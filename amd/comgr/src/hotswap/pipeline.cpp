#include "pipeline.h"
#include "code-object-utils.h"
#include "mc-state.h"
#include "raise-failure.h"
#include "raiser.h"

#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>

LLD_HAS_DRIVER(elf)

#define DEBUG_TYPE "transpiler"

namespace COMGR::hotswap {

namespace {

using TimingClock = std::chrono::steady_clock;

TimingClock::time_point timingStart(bool CollectTimings) {
  return CollectTimings ? TimingClock::now() : TimingClock::time_point{};
}

double timingElapsed(bool CollectTimings, TimingClock::time_point Start) {
  return CollectTimings
             ? std::chrono::duration<double>(TimingClock::now() - Start).count()
             : 0.0;
}

// Atomic write via a temp file + rename (writeToOutput), so a crash mid-write
// never leaves a truncated .o/.hsaco behind. Text vs binary is irrelevant on
// the Linux hotswap target, so a single raw writer covers both.
llvm::Error writeFile(llvm::StringRef Path, llvm::StringRef Contents) {
  return llvm::writeToOutput(Path, [&](llvm::raw_ostream &Out) -> llvm::Error {
    Out << Contents;
    return llvm::Error::success();
  });
}

llvm::Error writeFile(llvm::StringRef Path, llvm::ArrayRef<uint8_t> Data) {
  return writeFile(Path, llvm::toStringRef(Data));
}

// Best-effort write of a debug artifact: log and swallow any failure so a dump
// error never aborts the raise/compile pipeline.
void writeDebugFile(llvm::StringRef Path, llvm::StringRef Contents) {
  llvm::logAllUnhandledErrors(writeFile(Path, Contents), llvm::errs(),
                              "transpiler: ");
}

void writeDebugFile(llvm::StringRef Path, llvm::ArrayRef<uint8_t> Data) {
  llvm::logAllUnhandledErrors(writeFile(Path, Data), llvm::errs(),
                              "transpiler: ");
}

// Best-effort textual IR dump: print the module straight to the file so the
// production path never serializes IR into an in-memory string.
void writeDebugModule(llvm::StringRef Path, const llvm::Module &M) {
  llvm::logAllUnhandledErrors(
      llvm::writeToOutput(Path,
                          [&](llvm::raw_ostream &Out) -> llvm::Error {
                            M.print(Out, nullptr);
                            return llvm::Error::success();
                          }),
      llvm::errs(), "transpiler: ");
}

// Derive a filesystem-safe basename for an arbitrarily long kernel name.
// Most POSIX filesystems cap individual path components at 255 bytes, and
// Hotswap generates sibling files off the same stem (e.g. `<stem>.ll`,
// `<stem>.s`, `<stem>.dis`), so we leave a small suffix budget and fold
// anything longer down to a deterministic truncated+hashed form so two
// kernels with a shared 240-byte prefix don't collide on disk.
//
// The returned basename preserves a readable prefix of the original name
// for debuggability; it's only intended for temp-dir scratch files --
// symbol names inside the IR itself are unaffected.
std::string makeSafeBasename(llvm::StringRef KernelName,
                             size_t ReservedSuffixBytes = 8) {
  constexpr size_t MaxComponentBytes = 255;
  if (KernelName.size() + ReservedSuffixBytes <= MaxComponentBytes)
    return KernelName.str();

  uint64_t H = llvm::xxh3_64bits(KernelName);

  constexpr size_t HashHexBytes = 16;  // 64-bit hash as hex
  constexpr size_t SeparatorBytes = 1; // '_'
  const size_t PrefixBudget =
      MaxComponentBytes - ReservedSuffixBytes - HashHexBytes - SeparatorBytes;
  llvm::StringRef Prefix = KernelName.substr(0, PrefixBudget);
  return (Prefix + "_" + llvm::Twine::utohexstr(H)).str();
}

llvm::OptimizationLevel toOptimizationLevel(unsigned Level) {
  switch (Level) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  default:
    return llvm::OptimizationLevel::O3;
  }
}

llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
createHotswapTargetMachine(llvm::StringRef TargetISA, unsigned OptLevel) {
  std::string Err;
  llvm::Triple TheTriple(kAMDGPUTriple);
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TheTriple, Err);
  // The triple is hardcoded and the AMDGPU target is linked in, so a lookup
  // miss is a build misconfiguration rather than a kernel-level error.
  if (!TheTarget)
    return llvm::createStringError(
        llvm::Twine("transpiler: AMDGPU target not registered: ") + Err);
  llvm::CodeGenOptLevel CGOL = llvm::CodeGenOpt::getLevel(OptLevel).value_or(
      llvm::CodeGenOptLevel::Default);
  llvm::TargetOptions Opts;
  return std::unique_ptr<llvm::TargetMachine>(TheTarget->createTargetMachine(
      TheTriple, TargetISA, /*Features=*/"", Opts, llvm::Reloc::PIC_,
      /*CodeModel=*/std::nullopt, CGOL));
}

// In-process `opt`: run the default per-module pipeline at OptLevel.
void runOptPipeline(llvm::Module &M, llvm::TargetMachine &TM,
                    unsigned OptLevel) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB(&TM);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::OptimizationLevel OL = toOptimizationLevel(OptLevel);
  llvm::ModulePassManager MPM = OL == llvm::OptimizationLevel::O0
                                    ? PB.buildO0DefaultPipeline(OL)
                                    : PB.buildPerModuleDefaultPipeline(OL);
  MPM.run(M, MAM);
}

// Normalize raised setpc/swap_pc dispatch switches to ordinary branch trees
// before AMDGPU codegen. Raw switch terminators are not safe to hand to the
// irreducible-CFG path in the backend, so the pipeline calls this only for
// kernels that the raiser marked as containing enumerated setpc dispatch.
void lowerSwitchesToBranches(llvm::Module &M) {
  llvm::FunctionAnalysisManager FAM;
  llvm::PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);

  llvm::FunctionPassManager FPM;
  FPM.addPass(llvm::LowerSwitchPass());
  for (llvm::Function &F : M) {
    if (!F.isDeclaration())
      FPM.run(F, FAM);
  }
}

// Fail closed if switch lowering did not remove every switch terminator. This
// is deliberately module-wide: the setpc dispatch marker means the kernel
// requires branch-only IR before codegen, and silently letting any switch
// through would reintroduce the backend hazard this path exists to avoid.
llvm::Error checkNoSwitchTerminators(const llvm::Module &M,
                                     llvm::StringRef KernelName) {
  for (const llvm::Function &F : M) {
    for (const llvm::BasicBlock &BB : F) {
      if (!llvm::isa<llvm::SwitchInst>(BB.getTerminator()))
        continue;
      return llvm::createStringError(
          llvm::Twine("setpc dispatch switch remained after LowerSwitch for "
                      "kernel '") +
          KernelName + "' in function '" + F.getName() + "', block '" +
          BB.getName() + "'");
    }
  }
  return llvm::Error::success();
}

// In-process `llc`: run codegen for `M` and emit `FileType` to `OS`.
llvm::Error emitCodeGen(llvm::Module &M, llvm::TargetMachine &TM,
                        llvm::CodeGenFileType FileType,
                        llvm::raw_pwrite_stream &OS) {
  llvm::legacy::PassManager PM;
  if (TM.addPassesToEmitFile(PM, OS, /*DwoOut=*/nullptr, FileType))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "target cannot emit requested file type");

  PM.run(M);
  return llvm::Error::success();
}

struct DumpDir {
  llvm::SmallString<128> Path;
  bool Valid = false;
  bool Persistent = false;

  DumpDir() {
    static const std::optional<std::string> EnvDir =
        llvm::sys::Process::GetEnv("HSA_HOTSWAP_DUMP_DIR");
    if (EnvDir && !EnvDir->empty()) {
      Persistent = true;
      Path = *EnvDir;
      if (auto EC = llvm::sys::fs::create_directories(Path)) {
        llvm::errs() << "hotswap: failed to create dump dir '" << Path
                     << "': " << EC.message() << "\n";
        return;
      }
      // Create a unique subdirectory per invocation so parallel runs
      // don't clobber each other.
      llvm::SmallString<128> Sub;
      if (auto EC =
              llvm::sys::fs::createUniqueDirectory(Path + "/hotswap", Sub)) {
        llvm::errs() << "hotswap: failed to create subdir in '" << Path
                     << "': " << EC.message() << "\n";
        return;
      }
      Path = Sub;
      Valid = true;
    } else {
      if (auto EC = llvm::sys::fs::createUniqueDirectory("transpiler", Path)) {
        llvm::errs() << "hotswap: failed to create temp dir: " << EC.message()
                     << "\n";
      } else {
        Valid = true;
      }
    }
  }

  ~DumpDir() {
    if (Valid && !Persistent)
      llvm::sys::fs::remove_directories(Path);
  }

  DumpDir(const DumpDir &) = delete;
  DumpDir &operator=(const DumpDir &) = delete;

  std::string filePath(const llvm::Twine &Name) const {
    llvm::SmallString<256> P(Path);
    llvm::sys::path::append(P, Name);
    return std::string(P);
  }
};

} // anonymous namespace

static thread_local bool StrictModeOverrideActive = false;
static thread_local bool StrictModeOverrideValue = false;

ScopedStrictMode::ScopedStrictMode(bool Enabled)
    : PreviousActive(StrictModeOverrideActive),
      PreviousValue(StrictModeOverrideValue) {
  StrictModeOverrideActive = true;
  StrictModeOverrideValue = Enabled;
}

ScopedStrictMode::~ScopedStrictMode() {
  StrictModeOverrideActive = PreviousActive;
  StrictModeOverrideValue = PreviousValue;
}

bool isStrictMode() {
  if (StrictModeOverrideActive)
    return StrictModeOverrideValue;

  // Parsed once on first call; the result cannot change inside a process
  // because the env var is read once at the first transpile and reused for
  // the rest of the process lifetime. Treats any non-empty value as enabled
  // to keep the runner side (`HSA_HOTSWAP_STRICT=1`) and the pipeline side
  // decoupled; a future shell that writes `HSA_HOTSWAP_STRICT=true` still
  // works.
  static const bool Strict = [] {
    auto V = llvm::sys::Process::GetEnv("HSA_HOTSWAP_STRICT");
    return V && !V->empty();
  }();
  return Strict;
}

bool wantDumpInput() {
  // Parsed once, like isStrictMode(); reused by the raiser for the source
  // disassembly dump so both sides share one definition of the flag.
  static const bool Want = [] {
    auto V = llvm::sys::Process::GetEnv("HSA_HOTSWAP_DUMP_INPUT");
    return V && *V == "1";
  }();
  return Want;
}

// Raise one kernel to IR, then opt + codegen it to a relocatable .o.
// On success, writes the .o to ObjPath and returns true.
static bool raiseAndCompileKernel(
    const TextSection &Text, llvm::MemoryBufferRef CodeObjectData,
    llvm::StringRef KernelName, llvm::StringRef SourceISA,
    llvm::StringRef TargetISA, const DumpDir &TmpDir, llvm::StringRef ObjPath,
    PipelineResult &Result, const PipelineOptions &Options) {
  auto RaiseStart = timingStart(Options.CollectTimings);
  llvm::Expected<KernelMeta> MetaOrErr =
      extractKernelMeta(CodeObjectData, KernelName);
  KernelMeta Meta;
  if (MetaOrErr) {
    Meta = std::move(*MetaOrErr);
  } else {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << KernelName
                 << "': " << llvm::toString(MetaOrErr.takeError())
                 << ", using empty metadata\n";
  }
  if (Meta.Args.empty()) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << KernelName
                 << "', using empty metadata\n";
  }

  llvm::Expected<KernelSymbolExtent> KernelExtentOrErr =
      findKernelSymbolExtent(CodeObjectData, KernelName);
  if (!KernelExtentOrErr) {
    std::string Err = llvm::toString(KernelExtentOrErr.takeError());
    llvm::errs() << "transpiler: " << Err << "\n";
    Result.FailKernel = KernelName;
    Result.FailMnemonic = "__kernel_extent__";
    Result.FailReason = "KernelSymbolExtentLookupFailed";
    Result.FailFormat = "KernelSymbolExtentLookupFailed";
    Result.FailDetail = Err;
    Result.Timings.raiseSeconds +=
        timingElapsed(Options.CollectTimings, RaiseStart);
    return false;
  }
  uint64_t KernelOffset = KernelExtentOrErr->Offset;
  uint64_t KernelSize = KernelExtentOrErr->Size;
  LLVM_DEBUG(if (KernelOffset > 0) llvm::dbgs()
             << "transpiler: Kernel '" << KernelName << "' at .text offset 0x"
             << llvm::utohexstr(KernelOffset) << " size 0x"
             << llvm::utohexstr(KernelSize) << "\n");

  // Function-symbol extents let the raiser follow a tail-call into an outlined
  // device helper outside this kernel's own extent (see raiseToIR).
  // Best-effort: on failure fall back to an empty list, which keeps the strict
  // in-extent-only behavior.
  llvm::SmallVector<KernelSymbolExtent> FunctionExtents;
  if (llvm::Expected<llvm::SmallVector<KernelSymbolExtent>> ExtentsOrErr =
          listTextFunctionExtents(CodeObjectData)) {
    FunctionExtents = std::move(*ExtentsOrErr);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "hotswap: listTextFunctionExtents failed, "
                               "falling back to empty extents list\n");
    llvm::consumeError(ExtentsOrErr.takeError());
  }

  RaiseStats Stats;
  llvm::Expected<RaiseResult> RaisedOrErr = raiseToIR(
      Text.Bytes, SourceISA, KernelName, Meta, KernelOffset, KernelSize,
      TargetISA, Options.EnableWritelaneRewrite, Options.EnableWaveNative,
      Options.AssumeHipGlobalOffsetZero, FunctionExtents, &Stats);
  if (!RaisedOrErr) {
    llvm::errs() << "transpiler: Raising '" << KernelName
                 << "' to LLVM IR failed";
    Result.FailKernel = KernelName;
    bool IsFirstFailure = true;

    llvm::handleAllErrors(
        RaisedOrErr.takeError(),
        [&](RaiseFailure &Failure) {
          std::string RenderedFailure = Failure.message();
          llvm::errs() << " (" << RenderedFailure << ")";
          if (IsFirstFailure) {
            Result.FailMnemonic = Failure.Mnemonic;
            Result.FailReason = reasonString(Failure.Reason);
            Result.FailFormat = Failure.Format;
            Result.FailDetail = RenderedFailure;
            Result.FailOffset = Failure.Offset;
          }
          llvm::errs() << "\n";
          IsFirstFailure = false;
        },
        [&](const llvm::ErrorInfoBase &Err) {
          std::string RenderedFailure =
              "raiseToIR returned failure without a structured reason: " +
              Err.message();
          llvm::errs() << " (" << RenderedFailure << ")";
          if (IsFirstFailure) {
            Result.FailMnemonic = "";
            Result.FailReason = reasonString(RaiseFailureReason::InternalError);
            Result.FailFormat = "";
            Result.FailDetail = RenderedFailure;
            Result.FailOffset = 0;
          }
          llvm::errs() << "\n";
          IsFirstFailure = false;
        });

    Result.Timings.raiseSeconds +=
        timingElapsed(Options.CollectTimings, RaiseStart);
    return false;
  }

  RaiseResult Raised = std::move(*RaisedOrErr);
  Result.LiftedCount += Stats.LiftedCount;
  Result.TotalCount += Stats.TotalCount;
  if (Raised.UsesScratchPrivateSegment) {
    Result.UsesScratchPrivateSegment = true;
    if (Raised.SourcePrivateSegmentFixedSize >
        Result.SourcePrivateSegmentFixedSize)
      Result.SourcePrivateSegmentFixedSize =
          Raised.SourcePrivateSegmentFixedSize;
  }
  Result.C5SuppressedCount += Raised.C5SuppressedCount;
  if (Result.C5SuppressionReason.empty() && !Raised.C5SuppressionReason.empty())
    Result.C5SuppressionReason = Raised.C5SuppressionReason;
  Result.Timings.raiseSeconds +=
      timingElapsed(Options.CollectTimings, RaiseStart);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raised '" << KernelName << "' "
                          << Stats.LiftedCount << "/" << Stats.TotalCount
                          << " instructions\n");

  // Kernel names from Tensile et al. routinely exceed 255 bytes, which is
  // the per-component limit on ext4/xfs/tmpfs.  makeSafeBasename() hashes
  // the tail and truncates the head when the full name would blow the
  // budget; the symbol name inside the IR stays untouched, so debug
  // tooling can still resolve the long name from the LLVM module.
  std::string FileStem =
      makeSafeBasename(KernelName, /*ReservedSuffixBytes=*/5);

  if (!Raised.Module) {
    llvm::errs() << "transpiler: raiser produced no module for '" << KernelName
                 << "'\n";
    return false;
  }
  llvm::Module &M = *Raised.Module;

  // Codegen consumes the in-memory module directly; the .ll/.s/.dis files are
  // debug dumps only, so skip them unless a persistent dump dir was set (a
  // non-persistent temp dir is deleted on exit, taking the dumps with it).
  // The .ll is printed straight from the module here (pre-opt), so the
  // production path never serializes IR to text.
  auto WriteIrStart = timingStart(Options.CollectTimings);
  if (TmpDir.Persistent) {
    writeDebugModule(TmpDir.filePath(FileStem + ".ll"), M);
    if (!Raised.DisasmText.empty())
      writeDebugFile(TmpDir.filePath(FileStem + ".dis"), Raised.DisasmText);
  }
  Result.Timings.writeIrSeconds +=
      timingElapsed(Options.CollectTimings, WriteIrStart);

  llvm::Expected<std::unique_ptr<llvm::TargetMachine>> TMOrErr =
      createHotswapTargetMachine(TargetISA, Options.OptLevel);
  if (!TMOrErr) {
    llvm::errs() << "transpiler: failed to create TargetMachine for '"
                 << KernelName << "': " << llvm::toString(TMOrErr.takeError())
                 << "\n";
    return false;
  }
  std::unique_ptr<llvm::TargetMachine> TM = std::move(*TMOrErr);
  M.setDataLayout(TM->createDataLayout());

  auto OptStart = timingStart(Options.CollectTimings);
  runOptPipeline(M, *TM, Options.OptLevel);
  if (Raised.HasEnumeratedSetpcDispatch) {
    lowerSwitchesToBranches(M);
    if (llvm::Error Err = checkNoSwitchTerminators(M, KernelName)) {
      std::string Detail = llvm::toString(std::move(Err));
      llvm::errs() << "transpiler: " << Detail << "\n";
      Result.FailKernel = KernelName;
      Result.FailReason = reasonString(RaiseFailureReason::InternalError);
      Result.FailDetail = Detail;
      Result.Timings.optSeconds +=
          timingElapsed(Options.CollectTimings, OptStart);
      return false;
    }
  }
  Result.Timings.optSeconds += timingElapsed(Options.CollectTimings, OptStart);

  // Object codegen consumes the module, so clone it first when a debug
  // assembly dump is still needed.
  std::unique_ptr<llvm::Module> AsmModule;
  if (TmpDir.Persistent)
    AsmModule = llvm::CloneModule(M);

  llvm::SmallVector<char, 4096> ObjBytes;
  auto LlcStart = timingStart(Options.CollectTimings);
  llvm::Error Err = [&] {
    llvm::raw_svector_ostream OS(ObjBytes);
    return emitCodeGen(M, *TM, llvm::CodeGenFileType::ObjectFile, OS);
  }();
  Result.Timings.llcSeconds += timingElapsed(Options.CollectTimings, LlcStart);
  if (Err) {
    llvm::errs() << "transpiler: llc failed for '" << KernelName
                 << "': " << llvm::toString(std::move(Err)) << "\n";
    return false;
  }

  if (llvm::Error WriteErr = writeFile(
          ObjPath, llvm::StringRef(ObjBytes.data(), ObjBytes.size()))) {
    llvm::logAllUnhandledErrors(std::move(WriteErr), llvm::errs());
    return false;
  }

  // Textual assembly is a debug-only artifact emitted from the clone straight
  // to the file, so the object codegen above stays the canonical lowering and
  // the production path never materializes ASM text.
  if (AsmModule) {
    std::string AsmPath = TmpDir.filePath(FileStem + ".s");
    std::error_code EC;
    llvm::raw_fd_ostream AsmOut(AsmPath, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      llvm::logAllUnhandledErrors(llvm::createFileError(AsmPath, EC),
                                  llvm::errs(), "transpiler: ");
    } else if (llvm::Error Err =
                   emitCodeGen(*AsmModule, *TM,
                               llvm::CodeGenFileType::AssemblyFile, AsmOut)) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs());
    }
  }

  return true;
}

// Link one or more relocatable .o files into a shared HSACO using the
// in-process LLD ELF driver.
static llvm::Error linkObjects(llvm::ArrayRef<std::string> ObjPaths,
                               llvm::StringRef HsacoPath) {
  std::string HsacoPathStr = HsacoPath.str();
  llvm::SmallVector<const char *, 16> Args;
  Args.push_back("ld.lld");
  Args.push_back("-shared");
  Args.push_back("--threads=1");
  Args.push_back("-o");
  Args.push_back(HsacoPathStr.c_str());
  for (auto &O : ObjPaths)
    Args.push_back(O.c_str());

  // lld::lldMain drives a process-global CommonLinkerContext and is neither
  // re-entrant nor thread-safe; serialize all in-process links.
  static std::mutex LldMutex;
  std::lock_guard<std::mutex> LldLock(LldMutex);
  std::string ErrString;
  llvm::raw_string_ostream ErrStream(ErrString);
  lld::Result Ret = lld::lldMain(Args, llvm::nulls(), ErrStream,
                                 {{lld::Gnu, &lld::elf::link}});
  lld::CommonLinkerContext::destroy();
  if (Ret.retCode != 0 || !Ret.canRunAgain) {
    ErrStream.flush();
    return llvm::createStringError(
        "ld.lld failed return code: " + llvm::Twine(Ret.retCode) +
        " stderr: " + ErrString);
  }

  return llvm::Error::success();
}

void collectTargetPrivateSegmentMetadata(
    PipelineResult &Result, llvm::ArrayRef<std::string> KernelNames) {
  using namespace llvm::amdhsa;
  if (!Result.Hsaco || Result.Hsaco->getBufferSize() == 0)
    return;
  llvm::MemoryBufferRef HsacoBuf = Result.Hsaco->getMemBufferRef();
  for (llvm::StringRef KernelName : KernelNames) {
    llvm::Expected<KernelMeta> MetaOrErr =
        extractKernelMeta(HsacoBuf, KernelName);
    if (!MetaOrErr) {
      llvm::consumeError(MetaOrErr.takeError());
      continue;
    }
    KernelMeta &Meta = *MetaOrErr;
    if (!Meta.HasKernelDescriptor)
      continue;
    Result.TargetPrivateSegmentFixedSize =
        std::max(Result.TargetPrivateSegmentFixedSize,
                 static_cast<uint32_t>(Meta.PrivateSegmentFixedSize));
    const bool Enabled =
        (Meta.ComputePgmRsrc2 &
         (1u << COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT)) != 0;
    Result.TargetEnablePrivateSegment |= Enabled;
  }
}

// Shared body for both entry points: raise every kernel in `KernelNames`,
// link the objects into one HSACO, read it back, and collect target metadata.
// `Result` may already carry timings recorded by the caller (e.g.
// listKernelsSeconds); `TotalStart` anchors the overall timing.
static PipelineResult runPipelineImpl(llvm::MemoryBufferRef CodeObjectData,
                                      llvm::StringRef SourceISA,
                                      llvm::StringRef TargetISA,
                                      llvm::ArrayRef<std::string> KernelNames,
                                      const PipelineOptions &Options,
                                      TimingClock::time_point TotalStart,
                                      PipelineResult Result) {
  auto finish = [&]() {
    Result.Timings.totalSeconds =
        timingElapsed(Options.CollectTimings, TotalStart);
    return std::move(Result);
  };

  auto ExtractTextStart = timingStart(Options.CollectTimings);
  llvm::Expected<TextSection> TextOrErr = extractTextSection(CodeObjectData);
  Result.Timings.extractTextSeconds =
      timingElapsed(Options.CollectTimings, ExtractTextStart);
  if (!TextOrErr) {
    llvm::errs() << "transpiler: Failed to extract .text section: "
                 << llvm::toString(TextOrErr.takeError()) << "\n";
    return finish();
  }
  TextSection Text = std::move(*TextOrErr);

  auto TempDirStart = timingStart(Options.CollectTimings);
  DumpDir TmpDir;
  Result.Timings.createTempDirSeconds =
      timingElapsed(Options.CollectTimings, TempDirStart);
  if (!TmpDir.Valid)
    return finish();

  if (wantDumpInput())
    writeDebugFile(TmpDir.filePath("input.co"), CodeObjectData.getBuffer());

  llvm::SmallVector<std::string> ObjPaths;
  for (size_t I = 0; I < KernelNames.size(); ++I) {
    const std::string &KName = KernelNames[I];
    std::string ObjPath = TmpDir.filePath("k" + llvm::Twine(I) + ".o");

    LLVM_DEBUG(llvm::dbgs() << "transpiler:   [" << (I + 1) << "/"
                            << KernelNames.size() << "] " << KName << " ... ");

    if (!raiseAndCompileKernel(Text, CodeObjectData, KName, SourceISA,
                               TargetISA, TmpDir, ObjPath, Result, Options)) {
      LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
      Result.Success = false;
      return finish();
    }
    LLVM_DEBUG(llvm::dbgs() << "OK\n");
    ObjPaths.push_back(std::move(ObjPath));
  }

  std::string HsacoPath = TmpDir.filePath("merged.Hsaco");
  auto LinkStart = timingStart(Options.CollectTimings);
  if (llvm::Error Err = linkObjects(ObjPaths, HsacoPath)) {
    Result.FailDetail = llvm::toString(std::move(Err));
    return finish();
  }
  Result.Timings.linkSeconds +=
      timingElapsed(Options.CollectTimings, LinkStart);

  auto ReadHsacoStart = timingStart(Options.CollectTimings);
  if (auto HsacoBufOrErr =
          llvm::MemoryBuffer::getFile(HsacoPath, /*IsText=*/false)) {
    Result.Hsaco = std::move(*HsacoBufOrErr);
  } else {
    llvm::errs() << "transpiler: Cannot read HSACO: " << HsacoPath << ": "
                 << HsacoBufOrErr.getError().message() << "\n";
  }
  Result.Timings.readHsacoSeconds +=
      timingElapsed(Options.CollectTimings, ReadHsacoStart);
  if (!Result.Hsaco || Result.Hsaco->getBufferSize() == 0) {
    llvm::errs() << "transpiler: Failed to read HSACO\n";
    return finish();
  }

  auto CollectMetadataStart = timingStart(Options.CollectTimings);
  collectTargetPrivateSegmentMetadata(Result, KernelNames);
  Result.Timings.collectMetadataSeconds +=
      timingElapsed(Options.CollectTimings, CollectMetadataStart);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: HSACO generated: "
                          << Result.Hsaco->getBufferSize() << " bytes, "
                          << KernelNames.size() << " kernel(s)\n");
  Result.Success = true;
  return finish();
}

PipelineResult runPipeline(llvm::MemoryBufferRef CodeObjectData,
                           llvm::StringRef SourceISA, llvm::StringRef TargetISA,
                           llvm::StringRef KernelName,
                           PipelineOptions Options) {
  auto TotalStart = timingStart(Options.CollectTimings);
  llvm::SmallVector<std::string> KernelNames{KernelName.str()};
  return runPipelineImpl(CodeObjectData, SourceISA, TargetISA, KernelNames,
                         Options, TotalStart, PipelineResult{});
}

PipelineResult runPipelineAllKernels(llvm::MemoryBufferRef CodeObjectData,
                                     llvm::StringRef SourceISA,
                                     llvm::StringRef TargetISA,
                                     PipelineOptions Options) {
  auto TotalStart = timingStart(Options.CollectTimings);
  PipelineResult Result;
  auto finishEarly = [&]() {
    Result.Timings.totalSeconds =
        timingElapsed(Options.CollectTimings, TotalStart);
    return std::move(Result);
  };

  auto ListKernelsStart = timingStart(Options.CollectTimings);
  llvm::Expected<llvm::SmallVector<std::string>> KernelNamesOrErr =
      listKernelNames(CodeObjectData);
  Result.Timings.listKernelsSeconds =
      timingElapsed(Options.CollectTimings, ListKernelsStart);
  if (!KernelNamesOrErr) {
    llvm::errs() << "transpiler: No kernels found in code object: "
                 << llvm::toString(KernelNamesOrErr.takeError()) << "\n";
    return finishEarly();
  }
  if (KernelNamesOrErr->empty()) {
    llvm::errs() << "transpiler: No kernels found in code object\n";
    return finishEarly();
  }
  llvm::SmallVector<std::string> KernelNames = std::move(*KernelNamesOrErr);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raising " << KernelNames.size()
                          << " kernel(s) [" << SourceISA << " -> " << TargetISA
                          << "]\n");

  return runPipelineImpl(CodeObjectData, SourceISA, TargetISA, KernelNames,
                         Options, TotalStart, std::move(Result));
}

} // namespace COMGR::hotswap
