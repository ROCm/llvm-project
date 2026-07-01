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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>

LLD_HAS_DRIVER(elf)

#define DEBUG_TYPE "transpiler"

namespace COMGR::hotswap {

namespace {

using TimingClock = std::chrono::steady_clock;

double secondsBetween(TimingClock::time_point Start,
                      TimingClock::time_point End) {
  return std::chrono::duration<double>(End - Start).count();
}

TimingClock::time_point timingStart(bool CollectTimings) {
  return CollectTimings ? TimingClock::now() : TimingClock::time_point{};
}

double timingElapsed(bool CollectTimings, TimingClock::time_point Start) {
  return CollectTimings ? secondsBetween(Start, TimingClock::now()) : 0.0;
}

bool writeFile(llvm::StringRef Path, llvm::StringRef Contents) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "transpiler: Cannot write file: " << Path << ": "
                 << EC.message() << "\n";
    return false;
  }
  Out.write(Contents.data(), Contents.size());
  Out.flush();
  if (Out.has_error()) {
    llvm::errs() << "transpiler: write failed for: " << Path << "\n";
    return false;
  }
  return true;
}

bool writeFile(llvm::StringRef Path, llvm::ArrayRef<uint8_t> Data) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "transpiler: Cannot write file: " << Path << ": "
                 << EC.message() << "\n";
    return false;
  }
  Out.write(reinterpret_cast<const char *>(Data.data()), Data.size());
  Out.flush();
  if (Out.has_error()) {
    llvm::errs() << "transpiler: write failed for: " << Path << "\n";
    return false;
  }
  return true;
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
  std::string Prefix = KernelName.substr(0, PrefixBudget).str();
  std::string Hex = llvm::utohexstr(H, /*LowerCase=*/true, /*Width=*/16);
  return Prefix + "_" + Hex;
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

std::unique_ptr<llvm::TargetMachine>
createHotswapTargetMachine(llvm::StringRef TargetISA, unsigned OptLevel) {
  std::string Err;
  llvm::Triple TheTriple(kAMDGPUTriple);
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TheTriple, Err);
  // The triple is hardcoded and the AMDGPU target is linked in, so a lookup
  // miss is a build misconfiguration rather than a recoverable error.
  if (!TheTarget)
    llvm::report_fatal_error(
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
    static const char *EnvDir = std::getenv("HSA_HOTSWAP_DUMP_DIR");
    if (EnvDir && EnvDir[0]) {
      Persistent = true;
      Path = EnvDir;
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

  std::string filePath(llvm::StringRef Name) const {
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

  // Parsed once on first call. The handler implementations call this on
  // every relevant instruction, so going through the OS allocator
  // (`std::getenv`) repeatedly would be wasteful; the result also cannot
  // change inside a process because the env var is read once at the
  // first transpile and reused for the rest of the process lifetime.
  // Treats any non-empty value as enabled to keep the runner side
  // (`HSA_HOTSWAP_STRICT=1`) and the pipeline side decoupled; a future
  // shell that writes `HSA_HOTSWAP_STRICT=true` still works.
  static const bool Strict = []() {
    const char *V = std::getenv("HSA_HOTSWAP_STRICT");
    return V && V[0] != '\0';
  }();
  return Strict;
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
  if (!MetaOrErr) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << KernelName
                 << "': " << llvm::toString(MetaOrErr.takeError())
                 << ", using empty metadata\n";
  }
  KernelMeta Meta = MetaOrErr ? std::move(*MetaOrErr) : KernelMeta{};
  if (Meta.Args.empty()) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << KernelName
                 << "', using empty metadata\n";
  }

  auto KernelExtentOrErr = findKernelSymbolExtent(CodeObjectData, KernelName);
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

  auto Raised =
      raiseToIR(Text.Bytes, SourceISA, KernelName, Meta, KernelOffset,
                KernelSize, TargetISA, Options.EnableWritelaneRewrite,
                Options.EnableWaveNative, Options.AssumeHipGlobalOffsetZero);
  if (!Raised.Success) {
    llvm::errs() << "transpiler: Raising '" << KernelName
                 << "' to LLVM IR failed";
    Result.FailKernel = KernelName;
    RaiseFailure Failure =
        Raised.Failure.hasFailed()
            ? Raised.Failure
            : RaiseFailure::internalFailure(
                  "raiseToIR returned failure without a structured reason");
    std::string RenderedFailure = formatRaiseFailure(Failure);
    llvm::errs() << " (" << RenderedFailure << ")";
    Result.FailMnemonic = Failure.Mnemonic;
    Result.FailReason = reasonString(Failure.Reason);
    Result.FailFormat = Failure.Format;
    Result.FailDetail = RenderedFailure;
    Result.FailOffset = Failure.Offset;
    llvm::errs() << "\n";
    Result.Timings.raiseSeconds +=
        timingElapsed(Options.CollectTimings, RaiseStart);
    return false;
  }
  Result.LiftedCount += Raised.LiftedCount;
  Result.TotalCount += Raised.TotalCount;
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
  if (!Result.IrText.empty())
    Result.IrText += "\n";
  Result.IrText += Raised.IrText;

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raised '" << KernelName << "' "
                          << Raised.LiftedCount << "/" << Raised.TotalCount
                          << " instructions\n");

  // Kernel names from Tensile et al. routinely exceed 255 bytes, which is
  // the per-component limit on ext4/xfs/tmpfs.  makeSafeBasename() hashes
  // the tail and truncates the head when the full name would blow the
  // budget; the symbol name inside the IR stays untouched, so debug
  // tooling can still resolve the long name from the LLVM module.
  std::string FileStem =
      makeSafeBasename(KernelName, /*ReservedSuffixBytes=*/5);

  // Codegen consumes the in-memory module directly; the .ll/.s/.dis files are
  // debug dumps only, so skip them unless a persistent dump dir was set (a
  // non-persistent temp dir is deleted on exit, taking the dumps with it).
  auto WriteIrStart = timingStart(Options.CollectTimings);
  if (TmpDir.Persistent) {
    writeFile(TmpDir.filePath(FileStem + ".ll"), Raised.IrText);
    static const char *DumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
    if (DumpInput && DumpInput[0] == '1' && !Raised.DisasmText.empty())
      writeFile(TmpDir.filePath(FileStem + ".dis"), Raised.DisasmText);
  }
  Result.Timings.writeIrSeconds +=
      timingElapsed(Options.CollectTimings, WriteIrStart);

  if (!Raised.Module) {
    llvm::errs() << "transpiler: raiser produced no module for '" << KernelName
                 << "'\n";
    return false;
  }
  llvm::Module &M = *Raised.Module;

  std::unique_ptr<llvm::TargetMachine> TM =
      createHotswapTargetMachine(TargetISA, Options.OptLevel);
  if (!TM) {
    llvm::errs() << "transpiler: failed to create TargetMachine for '"
                 << KernelName << "'\n";
    return false;
  }
  M.setDataLayout(TM->createDataLayout());

  auto OptStart = timingStart(Options.CollectTimings);
  runOptPipeline(M, *TM, Options.OptLevel);
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

  if (!writeFile(ObjPath,
                 llvm::ArrayRef<uint8_t>(
                     reinterpret_cast<const uint8_t *>(ObjBytes.data()),
                     ObjBytes.size())))
    return false;

  // Textual assembly is a debug-only artifact emitted from the clone so the
  // object codegen above stays the canonical lowering.
  if (AsmModule) {
    llvm::SmallString<4096> AsmText;
    llvm::raw_svector_ostream OS(AsmText);
    if (llvm::Error Err = emitCodeGen(*AsmModule, *TM,
                                      llvm::CodeGenFileType::AssemblyFile, OS))
      llvm::consumeError(std::move(Err));
    else
      writeFile(TmpDir.filePath(FileStem + ".s"), AsmText);
  }

  return true;
}

// Link one or more relocatable .o files into a shared HSACO using the
// in-process LLD ELF driver.
static bool linkObjects(llvm::ArrayRef<std::string> ObjPaths,
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
  lld::Result Ret = lld::lldMain(Args, llvm::outs(), llvm::errs(),
                                 {{lld::Gnu, &lld::elf::link}});
  lld::CommonLinkerContext::destroy();
  if (Ret.retCode != 0 || !Ret.canRunAgain) {
    llvm::errs() << "transpiler: ld.lld failed\n";
    return false;
  }
  return true;
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

PipelineResult runPipeline(llvm::MemoryBufferRef CodeObjectData,
                           llvm::StringRef SourceISA, llvm::StringRef TargetISA,
                           llvm::StringRef KernelName,
                           PipelineOptions Options) {
  auto TotalStart = timingStart(Options.CollectTimings);
  PipelineResult Result;
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

  {
    static const char *DumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
    if (DumpInput && DumpInput[0] == '1')
      writeFile(TmpDir.filePath("input.co"),
                llvm::ArrayRef(reinterpret_cast<const uint8_t *>(
                                   CodeObjectData.getBufferStart()),
                               CodeObjectData.getBufferSize()));
  }

  std::string ObjPath = TmpDir.filePath("kernel.o");
  std::string HsacoPath = TmpDir.filePath("kernel.Hsaco");

  if (!raiseAndCompileKernel(Text, CodeObjectData, KernelName, SourceISA,
                             TargetISA, TmpDir, ObjPath, Result, Options))
    return finish();

  auto LinkStart = timingStart(Options.CollectTimings);
  if (!linkObjects({ObjPath}, HsacoPath))
    return finish();
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
  std::string KernelNameStr = KernelName.str();
  auto CollectMetadataStart = timingStart(Options.CollectTimings);
  collectTargetPrivateSegmentMetadata(Result, {KernelNameStr});
  Result.Timings.collectMetadataSeconds +=
      timingElapsed(Options.CollectTimings, CollectMetadataStart);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: HSACO generated: "
                          << Result.Hsaco->getBufferSize() << " bytes\n");
  Result.Success = true;
  return finish();
}

PipelineResult runPipelineAllKernels(llvm::MemoryBufferRef CodeObjectData,
                                     llvm::StringRef SourceISA,
                                     llvm::StringRef TargetISA,
                                     PipelineOptions Options) {
  auto TotalStart = timingStart(Options.CollectTimings);
  PipelineResult Result;
  auto finish = [&]() {
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
    return finish();
  }
  llvm::SmallVector<std::string> KernelNames = std::move(*KernelNamesOrErr);
  if (KernelNames.empty()) {
    llvm::errs() << "transpiler: No kernels found in code object\n";
    return finish();
  }

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raising " << KernelNames.size()
                          << " kernel(s) [" << SourceISA << " -> " << TargetISA
                          << "]\n");

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

  static const char *DumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
  if (DumpInput && DumpInput[0] == '1')
    writeFile(TmpDir.filePath("input.co"),
              llvm::ArrayRef(reinterpret_cast<const uint8_t *>(
                                 CodeObjectData.getBufferStart()),
                             CodeObjectData.getBufferSize()));

  std::vector<std::string> ObjPaths;
  for (size_t I = 0; I < KernelNames.size(); ++I) {
    const auto &KName = KernelNames[I];
    std::string ObjPath = TmpDir.filePath("k" + std::to_string(I) + ".o");

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
  if (!linkObjects(ObjPaths, HsacoPath))
    return finish();
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
    llvm::errs() << "transpiler: Failed to read merged HSACO\n";
    return finish();
  }
  auto CollectMetadataStart = timingStart(Options.CollectTimings);
  collectTargetPrivateSegmentMetadata(Result, KernelNames);
  Result.Timings.collectMetadataSeconds +=
      timingElapsed(Options.CollectTimings, CollectMetadataStart);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Merged HSACO: "
                          << Result.Hsaco->getBufferSize() << " bytes, "
                          << KernelNames.size() << " kernel(s)\n");
  Result.Success = true;
  return finish();
}

} // namespace COMGR::hotswap
