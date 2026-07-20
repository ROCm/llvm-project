// Per-file raiser CLI — two modes.
//
// Usage:
//   raise_cli <code-object.co|.hsaco> [--isa=<arch>] [--target-isa=<arch>]
//   raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] [--isa=<arch>]
//                                     [--target-isa=<arch>]
//
// Default (kerneldex-coverage) mode. For each kernel in the code object,
// forks a child that runs raiseToIR so that a fatal error
// (report_fatal_error / asan trap / ...) in one kernel doesn't poison
// the whole file.  Emits one or more lines per kernel on stdout:
//
//   OK   <kernel-name> (<lifted>/<total>)
//   FAIL <kernel-name> -> <mnemonic> [<format>] (<lifted>/<total>)
//   ALSO <kernel-name> -> <mnemonic> [<format>]
//   ALSO <kernel-name> -> __truncated__ [+N unique blockers not shown]
//
// The FAIL line carries the first blocker. Each additional unique
// (mnemonic, format) pair is reported on its own ALSO line up to the
// fixed shared-memory cap; the truncation line reports any excess.
// The (<lifted>/<total>) count on the FAIL line reflects how many
// instructions were successfully raised before and after all blockers.
//
// Kernels that crashed in the child (signal, non-zero exit, incomplete
// shm) are reported as a FAIL with mnemonic `__crash__` and a bracketed
// format such as `signal_<N>`, `exit_<N>`, or
// `status_incomplete` so they still land in the kerneldex worklist
// instead of being silently dropped.
//
// In default mode, exits 0 iff every kernel succeeded, 1 if any kernel
// failed/crashed, and a distinct non-zero infrastructure code when the
// parent cannot set up or monitor a child. ISA is auto-detected from the
// filename (look for `gfx<digits>[a-z]?`) when `--isa=` is not passed.
//
// --emit-ir mode. Designed for lit tests. Runs raiseToIR in-process (no
// fork), dumps the raised LLVM IR on stdout, and leaves stderr alone so
// FileCheck can match warnings / abort-gate diagnostics. Kernel
// selection:
//
//   --emit-ir              all kernels, in code-object order
//   --emit-ir=<kernel>     one kernel
//   --emit-ir=<k1>,<k2>    the listed kernels, in the given order
//
// When more than one kernel is emitted, each kernel's IR is preceded by
// a `; === raise_cli kernel: <name> ===` separator so a single
// FileCheck pass can anchor per-kernel checks. A kernel that fails to
// raise prints its diagnostic to stderr and the loop continues; the
// exit code is 0 iff every selected kernel raised successfully. This
// lets an all-refusing kernel set stay under a single `%not ... 2>&1`
// RUN line.
//
// --target-isa=<arch>. Optional. Controls the target ISA the raiser
// lowers for; defaults to the source ISA (same-wave translation). Use
// to exercise cross-wave paths from a single CO (e.g. a gfx1250 CO
// compiled for a wave64 target).
//
// --enable-writelane-rewrite / --disable-writelane-rewrite. Default
// **on** (post-Triton-corpus graduation; see raiser.hpp for the full
// rationale).  Controls the post-raise rewrite of cross-widen-divergent
// `v_writelane_b32` / `v_readlane_b32` sites into per-source-wave
// `select` / `ds_bpermute` primitives — see
// `rewrite_cross_lane_divergent.{hpp,cpp}` and
// hotswap/docs/wave-size-translation.md §5.6.3.
//
// `--enable-writelane-rewrite` is accepted for backward compatibility
// (the canonical flag name used by existing lit fixtures) and is a
// no-op since the default is already on; `--disable-writelane-rewrite`
// forces the pre-rewrite path and is used by the `REFUSE` / `UNCHANGED`
// sibling RUN lines in the writelane/readlane regression fixtures to
// pin the pre-rewrite contract.  Later-wins between the two flags is
// by command-line order (last occurrence decides).
//
// --enable-wave-native / --disable-wave-native. Default **on** as
// of the WaveNative graduation. Selects `WaveNativeProjection`
// instead of `ModuloReplicationProjection` for wave32 source →
// wave64 target cross-widening. Under wave-native the kernel entry
// emits `@llvm.amdgcn.init_whole_wave` so hardware EXEC = -1 for
// the body, which:
//   * makes the WMMA → MFMA pipeline in `wmma-lowering.cpp`
//     correct on the upper half of the Wave64 target (the original
//     design motivation — see wave-size-translation.md §5.6.1);
//   * projects kernels with `num_warps > 1` correctly by giving
//     each target lane its own modeled-EXEC bit (fixes the
//     `swiglu_fp32` / `corpus_layernorm_fp32` class documented in
//     hotswap/docs/modrep-predicate-chain.md §4.3 sub-case 1);
//   * renders the C5 classifier's MODREP-specific refusal
//     rationale inapplicable — target lanes have their own
//     modeled-EXEC bits rather than sharing source wave 0's. The
//     classifier's `waveNative` gate suppresses refusal on this
//     path. For `canary_bpermute_scan_fp32`, the underlying
//     miscompile that would otherwise surface is closed by the
//     VOPD-cndmask SGPR-condition fix
//     (modrep-predicate-chain.md §6.4) rather than by the
//     projection choice itself.
//
// `--disable-wave-native` opts back into `ModuloReplicationProjection`
// for the narrow class of pointwise / independent-half kernels where
// MODREP's "replicas of source wave 0" model is correct AND where
// the C5 refusal under MODREP is the desired loud-fail signal.
// No env-var override exists; `HSA_HOTSWAP_WAVE_NATIVE` was a
// transient test hook during the graduation sweep and has been
// removed so the opt-out path isn't silently bypassed.
//
// (The earlier `--enable-permlane16-xor3-partner` /
// `--enable-permlane16-swap-selfpreserve` flags were removed along
// with their rewrite passes once the asymmetric
// `v_permlane16_swap_b32` lift landed — see
// `handle-valu-cross-lane.cpp::emitPermLaneSwapEmulation` and
// matrix-translation.md §12.4.7.)

#include "comgr-metadata.h"
#include "comgr.h"
#include "hotswap/code-object-utils.h"
#include "hotswap/pipeline.h"
#include "hotswap/raise-failure.h"
#include "hotswap/raiser.h"

// raiser.hpp forward-declares llvm::LLVMContext and llvm::Module but
// RaiseResult holds them by unique_ptr, so the destructor synthesized in
// main() needs the complete types.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <utility>
#include <unistd.h>

namespace {

namespace cl = llvm::cl;

// Shared-memory block handed from each per-kernel child back to the parent.
// Using a fixed-size POD struct keeps the IPC trivially safe across fork().

// Maximum number of unique failure buckets emitted per kernel. Kernels that hit
// more distinct blockers will have the excess counted in numDroppedFailures and
// surfaced in the ALSO output as a truncation notice. The lifted/total count is
// always accurate regardless of the cap.
static constexpr int kMaxTrackedFailures = 32;
static constexpr int kMmapFailedExitCode = 3;
static constexpr int kChildOutputFileFailedExitCode = 4;
static constexpr int kForkFailedExitCode = 5;
static constexpr int kChildOutputWriteFailedExitCode = 6;

// Stable identity for one emitted failure bucket. The detail string is omitted
// deliberately: it is diagnostic text, not part of corpus bucketing.
struct FailureBucketKey {
  COMGR::hotswap::RaiseFailureReason Reason;
  std::string Mnemonic;
  std::string Format;
};

bool operator==(const FailureBucketKey &Lhs, const FailureBucketKey &Rhs) {
  return Lhs.Reason == Rhs.Reason && Lhs.Mnemonic == Rhs.Mnemonic &&
         Lhs.Format == Rhs.Format;
}

struct KernelRaiseStats {
  bool done;
  bool success;
  int lifted;
  int total;
  // Count of unique blockers that exceeded kMaxTrackedFailures and were not
  // emitted to the child's output file.
  int numDroppedFailures;
};

std::string autoDetectIsa(llvm::StringRef path) {
  // Look for ``gfx<digits>[a-z]?`` anywhere in the filename.
  for (size_t i = 0; i + 3 < path.size(); ++i) {
    if (path[i] == 'g' && path[i + 1] == 'f' && path[i + 2] == 'x') {
      size_t j = i + 3;
      while (j < path.size() &&
             std::isdigit(static_cast<unsigned char>(path[j])))
        ++j;
      if (j > i + 3) {
        if (j < path.size() && path[j] >= 'a' && path[j] <= 'z')
          ++j;
        return path.substr(i, j - i).str();
      }
    }
  }
  return {};
}

cl::opt<std::string> CoPathOpt(cl::Positional, cl::Required,
                               cl::desc("<code-object.co|.hsaco>"));

cl::opt<std::string> IsaOpt("isa", cl::value_desc("arch"),
                            cl::desc("Source ISA; inferred from the filename "
                                     "or ELF e_flags when not given."));

cl::opt<std::string>
    TargetIsaOpt("target-isa", cl::value_desc("arch"),
                                  cl::desc("Target ISA the raiser lowers for "
                                           "(default: same as --isa)."));

cl::opt<std::string>
    EmitIrOpt("emit-ir", cl::ValueOptional, cl::value_desc("kernel[,kernel...]"),
              cl::desc("Dump raised LLVM IR on stdout (no fork; stderr left "
                       "alone for FileCheck). Bare = all kernels; =<k>[,<k>...] "
                       "selects a subset in order."));

cl::opt<std::string>
    WriteHsacoOpt("write-hsaco", cl::value_desc("path"),
                  cl::desc("Run the full pipeline (raise + llc + lld) for a "
                           "single kernel and write the HSACO to <path>."));

cl::opt<std::string> KernelOpt("kernel", cl::value_desc("name"),
                               cl::desc("Kernel selected by --write-hsaco."));

cl::opt<bool> EnableWritelaneRewriteOpt(
    "enable-writelane-rewrite",
    cl::desc("Enable the cross-widen-divergent writelane/readlane rewrite "
             "(default on; later-wins with --disable-writelane-rewrite)."));
cl::opt<bool> DisableWritelaneRewriteOpt(
    "disable-writelane-rewrite",
    cl::desc("Pin the pre-rewrite REFUSE / UNCHANGED path."));

cl::opt<bool> EnableWaveNativeOpt(
    "enable-wave-native",
    cl::desc("Select WaveNativeProjection for wave32->wave64 cross-widening "
             "(default on; later-wins with --disable-wave-native)."));
cl::opt<bool> DisableWaveNativeOpt(
    "disable-wave-native",
                         cl::desc("Pin ModuloReplicationProjection."));

// Batch output de-duplicates failures by the structured fields visible in
// FAIL / ALSO lines.
FailureBucketKey failureBucketKey(
    const COMGR::hotswap::RaiseFailure &Failure) {
  return {Failure.Reason,
          Failure.Mnemonic.empty() ? std::string("unknown") : Failure.Mnemonic,
          Failure.Format.empty() ? std::string("unknown") : Failure.Format};
}

// Append the structured fields that distinguish otherwise identical
// mnemonic/format blockers in human diagnostics.
void printBatchFailureSuffix(llvm::raw_ostream &OS,
                             const COMGR::hotswap::RaiseFailure &Failure) {
  OS << " reason=" << COMGR::hotswap::reasonString(Failure.Reason)
     << " @offset=0x";
  OS.write_hex(Failure.Offset);
  if (!Failure.Detail.empty())
    OS << " :: " << Failure.Detail;
}

// Print one batch-mode failure record while preserving the historical
// `FAIL/ALSO <kernel> -> <mnemonic> [<format>]` prefix.
void printBatchFailureLine(llvm::raw_ostream &OS, llvm::StringRef Prefix,
                           llvm::StringRef KernelName,
                           const COMGR::hotswap::RaiseFailure &Failure,
                           int Lifted = -1, int Total = -1) {
  OS << Prefix << " " << KernelName << " -> "
     << (Failure.Mnemonic.empty() ? "unknown" : Failure.Mnemonic) << " ["
     << (Failure.Format.empty() ? "unknown" : Failure.Format) << "]";
  if (Lifted >= 0 && Total >= 0)
    OS << " (" << Lifted << "/" << Total << ")";
  printBatchFailureSuffix(OS, Failure);
  OS << "\n";
}

// Replay the child's buffered stdout after it exits so parent-side OK / FAIL
// records stay serialized even when a kernel crashes during raise.
bool replayChildOutput(llvm::StringRef Path) {
  auto BufferOrErr = llvm::MemoryBuffer::getFile(Path);
  if (!BufferOrErr) {
    llvm::errs() << "raise_cli: could not read child output " << Path << ": "
                 << BufferOrErr.getError().message() << "\n";
    return false;
  }
  llvm::outs() << (*BufferOrErr)->getBuffer();
  return true;
}

// Map the child's wait status into the same bracketed crash format the batch
// output already uses, so non-signal child failures remain diagnosable.
std::string childCrashFormat(int Status, bool ShmDone) {
  if (WIFSIGNALED(Status))
    return "signal_" + std::to_string(WTERMSIG(Status));
  if (WIFEXITED(Status)) {
    int ExitCode = WEXITSTATUS(Status);
    if (ExitCode == 0 && !ShmDone)
      return "status_incomplete";
    if (ExitCode == kChildOutputWriteFailedExitCode)
      return "child_output_write_failed";
    if (ExitCode != 0)
      return "exit_" + std::to_string(ExitCode);
  }
  if (!ShmDone)
    return "status_incomplete";
  return "wait_status_unknown";
}

cl::opt<bool> AssumeHipGlobalOffsetZeroOpt(
    "assume-hip-global-offset-zero",
    cl::desc("Assume HIP launch semantics for hidden_global_offset_{x,y,z}; "
             "generic HSA/OpenCL callers should leave this disabled."));

cl::opt<unsigned> OptLevel(
    "O", cl::Prefix, cl::init(0), cl::value_desc("level"),
    cl::desc("Optimization level (0-3) for the in-process opt + llc codegen "
             "stages (default 0)."));

// Resolve an --enable-/--disable- toggle pair, later occurrence wins.
bool resolveToggle(bool Default, const cl::opt<bool> &Enable,
                   const cl::opt<bool> &Disable) {
  unsigned EnablePos = Enable.getNumOccurrences() ? Enable.getPosition() : 0;
  unsigned DisablePos = Disable.getNumOccurrences() ? Disable.getPosition() : 0;
  if (!EnablePos && !DisablePos)
    return Default;
  return EnablePos >= DisablePos;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Per-kernel raiser CLI. Default mode emits per-kernel OK/FAIL lines on "
      "stdout; --emit-ir dumps raised IR for one or more kernels and "
      "--write-hsaco writes a single kernel's HSACO.\n");

  std::string coPath = CoPathOpt;
  std::string isa = IsaOpt;
  std::string targetIsa = TargetIsaOpt;
  bool emitIr = EmitIrOpt.getNumOccurrences() > 0;
  std::string emitIrKernel = EmitIrOpt;
  std::string writeHsacoPath = WriteHsacoOpt;
  std::string writeHsacoKernel = KernelOpt;
  // Both toggles default on (Triton-corpus / WaveNative graduations; see this
  // file's top-of-file comment and raiser.hpp). The --disable- forms pin the
  // pre-rewrite / MODREP paths for the lit fixtures.
  bool EnableWritelaneRewrite = resolveToggle(
      true, EnableWritelaneRewriteOpt, DisableWritelaneRewriteOpt);
  bool EnableWaveNative =
      resolveToggle(true, EnableWaveNativeOpt, DisableWaveNativeOpt);

  // Read the file up-front so we can fall back to the ELF e_flags
  // ISA when the filename heuristic fails (kerneldex corpora often
  // store kernels under hashed names with no `gfx*` substring; the
  // ELF MACH field is the only deterministic source).
  auto coBufOrErr = llvm::MemoryBuffer::getFile(coPath, /*IsText=*/false);
  if (!coBufOrErr) {
    llvm::errs() << "raise_cli: cannot read " << coPath << ": "
                 << coBufOrErr.getError().message() << "\n";
    return 2;
  }
  llvm::MemoryBufferRef coData = (*coBufOrErr)->getMemBufferRef();

  if (isa.empty()) {
    isa = autoDetectIsa(coPath);
    if (isa.empty()) {
      std::string elfIsa;
      if (COMGR::metadata::getElfIsaName(coData, elfIsa) ==
          AMD_COMGR_STATUS_SUCCESS)
        isa = std::move(elfIsa);
    }
    if (isa.empty()) {
      llvm::errs() << "raise_cli: could not infer ISA from " << coPath
                   << "; pass --isa=<arch>\n";
      return 2;
    }
  }

  auto kernelNamesOrErr = COMGR::hotswap::listKernelNames(coData);
  if (!kernelNamesOrErr) {
    llvm::errs() << "raise_cli: no kernels in " << coPath << ": "
                 << llvm::toString(kernelNamesOrErr.takeError()) << "\n";
    return 2;
  }
  llvm::SmallVector<std::string> kernelNames = std::move(*kernelNamesOrErr);
  if (kernelNames.empty()) {
    llvm::errs() << "raise_cli: no kernels in " << coPath << "\n";
    return 2;
  }

  auto textOrErr = COMGR::hotswap::extractTextSection(coData);
  if (!textOrErr) {
    llvm::errs() << "raise_cli: could not extract .text from " << coPath
                 << ": " << llvm::toString(textOrErr.takeError()) << "\n";
    return 2;
  }
  COMGR::hotswap::TextSection text = std::move(*textOrErr);

  // Function-symbol extents let the raiser follow tail-calls into outlined
  // device helpers outside a kernel's own extent. Best-effort: empty on failure
  // keeps the strict in-extent-only behavior.
  llvm::SmallVector<COMGR::hotswap::KernelSymbolExtent> functionExtents;
  if (llvm::Expected<llvm::SmallVector<COMGR::hotswap::KernelSymbolExtent>>
          extentsOrErr = COMGR::hotswap::listTextFunctionExtents(coData)) {
    functionExtents = std::move(*extentsOrErr);
  } else {
    llvm::consumeError(extentsOrErr.takeError());
  }

  // --emit-ir path — no fork, no stderr redirect. Used by lit tests that
  // FileCheck the raised IR on stdout and the raiser diagnostics on
  // stderr. Processes one or more kernels per invocation.
  if (emitIr) {
    // Resolve the target list: bare --emit-ir selects every kernel in
    // code-object order; --emit-ir=<k>[,<k>...] selects the listed
    // kernels in the given order.
    llvm::SmallVector<std::string> Targets;
    if (emitIrKernel.empty()) {
      Targets.assign(kernelNames.begin(), kernelNames.end());
    } else {
      llvm::SmallVector<llvm::StringRef> Requested;
      llvm::StringRef(emitIrKernel)
          .split(Requested, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (llvm::StringRef Req : Requested) {
        Req = Req.trim();
        bool Found = false;
        for (const auto &kn : kernelNames)
          if (kn == Req) {
            Targets.push_back(kn);
            Found = true;
            break;
          }
        if (!Found) {
          llvm::errs() << "raise_cli: kernel '" << Req << "' not found in "
                       << coPath << "\n";
          return 2;
        }
      }
    }
    if (Targets.empty()) {
      llvm::errs() << "raise_cli: no kernels selected by --emit-ir in " << coPath
                   << "\n";
      return 2;
    }

    // A multi-kernel dump prefixes each kernel with a separator so a
    // single FileCheck pass can anchor per-kernel CHECK-LABEL blocks.
    bool Multi = Targets.size() > 1;
    bool AnyFailed = false;
    for (const std::string &Target : Targets) {
      auto metaOrErr = COMGR::hotswap::extractKernelMeta(coData, Target);
      if (!metaOrErr) {
        llvm::errs() << "raise_cli: kernel '" << Target << "' metadata: "
                     << llvm::toString(metaOrErr.takeError()) << "\n";
        AnyFailed = true;
        continue;
      }
      COMGR::hotswap::KernelMeta meta = std::move(*metaOrErr);
      auto kernelExtentOrErr =
          COMGR::hotswap::findKernelSymbolExtent(coData, Target);
      if (!kernelExtentOrErr) {
        std::string err = llvm::toString(kernelExtentOrErr.takeError());
        llvm::errs() << "raise_cli: kernel '" << Target
                     << "' extent lookup failed: " << err << "\n";
        AnyFailed = true;
        continue;
      }
      uint64_t kernelOffset = kernelExtentOrErr->Offset;
      uint64_t kernelSize = kernelExtentOrErr->Size;
      llvm::Expected<COMGR::hotswap::RaiseResult> RaisedOrErr =
          COMGR::hotswap::raiseToIR(text.Bytes, isa, Target, meta, kernelOffset,
                                    kernelSize, targetIsa,
                                    EnableWritelaneRewrite, EnableWaveNative,
                                    AssumeHipGlobalOffsetZeroOpt, text.Address,
                                    text.ImageSections, functionExtents);
      if (!RaisedOrErr) {
        // raiseToIR only returns a module on the success path, so we cannot
        // dump partial IR here. Callers that need stderr diagnostics
        // (abort-gate lit tests, etc.) FileCheck the raiser's stderr — we
        // leave that untouched.
        llvm::handleAllErrors(
            RaisedOrErr.takeError(),
            [&](COMGR::hotswap::RaiseFailure &Failure) {
              llvm::errs() << "raise_cli: kernel '" << Target
                           << "' failed to raise: " << Failure.message() << "\n";
            },
            [&](const llvm::ErrorInfoBase &Err) {
              llvm::errs() << "raise_cli: kernel '" << Target
                           << "' failed to raise: raiseToIR returned failure "
                              "without a structured reason: "
                           << Err.message() << "\n";
            });
        AnyFailed = true;
        continue;
      }
      COMGR::hotswap::RaiseResult raised = std::move(*RaisedOrErr);
      if (!raised.Module) {
        llvm::errs() << "raise_cli: kernel '" << Target
                     << "' raised without a module\n";
        AnyFailed = true;
        continue;
      }
      if (Multi)
        llvm::outs() << "; === raise_cli kernel: " << Target << " ===\n";
      raised.Module->print(llvm::outs(), nullptr);
    }
    return AnyFailed ? 1 : 0;
  }

  // --write-hsaco path — runs the full pipeline (raise + llc + lld)
  // for a single kernel and writes the resulting HSACO to disk.
  // Triage-mode only: lets downstream tools (llvm-objdump) inspect the
  // exact bytes the gtest harness would launch, so we can walk the
  // Phase 6.5 rewrite end-to-end through the final ISA.
  if (!writeHsacoPath.empty()) {
    std::string target;
    if (writeHsacoKernel.empty()) {
      if (kernelNames.size() != 1) {
        llvm::errs() << "raise_cli: --write-hsaco requires --kernel=<name> "
                        "when the code object has "
                     << kernelNames.size() << " kernels\n";
        return 2;
      }
      target = kernelNames.front();
    } else {
      bool found = false;
      for (const auto &kn : kernelNames)
        if (kn == writeHsacoKernel) {
          target = kn;
          found = true;
          break;
        }
      if (!found) {
        llvm::errs() << "raise_cli: kernel '" << writeHsacoKernel
                     << "' not found in " << coPath << "\n";
        return 2;
      }
    }
    std::string effectiveTargetIsa = targetIsa.empty() ? isa : targetIsa;
    COMGR::hotswap::PipelineOptions pipelineOptions;
    pipelineOptions.EnableWritelaneRewrite = EnableWritelaneRewrite;
    pipelineOptions.EnableWaveNative = EnableWaveNative;
    pipelineOptions.AssumeHipGlobalOffsetZero = AssumeHipGlobalOffsetZeroOpt;
    pipelineOptions.OptLevel = std::min<unsigned>(OptLevel, 3);
    auto pipe = COMGR::hotswap::runPipeline(coData, isa, effectiveTargetIsa,
                                            target, pipelineOptions);
    if (!pipe.Success) {
      llvm::errs() << "raise_cli: pipeline failed for kernel '" << target
                   << "' (lifted=" << pipe.LiftedCount << "/" << pipe.TotalCount
                   << ", failure='" << pipe.FailDetail << "')\n";
      return 1;
    }
    FILE *fp = std::fopen(writeHsacoPath.c_str(), "wb");
    if (!fp) {
      llvm::errs() << "raise_cli: cannot open " << writeHsacoPath
                   << " for writing\n";
      return 2;
    }
    size_t hsacoSize = pipe.Hsaco ? pipe.Hsaco->getBufferSize() : 0;
    const char *hsacoData = pipe.Hsaco ? pipe.Hsaco->getBufferStart() : nullptr;
    size_t wrote = std::fwrite(hsacoData, 1, hsacoSize, fp);
    std::fclose(fp);
    if (wrote != hsacoSize) {
      llvm::errs() << "raise_cli: short write to " << writeHsacoPath << " ("
                   << wrote << " of " << hsacoSize << " bytes)\n";
      return 2;
    }
    llvm::errs() << "raise_cli: wrote " << hsacoSize << " byte HSACO for "
                 << "kernel '" << target << "' to " << writeHsacoPath
                 << " (lifted " << pipe.LiftedCount << "/" << pipe.TotalCount
                 << ")\n";
    return 0;
  }

  int totalKernels = 0, okKernels = 0, failKernels = 0, crashKernels = 0;

  for (auto &kName : kernelNames) {
    ++totalKernels;
    auto *shm = static_cast<KernelRaiseStats *>(
        mmap(nullptr, sizeof(KernelRaiseStats), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (shm == MAP_FAILED) {
      llvm::errs() << "raise_cli: mmap failed\n";
      return kMmapFailedExitCode;
    }
    auto UnmapShm = llvm::scope_exit([&] {
      if (munmap(shm, sizeof(KernelRaiseStats)) != 0)
        llvm::errs() << "raise_cli: munmap failed: errno=" << errno << "\n";
    });
    std::memset(shm, 0, sizeof(KernelRaiseStats));

    auto kernelExtentOrErr =
        COMGR::hotswap::findKernelSymbolExtent(coData, kName);
    if (!kernelExtentOrErr) {
      std::string err = llvm::toString(kernelExtentOrErr.takeError());
      llvm::errs() << "raise_cli: kernel '" << kName
                   << "' extent lookup failed: " << err << "\n";
      ++failKernels;
      llvm::outs() << "FAIL " << kName
                   << " -> __kernel_extent__ "
                      "[KernelSymbolExtentLookupFailed]\n";
      continue;
    }
    uint64_t kernelOffset = kernelExtentOrErr->Offset;
    uint64_t kernelSize = kernelExtentOrErr->Size;

    // Flush stdout so the child doesn't inherit pending bytes that
    // would re-emit after fork().
    llvm::outs().flush();

    int ChildOutputFD = -1;
    llvm::SmallString<128> ChildOutputPath;
    if (std::error_code EC = llvm::sys::fs::createTemporaryFile(
            "raise-cli", "out", ChildOutputFD, ChildOutputPath)) {
      llvm::errs() << "raise_cli: could not create child output file: "
                   << EC.message() << "\n";
      return kChildOutputFileFailedExitCode;
    }
    auto RemoveChildOutput = llvm::scope_exit([&] {
      if (std::error_code EC = llvm::sys::fs::remove(ChildOutputPath))
        llvm::errs() << "raise_cli: could not remove child output file "
                     << ChildOutputPath << ": " << EC.message() << "\n";
    });

    pid_t pid = fork();
    if (pid == 0) {
      llvm::raw_fd_ostream ChildOut(ChildOutputFD, /*shouldClose=*/true);
      // Silence the child's stderr: LLVM chatters a lot, and kerneldex
      // only cares about OK/FAIL on stdout plus the last stderr line
      // when the process as a whole crashes.
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
      auto metaOrErr = COMGR::hotswap::extractKernelMeta(coData, kName);
      COMGR::hotswap::KernelMeta meta;
      if (metaOrErr) {
        meta = std::move(*metaOrErr);
      } else {
        llvm::consumeError(metaOrErr.takeError());
      }
      COMGR::hotswap::RaiseStats Stats;
      llvm::Expected<COMGR::hotswap::RaiseResult> RaisedOrErr =
          COMGR::hotswap::raiseToIR(
              text.Bytes, isa, kName, meta, kernelOffset, kernelSize, targetIsa,
              EnableWritelaneRewrite, EnableWaveNative,
              AssumeHipGlobalOffsetZeroOpt, text.Address, text.ImageSections,
              functionExtents, &Stats);
      if (RaisedOrErr) {
        COMGR::hotswap::RaiseResult Raised = std::move(*RaisedOrErr);
        shm->done = true;
        shm->success = true;
        shm->lifted = Stats.LiftedCount;
        shm->total = Stats.TotalCount;
        shm->numDroppedFailures = 0;

        ChildOut << "OK " << kName << " (" << Stats.LiftedCount << "/"
                 << Stats.TotalCount << ")\n";
      } else {
        shm->done = true;
        shm->success = false;
        shm->lifted = Stats.LiftedCount;
        shm->total = Stats.TotalCount;
        shm->numDroppedFailures = 0;

        llvm::SmallVector<FailureBucketKey> Seen;
        int NumEmittedFailures = 0;
        auto EmitFailure = [&](const COMGR::hotswap::RaiseFailure &Failure,
                               llvm::StringRef Prefix) {
          FailureBucketKey Key = failureBucketKey(Failure);
          for (const FailureBucketKey &SeenKey : Seen) {
            if (SeenKey == Key)
              return;
          }
          if (NumEmittedFailures >= kMaxTrackedFailures) {
            ++shm->numDroppedFailures;
            return;
          }
          Seen.push_back(std::move(Key));
          // TODO: collect stats independently of RaiseResult and report here
          // again. int Lifted = Prefix == "FAIL" ? raised.LiftedCount : -1, int
          // Total = Prefix == "FAIL" ? raised.TotalCount : -1;
          printBatchFailureLine(ChildOut, Prefix, kName, Failure);
          ++NumEmittedFailures;
        };

        if (NumEmittedFailures == 0) {
          llvm::handleAllErrors(
              RaisedOrErr.takeError(),
              [&](COMGR::hotswap::RaiseFailure &Failure) {
                EmitFailure(Failure, NumEmittedFailures == 0 ? "FAIL" : "ALSO");
              },
              [&](const llvm::ErrorInfoBase &Err) {
                std::string RenderedFailure =
                    "FAIL: raiseToIR returned failure without a structured "
                    "reason: " +
                    Err.message();
                llvm::errs() << " (" << RenderedFailure << ")\n";
              });
        }

        if (shm->numDroppedFailures > 0) {
          ChildOut << "ALSO " << kName << " -> __truncated__ [+"
                   << shm->numDroppedFailures
                   << " unique blockers not shown]\n";
        }
      }

      ChildOut.flush();
      if (ChildOut.has_error())
        _exit(kChildOutputWriteFailedExitCode);
      _exit(0);
    }

    if (close(ChildOutputFD) != 0)
      llvm::errs() << "raise_cli: close child output file failed: errno="
                   << errno << "\n";

    if (pid < 0) {
      llvm::errs() << "raise_cli: fork failed\n";
      return kForkFailedExitCode;
    }

    int st = 0;
    bool WaitFailed = false;
    while (waitpid(pid, &st, 0) < 0) {
      if (errno == EINTR)
        continue;
      llvm::errs() << "raise_cli: waitpid failed: errno=" << errno << "\n";
      ++crashKernels;
      llvm::outs() << "FAIL " << kName
                   << " -> __crash__ [waitpid_failed]\n";
      WaitFailed = true;
      break;
    }
    if (WaitFailed)
      continue;

    auto ReplayOrCrash = [&]() {
      if (!replayChildOutput(ChildOutputPath)) {
        ++crashKernels;
        llvm::outs() << "FAIL " << kName << " -> __crash__ [output_missing]\n";
        return false;
      }
      return true;
    };

    bool ChildExitedCleanly = WIFEXITED(st) && WEXITSTATUS(st) == 0;
    if (!shm->done || !ChildExitedCleanly) {
      // Child never wrote the shm marker, or died by signal, or exited
      // with a nonzero status: surface this as a FAIL row with a
      // synthetic mnemonic so kerneldex still counts the kernel.
      ++crashKernels;
      llvm::outs() << "FAIL " << kName << " -> __crash__ ["
                   << childCrashFormat(st, shm->done) << "]\n";
    } else if (shm->success) {
      if (ReplayOrCrash())
        ++okKernels;
    } else {
      if (ReplayOrCrash())
        ++failKernels;
    }
  }

  llvm::errs() << "raise_cli: " << totalKernels << " kernels, " << okKernels
               << " ok, " << failKernels << " fail, " << crashKernels
               << " crash (" << coPath << ")\n";

  return (failKernels + crashKernels) == 0 ? 0 : 1;
}
