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
// fork), dumps the raised LLVM IR for a single kernel on stdout, and
// leaves stderr alone so FileCheck can match warnings / abort-gate
// diagnostics. Selects the only kernel when the code object has one, or
// requires the ``=<kernel>`` form when there are multiple. Exits 0 iff
// the kernel raised successfully; non-zero otherwise.
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
    EmitIrOpt("emit-ir", cl::ValueOptional, cl::value_desc("kernel"),
              cl::desc("Dump raised LLVM IR for a single kernel on stdout "
                       "(no fork; stderr left alone for FileCheck)."));

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
      "stdout; --emit-ir and --write-hsaco select the single-kernel modes.\n");

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

  // --emit-ir path — no fork, no stderr redirect. Used by lit tests that
  // FileCheck the raised IR on stdout and the raiser diagnostics on
  // stderr. One kernel per invocation.
  if (emitIr) {
    std::string target;
    if (emitIrKernel.empty()) {
      if (kernelNames.size() != 1) {
        llvm::errs() << "raise_cli: --emit-ir requires =<kernel> when the "
                        "code object has "
                     << kernelNames.size() << " kernels\n";
        return 2;
      }
      target = kernelNames.front();
    } else {
      bool found = false;
      for (const auto &kn : kernelNames)
        if (kn == emitIrKernel) {
          target = kn;
          found = true;
          break;
        }
      if (!found) {
        llvm::errs() << "raise_cli: kernel '" << emitIrKernel
                     << "' not found in " << coPath << "\n";
        return 2;
      }
    }
    auto metaOrErr = COMGR::hotswap::extractKernelMeta(coData, target);
    if (!metaOrErr) {
      llvm::errs() << "raise_cli: kernel '" << target << "' metadata: "
                   << llvm::toString(metaOrErr.takeError()) << "\n";
      return 1;
    }
    COMGR::hotswap::KernelMeta meta = std::move(*metaOrErr);
    auto kernelOffsetOrErr = COMGR::hotswap::findKernelSymbolOffset(coData, target);
    if (!kernelOffsetOrErr) {
      std::string err = llvm::toString(kernelOffsetOrErr.takeError());
      llvm::errs() << "raise_cli: kernel '" << target
                   << "' offset lookup failed: " << err << "\n";
      return 1;
    }
    uint64_t kernelOffset = *kernelOffsetOrErr;
    auto raised = COMGR::hotswap::raiseToIR(text.Bytes, isa, target, meta,
                                        kernelOffset, targetIsa,
                                        EnableWritelaneRewrite,
                                        EnableWaveNative);
    if (!raised.Success) {
      // Contract: raiseToIR only populates RaiseResult::IrText on the
      // success path (the last write before setting `success = true`),
      // so we cannot dump partial IR here. Callers that need stderr
      // diagnostics (abort-gate lit tests, etc.) FileCheck the raiser's
      // stderr — we leave that untouched.
      COMGR::hotswap::RaiseFailure Failure =
          raised.Failure.hasFailed()
              ? raised.Failure
              : COMGR::hotswap::RaiseFailure::internalFailure(
                    "raiseToIR returned failure without a structured reason");
      llvm::errs() << "raise_cli: kernel '" << target << "' failed to raise: "
                   << COMGR::hotswap::formatRaiseFailure(Failure) << "\n";
      return 1;
    }
    llvm::outs().write(raised.IrText.data(), raised.IrText.size());
    return 0;
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

    auto kernelOffsetOrErr = COMGR::hotswap::findKernelSymbolOffset(coData, kName);
    if (!kernelOffsetOrErr) {
      std::string err = llvm::toString(kernelOffsetOrErr.takeError());
      llvm::errs() << "raise_cli: kernel '" << kName
                   << "' offset lookup failed: " << err << "\n";
      ++failKernels;
      llvm::outs() << "FAIL " << kName
                   << " -> __kernel_offset__ "
                      "[KernelSymbolOffsetLookupFailed]\n";
      continue;
    }
    uint64_t kernelOffset = *kernelOffsetOrErr;

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
      auto raised = COMGR::hotswap::raiseToIR(text.Bytes, isa, kName, meta,
                                          kernelOffset, targetIsa,
                                          EnableWritelaneRewrite,
                                          EnableWaveNative);
      shm->done = true;
      shm->success = raised.Success;
      shm->lifted = raised.LiftedCount;
      shm->total = raised.TotalCount;
      shm->numDroppedFailures = 0;

      if (raised.Success) {
        ChildOut << "OK " << kName << " (" << raised.LiftedCount << "/"
                 << raised.TotalCount << ")\n";
      } else {
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
          printBatchFailureLine(ChildOut, Prefix, kName, Failure,
                                Prefix == "FAIL" ? raised.LiftedCount : -1,
                                Prefix == "FAIL" ? raised.TotalCount : -1);
          ++NumEmittedFailures;
        };

        for (const COMGR::hotswap::RaiseFailure &Failure : raised.AllFailures)
          EmitFailure(Failure, NumEmittedFailures == 0 ? "FAIL" : "ALSO");

        if (NumEmittedFailures == 0) {
          COMGR::hotswap::RaiseFailure Failure =
              raised.Failure.hasFailed()
                  ? raised.Failure
                  : COMGR::hotswap::RaiseFailure::internalFailure(
                        "raiseToIR returned failure without a structured reason");
          EmitFailure(Failure, "FAIL");
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
