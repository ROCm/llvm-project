#ifndef HOTSWAP_TRANSPILER_PIPELINE_H
#define HOTSWAP_TRANSPILER_PIPELINE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstdint>
#include <string>

namespace COMGR::hotswap {

struct PipelineTimings {
  double totalSeconds = 0.0;
  double listKernelsSeconds = 0.0;
  double extractTextSeconds = 0.0;
  double createTempDirSeconds = 0.0;
  double raiseSeconds = 0.0;
  double writeIrSeconds = 0.0;
  double optSeconds = 0.0;
  double llcSeconds = 0.0;
  double linkSeconds = 0.0;
  double readHsacoSeconds = 0.0;
  double collectMetadataSeconds = 0.0;
};

struct PipelineOptions {
  bool EnableWritelaneRewrite = true;
  bool EnableWaveNative = true;
  bool CollectTimings = false;
  // Optimization level (0-3) for the in-process opt + llc codegen stages.
  unsigned OptLevel = 0;
  // HIP launches handled by COMGR's HotSwap runtime have zero HSA grid-global
  // offset, so hidden_global_offset_{x,y,z} can be synthesized as zero.
  // Standalone callers keep this false and reject those source hidden args.
  bool AssumeHipGlobalOffsetZero = false;
  // Force ScaledModuloReplicationProjection for wave32->wave64 cross-widening.
  // Production never sets this -- the raiser auto-selects the projection on the
  // relevant C5 refusal. It exists so lit tests can exercise the projection
  // (and its refusals) on kernels that do not themselves trigger the refusal.
  bool ForceScaledModrep = false;
};

struct PipelineResult {
  std::unique_ptr<llvm::MemoryBuffer> Hsaco;
  PipelineTimings Timings;
  std::string FailMnemonic;
  std::string FailKernel;
  std::string FailReason;
  std::string FailFormat;
  std::string FailDetail;
  uint64_t FailOffset = 0;
  // Successful raises can still carry proof-relevant attribution. Today this
  // records C5 predicate-chain sites accepted under a projection-specific
  // proof (for example single-source-wave MODREP with no active replica
  // lanes); loader proof logs surface these fields on `hotswap_result`.
  int C5SuppressedCount = 0;
  std::string C5SuppressionReason;
  bool UsesScratchPrivateSegment = false;
  uint32_t SourcePrivateSegmentFixedSize = 0;
  bool TargetEnablePrivateSegment = false;
  uint32_t TargetPrivateSegmentFixedSize = 0;
  int LiftedCount = 0;
  int TotalCount = 0;
  // ScaledModuloReplicationProjection requirement for the (single) transpiled
  // kernel: the factor (W_t/W_s) the launch runtime must scale the block's x
  // extent by. 1 means no scaling. See raiser.h.
  unsigned ScaledDispatchFactor = 1;
  bool Success = false;
};

/// End-to-end pipeline: HSACO binary -> raise to LLVM IR -> llc -> HSACO.
/// Raises using `SourceISA` and lowers to `TargetISA`.  Single-ISA
/// callers pass the same string for both -- see the single-ISA note
/// below for why there is no separate 3-string overload.
///
///
/// Single-ISA convention: pass the same ISA string for both
/// `SourceISA` and `TargetISA` (e.g. `runPipeline(data, "gfx942",
/// "gfx942", "kernel", ...)`).  Earlier revisions exposed a separate
/// 3-string overload `(data, targetISA, kernel, ...)` to elide the
/// repetition, but that overload silently misbehaved under C++
/// overload resolution: a cross-arch call `runPipeline(data,
/// "gfx1250", "gfx942", "kernel_name")` with four `const char *`
/// literals picked up the 3-string overload (because `const char *
/// -> bool` is a *standard* pointer-to-bool conversion that outranks
/// the user-defined `const char * -> std::string` conversion needed
/// for the 4-string cross-arch overload).  The 4th literal was then
/// bound to `EnableWritelaneRewrite` as `true` and `KernelName`
/// silently became `"gfx942"`, which downstream surfaced as
/// `UserSgprLayout::fromKernelMeta: kernel 'gfx942' has no parsed
/// kernel descriptor` -- an easy-to-miss silent miscompile of the
/// API contract itself.  Removing the 3-string overload and
/// requiring the canonical 4-string form is the only way to make
/// the ambiguity unrepresentable (SFINAE / `string_view`-typed
/// alternatives were evaluated and rejected: every candidate
/// preserved some path where `const char *` beats `std::string`
/// under standard conversions).  Callers repeat the ISA string
/// instead; the ~5 single-ISA call sites that did this pre-fix are
/// all updated to the two-string form.
PipelineResult runPipeline(llvm::MemoryBufferRef CodeObjectData,
                           llvm::StringRef SourceISA, llvm::StringRef TargetISA,
                           llvm::StringRef KernelName,
                           PipelineOptions Options = {});

/// Raise and lower ALL kernels in a code object, producing a single merged
/// HSACO containing every kernel.  Returns success only if every kernel was
/// raised and compiled. On raise failure, the `fail*` fields carry the
/// structured `RaiseFailure` details for proof logs and corpus summaries.
///
/// `EnableWritelaneRewrite` / `EnableWaveNative` plumb through to the
/// per-kernel `raiseToIR` calls; see `raiser.hpp` for the contract and
/// the in-tree-debug-only caveat.
PipelineResult runPipelineAllKernels(llvm::MemoryBufferRef CodeObjectData,
                                     llvm::StringRef SourceISA,
                                     llvm::StringRef TargetISA,
                                     PipelineOptions Options = {});

/// Process-global "strict mode" toggle, controlled by the
/// `HSA_HOTSWAP_STRICT` environment variable. When set to a non-empty
/// value the hotswap transpiler promotes known silent-miscompile sites
/// to honest refusals instead of warning-and-continue. Today this
/// covers two sites flagged by the corpus runner's `INTEGRATION_GAP.md`
/// investigation:
///
///   * `s_setreg_imm32_b32 mode, imm` writes to the MODE register
///     (handle-sopk.cpp): silently dropped in non-strict mode but the
///     kernel may rely on the FP rounding / denormal / IEEE / FTZ bits
///     being set.
///   * Unsupported or unproved `llvm.amdgcn.implicitarg.ptr` lifts
///     (handle-smem.cpp): strict mode refuses dynamic source implicit-arg
///     offsets, unmatched source hidden-arg metadata, and hidden fields without
///     explicit source-ABI synthesis or target-ABI identity mapping. Supported
///     fields are materialized by `source-hidden-args.cpp`, not by applying
///     source byte offsets to the target runtime's hidden-arg block.
///
/// Parsed once on first call (`std::getenv("HSA_HOTSWAP_STRICT")`); the
/// callers (handler implementations) read the flag without round-tripping
/// through the OS allocator on every instruction. The GPT-OSS runner sets
/// `HSA_HOTSWAP_STRICT=1`; `compare_correctness` and the gtest binary do
/// not, so existing GPU tests stay passing.
bool isStrictMode();

/// Process-global `HSA_HOTSWAP_DUMP_INPUT` toggle (parsed once). When set to
/// `1`, debug dumps additionally include the raw input code object and the
/// per-kernel source disassembly. Shared by the pipeline and the raiser so
/// both agree on when to produce those artifacts.
bool wantDumpInput();

class ScopedStrictMode {
public:
  explicit ScopedStrictMode(bool Enabled);
  ~ScopedStrictMode();

  ScopedStrictMode(const ScopedStrictMode &) = delete;
  ScopedStrictMode &operator=(const ScopedStrictMode &) = delete;

private:
  bool PreviousActive = false;
  bool PreviousValue = false;
};

} // namespace COMGR::hotswap

#endif
