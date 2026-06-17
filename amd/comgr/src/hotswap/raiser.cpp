//===- raiser.cpp - Hotswap MC -> LLVM IR raiser scaffolding --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Disassembles a kernel's ELF text section into a typed `DecodedInst` stream
// and builds an `llvm::Module` with a kernel function whose body is `ret void`.
// See `raiser.h` for the full raise pipeline (ELF ingestion -> decode ->
// per-format handlers -> post-raise analyses).
//
//===----------------------------------------------------------------------===//

#include "raiser.h"
#include "amdgpu-formats.h"
#include "code-object-utils.h"
#include "decode.h"
#include "canonical-op.h"
#include "isa-profile.h"
#include "decoded-inst.h"
#include "parsed-reg.h"

#include "../comgr.h"
#include "mc-state.h"
#include "opcode-map.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "reg-file.h"
#include "kernarg-layout.h"
#include "raise-context.h"
#include "canonical-op-attrs.h"
#include "setpc-analysis.h"
#include "source-hidden-args.h"
#include "user-sgpr-layout.h"
#include "wave-projection.h"
#include "wave-size-obstruction.h"
#include "handlers.h"
#include "rewrite-cross-lane-divergent.h"
#include "c5-predicate-chain-classifier.h"
#include "ocml-runtime.h"
#include "tdm-runtime.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>
#include <utility>

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

llvm::DenseSet<uint64_t> collectInstructionOffsets(ArrayRef<DecodedInst> Insts) {
  llvm::DenseSet<uint64_t> Offsets;
  for (const DecodedInst &Di : Insts)
    Offsets.insert(Di.Offset);
  return Offsets;
}

void mergeDecodeResult(DecodeResult &Base, DecodeResult &&Extra) {
  llvm::DenseSet<uint64_t> Seen = collectInstructionOffsets(Base.Insts);
  for (DecodedInst &Di : Extra.Insts) {
    if (!Seen.insert(Di.Offset).second)
      continue;
    Base.Insts.push_back(std::move(Di));
  }
  llvm::sort(Base.Insts, [](const DecodedInst &A, const DecodedInst &B) {
    return A.Offset < B.Offset;
  });
  Base.BlockStarts.insert(Extra.BlockStarts.begin(), Extra.BlockStarts.end());
}

enum class ThreadLoopDecision {
  NotApplicable,
  EligibleButGateOff,
  EligibleAndGateOn,
  Ineligible,
};

struct ThreadLoopDecisionResult {
  ThreadLoopDecision Decision = ThreadLoopDecision::NotApplicable;
  std::string Reason;
};

ThreadLoopDecisionResult decideThreadLoopFallback(unsigned SourceWaveSize,
                                                  unsigned TargetWaveSize,
                                                  bool SgprForcedRefusal,
                                                  bool ThreadLoopEligible) {
  if (!SgprForcedRefusal)
    return {ThreadLoopDecision::NotApplicable, "no SGPR-forced refusal"};
  if (!ThreadLoopEligible) {
    return {ThreadLoopDecision::Ineligible,
            "SGPR-forced sink is outside the proven readlane/writelane -> "
            "explicit readfirstlane ThreadLoop class"};
  }
  if (TargetWaveSize <= SourceWaveSize) {
    return {ThreadLoopDecision::Ineligible,
            "thread-loop fallback is cross-widen-only"};
  }
  if ((TargetWaveSize % SourceWaveSize) != 0) {
    return {ThreadLoopDecision::Ineligible,
            "target wave size is not an integer multiple of source wave size"};
  }
  // Graduation gate for the narrow SGPR-forced post-raise refusal class.
  //
  // Objective trigger:
  //   * the SSA use-chain classifier has already refused a cross-widening
  //     writelane/readlane rewrite because the value flows into an explicit
  //     `llvm.amdgcn.readfirstlane` consumer; and
  //   * the target wave size is an integer multiple of the source wave size.
  //
  // This does not widen the rewrite allow-list. The original refusal remains
  // the proof obligation: only after the classifier names the proven
  // readfirstlane sink do we retry under ThreadLoopProjection, with the
  // rewrite disabled so source-wave-scoped readlane / writelane /
  // readfirstlane lowering owns the boundary. Other SGPR-forced sinks
  // (scalar memory operands, inline asm, unknown calls) still refuse loudly.
  constexpr bool kThreadLoopAutoActivateSgprForcedCrossWiden = true;
  if (kThreadLoopAutoActivateSgprForcedCrossWiden)
    return {ThreadLoopDecision::EligibleAndGateOn,
            "SGPR-forced cross-widen refusal is covered by ThreadLoopProjection"};
  return {ThreadLoopDecision::EligibleButGateOff,
          "eligible but graduation gate is off"};
}

static bool isSemOpInRange(CanonicalOp Op, CanonicalOp First, CanonicalOp Last) {
  auto V = static_cast<uint16_t>(Op);
  return V >= static_cast<uint16_t>(First) &&
         V <= static_cast<uint16_t>(Last);
}

static bool threadLoopUnsupportedWorkgroupMemoryOrBarrier(
    ArrayRef<DecodedInst> Insts, std::string &Detail) {
  for (const DecodedInst &Di : Insts) {
    StringRef Kind;
    switch (Di.CanonOp) {
    case CanonicalOp::S_BARRIER:
    case CanonicalOp::S_BARRIER_WAIT:
    case CanonicalOp::S_BARRIER_SIGNAL:
      Kind = "workgroup barrier";
      break;
    case CanonicalOp::BUFFER_LOAD_DWORD_LDS:
    case CanonicalOp::BUFFER_LOAD_DWORDX4_LDS:
    case CanonicalOp::TENSOR_LOAD_TO_LDS:
    case CanonicalOp::TENSOR_STORE_FROM_LDS:
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B8:
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B32:
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B64:
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B128:
      Kind = "LDS access";
      break;
    default:
      if (isSemOpInRange(Di.CanonOp, CanonicalOp::DS_LOAD_TR16_B128,
                         CanonicalOp::DS_SWIZZLE_B32))
        Kind = "LDS access";
      break;
    }

    if (!Kind.empty()) {
      Detail = (Twine("ThreadLoopProjection is not yet safe for kernels "
                      "containing ") +
                Kind + " (" + canonicalOpName(Di.CanonOp) + " at offset 0x" +
                Twine::utohexstr(Di.Offset) +
                "); barrier hoisting and LDS aliasing checks are still "
                "unimplemented, so refusing is safer than launching a "
                "translated kernel that can fault or miscompile.")
                   .str();
      return true;
    }
  }
  return false;
}

} // namespace

// parseReg, readOp32/64/ExecWidth, and OpResolver are in raise-context.h/cpp
// instructionWritesEXEC and the cross-wave gate live in wave-projection.h/cpp
// RaiseFailure + reasonString are in raise-failure.h/cpp

// ============================================================================
// Main raising function
// ============================================================================

static RaiseResult raiseToIRImpl(llvm::ArrayRef<uint8_t> TextBytes,
                                 llvm::StringRef SourceIsa,
                                 llvm::StringRef KernelName,
                                 const KernelMeta &Meta,
                                 uint64_t KernelOffset,
                                 uint64_t KernelSize,
                                 llvm::StringRef CompilationTargetIsa,
                                 bool EnableWritelaneRewrite,
                                 bool EnableWaveNative,
                                 bool ForceThreadLoopProjection,
                                 bool SuppressC5ForThreadLoopRoute) {
  RaiseResult Result;

  // Reject obviously-bad ISA inputs before reaching the MC stack -- an
  // empty or non-AMDGPU ISA string slips past `createMCSubtargetInfo`
  // (it returns a subtarget with no features) and only blows up later
  // in `createMCDisassembler` with an `llvm_unreachable`-flavoured
  // `LLVM ERROR: disassembly not yet supported for subtarget` that
  // aborts the process. Surface a structured failure instead.
  //
  // Callers may pass either the bare processor name (`gfx942`) or the
  // canonical AMDGPU ISA string (`amdgcn-amd-amdhsa--gfx942[:feat...]`).
  // Defer to Comgr's `parseTargetIdentifier` for the canonical form (it
  // handles the dash-separated Arch/Vendor/OS/Environ/Processor split
  // and the `:sramecc±:xnack±` feature suffix in one place);
  // `MCSubtargetInfo` only accepts the bare processor name, so we
  // forward `Ident.Processor` to the MC stack below.
  auto NormalizeIsa = [](StringRef Iso) -> StringRef {
    TargetIdentifier Ident;
    if (parseTargetIdentifier(Iso, Ident) == AMD_COMGR_STATUS_SUCCESS)
      return Ident.Processor;
    // Bare processor name (e.g. `gfx942`) -- not a 5-component canonical
    // ISA string. Return as-is and let the AMDGPU validator below decide.
    return Iso;
  };
  StringRef SourceCpu = NormalizeIsa(SourceIsa);
  if (SourceIsa.empty() ||
      AMDGPU::parseArchAMDGCN(SourceCpu) == AMDGPU::GK_NONE) {
    Result.Failure.Reason = RaiseFailureReason::BadInput;
    Result.Failure.Detail =
        (Twine("source ISA '") + SourceIsa +
         "' does not name an AMDGPU GPU")
            .str();
    return Result;
  }

  // Same normalisation for the target-side override (--target-isa on
  // raise_cli, or programmatic CompilationTargetIsa). Empty means
  // "translate in place -- reuse the source profile".
  StringRef TargetCpu = CompilationTargetIsa.empty()
                            ? CompilationTargetIsa
                            : NormalizeIsa(CompilationTargetIsa);

  // NOTE. The `HSA_HOTSWAP_WAVE_NATIVE=1` process-environment override
  // that lived here through the empirical graduation sweep (pre-
  // 2026-04-21) has been removed now that `enableWaveNative`
  // defaults to `true`. The override served one purpose -- flipping
  // every call-site's projection without editing each caller --
  // which is no longer needed. Keeping it around would subtly
  // break the opt-OUT path: `--disable-wave-native` on
  // `raise_cli` (and `enableWaveNative=false` on programmatic
  // callers) are how lit fixtures and operators pin MODREP for
  // projection-specific debugging, and a silent env-var that
  // unconditionally flips to WaveNative would defeat that. If
  // future evidence needs a global toggle, add a proper
  // `PipelineConfig` field rather than re-introducing the env var.

  MCState Mc;
  if (Error E = initMCState(Mc, SourceCpu)) {
    Result.Failure.Reason = RaiseFailureReason::BadInput;
    Result.Failure.Detail = toString(std::move(E));
    return Result;
  }

  ISAProfile Isa = ISAProfile::fromSubtarget(*Mc.SubtargetInfo);
  // When the caller does not specify a distinct compilation target we raise
  // in place and reuse the source profile; otherwise we spin up a throwaway
  // MCSubtargetInfo just to snapshot the target's feature bits.
  ISAProfile TargetIsa = Isa;
  std::unique_ptr<MCSubtargetInfo> TargetSti;
  if (!TargetCpu.empty()) {
    Expected<std::unique_ptr<MCSubtargetInfo>> StiOrErr =
        buildSubtargetInfo(*Mc.Target, TargetCpu);
    if (!StiOrErr) {
      Result.Failure.Reason = RaiseFailureReason::BadInput;
      Result.Failure.Detail = toString(StiOrErr.takeError());
      return Result;
    }
    TargetSti = std::move(*StiOrErr);
    TargetIsa = ISAProfile::fromSubtarget(*TargetSti);
  }

  // LLVMContext + common IR types are created here (earlier than they used
  // to be) so the WaveProjection has access to i32/i64 before the cross-
  // wave gate runs. The module is still created lazily in Phase 2 so
  // early-return paths (pre-translation aborts) don't leave behind a
  // half-built module.
  Result.Ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *Result.Ctx;
  auto *I32Ty = Type::getInt32Ty(C);
  auto *I64Ty = Type::getInt64Ty(C);

  // Projection choice.
  //
  // `ModuloReplicationProjection` is the long-standing default: it fans
  // each target lane onto `lane_id mod W_src` of the source EXEC mask
  // and truncates cross-wave ballots to source width. Correct under
  // the wave-size-obliviousness theorem (hotswap/docs/wave-size-
  // translation.md §6); insufficient for kernels whose WMMA -> MFMA
  // redistribute / collect pipeline needs hardware EXEC = -1 on the
  // upper half of the Wave64 target (lanes 32..63 would otherwise
  // never update their MFMA destination VGPRs -- see the file-header
  // comment in `wmma-lowering.cpp`).
  //
  // `WaveNativeProjection` is the opt-in alternative for wave32
  // source -> wave64 target. Its `emitInitialExec` calls
  // `@llvm.amdgcn.init_whole_wave` at kernel entry to force hardware
  // EXEC = -1 for the whole kernel body while saving the original
  // per-lane active mask into the (widened) EXEC alloca; every VGPR
  // write / memory store / LDS op already routes through
  // `emitUnderExec`, which rematerialises the per-lane predicate at
  // each side-effect site. The direction gate inside the
  // `WaveNativeProjection` constructor enforces that this projection
  // is only instantiated when `isa.isWave32() && !targetIsa.isWave32()`
  // -- other directions fatal-error loudly to prevent a decider bug
  // from silently picking an unsupported shape.
  //
  // Phantom-lane fallback to MODREP.  WaveNative's `init_whole_wave`
  // sets hardware EXEC = -1 and relies on SPE `emitUnderExec`
  // diamonds (gated by `saved_exec`) to keep inactive source lanes
  // from committing side effects.  That model is correct when every
  // target-wavefront lane has a source-kernel workitem -- i.e. when
  // the HSACO's `max_flat_workgroup_size` is at least
  // `targetWaveSize` so every launch fills the target wave.  When
  // `max_flat_workgroup_size < targetWaveSize` (the phantom-lane
  // regime, e.g. Triton's `num_warps=1` kernels whose source WG is
  // 32 on wave32 compiled for a wave64 target), the "extra" target
  // lanes have no source workitem: their `workitem.id.x()` is their
  // hardware lane index (e.g. 32..63 for a 32-thread block on
  // wave64), their VGPRs hold undef / dispatcher state, and their
  // cross-lane ops (`ds_bpermute`, `ds_swizzle`, `permlane*`) read
  // from / contribute to actively-masked source lanes with
  // undef-derived values -- producing addresses that fault on
  // subsequent SPE-gated loads (the active lane's pointer
  // arithmetic picks up undef data through a cross-lane op, then
  // the gated load fires with that poisoned address).  Empirically
  // surfaced by `compare_correctness`'s `matmul_fp16` /
  // `matmul_fp16_16x16` Triton recipes (HIP error 700 on every
  // shape under WaveNative; bumping `num_warps` to 2 fills the
  // target wavefront and eliminates the fault, confirming the
  // phantom-lane attribution).
  //
  // `ModuloReplicationProjection` leaves hardware EXEC at the
  // dispatcher's boot state (the source-wave-sized active mask,
  // with the target wave's upper lanes inactive) and uses
  // `lane_id mod W_src` to project the target mask onto the source
  // EXEC alloca.  Under MODREP, phantom lanes are hardware-inactive
  // for the entire kernel body -- every ISA instruction (VALU,
  // cross-lane, memory, control flow) is HW-EXEC-masked -- so
  // undef-VGPR contamination can't escape into active lanes.  The
  // trade-off is that MODREP cannot express WMMA -> MFMA layout
  // transposes that need all 64 target lanes active (see
  // `wmma-lowering.cpp`); those kernels will refuse at lift time
  // rather than silently running wrong.  That's the principled
  // outcome for the phantom-lane regime.
  const bool PhantomLaneRegime =
      Meta.MaxFlatWorkgroupSize > 0 &&
      static_cast<unsigned>(Meta.MaxFlatWorkgroupSize) < TargetIsa.WaveSize;
  const bool UseThreadLoop = ForceThreadLoopProjection;
  const bool UseWaveNative = !UseThreadLoop && EnableWaveNative &&
                             Isa.isWave32() && !TargetIsa.isWave32() &&
                             !PhantomLaneRegime;
  std::unique_ptr<WaveProjection> ProjectionPtr;
  if (UseThreadLoop) {
    ProjectionPtr = std::make_unique<ThreadLoopProjection>(
        Isa, TargetIsa, I32Ty, I64Ty);
    errs() << "transpiler: kernel '" << KernelName
           << "' selected ThreadLoopProjection (analysis-triggered "
              "cross-widen route; writelane/readlane rewrite may be "
              "disabled by the retry caller)\n";
  } else if (UseWaveNative) {
    ProjectionPtr = std::make_unique<WaveNativeProjection>(Isa, TargetIsa,
                                                             I32Ty, I64Ty);
  } else {
    ProjectionPtr = std::make_unique<ModuloReplicationProjection>(
        Isa, TargetIsa, I32Ty, I64Ty);
  }
  ProjectionPtr->setMaxFlatWorkgroupSize(Meta.MaxFlatWorkgroupSize);
  WaveProjection &Projection = *ProjectionPtr;

  if (!UseThreadLoop && EnableWaveNative && PhantomLaneRegime && Isa.isWave32() &&
      !TargetIsa.isWave32()) {
    // Log the fallback so operators can trace which kernels moved to
    // MODREP and why.  A regression that silently flips WaveNative's
    // selection on a phantom-lane kernel would then (re-)produce the
    // HIP-700 miscompile this fallback guards against.
    errs() << "transpiler: kernel '" << KernelName
           << "' is in phantom-lane regime (max_flat_workgroup_size="
           << Meta.MaxFlatWorkgroupSize << " < target wavefront width="
           << TargetIsa.WaveSize
           << "); falling back to ModuloReplicationProjection even "
              "though enableWaveNative=true, so phantom target lanes "
              "stay hardware-inactive and their undef-VGPR state "
              "cannot contaminate active-lane pointer arithmetic via "
              "cross-lane ops. See the block comment above in "
              "`raiser.cpp` for the full rationale.\n";
  }

  // Build opcode -> CanonicalOp map from MCInstrInfo
  OpcodeMap OpcMap;
  OpcMap.build(*Mc.InstrInfo);

  // Fail loudly if any MFMA-format CanonicalOp is missing a handler row. Cheap
  // startup walk that catches table drift before any kernel is lifted.
  verifyMFMACoverage(*Mc.InstrInfo, OpcMap);

  // Startup invariant: every MC opcode that implicitly defines EXEC must
  // map to a CanonicalOp that has `routesExecThroughStoreExec` set. Explicit-
  // operand EXEC writers (where EXEC is an operand value rather than a
  // TableGen def) stay the per-kernel Phase 1.5 gate's responsibility
  // since they depend on runtime operand values.
  verifyExecAttrCoverage(*Mc.InstrInfo, OpcMap);

  // ==== Phase 1: Disassemble + identify block boundaries ====
  //
  // The decode loop (and its two LLVM-drift guards) lives in decode.cpp so
  // this function stays focused on IR emission. decodeKernel returns a
  // linearised instruction stream + the set of CFG block-start offsets.
  if (KernelSize != 0 && KernelSize > UINT64_MAX - KernelOffset)
    report_fatal_error("transpiler: kernel decode extent overflows");
  const uint64_t KernelEndOffset =
      KernelSize == 0 ? 0 : KernelOffset + KernelSize;
  const uint64_t DecodeLimit =
      KernelEndOffset == 0 ? TextBytes.size() : KernelEndOffset;
  DecodeResult Decoded =
      decodeKernel(Mc, OpcMap,
                   ArrayRef<uint8_t>(TextBytes.data(), TextBytes.size()),
                   KernelOffset, KernelEndOffset);
  auto &Insts = Decoded.Insts;
  auto &BlockStarts = Decoded.BlockStarts;

  // ==== Phase 1.1: s_set_pc_i64 analysis ====
  //
  // Classify every s_set_pc_i64 site (Pattern A direct branch /
  // Pattern B subroutine return / Unresolvable) and discover the
  // extra basic-block leaders the indirect control-flow implies
  // (Pattern A targets + Pattern B return targets + the offset
  // immediately following each set-PC, which is otherwise unreachable
  // by linear fall-through). Merging the extra leaders into
  // `blockStarts` here is mandatory: Phase 3 only creates LLVM
  // BasicBlocks for offsets in this set, and the handler / call-site
  // rewrite both look up those BBs via `ctx.lookupBB`.
  // See setpc-analysis.h + canonical-op.h's `S_SET_PC_I64` doc for the
  // analysis contract.
  SetPcAnalysis SetpcAnalysis;
  // SetPC analysis can discover helper/subroutine regions that ordinary linear
  // decode did not reach. Decode every newly-discovered in-kernel target to a
  // fixpoint; crossing the selected symbol extent is a boundary violation, and
  // an in-extent target that cannot decode is a hard CFG recovery failure.
  while (true) {
    SetpcAnalysis = analyseSetPC(Insts, BlockStarts, Mc);
    llvm::DenseSet<uint64_t> InstOffsets = collectInstructionOffsets(Insts);
    bool AddedHelperRegion = false;
    for (uint64_t Addr : SetpcAnalysis.ExtraBlockStarts) {
      if (Addr < KernelOffset || Addr >= DecodeLimit) {
        Result.Failure = RaiseFailure::kernelBoundaryViolation(
            KernelName, Addr,
            "s_set_pc_i64 analysis discovered a target outside the selected "
            "kernel extent");
        return Result;
      }
      if (InstOffsets.count(Addr))
        continue;
      DecodeResult HelperDecoded =
          decodeKernel(Mc, OpcMap,
                       ArrayRef<uint8_t>(TextBytes.data(), TextBytes.size()),
                       Addr, KernelEndOffset, KernelOffset);
      if (HelperDecoded.Insts.empty()) {
        Result.Failure = RaiseFailure::kernelBoundaryViolation(
            KernelName, Addr,
            "s_set_pc_i64 analysis discovered an in-extent target that could "
            "not be decoded");
        return Result;
      }
      mergeDecodeResult(Decoded, std::move(HelperDecoded));
      AddedHelperRegion = true;
    }
    if (!AddedHelperRegion)
      break;
  }
  for (uint64_t Addr : SetpcAnalysis.ExtraBlockStarts) {
    if (Addr < KernelOffset || Addr >= DecodeLimit) {
      Result.Failure = RaiseFailure::kernelBoundaryViolation(
          KernelName, Addr,
          "s_set_pc_i64 analysis discovered a final block start outside the "
          "selected kernel extent");
      return Result;
    }
    BlockStarts.insert(Addr);
  }

  Result.TotalCount = static_cast<int>(Insts.size());

  {
    raw_string_ostream DisOs(Result.DisasmText);
    for (const auto &Di : Insts) {
      DisOs << format_hex_no_prefix(Di.Offset, 8) << ":  " << Di.FullText
            << "\n";
    }
  }

  // ==== Phase 1.4: Cross-wave legacy diagnostic (LLVM_DEBUG) ====
  //
  // Kept as a fallback diagnostic under `-debug-only=wave-projection`;
  // the structured classifier in Phase 1.4.5 below is the primary
  // decision surface. See wave-projection.cpp for the text of the
  // legacy diagnostic.
  emitCrossWaveWarning(Projection, Mc, Insts, SourceIsa,
                       CompilationTargetIsa);

  // ==== Phase 1.4.5: Wave-size obstruction classifier
  // (hotswap/docs/wave-size-translation.md §7) ====
  //
  // The classifier walks the decoded instruction stream and tags every
  // site that violates the wave-size-obliviousness theorem (see
  // wave-size-translation.md §6 for the precise definition). The
  // decider then applies the 3-outcome procedure:
  //   (a) no obstructions, or every obstruction is covered by an
  //       implemented rewrite -> emit modulo-replication.
  //   (b) at least one obstruction has a rewrite structurally
  //       recognised but not yet implemented (the "Pending rewrite"
  //       table in wave-size-translation.md §7) -> refuse with a
  //       `CrossWaveShuffleRewritePending` diagnostic naming the P-item.
  //   (c) at least one obstruction has no rewrite in the decision
  //       procedure's unrewritable table -> refuse with the kind-
  //       specific CrossWave* diagnostic (`CrossWaveLaneIdLeak`,
  //       `CrossWaveUnrewritableShuffle`, `CrossWaveReplicaRace`,
  //       `CrossWaveLanePredicatedExec`).
  //
  // Refusal diagnostics are written to `errs()` (user-visible) AND the
  // full per-site trace is routed through LLVM_DEBUG so operators can
  // inspect the oblivious/pass path under `-debug-only=wave-projection`
  // without recompiling.
  // Number of `WaveIdLiftScalarized` sites the classifier matched.
  // Needed after Phase 6.5 for the rewrite-pass safety net (see
  // below): when this is > 0, the rewrite pass is *expected* to have
  // rewritten at least one divergent writelane/readlane site; if it
  // rewrote zero, the oracle disagrees with the syntactic
  // classifier and we refuse post-raise rather than emit silently
  // unchanged IR that scalarises the divergent wave_id lift.
  unsigned ClassifierWaveIdLiftScalarizedSites = 0;
  {
    ObstructionReport Report =
        buildObstructionReport(Insts, Mc, Isa, TargetIsa,
                               EnableWritelaneRewrite);
    for (const auto &S : Report.Sites)
      if (S.Kind == ObstructionKind::WaveIdLiftScalarized)
        ++ClassifierWaveIdLiftScalarizedSites;
    std::string Trace = renderObstructionTrace(
        Report, KernelName, SourceIsa,
        CompilationTargetIsa.empty() ? SourceIsa : CompilationTargetIsa,
        Isa.WaveSize, TargetIsa.WaveSize);
    LLVM_DEBUG(dbgs() << Trace);
    if (Report.hasUnrewritable() || Report.hasPendingRewrite()) {
      RaiseFailure F = selectFailureFromReport(Report);
      // The factory names the class in `format`; surface the full trace in
      // `detail` so diagnostics can carry the per-site context forward without
      // re-invoking the classifier.
      if (!F.Detail.empty())
        F.Detail += "\n";
      F.Detail += Trace;
      // `format_hex(value, width)` prepends "0x" itself; do NOT add a
      // literal "0x" here or the output will read "0x0x...". Use
      // `format_hex_no_prefix` if a manual prefix is desired (the
      // trace-renderer below uses that variant).
      errs() << "transpiler: pre-translation abort: " << F.Format
             << " on '" << F.Mnemonic << "' at offset "
             << format_hex(F.Offset, 1) << " \u2014 "
             << (Report.firstUnrewritable()
                     ? "no rewrite in wave-size-translation.md "
                       "\u00a77's unrewritable table"
                     : "rewrite pending (wave-size-translation.md "
                       "\u00a77's pending-rewrite table)")
             << "\n"
             << Trace;
      Result.Failure = std::move(F);
      return Result;

    }
  }

  // ==== Phase 1.5: SPE A-level gate (EXEC-writer attribute check) ====
  //
  // SPE (SIMT Predicated Execution) is correct only when every runtime
  // change to EXEC either (a) propagates through the EXEC alloca via a
  // handler we have audited, or (b) follows the standard dataflow form
  // `exec = f(old_exec, sgprs, ...)` where `f` is a bitwise / shift /
  // move / compare-based scalar op -- the IR's live EXEC value then
  // matches the hardware EXEC that the backend re-materialises when it
  // lowers our predicated-store diamonds back to v_cmpx / s_and_saveexec
  // pairs. Anything outside this set risks silently generating IR that
  // looks well-typed but diverges from hardware semantics.
  //
  // The allow-list lives as per-CanonicalOp attributes in `sem_op_attrs.{hpp,
  // cpp}`; `verifyExecAttrCoverage` above already enforces it for
  // implicit-def EXEC writers at startup. This per-kernel scan covers
  // the remaining case: explicit-operand EXEC writers (e.g.
  // `s_mov_b32 exec_lo, s2`) where "writes EXEC" depends on the
  // runtime operand value rather than the MCInstrDesc alone.
  for (const DecodedInst &Di : Insts) {
    if (!instructionWritesEXEC(Di, Mc))
      continue;
    if (getCanonicalOpAttrs(Di.CanonOp).RoutesExecThroughStoreExec)
      continue;
    Result.Failure = RaiseFailure::speUnsafeExecWriter(Di);
    errs() << "transpiler: pre-translation abort: '" << Di.RawMnemonic
           << "' writes EXEC but its CanonicalOp (" << canonicalOpName(Di.CanonOp)
           << ") is not marked routesExecThroughStoreExec. Auditing "
              "the handler path against SPE (lane-active predication "
              "assumption) is required before declaring the CanonicalOp in "
              "the handler's get*Attrs() registration.\n";
    return Result;
  }

  // ==== Phase 2: Build LLVM IR module + function ====
  // LLVMContext + i32/i64 were created earlier for the WaveProjection.
  Result.Module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  TargetOptions Opts;
  std::unique_ptr<TargetMachine> Tm(Mc.Target->createTargetMachine(
      Triple("amdgcn-amd-amdhsa"),
      CompilationTargetIsa.empty() ? SourceIsa : CompilationTargetIsa,
      "", Opts, Reloc::PIC_));
  if (!Tm) {
    errs() << "transpiler: Failed to create TargetMachine\n";
    Result.Failure = RaiseFailure::targetMachineCreationFailed();
    return Result;
  }
  M.setDataLayout(Tm->createDataLayout());

  auto *VoidTy = Type::getVoidTy(C);
  auto *I1Ty = Type::getInt1Ty(C);
  auto *I8Ty = Type::getInt8Ty(C);
  auto *F32Ty = Type::getFloatTy(C);
  auto *PtrGlobalTy = PointerType::get(C, 1);

  // Build function signature: a single opaque
  // `ptr byref([N x i8]) align 16` placeholder whose only job is to
  // make the AMDGPU backend emit `kernarg_segment_size = N` and
  // `kernarg_segment_align = 16` in the lifted kernel's KD/metadata,
  // so the runtime's kernarg buffer reaches the kernel intact and
  // the metadata reports the AMDGPU ABI's 16-byte minimum.
  //
  // The handlers do NOT read this argument -- kernarg loads lift to
  // GEP+load against `amdgcn_kernarg_segment_ptr` and let the AMDGPU
  // backend re-select `s_load_*` against the kernarg segment. The
  // typed source-ABI signature (ptr addrspace(1) / i32 / i64 / per-
  // dword aggregate split) is therefore unnecessary on the lifted
  // side.
  //
  // Why `byref` + `align`: AMDGPULowerKernelArguments consults the
  // `align` parameter attribute only for byref kernel args (see
  // `MaybeAlign ParamAlign = IsByRef ? Arg.getParamAlign() :
  // std::nullopt;` in LLVM's `AMDGPULowerKernelArguments.cpp`). For
  // a non-byref `[N x i8]` arg, the IR-level alignment is the
  // type's natural alignment (1 byte), and the YAML metadata's
  // `.kernarg_segment_align` field reports a smaller value than the
  // ABI's 16-byte minimum. Using `byref` with an explicit
  // `align(16)` lets the backend honour the alignment without
  // forcing a vector or padding type, and the byref semantics --
  // "pointer to an aggregate that's actually placed in the kernarg
  // segment" -- match the placeholder's intent: a stable region of
  // `kernarg_segment_size` bytes that handlers don't need a typed
  // view of.
  //
  // AMDGPULowerKernelArguments skips load emission for arguments
  // that are `use_empty()` but still bumps the cumulative arg
  // offset, so the unused placeholder still contributes to
  // `kernarg_segment_size`.
  //
  // Test back-reference: every lit fixture under `lit_tests/` pins
  // either a `ptr addrspace(4)` GEP shape or an addrspace(1) global
  // GEP shape against the segment_ptr intrinsic -- none of them rely
  // on the kernarg buffer being a typed Function argument list.
  SmallVector<Type *, 1> ParamTypes;
  KernargLayout Kernargs;
  int ParamIdx = 0;
  Type *KernargByrefTy = nullptr;
  if (Meta.KernargSegmentSize > 0) {
    KernargByrefTy =
        ArrayType::get(I8Ty, static_cast<uint64_t>(Meta.KernargSegmentSize));
    ParamTypes.push_back(PointerType::get(C, /*addrspace=*/4));
    ParamIdx = 1;
  }
  Kernargs.ImplicitArgsBase = Meta.implicitArgsBase();
  Kernargs.Args = Meta.Args;
  Kernargs.KernargSegmentSize = Meta.KernargSegmentSize;

  auto *FuncTy = FunctionType::get(VoidTy, ParamTypes, false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  // Attach `byref([N x i8])` + `align(16)` to the placeholder kernarg
  // pointer. AMDGPULowerKernelArguments only honours param-align on
  // byref kernel args, so this combo is what gets the lifted KD's
  // kernarg-segment alignment to the AMDGPU ABI's 16-byte minimum
  // without forcing an aggregate / vector type for the parameter.
  if (KernargByrefTy != nullptr) {
    F->addParamAttr(0, Attribute::getWithByRefType(C, KernargByrefTy));
    F->addParamAttr(0, Attribute::getWithAlignment(C, Align(16)));
  }
  {
    // Pin the workgroup size to exactly what the source kernel declared, so
    // the backend lays out LDS / workitem IDs the same way the original
    // gfx1250 binary did.
    int MaxWg = Meta.MaxFlatWorkgroupSize > 0 ? Meta.MaxFlatWorkgroupSize : 1024;
    F->addFnAttr("amdgpu-flat-work-group-size",
                  std::to_string(MaxWg) + "," + std::to_string(MaxWg));

    // Deliberately do NOT set "amdgpu-waves-per-eu".  Pinning occupancy
    // constrains register allocation and caused spurious VGPR spills for
    // wide kernels (e.g. the Triton 128x128 matmul on gfx942), which then
    // triggered memory faults because our raised IR is register-pressure
    // heavy compared to a from-source compile.  Letting the backend choose
    // occupancy freely keeps register pressure safe.
    // TODO(gfx1250->gfx942): revisit once the raiser emits tighter IR; we may
    // want to propagate the source kernel's waves-per-eu for parity.

    // The hotswap caller still launches with the source kernel's host-side
    // kernarg buffer.  Hotswap materialises every source-visible value either
    // as a normal formal parameter, as source-ABI preloaded SGPR state seeded
    // explicitly in IR below, or as an intrinsic for architected dispatch
    // state.  Suppress backend-invented implicit kernarg slots so the emitted
    // descriptor keeps the source kernarg size instead of appending a
    // target-default hidden-arg block that the host never populated.
    F->addFnAttr("amdgpu-no-cluster-id-x");
    F->addFnAttr("amdgpu-no-cluster-id-y");
    F->addFnAttr("amdgpu-no-cluster-id-z");
    F->addFnAttr("amdgpu-no-completion-action");
    F->addFnAttr("amdgpu-no-default-queue");
    F->addFnAttr("amdgpu-no-dispatch-id");
    // Do not suppress dispatch-ptr: source hidden-arg synthesis materialises
    // values such as hidden_group_size_* and hidden_block_count_* from the
    // target dispatch packet, because the lifted HSACO intentionally does not
    // ask HIP to append source-ABI hidden args after the opaque kargs blob.
    F->addFnAttr("amdgpu-no-heap-ptr");
    F->addFnAttr("amdgpu-no-hostcall-ptr");
    F->addFnAttr("amdgpu-no-implicitarg-ptr");
    F->addFnAttr("amdgpu-no-lds-kernel-id");
    F->addFnAttr("amdgpu-no-multigrid-sync-arg");
    F->addFnAttr("amdgpu-no-queue-ptr");
    F->addFnAttr("amdgpu-no-workitem-id-x");
    F->addFnAttr("amdgpu-no-workitem-id-y");
    F->addFnAttr("amdgpu-no-workitem-id-z");
    F->addFnAttr("uniform-work-group-size", "true");
  }

  // Propagate static LDS allocation from the source kernel descriptor.
  //
  // The raiser's `ds_write_b128` / `ds_load_b128` / `ds_bpermute` emit
  // pointer-arithmetic into `addrspace(3)` DIRECTLY (via `inttoptr i64
  // to ptr addrspace(3)`), without declaring an LDS `GlobalVariable`.
  // LLVM's AMDGPU backend derives `group_segment_fixed_size` from
  // addrspace(3) GlobalVariables plus the `amdgpu-lds-size` function
  // attribute (see `AMDGPUMachineFunctionInfo` -- `LDSSizeRange.first`
  // is read from the attr), so a raised kernel that only manipulates
  // addrspace(3) via int-to-ptr conversion and never sets the attr
  // gets `group_segment_fixed_size: 0` in the emitted HSACO.  The
  // hardware then treats every LDS op as out-of-segment and returns
  // zero / drops writes.  This silently miscompiled every lifted
  // kernel with a non-trivial LDS round-trip, most visibly Triton's
  // `matmul_fp16` (mode-5 B-only-varying input returned all zeros
  // because the cross-thread LDS fragment shuffle read from an
  // uninitialised segment; see matrix-translation.md §12.4 for the
  // bisection).
  //
  // We mirror the source's `.group_segment_fixed_size` by setting the
  // per-function `amdgpu-lds-size` attribute in the source-declared
  // range.  The attribute takes "min,max" -- we pass the same value
  // for both since the source's static size is known exactly.
  if (Meta.GroupSegmentFixedSize > 0) {
    std::string SizeStr = std::to_string(Meta.GroupSegmentFixedSize);
    F->addFnAttr("amdgpu-lds-size", SizeStr + "," + SizeStr);
  }

  if (ParamIdx > 0)
    F->getArg(0)->setName("kargs");

  errs() << "transpiler: Kernel '" << KernelName
         << "' kernarg_segment_size=" << Meta.KernargSegmentSize << "\n";

  Function *FnWorkgroupIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_x);
  Function *FnWorkgroupIdY =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_y);
  Function *FnKargPtr =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_kernarg_segment_ptr);
  // Build the source-ISA user-SGPR ABI from the kernel descriptor.
  // Phase 4 seeding and handler-side ABI-sensitive decoding (e.g.
  // handle_smem's kernarg-pointer detection) both key off this layout.
  UserSgprLayout UserSgprLayout;
  std::string UserSgprFailureDetail;
  if (!UserSgprLayout::tryFromKernelMeta(Meta, Isa, SourceIsa, UserSgprLayout,
                                         UserSgprFailureDetail)) {
    Result.Failure = Meta.HasKernelDescriptor
                         ? RaiseFailure::userSgprLayoutMismatch(
                               KernelName, UserSgprFailureDetail)
                         : RaiseFailure::missingKernelDescriptor(KernelName);
    if (!UserSgprFailureDetail.empty())
      errs() << UserSgprFailureDetail << "\n";
    return Result;
  }
  if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo) &&
      Meta.hasNonDisabledClusterDims()) {
    Result.Failure = RaiseFailure::unsupportedSourceClusterDims(
        KernelName,
        Twine(".cluster_dims=[") + Twine(Meta.ClusterDims[0]) + "," +
            Twine(Meta.ClusterDims[1]) + "," + Twine(Meta.ClusterDims[2]) +
            "] requires real TTMP6 cluster workgroup state; the current "
            "HotSwap ABI model only supports disabled source clusters");
    return Result;
  }
  // ==== Phase 3: Create basic blocks ====
  // `blockStarts` is a std::set (see decode.h) so it iterates in
  // ascending source-address order, giving deterministic BB labels.
  // `offsetToBB` is a DenseMap and intentionally unordered; for the
  // thread-loop entry BB we need the lowest-address BB as InsertBefore
  // (so the entry sorts above the kernel body in IR), which we capture
  // explicitly during the create loop.
  llvm::DenseMap<uint64_t, BasicBlock *> OffsetToBb;
  BasicBlock *FirstBodyBb = nullptr;
  for (uint64_t Addr : BlockStarts) {
    BasicBlock *Bb =
        BasicBlock::Create(C, "bb_0x" + utohexstr(Addr - KernelOffset), F);
    OffsetToBb[Addr] = Bb;
    if (!FirstBodyBb)
      FirstBodyBb = Bb;
  }
  BasicBlock *EntryBb = UseThreadLoop
                            ? BasicBlock::Create(C, "entry", F, FirstBodyBb)
                            : OffsetToBb[KernelOffset];

  // ==== Phase 4: Init entry registers ====
  IRBuilder<> B(EntryBb);

  AllocaRegFile Regs;
  Regs.init(B, I32Ty, I1Ty, Isa, *Mc.RegInfo, Projection);

  // Seed kernel-entry SGPR state from the descriptor-derived user-SGPR ABI.
  //
  // Crucial invariant: never hardcode SGPR indices. Kernarg preload and
  // enable_sgpr_* toggles legally move the kernarg pointer and workgroup-id
  // SGPRs away from s[0:1]/s2/s3. Hardcoding those indices mis-seeds entry
  // state and turns real source values into undef reads on the JIT path.
  //
  // Seed the kernarg pair with ptrtoint(amdgcn_kernarg_segment_ptr) so the
  // generic GEP+load path in handle-smem.cpp materialises kernarg SMEM
  // loads as real scalar loads (the backend selects s_load_* off the
  // addrspace(4) cast). storeSGPR64 ptrtoint-splits the pointer into two
  // i32 halves; loadSGPR64 reconstructs and the SMEM handler casts back
  // to ptr addrspace(4).
  if (UserSgprLayout.KernargSegmentPtrSgpr >= 0) {
    Regs.storeSGPR64(B, UserSgprLayout.KernargSegmentPtrSgpr,
                     B.CreateCall(FnKargPtr, {}, "kernarg_ptr"));
  }
  if (UserSgprLayout.WorkgroupIdXSgpr >= 0) {
    Regs.storeSGPR32(B, UserSgprLayout.WorkgroupIdXSgpr,
                     B.CreateCall(FnWorkgroupIdX, {}, "wg_id_x"));
  }
  if (UserSgprLayout.WorkgroupIdYSgpr >= 0) {
    Regs.storeSGPR32(B, UserSgprLayout.WorkgroupIdYSgpr,
                     B.CreateCall(FnWorkgroupIdY, {}, "wg_id_y"));
  }
  SourceHiddenArgContext HiddenCtx{C, M, B, I8Ty, I32Ty, I64Ty, Meta.Args};
  auto EmitPreloadedHiddenKernargDword = [&](int ByteOffset) -> Value * {
    SourceHiddenArgValue Hidden = emitSourceHiddenDword(HiddenCtx, ByteOffset);
    if (!Hidden.Matched)
      return nullptr;
    if (!Hidden.Value)
      report_fatal_error(Twine("transpiler: preloaded hidden kernarg at byte "
                               "offset ") + Twine(ByteOffset) + ": " +
                         Hidden.FailureDetail);
    return Hidden.Value;
  };
  // Kernarg preload SGPRs carry dwords copied by hardware from the kernarg
  // segment before kernel entry. Materialize the same dwords by loading
  // through `amdgcn_kernarg_segment_ptr` so the AMDGPU backend handles the
  // ABI lowering uniformly: the GEP+load lowers back to `s_load_b32` (or a
  // hardware-preload SGPR read on gfx12+) against the kernarg segment, with
  // identical bytes to what the source kernel saw at entry.
  //
  // Hidden block counts (Triton's hidden_block_count_* ABI) still need
  // dispatch-packet synthesis since their values aren't stored in the
  // kernarg segment at all -- only `emitPreloadedHiddenKernargDword` can
  // materialize them from `amdgcn_dispatch_ptr`.
  for (size_t SgprIdx = 0; SgprIdx < UserSgprLayout.Entries.size(); ++SgprIdx) {
    const auto &Entry = UserSgprLayout.Entries[SgprIdx];
    if (Entry.SrcKind != UserSgprLayout::Source::PreloadedKernarg)
      continue;
    Value *Dw = EmitPreloadedHiddenKernargDword(Entry.KernargByteOffset);
    if (!Dw) {
      Value *SegPtr = B.CreateCall(FnKargPtr, {}, "preload_kernarg_ptr");
      Value *Gep = B.CreateInBoundsGEP(
          I8Ty, SegPtr, B.getInt64(Entry.KernargByteOffset), "preload_gep");
      Dw = B.CreateAlignedLoad(I32Ty, Gep, Align(4), "preload_dw");
    }
    Regs.storeSGPR32(B, static_cast<int>(SgprIdx), Dw);
  }
  auto SeedWorkitemX = [&](IRBuilder<> &SeedB) {
    Regs.storeVGPR32(SeedB, 0, Projection.emitWorkitemIdX(SeedB));
  };

  if (!UseThreadLoop)
    SeedWorkitemX(B);

  // On gfx12+ the hardware command processor uses TTMP registers for
  // workgroup scheduling (RDNA4+ / CDNA-next layout):
  //   ttmp7[15:0]  = workgroup_id_y  (low 16 bits)
  //   ttmp7[31:16] = workgroup_id_z  (high 16 bits; 0 when grid has no Z)
  //   ttmp8[29:25] = wave_id within workgroup (subgroup ID)
  //   ttmp9        = workgroup_id_x  (accelerated launch)
  // The packed-Y-and-Z layout in ttmp7 is from the AMDGPU backend's
  // `loadInputValue` path (see LLVM's `AMDGPULegalizerInfo.cpp` --
  // `WorkGroupIDY = ArgDescriptor::createRegister(TTMP7, 0xFFFFu)`,
  // `WorkGroupIDZ = ArgDescriptor::createRegister(TTMP7, 0xFFFF0000u)`).
  // Triton-generated gfx1250 kernels read the Y component via
  // `s_and_b32 sN, ttmp7, 0xffff` (e.g. matmul_fp16_16x16's `pid_n =
  // tl.program_id(1)` lowering), so a kernel raised without ttmp7
  // initialised always sees `workgroup_id_y == 0` -- only the
  // leftmost column of workgroups in a 2D-grid kernel writes its
  // tile, and the right-side tiles stay at whatever the destination
  // memory held at dispatch (verified empirically: matmul_fp16_16x16
  // M=32 with an all-1s input shows cols 0..15 = correct 32.0,
  // cols 16..31 = poison-fill from the host's pre-launch memset).
  // gfx11 (RDNA3) passes these via SGPRs set up by the CP instead.
  std::function<void(IRBuilder<> &)> SeedTtmp8 = [](IRBuilder<> &) {};
  if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo)) {
    // TTMP6 carries the source workgroup-cluster fields on gfx12+. This
    // HotSwap path models non-cluster source execution, so use the singleton
    // cluster encoding: per-cluster workgroup IDs and max IDs are all zero.
    B.CreateStore(B.getInt32(0), Regs.Ttmp[6]);
    B.CreateStore(B.CreateCall(FnWorkgroupIdX, {}, "ttmp9_wg_id"), Regs.Ttmp[9]);

    // ttmp7 = (workgroup_id_z << 16) | (workgroup_id_y & 0xFFFF).
    // We mask Y to 16 bits before shifting Z so a stray-high-bit Y
    // doesn't bleed into the Z field.  CAVEAT: upstream's mask is
    // conditional -- `AMDGPULegalizerInfo::loadInputValue` uses `~0u`
    // on no-Z-grid entry-function kernels (letting a consumer that
    // reads ttmp7 unmasked see the FULL 32-bit workgroup_id_y, for
    // Y up to UINT_MAX).  Our unconditional 16-bit mask clips Y on
    // no-Z grids with Y >= 65536, which is a hypothetical silent
    // miscompile.  We have not observed a lifted kernel that does
    // this in practice -- every Triton-emitted consumer I surveyed
    // reads via `s_and ttmp7, 0xffff` -- but if a Y >= 65536 no-Z
    // kernel shows up we'll need to either thread `hasWorkGroupIDZ`
    // through `meta` and emit the conditional mask here, or switch
    // to the `~0u` mask and let `s_and ttmp7, 0xffff` consumers
    // tolerate the Z bits bleeding into their read (they already do
    // per the consumer pattern definition).
    Value *WgIdY = B.CreateCall(FnWorkgroupIdY, {}, "ttmp7_wg_id_y");
    Function *FnWorkgroupIdZ =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_z);
    Value *WgIdZ = B.CreateCall(FnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
    Value *WgIdYLo = B.CreateAnd(WgIdY, B.getInt32(0xFFFF), "wg_id_y_lo16");
    Value *WgIdZHi = B.CreateShl(WgIdZ, B.getInt32(16), "wg_id_z_hi16");
    Value *Ttmp7Val = B.CreateOr(WgIdYLo, WgIdZHi, "ttmp7_val");
    B.CreateStore(Ttmp7Val, Regs.Ttmp[7]);

    SeedTtmp8 = [&](IRBuilder<> &SeedB) {
      // wave_id = workitem_id_x / wavefront_size (32 for gfx12)
      Value *TidForTtmp = Projection.emitWorkitemIdX(SeedB);
      TidForTtmp->setName("ttmp8_tid");
      Value *WaveId =
          SeedB.CreateLShr(TidForTtmp, SeedB.getInt32(5), "wave_id_in_wg");
      Value *Ttmp8Val =
          SeedB.CreateShl(WaveId, SeedB.getInt32(25), "ttmp8_val");
      SeedB.CreateStore(Ttmp8Val, Regs.Ttmp[8]);
    };
    if (!UseThreadLoop)
      SeedTtmp8(B);
  }

  auto SeedThreadLoopIterationState = [&](IRBuilder<> &SeedB) {
    for (auto *Slot : Regs.Sgpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Vgpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Agpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Ttmp)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.M0);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.FlatScr[0]);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.FlatScr[1]);

    // Mirror the entry-BB user-SGPR seeding above: the kernarg pair is
    // re-seeded with `amdgcn_kernarg_segment_ptr` so kernarg SMEM loads
    // inside the thread-loop iteration body lift through the same
    // GEP+load shape, and preloaded-kernarg SGPRs materialise their
    // dwords via the same intrinsic + GEP + i32 load. Hidden block
    // counts continue to flow through `emitPreloadedHiddenKernargDword`
    // (dispatch-packet synthesis, not in kernarg memory).
    if (UserSgprLayout.KernargSegmentPtrSgpr >= 0) {
      Regs.storeSGPR64(SeedB, UserSgprLayout.KernargSegmentPtrSgpr,
                       SeedB.CreateCall(FnKargPtr, {}, "kernarg_ptr"));
    }
    if (UserSgprLayout.WorkgroupIdXSgpr >= 0) {
      Regs.storeSGPR32(SeedB, UserSgprLayout.WorkgroupIdXSgpr,
                       SeedB.CreateCall(FnWorkgroupIdX, {}, "wg_id_x"));
    }
    if (UserSgprLayout.WorkgroupIdYSgpr >= 0) {
      Regs.storeSGPR32(SeedB, UserSgprLayout.WorkgroupIdYSgpr,
                       SeedB.CreateCall(FnWorkgroupIdY, {}, "wg_id_y"));
    }
    for (size_t SgprIdx = 0; SgprIdx < UserSgprLayout.Entries.size();
         ++SgprIdx) {
      const auto &Entry = UserSgprLayout.Entries[SgprIdx];
      if (Entry.SrcKind != UserSgprLayout::Source::PreloadedKernarg)
        continue;
      Value *Dw = EmitPreloadedHiddenKernargDword(Entry.KernargByteOffset);
      if (!Dw) {
        Value *SegPtr =
            SeedB.CreateCall(FnKargPtr, {}, "preload_kernarg_ptr");
        Value *Gep = SeedB.CreateInBoundsGEP(
            I8Ty, SegPtr, SeedB.getInt64(Entry.KernargByteOffset),
            "preload_gep");
        Dw = SeedB.CreateAlignedLoad(I32Ty, Gep, Align(4), "preload_dw");
      }
      Regs.storeSGPR32(SeedB, static_cast<int>(SgprIdx), Dw);
    }

    if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo)) {
      SeedB.CreateStore(SeedB.CreateCall(FnWorkgroupIdX, {}, "ttmp9_wg_id"),
                        Regs.Ttmp[9]);
      Value *WgIdY = SeedB.CreateCall(FnWorkgroupIdY, {}, "ttmp7_wg_id_y");
      Function *FnWorkgroupIdZ = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::amdgcn_workgroup_id_z);
      Value *WgIdZ = SeedB.CreateCall(FnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
      Value *WgIdYLo =
          SeedB.CreateAnd(WgIdY, SeedB.getInt32(0xFFFF), "wg_id_y_lo16");
      Value *WgIdZHi =
          SeedB.CreateShl(WgIdZ, SeedB.getInt32(16), "wg_id_z_hi16");
      Value *Ttmp7Val = SeedB.CreateOr(WgIdYLo, WgIdZHi, "ttmp7_val");
      SeedB.CreateStore(Ttmp7Val, Regs.Ttmp[7]);
      SeedTtmp8(SeedB);
    }

    SeedWorkitemX(SeedB);
    Regs.storeVCC(SeedB, ConstantInt::getFalse(I1Ty));
    Regs.storeSCC(SeedB, ConstantInt::getFalse(I1Ty));
    Regs.storeExec(SeedB, Projection.emitInitialExec(SeedB));
  };

  // ==== Phase 5: Raise each instruction; collect all failures in allFailures. ====

  auto *F16Ty = Type::getHalfTy(C);
  auto *F64Ty = Type::getDoubleTy(C);
  // `userSgprLayout` was built above before Phase 4 so entry SGPR seeding
  // and handler-side ABI decisions use the same descriptor-derived mapping.
  RaiseContext Ctx{C, M, B, Regs, Projection, Mc, Isa, TargetIsa, Kernargs,
                   &UserSgprLayout, F,
                   nullptr,
                   I1Ty, I8Ty, I32Ty, I64Ty, F32Ty, F16Ty, F64Ty,
                   PtrGlobalTy, OffsetToBb, KernelOffset, KernelEndOffset};
  Ctx.SetpcAnalysis = &SetpcAnalysis;
  Ctx.SourcePrivateSegmentFixedSize = Meta.PrivateSegmentFixedSize;
  Ctx.SourceComputePgmRsrc2 = Meta.ComputePgmRsrc2;
  Ctx.SourceKernelCodeProperties = Meta.KernelCodeProperties;

  // Dominance-safe SGPR wave-mask shadow storage.
  // One EXEC-width mask + one scalar-valid bit per SGPR base index.
  // Consumers can combine `(valid ? shadow : fallback)` across BBs without
  // carrying non-dominating SSA values in `lastSgprWaveMaskI1`.
  Ctx.SgprWaveMaskExecShadow.reserve(Regs.Sgpr.size());
  Ctx.SgprWaveMaskValidShadow.reserve(Regs.Sgpr.size());
  for (unsigned I = 0; I < Regs.Sgpr.size(); ++I) {
    auto *MaskA = B.CreateAlloca(Regs.ExecTy, nullptr,
                                 "sgpr_mask_shadow_" + std::to_string(I));
    auto *ValidA = B.CreateAlloca(I1Ty, nullptr,
                                  "sgpr_mask_valid_" + std::to_string(I));
    B.CreateStore(ConstantInt::get(Regs.ExecTy, 0), MaskA);
    B.CreateStore(B.getFalse(), ValidA);
    Ctx.SgprWaveMaskExecShadow.push_back(MaskA);
    Ctx.SgprWaveMaskValidShadow.push_back(ValidA);
  }

  // Wire the reg-file's EXEC-write invalidation hook to ctx's lane_active
  // memo. This catches every EXEC mutation -- ctx.storeExec, the various
  // ctx.writeReg*(EXEC, …) wrappers, *and* the handful of handlers that
  // still call ctx.Regs.storeExec / ctx.Regs.writeRegExecWidth directly
  // (SAVEEXEC family in handle_sop1, V_CMPX in handle_valu). Without
  // this hook those direct paths would leave the memo pointing at a
  // pre-write `lane_active`, silently mispredicating subsequent
  // emitUnderExec diamonds.
  Regs.OnExecWritten = [&Ctx] { Ctx.resetLaneActiveCache(); };

  // Wire the reg-file's per-SGPR write invalidation hook to ctx's
  // V_CMP -> V_CNDMASK per-lane-i1 shadow map
  // (`lastSgprWaveMaskI1`). Fires on every `storeSGPR32 / storeSGPR64`
  // and therefore on every path that mutates an SGPR -- including
  // handlers that bypass `writeReg32 / writeReg64` to call the
  // low-level stores directly (handle_smem's multi-dword load
  // splitting, handle_valu's SCC-flag SGPR writes, etc.). The V_CMP
  // wave-mask write path also fires this hook; the V_CMP handler
  // immediately re-populates the shadow with the per-lane `i1`
  // afterwards via `ctx.recordSgprWaveMaskI1`. See hotswap/docs/sgpr-
  // wave-mask-translation.md section 3.1 for the full contract.
  Regs.OnSgprWritten = [&Ctx](int Idx) { Ctx.invalidateSgprWaveMaskI1(Idx); };

  if (UseThreadLoop) {
    auto *IterA = B.CreateAlloca(I32Ty, nullptr, "tl_iter_alloca");
    B.CreateStore(B.getInt32(0), IterA);
    static_cast<ThreadLoopProjection *>(ProjectionPtr.get())
        ->setIterationAlloca(IterA);

    BasicBlock *CondBb = BasicBlock::Create(C, "tl_cond", F);
    BasicBlock *LatchBb = BasicBlock::Create(C, "tl_latch", F);
    BasicBlock *DoneBb = BasicBlock::Create(C, "tl_done", F);
    Ctx.ThreadLoopLatch = LatchBb;

    B.CreateBr(CondBb);
    B.SetInsertPoint(CondBb);

    Value *Iter = B.CreateLoad(I32Ty, IterA, "tl_iter_val");
    Value *IterOk = B.CreateICmpULT(
        Iter, B.getInt32(TargetIsa.WaveSize / Isa.WaveSize), "tl_iter_ok");
    Value *Lane = Projection.emitLaneIdx(B);
    Value *LaneOk =
        B.CreateICmpULT(Lane, B.getInt32(Isa.WaveSize), "tl_lane_ok");
    Value *EnterBody = B.CreateAnd(IterOk, LaneOk, "tl_enter_body");

    SeedThreadLoopIterationState(B);
    for (auto *ValidA : Ctx.SgprWaveMaskValidShadow)
      B.CreateStore(B.getFalse(), ValidA);

    B.CreateCondBr(EnterBody, OffsetToBb[KernelOffset], LatchBb);

    B.SetInsertPoint(LatchBb);
    Value *OldIter = B.CreateLoad(I32Ty, IterA, "tl_iter_old");
    Value *NextIter = B.CreateAdd(OldIter, B.getInt32(1), "tl_iter_next");
    B.CreateStore(NextIter, IterA);
    Value *More = B.CreateICmpULT(
        NextIter, B.getInt32(TargetIsa.WaveSize / Isa.WaveSize), "tl_more");
    B.CreateCondBr(More, CondBb, DoneBb);

    B.SetInsertPoint(DoneBb);
    B.CreateRetVoid();
  }

  int RaisedCount = 0;

  for (size_t InstIdx = 0; InstIdx < Insts.size(); ++InstIdx) {
    const DecodedInst &Di = Insts[InstIdx];

    // If a terminator ended the recovered CFG path and the next decoded
    // instruction is not a known block leader, that instruction is unreachable
    // fallthrough bytes (often code after an unconditional branch). Do not emit
    // it into the already-terminated LLVM block.
    auto BbIt = OffsetToBb.find(Di.Offset);
    if (B.GetInsertBlock()->hasTerminator() && BbIt == OffsetToBb.end())
      continue;

    // Source-BB boundary handling uses `B.GetInsertBlock()` rather than a
    // tracked `currentBB` so that intra-handler CFG splits (emitUnderExec
    // diamonds under SPE) propagate correctly: fall-through must leave
    // from whatever block the builder is currently at -- which is the
    // `spe_skip` tail when the last emission was wrapped -- not from the
    // block that started the source instruction.
    if (BbIt != OffsetToBb.end() && BbIt->second != B.GetInsertBlock()) {
      BasicBlock *InsertBb = B.GetInsertBlock();
      if (!InsertBb->hasTerminator())
        B.CreateBr(BbIt->second);
      B.SetInsertPoint(BbIt->second);
      // LLVM's AMDGPULowerVGPREncoding pass resets VGPR MSB mode at every
      // basic-block boundary (both before terminators and at BB fall-through
      // exits).  Mirror that behaviour so we do not inherit stale MSB state
      // from a previous linear instruction that does not control-flow into
      // this BB.
      Ctx.VgprMsBs = 0;
      // Drop the V_CMP -> V_CNDMASK per-lane-i1 shadow at every BB
      // transition. The cached `i1` SSA values dominate only the BB
      // they were emitted in; carrying them into a successor would
      // read an SSA value out of its dominance scope. A future
      // reaching-definitions pass on the raised IR could upgrade this
      // to a proper per-BB merge (see sgpr-wave-mask-translation.md
      // section 7 evolution path).
      Ctx.clearSgprWaveMaskShadow();
    }

    Ctx.computeVGPRAdjust(Di);
    // Invalidate the SPE lane_active memoisation at every instruction
    // boundary. Any instruction is a potential EXEC writer (either through
    // our modeled CanonicalOp allow-list, or through a path we haven't yet
    // covered), and emitLaneActiveBit is load-bearing for per-lane
    // predication correctness: reusing a stale lane_active from before an
    // EXEC write would silently mispredicate side effects. See
    // RaiseContext::resetLaneActiveCache in raise-context.h for the full
    // invalidation contract.
    Ctx.resetLaneActiveCache();
    OpResolver Op{Ctx, Di};

    // Dispatch to the format-specific handler by querying TSFlags (and
    // `AMDGPU::isVOPD` for the one encoding without a dedicated flag bit)
    // directly, rather than going through a hand-rolled FormatKind enum.
    // Check precedence mirrors LLVM's decoder:
    //   * VOPD first -- it has no TSFlags bit; detect by named-operand id.
    //   * IsMAI before VOP3 -- MFMA is a VOP3 subclass with its own handler.
    //   * DPP / SDWA / VOPC / VOP3P / VOP3 / VOP2 / VOP1 all route to
    //     handleVALU, so they're collapsed into one mask test; ordering
    //     within the VOP family is therefore irrelevant here.
    //   * Scalar / memory family bits are mutually exclusive.
    // `default: break;` semantics are preserved: anything without a matching
    // bit falls through with `hr.Handled == false` and hits the unsupported-
    // instruction error path below.
    const uint64_t KValu =
        SIInstrFlags::DPP | SIInstrFlags::SDWA | SIInstrFlags::VOP1 |
        SIInstrFlags::VOP2 | SIInstrFlags::VOP3 | SIInstrFlags::VOPC |
        SIInstrFlags::VOP3P;
    const uint64_t Flags = Di.TsFlags;
    const unsigned Opc = Di.Inst.getOpcode();
    HandlerResult Hr;
    if (AMDGPU::isVOPD(Opc))
      Hr = handleVOPD(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::IsMAI)
      Hr = handleMFMA(Ctx, Di, Op);
    else if (Flags & KValu)
      Hr = handleVALU(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::SOPP)
      Hr = handleSOPP(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::SOPC)
      Hr = handleSOPC(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::SOP1)
      Hr = handleSOP1(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::SOP2)
      Hr = handleSOP2(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::SOPK)
      Hr = handleSOPK(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::SMRD)
      Hr = handleSMEM(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::FLAT)
      Hr = handleFLAT(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::MUBUF)
      Hr = handleMUBUF(Ctx, Di, Op);
    else if (Flags & SIInstrFlags::DS)
      Hr = handleDS(Ctx, Di, Op);
    // VIMAGE TENSOR pseudo-instructions (`tensor_load_to_lds_d{2,4}`,
    // `tensor_store_from_lds_d{2,4}`, MIMGInstructions.td:2049-2113).
    // The pseudo extends `InstSI` directly and only sets `let VALU =
    // 1` and `let TENSOR_CNT = 1` (NOT `let VIMAGE = 1`), so the
    // `SIInstrFlags::VIMAGE` bit stays 0 on these. Dispatch on
    // `TENSOR_CNT` instead -- the only other carrier of that bit is
    // `s_wait_tensorcnt` (SOPP), which is already claimed by the
    // SOPP arm above and never reaches this fallthrough. Routed
    // late because TENSOR ops are exclusive to the gfx1250
    // (`isGFX125xOnly`) generation and the handler's only contract
    // today is a cross-target loud refusal; the same gating applies
    // when the same-target intrinsic-emit path lands.
    else if (Flags & SIInstrFlags::TENSOR_CNT)
      Hr = handleVIMAGE(Ctx, Di, Op);

    // Operand-read paths (`readOp32` / `readOp64`) cannot bail mid-
    // handler, so they record any unsupported-register failures into
    // `ctx.pendingFailure`. Promote that to the structured failure
    // *before* the `hr.Handled` check -- a handler that "succeeded"
    // by returning undef from a read is still an unraised kernel.
    if (Ctx.PendingFailure.hasFailed()) {
      if (!Result.Failure.hasFailed())
        Result.Failure = Ctx.PendingFailure;
      Result.AllFailures.push_back(std::move(Ctx.PendingFailure));
      Ctx.PendingFailure = RaiseFailure{};
      continue;
    }

    if (Hr.Handled) {
      if (Di.DefsScc && !Hr.SccHandled && Hr.SccResult) {
        Value *Zero = Constant::getNullValue(Hr.SccResult->getType());
        Ctx.Regs.storeSCC(Ctx.B, Ctx.B.CreateICmpNE(Hr.SccResult, Zero));
      }
      if (Di.DefsExec)
        Result.HasDivergentExec = true;
      // Pattern B call-site post-processing: if this s_add_co_ci_u32
      // is the high-half terminator of a getpc+add chain that feeds
      // a Pattern B `s_set_pc_i64` enumerated-dispatch cascade (i.e.
      // some downstream s_set_pc_i64 reads the same ret-pair this
      // chain populated), overwrite the ret-pair SGPR with the plain
      // i64 marker `resolvedReturnAddr` -- i.e. the source-MC byte
      // offset of the BB this chain meant to return to. The
      // downstream cascade compares against the same offsets via
      // `icmp eq i64 %marker, <offset_k>` for each enumerated
      // target; when this predecessor's marker matches one of the
      // enumerated offsets, mem2reg + SCCP + InstCombine fold the
      // compare to `i1 true` across the phi join and SimplifyCFG
      // collapses the cmp+br cascade into a direct
      // `br label %BB_<offset>`. The SOP2 handler has already done
      // its (binary-PC-producing) arithmetic above; this commit
      // happens *after* and clobbers that result on purpose -- that
      // value was an opaque runtime PC we never want to see
      // downstream.
      //
      // An earlier revision of this hook wrote
      // `ptrtoint(blockaddress(@kernel, %BB_returnAddr)) to i64`
      // here so the cascade could compare against a `blockaddress`
      // constant. That form survived mem2reg + SCCP unfolded in
      // irreducible tensilelite-shaped CFGs (the `storeSGPR64`
      // hi/lo split prevented the cross-phi fold), leaving a
      // `BlockAddress` SDNode alive into AMDGPU ISel, which has no
      // codegen pattern for it and aborts llc with
      //   `Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
      // Using a plain integer marker keeps `BlockAddress` solely
      // as a direct-branch `label` operand (which DOES have a
      // codegen pattern), sidestepping the ISel crash entirely.
      // See setpc-analysis.h + canonical-op.h's S_SET_PC_I64 doc +
      // `emitEnumeratedDispatch` in handle-sop1.cpp.
      if (Di.CanonOp == CanonicalOp::S_ADDC_U32) {
        auto It = SetpcAnalysis.ChainTerminators.find(Di.Offset);
        if (It != SetpcAnalysis.ChainTerminators.end()) {
          // Force the BB to exist so the downstream cascade's
          // direct branch has a destination; we don't use the
          // pointer here.
          (void)Ctx.lookupBB(It->second.ResolvedReturnAddr);
          Value *RetMarker =
              ConstantInt::get(Ctx.I64Ty, It->second.ResolvedReturnAddr);
          Ctx.Regs.storeSGPR64(Ctx.B,
                                static_cast<int>(It->second.RetPairLowReg),
                                RetMarker);
        }
      }

      RaisedCount++;
      continue;
    }

    // The handler either recognised the instruction but refused the
    // specific shape (Hr.Failure.Reason != None), or no handler claimed
    // it at all -- promote to `UnsupportedOpcode` and bucket by format.
    if (Hr.Failure.hasFailed()) {
      if (!Result.Failure.hasFailed())
        Result.Failure = Hr.Failure;
      Result.AllFailures.push_back(std::move(Hr.Failure));
    } else {
      RaiseFailure f = RaiseFailure::unsupportedOpcode(
          Di, formatName(Di.TsFlags, Di.Inst.getOpcode()));
      errs() << "transpiler: Unsupported instruction: " << Di.Mnemonic
             << " (raw: " << Di.RawMnemonic << ")"
             << " [format=" << f.Format << "]"
             << " at offset 0x" << format_hex(Di.Offset, 1) << "\n";
      if (!Result.Failure.hasFailed())
        Result.Failure = f;
      Result.AllFailures.push_back(std::move(f));
    }
  }

  // Ensure all BBs have terminators. An empty kernel (no decoded
  // instructions) leaves only the entry block with no terminator;
  // emit `ret void` so the lifted module behaves as a no-op kernel
  // rather than aborting on unreachable. Other unterminated blocks
  // (typically dead-fallthrough bytes after a recovered branch) keep
  // their defensive `unreachable`.
  for (auto &BB : *F) {
    if (!BB.hasTerminator()) {
      B.SetInsertPoint(&BB);
      if (&BB == &F->getEntryBlock() && F->size() == 1)
        B.CreateRetVoid();
      else
        B.CreateUnreachable();
    }
  }

  Result.LiftedCount = RaisedCount;

  // If any instructions failed to raise, skip Phases 6-7.
  if (!Result.AllFailures.empty()) {
    return Result;
  }

  // ==== Phase 6: Promote allocas to SSA ====
  {
    DominatorTree DT(*F);
    AssumptionCache AC(*F);
    SmallVector<AllocaInst *, 512> Allocas;
    Regs.collectAllocas(Allocas);
    Ctx.collectSgprWaveMaskShadowAllocas(Allocas);
    PromoteMemToReg(Allocas, DT, &AC);
  }

  // ==== Link and inline OCML device-library helpers ====
  // OCML-backed VALU lifts emit declared helper calls into COMGR's embedded
  // device libraries. Resolve and inline those calls before the cross-lane
  // rewrite so the DPP/readlane/writelane use-chain classifier sees the actual
  // arithmetic IR, not an unresolved ordinary call that must conservatively be
  // treated as an SGPR-forced/unknown consumer.
  // Linked OCML math bodies are pure arithmetic for this lowering policy; they
  // should not introduce workitem-id predicate-chain structure before the C5
  // classifier below.
  if (moduleUsesOCMLRuntime(M)) {
    StringRef OCMLTargetCpu = TargetCpu.empty() ? SourceCpu : TargetCpu;
    std::string OCMLLinkErr;
    if (!linkOCMLRuntime(M, OCMLTargetCpu, TargetIsa.WaveSize, OCMLLinkErr)) {
      errs() << "transpiler: OCML device-library link failed for kernel '"
             << KernelName << "'\n";
      Result.Failure =
          RaiseFailure::deviceLibraryLinkFailed(KernelName, OCMLLinkErr);
      return Result;
    }
  }

  // (Former Phase 6.035 "permlane16-swap-selfpreserve" and Phase
  // 6.04 "permlane16-xor3-partner" rewrites were deleted after
  // the asymmetric `v_permlane16_swap_b32` lift landed -- see
  // `handle-valu-cross-lane.cpp::emitPermLaneSwapEmulation` and
  // matrix-translation.md §12.4.7.  Both passes were transitional
  // bridges that compensated for the symmetric lift's
  // over-swap of the asymmetric-semantic's "unchanged" halves;
  // with the lift corrected, their fingerprints either no
  // longer match (xor3-partner) or actively corrupt the new
  // select shape (selfpreserve).)

  // ==== Phase 6.5: Cross-widen writelane/readlane rewrite ====
  //
  // Opt-in symmetric rewrite of `v_writelane_b32` / `v_readlane_b32`
  // sites under cross-widening. Disabled by default; the caller
  // (raise_cli's `--enable-writelane-rewrite`, PipelineConfig's
  // `enableWritelaneRewrite`) must ask for it explicitly. See
  // `rewrite_cross_lane_divergent.{hpp,cpp}` and
  // wave-size-translation.md §5.6.3 for the principled derivation,
  // and hotswap/docs/learnings.md for the asymmetric-rewrite bug
  // that motivated the symmetry-plus-use-chain design.
  //
  // Runs AFTER `PromoteMemToReg` by construction: the rewrite pass's
  // forward use-chain classifier needs post-mem2reg SSA so a
  // scratch-addrspace round-trip (load / store through an alloca) does
  // not obscure the fact that a writelane / readlane result eventually
  // reaches an SGPR-constrained consumer. No behavioural change on
  // same-wave / narrowing directions -- the rewrite pass short-
  // circuits internally on `targetWaveSize <= sourceWaveSize`.
  //
  // Refusal path. If any writelane / readlane site's forward use chain
  // reaches an SGPR-forced consumer that the classifier cannot prove
  // safe (`s_buffer_load` rsrc, `s_sendmsg` message, `readfirstlane`,
  // addrspace(4) load, inline asm with `"s"` constraint, or any
  // unaudited intrinsic / instruction), the rewrite pass performs
  // zero rewrites and populates `report.sgprForcedDetail`. The raiser
  // surfaces that detail as a `crossWaveRewriteOracleDisagreement`
  // refusal -- principled per the no-silent-miscompile contract:
  // rewriting the ds_bpermute output into an SGPR-forced consumer
  // would re-introduce `v_readfirstlane_b32` at the SGPR boundary and
  // recreate the source-wave collapse the rewrite exists to avoid.
  if (EnableWritelaneRewrite) {
    // `tm.get()` threaded through so `rewriteCrossLaneDivergent` can
    // build a `UniformityAnalysis` against the compilation target
    // for the §5.6.3 "UA-backed readfirstlane allow-gate" classifier
    // refinement. See the rewrite's header comment for the contract
    // (nullable -- null disables the gate and falls back to the
    // conservative pre-UA refusal behaviour).
    CrossLaneDivergentRewriteReport RewriteReport = rewriteCrossLaneDivergent(
        *F, Isa.WaveSize, TargetIsa.WaveSize, Tm.get());

    if (RewriteReport.refusedSgprForced()) {
      ThreadLoopDecisionResult TlDecision = decideThreadLoopFallback(
          Isa.WaveSize, TargetIsa.WaveSize, /*sgprForcedRefusal=*/true,
          RewriteReport.SgprForcedThreadLoopEligible);
      if (!ForceThreadLoopProjection &&
          TlDecision.Decision == ThreadLoopDecision::EligibleAndGateOn) {
        std::string ThreadLoopUnsupportedDetail;
        if (threadLoopUnsupportedWorkgroupMemoryOrBarrier(
                Insts, ThreadLoopUnsupportedDetail)) {
          errs() << "transpiler: thread-loop fallback not eligible for kernel '"
                 << KernelName << "': " << ThreadLoopUnsupportedDetail
                 << "\n";
          RaiseFailure F = RaiseFailure::crossWaveRewriteOracleDisagreement(
              KernelName, ThreadLoopUnsupportedDetail);
          errs() << "transpiler: post-raise abort: " << F.Format << " on '"
                 << F.Mnemonic << "' -- " << F.Detail << "\n";
          Result.Failure = std::move(F);
          return Result;
        }
        errs() << "transpiler: post-raise fallback: retrying kernel '"
               << KernelName
               << "' under ThreadLoopProjection after SGPR-forced cross-lane "
                  "rewrite refusal (analysis-triggered, no user opt-in)\n";
        errs() << "transpiler: thread-loop fallback trigger: "
               << RewriteReport.SgprForcedDetail << "\n";
        return raiseToIRImpl(TextBytes, SourceIsa, KernelName, Meta,
                             KernelOffset, KernelSize, CompilationTargetIsa,
                             /*enableWritelaneRewrite=*/false,
                             /*enableWaveNative=*/false,
                             /*forceThreadLoopProjection=*/true,
                             /*suppressC5ForThreadLoopRoute=*/true);
      }
      if (!ForceThreadLoopProjection &&
          TlDecision.Decision == ThreadLoopDecision::EligibleButGateOff) {
        errs() << "transpiler: thread-loop fallback candidate for kernel '"
               << KernelName << "' not activated: " << TlDecision.Reason
               << ". Keeping principled loud refusal.\n";
      }
      if (!ForceThreadLoopProjection &&
          TlDecision.Decision == ThreadLoopDecision::Ineligible) {
        errs() << "transpiler: thread-loop fallback not eligible for kernel '"
               << KernelName << "': " << TlDecision.Reason
               << ". Keeping principled loud refusal.\n";
      }
      RaiseFailure F = RaiseFailure::crossWaveRewriteOracleDisagreement(
          KernelName, RewriteReport.SgprForcedDetail);
      errs() << "transpiler: post-raise abort: " << F.Format << " on '"
             << F.Mnemonic << "' \u2014 " << F.Detail << "\n";
      Result.Failure = std::move(F);
      return Result;
    }

    // Unsupported `dpp_ctrl` on an i32 update.dpp site -- the rewrite
    // family covers quad_perm / row_shl / row_shr / row_xmask today
    // (all stay within a single 16-lane row).  Any ctrl outside that
    // set is either wave-size-dependent (wave_* shifts / rotations)
    // or hasn't been audited yet (row_mirror / row_half_mirror /
    // row_share).  Refusing loudly surfaces the demand so the next
    // extension has a concrete test pointer.  See
    // `buildDppLaneMap` in rewrite-cross-lane-divergent.cpp for
    // the per-ctrl widening protocol.
    if (RewriteReport.refusedUnsupportedDpp()) {
      RaiseFailure F = RaiseFailure::crossWaveRewriteOracleDisagreement(
          KernelName, RewriteReport.UnsupportedDppDetail);
      errs() << "transpiler: post-raise abort: " << F.Format << " on '"
             << F.Mnemonic << "' \u2014 " << F.Detail << "\n";
      Result.Failure = std::move(F);
      return Result;
    }

    // Second-order invariant: the syntactic Phase 1.4.5 classifier
    // matched `WaveIdLiftScalarized` iff the decoded instruction
    // stream contains at least one `v_writelane_b32` /
    // `v_readlane_b32`. Under the symmetry rule every such intrinsic
    // is rewritten (or the whole function refuses above), so a non-
    // zero classifier count MUST coincide with a non-zero count of
    // writelane + readlane rewrites specifically. Checking that
    // specific sum (not the grand total including `dppRewritten`)
    // matters: a kernel that emits DPP sites alongside missing
    // writelane / readlane would otherwise silently satisfy the
    // invariant via the DPP count, masking the handler-emission
    // regression this gate exists to catch.
    if (ClassifierWaveIdLiftScalarizedSites > 0 &&
        (RewriteReport.WritelaneRewritten +
         RewriteReport.ReadlaneRewritten) == 0) {
      std::string Msg;
      raw_string_ostream Os(Msg);
      Os << "classifier matched WaveIdLiftScalarized on "
         << ClassifierWaveIdLiftScalarizedSites
         << " site(s) but rewriteCrossLaneDivergent rewrote 0 \u2014 the "
            "raised IR is missing the writelane/readlane intrinsic(s) "
            "that the decoded instruction stream contained. This is a "
            "handler-emission regression, not a classifier/rewrite "
            "disagreement. Refusing rather than risk a silent "
            "miscompile (see wave-size-translation.md \u00a75.6.3).";
      RaiseFailure F = RaiseFailure::crossWaveRewriteOracleDisagreement(
          KernelName, Os.str());
      errs() << "transpiler: post-raise abort: " << F.Format << " on '"
             << F.Mnemonic << "' \u2014 " << F.Detail << "\n";
      Result.Failure = std::move(F);
      return Result;
    }
  }

  // ==== Phase 6.6: Cross-widen predicate-chain classifier (C5) ====
  //
  // Post-mem2reg classifier for the Class-5 predicate-chain class
  // documented in hotswap/docs/modrep-predicate-chain.md §5 (narrow-O1).
  // Walks every `@llvm.amdgcn.workitem.id.x()` call in the function and
  // refuses the lift if any call's forward use chain reaches an `icmp`
  // against a compile-time constant K in `(0, W_s - 1]` without being
  // AND-masked by `(W_s - 1)` first -- i.e. a lane-position-scoped
  // predicate (`tid < 2^s`, `tid < W_s/2`, quad-level masks) that would
  // evaluate differently on target replica-1 lanes than source wave 0
  // under modulo-replication despite sharing the source EXEC bit.
  //
  // Intentionally narrow: Phase-2 IR inspection (modrep-predicate-chain.md
  // §5 O1) established that the broader "any unmasked tid -> icmp ->
  // side-effect refuses" rule would also refuse baselines
  // `vecadd_f16` / `rope_fp32` / `canary_dpp_compound_add_fp32` (their
  // IR has structurally identical shapes but with a dynamic kernarg as
  // the icmp constant, not a compile-time K). The compile-time-K-only
  // rule catches `canary_bpermute_scan_fp32`'s Kogge-Stone scan-stage
  // predicates (K ∈ {1, 3, 7, 15}) while leaving the baselines green.
  //
  // Runs AFTER Phase 6 `PromoteMemToReg` so scratch-addrspace round-trips
  // are gone and the forward use-chain classifier operates on clean SSA.
  // Runs AFTER the Phase 6.5 writelane/readlane rewrite so the chain sees
  // the post-rewrite shapes (relevant when a future iteration widens the
  // classifier to audit additional users). Direction gate inside
  // `classifyPredicateChain` short-circuits when
  // `targetWaveSize <= sourceWaveSize`.
  //
  // No companion rewrite today. The design doc's §5 O2 "tid AND (W_s-1)"
  // rewrite is deferred (§6.2 documents the semantic-incorrectness of
  // the norm-family failing recipes and are a no-op for sub-case-2
  // scan-shaped recipes). If a future design iteration adds a principled
  // rewrite, pair it with a `RewriteId` alongside
  // `ObstructionKind::WorkitemIdPredicateChain`.
  {
    // Pass the projection actually selected for this kernel, not the
    // user-facing enable flag. Phantom-lane kernels route to MODREP above;
    // the classifier then decides whether that MODREP instance can have an
    // active replica lane before turning an observed C5 site into a refusal.
    PredicateChainProjection PredProjection =
        UseThreadLoop ? PredicateChainProjection::ThreadLoop
                      : (UseWaveNative
                             ? PredicateChainProjection::WaveNative
                             : PredicateChainProjection::ModuloReplication);
    PredicateChainClassifierReport PredReport =
        classifyPredicateChain(*F, Isa.WaveSize, TargetIsa.WaveSize,
                                PredProjection,
                                /*maxFlatWorkgroupSize=*/
                                Meta.MaxFlatWorkgroupSize > 0
                                    ? static_cast<unsigned>(
                                          Meta.MaxFlatWorkgroupSize)
                                    : 0u,
                                UseThreadLoop &&
                                    SuppressC5ForThreadLoopRoute);

    if (!PredReport.Refused && !PredReport.ObservedSites.empty()) {
      Result.C5SuppressedCount +=
          static_cast<int>(PredReport.ObservedSites.size());
      if (Result.C5SuppressionReason.empty())
        Result.C5SuppressionReason = PredReport.SuppressionReason;
      const char *ProjectionName =
          PredProjection == PredicateChainProjection::ThreadLoop
              ? "ThreadLoopProjection"
              : (PredProjection == PredicateChainProjection::WaveNative
                     ? "WaveNativeProjection"
                     : "ModuloReplicationProjection");
      LLVM_DEBUG({
        dbgs() << "c5-predicate-chain: observed "
               << PredReport.ObservedSites.size()
               << " C5-shape site(s) in '" << KernelName << "' under "
               << ProjectionName
               << " (refusal "
                  "suppressed per c5-predicate-chain-classifier.h "
                  "projection contract):\n";
        for (llvm::StringRef Site : PredReport.ObservedSites)
          dbgs() << "  - " << Site << "\n";
      });
    }

    if (PredReport.Refused) {
      auto HasMatrixOp = [&]() {
        const auto First =
            static_cast<uint16_t>(CanonicalOp::V_MFMA_F32_16x16x128_F8F6F4);
        const auto Last =
            static_cast<uint16_t>(CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4);
        for (const DecodedInst &Inst : Insts) {
          const auto Op = static_cast<uint16_t>(Inst.CanonOp);
          if (Op >= First && Op <= Last)
            return true;
        }
        return false;
      };
      constexpr bool kEnableThreadLoopC5Retry = false;
      const bool CanRetryThreadLoop =
          kEnableThreadLoopC5Retry &&
          PredReport.WaveNativeEqualityRefusal && !ForceThreadLoopProjection &&
          TargetIsa.WaveSize > Isa.WaveSize &&
          (TargetIsa.WaveSize % Isa.WaveSize) == 0 && !HasMatrixOp();
      if (CanRetryThreadLoop) {
        errs() << "transpiler: post-raise fallback: retrying kernel '"
               << KernelName
               << "' under ThreadLoopProjection after WaveNative C5 equality "
                  "refusal (analysis-triggered, no user opt-in)\n";
        errs() << "transpiler: thread-loop fallback trigger: "
               << PredReport.RefusalDetail << "\n";
        return raiseToIRImpl(TextBytes, SourceIsa, KernelName, Meta,
                             KernelOffset, KernelSize, CompilationTargetIsa,
                             /*enableWritelaneRewrite=*/false,
                             /*enableWaveNative=*/false,
                             /*forceThreadLoopProjection=*/true,
                             /*suppressC5ForThreadLoopRoute=*/true);
      }
      RaiseFailure F = RaiseFailure::crossWavePredicateChain(
          KernelName, PredReport.RefusalDetail);
      errs() << "transpiler: pre-translation abort: " << F.Format << " on '"
             << F.Mnemonic << "' \u2014 " << F.Detail << "\n";
      errs() << "  outcome: (c) refuse \u2014 "
                "WorkitemIdPredicateChain (\u00a73 Class 5"
             << (PredReport.WaveNativePhantomRefusal
                     ? " phantom-lane sub-case"
                     : "")
             << ")\n";
      Result.Failure = std::move(F);
      return Result;
    }
  }

  // ==== Phase 6.7: Link TDM emulation runtime ====
  // The cross-target VIMAGE handler emits calls to
  // `hotswap_tdm_load_to_lds` / `hotswap_tdm_store_from_lds` (declared,
  // no body) when the compilation target lacks the gfx1250 TENSORcnt
  // unit. Link the embedded HIP-authored runtime bitcode in here so
  // `verifyModule` sees a self-contained module and `llc` resolves the
  // calls at codegen time. No-op when the handler did not emit any
  // helper calls.
  if (moduleUsesTDMRuntime(M)) {
    if (!linkTDMRuntime(M, CompilationTargetIsa)) {
      errs() << "transpiler: TDM runtime link failed for kernel '" << KernelName << "'\n";
      Result.Failure = RaiseFailure::irVerificationFailed("TDM runtime bitcode link failed");
      return Result;
    }
  }

  // ==== Phase 7: Verify IR ====
  std::string VerifyErr;
  raw_string_ostream VerifyOs(VerifyErr);
  if (verifyModule(M, &VerifyOs)) {
    errs() << "transpiler: IR verification failed:\n" << VerifyErr << "\n";
    Result.Failure = RaiseFailure::irVerificationFailed(VerifyErr);
    return Result;
  }

  {
    raw_string_ostream IrOs(Result.IrText);
    M.print(IrOs, nullptr);
  }

  Result.UsesScratchPrivateSegment = Ctx.UsesScratchPrivateSegment;
  Result.SourcePrivateSegmentFixedSize = Ctx.SourcePrivateSegmentFixedSize;
  Result.Success = true;
  return Result;
}

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> TextBytes,
                      llvm::StringRef SourceIsa,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta,
                      llvm::StringRef CompilationTargetIsa,
                      bool EnableWritelaneRewrite,
                      bool EnableWaveNative) {
  return raiseToIR(TextBytes, SourceIsa, KernelName, Meta,
                   /*KernelOffset=*/0,
                   /*KernelSize=*/0, CompilationTargetIsa,
                   EnableWritelaneRewrite, EnableWaveNative);
}

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> TextBytes,
                      llvm::StringRef SourceIsa,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta,
                      uint64_t KernelOffset,
                      uint64_t KernelSize,
                      llvm::StringRef CompilationTargetIsa,
                      bool EnableWritelaneRewrite,
                      bool EnableWaveNative) {
  return raiseToIRImpl(TextBytes, SourceIsa, KernelName, Meta, KernelOffset,
                       KernelSize, CompilationTargetIsa, EnableWritelaneRewrite,
                       EnableWaveNative,
                       /*forceThreadLoopProjection=*/false,
                       /*suppressC5ForThreadLoopRoute=*/false);
}

} // namespace COMGR::hotswap
