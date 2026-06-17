//===- handle-sop1.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"
#include "canonical-op-attrs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Lower an analysis-enumerated indirect dispatch (the runtime i64
// value `targetInt` matches one of `targets` by setpc-analysis
// construction) into a cascade of cmp+br terminators rooted at the
// IRBuilder's current insertion block:
//
//   currBB:                                        ; B's current insert pt
//     %cmp_0 = icmp eq i64 %targetInt, <target_offset_0>
//     br i1 %cmp_0, label %bb_T0, label %dispatch_<off>_1
//   dispatch_<off>_1:
//     %cmp_1 = icmp eq i64 %targetInt, <target_offset_1>
//     br i1 %cmp_1, label %bb_T1, label %dispatch_<off>_2
//   ...
//   dispatch_<off>_{N-1}:
//     %cmp_{N-1} = icmp eq i64 %targetInt, <target_offset_{N-1}>
//     br i1 %cmp_{N-1}, label %bb_T{N-1},
//                       label %dispatch_<off>_unreachable
//   dispatch_<off>_unreachable:
//     unreachable
//
// On return the builder is positioned at the END of the unreachable
// block (after its terminator). Callers in this file return from the
// enclosing handler immediately afterwards, so no further code is
// emitted.
//
// Why a cascade and not `indirectbr` / `switch`:
//   LLVM's `FixIrreducible` pass (Transforms/Utils/FixIrreducible.cpp,
//   relied on by AMDGPU's structurizer) only handles `UncondBrInst`,
//   `CondBrInst` and `CallBrInst` as predecessors of an irreducible
//   cycle header -- it `llvm_unreachable`s for any other terminator.
//   Tensilelite-shaped lifted CFGs (kernels using `s_swappc_b64` for
//   activation-function dispatch) place the dispatch block inside an
//   irreducible cycle, so an `indirectbr` (or `switch`) terminator
//   there crashes llc with "unsupported block terminator". A cascade
//   of `br` is FixIrreducible-compatible.
//
// Why we compare against an integer marker (target offset) rather
// than a `blockaddress` pointer:
//   The raiser's chain-terminator hook stores a per-predecessor marker
//   into the ret-pair SGPRs. An earlier revision of this fix stored
//   `ptrtoint(blockaddress(@kernel, %bb_<retAddr>)) to i64` so the
//   cascade could compare against a `blockaddress` constant and let
//   LLVM's SCCP+InstCombine fold the cmp to `i1 true` on the hot
//   path. In practice the hi/lo split imposed by `storeSGPR64` (AMDGPU
//   SGPR pairs are two i32 halves joined back with shl/or at the
//   dispatch site) defeats that fold across phi joins, and the
//   `BlockAddress` SDNode survives into AMDGPU ISel -- which has no
//   pattern for materialising a `BlockAddress` as an i64 register
//   value (there is no relocation for "address of arbitrary BB inside
//   a kernel"). llc then aborts with
//     `LLVM ERROR: Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
//   Using the target's source-MC byte offset as a plain i64 marker
//   sidesteps the issue entirely: the marker is a normal integer
//   constant on every contributing predecessor path, folds cleanly
//   through mem2reg + SCCP + InstCombine, and `BlockAddress` only
//   appears as the `label` operand of the `br`, which DOES have a
//   codegen pattern (normal conditional branch). The hot-path folded
//   shape is identical to before (SimplifyCFG collapses the cascade
//   to a direct branch); the cold path does a bounded runtime integer
//   equality check before reaching the trap BB.
//
// `targetInt` must be of `ctx.I64Ty`; we assert this to catch
// regressions that forget to unpack the SGPR pair to i64 before
// calling.
//
// `targets` MUST be non-empty (the analysis never produces an empty
// dispatch set; the caller refuses Unresolvable sites earlier).
//
// `siteOffset` is the source-MC byte offset of the dispatching
// instruction; it is embedded in dispatch BB names to keep them
// unique across multiple dispatch sites in the same kernel.
void emitEnumeratedDispatch(RaiseContext &Ctx, Value *TargetInt,
                            ArrayRef<uint64_t> Targets,
                            uint64_t SiteOffset) {
  assert(!Targets.empty() && "enumerated dispatch needs ≥1 target");
  assert(TargetInt->getType() == Ctx.I64Ty &&
         "enumerated dispatch expects i64 target marker");

  SmallString<32> SitePrefixStorage;
  raw_svector_ostream(SitePrefixStorage) << "dispatch_0x"
                                         << utohexstr(SiteOffset);
  StringRef SitePrefix = SitePrefixStorage;

  IRBuilder<> &B = Ctx.B;

  // Pre-create the unreachable trap block so we can name it
  // deterministically and reference it from the last cascade step.
  BasicBlock *UnreachableBb = BasicBlock::Create(
      Ctx.C, SitePrefix.str() + "_unreachable", Ctx.Kernel);

  for (size_t I = 0; I < Targets.size(); ++I) {
    BasicBlock *TargetBb = Ctx.lookupBB(Targets[I]);
    Constant *MarkerCi = ConstantInt::get(Ctx.I64Ty, Targets[I]);
    SmallString<48> CmpName;
    raw_svector_ostream(CmpName) << SitePrefix << "_cmp_" << I;
    Value *Cmp = B.CreateICmpEQ(TargetInt, MarkerCi, CmpName);

    BasicBlock *FallthroughBb;
    if (I + 1 < Targets.size()) {
      SmallString<48> NextName;
      raw_svector_ostream(NextName) << SitePrefix << "_" << (I + 1);
      FallthroughBb = BasicBlock::Create(Ctx.C, NextName, Ctx.Kernel);
    } else {
      FallthroughBb = UnreachableBb;
    }
    B.CreateCondBr(Cmp, TargetBb, FallthroughBb);
    B.SetInsertPoint(FallthroughBb);
  }

  // Builder is now positioned at the start of unreachableBB. Emit the
  // unreachable terminator. The block is a BlockAddress-free terminal
  // sink -- no other code emits into it.
  B.CreateUnreachable();
}

// Lower a scalar F32-to-F32 SOP1 rounding instruction (s_ceil_f32,
// s_floor_f32, s_trunc_f32, s_rndne_f32) to the named LLVM intrinsic.
// The source SGPR is bitcast to F32, fed through the intrinsic, and the
// F32 result is bitcast back to I32 before being written to the
// destination SGPR. AMDGPU stores the integral result back in
// floating-point format -- this is not an integer conversion.
void emitScalarF32Rounding(RaiseContext &Ctx, OpResolver &Op,
                           Intrinsic::ID IntrinsicID, StringRef Name) {
  Value *SourceF32 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
  Function *IntrinsicFn =
      Intrinsic::getOrInsertDeclaration(&Ctx.M, IntrinsicID, {Ctx.F32Ty});
  Value *ResultF32 = Ctx.B.CreateCall(IntrinsicFn, {SourceF32}, Name);
  Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                      Ctx.B.CreateBitCast(ResultF32, Ctx.I32Ty));
}

} // namespace

// SPE attribute registrations. Every CanonicalOp listed here has been audited
// to route EXEC writes through `regs.storeExec` -- directly for the
// SAVEEXEC family, via `writeReg{32,64,ExecWidth}` -> `storeExec` for
// S_MOV_B{32,64} and S_NOT_B{32,64}. See AGENTS.md's SPE audit note
// before touching this list.
ArrayRef<CanonicalOpAttrSpec> getHandlerSOP1Attrs() {
  static constexpr CanonicalOpAttrSpec kAttrs[] = {
      {CanonicalOp::S_MOV_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_MOV_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_NOT_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_NOT_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_AND_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_OR_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_XOR_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_ANDN2_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_ORN2_SAVEEXEC_B32, {/*routesExecThroughStoreExec=*/true}},
  };
  return kAttrs;
}

HandlerResult handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  if (Sop == CanonicalOp::S_MOV_B32) {
    ParsedReg Dst = Op.dst();
    ParsedReg SrcReg = Op.isSrcReg(0) ? Op.srcReg(0) : ParsedReg{};
    // s_mov_b32 vcc_lo, sN restores a per-lane wave mask into VCC. The carry
    // is genuinely per-lane under wave32 -> wave64 widening (e.g. the
    // v_div_scale_f32 flag chain), so prefer the recorded per-lane shadow over
    // the default 32-bit path, which re-widens the truncated mask and loses the
    // upper lanes (see hotswap/docs/sgpr-wave-mask-translation.md). Mirrors the
    // V_CNDMASK_B32 SGPR-condition consumer in handle-valu-vop3p.cpp.
    if (Dst.RegKind == ParsedReg::VCC && SrcReg.RegKind == ParsedReg::SGPR &&
        SrcReg.BaseIdx >= 0) {
      // Same-BB: the producer's exact per-lane i1.
      if (Value *Shadow = Ctx.lookupSgprWaveMaskI1(SrcReg.BaseIdx)) {
        Ctx.Regs.storeVCC(Ctx.B, Shadow);
        Hr.Handled = true;
        return Hr;
      }
      // Cross-BB: the same-BB cache is cleared at a block boundary, but the
      // memory-backed shadow survives. Prefer it when valid, falling back to
      // re-widening the SGPR only when no producer recorded a shadow.
      if (Value *ShadowValid = Ctx.loadSgprWaveMaskValid(SrcReg.BaseIdx)) {
        Value *ShadowExec = Ctx.loadSgprWaveMaskExec(SrcReg.BaseIdx);
        Value *ShadowI1 =
            Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, ShadowExec);
        Value *SgprVal = Ctx.Isa.isWave32()
                             ? Ctx.Regs.loadSGPR32(Ctx.B, SrcReg.BaseIdx)
                             : Ctx.Regs.loadSGPR64(Ctx.B, SrcReg.BaseIdx);
        Value *Fallback =
            Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, SgprVal);
        Value *Sel = Ctx.B.CreateSelect(ShadowValid, ShadowI1, Fallback,
                                        "sgpr_mask_shadow_sel");
        Ctx.Regs.storeVCC(Ctx.B, Sel);
        Hr.Handled = true;
        return Hr;
      }
    }
    Value *Src = Op.src(0);
    Ctx.Regs.writeReg32(Ctx.B, Dst, Src);
    if (Dst.RegKind == ParsedReg::SGPR && SrcReg.RegKind == ParsedReg::EXEC) {
      Value *ExecI1 = Ctx.Projection.extractLaneBitFromWaveMask(
          Ctx.B, Ctx.Regs.loadExec(Ctx.B));
      Ctx.recordSgprWaveMaskI1(Dst.BaseIdx, ExecI1, /*isPair=*/false);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_MOV_B64) {
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Op.src64(0));
    Hr.Handled = true;
    return Hr;
  }
  // S_*_SAVEEXEC_B32 family -- save old EXEC into dst SGPR and
  // update EXEC via the family-specific combine.  The dst SGPR is
  // source-width (32-bit on wave32 source) so the i64 oldExec is
  // truncated when it lands in the alloca -- lossy under wave-native
  // cross-widening where the two halves of i64 EXEC can differ.
  //
  // Shadow propagation: we have the full-width `oldExec` in hand.
  // After `writeRegExecWidth` calls the reg-file's `onSgprWritten`
  // callback (which INVALIDATES the shadow for the dst SGPR),
  // re-record the shadow with the per-lane i1 extracted from
  // `oldExec` via the projection's `extractLaneBitFromWaveMask`.
  // Subsequent consumers (V_CNDMASK using this SGPR, or a
  // downstream S_XOR_B32 that ANDs the saved mask with the new
  // EXEC to compute the "else-branch" mask) see the correct
  // per-lane i1 instead of the narrow-mask fallback.
  //
  // Covers the Triton gfx1250 tl.sort at small BLOCK_N idiom
  // `s_and_saveexec_b32 sN, vcc; s_xor_b32 sN, exec_lo, sN` --
  // SAVEEXEC records `oldExec`'s i1 on sN, the sibling S_XOR_B32
  // handler extracts the current EXEC's i1 and XORs with the
  // shadowed sN i1, producing the wave-correct "lanes that became
  // inactive" mask for the V_CNDMASK consumer.
  //
  // Structurally safe: if the dst isn't an SGPR (e.g., dst == EXEC
  // itself -- non-saveexec form?  there isn't one for these
  // opcodes) the helper is a no-op.  The recorded i1 is a fresh
  // SSA value so `I2` (SSA-monotonic within a BB) holds.
  auto RecordOldExecShadowOnDst = [&](Value *OldExec) {
    ParsedReg Dst = Op.dst();
    if (Dst.RegKind != ParsedReg::SGPR)
      return;
    llvm::Value *OldExecI1 =
        Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, OldExec);
    Ctx.recordSgprWaveMaskI1(Dst.BaseIdx, OldExecI1, /*isPair=*/false);
  };

  if (Sop == CanonicalOp::S_AND_SAVEEXEC_B32) {
    Value *OldExec = Ctx.Regs.loadExec(Ctx.B);
    Value *Src = Op.srcExecWidth(0);
    Ctx.Regs.writeRegExecWidth(Ctx.B, Op.dst(), OldExec);
    RecordOldExecShadowOnDst(OldExec);
    Value *NewExec = Ctx.B.CreateAnd(OldExec, Src, "new_exec");
    Ctx.Regs.storeExec(Ctx.B, NewExec);
    Hr.SccResult = NewExec;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_OR_SAVEEXEC_B32) {
    Value *OldExec = Ctx.Regs.loadExec(Ctx.B);
    Value *Src = Op.srcExecWidth(0);
    Ctx.Regs.writeRegExecWidth(Ctx.B, Op.dst(), OldExec);
    RecordOldExecShadowOnDst(OldExec);
    Value *NewExec = Ctx.B.CreateOr(OldExec, Src, "new_exec");
    Ctx.Regs.storeExec(Ctx.B, NewExec);
    Hr.SccResult = NewExec;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_XOR_SAVEEXEC_B32) {
    Value *OldExec = Ctx.Regs.loadExec(Ctx.B);
    Value *Src = Op.srcExecWidth(0);
    Ctx.Regs.writeRegExecWidth(Ctx.B, Op.dst(), OldExec);
    RecordOldExecShadowOnDst(OldExec);
    Value *NewExec = Ctx.B.CreateXor(OldExec, Src, "new_exec");
    Ctx.Regs.storeExec(Ctx.B, NewExec);
    Hr.SccResult = NewExec;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ANDN2_SAVEEXEC_B32) {
    // S_AND_NOT1_SAVEEXEC (canonical ANDN2): D = EXEC; EXEC = S0 & ~EXEC.
    // "not1" inverts the EXEC operand, not the SGPR source: `Src & ~OldExec`.
    // Unlike commutative AND/OR/XOR_SAVEEXEC, the swapped `OldExec & ~Src` is a
    // different (not0) op and breaks the if/else divergence idiom.
    Value *OldExec = Ctx.Regs.loadExec(Ctx.B);
    Value *Src = Op.srcExecWidth(0);
    Ctx.Regs.writeRegExecWidth(Ctx.B, Op.dst(), OldExec);
    RecordOldExecShadowOnDst(OldExec);
    Value *NewExec = Ctx.B.CreateAnd(Src, Ctx.B.CreateNot(OldExec), "new_exec");
    Ctx.Regs.storeExec(Ctx.B, NewExec);
    Hr.SccResult = NewExec;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ORN2_SAVEEXEC_B32) {
    // S_OR_NOT1_SAVEEXEC (canonical ORN2): D = EXEC; EXEC = S0 | ~EXEC.
    // "not1" inverts EXEC, giving `Src | ~OldExec` (see ANDN2 above).
    Value *OldExec = Ctx.Regs.loadExec(Ctx.B);
    Value *Src = Op.srcExecWidth(0);
    Ctx.Regs.writeRegExecWidth(Ctx.B, Op.dst(), OldExec);
    RecordOldExecShadowOnDst(OldExec);
    Value *NewExec = Ctx.B.CreateOr(Src, Ctx.B.CreateNot(OldExec), "new_exec");
    Ctx.Regs.storeExec(Ctx.B, NewExec);
    Hr.SccResult = NewExec;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_GETPC_B64) {
    // Stub: the destination's symbolic PC is irrelevant for raised
    // IR. For Pattern A chains, the chain's binary value is never
    // read after we emit the `br label %target`. For Pattern B call
    // sites, the call-site rewrite in raiser.cpp overwrites the
    // ret-pair with a `blockaddress` after the chain's high-half
    // terminator runs, so the binary PC the chain would otherwise
    // produce is also discarded. Writing zero keeps SROA happy and
    // surfaces any stray downstream read as an obvious-zero use that
    // would crash the verifier rather than silently miscompile.
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), ConstantInt::get(Ctx.I64Ty, 0));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_SET_PC_I64) {
    // Look up the static analysis classification (Pattern A direct,
    // Pattern B enumerated-dispatch, or Unresolvable). Both patterns
    // emit a terminator into the current BB (and Pattern B / DispatchSet
    // also append a chain of dispatch sub-blocks via
    // `emitEnumeratedDispatch`); the raiser's BB-layout phase has
    // already promoted the next linear offset to a leader so subsequent
    // instructions land in their own BBs.
    if (!Ctx.SetpcAnalysis) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOP1",
          "s_set_pc_i64 reached without a SetPcAnalysis "
          "(raiser pipeline is missing the Phase 1.1 step)");
      return Hr;
    }
    auto It = Ctx.SetpcAnalysis->SetpcSites.find(Di.Offset);
    if (It == Ctx.SetpcAnalysis->SetpcSites.end()) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOP1",
          "s_set_pc_i64 site not classified by SetPcAnalysis");
      return Hr;
    }
    const SetPcSiteInfo &Info = It->second;
    switch (Info.SiteKind) {
    case SetPcSiteInfo::Kind::DirectA: {
      Ctx.B.CreateBr(Ctx.lookupBB(Info.DirectTarget));
      Hr.Handled = true;
      return Hr;
    }
    case SetPcSiteInfo::Kind::IndirectB:
    case SetPcSiteInfo::Kind::DispatchSet: {
      // Both shapes lower to the same enumerated-dispatch cascade:
      // read the source SGPR pair as i64 (it holds the per-predecessor
      // marker -- the resolved target's source-MC byte offset, written
      // either by the call-site chain-terminator hook in raiser.cpp for
      // IndirectB, or by the dispatch-target chain-terminator hook for
      // DispatchSet), then emit a cmp+br cascade against each
      // enumerated target offset. See `emitEnumeratedDispatch` above
      // for why this is a cascade of integer equality compares and not
      // `indirectbr` / a ptr-equality check against `blockaddress`.
      // The classification difference is purely semantic (return vs.
      // forward dispatch); the lowering mechanism is identical.
      Value *RetVal = Ctx.Regs.loadSGPR64(
          Ctx.B, static_cast<int>(Info.IndirectRetPairLowReg));
      RetVal->setName("ret_pc_marker");
      emitEnumeratedDispatch(Ctx, RetVal, Info.IndirectTargets,
                             Di.Offset);
      Hr.Handled = true;
      return Hr;
    }
    case SetPcSiteInfo::Kind::Unresolvable:
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(Di, "SOP1",
                                                  Info.RefusalReason);
      return Hr;
    }
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "SOP1", "s_set_pc_i64 SetPcSiteInfo::Kind not handled");
    return Hr;
  }
  if (Sop == CanonicalOp::S_SWAP_PC_I64) {
    // Branch-and-link. setpc_analysis classifies the call-target
    // pair (ssrc) as DirectA (chain resolves the absolute callee
    // offset intra-block), DispatchSet (inter-block dataflow
    // enumerates a bounded set of callee/branch targets reaching
    // this site through distinct CFG paths -- the tensilelite
    // "activation function dispatcher" shape), or Unresolvable (the
    // pair's value cannot be statically enumerated).
    //
    // For both DirectA and DispatchSet we materialise
    // `blockaddress(@kernel, %BB_returnAddr)` cast to i64 into sdst
    // BEFORE the terminator (so a downstream Pattern B
    // `s_set_pc_i64 sdst` in the callee can consume that
    // blockaddress via its enumerated-dispatch cascade). The
    // terminator itself is `br label %BB_callee` for DirectA or a
    // cmp+br cascade against `[list]` (via
    // `emitEnumeratedDispatch`) for DispatchSet. The
    // chain-terminator hook in raiser.cpp has already rewritten
    // ssrc to hold the matching BlockAddress on every contributing
    // CFG path, so each cascade `icmp eq` resolves to a constant
    // after mem2reg + SCCP rather than running a true runtime check.
    //
    // IndirectB on a swap_pc is NOT a valid classification: by
    // construction, IndirectB describes a return-side use (the pair
    // was written by some caller's chain terminator in a different
    // block) and a swap_pc reading such a pair would be a
    // function-pointer dispatch through a return slot. The analysis
    // never produces IndirectB for a swap_pc site (the source pair
    // is the call target, not a return address), so we refuse
    // loudly if it ever appears.
    //
    // Unresolvable is refused loudly with the analysis's diagnostic.
    // See canonical-op.h's S_SWAP_PC_I64 doc for the lowering contract.
    if (!Ctx.SetpcAnalysis) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOP1",
          "s_swap_pc_i64 reached without a SetPcAnalysis "
          "(raiser pipeline is missing the Phase 1.1 step)");
      return Hr;
    }
    auto It = Ctx.SetpcAnalysis->SetpcSites.find(Di.Offset);
    if (It == Ctx.SetpcAnalysis->SetpcSites.end()) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOP1",
          "s_swap_pc_i64 site not classified by SetPcAnalysis");
      return Hr;
    }
    const SetPcSiteInfo &Info = It->second;
    if (Info.SiteKind == SetPcSiteInfo::Kind::Unresolvable) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(Di, "SOP1",
                                                  Info.RefusalReason);
      return Hr;
    }
    if (Info.SiteKind == SetPcSiteInfo::Kind::IndirectB) {
      // Defensive: the analysis should never produce IndirectB for
      // a swap_pc site (a swap_pc's source pair is a call target,
      // not a return slot -- IndirectB is the return-side use of
      // such a pair). If it ever does, refuse loudly so the
      // mismatch surfaces rather than silently mis-lowering.
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOP1",
          "s_swap_pc_i64 classified as IndirectB by setpc_analysis "
          "(unexpected -- IndirectB is the return-side classification "
          "for s_set_pc_i64; a swap_pc reaching this code path "
          "indicates an analysis invariant violation)");
      return Hr;
    }
    // Materialise the return address marker (the offset of the BB
    // immediately after the swap) into sdst on both DirectA and
    // DispatchSet paths. Phase 1 of setpc_analysis has already
    // promoted `(di.Offset + di.size)` to a leader so subsequent
    // linear instructions live in their own BB; we simply write the
    // offset of that BB as a plain i64 constant, and the downstream
    // IndirectB consumer of sdst reads it back and compares it in a
    // cmp+br cascade. See `emitEnumeratedDispatch` above for why we
    // use an integer marker rather than `ptrtoint(blockaddress(...))`
    // (AMDGPU ISel cannot materialise a `BlockAddress` as an i64).
    uint64_t ReturnAddr = Di.Offset + Di.Size;
    // Force the target BB to exist in the lift so the subsequent
    // `br label %bb_<returnAddr>` has a valid destination; we don't
    // use the returned BB pointer here.
    (void)Ctx.lookupBB(ReturnAddr);
    Value *RetMarker = ConstantInt::get(Ctx.I64Ty, ReturnAddr);
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), RetMarker);

    if (Info.SiteKind == SetPcSiteInfo::Kind::DirectA) {
      Ctx.B.CreateBr(Ctx.lookupBB(Info.DirectTarget));
      Hr.Handled = true;
      return Hr;
    }
    // DispatchSet: emit an enumerated-dispatch cascade through the
    // source pair into the enumerated targets. The source pair holds
    // a per-predecessor i64 marker (the resolved callee's source-MC
    // byte offset), rewritten by the chain-terminator hook in
    // raiser.cpp on each contributing predecessor path.
    Value *CallTarget = Ctx.Regs.loadSGPR64(
        Ctx.B, static_cast<int>(Info.IndirectRetPairLowReg));
    CallTarget->setName("swap_call_target_marker");
    emitEnumeratedDispatch(Ctx, CallTarget, Info.IndirectTargets,
                           Di.Offset);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ADD_PC_I64) {
    std::optional<int64_t> ConstOpt = evalOperandAsConst(Di.Inst, 0);
    if (!ConstOpt) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOP1",
          "s_add_pc_i64 with non-literal source (SGPR-pair form unsupported)");
      return Hr;
    }
    uint64_t Target = Di.Offset + Di.Size + static_cast<uint64_t>(*ConstOpt);
    Ctx.B.CreateBr(Ctx.lookupBB(Target));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_NOT_B64) {
    Hr.SccResult = Ctx.B.CreateNot(Op.src64(0), "not64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_NOT_B32) {
    Hr.SccResult = Ctx.B.CreateNot(Op.src(0), "not32");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BREV_B32) {
    Function *Brev = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::bitreverse, {Ctx.I32Ty});
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateCall(Brev, {Op.src(0)}, "sbrev"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FF1_I32_B32) {
    Function *Cttz = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::cttz,
                                                       {Ctx.I32Ty});
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateCall(Cttz, {Op.src(0), ConstantInt::getTrue(Ctx.I1Ty)},
                         "ff1"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BCNT1_I32_B32) {
    Function *Ctpop = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ctpop, {Ctx.I32Ty});
    Hr.SccResult = Ctx.B.CreateCall(Ctpop, {Op.src(0)}, "bcnt1");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BCNT1_I32_B64) {
    Function *Ctpop64 = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ctpop, {Ctx.I64Ty});
    Value *R = Ctx.B.CreateCall(Ctpop64, {Op.src64(0)}, "bcnt1_64");
    Hr.SccResult = Ctx.B.CreateTrunc(R, Ctx.I32Ty);
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FF1_I32_B64) {
    Function *Cttz64 = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::cttz, {Ctx.I64Ty});
    Value *R = Ctx.B.CreateCall(
        Cttz64, {Op.src64(0), ConstantInt::getTrue(Ctx.I1Ty)}, "ff1_64");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateTrunc(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // s_ff0_i32_b{32,64} -- find first 0 bit (lowest position), -1 if
  // none. SOPInstructions.td:278-279 omits an LLVM ISel pattern, so
  // we lower directly: invert the source and reuse the cttz path
  // shared with V_FFBL_B32 (handle-valu-small-ops.cpp), then patch
  // the all-ones-input case to -1 since llvm.cttz with
  // is_zero_poison=false returns the bitwidth (32 / 64) for a zero
  // input rather than the AMDGPU's -1 sentinel.
  if (Sop == CanonicalOp::S_FF0_I32_B32) {
    Function *Cttz = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::cttz,
                                                       {Ctx.I32Ty});
    Value *Src = Op.src(0);
    Value *Inv = Ctx.B.CreateNot(Src, "ff0_inv");
    Value *Raw = Ctx.B.CreateCall(
        Cttz, {Inv, ConstantInt::getFalse(Ctx.I1Ty)}, "ff0_raw");
    Value *IsAllOnes = Ctx.B.CreateICmpEQ(
        Src, ConstantInt::getAllOnesValue(Ctx.I32Ty), "ff0_allones");
    Value *Res = Ctx.B.CreateSelect(
        IsAllOnes, Ctx.B.getInt32(-1), Raw, "ff0");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FF0_I32_B64) {
    Function *Cttz64 = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::cttz, {Ctx.I64Ty});
    Value *Src64 = Op.src64(0);
    Value *Inv = Ctx.B.CreateNot(Src64, "ff0_inv64");
    Value *Raw = Ctx.B.CreateCall(
        Cttz64, {Inv, ConstantInt::getFalse(Ctx.I1Ty)}, "ff0_raw64");
    Value *RawTrunc = Ctx.B.CreateTrunc(Raw, Ctx.I32Ty, "ff0_raw32");
    Value *IsAllOnes = Ctx.B.CreateICmpEQ(
        Src64, ConstantInt::getAllOnesValue(Ctx.I64Ty), "ff0_allones64");
    Value *Res = Ctx.B.CreateSelect(
        IsAllOnes, Ctx.B.getInt32(-1), RawTrunc, "ff0_64");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FLBIT_I32_B64) {
    Function *Ctlz64 = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ctlz, {Ctx.I64Ty});
    Value *R = Ctx.B.CreateCall(
        Ctlz64, {Op.src64(0), ConstantInt::getTrue(Ctx.I1Ty)}, "flbit64");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateTrunc(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FLBIT_I32_B32) {
    Function *Ctlz = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::ctlz,
                                                      {Ctx.I32Ty});
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateCall(Ctlz, {Op.src(0), ConstantInt::getTrue(Ctx.I1Ty)},
                         "flbit"));
    Hr.Handled = true;
    return Hr;
  }
  // s_flbit_i32 / s_flbit_i32_i64 -- signed find-leading-bit-not-equal-
  // to-sign-bit. SOPInstructions.td:296-298. Lower via the dedicated
  // llvm.amdgcn.sffbh intrinsic, which is overloaded on the source
  // integer type and selects back to v_ffbh_i32_e32 (or its 64-bit
  // pseudo equivalent) on AMDGPU. Hardware returns -1 for uniform-sign
  // input (0 or all-ones) -- the intrinsic shares the same convention,
  // so no explicit zero-fixup is needed.
  if (Sop == CanonicalOp::S_FLBIT_I32) {
    Function *Sffbh = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_sffbh, {Ctx.I32Ty});
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateCall(Sffbh, {Op.src(0)}, "sflbit"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FLBIT_I32_I64) {
    Function *Sffbh = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_sffbh, {Ctx.I64Ty});
    Value *R = Ctx.B.CreateCall(Sffbh, {Op.src64(0)}, "sflbit64");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateTrunc(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_SEXT_I32_I8) {
    Value *V = Ctx.B.CreateTrunc(Op.src(0), Ctx.I8Ty);
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateSExt(V, Ctx.I32Ty, "sext8"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_SEXT_I32_I16) {
    Value *V = Ctx.B.CreateTrunc(Op.src(0), Type::getInt16Ty(Ctx.C));
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateSExt(V, Ctx.I32Ty, "sext16"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CVT_F16_F32) {
    Value *Src = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *Half = Ctx.B.CreateFPTrunc(Src, Ctx.F16Ty, "s_cvt_h");
    Value *Bits = Ctx.B.CreateBitCast(Half, Type::getInt16Ty(Ctx.C));
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateZExt(Bits, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CVT_F32_F16 ||
      Sop == CanonicalOp::S_CVT_HI_F32_F16) {
    Value *Src = Op.src(0);
    if (Sop == CanonicalOp::S_CVT_HI_F32_F16)
      Src = Ctx.B.CreateLShr(Src, ConstantInt::get(Ctx.I32Ty, 16),
                             "s_cvt_hi_f16_bits32");
    Value *Bits = Ctx.B.CreateTrunc(Src, Type::getInt16Ty(Ctx.C),
                                    "s_cvt_f16_bits");
    Value *Half = Ctx.B.CreateBitCast(Bits, Ctx.F16Ty);
    Value *Result = Ctx.B.CreateFPExt(Half, Ctx.F32Ty, "s_cvt_f");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateBitCast(Result, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CVT_F32_U32) {
    Value *R = Ctx.B.CreateUIToFP(Op.src(0), Ctx.F32Ty, "s_cvt_f");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CVT_F32_I32) {
    Value *R = Ctx.B.CreateSIToFP(Op.src(0), Ctx.F32Ty, "s_cvt_f");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CVT_U32_F32) {
    Value *S = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateFPToUI(S, Ctx.I32Ty, "s_cvt_u"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CVT_I32_F32) {
    Value *S = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateFPToSI(S, Ctx.I32Ty, "s_cvt_i"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CEIL_F32) {
    emitScalarF32Rounding(Ctx, Op, Intrinsic::ceil, "s_ceil");
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FLOOR_F32) {
    emitScalarF32Rounding(Ctx, Op, Intrinsic::floor, "s_floor");
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_TRUNC_F32) {
    emitScalarF32Rounding(Ctx, Op, Intrinsic::trunc, "s_trunc");
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_RNDNE_F32) {
    emitScalarF32Rounding(Ctx, Op, Intrinsic::roundeven, "s_rndne");
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ABS_I32) {
    Function *AbsF =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::abs, {Ctx.I32Ty});
    Value *R = Ctx.B.CreateCall(AbsF, {Op.src(0), Ctx.B.getFalse()}, "s_abs");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), R);
    Hr.Handled = true;
    return Hr;
  }
  // s_bitset{0,1}_b{32,64}: clear or set a single bit in sdst.
  //   B32: bit index = src0[4:0], dst and tied read are 32-bit.
  //   B64: bit index = src0[5:0], dst and tied read are 64-bit (src0 is
  //        still an SReg_32 per LLVM's `SOP1_64_32` class).
  // These are read-modify-write: the destination's prior value is the
  // tied `sdst_in` operand in TableGen (`SOP1_32` / `SOP1_64_32` with
  // `tied_in=1` and `Constraints = "$sdst = $sdst_in"`), and the bit
  // index arrives in `src0` at src index 0.  SCC is not updated.
  //
  // The MC layer collapses the tied `$sdst_in` slot -- the AMDGPU
  // disassembler emits a 2-operand MCInst (`sdst`, `src0`) and the
  // tie is reconstituted only at MachineInstr lowering time. This
  // matches the S_CMOV_B{32,64} pattern below: the prior dst value
  // must be read explicitly via `regs.readReg{32,64}(op.dst())`, not
  // pulled from `op.src(1)`. (The `KKnownTiedIn` audit in
  // decode.cpp keeps `sdst_in` in the *driftCheck* allow-list -- i.e.
  // we declare it semantically a real input -- but no actual MCInst
  // operand survives disassembly to land in srcMap, so the read has
  // to come from the destination register itself.)
  if (Sop == CanonicalOp::S_BITSET0_B32 || Sop == CanonicalOp::S_BITSET1_B32 ||
      Sop == CanonicalOp::S_BITSET0_B64 || Sop == CanonicalOp::S_BITSET1_B64) {
    bool Is64 = (Sop == CanonicalOp::S_BITSET0_B64 || Sop == CanonicalOp::S_BITSET1_B64);
    bool IsSet = (Sop == CanonicalOp::S_BITSET1_B32 || Sop == CanonicalOp::S_BITSET1_B64);
    llvm::Type *Ty = Is64 ? Ctx.I64Ty : Ctx.I32Ty;
    // Hardware only consumes low log2(width) bits of the bit-index src;
    // mask explicitly so `shl 1, N` never becomes poison for N >= width.
    Value *BitIdx = Ctx.B.CreateAnd(Op.src(0),
                                    ConstantInt::get(Ctx.I32Ty,
                                                     Is64 ? 0x3F : 0x1F));
    if (Is64) BitIdx = Ctx.B.CreateZExt(BitIdx, Ctx.I64Ty);
    Value *Mask = Ctx.B.CreateShl(ConstantInt::get(Ty, 1), BitIdx);
    Value *Old = Is64 ? Ctx.Regs.readReg64(Ctx.B, Op.dst())
                      : Ctx.Regs.readReg32(Ctx.B, Op.dst());
    Value *Res = IsSet
                     ? Ctx.B.CreateOr(Old, Mask, "bitset1")
                     : Ctx.B.CreateAnd(Old, Ctx.B.CreateNot(Mask), "bitset0");
    if (Is64)
      Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Res);
    else
      Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  // s_cmov_b{32,64}: scalar conditional move on SCC. Hardware
  // semantics (per the gfx1250 ISA manual; see also
  // SOPInstructions.td `let Uses = [SCC]`):
  //   if (SCC) sdst = src; else sdst stays unchanged
  // SCC is read but not written.
  //
  // LLVM's SOP1_32/SOP1_64 pseudo for S_CMOV_B{32,64} declares
  //   `(outs sdst), (ins src0)`
  // *without* a tied sdst_in input -- the dst-on-SCC=0 read-modify
  // is implicit in the hardware encoding rather than modeled at
  // the MachineInstr level. So `op.nSrcs()` is 1 here (just src0)
  // and the prior dst value must be read explicitly via
  // `regs.readReg{32,64}(op.dst())`. The companion S_BITSET ops
  // above are the opposite case: their tied sdst_in is in srcMap
  // at index 1 because LLVM's `KKnownTiedIn` audit (decode.cpp)
  // keeps it. This asymmetry is a property of the LLVM .td
  // definitions, not a transpiler choice.
  if (Sop == CanonicalOp::S_CMOV_B32) {
    Value *Cond = Ctx.Regs.loadSCC(Ctx.B);
    Value *Src = Op.src(0);
    Value *OldDst = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateSelect(Cond, Src, OldDst, "scmov"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CMOV_B64) {
    Value *Cond = Ctx.Regs.loadSCC(Ctx.B);
    Value *Src = Op.src64(0);
    Value *OldDst = Ctx.Regs.readReg64(Ctx.B, Op.dst());
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(),
                        Ctx.B.CreateSelect(Cond, Src, OldDst, "scmov64"));
    Hr.Handled = true;
    return Hr;
  }
  // S_SET_VGPR_MSB is SOPP format -- handled in handleSOPP, not here.
  // GFX12+ `s_barrier_signal` appears in SOP1 encoding; model it as a no-op
  // (the paired SOPP `s_barrier_wait` does the actual rendezvous).
  if (Sop == CanonicalOp::S_BARRIER_SIGNAL) {
    Hr.Handled = true;
    return Hr;
  }
  return Hr;
}

} // namespace COMGR::hotswap
