//===- wave-projection.cpp - Hotswap transpiler ---------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wave-projection.h"

#include "decoded-inst.h"
#include "mc-state.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, EXEC_LO, EXEC_HI
#include "Utils/AMDGPUBaseInfo.h"            // AMDGPU::mc2PseudoReg

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace COMGR::hotswap {

// ----------------------------------------------------------------------------
// WaveProjection base: lane-id derivation is shared across every projection
// that keeps each target lane mapped 1:1 to a hardware lane. Subclasses that
// change that mapping (e.g. a future thread-loop projection) would override.
// ----------------------------------------------------------------------------

Value *WaveProjection::emitLaneIdx(IRBuilder<> &B) const {
  // Lane id (mbcnt vs all-ones, base 0) is EXEC-independent and
  // function-invariant: emit it once at a point that dominates the whole
  // function and reuse it everywhere. If nothing consumes it, DCE drops it.
  if (CachedLaneIdx)
    return CachedLaneIdx;

  // Emit in the entry block, after any leading allocas. The allocas must stay
  // at the top of the entry block or mem2reg/SROA may decline to promote them,
  // so insert at the first non-alloca instruction rather than the block start.
  BasicBlock &Entry = B.GetInsertBlock()->getParent()->getEntryBlock();
  IRBuilder<> EB(&Entry, Entry.getFirstNonPHIOrDbgOrAlloca());

  Module *M = Entry.getModule();
  Type *I32Ty = EB.getInt32Ty();
  Function *MbcntLo =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_lo);
  Value *AllOnes = ConstantInt::getSigned(I32Ty, -1);
  Value *Zero32 = ConstantInt::get(I32Ty, 0);
  Value *LaneId = EB.CreateCall(MbcntLo, {AllOnes, Zero32}, "lane_lo");
  if (WaveMaskTy != I32Ty) {
    Function *MbcntHi =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_hi);
    LaneId = EB.CreateCall(MbcntHi, {AllOnes, LaneId}, "lane_id");
  }
  CachedLaneIdx = LaneId;
  return LaneId;
}

Value *WaveProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  Module *M = B.GetInsertBlock()->getModule();
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_workitem_id_x);
  return B.CreateCall(Fn, {}, "tid");
}

Value *ModuloReplicationProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  Value *Raw = WaveProjection::emitWorkitemIdX(B);
  // When the target wave is wider than the source workgroup, the upper target
  // lanes [MaxFlatWG, WaveSize) carry no source workitem, so their raw workitem
  // id is just the hardware lane index. Clamp those lanes to workitem 0 so they
  // replicate lane 0's in-bounds addressing; real lanes are unchanged and every
  // committed result is identical. See hotswap/docs/modrep-predicate-chain.md
  // for why these lanes can still issue memory ops despite the modeled EXEC.
  if (Tgt.WaveSize > Src.WaveSize && MaxFlatWG > 0 &&
      MaxFlatWG < Tgt.WaveSize) {
    // The "real lane" test is the flat local id, not workitem.id.x. Under
    // modulo replication the target lane index is the flat local id (lanes are
    // laid out in flat order), so a lane is real iff its index is below the
    // flattened workgroup size. Comparing workitem.id.x against MaxFlatWG would
    // only be correct for 1D workgroups, where tid.x == flat local id; for a
    // multidimensional workgroup tid.x is just the X coordinate while MaxFlatWG
    // is the flattened total.
    Value *Limit = ConstantInt::get(I32Ty, MaxFlatWG);
    Value *FlatLaneId = emitLaneIdx(B);
    Value *IsRealLane = B.CreateICmpULT(FlatLaneId, Limit, "tid_is_real_lane");
    Raw = B.CreateSelect(IsRealLane, Raw, ConstantInt::get(I32Ty, 0),
                         "tid_phantom_clamp");
  }
  return Raw;
}

// Bit offsets of the Y/Z fields in the packed kernel-entry v0 workitem id
// (x[0:9] | y[10:19] | z[20:29])
static constexpr unsigned WorkitemIdYBitOffset = 10;
static constexpr unsigned WorkitemIdZBitOffset = 20;

Value *WaveProjection::packWorkitemId(IRBuilder<> &B, Value *X,
                                      unsigned NumDims) const {
  if (NumDims < 2)
    return X;
  Module *M = B.GetInsertBlock()->getModule();
  Function *FnY =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_workitem_id_y);
  Value *Y = B.CreateCall(FnY, {}, "tid_y");
  Value *Packed =
      B.CreateOr(X,
                 B.CreateShl(Y, ConstantInt::get(I32Ty, WorkitemIdYBitOffset),
                             "tid_y_shl"),
                 "tid_xy");
  if (NumDims < 3)
    return Packed;
  Function *FnZ =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_workitem_id_z);
  Value *Z = B.CreateCall(FnZ, {}, "tid_z");
  return B.CreateOr(Packed,
                    B.CreateShl(Z,
                                ConstantInt::get(I32Ty, WorkitemIdZBitOffset),
                                "tid_z_shl"),
                    "tid_xyz");
}

Value *WaveProjection::emitPackedWorkitemId(IRBuilder<> &B,
                                            unsigned NumDims) const {
  return packWorkitemId(B, emitWorkitemIdX(B), NumDims);
}

Value *WaveProjection::emitCurrentSourceWaveMask(IRBuilder<> &B, Value *Mask,
                                                 const Twine &Name) const {
  Type *SourceTy = sourceWaveMaskTy();
  assert(Mask->getType()->isIntegerTy() &&
         "emitCurrentSourceWaveMask expects an integer mask");
  if (Mask->getType() == SourceTy)
    return Mask;
  return B.CreateZExtOrTrunc(Mask, SourceTy, Name);
}

Value *
ModuloReplicationProjection::emitPackedWorkitemId(IRBuilder<> &B,
                                                  unsigned NumDims) const {
  // 1-D: identical to the (already phantom-clamped) X-only seed.
  if (NumDims < 2)
    return emitWorkitemIdX(B);
  // Build from the unclamped x so the phantom-lane clamp applies once to the
  // whole packed id; otherwise a clamped-to-0 x OR'd with a non-zero Y/Z would
  // leave phantom lanes with a stray id instead of replicating lane 0.
  Value *Raw = packWorkitemId(B, WaveProjection::emitWorkitemIdX(B), NumDims);
  if (Tgt.WaveSize > Src.WaveSize && MaxFlatWG > 0 &&
      MaxFlatWG < Tgt.WaveSize) {
    Value *Limit = ConstantInt::get(I32Ty, MaxFlatWG);
    Value *FlatLaneId = emitLaneIdx(B);
    Value *IsRealLane = B.CreateICmpULT(FlatLaneId, Limit, "tid_is_real_lane");
    // Clamp phantom upper lanes to the literal packed 0 (local id (0, 0, 0));
    // this copies nothing from lane 0. They stay hardware inactive and cannot
    // commit source-visible memory, so 0 is only an in-bounds address floor for
    // any address still computed for them.
    Raw = B.CreateSelect(IsRealLane, Raw, ConstantInt::get(I32Ty, 0),
                         "tid_phantom_clamp");
  }
  return Raw;
}

Value *WaveProjection::emitInitialExec(IRBuilder<> &B) const {
  // Default: the architectural boot state of a dispatched wave is
  // "every source lane active", i.e. all-ones in the source-width
  // EXEC storage. Projections that need to DECOUPLE the modeled
  // EXEC from the hardware EXEC (e.g. `WaveNativeProjection` below,
  // which forces hardware EXEC = -1 at entry via
  // `@llvm.amdgcn.init_whole_wave` and stores the captured original
  // per-lane active bit into the alloca) override this hook.
  return ConstantInt::getSigned(execStorageTy(), -1);
}

Value *WaveProjection::wrapAsWWMValue(IRBuilder<> &B, Value *V,
                                      const Twine &Name) const {
  if (providesFullWaveExecInvariant())
    return V;
  // `@llvm.amdgcn.strict.wwm`'s overload set in IntrinsicsAMDGPU.td is
  // `llvm_any_ty`, but in practice the backend lowers only a restricted
  // set of scalar and vector element types -- integers up to i64 and
  // float/half/bfloat/f32/f64 (plus their fixed-vector shapes).
  // Calling it with a pointer, aggregate, token, or other
  // backend-unsupported type would surface as a cryptic signature
  // error inside `Intrinsic::getOrInsertDeclaration`.  Assert on the
  // supported subset here so the misuse surfaces at the call site
  // instead.
  Type *T = V->getType();
  Type *ElemTy =
      T->isVectorTy() ? cast<FixedVectorType>(T)->getElementType() : T;
  (void)ElemTy;
  assert((ElemTy->isIntegerTy() || ElemTy->isFloatingPointTy()) &&
         "wrapAsWWMValue supports only integer / floating-point scalars "
         "and fixed-length vectors thereof; other types are not in the "
         "AMDGPU backend's strict.wwm lowering coverage (pointer, token, "
         "aggregate, etc. would produce a cryptic intrinsic-signature "
         "error).");
  Module *M = B.GetInsertBlock()->getModule();
  Function *WwmFn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_strict_wwm, {T});
  return B.CreateCall(WwmFn, {V}, Name);
}

// ----------------------------------------------------------------------------
// ModuloReplicationProjection.
// ----------------------------------------------------------------------------

Value *ModuloReplicationProjection::emitLaneActiveBit(IRBuilder<> &B,
                                                      Value *ExecVal) const {
  // Project the target-lane id onto the source EXEC mask under
  // modulo-replication: target lane L is active iff bit `L mod W_src` of
  // the source EXEC mask is set. Same-wave and narrowing cases collapse
  // to the identity because `lane_id < source_wave_bits` already; the
  // modulo is a no-op and the shift happens at source width.
  //
  // Shifting at source width also sidesteps the LLVM-IR poison rule that
  // `lshr iN, M` is poison for M >= N: the pre-modulo clamps the shift
  // into [0, execBits).
  Value *LaneId = emitLaneIdx(B);
  Type *ExecTy = ExecVal->getType();
  unsigned ExecBits = ExecTy->getPrimitiveSizeInBits();
  Value *LaneIdInExec = B.CreateZExtOrTrunc(LaneId, ExecTy, "spe_lane_idx");
  // execBits is a power of two (32 or 64), so modulo is bitwise AND.
  Value *LaneMod = B.CreateAnd(
      LaneIdInExec, ConstantInt::get(ExecTy, ExecBits - 1), "spe_lane_mod");
  Value *Shifted = B.CreateLShr(ExecVal, LaneMod, "spe_exec_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(ExecTy, 1), "spe_exec_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(ExecTy, 0), "spe_lane_active");
}

Value *ModuloReplicationProjection::ballotI1ToWidth(IRBuilder<> &B, Value *Pred,
                                                    Type *ResultTy,
                                                    const Twine &Name) const {
  assert(Pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *Ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy()});
  Value *WaveMask = B.CreateCall(Ballot, {Pred}, Name);
  unsigned WantedBits = ResultTy->getPrimitiveSizeInBits();
  unsigned WaveBits = WaveMaskTy->getPrimitiveSizeInBits();
  assert(WantedBits <= WaveBits &&
         "ballotI1ToWidth: wantedBits > waveBits (wave64 source on wave32 "
         "target) has no modulo-replication projection; this direction needs "
         "an explicit policy decision before use");
  if (WantedBits == WaveBits)
    return WaveMask;
  if (WantedBits < WaveBits)
    // MODREP: trunc is the modulo-replication projection of the target
    // ballot onto the source wave width.
    return B.CreateTrunc(WaveMask, ResultTy, Name + "_trunc");
  // `wantedBits > waveBits`: wave64 source on wave32 target. No correct
  // modulo-replication projection exists (the wider source wave has
  // lanes that do not exist in the narrower target), so zero-extending
  // would invent bits. Bail so a future wave64->wave32 lift lands here
  // rather than silently miscompiles.
}

Value *ModuloReplicationProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                               Value *V) const {
  if (V->getType() == B.getInt1Ty())
    return V;
  Type *I64Ty = B.getInt64Ty();
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, I64Ty);
  Type *TargetTy = WaveMaskTy;
  unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = TargetTy->getPrimitiveSizeInBits();
  if (SrcBits < DstBits) {
    // Cross-widening case: a narrow source-wave-width mask (e.g. the
    // 32-bit result of `ballotI1ToWidth(..., i32)` or a saved VCC-lo
    // read via `loadSGPR32`) has to be widened to the target wave-mask
    // width before the per-lane shift extracts a single bit.  A plain
    // `zext` zeros the upper `dstBits - srcBits` positions, which
    // under wave32 -> wave64 makes target lanes 32..63 always read a
    // zero (their `lane_id` shift lands in the zero-padded upper
    // half), and downstream every narrow-mask-guarded `v_cndmask_b32`
    // unconditionally picks its FALSE branch on those lanes.  In the
    // Triton SwiGLU shape (`corpus_swiglu_fp32`) that FALSE branch is
    // the 0x80000000 OOB-sentinel offset used to neutralise masked-
    // out buffer accesses, so all target-wave-upper-half stores land
    // out-of-bounds and the SRD bounds check silently drops them --
    // the observed "half of every target wave64's outputs stay at
    // their zero-initialised value" miscompile.  MODREP's contract
    // (`wave-size-translation.md` sec. 6 / class-"modulo-replication"
    // policy) says target lane L reads bit `L mod W_src` of the source
    // wave's mask, so the right widening *replicates* the narrow mask
    // into the upper half rather than zero-extending.  That matches
    // the `WaveNativeProjection::extractLaneBitFromWaveMask`
    // widen-by-replication path and makes a narrow-mask round-trip on
    // the consumer side symmetric with what both projections'
    // narrow-EXEC writers already do on the producer side; the full-
    // lane-id shift below then correctly selects the replicated bit
    // for every target lane.
    Value *Zext = B.CreateZExt(V, TargetTy);
    Value *Shifted = B.CreateShl(Zext, ConstantInt::get(TargetTy, SrcBits),
                                 "mask_widen_shl");
    V = B.CreateOr(Zext, Shifted, "mask_widen_replicate");
  } else if (SrcBits > DstBits) {
    V = B.CreateTrunc(V, TargetTy);
  } else if (V->getType() != TargetTy) {
    V = B.CreateBitCast(V, TargetTy);
  }
  Value *LaneIdx = emitLaneIdx(B);
  // Twine names are neutral (`mask_*`) rather than `vcc_*`: the helper
  // is called from every consumer that reads a wave mask as a per-lane
  // predicate -- the VCC consumer path via `readVCCAsWaveMask` AND the
  // SGPR-source `V_CNDMASK_B32_e64` consumer path in
  // `handle-valu-vop3p.cpp`. Keeping the old `vcc_` prefix would make
  // raised-IR dumps for e.g. the corpus_asin_fp32 kernel print
  // `%vcc_lane_idx` for reads of `s6`, which misleads.
  Value *LaneIdxExt = B.CreateZExtOrTrunc(LaneIdx, TargetTy, "mask_lane_idx");
  Value *Shifted = B.CreateLShr(V, LaneIdxExt, "mask_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(TargetTy, 1), "mask_lane_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(TargetTy, 0), "mask_lane_i1");
}

// ----------------------------------------------------------------------------
// ScaledModuloReplicationProjection.
//
// Remaps the hardware workitem-id.x of a scaled-dispatch launch back onto the
// logical source id, so hardware lane `W_s + i` (a replica) sees the same
// logical thread as hardware lane `i`. Everything else is inherited MODREP.
// ----------------------------------------------------------------------------

// logical_x = ((x_hw & ~(W_t-1)) >> log2(W_t/W_s)) | (x_hw & (W_s-1))
//
// The first term recovers the hardware wave index and rescales it to source
// lanes; the second term is the source lane within the wave (identical for a
// lane and its replica). For wave32->wave64 this is
// `((x_hw & ~63) >> 1) | (x_hw & 31)`.
static Value *emitScaledDispatchLogicalX(IRBuilder<> &B, Value *RawX,
                                         unsigned SrcWaveSize,
                                         unsigned TgtWaveSize) {
  assert(TgtWaveSize > SrcWaveSize && (TgtWaveSize % SrcWaveSize) == 0 &&
         "scaled-dispatch remap requires cross-widening with an integer "
         "wave-size ratio");
  Type *Ty = RawX->getType();
  const unsigned Ratio = TgtWaveSize / SrcWaveSize;
  const unsigned RatioLog2 = llvm::Log2_32(Ratio);
  // Hardware-wave-aligned high part, rescaled to source-lane units.
  Value *WaveAligned =
      B.CreateAnd(RawX,
                  ConstantInt::get(Ty, ~static_cast<uint64_t>(TgtWaveSize - 1u),
                                   /*IsSigned=*/true),
                  "dd_wave_aligned");
  Value *WaveScaled = B.CreateLShr(WaveAligned, ConstantInt::get(Ty, RatioLog2),
                                   "dd_wave_base");
  // Source lane within the wave (same for a lane and its replica).
  Value *SrcLane =
      B.CreateAnd(RawX, ConstantInt::get(Ty, SrcWaveSize - 1u), "dd_src_lane");
  return B.CreateOr(WaveScaled, SrcLane, "dd_logical_x");
}

Value *
ScaledModuloReplicationProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  // Deliberately bypass ModuloReplicationProjection::emitWorkitemIdX (the
  // phantom-lane clamp): a scaled dispatch has no phantom lanes, every
  // hardware lane is a real source thread or an exact replica of one.
  Value *Raw = WaveProjection::emitWorkitemIdX(B);
  return emitScaledDispatchLogicalX(B, Raw, Src.WaveSize, Tgt.WaveSize);
}

Value *ScaledModuloReplicationProjection::emitPackedWorkitemId(
    IRBuilder<> &B, unsigned NumDims) const {
  // Remapped x OR'd with the source's raw y/z fields. y/z are per-thread
  // correct as launched and become wave-uniform once x is doubled, so no
  // remap or clamp is applied to them.
  return packWorkitemId(B, emitWorkitemIdX(B), NumDims);
}

// ----------------------------------------------------------------------------
// WaveNativeProjection -- cross-widening (wave32 -> wave64).
//
// The base `WaveMaskTy` is already `tgtIsa.isWave32() ? i32 : i64`,
// which on the only supported direction (wave32 source -> wave64
// target) is `i64`. We reuse it directly for both the EXEC alloca
// storage and the ballot/lane-active arithmetic so the widths line up
// without any extra casting.
// ----------------------------------------------------------------------------

WaveNativeProjection::WaveNativeProjection(const ISAProfile &SrcIsa,
                                           const ISAProfile &TgtIsa,
                                           Type *I32Ty, Type *I64Ty)
    : WaveProjection(SrcIsa, TgtIsa, I32Ty, I64Ty) {
  // Restrict to the one translation direction where the wave-native
  // projection's extra invariants are well-defined. Same-wave paths
  // don't need a widened EXEC (ModRep already collapses to identity
  // there), and narrowing (wave64 source -> wave32 target) loses lanes
  // regardless of policy -- `ModuloReplicationProjection` in
  // `ballotI1ToWidth`; the wave-native projection is not a second
  //  answer for that direction.
  assert((SrcIsa.isWave32() && !TgtIsa.isWave32()) &&
         "WaveNativeProjection is defined only for wave32 source -> "
         "wave64 target cross-widening; other directions must use "
         "ModuloReplicationProjection (same-wave / narrowing) or a "
         "future ThreadLoopProjection implementation. See hotswap/"
         "docs/wave-size-translation.md 2.2 for the projection "
         "ladder.");

  // Widen EXEC storage to the target hardware mask and treat each half of
  // the target wave as a distinct source wave. `emitInitialExec` emits
  // `init_whole_wave` (HW EXEC=-1 kernel-wide), so mbcnt-derived EXEC writes
  // project into independent target-width masks and a narrow EXEC_LO write
  // broadcasts across both halves.
  ExecStorageTy = WaveMaskTy;
  NumSourceWavesPerTarget = 2;
  BroadcastNarrowExecLoWrite = true;
  ProvidesFullWaveExecInvariant = true;
  PreservesMbcntDerivedExec = true;
}

Value *WaveNativeProjection::emitInitialExec(IRBuilder<> &B) const {
  // Wave32 -> Wave64 cross-widening decouples the hardware EXEC (what
  // the target gfx942 wavefront actually applies) from the modeled
  // source EXEC (what the transpiler's `emitUnderExec` diamonds read
  // through the alloca). At kernel entry we call
  // `@llvm.amdgcn.init_whole_wave`, which:
  //
  //   (1) sets hardware EXEC = -1 (all 64 Wave64 lanes active), and
  //   (2) returns a per-lane i1 whose true-bits form the ORIGINAL
  //       hardware EXEC mask at dispatch time.
  //
  // We ballot (1) back into a wave-width i64 and return that as the
  // value to seed the EXEC alloca with. From this point on the
  // `emitUnderExec` diamonds guard every VGPR write, memory store,
  // LDS op, and atomic through an IR-level `br i1 %lane_active`
  // derived from the alloca -- the backend lowers those divergent
  // branches by setting hardware EXEC to the ballot of the
  // per-lane predicate inside each `do` block and restoring to
  // `EXEC = -1` afterwards, so no inactive source lane ever
  // commits a side effect. Between `emitUnderExec` diamonds the
  // hardware EXEC is -1, which is exactly what the WMMA -> MFMA
  // cross-lane pipeline in `wmma-lowering.cpp` needs to produce
  // correct per-lane output on all 64 Wave64 lanes.
  //
  // This replaces the prior per-MFMA-output `@llvm.amdgcn.strict.wwm`
  // strategy. See the long comment block on
  // `WaveProjection::emitInitialExec` for the register-allocator
  // pressure argument (`SIPreAllocateWWMRegs` requires dedicated
  // physregs for every vreg inside a WWM bracket, which a 128x128
  // matmul tile cannot satisfy).
  Module *M = B.GetInsertBlock()->getModule();
  Function *InitWw =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_init_whole_wave);
  Value *OriginalActive = B.CreateCall(InitWw, {}, "orig_active");
  // Ballot the per-lane i1 back into a wave-width mask. Reuses the
  // projection's own ballot emission so the width selection matches
  // `WaveMaskTy` (i64 on Wave64 target) and the single result-type
  // overload of `llvm.amdgcn.ballot` selected is the backend-
  // supported one for this subtarget.
  return ballotI1ToWidth(B, OriginalActive, WaveMaskTy, "saved_exec");
}

Value *WaveNativeProjection::emitLaneActiveBit(IRBuilder<> &B,
                                               Value *ExecVal) const {
  // Target lane L is active iff bit L of the widened EXEC is set. The
  // widened EXEC storage is `WaveMaskTy` (i64 on wave64 target), so
  // the shift index is the full target lane id (0..63) with no modulo
  // fold; that is the whole point of the wave-native projection
  // relative to `ModuloReplicationProjection::emitLaneActiveBit`,
  // which folds the target lane id into `lane_id mod W_src` and
  // thereby collapses target lanes 0..31 with 32..63.
  Value *LaneId = emitLaneIdx(B);
  Type *ExecTy = ExecVal->getType();
  assert(ExecTy == WaveMaskTy &&
         "WaveNativeProjection requires EXEC storage to match the "
         "target wave mask width; caller must size the alloca via "
         "execStorageTy()");
  Value *LaneIdInExec = B.CreateZExtOrTrunc(LaneId, ExecTy, "wn_lane_idx");
  Value *Shifted = B.CreateLShr(ExecVal, LaneIdInExec, "wn_exec_at_lane");
  Value *Bit = B.CreateAnd(Shifted, ConstantInt::get(ExecTy, 1), "wn_exec_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(ExecTy, 0), "wn_lane_active");
}

Value *WaveNativeProjection::ballotI1ToWidth(IRBuilder<> &B, Value *Pred,
                                             Type *ResultTy,
                                             const Twine &Name) const {
  assert(Pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *Ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy()});
  Value *WaveMask = B.CreateCall(Ballot, {Pred}, Name);
  unsigned WantedBits = ResultTy->getPrimitiveSizeInBits();
  unsigned WaveBits = WaveMaskTy->getPrimitiveSizeInBits();
  assert(WantedBits <= WaveBits &&
         "WaveNativeProjection::ballotI1ToWidth: wantedBits > waveBits "
         "is not defined for wave32 source -> wave64 target cross-"
         "widening; caller must request resultTy <= waveMaskTy");
  if (WantedBits == WaveBits)
    return WaveMask;
  if (WantedBits < WaveBits)
    // Narrowing the full target ballot to a source-width scalar loses
    // the upper half (target lanes 32..63). This is the one residual
    // truncation the wave-native projection accepts: source-ISA
    // instructions that name a single 32-bit SGPR destination (e.g.
    // `v_cmp_lt_u32_e64 s4, ...` on wave32) cannot hold a 64-bit
    // mask. The `handle-valu-vcmp.cpp` V_CMPX branch asks for
    // `resultTy = execStorageTy() = WaveMaskTy` and stays at full
    // width; only the V_CMP->SGPR branch asks for the narrower source
    // width and takes this trunc. Kernels that consume the truncated
    // mask as a per-lane wave mask downstream can still miscompile,
    // and such patterns remain the obstruction classifier's
    // responsibility to refuse (see `wave-size-obstruction.cpp`).
    return B.CreateTrunc(WaveMask, ResultTy, Name + "_trunc");
  // `wantedBits > waveBits`: wave32 target hardware ballot requested
  // wider than its native mask. This direction only arises on same-
  // target-wave lifts (not our wave32->wave64 cross-widening), so
  // reaching it under WaveNativeProjection is a raiser bug.
}

Value *WaveNativeProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                        Value *V) const {
  if (V->getType() == B.getInt1Ty())
    return V;
  Type *I64Ty = B.getInt64Ty();
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, I64Ty);
  Type *TargetTy = WaveMaskTy;
  unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = TargetTy->getPrimitiveSizeInBits();
  if (SrcBits < DstBits) {
    // Source-width wave mask (e.g. a 32-bit SGPR that caught the
    // output of `ballotI1ToWidth(..., i32, ...)` above) is widened
    // back to target width by *replication* so target lane K and
    // K+W_src read the same bit. Under wave-native this is the
    // conservative choice -- it matches what the V_CMP->SGPR trunc
    // already implicitly assumed when it picked lanes 0..W_src-1 as
    // canonical -- and it keeps `v_cndmask_b32` / `s_and_b64 exec,
    // ..., sN` rounds trips behaving like modulo-replication for
    // the residual save/restore pattern. Replacing replication with
    // a zero-extend would silently deactivate target lanes 32..63
    // whenever the kernel restores EXEC through a 32-bit SGPR; the
    // replication choice is the one that keeps the `v_cmpx ->
    // predicated store -> s_mov_b32 exec_lo, -1` shape working.
    Value *Zext = B.CreateZExt(V, TargetTy);
    Value *Shifted = B.CreateShl(Zext, SrcBits);
    V = B.CreateOr(Zext, Shifted, "wn_mask_widen");
  } else if (SrcBits > DstBits) {
    V = B.CreateTrunc(V, TargetTy);
  } else if (V->getType() != TargetTy) {
    V = B.CreateBitCast(V, TargetTy);
  }
  Value *LaneIdx = emitLaneIdx(B);
  // Neutral `mask_*` naming parity with the ModRep variant above --
  // same two-caller story (VCC consumer + SGPR-source V_CNDMASK_B32
  // consumer), same reason to avoid the old `wn_vcc_*` identifiers
  // surfacing in raised-IR dumps for kernels whose mask source is a
  // plain SGPR.
  Value *LaneIdxExt =
      B.CreateZExtOrTrunc(LaneIdx, TargetTy, "wn_mask_lane_idx");
  Value *Shifted = B.CreateLShr(V, LaneIdxExt, "wn_mask_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(TargetTy, 1), "wn_mask_lane_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(TargetTy, 0), "wn_mask_lane_i1");
}

Value *
WaveNativeProjection::emitCurrentSourceWaveMask(IRBuilder<> &B, Value *Mask,
                                                const Twine &Name) const {
  assert(Mask->getType()->isIntegerTy() &&
         "emitCurrentSourceWaveMask expects an "
         "integer mask");

  Type *SourceTy = sourceWaveMaskTy();
  unsigned SourceBits = SourceTy->getPrimitiveSizeInBits();
  unsigned MaskBits = Mask->getType()->getPrimitiveSizeInBits();
  if (MaskBits <= SourceBits)
    return B.CreateZExtOrTrunc(Mask, SourceTy, Name);

  Value *LaneId = emitLaneIdx(B);
  Value *SourceWaveBase = B.CreateAnd(
      LaneId, B.getInt32(~(static_cast<uint32_t>(Src.WaveSize) - 1u)),
      Name + "_base");
  Value *Shift =
      B.CreateZExtOrTrunc(SourceWaveBase, Mask->getType(), Name + "_shift");
  Value *AtSourceWave = B.CreateLShr(Mask, Shift, Name + "_at_srcwave");
  return B.CreateTrunc(AtSourceWave, SourceTy, Name);
}

// ----------------------------------------------------------------------------
// ThreadLoopProjection -- second rung of the coverage ladder described
// in hotswap/docs/wave-size-translation.md sec. 2.2.
//
// This implementation is intentionally conservative at the projection
// boundary: source-width EXEC storage, source-width lane selection for
// lane-active/wave-mask extraction, and source-width result typing for
// ballots. Selection remains opt-in in raiser.cpp.
// ----------------------------------------------------------------------------

ThreadLoopProjection::ThreadLoopProjection(const ISAProfile &SrcIsa,
                                           const ISAProfile &TgtIsa,
                                           Type *I32Ty, Type *I64Ty)
    : WaveProjection(SrcIsa, TgtIsa, I32Ty, I64Ty) {
  assert(TgtIsa.WaveSize > SrcIsa.WaveSize &&
         "ThreadLoopProjection is defined only for cross-widening "
         "(target wave > source wave)");

  assert((TgtIsa.WaveSize % SrcIsa.WaveSize) == 0 &&
         "ThreadLoopProjection requires target wave size to be an integer "
         "multiple of source wave size");

  ExecStorageTy = WaveMaskTy;
  NumSourceWavesPerTarget = TgtIsa.WaveSize / SrcIsa.WaveSize;
  SourceWaveScopedLaneOps = true;
}

Value *ThreadLoopProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  assert(IterationAlloca &&
         "ThreadLoopProjection::emitWorkitemIdX requires an iteration alloca; "
         "raiser must call setIterationAlloca before emitting source workitem "
         "ids");
  Module *M = B.GetInsertBlock()->getModule();
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_workitem_id_x);
  Value *Tid = B.CreateCall(Fn, {}, "tl_hw_tid");
  Value *LaneId = emitLaneIdx(B);
  const unsigned SrcBits = Src.WaveSize;
  const unsigned TgtBits = Tgt.WaveSize;
  Value *Iter = B.CreateLoad(B.getInt32Ty(), IterationAlloca, "tl_iter");
  Value *Base = B.CreateAnd(Tid, B.getInt32(~(TgtBits - 1u)), "tl_tid_base");
  Value *SourceLane =
      B.CreateAnd(LaneId, B.getInt32(SrcBits - 1u), "tl_source_lane");
  Value *WaveOffset =
      B.CreateMul(Iter, B.getInt32(SrcBits), "tl_source_wave_off");
  return B.CreateAdd(B.CreateAdd(Base, WaveOffset, "tl_tid_wave_base"),
                     SourceLane, "tl_tid");
}

Value *ThreadLoopProjection::emitLaneActiveBit(IRBuilder<> &B,
                                               Value *ExecVal) const {
  Value *LaneId = emitLaneIdx(B);
  Type *ExecTy = ExecVal->getType();
  const unsigned SourceBits = sourceWaveMaskTy()->getPrimitiveSizeInBits();
  Value *LaneIdInExec = B.CreateZExtOrTrunc(LaneId, ExecTy, "tl_lane_idx");
  Value *LaneMod = B.CreateAnd(
      LaneIdInExec, ConstantInt::get(ExecTy, SourceBits - 1), "tl_lane_mod");
  Value *Shifted = B.CreateLShr(ExecVal, LaneMod, "tl_exec_at_lane");
  Value *Bit = B.CreateAnd(Shifted, ConstantInt::get(ExecTy, 1), "tl_exec_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(ExecTy, 0), "tl_lane_active");
}

Value *ThreadLoopProjection::ballotI1ToWidth(IRBuilder<> &B, Value *Pred,
                                             Type *ResultTy,
                                             const Twine &Name) const {
  assert(Pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *Ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy()});
  Value *WaveMask = B.CreateCall(Ballot, {Pred}, Name);
  const unsigned WantedBits = ResultTy->getPrimitiveSizeInBits();
  const unsigned WaveBits = WaveMaskTy->getPrimitiveSizeInBits();
  assert(WantedBits <= WaveBits &&
         "ThreadLoopProjection::ballotI1ToWidth requires resultTy <= target "
         "wave mask width");
  if (WantedBits == WaveBits)
    return WaveMask;
  return B.CreateTrunc(WaveMask, ResultTy, Name + "_trunc");
}

Value *ThreadLoopProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                        Value *V) const {
  if (V->getType() == B.getInt1Ty())
    return V;
  Type *TargetTy = V->getType()->getPrimitiveSizeInBits() >
                           sourceWaveMaskTy()->getPrimitiveSizeInBits()
                       ? WaveMaskTy
                       : sourceWaveMaskTy();
  unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = TargetTy->getPrimitiveSizeInBits();
  if (SrcBits < DstBits) {
    V = B.CreateZExt(V, TargetTy);
  } else if (SrcBits > DstBits) {
    V = B.CreateTrunc(V, TargetTy);
  } else if (V->getType() != TargetTy) {
    V = B.CreateBitCast(V, TargetTy);
  }
  Value *LaneIdx = emitLaneIdx(B);
  Value *LaneIdxExt =
      B.CreateZExtOrTrunc(LaneIdx, TargetTy, "tl_mask_lane_idx");
  Value *ShiftIdx =
      (TargetTy == WaveMaskTy)
          ? LaneIdxExt
          : B.CreateAnd(LaneIdxExt, ConstantInt::get(TargetTy, DstBits - 1),
                        "tl_mask_lane_mod");
  Value *Shifted = B.CreateLShr(V, ShiftIdx, "tl_mask_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(TargetTy, 1), "tl_mask_lane_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(TargetTy, 0), "tl_mask_lane_i1");
}

// ----------------------------------------------------------------------------
// EXEC-writer detection.
// ----------------------------------------------------------------------------

bool instructionWritesEXEC(const DecodedInst &Di, const MCState &Mc) {
  if (Di.DefsExec)
    return true;
  // On a wave32 source, hardware EXEC is 32-bit (== EXEC_LO); EXEC_HI is a free
  // scratch scalar (see ParsedReg::EXEC_HI_SCRATCH), so an explicit def of
  // EXEC_HI alone is a scratch write, not an EXEC write.
  const bool SourceIsWave32 =
      Mc.SubtargetInfo->hasFeature(AMDGPU::FeatureWavefrontSize32);
  const MCInstrDesc &Desc = Mc.InstrInfo->get(Di.Inst.getOpcode());
  for (unsigned I = 0; I < Desc.getNumDefs() && I < Di.Inst.getNumOperands();
       ++I) {
    const MCOperand &Mop = Di.Inst.getOperand(I);
    if (!Mop.isReg() || !Mop.getReg())
      continue;
    MCRegister Reg = AMDGPU::mc2PseudoReg(Mop.getReg());
    if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO)
      return true;
    if (Reg == AMDGPU::EXEC_HI && !SourceIsWave32)
      return true;
  }
  return false;
}

// ----------------------------------------------------------------------------
// Phase 1.4 cross-wave warning.
// ----------------------------------------------------------------------------

bool emitCrossWaveWarning(const WaveProjection &Proj, const MCState &Mc,
                          ArrayRef<DecodedInst> Insts, StringRef SourceIsa,
                          StringRef TargetIsa) {
  if (Proj.sourceIsa().WaveSize == Proj.targetIsa().WaveSize)
    return false;

  const DecodedInst *FirstExecWriter = nullptr;
  for (const DecodedInst &Di : Insts) {
    if (instructionWritesEXEC(Di, Mc)) {
      FirstExecWriter = &Di;
      break;
    }
  }
  if (!FirstExecWriter)
    return false;

  // Route the legacy warn-only diagnostic through LLVM_DEBUG now that
  // the Phase 1.4.5 classifier (see `wave_size_obstruction.{hpp,cpp}`)
  // owns the gate decision. The structured decider in raiser.cpp emits
  // a precise per-obstruction trace via the same DEBUG_TYPE; this
  // legacy diagnostic remains only as a fallback that surfaces under
  // `-debug-only=wave-projection` when the classifier's trace is not
  // enough context. Enable via `raise_cli -debug-only=wave-projection`
  // or `llvm-opt -debug-only=wave-projection`.
  LLVM_DEBUG({
    dbgs() << "transpiler: WARNING: cross-wave translation of an "
              "EXEC-manipulating kernel relies on modulo-replication, "
              "which is not provably correct in general.\n"
           << "  source ISA wave size: " << Proj.sourceIsa().WaveSize << " ("
           << SourceIsa << ")\n"
           << "  target ISA wave size: " << Proj.targetIsa().WaveSize << " ("
           << (TargetIsa.empty() ? SourceIsa : TargetIsa) << ")\n"
           << "  first EXEC-writer: " << FirstExecWriter->RawMnemonic
           << " at offset 0x"
           << format_hex_no_prefix(FirstExecWriter->Offset, 4) << "\n"
           << "  rationale: the kernel manipulates EXEC; replicating it "
              "across wave halves will double per-lane side effects in a "
              "way the source author did not specify. Empirically this is "
              "correct for kernels whose EXEC writers are lane-position-"
              "independent (pointwise ops with bounds checks against a "
              "uniform >= target_wave_bits). The Phase 1.4.5 classifier "
              "(wave-size-obstruction.cpp) is the principled path for "
              "deciding between outcome (a)/(b)/(c) per hotswap/docs/"
              "wave-size-translation.md sec. 7.\n";
  });
  return true;
}

} // namespace COMGR::hotswap
