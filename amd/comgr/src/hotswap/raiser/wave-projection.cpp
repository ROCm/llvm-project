//===- wave-projection.cpp - Hotswap transpiler ---------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/wave-projection.h"

#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

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

#include <cstdint>

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace COMGR::hotswap {

static constexpr uint64_t DispatchWorkgroupSizeXOffset = 4;
static constexpr uint64_t DispatchWorkgroupSizeYOffset = 6;

// ----------------------------------------------------------------------------
// WaveProjection base: lane-id derivation is shared across every projection
// that keeps each target lane mapped 1:1 to a hardware lane.
// ----------------------------------------------------------------------------

Value *WaveProjection::emitLaneIdx(IRBuilder<> &B) const {
  // Lane id (mbcnt vs all-ones, base 0) is EXEC-independent and
  // function-invariant: emit it once at a point that dominates the whole
  // function and reuse it everywhere. If nothing consumes it, DCE drops it.
  //
  // The cache is keyed by the function it was emitted into, the kernel boundary
  // this class can observe: a projection reused across kernels re-emits per
  // function instead of returning a value that belongs to a different one.
  Function *F = B.GetInsertBlock()->getParent();
  if (CachedLaneIdx && CachedLaneIdxFunc == F)
    return CachedLaneIdx;

  // Emit in the entry block, after any leading allocas. The allocas must stay
  // at the top of the entry block or mem2reg/SROA may decline to promote them,
  // so insert at the first non-alloca instruction rather than the block start.
  BasicBlock &Entry = F->getEntryBlock();
  IRBuilder<> EB(&Entry, Entry.getFirstNonPHIOrDbgOrAlloca());

  Module *M = Entry.getModule();
  Type *I32Ty = EB.getInt32Ty();
  Function *MbcntLo =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_lo);
  Value *AllOnes = ConstantInt::getSigned(I32Ty, -1);
  Value *Zero32 = ConstantInt::get(I32Ty, 0);
  Value *LaneId = EB.CreateCall(MbcntLo, {AllOnes, Zero32}, "lane_lo");
  if (waveMaskTy() != I32Ty) {
    Function *MbcntHi =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_hi);
    LaneId = EB.CreateCall(MbcntHi, {AllOnes, LaneId}, "lane_id");
  }
  CachedLaneIdxFunc = F;
  CachedLaneIdx = LaneId;
  return LaneId;
}

Value *WaveProjection::emitTargetWorkitemId(IRBuilder<> &B,
                                            unsigned Dimension) const {
  assert(Dimension < CachedWorkitemIds.size() && "invalid work-item dimension");
  Function *F = B.GetInsertBlock()->getParent();
  if (CachedWorkitemIds[Dimension])
    return CachedWorkitemIds[Dimension];

  static constexpr Intrinsic::ID Ids[] = {Intrinsic::amdgcn_workitem_id_x,
                                          Intrinsic::amdgcn_workitem_id_y,
                                          Intrinsic::amdgcn_workitem_id_z};
  static constexpr const char *Names[] = {"tid_x", "tid_y", "tid_z"};
  BasicBlock &Entry = F->getEntryBlock();
  IRBuilder<> EB(&Entry, Entry.getFirstNonPHIOrDbgOrAlloca());
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(Entry.getModule(), Ids[Dimension]);
  return CachedWorkitemIds[Dimension] = EB.CreateCall(Fn, {}, Names[Dimension]);
}

Value *WaveProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  return emitTargetWorkitemId(B, 0);
}

Value *WaveProjection::emitTargetWaveId(IRBuilder<> &B) const {
  Module *M = B.GetInsertBlock()->getModule();
  if (Tgt.hasArchitectedSgprs()) {
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_wave_id);
    return B.CreateCall(Fn, {}, "target_wave_id");
  }

  Function *DispatchPtrFn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_dispatch_ptr);
  Value *DispatchPtr = B.CreateCall(DispatchPtrFn, {}, "dispatch_ptr");
  auto LoadWorkgroupSize = [&](uint64_t Offset, const Twine &Name) {
    Value *Ptr = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), DispatchPtr,
                                              Offset, Name + "_ptr");
    LoadInst *Size = B.CreateAlignedLoad(B.getInt16Ty(), Ptr, Align(2), Name);
    return B.CreateZExt(Size, B.getInt32Ty(), Name + "_i32");
  };
  Value *SizeX =
      LoadWorkgroupSize(DispatchWorkgroupSizeXOffset, "workgroup_size_x");
  Value *SizeY =
      LoadWorkgroupSize(DispatchWorkgroupSizeYOffset, "workgroup_size_y");

  Value *X = emitTargetWorkitemId(B, 0);
  Value *Y = emitTargetWorkitemId(B, 1);
  Value *Z = emitTargetWorkitemId(B, 2);
  Value *Row = B.CreateAdd(Y, B.CreateMul(Z, SizeY), "wave_flat_yz");
  Value *FlatId = B.CreateAdd(X, B.CreateMul(Row, SizeX), "wave_flat_id");
  return B.CreateUDiv(FlatId, B.getInt32(Tgt.waveSize()), "target_wave_id");
}

Value *ReplicationProjection::emitSourceWaveId(IRBuilder<> &B) const {
  return emitTargetWaveId(B);
}

Value *ReplicationProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  Value *Raw = WaveProjection::emitWorkitemIdX(B);
  // When the target wave is wider than the source workgroup, the upper target
  // lanes [MaxFlatWG, WaveSize) carry no source workitem, so their raw workitem
  // id is just the hardware lane index. Clamp those lanes to workitem 0 so they
  // replicate lane 0's in-bounds addressing; real lanes are unchanged and every
  // committed result is identical.
  if (Tgt.waveSize() > Src.waveSize() && MaxFlatWG > 0 &&
      MaxFlatWG < Tgt.waveSize()) {
    // The "real lane" test is the flat local id, not workitem.id.x. Under
    // replication the target lane index is the flat local id (lanes are
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
  Value *Y = emitTargetWorkitemId(B, 1);
  Value *Packed =
      B.CreateOr(X,
                 B.CreateShl(Y, ConstantInt::get(I32Ty, WorkitemIdYBitOffset),
                             "tid_y_shl"),
                 "tid_xy");
  if (NumDims < 3)
    return Packed;
  Value *Z = emitTargetWorkitemId(B, 2);
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
  const unsigned SourceBits = SourceTy->getPrimitiveSizeInBits();
  const unsigned MaskBits = Mask->getType()->getPrimitiveSizeInBits();
  if (Mask->getType() == SourceTy) {
    return Mask;
  }
  if (MaskBits < SourceBits) {
    return B.CreateZExtOrTrunc(Mask, SourceTy, Name);
  }

  Value *LaneId = emitLaneIdx(B);
  Value *SourceWaveBase = B.CreateAnd(
      LaneId, B.getInt32(~(static_cast<uint32_t>(Src.waveSize()) - 1u)),
      Name + "_base");
  Value *Shift =
      B.CreateZExtOrTrunc(SourceWaveBase, Mask->getType(), Name + "_shift");
  Value *AtSourceWave = B.CreateLShr(Mask, Shift, Name + "_at_srcwave");
  return B.CreateTrunc(AtSourceWave, SourceTy, Name);
}

Value *ReplicationProjection::emitPackedWorkitemId(IRBuilder<> &B,
                                                   unsigned NumDims) const {
  // 1-D: identical to the (already phantom-clamped) X-only seed.
  if (NumDims < 2)
    return emitWorkitemIdX(B);
  // Build from the unclamped x so the phantom-lane clamp applies once to the
  // whole packed id; otherwise a clamped-to-0 x OR'd with a non-zero Y/Z would
  // leave phantom lanes with a stray id instead of replicating lane 0.
  Value *Raw = packWorkitemId(B, WaveProjection::emitWorkitemIdX(B), NumDims);
  if (Tgt.waveSize() > Src.waveSize() && MaxFlatWG > 0 &&
      MaxFlatWG < Tgt.waveSize()) {
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
  // The architectural boot state of a dispatched wave is "every source lane
  // active", i.e. all-ones in the source-width EXEC storage. A projection that
  // decouples the modeled EXEC from the hardware EXEC overrides this hook.
  return ConstantInt::getSigned(execStorageTy(), -1);
}

Value *WaveProjection::wrapAsWWMValue(IRBuilder<> &B, Value *V,
                                      const Twine &Name) const {
  if (providesFullWaveExecInvariant())
    return V;
  // strict.wwm is declared llvm_any_ty, but the backend lowers only integer
  // and floating-point scalars and fixed vectors thereof. Assert on that
  // subset so an unsupported type fails here rather than at lowering.
  Type *T = V->getType();
  Type *ElemTy =
      T->isVectorTy() ? cast<FixedVectorType>(T)->getElementType() : T;
  (void)ElemTy;
  assert((ElemTy->isIntegerTy() || ElemTy->isFloatingPointTy()) &&
         "wrapAsWWMValue supports only integer / floating-point scalars "
         "and fixed-length vectors thereof");
  Module *M = B.GetInsertBlock()->getModule();
  Function *WwmFn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_strict_wwm, {T});
  return B.CreateCall(WwmFn, {V}, Name);
}

// ----------------------------------------------------------------------------
// ReplicationProjection.
// ----------------------------------------------------------------------------

Value *ReplicationProjection::emitLaneActiveBit(IRBuilder<> &B,
                                                Value *ExecVal) const {
  // Project the target-lane id onto the source EXEC mask under
  // replication: target lane L is active iff bit `L mod W_src` of
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

Value *ReplicationProjection::ballotI1ToWidth(IRBuilder<> &B, Value *Pred,
                                              Type *ResultTy,
                                              const Twine &Name) const {
  assert(Pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *Ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy()});
  Value *WaveMask = B.CreateCall(Ballot, {Pred}, Name);
  unsigned WantedBits = ResultTy->getPrimitiveSizeInBits();
  unsigned WaveBits = waveMaskTy()->getPrimitiveSizeInBits();
  assert(WantedBits <= WaveBits &&
         "ballotI1ToWidth: wantedBits > waveBits (wave64 source on wave32 "
         "target) has no replication projection; this direction needs "
         "an explicit policy decision before use");
  if (WantedBits == WaveBits)
    return WaveMask;
  if (WantedBits < WaveBits)
    // Truncation is the replication projection of the target ballot
    // onto the source wave width.
    return B.CreateTrunc(WaveMask, ResultTy, Name + "_trunc");
  // `wantedBits > waveBits`: wave64 source on wave32 target. No correct
  // replication projection exists (the wider source wave has lanes
  // that do not exist in the narrower target), so zero-extending would
  // invent bits; fall through without returning so this direction is not
  // silently miscompiled.
}

Value *ReplicationProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                         Value *V) const {
  if (V->getType() == B.getInt1Ty())
    return V;
  Type *I64Ty = B.getInt64Ty();
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, I64Ty);
  Type *TargetTy = waveMaskTy();
  unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = TargetTy->getPrimitiveSizeInBits();
  if (SrcBits < DstBits) {
    // A narrow source-wave-width mask must be widened to the target
    // wave-mask width before the per-lane shift extracts a single bit.
    // Replicate the narrow mask into the upper half rather than zero-
    // extending: replication's contract is that target lane L reads
    // bit `L mod W_src` of the source mask, so a zext would make lanes in
    // the upper half always read 0.
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
  // Neutral `mask_*` names: this helper reads any wave mask as a per-lane
  // predicate, not only VCC, so a `vcc_` prefix would mislabel SGPR sources.
  Value *LaneIdxExt = B.CreateZExtOrTrunc(LaneIdx, TargetTy, "mask_lane_idx");
  Value *Shifted = B.CreateLShr(V, LaneIdxExt, "mask_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(TargetTy, 1), "mask_lane_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(TargetTy, 0), "mask_lane_i1");
}

// ----------------------------------------------------------------------------
// ReplicationDoubledDispatchProjection.
//
// Remaps the hardware workitem-id.x of a doubled-dispatch launch back onto the
// logical source id, so hardware lane `W_s + i` (a replica) sees the same
// logical thread as hardware lane `i`. Everything else is inherited from
// ReplicationProjection.
// ----------------------------------------------------------------------------

// logical_x = ((x_hw & ~(W_t-1)) >> log2(W_t/W_s)) | (x_hw & (W_s-1))
//
// The first term recovers the hardware wave index and rescales it to source
// lanes; the second term is the source lane within the wave (identical for a
// lane and its replica). For wave32->wave64 this is
// `((x_hw & ~63) >> 1) | (x_hw & 31)`.
static Value *emitDoubledDispatchLogicalX(IRBuilder<> &B, Value *RawX,
                                          unsigned SrcWaveSize,
                                          unsigned TgtWaveSize) {
  assert(TgtWaveSize > SrcWaveSize && (TgtWaveSize % SrcWaveSize) == 0 &&
         "doubled-dispatch remap requires widening with an integer "
         "wave-size ratio");
  Type *Ty = RawX->getType();
  const unsigned Ratio = TgtWaveSize / SrcWaveSize;
  const unsigned RatioLog2 = llvm::Log2_32(Ratio);
  Value *WaveAligned = B.CreateAnd(
      RawX, ConstantInt::get(Ty, ~static_cast<uint64_t>(TgtWaveSize - 1u)),
      "dd_wave_aligned");
  Value *WaveScaled = B.CreateLShr(WaveAligned, ConstantInt::get(Ty, RatioLog2),
                                   "dd_wave_base");
  Value *SrcLane =
      B.CreateAnd(RawX, ConstantInt::get(Ty, SrcWaveSize - 1u), "dd_src_lane");
  return B.CreateOr(WaveScaled, SrcLane, "dd_logical_x");
}

Value *
ReplicationDoubledDispatchProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  // Deliberately bypass ReplicationProjection::emitWorkitemIdX (the
  // phantom-lane clamp): a doubled dispatch has no phantom lanes, every
  // hardware lane is a real source thread or an exact replica of one.
  Value *Raw = WaveProjection::emitWorkitemIdX(B);
  return emitDoubledDispatchLogicalX(B, Raw, Src.waveSize(), Tgt.waveSize());
}

Value *
ReplicationDoubledDispatchProjection::emitSourceWaveId(IRBuilder<> &B) const {
  return emitTargetWaveId(B);
}

Value *ReplicationDoubledDispatchProjection::emitPackedWorkitemId(
    IRBuilder<> &B, unsigned NumDims) const {
  // Remapped x OR'd with the source's raw y/z fields. y/z are per-thread
  // correct as launched and become wave-uniform once x is doubled, so no
  // remap or clamp is applied to them.
  return packWorkitemId(B, emitWorkitemIdX(B), NumDims);
}

// ----------------------------------------------------------------------------
// WaveNativeProjection -- widening (wave32 -> wave64).
//
// The target is wave64, so waveMaskTy() is i64; this projection uses it for
// both the EXEC alloca storage and the ballot/lane-active arithmetic.
// ----------------------------------------------------------------------------

WaveNativeProjection::WaveNativeProjection(const ISAProfile &SrcIsa,
                                           const ISAProfile &TgtIsa,
                                           Type *I32Ty, Type *I64Ty)
    : WaveProjection(SrcIsa, TgtIsa, I32Ty, I64Ty) {
  // Restrict to the one direction where the widened-EXEC invariants are
  // well-defined: same-wave needs no widening, and narrowing loses lanes
  // regardless of policy.
  assert((SrcIsa.isWave32() && !TgtIsa.isWave32()) &&
         "WaveNativeProjection is defined only for wave32 source -> "
         "wave64 target widening");

  // Widen EXEC storage to the target hardware mask and treat each half of the
  // target wave as a distinct source wave. `emitInitialExec` forces HW
  // EXEC=-1 kernel-wide, so mbcnt-derived EXEC writes project into independent
  // target-width masks and a narrow EXEC_LO write broadcasts across both
  // halves.
  ExecStorageTy = waveMaskTy();
  NumSourceWavesPerTarget = 2;
  BroadcastNarrowExecLoWrite = true;
  ProvidesFullWaveExecInvariant = true;
  PreservesMbcntDerivedExec = true;
}

Value *WaveNativeProjection::emitSourceWaveId(IRBuilder<> &B) const {
  Value *TargetWave = emitTargetWaveId(B);
  Value *FirstSourceWave = B.CreateMul(
      TargetWave, B.getInt32(numSourceWavesPerTarget()), "first_source_wave");
  Value *SourceWaveInTarget = B.CreateUDiv(
      emitLaneIdx(B), B.getInt32(Src.waveSize()), "source_wave_in_target");
  return B.CreateAdd(FirstSourceWave, SourceWaveInTarget, "source_wave_id");
}

Value *WaveNativeProjection::emitInitialExec(IRBuilder<> &B) const {
  // Call `@llvm.amdgcn.init_whole_wave` to set hardware EXEC = -1 (all target
  // lanes active) and capture the original per-lane active bit; ballot the
  // captured bit into a wave-width mask to seed the EXEC alloca. This keeps the
  // modeled source EXEC (read by emitUnderExec) separate from the hardware
  // EXEC, so the kernel body runs with all lanes hardware-active while stores
  // still honour the original mask.
  Module *M = B.GetInsertBlock()->getModule();
  Function *InitWw =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_init_whole_wave);
  Value *OriginalActive = B.CreateCall(InitWw, {}, "orig_active");
  // Ballot the per-lane i1 into a wave-width mask via the projection's own
  // ballot so the result type matches waveMaskTy().
  return ballotI1ToWidth(B, OriginalActive, waveMaskTy(), "saved_exec");
}

Value *WaveNativeProjection::emitLaneActiveBit(IRBuilder<> &B,
                                               Value *ExecVal) const {
  // Target lane L is active iff bit L of the widened EXEC (waveMaskTy()) is
  // set; the shift index is the full target lane id, with no modulo fold.
  Value *LaneId = emitLaneIdx(B);
  Type *ExecTy = ExecVal->getType();
  assert(ExecTy == waveMaskTy() &&
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
  unsigned WaveBits = waveMaskTy()->getPrimitiveSizeInBits();
  assert(WantedBits <= WaveBits &&
         "WaveNativeProjection::ballotI1ToWidth: wantedBits > waveBits "
         "is not defined for wave32 source -> wave64 target cross-"
         "widening; caller must request resultTy <= waveMaskTy");
  if (WantedBits == WaveBits)
    return WaveMask;
  if (WantedBits < WaveBits)
    // Narrowing the full target ballot to a source-width scalar loses the
    // upper half (target lanes 32..63): a source instruction naming a single
    // 32-bit SGPR destination cannot hold a 64-bit mask.
    return B.CreateTrunc(WaveMask, ResultTy, Name + "_trunc");
  // `wantedBits > waveBits` cannot occur for the wave32 -> wave64 direction
  // this projection handles, so fall through without returning rather than
  // zero-extending, which would invent bits the source wave does not have.
}

Value *WaveNativeProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                        Value *V) const {
  if (V->getType() == B.getInt1Ty())
    return V;
  Type *I64Ty = B.getInt64Ty();
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, I64Ty);
  Type *TargetTy = waveMaskTy();
  unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = TargetTy->getPrimitiveSizeInBits();
  if (SrcBits < DstBits) {
    // Widen a source-width mask back to target width by replication, so target
    // lane K and K+W_src read the same bit. A zero-extend would leave the upper
    // half always reading 0, deactivating target lanes 32..63 whenever EXEC is
    // restored through a source-width SGPR.
    Value *Zext = B.CreateZExt(V, TargetTy);
    Value *Shifted = B.CreateShl(Zext, SrcBits);
    V = B.CreateOr(Zext, Shifted, "wn_mask_widen");
  } else if (SrcBits > DstBits) {
    V = B.CreateTrunc(V, TargetTy);
  } else if (V->getType() != TargetTy) {
    V = B.CreateBitCast(V, TargetTy);
  }
  Value *LaneIdx = emitLaneIdx(B);
  // Neutral `mask_*` names: this helper reads any wave mask as a per-lane
  // predicate, not only VCC, so a `vcc_` prefix would mislabel SGPR sources.
  Value *LaneIdxExt =
      B.CreateZExtOrTrunc(LaneIdx, TargetTy, "wn_mask_lane_idx");
  Value *Shifted = B.CreateLShr(V, LaneIdxExt, "wn_mask_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(TargetTy, 1), "wn_mask_lane_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(TargetTy, 0), "wn_mask_lane_i1");
}

// ----------------------------------------------------------------------------
// ThreadLoopProjection.
// ----------------------------------------------------------------------------

ThreadLoopProjection::ThreadLoopProjection(const ISAProfile &SrcIsa,
                                           const ISAProfile &TgtIsa,
                                           Type *I32Ty, Type *I64Ty)
    : WaveProjection(SrcIsa, TgtIsa, I32Ty, I64Ty) {
  assert(TgtIsa.waveSize() > SrcIsa.waveSize() &&
         "ThreadLoopProjection is defined only for widening "
         "(target wave > source wave)");

  assert((TgtIsa.waveSize() % SrcIsa.waveSize()) == 0 &&
         "ThreadLoopProjection requires target wave size to be an integer "
         "multiple of source wave size");

  ExecStorageTy = waveMaskTy();
  NumSourceWavesPerTarget = TgtIsa.waveSize() / SrcIsa.waveSize();
  SourceWaveScopedLaneOps = true;
}

Value *ThreadLoopProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  assert(IterationAlloca &&
         "ThreadLoopProjection::emitWorkitemIdX requires an iteration alloca; "
         "raiser must call setIterationAlloca before emitting source workitem "
         "ids");
  Value *Tid = emitTargetWorkitemId(B, 0);
  Value *LaneId = emitLaneIdx(B);
  const unsigned SrcBits = Src.waveSize();
  const unsigned TgtBits = Tgt.waveSize();
  Value *Iter = B.CreateLoad(B.getInt32Ty(), IterationAlloca, "tl_iter");
  Value *Base = B.CreateAnd(Tid, B.getInt32(~(TgtBits - 1u)), "tl_tid_base");
  Value *SourceLane =
      B.CreateAnd(LaneId, B.getInt32(SrcBits - 1u), "tl_source_lane");
  Value *WaveOffset =
      B.CreateMul(Iter, B.getInt32(SrcBits), "tl_source_wave_off");
  return B.CreateAdd(B.CreateAdd(Base, WaveOffset, "tl_tid_wave_base"),
                     SourceLane, "tl_tid");
}

Value *ThreadLoopProjection::emitSourceWaveId(IRBuilder<> &B) const {
  assert(IterationAlloca &&
         "ThreadLoopProjection::emitSourceWaveId requires an iteration alloca");
  Value *TargetWave = emitTargetWaveId(B);
  Value *FirstSourceWave = B.CreateMul(
      TargetWave, B.getInt32(numSourceWavesPerTarget()), "first_source_wave");
  Value *Iteration =
      B.CreateLoad(B.getInt32Ty(), IterationAlloca, "source_wave_in_target");
  return B.CreateAdd(FirstSourceWave, Iteration, "source_wave_id");
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
  const unsigned WaveBits = waveMaskTy()->getPrimitiveSizeInBits();
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
                       ? waveMaskTy()
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
      (TargetTy == waveMaskTy())
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
  if (Di.defsExec())
    return true;
  // On a wave32 source, hardware EXEC is 32-bit (== EXEC_LO) and EXEC_HI is a
  // free scratch scalar, so an explicit def of EXEC_HI alone is a scratch
  // write, not an EXEC write.
  const bool SourceIsWave32 =
      Mc.SubtargetInfo->hasFeature(AMDGPU::FeatureWavefrontSize32);
  const MCInstrDesc &Desc = Mc.InstrInfo->get(Di.Inst.getOpcode());
  for (unsigned I = 0; I < Desc.getNumDefs() && I < Di.Inst.getNumOperands();
       ++I) {
    const MCOperand &Mop = Di.Inst.getOperand(I);
    if (!Mop.isReg() || !Mop.getReg())
      continue;
    // EXEC has no subtarget-specific MC aliases, so the raw MC register already
    // matches the pseudo-register id; no mc2PseudoReg normalization is needed.
    MCRegister Reg = Mop.getReg();
    if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO)
      return true;
    if (Reg == AMDGPU::EXEC_HI && !SourceIsWave32)
      return true;
  }
  return false;
}

// ----------------------------------------------------------------------------
// Cross-wave warning.
// ----------------------------------------------------------------------------

bool emitCrossWaveWarning(const WaveProjection &Proj, const MCState &Mc,
                          ArrayRef<DecodedInst> Insts, StringRef SourceIsa,
                          StringRef TargetIsa) {
  if (Proj.sourceIsa().waveSize() == Proj.targetIsa().waveSize())
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

  // Emit a warn-only diagnostic under -debug-only=wave-projection.
  LLVM_DEBUG({
    dbgs() << "transpiler: WARNING: cross-wave translation of an "
              "EXEC-manipulating kernel relies on replication, "
              "which is not provably correct in general.\n"
           << "  source ISA wave size: " << Proj.sourceIsa().waveSize() << " ("
           << SourceIsa << ")\n"
           << "  target ISA wave size: " << Proj.targetIsa().waveSize() << " ("
           << (TargetIsa.empty() ? SourceIsa : TargetIsa) << ")\n"
           << "  first EXEC-writer: " << getMnemonic(Mc, FirstExecWriter->Inst)
           << " at offset 0x"
           << format_hex_no_prefix(FirstExecWriter->Offset, 4) << "\n"
           << "  rationale: the kernel manipulates EXEC; replicating it "
              "across wave halves will double per-lane side effects in a "
              "way the source author did not specify. Empirically this is "
              "correct for kernels whose EXEC writers are lane-position-"
              "independent (pointwise ops with bounds checks against a "
              "uniform >= target_wave_bits).\n";
  });
  return true;
}

} // namespace COMGR::hotswap
