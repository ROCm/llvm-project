//===- wmma-lowering.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ============================================================================
// WMMA -> MFMA Lowering via Layout-Aware Lane Redistribution
// ============================================================================
//
// This file lowers Wave32 WMMA instructions (gfx1250 / RDNA4) to Wave64 MFMA
// instructions (gfx942 / CDNA3) using cross-lane data movement.
//
// Background
// ----------
// WMMA (Wave Matrix Multiply Accumulate) on gfx1250 is a Wave32 collective
// operation: 32 lanes cooperate to compute a 16x16 matrix multiply.  MFMA on
// gfx942 is a Wave64 collective: 64 lanes cooperate.  Because the per-lane
// fragment sizes differ (WMMA: <16 x half> / <8 x float>; MFMA: <4 x half> /
// <4 x float>), a simple intrinsic swap is impossible.
//
// The core problem is the wave-size mismatch.  When gfx1250 code is compiled
// for gfx942, the hardware groups 64 threads into one wavefront instead of 32.
// Threads 0-31 and 32-63 were in separate Wave32 wavefronts on the source
// architecture, computing different sub-tiles.  A single MFMA would mix their
// unrelated data.
//
// Register Layout Equations (from AMD Matrix Instruction Calculator)
// ------------------------------------------------------------------
//
// gfx12 (RDNA4) v_wmma_f32_16x16x{32,64}_{f16,bf16,fp8_*,bf8_*}, Wave32:
//
// All supported variants share the same per-Wave32-lane fragment shape:
// 8 VGPRs (= 32 bytes) per A side, 8 VGPRs per B side, 8 VGPRs of f32
// per C/D side. The K-dimension scales inversely with the element
// width (K=32 for 16-bit elements, K=64 for 8-bit elements), so the
// total bytes per lane stay constant. The lane redistribution math
// below operates on dwords (32-bit cells); it is therefore byte-
// identical across element widths -- the only per-variant divergence
// lives in (a) the MFMA intrinsic dispatched on the gfx942 side and
// (b) the per-MFMA bitcast / pack type. See `runGroupPass` and the
// `WMMAInputType` enum in `wmma-lowering.h` for the full enumeration.
//
//   A input -- 16-bit variants (8 VGPRs, <16 x {half|bfloat}>):
//     i = lane % 16
//     k = 8*floor(GPR/2) + 4*floor(lane/16) + 2*(GPR%2) + floor(bits/16)
//
//     Per-lane breakdown:
//       Lanes 0-15:   GPR 0->k={0,1}  GPR 1->k={2,3}  GPR 2->k={8,9}
//                     GPR 3->k={10,11} GPR 4->k={16,17} GPR 5->k={18,19}
//                     GPR 6->k={24,25} GPR 7->k={26,27}
//       Lanes 16-31:  GPR 0->k={4,5}  GPR 1->k={6,7}  GPR 2->k={12,13}
//                     GPR 3->k={14,15} GPR 4->k={20,21} GPR 5->k={22,23}
//                     GPR 6->k={28,29} GPR 7->k={30,31}
//
//   A input -- 8-bit variants (8 VGPRs, <8 x i32> = 32 packed fp8/bf8):
//     Same dword-grain layout as the 16-bit variants -- a fp8/bf8 byte
//     occupies the same byte slot inside its containing dword and
//     across lanes/GPRs that the corresponding 16-bit element would
//     have occupied. The K-stride doubles (each dword holds 4 fp8/bf8
//     bytes vs 2 halves) so the per-GPR K-range is twice as wide, but
//     the redistribution acts at dword granularity and does not see
//     the element-level interpretation.
//
//   C/D output (8 VGPRs, <8 x float>) -- invariant across variants:
//     i = 8*floor(lane/16) + GPR
//     j = lane % 16
//     -> Lanes 0-15: rows 0-7;  Lanes 16-31: rows 8-15
//
// gfx942 (CDNA3) MFMA targets:
//
//   16-bit MFMA (v_mfma_f32_16x16x16_{f16|bf16_1k}, K=16 per call):
//     A input (2 VGPRs, <4 x {half|i16}>):
//       i = lane % 16
//       k = 4*floor(lane/16) + 2*GPR + floor(bits/16)
//
//   8-bit MFMA (v_mfma_f32_16x16x32_{fp8|bf8}_{fp8|bf8}, K=32 per call):
//     A input (2 VGPRs, packed as i64 = 8 fp8/bf8 bytes):
//       Same per-lane VGPR width (2 dwords) as the 16-bit MFMA. The
//       K-fanout doubles to match the doubled WMMA K-range.
//
//   C/D output (4 VGPRs, <4 x float>) -- invariant across variants:
//     i = 4*floor(lane/16) + (GPR % 4)
//     j = lane % 16
//     -> Lanes 0-15: rows 0-3; 16-31: rows 4-7; 32-47: rows 8-11; 48-63: rows
//     12-15
//
// Approach
// --------
// We process each "virtual Wave32 group" (lanes 0-31 and 32-63) in a separate
// pass.  For each group:
//
//   1. REDISTRIBUTE: Use ds_bpermute to move WMMA fragments into the MFMA
//      layout.  The mapping is NOT a simple lane/2 -- it must account for the
//      interleaved k-distribution between lanes 0-15 and 16-31 in gfx12 WMMA,
//      and the 4-way lane-group distribution in gfx942 MFMA.
//
//      For each MFMA GPR, we read from the correct WMMA GPR and lane using
//      a 4-way select based on the MFMA lane group (floor(laneId/16)):
//
//        Lane group 0 (lanes 0-15):  reads from WMMA lower half (lanes 0-15)
//        Lane group 1 (lanes 16-31): reads from WMMA upper half (lanes 16-31)
//        Lane group 2 (lanes 32-47): reads from WMMA lower half
//        Lane group 3 (lanes 48-63): reads from WMMA upper half
//
//      With WMMA GPR selection cycling every 2 lane groups (GPR pairs {0,1},
//      {2,3} for first K=16; {4,5}, {6,7} for second K=16).
//
//   2. MFMA: Two v_mfma_f32_16x16x16_f16 calls (K=32 decomposed to 2x K=16),
//      chaining the accumulator.
//
//   3. COLLECT: Gather the 4-VGPR MFMA result back to 8-VGPR WMMA layout.
//      WMMA GPR_w reads from MFMA GPR (GPR_w % 4), lane computed as:
//        srcLane = 32*(w32Lane >= 16) + 16*(GPR_w >= 4) + (w32Lane & 15)
//
// After both passes, a lane-ID-based select picks the correct group's result.
//
// Partial-wave correctness and hardware EXEC
// -------------------------------------------
// The redistribute / MFMA / collect pipeline is semantically a Wave64
// collective: every MFMA input and every collect-time bpermute source
// is physically stored in SOME lane of the Wave64, and each destination
// lane's VGPR must hold the correct value for the next stage to read
// it back.
//
// `ds_bpermute` and `v_mfma_*` both READ all 64 source lanes regardless
// of EXEC -- but the WRITE of their per-lane result is EXEC-gated.  So
// a lane with EXEC=0 silently skips updating its destination VGPR, and
// any later cross-lane read of that VGPR returns stale / poison data.
//
// This is invisible when the kernel is launched with a blockDim that
// fills an entire Wave64 (every lane is active; MFMA inputs / outputs
// are written everywhere).  It manifests as a catastrophic correctness
// failure on partial-wave launches -- e.g. a Wave32 WMMA kernel
// launched with blockDim == 32 runs as a single Wave64 with EXEC =
// 0x0000_0000_FFFF_FFFF on gfx942.  Lanes 32-63 never update their
// mfmaA/B/C VGPRs, so MFMA reads garbage for k=2,3 (for the K=4 f32
// path) or for the entire upper k-half (for the K=32/K=64 path), and
// the collect-stage bpermute's reads from lanes 32-63 of the MFMA
// output return garbage too.  Rows 8-15 of the output come out as
// undefined / zero, and rows 0-7 get only a partial K-accumulation.
//
// The fix lives OUTSIDE this file, at the transpiler's kernel-entry
// plumbing: `WaveNativeProjection::emitInitialExec` (in
// `wave-projection.cpp`) emits `@llvm.amdgcn.init_whole_wave` at the
// very top of the lifted kernel, which (a) sets hardware EXEC = -1 for
// the remainder of the kernel and (b) captures the original per-lane
// active mask into the transpiler's EXEC alloca. Every VGPR write,
// memory store, LDS op, and atomic in the lifted IR already routes
// through `RaiseContext::emitUnderExec`, which reads the alloca-backed
// source EXEC and emits an `if (lane_active)` diamond -- the AMDGPU
// backend lowers those divergent branches by setting hardware EXEC
// to the ballot of the per-lane predicate inside each `do` block and
// restoring to EXEC = -1 afterwards. So between `emitUnderExec`
// diamonds (which is where the bpermute / MFMA / select chain here
// lives) hardware EXEC is -1, and all 64 lanes participate in the
// Wave64 collective exactly as required.
//
// This supersedes an earlier design that wrapped MFMA-output dwords in
// `@llvm.amdgcn.strict.wwm`. That design was semantically correct but
// unscalable: `SIPreAllocateWWMRegs` requires a DEDICATED physical
// VGPR per virtual register defined inside a WWM bracket, and the
// WWM def-chain from an MFMA output walks back through the entire
// accumulator initialisation. A 128x128 f16 matmul tile's entry
// region contains ~200 IMPLICIT_DEF / AV_MOV_B32 0 instructions for
// its accumulator ring, which together with the kernel's own VGPR
// demand exceeds gfx942's 256-VGPR pool and aborts the allocator
// with `physreg not found for WWM expression`. Moving the EXEC = -1
// guarantee to kernel entry sidesteps the allocator pressure entirely
// -- no intermediate vreg is ever "inside WWM" and regalloc is
// ordinary -- while preserving the partial-wave correctness property.
//
// From the perspective of this file, that means the redistribute +
// MFMA + collect chain emits ONLY ordinary IR (bpermute, bitcast,
// select, MFMA intrinsic) with no WWM markers. The hardware EXEC = -1
// invariant is the kernel-wide ambient set up by
// `WaveNativeProjection::emitInitialExec`, and the `writeRegVec` call
// in `handle-valu-vop3p.cpp` that consumes this file's return value
// takes care of gating the Wave32-layout destination VGPRs back to
// the original per-lane active mask.
//
// ============================================================================

#include "wmma-lowering.h"
#include "raise-context.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace COMGR::hotswap {

static Value *emitDSBpermute(IRBuilder<> &B, Module &M, Value *ByteOffset,
                             Value *SrcVal) {
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_ds_bpermute);
  return B.CreateCall(Fn, {ByteOffset, SrcVal}, "bperm");
}

static Value *emitLaneId(IRBuilder<> &B, Module &M, Type *I32Ty) {
  Function *MbcntLo =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_mbcnt_lo);
  Function *MbcntHi =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_mbcnt_hi);
  Value *AllOnes = ConstantInt::getSigned(I32Ty, -1);
  Value *Zero = ConstantInt::get(I32Ty, 0);
  Value *Lo = B.CreateCall(MbcntLo, {AllOnes, Zero}, "lane_lo");
  return B.CreateCall(MbcntHi, {AllOnes, Lo}, "lane_id");
}

static Value *packDwords(IRBuilder<> &B, Value **Dwords, unsigned NDwords,
                         Type *I32Ty, Type *TargetTy) {
  auto *VecTy = FixedVectorType::get(I32Ty, NDwords);
  Value *Vec = PoisonValue::get(VecTy);
  for (unsigned I = 0; I < NDwords; ++I)
    Vec = B.CreateInsertElement(Vec, Dwords[I], I, "pack");
  return B.CreateBitCast(Vec, TargetTy, "cast");
}

static void unpackDwords(IRBuilder<> &B, Value *Vec, unsigned NDwords,
                         Type *I32Ty, Value **Out) {
  auto *VecTy = FixedVectorType::get(I32Ty, NDwords);
  Value *AsI32 = B.CreateBitCast(Vec, VecTy, "toi32");
  for (unsigned I = 0; I < NDwords; ++I)
    Out[I] = B.CreateExtractElement(AsI32, I, "dw");
}

/// 4-way select based on lane group index (0..3).
/// Returns vals[laneGroup] for each lane.
static Value *selectByLaneGroup(IRBuilder<> &B, Value *LaneGroup, Value *V0,
                                Value *V1, Value *V2, Value *V3) {
  Value *S = V3;
  S = B.CreateSelect(B.CreateICmpEQ(LaneGroup, B.getInt32(2)), V2, S);
  S = B.CreateSelect(B.CreateICmpEQ(LaneGroup, B.getInt32(1)), V1, S);
  S = B.CreateSelect(B.CreateICmpEQ(LaneGroup, B.getInt32(0)), V0, S);
  return S;
}

/// Redistribute A or B input from gfx12 WMMA layout (8 VGPRs, Wave32)
/// to gfx942 MFMA layout (2 VGPRs x 2 passes, Wave64).
///
/// Source layout (A-Matrix; B-Matrix uses the same table with
/// M (rows) replaced by N (columns)):
///
///   Lanes 0-15  hold row M = L%16, K=0..7 in VGPR 0-3 (2 K's per dword,
///                                  packed lo/hi), K=16..23 in VGPR 4-7.
///   Lanes 16-31 hold row M = L%16, K=8..15 in VGPR 0-3, K=24..31 in
///                                  VGPR 4-7.
///
/// So lower wave32 half = K[3]==0 subset, upper half = K[3]==1 subset;
/// each VGPR-pair within a half covers 4 consecutive K values.
///
/// Target gfx942 MFMA (K=16 BF16/F16) per matrix calculator -- for both
/// A and B, 2 VGPRs per lane, 4 K-values per VGPR-pair:
///
///   LG 0 (lanes 0-15)  hold K=0..3,  with VGPR 0 = K{0,1}, VGPR 1 = K{2,3}
///   LG 1 (lanes 16-31) hold K=4..7
///   LG 2 (lanes 32-47) hold K=8..11
///   LG 3 (lanes 48-63) hold K=12..15
///
/// The second MFMA pass covers K=16..31 with the same LG slicing.
///
/// Source-(lane,VGPR) for MFMA dword G in pass p (p=0 -> K=0..15,
/// p=1 -> K=16..31):
///   src_w32_lane = (lane%16) + 16*(LG/2)        // lo half if LG<2
///   src_w32_vgpr = 4*p + 2*(LG%2) + G            // VGPR pair {0,1} or {2,3}
static void redistributeInput(IRBuilder<> &B, Module &M, Value **WmmaGpRs,
                              Value *AddrLo, Value *AddrHi, Value *LaneGroup,
                              Value **MfmaLo, Value **MfmaHi) {
  for (unsigned G = 0; G < 2; ++G) {
    Value *V0 = emitDSBpermute(B, M, AddrLo, WmmaGpRs[G]);     // LG0: lo, G
    Value *V1 = emitDSBpermute(B, M, AddrLo, WmmaGpRs[G + 2]); // LG1: lo, G+2
    Value *V2 = emitDSBpermute(B, M, AddrHi, WmmaGpRs[G]);     // LG2: hi, G
    Value *V3 = emitDSBpermute(B, M, AddrHi, WmmaGpRs[G + 2]); // LG3: hi, G+2
    MfmaLo[G] = selectByLaneGroup(B, LaneGroup, V0, V1, V2, V3);

    Value *U0 = emitDSBpermute(B, M, AddrLo, WmmaGpRs[G + 4]); // LG0: lo, G+4
    Value *U1 = emitDSBpermute(B, M, AddrLo, WmmaGpRs[G + 6]); // LG1: lo, G+6
    Value *U2 = emitDSBpermute(B, M, AddrHi, WmmaGpRs[G + 4]); // LG2: hi, G+4
    Value *U3 = emitDSBpermute(B, M, AddrHi, WmmaGpRs[G + 6]); // LG3: hi, G+6
    MfmaHi[G] = selectByLaneGroup(B, LaneGroup, U0, U1, U2, U3);
  }
}

/// Redistribute accumulator C from gfx12 WMMA layout (8 VGPRs, Wave32)
/// to gfx942 MFMA layout (4 VGPRs, Wave64).
///
/// gfx12 WMMA: i = 8*floor(lane/16) + GPR  ->  rows 0-7 in lanes 0-15, 8-15 in
/// lanes 16-31 gfx942 MFMA: i = 4*floor(lane/16) + GPR ->  rows 0-3 in LG0, 4-7
/// in LG1, 8-11 in LG2, 12-15 in LG3
///
/// MFMA GPR g needs:
///   LG 0: i = g      -> WMMA GPR g,   lower W32 half
///   LG 1: i = 4+g    -> WMMA GPR 4+g, lower W32 half
///   LG 2: i = 8+g    -> WMMA GPR g,   upper W32 half
///   LG 3: i = 12+g   -> WMMA GPR 4+g, upper W32 half
static void redistributeAcc(IRBuilder<> &B, Module &M, Value **CDwords,
                            Value *AddrLo, Value *AddrHi, Value *LaneGroup,
                            Value **MfmaC) {
  for (unsigned G = 0; G < 4; ++G) {
    Value *V0 = emitDSBpermute(B, M, AddrLo, CDwords[G]);
    Value *V1 = emitDSBpermute(B, M, AddrLo, CDwords[G + 4]);
    Value *V2 = emitDSBpermute(B, M, AddrHi, CDwords[G]);
    Value *V3 = emitDSBpermute(B, M, AddrHi, CDwords[G + 4]);
    MfmaC[G] = selectByLaneGroup(B, LaneGroup, V0, V1, V2, V3);
  }
}

/// Collect MFMA result (4 VGPRs, Wave64) back to WMMA layout (8 VGPRs, Wave32).
///
/// WMMA GPR_w reads MFMA GPR (GPR_w % 4) from:
///   srcLane = 32*(w32Lane >= 16) + 16*(GPR_w >= 4) + (w32Lane & 15)
static void collectResult(IRBuilder<> &B, Module &M, Value **MfmaDwords,
                          Value *W32Lane, Value **Out) {
  Value *W32Lo = B.CreateAnd(W32Lane, B.getInt32(15), "w32_lo");
  Value *IsUpper = B.CreateICmpUGE(W32Lane, B.getInt32(16), "is_upper");
  Value *UpperOff =
      B.CreateSelect(IsUpper, B.getInt32(32), B.getInt32(0), "upper_off");

  for (unsigned Gw = 0; Gw < 8; ++Gw) {
    Value *GprOff = B.getInt32((Gw >= 4) ? 16 : 0);
    Value *SrcLane =
        B.CreateAdd(B.CreateAdd(UpperOff, GprOff), W32Lo, "col_lane");
    Value *Addr = B.CreateShl(SrcLane, B.getInt32(2), "col_addr");
    Out[Gw] = emitDSBpermute(B, M, Addr, MfmaDwords[Gw % 4]);
  }
}

/// Run one full pass for a virtual Wave32 group:
/// redistribute -> 2x MFMA -> collect, wrapped in a single whole-wave
/// region so the cross-lane pipeline runs with EXEC = -1 regardless
/// of the caller-level EXEC mask (see file-header "Whole-wave mode").
///
/// \param groupBase  0 for group 0 (W64 lanes 0-31), 32 for group 1 (lanes
/// 32-63) \param inputType  selects MFMA intrinsic + per-MFMA pack/bitcast
/// type.
///                   The lane-redistribution math is element-type-agnostic
///                   across the entire WMMAInputType enumeration because
///                   every supported variant has the same per-Wave32-lane
///                   fragment size (8 VGPRs of A, 8 VGPRs of B, 8 VGPRs of
///                   f32 C/D) and the same K-decomposition factor (split
///                   into 2 chained MFMA calls per Wave32 group). The only
///                   per-variant divergence is the MFMA intrinsic name
///                   and the per-MFMA-call pack type:
///                     F16    -> mfma_f32_16x16x16f16,        <4 x half>
///                     BF16   -> mfma_f32_16x16x16bf16_1k,    <4 x i16>
///                     FP8_*  -> mfma_f32_16x16x32_<a>_<b>,   i64
///                     BF8_*  -> mfma_f32_16x16x32_<a>_<b>,   i64
///                   The bf16 -> i16 and fp8/bf8 -> i64 bitcasts are
///                   principled: the matching CDNA MFMA intrinsics were
///                   defined before the corresponding first-class LLVM
///                   types existed, and the storage containers are
///                   bit-for-bit identical.
static void runGroupPass(IRBuilder<> &B, Module &M, RaiseContext &Ctx,
                         unsigned GroupBase, Value *LaneId, Value **ADwords,
                         Value **BDwords, Value **CDwords,
                         WMMAInputType InputType, Value **ResultDwords) {
  Value *LaneMod16 = B.CreateAnd(LaneId, B.getInt32(15), "lane16");
  Value *LoLane = B.CreateAdd(LaneMod16, B.getInt32(GroupBase), "lo_lane");
  Value *HiLane = B.CreateAdd(LaneMod16, B.getInt32(GroupBase + 16), "hi_lane");
  Value *AddrLo = B.CreateShl(LoLane, B.getInt32(2), "addr_lo");
  Value *AddrHi = B.CreateShl(HiLane, B.getInt32(2), "addr_hi");
  Value *LaneGroup = B.CreateLShr(LaneId, B.getInt32(4), "lane_grp");

  Value *MfmaALo[2], *MfmaAHi[2];
  redistributeInput(B, M, ADwords, AddrLo, AddrHi, LaneGroup, MfmaALo, MfmaAHi);

  Value *MfmaBLo[2], *MfmaBHi[2];
  redistributeInput(B, M, BDwords, AddrLo, AddrHi, LaneGroup, MfmaBLo, MfmaBHi);

  Value *MfmaC[4];
  redistributeAcc(B, M, CDwords, AddrLo, AddrHi, LaneGroup, MfmaC);

  // Per-MFMA bitcast type and intrinsic dispatch -- the only point in
  // the lowering where the WMMA variants diverge.
  //
  // AB pack type:
  //   16-bit variants pack 2 redistributed dwords into a `<4 x t>` vector
  //   (4 elements per lane x 2 bytes = 8 bytes = 2 dwords). The 8-bit
  //   variants pack the same 2 dwords into a single `i64`; the CDNA fp8/
  //   bf8/i8 MFMA intrinsics were defined before fp8 became a first-class
  //   LLVM type (and there's no first-class packed-i8 vector type either),
  //   so they take 8 packed fp8/bf8/i8 bytes as i64 directly. Both packings
  //   are 64-bit and produced by the same 2-dword reduce (`packDwords`).
  //
  // Accumulator pack type:
  //   F32-accumulator MFMAs (everything except IU8) take `<4 x float>`.
  //   The IU8 path takes `<4 x i32>` to match the integer-accumulator
  //   `mfma_i32_16x16x32_i8` signature.
  Type *MfmaAbPackTy = nullptr;
  Type *MfmaAccPackTy = nullptr;
  Intrinsic::ID MfmaId;
  switch (InputType) {
  case WMMAInputType::F16:
    MfmaAbPackTy = FixedVectorType::get(Ctx.F16Ty, 4);
    MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
    MfmaId = Intrinsic::amdgcn_mfma_f32_16x16x16f16;
    break;
  case WMMAInputType::BF16:
    MfmaAbPackTy = FixedVectorType::get(Type::getInt16Ty(Ctx.C), 4);
    MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
    MfmaId = Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k;
    break;
  case WMMAInputType::FP8_FP8:
    MfmaAbPackTy = Ctx.I64Ty;
    MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
    MfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8;
    break;
  case WMMAInputType::FP8_BF8:
    MfmaAbPackTy = Ctx.I64Ty;
    MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
    MfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8;
    break;
  case WMMAInputType::BF8_FP8:
    MfmaAbPackTy = Ctx.I64Ty;
    MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
    MfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8;
    break;
  case WMMAInputType::BF8_BF8:
    MfmaAbPackTy = Ctx.I64Ty;
    MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
    MfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8;
    break;
  case WMMAInputType::IU8:
    MfmaAbPackTy = Ctx.I64Ty;
    MfmaAccPackTy = FixedVectorType::get(Ctx.I32Ty, 4);
    MfmaId = Intrinsic::amdgcn_mfma_i32_16x16x32_i8;
    break;
  }

  Value *SrcALo = packDwords(B, MfmaALo, 2, Ctx.I32Ty, MfmaAbPackTy);
  Value *SrcBLo = packDwords(B, MfmaBLo, 2, Ctx.I32Ty, MfmaAbPackTy);
  Value *Acc = packDwords(B, MfmaC, 4, Ctx.I32Ty, MfmaAccPackTy);

  Function *MfmaFn = Intrinsic::getOrInsertDeclaration(&M, MfmaId);
  Value *Cbsz = B.getInt32(0), *Abid = B.getInt32(0), *Blgp = B.getInt32(0);

  // MFMA is EXEC-gated on its WRITE: a lane with EXEC=0 skips
  // updating its destination VGPR.  Under `WaveNativeProjection` the
  // kernel-entry `init_whole_wave` keeps HW EXEC=-1 kernel-wide so
  // every lane writes its MFMA output and `wrapAsWWMValue` is a
  // no-op.  Under `ModuloReplicationProjection` (phantom-lane
  // fallback) HW EXEC = source-active-mask kernel-wide, so target
  // lanes 32..63 would never write their MFMA destination VGPR --
  // and the subsequent `collectResult` bpermute DOES read from
  // those target lanes (target lanes 16..31 pull rows 8..15 from
  // source lanes 32..47's MFMA output), so stale data would
  // corrupt output rows 8..15.  `wrapAsWWMValue` inserts a
  // `strict.wwm` marker on each MFMA output under MODREP, telling
  // the AMDGPU backend's `SIWholeQuadMode` pass to mark the MFMA
  // itself as WWM and emit `s_or_saveexec_b64 sN, -1` / `s_mov_b64
  // exec, sN` around it so every lane writes its output.
  //
  // We wrap the MFMA outputs specifically (not just the final collect
  // results) because SIWholeQuadMode's backward-propagation from a
  // `strict.wwm` on a later `ds_bpermute` result stops at the
  // bpermute boundary -- the backend sees the bpermute reads source
  // lanes 0..W_src-1 by address and concludes the MFMA output on
  // lanes W_src..2*W_src-1 is "not consumed", which is correct for
  // a single-pass MFMA but wrong for our cross-widening lowering
  // where the collect bpermute DOES pull from those upper-half
  // lanes to assemble rows 8..15.  Wrapping the MFMA result directly
  // forces the MFMA into the WWM backward slice.
  Value *Mfma1 = Ctx.Projection.wrapAsWWMValue(
      B, B.CreateCall(MfmaFn, {SrcALo, SrcBLo, Acc, Cbsz, Abid, Blgp}, "mfma1"),
      "mfma1_wwm");

  Value *SrcAHi = packDwords(B, MfmaAHi, 2, Ctx.I32Ty, MfmaAbPackTy);
  Value *SrcBHi = packDwords(B, MfmaBHi, 2, Ctx.I32Ty, MfmaAbPackTy);

  Value *Mfma2 = Ctx.Projection.wrapAsWWMValue(
      B,
      B.CreateCall(MfmaFn, {SrcAHi, SrcBHi, Mfma1, Cbsz, Abid, Blgp}, "mfma2"),
      "mfma2_wwm");

  Value *MfmaDst[4];
  unpackDwords(B, Mfma2, 4, Ctx.I32Ty, MfmaDst);

  Value *W32Lane = B.CreateAnd(LaneId, B.getInt32(31), "w32_lane");
  collectResult(B, M, MfmaDst, W32Lane, ResultDwords);

  // Also wrap each collect-stage output dword as WWM under MODREP,
  // so `SIWholeQuadMode` keeps the collect bpermutes in the WWM
  // region.  Without this, the bpermute WRITEs are EXEC-gated to
  // source-active lanes under normal HW EXEC, leaving target lanes
  // past the source active range with zeros (or stale values from
  // earlier instruction re-use of the same physical VGPR).  Target
  // lanes 16..31 in the wave32-source phantom-lane regime read
  // collect outputs from target lanes 32..47 / 48..63 to assemble
  // rows 8..15; those reads happen at ds_bpermute READ-time which
  // is un-gated, but the WRITE-back on target lanes 16..31 itself
  // is HW EXEC-gated and requires WWM for the chain to land the
  // correct per-lane value -- the bpermute-READ's result sits in
  // the target lane's register file only if that target lane is
  // HW-active at the bpermute WRITE, which under non-WWM MODREP is
  // the case for source-active lanes 0..31 but the destination
  // register allocation often aliases with values that were last
  // written in an earlier WWM region (where phantom lanes DID
  // write), so the stale phantom-lane content leaks through.
  // Wrapping the collect outputs forces the backend to treat the
  // bpermute writes as WWM, which clears the alias hazard.
  //
  // Under WaveNative this is an identity no-op.
  for (unsigned I = 0; I < 8; ++I)
    ResultDwords[I] =
        Ctx.Projection.wrapAsWWMValue(B, ResultDwords[I], "wmma_collect_wwm");
}

Expected<Value *> emitWMMAtoMFMA(RaiseContext &Ctx, Value *A, Value *Vb,
                                 Value *C, WMMAInputType InputType) {
  // The redistribute / 2xMFMA / collect chain is a Wave64 collective.
  // Per-MFMA-output `strict.wwm` markers inside `runGroupPass` handle
  // the two projections uniformly:
  //
  //   * WaveNativeProjection -- `init_whole_wave` at kernel entry
  //     already keeps HW EXEC=-1 kernel-wide.  `wrapAsWWMValue` is
  //     an identity no-op here to avoid the `SIPreAllocateWWMRegs`
  //     regalloc blow-up on large WMMA tiles (see
  //     `WaveProjection::emitInitialExec`'s block comment).
  //
  //   * ModuloReplicationProjection -- HW EXEC stays at the source-
  //     active mask kernel-wide; the per-MFMA `strict.wwm` tells
  //     the backend's `SIWholeQuadMode` pass to emit `s_or_saveexec
  //     _b64 sN, -1` / `s_mov_b64 exec, sN` around the MFMA and its
  //     redistribute inputs so every target lane writes its MFMA
  //     destination VGPR, which the subsequent collect bpermute
  //     then reads across the full Wave64.
  //
  // Number of passes depends on the projection's
  // `numSourceWavesPerTarget()`:
  //
  //   * 1 source wave per target (MODREP phantom-lane fallback):
  //     run only pass 0 (`groupBase=0`).  Pass 1 would pull data
  //     from source lanes W_src..2*W_src-1 which do not exist under
  //     phantom-lane, feeding undef into the MFMA cross-lane
  //     reduction.  Target lanes 32..63's output from pass 0 is
  //     well-defined (a correct MFMA group-0 output spread across
  //     the full Wave64) and harmlessly discarded by the post-lift
  //     `writeRegVec` `emitUnderExec`, whose HW-EXEC-gated store
  //     only fires on source-active lanes.
  //
  //   * 2 source waves per target (WaveNative cross-widen): run
  //     both passes; each source wave's WMMA maps to one pass.
  //     Target lanes 0..31 get pass-0's output (source wave 0's
  //     matmul), target lanes 32..63 get pass-1's (source wave 1's
  //     matmul), selected via a `laneId >= 32` comparison.
  IRBuilder<> &B = Ctx.B;
  Module &M = Ctx.M;

  Value *ADwords[8], *BDwords[8], *CDwords[8];
  unpackDwords(B, A, 8, Ctx.I32Ty, ADwords);
  unpackDwords(B, Vb, 8, Ctx.I32Ty, BDwords);
  unpackDwords(B, C, 8, Ctx.I32Ty, CDwords);

  Value *LaneId = emitLaneId(B, M, Ctx.I32Ty);

  const unsigned NumSrcWaves = Ctx.Projection.numSourceWavesPerTarget();
  if (NumSrcWaves != 1 && NumSrcWaves != 2)
    return createStringError(
        "WMMA->MFMA lowering defined only for wave32 source projections; "
        "numSourceWavesPerTarget() must be 1 (MODREP phantom-lane) or 2 "
        "(WaveNative cross-widen) -- a new projection class must declare "
        "which applies.");
  // Refusal-gate-still-in-place staging guard.  The K=4 f32 and K=32/K=64
  // refusal arms in `handle-valu-vop3p.cpp` return before reaching this
  // helper when `!providesFullWaveExecInvariant()`, i.e. under any non-
  // WaveNative projection.  The `numSrcWaves == 1` branch below is
  // Both `numSrcWaves == 1` (MODREP, single-source-wave per target
  // wave, the phantom-lane regime for kernels with
  // max_flat_workgroup_size < targetWaveSize) and `numSrcWaves == 2`
  // (WaveNative, wave32->wave64 cross-widen) fall through to the
  // two-branch pass logic below.  A broader `numSrcWaves != 2`
  // refusal guard lived here through 2026-04-23 to keep the MODREP
  // arm behind the `handle-valu-vop3p.cpp` refusal gate; it was
  // removed once the gate was narrowed to the multi-WMMA-per-K-iter
  // regime only (matmul_fp16 still refuses; matmul_fp16_16x16 and
  // other single-WMMA kernels now reach this path correctly).  See
  // matrix-translation.md sec. 12.4.4 for the layout characterisation
  // and handle-valu-vop3p.cpp's K=32/K=64 diagnostic for the
  // surgical refusal criterion.

  Value *Result0[8];
  runGroupPass(B, M, Ctx, /*groupBase=*/0, LaneId, ADwords, BDwords, CDwords,
               InputType, Result0);

  Value *FinalDwords[8];
  if (NumSrcWaves == 1) {
    for (unsigned I = 0; I < 8; ++I)
      FinalDwords[I] = Result0[I];
  } else {
    Value *Result1[8];
    runGroupPass(B, M, Ctx, /*groupBase=*/32, LaneId, ADwords, BDwords, CDwords,
                 InputType, Result1);
    Value *IsGroup1 = B.CreateICmpUGE(LaneId, B.getInt32(32), "is_group1");
    for (unsigned I = 0; I < 8; ++I)
      FinalDwords[I] = B.CreateSelect(IsGroup1, Result1[I], Result0[I], "sel");
  }

  // Result element type matches the dispatched MFMA accumulator
  // type (i32 for the IU8 integer-accumulator variant, f32 for
  // everything else).  The per-pass output dwords are i32
  // throughout; only the final reassembly cares about the element
  // semantics.
  Type *ResultElemTy =
      (InputType == WMMAInputType::IU8) ? Ctx.I32Ty : Ctx.F32Ty;
  return packDwords(B, FinalDwords, 8, Ctx.I32Ty,
                    FixedVectorType::get(ResultElemTy, 8));
}

// ----------------------------------------------------------------------
// v_wmma_f32_16x16x4_f32 -> mfma_f32_16x16x4f32 lowering
// ----------------------------------------------------------------------
//
// Source (gfx1250 RDNA4, Wave32):
//   int_amdgcn_wmma_f32_16x16x4_f32 -- `<8 x f32>` = (..., <2 x f32> A,
//   ..., <2 x f32> B, ..., <8 x f32> C, ...)
//
// Target (gfx942 CDNA3, Wave64):
//   int_amdgcn_mfma_f32_16x16x4f32 -- `<4 x f32>` = (f32 A, f32 B,
//   <4 x f32> C, i32 cbsz, i32 abid, i32 blgp)
//
// Register-layout equations
// -------------------------
// Source WMMA (Wave32, per-lane fragment):
//   A/B  -- <2 x f32> (2 VGPRs):  i = lane%16,  k = 2*floor(lane/16) + GPR
//     Lanes 0-15 GPR 0->k=0, GPR 1->k=1
//     Lanes 16-31 GPR 0->k=2, GPR 1->k=3
//   C/D  -- <8 x f32> (8 VGPRs):  i = 8*floor(lane/16) + GPR,  j = lane%16
//     Lanes 0-15 -> rows 0-7;   Lanes 16-31 -> rows 8-15
//
// Target MFMA (Wave64, per-lane fragment):
//   A/B  -- f32 (1 VGPR):          i = lane%16,  k = floor(lane/16)
//     LG0 (lanes 0-15)  -> k=0
//     LG1 (lanes 16-31) -> k=1
//     LG2 (lanes 32-47) -> k=2
//     LG3 (lanes 48-63) -> k=3
//   C/D  -- <4 x f32> (4 VGPRs):   i = 4*floor(lane/16) + GPR, j = lane%16
//     (same layout equation as the K=32/K=64 MFMA family, so the C
//     redistribution + result collection helpers above are reused
//     verbatim.)
//
// Redistribution
// --------------
// Per-group pass (`groupBase in {0, 32}`): the W32-group-N's data is
// held by W64 lanes `[groupBase .. groupBase+31]`. Each MFMA call
// spreads ONE Wave32 group's 32-lane x 2-dword A across all 64 Wave64
// lanes:
//
//   loAddr = 4 * ((lane%16) + groupBase)       // source for k=0..1
//   hiAddr = 4 * ((lane%16) + groupBase + 16)  // source for k=2..3
//
//   LG0 (lanes 0-15,  k=0): bpermute(loAddr, aDwords[0])
//   LG1 (lanes 16-31, k=1): bpermute(loAddr, aDwords[1])
//   LG2 (lanes 32-47, k=2): bpermute(hiAddr, aDwords[0])
//   LG3 (lanes 48-63, k=3): bpermute(hiAddr, aDwords[1])
//
// All four reads deliver the full K=4 range for ONE virtual Wave32
// group, which matches the K=4 MFMA signature -- so there is exactly
// ONE MFMA call per group (not 2 chained as in the K=32/K=64 path).
//
// B redistribution mirrors A exactly (same layout equation). The C
// redistribution reuses `redistributeAcc` (same WMMA C layout) and
// the result collection reuses `collectResult` (same WMMA D layout).
//
// Hardware EXEC: the redistribute / MFMA / collect chain relies on
// the kernel-wide EXEC = -1 invariant set up by
// `WaveNativeProjection::emitInitialExec`, so no in-file WWM marker
// is needed.  See the file-header "Partial-wave correctness and
// hardware EXEC" section for the correctness argument.
static void runGroupPassF32K4(IRBuilder<> &B, Module &M, RaiseContext &Ctx,
                              unsigned GroupBase, Value *LaneId,
                              Value **ADwords, Value **BDwords, Value **CDwords,
                              Value **ResultDwords) {
  Value *LaneMod16 = B.CreateAnd(LaneId, B.getInt32(15), "lane16");
  Value *LoLane = B.CreateAdd(LaneMod16, B.getInt32(GroupBase), "lo_lane");
  Value *HiLane = B.CreateAdd(LaneMod16, B.getInt32(GroupBase + 16), "hi_lane");
  Value *AddrLo = B.CreateShl(LoLane, B.getInt32(2), "addr_lo");
  Value *AddrHi = B.CreateShl(HiLane, B.getInt32(2), "addr_hi");
  Value *LaneGroup = B.CreateLShr(LaneId, B.getInt32(4), "lane_grp");

  // A input: single-dword MFMA fragment, one bpermute per (lane-group,
  // GPR) combination. aDwords[0] carries k=0 and k=2; aDwords[1]
  // carries k=1 and k=3 (lower vs upper WMMA half selects the +16
  // lane offset).
  Value *ALG0 = emitDSBpermute(B, M, AddrLo, ADwords[0]);
  Value *ALG1 = emitDSBpermute(B, M, AddrLo, ADwords[1]);
  Value *ALG2 = emitDSBpermute(B, M, AddrHi, ADwords[0]);
  Value *ALG3 = emitDSBpermute(B, M, AddrHi, ADwords[1]);
  Value *MfmaAI32 = selectByLaneGroup(B, LaneGroup, ALG0, ALG1, ALG2, ALG3);

  Value *BLG0 = emitDSBpermute(B, M, AddrLo, BDwords[0]);
  Value *BLG1 = emitDSBpermute(B, M, AddrLo, BDwords[1]);
  Value *BLG2 = emitDSBpermute(B, M, AddrHi, BDwords[0]);
  Value *BLG3 = emitDSBpermute(B, M, AddrHi, BDwords[1]);
  Value *MfmaBI32 = selectByLaneGroup(B, LaneGroup, BLG0, BLG1, BLG2, BLG3);

  Value *MfmaC[4];
  redistributeAcc(B, M, CDwords, AddrLo, AddrHi, LaneGroup, MfmaC);

  // Pack per-lane MFMA operands. The signatures are:
  //   A:f32         (scalar dword, not packed)
  //   B:f32         (scalar dword, not packed)
  //   C:<4 x f32>
  // The redistribution produced i32 dwords; bitcast A/B to f32 and
  // pack C into `<4 x float>` via the existing helper (which also
  // bitcasts).
  auto *MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
  Value *MfmaA = B.CreateBitCast(MfmaAI32, Ctx.F32Ty, "mfma_a");
  Value *MfmaB = B.CreateBitCast(MfmaBI32, Ctx.F32Ty, "mfma_b");
  Value *Acc = packDwords(B, MfmaC, 4, Ctx.I32Ty, MfmaAccPackTy);

  // cbsz / abid / blgp are the per-matrix broadcast-and-shift
  // modifiers hard-coded to zero here; the corpus kernels emit the
  // MFMA equivalent with these defaulted, matching what gfx1250
  // WMMA surfaces for the failing kerneldex Tensile GEMMs.
  Value *Cbsz = B.getInt32(0), *Abid = B.getInt32(0), *Blgp = B.getInt32(0);

  Function *MfmaFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_mfma_f32_16x16x4f32);
  // See the K=32 / K=64 `runGroupPass` helper above for the full
  // rationale: under MODREP phantom-lane, the MFMA's destination
  // VGPR must be written on every target lane so the subsequent
  // `collectResult` bpermute reads real data from lanes 32..47's
  // output.  `wrapAsWWMValue` inserts a `strict.wwm` marker under
  // MODREP (so the backend's `SIWholeQuadMode` pulls the MFMA into
  // a WWM region) and is an identity no-op under WaveNative (whose
  // kernel-entry `init_whole_wave` already keeps HW EXEC=-1).
  Value *Mfma = Ctx.Projection.wrapAsWWMValue(
      B, B.CreateCall(MfmaFn, {MfmaA, MfmaB, Acc, Cbsz, Abid, Blgp}, "mfma"),
      "mfma_wwm");

  Value *MfmaDst[4];
  unpackDwords(B, Mfma, 4, Ctx.I32Ty, MfmaDst);

  Value *W32Lane = B.CreateAnd(LaneId, B.getInt32(31), "w32_lane");
  collectResult(B, M, MfmaDst, W32Lane, ResultDwords);

  // Wrap collect outputs as WWM under MODREP -- see the equivalent
  // block comment in the K=32 / K=64 `runGroupPass` for the full
  // rationale.
  for (unsigned I = 0; I < 8; ++I)
    ResultDwords[I] =
        Ctx.Projection.wrapAsWWMValue(B, ResultDwords[I], "wmma_collect_wwm");
}

Expected<Value *> emitWmmAtoMfmaF3216x16x4(RaiseContext &Ctx, Value *A,
                                           Value *Vb, Value *C) {
  // K=4 f32 counterpart to `emitWMMAtoMFMA` above -- see that
  // function's block comment for the full design rationale
  // (projection-aware per-MFMA `strict.wwm` wrapping via
  // `wrapAsWWMValue`, and pass-1-skipped under MODREP phantom-
  // lane).  The only structural difference is that the K=4 f32
  // decomposition emits ONE MFMA per group pass (the source WMMA
  // is already K=4, exactly matching `mfma_f32_16x16x4f32`), not
  // the 2-chained-MFMA K=32->2xK=16 structure of the 16-/8-bit
  // family.
  IRBuilder<> &B = Ctx.B;
  Module &M = Ctx.M;

  Value *ADwords[2], *BDwords[2], *CDwords[8];
  unpackDwords(B, A, 2, Ctx.I32Ty, ADwords);
  unpackDwords(B, Vb, 2, Ctx.I32Ty, BDwords);
  unpackDwords(B, C, 8, Ctx.I32Ty, CDwords);

  Value *LaneId = emitLaneId(B, M, Ctx.I32Ty);

  const unsigned NumSrcWaves = Ctx.Projection.numSourceWavesPerTarget();
  if (NumSrcWaves != 1 && NumSrcWaves != 2)
    return createStringError(
        "WMMA->MFMA lowering defined only for wave32 source projections; "
        "numSourceWavesPerTarget() must be 1 (MODREP phantom-lane) or 2 "
        "(WaveNative cross-widen) -- a new projection class must declare "
        "which applies.");
  // Mirrors `emitWMMAtoMFMA` above: both MODREP (numSrcWaves==1) and
  // WaveNative (numSrcWaves==2) fall through to the two-branch pass
  // logic below; the old staging refusal was lifted once the
  // dispatcher's refusal gate was narrowed to the multi-WMMA-per-
  // K-iter pattern (see `handle-valu-vop3p.cpp`).

  Value *Result0[8];
  runGroupPassF32K4(B, M, Ctx, /*groupBase=*/0, LaneId, ADwords, BDwords,
                    CDwords, Result0);

  Value *FinalDwords[8];
  if (NumSrcWaves == 1) {
    for (unsigned I = 0; I < 8; ++I)
      FinalDwords[I] = Result0[I];
  } else {
    Value *Result1[8];
    runGroupPassF32K4(B, M, Ctx, /*groupBase=*/32, LaneId, ADwords, BDwords,
                      CDwords, Result1);
    Value *IsGroup1 = B.CreateICmpUGE(LaneId, B.getInt32(32), "is_group1");
    for (unsigned I = 0; I < 8; ++I)
      FinalDwords[I] = B.CreateSelect(IsGroup1, Result1[I], Result0[I], "sel");
  }

  return packDwords(B, FinalDwords, 8, Ctx.I32Ty,
                    FixedVectorType::get(Ctx.F32Ty, 8));
}

// ----------------------------------------------------------------------
// v_wmma_scale_f32_16x16x128_f8f6f4 -> v_mfma_scale_f32_16x16x128_f8f6f4
// ----------------------------------------------------------------------
//
// Source (gfx1250 RDNA4, Wave32, VOP3PX2):
//   int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4 -- 14-arg intrinsic.
//   Per Wave32 lane:
//     A: <aDwords x i32>   (aDwords = 16/12/8 for f8/f6/f4)
//     B: <bDwords x i32>   (same encoding)
//     C/D: <8 x f32>        (same as the K=32/K=64 WMMA family)
//
// Target (gfx950 CDNA4, Wave64, VOP3PX):
//   int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 -- 9-arg intrinsic, overloaded
//   on AB type.  Per handle-mfma.cpp convention we declare it with a uniform
//   <8 x i32> A/B type (the widest case, F8) and let cbsz / blgp select the
//   active subset of dwords per the LLVM TableGen rule.
//   Per Wave64 lane:
//     A, B: <8 x i32>       (logical: 8/6/4 dwords for f8/f6/f4; upper
//                           dwords are poison and ignored by hardware
//                           when cbsz / blgp narrow the width)
//     C/D:  <4 x f32>
//
// K decomposition:
//   None.  gfx950 MFMA scaled F8F6F4 covers the full K=128 in one call, so
//   the lowering is structurally simpler than emitWMMAtoMFMA's K=32 -> 2 x
//   K=16 chain.  We emit ONE MFMA call per virtual-W32-group pass.
//
// Lane redistribution (LINEAR-K model, validated on gfx950 silicon across
// 20 mxfp_attn_fwd_kernel shapes in rocm-hotswap-testing; matrix-class
// prediction median 0.1% error, 769 mfma_scale sites per kernel):
//   Per virtual Wave32 group at groupBase in {0, 32}, the A/B fragments live
//   in W64 lanes [groupBase .. groupBase+31].  The MFMA per-LG K-quarter
//   mapping is:
// clang-format off
//     LG0 (W64 lanes 0-15):   K = 0..31     <- WMMA dwords [0..mfmaDw-1] in W32 lower half
//     LG1 (W64 lanes 16-31):  K = 32..63    <- WMMA dwords [mfmaDw..2*mfmaDw-1] in W32 lower half
//     LG2 (W64 lanes 32-47):  K = 64..95    <- WMMA dwords [0..mfmaDw-1] in W32 upper half
//     LG3 (W64 lanes 48-63):  K = 96..127   <- WMMA dwords [mfmaDw..2*mfmaDw-1] in W32 upper half
// clang-format on
//   (mfmaDw = wmmaDw / 2 = 8/6/4 for f8/f6/f4.)
//
//   This is the "linear-K halves" model -- WMMA's two-lane-half split holds
//   the two K-halves linearly (lower-K then upper-K), and within each lane
//   the dwords run linearly over the K-range.  Contrast with the K=32 / K=64
//   16-bit family's interleaved layout (documented in this file's header
//   block) which alternates k between the lane halves.  Without an explicit
//   K=128 F8F6F4 layout doc in the AMD Matrix Instruction Calculator we
//   choose the simpler linear model; if the hardware layout turns out to be
//   the interleaved variant we adjust the redistribution below to match
//   (a one-line change in the lambda).
//
// C_mod application:
//   gfx1250 WMMA carries an i16 immediate (0 = none, 1 = neg, 2 = abs,
//   3 = neg(abs)) that gates the C-input modifier.  gfx950 MFMA has no
//   equivalent argument, so we apply it as IR fneg / fabs on the redistributed
//   <4 x f32> C input *before* the MFMA call.  The cMod argument is required
//   to be a ConstantInt -- callers pass an immediate-derived value, and any
//   non-constant trips the cast<ConstantInt> assertion (matching the
//   discipline of the WMMA-side fast path which also assumes a constant).
//
// Scale operand layout:
//   The MFMA scaled intrinsic takes one i32 scale per side, plus a 2-bit
//   op_sel encoding scale-format selection.  We pass scaleSrc0 / scaleSrc1
//   through directly (each Wave64 lane reads its own scale value, which
//   under WaveNativeProjection is already in the right per-source-lane
//   layout) and conservatively set op_sel = 0 for both A and B.  The exact
//   WMMA matrix_*_scale + matrix_*_scale_fmt -> MFMA op_sel mapping is not
//   documented in our reference materials; passing 0 selects the default
//   bit-position-zero scale, which is correct for the common UE8M0 scale
//   format that the MXFP attention kernels in our corpus use.  TODO: refine
//   if a kernel emits a non-default scale_fmt.
//
// EXEC handling:
//   Same as emitWMMAtoMFMA -- the MFMA/bpermute chain runs under the
//   kernel-wide HW EXEC = -1 ambient set by WaveNativeProjection's
//   init_whole_wave; under MODREP the wrapAsWWMValue calls keep the chain
//   inside SIWholeQuadMode's WWM region.

llvm::Expected<llvm::Value *> emitWMMAScaleF8F6F4toScaledMFMA(
    RaiseContext &ctx, Value *a, Value *b, Value *c, Value *matrixAFmt,
    Value *matrixBFmt, Value *cMod, Value *matrixAScale, Value *matrixAScaleFmt,
    Value *scaleSrc0, Value *matrixBScale, Value *matrixBScaleFmt,
    Value *scaleSrc1, unsigned aDwords, unsigned bDwords) {
  IRBuilder<> &B = ctx.B;
  Module &M = ctx.M;

  // Accept only the documented WMMA F8F6F4 widths.  The MC-pseudo decoder
  // in handle-valu-vop3p.cpp already rejects everything else with
  // unsupportedInstructionForm; we re-check here so that any future extension
  // that changes those values trips loudly rather than silently producing a
  // mismatched lane redistribution.
  if (!(aDwords == 16 || aDwords == 12 || aDwords == 8) ||
      !(bDwords == 16 || bDwords == 12 || bDwords == 8))
    return createStringError("unexpected WMMA F8F6F4 fragment width");

  const unsigned mfmaADw = aDwords / 2;
  const unsigned mfmaBDw = bDwords / 2;

  // Unused parameters that the WMMA encoding carries but the MFMA
  // intrinsic has no slot for. Reference them here so the build doesn't
  // warn about unused params; consult the operand-mapping comment above
  // for the rationale.
  (void)matrixAScale;
  (void)matrixAScaleFmt;
  (void)matrixBScale;
  (void)matrixBScaleFmt;

  // The MFMA scaled intrinsic is overloaded on the AB vector type.  Match
  // handle-mfma.cpp's convention: declare with the widest case (<8 x i32>)
  // and let cbsz / blgp select the active subset.
  auto *mfmaABTy = FixedVectorType::get(ctx.I32Ty, 8);
  auto *mfmaAccTy = FixedVectorType::get(ctx.F32Ty, 4);

  // Unpack source vectors into per-lane dword arrays.
  SmallVector<Value *, 16> aDwordsArr(aDwords);
  SmallVector<Value *, 16> bDwordsArr(bDwords);
  Value *cDwords[8];
  unpackDwords(B, a, aDwords, ctx.I32Ty, aDwordsArr.data());
  unpackDwords(B, b, bDwords, ctx.I32Ty, bDwordsArr.data());
  unpackDwords(B, c, 8, ctx.I32Ty, cDwords);

  Value *laneId = emitLaneId(B, M, ctx.I32Ty);

  const unsigned numSrcWaves = ctx.Projection.numSourceWavesPerTarget();
  if (numSrcWaves != 1 && numSrcWaves != 2)
    return createStringError(
        "WMMA-scale F8F6F4 -> MFMA lowering defined only for wave32 source "
        "projections; numSourceWavesPerTarget() must be 1 (MODREP) or 2 "
        "(WaveNative cross-widen).");

  // Extract C_mod constant for compile-time fneg / fabs application.
  // Callers in handle-valu-vop3p.cpp construct cMod via
  // ConstantInt::get(...), so it is always a ConstantInt by construction;
  // cast<> here documents the precondition and crashes nicely (via the
  // assertion build) if a future caller violates it.
  int64_t cModVal = cast<ConstantInt>(cMod)->getZExtValue();

  // cbsz / blgp come straight from the WMMA matrix_*_fmt operands.
  // Both encodings use the same {FP8, BF8, FP6, BF6, FP4} enum from
  // SIDefines.h::MatrixFMT, so direct passthrough is correct as long as
  // the enum stays unified across WMMA and MFMA scaled families (it does
  // today; if it ever forks, this is the place to remap).
  Value *cbsz = matrixAFmt;
  Value *blgp = matrixBFmt;

  // Conservative op_sel = 0 for both sides.  See block comment for why.
  Value *opSelA = ConstantInt::get(ctx.I32Ty, 0);
  Value *opSelB = ConstantInt::get(ctx.I32Ty, 0);

  // Lane-redistribution lambda: take a Wave32-layout dword array (wmmaDw
  // dwords per source-wave32 lane) and emit the per-Wave64-lane MFMA dword
  // array (wmmaDw / 2 entries) under the linear-K assumption documented
  // above.
  auto redistributeF8F6F4Input = [&](ArrayRef<Value *> wmmaDwords,
                                     Value *addrLo, Value *addrHi,
                                     Value *laneGroup, unsigned mfmaDw,
                                     SmallVectorImpl<Value *> &mfmaOut) {
    mfmaOut.resize(mfmaDw);
    for (unsigned i = 0; i < mfmaDw; ++i) {
      // LG0: lower-W32-half, first-half of WMMA dwords
      // LG1: lower-W32-half, second-half
      // LG2: upper-W32-half, first-half
      // LG3: upper-W32-half, second-half
      Value *v0 = emitDSBpermute(B, M, addrLo, wmmaDwords[i]);
      Value *v1 = emitDSBpermute(B, M, addrLo, wmmaDwords[i + mfmaDw]);
      Value *v2 = emitDSBpermute(B, M, addrHi, wmmaDwords[i]);
      Value *v3 = emitDSBpermute(B, M, addrHi, wmmaDwords[i + mfmaDw]);
      mfmaOut[i] = selectByLaneGroup(B, laneGroup, v0, v1, v2, v3);
    }
  };

  // One pass per virtual W32 group: groupBase in {0, 32}.
  auto runScaledPass = [&](unsigned groupBase,
                           SmallVectorImpl<Value *> &resultDwords) {
    Value *laneMod16 = B.CreateAnd(laneId, B.getInt32(15), "lane16");
    Value *loLane = B.CreateAdd(laneMod16, B.getInt32(groupBase), "lo_lane");
    Value *hiLane =
        B.CreateAdd(laneMod16, B.getInt32(groupBase + 16), "hi_lane");
    Value *addrLo = B.CreateShl(loLane, B.getInt32(2), "addr_lo");
    Value *addrHi = B.CreateShl(hiLane, B.getInt32(2), "addr_hi");
    Value *laneGroup = B.CreateLShr(laneId, B.getInt32(4), "lane_grp");

    SmallVector<Value *, 8> mfmaA, mfmaB;
    redistributeF8F6F4Input(aDwordsArr, addrLo, addrHi, laneGroup, mfmaADw,
                            mfmaA);
    redistributeF8F6F4Input(bDwordsArr, addrLo, addrHi, laneGroup, mfmaBDw,
                            mfmaB);

    Value *mfmaC[4];
    redistributeAcc(B, M, cDwords, addrLo, addrHi, laneGroup, mfmaC);

    // Pad MFMA A / B fragments to <8 x i32> with poison in the unused
    // upper lanes for f6 / f4 (cbsz / blgp tell hardware which subset
    // is active).
    Value *aPacked = PoisonValue::get(mfmaABTy);
    for (unsigned i = 0; i < mfmaADw; ++i)
      aPacked = B.CreateInsertElement(aPacked, mfmaA[i], i, "a_pack");
    Value *bPacked = PoisonValue::get(mfmaABTy);
    for (unsigned i = 0; i < mfmaBDw; ++i)
      bPacked = B.CreateInsertElement(bPacked, mfmaB[i], i, "b_pack");

    Value *acc = packDwords(B, mfmaC, 4, ctx.I32Ty, mfmaAccTy);

    // Apply C_mod (neg / abs / neg(abs)) on the redistributed C before
    // MFMA.  The bits are: 0 = none, 1 = neg, 2 = abs, 3 = neg(abs).
    if (cModVal & 2)
      acc = B.CreateUnaryIntrinsic(Intrinsic::fabs, acc, nullptr, "c_abs");
    if (cModVal & 1)
      acc = B.CreateFNeg(acc, "c_neg");

    // Redistribute scale_src per pass.  Pinned by the position-sweep
    // bisection in compare_correctness/kernels/wmma_scale_f32_16x16x128_*
    // against salmon-raised gfx950 .co: passing scaleSrc0 / scaleSrc1
    // through unchanged worked for pass 0 but produced D[odd-M][:] =
    // 64*16 + 64*1 for ODD source-lane scales in pass 1, indicating
    // that the MFMA's per-lane scale read does not see the source-lane
    // value at the right wave64 position for pass 1's group.  We
    // bpermute scale_src0 / scale_src1 from `addrLo` so that, for both
    // passes, wave64 lane L (L in 0..15) reads the scale value the
    // source kernel wrote at source-wave32 lane L of this pass's
    // virtual W32 group.  Pass 0 redistributes from source lanes 0..15
    // (no-op effectively); pass 1 redistributes from source lanes
    // 32..47, which under MODREP carry source lanes 0..15's
    // replicated values.
    Value *scaleSrc0Pass = emitDSBpermute(B, M, addrLo, scaleSrc0);
    Value *scaleSrc1Pass = emitDSBpermute(B, M, addrLo, scaleSrc1);

    Function *mfmaFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4,
        {mfmaABTy, mfmaABTy});
    Value *mfmaResult = ctx.Projection.wrapAsWWMValue(
        B,
        B.CreateCall(mfmaFn,
                     {aPacked, bPacked, acc, cbsz, blgp, opSelA, scaleSrc0Pass,
                      opSelB, scaleSrc1Pass},
                     "mfma_scale"),
        "mfma_scale_wwm");

    Value *mfmaDst[4];
    unpackDwords(B, mfmaResult, 4, ctx.I32Ty, mfmaDst);

    Value *w32Lane = B.CreateAnd(laneId, B.getInt32(31), "w32_lane");
    resultDwords.resize(8);
    collectResult(B, M, mfmaDst, w32Lane, resultDwords.data());

    // Wrap collect outputs as WWM under MODREP for the same reason as
    // emitWMMAtoMFMA -- the bpermute writes are HW-EXEC-gated and need
    // SIWholeQuadMode to widen the active mask.  No-op under WaveNative.
    for (unsigned i = 0; i < 8; ++i)
      resultDwords[i] = ctx.Projection.wrapAsWWMValue(B, resultDwords[i],
                                                      "wmma_scale_collect_wwm");
  };

  SmallVector<Value *, 8> result0;
  runScaledPass(0, result0);

  Value *finalDwords[8];
  if (numSrcWaves == 1) {
    for (unsigned i = 0; i < 8; ++i)
      finalDwords[i] = result0[i];
  } else {
    SmallVector<Value *, 8> result1;
    runScaledPass(32, result1);
    Value *isGroup1 = B.CreateICmpUGE(laneId, B.getInt32(32), "is_group1");
    for (unsigned i = 0; i < 8; ++i)
      finalDwords[i] = B.CreateSelect(isGroup1, result1[i], result0[i], "sel");
  }

  return packDwords(B, finalDwords, 8, ctx.I32Ty,
                    FixedVectorType::get(ctx.F32Ty, 8));
}

// gfx1250 v_wmma_scale_f32_16x16x128_f8f6f4 -> gfx942: decompose K=128 into 4
// K=32 unscaled MFMAs and apply the per-block scale to each partial via
// fmuladd. FP6/BF6/FP4 widen in-line to FP8/BF8 so the pipeline is
// format-agnostic. MODREP runs one pass; WaveNative cross-widen runs two
// (GroupBase 0 and 32) combined by a laneId select.

namespace {

// MatrixFMT enum values (mirrors SIDefines.h).
constexpr int FmtFP8 = 0;
constexpr int FmtBF8 = 1;
constexpr int FmtFP6 = 2;
constexpr int FmtBF6 = 3;
constexpr int FmtFP4 = 4;

// gfx942 unscaled K=32 fp8/bf8 MFMA family. Caller must have mapped any
// FP6/BF6/FP4 input to FP8/BF8 via effectiveFmtAfterWiden first.
Intrinsic::ID pickGfx942F8MfmaIntrinsic(int aFmt, int bFmt) {
  const bool aIsFp8 = (aFmt == FmtFP8);
  const bool aIsBf8 = (aFmt == FmtBF8);
  const bool bIsFp8 = (bFmt == FmtFP8);
  const bool bIsBf8 = (bFmt == FmtBF8);
  if (aIsFp8 && bIsFp8)
    return Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8;
  if (aIsFp8 && bIsBf8)
    return Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8;
  if (aIsBf8 && bIsFp8)
    return Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8;
  if (aIsBf8 && bIsBf8)
    return Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8;
  return Intrinsic::not_intrinsic;
}

// Post-widen format: FP6/FP4 -> FP8, BF6 -> BF8, FP8/BF8 pass through.
// Returns -1 for unknown formats.
int effectiveFmtAfterWiden(int srcFmt) {
  switch (srcFmt) {
  case FmtFP8:
    return FmtFP8;
  case FmtBF8:
    return FmtBF8;
  case FmtFP6:
    return FmtFP8;
  case FmtBF6:
    return FmtBF8;
  case FmtFP4:
    return FmtFP8;
  default:
    return -1;
  }
}

// FP4 (E2M1, bias 1) -> FP8 E4M3 (bias 7) over `<N x i32>` of nibbles.
// FP4's only subnormal is +/-0.5 (biased exp 6); the +/-0 override keeps the
// subnormal path from emitting 0x30 / 0xB0.
Value *widenF4NibbleVecToFP8(IRBuilder<> &B, Value *Nibbles) {
  auto *VecTy = cast<FixedVectorType>(Nibbles->getType());
  unsigned N = VecTy->getNumElements();
  auto *I32Ty = cast<IntegerType>(VecTy->getElementType());
  auto Splat = [&](uint64_t Val) -> Constant * {
    return ConstantVector::getSplat(ElementCount::getFixed(N),
                                    ConstantInt::get(I32Ty, Val));
  };

  Value *Sign =
      B.CreateAnd(B.CreateLShr(Nibbles, Splat(3)), Splat(1), "fp4_sign");
  Value *Exp =
      B.CreateAnd(B.CreateLShr(Nibbles, Splat(1)), Splat(3), "fp4_exp");
  Value *Mant = B.CreateAnd(Nibbles, Splat(1), "fp4_mant");

  Value *NormExp = B.CreateAdd(Exp, Splat(6), "norm_exp");
  Value *NormMant = B.CreateShl(Mant, Splat(2), "norm_mant");

  Value *IsSubnormal = B.CreateICmpEQ(Exp, Splat(0), "is_subnormal");
  Value *FP8Exp = B.CreateSelect(IsSubnormal, Splat(6), NormExp, "fp8_exp");
  Value *FP8Mant = B.CreateSelect(IsSubnormal, Splat(0), NormMant, "fp8_mant");

  Value *SignShifted = B.CreateShl(Sign, Splat(7), "fp8_sign_pos");
  Value *FP8WithSign = B.CreateOr(
      SignShifted, B.CreateOr(B.CreateShl(FP8Exp, Splat(3)), FP8Mant), "fp8");

  Value *IsZero =
      B.CreateAnd(IsSubnormal, B.CreateICmpEQ(Mant, Splat(0)), "is_zero");
  return B.CreateSelect(IsZero, SignShifted, FP8WithSign, "fp8_or_zero");
}

// Widen a full FP4 fragment (8 dwords / 64 nibbles per wave32 lane) to an
// FP8 fragment (16 dwords). Element k -> byte k.
void widenF4FragmentToFP8(IRBuilder<> &B, Module & /*M*/,
                          ArrayRef<Value *> SrcDwords,
                          SmallVectorImpl<Value *> &DstDwords) {
  assert(SrcDwords.size() == 8 && "FP4 fragment is 8 i32 dwords / lane");
  DstDwords.assign(16, nullptr);

  auto *I32Ty = B.getInt32Ty();
  auto *Vec8I8 = FixedVectorType::get(B.getInt8Ty(), 8);
  auto *Vec2I32 = FixedVectorType::get(I32Ty, 2);

  // Shifts {0,4,...,28} to spread a dword's 8 nibbles across the lanes.
  SmallVector<Constant *, 8> ShiftAmtsElts;
  for (unsigned i = 0; i < 8; ++i)
    ShiftAmtsElts.push_back(ConstantInt::get(I32Ty, 4 * i));
  Constant *ShiftAmts = ConstantVector::get(ShiftAmtsElts);
  Constant *NibbleMask = ConstantVector::getSplat(ElementCount::getFixed(8),
                                                  ConstantInt::get(I32Ty, 0xF));

  for (unsigned d = 0; d < 8; ++d) {
    Value *DwSplat = B.CreateVectorSplat(8, SrcDwords[d], "fp4_splat");
    Value *Shifted = B.CreateLShr(DwSplat, ShiftAmts, "fp4_shifted");
    Value *Nibbles = B.CreateAnd(Shifted, NibbleMask, "fp4_nibs");

    Value *FP8Vec = widenF4NibbleVecToFP8(B, Nibbles);

    // Each result is a byte; trunc + bitcast packs 8 of them as 2 dwords.
    Value *FP8Bytes = B.CreateTrunc(FP8Vec, Vec8I8, "fp4_bytes");
    Value *DwordPair = B.CreateBitCast(FP8Bytes, Vec2I32, "fp4_dw_pair");

    DstDwords[2 * d] =
        B.CreateExtractElement(DwordPair, B.getInt32(0), "fp8_dw_lo");
    DstDwords[2 * d + 1] =
        B.CreateExtractElement(DwordPair, B.getInt32(1), "fp8_dw_hi");
  }
}

// Extract the k-th 6-bit element from a 12-dword wave32-lane fragment.
// 6-bit elements pack contiguously, so some straddle a dword boundary;
// those combine two dwords as an i64 before shift/mask/truncate.
Value *extractF6Nibble(IRBuilder<> &B, ArrayRef<Value *> Dwords, unsigned k) {
  const unsigned bitOffset = 6 * k;
  const unsigned dwIdx = bitOffset / 32;
  const unsigned bitInDw = bitOffset % 32;
  const bool crosses = (bitInDw + 6) > 32;

  if (!crosses) {
    Value *Shifted = B.CreateLShr(Dwords[dwIdx], B.getInt32(bitInDw), "f6_shr");
    return B.CreateAnd(Shifted, B.getInt32(0x3F), "f6_nib");
  }
  // Straddles a dword boundary: combine as i64, shift, mask, truncate.
  Type *I64Ty = B.getInt64Ty();
  Type *I32Ty = B.getInt32Ty();
  Value *Lo = B.CreateZExt(Dwords[dwIdx], I64Ty, "f6_lo");
  Value *Hi = B.CreateZExt(Dwords[dwIdx + 1], I64Ty, "f6_hi");
  Value *Combined = B.CreateOr(Lo, B.CreateShl(Hi, B.getInt64(32)), "f6_pair");
  Value *Shifted = B.CreateLShr(Combined, B.getInt64(bitInDw), "f6_shr64");
  Value *Trunc = B.CreateTrunc(Shifted, I32Ty, "f6_trunc");
  return B.CreateAnd(Trunc, B.getInt32(0x3F), "f6_nib");
}

// FP6 (E2M3, bias 1) -> FP8 E4M3 (bias 7) over `<N x i32>` of nibbles.
// Same mantissa width, exp bias diff 6; subnormals renormalize via ctlz.
Value *widenFP6NibbleVecToFP8(IRBuilder<> &B, Module &M, Value *Nibbles) {
  auto *VecTy = cast<FixedVectorType>(Nibbles->getType());
  unsigned N = VecTy->getNumElements();
  auto *I32Ty = cast<IntegerType>(VecTy->getElementType());
  auto Splat = [&](uint64_t Val) -> Constant * {
    return ConstantVector::getSplat(ElementCount::getFixed(N),
                                    ConstantInt::get(I32Ty, Val));
  };

  Value *Sign =
      B.CreateAnd(B.CreateLShr(Nibbles, Splat(5)), Splat(1), "fp6_sign");
  Value *Exp =
      B.CreateAnd(B.CreateLShr(Nibbles, Splat(3)), Splat(3), "fp6_exp");
  Value *Mant = B.CreateAnd(Nibbles, Splat(7), "fp6_mant");

  Value *NormExp = B.CreateAdd(Exp, Splat(6), "norm_exp");

  Function *CtlzFn =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ctlz, {VecTy});
  Value *Ctlz = B.CreateCall(CtlzFn, {Mant, B.getInt1(true)}, "fp6_ctlz");
  Value *LZ = B.CreateSub(Ctlz, Splat(29), "fp6_lz");
  Value *SubExp = B.CreateSub(Splat(6), LZ, "sub_exp");
  Value *MantShifted =
      B.CreateShl(Mant, B.CreateAdd(LZ, Splat(1)), "sub_mant_shl");
  Value *SubMant = B.CreateAnd(MantShifted, Splat(7), "sub_mant");

  Value *IsSubnormal = B.CreateICmpEQ(Exp, Splat(0), "is_sub");
  Value *FP8Exp = B.CreateSelect(IsSubnormal, SubExp, NormExp, "fp8_exp");
  Value *FP8Mant = B.CreateSelect(IsSubnormal, SubMant, Mant, "fp8_mant");

  Value *SignShifted = B.CreateShl(Sign, Splat(7), "fp8_sign_pos");
  Value *FP8 = B.CreateOr(
      SignShifted, B.CreateOr(B.CreateShl(FP8Exp, Splat(3)), FP8Mant), "fp8");

  Value *IsZero =
      B.CreateAnd(IsSubnormal, B.CreateICmpEQ(Mant, Splat(0)), "is_zero");
  return B.CreateSelect(IsZero, SignShifted, FP8, "fp8_or_zero");
}

// BF6 (E3M2, bias 3) -> BF8 E5M2 (bias 15) over `<N x i32>` of nibbles.
// Same mantissa width, exp bias diff 12; subnormals renormalize via ctlz.
Value *widenBF6NibbleVecToBF8(IRBuilder<> &B, Module &M, Value *Nibbles) {
  auto *VecTy = cast<FixedVectorType>(Nibbles->getType());
  unsigned N = VecTy->getNumElements();
  auto *I32Ty = cast<IntegerType>(VecTy->getElementType());
  auto Splat = [&](uint64_t Val) -> Constant * {
    return ConstantVector::getSplat(ElementCount::getFixed(N),
                                    ConstantInt::get(I32Ty, Val));
  };

  Value *Sign =
      B.CreateAnd(B.CreateLShr(Nibbles, Splat(5)), Splat(1), "bf6_sign");
  Value *Exp =
      B.CreateAnd(B.CreateLShr(Nibbles, Splat(2)), Splat(7), "bf6_exp");
  Value *Mant = B.CreateAnd(Nibbles, Splat(3), "bf6_mant");

  Value *NormExp = B.CreateAdd(Exp, Splat(12), "norm_exp");

  Function *CtlzFn =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ctlz, {VecTy});
  Value *Ctlz = B.CreateCall(CtlzFn, {Mant, B.getInt1(true)}, "bf6_ctlz");
  Value *LZ = B.CreateSub(Ctlz, Splat(30), "bf6_lz");
  Value *SubExp = B.CreateSub(Splat(12), LZ, "sub_exp");
  Value *MantShifted =
      B.CreateShl(Mant, B.CreateAdd(LZ, Splat(1)), "sub_mant_shl");
  Value *SubMant = B.CreateAnd(MantShifted, Splat(3), "sub_mant");

  Value *IsSubnormal = B.CreateICmpEQ(Exp, Splat(0), "is_sub");
  Value *BF8Exp = B.CreateSelect(IsSubnormal, SubExp, NormExp, "bf8_exp");
  Value *BF8Mant = B.CreateSelect(IsSubnormal, SubMant, Mant, "bf8_mant");

  // BF8 mantissa width 2 -> exp shifts by 2.
  Value *SignShifted = B.CreateShl(Sign, Splat(7), "bf8_sign_pos");
  Value *BF8 = B.CreateOr(
      SignShifted, B.CreateOr(B.CreateShl(BF8Exp, Splat(2)), BF8Mant), "bf8");

  Value *IsZero =
      B.CreateAnd(IsSubnormal, B.CreateICmpEQ(Mant, Splat(0)), "is_zero");
  return B.CreateSelect(IsZero, SignShifted, BF8, "bf8_or_zero");
}

// Widen a full FP6 fragment (12 dwords / 64 elements) to FP8 (16 dwords),
// vectorized over 4 chunks of 16 elements. Element k -> byte k.
void widenFP6FragmentToFP8(IRBuilder<> &B, Module &M,
                           ArrayRef<Value *> SrcDwords,
                           SmallVectorImpl<Value *> &DstDwords) {
  assert(SrcDwords.size() == 12 && "FP6 fragment is 12 i32 dwords / lane");
  auto *I32Ty = B.getInt32Ty();
  auto *Vec16I32 = FixedVectorType::get(I32Ty, 16);
  auto *Vec16I8 = FixedVectorType::get(B.getInt8Ty(), 16);
  auto *Vec4I32 = FixedVectorType::get(I32Ty, 4);
  DstDwords.assign(16, nullptr);

  for (unsigned s = 0; s < 4; ++s) {
    Value *NibbleVec = PoisonValue::get(Vec16I32);
    for (unsigned i = 0; i < 16; ++i) {
      Value *Nibble = extractF6Nibble(B, SrcDwords, 16 * s + i);
      NibbleVec =
          B.CreateInsertElement(NibbleVec, Nibble, B.getInt32(i), "f6_lane");
    }

    Value *FP8Vec = widenFP6NibbleVecToFP8(B, M, NibbleVec);

    Value *FP8Bytes = B.CreateTrunc(FP8Vec, Vec16I8, "f6_bytes");
    Value *DwordVec = B.CreateBitCast(FP8Bytes, Vec4I32, "f6_dwords");
    for (unsigned i = 0; i < 4; ++i)
      DstDwords[4 * s + i] =
          B.CreateExtractElement(DwordVec, B.getInt32(i), "f6_dw");
  }
}

// Counterpart of `widenFP6FragmentToFP8` for BF6 -> BF8.
void widenBF6FragmentToBF8(IRBuilder<> &B, Module &M,
                           ArrayRef<Value *> SrcDwords,
                           SmallVectorImpl<Value *> &DstDwords) {
  assert(SrcDwords.size() == 12 && "BF6 fragment is 12 i32 dwords / lane");
  auto *I32Ty = B.getInt32Ty();
  auto *Vec16I32 = FixedVectorType::get(I32Ty, 16);
  auto *Vec16I8 = FixedVectorType::get(B.getInt8Ty(), 16);
  auto *Vec4I32 = FixedVectorType::get(I32Ty, 4);
  DstDwords.assign(16, nullptr);

  for (unsigned s = 0; s < 4; ++s) {
    Value *NibbleVec = PoisonValue::get(Vec16I32);
    for (unsigned i = 0; i < 16; ++i) {
      Value *Nibble = extractF6Nibble(B, SrcDwords, 16 * s + i);
      NibbleVec =
          B.CreateInsertElement(NibbleVec, Nibble, B.getInt32(i), "f6_lane");
    }

    Value *BF8Vec = widenBF6NibbleVecToBF8(B, M, NibbleVec);

    Value *BF8Bytes = B.CreateTrunc(BF8Vec, Vec16I8, "f6_bytes");
    Value *DwordVec = B.CreateBitCast(BF8Bytes, Vec4I32, "f6_dwords");
    for (unsigned i = 0; i < 4; ++i)
      DstDwords[4 * s + i] =
          B.CreateExtractElement(DwordVec, B.getInt32(i), "f6_dw");
  }
}

// matrix_*_scale_fmt selector values.
constexpr int ScaleFmtE8M0 = 0;
constexpr int ScaleFmtE5M3 = 1;
constexpr int ScaleFmtE4M3 = 2;

// Byte k of the 4-byte i32 scale source (ROW0 form; ROW1 gated by caller).
Value *extractScaleByte(IRBuilder<> &B, Value *Scale32, unsigned k) {
  Value *Shifted = B.CreateLShr(Scale32, B.getInt32(8 * k), "scale_shr");
  return B.CreateAnd(Shifted, B.getInt32(0xFF), "scale_byte");
}

// Decode one scale byte to f32. E8M0 via `ldexp(1.0, byte - 127)` with
// 0xFF -> qNaN. E4M3 via hw `cvt_f32_fp8` (same bit layout as the FP8
// E4M3 data format, including NaN). E5M3 not yet implemented.
Value *decodeScaleByte(IRBuilder<> &B, Module &M, Type *F32Ty, Value *Byte,
                       int Fmt) {
  switch (Fmt) {
  case ScaleFmtE8M0: {
    Value *IsNaN = B.CreateICmpEQ(Byte, B.getInt32(0xFF), "e8m0_is_nan");
    Value *Biased = B.CreateSub(Byte, B.getInt32(127), "e8m0_exp");
    Function *LdexpFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::ldexp, {F32Ty, B.getInt32Ty()});
    Value *Finite = B.CreateCall(LdexpFn, {ConstantFP::get(F32Ty, 1.0), Biased},
                                 "e8m0_finite");
    return B.CreateSelect(IsNaN, ConstantFP::getQNaN(F32Ty), Finite,
                          "e8m0_decoded");
  }
  case ScaleFmtE4M3: {
    // Scale format 2 is UE4M3 (unsigned). cvt.f32.fp8 decodes signed E4M3,
    // so mask the sign bit; byte 0x7F still decodes as +NaN, matching UE4M3.
    Value *Masked = B.CreateAnd(Byte, B.getInt32(0x7F), "ue4m3_byte_unsigned");
    Function *CvtFn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_cvt_f32_fp8);
    return B.CreateCall(CvtFn, {Masked, B.getInt32(0)}, "ue4m3_decoded");
  }
  case ScaleFmtE5M3:
    return nullptr; // TODO
  }
  return nullptr;
}

// Spec legal-combination rules: a non-E8M0 scale requires F4 data on that
// side, and F4 x F4 with non-E8M0 scales requires matching scale formats.
bool isLegalScaleDataCombo(int aFmt, int aScaleFmt, int bFmt, int bScaleFmt) {
  if (aScaleFmt != ScaleFmtE8M0 && aFmt != FmtFP4)
    return false;
  if (bScaleFmt != ScaleFmtE8M0 && bFmt != FmtFP4)
    return false;
  if (aFmt == FmtFP4 && bFmt == FmtFP4 && aScaleFmt != bScaleFmt)
    return false;
  return true;
}

// `<4 x float>` splat of factor_A * factor_B. E8M0 x E8M0 uses the
// combined-exponent shortcut 2^(byteA + byteB - 254); other combinations
// decode each side and fmul.
Value *buildScaleFactorVec(IRBuilder<> &B, Module &M, Type *F32Ty,
                           Value *ScaleAByte, Value *ScaleBByte, int AScaleFmt,
                           int BScaleFmt) {
  Value *Factor = nullptr;

  if (AScaleFmt == ScaleFmtE8M0 && BScaleFmt == ScaleFmtE8M0) {
    Value *AIsNaN =
        B.CreateICmpEQ(ScaleAByte, B.getInt32(0xFF), "scale_a_is_nan");
    Value *BIsNaN =
        B.CreateICmpEQ(ScaleBByte, B.getInt32(0xFF), "scale_b_is_nan");
    Value *AnyIsNaN = B.CreateOr(AIsNaN, BIsNaN, "scale_is_nan");

    Value *Sum = B.CreateAdd(ScaleAByte, ScaleBByte, "scale_sum");
    Value *Biased = B.CreateSub(Sum, B.getInt32(254), "scale_exp");
    Function *LdexpFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::ldexp, {F32Ty, B.getInt32Ty()});
    Value *FiniteFactor = B.CreateCall(
        LdexpFn, {ConstantFP::get(F32Ty, 1.0), Biased}, "finite_factor");
    Factor = B.CreateSelect(AnyIsNaN, ConstantFP::getQNaN(F32Ty), FiniteFactor,
                            "scale_factor");
  } else {
    Value *FactorA = decodeScaleByte(B, M, F32Ty, ScaleAByte, AScaleFmt);
    Value *FactorB = decodeScaleByte(B, M, F32Ty, ScaleBByte, BScaleFmt);
    if (!FactorA || !FactorB)
      return nullptr;
    Factor = B.CreateFMul(FactorA, FactorB, "scale_factor");
  }

  auto *Vec4 = FixedVectorType::get(F32Ty, 4);
  Value *Splat = B.CreateInsertElement(PoisonValue::get(Vec4), Factor,
                                       B.getInt32(0), "factor_lane0");
  return B.CreateShuffleVector(Splat, PoisonValue::get(Vec4),
                               ArrayRef<int>{0, 0, 0, 0}, "factor_v4");
}

} // namespace

Expected<Value *> emitWMMAScaleF8F6F4toMFMA(
    RaiseContext &ctx, Value *a, Value *b, Value *c, Value *matrixAFmt,
    Value *matrixBFmt, Value *cMod, Value *matrixAScale, Value *matrixAScaleFmt,
    Value *scaleSrc0, Value *matrixBScale, Value *matrixBScaleFmt,
    Value *scaleSrc1, unsigned aDwords, unsigned bDwords) {
  IRBuilder<> &B = ctx.B;
  Module &M = ctx.M;

  // Supported fragment widths per side: 16 (FP8/BF8), 12 (FP6/BF6), 8 (FP4).
  // The decoder already guarantees these; a mismatch is an invariant violation
  // rather than an unsupported form, so it stays loud in release builds.
  auto SupportedDwords = [](unsigned dw) {
    return dw == 16 || dw == 12 || dw == 8;
  };
  if (!SupportedDwords(aDwords) || !SupportedDwords(bDwords))
    return createStringError("unexpected WMMA F8F6F4 fragment width");

  auto AsConstInt = [](Value *V) -> int64_t {
    return cast<ConstantInt>(V)->getZExtValue();
  };
  int aFmt = static_cast<int>(AsConstInt(matrixAFmt));
  int bFmt = static_cast<int>(AsConstInt(matrixBFmt));
  int aScaleFmt = static_cast<int>(AsConstInt(matrixAScaleFmt));
  int bScaleFmt = static_cast<int>(AsConstInt(matrixBScaleFmt));
  int aScaleSel = static_cast<int>(AsConstInt(matrixAScale));
  int bScaleSel = static_cast<int>(AsConstInt(matrixBScale));

  // matrix_*_scale is the SCL_OPSEL[0] row selector. Only ROW0 is modeled;
  // ROW1 is rejected.
  if (aScaleSel != 0 || bScaleSel != 0)
    return createStringError(
        "matrix_*_scale ROW1 selector is not supported for scaled WMMA");

  // Scale formats: E8M0 and E4M3 supported, E5M3 not yet.
  auto SupportedScaleFmt = [](int fmt) {
    return fmt == ScaleFmtE8M0 || fmt == ScaleFmtE4M3;
  };
  if (!SupportedScaleFmt(aScaleFmt) || !SupportedScaleFmt(bScaleFmt))
    return createStringError(
        "unsupported matrix_*_scale_fmt (E5M3 is not yet supported)");

  if (!isLegalScaleDataCombo(aFmt, aScaleFmt, bFmt, bScaleFmt))
    return createStringError("illegal (matrix data, scale) format combination");

  int aFmtEff = effectiveFmtAfterWiden(aFmt);
  int bFmtEff = effectiveFmtAfterWiden(bFmt);
  if (aFmtEff < 0 || bFmtEff < 0)
    return createStringError("unsupported matrix_a_fmt / matrix_b_fmt");
  Intrinsic::ID MfmaId = pickGfx942F8MfmaIntrinsic(aFmtEff, bFmtEff);
  if (MfmaId == Intrinsic::not_intrinsic)
    return createStringError(
        "no gfx942 FP8 MFMA intrinsic for this matrix format pair");

  SmallVector<Value *, 16> aSrcDwords(aDwords);
  SmallVector<Value *, 16> bSrcDwords(bDwords);
  Value *cDwords[8];
  unpackDwords(B, a, aDwords, ctx.I32Ty, aSrcDwords.data());
  unpackDwords(B, b, bDwords, ctx.I32Ty, bSrcDwords.data());
  unpackDwords(B, c, 8, ctx.I32Ty, cDwords);

  // Widen FP6/BF6/FP4 so the pipeline sees a uniform 16-dword fragment.
  SmallVector<Value *, 16> aDwordsArr;
  SmallVector<Value *, 16> bDwordsArr;
  auto WidenFragment = [&](int Fmt, ArrayRef<Value *> Src,
                           SmallVectorImpl<Value *> &Dst) {
    switch (Fmt) {
    case FmtFP4:
      widenF4FragmentToFP8(B, M, Src, Dst);
      break;
    case FmtFP6:
      widenFP6FragmentToFP8(B, M, Src, Dst);
      break;
    case FmtBF6:
      widenBF6FragmentToBF8(B, M, Src, Dst);
      break;
    default:
      Dst.assign(Src.begin(), Src.end());
      break; // FP8 / BF8
    }
  };
  WidenFragment(aFmt, aSrcDwords, aDwordsArr);
  WidenFragment(bFmt, bSrcDwords, bDwordsArr);
  assert(aDwordsArr.size() == 16 && bDwordsArr.size() == 16 &&
         "post-widen fragments must be 16 fp8 dwords / lane");

  Value *LaneId = emitLaneId(B, M, ctx.I32Ty);

  // 1 wave (MODREP) or 2 (WaveNative cross-widen).
  const unsigned numSrcWaves = ctx.Projection.numSourceWavesPerTarget();
  if (numSrcWaves != 1 && numSrcWaves != 2)
    return createStringError(
        "scaled WMMA decomposition supports 1 or 2 source waves only");

  auto *AccTy = FixedVectorType::get(ctx.F32Ty, 4);
  Function *MfmaFn = Intrinsic::getOrInsertDeclaration(&M, MfmaId);
  Value *Cbsz = B.getInt32(0);
  Value *Abid = B.getInt32(0);
  Value *Blgp = B.getInt32(0);
  Value *ZeroAcc = ConstantAggregateZero::get(AccTy);
  int64_t cModVal = AsConstInt(cMod);

  // Lane indices shared across both passes.
  Value *LaneMod16 = B.CreateAnd(LaneId, B.getInt32(15), "lane16");
  Value *LaneGroup = B.CreateLShr(LaneId, B.getInt32(4), "lane_grp");
  Value *W32Lane = B.CreateAnd(LaneId, B.getInt32(31), "w32_lane");

  // One pass per virtual W32 group (GroupBase 0 / 32).
  auto runPass = [&](unsigned GroupBase, Value *Result[8]) {
    Value *LoLane = B.CreateAdd(LaneMod16, B.getInt32(GroupBase), "lo_lane");
    Value *HiLane =
        B.CreateAdd(LaneMod16, B.getInt32(GroupBase + 16), "hi_lane");
    Value *AddrLo = B.CreateShl(LoLane, B.getInt32(2), "addr_lo");
    Value *AddrHi = B.CreateShl(HiLane, B.getInt32(2), "addr_hi");

    // All 4 K-block scale bytes ride in one i32, so one redistribution per src
    // suffices. The scale must follow its A/B data: even lane groups (0,2) read
    // the lower W32 half (AddrLo), odd groups (1,3) the upper (AddrHi).
    // Constant sources are lane-uniform, so skip the bpermute.
    Value *IsOddGroup = B.CreateTrunc(LaneGroup, B.getInt1Ty(), "lg_odd");
    auto RedistributeScale = [&](Value *ScaleSrc) -> Value * {
      if (isa<Constant>(ScaleSrc))
        return ScaleSrc;
      Value *Lo = emitDSBpermute(B, M, AddrLo, ScaleSrc);
      Value *Hi = emitDSBpermute(B, M, AddrHi, ScaleSrc);
      return B.CreateSelect(IsOddGroup, Hi, Lo, "scale_redist");
    };
    Value *ScaleSrc0Pass = RedistributeScale(scaleSrc0);
    Value *ScaleSrc1Pass = RedistributeScale(scaleSrc1);

    Value *MfmaC[4];
    redistributeAcc(B, M, cDwords, AddrLo, AddrHi, LaneGroup, MfmaC);
    Value *Acc = packDwords(B, MfmaC, 4, ctx.I32Ty, AccTy);

    // C_mod: 0=none, 1=neg, 2=abs, 3=neg(abs).
    if (cModVal & 2)
      Acc = B.CreateUnaryIntrinsic(Intrinsic::fabs, Acc, nullptr, "c_abs");
    if (cModVal & 1)
      Acc = B.CreateFNeg(Acc, "c_neg");

    // Redistribute A and B for all 4 K-blocks (each call covers 2 blocks).
    Value *aPerKBlock[4][2];
    Value *bPerKBlock[4][2];
    redistributeInput(B, M, aDwordsArr.data(), AddrLo, AddrHi, LaneGroup,
                      aPerKBlock[0], aPerKBlock[1]);
    redistributeInput(B, M, aDwordsArr.data() + 8, AddrLo, AddrHi, LaneGroup,
                      aPerKBlock[2], aPerKBlock[3]);
    redistributeInput(B, M, bDwordsArr.data(), AddrLo, AddrHi, LaneGroup,
                      bPerKBlock[0], bPerKBlock[1]);
    redistributeInput(B, M, bDwordsArr.data() + 8, AddrLo, AddrHi, LaneGroup,
                      bPerKBlock[2], bPerKBlock[3]);

    for (unsigned kBlock = 0; kBlock < 4; ++kBlock) {
      Value *SrcA = packDwords(B, aPerKBlock[kBlock], 2, ctx.I32Ty, ctx.I64Ty);
      Value *SrcB = packDwords(B, bPerKBlock[kBlock], 2, ctx.I32Ty, ctx.I64Ty);

      Value *Partial = ctx.Projection.wrapAsWWMValue(
          B,
          B.CreateCall(MfmaFn, {SrcA, SrcB, ZeroAcc, Cbsz, Abid, Blgp},
                       "kblock_partial"),
          "kblock_partial_wwm");

      Value *ScaleAByte = extractScaleByte(B, ScaleSrc0Pass, kBlock);
      Value *ScaleBByte = extractScaleByte(B, ScaleSrc1Pass, kBlock);
      Value *FactorVec = buildScaleFactorVec(B, M, ctx.F32Ty, ScaleAByte,
                                             ScaleBByte, aScaleFmt, bScaleFmt);

      Acc = B.CreateIntrinsic(Intrinsic::fmuladd, {AccTy},
                              {Partial, FactorVec, Acc}, nullptr,
                              "kblock_fmuladd");
    }

    // Collect the wave64-layout accumulator back to wave32.
    Value *MfmaDst[4];
    unpackDwords(B, Acc, 4, ctx.I32Ty, MfmaDst);
    collectResult(B, M, MfmaDst, W32Lane, Result);

    for (unsigned i = 0; i < 8; ++i)
      Result[i] =
          ctx.Projection.wrapAsWWMValue(B, Result[i], "wmma_scale_collect_wwm");
  };

  Value *Result0[8];
  runPass(0, Result0);

  Value *FinalDwords[8];
  if (numSrcWaves == 1) {
    for (unsigned i = 0; i < 8; ++i)
      FinalDwords[i] = Result0[i];
  } else {
    Value *Result1[8];
    runPass(32, Result1);
    Value *IsGroup1 = B.CreateICmpUGE(LaneId, B.getInt32(32), "is_group1");
    for (unsigned i = 0; i < 8; ++i)
      FinalDwords[i] = B.CreateSelect(IsGroup1, Result1[i], Result0[i], "sel");
  }

  return packDwords(B, FinalDwords, 8, ctx.I32Ty,
                    FixedVectorType::get(ctx.F32Ty, 8));
}

} // namespace COMGR::hotswap
