//===- handle-sop2.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonical-op-attrs.h"
#include "handlers.h"
#include "hotswap/raise-failure.h"
#include "source-image-address.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace COMGR::hotswap {

// SPE attribute registrations. All of these route through
// `writeReg{32,64,ExecWidth}` which dispatch EXEC writes to
// `regs.storeExec`. Audit any addition before landing -- see
// AGENTS.md's SPE section.
ArrayRef<CanonicalOpAttrSpec> getHandlerSOP2Attrs() {
  static constexpr CanonicalOpAttrSpec kAttrs[] = {
      {CanonicalOp::S_AND_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_AND_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_OR_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_OR_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_XOR_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_XOR_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_ANDN2_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_ANDN2_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_ORN2_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_ORN2_B64, {/*routesExecThroughStoreExec=*/true}},
      // s_{nand,nor,xnor}_b{32,64}: `dst = ~(src0 OP src1)`. Identical
      // SPE shape to s_{and,or,xor}_b* -- they read EXEC-relative scalar
      // sources and dispatch the result through writeReg{32,64}, which
      // routes to storeExec when the destination operand is EXEC. See
      // SOPInstructions.td:789-803 for the LLVM patterns.
      {CanonicalOp::S_NAND_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_NAND_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_NOR_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_NOR_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_XNOR_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_XNOR_B64, {/*routesExecThroughStoreExec=*/true}},
      // s_absdiff_i32 returns an i32 magnitude; in principle the result
      // can target EXEC like any other SOP2 i32 writer, so route through
      // storeExec for safety.
      {CanonicalOp::S_ABSDIFF_I32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL_B32, {/*routesExecThroughStoreExec=*/true}},
      // s_lshl{1,2,3,4}_add_u32 write their i32 (shift-then-add) result
      // through writeReg32(), which routes an explicit EXEC destination
      // through storeExec (see reg-file.cpp). Triton's gfx12 divergent-
      // region exit emits `s_lshl2_add_u32 exec_lo, exec_lo, sMASK` as a
      // fused EXEC update; declare the attr so the SPE gate accepts it
      // (the handler already lowers it correctly).
      {CanonicalOp::S_LSHL1_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL2_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL3_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL4_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL_B64, {/*routesExecThroughStoreExec=*/true}},
      // s_lshl{1,2,3,4}_add_u32: dst = (src0 << N) + src1. Same SPE shape as
      // the plain shift/bitwise routers above -- the handler computes the
      // scalar result and dispatches via writeReg32(), which routes to
      // storeExec when the destination operand is EXEC. Autotuned Triton
      // kernels emit these as address/mask scalar math that can land in EXEC;
      // without these entries the SPE A-level gate aborts with
      // SPE-unmodeled-EXEC-writer. See the S_LSHL{1,2,3,4}_ADD_U32 handlers
      // below (all use writeReg32) and SOPInstructions.td:S_LSHLn_ADD_U32.
      {CanonicalOp::S_LSHL1_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL2_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL3_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHL4_ADD_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHR_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_LSHR_B64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_ASHR_I64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_BFM_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_BFM_B64, {/*routesExecThroughStoreExec=*/true}},
      // s_bfe_i64 writes its i64 sign-extracted result through
      // writeReg64(), so an explicit-EXEC destination (e.g.
      // `s_bfe_i64 exec, s[0:1], 0x10000`) routes through
      // storeExec like the other 64-bit SOP2 EXEC writers above.
      // The SPE gate requires the attr to be declared here.
      {CanonicalOp::S_BFE_I64, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_CSELECT_B32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_CSELECT_B64, {/*routesExecThroughStoreExec=*/true}},
      // s_{min,max}_{i,u}32 write an ordinary 32-bit scalar result. This
      // handler commits that result through writeReg32(), which routes explicit
      // EXEC destinations through storeExec.
      {CanonicalOp::S_MIN_I32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_MIN_U32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_MAX_I32, {/*routesExecThroughStoreExec=*/true}},
      {CanonicalOp::S_MAX_U32, {/*routesExecThroughStoreExec=*/true}},
  };
  return kAttrs;
}

// Look up the per-lane wave-width i1 for source operand `i`, covering
// three cases that each carry full wave-width information that would
// otherwise be lost when funneled through the source-width SGPR path:
//
//   1. SGPR shadowed by the V_CMP -> SGPR shadow cache in
//      `RaiseContext::lastSgprWaveMaskI1` (matmul128x128-class fix).
//   2. VCC -- the V_CMP -> VCC path stores a per-lane i1 directly in
//      the VCC alloca, so we can load it per-lane without the
//      ballot-then-truncate round-trip.
//   3. EXEC -- the EXEC alloca carries the full wave-width mask at
//      `execStorageTy()` (i64 on wave64 target).  Per-lane i1 comes
//      from `extractLaneBitFromWaveMask` on the loaded EXEC.
//
// Returns null for immediate / VGPR sources and for non-shadowed
// SGPRs.  Callers that handle all three wave-width-carrying kinds
// get correct per-lane results under cross-widening; callers that
// only look at the SGPR shadow get the narrower matmul-fix
// coverage.
//
// Context: cross-widening (wave32 source -> wave64 target) loses the
// upper 32 bits of a V_CMP-produced wave-mask when the mask is
// funneled through a 32-bit source-width SGPR.  Scalar binary ops
// (`s_xor_b32`, `s_and_b32`, `s_or_b32`) on two wave-width-carrying
// sources can PROPAGATE full wave-width information by computing
// the per-lane i1 of the result directly from the two input i1s.
// This closes three idiom classes from Triton's gfx1250 output:
//
//   * `v_cmp_X s2; v_cmp_Y s3; s_xor_b32 s2, s2, s3; v_cndmask ... s2`
//     (both sources shadowed SGPR -- core matmul-fix shape).
//   * `v_cmp_X vcc; s_and_saveexec_b32 s2, vcc; s_xor_b32 s2,
//     exec_lo, s2; v_cndmask ... s2` (right source = saved old_exec
//     in SGPR, left source = current exec_lo after saveexec -- the
//     "else-branch mask" idiom Triton's tl.sort at small BLOCK_N
//     emits between its bitonic stages).
//   * `s_and_b32 s2, s2, vcc_lo` / `s_or_b32 s2, s2, vcc_lo` where
//     one source is VCC.
static llvm::Value *tryGetSrcWaveMaskI1(RaiseContext &Ctx, OpResolver &Op,
                                        unsigned I) {
  if (!Op.isSrcReg(I)) {
    // Immediate / expr operands still denote source-width scalar wave masks in
    // SOP2 mask algebra (e.g. `s_xor_b32 sN, sMask, -1`). Lift them through
    // the same source->exec widening path and extract the per-lane bit so
    // shadow propagation can preserve full wave-width i1 semantics.
    llvm::Value *Mask = Op.srcExecWidth(I);
    return Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, Mask);
  }
  ParsedReg Pr = Op.srcReg(I);
  switch (Pr.RegKind) {
  case ParsedReg::SGPR: {
    if (llvm::Value *Fresh = Ctx.lookupSgprWaveMaskI1(Pr.BaseIdx))
      return Fresh;
    if (llvm::Value *ShadowValid = Ctx.loadSgprWaveMaskValid(Pr.BaseIdx)) {
      llvm::Value *ShadowExec = Ctx.loadSgprWaveMaskExec(Pr.BaseIdx);
      llvm::Value *ShadowI1 =
          Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, ShadowExec);
      llvm::Value *SgprMask = Ctx.Isa.isWave32()
                                  ? Ctx.Regs.loadSGPR32(Ctx.B, Pr.BaseIdx)
                                  : Ctx.Regs.loadSGPR64(Ctx.B, Pr.BaseIdx);
      llvm::Value *Fallback =
          Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, SgprMask);
      return Ctx.B.CreateSelect(ShadowValid, ShadowI1, Fallback,
                                "sop2_src_sgpr_mask_shadow_sel");
    }
    return nullptr;
  }
  case ParsedReg::VCC:
    // VCC alloca stores the per-lane i1 directly; load it to get
    // the correct wave-width i1 without the ballot-truncate-
    // replicate round-trip.
    return Ctx.Regs.loadVCC(Ctx.B);
  case ParsedReg::EXEC: {
    // EXEC storage is always the wave-width mask; extract the
    // per-lane bit via the projection's helper so the width /
    // replication policy matches what the sibling reader
    // (`extractLaneBitFromWaveMask` in the V_CNDMASK consumer
    // path) would produce.
    llvm::Value *ExecVal = Ctx.Regs.loadExec(Ctx.B);
    return Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, ExecVal);
  }
  default:
    return nullptr;
  }
}

// Record a derived wave-mask i1 on `dstReg`.  Handles the two
// destination kinds that carry wave-width information:
//
//   * SGPR -- record in the V_CMP shadow cache so the next
//     V_CNDMASK or scalar-op consumer in this BB picks up the
//     full-width i1 instead of the narrow-mask fallback.
//   * VCC -- OVERWRITE the VCC alloca's i1 with the wave-width
//     result.  The handler's earlier `writeReg32(VCC, i32_xor)`
//     derived an i1 from the lossy i32 via
//     `extractLaneBitFromWaveMask(trunc-replicate-extract)`;
//     we replace that with the structurally-correct per-lane i1
//     so downstream V_CNDMASK consumers that read VCC directly
//     (via the VCC alloca's load) get the right bit.
//
// EXEC destinations are the source-wave mask itself.  When both inputs
// supplied a per-lane i1, commit that full-width mask directly so
// wave-native cross-widening preserves independent masks for lanes 0..31
// and 32..63 instead of keeping the earlier low32 broadcast fallback.
// Other destination kinds (VGPR, M0, TTMP, immediate) are no-ops -- they
// either don't participate in the cross-widening ballot truncation this
// propagation addresses, or the earlier `writeReg32` already did the right
// thing.
static void recordDerivedWaveMaskI1(RaiseContext &Ctx, ParsedReg DstReg,
                                    llvm::Value *I1) {
  if (!I1)
    return;
  switch (DstReg.RegKind) {
  case ParsedReg::SGPR:
    Ctx.recordSgprWaveMaskI1(DstReg.BaseIdx, I1,
                             /*isPair=*/DstReg.WidthInDwords >= 2);
    return;
  case ParsedReg::VCC:
    // Overwrite VCC's stored i1 with the wave-width-correct value.
    // The earlier `writeReg32(VCC, i32)` landed a lossy i1; this
    // call replaces it.  Correctness invariant: SSA-monotonic
    // within this BB (the next reader sees the new i1).
    Ctx.Regs.storeVCC(Ctx.B, I1);
    return;
  case ParsedReg::EXEC: {
    llvm::Value *Mask = Ctx.Projection.ballotI1ToWidth(
        Ctx.B, I1, Ctx.Regs.ExecTy, "wave_mask_exec");
    Ctx.storeExec(Mask);
    return;
  }
  default:
    return;
  }
}

// Vector compares produce one bit per lane. When scalar code combines those
// values as masks, SCC should answer "is any modeled lane bit set?", not "is
// the current lane's bit set?". Rebuild the full mask with a ballot before
// testing it for zero.
static void storeSccFromWaveMaskI1(RaiseContext &Ctx, llvm::Value *I1,
                                   const Twine &Name) {
  Value *Mask = Ctx.Projection.ballotI1ToWidth(Ctx.B, I1, Ctx.Regs.ExecTy,
                                               Name + "_ballot");
  Ctx.Regs.storeSCC(
      Ctx.B, Ctx.B.CreateICmpNE(Mask, Constant::getNullValue(Mask->getType()),
                                Name + "_nonzero"));
}

// s_lshl{1,2,3,4}_add_u32: `D.u = (S0.u << N) + S1.u`, with
// `SCC = unsigned carry-out of the full sum`. The left shift can push
// significant bits past bit 31, so the destination truncation to i32
// loses carry information -- deriving `SCC = (D != 0)` from the truncated
// result is wrong (e.g. S0=0x80000000, N=1, S1=0 truncates D to 0 while
// SCC must be 1). Compute the shift-add once in i64, truncate for the
// destination, and set SCC from bits [63:32] being nonzero (equivalently
// the unbounded result exceeding 0xFFFFFFFF). See instruction_manual.pdf
// sec. S_LSHLn_ADD_U32 and the S_ADD_U32 handler above for the sibling
// carry-out pattern.
static void handleLshlAddU32(RaiseContext &Ctx, OpResolver &Op, unsigned ShAmt,
                             const Twine &Name, HandlerResult &Hr) {
  Value *Src0 = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty, Name + "_s0");
  Value *Src1 = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty, Name + "_s1");
  Value *Wide =
      Ctx.B.CreateAdd(Ctx.B.CreateShl(Src0, ConstantInt::get(Ctx.I64Ty, ShAmt)),
                      Src1, Name + "_wide");
  Value *Res = Ctx.B.CreateTrunc(Wide, Ctx.I32Ty, Name);
  Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
  // SCC = carry-out = (wide >> 32) != 0.
  Value *Carry = Ctx.B.CreateICmpUGT(
      Wide, ConstantInt::get(Ctx.I64Ty, 0xFFFFFFFFull), Name + "_scc");
  Ctx.Regs.storeSCC(Ctx.B, Carry);
  Hr.SccResult = Res;
  Hr.SccHandled = true;
  Hr.Handled = true;
}

Expected<HandlerResult> handleSOP2(RaiseContext &Ctx, const DecodedInst &Di,
                                   OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  // 32-bit binary ops -- auto SCC via sccResult unless the operands are known
  // lane masks, in which case SCC is derived from the full propagated mask.
  //
  // When both sources carry per-lane mask bits, compute the result in i1-space
  // and re-record it after the scalar write has invalidated the cache via
  // `onSgprWritten`. This keeps later mask consumers and SCC from falling back
  // to source-width scalar bits.
  if (Sop == CanonicalOp::S_AND_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult = Ctx.B.CreateAnd(Op.src(0), Op.src(1), "and");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *AndI1 = Ctx.B.CreateAnd(S0I1, S1I1, "wave_mask_and");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), AndI1);
      storeSccFromWaveMaskI1(Ctx, AndI1, "wave_mask_and_scc");
      Hr.SccHandled = true;
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_OR_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult = Ctx.B.CreateOr(Op.src(0), Op.src(1), "or");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *OrI1 = Ctx.B.CreateOr(S0I1, S1I1, "wave_mask_or");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), OrI1);
      storeSccFromWaveMaskI1(Ctx, OrI1, "wave_mask_or_scc");
      Hr.SccHandled = true;
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_LSHL_B32) {
    Hr.SccResult = Ctx.B.CreateShl(Op.src(0), Op.src(1), "shl");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_LSHR_B32) {
    Hr.SccResult = Ctx.B.CreateLShr(Op.src(0), Op.src(1), "lshr");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ASHR_I32) {
    Hr.SccResult = Ctx.B.CreateAShr(Op.src(0), Op.src(1), "ashr");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  // s_add_i32 / s_add_u32 (both CanonicalOp::S_ADD_U32)
  if (Sop == CanonicalOp::S_ADD_U32) { // Match by canonical semantic opcode,
                                       // not raw mnemonic string
    Value *Src0 = Op.src(0),
          *Src1 = Op.src(1); // Read source operands -- resolves SGPR, VGPR, or
                             // immediate to LLVM Value*
    Value *Res = Ctx.B.CreateAdd(
        Src0, Src1, "add"); // Emit LLVM IR: %add = add i32 %src0, %src1
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Res); // Store result into destination register's alloca
                              // (later promoted to SSA)
    auto *Ov =
        Ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow,
                              {Ctx.I32Ty}, // Compute carry-out using LLVM's
                                           // uadd.with.overflow intrinsic
                              {Src0, Src1});
    Ctx.Regs.storeSCC(Ctx.B, Ctx.B.CreateExtractValue(
                                 Ov, 1)); // Extract the overflow bit and write
                                          // it to SCC (Scalar Condition Code)
    Hr.SccHandled = true; // Tell the dispatch loop: "I wrote SCC myself, don't
                          // auto-compute it"
    Hr.Handled = true;    // Tell the dispatch loop: "This instruction was
                          // successfully raised"
    return Hr;
  }
  // s_sub_i32 / s_sub_u32 (both CanonicalOp::S_SUB_U32)
  if (Sop == CanonicalOp::S_SUB_U32) {
    Value *Src0 = Op.src(0), *Src1 = Op.src(1);
    Value *Res = Ctx.B.CreateSub(Src0, Src1, "sub");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    Ctx.Regs.storeSCC(Ctx.B, Ctx.B.CreateICmpULT(Src0, Src1));
    Hr.SccHandled = true;
    Hr.Handled = true;
    return Hr;
  }

  // Special SCC semantics -- handler writes SCC explicitly
  if (Sop == CanonicalOp::S_ADDC_U32) {
    Value *Src0 = Op.src(0), *Src1 = Op.src(1);
    Value *Cin = Ctx.B.CreateZExt(Ctx.Regs.loadSCC(Ctx.B), Ctx.I32Ty);
    Function *UaddOv = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::uadd_with_overflow, {Ctx.I32Ty});
    Value *Step1 = Ctx.B.CreateCall(UaddOv, {Src0, Src1});
    Value *Sum1 = Ctx.B.CreateExtractValue(Step1, 0);
    Value *C1 = Ctx.B.CreateExtractValue(Step1, 1);
    Value *Step2 = Ctx.B.CreateCall(UaddOv, {Sum1, Cin});
    Value *Res = Ctx.B.CreateExtractValue(Step2, 0, "addc");
    Value *C2 = Ctx.B.CreateExtractValue(Step2, 1);
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    Ctx.Regs.storeSCC(Ctx.B, Ctx.B.CreateOr(C1, C2));
    Hr.SccHandled = true;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_SUBB_U32) {
    Value *Src0 = Op.src(0), *Src1 = Op.src(1);
    Value *Borrow = Ctx.B.CreateZExt(Ctx.Regs.loadSCC(Ctx.B), Ctx.I32Ty);
    Value *Res = Ctx.B.CreateSub(Ctx.B.CreateSub(Src0, Src1), Borrow, "subb");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    Ctx.Regs.storeSCC(
        Ctx.B, Ctx.B.CreateOr(Ctx.B.CreateICmpULT(Src0, Src1),
                              Ctx.B.CreateAnd(Ctx.B.CreateICmpEQ(Src0, Src1),
                                              Ctx.Regs.loadSCC(Ctx.B))));
    Hr.SccHandled = true;
    Hr.Handled = true;
    return Hr;
  }

  // No SCC side-effect (di.defsSCC=false for these)
  if (Sop == CanonicalOp::S_MUL_I32) {
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateMul(Op.src(0), Op.src(1), "mul"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_MUL_HI_U32) {
    Value *A = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty),
          *B = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty);
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateTrunc(
            Ctx.B.CreateLShr(Ctx.B.CreateMul(A, B, "mulhi_wide"), 32),
            Ctx.I32Ty, "mulhi"));
    Hr.Handled = true;
    return Hr;
  }
  // s_mul_hi_i32: signed mul-high. Same widening pattern as
  // S_MUL_HI_U32 above, but sign-extend both operands so the wide
  // multiply produces a signed product. SOPInstructions.td .td
  // pattern is `mulhs SSrc_b32, SSrc_b32` (line ~849); the only
  // operational difference vs S_MUL_HI_U32 (`mulhu`) is the
  // extension semantics on the inputs.
  if (Sop == CanonicalOp::S_MUL_HI_I32) {
    Value *A = Ctx.B.CreateSExt(Op.src(0), Ctx.I64Ty),
          *B = Ctx.B.CreateSExt(Op.src(1), Ctx.I64Ty);
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateTrunc(
            Ctx.B.CreateLShr(Ctx.B.CreateMul(A, B, "mulhi_i_wide"), 32),
            Ctx.I32Ty, "mulhi_i"));
    Hr.Handled = true;
    return Hr;
  }
  // GFX12 scalar FP multiply
  if (Sop == CanonicalOp::S_MUL_F32) {
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), Ctx.F32Ty);
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateFMul(S0, S1, "s_fmul"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ADD_F32) {
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), Ctx.F32Ty);
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateFAdd(S0, S1, "s_fadd"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // gfx11+ scalar FP subtract. Direct mirror of S_ADD_F32 above;
  // the .td pattern is `any_fsub` (SOPInstructions.td:894). No
  // source modifiers on SOP2 -- the operands are bare i32-shaped
  // SGPRs that we reinterpret as f32.
  if (Sop == CanonicalOp::S_SUB_F32) {
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), Ctx.F32Ty);
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateFSub(S0, S1, "s_fsub"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_FMAAK_F32 || Sop == CanonicalOp::S_FMAMK_F32) {
    // Source order follows the MC operand order. For S_FMAAK this is
    // (src0, src1, literal); for S_FMAMK it is (src0, literal, src1), exactly
    // matching the manual's fma argument order.
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), Ctx.F32Ty);
    Value *S2 = Ctx.B.CreateBitCast(Op.src(2), Ctx.F32Ty);
    Function *Fma =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    const char *Name =
        (Sop == CanonicalOp::S_FMAAK_F32) ? "s_fmaak" : "s_fmamk";
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, S1, S2}, Name),
                            Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // gfx11+ scalar FP fused multiply-accumulate. Manual sec. 4.5.25 marks this
  // OPF_DACCUM and defines `D0.f32 = fma(S0.f32, S1.f32, D0.f32)`, so the
  // third operand is the old destination value, not a hidden source slot.
  if (Sop == CanonicalOp::S_FMAC_F32) {
    ParsedReg DstReg = Op.dst();
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), Ctx.F32Ty);
    Value *Acc =
        Ctx.B.CreateBitCast(Ctx.Regs.readReg32(Ctx.B, DstReg), Ctx.F32Ty);
    Function *Fma =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Ctx.Regs.writeReg32(
        Ctx.B, DstReg,
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, S1, Acc}, "s_fmac"),
                            Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // Scalar IEEE-754-2019 maximumNumber/minimumNumber. LLVM's canonical pseudo
  // is `S_{MAX,MIN}_F32`; `instruction_manual.pdf` sec. 4.5.39/sec. 4.5.45
  // names the gfx12+ real mnemonics `s_max_num_f32` / `s_min_num_f32`, with
  // `s_max_f32` / `s_min_f32` accepted as compatibility aliases. The manual's
  // pseudocode favors a numeric operand over NaN (including signaling NaN after
  // setting invalid), quiets all-NaN results, and orders signed zeros (+0 > -0
  // for max, -0 < +0 for min). LLVM's `maximumnum` / `minimumnum` intrinsics
  // model that NUM family; the NaN-propagating `maximum` / `minimum` intrinsics
  // are for the separate S_MAXIMUM_F32 / S_MINIMUM_F32 opcode family and must
  // not be used here.
  if (Sop == CanonicalOp::S_MAX_NUM_F32 || Sop == CanonicalOp::S_MIN_NUM_F32) {
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), Ctx.F32Ty);
    Intrinsic::ID Iid = (Sop == CanonicalOp::S_MAX_NUM_F32)
                            ? Intrinsic::maximumnum
                            : Intrinsic::minimumnum;
    const char *Name =
        (Sop == CanonicalOp::S_MAX_NUM_F32) ? "s_fmax_num" : "s_fmin_num";
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Iid, {Ctx.F32Ty});
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fn, {S0, S1}, Name), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // gfx12+ scalar IEEE-754-2019 NaN-propagating f16 maximum/minimum. The f16
  // operand lives in the low 16 bits of the SGPRs; the high half is ignored on
  // read and zeroed on write. Map to `llvm.maximum.f16` / `llvm.minimum.f16`.
  if (Sop == CanonicalOp::S_MAXIMUM_F16 || Sop == CanonicalOp::S_MINIMUM_F16) {
    Type *I16Ty = Type::getInt16Ty(Ctx.C);
    Value *S0Lo = Ctx.B.CreateTrunc(Op.src(0), I16Ty);
    Value *S1Lo = Ctx.B.CreateTrunc(Op.src(1), I16Ty);
    Value *S0 = Ctx.B.CreateBitCast(S0Lo, Ctx.F16Ty);
    Value *S1 = Ctx.B.CreateBitCast(S1Lo, Ctx.F16Ty);
    Intrinsic::ID Iid = (Sop == CanonicalOp::S_MAXIMUM_F16)
                            ? Intrinsic::maximum
                            : Intrinsic::minimum;
    const char *Name =
        (Sop == CanonicalOp::S_MAXIMUM_F16) ? "smaximum_f16" : "sminimum_f16";
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Iid, {Ctx.F16Ty});
    Value *R = Ctx.B.CreateCall(Fn, {S0, S1}, Name);
    Value *RBits = Ctx.B.CreateBitCast(R, I16Ty);
    Value *R32 = Ctx.B.CreateZExt(RBits, Ctx.I32Ty);
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), R32);
    Hr.Handled = true;
    return Hr;
  }
  // gfx12+ scalar IEEE-754-2019 NaN-propagating f32 maximum/minimum. SOP2 has
  // no source/output modifiers, so reinterpret the i32 SGPRs as f32 and map to
  // `llvm.maximum.f32` / `llvm.minimum.f32`.
  if (Sop == CanonicalOp::S_MAXIMUM_F32 || Sop == CanonicalOp::S_MINIMUM_F32) {
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), Ctx.F32Ty);
    Intrinsic::ID Iid = (Sop == CanonicalOp::S_MAXIMUM_F32)
                            ? Intrinsic::maximum
                            : Intrinsic::minimum;
    const char *Name =
        (Sop == CanonicalOp::S_MAXIMUM_F32) ? "s_fmaximum" : "s_fminimum";
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Iid, {Ctx.F32Ty});
    Ctx.Regs.writeReg32(
        Ctx.B, Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fn, {S0, S1}, Name), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // GFX12 scalar 64-bit ops
  if (Sop == CanonicalOp::S_MUL_U64) {
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(),
                        Ctx.B.CreateMul(Op.src64(0), Op.src64(1), "smul64"));
    Hr.Handled = true;
    return Hr;
  }
  // s_add/sub_nc_u64: gfx12 64-bit scalar add/sub, no carry. SCC is *not*
  // updated (the `nc` suffix) for either form; see
  // SOPInstructions.td ~661 for both opcodes' shared `no-Defs-[SCC]`
  // shape.  Opcode-map row: `opcode-map.cpp` folds LLVM's
  // `S_ADD_U64` pseudo into this single CanonicalOp (gfx12 renamed the
  // mnemonic).  An earlier version of this handler also matched a
  // dead `CanonicalOp::S_ADD_U64`; that enum entry is gone, see
  // opcode-map.cpp's S_ADD_U64 comment for the audit trail.
  if (Sop == CanonicalOp::S_ADD_NC_U64 || Sop == CanonicalOp::S_SUB_NC_U64) {
    auto SrcSourceImageAddr = [&](unsigned I) -> std::optional<uint64_t> {
      if (!Op.isSrcReg(I))
        return std::nullopt;
      ParsedReg SrcPr = Op.srcReg(I);
      if (SrcPr.RegKind != ParsedReg::SGPR || SrcPr.BaseIdx < 0)
        return std::nullopt;
      return Ctx.lookupSourceImageSgprPairAddr(SrcPr.BaseIdx);
    };
    auto SrcSignedImm = [&](unsigned I) -> std::optional<int64_t> {
      if (Op.isSrcReg(I))
        return std::nullopt;
      return evalOperandAsConst(Di.Inst, Op.srcIdx(I));
    };
    Value *Result = Sop == CanonicalOp::S_ADD_NC_U64
                        ? Ctx.B.CreateAdd(Op.src64(0), Op.src64(1), "sadd64")
                        : Ctx.B.CreateSub(Op.src64(0), Op.src64(1), "ssub64");
    ParsedReg Dst = Op.dst();
    Ctx.Regs.writeReg64(Ctx.B, Dst, Result);
    std::optional<uint64_t> SourceImageResult;
    std::optional<uint64_t> Src0SourceAddr = SrcSourceImageAddr(0);
    std::optional<uint64_t> Src1SourceAddr = SrcSourceImageAddr(1);
    std::optional<int64_t> Src0Imm = SrcSignedImm(0);
    std::optional<int64_t> Src1Imm = SrcSignedImm(1);
    if (Src0SourceAddr && Src1Imm) {
      Expected<uint64_t> NewSourceAddr =
          Sop == CanonicalOp::S_ADD_NC_U64
              ? applySourceImageByteOffset(Di, "SOP2", *Src0SourceAddr,
                                           *Src1Imm)
              : subtractSourceImageByteOffset(Di, "SOP2", *Src0SourceAddr,
                                              *Src1Imm);
      if (!NewSourceAddr)
        return NewSourceAddr.takeError();
      SourceImageResult = *NewSourceAddr;
    } else if (Sop == CanonicalOp::S_ADD_NC_U64 && Src1SourceAddr && Src0Imm) {
      Expected<uint64_t> NewSourceAddr =
          applySourceImageByteOffset(Di, "SOP2", *Src1SourceAddr, *Src0Imm);
      if (!NewSourceAddr)
        return NewSourceAddr.takeError();
      SourceImageResult = *NewSourceAddr;
    }
    if (SourceImageResult) {
      Ctx.recordSourceImageSgprPairAddr(Dst.BaseIdx, *SourceImageResult);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_MIN_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Hr.SccResult =
        Ctx.B.CreateSelect(Ctx.B.CreateICmpULT(S0, S1), S0, S1, "smin");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_MAX_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Hr.SccResult =
        Ctx.B.CreateSelect(Ctx.B.CreateICmpUGT(S0, S1), S0, S1, "smax");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_MIN_I32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Hr.SccResult =
        Ctx.B.CreateSelect(Ctx.B.CreateICmpSLT(S0, S1), S0, S1, "smin");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_MAX_I32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Hr.SccResult =
        Ctx.B.CreateSelect(Ctx.B.CreateICmpSGT(S0, S1), S0, S1, "smax");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_LSHL1_ADD_U32) {
    handleLshlAddU32(Ctx, Op, 1, "lshl1add", Hr);
    return Hr;
  }
  if (Sop == CanonicalOp::S_LSHL2_ADD_U32) {
    handleLshlAddU32(Ctx, Op, 2, "lshl2add", Hr);
    return Hr;
  }
  if (Sop == CanonicalOp::S_LSHL3_ADD_U32) {
    handleLshlAddU32(Ctx, Op, 3, "lshl3add", Hr);
    return Hr;
  }
  if (Sop == CanonicalOp::S_LSHL4_ADD_U32) {
    handleLshlAddU32(Ctx, Op, 4, "lshl4add", Hr);
    return Hr;
  }
  if (Sop == CanonicalOp::S_XOR_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult = Ctx.B.CreateXor(Op.src(0), Op.src(1), "xor");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *XorI1 = Ctx.B.CreateXor(S0I1, S1I1, "wave_mask_xor");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), XorI1);
      storeSccFromWaveMaskI1(Ctx, XorI1, "wave_mask_xor_scc");
      Hr.SccHandled = true;
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_XOR_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult = Ctx.B.CreateXor(Op.src64(0), Op.src64(1), "xor64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *XorI1 = Ctx.B.CreateXor(S0I1, S1I1, "wave_mask_xor64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), XorI1);
      storeSccFromWaveMaskI1(Ctx, XorI1, "wave_mask_xor64_scc");
      Hr.SccHandled = true;
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BFM_B64) {
    // s_bfm_b64 dst, width, offset: creates a 64-bit mask with `width` ones
    // starting at `offset`
    Value *Width = Ctx.B.CreateZExt(
        Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0x3F)),
        Ctx.I64Ty);
    Value *Offset = Ctx.B.CreateZExt(
        Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0x3F)),
        Ctx.I64Ty);
    Value *Mask =
        Ctx.B.CreateSub(Ctx.B.CreateShl(ConstantInt::get(Ctx.I64Ty, 1), Width),
                        ConstantInt::get(Ctx.I64Ty, 1));
    Hr.SccResult = Ctx.B.CreateShl(Mask, Offset, "bfm64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BFM_B32) {
    Value *Width =
        Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0x1F));
    Value *Offset =
        Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0x1F));
    Value *Mask =
        Ctx.B.CreateSub(Ctx.B.CreateShl(ConstantInt::get(Ctx.I32Ty, 1), Width),
                        ConstantInt::get(Ctx.I32Ty, 1));
    Hr.SccResult = Ctx.B.CreateShl(Mask, Offset, "bfm32");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BFE_U32) {
    // ---- Class 1 rescue: `s_bfe_u32 sDST, ttmp8, 0x50019` (wave_id lift) ----
    //
    // Canonical gfx1250 HIP prologue for reading `wave_id_in_workgroup`:
    //
    //     s_bfe_u32 sN, ttmp8, 0x50019   ; extract ttmp8[29:25]
    //
    // The command processor stores workgroup-scheduling metadata in the
    // `ttmp` bank, and -- on gfx12+ -- bits [29:25] of `ttmp8` carry the
    // wave's rank within its workgroup (0..max_waves_per_wg-1). The
    // `bfe(ttmp8, 0x50019)` immediate encodes (offset=25, width=5), so
    // the source semantics are exactly "read `wave_id_in_workgroup`".
    //
    // Why a pattern-lift is needed under cross-widening (wave32 -> wave64,
    // `WaveNativeProjection`):
    //   - `wave_id_in_workgroup` is a Class 1 value (`sec. 6` of
    //     `hotswap/docs/wave-size-translation.md`): it depends on the
    //     absolute lane position within the target wave, not merely
    //     `lane_id mod W_s`. Target lanes 0..W_s-1 correspond to one
    //     source wave and must read `wave_id = 2k`; target lanes
    //     W_s..2*W_s-1 correspond to the next source wave and must
    //     read `wave_id = 2k+1`.
    //   - The raiser's `ttmp8` seed (`raiser.cpp`, phase-4 entry init)
    //     stores the divergent expression `(workitem.id.x >> log2(W_s))
    //     << 25` into the `ttmp8` alloca. At the LLVM-IR level this is
    //     already per-lane divergent, and `mem2reg + InstCombine` fold
    //     the BFE round-trip back to `workitem.id.x >> log2(W_s) & 0x1F`.
    //   - Empirically, though, the formally-scalar `s_bfe_u32` shape --
    //     SGPR-class source (`ttmp8`) feeding an SGPR-class destination
    //     (`sDST`) -- loses its per-lane divergence somewhere in the
    //     gfx942 backend's scalarisation / divergence-analysis pipeline:
    //     downstream SGPR consumers see a single lane-0 value, so all
    //     64 target lanes read `wave_id = 0` and matmul tile-assignment
    //     collapses to a checkerboard (upper half writes onto lower
    //     half's tile). Refusing the kernel via `TtmpWaveIdLeak` made
    //     the symptom go away but blocked the GPT-OSS / matmul corpus.
    //
    // Principled rescue: emit the architectural expression
    // `(workitem.id.x >> log2(W_s)) & 0x1F` *directly* at the raise
    // site, as a fresh `@llvm.amdgcn.workitem.id.x` leaf that the
    // AMDGPU divergence analysis already marks divergent. The
    // destination alloca still round-trips, but the value now enters
    // the SGPR alloca from a known-divergent leaf rather than a chain
    // the backend later re-uniforms. Downstream uses of `sDST` see a
    // divergent VGPR value, preserving the per-source-wave distinction
    // through every consumer (address arithmetic, predicate
    // conversion, atomic indices).
    //
    // Same-wave translations (gfx942 -> gfx942, gfx1250 -> gfx1250) get
    // an identical IR shape -- the alloca path would have collapsed to
    // this expression anyway after InstCombine -- so the lift is safe
    // unconditionally and keeps one shape across projections.
    //
    // Scope: deliberately narrow. Only the *exact* canonical immediate
    // `0x50019` (offset=25, width=5) and *exact* `ttmp8` source get the
    // lift. Any other BFE against `ttmp` falls through to the generic
    // bitfield extract; those would indicate a non-canonical kernel
    // using `ttmp` for something the raiser's init does not model, and
    // forcing them through the lift would silently miscompile.
    //
    // See `hotswap/docs/wave-size-translation.md` sec. 5.6.2 (wave_id
    // lift) and sec. 6 (Class 1 obstructions) for the full contract.
    if (Op.isSrcReg(0) && !Op.isSrcReg(1)) {
      ParsedReg SrcPr = Op.srcReg(0);
      int64_t CtrlImm = Op.srcImm(1);
      if (SrcPr.RegKind == ParsedReg::TTMP && SrcPr.BaseIdx == 8 &&
          CtrlImm == 0x50019) {
        unsigned SrcWaveBits = Ctx.Isa.WaveSize;
        if (SrcWaveBits != 32 && SrcWaveBits != 64)
          return RaiseFailure::unsupportedInstructionForm(
              Di, "SOP2",
              "S_BFE_U32 wave_id lift: unsupported source wave size " +
                  Twine(SrcWaveBits) +
                  " (expected 32 or 64); extend the shift-amount dispatch "
                  "before using this path on a new source ISA.");

        unsigned LogWs = (SrcWaveBits == 64) ? 6 : 5;
        Value *Tid = Ctx.Projection.emitWorkitemIdX(Ctx.B);
        Tid->setName("wave_id_lift_tid");
        Value *WaveId = Ctx.B.CreateLShr(
            Tid, ConstantInt::get(Ctx.I32Ty, LogWs), "wave_id_in_wg");
        Value *Masked = Ctx.B.CreateAnd(
            WaveId, ConstantInt::get(Ctx.I32Ty, 0x1F), "wave_id_masked");
        Hr.SccResult = Masked;
        Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Masked);
        Hr.Handled = true;
        return Hr;
      }
    }
    // Generic scalar bitfield-extract.
    Value *Src = Op.src(0), *Ctrl = Op.src(1);
    Value *Offset = Ctx.B.CreateAnd(Ctrl, ConstantInt::get(Ctx.I32Ty, 0x1F));
    Value *Width = Ctx.B.CreateAnd(Ctx.B.CreateLShr(Ctrl, 16),
                                   ConstantInt::get(Ctx.I32Ty, 0x7F));
    Value *SafeWidth =
        Ctx.B.CreateAnd(Width, ConstantInt::get(Ctx.I32Ty, 0x1F));
    Value *Shifted = Ctx.B.CreateLShr(Src, Offset);
    Value *Mask = Ctx.B.CreateSub(
        Ctx.B.CreateShl(ConstantInt::get(Ctx.I32Ty, 1), SafeWidth),
        ConstantInt::get(Ctx.I32Ty, 1));
    Value *IsGE32 = Ctx.B.CreateICmpUGE(Width, ConstantInt::get(Ctx.I32Ty, 32));
    Mask =
        Ctx.B.CreateSelect(IsGE32, ConstantInt::getSigned(Ctx.I32Ty, -1), Mask);
    Value *IsZero = Ctx.B.CreateICmpEQ(Width, ConstantInt::get(Ctx.I32Ty, 0));
    Hr.SccResult = Ctx.B.CreateSelect(IsZero, ConstantInt::get(Ctx.I32Ty, 0),
                                      Ctx.B.CreateAnd(Shifted, Mask, "bfe"));
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  // s_bfe_i32: signed scalar Bit Field Extract.
  //   shift  = ctrl[4:0]
  //   length = ctrl[22:16]
  //   if length == 0: D = 0
  //   elif shift + length < 32:
  //       D = sign_ext((src << (32 - shift - length)) >> (32 - length))
  //   else:
  //       D = (int32)src >> shift   (length saturates to full width)
  // Matches native s_bfe_i32 exactly, including the shift-trick behavior
  // that diverges from a naive "mask and sign-extend from bit (length-1)"
  // implementation when shift + length >= 32.
  //
  // The `shl` / `ashr` amounts can legitimately be out-of-range on the
  // "wrong" side of the isShortEnough select (e.g. `32 - sum` wraps to a
  // huge value when sum >= 32, and `32 - length` is 32 when length == 0).
  // LLVM's select doesn't propagate poison from the unselected branch so
  // it's observationally safe, but we still mask every shift amount to 5
  // bits up front to remove the poison source entirely and keep future
  // optimizer passes from having to prove the guards are sound.
  if (Sop == CanonicalOp::S_BFE_I32) {
    Value *Src = Op.src(0), *Ctrl = Op.src(1);
    Value *C31 = ConstantInt::get(Ctx.I32Ty, 0x1F);
    Value *C32 = ConstantInt::get(Ctx.I32Ty, 32);
    Value *Shift = Ctx.B.CreateAnd(Ctrl, C31);
    Value *Length = Ctx.B.CreateAnd(Ctx.B.CreateLShr(Ctrl, 16),
                                    ConstantInt::get(Ctx.I32Ty, 0x7F));
    Value *Sum = Ctx.B.CreateAdd(Shift, Length);
    Value *IsShortEnough = Ctx.B.CreateICmpULT(Sum, C32);
    Value *ShlAmt = Ctx.B.CreateAnd(Ctx.B.CreateSub(C32, Sum), C31);
    Value *ShiftedLeft = Ctx.B.CreateShl(Src, ShlAmt);
    Value *ShrAmt = Ctx.B.CreateAnd(Ctx.B.CreateSub(C32, Length), C31);
    Value *Sx = Ctx.B.CreateAShr(ShiftedLeft, ShrAmt, "sbfe_i");
    // Fall-through branch (length saturates): arithmetic right shift by
    // `shift` gives "sign-extended src[31:shift]" in a single op.
    Value *Fallthrough = Ctx.B.CreateAShr(Src, Shift, "sbfe_i_sat");
    Value *Computed = Ctx.B.CreateSelect(IsShortEnough, Sx, Fallthrough);
    Value *IsZero = Ctx.B.CreateICmpEQ(Length, ConstantInt::get(Ctx.I32Ty, 0));
    Value *Result =
        Ctx.B.CreateSelect(IsZero, ConstantInt::get(Ctx.I32Ty, 0), Computed);
    // sccResult is an i32; downstream code derives SCC as (sccResult != 0),
    // matching the ISA's "SCC = D != 0" for s_bfe_*.
    Hr.SccResult = Result;
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  // s_bfe_i64: signed scalar Bit Field Extract, 64-bit.
  //   shift  = ctrl[5:0]
  //   length = ctrl[22:16]
  //   if length == 0: D = 0
  //   elif shift + length < 64:
  //       D = sign_ext((src << (64 - shift - length)) >> (64 - length))
  //   else:
  //       D = (int64)src >> shift   (length saturates to full width)
  // 64-bit analogue of S_BFE_I32. Src0 is a 64-bit SGPR pair; src1 is a
  // 32-bit control word in the same SOP2_64_32 shape as S_LSHL_B64. The
  // shift amounts are masked to 6 bits to keep the unselected select arm
  // poison-free (same justification as the I32 handler, just for i64).
  if (Sop == CanonicalOp::S_BFE_I64) {
    Value *Src = Op.src64(0);
    Value *Ctrl = Op.src(1);
    Value *Shift32 = Ctx.B.CreateAnd(Ctrl, ConstantInt::get(Ctx.I32Ty, 0x3F));
    Value *Length32 = Ctx.B.CreateAnd(Ctx.B.CreateLShr(Ctrl, 16),
                                      ConstantInt::get(Ctx.I32Ty, 0x7F));
    Value *Shift = Ctx.B.CreateZExt(Shift32, Ctx.I64Ty);
    Value *Length = Ctx.B.CreateZExt(Length32, Ctx.I64Ty);
    Value *C63 = ConstantInt::get(Ctx.I64Ty, 0x3F);
    Value *C64 = ConstantInt::get(Ctx.I64Ty, 64);
    Value *Sum = Ctx.B.CreateAdd(Shift, Length);
    Value *IsShortEnough = Ctx.B.CreateICmpULT(Sum, C64);
    Value *ShlAmt = Ctx.B.CreateAnd(Ctx.B.CreateSub(C64, Sum), C63);
    Value *ShiftedLeft = Ctx.B.CreateShl(Src, ShlAmt);
    Value *ShrAmt = Ctx.B.CreateAnd(Ctx.B.CreateSub(C64, Length), C63);
    Value *Sx = Ctx.B.CreateAShr(ShiftedLeft, ShrAmt, "sbfe_i64");
    // Fall-through branch (length saturates): arithmetic right shift by
    // `shift` gives "sign-extended src[63:shift]" in a single op.
    Value *Fallthrough = Ctx.B.CreateAShr(Src, Shift, "sbfe_i64_sat");
    Value *Computed = Ctx.B.CreateSelect(IsShortEnough, Sx, Fallthrough);
    Value *IsZero = Ctx.B.CreateICmpEQ(Length, ConstantInt::get(Ctx.I64Ty, 0));
    Value *Result =
        Ctx.B.CreateSelect(IsZero, ConstantInt::get(Ctx.I64Ty, 0), Computed);
    // sccResult is i64; downstream code derives SCC as (sccResult != 0),
    // matching the ISA's "SCC = D != 0" for s_bfe_*.
    Hr.SccResult = Result;
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_PACK_LL_B32_B16) {
    Value *Lo = Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0xFFFF));
    Value *Hi = Ctx.B.CreateShl(
        Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0xFFFF)), 16);
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateOr(Lo, Hi, "pack_ll"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_PACK_LH_B32_B16) {
    Value *Lo = Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0xFFFF));
    Value *Hi =
        Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0xFFFF0000u));
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Ctx.B.CreateOr(Lo, Hi, "pack_lh"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CSELECT_B32) {
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateSelect(Ctx.Regs.loadSCC(Ctx.B), Op.src(0),
                                           Op.src(1), "csel"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CSELECT_B64) {
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(),
                        Ctx.B.CreateSelect(Ctx.Regs.loadSCC(Ctx.B), Op.src64(0),
                                           Op.src64(1), "csel"));
    Hr.Handled = true;
    return Hr;
  }

  // 64-bit SOP2 -- auto SCC via sccResult.
  //
  // S_LSHL_B64 / S_LSHR_B64 / S_ASHR_I64 are `SOP2_64_32`: src1 is a
  // single 32-bit SGPR shift count. AMDGPU masks the count to the low 6
  // bits, so we zext to i64 and mask before shifting -- an unmasked
  // count >= 64 makes the LLVM shift poison.
  const uint64_t ShiftCountMask = (1u << 6) - 1;
  if (Sop == CanonicalOp::S_LSHL_B64) {
    Value *ShiftAmount = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty, "shift_amount");
    ShiftAmount = Ctx.B.CreateAnd(ShiftAmount, Ctx.B.getInt64(ShiftCountMask),
                                  "shift_amount_masked");
    Hr.SccResult = Ctx.B.CreateShl(Op.src64(0), ShiftAmount, "shl64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_LSHR_B64) {
    Value *ShiftAmount = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty, "shift_amount");
    ShiftAmount = Ctx.B.CreateAnd(ShiftAmount, Ctx.B.getInt64(ShiftCountMask),
                                  "shift_amount_masked");
    Hr.SccResult = Ctx.B.CreateLShr(Op.src64(0), ShiftAmount, "lshr64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ASHR_I64) {
    Value *ShiftAmount = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty, "shift_amount");
    ShiftAmount = Ctx.B.CreateAnd(ShiftAmount, Ctx.B.getInt64(ShiftCountMask),
                                  "shift_amount_masked");
    Hr.SccResult = Ctx.B.CreateAShr(Op.src64(0), ShiftAmount, "ashr64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_OR_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Value *Res = Ctx.B.CreateOr(Op.src64(0), Op.src64(1), "or64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Res);
    if (S0I1 && S1I1) {
      Value *OrI1 = Ctx.B.CreateOr(S0I1, S1I1, "wave_mask_or64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), OrI1);
      storeSccFromWaveMaskI1(Ctx, OrI1, "wave_mask_or64_scc");
      Hr.SccHandled = true;
    }
    if (!Hr.SccHandled)
      Hr.SccResult = Res;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_AND_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Value *Res = Ctx.B.CreateAnd(Op.src64(0), Op.src64(1), "and64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Res);
    if (S0I1 && S1I1) {
      Value *AndI1 = Ctx.B.CreateAnd(S0I1, S1I1, "wave_mask_and64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), AndI1);
      storeSccFromWaveMaskI1(Ctx, AndI1, "wave_mask_and64_scc");
      Hr.SccHandled = true;
    }
    if (!Hr.SccHandled)
      Hr.SccResult = Res;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ANDN2_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult =
        Ctx.B.CreateAnd(Op.src64(0), Ctx.B.CreateNot(Op.src64(1)), "andn2_64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *AndN2I1 =
          Ctx.B.CreateAnd(S0I1, Ctx.B.CreateNot(S1I1), "wave_mask_andn2_64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), AndN2I1);
      storeSccFromWaveMaskI1(Ctx, AndN2I1, "wave_mask_andn2_64_scc");
      Hr.SccHandled = true;
    }
    Hr.Handled = true;
    return Hr;
  }
  // Do not use the per-lane mask to set SCC for ORN2. A scalar complement can
  // set bits that are not modeled lane bits, e.g. with 4 lane bits:
  // A=0000, B=1111, A|~B=...11110000. Hardware SCC sees the full scalar
  // result, so it is true even though the modeled lane mask is empty.
  if (Sop == CanonicalOp::S_ORN2_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult =
        Ctx.B.CreateOr(Op.src64(0), Ctx.B.CreateNot(Op.src64(1)), "orn2_64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *OrN2I1 =
          Ctx.B.CreateOr(S0I1, Ctx.B.CreateNot(S1I1), "wave_mask_orn2_64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), OrN2I1);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ANDN2_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult =
        Ctx.B.CreateAnd(Op.src(0), Ctx.B.CreateNot(Op.src(1)), "andn2");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *AndN2I1 =
          Ctx.B.CreateAnd(S0I1, Ctx.B.CreateNot(S1I1), "wave_mask_andn2");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), AndN2I1);
      storeSccFromWaveMaskI1(Ctx, AndN2I1, "wave_mask_andn2_scc");
      Hr.SccHandled = true;
    }
    Hr.Handled = true;
    return Hr;
  }
  // See the ORN2_B64 note above: preserve scalar SCC, but keep the propagated
  // per-lane mask for downstream mask consumers.
  if (Sop == CanonicalOp::S_ORN2_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult =
        Ctx.B.CreateOr(Op.src(0), Ctx.B.CreateNot(Op.src(1)), "orn2");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *OrN2I1 =
          Ctx.B.CreateOr(S0I1, Ctx.B.CreateNot(S1I1), "wave_mask_orn2");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), OrN2I1);
    }
    Hr.Handled = true;
    return Hr;
  }
  // s_{nand,nor,xnor}_b{32,64} -- negated bitops, `dst = ~(src0 OP src1)`.
  // SCC follows writeReg32/64's standard rule (set when result != 0). As with
  // ORN2, the scalar complement can set bits outside the modeled lane mask, and
  // hardware SCC observes those bits. The per-lane mask remains useful for
  // downstream mask consumers.
  // Each opcode uses the same SOP2 operand triplet (sdst, src0, src1)
  // and identical sign-/zero-extension semantics as their non-negated
  // siblings (S_AND_B32 etc.), so we can reuse op.src/op.src64 directly.
  if (Sop == CanonicalOp::S_NAND_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult =
        Ctx.B.CreateNot(Ctx.B.CreateAnd(Op.src(0), Op.src(1), "and"), "nand");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *NandI1 =
          Ctx.B.CreateNot(Ctx.B.CreateAnd(S0I1, S1I1), "wave_mask_nand");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), NandI1);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_NAND_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult = Ctx.B.CreateNot(
        Ctx.B.CreateAnd(Op.src64(0), Op.src64(1), "and64"), "nand64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *NandI1 =
          Ctx.B.CreateNot(Ctx.B.CreateAnd(S0I1, S1I1), "wave_mask_nand64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), NandI1);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_NOR_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult =
        Ctx.B.CreateNot(Ctx.B.CreateOr(Op.src(0), Op.src(1), "or"), "nor");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *NorI1 =
          Ctx.B.CreateNot(Ctx.B.CreateOr(S0I1, S1I1), "wave_mask_nor");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), NorI1);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_NOR_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult = Ctx.B.CreateNot(
        Ctx.B.CreateOr(Op.src64(0), Op.src64(1), "or64"), "nor64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *NorI1 =
          Ctx.B.CreateNot(Ctx.B.CreateOr(S0I1, S1I1), "wave_mask_nor64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), NorI1);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_XNOR_B32) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult =
        Ctx.B.CreateNot(Ctx.B.CreateXor(Op.src(0), Op.src(1), "xor"), "xnor");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *XnorI1 =
          Ctx.B.CreateNot(Ctx.B.CreateXor(S0I1, S1I1), "wave_mask_xnor");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), XnorI1);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_XNOR_B64) {
    Value *S0I1 = tryGetSrcWaveMaskI1(Ctx, Op, 0);
    Value *S1I1 = tryGetSrcWaveMaskI1(Ctx, Op, 1);
    Hr.SccResult = Ctx.B.CreateNot(
        Ctx.B.CreateXor(Op.src64(0), Op.src64(1), "xor64"), "xnor64");
    Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Hr.SccResult);
    if (S0I1 && S1I1) {
      Value *XnorI1 =
          Ctx.B.CreateNot(Ctx.B.CreateXor(S0I1, S1I1), "wave_mask_xnor64");
      recordDerivedWaveMaskI1(Ctx, Op.dst(), XnorI1);
    }
    Hr.Handled = true;
    return Hr;
  }
  // s_absdiff_i32 -- `dst = |src0 - src1|` on signed i32. The hardware
  // wraps for INT_MIN (since `0 - INT_MIN` overflows back to itself in
  // two's complement), so we lower with `llvm.abs.i32(diff, false)` --
  // is_int_min_poison=false matches the wrapping behaviour exactly.
  // SCC follows writeReg32's standard (set when result != 0).
  if (Sop == CanonicalOp::S_ABSDIFF_I32) {
    Value *Diff = Ctx.B.CreateSub(Op.src(0), Op.src(1), "absdiff_sub");
    Value *Res = Ctx.B.CreateIntrinsic(Intrinsic::abs, {Ctx.I32Ty},
                                       {Diff, Ctx.B.getFalse()},
                                       /*FMFSource=*/nullptr, "absdiff");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    Hr.SccResult = Res;
    Hr.Handled = true;
    return Hr;
  }
  return Hr;
}

} // namespace COMGR::hotswap
