//===- handle-sopc.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

#include "llvm/IR/DerivedTypes.h"

using namespace llvm;

namespace COMGR::hotswap {

HandlerResult handleSOPC(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  // s_set_gpr_idx_on enables GPR dynamic indexing via M0.
  // In scalar model, we store the index value to M0 and treat this as a
  // control-flow nop. The actual VGPR indexing effect is not modeled.
  if (Sop == CanonicalOp::S_SET_GPR_IDX_ON) {
    ParsedReg M0reg;
    M0reg.RegKind = ParsedReg::M0;
    M0reg.BaseIdx = 0;
    Ctx.Regs.writeReg32(Ctx.B, M0reg, Op.src(0));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_SET_GPR_IDX_OFF) {
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_SETVSKIP) {
    Hr.Handled = true;
    return Hr;
  }
  // 64-bit unsigned SOPC compares (gfx8+ S_CMP_EQ_U64 / S_CMP_LG_U64).
  // Both operands are full 64-bit SGPR pairs per SOPInstructions.td's
  // `SOPC_CMP_64` record (the only SOPC compare shape that takes
  // 64-bit operands -- there are no signed or ordered 64-bit SOPC
  // compares on any AMDGPU generation). Read both with `op.src64`
  // and emit a single `icmp eq/ne i64` into SCC.
  //
  // Test back-reference: lit_tests/s_cmp_eq_u64/ pins the `icmp eq
  // i64` shape this lift produces. A regression that narrowed the
  // operands to i32 (a common shortcut against 64-bit SGPR pairs)
  // would break the corpus's per-thread-mask compares used in
  // tensilelite gemm dispatch.
  if (Sop == CanonicalOp::S_CMP_EQ_U64) {
    Value *Cmp64 =
        Ctx.B.CreateICmpEQ(Op.src64(0), Op.src64(1), "scmp64");
    Ctx.Regs.storeSCC(Ctx.B, Cmp64);
    Hr.SccHandled = true;
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CMP_LG_U64) {
    Value *Cmp64 =
        Ctx.B.CreateICmpNE(Op.src64(0), Op.src64(1), "scmp64");
    Ctx.Regs.storeSCC(Ctx.B, Cmp64);
    Hr.SccHandled = true;
    Hr.Handled = true;
    return Hr;
  }

  // SOPC bit-test family (SOPInstructions.td:1411-1414; gfx6+).
  // Shape per the GCN3/RDNA/CDNA ISA references
  // (e.g. RDNA3 ISA ref §4.3.2 "Scalar ALU Operations"):
  //
  //   S_BITCMP0_B32  SCC = (src0 & (1u  << (src1 & 0x1F))) == 0
  //   S_BITCMP1_B32  SCC = (src0 & (1u  << (src1 & 0x1F))) != 0
  //   S_BITCMP0_B64  SCC = (src0 & (1ul << (src1 & 0x3F))) == 0
  //   S_BITCMP1_B64  SCC = (src0 & (1ul << (src1 & 0x3F))) != 0
  //
  // We mask src1 explicitly (rather than relying on shift-wraparound
  // behaviour) so that the IR matches the hardware invariant
  // bit-exactly on every generation -- `shl i32 _, N` for N>=32 and
  // `shl i64 _, N` for N>=64 are UB in LLVM IR, and a corpus kernel
  // that feeds an SGPR with garbage in the high bits would otherwise
  // lift to poison under LLVM's IR rules while the hardware produces
  // a well-defined SCC.
  //
  // Both _0 and _1 variants share the shift-and-mask chain; only the
  // final icmp predicate differs.  A single classifier keeps the
  // parity with the SOPC compare family above (`S_CMP_*`) and makes
  // the four opcodes share exactly one code path.
  {
    bool Is64 = (Sop == CanonicalOp::S_BITCMP0_B64 || Sop == CanonicalOp::S_BITCMP1_B64);
    bool IsB32 = (Sop == CanonicalOp::S_BITCMP0_B32 || Sop == CanonicalOp::S_BITCMP1_B32);
    if (Is64 || IsB32) {
      Value *Src0 = Is64 ? Op.src64(0) : Op.src(0);
      Type *IntTy = Is64 ? Ctx.I64Ty : Ctx.I32Ty;
      uint64_t Mask = Is64 ? 0x3F : 0x1F;

      // Shift amount: src1 is always an SReg_32; mask to 5/6 bits and
      // widen to the src0 width before shifting to keep `shl` in
      // range.
      Value *Shamt = Op.src(1);
      if (Shamt->getType() != Ctx.I32Ty)
        Shamt = Ctx.B.CreateBitOrPointerCast(Shamt, Ctx.I32Ty);
      Shamt = Ctx.B.CreateAnd(Shamt, ConstantInt::get(Ctx.I32Ty, Mask),
                              "bitcmp_shamt");
      if (Is64) Shamt = Ctx.B.CreateZExt(Shamt, Ctx.I64Ty, "bitcmp_shamt64");

      Value *Bit = Ctx.B.CreateShl(ConstantInt::get(IntTy, 1), Shamt,
                                    "bitcmp_bit");
      Value *Masked = Ctx.B.CreateAnd(Src0, Bit, "bitcmp_mask");
      Value *Zero = ConstantInt::get(IntTy, 0);
      bool IsZeroPred =
          (Sop == CanonicalOp::S_BITCMP0_B32 || Sop == CanonicalOp::S_BITCMP0_B64);
      Value *Scc = IsZeroPred
                       ? Ctx.B.CreateICmpEQ(Masked, Zero, "bitcmp0")
                       : Ctx.B.CreateICmpNE(Masked, Zero, "bitcmp1");
      Ctx.Regs.storeSCC(Ctx.B, Scc);
      Hr.SccHandled = true;
      Hr.Handled = true;
      return Hr;
    }
  }

  Value *Src0 = Op.src(0);
  Value *Src1 = Op.src(1);
  Value *Cmp = nullptr;
  if (Sop == CanonicalOp::S_CMP_GT_I32)
    Cmp = Ctx.B.CreateICmpSGT(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_LT_I32)
    Cmp = Ctx.B.CreateICmpSLT(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_GE_I32)
    Cmp = Ctx.B.CreateICmpSGE(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_LE_I32)
    Cmp = Ctx.B.CreateICmpSLE(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_EQ_U32)
    Cmp = Ctx.B.CreateICmpEQ(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_LG_U32)
    Cmp = Ctx.B.CreateICmpNE(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_GE_U32)
    Cmp = Ctx.B.CreateICmpUGE(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_GT_U32)
    Cmp = Ctx.B.CreateICmpUGT(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_LT_U32)
    Cmp = Ctx.B.CreateICmpULT(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_LE_U32)
    Cmp = Ctx.B.CreateICmpULE(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_EQ_I32)
    Cmp = Ctx.B.CreateICmpEQ(Src0, Src1, "scmp");
  else if (Sop == CanonicalOp::S_CMP_LG_I32)
    Cmp = Ctx.B.CreateICmpNE(Src0, Src1, "scmp");
  // GFX12 scalar FP compares (ordered and unordered variants)
  else if (Sop >= CanonicalOp::S_CMP_EQ_F32 && Sop <= CanonicalOp::S_CMP_NLG_F32) {
    Value *F0 = Ctx.B.CreateBitCast(Src0, Ctx.F32Ty);
    Value *F1 = Ctx.B.CreateBitCast(Src1, Ctx.F32Ty);
    if (Sop == CanonicalOp::S_CMP_EQ_F32)
      Cmp = Ctx.B.CreateFCmpOEQ(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_LG_F32)
      Cmp = Ctx.B.CreateFCmpONE(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_GT_F32)
      Cmp = Ctx.B.CreateFCmpOGT(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_GE_F32)
      Cmp = Ctx.B.CreateFCmpOGE(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_LT_F32)
      Cmp = Ctx.B.CreateFCmpOLT(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_LE_F32)
      Cmp = Ctx.B.CreateFCmpOLE(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_NEQ_F32)
      Cmp = Ctx.B.CreateFCmpUNE(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_NLT_F32)
      Cmp = Ctx.B.CreateFCmpUGE(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_NLE_F32)
      Cmp = Ctx.B.CreateFCmpUGT(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_NGT_F32)
      Cmp = Ctx.B.CreateFCmpULE(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_NGE_F32)
      Cmp = Ctx.B.CreateFCmpULT(F0, F1, "scmpf");
    else if (Sop == CanonicalOp::S_CMP_NLG_F32)
      Cmp = Ctx.B.CreateFCmpUEQ(F0, F1, "scmpf");
  } else if (Sop >= CanonicalOp::S_CMP_EQ_F16 && Sop <= CanonicalOp::S_CMP_NLG_F16) {
    Type *F16Ty = Type::getHalfTy(Ctx.C);
    Value *F0 = Ctx.B.CreateBitCast(
        Ctx.B.CreateTrunc(Src0, Type::getInt16Ty(Ctx.C)), F16Ty);
    Value *F1 = Ctx.B.CreateBitCast(
        Ctx.B.CreateTrunc(Src1, Type::getInt16Ty(Ctx.C)), F16Ty);
    if (Sop == CanonicalOp::S_CMP_EQ_F16)
      Cmp = Ctx.B.CreateFCmpOEQ(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_LG_F16)
      Cmp = Ctx.B.CreateFCmpONE(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_GT_F16)
      Cmp = Ctx.B.CreateFCmpOGT(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_GE_F16)
      Cmp = Ctx.B.CreateFCmpOGE(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_LT_F16)
      Cmp = Ctx.B.CreateFCmpOLT(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_LE_F16)
      Cmp = Ctx.B.CreateFCmpOLE(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_NEQ_F16)
      Cmp = Ctx.B.CreateFCmpUNE(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_NLT_F16)
      Cmp = Ctx.B.CreateFCmpUGE(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_NLE_F16)
      Cmp = Ctx.B.CreateFCmpUGT(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_NGT_F16)
      Cmp = Ctx.B.CreateFCmpULE(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_NGE_F16)
      Cmp = Ctx.B.CreateFCmpULT(F0, F1, "scmpf16");
    else if (Sop == CanonicalOp::S_CMP_NLG_F16)
      Cmp = Ctx.B.CreateFCmpUEQ(F0, F1, "scmpf16");
  }
  if (Cmp) {
    Ctx.Regs.storeSCC(Ctx.B, Cmp);
    Hr.SccHandled = true;
    Hr.Handled = true;
    return Hr;
  }
  return Hr;
}

} // namespace COMGR::hotswap
