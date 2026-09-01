//===- handle-sopc.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/Instructions.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace llvm;

namespace COMGR::hotswap {
namespace {

// Return the LLVM predicate for a 32-bit integer comparison.
std::optional<CmpInst::Predicate> integerPredicate(CanonicalOp Opcode) {
  switch (Opcode) {
  case CanonicalOp::S_CMP_EQ_U32:
  case CanonicalOp::S_CMP_EQ_I32:
    return CmpInst::ICMP_EQ;
  case CanonicalOp::S_CMP_LG_U32:
  case CanonicalOp::S_CMP_LG_I32:
    return CmpInst::ICMP_NE;
  case CanonicalOp::S_CMP_GT_U32:
    return CmpInst::ICMP_UGT;
  case CanonicalOp::S_CMP_GE_U32:
    return CmpInst::ICMP_UGE;
  case CanonicalOp::S_CMP_LT_U32:
    return CmpInst::ICMP_ULT;
  case CanonicalOp::S_CMP_LE_U32:
    return CmpInst::ICMP_ULE;
  case CanonicalOp::S_CMP_GT_I32:
    return CmpInst::ICMP_SGT;
  case CanonicalOp::S_CMP_GE_I32:
    return CmpInst::ICMP_SGE;
  case CanonicalOp::S_CMP_LT_I32:
    return CmpInst::ICMP_SLT;
  case CanonicalOp::S_CMP_LE_I32:
    return CmpInst::ICMP_SLE;
  default:
    return std::nullopt;
  }
}

// Return the LLVM predicate for a floating-point comparison.
std::optional<CmpInst::Predicate> floatPredicate(CanonicalOp Opcode) {
  switch (Opcode) {
  case CanonicalOp::S_CMP_EQ_F32:
  case CanonicalOp::S_CMP_EQ_F16:
    return CmpInst::FCMP_OEQ;
  case CanonicalOp::S_CMP_LG_F32:
  case CanonicalOp::S_CMP_LG_F16:
    return CmpInst::FCMP_ONE;
  case CanonicalOp::S_CMP_GT_F32:
  case CanonicalOp::S_CMP_GT_F16:
    return CmpInst::FCMP_OGT;
  case CanonicalOp::S_CMP_GE_F32:
  case CanonicalOp::S_CMP_GE_F16:
    return CmpInst::FCMP_OGE;
  case CanonicalOp::S_CMP_LT_F32:
  case CanonicalOp::S_CMP_LT_F16:
    return CmpInst::FCMP_OLT;
  case CanonicalOp::S_CMP_LE_F32:
  case CanonicalOp::S_CMP_LE_F16:
    return CmpInst::FCMP_OLE;
  case CanonicalOp::S_CMP_NEQ_F32:
  case CanonicalOp::S_CMP_NEQ_F16:
    return CmpInst::FCMP_UNE;
  case CanonicalOp::S_CMP_NGT_F32:
  case CanonicalOp::S_CMP_NGT_F16:
    return CmpInst::FCMP_ULE;
  case CanonicalOp::S_CMP_NGE_F32:
  case CanonicalOp::S_CMP_NGE_F16:
    return CmpInst::FCMP_ULT;
  case CanonicalOp::S_CMP_NLT_F32:
  case CanonicalOp::S_CMP_NLT_F16:
    return CmpInst::FCMP_UGE;
  case CanonicalOp::S_CMP_NLE_F32:
  case CanonicalOp::S_CMP_NLE_F16:
    return CmpInst::FCMP_UGT;
  case CanonicalOp::S_CMP_NLG_F32:
  case CanonicalOp::S_CMP_NLG_F16:
    return CmpInst::FCMP_UEQ;
  case CanonicalOp::S_CMP_O_F32:
  case CanonicalOp::S_CMP_O_F16:
    return CmpInst::FCMP_ORD;
  case CanonicalOp::S_CMP_U_F32:
  case CanonicalOp::S_CMP_U_F16:
    return CmpInst::FCMP_UNO;
  default:
    return std::nullopt;
  }
}

// Return whether Opcode compares 16-bit floating-point values.
bool isFloat16Compare(CanonicalOp Opcode) {
  switch (Opcode) {
  case CanonicalOp::S_CMP_EQ_F16:
  case CanonicalOp::S_CMP_LG_F16:
  case CanonicalOp::S_CMP_GT_F16:
  case CanonicalOp::S_CMP_GE_F16:
  case CanonicalOp::S_CMP_LT_F16:
  case CanonicalOp::S_CMP_LE_F16:
  case CanonicalOp::S_CMP_NEQ_F16:
  case CanonicalOp::S_CMP_NGT_F16:
  case CanonicalOp::S_CMP_NGE_F16:
  case CanonicalOp::S_CMP_NLT_F16:
  case CanonicalOp::S_CMP_NLE_F16:
  case CanonicalOp::S_CMP_NLG_F16:
  case CanonicalOp::S_CMP_O_F16:
  case CanonicalOp::S_CMP_U_F16:
    return true;
  default:
    return false;
  }
}

// Raise a 32-bit integer comparison and write its result to SCC.
Error handleIntegerCompare(RaiseContext &Ctx, OperandResolver &Op,
                           CmpInst::Predicate Pred) {
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();
  Value *Result = Ctx.B.CreateICmp(Pred, *Src0, *Src1, "scmp");
  Ctx.registers().regFile().storeSCC(Ctx.B, Result);
  return Error::success();
}

// Raise a 64-bit integer comparison and write its result to SCC.
Error handleInteger64Compare(RaiseContext &Ctx, OperandResolver &Op,
                             CmpInst::Predicate Pred) {
  Expected<Value *> Src0 = Op.src64(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src64(1);
  if (!Src1)
    return Src1.takeError();
  Value *Result = Ctx.B.CreateICmp(Pred, *Src0, *Src1, "scmp64");
  Ctx.registers().regFile().storeSCC(Ctx.B, Result);
  return Error::success();
}

// Raise a floating-point comparison and write its result to SCC.
Error handleFloatCompare(RaiseContext &Ctx, OperandResolver &Op,
                         CmpInst::Predicate Pred, bool IsF16) {
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();

  Value *Bits0 = *Src0;
  Value *Bits1 = *Src1;
  if (IsF16) {
    Bits0 = Ctx.B.CreateTrunc(Bits0, Ctx.B.getInt16Ty(), "scmpf16_bits");
    Bits1 = Ctx.B.CreateTrunc(Bits1, Ctx.B.getInt16Ty(), "scmpf16_bits");
  }
  assert(Bits0->getType() == Bits1->getType());
  Type *FloatTy = Bits0->getType()->isIntegerTy(16) ? Ctx.B.getHalfTy()
                                                    : Ctx.B.getFloatTy();
  Value *Float0 = Ctx.B.CreateBitCast(Bits0, FloatTy, "scmpf_src");
  Value *Float1 = Ctx.B.CreateBitCast(Bits1, FloatTy, "scmpf_src");
  Value *Result = Ctx.B.CreateFCmp(Pred, Float0, Float1,
                                   FloatTy->isHalfTy() ? "scmpf16" : "scmpf");
  Ctx.registers().regFile().storeSCC(Ctx.B, Result);
  return Error::success();
}

// Raise a 32-bit bit test and write the requested bit value to SCC.
Error handleBitCompare32(RaiseContext &Ctx, OperandResolver &Op,
                         CmpInst::Predicate Pred) {
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();

  Value *Amount =
      Ctx.B.CreateAnd(*Src1, maskTrailingOnes<uint32_t>(5), "bitcmp_shamt");
  Value *Bit = Ctx.B.CreateShl(Ctx.B.getInt32(1), Amount, "bitcmp_bit");
  Value *Masked = Ctx.B.CreateAnd(*Src0, Bit, "bitcmp_mask");
  Value *Scc = Ctx.B.CreateICmp(Pred, Masked, Ctx.B.getInt32(0), "bitcmp");
  Ctx.registers().regFile().storeSCC(Ctx.B, Scc);
  return Error::success();
}

// Raise a 64-bit bit test and write the requested bit value to SCC.
Error handleBitCompare64(RaiseContext &Ctx, OperandResolver &Op,
                         CmpInst::Predicate Pred) {
  Expected<Value *> Src0 = Op.src64(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();

  Value *Amount32 =
      Ctx.B.CreateAnd(*Src1, maskTrailingOnes<uint32_t>(6), "bitcmp_shamt");
  Value *Amount =
      Ctx.B.CreateZExt(Amount32, Ctx.B.getInt64Ty(), "bitcmp_shamt64");
  Value *Bit = Ctx.B.CreateShl(Ctx.B.getInt64(1), Amount, "bitcmp_bit");
  Value *Masked = Ctx.B.CreateAnd(*Src0, Bit, "bitcmp_mask");
  Value *Scc = Ctx.B.CreateICmp(Pred, Masked, Ctx.B.getInt64(0), "bitcmp");
  Ctx.registers().regFile().storeSCC(Ctx.B, Scc);
  return Error::success();
}

} // namespace

// Raise one SOPC instruction and write its comparison result to SCC.
Error handleSOPC(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  if (std::optional<CmpInst::Predicate> Pred = integerPredicate(Di.CanonOp))
    return handleIntegerCompare(Ctx, Op, *Pred);

  if (Di.CanonOp == CanonicalOp::S_CMP_EQ_U64)
    return handleInteger64Compare(Ctx, Op, CmpInst::ICMP_EQ);
  if (Di.CanonOp == CanonicalOp::S_CMP_LG_U64)
    return handleInteger64Compare(Ctx, Op, CmpInst::ICMP_NE);

  if (std::optional<CmpInst::Predicate> Pred = floatPredicate(Di.CanonOp))
    return handleFloatCompare(Ctx, Op, *Pred, isFloat16Compare(Di.CanonOp));

  switch (Di.CanonOp) {
  case CanonicalOp::S_BITCMP0_B32:
    return handleBitCompare32(Ctx, Op, CmpInst::ICMP_EQ);
  case CanonicalOp::S_BITCMP1_B32:
    return handleBitCompare32(Ctx, Op, CmpInst::ICMP_NE);
  case CanonicalOp::S_BITCMP0_B64:
    return handleBitCompare64(Ctx, Op, CmpInst::ICMP_EQ);
  case CanonicalOp::S_BITCMP1_B64:
    return handleBitCompare64(Ctx, Op, CmpInst::ICMP_NE);
  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
