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

std::optional<CmpInst::Predicate> integerPredicate(const CanonicalInst &Inst) {
  bool IsSigned = Inst.Type == CanonicalType::I32;
  bool IsUnsigned = Inst.Type == CanonicalType::U32;
  if (!IsSigned && !IsUnsigned)
    return std::nullopt;

  switch (Inst.Op) {
  case CanonicalOp::S_CMP_EQ:
    return CmpInst::ICMP_EQ;
  case CanonicalOp::S_CMP_LG:
    return CmpInst::ICMP_NE;
  case CanonicalOp::S_CMP_GT:
    return IsSigned ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT;
  case CanonicalOp::S_CMP_GE:
    return IsSigned ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE;
  case CanonicalOp::S_CMP_LT:
    return IsSigned ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
  case CanonicalOp::S_CMP_LE:
    return IsSigned ? CmpInst::ICMP_SLE : CmpInst::ICMP_ULE;
  default:
    return std::nullopt;
  }
}

std::optional<CmpInst::Predicate> floatPredicate(CanonicalOp Opcode) {
  switch (Opcode) {
  case CanonicalOp::S_CMP_EQ:
    return CmpInst::FCMP_OEQ;
  case CanonicalOp::S_CMP_LG:
    return CmpInst::FCMP_ONE;
  case CanonicalOp::S_CMP_GT:
    return CmpInst::FCMP_OGT;
  case CanonicalOp::S_CMP_GE:
    return CmpInst::FCMP_OGE;
  case CanonicalOp::S_CMP_LT:
    return CmpInst::FCMP_OLT;
  case CanonicalOp::S_CMP_LE:
    return CmpInst::FCMP_OLE;
  case CanonicalOp::S_CMP_NEQ:
    return CmpInst::FCMP_UNE;
  case CanonicalOp::S_CMP_NGT:
    return CmpInst::FCMP_ULE;
  case CanonicalOp::S_CMP_NGE:
    return CmpInst::FCMP_ULT;
  case CanonicalOp::S_CMP_NLT:
    return CmpInst::FCMP_UGE;
  case CanonicalOp::S_CMP_NLE:
    return CmpInst::FCMP_UGT;
  case CanonicalOp::S_CMP_NLG:
    return CmpInst::FCMP_UEQ;
  case CanonicalOp::S_CMP_O:
    return CmpInst::FCMP_ORD;
  case CanonicalOp::S_CMP_U:
    return CmpInst::FCMP_UNO;
  default:
    return std::nullopt;
  }
}

// Raise a 32-bit integer comparison and write its result to SCC.
Error handleIntegerCompare(RaiseContext &Ctx, OpResolver &Op,
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
Error handleInteger64Compare(RaiseContext &Ctx, OpResolver &Op,
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
Error handleFloatCompare(RaiseContext &Ctx, OpResolver &Op,
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
Error handleBitCompare32(RaiseContext &Ctx, OpResolver &Op,
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
Error handleBitCompare64(RaiseContext &Ctx, OpResolver &Op,
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
Error handleSOPC(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  if (std::optional<CmpInst::Predicate> Pred = integerPredicate(Di.Canon))
    return handleIntegerCompare(Ctx, Op, *Pred);

  if (Di.Canon.Op == CanonicalOp::S_CMP_EQ &&
      Di.Canon.Type == CanonicalType::U64)
    return handleInteger64Compare(Ctx, Op, CmpInst::ICMP_EQ);
  if (Di.Canon.Op == CanonicalOp::S_CMP_LG &&
      Di.Canon.Type == CanonicalType::U64)
    return handleInteger64Compare(Ctx, Op, CmpInst::ICMP_NE);

  if (Di.Canon.Type == CanonicalType::F16 ||
      Di.Canon.Type == CanonicalType::F32) {
    if (std::optional<CmpInst::Predicate> Pred = floatPredicate(Di.Canon.Op))
      return handleFloatCompare(Ctx, Op, *Pred,
                                Di.Canon.Type == CanonicalType::F16);
  }

  switch (Di.Canon.Op) {
  case CanonicalOp::S_BITCMP0:
  case CanonicalOp::S_BITCMP1: {
    CmpInst::Predicate Pred = Di.Canon.Op == CanonicalOp::S_BITCMP0
                                  ? CmpInst::ICMP_EQ
                                  : CmpInst::ICMP_NE;
    if (Di.Canon.Type == CanonicalType::B32)
      return handleBitCompare32(Ctx, Op, Pred);
    if (Di.Canon.Type == CanonicalType::B64)
      return handleBitCompare64(Ctx, Op, Pred);
    return unsupportedInstruction(Ctx, Di);
  }
  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
