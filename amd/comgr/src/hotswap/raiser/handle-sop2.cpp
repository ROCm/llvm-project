//===- handle-sop2.cpp - Hotswap transpiler -------------------------------===//
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

#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace COMGR::hotswap {
namespace {

// Destination and source values for a binary SOP2 instruction.
struct BinaryOperands {
  ParsedReg Dst;
  Value *Src0;
  Value *Src1;
};

// Read the destination and two 32-bit sources of a binary instruction.
Expected<BinaryOperands> readBinary32(OpResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();
  return BinaryOperands{*Dst, *Src0, *Src1};
}

// Read the destination and two 64-bit sources of a binary instruction.
Expected<BinaryOperands> readBinary64(OpResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src64(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src64(1);
  if (!Src1)
    return Src1.takeError();
  return BinaryOperands{*Dst, *Src0, *Src1};
}

// Set SCC if Result is nonzero.
void storeNonzeroScc(RaiseContext &Ctx, Value *Result,
                     const Twine &Name = "scc") {
  Constant *Zero = Constant::getNullValue(Result->getType());
  Value *Nonzero = Ctx.B.CreateICmpNE(Result, Zero, Name);
  Ctx.registers().regFile().storeSCC(Ctx.B, Nonzero);
}

// Raise a shifted 32-bit addition and set SCC on unsigned overflow.
Error handleLshlAdd(RaiseContext &Ctx, OpResolver &Op, unsigned Shift,
                    const Twine &Name) {
  Expected<BinaryOperands> Args = readBinary32(Op);
  if (!Args)
    return Args.takeError();
  Value *Src0 = Ctx.B.CreateZExt(Args->Src0, Ctx.B.getInt64Ty(), Name + "_s0");
  Value *Src1 = Ctx.B.CreateZExt(Args->Src1, Ctx.B.getInt64Ty(), Name + "_s1");
  Value *Shifted = Ctx.B.CreateShl(Src0, Shift, Name + "_shifted");
  Value *Wide = Ctx.B.CreateAdd(Shifted, Src1, Name + "_wide");
  Value *Result = Ctx.B.CreateTrunc(Wide, Ctx.B.getInt32Ty(), Name);
  Ctx.registers().writeReg32(Args->Dst, Result);
  Value *Carry =
      Ctx.B.CreateICmpUGT(Wide, Ctx.B.getInt64(UINT32_MAX), Name + "_carry");
  Ctx.registers().regFile().storeSCC(Ctx.B, Carry);
  return Error::success();
}

// Raise a 32-bit binary instruction, writing the intrinsic result and overflow
// flag to the destination and SCC.
Error handleOverflowingBinary32(RaiseContext &Ctx, OpResolver &Op,
                                Intrinsic::ID IntrinsicID,
                                const Twine &ResultName,
                                const Twine &OverflowName) {
  Expected<BinaryOperands> Args = readBinary32(Op);
  if (!Args)
    return Args.takeError();
  Value *Pair = Ctx.B.CreateIntrinsic(IntrinsicID, {Ctx.B.getInt32Ty()},
                                      {Args->Src0, Args->Src1});
  Value *Result = Ctx.B.CreateExtractValue(Pair, 0, ResultName);
  Value *Overflow = Ctx.B.CreateExtractValue(Pair, 1, OverflowName);
  Ctx.registers().writeReg32(Args->Dst, Result);
  Ctx.registers().regFile().storeSCC(Ctx.B, Overflow);
  return Error::success();
}

// Return an unsupported-instruction failure for Di.
Error unsupported(RaiseContext &Ctx, const DecodedInst &Di) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

} // namespace

// Raise one SOP2 instruction and preserve its SCC side effects.
Error handleSOP2(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::S_ADD_U32:
    return handleOverflowingBinary32(Ctx, Op, Intrinsic::uadd_with_overflow,
                                     "add", "add_carry");
  case CanonicalOp::S_ADD_I32:
    return handleOverflowingBinary32(Ctx, Op, Intrinsic::sadd_with_overflow,
                                     "add", "add_overflow");
  case CanonicalOp::S_SUB_U32:
    return handleOverflowingBinary32(Ctx, Op, Intrinsic::usub_with_overflow,
                                     "sub", "sub_borrow");
  case CanonicalOp::S_SUB_I32:
    return handleOverflowingBinary32(Ctx, Op, Intrinsic::ssub_with_overflow,
                                     "sub", "sub_overflow");
  case CanonicalOp::S_ADDC_U32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Scc = Ctx.registers().regFile().loadSCC(Ctx.B);
    Value *CarryIn = Ctx.B.CreateZExt(Scc, Ctx.B.getInt32Ty(), "carry_in");
    Value *First =
        Ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow,
                              {Ctx.B.getInt32Ty()}, {Args->Src0, Args->Src1});
    Value *Sum = Ctx.B.CreateExtractValue(First, 0);
    Value *Second = Ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow,
                                          {Ctx.B.getInt32Ty()}, {Sum, CarryIn});
    Value *Result = Ctx.B.CreateExtractValue(Second, 0, "addc");
    Value *FirstCarry = Ctx.B.CreateExtractValue(First, 1);
    Value *SecondCarry = Ctx.B.CreateExtractValue(Second, 1);
    Value *Carry = Ctx.B.CreateOr(FirstCarry, SecondCarry, "addc_carry");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Carry);
    return Error::success();
  }
  case CanonicalOp::S_SUBB_U32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Scc = Ctx.registers().regFile().loadSCC(Ctx.B);
    Value *BorrowIn = Ctx.B.CreateZExt(Scc, Ctx.B.getInt32Ty(), "borrow_in");
    Value *First =
        Ctx.B.CreateIntrinsic(Intrinsic::usub_with_overflow,
                              {Ctx.B.getInt32Ty()}, {Args->Src0, Args->Src1});
    Value *Difference = Ctx.B.CreateExtractValue(First, 0);
    Value *Second =
        Ctx.B.CreateIntrinsic(Intrinsic::usub_with_overflow,
                              {Ctx.B.getInt32Ty()}, {Difference, BorrowIn});
    Value *Result = Ctx.B.CreateExtractValue(Second, 0, "subb");
    Value *FirstBorrow = Ctx.B.CreateExtractValue(First, 1);
    Value *SecondBorrow = Ctx.B.CreateExtractValue(Second, 1);
    Value *Borrow = Ctx.B.CreateOr(FirstBorrow, SecondBorrow, "subb_borrow");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Borrow);
    return Error::success();
  }

  case CanonicalOp::S_MUL_I32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Result = Ctx.B.CreateMul(Args->Src0, Args->Src1, "mul");
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::S_MUL_HI_U32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *A = Ctx.B.CreateZExt(Args->Src0, Ctx.B.getInt64Ty());
    Value *B = Ctx.B.CreateZExt(Args->Src1, Ctx.B.getInt64Ty());
    Value *Wide = Ctx.B.CreateMul(A, B, "mulhi_u_wide");
    Value *Shifted = Ctx.B.CreateLShr(Wide, 32);
    Value *High = Ctx.B.CreateTrunc(Shifted, Ctx.B.getInt32Ty(), "mulhi_u");
    Ctx.registers().writeReg32(Args->Dst, High);
    return Error::success();
  }
  case CanonicalOp::S_MUL_HI_I32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *A = Ctx.B.CreateSExt(Args->Src0, Ctx.B.getInt64Ty());
    Value *B = Ctx.B.CreateSExt(Args->Src1, Ctx.B.getInt64Ty());
    Value *Wide = Ctx.B.CreateMul(A, B, "mulhi_i_wide");
    Value *Shifted = Ctx.B.CreateLShr(Wide, 32);
    Value *High = Ctx.B.CreateTrunc(Shifted, Ctx.B.getInt32Ty(), "mulhi_i");
    Ctx.registers().writeReg32(Args->Dst, High);
    return Error::success();
  }
  case CanonicalOp::S_MUL_U64: {
    Expected<BinaryOperands> Args = readBinary64(Op);
    if (!Args)
      return Args.takeError();
    Value *Result = Ctx.B.CreateMul(Args->Src0, Args->Src1, "mul64");
    Ctx.registers().writeReg64(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::S_ADD_NC_U64: {
    Expected<BinaryOperands> Args = readBinary64(Op);
    if (!Args)
      return Args.takeError();
    Value *Result = Ctx.B.CreateAdd(Args->Src0, Args->Src1, "add64");
    Ctx.registers().writeReg64(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::S_SUB_NC_U64: {
    Expected<BinaryOperands> Args = readBinary64(Op);
    if (!Args)
      return Args.takeError();
    Value *Result = Ctx.B.CreateSub(Args->Src0, Args->Src1, "sub64");
    Ctx.registers().writeReg64(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::S_MIN_I32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Condition = Ctx.B.CreateICmpSLT(Args->Src0, Args->Src1);
    Value *Result =
        Ctx.B.CreateSelect(Condition, Args->Src0, Args->Src1, "min");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Condition);
    return Error::success();
  }
  case CanonicalOp::S_MIN_U32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Condition = Ctx.B.CreateICmpULT(Args->Src0, Args->Src1);
    Value *Result =
        Ctx.B.CreateSelect(Condition, Args->Src0, Args->Src1, "min");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Condition);
    return Error::success();
  }
  case CanonicalOp::S_MAX_I32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Condition = Ctx.B.CreateICmpSGE(Args->Src0, Args->Src1);
    Value *Result =
        Ctx.B.CreateSelect(Condition, Args->Src0, Args->Src1, "max");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Condition);
    return Error::success();
  }
  case CanonicalOp::S_MAX_U32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Condition = Ctx.B.CreateICmpUGE(Args->Src0, Args->Src1);
    Value *Result =
        Ctx.B.CreateSelect(Condition, Args->Src0, Args->Src1, "max");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Condition);
    return Error::success();
  }

  case CanonicalOp::S_LSHL1_ADD_U32:
    return handleLshlAdd(Ctx, Op, 1, "lshl1_add");
  case CanonicalOp::S_LSHL2_ADD_U32:
    return handleLshlAdd(Ctx, Op, 2, "lshl2_add");
  case CanonicalOp::S_LSHL3_ADD_U32:
    return handleLshlAdd(Ctx, Op, 3, "lshl3_add");
  case CanonicalOp::S_LSHL4_ADD_U32:
    return handleLshlAdd(Ctx, Op, 4, "lshl4_add");

  case CanonicalOp::S_ABSDIFF_I32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Diff = Ctx.B.CreateSub(Args->Src0, Args->Src1, "absdiff_sub");
    Value *IsNegative = Ctx.B.CreateICmpSLT(Diff, Ctx.B.getInt32(0));
    Value *Negated = Ctx.B.CreateNeg(Diff);
    Value *Result = Ctx.B.CreateSelect(IsNegative, Negated, Diff, "absdiff");
    Ctx.registers().writeReg32(Args->Dst, Result);
    storeNonzeroScc(Ctx, Result);
    return Error::success();
  }

  case CanonicalOp::S_CSELECT_B32: {
    Expected<BinaryOperands> Args = readBinary32(Op);
    if (!Args)
      return Args.takeError();
    Value *Scc = Ctx.registers().regFile().loadSCC(Ctx.B);
    Value *Result = Ctx.B.CreateSelect(Scc, Args->Src0, Args->Src1, "cselect");
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::S_CSELECT_B64: {
    Expected<BinaryOperands> Args = readBinary64(Op);
    if (!Args)
      return Args.takeError();
    Value *Scc = Ctx.registers().regFile().loadSCC(Ctx.B);
    Value *Result =
        Ctx.B.CreateSelect(Scc, Args->Src0, Args->Src1, "cselect64");
    Ctx.registers().writeReg64(Args->Dst, Result);
    return Error::success();
  }

  default:
    return unsupported(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
