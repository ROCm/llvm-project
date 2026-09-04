//===- handle-vop-shared.cpp - Shared VOP lowering helpers ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handle-vop-shared.h"

#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace COMGR::hotswap {

Error raiseMove32(RaiseContext &Ctx, const DecodedInst &Di,
                  OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() < 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src = Op.src(0);
  if (!Src)
    return Src.takeError();
  Ctx.registers().writeReg32(*Dst, *Src);
  return Error::success();
}

Error raiseMove64(RaiseContext &Ctx, const DecodedInst &Di,
                  OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() < 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src = Op.src64(0);
  if (!Src)
    return Src.takeError();
  Ctx.registers().writeReg64(*Dst, *Src);
  return Error::success();
}

Error raiseUnaryBit32(RaiseContext &Ctx, const DecodedInst &Di,
                      OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() < 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src = Op.src(0);
  if (!Src)
    return Src.takeError();

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_NOT_B32:
    Result = Ctx.B.CreateNot(*Src, "not");
    break;
  case CanonicalOp::V_BFREV_B32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::bitreverse, *Src, nullptr,
                                        "bfrev");
    break;
  case CanonicalOp::V_FFBH_U32:
  case CanonicalOp::V_FFBL_B32: {
    Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_FFBH_U32 ? Intrinsic::ctlz
                                                             : Intrinsic::cttz;
    Value *Count = Ctx.B.CreateIntrinsic(
        ID, {Ctx.B.getInt32Ty()}, {*Src, Ctx.B.getFalse()}, nullptr, "ffb");
    Value *IsZero = Ctx.B.CreateICmpEQ(*Src, Ctx.B.getInt32(0), "ffb.zero");
    Result = Ctx.B.CreateSelect(IsZero, Ctx.B.getInt32(-1), Count, "ffb");
    break;
  }
  case CanonicalOp::V_FFBH_I32: {
    Value *Sign = Ctx.B.CreateAShr(*Src, Ctx.B.getInt32(31), "ffbh.sign");
    Value *Normalized = Ctx.B.CreateXor(*Src, Sign, "ffbh.normalized");
    Value *Count = Ctx.B.CreateIntrinsic(Intrinsic::ctlz, {Ctx.B.getInt32Ty()},
                                         {Normalized, Ctx.B.getFalse()},
                                         nullptr, "ffbh.count");
    Value *AllSignBits =
        Ctx.B.CreateICmpEQ(Normalized, Ctx.B.getInt32(0), "ffbh.uniform");
    Result =
        Ctx.B.CreateSelect(AllSignBits, Ctx.B.getInt32(-1), Count, "ffbh.i32");
    break;
  }
  default:
    llvm_unreachable("not a unary integer bit operation");
  }
  Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

Error raiseBinary32(RaiseContext &Ctx, OperandResolver &Op,
                    BinaryBuilder Build) {
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args)
    return Args.takeError();
  Value *Result = Build(Ctx.B, Args->Src0, Args->Src1);
  Ctx.registers().writeReg32(Args->Dst, Result);
  return Error::success();
}

Error raiseBinary64(RaiseContext &Ctx, OperandResolver &Op,
                    BinaryBuilder Build) {
  Expected<BinaryOperands> Args = Op.readBinary64();
  if (!Args)
    return Args.takeError();
  Value *Result = Build(Ctx.B, Args->Src0, Args->Src1);
  Ctx.registers().writeReg64(Args->Dst, Result);
  return Error::success();
}

Value *maskShiftAmount(IRBuilder<> &B, Value *Amount, unsigned Width) {
  return B.CreateAnd(Amount, ConstantInt::get(Amount->getType(), Width - 1),
                     "shift_amount");
}

Error raiseBitMask(RaiseContext &Ctx, OperandResolver &Op) {
  return raiseBinary32(
      Ctx, Op, [](IRBuilder<> &B, Value *Width, Value *Offset) {
        Width = maskShiftAmount(B, Width, 32);
        Offset = maskShiftAmount(B, Offset, 32);
        Value *HighBit = B.CreateShl(B.getInt32(1), Width);
        Value *Ones = B.CreateSub(HighBit, B.getInt32(1), "bfm.ones");
        return B.CreateShl(Ones, Offset, "bfm");
      });
}

Error raiseBitCount(RaiseContext &Ctx, OperandResolver &Op) {
  return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
    Value *Count =
        B.CreateUnaryIntrinsic(Intrinsic::ctpop, Src0, nullptr, "bcnt");
    return B.CreateAdd(Count, Src1, "bcnt.add");
  });
}

Error raiseShiftLeft64(RaiseContext &Ctx, OperandResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Amount = Op.src(0);
  if (!Amount)
    return Amount.takeError();
  Expected<Value *> Operand = Op.src64(1);
  if (!Operand)
    return Operand.takeError();
  Value *MaskedAmount = maskShiftAmount(Ctx.B, *Amount, 64);
  Value *Shift = Ctx.B.CreateZExt(MaskedAmount, Ctx.B.getInt64Ty(), "shift64");
  Value *Result = Ctx.B.CreateShl(*Operand, Shift, "lshl64");
  Ctx.registers().writeReg64(*Dst, Result);
  return Error::success();
}

} // namespace COMGR::hotswap
