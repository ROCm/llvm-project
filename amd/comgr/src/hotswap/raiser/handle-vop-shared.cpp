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

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/MC/MCRegisterInfo.h"

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

Error raiseUnaryFloat32(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  if (Error Err = Ctx.validateF32Environment(Di))
    return Err;

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> SourceBits = Op.srcF(0);
  if (!SourceBits)
    return SourceBits.takeError();
  Value *Source = Ctx.B.CreateBitCast(*SourceBits, Ctx.B.getFloatTy());

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_FRACT_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_fract, Source,
                                        nullptr, "fract");
    break;
  case CanonicalOp::V_TRUNC_F32:
    Result =
        Ctx.B.CreateUnaryIntrinsic(Intrinsic::trunc, Source, nullptr, "trunc");
    break;
  case CanonicalOp::V_CEIL_F32:
    Result =
        Ctx.B.CreateUnaryIntrinsic(Intrinsic::ceil, Source, nullptr, "ceil");
    break;
  case CanonicalOp::V_RNDNE_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::roundeven, Source, nullptr,
                                        "rndne");
    break;
  case CanonicalOp::V_FLOOR_F32:
    Result =
        Ctx.B.CreateUnaryIntrinsic(Intrinsic::floor, Source, nullptr, "floor");
    break;
  case CanonicalOp::V_EXP_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_exp2, Source, nullptr,
                                        "exp");
    break;
  case CanonicalOp::V_LOG_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_log, Source, nullptr,
                                        "log");
    break;
  case CanonicalOp::V_RCP_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_rcp, Source, nullptr,
                                        "rcp");
    break;
  case CanonicalOp::V_RSQ_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_rsq, Source, nullptr,
                                        "rsq");
    break;
  case CanonicalOp::V_SQRT_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_sqrt, Source, nullptr,
                                        "sqrt");
    break;
  case CanonicalOp::V_SIN_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_sin, Source, nullptr,
                                        "sin");
    break;
  case CanonicalOp::V_COS_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_cos, Source, nullptr,
                                        "cos");
    break;
  case CanonicalOp::V_FREXP_EXP_I32_F32:
    Result = Ctx.B.CreateIntrinsic(Intrinsic::amdgcn_frexp_exp,
                                   {Ctx.B.getInt32Ty(), Ctx.B.getFloatTy()},
                                   {Source}, nullptr, "frexp.exp");
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  case CanonicalOp::V_FREXP_MANT_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_frexp_mant, Source,
                                        nullptr, "frexp.mant");
    break;
  default:
    llvm_unreachable("not a unary F32 operation");
  }

  Value *ResultBits = Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty());
  Ctx.registers().writeReg32(*Dst, ResultBits);
  return Error::success();
}

Error raiseFloatConversion32(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  if (Di.CanonOp == CanonicalOp::V_CVT_F16_F32 &&
      Ctx.Projection.SourceSTI.hasFeature(AMDGPU::FeatureRealTrue16Insts))
    return unsupportedInstruction(
        Ctx, Di, "true16 destination preservation is not supported");
  if (Error Err = Ctx.validateF32Environment(Di))
    return Err;
  if (Di.CanonOp == CanonicalOp::V_CVT_F16_F32) {
    if (Error Err = Ctx.validateF64Environment(Di))
      return Err;
  }

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_CVT_F32_I32:
  case CanonicalOp::V_CVT_F32_U32: {
    Expected<Value *> Source = Op.src(0);
    if (!Source)
      return Source.takeError();
    Value *Converted = Di.CanonOp == CanonicalOp::V_CVT_F32_I32
                           ? Ctx.B.CreateSIToFP(*Source, Ctx.B.getFloatTy())
                           : Ctx.B.CreateUIToFP(*Source, Ctx.B.getFloatTy());
    Result = Ctx.B.CreateBitCast(Converted, Ctx.B.getInt32Ty());
    break;
  }
  case CanonicalOp::V_CVT_I32_F32:
  case CanonicalOp::V_CVT_U32_F32: {
    Expected<Value *> SourceBits = Op.srcF(0);
    if (!SourceBits)
      return SourceBits.takeError();
    Value *Source = Ctx.B.CreateBitCast(*SourceBits, Ctx.B.getFloatTy());
    Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_CVT_I32_F32
                           ? Intrinsic::fptosi_sat
                           : Intrinsic::fptoui_sat;
    Result = Ctx.B.CreateIntrinsic(ID, {Ctx.B.getInt32Ty(), Ctx.B.getFloatTy()},
                                   {Source});
    break;
  }
  case CanonicalOp::V_CVT_F16_F32: {
    Expected<Value *> SourceBits = Op.srcF(0);
    if (!SourceBits)
      return SourceBits.takeError();
    Value *Source = Ctx.B.CreateBitCast(*SourceBits, Ctx.B.getFloatTy());
    Value *Half = Ctx.B.CreateFPTrunc(Source, Ctx.B.getHalfTy(), "cvt");
    Value *HalfBits = Ctx.B.CreateBitCast(Half, Ctx.B.getInt16Ty());
    Result = Ctx.B.CreateZExt(HalfBits, Ctx.B.getInt32Ty());
    break;
  }
  case CanonicalOp::V_CVT_F32_F16: {
    Expected<Value *> SourceBits = Op.src(0);
    if (!SourceBits)
      return SourceBits.takeError();
    unsigned Modifiers = Op.srcMod(0);
    constexpr unsigned AllowedModifiers =
        SISrcMods::NEG | SISrcMods::ABS | SISrcMods::OP_SEL_0;
    if (Modifiers & ~AllowedModifiers)
      return unsupportedInstruction(Ctx, Di, "unsupported source modifier");

    unsigned SourceIndex = Op.srcIdx(0);
    const MCRegisterInfo &MRI = *Ctx.MC.RegInfo;
    bool SourceIsHigh = (Modifiers & SISrcMods::OP_SEL_0) != 0 ||
                        (Di.isReg(SourceIndex) &&
                         AMDGPU::isHi16Reg(Di.getReg(SourceIndex), MRI));
    Value *SelectedBits = *SourceBits;
    if (SourceIsHigh)
      SelectedBits = Ctx.B.CreateLShr(SelectedBits, 16, "cvt.hi");
    Value *LowBits = Ctx.B.CreateTrunc(SelectedBits, Ctx.B.getInt16Ty());
    Value *Half = Ctx.B.CreateBitCast(LowBits, Ctx.B.getHalfTy());
    Half = Op.applyMods(0, Half);
    Value *Converted = Ctx.B.CreateFPExt(Half, Ctx.B.getFloatTy(), "cvt");
    Result = Ctx.B.CreateBitCast(Converted, Ctx.B.getInt32Ty());
    break;
  }
  case CanonicalOp::V_CVT_F32_UBYTE0:
  case CanonicalOp::V_CVT_F32_UBYTE1:
  case CanonicalOp::V_CVT_F32_UBYTE2:
  case CanonicalOp::V_CVT_F32_UBYTE3: {
    Expected<Value *> Source = Op.src(0);
    if (!Source)
      return Source.takeError();
    unsigned ByteIndex;
    switch (Di.CanonOp) {
    case CanonicalOp::V_CVT_F32_UBYTE0:
      ByteIndex = 0;
      break;
    case CanonicalOp::V_CVT_F32_UBYTE1:
      ByteIndex = 1;
      break;
    case CanonicalOp::V_CVT_F32_UBYTE2:
      ByteIndex = 2;
      break;
    case CanonicalOp::V_CVT_F32_UBYTE3:
      ByteIndex = 3;
      break;
    default:
      llvm_unreachable("not a byte conversion");
    }
    Value *Byte = *Source;
    if (ByteIndex != 0)
      Byte = Ctx.B.CreateLShr(Byte, ByteIndex * 8, "cvt.byte.shift");
    Byte = Ctx.B.CreateAnd(Byte, Ctx.B.getInt32(0xff), "cvt.byte");
    Value *Converted = Ctx.B.CreateUIToFP(Byte, Ctx.B.getFloatTy());
    Result = Ctx.B.CreateBitCast(Converted, Ctx.B.getInt32Ty());
    break;
  }
  default:
    llvm_unreachable("not a 32-bit floating-point conversion");
  }

  Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

Error raiseCndMask32(RaiseContext &Ctx, const DecodedInst &Di,
                     OperandResolver &Op) {
  if (Di.NumDefs != 1 || (Op.nSrcs() != 2 && Op.nSrcs() != 3))
    return unsupportedInstruction(
        Ctx, Di, "expected one destination and two values plus a condition");

  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args)
    return Args.takeError();
  Value *Source0 = Op.applyMods(0, Args->Src0);
  Value *Source1 = Op.applyMods(1, Args->Src1);

  Value *Condition;
  if (Op.nSrcs() == 2) {
    Condition = Ctx.registers().regFile().loadVCC(Ctx.B);
  } else {
    Expected<Value *> KnownCondition = Op.srcWaveMaskI1(2);
    if (!KnownCondition)
      return KnownCondition.takeError();
    if (*KnownCondition) {
      Condition = *KnownCondition;
    } else {
      Expected<Value *> Mask = Op.srcExecWidth(2);
      if (!Mask)
        return Mask.takeError();
      Condition = Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, *Mask);
    }
  }

  Value *Result = Ctx.B.CreateSelect(Condition, Source1, Source0, "cndmask");
  Ctx.registers().writeReg32(Args->Dst, Result);
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
