//===- handle-vop3.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-mc-tables.h"
#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/raiser/handle-vop-shared.h"
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/raise_failure.h"

#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <climits>
#include <cstdint>

using namespace llvm;

namespace COMGR::hotswap {
namespace {

/// Read the VOP3 clamp operand. Opcodes whose encoding reserves the field have
/// no named operand and are necessarily unclamped.
Expected<bool> readClamp(RaiseContext &Ctx, const DecodedInst &Di) {
  int Idx = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(),
                                               AMDGPU::OpName::clamp);
  if (Idx < 0)
    return false;
  if (!Di.isImm(Idx))
    return unsupportedInstruction(Ctx, Di, "clamp operand is not immediate");
  return Di.getImm(Idx) != 0;
}

/// Reject nonzero output multipliers on integer VOP3 instructions.
Error requireNoOutputMultiplier(RaiseContext &Ctx, const DecodedInst &Di) {
  int Idx = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(),
                                               AMDGPU::OpName::omod);
  if (Idx < 0)
    return Error::success();
  if (!Di.isImm(Idx))
    return unsupportedInstruction(Ctx, Di, "omod operand is not immediate");
  if (Di.getImm(Idx) != 0)
    return unsupportedInstruction(Ctx, Di,
                                  "integer output multiplier is not supported");
  return Error::success();
}

/// Reject source modifiers, which have floating-point rather than integer
/// semantics.
Error requireNoIntegerSourceModifiers(RaiseContext &Ctx, const DecodedInst &Di,
                                      OperandResolver &Op) {
  const unsigned NumSources = Op.nSrcs();
  for (unsigned I = 0; I != NumSources; ++I) {
    if (Op.srcMod(I) != 0)
      return unsupportedInstruction(
          Ctx, Di, "integer source modifiers are not supported");
  }
  return Error::success();
}

/// Read source I at the width selected by Is64.
Expected<Value *> readSrc(OperandResolver &Op, unsigned I, bool Is64) {
  return Is64 ? Op.src64(I) : Op.src(I);
}

/// Raise a wrapping binary integer operation at the selected width.
Error handleBinary(RaiseContext &Ctx, OperandResolver &Op,
                   Instruction::BinaryOps Opcode, bool Is64,
                   bool Reverse = false) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = readSrc(Op, 0, Is64);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = readSrc(Op, 1, Is64);
  if (!Src1)
    return Src1.takeError();
  Value *Lhs = Reverse ? *Src1 : *Src0;
  Value *Rhs = Reverse ? *Src0 : *Src1;
  Value *Result = Ctx.B.CreateBinOp(Opcode, Lhs, Rhs);
  if (Is64)
    Ctx.registers().writeReg64(*Dst, Result);
  else
    Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

/// Write a per-lane carry or borrow result to the instruction's second
/// destination.
Error writeCarryOut(RaiseContext &Ctx, const DecodedInst &Di,
                    OperandResolver &Op, Value *Carry) {
  if (Di.NumDefs != 2 || !Di.isReg(1))
    return unsupportedInstruction(Ctx, Di,
                                  "expected an explicit carry destination");
  Expected<ParsedReg> Dst = Op.dst(1);
  if (!Dst)
    return Dst.takeError();

  Carry = Ctx.B.CreateAnd(Carry, Ctx.registers().emitLaneActiveBit(),
                          "carry.active");
  switch (Dst->RegKind) {
  case ParsedReg::SGPR: {
    Type *MaskTy = Ctx.Projection.sourceWaveMaskTy();
    Value *Mask =
        Ctx.Projection.ballotI1ToWidth(Ctx.B, Carry, MaskTy, "carry.mask");
    Ctx.registers().writeRegExecWidth(*Dst, Mask);
    Ctx.registers().recordWaveMaskI1(*Dst, Carry);
    return Error::success();
  }
  case ParsedReg::VCC:
    Ctx.registers().regFile().storeVCC(Ctx.B, Carry);
    return Error::success();
  case ParsedReg::NOREG:
    return Error::success();
  default:
    return unsupportedInstruction(Ctx, Di,
                                  "invalid carry destination register");
  }
}

/// Raise a binary integer operation with an explicit carry or borrow output.
Error handleBinaryCarry(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op, Intrinsic::ID OverflowIntrinsic,
                        Intrinsic::ID SaturatingIntrinsic, bool Reverse,
                        bool Clamp) {
  if (Di.NumDefs != 2 || Op.nSrcs() < 2)
    return unsupportedInstruction(Ctx, Di,
                                  "expected two destinations and two sources");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();
  Value *Lhs = Reverse ? *Src1 : *Src0;
  Value *Rhs = Reverse ? *Src0 : *Src1;
  Value *Pair = Ctx.B.CreateIntrinsic(OverflowIntrinsic, {Ctx.B.getInt32Ty()},
                                      {Lhs, Rhs});
  Value *Wrapped = Ctx.B.CreateExtractValue(Pair, 0, "result");
  Value *Carry = Ctx.B.CreateExtractValue(Pair, 1, "carry");
  Value *Result =
      Clamp ? Ctx.B.CreateBinaryIntrinsic(SaturatingIntrinsic, Lhs, Rhs)
            : Wrapped;
  Ctx.registers().writeReg32(*Dst, Result);
  return writeCarryOut(Ctx, Di, Op, Carry);
}

/// Raise a binary integer operation with explicit carry or borrow input and
/// output.
Error handleBinaryCarryIn(RaiseContext &Ctx, const DecodedInst &Di,
                          OperandResolver &Op, Intrinsic::ID OverflowIntrinsic,
                          Intrinsic::ID SaturatingIntrinsic, bool Reverse,
                          bool Clamp) {
  if (Di.NumDefs != 2 || Op.nSrcs() < 3)
    return unsupportedInstruction(
        Ctx, Di, "expected two destinations and three sources");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();
  Expected<Value *> CarryInI1 = Op.srcWaveMaskI1(2);
  if (!CarryInI1)
    return CarryInI1.takeError();
  if (!*CarryInI1)
    return unsupportedInstruction(Ctx, Di, "invalid carry-in register");

  Value *Lhs = Reverse ? *Src1 : *Src0;
  Value *Rhs = Reverse ? *Src0 : *Src1;
  Value *CarryIn = Ctx.B.CreateZExt(*CarryInI1, Ctx.B.getInt32Ty(), "carry.in");
  Value *First = Ctx.B.CreateIntrinsic(OverflowIntrinsic, {Ctx.B.getInt32Ty()},
                                       {Lhs, Rhs});
  Value *FirstResult = Ctx.B.CreateExtractValue(First, 0);
  Value *FirstCarry = Ctx.B.CreateExtractValue(First, 1);
  Value *Second = Ctx.B.CreateIntrinsic(OverflowIntrinsic, {Ctx.B.getInt32Ty()},
                                        {FirstResult, CarryIn});
  Value *Wrapped = Ctx.B.CreateExtractValue(Second, 0, "result");
  Value *SecondCarry = Ctx.B.CreateExtractValue(Second, 1);
  Value *Carry = Ctx.B.CreateOr(FirstCarry, SecondCarry, "carry");
  Value *Result = Wrapped;
  if (Clamp) {
    Value *FirstSat =
        Ctx.B.CreateBinaryIntrinsic(SaturatingIntrinsic, Lhs, Rhs);
    Result =
        Ctx.B.CreateBinaryIntrinsic(SaturatingIntrinsic, FirstSat, CarryIn);
  }
  Ctx.registers().writeReg32(*Dst, Result);
  return writeCarryOut(Ctx, Di, Op, Carry);
}

/// Extend the low 24 bits of V to Ty according to the instruction's signedness.
Value *extendLow24(IRBuilder<> &B, Value *V, Type *Ty, bool IsSigned) {
  Value *Low = B.CreateTrunc(V, B.getIntNTy(24), "low24");
  return IsSigned ? B.CreateSExt(Low, Ty, "sext24")
                  : B.CreateZExt(Low, Ty, "zext24");
}

/// Extract the byte at ByteIndex from a packed 64-bit value.
Value *extractByte(IRBuilder<> &B, Value *Packed, Value *ByteIndex,
                   const Twine &Name) {
  constexpr unsigned ByteOffsetShift = 3;
  Value *BitOffset32 =
      B.CreateShl(ByteIndex, B.getInt32(ByteOffsetShift), Name + ".offset");
  Value *BitOffset =
      B.CreateZExt(BitOffset32, B.getInt64Ty(), Name + ".offset64");
  Value *Shifted = B.CreateLShr(Packed, BitOffset, Name + ".shifted");
  return B.CreateTrunc(Shifted, B.getInt8Ty(), Name);
}

/// Select one byte from Packed according to the V_PERM_B32 selector encoding.
/// Selectors 0-7 choose an input byte, 8-11 replicate a sign bit, 12 produces
/// zero, and larger selectors produce all ones.
Value *permuteByte(IRBuilder<> &B, Value *Packed, Value *Selector) {
  constexpr unsigned DataSelectorMask = 7;
  constexpr unsigned FirstSignSelector = 8;
  constexpr unsigned NumSignSelectors = 4;
  constexpr unsigned ZeroSelector = 12;
  constexpr unsigned FirstOnesSelector = 13;

  Value *ByteIndex =
      B.CreateAnd(Selector, B.getInt32(DataSelectorMask), "perm.index");
  Value *DataByte = extractByte(B, Packed, ByteIndex, "perm.data");

  Value *SignSelector =
      B.CreateSub(Selector, B.getInt32(FirstSignSelector), "perm.sign.sel");
  Value *IsSign = B.CreateICmpULT(SignSelector, B.getInt32(NumSignSelectors),
                                  "perm.is.sign");
  Value *SignIndex =
      B.CreateAnd(SignSelector, B.getInt32(NumSignSelectors - 1));
  Value *SignIndexTwice = B.CreateShl(SignIndex, B.getInt32(1));
  Value *SignByteIndex =
      B.CreateAdd(SignIndexTwice, B.getInt32(1), "perm.sign.index");
  Value *SignByte = extractByte(B, Packed, SignByteIndex, "perm.sign.byte");
  Value *SignBit = B.CreateICmpSLT(SignByte, B.getInt8(0), "perm.sign.bit");
  Value *SignFill =
      B.CreateSelect(SignBit, B.getInt8(-1), B.getInt8(0), "perm.sign.fill");
  Value *Result = B.CreateSelect(IsSign, SignFill, DataByte, "perm.sign");

  Value *IsZero =
      B.CreateICmpEQ(Selector, B.getInt32(ZeroSelector), "perm.is.zero");
  Result = B.CreateSelect(IsZero, B.getInt8(0), Result, "perm.zero");
  Value *IsOnes =
      B.CreateICmpUGE(Selector, B.getInt32(FirstOnesSelector), "perm.is.ones");
  return B.CreateSelect(IsOnes, B.getInt8(-1), Result, "perm.byte");
}

/// Raise V_PERM_B32 using generic integer operations.
Value *raiseBytePermute(IRBuilder<> &B, const TernaryOperands &Args) {
  constexpr unsigned BitsPerByte = 8;
  constexpr unsigned BytesPerDword = 4;
  constexpr unsigned HighDwordOffset = 32;
  constexpr unsigned SelectorMask = 0xff;

  Value *Low = B.CreateZExt(Args.Src1, B.getInt64Ty(), "perm.low");
  Value *High = B.CreateZExt(Args.Src0, B.getInt64Ty(), "perm.high");
  High = B.CreateShl(High, B.getInt64(HighDwordOffset), "perm.high.shifted");
  Value *Packed = B.CreateOr(High, Low, "perm.input");

  Value *Result = B.getInt32(0);
  for (unsigned I = 0; I != BytesPerDword; ++I) {
    Value *Selector = Args.Src2;
    if (I != 0)
      Selector =
          B.CreateLShr(Selector, B.getInt32(I * BitsPerByte), "perm.selector");
    Selector =
        B.CreateAnd(Selector, B.getInt32(SelectorMask), "perm.selector.byte");
    Value *Byte = permuteByte(B, Packed, Selector);
    Value *Extended = B.CreateZExt(Byte, B.getInt32Ty(), "perm.extended");
    if (I != 0)
      Extended =
          B.CreateShl(Extended, B.getInt32(I * BitsPerByte), "perm.positioned");
    Result = B.CreateOr(Result, Extended, "perm.result");
  }
  return Result;
}

/// Raise a nested min/max operation over three integer sources.
Error handleTernaryMinMax(RaiseContext &Ctx, OperandResolver &Op,
                          Intrinsic::ID Inner, Intrinsic::ID Outer,
                          StringRef Name) {
  Expected<TernaryOperands> Args = Op.readTernary32();
  if (!Args)
    return Args.takeError();
  Value *First = Ctx.B.CreateBinaryIntrinsic(Inner, Args->Src0, Args->Src1, {},
                                             Name + ".inner");
  Value *Result =
      Ctx.B.CreateBinaryIntrinsic(Outer, First, Args->Src2, {}, Name);
  Ctx.registers().writeReg32(Args->Dst, Result);
  return Error::success();
}

} // namespace

Error handleVOP3(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  if (Error Err = requireNoIntegerSourceModifiers(Ctx, Di, Op))
    return Err;
  if (Error Err = requireNoOutputMultiplier(Ctx, Di))
    return Err;

  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();

  switch (Di.CanonOp) {
  case CanonicalOp::V_MOV_B32:
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "integer move with clamp is not supported");
    return raiseMove32(Ctx, Di, Op);
  case CanonicalOp::V_MOV_B64:
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "integer move with clamp is not supported");
    return raiseMove64(Ctx, Di, Op);

  case CanonicalOp::V_NOT_B32:
  case CanonicalOp::V_BFREV_B32:
  case CanonicalOp::V_FFBH_U32:
  case CanonicalOp::V_FFBL_B32:
  case CanonicalOp::V_FFBH_I32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseUnaryBit32(Ctx, Di, Op);

  case CanonicalOp::V_AND_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAnd(Src0, Src1, "and");
    });
  case CanonicalOp::V_OR_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateOr(Src0, Src1, "or");
    });
  case CanonicalOp::V_XOR_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateXor(Src0, Src1, "xor");
    });
  case CanonicalOp::V_XNOR_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Xor = B.CreateXor(Src0, Src1, "xnor_xor");
      return B.CreateNot(Xor, "xnor");
    });
  case CanonicalOp::V_BFM_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBitMask(Ctx, Op);
  case CanonicalOp::V_BCNT_U32_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBitCount(Ctx, Op);
  case CanonicalOp::V_LSHLREV_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Amount = maskShiftAmount(B, Src0, 32);
      return B.CreateShl(Src1, Amount, "lshl");
    });
  case CanonicalOp::V_LSHRREV_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Amount = maskShiftAmount(B, Src0, 32);
      return B.CreateLShr(Src1, Amount, "lshr");
    });
  case CanonicalOp::V_ASHRREV_I32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Amount = maskShiftAmount(B, Src0, 32);
      return B.CreateAShr(Src1, Amount, "ashr");
    });
  case CanonicalOp::V_LSHLREV_B64:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseShiftLeft64(Ctx, Op);

  case CanonicalOp::V_LSHL_ADD_U32:
  case CanonicalOp::V_ADD_LSHL_U32:
  case CanonicalOp::V_LSHL_OR_B32:
  case CanonicalOp::V_AND_OR_B32:
  case CanonicalOp::V_OR3_B32:
  case CanonicalOp::V_XOR3_B32:
  case CanonicalOp::V_XAD_U32:
  case CanonicalOp::V_ALIGNBIT_B32:
  case CanonicalOp::V_BFE_U32:
  case CanonicalOp::V_BFE_I32:
  case CanonicalOp::V_BFI_B32:
  case CanonicalOp::V_PERM_B32: {
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();

    Value *Result;
    switch (Di.CanonOp) {
    case CanonicalOp::V_LSHL_ADD_U32: {
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src1, 32);
      Value *Shifted = Ctx.B.CreateShl(Args->Src0, Amount);
      Result = Ctx.B.CreateAdd(Shifted, Args->Src2, "lshl.add");
      break;
    }
    case CanonicalOp::V_ADD_LSHL_U32: {
      Value *Sum = Ctx.B.CreateAdd(Args->Src0, Args->Src1);
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src2, 32);
      Result = Ctx.B.CreateShl(Sum, Amount, "add.lshl");
      break;
    }
    case CanonicalOp::V_LSHL_OR_B32: {
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src1, 32);
      Value *Shifted = Ctx.B.CreateShl(Args->Src0, Amount);
      Result = Ctx.B.CreateOr(Shifted, Args->Src2, "lshl.or");
      break;
    }
    case CanonicalOp::V_AND_OR_B32: {
      Value *And = Ctx.B.CreateAnd(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateOr(And, Args->Src2, "and.or");
      break;
    }
    case CanonicalOp::V_OR3_B32: {
      Value *First = Ctx.B.CreateOr(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateOr(First, Args->Src2, "or3");
      break;
    }
    case CanonicalOp::V_XOR3_B32: {
      Value *First = Ctx.B.CreateXor(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateXor(First, Args->Src2, "xor3");
      break;
    }
    case CanonicalOp::V_XAD_U32: {
      Value *Xor = Ctx.B.CreateXor(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateAdd(Xor, Args->Src2, "xad");
      break;
    }
    case CanonicalOp::V_ALIGNBIT_B32: {
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src2, 32);
      Result = Ctx.B.CreateIntrinsic(Intrinsic::fshr, {Ctx.B.getInt32Ty()},
                                     {Args->Src0, Args->Src1, Amount}, nullptr,
                                     "alignbit");
      break;
    }
    case CanonicalOp::V_BFE_U32:
    case CanonicalOp::V_BFE_I32: {
      Value *Offset = maskShiftAmount(Ctx.B, Args->Src1, 32);
      Value *Width = maskShiftAmount(Ctx.B, Args->Src2, 32);
      Value *WidthNonZero =
          Ctx.B.CreateICmpNE(Width, Ctx.B.getInt32(0), "bfe.width.nonzero");
      Value *SafeWidth = Ctx.B.CreateSelect(
          WidthNonZero, Width, Ctx.B.getInt32(1), "bfe.safe.width");
      Value *MaskHighBit = Ctx.B.CreateShl(Ctx.B.getInt32(1), SafeWidth);
      Value *Mask = Ctx.B.CreateSub(MaskHighBit, Ctx.B.getInt32(1), "bfe.mask");
      Value *Shifted = Di.CanonOp == CanonicalOp::V_BFE_I32
                           ? Ctx.B.CreateAShr(Args->Src0, Offset)
                           : Ctx.B.CreateLShr(Args->Src0, Offset);
      Value *Field = Ctx.B.CreateAnd(Shifted, Mask, "bfe.field");
      if (Di.CanonOp == CanonicalOp::V_BFE_I32) {
        Value *SignOffset = Ctx.B.CreateSub(SafeWidth, Ctx.B.getInt32(1));
        Value *SignBit = Ctx.B.CreateShl(Ctx.B.getInt32(1), SignOffset);
        Value *SignFlipped = Ctx.B.CreateXor(Field, SignBit);
        Field = Ctx.B.CreateSub(SignFlipped, SignBit, "bfe.sign.extend");
      }
      Result =
          Ctx.B.CreateSelect(WidthNonZero, Field, Ctx.B.getInt32(0), "bfe");
      break;
    }
    case CanonicalOp::V_BFI_B32: {
      Value *Selected = Ctx.B.CreateAnd(Args->Src0, Args->Src1);
      Value *InvertedMask = Ctx.B.CreateNot(Args->Src0);
      Value *Preserved = Ctx.B.CreateAnd(InvertedMask, Args->Src2);
      Result = Ctx.B.CreateOr(Selected, Preserved, "bfi");
      break;
    }
    case CanonicalOp::V_PERM_B32:
      Result = raiseBytePermute(Ctx.B, *Args);
      break;
    default:
      llvm_unreachable("not a ternary integer bit operation");
    }
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_BITOP3_B32: {
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    if (Op.nSrcs() < 4 || !Di.isImm(Op.srcIdx(3)))
      return unsupportedInstruction(
          Ctx, Di, "expected three data sources and an immediate truth table");
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();

    Value *Not0 = Ctx.B.CreateNot(Args->Src0);
    Value *Not1 = Ctx.B.CreateNot(Args->Src1);
    Value *Not2 = Ctx.B.CreateNot(Args->Src2);
    constexpr unsigned TruthTableSize = 8;
    constexpr uint64_t TruthTableMask = UINT64_C(0xff);
    Value *Minterms[TruthTableSize];
    Value *First = Ctx.B.CreateAnd(Not0, Not1);
    Minterms[0] = Ctx.B.CreateAnd(First, Not2);
    Minterms[1] = Ctx.B.CreateAnd(First, Args->Src2);
    First = Ctx.B.CreateAnd(Not0, Args->Src1);
    Minterms[2] = Ctx.B.CreateAnd(First, Not2);
    Minterms[3] = Ctx.B.CreateAnd(First, Args->Src2);
    First = Ctx.B.CreateAnd(Args->Src0, Not1);
    Minterms[4] = Ctx.B.CreateAnd(First, Not2);
    Minterms[5] = Ctx.B.CreateAnd(First, Args->Src2);
    First = Ctx.B.CreateAnd(Args->Src0, Args->Src1);
    Minterms[6] = Ctx.B.CreateAnd(First, Not2);
    Minterms[7] = Ctx.B.CreateAnd(First, Args->Src2);
    uint64_t TruthTable = static_cast<uint64_t>(Op.srcImm(3)) & TruthTableMask;
    Value *Result = Ctx.B.getInt32(0);
    for (unsigned I = 0; I != TruthTableSize; ++I) {
      if (TruthTable & (UINT64_C(1) << I))
        Result = Ctx.B.CreateOr(Result, Minterms[I], "bitop3");
    }
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_LSHRREV_B64:
  case CanonicalOp::V_ASHRREV_I64: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "64-bit shift does not define clamp");
    if (Op.nSrcs() < 2)
      return unsupportedInstruction(Ctx, Di,
                                    "expected one destination and two sources");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Amount = Op.src(0);
    if (!Amount)
      return Amount.takeError();
    Expected<Value *> Source = Op.src64(1);
    if (!Source)
      return Source.takeError();
    Value *MaskedAmount = maskShiftAmount(Ctx.B, *Amount, 64);
    Value *Shift = Ctx.B.CreateZExt(MaskedAmount, Ctx.B.getInt64Ty());
    Value *Result = Di.CanonOp == CanonicalOp::V_LSHRREV_B64
                        ? Ctx.B.CreateLShr(*Source, Shift, "lshr64")
                        : Ctx.B.CreateAShr(*Source, Shift, "ashr64");
    Ctx.registers().writeReg64(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_LSHL_ADD_U64: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "64-bit shift-add does not define clamp");
    assert(Op.nSrcs() >= 3 && "shift-add instruction must have three sources");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src64(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src64(2);
    if (!S2)
      return S2.takeError();
    constexpr unsigned EncodedShiftValues = 8;
    constexpr unsigned MaxSupportedShift = 4;
    Value *EncodedShift = maskShiftAmount(Ctx.B, *S1, EncodedShiftValues);
    Value *ShiftSupported =
        Ctx.B.CreateICmpULE(EncodedShift, Ctx.B.getInt32(MaxSupportedShift));
    Value *Shift32 = Ctx.B.CreateSelect(ShiftSupported, EncodedShift,
                                        Ctx.B.getInt32(0), "shift.supported");
    Value *Shift = Ctx.B.CreateZExt(Shift32, Ctx.B.getInt64Ty());
    Value *Shifted = Ctx.B.CreateShl(*S0, Shift);
    Value *Result = Ctx.B.CreateAdd(Shifted, *S2, "lshl.add64");
    Ctx.registers().writeReg64(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_NC_U32:
  case CanonicalOp::V_SUB_NC_U32:
  case CanonicalOp::V_SUBREV_NC_U32: {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = Op.src(0);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = Op.src(1);
    if (!Src1)
      return Src1.takeError();
    const bool IsAdd = Di.CanonOp == CanonicalOp::V_ADD_NC_U32;
    const bool Reverse = Di.CanonOp == CanonicalOp::V_SUBREV_NC_U32;
    Value *Lhs = Reverse ? *Src1 : *Src0;
    Value *Rhs = Reverse ? *Src0 : *Src1;
    Value *Result;
    if (*Clamp)
      Result = Ctx.B.CreateBinaryIntrinsic(
          IsAdd ? Intrinsic::uadd_sat : Intrinsic::usub_sat, Lhs, Rhs);
    else
      Result = IsAdd ? Ctx.B.CreateAdd(Lhs, Rhs, "add")
                     : Ctx.B.CreateSub(Lhs, Rhs, "sub");
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_CO_U32:
    return handleBinaryCarry(Ctx, Di, Op, Intrinsic::uadd_with_overflow,
                             Intrinsic::uadd_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUB_CO_U32:
    return handleBinaryCarry(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                             Intrinsic::usub_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUBREV_CO_U32:
    return handleBinaryCarry(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                             Intrinsic::usub_sat, /*Reverse=*/true, *Clamp);
  case CanonicalOp::V_ADD_CO_CI_U32:
    return handleBinaryCarryIn(Ctx, Di, Op, Intrinsic::uadd_with_overflow,
                               Intrinsic::uadd_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUB_CO_CI_U32:
    return handleBinaryCarryIn(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                               Intrinsic::usub_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUBREV_CO_CI_U32:
    return handleBinaryCarryIn(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                               Intrinsic::usub_sat, /*Reverse=*/true, *Clamp);

  case CanonicalOp::V_MIN_I32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smin, Src0, Src1, {}, "min");
    });
  case CanonicalOp::V_MAX_I32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smax, Src0, Src1, {}, "max");
    });
  case CanonicalOp::V_MIN_U32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umin, Src0, Src1, {}, "min");
    });
  case CanonicalOp::V_MAX_U32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umax, Src0, Src1, {}, "max");
    });
  case CanonicalOp::V_MUL_U64:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary64(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateMul(Src0, Src1, "mul64");
    });

  case CanonicalOp::V_MUL_I32_I24:
  case CanonicalOp::V_MUL_HI_I32_I24:
  case CanonicalOp::V_MUL_U32_U24:
  case CanonicalOp::V_MUL_HI_U32_U24: {
    const bool Signed = Di.CanonOp == CanonicalOp::V_MUL_I32_I24 ||
                        Di.CanonOp == CanonicalOp::V_MUL_HI_I32_I24;
    const bool High = Di.CanonOp == CanonicalOp::V_MUL_HI_I32_I24 ||
                      Di.CanonOp == CanonicalOp::V_MUL_HI_U32_U24;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = Op.src(0);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = Op.src(1);
    if (!Src1)
      return Src1.takeError();
    Value *A = extendLow24(Ctx.B, *Src0, Ctx.B.getInt64Ty(), Signed);
    Value *B = extendLow24(Ctx.B, *Src1, Ctx.B.getInt64Ty(), Signed);
    Value *Product = Ctx.B.CreateMul(A, B, "mul24.wide");
    Value *Result;
    if (High) {
      Value *Shifted = Signed ? Ctx.B.CreateAShr(Product, 32, "mul24.high")
                              : Ctx.B.CreateLShr(Product, 32, "mul24.high");
      Result = Ctx.B.CreateTrunc(Shifted, Ctx.B.getInt32Ty(), "mul24.hi");
    } else if (*Clamp) {
      if (Signed) {
        Value *Lo = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MIN);
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MAX);
        Product = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Product, Lo);
        Product = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Product, Hi);
      } else {
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), UINT32_MAX);
        Product = Ctx.B.CreateBinaryIntrinsic(Intrinsic::umin, Product, Hi);
      }
      Result = Ctx.B.CreateTrunc(Product, Ctx.B.getInt32Ty(), "mul24.clamp");
    } else {
      Result = Ctx.B.CreateTrunc(Product, Ctx.B.getInt32Ty(), "mul24");
    }
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_NC_U64:
  case CanonicalOp::V_SUB_NC_U64:
  case CanonicalOp::V_ADD_I32:
  case CanonicalOp::V_SUB_I32: {
    const bool Is64 = Di.CanonOp == CanonicalOp::V_ADD_NC_U64 ||
                      Di.CanonOp == CanonicalOp::V_SUB_NC_U64;
    const bool Signed = !Is64;
    const bool IsAdd = Di.CanonOp == CanonicalOp::V_ADD_NC_U64 ||
                       Di.CanonOp == CanonicalOp::V_ADD_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = readSrc(Op, 0, Is64);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = readSrc(Op, 1, Is64);
    if (!Src1)
      return Src1.takeError();
    Intrinsic::ID Saturating =
        Signed ? (IsAdd ? Intrinsic::sadd_sat : Intrinsic::ssub_sat)
               : (IsAdd ? Intrinsic::uadd_sat : Intrinsic::usub_sat);
    Value *Result =
        *Clamp ? Ctx.B.CreateBinaryIntrinsic(Saturating, *Src0, *Src1)
               : Ctx.B.CreateBinOp(IsAdd ? Instruction::Add : Instruction::Sub,
                                   *Src0, *Src1);
    if (Is64)
      Ctx.registers().writeReg64(*Dst, Result);
    else
      Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MUL_LO_U32:
    // Clamp is reserved for this opcode. Refuse malformed input rather than
    // asserting while decoding an external binary.
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di, "multiply does not define clamp");
    return handleBinary(Ctx, Op, Instruction::Mul, false);
  case CanonicalOp::V_MUL_HI_U32:
  case CanonicalOp::V_MUL_HI_I32: {
    // Clamp is reserved for these opcodes as well.
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di, "multiply does not define clamp");
    const bool Signed = Di.CanonOp == CanonicalOp::V_MUL_HI_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = Op.src(0);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = Op.src(1);
    if (!Src1)
      return Src1.takeError();
    Value *A = Signed ? Ctx.B.CreateSExt(*Src0, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*Src0, Ctx.B.getInt64Ty());
    Value *B = Signed ? Ctx.B.CreateSExt(*Src1, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*Src1, Ctx.B.getInt64Ty());
    Value *Product = Ctx.B.CreateMul(A, B, "mul.wide");
    Value *High = Signed ? Ctx.B.CreateAShr(Product, 32, "mul.high")
                         : Ctx.B.CreateLShr(Product, 32, "mul.high");
    Ctx.registers().writeReg32(
        *Dst, Ctx.B.CreateTrunc(High, Ctx.B.getInt32Ty(), "mul.hi"));
    return Error::success();
  }

  case CanonicalOp::V_MAD_I32_I24:
  case CanonicalOp::V_MAD_U32_U24: {
    const bool Signed = Di.CanonOp == CanonicalOp::V_MAD_I32_I24;
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    Value *Result;
    if (*Clamp) {
      Value *A = extendLow24(Ctx.B, Args->Src0, Ctx.B.getInt64Ty(), Signed);
      Value *B = extendLow24(Ctx.B, Args->Src1, Ctx.B.getInt64Ty(), Signed);
      Value *C = Signed ? Ctx.B.CreateSExt(Args->Src2, Ctx.B.getInt64Ty())
                        : Ctx.B.CreateZExt(Args->Src2, Ctx.B.getInt64Ty());
      Value *Product = Ctx.B.CreateMul(A, B);
      Value *Wide = Ctx.B.CreateAdd(Product, C, "mad.wide");
      if (Signed) {
        Value *Lo = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MIN);
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MAX);
        Wide = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Wide, Lo);
        Wide = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Wide, Hi);
      } else {
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), UINT32_MAX);
        Wide = Ctx.B.CreateBinaryIntrinsic(Intrinsic::umin, Wide, Hi);
      }
      Result = Ctx.B.CreateTrunc(Wide, Ctx.B.getInt32Ty(), "mad.clamp");
    } else {
      Value *A = extendLow24(Ctx.B, Args->Src0, Ctx.B.getInt32Ty(), Signed);
      Value *B = extendLow24(Ctx.B, Args->Src1, Ctx.B.getInt32Ty(), Signed);
      // LLVM has no integer FMA intrinsic; separate integer operations retain
      // the instruction's wrapping multiply-add semantics.
      Value *Product = Ctx.B.CreateMul(A, B);
      Result = Ctx.B.CreateAdd(Product, Args->Src2, "mad24");
    }
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MAD_U32: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "instruction does not define clamp");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src(2);
    if (!S2)
      return S2.takeError();
    Value *Product = Ctx.B.CreateMul(*S0, *S1, "mad.mul");
    Ctx.registers().writeReg32(*Dst, Ctx.B.CreateAdd(Product, *S2, "mad"));
    return Error::success();
  }

  case CanonicalOp::V_ADD3_U32: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "instruction does not define clamp");
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    Value *First = Ctx.B.CreateAdd(Args->Src0, Args->Src1, "add3.first");
    Value *Result = Ctx.B.CreateAdd(First, Args->Src2, "add3");
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_MIN_I32:
  case CanonicalOp::V_ADD_MAX_I32:
  case CanonicalOp::V_ADD_MIN_U32:
  case CanonicalOp::V_ADD_MAX_U32: {
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    const bool Signed = Di.CanonOp == CanonicalOp::V_ADD_MIN_I32 ||
                        Di.CanonOp == CanonicalOp::V_ADD_MAX_I32;
    const bool Max = Di.CanonOp == CanonicalOp::V_ADD_MAX_I32 ||
                     Di.CanonOp == CanonicalOp::V_ADD_MAX_U32;
    Value *Sum = Ctx.B.CreateBinaryIntrinsic(Signed ? Intrinsic::sadd_sat
                                                    : Intrinsic::uadd_sat,
                                             Args->Src0, Args->Src1);
    Intrinsic::ID MinMax = Signed ? (Max ? Intrinsic::smax : Intrinsic::smin)
                                  : (Max ? Intrinsic::umax : Intrinsic::umin);
    Value *Result = Ctx.B.CreateBinaryIntrinsic(MinMax, Sum, Args->Src2);
    if (*Clamp && Signed)
      Result = Ctx.B.CreateBinaryIntrinsic(
          Intrinsic::smax, Result, ConstantInt::get(Ctx.B.getInt32Ty(), 0));
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MIN3_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smin, Intrinsic::smin,
                               "min3");
  case CanonicalOp::V_MAX3_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smax, Intrinsic::smax,
                               "max3");
  case CanonicalOp::V_MIN3_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umin, Intrinsic::umin,
                               "min3");
  case CanonicalOp::V_MAX3_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umax, Intrinsic::umax,
                               "max3");
  case CanonicalOp::V_MED3_I32: {
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    Value *Lo =
        Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Args->Src0, Args->Src1);
    Value *Hi =
        Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Args->Src0, Args->Src1);
    Value *Upper = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Hi, Args->Src2);
    Value *Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Lo, Upper);
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::V_MINMAX_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smin, Intrinsic::smax,
                               "minmax");
  case CanonicalOp::V_MAXMIN_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smax, Intrinsic::smin,
                               "maxmin");
  case CanonicalOp::V_MINMAX_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umin, Intrinsic::umax,
                               "minmax");
  case CanonicalOp::V_MAXMIN_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umax, Intrinsic::umin,
                               "maxmin");

  case CanonicalOp::V_MIN_I64:
  case CanonicalOp::V_MAX_I64:
  case CanonicalOp::V_MIN_U64:
  case CanonicalOp::V_MAX_U64: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "64-bit min/max does not define clamp");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src64(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src64(1);
    if (!S1)
      return S1.takeError();
    Intrinsic::ID ID;
    switch (Di.CanonOp) {
    case CanonicalOp::V_MIN_I64:
      ID = Intrinsic::smin;
      break;
    case CanonicalOp::V_MAX_I64:
      ID = Intrinsic::smax;
      break;
    case CanonicalOp::V_MIN_U64:
      ID = Intrinsic::umin;
      break;
    case CanonicalOp::V_MAX_U64:
      ID = Intrinsic::umax;
      break;
    default:
      llvm_unreachable("not a 64-bit min/max");
    }
    Ctx.registers().writeReg64(*Dst, Ctx.B.CreateBinaryIntrinsic(ID, *S0, *S1));
    return Error::success();
  }

  case CanonicalOp::V_MAD_NC_U64_U32:
  case CanonicalOp::V_MAD_NC_I64_I32: {
    const bool Signed = Di.CanonOp == CanonicalOp::V_MAD_NC_I64_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src64(2);
    if (!S2)
      return S2.takeError();
    Value *A = Signed ? Ctx.B.CreateSExt(*S0, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S0, Ctx.B.getInt64Ty());
    Value *B = Signed ? Ctx.B.CreateSExt(*S1, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S1, Ctx.B.getInt64Ty());
    Value *Product = Ctx.B.CreateMul(A, B, "mad64.mul");
    Value *Result =
        *Clamp ? Ctx.B.CreateBinaryIntrinsic(Signed ? Intrinsic::sadd_sat
                                                    : Intrinsic::uadd_sat,
                                             Product, *S2)
               : Ctx.B.CreateAdd(Product, *S2, "mad64");
    Ctx.registers().writeReg64(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MAD_U64_U32:
  case CanonicalOp::V_MAD_I64_I32: {
    if (Di.NumDefs != 2 || Op.nSrcs() < 3)
      return unsupportedInstruction(
          Ctx, Di, "expected two destinations and three sources");
    const bool Signed = Di.CanonOp == CanonicalOp::V_MAD_I64_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src64(2);
    if (!S2)
      return S2.takeError();
    Value *A = Signed ? Ctx.B.CreateSExt(*S0, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S0, Ctx.B.getInt64Ty());
    Value *B = Signed ? Ctx.B.CreateSExt(*S1, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S1, Ctx.B.getInt64Ty());
    Value *Product = Ctx.B.CreateMul(A, B, "mad64.mul");
    Value *Pair = Ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow,
                                        {Ctx.B.getInt64Ty()}, {Product, *S2});
    Value *Wrapped = Ctx.B.CreateExtractValue(Pair, 0, "mad64");
    Value *Carry = Ctx.B.CreateExtractValue(Pair, 1, "mad64.carry");
    if (Signed) {
      Value *ProductSign =
          Ctx.B.CreateICmpSLT(Product, ConstantInt::get(Ctx.B.getInt64Ty(), 0));
      Value *AccumulatorSign =
          Ctx.B.CreateICmpSLT(*S2, ConstantInt::get(Ctx.B.getInt64Ty(), 0));
      Value *SignsDiffer = Ctx.B.CreateXor(ProductSign, AccumulatorSign);
      Carry = Ctx.B.CreateXor(SignsDiffer, Carry, "mad64.high.bit");
    }
    Value *Result =
        *Clamp ? Ctx.B.CreateBinaryIntrinsic(Signed ? Intrinsic::sadd_sat
                                                    : Intrinsic::uadd_sat,
                                             Product, *S2)
               : Wrapped;
    Ctx.registers().writeReg64(*Dst, Result);
    return writeCarryOut(Ctx, Di, Op, Carry);
  }

  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
