//===- handle-sop2.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace COMGR::hotswap {
namespace {

// Read the destination, a 64-bit source, and a 32-bit source.
Expected<BinaryOperands> readBinary64x32(OpResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src64(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
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

// Set SCC if any lane in MaskI1 is active.
void storeWaveMaskScc(RaiseContext &Ctx, Value *MaskI1, const Twine &Name) {
  Value *Mask = Ctx.Projection.ballotI1ToWidth(
      Ctx.B, MaskI1, Ctx.Projection.waveMaskTy(), Name + "_ballot");
  Value *SourceMask =
      Ctx.Projection.emitCurrentSourceWaveMask(Ctx.B, Mask, Name + "_mask");
  Value *Any = Ctx.B.CreateICmpNE(
      SourceMask, ConstantInt::get(SourceMask->getType(), 0), Name);
  Ctx.registers().regFile().storeSCC(Ctx.B, Any);
}

// A bitwise operation applied to scalar values and wave-mask shadows.
enum class BitOp { And, Or, Xor, AndNot, OrNot, Nand, Nor, Xnor };

// Emit Op for A and Bv.
Value *emitBitOp(IRBuilder<> &B, BitOp Op, Value *A, Value *Bv,
                 const Twine &Name) {
  switch (Op) {
  case BitOp::And:
    return B.CreateAnd(A, Bv, Name);
  case BitOp::Or:
    return B.CreateOr(A, Bv, Name);
  case BitOp::Xor:
    return B.CreateXor(A, Bv, Name);
  case BitOp::AndNot: {
    Value *NotB = B.CreateNot(Bv);
    return B.CreateAnd(A, NotB, Name);
  }
  case BitOp::OrNot: {
    Value *NotB = B.CreateNot(Bv);
    return B.CreateOr(A, NotB, Name);
  }
  case BitOp::Nand: {
    Value *And = B.CreateAnd(A, Bv);
    return B.CreateNot(And, Name);
  }
  case BitOp::Nor: {
    Value *Or = B.CreateOr(A, Bv);
    return B.CreateNot(Or, Name);
  }
  case BitOp::Xnor: {
    Value *Xor = B.CreateXor(A, Bv);
    return B.CreateNot(Xor, Name);
  }
  }
  llvm_unreachable("all bit operations are handled");
}

// Raise a bitwise instruction and preserve wave-mask and SCC state.
Error handleBitOp(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op,
                  BitOp Kind, unsigned Width, const Twine &Name) {
  Expected<Value *> SrcMask0 = Op.srcWaveMaskI1(0);
  if (!SrcMask0)
    return SrcMask0.takeError();
  Expected<Value *> SrcMask1 = Op.srcWaveMaskI1(1);
  if (!SrcMask1)
    return SrcMask1.takeError();

  if (Ctx.Projection.numSourceWavesPerTarget() > 1 &&
      (!*SrcMask0 || !*SrcMask1))
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "bitwise operands lack full-width source-wave mask state");

  Expected<BinaryOperands> Args =
      Width == 64 ? Op.readBinary64() : Op.readBinary32();
  if (!Args)
    return Args.takeError();
  Value *Result = emitBitOp(Ctx.B, Kind, Args->Src0, Args->Src1, Name);
  if (Width == 64)
    Ctx.registers().writeReg64(Args->Dst, Result);
  else
    Ctx.registers().writeReg32(Args->Dst, Result);

  if (*SrcMask0 && *SrcMask1) {
    Value *MaskI1 =
        emitBitOp(Ctx.B, Kind, *SrcMask0, *SrcMask1, Name + "_wave_mask");
    Ctx.registers().recordWaveMaskI1(Args->Dst, MaskI1);
    // Complementing the second operand can set non-lane scalar bits for ORN2,
    // and the fully-negated operations do so unconditionally. Their SCC must
    // therefore be derived from the full scalar result.
    if (Kind == BitOp::And || Kind == BitOp::Or || Kind == BitOp::Xor ||
        Kind == BitOp::AndNot) {
      storeWaveMaskScc(Ctx, MaskI1, Name + "_scc");
      return Error::success();
    }
  }
  storeNonzeroScc(Ctx, Result, Name + "_scc");
  return Error::success();
}

// Raise a 32-bit shift and set SCC if its result is nonzero.
Error handleShift32(RaiseContext &Ctx, OpResolver &Op,
                    Instruction::BinaryOps Opcode, const Twine &Name) {
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args)
    return Args.takeError();
  Value *Amount = Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(31), "shamt");
  Value *Result = Ctx.B.CreateBinOp(Opcode, Args->Src0, Amount, Name);
  Ctx.registers().writeReg32(Args->Dst, Result);
  storeNonzeroScc(Ctx, Result);
  return Error::success();
}

// Raise a 64-bit shift and set SCC if its result is nonzero.
Error handleShift64(RaiseContext &Ctx, OpResolver &Op,
                    Instruction::BinaryOps Opcode, const Twine &Name) {
  Expected<BinaryOperands> Args = readBinary64x32(Op);
  if (!Args)
    return Args.takeError();
  Value *Amount32 =
      Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(63), "shamt64_masked");
  Value *Amount = Ctx.B.CreateZExt(Amount32, Ctx.B.getInt64Ty(), "shamt64");
  Value *Result = Ctx.B.CreateBinOp(Opcode, Args->Src0, Amount, Name);
  Ctx.registers().writeReg64(Args->Dst, Result);
  storeNonzeroScc(Ctx, Result);
  return Error::success();
}

// Raise a shifted 32-bit addition and set SCC on unsigned overflow.
Error handleLshlAdd(RaiseContext &Ctx, OpResolver &Op, unsigned Shift,
                    const Twine &Name) {
  Expected<BinaryOperands> Args = Op.readBinary32();
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
  Expected<BinaryOperands> Args = Op.readBinary32();
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

} // namespace

// Raise one SOP2 instruction and preserve its SCC side effects.
Error handleSOP2(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  switch (Di.Canon.Op) {
  case CanonicalOp::S_AND: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::And, Width,
                       Width == 64 ? "and64" : "and");
  }
  case CanonicalOp::S_OR: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::Or, Width,
                       Width == 64 ? "or64" : "or");
  }
  case CanonicalOp::S_XOR: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::Xor, Width,
                       Width == 64 ? "xor64" : "xor");
  }
  case CanonicalOp::S_ANDN2: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::AndNot, Width,
                       Width == 64 ? "andn2_64" : "andn2");
  }
  case CanonicalOp::S_ORN2: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::OrNot, Width,
                       Width == 64 ? "orn2_64" : "orn2");
  }
  case CanonicalOp::S_NAND: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::Nand, Width,
                       Width == 64 ? "nand64" : "nand");
  }
  case CanonicalOp::S_NOR: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::Nor, Width,
                       Width == 64 ? "nor64" : "nor");
  }
  case CanonicalOp::S_XNOR: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    return handleBitOp(Ctx, Di, Op, BitOp::Xnor, Width,
                       Width == 64 ? "xnor64" : "xnor");
  }

  case CanonicalOp::S_LSHL:
    return canonicalTypeBitWidth(Di.Canon.Type) == 64
               ? handleShift64(Ctx, Op, Instruction::Shl, "shl64")
               : handleShift32(Ctx, Op, Instruction::Shl, "shl");
  case CanonicalOp::S_LSHR:
    return canonicalTypeBitWidth(Di.Canon.Type) == 64
               ? handleShift64(Ctx, Op, Instruction::LShr, "lshr64")
               : handleShift32(Ctx, Op, Instruction::LShr, "lshr");
  case CanonicalOp::S_ASHR:
    return canonicalTypeBitWidth(Di.Canon.Type) == 64
               ? handleShift64(Ctx, Op, Instruction::AShr, "ashr64")
               : handleShift32(Ctx, Op, Instruction::AShr, "ashr");

  case CanonicalOp::S_ADD:
    if (Di.Canon.Type == CanonicalType::U32)
      return handleOverflowingBinary32(Ctx, Op, Intrinsic::uadd_with_overflow,
                                       "add", "add_carry");
    return handleOverflowingBinary32(Ctx, Op, Intrinsic::sadd_with_overflow,
                                     "add", "add_overflow");
  case CanonicalOp::S_SUB:
    if (Di.Canon.Type == CanonicalType::U32)
      return handleOverflowingBinary32(Ctx, Op, Intrinsic::usub_with_overflow,
                                       "sub", "sub_borrow");
    return handleOverflowingBinary32(Ctx, Op, Intrinsic::ssub_with_overflow,
                                     "sub", "sub_overflow");
  case CanonicalOp::S_ADDC: {
    Expected<BinaryOperands> Args = Op.readBinary32();
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
  case CanonicalOp::S_SUBB: {
    Expected<BinaryOperands> Args = Op.readBinary32();
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

  case CanonicalOp::S_MUL: {
    if (Di.Canon.Type == CanonicalType::U64) {
      Expected<BinaryOperands> Args = Op.readBinary64();
      if (!Args)
        return Args.takeError();
      Value *Result = Ctx.B.CreateMul(Args->Src0, Args->Src1, "mul64");
      Ctx.registers().writeReg64(Args->Dst, Result);
      return Error::success();
    }
    Expected<BinaryOperands> Args = Op.readBinary32();
    if (!Args)
      return Args.takeError();
    Value *Result = Ctx.B.CreateMul(Args->Src0, Args->Src1, "mul");
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::S_MUL_HI: {
    Expected<BinaryOperands> Args = Op.readBinary32();
    if (!Args)
      return Args.takeError();
    bool IsSigned = Di.Canon.Type == CanonicalType::I32;
    Value *A = IsSigned ? Ctx.B.CreateSExt(Args->Src0, Ctx.B.getInt64Ty())
                        : Ctx.B.CreateZExt(Args->Src0, Ctx.B.getInt64Ty());
    Value *B = IsSigned ? Ctx.B.CreateSExt(Args->Src1, Ctx.B.getInt64Ty())
                        : Ctx.B.CreateZExt(Args->Src1, Ctx.B.getInt64Ty());
    Value *Wide =
        Ctx.B.CreateMul(A, B, IsSigned ? "mulhi_i_wide" : "mulhi_u_wide");
    Value *Shifted = Ctx.B.CreateLShr(Wide, 32);
    Value *High = Ctx.B.CreateTrunc(Shifted, Ctx.B.getInt32Ty(),
                                    IsSigned ? "mulhi_i" : "mulhi_u");
    Ctx.registers().writeReg32(Args->Dst, High);
    return Error::success();
  }
  case CanonicalOp::S_ADD_NC: {
    Expected<BinaryOperands> Args = Op.readBinary64();
    if (!Args)
      return Args.takeError();
    Value *Result = Ctx.B.CreateAdd(Args->Src0, Args->Src1, "add64");
    Ctx.registers().writeReg64(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::S_SUB_NC: {
    Expected<BinaryOperands> Args = Op.readBinary64();
    if (!Args)
      return Args.takeError();
    Value *Result = Ctx.B.CreateSub(Args->Src0, Args->Src1, "sub64");
    Ctx.registers().writeReg64(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::S_MIN: {
    Expected<BinaryOperands> Args = Op.readBinary32();
    if (!Args)
      return Args.takeError();
    Value *Condition = Di.Canon.Type == CanonicalType::I32
                           ? Ctx.B.CreateICmpSLT(Args->Src0, Args->Src1)
                           : Ctx.B.CreateICmpULT(Args->Src0, Args->Src1);
    Value *Result =
        Ctx.B.CreateSelect(Condition, Args->Src0, Args->Src1, "min");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Condition);
    return Error::success();
  }
  case CanonicalOp::S_MAX: {
    Expected<BinaryOperands> Args = Op.readBinary32();
    if (!Args)
      return Args.takeError();
    Value *Condition = Di.Canon.Type == CanonicalType::I32
                           ? Ctx.B.CreateICmpSGE(Args->Src0, Args->Src1)
                           : Ctx.B.CreateICmpUGE(Args->Src0, Args->Src1);
    Value *Result =
        Ctx.B.CreateSelect(Condition, Args->Src0, Args->Src1, "max");
    Ctx.registers().writeReg32(Args->Dst, Result);
    Ctx.registers().regFile().storeSCC(Ctx.B, Condition);
    return Error::success();
  }
  case CanonicalOp::S_LSHL1_ADD:
    return handleLshlAdd(Ctx, Op, 1, "lshl1_add");
  case CanonicalOp::S_LSHL2_ADD:
    return handleLshlAdd(Ctx, Op, 2, "lshl2_add");
  case CanonicalOp::S_LSHL3_ADD:
    return handleLshlAdd(Ctx, Op, 3, "lshl3_add");
  case CanonicalOp::S_LSHL4_ADD:
    return handleLshlAdd(Ctx, Op, 4, "lshl4_add");

  case CanonicalOp::S_ABSDIFF: {
    Expected<BinaryOperands> Args = Op.readBinary32();
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

  case CanonicalOp::S_BFM: {
    Expected<BinaryOperands> Args = Op.readBinary32();
    if (!Args)
      return Args.takeError();
    unsigned WidthInBits = canonicalTypeBitWidth(Di.Canon.Type);
    Value *Width = Ctx.B.CreateAnd(Args->Src0, Ctx.B.getInt32(WidthInBits - 1));
    Value *Offset =
        Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(WidthInBits - 1));
    IntegerType *ResultType = Ctx.B.getIntNTy(WidthInBits);
    if (WidthInBits == 64) {
      Width = Ctx.B.CreateZExt(Width, ResultType);
      Offset = Ctx.B.CreateZExt(Offset, ResultType);
    }
    Value *OneShifted = Ctx.B.CreateShl(ConstantInt::get(ResultType, 1), Width);
    Value *Mask = Ctx.B.CreateSub(OneShifted, ConstantInt::get(ResultType, 1));
    Value *Result =
        Ctx.B.CreateShl(Mask, Offset, WidthInBits == 64 ? "bfm64" : "bfm32");
    if (WidthInBits == 64)
      Ctx.registers().writeReg64(Args->Dst, Result);
    else
      Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::S_BFE:
    if (Di.Canon.Type == CanonicalType::U32) {
      // gfx12 compute prologues expose wave_id_in_workgroup as ttmp8[29:25].
      if (Op.isSrcReg(0) && !Op.isSrcReg(1)) {
        Expected<std::optional<ParsedReg>> SrcReg = Op.srcReg(0);
        if (!SrcReg)
          return SrcReg.takeError();
        if (*SrcReg && (**SrcReg).RegKind == ParsedReg::TTMP &&
            (**SrcReg).BaseIdx == 8 && Op.srcImm(1) == 0x50019 &&
            Ctx.registers().isTTMP8EntryValueAvailable()) {
          if (!Ctx.Projection.sourceIsa().hasArchitectedSgprs())
            return RaiseFailure::atInstruction(
                RaiseFailureReason::UnsupportedInstructionForm,
                strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
                formatName(Di.TargetSpecificFlags),
                "TTMP8 entry wave ID is unavailable on the source ISA");
          Expected<ParsedReg> Dst = Op.dst();
          if (!Dst)
            return Dst.takeError();
          Value *WaveId = Ctx.Projection.emitSourceWaveId(Ctx.B);
          Value *Result =
              Ctx.B.CreateAnd(WaveId, Ctx.B.getInt32(0x1f), "wave_id_masked");
          Ctx.registers().writeReg32(*Dst, Result);
          storeNonzeroScc(Ctx, Result);
          return Error::success();
        }
      }

      Expected<BinaryOperands> Args = Op.readBinary32();
      if (!Args)
        return Args.takeError();
      Value *Shift = Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(0x1f));
      Value *PackedLength = Ctx.B.CreateLShr(Args->Src1, 16);
      Value *Length = Ctx.B.CreateAnd(PackedLength, Ctx.B.getInt32(0x7f));
      Value *SafeLength = Ctx.B.CreateAnd(Length, Ctx.B.getInt32(0x1f));
      Value *OneShifted = Ctx.B.CreateShl(Ctx.B.getInt32(1), SafeLength);
      Value *Mask = Ctx.B.CreateSub(OneShifted, Ctx.B.getInt32(1));
      Value *IsSaturated = Ctx.B.CreateICmpUGE(Length, Ctx.B.getInt32(32));
      Mask = Ctx.B.CreateSelect(IsSaturated, Ctx.B.getInt32(UINT32_MAX), Mask);
      Value *Shifted = Ctx.B.CreateLShr(Args->Src0, Shift);
      Value *Extract = Ctx.B.CreateAnd(Shifted, Mask);
      Value *IsEmpty = Ctx.B.CreateICmpEQ(Length, Ctx.B.getInt32(0));
      Value *Result =
          Ctx.B.CreateSelect(IsEmpty, Ctx.B.getInt32(0), Extract, "bfe");
      Ctx.registers().writeReg32(Args->Dst, Result);
      storeNonzeroScc(Ctx, Result);
      return Error::success();
    }
    if (Di.Canon.Type == CanonicalType::I32) {
      Expected<BinaryOperands> Args = Op.readBinary32();
      if (!Args)
        return Args.takeError();
      Value *Shift = Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(0x1f));
      Value *PackedLength = Ctx.B.CreateLShr(Args->Src1, 16);
      Value *Length = Ctx.B.CreateAnd(PackedLength, Ctx.B.getInt32(0x7f));
      Value *Sum = Ctx.B.CreateAdd(Shift, Length);
      Value *Short = Ctx.B.CreateICmpULT(Sum, Ctx.B.getInt32(32));
      Value *ShlDistance = Ctx.B.CreateSub(Ctx.B.getInt32(32), Sum);
      Value *ShlAmount = Ctx.B.CreateAnd(ShlDistance, Ctx.B.getInt32(0x1f));
      Value *ShrDistance = Ctx.B.CreateSub(Ctx.B.getInt32(32), Length);
      Value *ShrAmount = Ctx.B.CreateAnd(ShrDistance, Ctx.B.getInt32(0x1f));
      Value *Shifted = Ctx.B.CreateShl(Args->Src0, ShlAmount);
      Value *Extract = Ctx.B.CreateAShr(Shifted, ShrAmount, "bfe_i");
      Value *Saturated = Ctx.B.CreateAShr(Args->Src0, Shift, "bfe_i_sat");
      Value *Nonempty = Ctx.B.CreateSelect(Short, Extract, Saturated);
      Value *IsEmpty = Ctx.B.CreateICmpEQ(Length, Ctx.B.getInt32(0));
      Value *Result = Ctx.B.CreateSelect(IsEmpty, Ctx.B.getInt32(0), Nonempty,
                                         "bfe_i_result");
      Ctx.registers().writeReg32(Args->Dst, Result);
      storeNonzeroScc(Ctx, Result);
      return Error::success();
    }
    if (Di.Canon.Type == CanonicalType::I64) {
      Expected<BinaryOperands> Args = readBinary64x32(Op);
      if (!Args)
        return Args.takeError();
      Value *Shift32 = Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(0x3f));
      Value *PackedLength = Ctx.B.CreateLShr(Args->Src1, 16);
      Value *Length32 = Ctx.B.CreateAnd(PackedLength, Ctx.B.getInt32(0x7f));
      Value *Shift = Ctx.B.CreateZExt(Shift32, Ctx.B.getInt64Ty());
      Value *Length = Ctx.B.CreateZExt(Length32, Ctx.B.getInt64Ty());
      Value *Sum = Ctx.B.CreateAdd(Shift, Length);
      Value *Short = Ctx.B.CreateICmpULT(Sum, Ctx.B.getInt64(64));
      Value *ShlDistance = Ctx.B.CreateSub(Ctx.B.getInt64(64), Sum);
      Value *ShlAmount = Ctx.B.CreateAnd(ShlDistance, Ctx.B.getInt64(0x3f));
      Value *ShrDistance = Ctx.B.CreateSub(Ctx.B.getInt64(64), Length);
      Value *ShrAmount = Ctx.B.CreateAnd(ShrDistance, Ctx.B.getInt64(0x3f));
      Value *Shifted = Ctx.B.CreateShl(Args->Src0, ShlAmount);
      Value *Extract = Ctx.B.CreateAShr(Shifted, ShrAmount, "bfe_i64");
      Value *Saturated = Ctx.B.CreateAShr(Args->Src0, Shift, "bfe_i64_sat");
      Value *Nonempty = Ctx.B.CreateSelect(Short, Extract, Saturated);
      Value *IsEmpty = Ctx.B.CreateICmpEQ(Length, Ctx.B.getInt64(0));
      Value *Result = Ctx.B.CreateSelect(IsEmpty, Ctx.B.getInt64(0), Nonempty,
                                         "bfe_i64_result");
      Ctx.registers().writeReg64(Args->Dst, Result);
      storeNonzeroScc(Ctx, Result);
      return Error::success();
    }
    return unsupportedInstruction(Ctx, Di);

  case CanonicalOp::S_PACK_LL: {
    Expected<BinaryOperands> Args = Op.readBinary32();
    if (!Args)
      return Args.takeError();
    Value *Lo = Ctx.B.CreateAnd(Args->Src0, Ctx.B.getInt32(0xffff));
    Value *Hi16 = Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(0xffff));
    Value *Hi = Ctx.B.CreateShl(Hi16, 16);
    Value *Result = Ctx.B.CreateOr(Lo, Hi, "pack");
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::S_PACK_LH: {
    Expected<BinaryOperands> Args = Op.readBinary32();
    if (!Args)
      return Args.takeError();
    Value *Lo = Ctx.B.CreateAnd(Args->Src0, Ctx.B.getInt32(0xffff));
    Value *Hi = Ctx.B.CreateAnd(Args->Src1, Ctx.B.getInt32(0xffff0000u));
    Value *Result = Ctx.B.CreateOr(Lo, Hi, "pack");
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::S_CSELECT: {
    unsigned Width = canonicalTypeBitWidth(Di.Canon.Type);
    Expected<BinaryOperands> Args =
        Width == 64 ? Op.readBinary64() : Op.readBinary32();
    if (!Args)
      return Args.takeError();
    Value *Scc = Ctx.registers().regFile().loadSCC(Ctx.B);
    Value *Result = Ctx.B.CreateSelect(Scc, Args->Src0, Args->Src1,
                                       Width == 64 ? "cselect64" : "cselect");
    if (Width == 64)
      Ctx.registers().writeReg64(Args->Dst, Result);
    else
      Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
