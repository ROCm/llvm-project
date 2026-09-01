//===- handle-vop2.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>

using namespace llvm;

namespace COMGR::hotswap {

// Builds the result of a two-source instruction from its already-read sources.
using BinaryBuilder = function_ref<Value *(IRBuilder<> &, Value *, Value *)>;

static Error raiseFloatBinary(RaiseContext &Ctx, const DecodedInst &Di,
                              OperandResolver &Op,
                              Instruction::BinaryOps Opcode,
                              bool ReverseOperands) {
  if (Di.NumDefs != 1 || Di.numOperands() == 0 || !Di.isReg(0) ||
      Op.nSrcs() != 2) {
    return unsupported(Ctx, Di,
                       "expected one register destination and two sources");
  }

  if (Error Err = Ctx.validateF32Environment(Di)) {
    return Err;
  }

  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args) {
    return Args.takeError();
  }

  Value *Src0 = Ctx.B.CreateBitCast(Args->Src0, Ctx.B.getFloatTy());
  Value *Src1 = Ctx.B.CreateBitCast(Args->Src1, Ctx.B.getFloatTy());
  Value *Lhs = ReverseOperands ? Src1 : Src0;
  Value *Rhs = ReverseOperands ? Src0 : Src1;
  Value *Result = Ctx.B.CreateBinOp(Opcode, Lhs, Rhs);
  Ctx.registers().writeReg32(Args->Dst,
                             Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty()));
  return Error::success();
}

// Build a 32-bit result from two integer sources and write it to the
// destination.
static Error raiseBinary32(RaiseContext &Ctx, OperandResolver &Op,
                           BinaryBuilder Build) {
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args) {
    return Args.takeError();
  }
  Ctx.registers().writeReg32(Args->Dst, Build(Ctx.B, Args->Src0, Args->Src1));
  return Error::success();
}

// Raise a low-16-bit binary operation and zero-extend its result to 32 bits.
static Error raiseBinary16(RaiseContext &Ctx, OperandResolver &Op,
                           BinaryBuilder Build) {
  return raiseBinary32(Ctx, Op, [&](IRBuilder<> &B, Value *Src0, Value *Src1) {
    Type *I16Ty = B.getInt16Ty();
    Value *Lhs = B.CreateTrunc(Src0, I16Ty, "src0_i16");
    Value *Rhs = B.CreateTrunc(Src1, I16Ty, "src1_i16");
    return B.CreateZExt(Build(B, Lhs, Rhs), B.getInt32Ty(), "result_i32");
  });
}

// Write an EXEC-predicated vector result and store carry or borrow in VCC for
// active source lanes.
static void writeResultAndVCC(RaiseContext &Ctx, ParsedReg Dst, Value *Result,
                              Value *VCC) {
  Value *LaneActive = Ctx.registers().emitLaneActiveBit();
  Ctx.registers().writeReg32(Dst, Result);
  Ctx.registers().regFile().storeVCC(
      Ctx.B, Ctx.B.CreateAnd(LaneActive, VCC, "vcc_active"));
}

// Raise a binary operation that writes carry or borrow to VCC.
static Error raiseBinary32WriteVCC(RaiseContext &Ctx, OperandResolver &Op,
                                   Intrinsic::ID IntrinsicID,
                                   bool ReverseOperands) {
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args) {
    return Args.takeError();
  }
  Value *Lhs = ReverseOperands ? Args->Src1 : Args->Src0;
  Value *Rhs = ReverseOperands ? Args->Src0 : Args->Src1;
  Value *Pair =
      Ctx.B.CreateIntrinsic(IntrinsicID, {Ctx.B.getInt32Ty()}, {Lhs, Rhs});
  Value *Result = Ctx.B.CreateExtractValue(Pair, 0, "result");
  Value *VCC = Ctx.B.CreateExtractValue(Pair, 1, "vcc_out");
  writeResultAndVCC(Ctx, Args->Dst, Result, VCC);
  return Error::success();
}

// Raise a binary operation that reads carry or borrow from VCC and writes the
// updated carry or borrow back to VCC.
static Error raiseBinary32ReadWriteVCC(RaiseContext &Ctx, OperandResolver &Op,
                                       Intrinsic::ID IntrinsicID,
                                       bool ReverseOperands) {
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args) {
    return Args.takeError();
  }
  Value *Lhs = ReverseOperands ? Args->Src1 : Args->Src0;
  Value *Rhs = ReverseOperands ? Args->Src0 : Args->Src1;
  Value *VCCIn = Ctx.B.CreateZExt(Ctx.registers().regFile().loadVCC(Ctx.B),
                                  Ctx.B.getInt32Ty(), "vcc_in");
  Value *First =
      Ctx.B.CreateIntrinsic(IntrinsicID, {Ctx.B.getInt32Ty()}, {Lhs, Rhs});
  Value *Intermediate = Ctx.B.CreateExtractValue(First, 0);
  Value *Second = Ctx.B.CreateIntrinsic(IntrinsicID, {Ctx.B.getInt32Ty()},
                                        {Intermediate, VCCIn});
  Value *Result = Ctx.B.CreateExtractValue(Second, 0, "result");
  Value *FirstVCC = Ctx.B.CreateExtractValue(First, 1);
  Value *SecondVCC = Ctx.B.CreateExtractValue(Second, 1);
  Value *VCC = Ctx.B.CreateOr(FirstVCC, SecondVCC, "vcc_out");
  writeResultAndVCC(Ctx, Args->Dst, Result, VCC);
  return Error::success();
}

// Select src1 when the current lane's VCC bit is set, otherwise src0.
static Error raiseCndMask(RaiseContext &Ctx, OperandResolver &Op) {
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args) {
    return Args.takeError();
  }
  Value *VCC = Ctx.registers().regFile().loadVCC(Ctx.B);
  Value *Result = Ctx.B.CreateSelect(VCC, Args->Src1, Args->Src0, "cndmask");
  Ctx.registers().writeReg32(Args->Dst, Result);
  return Error::success();
}

// Accumulate signed products of packed source elements, using the destination's
// incoming value as the accumulator.
static Error raiseSignedDotAccumulate(RaiseContext &Ctx, OperandResolver &Op,
                                      unsigned ElementWidthInBits) {
  assert(Op.nSrcs() == 3 && "dot accumulate must have three sources");
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args) {
    return Args.takeError();
  }
  Expected<Value *> AccumulatorSource = Op.src(2);
  if (!AccumulatorSource) {
    return AccumulatorSource.takeError();
  }
  // Both sources pack their elements into a single 32-bit VGPR, so the number
  // of elements to accumulate follows from the element width.
  constexpr unsigned PackedWidthInBits = 32;
  unsigned NumElements = PackedWidthInBits / ElementWidthInBits;

  IntegerType *ElementTy = Ctx.B.getIntNTy(ElementWidthInBits);
  IntegerType *AccumulatorTy = Ctx.B.getInt32Ty();
  Value *Accumulator = *AccumulatorSource;
  for (unsigned I = 0; I != NumElements; ++I) {
    unsigned BitOffset = I * ElementWidthInBits;
    Value *LhsBits = Ctx.B.CreateLShr(Args->Src0, BitOffset, "dot_lhs_shifted");
    Value *LhsElement = Ctx.B.CreateTrunc(LhsBits, ElementTy, "dot_lhs_bits");
    Value *RhsBits = Ctx.B.CreateLShr(Args->Src1, BitOffset, "dot_rhs_shifted");
    Value *RhsElement = Ctx.B.CreateTrunc(RhsBits, ElementTy, "dot_rhs_bits");
    Value *Lhs = Ctx.B.CreateSExt(LhsElement, AccumulatorTy, "dot_lhs");
    Value *Rhs = Ctx.B.CreateSExt(RhsElement, AccumulatorTy, "dot_rhs");
    Value *Product = Ctx.B.CreateMul(Lhs, Rhs, "dot_product");
    Accumulator = Ctx.B.CreateAdd(Accumulator, Product, "dot_accumulate");
  }
  Ctx.registers().writeReg32(Args->Dst, Accumulator);
  return Error::success();
}

// Build a 64-bit result from two integer sources and write it to the
// destination.
static Error raiseBinary64(RaiseContext &Ctx, OperandResolver &Op,
                           BinaryBuilder Build) {
  Expected<BinaryOperands> Args = Op.readBinary64();
  if (!Args) {
    return Args.takeError();
  }
  Ctx.registers().writeReg64(Args->Dst, Build(Ctx.B, Args->Src0, Args->Src1));
  return Error::success();
}

// Reduce a shift amount to the low bits the hardware reads: `S0[4:0]` for a
// 32-bit shift and `S0[5:0]` for a 64-bit one. The mask is not redundant: an
// LLVM shift is poison once the amount reaches the operand width, where the
// hardware wraps instead.
static Value *maskShiftAmount(IRBuilder<> &B, Value *Amount, unsigned Width) {
  return B.CreateAnd(Amount, ConstantInt::get(Amount->getType(), Width - 1),
                     "shift_amount");
}

// Widen the low 24 bits of a source to `Ty`, which is how the `*_i24` and
// `*_u24` multiplies read their operands.
static Value *extendLow24(IRBuilder<> &B, Value *Source, Type *Ty,
                          bool IsSigned) {
  Value *Narrow = B.CreateTrunc(Source, B.getIntNTy(24), "narrow24");
  return IsSigned ? B.CreateSExt(Narrow, Ty, "sext24")
                  : B.CreateZExt(Narrow, Ty, "zext24");
}

// Raise a 24-bit multiply returning the low 32 bits of the product.
static Error raiseMul24(RaiseContext &Ctx, OperandResolver &Op, bool IsSigned) {
  return raiseBinary32(Ctx, Op,
                       [IsSigned](IRBuilder<> &B, Value *Src0, Value *Src1) {
                         Type *I32Ty = B.getInt32Ty();
                         Value *Lhs = extendLow24(B, Src0, I32Ty, IsSigned);
                         Value *Rhs = extendLow24(B, Src1, I32Ty, IsSigned);
                         return B.CreateMul(Lhs, Rhs, "mul24");
                       });
}

// Raise a 24-bit multiply returning bits [63:32] of the sign- or
// zero-extended 64-bit product.
static Error raiseMulHi24(RaiseContext &Ctx, OperandResolver &Op,
                          bool IsSigned) {
  return raiseBinary32(
      Ctx, Op, [IsSigned](IRBuilder<> &B, Value *Src0, Value *Src1) {
        Type *I64Ty = B.getInt64Ty();
        Value *Lhs = extendLow24(B, Src0, I64Ty, IsSigned);
        Value *Rhs = extendLow24(B, Src1, I64Ty, IsSigned);
        Value *Wide = B.CreateMul(Lhs, Rhs, "mul24_wide");
        Value *High = IsSigned ? B.CreateAShr(Wide, 32, "mul24_high")
                               : B.CreateLShr(Wide, 32, "mul24_high");
        return B.CreateTrunc(High, B.getInt32Ty(), "mul_hi24");
      });
}

// Raise `v_lshlrev_b64`, whose src0 is a 32-bit shift amount while its src1
// and destination are 64 bits.
static Error raiseShiftLeft64(RaiseContext &Ctx, OperandResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst) {
    return Dst.takeError();
  }
  Expected<Value *> Amount = Op.src(0);
  if (!Amount) {
    return Amount.takeError();
  }
  Expected<Value *> Operand = Op.src64(1);
  if (!Operand) {
    return Operand.takeError();
  }
  Value *Shift = Ctx.B.CreateZExt(maskShiftAmount(Ctx.B, *Amount, 64),
                                  Ctx.B.getInt64Ty(), "shift64");
  Ctx.registers().writeReg64(*Dst, Ctx.B.CreateShl(*Operand, Shift, "lshl64"));
  return Error::success();
}

Error handleVOP2(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::V_ADD_F32:
    return raiseFloatBinary(Ctx, Di, Op, Instruction::FAdd,
                            /*ReverseOperands=*/false);
  case CanonicalOp::V_MUL_F32:
    return raiseFloatBinary(Ctx, Di, Op, Instruction::FMul,
                            /*ReverseOperands=*/false);
  case CanonicalOp::V_SUB_F32:
    return raiseFloatBinary(Ctx, Di, Op, Instruction::FSub,
                            /*ReverseOperands=*/false);
  case CanonicalOp::V_SUBREV_F32:
    return raiseFloatBinary(Ctx, Di, Op, Instruction::FSub,
                            /*ReverseOperands=*/true);

  case CanonicalOp::V_ADD_NC_U32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAdd(Src0, Src1, "add");
    });
  case CanonicalOp::V_SUB_NC_U32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateSub(Src0, Src1, "sub");
    });
  case CanonicalOp::V_SUBREV_NC_U32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateSub(Src1, Src0, "subrev");
    });
  case CanonicalOp::V_ADD_CO_U32:
    return raiseBinary32WriteVCC(Ctx, Op, Intrinsic::uadd_with_overflow,
                                 /*ReverseOperands=*/false);
  case CanonicalOp::V_SUB_CO_U32:
    return raiseBinary32WriteVCC(Ctx, Op, Intrinsic::usub_with_overflow,
                                 /*ReverseOperands=*/false);
  case CanonicalOp::V_SUBREV_CO_U32:
    return raiseBinary32WriteVCC(Ctx, Op, Intrinsic::usub_with_overflow,
                                 /*ReverseOperands=*/true);
  case CanonicalOp::V_ADD_CO_CI_U32:
    return raiseBinary32ReadWriteVCC(Ctx, Op, Intrinsic::uadd_with_overflow,
                                     /*ReverseOperands=*/false);
  case CanonicalOp::V_SUB_CO_CI_U32:
    return raiseBinary32ReadWriteVCC(Ctx, Op, Intrinsic::usub_with_overflow,
                                     /*ReverseOperands=*/false);
  case CanonicalOp::V_SUBREV_CO_CI_U32:
    return raiseBinary32ReadWriteVCC(Ctx, Op, Intrinsic::usub_with_overflow,
                                     /*ReverseOperands=*/true);
  case CanonicalOp::V_CNDMASK_B32:
    return raiseCndMask(Ctx, Op);

  case CanonicalOp::V_MUL_I32_I24:
    return raiseMul24(Ctx, Op, /*IsSigned=*/true);
  case CanonicalOp::V_MUL_U32_U24:
    return raiseMul24(Ctx, Op, /*IsSigned=*/false);
  case CanonicalOp::V_MUL_HI_I32_I24:
    return raiseMulHi24(Ctx, Op, /*IsSigned=*/true);
  case CanonicalOp::V_MUL_HI_U32_U24:
    return raiseMulHi24(Ctx, Op, /*IsSigned=*/false);

  case CanonicalOp::V_MIN_I32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smin, Src0, Src1, {}, "min");
    });
  case CanonicalOp::V_MAX_I32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smax, Src0, Src1, {}, "max");
    });
  case CanonicalOp::V_MIN_U32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umin, Src0, Src1, {}, "min");
    });
  case CanonicalOp::V_MAX_U32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umax, Src0, Src1, {}, "max");
    });

  case CanonicalOp::V_AND_B32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAnd(Src0, Src1, "and");
    });
  case CanonicalOp::V_OR_B32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateOr(Src0, Src1, "or");
    });
  case CanonicalOp::V_XOR_B32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateXor(Src0, Src1, "xor");
    });
  case CanonicalOp::V_XNOR_B32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateNot(B.CreateXor(Src0, Src1, "xnor_xor"), "xnor");
    });

  // These take the shift amount in src0 and the value being shifted in src1.
  case CanonicalOp::V_LSHLREV_B32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateShl(Src1, maskShiftAmount(B, Src0, 32), "lshl");
    });
  case CanonicalOp::V_LSHRREV_B32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateLShr(Src1, maskShiftAmount(B, Src0, 32), "lshr");
    });
  case CanonicalOp::V_ASHRREV_I32:
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAShr(Src1, maskShiftAmount(B, Src0, 32), "ashr");
    });

  case CanonicalOp::V_ADD_NC_U64:
    return raiseBinary64(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAdd(Src0, Src1, "add64");
    });
  case CanonicalOp::V_SUB_NC_U64:
    return raiseBinary64(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateSub(Src0, Src1, "sub64");
    });
  case CanonicalOp::V_MUL_U64:
    return raiseBinary64(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateMul(Src0, Src1, "mul64");
    });
  case CanonicalOp::V_LSHLREV_B64:
    return raiseShiftLeft64(Ctx, Op);

  case CanonicalOp::V_ADD_U16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAdd(Src0, Src1, "add16");
    });
  case CanonicalOp::V_SUB_U16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateSub(Src0, Src1, "sub16");
    });
  case CanonicalOp::V_SUBREV_U16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateSub(Src1, Src0, "subrev16");
    });
  case CanonicalOp::V_MUL_LO_U16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateMul(Src0, Src1, "mul16");
    });
  case CanonicalOp::V_LSHLREV_B16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateShl(Src1, maskShiftAmount(B, Src0, 16), "lshl16");
    });
  case CanonicalOp::V_LSHRREV_B16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateLShr(Src1, maskShiftAmount(B, Src0, 16), "lshr16");
    });
  case CanonicalOp::V_ASHRREV_I16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAShr(Src1, maskShiftAmount(B, Src0, 16), "ashr16");
    });
  case CanonicalOp::V_MIN_I16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smin, Src0, Src1, {}, "min16");
    });
  case CanonicalOp::V_MAX_I16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smax, Src0, Src1, {}, "max16");
    });
  case CanonicalOp::V_MIN_U16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umin, Src0, Src1, {}, "min16");
    });
  case CanonicalOp::V_MAX_U16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umax, Src0, Src1, {}, "max16");
    });
  case CanonicalOp::V_DOT2C_I32_I16:
    return raiseSignedDotAccumulate(Ctx, Op, 16);
  case CanonicalOp::V_DOT4C_I32_I8:
    return raiseSignedDotAccumulate(Ctx, Op, 8);
  case CanonicalOp::V_DOT8C_I32_I4:
    return raiseSignedDotAccumulate(Ctx, Op, 4);

  default:
    return unsupported(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
