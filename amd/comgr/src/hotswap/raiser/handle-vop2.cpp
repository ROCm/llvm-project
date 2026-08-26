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
#include "hotswap/raiser/op-resolver.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {

// Builds the result of a two-source instruction from its already-read sources.
using BinaryBuilder = function_ref<Value *(IRBuilder<> &, Value *, Value *)>;

static Error raiseFloatBinary(RaiseContext &Ctx, const DecodedInst &Di,
                              OpResolver &Op, Instruction::BinaryOps Opcode,
                              bool ReverseOperands) {
  if (Di.NumDefs != 1 || Di.numOperands() == 0 || !Di.isReg(0) ||
      Op.nSrcs() != 2) {
    return unsupported(Ctx, Di,
                       "expected one register destination and two sources");
  }

  if (Error Err = Ctx.validateF32Environment(Di)) {
    return Err;
  }

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst) {
    return Dst.takeError();
  }
  Expected<Value *> Src0Bits = Op.src(0);
  if (!Src0Bits) {
    return Src0Bits.takeError();
  }
  Expected<Value *> Src1Bits = Op.src(1);
  if (!Src1Bits) {
    return Src1Bits.takeError();
  }

  Value *Src0 = Ctx.B.CreateBitCast(*Src0Bits, Ctx.B.getFloatTy());
  Value *Src1 = Ctx.B.CreateBitCast(*Src1Bits, Ctx.B.getFloatTy());
  Value *Lhs = ReverseOperands ? Src1 : Src0;
  Value *Rhs = ReverseOperands ? Src0 : Src1;
  Value *Result = Ctx.B.CreateBinOp(Opcode, Lhs, Rhs);
  Ctx.registers().writeReg32(*Dst,
                             Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty()));
  return Error::success();
}

// Build a 32-bit result from two integer sources and write it to the
// destination.
static Error raiseBinary32(RaiseContext &Ctx, OpResolver &Op,
                           BinaryBuilder Build) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst) {
    return Dst.takeError();
  }
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0) {
    return Src0.takeError();
  }
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1) {
    return Src1.takeError();
  }
  Ctx.registers().writeReg32(*Dst, Build(Ctx.B, *Src0, *Src1));
  return Error::success();
}

// Build a 64-bit result from two integer sources and write it to the
// destination.
static Error raiseBinary64(RaiseContext &Ctx, OpResolver &Op,
                           BinaryBuilder Build) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst) {
    return Dst.takeError();
  }
  Expected<Value *> Src0 = Op.src64(0);
  if (!Src0) {
    return Src0.takeError();
  }
  Expected<Value *> Src1 = Op.src64(1);
  if (!Src1) {
    return Src1.takeError();
  }
  Ctx.registers().writeReg64(*Dst, Build(Ctx.B, *Src0, *Src1));
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
static Error raiseMul24(RaiseContext &Ctx, OpResolver &Op, bool IsSigned) {
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
static Error raiseMulHi24(RaiseContext &Ctx, OpResolver &Op, bool IsSigned) {
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
static Error raiseShiftLeft64(RaiseContext &Ctx, OpResolver &Op) {
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

Error handleVOP2(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
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

  default:
    return unsupported(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
