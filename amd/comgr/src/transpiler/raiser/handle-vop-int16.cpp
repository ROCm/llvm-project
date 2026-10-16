//===- handle-vop-int16.cpp - Transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/handle-vop-shared.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cassert>
#include <cstdint>

using namespace llvm;

namespace COMGR::transpiler {

/// Reject the source modifiers that have no meaning on a 16-bit integer
/// operand. Only the op_sel bits, which pick the register halves the
/// instruction reads and writes, survive.
static Error requireOnlyHalfSelectModifiers(RaiseContext &Ctx,
                                            const DecodedInst &Di,
                                            OperandResolver &Op) {
  // src0 additionally carries the destination's half in DST_OP_SEL.
  constexpr unsigned AllowedSrc0 = SISrcMods::OP_SEL_0 | SISrcMods::DST_OP_SEL;
  for (unsigned I = 0, E = Op.nSrcs(); I != E; ++I) {
    unsigned Allowed = I == 0 ? AllowedSrc0 : SISrcMods::OP_SEL_0;
    if (Op.srcMod(I) & ~Allowed)
      return unsupported(Ctx, Di,
                         "16-bit integer source modifiers are not supported");
  }
  return Error::success();
}

/// Raise an add or subtract, saturating when the clamp bit is set.
static Error raiseAddSub16(RaiseContext &Ctx, OperandResolver &Op, bool Clamp,
                           bool IsSigned, bool IsSub, bool Reverse) {
  return raiseBinary16(Ctx, Op, [&](IRBuilder<> &B, Value *Src0, Value *Src1) {
    Value *Lhs = Reverse ? Src1 : Src0;
    Value *Rhs = Reverse ? Src0 : Src1;
    if (Clamp) {
      Intrinsic::ID ID =
          IsSub ? (IsSigned ? Intrinsic::ssub_sat : Intrinsic::usub_sat)
                : (IsSigned ? Intrinsic::sadd_sat : Intrinsic::uadd_sat);
      return B.CreateBinaryIntrinsic(ID, Lhs, Rhs, {}, "addsub16");
    }
    return IsSub ? B.CreateSub(Lhs, Rhs, "sub16")
                 : B.CreateAdd(Lhs, Rhs, "add16");
  });
}

/// Raise a shift whose count is src0 and whose value is src1, masked to the
/// four count bits the hardware reads.
static Error raiseShift16(RaiseContext &Ctx, OperandResolver &Op,
                          Instruction::BinaryOps Opcode) {
  return raiseBinary16(Ctx, Op, [&](IRBuilder<> &B, Value *Src0, Value *Src1) {
    return B.CreateBinOp(Opcode, Src1,
                         maskShiftAmount(B, Src0, /*Width=*/HalfWidthInBits),
                         "shift16");
  });
}

/// Raise a two-source min or max.
static Error raiseMinMax16(RaiseContext &Ctx, OperandResolver &Op,
                           Intrinsic::ID ID) {
  return raiseBinary16(Ctx, Op, [&](IRBuilder<> &B, Value *Src0, Value *Src1) {
    return B.CreateBinaryIntrinsic(ID, Src0, Src1, {}, "minmax16");
  });
}

/// Read the selected 16-bit halves of the first three sources.
static Expected<std::array<Value *, 3>> readTernary16(OperandResolver &Op) {
  std::array<Value *, 3> Sources;
  for (unsigned I = 0, E = Sources.size(); I != E; ++I) {
    Expected<Value *> Source = Op.src16(I);
    if (!Source)
      return Source.takeError();
    Sources[I] = *Source;
  }
  return Sources;
}

/// Raise a nested min/max over three sources, which is how the hardware
/// defines both the min3/max3 and the median opcodes.
static Error raiseTernaryMinMax16(RaiseContext &Ctx, OperandResolver &Op,
                                  Intrinsic::ID Inner, Intrinsic::ID Outer) {
  Expected<std::array<Value *, 3>> Sources = readTernary16(Op);
  if (!Sources)
    return Sources.takeError();
  auto [Src0, Src1, Src2] = *Sources;
  Value *First = Ctx.B.CreateBinaryIntrinsic(Inner, Src0, Src1, {}, "minmax3");
  Value *Result =
      Ctx.B.CreateBinaryIntrinsic(Outer, First, Src2, {}, "minmax3");
  return writeDestination16(Ctx, Op, Result);
}

/// Raise a median of three, which clamps src2 into the range spanned by the
/// other two sources.
static Error raiseMedian16(RaiseContext &Ctx, OperandResolver &Op,
                           bool IsSigned) {
  Expected<std::array<Value *, 3>> Sources = readTernary16(Op);
  if (!Sources)
    return Sources.takeError();
  auto [Src0, Src1, Src2] = *Sources;
  Intrinsic::ID Min = IsSigned ? Intrinsic::smin : Intrinsic::umin;
  Intrinsic::ID Max = IsSigned ? Intrinsic::smax : Intrinsic::umax;
  Value *Low = Ctx.B.CreateBinaryIntrinsic(Min, Src0, Src1);
  Value *High = Ctx.B.CreateBinaryIntrinsic(Max, Src0, Src1);
  Value *Upper = Ctx.B.CreateBinaryIntrinsic(Min, High, Src2);
  Value *Result = Ctx.B.CreateBinaryIntrinsic(Max, Low, Upper, {}, "med3.16");
  return writeDestination16(Ctx, Op, Result);
}

/// Raise a multiply-add. The sum is formed at 32 bits so that a set clamp bit
/// saturates the value the hardware computes rather than an already-wrapped
/// one.
static Error raiseMad16(RaiseContext &Ctx, OperandResolver &Op, bool Clamp,
                        bool IsSigned) {
  Expected<std::array<Value *, 3>> Sources = readTernary16(Op);
  if (!Sources)
    return Sources.takeError();
  auto [Src0, Src1, Src2] = *Sources;
  IntegerType *I32Ty = Ctx.B.getInt32Ty();
  auto Widen = [&](Value *V) {
    return IsSigned ? Ctx.B.CreateSExt(V, I32Ty) : Ctx.B.CreateZExt(V, I32Ty);
  };
  Value *Product = Ctx.B.CreateMul(Widen(Src0), Widen(Src1), "mad16.mul");
  Value *Sum = Ctx.B.CreateAdd(Product, Widen(Src2), "mad16.sum");
  if (Clamp) {
    // Zero-extended operands cannot sum to a negative value, so only the
    // signed form needs a lower bound.
    if (IsSigned)
      Sum = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Sum,
                                        Ctx.B.getInt32(INT16_MIN));
    Sum = Ctx.B.CreateBinaryIntrinsic(
        IsSigned ? Intrinsic::smin : Intrinsic::umin, Sum,
        Ctx.B.getInt32(IsSigned ? INT16_MAX : UINT16_MAX));
  }
  Value *Result = Ctx.B.CreateTrunc(Sum, Ctx.B.getInt16Ty(), "mad16");
  return writeDestination16(Ctx, Op, Result);
}

/// Raise V_BITOP3_B16, whose eight-entry truth table arrives as the immediate
/// fourth source.
static Error raiseBitOp3Int16(RaiseContext &Ctx, const DecodedInst &Di,
                              OperandResolver &Op) {
  assert(Op.nSrcs() >= 4 && Di.isImm(Op.srcIdx(3)) &&
         "bit operation must have three sources and a truth table immediate");
  Expected<std::array<Value *, 3>> Sources = readTernary16(Op);
  if (!Sources)
    return Sources.takeError();
  auto [Src0, Src1, Src2] = *Sources;
  Value *Result =
      emitBitOp3(Ctx.B, Src0, Src1, Src2, static_cast<uint8_t>(Op.srcImm(3)));
  return writeDestination16(Ctx, Op, Result);
}

/// Raise a select between the two source halves. The two-source encodings take
/// the lane condition from VCC and the three-source ones name a wave mask.
/// Both value sources additionally carry the f16 sign modifiers.
static Error raiseCndMask16(RaiseContext &Ctx, const DecodedInst &Di,
                            OperandResolver &Op) {
  assert((Op.nSrcs() == 2 || Op.nSrcs() == 3) &&
         "select must have two values and at most one named condition");
  constexpr unsigned SignModifiers = SISrcMods::NEG | SISrcMods::ABS;
  std::array<Value *, 2> Values;
  for (unsigned I = 0, E = Values.size(); I != E; ++I) {
    // src0 additionally carries the destination's half in DST_OP_SEL.
    unsigned Allowed = SISrcMods::OP_SEL_0 | SignModifiers |
                       (I == 0 ? SISrcMods::DST_OP_SEL : 0);
    if (Op.srcMod(I) & ~Allowed)
      return unsupported(Ctx, Di, "unsupported select source modifier");
    Expected<Value *> Source = Op.src16(I);
    if (!Source)
      return Source.takeError();
    Values[I] =
        applyFloat16SignModifiers(Ctx.B, *Source, Op.srcMod(I) & SignModifiers);
  }
  Expected<Value *> Condition = readCndMaskCondition(Ctx, Op);
  if (!Condition)
    return Condition.takeError();
  Value *Result =
      Ctx.B.CreateSelect(*Condition, Values[1], Values[0], "cndmask16");
  return writeDestination16(Ctx, Op, Result);
}

bool isInteger16Op(CanonicalOp Operation) {
  switch (Operation) {
  case CanonicalOp::V_ADD_U16:
  case CanonicalOp::V_ADD_I16:
  case CanonicalOp::V_SUB_U16:
  case CanonicalOp::V_SUB_I16:
  case CanonicalOp::V_SUBREV_U16:
  case CanonicalOp::V_MUL_LO_U16:
  case CanonicalOp::V_MAD_U16:
  case CanonicalOp::V_MAD_I16:
  case CanonicalOp::V_LSHLREV_B16:
  case CanonicalOp::V_LSHRREV_B16:
  case CanonicalOp::V_ASHRREV_I16:
  case CanonicalOp::V_MIN_I16:
  case CanonicalOp::V_MAX_I16:
  case CanonicalOp::V_MIN_U16:
  case CanonicalOp::V_MAX_U16:
  case CanonicalOp::V_MIN3_I16:
  case CanonicalOp::V_MAX3_I16:
  case CanonicalOp::V_MIN3_U16:
  case CanonicalOp::V_MAX3_U16:
  case CanonicalOp::V_MED3_I16:
  case CanonicalOp::V_MED3_U16:
  case CanonicalOp::V_AND_B16:
  case CanonicalOp::V_OR_B16:
  case CanonicalOp::V_XOR_B16:
  case CanonicalOp::V_NOT_B16:
  case CanonicalOp::V_BITOP3_B16:
  case CanonicalOp::V_MOV_B16:
  case CanonicalOp::V_CNDMASK_B16:
    return true;
  default:
    return false;
  }
}

Error handleInteger16(RaiseContext &Ctx, const DecodedInst &Di,
                      OperandResolver &Op) {

  // V_CNDMASK_B16 names a wave mask in src2, which is not a 16-bit operand and
  // whose half-select bits are reserved, so it is screened before the shared
  // modifier check.
  if (Di.CanonOp == CanonicalOp::V_CNDMASK_B16)
    return raiseCndMask16(Ctx, Di, Op);

  if (Error Err = requireOnlyHalfSelectModifiers(Ctx, Di, Op))
    return Err;
  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();

  switch (Di.CanonOp) {
  case CanonicalOp::V_ADD_U16:
    return raiseAddSub16(Ctx, Op, *Clamp, /*IsSigned=*/false, /*IsSub=*/false,
                         /*Reverse=*/false);
  case CanonicalOp::V_ADD_I16:
    return raiseAddSub16(Ctx, Op, *Clamp, /*IsSigned=*/true, /*IsSub=*/false,
                         /*Reverse=*/false);
  case CanonicalOp::V_SUB_U16:
    return raiseAddSub16(Ctx, Op, *Clamp, /*IsSigned=*/false, /*IsSub=*/true,
                         /*Reverse=*/false);
  case CanonicalOp::V_SUB_I16:
    return raiseAddSub16(Ctx, Op, *Clamp, /*IsSigned=*/true, /*IsSub=*/true,
                         /*Reverse=*/false);
  case CanonicalOp::V_SUBREV_U16:
    return raiseAddSub16(Ctx, Op, *Clamp, /*IsSigned=*/false, /*IsSub=*/true,
                         /*Reverse=*/true);
  case CanonicalOp::V_MAD_U16:
    return raiseMad16(Ctx, Op, *Clamp, /*IsSigned=*/false);
  case CanonicalOp::V_MAD_I16:
    return raiseMad16(Ctx, Op, *Clamp, /*IsSigned=*/true);
  default:
    break;
  }

  // Every remaining opcode leaves the result unmodified, so a set clamp bit
  // would have no representation.
  if (*Clamp)
    return unsupported(Ctx, Di,
                       "16-bit integer operation does not define clamp");

  switch (Di.CanonOp) {
  case CanonicalOp::V_MUL_LO_U16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateMul(Src0, Src1, "mul16");
    });
  case CanonicalOp::V_LSHLREV_B16:
    return raiseShift16(Ctx, Op, Instruction::Shl);
  case CanonicalOp::V_LSHRREV_B16:
    return raiseShift16(Ctx, Op, Instruction::LShr);
  case CanonicalOp::V_ASHRREV_I16:
    return raiseShift16(Ctx, Op, Instruction::AShr);

  case CanonicalOp::V_MIN_I16:
    return raiseMinMax16(Ctx, Op, Intrinsic::smin);
  case CanonicalOp::V_MAX_I16:
    return raiseMinMax16(Ctx, Op, Intrinsic::smax);
  case CanonicalOp::V_MIN_U16:
    return raiseMinMax16(Ctx, Op, Intrinsic::umin);
  case CanonicalOp::V_MAX_U16:
    return raiseMinMax16(Ctx, Op, Intrinsic::umax);
  case CanonicalOp::V_MIN3_I16:
    return raiseTernaryMinMax16(Ctx, Op, Intrinsic::smin, Intrinsic::smin);
  case CanonicalOp::V_MAX3_I16:
    return raiseTernaryMinMax16(Ctx, Op, Intrinsic::smax, Intrinsic::smax);
  case CanonicalOp::V_MIN3_U16:
    return raiseTernaryMinMax16(Ctx, Op, Intrinsic::umin, Intrinsic::umin);
  case CanonicalOp::V_MAX3_U16:
    return raiseTernaryMinMax16(Ctx, Op, Intrinsic::umax, Intrinsic::umax);
  case CanonicalOp::V_MED3_I16:
    return raiseMedian16(Ctx, Op, /*IsSigned=*/true);
  case CanonicalOp::V_MED3_U16:
    return raiseMedian16(Ctx, Op, /*IsSigned=*/false);

  case CanonicalOp::V_AND_B16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAnd(Src0, Src1, "and16");
    });
  case CanonicalOp::V_OR_B16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateOr(Src0, Src1, "or16");
    });
  case CanonicalOp::V_XOR_B16:
    return raiseBinary16(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateXor(Src0, Src1, "xor16");
    });
  case CanonicalOp::V_BITOP3_B16:
    return raiseBitOp3Int16(Ctx, Di, Op);

  case CanonicalOp::V_NOT_B16: {
    Expected<Value *> Source = Op.src16(0);
    if (!Source)
      return Source.takeError();
    return writeDestination16(Ctx, Op, Ctx.B.CreateNot(*Source, "not16"));
  }
  case CanonicalOp::V_MOV_B16: {
    Expected<Value *> Source = Op.src16(0);
    if (!Source)
      return Source.takeError();
    return writeDestination16(Ctx, Op, *Source);
  }

  default:
    return unsupported(Ctx, Di);
  }
}

} // namespace COMGR::transpiler
