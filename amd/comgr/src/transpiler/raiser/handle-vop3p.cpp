//===- handle-vop3p.cpp - Transpiler -------------------------------------===//
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
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"

#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <optional>

using namespace llvm;

namespace COMGR::transpiler {
namespace {

/// Read the VOP3P clamp operand. An absent operand is an unclamped result.
Expected<bool> readClamp(RaiseContext &Ctx, const DecodedInst &Di) {
  int Index = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                    AMDGPU::OpName::clamp);
  if (Index < 0)
    return false;
  if (!Di.isImm(static_cast<unsigned>(Index)))
    return unsupported(Ctx, Di, "clamp operand is not immediate");
  int64_t Clamp = Di.getImm(static_cast<unsigned>(Index));
  if (Clamp != 0 && Clamp != 1)
    return unsupported(Ctx, Di, "clamp operand is not 0 or 1");
  return Clamp != 0;
}

/// Read one two-lane packed floating-point source and apply its lane controls.
Expected<Value *> readPackedFloatSource(RaiseContext &Ctx,
                                        const DecodedInst &Di,
                                        OperandResolver &Op, unsigned Source,
                                        Type *ElementType) {
  assert((ElementType->isHalfTy() || ElementType->isFloatTy()) &&
         "unsupported packed floating-point element type");
  constexpr unsigned AllowedModifiers = SISrcMods::NEG | SISrcMods::NEG_HI |
                                        SISrcMods::OP_SEL_0 |
                                        SISrcMods::OP_SEL_1;
  unsigned Modifiers = Op.srcMod(Source);
  if (Modifiers & ~AllowedModifiers)
    return unsupported(Ctx, Di, "unsupported packed source modifier");

  FixedVectorType *VectorType = FixedVectorType::get(ElementType, 2);
  Value *NaturalLow;
  Value *NaturalHigh;
  bool IsVectorRegister = false;
  if (ElementType->isFloatTy()) {
    Expected<std::optional<ParsedReg>> SourceReg = Op.srcReg(Source);
    if (!SourceReg)
      return SourceReg.takeError();
    if (*SourceReg)
      IsVectorRegister = (*SourceReg)->RegKind == ParsedReg::VGPR ||
                         (*SourceReg)->RegKind == ParsedReg::AGPR;
  }

  Expected<Value *> SourceBits =
      IsVectorRegister ? Op.src64(Source) : Op.src(Source);
  if (!SourceBits)
    return SourceBits.takeError();

  if (ElementType->isFloatTy() && !IsVectorRegister) {
    // Packed F32 scalar sources provide one value for both result channels.
    Value *Scalar = Ctx.B.CreateBitCast(*SourceBits, ElementType, "pk.scalar");
    NaturalLow = Scalar;
    NaturalHigh = Scalar;
  } else {
    // Packed F16 sources and F32 vector sources carry both channel values.
    IntegerType *ElementIntegerType =
        ElementType->isFloatTy() ? Ctx.B.getInt32Ty() : Ctx.B.getInt16Ty();
    unsigned ElementBitWidth = ElementIntegerType->getBitWidth();
    Value *LowBits =
        Ctx.B.CreateTrunc(*SourceBits, ElementIntegerType, "pk.lo.bits");
    Value *ShiftedBits =
        Ctx.B.CreateLShr(*SourceBits, ElementBitWidth, "pk.hi.shifted");
    Value *HighBits =
        Ctx.B.CreateTrunc(ShiftedBits, ElementIntegerType, "pk.hi.bits");
    NaturalLow = Ctx.B.CreateBitCast(LowBits, ElementType, "pk.lo");
    NaturalHigh = Ctx.B.CreateBitCast(HighBits, ElementType, "pk.hi");
  }

  Value *Low = Modifiers & SISrcMods::OP_SEL_0 ? NaturalHigh : NaturalLow;
  Value *High = Modifiers & SISrcMods::OP_SEL_1 ? NaturalHigh : NaturalLow;
  if (Modifiers & SISrcMods::NEG)
    Low = Ctx.B.CreateFNeg(Low, "pk.neg.lo");
  if (Modifiers & SISrcMods::NEG_HI)
    High = Ctx.B.CreateFNeg(High, "pk.neg.hi");

  Value *Result = PoisonValue::get(VectorType);
  Result = Ctx.B.CreateInsertElement(Result, Low, uint64_t{0}, "pk.insert.lo");
  return Ctx.B.CreateInsertElement(Result, High, 1, "pk.insert.hi");
}

/// Raise packed floating-point add and multiply instructions.
Error raisePackedFloatBinary(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op, Type *ElementType,
                             bool IsAdd) {
  assert((Di.NumDefs == 1 && Di.numOperands() != 0 && Di.isReg(0) &&
          Op.nSrcs() == 2) &&
         "decoded packed float instruction has unexpected operands");

  if (Error Err = Ctx.validateFPEnvironment(Di, ElementType))
    return Err;

  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<Value *> Source0 =
      readPackedFloatSource(Ctx, Di, Op, 0, ElementType);
  if (!Source0)
    return Source0.takeError();
  Expected<Value *> Source1 =
      readPackedFloatSource(Ctx, Di, Op, 1, ElementType);
  if (!Source1)
    return Source1.takeError();

  Value *Result = IsAdd ? Ctx.B.CreateFAdd(*Source0, *Source1, "pk.add")
                        : Ctx.B.CreateFMul(*Source0, *Source1, "pk.mul");
  if (*Clamp) {
    FixedVectorType *VectorType = FixedVectorType::get(ElementType, 2);
    Function *Maximum = Intrinsic::getOrInsertDeclaration(
        Ctx.B.GetInsertBlock()->getModule(), Intrinsic::maxnum, {VectorType});
    Function *Minimum = Intrinsic::getOrInsertDeclaration(
        Ctx.B.GetInsertBlock()->getModule(), Intrinsic::minnum, {VectorType});
    Constant *Zero = ConstantVector::getSplat(
        ElementCount::getFixed(2), ConstantFP::get(ElementType, 0.0));
    Constant *One = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantFP::get(ElementType, 1.0));
    Result = Ctx.B.CreateCall(Maximum, {Result, Zero}, "pk.clamp.low");
    Result = Ctx.B.CreateCall(Minimum, {Result, One}, "pk.clamp");
  }

  if (ElementType->isFloatTy()) {
    Ctx.registers().writeRegVec(*Destination, Result);
  } else {
    Value *Packed = Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty(), "pk.pack");
    Ctx.registers().writeReg32(*Destination, Packed);
  }
  return Error::success();
}

} // namespace

Error handleVOP3P(RaiseContext &Ctx, const DecodedInst &Di,
                  OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::V_PK_ADD_F16:
  case CanonicalOp::V_PK_MUL_F16:
    return raisePackedFloatBinary(Ctx, Di, Op, Ctx.B.getHalfTy(),
                                  /*IsAdd=*/Di.CanonOp ==
                                      CanonicalOp::V_PK_ADD_F16);
  case CanonicalOp::V_PK_ADD_F32:
  case CanonicalOp::V_PK_MUL_F32:
    return raisePackedFloatBinary(Ctx, Di, Op, Ctx.B.getFloatTy(),
                                  /*IsAdd=*/Di.CanonOp ==
                                      CanonicalOp::V_PK_ADD_F32);
  default:
    return unsupported(Ctx, Di);
  }
}

} // namespace COMGR::transpiler
