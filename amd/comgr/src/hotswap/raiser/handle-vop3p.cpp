//===- handle-vop3p.cpp - Hotswap transpiler -----------------------------===//
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
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/wmma-lowering.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {
namespace {

/// Read the VOP3P clamp operand. An absent operand is an unclamped result.
Expected<bool> readClamp(RaiseContext &Ctx, const DecodedInst &Di) {
  int Index = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(),
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
                                        Type *ElementType, bool IsF32) {
  constexpr unsigned AllowedModifiers = SISrcMods::NEG | SISrcMods::NEG_HI |
                                        SISrcMods::OP_SEL_0 |
                                        SISrcMods::OP_SEL_1;
  unsigned Modifiers = Op.srcMod(Source);
  if (Modifiers & ~AllowedModifiers)
    return unsupported(Ctx, Di, "unsupported packed source modifier");

  FixedVectorType *VectorType = FixedVectorType::get(ElementType, 2);
  Value *NaturalLow;
  Value *NaturalHigh;
  if (IsF32 && Op.isSrcReg(Source)) {
    Expected<Value *> SourceBits = Op.src64(Source);
    if (!SourceBits)
      return SourceBits.takeError();
    Value *Vector = Ctx.B.CreateBitCast(*SourceBits, VectorType, "pk.src");
    NaturalLow = Ctx.B.CreateExtractElement(Vector, uint64_t(0), "pk.lo");
    NaturalHigh = Ctx.B.CreateExtractElement(Vector, uint64_t(1), "pk.hi");
  } else if (IsF32) {
    Expected<Value *> SourceBits = Op.src(Source);
    if (!SourceBits)
      return SourceBits.takeError();
    Value *Scalar = Ctx.B.CreateBitCast(*SourceBits, ElementType, "pk.literal");
    NaturalLow = Scalar;
    NaturalHigh = Scalar;
  } else {
    Expected<Value *> SourceBits = Op.src(Source);
    if (!SourceBits)
      return SourceBits.takeError();
    Value *Vector = Ctx.B.CreateBitCast(*SourceBits, VectorType, "pk.src");
    NaturalLow = Ctx.B.CreateExtractElement(Vector, uint64_t(0), "pk.lo");
    NaturalHigh = Ctx.B.CreateExtractElement(Vector, uint64_t(1), "pk.hi");
  }

  Value *Low = Modifiers & SISrcMods::OP_SEL_0 ? NaturalHigh : NaturalLow;
  Value *High = Modifiers & SISrcMods::OP_SEL_1 ? NaturalHigh : NaturalLow;
  if (Modifiers & SISrcMods::NEG)
    Low = Ctx.B.CreateFNeg(Low, "pk.neg.lo");
  if (Modifiers & SISrcMods::NEG_HI)
    High = Ctx.B.CreateFNeg(High, "pk.neg.hi");

  Value *Result = PoisonValue::get(VectorType);
  Result = Ctx.B.CreateInsertElement(Result, Low, uint64_t(0), "pk.insert.lo");
  return Ctx.B.CreateInsertElement(Result, High, uint64_t(1), "pk.insert.hi");
}

/// Raise packed floating-point add and multiply instructions.
Error raisePackedFloatBinary(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op, Type *ElementType, bool IsF32,
                             bool IsAdd) {
  if (Di.NumDefs != 1 || Di.numOperands() == 0 || !Di.isReg(0) ||
      Op.nSrcs() != 2)
    return unsupported(Ctx, Di,
                       "expected one register destination and two sources");

  if (IsF32) {
    if (Error Err = Ctx.validateF32Environment(Di))
      return Err;
  } else if (Error Err = Ctx.validateF16Environment(Di)) {
    return Err;
  }

  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<Value *> Source0 =
      readPackedFloatSource(Ctx, Di, Op, 0, ElementType, IsF32);
  if (!Source0)
    return Source0.takeError();
  Expected<Value *> Source1 =
      readPackedFloatSource(Ctx, Di, Op, 1, ElementType, IsF32);
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

  if (IsF32) {
    Ctx.registers().writeRegVec(*Destination, Result);
  } else {
    Value *Packed = Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty(), "pk.pack");
    Ctx.registers().writeReg32(*Destination, Packed);
  }
  return Error::success();
}

Expected<Value *> readWMMAAccumulator(RaiseContext &Ctx, const DecodedInst &Di,
                                      OperandResolver &Op,
                                      Type *AccumulatorTy) {
  if (Op.nSrcs() < 3)
    return unsupported(Ctx, Di, "WMMA requires an accumulator source");
  Expected<std::optional<ParsedReg>> Source = Op.srcReg(2);
  if (!Source)
    return Source.takeError();
  if (*Source)
    return Ctx.registers().regFile().readRegVec(Ctx.B, **Source, AccumulatorTy);
  if (!Di.isImm(Op.srcIdx(2)) || Di.getImm(Op.srcIdx(2)) != 0)
    return unsupported(Ctx, Di,
                       "only a zero immediate WMMA accumulator is supported");
  return ConstantAggregateZero::get(AccumulatorTy);
}

Error raiseWMMA(RaiseContext &Ctx, const DecodedInst &Di, OperandResolver &Op,
                WMMAInputType InputType) {
  if (Ctx.Projection.sourceWaveSize() != 32 ||
      Ctx.Projection.targetWaveSize() != 64)
    return unsupported(Ctx, Di, "WMMA remapping requires wave32 to wave64");
  if (!Ctx.Projection.TargetSTI.hasFeature(AMDGPU::FeatureMAIInsts))
    return unsupported(Ctx, Di, "target ISA does not support MFMA");
  if (Op.nSrcs() < 3)
    return unsupported(Ctx, Di, "WMMA requires three source operands");

  for (unsigned I = 0; I != 3; ++I) {
    if (Op.srcMod(I) != 0)
      return unsupported(Ctx, Di, "WMMA source modifiers are not supported");
  }
  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();
  if (*Clamp)
    return unsupported(Ctx, Di, "WMMA clamp is not supported");

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<std::optional<ParsedReg>> SourceA = Op.srcReg(0);
  if (!SourceA)
    return SourceA.takeError();
  Expected<std::optional<ParsedReg>> SourceB = Op.srcReg(1);
  if (!SourceB)
    return SourceB.takeError();
  if (!*SourceA || !*SourceB)
    return unsupported(Ctx, Di, "WMMA matrix inputs must be registers");

  Type *InputTy = FixedVectorType::get(Ctx.B.getInt32Ty(), 8);
  Type *ElementTy =
      InputType == WMMAInputType::IU8 ? Ctx.B.getInt32Ty() : Ctx.B.getFloatTy();
  Type *AccumulatorTy = FixedVectorType::get(ElementTy, 8);
  AllocaRegFile &Registers = Ctx.registers().regFile();
  Value *A = Registers.readRegVec(Ctx.B, **SourceA, InputTy);
  Value *B = Registers.readRegVec(Ctx.B, **SourceB, InputTy);
  Expected<Value *> C = readWMMAAccumulator(Ctx, Di, Op, AccumulatorTy);
  if (!C)
    return C.takeError();
  Expected<Value *> Result = emitWMMAtoMFMA(Ctx, A, B, *C, InputType);
  if (!Result)
    return Result.takeError();
  Ctx.registers().writeRegVec(*Destination, *Result);
  return Error::success();
}

} // namespace

Error handleVOP3P(RaiseContext &Ctx, const DecodedInst &Di,
                  OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::V_PK_ADD_F16:
  case CanonicalOp::V_PK_MUL_F16:
    return raisePackedFloatBinary(
        Ctx, Di, Op, Ctx.B.getHalfTy(), /*IsF32=*/false,
        /*IsAdd=*/Di.CanonOp == CanonicalOp::V_PK_ADD_F16);
  case CanonicalOp::V_PK_ADD_F32:
  case CanonicalOp::V_PK_MUL_F32:
    return raisePackedFloatBinary(
        Ctx, Di, Op, Ctx.B.getFloatTy(), /*IsF32=*/true,
        /*IsAdd=*/Di.CanonOp == CanonicalOp::V_PK_ADD_F32);
  case CanonicalOp::V_WMMA_F32_16x16x32_F16:
    return raiseWMMA(Ctx, Di, Op, WMMAInputType::F16);
  case CanonicalOp::V_WMMA_F32_16x16x32_BF16:
    return raiseWMMA(Ctx, Di, Op, WMMAInputType::BF16);
  case CanonicalOp::V_WMMA_I32_16x16x64_IU8:
    return raiseWMMA(Ctx, Di, Op, WMMAInputType::IU8);
  default:
    return unsupported(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
