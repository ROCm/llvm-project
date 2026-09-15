//===- handle-vop-cross-lane.cpp - Cross-lane VOP helpers ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handle-vop-cross-lane.h"

#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

using namespace llvm;

namespace COMGR::hotswap {

/// Reject cross-lane operations that would need lanes absent on the target.
static Error requireSupportedWaveDirection(RaiseContext &Ctx,
                                           const DecodedInst &Di) {
  if (Ctx.Projection.targetWaveSize() >= Ctx.Projection.sourceWaveSize())
    return Error::success();
  return unsupportedInstruction(
      Ctx, Di, "cross-lane VALU does not support wave-size narrowing");
}

/// Return the instruction destination after requiring a VGPR operand.
static Expected<ParsedReg> requireVectorDestination(RaiseContext &Ctx,
                                                    const DecodedInst &Di,
                                                    OperandResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  if (Dst->RegKind != ParsedReg::VGPR)
    return unsupportedInstruction(
        Ctx, Di, "v_writelane_b32 requires a VGPR destination");
  return *Dst;
}

/// Return the instruction destination after requiring writable scalar state.
static Expected<ParsedReg> requireScalarDestination(RaiseContext &Ctx,
                                                    const DecodedInst &Di,
                                                    OperandResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  switch (Dst->RegKind) {
  case ParsedReg::SGPR:
  case ParsedReg::VCC:
  case ParsedReg::EXEC:
  case ParsedReg::M0:
  case ParsedReg::FLAT_SCR:
  case ParsedReg::TTMP:
  case ParsedReg::VCC_HI_SCRATCH:
  case ParsedReg::EXEC_HI_SCRATCH:
  case ParsedReg::NOREG:
    return *Dst;
  default:
    return unsupportedInstruction(
        Ctx, Di, "cross-lane read requires a writable scalar destination");
  }
}

/// Return the first source register after requiring a VGPR operand.
static Expected<ParsedReg> requireVectorSource(RaiseContext &Ctx,
                                               const DecodedInst &Di,
                                               OperandResolver &Op) {
  Expected<std::optional<ParsedReg>> Src = Op.srcReg(0);
  if (!Src)
    return Src.takeError();
  if (!*Src || (**Src).RegKind != ParsedReg::VGPR)
    return unsupportedInstruction(Ctx, Di,
                                  "cross-lane read requires a VGPR source");
  return **Src;
}

/// Return the bit mask applied to a source-wave lane selector.
static Value *getSourceLaneMask(IRBuilder<> &B,
                                const WaveProjection &Projection) {
  return B.getInt32(Projection.sourceWaveSize() - 1);
}

/// Mask a lane selector to the source wave width.
static Value *emitSourceWaveLane(RaiseContext &Ctx, Value *Lane,
                                 const Twine &Name) {
  return Ctx.B.CreateAnd(Lane, getSourceLaneMask(Ctx.B, Ctx.Projection), Name);
}

/// Return the first target lane occupied by the current source-wave instance.
static Value *emitSourceWaveBase(RaiseContext &Ctx, const Twine &Name) {
  Value *Lane = Ctx.emitLaneIdx();
  uint32_t SourceMask = Ctx.Projection.sourceWaveSize() - 1;
  return Ctx.B.CreateAnd(Lane, Ctx.B.getInt32(~SourceMask), Name);
}

/// Read Src from SourceLane in the current source-wave instance.
static Value *emitSourceWaveRead(RaiseContext &Ctx, Value *Src,
                                 Value *SourceLane, const Twine &Name) {
  Value *Base = emitSourceWaveBase(Ctx, Name + ".base");
  Value *TargetLane = Ctx.B.CreateOr(Base, SourceLane, Name + ".lane");
  Value *ByteAddress =
      Ctx.B.CreateShl(TargetLane, Ctx.B.getInt32(2), Name + ".addr");
  Module *M = Ctx.B.GetInsertBlock()->getModule();
  Function *BPermute =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_ds_bpermute);
  Value *Gathered = Ctx.B.CreateCall(BPermute, {ByteAddress, Src}, Name);
  return Ctx.Projection.wrapAsWWMValue(Ctx.B, Gathered, Name + ".wwm");
}

Error raiseReadFirstLane32(RaiseContext &Ctx, const DecodedInst &Di,
                           OperandResolver &Op) {
  if (Error Err = requireSupportedWaveDirection(Ctx, Di))
    return Err;
  if (Op.nSrcs() != 1)
    return unsupportedInstruction(Ctx, Di, "expected one source operand");

  Expected<ParsedReg> Dst = requireScalarDestination(Ctx, Di, Op);
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src = Op.src(0);
  if (!Src)
    return Src.takeError();

  Value *Result = nullptr;
  if (Ctx.Projection.targetWaveSize() == Ctx.Projection.sourceWaveSize()) {
    Module *M = Ctx.B.GetInsertBlock()->getModule();
    Function *ReadFirstLane = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::amdgcn_readfirstlane, {Ctx.B.getInt32Ty()});
    Result = Ctx.B.CreateCall(ReadFirstLane, {*Src}, "readfirstlane");
  } else {
    Value *Exec = Ctx.registers().regFile().loadExec(Ctx.B);
    Value *SourceExec = Ctx.Projection.emitCurrentSourceWaveMask(
        Ctx.B, Exec, "readfirstlane.exec");
    Type *MaskTy = SourceExec->getType();
    Function *CountTrailingZeros = Intrinsic::getOrInsertDeclaration(
        Ctx.B.GetInsertBlock()->getModule(), Intrinsic::cttz, {MaskTy});
    Value *FirstActive =
        Ctx.B.CreateCall(CountTrailingZeros,
                         {SourceExec, ConstantInt::getFalse(Ctx.B.getInt1Ty())},
                         "readfirstlane.first");
    Value *ExecIsZero = Ctx.B.CreateICmpEQ(
        SourceExec, ConstantInt::get(MaskTy, 0), "readfirstlane.exec.zero");
    Value *Selected =
        Ctx.B.CreateSelect(ExecIsZero, ConstantInt::get(MaskTy, 0), FirstActive,
                           "readfirstlane.selected");
    Value *SourceLane = Ctx.B.CreateZExtOrTrunc(Selected, Ctx.B.getInt32Ty(),
                                                "readfirstlane.source.lane");
    Result = emitSourceWaveRead(Ctx, *Src, SourceLane, "readfirstlane");
  }

  Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

Error raiseReadLane32(RaiseContext &Ctx, const DecodedInst &Di,
                      OperandResolver &Op) {
  if (Error Err = requireSupportedWaveDirection(Ctx, Di))
    return Err;
  if (Op.nSrcs() != 2)
    return unsupportedInstruction(Ctx, Di, "expected two source operands");

  Expected<ParsedReg> Dst = requireScalarDestination(Ctx, Di, Op);
  if (!Dst)
    return Dst.takeError();
  Expected<ParsedReg> SrcReg = requireVectorSource(Ctx, Di, Op);
  if (!SrcReg)
    return SrcReg.takeError();
  Expected<Value *> Lane = Op.src(1);
  if (!Lane)
    return Lane.takeError();

  Value *Src = Ctx.registers().regFile().readReg32(Ctx.B, *SrcReg);
  Value *Lane32 =
      Ctx.B.CreateZExtOrTrunc(*Lane, Ctx.B.getInt32Ty(), "readlane.index");
  Value *SourceLane = emitSourceWaveLane(Ctx, Lane32, "readlane.source.lane");
  Value *Result = nullptr;
  if (Ctx.Projection.targetWaveSize() == Ctx.Projection.sourceWaveSize()) {
    Module *M = Ctx.B.GetInsertBlock()->getModule();
    Function *ReadLane = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::amdgcn_readlane, {Ctx.B.getInt32Ty()});
    Result = Ctx.B.CreateCall(ReadLane, {Src, SourceLane}, "readlane");
  } else {
    Result = emitSourceWaveRead(Ctx, Src, SourceLane, "readlane");
  }

  Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

Error raiseWriteLane32(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op) {
  if (Error Err = requireSupportedWaveDirection(Ctx, Di))
    return Err;
  if (Op.nSrcs() != 2)
    return unsupportedInstruction(Ctx, Di, "expected two source operands");

  Expected<ParsedReg> Dst = requireVectorDestination(Ctx, Di, Op);
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> ValueToWrite = Op.src(0);
  if (!ValueToWrite)
    return ValueToWrite.takeError();
  Expected<Value *> Lane = Op.src(1);
  if (!Lane)
    return Lane.takeError();
  Expected<Value *> Old = Op.dstValue();
  if (!Old)
    return Old.takeError();

  Value *Lane32 =
      Ctx.B.CreateZExtOrTrunc(*Lane, Ctx.B.getInt32Ty(), "writelane.index");
  Value *SourceLane = emitSourceWaveLane(Ctx, Lane32, "writelane.source.lane");
  Value *Result = nullptr;
  if (Ctx.Projection.targetWaveSize() == Ctx.Projection.sourceWaveSize()) {
    Module *M = Ctx.B.GetInsertBlock()->getModule();
    Function *WriteLane = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::amdgcn_writelane, {Ctx.B.getInt32Ty()});
    Result = Ctx.B.CreateCall(WriteLane, {*ValueToWrite, SourceLane, *Old},
                              "writelane");
  } else {
    Value *LaneId = Ctx.emitLaneIdx();
    Value *CurrentSourceLane =
        emitSourceWaveLane(Ctx, LaneId, "writelane.current.source.lane");
    Value *IsSelected = Ctx.B.CreateICmpEQ(CurrentSourceLane, SourceLane,
                                           "writelane.is.selected");
    Result = Ctx.B.CreateSelect(IsSelected, *ValueToWrite, *Old,
                                "writelane.source.wave");
  }

  // V_WRITELANE_B32 overrides EXEC for its VGPR write.
  Ctx.registers().writeReg32IgnoringExec(*Dst, Result);
  return Error::success();
}

} // namespace COMGR::hotswap
