//===- handle-vop-cross-lane.cpp - Cross-lane VOP helpers ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handle-vop-cross-lane.h"

#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/raise_failure.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cstdint>

using namespace llvm;

namespace COMGR::transpiler {

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

  Module *M = Ctx.B.GetInsertBlock()->getModule();
  // Modeled EXEC can differ from hardware EXEC at this instruction.
  Value *Exec = Ctx.registers().regFile().loadExec(Ctx.B);
  Function *CountTrailingZeros =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::cttz, {Exec->getType()});
  Value *FirstSet = Ctx.B.CreateCall(
      CountTrailingZeros, {Exec, Ctx.B.getFalse()}, "readfirstlane.first.set");
  Value *ExecIsZero = Ctx.B.CreateICmpEQ(
      Exec, ConstantInt::get(Exec->getType(), 0), "readfirstlane.exec.is.zero");
  Value *SourceLane =
      Ctx.B.CreateSelect(ExecIsZero, ConstantInt::get(Exec->getType(), 0),
                         FirstSet, "readfirstlane.source.lane");
  Value *Lane32 = Ctx.B.CreateZExtOrTrunc(SourceLane, Ctx.B.getInt32Ty(),
                                          "readfirstlane.index");
  Value *Result;
  if (Ctx.Projection.providesFullWaveExecInvariant()) {
    Result = emitSourceWaveRead(Ctx, *Src, Lane32, "readfirstlane");
  } else {
    Function *ReadLane = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::amdgcn_readlane, {Ctx.B.getInt32Ty()});
    Result = Ctx.B.CreateCall(ReadLane, {*Src, Lane32}, "readfirstlane");
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
  Expected<Value *> Src = Op.src(0);
  if (!Src)
    return Src.takeError();
  Expected<Value *> Lane = Op.src(1);
  if (!Lane)
    return Lane.takeError();

  Value *Lane32 =
      Ctx.B.CreateZExtOrTrunc(*Lane, Ctx.B.getInt32Ty(), "readlane.index");
  Value *SourceLane = emitSourceWaveLane(Ctx, Lane32, "readlane.source.lane");
  Value *Result = nullptr;
  if (Ctx.Projection.targetWaveSize() == Ctx.Projection.sourceWaveSize()) {
    Module *M = Ctx.B.GetInsertBlock()->getModule();
    Function *ReadLane = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::amdgcn_readlane, {Ctx.B.getInt32Ty()});
    Result = Ctx.B.CreateCall(ReadLane, {*Src, SourceLane}, "readlane");
  } else {
    Result = emitSourceWaveRead(Ctx, *Src, SourceLane, "readlane");
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

  Ctx.registers().regFile().writeReg32(Ctx.B, *Dst, Result);
  return Error::success();
}

} // namespace COMGR::transpiler
