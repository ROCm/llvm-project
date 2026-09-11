//===- wmma-lowering.cpp - Hotswap matrix remapping ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/wmma-lowering.h"

#include "hotswap/raiser/raise-context.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {
namespace {

Value *emitDSBpermute(IRBuilder<> &B, Module &M, Value *ByteOffset,
                      Value *Source) {
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_ds_bpermute);
  return B.CreateCall(Fn, {ByteOffset, Source}, "bperm");
}

Value *packDwords(IRBuilder<> &B, ArrayRef<Value *> Dwords, Type *TargetTy) {
  FixedVectorType *VectorTy =
      FixedVectorType::get(B.getInt32Ty(), Dwords.size());
  Value *Vector = PoisonValue::get(VectorTy);
  for (auto [Index, Dword] : enumerate(Dwords))
    Vector = B.CreateInsertElement(Vector, Dword, Index, "pack");
  return B.CreateBitCast(Vector, TargetTy, "pack.cast");
}

SmallVector<Value *> unpackDwords(IRBuilder<> &B, Value *Vector,
                                  unsigned Count) {
  FixedVectorType *VectorTy = FixedVectorType::get(B.getInt32Ty(), Count);
  Value *AsI32 = B.CreateBitCast(Vector, VectorTy, "dwords");
  SmallVector<Value *> Result;
  Result.reserve(Count);
  for (unsigned Index = 0; Index != Count; ++Index)
    Result.push_back(B.CreateExtractElement(AsI32, Index, "dword"));
  return Result;
}

Value *selectByLaneGroup(IRBuilder<> &B, Value *LaneGroup, Value *V0, Value *V1,
                         Value *V2, Value *V3) {
  Value *IsGroup2 = B.CreateICmpEQ(LaneGroup, B.getInt32(2), "is.group2");
  Value *Selected = B.CreateSelect(IsGroup2, V2, V3, "group23");
  Value *IsGroup1 = B.CreateICmpEQ(LaneGroup, B.getInt32(1), "is.group1");
  Selected = B.CreateSelect(IsGroup1, V1, Selected, "group123");
  Value *IsGroup0 = B.CreateICmpEQ(LaneGroup, B.getInt32(0), "is.group0");
  return B.CreateSelect(IsGroup0, V0, Selected, "group");
}

// Each target 16-lane group consumes one quarter of the source K dimension.
// The source stores alternating K quarters in its two 16-lane halves and in
// adjacent pairs of its eight fragment dwords.
void redistributeInput(IRBuilder<> &B, Module &M, ArrayRef<Value *> Source,
                       Value *AddrLo, Value *AddrHi, Value *LaneGroup,
                       MutableArrayRef<Value *> Low,
                       MutableArrayRef<Value *> High) {
  for (unsigned Gpr = 0; Gpr != 2; ++Gpr) {
    Value *V0 = emitDSBpermute(B, M, AddrLo, Source[Gpr]);
    Value *V1 = emitDSBpermute(B, M, AddrLo, Source[Gpr + 2]);
    Value *V2 = emitDSBpermute(B, M, AddrHi, Source[Gpr]);
    Value *V3 = emitDSBpermute(B, M, AddrHi, Source[Gpr + 2]);
    Low[Gpr] = selectByLaneGroup(B, LaneGroup, V0, V1, V2, V3);

    Value *U0 = emitDSBpermute(B, M, AddrLo, Source[Gpr + 4]);
    Value *U1 = emitDSBpermute(B, M, AddrLo, Source[Gpr + 6]);
    Value *U2 = emitDSBpermute(B, M, AddrHi, Source[Gpr + 4]);
    Value *U3 = emitDSBpermute(B, M, AddrHi, Source[Gpr + 6]);
    High[Gpr] = selectByLaneGroup(B, LaneGroup, U0, U1, U2, U3);
  }
}

// The source accumulator stores rows 0-7 in its low lane half and rows 8-15
// in its high half. The target stores four rows in each 16-lane group.
void redistributeAccumulator(IRBuilder<> &B, Module &M,
                             ArrayRef<Value *> Source, Value *AddrLo,
                             Value *AddrHi, Value *LaneGroup,
                             MutableArrayRef<Value *> Result) {
  for (unsigned Gpr = 0; Gpr != 4; ++Gpr) {
    Value *V0 = emitDSBpermute(B, M, AddrLo, Source[Gpr]);
    Value *V1 = emitDSBpermute(B, M, AddrLo, Source[Gpr + 4]);
    Value *V2 = emitDSBpermute(B, M, AddrHi, Source[Gpr]);
    Value *V3 = emitDSBpermute(B, M, AddrHi, Source[Gpr + 4]);
    Result[Gpr] = selectByLaneGroup(B, LaneGroup, V0, V1, V2, V3);
  }
}

// Reassemble four target accumulator dwords into the source's eight-dword
// layout. The high source row half comes from target lanes 32-63.
void collectResult(IRBuilder<> &B, Module &M, ArrayRef<Value *> Source,
                   Value *Wave32Lane, MutableArrayRef<Value *> Result) {
  Value *LaneInHalf = B.CreateAnd(Wave32Lane, B.getInt32(15), "lane.in.half");
  Value *IsUpper = B.CreateICmpUGE(Wave32Lane, B.getInt32(16), "lane.upper");
  Value *UpperOffset =
      B.CreateSelect(IsUpper, B.getInt32(32), B.getInt32(0), "upper.offset");
  for (unsigned Gpr = 0; Gpr != 8; ++Gpr) {
    Value *GprOffset = B.getInt32(Gpr >= 4 ? 16 : 0);
    Value *LaneBase = B.CreateAdd(UpperOffset, GprOffset, "collect.base");
    Value *SourceLane = B.CreateAdd(LaneBase, LaneInHalf, "collect.lane");
    Value *Address = B.CreateShl(SourceLane, B.getInt32(2), "collect.address");
    Result[Gpr] = emitDSBpermute(B, M, Address, Source[Gpr % 4]);
  }
}

void runGroupPass(RaiseContext &Ctx, unsigned GroupBase, Value *LaneId,
                  ArrayRef<Value *> A, ArrayRef<Value *> B, ArrayRef<Value *> C,
                  WMMAInputType InputType, MutableArrayRef<Value *> Result) {
  IRBuilder<> &Builder = Ctx.B;
  Module &M = *Builder.GetInsertBlock()->getModule();
  Value *LaneInGroup =
      Builder.CreateAnd(LaneId, Builder.getInt32(15), "matrix.lane");
  Value *LowLane =
      Builder.CreateAdd(LaneInGroup, Builder.getInt32(GroupBase), "low.lane");
  Value *HighLane = Builder.CreateAdd(
      LaneInGroup, Builder.getInt32(GroupBase + 16), "high.lane");
  Value *AddrLo = Builder.CreateShl(LowLane, Builder.getInt32(2), "addr.lo");
  Value *AddrHi = Builder.CreateShl(HighLane, Builder.getInt32(2), "addr.hi");
  Value *LaneGroup =
      Builder.CreateLShr(LaneId, Builder.getInt32(4), "lane.group");

  SmallVector<Value *, 2> ALow(2), AHigh(2), BLow(2), BHigh(2);
  SmallVector<Value *, 4> Accumulator(4);
  redistributeInput(Builder, M, A, AddrLo, AddrHi, LaneGroup, ALow, AHigh);
  redistributeInput(Builder, M, B, AddrLo, AddrHi, LaneGroup, BLow, BHigh);
  redistributeAccumulator(Builder, M, C, AddrLo, AddrHi, LaneGroup,
                          Accumulator);

  Type *InputTy;
  Type *AccumulatorTy;
  Intrinsic::ID IntrinsicId;
  switch (InputType) {
  case WMMAInputType::F16:
    InputTy = FixedVectorType::get(Builder.getHalfTy(), 4);
    AccumulatorTy = FixedVectorType::get(Builder.getFloatTy(), 4);
    IntrinsicId = Intrinsic::amdgcn_mfma_f32_16x16x16f16;
    break;
  case WMMAInputType::BF16:
    InputTy = FixedVectorType::get(Builder.getInt16Ty(), 4);
    AccumulatorTy = FixedVectorType::get(Builder.getFloatTy(), 4);
    IntrinsicId = Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k;
    break;
  case WMMAInputType::IU8:
    InputTy = Builder.getInt64Ty();
    AccumulatorTy = FixedVectorType::get(Builder.getInt32Ty(), 4);
    IntrinsicId = Intrinsic::amdgcn_mfma_i32_16x16x32_i8;
    break;
  }

  Value *ALowPack = packDwords(Builder, ALow, InputTy);
  Value *BLowPack = packDwords(Builder, BLow, InputTy);
  Value *AccumulatorPack = packDwords(Builder, Accumulator, AccumulatorTy);
  Function *Mfma = Intrinsic::getOrInsertDeclaration(&M, IntrinsicId);
  Value *Zero = Builder.getInt32(0);
  Value *FirstCall = Builder.CreateCall(
      Mfma, {ALowPack, BLowPack, AccumulatorPack, Zero, Zero, Zero}, "mfma.0");
  Value *First =
      Ctx.Projection.wrapAsWWMValue(Builder, FirstCall, "mfma.0.wwm");

  Value *AHighPack = packDwords(Builder, AHigh, InputTy);
  Value *BHighPack = packDwords(Builder, BHigh, InputTy);
  Value *SecondCall = Builder.CreateCall(
      Mfma, {AHighPack, BHighPack, First, Zero, Zero, Zero}, "mfma.1");
  Value *Second =
      Ctx.Projection.wrapAsWWMValue(Builder, SecondCall, "mfma.1.wwm");
  SmallVector<Value *> MfmaResult = unpackDwords(Builder, Second, 4);
  Value *Wave32Lane =
      Builder.CreateAnd(LaneId, Builder.getInt32(31), "wave32.lane");
  collectResult(Builder, M, MfmaResult, Wave32Lane, Result);
  for (Value *&Dword : Result)
    Dword = Ctx.Projection.wrapAsWWMValue(Builder, Dword, "matrix.collect.wwm");
}

} // namespace

Expected<Value *> emitWMMAtoMFMA(RaiseContext &Ctx, Value *A, Value *B,
                                 Value *C, WMMAInputType InputType) {
  if (Ctx.Projection.sourceWaveSize() != 32 ||
      Ctx.Projection.targetWaveSize() != 64)
    return createStringError(
        "WMMA to MFMA remapping requires wave32 to wave64");

  unsigned NumSourceWaves = Ctx.Projection.numSourceWavesPerTarget();
  if (NumSourceWaves != 1 && NumSourceWaves != 2)
    return createStringError(
        "WMMA to MFMA remapping requires one or two source waves per target");

  IRBuilder<> &Builder = Ctx.B;
  SmallVector<Value *> ADwords = unpackDwords(Builder, A, 8);
  SmallVector<Value *> BDwords = unpackDwords(Builder, B, 8);
  SmallVector<Value *> CDwords = unpackDwords(Builder, C, 8);
  Value *LaneId = Ctx.emitLaneIdx();

  SmallVector<Value *, 8> FirstResult(8);
  runGroupPass(Ctx, 0, LaneId, ADwords, BDwords, CDwords, InputType,
               FirstResult);
  SmallVector<Value *, 8> FinalResult = FirstResult;
  if (NumSourceWaves == 2) {
    SmallVector<Value *, 8> SecondResult(8);
    runGroupPass(Ctx, 32, LaneId, ADwords, BDwords, CDwords, InputType,
                 SecondResult);
    Value *UseSecond =
        Builder.CreateICmpUGE(LaneId, Builder.getInt32(32), "second.wave");
    for (unsigned I = 0; I != FinalResult.size(); ++I)
      FinalResult[I] = Builder.CreateSelect(UseSecond, SecondResult[I],
                                            FirstResult[I], "matrix.result");
  }

  Type *ElementTy = InputType == WMMAInputType::IU8 ? Builder.getInt32Ty()
                                                    : Builder.getFloatTy();
  Type *ResultTy = FixedVectorType::get(ElementTy, 8);
  return packDwords(Builder, FinalResult, ResultTy);
}

} // namespace COMGR::hotswap
