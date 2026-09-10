//===- WaveProjectionTest.cpp - wave projection unit tests ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Unit tests for the per-projection contract on WaveProjection and its
// subclasses. Each projection fixes its policy (EXEC storage width, source
// waves per target wave, doubled-dispatch factor, and the exec/mbcnt
// predicates) in its constructor, and the raiser and its passes branch on
// those values through a WaveProjection reference. These tests pin the value
// each projection reports, so a constructor that forgets to set a flag, or a
// change that flips a default, is caught here rather than downstream.
//
// The IR-emitting side of the interface (wrapAsWWMValue) is exercised on a
// small synthetic function.

#include "hotswap/raiser/wave-projection.h"

#include "hotswap/decoder/mc-state.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "gtest/gtest.h"

using namespace llvm;
using COMGR::hotswap::initMCState;
using COMGR::hotswap::MCState;
using COMGR::hotswap::ReplicationDoubledDispatchProjection;
using COMGR::hotswap::ReplicationProjection;
using COMGR::hotswap::ThreadLoopProjection;
using COMGR::hotswap::WaveNativeProjection;
using COMGR::hotswap::WaveProjection;

namespace {

// Every projection here covers the widening direction: a wave32 source
// (gfx1250) onto a wave64 target (gfx942). The projections consult only the
// wave size from live MCSubtargetInfo objects, so the fixture stands up the
// real AMDGPU MC stack for both ISAs.
class WaveProjectionContract : public ::testing::Test {
protected:
  void SetUp() override {
    Expected<MCState> SrcState = initMCState("gfx1250");
    ASSERT_TRUE(static_cast<bool>(SrcState)) << toString(SrcState.takeError());
    SrcMc = std::move(*SrcState);

    Expected<MCState> TgtState = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(TgtState)) << toString(TgtState.takeError());
    TgtMc = std::move(*TgtState);
  }

  const MCSubtargetInfo &srcSTI() const { return *SrcMc.SubtargetInfo; }
  const MCSubtargetInfo &tgtSTI() const { return *TgtMc.SubtargetInfo; }

  MCState SrcMc;
  MCState TgtMc;
};

} // namespace

// ----------------------------------------------------------------------------
// providesFullWaveExecInvariant: only WaveNative decouples hardware EXEC from
// the modeled source mask (via init_whole_wave), so only it reports true.
// Cross-lane collective lowerings gate on this before running.
// ----------------------------------------------------------------------------
TEST_F(WaveProjectionContract, ReplicationDoesNotProvideFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesFullWaveExecInvariant());
  EXPECT_EQ(&Proj.SourceSTI, &Src);
  EXPECT_EQ(&Proj.TargetSTI, &Tgt);
  EXPECT_EQ(Proj.sourceWaveSize(), 32u);
  EXPECT_EQ(Proj.targetWaveSize(), 64u);

  // Read through a base reference too: the value lives in the base subobject
  // the constructor set, so the non-virtual accessor resolves it correctly.
  const WaveProjection &Base = Proj;
  EXPECT_FALSE(Base.providesFullWaveExecInvariant());
}

TEST_F(WaveProjectionContract, WaveNativeProvidesFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  // The constructor asserts a wave32 -> wave64 direction.
  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_TRUE(Proj.providesFullWaveExecInvariant());
  EXPECT_EQ(Proj.execStorageTy(), I64Ty);
  EXPECT_TRUE(Proj.broadcastNarrowExecLoWrite());
  EXPECT_TRUE(Proj.preservesMbcntDerivedExec());

  const WaveProjection &Base = Proj;
  EXPECT_TRUE(Base.providesFullWaveExecInvariant());
}

// A minimal concrete projection that implements only the pure virtuals, used
// to pin the base-class defaults for the constructor-set configuration.
namespace {
class DefaultTestProjection final : public WaveProjection {
public:
  using WaveProjection::WaveProjection;
  llvm::Value *emitSourceWaveId(llvm::IRBuilder<> &) const override {
    return nullptr;
  }
  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &,
                                 llvm::Value *) const override {
    return nullptr;
  }
  llvm::Value *ballotI1ToWidth(llvm::IRBuilder<> &, llvm::Value *, llvm::Type *,
                               const llvm::Twine &) const override {
    return nullptr;
  }
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &,
                                          llvm::Value *) const override {
    return nullptr;
  }
};
} // namespace

TEST_F(WaveProjectionContract, BaseDefaults) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  DefaultTestProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesFullWaveExecInvariant());
  EXPECT_FALSE(Proj.usesDoubledDispatch());
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 1u);
}

TEST_F(WaveProjectionContract, ThreadLoop) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ThreadLoopProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesFullWaveExecInvariant());
  EXPECT_TRUE(Proj.sourceWaveScopedLaneOps());
  EXPECT_EQ(Proj.execStorageTy(), I64Ty);

  const WaveProjection &Base = Proj;
  EXPECT_FALSE(Base.providesFullWaveExecInvariant());
}

// ----------------------------------------------------------------------------
// numSourceWavesPerTarget: the per-source-wave pass count. A wrong value makes
// callers synthesise the wrong number of passes (a bogus second-wave pass for
// replication, or a skipped second wave for the widening projections).
// ----------------------------------------------------------------------------
TEST_F(WaveProjectionContract, ReplicationHasOneSourceWavePerTarget) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 1u);
}

TEST_F(WaveProjectionContract, WaveNativeHasTwoSourceWavesPerTarget) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 2u);
}

TEST_F(WaveProjectionContract, ThreadLoopReportsSourceWavesPerTargetRatio) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ThreadLoopProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 2u);
}

// ----------------------------------------------------------------------------
// Doubled dispatch: ReplicationDoubledDispatchProjection is the only projection
// that asks the runtime to scale the block's x extent. The factor is W_t / W_s.
// ----------------------------------------------------------------------------
TEST_F(WaveProjectionContract, ReplicationDoubledDispatch) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ReplicationDoubledDispatchProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_TRUE(Proj.usesDoubledDispatch());
  EXPECT_EQ(Proj.doubledDispatchFactor(), 2u);
  EXPECT_EQ(Proj.doubledDispatchDim(), 0u);
  // Doubled dispatch is replication underneath: one source wave per
  // target wave, upper lanes are replicas.
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 1u);

  // Plain replication must not report a doubled dispatch.
  ReplicationProjection Plain(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Plain.usesDoubledDispatch());
}

// ----------------------------------------------------------------------------
// wrapAsWWMValue: an identity no-op on projections that already guarantee
// hardware EXEC=-1 kernel-wide (WaveNative), and a single strict.wwm wrapper on
// those that do not (Replication). The WMMA lowering relies on both behaviours.
// ----------------------------------------------------------------------------
namespace {
struct IRScaffold {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  Argument *Arg;
  IRBuilder<> B;

  IRScaffold() : Ctx(), M(std::make_unique<Module>("t", Ctx)), B(Ctx) {
    auto *I32Ty = Type::getInt32Ty(Ctx);
    auto *FnTy = FunctionType::get(Type::getVoidTy(Ctx), {I32Ty}, false);
    F = Function::Create(FnTy, Function::ExternalLinkage, "f", M.get());
    BB = BasicBlock::Create(Ctx, "entry", F);
    Arg = F->getArg(0);
    B.SetInsertPoint(BB);
  }
};
} // namespace

TEST_F(WaveProjectionContract, WrapAsWWMValueIsNoOpOnWaveNative) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();
  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Result = Proj.wrapAsWWMValue(S.B, S.Arg);
  EXPECT_EQ(Result, S.Arg);
  EXPECT_TRUE(S.BB->empty());
}

TEST_F(WaveProjectionContract, WrapAsWWMValueEmitsStrictWWMOnReplication) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();
  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Result = Proj.wrapAsWWMValue(S.B, S.Arg);
  EXPECT_NE(Result, S.Arg);
  auto *Cb = dyn_cast<CallInst>(Result);
  ASSERT_NE(Cb, nullptr);
  Function *Callee = Cb->getCalledFunction();
  ASSERT_NE(Callee, nullptr);
  EXPECT_EQ(Callee->getIntrinsicID(), Intrinsic::amdgcn_strict_wwm);
  ASSERT_EQ(Cb->arg_size(), 1u);
  EXPECT_EQ(Cb->getArgOperand(0), S.Arg);
  EXPECT_EQ(Cb->getType(), I32Ty);
  EXPECT_EQ(S.BB->size(), 1u);
}

// wmma-lowering wraps both i32 (result dwords) and <4 x float> (MFMA outputs),
// so cover the vector overload too.
TEST_F(WaveProjectionContract, WrapAsWWMValueHandlesVectorFloatOverload) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);
  auto *F32Ty = Type::getFloatTy(S.Ctx);
  auto *V4f32Ty = FixedVectorType::get(F32Ty, 4);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();
  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Vec = PoisonValue::get(V4f32Ty);
  Value *Result = Proj.wrapAsWWMValue(S.B, Vec);

  auto *Cb = dyn_cast<CallInst>(Result);
  ASSERT_NE(Cb, nullptr);
  Function *Callee = Cb->getCalledFunction();
  ASSERT_NE(Callee, nullptr);
  EXPECT_EQ(Callee->getIntrinsicID(), Intrinsic::amdgcn_strict_wwm);
  EXPECT_EQ(Cb->getType(), V4f32Ty);
  ASSERT_EQ(Cb->arg_size(), 1u);
  EXPECT_EQ(Cb->getArgOperand(0)->getType(), V4f32Ty);
}
