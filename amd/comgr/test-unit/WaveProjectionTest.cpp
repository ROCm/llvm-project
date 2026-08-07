//===- wave_projection_test.cpp - wave_projection unit tests --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Unit tests for `WaveProjection::providesFullWaveExecInvariant()` —
// the virtual method introduced by the WMMA-refusal commit in the
// matmul_fp16 triage.  The method is the contract that
// `handle-valu-vop3p.cpp`'s WMMA → MFMA handlers consult before
// calling `emitWMMAtoMFMA*`; a regression that silently flips the
// return value for either concrete projection would silently
// unleash a WMMA miscompile on phantom-lane kernels (see the
// `wmma_phantom_lane_refuse/` lit fixture for the end-to-end
// regression fence).
//
// These tests cover:
//
//   * `ModuloReplicationProjection::providesFullWaveExecInvariant()`
//     inherits the base default of `false` — MODREP's
//     `emitInitialExec` returns source-width all-ones and does
//     NOT call `@llvm.amdgcn.init_whole_wave`, so hardware EXEC
//     stays at whatever the dispatcher set it to (the source-
//     wave active mask for a partial-wave launch).  Collective
//     lowerings that need all target-wave lanes active (WMMA →
//     MFMA redistribute-collect) must refuse when this is false.
//
//   * `WaveNativeProjection::providesFullWaveExecInvariant()`
//     overrides to `true` because
//     `WaveNativeProjection::emitInitialExec` explicitly emits
//     `@llvm.amdgcn.init_whole_wave` which sets HW EXEC = -1 for
//     the remainder of the kernel body.
//
//   * A base-class pointer dispatch works correctly — callers in
//     `handle-valu-vop3p.cpp` hold a `WaveProjection &` reference
//     and rely on virtual dispatch to pick the right answer per
//     kernel.
//
//   * `ThreadLoopProjection` is conservative-by-default and does not
//     provide the full-wave EXEC invariant. It still uses target-width
//     EXEC storage so source-wave predicate masks can be banked per
//     packed source wave.

#include "hotswap/wave-projection.h"

#include "hotswap/isa-profile.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "gtest/gtest.h"

using namespace llvm;
using COMGR::hotswap::ISAProfile;
using COMGR::hotswap::ModuloReplicationProjection;
using COMGR::hotswap::ThreadLoopProjection;
using COMGR::hotswap::WaveNativeProjection;
using COMGR::hotswap::WaveProjection;

namespace {

// Canonical source / target ISA profiles for the cross-widening
// direction the two projections cover.  Matches the source
// (gfx1250, wave32) / target (gfx942, wave64) cross-widening the
// matmul_fp16 triage surfaced the bug on.
//
// Constructed via `ISAProfile::forTesting(waveSize)` rather than
// `fromSubtarget` because standing up an `MCSubtargetInfo` would
// require the full LLVM AMDGPU init chain (multiple
// `InitializeAllTarget*` calls + target-lookup dance) and buys
// us nothing for the contract check this file pins — only the
// `waveSize` dimension is consulted by `WaveNativeProjection`'s
// direction-gate assertion, and
// `providesFullWaveExecInvariant()`'s return value is independent
// of the other feature flags.  See the `forTesting` docstring in
// `isa-profile.h` for the test-only scope of this factory.
ISAProfile makeGfx1250Profile() { return ISAProfile::forTesting(32); }
ISAProfile makeGfx942Profile() { return ISAProfile::forTesting(64); }

} // namespace

// ----------------------------------------------------------------------------
// MODREP: the base-class default of `false` is inherited; the
// projection does NOT decouple HW EXEC from the source active
// mask.  Pins the default so a future refactor that accidentally
// promotes MODREP to full-wave-EXEC semantics (without actually
// emitting `init_whole_wave`) would be caught here rather than by
// the WMMA handler silently accepting MODREP kernels and
// miscompiling them.
// ----------------------------------------------------------------------------
TEST(WaveProjectionContract, ModuloReplicationDoesNotProvideFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();

  ModuloReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesFullWaveExecInvariant());

  // Cross-check through a base-class reference to confirm the
  // virtual dispatch resolves to the correct override (or lack
  // thereof — MODREP inherits the base's `false`).
  const WaveProjection &Base = Proj;
  EXPECT_FALSE(Base.providesFullWaveExecInvariant());
}

// ----------------------------------------------------------------------------
// WaveNative: overrides to `true` because `emitInitialExec` emits
// `@llvm.amdgcn.init_whole_wave` which forces HW EXEC = -1 for the
// kernel body.  Pins the override so a future refactor that
// removes it (or moves the `init_whole_wave` emission site without
// updating the contract method) is caught here before it can
// silently gate off the WMMA handlers' acceptance of
// WaveNative-raised kernels.
// ----------------------------------------------------------------------------
TEST(WaveProjectionContract, WaveNativeProvidesFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  // WaveNativeProjection's constructor asserts `src.isWave32() &&
  // !tgt.isWave32()` (per its docstring — this projection is only
  // defined for wave32 → wave64 cross-widening), so we construct
  // with the canonical gfx1250 → gfx942 pair.  Any other direction
  // would fatal-error inside the constructor.
  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();

  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_TRUE(Proj.providesFullWaveExecInvariant());

  const WaveProjection &Base = Proj;
  EXPECT_TRUE(Base.providesFullWaveExecInvariant());
}

// ----------------------------------------------------------------------------
// The base class's default return value is `false`.  Tested through
// `ModuloReplicationProjection` above (which inherits the default),
// but also pinned explicitly here via a minimal subclass that
// implements only the pure virtuals.  This protects against a
// future author flipping the default to `true` and thereby silently
// graduating un-audited projection classes to "allowed to run WMMA"
// status.
// ----------------------------------------------------------------------------
namespace {
// Minimal concrete `WaveProjection` that only satisfies the pure
// virtuals required to instantiate — it is not a real projection and
// is used SOLELY to assert the base class's default return values
// for the concrete-but-permissive contract methods.
class DefaultTestProjection final : public WaveProjection {
public:
  using WaveProjection::WaveProjection;
  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &,
                                  llvm::Value *) const override {
    return nullptr;  // unused by these tests
  }
  llvm::Value *ballotI1ToWidth(llvm::IRBuilder<> &, llvm::Value *,
                                llvm::Type *,
                                const llvm::Twine &) const override {
    return nullptr;  // unused by these tests
  }
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &,
                                           llvm::Value *) const override {
    return nullptr;  // unused by these tests
  }
};
} // namespace

TEST(WaveProjectionContract, BaseDefaultIsNotFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();

  DefaultTestProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesFullWaveExecInvariant());
}

TEST(WaveProjectionContract, ThreadLoopDoesNotProvideFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();

  ThreadLoopProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesFullWaveExecInvariant());
  EXPECT_EQ(Proj.execStorageTy(), I64Ty);

  const WaveProjection &Base = Proj;
  EXPECT_FALSE(Base.providesFullWaveExecInvariant());
}

// ----------------------------------------------------------------------------
// `numSourceWavesPerTarget()` contract.  The WMMA → MFMA lowering in
// `wmma-lowering.cpp` iterates `groupBase ∈ {0, W_src, ..., (N-1) *
// W_src}` where N is this value, so a regression that flipped the
// return for MODREP (from 1 to 2) would synthesise a bogus pass-1
// MFMA reading undef from phantom lanes.  A regression that flipped
// WaveNative's return (from 2 to 1) would skip the second source
// wave's matmul entirely and leave target lanes 32..63 with
// uncomputed results.
// ----------------------------------------------------------------------------
TEST(WaveProjectionContract, ModuloReplicationHasOneSourceWavePerTarget) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();

  ModuloReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 1u);

  const WaveProjection &Base = Proj;
  EXPECT_EQ(Base.numSourceWavesPerTarget(), 1u);
}

TEST(WaveProjectionContract, WaveNativeHasTwoSourceWavesPerTarget) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();

  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 2u);

  const WaveProjection &Base = Proj;
  EXPECT_EQ(Base.numSourceWavesPerTarget(), 2u);
}

TEST(WaveProjectionContract, ThreadLoopReportsSourceWavesPerTargetRatio) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();

  ThreadLoopProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 2u);

  const WaveProjection &Base = Proj;
  EXPECT_EQ(Base.numSourceWavesPerTarget(), 2u);
}

// ----------------------------------------------------------------------------
// `wrapAsWWMValue` emission contract.  On projections that DO provide
// the full-wave-EXEC invariant kernel-wide (WaveNative) the helper is
// an identity no-op — it MUST NOT emit a `strict.wwm` marker.  On
// projections that do NOT (ModuloReplication) it emits exactly one
// `@llvm.amdgcn.strict.wwm` call wrapping its input value.  These
// tests pin the projection-aware behaviour by synthesising a tiny IR
// function, invoking the helper, and inspecting the resulting SSA
// shape directly.  The gate landing in `wmma-lowering.cpp` depends on
// the no-op behaviour on WaveNative to avoid the regalloc blow-up
// described in `WaveProjection::emitInitialExec`'s block comment; a
// regression that silently flips WaveNative to emit a marker (or
// silently flips MODREP to skip the marker) would silently re-open
// the 128×128-f16-matmul regalloc failure OR silently drop the
// MODREP-phantom-lane WMMA correctness scope respectively.
// ----------------------------------------------------------------------------
namespace {
// Minimal IR scaffold: a function `@f(i32 %x)` with a single block.
// The helper is invoked with the argument; the test inspects the
// returned value and surrounding instructions.
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

TEST(WaveProjectionContract, WrapAsWWMValueIsNoOpOnWaveNative) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();
  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Result = Proj.wrapAsWWMValue(S.B, S.Arg);

  // Identity — returned value IS the input, no new instruction emitted.
  EXPECT_EQ(Result, S.Arg);
  // The entry block has no instructions beyond the (implicit) label.
  EXPECT_TRUE(S.BB->empty())
      << "wrapAsWWMValue on WaveNativeProjection must not emit any "
         "instructions; found " << S.BB->size();
}

TEST(WaveProjectionContract, WrapAsWWMValueEmitsStrictWWMOnMODREP) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();
  ModuloReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Result = Proj.wrapAsWWMValue(S.B, S.Arg);

  // Result is NOT the input — a strict.wwm call wraps it.
  EXPECT_NE(Result, S.Arg);
  auto *Cb = dyn_cast<CallInst>(Result);
  ASSERT_NE(Cb, nullptr)
      << "wrapAsWWMValue on MODREP must return a CallInst wrapping "
         "the input value";

  Function *Callee = Cb->getCalledFunction();
  ASSERT_NE(Callee, nullptr);
  EXPECT_EQ(Callee->getIntrinsicID(), Intrinsic::amdgcn_strict_wwm)
      << "wrapAsWWMValue must emit @llvm.amdgcn.strict.wwm, got "
      << Callee->getName().str();
  // One argument — the wrapped input.
  ASSERT_EQ(Cb->arg_size(), 1u);
  EXPECT_EQ(Cb->getArgOperand(0), S.Arg);
  // The overload resolution picked the i32 variant.
  EXPECT_EQ(Cb->getType(), I32Ty)
      << "wrapAsWWMValue returned a value of the wrong type for the "
         "i32 input overload";

  // The emitted instruction should be the only one in the block.
  EXPECT_EQ(S.BB->size(), 1u)
      << "wrapAsWWMValue on MODREP should emit exactly one instruction; "
         "found " << S.BB->size();
}

// ----------------------------------------------------------------------------
// Polymorphic-overload coverage for `wrapAsWWMValue`.  `wmma-lowering.cpp`
// calls the helper with BOTH `i32` (collect output dwords) and
// `<4 x float>` (MFMA outputs before the unpackDwords -> collect chain).
// If a future LLVM version drops or narrows the `strict.wwm` overload set
// for vector fp types, only the scalar-only coverage above would catch
// the breakage — the MFMA-output path would silently fail to emit the
// intrinsic, and SIWholeQuadMode would never pull the MFMA into a WWM
// region, quietly reintroducing the rows-8..15-zero miscompile this
// helper exists to prevent.  This test exercises the vector overload
// directly by constructing a `<4 x float>` argument and asserting the
// intrinsic is invoked with the matching overloaded type.
// ----------------------------------------------------------------------------
TEST(WaveProjectionContract, WrapAsWWMValueHandlesVectorFloatOverload) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);
  auto *F32Ty = Type::getFloatTy(S.Ctx);
  auto *V4f32Ty = FixedVectorType::get(F32Ty, 4);

  ISAProfile Src = makeGfx1250Profile();
  ISAProfile Tgt = makeGfx942Profile();
  ModuloReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);

  // Build a <4 x float> value inside the scaffold's entry block so
  // `wrapAsWWMValue` has a local SSA Value to wrap.  The value itself
  // doesn't matter — we only inspect the emitted intrinsic call.
  Value *Vec = PoisonValue::get(V4f32Ty);
  Value *Result = Proj.wrapAsWWMValue(S.B, Vec);

  auto *Cb = dyn_cast<CallInst>(Result);
  ASSERT_NE(Cb, nullptr);
  Function *Callee = Cb->getCalledFunction();
  ASSERT_NE(Callee, nullptr);
  EXPECT_EQ(Callee->getIntrinsicID(), Intrinsic::amdgcn_strict_wwm);
  EXPECT_EQ(Cb->getType(), V4f32Ty)
      << "wrapAsWWMValue with a <4 x float> input must emit the "
         "<4 x float>-overloaded strict.wwm variant";
  ASSERT_EQ(Cb->arg_size(), 1u);
  EXPECT_EQ(Cb->getArgOperand(0)->getType(), V4f32Ty)
      << "operand type mismatch — strict.wwm's overload must match "
         "the input type";
}
