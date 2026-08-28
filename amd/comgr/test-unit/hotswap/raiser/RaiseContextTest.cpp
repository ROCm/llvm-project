//===- RaiseContextTest.cpp - raise context unit tests --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise-context.h"

#include "hotswap/common/kernel-meta.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>

using namespace llvm;
using namespace COMGR::hotswap;

namespace {

class RaiseContextTest : public ::testing::Test {
protected:
  // Offset the source kernel starts at, which the context maps to the entry
  // block. Deliberately not zero: the mapping tracks the kernel's own start,
  // not the start of the text section it sits in.
  static constexpr uint64_t KKernelStartOffset = 0x40;

  void SetUp() override {
    Expected<MCState> State = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
    Mc = std::move(*State);
    Env = std::make_unique<ContextEnvironment>(Mc);
  }

  struct ContextEnvironment {
    LLVMContext LLVMCtx;
    Module Mod;
    IRBuilder<> B;
    ISAProfile Isa;
    ReplicationProjection Projection;
    Function *Kernel;
    BasicBlock *Entry;
    std::optional<RaiseContext> Ctx;

    explicit ContextEnvironment(const MCState &Mc)
        : Mod("raise_context_test", LLVMCtx), B(LLVMCtx),
          Isa(ISAProfile::fromSubtarget(*Mc.SubtargetInfo)),
          Projection(Isa, Isa, B.getInt32Ty(), B.getInt64Ty()),
          Kernel(Function::Create(
              FunctionType::get(B.getVoidTy(), /*isVarArg=*/false),
              Function::ExternalLinkage, "kernel", Mod)),
          Entry(BasicBlock::Create(LLVMCtx, "entry", Kernel)) {
      B.SetInsertPoint(Entry);
      Ctx.emplace(cantFail(RaiseContext::create(
          B, Projection, Mc, KernelMeta(), ArrayRef<uint8_t>(), 0,
          ArrayRef<TextSection::ImageSection>(), KKernelStartOffset, 0)));
    }
  };

  MCState Mc;
  std::unique_ptr<ContextEnvironment> Env;
};

TEST_F(RaiseContextTest, ResolvesBlocksBySourceOffset) {
  EXPECT_EQ(Env->Ctx->lookupBB(KKernelStartOffset), Env->Entry);
}

} // namespace
