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
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>

namespace COMGR {
void ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
  });
}
} // namespace COMGR

using namespace llvm;
using namespace COMGR::hotswap;

namespace {

class RaiseContextTest : public ::testing::Test {
protected:
  // Offset the source kernel starts at. Deliberately not zero: the mapping
  // tracks the kernel's own start, not the start of the text section it sits
  // in.
  static constexpr uint64_t KKernelStartOffset = 0x40;
  // A second leader, standing in for a branch target inside the kernel.
  static constexpr uint64_t KSecondBlockOffset = 0x48;
  // Size of the text section the kernel sits in. The kernel is given no end
  // offset, so this is what bounds its extent.
  static constexpr uint64_t KTextSize = 0x100;

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
    std::array<uint8_t, KTextSize> Text{};
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
      std::set<uint64_t> BlockStarts = {KKernelStartOffset, KSecondBlockOffset};
      Ctx.emplace(cantFail(
          RaiseContext::create(B, Projection, Mc, KernelMeta(), Text, 0,
                               ArrayRef<TextSection::ImageSection>(),
                               KKernelStartOffset, 0, BlockStarts)));
    }
  };

  MCState Mc;
  std::unique_ptr<ContextEnvironment> Env;
};

TEST_F(RaiseContextTest, ResolvesBlocksBySourceOffset) {
  EXPECT_EQ(Env->Ctx->blocks().size(), 2u);
  EXPECT_EQ(Env->Ctx->lookupBB(KKernelStartOffset), Env->Ctx->blocks()[0].Bb);
  EXPECT_EQ(Env->Ctx->lookupBB(KSecondBlockOffset), Env->Ctx->blocks()[1].Bb);
  EXPECT_EQ(Env->Ctx->blocks()[0].Offset, KKernelStartOffset);
  EXPECT_EQ(Env->Ctx->blocks()[1].Offset, KSecondBlockOffset);
}

// The entry block holds the register file, so it must keep no predecessors --
// a source kernel may branch back to its own first instruction. It is
// therefore not one of the blocks a source offset resolves to.
TEST_F(RaiseContextTest, EntryBlockIsNotASourceBlock) {
  EXPECT_NE(Env->Ctx->lookupBB(KKernelStartOffset), Env->Entry);
  for (const RaiseContext::SourceBlock &Block : Env->Ctx->blocks())
    EXPECT_NE(Block.Bb, Env->Entry);
}

TEST_F(RaiseContextTest, OffsetsThatStartNoBlockResolveToNull) {
  EXPECT_EQ(Env->Ctx->findBB(KKernelStartOffset + 4), nullptr);
}

TEST_F(RaiseContextTest, KernelExtentBoundsBranchTargets) {
  EXPECT_FALSE(Env->Ctx->isInKernelExtent(KKernelStartOffset - 4));
  EXPECT_TRUE(Env->Ctx->isInKernelExtent(KKernelStartOffset));
  // A zero end offset runs the kernel to the end of the text section, and stops
  // there: a branch displacement can name an offset past it, or wrap past it.
  EXPECT_TRUE(Env->Ctx->isInKernelExtent(KTextSize - 4));
  EXPECT_FALSE(Env->Ctx->isInKernelExtent(KTextSize));
  EXPECT_FALSE(Env->Ctx->isInKernelExtent(UINT64_MAX));
}

} // namespace
