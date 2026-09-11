//===- RaiseContextTest.cpp - raise context unit tests --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise-context.h"

#include "hotswap/common/kernel-meta.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/handlers.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrInfo.h"
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
  // Offset the source kernel starts at. Deliberately not zero: the mapping
  // tracks the kernel's own start, not the start of the text section it sits
  // in.
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
    ReplicationProjection Projection;
    Function *Kernel;
    BasicBlock *Entry;
    std::optional<RaiseContext> Ctx;

    explicit ContextEnvironment(const MCState &Mc)
        : Mod("raise_context_test", LLVMCtx), B(LLVMCtx),
          Projection(*Mc.SubtargetInfo, *Mc.SubtargetInfo, B.getInt32Ty(),
                     B.getInt64Ty()),
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
  BasicBlock *Start = BasicBlock::Create(Env->LLVMCtx, "bb_start", Env->Kernel);
  Env->Ctx->defineBB(KKernelStartOffset, Start);
  EXPECT_EQ(Env->Ctx->lookupBB(KKernelStartOffset), Start);
}

TEST_F(RaiseContextTest, SetVgprMsbUsesLowImmediateByte) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  ContextEnvironment Gfx1250(*State);

  unsigned Opcode = State->InstrInfo->getNumOpcodes();
  for (unsigned I = 0; I != State->InstrInfo->getNumOpcodes(); ++I) {
    if (State->InstrInfo->getName(I) == "S_SET_VGPR_MSB") {
      Opcode = I;
      break;
    }
  }
  ASSERT_NE(Opcode, State->InstrInfo->getNumOpcodes());

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.Inst.addOperand(MCOperand::createImm(0xABD5));
  Di.CanonOp = CanonicalOp::S_SET_VGPR_MSB;
  Di.TargetSpecificFlags = State->InstrInfo->get(Opcode).TSFlags;
  OperandResolver Resolver{*Gfx1250.Ctx, Di};

  if (Error Err = handleSOPP(*Gfx1250.Ctx, Di, Resolver))
    FAIL() << toString(std::move(Err));
  EXPECT_EQ(Gfx1250.Ctx->registers().vgprMsBs(), 0xD5);

  unsigned MoveOpcode = State->InstrInfo->getNumOpcodes();
  MCRegister Vgpr0;
  MCRegister Vgpr1;
  for (unsigned I = 0; I != State->InstrInfo->getNumOpcodes(); ++I)
    if (State->InstrInfo->getName(I) == "V_MOV_B32_e32")
      MoveOpcode = I;
  for (unsigned I = 1; I != State->RegInfo->getNumRegs(); ++I) {
    StringRef Name = State->RegInfo->getName(I);
    if (Name == "VGPR0")
      Vgpr0 = MCRegister(I);
    else if (Name == "VGPR1")
      Vgpr1 = MCRegister(I);
  }
  ASSERT_NE(MoveOpcode, State->InstrInfo->getNumOpcodes());
  ASSERT_TRUE(Vgpr0);
  ASSERT_TRUE(Vgpr1);

  DecodedInst Move;
  Move.Inst.setOpcode(MoveOpcode);
  Move.Inst.addOperand(MCOperand::createReg(Vgpr1));
  Move.Inst.addOperand(MCOperand::createReg(Vgpr0));
  Gfx1250.Ctx->registers().computeVGPRAdjust(Move);
  Expected<ParsedReg> Destination = Gfx1250.Ctx->registers().parseReg(Move, 0);
  Expected<ParsedReg> Source = Gfx1250.Ctx->registers().parseReg(Move, 1);
  ASSERT_TRUE(static_cast<bool>(Destination))
      << toString(Destination.takeError());
  ASSERT_TRUE(static_cast<bool>(Source)) << toString(Source.takeError());
  EXPECT_EQ(Destination->BaseIdx, 769u);
  EXPECT_EQ(Source->BaseIdx, 256u);
}

} // namespace
