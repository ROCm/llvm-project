//===- OpResolverTest.cpp - operand resolver unit tests -------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/op-resolver.h"

#include "hotswap/common/kernel-meta.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>

using namespace llvm;
using namespace COMGR::hotswap;

namespace {

unsigned findOpcode(const MCInstrInfo &MII, StringRef Name) {
  for (unsigned Opc = 0; Opc != MII.getNumOpcodes(); ++Opc)
    if (MII.getName(Opc) == Name)
      return Opc;
  return MII.getNumOpcodes();
}

MCRegister findRegister(const MCRegisterInfo &MRI, StringRef Name) {
  for (unsigned Reg = 1; Reg != MRI.getNumRegs(); ++Reg)
    if (Name == MRI.getName(Reg))
      return MCRegister(Reg);
  return MCRegister();
}

class OpResolverTest : public ::testing::Test {
protected:
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
    std::optional<RaiseContext> Ctx;

    explicit ContextEnvironment(const MCState &Mc)
        : Mod("op_resolver_test", LLVMCtx), B(LLVMCtx),
          Isa(ISAProfile::fromSubtarget(*Mc.SubtargetInfo)),
          Projection(Isa, Isa, B.getInt32Ty(), B.getInt64Ty()),
          Kernel(Function::Create(
              FunctionType::get(B.getVoidTy(), /*isVarArg=*/false),
              Function::ExternalLinkage, "kernel", Mod)) {
      B.SetInsertPoint(BasicBlock::Create(LLVMCtx, "entry", Kernel));
      Ctx.emplace(cantFail(RaiseContext::create(
          B, Projection, Mc, KernelMeta(), ArrayRef<uint8_t>(), 0,
          ArrayRef<TextSection::ImageSection>(), 0, 0)));
    }
  };

  MCState Mc;
  std::unique_ptr<ContextEnvironment> Env;
};

TEST_F(OpResolverTest, ReportsRegisterFailures) {
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
  ASSERT_NE(Opc, Mc.InstrInfo->getNumOpcodes());
  MCRegister Reg = findRegister(*Mc.RegInfo, "XNACK_MASK_LO");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Di.Inst.addOperand(MCOperand::createReg(Reg));

  OpResolver Resolver{*Env->Ctx, Di};
  Expected<ParsedReg> Destination = Resolver.dst();
  ASSERT_FALSE(static_cast<bool>(Destination));
  EXPECT_NE(toString(Destination.takeError()).find("register-decode"),
            std::string::npos);
}

} // namespace
