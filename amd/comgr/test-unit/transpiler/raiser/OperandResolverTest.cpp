//===- OperandResolverTest.cpp - operand resolver unit tests --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/operand-resolver.h"

#include "transpiler/common/kernel-meta.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/handlers.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/wave-projection.h"

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

#include <climits>
#include <cstdint>
#include <memory>
#include <optional>

using namespace llvm;
using namespace COMGR::transpiler;

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

class OperandResolverTest : public ::testing::Test {
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
    ReplicationProjection Projection;
    Function *Kernel;
    std::optional<RaiseContext> Ctx;

    explicit ContextEnvironment(const MCState &Mc)
        : Mod("operand_resolver_test", LLVMCtx), B(LLVMCtx),
          Projection(*Mc.SubtargetInfo, *Mc.SubtargetInfo, B.getInt32Ty(),
                     B.getInt64Ty()),
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

TEST_F(OperandResolverTest, ReportsRegisterFailures) {
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
  ASSERT_NE(Opc, Mc.InstrInfo->getNumOpcodes());
  MCRegister Reg = findRegister(*Mc.RegInfo, "XNACK_MASK_LO");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Di.Inst.addOperand(MCOperand::createReg(Reg));

  OperandResolver Resolver{*Env->Ctx, Di};
  Expected<ParsedReg> Destination = Resolver.dst();
  ASSERT_FALSE(static_cast<bool>(Destination));
  EXPECT_NE(toString(Destination.takeError()).find("register-decode"),
            std::string::npos);
}

TEST_F(OperandResolverTest, ComparisonRejectsMalformedOperands) {
  unsigned Opcode = findOpcode(*Mc.InstrInfo, "V_CMP_LT_U32_e64_vi");
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());
  MCRegister Pair = findRegister(*Mc.RegInfo, "SGPR4_SGPR5");
  MCRegister Scalar = findRegister(*Mc.RegInfo, "SGPR4");
  MCRegister Vector = findRegister(*Mc.RegInfo, "VGPR0");
  ASSERT_TRUE(Pair && Scalar && Vector);

  MCOperand Source = MCOperand::createReg(Vector);
  MCOperand Destination = MCOperand::createReg(Pair);
  struct TestCase {
    SmallVector<MCOperand> Operands;
    StringRef Detail;
  };
  const TestCase Cases[] = {
      {{}, "expected a comparison mask destination"},
      {{MCOperand::createImm(0), Source, Source},
       "expected a comparison mask destination"},
      {{MCOperand::createReg(MCRegister()), Source, Source},
       "expected a comparison mask destination"},
      {{Source, Source, Source}, "register-decode"},
      {{MCOperand::createReg(Scalar), Source, Source}, "register-decode"},
      {{Destination, Source}, "expected two comparison sources"},
      {{Destination, Source, Destination}, "register-decode"},
  };
  for (const TestCase &Case : Cases) {
    DecodedInst Instruction;
    Instruction.Inst.setOpcode(Opcode);
    Instruction.CanonOp = CanonicalOp::V_CMP_LT_U32;
    Instruction.TargetSpecificFlags = Mc.InstrInfo->get(Opcode).TSFlags;
    Instruction.NumDefs = 1;
    Instruction.SrcMap = {1, 2};
    Instruction.ModMap = {UINT_MAX, UINT_MAX};
    for (MCOperand Operand : Case.Operands)
      Instruction.Inst.addOperand(Operand);
    OperandResolver Resolver{*Env->Ctx, Instruction};
    Error Result = handleVOP3(*Env->Ctx, Instruction, Resolver);
    ASSERT_TRUE(static_cast<bool>(Result));
    EXPECT_NE(toString(std::move(Result)).find(Case.Detail.str()),
              std::string::npos);
  }
}

} // namespace
