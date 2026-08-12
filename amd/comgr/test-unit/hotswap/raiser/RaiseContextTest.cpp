//===- RaiseContextTest.cpp - raise context unit tests --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise-context.h"

#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <memory>
#include <mutex>

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

class RaiseContextTest : public ::testing::Test {
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
    AllocaRegFile Regs;
    KernargLayout Kernargs;
    UserSgprLayout Layout;
    DenseMap<uint64_t, BasicBlock *> OffsetToBb;
    Function *Kernel;
    std::unique_ptr<RaiseContext> Ctx;

    explicit ContextEnvironment(const MCState &Mc)
        : Mod("raise_context_test", LLVMCtx), B(LLVMCtx),
          Isa(ISAProfile::fromSubtarget(*Mc.SubtargetInfo)),
          Projection(Isa, Isa, B.getInt32Ty(), B.getInt64Ty()),
          Kernel(Function::Create(
              FunctionType::get(B.getVoidTy(), /*isVarArg=*/false),
              Function::ExternalLinkage, "kernel", Mod)) {
      BasicBlock *Entry = BasicBlock::Create(LLVMCtx, "entry", Kernel);
      B.SetInsertPoint(Entry);
      Regs.init(B, B.getInt32Ty(), B.getInt1Ty(), Isa, *Mc.RegInfo, Projection);
      Ctx = std::make_unique<RaiseContext>(
          B, Regs, Projection, Mc, 6, Kernargs, Layout, nullptr, OffsetToBb,
          ArrayRef<uint8_t>(), 0, ArrayRef<TextSection::ImageSection>(), 0, 0);
    }
  };

  MCState Mc;
  std::unique_ptr<ContextEnvironment> Env;
};

TEST_F(RaiseContextTest, ParsesArchitecturalRegisterHalves) {
  auto Check = [&](StringRef Name, ParsedReg::Kind Kind, unsigned Base,
                   unsigned Width) {
    MCRegister Reg = findRegister(*Mc.RegInfo, Name);
    ASSERT_TRUE(Reg) << Name.str();
    DecodedInst Di;
    Di.Inst.setOpcode(findOpcode(*Mc.InstrInfo, "S_MOV_B32"));
    Di.Inst.addOperand(MCOperand::createReg(Reg));
    Expected<ParsedReg> Parsed = Env->Ctx->parseReg(Di, 0);
    ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
    ParsedReg Pr = *Parsed;
    EXPECT_EQ(Pr.RegKind, Kind) << Name.str();
    ASSERT_TRUE(Pr.BaseIdx) << Name.str();
    EXPECT_EQ(*Pr.BaseIdx, Base) << Name.str();
    EXPECT_EQ(Pr.WidthInDwords, Width) << Name.str();
  };

  Check("VCC_LO", ParsedReg::VCC, 0, 1);
  Check("VCC_HI", ParsedReg::VCC, 1, 1);
  Check("VCC", ParsedReg::VCC, 0, 2);
  Check("FLAT_SCR_LO", ParsedReg::FLAT_SCR, 0, 1);
  Check("FLAT_SCR_HI", ParsedReg::FLAT_SCR, 1, 1);
  Check("FLAT_SCR", ParsedReg::FLAT_SCR, 0, 2);
}

TEST_F(RaiseContextTest, KeepsWave32VccHighAsScratch) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  ContextEnvironment Gfx1250(*State);
  MCRegister Reg = findRegister(*State->RegInfo, "VCC_HI");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(findOpcode(*State->InstrInfo, "S_MOV_B32"));
  Di.Inst.addOperand(MCOperand::createReg(Reg));
  Expected<ParsedReg> Parsed = Gfx1250.Ctx->parseReg(Di, 0);
  ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
  ParsedReg Pr = *Parsed;
  EXPECT_EQ(Pr.RegKind, ParsedReg::VCC_HI_SCRATCH);
  EXPECT_EQ(Pr.WidthInDwords, 1u);
}

TEST_F(RaiseContextTest, AppliesVgprMsbsToBothVopdComponents) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  ContextEnvironment Gfx1250(*State);

  unsigned Opc =
      findOpcode(*State->InstrInfo, "V_DUAL_ADD_F32_e32_X_ADD_F32_e32_gfx1250");
  ASSERT_NE(Opc, State->InstrInfo->getNumOpcodes());
  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Gfx1250.Ctx->VgprMsBs = 0xD5;
  Gfx1250.Ctx->computeVGPRAdjust(Di);

  EXPECT_EQ(Gfx1250.Ctx->CurrentVgprAdjust[0], 768u);
  EXPECT_EQ(Gfx1250.Ctx->CurrentVgprAdjust[1], 768u);
  EXPECT_EQ(Gfx1250.Ctx->CurrentVgprAdjust[2], 256u);
  EXPECT_EQ(Gfx1250.Ctx->CurrentVgprAdjust[3], 256u);
  EXPECT_EQ(Gfx1250.Ctx->CurrentVgprAdjust[4], 256u);
  EXPECT_EQ(Gfx1250.Ctx->CurrentVgprAdjust[5], 256u);
  EXPECT_EQ(Gfx1250.Ctx->CurrentVgprAdjust.size(),
            State->InstrInfo->get(Opc).getNumOperands());
}

TEST_F(RaiseContextTest, SizesVgprAdjustmentsFromDescriptor) {
  unsigned Opcode = 0;
  unsigned MaxOperands = 0;
  for (unsigned I = 0; I != Mc.InstrInfo->getNumOpcodes(); ++I) {
    unsigned NumOperands = Mc.InstrInfo->get(I).getNumOperands();
    if (NumOperands > MaxOperands) {
      Opcode = I;
      MaxOperands = NumOperands;
    }
  }
  ASSERT_GT(MaxOperands, 16u);

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Env->Ctx->computeVGPRAdjust(Di);
  EXPECT_EQ(Env->Ctx->CurrentVgprAdjust.size(), MaxOperands);
}

TEST_F(RaiseContextTest, ReportsUnsupportedRegisterOperands) {
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32");
  ASSERT_NE(Opc, Mc.InstrInfo->getNumOpcodes());
  MCRegister Reg = findRegister(*Mc.RegInfo, "SRC_SHARED_BASE_LO");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Di.Inst.addOperand(MCOperand::createReg(Reg));
  Expected<Value *> Result = Env->Ctx->readOp32(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Message = toString(Result.takeError());
  EXPECT_NE(Message.find("register-decode"), std::string::npos);
  EXPECT_NE(Message.find("SRC_SHARED_BASE_LO"), std::string::npos);
}

TEST_F(RaiseContextTest, RejectsXnackMaskOperands) {
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32");
  ASSERT_NE(Opc, Mc.InstrInfo->getNumOpcodes());
  MCRegister Reg = findRegister(*Mc.RegInfo, "XNACK_MASK_LO");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Di.Inst.addOperand(MCOperand::createReg(Reg));
  Expected<Value *> Result = Env->Ctx->readOp32(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Message = toString(Result.takeError());
  EXPECT_NE(Message.find("unsupported-instruction-form"), std::string::npos);
  EXPECT_NE(Message.find("XNACK_MASK_LO"), std::string::npos);

  OpResolver Resolver{*Env->Ctx, Di};
  Expected<ParsedReg> Destination = Resolver.dst();
  ASSERT_FALSE(static_cast<bool>(Destination));
  EXPECT_NE(toString(Destination.takeError()).find("register-decode"),
            std::string::npos);
}

TEST_F(RaiseContextTest, DiscardsNullRegisterWrites) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  ContextEnvironment Gfx1250(*State);
  MCRegister Reg = findRegister(*State->RegInfo, "SGPR_NULL");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(findOpcode(*State->InstrInfo, "S_MOV_B32"));
  Di.Inst.addOperand(MCOperand::createReg(Reg));
  Expected<ParsedReg> Parsed = Gfx1250.Ctx->parseReg(Di, 0);
  ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
  ASSERT_EQ(Parsed->RegKind, ParsedReg::NOREG);

  BasicBlock *Block = Gfx1250.B.GetInsertBlock();
  size_t InstructionCount = Block->size();
  Gfx1250.Ctx->writeReg32(*Parsed, Gfx1250.B.getInt32(1));
  Gfx1250.Ctx->writeReg64(*Parsed, Gfx1250.B.getInt64(1));
  Gfx1250.Ctx->writeRegVec(*Parsed,
                           ConstantVector::getSplat(ElementCount::getFixed(2),
                                                    Gfx1250.B.getInt32(1)));
  Gfx1250.Ctx->writeRegExecWidth(*Parsed, Gfx1250.B.getInt32(1));
  EXPECT_EQ(Block->size(), InstructionCount);
}

TEST_F(RaiseContextTest, InvalidatesOverlappingPairShadows) {
  Env->Ctx->recordSgprWaveMaskI1(4, ConstantInt::getTrue(Env->LLVMCtx), true);
  Env->Ctx->recordSgprWaveMaskI1(6, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Ctx->recordSourceImageSgprPairAddr(4, 0x1000);

  Env->Ctx->invalidateSgprWaveMaskI1(5);

  EXPECT_EQ(Env->Ctx->lookupSgprWaveMaskI1(4), nullptr);
  EXPECT_NE(Env->Ctx->lookupSgprWaveMaskI1(6), nullptr);
  EXPECT_FALSE(Env->Ctx->lookupSourceImageSgprPairAddr(4));
}

TEST_F(RaiseContextTest, MaintainsStateOnRegisterWrites) {
  ParsedReg Sgpr;
  Sgpr.RegKind = ParsedReg::SGPR;
  Sgpr.BaseIdx = 5;
  Env->Ctx->recordSgprWaveMaskI1(4, ConstantInt::getTrue(Env->LLVMCtx), true);
  Env->Ctx->recordSourceImageSgprPairAddr(4, 0x1000);
  Env->Ctx->writeReg32(Sgpr, Env->B.getInt32(1));
  EXPECT_EQ(Env->Ctx->lookupSgprWaveMaskI1(4), nullptr);
  EXPECT_FALSE(Env->Ctx->lookupSourceImageSgprPairAddr(4));

  Sgpr.BaseIdx = 8;
  Sgpr.WidthInDwords = 2;
  Env->Ctx->recordSgprWaveMaskI1(8, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Ctx->recordSgprWaveMaskI1(9, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Ctx->writeRegExecWidth(Sgpr, Env->B.getInt64(1));
  EXPECT_EQ(Env->Ctx->lookupSgprWaveMaskI1(8), nullptr);
  EXPECT_EQ(Env->Ctx->lookupSgprWaveMaskI1(9), nullptr);

  ParsedReg M0;
  M0.RegKind = ParsedReg::M0;
  Env->Ctx->writeReg32(M0, Env->B.getInt32(7));
  EXPECT_EQ(Env->Ctx->getM0Const(), 7u);
  Env->Ctx->writeReg32(M0, UndefValue::get(Env->B.getInt32Ty()));
  EXPECT_FALSE(Env->Ctx->getM0Const());

  Value *OldLaneActive = Env->Ctx->emitLaneActiveBit();
  ParsedReg Exec;
  Exec.RegKind = ParsedReg::EXEC;
  Exec.BaseIdx = 0;
  Env->Ctx->writeReg32(Exec, Env->B.getInt32(1));
  EXPECT_NE(Env->Ctx->emitLaneActiveBit(), OldLaneActive);
}

} // namespace
