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
#include "hotswap/raiser/raise_failure.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

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

class FullWaveInvariantProjection : public ReplicationProjection {
public:
  FullWaveInvariantProjection(const ISAProfile &Isa, Type *I32Ty, Type *I64Ty,
                              bool FullWaveInvariant)
      : ReplicationProjection(Isa, Isa, I32Ty, I64Ty) {
    ProvidesFullWaveExecInvariant = FullWaveInvariant;
  }
};

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
    std::unique_ptr<WaveProjection> Projection;
    // Declared before Ctx, which holds views into this metadata.
    KernelMeta Meta;
    Function *Kernel;
    std::optional<RaiseContext> Ctx;

    explicit ContextEnvironment(const MCState &Mc, KernelMeta Meta = {},
                                bool FullWaveInvariant = false)
        : Mod("raise_context_test", LLVMCtx), B(LLVMCtx),
          Isa(ISAProfile::fromSubtarget(*Mc.SubtargetInfo)),
          Projection(std::make_unique<FullWaveInvariantProjection>(
              Isa, B.getInt32Ty(), B.getInt64Ty(), FullWaveInvariant)),
          Meta(std::move(Meta)),
          Kernel(Function::Create(
              FunctionType::get(B.getVoidTy(), /*isVarArg=*/false),
              Function::ExternalLinkage, "kernel", Mod)) {
      initialize(Mc);
    }

    ContextEnvironment(const MCState &SourceMc, const MCState &TargetMc)
        : Mod("raise_context_test", LLVMCtx), B(LLVMCtx),
          Isa(ISAProfile::fromSubtarget(*SourceMc.SubtargetInfo)),
          Projection(std::make_unique<WaveNativeProjection>(
              Isa, ISAProfile::fromSubtarget(*TargetMc.SubtargetInfo),
              B.getInt32Ty(), B.getInt64Ty())),
          Kernel(Function::Create(
              FunctionType::get(B.getVoidTy(), /*isVarArg=*/false),
              Function::ExternalLinkage, "kernel", Mod)) {
      initialize(SourceMc);
    }

    void initialize(const MCState &Mc) {
      BasicBlock *Entry = BasicBlock::Create(LLVMCtx, "entry", Kernel);
      B.SetInsertPoint(Entry);
      Ctx.emplace(cantFail(RaiseContext::create(
          B, *Projection, Mc, Meta, 6, DenseMap<uint64_t, BasicBlock *>(),
          nullptr, ArrayRef<uint8_t>(), 0,
          ArrayRef<TextSection::ImageSection>(), 0, 0)));
    }
  };

  // Kernel metadata enabling the kernarg segment pointer user SGPR.
  static KernelMeta makeKernargPtrMeta() {
    using namespace llvm::amdhsa;
    KernelMeta Meta;
    Meta.Name = "k";
    Meta.KernelCodeProperties =
        KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
    Meta.ComputePgmRsrc2 =
        2u << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT;
    return Meta;
  }

  static void promoteAndFold(ContextEnvironment &Environment) {
    SmallVector<AllocaInst *> Allocas;
    Environment.Ctx->collectAllocas(Allocas);
    DominatorTree DT(*Environment.Kernel);
    PromoteMemToReg(Allocas, DT);

    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (BasicBlock &BB : *Environment.Kernel) {
        for (Instruction &Instruction : make_early_inc_range(BB)) {
          if (Instruction.isTerminator()) {
            continue;
          }
          if (Constant *Folded = ConstantFoldInstruction(
                  &Instruction, Environment.Mod.getDataLayout())) {
            Instruction.replaceAllUsesWith(Folded);
            Instruction.eraseFromParent();
            Changed = true;
          }
        }
      }
    }
  }

  MCState Mc;
  std::unique_ptr<ContextEnvironment> Env;
};

TEST_F(RaiseContextTest, ParsesArchitecturalRegisterHalves) {
  auto Check = [&](StringRef OpcodeName, StringRef Name, ParsedReg::Kind Kind,
                   unsigned Base, unsigned Width) {
    MCRegister Reg = findRegister(*Mc.RegInfo, Name);
    ASSERT_TRUE(Reg) << Name.str();
    DecodedInst Di;
    Di.Inst.setOpcode(findOpcode(*Mc.InstrInfo, OpcodeName));
    Di.Inst.addOperand(MCOperand::createReg(Reg));
    Expected<ParsedReg> Parsed = Env->Ctx->parseReg(Di, 0);
    ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
    ParsedReg Pr = *Parsed;
    EXPECT_EQ(Pr.RegKind, Kind) << Name.str();
    ASSERT_TRUE(Pr.BaseIdx) << Name.str();
    EXPECT_EQ(*Pr.BaseIdx, Base) << Name.str();
    EXPECT_EQ(Pr.WidthInDwords, Width) << Name.str();
  };

  Check("S_MOV_B32", "VCC_LO", ParsedReg::VCC, 0, 1);
  Check("S_MOV_B32", "VCC_HI", ParsedReg::VCC, 1, 1);
  Check("S_MOV_B64", "VCC", ParsedReg::VCC, 0, 2);
  Check("S_MOV_B32", "FLAT_SCR_LO", ParsedReg::FLAT_SCR, 0, 1);
  Check("S_MOV_B32", "FLAT_SCR_HI", ParsedReg::FLAT_SCR, 1, 1);
  Check("S_MOV_B64", "FLAT_SCR", ParsedReg::FLAT_SCR, 0, 2);
}

TEST_F(RaiseContextTest, ParsesWideRegisterOperands) {
  const unsigned Opcode = findOpcode(*Mc.InstrInfo, "S_LOAD_DWORDX8_IMM");
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());
  const MCInstrDesc &Descriptor = Mc.InstrInfo->get(Opcode);
  ASSERT_GT(Descriptor.getNumOperands(), 0u);
  const MCOperandInfo &OperandInfo = Descriptor.operands()[0];
  const int16_t RegisterClassID = Mc.InstrInfo->getOpRegClassID(
      OperandInfo,
      Mc.SubtargetInfo->getHwMode(MCSubtargetInfo::HwMode_RegInfo));
  ASSERT_GE(RegisterClassID, 0);
  const MCRegisterClass &RegisterClass =
      Mc.RegInfo->getRegClass(RegisterClassID);
  ASSERT_GT(RegisterClass.getNumRegs(), 0u);

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.Inst.addOperand(MCOperand::createReg(RegisterClass.getRegister(0)));
  Expected<ParsedReg> Parsed = Env->Ctx->parseReg(Di, 0);
  ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
  EXPECT_EQ(Parsed->RegKind, ParsedReg::SGPR);
  EXPECT_EQ(Parsed->WidthInDwords, 8u);
}

TEST_F(RaiseContextTest, RejectsRegistersOutsideOperandClass) {
  const unsigned Opcode = findOpcode(*Mc.InstrInfo, "S_LOAD_DWORDX8_IMM_vi");
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());
  const MCRegister Register = findRegister(*Mc.RegInfo, "VGPR0");
  ASSERT_TRUE(Register);

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.TargetSpecificFlags = Mc.InstrInfo->get(Opcode).TSFlags;
  Di.Inst.addOperand(MCOperand::createReg(Register));
  Expected<ParsedReg> Parsed = Env->Ctx->parseReg(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Parsed));
  std::string Format;
  std::string Detail;
  handleAllErrors(Parsed.takeError(), [&](const RaiseFailure &Failure) {
    ASSERT_TRUE(Failure.format());
    Format = Failure.format()->str();
    Detail = Failure.detail().str();
  });
  EXPECT_EQ(Format, "SMEM");
  EXPECT_NE(Detail.find("register-decode"), std::string::npos);
  EXPECT_NE(Detail.find("not in operand register class"), std::string::npos);
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
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
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

TEST_F(RaiseContextTest, ReportsEncodingFormatForOperandFailures) {
  const unsigned Opcode = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.TargetSpecificFlags = Mc.InstrInfo->get(Opcode).TSFlags;
  Di.Inst.addOperand(MCOperand::createDFPImm(0));
  Expected<Value *> Result = Env->Ctx->readOp32(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Format;
  std::string Detail;
  handleAllErrors(Result.takeError(), [&](const RaiseFailure &Failure) {
    ASSERT_TRUE(Failure.format());
    Format = Failure.format()->str();
    Detail = Failure.detail().str();
  });
  EXPECT_EQ(Format, "SOP1");
  EXPECT_NE(Detail.find("operand-read"), std::string::npos);
}

TEST_F(RaiseContextTest, RejectsXnackMaskOperands) {
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
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
  BasicBlock *Block = Gfx1250.B.GetInsertBlock();
  size_t InstructionCount = Block->size();
  for (StringRef Name : {"SGPR_NULL", "SGPR_NULL_HI"}) {
    MCRegister Reg = findRegister(*State->RegInfo, Name);
    ASSERT_TRUE(Reg) << Name.str();

    DecodedInst Di;
    Di.Inst.setOpcode(findOpcode(*State->InstrInfo, "S_MOV_B32"));
    Di.Inst.addOperand(MCOperand::createReg(Reg));
    Expected<ParsedReg> Parsed = Gfx1250.Ctx->parseReg(Di, 0);
    ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
    ASSERT_EQ(Parsed->RegKind, ParsedReg::NOREG);

    Gfx1250.Ctx->writeReg32(*Parsed, Gfx1250.B.getInt32(1));
    Gfx1250.Ctx->writeReg64(*Parsed, Gfx1250.B.getInt64(1));
    Gfx1250.Ctx->writeRegVec(*Parsed,
                             ConstantVector::getSplat(ElementCount::getFixed(2),
                                                      Gfx1250.B.getInt32(1)));
    Gfx1250.Ctx->writeRegExecWidth(*Parsed, Gfx1250.B.getInt32(1));
  }
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

TEST_F(RaiseContextTest, ProjectsMaskReadsToCurrentSourceWave) {
  Expected<MCState> SourceState = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(SourceState))
      << toString(SourceState.takeError());
  ContextEnvironment Widening(*SourceState, Mc);
  Value *Lane = Widening.Projection->emitLaneIdx(Widening.B);
  Widening.Ctx->regs().storeVCC(
      Widening.B,
      Widening.B.CreateICmpUGE(Lane, Widening.B.getInt32(32), "upper_wave"));
  Widening.Ctx->regs().storeExec(Widening.B,
                                 Widening.B.getInt64(0xFFFFFFFF00000000));

  const unsigned Opcode = findOpcode(*SourceState->InstrInfo, "S_MOV_B32");
  ASSERT_NE(Opcode, SourceState->InstrInfo->getNumOpcodes());
  const auto ReadRegister = [&](MCRegister Register) -> Expected<Value *> {
    DecodedInst Instruction;
    Instruction.Inst.setOpcode(Opcode);
    Instruction.TargetSpecificFlags =
        SourceState->InstrInfo->get(Opcode).TSFlags;
    Instruction.Inst.addOperand(MCOperand::createReg(Register));
    return Widening.Ctx->readOp32(Instruction, 0);
  };
  const auto ExpectSourceWaveSlice = [](Value *Mask) {
    auto *Trunc = dyn_cast<TruncInst>(Mask);
    ASSERT_NE(Trunc, nullptr);
    auto *Shift = dyn_cast<BinaryOperator>(Trunc->getOperand(0));
    ASSERT_NE(Shift, nullptr);
    EXPECT_EQ(Shift->getOpcode(), Instruction::LShr);
  };

  for (const StringRef RegisterName : {"VCC_LO", "EXEC_LO"}) {
    const MCRegister Register =
        findRegister(*SourceState->RegInfo, RegisterName);
    ASSERT_TRUE(Register) << RegisterName.str();
    Expected<Value *> Result = ReadRegister(Register);
    ASSERT_TRUE(static_cast<bool>(Result)) << toString(Result.takeError());
    ExpectSourceWaveSlice(*Result);
  }
  for (const StringRef RegisterName : {"SRC_VCCZ", "SRC_EXECZ"}) {
    const MCRegister Register =
        findRegister(*SourceState->RegInfo, RegisterName);
    ASSERT_TRUE(Register) << RegisterName.str();
    Expected<Value *> Result = ReadRegister(Register);
    ASSERT_TRUE(static_cast<bool>(Result)) << toString(Result.takeError());
    auto *Extend = dyn_cast<ZExtInst>(*Result);
    ASSERT_NE(Extend, nullptr);
    auto *Compare = dyn_cast<ICmpInst>(Extend->getOperand(0));
    ASSERT_NE(Compare, nullptr);
    ExpectSourceWaveSlice(Compare->getOperand(0));
  }
}

TEST_F(RaiseContextTest, RetainsPairWidthAcrossBlocks) {
  Env->Ctx->recordSgprWaveMaskI1(2, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Ctx->recordSgprWaveMaskI1(4, ConstantInt::getTrue(Env->LLVMCtx), true);
  Env->Ctx->clearSgprWaveMaskShadow();

  Env->Ctx->invalidateSgprWaveMaskI1(3);
  Env->Ctx->invalidateSgprWaveMaskI1(5);
  Value *SingleValid = Env->Ctx->loadSgprWaveMaskValid(2);
  Value *PairValid = Env->Ctx->loadSgprWaveMaskValid(4);
  AllocaInst *ObservedSingle =
      Env->B.CreateAlloca(Env->B.getInt1Ty(), nullptr, "observed_single");
  AllocaInst *ObservedPair =
      Env->B.CreateAlloca(Env->B.getInt1Ty(), nullptr, "observed_pair");
  StoreInst *StoreSingle = Env->B.CreateStore(SingleValid, ObservedSingle);
  StoreInst *StorePair = Env->B.CreateStore(PairValid, ObservedPair);
  Env->B.CreateRetVoid();

  promoteAndFold(*Env);
  auto *StoredSingle = dyn_cast<ConstantInt>(StoreSingle->getValueOperand());
  auto *StoredPair = dyn_cast<ConstantInt>(StorePair->getValueOperand());
  ASSERT_NE(StoredSingle, nullptr);
  ASSERT_NE(StoredPair, nullptr);
  EXPECT_TRUE(StoredSingle->isOne());
  EXPECT_TRUE(StoredPair->isZero());
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
  Env->Ctx->writeReg32(M0, PoisonValue::get(Env->B.getInt32Ty()));
  EXPECT_FALSE(Env->Ctx->getM0Const());

  Value *OldLaneActive = Env->Ctx->emitLaneActiveBit();
  ParsedReg Exec;
  Exec.RegKind = ParsedReg::EXEC;
  Exec.BaseIdx = 0;
  Env->Ctx->writeReg32(Exec, Env->B.getInt32(1));
  EXPECT_NE(Env->Ctx->emitLaneActiveBit(), OldLaneActive);
}

TEST_F(RaiseContextTest, UnrelatedMemoryLoadPreservesKernargPtrOffset) {
  ContextEnvironment WithKernargPtr(Mc, makeKernargPtrMeta());
  WithKernargPtr.Ctx->setKernargPtrLiveEntryByteOffset(-16);

  WithKernargPtr.Ctx->noteSgprMemoryLoadForKernargProvenance(4, 2);

  const RaiseContext::KernargPtrProvenance Provenance =
      WithKernargPtr.Ctx->getKernargPtrProvenance();
  EXPECT_TRUE(Provenance.isLiveEntry());
  EXPECT_EQ(Provenance.EntryByteOffset, -16);
}

TEST_F(RaiseContextTest, InactiveSourceWavePreservesInvalidPairShadow) {
  ContextEnvironment FullWave(Mc, {}, true);
  FullWave.Ctx->regs().storeExec(FullWave.B, FullWave.B.getInt64(0));

  FullWave.Ctx->recordSourceWaveSgprPair(0, FullWave.B.getInt64(7));
  Value *Result =
      FullWave.Ctx->materializeSourceWaveSgprPair(0, FullWave.B.getInt64(42));
  AllocaInst *Observed =
      FullWave.B.CreateAlloca(FullWave.B.getInt64Ty(), nullptr, "observed");
  StoreInst *Store = FullWave.B.CreateStore(Result, Observed);
  FullWave.B.CreateRetVoid();

  promoteAndFold(FullWave);
  auto *Stored = dyn_cast<ConstantInt>(Store->getValueOperand());
  ASSERT_NE(Stored, nullptr);
  EXPECT_EQ(Stored->getZExtValue(), 42u);
}

TEST_F(RaiseContextTest, OwnsPerSgprShadowStorage) {
  SmallVector<AllocaInst *> RegisterFileAllocas;
  Env->Ctx->regs().collectAllocas(RegisterFileAllocas);
  SmallVector<AllocaInst *> Allocas;
  Env->Ctx->collectAllocas(Allocas);
  EXPECT_EQ(Allocas.size(),
            RegisterFileAllocas.size() + Env->Ctx->regs().Sgpr.size() * 5);
}

TEST_F(RaiseContextTest, InvalidatesPerSgprShadowStorage) {
  ContextEnvironment FullWave(Mc, {}, true);
  FullWave.Ctx->regs().storeExec(FullWave.B, FullWave.B.getInt64(UINT64_MAX));
  FullWave.Ctx->recordSourceWaveSgprPair(0, FullWave.B.getInt64(7));
  Value *Before =
      FullWave.Ctx->materializeSourceWaveSgprPair(0, FullWave.B.getInt64(42));
  AllocaInst *ObservedBefore = FullWave.B.CreateAlloca(
      FullWave.B.getInt64Ty(), nullptr, "observed_before");
  StoreInst *StoreBefore = FullWave.B.CreateStore(Before, ObservedBefore);

  FullWave.Ctx->invalidateSgprShadows();
  Value *After =
      FullWave.Ctx->materializeSourceWaveSgprPair(0, FullWave.B.getInt64(42));
  AllocaInst *ObservedAfter = FullWave.B.CreateAlloca(
      FullWave.B.getInt64Ty(), nullptr, "observed_after");
  StoreInst *StoreAfter = FullWave.B.CreateStore(After, ObservedAfter);
  FullWave.B.CreateRetVoid();

  promoteAndFold(FullWave);
  auto *StoredBefore = dyn_cast<ConstantInt>(StoreBefore->getValueOperand());
  auto *StoredAfter = dyn_cast<ConstantInt>(StoreAfter->getValueOperand());
  ASSERT_NE(StoredBefore, nullptr);
  ASSERT_NE(StoredAfter, nullptr);
  EXPECT_EQ(StoredBefore->getZExtValue(), 7u);
  EXPECT_EQ(StoredAfter->getZExtValue(), 42u);
}

} // namespace
