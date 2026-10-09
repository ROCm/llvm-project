//===- VectorCompareTest.cpp - Vector comparison tests --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/decoder/mc-state.h"
#include "transpiler/decoder/setpc-analysis.h"
#include "transpiler/raiser/handlers.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/wave-projection.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace COMGR::transpiler;

namespace {

// Evaluate the lane-zero result without executing target ballot intrinsics.
class LaneZeroProjection : public ReplicationProjection {
public:
  using ReplicationProjection::ReplicationProjection;
  Value *emitLaneActiveBit(IRBuilder<> &B, Value *Exec) const override {
    return B.CreateTrunc(Exec, B.getInt1Ty());
  }
};

class VectorCompareTest : public ::testing::Test {
protected:
  MCState MC;
  std::unique_ptr<TargetMachine> Machine;

  void SetUp() override {
    Expected<MCState> State = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
    MC = std::move(*State);
    Machine.reset(MC.Target->createTargetMachine(Triple("amdgcn-amd-amdhsa"),
                                                 "gfx942", "", TargetOptions(),
                                                 std::nullopt));
    ASSERT_NE(Machine, nullptr);
  }

  void check(CanonicalOp Opcode, uint64_t Left, uint64_t Right, bool Expected,
             bool Active = true, bool PoisonSource = false) {
    SCOPED_TRACE(canonicalOpName(Opcode).str());
    LLVMContext Context;
    Module TestModule("comparison", Context);
    TestModule.setTargetTriple(Triple("amdgcn-amd-amdhsa"));
    TestModule.setDataLayout(Machine->createDataLayout());
    IRBuilder<> B(Context);
    LaneZeroProjection Projection(*MC.SubtargetInfo, *MC.SubtargetInfo,
                                  B.getInt32Ty(), B.getInt64Ty());
    Function *TestFunction =
        Function::Create(FunctionType::get(B.getInt1Ty(), false),
                         Function::ExternalLinkage, "comparison", TestModule);
    B.SetInsertPoint(BasicBlock::Create(Context, "entry", TestFunction));
    SetPcAnalysis SetPc;
    DenseSet<uint64_t> PairedSplitBarriers;
    RaiseContext Raise = cantFail(
        RaiseContext::create(B, Projection, MC, SetPc, PairedSplitBarriers,
                             KernelMeta(), {}, 0, {}, 0, 0));
    Raise.registers().storeExec(B.getInt64(Active));

    DecodedInst Instruction;
    Instruction.CanonOp = Opcode;
    std::string Name = (canonicalOpName(Opcode) + "_e32_vi").str();
    unsigned MCOpcode = 0;
    for (; MCOpcode != MC.InstrInfo->getNumOpcodes(); ++MCOpcode)
      if (MC.InstrInfo->getName(MCOpcode) == Name)
        break;
    ASSERT_NE(MCOpcode, MC.InstrInfo->getNumOpcodes()) << Name;
    Instruction.Inst.setOpcode(MCOpcode);
    Instruction.setDefsVcc(true);
    Instruction.SrcMap = {0, 1};
    Instruction.ModMap = {UINT_MAX, UINT_MAX};
    Instruction.Inst.addOperand(MCOperand::createImm(Left));
    Instruction.Inst.addOperand(MCOperand::createImm(Right));
    if (PoisonSource) {
      MCRegister Source;
      for (unsigned I = 1; I != MC.RegInfo->getNumRegs(); ++I)
        if (StringRef(MC.RegInfo->getName(I)) == "VGPR0")
          Source = MCRegister(I);
      ASSERT_TRUE(Source);
      Instruction.Inst.getOperand(0) = MCOperand::createReg(Source);
      Raise.registers().regFile().writeReg32(B,
                                             ParsedReg{ParsedReg::VGPR, 0, 1},
                                             PoisonValue::get(B.getInt32Ty()));
    }
    OperandResolver Resolver{Raise, Instruction};
    Error Result = handleVOPC(Raise, Instruction, Resolver);
    ASSERT_FALSE(static_cast<bool>(Result)) << toString(std::move(Result));
    B.CreateRet(Raise.registers().regFile().loadVCC(B));
    ASSERT_FALSE(verifyModule(TestModule, &errs()));

    SmallVector<AllocaInst *> Allocas;
    Raise.registers().collectAllocas(Allocas);
    DominatorTree Dominators(*TestFunction);
    PromoteMemToReg(Allocas, Dominators);
    bool Changed;
    do {
      Changed = false;
      for (BasicBlock &Block : *TestFunction)
        for (llvm::Instruction &Value : make_early_inc_range(Block)) {
          if (Value.isTerminator())
            continue;
          if (llvm::Value *Folded = simplifyInstruction(
                  &Value, SimplifyQuery(TestModule.getDataLayout()))) {
            Value.replaceAllUsesWith(Folded);
            Value.eraseFromParent();
            Changed = true;
          }
        }
    } while (Changed);
    if (!getVectorCompareInfo(Opcode)->Predicate) {
      LoopAnalysisManager Loops;
      FunctionAnalysisManager Functions;
      CGSCCAnalysisManager CallGraph;
      ModuleAnalysisManager Modules;
      PassBuilder Passes(Machine.get());
      Passes.registerModuleAnalyses(Modules);
      Passes.registerCGSCCAnalyses(CallGraph);
      Passes.registerFunctionAnalyses(Functions);
      Passes.registerLoopAnalyses(Loops);
      Passes.crossRegisterProxies(Loops, Functions, CallGraph, Modules);
      FunctionPassManager Pipeline;
      Pipeline.addPass(InstCombinePass());
      Pipeline.run(*TestFunction, Functions);
    }
    auto *Returned = cast<ReturnInst>(TestFunction->back().getTerminator());
    auto *Constant = dyn_cast<ConstantInt>(Returned->getReturnValue());
    ASSERT_NE(Constant, nullptr);
    EXPECT_EQ(Constant->isOne(), Expected);
  }
};

TEST_F(VectorCompareTest, IntegerWidthsAndSignedness) {
  check(CanonicalOp::V_CMP_EQ_U16, 0x1234ffff, 0xabcdffff, true);
  check(CanonicalOp::V_CMP_NE_U16, 0x1234ffff, 0xabcdffff, false);
  check(CanonicalOp::V_CMP_LT_I16, 0x12348000, 0x56780001, true);
  check(CanonicalOp::V_CMP_LT_U16, 0x12348000, 0x56780001, false);
  check(CanonicalOp::V_CMP_LT_I64, 0x8000000000000000, 1, true);
  check(CanonicalOp::V_CMP_LT_U64, 0x8000000000000000, 1, false);
  check(CanonicalOp::V_CMP_EQ_U64, 0x100000001, 1, false);
  check(CanonicalOp::V_CMP_NE_U64, 0x100000001, 1, true);
  check(CanonicalOp::V_CMP_GT_I64, 0x100000000, 1, true);
  check(CanonicalOp::V_CMP_GE_I64, 0x100000000, 0x100000000, true);
  check(CanonicalOp::V_CMP_LE_I64, 0x100000000, 1, false);
  check(CanonicalOp::V_CMP_GT_U64, 0xffffffffffffffff, 1, true);
  check(CanonicalOp::V_CMP_GE_U64, 0x8000000000000000, 1, true);
  check(CanonicalOp::V_CMP_LE_U64, 0x100000000, 1, false);
  check(CanonicalOp::V_CMP_LT_I64, 0x80000000, 0, true);
  check(CanonicalOp::V_CMP_EQ_I64, 0xffffffff, UINT64_MAX, true);
  check(CanonicalOp::V_CMP_EQ_U64, 0xffffffff, UINT64_MAX, false);
  check(CanonicalOp::V_CMP_LT_I32, 0x80000000, 1, true);
  check(CanonicalOp::V_CMP_LT_U32, 0x80000000, 1, false);
}

TEST_F(VectorCompareTest, FloatPredicates) {
  struct Case {
    CanonicalOp Float32;
    CanonicalOp Float64;
    bool Less;
    bool Equal;
    bool Greater;
    bool Unordered;
  };
  const Case Cases[] = {
      {CanonicalOp::V_CMP_EQ_F32, CanonicalOp::V_CMP_EQ_F64, false, true, false,
       false},
      {CanonicalOp::V_CMP_GE_F32, CanonicalOp::V_CMP_GE_F64, false, true, true,
       false},
      {CanonicalOp::V_CMP_GT_F32, CanonicalOp::V_CMP_GT_F64, false, false, true,
       false},
      {CanonicalOp::V_CMP_LE_F32, CanonicalOp::V_CMP_LE_F64, true, true, false,
       false},
      {CanonicalOp::V_CMP_LG_F32, CanonicalOp::V_CMP_LG_F64, true, false, true,
       false},
      {CanonicalOp::V_CMP_LT_F32, CanonicalOp::V_CMP_LT_F64, true, false, false,
       false},
      {CanonicalOp::V_CMP_NEQ_F32, CanonicalOp::V_CMP_NEQ_F64, true, false,
       true, true},
      {CanonicalOp::V_CMP_NGE_F32, CanonicalOp::V_CMP_NGE_F64, true, false,
       false, true},
      {CanonicalOp::V_CMP_NGT_F32, CanonicalOp::V_CMP_NGT_F64, true, true,
       false, true},
      {CanonicalOp::V_CMP_NLE_F32, CanonicalOp::V_CMP_NLE_F64, false, false,
       true, true},
      {CanonicalOp::V_CMP_NLG_F32, CanonicalOp::V_CMP_NLG_F64, false, true,
       false, true},
      {CanonicalOp::V_CMP_NLT_F32, CanonicalOp::V_CMP_NLT_F64, false, true,
       true, true},
      {CanonicalOp::V_CMP_O_F32, CanonicalOp::V_CMP_O_F64, true, true, true,
       false},
      {CanonicalOp::V_CMP_U_F32, CanonicalOp::V_CMP_U_F64, false, false, false,
       true},
  };
  for (const Case &Test : Cases) {
    for (bool Is64 : {false, true}) {
      CanonicalOp Opcode = Is64 ? Test.Float64 : Test.Float32;
      uint64_t One = Is64 ? 0x3ff0000000000000 : 0x3f800000;
      uint64_t Two = Is64 ? 0x4000000000000000 : 0x40000000;
      uint64_t NegativeZero = Is64 ? 0x8000000000000000 : 0x80000000;
      uint64_t QuietNaN = Is64 ? 0x7ff8000000000001 : 0x7fc00001;
      check(Opcode, One, Two, Test.Less);
      check(Opcode, One, One, Test.Equal);
      check(Opcode, Two, One, Test.Greater);
      check(Opcode, 0, NegativeZero, Test.Equal);
      check(Opcode, QuietNaN, One, Test.Unordered);
      check(Opcode, One, QuietNaN, Test.Unordered);
      check(Opcode, QuietNaN, QuietNaN, Test.Unordered);
    }
  }
}

TEST_F(VectorCompareTest, FloatClasses) {
  const uint64_t Float32[] = {0x7f800001, 0x7fc00001, 0xff800000, 0xbf800000,
                              0x80000001, 0x80000000, 0,          1,
                              0x3f800000, 0x7f800000};
  const uint64_t Float64[] = {0x7ff0000000000001,
                              0x7ff8000000000001,
                              0xfff0000000000000,
                              0xbff0000000000000,
                              0x8000000000000001,
                              0x8000000000000000,
                              0,
                              1,
                              0x3ff0000000000000,
                              0x7ff0000000000000};
  for (unsigned Bit = 0; Bit != 10; ++Bit) {
    for (uint32_t Mask : {1u << Bit, 1023u ^ (1u << Bit), 3u, 0x204u}) {
      check(CanonicalOp::V_CMP_CLASS_F32, Float32[Bit], Mask,
            (Mask & (1u << Bit)) != 0);
      check(CanonicalOp::V_CMP_CLASS_F64, Float64[Bit], Mask,
            (Mask & (1u << Bit)) != 0);
    }
  }
}

TEST_F(VectorCompareTest, InactivePoisonDoesNotReachMask) {
  for (CanonicalOp Opcode :
       {CanonicalOp::V_CMP_EQ_U16, CanonicalOp::V_CMP_EQ_U32,
        CanonicalOp::V_CMP_EQ_F32, CanonicalOp::V_CMP_CLASS_F32})
    check(Opcode, 0, 0, false, false, true);
}

} // namespace
