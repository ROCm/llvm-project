//===- RegFileTest.cpp - alloca register-file unit tests ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for AllocaRegFile. The reg file exists to be written and read
// with straight-line stores/loads and then lifted to SSA by PromoteMemToReg,
// so the tests write a constant through the generic ParsedReg dispatch, read it
// back, promote, constant-fold, and check the value survives the round trip. A
// structural test pins that every emitted alloca is promotable and the module
// verifies.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/reg-file.h"

#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "gtest/gtest.h"

#include <mutex>

// AllocaRegFile links the hotswap::decoder MC stack, whose initMCState calls
// COMGR::ensureLLVMInitialized; its production definition lives in
// libamd_comgr. Provide the registration here so the test binary stays minimal.
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
using COMGR::hotswap::AllocaRegFile;
using COMGR::hotswap::initMCState;
using COMGR::hotswap::ISAProfile;
using COMGR::hotswap::MCState;
using COMGR::hotswap::ParsedReg;
using COMGR::hotswap::ReplicationProjection;

namespace {

// gfx942 is a wave64 CDNA ISA (it has AGPRs), so a same-ISA replication
// projection gives a self-consistent register file that exercises the AGPR
// bank and the i64 EXEC storage width.
class RegFileTest : public ::testing::Test {
protected:
  void SetUp() override {
    Expected<MCState> S = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(S)) << toString(S.takeError());
    Mc = std::move(*S);
  }

  ISAProfile srcIsa() const {
    return ISAProfile::fromSubtarget(*Mc.SubtargetInfo);
  }

  MCState Mc;
};

// Owns a module and an initialised register file. `begin` starts a function
// returning RetTy and initialises the reg file in its entry block, leaving the
// builder positioned to append register traffic.
struct RegFileFixture {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  ISAProfile Isa;
  ReplicationProjection Proj;
  const MCRegisterInfo &MRI;
  IRBuilder<> B;
  AllocaRegFile RF;
  Function *F = nullptr;

  RegFileFixture(const ISAProfile &SrcIsa, const MCRegisterInfo &Mri)
      : M(std::make_unique<Module>("regfile_test", Ctx)), Isa(SrcIsa),
        Proj(Isa, Isa, Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)), MRI(Mri),
        B(Ctx) {}

  void begin(Type *RetTy) {
    FunctionType *FT = FunctionType::get(RetTy, /*isVarArg=*/false);
    F = Function::Create(FT, Function::ExternalLinkage, "f", M.get());
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    B.SetInsertPoint(BB);
    RF.init(B, Type::getInt32Ty(Ctx), Type::getInt1Ty(Ctx), Isa, MRI, Proj);
  }
};

// Promote the register file's allocas to SSA and constant-fold the result, then
// return the value the function returns. With a constant written in, the read
// chain folds back to that constant.
Value *promoteAndFold(RegFileFixture &Fx) {
  SmallVector<AllocaInst *> Allocas;
  Fx.RF.collectAllocas(Allocas);
  DominatorTree DT(*Fx.F);
  PromoteMemToReg(Allocas, DT);

  const DataLayout &DL = Fx.M->getDataLayout();
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (BasicBlock &BB : *Fx.F)
      for (Instruction &I : make_early_inc_range(BB)) {
        if (I.isTerminator())
          continue;
        if (Constant *C = ConstantFoldInstruction(&I, DL)) {
          I.replaceAllUsesWith(C);
          I.eraseFromParent();
          Changed = true;
        }
      }
  }
  return cast<ReturnInst>(Fx.F->back().getTerminator())->getReturnValue();
}

ParsedReg reg(ParsedReg::Kind Kind, unsigned Idx, uint8_t Width) {
  ParsedReg R;
  R.RegKind = Kind;
  R.BaseIdx = Idx;
  R.WidthInDwords = Width;
  return R;
}

TEST_F(RegFileTest, SGPR32RoundTrip) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  Fx.begin(Type::getInt32Ty(Fx.Ctx));
  ParsedReg R = reg(ParsedReg::SGPR, 5, 1);
  Fx.RF.writeReg32(Fx.B, R, ConstantInt::get(Type::getInt32Ty(Fx.Ctx), 0x1234));
  Fx.B.CreateRet(Fx.RF.readReg32(Fx.B, R));

  auto *CI = dyn_cast_or_null<ConstantInt>(promoteAndFold(Fx));
  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getZExtValue(), 0x1234u);
}

TEST_F(RegFileTest, VGPR32RoundTrip) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  Fx.begin(Type::getInt32Ty(Fx.Ctx));
  ParsedReg R = reg(ParsedReg::VGPR, 7, 1);
  Fx.RF.writeReg32(Fx.B, R, ConstantInt::get(Type::getInt32Ty(Fx.Ctx), 0xABCD));
  Fx.B.CreateRet(Fx.RF.readReg32(Fx.B, R));

  auto *CI = dyn_cast_or_null<ConstantInt>(promoteAndFold(Fx));
  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getZExtValue(), 0xABCDu);
}

TEST_F(RegFileTest, M0RoundTrip) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  Fx.begin(Type::getInt32Ty(Fx.Ctx));
  ParsedReg R = reg(ParsedReg::M0, 0, 1);
  Fx.RF.writeReg32(Fx.B, R, ConstantInt::get(Type::getInt32Ty(Fx.Ctx), 0x55));
  Fx.B.CreateRet(Fx.RF.readReg32(Fx.B, R));

  auto *CI = dyn_cast_or_null<ConstantInt>(promoteAndFold(Fx));
  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getZExtValue(), 0x55u);
}

TEST_F(RegFileTest, SGPR64RoundTripSplitsAndRecombines) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  Fx.begin(Type::getInt64Ty(Fx.Ctx));
  ParsedReg R = reg(ParsedReg::SGPR, 4, 2);
  Constant *C =
      ConstantInt::get(Type::getInt64Ty(Fx.Ctx), 0x1122334455667788ULL);
  Fx.RF.writeReg64(Fx.B, R, C);
  Fx.B.CreateRet(Fx.RF.readReg64(Fx.B, R));

  auto *CI = dyn_cast_or_null<ConstantInt>(promoteAndFold(Fx));
  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getZExtValue(), 0x1122334455667788ULL);
}

TEST_F(RegFileTest, SccI1RoundTrip) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  Fx.begin(Type::getInt1Ty(Fx.Ctx));
  Fx.RF.storeSCC(Fx.B, ConstantInt::getTrue(Fx.Ctx));
  Fx.B.CreateRet(Fx.RF.loadSCC(Fx.B));

  auto *CI = dyn_cast_or_null<ConstantInt>(promoteAndFold(Fx));
  ASSERT_NE(CI, nullptr);
  EXPECT_TRUE(CI->isOne());
}

TEST_F(RegFileTest, VectorRoundTripAcrossContiguousVgprs) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  auto *VecTy = FixedVectorType::get(Type::getInt32Ty(Fx.Ctx), 4);
  Fx.begin(VecTy);
  ParsedReg R = reg(ParsedReg::VGPR, 8, 4);
  Constant *Vec =
      ConstantDataVector::get(Fx.Ctx, ArrayRef<uint32_t>{1, 2, 3, 4});
  Fx.RF.writeRegVec(Fx.B, R, Vec);
  Fx.B.CreateRet(Fx.RF.readRegVec(Fx.B, R, VecTy));

  // Constants are uniqued, so a correct round trip folds back to the same
  // constant object.
  EXPECT_EQ(promoteAndFold(Fx), Vec);
}

TEST_F(RegFileTest, InitProducesPromotableAllocasAndVerifies) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  Fx.begin(Type::getVoidTy(Fx.Ctx));
  Fx.RF.writeReg32(Fx.B, reg(ParsedReg::SGPR, 0, 1),
                   ConstantInt::get(Type::getInt32Ty(Fx.Ctx), 7));
  Fx.RF.writeReg64(Fx.B, reg(ParsedReg::VGPR, 2, 2),
                   ConstantInt::get(Type::getInt64Ty(Fx.Ctx), 9));
  Fx.B.CreateRetVoid();

  SmallVector<AllocaInst *> Allocas;
  Fx.RF.collectAllocas(Allocas);
  EXPECT_FALSE(Allocas.empty());

  DominatorTree DT(*Fx.F);
  PromoteMemToReg(Allocas, DT);

  unsigned RemainingAllocas = 0;
  for (Instruction &I : Fx.F->getEntryBlock())
    if (isa<AllocaInst>(I))
      ++RemainingAllocas;
  EXPECT_EQ(RemainingAllocas, 0u);

  std::string Err;
  raw_string_ostream OS(Err);
  EXPECT_FALSE(verifyModule(*Fx.M, &OS)) << Err;
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
TEST_F(RegFileTest, AbsentIndexOnIndexedKindAborts) {
  RegFileFixture Fx(srcIsa(), *Mc.RegInfo);
  Fx.begin(Type::getInt32Ty(Fx.Ctx));
  ParsedReg R;
  R.RegKind = ParsedReg::SGPR;
  R.WidthInDwords = 1; // BaseIdx left absent.
  EXPECT_DEATH((void)Fx.RF.readReg32(Fx.B, R), "base register index");
}
#endif

} // namespace
