//===- AMDGPUSQTTFunctionInstrument.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert the SQTT function entry and exit events in the functions marked with
// the "amdgpu-sqtt-instrument" attribute. The pass runs at the start of the
// pipeline, before the inliner.
//
// The events describe the execution of a wave, but they are inserted on the
// thread CFG. If a function has divergent returns, the exit event of each of
// them would execute for the same wave with a different exec mask. To avoid
// it, the divergent returns of an instrumented function are first merged into a
// single return block, which holds their exit event. The returns that are
// reached through uniform branches are executed by the whole wave, so they
// keep their own exit event.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUUnifyDivergentExitNodes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Target/AMDGPU/SQTTEvent.h"

using namespace llvm;

namespace {

constexpr StringLiteral InstrumentAttr = "amdgpu-sqtt-instrument";

static void insertEvent(Function &F, sqtt::Event Event, Instruction *InsertPt,
                        const DebugLoc &DL) {
  LLVMContext &C = F.getContext();
  IRBuilder<> Builder(InsertPt);
  Builder.SetCurrentDebugLocation(DL);
  Builder.CreateIntrinsic(Intrinsic::amdgcn_sqtt_event,
                          {MetadataAsValue::get(C, Event.toMetadata(C)),
                           PoisonValue::get(Builder.getInt32Ty()),
                           /*IsCULocal=*/Builder.getTrue()});
}

static DebugLoc getEntryDebugLoc(const Function &F) {
  if (DISubprogram *SP = F.getSubprogram())
    return DILocation::get(SP->getContext(), SP->getScopeLine(), 0, SP);
  return DebugLoc();
}

static DebugLoc getExitDebugLoc(const Function &F, const Instruction &Ret) {
  if (DebugLoc DL = Ret.getDebugLoc())
    return DL;
  if (DISubprogram *SP = F.getSubprogram())
    return DILocation::get(SP->getContext(), 0, 0, SP);
  return DebugLoc();
}

// The instruction that the exit event is inserted before: the return or, if
// there is one, the musttail call that precedes it.
static Instruction *getExitInsertPt(BasicBlock &BB) {
  if (CallInst *CI = BB.getTerminatingMustTailCall())
    return CI;
  return BB.getTerminator();
}

static bool run(Function &F, const TargetTransformInfo &TTI,
                const UniformityInfo &UA) {
  // Same as the generic entry/exit instrumentation: the asm of a naked function
  // can not be changed, and available_externally functions may not have a
  // definition outside of the module.
  if (F.isDeclaration() || F.hasFnAttribute(Attribute::Naked) ||
      F.hasAvailableExternallyLinkage())
    return false;

  if (!F.hasFnAttribute(InstrumentAttr))
    return false;

  // Consume the attribute so that the function is not instrumented twice.
  F.removeFnAttr(InstrumentAttr);

  // A musttail call must stay next to its return, and a return that is
  // uniformly reached is executed by the whole wave. Such returns are not
  // merged.
  SmallVector<BasicBlock *> DivergentReturns;
  SmallVector<BasicBlock *> OtherReturns;
  for (BasicBlock &BB : F) {
    if (!isa<ReturnInst>(BB.getTerminator()))
      continue;
    if (BB.getTerminatingMustTailCall() || isUniformlyReached(UA, BB))
      OtherReturns.push_back(&BB);
    else
      DivergentReturns.push_back(&BB);
  }

  DebugLoc EntryDL = getEntryDebugLoc(F);
  insertEvent(F, sqtt::Event::functionEntry(F),
              &*F.getEntryBlock().getFirstInsertionPt(), EntryDL);

  for (BasicBlock *BB : OtherReturns) {
    Instruction *InsertPt = getExitInsertPt(*BB);
    insertEvent(F, sqtt::Event::functionExit(F), InsertPt,
                getExitDebugLoc(F, *BB->getTerminator()));
  }

  if (DivergentReturns.empty())
    return true;

  DebugLoc ExitDL =
      getExitDebugLoc(F, *DivergentReturns.front()->getTerminator());
  BasicBlock *RetBlock = DivergentReturns.front();
  if (DivergentReturns.size() > 1)
    RetBlock = unifyReturnBlockSet(F, /*DTU=*/nullptr, DivergentReturns,
                                   "UnifiedReturnBlock", TTI);
  insertEvent(F, sqtt::Event::functionExit(F), RetBlock->getTerminator(),
              ExitDL);
  return true;
}

} // namespace

PreservedAnalyses
AMDGPUSQTTFunctionInstrumentPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  if (!::run(F, AM.getResult<TargetIRAnalysis>(F),
             AM.getResult<UniformityInfoAnalysis>(F)))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<TargetIRAnalysis>();
  return PA;
}
