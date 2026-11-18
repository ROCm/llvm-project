//===- AMDGPUSQTTLower.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimize llvm.amdgcn.sqtt.event uses and allocate unique event IDs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Target/AMDGPU/SQTTEvent.h"

#define DEBUG_TYPE "amdgpu-sqtt-lower"

using namespace llvm;

namespace {
static cl::opt<unsigned>
    CombineEventsThreshold("amdgpu-sqtt-combine-events-threshold",
                           cl::init(4 * TargetTransformInfo::TCC_Basic),
                           cl::Hidden,
                           cl::desc("Combine consecutive SQTT events separated "
                                    "by a latency of at most this threshold."));

static bool isEvent(const Instruction &I) {
  const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
  return II && II->getIntrinsicID() == Intrinsic::amdgcn_sqtt_event;
}

// Do not move events across these kinds of instructions.
static bool isEventFence(const Instruction &I) {
  return I.mayHaveSideEffects();
}

static MDNode *getEventMetadata(const Instruction &I) {
  assert(isEvent(I));
  const IntrinsicInst *II = cast<IntrinsicInst>(&I);
  return cast<MDNode>(
      cast<MetadataAsValue>(II->getArgOperand(0))->getMetadata());
}

static void assignEventIDs(Module &M, Function &SQTTEventFun) {
  FunctionType *FT = SQTTEventFun.getFunctionType();
  Type *IdTy = FT->getParamType(1);

  NamedMDNode *EventsMD = M.getOrInsertNamedMetadata(sqtt::EventsTableMetadata);

  for (User *U : SQTTEventFun.users()) {
    IntrinsicInst *II = cast<IntrinsicInst>(U);

    Use &EventIdOperand = II->getArgOperandUse(1);
    EventIdOperand.set(ConstantInt::get(IdTy, EventsMD->getNumOperands()));
    EventsMD->addOperand(getEventMetadata(*II));
  }
}

static void
removeRedundantSetEventTableEvents(const IntrinsicInst &First,
                                   SmallVectorImpl<sqtt::Event> &Events) {
  bool IsCULocal = cast<ConstantInt>(First.getArgOperand(2))->getZExtValue();

  // We want to preserve only the last setEventTableBase event. And only if the
  // BaseOffset actually changes at the end of the event.
  auto LastSetEventTableBase =
      find_if(Events.rbegin(), Events.rend(),
              std::mem_fn(&sqtt::Event::isSetEventTableBase));
  if (LastSetEventTableBase == Events.rend())
    return;

  auto EraseEnd = LastSetEventTableBase.base() - 1;
  auto EraseBegin = remove_if(Events.begin(), EraseEnd,
                              std::mem_fn(&sqtt::Event::isSetEventTableBase));

  // The last event sets the base offset to the same value as the incoming
  // offset, we can drop it too.
  uint32_t IncomingIndex = IsCULocal ? 0 : sqtt::SectionBase;
  if (LastSetEventTableBase->getIndex() == IncomingIndex)
    ++EraseEnd;

  Events.erase(EraseBegin, EraseEnd);
}

static void combineEvents(Module &M, const TargetMachine &TM,
                          Function &SQTTEventFun) {
  LLVMContext &C = M.getContext();
  MapVector<BasicBlock *, unsigned> IntrinsicBlocks;
  for (User *U : SQTTEventFun.users()) {
    IntrinsicInst *II = cast<IntrinsicInst>(U);
    ++IntrinsicBlocks[II->getParent()];
  }

  for (auto &[BB, IntrinsicCount] : IntrinsicBlocks) {
    if (IntrinsicCount < 2)
      continue;

    const TargetTransformInfo &TTI =
        TM.getTargetTransformInfo(*BB->getParent());

    Instruction *First = &*find_if(*BB, isEvent);
    SmallVector<Instruction *> ToMerge = {First};

    auto DoMerge = [&]() {
      if (ToMerge.size() < 2)
        return;

      IntrinsicInst &First = cast<IntrinsicInst>(*ToMerge.front());

      SmallVector<sqtt::Event> Events;
      Events.reserve(ToMerge.size());
      for (Instruction *I : ToMerge) {
        MDNode *MD = getEventMetadata(*I);
        Events.emplace_back(sqtt::Event::fromMetadata(MD));
      }

      removeRedundantSetEventTableEvents(First, Events);

      if (Events.empty()) {
        for_each(ToMerge, std::mem_fn(&Instruction::eraseFromParent));
        return;
      }

      sqtt::EventRecord Record(Events);
      First.setArgOperand(0, MetadataAsValue::get(C, Record.toMetadata(C)));
      for_each(drop_begin(ToMerge), std::mem_fn(&Instruction::eraseFromParent));
    };

    unsigned IntrinsicsAfterCur = IntrinsicCount - 1;
    while (IntrinsicsAfterCur > 0) {
      Instruction *LastSeen = ToMerge.back();
      Instruction *Next = &*std::find_if(LastSeen->getNextNode()->getIterator(),
                                         BB->end(), isEvent);

      InstructionCost Latency = 0;
      for (auto It = LastSeen->getNextNode()->getIterator(),
                End = Next->getIterator();
           Latency < CombineEventsThreshold && It != End; ++It) {
        const Instruction &I = *It;
        if (isEventFence(I))
          Latency = CombineEventsThreshold.getValue();
        else
          Latency +=
              TTI.getInstructionCost(&*It, TargetTransformInfo::TCK_Latency);
      }

      if (Latency < CombineEventsThreshold) {
        ToMerge.push_back(Next);
      } else {
        DoMerge();
        ToMerge = {Next};
      }
      --IntrinsicsAfterCur;
    }
    DoMerge();
  }
}

static bool canBeCalledFromOutsideTheCU(const Function &F) {
  return !F.hasLocalLinkage() ||
         F.hasAddressTaken(nullptr, /*IgnoreCallbackUses=*/true,
                           /*IgnoreAssumeLikeCalls=*/true,
                           /*IgnoreLLVMUsed=*/true);
}

static CallInst *insertSetEventTableBase(Instruction &InsertBefore,
                                         uint32_t Index, bool IsCULocal) {
  IRBuilder<> Builder(&InsertBefore);
  LLVMContext &C = Builder.getContext();
  Metadata *EventMD = sqtt::Event::setEventTableBase(Index).toMetadata(C);
  Value *Args[] = {MetadataAsValue::get(C, EventMD),
                   PoisonValue::get(Builder.getInt32Ty()),
                   Builder.getInt1(IsCULocal)};
  return Builder.CreateIntrinsicWithoutFolding(Intrinsic::amdgcn_sqtt_event,
                                               Args);
}

static CallInst *setEventTableToCUBase(Instruction &InsertBefore) {
  return insertSetEventTableBase(InsertBefore, /*Index=*/0,
                                 /*IsCULocal=*/false);
}

static CallInst *setEventTableToSectionBase(Instruction &InsertBefore) {
  return insertSetEventTableBase(InsertBefore, sqtt::SectionBase,
                                 /*IsCULocal=*/true);
}

static void insertCUBoundaryEventForFunction(Function &F) {
  BasicBlock &EntryBB = F.getEntryBlock();
  BasicBlock::iterator InsertBeforeEntry = EntryBB.getFirstNonPHIOrAlloca();
  setEventTableToCUBase(*InsertBeforeEntry);

  bool ReturnsToHost = AMDGPU::isKernel(F);
  if (ReturnsToHost)
    return;

  for (BasicBlock &BB : F) {
    ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!RI)
      continue;
    setEventTableToSectionBase(*RI);
  }
}

static void insertCUBoundaryEventForCallSite(CallInst &CI) {
  setEventTableToSectionBase(CI);
  setEventTableToCUBase(*CI.getNextNode());
}

static void insertCUBoundaryEvents(Module &M) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (!canBeCalledFromOutsideTheCU(F))
      continue;
    insertCUBoundaryEventForFunction(F);
  }

  for (Function &F : M) {
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      CallInst *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *CalledFunction = CI->getCalledFunction();
      if (CalledFunction && (CalledFunction->isIntrinsic() ||
                             !canBeCalledFromOutsideTheCU(*CalledFunction)))
        continue;
      insertCUBoundaryEventForCallSite(*CI);
    }
  }
}

static bool run(Module &M, const TargetMachine &TM) {
  Function *Intrinsic =
      M.getFunction(Intrinsic::getName(Intrinsic::amdgcn_sqtt_event));

  if (!Intrinsic || Intrinsic->user_empty())
    return false;

  if (AMDGPUTargetMachine::EnableObjectLinking)
    insertCUBoundaryEvents(M);

  if (CombineEventsThreshold > 0)
    combineEvents(M, TM, *Intrinsic);
  assignEventIDs(M, *Intrinsic);
  return true;
}

class AMDGPUSQTTLowerLegacy : public ModulePass {
public:
  static char ID;

  AMDGPUSQTTLowerLegacy() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  bool runOnModule(Module &M) override;
};

} // namespace

bool AMDGPUSQTTLowerLegacy::runOnModule(Module &M) {
  auto &TPC = getAnalysis<TargetPassConfig>();
  return ::run(M, TPC.getTM<AMDGPUTargetMachine>());
}

PreservedAnalyses AMDGPUSQTTLowerPass::run(Module &M, ModuleAnalysisManager &) {
  if (!::run(M, TM))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

INITIALIZE_PASS_BEGIN(AMDGPUSQTTLowerLegacy, DEBUG_TYPE, "AMDGPU SQTT lower",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUSQTTLowerLegacy, DEBUG_TYPE, "AMDGPU SQTT lower",
                    false, false)

char AMDGPUSQTTLowerLegacy::ID = 0;

ModulePass *llvm::createAMDGPUSQTTLowerLegacyPass() {
  return new AMDGPUSQTTLowerLegacy;
}
