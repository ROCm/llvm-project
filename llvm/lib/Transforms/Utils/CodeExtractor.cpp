//===- CodeExtractor.cpp - Pull code region into a new function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface to tear out a code region, such as an
// individual loop or a parallel section, into a new function, replacing it with
// a call to the new function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::PatternMatch;
using ProfileCount = Function::ProfileCount;

#define DEBUG_TYPE "code-extractor"

// Provide a command-line option to aggregate function arguments into a struct
// for functions produced by the code extractor. This is useful when converting
// extracted functions to pthread-based code, as only one argument (void*) can
// be passed in to pthread_create().
static cl::opt<bool>
AggregateArgsOpt("aggregate-extracted-args", cl::Hidden,
                 cl::desc("Aggregate arguments to code-extracted functions"));

/// Test whether a block is valid for extraction.
static bool isBlockValidForExtraction(const BasicBlock &BB,
                                      const SetVector<BasicBlock *> &Result,
                                      bool AllowVarArgs, bool AllowAlloca) {
  // taking the address of a basic block moved to another function is illegal
  if (BB.hasAddressTaken())
    return false;

  // don't hoist code that uses another basicblock address, as it's likely to
  // lead to unexpected behavior, like cross-function jumps
  SmallPtrSet<User const *, 16> Visited;
  SmallVector<User const *, 16> ToVisit(llvm::make_pointer_range(BB));

  while (!ToVisit.empty()) {
    User const *Curr = ToVisit.pop_back_val();
    if (!Visited.insert(Curr).second)
      continue;
    if (isa<BlockAddress const>(Curr))
      return false; // even a reference to self is likely to be not compatible

    if (isa<Instruction>(Curr) && cast<Instruction>(Curr)->getParent() != &BB)
      continue;

    for (auto const &U : Curr->operands()) {
      if (auto *UU = dyn_cast<User>(U))
        ToVisit.push_back(UU);
    }
  }

  // If explicitly requested, allow vastart and alloca. For invoke instructions
  // verify that extraction is valid.
  for (BasicBlock::const_iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
    if (isa<AllocaInst>(I)) {
       if (!AllowAlloca)
         return false;
       continue;
    }

    if (const auto *II = dyn_cast<InvokeInst>(I)) {
      // Unwind destination (either a landingpad, catchswitch, or cleanuppad)
      // must be a part of the subgraph which is being extracted.
      if (auto *UBB = II->getUnwindDest())
        if (!Result.count(UBB))
          return false;
      continue;
    }

    // All catch handlers of a catchswitch instruction as well as the unwind
    // destination must be in the subgraph.
    if (const auto *CSI = dyn_cast<CatchSwitchInst>(I)) {
      if (auto *UBB = CSI->getUnwindDest())
        if (!Result.count(UBB))
          return false;
      for (const auto *HBB : CSI->handlers())
        if (!Result.count(const_cast<BasicBlock*>(HBB)))
          return false;
      continue;
    }

    // Make sure that entire catch handler is within subgraph. It is sufficient
    // to check that catch return's block is in the list.
    if (const auto *CPI = dyn_cast<CatchPadInst>(I)) {
      for (const auto *U : CPI->users())
        if (const auto *CRI = dyn_cast<CatchReturnInst>(U))
          if (!Result.count(const_cast<BasicBlock*>(CRI->getParent())))
            return false;
      continue;
    }

    // And do similar checks for cleanup handler - the entire handler must be
    // in subgraph which is going to be extracted. For cleanup return should
    // additionally check that the unwind destination is also in the subgraph.
    if (const auto *CPI = dyn_cast<CleanupPadInst>(I)) {
      for (const auto *U : CPI->users())
        if (const auto *CRI = dyn_cast<CleanupReturnInst>(U))
          if (!Result.count(const_cast<BasicBlock*>(CRI->getParent())))
            return false;
      continue;
    }
    if (const auto *CRI = dyn_cast<CleanupReturnInst>(I)) {
      if (auto *UBB = CRI->getUnwindDest())
        if (!Result.count(UBB))
          return false;
      continue;
    }

    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      // musttail calls have several restrictions, generally enforcing matching
      // calling conventions between the caller parent and musttail callee.
      // We can't usually honor them, because the extracted function has a
      // different signature altogether, taking inputs/outputs and returning
      // a control-flow identifier rather than the actual return value.
      if (CI->isMustTailCall())
        return false;

      if (const Function *F = CI->getCalledFunction()) {
        auto IID = F->getIntrinsicID();
        if (IID == Intrinsic::vastart) {
          if (AllowVarArgs)
            continue;
          else
            return false;
        }

        // Currently, we miscompile outlined copies of eh_typid_for. There are
        // proposals for fixing this in llvm.org/PR39545.
        if (IID == Intrinsic::eh_typeid_for)
          return false;
      }
    }
  }

  return true;
}

/// Build a set of blocks to extract if the input blocks are viable.
static SetVector<BasicBlock *>
buildExtractionBlockSet(ArrayRef<BasicBlock *> BBs, DominatorTree *DT,
                        bool AllowVarArgs, bool AllowAlloca) {
  assert(!BBs.empty() && "The set of blocks to extract must be non-empty");
  SetVector<BasicBlock *> Result;

  // Loop over the blocks, adding them to our set-vector, and aborting with an
  // empty set if we encounter invalid blocks.
  for (BasicBlock *BB : BBs) {
    // If this block is dead, don't process it.
    if (DT && !DT->isReachableFromEntry(BB))
      continue;

    if (!Result.insert(BB))
      llvm_unreachable("Repeated basic blocks in extraction input");
  }

  LLVM_DEBUG(dbgs() << "Region front block: " << Result.front()->getName()
                    << '\n');

  for (auto *BB : Result) {
    if (!isBlockValidForExtraction(*BB, Result, AllowVarArgs, AllowAlloca))
      return {};

    // Make sure that the first block is not a landing pad.
    if (BB == Result.front()) {
      if (BB->isEHPad()) {
        LLVM_DEBUG(dbgs() << "The first block cannot be an unwind block\n");
        return {};
      }
      continue;
    }

    // All blocks other than the first must not have predecessors outside of
    // the subgraph which is being extracted.
    for (auto *PBB : predecessors(BB))
      if (!Result.count(PBB)) {
        LLVM_DEBUG(dbgs() << "No blocks in this region may have entries from "
                             "outside the region except for the first block!\n"
                          << "Problematic source BB: " << BB->getName() << "\n"
                          << "Problematic destination BB: " << PBB->getName()
                          << "\n");
        return {};
      }
  }

  return Result;
}

/// isAlignmentPreservedForAddrCast - Return true if the cast operation
/// for specified target preserves original alignment
static bool isAlignmentPreservedForAddrCast(const Triple &TargetTriple) {
  switch (TargetTriple.getArch()) {
  case Triple::ArchType::amdgcn:
  case Triple::ArchType::r600:
    return true;
  // TODO: Add other architectures for which we are certain that alignment
  // is preserved during address space cast operations.
  default:
    return false;
  }
  return false;
}

CodeExtractor::CodeExtractor(ArrayRef<BasicBlock *> BBs, DominatorTree *DT,
                             bool AggregateArgs, BlockFrequencyInfo *BFI,
                             BranchProbabilityInfo *BPI, AssumptionCache *AC,
                             bool AllowVarArgs, bool AllowAlloca,
                             BasicBlock *AllocationBlock, std::string Suffix,
                             bool ArgsInZeroAddressSpace)
    : DT(DT), AggregateArgs(AggregateArgs || AggregateArgsOpt), BFI(BFI),
      BPI(BPI), AC(AC), AllocationBlock(AllocationBlock),
      AllowVarArgs(AllowVarArgs),
      Blocks(buildExtractionBlockSet(BBs, DT, AllowVarArgs, AllowAlloca)),
      Suffix(Suffix), ArgsInZeroAddressSpace(ArgsInZeroAddressSpace) {}

/// definedInRegion - Return true if the specified value is defined in the
/// extracted region.
static bool definedInRegion(const SetVector<BasicBlock *> &Blocks, Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (Blocks.count(I->getParent()))
      return true;
  return false;
}

/// definedInCaller - Return true if the specified value is defined in the
/// function being code extracted, but not in the region being extracted.
/// These values must be passed in as live-ins to the function.
static bool definedInCaller(const SetVector<BasicBlock *> &Blocks, Value *V) {
  if (isa<Argument>(V)) return true;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (!Blocks.count(I->getParent()))
      return true;
  return false;
}

static BasicBlock *getCommonExitBlock(const SetVector<BasicBlock *> &Blocks) {
  BasicBlock *CommonExitBlock = nullptr;
  auto hasNonCommonExitSucc = [&](BasicBlock *Block) {
    for (auto *Succ : successors(Block)) {
      // Internal edges, ok.
      if (Blocks.count(Succ))
        continue;
      if (!CommonExitBlock) {
        CommonExitBlock = Succ;
        continue;
      }
      if (CommonExitBlock != Succ)
        return true;
    }
    return false;
  };

  if (any_of(Blocks, hasNonCommonExitSucc))
    return nullptr;

  return CommonExitBlock;
}

CodeExtractorAnalysisCache::CodeExtractorAnalysisCache(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &II : BB.instructionsWithoutDebug())
      if (auto *AI = dyn_cast<AllocaInst>(&II))
        Allocas.push_back(AI);

    findSideEffectInfoForBlock(BB);
  }
}

void CodeExtractorAnalysisCache::findSideEffectInfoForBlock(BasicBlock &BB) {
  for (Instruction &II : BB.instructionsWithoutDebug()) {
    unsigned Opcode = II.getOpcode();
    Value *MemAddr = nullptr;
    switch (Opcode) {
    case Instruction::Store:
    case Instruction::Load: {
      if (Opcode == Instruction::Store) {
        StoreInst *SI = cast<StoreInst>(&II);
        MemAddr = SI->getPointerOperand();
      } else {
        LoadInst *LI = cast<LoadInst>(&II);
        MemAddr = LI->getPointerOperand();
      }
      // Global variable can not be aliased with locals.
      if (isa<Constant>(MemAddr))
        break;
      Value *Base = MemAddr->stripInBoundsConstantOffsets();
      if (!isa<AllocaInst>(Base)) {
        SideEffectingBlocks.insert(&BB);
        return;
      }
      BaseMemAddrs[&BB].insert(Base);
      break;
    }
    default: {
      IntrinsicInst *IntrInst = dyn_cast<IntrinsicInst>(&II);
      if (IntrInst) {
        if (IntrInst->isLifetimeStartOrEnd())
          break;
        SideEffectingBlocks.insert(&BB);
        return;
      }
      // Treat all the other cases conservatively if it has side effects.
      if (II.mayHaveSideEffects()) {
        SideEffectingBlocks.insert(&BB);
        return;
      }
    }
    }
  }
}

bool CodeExtractorAnalysisCache::doesBlockContainClobberOfAddr(
    BasicBlock &BB, AllocaInst *Addr) const {
  if (SideEffectingBlocks.count(&BB))
    return true;
  auto It = BaseMemAddrs.find(&BB);
  if (It != BaseMemAddrs.end())
    return It->second.count(Addr);
  return false;
}

bool CodeExtractor::isLegalToShrinkwrapLifetimeMarkers(
    const CodeExtractorAnalysisCache &CEAC, Instruction *Addr) const {
  AllocaInst *AI = cast<AllocaInst>(Addr->stripInBoundsConstantOffsets());
  Function *Func = (*Blocks.begin())->getParent();
  for (BasicBlock &BB : *Func) {
    if (Blocks.count(&BB))
      continue;
    if (CEAC.doesBlockContainClobberOfAddr(BB, AI))
      return false;
  }
  return true;
}

BasicBlock *
CodeExtractor::findOrCreateBlockForHoisting(BasicBlock *CommonExitBlock) {
  BasicBlock *SinglePredFromOutlineRegion = nullptr;
  assert(!Blocks.count(CommonExitBlock) &&
         "Expect a block outside the region!");
  for (auto *Pred : predecessors(CommonExitBlock)) {
    if (!Blocks.count(Pred))
      continue;
    if (!SinglePredFromOutlineRegion) {
      SinglePredFromOutlineRegion = Pred;
    } else if (SinglePredFromOutlineRegion != Pred) {
      SinglePredFromOutlineRegion = nullptr;
      break;
    }
  }

  if (SinglePredFromOutlineRegion)
    return SinglePredFromOutlineRegion;

#ifndef NDEBUG
  auto getFirstPHI = [](BasicBlock *BB) {
    BasicBlock::iterator I = BB->begin();
    PHINode *FirstPhi = nullptr;
    while (I != BB->end()) {
      PHINode *Phi = dyn_cast<PHINode>(I);
      if (!Phi)
        break;
      if (!FirstPhi) {
        FirstPhi = Phi;
        break;
      }
    }
    return FirstPhi;
  };
  // If there are any phi nodes, the single pred either exists or has already
  // be created before code extraction.
  assert(!getFirstPHI(CommonExitBlock) && "Phi not expected");
#endif

  BasicBlock *NewExitBlock =
      CommonExitBlock->splitBasicBlock(CommonExitBlock->getFirstNonPHIIt());

  for (BasicBlock *Pred :
       llvm::make_early_inc_range(predecessors(CommonExitBlock))) {
    if (Blocks.count(Pred))
      continue;
    Pred->getTerminator()->replaceUsesOfWith(CommonExitBlock, NewExitBlock);
  }
  // Now add the old exit block to the outline region.
  Blocks.insert(CommonExitBlock);
  return CommonExitBlock;
}

// Find the pair of life time markers for address 'Addr' that are either
// defined inside the outline region or can legally be shrinkwrapped into the
// outline region. If there are not other untracked uses of the address, return
// the pair of markers if found; otherwise return a pair of nullptr.
CodeExtractor::LifetimeMarkerInfo
CodeExtractor::getLifetimeMarkers(const CodeExtractorAnalysisCache &CEAC,
                                  Instruction *Addr,
                                  BasicBlock *ExitBlock) const {
  LifetimeMarkerInfo Info;

  for (User *U : Addr->users()) {
    IntrinsicInst *IntrInst = dyn_cast<IntrinsicInst>(U);
    if (IntrInst) {
      // We don't model addresses with multiple start/end markers, but the
      // markers do not need to be in the region.
      if (IntrInst->getIntrinsicID() == Intrinsic::lifetime_start) {
        if (Info.LifeStart)
          return {};
        Info.LifeStart = IntrInst;
        continue;
      }
      if (IntrInst->getIntrinsicID() == Intrinsic::lifetime_end) {
        if (Info.LifeEnd)
          return {};
        Info.LifeEnd = IntrInst;
        continue;
      }
    }
    // Find untracked uses of the address, bail.
    if (!definedInRegion(Blocks, U))
      return {};
  }

  if (!Info.LifeStart || !Info.LifeEnd)
    return {};

  Info.SinkLifeStart = !definedInRegion(Blocks, Info.LifeStart);
  Info.HoistLifeEnd = !definedInRegion(Blocks, Info.LifeEnd);
  // Do legality check.
  if ((Info.SinkLifeStart || Info.HoistLifeEnd) &&
      !isLegalToShrinkwrapLifetimeMarkers(CEAC, Addr))
    return {};

  // Check to see if we have a place to do hoisting, if not, bail.
  if (Info.HoistLifeEnd && !ExitBlock)
    return {};

  return Info;
}

void CodeExtractor::findAllocas(const CodeExtractorAnalysisCache &CEAC,
                                ValueSet &SinkCands, ValueSet &HoistCands,
                                BasicBlock *&ExitBlock) const {
  Function *Func = (*Blocks.begin())->getParent();
  ExitBlock = getCommonExitBlock(Blocks);

  auto moveOrIgnoreLifetimeMarkers =
      [&](const LifetimeMarkerInfo &LMI) -> bool {
    if (!LMI.LifeStart)
      return false;
    if (LMI.SinkLifeStart) {
      LLVM_DEBUG(dbgs() << "Sinking lifetime.start: " << *LMI.LifeStart
                        << "\n");
      SinkCands.insert(LMI.LifeStart);
    }
    if (LMI.HoistLifeEnd) {
      LLVM_DEBUG(dbgs() << "Hoisting lifetime.end: " << *LMI.LifeEnd << "\n");
      HoistCands.insert(LMI.LifeEnd);
    }
    return true;
  };

  // Look up allocas in the original function in CodeExtractorAnalysisCache, as
  // this is much faster than walking all the instructions.
  for (AllocaInst *AI : CEAC.getAllocas()) {
    BasicBlock *BB = AI->getParent();
    if (Blocks.count(BB))
      continue;

    // As a prior call to extractCodeRegion() may have shrinkwrapped the alloca,
    // check whether it is actually still in the original function.
    Function *AIFunc = BB->getParent();
    if (AIFunc != Func)
      continue;

    LifetimeMarkerInfo MarkerInfo = getLifetimeMarkers(CEAC, AI, ExitBlock);
    bool Moved = moveOrIgnoreLifetimeMarkers(MarkerInfo);
    if (Moved) {
      LLVM_DEBUG(dbgs() << "Sinking alloca: " << *AI << "\n");
      SinkCands.insert(AI);
      continue;
    }

    // Find bitcasts in the outlined region that have lifetime marker users
    // outside that region. Replace the lifetime marker use with an
    // outside region bitcast to avoid unnecessary alloca/reload instructions
    // and extra lifetime markers.
    SmallVector<Instruction *, 2> LifetimeBitcastUsers;
    for (User *U : AI->users()) {
      if (!definedInRegion(Blocks, U))
        continue;

      if (U->stripInBoundsConstantOffsets() != AI)
        continue;

      Instruction *Bitcast = cast<Instruction>(U);
      for (User *BU : Bitcast->users()) {
        auto *IntrInst = dyn_cast<LifetimeIntrinsic>(BU);
        if (!IntrInst)
          continue;

        if (definedInRegion(Blocks, IntrInst))
          continue;

        LLVM_DEBUG(dbgs() << "Replace use of extracted region bitcast"
                          << *Bitcast << " in out-of-region lifetime marker "
                          << *IntrInst << "\n");
        LifetimeBitcastUsers.push_back(IntrInst);
      }
    }

    for (Instruction *I : LifetimeBitcastUsers) {
      Module *M = AIFunc->getParent();
      LLVMContext &Ctx = M->getContext();
      auto *Int8PtrTy = PointerType::getUnqual(Ctx);
      CastInst *CastI =
          CastInst::CreatePointerCast(AI, Int8PtrTy, "lt.cast", I->getIterator());
      I->replaceUsesOfWith(I->getOperand(1), CastI);
    }

    // Follow any bitcasts.
    SmallVector<Instruction *, 2> Bitcasts;
    SmallVector<LifetimeMarkerInfo, 2> BitcastLifetimeInfo;
    for (User *U : AI->users()) {
      if (U->stripInBoundsConstantOffsets() == AI) {
        Instruction *Bitcast = cast<Instruction>(U);
        LifetimeMarkerInfo LMI = getLifetimeMarkers(CEAC, Bitcast, ExitBlock);
        if (LMI.LifeStart) {
          Bitcasts.push_back(Bitcast);
          BitcastLifetimeInfo.push_back(LMI);
          continue;
        }
      }

      // Found unknown use of AI.
      if (!definedInRegion(Blocks, U)) {
        Bitcasts.clear();
        break;
      }
    }

    // Either no bitcasts reference the alloca or there are unknown uses.
    if (Bitcasts.empty())
      continue;

    LLVM_DEBUG(dbgs() << "Sinking alloca (via bitcast): " << *AI << "\n");
    SinkCands.insert(AI);
    for (unsigned I = 0, E = Bitcasts.size(); I != E; ++I) {
      Instruction *BitcastAddr = Bitcasts[I];
      const LifetimeMarkerInfo &LMI = BitcastLifetimeInfo[I];
      assert(LMI.LifeStart &&
             "Unsafe to sink bitcast without lifetime markers");
      moveOrIgnoreLifetimeMarkers(LMI);
      if (!definedInRegion(Blocks, BitcastAddr)) {
        LLVM_DEBUG(dbgs() << "Sinking bitcast-of-alloca: " << *BitcastAddr
                          << "\n");
        SinkCands.insert(BitcastAddr);
      }
    }
  }
}

bool CodeExtractor::isEligible() const {
  if (Blocks.empty())
    return false;
  BasicBlock *Header = *Blocks.begin();
  Function *F = Header->getParent();

  // For functions with varargs, check that varargs handling is only done in the
  // outlined function, i.e vastart and vaend are only used in outlined blocks.
  if (AllowVarArgs && F->getFunctionType()->isVarArg()) {
    auto containsVarArgIntrinsic = [](const Instruction &I) {
      if (const CallInst *CI = dyn_cast<CallInst>(&I))
        if (const Function *Callee = CI->getCalledFunction())
          return Callee->getIntrinsicID() == Intrinsic::vastart ||
                 Callee->getIntrinsicID() == Intrinsic::vaend;
      return false;
    };

    for (auto &BB : *F) {
      if (Blocks.count(&BB))
        continue;
      if (llvm::any_of(BB, containsVarArgIntrinsic))
        return false;
    }
  }
  // stacksave as input implies stackrestore in the outlined function.
  // This can confuse prolog epilog insertion phase.
  // stacksave's uses must not cross outlined function.
  for (BasicBlock *BB : Blocks) {
    for (Instruction &I : *BB) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      bool IsSave = II->getIntrinsicID() == Intrinsic::stacksave;
      bool IsRestore = II->getIntrinsicID() == Intrinsic::stackrestore;
      if (IsSave && any_of(II->users(), [&Blks = this->Blocks](User *U) {
            return !definedInRegion(Blks, U);
          }))
        return false;
      if (IsRestore && !definedInRegion(Blocks, II->getArgOperand(0)))
        return false;
    }
  }
  return true;
}

void CodeExtractor::findInputsOutputs(ValueSet &Inputs, ValueSet &Outputs,
                                      const ValueSet &SinkCands,
                                      bool CollectGlobalInputs) const {
  for (BasicBlock *BB : Blocks) {
    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      for (auto &OI : II.operands()) {
        Value *V = OI;
        if (!SinkCands.count(V) &&
            (definedInCaller(Blocks, V) ||
             (CollectGlobalInputs && llvm::isa<llvm::GlobalVariable>(V))))
          Inputs.insert(V);
      }

      for (User *U : II.users())
        if (!definedInRegion(Blocks, U)) {
          Outputs.insert(&II);
          break;
        }
    }
  }
}

/// severSplitPHINodesOfEntry - If a PHI node has multiple inputs from outside
/// of the region, we need to split the entry block of the region so that the
/// PHI node is easier to deal with.
void CodeExtractor::severSplitPHINodesOfEntry(BasicBlock *&Header) {
  unsigned NumPredsFromRegion = 0;
  unsigned NumPredsOutsideRegion = 0;

  if (Header != &Header->getParent()->getEntryBlock()) {
    PHINode *PN = dyn_cast<PHINode>(Header->begin());
    if (!PN) return;  // No PHI nodes.

    // If the header node contains any PHI nodes, check to see if there is more
    // than one entry from outside the region.  If so, we need to sever the
    // header block into two.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (Blocks.count(PN->getIncomingBlock(i)))
        ++NumPredsFromRegion;
      else
        ++NumPredsOutsideRegion;

    // If there is one (or fewer) predecessor from outside the region, we don't
    // need to do anything special.
    if (NumPredsOutsideRegion <= 1) return;
  }

  // Otherwise, we need to split the header block into two pieces: one
  // containing PHI nodes merging values from outside of the region, and a
  // second that contains all of the code for the block and merges back any
  // incoming values from inside of the region.
  BasicBlock *NewBB = SplitBlock(Header, Header->getFirstNonPHIIt(), DT);

  // We only want to code extract the second block now, and it becomes the new
  // header of the region.
  BasicBlock *OldPred = Header;
  Blocks.remove(OldPred);
  Blocks.insert(NewBB);
  Header = NewBB;

  // Okay, now we need to adjust the PHI nodes and any branches from within the
  // region to go to the new header block instead of the old header block.
  if (NumPredsFromRegion) {
    PHINode *PN = cast<PHINode>(OldPred->begin());
    // Loop over all of the predecessors of OldPred that are in the region,
    // changing them to branch to NewBB instead.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (Blocks.count(PN->getIncomingBlock(i))) {
        Instruction *TI = PN->getIncomingBlock(i)->getTerminator();
        TI->replaceUsesOfWith(OldPred, NewBB);
      }

    // Okay, everything within the region is now branching to the right block, we
    // just have to update the PHI nodes now, inserting PHI nodes into NewBB.
    BasicBlock::iterator AfterPHIs;
    for (AfterPHIs = OldPred->begin(); isa<PHINode>(AfterPHIs); ++AfterPHIs) {
      PHINode *PN = cast<PHINode>(AfterPHIs);
      // Create a new PHI node in the new region, which has an incoming value
      // from OldPred of PN.
      PHINode *NewPN = PHINode::Create(PN->getType(), 1 + NumPredsFromRegion,
                                       PN->getName() + ".ce");
      NewPN->insertBefore(NewBB->begin());
      PN->replaceAllUsesWith(NewPN);
      NewPN->addIncoming(PN, OldPred);

      // Loop over all of the incoming value in PN, moving them to NewPN if they
      // are from the extracted region.
      for (unsigned i = 0; i != PN->getNumIncomingValues(); ++i) {
        if (Blocks.count(PN->getIncomingBlock(i))) {
          NewPN->addIncoming(PN->getIncomingValue(i), PN->getIncomingBlock(i));
          PN->removeIncomingValue(i);
          --i;
        }
      }
    }
  }
}

/// severSplitPHINodesOfExits - if PHI nodes in exit blocks have inputs from
/// outlined region, we split these PHIs on two: one with inputs from region
/// and other with remaining incoming blocks; then first PHIs are placed in
/// outlined region.
void CodeExtractor::severSplitPHINodesOfExits() {
  for (BasicBlock *ExitBB : ExtractedFuncRetVals) {
    BasicBlock *NewBB = nullptr;

    for (PHINode &PN : ExitBB->phis()) {
      // Find all incoming values from the outlining region.
      SmallVector<unsigned, 2> IncomingVals;
      for (unsigned i = 0; i < PN.getNumIncomingValues(); ++i)
        if (Blocks.count(PN.getIncomingBlock(i)))
          IncomingVals.push_back(i);

      // Do not process PHI if there is one (or fewer) predecessor from region.
      // If PHI has exactly one predecessor from region, only this one incoming
      // will be replaced on codeRepl block, so it should be safe to skip PHI.
      if (IncomingVals.size() <= 1)
        continue;

      // Create block for new PHIs and add it to the list of outlined if it
      // wasn't done before.
      if (!NewBB) {
        NewBB = BasicBlock::Create(ExitBB->getContext(),
                                   ExitBB->getName() + ".split",
                                   ExitBB->getParent(), ExitBB);
        SmallVector<BasicBlock *, 4> Preds(predecessors(ExitBB));
        for (BasicBlock *PredBB : Preds)
          if (Blocks.count(PredBB))
            PredBB->getTerminator()->replaceUsesOfWith(ExitBB, NewBB);
        BranchInst::Create(ExitBB, NewBB);
        Blocks.insert(NewBB);
      }

      // Split this PHI.
      PHINode *NewPN = PHINode::Create(PN.getType(), IncomingVals.size(),
                                       PN.getName() + ".ce");
      NewPN->insertBefore(NewBB->getFirstNonPHIIt());
      for (unsigned i : IncomingVals)
        NewPN->addIncoming(PN.getIncomingValue(i), PN.getIncomingBlock(i));
      for (unsigned i : reverse(IncomingVals))
        PN.removeIncomingValue(i, false);
      PN.addIncoming(NewPN, NewBB);
    }
  }
}

void CodeExtractor::splitReturnBlocks() {
  for (BasicBlock *Block : Blocks)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(Block->getTerminator())) {
      BasicBlock *New =
          Block->splitBasicBlock(RI->getIterator(), Block->getName() + ".ret");
      if (DT) {
        // Old dominates New. New node dominates all other nodes dominated
        // by Old.
        DomTreeNode *OldNode = DT->getNode(Block);
        SmallVector<DomTreeNode *, 8> Children(OldNode->begin(),
                                               OldNode->end());

        DomTreeNode *NewNode = DT->addNewBlock(New, Block);

        for (DomTreeNode *I : Children)
          DT->changeImmediateDominator(I, NewNode);
      }
    }
}

Function *CodeExtractor::constructFunctionDeclaration(
    const ValueSet &inputs, const ValueSet &outputs, BlockFrequency EntryFreq,
    const Twine &Name, ValueSet &StructValues, StructType *&StructTy) {
  LLVM_DEBUG(dbgs() << "inputs: " << inputs.size() << "\n");
  LLVM_DEBUG(dbgs() << "outputs: " << outputs.size() << "\n");

  Function *oldFunction = Blocks.front()->getParent();
  Module *M = Blocks.front()->getModule();

  // Assemble the function's parameter lists.
  std::vector<Type *> ParamTy;
  std::vector<Type *> AggParamTy;
  const DataLayout &DL = M->getDataLayout();

  // Add the types of the input values to the function's argument list
  for (Value *value : inputs) {
    LLVM_DEBUG(dbgs() << "value used in func: " << *value << "\n");
    if (AggregateArgs && !ExcludeArgsFromAggregate.contains(value)) {
      AggParamTy.push_back(value->getType());
      StructValues.insert(value);
    } else
      ParamTy.push_back(value->getType());
  }

  // Add the types of the output values to the function's argument list.
  for (Value *output : outputs) {
    LLVM_DEBUG(dbgs() << "instr used in func: " << *output << "\n");
    if (AggregateArgs && !ExcludeArgsFromAggregate.contains(output)) {
      AggParamTy.push_back(output->getType());
      StructValues.insert(output);
    } else
      ParamTy.push_back(
          PointerType::get(output->getContext(), DL.getAllocaAddrSpace()));
  }

  assert(
      (ParamTy.size() + AggParamTy.size()) ==
          (inputs.size() + outputs.size()) &&
      "Number of scalar and aggregate params does not match inputs, outputs");
  assert((StructValues.empty() || AggregateArgs) &&
         "Expeced StructValues only with AggregateArgs set");

  // Concatenate scalar and aggregate params in ParamTy.
  if (!AggParamTy.empty()) {
    StructTy = StructType::get(M->getContext(), AggParamTy);
    ParamTy.push_back(PointerType::get(
        M->getContext(), ArgsInZeroAddressSpace ? 0 : DL.getAllocaAddrSpace()));
  }

  Type *RetTy = getSwitchType();
  LLVM_DEBUG({
    dbgs() << "Function type: " << *RetTy << " f(";
    for (Type *i : ParamTy)
      dbgs() << *i << ", ";
    dbgs() << ")\n";
  });

  FunctionType *funcType = FunctionType::get(
      RetTy, ParamTy, AllowVarArgs && oldFunction->isVarArg());

  // Create the new function
  Function *newFunction =
      Function::Create(funcType, GlobalValue::InternalLinkage,
                       oldFunction->getAddressSpace(), Name, M);

  // Propagate personality info to the new function if there is one.
  if (oldFunction->hasPersonalityFn())
    newFunction->setPersonalityFn(oldFunction->getPersonalityFn());

  // Inherit all of the target dependent attributes and white-listed
  // target independent attributes.
  //  (e.g. If the extracted region contains a call to an x86.sse
  //  instruction we need to make sure that the extracted region has the
  //  "target-features" attribute allowing it to be lowered.
  // FIXME: This should be changed to check to see if a specific
  //           attribute can not be inherited.
  for (const auto &Attr : oldFunction->getAttributes().getFnAttrs()) {
    if (Attr.isStringAttribute()) {
      if (Attr.getKindAsString() == "thunk")
        continue;
    } else
      switch (Attr.getKindAsEnum()) {
      // Those attributes cannot be propagated safely. Explicitly list them
      // here so we get a warning if new attributes are added.
      case Attribute::AllocSize:
      case Attribute::Builtin:
      case Attribute::Convergent:
      case Attribute::JumpTable:
      case Attribute::Naked:
      case Attribute::NoBuiltin:
      case Attribute::NoMerge:
      case Attribute::NoReturn:
      case Attribute::NoSync:
      case Attribute::ReturnsTwice:
      case Attribute::Speculatable:
      case Attribute::StackAlignment:
      case Attribute::WillReturn:
      case Attribute::AllocKind:
      case Attribute::PresplitCoroutine:
      case Attribute::Memory:
      case Attribute::NoFPClass:
      case Attribute::CoroDestroyOnlyWhenComplete:
      case Attribute::CoroElideSafe:
      case Attribute::NoDivergenceSource:
        continue;
      // Those attributes should be safe to propagate to the extracted function.
      case Attribute::AlwaysInline:
      case Attribute::Cold:
      case Attribute::DisableSanitizerInstrumentation:
      case Attribute::FnRetThunkExtern:
      case Attribute::Hot:
      case Attribute::HybridPatchable:
      case Attribute::NoRecurse:
      case Attribute::InlineHint:
      case Attribute::MinSize:
      case Attribute::NoCallback:
      case Attribute::NoDuplicate:
      case Attribute::NoFree:
      case Attribute::NoImplicitFloat:
      case Attribute::NoInline:
      case Attribute::NonLazyBind:
      case Attribute::NoRedZone:
      case Attribute::NoUnwind:
      case Attribute::NoSanitizeBounds:
      case Attribute::NoSanitizeCoverage:
      case Attribute::NullPointerIsValid:
      case Attribute::OptimizeForDebugging:
      case Attribute::OptForFuzzing:
      case Attribute::OptimizeNone:
      case Attribute::OptimizeForSize:
      case Attribute::SafeStack:
      case Attribute::ShadowCallStack:
      case Attribute::SanitizeAddress:
      case Attribute::SanitizeMemory:
      case Attribute::SanitizeNumericalStability:
      case Attribute::SanitizeThread:
      case Attribute::SanitizeType:
      case Attribute::SanitizeHWAddress:
      case Attribute::SanitizeMemTag:
      case Attribute::SanitizeRealtime:
      case Attribute::SanitizeRealtimeBlocking:
      case Attribute::SpeculativeLoadHardening:
      case Attribute::StackProtect:
      case Attribute::StackProtectReq:
      case Attribute::StackProtectStrong:
      case Attribute::StrictFP:
      case Attribute::UWTable:
      case Attribute::VScaleRange:
      case Attribute::NoCfCheck:
      case Attribute::MustProgress:
      case Attribute::NoProfile:
      case Attribute::SkipProfile:
        break;
      // These attributes cannot be applied to functions.
      case Attribute::Alignment:
      case Attribute::AllocatedPointer:
      case Attribute::AllocAlign:
      case Attribute::ByVal:
      case Attribute::Captures:
      case Attribute::Dereferenceable:
      case Attribute::DereferenceableOrNull:
      case Attribute::ElementType:
      case Attribute::InAlloca:
      case Attribute::InReg:
      case Attribute::Nest:
      case Attribute::NoAlias:
      case Attribute::NoUndef:
      case Attribute::NonNull:
      case Attribute::Preallocated:
      case Attribute::ReadNone:
      case Attribute::ReadOnly:
      case Attribute::Returned:
      case Attribute::SExt:
      case Attribute::StructRet:
      case Attribute::SwiftError:
      case Attribute::SwiftSelf:
      case Attribute::SwiftAsync:
      case Attribute::ZExt:
      case Attribute::ImmArg:
      case Attribute::ByRef:
      case Attribute::WriteOnly:
      case Attribute::Writable:
      case Attribute::DeadOnUnwind:
      case Attribute::Range:
      case Attribute::Initializes:
      case Attribute::NoExt:
      //  These are not really attributes.
      case Attribute::None:
      case Attribute::EndAttrKinds:
      case Attribute::EmptyKey:
      case Attribute::TombstoneKey:
      case Attribute::DeadOnReturn:
        llvm_unreachable("Not a function attribute");
      }

    newFunction->addFnAttr(Attr);
  }

  // Create scalar and aggregate iterators to name all of the arguments we
  // inserted.
  Function::arg_iterator ScalarAI = newFunction->arg_begin();

  // Set names and attributes for input and output arguments.
  ScalarAI = newFunction->arg_begin();
  for (Value *input : inputs) {
    if (StructValues.contains(input))
      continue;

    ScalarAI->setName(input->getName());
    if (input->isSwiftError())
      newFunction->addParamAttr(ScalarAI - newFunction->arg_begin(),
                                Attribute::SwiftError);
    ++ScalarAI;
  }
  for (Value *output : outputs) {
    if (StructValues.contains(output))
      continue;

    ScalarAI->setName(output->getName() + ".out");
    ++ScalarAI;
  }

  // Update the entry count of the function.
  if (BFI) {
    auto Count = BFI->getProfileCountFromFreq(EntryFreq);
    if (Count.has_value())
      newFunction->setEntryCount(
          ProfileCount(*Count, Function::PCT_Real)); // FIXME
  }

  return newFunction;
}

/// If the original function has debug info, we have to add a debug location
/// to the new branch instruction from the artificial entry block.
/// We use the debug location of the first instruction in the extracted
/// blocks, as there is no other equivalent line in the source code.
static void applyFirstDebugLoc(Function *oldFunction,
                               ArrayRef<BasicBlock *> Blocks,
                               Instruction *BranchI) {
  if (oldFunction->getSubprogram()) {
    any_of(Blocks, [&BranchI](const BasicBlock *BB) {
      return any_of(*BB, [&BranchI](const Instruction &I) {
        if (!I.getDebugLoc())
          return false;
        BranchI->setDebugLoc(I.getDebugLoc());
        return true;
      });
    });
  }
}

/// Erase lifetime.start markers which reference inputs to the extraction
/// region, and insert the referenced memory into \p LifetimesStart.
///
/// The extraction region is defined by a set of blocks (\p Blocks), and a set
/// of allocas which will be moved from the caller function into the extracted
/// function (\p SunkAllocas).
static void eraseLifetimeMarkersOnInputs(const SetVector<BasicBlock *> &Blocks,
                                         const SetVector<Value *> &SunkAllocas,
                                         SetVector<Value *> &LifetimesStart) {
  for (BasicBlock *BB : Blocks) {
    for (Instruction &I : llvm::make_early_inc_range(*BB)) {
      auto *II = dyn_cast<LifetimeIntrinsic>(&I);
      if (!II)
        continue;

      // Get the memory operand of the lifetime marker. If the underlying
      // object is a sunk alloca, or is otherwise defined in the extraction
      // region, the lifetime marker must not be erased.
      Value *Mem = II->getOperand(1)->stripInBoundsOffsets();
      if (SunkAllocas.count(Mem) || definedInRegion(Blocks, Mem))
        continue;

      if (II->getIntrinsicID() == Intrinsic::lifetime_start)
        LifetimesStart.insert(Mem);
      II->eraseFromParent();
    }
  }
}

/// Insert lifetime start/end markers surrounding the call to the new function
/// for objects defined in the caller.
static void insertLifetimeMarkersSurroundingCall(
    Module *M, ArrayRef<Value *> LifetimesStart, ArrayRef<Value *> LifetimesEnd,
    CallInst *TheCall) {
  LLVMContext &Ctx = M->getContext();
  auto NegativeOne = ConstantInt::getSigned(Type::getInt64Ty(Ctx), -1);
  Instruction *Term = TheCall->getParent()->getTerminator();

  // Emit lifetime markers for the pointers given in \p Objects. Insert the
  // markers before the call if \p InsertBefore, and after the call otherwise.
  auto insertMarkers = [&](Intrinsic::ID MarkerFunc, ArrayRef<Value *> Objects,
                           bool InsertBefore) {
    for (Value *Mem : Objects) {
      assert((!isa<Instruction>(Mem) || cast<Instruction>(Mem)->getFunction() ==
                                            TheCall->getFunction()) &&
             "Input memory not defined in original function");

      Function *Func =
          Intrinsic::getOrInsertDeclaration(M, MarkerFunc, Mem->getType());
      auto Marker = CallInst::Create(Func, {NegativeOne, Mem});
      if (InsertBefore)
        Marker->insertBefore(TheCall->getIterator());
      else
        Marker->insertBefore(Term->getIterator());
    }
  };

  if (!LifetimesStart.empty()) {
    insertMarkers(Intrinsic::lifetime_start, LifetimesStart,
                  /*InsertBefore=*/true);
  }

  if (!LifetimesEnd.empty()) {
    insertMarkers(Intrinsic::lifetime_end, LifetimesEnd,
                  /*InsertBefore=*/false);
  }
}

void CodeExtractor::moveCodeToFunction(Function *newFunction) {
  auto newFuncIt = newFunction->begin();
  for (BasicBlock *Block : Blocks) {
    // Delete the basic block from the old function, and the list of blocks
    Block->removeFromParent();

    // Insert this basic block into the new function
    // Insert the original blocks after the entry block created
    // for the new function. The entry block may be followed
    // by a set of exit blocks at this point, but these exit
    // blocks better be placed at the end of the new function.
    newFuncIt = newFunction->insert(std::next(newFuncIt), Block);
  }
}

void CodeExtractor::calculateNewCallTerminatorWeights(
    BasicBlock *CodeReplacer,
    const DenseMap<BasicBlock *, BlockFrequency> &ExitWeights,
    BranchProbabilityInfo *BPI) {
  using Distribution = BlockFrequencyInfoImplBase::Distribution;
  using BlockNode = BlockFrequencyInfoImplBase::BlockNode;

  // Update the branch weights for the exit block.
  Instruction *TI = CodeReplacer->getTerminator();
  SmallVector<unsigned, 8> BranchWeights(TI->getNumSuccessors(), 0);

  // Block Frequency distribution with dummy node.
  Distribution BranchDist;

  SmallVector<BranchProbability, 4> EdgeProbabilities(
      TI->getNumSuccessors(), BranchProbability::getUnknown());

  // Add each of the frequencies of the successors.
  for (unsigned i = 0, e = TI->getNumSuccessors(); i < e; ++i) {
    BlockNode ExitNode(i);
    uint64_t ExitFreq = ExitWeights.lookup(TI->getSuccessor(i)).getFrequency();
    if (ExitFreq != 0)
      BranchDist.addExit(ExitNode, ExitFreq);
    else
      EdgeProbabilities[i] = BranchProbability::getZero();
  }

  // Check for no total weight.
  if (BranchDist.Total == 0) {
    BPI->setEdgeProbability(CodeReplacer, EdgeProbabilities);
    return;
  }

  // Normalize the distribution so that they can fit in unsigned.
  BranchDist.normalize();

  // Create normalized branch weights and set the metadata.
  for (unsigned I = 0, E = BranchDist.Weights.size(); I < E; ++I) {
    const auto &Weight = BranchDist.Weights[I];

    // Get the weight and update the current BFI.
    BranchWeights[Weight.TargetNode.Index] = Weight.Amount;
    BranchProbability BP(Weight.Amount, BranchDist.Total);
    EdgeProbabilities[Weight.TargetNode.Index] = BP;
  }
  BPI->setEdgeProbability(CodeReplacer, EdgeProbabilities);
  TI->setMetadata(
      LLVMContext::MD_prof,
      MDBuilder(TI->getContext()).createBranchWeights(BranchWeights));
}

/// Erase debug info intrinsics which refer to values in \p F but aren't in
/// \p F.
static void eraseDebugIntrinsicsWithNonLocalRefs(Function &F) {
  for (Instruction &I : instructions(F)) {
    SmallVector<DbgVariableRecord *, 4> DbgVariableRecords;
    findDbgUsers(&I, DbgVariableRecords);
    for (DbgVariableRecord *DVR : DbgVariableRecords)
      if (DVR->getFunction() != &F)
        DVR->eraseFromParent();
  }
}

/// Fix up the debug info in the old and new functions. Following changes are
/// done.
/// 1. If a debug record points to a value that has been replaced, update the
///    record to use the new value.
/// 2. If an Input value that has been replaced was used as a location of a
///    debug record in the Parent function, then materealize a similar record in
///    the new function.
/// 3. Point line locations and debug intrinsics to the new subprogram scope
/// 4. Remove intrinsics which point to values outside of the new function.
static void fixupDebugInfoPostExtraction(Function &OldFunc, Function &NewFunc,
                                         CallInst &TheCall,
                                         const SetVector<Value *> &Inputs,
                                         ArrayRef<Value *> NewValues) {
  DISubprogram *OldSP = OldFunc.getSubprogram();
  LLVMContext &Ctx = OldFunc.getContext();

  if (!OldSP) {
    // Erase any debug info the new function contains.
    stripDebugInfo(NewFunc);
    // Make sure the old function doesn't contain any non-local metadata refs.
    eraseDebugIntrinsicsWithNonLocalRefs(NewFunc);
    return;
  }

  // Create a subprogram for the new function. Leave out a description of the
  // function arguments, as the parameters don't correspond to anything at the
  // source level.
  assert(OldSP->getUnit() && "Missing compile unit for subprogram");
  DIBuilder DIB(*OldFunc.getParent(), /*AllowUnresolved=*/false,
                OldSP->getUnit());
  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  DISubprogram::DISPFlags SPFlags = DISubprogram::SPFlagDefinition |
                                    DISubprogram::SPFlagOptimized |
                                    DISubprogram::SPFlagLocalToUnit;
  auto NewSP = DIB.createFunction(
      OldSP->getUnit(), NewFunc.getName(), NewFunc.getName(), OldSP->getFile(),
      /*LineNo=*/0, SPType, /*ScopeLine=*/0, DINode::FlagZero, SPFlags);
  NewFunc.setSubprogram(NewSP);

  auto UpdateOrInsertDebugRecord = [&](auto *DR, Value *OldLoc, Value *NewLoc,
                                       DIExpression *Expr, bool Declare) {
    if (DR->getParent()->getParent() == &NewFunc) {
      DR->replaceVariableLocationOp(OldLoc, NewLoc);
      return;
    }
    if (Declare) {
      DIB.insertDeclare(NewLoc, DR->getVariable(), Expr, DR->getDebugLoc(),
                        &NewFunc.getEntryBlock());
      return;
    }
    DIB.insertDbgValueIntrinsic(
        NewLoc, DR->getVariable(), Expr, DR->getDebugLoc(),
        NewFunc.getEntryBlock().getTerminator()->getIterator());
  };
  for (auto [Input, NewVal] : zip_equal(Inputs, NewValues)) {
    SmallVector<DbgVariableRecord *, 1> DPUsers;
    findDbgUsers(Input, DPUsers);
    DIExpression *Expr = DIB.createExpression();

    // Iterate the debud users of the Input values. If they are in the extracted
    // function then update their location with the new value. If they are in
    // the parent function then create a similar debug record.
    for (auto *DVR : DPUsers)
      UpdateOrInsertDebugRecord(DVR, Input, NewVal, Expr, DVR->isDbgDeclare());
  }

  auto IsInvalidLocation = [&NewFunc](Value *Location) {
    // Location is invalid if it isn't a constant, an instruction or an
    // argument, or is an instruction/argument but isn't in the new function.
    if (!Location || (!isa<Constant>(Location) && !isa<Argument>(Location) &&
                      !isa<Instruction>(Location)))
      return true;

    if (Argument *Arg = dyn_cast<Argument>(Location))
      return Arg->getParent() != &NewFunc;
    if (Instruction *LocationInst = dyn_cast<Instruction>(Location))
      return LocationInst->getFunction() != &NewFunc;
    return false;
  };

  // Debug intrinsics in the new function need to be updated in one of two
  // ways:
  //  1) They need to be deleted, because they describe a value in the old
  //     function.
  //  2) They need to point to fresh metadata, e.g. because they currently
  //     point to a variable in the wrong scope.
  SmallDenseMap<DINode *, DINode *> RemappedMetadata;
  SmallVector<DbgVariableRecord *, 4> DVRsToDelete;
  DenseMap<const MDNode *, MDNode *> Cache;

  auto GetUpdatedDIVariable = [&](DILocalVariable *OldVar) {
    DINode *&NewVar = RemappedMetadata[OldVar];
    if (!NewVar) {
      DILocalScope *NewScope = DILocalScope::cloneScopeForSubprogram(
          *OldVar->getScope(), *NewSP, Ctx, Cache);
      NewVar = DIB.createAutoVariable(
          NewScope, OldVar->getName(), OldVar->getFile(), OldVar->getLine(),
          OldVar->getType(), /*AlwaysPreserve=*/false, DINode::FlagZero,
          OldVar->getDWARFMemorySpace(),
          OldVar->getAlignInBits());
    }
    return cast<DILocalVariable>(NewVar);
  };

  auto UpdateDbgLabel = [&](auto *LabelRecord) {
    // Point the label record to a fresh label within the new function if
    // the record was not inlined from some other function.
    if (LabelRecord->getDebugLoc().getInlinedAt())
      return;
    DILabel *OldLabel = LabelRecord->getLabel();
    DINode *&NewLabel = RemappedMetadata[OldLabel];
    if (!NewLabel) {
      DILocalScope *NewScope = DILocalScope::cloneScopeForSubprogram(
          *OldLabel->getScope(), *NewSP, Ctx, Cache);
      NewLabel =
          DILabel::get(Ctx, NewScope, OldLabel->getName(), OldLabel->getFile(),
                       OldLabel->getLine(), OldLabel->getColumn(),
                       OldLabel->isArtificial(), OldLabel->getCoroSuspendIdx());
    }
    LabelRecord->setLabel(cast<DILabel>(NewLabel));
  };

  auto UpdateDbgRecordsOnInst = [&](Instruction &I) -> void {
    for (DbgRecord &DR : I.getDbgRecordRange()) {
      if (DbgLabelRecord *DLR = dyn_cast<DbgLabelRecord>(&DR)) {
        UpdateDbgLabel(DLR);
        continue;
      }

      DbgVariableRecord &DVR = cast<DbgVariableRecord>(DR);
      // If any of the used locations are invalid, delete the record.
      if (any_of(DVR.location_ops(), IsInvalidLocation)) {
        DVRsToDelete.push_back(&DVR);
        continue;
      }

      // DbgAssign intrinsics have an extra Value argument:
      if (DVR.isDbgAssign() && IsInvalidLocation(DVR.getAddress())) {
        DVRsToDelete.push_back(&DVR);
        continue;
      }

      // If the variable was in the scope of the old function, i.e. it was not
      // inlined, point the intrinsic to a fresh variable within the new
      // function.
      if (!DVR.getDebugLoc().getInlinedAt())
        DVR.setVariable(GetUpdatedDIVariable(DVR.getVariable()));
    }
  };

  for (Instruction &I : instructions(NewFunc))
    UpdateDbgRecordsOnInst(I);

  for (auto *DVR : DVRsToDelete)
    DVR->getMarker()->MarkedInstr->dropOneDbgRecord(DVR);
  DIB.finalizeSubprogram(NewSP);

  // Fix up the scope information attached to the line locations and the
  // debug assignment metadata in the new function.
  DenseMap<DIAssignID *, DIAssignID *> AssignmentIDMap;
  for (Instruction &I : instructions(NewFunc)) {
    if (const DebugLoc &DL = I.getDebugLoc())
      I.setDebugLoc(
          DebugLoc::replaceInlinedAtSubprogram(DL, *NewSP, Ctx, Cache));
    for (DbgRecord &DR : I.getDbgRecordRange())
      DR.setDebugLoc(DebugLoc::replaceInlinedAtSubprogram(DR.getDebugLoc(),
                                                          *NewSP, Ctx, Cache));

    // Loop info metadata may contain line locations. Fix them up.
    auto updateLoopInfoLoc = [&Ctx, &Cache, NewSP](Metadata *MD) -> Metadata * {
      if (auto *Loc = dyn_cast_or_null<DILocation>(MD))
        return DebugLoc::replaceInlinedAtSubprogram(Loc, *NewSP, Ctx, Cache);
      return MD;
    };
    updateLoopMetadataDebugLocations(I, updateLoopInfoLoc);
    at::remapAssignID(AssignmentIDMap, I);
  }
  if (!TheCall.getDebugLoc())
    TheCall.setDebugLoc(DILocation::get(Ctx, 0, 0, OldSP));

  eraseDebugIntrinsicsWithNonLocalRefs(NewFunc);
}

Function *
CodeExtractor::extractCodeRegion(const CodeExtractorAnalysisCache &CEAC) {
  ValueSet Inputs, Outputs;
  return extractCodeRegion(CEAC, Inputs, Outputs);
}

Function *
CodeExtractor::extractCodeRegion(const CodeExtractorAnalysisCache &CEAC,
                                 ValueSet &inputs, ValueSet &outputs) {
  if (!isEligible())
    return nullptr;

  // Assumption: this is a single-entry code region, and the header is the first
  // block in the region.
  BasicBlock *header = *Blocks.begin();
  Function *oldFunction = header->getParent();

  normalizeCFGForExtraction(header);

  // Remove @llvm.assume calls that will be moved to the new function from the
  // old function's assumption cache.
  for (BasicBlock *Block : Blocks) {
    for (Instruction &I : llvm::make_early_inc_range(*Block)) {
      if (auto *AI = dyn_cast<AssumeInst>(&I)) {
        if (AC)
          AC->unregisterAssumption(AI);
        AI->eraseFromParent();
      }
    }
  }

  ValueSet SinkingCands, HoistingCands;
  BasicBlock *CommonExit = nullptr;
  findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  assert(HoistingCands.empty() || CommonExit);

  // Find inputs to, outputs from the code region.
  findInputsOutputs(inputs, outputs, SinkingCands);

  // Collect objects which are inputs to the extraction region and also
  // referenced by lifetime start markers within it. The effects of these
  // markers must be replicated in the calling function to prevent the stack
  // coloring pass from merging slots which store input objects.
  ValueSet LifetimesStart;
  eraseLifetimeMarkersOnInputs(Blocks, SinkingCands, LifetimesStart);

  if (!HoistingCands.empty()) {
    auto *HoistToBlock = findOrCreateBlockForHoisting(CommonExit);
    Instruction *TI = HoistToBlock->getTerminator();
    for (auto *II : HoistingCands)
      cast<Instruction>(II)->moveBefore(TI->getIterator());
    computeExtractedFuncRetVals();
  }

  // CFG/ExitBlocks must not change hereafter

  // Calculate the entry frequency of the new function before we change the root
  //   block.
  BlockFrequency EntryFreq;
  DenseMap<BasicBlock *, BlockFrequency> ExitWeights;
  if (BFI) {
    assert(BPI && "Both BPI and BFI are required to preserve profile info");
    for (BasicBlock *Pred : predecessors(header)) {
      if (Blocks.count(Pred))
        continue;
      EntryFreq +=
          BFI->getBlockFreq(Pred) * BPI->getEdgeProbability(Pred, header);
    }

    for (BasicBlock *Succ : ExtractedFuncRetVals) {
      for (BasicBlock *Block : predecessors(Succ)) {
        if (!Blocks.count(Block))
          continue;

        // Update the branch weight for this successor.
        BlockFrequency &BF = ExitWeights[Succ];
        BF += BFI->getBlockFreq(Block) * BPI->getEdgeProbability(Block, Succ);
      }
    }
  }

  // Determine position for the replacement code. Do so before header is moved
  // to the new function.
  BasicBlock *ReplIP = header;
  while (ReplIP && Blocks.count(ReplIP))
    ReplIP = ReplIP->getNextNode();

  // Construct new function based on inputs/outputs & add allocas for all defs.
  std::string SuffixToUse =
      Suffix.empty()
          ? (header->getName().empty() ? "extracted" : header->getName().str())
          : Suffix;

  ValueSet StructValues;
  StructType *StructTy = nullptr;
  Function *newFunction = constructFunctionDeclaration(
      inputs, outputs, EntryFreq, oldFunction->getName() + "." + SuffixToUse,
      StructValues, StructTy);
  SmallVector<Value *> NewValues;

  emitFunctionBody(inputs, outputs, StructValues, newFunction, StructTy, header,
                   SinkingCands, NewValues);

  std::vector<Value *> Reloads;
  CallInst *TheCall = emitReplacerCall(
      inputs, outputs, StructValues, newFunction, StructTy, oldFunction, ReplIP,
      EntryFreq, LifetimesStart.getArrayRef(), Reloads);

  insertReplacerCall(oldFunction, header, TheCall->getParent(), outputs,
                     Reloads, ExitWeights);

  fixupDebugInfoPostExtraction(*oldFunction, *newFunction, *TheCall, inputs,
                               NewValues);

  LLVM_DEBUG(llvm::dbgs() << "After extractCodeRegion - newFunction:\n");
  LLVM_DEBUG(newFunction->dump());
  LLVM_DEBUG(llvm::dbgs() << "After extractCodeRegion - oldFunction:\n");
  LLVM_DEBUG(oldFunction->dump());
  LLVM_DEBUG(if (AC && verifyAssumptionCache(*oldFunction, *newFunction, AC))
                 report_fatal_error("Stale Asumption cache for old Function!"));
  return newFunction;
}

void CodeExtractor::normalizeCFGForExtraction(BasicBlock *&header) {
  // If we have any return instructions in the region, split those blocks so
  // that the return is not in the region.
  splitReturnBlocks();

  // If we have to split PHI nodes of the entry or exit blocks, do so now.
  severSplitPHINodesOfEntry(header);

  // If a PHI in an exit block has multiple incoming values from the outlined
  // region, create a new PHI for those values within the region such that only
  // PHI itself becomes an output value, not each of its incoming values
  // individually.
  computeExtractedFuncRetVals();
  severSplitPHINodesOfExits();
}

void CodeExtractor::computeExtractedFuncRetVals() {
  ExtractedFuncRetVals.clear();

  SmallPtrSet<BasicBlock *, 2> ExitBlocks;
  for (BasicBlock *Block : Blocks) {
    for (BasicBlock *Succ : successors(Block)) {
      if (Blocks.count(Succ))
        continue;

      bool IsNew = ExitBlocks.insert(Succ).second;
      if (IsNew)
        ExtractedFuncRetVals.push_back(Succ);
    }
  }
}

Type *CodeExtractor::getSwitchType() {
  LLVMContext &Context = Blocks.front()->getContext();

  assert(ExtractedFuncRetVals.size() < 0xffff &&
         "too many exit blocks for switch");
  switch (ExtractedFuncRetVals.size()) {
  case 0:
  case 1:
    return Type::getVoidTy(Context);
  case 2:
    // Conditional branch, return a bool
    return Type::getInt1Ty(Context);
  default:
    return Type::getInt16Ty(Context);
  }
}

void CodeExtractor::emitFunctionBody(
    const ValueSet &inputs, const ValueSet &outputs,
    const ValueSet &StructValues, Function *newFunction,
    StructType *StructArgTy, BasicBlock *header, const ValueSet &SinkingCands,
    SmallVectorImpl<Value *> &NewValues) {
  Function *oldFunction = header->getParent();
  LLVMContext &Context = oldFunction->getContext();

  // The new function needs a root node because other nodes can branch to the
  // head of the region, but the entry node of a function cannot have preds.
  BasicBlock *newFuncRoot =
      BasicBlock::Create(Context, "newFuncRoot", newFunction);

  // Now sink all instructions which only have non-phi uses inside the region.
  // Group the allocas at the start of the block, so that any bitcast uses of
  // the allocas are well-defined.
  for (auto *II : SinkingCands) {
    if (!isa<AllocaInst>(II)) {
      cast<Instruction>(II)->moveBefore(*newFuncRoot,
                                        newFuncRoot->getFirstInsertionPt());
    }
  }
  for (auto *II : SinkingCands) {
    if (auto *AI = dyn_cast<AllocaInst>(II)) {
      AI->moveBefore(*newFuncRoot, newFuncRoot->getFirstInsertionPt());
    }
  }

  Function::arg_iterator ScalarAI = newFunction->arg_begin();
  Argument *AggArg = StructValues.empty()
                         ? nullptr
                         : newFunction->getArg(newFunction->arg_size() - 1);

  // Rewrite all users of the inputs in the extracted region to use the
  // arguments (or appropriate addressing into struct) instead.
  for (unsigned i = 0, e = inputs.size(), aggIdx = 0; i != e; ++i) {
    Value *RewriteVal;
    if (StructValues.contains(inputs[i])) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(header->getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(header->getContext()), aggIdx);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, AggArg, Idx, "gep_" + inputs[i]->getName(), newFuncRoot);
      LoadInst *LoadGEP =
          new LoadInst(StructArgTy->getElementType(aggIdx), GEP,
                       "loadgep_" + inputs[i]->getName(), newFuncRoot);
      // If we load pointer, we can add optional !align metadata
      // The existence of the !align metadata on the instruction tells
      // the optimizer that the value loaded is known to be aligned to
      // a boundary specified by the integer value in the metadata node.
      // Example:
      // %res = load ptr, ptr %input, align 8, !align !align_md_node
      //                                 ^         ^
      //                                 |         |
      //            alignment of %input address    |
      //                                           |
      //                                     alignment of %res object
      if (StructArgTy->getElementType(aggIdx)->isPointerTy()) {
        unsigned AlignmentValue;
        const Triple &TargetTriple =
            newFunction->getParent()->getTargetTriple();
        const DataLayout &DL = header->getDataLayout();
        // Pointers without casting can provide more information about
        // alignment. Use pointers without casts if given target preserves
        // alignment information for cast the operation.
        if (isAlignmentPreservedForAddrCast(TargetTriple))
          AlignmentValue =
              inputs[i]->stripPointerCasts()->getPointerAlignment(DL).value();
        else
          AlignmentValue = inputs[i]->getPointerAlignment(DL).value();
        MDBuilder MDB(header->getContext());
        LoadGEP->setMetadata(
            LLVMContext::MD_align,
            MDNode::get(
                header->getContext(),
                MDB.createConstant(ConstantInt::get(
                    Type::getInt64Ty(header->getContext()), AlignmentValue))));
      }
      RewriteVal = LoadGEP;
      ++aggIdx;
    } else
      RewriteVal = &*ScalarAI++;

    NewValues.push_back(RewriteVal);
  }

  moveCodeToFunction(newFunction);

  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    Value *RewriteVal = NewValues[i];

    std::vector<User *> Users(inputs[i]->user_begin(), inputs[i]->user_end());
    for (User *use : Users)
      if (Instruction *inst = dyn_cast<Instruction>(use))
        if (Blocks.count(inst->getParent()))
          inst->replaceUsesOfWith(inputs[i], RewriteVal);
  }

  // Since there may be multiple exits from the original region, make the new
  // function return an unsigned, switch on that number.  This loop iterates
  // over all of the blocks in the extracted region, updating any terminator
  // instructions in the to-be-extracted region that branch to blocks that are
  // not in the region to be extracted.
  std::map<BasicBlock *, BasicBlock *> ExitBlockMap;

  // Iterate over the previously collected targets, and create new blocks inside
  // the function to branch to.
  for (auto P : enumerate(ExtractedFuncRetVals)) {
    BasicBlock *OldTarget = P.value();
    size_t SuccNum = P.index();

    BasicBlock *NewTarget = BasicBlock::Create(
        Context, OldTarget->getName() + ".exitStub", newFunction);
    ExitBlockMap[OldTarget] = NewTarget;

    Value *brVal = nullptr;
    Type *RetTy = getSwitchType();
    assert(ExtractedFuncRetVals.size() < 0xffff &&
           "too many exit blocks for switch");
    switch (ExtractedFuncRetVals.size()) {
    case 0:
    case 1:
      // No value needed.
      break;
    case 2: // Conditional branch, return a bool
      brVal = ConstantInt::get(RetTy, !SuccNum);
      break;
    default:
      brVal = ConstantInt::get(RetTy, SuccNum);
      break;
    }

    ReturnInst::Create(Context, brVal, NewTarget);
  }

  for (BasicBlock *Block : Blocks) {
    Instruction *TI = Block->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
      if (Blocks.count(TI->getSuccessor(i)))
        continue;
      BasicBlock *OldTarget = TI->getSuccessor(i);
      // add a new basic block which returns the appropriate value
      BasicBlock *NewTarget = ExitBlockMap[OldTarget];
      assert(NewTarget && "Unknown target block!");

      // rewrite the original branch instruction with this new target
      TI->setSuccessor(i, NewTarget);
    }
  }

  // Loop over all of the PHI nodes in the header and exit blocks, and change
  // any references to the old incoming edge to be the new incoming edge.
  for (BasicBlock::iterator I = header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!Blocks.count(PN->getIncomingBlock(i)))
        PN->setIncomingBlock(i, newFuncRoot);
  }

  // Connect newFunction entry block to new header.
  BranchInst *BranchI = BranchInst::Create(header, newFuncRoot);
  applyFirstDebugLoc(oldFunction, Blocks.getArrayRef(), BranchI);

  // Store the arguments right after the definition of output value.
  // This should be proceeded after creating exit stubs to be ensure that invoke
  // result restore will be placed in the outlined function.
  ScalarAI = newFunction->arg_begin();
  unsigned AggIdx = 0;

  for (Value *Input : inputs) {
    if (StructValues.contains(Input))
      ++AggIdx;
    else
      ++ScalarAI;
  }

  for (Value *Output : outputs) {
    // Find proper insertion point.
    // In case Output is an invoke, we insert the store at the beginning in the
    // 'normal destination' BB. Otherwise we insert the store right after
    // Output.
    BasicBlock::iterator InsertPt;
    if (auto *InvokeI = dyn_cast<InvokeInst>(Output))
      InsertPt = InvokeI->getNormalDest()->getFirstInsertionPt();
    else if (auto *Phi = dyn_cast<PHINode>(Output))
      InsertPt = Phi->getParent()->getFirstInsertionPt();
    else if (auto *OutI = dyn_cast<Instruction>(Output))
      InsertPt = std::next(OutI->getIterator());
    else {
      // Globals don't need to be updated, just advance to the next argument.
      if (StructValues.contains(Output))
        ++AggIdx;
      else
        ++ScalarAI;
      continue;
    }

    assert((InsertPt->getFunction() == newFunction ||
            Blocks.count(InsertPt->getParent())) &&
           "InsertPt should be in new function");

    if (StructValues.contains(Output)) {
      assert(AggArg && "Number of aggregate output arguments should match "
                       "the number of defined values");
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), AggIdx);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, AggArg, Idx, "gep_" + Output->getName(), InsertPt);
      new StoreInst(Output, GEP, InsertPt);
      ++AggIdx;
    } else {
      assert(ScalarAI != newFunction->arg_end() &&
             "Number of scalar output arguments should match "
             "the number of defined values");
      new StoreInst(Output, &*ScalarAI, InsertPt);
      ++ScalarAI;
    }
  }

  if (ExtractedFuncRetVals.empty()) {
    // Mark the new function `noreturn` if applicable. Terminators which resume
    // exception propagation are treated as returning instructions. This is to
    // avoid inserting traps after calls to outlined functions which unwind.
    if (none_of(Blocks, [](const BasicBlock *BB) {
          const Instruction *Term = BB->getTerminator();
          return isa<ReturnInst>(Term) || isa<ResumeInst>(Term);
        }))
      newFunction->setDoesNotReturn();
  }
}

CallInst *CodeExtractor::emitReplacerCall(
    const ValueSet &inputs, const ValueSet &outputs,
    const ValueSet &StructValues, Function *newFunction,
    StructType *StructArgTy, Function *oldFunction, BasicBlock *ReplIP,
    BlockFrequency EntryFreq, ArrayRef<Value *> LifetimesStart,
    std::vector<Value *> &Reloads) {
  LLVMContext &Context = oldFunction->getContext();
  Module *M = oldFunction->getParent();
  const DataLayout &DL = M->getDataLayout();

  // This takes place of the original loop
  BasicBlock *codeReplacer =
      BasicBlock::Create(Context, "codeRepl", oldFunction, ReplIP);
  if (AllocationBlock)
    assert(AllocationBlock->getParent() == oldFunction &&
           "AllocationBlock is not in the same function");
  BasicBlock *AllocaBlock =
      AllocationBlock ? AllocationBlock : &oldFunction->getEntryBlock();

  // Update the entry count of the function.
  if (BFI)
    BFI->setBlockFreq(codeReplacer, EntryFreq);

  std::vector<Value *> params;

  // Add inputs as params, or to be filled into the struct
  for (Value *input : inputs) {
    if (StructValues.contains(input))
      continue;

    params.push_back(input);
  }

  // Create allocas for the outputs
  std::vector<Value *> ReloadOutputs;
  for (Value *output : outputs) {
    if (StructValues.contains(output))
      continue;

    AllocaInst *alloca = new AllocaInst(
        output->getType(), DL.getAllocaAddrSpace(), nullptr,
        output->getName() + ".loc", AllocaBlock->getFirstInsertionPt());
    params.push_back(alloca);
    ReloadOutputs.push_back(alloca);
  }

  AllocaInst *Struct = nullptr;
  if (!StructValues.empty()) {
    Struct = new AllocaInst(StructArgTy, DL.getAllocaAddrSpace(), nullptr,
                            "structArg", AllocaBlock->getFirstInsertionPt());
    if (ArgsInZeroAddressSpace && DL.getAllocaAddrSpace() != 0) {
      auto *StructSpaceCast = new AddrSpaceCastInst(
          Struct, PointerType ::get(Context, 0), "structArg.ascast");
      StructSpaceCast->insertAfter(Struct->getIterator());
      params.push_back(StructSpaceCast);
    } else {
      params.push_back(Struct);
    }

    unsigned AggIdx = 0;
    for (Value *input : inputs) {
      if (!StructValues.contains(input))
        continue;

      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), AggIdx);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, Struct, Idx, "gep_" + input->getName());
      GEP->insertInto(codeReplacer, codeReplacer->end());
      new StoreInst(input, GEP, codeReplacer);

      ++AggIdx;
    }
  }

  // Emit the call to the function
  CallInst *call = CallInst::Create(
      newFunction, params, ExtractedFuncRetVals.size() > 1 ? "targetBlock" : "",
      codeReplacer);

  // Set swifterror parameter attributes.
  unsigned ParamIdx = 0;
  unsigned AggIdx = 0;
  for (auto input : inputs) {
    if (StructValues.contains(input)) {
      ++AggIdx;
    } else {
      if (input->isSwiftError())
        call->addParamAttr(ParamIdx, Attribute::SwiftError);
      ++ParamIdx;
    }
  }

  // Add debug location to the new call, if the original function has debug
  // info. In that case, the terminator of the entry block of the extracted
  // function contains the first debug location of the extracted function,
  // set in extractCodeRegion.
  if (codeReplacer->getParent()->getSubprogram()) {
    if (auto DL = newFunction->getEntryBlock().getTerminator()->getDebugLoc())
      call->setDebugLoc(DL);
  }

  // Reload the outputs passed in by reference, use the struct if output is in
  // the aggregate or reload from the scalar argument.
  for (unsigned i = 0, e = outputs.size(), scalarIdx = 0; i != e; ++i) {
    Value *Output = nullptr;
    if (StructValues.contains(outputs[i])) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), AggIdx);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, Struct, Idx, "gep_reload_" + outputs[i]->getName());
      GEP->insertInto(codeReplacer, codeReplacer->end());
      Output = GEP;
      ++AggIdx;
    } else {
      Output = ReloadOutputs[scalarIdx];
      ++scalarIdx;
    }
    LoadInst *load =
        new LoadInst(outputs[i]->getType(), Output,
                     outputs[i]->getName() + ".reload", codeReplacer);
    Reloads.push_back(load);
  }

  // Now we can emit a switch statement using the call as a value.
  SwitchInst *TheSwitch =
      SwitchInst::Create(Constant::getNullValue(Type::getInt16Ty(Context)),
                         codeReplacer, 0, codeReplacer);
  for (auto P : enumerate(ExtractedFuncRetVals)) {
    BasicBlock *OldTarget = P.value();
    size_t SuccNum = P.index();

    TheSwitch->addCase(ConstantInt::get(Type::getInt16Ty(Context), SuccNum),
                       OldTarget);
  }

  // Now that we've done the deed, simplify the switch instruction.
  Type *OldFnRetTy = TheSwitch->getParent()->getParent()->getReturnType();
  switch (ExtractedFuncRetVals.size()) {
  case 0:
    // There are no successors (the block containing the switch itself), which
    // means that previously this was the last part of the function, and hence
    // this should be rewritten as a `ret` or `unreachable`.
    if (newFunction->doesNotReturn()) {
      // If fn is no return, end with an unreachable terminator.
      (void)new UnreachableInst(Context, TheSwitch->getIterator());
    } else if (OldFnRetTy->isVoidTy()) {
      // We have no return value.
      ReturnInst::Create(Context, nullptr,
                         TheSwitch->getIterator()); // Return void
    } else if (OldFnRetTy == TheSwitch->getCondition()->getType()) {
      // return what we have
      ReturnInst::Create(Context, TheSwitch->getCondition(),
                         TheSwitch->getIterator());
    } else {
      // Otherwise we must have code extracted an unwind or something, just
      // return whatever we want.
      ReturnInst::Create(Context, Constant::getNullValue(OldFnRetTy),
                         TheSwitch->getIterator());
    }

    TheSwitch->eraseFromParent();
    break;
  case 1:
    // Only a single destination, change the switch into an unconditional
    // branch.
    BranchInst::Create(TheSwitch->getSuccessor(1), TheSwitch->getIterator());
    TheSwitch->eraseFromParent();
    break;
  case 2:
    // Only two destinations, convert to a condition branch.
    // Remark: This also swaps the target branches:
    // 0 -> false -> getSuccessor(2); 1 -> true -> getSuccessor(1)
    BranchInst::Create(TheSwitch->getSuccessor(1), TheSwitch->getSuccessor(2),
                       call, TheSwitch->getIterator());
    TheSwitch->eraseFromParent();
    break;
  default:
    // Otherwise, make the default destination of the switch instruction be one
    // of the other successors.
    TheSwitch->setCondition(call);
    TheSwitch->setDefaultDest(
        TheSwitch->getSuccessor(ExtractedFuncRetVals.size()));
    // Remove redundant case
    TheSwitch->removeCase(
        SwitchInst::CaseIt(TheSwitch, ExtractedFuncRetVals.size() - 1));
    break;
  }

  // Insert lifetime markers around the reloads of any output values. The
  // allocas output values are stored in are only in-use in the codeRepl block.
  insertLifetimeMarkersSurroundingCall(M, ReloadOutputs, ReloadOutputs, call);

  // Replicate the effects of any lifetime start/end markers which referenced
  // input objects in the extraction region by placing markers around the call.
  insertLifetimeMarkersSurroundingCall(oldFunction->getParent(), LifetimesStart,
                                       {}, call);

  return call;
}

void CodeExtractor::insertReplacerCall(
    Function *oldFunction, BasicBlock *header, BasicBlock *codeReplacer,
    const ValueSet &outputs, ArrayRef<Value *> Reloads,
    const DenseMap<BasicBlock *, BlockFrequency> &ExitWeights) {

  // Rewrite branches to basic blocks outside of the loop to new dummy blocks
  // within the new function. This must be done before we lose track of which
  // blocks were originally in the code region.
  std::vector<User *> Users(header->user_begin(), header->user_end());
  for (auto &U : Users)
    // The BasicBlock which contains the branch is not in the region
    // modify the branch target to a new block
    if (Instruction *I = dyn_cast<Instruction>(U))
      if (I->isTerminator() && I->getFunction() == oldFunction &&
          !Blocks.count(I->getParent()))
        I->replaceUsesOfWith(header, codeReplacer);

  // When moving the code region it is sufficient to replace all uses to the
  // extracted function values. Since the original definition's block
  // dominated its use, it will also be dominated by codeReplacer's switch
  // which joined multiple exit blocks.
  for (BasicBlock *ExitBB : ExtractedFuncRetVals)
    for (PHINode &PN : ExitBB->phis()) {
      Value *IncomingCodeReplacerVal = nullptr;
      for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
        // Ignore incoming values from outside of the extracted region.
        if (!Blocks.count(PN.getIncomingBlock(i)))
          continue;

        // Ensure that there is only one incoming value from codeReplacer.
        if (!IncomingCodeReplacerVal) {
          PN.setIncomingBlock(i, codeReplacer);
          IncomingCodeReplacerVal = PN.getIncomingValue(i);
        } else
          assert(IncomingCodeReplacerVal == PN.getIncomingValue(i) &&
                 "PHI has two incompatbile incoming values from codeRepl");
      }
    }

  for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
    Value *load = Reloads[i];
    std::vector<User *> Users(outputs[i]->user_begin(), outputs[i]->user_end());
    for (User *U : Users) {
      Instruction *inst = cast<Instruction>(U);
      if (inst->getParent()->getParent() == oldFunction)
        inst->replaceUsesOfWith(outputs[i], load);
    }
  }

  // Update the branch weights for the exit block.
  if (BFI && ExtractedFuncRetVals.size() > 1)
    calculateNewCallTerminatorWeights(codeReplacer, ExitWeights, BPI);
}

bool CodeExtractor::verifyAssumptionCache(const Function &OldFunc,
                                          const Function &NewFunc,
                                          AssumptionCache *AC) {
  for (auto AssumeVH : AC->assumptions()) {
    auto *I = dyn_cast_or_null<CallInst>(AssumeVH);
    if (!I)
      continue;

    // There shouldn't be any llvm.assume intrinsics in the new function.
    if (I->getFunction() != &OldFunc)
      return true;

    // There shouldn't be any stale affected values in the assumption cache
    // that were previously in the old function, but that have now been moved
    // to the new function.
    for (auto AffectedValVH : AC->assumptionsFor(I->getOperand(0))) {
      auto *AffectedCI = dyn_cast_or_null<CallInst>(AffectedValVH);
      if (!AffectedCI)
        continue;
      if (AffectedCI->getFunction() != &OldFunc)
        return true;
      auto *AssumedInst = cast<Instruction>(AffectedCI->getOperand(0));
      if (AssumedInst->getFunction() != &OldFunc)
        return true;
    }
  }
  return false;
}

void CodeExtractor::excludeArgFromAggregate(Value *Arg) {
  ExcludeArgsFromAggregate.insert(Arg);
}
