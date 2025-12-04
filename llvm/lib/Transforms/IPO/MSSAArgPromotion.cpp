//===------ MSSAArgPromotionPass.cpp - Promote by-reference arguments -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass promotes function argument passed by reference:
//  1. Input argument: if the argument is read it is promoted to the argument
//     passed by value. Callers load the argument's value and pass it to the
//     function.
//  2. Output argument: if the argument is modified the function return type is
//     transformed into an aggregate and the final argument's value is returned
//     as a component of the return value. Callers store the returned value
//     using the original argument pointer.
//  3. Input/Output argument: the combination of the above.
//
//  int foo(int a, int *x) {
//    *x += 2;
//    return a;
//  }
//  int MemVar;
//  int X = foo(1, &MemVar);
//
//  into:
//
//  struct { int, int } foo (int a, int x) {
//    return { a, x + 2 };
//  }
//  int MemVar;
//  struct { int, int } S = foo(1, MemVar);
//  int X = S.first;
//  MemVar = S.second;
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/MSSAArgPromotion.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mssaargpromotion"

STATISTIC(NumInArgCandidates, "Number of input argument candidates found");
STATISTIC(NumInArgPromoted, "Number of of input argument promoted");
STATISTIC(NumInOutArgCandidates, "Number of in/out argument candidates found");
STATISTIC(NumInOutArgPromoted, "Number of of in/out argument promoted");

// When searching for a clobber for an argument we constrain the number of
// expensive uncached MSSA walks.
static cl::opt<unsigned> MaxMSSAWalksNum(
    "argpromo-mssa-walks-limit", cl::Hidden, cl::init(10000),
    cl::desc(
        "Function argument promotion pass: the maximum number of MSSA walks"
        " per argument on a clobber search (default = 1000)"));

// Return dot prefixed string twine if S isn't empty (used for BB's names).
static inline Twine dot(const StringRef &S) {
  return !S.empty() ? Twine('.') + S : Twine();
}

// Structure describing argument for promotion.
struct ArgPromotionInfo {
  Argument *Arg;
  Type *ArgType;
  Align ArgAlign;
  uint32_t Preload : 1; // Argument requires initial value to be passed to
                        // the function.
  uint32_t Return : 1;  // Argument should be returned by the function.

  // When the argument is promoted we need a new argument for the incoming
  // preloaded value but the new function signature isn't known yet and
  // therefore isn't created. We use a dummy argument to start with and
  // after the new function is created its RAUWed with the function's
  // argument, see createNewFunction.
  std::unique_ptr<Argument> PreloadArgDummy;

  // Index of the value in the aggregated return type (insert/extract_value idx)
  unsigned ReturnValueIndex = (unsigned)-1;

  // If one candidate clobbers another this field denotes the relationship.
  // Used to find "declobbering" promotion sequence.
  ArgPromotionInfo *ClobberedBy = nullptr;

  AAMDNodes AAMD; // Merged AA metadata for the load/store.

  ArgPromotionInfo(Argument *Arg_ = nullptr, Type *ArgType_ = nullptr,
                   Align ArgAlign_ = Align())
      : Arg(Arg_), ArgType(ArgType_), ArgAlign(ArgAlign_) {
    Preload = Return = 0;
  }

  unsigned getArgNo() const { return Arg->getArgNo(); }

  bool isUnusedArg() const { return !Preload && !Return; }

  // Return true if this argument is promoted.
  bool isPromoted() const {
    return PreloadArgDummy || ReturnValueIndex != (unsigned)-1;
  }

  // TODO: this is a placeholder for checking GEP indexes
  bool isMyPtr(Value *Ptr) const { return Ptr && Ptr == Arg; }

  // Predicates returning true if the value is a load or store by this
  // argument (TODO: this will check GEPs later).
  bool isMyLoad(Value *V) const {
    LoadInst *LI = dyn_cast<LoadInst>(V);
    return LI ? isMyPtr(LI->getPointerOperand()) : false;
  }
  bool isMyStore(Value *V) const {
    StoreInst *SI = dyn_cast<StoreInst>(V);
    return SI ? isMyPtr(SI->getPointerOperand()) : false;
  }
  bool isMyLoadOrStore(Value *V) const {
    if (LoadInst *LI = dyn_cast<LoadInst>(V))
      return isMyPtr(LI->getPointerOperand());
    if (StoreInst *SI = dyn_cast<StoreInst>(V))
      return isMyPtr(SI->getPointerOperand());
    return false;
  }

  MemoryLocation getMemLoc() const {
    const auto &DL = Arg->getParent()->getParent()->getDataLayout();
    return MemoryLocation(Arg,
                          LocationSize::precise(DL.getTypeStoreSize(ArgType)));
  }

  bool isClobberedBy(const ArgPromotionInfo &A) const {
    const ArgPromotionInfo *P = this;
    while ((P = P->ClobberedBy)) {
      if (&A == P)
        return true;
    }
    return false;
  }

  Twine getParamName(StringRef &&LifeTimeOwner = StringRef()) const {
    // The problem with a twine is that StringRef it references should be alive
    // when the twine is alive: use LifeTimeOwner to keep the StringRef alive
    // at least for the lifetime of the full expression.
    LifeTimeOwner = Arg->getName();
    return LifeTimeOwner + ".0.val";
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD StringRef getKindStr() const {
    if (Preload && Return)
      return "inout";
    return Preload ? "in" : "out";
  }
#endif

  Argument *getOrCreatePreloadArgDummy() {
    if (!PreloadArgDummy)
      PreloadArgDummy = std::make_unique<Argument>(ArgType);
    return PreloadArgDummy.get();
  }

  LoadInst *createLoad(IRBuilder<NoFolder> &IRB, Value *Ptr,
                       const StringRef &Name) const {
    LoadInst *Load =
        IRB.CreateAlignedLoad(ArgType, Ptr, ArgAlign, Name + ".val");
    if (AAMD)
      Load->setAAMetadata(AAMD);
    return Load;
  }

  StoreInst *createStore(IRBuilder<NoFolder> &IRB, Value *V, Value *Ptr) const {
    StoreInst *Store = IRB.CreateAlignedStore(V, Ptr, ArgAlign);
    if (AAMD)
      Store->setAAMetadata(AAMD);
    return Store;
  }

  // Iterator to hide impl details on iterating promoted argument's users,
  // espesially when GEPs added, by now - minimal trivial implementation.
  class user_iterator
      : public iterator_facade_base<user_iterator, std::forward_iterator_tag,
                                    Value *> {
    Argument::user_iterator ArgUserI;
    friend struct ArgPromotionInfo;
    user_iterator(const Argument::user_iterator &I) : ArgUserI(I) {}

  public:
    value_type operator*() const { return *ArgUserI; }
    user_iterator &operator++() {
      ++ArgUserI;
      return *this;
    }
    user_iterator operator++(int) {
      auto R = *this;
      ++ArgUserI;
      return R;
    }
    bool operator==(const user_iterator &RHS) const {
      return ArgUserI == RHS.ArgUserI;
    }
  };
  user_iterator user_begin() const { return user_iterator(Arg->user_begin()); }
  user_iterator user_end() const { return user_iterator(Arg->user_end()); }
  iterator_range<user_iterator> users() const {
    return make_range(user_begin(), user_end());
  }
};

// Return true if Pred is true for all callers passing P.Arg.
static bool allCallersPass(
    const ArgPromotionInfo &P,
    function_ref<bool(CallBase *, Value *, const ArgPromotionInfo &)> Pred) {
  Function *Callee = P.Arg->getParent();
  for (User *U : Callee->users()) {
    assert(isa<CallBase>(U));
    CallBase *CB = cast<CallBase>(U);
    if (!Pred(CB, CB->getArgOperand(P.getArgNo()), P))
      return false;
  }
  return true;
}

// Given the function pointer argument that is only used by loads
// return true if the value pointed by the argument can be loaded before the
// function call and passed in:
//   either the value is loaded by the ptr arg on every function path
//   or the pointer is valid for all callsites in the program.
static bool isROCandidate(ArgPromotionInfo &Candidate) {
  SmallPtrSet<BasicBlock *, 16> ReadPerBB;
  for (Value *U : Candidate.users()) {
    assert(Candidate.isMyLoad(U));
    ReadPerBB.insert(cast<Instruction>(U)->getParent());
  }
  bool HasLoadOnEveryPath = true;
  Function *F = Candidate.Arg->getParent();
  auto *EntryBB = &F->getEntryBlock();
  for (auto DFI = df_begin(EntryBB), E = df_end(EntryBB); DFI != E;) {
    BasicBlock *BB = *DFI;
    if (ReadPerBB.count(BB)) {
      DFI.skipChildren(); // This path already have load - skipping children.
      continue;
    }
    if (isa<ReturnInst>(BB->getTerminator())) {
      HasLoadOnEveryPath = false;
      break;
    }
    ++DFI;
  }

  // Return true if we can prove that caller pass in a valid pointer.
  const DataLayout &DL = F->getParent()->getDataLayout();
  auto IsValidPtr = [&DL](Value *A, const ArgPromotionInfo &P) -> bool {
    return isDereferenceableAndAlignedPointer(A, P.ArgType, P.ArgAlign, DL);
  };

  Candidate.Preload =
      HasLoadOnEveryPath ||
      // Check if the argument itself is marked dereferenceable and aligned,
      IsValidPtr(Candidate.Arg, Candidate) ||
      // or this is true for all the callers
      allCallersPass(Candidate, [&IsValidPtr](CallBase *, Value *A,
                                              const ArgPromotionInfo &P) {
        return IsValidPtr(A, P);
      });

  LLVM_DEBUG(dbgs() << " - ";
             if (HasLoadOnEveryPath) dbgs() << "has a load on every path,";
             else dbgs()
             << (Candidate.Preload ? "" : "not")
             << " all callers pass a valid and aligned dereferenceable ptr,");

  return Candidate.Preload;
}

// Given the function pointer argument that is only used by stores and maybe
// loads return true if the value pointed by the argument can be stored after
// and loaded before the function call and passed in/returned by the function:
//   either the value is stored on every function path
//   or the pointer points to a thread local memory that doesn't escape before
//   the function call for every callsite in the program.
// Check is made if a load precedes stores on any path so the initial value
// should be passed in as a parameter.
static bool isRWCandidate(FunctionAnalysisManager &FAM,
                          ArgPromotionInfo &Candidate) {
  SmallDenseMap<BasicBlock *, unsigned, 16> RWPerBB;
  enum { HasReads = 1, HasWrites = 2 };
  for (Value *U : Candidate.users()) {
    assert(Candidate.isMyLoadOrStore(U));
    RWPerBB[cast<Instruction>(U)->getParent()] |=
        isa<LoadInst>(U) ? HasReads : HasWrites;
  }
  bool HasLoadBeforeStore = false;
  bool HasStoreOnEveryPath = true;
  Function *F = Candidate.Arg->getParent();
  auto *EntryBB = &F->getEntryBlock();
  for (auto DFI = df_begin(EntryBB), E = df_end(EntryBB); DFI != E;) {
    BasicBlock *BB = *DFI;
    auto RW = RWPerBB.find(BB);
    if (RW != RWPerBB.end()) { // There is load or store within the BB.
      if (!HasLoadBeforeStore && (RW->second & HasReads)) {
        if (RW->second & HasWrites) {
          // Determine if load locally dominates store.
          auto LorS = find_if(*BB, [&Candidate](Instruction &I) -> bool {
            return Candidate.isMyLoadOrStore(&I);
          });
          assert(LorS != BB->end());
          HasLoadBeforeStore = isa<LoadInst>(*LorS);
        } else
          HasLoadBeforeStore = true;
      }
      if (RW->second & HasWrites) {
        DFI.skipChildren(); // This path already have store - skipping children.
        continue;
      }
    }
    if (isa<ReturnInst>(BB->getTerminator()))
      HasStoreOnEveryPath = false;

    // Short-circuit: all the info is collected - nothing left to do.
    if (HasLoadBeforeStore && !HasStoreOnEveryPath)
      break;
    ++DFI;
  }

  auto ValidThreadLocalPtr = [&FAM, F](CallBase *CallInst, Value *ActualPtr,
                                       const ArgPromotionInfo &P) -> bool {
    Value *Object = getUnderlyingObject(ActualPtr);
    if (!isa<AllocaInst>(Object) &&
        !isAllocLikeFn(Object, &FAM.getResult<TargetLibraryAnalysis>(*F)))
      return false;

    // Get the dominator tree for the caller function (where CallInst is located)
    Function *Caller = CallInst->getParent()->getParent();
    return !PointerMayBeCapturedBefore(
        Object, /* ReturnCaptures */ false, CallInst,
        &FAM.getResult<DominatorTreeAnalysis>(*Caller),
        /* IncludeI */ false);  // Don't include the call itself
  };

  if (HasStoreOnEveryPath) {
    Candidate.Preload = HasLoadBeforeStore;
    Candidate.Return = true;
    LLVM_DEBUG(dbgs() << " - has store on every path,");
  } else {
    // Preload the value so it can be returned unchanged on some path.
    Candidate.Preload = Candidate.Return =
        allCallersPass(Candidate, ValidThreadLocalPtr);
    LLVM_DEBUG(dbgs() << " - " << (Candidate.Return ? "" : "not")
                      << " all callers pass a valid thread local ptr,");
  }
  return Candidate.Return;
}

// Fill Candidates with the list of arguments potentially suitable for promotion
static bool
getPromotionCandidates(FunctionAnalysisManager &FAM, Argument *PtrArg,
                       SmallVectorImpl<ArgPromotionInfo> &Candidates,
                       bool InArgsOnly) {
  LLVM_DEBUG(dbgs() << "  Trying arg: " << *PtrArg);

  unsigned NumLoads = 0, NumStores = 0;
  Type *ValueTy = nullptr;
  Align ArgAlign; // Receives max alignment among the instructions.
  for (auto *U : PtrArg->users()) {
    Type *InstType = nullptr;
    Align InstAlign;
    if (auto *LI = dyn_cast<LoadInst>(U)) {
      if (LI->isSimple()) {
        InstType = LI->getType();
        InstAlign = LI->getAlign();
        ++NumLoads;
      }
    } else if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->isSimple() && SI->getValueOperand() != PtrArg && !InArgsOnly) {
        InstType = SI->getValueOperand()->getType();
        InstAlign = SI->getAlign();
        ++NumStores;
      }
    }
    if (!InstType) {
      LLVM_DEBUG(dbgs() << " - unsupported use " << *U << '\n');
      return false;
    }
    if (!ValueTy) {
      if (!InstType->isSingleValueType()) {
        LLVM_DEBUG(dbgs() << " - unsupported type " << *InstType << '\n');
        return false;
      }
      ValueTy = InstType;
    } else if (ValueTy != InstType) {
      LLVM_DEBUG(dbgs() << " - loads/stores don't agree on the type " << *U
                        << '\n');
      return false;
    }
    if (NumStores && PtrArg->hasByValAttr()) { // Skip mutable byval.
      LLVM_DEBUG(dbgs() << " - byval has store " << *U << '\n');
      return false;
    }
    ArgAlign = std::max(ArgAlign, InstAlign);
  }

  // Check if the parameter has an explicit alignment attribute that's
  // insufficient for the loads/stores. If the parameter has an explicit
  // alignment guarantee that's less than what we need, we cannot promote.
  if (auto ParamAlign = PtrArg->getParamAlign()) {
    if (ParamAlign.value() < ArgAlign) {
      LLVM_DEBUG(dbgs() << " - insufficient alignment guarantee (param has "
                        << ParamAlign.value().value() << ", need "
                        << ArgAlign.value() << ")\n");
      return false;
    }
  }

  Candidates.emplace_back(PtrArg, ValueTy, ArgAlign);
  if (NumLoads + NumStores) {
    auto &C = Candidates.back();
    if (!(NumStores ? isRWCandidate(FAM, C) : isROCandidate(C))) {
      Candidates.pop_back();
      LLVM_DEBUG(dbgs() << " discard\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << " promote as " << C.getKindStr() << " arg\n");
  } else {
    // Otherwise - useless argument - to get rid off later.
    LLVM_DEBUG(dbgs() << " - unused arg, remove\n");
  }
  return true;
}

class ArgumentPromoter {
  Function *F;
  FunctionAnalysisManager &FAM;
  MemorySSA &MSSA;
  unsigned NumMSSAWalksLeft;
  SmallPtrSet<MemoryAccess *, 16> VisitedMA;

  enum ClobberTestResult {
    CheckOtherPhiPath,
    ContinueThisPhiPath,
    FoundClobber
  };
  using ClobberTestFx = enum ClobberTestResult(MemoryAccess *);

  MemoryAccess *getClobber(MemoryAccess *MA, const MemoryLocation &Loc,
                           function_ref<ClobberTestFx> ClobberTest,
                           SmallPtrSetImpl<MemoryAccess *> &Visited);

  MemoryAccess *getClobber(Instruction *I,
                           function_ref<ClobberTestFx> ClobberTest,
                           SmallPtrSetImpl<MemoryAccess *> &Visited);

  MemoryAccess *getInOutArgClobber(const ArgPromotionInfo &ArgInfo);

  using RetValuesMap =
      SmallDenseMap<ReturnInst *, SmallVector<TrackingVH<Value>, 4>>;
  void promoteInOutArg(ArgPromotionInfo &ArgInfo, RetValuesMap &RetValues);

  Type *promoteInOutCandidates(
      SmallVectorImpl<ArgPromotionInfo> &Candidates,
      SmallVectorImpl<ArgPromotionInfo *> &RetValuesStoreOrder);

  bool isInArgClobbered(const ArgPromotionInfo &ArgInfo);
  void promoteInArg(ArgPromotionInfo &ArgInfo);

  static Function *
  createNewFunction(Function *OldF, Type *RetTy,
                    const SmallVectorImpl<ArgPromotionInfo *> &PromotedArgs);

  static void promoteCallsite(
      CallBase &CB, Function *NF,
      const SmallVectorImpl<ArgPromotionInfo *> &PromotedArgs,
      const SmallVectorImpl<ArgPromotionInfo *> &RetValuesStoreOrder);

public:
  ArgumentPromoter(Function *F_, FunctionAnalysisManager &FAM_)
      : F(F_), FAM(FAM_), MSSA(FAM.getResult<MemorySSAAnalysis>(*F).getMSSA()) {
  }

  Function *run(SmallVectorImpl<ArgPromotionInfo> &Candidates);
};

// Search memory access that clobbers Loc starting from MA. Does a BFS search
// on phi paths. ClobberTest is run over every found clobber to negotiate it
// further by the ClobberTest's return value:
//   FoundClobber - stop search and return found clobber;
//   ContinueThisPhiPath - skip found clobber and continue searching the path;
//   CheckOtherPhiPath - skip found clobber and try other phi paths if any.
// Return found clobber, LiveOnEntryDef if no clobber or nullptr if the maximum
// number of uncached MSSA walks reached.
MemoryAccess *
ArgumentPromoter::getClobber(MemoryAccess *MA, const MemoryLocation &Loc,
                             function_ref<ClobberTestFx> ClobberTest,
                             SmallPtrSetImpl<MemoryAccess *> &Visited) {
  std::deque<MemoryAccess *> FIFO;
  do {
    while (true) {
      if (!Visited.insert(MA).second)
        break;
      if (MemoryPhi *Phi = dyn_cast<MemoryPhi>(MA)) {
        for (auto *DefMA : make_range(Phi->defs_begin(), Phi->defs_end()))
          FIFO.push_back(DefMA);
        break;
      }
      if (--NumMSSAWalksLeft == 0) // Constrain the number of uncached walks.
        return nullptr;
      auto *ClobberMA = MSSA.getWalker()->getClobberingMemoryAccess(MA, Loc);
      if (isa<MemoryPhi>(ClobberMA)) {
        MA = ClobberMA;
      } else if (!MSSA.isLiveOnEntryDef(ClobberMA)) {
        ClobberTestResult R = ClobberTest(ClobberMA);
        if (R == FoundClobber)
          return ClobberMA;
        else if (R == ContinueThisPhiPath)
          MA = cast<MemoryUseOrDef>(ClobberMA)->getDefiningAccess();
        else
          break; // CheckOtherPhiPath
      }
    }
    if (FIFO.empty())
      break;
    MA = FIFO.front();
    FIFO.pop_front();
  } while (true);
  return MSSA.getLiveOnEntryDef();
}

// Similar the previous routine but searches memory access that clobbers
// memory accessed by the I instruction.
MemoryAccess *
ArgumentPromoter::getClobber(Instruction *I,
                             function_ref<ClobberTestFx> ClobberTest,
                             SmallPtrSetImpl<MemoryAccess *> &Visited) {
  assert(MemoryLocation::getOrNone(I).has_value());
  auto *ClobberMA = MSSA.getWalker()->getClobberingMemoryAccess(I);
  if (MSSA.isLiveOnEntryDef(ClobberMA))
    return ClobberMA;
  if (isa<MemoryPhi>(ClobberMA))
    return getClobber(ClobberMA, MemoryLocation::get(I), ClobberTest, Visited);

  switch (ClobberTest(ClobberMA)) {
  case FoundClobber:
    return ClobberMA;
  case CheckOtherPhiPath:
    break; // No other path to test.
  case ContinueThisPhiPath:
    return getClobber(cast<MemoryUseOrDef>(ClobberMA)->getDefiningAccess(),
                      MemoryLocation::get(I), ClobberTest, Visited);
  }
  return MSSA.getLiveOnEntryDef();
}

// TODO: move this to the MemorySSA class
// Find last memory def or phi in the BB or in its dominating predecessors.
// Note that a def in non-dominating predecessor would create phi in the BB.
static MemoryAccess *getLastDef(BasicBlock *BB, MemorySSA &MSSA) {
  if (auto *Defs = MSSA.getBlockDefs(BB))
    return const_cast<MemoryAccess *>(&*Defs->rbegin());

  DomTreeNode *Node = MSSA.getDomTree().getNode(BB);
  while ((Node = Node->getIDom()))
    if (auto *Defs = MSSA.getBlockDefs(Node->getBlock()))
      return const_cast<MemoryAccess *>(&*Defs->rbegin());
  return MSSA.getLiveOnEntryDef();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
static void printClobber(raw_ostream &os, MemoryAccess *ClobberMA,
                         Instruction *I) {
  if (!ClobberMA) {
    os << "clobber search reached limit\n";
    return;
  }
  auto *ClobberI = cast<MemoryUseOrDef>(ClobberMA)->getMemoryInst();
  os << "found clobber:" << *I << '@' << I->getParent()->getName()
     << " is clobbered by" << *ClobberI << '@'
     << ClobberI->getParent()->getName() << '\n';
}
#endif

// Check if memops by the argument are clobbered by or clobber other memops.
// Return found clobber, LiveOnEntryDef if no clobber or nullptr if the maximum
// number of uncached MSSA walks reached.
MemoryAccess *
ArgumentPromoter::getInOutArgClobber(const ArgPromotionInfo &ArgInfo) {
  LLVM_DEBUG(dbgs() << "  Searching for a clobber for " << ArgInfo.getKindStr()
                    << " arg " << *ArgInfo.Arg << ": ");
  auto SkipMyStore = [&ArgInfo](MemoryAccess *MA) -> ClobberTestResult {
    return ArgInfo.isMyStore(cast<MemoryUseOrDef>(MA)->getMemoryInst())
               ? CheckOtherPhiPath
               : FoundClobber;
  };
  VisitedMA.clear(); // Using VisitedMA to track SkipMyStore condition tests.
  // Check if a load by the argument is clobbered by something else than
  // a store by the argument.
  for (Value *U : ArgInfo.users()) {
    assert(ArgInfo.isMyLoadOrStore(U));
    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      auto *Clob = getClobber(LI, SkipMyStore, VisitedMA);
      if (!MSSA.isLiveOnEntryDef(Clob)) {
        LLVM_DEBUG(printClobber(dbgs(), Clob, LI));
        return Clob;
      }
    }
  }
  // Check if the argument has been clobbered between last store by the arg
  // and return on any path.
  MemoryLocation Loc(ArgInfo.getMemLoc());
  for (auto &BB : *F) {
    if (!isa<ReturnInst>(BB.getTerminator()))
      continue;
    auto *Clob = getClobber(getLastDef(&BB, MSSA), Loc, SkipMyStore, VisitedMA);
    if (!MSSA.isLiveOnEntryDef(Clob)) {
      LLVM_DEBUG(printClobber(dbgs(), Clob, BB.getTerminator()));
      return Clob;
    }
  }
  // Check if any other load is clobbered by a store by the argument.
  AliasAnalysis &AA = FAM.getResult<AAManager>(*F);
  for (auto &BB : *F) {
    if (auto *L = MSSA.getBlockAccesses(&BB)) {
      for (auto &MA : *L) {
        if (auto *MU = dyn_cast<MemoryUse>(&MA)) {
          Instruction *UseI = MU->getMemoryInst();
          if (ArgInfo.isMyLoad(UseI))
            continue;
          auto UseLoc = MemoryLocation::getOrNone(UseI);
          if (!UseLoc.has_value()) {
            LLVM_DEBUG(dbgs() << "cannot get memloc for " << *UseI << '\n');
            // Conservatively consider this as a clobber.
            return const_cast<MemoryUse *>(MU);
          }
          if (AA.pointsToConstantMemory(UseLoc.value()))
            continue;
          auto FindMyStore = [&](MemoryAccess *MA) -> ClobberTestResult {
            Instruction *DefI = cast<MemoryUseOrDef>(MA)->getMemoryInst();
            if (ArgInfo.isMyStore(DefI))
              return FoundClobber;
            // If the UseI's location is definitely overwritten with the clober
            // we can skip this path, otherwise it can be clobbered earlier.
            auto DefLoc = MemoryLocation::getOrNone(DefI);
            if (DefLoc.has_value() &&
                AA.isMustAlias(DefLoc.value(), UseLoc.value()))
              return CheckOtherPhiPath;

            return ContinueThisPhiPath;
          };
          VisitedMA.clear();
          auto *Clob = getClobber(UseI, FindMyStore, VisitedMA);
          if (!MSSA.isLiveOnEntryDef(Clob)) {
            LLVM_DEBUG(printClobber(dbgs(), Clob, UseI));
            return Clob;
          }
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "no clobber\n");
  return MSSA.getLiveOnEntryDef();
}

// Annotate each return with a value for the argument ArgInfo.
// Create Phis and rewrites code.
void ArgumentPromoter::promoteInOutArg(ArgPromotionInfo &ArgInfo,
                                       RetValuesMap &RetValues) {
  SmallDenseMap<BasicBlock *, SmallVector<Instruction *, 4>, 16> MemInsts;
  SmallPtrSet<BasicBlock *, 4> DefBB;
  for (Value *U : ArgInfo.users()) {
    assert(ArgInfo.isMyLoadOrStore(U));

    Instruction *I = cast<Instruction>(U);
    if (MemInsts.empty())
      ArgInfo.AAMD = I->getAAMetadata();
    else if (ArgInfo.AAMD) // Merging AA metadata BTW.
      ArgInfo.AAMD.merge(I->getAAMetadata());

    BasicBlock *BB = I->getParent();
    if (isa<StoreInst>(I))
      DefBB.insert(BB);
    MemInsts[BB].push_back(I);
  }

  SmallDenseMap<BasicBlock *, TrackingVH<Value>, 16> BBExitValue;

  // Processing stores.
  for (BasicBlock *BB : DefBB) {
    auto &BBMemInsts = MemInsts[BB];
    // Sort mem instructions in the program order.
    sort(BBMemInsts, [this](Instruction *A, Instruction *B) {
      return MSSA.locallyDominates(MSSA.getMemoryAccess(A),
                                   MSSA.getMemoryAccess(B));
    });
    // Propagate store values down to the end of the basic block,
    // loads preceding the first store will be processed later.
    auto FirstStore =
        find_if(BBMemInsts, [](Instruction *I) { return isa<StoreInst>(I); });
    assert(FirstStore != BBMemInsts.end());
    Value *V = nullptr;
    for (Instruction *I : make_range(FirstStore, BBMemInsts.end())) {
      if (isa<LoadInst>(I)) {
        assert(V); // Since we started with a store.
        I->replaceAllUsesWith(V);
      } else
        V = cast<StoreInst>(I)->getValueOperand();
    }
    assert(V);
    BBExitValue[BB] = V;
  }

  SmallDenseMap<BasicBlock *, TrackingVH<Value>, 16> BBEntryValue;
  auto setEntryValue = [&](BasicBlock *BB, Value *V) {
    BBEntryValue[BB] = V;
    // Keep BBExitValue left from store processing.
    BBExitValue.try_emplace(BB, V);
  };

  if (ArgInfo.Preload)
    setEntryValue(&F->getEntryBlock(), ArgInfo.getOrCreatePreloadArgDummy());

  { // Inserting phis.
    SmallVector<BasicBlock *, 16> PHIBlocks;
    ForwardIDFCalculator IDF(MSSA.getDomTree());
    IDF.setDefiningBlocks(DefBB);
    IDF.calculate(PHIBlocks);

    for (auto *JoinBB : PHIBlocks) {
      auto P = MemInsts.find(JoinBB);
      // If JoinBB starts with a store then phi value isn't used.
      if (P == MemInsts.end() || isa<LoadInst>(P->second.front())) {
        PHINode *Phi = PHINode::Create(ArgInfo.ArgType, 2,
                                       ArgInfo.getParamName() +
                                           dot(JoinBB->getName()) + ".phi",
                                       JoinBB->begin());
        setEntryValue(JoinBB, Phi);
      }
    }
  }

  auto findIncomingValue = [&](BasicBlock *BB) -> Value * {
    DomTreeNode *Node = MSSA.getDomTree().getNode(BB);
    while ((Node = Node->getIDom())) {
      auto I = BBExitValue.find(Node->getBlock());
      if (I != BBExitValue.end())
        return I->second;
    }
    return UndefValue::get(ArgInfo.ArgType);
  };

  auto getBBExitValue = [&](BasicBlock *BB) -> Value * {
    auto I = BBExitValue.find(BB);
    if (I != BBExitValue.end())
      return I->second;
    return findIncomingValue(BB);
  };

  auto getBBEntryValue = [&](BasicBlock *BB) -> Value * {
    auto I = BBEntryValue.find(BB);
    if (I != BBEntryValue.end())
      return I->second;
    return findIncomingValue(BB);
  };

  // Processing phis.
  const DataLayout &DL = F->getParent()->getDataLayout();
  for (auto &P : BBEntryValue)
    if (PHINode *Phi = dyn_cast<PHINode>(&*P.second)) {
      for (BasicBlock *PredBB : predecessors(P.first))
        Phi->addIncoming(getBBExitValue(PredBB), PredBB);

      if (Value *V = simplifyInstruction(Phi, DL)) {
        Phi->replaceAllUsesWith(V);
        Phi->eraseFromParent();
      }
    }

  // Processing loads.
  for (auto &P : MemInsts) {
    auto &BBMemInsts = P.second;
    if (!isa<LoadInst>(BBMemInsts.front()))
      continue;
    Value *V = getBBEntryValue(P.first);
    auto I = BBMemInsts.begin(), E = BBMemInsts.end();
    do {
      (*I)->replaceAllUsesWith(V);
    } while (++I != E && isa<LoadInst>(*I));
  }

  // Annotate returns.
  for (BasicBlock &BB : *F)
    if (auto *RetInst = dyn_cast<ReturnInst>(BB.getTerminator()))
      RetValues[RetInst].push_back(getBBExitValue(&BB));

  // Finally erase load/stores.
  MemorySSAUpdater UMSSA(&MSSA);
  for (Value *U : make_early_inc_range(ArgInfo.users())) {
    assert(ArgInfo.isMyLoadOrStore(U));
    UMSSA.removeMemoryAccess(cast<Instruction>(U));
    cast<Instruction>(U)->eraseFromParent();
  }
#ifndef NDEBUG
  MSSA.verifyMemorySSA();
#endif
}

// Tries to promote [input/]output ptr arguments. It may happen that store
// instructions for several arguments clobber one another, to solve this
// an attempt to find an "unclobbering" promotion sequence is made.
// For example:
//     store PtrArgA(may alias), 1;
//     store PtrArgB(may alias), 0; <- clobbers store PtrArgA
//
// First PtrArgB is promoted unclobbering PtrArgA which is promoted second.
// Notice that it is only possible if such stores obey the same order in every
// basic block, otherwise we cannot unclobber these at all. Promoted stores are
// then placed in the caller in the same order making the transformation safe.
//
// This could be left for the following passes but it's better to perform such
// unclobbering all at once not only because of compilation speed but it also
// allows to simplify the return value of the function: otherwise we would have
// to deal with an onion-like aggregated return type with a bulky INSERT_VALUE/
// EXTRACT_VALUE sequence.
Type *ArgumentPromoter::promoteInOutCandidates(
    SmallVectorImpl<ArgPromotionInfo> &Candidates,
    SmallVectorImpl<ArgPromotionInfo *> &RetValuesStoreOrder) {

  // Priority queue is ordered so that clobbered candidates pop last.
  struct ClobberedPopLast {
    // Returns true if its first argument comes before its second argument in a
    // weak ordering. But because the priority queue outputs largest elements
    // first, the elements that "come before" are actually output last.
    bool operator()(const ArgPromotionInfo *A1,
                    const ArgPromotionInfo *A2) const {
      assert(!A1->isClobberedBy(*A2) || !A2->isClobberedBy(*A1));
      return A1->isClobberedBy(*A2);
    }
  };
  struct CandidateQueue
      : std::priority_queue<ArgPromotionInfo *,
                            SmallVector<ArgPromotionInfo *, 4>,
                            ClobberedPopLast> {
    CandidateQueue(SmallVectorImpl<ArgPromotionInfo> &Candidates) {
      // This might seem as a dirty hack but until ClobberedBy is set no order
      // on candidates can be established, so just store them as is
      for (auto &C : Candidates) {
        if (C.Return) {
          assert(!C.ClobberedBy); // but let's be carefull.
          c.push_back(&C);
        }
      }
    }
    // This is placed here because priority_queue container is protected.
    ArgPromotionInfo *findClobber(StoreInst *SI) const {
      auto Clobber =
          std::find_if(c.begin(), c.end(), [SI](const ArgPromotionInfo *A) {
            return A->isMyStore(SI);
          });
      return Clobber != c.end() ? *Clobber : nullptr;
    }
  } Queue(Candidates);

  RetValuesMap RetValues;
  unsigned NumPromoted = 0;
  while (!Queue.empty()) {
    ArgPromotionInfo &C = *Queue.top();
    Queue.pop();
    if (C.ClobberedBy && !C.ClobberedBy->isPromoted()) // [1]
      continue;                                        // the clobber isn't gone
    MemoryAccess *ClobberMA = getInOutArgClobber(C);
    if (MSSA.isLiveOnEntryDef(ClobberMA)) {
      promoteInOutArg(C, RetValues);
      // ReturnValueIndex is used as the index of the arg's value in the map
      // up until this function's exit, see below.
      C.ReturnValueIndex = NumPromoted++;
      continue;
    }
    MemoryDef *MDef = dyn_cast_or_null<MemoryDef>(ClobberMA);
    if (!MDef)
      continue;
    // If the clobbering store belongs to another candidate in the queue
    // enqueue the current candidate back with the ClobberedBy set so we can
    // retry it after the clobbering candidate has been promoted.
    StoreInst *SI = dyn_cast<StoreInst>(MDef->getMemoryInst());
    if (!SI || !SI->isSimple() || C.isMyStore(SI))
      continue;
    if (ArgPromotionInfo *Clobber = Queue.findClobber(SI)) {
      C.ClobberedBy = Clobber;
      if (!Clobber->isClobberedBy(C))
        Queue.push(&C);
      // Otherwise this is a circular dependency, other candidates will be
      // removed by the condition [1].
    }
  }

  Type *OldRetTy = F->getReturnType();
  if (!NumPromoted)
    return OldRetTy;

  SmallVector<Type *, 5> ReturnArgTypes;
  ReturnArgTypes.reserve(NumPromoted + 1);
  if (!OldRetTy->isVoidTy())
    ReturnArgTypes.push_back(OldRetTy);

  SmallVector<ArgPromotionInfo *, 4> ReturnArgs;
  ReturnArgs.reserve(NumPromoted);
  for (ArgPromotionInfo &C : Candidates) {
    if (C.isPromoted()) {
      assert(C.Return);
      ReturnArgs.push_back(&C);
      ReturnArgTypes.push_back(C.ArgType);
    }
  }

  Type *RetTy = ReturnArgTypes.size() > 1
                    ? StructType::get(F->getContext(), ReturnArgTypes)
                    : ReturnArgTypes.front();

  // Replace old return instructions using annotated return values.
  for (auto &P : RetValues) {
    ReturnInst *OldRetInst = P.first;
    const auto &Values = P.second;
    assert(Values.size() == NumPromoted);
    Value *RetValue;
    if (OldRetTy->isVoidTy() && NumPromoted == 1)
      RetValue = Values[0];
    else {
      SmallString<256> NameData;
      StringRef Name =
          (F->getName() + dot(OldRetInst->getParent()->getName()) + ".ret")
              .toStringRef(NameData);
      RetValue = UndefValue::get(RetTy);
      unsigned I = 0;
      if (!OldRetTy->isVoidTy()) {
        RetValue = InsertValueInst::Create(
            RetValue, OldRetInst->getReturnValue(), {I++}, Name, OldRetInst->getIterator());
      }
      for (const ArgPromotionInfo *C : ReturnArgs) {
        RetValue =
            InsertValueInst::Create(RetValue, Values[C->ReturnValueIndex], {I},
                                    Name + Twine(I), OldRetInst->getIterator());
        ++I;
      }
    }
    ReturnInst::Create(OldRetInst->getContext(), RetValue, OldRetInst->getIterator());
    OldRetInst->eraseFromParent();
  }

  RetValuesStoreOrder.resize(NumPromoted);
  for (unsigned I = 0; I < NumPromoted; I++) {
    ArgPromotionInfo *C = ReturnArgs[I];
    RetValuesStoreOrder[NumPromoted - 1 - C->ReturnValueIndex] = C;
    // ReturnValueIndex is now the index in the aggregated return type.
    C->ReturnValueIndex = I + (OldRetTy->isVoidTy() ? 0 : 1);
  }
  return RetTy;
}

bool ArgumentPromoter::isInArgClobbered(const ArgPromotionInfo &ArgInfo) {
  LLVM_DEBUG(dbgs() << "  Searching for a clobber for in arg " << *ArgInfo.Arg
                    << ": ");
  assert(!ArgInfo.Return && ArgInfo.Preload);
  auto *Walker = MSSA.getWalker();
  for (Value *U : ArgInfo.users()) {
    assert(ArgInfo.isMyLoad(U));
    LoadInst *LI = cast<LoadInst>(U);
    auto *ClobberMA = Walker->getClobberingMemoryAccess(LI);
    if (!MSSA.isLiveOnEntryDef(ClobberMA)) {
      LLVM_DEBUG(printClobber(dbgs(), ClobberMA, LI));
      return true;
    }
  }
  LLVM_DEBUG(dbgs() << "no clobber\n");
  return false;
}

void ArgumentPromoter::promoteInArg(ArgPromotionInfo &ArgInfo) {
  assert(!ArgInfo.Return && ArgInfo.Preload);
  MemorySSAUpdater UMSSA(&MSSA);
  bool FirstAAMD = true;
  for (Value *U : make_early_inc_range(ArgInfo.users())) {
    assert(ArgInfo.isMyLoad(U));
    LoadInst *LI = cast<LoadInst>(U);
    if (FirstAAMD) {
      ArgInfo.AAMD = LI->getAAMetadata();
      FirstAAMD = false;
    } else if (ArgInfo.AAMD)
      ArgInfo.AAMD.merge(LI->getAAMetadata());
    LI->replaceAllUsesWith(ArgInfo.getOrCreatePreloadArgDummy());
    UMSSA.removeMemoryAccess(LI);
    LI->eraseFromParent();
  }
#ifndef NDEBUG
  MSSA.verifyMemorySSA();
#endif
}

// Create the function with the new signature.
Function *ArgumentPromoter::createNewFunction(
    Function *OldF, Type *RetTy,
    const SmallVectorImpl<ArgPromotionInfo *> &PromotedArgs) {

  SmallVector<Type *, 8> Params;
  SmallVector<AttributeSet, 8> ParamAttr;
  AttributeList PAL = OldF->getAttributes();
  auto PA = PromotedArgs.begin();
  for (unsigned ArgNo = 0; ArgNo < OldF->arg_size(); ++ArgNo) {
    if (PA != PromotedArgs.end() && (*PA)->getArgNo() == ArgNo) {
      assert((*PA)->isPromoted() || (*PA)->isUnusedArg());
      if ((*PA)->PreloadArgDummy) {
        Params.push_back((*PA)->ArgType);
        ParamAttr.push_back(AttributeSet());
      }
      ++PA;
    } else {
      Params.push_back(OldF->getArg(ArgNo)->getType());
      ParamAttr.push_back(PAL.getParamAttrs(ArgNo));
    }
  }
  assert(PA == PromotedArgs.end());

  FunctionType *OldFTy = OldF->getFunctionType();
  FunctionType *NFTy = FunctionType::get(RetTy, Params, OldFTy->isVarArg());
  Function *NF = Function::Create(NFTy, OldF->getLinkage(),
                                  OldF->getAddressSpace(), OldF->getName());
  NF->copyAttributesFrom(OldF);
  NF->copyMetadata(OldF, 0);
  NF->setAttributes(AttributeList::get(OldF->getContext(), PAL.getFnAttrs(),
                                       PAL.getRetAttrs(), ParamAttr));

  // The new function will have the !dbg metadata copied from the original
  // function. The original function may not be deleted, and dbg metadata need
  // to be unique so we need to drop it.
  OldF->setSubprogram(nullptr);
  OldF->getParent()->getFunctionList().insert(OldF->getIterator(), NF);
  NF->takeName(OldF);
  NF->splice(NF->begin(), OldF);

  auto NewArgI = NF->arg_begin();
  PA = PromotedArgs.begin();
  for (unsigned ArgNo = 0; ArgNo < OldF->arg_size(); ++ArgNo) {
    Argument &OldArg = *OldF->getArg(ArgNo);
    if (PA != PromotedArgs.end() && (*PA)->getArgNo() == ArgNo) {
      assert((*PA)->isPromoted() || (*PA)->isUnusedArg());
      if ((*PA)->PreloadArgDummy) {
        (*PA)->PreloadArgDummy->replaceAllUsesWith(NewArgI);
        NewArgI->setName((*PA)->getParamName());
        // Replace potential metadata uses (like llvm.dbg.value) with undef.
        OldArg.replaceAllUsesWith(UndefValue::get(OldArg.getType()));
        ++NewArgI;
      }
      ++PA;
    } else {
      OldArg.replaceAllUsesWith(&*NewArgI);
      NewArgI->takeName(&OldArg);
      ++NewArgI;
    }
  }
  assert(PA == PromotedArgs.end());
  return NF;
}

// Promote callsite to call the new function signature inserting loads and
// stores before and after the callsite.
void ArgumentPromoter::promoteCallsite(
    CallBase &CB, Function *NF,
    const SmallVectorImpl<ArgPromotionInfo *> &PromotedArgs,
    const SmallVectorImpl<ArgPromotionInfo *> &RetValuesStoreOrder) {

  SmallVector<Value *, 16> Args;
  SmallVector<AttributeSet, 8> ArgsAttr;
  const AttributeList &CallPAL = CB.getAttributes();
  IRBuilder<NoFolder> IRB(&CB);
  auto PA = PromotedArgs.begin();
  for (unsigned ArgNo = 0; ArgNo < CB.arg_size(); ++ArgNo) {
    Value *CallOp = CB.getArgOperand(ArgNo);
    if (PA != PromotedArgs.end() && (*PA)->getArgNo() == ArgNo) {
      assert((*PA)->isPromoted() || (*PA)->isUnusedArg());
      if ((*PA)->PreloadArgDummy) {
        Args.push_back((*PA)->createLoad(IRB, CallOp, CallOp->getName()));
        ArgsAttr.push_back(AttributeSet());
      }
      ++PA;
    } else {
      Args.push_back(CallOp);
      ArgsAttr.push_back(CallPAL.getParamAttrs(ArgNo));
    }
  }
  assert(PA == PromotedArgs.end());

  SmallVector<OperandBundleDef, 1> OpBundles;
  CB.getOperandBundlesAsDefs(OpBundles);
  CallBase *NewCS = nullptr;
  if (InvokeInst *II = dyn_cast<InvokeInst>(&CB)) {
    NewCS = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args, OpBundles, "", CB.getIterator());
  } else {
    auto *NewCall = CallInst::Create(NF, Args, OpBundles, "", CB.getIterator());
    NewCall->setTailCallKind(cast<CallInst>(&CB)->getTailCallKind());
    NewCS = NewCall;
  }
  NewCS->setCallingConv(CB.getCallingConv());
  NewCS->copyMetadata(CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});
  NewCS->takeName(&CB);
  NewCS->setAttributes(AttributeList::get(
      NF->getContext(), CallPAL.getFnAttrs(), CallPAL.getRetAttrs(), ArgsAttr));

  if (RetValuesStoreOrder.empty()) {
    CB.replaceAllUsesWith(NewCS);
    return;
  }

  // Processing return values.
  bool OldRetTyIsVoid = CB.getCalledFunction()->getReturnType()->isVoidTy();
  if (OldRetTyIsVoid && RetValuesStoreOrder.size() == 1) {
    const ArgPromotionInfo *A = RetValuesStoreOrder.front();
    A->createStore(IRB, NewCS, CB.getArgOperand(A->getArgNo()));
  } else {
    if (!OldRetTyIsVoid && !CB.user_empty())
      CB.replaceAllUsesWith(
          IRB.CreateExtractValue(NewCS, {0}, NewCS->getName() + ".ret"));
    for (const ArgPromotionInfo *A : RetValuesStoreOrder) {
      Value *CallOp = CB.getArgOperand(A->getArgNo());
      Value *RetVal = IRB.CreateExtractValue(NewCS, {A->ReturnValueIndex},
                                             CallOp->getName() + ".val.ret");
      A->createStore(IRB, RetVal, CallOp);
    }
  }
}

// Try to promote function argument candidates and update callsites.
Function *ArgumentPromoter::run(SmallVectorImpl<ArgPromotionInfo> &Candidates) {
  // Reload MSSA uncached walks constraint.
  NumMSSAWalksLeft = MaxMSSAWalksNum * Candidates.size();

  SmallVector<ArgPromotionInfo *, 4> RetValuesStoreOrder;
  Type *RetType = promoteInOutCandidates(Candidates, RetValuesStoreOrder);

  SmallVector<ArgPromotionInfo *, 4> PromotedArgs;
  for (ArgPromotionInfo &C : Candidates) {
    if (C.Return) {
      ++NumInOutArgCandidates;
      if (C.isPromoted()) {
        PromotedArgs.push_back(&C);
        ++NumInOutArgPromoted;
      }
    } else if (C.Preload) {
      ++NumInArgCandidates;
      if (!isInArgClobbered(C)) {
        promoteInArg(C);
        PromotedArgs.push_back(&C);
        ++NumInArgPromoted;
      }
    } else {
      assert(C.isUnusedArg());
      PromotedArgs.push_back(&C); // Will be removed from the func signature.
    }
  }

  if (PromotedArgs.empty())
    return nullptr;

  Function *NF = createNewFunction(F, RetType, PromotedArgs);

  // Update callsites.
  for (auto *U : make_early_inc_range(F->users())) {
    assert(isa<CallBase>(U));
    CallBase &CB = *cast<CallBase>(U);
    assert(CB.getCalledFunction() == F && CB.getParent()->getParent() != F);
    promoteCallsite(CB, NF, PromotedArgs, RetValuesStoreOrder);
    CB.eraseFromParent();
  }
  return NF;
}

// This method checks the specified function to see if there're any
// promotable arguments and if it is safe to promote the function (for
// example, all callers are direct) and performs the promotion.
static Function *promoteArguments(Function *F, FunctionAnalysisManager &FAM) {
  if (F->hasOptNone())
    return nullptr;

  // Don't perform argument promotion for naked functions; otherwise we can end
  // up removing parameters that are seemingly 'not used' as they are referred
  // to in the assembly.
  if (F->hasFnAttribute(Attribute::Naked))
    return nullptr;

  // Make sure that it is local to this module.
  if (!F->hasLocalLinkage())
    return nullptr;

  // Don't promote arguments for variadic functions. Adding, removing, or
  // changing non-pack parameters can change the classification of pack
  // parameters. Frontends encode that classification at the call site in the
  // IR, while in the callee the classification is determined dynamically based
  // on the number of registers consumed so far.
  if (F->isVarArg())
    return nullptr;

  // Don't transform functions that receive inallocas, as the transformation may
  // not be safe depending on calling convention.
  if (F->getAttributes().hasAttrSomewhere(Attribute::InAlloca))
    return nullptr;

  // See if there are any pointer arguments.
  if (F->args().end() == find_if(F->args(), [](Argument &A) {
        return A.getType()->isPointerTy();
      }))
    return nullptr;

  LLVM_DEBUG(dbgs() << "Trying to promote arguments for " << F->getName()
                    << '\n');

  // If the function has attributes for the return value they most likely
  // would not make sense for the aggregated return value, so we discard any
  // in/out arguments. The same applies to the return attributes at callsites.
  bool InArgsOnly = F->getAttributes().getRetAttrs().hasAttributes();

  for (Use &U : F->uses()) {
    CallBase *CB = dyn_cast<CallBase>(U.getUser());
    // Must be a direct call.
    if (CB == nullptr || !CB->isCallee(&U)) // [1]
      return nullptr;

    // Must have matching function type (no bitcasts or type mismatches).
    if (CB->getCalledFunction() != F)
      return nullptr;

    // Don't promote if there are recursive calls.
    if (CB->getParent()->getParent() == F)
      return nullptr;

    // Can't change signature of musttail callee
    if (CB->isMustTailCall())
      return nullptr;

    if (!InArgsOnly && CB->getAttributes().getRetAttrs().hasAttributes())
      InArgsOnly = true;
  }

  // Can't change signature of musttail caller
  for (BasicBlock &BB : *F)
    if (BB.getTerminatingMustTailCall())
      return nullptr;

  SmallVector<ArgPromotionInfo, 4> Candidates;
  for (Argument &A : F->args())
    if (A.getType()->isPointerTy())
      getPromotionCandidates(FAM, &A, Candidates, InArgsOnly);

  if (Candidates.empty())
    return nullptr;

  { // Make sure preloaded arguments are ABI compatible.
    // TODO: Check individual arguments so we can promote a subset?
    SmallVector<Type *, 32> Types;
    for (auto &C : Candidates) {
      if (C.Preload)
        Types.push_back(C.ArgType);
    }
    if (!Types.empty()) {
      const TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(*F);
      for (const Use &U : F->uses()) {
        CallBase *CB = cast<CallBase>(U.getUser()); // due to check [1]
        if (!TTI.areTypesABICompatible(CB->getCaller(), F, Types))
          return nullptr;
      }
    }
  }

  return ArgumentPromoter(F, FAM).run(Candidates);
}

PreservedAnalyses MSSAArgPromotionPass::run(LazyCallGraph::SCC &C,
                                            CGSCCAnalysisManager &AM,
                                            LazyCallGraph &CG,
                                            CGSCCUpdateResult &UR) {
  bool Changed = false, LocalChange;
  do { // Iterate until we stop promoting from this SCC.
    LocalChange = false;
    for (LazyCallGraph::Node &N : C) {
      Function &OldF = N.getFunction();
      FunctionAnalysisManager &FAM =
          AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
      if (Function *NewF = promoteArguments(&OldF, FAM)) {
        // Directly substitute the functions in the call graph. Note that this
        // requires the old function to be completely dead and completely
        // replaced by the new function. It does no call graph updates, it
        // merely swaps out the particular function mapped to a particular node
        // in the graph.
        C.getOuterRefSCC().replaceNodeFunction(N, *NewF);
        FAM.clear(OldF, OldF.getName());
        OldF.eraseFromParent();
        LocalChange = true;
      }
    }
    Changed |= LocalChange;
  } while (LocalChange);

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none(); // Since the function signature is changed.
}
