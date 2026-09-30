//===- OpenMPKernelTraffic.cpp - Per-kernel memory traffic ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analyzes OpenMP offload kernels and estimates *per iteration* global memory
// traffic (number of streams, loads and stores) and compute operations as well
// as total operations.
// Results are recorded in a per-kernel global so that the offload runtime can
// use the data for grid size selection decisions. (This is a workaround for
// downstream, where not all kernels already have a KLE.)
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/OpenMPKernelTraffic.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "openmp-kernel-traffic"

STATISTIC(NumKernelsAnalyzed, "Number of OpenMP kernels analyzed");
STATISTIC(NumKernelsAnnotated, "Number of OpenMP kernels annotated");

/// Suffix of the global recording a kernel's execution mode. Unlike the kernel
/// environment, one is emitted for every offload kernel.
/// TODO: For downstream compatibility. Should soon not be needed anymore.
static constexpr StringRef ExecModeSuffix = "_exec_mode";

/// Suffix of the global this pass emits, read back by the offload plugin.
/// TODO: for downstream compatibility, Should be added to KLE later.
static constexpr StringRef KernelTrafficSuffix = "_kernel_traffic";
/// The counters in a <kernel>_kernel_traffic global, which precede its status.
static constexpr unsigned NumTrafficCounters = 8;

/// Loop property with which OpenMPWorkLoopMarkerPass marks the loop of a static
/// worksharing construct.
static constexpr StringRef WorkLoopAttr = "llvm.loop.omp.work";

/// Whether \p L or a loop nested in it is the loop of a worksharing construct.
static bool runsWorkLoop(const Loop &L) {
  return any_of(L.getLoopsInPreorder(), [](const Loop *Sub) {
    return findOptionMDForLoop(Sub, WorkLoopAttr);
  });
}

namespace {

/// Per-nest counters that are not bucketed by stream.
struct NestCounter {
  uint64_t LoadBytes = 0, StoreBytes = 0;
  uint64_t LoadCount = 0, StoreCount = 0;
  uint64_t ComputeOps = 0, TotalInsts = 0;

  bool operator<(const NestCounter &RHS) const {
    // Assume that stores are heavier than loads and that count matters more
    // than bytes.
    if (StoreCount != RHS.StoreCount)
      return StoreCount < RHS.StoreCount;
    if (LoadCount != RHS.LoadCount)
      return LoadCount < RHS.LoadCount;
    if (StoreBytes != RHS.StoreBytes)
      return StoreBytes < RHS.StoreBytes;
    if (LoadBytes != RHS.LoadBytes)
      return LoadBytes < RHS.LoadBytes;
    return std::tie(ComputeOps, TotalInsts) <
           std::tie(RHS.ComputeOps, RHS.TotalInsts);
  }

  bool operator>(const NestCounter &RHS) const { return RHS < *this; }

  NestCounter &operator+=(const NestCounter &RHS) {
    LoadBytes = SaturatingAdd(LoadBytes, RHS.LoadBytes);
    StoreBytes = SaturatingAdd(StoreBytes, RHS.StoreBytes);
    LoadCount = SaturatingAdd(LoadCount, RHS.LoadCount);
    StoreCount = SaturatingAdd(StoreCount, RHS.StoreCount);
    ComputeOps = SaturatingAdd(ComputeOps, RHS.ComputeOps);
    TotalInsts = SaturatingAdd(TotalInsts, RHS.TotalInsts);
    return *this;
  }

  /// The counters of \p Weight runs of what these count.
  NestCounter operator*(uint64_t Weight) const {
    NestCounter R;
    R.LoadBytes = SaturatingMultiply(LoadBytes, Weight);
    R.StoreBytes = SaturatingMultiply(StoreBytes, Weight);
    R.LoadCount = SaturatingMultiply(LoadCount, Weight);
    R.StoreCount = SaturatingMultiply(StoreCount, Weight);
    R.ComputeOps = SaturatingMultiply(ComputeOps, Weight);
    R.TotalInsts = SaturatingMultiply(TotalInsts, Weight);
    return R;
  }
};

/// The accesses of one iteration of a loop nest, or of one call to a function
/// made from inside a nest.
struct AccessSummary {
  /// Bytes accessed per underlying object. In a function's summary, an access
  /// through one of its parameters is keyed by that Argument until a call site
  /// maps it to what the caller passes.
  DenseMap<const Value *, uint64_t> Streams;
  NestCounter Counters;
  /// Whether the loop of a worksharing construct runs as part of this, in a
  /// callee. The nest the callee is called from then runs it, too.
  bool RunsWorkLoop = false;
};

/// How often the blocks of a function run per unit the estimate is counted in.
/// The unit is an iteration of the loop of a worksharing construct if a block
/// runs as part of one. Otherwise, it is a call of the function if \p
/// CalledFromNest, or an iteration of the outermost loop around the block. A
/// loop whose trip count is unknown counts once.
class IterationWeights {
public:
  IterationWeights(Function &F, FunctionAnalysisManager &FAM,
                   bool CalledFromNest)
      : LI(FAM.getResult<LoopAnalysis>(F)),
        SE(FAM.getResult<ScalarEvolutionAnalysis>(F)),
        CalledFromNest(CalledFromNest) {
    for (const Loop *Top : LI) {
      if (runsWorkLoop(*Top))
        WorkNests.insert(Top);
    }
  }

  /// Whether the loop of a worksharing construct runs in the function.
  bool hasWorkLoop() const { return !WorkNests.empty(); }

  uint64_t get(const BasicBlock &BB) { return get(LI.getLoopFor(&BB)); }

private:
  uint64_t get(const Loop *L) {
    if (!L)
      return 1;
    if (auto It = Cache.find(L); It != Cache.end())
      return It->second;
    uint64_t Weight = get(L->getParentLoop());
    if (isNestedInUnit(*L)) {
      if (unsigned TripCount = SE.getSmallConstantTripCount(L))
        Weight = SaturatingMultiply(Weight, uint64_t(TripCount));
    }
    Cache[L] = Weight;
    return Weight;
  }

  /// Whether \p L is nested in the unit-defining construct, depending on
  /// context:
  /// - the nest/function contains a work loop -> that loop is unit-defining
  ///   -> return true if L is nested in that loop
  /// - nest without a work loop -> the nest's outermost loop is unit-defining
  ///   -> return true if L is not the outermost loop
  /// - callee without a work loop -> no loop, the call is unit-defining
  ///   -> return true
  bool isNestedInUnit(const Loop &L) const {
    bool HasWorkLoop = CalledFromNest
                           ? hasWorkLoop()
                           : WorkNests.contains(L.getOutermostLoop());
    if (!HasWorkLoop)
      return CalledFromNest || L.getParentLoop();
    // Loops around the worksharing loop, or next to it, span units.
    for (const Loop *P = L.getParentLoop(); P; P = P->getParentLoop()) {
      if (findOptionMDForLoop(P, WorkLoopAttr))
        return true;
    }
    return false;
  }

  const LoopInfo &LI;
  ScalarEvolution &SE;
  bool CalledFromNest;
  SmallPtrSet<const Loop *, 4> WorkNests;
  DenseMap<const Loop *, uint64_t> Cache;
};

/// Map call arguments to function parameters. Only cares about funtion
/// args/parameters. Aka, maps function pointer parameters to the functions that
/// got passed to the callee by the caller.
using Bindings = SmallVector<Function *, 4>;

/// Per-iteration global memory traffic of a kernel's heaviest loop nest.
struct KernelTraffic {
  /// Sum of the widths of the nest's global accesses.
  uint64_t Bytes = 0;
  /// Number of distinct underlying objects those accesses reached.
  unsigned Streams = 0;
  /// The same accesses split by direction, and the nest's instruction mix.
  /// Not used by the policy at the moment; recorded so it can be refitted
  /// from runtime logs.
  NestCounter Counters;

  /// Total order over the recorded fields, so the heaviest nest is picked
  /// without depending on traversal order. Assume that the total number of
  /// moved bytes matters most. Then, compare the number of streams. The
  /// counters are used for tie-break
  bool operator<(const KernelTraffic &RHS) const {
    return std::tie(Bytes, Streams, Counters) <
           std::tie(RHS.Bytes, RHS.Streams, RHS.Counters);
  }

  bool operator>(const KernelTraffic &RHS) const { return RHS < *this; }
};

/// Valid, or why something is not analyzable.
using TrafficStatus = omp::KernelTrafficStatus;

/// Walks a kernel, bucketing the width of every access to global addrspace it
/// performs inside a loop by underlying object, and keeps the heaviest nest.
class TrafficEstimator {
public:
  TrafficEstimator(const DataLayout &DL, FunctionAnalysisManager &FAM,
                   unsigned GlobalAS)
      : DL(DL), FAM(FAM), GlobalAS(GlobalAS) {}

  /// Estimate the traffic of \p Kernel into \p Heaviest. Return why the
  /// kernel cannot be analyzed meaningfully, or Valid.
  TrafficStatus run(Function &Kernel, KernelTraffic &Heaviest) {
    Heaviest = KernelTraffic();
    Nests.clear();
    Visited.clear();
    FlatAS = FAM.getResult<TargetIRAnalysis>(Kernel).getFlatAddressSpace();
    if (TrafficStatus Status = visit(Kernel, Bindings());
        Status != TrafficStatus::Valid)
      return Status;

    // Once the device runtime is inlined, a kernel might have several loops
    // besides the actual work loop. Only nests that run the loop of a
    // worksharing construct should characterize the kernel if there is any such
    // nest. Otherwise (the loop was not marked, or not lowered from a static
    // worksharing construct), the heaviest nest has to do.
    KernelTraffic HeaviestUnmarked;
    bool SawWorkLoop = false;
    for (const auto &Nest : Nests) {
      KernelTraffic Traffic;
      for (const auto &Stream : Nest.second.Streams)
        Traffic.Bytes = SaturatingAdd(Traffic.Bytes, Stream.second);
      Traffic.Streams = Nest.second.Streams.size();
      Traffic.Counters = Nest.second.Counters;
      if (Nest.second.RunsWorkLoop || runsWorkLoop(*Nest.first.first)) {
        SawWorkLoop = true;
        if (Traffic > Heaviest)
          Heaviest = Traffic;
      } else if (Traffic > HeaviestUnmarked) {
        HeaviestUnmarked = Traffic;
      }
    }

    if (SawWorkLoop)
      return TrafficStatus::Valid;

    LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << Kernel.getName()
                      << ": no worksharing loop found, using the heaviest "
                         "loop nest\n");
    Heaviest = HeaviestUnmarked;

    // Finding nothing means the work loop was not visible far more
    // often than it means a kernel touches no global memory. Reporting no
    // estimate leaves the grid to the fall-back heuristic of the device;
    // reporting zero would claim the kernel is as light as possible.
    if (!Heaviest.Bytes && !Heaviest.Streams) {
      LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << Kernel.getName()
                        << ": no traffic found -> conservative fallback\n");
      return TrafficStatus::NoTraffic;
    }

    return TrafficStatus::Valid;
  }

private:
  /// Walk \p F, which runs outside any loop nest, with its parameters bound as
  /// \p B. Follow loops and calls.
  /// Return why the function is not analyzable, or Valid.
  TrafficStatus visit(Function &F, const Bindings &B);

  /// Add the instructions in \p BB, which run \p Weight times per call or per
  /// iteration of the nest \p AS describes, to \p AS. The parameters of the
  /// function \p BB is in are bound as \p B.
  /// Return why something is not analyzable, or Valid.
  TrafficStatus accumulate(const BasicBlock &BB, AccessSummary &AS,
                           const Bindings &B, uint64_t Weight);

  /// The accesses of one call to \p F with its parameters bound as \p B, or
  /// null and why \p F is not analyzable.
  std::pair<const AccessSummary *, TrafficStatus> summarize(Function &F,
                                                            const Bindings &B);

  /// Add \p Child, the summary of \p Callee, which \p Call makes \p Weight
  /// times, to \p Parent.
  void addSummary(AccessSummary &Parent, const AccessSummary &Child,
                  const CallBase &Call, const Function &Callee,
                  uint64_t Weight);

  /// Whether a call to the declaration \p CB (non-intrinsic) may access global
  /// memory in a way this estimator cannot see.
  bool mayAccessGlobalMemory(const CallBase &CB) const;

  /// Whether an access through \p Ptr may reach global memory.
  bool mayBeGlobal(const Value *Ptr) const;

  /// Whether the underlying object \p Obj may be in global memory.
  bool mayBeGlobalObject(const Value *Obj) const;

  /// The stream an access through \p Ptr is attributed to.
  const Value *getStream(const Value *Ptr) const;

  /// Add a \p W byte access through \p Ptr, made \p Weight times, to \p AS.
  void recordAccess(AccessSummary &AS, const Value *Ptr, uint64_t W,
                    bool IsLoad, uint64_t Weight);

  const DataLayout &DL;
  FunctionAnalysisManager &FAM;
  unsigned GlobalAS;
  /// The target's flat address space, or ~0U if it has none.
  unsigned FlatAS = ~0U;

  /// Per-iteration accesses keyed by outermost loop and the binding its
  /// function was walked with. A function walked with two bindings runs its
  /// loops as two different nests, which must not be added up.
  std::map<std::pair<const Loop *, Bindings>, AccessSummary> Nests;

  /// Functions walked by visit(), with the binding they were walked with.
  std::set<std::pair<const Function *, Bindings>> Visited;

  /// Per-call summaries, null along with why for functions that are not
  /// analyzable. They depend on the binding but not on the kernel, so they are
  /// kept across kernels.
  std::map<std::pair<const Function *, Bindings>,
           std::pair<std::unique_ptr<AccessSummary>, TrafficStatus>>
      Summaries;
  /// Functions whose summary is being computed, to detect recursion.
  SmallPtrSet<const Function *, 8> InProgress;
};

} // namespace

/// We require InferAddrspace so that we see the actual addrspaces of memory
/// accesses instead of the generic ones where possible. The rest is classified
/// by mayBeGlobal().
static TrafficStatus checkAnalyzable(const Function &F) {
  if (F.hasFnAttribute(Attribute::OptimizeNone)) {
    LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << F.getName()
                      << ": not analyzable (marked as optnone)\n");
    return TrafficStatus::OptNone;
  }
  return TrafficStatus::Valid;
}

bool TrafficEstimator::mayAccessGlobalMemory(const CallBase &CB) const {
  MemoryEffects ME = CB.getMemoryEffects();
  if (ME.onlyAccessesInaccessibleMem())
    return false;
  if (!ME.onlyAccessesInaccessibleOrArgMem())
    return true;

  // Only memory reachable from the arguments. Classify the pointers like the
  // ones of loads and stores.
  return any_of(CB.args(), [&](const Value *Arg) {
    return Arg->getType()->isPointerTy() && mayBeGlobal(Arg);
  });
}

/// The objects an access through \p Ptr may reach. Looks through selects and
/// phis.
static SmallVector<const Value *, 4> getObjects(const Value *Ptr) {
  SmallVector<const Value *, 4> Objects;
  getUnderlyingObjects(Ptr, Objects);
  return Objects;
}

bool TrafficEstimator::mayBeGlobal(const Value *Ptr) const {
  unsigned AS = Ptr->getType()->getPointerAddressSpace();
  if (AS == GlobalAS)
    return true;
  if (AS != FlatAS)
    return false;

  // A flat pointer counts unless it provably only reaches private memory or
  // LDS.
  return any_of(getObjects(Ptr),
                [&](const Value *Obj) { return mayBeGlobalObject(Obj); });
}

bool TrafficEstimator::mayBeGlobalObject(const Value *Obj) const {
  if (unsigned ObjAS = Obj->getType()->getPointerAddressSpace();
      ObjAS != FlatAS)
    return ObjAS == GlobalAS;
  if (isa<AllocaInst>(Obj))
    return false;
  if (auto *GV = dyn_cast<GlobalVariable>(Obj))
    return GV->getAddressSpace() == GlobalAS;
  // Parameters, loaded pointers, etc.: assume global.
  return true;
}

const Value *TrafficEstimator::getStream(const Value *Ptr) const {
  // Keying by underlying object gives the stream count for free. An access that
  // only reaches one global object is that object's stream. One that may reach
  // several global objects, e.g. through a select between two arrays, is keyed
  // by the instruction, aka a separate stream.
  SmallVector<const Value *, 4> Objects = getObjects(Ptr);
  const Value *Global = nullptr;
  for (const Value *Obj : Objects) {
    if (!mayBeGlobalObject(Obj))
      continue;
    if (Global)
      return getUnderlyingObject(Ptr);
    Global = Obj;
  }
  return Global ? Global : getUnderlyingObject(Ptr);
}

void TrafficEstimator::recordAccess(AccessSummary &AS, const Value *Ptr,
                                    uint64_t W, bool IsLoad, uint64_t Weight) {
  uint64_t Bytes = SaturatingMultiply(W, Weight);
  uint64_t &StreamBytes = AS.Streams[getStream(Ptr)];
  StreamBytes = SaturatingAdd(StreamBytes, Bytes);

  NestCounter &T = AS.Counters;
  if (IsLoad) {
    T.LoadBytes = SaturatingAdd(T.LoadBytes, Bytes);
    T.LoadCount = SaturatingAdd(T.LoadCount, Weight);
  } else {
    T.StoreBytes = SaturatingAdd(T.StoreBytes, Bytes);
    T.StoreCount = SaturatingAdd(T.StoreCount, Weight);
  }
}

/// The function \p V is known to point to when the parameters of the function
/// it is used in are bound as \p B, or null.
static Function *resolve(Value *V, const Bindings &B) {
  V = V->stripPointerCastsAndAliases();
  if (auto *Fn = dyn_cast<Function>(V))
    return Fn;
  if (auto *Arg = dyn_cast<Argument>(V); Arg && Arg->getArgNo() < B.size())
    return B[Arg->getArgNo()];
  return nullptr;
}

/// The binding \p Call establishes for the parameters of its callee, given that
/// the parameters of the caller are bound as \p B.
static Bindings bind(const CallBase &Call, const Bindings &B) {
  Bindings Callee;
  for (Value *Arg : Call.args())
    Callee.push_back(resolve(Arg, B));
  while (!Callee.empty() && !Callee.back())
    Callee.pop_back();
  return Callee;
}

void TrafficEstimator::addSummary(AccessSummary &Parent,
                                  const AccessSummary &Child,
                                  const CallBase &Call, const Function &Callee,
                                  uint64_t Weight) {
  for (auto [Obj, W] : Child.Streams) {
    // Map the parameters that the callee sees to the arguments that the caller
    // passed. That correctly handles:
    // - two calls to the same callee with different arguments -> two streams
    // - accesses to the same array from the caller and the callee -> one stream
    if (auto *Arg = dyn_cast<Argument>(Obj);
        Arg && Arg->getParent() == &Callee && Arg->getArgNo() < Call.arg_size())
      Obj = getStream(Call.getArgOperand(Arg->getArgNo()));
    uint64_t &StreamBytes = Parent.Streams[Obj];
    StreamBytes = SaturatingAdd(StreamBytes, SaturatingMultiply(W, Weight));
  }
  Parent.Counters += Child.Counters * Weight;
  Parent.RunsWorkLoop |= Child.RunsWorkLoop;
}

std::pair<const AccessSummary *, TrafficStatus>
TrafficEstimator::summarize(Function &F, const Bindings &B) {
  auto Key = std::make_pair(static_cast<const Function *>(&F), B);
  if (auto It = Summaries.find(Key); It != Summaries.end())
    return {It->second.first.get(), It->second.second};

  // Reaching a function again while it is still being summarized means it is
  // recursive and may run any number of times per iteration.
  if (!InProgress.insert(&F).second) {
    LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << F.getName()
                      << ": not analyzable (recursive)\n");
    return {nullptr, TrafficStatus::Recursive};
  }

  // Everything the function does happens once per call, its loops as often as
  // they iterate.
  auto Sum = std::make_unique<AccessSummary>();
  TrafficStatus Status = checkAnalyzable(F);
  if (Status == TrafficStatus::Valid) {
    IterationWeights Weights(F, FAM, /*CalledFromNest=*/true);
    for (const BasicBlock &BB : F) {
      Status = accumulate(BB, *Sum, B, Weights.get(BB));
      if (Status != TrafficStatus::Valid)
        break;
    }
    Sum->RunsWorkLoop |= Weights.hasWorkLoop();
  }
  InProgress.erase(&F);
  if (Status != TrafficStatus::Valid)
    Sum = nullptr;
  auto &Slot = Summaries[std::move(Key)];
  Slot = std::make_pair(std::move(Sum), Status);
  return {Slot.first.get(), Slot.second};
}

TrafficStatus TrafficEstimator::accumulate(const BasicBlock &BB,
                                           AccessSummary &AS, const Bindings &B,
                                           uint64_t Weight) {
  NestCounter &T = AS.Counters;

  for (const Instruction &I : BB) {
    T.TotalInsts = SaturatingAdd(T.TotalInsts, Weight);

    if (auto *CB = dyn_cast<CallBase>(&I)) {
      // memcpy/memset are one access of their length on either side.
      if (auto *MI = dyn_cast<AnyMemIntrinsic>(CB)) {
        bool DestIsGlobal = mayBeGlobal(MI->getRawDest());
        auto *MT = dyn_cast<AnyMemTransferInst>(MI);
        bool SrcIsGlobal = MT && mayBeGlobal(MT->getRawSource());
        if (!DestIsGlobal && !SrcIsGlobal)
          continue;

        auto *Len = dyn_cast<ConstantInt>(MI->getLength());
        if (!Len) {
          LLVM_DEBUG(dbgs()
                     << "openmp-kernel-traffic: " << I.getFunction()->getName()
                     << ": not analyzable (memory intrinsic with "
                        "non-constant length)\n");
          return TrafficStatus::MemIntrinsicLength;
        }

        uint64_t W = Len->getZExtValue();
        if (SrcIsGlobal)
          recordAccess(AS, MT->getRawSource(), W, /*IsLoad=*/true, Weight);
        if (DestIsGlobal)
          recordAccess(AS, MI->getRawDest(), W, /*IsLoad=*/false, Weight);
        continue;
      }

      T.ComputeOps = SaturatingAdd(T.ComputeOps, Weight);

      Function *Callee = resolve(CB->getCalledOperand(), B);
      if (!Callee) {
        LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: "
                          << I.getFunction()->getName()
                          << ": not analyzable (unresolved indirect call)\n");
        return TrafficStatus::IndirectCall;
      }

      if (Callee->isDeclaration()) {
        if (Callee->isIntrinsic() || !mayAccessGlobalMemory(*CB))
          continue;
        LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: "
                          << I.getFunction()->getName()
                          << ": not analyzable (call to " << Callee->getName()
                          << " is an unresolved non-intrinsic that might "
                             "access global memory)\n");
        return TrafficStatus::OpaqueCall;
      }

      // Every call adds the callee's accesses again, as it would if inlined.
      auto [Sum, Status] = summarize(*Callee, bind(*CB, B));
      if (!Sum)
        return Status;
      addSummary(AS, *Sum, *CB, *Callee, Weight);
      continue;
    }

    // Classify compute ops.
    if (I.isBinaryOp() || isa<UnaryOperator>(&I)) {
      T.ComputeOps = SaturatingAdd(T.ComputeOps, Weight);
      continue;
    }

    // Classify memory ops.
    const Value *Ptr = nullptr;
    Type *AccessedTy = nullptr;
    if (auto *Load = dyn_cast<LoadInst>(&I)) {
      Ptr = Load->getPointerOperand();
      AccessedTy = Load->getType();
    } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
      Ptr = Store->getPointerOperand();
      AccessedTy = Store->getValueOperand()->getType();
    } else {
      // Everything that doesn't move data: pointer computation (GEPs, pointer
      // casts), control flow, compares, PHIs, and, for now, atomicrmw/cmpxchg
      // is not traffic.
      continue;
    }

    if (!mayBeGlobal(Ptr))
      continue;

    TypeSize Width = DL.getTypeStoreSize(AccessedTy);
    if (Width.isScalable()) {
      LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: "
                        << I.getFunction()->getName()
                        << ": not analyzable (uses scalable type)\n");
      return TrafficStatus::ScalableType;
    }

    recordAccess(AS, Ptr, Width.getFixedValue(), isa<LoadInst>(&I), Weight);
  }
  return TrafficStatus::Valid;
}

TrafficStatus TrafficEstimator::visit(Function &F, const Bindings &B) {
  if (!Visited.insert({&F, B}).second)
    return TrafficStatus::Valid;

  if (TrafficStatus Status = checkAnalyzable(F); Status != TrafficStatus::Valid)
    return Status;

  const LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  IterationWeights Weights(F, FAM, /*CalledFromNest=*/false);

  for (const BasicBlock &BB : F) {
    if (const Loop *L = LI.getLoopFor(&BB)) {
      if (TrafficStatus Status = accumulate(
              BB, Nests[{L->getOutermostLoop(), B}], B, Weights.get(BB));
          Status != TrafficStatus::Valid)
        return Status;
    } else {
      // Outside any loop: prologue or epilogue, not per-iteration traffic ->
      // neither memory, nor compute ops matter (in our current model). A call
      // may still lead to the loop, though.
      for (const Instruction &I : BB) {
        const auto *CB = dyn_cast<CallBase>(&I);
        if (!CB)
          continue;

        Function *Callee = resolve(CB->getCalledOperand(), B);
        if (!Callee) {
          LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << F.getName()
                            << ": not analyzable (unresolved indirect call)\n");
          return TrafficStatus::IndirectCall;
        }

        if (Callee->isDeclaration()) {
          if (Callee->isIntrinsic() || !mayAccessGlobalMemory(*CB))
            continue;
          LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << F.getName()
                            << ": not analyzable (unresolved declaration that "
                               "might access global memory)\n");
          return TrafficStatus::OpaqueCall;
        }

        if (TrafficStatus Status = visit(*Callee, bind(*CB, B));
            Status != TrafficStatus::Valid)
          return Status;
      }
    }
  }

  return TrafficStatus::Valid;
}

/// Emit the <kernel>_kernel_traffic global of \p Kernel: \p Traffic, or -1 for
/// every counter if there is none, followed by \p Status.
///
/// The fields are written in the declaration order of KernelTrafficTy in
/// offload/include/Shared/Environment.h, which the host plugin reads them back
/// in; the order must not change.
static void recordTraffic(Function &Kernel, TrafficStatus Status,
                          const KernelTraffic *Traffic = nullptr) {
  Module &M = *Kernel.getParent();
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  SmallVector<Constant *, NumTrafficCounters + 1> Values;
  if (Traffic) {
    const uint64_t Fields[NumTrafficCounters] = {
        Traffic->Bytes,
        Traffic->Streams,
        Traffic->Counters.LoadBytes,
        Traffic->Counters.StoreBytes,
        Traffic->Counters.LoadCount,
        Traffic->Counters.StoreCount,
        Traffic->Counters.ComputeOps,
        Traffic->Counters.TotalInsts,
    };
    for (uint64_t F : Fields) {
      // The runtime reads these as signed; cap them to INT32_MAX when writing
      // below rather than wrap so a pathological kernel cannot be mistaken for
      // a cheap one.
      Values.push_back(ConstantInt::getSigned(
          Int32Ty, static_cast<int32_t>(std::min<uint64_t>(F, INT32_MAX))));
    }
  } else {
    Values.append(NumTrafficCounters, ConstantInt::getSigned(Int32Ty, -1));
  }
  Values.push_back(
      ConstantInt::getSigned(Int32Ty, static_cast<int32_t>(Status)));

  Constant *Init = ConstantStruct::getAnon(Values);
  auto *GV = new GlobalVariable(
      M, Init->getType(), /*isConstant=*/true, GlobalValue::WeakODRLinkage,
      Init, Kernel.getName() + KernelTrafficSuffix, nullptr,
      GlobalValue::NotThreadLocal,
      M.getDataLayout().getDefaultGlobalsAddressSpace());
  GV->setVisibility(GlobalValue::ProtectedVisibility);

  // Nothing references it, so it has to be marked explicitly as being used.
  appendToCompilerUsed(M, GV);
}

PreservedAnalyses OpenMPKernelTrafficPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  unsigned GlobalAS = DL.getDefaultGlobalsAddressSpace();

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  TrafficEstimator Estimator(DL, FAM, GlobalAS);

  // Kernels are found through their exec-mode global rather than their kernel
  // environment because downstream's specialized no-loop and big-jump-loop
  // kernels have no kernel environment.
  // TODO: this is a stupid workaround for the current downstream state.
  // Soonish, every kernel should have a KLE.
  SmallVector<Function *, 16> Kernels;
  for (GlobalVariable &GV : M.globals()) {
    StringRef Name = GV.getName();
    if (!Name.ends_with(ExecModeSuffix))
      continue;
    Function *Kernel = M.getFunction(Name.drop_back(ExecModeSuffix.size()));
    if (Kernel && !Kernel->isDeclaration())
      Kernels.push_back(Kernel);
  }

  // Collected first: the loop below adds globals to the module.
  for (Function *Kernel : Kernels) {
    // The pass may run more than once in a pipeline. Keep a valid estimate of
    // an earlier run, but retry a failed one: the kernel may have become
    // analyzable since.
    std::string Name = (Kernel->getName() + KernelTrafficSuffix).str();
    if (GlobalValue *Existing = M.getNamedValue(Name)) {
      auto *GV = dyn_cast<GlobalVariable>(Existing);
      if (!GV || !GV->hasInitializer())
        continue;
      auto *Status = dyn_cast_or_null<ConstantInt>(
          GV->getInitializer()->getAggregateElement(NumTrafficCounters));
      if (!Status ||
          Status->getSExtValue() == static_cast<int32_t>(TrafficStatus::Valid))
        continue;
      removeFromUsedLists(M, [GV](Constant *C) { return C == GV; });
      GV->eraseFromParent();
    }

    ++NumKernelsAnalyzed;
    KernelTraffic Traffic;
    if (TrafficStatus Status = Estimator.run(*Kernel, Traffic);
        Status != TrafficStatus::Valid) {
      LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << Kernel->getName()
                        << ": not analyzable\n");
      recordTraffic(*Kernel, Status);
      continue;
    }

    LLVM_DEBUG(dbgs() << "openmp-kernel-traffic: " << Kernel->getName() << ": "
                      << Traffic.Bytes << " bytes/iter accessed over "
                      << Traffic.Streams << " stream(s); "
                      << Traffic.Counters.LoadBytes << " bytes/iter loaded by "
                      << Traffic.Counters.LoadCount << " load(s); "
                      << Traffic.Counters.StoreBytes << " bytes/iter stored by "
                      << Traffic.Counters.StoreCount << " store(s); "
                      << Traffic.Counters.ComputeOps << " compute ops/iter; "
                      << Traffic.Counters.TotalInsts << " total ops/iter\n");
    recordTraffic(*Kernel, TrafficStatus::Valid, &Traffic);
    ++NumKernelsAnnotated;
  }

  // Globals were added and appendToCompilerUsed replaces @llvm.compiler.used,
  // so anything keyed on the module's globals is stale.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

/// Mark the outermost loops in \p LI for which \p IsWorkLoop holds. Loops
/// nested in a marked loop belong to it.
static void markWorkLoops(LoopInfo &LI,
                          function_ref<bool(const Loop &)> IsWorkLoop) {
  SmallVector<Loop *, 8> Worklist(LI.begin(), LI.end());
  while (!Worklist.empty()) {
    Loop *L = Worklist.pop_back_val();
    if (IsWorkLoop(*L))
      addStringMetadataToLoop(L, WorkLoopAttr);
    else
      append_range(Worklist, L->getSubLoops());
  }
}

PreservedAnalyses OpenMPWorkLoopMarkerPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // The entry points that take the loop body as a callback run the loop
    // themselves. Once one of them is inlined, its loop is the kernel's.
    StringRef FName = F.getName();
    if (FName.starts_with("__kmpc_for_static_loop") ||
        FName.starts_with("__kmpc_distribute_static_loop") ||
        FName.starts_with("__kmpc_distribute_for_static_loop")) {
      const Argument *LoopBody = F.getArg(1);
      auto CallsLoopBody = [&](const Loop &L) {
        return any_of(LoopBody->users(), [&](const User *U) {
          const auto *CB = dyn_cast<CallBase>(U);
          return CB && CB->getCalledOperand() == LoopBody && L.contains(CB);
        });
      };
      markWorkLoops(FAM.getResult<LoopAnalysis>(F), CallsLoopBody);
      continue;
    }

    // The other entry points only mark begin and end of the work.
    SmallVector<const Instruction *, 4> Inits, Finis;
    for (const Instruction &I : instructions(F)) {
      const auto *CB = dyn_cast<CallBase>(&I);
      const Function *Callee = CB ? CB->getCalledFunction() : nullptr;
      if (!Callee)
        continue;
      StringRef Name = Callee->getName();
      if (Name.starts_with("__kmpc_for_static_init_") ||
          Name.starts_with("__kmpc_distribute_static_init_"))
        Inits.push_back(&I);
      else if (Name == "__kmpc_for_static_fini" ||
               Name == "__kmpc_distribute_static_fini")
        Finis.push_back(&I);
    }
    if (Inits.empty() || Finis.empty())
      continue;

    // The work loop is the outermost loop that runs after the first call and
    // before the second. A loop around the whole construct contains the calls,
    // so it is not the loop of the construct.
    const auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    const auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
    auto IsBracketed = [&](const Loop &L) {
      const BasicBlock *Header = L.getHeader();
      auto IsBefore = [&](const Instruction *I) {
        return !L.contains(I) && DT.dominates(I->getParent(), Header);
      };
      auto IsAfter = [&](const Instruction *I) {
        return !L.contains(I) && PDT.dominates(I->getParent(), Header);
      };
      // The init and the fini must belong to the same construct. A loop
      // between the fini of one construct and the init of the next is not a
      // work loop, so no fini may lie between the init and the loop, and no
      // init between the loop and the fini.
      bool AfterInit = any_of(Inits, [&](const Instruction *Init) {
        return IsBefore(Init) && none_of(Finis, [&](const Instruction *Fini) {
                 return IsBefore(Fini) && DT.dominates(Init, Fini);
               });
      });
      bool BeforeFini = any_of(Finis, [&](const Instruction *Fini) {
        return IsAfter(Fini) && none_of(Inits, [&](const Instruction *Init) {
                 return IsAfter(Init) && PDT.dominates(Fini, Init);
               });
      });
      return AfterInit && BeforeFini;
    };
    markWorkLoops(FAM.getResult<LoopAnalysis>(F), IsBracketed);
  }

  return PreservedAnalyses::all();
}
