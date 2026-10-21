//===- NextUseAnalysisIncrementalTest.cpp - incremental NUA tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPUUnitTests.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include "llvm/ADT/PostOrderIterator.h"

#include <chrono>
#include <cstdlib>
#include <iterator>
#include <type_traits>
#include <utility>

using namespace llvm;

#ifndef LLVM_AMDGPU_NEXT_USE_MIR_DIR
#define LLVM_AMDGPU_NEXT_USE_MIR_DIR "test/CodeGen/AMDGPU/NextUseAnalysis"
#endif

namespace {

struct NextUseSnapshot {
  DenseMap<unsigned, int64_t> DefRegToClosestUseDist;
  DenseMap<unsigned, int64_t> BlockSpan;
};

using QueryDistances = DenseMap<unsigned, uint64_t>;

static void captureNextUseSnapshot(const AMDGPUNextUseAnalysis &NUA,
                                   const MachineFunction &MF,
                                   NextUseSnapshot &Snap) {
  Snap.DefRegToClosestUseDist.clear();
  Snap.BlockSpan.clear();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.def_empty(Reg))
      continue;
    const MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
    if (!DefMI)
      continue;

    SmallVector<const MachineOperand *> Uses;
    NUA.getReachableUses(Reg, LaneBitmask::getAll(), *DefMI, Uses);
    NextUseDistance Dist = Uses.empty()
                               ? NextUseDistance::unreachable()
                               : NUA.getShortestDistance(Reg, *DefMI, Uses);
    Snap.DefRegToClosestUseDist[Reg.id()] = Dist.getRawValue();
  }

  for (const MachineBasicBlock &MBB : MF)
    Snap.BlockSpan[MBB.getNumber()] = NUA.getSpan(MBB).getRawValue();
}

struct UpdateQueryTiming {
  double BeforeQueryUs = 0;
  double IncrementalUpdateUs = 0;
  double IncrementalQueryUs = 0;
  double FullRebuildUs = 0;
  double FullQueryUs = 0;
};

static double elapsedMicros(std::chrono::steady_clock::time_point Start) {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - Start)
      .count();
}

template <typename Fn>
static decltype(auto) timeMicros(double &OutUs, Fn &&Body) {
  const auto Start = std::chrono::steady_clock::now();
  if constexpr (std::is_void_v<decltype(std::forward<Fn>(Body)())>) {
    std::forward<Fn>(Body)();
    OutUs = elapsedMicros(Start);
  } else {
    decltype(auto) Result = std::forward<Fn>(Body)();
    OutUs = elapsedMicros(Start);
    return Result;
  }
}

static void reportUpdateQueryTiming(StringRef Label,
                                    const UpdateQueryTiming &T) {
  const double IncrementalTotalUs =
      T.IncrementalUpdateUs + T.IncrementalQueryUs;
  const double FullTotalUs = T.FullRebuildUs + T.FullQueryUs;
  const double Speedup =
      IncrementalTotalUs > 0 ? FullTotalUs / IncrementalTotalUs : 0.0;

  std::string Scenario;
  if (const testing::TestInfo *Info =
          testing::UnitTest::GetInstance()->current_test_info()) {
    Scenario = Info->name();
    if (!Label.empty())
      Scenario += ("/" + Label).str();
  } else {
    Scenario = Label.str();
  }

  const auto Ms = [](double Us) { return formatv("{0:F3}", Us / 1000.0); };

  llvm::outs() << "[ >NUA time] speedup=" << formatv("{0:F3}", Speedup) << "x "
               << Scenario << ": ";
  if (T.BeforeQueryUs > 0) {
    llvm::outs() << "before(query=" << Ms(T.BeforeQueryUs) << "ms) ";
    testing::Test::RecordProperty(Scenario + ".before_query_ms",
                                  Ms(T.BeforeQueryUs).str());
  }
  llvm::outs() << "incremental(update=" << Ms(T.IncrementalUpdateUs)
               << "ms query=" << Ms(T.IncrementalQueryUs)
               << "ms total=" << Ms(IncrementalTotalUs)
               << "ms) full(rebuild=" << Ms(T.FullRebuildUs)
               << "ms query=" << Ms(T.FullQueryUs)
               << "ms total=" << Ms(FullTotalUs) << "ms)\n";

  testing::Test::RecordProperty(Scenario + ".full_over_incremental_speedup",
                                formatv("{0:F3}", Speedup).str());
  testing::Test::RecordProperty(Scenario + ".incremental_update_ms",
                                Ms(T.IncrementalUpdateUs).str());
  testing::Test::RecordProperty(Scenario + ".incremental_query_ms",
                                Ms(T.IncrementalQueryUs).str());
  testing::Test::RecordProperty(Scenario + ".full_rebuild_ms",
                                Ms(T.FullRebuildUs).str());
  testing::Test::RecordProperty(Scenario + ".full_query_ms",
                                Ms(T.FullQueryUs).str());
}

static void expectSnapshotsEqual(const NextUseSnapshot &Inc,
                                 const NextUseSnapshot &Fresh) {
  EXPECT_EQ(Inc.DefRegToClosestUseDist.size(),
            Fresh.DefRegToClosestUseDist.size());
  for (const auto &KV : Inc.DefRegToClosestUseDist) {
    SCOPED_TRACE("vreg " + Twine(KV.first).str());
    auto It = Fresh.DefRegToClosestUseDist.find(KV.first);
    ASSERT_NE(It, Fresh.DefRegToClosestUseDist.end());
    EXPECT_EQ(KV.second, It->second);
  }

  EXPECT_EQ(Inc.BlockSpan.size(), Fresh.BlockSpan.size());
  for (const auto &KV : Inc.BlockSpan) {
    SCOPED_TRACE("block " + Twine(KV.first).str());
    auto It = Fresh.BlockSpan.find(KV.first);
    ASSERT_NE(It, Fresh.BlockSpan.end());
    EXPECT_EQ(KV.second, It->second);
  }
}

static AMDGPUNextUseAnalysis &
getNextUseAnalysis(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM,
                   AMDGPUNextUseAnalysis::Config Cfg, bool InvalidateFirst) {
  if (InvalidateFirst)
    MFAM.invalidate(MF, PreservedAnalyses::none());
  AMDGPUNextUseAnalysis &NUA = MFAM.getResult<AMDGPUNextUseAnalysisPass>(MF);
  NUA.setConfig(Cfg);
  return NUA;
}

static void insertNop(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator InsertPt) {
  const GCNSubtarget &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  BuildMI(MBB, InsertPt, DebugLoc(), TII.get(AMDGPU::S_NOP)).addImm(0);
}

static void expectInstrAddedMatchesFreshRecompute(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM,
    AMDGPUNextUseAnalysis::Config Cfg, MachineBasicBlock &ChangedMBB,
    function_ref<void(MachineFunction &)> ModifyMF) {
  AMDGPUNextUseAnalysis &Inc =
      getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);

  ModifyMF(MF);

  UpdateQueryTiming Timings;
  timeMicros(Timings.IncrementalUpdateUs, [&] { //
    Inc.instrAdded(ChangedMBB);
  });
  NextUseSnapshot AfterIncremental;
  timeMicros(Timings.IncrementalQueryUs, [&] { //
    captureNextUseSnapshot(Inc, MF, AfterIncremental);
  });
  AMDGPUNextUseAnalysis *FreshPtr = nullptr;
  timeMicros(Timings.FullRebuildUs, [&] {
    FreshPtr = &getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);
  });
  AMDGPUNextUseAnalysis &Fresh = *FreshPtr;
  NextUseSnapshot FreshSnap;
  timeMicros(Timings.FullQueryUs, [&] { //
    captureNextUseSnapshot(Fresh, MF, FreshSnap);
  });

  reportUpdateQueryTiming(ChangedMBB.getName(), Timings);
  expectSnapshotsEqual(AfterIncremental, FreshSnap);
}

// Exercise the operand-level hooks: rewrite uses in place, notify, and require
// the result to match a full rebuild. The warm-up query matters -- it populates
// the register-keyed caches, including the empty use lists that make a missed
// notification read as "no next use" rather than crash.
static void expectRegUsesChangedMatchesFreshRecompute(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM,
    AMDGPUNextUseAnalysis::Config Cfg, StringRef Label,
    function_ref<void(MachineFunction &, AMDGPUNextUseAnalysis &)>
        RewriteAndNotify) {
  AMDGPUNextUseAnalysis &Inc =
      getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);

  UpdateQueryTiming Timings;
  NextUseSnapshot WarmSnap;
  timeMicros(Timings.BeforeQueryUs,
             [&] { captureNextUseSnapshot(Inc, MF, WarmSnap); });
  ASSERT_TRUE(Inc.verifyRegUseCache());

  timeMicros(Timings.IncrementalUpdateUs, [&] { RewriteAndNotify(MF, Inc); });
  EXPECT_TRUE(Inc.verifyRegUseCache());

  NextUseSnapshot AfterIncremental;
  timeMicros(Timings.IncrementalQueryUs,
             [&] { captureNextUseSnapshot(Inc, MF, AfterIncremental); });

  AMDGPUNextUseAnalysis *FreshPtr = nullptr;
  timeMicros(Timings.FullRebuildUs, [&] {
    FreshPtr = &getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);
  });
  AMDGPUNextUseAnalysis &Fresh = *FreshPtr;
  NextUseSnapshot FreshSnap;
  timeMicros(Timings.FullQueryUs,
             [&] { captureNextUseSnapshot(Fresh, MF, FreshSnap); });

  reportUpdateQueryTiming(Label, Timings);
  expectSnapshotsEqual(AfterIncremental, FreshSnap);
}

// Repoint every use of \p OldReg in \p MI at \p NewReg, reporting the rewrite
// the way SSA repair would.
static void rewriteUsesInInstr(AMDGPUNextUseAnalysis &NUA, MachineInstr &MI,
                               Register OldReg, Register NewReg) {
  SmallVector<std::pair<const MachineOperand *, Register>, 4> Rewrites;
  for (MachineOperand &MO : MI.uses()) {
    if (!MO.isReg() || MO.getReg() != OldReg)
      continue;
    MO.setReg(NewReg);
    Rewrites.emplace_back(&MO, OldReg);
  }
  ASSERT_FALSE(Rewrites.empty());
  NUA.regUsesChanged(Rewrites);
}

static void removeMI(AMDGPUNextUseAnalysis &NUA, MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();
  NUA.instrRemoving(MI);
  MI.eraseFromParent();
  NUA.instrRemoved(*MBB);
}

static void expectInstrRemovedMatchesFreshRecompute(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM,
    AMDGPUNextUseAnalysis::Config Cfg, MachineBasicBlock &ChangedMBB,
    function_ref<void(AMDGPUNextUseAnalysis &, MachineFunction &)> ModifyMF) {
  AMDGPUNextUseAnalysis &Inc =
      getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);

  UpdateQueryTiming Timings;
  timeMicros(Timings.IncrementalUpdateUs, [&] { //
    ModifyMF(Inc, MF);
  });
  NextUseSnapshot AfterIncremental;
  timeMicros(Timings.IncrementalQueryUs, [&] { //
    captureNextUseSnapshot(Inc, MF, AfterIncremental);
  });
  AMDGPUNextUseAnalysis *FreshPtr = nullptr;
  timeMicros(Timings.FullRebuildUs, [&] {
    FreshPtr = &getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);
  });
  AMDGPUNextUseAnalysis &Fresh = *FreshPtr;
  NextUseSnapshot FreshSnap;
  timeMicros(Timings.FullQueryUs, [&] { //
    captureNextUseSnapshot(Fresh, MF, FreshSnap);
  });

  reportUpdateQueryTiming(ChangedMBB.getName(), Timings);
  expectSnapshotsEqual(AfterIncremental, FreshSnap);
}

static unsigned refBlockSize(const MachineBasicBlock &BB, bool CountPhis) {
  unsigned Size = BB.size();
  if (!CountPhis)
    Size -= std::distance(BB.begin(), BB.getFirstNonPHI());
  return Size;
}

static unsigned refRelativeLoopDepth(const MachineLoopInfo &MLI,
                                     const MachineBasicBlock *From,
                                     const MachineBasicBlock *To) {
  MachineLoop *LoopFrom = MLI.getLoopFor(From);
  MachineLoop *LoopTo = MLI.getLoopFor(To);

  if (!LoopFrom)
    return 0;

  if (!LoopTo)
    return LoopFrom->getLoopDepth();

  if (LoopFrom->contains(LoopTo))
    return 0;

  if (LoopTo->contains(LoopFrom))
    return LoopFrom->getLoopDepth() - LoopTo->getLoopDepth();

  unsigned Depth = 0;
  for (MachineLoop *A = LoopFrom; A != nullptr;
       A = A->getParentLoop(), ++Depth) {
    if (A->contains(LoopTo))
      return Depth;
  }
  return LoopFrom->getLoopDepth();
}

struct RefInterBlockDistance {
  NextUseDistance Weighted = NextUseDistance::unreachable();
  NextUseDistance Unweighted = NextUseDistance::unreachable();

  RefInterBlockDistance() = default;
  RefInterBlockDistance(NextUseDistance W, NextUseDistance U)
      : Weighted(W), Unweighted(U) {}

  bool operator==(const RefInterBlockDistance &O) const {
    return Weighted == O.Weighted && Unweighted == O.Unweighted;
  }
  bool operator!=(const RefInterBlockDistance &O) const {
    return !(*this == O);
  }
};

using RefInterBlockDistanceMap =
    DenseMap<unsigned, DenseMap<unsigned, RefInterBlockDistance>>;

static RefInterBlockDistanceMap computeReferenceInterBlockDistances(
    const MachineFunction &MF, const MachineLoopInfo &MLI, bool CountPhis) {
  RefInterBlockDistanceMap Distances;

  bool Changed = false;
  unsigned Iterations = 0;
  const unsigned MaxIterations = MF.getNumBlockIDs() + 2;
  do {
    if (++Iterations > MaxIterations) {
      ADD_FAILURE() << "reference inter-block distance fixpoint did not "
                       "converge";
      return Distances;
    }
    Changed = false;
    for (const MachineBasicBlock *MBB : post_order(&MF)) {
      unsigned MBBNum = MBB->getNumber();

      RefInterBlockDistanceMap::mapped_type Prev = std::move(Distances[MBBNum]);
      RefInterBlockDistanceMap::mapped_type Curr;
      Curr.reserve(Prev.size());

      for (const MachineBasicBlock *Succ : MBB->successors())
        Curr[Succ->getNumber()] = RefInterBlockDistance{0, 0};

      for (const MachineBasicBlock *Succ : MBB->successors()) {
        unsigned SuccNum = Succ->getNumber();
        const unsigned UnweightedSize = refBlockSize(*Succ, CountPhis);

        auto SuccRow = Distances.find(SuccNum);
        if (SuccRow == Distances.end())
          continue;

        for (const auto &[DestBlockNum, DestDist] : SuccRow->second) {
          if (DestBlockNum == MBBNum)
            continue;

          const MachineBasicBlock *DestMBB = MF.getBlockNumbered(DestBlockNum);

          NextUseDistance UnweightedDist = NextUseDistance::unreachable();
          if (DestDist.Unweighted.isReachable())
            UnweightedDist = UnweightedSize + DestDist.Unweighted;

          unsigned SuccToDestLoopDepth =
              refRelativeLoopDepth(MLI, Succ, DestMBB);

          NextUseDistance WeightedDist = NextUseDistance::unreachable();
          if (DestDist.Weighted.isReachable())
            WeightedDist =
                DestDist.Weighted +
                NextUseDistance::fromSize(UnweightedSize, SuccToDestLoopDepth);

          auto [I, First] =
              Curr.try_emplace(DestBlockNum, WeightedDist, UnweightedDist);
          if (!First) {
            RefInterBlockDistance &Slot = I->second;
            Slot.Weighted = min(Slot.Weighted, WeightedDist);
            Slot.Unweighted = min(Slot.Unweighted, UnweightedDist);
          }
        }
      }
      Changed |= (Prev != Curr);
      Distances[MBBNum] = std::move(Curr);
    }
  } while (Changed);

  return Distances;
}

static RefInterBlockDistance
lookupReferenceDistance(const RefInterBlockDistanceMap &Table, unsigned FromNum,
                        unsigned ToNum) {
  auto FromIt = Table.find(FromNum);
  if (FromIt == Table.end())
    return {};
  auto ToIt = FromIt->second.find(ToNum);
  if (ToIt == FromIt->second.end())
    return {};
  return ToIt->second;
}

static void expectInterBlockDistancesMatchReference(
    const AMDGPUNextUseAnalysis &NUA, const MachineFunction &MF,
    const MachineLoopInfo &MLI, AMDGPUNextUseAnalysis::Config Cfg) {
  const RefInterBlockDistanceMap Ref =
      computeReferenceInterBlockDistances(MF, MLI, Cfg.CountPhis);

  for (unsigned FromNum = 0, NumIds = MF.getNumBlockIDs(); FromNum != NumIds;
       ++FromNum) {
    const MachineBasicBlock *From = MF.getBlockNumbered(FromNum);
    if (!From)
      continue;

    for (unsigned ToNum = 0; ToNum != NumIds; ++ToNum) {
      if (FromNum == ToNum)
        continue;
      const MachineBasicBlock *To = MF.getBlockNumbered(ToNum);
      if (!To)
        continue;

      RefInterBlockDistance ExpectedRef =
          lookupReferenceDistance(Ref, FromNum, ToNum);
      if (Cfg.ForwardOnly && !NUA.isForwardReachable(From, To)) {
        ExpectedRef.Weighted = NextUseDistance::unreachable();
        ExpectedRef.Unweighted = NextUseDistance::unreachable();
      }

      SCOPED_TRACE("blocks " + Twine(FromNum) + " -> " + Twine(ToNum) +
                   " weighted");
      NextUseDistance GotWeighted =
          NUA.getInterBlockDistanceForTest(From, To, /*Unweighted*/ false);
      EXPECT_EQ(ExpectedRef.Weighted.getRawValue(), GotWeighted.getRawValue());

      SCOPED_TRACE("blocks " + Twine(FromNum) + " -> " + Twine(ToNum) +
                   " unweighted");
      NextUseDistance GotUnweighted =
          NUA.getInterBlockDistanceForTest(From, To, /*Unweighted*/ true);
      EXPECT_EQ(ExpectedRef.Unweighted.getRawValue(),
                GotUnweighted.getRawValue());
    }
  }
}

class CanonicalDigest {
  uint64_t H = 0xcbf29ce484222325ULL;

public:
  void add(StringRef S) {
    for (char C : S) {
      H ^= static_cast<unsigned char>(C);
      H *= 0x100000001b3ULL;
    }
  }

  void add(int64_t V) {
    add(Twine(V).str());
    add(";");
  }

  uint64_t value() const { return H; }
};

static uint64_t digestInterBlockDistances(const AMDGPUNextUseAnalysis &NUA,
                                          const MachineFunction &MF) {
  CanonicalDigest D;
  for (unsigned FromNum = 0, NumIds = MF.getNumBlockIDs(); FromNum != NumIds;
       ++FromNum) {
    const MachineBasicBlock *From = MF.getBlockNumbered(FromNum);
    if (!From)
      continue;
    for (unsigned ToNum = 0; ToNum != NumIds; ++ToNum) {
      if (FromNum == ToNum)
        continue;
      const MachineBasicBlock *To = MF.getBlockNumbered(ToNum);
      if (!To)
        continue;
      D.add(FromNum);
      D.add(ToNum);
      D.add(NUA.getInterBlockDistanceForTest(From, To, false).getRawValue());
      D.add(NUA.getInterBlockDistanceForTest(From, To, true).getRawValue());
    }
  }
  return D.value();
}

static uint64_t digestReachability(const AMDGPUNextUseAnalysis &NUA,
                                   const MachineFunction &MF) {
  CanonicalDigest D;
  for (unsigned FromNum = 0, NumIds = MF.getNumBlockIDs(); FromNum != NumIds;
       ++FromNum) {
    const MachineBasicBlock *From = MF.getBlockNumbered(FromNum);
    if (!From)
      continue;
    for (unsigned ToNum = 0; ToNum != NumIds; ++ToNum) {
      const MachineBasicBlock *To = MF.getBlockNumbered(ToNum);
      if (!To)
        continue;
      D.add(FromNum);
      D.add(ToNum);
      D.add(NUA.isReachable(From, To) ? 1 : 0);
      D.add(NUA.isForwardReachable(From, To) ? 1 : 0);
    }
  }
  return D.value();
}

//===----------------------------------------------------------------------===//
// Simulated AMDGPUEarlyRegisterSpilling query pattern
//
// ERS walks blocks in reverse post order and, wherever pressure exceeds the
// budget, ranks the registers live at that point by next-use distance, spills
// the furthest one, and continues. Two properties distinguish it from recovery
// and both are reproduced here:
//
//   - Queries are single-source per batch, but the source is the point itself
//     only outside a loop. Inside one it is the back of the outermost enclosing
//     loop's preheader (or the front of its header), so every pressure point in
//     a loop nest shares a source block.
//   - It mutates many times per function rather than once per attempt, and
//     each spill both inserts instructions and redirects a use, so it drives
//     instrAdded and regUsesChanged together.
//
// Two things are deliberately not simulated. Pressure accounting is replaced by
// "every Nth instruction in reverse post order", since what matters here is the
// query and notification shape, not which points ERS picks. And the spill is
// modelled by its analysis-visible footprint -- one instruction after the def,
// one before the redirected use, and the use rewritten to a fresh register --
// rather than by real spill opcodes and frame indices, because only block sizes
// and use lists feed next-use distance.
//===----------------------------------------------------------------------===//
struct ErsPressurePoint {
  // Stable across analysis invalidation and instruction insertion: MBBs and
  // MachineInstrs belong to the function, not to an analysis, and nothing here
  // erases instructions.
  MachineInstr *CurMI = nullptr;
  MachineBasicBlock *Preheader = nullptr; // null outside a loop
  MachineBasicBlock *Header = nullptr;
  SmallVector<std::pair<Register, LaneBitmask>, 0> Candidates;
};

// ERS recomputes StartMI from the loop each time, so resolve it at query time
// rather than caching an instruction that insertion may have displaced.
static MachineInstr *ersQuerySource(const ErsPressurePoint &P) {
  if (!P.Preheader)
    return P.CurMI;
  if (!P.Preheader->empty())
    return &P.Preheader->back();
  return P.Header->empty() ? nullptr : &P.Header->front();
}

static void planErsPressurePoints(MachineFunction &MF, LiveIntervals &LIS,
                                  const MachineLoopInfo &MLI, unsigned Stride,
                                  unsigned MaxPoints,
                                  SmallVectorImpl<ErsPressurePoint> &Points) {
  GCNDownwardRPTracker RPTracker(LIS);
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned Seen = 0;

  for (MachineBasicBlock *MBB : RPOT) {
    if (MBB->empty())
      continue;

    MachineLoop *Loop = MLI.getLoopFor(MBB);
    MachineLoop *Outermost = Loop ? Loop->getOutermostLoop() : nullptr;
    // ERS only takes the preheader path when it has one to take.
    MachineBasicBlock *Preheader =
        Outermost ? Outermost->getLoopPreheader() : nullptr;

    RPTracker.reset(*MBB->begin(), MBB->end());
    RPTracker.advance();

    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr())
        continue;

      if (Seen++ % Stride == 0 && Points.size() < MaxPoints) {
        ErsPressurePoint P;
        P.CurMI = &MI;
        if (Preheader) {
          P.Preheader = Preheader;
          P.Header = Outermost->getHeader();
        }
        for (auto [RegId, Mask] : RPTracker.getLiveRegs()) {
          Register Reg(RegId);
          if (!Reg.isVirtual() || !MF.getRegInfo().getUniqueVRegDef(Reg))
            continue;
          P.Candidates.emplace_back(Reg, Mask);
        }
        // getLiveRegs hands back a DenseMap; iteration order is not stable
        // between the two replays this plan feeds.
        llvm::sort(P.Candidates, [](const auto &A, const auto &B) {
          return A.first.id() < B.first.id();
        });
        if (!P.Candidates.empty())
          Points.push_back(std::move(P));
      }

      if (&MI == &MBB->back())
        break;
      if (MI.isPHI())
        RPTracker.reset(*MI.getNextNode(), MBB->end());
      RPTracker.advance();
    }
  }
}

// One ranked candidate, in ERS's terms. Dist is signed to match
// NextUseDistance::getRawValue, whose unreachable sentinel is int64_t's max.
struct ErsRankedCandidate {
  Register Reg;
  int64_t Dist = 0;
};

// Run the batch ERS runs at one pressure point: for every live candidate,
// collect reachable uses from the single source and take the shortest distance,
// plus the reachability probes ERS interleaves. Returns elapsed microseconds.
static double runErsRankingQuery(const AMDGPUNextUseAnalysis &NUA,
                                 const MachineFunction &MF,
                                 const ErsPressurePoint &P,
                                 SmallVectorImpl<ErsRankedCandidate> &Ranked) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineInstr *StartMI = ersQuerySource(P);
  if (!StartMI)
    return 0.0;
  const MachineBasicBlock *CurMBB = P.CurMI->getParent();

  const auto Start = std::chrono::steady_clock::now();
  for (auto [Reg, Mask] : P.Candidates) {
    const MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
    if (!Def || Def == StartMI)
      continue;

    // ERS asks about pairs unrelated to the query instruction, in both
    // directions; these are served by the same table as the distances.
    (void)NUA.isReachable(CurMBB, Def->getParent());

    SmallVector<const MachineOperand *> Uses;
    NUA.getReachableUses(Reg, Mask, *StartMI, Uses);

    // Loop-header PHI uses are skipped, as in getCandidates.
    if (P.Header)
      llvm::erase_if(Uses, [&](const MachineOperand *MO) {
        const MachineInstr *UseMI = MO->getParent();
        return UseMI->isPHI() && UseMI->getParent() == P.Header;
      });
    if (Uses.empty())
      continue;

    const MachineBasicBlock *FirstUseMBB =
        Uses.front()->getParent()->getParent();
    (void)NUA.isReachable(StartMI->getParent(), FirstUseMBB);

    Ranked.push_back(
        {Reg, NUA.getShortestDistance(Reg, *StartMI, Uses).getRawValue()});
  }
  const double ElapsedUs = elapsedMicros(Start);

  llvm::stable_sort(
      Ranked, [](const ErsRankedCandidate &A, const ErsRankedCandidate &B) {
        return std::tie(A.Dist, A.Reg) > std::tie(B.Dist, B.Reg);
      });
  return ElapsedUs;
}

// The analysis-visible footprint of spilling \p Reg: an instruction after its
// def, an instruction defining a fresh register before the use being
// redirected, and that use rewritten. Fills \p ChangedMBBs with every block the
// caller must pass to instrAdded, and reports the rewrite for regUsesChanged.
static void applyErsSpillFootprint(
    MachineFunction &MF, Register Reg, const MachineInstr &StartMI,
    SmallVectorImpl<MachineBasicBlock *> &ChangedMBBs,
    SmallVectorImpl<std::pair<const MachineOperand *, Register>> &Rewrites) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
  ASSERT_NE(Def, nullptr);

  // Redirect the first use reachable from the query source, which is the one
  // ERS's reload would dominate.
  SmallVector<const MachineOperand *> Uses;
  MachineOperand *UseMO = nullptr;
  for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
    if (MO.isUndef() || MO.getParent() == Def)
      continue;
    UseMO = &MO;
    break;
  }
  if (!UseMO)
    return;

  MachineInstr *UseMI = UseMO->getParent();
  MachineBasicBlock &DefMBB = *Def->getParent();
  MachineBasicBlock &UseMBB = *UseMI->getParent();

  BuildMI(DefMBB, std::next(Def->getIterator()), Def->getDebugLoc(),
          TII.get(AMDGPU::S_NOP))
      .addImm(0);

  Register ReloadReg = MRI.createVirtualRegister(MRI.getRegClass(Reg));
  BuildMI(UseMBB, UseMI->getIterator(), UseMI->getDebugLoc(),
          TII.get(AMDGPU::IMPLICIT_DEF), ReloadReg);

  const Register OldReg = UseMO->getReg();
  UseMO->setReg(ReloadReg);
  Rewrites.emplace_back(UseMO, OldReg);

  ChangedMBBs.push_back(&DefMBB);
  if (&UseMBB != &DefMBB)
    ChangedMBBs.push_back(&UseMBB);
}

// Walk the planned pressure points in order, keeping the analysis in sync
// either through the notification hooks or by full rebuild, and record what
// each batch answered.
static void runErsSweep(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM,
                        AMDGPUNextUseAnalysis::Config Cfg, LiveIntervals &LIS,
                        const MachineLoopInfo &MLI, bool Incremental,
                        unsigned Stride, unsigned MaxPoints,
                        UpdateQueryTiming &Timings,
                        SmallVectorImpl<QueryDistances> &PerPoint) {
  SmallVector<ErsPressurePoint, 8> Points;
  planErsPressurePoints(MF, LIS, MLI, Stride, MaxPoints, Points);
  ASSERT_FALSE(Points.empty());

  AMDGPUNextUseAnalysis *NUA =
      &getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);

  SmallVector<MachineBasicBlock *, 2> ChangedMBBs;
  SmallVector<std::pair<const MachineOperand *, Register>, 2> Rewrites;

  // Characterise the simulated workload, so a change in the MIR that quietly
  // reduces it to a handful of candidates is visible rather than silent.
  size_t TotalCandidates = 0, PreheaderSourced = 0, Notifications = 0;
  DenseSet<const MachineBasicBlock *> SourceBlocks;

  for (unsigned I = 0, E = Points.size(); I != E; ++I) {
    SCOPED_TRACE("pressure point " + std::to_string(I));

    // Catch up with the spill made at the previous point.
    if (I != 0) {
      const auto UpdateStart = std::chrono::steady_clock::now();
      if (Incremental) {
        for (MachineBasicBlock *MBB : ChangedMBBs) {
          NUA->instrAdded(*MBB);
          ++Notifications;
        }
        if (!Rewrites.empty())
          NUA->regUsesChanged(Rewrites);
      } else {
        NUA = &getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);
      }
      const double ElapsedUs = elapsedMicros(UpdateStart);
      if (Incremental) {
        Timings.IncrementalUpdateUs += ElapsedUs;
        EXPECT_TRUE(NUA->verifyRegUseCache());
      } else {
        Timings.FullRebuildUs += ElapsedUs;
      }
    }

    TotalCandidates += Points[I].Candidates.size();
    PreheaderSourced += Points[I].Preheader != nullptr;
    if (const MachineInstr *Src = ersQuerySource(Points[I]))
      SourceBlocks.insert(Src->getParent());

    SmallVector<ErsRankedCandidate, 32> Ranked;
    const double QueryUs = runErsRankingQuery(*NUA, MF, Points[I], Ranked);
    if (Incremental)
      Timings.IncrementalQueryUs += QueryUs;
    else
      Timings.FullQueryUs += QueryUs;

    QueryDistances Distances;
    for (const ErsRankedCandidate &C : Ranked)
      Distances[C.Reg.id()] = C.Dist;
    PerPoint.push_back(std::move(Distances));

    ChangedMBBs.clear();
    Rewrites.clear();
    if (Ranked.empty())
      continue;
    MachineInstr *StartMI = ersQuerySource(Points[I]);
    ASSERT_NE(StartMI, nullptr);
    ASSERT_NO_FATAL_FAILURE(applyErsSpillFootprint(
        MF, Ranked.front().Reg, *StartMI, ChangedMBBs, Rewrites));
  }

  if (Incremental)
    llvm::outs() << "[ >ERS sim ] blocks=" << MF.size()
                 << " points=" << Points.size()
                 << " candidates=" << TotalCandidates
                 << " preheader-sourced=" << PreheaderSourced
                 << " distinct-source-blocks=" << SourceBlocks.size()
                 << " instrAdded-calls=" << Notifications << '\n';
}

// Run the sweep twice, through the hooks and by full rebuild, and require every
// pressure point to rank identically.
static void expectErsSweepMatchesRebuild(
    StringRef Label, unsigned Stride, unsigned MaxPoints,
    function_ref<void(bool Incremental, UpdateQueryTiming &,
                      SmallVectorImpl<QueryDistances> &)>
        RunSweep) {
  UpdateQueryTiming Timings;
  SmallVector<QueryDistances, 8> IncPerPoint, FullPerPoint;

  ASSERT_NO_FATAL_FAILURE(RunSweep(/*Incremental*/ true, Timings, IncPerPoint));
  if (testing::Test::IsSkipped() || IncPerPoint.empty())
    return;

  ASSERT_NO_FATAL_FAILURE(
      RunSweep(/*Incremental*/ false, Timings, FullPerPoint));
  if (testing::Test::IsSkipped())
    return;

  ASSERT_EQ(IncPerPoint.size(), FullPerPoint.size());
  for (unsigned I = 0, E = IncPerPoint.size(); I != E; ++I) {
    SCOPED_TRACE("pressure point " + std::to_string(I));
    const QueryDistances &Inc = IncPerPoint[I];
    const QueryDistances &Full = FullPerPoint[I];
    EXPECT_EQ(Inc.size(), Full.size());
    for (const auto &KV : Inc) {
      SCOPED_TRACE("vreg " + std::to_string(KV.first));
      auto It = Full.find(KV.first);
      ASSERT_NE(It, Full.end());
      EXPECT_EQ(KV.second, It->second);
    }
  }

  reportUpdateQueryTiming((Label + "(" + Twine(IncPerPoint.size()) +
                           "points,stride" + Twine(Stride) + ")")
                              .str(),
                          Timings);
  (void)MaxPoints;
}

//===----------------------------------------------------------------------===//
// Read-only usage pattern
//
// What amdgpu-next-use-printer does, and what any pass that only consumes the
// analysis does: build once, then sweep every instruction asking for the
// furthest next use among the registers live there, with no mutation and so no
// notification. Timed separately from the build because the two regress for
// different reasons.
//===----------------------------------------------------------------------===//

// One pass of the printer's furthest-distance sweep. \Returns elapsed
// microseconds; \p Queries counts the instructions visited.
static double runFurthestDistanceSweep(const AMDGPUNextUseAnalysis &NUA,
                                       const MachineFunction &MF,
                                       LiveIntervals &LIS, size_t &Queries) {
  GCNDownwardRPTracker RPTracker(LIS);
  AMDGPUNextUseAnalysis::UseDistancePair Furthest, FurthestSubreg;
  DenseMap<const MachineOperand *, AMDGPUNextUseAnalysis::UseDistancePair>
      RelevantUses;

  const auto Start = std::chrono::steady_clock::now();
  for (const MachineBasicBlock &MBB : MF) {
    const MachineInstr *PrevMI = nullptr;
    for (const MachineInstr &MI : MBB) {
      if (!PrevMI || PrevMI->isPHI())
        RPTracker.reset(MI, MBB.end());
      RPTracker.advance();

      RelevantUses.clear();
      NUA.getNextUseDistances(RPTracker.getLiveRegs(), MI, Furthest,
                              &FurthestSubreg, &RelevantUses);
      ++Queries;
      PrevMI = &MI;
    }
  }
  return elapsedMicros(Start);
}

struct MirFuncCase {
  const char *File;
  const char *Func;
};

} // namespace

class NextUseAnalysisIncrementalTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu12.00--", "", ""); }

  bool parseMIRFromPath(StringRef Path) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFile(Path);
    if (!BufOrErr)
      return false;
    return parseMIR(BufOrErr.get()->getBuffer());
  }

  void parseNextUseMIROrSkip(StringRef FileName) {
    SmallString<256> Path(LLVM_AMDGPU_NEXT_USE_MIR_DIR);
    sys::path::append(Path, FileName);
    if (!sys::fs::exists(Path))
      GTEST_SKIP() << "MIR file not found: " << Path;
    ASSERT_TRUE(parseMIRFromPath(Path));
  }

  void parseTestForMIROrSkip() {
    parseNextUseMIROrSkip("simple-loop-3blocks.mir");
  }

  void resetParsedModule() {
    MFAM.clear();
    FAM.clear();
    CGAM.clear();
    MAM.clear();
    Mod.reset();
    MIR.reset();
  }

  static AMDGPUNextUseAnalysis::Config computeConfig() {
    return AMDGPUNextUseAnalysis::Config::Compute();
  }
};

namespace {

static void
runInterBlockReferenceCheck(MirFuncCase Case, AMDGPUNextUseAnalysis::Config Cfg,
                            NextUseAnalysisIncrementalTest &Fixture) {
  Fixture.resetParsedModule();
  Fixture.parseNextUseMIROrSkip(Case.File);
  if (Fixture.IsSkipped())
    return;

  MachineFunction &MF = Fixture.getMF(Case.Func);
  AMDGPUNextUseAnalysis &NUA =
      getNextUseAnalysis(MF, Fixture.MFAM, Cfg, /*Invalidate*/ true);
  const MachineLoopInfo &MLI = Fixture.MFAM.getResult<MachineLoopAnalysis>(MF);
  expectInterBlockDistancesMatchReference(NUA, MF, MLI, Cfg);
}

struct GoldenDigestEntry {
  const char *File;
  const char *Func;
  bool Compute;
  uint64_t Distances;
  uint64_t Reachability;
};

static constexpr bool GoldenDigestsPopulated = true;
static constexpr GoldenDigestEntry GoldenDigests[] = {
    {"acyclic-014bb.mir", "acyclic_014bb", false, 0x1176af95ef110adULL,
     0xaecb6c1cd15527cdULL},
    {"acyclic-014bb.mir", "acyclic_014bb", true, 0x46424941ee1c179dULL,
     0xaecb6c1cd15527cdULL},
    {"complex-control-flow-15blocks.mir", "test3", false, 0x54b67024ef25cf89ULL,
     0x7878373e7d00620dULL},
    {"complex-control-flow-15blocks.mir", "test3", true, 0xaadd7bc05a30fcbdULL,
     0x7878373e7d00620dULL},
    {"double-nested-loops-complex-cfg.mir", "test15", false,
     0xf6940ee249471d64ULL, 0x06fedd3e12c8c5bdULL},
    {"double-nested-loops-complex-cfg.mir", "test15", true,
     0x442de79b672b5a7bULL, 0x06fedd3e12c8c5bdULL},
    {"acyclic-770bb.mir", "acyclic_770bb", false, 0xfa31c411f1e8744bULL,
     0xc7bb20a425cf538dULL},
    {"acyclic-770bb.mir", "acyclic_770bb", true, 0x29d872a6b11b77fbULL,
     0xc7bb20a425cf538dULL},
};

static void runGoldenDigestCheck(const GoldenDigestEntry &Entry,
                                 NextUseAnalysisIncrementalTest &Fixture) {
  AMDGPUNextUseAnalysis::Config Cfg =
      Entry.Compute ? AMDGPUNextUseAnalysis::Config::Compute()
                    : AMDGPUNextUseAnalysis::Config::Graphics();
  Fixture.resetParsedModule();
  Fixture.parseNextUseMIROrSkip(Entry.File);
  if (Fixture.IsSkipped())
    return;

  MachineFunction &MF = Fixture.getMF(Entry.Func);
  AMDGPUNextUseAnalysis &NUA =
      getNextUseAnalysis(MF, Fixture.MFAM, Cfg, /*Invalidate*/ true);

  const uint64_t Distances = digestInterBlockDistances(NUA, MF);
  const uint64_t Reachability = digestReachability(NUA, MF);

  const bool Regen = getenv("NUA_GOLDEN_REGEN") != nullptr;
  if (Regen) {
    llvm::outs() << "  { \"" << Entry.File << "\", \"" << Entry.Func << "\", "
                 << (Entry.Compute ? "true" : "false") << ", 0x"
                 << Twine::utohexstr(Distances) << ", 0x"
                 << Twine::utohexstr(Reachability) << " },\n";
    return;
  }

  if (Distances != Entry.Distances || Reachability != Entry.Reachability) {
    ADD_FAILURE() << Entry.File
                  << " preset=" << (Entry.Compute ? "Compute" : "Graphics")
                  << ": distances expected 0x"
                  << Twine::utohexstr(Entry.Distances) << " got 0x"
                  << Twine::utohexstr(Distances) << "; reachability expected 0x"
                  << Twine::utohexstr(Entry.Reachability) << " got 0x"
                  << Twine::utohexstr(Reachability)
                  << ". If this change is intended, re-generate with "
                     "NUA_GOLDEN_REGEN=1 and justify the diff in the commit "
                     "message.";
  }
}

static constexpr MirFuncCase InterBlockReferenceCorpus[] = {
    {"simple-loop-3blocks.mir", "test4"},
    {"sequence_2_loops.mir", "test6"},
    {"two-sequential-loops.mir", "test7"},
    {"triple-nested-loops.mir", "test9"},
    {"complex-single-loop.mir", "test5"},
    {"complex-control-flow-11blocks.mir", "test2"},
    {"complex-control-flow-15blocks.mir", "test3"},
    {"acyclic-014bb.mir", "acyclic_014bb"},
    {"acyclic-cfg-with-self-loop.mir", "test11"},
    {"complex-acyclic-cfg-with-4-self-loops.mir", "test16"},
    {"inner_cfg_in_2_nested_loops.mir", "test10"},
    {"if_else_with_loops_nested_in_2_outer_loops.mir", "test13"},
    {"loop_nested_in_3_outer_loops_complex_cfg.mir", "test14"},
    {"nested-loops-with-side-exits-a.mir", "test12"},
    {"double-nested-loops-complex-cfg.mir", "test15"},
    {"three-tier-ranking-nested-loops.mir", "test_proper_nested_loops"},
};

} // namespace

TEST_F(NextUseAnalysisIncrementalTest, InterBlockDistanceMatchesReference) {
  for (bool Compute : {false, true}) {
    AMDGPUNextUseAnalysis::Config Cfg =
        Compute ? AMDGPUNextUseAnalysis::Config::Compute()
                : AMDGPUNextUseAnalysis::Config::Graphics();
    for (const MirFuncCase &Case : InterBlockReferenceCorpus) {
      SCOPED_TRACE(Twine(Case.File) + " " + (Compute ? "Compute" : "Graphics"));
      ASSERT_NO_FATAL_FAILURE(runInterBlockReferenceCheck(Case, Cfg, *this));
      if (IsSkipped())
        return;
    }
  }
}

TEST_F(NextUseAnalysisIncrementalTest, InterBlockDistanceGoldenDigest) {
  if (getenv("NUA_GOLDEN_REGEN")) {
    llvm::outs() << "static constexpr GoldenDigestEntry GoldenDigests[] = {\n";
    for (const GoldenDigestEntry &Entry : GoldenDigests)
      runGoldenDigestCheck(Entry, *this);
    llvm::outs() << "};\n";
    return;
  }

  if (!GoldenDigestsPopulated) {
    GTEST_SKIP() << "GoldenDigests not populated; run with NUA_GOLDEN_REGEN=1";
  }

  for (const GoldenDigestEntry &Entry : GoldenDigests)
    runGoldenDigestCheck(Entry, *this);
}

namespace {

static bool dfsBackedge(const MachineLoopInfo &MLI,
                        const MachineBasicBlock *From,
                        const MachineBasicBlock *To) {
  if (MachineLoop *L = MLI.getLoopFor(To))
    if (L->getHeader() == To && L->contains(From))
      return true;
  return false;
}

static bool dfsReachability(const MachineFunction &MF,
                            const MachineLoopInfo &MLI,
                            const MachineBasicBlock *From,
                            const MachineBasicBlock *To, bool ForwardOnly) {
  if (From == To)
    return MLI.getLoopFor(From) != nullptr;

  enum { VisitOp, PopOp };
  using MBBOpPair = std::pair<const MachineBasicBlock *, int>;
  SmallVector<MBBOpPair> Work{{From, VisitOp}};
  DenseSet<const MachineBasicBlock *> Visited{From};

  while (!Work.empty()) {
    auto [Current, Op] = Work.pop_back_val();
    if (Op == PopOp)
      continue;
    if (Current->succ_empty())
      continue;

    for (const MachineBasicBlock *Succ : Current->successors()) {
      if (ForwardOnly && dfsBackedge(MLI, Current, Succ))
        continue;
      if (Succ == To)
        return true;
      if (Visited.insert(Succ).second)
        Work.emplace_back(Succ, VisitOp);
    }
  }
  return false;
}

static void expectReachabilityMatchesDfs(const AMDGPUNextUseAnalysis &NUA,
                                         const MachineFunction &MF,
                                         const MachineLoopInfo &MLI) {
  for (unsigned FromNum = 0, N = MF.getNumBlockIDs(); FromNum != N; ++FromNum) {
    const MachineBasicBlock *From = MF.getBlockNumbered(FromNum);
    if (!From)
      continue;
    for (unsigned ToNum = 0; ToNum != N; ++ToNum) {
      const MachineBasicBlock *To = MF.getBlockNumbered(ToNum);
      if (!To)
        continue;
      SCOPED_TRACE("blocks " + Twine(FromNum) + " -> " + Twine(ToNum));
      EXPECT_EQ(NUA.isReachable(From, To),
                dfsReachability(MF, MLI, From, To, /*ForwardOnly*/ false));
      // Forward reachability uses initializeCfgPaths backedge marks, not
      // MachineLoop header edges, so an independent DFS oracle is not
      // compared here.
    }
  }
}

} // namespace

TEST_F(NextUseAnalysisIncrementalTest, ReachabilityClosureMatchesDfs) {
  for (bool Compute : {false, true}) {
    AMDGPUNextUseAnalysis::Config Cfg =
        Compute ? AMDGPUNextUseAnalysis::Config::Compute()
                : AMDGPUNextUseAnalysis::Config::Graphics();
    for (const MirFuncCase &Case : InterBlockReferenceCorpus) {
      SCOPED_TRACE(Twine(Case.File) + " " + (Compute ? "Compute" : "Graphics"));
      resetParsedModule();
      parseNextUseMIROrSkip(Case.File);
      if (IsSkipped())
        return;
      MachineFunction &MF = getMF(Case.Func);
      AMDGPUNextUseAnalysis &NUA =
          getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);
      const MachineLoopInfo &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
      expectReachabilityMatchesDfs(NUA, MF, MLI);
    }
  }
}

TEST_F(NextUseAnalysisIncrementalTest,
       ForwardReachabilityMatchesUnrestrictedOnAcyclicDag) {
  for (bool Compute : {false, true}) {
    AMDGPUNextUseAnalysis::Config Cfg =
        Compute ? AMDGPUNextUseAnalysis::Config::Compute()
                : AMDGPUNextUseAnalysis::Config::Graphics();
    resetParsedModule();
    parseNextUseMIROrSkip("acyclic-014bb.mir");
    if (IsSkipped())
      return;
    MachineFunction &MF = getMF("acyclic_014bb");
    AMDGPUNextUseAnalysis &NUA =
        getNextUseAnalysis(MF, MFAM, Cfg, /*Invalidate*/ true);

    for (unsigned FromNum = 0, N = MF.getNumBlockIDs(); FromNum != N;
         ++FromNum) {
      const MachineBasicBlock *From = MF.getBlockNumbered(FromNum);
      if (!From)
        continue;
      for (unsigned ToNum = 0; ToNum != N; ++ToNum) {
        const MachineBasicBlock *To = MF.getBlockNumbered(ToNum);
        if (!To)
          continue;
        SCOPED_TRACE("blocks " + Twine(FromNum) + " -> " + Twine(ToNum));
        EXPECT_EQ(NUA.isForwardReachable(From, To), NUA.isReachable(From, To))
            << "acyclic DAG has no backedges; forward and unrestricted "
               "reachability must agree";
      }
    }
  }
}

TEST_F(NextUseAnalysisIncrementalTest, ReachabilityUnchangedByInstrAdded) {
  parseTestForMIROrSkip();
  if (IsSkipped())
    return;
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);

  AMDGPUNextUseAnalysis &NUA =
      getNextUseAnalysis(MF, MFAM, computeConfig(), /*Invalidate*/ true);
  const uint64_t Before = digestReachability(NUA, MF);
  insertNop(MBB1, std::next(MBB1.getFirstNonPHI()));
  NUA.instrAdded(MBB1);
  const uint64_t After = digestReachability(NUA, MF);
  EXPECT_EQ(Before, After);
}

// Track the cost of the read-only pattern on the 770-block scaling benchmark,
// so that work added for the mutating clients does not quietly tax the passes
// that only read. Reports rather than asserts a wall-clock budget: the figures
// are for trending, and a machine-dependent threshold would be flaky. The one
// assertion is machine-independent -- a warm repeat of the same sweep over an
// unmodified function must not cost more than the cold one, which fails if the
// lazily populated path and distance caches stop being reused.
//
// The reported "total" is the sum of every phase, meant as the single number
// to compare between runs. For reference, over five runs on one machine it sat
// at 186.5-190.9ms, a spread of 2.3%, made up of build 24.8-26.2ms, sweep-cold
// 104.0-107.4ms, sweep-warm 56.1-58.0ms and def-to-use 0.39-0.41ms, for 2707
// queries over 770 blocks. The cold-to-warm ratio is the invariant, and it sat
// at 1.8-1.9x throughout.
TEST_F(NextUseAnalysisIncrementalTest, ReadOnlySweepAcyclic770bb) {
  parseNextUseMIROrSkip("acyclic-770bb.mir");
  MachineFunction &MF = getMF("acyclic_770bb");

  double BuildUs = 0;
  AMDGPUNextUseAnalysis *NUAPtr = nullptr;
  timeMicros(BuildUs, [&] {
    NUAPtr =
        &getNextUseAnalysis(MF, MFAM, computeConfig(), /*Invalidate*/ true);
  });
  AMDGPUNextUseAnalysis &NUA = *NUAPtr;

  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  size_t ColdQueries = 0, WarmQueries = 0;
  const double ColdUs = runFurthestDistanceSweep(NUA, MF, LIS, ColdQueries);
  const double WarmUs = runFurthestDistanceSweep(NUA, MF, LIS, WarmQueries);
  ASSERT_EQ(ColdQueries, WarmQueries);
  ASSERT_GT(ColdQueries, 0u);

  NextUseSnapshot Snap;
  double DefUseUs = 0;
  timeMicros(DefUseUs, [&] { captureNextUseSnapshot(NUA, MF, Snap); });
  EXPECT_FALSE(Snap.DefRegToClosestUseDist.empty());

  const double TotalUs = BuildUs + ColdUs + WarmUs + DefUseUs;

  const auto Ms = [](double Us) { return formatv("{0:F3}", Us / 1000.0); };
  llvm::outs() << "[ >NUA r/o ] total=" << Ms(TotalUs)
               << "ms blocks=" << MF.size() << " queries=" << ColdQueries
               << " build=" << Ms(BuildUs) << "ms sweep-cold=" << Ms(ColdUs)
               << "ms sweep-warm=" << Ms(WarmUs)
               << "ms def-to-use=" << Ms(DefUseUs) << "ms\n";

  RecordProperty("read_only.total_ms", Ms(TotalUs).str());
  RecordProperty("read_only.build_ms", Ms(BuildUs).str());
  RecordProperty("read_only.sweep_cold_ms", Ms(ColdUs).str());
  RecordProperty("read_only.sweep_warm_ms", Ms(WarmUs).str());
  RecordProperty("read_only.def_to_use_ms", Ms(DefUseUs).str());
  RecordProperty("read_only.queries", std::to_string(ColdQueries));

  EXPECT_LE(WarmUs, ColdUs);
}

TEST_F(NextUseAnalysisIncrementalTest, ErsSweepQueryPattern) {
  constexpr unsigned Stride = 3;
  constexpr unsigned MaxPoints = 8;

  auto RunSweep = [&](bool Incremental, UpdateQueryTiming &Timings,
                      SmallVectorImpl<QueryDistances> &PerPoint) {
    resetParsedModule();
    parseTestForMIROrSkip();
    if (IsSkipped())
      return;
    MachineFunction &MF = getMF("test4");
    LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
    const MachineLoopInfo &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
    ASSERT_NO_FATAL_FAILURE(runErsSweep(MF, MFAM, computeConfig(), LIS, MLI,
                                        Incremental, Stride, MaxPoints, Timings,
                                        PerPoint));
  };

  expectErsSweepMatchesRebuild("ErsSweepQueryPattern", Stride, MaxPoints,
                               RunSweep);
}

TEST_F(NextUseAnalysisIncrementalTest, NoChangeInstrAddedMatchesFresh) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  expectInstrAddedMatchesFreshRecompute(MF, MFAM, computeConfig(), MBB0,
                                        [](MachineFunction &) {});
}

TEST_F(NextUseAnalysisIncrementalTest, NopInEntryBlock) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  expectInstrAddedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), MBB0,
      [&](MachineFunction &) { insertNop(MBB0, MBB0.getFirstTerminator()); });
}

TEST_F(NextUseAnalysisIncrementalTest, NopInLoopBody) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  expectInstrAddedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), MBB1, [&](MachineFunction &) {
        insertNop(MBB1, std::next(MBB1.getFirstNonPHI()));
      });
}

TEST_F(NextUseAnalysisIncrementalTest, NopInExitBlock) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB2 = *MF.getBlockNumbered(2);
  expectInstrAddedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), MBB2,
      [&](MachineFunction &) { insertNop(MBB2, MBB2.getFirstNonPHI()); });
}

TEST_F(NextUseAnalysisIncrementalTest, TwoNopsInLoopBody) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  expectInstrAddedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), MBB1, [&](MachineFunction &) {
        auto InsertPt = std::next(MBB1.getFirstNonPHI());
        insertNop(MBB1, InsertPt);
        insertNop(MBB1, InsertPt);
      });
}

TEST_F(NextUseAnalysisIncrementalTest, NopInEntryBlockGraphicsConfig) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  expectInstrAddedMatchesFreshRecompute(
      MF, MFAM, AMDGPUNextUseAnalysis::Config::Graphics(), MBB0,
      [&](MachineFunction &) { insertNop(MBB0, MBB0.getFirstTerminator()); });
}

TEST_F(NextUseAnalysisIncrementalTest, AfterDistanceCacheQuery) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);

  AMDGPUNextUseAnalysis &Inc =
      getNextUseAnalysis(MF, MFAM, computeConfig(), /*Invalidate*/ true);
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  GCNDownwardRPTracker RPTracker(LIS);
  AMDGPUNextUseAnalysis::UseDistancePair Furthest;
  for (const MachineInstr &MI : MBB0) {
    RPTracker.reset(MI, MBB0.end());
    RPTracker.advance();
    Inc.getNextUseDistances(RPTracker.getLiveRegs(), MI, Furthest);
  }

  UpdateQueryTiming Timings;
  timeMicros(Timings.IncrementalUpdateUs, [&] {
    insertNop(MBB0, MBB0.getFirstTerminator());
    Inc.instrAdded(MBB0);
  });
  NextUseSnapshot AfterIncremental;
  timeMicros(Timings.IncrementalQueryUs, [&] { //
    captureNextUseSnapshot(Inc, MF, AfterIncremental);
  });
  AMDGPUNextUseAnalysis *FreshPtr = nullptr;
  timeMicros(Timings.FullRebuildUs, [&] {
    FreshPtr = &getNextUseAnalysis(MF, MFAM, computeConfig(),
                                   /*Invalidate*/ true);
  });
  AMDGPUNextUseAnalysis &Fresh = *FreshPtr;
  NextUseSnapshot FreshSnap;
  timeMicros(Timings.FullQueryUs, [&] { //
    captureNextUseSnapshot(Fresh, MF, FreshSnap);
  });

  reportUpdateQueryTiming("AfterDistanceCacheQuery", Timings);
  expectSnapshotsEqual(AfterIncremental, FreshSnap);
}

TEST_F(NextUseAnalysisIncrementalTest, TimingReportLoopBodyNop) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);

  constexpr unsigned Iterations = 32;
  UpdateQueryTiming Accumulated;

  for (unsigned I = 0; I != Iterations; ++I) {
    AMDGPUNextUseAnalysis &Inc =
        getNextUseAnalysis(MF, MFAM, computeConfig(), /*Invalidate*/ true);
    insertNop(MBB1, std::next(MBB1.getFirstNonPHI()));

    UpdateQueryTiming Timings;
    timeMicros(Timings.IncrementalUpdateUs, [&] { //
      Inc.instrAdded(MBB1);
    });
    timeMicros(Timings.IncrementalQueryUs, [&] {
      NextUseSnapshot Unused;
      captureNextUseSnapshot(Inc, MF, Unused);
    });
    AMDGPUNextUseAnalysis *FreshPtr = nullptr;
    timeMicros(Timings.FullRebuildUs, [&] {
      FreshPtr = &getNextUseAnalysis(MF, MFAM, computeConfig(),
                                     /*Invalidate*/ true);
    });
    AMDGPUNextUseAnalysis &Fresh = *FreshPtr;
    timeMicros(Timings.FullQueryUs, [&] {
      NextUseSnapshot Unused;
      captureNextUseSnapshot(Fresh, MF, Unused);
    });

    Accumulated.IncrementalUpdateUs += Timings.IncrementalUpdateUs;
    Accumulated.IncrementalQueryUs += Timings.IncrementalQueryUs;
    Accumulated.FullRebuildUs += Timings.FullRebuildUs;
    Accumulated.FullQueryUs += Timings.FullQueryUs;

    MachineInstr &Nop = *std::prev(MBB1.getFirstTerminator());
    removeMI(Fresh, Nop);
  }

  Accumulated.IncrementalUpdateUs /= Iterations;
  Accumulated.IncrementalQueryUs /= Iterations;
  Accumulated.FullRebuildUs /= Iterations;
  Accumulated.FullQueryUs /= Iterations;
  reportUpdateQueryTiming("TimingReportLoopBodyNop(avg32)", Accumulated);
}

TEST_F(NextUseAnalysisIncrementalTest, RemoveNopFromEntryBlock) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  expectInstrRemovedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), MBB0,
      [&](AMDGPUNextUseAnalysis &NUA, MachineFunction &) {
        insertNop(MBB0, MBB0.getFirstTerminator());
        MachineInstr &Nop = *std::prev(MBB0.getFirstTerminator());
        removeMI(NUA, Nop);
      });
}

TEST_F(NextUseAnalysisIncrementalTest, RemoveInstrFromLoopBody) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  expectInstrRemovedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), MBB1,
      [&](AMDGPUNextUseAnalysis &NUA, MachineFunction &) {
        MachineInstr &MI = *std::next(MBB1.getFirstNonPHI());
        removeMI(NUA, MI);
      });
}

TEST_F(NextUseAnalysisIncrementalTest, RewriteUseInLoopBody) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // %16 = V_LSHL_OR_B32_e64 %13, 8, %12 -- repoint the %13 use at %12.
  Register V12 = Register::index2VirtReg(12);
  Register V13 = Register::index2VirtReg(13);
  MachineInstr *MI = MRI.getVRegDef(Register::index2VirtReg(16));
  ASSERT_NE(MI, nullptr);

  expectRegUsesChangedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), "RewriteUseInLoopBody",
      [&](MachineFunction &, AMDGPUNextUseAnalysis &NUA) {
        rewriteUsesInInstr(NUA, *MI, V13, V12);
      });
}

TEST_F(NextUseAnalysisIncrementalTest, RewritePhiIncomingOperand) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // %6 = PHI %1, %bb.0, %7, %bb.1 -- repoint the bb.0 incoming value at %2.
  Register V1 = Register::index2VirtReg(1);
  Register V2 = Register::index2VirtReg(2);
  MachineInstr *Phi = MRI.getVRegDef(Register::index2VirtReg(6));
  ASSERT_NE(Phi, nullptr);
  ASSERT_TRUE(Phi->isPHI());

  expectRegUsesChangedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), "RewritePhiIncomingOperand",
      [&](MachineFunction &, AMDGPUNextUseAnalysis &NUA) {
        rewriteUsesInInstr(NUA, *Phi, V1, V2);
      });
}

// The hazardous direction: a register whose cached use list is empty gains a
// use. Without notification that empty list reads as "never used again".
TEST_F(NextUseAnalysisIncrementalTest, RewriteUseToPreviouslyUnusedReg) {
  StringRef MIR = R"MIR(
--- |
  define amdgpu_ps void @unused_def() {
  ret void
  }
...
---
name:            unused_def
tracksRegLiveness: true
body:             |
  bb.0:
    successors: %bb.1(0x80000000)
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = V_MOV_B32_e32 0, implicit $exec

  bb.1:
    %2:vgpr_32 = V_ADD_U32_e64 %1, %0, 0, implicit $exec
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)MIR";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("unused_def");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  Register V0 = Register::index2VirtReg(0);
  Register V2 = Register::index2VirtReg(2);
  ASSERT_TRUE(MRI.use_nodbg_empty(V2));
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  MachineInstr *Nop = &*std::next(MBB1.begin());
  ASSERT_EQ(Nop->getOpcode(), AMDGPU::S_NOP);

  expectRegUsesChangedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), "RewriteUseToPreviouslyUnusedReg",
      [&](MachineFunction &, AMDGPUNextUseAnalysis &NUA) {
        rewriteUsesInInstr(NUA, *Nop, V0, V2);
      });
}

// Without the notification the same rewrite leaves the cache stale, which
// verifyRegUseCache is there to catch.
TEST_F(NextUseAnalysisIncrementalTest, MissedRegUsesChangedLeavesCacheStale) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  AMDGPUNextUseAnalysis &Inc =
      getNextUseAnalysis(MF, MFAM, computeConfig(), /*Invalidate*/ true);
  NextUseSnapshot WarmSnap;
  captureNextUseSnapshot(Inc, MF, WarmSnap);
  ASSERT_TRUE(Inc.verifyRegUseCache());

  MachineInstr *MI = MRI.getVRegDef(Register::index2VirtReg(16));
  ASSERT_NE(MI, nullptr);
  for (MachineOperand &MO : MI->uses()) {
    if (MO.isReg() && MO.getReg() == Register::index2VirtReg(13))
      MO.setReg(Register::index2VirtReg(12));
  }

  EXPECT_FALSE(Inc.verifyRegUseCache());
}

// Growing an operand list reallocates MachineInstr's operand array, so every
// cached pointer into that instruction has to be dropped first.
TEST_F(NextUseAnalysisIncrementalTest, AddOperandToInstr) {
  parseTestForMIROrSkip();
  MachineFunction &MF = getMF("test4");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  MachineInstr *MI = MRI.getVRegDef(Register::index2VirtReg(16));
  ASSERT_NE(MI, nullptr);

  expectRegUsesChangedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), "AddOperandToInstr",
      [&](MachineFunction &MFRef, AMDGPUNextUseAnalysis &NUA) {
        NUA.instrOperandsChanging(*MI);
        MachineInstrBuilder(MFRef, MI).addReg(Register::index2VirtReg(14),
                                              RegState::Implicit);
        NUA.instrOperandsChanged(*MI);
      });
}

TEST_F(NextUseAnalysisIncrementalTest, SimpleAcyclicEmbeddedMIR) {
  StringRef MIR = R"MIR(
--- |
  define amdgpu_ps void @simple_acyclic() {
  ret void
  }
...
---
name:            simple_acyclic
tracksRegLiveness: true
body:             |
  bb.0:
    successors: %bb.1(0x80000000)
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = V_MOV_B32_e32 0, implicit $exec

  bb.1:
    %2:vgpr_32 = V_ADD_U32_e64 %1, %0, 0, implicit $exec
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)MIR";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("simple_acyclic");
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  expectInstrAddedMatchesFreshRecompute(
      MF, MFAM, computeConfig(), MBB0,
      [&](MachineFunction &) { insertNop(MBB0, MBB0.getFirstTerminator()); });
}
