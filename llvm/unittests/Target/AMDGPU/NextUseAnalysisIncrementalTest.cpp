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

// Step 2 of proposal-interblock-distances replaces the reachability oracle. It
// must leave every digest below unchanged. A digest change during step 2 means
// the closure disagrees with calcIsReachable and is a bug, not a golden-file
// update.
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
