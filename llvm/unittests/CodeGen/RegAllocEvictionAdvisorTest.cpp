#include "llvm/CodeGen/RegAllocEvictionAdvisor.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalUnion.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
#include "MFCommon.inc"

class CacheAccess : public RegAllocEvictionAdvisor {
public:
  using RegAllocEvictionAdvisor::ReassignmentCache;
};

using Cache = CacheAccess::ReassignmentCache;

class ReassignmentCacheTest : public testing::Test {
protected:
  LLVMContext Context;
  Module Mod{"reassignment-cache", Context};
  std::unique_ptr<MachineFunction> MF = createMachineFunction(Context, Mod);
  std::unique_ptr<SlotIndexes> Indexes;
  SmallVector<SlotIndex, 6> Positions;
  BumpPtrAllocator Values;
  LiveIntervalUnion::Allocator Allocator;
  LiveIntervalUnion Union{Allocator};
  Cache Blockers;
  unsigned NextReg = 0;

  void expectInlineStorage() {
    uintptr_t Begin = reinterpret_cast<uintptr_t>(&Blockers.Blockers);
    uintptr_t Buckets = reinterpret_cast<uintptr_t>(
        Blockers.Blockers.getPointerIntoBucketsArray());
    EXPECT_GE(Buckets, Begin);
    EXPECT_LT(Buckets, Begin + sizeof(Blockers.Blockers));
  }

  void SetUp() override {
    for (unsigned Count = 0; Count != 6; ++Count)
      MF->push_back(MF->CreateMachineBasicBlock());
    Indexes = std::make_unique<SlotIndexes>(*MF);
    for (auto &Block : *MF)
      Positions.push_back(Indexes->getMBBStartIdx(&Block));
  }

  std::unique_ptr<LiveInterval> interval(unsigned Begin, unsigned End) {
    auto Result = std::make_unique<LiveInterval>(
        Register::index2VirtReg(NextReg++), 1.0f);
    auto *Value = Result->getNextValue(Positions[Begin], Values);
    Result->addSegment({Positions[Begin], Positions[End], Value});
    return Result;
  }

  Cache::InterferenceKind check(const LiveInterval &Range,
                                unsigned Unit = 827) {
    auto Kind = Blockers.checkInterference(Range, MCRegUnit(Unit), Union);
    EXPECT_EQ(Kind != Cache::Free,
              LiveIntervalUnion::Query(Range, Union).checkInterference());
    return Kind;
  }
};

TEST_F(ReassignmentCacheTest, FreeSearchDoesNotAllocateOrPopulate) {
  auto Range = interval(0, 5);
  expectInlineStorage();
  const void *InlineBuckets = Blockers.Blockers.getPointerIntoBucketsArray();
  for (unsigned Unit = 0; Unit != 128; ++Unit)
    EXPECT_EQ(Cache::Free, check(*Range, Unit));
  EXPECT_TRUE(Blockers.Blockers.empty());
  EXPECT_EQ(InlineBuckets, Blockers.Blockers.getPointerIntoBucketsArray());
}

TEST_F(ReassignmentCacheTest,
       ShortSearchUsesInlineStorageAndReusesAcrossVRegs) {
  auto Owner = interval(1, 4);
  Union.unify(*Owner, *Owner);
  auto First = interval(2, 3);
  auto Second = interval(2, 4);
  auto Free = interval(4, 5);
  const void *InlineBuckets = Blockers.Blockers.getPointerIntoBucketsArray();
  EXPECT_EQ(Cache::Blocked, check(*First, 827));
  EXPECT_EQ(Cache::Cached, check(*Second, 827));
  EXPECT_EQ(Cache::Blocked, check(*Second, 828));
  EXPECT_EQ(Cache::Free, check(*Free, 829));
  EXPECT_EQ(2u, Blockers.Blockers.size());
  expectInlineStorage();
  EXPECT_EQ(InlineBuckets, Blockers.Blockers.getPointerIntoBucketsArray());
}

TEST_F(ReassignmentCacheTest, HalfOpenBoundariesAndContainment) {
  auto Owner = interval(2, 4);
  Union.unify(*Owner, *Owner);
  auto Before = interval(0, 2);
  auto After = interval(4, 5);
  auto Equal = interval(2, 4);
  auto Inside = interval(2, 3);
  auto Enclosing = interval(1, 5);
  EXPECT_EQ(Cache::Free, check(*Before));
  EXPECT_EQ(Cache::Free, check(*After));
  EXPECT_TRUE(Blockers.Blockers.empty());
  EXPECT_EQ(Cache::Blocked, check(*Equal));
  EXPECT_EQ(Cache::Free, check(*Before));
  EXPECT_EQ(Cache::Free, check(*After));
  EXPECT_EQ(Cache::Cached, check(*Inside));
  EXPECT_EQ(Cache::Cached, check(*Enclosing));
}

TEST_F(ReassignmentCacheTest, RangeMissReplacesBlocker) {
  auto FirstOwner = interval(0, 1);
  auto SecondOwner = interval(3, 5);
  Union.unify(*FirstOwner, *FirstOwner);
  Union.unify(*SecondOwner, *SecondOwner);
  auto First = interval(0, 1);
  auto Gap = interval(1, 3);
  auto Second = interval(3, 4);
  EXPECT_EQ(Cache::Blocked, check(*First));
  EXPECT_EQ(Cache::Free, check(*Gap));
  EXPECT_EQ(Cache::Blocked, check(*Second));
  EXPECT_EQ(Cache::Cached, check(*Second));
  EXPECT_EQ(1u, Blockers.Blockers.size());
}

TEST_F(ReassignmentCacheTest, UnionTagInvalidatesPositiveEntry) {
  auto Owner = interval(1, 4);
  auto Range = interval(2, 3);
  Union.unify(*Owner, *Owner);
  EXPECT_EQ(Cache::Blocked, check(*Range));
  Union.extract(*Owner, *Owner);
  EXPECT_EQ(Cache::Free, check(*Range));
  Union.unify(*Owner, *Owner);
  EXPECT_EQ(Cache::Blocked, check(*Range));
  EXPECT_EQ(Cache::Cached, check(*Range));
}

TEST_F(ReassignmentCacheTest, GrowthPreservesSparseEntries) {
  auto Owner = interval(1, 4);
  auto Range = interval(2, 3);
  Union.unify(*Owner, *Owner);
  for (unsigned Unit = 0; Unit != 128; ++Unit)
    EXPECT_EQ(Cache::Blocked, check(*Range, Unit * 7));
  EXPECT_EQ(128u, Blockers.Blockers.size());
  for (unsigned Unit = 0; Unit != 128; ++Unit)
    EXPECT_EQ(Cache::Cached, check(*Range, Unit * 7));
}

} // namespace