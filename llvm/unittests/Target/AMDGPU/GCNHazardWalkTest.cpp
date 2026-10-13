//===- GCNHazardWalkTest.cpp - GCNHazardWalk tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNHazardWalk.h"
#include "AMDGPUUnitTests.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "gtest/gtest.h"
#include <algorithm>

using namespace llvm;

namespace {

// Steps as (name, distance): V_MOV_B32_e32 N is "vN", anything else its
// opcode name.
using Steps = std::vector<std::pair<std::string, int>>;

class GCNHazardWalkTest : public AMDGPUCodeGenTestBase {
protected:
  void SetUp() override { setUpImpl("amdgcn-amd-amdhsa", "gfx90a", ""); }

  // Parses \p Body, the blocks of a function named f, and returns the
  // V_MOV_B32_e32 99 the walks start from.
  const MachineInstr &parse(StringRef Body) {
    parseModule("---\nname: f\nbody: |\n" + Body.str());
    return find(getMF("f"), 99);
  }

  // As parse(), for two functions f and g in one module: parsing a second
  // module would leave the analysis of the first one cached.
  std::pair<const MachineInstr *, const MachineInstr *>
  parseBoth(StringRef BodyF, StringRef BodyG) {
    parseModule("---\nname: f\nbody: |\n" + BodyF.str() +
                "---\nname: g\nbody: |\n" + BodyG.str());
    return {&find(getMF("f"), 99), &find(getMF("g"), 99)};
  }

  void parseModule(const std::string &MIR) {
    bool Parsed = parseMIR(MIR);
    EXPECT_TRUE(Parsed);
    if (!Parsed)
      report_fatal_error("cannot parse the test module");
  }

  static const MachineInstr &findOpcode(const MachineFunction &MF,
                                        unsigned Opcode) {
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB.instrs())
        if (MI.getOpcode() == Opcode)
          return MI;
    llvm_unreachable("no such instruction");
  }

  static const MachineInstr &find(const MachineFunction &MF, int64_t Imm) {
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB.instrs())
        if (MI.getOpcode() == AMDGPU::V_MOV_B32_e32 &&
            MI.getOperand(1).getImm() == Imm)
          return MI;
    llvm_unreachable("no such instruction");
  }

  static std::string name(const MachineInstr &MI) {
    if (MI.getOpcode() == AMDGPU::V_MOV_B32_e32)
      return "v" + std::to_string(MI.getOperand(1).getImm());
    const TargetInstrInfo &TII = *MI.getMF()->getSubtarget().getInstrInfo();
    return TII.getName(MI.getOpcode()).str();
  }

  // Runs \p Walk to the end, ending the path at each instruction named in
  // \p EndsPath, and checks that no distance is smaller than one before it.
  static Steps run(GCNHazardWalk Walk, ArrayRef<StringRef> EndsPath = {}) {
    Steps Result;
    for (auto [MI, Distance] : Walk) {
      if (!Result.empty())
        EXPECT_LE(Result.back().second, Distance);
      Result.emplace_back(name(MI), Distance);
      if (is_contained(EndsPath, name(MI)))
        Walk.endPath();
    }
    return Result;
  }

  // run() with ties in distance put in name order.
  static Steps sorted(Steps S) {
    llvm::sort(S, [](const auto &A, const auto &B) {
      return std::tie(A.second, A.first) < std::tie(B.second, B.first);
    });
    return S;
  }
};

TEST_F(GCNHazardWalkTest, StraightLine) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_NOP 2
    $vgpr2 = V_MOV_B32_e32 2, implicit $exec
    INLINEASM &"", 1
    KILL $vgpr2
    $vgpr3 = V_MOV_B32_e32 3, implicit $exec
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  Steps All = {{"v3", 0}, {"KILL", 1},  {"INLINEASM", 1},
               {"v2", 1}, {"S_NOP", 2}, {"v1", 5}};
  EXPECT_EQ(run(GCNHazardWalk(Start, 100)), All);
  EXPECT_EQ(run(GCNHazardWalk(Start, 2)), Steps(All.begin(), All.end() - 2));
  EXPECT_EQ(run(GCNHazardWalk(Start, 0)), Steps());
  EXPECT_EQ(run(GCNHazardWalk(Start, 100), {"KILL"}),
            Steps({{"v3", 0}, {"KILL", 1}}));

  // Narrowing the bound within a block stops the walk there, too.
  GCNHazardWalk Walk(Start, 100);
  Steps Result;
  for (auto [MI, Distance] : Walk) {
    Result.emplace_back(name(MI), Distance);
    if (name(MI) == "v3")
      Walk.narrowBound(2);
  }
  EXPECT_EQ(Result, Steps(All.begin(), All.end() - 2));
}

TEST_F(GCNHazardWalkTest, Bundle) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    BUNDLE implicit-def $vgpr2, implicit-def $vgpr3 {
      $vgpr2 = V_MOV_B32_e32 2, implicit $exec
      $vgpr3 = V_MOV_B32_e32 3, implicit $exec
    }
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  EXPECT_EQ(run(GCNHazardWalk(Start, 100)),
            Steps({{"v3", 0}, {"v2", 1}, {"v1", 2}}));
}

// Both arms branch explicitly, so the two layouts hold the same instructions
// and must give the same distances. The short arm reaches the entry block on
// its conditional branch; the long arm executes the unconditional one too and
// reaches the conditional branch only after the short arm has.
constexpr const char *DiamondLongFirst = R"(
  bb.0:
    successors: %bb.1, %bb.2
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_CBRANCH_SCC1 %bb.2, implicit undef $scc
    S_BRANCH %bb.1
  bb.1:
    successors: %bb.3
    S_NOP 4
    S_BRANCH %bb.3
  bb.2:
    successors: %bb.3
    $vgpr2 = V_MOV_B32_e32 2, implicit $exec
    S_BRANCH %bb.3
  bb.3:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)";

constexpr const char *DiamondShortFirst = R"(
  bb.0:
    successors: %bb.2, %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
    S_BRANCH %bb.2
  bb.1:
    successors: %bb.3
    $vgpr2 = V_MOV_B32_e32 2, implicit $exec
    S_BRANCH %bb.3
  bb.2:
    successors: %bb.3
    S_NOP 4
    S_BRANCH %bb.3
  bb.3:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)";

TEST_F(GCNHazardWalkTest, DiamondLayouts) {
  Steps Expected = {{"S_BRANCH", 0}, {"S_BRANCH", 0},       {"S_NOP", 1},
                    {"v2", 1},       {"S_CBRANCH_SCC1", 2}, {"v1", 3},
                    {"S_BRANCH", 6}};
  auto [LongFirst, ShortFirst] = parseBoth(DiamondLongFirst, DiamondShortFirst);
  EXPECT_EQ(sorted(run(GCNHazardWalk(*LongFirst, 100))), Expected);
  EXPECT_EQ(sorted(run(GCNHazardWalk(*ShortFirst, 100))), Expected);
}

// The defect the walk removes: a search that keeps the first path to reach a
// block measures v1 over the long arm in one layout. Here v1 comes first at
// its distance over the short arm, 3, in both layouts, and so does its
// deficit. The conditional branch is a step like any other, so it can be a
// producer.
TEST_F(GCNHazardWalkTest, ShortestPathInEitherLayout) {
  auto V1 = [](const MachineInstr &MI) -> std::optional<int> {
    return name(MI) == "v1" ? std::optional<int>(8) : std::nullopt;
  };
  auto Branch = [](const MachineInstr &MI) -> std::optional<int> {
    return name(MI) == "S_CBRANCH_SCC1" ? std::optional<int>(8) : std::nullopt;
  };
  auto [LongFirst, ShortFirst] = parseBoth(DiamondLongFirst, DiamondShortFirst);
  for (const MachineInstr *Start : {LongFirst, ShortFirst}) {
    std::optional<int> Nearest;
    for (auto [MI, Distance] : GCNHazardWalk(*Start, 100)) {
      if (name(MI) == "v1") {
        Nearest = Distance;
        break;
      }
    }
    EXPECT_EQ(Nearest, 3);
    EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(*Start, 8), V1), 8 - 3);
    EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(*Start, 8), Branch), 8 - 2);
  }
}

// A meta instruction is not a hazard, even where the predicate accepts it.
TEST_F(GCNHazardWalkTest, QueriesIgnoreMeta) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    $vgpr2 = V_MOV_B32_e32 2, implicit $exec
    KILL $vgpr1
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  auto UsesV1 = [](const MachineInstr &MI) {
    return name(MI) == "KILL" || name(MI) == "v1";
  };
  auto WindowFor = [&UsesV1](const MachineInstr &MI) -> std::optional<int> {
    if (!UsesV1(MI))
      return std::nullopt;
    return 4;
  };
  EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(Start, 4), WindowFor), 3);
}

// Control can come from one predecessor in more than one place: through the
// taken conditional branch, or past it to the unconditional one (f), or
// falling through as well (g, one place).
TEST_F(GCNHazardWalkTest, SeveralTransfersToOneSuccessor) {
  auto [TwoBranches, BranchAndFallthrough] = parseBoth(R"(
  bb.0:
    successors: %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
    S_BRANCH %bb.1
  bb.1:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)",
                                                       R"(
  bb.0:
    successors: %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
  bb.1:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  EXPECT_EQ(sorted(run(GCNHazardWalk(*TwoBranches, 100))),
            Steps({{"S_BRANCH", 0}, {"S_CBRANCH_SCC1", 0}, {"v1", 1}}));
  EXPECT_EQ(sorted(run(GCNHazardWalk(*BranchAndFallthrough, 100))),
            Steps({{"S_CBRANCH_SCC1", 0}, {"v1", 1}}));
}

// Two branches to bb.1 and a fallthrough into it: falling through executes
// both branches, so it adds no entry beyond the last one.
TEST_F(GCNHazardWalkTest, TwoBranchesAndFallthrough) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    successors: %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
    S_CBRANCH_VCCZ %bb.1, implicit undef $vcc
  bb.1:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  EXPECT_EQ(sorted(run(GCNHazardWalk(Start, 100))),
            Steps({{"S_CBRANCH_SCC1", 0}, {"S_CBRANCH_VCCZ", 0}, {"v1", 1}}));
}

// Control reaches bb.1 through the first branch, or past both branches by
// falling through.
TEST_F(GCNHazardWalkTest, BranchThenFallthrough) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    successors: %bb.1, %bb.2
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
    S_CBRANCH_VCCZ %bb.2, implicit undef $vcc
  bb.1:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
  bb.2:
    S_ENDPGM 0
)");
  EXPECT_EQ(sorted(run(GCNHazardWalk(Start, 100))),
            Steps({{"S_CBRANCH_SCC1", 0}, {"S_CBRANCH_VCCZ", 0}, {"v1", 1}}));
}

// From the second terminator of a self loop the first is nearest, and the
// backedge stops where the walk began. From the first, the backedge leads
// back to the start itself.
TEST_F(GCNHazardWalkTest, StartIsTerminator) {
  parseModule("---\nname: f\nbody: |\n" + std::string(R"(
  bb.0:
    successors: %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
  bb.1:
    successors: %bb.1, %bb.2
    $vgpr2 = V_MOV_B32_e32 2, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
    S_BRANCH %bb.2
  bb.2:
    S_ENDPGM 0
)"));
  const MachineInstr &CBranch = findOpcode(getMF("f"), AMDGPU::S_CBRANCH_SCC1);
  EXPECT_EQ(sorted(run(GCNHazardWalk(*CBranch.getNextNode(), 100))),
            Steps({{"S_CBRANCH_SCC1", 0}, {"v2", 1}, {"v1", 2}}));
  EXPECT_EQ(sorted(run(GCNHazardWalk(CBranch, 100))),
            Steps({{"v2", 0}, {"S_CBRANCH_SCC1", 1}, {"v1", 1}}));
}

// A cycle that comes back into the start block above a terminator start: the
// walk has been there on the way out, so the cycle stops at the branch.
TEST_F(GCNHazardWalkTest, CycleAboveTerminatorStart) {
  parseModule("---\nname: f\nbody: |\n" + std::string(R"(
  bb.0:
    successors: %bb.0
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_CBRANCH_SCC1 %bb.0, implicit undef $scc
    $sgpr0 = S_MOV_B32_term $sgpr2
    S_ENDPGM 0
)"));
  const MachineInstr &End = findOpcode(getMF("f"), AMDGPU::S_ENDPGM);
  EXPECT_EQ(sorted(run(GCNHazardWalk(End, 100))),
            Steps({{"S_MOV_B32_term", 0}, {"S_CBRANCH_SCC1", 1}, {"v1", 2}}));
}

// A meta instruction does not issue, so it weighs nothing under any weight.
TEST_F(GCNHazardWalkTest, MetaWeighsNothing) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    KILL $vgpr1
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  auto One = [](const MachineInstr &) -> unsigned { return 1; };
  EXPECT_EQ(run(GCNHazardWalk(Start, 100, One)),
            Steps({{"KILL", 0}, {"v1", 0}}));
}

// A bundle around the start, and one in a block two edges enter: headers are
// neither steps nor weighed.
TEST_F(GCNHazardWalkTest, BundlesAtStartAndJoin) {
  auto [InBundle, AtJoin] = parseBoth(R"(
  bb.0:
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    BUNDLE implicit-def $vgpr0, implicit-def $vgpr6 {
      $vgpr0 = V_MOV_B32_e32 99, implicit $exec
      $vgpr6 = V_MOV_B32_e32 6, implicit $exec
    }
    S_ENDPGM 0
)",
                                      R"(
  bb.0:
    successors: %bb.1, %bb.2
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    BUNDLE implicit-def $vgpr2, implicit-def $vgpr3 {
      $vgpr2 = V_MOV_B32_e32 2, implicit $exec
      $vgpr3 = V_MOV_B32_e32 3, implicit $exec
    }
    S_CBRANCH_SCC1 %bb.2, implicit undef $scc
    S_BRANCH %bb.1
  bb.1:
    successors: %bb.2
    S_BRANCH %bb.2
  bb.2:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  EXPECT_EQ(run(GCNHazardWalk(*InBundle, 100)), Steps({{"v1", 0}}));
  EXPECT_EQ(sorted(run(GCNHazardWalk(*AtJoin, 100))),
            Steps({{"S_BRANCH", 0},
                   {"S_CBRANCH_SCC1", 0},
                   {"S_BRANCH", 1},
                   {"v3", 1},
                   {"v2", 2},
                   {"v1", 3}}));
}

// Ending the short arm at v2 leaves v1 to the long arm.
TEST_F(GCNHazardWalkTest, EndPathOnOneArm) {
  Steps Expected = {{"S_BRANCH", 0}, {"S_BRANCH", 0}, {"S_NOP", 1},
                    {"v2", 1},       {"S_BRANCH", 6}, {"S_CBRANCH_SCC1", 7},
                    {"v1", 8}};
  auto [LongFirst, ShortFirst] = parseBoth(DiamondLongFirst, DiamondShortFirst);
  EXPECT_EQ(sorted(run(GCNHazardWalk(*LongFirst, 100), {"v2"})), Expected);
  EXPECT_EQ(sorted(run(GCNHazardWalk(*ShortFirst, 100), {"v2"})), Expected);
}

// The backedge leads through the rest of the block to the start itself, and
// stops where the walk began.
TEST_F(GCNHazardWalkTest, CycleThroughStart) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    successors: %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
  bb.1:
    successors: %bb.1, %bb.2
    $vgpr2 = V_MOV_B32_e32 2, implicit $exec
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    $vgpr3 = V_MOV_B32_e32 3, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
  bb.2:
    S_ENDPGM 0
)");
  EXPECT_EQ(sorted(run(GCNHazardWalk(Start, 100))),
            Steps({{"v2", 0},
                   {"S_CBRANCH_SCC1", 1},
                   {"v1", 1},
                   {"v3", 2},
                   {"v99", 3}}));
}

TEST_F(GCNHazardWalkTest, SelfLoopStartFirstInBlock) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    successors: %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
  bb.1:
    successors: %bb.1, %bb.2
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    $vgpr3 = V_MOV_B32_e32 3, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
  bb.2:
    S_ENDPGM 0
)");
  EXPECT_EQ(sorted(run(GCNHazardWalk(Start, 100))),
            Steps({{"S_CBRANCH_SCC1", 0}, {"v1", 0}, {"v3", 1}, {"v99", 2}}));
}

TEST_F(GCNHazardWalkTest, ZeroWeightCycle) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    successors: %bb.1
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
  bb.1:
    successors: %bb.2
  bb.2:
    successors: %bb.1, %bb.3
    S_CBRANCH_SCC1 %bb.1, implicit undef $scc
  bb.3:
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  auto Zero = [](const MachineInstr &) -> unsigned { return 0; };
  EXPECT_EQ(sorted(run(GCNHazardWalk(Start, 1, Zero))),
            Steps({{"S_CBRANCH_SCC1", 0}, {"v1", 0}}));
}

TEST_F(GCNHazardWalkTest, NarrowBound) {
  const MachineInstr &Start = parse(DiamondLongFirst);
  GCNHazardWalk Walk(Start, 100);
  Steps Result;
  for (auto [MI, Distance] : Walk) {
    Result.emplace_back(name(MI), Distance);
    if (name(MI) == "v2")
      Walk.narrowBound(3);
  }
  EXPECT_EQ(sorted(Result), Steps({{"S_BRANCH", 0},
                                   {"S_BRANCH", 0},
                                   {"S_NOP", 1},
                                   {"v2", 1},
                                   {"S_CBRANCH_SCC1", 2}}));
}

TEST_F(GCNHazardWalkTest, FindMaxWindowDeficit) {
  const MachineInstr &Start = parse(R"(
  bb.0:
    $vgpr1 = V_MOV_B32_e32 1, implicit $exec
    S_NOP 3
    $vgpr2 = V_MOV_B32_e32 2, implicit $exec
    $vgpr3 = V_MOV_B32_e32 3, implicit $exec
    $vgpr0 = V_MOV_B32_e32 99, implicit $exec
    S_ENDPGM 0
)");
  // v3 at 0, v2 at 1, v1 at 6.
  auto WindowFor = [](const MachineInstr &MI) -> std::optional<int> {
    if (name(MI) == "v2")
      return 4;
    if (name(MI) == "v1")
      return 8;
    return std::nullopt;
  };
  EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(Start, 8), WindowFor), 3);
  auto Far = [](const MachineInstr &MI) -> std::optional<int> {
    return name(MI) == "v1" ? std::optional<int>(8) : std::nullopt;
  };
  EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(Start, 8), Far), 2);
  EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(Start, 6), WindowFor), 3);
}

// Random CFGs against a label-correcting search that relaxes every position
// until nothing improves, written from the definitions alone.
class GCNHazardWalkRandomTest : public GCNHazardWalkTest {
protected:
  uint64_t Seed = 0x9e3779b97f4a7c15;

  unsigned random(unsigned N) {
    Seed ^= Seed << 13;
    Seed ^= Seed >> 7;
    Seed ^= Seed << 17;
    return Seed % N;
  }

  // A function with up to eight blocks of markers (V_MOV_B32 N; N >= 1000
  // ends a path), S_NOPs, and every kind of block end.
  std::string function(unsigned Index) {
    unsigned NumBlocks = 1 + random(8), Start = random(NumBlocks);
    std::string Body = "---\nname: f" + std::to_string(Index) + "\nbody: |\n";
    int Marker = 1;
    for (unsigned B = 0; B != NumBlocks; ++B) {
      bool Last = B + 1 == NumBlocks;
      unsigned X = random(NumBlocks), Y = random(NumBlocks);
      std::string Succ, Term;
      switch (random(Last ? 3 : 4)) {
      case 0:
        Term = Last ? "    S_ENDPGM 0\n" : "";
        Succ = Last ? "" : "%bb." + std::to_string(B + 1);
        break;
      case 1:
        Term = "    S_BRANCH %bb." + std::to_string(X) + "\n";
        Succ = "%bb." + std::to_string(X);
        break;
      case 2:
        // A terminator after the branch makes a start above which a cycle can
        // come back into the block.
        Term = "    S_CBRANCH_SCC1 %bb." + std::to_string(X) +
               ", implicit undef $scc\n";
        if (random(2))
          Term += "    $sgpr0 = S_MOV_B32_term $sgpr2\n";
        if (Last)
          Term += "    S_ENDPGM 0\n";
        Succ = "%bb." + std::to_string(X);
        if (!Last && X != B + 1)
          Succ += ", %bb." + std::to_string(B + 1);
        break;
      default:
        Term = "    S_CBRANCH_SCC1 %bb." + std::to_string(X) +
               ", implicit undef $scc\n    S_BRANCH %bb." + std::to_string(Y) +
               "\n";
        Succ = "%bb." + std::to_string(X);
        if (X != Y)
          Succ += ", %bb." + std::to_string(Y);
      }
      Body += "  bb." + std::to_string(B) + ":\n";
      if (!Succ.empty())
        Body += "    successors: " + Succ + "\n";
      unsigned NumInstrs = random(5), StartAt = random(NumInstrs + 1);
      for (unsigned I = 0; I <= NumInstrs; ++I) {
        if (B == Start && I == StartAt)
          Body += "    $vgpr0 = V_MOV_B32_e32 99, implicit $exec\n";
        if (I == NumInstrs)
          break;
        if (random(3) == 0) {
          Body += "    S_NOP " + std::to_string(random(4)) + "\n";
          continue;
        }
        int Imm = random(6) == 0 ? 1000 + Marker++ : Marker++;
        Body += "    $vgpr1 = V_MOV_B32_e32 " + std::to_string(Imm) +
                ", implicit $exec\n";
      }
      Body += Term;
    }
    return Body;
  }

  static bool endsPath(const MachineInstr &MI) {
    return MI.getOpcode() == AMDGPU::V_MOV_B32_e32 &&
           MI.getOperand(1).getImm() >= 1000;
  }

  // Written apart from getEntries on purpose. Where the two differ, this one
  // adds the last instruction once more, which changes no distance.
  //
  // The positions control can leave \p Pred from for \p MBB: every
  // terminator branching to \p MBB, and the last instruction if \p Pred falls
  // through to \p MBB or no terminator names it.
  static SmallVector<MachineBasicBlock::const_reverse_instr_iterator, 2>
  entries(const MachineBasicBlock &Pred, const MachineBasicBlock *MBB) {
    SmallVector<MachineBasicBlock::const_reverse_instr_iterator, 2> Result;
    for (const MachineInstr &MI : Pred.terminators())
      if (MI.isBranch() && MI.getNumOperands() > 0 &&
          MI.getOperand(0).isMBB() && MI.getOperand(0).getMBB() == MBB)
        Result.push_back(MI.getReverseIterator());
    bool FallsThrough =
        const_cast<MachineBasicBlock &>(Pred).getFallThrough(false) == MBB;
    if (Result.empty() ||
        (FallsThrough && Result.front() != Pred.instr_rbegin()))
      Result.push_back(Pred.instr_rbegin());
    return Result;
  }

  // Relaxes every position until nothing improves. Block tops are relaxed as
  // well, so that a cycle of empty blocks ends.
  struct Relaxation {
    int Bound;
    bool EndPaths;
    GCNHazardWalk::WeightFn Weight;
    DenseMap<const MachineInstr *, int> Dist;
    DenseMap<const MachineBasicBlock *, int> Top;

    template <typename MapT, typename KeyT>
    static bool improves(MapT &Map, KeyT Key, int Distance) {
      auto [It, Inserted] = Map.try_emplace(Key, Distance);
      if (!Inserted && It->second <= Distance)
        return false;
      It->second = Distance;
      return true;
    }

    void run(const MachineBasicBlock *MBB,
             MachineBasicBlock::const_reverse_instr_iterator Pos,
             int Distance) {
      for (; Pos != MBB->instr_rend(); ++Pos) {
        if (Pos->isBundle())
          continue;
        if (Distance >= Bound || !improves(Dist, &*Pos, Distance))
          return;
        if (EndPaths && endsPath(*Pos))
          return;
        Distance += Weight ? Weight(*Pos) : GCNHazardWalk::getWaitStates(*Pos);
      }
      if (Distance >= Bound || !improves(Top, MBB, Distance))
        return;
      for (const MachineBasicBlock *Pred : MBB->predecessors())
        for (auto Entry : entries(*Pred, MBB))
          run(Pred, Entry, Distance);
    }
  };

  // Every instruction of \p MF, as a start.
  static std::vector<const MachineInstr *> starts(MachineFunction &MF) {
    std::vector<const MachineInstr *> Result;
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB.instrs())
        Result.push_back(&MI);
    return Result;
  }

  // The distances below Bound of the instructions before \p From.
  static DenseMap<const MachineInstr *, int>
  relax(const MachineInstr &From, int Bound, bool EndPaths,
        GCNHazardWalk::WeightFn Weight = {}) {
    Relaxation R{Bound, EndPaths, Weight, {}, {}};
    R.run(From.getParent(), std::next(From.getReverseIterator()), 0);
    return std::move(R.Dist);
  }
};

TEST_F(GCNHazardWalkRandomTest, MatchesRelaxation) {
  constexpr unsigned NumFunctions = 400;
  std::vector<std::string> Functions;
  for (unsigned I = 0; I != NumFunctions; ++I)
    Functions.push_back(function(I));
  ASSERT_TRUE(parseMIR(join(Functions, "")));

  // From every instruction, with the default weights and with one per marker,
  // as a check that counts VALUs weighs.
  auto PerMarker = [](const MachineInstr &MI) -> unsigned {
    return MI.getOpcode() == AMDGPU::V_MOV_B32_e32;
  };
  const GCNHazardWalk::WeightFn Weights[] = {{}, PerMarker};
  for (unsigned I = 0; I != NumFunctions; ++I) {
    for (const MachineInstr *Start : starts(getMF("f" + std::to_string(I)))) {
      for (GCNHazardWalk::WeightFn Weight : Weights) {
        for (int Bound : {1, 3, 8, 1000}) {
          std::string Trace;
          raw_string_ostream(Trace) << "bound " << Bound << ", weights "
                                    << (Weight ? "per marker" : "default")
                                    << ", from " << *Start << "in\n"
                                    << Functions[I];
          SCOPED_TRACE(Trace);
          DenseMap<const MachineInstr *, int> Expected =
              relax(*Start, Bound, /*EndPaths=*/true, Weight);

          DenseMap<const MachineInstr *, int> Actual;
          int Last = 0;
          GCNHazardWalk Walk(*Start, Bound, Weight);
          for (auto [MI, Distance] : Walk) {
            EXPECT_LE(Last, Distance);
            Last = Distance;
            EXPECT_TRUE(Actual.try_emplace(&MI, Distance).second)
                << "yielded twice: " << MI;
            if (endsPath(MI))
              Walk.endPath();
          }
          EXPECT_EQ(Actual, Expected);

          // Pseudo-random windows of up to the bound on the markers, against
          // the distances of a relaxation in which no path ends.
          auto WindowFor =
              [Bound](const MachineInstr &MI) -> std::optional<int> {
            if (MI.getOpcode() != AMDGPU::V_MOV_B32_e32)
              return std::nullopt;
            return MI.getOperand(1).getImm() * 7 % (std::min(Bound, 12) + 1);
          };
          DenseMap<const MachineInstr *, int> All =
              relax(*Start, Bound, /*EndPaths=*/false, Weight);
          int ExpectedDeficit = 0;
          for (auto [MI, Distance] : All)
            if (std::optional<int> Window = WindowFor(*MI))
              ExpectedDeficit = std::max(ExpectedDeficit, *Window - Distance);
          EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(*Start, Bound, Weight),
                                         WindowFor),
                    ExpectedDeficit);

          // Every instruction once in any order, each at its distance: as the
          // only one with a window, the full bound, it leaves the bound minus
          // that.
          DenseMap<const MachineInstr *, int> Seen;
          auto Count = [&Seen](const MachineInstr &MI) -> std::optional<int> {
            ++Seen[&MI];
            return 0;
          };
          findMaxWindowDeficit(GCNHazardWalk(*Start, Bound, Weight), Count);
          EXPECT_EQ(Seen.size(), All.size());
          for (auto [MI, Distance] : All) {
            EXPECT_EQ(Seen.lookup(MI), 1) << "any order: " << *MI;
            const MachineInstr *Target = MI;
            auto Only = [Target, Bound](const MachineInstr &Other) {
              return &Other == Target ? std::optional<int>(Bound)
                                      : std::nullopt;
            };
            EXPECT_EQ(findMaxWindowDeficit(GCNHazardWalk(*Start, Bound, Weight),
                                           Only),
                      Bound - Distance)
                << "any order: " << *MI;
          }
        }
      }
    }
  }
}

} // end anonymous namespace
