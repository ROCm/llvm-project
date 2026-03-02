//===-- TraceCFGTest.cpp - Unit tests for trace CFG and loop analysis -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tools/tracecp/TraceUtil.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::tracecp;

namespace {

// Helper to create a simple CFG for testing
TraceCFG makeCFG(std::vector<std::tuple<uint64_t, uint64_t>> blocks,
                 std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> edges) {
  TraceCFG CFG;
  for (const auto &[start, end] : blocks) {
    BasicBlock BB;
    BB.StartPC = start;
    BB.EndPC = end;
    BB.NumInstructions = (end - start) / 4 + 1;
    BB.ExecutionCount = 1;
    CFG.Blocks[start] = BB;
  }
  for (const auto &[fromBlock, fromPC, toPC] : edges) {
    CFGEdge E;
    E.FromBlockPC = fromBlock;
    E.FromPC = fromPC;
    E.ToPC = toPC;
    E.Count = 1;
    CFG.Edges.push_back(E);
  }
  return CFG;
}

// Test: No loops (straight-line code)
// A -> B -> C
TEST(TraceCFGTest, NoLoops) {
  // Blocks: A[0x100-0x104], B[0x108-0x10c], C[0x110-0x114]
  // Edges: A->B, B->C
  TraceCFG CFG = makeCFG(
      {{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}},
      {{0x100, 0x104, 0x108}, {0x108, 0x10c, 0x110}});

  LoopInfo LI = detectLoops(CFG, {});

  EXPECT_EQ(LI.Loops.size(), 0u);
}

// Test: Simple single-block loop (header = latch)
// A -> A (self-loop)
TEST(TraceCFGTest, SingleBlockLoop) {
  // Block A[0x100-0x108] loops back to itself
  TraceCFG CFG = makeCFG({{0x100, 0x108}}, {{0x100, 0x108, 0x100}});

  LoopInfo LI = detectLoops(CFG, {});

  ASSERT_EQ(LI.Loops.size(), 1u);
  EXPECT_EQ(LI.Loops[0].HeaderPC, 0x100u);
  EXPECT_EQ(LI.Loops[0].LatchPCs.size(), 1u);
  EXPECT_EQ(LI.Loops[0].LatchPCs[0], 0x100u);
  EXPECT_EQ(LI.Loops[0].BodyBlockPCs.size(), 1u);
}

// Test: Simple multi-block loop
// A -> B -> C -> A
TEST(TraceCFGTest, SimpleLoop) {
  // Entry -> A -> B -> C -> A (back-edge)
  //                    \-> Exit
  // For simplicity, just A -> B -> C -> A
  TraceCFG CFG =
      makeCFG({{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}},
              {{0x100, 0x104, 0x108},   // A -> B
               {0x108, 0x10c, 0x110},   // B -> C
               {0x110, 0x114, 0x100}}); // C -> A (back-edge)

  LoopInfo LI = detectLoops(CFG, {});

  ASSERT_EQ(LI.Loops.size(), 1u);
  EXPECT_EQ(LI.Loops[0].HeaderPC, 0x100u);
  EXPECT_EQ(LI.Loops[0].LatchPCs.size(), 1u);
  EXPECT_EQ(LI.Loops[0].LatchPCs[0], 0x110u);
  EXPECT_EQ(LI.Loops[0].BodyBlockPCs.size(), 3u);
}

// Test: Nested loops
// Outer: A -> B -> C -> D -> A
// Inner: B -> C -> B
TEST(TraceCFGTest, NestedLoops) {
  // A (outer header) -> B (inner header) -> C -> B (inner back-edge)
  //                                           -> D -> A (outer back-edge)
  TraceCFG CFG =
      makeCFG({{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}, {0x118, 0x11c}},
              {{0x100, 0x104, 0x108},   // A -> B
               {0x108, 0x10c, 0x110},   // B -> C
               {0x110, 0x114, 0x108},   // C -> B (inner back-edge)
               {0x110, 0x114, 0x118},   // C -> D
               {0x118, 0x11c, 0x100}}); // D -> A (outer back-edge)

  LoopInfo LI = detectLoops(CFG, {});

  ASSERT_EQ(LI.Loops.size(), 2u);

  // Find inner and outer loops
  const Loop *Inner = nullptr;
  const Loop *Outer = nullptr;
  for (const Loop &L : LI.Loops) {
    if (L.HeaderPC == 0x108)
      Inner = &L;
    else if (L.HeaderPC == 0x100)
      Outer = &L;
  }

  ASSERT_NE(Inner, nullptr);
  ASSERT_NE(Outer, nullptr);

  // Check inner loop
  EXPECT_EQ(Inner->HeaderPC, 0x108u);
  EXPECT_EQ(Inner->LatchPCs.size(), 1u);
  EXPECT_EQ(Inner->LatchPCs[0], 0x110u);

  // Check outer loop
  EXPECT_EQ(Outer->HeaderPC, 0x100u);
  EXPECT_EQ(Outer->LatchPCs.size(), 1u);
  EXPECT_EQ(Outer->LatchPCs[0], 0x118u);

  // Inner should be nested in outer
  // Find indices
  int InnerIdx = -1, OuterIdx = -1;
  for (size_t I = 0; I < LI.Loops.size(); ++I) {
    if (LI.Loops[I].HeaderPC == 0x108)
      InnerIdx = I;
    else if (LI.Loops[I].HeaderPC == 0x100)
      OuterIdx = I;
  }
  EXPECT_EQ(LI.Loops[InnerIdx].ParentIdx, OuterIdx);
  EXPECT_EQ(LI.Loops[OuterIdx].ParentIdx, -1);
}

// Test: Multiple back-edges to same header (multiple latches)
// A -> B -> A, A -> C -> A
TEST(TraceCFGTest, MultipleLatches) {
  // A -> B -> A
  // A -> C -> A
  TraceCFG CFG =
      makeCFG({{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}},
              {{0x100, 0x104, 0x108},   // A -> B
               {0x108, 0x10c, 0x100},   // B -> A (back-edge 1)
               {0x100, 0x104, 0x110},   // A -> C
               {0x110, 0x114, 0x100}}); // C -> A (back-edge 2)

  LoopInfo LI = detectLoops(CFG, {});

  ASSERT_EQ(LI.Loops.size(), 1u);
  EXPECT_EQ(LI.Loops[0].HeaderPC, 0x100u);
  EXPECT_EQ(LI.Loops[0].LatchPCs.size(), 2u);

  // Both B and C should be latches
  std::set<uint64_t> Latches(LI.Loops[0].LatchPCs.begin(),
                             LI.Loops[0].LatchPCs.end());
  EXPECT_TRUE(Latches.count(0x108)); // B
  EXPECT_TRUE(Latches.count(0x110)); // C
}

// Test: Loop with if-else inside
// A -> B -> D -> A
// A -> C -> D -> A
TEST(TraceCFGTest, LoopWithIfElse) {
  // A (header) -> B (then) -> D (merge/latch) -> A
  //            -> C (else) -> D
  TraceCFG CFG = makeCFG(
      {{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}, {0x118, 0x11c}},
      {{0x100, 0x104, 0x108},   // A -> B (then)
       {0x100, 0x104, 0x110},   // A -> C (else)
       {0x108, 0x10c, 0x118},   // B -> D
       {0x110, 0x114, 0x118},   // C -> D
       {0x118, 0x11c, 0x100}}); // D -> A (back-edge)

  LoopInfo LI = detectLoops(CFG, {});

  ASSERT_EQ(LI.Loops.size(), 1u);
  EXPECT_EQ(LI.Loops[0].HeaderPC, 0x100u);
  EXPECT_EQ(LI.Loops[0].LatchPCs.size(), 1u);
  EXPECT_EQ(LI.Loops[0].LatchPCs[0], 0x118u);
  // All 4 blocks should be in the body
  EXPECT_EQ(LI.Loops[0].BodyBlockPCs.size(), 4u);
}

} // namespace
