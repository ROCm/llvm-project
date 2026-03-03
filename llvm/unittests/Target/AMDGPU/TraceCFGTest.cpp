//===-- TraceCFGTest.cpp - Unit tests for trace CFG reconstruction --------===//
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
    TraceBlock BB;
    BB.StartPC = start;
    BB.EndPC = end;
    BB.NumInstructions = (end - start) / 4 + 1;
    BB.ExecutionCount = 1;
    CFG.Blocks[start] = BB;
  }
  for (const auto &[fromBlock, fromPC, toPC] : edges) {
    TraceEdge E;
    E.FromBlockPC = fromBlock;
    E.FromPC = fromPC;
    E.ToPC = toPC;
    E.Count = 1;
    CFG.Edges.push_back(E);
  }
  return CFG;
}

// Test: CFG with multiple blocks
TEST(TraceCFGTest, MultipleBlocks) {
  // Blocks: A[0x100-0x104], B[0x108-0x10c], C[0x110-0x114]
  // Edges: A->B, B->C
  TraceCFG CFG = makeCFG(
      {{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}},
      {{0x100, 0x104, 0x108}, {0x108, 0x10c, 0x110}});

  EXPECT_EQ(CFG.Blocks.size(), 3u);
  EXPECT_EQ(CFG.Edges.size(), 2u);

  // Check blocks exist
  EXPECT_TRUE(CFG.Blocks.count(0x100));
  EXPECT_TRUE(CFG.Blocks.count(0x108));
  EXPECT_TRUE(CFG.Blocks.count(0x110));

  // Check block properties
  EXPECT_EQ(CFG.Blocks[0x100].StartPC, 0x100u);
  EXPECT_EQ(CFG.Blocks[0x100].EndPC, 0x104u);
}

// Test: CFG with back-edge (loop structure)
TEST(TraceCFGTest, BackEdge) {
  // A -> B -> C -> A (back-edge)
  TraceCFG CFG =
      makeCFG({{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}},
              {{0x100, 0x104, 0x108},   // A -> B
               {0x108, 0x10c, 0x110},   // B -> C
               {0x110, 0x114, 0x100}}); // C -> A (back-edge)

  EXPECT_EQ(CFG.Blocks.size(), 3u);
  EXPECT_EQ(CFG.Edges.size(), 3u);

  // Check that back-edge exists
  bool hasBackEdge = false;
  for (const TraceEdge &E : CFG.Edges) {
    if (E.ToPC == 0x100 && E.FromBlockPC == 0x110)
      hasBackEdge = true;
  }
  EXPECT_TRUE(hasBackEdge);
}

// Test: CFG with diamond pattern (if-else)
TEST(TraceCFGTest, DiamondPattern) {
  // A -> B -> D
  // A -> C -> D
  TraceCFG CFG = makeCFG(
      {{0x100, 0x104}, {0x108, 0x10c}, {0x110, 0x114}, {0x118, 0x11c}},
      {{0x100, 0x104, 0x108},   // A -> B (then)
       {0x100, 0x104, 0x110},   // A -> C (else)
       {0x108, 0x10c, 0x118},   // B -> D
       {0x110, 0x114, 0x118}}); // C -> D

  EXPECT_EQ(CFG.Blocks.size(), 4u);
  EXPECT_EQ(CFG.Edges.size(), 4u);

  // Block A has two outgoing edges
  unsigned outFromA = 0;
  for (const TraceEdge &E : CFG.Edges) {
    if (E.FromBlockPC == 0x100)
      outFromA++;
  }
  EXPECT_EQ(outFromA, 2u);
}

// Test: Single block CFG
TEST(TraceCFGTest, SingleBlock) {
  TraceCFG CFG = makeCFG({{0x100, 0x108}}, {});

  EXPECT_EQ(CFG.Blocks.size(), 1u);
  EXPECT_EQ(CFG.Edges.size(), 0u);
  EXPECT_EQ(CFG.Blocks[0x100].NumInstructions, 3u);
}

} // namespace
