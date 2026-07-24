//===- HotswapCfgTest.cpp - Unit tests for reglive CFG construction ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for reglive::buildCfg / reglive::reversePostOrder in
/// comgr-hotswap-cfg.cpp. Programs are assembled instruction-by-instruction
/// (with explicit numeric SOPP branch immediates) and concatenated, then
/// decoded through a real gfx1250 LLVMState so block partitioning and edge
/// resolution exercise the production MCInstrAnalysis / branch-target helpers.
///
/// COMGR::ensureLLVMInitialized() is provided by HotswapMCTest.cpp, linked into
/// the same HotswapMCTests binary; it is not redefined here.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-cfg.h"
#include "comgr-hotswap-internal.h"
#include "comgr.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

using namespace COMGR;
using namespace COMGR::hotswap;
using namespace COMGR::hotswap::reglive;

namespace {

bool hasSucc(const reglive::BasicBlock &B, unsigned To) {
  return std::find(B.Successors.begin(), B.Successors.end(), To) !=
         B.Successors.end();
}
bool hasPred(const reglive::BasicBlock &B, unsigned From) {
  return std::find(B.Predecessors.begin(), B.Predecessors.end(), From) !=
         B.Predecessors.end();
}

TargetIdentifier makeGfx1250Ident() {
  TargetIdentifier TI;
  TI.Arch = "amdgcn";
  TI.Vendor = "amd";
  TI.OS = "amdhsa";
  TI.Environ = "";
  TI.Processor = "gfx1250";
  return TI;
}

// Assemble each line separately and concatenate into one .text image, then
// decode. Keeping each line independent avoids label-fixup resolution (the
// capturing assembler path does not run layout), so branch destinations are
// expressed as raw SOPP simm16 dword deltas.
bool assembleProgram(const LLVMState &S, llvm::ArrayRef<std::string> Lines,
                     std::vector<InternalDecodedInst> &Decoded) {
  llvm::SmallVector<uint8_t> Bytes;
  for (const std::string &Line : Lines) {
    llvm::SmallVector<uint8_t> InstBytes = assembleSingleInst(Line, S);
    if (InstBytes.empty())
      return false;
    Bytes.append(InstBytes.begin(), InstBytes.end());
  }
  return decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded);
}

// -- Straight-line + terminator splitting -----------------------------------

TEST(RegliveCfg, TerminatorSplitsBlocksWithoutEdges) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Two s_endpgm terminators produce two blocks; the first exits (no edge).
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(
      S, {"v_mov_b32 v0, v1", "s_endpgm", "v_mov_b32 v2, v3", "s_endpgm"},
      Decoded));
  ASSERT_EQ(Decoded.size(), 4u);

  Cfg Graph = buildCfg(Decoded, S);
  ASSERT_EQ(Graph.Blocks.size(), 2u);
  EXPECT_EQ(Graph.Blocks[0].InstIndices.size(), 2u);
  EXPECT_EQ(Graph.Blocks[1].InstIndices.size(), 2u);
  EXPECT_TRUE(Graph.Blocks[0].Successors.empty());
  EXPECT_TRUE(Graph.Blocks[1].Successors.empty());
  EXPECT_TRUE(Graph.Blocks[0].Predecessors.empty());
  EXPECT_TRUE(Graph.Blocks[1].Predecessors.empty());

  std::vector<unsigned> Rpo = reversePostOrder(Graph);
  ASSERT_EQ(Rpo.size(), 2u);
}

// -- Diamond CFG (conditional + unconditional branch) -----------------------

TEST(RegliveCfg, DiamondBranchEdgesAndRpo) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Layout (all instructions 4 bytes):
  //   0  v_mov_b32 v2, v0
  //   4  s_cbranch_scc1 2   ; -> off 16 (taken) / fall through to off 8
  //   8  v_mov_b32 v3, v0
  //   12 s_branch 1         ; -> off 20 (unconditional)
  //   16 v_mov_b32 v4, v0
  //   20 s_endpgm
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(S,
                              {"v_mov_b32 v2, v0", "s_cbranch_scc1 2",
                               "v_mov_b32 v3, v0", "s_branch 1",
                               "v_mov_b32 v4, v0", "s_endpgm"},
                              Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  for (const InternalDecodedInst &DI : Decoded)
    ASSERT_EQ(DI.Size, 4u) << "unexpected instruction size in layout";

  Cfg Graph = buildCfg(Decoded, S);
  ASSERT_EQ(Graph.Blocks.size(), 4u);

  // Block start offsets: 0, 8, 16, 20.
  EXPECT_EQ(Graph.Blocks[0].StartOffset, 0u);
  EXPECT_EQ(Graph.Blocks[1].StartOffset, 8u);
  EXPECT_EQ(Graph.Blocks[2].StartOffset, 16u);
  EXPECT_EQ(Graph.Blocks[3].StartOffset, 20u);

  EXPECT_EQ(Graph.Blocks[0].InstIndices.size(), 2u);
  EXPECT_EQ(Graph.Blocks[1].InstIndices.size(), 2u);
  EXPECT_EQ(Graph.Blocks[2].InstIndices.size(), 1u);
  EXPECT_EQ(Graph.Blocks[3].InstIndices.size(), 1u);

  // B0 (cond branch): taken target B2 + fall-through B1.
  EXPECT_EQ(Graph.Blocks[0].Successors.size(), 2u);
  EXPECT_TRUE(hasSucc(Graph.Blocks[0], 1u));
  EXPECT_TRUE(hasSucc(Graph.Blocks[0], 2u));
  // B1 (unconditional branch to end): only B3.
  EXPECT_EQ(Graph.Blocks[1].Successors.size(), 1u);
  EXPECT_TRUE(hasSucc(Graph.Blocks[1], 3u));
  // B2 (non-terminator end): falls through to B3.
  EXPECT_EQ(Graph.Blocks[2].Successors.size(), 1u);
  EXPECT_TRUE(hasSucc(Graph.Blocks[2], 3u));
  // B3 (s_endpgm): exit.
  EXPECT_TRUE(Graph.Blocks[3].Successors.empty());

  EXPECT_EQ(Graph.Blocks[3].Predecessors.size(), 2u);
  EXPECT_TRUE(hasPred(Graph.Blocks[3], 1u));
  EXPECT_TRUE(hasPred(Graph.Blocks[3], 2u));

  // RPO: entry first, join last.
  std::vector<unsigned> Rpo = reversePostOrder(Graph);
  ASSERT_EQ(Rpo.size(), 4u);
  EXPECT_EQ(Rpo.front(), 0u);
  EXPECT_EQ(Rpo.back(), 3u);
}

// -- Scoped RPO -------------------------------------------------------------

TEST(RegliveCfg, ScopedRpoRestrictsToGivenBlocks) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(
      S, {"v_mov_b32 v0, v1", "s_endpgm", "v_mov_b32 v2, v3", "s_endpgm"},
      Decoded));
  Cfg Graph = buildCfg(Decoded, S);
  ASSERT_EQ(Graph.Blocks.size(), 2u);

  // Restrict the traversal to the second block only.
  std::vector<unsigned> Scope = {1u};
  std::vector<unsigned> Rpo = reversePostOrder(Graph, Scope);
  ASSERT_EQ(Rpo.size(), 1u);
  EXPECT_EQ(Rpo.front(), 1u);
}

} // namespace
