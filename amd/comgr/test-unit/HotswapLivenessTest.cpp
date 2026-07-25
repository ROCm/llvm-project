//===- HotswapLivenessTest.cpp - Unit tests for reglive liveness ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for reglive::LivenessAnalysis in comgr-hotswap-liveness-analysis.cpp.
/// Programs are assembled instruction-by-instruction (with explicit numeric
/// SOPP branch immediates) and decoded through a real gfx1250 LLVMState, then
/// the CFG + backward dataflow solver are exercised end to end.
///
/// COMGR::ensureLLVMInitialized() is provided by HotswapMCTest.cpp, linked into
/// the same HotswapMCTests binary; it is not redefined here.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-cfg.h"
#include "comgr-hotswap-internal.h"
#include "comgr-hotswap-liveness-analysis.h"
#include "comgr.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace COMGR;
using namespace COMGR::hotswap;
using namespace COMGR::hotswap::reglive;

namespace {

TargetIdentifier makeGfx1250Ident() {
  TargetIdentifier TI;
  TI.Arch = "amdgcn";
  TI.Vendor = "amd";
  TI.OS = "amdhsa";
  TI.Environ = "";
  TI.Processor = "gfx1250";
  return TI;
}

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

RegisterRef sgpr(uint16_t Index) { return {RegClass::SGPR, Index, 1}; }
RegisterRef vgpr(uint16_t Index) { return {RegClass::VGPR, Index, 1}; }

// -- Scalar defs kill (precise scalar liveness) -----------------------------

TEST(RegliveLiveness, ScalarDefKillsSource) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(
      S, {"s_mov_b32 s0, s1", "s_mov_b32 s2, s0", "s_endpgm"}, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);

  Cfg Graph = buildCfg(Decoded, S);
  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);

  // Before inst0, only s1 is live; s0 is defined here, so it is not live-in.
  EXPECT_TRUE(LA.isLiveBefore(0, sgpr(1)));
  EXPECT_FALSE(LA.isLiveBefore(0, sgpr(0)));
  // Before inst1, s0 is live (produced by inst0, consumed here).
  EXPECT_TRUE(LA.isLiveBefore(1, sgpr(0)));
}

// -- EXEC-masked vector defs do not kill ------------------------------------

TEST(RegliveLiveness, ExecMaskedVectorDefDoesNotKill) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(
      S, {"v_mov_b32 v0, v5", "v_mov_b32 v6, v0", "s_endpgm"}, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);

  Cfg Graph = buildCfg(Decoded, S);
  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);

  // v0 is written by inst0 but the write is EXEC-masked, so v0 is still treated
  // as live before inst0 (inactive lanes preserve the old value).
  EXPECT_TRUE(LA.isLiveBefore(0, vgpr(0)));
  EXPECT_TRUE(LA.isLiveBefore(0, vgpr(5)));
}

// -- Diamond: successor live-ins union at the branch ------------------------

TEST(RegliveLiveness, DiamondUnionsSuccessorLiveIn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // 0  s_mov_b32 s0, s1
  // 4  s_cbranch_scc1 2   ; -> off 16
  // 8  s_mov_b32 s3, s0   ; uses s0
  // 12 s_branch 1         ; -> off 20
  // 16 s_mov_b32 s4, s0   ; uses s0
  // 20 s_endpgm
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(S,
                              {"s_mov_b32 s0, s1", "s_cbranch_scc1 2",
                               "s_mov_b32 s3, s0", "s_branch 1",
                               "s_mov_b32 s4, s0", "s_endpgm"},
                              Decoded));
  ASSERT_EQ(Decoded.size(), 6u);

  Cfg Graph = buildCfg(Decoded, S);
  ASSERT_EQ(Graph.Blocks.size(), 4u);
  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);

  const unsigned Entry = Graph.OffsetToBlock.lookup(0);
  // s0 is used on both arms, so it is live out of the entry block; s1 (only
  // read by the entry block itself) is live in but not out.
  EXPECT_TRUE(LA.blockLiveness(Entry).LiveOut.contains(sgpr(0)));
  EXPECT_TRUE(LA.blockLiveness(Entry).LiveIn.contains(sgpr(1)));
  EXPECT_FALSE(LA.blockLiveness(Entry).LiveIn.contains(sgpr(0)));

  // s0 is live before the first instruction of each arm.
  EXPECT_TRUE(LA.isLiveBefore(2, sgpr(0)));
  EXPECT_TRUE(LA.isLiveBefore(4, sgpr(0)));
}

// -- Loop: value used in the body stays live across the backedge ------------

TEST(RegliveLiveness, LoopBackedgeKeepsValueLive) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // 0  s_mov_b32 s3, s0     ; body reads s0
  // 4  s_cbranch_scc1 -2    ; backedge to off 0, else fall through
  // 8  s_endpgm
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(
      S, {"s_mov_b32 s3, s0", "s_cbranch_scc1 -2", "s_endpgm"}, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);

  Cfg Graph = buildCfg(Decoded, S);
  const unsigned Header = Graph.OffsetToBlock.lookup(0);
  // The header must branch back to itself for this to be a loop.
  ASSERT_FALSE(Graph.Blocks[Header].Successors.empty());

  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);
  // s0 flows around the backedge, so it is live out of the header and live
  // before the loop body's first instruction.
  EXPECT_TRUE(LA.blockLiveness(Header).LiveOut.contains(sgpr(0)));
  EXPECT_TRUE(LA.isLiveBefore(0, sgpr(0)));
  EXPECT_FALSE(LA.isLiveBefore(0, sgpr(3)));
}

} // namespace
