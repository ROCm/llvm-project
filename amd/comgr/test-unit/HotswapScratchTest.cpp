//===- HotswapScratchTest.cpp - Unit tests for reglive scratch finders ---===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the scratch-allocation queries on reglive::LivenessAnalysis
/// (findFreeRun / findFreeSgpr / findFreeSgprPair and the MinFreeVgpr floor)
/// in comgr-hotswap-liveness-analysis.cpp. Programs are assembled and decoded
/// through a real gfx1250 LLVMState so the dead-register search runs over the
/// same live-before sets a lowering would query.
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

#include <optional>
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

// -- Free SGPR / SGPR pair --------------------------------------------------

TEST(RegliveScratch, FindsFreeSgprAboveLiveSet) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Before inst0, s0 and s1 are live (read by the add); s2+ are dead.
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(
      assembleProgram(S, {"s_add_co_u32 s10, s0, s1", "s_endpgm"}, Decoded));
  Cfg Graph = buildCfg(Decoded, S);
  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);

  std::optional<uint16_t> Free = LA.findFreeSgpr(0);
  ASSERT_TRUE(Free.has_value());
  EXPECT_EQ(*Free, 2u);

  // A search start skips lower indices even when they are dead.
  std::optional<uint16_t> FreeFrom5 = LA.findFreeSgpr(0, /*SearchStart=*/5);
  ASSERT_TRUE(FreeFrom5.has_value());
  EXPECT_EQ(*FreeFrom5, 5u);
}

TEST(RegliveScratch, FindsEvenAlignedFreeSgprPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(
      assembleProgram(S, {"s_add_co_u32 s10, s0, s1", "s_endpgm"}, Decoded));
  Cfg Graph = buildCfg(Decoded, S);
  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);

  // {s0,s1} live so the first free even-aligned pair is {s2,s3}.
  std::optional<uint16_t> Pair = LA.findFreeSgprPair(0);
  ASSERT_TRUE(Pair.has_value());
  EXPECT_EQ(*Pair, 2u);

  // An odd search start rounds up to keep the pair even-aligned.
  std::optional<uint16_t> PairFrom3 = LA.findFreeSgprPair(0, /*SearchStart=*/3);
  ASSERT_TRUE(PairFrom3.has_value());
  EXPECT_EQ(*PairFrom3, 4u);
}

// -- Free VGPR run ----------------------------------------------------------

TEST(RegliveScratch, FindsFreeVgprRun) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Before inst0, v0 and v1 are live (read by the add); v2+ are dead.
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(
      assembleProgram(S, {"v_add_f32 v10, v0, v1", "s_endpgm"}, Decoded));
  Cfg Graph = buildCfg(Decoded, S);
  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);

  std::optional<uint16_t> One = LA.findFreeRun(0, /*Count=*/1);
  ASSERT_TRUE(One.has_value());
  EXPECT_EQ(*One, 2u);

  std::optional<uint16_t> Four = LA.findFreeRun(0, /*Count=*/4);
  ASSERT_TRUE(Four.has_value());
  EXPECT_EQ(*Four, 2u);
}

// -- MinFreeVgpr allocation floor -------------------------------------------

TEST(RegliveScratch, MinFreeVgprFloorRaisesSearchStart) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(S, {"s_nop 0", "s_endpgm"}, Decoded));
  Cfg Graph = buildCfg(Decoded, S);

  // No VGPRs are live, so the default analysis hands out v0.
  LivenessAnalysis Default(Decoded, Graph, *S.MCII, *S.MRI);
  std::optional<uint16_t> DefaultRun = Default.findFreeRun(0, /*Count=*/1);
  ASSERT_TRUE(DefaultRun.has_value());
  EXPECT_EQ(*DefaultRun, 0u);

  // With a floor of 8, the same dead-register search starts at v8.
  LivenessAnalysisOptions Options;
  Options.MinFreeVgpr = 8;
  LivenessAnalysis Floored(Decoded, Graph, *S.MCII, *S.MRI,
                           llvm::ArrayRef<unsigned>{}, Options);
  std::optional<uint16_t> FlooredRun = Floored.findFreeRun(0, /*Count=*/1);
  ASSERT_TRUE(FlooredRun.has_value());
  EXPECT_EQ(*FlooredRun, 8u);
}

// -- No result: unanalyzed instruction or no run fits -----------------------

TEST(RegliveScratch, ReturnsNulloptWhenUnavailable) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(assembleProgram(S, {"s_nop 0", "s_endpgm"}, Decoded));
  Cfg Graph = buildCfg(Decoded, S);

  LivenessAnalysis LA(Decoded, Graph, *S.MCII, *S.MRI);
  // An instruction index that was never analyzed yields no allocation.
  EXPECT_FALSE(LA.findFreeRun(9999, /*Count=*/1).has_value());
  EXPECT_FALSE(LA.findFreeSgpr(9999).has_value());
  // A count of zero is rejected.
  EXPECT_FALSE(LA.findFreeRun(0, /*Count=*/0).has_value());

  // A floor so high that the requested run cannot fit yields no allocation.
  LivenessAnalysisOptions Options;
  Options.MinFreeVgpr = RegisterSetMaxVgprs - 1;
  LivenessAnalysis HighFloor(Decoded, Graph, *S.MCII, *S.MRI,
                             llvm::ArrayRef<unsigned>{}, Options);
  EXPECT_FALSE(HighFloor.findFreeRun(0, /*Count=*/4).has_value());
}

} // namespace
