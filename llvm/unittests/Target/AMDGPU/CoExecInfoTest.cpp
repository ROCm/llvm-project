//===- CoExecInfoTest.cpp - Unit tests for AMDGPUCoExecInfo ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCoExecInfo.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

TEST(CoExecInfoTest, BuildBasicPattern) {
  // Build a basic pattern
  auto Info = CoExecInfo::build(4, 6, "0EEIVV", 3, false);

  EXPECT_EQ(Info.Occupancy, 4u);
  EXPECT_EQ(Info.TotalWindow, 6u);
  EXPECT_EQ(Info.LastIStage, 3u);
  EXPECT_FALSE(Info.HasScaling);

  // Check slot masks
  EXPECT_EQ(Info.getMask(0), CoExecMask::StageE0);
  EXPECT_EQ(Info.getMask(1), CoExecMask::StageE);
  EXPECT_EQ(Info.getMask(2), CoExecMask::StageE);
  EXPECT_EQ(Info.getMask(3), CoExecMask::StageI);
  EXPECT_EQ(Info.getMask(4), CoExecMask::StageV);
  EXPECT_EQ(Info.getMask(5), CoExecMask::StageV);
}

TEST(CoExecInfoTest, CanCoExec) {
  // Build a pattern with different slot types
  auto Info = CoExecInfo::build(4, 6, "0EEIVV", 3, false);

  // E0 slot (index 0) - only CTRL allowed
  EXPECT_TRUE(Info.canCoExec(CoExecMask::CTRL, 0));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::VALU, 0));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::TRANS, 0));

  // E slots (index 1, 2) - no VALU/TRANS
  EXPECT_FALSE(Info.canCoExec(CoExecMask::VALU, 1));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::TRANS, 2));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::SALU, 1));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::DS, 2));

  // I slot (index 3) - VALU and TRANS allowed
  EXPECT_TRUE(Info.canCoExec(CoExecMask::VALU, 3));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::TRANS, 3));

  // V slots (index 4, 5) - no VALU/TRANS, but WMMA allowed
  EXPECT_FALSE(Info.canCoExec(CoExecMask::VALU, 4));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::TRANS, 5));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 4));
}

TEST(CoExecInfoTest, Stage3Avoidances) {
  // Test that stage 3 avoidances are set correctly for 10-cycle patterns
  // This tests the new avoidances added for stage 3 (DS/SALU/VMEM)
  auto Info = CoExecInfo::build(8, 10, "0EEIEEIIVV", 7, false)
                  .preferring(3, flavorBit(InstructionFlavor::TRANS))
                  .avoiding(3, flavorBit(InstructionFlavor::DS) |
                                   flavorBit(InstructionFlavor::SALU) |
                                   flavorBit(InstructionFlavor::VMEM));

  // Stage 3 should prefer TRANS
  EXPECT_TRUE(Info.prefersFlavor(3, InstructionFlavor::TRANS));

  // Stage 3 should avoid DS, SALU, and VMEM
  EXPECT_TRUE(Info.avoidsFlavor(3, InstructionFlavor::DS));
  EXPECT_TRUE(Info.avoidsFlavor(3, InstructionFlavor::SALU));
  EXPECT_TRUE(Info.avoidsFlavor(3, InstructionFlavor::VMEM));

  // Stage 3 should NOT avoid SingleCycleVALU or TRANS
  EXPECT_FALSE(Info.avoidsFlavor(3, InstructionFlavor::SingleCycleVALU));
  EXPECT_FALSE(Info.avoidsFlavor(3, InstructionFlavor::TRANS));
}

TEST(CoExecInfoTest, Stage7TRANSPreferenceNonScaled) {
  // Test that stage 7 prefers TRANS for non-scaled patterns
  auto Info = CoExecInfo::build(8, 10, "0EEIEEIIVV", 7, false)
                  .preferring(7, flavorBit(InstructionFlavor::TRANS));

  // Stage 7 should prefer TRANS for non-scaled
  EXPECT_TRUE(Info.prefersFlavor(7, InstructionFlavor::TRANS));
  EXPECT_FALSE(Info.prefersFlavor(7, InstructionFlavor::SingleCycleVALU));
  EXPECT_FALSE(Info.prefersFlavor(7, InstructionFlavor::WMMA));
}

TEST(CoExecInfoTest, Stage7WMMAPreferenceScaled) {
  // Test that stage 7 prefers WMMA for scaled patterns
  auto Info = CoExecInfo::build(8, 10, "0EEIEEISVV", 7, true)
                  .preferring(7, flavorBit(InstructionFlavor::WMMA));

  // Stage 7 should prefer WMMA for scaled
  EXPECT_TRUE(Info.prefersFlavor(7, InstructionFlavor::WMMA));
  EXPECT_FALSE(Info.prefersFlavor(7, InstructionFlavor::SingleCycleVALU));
  EXPECT_FALSE(Info.prefersFlavor(7, InstructionFlavor::TRANS));
}

TEST(CoExecInfoTest, TypeIndexComputation) {
  // Test that TypeIndex is computed correctly for mixed patterns
  auto Info = CoExecInfo::build(8, 10, "0EEIEEIIVV", 7, false);

  // E0 at index 0 should have TypeIndex 0
  EXPECT_EQ(Info.getTypeIndex(0), 0u);

  // E slots at indices 1, 2 should have TypeIndex 0, 1
  EXPECT_EQ(Info.getTypeIndex(1), 0u);
  EXPECT_EQ(Info.getTypeIndex(2), 1u);

  // I slot at index 3 should have TypeIndex 0
  EXPECT_EQ(Info.getTypeIndex(3), 0u);

  // E slots at indices 4, 5 should have TypeIndex 2, 3
  EXPECT_EQ(Info.getTypeIndex(4), 2u);
  EXPECT_EQ(Info.getTypeIndex(5), 3u);

  // I slots at indices 6, 7 should have TypeIndex 1, 2
  EXPECT_EQ(Info.getTypeIndex(6), 1u);
  EXPECT_EQ(Info.getTypeIndex(7), 2u);

  // V slots at indices 8, 9 should have TypeIndex 0, 1
  EXPECT_EQ(Info.getTypeIndex(8), 0u);
  EXPECT_EQ(Info.getTypeIndex(9), 1u);
}

TEST(CoExecInfoTest, StageTypeDetection) {
  // Test getType() returns correct stage types
  auto Info = CoExecInfo::build(8, 10, "0EEIEEIIVV", 7, false);

  EXPECT_EQ(Info.getType(0), CoExecStageType::E0);
  EXPECT_EQ(Info.getType(1), CoExecStageType::E);
  EXPECT_EQ(Info.getType(2), CoExecStageType::E);
  EXPECT_EQ(Info.getType(3), CoExecStageType::I);
  EXPECT_EQ(Info.getType(4), CoExecStageType::E);
  EXPECT_EQ(Info.getType(5), CoExecStageType::E);
  EXPECT_EQ(Info.getType(6), CoExecStageType::I);
  EXPECT_EQ(Info.getType(7), CoExecStageType::I);
  EXPECT_EQ(Info.getType(8), CoExecStageType::V);
  EXPECT_EQ(Info.getType(9), CoExecStageType::V);
}

TEST(CoExecInfoTest, ScaledPatternWithSSlot) {
  // Test scaled pattern with S slot (IS mask)
  auto Info = CoExecInfo::build(8, 10, "0EEIEEISVV", 7, true);

  EXPECT_TRUE(Info.HasScaling);
  // S slot should have StageIS mask (I + WMMA)
  EXPECT_EQ(Info.getMask(7), CoExecMask::StageIS);
  EXPECT_TRUE(Info.canCoExec(CoExecMask::VALU, 7));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 7));
}

} // namespace
