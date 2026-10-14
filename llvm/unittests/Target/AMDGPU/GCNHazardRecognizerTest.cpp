//===- GCNHazardRecognizerTest.cpp - GCNHazardRecognizer tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNHazardRecognizer.h"
#include "AMDGPUUnitTests.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class GCNHazardRecognizerTest : public AMDGPUCodeGenTestBase {
protected:
  void SetUp() override { setUpImpl("amdgcn-amd-amdhsa", "gfx1250", ""); }

  // Parses \p Body, the block of a function named f, emits everything before
  // the last instruction above the terminator as a post-RA scheduler would,
  // stalls \p Stalls cycles, and asks about that last instruction. Sixteen
  // stalls cover the data hazards of these tests, but no co-execution window,
  // which counts VALUs.
  ScheduleHazardRecognizer::HazardType hazardAtLast(StringRef Body,
                                                    int Stalls = 16) {
    bool Parsed = parseMIR("---\nname: f\nbody: |\n  bb.0:\n" + Body.str());
    EXPECT_TRUE(Parsed);
    if (!Parsed)
      report_fatal_error("cannot parse the test module");
    MachineFunction &MF = getMF("f");
    MachineBasicBlock &MBB = MF.front();
    MachineInstr &Last = *std::prev(MBB.getFirstTerminator());
    GCNHazardRecognizer HR(MF, GCNHazardRecognizer::OperatingMode::PostRA);
    for (MachineInstr &MI :
         make_range(MBB.begin(), MachineBasicBlock::iterator(Last))) {
      HR.EmitInstruction(&MI);
      HR.AdvanceCycle();
    }
    for (int Stall = 0; Stall != Stalls; ++Stall)
      HR.AdvanceCycle();
    SUnit SU(&Last, 0);
    return HR.getHazardType(&SU, 0);
  }
};

// Two WMMA results the add reads are in flight: the FP8 one is one VALU back
// and asks for one, the IU8 one is two back and asks for eight. The IU8
// decides.
TEST_F(GCNHazardRecognizerTest, SchedulerTakesLargestWMMACoexecutionDeficit) {
  EXPECT_EQ(hazardAtLast(R"(
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_I32_16X16X64_IU8_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    early-clobber $vgpr40_vgpr41_vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47 = V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr $vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31, $vgpr32_vgpr33_vgpr34_vgpr35_vgpr36_vgpr37_vgpr38_vgpr39, 8, $vgpr40_vgpr41_vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47, 0, 0, 0, 0, implicit $exec
    $vgpr64 = V_ADD_U32_e32 1, $vgpr64, implicit $exec
    $vgpr72 = V_ADD_U32_e32 $vgpr16, $vgpr40, implicit $exec
    S_ENDPGM 0
)"),
            ScheduleHazardRecognizer::Hazard);
}

// Seven VALUs after the FP8 one put the IU8 one eight back: nothing is left.
TEST_F(GCNHazardRecognizerTest, SchedulerWMMACoexecutionDeficitCovered) {
  EXPECT_EQ(hazardAtLast(R"(
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_I32_16X16X64_IU8_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    early-clobber $vgpr40_vgpr41_vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47 = V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr $vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31, $vgpr32_vgpr33_vgpr34_vgpr35_vgpr36_vgpr37_vgpr38_vgpr39, 8, $vgpr40_vgpr41_vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47, 0, 0, 0, 0, implicit $exec
    $vgpr64 = V_ADD_U32_e32 1, $vgpr64, implicit $exec
    $vgpr65 = V_ADD_U32_e32 1, $vgpr65, implicit $exec
    $vgpr66 = V_ADD_U32_e32 1, $vgpr66, implicit $exec
    $vgpr67 = V_ADD_U32_e32 1, $vgpr67, implicit $exec
    $vgpr68 = V_ADD_U32_e32 1, $vgpr68, implicit $exec
    $vgpr69 = V_ADD_U32_e32 1, $vgpr69, implicit $exec
    $vgpr70 = V_ADD_U32_e32 1, $vgpr70, implicit $exec
    $vgpr72 = V_ADD_U32_e32 $vgpr16, $vgpr40, implicit $exec
    S_ENDPGM 0
)"),
            ScheduleHazardRecognizer::NoHazard);
}

class GCNHazardRecognizerGFX90ATest : public GCNHazardRecognizerTest {
protected:
  void SetUp() override { setUpImpl("amdgcn-amd-amdhsa", "gfx90a", ""); }
};

// S_NOP 2 is three wait states, so the lane select is three back where the read
// needs four.
TEST_F(GCNHazardRecognizerGFX90ATest, SchedulerCountsSNopOnce) {
  EXPECT_EQ(hazardAtLast(R"(
    $sgpr0 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec
    S_NOP 2
    $sgpr1 = V_READLANE_B32 $vgpr2, $sgpr0
    S_ENDPGM 0
)",
                         /*Stalls=*/0),
            ScheduleHazardRecognizer::Hazard);
}

// The same for a window: S_NOP 6 is seven wait states of the eleven a read of
// a 16x16 MFMA result needs. Using an AGPR makes the history long enough for
// MFMA windows.
TEST_F(GCNHazardRecognizerGFX90ATest, SchedulerWindowCountsSNopOnce) {
  EXPECT_EQ(hazardAtLast(R"(
    $agpr0 = V_ACCVGPR_WRITE_B32_e64 $vgpr20, implicit $exec
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X16F16_vgprcd_e64 $vgpr8_vgpr9, $vgpr10_vgpr11, $vgpr0_vgpr1_vgpr2_vgpr3, 0, 0, 0, implicit $mode, implicit $exec
    S_NOP 6
    $vgpr4 = V_MOV_B32_e32 $vgpr2, implicit $exec
    S_ENDPGM 0
)",
                         /*Stalls=*/0),
            ScheduleHazardRecognizer::Hazard);
}

// S_NOP 3 is four wait states, all the read needs.
TEST_F(GCNHazardRecognizerGFX90ATest, SchedulerSNopCovers) {
  EXPECT_EQ(hazardAtLast(R"(
    $sgpr0 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec
    S_NOP 3
    $sgpr1 = V_READLANE_B32 $vgpr2, $sgpr0
    S_ENDPGM 0
)",
                         /*Stalls=*/0),
            ScheduleHazardRecognizer::NoHazard);
}

// S_NOP 6 and S_NOP 3 are the eleven wait states the read needs.
TEST_F(GCNHazardRecognizerGFX90ATest, SchedulerSNopsCoverWindow) {
  EXPECT_EQ(hazardAtLast(R"(
    $agpr0 = V_ACCVGPR_WRITE_B32_e64 $vgpr20, implicit $exec
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X16F16_vgprcd_e64 $vgpr8_vgpr9, $vgpr10_vgpr11, $vgpr0_vgpr1_vgpr2_vgpr3, 0, 0, 0, implicit $mode, implicit $exec
    S_NOP 6
    S_NOP 3
    $vgpr4 = V_MOV_B32_e32 $vgpr2, implicit $exec
    S_ENDPGM 0
)",
                         /*Stalls=*/0),
            ScheduleHazardRecognizer::NoHazard);
}

// A bundle records its meta instructions too, and they take no wait state:
// S_NOP 1 and the V_MOV put the lane select three back.
TEST_F(GCNHazardRecognizerGFX90ATest, SchedulerBundledMetaTakesNoWaitState) {
  EXPECT_EQ(hazardAtLast(R"(
    $sgpr0 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec
    S_NOP 1
    BUNDLE implicit-def $vgpr5, implicit-def $vgpr6, implicit $exec {
      $vgpr5 = IMPLICIT_DEF
      $vgpr6 = V_MOV_B32_e32 1, implicit $exec
    }
    $sgpr1 = V_READLANE_B32 $vgpr2, $sgpr0
    S_ENDPGM 0
)",
                         /*Stalls=*/0),
            ScheduleHazardRecognizer::Hazard);
}

// The same for a window: S_NOP 6, S_NOP 1 and the V_MOV are ten of the eleven
// wait states the read needs.
TEST_F(GCNHazardRecognizerGFX90ATest,
       SchedulerWindowBundledMetaTakesNoWaitState) {
  EXPECT_EQ(hazardAtLast(R"(
    $agpr0 = V_ACCVGPR_WRITE_B32_e64 $vgpr20, implicit $exec
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X16F16_vgprcd_e64 $vgpr8_vgpr9, $vgpr10_vgpr11, $vgpr0_vgpr1_vgpr2_vgpr3, 0, 0, 0, implicit $mode, implicit $exec
    S_NOP 6
    S_NOP 1
    BUNDLE implicit-def $vgpr30, implicit-def $vgpr31, implicit $exec {
      $vgpr30 = IMPLICIT_DEF
      $vgpr31 = V_MOV_B32_e32 1, implicit $exec
    }
    $vgpr4 = V_MOV_B32_e32 $vgpr2, implicit $exec
    S_ENDPGM 0
)",
                         /*Stalls=*/0),
            ScheduleHazardRecognizer::Hazard);
}

} // end anonymous namespace
