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
  // and asks about that last instruction. Sixteen stalls cover the data
  // hazards of these tests, but no co-execution window, which counts VALUs.
  ScheduleHazardRecognizer::HazardType hazardAtLast(StringRef Body) {
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
    for (int Stall = 0; Stall != 16; ++Stall)
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

} // end anonymous namespace
