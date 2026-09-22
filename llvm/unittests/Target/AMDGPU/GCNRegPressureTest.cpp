//===- GCNRegPressureTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNRegPressure.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "LIRP.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Passes/PassBuilder.h"
#include "gtest/gtest.h"

using namespace llvm;

class GCNRegPressureTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu9.08--", "", ""); }
};

TEST_F(GCNRegPressureTest, DownwardTrackerEndOnDbgVal) {
  StringRef MIR = R"(
name:            DownwardTrackerEndOnDbgVal
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = IMPLICIT_DEF
  
  bb.1:
    DBG_VALUE %0
    DBG_VALUE %1
    %2:vgpr_32 = IMPLICIT_DEF
  
  bb.3:
    S_NOP 0, implicit %0, implicit %1, implicit %2
    S_ENDPGM 0
...
)";
  EXPECT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("DownwardTrackerEndOnDbgVal");
  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  // MBB1 live-in pressure is equivalent to MBB0 live-out pressure.
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  GCNRPTracker::LiveRegSet MBB1LiveIns =
      getLiveRegs(LIS.getInstructionIndex(*MBB0.rbegin()).getDeadSlot(), LIS,
                  MF.getRegInfo());

  // Track pressure across MBB1.
  {
    GCNDownwardRPTracker RPTracker(LIS), RPTrackerNoLiveIns(LIS);

    // There is a non-debug instruction in bb.1 (%2's def), so advance should
    // return true.
    EXPECT_TRUE(RPTracker.advance(MBB1.begin(), MBB1.end(), &MBB1LiveIns));
    EXPECT_TRUE(RPTrackerNoLiveIns.advance(MBB1.begin(), MBB1.end(), nullptr));

    // When advance returns true, maximum pressure should be the pressured
    // induced by the block's live-ins plus %2's def i.e., 3 VGPRs.
    EXPECT_EQ(RPTracker.moveMaxPressure().getVGPRNum(false), 3U);
    EXPECT_EQ(RPTrackerNoLiveIns.moveMaxPressure().getVGPRNum(false), 3U);
  }

  // Track pressure just across the first debug value of bb.1.
  {
    MachineBasicBlock::iterator Dbg1 = std::next(MBB1.begin());
    GCNDownwardRPTracker RPTracker(LIS), RPTrackerNoLiveIns(LIS);

    // The following unpacks a call to
    // advance(*MBB1.begin(), Dbg1, [MBB1LiveIns|nullptr])
    // which would return false in this case.
    //
    // There aren't any non-debug instruction between the beginning of bb1 and
    // Dbg1 (exclusive), the reset is therefore unsuccessful. The advance caller
    // returns early on a failure to reset. Calling advance after this does
    // nothing and produces false because the internal iterator already points
    // to the second debug instruction.
    EXPECT_FALSE(RPTracker.reset(*MBB1.begin(), Dbg1, &MBB1LiveIns));
    EXPECT_FALSE(RPTrackerNoLiveIns.reset(*MBB1.begin(), Dbg1, nullptr));
    EXPECT_FALSE(RPTracker.advance(Dbg1));
    EXPECT_FALSE(RPTrackerNoLiveIns.advance(Dbg1));

    // Register pressure should be the one at the block's live-ins.
    EXPECT_EQ(RPTracker.moveMaxPressure().getVGPRNum(false), 2U);
    EXPECT_EQ(RPTrackerNoLiveIns.moveMaxPressure().getVGPRNum(false), 2U);
  }
}

TEST_F(GCNRegPressureTest, DownwardTrackerAllDbgVal) {
  StringRef MIR = R"(
name:            DownwardTrackerAllDbgVal
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF

  bb.1:
    DBG_VALUE %0
  
  bb.2:
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";
  EXPECT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("DownwardTrackerAllDbgVal");
  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  // MBB1 live-in pressure is equivalent to MBB0 live-out pressure.
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  GCNRPTracker::LiveRegSet MBB1LiveIns =
      getLiveRegs(LIS.getInstructionIndex(*MBB0.rbegin()).getDeadSlot(), LIS,
                  MF.getRegInfo());

  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  GCNDownwardRPTracker RPTracker(LIS), RPTrackerNoLiveIns(LIS);

  // The following unpacks a call to
  // advance(MBB1.begin(), MBB1.end(), [MBB1LiveIns|nullptr])
  // which would return false in this case.
  //
  // There aren't any non-debug instruction in bb.2, the reset is therefore
  // unsuccessful. The advance caller returns early on a failure to reset.
  // Calling advance after this does nothing and produces false because the
  // internal iterator is already at the block's end.
  EXPECT_FALSE(RPTracker.reset(*MBB1.begin(), MBB1.end(), &MBB1LiveIns));
  EXPECT_FALSE(RPTrackerNoLiveIns.reset(*MBB1.begin(), MBB1.end(), nullptr));
  EXPECT_FALSE(RPTracker.advance(MBB1.end()));
  EXPECT_FALSE(RPTrackerNoLiveIns.advance(MBB1.end()));

  // Register pressure should be the one at the block's live-ins.
  EXPECT_EQ(RPTracker.moveMaxPressure().getVGPRNum(false), 1U);
  EXPECT_EQ(RPTrackerNoLiveIns.moveMaxPressure().getVGPRNum(false), 1U);
}

// Tests the correct handling of multiple uses of the same virtual register
// in bumpDownwardPressure (speculative estimate of register pressure).
TEST_F(GCNRegPressureTest, BumpDownwardPressureLastUseAfterCommit) {
  StringRef MIR = R"(
name:            BumpDownwardPressureLastUseAfterCommit
tracksRegLiveness: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vreg_256_align2 = IMPLICIT_DEF
    S_NOP 0, implicit %1
    S_NOP 0, implicit %1
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("BumpDownwardPressureLastUseAfterCommit");
  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : MBB)
    Instrs.push_back(&MI);
  // 0: def %0, 1: def %1, 2: U1 (use %1), 3: U2 (last use %1),
  // 4: use %0, 5: S_ENDPGM
  MachineInstr *DefV0 = Instrs[0];
  MachineInstr *DefV1 = Instrs[1];
  MachineInstr *U1 = Instrs[2];
  MachineInstr *U2 = Instrs[3];

  GCNDownwardRPTracker RPTracker(LIS);
  GCNRPTracker::LiveRegSet Empty;
  RPTracker.reset(MRI, Empty);

  // Commit the defs and the first use of %1 via the externally-managed
  // iterator (same as while scheduling).
  RPTracker.advance(DefV0, /*UseInternalIterator=*/false);
  RPTracker.advance(DefV1, /*UseInternalIterator=*/false);
  RPTracker.advance(U1, /*UseInternalIterator=*/false);

  // After committing U1, both %0 (1 VGPR) and %1 (vreg_256 = 8 VGPRs) are live.
  EXPECT_EQ(RPTracker.getPressure().getArchVGPRNum(), 9U);

  // Speculate the last use of %1. %1 must die here, dropping its 8 VGPRs and
  // leaving only %0 live.
  GCNRegPressure P = RPTracker.bumpDownwardPressure(U2, TRI);
  EXPECT_EQ(P.getArchVGPRNum(), 1U);
}

// Tests bumpDownwardPressure for an instruction that uses and redefines
// the same register.
TEST_F(GCNRegPressureTest, BumpDownwardPressureUseAndRedef) {
  StringRef MIR = R"(
name:            BumpDownwardPressureUseAndRedef
tracksRegLiveness: true
body:             |
  bb.0:
    %0:sgpr_32 = IMPLICIT_DEF
    %0:sgpr_32 = S_OR_B32 %0:sgpr_32, 1, implicit-def dead $scc
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("BumpDownwardPressureUseAndRedef");
  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : MBB)
    Instrs.push_back(&MI);
  // 0: def %0 (sgpr_32 = 1 SGPR)
  // 1: use+redef of %0
  // 2: use %0
  // 3: S_ENDPGM
  MachineInstr *DefS0 = Instrs[0];
  MachineInstr *UseRedef = Instrs[1];

  GCNDownwardRPTracker RPTracker(LIS);
  GCNRPTracker::LiveRegSet Empty;
  RPTracker.reset(MRI, Empty);

  // Commit the def; %0 occupies 1 SGPR and stays live.
  RPTracker.advance(DefS0, /*UseInternalIterator=*/false);
  EXPECT_EQ(RPTracker.getPressure().getSGPRNum(), 1U);

  // Speculate the instruction. It uses and redefines %0, which stays live, so
  // pressure must be unchanged.
  GCNRegPressure P = RPTracker.bumpDownwardPressure(UseRedef, TRI);
  EXPECT_EQ(P.getSGPRNum(), 1U);
}

class LIRPAllocTableTest : public llvm::CodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgcn--", "gfx1250", ""); }

  MachineFunction &getEmptyMF(StringRef Name) {
    std::string MIR =
        ("name: " + Name +
         "\ntracksRegLiveness: true\n"
         "body: |\n  bb.0:\n    S_ENDPGM 0\n...\n")
            .str();
    EXPECT_TRUE(parseMIR(MIR));
    return getMF(Name);
  }
};

// LIRP: reg pressure computation.
TEST_F(LIRPAllocTableTest, MaxSlotPressure) {
  MachineFunction &MF = getEmptyMF("maxslot");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  auto V = [&] { return MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass); };
  Register A = V(), B = V(), C = V(), D = V(), E = V();

  LIRPAllocTable T(MRI);
  // Schedule (defs and last uses), in order:
  T.open(A, 1);   // A def
  T.open(B, 1);   // B def
  T.close(A, 1);  // A last use (gap begins)
  T.open(C, 1);   // C def
  T.close(B, 1);  // B last use
  T.open(D, 1);   // D def
  T.close(C, 1);  // C last use
  T.open(E, 1);   // E def
  EXPECT_EQ(T.getRegNum(), 2U);

  T.open(A, 1);   // A redef (5-cycle interference graph) ->
                  // conflict -> rebuild
  EXPECT_EQ(T.getRegNum(), 3U);
}

// LIRP: a wide value cannot reuse a too-small gap.
TEST_F(LIRPAllocTableTest, TupleFragmentation) {
  MachineFunction &MF = getEmptyMF("frag");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  auto V32 = [&] { return MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass); };
  auto V128 = [&] {
    return MRI.createVirtualRegister(&AMDGPU::VReg_128RegClass);
  };

  Register A = V32(), B = V32(), W = V128();
  const uint64_t S0 = 1;       // one 32-bit slot
  const uint64_t W4 = 0x0F;    // four contiguous 32-bit slots

  LIRPAllocTable T(MRI);
  T.open(A, S0);   // slot 0
  T.open(B, S0);   // slot 1
  T.close(A, S0);  // free slot 0 (a 1-wide gap)
  T.open(W, W4);   // needs 4 contiguous slots; cannot use slot-0 hole
  // B occupies slot 1, so W must start at slot 2 -> uses [2,6) -> 6 slots.
  EXPECT_EQ(T.getRegNum(), 6U);
}

// LIRP: alignment forces a wide value past an odd free slot.
TEST_F(LIRPAllocTableTest, AlignedWidePastOddSlot) {
  MachineFunction &MF = getEmptyMF("aligned");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  auto V32 = [&] { return MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass); };
  auto V256A2 = [&] {
    return MRI.createVirtualRegister(&AMDGPU::VReg_256_Align2RegClass);
  };

  Register A = V32(), W = V256A2();
  const uint64_t S0 = 1;           // one 32-bit slot
  const uint64_t W8 = 0xFF;        // eight contiguous 32-bit slots

  LIRPAllocTable T(MRI);
  T.open(A, S0);   // slot 0
  T.open(W, W8);   // 8 wide, align 2 -> base 2 -> [2,10)
  EXPECT_EQ(T.getRegNum(), 10U);
}

// LIRP: a partial (single-slot) redef of a tuple conflicts with a value
// that reused one of its slots.
TEST_F(LIRPAllocTableTest, PartialRedef) {
  MachineFunction &MF = getEmptyMF("partialredef");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  Register A = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
  Register B = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  const uint64_t S0 = 1;      // slot offset 0 (sub0 of A)
  const uint64_t A2 = 0x03;   // both of A's slots {0,1}

  LIRPAllocTable T(MRI);
  T.open(A, A2);   // A -> base 0, slots {0,1}
  T.close(A, A2);  // free 0,1
  T.open(B, S0);   // B -> slot 0 (reuse)
  T.open(A, S0);   // A redefs slot 0 (owned by B) -> rebuild
  // Rebuild places B at slot 1, max == 2.
  EXPECT_EQ(T.getRegNum(), 2U);

  T.close(B, S0);
  T.close(A, S0);
}

// LIRP: tracker routing (SGPR vs VGPR).
TEST_F(LIRPAllocTableTest, SlotTrackerRouting) {
  MachineFunction &MF = getEmptyMF("routing");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  Register V0 = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register V1 = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register S0 = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  auto Full = [&](Register R) { return MRI.getMaxLaneMaskForVReg(R); };
  LaneBitmask None = LaneBitmask::getNone();

  LIRPTracker T;
  LIRPTracker::LiveRegSet NoLiveIns;
  T.reset(MRI, NoLiveIns);
  T.update(V0, None, Full(V0));
  T.update(S0, None, Full(S0));
  T.update(V1, None, Full(V1));

  EXPECT_EQ(T.getVGPRNum(), 2U);
  EXPECT_EQ(T.getSGPRNum(), 1U);
}

// LIRP: speculate reg pressure.
TEST_F(LIRPAllocTableTest, SlotTrackerSpeculate) {
  MachineFunction &MF = getEmptyMF("speculate");
  MachineRegisterInfo &MRI = MF.getRegInfo();

  Register A = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register B = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register C = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  auto Full = [&](Register R) { return MRI.getMaxLaneMaskForVReg(R); };
  using Event = LIRPTracker::Event;

  LIRPTracker T;
  LIRPTracker::LiveRegSet NoLiveIns;
  T.reset(MRI, NoLiveIns);
  T.update(A, LaneBitmask::getNone(), Full(A));  // A live, slot 0
  T.update(B, LaneBitmask::getNone(), Full(B));  // B live, slot 1
  EXPECT_EQ(T.getVGPRNum(), 2U);

  // Speculate: close A, open C (new). C reuses A's freed slot -> still 2.
  LaneBitmask None = LaneBitmask::getNone();
  Event CloseA{A, Full(A), None}, OpenC{C, None, Full(C)};
  auto [Sgpr, Vgpr] = T.speculate({CloseA, OpenC});
  EXPECT_EQ(Vgpr, 2U);
  EXPECT_EQ(Sgpr, 0U);
  EXPECT_EQ(T.getVGPRNum(), 2U);

  // Applying the same batch results in the the same pressure.
  T.update(A, Full(A), LaneBitmask::getNone());
  T.update(C, LaneBitmask::getNone(), Full(C));
  EXPECT_EQ(T.getVGPRNum(), 2U);

  // Speculate opening new vregs.
  Register D = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register E = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Event OpenD{D, None, Full(D)}, OpenE{E, None, Full(E)};
  auto [Sgpr2, Vgpr2] = T.speculate({OpenD, OpenE});
  EXPECT_EQ(Vgpr2, 4U);
  EXPECT_EQ(Sgpr2, 0U);
  EXPECT_EQ(T.getVGPRNum(), 2U);

  T.update(D, LaneBitmask::getNone(), Full(D));
  T.update(E, LaneBitmask::getNone(), Full(E));
  EXPECT_EQ(T.getVGPRNum(), 4U);
}

// LIRP: true16 handling.
TEST_F(LIRPAllocTableTest, SlotTrackerPartialLane) {
  MachineFunction &MF = getEmptyMF("partiallane");
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  // sreg_64: two 32-bit slots (sub0/sub1), sub0 split into lo16/hi16 lanes.
  Register A = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
  auto Full = [&](Register R) { return MRI.getMaxLaneMaskForVReg(R); };
  LaneBitmask Sub0Lo = TRI->getSubRegIndexLaneMask(AMDGPU::lo16);
  LaneBitmask Sub0Hi = TRI->getSubRegIndexLaneMask(AMDGPU::hi16);
  LaneBitmask None = LaneBitmask::getNone();

  LIRPTracker T;
  LIRPTracker::LiveRegSet NoLiveIns;
  T.reset(MRI, NoLiveIns);

  // allocate register -> 2 slots
  T.update(A, None, Full(A));
  EXPECT_EQ(T.getSGPRNum(), 2U);

  // close sub0's high 16 bits,
  // sub0's low half is still live -> slot stays occupied
  LaneBitmask PrevMask = Full(A);
  LaneBitmask NewMask = Full(A) & ~Sub0Hi;
  T.update(A, PrevMask, NewMask);
  EXPECT_EQ(T.getSGPRNum(), 2U);

  // allocate new slot
  Register B = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  T.update(B, None, Full(B));
  EXPECT_EQ(T.getSGPRNum(), 3U);

  // close sub0's low 16 bits
  PrevMask = NewMask;
  NewMask &= ~Sub0Lo;
  T.update(A, PrevMask, NewMask);
  EXPECT_EQ(T.getSGPRNum(), 3U);

  // C reuses the slot freed by A's sub0.
  Register C = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  T.update(C, None, Full(C));
  EXPECT_EQ(T.getSGPRNum(), 3U);
}

TEST(RangeUnionTest, OpenClose) {
  RangeUnion S;
  EXPECT_TRUE(S.empty());
  EXPECT_FALSE(S.isOpen());

  S.appendOpen(10);
  EXPECT_FALSE(S.empty());
  EXPECT_TRUE(S.isOpen()); // last segment ends at Inf
  EXPECT_EQ(S.segments().back().first, 10U);
  EXPECT_EQ(S.segments().back().second, RangeUnion::Inf);

  S.closeLast(20);
  EXPECT_FALSE(S.isOpen());
  EXPECT_EQ(S.segments().back().second, 20U);

  S.appendOpen(30);
  EXPECT_TRUE(S.isOpen());
  EXPECT_EQ(S.segments().size(), 2U);
  EXPECT_EQ(S.segments().back().first, 30U);
}

TEST(RangeUnionTest, Overlaps) {
  RangeUnion U;
  U.append(0, 10);
  U.append(20, 30);

  auto Span = [](unsigned Lo, unsigned Hi) {
    RangeUnion S;
    S.append(Lo, Hi);
    return S;
  };

  EXPECT_FALSE(U.overlaps(Span(10, 20)));
  EXPECT_TRUE(U.overlaps(Span(15, 25)));
  EXPECT_TRUE(U.overlaps(Span(2, 5)));
  EXPECT_FALSE(U.overlaps(Span(30, 40)));
  EXPECT_TRUE(U.overlaps(Span(25, RangeUnion::Inf)));
  EXPECT_FALSE(U.overlaps(Span(30, RangeUnion::Inf)));
}

TEST(RangeUnionTest, Merge) {
  RangeUnion U;
  U.append(0, 10);

  RangeUnion Other;
  Other.append(20, 30);

  U.merge(Other);
  ASSERT_EQ(U.segments().size(), 2U);
  EXPECT_EQ(U.segments()[0], std::make_pair(0u, 10u));
  EXPECT_EQ(U.segments()[1], std::make_pair(20u, 30u));

  RangeUnion Other2;
  Other2.append(12, 15);
  U.merge(Other2);
  ASSERT_EQ(U.segments().size(), 3U);
  EXPECT_EQ(U.segments()[0], std::make_pair(0u, 10u));
  EXPECT_EQ(U.segments()[1], std::make_pair(12u, 15u));
  EXPECT_EQ(U.segments()[2], std::make_pair(20u, 30u));
}

TEST(RangeUnionTest, MergeCoalesce) {
  {
    RangeUnion A, B;
    A.append(0, 10);
    B.append(5, 20);
    A.merge(B);
    ASSERT_EQ(A.segments().size(), 1U);
    EXPECT_EQ(A.segments()[0], std::make_pair(0u, 20u));
  }
  {
    RangeUnion A, B;
    A.append(0, 10);
    B.append(10, 20);
    A.merge(B);
    ASSERT_EQ(A.segments().size(), 1U);
    EXPECT_EQ(A.segments()[0], std::make_pair(0u, 20u));
  }
  {
    RangeUnion A, B;
    A.appendOpen(10);      // [10, Inf)
    B.append(20, 30);      // inside the open range
    A.merge(B);
    ASSERT_EQ(A.segments().size(), 1U);
    EXPECT_EQ(A.segments()[0], std::make_pair(10u, RangeUnion::Inf));
  }
  {
    RangeUnion A, B;
    A.append(0, 10);
    B.append(20, 30);
    A.merge(B);
    EXPECT_EQ(A.segments().size(), 2U);
  }
  {
    // A single incoming segment that bridges several existing ones coalesces
    // them all into one.
    RangeUnion A, B;
    A.append(0, 10);
    A.append(20, 30);
    A.append(40, 50);
    B.append(5, 45); // spans into [0,10), across [20,30), into [40,50)
    A.merge(B);
    ASSERT_EQ(A.segments().size(), 1U);
    EXPECT_EQ(A.segments()[0], std::make_pair(0u, 50u));
  }
  {
    // Bridge exactly two segments via adjacency (touch, not overlap).
    RangeUnion A, B;
    A.append(0, 10);
    A.append(20, 30);
    B.append(10, 20);
    A.merge(B);
    ASSERT_EQ(A.segments().size(), 1U);
    EXPECT_EQ(A.segments()[0], std::make_pair(0u, 30u));
  }
}
