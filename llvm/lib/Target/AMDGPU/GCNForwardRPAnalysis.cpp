//===- GCNForwardRPAnalysis.cpp - Forward register pressure analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass walks a machine function in forward (program) order and compute
/// the SGPR and VGPR register pressure after each non-debug instruction using
/// GCNDownwardRPTracker. We need to know these information before the 1st
/// register allocation pass, which allocates the per-lane VGPRs: we need to
/// know how many VGPRs we need to reserve for SGPR-spilling. However, the SGPR
/// pressure computed by GCNDownwardRPTracker is not accurate because it does
/// not account for the virtual SGPRs that will be created during
/// AMDGPUWavenTransform, which happens after the 1st register allocation pass.
/// BranchMaskIntervals and JoinMaskIntervals are created trying to estimate
/// the lifetime of those lane-mask virtual SGPRs that AMDGPUWavenTransform
/// likely will create.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-forward-rp"

namespace {

typedef struct LaneMaskInterval {
  SlotIndex Start;
  SlotIndex End;
} LaneMaskInterval;

class GCNForwardRPAnalysis : public MachineFunctionPass {
public:
  static char ID;

  GCNForwardRPAnalysis() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "GCN Forward Register Pressure Analysis";
  }
};

} // end anonymous namespace

char GCNForwardRPAnalysis::ID = 0;

INITIALIZE_PASS_BEGIN(GCNForwardRPAnalysis, "gcn-forward-rp",
                      "GCN Forward Register Pressure Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(GCNForwardRPAnalysis, "gcn-forward-rp",
                    "GCN Forward Register Pressure Analysis", false, true)

char &llvm::GCNForwardRPAnalysisID = GCNForwardRPAnalysis::ID;

FunctionPass *llvm::createGCNForwardRPAnalysisPass() {
  return new GCNForwardRPAnalysis();
}

bool GCNForwardRPAnalysis::runOnMachineFunction(MachineFunction &MF) {
  const LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  const MachineLoopInfo &MLI =
      getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  const MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  const MachinePostDominatorTree &MPDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();

  errs() << "=== GCN Forward RP Analysis: " << MF.getName() << " ===\n";

  std::vector<LaneMaskInterval> BranchMaskIntervals;
  std::map<SlotIndex, LaneMaskInterval> JoinMaskIntervals;

  for (const MachineBasicBlock &MBB : MF) {
    if (MBB.empty())
      continue;

    auto TerminatorMI = MBB.getFirstTerminator();
    if (TerminatorMI == MBB.end())
      continue;
    if (TerminatorMI->getOpcode() != AMDGPU::SI_BRCOND &&
        TerminatorMI->getOpcode() != AMDGPU::SI_BRCOND_Z)
      continue;
    // get the Immediate Post Dominator of the MBB, which is the join point for
    // the branch
    auto *IPD = MPDT.getNode(&MBB)->getIDom();
    if (!IPD)
      continue;
    MachineBasicBlock *JoinMBB = IPD->getBlock();
    if (!JoinMBB)
      continue;
    // First create BranchMaskInterval for the branch instruction to the join
    // point.
    SlotIndex EndSlotIndex = LIS.getMBBStartIdx(JoinMBB);
    SlotIndex StartSlotIndex = LIS.getInstructionIndex(*TerminatorMI);
    // Need to extend interval for loop cases
    if (MLI.getLoopFor(&MBB) != MLI.getLoopFor(JoinMBB)) {
      SlotIndex LoopHeaderStart =
          LIS.getMBBStartIdx(MLI.getLoopFor(&MBB)->getHeader());
      if (StartSlotIndex > LoopHeaderStart)
        StartSlotIndex = LoopHeaderStart;
    }
    assert(StartSlotIndex < EndSlotIndex &&
           "Start index must be less than end index");
    BranchMaskIntervals.push_back({StartSlotIndex, EndSlotIndex});
    // Next create/update JoinMaskInterval for the join point to the branch
    // instruction.
    auto JoinIt = JoinMaskIntervals.find(EndSlotIndex);
    if (JoinIt == JoinMaskIntervals.end()) {
      JoinMaskIntervals[EndSlotIndex] = {StartSlotIndex, EndSlotIndex};
    } else {
      SlotIndex OldStartIndex = JoinIt->second.Start;
      MachineBasicBlock *OldStartMBB = LIS.getMBBFromIndex(OldStartIndex);
      MachineBasicBlock *NewStartMBB = LIS.getMBBFromIndex(StartSlotIndex);
      // Find the common dominator of the old start and new start.
      MachineBasicBlock *CommonDom =
          MDT.findNearestCommonDominator(OldStartMBB, NewStartMBB);
      if (CommonDom) {
        SlotIndex CommonDomStart = LIS.getMBBEndIdx(CommonDom);
        if (CommonDomStart < StartSlotIndex)
          StartSlotIndex = CommonDomStart;
      }
      if (StartSlotIndex < JoinIt->second.Start)
        JoinIt->second.Start = StartSlotIndex;
    }
  }

  // Now we want to have both BranchMaskIntervals and JoinMaskIntervals into a
  // single sorted list of intervals to process in order.
  std::vector<LaneMaskInterval> AllIntervals;
  for (const auto &Interval : BranchMaskIntervals) {
    AllIntervals.push_back(Interval);
  }
  for (const auto &Interval : JoinMaskIntervals) {
    AllIntervals.push_back(Interval.second);
  }
  std::sort(AllIntervals.begin(), AllIntervals.end(),
            [](const LaneMaskInterval &A, const LaneMaskInterval &B) {
              return A.Start < B.Start;
            });

  // Create a list of active intervals whose start is less than or equal to the
  // current slot index and whose end is greater than the current slot index.
  // The size of this list is the number of active lane masks at the current
  // slot index.
  unsigned TotalIntervals = AllIntervals.size();
  std::vector<LaneMaskInterval> ActiveIntervals;
  auto RemoveExpiredIntervals = [&](SlotIndex CurSlotIndex) {
    ActiveIntervals.erase(
        std::remove_if(ActiveIntervals.begin(), ActiveIntervals.end(),
                       [CurSlotIndex](const LaneMaskInterval &Interval) {
                         return Interval.End <= CurSlotIndex;
                       }),
        ActiveIntervals.end());
  };
  unsigned NextIntervalIndex = 0;
  auto AddNewlyActiveIntervals = [&](SlotIndex CurSlotIndex) {
    while (NextIntervalIndex < TotalIntervals &&
           AllIntervals[NextIntervalIndex].Start <= CurSlotIndex) {
      if (AllIntervals[NextIntervalIndex].End > CurSlotIndex) {
        ActiveIntervals.push_back(AllIntervals[NextIntervalIndex]);
      }
      NextIntervalIndex++;
    }
  };

  unsigned MaxSGPR = 0;
  unsigned MaxVGPR = 0;
  unsigned MaxLaneMaskSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    dbgs() << "\nBB#" << MBB.getNumber();
    if (const auto *BB = MBB.getBasicBlock())
      if (BB->hasName())
        dbgs() << " %" << BB->getName();
    dbgs() << ":\n";

    if (MBB.empty())
      continue;

    GCNDownwardRPTracker RPT(LIS);
    // Initialize tracker at the first instruction; fills LiveRegs from LIS.
    RPT.reset(MBB.front(), MBB.end());
    SlotIndex CurSlotIndex = LIS.getMBBStartIdx(&MBB);
    RemoveExpiredIntervals(CurSlotIndex);
    AddNewlyActiveIntervals(CurSlotIndex);

    dbgs() << "[Live-in] SGPR=" << RPT.getPressure().getSGPRNum()
           << " LaneMask=" << ActiveIntervals.size()
           << " VGPR=" << RPT.getPressure().getVGPRNum(false) << "\n";

    MaxLaneMaskSize =
        std::max(MaxLaneMaskSize, (unsigned)ActiveIntervals.size());

    // Walk forward: advanceBeforeNext returns true when the block end is
    // reached (no more non-debug instructions to process).
    while (!RPT.advanceBeforeNext()) {
      // getPressure() here is the pressure *before* the next instruction
      // (uses are already killed, but the def is not yet live).
      GCNRegPressure RPBefore = RPT.getPressure();

      // Commit the instruction: add defs to LiveRegs.
      RPT.advanceToNext();

      // getPressure() is now the pressure *after* the instruction.
      GCNRegPressure RPAfter = RPT.getPressure();

      const MachineInstr *MI = RPT.getLastTrackedMI();
      assert(MI && "advanceToNext must set LastTrackedMI");
      CurSlotIndex = LIS.getInstructionIndex(*MI);
      RemoveExpiredIntervals(CurSlotIndex);
      AddNewlyActiveIntervals(CurSlotIndex);
      MaxLaneMaskSize =
          std::max(MaxLaneMaskSize, (unsigned)ActiveIntervals.size());

      dbgs() << "  [after]  SGPR=" << RPAfter.getSGPRNum()
             << " LaneMask=" << ActiveIntervals.size()
             << " VGPR=" << RPAfter.getVGPRNum(false) << "\n";
    }

    MaxSGPR = std::max(MaxSGPR, RPT.getPressure().getSGPRNum());
    MaxVGPR = std::max(MaxVGPR, RPT.getPressure().getVGPRNum(false));
  }
  errs() << "=== Max SGPR=" << MaxSGPR
         << " LaneMask=" << MaxLaneMaskSize
         << " VGPR=" << MaxVGPR << " ===\n";

  // Analysis pass — never modifies the function.
  return false;
}
