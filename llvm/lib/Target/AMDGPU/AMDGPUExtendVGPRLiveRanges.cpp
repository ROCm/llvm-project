//===-- AMDGPUExtendVGPRLiveRanges.cpp - Fix Phy-VGPR live-ranges ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass assumes that we have done register-allocation for per-thread VGPR
/// values. It extends the live-ranges of those physical VGPRs in order to
/// create the correct interference with those WWM/WQM values during the last
/// register-allocation pass for those WWM/WQM values.
//
// TODO: Ihis piece may still need some significant rework. As it is, we
// do not cap those extended physical live-ranges because we do not how to
// do it. Does it create too much extra interference for whole-wave-mode
// register allocation?
// Is there a different way to tackle this problem? Note that we really need
// is to extend those physical VGPRs live-range through those WWM definitions
// for the correct interference. But what is the right way to pass that info
// to register allocator? Can it be achieved by simply adding RegMask operands
// to those WWM definition instructions?
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "extend-vgpr-live-ranges"

namespace {

class AMDGPUExtendVGPRLiveRanges : public MachineFunctionPass {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  MachinePostDominatorTree *PDT;

  // Control dependencies map: for each block, the set of blocks it is directly
  // control-dependent on.
  DenseMap<MachineBasicBlock *, SmallPtrSet<MachineBasicBlock *, 2>> CtrlDeps;
  // All the divergent control block that influences one of the WWMBBs
  DenseSet<MachineBasicBlock *> DivergentCtrlBBs;

  void buildControlDependences(MachineFunction &MF);
  void findAllCtrlMBBs(MachineBasicBlock *DepMBB,
                       SmallPtrSetImpl<MachineBasicBlock *> &Result);
  //bool influences(MachineBasicBlock *CtrlMBB, MachineBasicBlock *DepMBB) {
  //  SmallPtrSet<MachineBasicBlock *, 8> AllCtrlMBBs;
  //  findAllCtrlMBBs(DepMBB, AllCtrlMBBs);
  //  return (AllCtrlMBBs.count(CtrlMBB));
  //}
  void findDivergentCtrlMBBs(MachineBasicBlock *DepMBB,
                             DenseSet<MachineBasicBlock *> &Result);

public:
  static char ID;

  AMDGPUExtendVGPRLiveRanges() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUExtendVGPRLiveRanges, DEBUG_TYPE,
                      "Extend VGPR Live Ranges", false, false)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUExtendVGPRLiveRanges, DEBUG_TYPE,
                    "Extend VGPR Live Ranges", false, false)

char AMDGPUExtendVGPRLiveRanges::ID = 0;

char &llvm::AMDGPUExtendVGPRLiveRangesID = AMDGPUExtendVGPRLiveRanges::ID;

FunctionPass *llvm::createAMDGPUExtendVGPRLiveRangesPass() {
  return new AMDGPUExtendVGPRLiveRanges();
}

static bool hasWWMInMBB(const MachineBasicBlock &MBB) {
  for (const MachineInstr &MI : MBB) {
    if (MI.getOpcode() == AMDGPU::V_SET_INACTIVE_B32 ||
        MI.getOpcode() == AMDGPU::SI_SPILL_S32_TO_VGPR ||
        MI.getOpcode() == AMDGPU::ENTER_STRICT_WWM ||
        MI.getOpcode() == AMDGPU::ENTER_STRICT_WQM) {
      return true;
    }
  }
  return false;
}

// TODO: A block ending with a divergent branch should have
// an instruction updating the EXEC register followed by a branch
// using EXEC as the condition.
static bool endsWithDivergentBranch(const MachineBasicBlock &MBB,
                                    const SIInstrInfo *TII) {
  MachineBasicBlock *TrueMBB = nullptr;
  MachineBasicBlock *FalseMBB = nullptr;
  SmallVector<MachineOperand, 1> Cond;
  TII->analyzeBranch(const_cast<MachineBasicBlock &>(MBB), TrueMBB, FalseMBB,
                     Cond);

  if (!Cond.size())
    return false;

  auto CondOpnd = Cond.back();
  if (CondOpnd.getReg() == AMDGPU::EXEC ||
      CondOpnd.getReg() == AMDGPU::EXEC_LO ||
      CondOpnd.getReg() == AMDGPU::EXEC_HI)
    return true;

  return false;
}

void AMDGPUExtendVGPRLiveRanges::findAllCtrlMBBs(
    MachineBasicBlock *DepMBB, SmallPtrSetImpl<MachineBasicBlock *> &Result) {
  Result.clear();
  if (CtrlDeps.find(DepMBB) == CtrlDeps.end())
    return;

  SmallVector<MachineBasicBlock *, 8> WL;
  for (auto *ParMBB : CtrlDeps[DepMBB]) {
    WL.push_back(ParMBB);
    Result.insert(ParMBB);
  }

  while (!WL.empty()) {
    auto *MBB = WL.back();
    WL.pop_back();
    if (CtrlDeps.find(MBB) == CtrlDeps.end())
      continue;
    for (auto *ParMBB : CtrlDeps[DepMBB]) {
      if (Result.count(ParMBB))
        continue;
      WL.push_back(ParMBB);
      Result.insert(ParMBB);
    }
  }
}

void AMDGPUExtendVGPRLiveRanges::findDivergentCtrlMBBs(
    MachineBasicBlock *DepMBB, DenseSet<MachineBasicBlock *> &Result) {
  SmallPtrSet<MachineBasicBlock *, 8> AllCtrlMBBs;
  findAllCtrlMBBs(DepMBB, AllCtrlMBBs);
  for (auto *MBB : AllCtrlMBBs) {
    if (endsWithDivergentBranch(*MBB, TII)) {
      Result.insert(MBB);
    }
  }
}

// Build the control-dependence graph for the function. Also find all the
// divergent control blocks that influence WWM blocks.
void AMDGPUExtendVGPRLiveRanges::buildControlDependences(MachineFunction &MF) {
  DivergentCtrlBBs.clear();
  CtrlDeps.clear();
  // Set of blocks that contain instructions that may write to VGPR in
  // whole-wave mode.
  DenseSet<MachineBasicBlock *> WWMBBs;
  for (auto *MBB : nodes(&MF)) {
    // skip
    if (MBB->getSingleSuccessor())
      continue;

    // For each successor of MBB
    for (auto *SuccMBB : MBB->successors()) {
      auto *PostDomMBB = PDT->findNearestCommonDominator(MBB, SuccMBB);
      if (PostDomMBB == MBB) {
        if (auto *ParentNode = PDT->getNode(MBB)->getIDom())
          PostDomMBB = ParentNode->getBlock();
      }
      // walk PDT from SuccMBB to PostDomMBB
      // add MBB as the control-parent of the blocks along the path (except
      // PostDomBB)
      for (auto *Node = PDT->getNode(SuccMBB);
           Node && Node->getBlock() != PostDomMBB; Node = Node->getIDom()) {
        auto *PathMBB = Node->getBlock();
        CtrlDeps[PathMBB].insert(MBB);
        if (hasWWMInMBB(*PathMBB))
          WWMBBs.insert(PathMBB);
      }
    }
  }
  for (auto *WWMMBB : WWMBBs) {
    findDivergentCtrlMBBs(WWMMBB, DivergentCtrlBBs);
  }
}

bool AMDGPUExtendVGPRLiveRanges::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "AMDGPUExtendVGPRLiveRanges: function " << MF.getName()
                    << "\n");

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  PDT = &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();

  buildControlDependences(MF);

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  bool Changed = false;

  // Iterate through the CFG in RPO oder.
  for (MachineBasicBlock *MBB : RPOT) {
    SmallPtrSet<MachineBasicBlock *, 3> CtrlBBs;
    findAllCtrlMBBs(MBB, CtrlBBs);
    // Find the union of the live RegUnits at the immediate post-dominators
    // of those control-blocks.
    LiveRegUnits LiveRegs(*TRI);
    for (auto *CtrlMBB : CtrlBBs) {
      // Only counts control-blocks that are divergent and influence WWMBBs.
      if (!DivergentCtrlBBs.count(CtrlMBB))
        continue;
      auto *IPD = PDT->getNode(CtrlMBB)->getIDom()->getBlock();
      LiveRegs.addLiveIns(*IPD);
    }
    if (LiveRegs.empty())
      continue;
    // Iterate through all the MachineInstr in MBB, check their defs against
    // LiveRegs, and extend the live-ranges if needed.
    for (MachineInstr &MI : *MBB) {
      for (MachineOperand &MO : MI.defs()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        if (!TRI->isVectorRegister(*MRI, MO.getReg()))
          continue;
        MCPhysReg PhysReg = MO.getReg();
        if (LiveRegs.available(PhysReg))
          continue;
        // Add implicit use to extend the live-range of PhysReg.
        bool UseExists = false;
        for (auto Opnd : MI.all_uses()) {
          if (Opnd.isReg() && Opnd.getReg() == PhysReg) {
            UseExists = true;
            break;
          }
        }
        if (!UseExists) {
          MI.addOperand(MF, MachineOperand::CreateReg(PhysReg, false, true));
          Changed = true;
        }
      }
    }
  }

  if (Changed) {
    // recompute liveness
    std::vector<MachineBasicBlock *> PostOrder;
    for (auto MBB : reverse(RPOT)) {
      PostOrder.push_back(MBB);
    }
    fullyRecomputeLiveIns(PostOrder);
    for (auto *MBB : RPOT) {
      recomputeLivenessFlags(*MBB);
    }
  }

  return Changed;
}
