//===------- AMDGPUPreWaveTransform.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass prepares the machine function for wave transform.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-wave-transform"

namespace {

class AMDGPUPreWaveTransform : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPUPreWaveTransform() : MachineFunctionPass(ID) {
    initializeAMDGPUPreWaveTransformPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre Wave Transform";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPreWaveTransform, DEBUG_TYPE,
                "AMDGPU Pre Wave Transform", false, false)

char AMDGPUPreWaveTransform::ID = 0;
char &llvm::AMDGPUPreWaveTransformID = AMDGPUPreWaveTransform::ID;

FunctionPass *llvm::createAMDGPUPreWaveTransformPass() {
  return new AMDGPUPreWaveTransform();
}

// Drop SI_BRCOND/SI_BRCOND_Z when both arms target the same block.
static bool dropRedundantBrCond(MachineBasicBlock &MBB,
                                MachineRegisterInfo &MRI) {
  MachineInstr *BrCond = nullptr;
  for (MachineInstr &MI : MBB.terminators()) {
    unsigned Op = MI.getOpcode();
    if (Op == AMDGPU::SI_BRCOND || Op == AMDGPU::SI_BRCOND_Z) {
      BrCond = &MI;
      break;
    }
  }
  if (!BrCond)
    return false;

  MachineBasicBlock *CondTarget = BrCond->getOperand(0).getMBB();
  MachineBasicBlock *OtherTarget = nullptr;
  MachineBasicBlock::iterator It(BrCond);
  for (++It; It != MBB.end(); ++It) {
    if (It->getOpcode() == AMDGPU::S_BRANCH) {
      OtherTarget = It->getOperand(0).getMBB();
      break;
    }
  }
  if (!OtherTarget)
    OtherTarget = MBB.getFallThrough();

  if (CondTarget != OtherTarget)
    return false;

  Register Cond = BrCond->getOperand(1).getReg();
  BrCond->eraseFromParent();

  if (Cond.isVirtual() && MRI.use_nodbg_empty(Cond)) {
    MachineInstr *Def = MRI.getUniqueVRegDef(Cond);
    if (Def && !Def->mayLoad() && !Def->mayStore() &&
        !Def->hasUnmodeledSideEffects())
      Def->eraseFromParent();
  }

  return true;
}

bool AMDGPUPreWaveTransform::runOnMachineFunction(MachineFunction &MF) {
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.terminators()) {
      if (MI.getOpcode() == AMDGPU::SI_WATERFALL_LOOP) {
        MI.setDesc(TII->get(AMDGPU::S_CBRANCH_EXECNZ));
        Changed = true;
      }
    }
    Changed |= dropRedundantBrCond(MBB, MRI);
  }

  return Changed;
}
