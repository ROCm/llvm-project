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
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-wave-transform"

static bool preWaveTransform(MachineFunction &MF) {
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.terminators()) {
      if (MI.getOpcode() == AMDGPU::SI_WATERFALL_LOOP) {
        MI.setDesc(TII->get(AMDGPU::S_CBRANCH_EXECNZ));
        Changed = true;
      }
    }
  }

  return Changed;
}

namespace {

class AMDGPUPreWaveTransformLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreWaveTransformLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUPreWaveTransformLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    return preWaveTransform(MF);
  }

  StringRef getPassName() const override {
    return "AMDGPU Pre Wave Transform";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPreWaveTransformLegacy, DEBUG_TYPE,
                "AMDGPU Pre Wave Transform", false, false)

char AMDGPUPreWaveTransformLegacy::ID = 0;
char &llvm::AMDGPUPreWaveTransformID = AMDGPUPreWaveTransformLegacy::ID;

FunctionPass *llvm::createAMDGPUPreWaveTransformPass() {
  return new AMDGPUPreWaveTransformLegacy();
}

PreservedAnalyses
llvm::AMDGPUPreWaveTransformPass::run(MachineFunction &MF,
                                      MachineFunctionAnalysisManager &MFAM) {
  if (!preWaveTransform(MF))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
