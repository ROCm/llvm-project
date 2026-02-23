//===-- AMDGPUReEmitLiveDebugVariables.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The Late Wave Transform (LWT) pass transforms the machine CFG from
/// thread-level to wave-level control flow. This invalidates SlotIndexes,
/// LiveIntervals, and MBB pointers that LiveDebugVariables (LDV) uses
/// internally to track debug variable locations.
///
/// This pass re-emits the debug data collected by LDV back into MIR at an
/// explicit pipeline point — after VGPR register allocation but before LWT
/// modifies the CFG. VirtRegMap is used to resolve any virtual registers that
/// were assigned physical registers by the preceding allocation stage.
///
/// After LWT restructures the CFG, a fresh LDV instance re-collects the
/// debug instructions from the transformed MIR for the subsequent SGPR/WWM/VGPR
/// allocation stages.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUReEmitLiveDebugVariables.h"
#include "AMDGPU.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/Function.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-re-emit-ldv"

namespace {

class AMDGPUReEmitLiveDebugVariablesLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUReEmitLiveDebugVariablesLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUReEmitLiveDebugVariablesLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Re-emit LiveDebugVariables Before Wave Transform";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveDebugVariablesWrapperLegacy>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUReEmitLiveDebugVariablesLegacy, DEBUG_TYPE,
                "AMDGPU Re-emit LiveDebugVariables Before Wave Transform",
                false, false)

char AMDGPUReEmitLiveDebugVariablesLegacy::ID = 0;

char &llvm::AMDGPUReEmitLiveDebugVariablesLegacyID =
    AMDGPUReEmitLiveDebugVariablesLegacy::ID;

bool AMDGPUReEmitLiveDebugVariablesLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (!MF.getFunction().getSubprogram())
    return false;
  auto &LDV = getAnalysis<LiveDebugVariablesWrapperLegacy>().getLDV();
  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LDV.revertCollection(&VRM);
  return true;
}

PreservedAnalyses AMDGPUReEmitLiveDebugVariablesPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  if (!MF.getFunction().getSubprogram())
    return PreservedAnalyses::all();

  auto &LDV = MFAM.getResult<LiveDebugVariablesAnalysis>(MF);
  auto &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  LDV.revertCollection(&VRM);
  return PreservedAnalyses::all();
}
