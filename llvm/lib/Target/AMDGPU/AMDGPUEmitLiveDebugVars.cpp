//===--------------------- AMDGPUEmitLiveDebugVars.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// In the LWT (Late Wave Transform) pipeline, register allocation runs in
/// multiple phases: VGPR alloc, then WaveTransform, then SGPR/WWM alloc.
/// LiveDebugVariables (LDV) collects DBG_VALUE instructions during VGPR alloc
/// and must emit them back into MIR before WaveTransform invalidates LDV.
/// Without this intermediate emission, LDV's releaseMemory() would trigger an
/// assertion because ModifiedMF=true but EmitDone=false.
///
/// This pass calls emitDebugValues with KeepUnassignedVRegs=true, which:
/// - Rewrites allocated VGPR locations to physical registers
/// - Keeps unallocated SGPR vregs as-is for re-collection in the next phase
/// - Sets EmitDone=true so the assertion in clear() passes
///
/// By not preserving LDV, this pass intentionally invalidates it so the next
/// RA phase re-collects from post-WaveTransform MIR.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-emit-live-debug-vars"

namespace {

class AMDGPUEmitLiveDebugVars {
public:
  bool run(MachineFunction &MF, LiveDebugVariables &LDV, VirtRegMap &VRM);
};

class AMDGPUEmitLiveDebugVarsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUEmitLiveDebugVarsLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUEmitLiveDebugVarsLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.getFunction().getSubprogram())
      return false;
    auto &LDV = getAnalysis<LiveDebugVariablesWrapperLegacy>().getLDV();
    auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    return AMDGPUEmitLiveDebugVars().run(MF, LDV, VRM);
  }

  StringRef getPassName() const override {
    return "AMDGPU Emit Live Debug Variables";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveDebugVariablesWrapperLegacy>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    // Intentionally do NOT preserve LDV — it must be invalidated so that
    // the next RA phase re-collects from post-WaveTransform MIR.
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUEmitLiveDebugVarsLegacy, DEBUG_TYPE,
                      "AMDGPU Emit Live Debug Variables", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariablesWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_END(AMDGPUEmitLiveDebugVarsLegacy, DEBUG_TYPE,
                    "AMDGPU Emit Live Debug Variables", false, false)

char AMDGPUEmitLiveDebugVarsLegacy::ID = 0;
char &llvm::AMDGPUEmitLiveDebugVarsLegacyID = AMDGPUEmitLiveDebugVarsLegacy::ID;

FunctionPass *llvm::createAMDGPUEmitLiveDebugVarsPass() {
  return new AMDGPUEmitLiveDebugVarsLegacy();
}

bool AMDGPUEmitLiveDebugVars::run(MachineFunction &MF, LiveDebugVariables &LDV,
                                  VirtRegMap &VRM) {
  LDV.emitDebugValues(&VRM, /*KeepUnassignedVRegs=*/true);
  return true;
}

PreservedAnalyses
llvm::AMDGPUEmitLiveDebugVarsPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  if (!MF.getFunction().getSubprogram())
    return PreservedAnalyses::all();
  auto &LDV = MFAM.getResult<LiveDebugVariablesAnalysis>(MF);
  auto &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  AMDGPUEmitLiveDebugVars().run(MF, LDV, VRM);
  // Intentionally do NOT preserve LDV.
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
