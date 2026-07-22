//===-- AMDGPUPartitionVGPRsForRA.cpp - Partition VGPRs before RA ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass is responsible for partitioning the available VGPRs into two sets
/// such that the perlane-VGPR and wwm-VGPR regalloc pipelines can use distinct
/// registers for allocation. This is truly needed as their allocations are
/// attempted separately due to the incorrect liveness computation of the two
/// separate value classes - one being the perlane and the other being the
/// whole-wave in nature. Until we fix their liveness and further enhance the
/// interference computation during various RA stages, we still have to use this
/// suboptimal partitioning to ensure enough registers for their allocations.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPartitionVGPRsForRA.h"
#include "AMDGPU.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-partition-vgprs-for-ra"

// Minimum number of VGPRs reserved for WWM register allocation. This is a
// conservative estimate of the VGPRs needed for SGPR spill lanes and virtual
// WWM registers, assuming the worst case when perlane allocation consumes all
// remaining VGPRs.
static cl::opt<unsigned> MinNumWWMRegs(
    "amdgpu-min-vgprs-for-wwm-regalloc", cl::init(3), cl::Hidden,
    cl::desc("Minimum VGPRs reserved for WWM register allocation"));

namespace {

class AMDGPUPartitionVGPRsForRALegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPartitionVGPRsForRALegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Partition VGPRs for RA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class AMDGPUPartitionVGPRsForRA {
public:
  bool run(MachineFunction &MF);
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPartitionVGPRsForRALegacy, DEBUG_TYPE,
                "AMDGPU Partition VGPRs for RA", false, false)

char AMDGPUPartitionVGPRsForRALegacy::ID = 0;

char &llvm::AMDGPUPartitionVGPRsForRALegacyID =
    AMDGPUPartitionVGPRsForRALegacy::ID;

bool AMDGPUPartitionVGPRsForRALegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return AMDGPUPartitionVGPRsForRA().run(MF);
}

PreservedAnalyses
AMDGPUPartitionVGPRsForRAPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &) {
  AMDGPUPartitionVGPRsForRA().run(MF);
  return PreservedAnalyses::all();
}

bool AMDGPUPartitionVGPRsForRA::run(MachineFunction &MF) {
  SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  BitVector WwmRegMask(TRI->getNumRegs());
  TRI->determineVGPRsForWwmAlloc(MF, WwmRegMask, MinNumWWMRegs);
  MFI.updateVGPRAllocMask(WwmRegMask);

  return true;
}
