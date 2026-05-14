//===-- AMDGPUReserveAllocatedVGPRs.cpp - Reserve perlane VGPRs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// In the late-WaveTransform (LWT) pipeline, perlane VGPR allocation runs
/// before the SGPR-spill and WWM-regalloc rounds. The physical VGPRs picked by
/// that early perlane allocation must be preserved so that subsequent passes
/// (SILowerSGPRSpills, WWM regalloc, PEI / scavenger, ...) do not reuse them.
///
/// This pass should be invoked at the end of the perlane VGPR allocation in
/// the LWT flow. It records the perlane-allocated physical VGPRs into
/// SIMachineFunctionInfo's VGPRAllocMask, so that later phases see them as
/// reserved through SIRegisterInfo::getReservedRegs().
//
//===----------------------------------------------------------------------===//

#include "AMDGPUReserveAllocatedVGPRs.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <charconv>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-reserve-allocated-vgprs"

namespace {

class AMDGPUReserveAllocatedVGPRsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUReserveAllocatedVGPRsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Reserve Allocated VGPRs";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class AMDGPUReserveAllocatedVGPRs {
public:
  bool run(MachineFunction &MF);
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUReserveAllocatedVGPRsLegacy, DEBUG_TYPE,
                "AMDGPU Reserve Allocated VGPRs", false, false)

char AMDGPUReserveAllocatedVGPRsLegacy::ID = 0;

char &llvm::AMDGPUReserveAllocatedVGPRsLegacyID =
    AMDGPUReserveAllocatedVGPRsLegacy::ID;

bool AMDGPUReserveAllocatedVGPRsLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return AMDGPUReserveAllocatedVGPRs().run(MF);
}

PreservedAnalyses
AMDGPUReserveAllocatedVGPRsPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &) {
  AMDGPUReserveAllocatedVGPRs().run(MF);
  return PreservedAnalyses::all();
}

bool AMDGPUReserveAllocatedVGPRs::run(MachineFunction &MF) {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  BitVector RegMask(TRI->getNumRegs());
  auto [MaxNumVGPRs, MaxNumAGPRs] = ST.getMaxNumVectorRegs(MF.getFunction());
  for (unsigned RegI = AMDGPU::VGPR0, RegE = AMDGPU::VGPR0 + MaxNumVGPRs;
       RegI < RegE; ++RegI) {
    if (!MRI.isPhysRegUsed(RegI, /*SkipRegMaskTest=*/true))
      continue;
    TRI->markSuperRegs(RegMask, RegI);
  }
  RegMask.clearBitsNotInMask(TRI->getAllVGPRRegMask());
  MFI->updateVGPRAllocMask(RegMask);

  // The renamable flag can't be set for reserved registers. Reset the flag on
  // machine operands referencing the perlane-allocated VGPRs (and their tuple
  // super-registers) as they are marked as reserved during the subsequent
  // SGPR/WWM regalloc rounds.
  for (unsigned Reg : RegMask.set_bits()) {
    for (MachineOperand &MO : MRI.reg_operands(Reg))
      MO.setIsRenamable(false);
  }

  // As we are calling this pass just before a non-RA pass, we need to manually
  // freeze the reserved registers to update the VGPRAllocMask.
  MF.getRegInfo().freezeReservedRegs();

  return RegMask.any();
}
