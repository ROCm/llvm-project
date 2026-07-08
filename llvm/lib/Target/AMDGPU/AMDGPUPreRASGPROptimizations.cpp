//===-- AMDGPUPreRASGPROptimizations.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split 64-bit SGPR register tuple initialization into
/// a single rematerializable pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This enables rematerialization instead of spilling. It must run after
/// register coalescing (so coalescer can handle the SGPR copies first) and
/// before SGPR allocation (so the allocator can rematerialize).
///
/// In the legacy pipeline, this optimization runs as part of
/// GCNPreRAOptimizations. In the LWT pipeline, it is split out here
/// because it must run after register coalescing (before SGPR alloc),
/// while vector register hints must run earlier (before VGPR alloc).
///
//===----------------------------------------------------------------------===//

#include "AMDGPUPreRASGPROptimizations.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-sgpr-optimizations"

namespace {

class AMDGPUPreRASGPROptimizationsImpl {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  bool processReg(Register Reg);

public:
  AMDGPUPreRASGPROptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class AMDGPUPreRASGPROptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreRASGPROptimizationsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA SGPR Optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUPreRASGPROptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA SGPR Optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPUPreRASGPROptimizationsLegacy, DEBUG_TYPE,
                    "AMDGPU Pre-RA SGPR Optimizations", false, false)

char AMDGPUPreRASGPROptimizationsLegacy::ID = 0;

char &llvm::AMDGPUPreRASGPROptimizationsID =
    AMDGPUPreRASGPROptimizationsLegacy::ID;

FunctionPass *llvm::createAMDGPUPreRASGPROptimizationsLegacyPass() {
  return new AMDGPUPreRASGPROptimizationsLegacy();
}

bool AMDGPUPreRASGPROptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return false;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool AMDGPUPreRASGPROptimizationsLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return AMDGPUPreRASGPROptimizationsImpl(LIS).run(MF);
}

PreservedAnalyses
AMDGPUPreRASGPROptimizationsPass::run(MachineFunction &MF,
                                      MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  AMDGPUPreRASGPROptimizationsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

bool AMDGPUPreRASGPROptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if (RC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC))
      continue;

    Changed |= processReg(Reg);
  }

  return Changed;
}
