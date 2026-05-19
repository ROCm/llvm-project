//===-- AMDGPUSetBankHints.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Process V_SET_BANK_B* pseudo instructions: read the bank operand, set a
// register allocation hint on the destination vreg, and replace the pseudo
// with a COPY.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-set-bank-hints"

/// Walk the SSA use chain of \p Reg and propagate bank hints through
/// COPY, REG_SEQUENCE, INSERT_SUBREG, and SUBREG_TO_REG so that the
/// bank preference survives register coalescing.
static void propagateBankHint(MachineRegisterInfo &MRI, Register Reg,
                              unsigned Bank,
                              SmallDenseSet<unsigned, 32> &Visited) {
  if (!Reg.isVirtual() || !Visited.insert(Reg.id()).second)
    return;

  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    Register DefReg;
    switch (UseMI.getOpcode()) {
    case TargetOpcode::COPY:
    case TargetOpcode::REG_SEQUENCE:
    case TargetOpcode::INSERT_SUBREG:
    case TargetOpcode::SUBREG_TO_REG:
      DefReg = UseMI.getOperand(0).getReg();
      break;
    default:
      continue;
    }
    if (!DefReg.isVirtual())
      continue;

    const auto &Existing = MRI.getRegAllocationHint(DefReg);
    if (Existing.first != 0 &&
        (Existing.first != AMDGPURI::BankHint || Existing.second != Bank))
      continue;
    MRI.setRegAllocationHint(DefReg, AMDGPURI::BankHint, Bank);
    propagateBankHint(MRI, DefReg, Bank, Visited);
  }
}

namespace {

class AMDGPUSetBankHints : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUSetBankHints() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU Set Bank Hints"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUSetBankHints, DEBUG_TYPE,
                "AMDGPU Set VGPR Bank Hints", false, false)

char AMDGPUSetBankHints::ID = 0;

char &llvm::AMDGPUSetBankHintsID = AMDGPUSetBankHints::ID;

static bool isSetBankPseudo(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::V_SET_BANK_B32:
  case AMDGPU::V_SET_BANK_B64:
  case AMDGPU::V_SET_BANK_B128:
  case AMDGPU::V_SET_BANK_B256:
    return true;
  default:
    return false;
  }
}

bool AMDGPUSetBankHints::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  SmallVector<MachineInstr *, 16> ToErase;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSetBankPseudo(MI.getOpcode()))
        continue;

      Register DstReg = MI.getOperand(0).getReg();
      Register SrcReg = MI.getOperand(1).getReg();
      unsigned Bank = MI.getOperand(2).getImm();

      if (Bank > 3)
        Bank = 0;

      // Set bank hint on both destination and source vregs so the hint
      // survives COPY coalescing regardless of which register is eliminated.
      if (DstReg.isVirtual())
        MRI.setRegAllocationHint(DstReg, AMDGPURI::BankHint, Bank);
      if (SrcReg.isVirtual())
        MRI.setRegAllocationHint(SrcReg, AMDGPURI::BankHint, Bank);

      // Propagate hints through REG_SEQUENCE / COPY / INSERT_SUBREG so
      // the bank preference reaches larger registers that survive coalescing.
      SmallDenseSet<unsigned, 32> Visited;
      if (DstReg.isVirtual())
        propagateBankHint(MRI, DstReg, Bank, Visited);

      // Replace pseudo with COPY.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), DstReg)
          .addReg(SrcReg, getRegState(MI.getOperand(1)));

      ToErase.push_back(&MI);
      Changed = true;
    }
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();

  return Changed;
}
