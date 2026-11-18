//===-- AMDGPUSchedBank.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower the V_SCHEDULE_BANK_B* pseudos produced from llvm.amdgcn.sched.bank,
/// and the V_SCHEDULE_VGPR_B*, S_SCHEDULE_SGPR_B*, A_SCHEDULE_AGPR_B* pseudos
/// from llvm.amdgcn.schedule.{vgpr,sgpr,agpr}.
///
/// For each pseudo this pass:
///   1. reads the requested bank or register number from the immediate operand,
///   2. attaches an appropriate register allocation hint to the destination
///      (and source) vreg so the allocator prefers the target bank or register,
///   3. propagates the hint through COPY / REG_SEQUENCE / INSERT_SUBREG /
///      SUBREG_TO_REG so it survives coalescing, and
///   4. replaces the pseudo with a plain COPY.
///
/// The hint is advisory: SIRegisterInfo::getRegAllocationHints only reorders the
/// allocation candidates, so a full bank (or unavailable register) falls back to
/// the default order and the pass never forces a spill.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSchedBank.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-sched-bank"

/// Classification of schedule-hint pseudos.
enum ScheduleHintKind { SHK_None, SHK_Bank, SHK_VGPR, SHK_SGPR, SHK_AGPR };

static ScheduleHintKind getScheduleHintKind(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::V_SCHEDULE_BANK_B32:
  case AMDGPU::V_SCHEDULE_BANK_B64:
  case AMDGPU::V_SCHEDULE_BANK_B128:
  case AMDGPU::V_SCHEDULE_BANK_B256:
    return SHK_Bank;
  case AMDGPU::V_SCHEDULE_VGPR_B32:
  case AMDGPU::V_SCHEDULE_VGPR_B64:
  case AMDGPU::V_SCHEDULE_VGPR_B128:
  case AMDGPU::V_SCHEDULE_VGPR_B256:
  case AMDGPU::V_SCHEDULE_VGPR_B512:
    return SHK_VGPR;
  case AMDGPU::S_SCHEDULE_SGPR_B32:
  case AMDGPU::S_SCHEDULE_SGPR_B64:
  case AMDGPU::S_SCHEDULE_SGPR_B128:
  case AMDGPU::S_SCHEDULE_SGPR_B256:
  case AMDGPU::S_SCHEDULE_SGPR_B512:
    return SHK_SGPR;
  case AMDGPU::A_SCHEDULE_AGPR_B32:
  case AMDGPU::A_SCHEDULE_AGPR_B64:
  case AMDGPU::A_SCHEDULE_AGPR_B128:
  case AMDGPU::A_SCHEDULE_AGPR_B256:
  case AMDGPU::A_SCHEDULE_AGPR_B512:
    return SHK_AGPR;
  default:
    return SHK_None;
  }
}

/// Walk the SSA use chain of \p Reg and propagate the hint through
/// value-preserving pseudos so the preference reaches the vregs that survive
/// coalescing.
static void propagateHint(MachineRegisterInfo &MRI, Register Reg,
                          unsigned HintValue, unsigned HintKind,
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

    // Do not clobber an existing, conflicting hint.
    std::pair<unsigned, Register> Existing = MRI.getRegAllocationHint(DefReg);
    if (Existing.first != 0 &&
        (Existing.first != HintKind || Existing.second != HintValue))
      continue;

    MRI.setRegAllocationHint(DefReg, HintKind, HintValue);
    propagateHint(MRI, DefReg, HintValue, HintKind, Visited);
  }
}

// Shared implementation used by both the legacy and new-PM passes.
static bool runScheduleBank(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  SmallVector<MachineInstr *, 16> ToErase;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      ScheduleHintKind Kind = getScheduleHintKind(MI.getOpcode());
      if (Kind == SHK_None)
        continue;

      Register DstReg = MI.getOperand(0).getReg();
      const MachineOperand &SrcMO = MI.getOperand(1);
      Register SrcReg = SrcMO.getReg();

      unsigned HintKind;
      unsigned HintValue;

      if (Kind == SHK_Bank) {
        // Bank pseudos are only valid on targets with 1024 addressable VGPRs.
        if (!ST.has1024AddressableVGPRs()) {
          // Replace with COPY and skip hint setup.
          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY),
                  DstReg)
              .addReg(SrcReg, getRegState(SrcMO), SrcMO.getSubReg());
          ToErase.push_back(&MI);
          Changed = true;
          continue;
        }

        unsigned EncodedBank = MI.getOperand(2).getImm();
        bool Strict = EncodedBank & 0x4;
        unsigned Bank = EncodedBank & 0x3;

        if (EncodedBank > 7) {
          Bank = 0;
          Strict = false;
        }

        HintKind = Strict ? AMDGPURI::StrictBankHint : AMDGPURI::BankHint;
        HintValue = Bank;

        LLVM_DEBUG(dbgs() << "  sched.bank: " << printReg(DstReg) << " <- "
                          << printReg(SrcReg) << " bank " << Bank
                          << " strict " << Strict << '\n');
      } else {
        // Exact register pseudos: operand 2 = regnum, operand 3 = strict.
        unsigned RegNum = MI.getOperand(2).getImm();
        bool Strict = MI.getOperand(3).getImm() != 0;

        switch (Kind) {
        case SHK_VGPR:
          HintKind = Strict ? AMDGPURI::StrictVGPRRegHint
                            : AMDGPURI::VGPRRegHint;
          break;
        case SHK_SGPR:
          HintKind = Strict ? AMDGPURI::StrictSGPRRegHint
                            : AMDGPURI::SGPRRegHint;
          break;
        case SHK_AGPR:
          HintKind = Strict ? AMDGPURI::StrictAGPRRegHint
                            : AMDGPURI::AGPRRegHint;
          break;
        default:
          llvm_unreachable("unexpected schedule hint kind");
        }
        HintValue = RegNum;

        LLVM_DEBUG(dbgs() << "  schedule."
                          << (Kind == SHK_VGPR   ? "vgpr"
                              : Kind == SHK_SGPR ? "sgpr"
                                                 : "agpr")
                          << ": " << printReg(DstReg) << " <- "
                          << printReg(SrcReg) << " reg " << RegNum
                          << " strict " << Strict << '\n');
      }

      // Hint both the destination and the source so the preference survives
      // whichever side coalescing keeps. Don't overwrite a strict hint with a
      // soft hint of the same category.
      bool IsStrict = (HintKind == AMDGPURI::StrictBankHint ||
                       HintKind == AMDGPURI::StrictVGPRRegHint ||
                       HintKind == AMDGPURI::StrictSGPRRegHint ||
                       HintKind == AMDGPURI::StrictAGPRRegHint);
      // The strict variant is always HintKind | 1 for each category.
      unsigned StrictVariant = HintKind | 1;

      if (DstReg.isVirtual() &&
          (IsStrict ||
           MRI.getRegAllocationHint(DstReg).first != StrictVariant))
        MRI.setRegAllocationHint(DstReg, HintKind, HintValue);
      if (SrcReg.isVirtual() &&
          (IsStrict ||
           MRI.getRegAllocationHint(SrcReg).first != StrictVariant))
        MRI.setRegAllocationHint(SrcReg, HintKind, HintValue);

      SmallDenseSet<unsigned, 32> Visited;
      if (DstReg.isVirtual())
        propagateHint(MRI, DstReg, HintValue, HintKind, Visited);

      // Replace the pseudo with a plain COPY.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), DstReg)
          .addReg(SrcReg, getRegState(SrcMO), SrcMO.getSubReg());

      ToErase.push_back(&MI);
      Changed = true;
    }
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();

  return Changed;
}

namespace {

class AMDGPUSchedBank : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUSchedBank() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return runScheduleBank(MF);
  }

  StringRef getPassName() const override { return "AMDGPU Sched Bank"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUSchedBank, DEBUG_TYPE, "AMDGPU Sched Bank",
                false, false)

char AMDGPUSchedBank::ID = 0;

char &llvm::AMDGPUSchedBankID = AMDGPUSchedBank::ID;

PreservedAnalyses
AMDGPUSchedBankPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  if (!runScheduleBank(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
