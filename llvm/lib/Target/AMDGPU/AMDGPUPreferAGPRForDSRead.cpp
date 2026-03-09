//===-- AMDGPUPreferAGPRForDSRead.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Forces DS load destinations marked with the NonTemporal flag to be allocated
// into AccVGPR (AGPR) physical registers, then propagates this constraint
// through COPY and PHI chains so that loop-carried values (scf.for iter_args)
// also stay in AGPR across loop boundaries.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPreferAGPRForDSRead.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-prefer-agpr-for-ds-read"

static cl::opt<bool> EnablePreferAGPRForDSRead(
    "amdgpu-prefer-agpr-for-ds-read",
    cl::desc("Set AGPR allocation hints on ALL DS read destinations"),
    cl::init(false), cl::Hidden);

namespace {

class AMDGPUPreferAGPRForDSReadImpl {
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;

  bool tryConstrainToAGPR(Register VirtReg, DenseSet<Register> &Constrained);

public:
  AMDGPUPreferAGPRForDSReadImpl(const GCNSubtarget &ST,
                                MachineRegisterInfo &MRI)
      : TII(*ST.getInstrInfo()),
        TRI(*static_cast<const SIRegisterInfo *>(ST.getRegisterInfo())),
        MRI(MRI) {}

  bool run(MachineFunction &MF);
};

class AMDGPUPreferAGPRForDSReadLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreferAGPRForDSReadLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUPreferAGPRForDSReadLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Prefer AGPR For DS Read";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUPreferAGPRForDSReadLegacy, DEBUG_TYPE,
                      "AMDGPU Prefer AGPR For DS Read", false, false)
INITIALIZE_PASS_END(AMDGPUPreferAGPRForDSReadLegacy, DEBUG_TYPE,
                    "AMDGPU Prefer AGPR For DS Read", false, false)

char AMDGPUPreferAGPRForDSReadLegacy::ID = 0;

char &llvm::AMDGPUPreferAGPRForDSReadLegacyID =
    AMDGPUPreferAGPRForDSReadLegacy::ID;

bool AMDGPUPreferAGPRForDSReadLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  return AMDGPUPreferAGPRForDSReadImpl(ST, MF.getRegInfo()).run(MF);
}

PreservedAnalyses
AMDGPUPreferAGPRForDSReadPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  AMDGPUPreferAGPRForDSReadImpl(ST, MF.getRegInfo()).run(MF);
  return PreservedAnalyses::all();
}

bool AMDGPUPreferAGPRForDSReadImpl::tryConstrainToAGPR(
    Register VirtReg, DenseSet<Register> &Constrained) {
  if (Constrained.count(VirtReg))
    return false;

  const TargetRegisterClass *CurRC = MRI.getRegClass(VirtReg);
  if (SIRegisterInfo::isAGPRClass(CurRC)) {
    Constrained.insert(VirtReg);
    return false;
  }

  const TargetRegisterClass *ARC = TRI.getEquivalentAGPRClass(CurRC);
  if (!ARC)
    return false;

  const TargetRegisterClass *NewRC = MRI.constrainRegClass(VirtReg, ARC);
  if (!NewRC)
    return false;

  MRI.setRegAllocationHint(VirtReg, AMDGPURI::PreferAGPR, Register());
  Constrained.insert(VirtReg);
  return true;
}

bool AMDGPUPreferAGPRForDSReadImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (!ST.hasGFX90AInsts())
    return false;

  DenseSet<Register> AGPRConstrained;
  bool Changed = false;

  // Phase 1: Constrain nontemporal DS load destinations to AGPR.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!TII.isDS(MI) || !MI.mayLoad())
        continue;

      bool MarkedNonTemporal = llvm::any_of(MI.memoperands(),
          [](const MachineMemOperand *MMO) { return MMO->isNonTemporal(); });

      if (!MarkedNonTemporal && !EnablePreferAGPRForDSRead)
        continue;

      for (MachineOperand &MO : MI.defs()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;

        if (tryConstrainToAGPR(MO.getReg(), AGPRConstrained))
          Changed = true;
        break;
      }
    }
  }

  if (AGPRConstrained.empty())
    return Changed;

  // Phase 2: Propagate AGPR constraints through COPY and PHI chains.
  // When a DS load result flows into a PHI (e.g., scf.for iter_args),
  // the PHI destination and ALL PHI sources must share a compatible
  // register class.  We constrain them all to AGPR so the register
  // coalescer doesn't widen the class back to AV_*.
  bool Propagated = true;
  while (Propagated) {
    Propagated = false;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!MI.isPHI() && !MI.isCopy())
          continue;

        Register DestReg = MI.getOperand(0).getReg();
        if (!DestReg.isVirtual())
          continue;

        // Collect source virtual registers.
        SmallVector<Register, 8> SrcRegs;
        if (MI.isCopy()) {
          Register Src = MI.getOperand(1).getReg();
          if (Src.isVirtual())
            SrcRegs.push_back(Src);
        } else {
          // PHI: operands are (dest, src1, bb1, src2, bb2, ...)
          for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
            Register Src = MI.getOperand(i).getReg();
            if (Src.isVirtual())
              SrcRegs.push_back(Src);
          }
        }

        // Check if dest or any source is already AGPR-constrained.
        bool AnyAGPR = AGPRConstrained.count(DestReg);
        for (Register Src : SrcRegs) {
          if (AGPRConstrained.count(Src)) {
            AnyAGPR = true;
            break;
          }
        }

        if (!AnyAGPR)
          continue;

        // Forward propagation: constrain dest.
        if (tryConstrainToAGPR(DestReg, AGPRConstrained)) {
          Propagated = true;
          Changed = true;
        }

        // Backward propagation: constrain all sources so the PHI/COPY
        // coalescing sees a uniform AGPR class and doesn't widen.
        for (Register Src : SrcRegs) {
          if (tryConstrainToAGPR(Src, AGPRConstrained)) {
            Propagated = true;
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}
