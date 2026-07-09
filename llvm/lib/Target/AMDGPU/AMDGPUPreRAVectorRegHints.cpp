//===-- AMDGPUPreRAVectorRegHints.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass adds vector register allocation hints before VGPR allocation
/// in the LWT (Late Wave Transform) pipeline:
///
/// 1. True16 COPY hints: When using True16, COPY instructions moving 16-bit
///    values between VGPR_32 and VGPR_16 get allocation hints so the allocator
///    can place the VGPR_16 in the lo16 sub-register of the VGPR_32, allowing
///    the COPY to be eliminated entirely.
///
/// 2. AGPR copy propagation: For AGPR-to-AGPR copies that go through an
///    intermediate VGPR, this pass propagates the source of the defining
///    V_ACCVGPR_WRITE_B32_e64 directly to the COPY, eliminating the need
///    for the intermediate register.
///
/// 3. BVH stack optimization: Adds implicit uses to DS_BVH_STACK instructions
///    to avoid partial register re-allocation that could trigger premature
///    s_wait_bvhcnt instructions.
///
/// In the legacy pipeline, all these optimizations run as part of
/// GCNPreRAOptimizations. In the LWT pipeline, they are split out here
/// because they must run before VGPR allocation, while the SGPR constant
/// fusion runs later before SGPR allocation.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUPreRAVectorRegHints.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-vector-reg-hints"

namespace {

class AMDGPUPreRAVectorRegHintsImpl {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  bool processAGPRReg(Register Reg);
  void hintTrue16Copy(const MachineInstr &MI);
  bool optimizeBVHStack(MachineInstr &MI);

public:
  AMDGPUPreRAVectorRegHintsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class AMDGPUPreRAVectorRegHintsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreRAVectorRegHintsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA Vector Register Hints";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUPreRAVectorRegHintsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA Vector Register Hints", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPUPreRAVectorRegHintsLegacy, DEBUG_TYPE,
                    "AMDGPU Pre-RA Vector Register Hints", false, false)

char AMDGPUPreRAVectorRegHintsLegacy::ID = 0;

char &llvm::AMDGPUPreRAVectorRegHintsID = AMDGPUPreRAVectorRegHintsLegacy::ID;

FunctionPass *llvm::createAMDGPUPreRAVectorRegHintsLegacyPass() {
  return new AMDGPUPreRAVectorRegHintsLegacy();
}

bool AMDGPUPreRAVectorRegHintsImpl::processAGPRReg(Register Reg) {
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermediate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.
      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          const MachineOperand &DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    }
  }

  if (Changed) {
    for (Register RegToUpdate : ModifiedRegs) {
      LIS->removeInterval(RegToUpdate);
      LIS->createAndComputeVirtRegInterval(RegToUpdate);
    }
  }

  return Changed;
}

bool AMDGPUPreRAVectorRegHintsLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return AMDGPUPreRAVectorRegHintsImpl(LIS).run(MF);
}

PreservedAnalyses
AMDGPUPreRAVectorRegHintsPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  AMDGPUPreRAVectorRegHintsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

void AMDGPUPreRAVectorRegHintsImpl::hintTrue16Copy(const MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  const TargetRegisterClass *DstRC = TRI->getRegClassForReg(*MRI, Dst);
  bool IsDst16Bit = AMDGPU::VGPR_16RegClass.hasSubClassEq(DstRC);
  if (Dst.isVirtual() && IsDst16Bit && Src.isPhysical() &&
      TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
  if (Src.isVirtual() && MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
      Dst.isPhysical() && DstRC == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
  if (!Dst.isVirtual() || !Src.isVirtual())
    return;
  if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
      MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
    MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
  }
  if (IsDst16Bit && MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
}

bool AMDGPUPreRAVectorRegHintsImpl::optimizeBVHStack(MachineInstr &MI) {
  SmallVector<Register, 2> UseRegs;

  // Find BVH sources for this DS_BVH_STACK instruction.
  auto CheckUse = [&](MachineOperand &Use) {
    Register Reg = Use.getReg();
    for (const MachineInstr &Src : MRI->def_instructions(Reg)) {
      if (!SIInstrInfo::isImage(Src))
        continue;
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Src.getOpcode());
      const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
          AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
      if (!BaseInfo->BVH)
        continue;
      UseRegs.push_back(Reg);
      break;
    }
  };
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data0));
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data1));

  if (UseRegs.empty())
    return false;

  // Add implicit uses for entire BVH source registers.
  // This avoids partial reallocation of register which could
  // introduce a premature s_wait_bvhcnt.
  for (Register Reg : UseRegs) {
    MI.addOperand(MachineOperand::CreateReg(Reg, false, true));
    LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
  }
  LLVM_DEBUG(dbgs() << "Added implicit uses to: " << MI);

  return true;
}

bool AMDGPUPreRAVectorRegHintsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;

  // Process AGPR copy propagation.
  if (ST.hasGFX90AInsts()) {
    for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (!LIS->hasInterval(Reg))
        continue;
      const TargetRegisterClass *RC = MRI->getRegClass(Reg);
      if (!TRI->isAGPRClass(RC))
        continue;

      Changed |= processAGPRReg(Reg);
    }
  }

  const bool HasBVHStack = ST.hasBVHDualAndBVH8Insts();
  const bool HasRealTrue16 = ST.useRealTrue16Insts();

  if (!HasRealTrue16 && !HasBVHStack)
    return Changed;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Add RA hints to improve True16 COPY elimination.
      if (HasRealTrue16 && MI.getOpcode() == AMDGPU::COPY) {
        hintTrue16Copy(MI);
        continue;
      }
      // Add implicit uses to avoid early wait on intersect ray instructions.
      if (HasBVHStack &&
          (MI.getOpcode() == AMDGPU::DS_BVH_STACK_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP1_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP2_RTN_B64)) {
        Changed |= optimizeBVHStack(MI);
        continue;
      }
    }
  }

  return Changed;
}
