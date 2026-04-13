//===------- AMDGPUFinalizeISelWaveTransform.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass runs at the end of DAG-ISel in late wave-transform mode.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SILowerI1Copies.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-finalize-isel-wave-transform"

namespace {

class Vreg1WideningHelper : public PhiLoweringHelper {
public:
  Vreg1WideningHelper(MachineFunction *MF);

private:
  DenseSet<Register> ConstrainRegs;
  void buildLaneMaskToVGPR(MachineBasicBlock &MBB, MachineInstr &InsertBefore,
                           const DebugLoc &DL, Register DstReg,
                           Register SrcReg) {
    ConstrainRegs.insert(SrcReg);
    BuildMI(MBB, InsertBefore, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
        .addImm(0)
        .addImm(0)
        .addImm(0)
        .addImm(-1)
        .addReg(SrcReg);
  }
  void buildVGPRToLaneMask(MachineBasicBlock &MBB, MachineInstr &InsertBefore,
                           const DebugLoc &DL, Register DstReg,
                           Register SrcReg) {
    BuildMI(MBB, InsertBefore, DL, TII->get(AMDGPU::V_CMP_NE_U32_e64), DstReg)
        .addReg(SrcReg)
        .addImm(0);
  }

public:
  void markAsLaneMask(Register DstReg) const override {
    MRI->setRegClass(DstReg, ST->getBoolRC());
  }
  void getCandidatesForLowering(
      SmallVectorImpl<MachineInstr *> &Vreg1Phis) const override {}
  void collectIncomingValuesFromPhi(
      const MachineInstr *MI,
      SmallVectorImpl<Incoming> &Incomings) const override {}
  void replaceDstReg(Register NewReg, Register OldReg,
                     MachineBasicBlock *MBB) override {}
  void buildMergeLaneMasks(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, const DebugLoc &DL,
                           Register DstReg, Register PrevReg,
                           Register CurReg) override {}
  void constrainAsLaneMask(Incoming &In) override {}

  bool widenVreg1s();

  bool cleanConstrainRegs(bool Changed) {
    assert(Changed || ConstrainRegs.empty());
    for (Register Reg : ConstrainRegs)
      MRI->constrainRegClass(Reg, TII->getRegisterInfo().getWaveMaskRegClass());
    ConstrainRegs.clear();
    return Changed;
  }
  bool isVreg1(Register Reg) const {
    return Reg.isVirtual() && MRI->getRegClass(Reg) == &AMDGPU::VReg_1RegClass;
  }
  bool isVreg32(Register Reg) const {
    return Reg.isVirtual() && MRI->getRegClass(Reg) == &AMDGPU::VGPR_32RegClass;
  }
};

Vreg1WideningHelper::Vreg1WideningHelper(MachineFunction *MF)
    : PhiLoweringHelper(MF, nullptr, nullptr) {}

// When WaveTransform happens later and CFG is not structurized,
// We need to apply a different algorithm for lowering vreg_1
// PhiNodes. Plus maybe some other lowering work needed?
class AMDGPUFinalizeISelWaveTransform : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPUFinalizeISelWaveTransform() : MachineFunctionPass(ID) {
    initializeAMDGPUFinalizeISelWaveTransformPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Finalize ISel for Wave Transform";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUFinalizeISelWaveTransform, DEBUG_TYPE,
                      "AMDGPU Finalize ISel Wave Transform", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUFinalizeISelWaveTransform, DEBUG_TYPE,
                    "AMDGPU Finalize ISel Wave Transform", false, false)

char AMDGPUFinalizeISelWaveTransform::ID = 0;
char &llvm::AMDGPUFinalizeISelWaveTransformID =
    AMDGPUFinalizeISelWaveTransform::ID;

FunctionPass *llvm::createAMDGPUFinalizeISelWaveTransformPass() {
  return new AMDGPUFinalizeISelWaveTransform();
}

//===----------------------------------------------------------------------===//
// MIR-level PHI simplification for uniform (SGPR) PHIs, analogous to
// simplifyPHINode() in InstructionSimplify.cpp for LLVM IR.
//
// Detect PHIs that even though UA identifies them as uniform, they actually
// merge values through divergent control flow.
//
// The only possible scenario for such PHIs is when all incoming edges carry
// either undef/poison (IMPLICIT_DEF) or the same uniquely defined value.
// This case is only observable in -O0 mode for the late wave-transform
// pipeline (for other modes, codegen would already have optimized such PHIs
// before ISel).  As for the early structurizer-enabled pipeline, the structurizer pass would
// optimize such PHIs after restructurizing the CFG, irrespective of the
// optimization level.
//
// Only SGPR (scalar) PHIs are candidates: every source and destination
// register must reside in an SGPR register class.
//===----------------------------------------------------------------------===//

/// Return the single common register that a uniform machine PHI folds to, or
/// an invalid Register if the PHI cannot be simplified.  Only PHIs whose
/// destination and all source registers are SGPRs are considered.
static Register simplifyMachinePHI(MachineInstr &PHI, MachineRegisterInfo &MRI,
                                   MachineDominatorTree &MDT,
                                   const TargetRegisterClass *WaveMaskRC) {
  assert(PHI.isPHI());
  Register DstReg = PHI.getOperand(0).getReg();
  if (!DstReg.isVirtual())
    return Register();
  const TargetRegisterClass *DstRC = MRI.getRegClass(DstReg);

  if (!SIRegisterInfo::isSGPRClass(DstRC) || WaveMaskRC == DstRC)
    return Register();

  Register CommonReg;
  bool HasImplicitDefInput = false;

  for (unsigned i = 1, e = PHI.getNumOperands(); i < e; i += 2) {
    Register Incoming = PHI.getOperand(i).getReg();

    if (!SIRegisterInfo::isSGPRClass(MRI.getRegClass(Incoming)))
      return Register();

    if (Incoming == DstReg)
      continue;

    MachineInstr *Def = MRI.getVRegDef(Incoming);
    if (Def && Def->isImplicitDef()) {
      HasImplicitDefInput = true;
      continue;
    }

    if (CommonReg && CommonReg != Incoming)
      return Register();
    CommonReg = Incoming;
  }

  if (!CommonReg)
    return Register();

  // When IMPLICIT_DEF inputs are present the common value must dominate the
  // PHI block, otherwise the replacement would be invalid on paths that
  // originally carried IMPLICIT_DEF.
  if (HasImplicitDefInput) {
    MachineInstr *DefCommonReg = MRI.getVRegDef(CommonReg);
    MachineBasicBlock *DefMBB = DefCommonReg->getParent();
    MachineBasicBlock *PHIBB = PHI.getParent();
    if (DefCommonReg && MDT.dominates(DefMBB, PHIBB))
      return CommonReg;
  }

  return Register();
}

/// Walk every PHI in the function and try to replace uniform (SGPR) PHIs that
/// merge values through divergent control flow with their single real incoming
/// value.
static bool simplifyMachinePHIs(MachineFunction &MF,
                                MachineDominatorTree &MDT) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const TargetRegisterClass *WaveMaskRC =
      ST.getRegisterInfo()->getWaveMaskRegClass();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB.phis())) {
      Register NewReg = simplifyMachinePHI(MI, MRI, MDT, WaveMaskRC);
      if (!NewReg)
        continue;

      Register OldReg = MI.getOperand(0).getReg();

      LLVM_DEBUG(dbgs() << "Simplifying PHI: " << MI << "  -> "
                        << printReg(NewReg) << "\n");

      if (!MRI.hasOneUse(OldReg))
        MRI.clearKillFlags(NewReg);
      MRI.replaceRegWith(OldReg, NewReg);
      
      MI.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
//
// This pass lowers all occurrences of i1 values (with a vreg_1 register class)
// to either vreg_32 (32-bit vgpr per lane) or wave-mask sgpr. The pass assumes
// machine SSA form and a per-thread control flow graph.
//
// Before this pass, values that are semantically i1 and are defined and used
// within the same basic block are already represented as lane masks in scalar
// registers. However, values that cross basic blocks are always transferred
// between basic blocks in vreg_1 virtual registers and are lowered by this
// pass.
//
// The only instructions that define vreg_1 virtual registers are COPY, PHI,
// and IMPLICIT_DEF.
// SIFixSGPRCopies may introduce scalar consumers (e.g. S_AND_B64) that directly
// use vreg_1 values.
// Vreg_1 regions that transitively feed such consumers through COPY/PHI chains
// are preserved as wave-mask sgprs; all others are widened to vreg_32.
//
//===----------------------------------------------------------------------===//
bool Vreg1WideningHelper::widenVreg1s() {
  bool Changed = false;
  SmallSetVector<MachineInstr *, 8> DeadCopies;
  SmallVector<MachineInstr *, 8> DeferredCopyFixups;
  DenseSet<Register> Vreg32Set;

  // Pre-pass: classify which vreg_1 registers must stay in scalar
  // lane-mask form.  Seed the worklist with vreg_1 registers that have
  // a non-structural consumer (anything other than COPY/PHI/IMPLICIT_DEF,
  // e.g. S_AND_B64), then backward-propagate through defining COPY/PHI
  // chains so the entire feeding region is preserved.
  DenseSet<Register> PreservedLaneMaskRegs;
  {
    SmallVector<Register, 8> Worklist;

    for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (!isVreg1(Reg))
        continue;

      for (MachineInstr &Use : MRI->use_instructions(Reg)) {
        if (!Use.isPHI() && Use.getOpcode() != AMDGPU::COPY &&
            Use.getOpcode() != AMDGPU::IMPLICIT_DEF) {
          if (PreservedLaneMaskRegs.insert(Reg).second)
            Worklist.push_back(Reg);
          break;
        }
      }
    }

    while (!Worklist.empty()) {
      Register Reg = Worklist.pop_back_val();
      MachineInstr *DefMI = MRI->getVRegDef(Reg);
      if (!DefMI)
        continue;
      auto MarkPreserved = [&](Register Src) {
        if (isVreg1(Src) && PreservedLaneMaskRegs.insert(Src).second)
          Worklist.push_back(Src);
      };
      if (DefMI->getOpcode() == AMDGPU::COPY)
        MarkPreserved(DefMI->getOperand(1).getReg());
      else if (DefMI->isPHI())
        for (unsigned I = 1, E = DefMI->getNumOperands(); I < E; I += 2)
          MarkPreserved(DefMI->getOperand(I).getReg());
    }
  }

  // Round#1, create the replacing instruction per Vreg1 definition.
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      if (!MI.isPHI() && MI.getOpcode() != AMDGPU::COPY &&
          MI.getOpcode() != AMDGPU::IMPLICIT_DEF)
        continue;

      // Collect COPYs to be revisited after vreg_1 defs are reclassed.
      if (MI.getOpcode() == AMDGPU::COPY) {
        auto SrcReg = MI.getOperand(1).getReg();
        // If SrcReg has been reclassed, it is in Vreg32Set or
        // PreservedLaneMaskRegs.
        if (isVreg1(SrcReg) || Vreg32Set.contains(SrcReg) ||
            PreservedLaneMaskRegs.contains(SrcReg))
          DeferredCopyFixups.push_back(&MI);
      }

      Register DstReg = MI.getOperand(0).getReg();
      if (!isVreg1(DstReg))
        continue;

      Changed = true;

      // If this vreg_1 must stay as a scalar lane-mask, preserve it by
      // reclassing to the wave-mask sgpr class instead of widening.
      if (PreservedLaneMaskRegs.contains(DstReg)) {
        LLVM_DEBUG(dbgs() << "preserve lane mask for vreg1 def: " << MI);
        markAsLaneMask(DstReg);
        continue;
      }

      LLVM_DEBUG(dbgs() << "create vreg32 def that replaces vreg1 def: " << MI);
      DebugLoc DL = MI.getDebugLoc();

      assert(!MI.getOperand(0).getSubReg());

      Register DefReg32b = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
      Vreg32Set.insert(DefReg32b);

      if (MI.getOpcode() == AMDGPU::IMPLICIT_DEF || MI.isPHI()) {
        // Simply replace the register with on existing instructions.
        MRI->replaceRegWith(DstReg, DefReg32b);
      } else if (MI.getOpcode() == AMDGPU::COPY) {
        Register SrcReg = MI.getOperand(1).getReg();
        assert(!MI.getOperand(1).getSubReg());
        if (isLaneMaskReg(SrcReg)) {
          buildLaneMaskToVGPR(MBB, MI, DL, DefReg32b, SrcReg);
          DeadCopies.insert(&MI);
        } else {
          assert(isVreg1(SrcReg) || Vreg32Set.contains(SrcReg));
        }

        MRI->replaceRegWith(DstReg, DefReg32b);
      }
    } // For MI.
  } // For MBB.

  // Round#2, rewrite deferred COPYs at representation boundaries.
  for (auto *MI : DeferredCopyFixups) {
    if (DeadCopies.contains(MI))
      continue;

    auto SrcReg = MI->getOperand(1).getReg();
    auto DstReg = MI->getOperand(0).getReg();
    DebugLoc DL = MI->getDebugLoc();

    if (PreservedLaneMaskRegs.contains(SrcReg)) {
      if (!isLaneMaskReg(DstReg)) {
        buildLaneMaskToVGPR(*MI->getParent(), *MI, DL, DstReg, SrcReg);
        DeadCopies.insert(MI);
      }
      // If dest is also a lane mask, the COPY is already legal.
      continue;
    }

    // Should have been renamed to VGPR_32.
    assert(isVreg32(SrcReg));
    if (isLaneMaskReg(DstReg)) {
      buildVGPRToLaneMask(*MI->getParent(), *MI, DL, DstReg, SrcReg);
      DeadCopies.insert(MI);
    } else
      assert(isVreg32(DstReg));
  }

  // Round#2b, fix PHIs with lane mask dst that received widened vgpr_32 operands.
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB.phis()) {
      // Skip instructions with no lane mask destination register (SGPR).
      if (!isLaneMaskReg(MI.getOperand(0).getReg()))
        continue;

      for (unsigned I = 1; I < MI.getNumOperands(); I += 2) {
        assert(I + 1 < MI.getNumOperands());
        // Skip operands that were not widened by Round#1.
        if (!Vreg32Set.contains(MI.getOperand(I).getReg()))
          continue;
        // Convert vgpr_32 back to lane mask in the predecessor block.
        MachineBasicBlock *PredMBB = MI.getOperand(I + 1).getMBB();
        Register LaneMaskReg = MRI->createVirtualRegister(
            TII->getRegisterInfo().getWaveMaskRegClass());
        BuildMI(*PredMBB, PredMBB->getFirstTerminator(), DebugLoc(),
                TII->get(AMDGPU::V_CMP_NE_U32_e64), LaneMaskReg)
            .addReg(MI.getOperand(I).getReg())
            .addImm(0);
        MI.getOperand(I).setReg(LaneMaskReg);
      }
    }
  }

  for (MachineInstr *MI : DeadCopies)
    MI->eraseFromParent();
  DeadCopies.clear();

  return Changed;
}

bool AMDGPUFinalizeISelWaveTransform::runOnMachineFunction(
    MachineFunction &MF) {
  // Only need to run this in SelectionDAG path.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::Selected))
    return false;

  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  bool Changed = simplifyMachinePHIs(MF, MDT);

  Vreg1WideningHelper Helper(&MF);
  Changed |= Helper.widenVreg1s();
  return Helper.cleanConstrainRegs(Changed);
}
