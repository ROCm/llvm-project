//===- SISGPRCopyOpt.cpp - Hoist SGPR->VGPR copies out of loops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass optimizes SGPR to VGPR copies that appear at loop headers as a
/// result of PHI elimination. When a VGPR PHI has an SGPR initial value, the
/// PHI elimination pass inserts a V_MOV from SGPR to VGPR at the beginning of
/// the loop body. If this SGPR value is loop-invariant, we can hoist the MOV
/// to the loop preheader, avoiding the copy on every iteration.
///
/// Copies that cannot be hoisted but are provably dead (their destination is
/// fully overwritten before any use within the loop) are deleted instead.
///
/// This pass runs after register allocation and operates on physical registers.
///
/// Pattern detected:
///   bb.preheader:
///     ; empty or other code
///     S_BRANCH bb.loop
///
///   bb.loop:
///     $vgpr = V_MOV_B64_e32 $sgpr  ; loop-invariant!
///     ...
///     $vgpr = WMMA ...  ; $vgpr redefined by loop body
///     ...
///     S_CBRANCH bb.loop
///
/// Optimization:
///   bb.preheader:
///     $vgpr = V_MOV_B64_e32 $sgpr  ; hoisted!
///     S_BRANCH bb.loop
///
///   bb.loop:
///     ; MOV removed
///     ...
///     $vgpr = WMMA ...
///     ...
///     S_CBRANCH bb.loop
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-sgpr-copy-opt"

static cl::opt<bool> EnableSGPRCopyOpt(
    "amdgpu-enable-sgpr-copy-opt",
    cl::desc("Hoist loop-invariant SGPR->VGPR copies to loop preheaders"),
    cl::Hidden, cl::init(true));

// Upper bound on the number of copies hoisted out of a single loop. Because a
// hoisted copy's destination is provably not redefined in the loop, hoisting is
// occupancy-neutral post-RA; this knob only throttles the transform while
// investigating scheduling interactions. 0 means unlimited.
static cl::opt<unsigned> MaxHoistPerLoop(
    "amdgpu-sgpr-copy-opt-max-hoist", cl::init(0), cl::Hidden,
    cl::desc("Max SGPR->VGPR copies to hoist per loop (0 = unlimited)"));

namespace {

class SISGPRCopyOpt {
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  MachineLoopInfo *MLI;

public:
  SISGPRCopyOpt(MachineLoopInfo *MLI) : MLI(MLI) {}

  bool run(MachineFunction &MF);

private:
  /// Check if a register is an SGPR and is loop-invariant (not defined in the
  /// loop).
  bool isLoopInvariantSGPR(Register Reg, MachineLoop *Loop) const;

  /// Check if a V_MOV instruction can be hoisted to the preheader.
  bool canHoistMov(MachineInstr &MI, MachineLoop *Loop) const;

  /// Check whether Reg is defined by any instruction in the loop other than
  /// Except (overlapping registers count). Used to guarantee a hoisted MOV is
  /// the sole in-loop definition of its destination.
  bool isRegDefinedInLoop(Register Reg, MachineLoop *Loop,
                          const MachineInstr *Except) const;

  /// Check whether Reg is referenced anywhere in the preheader, which would
  /// make inserting a redefining MOV there unsafe.
  bool isRegTouchedInPreheader(Register Reg,
                               MachineBasicBlock *Preheader) const;

  /// Check whether MI's destination DstReg is fully overwritten before any use
  /// later in the same block, which makes the MOV dead and safe to delete.
  bool isMovDeadInBlock(MachineInstr &MI, Register DstReg) const;

  /// Hoist eligible V_MOV instructions from loop header to preheader.
  bool hoistLoopInvariantCopies(MachineLoop *Loop);
};

class SISGPRCopyOptLegacy : public MachineFunctionPass {
public:
  static char ID;

  SISGPRCopyOptLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    SISGPRCopyOpt Impl(MLI);
    return Impl.run(MF);
  }

  StringRef getPassName() const override { return "SI SGPR Copy Optimization"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SISGPRCopyOptLegacy, DEBUG_TYPE,
                      "SI SGPR Copy Optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SISGPRCopyOptLegacy, DEBUG_TYPE,
                    "SI SGPR Copy Optimization", false, false)

char SISGPRCopyOptLegacy::ID = 0;

char &llvm::SISGPRCopyOptLegacyID = SISGPRCopyOptLegacy::ID;

FunctionPass *llvm::createSISGPRCopyOptLegacyPass() {
  return new SISGPRCopyOptLegacy();
}

bool SISGPRCopyOpt::isLoopInvariantSGPR(Register Reg, MachineLoop *Loop) const {
  if (!Reg.isPhysical() || !TRI->isSGPRPhysReg(Reg))
    return false;

  // Loop-invariant means the register is not defined anywhere in the loop.
  for (MachineBasicBlock *MBB : Loop->blocks())
    for (MachineInstr &MI : *MBB)
      for (const MachineOperand &MO : MI.defs())
        if (MO.isReg() && MO.getReg().isPhysical() &&
            TRI->regsOverlap(MO.getReg(), Reg))
          return false;

  return true;
}

bool SISGPRCopyOpt::isRegDefinedInLoop(Register Reg, MachineLoop *Loop,
                                       const MachineInstr *Except) const {
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (MachineInstr &MI : *MBB) {
      if (&MI == Except)
        continue;
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical() &&
            TRI->regsOverlap(MO.getReg(), Reg))
          return true;
      }
    }
  }
  return false;
}

bool SISGPRCopyOpt::isRegTouchedInPreheader(
    Register Reg, MachineBasicBlock *Preheader) const {
  for (MachineInstr &MI : *Preheader) {
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.getReg().isPhysical() &&
          TRI->regsOverlap(MO.getReg(), Reg))
        return true;
    }
  }
  return false;
}

bool SISGPRCopyOpt::isMovDeadInBlock(MachineInstr &MI, Register DstReg) const {
  // Register units of the destination that must be redefined before any use for
  // the MOV to be dead.
  SmallDenseSet<MCRegUnit, 16> NeededUnits;
  for (MCRegUnit U : TRI->regunits(DstReg.asMCReg()))
    NeededUnits.insert(U);

  MachineBasicBlock *MBB = MI.getParent();
  for (auto I = std::next(MI.getIterator()); I != MBB->end(); ++I) {
    // A read of any still-live unit means the MOV's value is consumed, so it is
    // not dead. Check uses before defs so a read-modify-write counts as a use.
    for (const MachineOperand &MO : I->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical() || !MO.isUse() ||
          MO.isUndef())
        continue;
      for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg()))
        if (NeededUnits.contains(U))
          return false;
    }
    // Retire the units this instruction overwrites.
    for (const MachineOperand &MO : I->operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical())
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg()))
          NeededUnits.erase(U);
    }
    if (NeededUnits.empty())
      return true; // fully redefined before any use within the block
  }

  // Reached the end of the block without a full redefinition: the value may be
  // live-out. Be conservative.
  return false;
}

bool SISGPRCopyOpt::canHoistMov(MachineInstr &MI, MachineLoop *Loop) const {
  unsigned Opc = MI.getOpcode();

  // Check for V_MOV_B32 or V_MOV_B64 from SGPR
  if (Opc != AMDGPU::V_MOV_B32_e32 && Opc != AMDGPU::V_MOV_B64_e32)
    return false;

  const MachineOperand &DstOp = MI.getOperand(0);
  const MachineOperand &SrcOp = MI.getOperand(1);

  if (!DstOp.isReg() || !SrcOp.isReg())
    return false;

  Register DstReg = DstOp.getReg();
  Register SrcReg = SrcOp.getReg();

  if (!DstReg.isPhysical() || !SrcReg.isPhysical())
    return false;

  // Source must be a loop-invariant SGPR
  if (!isLoopInvariantSGPR(SrcReg, Loop))
    return false;

  // Destination must be a VGPR
  if (!TRI->isVGPR(*MRI, DstReg))
    return false;

  // If DstReg is used before this MOV in the header, the previous iteration's
  // value is needed and hoisting would be incorrect. Implicit operands are
  // ignored: they are usually super-register liveness bookkeeping rather than
  // real data dependencies.
  MachineBasicBlock *HeaderBB = Loop->getHeader();
  for (auto I = HeaderBB->begin(); &*I != &MI; ++I) {
    for (const MachineOperand &MO : I->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical() || MO.isDef() ||
          MO.isImplicit())
        continue;
      if (TRI->regsOverlap(MO.getReg(), DstReg))
        return false;
    }
  }

  // The hoisted MOV must be the only in-loop definition of the destination. If
  // anything else in the loop writes it (e.g. a load into the data half of a
  // matrix operand, or reuse of the physical VGPR as scratch), removing the
  // per-iteration MOV would let a later iteration read the clobbered value.
  if (isRegDefinedInLoop(DstReg, Loop, &MI))
    return false;

  // Do not clobber a value the preheader itself defines or consumes: the copy
  // is inserted before the preheader terminator.
  if (MachineBasicBlock *Preheader = Loop->getLoopPreheader())
    if (isRegTouchedInPreheader(DstReg, Preheader))
      return false;

  return true;
}

bool SISGPRCopyOpt::hoistLoopInvariantCopies(MachineLoop *Loop) {
  MachineBasicBlock *Header = Loop->getHeader();
  MachineBasicBlock *Preheader = Loop->getLoopPreheader();

  if (!Preheader)
    return false;

  // Find the insertion point in the preheader (before the terminator)
  MachineBasicBlock::iterator InsertPt = Preheader->getFirstTerminator();

  bool Changed = false;
  SmallVector<MachineInstr *, 16> ToHoist;
  SmallVector<MachineInstr *, 16> ToDelete;

  // Collect movs to hoist - only look at the beginning of the header
  // (before any other non-copy/non-mov instructions)
  for (MachineInstr &MI : *Header) {
    // Stop when we hit a non-mov/non-copy/non-fence instruction
    unsigned Opc = MI.getOpcode();
    if (Opc != AMDGPU::V_MOV_B32_e32 && Opc != AMDGPU::V_MOV_B64_e32 &&
        Opc != AMDGPU::COPY && !MI.isMetaInstruction() &&
        Opc != AMDGPU::ATOMIC_FENCE && Opc != AMDGPU::WAVE_BARRIER)
      break;

    if (Opc != AMDGPU::V_MOV_B64_e32 && Opc != AMDGPU::V_MOV_B32_e32)
      continue;

    bool HoistCapReached = MaxHoistPerLoop && ToHoist.size() >= MaxHoistPerLoop;
    if (!HoistCapReached && canHoistMov(MI, Loop)) {
      ToHoist.push_back(&MI);
      continue;
    }

    // Not hoistable: delete it if it is provably dead within the loop. This
    // reclaims the pad fills that a later load overwrites before any use,
    // without the risk of hoisting a register that is redefined in the loop.
    Register DstReg = MI.getOperand(0).getReg();
    if (DstReg.isPhysical() && TRI->isVGPR(*MRI, DstReg) &&
        isMovDeadInBlock(MI, DstReg))
      ToDelete.push_back(&MI);
  }

  for (MachineInstr *MI : ToHoist) {
    LLVM_DEBUG(dbgs() << "Hoisting to preheader: " << *MI);
    MI->removeFromParent();
    Preheader->insert(InsertPt, MI);
    Changed = true;
  }

  for (MachineInstr *MI : ToDelete) {
    LLVM_DEBUG(dbgs() << "Deleting dead loop MOV: " << *MI);
    MI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool SISGPRCopyOpt::run(MachineFunction &MF) {
  if (!EnableSGPRCopyOpt)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  // Process loops innermost-first.
  for (MachineLoop *L : MLI->getLoopsInPreorder())
    Changed |= hoistLoopInvariantCopies(L);

  return Changed;
}
