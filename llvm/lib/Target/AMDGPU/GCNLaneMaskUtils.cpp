//===- GCNLaneMaskUtils.cpp --------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNLaneMaskUtils.h"

#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gcn-lane-mask-utils"

using namespace llvm;

/// Check whether the register could be a lane-mask register.
///
/// It does not distinguish between lane-masks and scalar registers that happen
/// to have the right bitsize.
bool GCNLaneMaskUtils::maybeLaneMask(Register Reg) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  return TII->getRegisterInfo().isSGPRReg(MRI, Reg) &&
         TII->getRegisterInfo().getRegSizeInBits(Reg, MRI) ==
             ST.getWavefrontSize();
}

/// Determine whether the lane-mask register \p Reg is a wave-wide constant.
/// If so, the value is stored in \p Val.
bool GCNLaneMaskUtils::isConstantLaneMask(
    Register Reg, bool &Val, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBIter) const {
  MachineInstr *MI = nullptr;
  for (;;) {
    MI = getRegisterInfo().getDomVRegDefInBasicBlock(Reg, MBB, MBBIter);
    if (!MI) {
      // This can happen when called from GCNLaneMaskUpdater, where Reg can
      // be a placeholder that has not yet been filled in.
      return false;
    }

    if (MI->getOpcode() == AMDGPU::IMPLICIT_DEF)
      return true;

    if (MI->getOpcode() != AMDGPU::COPY)
      break;

    Reg = MI->getOperand(1).getReg();
    if (!Register::isVirtualRegister(Reg))
      return false;
    if (!maybeLaneMask(Reg))
      return false;
    MBBIter = MI->getIterator();
  }

  if (MI->getOpcode() != LMC.MovOpc)
    return false;

  if (!MI->getOperand(1).isImm())
    return false;

  int64_t Imm = MI->getOperand(1).getImm();
  if (Imm == 0) {
    Val = false;
    return true;
  }
  if (Imm == -1) {
    Val = true;
    return true;
  }

  return false;
}

/// Create a virtual lanemask register.
Register GCNLaneMaskUtils::createLaneMaskReg() const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  return MRI.createVirtualRegister(LMC.LaneMaskRC);
}

/// Insert the moral equivalent of
///
///    DstReg = PrevReg | (CurReg & EXEC)
///
/// before \p I in basic block \p MBB. Some simplifications are applied on the
/// fly based on constant inputs and analysis via \p LMA
///
/// \param DstReg The virtual register into which the merged mask is written.
/// \param PrevReg The virtual register with the "previous" lane mask value;
///                may be ZeroReg or Accumulator.
/// \param CurReg The virtual register with the "current" lane mask value to
///               be merged into "previous".
/// \param LMA If non-null, used to test whether CurReg may already be a subset
///            of EXEC.
/// \param isPrevZeroReg Indicates that PrevReg is a zero register.
void GCNLaneMaskUtils::buildMergeLaneMasks(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator I,
                                           const DebugLoc &DL, Register DstReg,
                                           Register PrevReg, Register CurReg,
                                           GCNLaneMaskAnalysis *LMA,
                                           bool isPrevZeroReg) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool PrevVal = false;
  bool PrevConstant = !PrevReg || isPrevZeroReg;
  bool CurVal = false;
  bool CurConstant = isConstantLaneMask(CurReg, CurVal, MBB, I);

  assert(PrevReg);

  if (PrevConstant && CurConstant) {
    if (PrevVal == CurVal) {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(CurReg);
    } else if (CurVal) {
      // If PrevReg is undef, prefer to propagate a full constant.
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg)
          .addReg(PrevReg ? LMC.ExecReg : CurReg);
    } else {
      BuildMI(MBB, I, DL, TII->get(LMC.XorOpc), DstReg)
          .addReg(LMC.ExecReg)
          .addImm(-1);
    }
    return;
  }

  MachineInstr *PrevMaskedBuilt = nullptr;
  MachineInstr *CurMaskedBuilt = nullptr;
  Register PrevMaskedReg;
  Register CurMaskedReg;
  if (!PrevConstant) {
    PrevMaskedReg = PrevReg;
  }
  if (!CurConstant) {
    if ((PrevConstant && PrevVal) ||
        (LMA && LMA->isSubsetOfExec(CurReg, MBB, I))) {
      CurMaskedReg = CurReg;
    } else {
      CurMaskedReg = createLaneMaskReg();
      CurMaskedBuilt = BuildMI(MBB, I, DL, TII->get(LMC.AndOpc), CurMaskedReg)
                           .addReg(CurReg)
                           .addReg(LMC.ExecReg);
    }
  }

  // TODO-NOW: reevaluate the masking logic in case of CurConstant && CurVal

  if (PrevConstant && !PrevVal) {
    if (CurMaskedBuilt) {
      CurMaskedBuilt->getOperand(0).setReg(DstReg);
    } else {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(CurMaskedReg);
    }
  } else if (CurConstant && !CurVal) {
    if (PrevMaskedBuilt) {
      PrevMaskedBuilt->getOperand(0).setReg(DstReg);
    } else {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(PrevMaskedReg);
    }
  } else if (PrevConstant && PrevVal) {
    BuildMI(MBB, I, DL, TII->get(LMC.OrN2Opc), DstReg)
        .addReg(CurMaskedReg)
        .addReg(LMC.ExecReg);
  } else {
    BuildMI(MBB, I, DL, TII->get(LMC.OrOpc), DstReg)
        .addReg(PrevMaskedReg)
        .addReg(CurMaskedReg ? CurMaskedReg : LMC.ExecReg);
  }
}

/// Conservatively determine whether the \p Reg is a subset of EXEC for
/// \p UseBlock, i.e. it returns true if it can statically prove that
/// (Reg & EXEC) == Reg when used in \p UseBlock.
bool GCNLaneMaskAnalysis::isSubsetOfExec(Register Reg,
                                         MachineBasicBlock &UseBlock,
                                         MachineBasicBlock::iterator I,
                                         unsigned RemainingDepth) {
  MachineInstr *DefInstr = nullptr;
  const AMDGPU::LaneMaskConstants &LMC = LMU.getLaneMaskConsts();

  for (;;) {
    if (!Register::isVirtualRegister(Reg)) {
      if (Reg == LMC.ExecReg &&
          (!DefInstr || DefInstr->getParent() == &UseBlock))
        return true;
      return false;
    }

    DefInstr =
        LMU.getRegisterInfo().getDomVRegDefInBasicBlock(Reg, UseBlock, I);
    if (!DefInstr)
      return false;
    if (DefInstr->getOpcode() == AMDGPU::COPY) {
      Reg = DefInstr->getOperand(1).getReg();
      I = DefInstr->getIterator();
      continue;
    }

    if (DefInstr->getOpcode() == LMC.MovOpc) {
      if (DefInstr->getOperand(1).isImm() &&
          DefInstr->getOperand(1).getImm() == 0)
        return true;
      return false;
    }

    break;
  }

  if (DefInstr->getParent() != &UseBlock)
    return false;

  // V_CMP_xx always return a subset of EXEC.
  if (DefInstr->isCompare() &&
      (SIInstrInfo::isVOPC(*DefInstr) || SIInstrInfo::isVOP3(*DefInstr)))
    return true;

  if (!RemainingDepth--)
    return false;

  bool LikeOr = DefInstr->getOpcode() == LMC.OrOpc ||
                DefInstr->getOpcode() == LMC.XorOpc ||
                DefInstr->getOpcode() == LMC.CSelectOpc;
  bool IsAnd = DefInstr->getOpcode() == LMC.AndOpc;
  bool IsAndN2 = DefInstr->getOpcode() == LMC.AndN2Opc;
  if (LikeOr || IsAnd || IsAndN2) {
    const MachineOperand &Op1 = DefInstr->getOperand(1);
    bool FirstIsSubset = (Op1.isImm() && Op1.getImm() == 0) ||
                         (Op1.isReg() && isSubsetOfExec(Op1.getReg(), UseBlock,
                                                        DefInstr->getIterator(),
                                                        RemainingDepth));
    if (!FirstIsSubset && (LikeOr || IsAndN2))
      return false;

    if (FirstIsSubset && (IsAnd || IsAndN2))
      return true;

    const MachineOperand &Op2 = DefInstr->getOperand(2);
    bool SecondIsSubset =
        (Op2.isImm() && Op2.getImm() == 0) ||
        (Op2.isReg() &&
         isSubsetOfExec(Op2.getReg(), UseBlock, DefInstr->getIterator(),
                        RemainingDepth));
    if (!SecondIsSubset)
      return false;

    return true;
  }

  return false;
}

/// Initialize the updater.
void GCNLaneMaskUpdater::init() {
  Processed = false;
  Blocks.clear();
  // SSAUpdater.Initialize(LMU.getLaneMaskConsts().LaneMaskRC);
  Accumulator = AMDGPU::NoRegister;
}

/// Optional cleanup, may remove stray instructions.
void GCNLaneMaskUpdater::cleanup() {
  Processed = false;
  Blocks.clear();
  Accumulator = AMDGPU::NoRegister;
  MachineRegisterInfo &MRI = LMU.function()->getRegInfo();

  if (ZeroReg && MRI.use_empty(ZeroReg)) {
    MRI.getVRegDef(ZeroReg)->eraseFromParent();
    ZeroReg = AMDGPU::NoRegister;
  }

  for (MachineInstr *MI : PotentiallyDead) {
    Register DefReg = MI->getOperand(0).getReg();
    if (MRI.use_empty(DefReg))
      MI->eraseFromParent();
  }
  PotentiallyDead.clear();
}

/// Indicate that a reset should occur in the given block.
///
/// Can be called multiple times for the same block, flags accumulate.
void GCNLaneMaskUpdater::addReset(MachineBasicBlock &Block, ResetFlags Flags) {
  assert(!Processed);

  auto BlockIt = findBlockInfo(Block);
  if (BlockIt == Blocks.end()) {
    Blocks.emplace_back(&Block);
    BlockIt = Blocks.end() - 1;
  }

  BlockIt->Flags |= Flags;
}

/// Indicate that a new value is available in \p block. Lane mask bits
/// (per-thread boolean values) are updated.
///
/// \param Value A virtual lane mask register; the lane bits are masked by the
///              block's effective EXEC.
void GCNLaneMaskUpdater::addAvailable(MachineBasicBlock &Block,
                                      Register Value) {
  assert(!Processed);

  auto BlockIt = findBlockInfo(Block);
  if (BlockIt == Blocks.end()) {
    Blocks.emplace_back(&Block);
    BlockIt = Blocks.end() - 1;
  }
  assert(!BlockIt->Value);

  BlockIt->Value = Value;
}

/// Return the value in the middle of the block, i.e. before any change that
/// was registered via \ref addAvailable.
Register GCNLaneMaskUpdater::getValueInMiddleOfBlock(MachineBasicBlock &Block) {
  if (!Processed)
    process();
  return Accumulator;
}

/// Return the value at the end of the given block, i.e. after any change that
/// was registered via \ref addAvailable.
///
/// Note: If \p Block is the reset block with ResetAtEnd
///       reset mode, then this value will be 0. You likely want
///       \ref getPreReset instead.
Register GCNLaneMaskUpdater::getValueAtEndOfBlock(MachineBasicBlock &Block) {
  if (!Processed)
    process();
  return Accumulator;
}

/// Return the value in \p Block after the value merge (if any).
Register GCNLaneMaskUpdater::getValueAfterMerge(MachineBasicBlock &Block) {
  if (!Processed)
    process();

  auto BlockIt = findBlockInfo(Block);
  if (BlockIt != Blocks.end()) {
    if (BlockIt->Value)
      return Accumulator;
    if (BlockIt->Flags & ResetInMiddle)
      return ZeroReg;
  }

  // We didn't merge anything in the block, but the block may still be
  // ResetAtEnd, in which case we need the pre-reset value.
  return Accumulator;
}

/// Determine whether \p MI defines and/or uses SCC.
static void instrDefsUsesSCC(const MachineInstr &MI, bool &Def, bool &Use) {
  Def = false;
  Use = false;

  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() == AMDGPU::SCC) {
      if (MO.isUse())
        Use = true;
      else
        Def = true;
    }
  }
}

/// Return a point at the end of the given \p MBB to insert SALU instructions
/// for lane mask calculation. Take terminators, INLINEASM_BR, and SCC into
/// account.
static MachineBasicBlock::iterator
getSaluInsertionAtEnd(MachineBasicBlock &MBB) {
  auto InsertionPt = MBB.getFirstTerminator();

  // INLINEASM_BR is not marked as a terminator, but lane mask contributions
  // must be placed before it. Walk back past the INLINEASM_BR.
  if (InsertionPt != MBB.begin() &&
      std::prev(InsertionPt)->getOpcode() == TargetOpcode::INLINEASM_BR)
    --InsertionPt;

  bool TerminatorsUseSCC = false;
  for (auto I = InsertionPt, E = MBB.end(); I != E; ++I) {
    bool DefsSCC;
    instrDefsUsesSCC(*I, DefsSCC, TerminatorsUseSCC);
    if (TerminatorsUseSCC || DefsSCC)
      break;
  }

  if (!TerminatorsUseSCC)
    return InsertionPt;

  while (InsertionPt != MBB.begin()) {
    InsertionPt--;

    bool DefSCC, UseSCC;
    instrDefsUsesSCC(*InsertionPt, DefSCC, UseSCC);
    if (DefSCC)
      return InsertionPt;
  }

  // We should have at least seen an IMPLICIT_DEF or COPY
  llvm_unreachable("SCC used by terminator but no def in block");
}

/// Internal method to insert merge instructions.
void GCNLaneMaskUpdater::process() {
  const SIInstrInfo *TII =
      LMU.function()->getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock &Entry = LMU.function()->front();

  if (!ZeroReg) {
    ZeroReg = LMU.createLaneMaskReg();
    BuildMI(Entry, Entry.getFirstTerminator(), {},
            TII->get(LMU.getLaneMaskConsts().MovOpc), ZeroReg)
        .addImm(0);
  }

  if (!Accumulator) {
    Accumulator = LMU.createLaneMaskReg();
    AllAccumulators.insert(Accumulator);
    BuildMI(Entry, Entry.getFirstTerminator(), {},
            TII->get(LMU.getLaneMaskConsts().MovOpc), Accumulator)
        .addImm(0);
  }

  // Reset accumulator.
  for (BlockInfo &Info : Blocks) {
    assert(Info.Flags || Info.Value);
    if (Info.Flags)
      AccumulatorResetBlocks[Info.Block].push_back({Accumulator, Info.Flags});
  }

  // Once the SSA updater is ready, we can fill in all merge code, relying
  // on the SSA updater to insert required PHIs.
  for (BlockInfo &Info : Blocks) {
    if (!Info.Value)
      continue;

    // Determine the "previous" value.
    Register Previous;
    if (Info.Block != &LMU.function()->front() && !(Info.Flags & ResetInMiddle))
      Previous = Accumulator;
    else
      Previous = ZeroReg;

    // Insert merge logic.
    MachineBasicBlock::iterator insertPt = getSaluInsertionAtEnd(*Info.Block);
    LMU.buildMergeLaneMasks(*Info.Block, insertPt, {}, Accumulator, Previous,
                            Info.Value, LMA, Previous == ZeroReg);
  }

  Processed = true;
}

/// Find a block in the \ref Blocks structure.
SmallVectorImpl<GCNLaneMaskUpdater::BlockInfo>::iterator
GCNLaneMaskUpdater::findBlockInfo(MachineBasicBlock &Block) {
  return llvm::find_if(
      Blocks, [&](const auto &Entry) { return Entry.Block == &Block; });
}

void GCNLaneMaskUpdater::insertAccumulatorResets() {
  const SIInstrInfo *TII =
      LMU.function()->getSubtarget<GCNSubtarget>().getInstrInfo();
  for (auto &[B, AccFlagPairs] : AccumulatorResetBlocks) {

    // TODO : We only need to compute EndInsertPt if any of B's AccFlagPairs has
    // ResetAtEnd
    MachineBasicBlock::iterator EndInsertPt;
    EndInsertPt = B->getFirstTerminator();
    if (EndInsertPt != B->end() && EndInsertPt->getOpcode() == LMU.getLaneMaskConsts().MovTermOpc &&
        EndInsertPt->getOperand(0).getReg() ==
            LMU.getLaneMaskConsts().ExecReg) {
      EndInsertPt->setDesc(TII->get(LMU.getLaneMaskConsts().MovOpc));
      EndInsertPt++;
    }

    for (auto &[Acc, Flags] : AccFlagPairs) {
      if (Flags & ResetInMiddle) {
        // Insert at beginning of basic block for ResetInMiddle
        BuildMI(*B, B->begin(), {}, TII->get(LMU.getLaneMaskConsts().MovOpc),
                Acc)
            .addImm(0);
      }
      if (Flags & ResetAtEnd) {
        // Insert at end of basic block for ResetAtEnd
        BuildMI(*B, EndInsertPt, {}, TII->get(LMU.getLaneMaskConsts().MovOpc),
                Acc)
            .addImm(0);
      }
    }
  }
}
