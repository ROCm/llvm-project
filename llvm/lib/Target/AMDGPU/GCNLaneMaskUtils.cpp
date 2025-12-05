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
bool GCNLaneMaskUtils::isConstantLaneMask(Register Reg, bool &Val, MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBIter) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  LLVM_DEBUG(dbgs() << "isConstantLaneMask(" << printReg(Reg, MRI.getTargetRegisterInfo(), 0, &MRI) << "," << MBB.name() << ") : \n");
  LLVM_DEBUG(dbgs() << "MBBIter:");
  if(MBBIter != MBB.end()) MBBIter->dump();
  else LLVM_DEBUG(dbgs() << "end of block");
  LLVM_DEBUG(dbgs() << "\n");

  MachineInstr *MI = nullptr;
  for (;;) {
    MI = getRegisterInfo().getDomVRegDefInBasicBlock(Reg, MBB, MBBIter);
    if (!MI) {
      // This can happen when called from GCNLaneMaskUpdater, where Reg can
      // be a placeholder that has not yet been filled in.
      LLVM_DEBUG(dbgs() << "MI == nullptr, return false\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "MI:");
    MI->dump();
    LLVM_DEBUG(dbgs() << "\n");
    
    if (MI->getOpcode() == AMDGPU::IMPLICIT_DEF){
      LLVM_DEBUG(dbgs() << "MI->getOpcode() == AMDGPU::IMPLICIT_DEF, return true;\n");
      return true;
    }
    if (MI->getOpcode() != AMDGPU::COPY)
      break;

    Reg = MI->getOperand(1).getReg();
    if (!Register::isVirtualRegister(Reg)){
      LLVM_DEBUG(dbgs() << "!Register::isVirtualRegister(Reg), return false\n");
      return false;}
    if (!maybeLaneMask(Reg)){
      LLVM_DEBUG(dbgs() << "!maybeLaneMask(Reg), return false\n");
      return false;}

    MBBIter = MI->getIterator();
  }

  LLVM_DEBUG(dbgs() << "MI after loop:");
  MI->dump();
  LLVM_DEBUG(dbgs() << "\n");

  if (MI->getOpcode() != LMC.MovOpc){
    LLVM_DEBUG(dbgs() << "MI->getOpcode() != LMC.MovOpc, return false\n");
    return false;}

    if (!MI->getOperand(1).isImm()){
      LLVM_DEBUG(dbgs() << "!MI->getOperand(1).isImm(), return false\n");
      return false;}

  int64_t Imm = MI->getOperand(1).getImm();
  if (Imm == 0) {
    LLVM_DEBUG(dbgs() << "Imm == 0, Val = false, return true\n");
    Val = false;
    return true;
  }
  if (Imm == -1) {
    LLVM_DEBUG(dbgs() << "Imm == -1, Val = true, return true\n");
    Val = true;
    return true;
  }

  LLVM_DEBUG(dbgs() << "End of isConstantLaneMask, return false\n");
  return false;
}

/// Create a virtual lanemask register.
Register GCNLaneMaskUtils::createLaneMaskReg() const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  return MRI.createVirtualRegister(LMC.LaneMaskRC);
}

/// Insert the moral equivalent of
///
///    DstReg = (PrevReg & ~EXEC) | (CurReg & EXEC)
///
/// before \p I in basic block \p MBB. Some simplifications are applied on the
/// fly based on constant inputs and analysis via \p LMA
///
/// \param DstReg The virtual register into which the merged mask is written.
/// \param PrevReg The virtual register with the "previous" lane mask value;
///                may be null to indicate an undef value.
/// \param CurReg The virtual register with the "current" lane mask value to
///               be merged into "previous".
/// \param LMA If non-null, used to test whether CurReg may already be a subset
///            of EXEC.
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
  MachineRegisterInfo &MRI = MF.getRegInfo();

  Printable destRegPrintable = printReg(DstReg , MRI.getTargetRegisterInfo(), 0, &MRI);
  Printable curRegPrintable = printReg(CurReg , MRI.getTargetRegisterInfo(), 0, &MRI);
  Printable prevRegPrintable = printReg(PrevReg , MRI.getTargetRegisterInfo(), 0, &MRI);

  LLVM_DEBUG(dbgs() << "\t\tGCNLaneMaskUtils::buildMergeLaneMasks(" << MBB.name() << ",...):\n");
  LLVM_DEBUG(dbgs() << "\t\t DstReg : BlockInfo.Merged : " << destRegPrintable << "\n");
  LLVM_DEBUG(dbgs() << "\t\t PrevReg : Previous : " << prevRegPrintable << "\n");
  LLVM_DEBUG(dbgs() << "\t\t CurReg : BlockInfo.Value : " << curRegPrintable << "\n");
  LLVM_DEBUG(dbgs() << "\t\t Create instr : " << destRegPrintable << " = (" << prevRegPrintable << " & ~EXEC) | (" << curRegPrintable << " & EXEC) : \n");
  LLVM_DEBUG(dbgs() << "\t\tPrevConstant:" << PrevConstant << " CurConstant:" << CurConstant << "\n");
  LLVM_DEBUG(dbgs() << "\t\tPrevVal:" << PrevVal << " CurVal:" << CurVal << "\n");
  LLVM_DEBUG(dbgs() << "\t\tIterator I:");
  if(I != MBB.end()) I->dump();
  else LLVM_DEBUG(dbgs() << "end of block");
  LLVM_DEBUG(dbgs() << "\n");

  assert(PrevReg);

  if (PrevConstant && CurConstant) {// is wave wide constant?
    if (PrevVal == CurVal) {
      LLVM_DEBUG(dbgs() << "\t ");
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(CurReg)->dump();
    } else if (CurVal) {
      // If PrevReg is undef, prefer to propagate a full constant.
      LLVM_DEBUG(dbgs() << "\t ");
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg)
          .addReg(PrevReg ? LMC.ExecReg : CurReg)->dump();
    } else {
      LLVM_DEBUG(dbgs() << "\t ");
      BuildMI(MBB, I, DL, TII->get(LMC.XorOpc), DstReg)
          .addReg(LMC.ExecReg)
          .addImm(-1)->dump();
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
    bool isCurRegSubsetOfExec = LMA && LMA->isSubsetOfExec(CurReg, MBB, I);
    LLVM_DEBUG(dbgs() << "isSubsetOfExec(" << printReg(CurReg, MRI.getTargetRegisterInfo(), 0, &MRI) << "," << MBB.name() << ") : " << isCurRegSubsetOfExec << "\n");
    if ((PrevConstant && PrevVal) || isCurRegSubsetOfExec) {
      CurMaskedReg = CurReg;
    } else {
      CurMaskedReg = createLaneMaskReg();
      LLVM_DEBUG(dbgs() << "\t ");
      CurMaskedBuilt = BuildMI(MBB, I, DL, TII->get(LMC.AndOpc), CurMaskedReg)
                           .addReg(CurReg)
                           .addReg(LMC.ExecReg);
      CurMaskedBuilt->dump();
    }
  }

  // TODO-NOW: reevaluate the masking logic in case of CurConstant && CurVal &&
  // accumulating

  if (PrevConstant && !PrevVal) {
    if (CurMaskedBuilt) {
      CurMaskedBuilt->getOperand(0).setReg(DstReg);
      LLVM_DEBUG(dbgs() << "\t ");
      CurMaskedBuilt->dump();
    } else {
      LLVM_DEBUG(dbgs() << "\t ");
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(CurMaskedReg)->dump();
    }
  } else if (CurConstant && !CurVal) {
    if (PrevMaskedBuilt) {
      PrevMaskedBuilt->getOperand(0).setReg(DstReg);
      LLVM_DEBUG(dbgs() << "\t ");
      PrevMaskedBuilt->dump();
    } else {
      LLVM_DEBUG(dbgs() << "\t ");
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(PrevMaskedReg)->dump();
    }
  } else if (PrevConstant && PrevVal) {
    LLVM_DEBUG(dbgs() << "\t ");
    BuildMI(MBB, I, DL, TII->get(LMC.OrN2Opc), DstReg)
        .addReg(CurMaskedReg)
        .addReg(LMC.ExecReg)->dump();
  } else {
    LLVM_DEBUG(dbgs() << "\t ");
    BuildMI(MBB, I, DL, TII->get(LMC.OrOpc), DstReg)
        .addReg(PrevMaskedReg)
        .addReg(CurMaskedReg ? CurMaskedReg : LMC.ExecReg)->dump();
  }
  LLVM_DEBUG(dbgs() << "\t\tGCNLaneMaskUtils::buildMergeLaneMasks() ends\n");
}

/// Conservatively determine whether the \p Reg is a subset of EXEC for
/// \p UseBlock, i.e. it returns true if it can statically prove that
/// (Reg & EXEC) == Reg when used in \p UseBlock.
bool GCNLaneMaskAnalysis::isSubsetOfExec(Register Reg,
                                         MachineBasicBlock &UseBlock,
                                         MachineBasicBlock::iterator I,
                                         unsigned RemainingDepth) {
  MachineRegisterInfo &MRI = LMU.function()->getRegInfo();
  MachineInstr* DefInstr = nullptr;
  const AMDGPU::LaneMaskConstants &LMC = LMU.getLaneMaskConsts();
  LLVM_DEBUG(dbgs() << "isSubsetOfExec(" << printReg(Reg, MRI.getTargetRegisterInfo(), 0, &MRI) << "," << UseBlock.name() << ") : \n");
  if(I != UseBlock.end()) I->dump();
  else LLVM_DEBUG(dbgs() << "I: end of block\n");
  
  for (;;) {
    if (!Register::isVirtualRegister(Reg)) {
      if (Reg == LMC.ExecReg &&
          (!DefInstr || DefInstr->getParent() == &UseBlock)){
            LLVM_DEBUG(dbgs() << "Reg is EXEC in same BB, return true\n");
            return true;
        }
        LLVM_DEBUG(dbgs() << "Reg is not EXEC or is in other BB, return false\n");
      return false;
    }

    DefInstr = LMU.getRegisterInfo().getDomVRegDefInBasicBlock(Reg, UseBlock, I);
    if(!DefInstr){
      LLVM_DEBUG(dbgs() << "DefInstr == nullptr, return false\n");
      return false;}
    if (DefInstr->getOpcode() == AMDGPU::COPY) {
      Reg = DefInstr->getOperand(1).getReg();
      I = DefInstr->getIterator(); //pointer to iterator
      continue;
    }

    if (DefInstr->getOpcode() == LMC.MovOpc) {
      if (DefInstr->getOperand(1).isImm() &&
          DefInstr->getOperand(1).getImm() == 0){
            LLVM_DEBUG(dbgs() << "MOV 0, return true\n");
        return true;}
        LLVM_DEBUG(dbgs() << "MOV is not imm or not 0, return false\n");
      return false;
    }

    break;
  }

  LLVM_DEBUG(dbgs() << "DefInstr:");
  if(DefInstr) DefInstr->dump();
  LLVM_DEBUG(dbgs() << "\n");

  if (DefInstr->getParent() != &UseBlock){
    LLVM_DEBUG(dbgs() << "DefInstr->getParent() != &UseBlock, return false\n");
    return false;}

  auto CacheIt = SubsetOfExec.find(Reg);
  if (CacheIt != SubsetOfExec.end()){
    LLVM_DEBUG(dbgs() << "CacheIt != SubsetOfExec.end(), return CacheIt->second: " << CacheIt->second << " \n");
    return CacheIt->second;
  }
  // V_CMP_xx always return a subset of EXEC.
  if (DefInstr->isCompare() &&
      (SIInstrInfo::isVOPC(*DefInstr) || SIInstrInfo::isVOP3(*DefInstr))) {
    SubsetOfExec[Reg] = true;
    LLVM_DEBUG(dbgs() << "DefInstr is VOPC or VOP3, return true\n");
    return true;
  }

  if (!RemainingDepth--){
    LLVM_DEBUG(dbgs() << "RemainingDepth-- is 0, return false\n");
    return false;
  }

  bool LikeOr = DefInstr->getOpcode() == LMC.OrOpc ||
                DefInstr->getOpcode() == LMC.XorOpc ||
                DefInstr->getOpcode() == LMC.CSelectOpc;
  bool IsAnd = DefInstr->getOpcode() == LMC.AndOpc;
  bool IsAndN2 = DefInstr->getOpcode() == LMC.AndN2Opc;
  LLVM_DEBUG(dbgs() << "LikeOr: " << LikeOr << " IsAnd: " << IsAnd << " IsAndN2: " << IsAndN2 << "\n");
  if ((LikeOr || IsAnd || IsAndN2) &&
      (DefInstr->getOperand(1).isReg() && DefInstr->getOperand(2).isReg())) {
    bool FirstIsSubset = isSubsetOfExec(DefInstr->getOperand(1).getReg(),
                                        UseBlock, DefInstr->getIterator(), RemainingDepth);//Definstr should be iterator

    LLVM_DEBUG(dbgs() << "FirstIsSubset: " << FirstIsSubset << "\n");
    
    if (!FirstIsSubset && (LikeOr || IsAndN2)){
      bool res = SubsetOfExec.try_emplace(Reg, false).first->second;
      LLVM_DEBUG(dbgs() << "FirstIsSubset is false and (LikeOr || IsAndN2), return res: " << res << "\n");
      return res;}

    if (FirstIsSubset && (IsAnd || IsAndN2)) {
      SubsetOfExec[Reg] = true;
      LLVM_DEBUG(dbgs() << "FirstIsSubset is true and (IsAnd || IsAndN2), return true\n");
      return true;
    }

    bool SecondIsSubset = isSubsetOfExec(DefInstr->getOperand(2).getReg(),
                                         UseBlock, DefInstr->getIterator(), RemainingDepth);//Definstr should be iterator
    LLVM_DEBUG(dbgs() << "SecondIsSubset: " << SecondIsSubset << "\n");
    if (!SecondIsSubset){
      bool res = SubsetOfExec.try_emplace(Reg, false).first->second;
      LLVM_DEBUG(dbgs() << "SecondIsSubset is false, return res: " << res << "\n");
      return res;}

    SubsetOfExec[Reg] = true;
    LLVM_DEBUG(dbgs() << "SecondIsSubset is true, return true\n");
    return true;
  }

  LLVM_DEBUG(dbgs() << "End of function ,return false\n");
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
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::addAvailable(" << Block.name() << "," << printReg(Value, MRI.getTargetRegisterInfo(), 0, &MRI) << ")\n");

  BlockIt->Value = Value;
}

/// Return the value in the middle of the block, i.e. before any change that
/// was registered via \ref addAvailable.
Register GCNLaneMaskUpdater::getValueInMiddleOfBlock(MachineBasicBlock &Block) {
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueInMiddleOfBlock(" << Block.name() << ")\n");
  if (!Processed)
    process();
  Register reg = Accumulator;
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueInMiddleOfBlock(" << Block.name() << "," << printReg(reg, MRI.getTargetRegisterInfo(), 0, &MRI) << ")\n");
  return reg;
}

/// Return the value at the end of the given block, i.e. after any change that
/// was registered via \ref addAvailable.
///
/// Note: If \p Block is the reset block with ResetAtEnd
///       reset mode, then this value will be 0. You likely want
///       \ref getPreReset instead.
Register GCNLaneMaskUpdater::getValueAtEndOfBlock(MachineBasicBlock &Block) {
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueAtEndOfBlock(" << Block.name() << ")\n");
  if (!Processed)
    process();
  Register reg = Accumulator;
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueAtEndOfBlock(" << Block.name() << "," << printReg(reg, MRI.getTargetRegisterInfo(), 0, &MRI) << ")\n");
  return reg;
}

/// Return the value in \p Block after the value merge (if any).
Register GCNLaneMaskUpdater::getValueAfterMerge(MachineBasicBlock &Block) {
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueAfterMerge(" << Block.name() << ")\n");
  if (!Processed)
    process();
  Register reg = AMDGPU::NoRegister;
  auto BlockIt = findBlockInfo(Block);
  if (BlockIt != Blocks.end()) {
    if (BlockIt->Value){
      reg = Accumulator;
      LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueAfterMerge(" << Block.name() << "," << printReg(reg, MRI.getTargetRegisterInfo(), 0, &MRI) << ") returning Merged.\n");
      return reg;
    }
    if (BlockIt->Flags & ResetInMiddle){
      reg = ZeroReg;
      LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueAfterMerge(" << Block.name() << "," << printReg(reg, MRI.getTargetRegisterInfo(), 0, &MRI) << ") returning ZeroReg.\n");
      return reg;
    }
  }

  // We didn't merge anything in the block, but the block may still be
  // ResetAtEnd, in which case we need the pre-reset value.
  reg = Accumulator;
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::getValueAfterMerge(" << Block.name() << "," << printReg(reg, MRI.getTargetRegisterInfo(), 0, &MRI) << ")\n");
  return reg;
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
/// for lane mask calculation. Take terminators and SCC into account.
static MachineBasicBlock::iterator
getSaluInsertionAtEnd(MachineBasicBlock &MBB) {
  auto InsertionPt = MBB.getFirstTerminator();
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

void GCNLaneMaskUpdater::insertAccumulatorResets() {
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::insertAccumulatorResets()\n");
  const SIInstrInfo *TII = LMU.function()->getSubtarget<GCNSubtarget>().getInstrInfo();

  for (auto &[B, Accumulators] : AccumulatorResetBlocks) {

    MachineBasicBlock::iterator I = B->getFirstTerminator();
    if (I->getOpcode() == LMU.getLaneMaskConsts().MovTermOpc && I->getOperand(0).getReg() == LMU.getLaneMaskConsts().ExecReg) {
      LLVM_DEBUG(dbgs() << "    MovTermOpc to EXEC found, set Desc to MovOpc\n");
      I->setDesc(TII->get(LMU.getLaneMaskConsts().MovOpc));
      I++;
    }
    
    LLVM_DEBUG(dbgs() << "    insertion point:");
    if(I == B->end())
      LLVM_DEBUG(dbgs() << "    end of block");
    else
      I->dump();
    LLVM_DEBUG(dbgs() << "\n");

    for (Register Acc : Accumulators) {
      LLVM_DEBUG(dbgs() << "    Resetting accumulator: " << printReg(Acc, MRI.getTargetRegisterInfo(), 0, &MRI) << "@" << B->name()<< "\n");
      BuildMI(*B, I, {}, TII->get(LMU.getLaneMaskConsts().MovOpc), Acc).addImm(0)->dump();
    }
  }
}

/// Internal method to insert merge instructions.
void GCNLaneMaskUpdater::process() {
  LLVM_DEBUG(dbgs() << "\n\tGCNLaneMaskUpdater::process() begins\n");
  MachineRegisterInfo &MRI = LMU.function()->getRegInfo();
  const SIInstrInfo *TII =
      LMU.function()->getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock &Entry = LMU.function()->front();

  if (!ZeroReg) {
    ZeroReg = LMU.createLaneMaskReg();
    BuildMI(Entry, Entry.getFirstTerminator(), {},
            TII->get(LMU.getLaneMaskConsts().MovOpc), ZeroReg)
        .addImm(0);
  }
  LLVM_DEBUG(dbgs() << "\tZeroReg:" << printReg(ZeroReg, MRI.getTargetRegisterInfo(), 0, &MRI) << "\n");
  LLVM_DEBUG(dbgs() << "\n\tAdding available values:\n");

  if (!Accumulator) {
    Accumulator = LMU.createLaneMaskReg();
    LLVM_DEBUG(dbgs() << "\tCreating Accumulator:" << printReg(Accumulator, MRI.getTargetRegisterInfo(), 0, &MRI) << "\n");
    BuildMI(Entry, Entry.getFirstTerminator(), {},
            TII->get(LMU.getLaneMaskConsts().MovOpc), Accumulator)
        .addImm(0);
  }
  LLVM_DEBUG(dbgs() << "\n\tMachineSSAUpdater ready, begin merging\n");

  // Add available values.
  for (BlockInfo &Info : Blocks) {
    LLVM_DEBUG(dbgs() << "\tAdd avail value for BlockInfo:" << Info.Block->name() << "\n\t");
    assert(Info.Flags || Info.Value);
    Info.dump(MRI);
    if(!Info.Value || (Info.Flags & ResetAtEnd)){
      LLVM_DEBUG(dbgs() << "  !Info.Value || (Info.Flags & ResetAtEnd) is true\n");
      LLVM_DEBUG(dbgs() << "  AccumulatorResetBlocks[" << Info.Block->name() << "]:" << printReg(Accumulator, MRI.getTargetRegisterInfo(), 0, &MRI) << "\n");
      AccumulatorResetBlocks[Info.Block].insert(Accumulator);
    }
  }
  
  // Once the SSA updater is ready, we can fill in all merge code, relying
  // on the SSA updater to insert required PHIs.
  for (BlockInfo &Info : Blocks) {
    if (!Info.Value)
      continue;
    
    LLVM_DEBUG(dbgs() << "\tmerge ");
    Info.dump(MRI);
    LLVM_DEBUG(dbgs() << "\n");
    // Determine the "previous" value, if any.
    Register Previous;
    if (Info.Block != &LMU.function()->front() &&
        !(Info.Flags & ResetInMiddle)) {
      Previous = Accumulator;
    } else {
      LLVM_DEBUG(dbgs() << "\tEither one of the following 2 conds are true:\n");
      LLVM_DEBUG(dbgs() << "\tInfo.Block == &LMU.function()->front():" << (Info.Block == &LMU.function()->front()) << "\n");
      LLVM_DEBUG(dbgs() << "\tInfo.Flags & ResetInMiddle:" << (Info.Flags & ResetInMiddle) << "\n");
      Previous = ZeroReg;
      LLVM_DEBUG(dbgs() << "\tBlock:" << Info.Block->name() << " Previous is ZeroReg:" << printReg(Previous , MRI.getTargetRegisterInfo(), 0, &MRI) << "\n");
      
    }

    // Insert merge logic.
    MachineBasicBlock::iterator insertPt = getSaluInsertionAtEnd(*Info.Block);
    LMU.buildMergeLaneMasks(*Info.Block, insertPt, {}, Accumulator, Previous,
                            Info.Value, LMA, (Previous == ZeroReg));


      /*if (Info.Flags & ResetAtEnd) {
      // We enter this if block if Info.Block is Ti and Ri
      // Here we check if Accumulator was set by a simple copy, if so, we use the corresponding register
      // This is a copy propogation optimization.
      // It depends on getting the latest def of Accumulator in Info.Block and checking if it has no uses.
      // TODO : Swithing off this optimization for nonSSA context since Accumulator will 
      // have a use at the end of Info.Block : Set Accumumlator to 0 (since Info.Block is Ri)
      // Will implement a nonSSA variant for the same.
      
      MachineInstr *mergeInstr = MRI.getVRegDef(Info.Merged);
      LLVM_DEBUG(dbgs() << "\tmergeInstr:");
      mergeInstr->dump();
      LLVM_DEBUG(dbgs() << "\n");
      if (mergeInstr->getOpcode() == AMDGPU::COPY &&
          mergeInstr->getOperand(1).getReg().isVirtual()) {
        assert(MRI.use_empty(Info.Merged));
        Info.Merged = mergeInstr->getOperand(1).getReg();
        LLVM_DEBUG(dbgs() << "\tset Merged:" << printReg(Info.Merged , MRI.getTargetRegisterInfo(), 0, &MRI) << " for block " << Info.Block->name() << "\n");
        LLVM_DEBUG(dbgs() << "\tErase mergeInstr\n");
        mergeInstr->eraseFromParent();
      }
    }*/
  }

  Processed = true;
  LLVM_DEBUG(dbgs() << "GCNLaneMaskUpdater::process() ends\n");

}

/// Find a block in the \ref Blocks structure.
SmallVectorImpl<GCNLaneMaskUpdater::BlockInfo>::iterator
GCNLaneMaskUpdater::findBlockInfo(MachineBasicBlock &Block) {
  return llvm::find_if(
      Blocks, [&](const auto &Entry) { return Entry.Block == &Block; });
}
