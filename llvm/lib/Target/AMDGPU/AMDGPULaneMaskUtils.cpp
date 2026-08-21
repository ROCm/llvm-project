//===- AMDGPULaneMaskUtils.cpp -----------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULaneMaskUtils.h"

#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "amdgpu-lane-mask-utils"

using namespace llvm;

static MachineBasicBlock::iterator
getSaluInsertionAtEnd(MachineBasicBlock &MBB);

/// Check whether the register could be a lane-mask register.
///
/// It does not distinguish between lane-masks and scalar registers that happen
/// to have the right bitsize.
bool AMDGPULaneMaskUtils::maybeLaneMask(Register Reg) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  return TII->getRegisterInfo().isSGPRReg(MRI, Reg) &&
         TII->getRegisterInfo().getRegSizeInBits(Reg, MRI) ==
             ST.getWavefrontSize();
}

/// Determine whether the lane-mask register \p Reg is a wave-wide constant.
/// If so, the value is stored in \p Val.
bool AMDGPULaneMaskUtils::isConstantLaneMask(
    Register Reg, bool &Val, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBIter) const {
  MachineInstr *MI = nullptr;
  for (;;) {
    MI = getRegisterInfo().getDomVRegDefInBasicBlock(Reg, MBB, MBBIter);
    if (!MI) {
      // This can happen when called from AMDGPULaneMaskUpdater, where Reg can
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
Register AMDGPULaneMaskUtils::createLaneMaskReg() const {
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
void AMDGPULaneMaskUtils::buildMergeLaneMasks(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator I,
                                           const DebugLoc &DL, Register DstReg,
                                           Register PrevReg, Register CurReg,
                                           AMDGPULaneMaskAnalysis *LMA,
                                           bool isPrevZeroReg) const {
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  assert(PrevReg &&
         "previous lane mask must be the zero reg or an accumulator");

  auto buildCopy = [&](Register Dst, Register Src) {
    BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), Dst).addReg(Src);
  };
  auto buildBinOp = [&](unsigned Opc, Register Dst, Register A, Register B) {
    BuildMI(MBB, I, DL, TII->get(Opc), Dst).addReg(A).addReg(B);
  };

  bool CurVal = false;
  bool CurIsConstant = isConstantLaneMask(CurReg, CurVal, MBB, I);

  // Case A -- previous is zero: DstReg = CurReg & EXEC.
  if (isPrevZeroReg) {
    // Constant current: -1 & EXEC == EXEC, 0 & EXEC == 0.
    if (CurIsConstant)
      buildCopy(DstReg, CurVal ? LMC.ExecReg : CurReg);
    else if (LMA && LMA->isSubsetOfExec(CurReg, MBB, I))
      buildCopy(DstReg, CurReg);
    else
      buildBinOp(LMC.AndOpc, DstReg, CurReg, LMC.ExecReg);
    return;
  }

  // Case B -- previous is a real accumulator:
  //   DstReg = PrevReg | (CurReg & EXEC).
  if (CurIsConstant && !CurVal) {
    // CurReg & EXEC == 0, so PrevReg | 0 == PrevReg.
    buildCopy(DstReg, PrevReg);
    return;
  }

  // Reduce (CurReg & EXEC) to a single OR operand.
  Register CurMasked;
  if (CurIsConstant)
    // CurVal is true here: -1 & EXEC == EXEC.
    CurMasked = LMC.ExecReg;
  else if (LMA && LMA->isSubsetOfExec(CurReg, MBB, I))
    CurMasked = CurReg;
  else {
    CurMasked = createLaneMaskReg();
    buildBinOp(LMC.AndOpc, CurMasked, CurReg, LMC.ExecReg);
  }

  buildBinOp(LMC.OrOpc, DstReg, PrevReg, CurMasked);
}

/// Conservatively determine whether the \p Reg is a subset of EXEC for
/// \p UseBlock, i.e. it returns true if it can statically prove that
/// (Reg & EXEC) == Reg when used in \p UseBlock.
bool AMDGPULaneMaskAnalysis::isSubsetOfExec(Register Reg,
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
void AMDGPULaneMaskUpdater::init() {
  Blocks.clear();

  const SIInstrInfo *TII =
      LMU.function()->getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock &Entry = LMU.function()->front();

  if (!ZeroReg) {
    ZeroReg = LMU.createLaneMaskReg();
    BuildMI(Entry, Entry.getFirstTerminator(), {},
            TII->get(LMU.getLaneMaskConsts().MovOpc), ZeroReg)
        .addImm(0);
  }

  Accumulator = LMU.createLaneMaskReg();
  AllAccumulators.insert(Accumulator);
  BuildMI(Entry, Entry.getFirstTerminator(), {},
          TII->get(LMU.getLaneMaskConsts().MovOpc), Accumulator)
      .addImm(0);
}

/// Optional cleanup, may remove stray instructions.
void AMDGPULaneMaskUpdater::cleanup() {
  Blocks.clear();
  Accumulator = AMDGPU::NoRegister;
  MachineRegisterInfo &MRI = LMU.function()->getRegInfo();

  if (ZeroReg && MRI.use_empty(ZeroReg)) {
    MRI.getVRegDef(ZeroReg)->eraseFromParent();
    ZeroReg = AMDGPU::NoRegister;
  }
}

/// Indicate that a reset should occur in the given block.
///
/// Can be called multiple times for the same block, flags accumulate.
void AMDGPULaneMaskUpdater::addReset(MachineBasicBlock &Block,
                                     ResetFlags Flags) {
  auto BlockIt = findBlockInfo(Block);
  if (BlockIt == Blocks.end()) {
    Blocks.emplace_back(&Block);
    BlockIt = Blocks.end() - 1;
  }

  BlockIt->Flags |= Flags;
  AccumulatorResetBlocks[&Block].push_back({Accumulator, Flags});
}

///
/// \param Value A virtual lane mask register; the lane bits are masked by the
///              block's effective EXEC.
void AMDGPULaneMaskUpdater::addAvailable(MachineBasicBlock &Block,
                                         Register Value) {
  auto BlockIt = findBlockInfo(Block);
  if (BlockIt == Blocks.end()) {
    Blocks.emplace_back(&Block);
    BlockIt = Blocks.end() - 1;
  }
  assert(!BlockIt->Value);

  BlockIt->Value = Value;
  Register Previous;
  if (&Block != &LMU.function()->front() && !(BlockIt->Flags & ResetInMiddle))
    Previous = Accumulator;
  else
    Previous = ZeroReg;
  LMU.buildMergeLaneMasks(Block, getSaluInsertionAtEnd(Block), {}, Accumulator,
                          Previous, Value, LMA, Previous == ZeroReg);
}

/// Return the accumulated lane mask after \p Block's merge: \ref Accumulator,
/// or the zero register for a reset-in-middle block that contributed no value.
Register AMDGPULaneMaskUpdater::getMergedMask(MachineBasicBlock &Block) {
  auto BlockIt = findBlockInfo(Block);
  if (BlockIt != Blocks.end() && !BlockIt->Value &&
      (BlockIt->Flags & ResetInMiddle))
    return ZeroReg;

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

/// Find a block in the \ref Blocks structure.
SmallVectorImpl<AMDGPULaneMaskUpdater::BlockInfo>::iterator
AMDGPULaneMaskUpdater::findBlockInfo(MachineBasicBlock &Block) {
  return llvm::find_if(
      Blocks, [&](const auto &Entry) { return Entry.Block == &Block; });
}

void AMDGPULaneMaskUpdater::insertAccumulatorResets() {
  const SIInstrInfo *TII =
      LMU.function()->getSubtarget<GCNSubtarget>().getInstrInfo();
  for (auto &[B, AccFlagPairs] : AccumulatorResetBlocks) {

    // TODO : We only need to compute EndInsertPt if any of B's AccFlagPairs has
    // ResetAtEnd
    MachineBasicBlock::iterator EndInsertPt;
    EndInsertPt = B->getFirstTerminator();

    // Keep the EXEC-narrowing S_MOV_*_term opcode a terminator, since RA
    // inserts end-of-block spills at the first terminator, and spill code below
    // the narrowing would store only the lanes still active.
    MachineInstr *ExecWrite = nullptr;
    if (EndInsertPt != B->end() &&
        EndInsertPt->getOpcode() == LMU.getLaneMaskConsts().MovTermOpc &&
        EndInsertPt->getOperand(0).getReg() ==
            LMU.getLaneMaskConsts().ExecReg) {
      ExecWrite = &*EndInsertPt;
    }

    for (auto &[Acc, Flags] : AccFlagPairs) {
      if (Flags & ResetInMiddle) {
        // Insert at beginning of basic block for ResetInMiddle
        BuildMI(*B, B->begin(), {}, TII->get(LMU.getLaneMaskConsts().MovOpc),
                Acc)
            .addImm(0);
      }
      if (Flags & ResetAtEnd) {
        // The Acc reset following the EXEC write cannot be a
        // terminator itself, since InlineSpiller cannot spill a vreg defined by
        // a terminator. So, Copy Acc into a temporary for the EXEC write to
        // read, so the Acc reset stays an ordinary instruction ahead of it.
        if (ExecWrite && ExecWrite->getOperand(1).isReg() &&
            ExecWrite->getOperand(1).getReg() == Acc) {
          Register Staged = LMU.createLaneMaskReg();
          BuildMI(*B, EndInsertPt, {}, TII->get(TargetOpcode::COPY), Staged)
              .addReg(Acc);
          ExecWrite->getOperand(1).setReg(Staged);
        }
        // Insert at end of basic block for ResetAtEnd
        BuildMI(*B, EndInsertPt, {}, TII->get(LMU.getLaneMaskConsts().MovOpc),
                Acc)
            .addImm(0);
      }
    }
  }
}
