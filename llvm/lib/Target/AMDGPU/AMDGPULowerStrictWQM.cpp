//===-- AMDGPULowerStrictWQM.cpp - Lower Strict WWM/WQM operations --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass lowers Strict Whole Wave Mode (StrictWWM) and Strict Whole Quad
/// Mode (StrictWQM) operations. It is originally part of the SIWholeQuadMode
/// pass, decoupled to run earlier in the Late Wave Transform (LWT) pipeline
/// before SIPreAllocateWWMRegs.
///
/// This pass handles:
/// - STRICT_WWM and STRICT_WQM pseudo-instruction lowering to V_MOV/COPY
/// - Insertion of ENTER_STRICT_WWM/EXIT_STRICT_WWM and
///   ENTER_STRICT_WQM/EXIT_STRICT_WQM brackets around strict regions
/// - V_SET_INACTIVE_B32 promotion to WWM when used in strict context,
///   or lowering to COPY when not
/// - Patching V_SET_INACTIVE_B32 operand 5 with the saved EXEC register
///   from the enclosing ENTER bracket
/// - LDS_PARAM_LOAD/DS_PARAM_LOAD/LDS_DIRECT_LOAD/DS_DIRECT_LOAD handling
///   (marked as needing StrictWQM)
/// - Dual-source blend EXP source operand handling (marked as StrictWQM)
///
/// It does NOT handle WQM/Exact transitions, kill/demote lowering, init exec,
/// live mask management, or SOFT_WQM — those remain in AMDGPULowerWQMOperations
/// which runs after WaveTransform.
///
//===----------------------------------------------------------------------===//

#include "AMDGPULowerStrictWQM.h"
#include "AMDGPU.h"
#include "AMDGPULaneMaskUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-strict-wqm"

namespace {

enum {
  StateStrictWWM = 0x2,
  StateStrictWQM = 0x4,
  StateStrict = StateStrictWWM | StateStrictWQM,
};

struct PrintState {
public:
  int State;

  explicit PrintState(int State) : State(State) {}
};

#ifndef NDEBUG
static raw_ostream &operator<<(raw_ostream &OS, const PrintState &PS) {
  static const std::pair<char, const char *> Mapping[] = {
      std::pair(StateStrictWWM, "StrictWWM"),
      std::pair(StateStrictWQM, "StrictWQM")};
  char State = PS.State;
  for (auto M : Mapping) {
    if (State & M.first) {
      OS << M.second;
      State &= ~M.first;

      if (State)
        OS << '|';
    }
  }
  return OS;
}
#endif

struct InstrInfo {
  char Needs = 0;
  char Disabled = 0;
  char OutNeeds = 0;
  char MarkedStates = 0;
};

struct BlockInfo {
  char Needs = 0;
  bool NeedsLowering = false;
};

struct WorkItem {
  MachineInstr *MI = nullptr;

  WorkItem() = default;
  WorkItem(MachineInstr *MI) : MI(MI) {}
};

class AMDGPULowerStrictWQM {
public:
  AMDGPULowerStrictWQM(MachineFunction &MF, LiveIntervals *LIS)
      : ST(&MF.getSubtarget<GCNSubtarget>()), TII(ST->getInstrInfo()),
        TRI(&TII->getRegisterInfo()), MRI(&MF.getRegInfo()), LIS(LIS),
        LMC(AMDGPU::LaneMaskConstants::get(*ST)) {}
  bool run(MachineFunction &MF);

private:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  const AMDGPU::LaneMaskConstants &LMC;

  DenseMap<const MachineInstr *, InstrInfo> Instructions;
  MapVector<MachineBasicBlock *, BlockInfo> Blocks;
  DenseMap<const MachineInstr *, char> StateTransition;

  SmallVector<MachineInstr *, 4> LowerToMovInstrs;
  SmallSetVector<MachineInstr *, 4> LowerToCopyInstrs;
  SmallVector<MachineInstr *, 4> SetInactiveInstrs;

  void printInfo();

  void markInstruction(MachineInstr &MI, char Flag,
                       std::vector<WorkItem> &Worklist);
  void markDefs(const MachineInstr &UseMI, LiveRange &LR,
                VirtRegOrUnit VRegOrUnit, unsigned SubReg, char Flag,
                std::vector<WorkItem> &Worklist);
  void markOperand(const MachineInstr &MI, const MachineOperand &Op, char Flag,
                   std::vector<WorkItem> &Worklist);
  void markInstructionUses(const MachineInstr &MI, char Flag,
                           std::vector<WorkItem> &Worklist);
  char scanInstructions(MachineFunction &MF, std::vector<WorkItem> &Worklist);
  void propagateInstruction(MachineInstr &MI, std::vector<WorkItem> &Worklist);

  MachineBasicBlock::iterator saveSCC(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator Before);
  MachineBasicBlock::iterator
  prepareInsertion(MachineBasicBlock &MBB, MachineBasicBlock::iterator First,
                   MachineBasicBlock::iterator Last, bool PreferLast,
                   bool SaveSCC);
  void toStrictMode(MachineBasicBlock &MBB, MachineBasicBlock::iterator Before,
                    Register SaveOrig, char StrictStateNeeded);
  void fromStrictMode(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator Before, Register SavedOrig,
                      char CurrentStrictState);

  void processBlock(MachineBasicBlock &MBB, BlockInfo &BI);
  void lowerBlock(MachineBasicBlock &MBB, BlockInfo &BI);
  bool lowerCopyInstrs();
};

class AMDGPULowerStrictWQMLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULowerStrictWQMLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Lower Strict WQM Operations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char AMDGPULowerStrictWQMLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPULowerStrictWQMLegacy, DEBUG_TYPE,
                      "AMDGPU Lower Strict WQM Operations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPULowerStrictWQMLegacy, DEBUG_TYPE,
                    "AMDGPU Lower Strict WQM Operations", false, false)

char &llvm::AMDGPULowerStrictWQMLegacyID = AMDGPULowerStrictWQMLegacy::ID;

FunctionPass *llvm::createAMDGPULowerStrictWQMLegacyPass() {
  return new AMDGPULowerStrictWQMLegacy;
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void AMDGPULowerStrictWQM::printInfo() {
  for (const auto &BII : Blocks) {
    dbgs() << "\n"
           << printMBBReference(*BII.first) << ":\n"
           << "  Needs = " << PrintState(BII.second.Needs) << "\n\n";

    for (const MachineInstr &MI : *BII.first) {
      auto III = Instructions.find(&MI);
      if (III != Instructions.end()) {
        dbgs() << "  " << MI << "    Needs = " << PrintState(III->second.Needs)
               << '\n';
      }
    }
  }
}
#endif

void AMDGPULowerStrictWQM::markInstruction(MachineInstr &MI, char Flag,
                                           std::vector<WorkItem> &Worklist) {
  InstrInfo &II = Instructions[&MI];

  assert(Flag != 0);

  // Capture all states requested in marking including disabled ones.
  II.MarkedStates |= Flag;

  // Remove any disabled states from the flag.
  Flag &= ~II.Disabled;

  // Ignore if the flag is already encompassed by the existing needs, or we
  // just disabled everything.
  if ((II.Needs & Flag) == Flag)
    return;

  LLVM_DEBUG(dbgs() << "markInstruction " << PrintState(Flag) << ": " << MI);
  II.Needs |= Flag;
  Worklist.emplace_back(&MI);
}

/// Mark all relevant definitions of register in usage.
void AMDGPULowerStrictWQM::markDefs(const MachineInstr &UseMI, LiveRange &LR,
                                    VirtRegOrUnit VRegOrUnit, unsigned SubReg,
                                    char Flag,
                                    std::vector<WorkItem> &Worklist) {
  LLVM_DEBUG(dbgs() << "markDefs " << PrintState(Flag) << ": " << UseMI);

  LiveQueryResult UseLRQ = LR.Query(LIS->getInstructionIndex(UseMI));
  const VNInfo *Value = UseLRQ.valueIn();
  if (!Value)
    return;

  const LaneBitmask UseLanes =
      SubReg ? TRI->getSubRegIndexLaneMask(SubReg)
             : (VRegOrUnit.isVirtualReg()
                    ? MRI->getMaxLaneMaskForVReg(VRegOrUnit.asVirtualReg())
                    : LaneBitmask::getNone());

  struct PhiEntry {
    const VNInfo *Phi;
    unsigned PredIdx;
    LaneBitmask DefinedLanes;

    PhiEntry(const VNInfo *Phi, unsigned PredIdx, LaneBitmask DefinedLanes)
        : Phi(Phi), PredIdx(PredIdx), DefinedLanes(DefinedLanes) {}
  };
  using VisitKey = std::pair<const VNInfo *, LaneBitmask>;
  SmallVector<PhiEntry, 2> PhiStack;
  SmallSet<VisitKey, 4> Visited;
  LaneBitmask DefinedLanes;
  unsigned NextPredIdx = 0;
  do {
    const VNInfo *NextValue = nullptr;
    const VisitKey Key(Value, DefinedLanes);

    if (Visited.insert(Key).second) {
      NextPredIdx = 0;
    }

    if (Value->isPHIDef()) {
      const MachineBasicBlock *MBB = LIS->getMBBFromIndex(Value->def);
      assert(MBB && "Phi-def has no defining MBB");

      unsigned Idx = NextPredIdx;
      const auto *PI = MBB->pred_begin() + Idx;
      const auto *PE = MBB->pred_end();
      for (; PI != PE && !NextValue; ++PI, ++Idx) {
        if (const VNInfo *VN = LR.getVNInfoBefore(LIS->getMBBEndIdx(*PI))) {
          if (!Visited.count(VisitKey(VN, DefinedLanes)))
            NextValue = VN;
        }
      }

      if (PI != PE)
        PhiStack.emplace_back(Value, Idx, DefinedLanes);
    } else {
      MachineInstr *MI = LIS->getInstructionFromIndex(Value->def);
      assert(MI && "Def has no defining instruction");

      if (VRegOrUnit.isVirtualReg()) {
        bool HasDef = false;
        for (const MachineOperand &Op : MI->all_defs()) {
          if (Op.getReg() != VRegOrUnit.asVirtualReg())
            continue;

          LaneBitmask OpLanes =
              Op.isUndef() ? LaneBitmask::getAll()
                           : TRI->getSubRegIndexLaneMask(Op.getSubReg());
          LaneBitmask Overlap = (UseLanes & OpLanes);

          HasDef |= Overlap.any();
          DefinedLanes |= OpLanes;
        }

        if ((DefinedLanes & UseLanes) != UseLanes) {
          LiveQueryResult LRQ = LR.Query(LIS->getInstructionIndex(*MI));
          if (const VNInfo *VN = LRQ.valueIn()) {
            if (!Visited.count(VisitKey(VN, DefinedLanes)))
              NextValue = VN;
          }
        }

        if (HasDef)
          markInstruction(*MI, Flag, Worklist);
      } else {
        markInstruction(*MI, Flag, Worklist);
      }
    }

    if (!NextValue && !PhiStack.empty()) {
      PhiEntry &Entry = PhiStack.back();
      NextValue = Entry.Phi;
      NextPredIdx = Entry.PredIdx;
      DefinedLanes = Entry.DefinedLanes;
      PhiStack.pop_back();
    }

    Value = NextValue;
  } while (Value);
}

void AMDGPULowerStrictWQM::markOperand(const MachineInstr &MI,
                                       const MachineOperand &Op, char Flag,
                                       std::vector<WorkItem> &Worklist) {
  assert(Op.isReg());
  Register Reg = Op.getReg();

  switch (Reg) {
  case AMDGPU::EXEC:
  case AMDGPU::EXEC_LO:
    return;
  default:
    break;
  }

  LLVM_DEBUG(dbgs() << "markOperand " << PrintState(Flag) << ": " << Op
                    << " for " << MI);
  if (Reg.isVirtual()) {
    LiveRange &LR = LIS->getInterval(Reg);
    markDefs(MI, LR, VirtRegOrUnit(Reg), Op.getSubReg(), Flag, Worklist);
  } else {
    for (MCRegUnit Unit : TRI->regunits(Reg.asMCReg())) {
      LiveRange &LR = LIS->getRegUnit(Unit);
      const VNInfo *Value = LR.Query(LIS->getInstructionIndex(MI)).valueIn();
      if (Value)
        markDefs(MI, LR, VirtRegOrUnit(Unit), AMDGPU::NoSubRegister, Flag,
                 Worklist);
    }
  }
}

void AMDGPULowerStrictWQM::markInstructionUses(
    const MachineInstr &MI, char Flag, std::vector<WorkItem> &Worklist) {
  LLVM_DEBUG(dbgs() << "markInstructionUses " << PrintState(Flag) << ": "
                    << MI);

  for (const MachineOperand &Use : MI.all_uses())
    markOperand(MI, Use, Flag, Worklist);
}

char AMDGPULowerStrictWQM::scanInstructions(MachineFunction &MF,
                                            std::vector<WorkItem> &Worklist) {
  char GlobalFlags = 0;

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT) {
    BlockInfo &BBI = Blocks[MBB];

    for (MachineInstr &MI : *MBB) {
      InstrInfo &III = Instructions[&MI];
      unsigned Opcode = MI.getOpcode();

      if (Opcode == AMDGPU::STRICT_WWM) {
        markInstructionUses(MI, StateStrictWWM, Worklist);
        GlobalFlags |= StateStrictWWM;
        LowerToMovInstrs.push_back(&MI);
      } else if (Opcode == AMDGPU::STRICT_WQM ||
                 TII->isDualSourceBlendEXP(MI)) {
        markInstructionUses(MI, StateStrictWQM, Worklist);
        GlobalFlags |= StateStrictWQM;

        if (Opcode == AMDGPU::STRICT_WQM) {
          LowerToMovInstrs.push_back(&MI);
        } else {
          // Dual source blend export: sources need StrictWQM,
          // but the export itself should not execute in strict mode.
          III.Disabled = StateStrict;
        }
      } else if (Opcode == AMDGPU::LDS_PARAM_LOAD ||
                 Opcode == AMDGPU::DS_PARAM_LOAD ||
                 Opcode == AMDGPU::LDS_DIRECT_LOAD ||
                 Opcode == AMDGPU::DS_DIRECT_LOAD) {
        III.Needs |= StateStrictWQM;
        BBI.Needs |= StateStrictWQM;
        GlobalFlags |= StateStrictWQM;
      } else if (Opcode == AMDGPU::V_SET_INACTIVE_B32) {
        III.Disabled = StateStrict;
        MachineOperand &Inactive = MI.getOperand(4);
        if (Inactive.isReg()) {
          if (Inactive.isUndef() && MI.getOperand(3).getImm() == 0)
            LowerToCopyInstrs.insert(&MI);
          else
            markOperand(MI, Inactive, StateStrictWWM, Worklist);
        }
        SetInactiveInstrs.push_back(&MI);
        BBI.NeedsLowering = true;
      }
    }
  }

  return GlobalFlags;
}

void AMDGPULowerStrictWQM::propagateInstruction(
    MachineInstr &MI, std::vector<WorkItem> &Worklist) {
  MachineBasicBlock *MBB = MI.getParent();
  InstrInfo II =
      Instructions[&MI]; // take a copy to prevent dangling references
  BlockInfo &BI = Blocks[MBB];

  // Propagate to block level
  if (II.Needs & StateStrictWWM)
    BI.Needs |= StateStrictWWM;
  if (II.Needs & StateStrictWQM)
    BI.Needs |= StateStrictWQM;

  // Propagate strict needs backward through def-use chains
  if (II.Needs != 0)
    markInstructionUses(MI, II.Needs, Worklist);
}

MachineBasicBlock::iterator
AMDGPULowerStrictWQM::saveSCC(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator Before) {
  Register SaveReg = MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);

  MachineInstr *Save =
      BuildMI(MBB, Before, DebugLoc(), TII->get(AMDGPU::COPY), SaveReg)
          .addReg(AMDGPU::SCC);
  MachineInstr *Restore =
      BuildMI(MBB, Before, DebugLoc(), TII->get(AMDGPU::COPY), AMDGPU::SCC)
          .addReg(SaveReg);

  LIS->InsertMachineInstrInMaps(*Save);
  LIS->InsertMachineInstrInMaps(*Restore);
  LIS->createAndComputeVirtRegInterval(SaveReg);

  return Restore;
}

MachineBasicBlock::iterator AMDGPULowerStrictWQM::prepareInsertion(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator First,
    MachineBasicBlock::iterator Last, bool PreferLast, bool SaveSCC) {
  if (!SaveSCC)
    return PreferLast ? Last : First;

  LiveRange &LR =
      LIS->getRegUnit(*TRI->regunits(MCRegister::from(AMDGPU::SCC)).begin());
  auto MBBE = MBB.end();
  auto FirstNonDbg = skipDebugInstructionsForward(First, MBBE);
  auto LastNonDbg = skipDebugInstructionsForward(Last, MBBE);
  SlotIndex FirstIdx = FirstNonDbg != MBBE
                           ? LIS->getInstructionIndex(*FirstNonDbg)
                           : LIS->getMBBEndIdx(&MBB);
  SlotIndex LastIdx = LastNonDbg != MBBE ? LIS->getInstructionIndex(*LastNonDbg)
                                         : LIS->getMBBEndIdx(&MBB);
  SlotIndex Idx = PreferLast ? LastIdx : FirstIdx;
  const LiveRange::Segment *S;

  for (;;) {
    S = LR.getSegmentContaining(Idx);
    if (!S)
      break;

    if (PreferLast) {
      SlotIndex Next = S->start.getBaseIndex();
      if (Next < FirstIdx)
        break;
      Idx = Next;
    } else {
      MachineInstr *EndMI = LIS->getInstructionFromIndex(S->end.getBaseIndex());
      assert(EndMI && "Segment does not end on valid instruction");
      auto NextI = next_nodbg(EndMI->getIterator(), MBB.instr_end());
      if (NextI == MBB.instr_end())
        break;
      SlotIndex Next = LIS->getInstructionIndex(*NextI);
      if (Next > LastIdx)
        break;
      Idx = Next;
    }
  }

  MachineBasicBlock::iterator MBBI;

  if (MachineInstr *MI = LIS->getInstructionFromIndex(Idx))
    MBBI = MI;
  else {
    assert(Idx == LIS->getMBBEndIdx(&MBB));
    MBBI = MBB.end();
  }

  // Move insertion point past any operations modifying EXEC.
  while (MBBI != Last) {
    bool IsExecDef = false;
    for (const MachineOperand &MO : MBBI->all_defs()) {
      IsExecDef |=
          MO.getReg() == AMDGPU::EXEC_LO || MO.getReg() == AMDGPU::EXEC;
    }
    if (!IsExecDef)
      break;
    MBBI++;
    S = nullptr;
  }

  if (S)
    MBBI = saveSCC(MBB, MBBI);

  return MBBI;
}

void AMDGPULowerStrictWQM::toStrictMode(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator Before,
                                        Register SaveOrig,
                                        char StrictStateNeeded) {
  MachineInstr *MI;
  assert(SaveOrig);
  assert(StrictStateNeeded == StateStrictWWM ||
         StrictStateNeeded == StateStrictWQM);

  const DebugLoc &DL = MBB.findDebugLoc(Before);

  if (StrictStateNeeded == StateStrictWWM) {
    MI = BuildMI(MBB, Before, DL, TII->get(AMDGPU::ENTER_STRICT_WWM), SaveOrig)
             .addImm(-1);
  } else {
    MI = BuildMI(MBB, Before, DL, TII->get(AMDGPU::ENTER_STRICT_WQM), SaveOrig)
             .addImm(-1);
  }
  LIS->InsertMachineInstrInMaps(*MI);
  StateTransition[MI] = StrictStateNeeded;
}

void AMDGPULowerStrictWQM::fromStrictMode(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator Before,
                                          Register SavedOrig,
                                          char CurrentStrictState) {
  MachineInstr *MI;

  assert(SavedOrig);
  assert(CurrentStrictState == StateStrictWWM ||
         CurrentStrictState == StateStrictWQM);

  const DebugLoc &DL = MBB.findDebugLoc(Before);

  if (CurrentStrictState == StateStrictWWM) {
    MI =
        BuildMI(MBB, Before, DL, TII->get(AMDGPU::EXIT_STRICT_WWM), LMC.ExecReg)
            .addReg(SavedOrig);
  } else {
    MI =
        BuildMI(MBB, Before, DL, TII->get(AMDGPU::EXIT_STRICT_WQM), LMC.ExecReg)
            .addReg(SavedOrig);
  }
  LIS->InsertMachineInstrInMaps(*MI);
  StateTransition[MI] = 0;
}

void AMDGPULowerStrictWQM::processBlock(MachineBasicBlock &MBB, BlockInfo &BI) {
  if (!(BI.Needs & StateStrict))
    return;

  LLVM_DEBUG(dbgs() << "\nProcessing block " << printMBBReference(MBB)
                    << ":\n");

  Register SavedNonStrictReg;
  char State = 0; // not in strict mode
  const TargetRegisterClass *BoolRC = TRI->getBoolRC();

  auto II = MBB.getFirstNonPHI(), IE = MBB.end();

  // This stores the first instruction where it's safe to switch to/from
  // strict mode.
  MachineBasicBlock::iterator FirstStrict = IE;

  for (;;) {
    MachineBasicBlock::iterator Next = II;
    char Needs = 0; // default: doesn't need strict
    bool IsExecIndependent = false;

    if (FirstStrict == IE)
      FirstStrict = II;

    if (II != IE) {
      MachineInstr &MI = *II;

      if (MI.isTerminator() || TII->mayReadEXEC(*MRI, MI)) {
        auto III = Instructions.find(&MI);
        if (III != Instructions.end()) {
          if (III->second.Needs & StateStrictWWM)
            Needs = StateStrictWWM;
          else if (III->second.Needs & StateStrictWQM)
            Needs = StateStrictWQM;
          else
            Needs &= ~III->second.Disabled;
        }
      } else {
        // If the instruction doesn't read EXEC (e.g. SGPR-only operations,
        // meta instructions), it can safely execute in any mode. So keeping the
        // current state to avoid exiting and immediately re-entering.
        Needs = State;
        IsExecIndependent = true;
      }

      ++Next;
    }

    // Transition if the current state doesn't satisfy what's needed and at
    // least one side involves strict mode. The second condition prevents
    // false positives when both Needs and State are 0 (non-strict instruction
    // outside any strict region), which would otherwise trigger
    // prepareInsertion and potentially insert dead SCC save/restore pairs.
    if (!(Needs & State) && ((State | Needs) & StateStrict)) {
      MachineBasicBlock::iterator Before =
          prepareInsertion(MBB, FirstStrict, II, false, true);

      if (State & StateStrict) {
        fromStrictMode(MBB, Before, SavedNonStrictReg, State);
        LIS->createAndComputeVirtRegInterval(SavedNonStrictReg);
        SavedNonStrictReg = Register();
        State = 0;
      }

      if (Needs & StateStrict) {
        SavedNonStrictReg = MRI->createVirtualRegister(BoolRC);
        toStrictMode(MBB, Before, SavedNonStrictReg, Needs);
        State = Needs;
      }
    }

    // Reset FirstStrict for EXEC-dependent instructions to prevent
    // accumulation past instructions that are affected by EXEC changes.
    // Preserve FirstStrict for EXEC-independent instructions (SGPR-only,
    // meta) to give prepareInsertion a wider [FirstStrict, II] range for
    // finding SCC-dead insertion points, matching SIWholeQuadMode's design.
    if (!IsExecIndependent)
      FirstStrict = IE;

    if (II == IE)
      break;

    II = Next;
  }
  assert(!SavedNonStrictReg && "Strict mode not closed at end of block");
}

void AMDGPULowerStrictWQM::lowerBlock(MachineBasicBlock &MBB, BlockInfo &BI) {
  if (!BI.NeedsLowering)
    return;

  LLVM_DEBUG(dbgs() << "\nLowering block " << printMBBReference(MBB) << ":\n");

  Register ActiveLanesReg;

  for (MachineInstr &MI : llvm::make_early_inc_range(
           llvm::make_range(MBB.getFirstNonPHI(), MBB.end()))) {
    switch (MI.getOpcode()) {
    case AMDGPU::ENTER_STRICT_WWM:
      ActiveLanesReg = MI.getOperand(0).getReg();
      break;
    case AMDGPU::EXIT_STRICT_WWM:
      ActiveLanesReg = Register();
      break;
    case AMDGPU::V_SET_INACTIVE_B32:
      if (ActiveLanesReg) {
        LiveInterval &LI = LIS->getInterval(MI.getOperand(5).getReg());
        MRI->constrainRegClass(ActiveLanesReg, TRI->getWaveMaskRegClass());
        MI.getOperand(5).setReg(ActiveLanesReg);
        LIS->shrinkToUses(&LI);
      }
      // else: not inside bracket → will be lowered to COPY later
      break;
    default:
      break;
    }
  }
}

bool AMDGPULowerStrictWQM::lowerCopyInstrs() {
  for (MachineInstr *MI : LowerToMovInstrs) {
    assert(MI->getNumExplicitOperands() == 2);

    const Register Reg = MI->getOperand(0).getReg();

    const TargetRegisterClass *regClass =
        TRI->getRegClassForOperandReg(*MRI, MI->getOperand(0));
    if (TRI->isVGPRClass(regClass)) {
      const unsigned MovOp = TII->getMovOpcode(regClass);
      MI->setDesc(TII->get(MovOp));

      assert(any_of(MI->implicit_operands(), [](const MachineOperand &MO) {
        return MO.isUse() && MO.getReg() == AMDGPU::EXEC;
      }));
    } else {
      // Remove early-clobber and exec dependency from simple SGPR copies.
      LLVM_DEBUG(dbgs() << "simplify SGPR copy: " << *MI);
      if (MI->getOperand(0).isEarlyClobber()) {
        LIS->removeInterval(Reg);
        MI->getOperand(0).setIsEarlyClobber(false);
        LIS->createAndComputeVirtRegInterval(Reg);
      }
      int Index = MI->findRegisterUseOperandIdx(AMDGPU::EXEC, /*TRI=*/nullptr);
      while (Index >= 0) {
        MI->removeOperand(Index);
        Index = MI->findRegisterUseOperandIdx(AMDGPU::EXEC, /*TRI=*/nullptr);
      }
      MI->setDesc(TII->get(AMDGPU::COPY));
      LLVM_DEBUG(dbgs() << "  -> " << *MI);
    }
  }
  for (MachineInstr *MI : LowerToCopyInstrs) {
    LLVM_DEBUG(dbgs() << "simplify: " << *MI);

    if (MI->getOpcode() == AMDGPU::V_SET_INACTIVE_B32) {
      assert(MI->getNumExplicitOperands() == 6);

      LiveInterval *RecomputeLI = nullptr;
      Register RecomputePhysReg;
      if (MI->getOperand(4).isReg()) {
        Register Reg = MI->getOperand(4).getReg();
        if (Reg.isVirtual())
          RecomputeLI = &LIS->getInterval(Reg);
        else
          RecomputePhysReg = Reg;
      }

      MI->removeOperand(5);
      MI->removeOperand(4);
      MI->removeOperand(3);
      MI->removeOperand(1);

      if (RecomputeLI)
        LIS->shrinkToUses(RecomputeLI);
      else if (RecomputePhysReg) {
        for (MCRegUnit Unit : TRI->regunits(RecomputePhysReg)) {
          LIS->removeRegUnit(Unit);
          LIS->getRegUnit(Unit);
        }
      }
    } else {
      assert(MI->getNumExplicitOperands() == 2);
    }

    unsigned CopyOp = MI->getOperand(1).isReg()
                          ? (unsigned)AMDGPU::COPY
                          : TII->getMovOpcode(TRI->getRegClassForOperandReg(
                                *MRI, MI->getOperand(0)));
    MI->setDesc(TII->get(CopyOp));
    LLVM_DEBUG(dbgs() << " -> " << *MI);
  }
  return !LowerToCopyInstrs.empty() || !LowerToMovInstrs.empty();
}

bool AMDGPULowerStrictWQM::run(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "AMDGPU Lower Strict WQM on " << MF.getName()
                    << " ------------- \n");
  LLVM_DEBUG(MF.dump(););

  Instructions.clear();
  Blocks.clear();
  LowerToMovInstrs.clear();
  LowerToCopyInstrs.clear();
  SetInactiveInstrs.clear();
  StateTransition.clear();

  // Phase 1: Analysis (strict-only)
  std::vector<WorkItem> Worklist;
  char GlobalFlags = scanInstructions(MF, Worklist);

  if (!(GlobalFlags & StateStrict)) {
    // No strict operations found. V_SET_INACTIVE_B32 with undef inactive
    // and flag=0 still need to be lowered to COPY. Any remaining
    // SetInactiveInstrs without strict marking also get lowered.
    for (MachineInstr *MI : SetInactiveInstrs) {
      if (!LowerToCopyInstrs.contains(MI))
        LowerToCopyInstrs.insert(MI);
    }
    if (!LowerToCopyInstrs.empty())
      return lowerCopyInstrs();
    return false;
  }

  // Worklist propagation (instruction-level only, no block-level)
  while (!Worklist.empty()) {
    WorkItem WI = Worklist.back();
    Worklist.pop_back();
    assert(WI.MI);
    propagateInstruction(*WI.MI, Worklist);
  }

  // V_SET_INACTIVE promotion: if touched by strict marking, promote to WWM;
  // otherwise lower to COPY.
  for (MachineInstr *MI : SetInactiveInstrs) {
    if (LowerToCopyInstrs.contains(MI))
      continue;
    auto &Info = Instructions[MI];
    if (Info.MarkedStates & StateStrict) {
      Info.Needs |= StateStrictWWM;
      Info.Disabled &= ~StateStrictWWM;
      Blocks[MI->getParent()].Needs |= StateStrictWWM;
    } else {
      LLVM_DEBUG(dbgs() << "Has no WWM marking: " << *MI);
      LowerToCopyInstrs.insert(MI);
    }
  }

  LLVM_DEBUG(printInfo());

  // Phase 2: Transformation
  for (auto &BII : Blocks)
    processBlock(*BII.first, BII.second);
  for (auto &BII : Blocks)
    lowerBlock(*BII.first, BII.second);
  lowerCopyInstrs();

  // Clean up
  LIS->removeAllRegUnitsForPhysReg(AMDGPU::SCC);

  return true;
}

bool AMDGPULowerStrictWQMLegacy::runOnMachineFunction(MachineFunction &MF) {
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  AMDGPULowerStrictWQM Impl(MF, LIS);
  return Impl.run(MF);
}

PreservedAnalyses
AMDGPULowerStrictWQMPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  AMDGPULowerStrictWQM Impl(MF, LIS);
  bool Changed = Impl.run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<SlotIndexesAnalysis>();
  PA.preserve<LiveIntervalsAnalysis>();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
