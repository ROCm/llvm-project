//===-- AMDGPULowerWQMOperations.cpp - Lower WQM/Exact operations
//----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass handles WQM (Whole Quad Mode) and Exact mode transitions for
/// pixel shaders. It is originally part of the SIWholeQuadMode pass, decoupled
/// to run later in the Late Wave Transform (LWT) pipeline after WaveTransform.
///
/// This pass handles:
/// - WQM and SOFT_WQM pseudo-instruction lowering to COPY
/// - WQM-to-Exact and Exact-to-WQM transitions via EXEC mask manipulation
/// - LiveMask register creation and management
/// - Kill/demote instruction lowering (SI_KILL_*, SI_DEMOTE_I1)
/// - Init exec instruction lowering (SI_INIT_EXEC, SI_INIT_EXEC_FROM_INPUT,
///   SI_INIT_WHOLE_WAVE)
/// - Live mask queries (SI_PS_LIVE, SI_LIVE_MASK)
///
/// It does NOT handle Strict WWM/WQM operations (STRICT_WWM, STRICT_WQM,
/// V_SET_INACTIVE_B32, LDS_PARAM_LOAD, etc.) — those are handled by
/// AMDGPULowerStrictWQM which runs earlier, before SIPreAllocateWWMRegs.
///
/// This pass treats pre-existing ENTER_STRICT_WWM/WQM and EXIT_STRICT_WWM/WQM
/// brackets as opaque regions: no WQM/Exact transition decisions are made
/// inside these brackets, and the WQM/Exact state is restored after EXIT.
///
//===----------------------------------------------------------------------===//

#include "AMDGPULowerWQMOperations.h"
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
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-wqm-operations"

namespace {

enum {
  StateWQM = 0x1,
  StateStrictWWM = 0x2,
  StateStrictWQM = 0x4,
  StateExact = 0x8,
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
      std::pair(StateWQM, "WQM"), std::pair(StateStrictWWM, "StrictWWM"),
      std::pair(StateStrictWQM, "StrictWQM"), std::pair(StateExact, "Exact")};
  char State = PS.State;
  for (auto M : Mapping) {
    if (State & M.first) {
      OS << M.second;
      State &= ~M.first;

      if (State)
        OS << '|';
    }
  }
  assert(State == 0);
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
  char InNeeds = 0;
  char OutNeeds = 0;
  char InitialState = 0;
  bool NeedsLowering = false;
};

struct WorkItem {
  MachineBasicBlock *MBB = nullptr;
  MachineInstr *MI = nullptr;

  WorkItem() = default;
  WorkItem(MachineBasicBlock *MBB) : MBB(MBB) {}
  WorkItem(MachineInstr *MI) : MI(MI) {}
};

class AMDGPULowerWQMOperations {
public:
  AMDGPULowerWQMOperations(MachineFunction &MF, LiveIntervals *LIS,
                           MachineDominatorTree *MDT,
                           MachinePostDominatorTree *PDT)
      : ST(&MF.getSubtarget<GCNSubtarget>()), TII(ST->getInstrInfo()),
        TRI(&TII->getRegisterInfo()), MRI(&MF.getRegInfo()), LIS(LIS), MDT(MDT),
        PDT(PDT), LMC(AMDGPU::LaneMaskConstants::get(*ST)) {}
  bool run(MachineFunction &MF);

private:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  MachineDominatorTree *MDT;
  MachinePostDominatorTree *PDT;
  const AMDGPU::LaneMaskConstants &LMC;

  Register LiveMaskReg;

  DenseMap<const MachineInstr *, InstrInfo> Instructions;
  MapVector<MachineBasicBlock *, BlockInfo> Blocks;

  // Tracks state (WQM/Exact) after a given instruction
  DenseMap<const MachineInstr *, char> StateTransition;

  SmallVector<MachineInstr *, 2> LiveMaskQueries;
  SmallSetVector<MachineInstr *, 4> LowerToCopyInstrs;
  SmallVector<MachineInstr *, 4> KillInstrs;
  SmallVector<MachineInstr *, 4> InitExecInstrs;

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
  char scanInstructions(MachineFunction &MF, std::vector<WorkItem> &Worklist,
                        SmallVector<MachineInstr *> &ExeczSideEffectInstrs);
  void propagateInstruction(MachineInstr &MI, std::vector<WorkItem> &Worklist);
  void propagateBlock(MachineBasicBlock &MBB, std::vector<WorkItem> &Worklist);
  char analyzeFunction(MachineFunction &MF);

  MachineBasicBlock::iterator saveSCC(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator Before);
  MachineBasicBlock::iterator
  prepareInsertion(MachineBasicBlock &MBB, MachineBasicBlock::iterator First,
                   MachineBasicBlock::iterator Last, bool PreferLast,
                   bool SaveSCC);
  void toExact(MachineBasicBlock &MBB, MachineBasicBlock::iterator Before,
               Register SaveWQM);
  void toWQM(MachineBasicBlock &MBB, MachineBasicBlock::iterator Before,
             Register SavedWQM);

  void splitBlock(MachineInstr *TermMI);
  MachineInstr *lowerKillI1(MachineInstr &MI, bool IsWQM);
  MachineInstr *lowerKillF32(MachineInstr &MI);

  void lowerBlock(MachineBasicBlock &MBB, BlockInfo &BI);
  void processBlock(MachineBasicBlock &MBB, BlockInfo &BI, bool IsEntry);

  bool lowerLiveMaskQueries();
  bool lowerCopyInstrs();
  bool lowerKillInstrs(bool IsWQM);
  void lowerInitExec(MachineInstr &MI);
  MachineBasicBlock::iterator lowerInitExecInstrs(MachineBasicBlock &Entry,
                                                  bool &Changed);
};

class AMDGPULowerWQMOperationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULowerWQMOperationsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Lower WQM Operations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachinePostDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};
} // end anonymous namespace

char AMDGPULowerWQMOperationsLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPULowerWQMOperationsLegacy, DEBUG_TYPE,
                      "AMDGPU Lower WQM Operations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPULowerWQMOperationsLegacy, DEBUG_TYPE,
                    "AMDGPU Lower WQM Operations", false, false)

char &llvm::AMDGPULowerWQMOperationsLegacyID =
    AMDGPULowerWQMOperationsLegacy::ID;

FunctionPass *llvm::createAMDGPULowerWQMOperationsLegacyPass() {
  return new AMDGPULowerWQMOperationsLegacy;
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void AMDGPULowerWQMOperations::printInfo() {
  for (const auto &BII : Blocks) {
    dbgs() << "\n"
           << printMBBReference(*BII.first) << ":\n"
           << "  InNeeds = " << PrintState(BII.second.InNeeds)
           << ", Needs = " << PrintState(BII.second.Needs)
           << ", OutNeeds = " << PrintState(BII.second.OutNeeds) << "\n\n";

    for (const MachineInstr &MI : *BII.first) {
      auto III = Instructions.find(&MI);
      if (III != Instructions.end()) {
        dbgs() << "  " << MI << "    Needs = " << PrintState(III->second.Needs)
               << ", OutNeeds = " << PrintState(III->second.OutNeeds) << '\n';
      }
    }
  }
}
#endif

void AMDGPULowerWQMOperations::markInstruction(
    MachineInstr &MI, char Flag, std::vector<WorkItem> &Worklist) {
  InstrInfo &II = Instructions[&MI];

  assert(!(Flag & StateExact) && Flag != 0);

  II.MarkedStates |= Flag;
  Flag &= ~II.Disabled;

  if ((II.Needs & Flag) == Flag)
    return;

  LLVM_DEBUG(dbgs() << "markInstruction " << PrintState(Flag) << ": " << MI);
  II.Needs |= Flag;
  Worklist.emplace_back(&MI);
}

void AMDGPULowerWQMOperations::markDefs(const MachineInstr &UseMI,
                                        LiveRange &LR, VirtRegOrUnit VRegOrUnit,
                                        unsigned SubReg, char Flag,
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

void AMDGPULowerWQMOperations::markOperand(const MachineInstr &MI,
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

void AMDGPULowerWQMOperations::markInstructionUses(
    const MachineInstr &MI, char Flag, std::vector<WorkItem> &Worklist) {
  LLVM_DEBUG(dbgs() << "markInstructionUses " << PrintState(Flag) << ": "
                    << MI);

  for (const MachineOperand &Use : MI.all_uses())
    markOperand(MI, Use, Flag, Worklist);
}

char AMDGPULowerWQMOperations::scanInstructions(
    MachineFunction &MF, std::vector<WorkItem> &Worklist,
    SmallVector<MachineInstr *> &ExeczSideEffectInstrs) {
  char GlobalFlags = 0;
  bool WQMOutputs = MF.getFunction().hasFnAttribute("amdgpu-ps-wqm-outputs");
  SmallVector<MachineInstr *, 4> SoftWQMInstrs;
  bool HasImplicitDerivatives =
      MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS;

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT) {
    BlockInfo &BBI = Blocks[MBB];

    for (MachineInstr &MI : *MBB) {
      InstrInfo &III = Instructions[&MI];
      unsigned Opcode = MI.getOpcode();
      char Flags = 0;

      if (TII->isWQM(Opcode)) {
        if (ST->hasExtendedImageInsts() && HasImplicitDerivatives) {
          markInstructionUses(MI, StateWQM, Worklist);
          GlobalFlags |= StateWQM;
        }
      } else if (Opcode == AMDGPU::WQM) {
        Flags = StateWQM;
        LowerToCopyInstrs.insert(&MI);
      } else if (Opcode == AMDGPU::SOFT_WQM) {
        LowerToCopyInstrs.insert(&MI);
        SoftWQMInstrs.push_back(&MI);
      } else if (TII->isDualSourceBlendEXP(MI)) {
        // Sources already handled by AMDGPULowerStrictWQM (in ENTER/EXIT
        // brackets). The export itself needs Exact.
        BBI.Needs |= StateExact;
        if (!(BBI.InNeeds & StateExact)) {
          BBI.InNeeds |= StateExact;
          Worklist.emplace_back(MBB);
        }
        GlobalFlags |= StateExact;
        III.Disabled = StateWQM | StateStrict;
      } else if (TII->isDisableWQM(MI)) {
        BBI.Needs |= StateExact;
        if (!(BBI.InNeeds & StateExact)) {
          BBI.InNeeds |= StateExact;
          Worklist.emplace_back(MBB);
        }
        GlobalFlags |= StateExact;
        III.Disabled = StateWQM | StateStrict;
      } else if (Opcode == AMDGPU::SI_PS_LIVE ||
                 Opcode == AMDGPU::SI_LIVE_MASK) {
        LiveMaskQueries.push_back(&MI);
      } else if (Opcode == AMDGPU::SI_KILL_I1_TERMINATOR ||
                 Opcode == AMDGPU::SI_KILL_F32_COND_IMM_TERMINATOR ||
                 Opcode == AMDGPU::SI_DEMOTE_I1) {
        KillInstrs.push_back(&MI);
        BBI.NeedsLowering = true;
      } else if (Opcode == AMDGPU::SI_INIT_EXEC ||
                 Opcode == AMDGPU::SI_INIT_EXEC_FROM_INPUT ||
                 Opcode == AMDGPU::SI_INIT_WHOLE_WAVE) {
        InitExecInstrs.push_back(&MI);
      } else if (WQMOutputs) {
        for (const MachineOperand &MO : MI.defs()) {
          Register Reg = MO.getReg();
          if (Reg.isPhysical() &&
              TRI->hasVectorRegisters(TRI->getPhysRegBaseClass(Reg))) {
            Flags = StateWQM;
            break;
          }
        }
      }

      // STRICT_WWM, STRICT_WQM, V_SET_INACTIVE_B32, LDS_PARAM_LOAD etc.
      // are already lowered by AMDGPULowerStrictWQM — we skip them here.

      if (TII->hasUnwantedEffectsWhenEXECEmpty(MI)) {
        for (auto &Op : MI.uses()) {
          if (!Op.isReg())
            continue;
          if (!TRI->isVectorRegister(*MRI, Op.getReg()))
            continue;

          ExeczSideEffectInstrs.push_back(&MI);
          break;
        }
      }

      if (Flags) {
        markInstruction(MI, Flags, Worklist);
        GlobalFlags |= Flags;
      }
    }
  }

  // SOFT_WQM marking
  if (GlobalFlags & StateWQM) {
    for (MachineInstr *MI : SoftWQMInstrs)
      markInstruction(*MI, StateWQM, Worklist);
  }

  return GlobalFlags;
}

void AMDGPULowerWQMOperations::propagateInstruction(
    MachineInstr &MI, std::vector<WorkItem> &Worklist) {
  MachineBasicBlock *MBB = MI.getParent();
  InstrInfo II =
      Instructions[&MI]; // take a copy to prevent dangling references
  BlockInfo &BI = Blocks[MBB];

  if ((II.OutNeeds & StateWQM) && !(II.Disabled & StateWQM) &&
      (MI.isTerminator() || (TII->usesVM_CNT(MI) && MI.mayStore()))) {
    Instructions[&MI].Needs = StateWQM;
    II.Needs = StateWQM;
  }

  if (II.Needs & StateWQM) {
    BI.Needs |= StateWQM;
    if (!(BI.InNeeds & StateWQM)) {
      BI.InNeeds |= StateWQM;
      Worklist.emplace_back(MBB);
    }
  }

  if (MachineInstr *PrevMI = MI.getPrevNode()) {
    char InNeeds = (II.Needs & ~StateStrict) | II.OutNeeds;
    if (!PrevMI->isPHI()) {
      InstrInfo &PrevII = Instructions[PrevMI];
      if ((PrevII.OutNeeds | InNeeds) != PrevII.OutNeeds) {
        PrevII.OutNeeds |= InNeeds;
        Worklist.emplace_back(PrevMI);
      }
    }
  }

  assert(!(II.Needs & StateExact));

  if (II.Needs != 0)
    markInstructionUses(MI, II.Needs, Worklist);
}

void AMDGPULowerWQMOperations::propagateBlock(MachineBasicBlock &MBB,
                                              std::vector<WorkItem> &Worklist) {
  // Pass-through WQM propagation fix for wave-transform merge blocks.
  if (Blocks[&MBB].Needs == 0 && (Blocks[&MBB].InNeeds & StateWQM))
    Blocks[&MBB].OutNeeds |= StateWQM;

  BlockInfo BI = Blocks[&MBB]; // Make a copy to prevent dangling references.

  if (!MBB.empty()) {
    MachineInstr *LastMI = &*MBB.rbegin();
    InstrInfo &LastII = Instructions[LastMI];
    if ((LastII.OutNeeds | BI.OutNeeds) != LastII.OutNeeds) {
      LastII.OutNeeds |= BI.OutNeeds;
      Worklist.emplace_back(LastMI);
    }
  }

  for (MachineBasicBlock *Pred : MBB.predecessors()) {
    BlockInfo &PredBI = Blocks[Pred];
    if ((PredBI.OutNeeds | BI.InNeeds) == PredBI.OutNeeds)
      continue;

    PredBI.OutNeeds |= BI.InNeeds;
    PredBI.InNeeds |= BI.InNeeds;
    Worklist.emplace_back(Pred);
  }

  for (MachineBasicBlock *Succ : MBB.successors()) {
    BlockInfo &SuccBI = Blocks[Succ];
    if ((SuccBI.InNeeds | BI.OutNeeds) == SuccBI.InNeeds)
      continue;

    SuccBI.InNeeds |= BI.OutNeeds;
    Worklist.emplace_back(Succ);
  }
}

char AMDGPULowerWQMOperations::analyzeFunction(MachineFunction &MF) {
  std::vector<WorkItem> Worklist;
  SmallVector<MachineInstr *> ExeczSideEffectInstrs;
  char GlobalFlags = scanInstructions(MF, Worklist, ExeczSideEffectInstrs);

  while (!Worklist.empty()) {
    WorkItem WI = Worklist.back();
    Worklist.pop_back();

    if (WI.MI)
      propagateInstruction(*WI.MI, Worklist);
    else
      propagateBlock(*WI.MBB, Worklist);

    if (Worklist.empty()) {
      for (auto *MI : ExeczSideEffectInstrs) {
        InstrInfo II = Instructions[MI];
        if (II.OutNeeds & StateWQM)
          markInstructionUses(*MI, StateWQM, Worklist);
      }
      ExeczSideEffectInstrs.clear();
    }
  }

  return GlobalFlags;
}

MachineBasicBlock::iterator
AMDGPULowerWQMOperations::saveSCC(MachineBasicBlock &MBB,
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

void AMDGPULowerWQMOperations::splitBlock(MachineInstr *TermMI) {
  MachineBasicBlock *BB = TermMI->getParent();
  LLVM_DEBUG(dbgs() << "Split block " << printMBBReference(*BB) << " @ "
                    << *TermMI << "\n");

  MachineBasicBlock *SplitBB =
      BB->splitAt(*TermMI, /*UpdateLiveIns*/ true, LIS);

  unsigned NewOpcode = 0;
  switch (TermMI->getOpcode()) {
  case AMDGPU::S_AND_B32:
    NewOpcode = AMDGPU::S_AND_B32_term;
    break;
  case AMDGPU::S_AND_B64:
    NewOpcode = AMDGPU::S_AND_B64_term;
    break;
  case AMDGPU::S_MOV_B32:
    NewOpcode = AMDGPU::S_MOV_B32_term;
    break;
  case AMDGPU::S_MOV_B64:
    NewOpcode = AMDGPU::S_MOV_B64_term;
    break;
  case AMDGPU::S_ANDN2_B32:
    NewOpcode = AMDGPU::S_ANDN2_B32_term;
    break;
  case AMDGPU::S_ANDN2_B64:
    NewOpcode = AMDGPU::S_ANDN2_B64_term;
    break;
  default:
    llvm_unreachable("Unexpected instruction");
  }

  TermMI->setDesc(TII->get(NewOpcode));

  if (SplitBB != BB) {
    using DomTreeT = DomTreeBase<MachineBasicBlock>;
    SmallVector<DomTreeT::UpdateType, 16> DTUpdates;
    for (MachineBasicBlock *Succ : SplitBB->successors()) {
      DTUpdates.push_back({DomTreeT::Insert, SplitBB, Succ});
      DTUpdates.push_back({DomTreeT::Delete, BB, Succ});
    }
    DTUpdates.push_back({DomTreeT::Insert, BB, SplitBB});
    if (MDT)
      MDT->applyUpdates(DTUpdates);
    if (PDT)
      PDT->applyUpdates(DTUpdates);
  }
}

MachineInstr *AMDGPULowerWQMOperations::lowerKillF32(MachineInstr &MI) {
  assert(LiveMaskReg.isVirtual());

  const DebugLoc &DL = MI.getDebugLoc();
  unsigned Opcode = 0;

  assert(MI.getOperand(0).isReg());

  switch (MI.getOperand(2).getImm()) {
  case ISD::SETUEQ:
    Opcode = AMDGPU::V_CMP_LG_F32_e64;
    break;
  case ISD::SETUGT:
    Opcode = AMDGPU::V_CMP_GE_F32_e64;
    break;
  case ISD::SETUGE:
    Opcode = AMDGPU::V_CMP_GT_F32_e64;
    break;
  case ISD::SETULT:
    Opcode = AMDGPU::V_CMP_LE_F32_e64;
    break;
  case ISD::SETULE:
    Opcode = AMDGPU::V_CMP_LT_F32_e64;
    break;
  case ISD::SETUNE:
    Opcode = AMDGPU::V_CMP_EQ_F32_e64;
    break;
  case ISD::SETO:
    Opcode = AMDGPU::V_CMP_O_F32_e64;
    break;
  case ISD::SETUO:
    Opcode = AMDGPU::V_CMP_U_F32_e64;
    break;
  case ISD::SETOEQ:
  case ISD::SETEQ:
    Opcode = AMDGPU::V_CMP_NEQ_F32_e64;
    break;
  case ISD::SETOGT:
  case ISD::SETGT:
    Opcode = AMDGPU::V_CMP_NLT_F32_e64;
    break;
  case ISD::SETOGE:
  case ISD::SETGE:
    Opcode = AMDGPU::V_CMP_NLE_F32_e64;
    break;
  case ISD::SETOLT:
  case ISD::SETLT:
    Opcode = AMDGPU::V_CMP_NGT_F32_e64;
    break;
  case ISD::SETOLE:
  case ISD::SETLE:
    Opcode = AMDGPU::V_CMP_NGE_F32_e64;
    break;
  case ISD::SETONE:
  case ISD::SETNE:
    Opcode = AMDGPU::V_CMP_NLG_F32_e64;
    break;
  default:
    llvm_unreachable("invalid ISD:SET cond code");
  }

  MachineBasicBlock &MBB = *MI.getParent();

  MachineInstr *VcmpMI;
  const MachineOperand &Op0 = MI.getOperand(0);
  const MachineOperand &Op1 = MI.getOperand(1);

  if (TRI->isVGPR(*MRI, Op0.getReg())) {
    Opcode = AMDGPU::getVOPe32(Opcode);
    VcmpMI = BuildMI(MBB, &MI, DL, TII->get(Opcode)).add(Op1).add(Op0);
  } else {
    VcmpMI = BuildMI(MBB, &MI, DL, TII->get(Opcode))
                 .addReg(LMC.VccReg, RegState::Define)
                 .addImm(0) // src0 modifiers
                 .add(Op1)
                 .addImm(0) // src1 modifiers
                 .add(Op0)
                 .addImm(0); // omod
  }

  MachineInstr *MaskUpdateMI =
      BuildMI(MBB, MI, DL, TII->get(LMC.AndN2Opc), LiveMaskReg)
          .addReg(LiveMaskReg)
          .addReg(LMC.VccReg);

  MachineInstr *EarlyTermMI =
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::SI_EARLY_TERMINATE_SCC0));

  MachineInstr *ExecMaskMI =
      BuildMI(MBB, MI, DL, TII->get(LMC.AndN2Opc), LMC.ExecReg)
          .addReg(LMC.ExecReg)
          .addReg(LMC.VccReg);

  assert(MBB.succ_size() == 1);

  LIS->ReplaceMachineInstrInMaps(MI, *VcmpMI);
  MBB.remove(&MI);

  LIS->InsertMachineInstrInMaps(*MaskUpdateMI);
  LIS->InsertMachineInstrInMaps(*EarlyTermMI);
  LIS->InsertMachineInstrInMaps(*ExecMaskMI);

  return ExecMaskMI;
}

MachineInstr *AMDGPULowerWQMOperations::lowerKillI1(MachineInstr &MI,
                                                    bool IsWQM) {
  assert(LiveMaskReg.isVirtual());

  MachineBasicBlock &MBB = *MI.getParent();

  const DebugLoc &DL = MI.getDebugLoc();
  MachineInstr *MaskUpdateMI = nullptr;

  const bool IsDemote = IsWQM && (MI.getOpcode() == AMDGPU::SI_DEMOTE_I1);
  const MachineOperand &Op = MI.getOperand(0);
  int64_t KillVal = MI.getOperand(1).getImm();
  MachineInstr *ComputeKilledMaskMI = nullptr;
  Register CndReg = !Op.isImm() ? Op.getReg() : Register();
  Register TmpReg;

  if (Op.isImm()) {
    if (Op.getImm() == KillVal) {
      MaskUpdateMI = BuildMI(MBB, MI, DL, TII->get(LMC.AndN2Opc), LiveMaskReg)
                         .addReg(LiveMaskReg)
                         .addReg(LMC.ExecReg);
    } else {
      bool IsLastTerminator = std::next(MI.getIterator()) == MBB.end();
      if (!IsLastTerminator) {
        LIS->RemoveMachineInstrFromMaps(MI);
      } else {
        assert(MBB.succ_size() == 1);
        MachineInstr *NewTerm = BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_BRANCH))
                                    .addMBB(*MBB.succ_begin());
        LIS->ReplaceMachineInstrInMaps(MI, *NewTerm);
      }
      MBB.remove(&MI);
      return nullptr;
    }
  } else {
    if (!KillVal) {
      TmpReg = MRI->createVirtualRegister(TRI->getBoolRC());
      ComputeKilledMaskMI = BuildMI(MBB, MI, DL, TII->get(LMC.AndN2Opc), TmpReg)
                                .addReg(LMC.ExecReg)
                                .add(Op);
      MaskUpdateMI = BuildMI(MBB, MI, DL, TII->get(LMC.AndN2Opc), LiveMaskReg)
                         .addReg(LiveMaskReg)
                         .addReg(TmpReg);
    } else {
      MaskUpdateMI = BuildMI(MBB, MI, DL, TII->get(LMC.AndN2Opc), LiveMaskReg)
                         .addReg(LiveMaskReg)
                         .add(Op);
    }
  }

  MachineInstr *EarlyTermMI =
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::SI_EARLY_TERMINATE_SCC0));

  MachineInstr *NewTerm;
  MachineInstr *WQMMaskMI = nullptr;
  Register LiveMaskWQM;
  if (IsDemote) {
    LiveMaskWQM = MRI->createVirtualRegister(TRI->getBoolRC());
    WQMMaskMI = BuildMI(MBB, MI, DL, TII->get(LMC.WQMOpc), LiveMaskWQM)
                    .addReg(LiveMaskReg);
    NewTerm = BuildMI(MBB, MI, DL, TII->get(LMC.AndOpc), LMC.ExecReg)
                  .addReg(LMC.ExecReg)
                  .addReg(LiveMaskWQM);
  } else {
    if (Op.isImm()) {
      NewTerm =
          BuildMI(MBB, &MI, DL, TII->get(LMC.MovOpc), LMC.ExecReg).addImm(0);
    } else if (!IsWQM) {
      NewTerm = BuildMI(MBB, &MI, DL, TII->get(LMC.AndOpc), LMC.ExecReg)
                    .addReg(LMC.ExecReg)
                    .addReg(LiveMaskReg);
    } else {
      unsigned Opcode = KillVal ? LMC.AndN2Opc : LMC.AndOpc;
      NewTerm = BuildMI(MBB, &MI, DL, TII->get(Opcode), LMC.ExecReg)
                    .addReg(LMC.ExecReg)
                    .add(Op);
    }
  }

  LIS->RemoveMachineInstrFromMaps(MI);
  MBB.remove(&MI);
  assert(EarlyTermMI);
  assert(MaskUpdateMI);
  assert(NewTerm);
  if (ComputeKilledMaskMI)
    LIS->InsertMachineInstrInMaps(*ComputeKilledMaskMI);
  LIS->InsertMachineInstrInMaps(*MaskUpdateMI);
  LIS->InsertMachineInstrInMaps(*EarlyTermMI);
  if (WQMMaskMI)
    LIS->InsertMachineInstrInMaps(*WQMMaskMI);
  LIS->InsertMachineInstrInMaps(*NewTerm);

  if (CndReg) {
    LIS->removeInterval(CndReg);
    LIS->createAndComputeVirtRegInterval(CndReg);
  }
  if (TmpReg)
    LIS->createAndComputeVirtRegInterval(TmpReg);
  if (LiveMaskWQM)
    LIS->createAndComputeVirtRegInterval(LiveMaskWQM);

  return NewTerm;
}

void AMDGPULowerWQMOperations::lowerBlock(MachineBasicBlock &MBB,
                                          BlockInfo &BI) {
  if (!BI.NeedsLowering)
    return;

  LLVM_DEBUG(dbgs() << "\nLowering block " << printMBBReference(MBB) << ":\n");

  SmallVector<MachineInstr *, 4> SplitPoints;
  char State = BI.InitialState;

  for (MachineInstr &MI : llvm::make_early_inc_range(
           llvm::make_range(MBB.getFirstNonPHI(), MBB.end()))) {
    auto MIState = StateTransition.find(&MI);
    if (MIState != StateTransition.end())
      State = MIState->second;

    MachineInstr *SplitPoint = nullptr;
    switch (MI.getOpcode()) {
    case AMDGPU::SI_DEMOTE_I1:
    case AMDGPU::SI_KILL_I1_TERMINATOR:
      SplitPoint = lowerKillI1(MI, State == StateWQM);
      break;
    case AMDGPU::SI_KILL_F32_COND_IMM_TERMINATOR:
      SplitPoint = lowerKillF32(MI);
      break;
    default:
      break;
    }
    if (SplitPoint)
      SplitPoints.push_back(SplitPoint);
  }

  for (MachineInstr *MI : SplitPoints)
    splitBlock(MI);
}

MachineBasicBlock::iterator AMDGPULowerWQMOperations::prepareInsertion(
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

void AMDGPULowerWQMOperations::toExact(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator Before,
                                       Register SaveWQM) {
  assert(LiveMaskReg.isVirtual());

  bool IsTerminator = Before == MBB.end();
  if (!IsTerminator) {
    auto FirstTerm = MBB.getFirstTerminator();
    if (FirstTerm != MBB.end()) {
      SlotIndex FirstTermIdx = LIS->getInstructionIndex(*FirstTerm);
      SlotIndex BeforeIdx = LIS->getInstructionIndex(*Before);
      IsTerminator = BeforeIdx > FirstTermIdx;
    }
  }

  const DebugLoc &DL = MBB.findDebugLoc(Before);
  MachineInstr *MI;

  if (SaveWQM) {
    unsigned Opcode =
        IsTerminator ? LMC.AndSaveExecTermOpc : LMC.AndSaveExecOpc;
    MI =
        BuildMI(MBB, Before, DL, TII->get(Opcode), SaveWQM).addReg(LiveMaskReg);
  } else {
    unsigned Opcode = IsTerminator ? LMC.AndTermOpc : LMC.AndOpc;
    MI = BuildMI(MBB, Before, DL, TII->get(Opcode), LMC.ExecReg)
             .addReg(LMC.ExecReg)
             .addReg(LiveMaskReg);
  }

  LIS->InsertMachineInstrInMaps(*MI);
  LIS->removeAllRegUnitsForPhysReg(AMDGPU::EXEC);
  StateTransition[MI] = StateExact;
}

void AMDGPULowerWQMOperations::toWQM(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator Before,
                                     Register SavedWQM) {
  const DebugLoc &DL = MBB.findDebugLoc(Before);
  MachineInstr *MI;

  if (SavedWQM) {
    MI = BuildMI(MBB, Before, DL, TII->get(AMDGPU::COPY), LMC.ExecReg)
             .addReg(SavedWQM);
  } else {
    MI = BuildMI(MBB, Before, DL, TII->get(LMC.WQMOpc), LMC.ExecReg)
             .addReg(LMC.ExecReg);
  }

  LIS->InsertMachineInstrInMaps(*MI);
  StateTransition[MI] = StateWQM;
}

void AMDGPULowerWQMOperations::processBlock(MachineBasicBlock &MBB,
                                            BlockInfo &BI, bool IsEntry) {
  // This is a non-entry block that is WQM throughout, so no need to do
  // anything.
  if (!IsEntry && BI.Needs == StateWQM && BI.OutNeeds != StateExact) {
    BI.InitialState = StateWQM;
    return;
  }

  LLVM_DEBUG(dbgs() << "\nProcessing block " << printMBBReference(MBB)
                    << ":\n");

  Register SavedWQMReg;
  bool WQMFromExec = IsEntry;
  char State = (IsEntry || !(BI.InNeeds & StateWQM)) ? StateExact : StateWQM;
  const TargetRegisterClass *BoolRC = TRI->getBoolRC();

  auto II = MBB.getFirstNonPHI(), IE = MBB.end();
  if (IsEntry) {
    // Skip the instruction that saves LiveMask
    if (II != IE && II->getOpcode() == AMDGPU::COPY &&
        II->getOperand(1).getReg() == LMC.ExecReg)
      ++II;
  }

  MachineBasicBlock::iterator FirstWQM = IE;

  // Track whether we're inside an ENTER/EXIT strict bracket
  bool InStrictBracket = false;
  char SavedStateBeforeStrict = 0;

  BI.InitialState = State;

  for (unsigned Idx = 0;; ++Idx) {
    MachineBasicBlock::iterator Next = II;

    // ─── ENTER/EXIT bracket detection ───
    if (II != IE) {
      unsigned Opcode = II->getOpcode();
      if (Opcode == AMDGPU::ENTER_STRICT_WWM ||
          Opcode == AMDGPU::ENTER_STRICT_WQM) {
        SavedStateBeforeStrict = State;
        InStrictBracket = true;
        ++Next;
        II = Next;
        continue;
      }
      if (Opcode == AMDGPU::EXIT_STRICT_WWM ||
          Opcode == AMDGPU::EXIT_STRICT_WQM) {
        State = SavedStateBeforeStrict;
        InStrictBracket = false;
        FirstWQM = IE; // reset safe insertion point
        ++Next;
        II = Next;
        continue;
      }
    }

    // ─── Skip all instructions inside brackets ───
    if (InStrictBracket) {
      if (II == IE)
        break;
      II = ++Next;
      continue;
    }

    // ─── Regular WQM/Exact transition logic ───
    char Needs = StateExact | StateWQM;
    char OutNeeds = 0;

    if (FirstWQM == IE)
      FirstWQM = II;

    // Adjust needs if this is first instruction of WQM requiring shader.
    if (IsEntry && Idx == 0 && (BI.InNeeds & StateWQM) && II != IE &&
        !TII->isDisableWQM(*II))
      Needs = StateWQM;

    if (II != IE) {
      MachineInstr &MI = *II;

      if (MI.isTerminator() || TII->mayReadEXEC(*MRI, MI)) {
        auto III = Instructions.find(&MI);
        if (III != Instructions.end()) {
          if (III->second.Needs & StateWQM)
            Needs = StateWQM;
          else
            Needs &= ~III->second.Disabled;
          OutNeeds = III->second.OutNeeds;
        }
      } else {
        Needs = StateExact | StateWQM;
      }

      if (MI.isBranch() && OutNeeds == StateExact)
        Needs = StateExact;

      ++Next;
    } else {
      if (BI.OutNeeds & StateWQM)
        Needs = StateWQM;
      else if (BI.OutNeeds == StateExact)
        Needs = StateExact;
      else
        Needs = StateWQM | StateExact;
    }

    // Now, transition if necessary.
    if (!(Needs & State)) {
      MachineBasicBlock::iterator First = FirstWQM;

      bool SaveSCC = false;
      switch (State) {
      case StateExact:
        SaveSCC = (Needs & StateWQM) && WQMFromExec;
        break;
      case StateWQM:
        SaveSCC = !(Needs & StateWQM);
        break;
      default:
        llvm_unreachable("Unknown state");
        break;
      }

      bool WQMToExact =
          State == StateWQM && (Needs & StateExact) && !(Needs & StateWQM);
      bool ExactToWQM =
          State == StateExact && (Needs & StateWQM) && !(Needs & StateExact);
      bool PreferLast = Needs == StateWQM;

      if ((WQMToExact && (OutNeeds & StateWQM)) || ExactToWQM) {
        for (MachineBasicBlock::iterator I = First; I != II; ++I) {
          if (TII->hasUnwantedEffectsWhenEXECEmpty(*I)) {
            PreferLast = WQMToExact;
            break;
          }
        }
      }
      MachineBasicBlock::iterator Before =
          prepareInsertion(MBB, First, II, PreferLast, SaveSCC);

      if (WQMToExact) {
        if (!WQMFromExec && (OutNeeds & StateWQM)) {
          assert(!SavedWQMReg);
          SavedWQMReg = MRI->createVirtualRegister(BoolRC);
        }
        Before = skipDebugInstructionsForward(Before, MBB.end());
        toExact(MBB, Before, SavedWQMReg);
        State = StateExact;
      } else if (ExactToWQM) {
        assert(WQMFromExec == (SavedWQMReg == 0));

        toWQM(MBB, Before, SavedWQMReg);

        if (SavedWQMReg) {
          LIS->createAndComputeVirtRegInterval(SavedWQMReg);
          SavedWQMReg = Register();
        }
        State = StateWQM;
      } else {
        assert(Needs & State);
      }
    }

    if (Needs != (StateExact | StateWQM))
      FirstWQM = IE;

    if (II == IE)
      break;

    II = Next;
  }
  assert(!SavedWQMReg);
}

bool AMDGPULowerWQMOperations::lowerLiveMaskQueries() {
  for (MachineInstr *MI : LiveMaskQueries) {
    const DebugLoc &DL = MI->getDebugLoc();
    Register Dest = MI->getOperand(0).getReg();

    MachineInstr *Copy =
        BuildMI(*MI->getParent(), MI, DL, TII->get(AMDGPU::COPY), Dest)
            .addReg(LiveMaskReg);

    LIS->ReplaceMachineInstrInMaps(*MI, *Copy);
    MI->eraseFromParent();
  }
  return !LiveMaskQueries.empty();
}

bool AMDGPULowerWQMOperations::lowerCopyInstrs() {
  for (MachineInstr *MI : LowerToCopyInstrs) {
    LLVM_DEBUG(dbgs() << "simplify: " << *MI);

    assert(MI->getNumExplicitOperands() == 2);

    unsigned CopyOp = MI->getOperand(1).isReg()
                          ? (unsigned)AMDGPU::COPY
                          : TII->getMovOpcode(TRI->getRegClassForOperandReg(
                                *MRI, MI->getOperand(0)));
    MI->setDesc(TII->get(CopyOp));
    LLVM_DEBUG(dbgs() << " -> " << *MI);
  }
  return !LowerToCopyInstrs.empty();
}

bool AMDGPULowerWQMOperations::lowerKillInstrs(bool IsWQM) {
  for (MachineInstr *MI : KillInstrs) {
    MachineInstr *SplitPoint = nullptr;
    switch (MI->getOpcode()) {
    case AMDGPU::SI_DEMOTE_I1:
    case AMDGPU::SI_KILL_I1_TERMINATOR:
      SplitPoint = lowerKillI1(*MI, IsWQM);
      break;
    case AMDGPU::SI_KILL_F32_COND_IMM_TERMINATOR:
      SplitPoint = lowerKillF32(*MI);
      break;
    }
    if (SplitPoint)
      splitBlock(SplitPoint);
  }
  return !KillInstrs.empty();
}

void AMDGPULowerWQMOperations::lowerInitExec(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();

  if (MI.getOpcode() == AMDGPU::SI_INIT_WHOLE_WAVE) {
    assert(MBB == &MBB->getParent()->front() &&
           "init whole wave not in entry block");
    Register EntryExec = MRI->createVirtualRegister(TRI->getBoolRC());
    MachineInstr *SaveExec = BuildMI(*MBB, MBB->begin(), MI.getDebugLoc(),
                                     TII->get(LMC.OrSaveExecOpc), EntryExec)
                                 .addImm(-1);

    MRI->replaceRegWith(MI.getOperand(0).getReg(), EntryExec);

    if (LIS) {
      LIS->RemoveMachineInstrFromMaps(MI);
    }

    MI.eraseFromParent();

    if (LIS) {
      LIS->InsertMachineInstrInMaps(*SaveExec);
      LIS->createAndComputeVirtRegInterval(EntryExec);
    }
    return;
  }

  if (MI.getOpcode() == AMDGPU::SI_INIT_EXEC) {
    MachineInstr *InitMI = BuildMI(*MBB, MBB->begin(), MI.getDebugLoc(),
                                   TII->get(LMC.MovOpc), LMC.ExecReg)
                               .addImm(MI.getOperand(0).getImm());
    if (LIS) {
      LIS->RemoveMachineInstrFromMaps(MI);
      LIS->InsertMachineInstrInMaps(*InitMI);
    }
    MI.eraseFromParent();
    return;
  }

  Register InputReg = MI.getOperand(0).getReg();
  MachineInstr *FirstMI = &*MBB->begin();
  if (InputReg.isVirtual()) {
    MachineInstr *DefInstr = MRI->getVRegDef(InputReg);
    assert(DefInstr && DefInstr->isCopy());
    if (DefInstr->getParent() == MBB) {
      if (DefInstr != FirstMI) {
        DefInstr->removeFromParent();
        MBB->insert(FirstMI, DefInstr);
        if (LIS)
          LIS->handleMove(*DefInstr);
      } else {
        FirstMI = &*std::next(FirstMI->getIterator());
      }
    }
  }

  const DebugLoc &DL = MI.getDebugLoc();
  const unsigned WavefrontSize = ST->getWavefrontSize();
  const unsigned Mask = (WavefrontSize << 1) - 1;
  Register CountReg = MRI->createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  auto BfeMI = BuildMI(*MBB, FirstMI, DL, TII->get(AMDGPU::S_BFE_U32), CountReg)
                   .addReg(InputReg)
                   .addImm((MI.getOperand(1).getImm() & Mask) | 0x70000);
  auto BfmMI = BuildMI(*MBB, FirstMI, DL, TII->get(LMC.BfmOpc), LMC.ExecReg)
                   .addReg(CountReg)
                   .addImm(0);
  auto CmpMI = BuildMI(*MBB, FirstMI, DL, TII->get(AMDGPU::S_CMP_EQ_U32))
                   .addReg(CountReg, RegState::Kill)
                   .addImm(WavefrontSize);
  auto CmovMI =
      BuildMI(*MBB, FirstMI, DL, TII->get(LMC.CMovOpc), LMC.ExecReg).addImm(-1);

  if (!LIS) {
    MI.eraseFromParent();
    return;
  }

  LIS->RemoveMachineInstrFromMaps(MI);
  MI.eraseFromParent();

  LIS->InsertMachineInstrInMaps(*BfeMI);
  LIS->InsertMachineInstrInMaps(*BfmMI);
  LIS->InsertMachineInstrInMaps(*CmpMI);
  LIS->InsertMachineInstrInMaps(*CmovMI);

  LIS->removeInterval(InputReg);
  LIS->createAndComputeVirtRegInterval(InputReg);
  LIS->createAndComputeVirtRegInterval(CountReg);
}

MachineBasicBlock::iterator
AMDGPULowerWQMOperations::lowerInitExecInstrs(MachineBasicBlock &Entry,
                                              bool &Changed) {
  MachineBasicBlock::iterator InsertPt = Entry.getFirstNonPHI();

  for (MachineInstr *MI : InitExecInstrs) {
    if (MI->getParent() == &Entry)
      InsertPt = std::next(MI->getIterator());

    lowerInitExec(*MI);
    Changed = true;
  }

  return InsertPt;
}

bool AMDGPULowerWQMOperations::run(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "AMDGPU Lower WQM Operations on " << MF.getName()
                    << " ------------- \n");
  LLVM_DEBUG(MF.dump(););

  Instructions.clear();
  Blocks.clear();
  LiveMaskQueries.clear();
  LowerToCopyInstrs.clear();
  KillInstrs.clear();
  InitExecInstrs.clear();
  StateTransition.clear();

  const char GlobalFlags = analyzeFunction(MF);
  bool Changed = false;

  LiveMaskReg = LMC.ExecReg;

  MachineBasicBlock &Entry = MF.front();
  MachineBasicBlock::iterator EntryMI = lowerInitExecInstrs(Entry, Changed);

  // Store a copy of the original live mask when required
  const bool HasLiveMaskQueries = !LiveMaskQueries.empty();
  const bool HasWaveModes = GlobalFlags & ~StateExact;
  const bool HasKills = !KillInstrs.empty();
  const bool UsesWQM = GlobalFlags & StateWQM;
  if (HasKills || UsesWQM || (HasWaveModes && HasLiveMaskQueries)) {
    LiveMaskReg = MRI->createVirtualRegister(TRI->getBoolRC());
    MachineInstr *MI =
        BuildMI(Entry, EntryMI, DebugLoc(), TII->get(AMDGPU::COPY), LiveMaskReg)
            .addReg(LMC.ExecReg);
    LIS->InsertMachineInstrInMaps(*MI);
    Changed = true;
  }

  LLVM_DEBUG(printInfo());

  Changed |= lowerLiveMaskQueries();
  Changed |= lowerCopyInstrs();

  if (!HasWaveModes) {
    // No wave mode execution
    Changed |= lowerKillInstrs(false);
  } else if (GlobalFlags == StateWQM) {
    // Shader only needs WQM
    auto MI =
        BuildMI(Entry, EntryMI, DebugLoc(), TII->get(LMC.WQMOpc), LMC.ExecReg)
            .addReg(LMC.ExecReg);
    LIS->InsertMachineInstrInMaps(*MI);
    lowerKillInstrs(true);
    Changed = true;
  } else {
    // Mixed mode → full processBlock + lowerBlock
    if (GlobalFlags & StateWQM)
      Blocks[&Entry].InNeeds |= StateWQM;
    for (auto &BII : Blocks)
      processBlock(*BII.first, BII.second, BII.first == &Entry);
    for (auto &BII : Blocks)
      lowerBlock(*BII.first, BII.second);
    Changed = true;
  }

  // Compute live range for live mask
  if (LiveMaskReg != LMC.ExecReg)
    LIS->createAndComputeVirtRegInterval(LiveMaskReg);

  LIS->removeAllRegUnitsForPhysReg(AMDGPU::SCC);

  if (!KillInstrs.empty() || !InitExecInstrs.empty())
    LIS->removeAllRegUnitsForPhysReg(AMDGPU::EXEC);

  return Changed;
}

bool AMDGPULowerWQMOperationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachinePostDominatorTree *PDT =
      &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  AMDGPULowerWQMOperations Impl(MF, LIS, MDT, PDT);
  return Impl.run(MF);
}

PreservedAnalyses
AMDGPULowerWQMOperationsPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);

  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  MachineDominatorTree *MDT = &MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachinePostDominatorTree *PDT =
      &MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF);
  AMDGPULowerWQMOperations Impl(MF, LIS, MDT, PDT);
  bool Changed = Impl.run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<SlotIndexesAnalysis>();
  PA.preserve<LiveIntervalsAnalysis>();
  PA.preserve<MachineDominatorTreeAnalysis>();
  PA.preserve<MachinePostDominatorTreeAnalysis>();
  return PA;
}
