//===- AMDGPUSim/Simulator.cpp - Core Simulation Logic --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements the core simulation logic for the AMDGPU static simulator.
//
//===----------------------------------------------------------------------===//

#include "Simulator.h"
#include "SimInstInfo.h"
#include <algorithm>

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Internal Stall Sources (file-local)
//===----------------------------------------------------------------------===//

struct StallSources {
  unsigned Unit = 0;
  unsigned VALUSlot = 0;
  unsigned CoExec = 0;                 // Combined FU+CoExec total
  unsigned CoExecFromEffective = 0;    // Incremental co-exec only (for verbose)
  unsigned EffectiveCycle = 0;         // Cycle after base stall (for verbose)
  bool HasFUCoExecInteraction = false; // FU stall preceded CoExec
  unsigned DelayAlu = 0;
  unsigned WaitCnt = 0;
  unsigned MemFIFO = 0;
  unsigned RegBank = 0;
  unsigned LongLatVALU = 0;
  unsigned LOLVALUTRANSHazard = 0;
  unsigned SSRC = 0;
  unsigned VaVdst = 0;
  unsigned RAW = 0;
  std::string CachePattern;

  unsigned CacheHits = 0;
  unsigned CacheMisses = 0;
  unsigned CacheEvictions = 0;

  unsigned WMMAStartCycle = 0;
  bool IsScaledWMMA = false;
  bool RegBankInWMMAWindow = false;

  unsigned total() const {
    unsigned EffectiveRegBank = RegBankInWMMAWindow ? 0 : RegBank;
    return std::max({Unit, VALUSlot, CoExec, DelayAlu, WaitCnt, MemFIFO,
                     EffectiveRegBank, LongLatVALU, LOLVALUTRANSHazard, SSRC,
                     VaVdst, RAW});
  }
};

//===----------------------------------------------------------------------===//
// Pure Utility Functions (file-local, no member access needed)
//===----------------------------------------------------------------------===//

static void applyStall(unsigned &IssueCycle, unsigned CurrentCycle,
                       unsigned StallUntil) {
  if (StallUntil > IssueCycle)
    IssueCycle = StallUntil;
}

static StallReason getDominantStallReason(const StallSources &S) {
  unsigned Max = 0;
  StallReason Reason = StallReason::NONE;

  auto Check = [&](unsigned Val, StallReason R) {
    if (Val > Max) {
      Max = Val;
      Reason = R;
    }
  };

  Check(S.WaitCnt, StallReason::WAITCNT);
  Check(S.DelayAlu, StallReason::DELAY_ALU);
  Check(S.LongLatVALU, StallReason::LONG_LAT_VALU);
  Check(S.LOLVALUTRANSHazard, StallReason::LOLVALU_TRANS_HAZARD);
  Check(S.CoExec, StallReason::COEXEC_BLOCKED);
  Check(S.MemFIFO, StallReason::MEM_FIFO);
  Check(S.Unit, StallReason::FU_BUSY);
  Check(S.SSRC, StallReason::VA_SSRC_STALL);
  Check(S.VaVdst, StallReason::VA_VDST_WAIT);
  Check(S.RAW, StallReason::RAW_HAZARD);
  if (!S.RegBankInWMMAWindow)
    Check(S.RegBank, StallReason::REG_BANK);

  return Reason;
}

static bool canMSBSetFuse(InstClass PrevIC) {
  switch (PrevIC) {
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::BARRIER:
  case InstClass::WAITCNT:
    return false;
  case InstClass::VALU:
  case InstClass::TRANS:
  case InstClass::SALU:
  case InstClass::WMMA:
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
  case InstClass::SMEM:
  case InstClass::TDM:
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Simulator::Impl — PIMPL Implementation
//===----------------------------------------------------------------------===//

class Simulator::Impl {
  const SimInstInfo &InstInfo;
  const HWModel &Model;
  GPUSimState State;
  SimulatorConfig Config;

  /// Cached verbose log stream (nullptr if logging disabled).
  raw_ostream *Log;

  //--- delay_alu helpers ---

  unsigned decodeDelayDep(unsigned Dep) const {
    if (Dep == 0)
      return 0;

    // VALU_DEP_1 to VALU_DEP_4 (values 1-4)
    if (Dep >= 1 && Dep <= 4) {
      unsigned Index = Dep - 1;
      if (Index < State.RecentVALU.size()) {
        auto &Recent = State.RecentVALU[State.RecentVALU.size() - 1 - Index];
        unsigned Elapsed = State.CurrentCycle - Recent.IssueCycle;
        if (Elapsed < Recent.Latency)
          return Recent.Latency - Elapsed;
      }
      return 0;
    }

    // TRANS32_DEP_1 to TRANS32_DEP_3 (values 5-7)
    if (Dep >= 5 && Dep <= 7) {
      unsigned Index = Dep - 5;
      if (Index < State.RecentTRANS.size()) {
        auto &Recent = State.RecentTRANS[State.RecentTRANS.size() - 1 - Index];
        unsigned Elapsed = State.CurrentCycle - Recent.IssueCycle;
        if (Elapsed < Recent.Latency)
          return Recent.Latency - Elapsed;
      }
      return 0;
    }

    // SALU_CYCLE_1 to SALU_CYCLE_4 (values 9-12)
    if (Dep >= 9 && Dep <= 12) {
      unsigned WaitCycles = Dep - 8;
      unsigned Elapsed = State.CurrentCycle - State.LastSALUCycle;
      if (Elapsed < WaitCycles)
        return WaitCycles - Elapsed;
      return 0;
    }

    return 0;
  }

  unsigned checkPendingDelayAlu(bool SkipApply = false) {
    if (!State.PendingInstId1)
      return 0;

    auto &Pending = *State.PendingInstId1;
    if (Pending.InstructionsLeft > 0) {
      Pending.InstructionsLeft--;
      return 0;
    }

    if (SkipApply)
      return 0;

    unsigned Stall = decodeDelayDep(Pending.DepType);
    if (Log && Stall > 0) {
      *Log << "    PendingInstId1: Dep=" << Pending.DepType
           << " stall=" << Stall << "\n";
    }
    State.PendingInstId1.reset();
    return Stall;
  }

  unsigned parseDelayAlu(const SimInst &Inst) {
    if (Inst.Class != InstClass::DELAY_ALU)
      return 0;

    unsigned Imm = InstInfo.getDelayAluImm(Inst);
    if (Imm == 0)
      return 0;

    unsigned Dep1 = Imm & 0xF;
    unsigned Skip = (Imm >> 4) & 0x7;
    unsigned Dep2 = (Imm >> 7) & 0xF;

    unsigned Stall1 = decodeDelayDep(Dep1);

    if (Dep2 != 0) {
      State.PendingInstId1 =
          GPUSimState::PendingDelayAlu{Dep2, Skip, State.CurrentCycle};
      if (Log)
        *Log << "    DelayALU: instid0=" << Dep1 << " (stall " << Stall1
             << "), skip=" << Skip << ", instid1=" << Dep2 << " (pending)\n";
    } else if (Log) {
      *Log << "    DelayALU: instid0=" << Dep1 << " (stall " << Stall1 << ")\n";
    }

    return Stall1;
  }

  //--- Wait instruction handling ---

  unsigned computeWaitStall(const SimInst &Inst) {
    if (Inst.Class != InstClass::WAITCNT)
      return 0;

    auto WaitInfo = InstInfo.getWaitInfo(Inst);
    WaitType Wait = WaitInfo.first;
    unsigned WaitCount = WaitInfo.second;

    switch (Wait) {
    case WaitType::DS:
      return State.waitDS(WaitCount);
    case WaitType::VMEMLoad:
      return State.waitVMEMLoad(WaitCount);
    case WaitType::VMEMStore:
      return State.waitVMEMStore(WaitCount);
    case WaitType::SMEM:
      return State.waitSMEM(WaitCount);
    case WaitType::Tensor:
      return State.waitTensor(WaitCount);
    case WaitType::DepCtr: {
      unsigned VaVdstTarget = InstInfo.getVaVdstTarget(Inst);
      if (VaVdstTarget < 15) {
        unsigned ReadyCycle = State.getVaVdstReadyCycle(VaVdstTarget);
        if (ReadyCycle > State.CurrentCycle)
          return ReadyCycle - State.CurrentCycle;
      }
      return 0;
    }
    case WaitType::XCnt:
    case WaitType::None:
      return 0;
    }
    return 0;
  }

  //--- Stall source computation ---

  unsigned computeVALUResourceStall(InstClass IC, unsigned IssueCycle) const {
    unsigned Stall = 0;
    if (IC == InstClass::VALU && State.VALUResourceBusyUntil > IssueCycle)
      Stall = State.VALUResourceBusyUntil - IssueCycle;
    if (IC == InstClass::TRANS && State.inWMMAWindow() &&
        State.VALUResourceBusyUntil > IssueCycle)
      Stall = std::max(Stall, State.VALUResourceBusyUntil - IssueCycle);
    return Stall;
  }

  void computeWMMAScaleStall(unsigned &IssueCycle, StallSources &S) const {
    unsigned ScaleReadCycle = IssueCycle;

    ScaleReadCycle = std::max(ScaleReadCycle, State.VALUResourceBusyUntil);

    if (State.inWMMAWindow()) {
      unsigned CoExecStall = State.getCoExecStall(InstClass::VALU);
      ScaleReadCycle =
          std::max(ScaleReadCycle, State.CurrentCycle + CoExecStall);
    }

    unsigned WMMAStartCycle = ScaleReadCycle + 1;
    unsigned XDLFreeAt = State.getUnitBusyUntil(FunctionalUnit::XDL);

    if (WMMAStartCycle < XDLFreeAt) {
      unsigned DesiredScaleSlot = XDLFreeAt - 1;

      if (State.inWMMAWindow() &&
          DesiredScaleSlot >= State.ActiveWMMA.StartCycle &&
          DesiredScaleSlot < State.ActiveWMMA.EndCycle) {
        auto StageOpt = State.ActiveWMMA.getCurrentStage(DesiredScaleSlot);
        if (StageOpt &&
            State.ActiveWMMA.Info.canCoExec(InstClass::VALU, *StageOpt)) {
          ScaleReadCycle = DesiredScaleSlot;
          WMMAStartCycle = XDLFreeAt;
        } else {
          ScaleReadCycle = State.ActiveWMMA.EndCycle;
          WMMAStartCycle = ScaleReadCycle + 1;
        }
      } else {
        ScaleReadCycle = DesiredScaleSlot;
        WMMAStartCycle = XDLFreeAt;
      }
    }

    IssueCycle = ScaleReadCycle;
    S.WMMAStartCycle = WMMAStartCycle;
    S.VALUSlot = IssueCycle - State.CurrentCycle;
  }

  unsigned computeRAWStall(const SimInst &Inst) const {
    if (!Config.EnableScoreboard)
      return 0;

    unsigned MaxRAW = 0;
    SmallVector<RegOperand, 8> SrcRegs;
    InstInfo.getSrcRegs(Inst, SrcRegs);

    for (const RegOperand &Op : SrcRegs) {
      if (Op.RegType == RegOperand::Type::VGPR) {
        for (unsigned i = 0; i < Op.NumComponents; ++i) {
          unsigned RAW = State.getRAWStall(Op.HWIndex + i, true);
          MaxRAW = std::max(MaxRAW, RAW);
        }
      } else if (Op.RegType == RegOperand::Type::SGPR) {
        for (unsigned i = 0; i < Op.NumComponents; ++i) {
          unsigned RAW = State.getRAWStall(Op.HWIndex + i, false);
          MaxRAW = std::max(MaxRAW, RAW);
        }
      }
    }
    if (Log && MaxRAW > 0)
      *Log << "    RAW dependency: stall=" << MaxRAW << "\n";
    return MaxRAW;
  }

  unsigned computeMemFIFOStall(InstClass IC) const {
    switch (IC) {
    case InstClass::DS_READ:
    case InstClass::DS_WRITE:
      return State.getDSFIFOStall();
    case InstClass::VMEM_READ:
    case InstClass::VMEM_WRITE:
      return State.getVMEMBufferStall();
    case InstClass::TDM:
      return State.getTDMFIFOStall();
    default:
      return 0;
    }
  }

  StallSources computeStallSources(const SimInst &Inst) {
    StallSources S;
    InstClass IC = Inst.Class;
    FunctionalUnit Unit = Inst.Unit;
    unsigned IssueCycle = State.CurrentCycle;
    bool IsLOLVALU = InstInfo.isLOLVALU(Inst);

    // 1. Pending delay_alu from previous instruction
    unsigned PendingDelay = checkPendingDelayAlu(/*SkipApply=*/false);
    if (PendingDelay > 0) {
      S.DelayAlu = PendingDelay;
      applyStall(IssueCycle, State.CurrentCycle,
                 State.CurrentCycle + PendingDelay);
    }

    // 2. Functional unit availability
    unsigned BusyUntil = State.getUnitBusyUntil(Unit);
    if (BusyUntil > IssueCycle) {
      S.Unit = BusyUntil - State.CurrentCycle;
      IssueCycle = BusyUntil;
    }

    // 3. VALU resource stalls (LOLVALU, TRANS in WMMA window)
    unsigned VALUResStall = computeVALUResourceStall(IC, IssueCycle);
    if (VALUResStall > 0) {
      S.Unit = std::max(S.Unit, VALUResStall);
      IssueCycle = std::max(IssueCycle, State.VALUResourceBusyUntil);
    }

    // 4. LOLVALU <-> TRANS 1-cycle mutual exclusion hazard
    if ((IC == InstClass::TRANS || IsLOLVALU) &&
        State.LOLVALUTRANSHazardUntil > IssueCycle) {
      S.LOLVALUTRANSHazard = State.LOLVALUTRANSHazardUntil - IssueCycle;
      IssueCycle = State.LOLVALUTRANSHazardUntil;
    }

    // Get source regs for bank stall calculation
    SmallVector<RegOperand, 8> SrcRegs;
    InstInfo.getSrcRegs(Inst, SrcRegs);

    // 5. WMMA-specific stalls
    if (IC == InstClass::WMMA) {
      bool HasScaling = InstInfo.hasScaling(Inst);
      S.IsScaledWMMA = HasScaling;

      unsigned TRANSStall = State.getWMMATRANSStall();
      applyStall(IssueCycle, State.CurrentCycle,
                 State.CurrentCycle + TRANSStall);

      if (HasScaling) {
        computeWMMAScaleStall(IssueCycle, S);
      } else {
        S.WMMAStartCycle = IssueCycle;
      }

      // Track WMMA cache using only A and B matrix VGPR sources
      SmallVector<RegOperand, 4> WMMASrcRegs;
      InstInfo.getWMMASrcRegs(Inst, WMMASrcRegs);
      auto RB = State.RegFile.getRegBankStalls(WMMASrcRegs);
      S.CachePattern = RB.CachePattern;
      S.CacheHits = RB.CacheHits;
      S.CacheMisses = RB.CacheMisses;
      S.CacheEvictions = RB.CacheEvictions;
    }

    // 6. Register bank stalls (VALU/TRANS/SALU)
    if (IC == InstClass::VALU || IC == InstClass::TRANS ||
        IC == InstClass::SALU) {
      auto RB = State.RegFile.getRegBankStalls(SrcRegs);
      S.RegBank = RB.Stalls;
      S.CachePattern = RB.CachePattern;
      S.CacheHits = RB.CacheHits;
      S.CacheMisses = RB.CacheMisses;
      S.CacheEvictions = RB.CacheEvictions;
      if (State.inWMMAWindow()) {
        S.RegBankInWMMAWindow = true;
      } else {
        IssueCycle += RB.Stalls;
      }
    }

    // 7. VA_SSRC: VALU with SGPR blocks SALU
    if (IC == InstClass::SALU && State.VaSSRCBusyUntil > IssueCycle) {
      S.SSRC = State.VaSSRCBusyUntil - IssueCycle;
      IssueCycle = State.VaSSRCBusyUntil;
    }

    // 8. RAW hazards (optional scoreboard mode)
    unsigned RAWStall = computeRAWStall(Inst);
    if (RAWStall > 0) {
      S.RAW = RAWStall;
      applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + RAWStall);
    }

    // 9. s_wait_alu va_vdst(N)
    auto WaitInfo = InstInfo.getWaitInfo(Inst);
    if (WaitInfo.first == WaitType::DepCtr) {
      unsigned VaVdstTarget = InstInfo.getVaVdstTarget(Inst);
      if (VaVdstTarget < 15) {
        unsigned ReadyCycle = State.getVaVdstReadyCycle(VaVdstTarget);
        if (ReadyCycle > State.CurrentCycle) {
          S.VaVdst = ReadyCycle - State.CurrentCycle;
          applyStall(IssueCycle, State.CurrentCycle, ReadyCycle);
          if (Log) {
            *Log << "    s_wait_alu: va_vdst(" << VaVdstTarget
                 << "), pending=" << State.getVaVdst() << ", stall=" << S.VaVdst
                 << "\n";
          }
        }
      }
    }

    // 10. WMMA co-execution window rules
    if (State.inWMMAWindow() && IC != InstClass::WMMA) {
      if (IsLOLVALU) {
        // LOLVALU can't co-execute - waits for entire window
        if (State.ActiveWMMA.EndCycle > IssueCycle) {
          S.LongLatVALU = State.ActiveWMMA.EndCycle - IssueCycle;
          IssueCycle = State.ActiveWMMA.EndCycle;
        }
      } else {
        unsigned CoExecStall = State.getCoExecStallAt(IC, IssueCycle);
        if (CoExecStall > 0) {
          S.EffectiveCycle = IssueCycle;
          S.CoExecFromEffective = CoExecStall;
          S.HasFUCoExecInteraction = (IssueCycle > State.CurrentCycle);
          IssueCycle += CoExecStall;
        }
      }
      S.CoExec = IssueCycle - State.CurrentCycle; // Combined total
    }

    // 11. delay_alu stall
    if (IC == InstClass::DELAY_ALU) {
      unsigned DelayStall = parseDelayAlu(Inst);
      S.DelayAlu = std::max(S.DelayAlu, DelayStall);
      applyStall(IssueCycle, State.CurrentCycle,
                 State.CurrentCycle + DelayStall);
    }

    // 12. Wait instruction stall (preview, actual apply later)
    if (IC == InstClass::WAITCNT) {
      unsigned Pending = 0, Completion = 0;
      unsigned WaitCount = WaitInfo.second;
      switch (WaitInfo.first) {
      case WaitType::DS:
        Pending = State.PendingDS.size();
        if (Pending > WaitCount) {
          unsigned Idx = Pending - WaitCount - 1;
          Completion = State.PendingDS[Idx].CompletionCycle;
        }
        break;
      case WaitType::VMEMLoad:
        Pending = State.PendingVMEMLoad.size();
        if (Pending > WaitCount) {
          unsigned Idx = Pending - WaitCount - 1;
          Completion = State.PendingVMEMLoad[Idx].CompletionCycle;
        }
        break;
      case WaitType::VMEMStore:
        Pending = State.PendingVMEMStore.size();
        if (Pending > WaitCount) {
          unsigned Idx = Pending - WaitCount - 1;
          Completion = State.PendingVMEMStore[Idx].CompletionCycle;
        }
        break;
      case WaitType::SMEM:
        Pending = State.PendingSMEM.size();
        if (Pending > WaitCount) {
          unsigned Idx = Pending - WaitCount - 1;
          Completion = State.PendingSMEM[Idx].CompletionCycle;
        }
        break;
      case WaitType::Tensor:
        Pending = State.PendingTDM.size();
        if (Pending > WaitCount) {
          unsigned Idx = Pending - WaitCount - 1;
          Completion = State.PendingTDM[Idx].CompletionCycle;
        }
        break;
      default:
        break;
      }
      if (Completion > State.CurrentCycle)
        S.WaitCnt = Completion - State.CurrentCycle;
      applyStall(IssueCycle, State.CurrentCycle,
                 State.CurrentCycle + S.WaitCnt);
    }

    // 13. Memory FIFO stall
    unsigned FIFOStall = computeMemFIFOStall(IC);
    S.MemFIFO = FIFOStall;
    applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + FIFOStall);

    return S;
  }

  //--- Record instruction state updates ---

  void recordInstruction(const SimInst &Inst, unsigned WMMAStartCycle) {
    InstClass IC = Inst.Class;

    switch (IC) {
    case InstClass::VALU: {
      State.trackVALU(Inst.Latency);
      State.trackVALUForWMMA(IC);
      State.trackVaVdst(Inst.Latency, Model.VaVdstMultiplier);
      if (InstInfo.hasSGPROperands(Inst))
        State.VaSSRCBusyUntil =
            std::max(State.VaSSRCBusyUntil, State.CurrentCycle + Inst.Latency);
      bool IsLOLVALU = InstInfo.isLOLVALU(Inst);
      if (IsLOLVALU && State.inWMMAWindow())
        State.holdVALUResourceInWindow(InstInfo.getRepeatRate(Inst));
      if (IsLOLVALU)
        State.LOLVALUTRANSHazardUntil =
            std::max(State.LOLVALUTRANSHazardUntil, State.CurrentCycle + 2);
      break;
    }

    case InstClass::SALU:
      State.LastSALUCycle = State.CurrentCycle;
      break;

    case InstClass::TRANS:
      State.trackTRANS(Inst.Latency);
      State.trackVALUForWMMA(IC);
      State.trackVaVdst(Inst.Latency, Model.VaVdstMultiplier);
      State.holdVALUResourceInWindow(InstInfo.getResourceCycles(Inst));
      State.LOLVALUTRANSHazardUntil =
          std::max(State.LOLVALUTRANSHazardUntil, State.CurrentCycle + 2);
      break;

    case InstClass::WMMA: {
      State.trackTRANS(Inst.Latency);
      bool IsScaled = InstInfo.hasScaling(Inst);

      unsigned EffectiveStart = IsScaled ? WMMAStartCycle : State.CurrentCycle;
      unsigned Occupancy = State.startWMMAWindow(InstInfo.getWMMAVariant(Inst),
                                                 IsScaled, EffectiveStart);

      if (IsScaled) {
        State.VALUResourceBusyUntil =
            std::max(State.VALUResourceBusyUntil, State.CurrentCycle + 1);
        State.LastVALUCycle = State.CurrentCycle;
      }

      if (InstInfo.hasSGPROperands(Inst))
        State.VaSSRCBusyUntil =
            std::max(State.VaSSRCBusyUntil, EffectiveStart + Occupancy);

      State.PendingVaVdst.push_back(
          {EffectiveStart + Occupancy * Model.VaVdstMultiplier});

      if (Log) {
        *Log << "  Class: WMMA | Unit: XDL | Occupancy: " << Occupancy
             << " | Window: " << State.ActiveWMMA.Info.TotalWindow << "\n";
      }
      break;
    }

    case InstClass::DS_READ: {
      auto DestInfo = InstInfo.getDestRegInfo(Inst, true);
      State.issueDS(Inst.Latency, DestInfo.first, std::max(DestInfo.second, 1u),
                    true);
      break;
    }

    case InstClass::DS_WRITE:
      State.issueDS(Inst.Latency, 0, 0, false);
      break;

    case InstClass::VMEM_READ: {
      auto DestInfo = InstInfo.getDestRegInfo(Inst, true);
      State.issueVMEM(Inst.Latency, DestInfo.first,
                      std::max(DestInfo.second, 1u), true);
      break;
    }

    case InstClass::VMEM_WRITE:
      State.issueVMEM(Inst.Latency, 0, 0, false);
      break;

    case InstClass::SMEM: {
      auto DestInfo = InstInfo.getDestRegInfo(Inst, false);
      State.issueSMEM(Inst.Latency, DestInfo.first,
                      std::max(DestInfo.second, 1u));
      break;
    }

    case InstClass::TDM:
      State.issueTDM(Inst.Latency);
      break;

    default:
      break;
    }

    // Update functional unit busy time (except WMMA which is handled above)
    if (IC != InstClass::WMMA) {
      unsigned ResourceCycles = InstInfo.getResourceCycles(Inst);
      State.setUnitBusyUntil(Inst.Unit, State.CurrentCycle + ResourceCycles);
    }

    // Scoreboard: record destination registers and handle implicit waits
    if (Config.EnableScoreboard) {
      if (InstInfo.waitsForVALU(Inst)) {
        State.clearRegScoreboard();
        if (Log)
          *Log << "  \xe2\x86\x92 Scoreboard cleared (implicit VA_VDST wait)\n";
      }
      if (IC == InstClass::VALU || IC == InstClass::TRANS ||
          IC == InstClass::SALU || IC == InstClass::WMMA) {
        SmallVector<RegOperand, 4> DstRegs;
        InstInfo.getDstRegs(Inst, DstRegs);
        for (const RegOperand &Op : DstRegs) {
          if (Op.RegType == RegOperand::Type::VGPR) {
            for (unsigned i = 0; i < Op.NumComponents; ++i)
              State.recordRegWrite(Op.HWIndex + i, true, Inst.Latency);
          } else if (Op.RegType == RegOperand::Type::SGPR) {
            for (unsigned i = 0; i < Op.NumComponents; ++i)
              State.recordRegWrite(Op.HWIndex + i, false, Inst.Latency);
          }
        }
      }
    }
  }

  //--- Populate InstrSimInfo ---

  void populateInstrSimInfo(InstrSimInfo &Info, const StallSources &S,
                            InstClass IC) const {
    Info.StallCycles = S.total();
    Info.Reason = getDominantStallReason(S);
    Info.CachePattern = S.CachePattern;
    Info.CacheHits = S.CacheHits;
    Info.CacheMisses = S.CacheMisses;
    Info.CacheEvictions = S.CacheEvictions;
    Info.RegBankStalls = S.RegBank;
    Info.RegBankInWMMAWindow = S.RegBankInWMMAWindow;

    Info.Breakdown.FU = S.Unit;
    Info.Breakdown.VALUSlot = S.VALUSlot;
    Info.Breakdown.CoExec = S.CoExec;
    Info.Breakdown.CoExecFromEffective = S.CoExecFromEffective;
    Info.Breakdown.EffectiveCycle = S.EffectiveCycle;
    Info.Breakdown.HasFUCoExecInteraction = S.HasFUCoExecInteraction;
    Info.Breakdown.DelayAlu = S.DelayAlu;
    Info.Breakdown.WaitCnt = S.WaitCnt;
    Info.Breakdown.MemFIFO = S.MemFIFO;
    Info.Breakdown.RegBank = S.RegBank;
    Info.Breakdown.LongLatVALU = S.LongLatVALU;
    Info.Breakdown.LOLVALUTRANSHazard = S.LOLVALUTRANSHazard;
    Info.Breakdown.SSRC = S.SSRC;
    Info.Breakdown.VaVdst = S.VaVdst;
    Info.Breakdown.RAW = S.RAW;
    Info.Breakdown.RegBankInWMMAWindow = S.RegBankInWMMAWindow;
    Info.Breakdown.IsScaledWMMA = S.IsScaledWMMA;
    Info.Breakdown.WMMAStartCycle = S.WMMAStartCycle;

    if (IC == InstClass::DELAY_ALU)
      Info.WasFused = true;

    if (State.inWMMAWindow() && IC != InstClass::WMMA) {
      Info.InWMMAWindow = true;
      Info.WMMATotalWindow = State.ActiveWMMA.Info.TotalWindow;

      auto StageOpt = State.getWMMAStage();
      if (StageOpt) {
        Info.WMMAStage = *StageOpt;
        uint8_t Mask = State.ActiveWMMA.Info.StageMask[*StageOpt];
        Info.StageType = CoExecMask::getStageType(Mask);
      }

      Info.CoExecuted = (S.CoExec == 0 && S.LongLatVALU == 0);
    }
  }

  //--- Verbose logging ---

  void logStalls(const StallSources &Stalls) const {
    *Log << "  Stalls: ";
    if (Stalls.total() == 0) {
      *Log << "(none)";
    } else {
      bool First = true;
      auto printStall = [&](const char *Name, unsigned Val) {
        if (Val > 0) {
          if (!First)
            *Log << ", ";
          *Log << Name << "=" << Val;
          First = false;
        }
      };
      printStall("FU", Stalls.Unit);
      printStall("VALUSlot", Stalls.VALUSlot);
      printStall("WMMACoExecMiss", Stalls.CoExecFromEffective);
      printStall("LongLatVALU", Stalls.LongLatVALU);
      printStall("LOLVALUxTRANS", Stalls.LOLVALUTRANSHazard);
      printStall("SSRC", Stalls.SSRC);
      printStall("VaVdst", Stalls.VaVdst);
      printStall("RAW", Stalls.RAW);
      printStall("DelayALU", Stalls.DelayAlu);
      printStall("WaitCnt", Stalls.WaitCnt);
      printStall("MemFIFO", Stalls.MemFIFO);
      printStall("RegBank", Stalls.RegBank);
      printStall("ISFetch", 0u); // IS cache is pass-level
      if (Stalls.RegBankInWMMAWindow && Stalls.RegBank > 0)
        *Log << " [in WMMA window, not counted]";
    }
    *Log << " \xe2\x86\x92 Total: " << Stalls.total();
    if (!Stalls.CachePattern.empty())
      *Log << " Cache" << Stalls.CachePattern;
    *Log << "\n";

    if (Stalls.HasFUCoExecInteraction) {
      auto EffectiveStage =
          State.ActiveWMMA.getCurrentStage(Stalls.EffectiveCycle);
      *Log << "    (Base stall lands at cycle " << Stalls.EffectiveCycle;
      if (EffectiveStage) {
        uint8_t Mask = State.ActiveWMMA.Info.StageMask[*EffectiveStage];
        WMMAStageType StageType = CoExecMask::getStageType(Mask);
        const char *StageName = StageType == WMMAStageType::E0  ? "E0"
                                : StageType == WMMAStageType::E ? "E"
                                : StageType == WMMAStageType::I ? "I"
                                : StageType == WMMAStageType::V ? "V"
                                                                : "?";
        *Log << " [stage " << *EffectiveStage << "/"
             << State.ActiveWMMA.Info.TotalWindow << " " << StageName
             << " - blocked]";
      } else {
        *Log << " [outside window]";
      }
      *Log << " \xe2\x86\x92 additional CoExec=" << Stalls.CoExecFromEffective
           << ")\n";
    }
  }

  void logWMMAWindow(InstClass IC) const {
    if (!State.inWMMAWindow() || IC == InstClass::WMMA)
      return;

    auto Stage = State.ActiveWMMA.getCurrentStage(State.CurrentCycle);
    *Log << "  WMMA Window: [" << (Stage ? *Stage : ~0U) << "/"
         << State.ActiveWMMA.Info.TotalWindow << "]";
    if (Stage) {
      uint8_t Mask = State.ActiveWMMA.Info.StageMask[*Stage];
      WMMAStageType ST = CoExecMask::getStageType(Mask);
      const char *StageNames[] = {"?", "E0", "E", "I", "V"};
      *Log << " " << StageNames[(int)ST];
    }
    *Log << " (cycles " << State.ActiveWMMA.StartCycle << "-"
         << State.ActiveWMMA.EndCycle << ")\n";
  }

  void logUnitAndMemState(InstClass IC, FunctionalUnit Unit) const {
    if (Unit != FunctionalUnit::NONE) {
      *Log << "  \xe2\x86\x92 UnitBusyUntil[" << getUnitName(Unit)
           << "] = " << State.getUnitBusyUntil(Unit) << "\n";
    }

    if (IC == InstClass::VALU)
      *Log << "  \xe2\x86\x92 LastVALUCycle = " << State.LastVALUCycle << "\n";
    else if (IC == InstClass::TRANS)
      *Log << "  \xe2\x86\x92 LastTRANSCycle = " << State.LastTRANSCycle
           << "\n";

    switch (IC) {
    case InstClass::DS_READ:
    case InstClass::DS_WRITE:
      *Log << "  \xe2\x86\x92 PendingDS: " << State.PendingDS.size()
           << ", Counter[LGKM]="
           << State.MemCounters[(unsigned)MemCounter::LGKM] << "\n";
      break;
    case InstClass::VMEM_READ:
      *Log << "  \xe2\x86\x92 PendingVMEMLoad: " << State.PendingVMEMLoad.size()
           << ", Counter[VMEM]="
           << State.MemCounters[(unsigned)MemCounter::VMEM] << "\n";
      break;
    case InstClass::VMEM_WRITE:
      *Log << "  \xe2\x86\x92 PendingVMEMStore: "
           << State.PendingVMEMStore.size() << ", Counter[VMEM]="
           << State.MemCounters[(unsigned)MemCounter::VMEM] << "\n";
      break;
    case InstClass::SMEM:
      *Log << "  \xe2\x86\x92 PendingSMEM: " << State.PendingSMEM.size()
           << ", Counter[LGKM]="
           << State.MemCounters[(unsigned)MemCounter::LGKM] << "\n";
      break;
    case InstClass::TDM:
      *Log << "  \xe2\x86\x92 PendingTDM: " << State.PendingTDM.size() << "\n";
      break;
    case InstClass::WMMA:
      *Log << "  \xe2\x86\x92 ActiveWMMA: cycles "
           << State.ActiveWMMA.StartCycle << "-" << State.ActiveWMMA.EndCycle;
      if (State.ActiveWMMA.IsBackToBack)
        *Log << " [back-to-back]";
      *Log << "\n";
      break;
    default:
      break;
    }
  }

  //--- Core simulation ---

  InstrSimInfo simulateInstCore(const SimInst &Inst,
                                ArrayRef<SimInst> Lookahead) {
    InstrSimInfo Info;
    InstClass IC = Inst.Class;

    // Handle MSB_SET specially -- no verbose logging here (pass handles it)
    if (IC == InstClass::MSB_SET) {
      checkPendingDelayAlu(true);

      bool CanFuse = canMSBSetFuse(State.PreviousInstClass);

      if (CanFuse) {
        Info.WasFused = true;
      } else {
        bool IsMasked = false;
        if (State.inWMMAWindow() && !Lookahead.empty()) {
          unsigned NextCoExecStall = State.getCoExecStall(Lookahead[0].Class);
          IsMasked = (NextCoExecStall >= 1);
        }

        if (IsMasked) {
          Info.WasExposed = true;
          Info.WasMasked = true;
        } else {
          Info.WasExposed = true;
          Info.StallCycles = 1;
          Info.Reason = StallReason::MSB_SET_EXPOSED;
          State.advanceCycle(1);
        }
      }
      State.PreviousInstClass = InstClass::SALU;
      return Info;
    }

    // Compute all stall sources
    StallSources Stalls = computeStallSources(Inst);

    // Populate the result info
    populateInstrSimInfo(Info, Stalls, IC);

    // Verbose: log stall breakdown
    if (Log)
      logStalls(Stalls);

    // Advance to ready cycle if needed
    unsigned TotalStall = Stalls.total();
    if (TotalStall > 0) {
      if (Log)
        *Log << "  \xe2\x86\x92 Advancing cycle: " << State.CurrentCycle
             << " \xe2\x86\x92 " << (State.CurrentCycle + TotalStall) << "\n";
      State.advanceToCycle(State.CurrentCycle + TotalStall);
    }

    // Apply wait instruction effects
    if (IC == InstClass::WAITCNT) {
      computeWaitStall(Inst);
    }

    // Verbose: log WMMA window state
    if (Log)
      logWMMAWindow(IC);

    // Record instruction effects on state
    recordInstruction(Inst, Stalls.WMMAStartCycle);

    // Invalidate VGPR cache for written registers
    SmallVector<RegOperand, 4> DstRegs;
    InstInfo.getDstRegs(Inst, DstRegs);
    State.RegFile.invalidateWrites(DstRegs);

    // Verbose: log unit and memory state
    if (Log)
      logUnitAndMemState(IC, Inst.Unit);

    // Advance one cycle for instruction issue
    State.advanceCycle(1);
    State.PreviousInstClass = IC;

    // Populate WMMA-specific info
    if (IC == InstClass::WMMA) {
      Info.IsWMMA = true;
      Info.WMMAPattern = State.ActiveWMMA.Info.Pattern;
    }

    return Info;
  }

public:
  Impl(const SimInstInfo &II, const HWModel &M, SimulatorConfig C)
      : InstInfo(II), Model(M), Config(C), Log(C.Log) {
    State.reset();
    if (Config.EnableISCache)
      State.ISCache.init(Model.ISCacheNumLines, Model.ISCacheLineSize);
  }

  InstrSimInfo simulateInst(const SimInst &Inst, ArrayRef<SimInst> Lookahead) {
    // IS cache: pre-simulation stall (wait for current line to be ready)
    unsigned ISPreStall = 0;
    if (Config.EnableISCache) {
      ISPreStall = State.ISCache.getCurrentLineStall(State.CurrentCycle);
      if (ISPreStall > 0) {
        State.advanceToCycle(State.CurrentCycle + ISPreStall);
        if (Log)
          *Log << "    IS fetch stall: line " << State.ISCache.CurrentLine
               << " not ready, stall=" << ISPreStall << "\n";
      }
    }

    // Core simulation
    InstrSimInfo Info = simulateInstCore(Inst, Lookahead);

    // IS cache: post-simulation byte consumption and line transition stall
    if (Config.EnableISCache) {
      unsigned InstBytes = InstInfo.getInstBytes(Inst);
      unsigned FetchesBefore = State.ISCache.NumFetchesTriggered;
      unsigned ISPostStall = State.ISCache.consumeBytes(
          InstBytes, State.CurrentCycle, Model.SQCToISLatency);

      if (ISPostStall > 0) {
        State.advanceToCycle(State.CurrentCycle + ISPostStall);
        if (Log)
          *Log << "    IS line transition stall: +" << ISPostStall
               << " cycles\n";
      }

      unsigned FetchesTriggered =
          State.ISCache.NumFetchesTriggered - FetchesBefore;
      if (FetchesTriggered > 0 && Log) {
        *Log << "    IS fetch triggered: line "
             << ((State.ISCache.CurrentLine + State.ISCache.NumLines - 1) %
                 State.ISCache.NumLines)
             << " \xe2\x86\x92 ready @ "
             << (State.CurrentCycle + Model.SQCToISLatency)
             << ", now issuing from line " << State.ISCache.CurrentLine
             << " (byte " << State.ISCache.BytesConsumed << ")\n";
      }

      Info.ISFetchStall = ISPreStall + ISPostStall;
      Info.ISFetchesTriggered = FetchesTriggered;
      Info.ISBytesConsumed = InstBytes;
    }

    return Info;
  }

  void reset() {
    State.reset();
    if (Config.EnableISCache)
      State.ISCache.init(Model.ISCacheNumLines, Model.ISCacheLineSize);
  }

  void advanceCycles(unsigned N) {
    if (N > 0)
      State.advanceToCycle(State.CurrentCycle + N);
  }

  const GPUSimState &getState() const { return State; }
  const SimulatorConfig &getConfig() const { return Config; }
  const HWModel &getModel() const { return Model; }
};

//===----------------------------------------------------------------------===//
// Simulator Public API — Delegates to Impl
//===----------------------------------------------------------------------===//

Simulator::Simulator(const SimInstInfo &II, const HWModel &M, SimulatorConfig C)
    : PImpl(std::make_unique<Impl>(II, M, C)) {}

Simulator::~Simulator() = default;

InstrSimInfo Simulator::simulateInst(const SimInst &Inst,
                                     ArrayRef<SimInst> Lookahead) {
  return PImpl->simulateInst(Inst, Lookahead);
}

void Simulator::reset() { PImpl->reset(); }

void Simulator::advanceCycles(unsigned N) { PImpl->advanceCycles(N); }

const GPUSimState &Simulator::getState() const { return PImpl->getState(); }

const SimulatorConfig &Simulator::getConfig() const {
  return PImpl->getConfig();
}

const HWModel &Simulator::getModel() const { return PImpl->getModel(); }

} // namespace AMDGPUSim
} // namespace llvm
