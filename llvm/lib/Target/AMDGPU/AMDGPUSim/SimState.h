//===- AMDGPUSim/SimState.h - Simulation State ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the GPU simulation state (GPUSimState) and related structures.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMSTATE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMSTATE_H

#include "HWModel.h"
#include "SimInst.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <array>
#include <climits>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Memory Counter Types
//===----------------------------------------------------------------------===//

enum class MemCounter : unsigned {
  LGKM = 0,
  VMEM = 1,
  VS = 2,
  TENSOR = 3,
  XCNT = 4,
  NUM_COUNTERS
};

//===----------------------------------------------------------------------===//
// PendingMemOp
//===----------------------------------------------------------------------===//

struct PendingMemOp {
  unsigned IssueCycle;
  unsigned CompletionCycle;
  unsigned DestReg;
  unsigned NumRegs;
  MemCounter Counter;
  bool IsLoad;

  PendingMemOp(unsigned Issue, unsigned Complete, unsigned Dest, unsigned NRegs,
               MemCounter Cnt, bool Load)
      : IssueCycle(Issue), CompletionCycle(Complete), DestReg(Dest),
        NumRegs(NRegs), Counter(Cnt), IsLoad(Load) {}

  bool writesToAny(ArrayRef<unsigned> Regs) const {
    for (unsigned R : Regs) {
      for (unsigned i = 0; i < NumRegs; ++i) {
        if (DestReg + i == R)
          return true;
      }
    }
    return false;
  }
};

//===----------------------------------------------------------------------===//
// VGPRSourceCache
//===----------------------------------------------------------------------===//

struct VGPRSourceCache {
  static constexpr unsigned NumBanks = 8;
  static constexpr unsigned NumPorts = 3;
  static constexpr unsigned CacheDepth = 4;

  std::array<std::array<SmallVector<unsigned, 4>, NumPorts>, NumBanks> Cache;

  unsigned CycleHits = 0;
  unsigned CycleMisses = 0;
  unsigned CycleEvictions = 0;

  void resetCycleStats() {
    CycleHits = 0;
    CycleMisses = 0;
    CycleEvictions = 0;
  }

  bool checkHit(unsigned HWReg, unsigned Port) {
    unsigned Bank = HWReg % NumBanks;
    auto &C = Cache[Bank][Port];
    auto It = std::find(C.begin(), C.end(), HWReg);
    if (It != C.end()) {
      C.erase(It);
      C.push_back(HWReg);
      CycleHits++;
      return true;
    }
    return false;
  }

  void recordMiss(unsigned HWReg, unsigned Port) {
    unsigned Bank = HWReg % NumBanks;
    auto &C = Cache[Bank][Port];
    if (C.size() >= CacheDepth) {
      C.erase(C.begin());
      CycleEvictions++;
    }
    C.push_back(HWReg);
    CycleMisses++;
  }

  void invalidate(unsigned HWReg) {
    unsigned Bank = HWReg % NumBanks;
    for (unsigned Port = 0; Port < NumPorts; ++Port) {
      auto &C = Cache[Bank][Port];
      C.erase(std::remove(C.begin(), C.end(), HWReg), C.end());
    }
  }
};

//===----------------------------------------------------------------------===//
// RegBankResult
//===----------------------------------------------------------------------===//

struct RegBankResult {
  unsigned Stalls = 0;
  std::string CachePattern;
  unsigned CacheHits = 0;
  unsigned CacheMisses = 0;
  unsigned CacheEvictions = 0;
};

//===----------------------------------------------------------------------===//
// RegisterFile
//===----------------------------------------------------------------------===//

struct RegisterFile {
  VGPRSourceCache SrcCache;

  static unsigned countBankConflicts(ArrayRef<unsigned> HWRegs,
                                     unsigned NumBanks) {
    SmallVector<unsigned, 8> BankCount(NumBanks, 0);
    for (unsigned HWReg : HWRegs)
      BankCount[HWReg % NumBanks]++;
    unsigned MaxReads = *std::max_element(BankCount.begin(), BankCount.end());
    return MaxReads > 1 ? MaxReads - 1 : 0;
  }

  RegBankResult getRegBankStalls(ArrayRef<RegOperand> SrcRegs) {
    RegBankResult Result;
    SrcCache.resetCycleStats();
    SmallVector<unsigned, 16> VGPRMisses;
    SmallVector<unsigned, 8> SGPRHWRegs;
    std::string Pattern;

    unsigned PortIdx = 0;
    for (const RegOperand &Op : SrcRegs) {
      unsigned Port = PortIdx % 3;

      if (Op.RegType == RegOperand::Type::VGPR) {
        bool AllHit = true;
        for (unsigned i = 0; i < Op.NumComponents; ++i) {
          unsigned HWReg = Op.HWIndex + i;
          if (!SrcCache.checkHit(HWReg, Port)) {
            VGPRMisses.push_back(HWReg);
            SrcCache.recordMiss(HWReg, Port);
            AllHit = false;
          }
        }
        Pattern += AllHit ? '$' : '-';
      } else if (Op.RegType == RegOperand::Type::SGPR) {
        for (unsigned i = 0; i < Op.NumComponents; ++i)
          SGPRHWRegs.push_back(Op.HWIndex + i);
        // No pattern character for SGPRs (no cache for SGPRs)
      } else {
        // No pattern character for non-register operands
      }
      PortIdx++;
    }

    Result.Stalls =
        countBankConflicts(VGPRMisses, 8) + countBankConflicts(SGPRHWRegs, 4);
    if (!Pattern.empty())
      Result.CachePattern = "(" + Pattern + ")";
    Result.CacheHits = SrcCache.CycleHits;
    Result.CacheMisses = SrcCache.CycleMisses;
    Result.CacheEvictions = SrcCache.CycleEvictions;
    return Result;
  }

  void invalidateWrites(ArrayRef<RegOperand> DstRegs) {
    for (const RegOperand &Op : DstRegs) {
      if (Op.RegType != RegOperand::Type::VGPR)
        continue;
      for (unsigned i = 0; i < Op.NumComponents; ++i)
        SrcCache.invalidate(Op.HWIndex + i);
    }
  }
};

//===----------------------------------------------------------------------===//
// GPUSimState
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Store (IS) Cache State
//===----------------------------------------------------------------------===//

struct ISCacheState {
  unsigned NumLines = 4;
  unsigned LineSizeBytes = 64;
  unsigned CurrentLine = 0;
  unsigned BytesConsumed = 0;
  SmallVector<unsigned, 4> LineReadyCycle;
  unsigned TotalFetchStalls = 0;
  unsigned NumFetchesTriggered = 0;

  ISCacheState() : LineReadyCycle(4, 0) {}

  void init(unsigned NLines, unsigned LineSize) {
    NumLines = NLines;
    LineSizeBytes = LineSize;
    LineReadyCycle.assign(NumLines, 0);
  }

  unsigned consumeBytes(unsigned Bytes, unsigned CurrentCycle,
                        unsigned FetchLatency) {
    unsigned Stall = 0;
    while (Bytes > 0) {
      unsigned RemainingInLine = LineSizeBytes - BytesConsumed;
      if (Bytes <= RemainingInLine) {
        BytesConsumed += Bytes;
        Bytes = 0;
      } else {
        Bytes -= RemainingInLine;
        BytesConsumed = LineSizeBytes;
      }
      if (BytesConsumed >= LineSizeBytes) {
        unsigned FinishedLine = CurrentLine;
        LineReadyCycle[FinishedLine] = CurrentCycle + FetchLatency;
        NumFetchesTriggered++;
        CurrentLine = (CurrentLine + 1) % NumLines;
        BytesConsumed = 0;
        if (LineReadyCycle[CurrentLine] > CurrentCycle) {
          unsigned LineStall = LineReadyCycle[CurrentLine] - CurrentCycle;
          Stall += LineStall;
          TotalFetchStalls += LineStall;
        }
      }
    }
    return Stall;
  }

  unsigned getCurrentLineStall(unsigned CurrentCycle) const {
    if (LineReadyCycle[CurrentLine] > CurrentCycle)
      return LineReadyCycle[CurrentLine] - CurrentCycle;
    return 0;
  }

  void reset() {
    CurrentLine = 0;
    BytesConsumed = 0;
    LineReadyCycle.assign(NumLines, 0);
    TotalFetchStalls = 0;
    NumFetchesTriggered = 0;
  }
};

//===----------------------------------------------------------------------===//
// GPUSimState
//===----------------------------------------------------------------------===//

struct GPUSimState {
  unsigned CurrentCycle = 0;
  RegisterFile RegFile;

  //--- Instruction Store cache ---
  ISCacheState ISCache;

  //--- Functional unit tracking ---
  std::array<unsigned, static_cast<size_t>(FunctionalUnit::NUM_UNITS)>
      UnitBusyUntil = {};

  //--- WMMA co-execution state ---
  struct WMMACoExecState {
    unsigned StartCycle = 0;
    unsigned EndCycle = 0;
    unsigned OccupancyCycle = 0;
    bool Active = false;
    bool IsBackToBack = false;
    WMMACoExecInfo Info;

    std::optional<unsigned> getCurrentStage(unsigned Cycle) const {
      if (!Active || Cycle < StartCycle || Cycle >= EndCycle)
        return std::nullopt;
      return Cycle - StartCycle;
    }
  };
  WMMACoExecState ActiveWMMA;
  bool HadPreviousWMMA = false;

  //--- Register ready times ---
  std::array<unsigned, 1024> VGPRReadyTimes = {};
  std::array<unsigned, 128> SGPRReadyTimes = {};

  //--- Recent instruction tracking for delay_alu ---
  struct RecentInst {
    unsigned IssueCycle;
    unsigned Latency;
  };
  std::deque<RecentInst> RecentVALU;
  std::deque<RecentInst> RecentTRANS;
  unsigned LastSALUCycle = 0;

  //--- VA_VDST tracking ---
  struct PendingVALUWrite {
    unsigned ReadyCycle;
  };
  std::deque<PendingVALUWrite> PendingVaVdst;

  unsigned getVaVdst() const {
    unsigned count = 0;
    for (const auto &e : PendingVaVdst)
      if (e.ReadyCycle > CurrentCycle)
        count++;
    return std::min(count, 15u);
  }

  unsigned getVaVdstReadyCycle(unsigned target) const {
    unsigned currentCount = getVaVdst();
    if (currentCount <= target)
      return CurrentCycle;

    unsigned toRetire = currentCount - target;
    std::vector<unsigned> retireTimes;
    unsigned lastRetire = CurrentCycle;
    for (const auto &e : PendingVaVdst) {
      if (e.ReadyCycle > CurrentCycle) {
        lastRetire = std::max(e.ReadyCycle, lastRetire);
        retireTimes.push_back(lastRetire);
      }
    }

    if (toRetire <= retireTimes.size())
      return retireTimes[toRetire - 1];
    return CurrentCycle;
  }

  void trackVaVdst(unsigned Latency, unsigned Multiplier = 4) {
    PendingVaVdst.push_back({CurrentCycle + Latency * Multiplier});
  }

  //--- VALU/TRANS cycle tracking ---
  unsigned LastVALUCycle = ~0u;
  unsigned LastTRANSCycle = ~0u;

  //--- Resource hazard tracking ---
  unsigned VALUResourceBusyUntil = 0;
  unsigned VaSSRCBusyUntil = 0;
  unsigned LOLVALUTRANSHazardUntil = 0;

  //--- Register scoreboard ---
  DenseMap<unsigned, unsigned> RegScoreboard;

  unsigned getRAWStall(unsigned RegIdx, bool IsVGPR) const {
    auto It = RegScoreboard.find((IsVGPR ? 0x10000 : 0) | RegIdx);
    if (It != RegScoreboard.end() && It->second > CurrentCycle)
      return It->second - CurrentCycle;
    return 0;
  }

  void recordRegWrite(unsigned RegIdx, bool IsVGPR, unsigned Latency) {
    unsigned Key = (IsVGPR ? 0x10000 : 0) | RegIdx;
    RegScoreboard[Key] = CurrentCycle + Latency;
  }

  void clearRegScoreboard() { RegScoreboard.clear(); }

  unsigned getMaxPendingRAW() const {
    unsigned MaxStall = 0;
    for (const auto &KV : RegScoreboard) {
      if (KV.second > CurrentCycle)
        MaxStall = std::max(MaxStall, KV.second - CurrentCycle);
    }
    return MaxStall;
  }

  //--- Pending delay_alu ---
  struct PendingDelayAlu {
    unsigned DepType;
    unsigned InstructionsLeft;
    unsigned IssueCycle;
  };
  std::optional<PendingDelayAlu> PendingInstId1;

  InstClass PreviousInstClass = InstClass::OTHER;

  //--- Memory operation queues ---
  std::deque<PendingMemOp> PendingDS;
  std::deque<PendingMemOp> PendingVMEMLoad;
  std::deque<PendingMemOp> PendingVMEMStore;
  std::deque<PendingMemOp> PendingSMEM;
  std::deque<PendingMemOp> PendingTDM;

  std::array<unsigned, static_cast<size_t>(MemCounter::NUM_COUNTERS)>
      MemCounters = {};

  //=== Query methods ===

  bool inWMMAWindow() const { return ActiveWMMA.Active; }

  std::optional<unsigned> getWMMAStage() const {
    return ActiveWMMA.getCurrentStage(CurrentCycle);
  }

  unsigned getUnitBusyUntil(FunctionalUnit Unit) const {
    if (Unit == FunctionalUnit::NONE)
      return 0;
    return UnitBusyUntil[static_cast<size_t>(Unit)];
  }

  unsigned getUnitStall(FunctionalUnit Unit) const {
    if (Unit == FunctionalUnit::NONE)
      return 0;
    unsigned Busy = getUnitBusyUntil(Unit);
    return (Busy > CurrentCycle) ? (Busy - CurrentCycle) : 0;
  }

  unsigned getWMMATRANSStall() const {
    if (LastTRANSCycle == ~0u)
      return 0;
    unsigned TRANSEndCycle = LastTRANSCycle + 2;
    return (CurrentCycle < TRANSEndCycle) ? (TRANSEndCycle - CurrentCycle) : 0;
  }

  unsigned getCoExecStallAt(InstClass IC, unsigned AtCycle) const {
    if (!ActiveWMMA.Active)
      return 0;

    auto StageOpt = ActiveWMMA.getCurrentStage(AtCycle);
    if (!StageOpt)
      return 0;

    unsigned Stage = *StageOpt;
    const WMMACoExecInfo &Info = ActiveWMMA.Info;

    if (Info.canCoExec(IC, Stage))
      return 0;

    unsigned SearchFrom = Stage + 1;
    auto NextStage = Info.findNextAllowedStage(IC, SearchFrom);

    if (NextStage)
      return *NextStage - Stage;

    return ActiveWMMA.EndCycle - AtCycle;
  }

  unsigned getCoExecStall(InstClass IC) const {
    return getCoExecStallAt(IC, CurrentCycle);
  }

  //=== State modification methods ===

  void setUnitBusyUntil(FunctionalUnit Unit, unsigned Cycle) {
    if (Unit == FunctionalUnit::NONE)
      return;
    UnitBusyUntil[static_cast<size_t>(Unit)] = Cycle;
  }

  void advanceCycle(unsigned N = 1) { advanceToCycle(CurrentCycle + N); }

  unsigned advanceToCycle(unsigned TargetCycle) {
    if (TargetCycle <= CurrentCycle)
      return 0;
    unsigned Delta = TargetCycle - CurrentCycle;
    CurrentCycle = TargetCycle;
    if (ActiveWMMA.Active && CurrentCycle >= ActiveWMMA.EndCycle)
      ActiveWMMA.Active = false;
    retireCompletedMemOps();
    while (!PendingVaVdst.empty() &&
           PendingVaVdst.front().ReadyCycle <= CurrentCycle)
      PendingVaVdst.pop_front();
    return Delta;
  }

  void trackVALUForWMMA(InstClass IC) {
    if (IC == InstClass::VALU)
      LastVALUCycle = CurrentCycle;
    else if (IC == InstClass::TRANS)
      LastTRANSCycle = CurrentCycle;
  }

  void holdVALUResourceInWindow(unsigned Cycles) {
    if (inWMMAWindow())
      VALUResourceBusyUntil =
          std::max(VALUResourceBusyUntil, CurrentCycle + Cycles);
  }

  unsigned getVALUResourceStallInWindow() const {
    if (!inWMMAWindow())
      return 0;
    return (VALUResourceBusyUntil > CurrentCycle)
               ? (VALUResourceBusyUntil - CurrentCycle)
               : 0;
  }

  void setVGPRReady(unsigned RegIdx, unsigned Latency) {
    if (RegIdx < VGPRReadyTimes.size())
      VGPRReadyTimes[RegIdx] = CurrentCycle + Latency;
  }

  void setSGPRReady(unsigned RegIdx, unsigned Latency) {
    if (RegIdx < SGPRReadyTimes.size())
      SGPRReadyTimes[RegIdx] = CurrentCycle + Latency;
  }

  void trackVALU(unsigned Latency) {
    RecentVALU.push_back({CurrentCycle, Latency});
    if (RecentVALU.size() > 5)
      RecentVALU.pop_front();
  }

  void trackTRANS(unsigned Latency) {
    RecentTRANS.push_back({CurrentCycle, Latency});
    if (RecentTRANS.size() > 4)
      RecentTRANS.pop_front();
  }

  unsigned startWMMAWindow(WMMAVariant Variant, bool HasScaling,
                           unsigned WMMAStartCycle) {
    WMMACoExecInfo Info = getWMMACoExecInfo(Variant, HasScaling);

    bool BackToBack = HadPreviousWMMA &&
                      WMMAStartCycle >= ActiveWMMA.OccupancyCycle &&
                      WMMAStartCycle < ActiveWMMA.EndCycle;

    HadPreviousWMMA = true;

    ActiveWMMA.StartCycle = WMMAStartCycle;
    ActiveWMMA.EndCycle = WMMAStartCycle + Info.TotalWindow;
    ActiveWMMA.OccupancyCycle = WMMAStartCycle + Info.Occupancy;
    ActiveWMMA.Active = true;
    ActiveWMMA.IsBackToBack = BackToBack;
    ActiveWMMA.Info = Info;

    setUnitBusyUntil(FunctionalUnit::XDL, ActiveWMMA.OccupancyCycle);

    return Info.Occupancy;
  }

  //=== Memory FIFO methods ===

  unsigned getFIFOStall(const std::deque<PendingMemOp> &Queue,
                        unsigned MaxInFlight) const {
    if (Queue.size() < MaxInFlight)
      return 0;
    unsigned OldestComplete = Queue.front().CompletionCycle;
    return (OldestComplete > CurrentCycle) ? (OldestComplete - CurrentCycle)
                                           : 0;
  }

  unsigned getDSFIFOStall() const {
    return getFIFOStall(PendingDS, MemLimits::MaxDSInFlight);
  }

  unsigned getVMEMBufferStall() const {
    unsigned TotalPending = PendingVMEMLoad.size() + PendingVMEMStore.size();
    if (TotalPending < MemLimits::MaxVMEMInFlight)
      return 0;
    unsigned OldestComplete = UINT_MAX;
    if (!PendingVMEMLoad.empty())
      OldestComplete =
          std::min(OldestComplete, PendingVMEMLoad.front().CompletionCycle);
    if (!PendingVMEMStore.empty())
      OldestComplete =
          std::min(OldestComplete, PendingVMEMStore.front().CompletionCycle);
    return (OldestComplete > CurrentCycle) ? (OldestComplete - CurrentCycle)
                                           : 0;
  }

  unsigned getTDMFIFOStall() const {
    return getFIFOStall(PendingTDM, MemLimits::MaxTDMInFlight);
  }

  void issueMemOp(std::deque<PendingMemOp> &Queue, MemCounter Cnt,
                  unsigned Latency, unsigned BaseReg, unsigned NumRegs,
                  bool IsLoad, bool UpdateVGPR, bool UpdateSGPR) {
    Queue.emplace_back(CurrentCycle, CurrentCycle + Latency, BaseReg, NumRegs,
                       Cnt, IsLoad);
    MemCounters[static_cast<size_t>(Cnt)]++;
    unsigned ReadyTime = CurrentCycle + Latency;
    if (UpdateVGPR) {
      for (unsigned i = 0; i < NumRegs && (BaseReg + i) < VGPRReadyTimes.size();
           ++i)
        VGPRReadyTimes[BaseReg + i] = ReadyTime;
    }
    if (UpdateSGPR) {
      for (unsigned i = 0; i < NumRegs && (BaseReg + i) < SGPRReadyTimes.size();
           ++i)
        SGPRReadyTimes[BaseReg + i] = ReadyTime;
    }
  }

  void issueDS(unsigned Latency, unsigned BaseVGPR, unsigned NumRegs,
               bool IsLoad) {
    issueMemOp(PendingDS, MemCounter::LGKM, Latency, BaseVGPR, NumRegs, IsLoad,
               IsLoad, false);
  }

  void issueVMEM(unsigned Latency, unsigned BaseVGPR, unsigned NumRegs,
                 bool IsLoad) {
    MemCounter Cnt = IsLoad ? MemCounter::VMEM : MemCounter::VS;
    auto &Queue = IsLoad ? PendingVMEMLoad : PendingVMEMStore;
    issueMemOp(Queue, Cnt, Latency, BaseVGPR, NumRegs, IsLoad, IsLoad, false);
  }

  void issueSMEM(unsigned Latency, unsigned BaseSGPR, unsigned NumRegs) {
    issueMemOp(PendingSMEM, MemCounter::LGKM, Latency, BaseSGPR, NumRegs, true,
               false, true);
  }

  void issueTDM(unsigned Latency) {
    issueMemOp(PendingTDM, MemCounter::TENSOR, Latency, 0, 0, true, false,
               false);
  }

  //=== Wait methods ===

  unsigned computeWaitStall(const std::deque<PendingMemOp> &Queue,
                            unsigned WaitCount) const {
    unsigned Pending = Queue.size();
    if (Pending <= WaitCount)
      return 0;
    unsigned WaitForIndex = Pending - WaitCount - 1;
    unsigned CompletionCycle = Queue[WaitForIndex].CompletionCycle;
    return (CompletionCycle > CurrentCycle) ? (CompletionCycle - CurrentCycle)
                                            : 0;
  }

  unsigned applyWait(std::deque<PendingMemOp> &Queue, unsigned WaitCount) {
    if (Queue.size() <= WaitCount)
      return 0;
    unsigned ToRetire = Queue.size() - WaitCount;
    while (Queue.size() > WaitCount)
      Queue.pop_front();
    return ToRetire;
  }

  unsigned waitDS(unsigned WaitCount) {
    unsigned Stall = computeWaitStall(PendingDS, WaitCount);
    applyWait(PendingDS, WaitCount);
    return Stall;
  }

  unsigned waitVMEMLoad(unsigned WaitCount) {
    unsigned Stall = computeWaitStall(PendingVMEMLoad, WaitCount);
    applyWait(PendingVMEMLoad, WaitCount);
    return Stall;
  }

  unsigned waitVMEMStore(unsigned WaitCount) {
    unsigned Stall = computeWaitStall(PendingVMEMStore, WaitCount);
    applyWait(PendingVMEMStore, WaitCount);
    return Stall;
  }

  unsigned waitTensor(unsigned WaitCount) {
    unsigned Stall = computeWaitStall(PendingTDM, WaitCount);
    applyWait(PendingTDM, WaitCount);
    return Stall;
  }

  unsigned waitSMEM(unsigned WaitCount) {
    unsigned Stall = computeWaitStall(PendingSMEM, WaitCount);
    applyWait(PendingSMEM, WaitCount);
    return Stall;
  }

  //=== Utility methods ===

  unsigned getCounter(MemCounter Cnt) const {
    return MemCounters[static_cast<size_t>(Cnt)];
  }

  unsigned getNumPendingDS() const { return PendingDS.size(); }
  unsigned getNumPendingVMEMLoad() const { return PendingVMEMLoad.size(); }
  unsigned getNumPendingVMEMStore() const { return PendingVMEMStore.size(); }
  unsigned getNumPendingSMEM() const { return PendingSMEM.size(); }
  unsigned getNumPendingTDM() const { return PendingTDM.size(); }

  void retireCompletedMemOps() {
    auto RetireFrom = [this](std::deque<PendingMemOp> &Queue) {
      while (!Queue.empty() && Queue.front().CompletionCycle <= CurrentCycle)
        Queue.pop_front();
    };
    RetireFrom(PendingDS);
    RetireFrom(PendingVMEMLoad);
    RetireFrom(PendingVMEMStore);
    RetireFrom(PendingSMEM);
    RetireFrom(PendingTDM);
  }

  void reset() {
    CurrentCycle = 0;
    RegFile = RegisterFile();
    UnitBusyUntil.fill(0);
    ActiveWMMA = WMMACoExecState();
    HadPreviousWMMA = false;
    VGPRReadyTimes.fill(0);
    SGPRReadyTimes.fill(0);
    RecentVALU.clear();
    RecentTRANS.clear();
    LastSALUCycle = 0;
    PendingVaVdst.clear();
    LastVALUCycle = ~0u;
    LastTRANSCycle = ~0u;
    VALUResourceBusyUntil = 0;
    VaSSRCBusyUntil = 0;
    LOLVALUTRANSHazardUntil = 0;
    RegScoreboard.clear();
    PendingInstId1.reset();
    PreviousInstClass = InstClass::OTHER;
    PendingDS.clear();
    PendingVMEMLoad.clear();
    PendingVMEMStore.clear();
    PendingSMEM.clear();
    PendingTDM.clear();
    MemCounters.fill(0);
    ISCache.reset();
  }
};

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMSTATE_H
