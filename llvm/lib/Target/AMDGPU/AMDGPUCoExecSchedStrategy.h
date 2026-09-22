//===- AMDGPUCoExecSchedStrategy.h - CoExec Scheduling Strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Coexecution-focused scheduling strategy for AMDGPU.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECSCHEDSTRATEGY_H

#include "AMDGPUCoExecInfo.h"
#include "GCNSchedStrategy.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class MachineLoopInfo;

// classifyFlavor is declared in AMDGPUCoExecInfo.h but defined in cpp
namespace AMDGPU {
namespace DefaultBufferSizes {
constexpr unsigned DS = 16;
} // namespace DefaultBufferSizes

enum class CarriedLatency : uint8_t { Off, Fence, All };

} // namespace AMDGPU

namespace AMDGPU {

/// AMDGPU-specific scheduling decision reasons. These provide more granularity
/// than the generic CandReason enum for debugging purposes.
enum class AMDGPUSchedReason : uint8_t {
  None,
  Stall,
  MemoryPipeline,
  CoexecSlot,
  CritResourceBalance, // tryCriticalResource chose based on resource pressure
  CritResourceDep,     // tryCriticalResourceDependency chose based on enabling
  ShadowMix,           // tryShadowMix deferred/prioritized for co-exec filling
  NUM_REASONS
};

inline StringRef getReasonName(AMDGPUSchedReason R) {
  switch (R) {
  case AMDGPUSchedReason::None:
    return "None";
  case AMDGPUSchedReason::Stall:
    return "Stall";
  case AMDGPUSchedReason::MemoryPipeline:
    return "MemoryPipeline";
  case AMDGPUSchedReason::CoexecSlot:
    return "CoexecSlot";
  case AMDGPUSchedReason::CritResourceBalance:
    return "CritResource";
  case AMDGPUSchedReason::CritResourceDep:
    return "CritResourceDep";
  case AMDGPUSchedReason::ShadowMix:
    return "ShadowMix";
  case AMDGPUSchedReason::NUM_REASONS:
    llvm_unreachable("Unknown AMDGPUSchedReason");
  }
  llvm_unreachable("Unknown AMDGPUSchedReason");
}

} // End namespace AMDGPU

//===----------------------------------------------------------------------===//
// Roofline Co-Execution Analysis Result
//===----------------------------------------------------------------------===//

/// Results of the roofline co-execution analysis for a scheduling region.
/// Computes the maximum number of WMMA co-exec slots that can be filled
/// by available consumer instructions (via max-flow), yielding an exact
/// lower bound on unavoidable stall cycles under the "arbitrary reorder,
/// coexec-only" abstraction.
struct RooflineResult {
  unsigned TotalSlots = 0;       // Total co-exec slots from all WMMAs (excl. E0)
  unsigned MaxFilledSlots = 0;   // Max-flow: best possible slot filling
  unsigned LowerBoundStalls = 0; // TotalSlots - MaxFilledSlots
  unsigned TotalConsumers = 0;   // Total consumer instructions in region

  // Per-consumer-class counts indexed by CoExecMask bit position (0-7):
  // 0=CTRL, 1=VALU, 2=TRANS, 3=SALU, 4=DS, 5=VMEM, 6=SMEM, 7=WMMA
  unsigned ConsumerCount[8] = {};

  // Per-class consumers that the max-flow could not place into a slot.
  // ExposedByClass[k] = max(0, ConsumerCount[k] - flow assigned to k).
  // Same indexing as ConsumerCount.
  unsigned ExposedByClass[8] = {};
  unsigned WMMACoexecByClass[8] = {};
  unsigned WMMACount = 0;

  unsigned WMMACorrectionPercent = 75;

  bool isValid() const { return TotalSlots > 0; }

  unsigned getWMMACoexecByFlavor(AMDGPU::InstructionFlavor Flavor);

  unsigned getWMMAISlotSupplySlots() {
    if (WMMACount == 0)
      return 0;
    unsigned WMMACorrection = WMMACount * WMMACorrectionPercent / 100;
    unsigned TotalISupply =
        getWMMACoexecByFlavor(AMDGPU::InstructionFlavor::SingleCycleVALU) +
        getWMMACoexecByFlavor(AMDGPU::InstructionFlavor::TRANS);
    return (TotalISupply + WMMACorrection) / WMMACount;
  }

  unsigned getWMMAESlotSupplySlots() {
    if (WMMACount == 0)
      return 0;
    unsigned WMMACorrection = WMMACount * WMMACorrectionPercent / 100;
    unsigned TotalESupply =
        getWMMACoexecByFlavor(AMDGPU::InstructionFlavor::SALU) +
        getWMMACoexecByFlavor(AMDGPU::InstructionFlavor::DS);
    return (TotalESupply + WMMACorrection) / WMMACount;
  }

  float getSlotUtilization() const {
    if (TotalSlots == 0)
      return 0.0f;
    return static_cast<float>(MaxFilledSlots) / TotalSlots;
  }
};

class CandidateHeuristics;

//===----------------------------------------------------------------------===//
// Hardware Unit Information
//===----------------------------------------------------------------------===//

/// HardwareUnitInfo is a wrapper class which maps to some real hardware
/// resource. This is used to model hardware resource pressure per region, and
/// guide scheduling heuristics.
class HardwareUnitInfo {
private:
  /// PrioritySUs maintains a list of the SUs we want to prioritize scheduling
  /// for this HardwareUnit. This is used for agreement between
  /// tryCriticalResourceDependency and tryCriticalResource: we schedule the
  /// dependencies for a SU on critical resource, then schedule that same SU on
  /// the critical resource. This agreement results in shorter live ranges and
  /// more regular HardwareUnit access patterns. SUs are prioritized based on
  /// depth for top-down scheduling.
  SmallSetVector<SUnit *, 16> PrioritySUs;
  /// All the SUs in the region that consume this resource.
  SmallSetVector<SUnit *, 16> AllSUs;
  /// All the SUs for this HardwareUnit that have already been scheduled.
  SmallVector<SUnit *, 16> ScheduledSUs;
  /// The total number of busy cycles for this HardwareUnit for a given region.
  unsigned TotalCycles = 0;
  /// InstructionFlavor mapping.
  AMDGPU::InstructionFlavor Type;
  /// Whether or not instructions on this HardwareUnit may produce a window in
  /// which instructions in other HardwareUnits can coexecute. For example, WMMA
  /// / MFMA instructions may take multiple cycles, which may be overlapped with
  /// instructions on other HardwareUnits.
  bool ProducesCoexecWindow = false;
  /// How many instructions can be held simultaneously for this HardwareUnit.
  /// A value of 0 means there is no limit. A value of 1 models an unbuffered
  /// resource with a single in-flight instruction.
  unsigned BufferSize = 0;
  /// How many cycles it takes for an instruction to clear the buffer.
  unsigned BufferCycles = 0;
  /// Estimated cycles of this flavor that will execute outside any coexec
  /// window (i.e. cannot be hidden behind a producer's shadow). Populated
  /// once at region init from either a hand-ordered allocation or the
  /// roofline solver, then decremented as instructions of this flavor are
  /// scheduled outside an active window. Read by the critical-resource sort
  /// when CoexecExposedSort != Off.
  unsigned RemainingExposed = 0;

  unsigned RemainingCycles = 0;

public:
  HardwareUnitInfo() {}

  auto begin() { return AllSUs.begin(); }
  auto end() { return AllSUs.end(); }
  auto rbegin() { return AllSUs.rbegin(); }
  auto rend() { return AllSUs.rend(); }

  unsigned size() const { return AllSUs.size(); }

  unsigned getTotalCycles() const { return TotalCycles; }

  void setType(unsigned TheType) {
    assert(TheType < (unsigned)AMDGPU::InstructionFlavor::NUM_FLAVORS);
    Type = (AMDGPU::InstructionFlavor)(TheType);
  }

  AMDGPU::InstructionFlavor getType() const { return Type; }

  bool producesCoexecWindow() const { return ProducesCoexecWindow; }

  void setProducesCoexecWindow(bool Val) { ProducesCoexecWindow = Val; }

  bool contains(SUnit *SU) const { return AllSUs.contains(SU); }

  void setBufferSize(unsigned Size) { BufferSize = Size; }

  unsigned getBufferSize() { return BufferSize; }

  unsigned getRemainingExposed() const { return RemainingExposed; }
  void setExposedCount(unsigned N) { RemainingExposed = N; }
  void reduceRemainingExposed() {
    if (RemainingExposed)
      --RemainingExposed;
  }

  unsigned getRemainingCycles() const { return RemainingCycles; }
  void setRemainingCycles(unsigned N) { RemainingExposed = N; }

  /// \returns the next cycle where there is space in the buffer.
  unsigned getBufferAvailableCycle(unsigned CurrCycle) {
    // An unlimited buffer is always available.
    if (BufferSize == 0)
      return CurrCycle;

    // Buffer is available now.
    if (ScheduledSUs.size() < BufferSize)
      return CurrCycle;

    return BufferCycles +
           ScheduledSUs[ScheduledSUs.size() - BufferSize]->TopReadyCycle;
  }

  /// \returns the most recently scheduled SU for this HardwareUnit.
  SUnit *getLastScheduledSU() const {
    unsigned ScheduledCount = ScheduledSUs.size();
    if (!ScheduledCount)
      return nullptr;

    return ScheduledSUs[ScheduledCount - 1];
  }

  /// \returns the list of scheduled SUs in scheduling order.
  ArrayRef<SUnit *> getScheduledSUs() const { return ScheduledSUs; }

  /// \returns the SUnit with higher priority or nullptr if they are the same.
  /// This method looks through the PrioritySUs to determine if one SU is more
  /// prioritized than the other. If neither are in the PrioritySUs list, then
  /// neither have priority over each other.
  SUnit *getHigherPriority(SUnit *SU, SUnit *Other) const {
    for (SUnit *SUOrder : PrioritySUs) {
      if (SUOrder == SU)
        return SU;

      if (SUOrder == Other)
        return Other;
    }
    return nullptr;
  }

  void reset() {
    AllSUs.clear();
    PrioritySUs.clear();
    ScheduledSUs.clear();
    TotalCycles = 0;
    Type = AMDGPU::InstructionFlavor::Other;
    ProducesCoexecWindow = false;
    BufferSize = 0;
    BufferCycles = 0;
    RemainingExposed = 0;
    RemainingCycles = 0;
  }

  /// \returns the next SU in PrioritySUs that is not ready. If \p LookDeep is
  /// set, we will look beyond the PrioritySUs (if all the PrioritySUs are
  /// ready) to AllSUs to attempt to find a target SU. When looking through
  /// AllSUs we sort pick the target SU by minimal depth for top-down
  /// scheduling. getNextTargetSU is useful for determining which SU on this
  /// HardwareUnit we are trying to schedule - this info helps us determine
  /// which dependencies to schedule. LookDeep is useful if the dependencies are
  /// long latency (e.g. memory instructions). If we have many long latency
  /// dependencies, it is beneficial to enable SUs multiple levels ahead.
  SUnit *getNextTargetSU(bool LookDeep = false) const;
  /// Insert the \p SU into AllSUs and account its \p BlockingCycles into
  /// the TotalCycles. This maintains the list of PrioritySUs.
  void insert(SUnit *SU, unsigned BlockingCycles);
  /// Update the state for \p SU being scheduled by removing it from the AllSUs
  /// and reducing its \p BlockingCycles from the TotalCycles. This maintains
  /// the list of PrioritySUs.
  void markScheduled(SUnit *SU, unsigned BlockingCycles);
  /// After we've collected all the region pressure for this HWUI, correct for
  /// any specifics of the behavior of this resource. For example, if we the
  /// HardwareUnit can hold N instructions simultaneously, then there is no
  /// penalty for scheduling N instructions back to back.
  void finalizeCycles();
};

//===----------------------------------------------------------------------===//
// Window Slot Demand
//===----------------------------------------------------------------------===//

struct RegionMixInfo; // Forward declaration

/// Slot demand derived from a WMMA CoExecInfo template.
/// Instead of hardcoded filler thresholds, this counts actual I-slots and
/// E-slots from the WMMA's window pattern, telling the scheduler exactly
/// how many compatible fillers are needed.
struct WindowSlotDemand {
  unsigned ISlots = 0; // Stages accepting VALU/TRANS (CoExecMask::StageI)
  unsigned ESlots = 0; // Stages accepting SALU/DS only (CoExecMask::StageE)
  unsigned TRSlots = 0; // Stages accepting all but TRANS (CoexecMask::StageTR)
  unsigned VSlots = 0; // Vacant stages (next WMMA only, CoExecMask::StageV)

  /// Compute demand from a CoExecInfo template.
  static WindowSlotDemand fromCoExecInfo(const AMDGPU::CoExecInfo &Info);

  /// Check if ready fillers in \p Mix meet the slot demand.
  bool isSatisfied(const RegionMixInfo &Mix) const;

  /// Return the flavor with the largest gap between demand and ready count.
  AMDGPU::InstructionFlavor getMostDeficientFlavor(const RegionMixInfo &Mix) const;

  bool hasSlots() const { return ISlots > 0 || ESlots > 0 || TRSlots > 0; }

  void clamp(RooflineResult &Roofline) {
    ISlots = std::min(ISlots, Roofline.getWMMAISlotSupplySlots());
    ESlots = std::min(ESlots, Roofline.getWMMAESlotSupplySlots());
  }
};

//===----------------------------------------------------------------------===//
// Coexec Window
//===----------------------------------------------------------------------===//

/// Tracks the lifecycle of a co-execution window produced by a multi-cycle
/// instruction (WMMA, TRANS, MultiCycleVALU).
///
/// Window lifecycle (top-down scheduling):
///   Unpopulated → Populated (demand computed from template)
///                → Active   (producer scheduled, StartCycle/EndCycle set)
///                → Expired  (SU->TopReadyCycle >= EndCycle → rotate)
///
/// The scheduler maintains two windows: CurrentWindow and NextWindow.
/// When CurrentWindow is active and its demand is satisfied, TargetWindow
/// switches to NextWindow so the scheduler prepares fillers for the upcoming
/// producer while the current one executes.
struct CoexecWindow {
  /// The flavor of the instruction that produces this window.
  AMDGPU::InstructionFlavor ProducerFlavor = AMDGPU::InstructionFlavor::Other;
  /// Template-derived slot demand for this window.
  WindowSlotDemand Demand;
  /// Cycle at which the producer was scheduled (top-down).
  unsigned StartCycle = 0;
  /// Cycle at which the window expires (StartCycle + latency - 1).
  unsigned EndCycle = 0;
  /// Whether this window has been populated with demand info.
  bool IsPopulated = false;
  /// Whether the producer has been scheduled (window is "open").
  bool IsActive = false;

  void clear() {
    ProducerFlavor = AMDGPU::InstructionFlavor::Other;
    Demand = WindowSlotDemand();
    StartCycle = 0;
    EndCycle = 0;
    IsPopulated = false;
    IsActive = false;
  }

  /// Populate this window from the best available producer in the region.
  /// If \p PreferredFlavor is a window producer, use that flavor directly.
  /// Otherwise, select the producer based on ready instructions and demand
  /// satisfaction: prefer producers that are ready and have satisfied demand,
  /// tiebreaking by fewest slots missed, then by largest window size.
  void populate(AMDGPU::InstructionFlavor PreferredFlavor,
                const SmallVectorImpl<HardwareUnitInfo> &HWUInfo,
                const RegionMixInfo &MixInfo, const SIInstrInfo &SII,
                RooflineResult &Roofline);

  /// Check if the window has expired given the current \p Cycle.
  bool isExpired(unsigned Cycle) const {
    return IsActive && Cycle >= EndCycle;
  }

  /// Activate the window: the producer has been scheduled at \p Cycle with
  /// the given \p Latency.
  void activate(unsigned Cycle, unsigned Latency) {
    IsActive = true;
    StartCycle = Cycle;
    EndCycle = Cycle + Latency - 1;
  }
};

//===----------------------------------------------------------------------===//
// Region Mix Info
//===----------------------------------------------------------------------===//

/// Tracks per-flavor instruction counts across a scheduling region.
/// Used by ShadowMix heuristics to determine when enough co-execution
/// fillers are ready to justify scheduling a window-producing instruction.
///
/// `ReadyCount[]` is a snapshot of the boundary's Available + Pending queues
/// (both DAG-ready, the latter would just stall this cycle). The snapshot is
/// lazy: invalidated whenever a node is scheduled, refreshed on first read.
/// Callers that consume `ReadyCount` MUST call `refreshFromBoundary(Zone)`
/// first, or the value is stale.
struct RegionMixInfo {
  unsigned ReadyCount[static_cast<unsigned>(AMDGPU::InstructionFlavor::NUM_FLAVORS)] = {};
  unsigned ReadyCycles[static_cast<unsigned>(
      AMDGPU::InstructionFlavor::NUM_FLAVORS)] = {};
  unsigned ScheduledCount[static_cast<unsigned>(AMDGPU::InstructionFlavor::NUM_FLAVORS)] = {};

  void reset();
  void recordScheduled(AMDGPU::InstructionFlavor Flavor);
  void invalidate() { SnapshotDirty = true; }
  void refreshFromBoundary(SchedBoundary &Zone, const SIInstrInfo &SII,
                           CandidateHeuristics *Heurs);

  unsigned getReadyCount(AMDGPU::InstructionFlavor F) const {
    return ReadyCount[static_cast<unsigned>(F)];
  }

  unsigned getReadyCycles(AMDGPU::InstructionFlavor F) const {
    return ReadyCycles[static_cast<unsigned>(F)];
  }

private:
  bool SnapshotDirty = true;
};

//===----------------------------------------------------------------------===//
// Candidate Heuristics
//===----------------------------------------------------------------------===//

/// CandidateHeuristics contains state and implementations to facilitate making
/// per instruction scheduling decisions; it contains methods used in
/// tryCandidate to decide which instruction to schedule next.
class CandidateHeuristics {
protected:
  ScheduleDAGMI *DAG;
  const SIInstrInfo *SII;
  const SIRegisterInfo *SRI;
  const TargetSchedModel *SchedModel;
  SmallVector<HardwareUnitInfo, 8> HWUInfo;
  AMDGPU::CarriedLatency RegionCarriedLatency = AMDGPU::CarriedLatency::Off;
  DenseMap<MachineInstr *, unsigned> CarriedLatencies;
  RegionMixInfo MixInfo;
  /// Current co-execution window being filled/active.
  CoexecWindow CurrentWindow;
  /// Next window — populated for lookahead when CurrentWindow is satisfied.
  CoexecWindow NextWindow;

  /// Recorded next-iter consumer position per loop-carried lane slice.
  /// Keyed by vreg, each entry holds a small list of (lane mask, slot)
  /// pairs — one entry per LC slice on that vreg that has been observed
  /// at the top of the next iteration. The SlotIndex is the consumer MI's
  /// register slot at the time the slice was discovered; ScheduleDAGMI
  /// keeps LIS up to date as instructions are moved, so SlotIndex order
  /// reflects scheduled program order. Masks here are a subset of the
  /// masks in LoopCarriedDSReadDefLanes for the same Reg.
  DenseMap<Register, SmallVector<std::pair<LaneBitmask, SlotIndex>, 2>>
      LoopTopVGPRConsumerPos;

  /// Lane slices on each vreg that are defined by a DS_READ in the current
  /// region's MBB and whose only in-MBB uses precede that DS_READ in MIR
  /// order (so the real consumer crosses the back-edge). Slices for the
  /// same Reg are pairwise disjoint.
  DenseMap<Register, SmallVector<LaneBitmask, 2>> LoopCarriedDSReadDefLanes;

  /// Total number of LC slices across all vregs (sum of inner vector
  /// sizes in LoopCarriedDSReadDefLanes). Set once by the seeder; used
  /// by recordLoopTopConsumption to early-exit once every slice has a
  /// recorded next-iter consumer slot.
  unsigned LoopCarriedSliceCount = 0;

  /// Running total of LC slices that have been recorded with a next-iter
  /// consumer slot (one per (Reg, mask) entry in LoopTopVGPRConsumerPos).
  /// Drives the early-exit in recordLoopTopConsumption.
  unsigned RecordedConsumerSliceCount = 0;

  /// Reset cross-iter DSRead consumption-tracking state.
  void resetDSReadByConsumerOrderState() {
    LoopTopVGPRConsumerPos.clear();
    LoopCarriedDSReadDefLanes.clear();
    LoopCarriedSliceCount = 0;
    RecordedConsumerSliceCount = 0;
  }

  /// Walk the current region's MBB and populate LoopCarriedDSReadDefs.
  void seedLoopCarriedDSReadDefs();

  /// Record consumption of DSRead along the backedge for wait count
  /// reduction as we schedule instructions.
  void recordLoopTopConsumption(SUnit *SU);

  // Treat structural, latency, buffer-full, carried-latency, and fence stalls
  // as a single scheduling cost for the current cycle.
  struct StallCosts {
    unsigned Ready = 0;
    unsigned Structural = 0;
    unsigned Latency = 0;
    unsigned Carried = 0;
    unsigned Buffer = 0;
    unsigned Fence = 0;
    unsigned RAWVdst = 0;
    unsigned WARVdst = 0;
    unsigned Effective = 0;
  };

  /// Walk over the region and collect characteristics for the various
  /// heuristics.
  void collectRegionSummary();

  /// \returns the maximum blocking cycles according to the SchedModel for a
  /// given MCSchedClassDesc \p SC
  unsigned getMaxBlockingCycles(const MCSchedClassDesc *SC,
                                const MachineInstr *MI);

  /// Compute the roofline co-execution analysis for the current region.
  /// Aggregates WMMA co-exec slots by compatibility set and consumer
  /// instructions by class, then solves a bipartite max-flow to find
  /// the maximum number of fillable slots.
  void computeRooflineCoExec();

  /// Populate per-HWUI RemainingExposed using a hand-ordered allocation
  /// (DS, SALU, TRANS, VALU compete for WMMA shadow + MultiVALU shadow).
  /// Mirrors the prior PipelinedScheduler heuristic, including the DSBound
  /// guard which leaves all exposed counts at 0 for DS-bound regions.
  void initExposedGreedy();

  /// Populate per-HWUI RemainingExposed from RooflineResult::ExposedByClass
  /// (per-class flow recovered from the max-flow solver). Covers every
  /// flavor mapped by flavorToCoExecMask.
  void initExposedRoofline();

  RooflineResult Roofline;

  /// Estimate the block carried latency from loads for a given \p SU. This is
  /// essentially global scheduling info that our local scheduling
  /// infrastructure lacks the necessary infrastructure to accurately measure.
  /// Thus, this method just attempts to find a reasonable upper bound for
  /// carried load latency to avoid long stalls.
  unsigned getCarriedLatency(SUnit *SU);

  /// Count how many successors of \p SU match \p TargetFlavor and would become
  /// ready (NumPredsLeft == 1) if \p SU were scheduled.
  unsigned countDirectlyEnabledByFlavor(SUnit *SU,
                                        AMDGPU::InstructionFlavor TargetFlavor);

  /// BFS to find the nearest pending SU matching \p TargetFlavor reachable from
  /// successors of instructions in the DAG. Returns the distance or nullopt.
  std::optional<unsigned>
  findNearestPendingByFlavor(AMDGPU::InstructionFlavor TargetFlavor,
                             unsigned MaxDepth);

  /// Returns true if scheduling \p SU would directly or transitively help
  /// enable \p TargetSU.
  bool wouldHelpEnable(SUnit *SU, SUnit *TargetSU);

  /// MachineLoopInfo for the current function. May be null; only consulted
  /// by the loop-body gate of the DS_READ-by-consumer-order tie-breaker,
  /// which falls back to a self-edge check when MLI is unavailable.
  const MachineLoopInfo *MLI = nullptr;

  /// Node-visit budget for the ShadowMix lookahead BFS
  /// (findNearestPendingByFlavor). 0 = skip the lookahead entirely;
  /// UINT_MAX = unbounded. Derived per-region from the coexec lookahead
  /// effort level (see getCoexecLookaheadEffort).
  unsigned ShadowMixNodeBudget = 32;

  /// Whether the critical-resource-dependency reachability heuristic runs.
  /// Disabled only at the "off" effort level.
  bool CriticalResourceDepEnabled = true;

  /// Region-lifetime cache of ancestor sets keyed by target SUnit: maps a
  /// TargetSU to the NodeNums of every SUnit that can reach it (its
  /// predecessor closure). Used to answer DAG->IsReachable(TargetSU, X)
  /// == "does X enable TargetSU" in O(1) after one BFS, instead of the
  /// per-query DFS that LLVM re-runs for every positive result. DAG
  /// dependency edges are immutable during a region's list scheduling, so
  /// the closure is valid for the whole region; cleared in initialize().
  mutable DenseMap<const SUnit *, DenseSet<unsigned>> AncestorCache;

  /// \returns true if scheduling \p Cand would enable \p TargetSU, i.e.
  /// \p Cand is in the predecessor closure of \p TargetSU. Equivalent to
  /// DAG->IsReachable(TargetSU, Cand) but memoized via AncestorCache.
  bool candEnablesTarget(const SUnit *Cand, const SUnit *TargetSU) const;

public:
  CandidateHeuristics() = default;

  void initialize(ScheduleDAGMI *DAG, const TargetSchedModel *SchedModel,
                  const TargetRegisterInfo *TRI,
                  const MachineLoopInfo *MLI = nullptr);

  /// Given a \p Flavor , find the corresponding HardwareUnit. \returns the
  /// mapped HardwareUnit.
  HardwareUnitInfo *getHWUIFromFlavor(AMDGPU::InstructionFlavor Flavor);

  /// Compute the blocking cycles for the appropriate HardwareUnit given an \p
  /// SU
  unsigned getHWUICyclesForSU(SUnit *SU);
  /// Compute the blocking cycles for the appropriate HardwareUnit given an \p
  /// MI
  unsigned getHWUICyclesForMI(MachineInstr *MI);

  /// Update the state to reflect that \p SU is going to be scheduled.
  /// \p Zone provides cycle information for window lifecycle management.
  void updateForScheduling(SUnit *SU, SchedBoundary *Zone);

  /// Sort the HWUInfo vector. After sorting, the HardwareUnits that are highest
  /// priority are first. Priority is determined by maximizing coexecution and
  /// keeping the critical HardwareUnit busy.
  void sortHWUIResources(SchedBoundary *Zone, bool UseDependencySort = false);

  unsigned getStructuralStallCycles(SchedBoundary &Zone, SUnit *SU);

  unsigned getStallCosts(SUnit *SU, SchedBoundary &Zone, StallCosts &Costs);

  /// Calculate the number of missed slots if we schedule SU now.
  /// Currently only handles the case where we're in an active WMMA coexec
  /// window, the next slot is an I slot, and SU is a DS instruction.
  /// Returns the count of I-slots that would be missed (wasted).
  unsigned getMissedSlotCost(SUnit *SU, SchedBoundary &Zone);

  bool tryEffectiveStall(GenericSchedulerBase::SchedCandidate &TryCand,
                         GenericSchedulerBase::SchedCandidate &Cand,
                         SchedBoundary &Zone);
  /// If we are in an active coexecution slot that has preferences and / or
  /// avoidances tryCoexecSlot will try to honor that by prefering the
  /// preferences and avoiding the avoidances.
  bool tryCoexecSlot(GenericSchedulerBase::SchedCandidate &TryCand,
                     GenericSchedulerBase::SchedCandidate &Cand,
                     SchedBoundary *Zone);

  /// Prioritize instructions involved the memory pipeline. Currently we don't have
  /// any modelling of pipelined loads, so we control the layout of the pipeline
  /// per iteration by giving the user some control over the stalls (e.g. between
  /// s_barrier_signal and s_barrier_wait) and scheduling the pipeline instructions
  /// as soon as they are ready.
  ///
  /// TODO -- add better modelling and heuristics for pipelining based scheduling.
  bool tryMemoryPipeline(GenericSchedulerBase::SchedCandidate &TryCand,
                         GenericSchedulerBase::SchedCandidate &Cand,
                         SchedBoundary *Zone);

  /// Get the roofline co-execution analysis result for the current region.
  const RooflineResult &getRooflineResult() const { return Roofline; }

  /// Check for critical resource consumption. Prefer the candidate that uses
  /// the most prioritized HardwareUnit. If both candidates use the same
  /// HarwareUnit, prefer the candidate with higher priority on that
  /// HardwareUnit.
  bool tryCriticalResource(GenericSchedulerBase::SchedCandidate &TryCand,
                           GenericSchedulerBase::SchedCandidate &Cand,
                           SchedBoundary *Zone) const;

  /// Check for dependencies of instructions that use prioritized HardwareUnits.
  /// Prefer the candidate that is a dependency of an instruction that uses the
  /// most prioritized HardwareUnit. If both candidates enable the same
  /// HardwareUnit, prefer the candidate that enables the higher priority
  /// instruction on that HardwareUnit.
  bool
  tryCriticalResourceDependency(GenericSchedulerBase::SchedCandidate &TryCand,
                                GenericSchedulerBase::SchedCandidate &Cand,
                                SchedBoundary *Zone) const;

  bool tryCriticalResourcePrio(GenericSchedulerBase::SchedCandidate &TryCand,
                               GenericSchedulerBase::SchedCandidate &Cand,
                               SchedBoundary *Zone) const;

  /// ShadowMix heuristic: prefer window-producing instructions (WMMA, TRANS,
  /// MultiCycleVALU) over compatible fillers so fillers execute in the
  /// producer's shadow. When fillers are insufficient, steer enablement
  /// toward the most deficient filler flavor. When fillers are sufficient,
  /// promote the producer.
  bool tryShadowMix(GenericSchedulerBase::SchedCandidate &TryCand,
                    GenericSchedulerBase::SchedCandidate &Cand,
                    SchedBoundary *Zone);

  /// Tie-break between two ready DS_READ candidates by the recorded
  /// next-iter consumer position. Called from tryCriticalResourceDependency
  /// as the cross-loop-edge extension to its in-iter DS_READ → WMMA
  /// prioritization. Sets Reason = RegCritical on the winner.
  bool
  tryLoopCarriedDSReadOrder(GenericSchedulerBase::SchedCandidate &TryCand,
                            GenericSchedulerBase::SchedCandidate &Cand) const;

  void dumpRegionSummary();
};

class AMDGPUCoExecSchedStrategy final : public GCNSchedStrategy {
protected:
  AMDGPU::AMDGPUSchedReason LastAMDGPUReason = AMDGPU::AMDGPUSchedReason::None;
  CandidateHeuristics Heurs;

#ifndef NDEBUG
  void dumpPickSummary(SUnit *SU, bool IsTopNode, SchedCandidate &Cand);
#endif

  bool tryCandidateCoexec(SchedCandidate &Cand, SchedCandidate &TryCand,
                          SchedBoundary *Zone);
  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand, bool &PickedPending,
                         bool IsBottomUp);

public:
  AMDGPUCoExecSchedStrategy(const MachineSchedContext *C);

  void initPolicy(MachineBasicBlock::iterator Begin,
                  MachineBasicBlock::iterator End,
                  unsigned NumRegionInstrs) override;
  void initialize(ScheduleDAGMI *DAG) override;
  SUnit *pickNode(bool &IsTopNode) override;
  void schedNode(SUnit *SU, bool IsTopNode) override;
};

ScheduleDAGInstrs *createGCNCoExecMachineScheduler(MachineSchedContext *C);
ScheduleDAGInstrs *createGCNNoopPostMachineScheduler(MachineSchedContext *C);

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECSCHEDSTRATEGY_H
