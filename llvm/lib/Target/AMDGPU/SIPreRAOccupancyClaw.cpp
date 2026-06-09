//===-- SIPreRAOccupancyClaw.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  ############################################################################
//  ##  TEMPORARY PROTOTYPE / TUNING SCAFFOLD - DO NOT UPSTREAM AS A PASS.     ##
//  ##  This whole standalone pass exists only to iterate on the SGPR          ##
//  ##  occupancy-claw heuristic with cl::opt knobs and instrumentation.       ##
//  ##  Before any upstream patch, FOLD this logic into an existing pre-RA     ##
//  ##  pass that already holds LiveIntervals (candidate:                      ##
//  ##  GCNPreRAOptimizations) and DELETE this file + its pipeline/registry    ##
//  ##  entries. No new codegen pass should land just for this heuristic.      ##
//  ##  See /work/mselehov/tickets/2277/REGALLOC-DESIGN.md.                    ##
//  ############################################################################
//
/// \file
/// \brief Experimental (LCOMPILER-2277) pre-RA heuristic that lets the SGPR
/// register allocator claw back occupancy lost to SGPR pressure.
///
/// On gfx90a+ a kernel can be pinned below the occupancy that LDS/workgroup
/// size already permit, just because SGPR pressure spills over an occupancy
/// granule (e.g. 81 SGPRs -> 7 waves while LDS allows 8). Surplus SGPRs can be
/// spilled very cheaply into VGPR lanes, but the allocator does not do this for
/// occupancy on its own: its SGPR budget is derived from the function's minimum
/// waves/EU, not from the achievable occupancy ceiling.
///
/// This pass runs right before the SGPR allocation phase. It compares the
/// natural SGPR pressure peak (from LiveIntervals) against the occupancy that
/// LDS/WG allow. Only when SGPR pressure actually pins occupancy below that
/// ceiling does it record a tighter SGPR budget in the MachineFunctionInfo;
/// SIRegisterInfo::getReservedRegs then reserves the surplus SGPRs so greedy
/// spills them (into VGPR lanes) to reach the target occupancy.
///
/// The pass only sets a budget; the existing greedy allocator +
/// SILowerSGPRSpills machinery do the actual spilling. It is gated behind
/// -amdgpu-enable-sgpr-occupancy-claw (default off) and only fires on gfx90a+.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-sgpr-occupancy-claw"

static cl::opt<bool> EnableSGPROccupancyClaw(
    // NB: option name must differ from the pass arg/DEBUG_TYPE
    // ("amdgpu-sgpr-occupancy-claw"); opt's legacy PassNameParser turns the pass
    // arg into a cl flag, so a same-named cl::opt aborts opt with "registered
    // more than once".
    "amdgpu-enable-sgpr-occupancy-claw", cl::init(false), cl::ReallyHidden,
    cl::desc("Experimental: spill surplus SGPRs (into VGPR lanes) to preserve "
             "the occupancy that LDS/WG already allow (gfx90a+)"));

// Testing/tuning knob: force a target occupancy, bypassing the natural-pressure
// profitability gate. 0 = disabled (use the heuristic). Lets us exercise the
// claw on standalone IR that does not reproduce the full-pipeline cliff.
static cl::opt<unsigned> ForceClawOccupancy(
    "amdgpu-sgpr-claw-force-occupancy", cl::init(0), cl::ReallyHidden,
    cl::desc("Force the SGPR occupancy-claw target occupancy (0 = use the "
             "heuristic)"));

// Cost gate: maximum estimated dynamic spill traffic (writelane+readlane
// executions, in units of entry-block executions) the claw is willing to pay.
// Negative = report only / do not gate on cost (used to gather numbers before
// picking a real threshold).
static cl::opt<double> MaxClawCost(
    "amdgpu-sgpr-claw-max-cost", cl::init(-1.0), cl::ReallyHidden,
    cl::desc("Max estimated dynamic spill traffic (entry-frequency units) the "
             "SGPR occupancy claw will pay; negative = do not gate on cost"));

namespace {

struct PressureInfo {
  unsigned MaxSGPR = 0;
  unsigned MaxVGPR = 0;
  // SGPR live set at the max-SGPR-pressure point (the binding peak); the
  // candidate set the allocator would spill from to relieve SGPR pressure.
  GCNRPTracker::LiveRegSet PeakLive;
};

/// Estimate the peak number of simultaneously-live SGPRs and VGPRs across the
/// function (the two peaks may occur at different program points), and capture
/// the live set at the SGPR peak.
static PressureInfo computeMaxPressure(const MachineFunction &MF,
                                       const LiveIntervals &LIS) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  // gfx90a+ uses a unified VGPR file; we only run there.
  const bool UnifiedVGPR = MF.getSubtarget<GCNSubtarget>().hasGFX90AInsts();
  PressureInfo PI;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      GCNRPTracker::LiveRegSet Live = getLiveRegsAfter(MI, LIS);
      GCNRegPressure RP = getRegPressure(MRI, Live);
      if (RP.getSGPRNum() > PI.MaxSGPR) {
        PI.MaxSGPR = RP.getSGPRNum();
        PI.PeakLive = Live;
      }
      PI.MaxVGPR = std::max(PI.MaxVGPR, RP.getVGPRNum(UnifiedVGPR));
    }
  }
  return PI;
}

/// Estimate the dynamic spill traffic (writelane + readlane executions, in
/// units of entry-block executions) of clawing the SGPR peak down by \p
/// Overflow registers. The allocator spills the cheapest (coldest) ranges
/// first, so we sum the def/use block frequencies of the cheapest SGPR ranges
/// live at the peak until their combined width covers the overflow. Working per
/// range (not per peak block) captures the "cold range crossing a hot block"
/// case: such a range is cheap to spill even though the peak itself is hot.
static double estimateSpillCost(const MachineFunction &MF,
                               const MachineBlockFrequencyInfo &MBFI,
                               const SIRegisterInfo &TRI,
                               const GCNRPTracker::LiveRegSet &PeakLive,
                               unsigned Overflow) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  struct Cand {
    double Cost; // dynamic writelane/readlane executions
    unsigned Width;
  };
  SmallVector<Cand, 32> Cands;
  for (auto [RegU, Mask] : PeakLive) {
    Register Reg(RegU);
    if (!Reg.isVirtual())
      continue;
    const TargetRegisterClass *RC = MRI.getRegClass(Reg);
    if (!TRI.isSGPRClass(RC))
      continue;
    unsigned Width = TRI.getRegSizeInBits(*RC) / 32;
    double Cost = 0.0;
    for (const MachineInstr &MI : MRI.reg_nodbg_instructions(Reg))
      Cost += double(MBFI.getBlockFreq(MI.getParent()).getFrequency());
    Cands.push_back({Cost, Width});
  }
  // Cheapest pressure relief first (lowest cost per reclaimed register).
  llvm::sort(Cands, [](const Cand &A, const Cand &B) {
    return A.Cost * B.Width < B.Cost * A.Width;
  });
  double Total = 0.0;
  unsigned Relieved = 0;
  for (const Cand &C : Cands) {
    if (Relieved >= Overflow)
      break;
    Total += C.Cost;
    Relieved += C.Width;
  }
  uint64_t EntryFreq = MBFI.getEntryFreq().getFrequency();
  return EntryFreq ? Total / double(EntryFreq) : Total;
}

/// Net SGPR budget that yields \p Occ waves/EU, mirroring
/// GCNSubtarget::getMaxNumSGPRs(MF) (i.e. minus the reserved special SGPRs).
static unsigned getNetNumSGPRsForOccupancy(const GCNSubtarget &ST,
                                           const MachineFunction &MF,
                                           unsigned Occ) {
  unsigned Reserved = ST.getReservedNumSGPRs(MF);
  unsigned Budget = ST.getMaxNumSGPRs(Occ, /*Addressable=*/false);
  return std::min(Budget > Reserved ? Budget - Reserved : 0u,
                  ST.getMaxNumSGPRs(Occ, /*Addressable=*/true));
}

class SIPreRAOccupancyClaw {
public:
  bool run(MachineFunction &MF, const LiveIntervals &LIS,
           const MachineBlockFrequencyInfo &MBFI);
};

bool SIPreRAOccupancyClaw::run(MachineFunction &MF, const LiveIntervals &LIS,
                               const MachineBlockFrequencyInfo &MBFI) {
  if (!EnableSGPROccupancyClaw)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  // SGPRs can only limit occupancy on pre-GFX10; we restrict to gfx90a+ (MI200
  // and newer), the generations the compiler officially targets for perf.
  if (!ST.hasGFX90AInsts())
    return false;

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  PressureInfo PI = computeMaxPressure(MF, LIS);

  // The generic pre-RA register-pressure tracker UNDER-counts relative to the
  // real allocation (live-range splits, copies, alignment/fragmentation: e.g. a
  // measured SGPR peak of 80 came out as 90 allocated). The machine scheduler
  // compensates with a hardcoded safety margin (GCNSchedStrategy::ErrorMargin,
  // subtracted from its register limits). Reuse that SAME margin here instead of
  // inventing another one: treat the measured peaks as this much higher before
  // trusting them for the ceiling, the VGPR-feasibility check and the overflow.
  // TODO(upstream): promote GCNSchedStrategy::ErrorMargin to a shared constant
  // and reference it directly rather than mirroring the value.
  const unsigned ErrorMargin = 3; // keep in sync with GCNSchedStrategy.h
  unsigned PeakSGPR = PI.MaxSGPR + ErrorMargin;
  unsigned PeakVGPR = PI.MaxVGPR + ErrorMargin;

  // Occupancy ceiling that everything EXCEPT SGPR pressure allows, reusing the
  // existing accounting:
  //  - MFI->getWavesPerEU().second already folds in LDS, workgroup size AND the
  //    amdgpu-waves-per-eu attribute (via getEffectiveWavesPerEU).
  //  - getOccupancyWithNumVGPRs() adds the VGPR-pressure cap.
  unsigned DynVGPRBlockSize =
      AMDGPU::getDynamicVGPRBlockSize(MF.getFunction());
  unsigned Ceiling = std::min(MFI->getWavesPerEU().second,
                              ST.getOccupancyWithNumVGPRs(PeakVGPR,
                                                          DynVGPRBlockSize));

  // We do NOT predict whether SGPRs pin occupancy below the ceiling from the
  // (optimistic) pre-RA pressure. Instead we request a target occupancy and let
  // the allocator react to the real numbers: it spills only if natural pressure
  // does not fit budget(target), and does nothing otherwise (never-loosen).
  //
  // Occupancy ladder: SGPRs admit only a few occupancy levels, and the spill
  // cost to reach a level grows as the target rises (lower target => smaller
  // overflow => fewer/cheaper ranges spilled). So walk DOWN from the ceiling and
  // take the HIGHEST occupancy whose spill traffic is affordable, rather than
  // bailing all the way to natural when the ceiling alone is too hot ("can't
  // afford 8, but 7 is cheap" still wins +1 wave). force-occupancy pins a single
  // level (testing); cost gate disabled when MaxClawCost is negative.
  unsigned HighestOcc = ForceClawOccupancy ? ForceClawOccupancy : Ceiling;
  unsigned FloorOcc = ForceClawOccupancy ? ForceClawOccupancy : 1;
  LLVM_DEBUG(dbgs() << "[sgpr-claw] " << MF.getName() << ": peakSGPR=" << PeakSGPR
                    << " peakVGPR=" << PeakVGPR << " ceiling=" << Ceiling
                    << " (wavesPerEU.max=" << MFI->getWavesPerEU().second
                    << ")\n");

  for (unsigned TargetOcc = HighestOcc; TargetOcc >= FloorOcc; --TargetOcc) {
    unsigned ClawNumSGPRs = getNetNumSGPRsForOccupancy(ST, MF, TargetOcc);
    // This level does not tighten the budget, so no lower level will either.
    if (ClawNumSGPRs >= ST.getMaxNumSGPRs(MF))
      break;

    unsigned Overflow = PeakSGPR > ClawNumSGPRs ? PeakSGPR - ClawNumSGPRs : 0;

    // VGPR feasibility: the SGPR spills land in VGPR lanes, so they RAISE VGPR
    // pressure. Reuse the backend's canonical estimate of that cost
    // (RegExcess in GCNRegPressure.cpp: divideCeil(excessSGPR, wavefrontSize)
    // extra VGPRs) and the canonical VGPR->occupancy map. PeakVGPR already
    // carries the ErrorMargin above. Skip this target if holding the spills
    // would itself pull VGPR occupancy below it (we would pay the spill traffic
    // and not gain the wave - or worse, force VGPR spills).
    unsigned VGPRForSGPRSpills = divideCeil(Overflow, ST.getWavefrontSize());
    if (ST.getOccupancyWithNumVGPRs(PeakVGPR + VGPRForSGPRSpills,
                                    DynVGPRBlockSize) < TargetOcc) {
      LLVM_DEBUG(dbgs() << "[sgpr-claw] " << MF.getName() << ": occ " << TargetOcc
                        << " not VGPR-feasible (peakVGPR " << PeakVGPR << " + "
                        << VGPRForSGPRSpills << " spill VGPRs), trying lower\n");
      continue;
    }

    double Cost = estimateSpillCost(MF, MBFI, *ST.getRegisterInfo(), PI.PeakLive,
                                    Overflow);
    LLVM_DEBUG(dbgs() << "[sgpr-claw] " << MF.getName() << ": try occ="
                      << TargetOcc << " SGPR budget " << ST.getMaxNumSGPRs(MF)
                      << " -> " << ClawNumSGPRs << " overflow=" << Overflow
                      << " estSpillCost=" << Cost << '\n');
    if (MaxClawCost >= 0.0 && Cost > MaxClawCost) {
      LLVM_DEBUG(dbgs() << "[sgpr-claw] " << MF.getName() << ": occ " << TargetOcc
                        << " too hot (" << Cost << " > " << MaxClawCost
                        << "), trying lower\n");
      continue;
    }
    LLVM_DEBUG(dbgs() << "[sgpr-claw] " << MF.getName() << ": picked occ="
                      << TargetOcc << " budget=" << ClawNumSGPRs << '\n');
    MFI->setSGPRClawNumSGPRs(ClawNumSGPRs);
    return false;
  }

  return false;
}

class SIPreRAOccupancyClawLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIPreRAOccupancyClawLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    const MachineBlockFrequencyInfo &MBFI =
        getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
    return SIPreRAOccupancyClaw().run(MF, LIS, MBFI);
  }

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA SGPR Occupancy Claw";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SIPreRAOccupancyClawLegacy::ID = 0;

char &llvm::SIPreRAOccupancyClawID = SIPreRAOccupancyClawLegacy::ID;

INITIALIZE_PASS_BEGIN(SIPreRAOccupancyClawLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA SGPR Occupancy Claw", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(SIPreRAOccupancyClawLegacy, DEBUG_TYPE,
                    "AMDGPU Pre-RA SGPR Occupancy Claw", false, false)
