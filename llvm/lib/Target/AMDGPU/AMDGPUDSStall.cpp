//===--- AMDGPUDSStall.cpp - AMDGPU DS Read Interleaving  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation that adds weak order
///       edges between consecutive DS read instructions, routing them through
///       high-latency compute fillers (VALU/SALU). This discourages the
///       scheduler from placing DS reads back-to-back, hiding LDS latency
///       and reducing register pressure from live-range overlap.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUDSStall.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ds-stall"

static cl::opt<bool> DisableDSStall("amdgpu-disable-ds-stall",
  cl::desc("Disable DS read interleaving mutation"),
  cl::init(false), cl::Hidden);

namespace {

class DSStall : public ScheduleDAGMutation {
public:
  void apply(ScheduleDAGInstrs *DAG) override;
};

static bool isDSRead(const SUnit &SU) {
  const MachineInstr *MI = SU.getInstr();
  return MI && SIInstrInfo::isDS(*MI) && MI->mayLoad() && !MI->mayStore();
}

static bool isComputeFiller(const SUnit &SU) {
  const MachineInstr *MI = SU.getInstr();
  if (!MI)
    return false;
  // Exclude SOPP (s_barrier, s_cbranch_*, etc.) which pass isSALU but are
  // not useful compute work.
  return SIInstrInfo::isVALU(*MI) ||
         (SIInstrInfo::isSALU(*MI) && !SIInstrInfo::isSOPP(*MI));
}

void DSStall::apply(ScheduleDAGInstrs *DAG) {
  if (DisableDSStall)
    return;

  const TargetSchedModel *SchedModel = DAG->getSchedModel();

  if (!SchedModel->hasInstrSchedModel())
    return;

  SmallVector<SUnit *, 16> DSReads;
  for (SUnit &SU : DAG->SUnits) {
    if (isDSRead(SU))
      DSReads.push_back(&SU);
  }

  if (DSReads.size() < 2)
    return;

  LLVM_DEBUG(dbgs() << "DSStall: " << DSReads.size() << " DS reads\n");

  for (unsigned I = 0, E = DSReads.size() - 1; I < E; ++I) {
    SUnit *DSA = DSReads[I];
    SUnit *DSB = DSReads[I + 1];

    if (DAG->IsReachable(DSB, DSA))
      continue;

    SUnit *Best = nullptr;
    unsigned BestLat = 0;

    // Phase 1: Search between DSA and DSB for a compute filler.
    for (unsigned N = DSA->NodeNum + 1; N < DSB->NodeNum; ++N) {
      SUnit &Cand = DAG->SUnits[N];
      if (!isComputeFiller(Cand))
        continue;
      if (!DAG->canAddEdge(&Cand, DSA) || !DAG->canAddEdge(DSB, &Cand))
        continue;
      unsigned Lat = SchedModel->computeInstrLatency(Cand.getInstr());
      if (Lat > BestLat) {
        Best = &Cand;
        BestLat = Lat;
      }
    }

    // Phase 2: Widen to the entire scheduling region.
    if (!Best) {
      for (SUnit &Cand : DAG->SUnits) {
        if (Cand.NodeNum > DSA->NodeNum && Cand.NodeNum < DSB->NodeNum)
          continue; // already visited in Phase 1
        if (!isComputeFiller(Cand))
          continue;
        if (!DAG->canAddEdge(&Cand, DSA) || !DAG->canAddEdge(DSB, &Cand))
          continue;
        unsigned Lat = SchedModel->computeInstrLatency(Cand.getInstr());
        if (Lat > BestLat) {
          Best = &Cand;
          BestLat = Lat;
        }
      }
    }

    if (Best) {
      bool AddedAF = DAG->addEdge(Best, SDep(DSA, SDep::Weak));
      bool AddedFB = DAG->addEdge(DSB, SDep(Best, SDep::Weak));
      LLVM_DEBUG(dbgs() << "  Routed: SU(" << DSA->NodeNum << ") -> SU("
                        << Best->NodeNum << ") -> SU(" << DSB->NodeNum
                        << ") lat=" << BestLat
                        << " edges=" << AddedAF << "," << AddedFB << "\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "  Skip: no routable filler for SU("
                      << DSA->NodeNum << ") -> SU(" << DSB->NodeNum << ")\n");
  }
}

} // end anonymous namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createAMDGPUDSStallDAGMutation() {
  return std::make_unique<DSStall>();
}

} // end namespace llvm
