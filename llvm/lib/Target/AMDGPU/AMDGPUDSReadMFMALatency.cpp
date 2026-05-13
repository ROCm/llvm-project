//===--- AMDGPUDSReadMFMALatency.cpp - AMDGPU DSRead-MFMA Latency Adjustment ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to adjust the
///       latency of data edges between DS_READ and MFMA instructions
///       when the kernel is memory bound and DS_READs are liable to
///       take longer latency than the normal scheduling model.
//
//===---------------------------------------------------------------------------===//

#include "AMDGPUDSReadMFMALatency.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "AMDGPUMachineFunctionInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

#define DEBUG_TYPE "dsread mfma latency"

using namespace llvm;

namespace {

static bool isDSReadB128OrPureDSReadB128Bundle(const MachineInstr *MI) {
  if (MI->getOpcode() == AMDGPU::DS_READ_B128_gfx9)
    return true;

  if (MI->getOpcode() != TargetOpcode::BUNDLE)
    return false;

  MachineBasicBlock::const_instr_iterator It = MI->getIterator();
  ++It; // first instruction after the BUNDLE header

  bool SawAny = false;
  for (; It != MI->getParent()->instr_end() && It->isBundledWithPred(); ++It) {
    SawAny = true;
    if (It->getOpcode() != AMDGPU::DS_READ_B128_gfx9)
      return false;
  }

  return SawAny;
}

unsigned EstimateMaxSchedulableGap(unsigned CritPathLen, const SUnit &A,
                                   const SUnit &B) {
  // Assuming A and B have dependence and B in A.succs
  // MaxGap ~= ALAP(B) - ASAP(A)
  //         = (CritPathLen - height(B)) - depth(A)
  //
  // CritPathLen - height(B) calculates the latest cycle at
  //               which B can be scheduled
  // depth(A) calculates the earliest cycle at which A can be scheduled
  unsigned ALAP = CritPathLen > B.getHeight() ?
                    CritPathLen - B.getHeight() : 0;
  unsigned ASAP = A.getDepth();
  unsigned Gap = ALAP > ASAP ?
                   ALAP - ASAP : 0;
  LLVM_DEBUG( 
    dbgs() << "A.depth=ASAP=" << ASAP << " CritPathLen=" << CritPathLen 
         << " B.height=" << B.getHeight() << " ALAP=" << ALAP << "\n";
  );
  return Gap;
}

unsigned ComputeNewLat(unsigned MaxSchedulableGap) {
  unsigned MaxLatency = 48;
  unsigned MinLatency = 24;
  unsigned K = 8;

  unsigned Num = (MaxLatency - MinLatency) * K;
  return MinLatency + Num / (MaxSchedulableGap + K);
}

class DSReadMFMALatency : public ScheduleDAGMutation {
  const GCNSubtarget &ST;

public:
  DSReadMFMALatency(MachineFunction *MF) : ST(MF->getSubtarget<GCNSubtarget>()) {}
  void apply(ScheduleDAGInstrs *DAG) override {
    const SIInstrInfo *TII = ST.getInstrInfo();
    const auto *MFI = DAG->MF.getInfo<AMDGPUMachineFunctionInfo>();
    bool IsMemBound = MFI->MemoryBound;
    if (!IsMemBound)
      return;

    unsigned CritPathLen = 0;
    for (SUnit &SU : DAG->SUnits) {
      if (SU.getDepth() + SU.getHeight() > 0)
        CritPathLen = std::max(CritPathLen, SU.getDepth() + SU.getHeight() - 1);
    }
    // Go through SU's and raise latencies on both edges between
    // DS_READ and MFMA
    LLVM_DEBUG(
      dbgs() << "Applying DSRead-MFMA DAG mutations..\n";
    );
    for (SUnit &SU : DAG->SUnits) {
      MachineInstr *MI = SU.getInstr();
      if (!MI)
        continue;

      if (isDSReadB128OrPureDSReadB128Bundle(MI)) {
        for (SDep &SuccDep : SU.Succs) {
          if (SuccDep.isCtrl())
            continue;

          SUnit *DstSU = SuccDep.getSUnit();
          MachineInstr *DstMI = DstSU ? DstSU->getInstr() : nullptr;
          if (!DstMI)
            continue;

          if (!TII->isMFMA(*DstMI))
            continue;

          unsigned MaxGap =
              EstimateMaxSchedulableGap(CritPathLen, SU, *DstSU);
          unsigned OldLat = SuccDep.getLatency();
          unsigned NewLat = ComputeNewLat(MaxGap);
          SuccDep.setLatency(NewLat);

          LLVM_DEBUG(dbgs() << "New Latency: SU(" << SU.NodeNum
                            << ")->SU(" << DstSU->NodeNum << ")="
                            << OldLat << "->" << NewLat
                            << ". MaxGap=" << MaxGap << "\n";);
        }
      } else if (TII->isMFMA(*MI)) {
        for (SDep &PredDep : SU.Preds) {
          if (PredDep.isCtrl())
            continue;

          SUnit *SrcSU = PredDep.getSUnit();
          MachineInstr *SrcMI = SrcSU ? SrcSU->getInstr() : nullptr;
          if (!SrcMI)
            continue;

          if (SrcMI->getOpcode() == AMDGPU::DS_READ_B128)
            continue;

          unsigned MaxGap =
              EstimateMaxSchedulableGap(CritPathLen, *SrcSU, SU);
          unsigned OldLat = PredDep.getLatency();
          unsigned NewLat = ComputeNewLat(MaxGap);
          PredDep.setLatency(NewLat);

          LLVM_DEBUG(dbgs() << "New Latency: SU(" << SrcSU->NodeNum
                            << ")<-SU(" << SU.NodeNum << ")="
                            << OldLat << "->" << NewLat 
                            << ". MaxGap=" << MaxGap << "\n";);
        }
      }
    }
  }
};

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUDSReadMFMALatencyDAGMutation(MachineFunction *MF) {
  return std::make_unique<DSReadMFMALatency>(MF);
}
//
