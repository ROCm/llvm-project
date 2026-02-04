//===- AMDGPUStaticSimulator.h - Static Performance Simulator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Static simulator for AMDGPU kernels that estimates performance metrics
/// without running on hardware. Produces instruction counts, stall estimates,
/// and efficiency metrics as assembly comments.
///
/// Uses the AMDGPUSim library for core simulation logic.
/// Currently enabled only for gfx1250. Target workloads: GEMM, Flash Attention.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSTATICSIMULATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSTATICSIMULATOR_H

#include "AMDGPUSim/AMDGPUSim.h"
#include "AMDGPUSim/MIRAdapter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class MachineFunction;
class MachineBasicBlock;
class MachineInstr;
class GCNSubtarget;

namespace AMDGPU {

// Import commonly used types from the AMDGPUSim library
using AMDGPUSim::FunctionalUnit;
using AMDGPUSim::getInstClassName;
using AMDGPUSim::getUnitName;
using AMDGPUSim::GPUSimState;
using AMDGPUSim::HWModel;
using AMDGPUSim::InstClass;
using AMDGPUSim::InstrSimInfo;
using AMDGPUSim::MachineInstrInfo;
using AMDGPUSim::PendingMemOp;
using AMDGPUSim::SimInst;
using AMDGPUSim::Simulator;
using AMDGPUSim::SimulatorConfig;
using AMDGPUSim::StallBreakdown;
using AMDGPUSim::StallReason;

//===----------------------------------------------------------------------===//
// BlockMetrics - Per-block accumulated counts and cycles
//===----------------------------------------------------------------------===//

/// Metrics accumulated for a basic block or region. All fields are additive.
struct BlockMetrics {
  // Instruction Counts
  unsigned NumInstructions = 0;
  unsigned NumVALU = 0;
  unsigned NumSALU = 0;
  unsigned NumTRANS = 0;
  unsigned NumWMMA = 0;
  unsigned NumVOPD = 0;
  unsigned NumPacked = 0;
  unsigned NumDSRead = 0;
  unsigned NumDSWrite = 0;
  unsigned NumVMEM = 0;
  unsigned NumSMEM = 0;
  unsigned NumTDM = 0;
  unsigned NumBranch = 0;
  unsigned NumBarrier = 0;
  unsigned NumNop = 0;
  unsigned NumDelayAlu = 0;
  unsigned NumMSBSet = 0;
  unsigned NumMSBSetExposed = 0;
  unsigned NumMSBSetMasked = 0;
  unsigned NumSpill = 0;
  unsigned NumReload = 0;
  unsigned NumSGPRToVGPR = 0;
  unsigned NumVGPRToSGPR = 0;

  // Wait Counts
  unsigned NumWaitcnt = 0;
  unsigned WaitLGKM = 0;
  unsigned WaitVMEM = 0;
  unsigned WaitEXP = 0;
  unsigned NumFalseWaits = 0;

  // Cycle Estimates
  unsigned TotalCycles = 0;

  // Stall Breakdown
  unsigned StallFunctionalUnit = 0;
  unsigned StallCoExec = 0;
  unsigned StallDelayAlu = 0;
  unsigned StallMemFIFO = 0;
  unsigned StallWaitCnt = 0;
  unsigned StallFalseWait = 0;

  // Per-unit stall breakdown
  unsigned StallXDL = 0;
  unsigned StallVALU = 0;
  unsigned StallSALU = 0;
  unsigned StallTRANSUnit = 0;
  unsigned StallLDS = 0;
  unsigned StallVMEMUnit = 0;

  unsigned StallRegBankConflict = 0;
  unsigned RegBankConflictsInWMMAWindow = 0;
  unsigned StallLongLatVALU = 0;
  unsigned StallLOLVALUTRANS = 0;
  unsigned StallVaSSRC = 0;
  unsigned StallVaVdst = 0;
  unsigned StallRAW = 0;
  unsigned StallISFetch = 0;
  unsigned ISFetchesTriggered = 0;
  unsigned ISBytesConsumed = 0;

  unsigned VGPRCacheHits = 0;
  unsigned VGPRCacheMisses = 0;
  unsigned VGPRCacheEvictions = 0;

  float VGPRCacheHitRate() const {
    unsigned Total = VGPRCacheHits + VGPRCacheMisses;
    return Total > 0 ? static_cast<float>(VGPRCacheHits) / Total : 0.0f;
  }

  unsigned StallCycles() const {
    return NumMSBSetExposed + StallFunctionalUnit + StallCoExec +
           StallDelayAlu + StallMemFIFO + StallWaitCnt + StallRegBankConflict +
           StallLOLVALUTRANS + StallVaSSRC + StallVaVdst + StallRAW +
           StallISFetch;
  }

  // WMMA Co-execution
  unsigned WMMAWindowCycles = 0;
  unsigned WMMACoExecUsed = 0;
  unsigned WMMACoExecBlocked = 0;
  unsigned WMMAStarved = 0;

  // Co-exec miss breakdown by instruction class
  unsigned CoExecMissVALU = 0;
  unsigned CoExecMissTRANS = 0;
  unsigned CoExecMissMemory = 0;
  unsigned CoExecMissOther = 0;

  // I-slot utilization
  unsigned ISlotTotal = 0;
  unsigned ISlotUsedByVALU = 0;
  unsigned ISlotWastedOnNonVALU = 0;

  /// Scale all metrics by a factor
  BlockMetrics operator*(float Factor) const;
  friend BlockMetrics operator*(float Factor, const BlockMetrics &M) {
    return M * Factor;
  }

  /// Sum two metric sets
  BlockMetrics operator+(const BlockMetrics &O) const;

  // === Formatting helpers ===
  void printInstBreakdown(raw_ostream &OS) const;
  void printStallBreakdown(raw_ostream &OS) const;
  void printFUBreakdown(raw_ostream &OS) const;
};

//===----------------------------------------------------------------------===//
// PerBlockInfo - Per-block metrics for assembly output
//===----------------------------------------------------------------------===//

struct PerBlockInfo {
  BlockMetrics Cold;
  BlockMetrics Warm;
  unsigned TripCount = 1;
  float Frequency = 1.0f;
  bool IsLoopHeader = false;
  bool InLoop = false;

  BlockMetrics getScaled() const {
    if (TripCount <= 1)
      return Cold;
    return Cold + Warm * (TripCount - 1);
  }
};

//===----------------------------------------------------------------------===//
// KernelPerfReport - Final aggregated performance report
//===----------------------------------------------------------------------===//

struct KernelPerfReport {
  BlockMetrics Raw;
  BlockMetrics Scaled;
  BlockMetrics ColdTotal;
  BlockMetrics WarmTotal;

  DenseMap<const MachineBasicBlock *, PerBlockInfo> PerBlock;
  DenseMap<const MachineInstr *, InstrSimInfo> PerInstr;

  // Derived metrics
  float IPC = 0.0f;
  float StallRatio = 0.0f;
  float CoExecEfficiency = 0.0f;
  float FalseWaitRatio = 0.0f;
  unsigned EstimatedWaves = 0;

  // CFG info
  unsigned NumLoops = 0;
  unsigned MaxLoopDepth = 0;
  unsigned MaxTripCount = 0;
  unsigned NumBranches = 0;

  std::string FunctionName;

  void finalize() {
    if (Scaled.TotalCycles > 0) {
      unsigned ComputeOps = Scaled.NumVALU + Scaled.NumSALU +
                            Scaled.NumTRANS + Scaled.NumWMMA;
      IPC = static_cast<float>(ComputeOps) / Scaled.TotalCycles;
      StallRatio =
          static_cast<float>(Scaled.StallCycles()) / Scaled.TotalCycles;
    }
    if (Scaled.WMMAWindowCycles > 0) {
      CoExecEfficiency =
          static_cast<float>(Scaled.WMMACoExecUsed) / Scaled.WMMAWindowCycles;
    }
    if (Scaled.NumWaitcnt > 0) {
      FalseWaitRatio =
          static_cast<float>(Scaled.NumFalseWaits) / Scaled.NumWaitcnt;
    }
  }

  void print(raw_ostream &OS, StringRef FuncName = "") const;
};

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSTATICSIMULATOR_H
