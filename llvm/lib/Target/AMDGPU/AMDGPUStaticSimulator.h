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
/// Currently enabled only for gfx1250. Target workloads: GEMM, Flash Attention.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSTATICSIMULATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSTATICSIMULATOR_H

#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include <array>
#include <deque>
#include <optional>

namespace llvm {

class MachineFunction;
class MachineBasicBlock;
class GCNSubtarget;

namespace AMDGPU {

//===----------------------------------------------------------------------===//
// Latency Constants (gfx1250 defaults)
//===----------------------------------------------------------------------===//

/// Default latency values for gfx1250.
namespace DefaultLatency {
constexpr unsigned VALU = 5;
constexpr unsigned SALU = 2;
constexpr unsigned TRANS = 8;      // Transcendental / WMMA
constexpr unsigned DS_READ = 50;   // LDS load
constexpr unsigned DS_WRITE = 8;   // LDS store
constexpr unsigned VMEM = 300;     // Global memory (conservative)
constexpr unsigned SMEM = 20;      // Scalar memory
constexpr unsigned BARRIER = 32;   // s_barrier
} // namespace DefaultLatency

//===----------------------------------------------------------------------===//
// Instruction Classification
//===----------------------------------------------------------------------===//

/// Instruction classification for simulation and latency queries.
/// Used both for counting and for latency lookups.
enum class InstClass {
  VALU,
  SALU,
  TRANS,       // Transcendentals (V_EXP, V_LOG, V_RCP, V_RSQ, V_SQRT, etc.)
  WMMA,        // Matrix multiply
  DS_READ,
  DS_WRITE,
  VMEM_READ,
  VMEM_WRITE,
  SMEM,
  TDM,         // Tensor DMA (TENSOR_LOAD_TO_LDS etc.)
  BARRIER,
  WAITCNT,
  DELAY_ALU,
  MSB_SET,     // s_set_vgpr_msb (gfx1250 overhead for >256 VGPRs)
  NOP,
  BRANCH,
  OTHER
};

/// Get the latency for an instruction class.
inline unsigned getLatencyForClass(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:       return DefaultLatency::VALU;
  case InstClass::SALU:       return DefaultLatency::SALU;
  case InstClass::TRANS:      return DefaultLatency::TRANS;
  case InstClass::WMMA:       return DefaultLatency::TRANS;  // Similar latency
  case InstClass::DS_READ:    return DefaultLatency::DS_READ;
  case InstClass::DS_WRITE:   return DefaultLatency::DS_WRITE;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE: return DefaultLatency::VMEM;
  case InstClass::SMEM:       return DefaultLatency::SMEM;
  case InstClass::TDM:        return DefaultLatency::DS_READ; // Similar to DS
  case InstClass::BARRIER:    return DefaultLatency::BARRIER;
  default:                    return 1;
  }
}

// WMMA window size is per-variant via SchedModel (8 for BF16/F16, 16 for IU8/IU4, etc.)

//===----------------------------------------------------------------------===//
// Per-Instruction Simulation Info (for assembly annotations)
//===----------------------------------------------------------------------===//

/// Reason for stall cycles on an instruction
enum class StallReason : uint8_t {
  NONE = 0,
  FU_BUSY,         // Functional unit not ready
  COEXEC_BLOCKED,  // Blocked by WMMA co-execution rules
  WAITCNT,         // Memory wait (s_wait_*)
  DELAY_ALU,       // RAW dependency (s_delay_alu)
  MEM_FIFO,        // Memory FIFO full
  MSB_SET_EXPOSED, // s_set_vgpr_msb not fused
  REG_BANK         // Register bank conflict (operands in same sub-bank)
};

/// Stage type for WMMA co-execution (for annotation display)
enum class WMMAStageType : uint8_t {
  NONE = 0,  // Not in WMMA window
  E0,        // Issue cycle - control only
  E,         // External - MEM/SALU allowed
  I,         // Internal - MEM/SALU/VALU allowed
  V          // Vacant - MEM/SALU/WMMA allowed, no VALU
};

/// Per-instruction simulation data for assembly annotation
struct InstrSimInfo {
  unsigned StallCycles = 0;
  StallReason Reason = StallReason::NONE;
  bool InWMMAWindow = false;
  uint8_t WMMAStage = 0;
  uint8_t WMMATotalWindow = 0;
  WMMAStageType StageType = WMMAStageType::NONE;
  bool CoExecuted = false;
  bool WasFused = false;
  bool WasExposed = false;
  bool IsWMMA = false;
  bool LDScaleBlocked = false;
  StringRef WMMAPattern;
  std::string CachePattern;  // e.g., "($--)" for VGPR cache hits ($=hit, -=miss)
  
  /// Get human-readable reason string
  const char *getReasonString() const {
    switch (Reason) {
    case StallReason::NONE:           return nullptr;
    case StallReason::FU_BUSY:        return "FU busy";
    case StallReason::COEXEC_BLOCKED: return "CoExec blocked";
    case StallReason::WAITCNT:        return "WaitCnt";
    case StallReason::DELAY_ALU:      return "DelayAlu";
    case StallReason::MEM_FIFO:       return "FIFO full";
    case StallReason::MSB_SET_EXPOSED: return "MSB exposed";
    case StallReason::REG_BANK:       return "RegBank conflict";
    }
    return "Unknown";
  }
  
  /// Get stage type character for compact display
  char getStageChar() const {
    switch (StageType) {
    case WMMAStageType::E0: return '0';
    case WMMAStageType::E:  return 'E';
    case WMMAStageType::I:  return 'I';
    case WMMAStageType::V:  return 'V';
    default: return '?';
    }
  }
  
  /// Get stage type name
  const char *getStageName() const {
    switch (StageType) {
    case WMMAStageType::E0: return "E0";
    case WMMAStageType::E:  return "E";
    case WMMAStageType::I:  return "I";
    case WMMAStageType::V:  return "V";
    default: return "?";
    }
  }
};

//===----------------------------------------------------------------------===//
// Functional Units
//===----------------------------------------------------------------------===//

/// Hardware functional units for per-unit busy tracking
enum class FunctionalUnit : unsigned {
  NONE = 0,   // No unit (NOPs, WAITCNTs, BARRIERs) - no busy tracking
  XDL,        // Matrix/WMMA unit
  VALU,       // Vector ALU
  SALU,       // Scalar ALU
  TRANS,      // Transcendental unit
  LDS,        // Local Data Share
  VMEM,       // Global memory unit
  SMEM,       // Scalar memory unit
  BRANCH,     // Branch unit
  NUM_UNITS
};

/// Map InstClass to FunctionalUnit
inline FunctionalUnit getUnitForClass(InstClass IC) {
  switch (IC) {
  case InstClass::WMMA:       return FunctionalUnit::XDL;
  case InstClass::VALU:       return FunctionalUnit::VALU;
  case InstClass::TRANS:      return FunctionalUnit::TRANS;
  case InstClass::SALU:
  case InstClass::DELAY_ALU:  // s_delay_alu is SALU
  case InstClass::MSB_SET:    // s_set_vgpr_msb is SALU (co-fuses with following VALU)
    return FunctionalUnit::SALU;
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::TDM:        return FunctionalUnit::LDS;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE: return FunctionalUnit::VMEM;
  case InstClass::SMEM:       return FunctionalUnit::SMEM;
  case InstClass::BRANCH:     return FunctionalUnit::BRANCH;
  // These don't occupy functional units - they're cycle padding or control flow
  // NOPs are inserted by hazard recognizer to resolve stalls, not cause new ones
  case InstClass::NOP:
  case InstClass::WAITCNT:
  case InstClass::BARRIER:
  case InstClass::OTHER:
   return FunctionalUnit::NONE;
  }
}

// Resource hold time queried via getResourceCycles() in .cpp.

//===----------------------------------------------------------------------===//
// WMMA Co-execution Stage Rules
//===----------------------------------------------------------------------===//

/// Stage types for WMMA co-execution windows
/// Each WMMA cycle has a stage type determining what can co-execute.
enum class StageType : uint8_t {
  E0,   // Issue cycle - only control co-executes (s_delay_alu, s_set_vgpr_msb)
  E,    // External - mem/salu primary, no valu/trans
  I,    // Internal - mem/salu/valu/trans all allowed
  V,    // Vacant - mem/salu/next-wmma ok, NO VALU/TRANS
  ANY,  // Default - all instruction types allowed (fallback)
};

/// Bitmask for instruction types allowed to co-execute at a stage
namespace CoExecMask {
  constexpr uint8_t None  = 0;
  constexpr uint8_t CTRL  = 1 << 0;   // Control: s_delay_alu, s_set_vgpr_msb
  constexpr uint8_t VALU  = 1 << 1;   // Vector ALU
  constexpr uint8_t TRANS = 1 << 2;   // Transcendentals (V_EXP etc)
  constexpr uint8_t SALU  = 1 << 3;   // Scalar ALU
  constexpr uint8_t DS    = 1 << 4;   // LDS read/write
  constexpr uint8_t VMEM  = 1 << 5;   // Global memory
  constexpr uint8_t SMEM  = 1 << 6;   // Scalar memory
  constexpr uint8_t WMMA  = 1 << 7;   // Next WMMA (V stages only)
  constexpr uint8_t All   = 0xFF;
  
  constexpr uint8_t MEM = DS | VMEM | SMEM;
  constexpr uint8_t StageE0 = CTRL;                           // Issue: control only
  constexpr uint8_t StageE  = CTRL | SALU | MEM;              // External: mem/salu
  constexpr uint8_t StageI  = CTRL | SALU | MEM | VALU | TRANS; // Internal: all ALU
  constexpr uint8_t StageV  = CTRL | SALU | MEM | WMMA;       // Vacant: no valu/trans
  
  /// Get WMMAStageType from a stage mask value
  inline WMMAStageType getStageType(uint8_t Mask) {
    if (Mask == StageE0) return WMMAStageType::E0;
    if (Mask == StageE)  return WMMAStageType::E;
    if (Mask == StageI)  return WMMAStageType::I;
    if (Mask == StageV)  return WMMAStageType::V;
    // For 'All' or unknown, return based on what's allowed
    if (Mask & VALU) return WMMAStageType::I;  // If VALU allowed, it's I-like
    if (Mask & WMMA) return WMMAStageType::V;  // If WMMA allowed (not VALU), V-like
    return WMMAStageType::E;  // Default to E
  }
} // namespace CoExecMask

/// Max stages: INT8 16x16x64 = 17 cycles, round up for safety
constexpr unsigned MaxWMMAStages = 20;

/// Per-WMMA-variant co-execution rules
struct WMMACoExecInfo {
  unsigned Occupancy;                         // Cycles until XDL frees (next WMMA can issue)
  unsigned TotalWindow;                       // Total window including V tail
  uint8_t StageMask[MaxWMMAStages] = {};      // Per-stage allowed instruction mask
  unsigned LastIStage;                        // Last I-stage index (for LD_SCALE rule)
  bool HasScaling;                            // True for FP8/FP6/FP4 scaled variants
  StringRef Pattern;                          // Pattern string for display (e.g., "0EEIIIVV")
  
  /// Default constructor - initialize to safe defaults
  WMMACoExecInfo() : Occupancy(0), TotalWindow(0), LastIStage(0), HasScaling(false) {
    for (unsigned i = 0; i < MaxWMMAStages; ++i)
      StageMask[i] = CoExecMask::All;  // Default: permissive
  }
  
  /// Get the mask bit for an instruction class
  static uint8_t getMaskForIC(InstClass IC) {
    switch (IC) {
    case InstClass::VALU:
      return CoExecMask::VALU;
    case InstClass::TRANS:
      return CoExecMask::TRANS;
    case InstClass::SALU:
    case InstClass::BARRIER:   // Barriers use SALU, can co-exec like SALU
    case InstClass::WAITCNT:   // Wait instructions use SALU
    case InstClass::BRANCH:    // Branch instructions use SALU
      return CoExecMask::SALU;
    case InstClass::DELAY_ALU:
    case InstClass::MSB_SET:
      return CoExecMask::CTRL;
    case InstClass::DS_READ:
    case InstClass::DS_WRITE:
    case InstClass::TDM:
      return CoExecMask::DS;
    case InstClass::VMEM_READ:
    case InstClass::VMEM_WRITE:
      return CoExecMask::VMEM;
    case InstClass::SMEM:
      return CoExecMask::SMEM;
    case InstClass::WMMA:
      return CoExecMask::WMMA;
    case InstClass::NOP:
      // NOPs are V_ALU for now
      return CoExecMask::VALU;
    default:
      return 0;
    }
  }

  /// Check if InstClass can co-execute at given stage
  bool canCoExec(InstClass IC, unsigned Stage) const {
    if (Stage >= TotalWindow)
      return false;
    return StageMask[Stage] & getMaskForIC(IC);
  }
  
  /// Find the next stage (>= CurrentStage) where IC can co-execute.
  std::optional<unsigned> findNextAllowedStage(InstClass IC, unsigned CurrentStage) const {
    uint8_t Needed = getMaskForIC(IC);
    if (Needed == 0)
      return std::nullopt;
    
    for (unsigned S = CurrentStage; S < TotalWindow; ++S) {
      if (StageMask[S] & Needed)
        return S;
    }
    return std::nullopt;
  }
  
  /// Check if this is a back-to-back WMMA scenario (next WMMA at Occupancy)
  bool isBackToBack(unsigned NextWMMAStage) const {
    return NextWMMAStage >= Occupancy && NextWMMAStage < TotalWindow;
  }
  
};

/// Helper to create WMMACoExecInfo from a pattern string
/// Pattern chars: '0'=E0, 'E'=External, 'I'=Internal, 'V'=Vacant, 'A'=Any
inline WMMACoExecInfo makeWMMACoExecInfo(unsigned Occupancy, unsigned TotalWindow,
                                         const char* Pattern, unsigned LastIStage,
                                         bool HasScaling) {
  WMMACoExecInfo Info;
  Info.Occupancy = Occupancy;
  Info.TotalWindow = TotalWindow;
  Info.LastIStage = LastIStage;
  Info.HasScaling = HasScaling;
  Info.Pattern = Pattern;  // Store pattern for display
  
  for (unsigned i = 0; i < TotalWindow && Pattern[i]; ++i) {
    switch (Pattern[i]) {
    case '0': Info.StageMask[i] = CoExecMask::StageE0; break;  // E0: control only
    case 'E': Info.StageMask[i] = CoExecMask::StageE; break;   // External
    case 'I': Info.StageMask[i] = CoExecMask::StageI; break;   // Internal
    case 'V': Info.StageMask[i] = CoExecMask::StageV; break;   // Vacant
    case 'A': Info.StageMask[i] = CoExecMask::All; break;      // All allowed
    default:  Info.StageMask[i] = CoExecMask::All; break;      // Safe default
    }
  }
  return Info;
}

/// Get co-execution info for a WMMA/MFMA instruction.
inline WMMACoExecInfo getWMMACoExecInfo(const MachineInstr &MI,
                                        const SIInstrInfo &TII) {
  StringRef Name = TII.getName(MI.getOpcode());
  
  // Check for scaled variants (LD_SCALE rule applies)
  bool HasScaling = Name.contains_insensitive("scale");
  
  if (Name.contains_insensitive("16x16x64_iu8")) {
    return makeWMMACoExecInfo(16, 17, "0EIIEEIIEEIIEEIIV", 15, HasScaling);
  }
  
  // F8F6F4 16x16x128: window size depends on operand formats
  if (Name.contains_insensitive("16x16x128_f8f6f4")) {
    // Check if both operands are FP4 (shorter window)
    bool BothF4 = false;
    if (const MachineOperand *FmtA = TII.getNamedOperand(MI, AMDGPU::OpName::matrix_a_fmt)) {
      if (const MachineOperand *FmtB = TII.getNamedOperand(MI, AMDGPU::OpName::matrix_b_fmt)) {
        BothF4 = (FmtA->getImm() == WMMA::MATRIX_FMT_FP4 &&
                  FmtB->getImm() == WMMA::MATRIX_FMT_FP4);
      }
    }
    
    if (BothF4) {
      // f4×f4: 4-cycle occupancy, 6-cycle window
      return makeWMMACoExecInfo(4, 6, "0EEIVV", 3, HasScaling);
    } else {
      // f8×*, f6×*, or mixed: 8-cycle occupancy, 10-cycle window
      return makeWMMACoExecInfo(8, 10, "0EEIEEIIVV", 7, HasScaling);
    }
  }
  
  // FP8/BF8 16x16x64: 4-cycle occupancy, 6-cycle window  
  if (Name.contains_insensitive("16x16x64_fp8") ||
      Name.contains_insensitive("16x16x64_bf8")) {
    return makeWMMACoExecInfo(4, 6, "0EEIVV", 3, HasScaling);
  }
  
  // F16/BF16 16x16x32: 8-cycle occupancy, 9-cycle window
  if (Name.contains_insensitive("16x16x32_f16") ||
      Name.contains_insensitive("16x16x32_bf16")) {
    return makeWMMACoExecInfo(8, 9, "0EIIEEIIV", 7, HasScaling);
  }
  
  // FP8/BF8 16x16x128: 8-cycle occupancy, 10-cycle window
  if (Name.contains_insensitive("16x16x128_fp8") ||
      Name.contains_insensitive("16x16x128_bf8")) {
    return makeWMMACoExecInfo(8, 10, "0EEIEEIIVV", 7, HasScaling);
  }
  
  // 32x16x128 F4 variants
  if (Name.contains_insensitive("32x16x128_f4")) {
    return makeWMMACoExecInfo(4, 6, "0EEIVV", 3, HasScaling);
  }
  
  // Default fallback: permissive 8-cycle pattern
  return makeWMMACoExecInfo(8, 9, "AAAAAAAAA", 7, HasScaling);
}

//===----------------------------------------------------------------------===//
// PendingMemOp - Tracks in-flight memory operations
//===----------------------------------------------------------------------===//

/// Memory counter types for s_wait_* tracking
enum class MemCounter : unsigned {
  LGKM = 0,
  VMEM = 1,
  VS = 2,
  TENSOR = 3,
  XCNT = 4,
  NUM_COUNTERS
};

/// In-flight memory operation
struct PendingMemOp {
  unsigned IssueCycle;
  unsigned CompletionCycle;
  unsigned DestVGPR;
  unsigned NumRegs;
  MemCounter Counter;
  bool IsLoad;
  
  PendingMemOp(unsigned Issue, unsigned Complete, unsigned Dest, unsigned NRegs,
               MemCounter Cnt, bool Load)
      : IssueCycle(Issue), CompletionCycle(Complete), DestVGPR(Dest),
        NumRegs(NRegs), Counter(Cnt), IsLoad(Load) {}
  
  bool writesToAnyOf(const SmallSet<unsigned, 16> &Regs) const {
    for (unsigned i = 0; i < NumRegs; ++i) {
      if (Regs.contains(DestVGPR + i))
        return true;
    }
    return false;
  }
};

namespace MemLimits {
  constexpr unsigned MaxDSInFlight = 10;
  constexpr unsigned MaxVMEMInFlight = 16;
  constexpr unsigned MaxSMEMInFlight = 10;
  constexpr unsigned MaxTDMInFlight = 4;
}

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

  unsigned VGPRCacheHits = 0;
  unsigned VGPRCacheMisses = 0;
  unsigned VGPRCacheEvictions = 0;

  float VGPRCacheHitRate() const {
    unsigned Total = VGPRCacheHits + VGPRCacheMisses;
    return Total > 0 ? static_cast<float>(VGPRCacheHits) / Total : 0.0f;
  }

  unsigned StallCycles() const {
    return NumMSBSetExposed + StallFunctionalUnit + StallCoExec + 
           StallDelayAlu + StallMemFIFO + StallWaitCnt + StallRegBankConflict;
  }

  // WMMA Co-execution
  unsigned WMMAWindowCycles = 0;
  unsigned WMMACoExecUsed = 0;
  unsigned WMMACoExecBlocked = 0;
  unsigned WMMAStarved = 0;

  /// Scale all metrics by a factor (for loop trip counts, branch probabilities)
  BlockMetrics operator*(float Factor) const {
    // Helper to scale and round (not truncate)
    auto scale = [Factor](unsigned V) -> unsigned {
      return static_cast<unsigned>(V * Factor + 0.5f);
    };
    
    BlockMetrics Result;
    Result.NumInstructions = scale(NumInstructions);
    Result.NumVALU = scale(NumVALU);
    Result.NumSALU = scale(NumSALU);
    Result.NumTRANS = scale(NumTRANS);
    Result.NumWMMA = scale(NumWMMA);
    Result.NumVOPD = scale(NumVOPD);
    Result.NumDSRead = scale(NumDSRead);
    Result.NumDSWrite = scale(NumDSWrite);
    Result.NumVMEM = scale(NumVMEM);
    Result.NumSMEM = scale(NumSMEM);
    Result.NumTDM = scale(NumTDM);
    Result.NumBranch = scale(NumBranch);
    Result.NumBarrier = scale(NumBarrier);
    Result.NumNop = scale(NumNop);
    Result.NumDelayAlu = scale(NumDelayAlu);
    Result.NumMSBSet = scale(NumMSBSet);
    Result.NumSpill = scale(NumSpill);
    Result.NumReload = scale(NumReload);
    Result.NumSGPRToVGPR = scale(NumSGPRToVGPR);
    Result.NumVGPRToSGPR = scale(NumVGPRToSGPR);

    Result.NumWaitcnt = scale(NumWaitcnt);
    Result.WaitLGKM = scale(WaitLGKM);
    Result.WaitVMEM = scale(WaitVMEM);
    Result.WaitEXP = scale(WaitEXP);
    Result.NumFalseWaits = scale(NumFalseWaits);

    Result.TotalCycles = scale(TotalCycles);
    // Stall breakdown
    Result.StallFunctionalUnit = scale(StallFunctionalUnit);
    Result.StallCoExec = scale(StallCoExec);
    Result.StallDelayAlu = scale(StallDelayAlu);
    Result.StallMemFIFO = scale(StallMemFIFO);
    Result.StallWaitCnt = scale(StallWaitCnt);
    Result.StallFalseWait = scale(StallFalseWait);
    // Per-unit stall breakdown
    Result.StallXDL = scale(StallXDL);
    Result.StallVALU = scale(StallVALU);
    Result.StallSALU = scale(StallSALU);
    Result.StallTRANSUnit = scale(StallTRANSUnit);
    Result.StallLDS = scale(StallLDS);
    Result.StallVMEMUnit = scale(StallVMEMUnit);
    Result.StallRegBankConflict = scale(StallRegBankConflict);

    Result.VGPRCacheHits = scale(VGPRCacheHits);
    Result.VGPRCacheMisses = scale(VGPRCacheMisses);
    Result.VGPRCacheEvictions = scale(VGPRCacheEvictions);

    Result.WMMAWindowCycles = scale(WMMAWindowCycles);
    Result.WMMACoExecUsed = scale(WMMACoExecUsed);
    Result.WMMACoExecBlocked = scale(WMMACoExecBlocked);
    Result.WMMAStarved = scale(WMMAStarved);
    return Result;
  }

  /// Allow factor * metrics
  friend BlockMetrics operator*(float Factor, const BlockMetrics &M) {
    return M * Factor;
  }

  /// Sum two metric sets
  BlockMetrics operator+(const BlockMetrics &O) const {
    BlockMetrics Result;
    Result.NumInstructions = NumInstructions + O.NumInstructions;
    Result.NumVALU = NumVALU + O.NumVALU;
    Result.NumSALU = NumSALU + O.NumSALU;
    Result.NumTRANS = NumTRANS + O.NumTRANS;
    Result.NumWMMA = NumWMMA + O.NumWMMA;
    Result.NumVOPD = NumVOPD + O.NumVOPD;
    Result.NumDSRead = NumDSRead + O.NumDSRead;
    Result.NumDSWrite = NumDSWrite + O.NumDSWrite;
    Result.NumVMEM = NumVMEM + O.NumVMEM;
    Result.NumSMEM = NumSMEM + O.NumSMEM;
    Result.NumTDM = NumTDM + O.NumTDM;
    Result.NumBranch = NumBranch + O.NumBranch;
    Result.NumBarrier = NumBarrier + O.NumBarrier;
    Result.NumNop = NumNop + O.NumNop;
    Result.NumDelayAlu = NumDelayAlu + O.NumDelayAlu;
    Result.NumMSBSet = NumMSBSet + O.NumMSBSet;
    Result.NumSpill = NumSpill + O.NumSpill;
    Result.NumReload = NumReload + O.NumReload;
    Result.NumSGPRToVGPR = NumSGPRToVGPR + O.NumSGPRToVGPR;
    Result.NumVGPRToSGPR = NumVGPRToSGPR + O.NumVGPRToSGPR;

    Result.NumWaitcnt = NumWaitcnt + O.NumWaitcnt;
    Result.WaitLGKM = WaitLGKM + O.WaitLGKM;
    Result.WaitVMEM = WaitVMEM + O.WaitVMEM;
    Result.WaitEXP = WaitEXP + O.WaitEXP;
    Result.NumFalseWaits = NumFalseWaits + O.NumFalseWaits;

    Result.TotalCycles = TotalCycles + O.TotalCycles;
    Result.StallFunctionalUnit = StallFunctionalUnit + O.StallFunctionalUnit;
    Result.StallCoExec = StallCoExec + O.StallCoExec;
    Result.StallDelayAlu = StallDelayAlu + O.StallDelayAlu;
    Result.StallMemFIFO = StallMemFIFO + O.StallMemFIFO;
    Result.StallWaitCnt = StallWaitCnt + O.StallWaitCnt;
    Result.StallFalseWait = StallFalseWait + O.StallFalseWait;
    // Per-unit stall breakdown
    Result.StallXDL = StallXDL + O.StallXDL;
    Result.StallVALU = StallVALU + O.StallVALU;
    Result.StallSALU = StallSALU + O.StallSALU;
    Result.StallTRANSUnit = StallTRANSUnit + O.StallTRANSUnit;
    Result.StallLDS = StallLDS + O.StallLDS;
    Result.StallVMEMUnit = StallVMEMUnit + O.StallVMEMUnit;
    Result.StallRegBankConflict = StallRegBankConflict + O.StallRegBankConflict;

    Result.VGPRCacheHits = VGPRCacheHits + O.VGPRCacheHits;
    Result.VGPRCacheMisses = VGPRCacheMisses + O.VGPRCacheMisses;
    Result.VGPRCacheEvictions = VGPRCacheEvictions + O.VGPRCacheEvictions;

    Result.WMMAWindowCycles = WMMAWindowCycles + O.WMMAWindowCycles;
    Result.WMMACoExecUsed = WMMACoExecUsed + O.WMMACoExecUsed;
    Result.WMMACoExecBlocked = WMMACoExecBlocked + O.WMMACoExecBlocked;
    Result.WMMAStarved = WMMAStarved + O.WMMAStarved;
    return Result;
  }

  // === Formatting helpers ===
  
  /// Print instruction breakdown: "VALU:12 SALU:2 DS:4 ..."
  void printInstBreakdown(raw_ostream &OS) const {
    bool First = true;
    auto Emit = [&](const char *Name, unsigned Val) {
      if (Val) {
        if (!First) OS << " ";
        OS << Name << ":" << Val;
        First = false;
      }
    };
    // Compute ops
    Emit("VALU", NumVALU);
    Emit("SALU", NumSALU);
    Emit("TRANS", NumTRANS);
    Emit("WMMA", NumWMMA);
    // Memory ops
    Emit("DS", NumDSRead + NumDSWrite);
    Emit("VMEM", NumVMEM);
    Emit("SMEM", NumSMEM);
    Emit("TDM", NumTDM);
    // Control flow (grouped as "Ctrl" to keep output concise)
    unsigned NumCtrl = NumWaitcnt + NumBarrier + NumDelayAlu + NumMSBSet + NumNop + NumBranch;
    Emit("Ctrl", NumCtrl);
    // Spills/reloads (show separately for clarity)
    Emit("Spill", NumSpill);
    Emit("Reload", NumReload);
  }

  /// Print stall breakdown: "FU:10 | WMMACoExecMiss:5 | Wait:3"
  void printStallBreakdown(raw_ostream &OS) const {
    bool First = true;
    auto Emit = [&](const char *Name, unsigned Val) {
      if (Val) {
        if (!First) OS << " | ";
        OS << Name << ":" << Val;
        First = false;
      }
    };
    Emit("FU", StallFunctionalUnit);
    Emit("WMMACoExecMiss", StallCoExec);  // Missed WMMA co-execution slots
    Emit("DelayAlu", StallDelayAlu);
    Emit("MemFIFO", StallMemFIFO);
    Emit("Wait", StallWaitCnt);
    Emit("RegBank", StallRegBankConflict);
  }

  /// Print FU stall detail: "XDL:5 VALU:3 TRANS:2"
  void printFUBreakdown(raw_ostream &OS) const {
    bool First = true;
    auto Emit = [&](const char *Name, unsigned Val) {
      if (Val) {
        if (!First) OS << " ";
        OS << Name << ":" << Val;
        First = false;
      }
    };
    Emit("XDL", StallXDL);
    Emit("VALU", StallVALU);
    Emit("TRANS", StallTRANSUnit);
    Emit("SALU", StallSALU);
    Emit("LDS", StallLDS);
    Emit("VMEM", StallVMEMUnit);
  }
};

//===----------------------------------------------------------------------===//
// RegisterFile - Bank conflict + source cache modeling
//===----------------------------------------------------------------------===//

/// VGPR source cache: 8 banks × 3 ports × 4-entry LRU.
struct VGPRSourceCache {
  static constexpr unsigned NumBanks = 8;
  static constexpr unsigned NumPorts = 3;
  static constexpr unsigned CacheDepth = 4;
  
  std::array<std::array<SmallVector<unsigned, CacheDepth>, NumPorts>, NumBanks> Cache;
  
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
    auto It = llvm::find(C, HWReg);
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
    for (unsigned Port = 0; Port < NumPorts; ++Port)
      llvm::erase(Cache[Bank][Port], HWReg);
  }
};

struct RegBankResult {
  unsigned Stalls = 0;
  std::string CachePattern;
  unsigned CacheHits = 0;
  unsigned CacheMisses = 0;
  unsigned CacheEvictions = 0;
};

/// VGPR: 8 banks + source cache. SGPR: 4 banks.
struct RegisterFile {
  const SIRegisterInfo *TRI = nullptr;
  VGPRSourceCache SrcCache;
  
  RegisterFile() = default;
  explicit RegisterFile(const SIRegisterInfo *TRI) : TRI(TRI) {}
  
  static unsigned countBankConflicts(ArrayRef<unsigned> HWRegs, unsigned NumBanks) {
    SmallVector<unsigned, 8> BankCount(NumBanks, 0);
    for (unsigned HWReg : HWRegs)
      BankCount[HWReg % NumBanks]++;
    unsigned MaxReads = *std::max_element(BankCount.begin(), BankCount.end());
    return MaxReads > 1 ? MaxReads - 1 : 0;
  }
  
  RegBankResult getRegBankStalls(const MachineInstr &MI) {
    RegBankResult Result;
    if (!TRI) return Result;
    
    SrcCache.resetCycleStats();
    SmallVector<unsigned, 16> VGPRMisses;
    SmallVector<unsigned, 8> SGPRHWRegs;
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    std::string Pattern;
    
    unsigned PortIdx = 0;
    for (const MachineOperand &MO : MI.explicit_uses()) {
      if (!MO.isReg()) { PortIdx++; continue; }
      if (!MO.getReg().isPhysical()) { PortIdx++; continue; }
      
      Register Reg = MO.getReg();
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      unsigned NumComponents = (RC ? TRI->getRegSizeInBits(*RC) : 32) / 32;
      unsigned BaseHWReg = TRI->getHWRegIndex(Reg);
      unsigned Port = PortIdx % 3;
      
      if (TRI->isVGPR(MRI, Reg)) {
        bool AllHit = true;
        for (unsigned i = 0; i < NumComponents; ++i) {
          unsigned HWReg = BaseHWReg + i;
          if (!SrcCache.checkHit(HWReg, Port)) {
            VGPRMisses.push_back(HWReg);
            SrcCache.recordMiss(HWReg, Port);
            AllHit = false;
          }
        }
        Pattern += AllHit ? '$' : '-';
      } else if (TRI->isSGPRReg(MRI, Reg)) {
        for (unsigned i = 0; i < NumComponents; ++i)
          SGPRHWRegs.push_back(BaseHWReg + i);
        Pattern += 's';
      } else {
        Pattern += '.';
      }
      PortIdx++;
    }
    
    Result.Stalls = countBankConflicts(VGPRMisses, 8) + countBankConflicts(SGPRHWRegs, 4);
    if (!Pattern.empty())
      Result.CachePattern = "(" + Pattern + ")";
    Result.CacheHits = SrcCache.CycleHits;
    Result.CacheMisses = SrcCache.CycleMisses;
    Result.CacheEvictions = SrcCache.CycleEvictions;
    return Result;
  }
  
  void invalidateWrites(const MachineInstr &MI) {
    if (!TRI) return;
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    
    for (const MachineOperand &MO : MI.defs()) {
      if (!MO.isReg() || !MO.getReg().isPhysical()) continue;
      
      Register Reg = MO.getReg();
      if (!TRI->isVGPR(MRI, Reg)) continue;
      
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      unsigned NumComponents = (RC ? TRI->getRegSizeInBits(*RC) : 32) / 32;
      unsigned BaseHWReg = TRI->getHWRegIndex(Reg);
      
      for (unsigned i = 0; i < NumComponents; ++i)
        SrcCache.invalidate(BaseHWReg + i);
    }
  }
};

//===----------------------------------------------------------------------===//
// GPUSimState - Hardware simulation state carried across blocks
//===----------------------------------------------------------------------===//

/// Simulated GPU hardware state
struct GPUSimState {
  unsigned CurrentCycle = 0;
  RegisterFile RegFile;

  std::array<unsigned, static_cast<size_t>(FunctionalUnit::NUM_UNITS)>
      UnitBusyUntil = {};

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

  std::array<unsigned, 1024> VGPRReadyTimes = {};
  std::array<unsigned, 128> SGPRReadyTimes = {};

  struct RecentInst {
    unsigned IssueCycle;
    unsigned Latency;
  };
  std::deque<RecentInst> RecentVALU;
  std::deque<RecentInst> RecentTRANS;
  unsigned LastSALUCycle = 0;

  unsigned LastVALUCycle = ~0u;
  unsigned LastTRANSCycle = ~0u;

  InstClass PreviousInstClass = InstClass::OTHER;

  std::deque<PendingMemOp> PendingDS;
  std::deque<PendingMemOp> PendingVMEMLoad;
  std::deque<PendingMemOp> PendingVMEMStore;
  std::deque<PendingMemOp> PendingSMEM;
  std::deque<PendingMemOp> PendingTDM;
  
  std::array<unsigned, static_cast<size_t>(MemCounter::NUM_COUNTERS)>
      MemCounters = {};

  bool inWMMAWindow() const { return ActiveWMMA.Active; }

  std::optional<unsigned> getWMMAStage() const {
    return ActiveWMMA.getCurrentStage(CurrentCycle);
  }

  void advanceCycle(unsigned N = 1) {
    advanceToCycle(CurrentCycle + N);
  }
  
  unsigned advanceToCycle(unsigned TargetCycle) {
    if (TargetCycle <= CurrentCycle)
      return 0;
    unsigned Delta = TargetCycle - CurrentCycle;
    CurrentCycle = TargetCycle;
    if (ActiveWMMA.Active && CurrentCycle >= ActiveWMMA.EndCycle)
      ActiveWMMA.Active = false;
    retireCompletedMemOps();
    return Delta;
  }

  unsigned startWMMAWindow(const MachineInstr &MI, const SIInstrInfo &TII) {
    WMMACoExecInfo Info = getWMMACoExecInfo(MI, TII);
    
    bool BackToBack = HadPreviousWMMA && 
                      CurrentCycle >= ActiveWMMA.OccupancyCycle &&
                      CurrentCycle < ActiveWMMA.EndCycle;
    
    HadPreviousWMMA = true;
    
    ActiveWMMA.StartCycle = CurrentCycle;
    ActiveWMMA.EndCycle = CurrentCycle + Info.TotalWindow;
    ActiveWMMA.OccupancyCycle = CurrentCycle + Info.Occupancy;
    ActiveWMMA.Active = true;
    ActiveWMMA.IsBackToBack = BackToBack;
    ActiveWMMA.Info = Info;
    
    setUnitBusyUntil(FunctionalUnit::XDL, ActiveWMMA.OccupancyCycle);
    
    return Info.Occupancy;
  }

  unsigned getUnitBusyUntil(FunctionalUnit Unit) const {
    if (Unit == FunctionalUnit::NONE)
      return 0;
    return UnitBusyUntil[static_cast<size_t>(Unit)];
  }

  void setUnitBusyUntil(FunctionalUnit Unit, unsigned Cycle) {
    if (Unit == FunctionalUnit::NONE)
      return;
    UnitBusyUntil[static_cast<size_t>(Unit)] = Cycle;
  }

  unsigned getUnitStall(FunctionalUnit Unit) const {
    if (Unit == FunctionalUnit::NONE)
      return 0;
    unsigned Busy = getUnitBusyUntil(Unit);
    return (Busy > CurrentCycle) ? (Busy - CurrentCycle) : 0;
  }

  void trackVALUForWMMA(InstClass IC) {
    if (IC == InstClass::VALU)
      LastVALUCycle = CurrentCycle;
    else if (IC == InstClass::TRANS)
      LastTRANSCycle = CurrentCycle;
  }

  unsigned getWMMATRANSStall() const {
    if (LastTRANSCycle == ~0u)
      return 0;
    unsigned TRANSEndCycle = LastTRANSCycle + 2;
    return (CurrentCycle < TRANSEndCycle) ? (TRANSEndCycle - CurrentCycle) : 0;
  }

  unsigned getLDScaleStall(unsigned EffectiveCycle) const {
    if (LastVALUCycle != ~0u && LastVALUCycle == EffectiveCycle - 1)
      return 1;
    if (LastTRANSCycle != ~0u && LastTRANSCycle == EffectiveCycle - 2)
      return 1;
    return 0;
  }

  bool canCoExecuteWithWMMA(InstClass IC) const {
    return getCoExecStall(IC) == 0;
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

  unsigned getVGPRStall(unsigned RegIdx) const {
    if (RegIdx >= VGPRReadyTimes.size())
      return 0;
    unsigned Ready = VGPRReadyTimes[RegIdx];
    return (Ready > CurrentCycle) ? (Ready - CurrentCycle) : 0;
  }

  unsigned getSGPRStall(unsigned RegIdx) const {
    if (RegIdx >= SGPRReadyTimes.size())
      return 0;
    unsigned Ready = SGPRReadyTimes[RegIdx];
    return (Ready > CurrentCycle) ? (Ready - CurrentCycle) : 0;
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

  void retireCompletedMemOps();

  unsigned getFIFOStall(const std::deque<PendingMemOp> &Queue,
                        unsigned MaxInFlight) const {
    if (Queue.size() < MaxInFlight)
      return 0;
    unsigned OldestComplete = Queue.front().CompletionCycle;
    return (OldestComplete > CurrentCycle) ? (OldestComplete - CurrentCycle) : 0;
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
      OldestComplete = std::min(OldestComplete, PendingVMEMLoad.front().CompletionCycle);
    if (!PendingVMEMStore.empty())
      OldestComplete = std::min(OldestComplete, PendingVMEMStore.front().CompletionCycle);
    return (OldestComplete > CurrentCycle) ? (OldestComplete - CurrentCycle) : 0;
  }

  void issueMemOp(std::deque<PendingMemOp> &Queue, MemCounter Cnt,
                  unsigned Latency, unsigned BaseReg, unsigned NumRegs,
                  bool IsLoad, bool UpdateVGPR, bool UpdateSGPR) {
    Queue.emplace_back(CurrentCycle, CurrentCycle + Latency, BaseReg, NumRegs, Cnt, IsLoad);
    MemCounters[static_cast<size_t>(Cnt)]++;
    unsigned ReadyTime = CurrentCycle + Latency;
    if (UpdateVGPR) {
      for (unsigned i = 0; i < NumRegs && (BaseReg + i) < VGPRReadyTimes.size(); ++i)
        VGPRReadyTimes[BaseReg + i] = ReadyTime;
    }
    if (UpdateSGPR) {
      for (unsigned i = 0; i < NumRegs && (BaseReg + i) < SGPRReadyTimes.size(); ++i)
        SGPRReadyTimes[BaseReg + i] = ReadyTime;
    }
  }

  void issueDS(unsigned Latency, unsigned BaseVGPR, unsigned NumRegs, bool IsLoad) {
    issueMemOp(PendingDS, MemCounter::LGKM, Latency, BaseVGPR, NumRegs, IsLoad,
               /*UpdateVGPR=*/IsLoad, /*UpdateSGPR=*/false);
  }

  void issueVMEM(unsigned Latency, unsigned BaseVGPR, unsigned NumRegs, bool IsLoad) {
    MemCounter Cnt = IsLoad ? MemCounter::VMEM : MemCounter::VS;
    auto &Queue = IsLoad ? PendingVMEMLoad : PendingVMEMStore;
    issueMemOp(Queue, Cnt, Latency, BaseVGPR, NumRegs, IsLoad,
               /*UpdateVGPR=*/IsLoad, /*UpdateSGPR=*/false);
  }

  void issueSMEM(unsigned Latency, unsigned BaseSGPR, unsigned NumRegs) {
    issueMemOp(PendingSMEM, MemCounter::LGKM, Latency, BaseSGPR, NumRegs,
               /*IsLoad=*/true, /*UpdateVGPR=*/false, /*UpdateSGPR=*/true);
  }

  void issueTDM(unsigned Latency) {
    issueMemOp(PendingTDM, MemCounter::TENSOR, Latency, /*BaseReg=*/0, /*NumRegs=*/0,
               /*IsLoad=*/true, /*UpdateVGPR=*/false, /*UpdateSGPR=*/false);
  }

  unsigned getTDMFIFOStall() const {
    return getFIFOStall(PendingTDM, MemLimits::MaxTDMInFlight);
  }

  unsigned getCounter(MemCounter Cnt) const {
    return MemCounters[static_cast<size_t>(Cnt)];
  }

  unsigned getDSCompletionCycle(unsigned Index) const {
    if (Index >= PendingDS.size()) return 0;
    return PendingDS[Index].CompletionCycle;
  }
  
  unsigned getVMEMLoadCompletionCycle(unsigned Index) const {
    if (Index >= PendingVMEMLoad.size()) return 0;
    return PendingVMEMLoad[Index].CompletionCycle;
  }
  
  unsigned getVMEMStoreCompletionCycle(unsigned Index) const {
    if (Index >= PendingVMEMStore.size()) return 0;
    return PendingVMEMStore[Index].CompletionCycle;
  }

  unsigned getTDMCompletionCycle(unsigned Index) const {
    if (Index >= PendingTDM.size()) return 0;
    return PendingTDM[Index].CompletionCycle;
  }

  unsigned getNumPendingDS() const { return PendingDS.size(); }
  unsigned getNumPendingVMEMLoad() const { return PendingVMEMLoad.size(); }
  unsigned getNumPendingVMEMStore() const { return PendingVMEMStore.size(); }
  unsigned getNumPendingSMEM() const { return PendingSMEM.size(); }
  unsigned getNumPendingTDM() const { return PendingTDM.size(); }

  unsigned computeWaitStall(const std::deque<PendingMemOp> &Queue,
                            unsigned WaitCount) const {
    unsigned Pending = Queue.size();
    if (Pending <= WaitCount)
      return 0;
    unsigned WaitForIndex = Pending - WaitCount - 1;
    unsigned CompletionCycle = Queue[WaitForIndex].CompletionCycle;
    return (CompletionCycle > CurrentCycle) ? (CompletionCycle - CurrentCycle) : 0;
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
      StallRatio = static_cast<float>(Scaled.StallCycles()) / Scaled.TotalCycles;
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

