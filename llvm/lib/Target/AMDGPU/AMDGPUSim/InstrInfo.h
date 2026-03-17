//===- AMDGPUSim/InstrInfo.h - Per-Instruction Output -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the per-instruction simulation output (InstrSimInfo) returned
/// by the simulator core.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_INSTRINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_INSTRINFO_H

#include "SimInst.h"
#include <algorithm>
#include <string>

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Stall Reasons
//===----------------------------------------------------------------------===//

/// Reason for stall cycles on an instruction.
enum class StallReason : uint8_t {
  NONE = 0,
  FU_BUSY,              // Functional unit not ready
  COEXEC_BLOCKED,       // Blocked by WMMA co-execution rules
  LONG_LAT_VALU,        // Long-latency VALU blocked by WMMA window
  LOLVALU_TRANS_HAZARD, // 1-cycle mutual exclusion: LOLVALU <-> TRANS
  VA_SSRC_STALL,        // VA_SSRC: VALU/WMMA with SGPR blocks SALU
  VA_VDST_WAIT,         // VA_VDST: s_wait_alu depctr_va_vdst stall
  RAW_HAZARD,           // RAW: register dependency (scoreboard)
  WAITCNT,              // Memory wait (s_wait_*)
  DELAY_ALU,            // RAW dependency (s_delay_alu)
  MEM_FIFO,             // Memory FIFO full
  MSB_SET_EXPOSED,      // s_set_vgpr_msb not fused
  REG_BANK,             // Register bank conflict
  IS_FETCH              // Instruction Store cache miss
};

/// Get enum name for StallReason.
inline const char *getStallReasonName(StallReason Reason) {
  switch (Reason) {
  case StallReason::NONE:
    return nullptr;
  case StallReason::FU_BUSY:
    return "FU_BUSY";
  case StallReason::COEXEC_BLOCKED:
    return "COEXEC_BLOCKED";
  case StallReason::LONG_LAT_VALU:
    return "LONG_LAT_VALU";
  case StallReason::LOLVALU_TRANS_HAZARD:
    return "LOLVALU_TRANS_HAZARD";
  case StallReason::VA_SSRC_STALL:
    return "VA_SSRC_STALL";
  case StallReason::VA_VDST_WAIT:
    return "VA_VDST_WAIT";
  case StallReason::RAW_HAZARD:
    return "RAW_HAZARD";
  case StallReason::WAITCNT:
    return "WAITCNT";
  case StallReason::DELAY_ALU:
    return "DELAY_ALU";
  case StallReason::MEM_FIFO:
    return "MEM_FIFO";
  case StallReason::MSB_SET_EXPOSED:
    return "MSB_SET_EXPOSED";
  case StallReason::REG_BANK:
    return "REG_BANK";
  case StallReason::IS_FETCH:
    return "IS_FETCH";
  }
  return "UNKNOWN";
}

/// Get human-readable reason string.
inline const char *getStallReasonString(StallReason Reason) {
  switch (Reason) {
  case StallReason::NONE:
    return nullptr;
  case StallReason::FU_BUSY:
    return "FU busy";
  case StallReason::COEXEC_BLOCKED:
    return "CoExec blocked";
  case StallReason::LONG_LAT_VALU:
    return "LongLatVALU blocked";
  case StallReason::LOLVALU_TRANS_HAZARD:
    return "LOLVALU<->TRANS hazard";
  case StallReason::VA_SSRC_STALL:
    return "VA_SSRC blocked";
  case StallReason::VA_VDST_WAIT:
    return "VA_VDST wait";
  case StallReason::RAW_HAZARD:
    return "RAW hazard";
  case StallReason::WAITCNT:
    return "WaitCnt";
  case StallReason::DELAY_ALU:
    return "DelayAlu";
  case StallReason::MEM_FIFO:
    return "FIFO full";
  case StallReason::MSB_SET_EXPOSED:
    return "MSB exposed";
  case StallReason::REG_BANK:
    return "RegBank conflict";
  case StallReason::IS_FETCH:
    return "IS fetch";
  }
  return "Unknown";
}

//===----------------------------------------------------------------------===//
// Per-Instruction Simulation Info
//===----------------------------------------------------------------------===//

/// Detailed stall breakdown for per-source attribution.
/// Populated by the simulator core; used by the pass for verbose logging
/// and BlockMetrics attribution.
struct StallBreakdown {
  unsigned FU = 0;
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
  unsigned ISFetch = 0;
  bool RegBankInWMMAWindow = false;
  bool IsScaledWMMA = false;
  unsigned WMMAStartCycle = 0;

  unsigned total() const {
    unsigned EffRB = RegBankInWMMAWindow ? 0 : RegBank;
    return std::max({FU, VALUSlot, CoExec, DelayAlu, WaitCnt, MemFIFO, EffRB,
                     LongLatVALU, LOLVALUTRANSHazard, SSRC, VaVdst, RAW,
                     ISFetch});
  }
};

/// Per-instruction simulation result returned by simulateInst().
struct InstrSimInfo {
  //--- Stall information ---
  unsigned StallCycles = 0;
  StallReason Reason = StallReason::NONE;
  StallBreakdown Breakdown;

  //--- WMMA window state at instruction issue ---
  bool InWMMAWindow = false;
  uint8_t WMMAStage = 0;
  uint8_t WMMATotalWindow = 0;
  WMMAStageType StageType = WMMAStageType::NONE;
  bool CoExecuted = false;
  bool LDScaleBlocked = false;

  //--- Instruction outcome flags ---
  bool WasFused = false;
  bool WasExposed = false;
  bool WasMasked = false; // MSB_SET exposed but masked by next instr stall
  bool IsWMMA = false;
  std::string_view WMMAPattern;

  //--- Cache/register bank analysis ---
  std::string CachePattern;
  unsigned CacheHits = 0;
  unsigned CacheMisses = 0;
  unsigned CacheEvictions = 0;
  unsigned RegBankStalls = 0;
  bool RegBankInWMMAWindow = false;

  //--- IS cache (populated by library when EnableISCache is set) ---
  unsigned ISFetchStall = 0;
  unsigned ISFetchesTriggered = 0;
  unsigned ISBytesConsumed = 0;

  /// Get human-readable reason string.
  const char *getReasonString() const { return getStallReasonString(Reason); }

  /// Get stage type character for compact display.
  char getStageChar() const {
    switch (StageType) {
    case WMMAStageType::E0:
      return '0';
    case WMMAStageType::E:
      return 'E';
    case WMMAStageType::I:
      return 'I';
    case WMMAStageType::V:
      return 'V';
    default:
      return '?';
    }
  }

  /// Get stage type name.
  const char *getStageName() const {
    switch (StageType) {
    case WMMAStageType::E0:
      return "E0";
    case WMMAStageType::E:
      return "E";
    case WMMAStageType::I:
      return "I";
    case WMMAStageType::V:
      return "V";
    default:
      return "?";
    }
  }
};

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_INSTRINFO_H
