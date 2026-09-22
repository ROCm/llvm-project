//===-- AMDGPUCoExecInfo.h - Co-execution info ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Shared types for co-execution modeling used by GCNHazardRecognizer
/// and AMDGPUStaticSimulator, and the schedulers.
///
/// Multi-cycle instructions (WMMA, TRANS, etc.) have execution windows where
/// other instruction types can co-execute. For WMMA, slot patterns depend on
/// the variant:
///
///   E0 (Issue): Control instructions only (s_delay_alu, s_set_vgpr_msb)
///   E (External): Memory and SALU can co-execute, no VALU
///   I (Internal): VALU, TRANS, memory, and SALU can all co-execute
///   V (Vacant): Memory/SALU/next-WMMA ok, NO VALU/TRANS
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECINFO_H

#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <cassert>
#include <cstdint>
#include <optional>

namespace llvm {

namespace AMDGPU {

//===----------------------------------------------------------------------===//
// Co-execution Bitmasks
//===----------------------------------------------------------------------===//

using CoExecMaskT = uint16_t;

/// Bitmask for instruction types allowed to co-execute at a stage.
namespace CoExecMask {
constexpr CoExecMaskT None = 0;
/// RooflineResult relies on the first 8 masks having only one of the rightmost
/// 8 bits being set: as such do not change values of existing masks without
/// adjusting RooflineResult accordingly.
constexpr CoExecMaskT CTRL = 1 << 0;  // Control: s_delay_alu, s_set_vgpr_msb
constexpr CoExecMaskT VALU = 1 << 1;  // Vector ALU
constexpr CoExecMaskT TRANS = 1 << 2; // Transcendentals (V_EXP etc)
constexpr CoExecMaskT SALU = 1 << 3;  // Scalar ALU
constexpr CoExecMaskT DS = 1 << 4;    // LDS read/write
constexpr CoExecMaskT VMEM = 1 << 5;  // Global memory
constexpr CoExecMaskT SMEM = 1 << 6;  // Scalar memory
constexpr CoExecMaskT WMMA = 1 << 7;  // Next WMMA (V stages only), or MFMA
constexpr CoExecMaskT All = 0xFFFF;

constexpr CoExecMaskT MEM = DS | VMEM | SMEM;
constexpr CoExecMaskT StageE0 = CTRL;             // Issue: control only
constexpr CoExecMaskT StageE = CTRL | SALU | MEM; // External: mem/salu
constexpr CoExecMaskT StageI =
    CTRL | SALU | MEM | VALU | TRANS; // Internal: all ALU
// Internal + scaled-WMMA absorb: same as StageI but the next scaled
// WMMA may issue here — its LD_SCALE consumes the I cycle and the matrix
// multiply lands in the V slot that follows. Used for the last I before
// V of HasScaling patterns.
constexpr CoExecMaskT StageIS = StageI | WMMA;
constexpr CoExecMaskT StageV =
    CTRL | SALU | MEM | WMMA;                 // Vacant: no valu/trans
constexpr CoExecMaskT StageTR = All & ~TRANS; // TRANS co-exec: no TRANS
} // namespace CoExecMask

//===----------------------------------------------------------------------===//
// VALU to Memory Latency (GFX1250)
//===----------------------------------------------------------------------===//

/// Latency in cycles for VALU/WMMA writes to VGPR that VMEM/DS reads.
/// These values apply to GFX1250 and define the RAW hazard latency.
namespace VALUToMemLatency {
constexpr unsigned VALU = 16; // Non-WMMA VALU -> VMEM/DS
constexpr unsigned WMMA = 24; // WMMA -> VMEM/DS
} // namespace VALUToMemLatency

/// Additional cycles beyond instruction latency for WAR hazard between
/// VALU/WMMA reads and subsequent DS/VMEM writes to the same VGPR.
namespace WARVdstLatency {
constexpr unsigned AdditionalCycles = 16;
} // namespace WARVdstLatency

//===----------------------------------------------------------------------===//
// Instruction Flavor Classification
//===----------------------------------------------------------------------===//

/// Classification of instructions by execution characteristics.
/// Used for scheduling decisions and co-execution slot preferences.
enum class InstructionFlavor : uint8_t {
  WMMA,            // WMMA/MFMA matrix operations
  SingleCycleVALU, // Single-cycle VALU (not TRANS, not multi-cycle CVT)
  TRANS,           // Transcendental ops (v_exp, v_log, etc.)
  MultiCycleVALU,  // VALU instructions with repeat rate > 1
  VMEM,            // FLAT/GLOBAL memory operations
  SMEM,            // Scalar memory operations
  DS,              // LDS/GDS operations
  SALU,            // Scalar ALU
  DMA,             // Tensor DMA operations
  Fence,           // Fences and waits
  Other,           // Everything else
  NUM_FLAVORS
};

InstructionFlavor classifyFlavor(const MachineInstr &MI,
                                 const SIInstrInfo &SII);

// This function is used by RooflineResult which supports limited number of
// CoExecMask's. It will handle zero-ed (i.e. None) mask just fine, but adding
// a new return here without updating RooflineResult would lead to assertions
// and out-of-bounds access.
inline CoExecMaskT flavorToCoExecMask(InstructionFlavor F) {
  switch (F) {
  case InstructionFlavor::SingleCycleVALU:
  case InstructionFlavor::MultiCycleVALU:
    return CoExecMask::VALU;
  case InstructionFlavor::TRANS:
    return CoExecMask::TRANS;
  case InstructionFlavor::SALU:
    return CoExecMask::SALU;
  case InstructionFlavor::DS:
  case InstructionFlavor::DMA:
    return CoExecMask::DS;
  case InstructionFlavor::VMEM:
    return CoExecMask::VMEM;
  case InstructionFlavor::SMEM:
    return CoExecMask::SMEM;
  case InstructionFlavor::WMMA:
    return CoExecMask::WMMA;
  case InstructionFlavor::Fence:
  case InstructionFlavor::Other:
  case InstructionFlavor::NUM_FLAVORS:
    return CoExecMask::None;
  }

  return CoExecMask::None;
}

inline CoExecMaskT
getCoExecMaskForMI(const MachineInstr &MI, const SIInstrInfo &TII) {
  // classifyFlavor treats LDSDMA as DS, whilst the original HazardRecognizer
  // behavior expected them to be treated as VALU.
  if (TII.isLDSDMA(MI))
    return CoExecMask::VALU;

  InstructionFlavor Flavor = classifyFlavor(MI, TII);
  // There is no Control InstructionFlavor
  if (Flavor == InstructionFlavor::Other ||
      Flavor == InstructionFlavor::Fence) {
    // Control instructions (s_delay_alu, s_waitcnt, etc.) - always allowed.
    return CoExecMask::CTRL;
  }

  return flavorToCoExecMask(Flavor);
}

constexpr StringRef getFlavorName(InstructionFlavor F) {
  switch (F) {
  case InstructionFlavor::WMMA:
    return "WMMA";
  case InstructionFlavor::SingleCycleVALU:
    return "VALU(1c)";
  case InstructionFlavor::TRANS:
    return "TRANS";
  case InstructionFlavor::MultiCycleVALU:
    return "VALU(Nc)";
  case InstructionFlavor::VMEM:
    return "VMEM";
  case InstructionFlavor::SMEM:
    return "SMEM";
  case InstructionFlavor::DS:
    return "DS";
  case InstructionFlavor::SALU:
    return "SALU";
  case InstructionFlavor::DMA:
    return "DMA";
  case InstructionFlavor::Fence:
    return "Fence";
  case InstructionFlavor::Other:
    return "Other";
  case InstructionFlavor::NUM_FLAVORS:
    return "???";
  }
  llvm_unreachable("Unknown InstructionFlavor");
}

/// Returns the VALU->Memory latency for a given instruction flavor.
/// Returns 0 for flavors that don't have this RAW hazard.
inline unsigned getVALUToMemLatency(InstructionFlavor Flavor) {
  switch (Flavor) {
  case InstructionFlavor::SingleCycleVALU:
  case InstructionFlavor::MultiCycleVALU:
  case InstructionFlavor::TRANS:
    return VALUToMemLatency::VALU;
  case InstructionFlavor::WMMA:
    return VALUToMemLatency::WMMA;
  default:
    return 0;
  }
}

/// Bitmask type for flavor sets. Supports up to 16 flavors.
using FlavorMask = uint16_t;

/// Convert a single flavor to its bitmask representation.
inline constexpr FlavorMask flavorBit(InstructionFlavor F) {
  return 1u << static_cast<unsigned>(F);
}

/// Predefined flavor masks for common combinations.
namespace FlavorMasks {
constexpr FlavorMask None = 0;
constexpr FlavorMask All =
    (1u << static_cast<unsigned>(InstructionFlavor::NUM_FLAVORS)) - 1;
constexpr FlavorMask AllVALU = flavorBit(InstructionFlavor::SingleCycleVALU) |
                               flavorBit(InstructionFlavor::TRANS) |
                               flavorBit(InstructionFlavor::MultiCycleVALU);
constexpr FlavorMask AllMem = flavorBit(InstructionFlavor::VMEM) |
                              flavorBit(InstructionFlavor::DS) |
                              flavorBit(InstructionFlavor::DMA);
} // namespace FlavorMasks

//===----------------------------------------------------------------------===//
// Co-execution Stage Type
//===----------------------------------------------------------------------===//

/// Stage type for co-execution (for annotation/display).
enum class CoExecStageType : uint8_t {
  NONE = 0, // Not in co-exec window
  E0,       // Issue cycle - control only
  E,        // External - MEM/SALU allowed
  I,        // Internal - MEM/SALU/VALU allowed
  IS,       // Internal + scaled-WMMA absorb (I plus next-WMMA issue)
  V,        // Vacant - MEM/SALU/WMMA allowed, no VALU
  TR        // TRANS co-exec - everything except TRANS
};

inline const char *getStageTypeName(CoExecStageType T) {
  switch (T) {
  case CoExecStageType::NONE: return "--";
  case CoExecStageType::E0:   return "E0";
  case CoExecStageType::E:    return "E";
  case CoExecStageType::I:    return "I";
  case CoExecStageType::IS:   return "IS";
  case CoExecStageType::V:    return "V";
  case CoExecStageType::TR:   return "TR";
  }
  llvm_unreachable("Unknown CoExecStageType");
}

/// Return a human-readable name for a CoExecMask bitmask value.
inline const char *getCoExecMaskName(CoExecMaskT Mask) {
  switch (Mask) {
  case CoExecMask::CTRL:  return "CTRL";
  case CoExecMask::VALU:  return "VALU";
  case CoExecMask::TRANS: return "TRANS";
  case CoExecMask::SALU:  return "SALU";
  case CoExecMask::DS:    return "DS";
  case CoExecMask::VMEM:  return "VMEM";
  case CoExecMask::SMEM:  return "SMEM";
  case CoExecMask::WMMA:  return "WMMA";
  default:                return "???";
  }
}

/// Return a single character for a CoExecMask value (for visual window logs).
inline char getCoExecMaskChar(CoExecMaskT Mask) {
  switch (Mask) {
  case CoExecMask::WMMA:  return 'W';
  case CoExecMask::VALU:  return 'V';
  case CoExecMask::TRANS: return 'T';
  case CoExecMask::SALU:  return 'S';
  case CoExecMask::DS:    return 'D';
  case CoExecMask::VMEM:  return 'M';
  case CoExecMask::SMEM:  return 'm';
  case CoExecMask::CTRL:  return 'c';
  default:                return '?';
  }
}

/// Get the CoExecMask for a COPY instruction based on register classes.
/// SGPR-to-SGPR uses SALU, everything else (VGPR↔VGPR, cross-class) uses VALU.
inline CoExecMaskT getCoExecMaskForCopy(const MachineInstr &MI,
                                        const MachineRegisterInfo &MRI,
                                        const SIRegisterInfo &TRI) {
  assert(MI.isCopy());
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();

  auto getRegClass = [&](Register Reg) -> const TargetRegisterClass * {
    if (Reg.isVirtual())
      return MRI.getRegClass(Reg);
    return TRI.getPhysRegBaseClass(Reg);
  };

  const TargetRegisterClass *DstRC = getRegClass(Dst);
  const TargetRegisterClass *SrcRC = getRegClass(Src);

  bool DstIsVGPR = DstRC && TRI.isVGPRClass(DstRC);
  bool SrcIsVGPR = SrcRC && TRI.isVGPRClass(SrcRC);

  // SGPR-to-SGPR is SALU, everything else (VGPR↔VGPR, cross-class) is VALU.
  if (!DstIsVGPR && !SrcIsVGPR)
    return CoExecMask::SALU;
  return CoExecMask::VALU;
}

/// Max stages: INT8 16x16x64 = 17 cycles, round up for safety.
constexpr unsigned MaxCoExecStages = 32;

//===----------------------------------------------------------------------===//
// Co-execution Slot Info
//===----------------------------------------------------------------------===//

/// Per-slot info including capabilities and scheduling preferences.
struct CoExecSlotInfo {
  CoExecMaskT Mask = CoExecMask::All; // What CAN execute (correctness)
  FlavorMask PreferredFlavors = FlavorMasks::None; // Flavors to prefer here
  FlavorMask AvoidedFlavors = FlavorMasks::None;   // Flavors to avoid here
  uint8_t TypeIndex = 0; // Index within type (0=first E, etc)
};

//===----------------------------------------------------------------------===//
// Co-execution Info
//===----------------------------------------------------------------------===//

/// Co-execution characteristics for a multi-cycle instruction.
/// Used by scheduler, hazard recognizer, and static simulator.
struct CoExecInfo {
  /// Cycles until unit is free (next instruction of same type can issue).
  unsigned Occupancy = 0;
  /// Total co-execution window size including tail.
  unsigned TotalWindow = 0;
  /// Per-stage slot info (mask, preferences, type index).
  CoExecSlotInfo Slots[MaxCoExecStages];
  /// Last I-stage index (for LD_SCALE rule).
  unsigned LastIStage = 0;
  /// True for FP8/FP6/FP4 scaled variants.
  bool HasScaling = false;
  /// Pattern string for display (e.g., "0EIIEEIIV").
  StringRef Pattern;

  /// Default constructor - initialize to safe defaults.
  CoExecInfo() {
    for (unsigned I = 0; I < MaxCoExecStages; ++I)
      Slots[I].Mask = CoExecMask::All; // Default: permissive
  }

  /// Get capability mask for a stage.
  CoExecMaskT getMask(unsigned Stage) const {
    return Stage < MaxCoExecStages ? Slots[Stage].Mask : CoExecMask::All;
  }

  /// Get preferred flavors for a stage.
  FlavorMask getPreferredFlavors(unsigned Stage) const {
    return Stage < TotalWindow ? Slots[Stage].PreferredFlavors
                               : FlavorMasks::None;
  }

  /// Get avoided flavors for a stage.
  FlavorMask getAvoidedFlavors(unsigned Stage) const {
    return Stage < TotalWindow ? Slots[Stage].AvoidedFlavors
                               : FlavorMasks::None;
  }

  /// Get type index for a stage (e.g., 0 = first E, 1 = second E).
  uint8_t getTypeIndex(unsigned Stage) const {
    return Stage < TotalWindow ? Slots[Stage].TypeIndex : 0;
  }

  /// Check if this is the first slot of its type.
  bool isFirstOfType(unsigned Stage) const { return getTypeIndex(Stage) == 0; }

  /// Check if slot is at a specific position within its type.
  bool isAtTypeIndex(unsigned Stage, unsigned Index) const {
    return getTypeIndex(Stage) == Index;
  }

  /// Check if a flavor is preferred at a stage.
  bool prefersFlavor(unsigned Stage, InstructionFlavor F) const {
    return (getPreferredFlavors(Stage) & flavorBit(F)) != 0;
  }

  /// Check if a flavor should be avoided at a stage.
  bool avoidsFlavor(unsigned Stage, InstructionFlavor F) const {
    return (getAvoidedFlavors(Stage) & flavorBit(F)) != 0;
  }

  /// Check if a flavor is allowed at a stage (based on slot mask).
  bool allowsFlavor(unsigned Stage, InstructionFlavor F) const {
    return (Slots[Stage].Mask & flavorBit(F)) != 0;
  }

  /// Check if instruction class mask can co-execute at a given stage.
  bool canCoExec(CoExecMaskT InstMask, unsigned Stage) const {
    if (Stage >= TotalWindow)
      return true;
    return (Slots[Stage].Mask & InstMask) != 0;
  }

  /// Find next stage where instruction class is allowed.
  std::optional<unsigned> findNextAllowedStage(CoExecMaskT InstMask,
                                               unsigned FromStage) const {
    for (unsigned I = FromStage; I < TotalWindow; ++I) {
      if ((Slots[I].Mask & InstMask) != 0)
        return I;
    }
    return std::nullopt;
  }

  unsigned getCoExecStageCount(CoExecMaskT InstMask) const {
    unsigned Counter = 0;
    for (unsigned I = 0; I < TotalWindow; ++I) {
      if ((Slots[I].Mask & InstMask) != 0)
        ++Counter;
    }
    return Counter;
  }

  /// Get stage type from mask for display.
  static CoExecStageType getStageType(CoExecMaskT Mask) {
    using namespace CoExecMask;
    if (Mask == StageE0)
      return CoExecStageType::E0;
    if (Mask == StageE)
      return CoExecStageType::E;
    if (Mask == StageIS)
      return CoExecStageType::IS;
    if (Mask == StageI)
      return CoExecStageType::I;
    if (Mask == StageV)
      return CoExecStageType::V;
    if (Mask == StageTR)
      return CoExecStageType::TR;
    // For 'All' or unknown, return based on what's allowed
    if (Mask & VALU)
      return CoExecStageType::I; // If VALU allowed, it's I-like
    if (Mask & CoExecMask::WMMA)
      return CoExecStageType::V; // If WMMA allowed (not VALU), V-like
    return CoExecStageType::E;   // Default to E
  }

  /// Get stage type for a specific stage.
  CoExecStageType getType(unsigned Stage) const {
    return getStageType(getMask(Stage));
  }

  /// Set preferred flavors for a stage. Returns *this for chaining.
  CoExecInfo &preferring(unsigned Stage, FlavorMask Flavors) {
    assert(Stage < TotalWindow);
    Slots[Stage].PreferredFlavors = Flavors;
    return *this;
  }

  /// Set avoided flavors for a stage. Returns *this for chaining.
  CoExecInfo &avoiding(unsigned Stage, FlavorMask Flavors) {
    assert(Stage < TotalWindow);
    Slots[Stage].AvoidedFlavors = Flavors;
    return *this;
  }

  /// Build a CoExecInfo from pattern string (fluent interface entry point).
  static CoExecInfo build(unsigned Occupancy, unsigned TotalWindow,
                          const char *Pattern, unsigned LastIStage,
                          bool HasScaling);

  /// Build a uniform CoExecInfo where all stages have the same slot type.
  /// Useful for TRANS and MultiCycleVALU which have simpler patterns.
  static CoExecInfo buildUniform(unsigned Occupancy, unsigned TotalWindow,
                                 char Slot, unsigned LastIStage,
                                 bool HasScaling, FlavorMask Prefer,
                                 FlavorMask Avoids);
};

//===----------------------------------------------------------------------===//
// Co-execution Info Construction
//===----------------------------------------------------------------------===//

/// Build CoExecInfo from a pattern string.
/// Pattern chars: '0'=E0, 'E'=External, 'I'=Internal, 'V'=Vacant,
///                 'S'=Internal+ScaleWMMAAbsorb (I plus next scaled WMMA),
///                 'T'=TRANS co-exec (all except TRANS), 'A'=Any
///
/// Example defining slot preferences with fluent interface:
/// \code
///   return CoExecInfo::build(8, 9, "0EIIEEIIV", 7, HasScaling)
///       .avoiding(2, flavorBit(InstructionFlavor::TRANS) |
///                    flavorBit(InstructionFlavor::MultiCycleVALU))  // I0
///       .preferring(6, flavorBit(InstructionFlavor::TRANS))         // I2
///       .preferring(8, flavorBit(InstructionFlavor::WMMA));         // V0
/// \endcode
///
/// Example scheduler usage:
/// \code
///   unsigned Stage = HazardRec->getCurrentCoExecStage();
///   const CoExecInfo &Info = HazardRec->getActiveCoExecInfo();
///   InstructionFlavor Flavor = classifyFlavor(*MI, TII);
///
///   if (Info.avoidsFlavor(Stage, Flavor)) {
///     // Deprioritize this instruction at this slot
///     return false;
///   }
///
///   // Position-aware decision: prefer TRANS on second I-slot
///   if (Info.getType(Stage) == CoExecStageType::I &&
///       Info.isAtTypeIndex(Stage, 1) &&
///       Flavor == InstructionFlavor::TRANS) {
///     return true; // Boost priority
///   }
/// \endcode
inline CoExecInfo CoExecInfo::build(unsigned Occupancy, unsigned TotalWindow,
                                    const char *Pattern, unsigned LastIStage,
                                    bool HasScaling) {
  CoExecInfo Info;
  assert(TotalWindow <= MaxCoExecStages && "Co-execution window is too long");
  assert(StringRef(Pattern).size() == TotalWindow &&
         "Pattern must describe every cycle of the co-execution window");
  Info.Occupancy = Occupancy;
  Info.TotalWindow = TotalWindow;
  Info.LastIStage = LastIStage;
  Info.HasScaling = HasScaling;
  Info.Pattern = Pattern;

  // Track count of each type for TypeIndex computation
  unsigned ECount = 0, ICount = 0, VCount = 0;

  for (unsigned I = 0; I < Info.TotalWindow; ++I) {
    switch (Pattern[I]) {
    case '0':
      Info.Slots[I].Mask = CoExecMask::StageE0;
      Info.Slots[I].TypeIndex = 0; // E0 is always unique
      break;
    case 'E':
      Info.Slots[I].Mask = CoExecMask::StageE;
      Info.Slots[I].TypeIndex = ECount++;
      break;
    case 'I':
      Info.Slots[I].Mask = CoExecMask::StageI;
      Info.Slots[I].TypeIndex = ICount++;
      break;
    case 'S':
      // I + scaled-WMMA absorb: same I-flavor capacity as 'I' plus the
      // ability for the next scaled WMMA to issue here. Counts toward
      // ICount so getTypeIndex(...) reports it as "the Nth I slot".
      Info.Slots[I].Mask = CoExecMask::StageIS;
      Info.Slots[I].TypeIndex = ICount++;
      break;
    case 'V':
      Info.Slots[I].Mask = CoExecMask::StageV;
      Info.Slots[I].TypeIndex = VCount++;
      break;
    case 'T':
      Info.Slots[I].Mask = CoExecMask::StageTR;
      Info.Slots[I].TypeIndex = 0;
      break;
    case 'A':
    default:
      Info.Slots[I].Mask = CoExecMask::All;
      Info.Slots[I].TypeIndex = 0;
      break;
    }
  }
  return Info;
}

/// Build a uniform CoExecInfo where all stages have the same slot type.
inline CoExecInfo CoExecInfo::buildUniform(unsigned Occupancy,
                                           unsigned TotalWindow, char Slot,
                                           unsigned LastIStage, bool HasScaling,
                                           FlavorMask Prefer,
                                           FlavorMask Avoids) {
  CoExecInfo Info;
  assert(TotalWindow <= MaxCoExecStages && "Co-execution window is too long");
  Info.Occupancy = Occupancy;
  Info.TotalWindow = TotalWindow;
  Info.LastIStage = LastIStage;
  Info.HasScaling = HasScaling;

  unsigned ECount = 0, ICount = 0, VCount = 0;

  for (unsigned I = 0; I < Info.TotalWindow; ++I) {
    Info.Slots[I].PreferredFlavors = Prefer;
    Info.Slots[I].AvoidedFlavors = Avoids;

    switch (Slot) {
    case '0':
      Info.Slots[I].Mask = CoExecMask::StageE0;
      Info.Slots[I].TypeIndex = 0;
      break;
    case 'E':
      Info.Slots[I].Mask = CoExecMask::StageE;
      Info.Slots[I].TypeIndex = ECount++;
      break;
    case 'I':
      Info.Slots[I].Mask = CoExecMask::StageI;
      Info.Slots[I].TypeIndex = ICount++;
      break;
    case 'S':
      Info.Slots[I].Mask = CoExecMask::StageIS;
      Info.Slots[I].TypeIndex = ICount++;
      break;
    case 'V':
      Info.Slots[I].Mask = CoExecMask::StageV;
      Info.Slots[I].TypeIndex = VCount++;
      break;
    case 'T':
      Info.Slots[I].Mask = CoExecMask::StageTR;
      Info.Slots[I].TypeIndex = 0;
      break;
    case 'A':
    default:
      Info.Slots[I].Mask = CoExecMask::All;
      Info.Slots[I].TypeIndex = 0;
      break;
    }
  }
  return Info;
}

/// Get co-execution info for a gfx950 MFMA instruction.
CoExecInfo getMFMACoExecInfo(unsigned Opcode);

/// Get co-execution info for a multi-cycle instruction.
/// For WMMA: returns detailed pattern based on opcode.
/// For TRANS/MultiCycleVALU: returns uniform pattern based on repeat rate.
inline CoExecInfo getCoExecInfo(const MachineInstr &MI,
                                const SIInstrInfo &TII) {
  // Handle TRANS and MultiCycleVALU instructions first.
  if (!TII.isMFMAorWMMA(MI)) {
    unsigned RepeatRate = TII.getRepeatRate(MI);
    if (RepeatRate <= 1)
      return CoExecInfo();

    bool IsTRANS = TII.isTRANS(MI);
    // TRANS: uses 'T' slots (all except TRANS), prefers SingleCycleVALU.
    // MultiCycleVALU: uses 'E' slots (external, no VALU).
    return CoExecInfo::buildUniform(
        RepeatRate - 1, RepeatRate - 1, IsTRANS ? 'T' : 'E',
        IsTRANS ? RepeatRate - 2 : 0, false,
        IsTRANS ? flavorBit(InstructionFlavor::SingleCycleVALU) : FlavorMask(),
        FlavorMask());
  }

  StringRef Name = TII.getName(MI.getOpcode());

  // Check for scaled variants (LD_SCALE rule applies)
  bool HasScaling = Name.contains_insensitive("scale");

  if (TII.isMFMA(MI))
    return getMFMACoExecInfo(MI.getOpcode());

  if (Name.contains_insensitive("16x16x64_iu8")) {
    return CoExecInfo::build(16, 17, "0EIIEEIIEEIIEEIIV", 15, HasScaling);
  }

  // F8F6F4 16x16x128: window size depends on operand formats
  if (Name.contains_insensitive("16x16x128_f8f6f4")) {
    // Check if both operands are FP4 (shorter window)
    bool BothF4 = false;
    if (const MachineOperand *FmtA =
            TII.getNamedOperand(MI, AMDGPU::OpName::matrix_a_fmt)) {
      if (const MachineOperand *FmtB =
              TII.getNamedOperand(MI, AMDGPU::OpName::matrix_b_fmt)) {
        BothF4 = (FmtA->getImm() == AMDGPU::WMMA::MATRIX_FMT_FP4 &&
                  FmtB->getImm() == AMDGPU::WMMA::MATRIX_FMT_FP4);
      }
    }

    if (BothF4) {
      // f4×f4: 4-cycle occupancy, 6-cycle window. The scaled variant uses
      // 'S' at the last I — the next scaled WMMA's LD_SCALE absorbs there.
      const char *Pattern = HasScaling ? "0EESVV" : "0EEIVV";
      return CoExecInfo::build(4, 6, Pattern, 3, HasScaling)
          .preferring(1, flavorBit(InstructionFlavor::DS))
          .avoiding(2, flavorBit(InstructionFlavor::DS))
          .preferring(3, HasScaling
                             ? flavorBit(InstructionFlavor::WMMA)
                             : flavorBit(InstructionFlavor::SingleCycleVALU))
          .preferring(4, flavorBit(InstructionFlavor::WMMA))
          .preferring(5, flavorBit(InstructionFlavor::WMMA));
    }
    // f8×*, f6×*, or mixed: 8-cycle occupancy, 10-cycle window.
    // Non-scaled: 0EEIEEIIVV - stages 3,6,7 are I slots (VALU/TRANS allowed)
    // Scaled:     0EEIEEISVV - stage 7 is S slot (scaled WMMA)
    const char *Pattern = HasScaling ? "0EEIEEISVV" : "0EEIEEIIVV";
    return CoExecInfo::build(8, 10, Pattern, 7, HasScaling)
        .preferring(1, flavorBit(InstructionFlavor::DS))
        .avoiding(2, flavorBit(InstructionFlavor::DS))
        .preferring(3, flavorBit(InstructionFlavor::TRANS))
        .avoiding(3, flavorBit(InstructionFlavor::DS) |
                         flavorBit(InstructionFlavor::SALU) |
                         flavorBit(InstructionFlavor::VMEM))
        .preferring(4, flavorBit(InstructionFlavor::DS))
        .avoiding(5, flavorBit(InstructionFlavor::DS))
        .preferring(6, flavorBit(InstructionFlavor::SingleCycleVALU))
        .avoiding(6, HasScaling ? flavorBit(InstructionFlavor::TRANS)
                                : FlavorMasks::None)
        // Stage 7: For non-scaled, this is an I slot - prefer TRANS here
        // (last I slot before V slots). For scaled, it's S slot for WMMA.
        .preferring(7, HasScaling ? flavorBit(InstructionFlavor::WMMA)
                                  : flavorBit(InstructionFlavor::TRANS))
        .preferring(8, flavorBit(InstructionFlavor::WMMA))
        .preferring(9, flavorBit(InstructionFlavor::WMMA));
  }

  // FP8/BF8 16x16x64: 4-cycle occupancy, 6-cycle window
  if (Name.contains_insensitive("16x16x64_fp8") ||
      Name.contains_insensitive("16x16x64_bf8")) {
    const char *Pattern = HasScaling ? "0EESVV" : "0EEIVV";
    return CoExecInfo::build(4, 6, Pattern, 3, HasScaling)
        .preferring(1, flavorBit(InstructionFlavor::DS))
        .avoiding(2, flavorBit(InstructionFlavor::DS))
        .preferring(3, HasScaling
                           ? flavorBit(InstructionFlavor::WMMA)
                           : flavorBit(InstructionFlavor::SingleCycleVALU))
        .preferring(4, flavorBit(InstructionFlavor::WMMA))
        .preferring(5, flavorBit(InstructionFlavor::WMMA));
  }

  // F16/BF16 16x16x32: 8-cycle occupancy, 9-cycle window
  if (Name.contains_insensitive("16x16x32_f16") ||
      Name.contains_insensitive("16x16x32_bf16")) {
    const char *Pattern = HasScaling ? "0EIIEEISV" : "0EIIEEIIV";
    return CoExecInfo::build(8, 9, Pattern, 7, HasScaling)
        .avoiding(1, flavorBit(InstructionFlavor::DS))
        .preferring(2, flavorBit(InstructionFlavor::SingleCycleVALU))
        .preferring(3, flavorBit(InstructionFlavor::TRANS))
        .avoiding(3, flavorBit(InstructionFlavor::DS) |
                         flavorBit(InstructionFlavor::SALU) |
                         flavorBit(InstructionFlavor::VMEM))
        .preferring(4, flavorBit(InstructionFlavor::DS))
        .avoiding(5, flavorBit(InstructionFlavor::DS))
        .preferring(6, flavorBit(InstructionFlavor::SingleCycleVALU))
        .avoiding(6, HasScaling ? flavorBit(InstructionFlavor::TRANS)
                                : FlavorMasks::None)
        // Stage 7: For non-scaled, this is an I slot - prefer TRANS here
        // (last I slot before V slots). For scaled, it's S slot for WMMA.
        .preferring(7, HasScaling ? flavorBit(InstructionFlavor::WMMA)
                                  : flavorBit(InstructionFlavor::TRANS))
        .preferring(8, flavorBit(InstructionFlavor::WMMA));
  }

  // FP8/BF8 16x16x128: 8-cycle occupancy, 10-cycle window
  if (Name.contains_insensitive("16x16x128_fp8") ||
      Name.contains_insensitive("16x16x128_bf8")) {
    const char *Pattern = HasScaling ? "0EEIEEISVV" : "0EEIEEIIVV";
    return CoExecInfo::build(8, 10, Pattern, 7, HasScaling)
        .preferring(1, flavorBit(InstructionFlavor::DS))
        .avoiding(2, flavorBit(InstructionFlavor::DS))
        .preferring(3, flavorBit(InstructionFlavor::TRANS))
        .avoiding(3, flavorBit(InstructionFlavor::DS) |
                         flavorBit(InstructionFlavor::SALU) |
                         flavorBit(InstructionFlavor::VMEM))
        .preferring(4, flavorBit(InstructionFlavor::DS))
        .avoiding(5, flavorBit(InstructionFlavor::DS))
        .preferring(6, flavorBit(InstructionFlavor::SingleCycleVALU))
        .avoiding(6, HasScaling ? flavorBit(InstructionFlavor::TRANS)
                                : FlavorMasks::None)
        // Stage 7: For non-scaled, this is an I slot - prefer TRANS here
        // (last I slot before V slots). For scaled, it's S slot for WMMA.
        .preferring(7, HasScaling ? flavorBit(InstructionFlavor::WMMA)
                                  : flavorBit(InstructionFlavor::TRANS))
        .preferring(8, flavorBit(InstructionFlavor::WMMA))
        .preferring(9, flavorBit(InstructionFlavor::WMMA));
  }

  // 32x16x128 F4 variants
  if (Name.contains_insensitive("32x16x128_f4")) {
    const char *Pattern = HasScaling ? "0EEIEIESVV" : "0EEIEIEIVV";
    return CoExecInfo::build(8, 10, Pattern, 7, HasScaling)
        .preferring(1, flavorBit(InstructionFlavor::DS))
        .avoiding(2, flavorBit(InstructionFlavor::DS))
        .preferring(3, flavorBit(InstructionFlavor::TRANS))
        .avoiding(3, flavorBit(InstructionFlavor::DS) |
                         flavorBit(InstructionFlavor::SALU) |
                         flavorBit(InstructionFlavor::VMEM))
        .avoiding(4, flavorBit(InstructionFlavor::DS))
        .preferring(5, flavorBit(InstructionFlavor::TRANS))
        .avoiding(5, flavorBit(InstructionFlavor::DS) |
                         flavorBit(InstructionFlavor::SALU) |
                         flavorBit(InstructionFlavor::VMEM))
        .avoiding(6, flavorBit(InstructionFlavor::DS))
        // Stage 7: For non-scaled, this is an I slot - prefer TRANS here
        // (last I slot before V slots). For scaled, it's S slot for WMMA.
        .preferring(7, HasScaling ? flavorBit(InstructionFlavor::WMMA)
                                  : flavorBit(InstructionFlavor::TRANS))
        .preferring(8, flavorBit(InstructionFlavor::WMMA))
        .preferring(9, flavorBit(InstructionFlavor::WMMA));
  }

  // Default fallback: permissive 8-cycle pattern
  return CoExecInfo::build(8, 9, "AAAAAAAAA", 7, HasScaling);
}

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECINFO_H
