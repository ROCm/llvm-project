//===- AMDGPUSim/SimInst.h - Abstract Instruction Type ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the abstract instruction type (SimInst) for the AMDGPU static
/// simulator library. SimInst is a lightweight wrapper holding a pointer to
/// the actual instruction plus cached basic properties. Detailed property
/// queries go through the SimInstInfo interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINST_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINST_H

#include <cstdint>

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Instruction Classification
//===----------------------------------------------------------------------===//

/// Instruction classification for simulation.
enum class InstClass : uint8_t {
  VALU,
  SALU,
  TRANS, // Transcendentals (V_EXP, V_LOG, V_RCP, V_RSQ, V_SQRT, etc.)
  WMMA,  // Matrix multiply
  DS_READ,
  DS_WRITE,
  VMEM_READ,
  VMEM_WRITE,
  SMEM,
  TDM, // Tensor DMA (TENSOR_LOAD_TO_LDS etc.)
  BARRIER,
  WAITCNT,
  DELAY_ALU,
  MSB_SET, // s_set_vgpr_msb (gfx1250 overhead for >256 VGPRs)
  NOP,
  BRANCH,
  OTHER
};

//===----------------------------------------------------------------------===//
// Functional Units
//===----------------------------------------------------------------------===//

/// Hardware functional units for per-unit busy tracking.
enum class FunctionalUnit : uint8_t {
  NONE = 0, // No unit (NOPs, WAITCNTs, BARRIERs) - no busy tracking
  XDL,      // Matrix/WMMA unit
  VALU,     // Vector ALU
  SALU,     // Scalar ALU
  TRANS,    // Transcendental unit
  LDS,      // Local Data Share
  VMEM,     // Global memory unit
  SMEM,     // Scalar memory unit
  BRANCH,   // Branch unit
  NUM_UNITS
};

/// Get human-readable name for an InstClass.
inline const char *getInstClassName(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:
    return "VALU";
  case InstClass::SALU:
    return "SALU";
  case InstClass::TRANS:
    return "TRANS";
  case InstClass::WMMA:
    return "WMMA";
  case InstClass::DS_READ:
    return "DS_READ";
  case InstClass::DS_WRITE:
    return "DS_WRITE";
  case InstClass::VMEM_READ:
    return "VMEM_READ";
  case InstClass::VMEM_WRITE:
    return "VMEM_WRITE";
  case InstClass::SMEM:
    return "SMEM";
  case InstClass::TDM:
    return "TDM";
  case InstClass::BARRIER:
    return "BARRIER";
  case InstClass::WAITCNT:
    return "WAITCNT";
  case InstClass::DELAY_ALU:
    return "DELAY_ALU";
  case InstClass::MSB_SET:
    return "MSB_SET";
  case InstClass::NOP:
    return "NOP";
  case InstClass::BRANCH:
    return "BRANCH";
  case InstClass::OTHER:
    return "OTHER";
  }
  return "UNKNOWN";
}

/// Get human-readable name for a FunctionalUnit.
inline const char *getUnitName(FunctionalUnit U) {
  switch (U) {
  case FunctionalUnit::NONE:
    return "NONE";
  case FunctionalUnit::XDL:
    return "XDL";
  case FunctionalUnit::VALU:
    return "VALU";
  case FunctionalUnit::SALU:
    return "SALU";
  case FunctionalUnit::TRANS:
    return "TRANS";
  case FunctionalUnit::LDS:
    return "LDS";
  case FunctionalUnit::VMEM:
    return "VMEM";
  case FunctionalUnit::SMEM:
    return "SMEM";
  case FunctionalUnit::BRANCH:
    return "BRANCH";
  case FunctionalUnit::NUM_UNITS:
    return "NUM_UNITS";
  }
  return "UNKNOWN";
}

/// Map InstClass to FunctionalUnit.
inline FunctionalUnit getUnitForClass(InstClass IC) {
  switch (IC) {
  case InstClass::WMMA:
    return FunctionalUnit::XDL;
  case InstClass::VALU:
    return FunctionalUnit::VALU;
  case InstClass::TRANS:
    return FunctionalUnit::TRANS;
  case InstClass::SALU:
  case InstClass::DELAY_ALU:
  case InstClass::MSB_SET:
    return FunctionalUnit::SALU;
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::TDM:
    return FunctionalUnit::LDS;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return FunctionalUnit::VMEM;
  case InstClass::SMEM:
    return FunctionalUnit::SMEM;
  case InstClass::BRANCH:
    return FunctionalUnit::BRANCH;
  case InstClass::NOP:
  case InstClass::WAITCNT:
  case InstClass::BARRIER:
  case InstClass::OTHER:
    return FunctionalUnit::NONE;
  }
  return FunctionalUnit::NONE;
}

//===----------------------------------------------------------------------===//
// WMMA Stage Types
//===----------------------------------------------------------------------===//

/// Stage type for WMMA co-execution windows.
enum class WMMAStageType : uint8_t {
  NONE = 0, // Not in WMMA window
  E0,       // Issue cycle - control only (s_delay_alu, s_set_vgpr_msb)
  E,        // External - MEM/SALU allowed, no VALU/TRANS
  I,        // Internal - MEM/SALU/VALU/TRANS all allowed
  V         // Vacant - MEM/SALU/WMMA allowed, no VALU/TRANS
};

//===----------------------------------------------------------------------===//
// Wait Instruction Types
//===----------------------------------------------------------------------===//

/// Wait instruction type for counter-based waits.
enum class WaitType : uint8_t {
  None = 0,
  DS,        // s_wait_dscnt
  VMEMLoad,  // s_wait_loadcnt
  VMEMStore, // s_wait_storecnt
  SMEM,      // s_wait_kmcnt
  Tensor,    // s_wait_tensorcnt
  XCnt,      // s_wait_xcnt
  DepCtr     // s_waitcnt_depctr (va_vdst)
};

//===----------------------------------------------------------------------===//
// WMMA Variants
//===----------------------------------------------------------------------===//

/// WMMA instruction variant for co-execution rule selection.
enum class WMMAVariant : uint8_t {
  Default = 0,
  IU8_16x16x64,
  F8F6F4_16x16x128,
  F8F6F4_16x16x128_BothF4,
  FP8_16x16x64,
  BF8_16x16x64,
  F16_16x16x32,
  BF16_16x16x32,
  FP8_16x16x128,
  BF8_16x16x128,
  F4_32x16x128,
};

//===----------------------------------------------------------------------===//
// Register Operand Info
//===----------------------------------------------------------------------===//

/// Register operand info for bank conflict analysis.
struct RegOperand {
  enum class Type : uint8_t { VGPR, SGPR, Other };

  Type RegType = Type::Other;
  unsigned HWIndex = 0;       // Hardware register index
  unsigned NumComponents = 1; // Number of 32-bit components

  RegOperand() = default;
  RegOperand(Type T, unsigned Idx, unsigned N = 1)
      : RegType(T), HWIndex(Idx), NumComponents(N) {}
};

//===----------------------------------------------------------------------===//
// Abstract Instruction (Lightweight Wrapper)
//===----------------------------------------------------------------------===//

/// Lightweight instruction wrapper for simulation.
/// Holds a pointer to the actual instruction plus cached basic properties.
/// Detailed property queries go through the SimInstInfo interface.
struct SimInst {
  /// Pointer to the actual instruction (MachineInstr*, MCInst*, etc.)
  void *InstPtr = nullptr;

  /// Cached basic properties (populated by SimInstInfo::createSimInst)
  InstClass Class = InstClass::OTHER;
  unsigned Latency = 1;
  FunctionalUnit Unit = FunctionalUnit::NONE;

  /// Optional instruction identifier for debugging/logging
  unsigned InstIndex = 0;

  SimInst() = default;
  SimInst(void *Ptr, InstClass C, unsigned Lat, FunctionalUnit U,
          unsigned Idx = 0)
      : InstPtr(Ptr), Class(C), Latency(Lat), Unit(U), InstIndex(Idx) {}

  /// Get the instruction pointer cast to the appropriate type.
  template <typename T> T *getAs() const { return static_cast<T *>(InstPtr); }
};

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINST_H
