//===- AMDGPUSim/SimInstInfo.h - Instruction Info Interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the abstract SimInstInfo interface for querying instruction
/// properties. Concrete implementations (MachineInstrInfo, MCInstInfo) provide
/// the actual property extraction from the underlying instruction type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINSTINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINSTINFO_H

#include "SimInst.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace llvm {
namespace AMDGPUSim {

/// Abstract interface for querying instruction properties.
/// Similar in concept to SIInstrInfo, but operating on SimInst.
/// Concrete implementations know how to extract properties from the
/// underlying instruction type (MachineInstr, MCInst, etc.).
class SimInstInfo {
public:
  virtual ~SimInstInfo() = default;

  //===--------------------------------------------------------------------===//
  // VALU/TRANS Properties
  //===--------------------------------------------------------------------===//

  /// Get the repeat rate (1 for normal VALU, >1 for long-latency VALU).
  virtual unsigned getRepeatRate(const SimInst &SI) const = 0;

  /// Check if this is a long-latency VALU instruction.
  virtual bool isLOLVALU(const SimInst &SI) const = 0;

  /// Check if this is a TRANS instruction (explicit flag for hazard tracking).
  virtual bool isTRANS(const SimInst &SI) const = 0;

  /// Get resource cycles (how long the functional unit is busy).
  virtual unsigned getResourceCycles(const SimInst &SI) const = 0;

  //===--------------------------------------------------------------------===//
  // delay_alu Properties
  //===--------------------------------------------------------------------===//

  /// Get the encoded delay_alu immediate value.
  virtual unsigned getDelayAluImm(const SimInst &SI) const = 0;

  //===--------------------------------------------------------------------===//
  // Wait Instruction Properties
  //===--------------------------------------------------------------------===//

  /// Get wait type and count from a wait instruction.
  virtual std::pair<WaitType, unsigned>
  getWaitInfo(const SimInst &SI) const = 0;

  /// Get VA_VDST target for s_waitcnt_depctr (15 = don't wait).
  virtual unsigned getVaVdstTarget(const SimInst &SI) const = 0;

  //===--------------------------------------------------------------------===//
  // Memory Instruction Properties
  //===--------------------------------------------------------------------===//

  /// Get destination register base index and count for memory ops.
  /// Returns {BaseIndex, NumRegs}.
  virtual std::pair<unsigned, unsigned> getDestRegInfo(const SimInst &SI,
                                                       bool IsVGPR) const = 0;

  //===--------------------------------------------------------------------===//
  // WMMA Properties
  //===--------------------------------------------------------------------===//

  /// Get the WMMA variant for co-execution rule selection.
  virtual WMMAVariant getWMMAVariant(const SimInst &SI) const = 0;

  /// Check if this is a scaled WMMA instruction.
  virtual bool hasScaling(const SimInst &SI) const = 0;

  //===--------------------------------------------------------------------===//
  // Register Properties
  //===--------------------------------------------------------------------===//

  /// Check if the instruction has explicit SGPR operands.
  virtual bool hasSGPROperands(const SimInst &SI) const = 0;

  /// Get source register operands for bank conflict analysis.
  /// Includes all explicit_uses (with non-register placeholders) to preserve
  /// port indexing for the VGPR source cache.
  virtual void getSrcRegs(const SimInst &SI,
                          SmallVectorImpl<RegOperand> &Regs) const = 0;

  /// Get WMMA source register operands for cache tracking.
  /// Only returns the A and B matrix VGPR sources (src0, src1),
  /// skipping C (tied-def) and scale registers.
  virtual void getWMMASrcRegs(const SimInst &SI,
                              SmallVectorImpl<RegOperand> &Regs) const = 0;

  /// Get destination register operands.
  virtual void getDstRegs(const SimInst &SI,
                          SmallVectorImpl<RegOperand> &Regs) const = 0;

  //===--------------------------------------------------------------------===//
  // Counting Flags
  //===--------------------------------------------------------------------===//

  /// Check if this is a VOPD instruction.
  virtual bool isVOPD(const SimInst &SI) const = 0;

  /// Check if this is a packed instruction.
  virtual bool isPacked(const SimInst &SI) const = 0;

  //===--------------------------------------------------------------------===//
  // Scoreboard Properties
  //===--------------------------------------------------------------------===//

  /// Check if instruction implicitly waits for all VALU to complete
  /// (VA_VDST==0). Used for scoreboard clearing.
  /// Instructions that wait: DS, EXP, FLAT, MIMG, MTBUF, MUBUF, etc.
  virtual bool waitsForVALU(const SimInst &SI) const = 0;

  //===--------------------------------------------------------------------===//
  // Miscellaneous
  //===--------------------------------------------------------------------===//

  /// Get instruction size in bytes.
  virtual unsigned getInstBytes(const SimInst &SI) const = 0;
};

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMINSTINFO_H
