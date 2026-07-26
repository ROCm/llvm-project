//===- AMDGPUOperandInfo.h - AMDGPU MC operand info -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Stable, target-specific access to named operands in AMDGPU MCInsts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_AMDGPU_AMDGPUOPERANDINFO_H
#define LLVM_TARGET_AMDGPU_AMDGPUOPERANDINFO_H

#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <optional>

namespace llvm {

class MCInst;

namespace AMDGPU {

/// Semantic roles exposed by AMDGPU's generated named-operand tables.
///
/// This enum deliberately insulates MC clients from the generated OpName enum,
/// which is a backend implementation detail and is not installed with LLVM.
///
/// The explicit numeric values are stable public identifiers. Existing values
/// must not be renumbered or reused; new roles must be appended.
enum class MCOperandRole : uint8_t {
  VDst = 0,
  Src0 = 1,
  Src1 = 2,
  Src2 = 3,
  Src0Modifiers = 4,
  Src1Modifiers = 5,
  Src2Modifiers = 6,
  ScaleSrc0 = 7,
  ScaleSrc1 = 8,
  MatrixAFormat = 9,
  MatrixBFormat = 10,
  MatrixAScale = 11,
  MatrixBScale = 12,
  MatrixAScaleFormat = 13,
  MatrixBScaleFormat = 14,
  MatrixAReuse = 15,
  MatrixBReuse = 16,
  NegLo = 17,
  NegHi = 18,
  Clamp = 19,
  OMod = 20,
  OpSel = 21,
  OpSelHi = 22,
};

/// Return the operand index for \p Role in the AMDGPU \p Inst.
///
/// Returns std::nullopt when the opcode is invalid, the role is not present,
/// the role value is not recognized, or the operand was not materialized in
/// this MCInst. The latter can occur when a decoder omits trailing operands
/// whose values are implicit in the encoding.
LLVM_ABI std::optional<unsigned> getMCOperandIndex(const MCInst &Inst,
                                                   MCOperandRole Role);

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_TARGET_AMDGPU_AMDGPUOPERANDINFO_H
