//===- AMDGPUOperandInfo.h - Public AMDGPU MC operand roles ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes a stable subset of AMDGPU's TableGen named-operand
// metadata to clients that operate on MCInsts without access to the
// backend-private AMDGPUBaseInfo.h header.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_AMDGPUOPERANDINFO_H
#define LLVM_TARGETPARSER_AMDGPUOPERANDINFO_H

#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace llvm {
namespace AMDGPU {

enum class MCNamedOperand : uint8_t {
  VDst,
  Src0Modifiers,
  Src0,
  Src1Modifiers,
  Src1,
  Src2Modifiers,
  Src2,
  ScaleSrc0,
  ScaleSrc1,
  Clamp,
  MatrixAFmt,
  MatrixBFmt,
  MatrixAScale,
  MatrixBScale,
  MatrixAScaleFmt,
  MatrixBScaleFmt,
  MatrixAReuse,
  MatrixBReuse,
  NegLo,
  NegHi,
};

/// Return the MCInst operand index for \p Name in \p Opcode, or -1 when that
/// opcode does not carry the named operand.
LLVM_ABI int16_t getNamedOperandIdx(uint32_t Opcode, MCNamedOperand Name);

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_TARGETPARSER_AMDGPUOPERANDINFO_H
