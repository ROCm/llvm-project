//===- AMDGPUMCOperand.h - Public AMDGPU MC operand names -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Public access to the named-operand metadata generated for AMDGPU
/// instructions. This keeps MC clients from depending on target-private
/// TableGen headers or fixed operand positions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_AMDGPUMCOPERAND_H
#define LLVM_TARGETPARSER_AMDGPUMCOPERAND_H

#include "llvm/Support/Compiler.h"

#include <cstdint>

namespace llvm::AMDGPU {

enum class NamedOperand : uint8_t {
  VAddr1,
  SDst,
  Src0,
  Src1,
  Src2,
  Count,
};

/// Return the MC operand index associated with \p Name for \p Opcode, or -1
/// when the instruction does not define that named operand. Concrete
/// subtarget opcodes inherit operand names from their generated pseudo-opcode
/// sources.
LLVM_ABI int16_t getNamedOperandIdx(uint32_t Opcode, NamedOperand Name);

} // namespace llvm::AMDGPU

#endif // LLVM_TARGETPARSER_AMDGPUMCOPERAND_H
