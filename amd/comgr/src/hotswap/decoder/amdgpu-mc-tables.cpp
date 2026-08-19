//===- amdgpu-mc-tables.cpp - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGPUBaseInfo.h declares the named-operand lookup and the opcode-form
// mappings the decoder uses, but libLLVM.so exports neither, so a dylib build
// of amd_comgr cannot resolve them. Instantiate the tables here from the same
// generated header AMDGPUBaseInfo.cpp instantiates them from, which keeps them
// in step with the AMDGPU target the rest of the decoder already builds
// against.
//
// A static build resolves the definitions from LLVMAMDGPUUtils and would
// reject these as duplicates, so CMakeLists.txt compiles this file only when
// comgr links the LLVM dylib.
//
//===----------------------------------------------------------------------===//

// GET_INSTRINFO_ENUM, for the opcode enumerators the tables are indexed by.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
// GET_INSTRINFO_OPERAND_ENUM, for the OpName the lookup is keyed by.
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/Support/Compiler.h"

#include <cstdint>

#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRMAP_INFO
#include "AMDGPUGenInstrInfo.inc"

namespace llvm::AMDGPU {

// The generated lookup takes the encoding family as `enum Subtarget`, which is
// declared by the same block that defines the lookup and so cannot be named by
// a caller. AMDGPUBaseInfo.h declares this unsigned-taking wrapper for them.
int32_t getMCOpcode(uint32_t Opcode, unsigned Gen) {
  return getMCOpcodeGen(Opcode, static_cast<Subtarget>(Gen));
}

} // namespace llvm::AMDGPU
