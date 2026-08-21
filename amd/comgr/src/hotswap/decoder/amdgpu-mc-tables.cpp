//===- amdgpu-mc-tables.cpp - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amdgpu-mc-tables.h"

// AMDGPU target-private headers.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"

#include <cstdint>

namespace COMGR::hotswap {

// Nesting `llvm::AMDGPU` here keeps these definitions distinct from the ones a
// static build puts on the same link line; the using-directive is what lets the
// generated code's unqualified lookups still reach the real namespace.
namespace tables {
using namespace ::llvm::AMDGPU;
#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRMAP_INFO
#include "AMDGPUGenInstrInfo.inc"
} // namespace tables

int16_t getNamedOperandIdx(uint32_t Opcode, llvm::AMDGPU::OpName Name) {
  return tables::llvm::AMDGPU::getNamedOperandIdx(Opcode, Name);
}

int32_t getMCOpcode(uint32_t Opcode, unsigned Gen) {
  using Subtarget = tables::llvm::AMDGPU::Subtarget;
  return tables::llvm::AMDGPU::getMCOpcodeGen(Opcode,
                                              static_cast<Subtarget>(Gen));
}

int32_t getVOPe64(uint32_t Opcode) {
  return tables::llvm::AMDGPU::getVOPe64(Opcode);
}

int32_t getDPPOp32(uint32_t Opcode) {
  return tables::llvm::AMDGPU::getDPPOp32(Opcode);
}

int32_t getDPPOp64(uint32_t Opcode) {
  return tables::llvm::AMDGPU::getDPPOp64(Opcode);
}

int32_t getBasicFromSDWAOp(uint32_t Opcode) {
  return tables::llvm::AMDGPU::getBasicFromSDWAOp(Opcode);
}

int32_t getGlobalVaddrOp(uint32_t Opcode) {
  return tables::llvm::AMDGPU::getGlobalVaddrOp(Opcode);
}

} // namespace COMGR::hotswap
