//===- amdgpu-mc-tables.h - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AMDGPU TableGen lookups the decoder needs: the index of a named operand,
// and the opcode of the other encoding forms of an instruction. Comgr carries
// its own copy of these tables because libLLVM.so exports none of them.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_AMDGPU_MC_TABLES_H
#define HOTSWAP_TRANSPILER_AMDGPU_MC_TABLES_H

#include "Utils/AMDGPUBaseInfo.h"

#include <cstdint>

namespace COMGR::hotswap {

/// Index of the operand named `Name` in `Opcode`, or -1 if it has no operand of
/// that name.
int16_t getNamedOperandIdx(uint32_t Opcode, llvm::AMDGPU::OpName Name);

/// The opcode `Opcode` encodes as on encoding family `Gen` (an
/// `AMDGPUEncodingFamily`), or -1 if that family does not encode it.
int32_t getMCOpcode(uint32_t Opcode, unsigned Gen);

/// The VOP3 form of the VOP1/VOP2/VOPC opcode `Opcode`, or -1 if it has none.
int32_t getVOPe64(uint32_t Opcode);

/// The DPP form of `Opcode` at the given width, or -1 if it has none.
int32_t getDPPOp32(uint32_t Opcode);
int32_t getDPPOp64(uint32_t Opcode);

/// The non-SDWA form of the SDWA opcode `Opcode`, or -1 if it has none.
int32_t getBasicFromSDWAOp(uint32_t Opcode);

/// The vaddr form of the saddr FLAT/global opcode `Opcode`, or -1 if it has
/// none.
int32_t getGlobalVaddrOp(uint32_t Opcode);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_AMDGPU_MC_TABLES_H
