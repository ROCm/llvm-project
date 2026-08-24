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

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

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

/// The operand indices the four two-bit fields of S_SET_VGPR_MSB apply to.
/// Elements 0 through 3 correspond to src0, src1, src2 and dst. Each element
/// holds the X and Y operand index of one field; the Y index is absent for
/// everything but VOPD.
using VGPRMSBOperandIndices =
    std::array<std::pair<std::optional<unsigned>, std::optional<unsigned>>, 4>;

/// The operand indices the S_SET_VGPR_MSB fields select for `Desc`. An index is
/// absent when the instruction has no operand in that slot.
VGPRMSBOperandIndices getVGPRMSBOperandIndices(const llvm::MCInstrDesc &Desc);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_AMDGPU_MC_TABLES_H
