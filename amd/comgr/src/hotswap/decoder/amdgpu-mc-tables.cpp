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
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"

#include <cassert>
#include <cstdint>

namespace COMGR::hotswap {

// Nesting `llvm::AMDGPU` here keeps these definitions distinct from the ones a
// static build puts on the same link line; the using-directive is what lets the
// generated code's unqualified lookups still reach the real namespace.
namespace tables {
using namespace ::llvm::AMDGPU;
using ::llvm::ArrayRef;

// Row types for the searchable tables below. The emitter expects the including
// translation unit to have declared them.
struct VOPDComponentInfo {
  uint16_t BaseVOP;
  uint16_t VOPDOp;
};

struct VOPDInfo {
  uint32_t Opcode;
  uint16_t OpX;
  uint16_t OpY;
  uint16_t Subtarget;
  bool VOPD3;
};

#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRMAP_INFO
#include "AMDGPUGenInstrInfo.inc"

#define GET_VOPDComponentTable_DECL
#define GET_VOPDComponentTable_IMPL
#define GET_VOPDPairs_DECL
#define GET_VOPDPairs_IMPL
#include "AMDGPUGenSearchableTables.inc"
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

// The index of the operand `OperandNames` names for `Slot`, or nullopt when the
// table leaves that slot empty or the instruction has no such operand.
static std::optional<unsigned>
operandIndexForSlot(const llvm::MCInstrDesc &Desc,
                    const llvm::AMDGPU::OpName *OperandNames, unsigned Slot) {
  if (!OperandNames ||
      OperandNames[Slot] == llvm::AMDGPU::OpName::NUM_OPERAND_NAMES)
    return std::nullopt;
  // Qualified because ADL on the argument type also finds the LLVM overload,
  // the one a dylib build does not export.
  int16_t OperandIndex =
      COMGR::hotswap::getNamedOperandIdx(Desc.getOpcode(), OperandNames[Slot]);
  if (OperandIndex < 0)
    return std::nullopt;
  return static_cast<unsigned>(OperandIndex);
}

// The operand names the four S_SET_VGPR_MSB fields address, per instruction
// format. A slot the format does not use is spelled NUM_OPERAND_NAMES. The
// second table is non-null only for VOPD, whose two components each take one
// half of every field.
static std::pair<const llvm::AMDGPU::OpName *, const llvm::AMDGPU::OpName *>
vgprLoweringOperandTables(const llvm::MCInstrDesc &Desc) {
  using llvm::AMDGPU::OpName;
  static const OpName VOPOps[4] = {OpName::src0, OpName::src1, OpName::src2,
                                   OpName::vdst};
  static const OpName VDSOps[4] = {OpName::addr, OpName::data0, OpName::data1,
                                   OpName::vdst};
  static const OpName FLATOps[4] = {OpName::vaddr, OpName::vdata,
                                    OpName::NUM_OPERAND_NAMES, OpName::vdst};
  static const OpName BUFOps[4] = {OpName::vaddr, OpName::NUM_OPERAND_NAMES,
                                   OpName::NUM_OPERAND_NAMES, OpName::vdata};
  static const OpName VIMGOps[4] = {OpName::vaddr0, OpName::vaddr1,
                                    OpName::vaddr2, OpName::vdata};
  // A VOPD Y component shares the MSB of its X counterpart, so the two tables
  // are indexed by the same field.
  static const OpName VOPDOpsX[4] = {OpName::src0X, OpName::vsrc1X,
                                     OpName::vsrc2X, OpName::vdstX};
  static const OpName VOPDOpsY[4] = {OpName::src0Y, OpName::vsrc1Y,
                                     OpName::vsrc2Y, OpName::vdstY};
  // MADMK encodes a literal where the second source would be, shifting the
  // remaining source down a field.
  static const OpName VOP2MADMKOps[4] = {
      OpName::src0, OpName::NUM_OPERAND_NAMES, OpName::src1, OpName::vdst};
  static const OpName VOPDFMAMKOpsX[4] = {
      OpName::src0X, OpName::NUM_OPERAND_NAMES, OpName::vsrc1X, OpName::vdstX};
  static const OpName VOPDFMAMKOpsY[4] = {
      OpName::src0Y, OpName::NUM_OPERAND_NAMES, OpName::vsrc1Y, OpName::vdstY};

  namespace SIInstrFlags = llvm::SIInstrFlags;
  namespace AMDGPU = llvm::AMDGPU;
  if (SIInstrFlags::isVOP1(Desc) || SIInstrFlags::isVOP2(Desc) ||
      SIInstrFlags::isVOP3Like(Desc) || SIInstrFlags::isVOPC(Desc) ||
      SIInstrFlags::isDPP(Desc)) {
    switch (Desc.getOpcode()) {
    // The scale operands are not VGPR addresses, so no field applies.
    case AMDGPU::V_WMMA_LD_SCALE_PAIRED_B32:
    case AMDGPU::V_WMMA_LD_SCALE_PAIRED_B32_gfx1250:
    case AMDGPU::V_WMMA_LD_SCALE16_PAIRED_B64:
    case AMDGPU::V_WMMA_LD_SCALE16_PAIRED_B64_gfx1250:
      return {};
    case AMDGPU::V_FMAMK_F16:
    case AMDGPU::V_FMAMK_F16_t16:
    case AMDGPU::V_FMAMK_F16_t16_gfx12:
    case AMDGPU::V_FMAMK_F16_fake16:
    case AMDGPU::V_FMAMK_F16_fake16_gfx12:
    case AMDGPU::V_FMAMK_F32:
    case AMDGPU::V_FMAMK_F32_gfx12:
    case AMDGPU::V_FMAMK_F64:
    case AMDGPU::V_FMAMK_F64_gfx1250:
      return {VOP2MADMKOps, nullptr};
    default:
      break;
    }
    return {VOPOps, nullptr};
  }

  if (SIInstrFlags::isDS(Desc))
    return {VDSOps, nullptr};
  if (SIInstrFlags::isFLAT(Desc))
    return {FLATOps, nullptr};
  if (SIInstrFlags::isBuffer(Desc))
    return {BUFOps, nullptr};
  if (SIInstrFlags::isVIMAGE(Desc))
    return {VIMGOps, nullptr};

  // A VOPD instruction is one carrying an X-component source operand.
  if (COMGR::hotswap::getNamedOperandIdx(Desc.getOpcode(), OpName::src0X) >=
      0) {
    const tables::VOPDInfo *Pair =
        tables::getVOPDOpcodeHelper(Desc.getOpcode());
    assert(Pair && "VOPD opcode is absent from the VOPD pair table");
    const tables::VOPDComponentInfo *OpX =
        tables::getVOPDBaseFromComponent(Pair->OpX);
    const tables::VOPDComponentInfo *OpY =
        tables::getVOPDBaseFromComponent(Pair->OpY);
    assert(OpX && OpY && "VOPD component is absent from the component table");
    return {OpX->BaseVOP == AMDGPU::V_FMAMK_F32 ? VOPDFMAMKOpsX : VOPDOpsX,
            OpY->BaseVOP == AMDGPU::V_FMAMK_F32 ? VOPDFMAMKOpsY : VOPDOpsY};
  }

  // Sampling and export never reach the transpiler, and MIMG is superseded by
  // VIMAGE on every ISA it supports.
  assert(!SIInstrFlags::isMIMG(Desc) && !SIInstrFlags::isVSAMPLE(Desc) &&
         !SIInstrFlags::isEXP(Desc) &&
         "VGPR MSB lowering is not modelled for this instruction format");
  return {};
}

VGPRMSBOperandIndices getVGPRMSBOperandIndices(const llvm::MCInstrDesc &Desc) {
  std::pair<const llvm::AMDGPU::OpName *, const llvm::AMDGPU::OpName *>
      OperandNames = vgprLoweringOperandTables(Desc);

  VGPRMSBOperandIndices Indices;
  for (unsigned Slot = 0; Slot != Indices.size(); ++Slot)
    Indices[Slot] = {operandIndexForSlot(Desc, OperandNames.first, Slot),
                     operandIndexForSlot(Desc, OperandNames.second, Slot)};
  return Indices;
}

} // namespace COMGR::hotswap
