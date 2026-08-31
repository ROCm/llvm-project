//===- opcode-map.cpp - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "opcode-map.h"

#include "amdgpu-mc-tables.h"

#include <cassert>
#include <optional>

// AMDGPU target-private headers.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Opcode named by an AMDGPU InstrMapping helper result, or nullopt for either
// way the helpers signal "no mapping": -1 and INSTRUCTION_LIST_END.
std::optional<unsigned> mappedOpcode(int Result) {
  if (Result <= 0 || Result >= AMDGPU::INSTRUCTION_LIST_END)
    return std::nullopt;
  return Result;
}

// One kCanonTable row: a canonical AMDGPU pseudo opcode and its instruction.
struct Entry {
  unsigned Opc;
  CanonicalInst Inst;
};

#define E(OP, SEM, TYPE, ELEMENT_TYPE)                                         \
  Entry {                                                                      \
    AMDGPU::OP, {                                                              \
      CanonicalOp::SEM, CanonicalType::TYPE, CanonicalType::ELEMENT_TYPE       \
    }                                                                          \
  }

static const Entry kCanonTable[] = {
    // clang-format off
    E(S_MOV_B32, S_MOV, B32, None),
    E(S_ADD_U32, S_ADD, U32, None),
    E(S_ADD_I32, S_ADD, I32, None),
    E(S_ADDC_U32, S_ADDC, U32, None),
    E(S_SUB_U32, S_SUB, U32, None),
    E(S_SUB_I32, S_SUB, I32, None),
    E(S_SUBB_U32, S_SUBB, U32, None),
    E(S_AND_B32, S_AND, B32, None),
    E(S_AND_B64, S_AND, B64, None),
    E(S_OR_B32, S_OR, B32, None),
    E(S_OR_B64, S_OR, B64, None),
    E(S_XOR_B32, S_XOR, B32, None),
    E(S_XOR_B64, S_XOR, B64, None),
    E(S_ANDN2_B32, S_ANDN2, B32, None),
    E(S_ANDN2_B64, S_ANDN2, B64, None),
    E(S_ORN2_B32, S_ORN2, B32, None),
    E(S_ORN2_B64, S_ORN2, B64, None),
    E(S_NAND_B32, S_NAND, B32, None),
    E(S_NAND_B64, S_NAND, B64, None),
    E(S_NOR_B32, S_NOR, B32, None),
    E(S_NOR_B64, S_NOR, B64, None),
    E(S_XNOR_B32, S_XNOR, B32, None),
    E(S_XNOR_B64, S_XNOR, B64, None),
    E(S_ABSDIFF_I32, S_ABSDIFF, I32, None),
    E(S_LSHL_B32, S_LSHL, B32, None),
    E(S_LSHL_B64, S_LSHL, B64, None),
    E(S_LSHR_B32, S_LSHR, B32, None),
    E(S_LSHR_B64, S_LSHR, B64, None),
    E(S_ASHR_I32, S_ASHR, I32, None),
    E(S_ASHR_I64, S_ASHR, I64, None),
    E(S_MUL_I32, S_MUL, I32, None),
    E(S_MUL_HI_U32, S_MUL_HI, U32, None),
    E(S_MUL_HI_I32, S_MUL_HI, I32, None),
    E(S_MUL_U64, S_MUL, U64, None),
    E(S_BFE_U32, S_BFE, U32, None),
    E(S_BFE_I32, S_BFE, I32, None),
    E(S_BFE_I64, S_BFE, I64, None),
    E(S_BFM_B32, S_BFM, B32, None),
    E(S_BFM_B64, S_BFM, B64, None),
    E(S_CSELECT_B32, S_CSELECT, B32, None),
    E(S_CSELECT_B64, S_CSELECT, B64, None),
    E(S_MIN_I32, S_MIN, I32, None),
    E(S_MIN_U32, S_MIN, U32, None),
    E(S_MAX_I32, S_MAX, I32, None),
    E(S_MAX_U32, S_MAX, U32, None),
    E(S_PACK_LL_B32_B16, S_PACK_LL, B32, B16),
    E(S_PACK_LH_B32_B16, S_PACK_LH, B32, B16),
    E(S_LSHL1_ADD_U32, S_LSHL1_ADD, U32, None),
    E(S_LSHL2_ADD_U32, S_LSHL2_ADD, U32, None),
    E(S_LSHL3_ADD_U32, S_LSHL3_ADD, U32, None),
    E(S_LSHL4_ADD_U32, S_LSHL4_ADD, U32, None),
    // gfx12 renamed the assembly mnemonics but retained these pseudos.
    E(S_ADD_U64, S_ADD_NC, U64, None),
    E(S_SUB_U64, S_SUB_NC, U64, None),
    E(S_CMP_EQ_U32, S_CMP_EQ, U32, None),
    E(S_CMP_LG_U32, S_CMP_LG, U32, None),
    E(S_CMP_GT_U32, S_CMP_GT, U32, None),
    E(S_CMP_GE_U32, S_CMP_GE, U32, None),
    E(S_CMP_LT_U32, S_CMP_LT, U32, None),
    E(S_CMP_LE_U32, S_CMP_LE, U32, None),
    E(S_CMP_EQ_I32, S_CMP_EQ, I32, None),
    E(S_CMP_LG_I32, S_CMP_LG, I32, None),
    E(S_CMP_GT_I32, S_CMP_GT, I32, None),
    E(S_CMP_GE_I32, S_CMP_GE, I32, None),
    E(S_CMP_LT_I32, S_CMP_LT, I32, None),
    E(S_CMP_LE_I32, S_CMP_LE, I32, None),
    E(S_CMP_EQ_U64, S_CMP_EQ, U64, None),
    E(S_CMP_LG_U64, S_CMP_LG, U64, None),
    E(S_CMP_EQ_F32, S_CMP_EQ, F32, None),
    E(S_CMP_LG_F32, S_CMP_LG, F32, None),
    E(S_CMP_GT_F32, S_CMP_GT, F32, None),
    E(S_CMP_GE_F32, S_CMP_GE, F32, None),
    E(S_CMP_LT_F32, S_CMP_LT, F32, None),
    E(S_CMP_LE_F32, S_CMP_LE, F32, None),
    E(S_CMP_NEQ_F32, S_CMP_NEQ, F32, None),
    E(S_CMP_NGT_F32, S_CMP_NGT, F32, None),
    E(S_CMP_NGE_F32, S_CMP_NGE, F32, None),
    E(S_CMP_NLT_F32, S_CMP_NLT, F32, None),
    E(S_CMP_NLE_F32, S_CMP_NLE, F32, None),
    E(S_CMP_NLG_F32, S_CMP_NLG, F32, None),
    E(S_CMP_O_F32, S_CMP_O, F32, None),
    E(S_CMP_U_F32, S_CMP_U, F32, None),
    E(S_CMP_EQ_F16, S_CMP_EQ, F16, None),
    E(S_CMP_LG_F16, S_CMP_LG, F16, None),
    E(S_CMP_GT_F16, S_CMP_GT, F16, None),
    E(S_CMP_GE_F16, S_CMP_GE, F16, None),
    E(S_CMP_LT_F16, S_CMP_LT, F16, None),
    E(S_CMP_LE_F16, S_CMP_LE, F16, None),
    E(S_CMP_NEQ_F16, S_CMP_NEQ, F16, None),
    E(S_CMP_NGT_F16, S_CMP_NGT, F16, None),
    E(S_CMP_NGE_F16, S_CMP_NGE, F16, None),
    E(S_CMP_NLT_F16, S_CMP_NLT, F16, None),
    E(S_CMP_NLE_F16, S_CMP_NLE, F16, None),
    E(S_CMP_NLG_F16, S_CMP_NLG, F16, None),
    E(S_CMP_O_F16, S_CMP_O, F16, None),
    E(S_CMP_U_F16, S_CMP_U, F16, None),
    E(S_BITCMP0_B32, S_BITCMP0, B32, None),
    E(S_BITCMP1_B32, S_BITCMP1, B32, None),
    E(S_BITCMP0_B64, S_BITCMP0, B64, None),
    E(S_BITCMP1_B64, S_BITCMP1, B64, None),
    E(S_ENDPGM, S_ENDPGM, None, None),
    E(S_WAITCNT, S_WAITCNT, None, None),
    E(S_WAIT_LOADCNT, S_WAIT_LOADCNT, None, None),
    E(S_WAIT_STORECNT, S_WAIT_STORECNT, None, None),
    E(S_WAIT_DSCNT, S_WAIT_DSCNT, None, None),
    E(S_WAIT_KMCNT, S_WAIT_KMCNT, None, None),
    E(S_WAIT_EXPCNT, S_WAIT_EXPCNT, None, None),
    E(S_WAIT_LOADCNT_DSCNT, S_WAIT_LOADCNT_DSCNT, None, None),
    E(S_WAIT_STORECNT_DSCNT, S_WAIT_STORECNT_DSCNT, None, None),
    E(S_WAIT_IDLE, S_WAIT_IDLE, None, None),
    E(S_WAIT_ASYNCCNT, S_WAIT_ASYNCCNT, None, None),
    E(S_WAIT_TENSORCNT, S_WAIT_TENSORCNT, None, None),
    E(S_WAIT_XCNT, S_WAIT_XCNT, None, None),
    // gfx12 renamed the mnemonic to `s_wait_alu`, but the pseudo LLVM keys on
    // still carries the original `S_WAITCNT_DEPCTR` spelling.
    E(S_WAITCNT_DEPCTR, S_WAIT_ALU, None, None),
    E(S_NOP, S_NOP, None, None),
    E(S_CLAUSE, S_CLAUSE, None, None),
    E(S_DELAY_ALU, S_DELAY_ALU, None, None),
    E(S_SLEEP, S_SLEEP, None, None),
    E(S_SETPRIO, S_SETPRIO, None, None),
    E(S_MONITOR_SLEEP, S_MONITOR_SLEEP, None, None),
    E(S_WAKEUP, S_WAKEUP, None, None),
    E(S_SETPRIO_INC_WG, S_SETPRIO_INC_WG, None, None),
    E(S_CODE_END, S_CODE_END, None, None),
    E(S_INCPERFLEVEL, S_INCPERFLEVEL, None, None),
    E(S_DECPERFLEVEL, S_DECPERFLEVEL, None, None),
    E(S_TTRACEDATA, S_TTRACEDATA, None, None),
    E(S_TTRACEDATA_IMM, S_TTRACEDATA_IMM, None, None),
    E(S_ICACHE_INV, S_ICACHE_INV, None, None),
    E(S_LOAD_DWORD_IMM, S_LOAD, B32, None),
    E(S_LOAD_DWORD_SGPR, S_LOAD, B32, None),
    E(S_LOAD_DWORD_SGPR_IMM, S_LOAD, B32, None),
    E(S_LOAD_DWORDX2_IMM, S_LOAD, B64, None),
    E(S_LOAD_DWORDX2_SGPR, S_LOAD, B64, None),
    E(S_LOAD_DWORDX2_SGPR_IMM, S_LOAD, B64, None),
    E(S_LOAD_DWORDX4_IMM, S_LOAD, B128, None),
    E(S_LOAD_DWORDX4_SGPR, S_LOAD, B128, None),
    E(S_LOAD_DWORDX4_SGPR_IMM, S_LOAD, B128, None),
    E(V_ADD_F32_e64, V_ADD, F32, None),
    E(V_MUL_F32_e64, V_MUL, F32, None),
    E(V_SUB_F32_e64, V_SUB, F32, None),
    E(V_SUBREV_F32_e64, V_SUBREV, F32, None),
    // gfx10 renamed the assembly mnemonics but retained these pseudos.
    E(V_ADD_U32_e64, V_ADD_NC, U32, None),
    E(V_SUB_U32_e64, V_SUB_NC, U32, None),
    E(V_SUBREV_U32_e64, V_SUBREV_NC, U32, None),
    E(V_ADD_CO_U32_e64, V_ADD_CO, U32, None),
    E(V_SUB_CO_U32_e64, V_SUB_CO, U32, None),
    E(V_SUBREV_CO_U32_e64, V_SUBREV_CO, U32, None),
    E(V_ADDC_U32_e64, V_ADD_CO_CI, U32, None),
    E(V_SUBB_U32_e64, V_SUB_CO_CI, U32, None),
    E(V_SUBBREV_U32_e64, V_SUBREV_CO_CI, U32, None),
    E(V_CNDMASK_B32_e64, V_CNDMASK, B32, None),
    E(V_MUL_I32_I24_e64, V_MUL, I32, I24),
    E(V_MUL_HI_I32_I24_e64, V_MUL_HI, I32, I24),
    E(V_MUL_U32_U24_e64, V_MUL, U32, U24),
    E(V_MUL_HI_U32_U24_e64, V_MUL_HI, U32, U24),
    E(V_MIN_I32_e64, V_MIN, I32, None),
    E(V_MAX_I32_e64, V_MAX, I32, None),
    E(V_MIN_U32_e64, V_MIN, U32, None),
    E(V_MAX_U32_e64, V_MAX, U32, None),
    E(V_AND_B32_e64, V_AND, B32, None),
    E(V_OR_B32_e64, V_OR, B32, None),
    E(V_XOR_B32_e64, V_XOR, B32, None),
    E(V_XNOR_B32_e64, V_XNOR, B32, None),
    E(V_LSHLREV_B32_e64, V_LSHLREV, B32, None),
    E(V_LSHRREV_B32_e64, V_LSHRREV, B32, None),
    E(V_ASHRREV_I32_e64, V_ASHRREV, I32, None),
    E(V_ADD_U64_e64, V_ADD_NC, U64, None),
    E(V_SUB_U64_e64, V_SUB_NC, U64, None),
    E(V_MUL_U64_e64, V_MUL, U64, None),
    E(V_LSHLREV_B64_pseudo_e64, V_LSHLREV, B64, None),
    E(V_ADD_U16_e64, V_ADD, U16, None),
    E(V_SUB_U16_e64, V_SUB, U16, None),
    E(V_SUBREV_U16_e64, V_SUBREV, U16, None),
    E(V_MUL_LO_U16_e64, V_MUL_LO, U16, None),
    E(V_LSHLREV_B16_e64, V_LSHLREV, B16, None),
    E(V_LSHRREV_B16_e64, V_LSHRREV, B16, None),
    E(V_ASHRREV_I16_e64, V_ASHRREV, I16, None),
    E(V_MIN_I16_e64, V_MIN, I16, None),
    E(V_MAX_I16_e64, V_MAX, I16, None),
    E(V_MIN_U16_e64, V_MIN, U16, None),
    E(V_MAX_U16_e64, V_MAX, U16, None),
    E(V_DOT2C_I32_I16_e64, V_DOT2C, I32, I16),
    E(V_DOT4C_I32_I8_e64, V_DOT4C, I32, I8),
    E(V_DOT8C_I32_I4_e64, V_DOT8C, I32, I4),
    // clang-format on
};

#undef E

// Update this bound when SIEncodingFamily gains a new value, otherwise opcodes
// using that encoding remain unmapped.
constexpr unsigned KNumEncodingFamilies =
    static_cast<unsigned>(SIEncodingFamily::GFX13) + 1;

// Reverse map MC opcode -> canonical pseudo, built by scanning the first
// `NumOpc` pseudos across every encoding family.
DenseMap<unsigned, unsigned> buildMcToPseudoMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    for (unsigned Gen = 0; Gen < KNumEncodingFamilies; ++Gen) {
      std::optional<unsigned> Mc = mappedOpcode(hotswap::getMCOpcode(P, Gen));
      if (Mc && *Mc != P)
        Result.try_emplace(*Mc, P);
    }
  }
  return Result;
}

// Reverse map DPP opcode -> base VOP opcode, built by scanning the first
// `NumOpc` opcodes because only the forward mappings are exposed.
DenseMap<unsigned, unsigned> buildDppToBaseMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    if (std::optional<unsigned> D32 = mappedOpcode(hotswap::getDPPOp32(P)))
      Result.try_emplace(*D32, P);
    if (std::optional<unsigned> D64 = mappedOpcode(hotswap::getDPPOp64(P)))
      Result.try_emplace(*D64, P);
  }
  return Result;
}

// Map `Mc` to the canonical pseudo used by kCanonTable.
unsigned canonicalize(unsigned Mc, const MCInstrInfo &MCII,
                      const DenseMap<unsigned, unsigned> &McToPseudo,
                      const DenseMap<unsigned, unsigned> &DppToBase) {
  unsigned P = Mc;

  DenseMap<unsigned, unsigned>::const_iterator PseudoIt = McToPseudo.find(P);
  if (PseudoIt != McToPseudo.end())
    P = PseudoIt->second;

  DenseMap<unsigned, unsigned>::const_iterator DppIt = DppToBase.find(P);
  if (DppIt != DppToBase.end())
    P = DppIt->second;

  if (std::optional<unsigned> Base =
          mappedOpcode(hotswap::getBasicFromSDWAOp(P)))
    P = *Base;

  if (std::optional<unsigned> E64 = mappedOpcode(hotswap::getVOPe64(P)))
    P = *E64;

  // Testing the format flag first avoids a table lookup for every non-FLAT
  // opcode.
  if (P < MCII.getNumOpcodes() && SIInstrFlags::isFLAT(MCII, P)) {
    if (std::optional<unsigned> Vaddr =
            mappedOpcode(hotswap::getGlobalVaddrOp(P)))
      P = *Vaddr;
  }

  return P;
}

} // namespace

CanonicalInst OpcodeMap::lookup(unsigned Opcode) const {
  DenseMap<unsigned, CanonicalInst>::const_iterator It = Map.find(Opcode);
  return It != Map.end() ? It->second : CanonicalInst{};
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // A duplicate opcode would silently keep only the first row and route the
  // rest through the wrong canonical instruction, so it is a table-authoring
  // bug.
  DenseMap<unsigned, CanonicalInst> CanonToInst;
  CanonToInst.reserve(std::size(kCanonTable));
  for (const Entry &E : kCanonTable) {
    bool Inserted = CanonToInst.try_emplace(E.Opc, E.Inst).second;
    assert(Inserted && "kCanonTable maps one MC opcode twice");
    (void)Inserted;
  }

  const unsigned NumOpc = MCII.getNumOpcodes();
  const DenseMap<unsigned, unsigned> McToPseudo = buildMcToPseudoMap(NumOpc);
  const DenseMap<unsigned, unsigned> DppToBase = buildDppToBaseMap(NumOpc);

  Map.clear();
  for (unsigned Mc = 0; Mc < NumOpc; ++Mc) {
    const unsigned Canon = canonicalize(Mc, MCII, McToPseudo, DppToBase);
    DenseMap<unsigned, CanonicalInst>::const_iterator It =
        CanonToInst.find(Canon);
    if (It != CanonToInst.end())
      Map[Mc] = It->second;
  }
}

} // namespace COMGR::hotswap
