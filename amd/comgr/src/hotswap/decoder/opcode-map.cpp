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

// One kCanonTable row: a canonical AMDGPU pseudo opcode and its CanonicalOp.
struct Entry {
  unsigned Opc;
  CanonicalOp Sem;
};

#define E(OP, SEM)                                                             \
  Entry { AMDGPU::OP, CanonicalOp::SEM }

static const Entry kCanonTable[] = {
    // clang-format off
    E(S_MOV_B32, S_MOV_B32),
    E(S_ADD_U32, S_ADD_U32),
    E(S_ADD_I32, S_ADD_I32),
    E(S_ADDC_U32, S_ADDC_U32),
    E(S_SUB_U32, S_SUB_U32),
    E(S_SUB_I32, S_SUB_I32),
    E(S_SUBB_U32, S_SUBB_U32),
    E(S_AND_B32, S_AND_B32),
    E(S_AND_B64, S_AND_B64),
    E(S_OR_B32, S_OR_B32),
    E(S_OR_B64, S_OR_B64),
    E(S_XOR_B32, S_XOR_B32),
    E(S_XOR_B64, S_XOR_B64),
    E(S_ANDN2_B32, S_ANDN2_B32),
    E(S_ANDN2_B64, S_ANDN2_B64),
    E(S_ORN2_B32, S_ORN2_B32),
    E(S_ORN2_B64, S_ORN2_B64),
    E(S_NAND_B32, S_NAND_B32),
    E(S_NAND_B64, S_NAND_B64),
    E(S_NOR_B32, S_NOR_B32),
    E(S_NOR_B64, S_NOR_B64),
    E(S_XNOR_B32, S_XNOR_B32),
    E(S_XNOR_B64, S_XNOR_B64),
    E(S_ABSDIFF_I32, S_ABSDIFF_I32),
    E(S_LSHL_B32, S_LSHL_B32),
    E(S_LSHL_B64, S_LSHL_B64),
    E(S_LSHR_B32, S_LSHR_B32),
    E(S_LSHR_B64, S_LSHR_B64),
    E(S_ASHR_I32, S_ASHR_I32),
    E(S_ASHR_I64, S_ASHR_I64),
    E(S_MUL_I32, S_MUL_I32),
    E(S_MUL_HI_U32, S_MUL_HI_U32),
    E(S_MUL_HI_I32, S_MUL_HI_I32),
    E(S_MUL_U64, S_MUL_U64),
    E(S_BFE_U32, S_BFE_U32),
    E(S_BFE_I32, S_BFE_I32),
    E(S_BFE_I64, S_BFE_I64),
    E(S_BFM_B32, S_BFM_B32),
    E(S_BFM_B64, S_BFM_B64),
    E(S_CSELECT_B32, S_CSELECT_B32),
    E(S_CSELECT_B64, S_CSELECT_B64),
    E(S_MIN_I32, S_MIN_I32),
    E(S_MIN_U32, S_MIN_U32),
    E(S_MAX_I32, S_MAX_I32),
    E(S_MAX_U32, S_MAX_U32),
    E(S_PACK_LL_B32_B16, S_PACK_LL_B32_B16),
    E(S_PACK_LH_B32_B16, S_PACK_LH_B32_B16),
    E(S_LSHL1_ADD_U32, S_LSHL1_ADD_U32),
    E(S_LSHL2_ADD_U32, S_LSHL2_ADD_U32),
    E(S_LSHL3_ADD_U32, S_LSHL3_ADD_U32),
    E(S_LSHL4_ADD_U32, S_LSHL4_ADD_U32),
    // gfx12 renamed the assembly mnemonics but retained these pseudos.
    E(S_ADD_U64, S_ADD_NC_U64),
    E(S_SUB_U64, S_SUB_NC_U64),
    E(S_ENDPGM, S_ENDPGM),
    E(S_WAITCNT, S_WAITCNT),
    E(S_WAIT_LOADCNT, S_WAIT_LOADCNT),
    E(S_WAIT_STORECNT, S_WAIT_STORECNT),
    E(S_WAIT_DSCNT, S_WAIT_DSCNT),
    E(S_WAIT_KMCNT, S_WAIT_KMCNT),
    E(S_WAIT_EXPCNT, S_WAIT_EXPCNT),
    E(S_WAIT_LOADCNT_DSCNT, S_WAIT_LOADCNT_DSCNT),
    E(S_WAIT_STORECNT_DSCNT, S_WAIT_STORECNT_DSCNT),
    E(S_WAIT_IDLE, S_WAIT_IDLE),
    E(S_WAIT_ASYNCCNT, S_WAIT_ASYNCCNT),
    E(S_WAIT_TENSORCNT, S_WAIT_TENSORCNT),
    E(S_WAIT_XCNT, S_WAIT_XCNT),
    // gfx12 renamed the mnemonic to `s_wait_alu`, but the pseudo LLVM keys on
    // still carries the original `S_WAITCNT_DEPCTR` spelling.
    E(S_WAITCNT_DEPCTR, S_WAIT_ALU),
    E(S_NOP, S_NOP),
    E(S_CLAUSE, S_CLAUSE),
    E(S_DELAY_ALU, S_DELAY_ALU),
    E(S_SLEEP, S_SLEEP),
    E(S_SETPRIO, S_SETPRIO),
    E(S_MONITOR_SLEEP, S_MONITOR_SLEEP),
    E(S_WAKEUP, S_WAKEUP),
    E(S_SETPRIO_INC_WG, S_SETPRIO_INC_WG),
    E(S_CODE_END, S_CODE_END),
    E(S_INCPERFLEVEL, S_INCPERFLEVEL),
    E(S_DECPERFLEVEL, S_DECPERFLEVEL),
    E(S_TTRACEDATA, S_TTRACEDATA),
    E(S_TTRACEDATA_IMM, S_TTRACEDATA_IMM),
    E(S_ICACHE_INV, S_ICACHE_INV),
    E(S_LOAD_DWORD_IMM, S_LOAD_B32),
    E(S_LOAD_DWORD_SGPR, S_LOAD_B32),
    E(S_LOAD_DWORD_SGPR_IMM, S_LOAD_B32),
    E(S_LOAD_DWORDX2_IMM, S_LOAD_B64),
    E(S_LOAD_DWORDX2_SGPR, S_LOAD_B64),
    E(S_LOAD_DWORDX2_SGPR_IMM, S_LOAD_B64),
    E(S_LOAD_DWORDX4_IMM, S_LOAD_B128),
    E(S_LOAD_DWORDX4_SGPR, S_LOAD_B128),
    E(S_LOAD_DWORDX4_SGPR_IMM, S_LOAD_B128),
    E(V_ADD_F32_e64, V_ADD_F32),
    E(V_MUL_F32_e64, V_MUL_F32),
    E(V_SUB_F32_e64, V_SUB_F32),
    E(V_SUBREV_F32_e64, V_SUBREV_F32),
    // gfx10 renamed the assembly mnemonics but retained these pseudos.
    E(V_ADD_U32_e64, V_ADD_NC_U32),
    E(V_SUB_U32_e64, V_SUB_NC_U32),
    E(V_SUBREV_U32_e64, V_SUBREV_NC_U32),
    E(V_ADD_CO_U32_e64, V_ADD_CO_U32),
    E(V_SUB_CO_U32_e64, V_SUB_CO_U32),
    E(V_SUBREV_CO_U32_e64, V_SUBREV_CO_U32),
    E(V_ADDC_U32_e64, V_ADD_CO_CI_U32),
    E(V_SUBB_U32_e64, V_SUB_CO_CI_U32),
    E(V_SUBBREV_U32_e64, V_SUBREV_CO_CI_U32),
    E(V_CNDMASK_B32_e64, V_CNDMASK_B32),
    E(V_MUL_I32_I24_e64, V_MUL_I32_I24),
    E(V_MUL_HI_I32_I24_e64, V_MUL_HI_I32_I24),
    E(V_MUL_U32_U24_e64, V_MUL_U32_U24),
    E(V_MUL_HI_U32_U24_e64, V_MUL_HI_U32_U24),
    E(V_MIN_I32_e64, V_MIN_I32),
    E(V_MAX_I32_e64, V_MAX_I32),
    E(V_MIN_U32_e64, V_MIN_U32),
    E(V_MAX_U32_e64, V_MAX_U32),
    E(V_AND_B32_e64, V_AND_B32),
    E(V_OR_B32_e64, V_OR_B32),
    E(V_XOR_B32_e64, V_XOR_B32),
    E(V_XNOR_B32_e64, V_XNOR_B32),
    E(V_LSHLREV_B32_e64, V_LSHLREV_B32),
    E(V_LSHRREV_B32_e64, V_LSHRREV_B32),
    E(V_ASHRREV_I32_e64, V_ASHRREV_I32),
    E(V_ADD_U64_e64, V_ADD_NC_U64),
    E(V_SUB_U64_e64, V_SUB_NC_U64),
    E(V_MUL_U64_e64, V_MUL_U64),
    E(V_LSHLREV_B64_pseudo_e64, V_LSHLREV_B64),
    E(V_ADD_U16_e64, V_ADD_U16),
    E(V_SUB_U16_e64, V_SUB_U16),
    E(V_SUBREV_U16_e64, V_SUBREV_U16),
    E(V_MUL_LO_U16_e64, V_MUL_LO_U16),
    E(V_LSHLREV_B16_e64, V_LSHLREV_B16),
    E(V_LSHRREV_B16_e64, V_LSHRREV_B16),
    E(V_ASHRREV_I16_e64, V_ASHRREV_I16),
    E(V_MIN_I16_e64, V_MIN_I16),
    E(V_MAX_I16_e64, V_MAX_I16),
    E(V_MIN_U16_e64, V_MIN_U16),
    E(V_MAX_U16_e64, V_MAX_U16),
    E(V_DOT2C_I32_I16_e64, V_DOT2C_I32_I16),
    E(V_DOT4C_I32_I8_e64, V_DOT4C_I32_I8),
    E(V_DOT8C_I32_I4_e64, V_DOT8C_I32_I4),
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

CanonicalOp OpcodeMap::lookup(unsigned Opcode) const {
  DenseMap<unsigned, CanonicalOp>::const_iterator It = Map.find(Opcode);
  return It != Map.end() ? It->second : CanonicalOp::Unknown;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // A duplicate opcode would silently keep only the first row and route the
  // rest through the wrong CanonicalOp, so it is a table-authoring bug.
  DenseMap<unsigned, CanonicalOp> CanonToSem;
  CanonToSem.reserve(std::size(kCanonTable));
  for (const Entry &E : kCanonTable) {
    bool Inserted = CanonToSem.try_emplace(E.Opc, E.Sem).second;
    assert(Inserted && "kCanonTable maps one MC opcode to two CanonicalOps");
    (void)Inserted;
  }

  const unsigned NumOpc = MCII.getNumOpcodes();
  const DenseMap<unsigned, unsigned> McToPseudo = buildMcToPseudoMap(NumOpc);
  const DenseMap<unsigned, unsigned> DppToBase = buildDppToBaseMap(NumOpc);

  Map.clear();
  for (unsigned Mc = 0; Mc < NumOpc; ++Mc) {
    const unsigned Canon = canonicalize(Mc, MCII, McToPseudo, DppToBase);
    DenseMap<unsigned, CanonicalOp>::const_iterator It = CanonToSem.find(Canon);
    if (It != CanonToSem.end())
      Map[Mc] = It->second;
  }
}

} // namespace COMGR::hotswap
