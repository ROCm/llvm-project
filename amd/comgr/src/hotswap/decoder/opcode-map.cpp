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
    E(S_MOV_B32, S_MOV_B32),
    E(S_ENDPGM, S_ENDPGM),
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
