//===- opcode-map.cpp - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "opcode-map.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

// AMDGPU target-private headers.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
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

// Iteration bound for SIEncodingFamily, a closed numeric set whose current
// maximum is GFX13. The static_assert fails if that sentinel is renamed.
static_assert(SIEncodingFamily::GFX13 >= SIEncodingFamily::SI,
              "SIEncodingFamily enum layout changed unexpectedly");
constexpr unsigned KNumEncodingFamilies =
    static_cast<unsigned>(SIEncodingFamily::GFX13) + 1;

// Reverse map MC opcode -> canonical pseudo, built by scanning the first
// `NumOpc` pseudos across every encoding family.
DenseMap<unsigned, unsigned> buildMcToPseudoMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    for (unsigned Gen = 0; Gen < KNumEncodingFamilies; ++Gen) {
      if (auto Mc = mappedOpcode(AMDGPU::getMCOpcode(P, Gen)); Mc && *Mc != P)
        Result.try_emplace(*Mc, P);
    }
  }
  return Result;
}

// Invariant the source pseudo and the target it collapses onto must satisfy
// for the collapse to be semantically safe.
using RulePredicate = bool (*)(const MCInstrDesc &Src, const MCInstrDesc &Tgt);

// Source is an atomic that returns the old value and the target is the same
// atomic without a return, so their def counts differ accordingly.
static bool atomicRetToNoRet(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  constexpr uint64_t KRet = SIInstrFlags::IsAtomicRet;
  constexpr uint64_t KNoRet = SIInstrFlags::IsAtomicNoRet;
  return (Src.TSFlags & KRet) && (Tgt.TSFlags & KNoRet) &&
         Src.getNumDefs() > 0 && Tgt.getNumDefs() == 0;
}

// Both source and target must be MFMA pseudos.
static bool bothAreMAI(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  constexpr uint64_t KMAI = SIInstrFlags::IsMAI;
  return (Src.TSFlags & KMAI) && (Tgt.TSFlags & KMAI);
}

// TSFlags bits that must match between source and target for a collapse to be
// safe. Encoding-only bits such as VOP3_OPSEL are excluded because
// operand-size and subtarget variants legitimately toggle them.
static constexpr uint64_t KSemanticShapeMask =
    // Instruction families.
    SIInstrFlags::SOP1 | SIInstrFlags::SOP2 | SIInstrFlags::SOPC |
    SIInstrFlags::SOPK | SIInstrFlags::SOPP | SIInstrFlags::VOP1 |
    SIInstrFlags::VOP2 | SIInstrFlags::VOPC | SIInstrFlags::VOP3 |
    SIInstrFlags::VOP3P | SIInstrFlags::SDWA | SIInstrFlags::DPP |
    SIInstrFlags::MUBUF | SIInstrFlags::MTBUF | SIInstrFlags::SMRD |
    SIInstrFlags::FLAT | SIInstrFlags::DS | SIInstrFlags::MIMG |
    // Semantic classification (atomic kind, MAI).
    SIInstrFlags::IsAtomicRet | SIInstrFlags::IsAtomicNoRet |
    SIInstrFlags::IsMAI;

// Source and target agree on every KSemanticShapeMask bit and on def arity.
static bool sameSemanticShape(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Src.getNumDefs() == Tgt.getNumDefs();
}

// Same shape as sameSemanticShape, except the target may carry one more def
// than the source: the scalar dst the source drops.
static bool nosdstDropsScalarDef(const MCInstrDesc &Src,
                                 const MCInstrDesc &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Tgt.getNumDefs() >= Src.getNumDefs() &&
         Tgt.getNumDefs() - Src.getNumDefs() <= 1;
}

// Alias map collapsing the several pseudos LLVM generates for one instruction
// onto a single canonical pseudo, e.g.
//   DS_WRITE_B16_gfx9        -> DS_WRITE_B16
//   V_ADD_F16_t16_e64        -> V_ADD_F16_e64
//   V_ADD_F16_fake16_e64     -> V_ADD_F16_e64
// No helper exposes this collapse, so it is matched on pseudo name.
DenseMap<unsigned, unsigned> buildPseudoAliasMap(const MCInstrInfo &MCII) {
  unsigned NumOpc = MCII.getNumOpcodes();

  llvm::StringMap<unsigned> ByName;
  for (unsigned P = 0; P < NumOpc; ++P)
    ByName.try_emplace(MCII.getName(P), P);

  // A pseudo-name marker to strip and the invariant the alias must preserve.
  struct Rule {
    llvm::StringRef Needle;
    bool IsSuffix;
    RulePredicate Pred;
  };
  static const Rule Rules[] = {
      // Per-subtarget pseudos sharing the base's TableGen class.
      {"_vi_gfx9", true, sameSemanticShape},
      {"_gfx9", true, sameSemanticShape},
      {"_gfx1250", true, sameSemanticShape},
      {"_gfx1250_", false, sameSemanticShape},
      {"_pseudo_", false, sameSemanticShape},
      // 16-bit operand encoding variant; toggles VOP3_OPSEL but keeps dispatch
      // identity.
      {"_t16_", false, sameSemanticShape},
      {"_fake16_", false, sameSemanticShape},
      // gfx11+ encoding variant of the fp8/bf8 cvt ops: adds a byte_sel operand
      // but keeps dispatch identity.
      {"_OP_SEL_", false, sameSemanticShape},
      // gfx11+ VOPC CMPX drops the scalar dst and writes EXEC only.
      {"_nosdst_", false, nosdstDropsScalarDef},
      // MFMA VGPR-dest and accumulate variants.
      {"_vgprcd_", false, bothAreMAI},
      {"_mac_", false, bothAreMAI},
      // AGPR-dest and atomic return-value variants.
      {"_agpr", true, sameSemanticShape},
      {"_RTN", true, atomicRetToNoRet},
  };

  // Returns the index of the firing rule, or nullopt if no rule applies.
  auto StripOnce = [&](llvm::StringRef Name,
                       std::string &Out) -> std::optional<size_t> {
    for (size_t I = 0; I < std::size(Rules); ++I) {
      const Rule &R = Rules[I];
      if (R.IsSuffix) {
        if (!Name.ends_with(R.Needle))
          continue;
        Out = Name.drop_back(R.Needle.size()).str();
        return I;
      }
      size_t Pos = Name.find(R.Needle);
      if (Pos == llvm::StringRef::npos)
        continue;
      Out = (Name.substr(0, Pos).str() + std::string("_") +
             Name.substr(Pos + R.Needle.size()).str());
      return I;
    }
    return std::nullopt;
  };

  DenseMap<unsigned, unsigned> Alias;
  for (const auto &Kv : ByName) {
    StringRef Name = Kv.first();
    // Most pseudos carry no marker, so skip them before any string allocation.
    if (llvm::none_of(Rules,
                      [&](const Rule &R) { return Name.contains(R.Needle); }))
      continue;

    // A pseudo may carry several markers, so strip them one at a time until
    // none remain, validating each step against the opcode it lands on.
    std::string Cur = Name.str();
    unsigned CurOpc = Kv.second;
    unsigned FinalOpc = Kv.second;
    while (true) {
      std::string Next;
      std::optional<size_t> RuleIdx = StripOnce(Cur, Next);
      if (!RuleIdx)
        break;
      auto It = ByName.find(Next);
      if (It != ByName.end() && It->second != Kv.second) {
        bool Ok = Rules[*RuleIdx].Pred(MCII.get(CurOpc), MCII.get(It->second));
        assert(Ok && "alias rule broke its semantic invariant; LLVM likely "
                     "renamed or repurposed a pseudo");
        if (Ok) {
          CurOpc = It->second;
          FinalOpc = CurOpc;
        }
      }
      Cur = std::move(Next);
    }
    if (FinalOpc != Kv.second)
      Alias.try_emplace(Kv.second, FinalOpc);
  }
  return Alias;
}

// Reverse map DPP opcode -> base VOP opcode, built by scanning the first
// `NumOpc` opcodes because only the forward mappings are exposed.
DenseMap<unsigned, unsigned> buildDppToBaseMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    if (auto D32 = mappedOpcode(AMDGPU::getDPPOp32(P)))
      Result.try_emplace(*D32, P);
    if (auto D64 = mappedOpcode(AMDGPU::getDPPOp64(P)))
      Result.try_emplace(*D64, P);
  }
  return Result;
}

// Canonicalize any MC opcode `mc` to the pseudo form matched in kCanonTable.
// The chain is:
//   MC -> pseudo                (TableGen Subtarget map)
//   pseudo -> base VOP          (strip DPP / SDWA)
//   e32 -> e64                  (collapse VOP encoding variants)
//   SADDR -> VADDR              (FLAT/GLOBAL global-saddr table)
unsigned canonicalize(unsigned Mc, const MCInstrInfo &MCII,
                      const DenseMap<unsigned, unsigned> &McToPseudo,
                      const DenseMap<unsigned, unsigned> &PseudoAlias,
                      const DenseMap<unsigned, unsigned> &DppToBase) {
  unsigned P = Mc;

  // MC (subtarget-specific real) -> pseudo.
  if (auto It = McToPseudo.find(P); It != McToPseudo.end())
    P = It->second;

  // Strip subtarget/operand markers to the base pseudo (_gfx9, _t16_, ...).
  if (auto It = PseudoAlias.find(P); It != PseudoAlias.end())
    P = It->second;

  // DPP -> base, for both the _dpp and _e64_dpp pseudos.
  if (auto It = DppToBase.find(P); It != DppToBase.end())
    P = It->second;

  // SDWA -> base.
  if (auto Base = mappedOpcode(AMDGPU::getBasicFromSDWAOp(P)))
    P = *Base;

  // e32 -> e64.
  if (auto E64 = mappedOpcode(AMDGPU::getVOPe64(P)))
    P = *E64;

  // Re-run the marker strip: getVOPe64 can expose a marked _e64 pseudo (e.g.
  // V_LSHLREV_B64_pseudo_e64) that only reduces once the _e64 suffix is
  // present.
  if (auto It = PseudoAlias.find(P); It != PseudoAlias.end())
    P = It->second;

  // FLAT/GLOBAL SADDR -> VADDR. Testing the format flag first avoids a table
  // lookup for every non-FLAT opcode.
  if (P < MCII.getNumOpcodes() &&
      (MCII.get(P).TSFlags & SIInstrFlags::FLAT) != 0) {
    if (auto Vaddr = mappedOpcode(AMDGPU::getGlobalVaddrOp(P)))
      P = *Vaddr;
  }

  return P;
}

} // namespace

CanonicalOp OpcodeMap::lookup(unsigned Opcode) const {
  auto It = Map.find(Opcode);
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
  const auto McToPseudo = buildMcToPseudoMap(NumOpc);
  const auto PseudoAlias = buildPseudoAliasMap(MCII);
  const auto DppToBase = buildDppToBaseMap(NumOpc);

  Map.clear();
  // Rough guess at the mapped fraction; a one-shot init tolerates resizes.
  Map.reserve(NumOpc / 4);
  for (unsigned Mc = 0; Mc < NumOpc; ++Mc) {
    const unsigned Canon =
        canonicalize(Mc, MCII, McToPseudo, PseudoAlias, DppToBase);
    if (auto It = CanonToSem.find(Canon); It != CanonToSem.end())
      Map[Mc] = It->second;
  }
}

} // namespace COMGR::hotswap
