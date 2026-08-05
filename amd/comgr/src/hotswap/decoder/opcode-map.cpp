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

// The AMDGPU InstrMapping helpers (getMCOpcode, getVOPe64, getDPPOp32/64,
// getBasicFromSDWAOp, getGlobalVaddrOp) return int32_t. They signal "no
// mapping" two ways: -1 when the opcode is absent from the table, and
// INSTRUCTION_LIST_END when it is present but the requested column is empty.
// Normalise both to nullopt so callers get a real opcode or nothing.
std::optional<unsigned> mappedOpcode(int Result) {
  if (Result <= 0 || Result >= AMDGPU::INSTRUCTION_LIST_END)
    return std::nullopt;
  return Result;
}

// One kCanonTable row: a canonical AMDGPU pseudo opcode and the CanonicalOp the
// raiser dispatches on.
struct Entry {
  unsigned Opc;
  CanonicalOp Sem;
};

#define E(OP, SEM)                                                             \
  Entry { AMDGPU::OP, CanonicalOp::SEM }

static const Entry kCanonTable[] = {
    // One row per landed handler; every unlisted MC opcode stays Unknown and
    // is refused by the raiser's dispatch.
    E(S_MOV_B32, S_MOV_B32),
    E(S_ENDPGM, S_ENDPGM),
};

#undef E

// Iteration bound for SIEncodingFamily: the enum in SIDefines.h is a closed
// numeric set with GFX13 as the current maximum, so we scan [0, GFX13] when
// inverting the pseudo -> MC map.  If LLVM adds a new family the next
// enumerator value appears here automatically and the build still compiles;
// the static_assert keeps us honest if LLVM ever renames the sentinel we use.
static_assert(SIEncodingFamily::GFX13 >= SIEncodingFamily::SI,
              "SIEncodingFamily enum layout changed unexpectedly");
constexpr unsigned KNumEncodingFamilies =
    static_cast<unsigned>(SIEncodingFamily::GFX13) + 1;

// Build a reverse map MC-opcode -> canonical pseudo by scanning every pseudo
// opcode across all subtarget generations. This is ~O(N * 15) work at init
// time (N ~= 70k AMDGPU opcodes on recent LLVM), which is well under a
// millisecond on modern hardware and done once per raiser.
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

// Rule predicates: an optional semantic invariant the alias must preserve.
// Every time an alias step is committed (source pseudo S collapses onto
// target pseudo T), the firing rule's predicate is evaluated against
// MCInstrDesc(S) and MCInstrDesc(T). A violation means LLVM renamed or
// repurposed a pseudo in a way that breaks our naming contract, and is
// reported as a fatal error at init time rather than silently producing
// wrong IR at runtime.
using RulePredicate = bool (*)(const MCInstrDesc &Src, const MCInstrDesc &Tgt);

// Source is an atomic that returns the old value; target is the same atomic
// without a return. numDefs carries that distinction downstream, so it must
// differ accordingly.
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

// Flags that must match between source and target for an alias to be
// semantically safe: instruction family, atomic kind, and MAI. Encoding-only
// bits such as VOP3_OPSEL are deliberately excluded because operand-size and
// subtarget variants toggle them.
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

// Subtarget-/operand-class variants (`_gfx9`, `_t16_`, `_fake16_`, `_agpr`,
// etc.) may legitimately toggle encoding flags such as `VOP3_OPSEL` or
// `renamedInGFX9` between source and target, but they must preserve the
// instruction's dispatch identity: same family, same atomic kind, same MAI
// classification, same def arity. A violation means LLVM renamed or
// repurposed a pseudo in a way our alias map cannot safely collapse.
static bool sameSemanticShape(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Src.getNumDefs() == Tgt.getNumDefs();
}

// `_nosdst_` collapse: same dispatch identity as the base, and the source
// never has more defs than the target (the scalar dst is either dropped
// entirely or added back on the target's e64 form).
static bool nosdstDropsScalarDef(const MCInstrDesc &Src,
                                 const MCInstrDesc &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Tgt.getNumDefs() >= Src.getNumDefs() &&
         Tgt.getNumDefs() - Src.getNumDefs() <= 1;
}

// Build an alias map that collapses "parallel" pseudos LLVM generates for the
// same semantic instruction into a single canonical pseudo. Examples:
//   DS_WRITE_B16_gfx9        -> DS_WRITE_B16
//   V_ADD_F16_t16_e64        -> V_ADD_F16_e64
//   V_ADD_F16_fake16_e64     -> V_ADD_F16_e64
// LLVM does not expose a helper for this collapse, so we match on pseudo name
// at init time. Name lookups are confined to this one-shot scan over
// `MCII.getNumOpcodes()`; runtime lookups remain pure DenseMap hits.
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
      // Per-subtarget pseudos (`_gfx9`, `_gfx1250`, `_vi_gfx9`, ...) sharing
      // the
      // base's TableGen class.
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
      // gfx11+ VOPC CMPX drops the scalar dst (writes EXEC only); the handler
      // ignores it.
      {"_nosdst_", false, nosdstDropsScalarDef},
      // MFMA VGPR-dest (`_vgprcd_`) and accumulate (`_mac_`) variants.
      {"_vgprcd_", false, bothAreMAI},
      {"_mac_", false, bothAreMAI},
      // Atomic return-value (`_RTN`) and AGPR-dest (`_agpr`) variants; whether
      // the old value is written is derived from numDefs downstream.
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

// Build a reverse DPP map: DPP opcode -> base VOP opcode. LLVM only provides
// forward mappings (base -> DPP32 / DPP64), so we invert by scanning.
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

  // DPP -> base. This handles both VOP2-like _dpp pseudos and VOP3-like
  // _e64_dpp pseudos; the reverse map was built from both getDPPOp32 and
  // getDPPOp64.
  if (auto It = DppToBase.find(P); It != DppToBase.end())
    P = It->second;

  // SDWA -> base. LLVM provides a forward helper for this direction.
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

  // FLAT/GLOBAL SADDR -> VADDR. Only applicable to instructions tagged with
  // the FLAT format flag; the helper returns -1 for non-FLAT opcodes but
  // checking the flag first avoids the lookup for every non-FLAT opcode.
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
  // Flatten kCanonTable into a DenseMap for O(1) lookups during the scan below.
  // A duplicate MC opcode would silently keep only the first row and route the
  // rest through the wrong CanonicalOp, so a duplicate key is a table-authoring
  // bug.
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
  // Heuristic: roughly a quarter of MC opcodes carry a CanonicalOp in practice;
  // resizing a few times is fine for a one-shot init.
  Map.reserve(NumOpc / 4);
  for (unsigned Mc = 0; Mc < NumOpc; ++Mc) {
    const unsigned Canon =
        canonicalize(Mc, MCII, McToPseudo, PseudoAlias, DppToBase);
    if (auto It = CanonToSem.find(Canon); It != CanonToSem.end())
      Map[Mc] = It->second;
    // Every other MC opcode stays Unknown so the raiser's dispatch refuses it.
  }
}

} // namespace COMGR::hotswap
