//===- opcode-map.cpp - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "opcode-map.h"

#include <cstdint>
#include <optional>
#include <string>

// AMDGPU target-private headers. They expose:
//   AMDGPU::getMCOpcode           (declared in Utils/AMDGPUBaseInfo.h)
//   AMDGPU::getVOPe64 / getVOPe32 / getDPPOp32 / getDPPOp64 /
//   getSDWAOp / getBasicFromSDWAOp / getGlobalVaddrOp
//                                  (declared in SIInstrInfo.h, implemented
//                                   in the TableGen-generated
//                                   AMDGPUGenInstrInfo.inc under
//                                   `#define GET_INSTRMAP_INFO`, linked from
//                                   libLLVMAMDGPUUtils.a).
//
// SIInstrInfo.h drags in the CodeGen TargetInstrInfo base, which we do not
// use at runtime, but pulling it in is preferable to hand-rolling forward
// declarations that would silently go stale if LLVM changes a signature.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Maps a canonical AMDGPU pseudo opcode to the CanonicalOp the raiser
// dispatches on. The canonical form is what comes out of the canonicalization
// chain below:
//   MC opcode -> pseudo
//   pseudo    -> e64 (if VOP/VOPC has e32/e64 split)
//   pseudo    -> base (if SDWA/DPP)
//   pseudo    -> VADDR (if FLAT/GLOBAL SADDR)
//
// Using AMDGPU:: enum constants instead of strings gives us compile-time
// checking: if LLVM renames a pseudo, the build fails here rather than the
// lookup silently returning CanonicalOp::Unknown at runtime.
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
      int Mc = AMDGPU::getMCOpcode(P, Gen);
      if (Mc > 0 && static_cast<unsigned>(Mc) != P)
        Result.try_emplace(static_cast<unsigned>(Mc), P);
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

// `_RTN` collapse: source must be an atomic with a return value; target must
// be the same atomic without one. The raiser uses `numDefs` as the
// "publishes old value" signal, so that invariant must also hold.
static bool atomicRetToNoRet(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  constexpr uint64_t KRet = SIInstrFlags::IsAtomicRet;
  constexpr uint64_t KNoRet = SIInstrFlags::IsAtomicNoRet;
  return (Src.TSFlags & KRet) && (Tgt.TSFlags & KNoRet) &&
         Src.getNumDefs() > 0 && Tgt.getNumDefs() == 0;
}

// `_vgprcd_` / `_mac_` collapse: both source and target must be MFMA
// (matrix-accumulate) pseudos.
static bool bothAreMAI(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  constexpr uint64_t KMAI = SIInstrFlags::IsMAI;
  return (Src.TSFlags & KMAI) && (Tgt.TSFlags & KMAI);
}

// `_nosdst_` collapse: starting with GFX11, VOPC CMPX instructions no longer
// write a scalar destination register (EXEC receives the mask directly) and
// LLVM represents this as a `_nosdst_` variant. The non-`_nosdst_` target
// form keeps the scalar dst (for older subtargets). Both forms share
// dispatch-relevant TSFlags; the raiser's CMPX handler only writes EXEC and
// ignores the optional sdst, so collapsing the variant onto the base is
// safe. The source has one fewer def when the base includes sdst (e64
// forms) and the same number of defs otherwise (e32, where both lack sdst).

// Bits we require to be identical between source and target for an alias
// collapse to be considered semantically safe. Deliberately excludes encoding
// variation flags like `VOP3_OPSEL` (set on `_t16_` op-sel encodings but not
// on the base `_e64`) and `renamedInGFX9` (set only on the subtarget-specific
// pseudo). Everything listed below represents *what the handler dispatches
// on*: instruction family (SOP/VOP/FLAT/DS/...), atomic kind, and MAI.
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

  struct Rule {
    llvm::StringRef Needle;
    bool IsSuffix;
    // Optional semantic check on (source, target) MCInstrDesc. A null
    // predicate means "no validation yet" (see the older subtarget/operand
    // markers below).
    RulePredicate Pred;
  };
  // Subtarget-specific markers ("_gfx9", "_gfx1250", ...) and operand-size
  // markers ("_t16_", "_fake16_") that LLVM injects into the pseudo name.
  // A single pseudo can carry multiple markers (e.g.
  // `V_BITOP3_B16_gfx1250_fake16_e64`), so the outer loop below applies these
  // rules iteratively until the name stops shrinking.
  static const Rule Rules[] = {
      // Subtarget-specific markers. LLVM emits a dedicated pseudo per
      // subtarget (e.g. `_gfx9`, `_gfx1250`, `_vi_gfx9`) with the same
      // TableGen class as the base; collapsing them is sound as long as
      // TSFlags and def arity match.
      {"_vi_gfx9", true, sameSemanticShape},
      {"_gfx9", true, sameSemanticShape},
      {"_gfx1250", true, sameSemanticShape},
      {"_gfx1250_", false, sameSemanticShape},
      {"_pseudo_", false, sameSemanticShape},
      // True16 / Fake16 mark the 16-bit operand encoding variant; LLVM has
      // no dedicated TSFlag bit for this (the distinction lives in
      // True16Predicate on the TableGen side), and the t16 encoding toggles
      // `VOP3_OPSEL`. We cross-check that dispatch-relevant TSFlags and def
      // arity are preserved, but tolerate encoding-bit drift.
      {"_t16_", false, sameSemanticShape},
      {"_fake16_", false, sameSemanticShape},
      // `_OP_SEL_` infix marks the gfx11+ encoding variant for VOP1 cvt
      // instructions (e.g. v_cvt_f32_{fp8,bf8}, v_cvt_pk_f32_{fp8,bf8}).
      // The OP_SEL pseudo carries an extra `byte_sel` immediate operand and
      // toggles `VOP3_OPSEL`/`maybeAtomic`/`ASYNC_CNT` bits relative to the
      // base e64 pseudo, but dispatch identity (instruction family + def
      // arity + atomic/MAI classification) is preserved. Collapsing onto
      // the base pseudo lets a single CanonicalOp handler service both the
      // pre-gfx11 SDWA/byte_sel-via-disassembly form and the gfx11+ encoded
      // byte_sel form; the handler reads the byte_sel from the disassembly
      // text (`op_sel:`), which is identical for both.
      {"_OP_SEL_", false, sameSemanticShape},
      // GFX11+ VOPC CMPX family drops the scalar destination register; the
      // raiser's CMPX handler only touches EXEC so the `_nosdst_` form
      // collapses cleanly onto the base pseudo of the same encoding width.
      {"_nosdst_", false, nosdstDropsScalarDef},
      // MFMA register-class modifiers.  `_vgprcd_` marks a VGPR destination
      // variant; `_mac_` marks a multiply-accumulate (tied dst/src2) variant.
      // Both keep the same TableGen intrinsic and semantic shape, so they
      // collapse onto the base `_e64` pseudo.
      {"_vgprcd_", false, bothAreMAI},
      {"_mac_", false, bothAreMAI},
      // Atomic return-value variants: LLVM emits distinct `_RTN` pseudos for
      // the forms that return the pre-modification value, plus `_agpr`
      // variants that just pick an AGPR destination register class. These
      // pseudos carry the same TableGen intrinsic and identical semantics;
      // the only difference is whether the handler should write the result
      // back, which the raiser already derives from `di.numDefs`
      // (MCInstrDesc::getNumDefs()). Collapse them onto the non-RTN pseudo
      // so both forms share a single CanonicalOp.
      {"_agpr", true, sameSemanticShape},
      {"_RTN", true, atomicRetToNoRet},
  };

  // Returns the index of the firing rule or -1 if no rule applies.
  auto StripOnce = [&](llvm::StringRef Name, std::string &Out) -> int {
    for (size_t I = 0; I < std::size(Rules); ++I) {
      const Rule &R = Rules[I];
      if (R.IsSuffix) {
        if (!Name.ends_with(R.Needle))
          continue;
        Out = Name.drop_back(R.Needle.size()).str();
        return static_cast<int>(I);
      }
      size_t Pos = Name.find(R.Needle);
      if (Pos == llvm::StringRef::npos)
        continue;
      Out = (Name.substr(0, Pos).str() + std::string("_") +
             Name.substr(Pos + R.Needle.size()).str());
      return static_cast<int>(I);
    }
    return -1;
  };

  DenseMap<unsigned, unsigned> Alias;
  for (const auto &Kv : ByName) {
    std::string Cur = Kv.first().str();
    unsigned CurOpc = Kv.second;
    unsigned FinalOpc = Kv.second;
    while (true) {
      std::string Next;
      int RuleIdx = StripOnce(Cur, Next);
      if (RuleIdx < 0)
        break;
      auto It = ByName.find(Next);
      if (It != ByName.end() && It->second != Kv.second) {
        const Rule &R = Rules[RuleIdx];
        if (R.Pred && !R.Pred(MCII.get(CurOpc), MCII.get(It->second))) {
          report_fatal_error(
              Twine("opcode_map: alias rule '") + R.Needle +
              "' broke its semantic invariant while collapsing '" + Cur +
              "' -> '" + Next +
              "'. LLVM likely renamed or repurposed a pseudo; update the "
              "alias rules or the predicate.");
        }
        CurOpc = It->second;
        FinalOpc = CurOpc;
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
    int D32 = AMDGPU::getDPPOp32(P);
    if (D32 > 0)
      Result.try_emplace(static_cast<unsigned>(D32), P);
    int D64 = AMDGPU::getDPPOp64(P);
    if (D64 > 0)
      Result.try_emplace(static_cast<unsigned>(D64), P);
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

  // Parallel-pseudo alias -> base pseudo (strips _gfx9, _t16_, _fake16_).
  if (auto It = PseudoAlias.find(P); It != PseudoAlias.end())
    P = It->second;

  // DPP -> base. This handles both VOP2-like _dpp pseudos and VOP3-like
  // _e64_dpp pseudos; the reverse map was built from both getDPPOp32 and
  // getDPPOp64.
  if (auto It = DppToBase.find(P); It != DppToBase.end())
    P = It->second;

  // SDWA -> base. LLVM provides a forward helper for this direction.
  int Base = AMDGPU::getBasicFromSDWAOp(P);
  if (Base > 0)
    P = static_cast<unsigned>(Base);

  // e32 -> e64.
  int E64 = AMDGPU::getVOPe64(P);
  if (E64 > 0)
    P = static_cast<unsigned>(E64);

  // Re-apply the pseudo-alias step. `getVOPe64` can resolve an `_e32` pseudo
  // (e.g. `V_LSHLREV_B64_pseudo_e32`) to an `_e64` pseudo with a parallel
  // variant marker (`V_LSHLREV_B64_pseudo_e64`) that only collapses once
  // both the `_pseudo_` infix and `_e64` suffix are visible together.
  if (auto It = PseudoAlias.find(P); It != PseudoAlias.end())
    P = It->second;

  // FLAT/GLOBAL SADDR -> VADDR. Only applicable to instructions tagged with
  // the FLAT format flag; the helper returns -1 for non-FLAT opcodes but
  // checking the flag first avoids the lookup for every non-FLAT opcode.
  if (P < MCII.getNumOpcodes() &&
      (MCII.get(P).TSFlags & SIInstrFlags::FLAT) != 0) {
    int Vaddr = AMDGPU::getGlobalVaddrOp(P);
    if (Vaddr > 0)
      P = static_cast<unsigned>(Vaddr);
  }

  return P;
}

} // namespace

CanonicalOp OpcodeMap::lookup(unsigned Opcode) const {
  auto It = Map.find(Opcode);
  return It != Map.end() ? It->second : CanonicalOp::Unknown;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // Flatten the static kCanonTable into a DenseMap for O(1) lookups
  // during the subsequent scan over every MC opcode.  This flatten
  // step also serves as a duplicate-key audit for `kCanonTable`: the
  // earlier implementation used a plain `canonToSem.try_emplace` loop,
  // which silently keeps the first insertion on key collision.  That
  // silent-first-wins behaviour is exactly what let S_ADD_U64 get
  // mapped twice (once as SOP2-block `E(S_ADD_U64, S_ADD_U64)` and
  // once as gfx12-rename `E(S_ADD_U64, S_ADD_NC_U64)`) with the
  // second row silently losing the routing race and leaving every
  // `s_add_u64` lift routed through the wrong CanonicalOp -- see commit
  // eaee0a0e88 for the repair.  The loop below now returns an error on
  // duplicate keys so a future add that re-introduces the collision
  // is caught at transpiler init time instead of quietly miscompiling
  // everything under the duplicated opcode.
  //
  // A returned error (rather than `assert`) keeps the check active in
  // release builds too -- the cost is a single `try_emplace` per table
  // row at process start, which is negligible for a ~800-entry table.
  //
  // "Same CanonicalOp twice" is also rejected.  In principle a redundant
  // row that maps the same MC opcode to the same CanonicalOp is just
  // noise the first-wins rule would swallow harmlessly, but we
  // refuse it anyway so the `kCanonTable` stays honest -- a
  // redundant row is always an editing mistake, never an
  // intentional design, and the abort makes the fix mechanical
  // (delete one of the two rows).
  DenseMap<unsigned, CanonicalOp> CanonToSem;
  CanonToSem.reserve(std::size(kCanonTable));
  for (const Entry &E : kCanonTable) {
    auto [existing, inserted] = CanonToSem.try_emplace(E.Opc, E.Sem);
    if (!inserted) {
      std::string Msg;
      raw_string_ostream Os(Msg);
      Os << "opcode-map.cpp: kCanonTable maps MC opcode '"
         << MCII.getName(E.Opc) << "' (enum value " << E.Opc
         << ") to TWO CanonicalOps: " << "first = CanonicalOp::"
         << canonicalOpName(existing->second)
         << ", second = CanonicalOp::" << canonicalOpName(E.Sem);
      if (existing->second == E.Sem) {
        Os << ".  (Both targets are the same -- the row is redundant; "
              "remove one.)";
      } else {
        Os << ".  `canonToSem.try_emplace` keeps the first insertion, "
              "so every `"
           << MCII.getName(E.Opc)
           << "` lift silently routes through CanonicalOp::"
           << canonicalOpName(existing->second)
           << " and the second row is dead.  Pick ONE CanonicalOp target "
              "and remove the loser row.";
      }
      report_fatal_error(Msg.c_str());
    }
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
