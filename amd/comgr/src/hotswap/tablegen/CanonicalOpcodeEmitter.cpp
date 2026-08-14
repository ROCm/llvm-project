//===- CanonicalOpcodeEmitter.cpp - Hotswap canonical opcode tables -------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits the hotswap raiser's canonical opcode tables from
// `src/hotswap/CanonicalOpcodes.td`.
//
// The raiser dispatches on `CanonicalOp`, an architecture-neutral instruction
// identity. Getting from an MC opcode (what the AMDGPU disassembler hands us)
// to a CanonicalOp requires collapsing every encoding and subtarget variant
// LLVM generates for one semantic instruction onto a single canonical pseudo,
// then looking that pseudo up in a hand-maintained table. That collapse is a
// pure function of the AMDGPU TableGen files, so it is computed here, once, at
// build time. The raiser is left with a dense array index.
//
// The canonicalization chain mirrors `AMDGPU::getMCOpcode` and friends:
//
//   MC (subtarget-specific real) -> pseudo   (getMCOpcodeGen, inverted)
//   pseudo -> base pseudo                    (name-marker stripping)
//   DPP -> base                              (getDPPOp32/getDPPOp64, inverted)
//   SDWA -> base                             (getBasicFromSDWAOp)
//   e32 -> e64                               (getVOPe64)
//   pseudo -> base pseudo                    (again; see canonicalize())
//   FLAT/GLOBAL SADDR -> VADDR               (getGlobalVaddrOp)
//
// Two outputs are produced from a single parse of AMDGPU.td (which dominates
// the run time, so we do not want to pay for it twice):
//
//   <prefix>.inc      the CanonicalOp enum body and its name table, included
//                     by canonical-op.h
//   <prefix>Impl.cpp  a translation unit with the dense lookup tables and the
//                     out-of-line definitions of canonicalOpFor(),
//                     vcmpMetaFor() and canonicalOpName()
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenTarget.h"

// AMDGPU target-private header, for the `SIInstrFlags` TSFlags bit masks. The
// alias-collapse predicates below are literal ports of the runtime predicates
// that used to live in `src/hotswap/opcode-map.cpp`, and reading the same
// flags from the same enum is what keeps them literal.
#include "SIDefines.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// InstrMapping evaluation
//===----------------------------------------------------------------------===//

// A queryable view of an `InstrMapping` record (llvm/include/llvm/Target/
// Target.td). This is the same relation that `-gen-instr-info` turns into the
// `getVOPe64` / `getMCOpcodeGen` / ... query functions in
// AMDGPUGenInstrInfo.inc; we evaluate it directly instead of calling the
// generated functions, because this runs before (and instead of) that
// generated code being available.
//
// Semantics are taken from `MapTableEmitter` in
// llvm/utils/TableGen/CodeGenMapTable.cpp: rows are keyed on the uniqued
// `Init`s of the RowFields, columns on the unquoted string form of the
// ColFields. Reading `FilterClass` / `RowFields` / `ColFields` / `KeyCol` /
// `ValueCols` off the record rather than hard-coding them means the backend
// follows SIInstrInfo.td if LLVM re-shapes one of these maps.
class InstrMappingTable {
public:
  InstrMappingTable(const RecordKeeper &Records, StringRef MapName)
      : Name(MapName.str()) {
    const Record *MapRec = Records.getDef(MapName);
    if (!MapRec)
      PrintFatalError("hotswap-tblgen: no InstrMapping named '" + Name +
                      "'. AMDGPU may have renamed or removed it; update "
                      "CanonicalOpcodeEmitter.cpp.");

    // getAsUnquotedString returns by value; hold it, don't alias it.
    std::string FilterClass =
        MapRec->getValue("FilterClass")->getValue()->getAsUnquotedString();
    const ListInit *RowFields = MapRec->getValueAsListInit("RowFields");
    const ListInit *ColFields = MapRec->getValueAsListInit("ColFields");
    const ListInit *KeyCol = MapRec->getValueAsListInit("KeyCol");

    KeyColKey = joinColumn(KeyCol);
    for (const Init *VC :
         MapRec->getValueAsListInit("ValueCols")->getElements())
      ValueColKeys.push_back(joinColumn(cast<ListInit>(VC)));

    for (const Record *Instr : Records.getAllDerivedDefinitions(FilterClass)) {
      std::vector<const Init *> RowKey;
      for (const Init *RF : RowFields->getElements())
        RowKey.push_back(fieldValue(Instr, RF));

      std::string ColKey;
      for (const Init *CF : ColFields->getElements()) {
        if (!ColKey.empty())
          ColKey += '\x1f';
        ColKey += fieldValue(Instr, CF)->getAsUnquotedString();
      }

      std::vector<Cell> &Row = Rows[RowKey];
      Row.push_back({ColKey, Instr});
      RowOf[Instr] = &Row;
      if (ColKey == KeyColKey)
        KeyInstrs.insert(Instr);
    }
  }

  unsigned getNumValueCols() const { return ValueColKeys.size(); }

  // Returns the instruction sitting in `From`'s row at value column
  // `ValueColIdx`, or null. Mirrors the generated query function: a
  // non-key-column instruction has no row of its own to search, so it maps to
  // nothing.
  const Record *lookup(const Record *From, unsigned ValueColIdx) const {
    if (!KeyInstrs.contains(From))
      return nullptr;
    typename DenseMap<const Record *, const std::vector<Cell> *>::const_iterator
        It = RowOf.find(From);
    if (It == RowOf.end())
      return nullptr;

    const Record *Match = nullptr;
    for (const Cell &C : *It->second) {
      if (C.ColKey != ValueColKeys[ValueColIdx])
        continue;
      if (Match)
        PrintFatalError(From->getLoc(),
                        "hotswap-tblgen: multiple matches for '" +
                            From->getName() + "' in relation '" + Name + "'");
      Match = C.Instr;
    }
    return Match;
  }

private:
  struct Cell {
    std::string ColKey;
    const Record *Instr;
  };

  static std::string joinColumn(const ListInit *Col) {
    std::string S;
    for (const Init *I : Col->getElements()) {
      if (!S.empty())
        S += '\x1f';
      S += I->getAsUnquotedString();
    }
    return S;
  }

  const Init *fieldValue(const Record *Instr, const Init *Field) const {
    const RecordVal *RV = Instr->getValue(Field);
    if (!RV)
      PrintFatalError(Instr->getLoc(),
                      "hotswap-tblgen: no field " + Field->getAsString() +
                          " on '" + Instr->getName() +
                          "', required by relation '" + Name + "'");
    return RV->getValue();
  }

  std::string Name;
  std::string KeyColKey;
  std::vector<std::string> ValueColKeys;
  // std::map node addresses are stable, so `RowOf` may point into it.
  std::map<std::vector<const Init *>, std::vector<Cell>> Rows;
  DenseMap<const Record *, const std::vector<Cell> *> RowOf;
  DenseSet<const Record *> KeyInstrs;
};

//===----------------------------------------------------------------------===//
// Alias-collapse predicates
//===----------------------------------------------------------------------===//

// Bits we require to be identical between source and target for an alias
// collapse to be considered semantically safe. Deliberately excludes encoding
// variation flags like `VOP3_OPSEL` (set on `_t16_` op-sel encodings but not
// on the base `_e64`) and `renamedInGFX9` (set only on the subtarget-specific
// pseudo). Everything listed below represents *what the handler dispatches
// on*: instruction family (SOP/VOP/FLAT/DS/...), atomic kind, and MAI.
constexpr uint64_t KSemanticShapeMask =
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

// The dispatch-relevant facts about an instruction: the same two things the
// runtime predicates used to read out of `MCInstrDesc` (`TSFlags` and
// `getNumDefs()`), read here off the TableGen record instead.
struct InstrShape {
  uint64_t TSFlags = 0;
  unsigned NumDefs = 0;
};

using RulePredicate = bool (*)(const InstrShape &Src, const InstrShape &Tgt);

// `_RTN` collapse: source must be an atomic with a return value; target must
// be the same atomic without one. The raiser uses `numDefs` as the
// "publishes old value" signal, so that invariant must also hold.
bool atomicRetToNoRet(const InstrShape &Src, const InstrShape &Tgt) {
  constexpr uint64_t KRet = SIInstrFlags::IsAtomicRet;
  constexpr uint64_t KNoRet = SIInstrFlags::IsAtomicNoRet;
  return (Src.TSFlags & KRet) && (Tgt.TSFlags & KNoRet) && Src.NumDefs > 0 &&
         Tgt.NumDefs == 0;
}

// `_vgprcd_` / `_mac_` collapse: both source and target must be MFMA
// (matrix-accumulate) pseudos.
bool bothAreMAI(const InstrShape &Src, const InstrShape &Tgt) {
  constexpr uint64_t KMAI = SIInstrFlags::IsMAI;
  return (Src.TSFlags & KMAI) && (Tgt.TSFlags & KMAI);
}

// Subtarget-/operand-class variants (`_gfx9`, `_t16_`, `_fake16_`, `_agpr`,
// etc.) may legitimately toggle encoding flags such as `VOP3_OPSEL` or
// `renamedInGFX9` between source and target, but they must preserve the
// instruction's dispatch identity: same family, same atomic kind, same MAI
// classification, same def arity. A violation means LLVM renamed or
// repurposed a pseudo in a way our alias map cannot safely collapse.
bool sameSemanticShape(const InstrShape &Src, const InstrShape &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Src.NumDefs == Tgt.NumDefs;
}

// `_nosdst_` collapse: starting with GFX11, VOPC CMPX instructions no longer
// write a scalar destination register (EXEC receives the mask directly) and
// LLVM represents this as a `_nosdst_` variant. The non-`_nosdst_` target
// form keeps the scalar dst (for older subtargets). Both forms share
// dispatch-relevant TSFlags; the raiser's CMPX handler only writes EXEC and
// ignores the optional sdst, so collapsing the variant onto the base is safe.
bool nosdstDropsScalarDef(const InstrShape &Src, const InstrShape &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Tgt.NumDefs >= Src.NumDefs && Tgt.NumDefs - Src.NumDefs <= 1;
}

//===----------------------------------------------------------------------===//
// Vector-compare metadata
//===----------------------------------------------------------------------===//

// Mirror of `COMGR::hotswap::VCmpMeta`. `Pred` is spelled as the C++ enumerator
// text so the emitted table needs no numeric coupling to CmpInst::Predicate.
struct VCmpMeta {
  std::string Pred;
  unsigned Bits = 0;
  bool IsFloat = false;
  bool IsClass = false;

  bool operator<(const VCmpMeta &O) const {
    return std::tie(Pred, Bits, IsFloat, IsClass) <
           std::tie(O.Pred, O.Bits, O.IsFloat, O.IsClass);
  }
};

// Parse a canonical vector-compare pseudo name into (predicate, bits, kind).
// Accepted shape: `V_CMP_<PRED>_<TYPE><BITS>_e64` where
//   PRED  in {EQ, NE, GT, GE, LT, LE, LG, NEQ, NLT, NLE, NGT, NGE, NLG, U, O,
//            CLASS}
//   TYPE  in {U, I, F} (CLASS only ever appears with TYPE=F)
//   BITS  in {16, 32, 64}
// and an optional `V_CMPX_` prefix plays the role of `V_CMP_`. Returns
// `std::nullopt` for anything else.
//
// Rationale: LLVM exposes `AMDGPU::getVCMPXOpFromVCMP` as a V_CMP -> V_CMPX
// mapping, but no relation that hands back a CmpInst::Predicate or element
// width. Rather than hand-list ~100 opcode/metadata pairs we parse the pseudo
// name; the same token grammar is already hard-coded in LLVM's TableGen for
// these instructions.
//
// CLASS is special: `V_CMP_CLASS_F<bits>` is *not* a predicate compare. src1
// is an i32 mask of FP classes, and the result lane bit is set iff src0's IEEE
// class matches any enabled bit in the mask. It collapses onto the same
// `V_CMP` / `V_CMPX` CanonicalOps and signals the special-case lift via
// `VCmpMeta::IsClass`.
std::optional<VCmpMeta> parseVCmpPseudoName(StringRef Name) {
  StringRef Rest = Name;
  if (!Rest.consume_front("V_CMPX_") && !Rest.consume_front("V_CMP_"))
    return std::nullopt;
  if (!Rest.consume_back("_e64"))
    return std::nullopt;

  std::pair<StringRef, StringRef> Split = Rest.rsplit('_');
  StringRef PredTok = Split.first;
  StringRef TypeTok = Split.second;
  if (PredTok.empty() || TypeTok.size() < 2)
    return std::nullopt;

  const char TypeCh = TypeTok[0];
  unsigned Bits = 0;
  if (TypeTok.drop_front().getAsInteger(10, Bits))
    return std::nullopt;
  if (Bits != 16 && Bits != 32 && Bits != 64)
    return std::nullopt;

  VCmpMeta M;
  M.Pred = "BAD_ICMP_PREDICATE";
  M.Bits = Bits;

  // V_CMP_CLASS_F<bits> / V_CMPX_CLASS_F<bits>: floating-point classification
  // mask, not a predicate compare. The handler takes the `IsClass` branch and
  // ignores `Pred`; we leave `Pred` as BAD_ICMP_PREDICATE so any accidental
  // FCmp/ICmp use would assert loudly rather than silently miscompile.
  if (PredTok == "CLASS") {
    if (TypeCh != 'F')
      return std::nullopt;
    M.IsFloat = true;
    M.IsClass = true;
    return M;
  }

  if (TypeCh == 'F') {
    M.IsFloat = true;
    // Float predicates: ordered variants set the O-prefix predicates;
    // N-prefixed AMDGPU names select the "unordered-or-..." complements.
    if (PredTok == "EQ")
      M.Pred = "FCMP_OEQ";
    else if (PredTok == "GT")
      M.Pred = "FCMP_OGT";
    else if (PredTok == "GE")
      M.Pred = "FCMP_OGE";
    else if (PredTok == "LT")
      M.Pred = "FCMP_OLT";
    else if (PredTok == "LE")
      M.Pred = "FCMP_OLE";
    // LG ("less or greater"), NE, and NEQ all mean "ordered and !=" in
    // AMDGPU's model and all lower to FCMP_ONE.
    else if (PredTok == "LG" || PredTok == "NE" || PredTok == "NEQ")
      M.Pred = "FCMP_ONE";
    else if (PredTok == "NLT")
      M.Pred = "FCMP_UGE";
    else if (PredTok == "NLE")
      M.Pred = "FCMP_UGT";
    else if (PredTok == "NGT")
      M.Pred = "FCMP_ULE";
    else if (PredTok == "NGE")
      M.Pred = "FCMP_ULT";
    // NLG ("not (less or greater)") is the unordered-or-equal complement.
    else if (PredTok == "NLG")
      M.Pred = "FCMP_UEQ";
    else if (PredTok == "U")
      M.Pred = "FCMP_UNO";
    else if (PredTok == "O")
      M.Pred = "FCMP_ORD";
    else
      return std::nullopt;
  } else if (TypeCh == 'U' || TypeCh == 'I') {
    const bool IsSigned = TypeCh == 'I';
    if (PredTok == "EQ")
      M.Pred = "ICMP_EQ";
    else if (PredTok == "NE")
      M.Pred = "ICMP_NE";
    else if (PredTok == "GT")
      M.Pred = IsSigned ? "ICMP_SGT" : "ICMP_UGT";
    else if (PredTok == "GE")
      M.Pred = IsSigned ? "ICMP_SGE" : "ICMP_UGE";
    else if (PredTok == "LT")
      M.Pred = IsSigned ? "ICMP_SLT" : "ICMP_ULT";
    else if (PredTok == "LE")
      M.Pred = IsSigned ? "ICMP_SLE" : "ICMP_ULE";
    else
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  return M;
}

//===----------------------------------------------------------------------===//
// The emitter
//===----------------------------------------------------------------------===//

class CanonicalOpcodeEmitter {
public:
  explicit CanonicalOpcodeEmitter(const RecordKeeper &Records)
      : Records(Records) {}

  TableGenOutputFiles run(StringRef FilenamePrefix);

private:
  void collectCanonOps();
  void collectInstructions();
  void buildMcToPseudoMap();
  void buildPseudoAliasMap();
  void buildDppToBaseMap();
  void buildCanonToSem();
  unsigned canonicalize(unsigned Mc) const;
  void buildTables();

  void emitEnum(raw_ostream &OS) const;
  void emitImpl(raw_ostream &OS) const;

  const RecordKeeper &Records;

  // CanonicalOp enumerators, in .td source order. Index + 1 is the enum value
  // (0 is reserved for `Unknown`).
  std::vector<const Record *> CanonOps;
  DenseMap<const Record *, unsigned> CanonOpValue;
  StringMap<unsigned> CanonOpByEnumName;
  // Indexed by enum value; the spelling canonicalOpName() hands back.
  std::vector<std::string> CanonOpNames;
  unsigned NumCanonOpValues = 0;

  // Instructions in MC opcode order (matches AMDGPUGenInstrInfo.inc).
  std::vector<const Record *> InstrRecs;
  std::vector<InstrShape> Shapes;
  DenseMap<const Record *, unsigned> OpcodeOf;
  StringMap<unsigned> OpcodeByName;
  unsigned NumOpc = 0;

  DenseMap<unsigned, unsigned> McToPseudo;
  DenseMap<unsigned, unsigned> PseudoAlias;
  DenseMap<unsigned, unsigned> DppToBase;
  // Relations consulted directly by canonicalize().
  std::unique_ptr<InstrMappingTable> SdwaToBase;
  std::unique_ptr<InstrMappingTable> VOPe64;
  std::unique_ptr<InstrMappingTable> GlobalVaddr;
  // Canonical pseudo opcode -> CanonicalOp enum value.
  DenseMap<unsigned, unsigned> CanonToSem;
  // Canonical pseudo opcode -> the .td row that produced it, for diagnostics
  // and for the emitted static_asserts.
  std::vector<std::pair<unsigned, unsigned>> CanonRows;

  // Final dense tables.
  std::vector<unsigned> CanonByOpcode;
  std::vector<unsigned> VCmpIndexByOpcode;
  std::vector<VCmpMeta> VCmpPool;
};

//===----------------------------------------------------------------------===//

void CanonicalOpcodeEmitter::collectCanonOps() {
  ArrayRef<const Record *> All =
      Records.getAllDerivedDefinitions("CanonOpBase");
  std::vector<const Record *> Defs(All.begin(), All.end());
  // getAllDerivedDefinitions sorts by record name; restore .td source order so
  // the generated enum keeps the authored grouping.
  llvm::sort(Defs, [](const Record *A, const Record *B) {
    return A->getID() < B->getID();
  });

  StringMap<const Record *> ByEnumName;
  unsigned NextValue = 1; // 0 is CanonicalOp::Unknown.
  for (const Record *R : Defs) {
    // CanonOpBase derives EnumName as `!substr(NAME, 4)`; a def that skipped
    // the `COP_` prefix would silently lose its first four characters.
    if (!R->getName().starts_with("COP_"))
      PrintFatalError(R->getLoc(), "hotswap-tblgen: CanonOp def '" +
                                       R->getName() +
                                       "' must be named COP_<enumerator>");
    StringRef EnumName = R->getValueAsString("EnumName");
    if (EnumName.empty())
      PrintFatalError(R->getLoc(), "hotswap-tblgen: CanonOp '" + R->getName() +
                                       "' has an empty EnumName");
    std::pair<StringMap<const Record *>::iterator, bool> Ins =
        ByEnumName.try_emplace(EnumName, R);
    if (!Ins.second)
      PrintFatalError(R->getLoc(),
                      "hotswap-tblgen: two CanonOp defs share EnumName '" +
                          EnumName + "': '" + Ins.first->second->getName() +
                          "' and '" + R->getName() + "'");

    unsigned Value;
    if (const RecordVal *RV = R->getValue("AliasOf")) {
      // `NAME = Target` in the generated enum. C++ requires the target to be
      // declared first, and so do we.
      const Record *Target = cast<DefInit>(RV->getValue())->getDef();
      DenseMap<const Record *, unsigned>::const_iterator It =
          CanonOpValue.find(Target);
      if (It == CanonOpValue.end())
        PrintFatalError(R->getLoc(),
                        "hotswap-tblgen: '" + R->getName() + "' aliases '" +
                            Target->getName() +
                            "', which is not defined earlier in the file");
      Value = It->second;
    } else {
      Value = NextValue++;
    }

    CanonOpValue[R] = Value;
    CanonOpByEnumName[EnumName] = Value;
    CanonOps.push_back(R);
    if (!R->getValueAsBit("IsSentinel")) {
      if (Value >= CanonOpNames.size())
        CanonOpNames.resize(Value + 1);
      if (!CanonOpNames[Value].empty())
        PrintFatalError(R->getLoc(),
                        "hotswap-tblgen: '" + EnumName + "' and '" +
                            CanonOpNames[Value] +
                            "' share a value but neither is marked "
                            "IsSentinel, so canonicalOpName would be "
                            "ambiguous");
      CanonOpNames[Value] = EnumName.str();
    }
  }

  if (CanonOps.empty())
    PrintFatalError("hotswap-tblgen: no CanonOp definitions found");

  NumCanonOpValues = NextValue;
  CanonOpNames.resize(NumCanonOpValues);
  CanonOpNames[0] = "Unknown";
  for (unsigned I = 0; I != NumCanonOpValues; ++I)
    if (CanonOpNames[I].empty())
      PrintFatalError("hotswap-tblgen: CanonicalOp value " + Twine(I) +
                      " has only sentinel spellings; one of them must drop "
                      "`let IsSentinel = 1;`");
}

void CanonicalOpcodeEmitter::collectInstructions() {
  CodeGenTarget Target(Records);
  // Ordered by enum value, i.e. the numbering AMDGPUGenInstrInfo.inc uses.
  ArrayRef<const CodeGenInstruction *> Insts = Target.getInstructions();
  NumOpc = Insts.size();

  InstrRecs.reserve(NumOpc);
  Shapes.reserve(NumOpc);
  for (unsigned I = 0; I != NumOpc; ++I) {
    const Record *R = Insts[I]->TheDef;
    InstrRecs.push_back(R);
    OpcodeOf[R] = I;
    OpcodeByName.try_emplace(R->getName(), I);

    InstrShape S;
    S.NumDefs = Insts[I]->Operands.NumDefs;
    // Target-independent opcodes (PHI, INLINEASM, G_*) are not InstSI
    // subclasses and carry no TSFlags field; they stay all-zero, which is
    // what MCInstrDesc reports for them too.
    if (const RecordVal *RV = R->getValue("TSFlags")) {
      if (const BitsInit *Bits = dyn_cast<BitsInit>(RV->getValue())) {
        if (std::optional<int64_t> V = Bits->convertInitializerToInt())
          S.TSFlags = static_cast<uint64_t>(*V);
      }
    }
    Shapes.push_back(S);
  }
}

// Build a reverse map MC-opcode -> canonical pseudo.
//
// The runtime version inverted `AMDGPU::getMCOpcode(P, Gen)` by scanning every
// pseudo across all 15 encoding families. Here we evaluate `getMCOpcodeGen`
// directly and invert it in one pass. Each real instruction sits in exactly
// one column of exactly one row, so the inversion is unambiguous.
void CanonicalOpcodeEmitter::buildMcToPseudoMap() {
  InstrMappingTable Map(Records, "getMCOpcodeGen");
  const unsigned NumCols = Map.getNumValueCols();
  for (unsigned P = 0; P != NumOpc; ++P) {
    for (unsigned Col = 0; Col != NumCols; ++Col) {
      const Record *Real = Map.lookup(InstrRecs[P], Col);
      if (!Real)
        continue;
      DenseMap<const Record *, unsigned>::const_iterator It =
          OpcodeOf.find(Real);
      if (It == OpcodeOf.end() || It->second == P)
        continue;
      McToPseudo.try_emplace(It->second, P);
    }
  }
}

// Build an alias map that collapses "parallel" pseudos LLVM generates for the
// same semantic instruction into a single canonical pseudo. Examples:
//   DS_WRITE_B16_gfx9        -> DS_WRITE_B16
//   V_ADD_F16_t16_e64        -> V_ADD_F16_e64
//   V_ADD_F16_fake16_e64     -> V_ADD_F16_e64
// LLVM has no relation for this collapse, so we match on pseudo name.
void CanonicalOpcodeEmitter::buildPseudoAliasMap() {
  struct Rule {
    StringRef Needle;
    bool IsSuffix;
    // Optional semantic check on (source, target). A null predicate means "no
    // validation yet".
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
      // arity + atomic/MAI classification) is preserved.
      {"_OP_SEL_", false, sameSemanticShape},
      // GFX11+ VOPC CMPX family drops the scalar destination register; the
      // raiser's CMPX handler only touches EXEC so the `_nosdst_` form
      // collapses cleanly onto the base pseudo of the same encoding width.
      {"_nosdst_", false, nosdstDropsScalarDef},
      // MFMA register-class modifiers. `_vgprcd_` marks a VGPR destination
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
      // back, which the raiser already derives from `MCInstrDesc::getNumDefs`.
      {"_agpr", true, sameSemanticShape},
      {"_RTN", true, atomicRetToNoRet},
  };

  // Returns the index of the firing rule or -1 if no rule applies.
  auto StripOnce = [&](StringRef Name, std::string &Out) -> int {
    for (size_t I = 0; I < std::size(Rules); ++I) {
      const Rule &R = Rules[I];
      if (R.IsSuffix) {
        if (!Name.ends_with(R.Needle))
          continue;
        Out = Name.drop_back(R.Needle.size()).str();
        return static_cast<int>(I);
      }
      size_t Pos = Name.find(R.Needle);
      if (Pos == StringRef::npos)
        continue;
      Out = (Name.substr(0, Pos).str() + std::string("_") +
             Name.substr(Pos + R.Needle.size()).str());
      return static_cast<int>(I);
    }
    return -1;
  };

  for (unsigned P = 0; P != NumOpc; ++P) {
    std::string Cur = InstrRecs[P]->getName().str();
    unsigned CurOpc = P;
    unsigned FinalOpc = P;
    while (true) {
      std::string Next;
      int RuleIdx = StripOnce(Cur, Next);
      if (RuleIdx < 0)
        break;
      StringMap<unsigned>::const_iterator It = OpcodeByName.find(Next);
      if (It != OpcodeByName.end() && It->second != P) {
        const Rule &R = Rules[RuleIdx];
        if (R.Pred && !R.Pred(Shapes[CurOpc], Shapes[It->second]))
          PrintFatalError(
              InstrRecs[CurOpc]->getLoc(),
              Twine("hotswap-tblgen: alias rule '") + R.Needle +
                  "' broke its semantic invariant while collapsing '" + Cur +
                  "' -> '" + Next +
                  "'. LLVM likely renamed or repurposed a pseudo; update the "
                  "alias rules or the predicate in "
                  "CanonicalOpcodeEmitter.cpp.");
        CurOpc = It->second;
        FinalOpc = CurOpc;
      }
      Cur = std::move(Next);
    }
    if (FinalOpc != P)
      PseudoAlias.try_emplace(P, FinalOpc);
  }
}

// Build a reverse DPP map: DPP opcode -> base VOP opcode. The relations only
// go base -> DPP32 / DPP64, so we invert them.
void CanonicalOpcodeEmitter::buildDppToBaseMap() {
  InstrMappingTable Dpp32(Records, "getDPPOp32");
  InstrMappingTable Dpp64(Records, "getDPPOp64");
  for (unsigned P = 0; P != NumOpc; ++P) {
    for (const InstrMappingTable *M : {&Dpp32, &Dpp64}) {
      const Record *D = M->lookup(InstrRecs[P], 0);
      if (!D)
        continue;
      DenseMap<const Record *, unsigned>::const_iterator It = OpcodeOf.find(D);
      if (It != OpcodeOf.end())
        DppToBase.try_emplace(It->second, P);
    }
  }
}

// Canonicalize any MC opcode `Mc` to the pseudo form matched by the .td rows.
unsigned CanonicalOpcodeEmitter::canonicalize(unsigned Mc) const {
  unsigned P = Mc;

  // MC (subtarget-specific real) -> pseudo.
  if (DenseMap<unsigned, unsigned>::const_iterator It = McToPseudo.find(P);
      It != McToPseudo.end())
    P = It->second;

  // Parallel-pseudo alias -> base pseudo (strips _gfx9, _t16_, _fake16_).
  if (DenseMap<unsigned, unsigned>::const_iterator It = PseudoAlias.find(P);
      It != PseudoAlias.end())
    P = It->second;

  // DPP -> base. This handles both VOP2-like _dpp pseudos and VOP3-like
  // _e64_dpp pseudos; the reverse map was built from both relations.
  if (DenseMap<unsigned, unsigned>::const_iterator It = DppToBase.find(P);
      It != DppToBase.end())
    P = It->second;

  // SDWA -> base.
  if (const Record *B = SdwaToBase->lookup(InstrRecs[P], 0)) {
    DenseMap<const Record *, unsigned>::const_iterator It = OpcodeOf.find(B);
    if (It != OpcodeOf.end() && It->second > 0)
      P = It->second;
  }

  // e32 -> e64.
  if (const Record *E = VOPe64->lookup(InstrRecs[P], 0)) {
    DenseMap<const Record *, unsigned>::const_iterator It = OpcodeOf.find(E);
    if (It != OpcodeOf.end() && It->second > 0)
      P = It->second;
  }

  // Re-apply the pseudo-alias step. The e32 -> e64 step can resolve an `_e32`
  // pseudo (e.g. `V_LSHLREV_B64_pseudo_e32`) to an `_e64` pseudo with a
  // parallel variant marker (`V_LSHLREV_B64_pseudo_e64`) that only collapses
  // once both the `_pseudo_` infix and `_e64` suffix are visible together.
  if (DenseMap<unsigned, unsigned>::const_iterator It = PseudoAlias.find(P);
      It != PseudoAlias.end())
    P = It->second;

  // FLAT/GLOBAL SADDR -> VADDR. Only applicable to instructions tagged with
  // the FLAT format flag.
  if ((Shapes[P].TSFlags & SIInstrFlags::FLAT) != 0) {
    if (const Record *V = GlobalVaddr->lookup(InstrRecs[P], 0)) {
      DenseMap<const Record *, unsigned>::const_iterator It = OpcodeOf.find(V);
      if (It != OpcodeOf.end() && It->second > 0)
        P = It->second;
    }
  }

  return P;
}

// Flatten the `Canon` rows into a pseudo-opcode -> CanonicalOp map.
//
// The duplicate-key audit is the build-time replacement for the runtime one
// that used to run in `OpcodeMap::build`: a `try_emplace` loop silently keeps
// the first insertion on key collision, which is exactly what once let
// S_ADD_U64 get mapped twice with the second row losing the routing race.
// Rejecting it here means the mistake cannot ship.
void CanonicalOpcodeEmitter::buildCanonToSem() {
  ArrayRef<const Record *> All = Records.getAllDerivedDefinitions("Canon");
  std::vector<const Record *> Rows(All.begin(), All.end());
  llvm::sort(Rows, [](const Record *A, const Record *B) {
    return A->getID() < B->getID();
  });

  DenseMap<unsigned, const Record *> RowFor;
  for (const Record *Row : Rows) {
    const Record *Pseudo = Row->getValueAsDef("Pseudo");
    const Record *Op = Row->getValueAsDef("Op");

    DenseMap<const Record *, unsigned>::const_iterator PIt =
        OpcodeOf.find(Pseudo);
    if (PIt == OpcodeOf.end())
      PrintFatalError(Row->getLoc(), "hotswap-tblgen: '" + Pseudo->getName() +
                                         "' is not an AMDGPU instruction");
    DenseMap<const Record *, unsigned>::const_iterator OIt =
        CanonOpValue.find(Op);
    if (OIt == CanonOpValue.end())
      PrintFatalError(Row->getLoc(), "hotswap-tblgen: '" + Op->getName() +
                                         "' is not a CanonOp");

    std::pair<DenseMap<unsigned, unsigned>::iterator, bool> Ins =
        CanonToSem.try_emplace(PIt->second, OIt->second);
    if (!Ins.second) {
      const Record *First = RowFor[PIt->second];
      StringRef FirstName =
          First->getValueAsDef("Op")->getValueAsString("EnumName");
      StringRef SecondName = Op->getValueAsString("EnumName");
      std::string Msg =
          ("hotswap-tblgen: CanonicalOpcodes.td maps MC opcode '" +
           Pseudo->getName() + "' to TWO CanonicalOps: first = CanonicalOp::" +
           FirstName + ", second = CanonicalOp::" + SecondName)
              .str();
      if (FirstName == SecondName)
        Msg += ".  (Both targets are the same -- the row is redundant; remove "
               "one.)";
      else
        Msg += ".  Only the first row would take effect and the second would "
               "be dead.  Pick ONE CanonicalOp target and remove the loser "
               "row.";
      PrintFatalError(Row->getLoc(), Msg);
    }
    RowFor[PIt->second] = Row;
    CanonRows.emplace_back(PIt->second, OIt->second);
  }
}

void CanonicalOpcodeEmitter::buildTables() {
  CanonByOpcode.assign(NumOpc, 0);
  VCmpIndexByOpcode.assign(NumOpc, 0);
  // Index 0 of the pool is the "not a compare" slot and is never handed out.
  VCmpMeta None;
  None.Pred = "BAD_ICMP_PREDICATE";
  VCmpPool.push_back(None);
  std::map<VCmpMeta, unsigned> PoolIndex;

  for (unsigned Mc = 0; Mc != NumOpc; ++Mc) {
    const unsigned Canon = canonicalize(Mc);
    if (DenseMap<unsigned, unsigned>::const_iterator It =
            CanonToSem.find(Canon);
        It != CanonToSem.end()) {
      CanonByOpcode[Mc] = It->second;
      continue;
    }

    // The canonical pseudo was not enumerated in the .td. Check if it belongs
    // to the V_CMP / V_CMPX family, which is handled via the metadata side
    // table rather than per-opcode enumeration. Use the canonical pseudo's
    // name so DPP/SDWA variants (already folded by canonicalize) need no
    // re-canonicalization.
    StringRef CanonName = InstrRecs[Canon]->getName();
    const bool IsCmp = CanonName.starts_with("V_CMP_");
    const bool IsCmpX = CanonName.starts_with("V_CMPX_");
    if (!IsCmp && !IsCmpX)
      continue;
    std::optional<VCmpMeta> Meta = parseVCmpPseudoName(CanonName);
    if (!Meta)
      // Names that start with V_CMP_ but don't parse (e.g. a hypothetical
      // future family) are left as CanonicalOp::Unknown so the raiser reports
      // them loudly rather than silently producing wrong IR.
      continue;

    StringRef Collapsed = IsCmpX ? "V_CMPX" : "V_CMP";
    CanonByOpcode[Mc] = CanonOpByEnumName.lookup(Collapsed);
    if (CanonByOpcode[Mc] == 0)
      PrintFatalError("hotswap-tblgen: CanonicalOpcodes.td must define a "
                      "CanonOp with EnumName \"" +
                      Collapsed +
                      "\"; the whole V_CMP/V_CMPX family "
                      "collapses onto it");

    std::pair<std::map<VCmpMeta, unsigned>::iterator, bool> Ins =
        PoolIndex.try_emplace(*Meta, VCmpPool.size());
    if (Ins.second)
      VCmpPool.push_back(*Meta);
    VCmpIndexByOpcode[Mc] = Ins.first->second;
  }
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

// Re-indent a `code` block so it can be emitted as a C++ comment: drop leading
// and trailing blank lines, then remove the common indentation.
std::vector<std::string> docLines(StringRef Doc) {
  SmallVector<StringRef, 32> Raw;
  Doc.split(Raw, '\n');
  while (!Raw.empty() && Raw.front().trim().empty())
    Raw.erase(Raw.begin());
  while (!Raw.empty() && Raw.back().trim().empty())
    Raw.pop_back();

  size_t Common = StringRef::npos;
  for (StringRef L : Raw) {
    if (L.trim().empty())
      continue;
    Common = std::min(Common, L.size() - L.ltrim().size());
  }
  if (Common == StringRef::npos)
    Common = 0;

  std::vector<std::string> Out;
  for (StringRef L : Raw)
    Out.push_back(L.size() >= Common ? L.drop_front(Common).rtrim().str()
                                     : L.trim().str());
  return Out;
}

void CanonicalOpcodeEmitter::emitEnum(raw_ostream &OS) const {
  emitSourceFileHeader("Hotswap canonical opcode enum", OS, Records);

  OS << "#ifdef GET_CANONICAL_OP_ENUM\n";
  OS << "#undef GET_CANONICAL_OP_ENUM\n\n";
  OS << "  Unknown = 0,\n";
  for (const Record *R : CanonOps) {
    std::vector<std::string> Doc = docLines(R->getValueAsString("Doc"));
    if (!Doc.empty())
      OS << '\n';
    for (const std::string &L : Doc)
      OS << (L.empty() ? "  //\n" : "  // " + L + "\n");
    OS << "  " << R->getValueAsString("EnumName") << " = ";
    // Spell an alias as `NAME = Target` rather than as a bare number, so the
    // header still shows which enumerators deliberately share a value.
    if (const RecordVal *RV = R->getValue("AliasOf"))
      OS << cast<DefInit>(RV->getValue())
                ->getDef()
                ->getValueAsString("EnumName")
         << ",\n";
    else
      OS << CanonOpValue.lookup(R) << ",\n";
  }
  OS << "\n  CanonicalOp_COUNT = " << NumCanonOpValues << ",\n";
  OS << "\n#endif // GET_CANONICAL_OP_ENUM\n";
}

void CanonicalOpcodeEmitter::emitImpl(raw_ostream &OS) const {
  emitSourceFileHeader("Hotswap canonical opcode tables", OS, Records);

  // This file is written to the build tree, so it cannot use the bare
  // `"canonical-op.h"` spelling its siblings in the source tree use; go
  // through the target's public `hotswap/` include root instead.
  OS << "#include \"hotswap/canonical-op.h\"\n";
  OS << "#include \"hotswap/opcode-map.h\"\n\n";
  OS << "// AMDGPU:: opcode enumerators, for the numbering static_asserts "
        "below.\n";
  OS << "#include \"MCTargetDesc/AMDGPUMCTargetDesc.h\"\n\n";
  OS << "#include <cstdint>\n#include <iterator>\n\n";
  OS << "namespace COMGR::hotswap {\nnamespace {\n\n";

  // -- CanonicalOp names -----------------------------------------------------
  OS << "// Indexed by CanonicalOp. Range sentinels share a value with a real\n"
        "// canonical opcode and do not get a slot of their own.\n";
  OS << "const char *const CanonicalOpNames[] = {\n";
  for (const std::string &N : CanonOpNames)
    OS << "    \"" << N << "\",\n";
  OS << "};\n";
  OS << "static_assert(std::size(CanonicalOpNames) ==\n"
        "                  static_cast<unsigned>(CanonicalOp::CanonicalOp_"
        "COUNT),\n"
        "              \"CanonicalOp enum and name table are out of sync\");\n"
        "\n";

  // -- MC opcode -> CanonicalOp ---------------------------------------------
  OS << "// Dense, indexed by MC opcode. Zero means CanonicalOp::Unknown.\n";
  OS << "constexpr uint16_t CanonByOpcode[] = {\n";
  for (unsigned I = 0; I != NumOpc; ++I) {
    OS << (I % 16 == 0 ? "    " : " ") << CanonByOpcode[I] << ',';
    if (I % 16 == 15 || I + 1 == NumOpc)
      OS << '\n';
  }
  OS << "};\n\n";

  // -- V_CMP / V_CMPX metadata ----------------------------------------------
  OS << "// V_CMP / V_CMPX side table. Index 0 is the \"not a compare\" "
        "slot.\n";
  OS << "constexpr VCmpMeta VCmpPool[] = {\n";
  for (const VCmpMeta &M : VCmpPool)
    OS << "    {llvm::CmpInst::" << M.Pred << ", " << M.Bits << ", "
       << (M.IsFloat ? "true" : "false") << ", "
       << (M.IsClass ? "true" : "false") << "},\n";
  OS << "};\n\n";

  OS << "constexpr uint16_t VCmpIndexByOpcode[] = {\n";
  for (unsigned I = 0; I != NumOpc; ++I) {
    OS << (I % 16 == 0 ? "    " : " ") << VCmpIndexByOpcode[I] << ',';
    if (I % 16 == 15 || I + 1 == NumOpc)
      OS << '\n';
  }
  OS << "};\n\n";

  // -- Numbering guards ------------------------------------------------------
  OS << "// The tables above are indexed by the opcode numbering "
        "hotswap-tblgen\n"
        "// computed from AMDGPU.td. These asserts prove that numbering still "
        "matches\n"
        "// the one AMDGPUGenInstrInfo.inc was generated with: the count, "
        "plus one\n"
        "// spot check per row of CanonicalOpcodes.td.\n";
  OS << "static_assert(std::size(CanonByOpcode) == "
        "llvm::AMDGPU::INSTRUCTION_LIST_END,\n"
        "              \"hotswap-tblgen and AMDGPUGenInstrInfo.inc disagree "
        "on the\"\n"
        "              \" AMDGPU opcode count; regenerate the hotswap "
        "tables\");\n";
  OS << "static_assert(std::size(VCmpIndexByOpcode) == "
        "llvm::AMDGPU::INSTRUCTION_LIST_END);\n\n";
  for (const std::pair<unsigned, unsigned> &Row : CanonRows)
    OS << "static_assert(CanonByOpcode[llvm::AMDGPU::"
       << InstrRecs[Row.first]->getName() << "] == " << Row.second << ");\n";
  OS << '\n';

  OS << "} // namespace\n\n";

  // -- Accessors -------------------------------------------------------------
  OS << "const char *canonicalOpName(CanonicalOp Op) {\n"
        "  unsigned I = static_cast<unsigned>(Op);\n"
        "  return I < std::size(CanonicalOpNames) ? CanonicalOpNames[I]\n"
        "                                        : \"<unknown "
        "CanonicalOp>\";\n"
        "}\n\n";

  OS << "CanonicalOp canonicalOpFor(unsigned Opcode) {\n"
        "  return Opcode < std::size(CanonByOpcode)\n"
        "             ? static_cast<CanonicalOp>(CanonByOpcode[Opcode])\n"
        "             : CanonicalOp::Unknown;\n"
        "}\n\n";

  OS << "const VCmpMeta *vcmpMetaFor(unsigned Opcode) {\n"
        "  if (Opcode >= std::size(VCmpIndexByOpcode))\n"
        "    return nullptr;\n"
        "  uint16_t I = VCmpIndexByOpcode[Opcode];\n"
        "  return I ? &VCmpPool[I] : nullptr;\n"
        "}\n\n";

  OS << "} // namespace COMGR::hotswap\n";
}

TableGenOutputFiles CanonicalOpcodeEmitter::run(StringRef /*FilenamePrefix*/) {
  collectCanonOps();
  collectInstructions();
  SdwaToBase =
      std::make_unique<InstrMappingTable>(Records, "getBasicFromSDWAOp");
  VOPe64 = std::make_unique<InstrMappingTable>(Records, "getVOPe64");
  GlobalVaddr =
      std::make_unique<InstrMappingTable>(Records, "getGlobalVaddrOp");
  buildMcToPseudoMap();
  buildPseudoAliasMap();
  buildDppToBaseMap();
  buildCanonToSem();
  buildTables();

  std::string Enum;
  raw_string_ostream EnumOS(Enum);
  emitEnum(EnumOS);

  std::string Impl;
  raw_string_ostream ImplOS(Impl);
  emitImpl(ImplOS);

  return {std::move(Enum), {{"Impl.cpp", std::move(Impl)}}};
}

} // namespace

static TableGen::Emitter::MultiFileOptClass<CanonicalOpcodeEmitter>
    X("gen-hotswap-canonical-opcodes",
      "Generate the hotswap raiser's canonical opcode tables");
