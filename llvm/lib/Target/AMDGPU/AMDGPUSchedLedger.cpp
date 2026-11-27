//===- AMDGPUSchedLedger.cpp - Profile-guided scheduling ledger -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// A ledger is a JSON file of scheduling hints produced outside the compiler
/// from profiles. This mutation applies the per-region part of it:
///
///   latency  overrides the latency of an instruction's outgoing data edges
///   order    adds artificial ordering edges between named instructions
///
/// Both only constrain or re-weight the existing dependency graph; an edge that
/// would create a cycle is rejected, so every resulting schedule is legal.
///
/// A region entry names its region by content: a signature of the region's
/// instructions (opcodes, immediates, physical registers and the position of
/// each use's definition inside the region), and names instructions by their
/// offset in the region. Code added elsewhere in the function therefore does
/// not invalidate or shift an entry, and an entry is applied to every region
/// with its signature in every function its (possibly broad) name match
/// selects, so one entry serves all instantiations that contain the same code.
/// Nothing here reads debug info: a build without it gets the same schedule. With
/// PSCHED_SALVAGE set, an entry whose region changed is carried over to the
/// most similar region by aligning the two opcode sequences (reported).
/// Version 1 entries (absolute ids, opcode hash) are still accepted.
///
/// Internally instructions carry a per-function id assigned in program order
/// before the first scheduling region is built. The id is carried on the
/// instruction as !pcsections metadata, which makes AsmPrinter emit an
/// (address, function hash, id) table in section "psched_ids". The same
/// mutation can dump each region's DAG as one JSON line for offline analysis.
///
/// Inputs (command line option, else environment variable):
///   -amdgpu-sched-ledger / PSCHED_LEDGER         colon-separated files or dirs
///   -amdgpu-sched-ledger-dump / PSCHED_DUMP_DIR  directory for <function>.jsonl
///   PSCHED_DUMP_FILTER                           dump only functions whose name contains this
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSchedLedger.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <list>

using namespace llvm;

static cl::opt<std::string> LedgerPathsOpt(
    "amdgpu-sched-ledger", cl::Hidden,
    cl::desc("Colon-separated scheduling ledger files or directories "
             "(default: $PSCHED_LEDGER)"));

static cl::opt<std::string> DumpDirOpt(
    "amdgpu-sched-ledger-dump", cl::Hidden,
    cl::desc("Directory receiving one <function>.jsonl scheduling DAG dump "
             "per function (default: $PSCHED_DUMP_DIR)"));

namespace {

constexpr StringLiteral IdSection = "psched_ids";

struct RegionEntry {
  std::string Hash; // v1: opcode hash of the region; v2: content signature
  std::string Origin;
  // v2: Latency/Order hold offsets within the region instead of ids, and the
  // entry is a template that is bound to every region with this signature.
  bool Relative = false;
  unsigned NumInsts = 0;
  std::vector<std::string> Ops; // opcode per offset, for salvage (optional)
  unsigned Bound = 0;           // regions bound in the current function
  bool NearMiss = false;        // reported as changed in the current function
  unsigned BoundTotal = 0;      // regions bound in this compilation
  unsigned Offered = 0;         // functions it was selected for in this compilation
  unsigned BestSim = 0;         // best alignment seen where it did not bind
  std::string BestFn;
  std::vector<std::pair<unsigned, unsigned>> Latency;
  std::vector<std::vector<unsigned>> Order;
  bool Validated = false;
  bool Stale = false;
  bool CycleWarned = false;
  unsigned PairGuarded = 0; // order edges skipped because they would split a dual-issue pair
};

struct FunctionEntry {
  std::string Match;
  std::string Source;
  std::vector<RegionEntry> Regions;
};

/// PSCHED_DUMP_FILTER=<substring>: dump only functions whose name contains it.
/// A dump of every function of a large library is gigabytes.
bool dumpWanted(StringRef FnName) {
  static const char *Filter = std::getenv("PSCHED_DUMP_FILTER");
  return !Filter || !*Filter || FnName.contains(Filter);
}

std::string envOr(const cl::opt<std::string> &Opt, const char *Env) {
  if (!Opt.empty())
    return Opt;
  const char *V = std::getenv(Env);
  return V ? V : "";
}

void parseLedgerFile(StringRef Path, std::vector<FunctionEntry> &Out) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(Path);
  if (!Buf) {
    errs() << "psched: cannot read ledger '" << Path << "'\n";
    return;
  }
  Expected<json::Value> Root = json::parse((*Buf)->getBuffer());
  if (!Root) {
    errs() << "psched: ledger '" << Path
           << "' is not valid JSON: " << toString(Root.takeError()) << "\n";
    return;
  }
  const json::Object *Obj = Root->getAsObject();
  const json::Array *Fns = Obj ? Obj->getArray("functions") : nullptr;
  if (!Fns)
    return;
  for (const json::Value &V : *Fns) {
    const json::Object *E = V.getAsObject();
    const json::Array *Regions = E ? E->getArray("regions") : nullptr;
    if (!Regions)
      continue;
    FunctionEntry FE;
    FE.Match = E->getString("match").value_or("").str();
    FE.Source = E->getString("source").value_or("").str();
    for (const json::Value &RV : *Regions) {
      const json::Object *R = RV.getAsObject();
      if (!R)
        continue;
      RegionEntry RE;
      RE.Hash = R->getString("hash").value_or("").str();
      RE.Origin = Path.str();
      if (auto Sig = R->getString("sig")) {
        RE.Relative = true;
        RE.Hash = Sig->str();
        RE.NumInsts = R->getInteger("n").value_or(0);
        const json::Array *Tab = R->getArray("optab");
        const json::Array *Ops = R->getArray("ops");
        if (Tab && Ops)
          for (const json::Value &OV : *Ops) {
            auto Idx = OV.getAsInteger();
            auto Name = Idx && *Idx >= 0 && size_t(*Idx) < Tab->size()
                            ? (*Tab)[*Idx].getAsString()
                            : std::nullopt;
            RE.Ops.push_back(Name ? Name->str() : "?");
          }
      }
      if (const json::Array *L = R->getArray("latency"))
        for (const json::Value &LV : *L)
          if (const json::Object *LO = LV.getAsObject())
            if (auto Id = LO->getInteger(RE.Relative ? "at" : "id"))
              RE.Latency.emplace_back(*Id, LO->getInteger("lat").value_or(0));
      if (const json::Array *O = R->getArray("order"))
        for (const json::Value &CV : *O)
          if (const json::Array *Chain = CV.getAsArray()) {
            std::vector<unsigned> Ids;
            for (const json::Value &IV : *Chain)
              if (auto Id = IV.getAsInteger())
                Ids.push_back(*Id);
            if (Ids.size() > 1)
              RE.Order.push_back(std::move(Ids));
          }
      if (!RE.Latency.empty() || !RE.Order.empty())
        FE.Regions.push_back(std::move(RE));
    }
    if (!FE.Match.empty() && !FE.Regions.empty())
      Out.push_back(std::move(FE));
  }
}

std::vector<FunctionEntry> &getLedger();

/// End-of-compilation report (runs from llvm_shutdown). A version 2 entry is
/// offered to every function its name match selects (often all instantiations
/// of a template) and is expected to be absent from most of them, so whether
/// it went stale can only be said once all of them have been compiled.
struct LedgerSummary {
  ~LedgerSummary() {
    for (FunctionEntry &FE : getLedger())
      for (RegionEntry &R : FE.Regions) {
        // Not offered: this compilation does not contain the code the entry
        // is for (device code is generated per translation unit).
        if (!R.Relative || R.BoundTotal || !R.Offered)
          continue;
        errs() << "psched: warning: region " << R.Hash << " (" << R.Origin
               << ") matched no code in this compilation (stale ledger?)";
        if (R.BestSim)
          errs() << "; the closest region aligned " << R.BestSim << "%, in "
                 << R.BestFn << " (PSCHED_SALVAGE=1 carries the entry over)";
        errs() << "\n";
      }
  }
};
ManagedStatic<LedgerSummary> Summary;

/// All region-level ledger entries, loaded once per process.
std::vector<FunctionEntry> &getLedger() {
  static std::vector<FunctionEntry> Ledger;
  static bool Loaded = false;
  if (Loaded)
    return Ledger;
  Loaded = true;
  (void)*Summary;
  auto Bypass = sys::sandbox::scopedDisable();
  std::string Spec = envOr(LedgerPathsOpt, "PSCHED_LEDGER");
  SmallVector<StringRef, 4> Paths;
  StringRef(Spec).split(Paths, ':', -1, false);
  for (StringRef P : Paths) {
    if (!sys::fs::is_directory(P)) {
      parseLedgerFile(P, Ledger);
      continue;
    }
    std::vector<std::string> Files;
    std::error_code EC;
    for (sys::fs::directory_iterator I(P, EC), E; I != E && !EC; I.increment(EC))
      if (sys::path::extension(I->path()) == ".json")
        Files.push_back(I->path());
    llvm::sort(Files);
    for (const std::string &F : Files)
      parseLedgerFile(F, Ledger);
  }
  return Ledger;
}

/// State for the function currently being compiled. Machine functions are
/// compiled one at a time, pre-RA scheduling before post-RA scheduling.
struct FunctionState {
  std::string Name;
  unsigned Number = ~0u;
  uint32_t NameHash = 0;
  bool IdsAssigned = false;
  bool DumpStarted = false;
  bool PressureStarted = false;
  std::vector<RegionEntry *> Regions;
  DenseSet<unsigned> Dumped;
  // Version 2: templates selected for this function, the regions they were
  // bound to (with absolute ids), and the function's region table.
  std::vector<RegionEntry *> Templates;
  std::list<RegionEntry> BoundStore;
  struct RegionInfo {
    std::string Sig;
    unsigned NumInsts = 0;
  };
  std::vector<RegionInfo> Table;
  DenseMap<unsigned, std::pair<unsigned, unsigned>> IdPos; // id -> (region, offset)
  bool TableBuilt = false;

  void report() {
    for (RegionEntry *R : Regions)
      if (!R->Validated && !R->Stale)
        errs() << "psched: warning: region " << R->Hash << " (" << R->Origin
               << ") matched nothing in " << Name << " (stale ledger?)\n";
    // Whether a version 2 entry is stale is decided at the end of the
    // compilation (LedgerSummary); per function its absence is only a note.
    if (TableBuilt && std::getenv("PSCHED_VERBOSE"))
      for (RegionEntry *T : Templates)
        if (!T->Bound && !T->NearMiss)
          errs() << "psched: region " << T->Hash << " (" << T->Origin
                 << ") is not in " << Name << "\n";
    Regions.clear();
    Templates.clear();
    BoundStore.clear();
    Table.clear();
    IdPos.clear();
    TableBuilt = false;
  }
  ~FunctionState() { report(); }
};

FunctionState &state() {
  static FunctionState S;
  return S;
}

uint32_t fnv1a(StringRef S, uint32_t H = 2166136261u) {
  for (unsigned char C : S)
    H = (H ^ C) * 16777619u;
  return H;
}

FunctionState &enterFunction(MachineFunction &MF) {
  FunctionState &S = state();
  if (S.Name == MF.getName() && S.Number == MF.getFunctionNumber())
    return S;
  S.report();
  S.Name = MF.getName().str();
  S.Number = MF.getFunctionNumber();
  S.NameHash = fnv1a(MF.getName());
  S.IdsAssigned = S.DumpStarted = false;
  S.Dumped.clear();
  // Device code is usually generated at LTO link time, where the module name
  // no longer identifies the translation unit; use the function's debug info.
  // Without debug info the source filter cannot be evaluated and is ignored.
  const DISubprogram *SP = MF.getFunction().getSubprogram();
  if (std::getenv("PSCHED_NO_REGIONS"))
    return S;
  for (FunctionEntry &FE : getLedger()) {
    if (FE.Match != "*" && !MF.getName().contains(FE.Match))
      continue;
    if (!FE.Source.empty() && SP && !SP->getFilename().contains(FE.Source))
      continue;
    for (RegionEntry &R : FE.Regions) {
      R.Validated = R.Stale = R.CycleWarned = false;
      if (R.Relative) {
        R.Bound = 0;
        R.NearMiss = false;
        S.Templates.push_back(&R);
      } else {
        S.Regions.push_back(&R);
      }
    }
  }
  return S;
}

std::optional<unsigned> getId(const MachineInstr &MI) {
  const MDNode *MD = MI.getPCSections();
  if (!MD || MD->getNumOperands() != 2)
    return std::nullopt;
  const auto *Sec = dyn_cast<MDString>(MD->getOperand(0));
  const auto *Aux = dyn_cast<MDNode>(MD->getOperand(1));
  if (!Sec || Sec->getString() != IdSection || !Aux || Aux->getNumOperands() != 2)
    return std::nullopt;
  return mdconst::extract<ConstantInt>(Aux->getOperand(1))->getZExtValue();
}

void assignIds(MachineFunction &MF, FunctionState &S) {
  if (S.IdsAssigned)
    return;
  S.IdsAssigned = true;
  LLVMContext &Ctx = MF.getFunction().getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Metadata *Sec = MDString::get(Ctx, IdSection);
  Metadata *Hash = ConstantAsMetadata::get(ConstantInt::get(I32, S.NameHash));
  // PSCHED_TEST_ID_SKEW=N starts the ids at N, which is what N instructions
  // added in front of everything else in the function would do (test only).
  const char *Skew = std::getenv("PSCHED_TEST_ID_SKEW");
  unsigned Id = Skew ? std::atoi(Skew) : 0;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr() || MI.isPHI() || MI.getPCSections())
        continue;
      Metadata *Aux = MDNode::get(
          Ctx, {Hash, ConstantAsMetadata::get(ConstantInt::get(I32, Id++))});
      MI.setPCSections(MF, MDNode::get(Ctx, {Sec, Aux}));
    }
}

/// Content signature of a region: its instructions in id (original program)
/// order, with opcode names, immediates, physical registers, subregister
/// indices and, for every use of a virtual register, the offset inside the
/// region of the instruction that defines it (or "in" for a value that enters
/// the region). Virtual register numbers and instruction ids do not enter it,
/// so code elsewhere in the function leaves the signature unchanged, while two
/// regions with the same opcodes but different dataflow or constants differ.
std::string regionSig(ArrayRef<const MachineInstr *> Insts,
                      const TargetInstrInfo &TII, const TargetRegisterInfo &TRI) {
  uint32_t H = 2166136261u;
  DenseMap<Register, unsigned> LastDef;
  for (unsigned R = 0; R < Insts.size(); ++R) {
    const MachineInstr &MI = *Insts[R];
    H = fnv1a(TII.getName(MI.getOpcode()), H);
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg()) {
        Register Reg = MO.getReg();
        if (!Reg) {
          H = fnv1a("_", H);
        } else if (Reg.isPhysical()) {
          H = fnv1a(MO.isDef() ? "D" : "U", H);
          H = fnv1a(TRI.getName(Reg), H);
        } else if (MO.isDef()) {
          H = fnv1a("d", H);
        } else {
          auto It = LastDef.find(Reg);
          H = fnv1a(It == LastDef.end() ? std::string("in")
                                        : "u" + utostr(It->second),
                    H);
        }
        if (MO.getSubReg())
          H = fnv1a(TRI.getSubRegIndexName(MO.getSubReg()), H);
      } else if (MO.isImm()) {
        H = fnv1a("i" + itostr(MO.getImm()), H);
      } else if (MO.isGlobal()) {
        H = fnv1a(MO.getGlobal()->getName(), H);
      } else {
        H = fnv1a("t" + utostr(unsigned(MO.getType())), H);
      }
      H = fnv1a(",", H);
    }
    for (const MachineOperand &MO : MI.all_defs())
      if (MO.getReg().isVirtual())
        LastDef[MO.getReg()] = R;
    H = fnv1a(";", H);
  }
  return utohexstr(H, /*LowerCase=*/true, 8);
}

/// Longest common subsequence of two opcode sequences. Returns its length and
/// fills Map[i] with the position in B aligned to position i of A, or ~0u.
unsigned alignOps(ArrayRef<unsigned> A, ArrayRef<unsigned> B,
                  std::vector<unsigned> &Map) {
  size_t N = A.size(), M = B.size();
  std::vector<uint16_t> T((N + 1) * (M + 1), 0);
  auto At = [&](size_t I, size_t J) -> uint16_t & { return T[I * (M + 1) + J]; };
  for (size_t I = N; I-- > 0;)
    for (size_t J = M; J-- > 0;)
      At(I, J) = A[I] == B[J] ? At(I + 1, J + 1) + 1
                              : std::max(At(I + 1, J), At(I, J + 1));
  Map.assign(N, ~0u);
  for (size_t I = 0, J = 0; I < N && J < M;) {
    if (A[I] == B[J]) {
      Map[I] = J;
      ++I, ++J;
    } else if (At(I + 1, J) >= At(I, J + 1)) {
      ++I;
    } else {
      ++J;
    }
  }
  return At(0, 0);
}

/// Splits the function into scheduling regions the way the machine scheduler
/// does, records each region's signature and every instruction's (region,
/// offset), and binds the version 2 ledger templates selected for this
/// function: to every region with the template's signature, or, when there is
/// none and salvage is on, to the most similar region.
void buildTableAndBind(MachineFunction &MF, FunctionState &S) {
  if (S.TableBuilt)
    return;
  S.TableBuilt = true;
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  std::vector<std::vector<const MachineInstr *>> Regions;
  for (MachineBasicBlock &MBB : MF) {
    std::vector<const MachineInstr *> Cur;
    auto Flush = [&] {
      if (Cur.size() > 1)
        Regions.push_back(std::move(Cur));
      Cur.clear();
    };
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugOrPseudoInstr())
        continue;
      if (MI.isCall() || MI.isFakeUse() || MI.isPHI() ||
          TII.isSchedulingBoundary(MI, &MBB, MF))
        Flush();
      else if (getId(MI))
        Cur.push_back(&MI);
    }
    Flush();
  }
  for (unsigned RI = 0; RI < Regions.size(); ++RI) {
    S.Table.push_back({regionSig(Regions[RI], TII, TRI),
                       unsigned(Regions[RI].size())});
    for (unsigned Off = 0; Off < Regions[RI].size(); ++Off)
      S.IdPos[*getId(*Regions[RI][Off])] = {RI, Off};
  }
  if (S.Templates.empty())
    return;

  for (RegionEntry *T : S.Templates)
    ++T->Offered;

  static const bool Verbose = std::getenv("PSCHED_VERBOSE") != nullptr;
  static const bool Salvage = std::getenv("PSCHED_SALVAGE") != nullptr;
  static const unsigned MinSim = [] {
    const char *V = std::getenv("PSCHED_SALVAGE_MIN");
    return V ? unsigned(std::atoi(V)) : 75u;
  }();

  // Binds T to region RI. Map translates a template offset to an offset in the
  // region (identity for an exact match).
  auto Bind = [&](RegionEntry &T, unsigned RI, const std::vector<unsigned> *Map,
                  unsigned &Edges, unsigned &Kept) {
    auto Id = [&](unsigned Off) -> std::optional<unsigned> {
      if (Map)
        Off = Off < Map->size() ? (*Map)[Off] : ~0u;
      if (Off >= Regions[RI].size())
        return std::nullopt;
      return getId(*Regions[RI][Off]);
    };
    RegionEntry B;
    B.Hash = T.Hash;
    B.Origin = T.Origin;
    B.Validated = true;
    for (auto &[Off, Lat] : T.Latency)
      if (auto I = Id(Off))
        B.Latency.emplace_back(*I, Lat);
    Edges = Kept = 0;
    for (const std::vector<unsigned> &Chain : T.Order)
      for (unsigned I = 0; I + 1 < Chain.size(); ++I) {
        ++Edges;
        auto A = Id(Chain[I]), Z = Id(Chain[I + 1]);
        if (A && Z) {
          ++Kept;
          B.Order.push_back({*A, *Z});
        }
      }
    S.BoundStore.push_back(std::move(B));
    S.Regions.push_back(&S.BoundStore.back());
    ++T.Bound;
    ++T.BoundTotal;
  };

  StringMap<unsigned> OpIds;
  auto OpId = [&](StringRef Name) {
    return OpIds.try_emplace(Name, OpIds.size()).first->second;
  };
  for (RegionEntry *T : S.Templates) {
    unsigned Edges = 0, Kept = 0;
    for (unsigned RI = 0; RI < Regions.size(); ++RI)
      if (S.Table[RI].Sig == T->Hash &&
          (!T->NumInsts || T->NumInsts == S.Table[RI].NumInsts)) {
        Bind(*T, RI, nullptr, Edges, Kept);
        if (Verbose)
          errs() << "psched: " << MF.getName() << " <- region " << T->Hash
                 << " (" << S.Table[RI].NumInsts << " instructions): "
                 << T->Latency.size() << " latency override(s), " << Kept
                 << " order edge(s) (" << T->Origin << ")\n";
      }
  }

  // Templates whose signature is in no region: find the region each most
  // likely was, by aligning opcode sequences. A region is given to at most one
  // such template (the best alignment first), so two templates derived from
  // two similar regions do not both land on the same one.
  struct Cand {
    unsigned Sim, RI, TI;
    std::vector<unsigned> Map;
  };
  std::vector<Cand> Cands;
  DenseSet<unsigned> Taken;
  for (RegionEntry &B : S.BoundStore)
    for (auto &Chain : B.Order)
      if (auto It = S.IdPos.find(Chain[0]); It != S.IdPos.end())
        Taken.insert(It->second.first);
  for (unsigned TI = 0; TI < S.Templates.size(); ++TI) {
    RegionEntry *T = S.Templates[TI];
    if (T->Bound || T->Ops.empty())
      continue;
    std::vector<unsigned> A;
    for (const std::string &Op : T->Ops)
      A.push_back(OpId(Op));
    for (unsigned RI = 0; RI < Regions.size(); ++RI) {
      size_t M = Regions[RI].size();
      if (Taken.count(RI) || M * 4 < A.size() * 3 || M * 3 > A.size() * 4 ||
          M > 60000 || A.size() > 60000)
        continue;
      std::vector<unsigned> B, Map;
      for (const MachineInstr *MI : Regions[RI])
        B.push_back(OpId(TII.getName(MI->getOpcode())));
      unsigned Sim = 200 * alignOps(A, B, Map) / (A.size() + B.size());
      Cands.push_back({Sim, RI, TI, std::move(Map)});
    }
  }
  llvm::stable_sort(Cands, [](const Cand &X, const Cand &Y) {
    return std::tie(Y.Sim, X.RI, X.TI) < std::tie(X.Sim, Y.RI, Y.TI);
  });
  for (Cand &C : Cands) {
    RegionEntry *T = S.Templates[C.TI];
    if (T->Bound || T->NearMiss || Taken.count(C.RI))
      continue;
    if (C.Sim > T->BestSim) {
      T->BestSim = C.Sim;
      T->BestFn = MF.getName().str();
    }
    if (C.Sim < MinSim) {
      if (Verbose)
        errs() << "psched: region " << T->Hash << " (" << T->Origin << ", "
               << T->Ops.size() << " instructions): closest region in "
               << MF.getName() << " is " << S.Table[C.RI].Sig << " ("
               << S.Table[C.RI].NumInsts << " instructions), aligned " << C.Sim
               << "%, below the salvage threshold of " << MinSim << "%\n";
      T->NearMiss = true;
      continue;
    }
    T->NearMiss = true;
    Taken.insert(C.RI);
    unsigned Edges = 0, Kept = 0;
    if (Salvage) {
      Bind(*T, C.RI, &C.Map, Edges, Kept);
      errs() << "psched: warning: region " << T->Hash << " (" << T->Origin
             << ") has changed in " << MF.getName() << ": " << C.Sim
             << "% of its instructions align with region " << S.Table[C.RI].Sig
             << "; salvaged " << Kept << " of " << Edges
             << " order edge(s). Measure again and re-derive the entry.\n";
    } else if (Verbose) {
      // Expected in most functions the entry is offered to (another
      // instantiation of the same code); LedgerSummary warns if it bound
      // nowhere at all.
      errs() << "psched: region " << T->Hash << " (" << T->Origin << ") is not in "
             << MF.getName() << ": the closest region (" << S.Table[C.RI].Sig
             << ") aligns " << C.Sim
             << "%; not applied here (PSCHED_SALVAGE=1 carries it over)\n";
    }
  }
}

StringRef classify(const MachineInstr &MI) {
  if (SIInstrInfo::isMFMAorWMMA(MI))
    return "matrix";
  if (SIInstrInfo::isDS(MI))
    return MI.mayStore() ? "ds_write" : "ds_read";
  if (SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI))
    return MI.mayStore() ? "vmem_write" : "vmem_read";
  if (SIInstrInfo::isSMRD(MI))
    return "smem";
  if (SIInstrInfo::isTRANS(MI))
    return "trans";
  if (SIInstrInfo::isVALU(MI, /*AllowLDSDMA=*/false))
    return "valu";
  if (SIInstrInfo::isSALU(MI))
    return "salu";
  return "other";
}

StringRef depKind(const SDep &D) {
  switch (D.getKind()) {
  case SDep::Data:
    return "data";
  case SDep::Anti:
    return "anti";
  case SDep::Output:
    return "output";
  case SDep::Order:
    break;
  }
  if (D.isCluster())
    return "cluster";
  if (D.isArtificial())
    return "artificial";
  if (D.isWeak())
    return "weak";
  if (D.isBarrier())
    return "barrier";
  if (D.isNormalMemory())
    return "memory";
  return "order";
}

std::string locString(const DILocation *L) {
  return (sys::path::filename(L->getFilename()) + ":" + Twine(L->getLine())).str();
}

class SchedLedger : public ScheduleDAGMutation {
  using IdMap = DenseMap<unsigned, SUnit *>;

  void dump(ScheduleDAGMI &DAG, FunctionState &S, const IdMap &Ids,
            unsigned Key, StringRef Hash, bool PostRA);
  void applyRegion(ScheduleDAGMI &DAG, RegionEntry &R, const IdMap &Ids,
                   StringRef Hash, bool PostRA);

public:
  void apply(ScheduleDAGInstrs *DAGInstrs) override;
};

void SchedLedger::dump(ScheduleDAGMI &DAG, FunctionState &S, const IdMap &Ids,
                       unsigned Key, StringRef Hash, bool PostRA) {
  std::string Dir = envOr(DumpDirOpt, "PSCHED_DUMP_DIR");
  // Post-RA regions are dumped too (marked "post_ra"): the post-RA scheduler
  // has additional dependencies, and an ordering edge must be legal there as
  // well. Ids are below 2^31, so the top bit separates the two key spaces.
  if (Dir.empty() || !dumpWanted(S.Name) ||
      !S.Dumped.insert(Key | (PostRA ? 0x80000000u : 0)).second)
    return;
  auto Bypass = sys::sandbox::scopedDisable();
  sys::fs::create_directories(Dir);
  SmallString<256> Path(Dir);
  sys::path::append(Path, S.Name + ".jsonl");
  std::error_code EC;
  raw_fd_ostream OS(Path, EC,
                    S.DumpStarted ? sys::fs::OF_Append : sys::fs::OF_None);
  if (EC) {
    errs() << "psched: cannot write '" << Path << "': " << EC.message() << "\n";
    return;
  }
  S.DumpStarted = true;

  const MachineRegisterInfo &MRI = DAG.MF.getRegInfo();
  const auto *TRI = static_cast<const SIRegisterInfo *>(DAG.TRI);
  json::Array Nodes, Edges;
  // Virtual VGPR registers touched in the region, for the live-range report:
  // each node lists [register, width in 32-bit registers] for its defs and
  // uses, and the region lists the registers still live after it.
  auto VGPRWidth = [&](Register R) -> unsigned {
    if (!R.isVirtual())
      return 0;
    const TargetRegisterClass *RC = MRI.getRegClass(R);
    return SIRegisterInfo::isVGPRClass(RC) ? TRI->getRegSizeInBits(*RC) / 32
                                           : 0;
  };
  SmallDenseSet<Register, 64> Touched;
  for (SUnit &SU : DAG.SUnits) {
    const MachineInstr &MI = *SU.getInstr();
    std::optional<unsigned> Id = getId(MI);
    if (!Id)
      continue;
    unsigned VDef = 0;
    for (const MachineOperand &MO : MI.all_defs())
      if (MO.getReg().isVirtual())
        if (const TargetRegisterClass *RC = MRI.getRegClass(MO.getReg());
            SIRegisterInfo::isVGPRClass(RC))
          VDef += TRI->getRegSizeInBits(*RC) / 32;
    json::Object N{{"id", *Id},
                   {"off", [&]() -> int64_t {
                      auto It = S.IdPos.find(*Id);
                      return It == S.IdPos.end() ? -1 : int64_t(It->second.second);
                    }()},
                   {"op", DAG.TII->getName(MI.getOpcode())},
                   {"class", classify(MI)},
                   {"lat", SU.Latency},
                   {"pos", SU.NodeNum},
                   {"vdef", VDef}};
    json::Array RDefs, RUses;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || MO.isDebug())
        continue;
      if (unsigned W = VGPRWidth(MO.getReg())) {
        Touched.insert(MO.getReg());
        (MO.isDef() ? RDefs : RUses)
            .push_back(json::Array{MO.getReg().virtRegIndex(), W});
      }
    }
    if (!RDefs.empty())
      N["rdefs"] = std::move(RDefs);
    if (!RUses.empty())
      N["ruses"] = std::move(RUses);
    if (const DILocation *L = MI.getDebugLoc().get()) {
      const DILocation *Outer = L;
      while (Outer->getInlinedAt())
        Outer = Outer->getInlinedAt();
      N["loc"] = locString(Outer);
      if (Outer != L)
        N["inner"] = locString(L);
    }
    Nodes.push_back(std::move(N));
    for (const SDep &P : SU.Preds) {
      const SUnit *Pred = P.getSUnit();
      if (Pred->isBoundaryNode())
        continue;
      if (std::optional<unsigned> PId = getId(*Pred->getInstr()))
        Edges.push_back(json::Object{{"from", *PId},
                                     {"to", *Id},
                                     {"kind", depKind(P)},
                                     {"lat", P.getLatency()}});
    }
  }
  json::Array LiveOut;
  if (LiveIntervals *LIS = PostRA ? nullptr : DAG.getLIS()) {
    MachineBasicBlock *BB = DAG.begin()->getParent();
    MachineBasicBlock::iterator E =
        skipDebugInstructionsForward(DAG.end(), BB->end());
    SlotIndex EndIdx = E == BB->end()
                           ? LIS->getMBBEndIdx(BB).getPrevSlot()
                           : LIS->getInstructionIndex(*E).getBaseIndex();
    for (Register R : Touched)
      if (LIS->hasInterval(R) && LIS->getInterval(R).liveAt(EndIdx))
        LiveOut.push_back(R.virtRegIndex());
  }
  // Signature and size of the region as the ledger names it: the one from the
  // function's region table that holds this DAG's first instruction.
  std::string Sig;
  unsigned SigN = 0;
  if (auto It = S.IdPos.find(Key); It != S.IdPos.end()) {
    Sig = S.Table[It->second.first].Sig;
    SigN = S.Table[It->second.first].NumInsts;
  }
  json::Object Rec{{"fn", S.Name},
                   {"fn_hash", S.NameHash},
                   {"sig", Sig},
                   {"n", SigN},
                   {"block", DAG.begin()->getParent()->getNumber()},
                   {"key", Key},
                   {"hash", Hash},
                   {"post_ra", PostRA},
                   {"live_out", std::move(LiveOut)},
                   {"nodes", std::move(Nodes)},
                   {"edges", std::move(Edges)}};
  OS << json::Value(std::move(Rec)) << "\n";
}

void SchedLedger::applyRegion(ScheduleDAGMI &DAG, RegionEntry &R,
                              const IdMap &Ids, StringRef Hash, bool PostRA) {
  if (R.Stale)
    return;
  // PSCHED_NO_POSTRA=1: apply the ledger before register allocation only and
  // leave the post-RA scheduler free (it forms the dual-issue pairs).
  static const bool NoPostRA = std::getenv("PSCHED_NO_POSTRA") != nullptr;
  if (PostRA && NoPostRA)
    return;
  unsigned Present = 0, Total = 0;
  auto Count = [&](unsigned Id) {
    ++Total;
    Present += Ids.count(Id);
  };
  for (auto &[Id, Lat] : R.Latency)
    Count(Id);
  for (const std::vector<unsigned> &Chain : R.Order)
    for (unsigned Id : Chain)
      Count(Id);
  if (!Present)
    return;

  if (!PostRA) {
    // The pre-RA region is the unit of ownership: every id must be here and
    // the opcode hash must match what the ledger was derived from.
    // Staleness is decided the first time the region is seen. Later
    // rescheduling stages may have rematerialised or removed instructions, which
    // changes the hash without making the entry wrong; there the entry keeps
    // applying to whichever of its instructions are still present.
    if (!R.Validated &&
        (Present != Total || (!R.Hash.empty() && R.Hash != Hash))) {
      R.Stale = true;
      errs() << "psched: warning: stale region " << R.Hash << " (" << R.Origin
             << ") in " << DAG.MF.getName() << ": "
             << (Present != Total ? "instruction ids missing" : "hash mismatch")
             << "; entry ignored\n";
      return;
    }
    if (!R.Validated && std::getenv("PSCHED_VERBOSE")) {
      unsigned Edges = 0;
      for (const std::vector<unsigned> &Chain : R.Order)
        Edges += Chain.size() - 1;
      errs() << "psched: " << DAG.MF.getName() << " <- region " << Hash << ": "
             << R.Latency.size() << " latency override(s), " << Edges
             << " order edge(s) (" << R.Origin << ")\n";
    }
    R.Validated = true;
  } else if (!R.Validated) {
    return;
  }

  for (auto &[Id, Lat] : R.Latency) {
    auto It = Ids.find(Id);
    if (It == Ids.end())
      continue;
    SUnit *SU = It->second;
    for (SDep &Succ : SU->Succs) {
      if (Succ.getKind() != SDep::Data)
        continue;
      SUnit *User = Succ.getSUnit();
      Succ.setLatency(Lat);
      for (SDep &Pred : User->Preds)
        if (Pred.getSUnit() == SU && Pred.getKind() == SDep::Data)
          Pred.setLatency(Lat);
      User->setDepthDirty();
    }
    SU->setHeightDirty();
  }

  // After register allocation the VOPD pairing mutation has already clustered
  // pairs of instructions that can be issued as one dual instruction. An order
  // edge that forces something between the two members of such a pair splits
  // it (one extra instruction per pair), so it is skipped. A ledger entry
  // cannot ask for that; PSCHED_NO_PAIR_GUARD=1 turns the guard off.
  static const bool PairGuard = !std::getenv("PSCHED_NO_PAIR_GUARD");
  // Pairs from this DAG's own nodes (the cluster list itself may hold entries
  // of earlier regions).
  SmallVector<std::pair<SUnit *, SUnit *>, 32> Pairs;
  if (PostRA && PairGuard) {
    DenseMap<unsigned, SmallVector<SUnit *, 2>> ByCluster;
    for (SUnit &SU : DAG.SUnits)
      if (SU.isClustered())
        ByCluster[SU.ParentClusterIdx].push_back(&SU);
    for (auto &[Idx, Members] : ByCluster)
      if (Members.size() == 2)
        Pairs.emplace_back(Members[0]->NodeNum < Members[1]->NodeNum ? Members[0] : Members[1],
                           Members[0]->NodeNum < Members[1]->NodeNum ? Members[1] : Members[0]);
  }
  auto SplitsPair = [&](SUnit *Pred, SUnit *Succ) {
    for (auto [P, Q] : Pairs) {
      // the pair is P then Q: Pred after P and Succ before Q puts both between
      bool AfterP = Pred == P || DAG.IsReachable(Pred, P);
      bool BeforeQ = Succ == Q || DAG.IsReachable(Q, Succ);
      if (AfterP && BeforeQ && !(Pred == P && Succ == Q))
        return true;
    }
    return false;
  };

  for (const std::vector<unsigned> &Chain : R.Order)
    for (unsigned I = 0; I + 1 < Chain.size(); ++I) {
      auto A = Ids.find(Chain[I]), B = Ids.find(Chain[I + 1]);
      if (A == Ids.end() || B == Ids.end())
        continue;
      if (SplitsPair(A->second, B->second)) {
        if (R.PairGuarded++ == 0 && std::getenv("PSCHED_VERBOSE"))
          errs() << "psched: " << DAG.MF.getName() << ": order " << Chain[I] << " -> " << Chain[I + 1]
                 << " (" << R.Origin << ") would split a dual-issue pair; skipped (further ones counted silently)\n";
        continue;
      }
      if (DAG.canAddEdge(B->second, A->second)) {
        DAG.addEdge(B->second, SDep(A->second, SDep::Artificial));
      } else if (!R.CycleWarned) {
        R.CycleWarned = true;
        errs() << "psched: warning: order " << Chain[I] << " -> " << Chain[I + 1]
               << " (" << R.Origin << ") contradicts a dependency in "
               << DAG.MF.getName() << "; edge skipped\n";
      }
    }
}

void SchedLedger::apply(ScheduleDAGInstrs *DAGInstrs) {
  auto &DAG = *static_cast<ScheduleDAGMI *>(DAGInstrs);
  FunctionState &S = enterFunction(DAG.MF);
  bool PostRA = DAG.MF.getProperties().hasNoVRegs();
  if (!PostRA) {
    assignIds(DAG.MF, S);
    if (!S.Templates.empty() || !envOr(DumpDirOpt, "PSCHED_DUMP_DIR").empty())
      buildTableAndBind(DAG.MF, S);
  }

  IdMap Ids;
  std::vector<std::pair<unsigned, unsigned>> Opcodes;
  for (SUnit &SU : DAG.SUnits)
    if (std::optional<unsigned> Id = getId(*SU.getInstr())) {
      Ids[*Id] = &SU;
      Opcodes.emplace_back(*Id, SU.getInstr()->getOpcode());
    }
  if (Ids.empty())
    return;
  llvm::sort(Opcodes);
  uint32_t H = 2166136261u;
  for (auto &[Id, Opc] : Opcodes)
    H = fnv1a(DAG.TII->getName(Opc), H);
  std::string Hash = utohexstr(H, /*LowerCase=*/true, 8);

  dump(DAG, S, Ids, Opcodes.front().first, Hash, PostRA);
  for (RegionEntry *R : S.Regions)
    applyRegion(DAG, *R, Ids, Hash, PostRA);
}

} // namespace

void llvm::AMDGPU::addSchedLedgerDAGMutation(ScheduleDAGMI *DAG,
                                             MachineFunction *MF) {
  FunctionState &S = enterFunction(*MF);
  if (S.Regions.empty() && S.Templates.empty() &&
      envOr(DumpDirOpt, "PSCHED_DUMP_DIR").empty())
    return;
  DAG->addMutation(std::make_unique<SchedLedger>());
}

void llvm::AMDGPU::reportRegionPressure(MachineFunction &MF,
                                        const MachineInstr *FirstMI,
                                        const MachineInstr *EndMI,
                                        StringRef Stage, unsigned VGPRBefore,
                                        unsigned VGPRAfter, unsigned VGPRKept,
                                        unsigned SGPRBefore, unsigned SGPRAfter,
                                        unsigned SGPRKept) {
  std::string Dir = envOr(DumpDirOpt, "PSCHED_DUMP_DIR");
  if (Dir.empty() || !FirstMI || !dumpWanted(MF.getName()))
    return;
  FunctionState &S = enterFunction(MF);
  unsigned Key = ~0u, Num = 0;
  for (const MachineInstr *MI = FirstMI; MI && MI != EndMI;
       MI = MI->getNextNode()) {
    if (MI->isDebugInstr())
      continue;
    ++Num;
    if (std::optional<unsigned> Id = getId(*MI))
      Key = std::min(Key, *Id);
  }
  if (Key == ~0u)
    return;
  auto Bypass = sys::sandbox::scopedDisable();
  sys::fs::create_directories(Dir);
  SmallString<256> Path(Dir);
  sys::path::append(Path, S.Name + ".pressure.jsonl");
  std::error_code EC;
  raw_fd_ostream OS(Path, EC,
                    S.PressureStarted ? sys::fs::OF_Append : sys::fs::OF_None);
  if (EC)
    return;
  S.PressureStarted = true;
  OS << json::Value(json::Object{{"key", Key},
                                 {"nodes", Num},
                                 {"stage", Stage},
                                 {"vgpr_before", VGPRBefore},
                                 {"vgpr_after", VGPRAfter},
                                 {"vgpr_kept", VGPRKept},
                                 {"sgpr_before", SGPRBefore},
                                 {"sgpr_after", SGPRAfter},
                                 {"sgpr_kept", SGPRKept}})
     << "\n";
}
