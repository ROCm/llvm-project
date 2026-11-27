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
/// Instructions are named by a per-function id assigned in program order
/// before the first scheduling region is built. The id is carried on the
/// instruction as !pcsections metadata, which makes AsmPrinter emit an
/// (address, function hash, id) table in section "psched_ids". The same
/// mutation can dump each region's DAG as one JSON line for offline analysis.
///
/// Inputs (command line option, else environment variable):
///   -amdgpu-sched-ledger / PSCHED_LEDGER         colon-separated files or dirs
///   -amdgpu-sched-ledger-dump / PSCHED_DUMP_DIR  directory for <function>.jsonl
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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

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
  std::string Hash;
  std::string Origin;
  std::vector<std::pair<unsigned, unsigned>> Latency;
  std::vector<std::vector<unsigned>> Order;
  bool Validated = false;
  bool Stale = false;
  bool CycleWarned = false;
};

struct FunctionEntry {
  std::string Match;
  std::string Source;
  std::vector<RegionEntry> Regions;
};

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
      if (const json::Array *L = R->getArray("latency"))
        for (const json::Value &LV : *L)
          if (const json::Object *LO = LV.getAsObject())
            if (auto Id = LO->getInteger("id"))
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

/// All region-level ledger entries, loaded once per process.
std::vector<FunctionEntry> &getLedger() {
  static std::vector<FunctionEntry> Ledger;
  static bool Loaded = false;
  if (Loaded)
    return Ledger;
  Loaded = true;
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

  void report() {
    for (RegionEntry *R : Regions)
      if (!R->Validated && !R->Stale)
        errs() << "psched: warning: region " << R->Hash << " (" << R->Origin
               << ") matched nothing in " << Name << " (stale ledger?)\n";
    Regions.clear();
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
      S.Regions.push_back(&R);
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
  unsigned Id = 0;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr() || MI.isPHI() || MI.getPCSections())
        continue;
      Metadata *Aux = MDNode::get(
          Ctx, {Hash, ConstantAsMetadata::get(ConstantInt::get(I32, Id++))});
      MI.setPCSections(MF, MDNode::get(Ctx, {Sec, Aux}));
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
  if (Dir.empty() || !S.Dumped.insert(Key | (PostRA ? 0x80000000u : 0)).second)
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
  json::Object Rec{{"fn", S.Name},
                   {"fn_hash", S.NameHash},
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

  for (const std::vector<unsigned> &Chain : R.Order)
    for (unsigned I = 0; I + 1 < Chain.size(); ++I) {
      auto A = Ids.find(Chain[I]), B = Ids.find(Chain[I + 1]);
      if (A == Ids.end() || B == Ids.end())
        continue;
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
  if (!PostRA)
    assignIds(DAG.MF, S);

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
  if (S.Regions.empty() && envOr(DumpDirOpt, "PSCHED_DUMP_DIR").empty())
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
  if (Dir.empty() || !FirstMI)
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
