//===- SummaryCallGraph.cpp - Call graph of a summary index ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SummaryCallGraph.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

AnalysisKey SummaryCallGraphAnalysis::Key;
AnalysisKey SummaryAddressTakenAnalysis::Key;

/// Returns the copy of \p VI that the link uses, or null if no copy or more
/// than one copy is in use. A copy is in use if it is live, and it is local,
/// promoted, or prevailing. An available_externally copy is never in use.
static GlobalValueSummary *getPrevailingCopy(
    ValueInfo VI, const ModuleSummaryIndex &Index,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        IsPrevailing) {
  GlobalValueSummary *Prevailing = nullptr;
  for (const auto &S : VI.getSummaryList()) {
    if (!Index.isGlobalValueLive(S.get()))
      continue;
    GlobalValue::LinkageTypes Linkage = S->linkage();
    if (GlobalValue::isAvailableExternallyLinkage(Linkage))
      continue;
    // The link does not resolve local symbols, so a local copy is in use. A
    // promoted copy was local before the thin link.
    if (!GlobalValue::isLocalLinkage(Linkage) && !S->wasPromoted() &&
        !IsPrevailing(VI.getGUID(), S.get()))
      continue;
    // More than one copy is in use only if the GUIDs of local values collide.
    // A call can then reach any of them.
    if (Prevailing)
      return nullptr;
    Prevailing = S.get();
  }
  return Prevailing;
}

/// Returns the GUIDs of the values that the lists llvm.global_ctors,
/// llvm.global_dtors, llvm.used, and llvm.compiler.used of the modules refer
/// to, directly or through other variables. Code outside the summaries can
/// call the functions among them, also if they are not preserved symbols: the
/// runtime calls the constructors and the destructors, and the users of a used
/// value are not in the summaries.
static DenseSet<GlobalValue::GUID>
getRuntimeEntries(const ModuleSummaryIndex &Index) {
  DenseSet<GlobalValue::GUID> Entries;
  SmallVector<ValueInfo> Worklist;
  for (StringRef Name : {"llvm.global_ctors", "llvm.global_dtors", "llvm.used",
                         "llvm.compiler.used"})
    if (ValueInfo VI = Index.getValueInfo(
            GlobalValue::getGUIDAssumingExternalLinkage(Name)))
      Worklist.push_back(VI);
  while (!Worklist.empty()) {
    ValueInfo VI = Worklist.pop_back_val();
    for (const auto &S : VI.getSummaryList())
      if (isa<GlobalVarSummary>(S.get()) && Index.isGlobalValueLive(S.get()))
        for (ValueInfo Ref : S->refs())
          if (Entries.insert(Ref.getGUID()).second)
            Worklist.push_back(Ref);
  }
  return Entries;
}

SummaryCallGraph::Node *SummaryCallGraph::createNode(ValueInfo VI) {
  Node *N = new (Allocator.Allocate()) Node(VI);
  Nodes.push_back(N);
  [[maybe_unused]] bool Inserted = NodeMap.try_emplace(VI.getGUID(), N).second;
  assert(Inserted && "Expected one node for each GUID");
  return N;
}

void SummaryCallGraph::computeSCCs() {
  // Start from a root that calls every node, so that the walk reaches all
  // nodes. Nothing calls the root, so it is alone in the last SCC.
  Node Root((ValueInfo()));
  Root.Callees.push_back(Edge(ExternalCallingNode, CallEdge));
  for (Node *N : Nodes)
    Root.Callees.push_back(Edge(N, CallEdge));
  Root.Callees.push_back(Edge(CallsExternalNode, CallEdge));

  // Reserve the space first, so that the SCCs stay valid.
  SCCNodes.reserve(Nodes.size() + 2);
  for (scc_iterator<Node *> I = scc_begin(&Root); !I.isAtEnd(); ++I) {
    const std::vector<Node *> &Component = *I;
    if (Component.front() == &Root)
      continue;
    size_t Begin = SCCNodes.size();
    llvm::append_range(SCCNodes, Component);
    SCCs.push_back(ArrayRef(SCCNodes).drop_front(Begin));
  }
  assert(SCCNodes.size() == Nodes.size() + 2 &&
         "Expected each node in one SCC");
}

SummaryCallGraph::SummaryCallGraph(
    ModuleSummaryIndex &Index, const ThinLinkInfoAnalysis::Result &Info,
    const DenseSet<GlobalValue::GUID> &AddressTaken, bool IncludeIndirectCalls)
    : IncludeIndirectCalls(IncludeIndirectCalls) {
  ExternalCallingNode = new (Allocator.Allocate()) Node(ValueInfo());
  CallsExternalNode = new (Allocator.Allocate()) Node(ValueInfo());

  // Create a node for each value with a live function or alias summary.
  for (const auto &Entry : Index) {
    ValueInfo VI = Index.getValueInfo(Entry);
    SmallVector<FunctionSummary *, 1> Copies;
    bool HasAlias = false;
    for (const auto &S : VI.getSummaryList()) {
      if (!Index.isGlobalValueLive(S.get()))
        continue;
      if (auto *FS = dyn_cast<FunctionSummary>(S.get()))
        Copies.push_back(FS);
      else
        HasAlias |= isa<AliasSummary>(S.get());
    }
    if (Copies.empty() && !HasAlias)
      continue;
    Node *N = createNode(VI);
    N->Copies = std::move(Copies);
    GlobalValueSummary *S = getPrevailingCopy(VI, Index, Info.IsPrevailing);
    if (S && !isa<GlobalVarSummary>(S))
      N->Prevailing = S;
  }

  // The graph knows all callers and all callees of the indirect calls only if
  // it includes them and the link assumes a closed world. Otherwise, the
  // special nodes also stand for the indirect calls.
  const bool KnowsIndirectCalls = IncludeIndirectCalls && Info.ClosedWorld;

  // The possible targets of the indirect calls with each call type ID: the
  // functions whose address can be taken and that have the ID.
  DenseMap<GlobalValue::GUID, SmallVector<Node *, 4>> Targets;
  for (Node *N : Nodes) {
    if (!IncludeIndirectCalls || !AddressTaken.contains(N->VI.getGUID()))
      continue;
    SmallVector<GlobalValue::GUID, 1> Ids;
    for (FunctionSummary *FS : N->Copies)
      if (FS->callTypeId() && !is_contained(Ids, FS->callTypeId()))
        Ids.push_back(FS->callTypeId());
    for (GlobalValue::GUID Id : Ids)
      Targets[Id].push_back(N);
  }

  // Returns the node of a callee. A callee without a function or alias summary
  // gets a node, which calls the calls external node.
  auto GetNode = [&](ValueInfo VI) {
    if (Node *N = NodeMap.lookup(VI.getGUID()))
      return N;
    return createNode(VI);
  };

  auto SetCallees = [](Node *N, const MapVector<Node *, unsigned> &Edges) {
    for (const auto &[Callee, Kinds] : Edges)
      N->Callees.push_back(Edge(Callee, Kinds));
  };

  // Add the edges to the callees. The loop also visits the nodes that it
  // creates.
  MapVector<Node *, unsigned> Edges;
  for (size_t I = 0; I != Nodes.size(); ++I) {
    Node *N = Nodes[I];
    Edges.clear();
    // The thin link does not know the code that a call to the value runs.
    if (!N->Prevailing)
      Edges[CallsExternalNode] |= CallEdge;
    for (FunctionSummary *FS : N->Copies) {
      for (const FunctionSummary::EdgeTy &E : FS->calls())
        Edges[GetNode(E.first)] |= CallEdge;
      if (IncludeIndirectCalls)
        for (GlobalValue::GUID Id : FS->indirectCallTypeIds()) {
          auto It = Targets.find(Id);
          if (It != Targets.end())
            for (Node *Target : It->second)
              Edges[Target] |= IndirectEdge;
        }
      // The summary does not tell if each call with an unknown callee is an
      // indirect call with a call type ID. A call to inline assembly or to an
      // ifunc has no ID.
      if (FS->fflags().HasUnknownCall ||
          (!KnowsIndirectCalls && !FS->indirectCallTypeIds().empty()))
        Edges[CallsExternalNode] |= IndirectEdge;
    }
    for (const auto &S : N->VI.getSummaryList())
      if (auto *AS = dyn_cast<AliasSummary>(S.get());
          AS && AS->hasAliasee() && Index.isGlobalValueLive(AS))
        Edges[GetNode(AS->getAliaseeVI())] |= CallEdge;
    SetCallees(N, Edges);
  }

  // Code outside the summaries can call the preserved symbols and the entries
  // of the runtime. Unless the graph knows the indirect calls, the external
  // calling node also calls each function whose address can be taken.
  auto ByGUID = [](const Node *A, const Node *B) {
    return A->VI.getGUID() < B->VI.getGUID();
  };
  SmallVector<Node *> Preserved;
  for (GlobalValue::GUID GUID : Info.GUIDPreservedSymbols)
    if (Node *N = NodeMap.lookup(GUID))
      Preserved.push_back(N);
  llvm::sort(Nodes, ByGUID);
  llvm::sort(Preserved, ByGUID);
  Edges.clear();
  for (Node *N : Preserved)
    Edges[N] |= CallEdge;
  DenseSet<GlobalValue::GUID> Entries = getRuntimeEntries(Index);
  for (Node *N : Nodes)
    if (Entries.contains(N->VI.getGUID()) ||
        (!KnowsIndirectCalls && AddressTaken.contains(N->VI.getGUID())))
      Edges[N] |= IndirectEdge;
  SetCallees(ExternalCallingNode, Edges);

  // Add the edges from the callers.
  auto AddCallers = [](Node *N) {
    for (const Edge &E : N->Callees)
      E.N->Callers.push_back(Edge(N, E.Kinds));
  };
  AddCallers(ExternalCallingNode);
  for (Node *N : Nodes)
    AddCallers(N);

  computeSCCs();
}

/// Prints the name of \p N.
static void printName(raw_ostream &OS, const SummaryCallGraph &CG,
                      const SummaryCallGraph::Node *N) {
  if (N == CG.getExternalCallingNode())
    OS << "<external callers>";
  else if (N == CG.getCallsExternalNode())
    OS << "<external callees>";
  else
    OS << N->getValueInfo();
}

void SummaryCallGraph::print(raw_ostream &OS) const {
  auto PrintNode = [&](const Node *N) {
    OS << "  ";
    printName(OS, *this, N);
    if (isa_and_nonnull<AliasSummary>(N->Prevailing))
      OS << " [alias]";
    else if (N->Prevailing)
      OS << " [" << N->Prevailing->modulePath() << "]";
    else if (N != ExternalCallingNode && N != CallsExternalNode)
      OS << " [unknown]";
    SmallVector<StringRef, 2> OtherCopies;
    for (const FunctionSummary *FS : N->Copies)
      if (FS != N->Prevailing)
        OtherCopies.push_back(FS->modulePath());
    if (!OtherCopies.empty())
      OS << " [other copies: " << join(OtherCopies, ", ") << "]";
    OS << "\n";
    for (const Edge &E : N->Callees) {
      OS << "    calls ";
      printName(OS, *this, E.getNode());
      if (!E.isCall())
        OS << " (indirect)";
      else if (E.isIndirect())
        OS << " (also indirect)";
      OS << "\n";
    }
  };

  OS << "Summary call graph:\n";
  PrintNode(ExternalCallingNode);
  for (const Node *N : Nodes)
    PrintNode(N);
  PrintNode(CallsExternalNode);
  OS << "SCCs in bottom-up order:\n";
  for (SCC C : SCCs) {
    OS << "  [";
    ListSeparator LS;
    for (const Node *N : C) {
      OS << LS;
      printName(OS, *this, N);
    }
    OS << "]\n";
  }
}

const SummaryCallGraph &
SummaryCallGraphAnalysis::Result::getGraph(bool IncludeIndirectCalls) {
  std::unique_ptr<SummaryCallGraph> &CG =
      IncludeIndirectCalls ? GraphWithIndirectCalls : Graph;
  if (!CG)
    CG = std::make_unique<SummaryCallGraph>(Index, Info, AddressTaken,
                                            IncludeIndirectCalls);
  return *CG;
}

bool SummaryCallGraphAnalysis::Result::invalidate(
    ModuleSummaryIndex &Index, const PreservedAnalyses &PA,
    SummaryAnalysisManager::Invalidator &Inv) {
  // The result also refers to the set of the values whose address can be
  // taken.
  auto PAC = PA.getChecker<SummaryCallGraphAnalysis>();
  return !(PAC.preserved() ||
           PAC.preservedSet<AllAnalysesOn<ModuleSummaryIndex>>()) ||
         Inv.invalidate<SummaryAddressTakenAnalysis>(Index, PA);
}

SummaryCallGraphAnalysis::Result
SummaryCallGraphAnalysis::run(ModuleSummaryIndex &Index,
                              SummaryAnalysisManager &AM) {
  return Result(Index, AM.getResult<ThinLinkInfoAnalysis>(Index),
                AM.getResult<SummaryAddressTakenAnalysis>(Index));
}

SummaryAddressTakenAnalysis::Result
SummaryAddressTakenAnalysis::run(ModuleSummaryIndex &Index,
                                 SummaryAnalysisManager &AM) {
  auto &Info = AM.getResult<ThinLinkInfoAnalysis>(Index);
  // Code outside the summaries can take the address of a preserved symbol.
  // Code in the summaries refers to a value for each use other than as the
  // callee of a direct call.
  Result AddressTaken(Info.GUIDPreservedSymbols.begin(),
                      Info.GUIDPreservedSymbols.end());
  for (const auto &[GUID, VInfo] : Index)
    for (const auto &S : VInfo.getSummaryList())
      if (Index.isGlobalValueLive(S.get()))
        for (ValueInfo Ref : S->refs())
          AddressTaken.insert(Ref.getGUID());

  // The address of an alias is the address of its aliasee. The GUID of an
  // aliasee can be an alias in another module, so also visit the GUIDs that
  // this step adds.
  SmallVector<GlobalValue::GUID> Worklist;
  auto AddAliasees = [&](ValueInfo VI) {
    for (const auto &S : VI.getSummaryList()) {
      auto *AS = dyn_cast<AliasSummary>(S.get());
      if (AS && AS->hasAliasee() && Index.isGlobalValueLive(AS) &&
          AddressTaken.insert(AS->getAliaseeGUID()).second)
        Worklist.push_back(AS->getAliaseeGUID());
    }
  };
  for (const auto &Entry : Index)
    if (AddressTaken.contains(Entry.first))
      AddAliasees(Index.getValueInfo(Entry));
  while (!Worklist.empty())
    AddAliasees(Index.getValueInfo(Worklist.pop_back_val()));
  return AddressTaken;
}

PreservedAnalyses SummaryCallGraphPrinterPass::run(ModuleSummaryIndex &Index,
                                                   SummaryAnalysisManager &AM) {
  AM.getResult<SummaryCallGraphAnalysis>(Index)
      .getGraph(IncludeIndirectCalls)
      .print(OS);
  return PreservedAnalyses::all();
}
