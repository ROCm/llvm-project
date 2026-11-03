//===- SummaryCallGraph.cpp - Call graph of a summary index ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SummaryCallGraph.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
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

SummaryCallGraph::Node *SummaryCallGraph::createNode(ValueInfo VI,
                                                     FunctionSummary *FS) {
  Node *N = new (Allocator.Allocate()) Node(VI, FS);
  Nodes.push_back(N);
  [[maybe_unused]] bool Inserted = NodeMap.try_emplace(VI.getGUID(), N).second;
  assert(Inserted && "Expected one node for each GUID");
  return N;
}

SummaryCallGraph::SummaryCallGraph(
    ModuleSummaryIndex &Index,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        IsPrevailing) {
  // Create a node for each function with a prevailing copy. Record the aliasee
  // of each alias with a prevailing copy.
  DenseMap<GlobalValue::GUID, ValueInfo> Aliasees;
  for (const auto &Entry : Index) {
    ValueInfo VI = Index.getValueInfo(Entry);
    GlobalValueSummary *S = getPrevailingCopy(VI, Index, IsPrevailing);
    if (auto *FS = dyn_cast_or_null<FunctionSummary>(S))
      createNode(VI, FS);
    else if (auto *AS = dyn_cast_or_null<AliasSummary>(S);
             AS && AS->hasAliasee())
      Aliasees[VI.getGUID()] = AS->getAliaseeVI();
  }

  // Returns the value that a use of VI reaches. The aliasee in a summary is
  // not an alias, but its GUID can have an alias as the prevailing copy (for
  // example, a C++ constructor alias in another module). The walk stops in a
  // cycle of aliases, and the caller then finds no function.
  auto ResolveAliases = [&](ValueInfo VI) {
    for (size_t Steps = 0; Steps <= Aliasees.size(); ++Steps) {
      ValueInfo Aliasee = Aliasees.lookup(VI.getGUID());
      if (!Aliasee)
        break;
      VI = Aliasee;
    }
    return VI;
  };

  // Add the call edges. Only the nodes that exist now have summaries, so the
  // loop does not visit the nodes that it creates.
  SmallPtrSet<Node *, 8> Seen;
  for (size_t I = 0, E = Nodes.size(); I != E; ++I) {
    Node *Caller = Nodes[I];
    Seen.clear();
    for (const FunctionSummary::EdgeTy &Edge : Caller->FS->calls()) {
      ValueInfo CalleeVI = ResolveAliases(Edge.first);
      Node *Callee = NodeMap.lookup(CalleeVI.getGUID());
      // The thin link does not know the code of the callee.
      if (!Callee)
        Callee = createNode(CalleeVI, /*FS=*/nullptr);
      if (!Seen.insert(Callee).second)
        continue;
      Caller->Callees.push_back(Callee);
      Callee->Callers.push_back(Caller);
    }
  }

  // Code outside the summaries can call the preserved symbols.
  for (GlobalValue::GUID GUID : GUIDPreservedSymbols) {
    ValueInfo VI = Index.getValueInfo(GUID);
    if (!VI)
      continue;
    if (Node *N = NodeMap.lookup(ResolveAliases(VI).getGUID()))
      N->HasUnknownCallers = true;
  }

  llvm::sort(Nodes, [](const Node *A, const Node *B) {
    return A->VI.getGUID() < B->VI.getGUID();
  });

  // Find the SCCs. Start from a root that calls every node, so that the walk
  // reaches all nodes. Nothing calls the root, so it is alone in the last SCC.
  Node Root(ValueInfo(), /*FS=*/nullptr);
  Root.Callees.assign(Nodes.begin(), Nodes.end());
  // Reserve the space first, so that the SCCs stay valid.
  SCCNodes.reserve(Nodes.size());
  for (scc_iterator<Node *> I = scc_begin(&Root); !I.isAtEnd(); ++I) {
    const std::vector<Node *> &Component = *I;
    if (Component.front() == &Root)
      continue;
    size_t Begin = SCCNodes.size();
    llvm::append_range(SCCNodes, Component);
    SCCs.push_back(ArrayRef(SCCNodes).drop_front(Begin));
  }
  assert(SCCNodes.size() == Nodes.size() && "Expected each node in one SCC");
}

void SummaryCallGraph::print(raw_ostream &OS) const {
  OS << "Summary call graph:\n";
  for (const Node *N : Nodes) {
    OS << "  " << N->VI;
    if (N->FS)
      OS << " [" << N->FS->modulePath() << "]";
    else
      OS << " [no summary]";
    if (N->HasUnknownCallers)
      OS << " [unknown callers]";
    OS << "\n";
    for (const Node *Callee : N->Callees)
      OS << "    calls " << Callee->VI << "\n";
  }
  OS << "SCCs in bottom-up order:\n";
  for (SCC C : SCCs) {
    OS << "  [";
    ListSeparator LS;
    for (const Node *N : C)
      OS << LS << N->VI;
    OS << "]\n";
  }
}

SummaryCallGraph SummaryCallGraphAnalysis::run(ModuleSummaryIndex &Index,
                                               SummaryAnalysisManager &AM) {
  auto &Info = AM.getResult<ThinLinkInfoAnalysis>(Index);
  return SummaryCallGraph(Index, Info.GUIDPreservedSymbols, Info.IsPrevailing);
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
  AM.getResult<SummaryCallGraphAnalysis>(Index).print(OS);
  return PreservedAnalyses::all();
}
