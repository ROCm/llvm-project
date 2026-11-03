//===- SummaryCallGraph.h - Call graph of a summary index -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analyses of a combined summary index for the passes of the ThinLTO thin
// link: the call graph of the functions, and the set of values whose address
// can be taken.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SUMMARYCALLGRAPH_H
#define LLVM_ANALYSIS_SUMMARYCALLGRAPH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/SummaryPassManager.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include <vector>

namespace llvm {

class raw_ostream;

/// The call graph of a combined index, after the link resolved the symbols.
/// Code at the thin link walks calls with this graph, not with
/// FunctionSummary::calls().
///
/// A node is a value with a live function or alias summary, or a callee
/// without one. Like the IR call graph (CallGraph), the graph also has two
/// special nodes: the external calling node calls each function that code
/// outside the summaries can call (a preserved symbol), and the calls external
/// node is a callee of each node whose callees are not all known.
///
/// The edges of a node come from all its live copies. The copy that prevails
/// runs when code calls the symbol, and a module can still inline its own copy
/// that does not prevail. An alias calls its aliasee.
///
/// An edge has kinds. A call edge comes from FunctionSummary::calls(): a direct
/// call, or a target that a profile or !callees metadata gives for an indirect
/// call. An indirect edge goes to a possible target of an indirect call: a
/// function whose address can be taken and that has the call type ID of the
/// call. A walk can leave out the indirect edges.
///
/// The graph does not show calls through a function pointer that escapes to
/// code outside the summaries.
class SummaryCallGraph {
public:
  class Node;

  /// The kinds of an edge.
  enum EdgeKind : unsigned {
    /// A call edge in the summary.
    CallEdge = 1 << 0,
    /// A possible target of an indirect call, or a call with a callee that the
    /// summary does not know.
    IndirectEdge = 1 << 1,
  };

  /// An edge to a callee or from a caller.
  class Edge {
  public:
    Edge(Node *N, unsigned Kinds) : N(N), Kinds(Kinds) {}

    /// Returns the node at the other end: the callee or the caller.
    Node *getNode() const { return N; }

    bool isCall() const { return Kinds & CallEdge; }
    bool isIndirect() const { return Kinds & IndirectEdge; }

  private:
    friend class SummaryCallGraph;

    Node *N;
    unsigned Kinds;
  };

  class Node {
  public:
    /// Returns the value of the node. It is empty for the special nodes.
    ValueInfo getValueInfo() const { return VI; }

    /// Returns the copy that prevails: a function or an alias summary. It is
    /// null if the thin link does not know the code that a call to the value
    /// runs (for example, the definition is in a native object), and for the
    /// special nodes.
    GlobalValueSummary *getPrevailingCopy() const { return Prevailing; }

    /// Returns the copy that prevails if it is a function, or null.
    FunctionSummary *getSummary() const {
      return dyn_cast_or_null<FunctionSummary>(Prevailing);
    }

    /// Returns the live function summaries of the value: its copies in the
    /// modules.
    ArrayRef<FunctionSummary *> copies() const { return Copies; }

    /// Returns the edges to the callees. Each callee is in one edge, which has
    /// the kinds of all calls to it. If \p IncludeIndirect is false, returns
    /// only the edges that have the call kind.
    ArrayRef<Edge> callees(bool IncludeIndirect = true) const {
      return IncludeIndirect ? ArrayRef(Callees)
                             : ArrayRef(Callees).take_front(NumCallCallees);
    }

    /// Returns the edges from the callers, like callees().
    ArrayRef<Edge> callers(bool IncludeIndirect = true) const {
      return IncludeIndirect ? ArrayRef(Callers)
                             : ArrayRef(Callers).take_front(NumCallCallers);
    }

  private:
    friend class SummaryCallGraph;

    explicit Node(ValueInfo VI) : VI(VI) {}

    ValueInfo VI;
    GlobalValueSummary *Prevailing = nullptr;
    SmallVector<FunctionSummary *, 1> Copies;
    // The edges that have the call kind come first.
    SmallVector<Edge, 0> Callees;
    SmallVector<Edge, 0> Callers;
    unsigned NumCallCallees = 0;
    unsigned NumCallCallers = 0;
  };

  /// A strongly connected component: functions that can call each other.
  using SCC = ArrayRef<Node *>;

  /// Builds the call graph of \p Index. \p GUIDPreservedSymbols and
  /// \p IsPrevailing are the facts of the link (see ThinLinkInfoAnalysis).
  /// \p AddressTaken holds the values whose address can be taken (see
  /// SummaryAddressTakenAnalysis).
  LLVM_ABI SummaryCallGraph(
      ModuleSummaryIndex &Index,
      const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
      const DenseSet<GlobalValue::GUID> &AddressTaken,
      function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
          IsPrevailing);

  SummaryCallGraph(SummaryCallGraph &&) = default;

  /// Returns the node that calls each function that code outside the
  /// summaries can call.
  Node *getExternalCallingNode() const { return ExternalCallingNode; }

  /// Returns the node that each node with unknown callees calls.
  Node *getCallsExternalNode() const { return CallsExternalNode; }

  /// Returns the nodes of the values, in the order of their GUIDs. The special
  /// nodes are not in the list.
  ArrayRef<Node *> nodes() const { return Nodes; }

  /// Returns the node of the value with \p GUID, or null.
  Node *lookup(GlobalValue::GUID GUID) const { return NodeMap.lookup(GUID); }

  /// Returns the SCCs of all nodes, also the special nodes, in bottom-up
  /// order: each SCC comes before the SCCs that call into it. Iterate in
  /// reverse for the top-down order. If \p IncludeIndirect is false, returns
  /// the SCCs of the edges that have the call kind.
  ArrayRef<SCC> sccs(bool IncludeIndirect = true) const {
    return IncludeIndirect ? SCCs.SCCs : CallSCCs.SCCs;
  }

  LLVM_ABI void print(raw_ostream &OS) const;

private:
  struct SCCList {
    // The nodes of all SCCs, in the order of the SCCs. The SCCs refer to it.
    std::vector<Node *> Nodes;
    std::vector<SCC> SCCs;
  };

  Node *createNode(ValueInfo VI);
  template <bool IncludeIndirect> void computeSCCs(SCCList &List);

  SpecificBumpPtrAllocator<Node> Allocator;
  Node *ExternalCallingNode;
  Node *CallsExternalNode;
  std::vector<Node *> Nodes;
  DenseMap<GlobalValue::GUID, Node *> NodeMap;
  SCCList SCCs;
  SCCList CallSCCs;
};

/// Walks all edges of the graph.
template <> struct GraphTraits<SummaryCallGraph::Node *> {
  using NodeRef = SummaryCallGraph::Node *;

  static NodeRef getNode(const SummaryCallGraph::Edge &E) {
    return E.getNode();
  }
  using ChildIteratorType =
      mapped_iterator<ArrayRef<SummaryCallGraph::Edge>::iterator,
                      decltype(&getNode)>;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) {
    return map_iterator(N->callees().begin(), &getNode);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return map_iterator(N->callees().end(), &getNode);
  }
};

/// Computes the SummaryCallGraph of the index. Uses ThinLinkInfoAnalysis and
/// SummaryAddressTakenAnalysis.
class SummaryCallGraphAnalysis
    : public AnalysisInfoMixin<SummaryCallGraphAnalysis> {
public:
  using Result = SummaryCallGraph;

  LLVM_ABI Result run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);

private:
  friend AnalysisInfoMixin<SummaryCallGraphAnalysis>;
  LLVM_ABI static AnalysisKey Key;
};

/// Computes the GUIDs of the values whose address code can take: the values
/// that a live summary refers to, and the preserved symbols. For an alias in
/// the set, the set also holds its aliasee. A function in the set can be the
/// target of an indirect call. Uses ThinLinkInfoAnalysis.
class SummaryAddressTakenAnalysis
    : public AnalysisInfoMixin<SummaryAddressTakenAnalysis> {
public:
  using Result = DenseSet<GlobalValue::GUID>;

  LLVM_ABI Result run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);

private:
  friend AnalysisInfoMixin<SummaryAddressTakenAnalysis>;
  LLVM_ABI static AnalysisKey Key;
};

/// Prints the SummaryCallGraph of the index.
class SummaryCallGraphPrinterPass
    : public RequiredPassInfoMixin<SummaryCallGraphPrinterPass> {
  raw_ostream &OS;

public:
  explicit SummaryCallGraphPrinterPass(raw_ostream &OS) : OS(OS) {}

  LLVM_ABI PreservedAnalyses run(ModuleSummaryIndex &Index,
                                 SummaryAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_SUMMARYCALLGRAPH_H
