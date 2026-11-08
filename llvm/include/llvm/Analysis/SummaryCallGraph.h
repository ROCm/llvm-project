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
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/SummaryPassManager.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include <memory>
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
/// outside the summaries can call, and the calls external node is a callee of
/// each node whose callees are not all known.
///
/// The edges of a node come from all its live copies. The copy that prevails
/// runs when code calls the symbol, and a module can still inline its own copy
/// that does not prevail. An alias calls its aliasee.
///
/// The external calling node calls each preserved symbol, and each function
/// that a module lists as a constructor, a destructor, or a used value
/// (llvm.global_ctors, llvm.global_dtors, llvm.used, llvm.compiler.used),
/// also through a variable. When a pass asks SummaryCallGraphAnalysis for the
/// graph, it selects whether the graph includes the indirect calls:
/// - Without them, an indirect call calls the calls external node, and the
///   external calling node also calls each function whose address can be
///   taken. As in the IR call graph, the special nodes stand for the callees
///   and the callers that the graph does not know.
/// - With them, an indirect call calls each of its possible targets: a
///   function whose address can be taken and that has the call type ID of the
///   call. A summary without call type IDs (see
///   FunctionSummary::hasCallTypeIds()) can have any ID. These edges can be
///   many: for each call type ID, the number of callers times the number of
///   targets. A pass that needs them accepts their cost. If the link does not
///   assume a closed world (see
///   ThinLinkInfoAnalysis), the graph also has the edges of the other form:
///   code outside the summaries can call a function whose address escapes to
///   it, and an indirect call can reach code outside the summaries.
///
/// An edge has kinds. A call edge comes from FunctionSummary::calls(): a direct
/// call, or a target that a profile or !callees metadata gives for an indirect
/// call. An indirect edge stands for calls that the summary does not name: an
/// edge to a possible target of an indirect call, an edge from an indirect
/// call or an unknown call to the calls external node, or an edge from the
/// external calling node to a function whose address can be taken.
class SummaryCallGraph {
public:
  class Node;

  /// The kinds of an edge.
  enum EdgeKind : unsigned {
    /// A call edge in the summary.
    CallEdge = 1 << 0,
    /// A call that the summary does not name.
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
    /// the kinds of all calls to it.
    ArrayRef<Edge> callees() const { return Callees; }

    /// Returns the edges from the callers, like callees().
    ArrayRef<Edge> callers() const { return Callers; }

  private:
    friend class SummaryCallGraph;

    explicit Node(ValueInfo VI) : VI(VI) {}

    ValueInfo VI;
    GlobalValueSummary *Prevailing = nullptr;
    SmallVector<FunctionSummary *, 1> Copies;
    SmallVector<Edge, 0> Callees;
    SmallVector<Edge, 0> Callers;
  };

  /// A strongly connected component: functions that can call each other.
  using SCC = ArrayRef<Node *>;

  /// Builds the call graph of \p Index. \p Info holds the facts of the link.
  /// \p AddressTaken holds the values whose address can be taken (see
  /// SummaryAddressTakenAnalysis). If \p IncludeIndirectCalls is true, the
  /// graph has the edges from the indirect calls to their possible targets.
  LLVM_ABI SummaryCallGraph(ModuleSummaryIndex &Index,
                            const ThinLinkInfoAnalysis::Result &Info,
                            const DenseSet<GlobalValue::GUID> &AddressTaken,
                            bool IncludeIndirectCalls);

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
  /// reverse for the top-down order.
  ArrayRef<SCC> sccs() const { return SCCs; }

  /// Returns true if the graph has the edges from the indirect calls to their
  /// possible targets.
  bool includesIndirectCalls() const { return IncludeIndirectCalls; }

  LLVM_ABI void print(raw_ostream &OS) const;

private:
  Node *createNode(ValueInfo VI);
  void computeSCCs();

  SpecificBumpPtrAllocator<Node> Allocator;
  Node *ExternalCallingNode;
  Node *CallsExternalNode;
  std::vector<Node *> Nodes;
  DenseMap<GlobalValue::GUID, Node *> NodeMap;
  // The nodes of all SCCs, in the order of the SCCs. The SCCs refer to it.
  std::vector<Node *> SCCNodes;
  std::vector<SCC> SCCs;
  bool IncludeIndirectCalls;
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

/// Gives the SummaryCallGraph of the index in the form that a pass selects.
/// Uses ThinLinkInfoAnalysis and SummaryAddressTakenAnalysis.
///
/// The graph reads these data of the index. A pass that changes them must not
/// preserve this analysis:
/// - the summaries of each value, and for each summary, whether it is live,
///   its linkage, and whether it was promoted. They give the copies and the
///   prevailing copy of a node.
/// - the calls of a function summary (FunctionSummary::calls()), its call type
///   IDs (FunctionSummary::getCallTypeIdInfo()), and its flag HasUnknownCall.
/// - the aliasee of an alias summary.
/// - the references of the summaries (GlobalValueSummary::refs()). They give
///   the values whose address can be taken (see SummaryAddressTakenAnalysis),
///   and the functions that the runtime can call, for example through
///   llvm.global_ctors.
/// The facts of ThinLinkInfoAnalysis do not change while passes run. The other
/// data of the index do not change the graph, for example the other function
/// flags, the parameter access ranges, and the entries of summary extensions.
/// A pass that changes only such data can preserve this analysis. The result is
/// also invalid if SummaryAddressTakenAnalysis is.
class SummaryCallGraphAnalysis
    : public AnalysisInfoMixin<SummaryCallGraphAnalysis> {
public:
  /// Builds each form of the graph when a pass first asks for it. A pass that
  /// does not need the indirect calls does not pay for their edges.
  class Result {
  public:
    Result(ModuleSummaryIndex &Index, const ThinLinkInfoAnalysis::Result &Info,
           const DenseSet<GlobalValue::GUID> &AddressTaken)
        : Index(Index), Info(Info), AddressTaken(AddressTaken) {}

    /// Returns the call graph of the index. If \p IncludeIndirectCalls is
    /// true, the graph has the edges from the indirect calls to their possible
    /// targets (see SummaryCallGraph).
    LLVM_ABI const SummaryCallGraph &getGraph(bool IncludeIndirectCalls);

    LLVM_ABI bool invalidate(ModuleSummaryIndex &Index,
                             const PreservedAnalyses &PA,
                             SummaryAnalysisManager::Invalidator &Inv);

  private:
    ModuleSummaryIndex &Index;
    const ThinLinkInfoAnalysis::Result &Info;
    const DenseSet<GlobalValue::GUID> &AddressTaken;
    std::unique_ptr<SummaryCallGraph> Graph;
    std::unique_ptr<SummaryCallGraph> GraphWithIndirectCalls;
  };

  LLVM_ABI Result run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);

private:
  friend AnalysisInfoMixin<SummaryCallGraphAnalysis>;
  LLVM_ABI static AnalysisKey Key;
};

/// Computes the GUIDs of the values whose address code can take: the values
/// that a live summary refers to, and the preserved symbols. For an alias in
/// the set, the set also holds its aliasee. A function in the set can be the
/// target of an indirect call. Uses ThinLinkInfoAnalysis. A pass that changes
/// whether a summary is live, the references of a summary, or the aliasee of an
/// alias must not preserve this analysis.
class SummaryAddressTakenAnalysis
    : public AnalysisInfoMixin<SummaryAddressTakenAnalysis> {
public:
  using Result = DenseSet<GlobalValue::GUID>;

  LLVM_ABI Result run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);

private:
  friend AnalysisInfoMixin<SummaryAddressTakenAnalysis>;
  LLVM_ABI static AnalysisKey Key;
};

/// Prints the SummaryCallGraph of the index. If \p IncludeIndirectCalls is
/// true, prints the graph that includes the indirect calls.
class SummaryCallGraphPrinterPass
    : public RequiredPassInfoMixin<SummaryCallGraphPrinterPass> {
  raw_ostream &OS;
  bool IncludeIndirectCalls;

public:
  SummaryCallGraphPrinterPass(raw_ostream &OS, bool IncludeIndirectCalls)
      : OS(OS), IncludeIndirectCalls(IncludeIndirectCalls) {}

  LLVM_ABI PreservedAnalyses run(ModuleSummaryIndex &Index,
                                 SummaryAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_SUMMARYCALLGRAPH_H
