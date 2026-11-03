//===- SummaryCallGraph.h - Call graph of a summary index -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analyses of a combined summary index for the passes of the ThinLTO thin
// link: the call graph over the copies of the functions that the link uses,
// and the set of values whose address can be taken.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SUMMARYCALLGRAPH_H
#define LLVM_ANALYSIS_SUMMARYCALLGRAPH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/SummaryPassManager.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include <vector>

namespace llvm {

class raw_ostream;

/// A call graph over the function summaries of a combined index, after the
/// link resolved the symbols. Each node is a function. The summary of a node
/// is the copy of the function that the link uses (the prevailing copy): the
/// live copy that is local, promoted, or prevailing. A call to an alias is an
/// edge to the aliasee.
///
/// A node has no summary if the thin link does not know the code of the
/// function. For example, the function is defined in a native object, or two
/// local copies have the same GUID. Such a node has no callees, so passes must
/// treat it conservatively.
///
/// The edges are the direct calls in the summaries. Code outside the summaries
/// can call a function through its symbol (see Node::hasUnknownCallers()). The
/// graph does not show calls through a function pointer that escapes to such
/// code.
class SummaryCallGraph {
public:
  class Node {
  public:
    /// Returns the function. For a function without a summary, this is the
    /// callee of the calls.
    ValueInfo getValueInfo() const { return VI; }

    /// Returns the prevailing copy of the function, or null if the thin link
    /// does not know the code of the function.
    FunctionSummary *getSummary() const { return FS; }

    /// Returns the functions that this function calls directly. Each function
    /// is in the list one time.
    ArrayRef<Node *> callees() const { return Callees; }

    /// Returns the functions that call this function directly. Each function
    /// is in the list one time.
    ArrayRef<Node *> callers() const { return Callers; }

    /// Returns true if code outside the summaries can call this function: the
    /// function or an alias of it is a preserved symbol.
    bool hasUnknownCallers() const { return HasUnknownCallers; }

  private:
    friend class SummaryCallGraph;

    Node(ValueInfo VI, FunctionSummary *FS) : VI(VI), FS(FS) {}

    ValueInfo VI;
    FunctionSummary *FS;
    SmallVector<Node *, 0> Callees;
    SmallVector<Node *, 0> Callers;
    bool HasUnknownCallers = false;
  };

  /// A strongly connected component: functions that can call each other.
  using SCC = ArrayRef<Node *>;

  /// Builds the call graph of \p Index. \p GUIDPreservedSymbols and
  /// \p IsPrevailing are the facts of the link (see ThinLinkInfoAnalysis).
  LLVM_ABI SummaryCallGraph(
      ModuleSummaryIndex &Index,
      const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
      function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
          IsPrevailing);

  SummaryCallGraph(SummaryCallGraph &&) = default;

  /// Returns all nodes, in the order of their GUIDs.
  ArrayRef<Node *> nodes() const { return Nodes; }

  /// Returns the node of the function with \p GUID, or null. The graph has a
  /// node for each function with a prevailing copy, and for each function that
  /// such a function calls.
  Node *lookup(GlobalValue::GUID GUID) const { return NodeMap.lookup(GUID); }

  /// Returns the SCCs in bottom-up order: each SCC comes before the SCCs that
  /// call into it. Iterate in reverse for the top-down order.
  ArrayRef<SCC> sccs() const { return SCCs; }

  LLVM_ABI void print(raw_ostream &OS) const;

private:
  Node *createNode(ValueInfo VI, FunctionSummary *FS);

  SpecificBumpPtrAllocator<Node> Allocator;
  std::vector<Node *> Nodes;
  DenseMap<GlobalValue::GUID, Node *> NodeMap;
  // The nodes of all SCCs, in the order of sccs(). The SCCs refer to it.
  std::vector<Node *> SCCNodes;
  std::vector<SCC> SCCs;
};

template <> struct GraphTraits<SummaryCallGraph::Node *> {
  using NodeRef = SummaryCallGraph::Node *;
  using ChildIteratorType = ArrayRef<SummaryCallGraph::Node *>::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) {
    return N->callees().begin();
  }
  static ChildIteratorType child_end(NodeRef N) { return N->callees().end(); }
};

/// Computes the SummaryCallGraph of the index. Uses ThinLinkInfoAnalysis.
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
