//===- SummaryDataflow.h - Dataflow on the summary call graph ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A solver for dataflow problems on the call graph of a summary index
// (SummaryCallGraph), for the passes of the ThinLTO thin link. A top-down
// problem goes from the callers to the callees, for example a range of values
// that the kernels which reach a function give it. A bottom-up problem goes
// from the callees to the callers, for example a fact that is true if some
// callee uses a resource.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SUMMARYDATAFLOW_H
#define LLVM_ANALYSIS_SUMMARYDATAFLOW_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/SummaryCallGraph.h"
#include <utility>

namespace llvm {

/// The direction of a dataflow problem on the summary call graph.
enum class SummaryDataflowDirection {
  /// The input of a node joins the outputs of its callers.
  TopDown,
  /// The input of a node joins the outputs of its callees.
  BottomUp,
};

/// Solves a dataflow problem on the summary call graph \p CG, one SCC at a
/// time, in the order of \p Direction. In an SCC, the solver computes the
/// outputs again until no output changes. If \p IncludeIndirect is false, the
/// solver follows only the edges that have the call kind.
///
/// \p Problem provides:
/// - ValueT: the type of the values, which can be copied and compared with
///   operator==.
/// - ValueT getInitialValue(const SummaryCallGraph::Node &N): the bottom of
///   the values. It is the output of N before its first transfer, and the
///   start of the join for the input of N.
/// - ValueT getExternalValue(): the output of the code outside the summaries,
///   which the external calling node and the calls external node stand for.
/// - void join(ValueT &Into, const ValueT &V): joins \p V into \p Into.
/// - ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In): the
///   output of N for the input \p In. It is not called for the special nodes.
///
/// The transfer must be monotone, and the values must have a finite height, so
/// that the iteration in an SCC ends.
///
/// Returns the output of each node, also of the special nodes.
template <typename ProblemT>
DenseMap<const SummaryCallGraph::Node *, typename ProblemT::ValueT>
solveSummaryDataflow(const SummaryCallGraph &CG, ProblemT &Problem,
                     SummaryDataflowDirection Direction,
                     bool IncludeIndirect = true) {
  using Node = SummaryCallGraph::Node;
  using ValueT = typename ProblemT::ValueT;
  const bool TopDown = Direction == SummaryDataflowDirection::TopDown;

  DenseMap<const Node *, ValueT> Out;
  auto IsExternal = [&](const Node *N) {
    return N == CG.getExternalCallingNode() || N == CG.getCallsExternalNode();
  };
  // The nodes that give the input of N: its callers or its callees. The order
  // of the SCCs puts them before N, or into the SCC of N.
  auto GetSources = [&](const Node *N) {
    return TopDown ? N->callers(IncludeIndirect) : N->callees(IncludeIndirect);
  };

  auto Solve = [&](SummaryCallGraph::SCC C) {
    for (const Node *N : C)
      Out.try_emplace(N, IsExternal(N) ? Problem.getExternalValue()
                                       : Problem.getInitialValue(*N));
    // A node that is alone in its SCC and that is not its own source needs
    // one transfer.
    bool Iterate = C.size() > 1 || any_of(GetSources(C.front()),
                                          [&](const SummaryCallGraph::Edge &E) {
                                            return E.getNode() == C.front();
                                          });
    bool Changed;
    do {
      Changed = false;
      for (const Node *N : C) {
        if (IsExternal(N))
          continue;
        ValueT In = Problem.getInitialValue(*N);
        for (const SummaryCallGraph::Edge &E : GetSources(N))
          Problem.join(In, Out.find(E.getNode())->second);
        ValueT NewOut = Problem.transfer(*N, In);
        ValueT &OldOut = Out.find(N)->second;
        if (!(NewOut == OldOut)) {
          OldOut = std::move(NewOut);
          Changed = true;
        }
      }
    } while (Changed && Iterate);
  };

  ArrayRef<SummaryCallGraph::SCC> SCCs = CG.sccs(IncludeIndirect);
  if (TopDown)
    for (SummaryCallGraph::SCC C : reverse(SCCs))
      Solve(C);
  else
    for (SummaryCallGraph::SCC C : SCCs)
      Solve(C);
  return Out;
}

} // end namespace llvm

#endif // LLVM_ANALYSIS_SUMMARYDATAFLOW_H
