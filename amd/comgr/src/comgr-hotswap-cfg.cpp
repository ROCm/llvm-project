//===- comgr-hotswap-cfg.cpp - HotSwap register-liveness CFG -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of reglive::buildCfg and reglive::reversePostOrder. See
/// comgr-hotswap-cfg.h. Not wired into any production rewrite path.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-cfg.h"

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCInstrAnalysis.h"

#include <algorithm>
#include <optional>
#include <set>
#include <utility>

namespace COMGR {
namespace hotswap {
namespace reglive {

namespace {

// A program terminator has no static successor within this local CFG: an
// s_endpgm variant or an analyzed return.
bool isProgramTerminator(const InternalDecodedInst &DI, const LLVMState &LS) {
  const unsigned Op = DI.Inst.getOpcode();
  if (Op == LS.SEndPgmOpcode || Op == LS.SEndPgmSavedOpcode)
    return true;
  return LS.MIA && LS.MIA->isReturn(DI.Inst);
}

// A direct (statically resolvable) branch: an ordinary branch that is neither
// an indirect branch nor a call.
bool isDirectBranch(const InternalDecodedInst &DI, const LLVMState &LS) {
  return LS.MIA && LS.MIA->isBranch(DI.Inst) &&
         !LS.MIA->isIndirectBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst);
}

// Whether \p DI ends a basic block: any branch (direct or indirect), a return,
// or a program terminator. Calls are intentionally not terminators -- they
// fall through to the return site.
bool endsBlock(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (isProgramTerminator(DI, LS))
    return true;
  return LS.MIA && LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst);
}

// Whether a block ending in \p DI exposes no statically-known successor.
bool hasNoStaticSuccessor(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (isProgramTerminator(DI, LS))
    return true;
  return LS.MIA && LS.MIA->isIndirectBranch(DI.Inst);
}

// Iterative post-order DFS from \p Start over blocks flagged in \p Allowed.
void dfsPostOrder(unsigned Start, const Cfg &Graph,
                  const std::vector<char> &Allowed, std::vector<char> &Visited,
                  std::vector<unsigned> &PostOrder) {
  if (!Allowed[Start] || Visited[Start])
    return;
  Visited[Start] = 1;

  std::vector<std::pair<unsigned, size_t>> Stack;
  Stack.emplace_back(Start, 0);
  while (!Stack.empty()) {
    const unsigned Block = Stack.back().first;
    const llvm::SmallVectorImpl<unsigned> &Succs =
        Graph.Blocks[Block].Successors;
    const size_t Idx = Stack.back().second;
    if (Idx < Succs.size()) {
      Stack.back().second = Idx + 1; // advance before any reallocating push.
      const unsigned Succ = Succs[Idx];
      if (Allowed[Succ] && !Visited[Succ]) {
        Visited[Succ] = 1;
        Stack.emplace_back(Succ, 0);
      }
      continue;
    }
    PostOrder.push_back(Block);
    Stack.pop_back();
  }
}

} // namespace

Cfg buildCfg(llvm::ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
             llvm::ArrayRef<uint64_t> ExtraLeaders) {
  Cfg Graph;
  if (Decoded.empty())
    return Graph;

  const uint64_t StartOffset = Decoded.front().Offset;
  const InternalDecodedInst &Last = Decoded.back();
  const uint64_t SectionEnd = Last.Offset + Last.Size;

  // -- Leader detection -----------------------------------------------------
  std::set<uint64_t> Leaders;
  Leaders.insert(StartOffset);
  for (uint64_t Leader : ExtraLeaders) {
    if (Leader >= StartOffset && Leader < SectionEnd)
      Leaders.insert(Leader);
  }
  for (const InternalDecodedInst &DI : Decoded) {
    const uint64_t Next = DI.Offset + DI.Size;
    if (endsBlock(DI, LS) && Next < SectionEnd)
      Leaders.insert(Next);
    if (isDirectBranch(DI, LS)) {
      if (std::optional<uint64_t> Target =
              evaluateDirectControlFlowTarget(DI, LS)) {
        if (*Target >= StartOffset && *Target < SectionEnd)
          Leaders.insert(*Target);
      }
    }
  }

  // -- Block partitioning ---------------------------------------------------
  for (size_t I = 0, E = Decoded.size(); I < E;) {
    BasicBlock Block;
    Block.StartOffset = Decoded[I].Offset;
    while (I < E) {
      const InternalDecodedInst &DI = Decoded[I];
      const uint64_t Next = DI.Offset + DI.Size;
      const bool Terminates = endsBlock(DI, LS);
      Block.InstIndices.push_back(I);
      Block.EndOffset = Next;
      ++I;
      if (Terminates || (I < E && Leaders.count(Next)))
        break;
    }
    Graph.OffsetToBlock.try_emplace(Block.StartOffset,
                                    static_cast<unsigned>(Graph.Blocks.size()));
    Graph.Blocks.push_back(std::move(Block));
  }

  // -- Edge construction ----------------------------------------------------
  auto addEdge = [&](unsigned From, unsigned To) {
    if (llvm::is_contained(Graph.Blocks[From].Successors, To))
      return;
    Graph.Blocks[From].Successors.push_back(To);
    Graph.Blocks[To].Predecessors.push_back(From);
  };

  for (unsigned BI = 0, BE = Graph.Blocks.size(); BI < BE; ++BI) {
    const size_t LastIndex = Graph.Blocks[BI].InstIndices.back();
    const InternalDecodedInst &Term = Decoded[LastIndex];
    if (hasNoStaticSuccessor(Term, LS))
      continue;

    bool Unconditional = false;
    if (isDirectBranch(Term, LS)) {
      Unconditional = LS.MIA && LS.MIA->isUnconditionalBranch(Term.Inst);
      if (std::optional<uint64_t> Target =
              evaluateDirectControlFlowTarget(Term, LS)) {
        llvm::DenseMap<uint64_t, unsigned>::iterator It =
            Graph.OffsetToBlock.find(*Target);
        if (It != Graph.OffsetToBlock.end())
          addEdge(BI, It->second);
      }
    }

    // Conditional branches and non-branch block ends fall through; direct
    // unconditional branches do not.
    if (!Unconditional) {
      llvm::DenseMap<uint64_t, unsigned>::iterator It =
          Graph.OffsetToBlock.find(Graph.Blocks[BI].EndOffset);
      if (It != Graph.OffsetToBlock.end())
        addEdge(BI, It->second);
    }
  }

  return Graph;
}

std::vector<unsigned> reversePostOrder(const Cfg &Graph,
                                       llvm::ArrayRef<unsigned> Scope) {
  const size_t N = Graph.Blocks.size();
  std::vector<char> Allowed(N, 0);
  std::vector<unsigned> Roots;
  if (Scope.empty()) {
    Roots.reserve(N);
    for (size_t I = 0; I < N; ++I) {
      Allowed[I] = 1;
      Roots.push_back(static_cast<unsigned>(I));
    }
  } else {
    for (unsigned Block : Scope)
      if (Block < N)
        Allowed[Block] = 1;
    for (unsigned Block : Scope)
      if (Block < N)
        Roots.push_back(Block);
  }

  std::vector<char> Visited(N, 0);
  std::vector<unsigned> PostOrder;
  PostOrder.reserve(N);
  for (unsigned Root : Roots)
    dfsPostOrder(Root, Graph, Allowed, Visited, PostOrder);

  std::reverse(PostOrder.begin(), PostOrder.end());
  return PostOrder;
}

} // namespace reglive
} // namespace hotswap
} // namespace COMGR
