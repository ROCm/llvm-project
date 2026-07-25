//===- comgr-hotswap-liveness-analysis.cpp - HotSwap register liveness ---===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of reglive::LivenessAnalysis. See
/// comgr-hotswap-liveness-analysis.h. Not wired into any production rewrite
/// path.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-liveness-analysis.h"

#include "comgr-hotswap-def-use.h"
#include "comgr-hotswap-internal.h"

#include <deque>

namespace COMGR {
namespace hotswap {
namespace reglive {

namespace {

// The registers an instruction definitely overwrites. Predicated defs preserve
// the old value on at least one path, and EXEC-masked vector defs preserve it
// on inactive lanes; until EXEC state is tracked per program point, neither can
// be treated as an unconditional liveness kill.
RegisterSet killDefs(const InstDefUse &DU) {
  if (DU.HasPredicatedDef)
    return {};
  RegisterSet Kills = DU.Defs;
  if (DU.HasExecMaskedVectorDef) {
    Kills.clearClass(RegClass::VGPR);
    Kills.clearClass(RegClass::ACC_VGPR);
  }
  return Kills;
}

} // namespace

LivenessAnalysis::LivenessAnalysis(llvm::ArrayRef<InternalDecodedInst> Decoded,
                                   const Cfg &Graph,
                                   const llvm::MCInstrInfo &MCII,
                                   const llvm::MCRegisterInfo &MRI,
                                   llvm::ArrayRef<unsigned> Scope) {
  const size_t N = Graph.Blocks.size();
  BlockState.resize(N);

  std::vector<char> InScope(N, 0);
  if (Scope.empty()) {
    for (size_t I = 0; I < N; ++I)
      InScope[I] = 1;
  } else {
    for (unsigned B : Scope)
      if (B < N)
        InScope[B] = 1;
  }

  // Local transfer functions. `Gen` keeps only uses not already killed earlier
  // in the block; `Kill` accumulates every local kill.
  for (size_t I = 0; I < N; ++I) {
    if (!InScope[I])
      continue;
    BlockLiveness &State = BlockState[I];
    for (size_t InstIndex : Graph.Blocks[I].InstIndices) {
      InstDefUse DU(Decoded[InstIndex].Inst, MCII, MRI);
      RegisterSet Kills = killDefs(DU);
      RegisterSet UpwardUses = DU.Uses;
      UpwardUses -= State.Kill;
      State.Gen |= UpwardUses;
      State.Kill |= Kills;
    }
  }

  // Iterate to a fixed point. Seeding the worklist in reverse-post-order makes
  // a backward analysis converge quickly; predecessors are re-enqueued
  // whenever a block's live-in changes.
  const std::vector<unsigned> Rpo = reversePostOrder(Graph, Scope);
  std::deque<size_t> Worklist;
  std::vector<char> InWorklist(N, 0);
  auto enqueue = [&](size_t Index) {
    if (Index >= N || !InScope[Index] || InWorklist[Index])
      return;
    InWorklist[Index] = 1;
    Worklist.push_back(Index);
  };
  for (unsigned Block : Rpo)
    enqueue(Block);

  while (!Worklist.empty()) {
    const size_t Index = Worklist.front();
    Worklist.pop_front();
    InWorklist[Index] = 0;

    RegisterSet LiveOut;
    for (unsigned Succ : Graph.Blocks[Index].Successors)
      if (Succ < N && InScope[Succ])
        LiveOut |= BlockState[Succ].LiveIn;

    RegisterSet LiveIn = LiveOut;
    LiveIn -= BlockState[Index].Kill;
    LiveIn |= BlockState[Index].Gen;

    BlockLiveness &State = BlockState[Index];
    const bool LiveInChanged = State.LiveIn != LiveIn;
    if (State.LiveOut != LiveOut || LiveInChanged) {
      State.LiveOut = LiveOut;
      State.LiveIn = LiveIn;
      if (LiveInChanged) {
        for (unsigned Pred : Graph.Blocks[Index].Predecessors)
          enqueue(Pred);
      }
    }
  }

  // Materialize per-instruction live-before by replaying each block's transfer
  // function backward from its live-out. The transfer is applied per
  // instruction so a read-modify-write keeps its source live before the
  // instruction even though the same register is also defined here.
  for (size_t I = 0; I < N; ++I) {
    if (!InScope[I])
      continue;
    const BasicBlock &Block = Graph.Blocks[I];
    RegisterSet Live = BlockState[I].LiveOut;
    for (auto It = Block.InstIndices.rbegin(); It != Block.InstIndices.rend();
         ++It) {
      const size_t InstIndex = *It;
      InstDefUse DU(Decoded[InstIndex].Inst, MCII, MRI);
      RegisterSet Kills = killDefs(DU);
      Live -= Kills;
      Live |= DU.Uses;
      LiveBeforeByIndex.emplace(InstIndex, Live);
    }
  }
}

const BlockLiveness &
LivenessAnalysis::blockLiveness(unsigned BlockIndex) const {
  static const BlockLiveness EmptyState;
  return BlockIndex < BlockState.size() ? BlockState[BlockIndex] : EmptyState;
}

const RegisterSet &LivenessAnalysis::liveBefore(size_t InstIndex) const {
  auto It = LiveBeforeByIndex.find(InstIndex);
  return It != LiveBeforeByIndex.end() ? It->second : Empty;
}

bool LivenessAnalysis::isLiveBefore(size_t InstIndex, RegisterRef Ref) const {
  return liveBefore(InstIndex).contains(Ref);
}

} // namespace reglive
} // namespace hotswap
} // namespace COMGR
