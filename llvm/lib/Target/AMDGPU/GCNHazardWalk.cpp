//===- GCNHazardWalk.cpp - Backward walk for GCN hazard checks ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dijkstra's algorithm on instruction positions, so a position is first reached
// at its distance. A block's top is left once, so paths can only meet where
// control leaves a block from several places: among its terminators and at the
// body above them. Only those positions are recorded, and AtStart stops a cycle
// back to the start. In strict order the closest cursor always moves next; in
// any order a cursor gives way to a closer one only at those positions, since
// between them a position has a single path to it. No cursor enters the start
// block above the start: every path leaves the block past the instructions
// above the start, so the first cursor came by sooner. Otherwise the current
// cursor runs on without touching the heap, so a query that ends in its own
// block does not allocate.
//
//===----------------------------------------------------------------------===//

#include "GCNHazardWalk.h"

using namespace llvm;

// The positions control can leave \p Pred from for \p MBB: each terminator that
// branches to \p MBB, and the last instruction if control also falls through.
// An edge that no terminator names is a fallthrough past every terminator.
// Branches from \p Passed up are left out: the walk has been through them.
static void getEntries(
    const MachineBasicBlock &Pred, const MachineBasicBlock *MBB,
    MachineBasicBlock::const_reverse_instr_iterator Passed,
    SmallVectorImpl<MachineBasicBlock::const_reverse_instr_iterator> &Entries) {
  bool Named = false, Above = false;
  for (auto I = Pred.instr_rbegin(), E = Pred.instr_rend();
       I != E && I->isTerminator(); ++I) {
    Above = Above || I == Passed;
    if (I->isBranch() && I->getNumOperands() > 0 && I->getOperand(0).isMBB() &&
        I->getOperand(0).getMBB() == MBB) {
      Named = true;
      if (!Above)
        Entries.push_back(I);
    }
  }
  if (!Named || ((Entries.empty() || Entries.front() != Pred.instr_rbegin()) &&
                 Pred.isLayoutSuccessor(MBB) && !Pred.back().isBarrier()))
    Entries.push_back(Pred.instr_rbegin());
}

GCNHazardWalk::GCNHazardWalk(const MachineInstr &MI, int Bound, WeightFn Weight)
    : Weight(Weight), Bound(Bound), Limit(Bound - 1) {
  const MachineBasicBlock *MBB = MI.getParent();
  auto Pos = std::next(MI.getReverseIterator());
  while (Pos != MBB->instr_rend() && Pos->isBundle())
    ++Pos;
  StartMBB = MBB;
  StartPos = Pos;
  Cursor First{MBB, Pos, 0};
  First.FromStart = true;
  Current = First;
}

// Whether another path can reach the position of \p C too: cursors enter a
// block at its end or at a branch, so among shared terminators paths meet
// there.
bool GCNHazardWalk::isJoin(const Cursor &C) {
  return C.Shared && (C.Pos == C.MBB->instr_rbegin() ||
                      (C.Pos != C.MBB->instr_rend() && C.Pos->isBranch()));
}

bool GCNHazardWalk::leaveSpan() {
  if (Current) {
    Cursor &C = *Current;
    // A pending cursor at the same distance need not go first.
    if (C.Distance >= Bound) {
      Current.reset();
    } else if (Ordered && !Pending.empty() &&
               Pending.front().Distance < C.Distance) {
      pushPending(C);
      Current.reset();
    }
  }
  return findStep();
}

bool GCNHazardWalk::findStep() {
  for (;;) {
    if (!Current) {
      // The nearest pending cursor is past a narrowed bound, so all are.
      if (Pending.empty() || Pending.front().Distance >= Bound)
        return false;
      std::pop_heap(Pending.begin(), Pending.end(), IsFarther());
      Current = Pending.pop_back_val();
    }

    Cursor &C = *Current;
    auto End = C.MBB->instr_rend();
    while (C.Pos != End && C.Pos->isBundle())
      ++C.Pos;

    if (C.MBB == StartMBB && C.Pos == StartPos) {
      if (!AtStart) {
        Current.reset();
        continue;
      }
      AtStart = false;
    } else if (isJoin(C)) {
      // In any order this cursor may have run ahead of a closer one, which
      // goes first.
      if (!Pending.empty() && Pending.front().Distance < C.Distance) {
        pushPending(C);
        Current.reset();
        continue;
      }
      PointerUnion<const MachineInstr *, const MachineBasicBlock *> Key = C.MBB;
      if (C.Pos != End)
        Key = &*C.Pos;
      if (!Reached)
        Reached.emplace();
      if (!Reached->insert(Key).second) {
        Current.reset();
        continue;
      }
    }

    if (C.Pos == End) {
      enterPredecessors(C.MBB, C.Distance);
      continue;
    }

    C.Shared = C.Shared && C.Pos->isTerminator();

    // A cursor that came back into the start block stops at the start. Among
    // shared terminators it stops at the next branch, where another path may
    // join.
    SpanEnd = End;
    if (C.MBB == StartMBB && !C.FromStart)
      SpanEnd = StartPos;
    if (C.Shared) {
      for (auto I = std::next(C.Pos);
           I != SpanEnd && I != End && I->isTerminator(); ++I) {
        if (I->isBranch()) {
          SpanEnd = I;
          break;
        }
      }
    }
    Limit = Bound - 1;
    if (Ordered && !Pending.empty())
      Limit = std::min(Limit, Pending.front().Distance);
    CurrentMI = &*C.Pos;
    return true;
  }
}

// Replaces the cursor at the top of \p MBB by one per entry of a predecessor.
void GCNHazardWalk::enterPredecessors(const MachineBasicBlock *MBB,
                                      int Distance) {
  Current.reset();
  SmallVector<MachineBasicBlock::const_reverse_instr_iterator, 2> Entries;
  for (const MachineBasicBlock *Pred : MBB->predecessors()) {
    Entries.clear();
    getEntries(*Pred, MBB, Pred == StartMBB ? StartPos : Pred->instr_rend(),
               Entries);
    bool Shared = Pred->succ_size() > 1 || Entries.size() > 1;
    for (auto Pos : Entries) {
      Cursor Entry{Pred, Pos, Distance};
      Entry.Shared = Shared;
      if (Current)
        pushPending(Entry);
      else
        Current = Entry;
    }
  }
}

void GCNHazardWalk::pushPending(const Cursor &C) {
  Pending.push_back(C);
  std::push_heap(Pending.begin(), Pending.end(), IsFarther());
}
