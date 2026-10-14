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
// control leaves a block from several places: among its exits and at the body
// above them. Only those positions are recorded, and AtStart stops a cycle back
// to the start. In strict order the closest cursor always moves next; in any
// order a cursor gives way to a closer one only at those positions, since
// between them a position has a single path to it. No cursor enters the start
// block above the start: every path leaves the block past the instructions
// above the start, so the first cursor came by sooner. Otherwise the current
// cursor runs on without touching the heap, so a query that ends in its own
// block does not allocate.
//
//===----------------------------------------------------------------------===//

#include "GCNHazardWalk.h"

using namespace llvm;

// INLINEASM_BR is not a terminator, and copies of its results can follow it,
// but control leaves the block at it for its indirect targets. In a block that
// may hold one, the walk takes every instruction as an exit and the asm as a
// branch, so that the paths that enter at it meet there.
static bool isExit(const MachineInstr &MI) {
  return MI.isTerminator() || MI.getParent()->mayHaveInlineAsmBr();
}

static bool isBranch(const MachineInstr &MI) {
  return MI.isBranch() || MI.getOpcode() == TargetOpcode::INLINEASM_BR;
}

static bool branchesTo(const MachineInstr &MI, const MachineBasicBlock *MBB) {
  if (MI.getOpcode() == TargetOpcode::INLINEASM_BR)
    return any_of(MI.operands(), [MBB](const MachineOperand &MO) {
      return MO.isMBB() && MO.getMBB() == MBB;
    });
  return MI.isBranch() && MI.getNumOperands() > 0 && MI.getOperand(0).isMBB() &&
         MI.getOperand(0).getMBB() == MBB;
}

// The positions control can leave \p Pred from for \p MBB: each exit that
// branches to \p MBB, and the last instruction if control also falls through.
// An edge that no exit names is a fallthrough past every terminator. Exits from
// \p Passed up are left out: the walk has been through them.
static void getEntries(
    const MachineBasicBlock &Pred, const MachineBasicBlock *MBB,
    MachineBasicBlock::const_reverse_instr_iterator Passed,
    SmallVectorImpl<MachineBasicBlock::const_reverse_instr_iterator> &Entries) {
  // Only an indirect target is entered from an INLINEASM_BR.
  bool AsmTarget = MBB->isInlineAsmBrIndirectTarget();
  bool Named = false, Above = false;
  for (auto I = Pred.instr_rbegin(), E = Pred.instr_rend();
       I != E && (I->isTerminator() || AsmTarget); ++I) {
    Above = Above || I == Passed;
    if (branchesTo(*I, MBB)) {
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
// block at its end or at a branch, so among shared exits paths meet there.
bool GCNHazardWalk::isJoin(const Cursor &C) {
  return C.Shared && (C.Pos == C.MBB->instr_rbegin() ||
                      (C.Pos != C.MBB->instr_rend() && isBranch(*C.Pos)));
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

    C.Shared = C.Shared && isExit(*C.Pos);

    // A cursor that came back into the start block stops at the start. Among
    // shared exits it stops at the next branch, where another path may join.
    SpanEnd = End;
    if (C.MBB == StartMBB && !C.FromStart)
      SpanEnd = StartPos;
    if (C.Shared) {
      for (auto I = std::next(C.Pos); I != SpanEnd && I != End && isExit(*I);
           ++I) {
        if (isBranch(*I)) {
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
