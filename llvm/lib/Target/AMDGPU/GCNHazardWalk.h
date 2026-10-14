//===- GCNHazardWalk.h - Backward walk for GCN hazard checks ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// A backward walk over the instructions that can execute before a given one,
/// nearest first, and the hazard queries built on it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNHAZARDWALK_H
#define LLVM_LIB_TARGET_AMDGPU_GCNHAZARDWALK_H

#include "SIInstrInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>

namespace llvm {

/// An instruction before the walk's start and its distance: the least total
/// weight of the instructions between the two, over all paths.
struct GCNHazardStep {
  const MachineInstr &MI;
  int Distance;
};

/// A minimal iterator for range-for loops over \p RangeT, which provides
/// getStep() and advance(), false at the end. Copies share the range;
/// advancing one invalidates the others.
template <typename RangeT> class GCNHazardStepIterator {
public:
  GCNHazardStepIterator() = default;
  GCNHazardStep operator*() const { return Range->getStep(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE GCNHazardStepIterator &operator++() {
    if (!Range->advance())
      Range = nullptr;
    return *this;
  }
  bool operator==(const GCNHazardStepIterator &RHS) const {
    return Range == RHS.Range;
  }
  bool operator!=(const GCNHazardStepIterator &RHS) const {
    return Range != RHS.Range;
  }

private:
  friend RangeT;
  explicit GCNHazardStepIterator(RangeT *Range) : Range(Range) {}

  RangeT *Range = nullptr;
};

/// Walks the CFG backwards from an instruction and yields each instruction
/// that can execute before it once, at its distance, nearest first. Steps at
/// the bound or further away are not yielded. Since distances never decrease,
/// a check may stop at the first step that settles its answer.
///
/// endPath() ends the path through the current step. A later path to the same
/// instruction is no shorter and is not offered again, so whether a check ends
/// a path may depend only on the step, not on its distance or path.
///
/// A predecessor's terminators, and the instructions after an INLINEASM_BR, are
/// yielded only if the edge executes them, and the start is yielded where a
/// cycle leads back to it.
///
/// A walk can be iterated once.
class GCNHazardWalk {
public:
  using WeightFn = function_ref<unsigned(const MachineInstr &)>;

  /// The default weight. Inline asm weighs 0, as it may be empty.
  static unsigned getWaitStates(const MachineInstr &MI) {
    return MI.isInlineAsm() ? 0 : SIInstrInfo::getNumWaitStates(MI);
  }

  /// Walks back from \p MI. Distances add up \p Weight, or getWaitStates() if
  /// null; a meta instruction, which does not issue, weighs 0 either way.
  /// \p Weight must outlive the walk, which holds it by reference.
  GCNHazardWalk(const MachineInstr &MI, int Bound, WeightFn Weight = {});

  GCNHazardWalk(const GCNHazardWalk &) = delete;

  using iterator = GCNHazardStepIterator<GCNHazardWalk>;

  iterator begin() {
    assert(!Started && "a walk is a single pass");
    Started = true;
    if (Bound <= 0 || !findStep())
      return end();
    return iterator(this);
  }
  iterator end() { return iterator(); }

  /// Yields nothing behind the current step on its path. The step must not be
  /// read again; the next operation is the increment.
  void endPath() { Current.reset(); }

  /// Yields no more steps at \p NewBound or further away.
  void narrowBound(int NewBound) {
    Bound = std::min(Bound, NewBound);
    Limit = std::min(Limit, Bound - 1);
  }

  int getBound() const { return Bound; }

private:
  friend iterator;
  template <typename StepsT, typename WindowForT>
  friend int findMaxWindowDeficit(StepsT Steps, WindowForT WindowFor);
  template <typename StepsT, typename IsHazardT>
  friend int findNearestDistance(StepsT Steps, IsHazardT IsHazard);
  template <typename IsHazardT, typename IsExpiredT>
  friend bool hasActiveHazard(GCNHazardWalk Walk, IsHazardT IsHazard,
                              IsExpiredT IsExpired);

  // A position to continue upwards from, Distance away from the start.
  struct Cursor {
    const MachineBasicBlock *MBB;
    MachineBasicBlock::const_reverse_instr_iterator Pos;
    int Distance;
    // Pos is among the exits of a block that control leaves from several
    // places, so other cursors may come this way too.
    bool Shared = false;
    // The cursor the walk began with.
    bool FromStart = false;
  };

  // Orders Pending as a min-heap by distance.
  struct IsFarther {
    bool operator()(const Cursor &A, const Cursor &B) const {
      return A.Distance > B.Distance;
    }
  };

  // Lets steps come in any order, each still once and at its distance: a
  // cursor gives way to a closer one only where paths can meet.
  void allowAnyOrder() {
    assert(!Started && "the order is fixed once the walk has begun");
    Ordered = false;
  }

  unsigned getWeight(const MachineInstr &MI) const {
    if (!Weight)
      return getWaitStates(MI);
    return MI.isMetaInstruction() ? 0 : Weight(MI);
  }

  GCNHazardStep getStep() const {
    assert(Current && "the path through the step has ended");
    return {*CurrentMI, Current->Distance};
  }

  // Within findStep()'s span, moving past the current step is all it takes.
  // Every step of every query comes through here.
  LLVM_ATTRIBUTE_ALWAYS_INLINE bool advance() {
    if (Current) {
      Cursor &C = *Current;
      C.Distance += getWeight(*CurrentMI);
      ++C.Pos;
      if (C.Pos != SpanEnd && C.Distance <= Limit) {
        CurrentMI = &*C.Pos;
        if (!CurrentMI->isBundle())
          return true;
      }
    }
    return leaveSpan();
  }

  static bool isJoin(const Cursor &C);
  bool leaveSpan();
  bool findStep();
  void enterPredecessors(const MachineBasicBlock *MBB, int Distance);
  void pushPending(const Cursor &C);

  const WeightFn Weight;
  int Bound;
  // The farthest distance the current cursor still yields: below Bound, and in
  // strict order no farther than the nearest pending cursor.
  int Limit;
  bool Started = false;
  bool Ordered = true;

  const MachineBasicBlock *StartMBB;
  MachineBasicBlock::const_reverse_instr_iterator StartPos;
  bool AtStart = true;
  // Current is the stepped cursor and CurrentMI its instruction; advance()
  // moves it up to SpanEnd, where something can happen, without findStep().
  // Pending holds the other cursors as a min-heap. Reached holds the reached
  // positions where paths can meet; AtStart stands in for the start position.
  std::optional<Cursor> Current;
  const MachineInstr *CurrentMI = nullptr;
  MachineBasicBlock::const_reverse_instr_iterator SpanEnd;
  SmallVector<Cursor, 4> Pending;
  // Built on the first join, so that a query in one block need not.
  std::optional<SmallDenseSet<
      PointerUnion<const MachineInstr *, const MachineBasicBlock *>, 8>>
      Reached;
};

/// The largest Window - Distance over \p Steps, or 0 if none is positive.
/// \p Steps is a GCNHazardWalk or another single-pass range of GCNHazardStep
/// with getBound(), narrowBound() and allowAnyOrder(). \p WindowFor(MI) gives
/// MI's window, at most the bound, or std::nullopt; it is a template parameter
/// so that it is inlined. A meta instruction, which does not issue, has no
/// window.
template <typename StepsT, typename WindowForT>
int findMaxWindowDeficit(StepsT Steps, WindowForT WindowFor) {
  const int MaxWindow = Steps.getBound();
  Steps.allowAnyOrder();
  int Deficit = 0;
  for (auto [MI, Distance] : Steps) {
    std::optional<int> Window = WindowFor(MI);
    // Only a window pays for the meta test.
    if (!Window || MI.isMetaInstruction())
      continue;
    assert(*Window >= 0 && *Window <= MaxWindow && "window out of bounds");
    if (*Window - Distance <= Deficit)
      continue;
    Deficit = *Window - Distance;
    // No window exceeds MaxWindow, so nothing this far away can beat Deficit.
    Steps.narrowBound(MaxWindow - Deficit);
  }
  return Deficit;
}

/// The distance to the nearest of \p Steps that \p IsHazard(MI) accepts, or
/// std::numeric_limits<int>::max() if there is none. A meta instruction does
/// not issue, so it is never the hazard. \p Steps is as for
/// findMaxWindowDeficit.
template <typename StepsT, typename IsHazardT>
int findNearestDistance(StepsT Steps, IsHazardT IsHazard) {
  Steps.allowAnyOrder();
  int Nearest = std::numeric_limits<int>::max();
  for (auto [MI, Distance] : Steps) {
    // Only a match pays for the meta test.
    if (!IsHazard(MI) || MI.isMetaInstruction())
      continue;
    // Only a nearer step can come after this one.
    Nearest = Distance;
    Steps.narrowBound(Distance);
  }
  return Nearest;
}

/// Whether a hazard is still active at the start of \p Walk: some path reaches
/// a step \p IsHazard(MI) accepts without first passing one \p IsExpired(MI)
/// accepts. A step both accept is a hazard; a meta instruction, which does not
/// issue, is neither. Inline asm is offered to both.
template <typename IsHazardT, typename IsExpiredT>
bool hasActiveHazard(GCNHazardWalk Walk, IsHazardT IsHazard,
                     IsExpiredT IsExpired) {
  Walk.allowAnyOrder();
  for (auto [MI, Distance] : Walk) {
    // Only a match pays for the meta test.
    if (IsHazard(MI) && !MI.isMetaInstruction())
      return true;
    if (IsExpired(MI) && !MI.isMetaInstruction())
      Walk.endPath();
  }
  return false;
}

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNHAZARDWALK_H
