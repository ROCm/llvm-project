//===- LIRP.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Live interval-based register pressure (LIRP) estimation utilities.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_LIRP_H
#define LLVM_LIB_TARGET_AMDGPU_LIRP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/LaneBitmask.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

namespace llvm {

class MachineRegisterInfo;
class raw_ostream;

/// A union of non-overlapping half-open [open, close) segments. The last
/// segment may be open-ended (close == Inf).
class RangeUnion {
public:
  using Segment = std::pair<unsigned, unsigned>;
  static constexpr unsigned Inf = ~0u;

  /// Append a segment.
  /// \p Open must be finite and strictly less than \p Close, the segment
  /// must start at or after the previous segment's end.
  void append(unsigned Open, unsigned Close);

  /// Append an open-ended segment [Open, Inf).
  void appendOpen(unsigned Open) { append(Open, Inf); }

  /// Close the last (open) segment at \p Close.
  void closeLast(unsigned Close);

  /// \returns true if any segment of \p Other overlaps a segment of this union.
  bool overlaps(const RangeUnion &Other) const;

  /// Merge \p Other's segments into this union.
  void merge(const RangeUnion &Other);

  void clear() { Segments.clear(); }

  bool empty() const { return Segments.empty(); }

  /// \returns true if the last segment is still open (close == Inf).
  bool isOpen() const {
    return !Segments.empty() && Segments.back().second == Inf;
  }

  ArrayRef<Segment> segments() const { return Segments; }

  void dump(raw_ostream &OS) const;

private:
  SmallVector<Segment, 4> Segments;
};

/// Greedy, no-split allocator that estimates live interval-based register
/// pressure (LIRP) incrementally, suitable for use during scheduling.
///
/// Unlike instantaneous register pressure (max number of simultaneously live
/// registers), LIRP also accounts for:
///  - registers that die and are later redefined (use the same assignment),
///  - tuple contiguity/alignment (fragmentation).
///
/// The reported value (getRegNum) is an upper bound on the true minimum
/// (graph chromatic number), which is itself an upper bound on instant
/// pressure.
///
/// Events are fed in scheduled order as open(Reg, Mask) / close(Reg, Mask),
/// where Mask is a bitmask for 32-bit subregisters.
class LIRPAllocTable {
public:
  struct Event {
    Register Reg;
    uint64_t PrevSubRegs;
    uint64_t NewSubRegs;
  };

  explicit LIRPAllocTable(const MachineRegisterInfo &MRI) : MRI(&MRI) {}

  /// Apply \p Reg's subregister transition \p PrevSubRegs -> \p NewSubRegs.
  void update(Register Reg, uint64_t PrevSubRegs, uint64_t NewSubRegs);

  /// Mark subregisters of \p Reg as alive, allocating more space if needed.
  void open(Register Reg, uint64_t Offsets);

  /// Mark subregisters of \p Reg dead.
  void close(Register Reg, uint64_t Offsets);

  /// \returns the size of current allocation.
  unsigned getRegNum() const { return Slots.size(); }

  /// \returns what getRegNum() would become after applying \p Events (in
  /// order), without modifying the table.
  unsigned speculate(ArrayRef<Event> Events) const;

  void clear();

private:
  unsigned getRegWidth(Register Reg) const;
  unsigned getRegAlign(Register Reg) const;

  /// Record subregisters of \p Reg becoming live (\p IsOpen) or dead
  /// in its per-lane live ranges. \returns the assigned event index.
  unsigned updateRegLanes(Register Reg, uint64_t Offsets, bool IsOpen);

  /// Greedily assign \p Reg to the lowest aligned slot compatible with Reg.
  void assign(Register Reg);

  /// Commit \p Reg at a known slot base \p B: record the base and merge
  /// its lane ranges into the slot range.
  void assignAt(Register Reg, unsigned B);

  /// Discard all assignments and replay RegLanes in first-appearance order,
  /// assigning every register greedily.
  void rebuild();

  /// Rebuild reusing the existing bases for the first \p StartSeq registers
  /// and re-assigning the rest greedily.
  void rebuildFrom(unsigned StartSeq);

  const MachineRegisterInfo *MRI = nullptr;

  // Per-register per-lane ranges.
  SmallMapVector<Register, SmallVector<RangeUnion>, 32> RegLanes;
  unsigned NextIdx = 0;

  // First-appearance sequence index per register.
  DenseMap<Register, unsigned> SeqIdx;

  // Occupancy of one 32-bit slot: live range and current slot owner.
  struct SlotInfo {
    RangeUnion Range;
    Register Owner;
  };

  // Current slot assignments and per-slot occupancy.
  DenseMap<Register, unsigned> RegBase; // register -> assigned base slot
  SmallVector<SlotInfo> Slots;
};

/// LIRP tracker: routes pressure tracking over SGPR and VGPR (VGPR/AGPR/AVGPR)
/// register files.
class LIRPTracker {
public:
  /// Register lane liveness transitions.
  struct Event {
    Register Reg;
    LaneBitmask PrevMask;
    LaneBitmask NewMask;
  };
  using LiveRegSet = DenseMap<unsigned, LaneBitmask>;

  LIRPTracker() = default;

  /// (Re)create both tables for \p MRI and open all \p LiveIns for a new
  /// region.
  void reset(const MachineRegisterInfo &MRI, const LiveRegSet &LiveIns);

  /// Update corresponding (SGPR or VGPR) allocation table
  void update(Register Reg, LaneBitmask PrevMask, LaneBitmask NewMask);

  /// Speculate the per-file register pressure after applying \p Events
  /// (in order), without mutating state. \returns {SGPR num, VGPR num}.
  std::pair<unsigned, unsigned> speculate(ArrayRef<Event> Events) const;

  unsigned getSGPRNum() const { return table(SGPRTable).getRegNum(); }
  unsigned getVGPRNum() const { return table(VGPRTable).getRegNum(); }

private:
  bool isSGPR(Register Reg) const;

  static LIRPAllocTable &table(std::optional<LIRPAllocTable> &T) {
    assert(T && "LIRPTracker used before reset()");
    return *T;
  }
  static const LIRPAllocTable &table(const std::optional<LIRPAllocTable> &T) {
    assert(T && "LIRPTracker used before reset()");
    return *T;
  }

  /// Convert a 16-bit-based lane mask (two bits correspond to a 32-bit
  /// subregister) into a 32-bit-based subregister mask (each bit corresponds
  /// to a 32-bit subregister).
  static uint64_t lanesToSubRegs(LaneBitmask Lanes);

  const MachineRegisterInfo *MRI = nullptr;
  std::optional<LIRPAllocTable> SGPRTable;
  std::optional<LIRPAllocTable> VGPRTable;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_LIRP_H
