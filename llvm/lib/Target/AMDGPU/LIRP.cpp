//===- LIRP.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements live interval-based register pressure (LIRP)
/// estimation utilities.
///
//===----------------------------------------------------------------------===//

#include "LIRP.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "lirp"

namespace {
const SIRegisterInfo &getTRI(const MachineRegisterInfo &MRI) {
  return *static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo());
}

template <typename Fn> void forEachSetBit(uint64_t Mask, Fn Cb) {
  for (unsigned Off = 0; Mask; ++Off, Mask >>= 1)
    if (Mask & 1)
      Cb(Off);
}

/// Return the lowest aligned base (a multiple of \p Align, starting at 0) for
/// which \p FitsFn returns true.
template <typename FitsFn>
unsigned lowestAlignedBase(unsigned SlotCount, unsigned Align, FitsFn Fits) {
  [[maybe_unused]] unsigned Limit = alignTo(SlotCount, Align);
  for (unsigned B = 0;; B += Align) {
    assert(B <= Limit && "base search must fit by the end of the slot range");
    if (Fits(B))
      return B;
  }
}
} // namespace

void RangeUnion::dump(raw_ostream &OS) const {
  OS << "{";
  bool First = true;
  for (const auto &S : Segments) {
    OS << (First ? "" : ",") << "[" << S.first << ",";
    if (S.second == Inf)
      OS << "inf";
    else
      OS << S.second;
    OS << ")";
    First = false;
  }
  OS << "}";
}

void RangeUnion::append(unsigned Open, unsigned Close) {
  assert(Open != Inf && "open index must be finite");
  assert(Open < Close && "empty or inverted segment");
  assert((Segments.empty() || Segments.back().second <= Open) &&
         "segment must follow the previous one");
  Segments.emplace_back(Open, Close);
}

void RangeUnion::closeLast(unsigned Close) {
  assert(isOpen() && "closeLast on a union with no open segment");
  assert(Close != Inf && "close index must be finite");
  assert(Segments.back().first < Close &&
         "close index must be after the segment's open");
  Segments.back().second = Close;
}

bool RangeUnion::overlaps(const RangeUnion &Other) const {
  for (const auto &Seg : Other.Segments) {
    unsigned Start = Seg.first, End = Seg.second;
    // Find the first segment in this union whose end > Start.
    // Segments are sorted by start and non-overlapping.
    auto It = llvm::partition_point(
        Segments, [&](const std::pair<unsigned, unsigned> &S) {
          // <= bc if segment ends at Start it does not
          // overlap with [Start, End)
          return S.second <= Start;
        });
    if (It != Segments.end() && It->first < End)
      return true;
  }
  return false;
}

void RangeUnion::merge(const RangeUnion &Other) {
  if (Other.Segments.empty())
    return;
  if (Segments.empty()) {
    Segments = Other.Segments;
    return;
  }

  SmallVector<Segment, 16> Out;
  Out.reserve(Segments.size() + Other.Segments.size());
  auto Push = [&](Segment S) {
    if (!Out.empty() && S.first <= Out.back().second)
      Out.back().second = std::max(Out.back().second, S.second);
    else
      Out.push_back(S);
  };

  unsigned I = 0, J = 0, N = Segments.size(), M = Other.Segments.size();
  while (I < N && J < M) {
    if (Segments[I].first <= Other.Segments[J].first)
      Push(Segments[I++]);
    else
      Push(Other.Segments[J++]);
  }
  while (I < N)
    Push(Segments[I++]);
  while (J < M)
    Push(Other.Segments[J++]);

  Segments = std::move(Out);
}

unsigned LIRPAllocTable::updateRegLanes(Register Reg, uint64_t Offsets,
                                        bool IsOpen) {
  unsigned Idx = NextIdx++;
  auto [It, Inserted] = RegLanes.try_emplace(Reg);
  if (Inserted) {
    It->second.resize(getRegWidth(Reg));
    unsigned Seq = SeqIdx.size();
    SeqIdx[Reg] = Seq;
  }
  SmallVector<RangeUnion> &L = It->second;
  for (unsigned Off = 0; Offsets; ++Off, Offsets >>= 1) {
    if (!(Offsets & 1))
      continue;
    if (IsOpen) {
      assert(!L[Off].isOpen() && "opening a lane that is already open");
      L[Off].appendOpen(Idx);
    } else {
      assert(L[Off].isOpen() && "closing a lane that is not open");
      L[Off].closeLast(Idx);
    }
  }
  return Idx;
}

void LIRPAllocTable::clear() {
  RegLanes.clear();
  NextIdx = 0;
  SeqIdx.clear();
  RegBase.clear();
  Slots.clear();
}

unsigned LIRPAllocTable::getRegWidth(Register Reg) const {
  const TargetRegisterClass *RC = MRI->getRegClass(Reg);
  return getTRI(*MRI).getRegClassWeight(RC).RegWeight;
}

unsigned LIRPAllocTable::getRegAlign(Register Reg) const {
  const TargetRegisterClass *RC = MRI->getRegClass(Reg);
  unsigned Bits = getTRI(*MRI).getRegClassAlignmentNumBits(RC);
  return std::max(1u, Bits / 32);
}

void LIRPAllocTable::assignAt(Register Reg, unsigned B) {
  ArrayRef<RangeUnion> Lanes = RegLanes.at(Reg);
  unsigned Width = Lanes.size();
  LLVM_DEBUG({
    dbgs() << "    assignAt " << printReg(Reg, &getTRI(*MRI))
           << " seq=" << SeqIdx.lookup(Reg) << " base=" << B << " lanes:";
    for (unsigned Off = 0; Off < Width; ++Off) {
      dbgs() << " [" << Off << "]=";
      Lanes[Off].dump(dbgs());
    }
    dbgs() << "\n";
  });

  RegBase[Reg] = B;
  if (B + Width > Slots.size())
    Slots.resize(B + Width);
  for (unsigned Off = 0; Off < Width; ++Off)
    if (!Lanes[Off].empty()) {
      SlotInfo &SI = Slots[B + Off];
      assert(!SI.Range.overlaps(Lanes[Off]) &&
             "assigning a register into a incompatible slot");

      SI.Range.merge(Lanes[Off]);
      if (Lanes[Off].isOpen())
        SI.Owner = Reg;
    }
}

void LIRPAllocTable::assign(Register Reg) {
  ArrayRef<RangeUnion> Lanes = RegLanes.find(Reg)->second;
  unsigned Width = Lanes.size(), Align = getRegAlign(Reg);

  // A subregister fits at base B if its lane range does not overlap the target
  // slot's range in time; empty lanes never conflict.
  unsigned B = lowestAlignedBase(Slots.size(), Align, [&](unsigned B) {
    for (unsigned Off = 0; Off < Width; ++Off) {
      const RangeUnion &Lane = Lanes[Off];
      unsigned S = B + Off;
      if (!Lane.empty() && S < Slots.size() && Slots[S].Range.overlaps(Lane))
        return false;
    }
    return true;
  });

  assignAt(Reg, B);
}

void LIRPAllocTable::rebuild() {
  LLVM_DEBUG(dbgs() << "  REBUILD (full)\n");
  RegBase.clear();
  Slots.clear();
  // Greedily assign each register once, in first-appearance order,
  // using the known register live ranges.
  for (Register Reg : make_first_range(RegLanes.getArrayRef()))
    assign(Reg);
}

void LIRPAllocTable::rebuildFrom(unsigned StartSeq) {
  LLVM_DEBUG(dbgs() << "  REBUILD-FROM startSeq=" << StartSeq << " of "
                    << RegLanes.size() << " regs\n");
  // Snapshot the prefix's existing bases before tearing down the slot state.
  SmallVector<std::pair<Register, unsigned>, 32> Prefix;
  Prefix.reserve(StartSeq);
  for (auto [Reg, Lanes] : RegLanes.getArrayRef().take_front(StartSeq))
    Prefix.emplace_back(Reg, RegBase[Reg]);

  RegBase.clear();
  Slots.clear();

  // Prefix keeps its bases; the tail is re-assigned greedily.
  for (auto [Reg, Base] : Prefix)
    assignAt(Reg, Base);
  for (auto [Reg, Lanes] : drop_begin(RegLanes.getArrayRef(), StartSeq))
    assign(Reg);
}

void LIRPAllocTable::update(Register Reg, uint64_t PrevSubRegs,
                            uint64_t NewSubRegs) {
  if (uint64_t CloseSubRegs = PrevSubRegs & ~NewSubRegs)
    close(Reg, CloseSubRegs);
  if (uint64_t OpenSubRegs = NewSubRegs & ~PrevSubRegs)
    open(Reg, OpenSubRegs);
}

void LIRPAllocTable::open(Register Reg, uint64_t Offsets) {
  bool WasNew = !RegBase.count(Reg);
  unsigned OpenIdx = updateRegLanes(Reg, Offsets, /*IsOpen=*/true);
  LLVM_DEBUG(dbgs() << "OPEN " << printReg(Reg, &getTRI(*MRI)) << " slotmask="
                    << format_hex(Offsets, 4) << " idx=" << OpenIdx
                    << (WasNew ? " (new)" : " (revival)") << "\n");

  auto BaseIt = RegBase.find(Reg);
  if (BaseIt == RegBase.end()) {
    // Open new register: allocate slots
    assign(Reg);
    return;
  }

  // Revival (redef) of the register: check if anyone reused the slots and
  // still holds them.
  unsigned B = BaseIt->second;
  unsigned StartSeq = ~0u;
  forEachSetBit(Offsets, [&](unsigned Off) {
    if (!Slots[B + Off].Range.isOpen())
      return;
    Register Owner = Slots[B + Off].Owner;
    LLVM_DEBUG(dbgs() << "  conflict slot " << (B + Off) << " owner="
                      << printReg(Owner, &getTRI(*MRI))
                      << " ownerSeq=" << SeqIdx.lookup(Owner) << "\n");

    StartSeq = std::min(StartSeq, SeqIdx.lookup(Owner));
  });
  // If there's a conflict, rebuild the table starting from the conflicting
  // register.
  if (StartSeq != ~0u) {
    rebuildFrom(StartSeq);
    return;
  }

  // All revived slots are available, update slot ranges.
  forEachSetBit(Offsets, [&](unsigned Off) {
    Slots[B + Off].Range.appendOpen(OpenIdx);
    Slots[B + Off].Owner = Reg;
  });
}

void LIRPAllocTable::close(Register Reg, uint64_t Offsets) {
  unsigned CloseIdx = updateRegLanes(Reg, Offsets, /*IsOpen=*/false);
  LLVM_DEBUG(dbgs() << "CLOSE " << printReg(Reg, &getTRI(*MRI)) << " slotmask="
                    << format_hex(Offsets, 4) << " idx=" << CloseIdx << "\n");

  auto BaseIt = RegBase.find(Reg);
  assert(BaseIt != RegBase.end() && "closing a register that was never assigned");
  unsigned B = BaseIt->second;

  forEachSetBit(Offsets, [&](unsigned Off) {
    Slots[B + Off].Range.closeLast(CloseIdx);
    Slots[B + Off].Owner = Register();
  });
}

unsigned LIRPAllocTable::speculate(ArrayRef<Event> Events) const {
  // Estimate getRegNum() after applying Events, without mutating the table.
  //
  // Fast path: closes never grow the table, and a single open of a brand-new
  // register grows it by at most the run it needs. Anything that could trigger
  // a rebuild (a revival) falls back to copy-and-replay.
  const Event *Open = nullptr;
  bool FastSpec = true;
  for (const Event &E : Events) {
    uint64_t OpenSubRegs = E.NewSubRegs & ~E.PrevSubRegs;
    if (!OpenSubRegs)
      continue;
    if (Open || RegLanes.contains(E.Reg)) {
      FastSpec = false;
      break;
    }
    Open = &E;
  }

  if (FastSpec) {
    if (!Open)
      return getRegNum(); // closes only

    // Slots freed by this batch's closes become usable by the open.
    SmallSet<unsigned, 8> Freed;
    for (const Event &E : Events) {
      uint64_t CloseSubRegs = E.PrevSubRegs & ~E.NewSubRegs;
      auto It = RegBase.find(E.Reg);
      if (!CloseSubRegs || It == RegBase.end())
        continue;
      unsigned B = It->second;
      forEachSetBit(CloseSubRegs, [&](unsigned Off) { Freed.insert(B + Off); });
    }

    auto IsFreed = [&](unsigned S) { return llvm::is_contained(Freed, S); };

    unsigned Width = getRegWidth(Open->Reg);
    unsigned Align = getRegAlign(Open->Reg);
    uint64_t Offsets = Open->NewSubRegs & ~Open->PrevSubRegs;

    // A subregister fits at base B if its slot is not currently live, treating
    // slots this batch's closes will free as available. The new register's
    // lane range is [now, inf), so a currently-live slot is the only conflict.
    unsigned B = lowestAlignedBase(Slots.size(), Align, [&](unsigned B) {
      for (uint64_t Mask = Offsets, Off = 0; Mask; ++Off, Mask >>= 1) {
        if (!(Mask & 1))
          continue;
        unsigned S = B + Off;
        if (S < Slots.size() && Slots[S].Range.isOpen() && !IsFreed(S))
          return false;
      }
      return true;
    });
    return std::max<unsigned>(Slots.size(), B + Width);
  }

  // Slow path: copy and replay (may trigger a rebuild).
  LIRPAllocTable Copy = *this;
  for (const Event &E : Events)
    Copy.update(E.Reg, E.PrevSubRegs, E.NewSubRegs);
  return Copy.getRegNum();
}

uint64_t LIRPTracker::lanesToSubRegs(LaneBitmask Lanes) {
  // Each 32-bit subregister corresponds to a 2-bit lane group;
  // a 32-bit subregister is considered alive if any of its lanes are alive.
  LaneBitmask::Type Mask = Lanes.getAsInteger();
  uint64_t Offsets = 0;
  for (unsigned Slot = 0; Mask; ++Slot, Mask >>= 2)
    if (Mask & 0x3)
      Offsets |= (1ULL << Slot);
  return Offsets;
}

void LIRPTracker::reset(const MachineRegisterInfo &MRI,
                        const LiveRegSet &LiveIns) {
  LLVM_DEBUG(dbgs() << "==== LIRP RESET region, " << LiveIns.size()
                    << " live-ins ====\n");
  this->MRI = &MRI;
  SGPRTable.emplace(MRI);
  VGPRTable.emplace(MRI);
  for (const auto &[RegNum, LaneMask] : LiveIns)
    update(Register(RegNum), LaneBitmask::getNone(), LaneMask);
}

bool LIRPTracker::isSGPR(Register Reg) const {
  assert(MRI && "LIRPTracker used before reset()");
  return getTRI(*MRI).isSGPRClass(MRI->getRegClass(Reg));
}

void LIRPTracker::update(Register Reg, LaneBitmask PrevMask,
                         LaneBitmask NewMask) {
  if (!Reg.isVirtual())
    return;
  LIRPAllocTable &T = table(isSGPR(Reg) ? SGPRTable : VGPRTable);
  T.update(Reg, lanesToSubRegs(PrevMask), lanesToSubRegs(NewMask));
}

std::pair<unsigned, unsigned>
LIRPTracker::speculate(ArrayRef<Event> Events) const {
  SmallVector<LIRPAllocTable::Event, 8> SEvents, VEvents;
  for (const Event &E : Events) {
    auto &Dst = isSGPR(E.Reg) ? SEvents : VEvents;
    Dst.push_back(
        {E.Reg, lanesToSubRegs(E.PrevMask), lanesToSubRegs(E.NewMask)});
  }

  return {table(SGPRTable).speculate(SEvents),
          table(VGPRTable).speculate(VEvents)};
}
