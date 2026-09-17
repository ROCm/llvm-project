//===- setpc-analysis.cpp - Transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "setpc-analysis.h"

#include "mc-state.h"
#include "transpiler/decoder/decode.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

using namespace llvm;

namespace COMGR::transpiler {

namespace {

// Index of the SGPR `Reg` names, or nullopt when it is not one of the general
// scalar registers the analysis follows. A pair is named by its low register,
// which is the index this returns for either width.
std::optional<unsigned> sgprIndex(const MCRegisterInfo &MRI, MCRegister Reg) {
  if (!Reg)
    return std::nullopt;
  MCRegister Low = MRI.getSubReg(Reg, AMDGPU::sub0);
  Low = stripRegEncoding(Low ? Low : Reg);

  // The condition, mode and aperture registers encode in the scalar range but
  // are not storage a program-counter chain can be carried in.
  switch (Low) {
  case AMDGPU::VCC_LO:
  case AMDGPU::VCC_HI:
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
  case AMDGPU::SCC:
  case AMDGPU::MODE:
  case AMDGPU::M0:
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
  case AMDGPU::LDS_DIRECT:
    return std::nullopt;
  default:
    break;
  }

  if (!MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Low))
    return std::nullopt;
  unsigned Encoding = MRI.getEncodingValue(Low);
  if (Encoding & (AMDGPU::HWEncoding::IS_VGPR | AMDGPU::HWEncoding::IS_AGPR))
    return std::nullopt;
  return Encoding & AMDGPU::HWEncoding::REG_IDX_MASK;
}

// Most source offsets one indirect jump may reach before the analysis stops
// enumerating and refuses it. A dispatcher a compiler wrote stays far below
// this, so a wider fan-out is more likely a value the analysis got wrong than
// a table it can state.
constexpr unsigned MaxSetPcTargets = 16;

// The source offsets one register pair may hold, as a dataflow lattice.
//
// The bottom element is the empty offset set: nothing reaches the pair. A join
// unions the offset sets, and a value only ever moves up. The two top elements
// carry no offsets. Both refuse the jumps that read them, and differ only in
// the refusal they produce.
class TargetOffsets {
public:
  TargetOffsets() = default;

  static TargetOffsets single(uint64_t Offset) {
    TargetOffsets Result;
    Result.Offsets.push_back(Offset);
    return Result;
  }

  // Top: some path leaves the pair holding a value the analysis cannot name.
  static TargetOffsets unnameable() {
    TargetOffsets Result;
    Result.Top = Reason::Unnameable;
    return Result;
  }

  bool isUnnameable() const { return Top == Reason::Unnameable; }
  bool isTooMany() const { return Top == Reason::TooMany; }
  bool isTop() const { return Top != Reason::None; }

  // The offsets that reach the pair, ascending and distinct. Empty at the top
  // and at the bottom alike.
  ArrayRef<uint64_t> offsets() const { return Offsets; }

  // Join `Other` into this value. Return true if this value changed, which is
  // what keeps the worklist going.
  bool join(const TargetOffsets &Other) {
    if (Other.Top > Top) {
      Top = Other.Top;
      Offsets.clear();
      return true;
    }
    if (isTop())
      return false;

    bool Changed = false;
    for (uint64_t Offset : Other.Offsets) {
      if (is_contained(Offsets, Offset))
        continue;
      if (Offsets.size() == MaxSetPcTargets) {
        Top = Reason::TooMany;
        Offsets.clear();
        return true;
      }
      Offsets.push_back(Offset);
      Changed = true;
    }
    if (Changed)
      sort(Offsets);
    return Changed;
  }

  // Raise this value to unnameable. Return true if that changed it.
  bool raiseToUnnameable() { return join(unnameable()); }

private:
  // Why the value is at the top. A join keeps the larger reason, so a pair that
  // is both unnameable and over the cap reads as unnameable.
  enum class Reason { None, TooMany, Unnameable };

  SmallVector<uint64_t> Offsets;
  Reason Top = Reason::None;
};

// What every register pair holds at one point, keyed by the pair's low register
// index. A pair with no entry is one the surrounding code says nothing about.
using PairOffsets = DenseMap<unsigned, TargetOffsets>;

// The displacement a program-counter capture has been carried by so far, and
// whether any has been applied. An undisplaced capture is not yet a target:
// the source computes one by adding to the capture, and a jump reading the
// bare capture jumps back onto the instruction that follows the capture.
struct PcChain {
  uint64_t Value = 0;
  bool Displaced = false;
};

// What the analysis knows about the scalar registers while it walks one block.
// A chain lives entirely within one block: it runs from the capture that starts
// it to the jump that reads it.
//
// Every register an instruction writes must go through `markWritten` before the
// instruction records what it knows. A register pair is named by its low
// register, so a write to either half kills the chain the pair carried, and a
// caller that marks only the register it tracks leaves the other half looking
// live to the jumps that read it.
class BlockState {
public:
  // Forget what the register at `Idx` held, and forget the chain of the pair it
  // is the high half of. Note the write, so the block reports it.
  void markWritten(unsigned Idx) {
    Chains.erase(Idx);
    Scalars.erase(Idx);
    Written.insert(Idx);
    if (Idx > 0)
      Chains.erase(Idx - 1);
  }

  // Start a chain at `Offset`, the source offset the captured program counter
  // names.
  void recordCapture(unsigned Idx, uint64_t Offset) {
    Chains[Idx] = PcChain{Offset, /*Displaced=*/false};
  }

  PcChain *chain(unsigned Idx) {
    auto It = Chains.find(Idx);
    return It == Chains.end() ? nullptr : &It->second;
  }

  // Carry the chain based at `Idx` to `Value`.
  void displace(unsigned Idx, uint64_t Value) {
    Chains[Idx] = PcChain{Value, /*Displaced=*/true};
  }

  void recordScalar(unsigned Idx, uint32_t Value) { Scalars[Idx] = Value; }

  std::optional<uint32_t> scalar(unsigned Idx) const {
    auto It = Scalars.find(Idx);
    return It == Scalars.end() ? std::nullopt
                               : std::optional<uint32_t>(It->second);
  }

  // Whether the block wrote either half of the pair based at `Idx`.
  bool pairIsWritten(unsigned Idx) const {
    return Written.contains(Idx) || Written.contains(Idx + 1);
  }

  // Record what every pair the block writes holds where the block ends: the
  // source offset a displaced chain names, or an unnameable value when the
  // writes left no offset behind. Pairs the block leaves alone stay out of
  // `Out`, because they keep whatever reached the block.
  void summarize(PairOffsets &Out) const {
    auto Record = [&](unsigned Base) {
      auto It = Chains.find(Base);
      Out[Base] = It != Chains.end() && It->second.Displaced
                      ? TargetOffsets::single(It->second.Value)
                      : TargetOffsets::unnameable();
    };
    for (unsigned Idx : Written) {
      Record(Idx);
      if (Idx > 0)
        Record(Idx - 1);
    }
  }

private:
  DenseMap<unsigned, PcChain> Chains;
  DenseMap<unsigned, uint32_t> Scalars;
  DenseSet<unsigned> Written;
};

// Mark every scalar register `Di` writes, including each register a wide
// destination spans.
void markDefsWritten(const DecodedInst &Di, const MCRegisterInfo &MRI,
                     BlockState &State) {
  for (unsigned I = 0; I != Di.NumDefs && I != Di.numOperands(); ++I) {
    if (!Di.isReg(I))
      continue;
    MCRegister Reg = Di.getReg(I);
    if (std::optional<unsigned> Idx = sgprIndex(MRI, Reg))
      State.markWritten(*Idx);
    for (MCPhysReg Sub : MRI.subregs(Reg))
      if (std::optional<unsigned> Idx = sgprIndex(MRI, Sub))
        State.markWritten(*Idx);
  }
}

// The value of operand `Index` when it is a compile-time constant that fits in
// 32 bits.
std::optional<uint32_t> immediate32(const MCInst &Inst, unsigned Index) {
  std::optional<int64_t> Value = evalOperandAsConst(Inst, Index);
  if (!Value)
    return std::nullopt;
  return static_cast<uint32_t>(*Value);
}

// Name a register pair the way the source assembly spells it, for a refusal.
std::string pairName(unsigned Idx) {
  return (Twine("s[") + Twine(Idx) + ":" + Twine(Idx + 1) + "]").str();
}

// Whether `Op` is an indirect jump: it moves the program counter to what a
// register pair holds. s_swap_pc_i64 also saves a return address, so it is a
// call, but the analysis treats both the same way.
bool isIndirectJump(CanonicalOp Op) {
  return Op == CanonicalOp::S_SETPC_B64 || Op == CanonicalOp::S_SWAPPC_B64;
}

// Every offset an indirect jump at a given source offset has been found to
// reach, keyed by the jump's source offset.
using JumpTargets = DenseMap<uint64_t, DenseSet<uint64_t>>;

// Work out where every indirect jump in `Insts` goes, splitting the walk at
// `WalkBlockStarts` and drawing the edges out of a jump from `KnownTargets`.
// `InstOffsets` holds the offset of every decoded instruction. `Insts` must not
// be empty.
Expected<SetPcAnalysis> classifyAgainstBlockStarts(
    ArrayRef<DecodedInst> Insts, const DenseSet<uint64_t> &WalkBlockStarts,
    const DenseSet<uint64_t> &InstOffsets, const JumpTargets &KnownTargets,
    const MCRegisterInfo &MRI) {
  assert(!Insts.empty() && "the walk needs at least one instruction to split");

  // One recovered block, in source order.
  struct Block {
    // Source offset of the first instruction of the block.
    uint64_t Start = 0;
    // Index into `Insts` of the last instruction of the block.
    unsigned LastInst = 0;
    // What the block itself leaves in the register pairs it writes. Pairs it
    // does not write are absent and keep what reached the block.
    PairOffsets Writes;
    // Source offsets of the blocks control can go to from here.
    SmallVector<uint64_t> Successors;
    // What the paths into the block leave in the register pairs, joined. Only
    // meaningful once `Reachable` is set.
    PairOffsets Entry;
    // Whether the dataflow has reached this block over any path.
    bool Reachable = false;

    // Join what a predecessor leaves into `Entry`. Return true if `Entry`
    // changed. A register pair only one side names is one the other side says
    // nothing about, which is as incomplete as a write that left no offset
    // behind.
    bool joinEntry(const PairOffsets &FromPredecessor) {
      if (!Reachable) {
        Reachable = true;
        Entry = FromPredecessor;
        return true;
      }

      bool Changed = false;
      for (auto &Held : Entry) {
        auto It = FromPredecessor.find(Held.first);
        Changed |= It == FromPredecessor.end() ? Held.second.raiseToUnnameable()
                                               : Held.second.join(It->second);
      }
      for (const auto &Held : FromPredecessor) {
        if (Entry.count(Held.first))
          continue;
        TargetOffsets Offsets = Held.second;
        Offsets.raiseToUnnameable();
        Entry[Held.first] = std::move(Offsets);
        Changed = true;
      }
      return Changed;
    }
  };
  SmallVector<Block> Blocks;
  // Index into `Blocks` of the block starting at a source offset.
  DenseMap<uint64_t, unsigned> BlockOf;

  // A jump whose own block leaves the register pair it reads alone. It waits
  // for the dataflow to say what the paths into the block leave there.
  struct DeferredSite {
    uint64_t Offset;
    unsigned BlockIndex;
    unsigned PairBase;
  };
  SmallVector<DeferredSite> Deferred;

  SetPcAnalysis Result;

  // Work out where one jump goes from the chain its own block built.
  auto classify = [&](const DecodedInst &Di, BlockState &State) {
    SetPcSite Site;
    unsigned SourceIndex = Di.FirstSrcIdx;
    std::optional<unsigned> Source;
    if (SourceIndex < Di.numOperands() && Di.isReg(SourceIndex))
      Source = sgprIndex(MRI, Di.getReg(SourceIndex));
    if (!Source) {
      Site.RefusalReason = "reads its target from something other than a "
                           "scalar register pair";
      Result.Sites[Di.Offset] = std::move(Site);
      return;
    }

    PcChain *Chain = State.chain(*Source);
    if (!Chain || !Chain->Displaced) {
      if (!State.pairIsWritten(*Source)) {
        Deferred.push_back(
            {Di.Offset, static_cast<unsigned>(Blocks.size() - 1), *Source});
        return;
      }
      Site.RefusalReason =
          (Twine("reads ") + pairName(*Source) +
           ", which its block writes without computing a source offset in it")
              .str();
      Result.Sites[Di.Offset] = std::move(Site);
      return;
    }

    if (!InstOffsets.contains(Chain->Value)) {
      Site.RefusalReason =
          (Twine("reaches source offset 0x") + Twine::utohexstr(Chain->Value) +
           ", which no decoded instruction starts at")
              .str();
      Result.Sites[Di.Offset] = std::move(Site);
      return;
    }

    Site.Targets.push_back(Chain->Value);
    Result.Sites[Di.Offset] = std::move(Site);
    Result.ExtraBlockStarts.insert(Chain->Value);
  };

  assert(Insts.size() <= std::numeric_limits<unsigned>::max() &&
         "instruction and block indices are tracked as unsigned");

  BlockState State;
  for (unsigned I = 0, E = Insts.size(); I != E; ++I) {
    const DecodedInst &Di = Insts[I];
    if (Blocks.empty() || WalkBlockStarts.contains(Di.Offset)) {
      if (!Blocks.empty())
        State.summarize(Blocks.back().Writes);
      State = BlockState();
      BlockOf[Di.Offset] = Blocks.size();
      Blocks.push_back(Block());
      Blocks.back().Start = Di.Offset;
    }
    Blocks.back().LastInst = I;

    switch (Di.CanonOp) {
    case CanonicalOp::S_GETPC_B64: {
      // The capture names the offset of the instruction that follows it. It
      // writes both halves of the pair.
      std::optional<unsigned> Dst;
      if (Di.NumDefs >= 1 && Di.numOperands() >= 1 && Di.isReg(0))
        Dst = sgprIndex(MRI, Di.getReg(0));
      if (Dst) {
        State.markWritten(*Dst);
        State.markWritten(*Dst + 1);
        State.recordCapture(*Dst, Di.Offset + Di.sizeInBytes());
        continue;
      }
      break;
    }

    case CanonicalOp::S_ADD_U32: {
      // Low half of a displacement split across two adds, or a plain constant
      // fold that a later add can use as its addend.
      if (Di.NumDefs < 1 || !Di.isReg(0))
        break;
      std::optional<unsigned> Dst = sgprIndex(MRI, Di.getReg(0));
      if (!Dst)
        break;
      unsigned Src0 = Di.FirstSrcIdx;
      unsigned Src1 = Src0 + 1;
      if (Src1 >= Di.numOperands())
        break;
      std::optional<uint32_t> Src0Imm = immediate32(Di.Inst, Src0);
      std::optional<uint32_t> Src1Imm = immediate32(Di.Inst, Src1);
      if (Src0Imm && Src1Imm) {
        State.markWritten(*Dst);
        State.recordScalar(*Dst, *Src0Imm + *Src1Imm);
        continue;
      }
      std::optional<unsigned> Src0Idx;
      if (Di.isReg(Src0))
        Src0Idx = sgprIndex(MRI, Di.getReg(Src0));
      if (Src0Idx != Dst)
        break;
      PcChain *Chain = State.chain(*Dst);
      if (!Chain || Chain->Displaced)
        break;
      std::optional<uint32_t> Addend = Src1Imm;
      if (!Addend && Di.isReg(Src1)) {
        std::optional<unsigned> Src1Idx = sgprIndex(MRI, Di.getReg(Src1));
        if (Src1Idx)
          Addend = State.scalar(*Src1Idx);
      }
      if (!Addend)
        break;
      uint64_t Displaced = Chain->Value + *Addend;
      State.markWritten(*Dst);
      State.displace(*Dst, Displaced);
      continue;
    }

    case CanonicalOp::S_ADDC_U32: {
      // High half of a split displacement. The low add already folded the
      // whole displacement into the chain when the source offset stays within
      // four gigabytes of the capture, so this only carries the high addend.
      if (Di.NumDefs < 1 || !Di.isReg(0))
        break;
      std::optional<unsigned> Dst = sgprIndex(MRI, Di.getReg(0));
      if (!Dst || *Dst == 0)
        break;
      unsigned Low = *Dst - 1;
      PcChain *Chain = State.chain(Low);
      if (!Chain || !Chain->Displaced)
        break;
      unsigned Src0 = Di.FirstSrcIdx;
      unsigned Src1 = Src0 + 1;
      if (Src1 >= Di.numOperands() || !Di.isReg(Src0))
        break;
      if (sgprIndex(MRI, Di.getReg(Src0)) != Dst)
        break;
      std::optional<uint32_t> Addend = immediate32(Di.Inst, Src1);
      if (!Addend)
        break;
      // The add writes the high half of the pair the chain is based at, so the
      // chain has to be read before the write and put back after it.
      uint64_t Displaced =
          Chain->Value + (static_cast<uint64_t>(*Addend) << 32);
      State.markWritten(*Dst);
      State.displace(Low, Displaced);
      continue;
    }

    case CanonicalOp::S_ADD_NC_U64: {
      // The whole displacement folded into one add, which writes both halves of
      // the pair. It commutes, so the capture stands on either side of the
      // constant.
      if (Di.NumDefs < 1 || !Di.isReg(0))
        break;
      std::optional<unsigned> Dst = sgprIndex(MRI, Di.getReg(0));
      if (!Dst)
        break;
      PcChain *Chain = State.chain(*Dst);
      if (!Chain || Chain->Displaced || Di.SrcMap.size() < 2)
        break;
      unsigned SrcA = Di.SrcMap[0];
      unsigned SrcB = Di.SrcMap[1];
      std::optional<unsigned> SrcAIdx;
      if (Di.isReg(SrcA))
        SrcAIdx = sgprIndex(MRI, Di.getReg(SrcA));
      std::optional<unsigned> SrcBIdx;
      if (Di.isReg(SrcB))
        SrcBIdx = sgprIndex(MRI, Di.getReg(SrcB));
      std::optional<int64_t> Displacement;
      if (SrcAIdx == Dst && !SrcBIdx)
        Displacement = evalOperandAsConst(Di.Inst, SrcB);
      else if (SrcBIdx == Dst && !SrcAIdx)
        Displacement = evalOperandAsConst(Di.Inst, SrcA);
      if (!Displacement)
        break;
      uint64_t Displaced = Chain->Value + static_cast<uint64_t>(*Displacement);
      State.markWritten(*Dst);
      State.markWritten(*Dst + 1);
      State.displace(*Dst, Displaced);
      continue;
    }

    case CanonicalOp::S_SETPC_B64:
      classify(Di, State);
      continue;

    case CanonicalOp::S_SWAPPC_B64: {
      classify(Di, State);
      // The call writes the offset it returns to over whatever its destination
      // pair held, which is a source offset like any a chain computes.
      if (Di.NumDefs >= 1 && Di.numOperands() >= 1 && Di.isReg(0)) {
        if (std::optional<unsigned> Dst = sgprIndex(MRI, Di.getReg(0))) {
          State.markWritten(*Dst);
          State.markWritten(*Dst + 1);
          uint64_t Return = Di.Offset + Di.sizeInBytes();
          if (InstOffsets.contains(Return))
            State.displace(*Dst, Return);
        }
      }
      continue;
    }

    default:
      break;
    }

    markDefsWritten(Di, MRI, State);
  }
  State.summarize(Blocks.back().Writes);

  // Draw the edges out of every block. An indirect jump names no offset the
  // decode could follow, so its edges are the targets already found for it:
  // what this walk resolved, plus what the earlier walks reached.
  //
  // A jump nothing has resolved yet draws no edges at all, which is optimistic:
  // the blocks it reaches miss what it would have left them. That is sound only
  // because a jump still unresolved once the rounds settle refuses the whole
  // kernel, so nothing computed from the missing edges reaches the raise.
  for (unsigned I = 0, E = Blocks.size(); I != E; ++I) {
    Block &Blk = Blocks[I];
    const DecodedInst &Last = Insts[Blk.LastInst];
    if (isIndirectJump(Last.CanonOp)) {
      auto Site = Result.Sites.find(Last.Offset);
      if (Site != Result.Sites.end())
        Blk.Successors.assign(Site->second.Targets);
      auto Known = KnownTargets.find(Last.Offset);
      if (Known != KnownTargets.end())
        for (uint64_t Target : Known->second)
          if (!is_contained(Blk.Successors, Target))
            Blk.Successors.push_back(Target);
      continue;
    }
    std::optional<uint64_t> Next;
    if (I + 1 != E)
      Next = Blocks[I + 1].Start;
    Expected<SmallVector<uint64_t>> Successors =
        computeDecodedBlockSuccessors(Last, Next);
    if (!Successors)
      return Successors.takeError();
    Blk.Successors = std::move(*Successors);
  }

  // Run the forward dataflow to a fixpoint. The lattice is bounded: a register
  // pair holds at most MaxSetPcTargets offsets before it goes to the top, and a
  // join only ever moves a value up, so the worklist runs dry.
  Blocks[0].Reachable = true;
  SetVector<unsigned> Worklist;
  Worklist.insert(0);
  while (!Worklist.empty()) {
    unsigned I = Worklist.pop_back_val();

    PairOffsets Exit = Blocks[I].Entry;
    for (const auto &Written : Blocks[I].Writes)
      Exit[Written.first] = Written.second;

    for (uint64_t Successor : Blocks[I].Successors) {
      auto It = BlockOf.find(Successor);
      if (It == BlockOf.end())
        continue;
      if (Blocks[It->second].joinEntry(Exit))
        Worklist.insert(It->second);
    }
  }

  // A deferred jump reads a register pair its own block leaves alone, so what
  // the paths into the block leave there is what the jump reads.
  for (const DeferredSite &Site : Deferred) {
    SetPcSite Classified;
    const Block &Blk = Blocks[Site.BlockIndex];
    if (!Blk.Reachable) {
      Classified.RefusalReason =
          (Twine("reads ") + pairName(Site.PairBase) +
           ", and no path the decode recovered reaches its block")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }

    auto Held = Blk.Entry.find(Site.PairBase);
    if (Held == Blk.Entry.end()) {
      Classified.RefusalReason = (Twine("reads ") + pairName(Site.PairBase) +
                                  ", which no path reaching its block writes")
                                     .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }

    const TargetOffsets &Offsets = Held->second;
    if (Offsets.isUnnameable()) {
      Classified.RefusalReason =
          (Twine("reads ") + pairName(Site.PairBase) +
           ", which some path reaching its block leaves without a source "
           "offset")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }
    if (Offsets.isTooMany()) {
      Classified.RefusalReason =
          (Twine("reads ") + pairName(Site.PairBase) + ", which more than " +
           Twine(MaxSetPcTargets) + " source offsets reach")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }

    assert(!Offsets.offsets().empty() &&
           "a reachable block joins at least one predecessor, which either "
           "names an offset or raises the pair to the top");
    const uint64_t *Undecoded = find_if(Offsets.offsets(), [&](uint64_t Value) {
      return !InstOffsets.contains(Value);
    });
    if (Undecoded != Offsets.offsets().end()) {
      Classified.RefusalReason =
          (Twine("reaches source offset 0x") + Twine::utohexstr(*Undecoded) +
           ", which no decoded instruction starts at")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }

    Classified.Targets.assign(Offsets.offsets());
    Result.ExtraBlockStarts.insert_range(Offsets.offsets());
    Result.Sites[Site.Offset] = std::move(Classified);
  }

  return Result;
}

} // namespace

Expected<SetPcAnalysis> analyzeSetPc(ArrayRef<DecodedInst> Insts,
                                     const std::set<uint64_t> &BlockStarts,
                                     const MCState &Mc) {
  SetPcAnalysis Result;
  if (Insts.empty())
    return Result;

  DenseSet<uint64_t> InstOffsets;
  InstOffsets.reserve(Insts.size());
  for (const DecodedInst &Di : Insts)
    InstOffsets.insert(Di.Offset);

  // An indirect jump ends the block it sits in, so what follows it leads a
  // block of its own. For a call that block is where the callee returns to; for
  // a jump nothing falls into it, but the instructions there still need a block
  // to be raised into. The offset one past the last instruction leads nothing.
  //
  // Splitting here also keeps each jump the last instruction of its block, so a
  // jump reads the chain state of the instructions before it and of nothing
  // else.
  DenseSet<uint64_t> WalkBlockStarts(BlockStarts.begin(), BlockStarts.end());
  for (const DecodedInst &Di : Insts) {
    if (!isIndirectJump(Di.CanonOp))
      continue;
    uint64_t Fallthrough = Di.Offset + Di.sizeInBytes();
    if (InstOffsets.contains(Fallthrough))
      WalkBlockStarts.insert(Fallthrough);
  }

  // Resolving a jump both names a block the walk did not split at and draws an
  // edge the walk did not have. Either changes what reaches the other jumps, so
  // the walk repeats until it learns nothing new. Both the block starts and the
  // edges only ever grow, so a round that learns nothing is the fixpoint.
  //
  // `KnownTargets` keeps every edge any round drew, including edges out of a
  // jump a later round stops resolving. A stale edge can carry offsets into a
  // block along a path the final classification no longer believes in. That is
  // harmless only because the jump that stopped resolving refuses the whole
  // kernel, so no answer computed from its stale edge reaches the raise.
  //
  // That growth also bounds the number of rounds, and a walk that runs past the
  // bound has broken the argument. Refusing the kernel beats spinning on it.
  const size_t MaxRounds = Insts.size() * (MaxSetPcTargets + 1) + 1;
  JumpTargets KnownTargets;
  for (size_t Round = 0;; ++Round) {
    if (Round == MaxRounds)
      return createStringError(
          inconvertibleErrorCode(),
          "program-counter analysis did not settle in %zu rounds", MaxRounds);

    Expected<SetPcAnalysis> Classified = classifyAgainstBlockStarts(
        Insts, WalkBlockStarts, InstOffsets, KnownTargets, *Mc.RegInfo);
    if (!Classified)
      return Classified.takeError();

    bool Learned = false;
    for (uint64_t Start : Classified->ExtraBlockStarts)
      Learned |= WalkBlockStarts.insert(Start).second;
    for (const auto &Site : Classified->Sites)
      for (uint64_t Target : Site.second.Targets)
        Learned |= KnownTargets[Site.first].insert(Target).second;
    if (Learned)
      continue;

    // The raise must build the blocks the analysis walked, or its answers stop
    // holding. Report every split the walk made that the decode did not.
    Result = std::move(*Classified);
    Result.ExtraBlockStarts = std::move(WalkBlockStarts);
    for (uint64_t Start : BlockStarts)
      Result.ExtraBlockStarts.erase(Start);
    return Result;
  }
}

} // namespace COMGR::transpiler
