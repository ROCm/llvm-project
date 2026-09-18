//===- setpc-analysis.cpp - Transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "setpc-analysis.h"

#include "decode.h"
#include "mc-state.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
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

// The displacement a program-counter capture has been carried by so far, and
// whether any has been applied. An undisplaced capture is not yet a target:
// the source computes one by adding to the capture, and a transfer reading the
// bare capture jumps back onto the instruction that follows the capture.
struct PcChain {
  uint64_t Value = 0;
  bool Displaced = false;
};

// What the analysis knows about the scalar registers while walking one block.
// Everything here is block-local: a chain is only followed from the capture
// that starts it to the transfer that reads it, both within one block.
class BlockState {
public:
  // Start a chain at `Offset`, the source offset the captured program counter
  // names.
  void recordCapture(unsigned Idx, uint64_t Offset) {
    Chains[Idx] = PcChain{Offset, /*Displaced=*/false};
    Dirty.insert(Idx);
  }

  PcChain *chain(unsigned Idx) {
    auto It = Chains.find(Idx);
    return It == Chains.end() ? nullptr : &It->second;
  }

  // Carry the chain based at `Idx` to `Value`.
  void displace(unsigned Idx, uint64_t Value) {
    PcChain &Chain = Chains[Idx];
    Chain.Value = Value;
    Chain.Displaced = true;
    Dirty.insert(Idx);
  }

  void recordScalar(unsigned Idx, uint32_t Value) {
    Scalars[Idx] = Value;
    Dirty.insert(Idx);
  }

  std::optional<uint32_t> scalar(unsigned Idx) const {
    auto It = Scalars.find(Idx);
    return It == Scalars.end() ? std::nullopt
                               : std::optional<uint32_t>(It->second);
  }

  // Forget everything known about the register at `Idx`, and about the pair it
  // is the high half of.
  void invalidate(unsigned Idx) {
    Chains.erase(Idx);
    Scalars.erase(Idx);
    Dirty.insert(Idx);
    if (Idx > 0)
      Chains.erase(Idx - 1);
  }

  // Whether either half of the pair based at `Idx` was written in this block.
  bool pairIsDirty(unsigned Idx) const {
    return Dirty.contains(Idx) || Dirty.contains(Idx + 1);
  }

  // What every pair the block writes holds where the block ends: the source
  // offset a completed chain names, or nothing when the writes left no offset
  // behind. Pairs the block does not write are absent, since those keep
  // whatever reached the block.
  void summarize(DenseMap<unsigned, std::optional<uint64_t>> &Out) const {
    auto Record = [&](unsigned Base) {
      auto It = Chains.find(Base);
      Out[Base] = It != Chains.end() && It->second.Displaced
                      ? std::optional<uint64_t>(It->second.Value)
                      : std::nullopt;
    };
    for (unsigned Idx : Dirty) {
      Record(Idx);
      if (Idx > 0)
        Record(Idx - 1);
    }
  }

private:
  DenseMap<unsigned, PcChain> Chains;
  DenseMap<unsigned, uint32_t> Scalars;
  DenseSet<unsigned> Dirty;
};

// Forget what every scalar register `Di` writes held. Instructions the walk
// below models return before reaching this, so anything that gets here is an
// instruction whose effect on the chain is unknown.
void invalidateDefs(const DecodedInst &Di, const MCRegisterInfo &MRI,
                    BlockState &State) {
  for (unsigned I = 0; I != Di.NumDefs && I != Di.numOperands(); ++I) {
    if (!Di.isReg(I))
      continue;
    MCRegister Reg = Di.getReg(I);
    // A destination wider than one register covers several of the indices the
    // chains are keyed by, so each register it spans is invalidated in turn.
    if (std::optional<unsigned> Idx = sgprIndex(MRI, Reg))
      State.invalidate(*Idx);
    for (MCPhysReg Sub : MRI.subregs(Reg))
      if (std::optional<unsigned> Idx = sgprIndex(MRI, Sub))
        State.invalidate(*Idx);
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

// Whether `Op` transfers control through a register value.
bool isRegisterIndirectTransfer(CanonicalOp Op) {
  return Op == CanonicalOp::S_SETPC_B64 || Op == CanonicalOp::S_SWAPPC_B64;
}

// The source offsets one register pair may hold where a block begins. Either
// flag makes the set unusable: some path leaves the pair holding something the
// analysis cannot name, or more offsets reach it than may be enumerated.
struct PairFacts {
  SmallVector<uint64_t, 4> Values;
  bool Unknown = false;
  bool Overflowed = false;
};

bool operator==(const PairFacts &A, const PairFacts &B) {
  return A.Unknown == B.Unknown && A.Overflowed == B.Overflowed &&
         A.Values == B.Values;
}

// What every pair a block can be entered holding. A pair with no entry is one
// no path reaching the block says anything about.
using BlockFacts = DenseMap<unsigned, PairFacts>;

// What a block leaves in the pairs it writes: the source offset it computed,
// or nothing when its writes left no offset behind.
using BlockEffect = DenseMap<unsigned, std::optional<uint64_t>>;

// Add the offsets of `Src` to `Dst`, keeping them ascending and distinct. Past
// the cap the set stops growing and stands as overflowed, which refuses the
// sites reading it rather than dispatching on a truncated table.
void joinPairFacts(PairFacts &Dst, const PairFacts &Src) {
  Dst.Unknown |= Src.Unknown;
  Dst.Overflowed |= Src.Overflowed;
  for (uint64_t Value : Src.Values) {
    if (is_contained(Dst.Values, Value))
      continue;
    if (Dst.Values.size() == kMaxSetPcTargets) {
      Dst.Overflowed = true;
      break;
    }
    Dst.Values.push_back(Value);
  }
  sort(Dst.Values);
}

// Merge what one predecessor leaves into what a block is known to be entered
// holding, answering whether that changed anything. A pair only one side names
// is one the other leaves unknown, which is incomplete for the same reason a
// pair written without an offset is.
bool joinBlockFacts(BlockFacts &Dst, bool &DstSeeded, const BlockFacts &Src) {
  if (!DstSeeded) {
    DstSeeded = true;
    Dst = Src;
    return true;
  }

  BlockFacts Joined;
  for (const auto &Entry : Dst) {
    PairFacts Facts = Entry.second;
    auto It = Src.find(Entry.first);
    if (It == Src.end())
      Facts.Unknown = true;
    else
      joinPairFacts(Facts, It->second);
    Joined[Entry.first] = std::move(Facts);
  }
  for (const auto &Entry : Src) {
    if (Dst.count(Entry.first))
      continue;
    PairFacts Facts = Entry.second;
    Facts.Unknown = true;
    Joined[Entry.first] = std::move(Facts);
  }

  if (Joined.size() == Dst.size() && all_of(Joined, [&](const auto &E) {
        return Dst.lookup(E.first) == E.second;
      }))
    return false;
  Dst = std::move(Joined);
  return true;
}

// Every offset a transfer at a given source offset has been found to reach.
using TransferTargets = DenseMap<uint64_t, DenseSet<uint64_t>>;

// Classify every transfer in `Insts` into `Result`, splitting the walk at
// `WalkBlockStarts` and drawing the edges out of a transfer from
// `KnownTargets`. `InstOffsets` holds the offset of every decoded instruction.
Error classifyAgainstBlockStarts(
    ArrayRef<DecodedInst> Insts, const std::set<uint64_t> &WalkBlockStarts,
    const DenseSet<uint64_t> &InstOffsets, const TransferTargets &KnownTargets,
    uint64_t EntryOffset, const MCRegisterInfo &MRI, SetPcAnalysis &Result) {
  // One recovered block, in source order.
  struct Block {
    uint64_t Start = 0;
    size_t LastInst = 0;
    BlockEffect Effect;
    SmallVector<uint64_t> Successors;
    BlockFacts Entry;
    bool Reachable = false;
  };
  SmallVector<Block> Blocks;
  DenseMap<uint64_t, unsigned> BlockOf;

  // A transfer whose block leaves its pair alone, waiting for the dataflow to
  // say what reaches it.
  struct DeferredSite {
    uint64_t Offset;
    unsigned BlockIdx;
    unsigned Pair;
  };
  SmallVector<DeferredSite> Deferred;

  // Resolve one transfer's source pair, recording what was found for it.
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
      if (!State.pairIsDirty(*Source)) {
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
      Result.UndecodedTargets.insert(Chain->Value);
      return;
    }

    Site.SiteKind = SetPcSite::Kind::Direct;
    Site.DirectTarget = Chain->Value;
    Result.Sites[Di.Offset] = std::move(Site);
    Result.ExtraBlockStarts.insert(Chain->Value);
  };

  BlockState State;
  for (size_t I = 0, E = Insts.size(); I != E; ++I) {
    const DecodedInst &Di = Insts[I];
    if (Blocks.empty() || WalkBlockStarts.count(Di.Offset)) {
      if (!Blocks.empty())
        State.summarize(Blocks.back().Effect);
      State = BlockState();
      BlockOf[Di.Offset] = Blocks.size();
      Blocks.push_back(Block());
      Blocks.back().Start = Di.Offset;
    }
    Blocks.back().LastInst = I;

    switch (Di.CanonOp) {
    case CanonicalOp::S_GETPC_B64: {
      // The capture names the offset of the instruction that follows it.
      std::optional<unsigned> Dst;
      if (Di.NumDefs >= 1 && Di.numOperands() >= 1 && Di.isReg(0))
        Dst = sgprIndex(MRI, Di.getReg(0));
      if (Dst) {
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
      State.displace(*Dst, Chain->Value + *Addend);
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
      State.displace(Low,
                     Chain->Value + (static_cast<uint64_t>(*Addend) << 32));
      continue;
    }

    case CanonicalOp::S_ADD_NC_U64: {
      // The whole displacement folded into one add. It commutes, so the
      // capture stands on either side of the constant.
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
      State.displace(*Dst, Chain->Value + static_cast<uint64_t>(*Displacement));
      continue;
    }

    case CanonicalOp::S_SETPC_B64:
      classify(Di, State);
      continue;

    case CanonicalOp::S_SWAPPC_B64: {
      classify(Di, State);
      // The call writes the offset it returns to over whatever its destination
      // held, which is a source offset like any a chain computes.
      if (Di.NumDefs >= 1 && Di.numOperands() >= 1 && Di.isReg(0)) {
        if (std::optional<unsigned> Dst = sgprIndex(MRI, Di.getReg(0))) {
          State.invalidate(*Dst);
          State.invalidate(*Dst + 1);
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

    invalidateDefs(Di, MRI, State);
  }
  State.summarize(Blocks.back().Effect);

  // A transfer names no offset the decode could follow, so the edges out of
  // the block it ends are the ones already found for it: what this walk just
  // resolved, plus whatever earlier walks reached.
  for (unsigned I = 0, E = Blocks.size(); I != E; ++I) {
    Block &B = Blocks[I];
    const DecodedInst &Last = Insts[B.LastInst];
    if (isRegisterIndirectTransfer(Last.CanonOp)) {
      auto Site = Result.Sites.find(Last.Offset);
      if (Site != Result.Sites.end() &&
          Site->second.SiteKind == SetPcSite::Kind::Direct)
        B.Successors.push_back(Site->second.DirectTarget);
      auto Known = KnownTargets.find(Last.Offset);
      if (Known != KnownTargets.end())
        for (uint64_t Target : Known->second)
          if (!is_contained(B.Successors, Target))
            B.Successors.push_back(Target);
      continue;
    }
    std::optional<uint64_t> Next;
    if (I + 1 != E)
      Next = Blocks[I + 1].Start;
    Expected<SmallVector<uint64_t>> Successors =
        computeDecodedBlockSuccessors(Last, Next);
    if (!Successors)
      return Successors.takeError();
    B.Successors = std::move(*Successors);
  }

  // Forward dataflow to a fixpoint. The lattice is bounded: a pair holds at
  // most kMaxSetPcTargets offsets and one incomplete bit, and a join only ever
  // adds to that, so the worklist runs dry.
  //
  // Control enters at the entry block, which need not be the lowest-addressed
  // one: a callee followed into the decode may sit below its caller.
  auto Entry = BlockOf.find(EntryOffset);
  assert(Entry != BlockOf.end() && "the entry offset leads a block");
  unsigned EntryIdx = Entry->second;
  Blocks[EntryIdx].Reachable = true;
  SmallVector<unsigned> Worklist{EntryIdx};
  DenseSet<unsigned> Queued{EntryIdx};
  while (!Worklist.empty()) {
    unsigned I = Worklist.pop_back_val();
    Queued.erase(I);

    BlockFacts Exit = Blocks[I].Entry;
    for (const auto &Written : Blocks[I].Effect) {
      PairFacts Facts;
      if (Written.second)
        Facts.Values.push_back(*Written.second);
      else
        Facts.Unknown = true;
      Exit[Written.first] = std::move(Facts);
    }

    for (uint64_t Successor : Blocks[I].Successors) {
      auto It = BlockOf.find(Successor);
      if (It == BlockOf.end())
        continue;
      Block &Succ = Blocks[It->second];
      if (!joinBlockFacts(Succ.Entry, Succ.Reachable, Exit))
        continue;
      if (Queued.insert(It->second).second)
        Worklist.push_back(It->second);
    }
  }

  // The pairs the deferred sites read are untouched by their own blocks, so
  // what reaches each block is what its transfer reads.
  for (const DeferredSite &Site : Deferred) {
    SetPcSite Classified;
    const Block &B = Blocks[Site.BlockIdx];
    const PairFacts *Facts = B.Entry.find(Site.Pair) == B.Entry.end()
                                 ? nullptr
                                 : &B.Entry.find(Site.Pair)->second;
    if (!B.Reachable || !Facts) {
      Classified.RefusalReason =
          (Twine("reads ") + pairName(Site.Pair) +
           ", which nothing reaching its block gives a source offset")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }
    if (Facts->Unknown) {
      Classified.RefusalReason =
          (Twine("reads ") + pairName(Site.Pair) +
           ", which some path reaching its block leaves without a source "
           "offset")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }
    if (Facts->Overflowed) {
      Classified.RefusalReason =
          (Twine("reads ") + pairName(Site.Pair) + ", which more than " +
           Twine(kMaxSetPcTargets) + " source offsets reach")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      continue;
    }

    assert(!Facts->Values.empty() && "a complete pair names an offset");
    const uint64_t *Undecoded = find_if(Facts->Values, [&](uint64_t Value) {
      return !InstOffsets.contains(Value);
    });
    if (Undecoded != Facts->Values.end()) {
      Classified.RefusalReason =
          (Twine("reaches source offset 0x") + Twine::utohexstr(*Undecoded) +
           ", which no decoded instruction starts at")
              .str();
      Result.Sites[Site.Offset] = std::move(Classified);
      for (uint64_t Value : Facts->Values)
        if (!InstOffsets.contains(Value))
          Result.UndecodedTargets.insert(Value);
      continue;
    }

    if (Facts->Values.size() == 1) {
      Classified.SiteKind = SetPcSite::Kind::Direct;
      Classified.DirectTarget = Facts->Values.front();
    } else {
      Classified.SiteKind = SetPcSite::Kind::Enumerated;
      Classified.Targets = Facts->Values;
    }
    Result.ExtraBlockStarts.insert(Facts->Values.begin(), Facts->Values.end());
    Result.Sites[Site.Offset] = std::move(Classified);
  }

  return Error::success();
}

} // namespace

Expected<SetPcAnalysis> analyzeSetPc(ArrayRef<DecodedInst> Insts,
                                     const std::set<uint64_t> &BlockStarts,
                                     uint64_t EntryOffset, const MCState &Mc) {
  SetPcAnalysis Result;
  if (Insts.empty())
    return Result;

  DenseSet<uint64_t> InstOffsets;
  InstOffsets.reserve(Insts.size());
  for (const DecodedInst &Di : Insts)
    InstOffsets.insert(Di.Offset);

  // A transfer ends the block it sits in, so what follows it leads a block of
  // its own. For a call that block is where the callee returns to; for a jump
  // nothing reaches it, but the instructions there still need a block to be
  // raised into. The offset one past the last instruction leads nothing.
  //
  // Walking the blocks this way also keeps each transfer the last instruction
  // of its block, so the chain state a transfer reads is the state of the
  // instructions before it and of nothing else.
  std::set<uint64_t> WalkBlockStarts(BlockStarts.begin(), BlockStarts.end());
  // Control enters here, so this leads a block however the decode was split.
  WalkBlockStarts.insert(EntryOffset);
  for (const DecodedInst &Di : Insts) {
    if (!isRegisterIndirectTransfer(Di.CanonOp))
      continue;
    uint64_t Fallthrough = Di.Offset + Di.sizeInBytes();
    if (!InstOffsets.contains(Fallthrough))
      continue;
    WalkBlockStarts.insert(Fallthrough);
    Result.ExtraBlockStarts.insert(Fallthrough);
  }

  // Classifying a transfer both names a block the walk did not split at and
  // draws an edge the walk did not have. Either changes what reaches the
  // transfers, so the walk repeats until it learns nothing new. Both the block
  // starts and the edges only ever grow, and there are finitely many of each,
  // so the rounds run out.
  TransferTargets KnownTargets;
  for (;;) {
    Result.Sites.clear();
    Result.UndecodedTargets.clear();
    if (Error E = classifyAgainstBlockStarts(Insts, WalkBlockStarts,
                                             InstOffsets, KnownTargets,
                                             EntryOffset, *Mc.RegInfo, Result))
      return std::move(E);

    bool Learned = false;
    for (uint64_t Start : Result.ExtraBlockStarts)
      Learned |= WalkBlockStarts.insert(Start).second;
    for (const auto &Site : Result.Sites) {
      DenseSet<uint64_t> &Targets = KnownTargets[Site.first];
      if (Site.second.SiteKind == SetPcSite::Kind::Direct)
        Learned |= Targets.insert(Site.second.DirectTarget).second;
      for (uint64_t Target : Site.second.Targets)
        Learned |= Targets.insert(Target).second;
    }
    if (!Learned)
      return Result;
  }
}

} // namespace COMGR::transpiler
