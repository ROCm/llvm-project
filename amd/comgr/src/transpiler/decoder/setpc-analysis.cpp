//===- setpc-analysis.cpp - Transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "setpc-analysis.h"

#include "mc-state.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"

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

} // namespace

SetPcAnalysis analyzeSetPc(ArrayRef<DecodedInst> Insts,
                           const std::set<uint64_t> &BlockStarts,
                           const MCState &Mc) {
  SetPcAnalysis Result;
  if (Insts.empty())
    return Result;

  const MCRegisterInfo &MRI = *Mc.RegInfo;

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
  DenseSet<uint64_t> WalkBlockStarts(BlockStarts.begin(), BlockStarts.end());
  for (const DecodedInst &Di : Insts) {
    if (!isRegisterIndirectTransfer(Di.CanonOp))
      continue;
    uint64_t Fallthrough = Di.Offset + Di.sizeInBytes();
    if (!InstOffsets.contains(Fallthrough))
      continue;
    WalkBlockStarts.insert(Fallthrough);
    Result.ExtraBlockStarts.insert(Fallthrough);
  }

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
      Site.RefusalReason =
          (Twine("reads ") + pairName(*Source) + ", which " +
           (State.pairIsDirty(*Source)
                ? "its block writes without computing a source offset in it"
                : "nothing in its block gives a source offset"))
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

    Site.SiteKind = SetPcSite::Kind::Direct;
    Site.DirectTarget = Chain->Value;
    Result.Sites[Di.Offset] = std::move(Site);
    Result.ExtraBlockStarts.insert(Chain->Value);
    // The pair is consumed here; a block reached over a back edge must not see
    // the chain as still standing.
    State.invalidate(*Source);
  };

  BlockState State;
  for (const DecodedInst &Di : Insts) {
    if (WalkBlockStarts.contains(Di.Offset))
      State = BlockState();

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
      // The call leaves the return offset in its destination, so whatever the
      // pair held is gone.
      if (Di.NumDefs >= 1 && Di.numOperands() >= 1 && Di.isReg(0)) {
        if (std::optional<unsigned> Dst = sgprIndex(MRI, Di.getReg(0))) {
          State.invalidate(*Dst);
          State.invalidate(*Dst + 1);
        }
      }
      continue;
    }

    default:
      break;
    }

    invalidateDefs(Di, MRI, State);
  }

  return Result;
}

} // namespace COMGR::transpiler
