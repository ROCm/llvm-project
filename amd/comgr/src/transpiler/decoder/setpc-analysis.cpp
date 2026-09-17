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

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdint>
#include <optional>

using namespace llvm;

namespace COMGR::transpiler {

namespace {

// Index of the SGPR that `Reg` names, or nullopt when it is not one of the
// general scalar registers the analysis follows. A pair is named by its low
// register, which is the index this returns for either width.
std::optional<unsigned> sgprIndex(const MCRegisterInfo &MRI, MCRegister Reg) {
  if (!Reg)
    return std::nullopt;
  MCRegister Low = MRI.getSubReg(Reg, AMDGPU::sub0);
  Low = stripRegEncoding(Low ? Low : Reg);

  if (!MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Low))
    return std::nullopt;
  // That class also holds the two halves of the condition register, which
  // cannot carry a captured program counter.
  if (Low == AMDGPU::VCC_LO || Low == AMDGPU::VCC_HI)
    return std::nullopt;
  return MRI.getEncodingValue(Low) & AMDGPU::HWEncoding::REG_IDX_MASK;
}

// What the analysis knows about one scalar register within a block.
struct RegState {
  // Whether the block writes this register at all.
  bool Written = false;
  // The constant that the block folded into this register.
  std::optional<uint32_t> Constant;
  // The source offset carried by the pair that starts at this register.
  std::optional<uint64_t> PairOffset;
};

// What the analysis knows about the scalar registers while walking one block.
// Everything here is block-local: an offset is only followed from the capture
// that starts it to the transfer that reads it, both within one block.
struct BlockState {
  // Record that the block writes the register at `Idx`. The write drops the
  // constant that the register held and the offsets of both pairs it belongs
  // to, the pair it starts and the pair it ends. The constant at `Idx - 1`
  // belongs to a different register and stands.
  void writeReg(unsigned Idx) {
    RegState &Reg = Regs[Idx];
    Reg.Written = true;
    Reg.Constant.reset();
    Reg.PairOffset.reset();
    if (Idx > 0)
      Regs[Idx - 1].PairOffset.reset();
    // A pending carry belongs to the value that the low add produced, and this
    // write replaces that value.
    if (CarryPair == Idx)
      CarryPair.reset();
  }

  // Everything the block did to each register, keyed by register index.
  // Registers the block has not touched read back as a default RegState.
  DenseMap<unsigned, RegState> Regs;
  // The pair whose low-half add left its carry in SCC, for as long as SCC
  // still holds it. Only the high-half add of that same pair may consume it.
  std::optional<unsigned> CarryPair;
};

// Record the write to every scalar register that `Di` defines. The walk below
// returns before reaching this for every instruction it models, so whatever
// gets here has an unknown effect on a captured program counter.
void writeDefs(const DecodedInst &Di, const MCRegisterInfo &MRI,
               BlockState &State) {
  for (unsigned I = 0; I != Di.NumDefs && I != Di.numOperands(); ++I) {
    if (!Di.isReg(I))
      continue;
    MCRegister Reg = Di.getReg(I);
    // A destination wider than one register covers several of the indices that
    // key the state, so writing it writes each register it spans.
    if (std::optional<unsigned> Idx = sgprIndex(MRI, Reg))
      State.writeReg(*Idx);
    for (MCPhysReg Sub : MRI.subregs(Reg))
      if (std::optional<unsigned> Idx = sgprIndex(MRI, Sub))
        State.writeReg(*Idx);
  }
}

// The value of operand `Index` when it is a compile-time constant. Every
// operand this is asked about belongs to a 32-bit scalar add, so a constant
// there always fits in 32 bits, signed or unsigned.
std::optional<uint32_t> immediate32(const MCInst &Inst, unsigned Index) {
  std::optional<int64_t> Value = evalOperandAsConst(Inst, Index);
  if (!Value)
    return std::nullopt;
  assert((isInt<32>(*Value) || isUInt<32>(*Value)) &&
         "32-bit scalar operand holds a value wider than 32 bits");
  return static_cast<uint32_t>(*Value);
}

// Whether `Op` transfers control through a register value.
bool isRegisterIndirectTransfer(CanonicalOp Op) {
  return Op == CanonicalOp::S_SETPC_B64 || Op == CanonicalOp::S_SWAPPC_B64;
}

// The destination register index of an instruction that writes one scalar
// destination, or nullopt when the destination is not a register the analysis
// follows.
std::optional<unsigned> destIndex(const DecodedInst &Di,
                                  const MCRegisterInfo &MRI) {
  assert(Di.NumDefs >= 1 && Di.numOperands() >= 1 && Di.isReg(0) &&
         "scalar destination expected in operand 0");
  return sgprIndex(MRI, Di.getReg(0));
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

  // A transfer ends the block it sits in, so whatever follows it leads a block
  // of its own. For a call that block is where the callee returns to; for a
  // jump nothing reaches it, but the instructions there still need a block to
  // be raised into. The offset one past the last instruction leads nothing.
  //
  // Walking the blocks this way also keeps each transfer at the end of its
  // block, so a transfer reads exactly the state that the instructions before
  // it left behind.
  DenseSet<uint64_t> WalkBlockStarts(llvm::from_range, BlockStarts);
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
  auto resolveTransfer = [&](const DecodedInst &Di, const BlockState &State) {
    SetPcSite Site = SetPcUnresolvable{SetPcRefusal::NotARegisterPair, 0};
    unsigned SourceIndex = Di.FirstSrcIdx;
    std::optional<unsigned> Source;
    if (SourceIndex < Di.numOperands() && Di.isReg(SourceIndex))
      Source = sgprIndex(MRI, Di.getReg(SourceIndex));

    if (Source) {
      const RegState &Low = State.Regs.lookup(*Source);
      if (!Low.PairOffset) {
        bool Written = Low.Written || State.Regs.lookup(*Source + 1).Written;
        Site =
            SetPcUnresolvable{Written ? SetPcRefusal::PairWrittenWithoutOffset
                                      : SetPcRefusal::PairNeverHeldOffset,
                              *Source};
      } else if (!InstOffsets.contains(*Low.PairOffset)) {
        Site = SetPcUnresolvable{SetPcRefusal::TargetNotAnInstruction,
                                 *Low.PairOffset};
      } else {
        Site = SetPcDirect{*Low.PairOffset};
        Result.ExtraBlockStarts.insert(*Low.PairOffset);
      }
    }

    Result.Sites.insert({Di.Offset, Site});
  };

  BlockState State;
  for (const DecodedInst &Di : Insts) {
    if (WalkBlockStarts.contains(Di.Offset))
      State = BlockState();

    // SCC holds the carry out of a low-half add only until the next
    // instruction writes it, so anything that writes SCC ends the relation.
    // Only which pair owns the carry matters here; its value depends on the
    // program counter, which the analysis never knows. The low-half add below
    // re-establishes ownership for the pair it just wrote.
    std::optional<unsigned> Carry = State.CarryPair;
    if (Di.defsScc())
      State.CarryPair.reset();

    switch (Di.CanonOp) {
    case CanonicalOp::S_GETPC_B64: {
      // The capture names the offset of the instruction that follows it.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Dst)
        break;
      State.writeReg(*Dst);
      State.writeReg(*Dst + 1);
      State.Regs[*Dst].PairOffset = Di.Offset + Di.sizeInBytes();
      continue;
    }

    case CanonicalOp::S_ADD_U32: {
      // A displacement added to the offset that a pair already carries, or a
      // plain constant fold that a later add can use as its addend.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Dst)
        break;
      unsigned Src0 = Di.FirstSrcIdx;
      unsigned Src1 = Src0 + 1;
      assert(Src1 < Di.numOperands() && "scalar add has two source operands");
      std::optional<uint32_t> Src0Imm = immediate32(Di.Inst, Src0);
      std::optional<uint32_t> Src1Imm = immediate32(Di.Inst, Src1);
      if (Src0Imm && Src1Imm) {
        // A fold of two constants. No displacement chain can continue from its
        // carry out, and the clear above already gave up ownership of SCC, so
        // only the folded value is worth recording.
        State.writeReg(*Dst);
        State.Regs[*Dst].Constant = *Src0Imm + *Src1Imm;
        continue;
      }
      std::optional<unsigned> Src0Idx;
      if (Di.isReg(Src0))
        Src0Idx = sgprIndex(MRI, Di.getReg(Src0));
      if (Src0Idx != Dst)
        break;
      std::optional<uint64_t> Base = State.Regs.lookup(*Dst).PairOffset;
      if (!Base)
        break;
      std::optional<uint32_t> Addend = Src1Imm;
      if (!Addend && Di.isReg(Src1)) {
        std::optional<unsigned> Src1Idx = sgprIndex(MRI, Di.getReg(Src1));
        if (Src1Idx)
          Addend = State.Regs.lookup(*Src1Idx).Constant;
      }
      if (!Addend)
        break;
      uint64_t Target = *Base + *Addend;
      State.writeReg(*Dst);
      State.Regs[*Dst].PairOffset = Target;
      // A high-half add may now consume the carry out of this add.
      State.CarryPair = *Dst;
      continue;
    }

    case CanonicalOp::S_ADDC_U32: {
      // High half of a split displacement. This adds SCC, so it continues the
      // low half only while SCC still holds that half's carry, and `Carry`
      // names the pair that carry belongs to. Everything else that writes the
      // same high register is an ordinary write, whichever pair it sits in.
      //
      // The low add already folded the whole displacement into the offset
      // whenever the source offset stays within four gigabytes of the capture,
      // so this only adds the high addend.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Carry || Dst != *Carry + 1)
        break;
      unsigned Low = *Carry;
      std::optional<uint64_t> Target = State.Regs.lookup(Low).PairOffset;
      if (!Target)
        break;
      unsigned Src0 = Di.FirstSrcIdx;
      unsigned Src1 = Src0 + 1;
      assert(Src1 < Di.numOperands() && "scalar add has two source operands");
      if (!Di.isReg(Src0) || sgprIndex(MRI, Di.getReg(Src0)) != Dst)
        break;
      std::optional<uint32_t> Addend = immediate32(Di.Inst, Src1);
      if (!Addend)
        break;
      State.writeReg(*Dst);
      State.Regs[Low].PairOffset =
          *Target + (static_cast<uint64_t>(*Addend) << 32);
      continue;
    }

    case CanonicalOp::S_ADD_NC_U64: {
      // The whole displacement folded into one add. It commutes, so the pair
      // carrying the offset stands on either side of the constant.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Dst)
        break;
      std::optional<uint64_t> Base = State.Regs.lookup(*Dst).PairOffset;
      if (!Base || Di.SrcMap.size() < 2)
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
      uint64_t Target = *Base + static_cast<uint64_t>(*Displacement);
      State.writeReg(*Dst);
      State.writeReg(*Dst + 1);
      State.Regs[*Dst].PairOffset = Target;
      continue;
    }

    case CanonicalOp::S_SETPC_B64:
      resolveTransfer(Di, State);
      continue;

    case CanonicalOp::S_SWAPPC_B64: {
      resolveTransfer(Di, State);
      // The call leaves the return offset in its destination, so whatever the
      // pair held is gone.
      if (std::optional<unsigned> Dst = destIndex(Di, MRI)) {
        State.writeReg(*Dst);
        State.writeReg(*Dst + 1);
      }
      continue;
    }

    default:
      break;
    }

    writeDefs(Di, MRI, State);
  }

  return Result;
}

} // namespace COMGR::transpiler
