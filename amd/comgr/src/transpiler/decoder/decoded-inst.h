//===- decoded-inst.h - Transpiler ----------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_DECODED_INST_H
#define TRANSPILER_DECODED_INST_H

#include "transpiler/decoder/canonical-op.h"

#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>

namespace COMGR::transpiler {

// If operand `OpIdx` of `Inst` is a compile-time constant -- an immediate, or
// an expression that folds to one -- returns its value; otherwise std::nullopt
// (including when `OpIdx` is out of range).
inline std::optional<int64_t> evalOperandAsConst(const llvm::MCInst &Inst,
                                                 unsigned OpIdx) {
  if (OpIdx >= Inst.getNumOperands())
    return std::nullopt;
  const llvm::MCOperand &Operand = Inst.getOperand(OpIdx);
  if (Operand.isImm())
    return Operand.getImm();
  if (Operand.isExpr()) {
    int64_t Val = 0;
    if (Operand.getExpr()->evaluateAsAbsolute(Val))
      return Val;
  }
  return std::nullopt;
}

// One decoded source-ISA instruction: the MC instruction plus the facts the
// raiser dispatches on. The mnemonic is not stored -- reconstruct it on demand
// from `Inst` with strippedMnemonic (see mc-state.h) where a diagnostic needs
// it.
struct DecodedInst {
  llvm::MCInst Inst;
  CanonicalOp CanonOp = CanonicalOp::Unknown;

  // The instruction's MCInstrDesc::TSFlags, whose AMDGPU-specific bits carry
  // the instruction-format the raiser dispatches on (see amdgpu-formats.h).
  uint64_t TargetSpecificFlags = 0;

  // Byte offset of this instruction within the kernel's .text.
  uint64_t Offset = 0;

  // Number of leading MCInst operands that are definitions (results); the
  // logical sources start at operand index FirstSrcIdx.
  unsigned NumDefs = 0;
  unsigned FirstSrcIdx = 0;

  // Operand index of each logical source, and of its source modifier (or
  // UINT_MAX when the source has none). Parallel, one entry per source.
  llvm::SmallVector<unsigned> SrcMap;
  llvm::SmallVector<unsigned> ModMap;

  // Structural view of a VOPD packet. A packet contains two component VOPs
  // in one MCInst, so the ordinary one-opcode CanonOp/SrcMap view cannot
  // describe it. Operand indices still refer to Inst and therefore retain
  // the register classes and S_SET_VGPR_MSB adjustments of the full packet.
  struct VOPDHalf {
    CanonicalOp CanonOp = CanonicalOp::Unknown;
    unsigned SrcIdx[3] = {};

    unsigned destinationIndex() const {
      return llvm::Bitfield::get<DstIdxField>(DstAndFlags);
    }
    void setDestinationIndex(unsigned I) {
      assert(I < (1u << 29) && "VOPD destination index does not fit");
      llvm::Bitfield::set<DstIdxField>(DstAndFlags, I);
    }

    unsigned numSources() const {
      return llvm::Bitfield::get<NumSrcsField>(DstAndFlags);
    }
    unsigned appendSource(unsigned OperandIdx) {
      unsigned I = numSources();
      assert(I < 3 && "too many VOPD component sources");
      SrcIdx[I] = OperandIdx;
      llvm::Bitfield::set<NumSrcsField>(DstAndFlags, I + 1);
      return I;
    }

    uint8_t sourceModifier(unsigned I) const {
      assert(I < 3 && "VOPD source modifier index out of range");
      return static_cast<uint8_t>(PackedModifiers >> (I * 8));
    }
    void setSourceModifier(unsigned I, uint8_t Mods) {
      assert(I < 3 && "VOPD source modifier index out of range");
      const unsigned Shift = I * 8;
      PackedModifiers = (PackedModifiers & ~(UINT32_C(0xff) << Shift)) |
                        (static_cast<uint32_t>(Mods) << Shift);
    }

    bool hasBitOp3() const {
      return llvm::Bitfield::get<HasBitOp3Field>(DstAndFlags);
    }
    uint8_t bitOp3() const {
      return static_cast<uint8_t>(PackedModifiers >> 24);
    }
    void setBitOp3(uint8_t TruthTable) {
      llvm::Bitfield::set<HasBitOp3Field>(DstAndFlags, true);
      PackedModifiers = (PackedModifiers & UINT32_C(0x00ffffff)) |
                        (static_cast<uint32_t>(TruthTable) << 24);
    }

  private:
    // MC operand indices are far smaller than 2^29. Pack the destination next
    // to the two-bit source count and bitop-presence flag, and store the three
    // source modifiers and eight-bit bitop truth table in one word.
    using DstIdxField = llvm::Bitfield::Element<unsigned, 0, 29>;
    using NumSrcsField = llvm::Bitfield::Element<unsigned, 29, 2>;
    using HasBitOp3Field = llvm::Bitfield::Element<bool, 31, 1>;
    uint32_t DstAndFlags = 0;
    uint32_t PackedModifiers = 0;
  };

  std::optional<std::array<VOPDHalf, 2>> VOPD;

  // Instruction length in bytes. The AMDGPU maximum is 20
  // (AMDGPUMCAsmInfo::MaxInstLength), so five bits suffice.
  unsigned sizeInBytes() const { return llvm::Bitfield::get<SizeField>(Flags); }
  void setSizeInBytes(unsigned Bytes) {
    llvm::Bitfield::set<SizeField>(Flags, Bytes);
  }

  // Whether the instruction writes the SCC / VCC / EXEC condition registers.
  bool defsScc() const { return llvm::Bitfield::get<DefsSccField>(Flags); }
  void setDefsScc(bool V) { llvm::Bitfield::set<DefsSccField>(Flags, V); }
  bool defsVcc() const { return llvm::Bitfield::get<DefsVccField>(Flags); }
  void setDefsVcc(bool V) { llvm::Bitfield::set<DefsVccField>(Flags, V); }
  bool defsExec() const { return llvm::Bitfield::get<DefsExecField>(Flags); }
  void setDefsExec(bool V) { llvm::Bitfield::set<DefsExecField>(Flags, V); }

  unsigned numOperands() const { return Inst.getNumOperands(); }
  bool isReg(unsigned I) const {
    assert(I < numOperands() && "operand index out of range");
    return Inst.getOperand(I).isReg();
  }
  bool isImm(unsigned I) const {
    assert(I < numOperands() && "operand index out of range");
    return Inst.getOperand(I).isImm();
  }
  unsigned getReg(unsigned I) const { return Inst.getOperand(I).getReg(); }
  int64_t getImm(unsigned I) const { return Inst.getOperand(I).getImm(); }

private:
  // `Flags` byte layout: the 5-bit instruction size packed next to the three
  // condition-register def bits.
  using SizeField = llvm::Bitfield::Element<unsigned, 0, 5>;
  using DefsSccField = llvm::Bitfield::Element<bool, 5, 1>;
  using DefsVccField = llvm::Bitfield::Element<bool, 6, 1>;
  using DefsExecField = llvm::Bitfield::Element<bool, 7, 1>;
  uint8_t Flags = 0;
};

} // namespace COMGR::transpiler

#endif
