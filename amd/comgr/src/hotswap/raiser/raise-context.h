//===- raise-context.h - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_CONTEXT_H
#define HOTSWAP_TRANSPILER_RAISE_CONTEXT_H

#include "hotswap/common/kernel-meta.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/loader/code-object-utils.h"
#include "raise_failure.h"
#include "reg-file.h"
#include "user-sgpr-layout.h"
#include "wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <climits>
#include <cstdint>
#include <optional>
#include <utility>

namespace COMGR::hotswap {

// Shared state threaded through every format handler.
class RaiseContext {
public:
  // Build the context for the source kernel described by Meta. B must be
  // positioned in the entry block: the register file and the cross-block
  // shadow storage are allocated there. Fails when the kernel descriptor and
  // the metadata disagree on the user-SGPR layout.
  static llvm::Expected<RaiseContext>
  create(llvm::IRBuilder<> &B, const WaveProjection &Projection,
         const MCState &MC, const KernelMeta &Meta,
         llvm::DenseMap<uint64_t, llvm::BasicBlock *> OffsetToBb,
         llvm::ArrayRef<uint8_t> SourceTextBytes,
         uint64_t SourceTextBaseAddress,
         llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
         uint64_t KernelStartOffset, uint64_t KernelEndOffset);

  // Builder every handler emits into. Its insertion point moves as raising
  // progresses.
  llvm::IRBuilder<> &B;
  // Translation between the source and target wave sizes.
  const WaveProjection &Projection;
  // MC layer for the source ISA, shared by every kernel in the code object.
  const MCState &MC;

  // Alloca-backed storage for the source architectural registers.
  AllocaRegFile &regs() { return Regs; }
  // Source user-SGPR layout derived from the kernel descriptor.
  const UserSgprLayout &layout() const { return Layout; }

  // Source text section, and the address the source code object loads it at.
  // PC-relative literals are materialized by reading out of these.
  llvm::ArrayRef<uint8_t> sourceTextBytes() const { return SourceTextBytes; }
  uint64_t sourceTextBaseAddress() const { return SourceTextBaseAddress; }
  // Source code-object sections a proven PC-relative address can land in.
  llvm::ArrayRef<TextSection::ImageSection> sourceImageSections() const {
    return SourceImageSections;
  }

  // Offset of the source kernel's first byte within the source text section.
  uint64_t kernelStartOffset() const { return KernelStartOffset; }
  // Offset one past the source kernel's last byte, or 0 when the kernel runs
  // to the end of the source text section.
  uint64_t kernelEndOffset() const { return KernelEndOffset; }

  // Source scratch allocation, disjoint from target spills. Null until a
  // handler needs source scratch.
  llvm::AllocaInst *scratchPrivateSegmentAlloca() const {
    return ScratchPrivateSegmentAlloca;
  }
  void setScratchPrivateSegmentAlloca(llvm::AllocaInst *Alloca) {
    ScratchPrivateSegmentAlloca = Alloca;
  }
  // Whether the raised kernel reads or writes the source private segment.
  bool usesScratchPrivateSegment() const {
    return ScratchPrivateSegmentAlloca != nullptr;
  }

  // Active low byte of S_SET_VGPR_MSB. Each two-bit field selects the high
  // VGPR bank for a format-defined operand slot.
  uint8_t vgprMsBs() const { return VgprMsBs; }
  void setVgprMsBs(uint8_t Value) { VgprMsBs = Value; }

  // VGPR index adjustments for the instruction passed to computeVGPRAdjust,
  // indexed by MC operand index.
  llvm::ArrayRef<unsigned> currentVgprAdjust() const {
    return CurrentVgprAdjust;
  }

  // Compute VGPR bank adjustments for the instruction's format-defined slots.
  void computeVGPRAdjust(const DecodedInst &Di);

  // Return the block raised from the source instruction at Addr. A missing
  // block is a raiser bug and aborts.
  llvm::BasicBlock *lookupBB(uint64_t Addr);

  // Resolve the register operand at OperandIndex to the source register it
  // names. Fails on a register this raiser does not model.
  llvm::Expected<ParsedReg> parseReg(const DecodedInst &Di,
                                     unsigned OperandIndex) const;

  // Read the operand at OpIdx as a 32-bit value, resolving registers through
  // the reg-file and immediates through the MC operand.
  llvm::Expected<llvm::Value *> readOp32(const DecodedInst &Di, unsigned OpIdx);
  // Read the operand at OpIdx as a 64-bit value, pairing adjacent registers.
  llvm::Expected<llvm::Value *> readOp64(const DecodedInst &Di, unsigned OpIdx);
  // Read a mask at target EXEC width, replicating narrower source-wave bits.
  llvm::Expected<llvm::Value *> readOpExecWidth(const DecodedInst &Di,
                                                unsigned OpIdx);
  // Read the mask a source-wave instruction should see, e.g. for `v_mbcnt_lo`.
  // EXEC/VCC/SGPR-shadow masks are projected; scalars use readOp32.
  llvm::Expected<llvm::Value *> readOpSourceWaveMask32(const DecodedInst &Di,
                                                       unsigned OpIdx);

  // Target-hardware lane id (i32), emitted once per kernel and reused.
  llvm::Value *emitLaneIdx();

  // Freeze per-lane addresses when widening wave32 to wave64. New target lanes
  // may hold poison from an earlier inactive definition, which would make even
  // an EXEC-predicated memory operation undefined. Other wave-size directions
  // return the address unchanged.
  llvm::Value *freezeMemAddr(llvm::Value *Addr);

  // Return whether the current target lane is active in the logical EXEC mask.
  // The result is cached for one source instruction and invalidated by EXEC
  // writes.
  llvm::Value *emitLaneActiveBit();

  // Invalidate cached lane activity after an EXEC write or instruction
  // boundary.
  void resetLaneActiveCache() { CachedLaneActive = nullptr; }

  // Store EXEC and invalidate cached lane activity.
  void storeExec(llvm::Value *V) {
    Regs.storeExec(B, V);
    resetLaneActiveCache();
  }

  // Write V to the register Pr names, at the register's width. VGPR and AGPR
  // writes are predicated on EXEC; scalar writes commit for the whole wave.
  void writeReg32(ParsedReg Pr, llvm::Value *V);
  void writeReg64(ParsedReg Pr, llvm::Value *V);
  // Write a value spanning more than two dwords, such as an image descriptor.
  void writeRegVec(ParsedReg Pr, llvm::Value *V);
  // Write a wave mask at the target EXEC width.
  void writeRegExecWidth(ParsedReg Pr, llvm::Value *V);

  // Write V to a vector register by resolved index, predicated on EXEC.
  void storeVGPR32(unsigned Idx, llvm::Value *V);
  void storeVGPR64(unsigned Idx, llvm::Value *V);
  void storeAGPR32(unsigned Idx, llvm::Value *V);

  // Emit Body in a lane-active control-flow diamond and leave the builder at
  // its merge block. This preserves inactive lanes for per-lane side effects.
  void emitUnderExec(llvm::function_ref<void()> Body);

  // Record CmpI1 as the per-lane compare a V_CMP wrote to SGPR BaseIdx, both
  // for reuse within the block and in the cross-block shadow storage. IsPair
  // says whether the destination spans BaseIdx and its successor.
  void recordSgprWaveMaskI1(unsigned BaseIdx, llvm::Value *CmpI1, bool IsPair);

  // Emit a test of whether the source wave holding the current target lane has
  // any lane active in EXEC.
  llvm::Value *emitCurrentSourceWaveHasActiveLane();

  // Record V as the value SGPR pair BaseIdx holds for the source wave holding
  // the current target lane, leaving the value recorded for other source waves
  // in place. Does nothing unless the projection packs whole source waves.
  void recordSourceWaveSgprPair(unsigned BaseIdx, llvm::Value *V);

  // Emit a read of the value recorded for SGPR pair BaseIdx, or Fallback when
  // nothing was recorded.
  llvm::Value *materializeSourceWaveSgprPair(unsigned BaseIdx,
                                             llvm::Value *Fallback);

  // Return the compare recorded for SGPR BaseIdx in this block, or null when
  // none is valid.
  llvm::Value *lookupSgprWaveMaskI1(unsigned BaseIdx) const {
    auto It = LastSgprWaveMaskI1.find(BaseIdx);
    return It == LastSgprWaveMaskI1.end() ? nullptr : It->second.I1;
  }

  // Emit a read of the wave mask shadowed for SGPR BaseIdx, and of the bit
  // saying whether that shadow is valid. Both return null for an SGPR with no
  // shadow storage.
  llvm::Value *loadSgprWaveMaskExec(unsigned BaseIdx) const;
  llvm::Value *loadSgprWaveMaskValid(unsigned BaseIdx) const;

  // Invalidate the facts a write to SGPR BaseIdx invalidates. A pair is keyed
  // by its low SGPR, so writing its high SGPR also invalidates the preceding
  // entry. Single-SGPR entries remain independent.
  void invalidateSgprWaveMaskI1(unsigned BaseIdx);

  // Drop the facts whose SSA values cannot cross a block boundary.
  void clearSgprWaveMaskShadow() {
    LastSgprWaveMaskI1.clear();
    SourceImageSgprPairAddrShadow.clear();
  }

  // Record that SGPR pair BaseIdx holds source code-object address Value.
  void recordSourceImageSgprPairAddr(unsigned BaseIdx, uint64_t Value) {
    SourceImageSgprPairAddrShadow[BaseIdx] = Value;
  }

  // Return the source code-object address recorded for SGPR pair BaseIdx in
  // this block, if any.
  std::optional<uint64_t> lookupSourceImageSgprPairAddr(unsigned BaseIdx) const;

  // Track the value written to M0, which V_MOVREL needs as a constant to
  // resolve its VGPR index while the indexed register file is built. A
  // non-constant write, and any block boundary, gives up the constant.
  void updateM0Const(llvm::Value *V);
  void clearM0Const() { M0Const = std::nullopt; }
  std::optional<uint64_t> getM0Const() const { return M0Const; }

  // Emit stores marking every cross-block SGPR shadow invalid.
  void invalidateSgprShadows();

  // Append every alloca backing the context, register file included, to Out
  // for SSA promotion.
  void collectAllocas(llvm::SmallVectorImpl<llvm::AllocaInst *> &Out) const;

private:
  RaiseContext(llvm::IRBuilder<> &B, const WaveProjection &Projection,
               const MCState &MC, UserSgprLayout Layout,
               llvm::DenseMap<uint64_t, llvm::BasicBlock *> OffsetToBb,
               llvm::ArrayRef<uint8_t> SourceTextBytes,
               uint64_t SourceTextBaseAddress,
               llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
               uint64_t KernelStartOffset, uint64_t KernelEndOffset);

  // Storage shadowing one SGPR across block boundaries.
  struct SgprShadow {
    // Wave mask last written to this SGPR, at target EXEC width.
    llvm::AllocaInst *WaveMask;
    // Whether WaveMask still describes what the SGPR holds.
    llvm::AllocaInst *WaveMaskValid;
    // Whether the mask spans this SGPR and its successor.
    llvm::AllocaInst *WaveMaskIsPair;
    // Value this SGPR pair holds for the current source wave.
    llvm::AllocaInst *SourceWavePair;
    // Whether SourceWavePair was ever written for this source wave.
    llvm::AllocaInst *SourceWavePairValid;
  };

  // A V_CMP result kept for reuse while the SGPR it wrote remains valid.
  struct WaveMaskEntry {
    llvm::Value *I1 = nullptr;
    // Whether the destination spans this SGPR and its successor.
    bool IsPair = false;
  };

  // Source architectural registers, allocated in the entry block.
  AllocaRegFile Regs;
  // What each SGPR holds at source kernel entry.
  UserSgprLayout Layout;
  // Block raised from each source instruction offset that starts one.
  llvm::DenseMap<uint64_t, llvm::BasicBlock *> OffsetToBb;

  // Source code object, read to materialize proven PC-relative literals.
  llvm::ArrayRef<uint8_t> SourceTextBytes;
  uint64_t SourceTextBaseAddress = 0;
  llvm::ArrayRef<TextSection::ImageSection> SourceImageSections;

  // Extent of the source kernel within the source text section.
  uint64_t KernelStartOffset = 0;
  uint64_t KernelEndOffset = 0;

  // Allocation backing the source private segment, made on first use.
  llvm::AllocaInst *ScratchPrivateSegmentAlloca = nullptr;
  // Active low byte of S_SET_VGPR_MSB.
  uint8_t VgprMsBs = 0;
  // VGPR bank adjustment per MC operand of the instruction being raised.
  llvm::SmallVector<unsigned> CurrentVgprAdjust;

  // Lane-active bit reused until an EXEC write or instruction boundary.
  llvm::Value *CachedLaneActive = nullptr;
  // Block-local compares, keyed by the SGPR they were written to.
  llvm::DenseMap<unsigned, WaveMaskEntry> LastSgprWaveMaskI1;
  // Block-local source-image addresses proven for PC-relative literal loads.
  llvm::DenseMap<unsigned, uint64_t> SourceImageSgprPairAddrShadow;
  // Block-local constant value last stored to M0.
  std::optional<uint64_t> M0Const;

  // Shadow storage per SGPR. Cross-block values live in allocas to avoid
  // carrying SSA values that do not dominate their uses.
  llvm::SmallVector<SgprShadow> SgprShadows;
};

// Reads a handler's source operands via the decoded srcMap, applying VOP3
// neg/abs modifiers on the float paths.
struct OpResolver {
  // Context the operands are read through.
  RaiseContext &Ctx;
  // Instruction whose operands are being read.
  const DecodedInst &Di;

  // MC operand index of the I-th source.
  unsigned srcIdx(unsigned I) const {
    assert(I < Di.SrcMap.size() && "source index out of range");
    return Di.SrcMap[I];
  }
  // Number of sources the instruction takes.
  unsigned nSrcs() const { return static_cast<unsigned>(Di.SrcMap.size()); }

  // Modifier bits attached to the I-th source, 0 when it carries none. Bit 0
  // negates and bit 1 takes the absolute value.
  unsigned srcMod(unsigned I) const;

  // Apply the I-th source's modifiers to V, which the caller has already read.
  // An integer-typed V round-trips through float, since the modifiers are
  // defined on the float interpretation of the bits.
  llvm::Value *applyMods(unsigned I, llvm::Value *V);

  // Read the I-th source as a 32-bit value.
  llvm::Expected<llvm::Value *> src(unsigned I) {
    return Ctx.readOp32(Di, srcIdx(I));
  }
  // Read the I-th source as a 32-bit value with its modifiers applied.
  llvm::Expected<llvm::Value *> srcF(unsigned I);
  // Read the I-th source as a 64-bit value.
  llvm::Expected<llvm::Value *> src64(unsigned I) {
    return Ctx.readOp64(Di, srcIdx(I));
  }
  // Read the I-th source as a wave mask at target EXEC width.
  llvm::Expected<llvm::Value *> srcExecWidth(unsigned I) {
    return Ctx.readOpExecWidth(Di, srcIdx(I));
  }
  // Value of the I-th source, which must be an immediate.
  int64_t srcImm(unsigned I) {
    unsigned Index = srcIdx(I);
    assert(Di.isImm(Index) && "source operand must be an immediate");
    return Di.getImm(Index);
  }

  // Register the I-th destination names.
  llvm::Expected<ParsedReg> dst(unsigned I = 0) {
    assert(Di.isReg(I) && "destination operand must be a register");
    return Ctx.parseReg(Di, I);
  }
  // Whether the I-th source is a register rather than an immediate.
  bool isSrcReg(unsigned I) { return Di.isReg(srcIdx(I)); }
  // Register the I-th source names, or no value when it is an immediate.
  llvm::Expected<std::optional<ParsedReg>> srcReg(unsigned I);
};

} // namespace COMGR::hotswap

#endif
