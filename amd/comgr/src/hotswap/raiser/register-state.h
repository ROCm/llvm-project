//===- register-state.h - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_REGISTER_STATE_H
#define HOTSWAP_TRANSPILER_REGISTER_STATE_H

#include "hotswap/common/kernel-meta.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/raiser/reg-file.h"
#include "hotswap/raiser/user-sgpr-layout.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace COMGR::hotswap {

// The source architectural registers as the raiser sees them: alloca-backed
// storage, the operand reads and writes that resolve through it, EXEC
// predication of the per-lane ones, and the facts derived from a scalar write
// that a later write to the same register invalidates.
class RegisterState {
public:
  // Build the register state for the source kernel described by Meta, with the
  // SGPRs the source ABI preloads before entry already seeded. B must be
  // positioned in the entry block: the register file, the cross-block shadow
  // storage, and the seeds are emitted there. Fails when the kernel descriptor
  // and the metadata disagree on the user-SGPR layout, and when the layout
  // preloads an entry source the target cannot reproduce.
  static llvm::Expected<RegisterState> create(llvm::IRBuilder<> &B,
                                              const WaveProjection &Projection,
                                              const MCState &MC,
                                              const KernelMeta &Meta);

  // Alloca-backed storage for the source architectural registers.
  AllocaRegFile &regFile() { return Regs; }
  // What each SGPR holds at source kernel entry.
  const UserSgprLayout &layout() const { return Layout; }

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

  // Start raising a new source block, dropping every raise-time fact that does
  // not survive a block boundary: the compares and source-image addresses
  // recorded per SGPR, the M0 constant, the VGPR MSB mode, and the cached
  // lane-active bit. The alloca-backed shadow storage is unaffected.
  void enterBlock();

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
  std::optional<uint64_t> getM0Const() const { return M0Const; }

  // Emit stores marking every cross-block SGPR shadow invalid.
  void invalidateSgprShadows();

  // Append every alloca backing the register state, register file included, to
  // Out for SSA promotion.
  void collectAllocas(llvm::SmallVectorImpl<llvm::AllocaInst *> &Out) const;

private:
  RegisterState(llvm::IRBuilder<> &B, const WaveProjection &Projection,
                const MCState &MC, UserSgprLayout Layout);

  // Give the preloaded entry SGPRs the values the source ABI hands them.
  llvm::Error seedEntrySgprs();

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

  // Builder the register accesses are emitted into. Its insertion point moves
  // as raising progresses.
  llvm::IRBuilder<> &B;
  // Translation between the source and target wave sizes.
  const WaveProjection &Projection;
  // MC layer for the source ISA, shared by every kernel in the code object.
  const MCState &MC;

  // Source architectural registers, allocated in the entry block.
  AllocaRegFile Regs;
  // What each SGPR holds at source kernel entry.
  UserSgprLayout Layout;

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

} // namespace COMGR::hotswap

#endif
