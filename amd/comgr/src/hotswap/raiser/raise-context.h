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
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/loader/code-object-utils.h"
#include "kernarg-layout.h"
#include "raise_failure.h"
#include "reg-file.h"
#include "user-sgpr-layout.h"
#include "wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <map>
#include <optional>
#include <utility>

namespace COMGR::hotswap {

// Shared state threaded through every format handler.
struct RaiseContext {
  llvm::IRBuilder<> &B;
  AllocaRegFile &Regs;
  const WaveProjection &Projection;
  const MCState &MC;
  const ISAProfile &Isa; // source ISA (for disassembly / instruction semantics)
  ISAProfile
      TargetIsa; // compilation target ISA (for code generation decisions)
  // Target hidden-argument offsets depend on the code object version.
  unsigned TargetCodeObjectVersion = 6;
  KernargLayout &Kernargs;
  // Non-owning source user-SGPR layout derived from the kernel descriptor.
  const UserSgprLayout *Layout = nullptr;
  llvm::Function *Kernel;
  llvm::BasicBlock *ThreadLoopLatch = nullptr;

  llvm::DenseMap<uint64_t, llvm::BasicBlock *> &OffsetToBb;
  // Source sections used to materialize proven PC-relative literals.
  llvm::ArrayRef<uint8_t> SourceTextBytes;
  uint64_t SourceTextBaseAddress = 0;
  llvm::ArrayRef<TextSection::ImageSection> SourceImageSections;
  uint64_t KernelStartOffset = 0;
  uint64_t KernelEndOffset = 0;

  RaiseContext(llvm::IRBuilder<> &B, AllocaRegFile &Regs,
               const WaveProjection &Projection, const MCState &MC,
               const ISAProfile &Isa, ISAProfile TargetIsa,
               unsigned TargetCodeObjectVersion, KernargLayout &Kernargs,
               const UserSgprLayout *Layout, llvm::Function *Kernel,
               llvm::BasicBlock *ThreadLoopLatch,
               llvm::DenseMap<uint64_t, llvm::BasicBlock *> &OffsetToBb,
               llvm::ArrayRef<uint8_t> SourceTextBytes,
               uint64_t SourceTextBaseAddress,
               llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
               uint64_t KernelStartOffset, uint64_t KernelEndOffset);

  // Source scratch allocation. LLVM's frame layout keeps it disjoint from
  // target spills.
  uint32_t SourcePrivateSegmentFixedSize = 0;
  uint32_t SourceComputePgmRsrc2 = 0;
  uint16_t SourceKernelCodeProperties = 0;
  bool UsesScratchPrivateSegment = false;
  llvm::AllocaInst *ScratchPrivateSegmentAlloca = nullptr;

  // Active low byte of S_SET_VGPR_MSB. Each two-bit field selects the high
  // VGPR bank for a format-defined operand slot.
  uint8_t VgprMsBs = 0;
  bool AssumeHipGlobalOffsetZero = false;

  // Per-instruction VGPR index adjustments, indexed by MC operand index.
  // Exceeding this capacity is an error rather than a truncated register index.
  static constexpr unsigned KMaxOps = 16;
  unsigned CurrentVgprAdjust[KMaxOps] = {};

  // Compute VGPR bank adjustments for the instruction's format-defined slots.
  llvm::Error computeVGPRAdjust(const DecodedInst &Di);

  llvm::BasicBlock *lookupBB(uint64_t Addr);

  ParsedReg parseReg(llvm::MCRegister Reg, int MciOpIdx = -1) const;

  // Read the operand at OpIdx as a 32-bit value, resolving registers through
  // the reg-file and immediates through the MC operand.
  llvm::Expected<llvm::Value *> readOp32(const DecodedInst &Di, unsigned OpIdx);
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
  void resetLaneActiveCache() {
    CachedLaneActive = nullptr;
    CachedLaneActiveBb = nullptr;
  }

  // Store EXEC and invalidate cached lane activity.
  void storeExec(llvm::Value *V) {
    Regs.storeExec(B, V);
    resetLaneActiveCache();
  }

  // Predicate VGPR and AGPR writes on EXEC; commit scalar writes for the wave.
  // The store methods accept an already resolved vector-register index.
  void writeReg32(ParsedReg Pr, llvm::Value *V);
  void writeReg64(ParsedReg Pr, llvm::Value *V);
  void writeRegVec(ParsedReg Pr, llvm::Value *V);
  void writeRegExecWidth(ParsedReg Pr, llvm::Value *V);
  void storeVGPR32(unsigned Idx, llvm::Value *V);
  void storeVGPR64(unsigned Idx, llvm::Value *V);
  void storeAGPR32(unsigned Idx, llvm::Value *V);

  // Provenance of one source kernarg pointer SGPR lane.
  enum class KernargPtrLaneProvenance {
    LiveEntry,
    NonEntry,
    Unknown,
  };

  // Pair provenance used to decide whether hidden-argument synthesis is safe.
  struct KernargPtrProvenance {
    KernargPtrLaneProvenance Low = KernargPtrLaneProvenance::Unknown;
    KernargPtrLaneProvenance High = KernargPtrLaneProvenance::Unknown;
    int64_t EntryByteOffset = 0;

    bool operator==(KernargPtrProvenance Other) const {
      return Low == Other.Low && High == Other.High &&
             EntryByteOffset == Other.EntryByteOffset;
    }

    bool isLiveEntry() const {
      return Low == KernargPtrLaneProvenance::LiveEntry &&
             High == KernargPtrLaneProvenance::LiveEntry;
    }

    bool isNonEntry() const {
      return Low == KernargPtrLaneProvenance::NonEntry &&
             High == KernargPtrLaneProvenance::NonEntry;
    }
  };

  // Join lane facts, preserving only agreement.
  static KernargPtrLaneProvenance
  joinKernargPtrLaneProvenance(KernargPtrLaneProvenance Lhs,
                               KernargPtrLaneProvenance Rhs) {
    if (Lhs == Rhs)
      return Lhs;
    return KernargPtrLaneProvenance::Unknown;
  }

  // Join pair facts; unequal live-entry offsets become unknown.
  static KernargPtrProvenance
  joinKernargPtrProvenance(KernargPtrProvenance Lhs, KernargPtrProvenance Rhs) {
    KernargPtrProvenance Result = {
        joinKernargPtrLaneProvenance(Lhs.Low, Rhs.Low),
        joinKernargPtrLaneProvenance(Lhs.High, Rhs.High), 0};
    if (Result.isLiveEntry()) {
      if (Lhs.isLiveEntry() && Rhs.isLiveEntry() &&
          Lhs.EntryByteOffset == Rhs.EntryByteOffset)
        Result.EntryByteOffset = Lhs.EntryByteOffset;
      else
        Result.Low = Result.High = KernargPtrLaneProvenance::Unknown;
    }
    return Result;
  }

  // True when `Base` names the descriptor-provided kernarg pointer SGPR pair.
  // Kernels that do not enable that user SGPR never match.
  bool isEntryKernargSegmentPtrSgpr(ParsedReg Base) const {
    assert(Layout && "RaiseContext requires descriptor-derived SGPR layout");
    if (Base.RegKind != ParsedReg::SGPR)
      return false;
    assert(Base.BaseIdx && "SGPR must have a base register index");
    std::optional<unsigned> KernargPtrSgpr = Layout->kernargSegmentPtrSgpr();
    return KernargPtrSgpr && Base.BaseIdx == KernargPtrSgpr;
  }

  KernargPtrProvenance getKernargPtrProvenance() const {
    return CurrentKernargPtrProvenance;
  }

  // Record a proven constant offset from the entry kernarg pointer.
  void setKernargPtrLiveEntryByteOffset(int64_t ByteOffset) {
    CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::LiveEntry;
    CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::LiveEntry;
    CurrentKernargPtrProvenance.EntryByteOffset = ByteOffset;
  }

  // Record a proven non-entry pointer.
  void setKernargPtrNonEntry() {
    CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::NonEntry;
    CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::NonEntry;
    CurrentKernargPtrProvenance.EntryByteOffset = 0;
  }

  // Invalidate provenance for a written kernarg pointer lane.
  void noteSgprWriteForKernargProvenance(unsigned Idx) {
    assert(Layout && "RaiseContext requires descriptor-derived SGPR layout");
    std::optional<unsigned> KernargPtrSgpr = Layout->kernargSegmentPtrSgpr();
    if (!KernargPtrSgpr)
      return;
    if (Idx == *KernargPtrSgpr)
      CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::Unknown;
    else if (Idx == *KernargPtrSgpr + 1)
      CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::Unknown;
    else
      return;
    CurrentKernargPtrProvenance.EntryByteOffset = 0;
  }

  // Mark kernarg pointer lanes overwritten by a scalar memory load as
  // non-entry.
  void noteSgprMemoryLoadForKernargProvenance(unsigned BaseIdx,
                                              unsigned WidthDwords) {
    assert(Layout && "RaiseContext requires descriptor-derived SGPR layout");
    assert(WidthDwords > 0 && "SMEM destination width must be non-zero");
    std::optional<unsigned> KernargPtrSgpr = Layout->kernargSegmentPtrSgpr();
    if (!KernargPtrSgpr)
      return;
    unsigned EndIdx = BaseIdx + WidthDwords - 1;
    if (BaseIdx <= *KernargPtrSgpr && EndIdx >= *KernargPtrSgpr)
      CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::NonEntry;
    if (BaseIdx <= *KernargPtrSgpr + 1 && EndIdx >= *KernargPtrSgpr + 1)
      CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::NonEntry;
    CurrentKernargPtrProvenance.EntryByteOffset = 0;
  }

  // Record the precomputed provenance at a source basic-block entry.
  void setKernargPtrProvenanceForBlock(llvm::BasicBlock *BB,
                                       KernargPtrProvenance Provenance) {
    KernargSegmentPtrProvenanceByBB[BB] = Provenance;
  }

  // Enter a source basic block with its precomputed provenance.
  void enterKernargPtrProvenanceForBlock(llvm::BasicBlock *BB) {
    assert(BB && "cannot enter kernarg provenance for null basic block");
    if (!HasKernargPtrProvenanceByBB) {
      CurrentKernargPtrProvenance = {};
      return;
    }
    auto It = KernargSegmentPtrProvenanceByBB.find(BB);
    assert(It != KernargSegmentPtrProvenanceByBB.end() &&
           "missing kernarg provenance for source basic block");
    CurrentKernargPtrProvenance = It->second;
  }

  // Emit Body in a lane-active control-flow diamond and leave the builder at
  // its merge block. This preserves inactive lanes for per-lane side effects.
  void emitUnderExec(llvm::function_ref<void()> Body);

  // Cached lane-active value and its defining block.
  llvm::Value *CachedLaneActive = nullptr;
  llvm::BasicBlock *CachedLaneActiveBb = nullptr;

  // Same-block per-lane values for V_CMP writes to SGPR destinations. Under
  // widening, the stored SGPR mask can be narrower than target EXEC, so a
  // consumer uses this value while it remains valid.
  struct WaveMaskEntry {
    llvm::Value *I1 = nullptr;
    // Whether the destination spans this SGPR and its successor.
    bool IsPair = false;
  };

  llvm::DenseMap<unsigned, WaveMaskEntry> LastSgprWaveMaskI1;

  // Cross-block SGPR wave-mask values and validity bits. Alloca storage avoids
  // carrying non-dominating SSA values across recovered control flow.
  llvm::SmallVector<llvm::AllocaInst *> SgprWaveMaskExecShadow;
  llvm::SmallVector<llvm::AllocaInst *> SgprWaveMaskValidShadow;
  llvm::SmallVector<llvm::AllocaInst *> SourceWaveSgprPairShadow;
  llvm::SmallVector<llvm::AllocaInst *> SourceWaveSgprPairValidShadow;
  // Same-block source-image addresses proven for PC-relative literal loads.
  llvm::DenseMap<unsigned, uint64_t> SourceImageSgprPairAddrShadow;

  // Block-local constant value last stored to M0.
  std::optional<uint64_t> M0Const;

  // Conservative kernarg pointer provenance at source basic-block entries.
  // Disagreement remains unknown so strict hidden-argument handling fails
  // loudly.
  bool HasKernargPtrProvenanceByBB = false;
  llvm::DenseMap<llvm::BasicBlock *, KernargPtrProvenance>
      KernargSegmentPtrProvenanceByBB;
  KernargPtrProvenance CurrentKernargPtrProvenance;

  // Record the latest per-lane compare written to an SGPR destination.
  void recordSgprWaveMaskI1(unsigned BaseIdx, llvm::Value *CmpI1, bool IsPair) {
    LastSgprWaveMaskI1[BaseIdx] = WaveMaskEntry{CmpI1, IsPair};
    if (BaseIdx < SgprWaveMaskExecShadow.size()) {
      llvm::Value *ExecMask = Projection.ballotI1ToWidth(
          B, CmpI1, Projection.execStorageTy(), "wm_shadow_exec");
      B.CreateStore(ExecMask, SgprWaveMaskExecShadow[BaseIdx]);
      B.CreateStore(B.getTrue(), SgprWaveMaskValidShadow[BaseIdx]);
    }
  }

  // Return true when the source wave containing the current target lane has any
  // active lane in EXEC.
  llvm::Value *emitCurrentSourceWaveHasActiveLane() {
    llvm::Value *Exec = Regs.loadExec(B);
    if (!Projection.providesFullWaveExecInvariant())
      return emitLaneActiveBit();
    unsigned SourceBits = Isa.waveSize();
    assert(Isa.hasValidWaveSize() && "source wave size must be 32 or 64");
    if (SourceBits >= 64)
      return B.CreateICmpNE(Exec, llvm::ConstantInt::get(Exec->getType(), 0),
                            "source_wave_active");
    llvm::Type *ExecTy = Exec->getType();
    llvm::Value *Lane =
        B.CreateZExtOrTrunc(emitLaneIdx(), ExecTy, "source_wave_lane");
    llvm::Value *Group = B.CreateUDiv(
        Lane, llvm::ConstantInt::get(ExecTy, SourceBits), "source_wave_group");
    llvm::Value *Shift = B.CreateMul(
        Group, llvm::ConstantInt::get(ExecTy, SourceBits), "source_wave_shift");
    llvm::Value *Shifted = B.CreateLShr(Exec, Shift, "source_wave_exec");
    uint64_t Mask = (uint64_t{1} << SourceBits) - 1;
    llvm::Value *GroupMask = B.CreateAnd(
        Shifted, llvm::ConstantInt::get(ExecTy, Mask), "source_wave_mask");
    return B.CreateICmpNE(GroupMask, llvm::ConstantInt::get(ExecTy, 0),
                          "source_wave_active");
  }

  // Record an SGPR-pair value for the active source wave while preserving the
  // value for inactive source waves.
  void recordSourceWaveSgprPair(unsigned BaseIdx, llvm::Value *V) {
    if (!Projection.providesFullWaveExecInvariant())
      return;
    if (BaseIdx >= SourceWaveSgprPairShadow.size())
      return;
    llvm::Value *Old =
        B.CreateLoad(B.getInt64Ty(), SourceWaveSgprPairShadow[BaseIdx],
                     "source_wave_sgpr_pair_old");
    llvm::Value *Merged = B.CreateSelect(emitCurrentSourceWaveHasActiveLane(),
                                         V, Old, "source_wave_sgpr_pair");
    B.CreateStore(Merged, SourceWaveSgprPairShadow[BaseIdx]);
    B.CreateStore(B.getTrue(), SourceWaveSgprPairValidShadow[BaseIdx]);
  }

  // Return the recorded source-wave value or Fallback.
  llvm::Value *materializeSourceWaveSgprPair(unsigned BaseIdx,
                                             llvm::Value *Fallback) {
    if (!Projection.providesFullWaveExecInvariant() ||
        BaseIdx >= SourceWaveSgprPairShadow.size())
      return Fallback;
    llvm::Value *Shadow =
        B.CreateLoad(B.getInt64Ty(), SourceWaveSgprPairShadow[BaseIdx],
                     "source_wave_sgpr_pair");
    llvm::Value *Valid =
        B.CreateLoad(B.getInt1Ty(), SourceWaveSgprPairValidShadow[BaseIdx],
                     "source_wave_sgpr_pair_valid");
    return B.CreateSelect(Valid, Shadow, Fallback, "source_wave_sgpr_pair_sel");
  }

  // Return the current block's cached compare, or null when none is valid.
  llvm::Value *lookupSgprWaveMaskI1(unsigned BaseIdx) const {
    auto It = LastSgprWaveMaskI1.find(BaseIdx);
    return It == LastSgprWaveMaskI1.end() ? nullptr : It->second.I1;
  }

  llvm::Value *loadSgprWaveMaskExec(unsigned BaseIdx) const {
    if (BaseIdx >= SgprWaveMaskExecShadow.size())
      return nullptr;
    return B.CreateLoad(Projection.execStorageTy(),
                        SgprWaveMaskExecShadow[BaseIdx], "sgpr_mask_exec");
  }

  llvm::Value *loadSgprWaveMaskValid(unsigned BaseIdx) const {
    if (BaseIdx >= SgprWaveMaskValidShadow.size())
      return nullptr;
    return B.CreateLoad(B.getInt1Ty(), SgprWaveMaskValidShadow[BaseIdx],
                        "sgpr_mask_valid");
  }

  // Invalidate facts overlapping BaseIdx. A pair is keyed by its low SGPR, so
  // writing its high SGPR also invalidates the preceding entry. Single-SGPR
  // entries remain independent.
  void invalidateSgprWaveMaskI1(unsigned BaseIdx) {
    noteSgprWriteForKernargProvenance(BaseIdx);
    LastSgprWaveMaskI1.erase(BaseIdx);
    SourceImageSgprPairAddrShadow.erase(BaseIdx);
    if (BaseIdx < SgprWaveMaskValidShadow.size())
      B.CreateStore(B.getFalse(), SgprWaveMaskValidShadow[BaseIdx]);
    if (BaseIdx < SourceWaveSgprPairValidShadow.size())
      B.CreateStore(B.getFalse(), SourceWaveSgprPairValidShadow[BaseIdx]);
    if (BaseIdx > 0) {
      auto Prev = LastSgprWaveMaskI1.find(BaseIdx - 1);
      if (Prev != LastSgprWaveMaskI1.end() && Prev->second.IsPair) {
        LastSgprWaveMaskI1.erase(Prev);
        if (BaseIdx - 1 < SgprWaveMaskValidShadow.size())
          B.CreateStore(B.getFalse(), SgprWaveMaskValidShadow[BaseIdx - 1]);
      }
      if (BaseIdx - 1 < SourceWaveSgprPairValidShadow.size())
        B.CreateStore(B.getFalse(), SourceWaveSgprPairValidShadow[BaseIdx - 1]);
      SourceImageSgprPairAddrShadow.erase(BaseIdx - 1);
    }
  }

  // Clear block-local facts whose SSA values cannot cross a block boundary.
  void clearSgprWaveMaskShadow() {
    LastSgprWaveMaskI1.clear();
    SourceImageSgprPairAddrShadow.clear();
  }

  // Record a source code-object address held by an SGPR pair in this block.
  void recordSourceImageSgprPairAddr(unsigned BaseIdx, uint64_t Value) {
    SourceImageSgprPairAddrShadow[BaseIdx] = Value;
  }

  // Return a proven source-image address for an SGPR pair, if present.
  std::optional<uint64_t>
  lookupSourceImageSgprPairAddr(unsigned BaseIdx) const {
    auto It = SourceImageSgprPairAddrShadow.find(BaseIdx);
    if (It == SourceImageSgprPairAddrShadow.end())
      return std::nullopt;
    return It->second;
  }

  // V_MOVREL resolves a VGPR index from base + M0, which must be known while
  // constructing the indexed register file. Track constants only within a
  // basic block.
  void updateM0Const(llvm::Value *V) {
    if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V))
      M0Const = CI->getZExtValue();
    else
      M0Const = std::nullopt;
  }
  void clearM0Const() { M0Const = std::nullopt; }
  std::optional<uint64_t> getM0Const() const { return M0Const; }

  void collectSgprWaveMaskShadowAllocas(
      llvm::SmallVectorImpl<llvm::AllocaInst *> &Out) const {
    Out.append(SgprWaveMaskExecShadow);
    Out.append(SgprWaveMaskValidShadow);
    Out.append(SourceWaveSgprPairShadow);
    Out.append(SourceWaveSgprPairValidShadow);
  }
};

// Result of a format handler. An unhandled result permits dispatch to fall
// through; refusals are returned as llvm::Error.
struct HandlerResult {
  bool Handled = false;
  llvm::Value *SccResult = nullptr;
  bool SccHandled = false;
};

// Reads a handler's source operands via the decoded srcMap, applying VOP3
// neg/abs modifiers on the float paths.
struct OpResolver {
  RaiseContext &Ctx;
  const DecodedInst &Di;

  unsigned srcIdx(unsigned I) const {
    assert(I < Di.SrcMap.size() && "source index out of range");
    return Di.SrcMap[I];
  }
  unsigned nSrcs() const { return static_cast<unsigned>(Di.SrcMap.size()); }

  unsigned srcMod(unsigned I) const {
    unsigned ModIdx = Di.ModMap[I];
    if (ModIdx == UINT_MAX)
      return 0;
    if (!Di.isImm(ModIdx))
      return 0;
    return static_cast<unsigned>(Di.getImm(ModIdx) & 0xF);
  }

  llvm::Value *applyMods(unsigned I, llvm::Value *V) {
    unsigned Mods = srcMod(I);
    if (Mods == 0)
      return V;
    bool IsI32 = (V->getType() == Ctx.B.getInt32Ty());
    if (IsI32)
      V = Ctx.B.CreateBitCast(V, Ctx.B.getFloatTy());
    if (Mods & 2)
      V = Ctx.B.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, V, nullptr, "abs");
    if (Mods & 1)
      V = Ctx.B.CreateFNeg(V, "neg");
    if (IsI32)
      V = Ctx.B.CreateBitCast(V, Ctx.B.getInt32Ty());
    return V;
  }

  llvm::Expected<llvm::Value *> src(unsigned I) {
    return Ctx.readOp32(Di, srcIdx(I));
  }
  llvm::Expected<llvm::Value *> srcF(unsigned I) {
    llvm::Expected<llvm::Value *> V = Ctx.readOp32(Di, srcIdx(I));
    if (!V)
      return V.takeError();
    return applyMods(I, *V);
  }
  llvm::Expected<llvm::Value *> src64(unsigned I) {
    return Ctx.readOp64(Di, srcIdx(I));
  }
  llvm::Expected<llvm::Value *> srcExecWidth(unsigned I) {
    return Ctx.readOpExecWidth(Di, srcIdx(I));
  }
  int64_t srcImm(unsigned I) { return Di.getImm(srcIdx(I)); }

  ParsedReg dst(unsigned I = 0) { return Ctx.parseReg(Di.getReg(I), I); }
  bool isSrcReg(unsigned I) { return Di.isReg(srcIdx(I)); }

  ParsedReg srcReg(unsigned I) {
    unsigned Idx = srcIdx(I);
    if (!Di.isReg(Idx)) {
      ParsedReg Pr;
      Pr.RegKind = ParsedReg::OTHER;
      return Pr;
    }
    return Ctx.parseReg(Di.getReg(Idx), Idx);
  }
};

} // namespace COMGR::hotswap

#endif
