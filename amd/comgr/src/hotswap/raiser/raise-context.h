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
#include "hotswap/loader/code-object-utils.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/isa-profile.h"
#include "kernarg-layout.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/parsed-reg.h"
#include "raise_failure.h"
#include "reg-file.h"
#include "user-sgpr-layout.h"
#include "wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <map>
#include <optional>
#include <utility>

namespace COMGR::hotswap {

// Shared state threaded through every format handler.
struct RaiseContext {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::IRBuilder<> &B;
  AllocaRegFile &Regs;
  const WaveProjection &Projection;
  const MCState &Mc;
  const ISAProfile &Isa; // source ISA (for disassembly / instruction semantics)
  ISAProfile
      TargetIsa; // compilation target ISA (for code generation decisions)
  // Target hidden-arg offsets are code-object-version dependent; default to
  // the backend's current emission contract and let the raiser override it.
  unsigned TargetCodeObjectVersion = 6;
  KernargLayout &Kernargs;
  // Source-ISA user-SGPR ABI derived from the kernel descriptor, used to
  // identify a specific SGPR by its source-ABI role (e.g. which SGPR holds
  // the kernarg-segment pointer). Owned by the raiser; non-null in production.
  const UserSgprLayout *Layout = nullptr;
  llvm::Function *Kernel;
  llvm::BasicBlock *ThreadLoopLatch = nullptr;

  llvm::IntegerType *I1Ty;
  llvm::IntegerType *I8Ty;
  llvm::IntegerType *I16Ty;
  llvm::IntegerType *I32Ty;
  llvm::IntegerType *I64Ty;
  llvm::Type *F32Ty;
  llvm::Type *F16Ty;
  llvm::Type *F64Ty;
  llvm::Type *PtrGlobalTy;

  llvm::DenseMap<uint64_t, llvm::BasicBlock *> &OffsetToBb;
  // Source code-object bytes used to materialise proven PC-relative literals.
  // `SourceTextBytes` remains the disassembly image; `SourceImageSections`
  // carries allocated sections addressable by source code-object address.
  llvm::ArrayRef<uint8_t> SourceTextBytes;
  uint64_t SourceTextBaseAddress = 0;
  llvm::ArrayRef<TextSection::ImageSection> SourceImageSections;
  uint64_t KernelStartOffset = 0;
  uint64_t KernelEndOffset = 0;

  RaiseContext(llvm::LLVMContext &C, llvm::Module &M, llvm::IRBuilder<> &B,
               AllocaRegFile &Regs, const WaveProjection &Projection,
               const MCState &Mc, const ISAProfile &Isa, ISAProfile TargetIsa,
               unsigned TargetCodeObjectVersion, KernargLayout &Kernargs,
               const UserSgprLayout *Layout, llvm::Function *Kernel,
               llvm::BasicBlock *ThreadLoopLatch,
               llvm::DenseMap<uint64_t, llvm::BasicBlock *> &OffsetToBb,
               llvm::ArrayRef<uint8_t> SourceTextBytes,
               uint64_t SourceTextBaseAddress,
               llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
               uint64_t KernelStartOffset, uint64_t KernelEndOffset);

  // Source KD private/scratch allocation. Set when a `scratch_*` instruction
  // is lowered; the on-demand private alloca is managed by LLVM's frame layout
  // so target spills cannot overlap translated source scratch slots.
  uint32_t SourcePrivateSegmentFixedSize = 0;
  uint32_t SourceComputePgmRsrc2 = 0;
  uint16_t SourceKernelCodeProperties = 0;
  bool UsesScratchPrivateSegment = false;
  llvm::AllocaInst *ScratchPrivateSegmentAlloca = nullptr;

  // gfx1250 s_set_vgpr_msb state: only the LOW 8 bits of the instruction's
  // 16-bit immediate carry runtime meaning.  They encode four 2-bit MSB
  // fields, one per slot, that apply to subsequent instructions until the
  // next s_set_vgpr_msb:
  //
  //   [1:0]  slot 0 MSB       [3:2]  slot 1 MSB
  //   [5:4]  slot 2 MSB       [7:6]  slot 3 MSB
  //
  // Each 2-bit field adds (value * 256) to the corresponding operand's VGPR
  // index. The slot->operand mapping is instruction-format-specific and is NOT
  // VALU's positional src0/src1/src2/vdst order in general: VBUFFER maps slot 0
  // to vaddr and slot 3 to vdata, VDS maps slots 0/1/2 to addr/data0/data1, and
  // so on. `computeVGPRAdjust` resolves it via
  // AMDGPU::getVGPRLoweringOperandTables (the same per-format tables the AMDGPU
  // backend uses to lower VGPR encoding).
  //
  // The HIGH 8 bits record the previous mode value for compiler bookkeeping;
  // hardware ignores them, so mask to 8 bits on store.
  //
  // VOPD dual-issue X-op and Y-op share the same MSB pair per operand slot, so
  // applying the same 8-bit state to both halves is correct.
  uint8_t VgprMsBs = 0;
  bool AssumeHipGlobalOffsetZero = false;

  // Per-instruction VGPR index adjustment, indexed by MCInst operand index.
  // Computed from vgprMSBs before each instruction dispatch. computeVGPRAdjust
  // returns an error if LLVM TableGen grows a VGPR-MSB-controlled operand
  // beyond this bound; silently dropping such an adjustment would misdecode the
  // source register bank.
  static constexpr unsigned KMaxOps = 16;
  unsigned CurrentVgprAdjust[KMaxOps] = {};

  // Compute currentVGPRAdjust for the given instruction based on vgprMSBs.
  llvm::Error computeVGPRAdjust(const DecodedInst &Di);

  llvm::BasicBlock *lookupBB(uint64_t Addr);

  ParsedReg parseReg(llvm::MCRegister Reg, int MciOpIdx = -1) const;

  // Read the operand at OpIdx as a 32-bit value, resolving registers through
  // the reg-file and immediates through the MC operand.
  llvm::Value *readOp32(const DecodedInst &Di, unsigned OpIdx);
  llvm::Value *readOp64(const DecodedInst &Di, unsigned OpIdx);
  llvm::Value *readOpExecWidth(const DecodedInst &Di, unsigned OpIdx);
  // Read the mask a source-wave instruction should see, e.g. for `v_mbcnt_lo`.
  // EXEC/VCC/SGPR-shadow masks are projected; scalars use readOp32.
  llvm::Value *readOpSourceWaveMask32(const DecodedInst &Di, unsigned OpIdx);

  // Target-hardware lane id (i32), emitted once per kernel and reused.
  llvm::Value *emitLaneIdx();

  // Neutralise a per-lane memory ADDRESS against poison on inactive lanes
  // when widening wave32 -> wave64.
  //
  // A per-lane address VGPR is produced under one EXEC-gated `emitUnderExec`
  // region and consumed by a memory op under a later, possibly different one.
  // On lanes that were inactive when the address was produced, the alloca
  // reg-file carries `undef` (the inactive arm of the first-def phi after
  // mem2reg). Those lanes never commit the memory op -- the op is itself
  // wrapped in `emitUnderExec` -- but feeding `undef`/poison through
  // `inttoptr` into a load/store is UB, which the AMDGPU backend is entitled
  // to exploit (it may drop the divergent branch and issue the access at the
  // wave-native HW EXEC = -1 forced by `init_whole_wave`, faulting on the
  // undef address). Freezing the address integer replaces poison with an
  // arbitrary but well-defined value, removing the UB while leaving the
  // per-lane `emitUnderExec` gate to keep the access off inactive lanes.
  //
  // Gated on widening (`Isa.isWave32() && !TargetIsa.isWave32()`) so
  // same-wave and narrowing lifts keep byte-identical codegen. `Addr` is the
  // integer address (returned unchanged when not widening).
  llvm::Value *freezeMemAddr(llvm::Value *Addr);

  // SIMT predicated-execution helpers.
  //
  // emitLaneActiveBit() returns an i1 true iff the current lane's bit in the
  // EXEC-mask alloca is set. Wave-size-aware via targetIsa: the lane id is
  // built from llvm.amdgcn.mbcnt.lo for wave32 and mbcnt.lo+mbcnt.hi for
  // wave64. The alloca itself is the current SSA-tracked EXEC value, so
  // uniform code (EXEC provably -1 by SROA) folds this to `true`.
  //
  // Caching: the result is memoised for one decoded instruction's dispatch, so
  // handlers that emit multiple `emitUnderExec` diamonds reuse the same
  // `lane_active` i1. Invalidated at every new source instruction, on an
  // insertion-block change (the cached i1 no longer dominates), and on any
  // EXEC write. The cache keeps the raw raised IR readable for lit tests that
  // FileCheck the unoptimised output shape.
  llvm::Value *emitLaneActiveBit();

  // Invalidate the lane_active memoisation. Called by the main raiser
  // loop between instructions and by `storeExec`. Handlers that know
  // they have mutated EXEC through a lower-level path (e.g. the few
  // places that call `regs.storeExec` directly) must also invoke this.
  void resetLaneActiveCache() {
    CachedLaneActive = nullptr;
    CachedLaneActiveBb = nullptr;
  }

  // Wrap `regs.storeExec` with cache invalidation. Handlers should prefer
  // this over `regs.storeExec` so the lane_active memo is always
  // consistent with the live EXEC value.
  void storeExec(llvm::Value *V) {
    Regs.storeExec(B, V);
    resetLaneActiveCache();
  }

  // Predicated register-commit API. VGPR/AGPR writes are per-lane side
  // effects and MUST be wrapped in an emitUnderExec diamond so inactive
  // lanes keep their prior VGPR value; SGPR/VCC/SCC/EXEC/M0/FLAT_SCR/TTMP
  // writes are wave-level and pass through unchanged. Handlers should
  // call these instead of reaching into `regs.write*` directly.
  //
  // `storeVGPR32` / `storeVGPR64` / `storeAGPR32` are the direct-index
  // variants used when the register index is already known.
  void writeReg32(ParsedReg Pr, llvm::Value *V);
  void writeReg64(ParsedReg Pr, llvm::Value *V);
  void writeRegVec(ParsedReg Pr, llvm::Value *V);
  void writeRegExecWidth(ParsedReg Pr, llvm::Value *V);
  void storeVGPR32(int Idx, llvm::Value *V);
  void storeVGPR64(int Idx, llvm::Value *V);
  void storeAGPR32(int Idx, llvm::Value *V);

  // Provenance fact for the physical SGPR pair that originally held the
  // source-ABI kernarg-segment pointer.
  //
  //   LiveEntry - all incoming CFG paths carry the dispatch-provided entry
  //               pointer plus EntryByteOffset. Both lanes must be LiveEntry
  //               for the offset to be meaningful.
  //   NonEntry  - all incoming CFG paths have overwritten the pair with a value
  //               loaded from memory rather than the dispatch-provided entry
  //               SGPR value. Constant rebases of such a value remain NonEntry.
  //   Unknown   - paths disagree, are unreachable, include an unclassified
  //               write, or carry different EntryByteOffset values.
  enum class KernargPtrLaneProvenance {
    LiveEntry,
    NonEntry,
    Unknown,
  };

  // Consumers classify the pair by combining the two lane facts: both LiveEntry
  // permits source hidden-arg synthesis at EntryByteOffset + instruction
  // offset, both NonEntry uses ordinary memory lowering, and any mixed/Unknown
  // state is ambiguous in strict mode.
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

  // Merge facts from two control-flow paths. Equal lane facts survive; any
  // disagreement becomes Unknown. A LiveEntry pair keeps EntryByteOffset only
  // when every incoming path has the same offset.
  static KernargPtrLaneProvenance
  joinKernargPtrLaneProvenance(KernargPtrLaneProvenance Lhs,
                               KernargPtrLaneProvenance Rhs) {
    if (Lhs == Rhs)
      return Lhs;
    return KernargPtrLaneProvenance::Unknown;
  }

  // Pair-wise control-flow join for provenance carried through IR diamonds.
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

  // Restore a proven entry-pointer fact after a constant-preserving rebase.
  void setKernargPtrLiveEntryByteOffset(int64_t ByteOffset) {
    CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::LiveEntry;
    CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::LiveEntry;
    CurrentKernargPtrProvenance.EntryByteOffset = ByteOffset;
  }

  // Restore a proven non-entry pointer fact after a constant-preserving rebase.
  void setKernargPtrNonEntry() {
    CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::NonEntry;
    CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::NonEntry;
    CurrentKernargPtrProvenance.EntryByteOffset = 0;
  }

  // Update the current intra-BB provenance state after an ordinary SGPR write.
  // Only writes to either kernarg-pointer lane change this fact. A generic
  // register write kills LiveEntry but does not prove a non-entry value.
  void noteSgprWriteForKernargProvenance(int Idx) {
    assert(Layout && "RaiseContext requires descriptor-derived SGPR layout");
    assert(Idx >= 0 && "SGPR write must have a valid register index");
    unsigned SgprIdx = static_cast<unsigned>(Idx);
    std::optional<unsigned> KernargPtrSgpr = Layout->kernargSegmentPtrSgpr();
    if (!KernargPtrSgpr)
      return;
    if (SgprIdx == *KernargPtrSgpr)
      CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::Unknown;
    else if (SgprIdx == *KernargPtrSgpr + 1)
      CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::Unknown;
    else
      return;
    CurrentKernargPtrProvenance.EntryByteOffset = 0;
  }

  // Record that an SMEM-style load wrote one or more SGPR lanes from memory.
  // This is stronger than a generic register write: any overlapped kernarg
  // pointer lane no longer carries the dispatch-provided entry SGPR value.
  void noteSgprMemoryLoadForKernargProvenance(int BaseIdx, int WidthDwords) {
    assert(Layout && "RaiseContext requires descriptor-derived SGPR layout");
    assert(BaseIdx >= 0 && "SMEM destination base must be a valid SGPR index");
    assert(WidthDwords > 0 && "SMEM destination width must be non-zero");
    std::optional<unsigned> KernargPtrSgpr = Layout->kernargSegmentPtrSgpr();
    if (!KernargPtrSgpr)
      return;
    unsigned SgprBaseIdx = static_cast<unsigned>(BaseIdx);
    unsigned EndIdx = SgprBaseIdx + static_cast<unsigned>(WidthDwords) - 1;
    if (SgprBaseIdx <= *KernargPtrSgpr && EndIdx >= *KernargPtrSgpr)
      CurrentKernargPtrProvenance.Low = KernargPtrLaneProvenance::NonEntry;
    if (SgprBaseIdx <= *KernargPtrSgpr + 1 && EndIdx >= *KernargPtrSgpr + 1)
      CurrentKernargPtrProvenance.High = KernargPtrLaneProvenance::NonEntry;
    CurrentKernargPtrProvenance.EntryByteOffset = 0;
  }

  // Record the prepass-computed entry fact for a recovered source BB.
  void setKernargPtrProvenanceForBlock(llvm::BasicBlock *BB,
                                       KernargPtrProvenance Provenance) {
    KernargSegmentPtrProvenanceByBB[BB] = Provenance;
  }

  // Load the prepass entry fact when lowering reaches a recovered source BB.
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

  // emitUnderExec(body) wraps `body()` in an `if (lane_active)` diamond:
  //
  //   %active = emitLaneActiveBit()
  //   br i1 %active, label %exec_do, label %exec_skip
  //   exec_do:
  //     body()                 (whatever side-effectful IR the handler emits)
  //     br label %exec_skip    (only if body() did not itself terminate)
  //   exec_skip:
  //     ...                    (insertion point on return)
  //
  // Because %active is data-dependent on workitem.id.x, LLVM's divergence
  // analysis treats the branch as divergent and the AMDGPU backend
  // rematerialises hardware-level v_cmpx around the do-block. Uniform code
  // collapses: when %active folds to `true` the diamond vanishes, so this
  // is a no-op in IR size + codegen terms for non-divergent sites.
  //
  // On return, the builder's insertion point is at the start of %exec_skip,
  // so subsequent handler emission continues in the skip block (which is
  // topologically the "after" of the wrapped op, exactly like before).
  void emitUnderExec(llvm::function_ref<void()> Body);

  // Memoised lane_active for this instruction's emission. Mutate only via
  // `resetLaneActiveCache` / `emitLaneActiveBit`.
  llvm::Value *CachedLaneActive = nullptr;
  llvm::BasicBlock *CachedLaneActiveBb = nullptr;

  // Per-BB cache of the per-lane i1 compare result produced by the
  // most recent V_CMP_*_e64 writer targeting a given SGPR in this
  // basic block. Keyed by source-ABI SGPR baseIdx (the low SGPR of
  // an SGPR pair on wave64 source; the single SGPR on wave32
  // source). `isPair` distinguishes a wave64-source pair entry
  // (the value spans [baseIdx, baseIdx+1]) from a wave32-source
  // single entry (baseIdx only), which matters for the adjacent-
  // invalidation rule in `invalidateSgprWaveMaskI1`.
  //
  // A wave64-source V_CMP -> SGPR store truncates the full target-hardware
  // ballot to the source SGPR's 32-bit width, losing lanes 32..63 under
  // wave32 -> wave64 widening. A same-BB V_CNDMASK_B32 consumer cannot
  // recover those bits from the narrow SGPR, so this cache carries the full
  // per-lane i1 across to the consumer. When absent, the consumer takes the
  // narrow-mask fallback. Entries are invalidated on any overlapping SGPR
  // write and cleared at every BB boundary.
  struct WaveMaskEntry {
    llvm::Value *I1 = nullptr;
    // True if this entry is a wave64-source V_CMP whose destination is the
    // SGPR pair [baseIdx, baseIdx+1], false for a single wave32-source SGPR.
    // A write to baseIdx+1 invalidates a pair entry at baseIdx.
    bool IsPair = false;
  };

  llvm::DenseMap<int, WaveMaskEntry> LastSgprWaveMaskI1;

  // Cross-BB, dominance-safe shadow storage for SGPR wave masks.
  // Each SGPR base index has:
  //   * sgprWaveMaskExecShadow[idx]  : EXEC-width mask value (i32/i64)
  //   * sgprWaveMaskValidShadow[idx] : scalar i1 validity bit
  //
  // Record/write sites update both allocas; invalidation writes
  // `valid=false`. Consumers can load-valid+load-mask and pick between
  // shadow and fallback via `select`, avoiding SSA-dominance hazards.
  llvm::SmallVector<llvm::AllocaInst *> SgprWaveMaskExecShadow;
  llvm::SmallVector<llvm::AllocaInst *> SgprWaveMaskValidShadow;
  llvm::SmallVector<llvm::AllocaInst *> SourceWaveSgprPairShadow;
  llvm::SmallVector<llvm::AllocaInst *> SourceWaveSgprPairValidShadow;
  // Same-BB source-image address facts for PC-relative literal loads. This is
  // not a generic constant tracker: only s_get_pc_i64 seeds it, only constant
  // s_add/sub_nc_u64 propagates it, and only SMEM literal materialisation reads
  // it.
  llvm::DenseMap<int, uint64_t> SourceImageSgprPairAddrShadow;

  // Raise-time constant shadow of M0; see updateM0Const / getM0Const.
  std::optional<uint64_t> M0Const;

  // Conservative lane-wise kernarg-pointer provenance for the strict hidden-arg
  // SMEM gate. Filled before instruction lowering by a fixed-point over the
  // decoded CFG. Mixed incoming states become Unknown and keep strict mode
  // loud. False means tracking is inactive and BB entry uses Unknown without
  // lookup.
  bool HasKernargPtrProvenanceByBB = false;
  llvm::DenseMap<llvm::BasicBlock *, KernargPtrProvenance>
      KernargSegmentPtrProvenanceByBB;
  KernargPtrProvenance CurrentKernargPtrProvenance;

  // Record the per-lane compare i1 produced by a V_CMP_*_e64 write
  // to SGPR baseIdx in the current BB. Overwrites any prior entry
  // (last-writer wins -- a later V_CMP obviates the earlier value
  // for any consumer that reads after the write). `isPair` should
  // be true iff the V_CMP's destination ParsedReg has `width >= 2`
  // (a wave64-source SGPR pair), so subsequent writes to baseIdx+1
  // correctly invalidate this entry via
  // `invalidateSgprWaveMaskI1`'s pair-aware branch.
  void recordSgprWaveMaskI1(int BaseIdx, llvm::Value *CmpI1, bool IsPair) {
    LastSgprWaveMaskI1[BaseIdx] = WaveMaskEntry{CmpI1, IsPair};
    if (BaseIdx >= 0 &&
        static_cast<size_t>(BaseIdx) < SgprWaveMaskExecShadow.size()) {
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

  // Record an SGPR-pair marker for the currently active source wave, preserving
  // the old marker for inactive source waves.
  void recordSourceWaveSgprPair(int BaseIdx, llvm::Value *V) {
    if (!Projection.providesFullWaveExecInvariant())
      return;
    if (BaseIdx < 0 ||
        static_cast<size_t>(BaseIdx) >= SourceWaveSgprPairShadow.size())
      return;
    llvm::Value *Old = B.CreateLoad(I64Ty, SourceWaveSgprPairShadow[BaseIdx],
                                    "source_wave_sgpr_pair_old");
    llvm::Value *Merged = B.CreateSelect(emitCurrentSourceWaveHasActiveLane(),
                                         V, Old, "source_wave_sgpr_pair");
    B.CreateStore(Merged, SourceWaveSgprPairShadow[BaseIdx]);
    B.CreateStore(B.getTrue(), SourceWaveSgprPairValidShadow[BaseIdx]);
  }

  // Load the source-wave marker when one was recorded; otherwise use the normal
  // SGPR-pair value.
  llvm::Value *materializeSourceWaveSgprPair(int BaseIdx,
                                             llvm::Value *Fallback) {
    if (!Projection.providesFullWaveExecInvariant() || BaseIdx < 0 ||
        static_cast<size_t>(BaseIdx) >= SourceWaveSgprPairShadow.size())
      return Fallback;
    llvm::Value *Shadow = B.CreateLoad(I64Ty, SourceWaveSgprPairShadow[BaseIdx],
                                       "source_wave_sgpr_pair");
    llvm::Value *Valid =
        B.CreateLoad(I1Ty, SourceWaveSgprPairValidShadow[BaseIdx],
                     "source_wave_sgpr_pair_valid");
    return B.CreateSelect(Valid, Shadow, Fallback, "source_wave_sgpr_pair_sel");
  }

  // Look up the cached per-lane i1 for SGPR baseIdx in the current
  // BB, or null if none (either no V_CMP wrote it, or the entry
  // was invalidated by a scalar write, or the BB boundary cleared
  // the map). Callers treat null as "fall back to the standard
  // extractLaneBitFromWaveMask".
  llvm::Value *lookupSgprWaveMaskI1(int BaseIdx) const {
    auto It = LastSgprWaveMaskI1.find(BaseIdx);
    return It == LastSgprWaveMaskI1.end() ? nullptr : It->second.I1;
  }

  llvm::Value *loadSgprWaveMaskExec(int BaseIdx) const {
    if (BaseIdx < 0 ||
        static_cast<size_t>(BaseIdx) >= SgprWaveMaskExecShadow.size())
      return nullptr;
    return B.CreateLoad(Projection.execStorageTy(),
                        SgprWaveMaskExecShadow[BaseIdx], "sgpr_mask_exec");
  }

  llvm::Value *loadSgprWaveMaskValid(int BaseIdx) const {
    if (BaseIdx < 0 ||
        static_cast<size_t>(BaseIdx) >= SgprWaveMaskValidShadow.size())
      return nullptr;
    return B.CreateLoad(I1Ty, SgprWaveMaskValidShadow[BaseIdx],
                        "sgpr_mask_valid");
  }

  // Invalidate the cached per-lane i1 for SGPR baseIdx. Called by
  // AllocaRegFile on any SGPR write so the next consumer takes the
  // narrow-mask fallback rather than a stale i1 whose bits no
  // longer correspond to the scalar value just stored. Idempotent;
  // safe to call on an SGPR that had no cached entry.
  //
  // Pair-aware adjacent invalidation. On wave64 source, V_CMP_e64
  // writes an SGPR pair [baseIdx, baseIdx+1] but records a single
  // entry keyed on baseIdx (via `recordSgprWaveMaskI1(..., /*isPair=*/true)`).
  // If later code writes to baseIdx+1 alone (e.g.
  // `s_mov_b32 sHi, imm`), the high half of the pair is clobbered
  // but the entry at baseIdx would otherwise survive and silently
  // return a cmp that no longer matches the pair's current value.
  // So on invalidate(K), if entry at K-1 exists AND is flagged
  // `isPair`, invalidate K-1 too. The guard on `isPair` avoids
  // over-invalidation: a single-SGPR wave32 entry at K-1 is
  // unrelated to a scalar write at K and must NOT be invalidated.
  void invalidateSgprWaveMaskI1(int BaseIdx) {
    noteSgprWriteForKernargProvenance(BaseIdx);
    LastSgprWaveMaskI1.erase(BaseIdx);
    SourceImageSgprPairAddrShadow.erase(BaseIdx);
    if (BaseIdx >= 0 &&
        static_cast<size_t>(BaseIdx) < SgprWaveMaskValidShadow.size())
      B.CreateStore(B.getFalse(), SgprWaveMaskValidShadow[BaseIdx]);
    if (BaseIdx >= 0 &&
        static_cast<size_t>(BaseIdx) < SourceWaveSgprPairValidShadow.size())
      B.CreateStore(B.getFalse(), SourceWaveSgprPairValidShadow[BaseIdx]);
    if (BaseIdx > 0) {
      auto Prev = LastSgprWaveMaskI1.find(BaseIdx - 1);
      if (Prev != LastSgprWaveMaskI1.end() && Prev->second.IsPair) {
        LastSgprWaveMaskI1.erase(Prev);
        if (static_cast<size_t>(BaseIdx - 1) < SgprWaveMaskValidShadow.size())
          B.CreateStore(B.getFalse(), SgprWaveMaskValidShadow[BaseIdx - 1]);
      }
      if (static_cast<size_t>(BaseIdx - 1) <
          SourceWaveSgprPairValidShadow.size())
        B.CreateStore(B.getFalse(), SourceWaveSgprPairValidShadow[BaseIdx - 1]);
      SourceImageSgprPairAddrShadow.erase(BaseIdx - 1);
    }
  }

  // Drop every cached entry. Called at every BB boundary so cross-BB
  // V_CMP / V_CNDMASK pairs conservatively fall back to the narrow extract
  // rather than relying on an i1 that no longer dominates the consumer.
  void clearSgprWaveMaskShadow() {
    LastSgprWaveMaskI1.clear();
    SourceImageSgprPairAddrShadow.clear();
  }

  // Record that SGPR pair [BaseIdx:BaseIdx+1] currently holds a source
  // code-object address, in the same address domain as s_get_pc_i64. This is
  // used only for PC-relative literal-table sequences and is cleared on any
  // overlapping SGPR write or BB boundary.
  void recordSourceImageSgprPairAddr(int BaseIdx, uint64_t Value) {
    if (BaseIdx < 0)
      llvm::report_fatal_error(
          "transpiler: source-image SGPR pair record has invalid base index");
    SourceImageSgprPairAddrShadow[BaseIdx] = Value;
  }

  // Return the source-image address fact for SGPR pair [BaseIdx:BaseIdx+1], if
  // the current BB has proven one. Absence means the SMEM handler must use the
  // ordinary runtime memory path or refuse according to its own operand rules.
  std::optional<uint64_t> lookupSourceImageSgprPairAddr(int BaseIdx) const {
    auto It = SourceImageSgprPairAddrShadow.find(BaseIdx);
    if (It == SourceImageSgprPairAddrShadow.end())
      return std::nullopt;
    return It->second;
  }

  // --- M0 raise-time constant shadow ---------------------------------------
  // v_movrel* resolve their VGPR index from `base + M0`. Because the reg
  // file promotes VGPRs to SSA by index, the index must be known at raise
  // time. `M0Const` tracks the last constant stored to M0 within the
  // current basic block; it is cleared on any non-constant M0 store and at
  // every BB boundary (M0 is uniform, but a value written in a predecessor
  // no longer dominates trivially, so we stay conservative).
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
    Out.append(SgprWaveMaskExecShadow.begin(), SgprWaveMaskExecShadow.end());
    Out.append(SgprWaveMaskValidShadow.begin(), SgprWaveMaskValidShadow.end());
    Out.append(SourceWaveSgprPairShadow.begin(),
               SourceWaveSgprPairShadow.end());
    Out.append(SourceWaveSgprPairValidShadow.begin(),
               SourceWaveSgprPairValidShadow.end());
  }

  // Record an operand-read failure. Read paths cannot bail mid-handler (they
  // must return a Value*), so they report the failure here and the dispatch
  // loop promotes it to a structured kernel-raise failure at the next
  // instruction boundary.
  llvm::function_ref<void(llvm::Error Err)> recordReadFailure;
};

// Return value from every format handler, carried inside an
// `llvm::Expected<HandlerResult>`.
//
// Handlers communicate back in three ways:
//   * `Handled = true` -> the handler fully lowered the instruction.
//   * `Handled = false` (no Error) -> this handler does not claim the
//     instruction; the main loop falls through to the generic
//     `UnsupportedOpcode` diagnostic.
//   * an `llvm::Error` (a `RaiseFailure`) -> the handler recognised the
//     instruction but refuses to lower it (e.g. operand shape
//     unsupported); the main loop records the structured failure and
//     aborts without consulting other handlers.
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
    bool IsI32 = (V->getType() == Ctx.I32Ty);
    if (IsI32)
      V = Ctx.B.CreateBitCast(V, Ctx.F32Ty);
    if (Mods & 2)
      V = Ctx.B.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, V, nullptr, "abs");
    if (Mods & 1)
      V = Ctx.B.CreateFNeg(V, "neg");
    if (IsI32)
      V = Ctx.B.CreateBitCast(V, Ctx.I32Ty);
    return V;
  }

  llvm::Value *src(unsigned I) { return Ctx.readOp32(Di, srcIdx(I)); }
  llvm::Value *srcF(unsigned I) {
    return applyMods(I, Ctx.readOp32(Di, srcIdx(I)));
  }
  llvm::Value *src64(unsigned I) { return Ctx.readOp64(Di, srcIdx(I)); }
  llvm::Value *srcExecWidth(unsigned I) {
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
