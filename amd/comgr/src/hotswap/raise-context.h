//===- raise-context.h - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_CONTEXT_H
#define HOTSWAP_TRANSPILER_RAISE_CONTEXT_H

#include "decoded-inst.h"
#include "isa-profile.h"
#include "kernarg-layout.h"
#include "mc-state.h"
#include "parsed-reg.h"
#include "raise-failure.h"
#include "reg-file.h"
#include "setpc-analysis.h"
#include "user-sgpr-layout.h"
#include "wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegister.h"

#include <map>

namespace COMGR::hotswap {

// Shared state threaded through every format handler.
struct RaiseContext {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::IRBuilder<> &B;
  AllocaRegFile &Regs;
  const WaveProjection &Projection;
  const MCState &Mc;
  const ISAProfile &Isa;       // source ISA (for disassembly / instruction semantics)
  ISAProfile TargetIsa;        // compilation target ISA (for code generation decisions)
  KernargLayout &Kernargs;
  // Source-ISA user-SGPR ABI derived from the kernel descriptor. Owned by
  // the raiser; threaded into every handler that needs to identify a
  // specific SGPR by its source-ABI role (e.g. handle-smem.cpp must know
  // which SGPR holds kernarg_segment_ptr to recognise s_load_b* against
  // the kernarg segment). Always non-null in production; the raiser
  // populates it from `UserSgprLayout::fromKernelMeta` before constructing
  // the context.
  const UserSgprLayout *Layout = nullptr;
  llvm::Function *Kernel;
  llvm::BasicBlock *ThreadLoopLatch = nullptr;

  llvm::Type *I1Ty;
  llvm::Type *I8Ty;
  llvm::Type *I32Ty;
  llvm::Type *I64Ty;
  llvm::Type *F32Ty;
  llvm::Type *F16Ty;
  llvm::Type *F64Ty;
  llvm::Type *PtrGlobalTy;

  llvm::DenseMap<uint64_t, llvm::BasicBlock *> &OffsetToBb;
  uint64_t KernelStartOffset = 0;
  uint64_t KernelEndOffset = 0;

  // Source KD private/scratch allocation. `handle-flat.cpp` sets
  // `usesScratchPrivateSegment` when it lowers a `scratch_*` instruction; the
  // on-demand private alloca is deliberately managed by LLVM's frame layout so
  // target spills cannot overlap translated source scratch slots.
  uint32_t SourcePrivateSegmentFixedSize = 0;
  uint32_t SourceComputePgmRsrc2 = 0;
  uint16_t SourceKernelCodeProperties = 0;
  bool UsesScratchPrivateSegment = false;
  llvm::AllocaInst *ScratchPrivateSegmentAlloca = nullptr;

  // Result of the static analysis that classifies every s_set_pc_i64
  // site. Owned by the raiser; the handler reads it to decide
  // between Pattern A (direct br) and Pattern B / DispatchSet
  // (cmp+br cascade via `emitEnumeratedDispatch`) lowerings, and
  // the raiser's main loop reads `chainTerminators` to materialise
  // call-site blockaddress writes into ret-pair SGPRs. See
  // setpc-analysis.h for the full contract; see canonical-op.h's
  // `S_SET_PC_I64` doc for the lowering shapes.
  const SetPcAnalysis *SetpcAnalysis = nullptr;

  // gfx1250 s_set_vgpr_msb state: only the LOW 8 bits of the instruction's
  // 16-bit immediate carry runtime meaning.  They encode the MSB bit pair for
  // every operand slot of the next ALU instruction:
  //
  //   [1:0]  src0 MSB        [3:2]  src1 MSB
  //   [5:4]  src2 MSB        [7:6]  vdst MSB
  //
  // Each 2-bit field adds (value * 256) to the corresponding VGPR index.
  //
  // The HIGH 8 bits of the `s_set_vgpr_msb` immediate record the PREVIOUS
  // mode value for compiler bookkeeping (see LLVM's AMDGPULowerVGPREncoding
  // pass, `setMode()`: "Record previous mode into high 8 bits of the
  // immediate."). The hardware ignores them, so we mask to 8 bits on store.
  //
  // For VOPD dual-issue instructions, the X-op and Y-op share the same MSB
  // pair per operand slot: LLVM asserts that if X-op and Y-op both reference
  // the same slot, they must carry identical MSBs ("Invalid VOPD pair was
  // created" in `computeMode`).  So applying the same 8-bit state to both
  // halves is correct.
  uint8_t VgprMsBs = 0;

  // Per-instruction VGPR index adjustment, indexed by MCInst operand index.
  // Computed from vgprMSBs before each instruction dispatch.
  static constexpr unsigned KMaxOps = 16;
  unsigned CurrentVgprAdjust[KMaxOps] = {};

  // Compute currentVGPRAdjust for the given instruction based on vgprMSBs.
  void computeVGPRAdjust(const DecodedInst &Di);

  llvm::BasicBlock *lookupBB(uint64_t Addr);

  ParsedReg parseReg(llvm::MCRegister Reg, int MciOpIdx = -1) const;

  // Operand reading -- mirrors the lambdas in the original raiseToIR.
  llvm::Value *readOp32(const DecodedInst &Di, unsigned OpIdx);
  llvm::Value *readOp64(const DecodedInst &Di, unsigned OpIdx);
  llvm::Value *readOpExecWidth(const DecodedInst &Di, unsigned OpIdx);

  // Emit `llvm.amdgcn.update.dpp.<i32>(old, src, ctrl, row_mask, bank_mask,
  // bound_ctrl)` -- the P5 lowering for src0-path DPP modifiers; see the
  // DPP row of hotswap/docs/wave-size-translation.md §5.3 for the rewrite
  // contract. `src` and `old` must be 32-bit (i32 or f32/bitcastable); a
  // future 64-bit lift would extend the intrinsic overload set and the
  // bitcast-bridge below. The return type matches `src->getType()` so
  // callers can feed the result back into the original instruction's
  // ALU path without reshuffling.
  //
  // `OpResolver::src(0)` / `srcF(0)` wrap their return through this helper
  // when `DecodedInst::hasDpp` is true, so handlers dispatched by CanonicalOp
  // stay DPP-agnostic. See `decoded-inst.h`'s DPP-state block for how
  // the modifier operand values reach us.
  llvm::Value *emitUpdateDpp(llvm::Value *OldVal, llvm::Value *Src,
                              uint16_t Ctrl, uint8_t RowMask,
                              uint8_t BankMask, bool BoundCtrl);

  // Materialise the target-hardware lane id (i32) at the current
  // builder insertion point, with per-BB memoisation. Mirrors the
  // `cachedLaneActive` pattern: handlers that need the lane id can
  // call `ctx.emitLaneIdx()` repeatedly within an instruction's
  // dispatch and reuse the same SSA value, avoiding redundant
  // `mbcnt_lo` / `mbcnt_hi` chains in the raw IR.
  //
  // The cache is invalidated on every source-instruction boundary
  // by the same `resetLaneActiveCache` call that resets
  // `cachedLaneActive`, since the dominance lifecycle is identical
  // (the cached i32 dominates the BB it was emitted in but not
  // arbitrary later BBs).
  //
  // Delegates to `projection.emitLaneIdx(B)` for the actual mbcnt
  // sequence -- `WaveProjection` remains the single source of truth
  // for *how* lane id is computed; `RaiseContext` only handles
  // caching.
  llvm::Value *emitLaneIdx();

  // ==== SIMT Predicated Execution (SPE) helpers
  // (see hotswap/docs/wave-size-translation.md §5.1). ================
  //
  // emitLaneActiveBit() returns an i1 true iff the current lane's bit in the
  // EXEC-mask alloca is set. Wave-size-aware via targetIsa: the lane id is
  // built from llvm.amdgcn.mbcnt.lo for wave32 and mbcnt.lo+mbcnt.hi for
  // wave64. The alloca itself is the current SSA-tracked EXEC value, so
  // uniform code (EXEC provably -1 by SROA) folds this to `true`.
  //
  // Caching: the result is memoised for the duration of a single decoded
  // instruction's dispatch (`cachedLaneActive` + `cachedLaneActiveBB`).
  // Handlers that emit multiple `emitUnderExec` diamonds for one source
  // instruction (e.g. a multi-dword MUBUF store writing four VGPRs) reuse
  // the same `lane_active` i1 instead of re-emitting the mbcnt / lshr /
  // and / icmp chain. Invalidated on:
  //   - Entry to every new source instruction (`resetLaneActiveCache`),
  //     because intervening instructions may have written EXEC.
  //   - Insertion-block change within a handler (the cached i1 no longer
  //     dominates emission points in a new BB).
  //   - Any explicit EXEC write via `ctx.storeExec`.
  // LLVM's mem2reg + CSE would clean up the redundancy anyway; the cache
  // keeps the raw raised IR readable for lit tests that FileCheck the
  // unoptimised output shape.
  llvm::Value *emitLaneActiveBit();

  // Invalidate the lane_active and lane_idx memoisations. Called by
  // the main raiser loop between instructions and by `storeExec`.
  // Handlers that know they have mutated EXEC through a lower-level
  // path (e.g. the few places that call `regs.storeExec` directly)
  // must also invoke this. The lane_idx cache shares the same
  // invalidation contract: per-BB lifetime, reset at every
  // instruction boundary so a handler that hops basic blocks (e.g.
  // SPE diamonds) never reads a cached SSA value out of its
  // dominance scope.
  void resetLaneActiveCache() {
    CachedLaneActive = nullptr;
    CachedLaneActiveBb = nullptr;
    CachedLaneIdx = nullptr;
    CachedLaneIdxBb = nullptr;
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
  // variants used when a handler already knows the register index (e.g.
  // VOPD's dstX/dstY pair or DS-transpose's contiguous sub-VGPR loop).
  void writeReg32(ParsedReg Pr, llvm::Value *V);
  void writeReg64(ParsedReg Pr, llvm::Value *V);
  void writeRegVec(ParsedReg Pr, llvm::Value *V);
  void writeRegExecWidth(ParsedReg Pr, llvm::Value *V);
  void storeVGPR32(int Idx, llvm::Value *V);
  void storeVGPR64(int Idx, llvm::Value *V);
  void storeAGPR32(int Idx, llvm::Value *V);

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

  // Memoised lane_active for this instruction's emission. Kept as public
  // members (rather than `private:`) so RaiseContext remains an aggregate
  // and can be brace-initialised from the raiser. Mutate only via
  // `resetLaneActiveCache` / `emitLaneActiveBit`.
  llvm::Value *CachedLaneActive = nullptr;
  llvm::BasicBlock *CachedLaneActiveBb = nullptr;

  // Memoised lane id for this instruction's emission, mirroring the
  // cachedLaneActive pair above. Used by `emitLaneIdx`. Same public-
  // field rationale (aggregate brace-init); mutate only via
  // `resetLaneActiveCache` / `emitLaneIdx`.
  llvm::Value *CachedLaneIdx = nullptr;
  llvm::BasicBlock *CachedLaneIdxBb = nullptr;

  // Per-BB cache of the per-lane i1 compare result produced by the
  // most recent V_CMP_*_e64 writer targeting a given SGPR in this
  // basic block. Keyed by source-ABI SGPR baseIdx (the low SGPR of
  // an SGPR pair on wave64 source; the single SGPR on wave32
  // source). `isPair` distinguishes a wave64-source pair entry
  // (the value spans [baseIdx, baseIdx+1]) from a wave32-source
  // single entry (baseIdx only), which matters for the adjacent-
  // invalidation rule in `invalidateSgprWaveMaskI1`.
  //
  // Motivation. The V_CMP -> SGPR store at `handle-valu-vcmp.cpp`
  // truncates the full target-hardware ballot down to the source
  // SGPR's physical 32-bit width (`ballotI1ToWidth(cmp, source
  // WaveMaskTy, ...)` -> `CreateTrunc`), destroying lanes 32..63's
  // compare results on wave32 source / wave64 target cross-
  // widening. A V_CNDMASK_B32_e64 consumer in the same BB has no
  // way to recover those bits from the narrow SGPR -- but it does
  // not need to, because the writer still holds the full per-lane
  // `i1` SSA value. This cache carries that `i1` across to the
  // consumer.
  //
  // Invariants (maintained in concert by
  //   * handle-valu-vcmp.cpp V_CMP -> SGPR path writes `cmp` here,
  //   * handle-valu-vop3p.cpp V_CNDMASK_B32 SGPR-source path reads,
  //   * AllocaRegFile::onSgprWritten invalidates on any SGPR write
  //     (with pair-aware adjacent invalidation for the high half
  //     of a pair rooted at baseIdx-1), and
  //   * `clearSgprWaveMaskShadow` drops the whole map on BB entry):
  //
  //   I1 - Additive. The narrow ballot store and its
  //        `extractLaneBitFromWaveMask` reader chain are both
  //        preserved. When the cache is absent, the consumer takes
  //        the existing (lossy-under-cross-widening) path. No case
  //        where the fix regresses a previously-correct lowering.
  //   I2 - SSA-monotonic within a BB. The SSA value returned by
  //        `lookupSgprWaveMaskI1(N)` is the exact `cmp` produced by
  //        the last V_CMP writer to sN in the current BB, with no
  //        intervening write to sN (or to sN+1 if the entry at sN
  //        is a pair). Linear handler dispatch over the
  //        instruction stream guarantees this.
  //   I3 - Any interference defeats the cache. Scalar writes
  //        invalidate the entry via the reg-file callback, AND
  //        invalidate a preceding pair entry whose high half this
  //        write just clobbered (see `invalidateSgprWaveMaskI1`);
  //        subsequent V_CMP writes overwrite the entry (last-writer
  //        wins); BB transition clears the whole map.
  //
  // Design rationale: see hotswap/docs/sgpr-wave-mask-translation.md
  // section 3.1 (chosen approach) and section 4 (the
  // widened-SGPR-storage alternative we deliberately did not
  // schedule).
  struct WaveMaskEntry {
    llvm::Value *I1 = nullptr;
    // `true` if this entry was recorded for a wave64-source V_CMP
    // whose destination is an SGPR pair [baseIdx, baseIdx+1]. `false`
    // for a wave32-source V_CMP whose destination is a single SGPR at
    // baseIdx. Consulted by `invalidateSgprWaveMaskI1` to decide
    // whether a write to baseIdx+1 should also invalidate the entry
    // at baseIdx (pair-aware half-overwrite protection).
    bool IsPair = false;
  };

  // Default-construct via an explicit zero-arg temporary to disambiguate
  // from DenseMap's `explicit` single-unsigned-int constructor under
  // RaiseContext's aggregate brace-init. (With `= {}` or no initializer
  // GCC warns: "converting … from initializer list would use explicit
  // constructor DenseMap(unsigned int)".)
  llvm::DenseMap<int, WaveMaskEntry> LastSgprWaveMaskI1 =
      llvm::DenseMap<int, WaveMaskEntry>();

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
          B, CmpI1, Regs.ExecTy, "wm_shadow_exec");
      B.CreateStore(ExecMask, SgprWaveMaskExecShadow[BaseIdx]);
      B.CreateStore(B.getTrue(), SgprWaveMaskValidShadow[BaseIdx]);
    }
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
    if (BaseIdx < 0 || static_cast<size_t>(BaseIdx) >= SgprWaveMaskExecShadow.size())
      return nullptr;
    return B.CreateLoad(Regs.ExecTy, SgprWaveMaskExecShadow[BaseIdx],
                        "sgpr_mask_exec");
  }

  llvm::Value *loadSgprWaveMaskValid(int BaseIdx) const {
    if (BaseIdx < 0 || static_cast<size_t>(BaseIdx) >= SgprWaveMaskValidShadow.size())
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
    LastSgprWaveMaskI1.erase(BaseIdx);
    if (BaseIdx >= 0 &&
        static_cast<size_t>(BaseIdx) < SgprWaveMaskValidShadow.size())
      B.CreateStore(B.getFalse(), SgprWaveMaskValidShadow[BaseIdx]);
    if (BaseIdx > 0) {
      auto Prev = LastSgprWaveMaskI1.find(BaseIdx - 1);
      if (Prev != LastSgprWaveMaskI1.end() && Prev->second.IsPair) {
        LastSgprWaveMaskI1.erase(Prev);
        if (static_cast<size_t>(BaseIdx - 1) < SgprWaveMaskValidShadow.size())
          B.CreateStore(B.getFalse(), SgprWaveMaskValidShadow[BaseIdx - 1]);
      }
    }
  }

  // Drop every cached entry. Called at every BB boundary in the
  // raiser's main loop (alongside `vgprMSBs = 0`) so that cross-BB
  // V_CMP / V_CNDMASK pairs conservatively fall back to the narrow
  // extract rather than relying on an i1 that no longer dominates
  // the consumer. A future reaching-definitions pass on the raised
  // IR (see sgpr-wave-mask-translation.md section 7 evolution
  // path) can upgrade this to a proper per-BB merge.
  void clearSgprWaveMaskShadow() { LastSgprWaveMaskI1.clear(); }

  void collectSgprWaveMaskShadowAllocas(
      llvm::SmallVectorImpl<llvm::AllocaInst *> &Out) const {
    Out.append(SgprWaveMaskExecShadow.begin(), SgprWaveMaskExecShadow.end());
    Out.append(SgprWaveMaskValidShadow.begin(), SgprWaveMaskValidShadow.end());
  }

  // Pending failure raised during operand-read dispatch (e.g.
  // `readOp32` / `readOp64` encountering an unmodeled aperture
  // register such as SRC_SHARED_BASE / SRC_FLAT_SCRATCH_BASE_LO).
  // Read paths cannot bail mid-handler -- they must return some
  // Value* -- so they record the failure here and the per-instruction
  // dispatch loop in `raiser.cpp` checks `pendingFailure` after each
  // handler returns and aborts the kernel raise. Set via
  // `recordReadFailure`; cleared per instruction by the dispatch
  // loop after consumption (or before the next instruction starts).
  RaiseFailure PendingFailure;

  // Record an operand-read failure. Only the first failure per
  // instruction is captured; subsequent reads of the same kernel
  // simply return undef and let the dispatch loop bail on the
  // first one we observed.
  void recordReadFailure(RaiseFailure F) {
    if (!PendingFailure.hasFailed())
      PendingFailure = std::move(F);
  }
};

// Return value from every format handler.
//
// Handlers communicate back in three ways:
//   * `handled = true` -> the handler fully lowered the instruction.
//   * `handled = false`, `failure.reason = None` -> this handler does
//     not claim the instruction; the main loop falls through to the
//     generic `UnsupportedOpcode` diagnostic.
//   * `handled = false`, `failure.reason != None` -> the handler
//     recognised the instruction but refuses to lower it (e.g. operand
//     shape unsupported); the main loop records the structured failure
//     and aborts without consulting other handlers.
struct HandlerResult {
  bool Handled = false;
  llvm::Value *SccResult = nullptr;
  bool SccHandled = false;
  RaiseFailure Failure;
};

// Reads source operands via srcMap, skipping VOP3 modifiers. If the
// decoded instruction carries a DPP modifier (`DecodedInst::hasDpp`),
// `src(0)` / `srcF(0)` / `src64(0)` transparently wrap their result
// through `RaiseContext::emitUpdateDpp` using the modifier operand
// values from `di`, so handlers dispatched on the canonicalised
// CanonicalOp stay DPP-agnostic. See `decoded-inst.h`'s DPP-state block
// for the data-flow contract.
struct OpResolver {
  RaiseContext &Ctx;
  const DecodedInst &Di;

  unsigned srcIdx(unsigned I) const {
    assert(I < Di.NumSrcs && "source index out of range");
    return Di.SrcMap[I];
  }
  unsigned nSrcs() const { return Di.NumSrcs; }

  unsigned srcMod(unsigned I) const {
    unsigned ModIdx = Di.ModMap[I];
    if (ModIdx == UINT_MAX) return 0;
    if (!Di.isImm(ModIdx)) return 0;
    return static_cast<unsigned>(Di.getImm(ModIdx) & 0xF);
  }

  llvm::Value *applyMods(unsigned I, llvm::Value *V) {
    unsigned Mods = srcMod(I);
    if (Mods == 0) return V;
    bool IsI32 = (V->getType() == Ctx.I32Ty);
    if (IsI32) V = Ctx.B.CreateBitCast(V, Ctx.F32Ty);
    if (Mods & 2)
      V = Ctx.B.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, V, nullptr, "abs");
    if (Mods & 1)
      V = Ctx.B.CreateFNeg(V, "neg");
    if (IsI32) V = Ctx.B.CreateBitCast(V, Ctx.I32Ty);
    return V;
  }

  // DPP src-path wrapping. DPP is a src0-only data-pathway modifier: on
  // hardware, src0 is shuffled across lanes per the DPP control bits
  // *before* any fmods (neg/abs) and before the ALU op consumes it.
  // Callers apply ordering: DPP wrap -> applyMods -> ALU.
  //
  // `%old` -- the intrinsic's "inactive-lane value" operand -- is the
  // current VGPR value of the instruction's vdst (MCInst operand 0);
  // we read it lazily and cache within this OpResolver because
  // handlers that emit multiple side effects for a single source
  // instruction may call `src(0)` more than once (e.g. VOPD's two
  // halves). `OpResolver` instances are short-lived so the cache is
  // strictly scoped.
  //
  // Width dispatch: `src(0)` / `srcF(0)` go through the 32-bit read
  // path and cache `cachedDppOldVdst32`; `src64(0)` goes through the
  // 64-bit read path and caches `cachedDppOldVdst64`. Handlers that
  // mix widths on the same instruction (which would be unusual -- a
  // VALU op reads its src0 at a single width) would hit both caches
  // but each lookup stays coherent. `emitUpdateDpp` supports both
  // widths; other widths abort loudly.
  llvm::Value *wrapDppIfNeeded(unsigned LogicalSrc, llvm::Value *Raw) {
    if (!Di.HasDpp || LogicalSrc != 0) return Raw;
    if (rejectDpp16FetchInactiveIfNeeded()) return Raw;
    if (!CachedDppOldVdst32)
      CachedDppOldVdst32 = Ctx.readOp32(Di, 0);
    return Ctx.emitUpdateDpp(CachedDppOldVdst32, Raw, Di.DppCtrl, Di.DppRowMask,
                              Di.DppBankMask, Di.DppBoundCtrl);
  }

  llvm::Value *src(unsigned I) {
    return wrapDppIfNeeded(I, Ctx.readOp32(Di, srcIdx(I)));
  }
  llvm::Value *srcF(unsigned I) {
    return applyMods(I, wrapDppIfNeeded(I, Ctx.readOp32(Di, srcIdx(I))));
  }
  llvm::Value *src64(unsigned I) {
    llvm::Value *Raw = Ctx.readOp64(Di, srcIdx(I));
    if (!Di.HasDpp || I != 0)
      return Raw;
    if (rejectDpp16FetchInactiveIfNeeded()) return Raw;
    if (!CachedDppOldVdst64)
      CachedDppOldVdst64 = Ctx.readOp64(Di, 0);
    return Ctx.emitUpdateDpp(CachedDppOldVdst64, Raw, Di.DppCtrl, Di.DppRowMask,
                              Di.DppBankMask, Di.DppBoundCtrl);
  }
  llvm::Value *srcExecWidth(unsigned I) { return Ctx.readOpExecWidth(Di, srcIdx(I)); }
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

  // Memoised src-0 %old values for DPP wrapping, one per supported
  // bit-width (32 / 64). Each is populated lazily on the first
  // `src(0)` / `srcF(0)` / `src64(0)` call for a DPP-flagged
  // instruction. Kept public so `OpResolver` remains an aggregate and
  // can be brace-initialised from the raiser
  // (`OpResolver op{ctx, di};`). Mutate only through the src*
  // wrappers. Mirrors the `cachedLaneActive` public-field convention
  // on `RaiseContext` above.
  llvm::Value *CachedDppOldVdst32 = nullptr;
  llvm::Value *CachedDppOldVdst64 = nullptr;

private:
  /// `llvm.amdgcn.update.dpp` has no FI operand. Record refusal when
  /// Table-57 fetch-inactive is requested; callers must return their
  /// `Raw` value unchanged when this returns true.
  bool rejectDpp16FetchInactiveIfNeeded() {
    if (!Di.DppFi)
      return false;
    Ctx.recordReadFailure(RaiseFailure::unsupportedInstructionForm(
        Di, "DPP",
        "DPP16 FI fetch-inactive form: llvm.amdgcn.update.dpp has no FI "
        "operand; extending DPP lifting requires modelling Table 57 "
        "fetch-inactive semantics rather than silently dropping FI"));
    return true;
  }
};

} // namespace COMGR::hotswap

#endif
