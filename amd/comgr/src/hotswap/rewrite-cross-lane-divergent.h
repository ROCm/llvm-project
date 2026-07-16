//===- rewrite-cross-lane-divergent.h - Hotswap transpiler ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_REWRITE_CROSS_LANE_DIVERGENT_H
#define HOTSWAP_TRANSPILER_REWRITE_CROSS_LANE_DIVERGENT_H

#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
class Function;
class TargetMachine;
} // namespace llvm

namespace COMGR::hotswap {

// ============================================================================
// Post-raise rewrite: per-source-wave cross-lane primitives under
// cross-widen divergence. See hotswap/docs/wave-size-translation.md
// sec. 5.6.3 for the principled derivation of the rewrite shapes below.
// ============================================================================
//
// PROBLEM. The AMDGPU backend lowers `llvm.amdgcn.writelane(val, lane,
// old)` / `llvm.amdgcn.readlane(src, lane)` through the
// `v_writelane_b32` / `v_readlane_b32` machine opcodes. Those opcodes
// accept the value / src as an *SGPR*, and the backend inserts an
// implicit `v_readfirstlane_b32` on any divergent-in-a-VGPR input to
// scalarise it. Under cross-widening (source wave32 -> target wave64),
// a value that is *intentionally* per-lane-divergent within a target
// wave -- e.g. the `wave_id_in_workgroup` divergent-VGPR lifted out of
// `s_bfe_u32 ttmp8, 0x50019` (handle-sop2.cpp sec. 5.6.2 rescue) -- gets
// collapsed by the readfirstlane to one lane's value, and the other
// target lanes lose their per-source-wave meaning. Every downstream
// computation keyed on the collapsed value (tile-column address, EXEC
// predicate, scalar broadcast) then reads the wrong source wave's data,
// which miscompiles every Matmul128x128-class kernel on gfx1250 ->
// gfx942 translation.
//
// The same cross-widen fault class extends to `@llvm.amdgcn.update.dpp`:
// source gfx1250 (wave32) reduction trees encode their per-step
// `dpp_ctrl` / `row_mask` / `bank_mask` / `bound_ctrl` assuming a
// 32-lane wave topology (2 rows of 16 lanes).  Preserving the same
// bits verbatim on wave64 (4 rows of 16 lanes) keeps within-16-lane-
// row shifts semantically correct BUT leaves target-wave-specific
// mask-write semantics divergent from the wave64-native reduction
// tree Triton would have compiled at the target -- producing byte-
// different results on kernels whose reduction accumulators feed
// downstream arithmetic (softmax, tl.sum(axis=1), every 32-col-or-
// wider reduction; pinned by the `topk_forward_bisect_*` recipe
// family in `compare_correctness`).  The DPP case is included in
// this pass under the same rewrite invariant: under cross-widening,
// every cross-lane primitive in the lifted IR is rewritten to an
// ISA-neutral `ds_bpermute + select` form whose correctness depends
// only on ds_bpermute's explicit per-lane read semantics (stable
// across gfx9+) rather than on target ISA and source ISA sharing
// the same mask-bit interpretation.
//
// REWRITE. Replace the cross-lane primitive with a per-source-wave
// shape that keeps the scalar operand in a VGPR and preserves per-
// source-wave state across the target wave:
//
//   * `writelane(val, lane_idx, old)` ->
//       `select ((lane_id & (W_s-1)) == lane_idx), val, old`
//     Semantics: within each virtual source wave (each contiguous
//     W_s lanes of the target wave), the lane whose source-wave
//     relative index equals `lane_idx` receives `val`'s per-lane
//     value; the other W_s-1 lanes keep `old`'s per-lane value. This
//     matches the source kernel's single-wave writelane exactly
//     under modulo-replication / wave-native projection.
//
//   * `readlane(src, lane_idx)` ->
//       `ds_bpermute(selector = ((lane_id & ~(W_s-1)) | lane_idx) << 2,
//                    src)`
//     Semantics: broadcast lane `lane_idx` of the current source wave
//     to all lanes in that source wave. The selector `<< 2` multiplier
//     is `ds_bpermute`'s byte-to-lane offset convention
//     (IntrinsicsAMDGPU.td comment on `int_amdgcn_ds_bpermute`); the
//     base mask `~(W_s-1)` aligns the selector to the source-wave
//     boundary so each source wave broadcasts only within itself.
//
//   * `update.dpp(old, src, dpp_ctrl, row_mask, bank_mask,
//                 bound_ctrl)` ->
//       per-lane `ds_bpermute(srcLaneAbs << 2, src)` + `select`
//       chain.  Per-target-lane L, with row = (L >> 4) & 3, bank =
//       (L >> 2) & 3, withinRow = L & 0xF:
//         srcWithinRow, inRange = decode(dpp_ctrl, withinRow)
//         srcLaneAbs            = (L & ~0xF) | srcWithinRow
//         bperm                 = ds_bpermute(srcLaneAbs << 2, src)
//         oob                   = bound_ctrl ? 0 : old
//         dppVal                = inRange ? bperm : oob
//         laneActive            = rowMaskBit(row) & bankMaskBit(bank)
//         result                = laneActive ? dppVal : old
//     Supported dpp_ctrl values (covers the observed Triton corpus):
//     `quad_perm[*]` (0x000..0x0FF), `row_shl:1..15` (0x101..0x10F),
//     `row_shr:1..15` (0x111..0x11F).  All three families stay within
//     a 16-lane row, so the rewrite's srcLaneAbs computation is
//     wave-size-oblivious -- the source-wave boundary between row 0
//     and row 1 (lanes 0..15 vs 16..31) is the same topological bit
//     on wave32 and wave64.  Other dpp_ctrl families (row_rotate,
//     row_mirror, row_half_mirror, row_share, row_xmask, and the
//     wave-wide / bcast variants) DO shift across row-pair
//     boundaries in a wave-size-dependent way; the rewrite refuses
//     those loudly via `unsupportedDppDetail` rather than producing
//     a silently-wrong `ds_bpermute` expansion.  The faithful lift
//     path (`@llvm.amdgcn.update.dpp` passed through to the backend)
//     remains for i64 DPP operands, which AMDGPU's backend splits
//     into two i32 DPP ops internally -- the rewrite's i32-only scope
//     is intentional since every Triton reduction corpus we have
//     uses i32 DPP exclusively.
//
// WRITELANE / READLANE SYMMETRY. ALL writelane and ALL readlane sites
// are rewritten under cross-widening, independent of whether operands
// are divergent or uniform. The `select` / `ds_bpermute` forms are
// semantically equivalent to the source opcodes for every combination
// of operand divergence (uniform operands agree on lanes `N` and
// `N + W_s`; divergent operands carry their per-source-wave split),
// and unconditional rewriting is the only rule that keeps a
// writelane/readlane pair on a shared VGPR self-consistent.
//
// Asymmetric rewriting (the pre-symmetry rule that preserved
// "uniform" sites as native `v_writelane_b32` / `v_readlane_b32`) is
// silently unsound: a preserved uniform writelane writes hardware
// lane `N` only, leaving lane `N + W_s` at its prior value (undef for
// a fresh spill slot), while a sibling `ds_bpermute`-rewritten
// readlane on the same VGPR reads BOTH `N` and `N + W_s`. The upper
// replica returns undef, which flows into downstream address math and
// faults the kernel with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
// (observed historically on matmul cross-widening kernels after
// WaveNative graduation).
//
// USE-CHAIN CONSTRAINT. The rewrite is correct for any kernel whose
// writelane / readlane results flow only into VGPR-safe consumers
// (global / scratch / LDS memory ops, VALU arithmetic, other
// rewritten cross-lane primitives, branches, etc.). If any site's
// result transitively reaches an SGPR-constrained operand --
// `llvm.amdgcn.s.buffer.load` rsrc, `llvm.amdgcn.s.sendmsg` message,
// an explicit `llvm.amdgcn.readfirstlane`, a `load` from
// addrspace(4), inline asm with `"s"` constraint, or any unknown
// sink the classifier cannot prove safe -- the backend will insert
// `v_readfirstlane` on the `ds_bpermute` result and re-introduce the
// cross-source-wave collapse that the rewrite was designed to
// eliminate. We detect this pre-rewrite via `classifyUseChain`
// (internal to `rewrite-cross-lane-divergent.cpp`) and refuse the
// whole function by returning a populated
// `refusal.sgprForcedDetail`. The refusal is principled per the
// project's no-silent-miscompile contract: if we cannot prove every
// consumer is VGPR-safe, we do not emit a best-effort rewrite.
//
// PRECONDITION. Run AFTER `PromoteMemToReg` so the use-chain
// classifier operates on post-mem2reg SSA: scratch-addrspace allocas
// that would otherwise obscure a use chain through a round-trip
// load/store are gone. The rewrite assumes `writelane` / `readlane`
// are overloaded on `i32` (the only shape the raiser emits today --
// see `handle-valu-cross-lane.cpp`'s `V_WRITELANE_B32` /
// `V_READLANE_B32` branches).
//
// DIRECTION GATE. Only runs when `targetWaveSize > sourceWaveSize`
// (cross-widening). Same-wave and narrowing directions leave every
// site unchanged -- the implicit readfirstlane does not collapse
// per-source-wave state when a source wave *is* a target wave.

enum class SgprForcedConsumerKind {
  None,
  ExplicitReadFirstLane,
  ScalarSendMsg,
  ScalarMemoryOperand,
  ConstantAddressSpaceMemory,
  InlineAsm,
  IndirectCall,
  OrdinaryCall,
  Unknown,
};

struct CrossLaneDivergentRewriteReport {
  // Number of `amdgcn.writelane` calls rewritten to a per-lane
  // `select`. Under cross-widening with a VGPR-safe use chain this
  // equals the total number of writelane sites in the function.
  unsigned WritelaneRewritten = 0;

  // Number of `amdgcn.readlane` calls rewritten to a `ds_bpermute`.
  // Under cross-widening with a VGPR-safe use chain this equals the
  // total number of readlane sites in the function.
  unsigned ReadlaneRewritten = 0;

  // Number of `amdgcn.update.dpp` calls rewritten to a `ds_bpermute`
  // + `select` chain. Under cross-widening with a VGPR-safe use
  // chain AND all-supported dpp_ctrls this equals the total number
  // of i32 DPP sites in the function. i64 DPP sites pass through
  // unmodified (see the `@llvm.amdgcn.update.dpp` section of the
  // header comment above).
  unsigned DppRewritten = 0;

  // Non-empty iff the use-chain classifier rejected at least one
  // writelane / readlane / DPP site because a transitive consumer
  // requires an SGPR (e.g. `s_buffer_load` rsrc, `s_sendmsg` message,
  // `readfirstlane`, addrspace(4) load, inline asm `"s"`, or any
  // unknown sink). When populated: no sites were rewritten (refusal
  // is all-or-nothing to avoid the asymmetric-rewrite trap), the
  // raiser translates this into a
  // `RaiseFailure::crossWaveRewriteOracleDisagreement` refusal, and
  // the detail string names the offending intrinsic / memory op so
  // the next investigation knows where to start.
  std::string SgprForcedDetail;
  SgprForcedConsumerKind SgprForcedKind = SgprForcedConsumerKind::None;
  bool SgprForcedThreadLoopEligible = false;

  // Non-empty iff the DPP-rewrite encountered a `dpp_ctrl` value
  // outside the supported family (quad_perm / row_shl / row_shr).
  // The detail string names the offending ctrl value. When
  // populated: no DPP sites were rewritten (refusal is all-or-nothing
  // across the pass's primitive families to preserve the symmetry
  // invariant with writelane / readlane), the raiser translates this
  // into a refusal, and the next step is either widening the
  // supported-ctrl table in `buildDppLaneMap` (after proving the new
  // ctrl is wave-size-oblivious within a 16-lane row) or refusing
  // the kernel at raise time as genuinely untranslatable.
  std::string UnsupportedDppDetail;

  // Total sites rewritten. Callers use this only for diagnostic
  // formatting -- under the symmetry contract every reachable site is
  // rewritten (or the whole function refuses), so `totalRewritten()
  // == 0` on a non-empty-site kernel always implies at least one of
  // `sgprForcedDetail` / `unsupportedDppDetail` is populated.
  unsigned totalRewritten() const {
    return WritelaneRewritten + ReadlaneRewritten + DppRewritten;
  }

  // True iff the classifier refused due to an SGPR-forced consumer.
  // When true, the rewrite pass performed zero rewrites and the
  // caller must surface `sgprForcedDetail` as a raise-time refusal
  // diagnostic.
  bool refusedSgprForced() const { return !SgprForcedDetail.empty(); }

  // True iff the DPP-rewrite encountered an unsupported `dpp_ctrl`.
  // When true, the rewrite pass performed zero rewrites and the
  // caller must surface `unsupportedDppDetail` as a raise-time
  // refusal diagnostic.
  bool refusedUnsupportedDpp() const { return !UnsupportedDppDetail.empty(); }

  // True iff the pass refused the function for any reason.
  bool refused() const {
    return refusedSgprForced() || refusedUnsupportedDpp();
  }
};

// Rewrite every cross-lane primitive site in `F` under cross-widening
// IF the forward use-chain classifier proves every site VGPR-safe.
// Otherwise perform zero rewrites and return the refusal detail in
// `sgprForcedDetail`. No-op when `targetWaveSize <= sourceWaveSize`.
//
// `ProvidesFullWaveExecInvariant` says whether the active projection
// guarantees hardware EXEC = -1 kernel-wide (i.e.
// `WaveProjection::providesFullWaveExecInvariant()`, true only for
// `WaveNativeProjection` via its kernel-entry `init_whole_wave`).  It
// governs the whole-wave gather for the `readlane` / `readfirstlane`
// rewrites: the source ops ignore EXEC (they read the selected lane's
// real VGPR even when that lane is inactive), but `ds_bpermute`
// returns 0 for a read of an EXEC-inactive lane.  When the invariant
// does NOT hold (MODREP / ThreadLoop, where hardware EXEC tracks the
// modeled per-lane active mask), the rewritten `ds_bpermute` gather is
// wrapped in `@llvm.amdgcn.strict.wwm` so the backend runs it under
// HW EXEC = -1 and every lane stages its real value.  When the
// invariant holds the wrap is omitted (it would be redundant and the
// `SIPreAllocateWWMRegs` pressure is best avoided).
//
// `TM` is an optional (default-null) handle to the compilation
// target's `TargetMachine`. It is currently unused -- the classifier
// operates purely on the lifted IR's structural shape. The parameter
// is kept in the signature so callers (`raiser.cpp`) can pass the
// already-constructed target machine, and any future refinement that
// wants to consult a target-aware analysis (`UniformityAnalysis`,
// `TargetTransformInfo`, etc.) can wire it in without a signature
// change. A prior attempt to use UA directly for a
// `readfirstlane`-in-chain allow-gate was shown unsound under
// modulo-replication (pre-rewrite UA sees `amdgcn.readlane` as
// `AlwaysUniform`, but the rewrite lowers it to a source-wave-scoped
// `ds_bpermute` whose result is MODREP-replica-divergent -- so a
// pre-rewrite UA verdict does not survive the rewrite), so no such
// refinement is active today.
llvm::Expected<CrossLaneDivergentRewriteReport>
rewriteCrossLaneDivergent(llvm::Function &F, unsigned SourceWaveSize,
                          unsigned TargetWaveSize,
                          bool ProvidesFullWaveExecInvariant = false,
                          llvm::TargetMachine *TM = nullptr);

} // namespace COMGR::hotswap

#endif
