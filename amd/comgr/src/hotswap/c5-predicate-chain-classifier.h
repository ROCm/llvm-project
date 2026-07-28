//===- c5-predicate-chain-classifier.h - Hotswap transpiler ---------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_C5_PREDICATE_CHAIN_CLASSIFIER_H
#define HOTSWAP_TRANSPILER_C5_PREDICATE_CHAIN_CLASSIFIER_H

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace llvm {
class Function;
} // namespace llvm

namespace COMGR::hotswap {

// ============================================================================
// Narrow-O1 post-mem2reg IR-level classifier for the Class-5
// predicate-chain class. See hotswap/docs/modrep-predicate-chain.md sec. 5
// (O1) for the design and narrowing rationale.
// ============================================================================
//
// PROBLEM. Under modulo-replication (`ModuloReplicationProjection`),
// a wave32 source kernel's `llvm.amdgcn.workitem.id.x()` value
// reaches an `icmp` against a compile-time constant K with
// `0 < K <= W_s - 1`, where the icmp result gates a global-memory
// side effect. The predicate is structurally "which lane-position-
// within-source-wave am I" -- source wave 0's lane L and target
// replica-1's lane L+W_s share the same source EXEC bit but have
// different architectural `tid` values, so `tid < K` evaluates
// differently between the replicas. Scan-stage guards
// (`tid >= 2^s` for Kogge-Stone), half-wave broadcasts
// (`if tid < W_s/2`), and quad-level masks (`tid & 3`) all fit
// this shape. The `canary_bpermute_scan_fp32` recipe
// (`compare_correctness` Triton suite) is the canary that pins it.
//
// NARROWING RULE (the key principled bit -- Phase-2 evidence in
// `modrep-predicate-chain.md` sec. 5 O1). "Refuse any unmasked
// `tid -> icmp -> side-effect`" -- the literal O1 wording -- would also
// refuse currently-passing baselines (`vecadd_f16`, `rope_fp32`,
// elementwise Triton kernels) whose IR has structurally identical
// shapes but compares against a *dynamic* kernarg (`tid < N` with
// N a runtime size). The constant-K-only rule distinguishes
// scan-stage / broadcast / lane-position gates (refuse) from
// bounds-against-kernarg (don't refuse).
//
// TWO-PASS WALK. A single forward pass from `tid` with an
// `otherOperand(icmp, V)` check works for the straightforward
// `icmp tid, K` / `icmp f(tid), K` shapes, but couples the
// walker's operand-ordering heuristics to the refusal rule. The
// two-pass form decouples them: Pass 1 builds a tid-reachable
// value set once, Pass 2 scans all icmps and applies the rule
// operand-by-operand against that set.
//
//   Pass 1: forward-walk from each `@llvm.amdgcn.workitem.id.x()`
//           call, tagging every tid-reachable value as either
//           "unmasked" (still carries replica-1-vs-source-wave-0
//           divergence) or "masked" (passed through `and V,
//           K<=W_s-1` on this walk path, collapsing the value onto
//           `[0, W_s)`). A value can be in both sets when it joins
//           a masked arm and an unmasked arm (e.g. via a phi).
//
//   Pass 2: scan every `icmp` in the function. Refuse iff the icmp
//           has BOTH (a) at least one operand in `unmaskedVisited`
//           AND (b) at least one operand that is a compile-time
//           constant K in `(0, W_s-1]`. `maskedVisited` is tracked
//           by Pass 1 only so the walker can distinguish the
//           collapsed path from the divergent one; it is NOT
//           sufficient on its own to drive refusal (see design
//           note below).
//
// DESIGN NOTE: why not refuse on `unmasked + masked` cross-subtree.
// An earlier iteration of this classifier also refused icmps of
// the form `icmp f(tid)_unmasked, g(tid)_masked` on the theory
// that replica-0 and replica-1 evaluate the predicate on different
// `f` values while agreeing on `g` values. The compare_correctness
// hotswap-path sweep on 2026-04-21 falsified this for `ult`/`ugt`/
// `slt`/`sgt`/`ule`/etc.: the baseline kernels `vecadd_f16`,
// `corpus_add_fp32`, `corpus_asin_fp32`, and
// `canary_dpp_reduce_fp32` all emit that exact shape (Triton's
// bounds-check against a tid-derived end bound) and empirically
// produce identical outputs across replicas under MODREP because
// the specific tid-value ranges make the inequality evaluate
// consistently. Only `eq`/`ne` variants actually diverge -- and
// tightening per-predicate requires value-range reasoning the
// classifier does not do today. The safe direction is to leave
// cross-subtree shapes un-refused and let `compare_correctness`
// be the end-to-end fence for any kernel that surfaces a genuine
// `eq`/`ne` miscompile (none currently known in the corpus).
//
// Walk rules for Pass 1:
//
//   * `and %v, K` with `K <= W_s - 1`: walk transitions into the
//     masked state. The result enters `maskedVisited`. (This is the
//     sec. 5.6.2 `wave_id` lift's mask shape and the SPE prelude's own
//     `lane_id & (W_s-1)` -- both produce safe values.)
//   * Pure propagators (cast, binop incl. non-masking `and`, phi,
//     select, gep, freeze, insert/extract{element,value},
//     shufflevector, and the numeric intrinsics
//     `llvm.{smin,smax,umin,umax,abs,ctlz,cttz,ctpop,bitreverse,
//     bswap,fshl,fshr,sadd_sat,ssub_sat,uadd_sat,usub_sat}`): push
//     the result onto the worklist, carrying the current
//     masked/unmasked state forward. The intrinsic list is
//     explicit rather than defaulting every call to propagator --
//     unknown intrinsics fall into the "sink" bucket where we
//     neither propagate nor refuse.
//   * `icmp`: the walker does not propagate through the `i1`
//     result; Pass 2 audits icmps.
//   * Non-Instruction users: structurally unreachable (ConstantExpr
//     operands must be Constants, not Instructions). If the branch
//     ever fires we refuse loudly -- unknown IR shape the
//     classifier cannot reason about.
//   * Unknown Instruction users (store, load, call, branch, return,
//     atomic, unknown intrinsics): sink. Stop walking this user.
//     The narrow-O1 class scopes refusal to the icmp shape; stores
//     of tid-derived values are the baseline pattern for
//     `vecadd_f16` / `rope_fp32` and are intentionally not refused.
//
// DIRECTION GATE. Runs only when `targetWaveSize > sourceWaveSize`.
// Same-wave or narrowing: no replica-1 exists, no lane-position
// ambiguity, the classifier is a structural no-op.
//
// COVERAGE + INTENTIONAL NON-COVERAGE. Per sec. 5 O1 of the design doc:
//
//   * `canary_bpermute_scan_fp32`: Kogge-Stone scan stages emit
//     `icmp ult i32 K, %tid` with K in {1, 3, 7, 15} (all
//     `<= W_s - 1 = 31`). Classifier refuses when MODREP can have
//     active replica lanes. Under the WaveNative default the refusal
//     is short-circuited (see `projection` docstring below); the
//     recipe's previously-observed WRONG
//     numerics turned out to be an orthogonal VOPD-cndmask bug
//     (design-doc sec. 6.4), fixed independently of this classifier.
//   * `rmsnorm_fp32`, `swiglu_fp32`, `corpus_layernorm_fp32`:
//     their kernel-level icmps compare `tid` against a dynamic
//     kernarg (`%arg4` / `%arg2`), not a compile-time constant.
//     Classifier does NOT refuse. Their end-to-end status is
//     MATCH today; the relevant fixes were orthogonal to this
//     classifier (see design-doc sec. 6.3 / sec. 6.4).
//   * `vecadd_f16`, `rope_fp32`, `canary_dpp_compound_add_fp32`
//     (baselines): no icmps with the C5 shape. Classifier does
//     NOT refuse. Baselines stay green.
//
// RELATIONSHIP TO sec. 5 O2 (the mask rewrite). O2 is deferred
// indefinitely per sec. 6.2 of the design doc: the mask's shape is
// semantically incorrect for multi-warp kernels and a structural
// no-op for single-warp sub-case 2. This classifier is
// refuse-only; there is no `RewriteId` entry paired with
// `ObstructionKind::WorkitemIdPredicateChain` today.

enum class PredicateChainProjection {
  ModuloReplication,
  WaveNative,
  ThreadLoop,
  // Scaled dispatch: each target wave hosts one source wave with the upper
  // lanes as replicas, so a lane and its replica share every predicate. Never
  // refuses C5. See `ScaledModuloReplicationProjection`.
  ModuloReplicationScaled,
};

struct PredicateChainClassifierReport {
  // True iff the classifier refused at least one icmp under the
  // current projection. When true, the raiser translates this into a
  // `RaiseFailure::crossWavePredicateChain` refusal. The decision is
  // projection- and launch-regime-aware: MODREP refuses only when an
  // active target replica lane can exist, WaveNative refuses only for
  // the defensive phantom-lane sub-case, and ThreadLoop suppresses only
  // for its separately-proven retry route.
  bool Refused = false;

  // When `refused == true` AND the trigger was the defensive
  // phantom-lane sub-case of the WaveNative arm (not the baseline
  // MODREP arm), `refusalDetail` is augmented with the phantom-lane
  // explanation so the diagnostic names the specific cause. This bit
  // lets callers (and the lit fixtures) discriminate between the two
  // refusal paths without string-matching the detail.
  bool WaveNativePhantomRefusal = false;

  // True iff WaveNative refused specifically because an equality predicate
  // (`eq`/`ne`) matched the C5 lane-position shape.  This class can be retried
  // under ThreadLoopProjection, which keeps per-source-wave predicate masks
  // distinct instead of truncating them through a single source-width SGPR.
  bool WaveNativeEqualityRefusal = false;

  // True iff WaveNative refused on a `workitem.id.y()`/`.z()`-derived predicate
  // reaching a memory address. The raiser retries such kernels under
  // `ScaledModuloReplicationProjection` when they are eligible.
  bool WaveNativeYzRefusal = false;

  // True iff a WaveNative equality (`eq`/`ne`) C5 site was observed. These
  // sites are accepted only by the target-width mask-shadow contract; callers
  // surface the count/reason in proof logs so acceptance is explicit.
  bool WaveNativeEqualityObserved = false;

  // Detail string for the first refused site. Empty iff `!refused`.
  // Stable-enough-for-lit substring format (see lit fixtures under
  // `lit_tests/c5_predicate_chain_*`).
  std::string RefusalDetail;

  // All C5-shape sites the classifier observed, regardless of
  // whether they were refused. Populated on suppressed projection
  // routes so callers can emit attribution breadcrumbs and so a
  // future iteration can widen the suppression-vs-refusal decision
  // without rerunning Pass 2. Each entry is the same kind of string
  // that `refusalDetail` carries.
  llvm::SmallVector<std::string> ObservedSites;

  // Non-empty iff `!refused && !observedSites.empty()`. Names the
  // projection-specific proof that converted the observed C5 site(s)
  // into an accepted kernel instead of a refusal. This is intentionally
  // separate from `refusalDetail` so proof logs can surface safe-C5
  // attribution without pretending it was an error.
  std::string SuppressionReason;

  // Number of `@llvm.amdgcn.workitem.id.x()` calls the classifier
  // visited. Informational; used by the lit fixtures to assert
  // "classifier ran at all" in the non-refusal negative case.
  unsigned VisitedCalls = 0;
};

// Classify every `@llvm.amdgcn.workitem.id.x()` call in F against
// the narrowing rule above.
//
// Returns `!refused && visitedCalls == 0` when
// `targetWaveSize <= sourceWaveSize` (same-wave / narrowing -- no
// replica-1 exists, nothing to refuse).
//
// `projection` parameter semantics. The caller must pass the projection
// actually selected for this raise, not a user-facing enable flag. The
// walk always runs; the projection only decides whether an observed C5
// site becomes a refusal or an attribution breadcrumb. Successful raises with
// suppressed C5 sites propagate `suppressionReason` to the loader proof log.
//
// `PredicateChainProjection::WaveNative` SUPPRESSES refusal in the normal
// no-phantom regime. Rationale:
//
//   * Under `WaveNativeProjection` the `init_whole_wave` +
//     per-source-lane modeled-EXEC model means the architectural
//     `tid` a target lane sees IS its own source-wave tid (for a
//     num_warps > 1 kernel: target wavefront 0's lanes 0..31 run
//     source wave 0 with tids 0..31, lanes 32..63 run source
//     wave 1 with tids 32..63). A predicate like `tid < K` then
//     evaluates the same way the source HW would have evaluated
//     it. The MODREP-specific "replica-1 shares source wave 0's
//     EXEC" trap -- the exact rationale the refusal exists for --
//     does not apply.
//
//   * This is a projection-model statement, not a universal
//     safety proof. If a launch config ever materialises with
//     "phantom" target lanes that are NOT a 1:1 mapping of source
//     lanes -- e.g. a target wavefront whose
//     `max_flat_workgroup_size` is smaller than the target
//     wavefront width, leaving lanes outside the source lane
//     index space -- then `tid` on those lanes is unmodelled by
//     either projection and the classifier's refusal is again
//     the principled answer.
//
//     The `maxFlatWorkgroupSize` parameter below (added after
//     `canary_bitmatrix_composite` empirically falsified the
//     pre-phantom-lane WaveNative suppression -- see the canary's
//     commit message) narrows the WaveNative suppression to
//     exactly the
//     "no phantom lanes" regime: when `maxFlatWorkgroupSize <
//     targetWaveSize` (and it's known, i.e. > 0), the HSACO
//     cannot be launched with enough threads to fill a target
//     wavefront, phantom lanes are GUARANTEED for every launch,
//     and the classifier refuses the same way it would under
//     MODREP. When `maxFlatWorkgroupSize >= targetWaveSize` or
//     the caller passes 0 ("unknown"), phantom lanes are
//     possible but not provable -- the classifier stays
//     permissive and logs the site as an attribution breadcrumb
//     (unchanged from the pre-phantom-lane behaviour). The normal raiser
//     path routes statically-known phantom-lane kernels to MODREP before
//     this classifier runs; this WaveNative refusal is a defensive guard
//     for direct callers and future projection-selection bugs.
//
//   * The walk still runs so the report's `observedSites` is
//     populated. raiser.cpp emits `LLVM_DEBUG` for every
//     observed site on the WaveNative path as an attribution
//     breadcrumb -- if a C5-shape kernel ever miscompiles under
//     WaveNative, the debug log names the icmp site the
//     classifier would have refused under MODREP.
// `PredicateChainProjection::ModuloReplication` refuses only when an
// active target replica lane can exist. The proof relies on the HSACO
// metadata contract: `.max_flat_workgroup_size` is a hard upper bound on
// the runtime workgroup size for the kernel descriptor Hotswap emits. If
// `0 < maxFlatWorkgroupSize <= sourceWaveSize`, the launch cannot
// activate lanes outside the source wave's lane-index domain; under
// MODREP those upper target lanes remain hardware-inactive for the whole
// kernel, so the replica-divergence proof obligation does not apply. If
// `maxFlatWorkgroupSize == 0` the evidence is unknown and the classifier
// fails conservative; if `maxFlatWorkgroupSize > sourceWaveSize`, at
// least one active target lane can evaluate the unmasked `tid` predicate
// outside `[0, W_s)`, so C5 remains a loud refusal.
//
// `maxFlatWorkgroupSize`: the kernel's `max_flat_workgroup_size`
// metadata field from the HSACO's `.amdgpu_metadata` section, or
// 0 if the caller does not have the information. Used by WaveNative's
// defensive phantom-lane refusal and by MODREP's active-replica-lane
// proof above.
//
// `suppressThreadLoopC5` is meaningful only with
// `PredicateChainProjection::ThreadLoop`. Callers must set it only for a
// separately-proven ThreadLoop route; raiser.cpp does so only for the
// SGPR-forced readlane/writelane -> explicit-readfirstlane retry. It is
// not a blanket "ThreadLoop solves C5" assertion.
PredicateChainClassifierReport classifyPredicateChain(
    llvm::Function &F, unsigned SourceWaveSize, unsigned TargetWaveSize,
    PredicateChainProjection Projection =
        PredicateChainProjection::ModuloReplication,
    unsigned MaxFlatWorkgroupSize = 0, bool SuppressThreadLoopC5 = false);

} // namespace COMGR::hotswap

#endif
