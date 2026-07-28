# MODREP predicate-chain class

**Status.** Option O1 (projection-aware loud-refuse classifier) landed; O2
(mask rewrite) deferred indefinitely on semantic grounds; O3 is now active
only as the analysis-selected ThreadLoop retry for the §5.6.3
explicit-readfirstlane SGPR-forced class. Under `WaveNativeProjection` (the
default for wave32 -> wave64 cross-widening) the class is structurally
suppressed except for the defensive phantom-lane sub-case. Under
`ModuloReplicationProjection`, the narrow-O1 classifier in
`c5_predicate_chain_classifier.{hpp,cpp}` refuses only when an
active target replica lane can exist; statically single-source-wave launches
(`0 < max_flat_workgroup_size ≤ W_s`) remain supported because upper target
lanes are hardware-inactive for the entire kernel body.

**Boundary with the §5.6.3 cross-lane safety-net.** This document's
C5 predicate-chain refusal is projection-dependent (suppressed under
WaveNative except the defensive phantom-lane sub-case, active under MODREP
only when launch metadata does not prove the no-active-replica regime).
That is distinct from the post-raise
`rewriteCrossLaneDivergent` use-chain safety-net in
`rewrite_cross_lane_divergent.{hpp,cpp}` (the
`writelane/readlane-post-raise-safety-net` refusal path): that guard
is projection-agnostic and can refuse under both WaveNative and
MODREP when a rewritten cross-lane value reaches an SGPR-forced sink
(for example `llvm.amdgcn.readfirstlane`).

**ThreadLoop activation boundary.** The internal retry is now graduated for
one class only: cross-widening post-raise refusals whose structured
use-chain verdict is `ExplicitReadFirstLane`, with an integer target/source
wave-size ratio. It is still analysis-driven and invisible to users. The
retry lowers `readlane`, `writelane`, and explicit `readfirstlane` as
source-wave-scoped operations and keeps the original loud refusal for shapes
outside that proof. This is orthogonal to MODREP C5:
`WorkitemIdPredicateChain` remains a refusal under MODREP whenever active
replica lanes can exist and remains suppressed under WaveNative's
non-phantom contract. The ThreadLoop retry suppresses the C5 refusal only on
this narrowed route; it is not a general "ThreadLoop solves C5" claim.

**Scope.** Wave-size axis: kernels compiled for a source wave
width `W_s` that reach cross-widening (`W_t > W_s`) under
modulo-replication, pass the `wave-size-translation.md §6`
obstruction classes C1–C4, raise cleanly through `raise_cli`,
and yet compute mathematically wrong values on the target
because their kernel-emitted predicates reference
`workitem.id.x()` directly without wave-size normalisation.
Orthogonal translation axes (ABI, sync, matrix, TDM) are out of
scope.

**Audience.** Hotswap raiser contributors and reviewers adding /
modifying cross-widening machinery; anyone wanting to know
whether a given source kernel shape is wave-size-safe under
MODREP.

---

## 1. Problem statement

The `wave-size-translation.md §6` obstruction classes (C1–C4)
enumerate the *structural* ways modulo-replication can be wrong:
absolute lane-ID leaks (C1), wave-baked cross-lane ops (C2),
inter-replica communication via shared state (C3), and
lane-position-dependent EXEC writes (C4). Every site G1 refuses
is one of these; everything else emits modulo-replication code
that the §5 SIMT Predicated Execution (SPE) projection assumes is
wave-size-oblivious by construction.

A recipe passes G1 iff every site in its disassembly is either
(a) wave-invariant, (b) rewritable per §5.3's landed table, or
(c) pending-rewrite-recognised. §7's soundness claim reads:

> Every kernel the tool emits code for is provably wave-size-
> oblivious; every kernel it refuses is one we cannot prove safe.
> There is no third category of "the tool ran and produced wrong
> code."

That claim holds for the C1–C4 axes. It does *not* hold for a
fourth concern G1 does not audit: **the kernel's downstream use
of `workitem.id.x()` inside predicate chains that gate side
effects**. When a source-wave kernel packs its per-lane work for
`W_s` threads and uses `tid` directly as a control variable, the
raiser lifts `workitem.id.x()` through the SPE prelude unchanged
-- so on the target wave the predicate evaluates on the
target-wave `tid` range rather than the source-wave `[0, W_s)`.
For kernels that use this predicate as "which lanes participate
in stage `s` of a scan" or "which lanes write which slot of an
output tile", the result is a silent miscompile that is neither
classified by G1 nor rewritten by §5.3.

This document defines the class, argues it is orthogonal to
C1–C4, enumerates the fix options, and documents the landed
design.

---

## 2. Class signature

A kernel exhibits the predicate-chain class iff all three
ingredients hold at some site:

1. **Source**: `@llvm.amdgcn.workitem.id.x()` (or a one-hop
   derivation -- `add`, `sub`, `lshr`, `and` with a constant
   ≠ `W_s − 1`, phi, select, cast) is read in the raised IR.
2. **Gate**: the derived value feeds either
   - an `icmp` whose result gates a side effect (store,
     SPE-wrapped compute, scan-stage update), or
   - an address expression that feeds a store.
3. **Wave-size sensitivity**: the icmp's other operand is a
   compile-time constant `K` with `0 < K ≤ W_s − 1` (the
   lane-position-gate shape), or the address expression
   admits target-wave lanes into a branch / slot the source
   wave never would have taken.

### Canonical example -- Kogge-Stone scan

Triton's `tl.cumsum` lowers to a Kogge-Stone tree whose
stage-`s` guard is `if tid >= 2^s: x += bpermute(lane_id − 2^s)`.
For wave32 source with `num_warps = 1`, `tid ∈ [0, 32)` and the
guard partitions lanes correctly per stage (stages `s = 0..4`,
`2^s ∈ {1, 2, 4, 8, 16}` -- all within `(0, W_s − 1]`).

The raised IR for this shape is:

```
%tid = call i32 @llvm.amdgcn.workitem.id.x()
...
%vcmp215 = icmp ult i32 1, %tid       ; stage-1: tid > 1
%vcmp359 = icmp ult i32 3, %tid       ; stage-2: tid > 3
%vcmp482 = icmp ult i32 7, %tid       ; stage-3: tid > 7
%vcmp592 = icmp ult i32 15, %tid      ; stage-4: tid > 15
```

Each of those four icmps matches ingredients (1) + (2) + (3):
the compile-time constants (1, 3, 7, 15) are all in
`(0, W_s − 1 = 31]`. Under modulo-replication to wave64, target
lanes in the upper half of each wavefront have `tid` values the
source compile did not plan for; the predicates evaluate
differently than they would have in the source wave.

### Passing baseline -- Triton bounds check

Kernels like `vecadd_f16` emit the same structural IR shape but
the icmp operand is a *dynamic* kernarg rather than a
compile-time constant:

```
%tid = call i32 @llvm.amdgcn.workitem.id.x()
%vcmp = icmp ult i32 %tid, %arg_num_elements    ; dynamic bound
br i1 %vcmp, label %store_arm, label %skip_arm
```

This matches ingredients (1) + (2) but **not** (3) under the
narrow-O1 shape -- the classifier is defined to treat
dynamic-operand icmps as bounds checks, not lane-position gates.
The decision to narrow ingredient (3) to compile-time K is what
keeps baselines like `vecadd_f16`, `rope_fp32`, `corpus_add_fp32`,
`corpus_asin_fp32` green. See §5 O1's rationale for why the
broader "unmasked tid reaches any icmp" rule was rejected.

---

## 3. Why existing obstruction classes don't catch this

Each of C1–C4 checked against the canonical example:

- **C1. Absolute lane-ID leak.** Predicated on
  `v_mbcnt_hi_u32_b32`, `v_readlane_b32` / `v_writelane_b32` with
  operand ≥ `W_s`, `llvm.amdgcn.ballot`,
  `llvm.amdgcn.wavefrontsize`. None of these. `%tid` via
  `llvm.amdgcn.workitem.id.x` is structurally NOT an absolute
  lane ID (its range is `[0, WG)`, not `[0, wave)`, and the two
  coincide only for single-wave workgroups). G1 correctly clears
  the scan-shaped kernel on C1.
- **C2. Wave-baked cross-lane op.** `ds_bpermute_b32`,
  `v_permlanex16`, `ds_swizzle_b32`, DPP with wave-wide
  semantics. All landed rewrites per §5.3. A Kogge-Stone scan
  emits many `ds_bpermute_b32`; the P1 lift produces
  correctness-equivalent IR per the handler's MODREP comment.
  The class's miscompile is downstream of the cross-lane op,
  not at the op itself.
- **C3. Non-commutative atomic.** Zero in all failing kernels --
  consistent with the wider corpus's zero hit rate.
- **C4. Lane-position-dependent EXEC write.** `v_cmpx_*`,
  `s_*_saveexec_b32` against a lane-id-derived mask. The SPE
  diamond's own `s_and_saveexec` is classified safe by the
  allow-list gate (G2).

So the classifier is unanimously green -- modulo-replication is
emitted -- and the kernel miscompiles anyway. The class gap
this document closes is:

> Predicates on `workitem.id.x()` (or derived SSA values) that
> gate side effects -- stores, addr arithmetic, scan / reduction
> predicates -- are NOT audited by any existing C-class and are
> NOT rewritten by §5.3, even though their semantics change
> under MODREP when `W_t > W_s`.

---

## 4. Root cause

### 4.1 The MODREP contract

`wave-size-translation.md §5.2` states the projection formally:

> The target wave runs as `R = W_t / W_s` replicas of the source
> wave. Correctness requires every replica observes the same
> state as the source -- at every observable side effect, replica
> `r` with lane `l_target = r · W_s + l_source` must compute the
> same value the source's lane `l_source` would have.

The §5.4 SPE allow-list gate + §5.3 cross-lane rewrite table
jointly ensure this for every SemOp. The SPE prelude masks
`mbcnt`-derived `lane_id` by `W_s − 1` so the active-lane gate
evaluates on the source-wave-scoped `l_source`. Each landed
rewrite (P1–P6) preserves replica-independence at its
instruction-level boundary.

### 4.2 Why `workitem.id.x()` escapes the contract

`workitem.id.x()` is NOT a SemOp the rewrite table covers -- it
is an `amdgcn` intrinsic the raiser passes through verbatim to
the LLVM backend, which re-lowers it to a VGPR read of the
architectural workitem-id register. That register returns
`l_target ∈ [0, W_t · num_warps · W_s)` on the target wave,
NOT `l_source`. Any downstream use of this value in a
kernel-emitted predicate, address computation, or control-flow
decision silently reads a target-wave-scoped value.

### 4.3 Sub-cases by launch shape

The class manifests differently depending on the dispatch
`blockDim.x` the runtime hands HIP, which is determined by the
source compile's `max_flat_workgroup_size` (the hotswap path
honours the source WG size via the raised HSACO's kernel
descriptor; it does not upscale to target-wave alignment):

- **Sub-case 1: `blockDim.x = num_warps × W_s` and `num_warps > 1`
  forces `blockDim.x > W_t`.** The target dispatches multiple
  wave64 wavefronts, each with all 64 lanes active and `tid`
  values in `[0, W_t)` within each wavefront. Lanes in the upper
  half of each target wavefront (indices `W_s..W_t−1`) are
  MODREP replica-1; their `tid` value exceeds `W_s` and any
  kernel-emitted predicate `tid < K` with `K ≤ W_s − 1`
  evaluates *differently* from the source's corresponding
  replica-0 lanes.
- **Sub-case 2: `blockDim.x = W_s` (single-warp kernel).** The
  target dispatches one wave64 wavefront with only the low
  `W_s` lanes in EXEC at entry. Active lanes' `tid ∈ [0, W_s)`
  and kernel predicates evaluate correctly on the active subset;
  no active target replica lane can disagree with its source-lane
  counterpart. Hotswap therefore routes this statically-known
  phantom-lane launch to `ModuloReplicationProjection`, not
  `WaveNativeProjection`: upper target lanes stay hardware-inactive
  for the whole kernel, so their undefined VGPR state cannot
  participate in VALU, memory, or cross-lane side effects. This is
  the GPT-OSS/SGLang rotary-kernel shape (`max_flat_workgroup_size =
  W_s = 32` on a wave64 target).

### 4.4 Class summary

Every site exhibiting the class shares the three ingredients of
§2 plus active target lanes outside the source-wave lane domain
(§4.3 sub-case 1, or unknown launch metadata that cannot rule it
out). The class is orthogonal to C1–C4: C1 is about `mbcnt_hi` /
`ballot` leaks (already caught); this class is about
`workitem.id.x()`.

The per-instruction cross-lane rewrites (P1–P6) are *necessary*
for wave-size obliviousness but not *sufficient* -- a kernel that
passes the per-instruction audit can still miscompile if its
kernel-level predicate chain is wave-size-sensitive.

---

## 5. Fix options

Four options, each with a different coverage / invasiveness
profile.

### O1. G1 classifier extension + loud refusal

Add a post-raise IR-level classifier that identifies the class
signature of §2 and refuses the kernel with a new
`CrossWavePredicateChain` diagnostic. No rewrite attempted; the
classifier's correctness depends only on a forward use-chain
walk from each `@llvm.amdgcn.workitem.id.x()` call.

**Shape (narrow-O1).** Refuse iff the icmp has at least one
operand in the tid-unmasked-reachable set AND at least one
operand that is a compile-time constant K with
`0 < K ≤ W_s − 1`. Chains that pass through `and V, K`
with `K ≤ W_s − 1` mask the tid-divergence away and do not
contribute to the unmasked set.

**Projection / launch gate.** The shape becomes a refusal only when the
selected projection can activate a target lane outside the source-wave lane
domain. This relies on the HSACO contract that `max_flat_workgroup_size` is a
hard launch upper bound for the kernel descriptor Hotswap emits. MODREP
refuses when `max_flat_workgroup_size == 0` (unknown) or
`max_flat_workgroup_size > W_s`; it records a structured safe-C5 proof note
but does not refuse when `0 < max_flat_workgroup_size ≤ W_s`. WaveNative
suppresses the shape in the normal no-phantom regime and retains a defensive
phantom-lane refusal for direct callers. ThreadLoop suppresses only for the
separately-proven §5.6.3 retry route.

**Coverage.** Catches the Kogge-Stone scan shape (stage guards
`tid > 2^s − 1` with `s < log2(W_s)`). Does not catch kernels
that use dynamic-operand bounds checks (`vecadd_f16` and
friends), which is the design intent.

**Invasiveness.** ~280 LoC classifier + ~150 LoC lit fixtures
+ ~330 LoC unit tests + ~20 LoC wiring into `raiser.cpp`.

**Risk.** Sound-but-incomplete by construction: false positives
(refusing a safe kernel whose predicate happens to match the
narrow-O1 shape and has possible active replica lanes) are benign;
false negatives (missing a wave-size-sensitive predicate whose operand is
dynamic) leave a residual silent-miscompile class for dynamic-operand
shapes, which §6 argues is covered empirically by the WaveNative-default
suppression + end-to-end corpus regression testing.

### O2. Predicate-chain rewrite

A post-raise pass that rewrites every `@llvm.amdgcn.workitem.id.x()`
value in a predicate chain (icmp operand or GEP index feeding a
store) to `workitem.id.x() AND (W_s − 1)` -- i.e., the
source-wave-scoped lane id. The rewrite would run before the
narrow-O1 classifier and would convert C5-refused kernels to
correctness-preserving lifts.

**Coverage.** Would fix sub-case 1 (multi-warp kernels where
replica-1 lanes' `tid` exceeds `W_s`, and the mask collapses
them back onto `[0, W_s)`). Does NOT fix sub-case 2: on
sub-case 2, active lanes already have `tid ∈ [0, W_s)` so the
mask is a no-op on the active set; the cross-lane-primitive
inactive-lane-leak is outside the rewrite's scope.

**Invasiveness.** ~500 LoC rewrite pass + use-chain classifier
+ ~300 LoC lit tests. Structurally analogous to §5.6.3's
`rewrite-cross-lane-divergent.cpp`.

**Risk.** False-positive rewrites that change the semantic of a
kernel that intentionally uses `[0, W_t)` scope. The mask also
imposes a hidden invariant: if `num_warps × W_s` crosses a
power-of-2 boundary the mask's arithmetic is wrong. More
fundamentally, the mask is *semantically incorrect* for
`num_warps > 1` multi-warp kernels where the source compile
expects source wave 1's lanes to see `tid ≥ W_s` (e.g. for
per-wave partitioning) -- masking those lanes' `tid` by `W_s − 1`
forces their predicate to evaluate on the wrong range. This is
the primary reason O2 is deferred; see §6.

### O3. ThreadLoopProjection for scan-shaped patterns

The `ThreadLoopProjection` escape hatch: lower the kernel as a
target-wave-scoped serial loop over the `R = W_t / W_s` source
replicas. For each replica `r ∈ [0, R)`, run the kernel body
once with `workitem.id.x()` remapped to `(r · W_s) + lane_mod`.
Cost: `1/R` throughput (2× slowdown for wave32 -> wave64).

**Coverage.** Strictly more than O2 -- handles anything that can
be expressed as "run the source kernel R times, then reconcile",
including sub-case 2's inactive-lane-leak (each replica runs
serially with full EXEC, so inactive-lane VGPRs never participate
in a gather).

**Invasiveness.** The full serial-body loop remains the largest of the four.
The landed implementation is narrower: `ThreadLoopProjection` supplies the
projection boundary needed by the §5.6.3 SGPR-forced route and makes
lane-indexed primitives source-wave-scoped there. It does not claim to cover
arbitrary scan-shaped C5 kernels by itself; those still need either the full
loop body transform or another explicit rewrite.

**Risk.** Bounded to the explicit-readfirstlane post-raise class by the
activation gate. The canaries assert the MODREP C5 refusal remains intact and
that non-trigger writelane/readlane rewrite cases stay on their existing path.

### O4. Harness-side constraint on `num_warps`

Add a check in the AOT compile flow that refuses to compile a recipe for
gfx1250 unless `num_warps × W_s ≥ W_t`. Ensures the target WG
is at least one full target wave and MODREP's replica factor
is 1.

**Coverage.** Narrow. Would force
`canary_bpermute_scan_fp32` to re-compile with `num_warps = 2`
to target wave64 -- but does NOT fix kernels in
`scope_discovery/kernels/` captured from upstream corpora
where `num_warps` is set by the original author.

**Invasiveness.** Trivial -- a single assertion. ~10 LoC.

**Risk.** Doesn't address the class; treating it as "the fix"
would leave a trap for recipes that happen to satisfy the
assertion today but break on a `num_warps` bump.

---

## 6. Landed design

### 6.1 Narrow-O1 classifier

Implementation:
`c5_predicate_chain_classifier.{hpp,cpp}`. Wired into
`raiser.cpp` Phase 6.6 (post-mem2reg). Emits
`RaiseFailure::crossWavePredicateChain` on refusal, with
`ObstructionKind::WorkitemIdPredicateChain` +
`RaiseFailureReason::CrossWavePredicateChain`.

Two-pass walker:

- **Pass 1**: forward-walk from each `@llvm.amdgcn.workitem.id.x()`
  call, tagging every tid-reachable value as `unmaskedVisited`
  (walk did not cross an `and V, K ≤ W_s − 1` mask) or
  `maskedVisited` (walk did). A value can appear in both sets
  when it joins a masked arm and an unmasked arm via a phi.
  Propagators: casts, phi, select, gep, freeze, insert/extract
  element/value, shufflevector, integer binops (add/sub/mul/shl/
  lshr/ashr/or/xor/and -- including a non-masking `and`), unary
  ops, and the numeric intrinsics
  `llvm.{smin,smax,umin,umax,abs,ctlz,cttz,ctpop,bitreverse,
  bswap,fshl,fshr,sadd_sat,ssub_sat,uadd_sat,usub_sat}`.
- **Pass 2**: scan every `icmp` in the function. Refuse iff the
  icmp has at least one operand in `unmaskedVisited` AND at
  least one operand that is a compile-time constant K with
  `0 < K ≤ W_s − 1`.

**Direction gate.** The classifier no-ops when
`W_t ≤ W_s` (same-wave / narrowing have no replica-1; the class
cannot manifest).

**Projection / launch gate.** The caller passes the selected projection, not
the user-facing `enableWaveNative` flag. The walk still runs and populates
`observedSites`; the projection/launch gate decides whether a site becomes a
refusal or an attribution breadcrumb.

- `WaveNativeProjection`: refusal is suppressed in the normal no-phantom
  regime. `init_whole_wave` + per-source-lane modeled EXEC means each target
  lane's `tid` IS its own source-wave tid (for `num_warps > 1`, target
  wavefront 0's lanes 0..31 run source wave 0 with tids 0..31, lanes 32..63
  run source wave 1 with tids 32..63); the MODREP "replica-1 shares source
  wave 0's EXEC" assumption does not apply. A defensive refusal remains if a
  direct caller selects WaveNative with `0 < max_flat_workgroup_size < W_t`.
- `ModuloReplicationProjection`: refusal fires only when active replica lanes
  can exist (`max_flat_workgroup_size == 0` unknown, or
  `max_flat_workgroup_size > W_s`). If `0 < max_flat_workgroup_size ≤ W_s`,
  the HSACO launch-bound metadata proves upper target lanes are
  hardware-inactive for the whole kernel and cannot observe the divergent
  predicate. Successful Hotswap proof JSON carries
  `c5_suppressed_count` / `c5_suppression_reason` when this accepted-C5 path
  is exercised.
- `ThreadLoopProjection`: refusal is suppressed only for the existing
  analysis-triggered SGPR-forced explicit-readfirstlane retry route.

**Regression guards.** See §7.

### 6.2 O2 deferred indefinitely

The proposed `tid AND (W_s − 1)` mask rewrite is semantically
incorrect for the multi-warp (`num_warps > 1`) shape and a
structural no-op for single-warp sub-case 2 (active lanes
already have `tid ∈ [0, W_s)`). No recipe in the current corpus
would be helped by it. Reopening criteria in §6.5.

### 6.3 `WaveNativeProjection` as cross-widening default

`WaveNativeProjection` is the default for wave32 -> wave64
cross-widening (opt out via `--disable-wave-native` on
`raise_cli` or `enableWaveNative=false` on programmatic
callers). Under WaveNative:

- `@llvm.amdgcn.init_whole_wave` at kernel entry forces
  hardware `EXEC = -1` kernel-wide; the original per-lane
  active mask is saved into the (widened) EXEC alloca.
- Each target lane has its own modeled-EXEC bit at source
  width. For `num_warps > 1`, target wavefront 0's lanes
  0..31 model source wave 0 with their own EXEC, lanes 32..63
  model source wave 1 with its own EXEC. No replica sharing.
- Cross-lane primitives (`ds_bpermute`, WMMA collectives) run
  with hardware EXEC = -1 so every lane participates,
  eliminating sub-case 2's inactive-lane-leak class.

The predicate-chain class the narrow-O1 classifier detects is
MODREP-specific by construction -- under WaveNative the replicas
don't exist and each target lane's `tid` is its own source-wave
`tid`. The classifier short-circuits the refusal under
WaveNative by design (§6.1's projection gate), except for the
defensive phantom-lane direct-caller guard.

Rationale for the default choice:

- **WMMA / matrix-kernel correctness.** `wave-size-translation.md
  §5.6.1` documents the Wave64-collective correctness invariant:
  the WMMA -> MFMA lowering and every convergent cross-lane
  primitive issues a single Wave64 intrinsic that reads all 64
  lanes regardless of EXEC; under partial-wave hardware EXEC
  those lanes' VGPR writes don't fire and downstream readers
  see undefined data. `init_whole_wave` + the kernel-wide
  HW EXEC = -1 + per-lane modeled EXEC is the documented
  solution.
- **`num_warps > 1` correctness.** MODREP's "replicas of source
  wave 0" model cannot express genuinely distinct source-wave
  EXEC registers. WaveNative's per-lane modeled EXEC does.

MODREP code is fully retained:

1. Same-wave and narrowing directions bypass WaveNative via
   its constructor's direction gate; MODREP is the only
   projection emitted there.
2. Explicit opt-out via `--disable-wave-native` /
   `enableWaveNative=false` is preserved for lit fixtures that
   pin MODREP-shape IR invariants
   (`c5_predicate_chain_tid`, `v_cmp_cndmask_sgpr_scalar_clobber`)
   and for operators debugging projection-specific behaviour.
3. The narrow-O1 classifier's refusal path runs under MODREP when active
   replica lanes can exist, short-circuits under single-source-wave MODREP,
   and short-circuits under WaveNative per §6.1.

### 6.4 Scope note: orthogonal fixes for canary / corpus_layernorm

`canary_bpermute_scan_fp32` and `corpus_layernorm_fp32` -- the
two recipes originally expected to exhibit the predicate-chain
class as their end-to-end miscompile -- have MATCH verdicts
today under the WaveNative default. Their mechanism, however,
turned out to be entirely orthogonal to this document's class:

A silent fallback in `handle-vopd.cpp::v_cndmask_b32` hardcoded
`loadVCC()` regardless of the instruction's explicit scalar
operand. gfx1250 VOPD's `v_dual_cndmask_b32` encoding can bind
the scalar operand to either `vcc_lo` OR an arbitrary `sN`; the
pre-fix handler routed both halves through VCC unconditionally,
collapsing distinct scalar conditions onto whatever last wrote
VCC. Triton's Kogge-Stone scan at distances 8 / 16 emits
paired `v_dual_cndmask_b32 ..., s0 :: v_dual_cndmask_b32 ...,
vcc_lo` where s0 is the stage's selector-advance guard and
vcc_lo is the value-update guard; the pre-fix handler mis-wired
the selector-advance to use the value-update guard's predicate.

The fix (`handle-vopd.cpp::v_cndmask_b32` SGPR-aware routing,
mirroring the `V_CNDMASK_B32_e64` handler in
`handle-valu-vop3p.cpp`) + a sibling fix extending the same
routing pattern to the six `V_{ADD,SUB,SUBREV}_CO_(CI_)U32`
carry-chain handlers in `handle-valu.cpp` closes the VOPD /
VOP3B SGPR-operand class independently of the predicate-chain
class this document scopes. See the lit fixtures in §7.2 for
the IR-level regression fences and the Triton-corpus end-to-end
verdicts.

The narrow-O1 classifier (§6.1) still serves as the MODREP-path
regression guard for the predicate-chain class -- on a future
kernel that genuinely hits the class under MODREP with possible active
replica lanes, the classifier loud-refuses rather than silently
miscompiling.

### 6.5 Deferred options -- reopening criteria

O2 / O3 / O4 reopen if:

- **A.** A new corpus recipe appears whose `workitem.id.x()`
  flows into an `icmp` against a compile-time constant in
  `(0, W_s − 1]` AND whose downstream miscompile mechanism is
  demonstrably the predicate-chain class (not a VOPD-cndmask /
  carry-chain SGPR-operand issue like §6.4, and not an
  inactive-lane-leak from sub-case 2). Triggers an O2 redesign
  with a real mechanism trace of the new recipe.
- **B.** A corpus recipe is shown to need the
  `ThreadLoopProjection` escape hatch (O3) -- typically a kernel
  that legitimately requires `[0, W_t)` scope for some
  predicate and cannot be expressed with an AND-mask.
- **C.** A scope-discovery / GPT-OSS kernel surfaces a fix-up
  need beyond either O1's refusal or an eventual O2 mask -- e.g.
  a mixed bitmatrix / scan / writelane pattern. Triggers an
  O3 scope paper or an entirely new class doc.

---

## 7. Test surface

### 7.1 Predicate-chain class (O1) regression guards

- **`lit_tests/c5_predicate_chain_tid/`** -- paired RUN lines
  pin both projection paths:
  - `--disable-wave-native` MODREP path: expects loud refusal
    with `cross-wave-predicate-chain` diagnostic, the
    `compile-time constant 16` detail string, and
    `Class 5 / WorkitemIdPredicateChain` per-site trace.
  - WaveNative default path: expects clean raise + IR
    emission (the classifier's projection gate suppresses
    refusal).
- **`lit_tests/c5_predicate_chain_phantom_lane/`** -- same C5 shape with
  `max_flat_workgroup_size: 32` on a wave64 target. Expects the raiser to
  select MODREP for phantom-lane safety and then accept the C5 site because
  no active target replica lanes exist.
- **`lit_tests/c5_predicate_chain_masked/`** -- non-refusal
  sibling. Same K=16 icmp preceded by `and tid, 31` (= W_s−1);
  expects raise_cli success and the `and i32 ..., 31` anchor
  in the emitted IR.
- **`tests/c5_predicate_chain_test.cpp`** -- `C5PredicateChain.*`
  gtest suite. Audits the classifier on synthesised
  IR in isolation from the MC-level pipeline:
  `TidDirectSmallConstRefuses`, `TidMaskedBeforeCmpAccepts`,
  `TidSmallConstZeroAccepts`, `TidLargeConstAccepts`,
  `TidDynamicCmpAccepts`, `SameWaveDirectionGate`,
  `NarrowingDirectionGate`, `WaveNativeProjectionGate`,
  `ThreadLoopProjectionGate`, `WaveNativePhantomLaneRegimeRefuses`,
  `WaveNativeUnknownWorkgroupSizeKeepsSuppression`,
  `ModrepSingleSourceWaveAccepts`,
  `ModrepUnknownOrActiveReplicaRefuses`, `NoCallsIsNoOp`,
  `PhiPropagatesTidDerivation`,
  `MaskedPhiThroughUnmaskedArmRefuses`,
  `CrossSubtreeMaskedVsUnmaskedAccepts` (pins that the
  `icmp unmasked-tid, masked-tid` cross-subtree shape is NOT
  refused -- that widening was explored and falsified),
  `IntrinsicPropagatorRefuses` (pins the numeric-intrinsic
  propagator audit via `@llvm.umin`).
- **`lit_tests/cross_wave_warn/`** and **`lit_tests/v_cmpx_ballot/`**
  -- updated to use `K ≥ W_s` constants (64 / 96) so their
  cross-wave EXEC-writer / ballot-routing coverage is not
  conflated with the C5 predicate-chain coverage. Pre-update
  those fixtures exercised K ≤ W_s − 1 predicates which the
  narrow-O1 classifier correctly refuses as latent-bug kernels.

### 7.2 Orthogonal VOPD / VOP3B SGPR-operand fixes

Separate class from §6.1, documented in §6.4 and cross-referenced
here because they close `canary_bpermute_scan_fp32` /
`corpus_layernorm_fp32` end-to-end:

- **`lit_tests/v_dual_cndmask_b32_sgpr_cond/`** -- regression
  fence for the VOPD `v_dual_cndmask_b32` SGPR-condition fix.
  Inline asm emits a paired `v_dual_cndmask_b32 ..., s0 ::
  v_dual_cndmask_b32 ..., vcc_lo` with DIFFERENT scalar
  conditions on the two halves; FileCheck expects the two
  `vopd_cndmask` selects to consume different `i1` SSA values
  (the SGPR-path cmp and the VCC-path cmp). A regression that
  re-introduces hardcoded-VCC collapses the two conditions onto
  the same i1 and fails the CHECK pattern.
- **`lit_tests/v_add_co_u32_sgpr_carry/`** -- regression fence
  for the carry-chain (`V_{ADD,SUB,SUBREV}_CO_(CI_)U32`) SGPR-
  operand fix. Inline asm emits `v_add_co_u32 ..., s0, ... ::
  v_add_co_ci_u32_e64 ..., s0, ..., s0` -- the canonical 64-bit
  address-chain shape under VCC pressure. FileCheck expects the
  carry-out ballots through `amdgcn.ballot.i64` into the SGPR
  path, the second add's carry-in forwards the SAME `i1` SSA
  value the first add produced (fresh V_CMP shadow lookup -- no
  `load i1, ptr %vcc` in between). Negative CHECK-NOT pin on
  `load i1, ptr %vcc` catches hardcoded-VCC regressions.

### 7.3 End-to-end regression surface

External end-to-end validation on the Triton corpus (run
against `gfx1250 -> gfx942` cross-widening, WaveNative default):

- `canary_bpermute_scan_fp32` -- MATCH 4/4 under WaveNative
  default; loud-refused 4/4 under `--disable-wave-native`
  (MODREP + narrow-O1).
- `swiglu_fp32`, `rmsnorm_fp32`, `corpus_layernorm_fp32` --
  MATCH 4/4.
- 7 baselines (`vecadd_f16`, `rope_fp32`, `corpus_add_fp32`,
  `corpus_asin_fp32`, `canary_dpp_compound_add_fp32`,
  `canary_dpp_reduce_fp32`, `canary_permlanex16_rowmax_fp32`)
  -- MATCH 4/4.
- `corpus_softmax_fp32` -- EXIT=2 via the writelane safety net
  (`wave-size-translation.md §5.6.3`); orthogonal class.

`ctest` + `llvm-lit`: no regressions introduced by the landed
design. Batch-raise corpus coverage is unchanged. The Matmul128 family
remains covered by the V_CMP -> V_CNDMASK per-lane-i1 shadow tests
documented in `learnings.md` (2026-04-21).

### 7.4 Repro

Raise a single kernel to IR under each projection:

```bash
# WaveNative default
<hotswap-build>/raise_cli <kernel>.gfx1250.co \
  --isa=gfx1250 --target-isa=gfx942 \
  --emit-ir=<kernel_name>

# MODREP (loud-refuses predicate-chain shapes)
<hotswap-build>/raise_cli <kernel>.gfx1250.co \
  --isa=gfx1250 --target-isa=gfx942 \
  --disable-wave-native \
  --emit-ir=<kernel_name>
```

---

## 8. References

- [`wave-size-translation.md`](wave-size-translation.md) -- the
  canonical wave-size axis spec. Specifically §5.2
  (modulo-replication definition), §5.3 (cross-lane rewrite
  table), §5.6.1 (hardware vs modeled EXEC under
  cross-widening), §5.6.3 (writelane/readlane post-raise
  rewrite), §6 (obstruction classes C1–C4), §7 (decision
  procedure), §8 (principled fail-loudly gates).
- [`target-capability-dispatch.md`](target-capability-dispatch.md)
  §5 -- `SemOpAttrs` extension pattern (relevant if a future
  revision hangs a `predicateChainLaneScoped` bit on tid-
  emitting SemOps).
- `c5_predicate_chain_classifier.{hpp,cpp}` -- the
  narrow-O1 classifier. Header carries the full design contract
  and the `waveNative` parameter docstring.
- `handle-vopd.cpp` -- the VOPD `v_cndmask_b32`
  SGPR-condition handler (§6.4).
- `handle-valu.cpp` -- the six carry-chain handlers
  with `readCarryInI1` / `writeCarryOutI1` helpers (§6.4).
- `handle-valu-vop3p.cpp` -- the non-VOPD
  `V_CNDMASK_B32` handler; the reference implementation whose
  SGPR-aware routing the VOPD + carry-chain fixes mirror.
- `rewrite-cross-lane-divergent.cpp` -- the
  writelane/readlane post-raise rewrite. Structural template
  for a future O2 landing.

---

## 9. Open questions

1. Is there evidence of a GPT-OSS / AITER kernel in the broader
   corpus (beyond the current Triton set) that
   genuinely hits the predicate-chain class -- i.e., matches the
   §2 signature AND miscompiles in a way not explained by the
   VOPD-cndmask / carry-chain class (§6.4) or the
   inactive-lane-leak sub-case 2 mode? If so, it's the
   reopening trigger A in §6.5 and motivates actually
   implementing O2.
2. Should `target-capability-dispatch.md`'s `SemOpAttrs`
   extension grow a `predicateChainLaneScoped` bit on
   `workitem.id.x`-emitting SemOps, so the classifier's C5
   pass can enumerate emission sites without hand-maintaining
   a list? This is the "hang new attrs off `SemOpAttrs`"
   pattern the README index section recommends.
3. Is the narrow-O1 classifier's constant-K range
   `(0, W_s − 1]` the right bound, or should it extend to
   `[1, W_t − 1]` to catch predicates that partition the
   target wave (e.g., `tid < W_t/2` half-wave broadcasts)?
   Current evidence is insufficient -- no corpus kernel emits
   this shape -- and the over-approximation discipline argues
   for the tighter bound. If a corpus kernel surfaces with
   `K ∈ (W_s − 1, W_t − 1]`, widen the bound to match.

## 10. Scaled-dispatch projection (`ScaledModuloReplicationProjection`)

This section documents the correct-by-construction resolution of the
predicate-chain / divergent-early-exit class. It is the resolution of the
WaveNative `workitem.id.y()`/`.z()`-derived refusal
(sec. 6.3 / `c5-predicate-chain-classifier`).

### 10.1 Why the other three projections cannot run the reduce

| Projection | Result | Blocker |
|---|---|---|
| WaveNative | faults on odd N | two packed source waves share one HW EXEC; a divergent early exit drags one packed wave through masked with a stale base VGPR into a faulting load. Systemic; no small patch. |
| MODREP (as-is) | drops work | valid only when `max_flat_workgroup_size <= W_s` (phantom regime). For a full block the HW packs `W_t` real threads/wave, so MODREP's replica lanes collide with real block threads. |
| ThreadLoop | refuses | `threadLoopUnsupportedWorkgroupMemoryOrBarrier` refuses any kernel with `s_barrier`/LDS -- the reduce has thousands. |

### 10.2 The design

Make MODREP's "upper lanes are replicas, not real threads" assumption TRUE by
construction: the runtime launches the block with a `W_t / W_s`-scaled extent
along the fastest (wave-carrying) dimension **x**, so each target wave hosts
exactly one source wave in lanes `0..W_s-1` and exact replicas in `W_s..W_t-1`.

Concretely for wave32 -> wave64 (factor 2):

- The launch shim doubles `blockDim.x` (grid in blocks unchanged).
- The raised kernel maps hardware `workitem.id.x` back to the logical source id:

  ```
  logical_x = ((x_hw & ~(W_t-1)) >> log2(W_t/W_s)) | (x_hw & (W_s-1))
            = ((x_hw & ~63) >> 1) | (x_hw & 31)          (wave32->wave64)
  ```

  so hardware lane `W_s + i` sees the same logical thread as lane `i`.
- The raiser halves the in-kernel workgroup/grid-size query along x
  (`source-hidden-args.cpp`), so loops and reduction bounds still observe the
  source block size. All derived hidden args (`block_count = grid/group`,
  `remainder = grid%group`) stay correct because each halved read reproduces the
  exact source size, so anything computed from them is computed from source
  values.

**Always x, not the predicate's dimension.** The kernel that surfaced this is
PyTorch `reduce_kernel<512,1>` launched `(32,16,1)`: `blockDim.x = 32 = W_s`, and
the divergent early exit is on `id.y`. Doubling **x** (32 -> 64) makes each target
wave64 cover exactly one `id.y` row, so `id.y` -- and any `.y`/`.z`-derived
predicate -- becomes *wave-uniform* and the divergence disappears; simultaneously
the x-doubling is what interleaves lanes `0..31` and `32..63` as replicas. This
resolves the "which dim to double" question: it is always the fastest
wave-carrying dimension (x), whose `blockDim.x` is a positive multiple of `W_s`
for any warp-primitive kernel.

### 10.3 Why it is correct

- **Divergence gone.** Lane `i` and replica `W_s + i` compute identically
  (same logical id, same wave-uniform y/z), so they share every predicate. A
  divergent `s_cbranch_execz` early exit takes both out together -- no phantom
  wave, no stale VGPR, no aperture fault.
- **Cross-lane ops correct.** A reduce is all `ds_bpermute` / `permlane` /
  `readlane` / DPP. MODREP already scopes these per source wave
  (`ballotI1ToWidth` source-wave-base shift, `emitLaneActiveBit`'s `L mod W_s`,
  mask replication in `extractLaneBitFromWaveMask`). Because the replicas make
  the target ballot's upper half mirror its lower half, every one of these ops
  reads valid duplicate data instead of undef, and the source-width (i32) EXEC
  alloca is exactly the right model: lane `i` and `W_s + i` genuinely should
  share one EXEC bit.
- **mbcnt-derived `v_cmpx` / `saveexec`.** The mbcnt lift already folds the lane
  id to `lane_id & (W_s-1)` (source-wave-local), so a lane and its replica
  compare equal and the `v_cmpx` ballot into the i32 EXEC alloca is consistent.
  The obstruction classifier accepts `CmpxFromLaneId` (`RewriteId::CmpxLaneRelative`)
  and `SaveExecFromLaneId` (`RewriteId::SaveExecLaneRelative`, via
  `numSourceWavesPerTarget() == 1`) under this projection -- the normal handlers
  already emit correct source-wave-local IR; no target-width projection is needed.
- **Store-only atomics.** MODREP's `AtomicOneReplica` gate (`lane_id < W_s`) is
  exactly right: only the real lower half issues each atomic, matching native
  wave32.

Cost: ~50% lane utilisation (the upper half redoes the lower half's work), i.e.
2x slower than WaveNative packing. It is the safe correctness fallback, not the
fast path; WaveNative stays the opt-in fast path for kernels it can represent.

### 10.4 Refusals

- **wmma / mfma:** a matrix fragment spans all `W_t` lanes and cannot be fed from
  `W_s` logical lanes + replicas. Refuse.
- **Source blocks that don't fit:** the scaled flat size
  (`max_flat_workgroup_size * W_t/W_s`) must not exceed the target hardware
  threads/block max (1024 for gfx9/CDNA). The reduce is 512 -> 1024, exactly the
  limit. Refuse larger blocks rather than truncate.

### 10.5 Selection and plumbing

- `wave-projection.{h,cpp}`: `ScaledModuloReplicationProjection` derives from
  `ModuloReplicationProjection`, overriding the workitem-id-x remap
  (`emitWorkitemIdX`) and the packed-id emission (`emitPackedWorkitemId`, which
  bypasses the base phantom-lane clamp and packs the raw y/z), and exposing
  `usesScaledDispatch()` / `scaledDispatchFactor()`. The scaled dimension is
  always x (the wave-carrying dimension), so it is not exposed or reported --
  scaling any other dimension would spread the extra threads across different
  rows/planes instead of aliasing them as replicas.
- `c5-predicate-chain-classifier`: new `PredicateChainProjection::ModuloReplicationScaled`
  never refuses C5 (both `.x` lane-position and `.y`/`.z` wave-spanning predicates
  are correct by construction); the report flags `WaveNativeYzRefusal` so the
  raiser knows the upgrade applies.
- `raiser.cpp`: when the WaveNative C5 refusal is the `.y`/`.z` case, no matrix
  op is present, and the scaled block fits the hardware max, the raiser
  **automatically** retries under the scaled projection (no flag, no env --
  this is the default resolution). It widens
  `amdgpu-flat-work-group-size` by the factor and records the scaled factor
  on the transpile result. Ineligible kernels (matrix, or scaled size > hardware
  max) keep the refusal. `--force-scaled-modrep` selects it unconditionally for
  offline testing of kernels that do not hit the refusal.
- `source-hidden-args.cpp`: halves the workgroup/grid-size-x reads.
- Per-kernel runtime signal: the scaled factor is threaded raiser ->
  `RaiseResult` -> `PipelineResult` -> comgr transpile result
  (`amd_comgr_hotswap_transpile_result_get_info`, `..._SCALED_DISPATCH_FACTOR`)
  -> hsa loader (`HotSwapKernelRecord`) -> `PrepareHotSwapDispatchKernelObject`
  -> `PatchPublishedKernelDispatchPacket`, which scales `workgroup_size_x` and
  `grid_size_x` for exactly the scaled kernels. The scaled dimension is always x,
  so only the factor is threaded through. The factor is persisted in the
  translation cache so cache hits stay correct. No AQL field mutation beyond the
  x extent and no `group_segment_size` change are needed (LDS is per-block,
  logical-tid addressed).

### 10.6 Flags and tests

- No flag or env is needed to trigger the fix -- the y/z-refusal -> scaled
  upgrade is automatic. `raise_cli --force-scaled-modrep` (and
  `PipelineOptions::ForceScaledModrep`) is an offline testing knob only.
- Offline regression: `c5_predicate_chain_workitem_id_y_scaled_modrep.s`
  (default auto-upgrade + `--force` both raise with the x-remap, scaled
  `amdgpu-flat-work-group-size`, the `hotswap-scaled-dispatch` marker,
  and no `init_whole_wave`); `scaled_modrep_too_large_refuse.s` (default and
  `--force` both refuse an ineligible >512 block via the size gate);
  `scaled_modrep_wmma_refuse.s` (matrix refusal);
  `scaled_modrep_wgsize_virtualize.s` (x workgroup-size halving).
