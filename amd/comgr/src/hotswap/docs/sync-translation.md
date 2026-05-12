# Sync Translation — Barriers, Waitcnt, Fences, Memory Scopes

> **Status:** design proposal. Current raiser implements the 80% path
> (unified-barrier lowering, waitcnt-as-no-op, SC atomics) which is
> correct for workgroup-synchronous kernels that match LLVM's default
> memory model. This doc specifies the remaining semantics and the
> gates needed to refuse kernels whose synchronisation patterns fall
> outside that model.
>
> **Scope:** gfx1250 → gfx950. The barrier split is gfx12-only in our
> corpus, but the framework applies to any future ISA pair that
> diverges on synchronisation primitives.

---

## 1. Problem in one paragraph

The source binary encodes its synchronisation contract in five
intertwined surfaces: (a) workgroup rendezvous (`s_barrier*`), (b)
per-wave memory ordering (`s_waitcnt` and its split counterparts),
(c) memory-scope tags on atomics and fences, (d) cache control
(`global_inv`, `global_wb`, `buffer_inv`, …), and (e) cluster-level
synchronisation for multi-XCD dispatches. The target ISA has a
different primitive for each, and in several cases fewer primitives
than the source. Translation must preserve the **observable
happens-before graph** across waves and across memory, even when the
syntactic primitives disappear. That reduces to: express every
synchronisation act in LLVM IR at a semantic level the target
backend will re-materialise correctly on its own ISA, and fail loudly
when the source expresses something that has no LLVM-level
equivalent.

## 2. Why LLVM IR can carry most of this for us

Three facts make the translation tractable:

1. **LLVM owns waitcnt.** The backend inserts `s_waitcnt` at code
   emission based on the IR's memory-model ordering of loads, stores,
   and atomics. Any source-side `s_waitcnt` is redundant metadata
   once we have the IR dependences right — the backend rebuilds it
   for the target generation (unified `s_waitcnt` on gfx9, split
   counters on gfx12). This is why `handle_sopp.cpp:94` treats
   waitcnt as a no-op. Current correctness notes do not attribute any
   tracked Hotswap regression to dropped source waitcnts.
2. **LLVM owns atomic ordering and scope.** `AtomicOrdering` +
   `syncscope` are IR-level tags; the backend maps them to target
   cache-control bits (`glc`, `slc`, `dlc`, `scc` on CDNA;
   `th`/`scope` on gfx12). As long as the raiser tags atomics with
   the right (ordering, scope) pair, cross-ISA translation is
   automatic.
3. **Workgroup barriers are an intrinsic.** `llvm.amdgcn.s.barrier`
   is a stable cross-ISA intrinsic that lowers to the right barrier
   family on every target. That covers the 80% path where every wave
   in the workgroup reaches every barrier.

The remaining 20% is the set of source patterns that do *not* reduce
to those IR primitives. Named barriers, fences with non-standard
scopes, and multi-XCD cluster sync are the cases we must refuse.

## 3. Source model — gfx1250 (gfx12)

### 3.1 Split barriers

gfx12 replaced the monolithic `s_barrier` with two primitives:

- `s_barrier_signal imm16` — signals barrier `imm16`; does not
  block the wave. Lanes continue executing.
- `s_barrier_wait imm16` — blocks the wave until barrier `imm16`
  has been signalled by the required number of waves.

`imm16` is a barrier resource index (0..31 typical; 0 is the default
workgroup barrier that behaves like the legacy unified `s_barrier`).
Additional barrier resources (named barriers) can be set up at launch
time to coordinate subsets of waves — the canonical use case is
Triton's `warp_specialize` where producer and consumer waves rendezvous
on different named barriers.

SemOps: `S_BARRIER`, `S_BARRIER_SIGNAL`, `S_BARRIER_WAIT`
(`semop.hpp:24`).

### 3.2 Split wait counters

gfx12 split the combined `s_waitcnt` into category-specific waits:

- `s_wait_loadcnt N` — VMEM loads outstanding ≤ N
- `s_wait_storecnt N` — VMEM stores outstanding ≤ N
- `s_wait_dscnt N` — LDS ops outstanding ≤ N
- `s_wait_kmcnt N` — SMEM / kernarg loads outstanding ≤ N
- `s_wait_xcnt N` — export/message count ≤ N
- `s_wait_loadcnt_dscnt` — fused LOAD + DS wait
- `s_wait_alu imm16` — VALU dependency wait (gfx12 specific)
- `s_delay_alu`, `s_clause` — scheduling hints, not memory waits

SemOps: all enumerated at `semop.hpp:18-20`.

### 3.3 Atomic memory scopes

gfx12 atomics carry a two-level scope+scope-of-return encoding in
`th`/`scope` instruction modifiers. The typical scopes used by Triton
/ Gluon kernels:

- `agent` (device-scope) — for cross-workgroup atomics
- `workgroup` — for cross-wave-within-workgroup atomics (rare; LDS
  atomics cover most of these)
- `wavefront` — for cross-lane atomics (usually absent; LDS suffices)

The source code's intent is carried by how Triton compiles its
`atomic_add(..., sem="release", scope="gpu")` etc. down to MC
instructions. The raiser must recover that intent.

### 3.4 Cache control

Stand-alone cache operations:

- `global_inv scope` — invalidate a cache level
- `global_wb scope` — write back a cache level
- `global_wbinv scope` — both
- `buffer_inv scope` — same for buffer/V# accesses

These exist on gfx12 as first-class SOPP/FLAT instructions. They do
not have SemOps today. They appear in the kerneldex histograms for
some GPT-OSS kernels (RCP, softmax normalisation patterns).

### 3.5 Cluster barriers (multi-XCD)

gfx12 with multi-XCD SKUs exposes `cluster.arrive` / `cluster.wait`
via Gluon; these compile to a coordinated DS-based barrier across
XCDs. Not present in the captured corpus; listed for gate
completeness.

## 4. Target model — gfx950 (CDNA4)

### 4.1 What gfx950 has

- `s_barrier` — unified monolithic barrier. All waves in workgroup
  arrive, all leave. No named barriers.
- `s_waitcnt vmcnt(N) lgkmcnt(N) expcnt(N)` — combined, three
  counters. The backend inserts these.
- `AtomicOrdering` + `syncscope(…)` lower to `glc`, `slc`, `dlc`,
  `scc` bits on the atomic.
- `buffer_wbinvl1`, `buffer_wbinvl1_vol`, `buffer_wbl2` — cache
  control at L1/L2. Each is a SOPP instruction the backend emits.
- LDS atomics supported natively; cross-wave sync is via
  `s_barrier` + LDS.

### 4.2 What gfx950 does *not* have

- Named / split barriers. Only one workgroup rendezvous per
  `s_barrier`.
- Split wait counters.
- Cluster sync across XCDs. Multi-XCD coordination on CDNA is done
  at the CU scheduling level; there is no in-kernel primitive.
- `s_wait_alu`. The gfx950 backend resolves VALU hazards with
  register-read stall cycles, not explicit waits.

## 5. Translation design — per primitive

### 5.0 Target-capability dispatch (native vs. collapse)

Every sync primitive in the source has a target-capability branch that
mirrors the matrix axis (`matrix-translation.md §5.0`): emit the same
primitive natively when the target supports it, and decompose (or
collapse, or refuse) when it does not.

The sync axis is distinct in that some primitives are **added** in
gfx12 (split barriers, split wait counters, `s_wait_alu`), while
others are **shared** (atomic scopes via LLVM's `syncscope`, workgroup
barrier). The dispatch table is therefore denser than for matrix.

#### 5.0.1 Capability bits

Extend `ISAProfile` with:

| New bit | Backing feature | Governs |
|---|---|---|
| `hasSplitBarriers` | `FeatureGFX12Insts` (split `s_barrier_signal` / `s_barrier_wait`) | §5.1 |
| `hasSplitWaitCnt` | `FeatureGFX12Insts` (split `s_wait_loadcnt`, `s_wait_dscnt`, …) | §5.2 |
| `hasSWaitAlu` | `FeatureGFX12Insts` (`s_wait_alu`) | §5.2 |
| `hasClusterBarriers` | `FeatureClusters` (upstream TableGen name) | §5.5 |

`hasCacheScopedOps` (for `global_inv` / `global_wb` / …) is not a
separate bit — gfx950 and gfx12 have overlapping but non-identical
families. §5.4 handles the per-op mapping instead of treating it as
one capability.

#### 5.0.2 Dispatch table

| Source primitive | Target has capability | Action | Target lacks capability | Action |
|---|---|---|---|---|
| `S_BARRIER_SIGNAL/WAIT` (`imm16=0`) | `hasSplitBarriers` | emit split intrinsic pair (§5.1.a) | — | collapse to `llvm.amdgcn.s.barrier` (§5.1.b, existing) |
| `S_BARRIER_SIGNAL/WAIT` (`imm16≠0`) | `hasSplitBarriers` | emit named-barrier intrinsic (§5.1.a) | — | **refuse** (`namedBarrierUnsupported`; §5.1) |
| `S_WAIT_{LOAD,STORE,DS,KM,X}CNT` | `hasSplitWaitCnt` | emit matching `llvm.amdgcn.s.waitcnt.*` | — | no-op (backend re-derives on combined counter) (§5.2) |
| `S_WAIT_ALU` | `hasSWaitAlu` | emit native intrinsic | — | no-op (backend handles VALU hazards) (§5.2) |
| `S_BARRIER` (monolithic) | always (every AMDGPU target) | emit `llvm.amdgcn.s.barrier` | — | — |
| Atomics with `(ordering, scope)` | always (LLVM IR is shared) | emit `atomicrmw` with tags (§5.3) | — | — |
| `GLOBAL_INV/WB/WBINV`, `BUFFER_INV` | per-target table (§5.4) | emit matching intrinsic | — | refuse |
| Cluster barriers | `hasClusterBarriers` | emit cluster intrinsic pair | — | refuse (§5.5) |

#### 5.0.3 Consequences for same-family retarget

The gfx1251 → gfx1250 path takes the native branch for every sync
primitive: split barriers stay split, split wait counters stay split,
`s_wait_alu` stays, and atomics keep their decoded `(ordering, scope)`
tuple. The "collapse" paths below (§5.1.b's monolithic barrier,
§5.2's no-op waitcnt) are reached only when targeting pre-gfx12 ISAs
like gfx942/gfx950. Same as matrix: there is no fast path in the
byte-patching sense — the capability branch above **is** the fast
path.

### 5.1 Barriers

#### 5.1.a Native split path — `hasSplitBarriers` targets

When the target ISA also exposes split barriers (gfx12+), emit the
split intrinsic pair directly:

```cpp
if (sop == SemOp::S_BARRIER_SIGNAL && ctx.targetIsa.hasSplitBarriers) {
  auto *fn = Intrinsic::getOrInsertDeclaration(
      &ctx.M, Intrinsic::amdgcn_s_barrier_signal);
  ctx.B.CreateCall(fn, {ctx.B.getInt32(di.getImm(0))});
  hr.handled = true;
  return hr;
}
if (sop == SemOp::S_BARRIER_WAIT && ctx.targetIsa.hasSplitBarriers) {
  auto *fn = Intrinsic::getOrInsertDeclaration(
      &ctx.M, Intrinsic::amdgcn_s_barrier_wait);
  ctx.B.CreateCall(fn, {ctx.B.getInt32(di.getImm(0))});
  hr.handled = true;
  return hr;
}
```

The `imm16` barrier resource id passes through unchanged — named
barriers are valid on split-capable targets. G1 (§7) only fires on
pre-gfx12 targets where the collapse in §5.1.b cannot represent them.

#### 5.1.b Collapse path — pre-gfx12 targets (existing policy)

When the target is pre-gfx12 (`!hasSplitBarriers`), collapse the split
primitives to the monolithic barrier. Current implementation in
`handle_sopp.cpp:70-84`:

- `S_BARRIER`, `S_BARRIER_WAIT` → `call void @llvm.amdgcn.s.barrier()`
- `S_BARRIER_SIGNAL` → no-op (handled = true; emits nothing)

This is correct iff the source uses the **default barrier resource
(0)** exclusively and *every* wave reaches *every* signal and wait.
Under that restriction, the wait becomes a full workgroup barrier and
the signal is redundant with respect to the wait. Since every wave
signals before it waits, the combined behaviour matches a monolithic
`s_barrier`.

It is incorrect in three cases:

| Case | Why it breaks |
|---|---|
| `imm16 != 0` | Source uses named barriers to rendezvous a **subset** of waves. Monolithic lowering rendezvouses all waves, overclosing the happens-before graph and deadlocking if some waves never reach this wait. |
| Producer signals `imm16=0` but consumer waits on `imm16=1` (or vice versa) | Different barrier identities — collapsing to one barrier reconverges waves that source kept divergent. |
| One wave signals twice without a wait in between | The signal count matters for named barriers. The collapse loses that. |

**Refinement:** the named-barrier refusal only applies on the
collapse path. Per-kernel gate: if `!targetIsa.hasSplitBarriers` and
any `S_BARRIER_SIGNAL` / `S_BARRIER_WAIT` instruction has `imm16 != 0`,
refuse with `RaiseFailure::namedBarrierUnsupported`. When the target
has split barriers, named barriers pass through unchanged via §5.1.a.

At the raiser level that means:

```cpp
if (sop == SemOp::S_BARRIER_SIGNAL || sop == SemOp::S_BARRIER_WAIT) {
  int64_t barrierId = di.getImm(0);
  if (ctx.targetIsa.hasSplitBarriers) {
    // §5.1.a: native split path, barrierId preserved
    return emitNativeSplitBarrier(ctx, sop, barrierId);
  }
  if (barrierId != 0)
    return HandlerResult::fail(
        RaiseFailure::namedBarrierUnsupported(di, barrierId));
  // else: fall through to existing collapse policy (§5.1.b)
}
```

Named barriers are the **load-bearing primitive** of Triton's
`warp_specialize`. Refusing on the collapse path is the correct
behaviour for pre-gfx12 targets until we have a real producer/consumer
model in IR; on gfx12+ targets the primitive exists and we pass it
through.

### 5.2 Wait counters

#### 5.2.a Native split path — `hasSplitWaitCnt` targets

When the target also has split counters (gfx12+), preserve them
directly. Each `S_WAIT_*CNT` SemOp maps 1:1 to
`llvm.amdgcn.s.waitcnt.*`; `S_WAIT_ALU` lowers when `hasSWaitAlu` is
set. The backend re-emits the same mnemonic on output. No information
is discarded.

#### 5.2.b No-op collapse path — pre-gfx12 targets (existing policy)

When the target has only the combined counter (`!hasSplitWaitCnt`),
all waitcnt SemOps fall through as no-ops (`handle_sopp.cpp:94`). The
backend's memory-model-driven waitcnt insertion on gfx950 replaces
them.

This is correct on two conditions:

1. The raised IR faithfully represents every memory dependence the
   source's waitcnt was protecting. That is the default: `load` and
   `store` SSA values carry dependences, and the backend inserts
   `s_waitcnt` in the output.
2. No waitcnt is acting as a **cross-wave** barrier. Waitcnt is a
   per-wave primitive by definition, so this is tautologically true —
   but some programming styles use `s_waitcnt` + volatile memory
   accesses to implement a hand-rolled producer/consumer across
   waves. That pattern requires a barrier, not a waitcnt, and if a
   kernel relies on it we would detect it only empirically.

**No rewrite needed.** The sem_op_attrs row for every waitcnt SemOp
sets `isScheduleHint = true` (a new attr bit) and the raiser's
per-kernel gate verifies that is the final classification. If a
future waitcnt variant has real IR-level semantics (e.g., the gfx12
`s_wait_alu` with VALU dependency tracking that the backend's memory
model does not capture), add a new attr bit and a handler, don't
special-case inline.

### 5.3 Atomic memory scopes

Current policy (`handle_flat.cpp:291` etc.): every atomic emits
`AtomicOrdering::SequentiallyConsistent` (for `atomicrmw`) or
`Monotonic` (for `cmpxchg`), with the default `syncscope` (system).

This is **conservative-correct for gfx950**: SC + system emits the
strongest cache-flush pattern, which subsumes every weaker scope the
source could have wanted. It is also **performance-wrong** for the
common Triton case where the source intent is `release`
`agent`-scope. For correctness of the translated kernel, the
conservative policy ships today; for parity with the source's
performance, we need to recover the source intent from the gfx12
th/scope bits.

**Refinement:** extend `DecodedInst` with parsed `(scope, ordering)`
fields from the gfx12 th/scope modifiers, and use them in
`handle_flat.cpp` / `handle_mubuf.cpp` when emitting `atomicrmw`.
The mapping table:

| gfx12 encoding | IR ordering | IR syncscope |
|---|---|---|
| `th=0, scope=CU` | `monotonic` | `"workgroup-one-as"` |
| `th=0, scope=DEV` | `monotonic` | `"agent-one-as"` |
| `th=0, scope=SYS` | `monotonic` | `""` (system) |
| `th=RT (release)` | `release` | <scope as above> |
| `th=NT (nontemporal)` | `monotonic` | scope + nontemporal metadata |
| `th=HT (high temporal)` | `monotonic` | scope (no extra tag) |
| `th=BYPASS` | `monotonic` | scope, plus `!nontemporal` |

The `"-one-as"` scope variants are the AMDGPU-specific single-address-
space scopes; they lower correctly on both gfx12 and gfx9.

**Refusal case:** a scope the target does not support. `wavefront`
scope is not a standard LLVM AMDGPU scope — reject until the
AMDGPULowerMemoryModel pass learns it.

### 5.4 Standalone cache operations

These SemOps do not yet exist. Adding them is the first real ISA
divergence handled at the sync-translation level:

| New SemOp | gfx12 source | gfx950 lowering |
|---|---|---|
| `GLOBAL_INV` | `global_inv scope` | `call void @llvm.amdgcn.buffer.wbinvl1()` for CU/DEV; refuse SYS |
| `GLOBAL_WB` | `global_wb scope` | `call void @llvm.amdgcn.buffer.wbl2()` for DEV |
| `GLOBAL_WBINV` | `global_wbinv scope` | composition of the two |
| `BUFFER_INV` | `buffer_inv scope` | same as `GLOBAL_INV` on gfx950 (unified L1) |

If the source's `scope` is not representable with the target's
intrinsics (e.g., gfx950 has no per-XCD cache op), refuse.

Most Triton/Gluon kernels never emit standalone cache ops — the
AMDGPULowerMemoryModel pass derives them from atomic ordering+scope.
We still need the SemOps so we can catch hand-written kernels (like
some Tensilelite persistent GEMMs) that emit them explicitly.

### 5.5 Cluster barriers

Not yet a SemOp. Cluster sync compiles to `ds_wrap_rtn_b32` +
`s_waitcnt_vscnt` + a coordinated write-back pattern. Dispatched by
`ISAProfile::hasClusterBarriers`:

- **Target has cluster barriers** → preserve the primitive (emit the
  Gluon-level cluster intrinsic pair, `cluster.arrive` /
  `cluster.wait`).
- **Target lacks cluster barriers** → refuse. Multi-XCD coordination
  has no intra-kernel equivalent on pre-cluster ISAs.

Detectable at decode via `hidden_cluster_size` or the Gluon-emitted
pattern at the disassembly level. Startup-era gate only once a real
kernel using it lands; placeholder in the refusal taxonomy.

## 6. Decision procedure (per kernel)

Run after ABI gates (`abi-translation.md §7`), before any IR
emission for sync-related SemOps:

```
for each decoded instruction:
  if sop is barrier family:
    if targetIsa.hasSplitBarriers:
      emit matching split intrinsic (§5.1.a), imm16 preserved
    elif imm16 != 0:
      refuse (namedBarrierUnsupported)
    else:
      emit llvm.amdgcn.s.barrier (§5.1.b)
  if sop is waitcnt family:
    assert attrs.isScheduleHint == true
    if targetIsa.hasSplitWaitCnt:
      emit matching split waitcnt intrinsic (§5.2.a)
    else:
      skip (no-op, §5.2.b)
  if sop is atomic:
    parse th/scope from decoded modifiers
    map to (ordering, syncscope)
    if scope is unrepresentable: refuse
    emit AtomicRMW with tags
  if sop is GLOBAL_{INV,WB,WBINV} / BUFFER_INV:
    map to target intrinsic if supported
    else: refuse
  if sop is cluster barrier:
    if targetIsa.hasClusterBarriers:
      emit matching intrinsic
    else:
      refuse
```

Every refusal carries the decoded instruction offset and a reason
string. No silent downgrades.

## 7. Principled fail-loudly gates

### G1 — Named-barrier gate (per-kernel)

Fires on any `S_BARRIER_SIGNAL` / `S_BARRIER_WAIT` with `imm16 != 0`.
Reason: `namedBarrierUnsupported`.

### G2 — Waitcnt classification (startup)

New `SemOpAttrs::isScheduleHint` bit. Startup verifier:
`verifyWaitcntAttrCoverage` ensures every SemOp whose MCInstrDesc
flags it as SOPP *and* whose mnemonic matches `s_wait*|s_clause|s_delay_alu`
has `isScheduleHint = true`. Catches new LLVM-emitted wait variants
that we haven't audited.

### G3 — Atomic scope gate (per-kernel)

Wrapped around each atomic emission: if decoded `(th, scope)` does
not map to a supported IR `(ordering, syncscope)` tuple, refuse with
`atomicScopeUnsupported`.

### G4 — Cache-op coverage (per-kernel)

For every instruction whose mnemonic matches the cache-op family,
require a handler that either emits the target equivalent or raises
`cacheOpUnsupported`. Default (no handler row) is refuse.

### G5 — Cluster sync (per-kernel)

Pattern-match the cluster barrier signature (DS wrap + scoped
waitcnt) at raise time; refuse on hit.

## 8. Open design questions

1. **Per-wave vs per-workgroup split in the named-barrier refusal.**
   A kernel that uses named barrier `0` only (which is the default
   workgroup barrier) is safe. A kernel that uses barriers `0` and
   `1` with `1` as a partition of waves is not. Does it matter for
   the refusal granularity? Current answer: no, single refusal is
   fine. If a realistic kernel shows up with *only* `imm16=0` reads
   that the disassembler happens to render with an explicit operand,
   the gate needs a second reading to check the canonical form.
2. **Waitcnt correctness proof.** "Backend handles it" is correct
   *assuming* the raised IR encodes all memory dependences. Are
   there source patterns (e.g., `s_load_*` under divergent EXEC
   whose result is read conditionally) where the source's explicit
   waitcnt carries a dependence the IR loses? We have not seen one,
   but the question is worth auditing once SPE is tightened
   (`wave-size-translation.md §5.1`).
3. **Atomic scope default.** When decoded `th/scope` are absent
   (instruction has no modifiers), we currently default to SC+system.
   Triton usually emits with explicit modifiers; Tensilelite sometimes
   does not. Should the default be `monotonic` + `agent`? That
   would match the usual intent but weakens the conservative-correct
   posture. Recommendation: keep SC+system default until we have
   specific kernels failing performance-wise, then revisit with a
   per-kernel override rather than a global policy change.
4. **Scope recovery when source uses raw `s_memrealtime` / fence
   intrinsics.** These appear in a handful of Tensilelite persistent
   kernels. Not in scope for the GPT-OSS North Star; mentioning so
   the taxonomy stays complete.

## 9. Engineering tasks

### T1 — Named-barrier gate (G1)

Modify `handle_sopp.cpp:74-84` to read `di.getImm(0)` and refuse on
non-zero. 15 LoC + one test.

### T2 — Waitcnt classification (G2)

Add `isScheduleHint` to `SemOpAttrs`. Register all `S_WAIT_*`,
`S_CLAUSE`, `S_DELAY_ALU`, `S_WAITCNT` SemOps in a new
`getHandlerSOPPScheduleAttrs()` spec list. Add
`verifyWaitcntAttrCoverage()` called from raiser init. 80 LoC.

### T3 — Atomic scope recovery (G3)

Extend `DecodedInst` with parsed `th`/`scope` fields from gfx12
modifiers (decode-side change in `decode.cpp`). Rewrite
`handle_flat.cpp:291-293` / `handle_mubuf.cpp` to use them. Add
mapping table. ~250 LoC.

### T4 — Cache-op SemOps (G4)

Add `GLOBAL_INV`, `GLOBAL_WB`, `GLOBAL_WBINV`, `BUFFER_INV`,
`S_MEMREALTIME` (latter for gate only). Handlers emit the right
intrinsic on gfx950. ~200 LoC.

### T5 — Cluster barrier gate (G5)

Pattern matcher in the post-decode pass; refuse on hit. 80 LoC.

Priority order: **T1 first** (cheap, eliminates a correctness hole
for any future warp-specialised kernel). T2 before T3 (T2 formalises
the current behaviour; T3 improves performance but doesn't fix
correctness). T4 and T5 follow once real kernels hit them.

## 10. Relationship to other axes

- **SPE (`wave-size-translation.md`):** barriers under divergent EXEC are
  SPE's responsibility — a barrier inside a predicated diamond is
  wrong for all source ISAs and SPE already refuses that pattern
  (every wave must reach a barrier). Sync-translation assumes SPE
  is already enforced.
- **Matrix (`matrix-translation.md`):** MFMA and WMMA both have
  implicit waitcnt requirements the backend handles. No
  sync-specific handling required.
- **Async / tensor-copy families:** gfx12 async-copy + `async_wait`
  is a category of wait counter (`s_wait_xcnt`) that we handle as a
  no-op today when emulation lowers the operation to synchronous
  buffer loads. Once async-copy is natively lowered, `s_wait_xcnt`
  has to become a real dependency in the IR.
- **Cross-cutting capability dispatch:** §5.0 is the sync-axis
  instance of the project-wide "emit native when the target supports
  it, decompose only when it does not" principle. See
  `target-capability-dispatch.md`.
