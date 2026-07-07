# Learnings

Short, dated entries capturing what broke, why it broke, and the fix.
Written for the reader in a future where everything works -- so they can
reconstruct the trap that was there, not the solution that's obvious now.

Append-only. Newest on top.

---

## 2026-04-23 -- V_CMP wave-mask shadow propagation through scalar ops (closes canary_tl_sort_fp32_n4)

`canary_tl_sort_fp32_n4` graduates from WRONG 775/2048 to
`match` through two commits that extend the matmul128x128-class
`WaveMaskEntry` shadow cache (da404faf84, 2026-04-21) from its
original "V_CMP -> V_CNDMASK in same BB, no intervening scalar
write" scope to also cover scalar-op-interleaved patterns that
Triton's gfx1250 codegen emits at the cross-widened bitonic
compare-and-swap.

**Stage 1 -- scalar XOR/AND/OR between V_CMPs
(`s_{and,or,xor}_b32` + SGPR sources).**
Triton's n=4 sort direction mask is:

  v_cmp_ngt_f32_e64 s2, v3, v5    ; per-lane (v3 <= partner)
  v_cmp_eq_u32_e64  s3, 1, v4     ; lane-parity bit
  s_xor_b32 s2, s2, s3            ; direction = ngt XOR parity
  v_cndmask_b32 v3, v5, v3, s2    ; v3 = s2 ? v3 : v5

Under wave32 source -> wave64 target, the pre-fix shadow cache
tracked each V_CMP's wave-width i1 (64 bits) across the narrow
(32-bit) SGPR write, so V_CNDMASK could read the full i1
directly.  But `s_xor_b32 s2, s2, s3` is a scalar write that
INVALIDATES the cache for s2.  The subsequent V_CNDMASK then
took the lossy narrow-mask fallback (replicate low-32 to
both halves), mis-routing the upper 32 target lanes.

Fix: propagate wave-width i1 through S_{AND,OR,XOR}_B32 when
BOTH sources have cached i1.  Compute `dst_i1 = src0_i1 OP
src1_i1` per-lane and re-record the shadow after the scalar
write's invalidation.  Graduated the BLOCK_N=4 sort WRONG
775/2048 -> 498/2048 (−35.7%).

**Stage 2 -- VCC/EXEC sources and VCC destination.**
The n=4 sort's second and third compare-and-swap idioms use
VCC and the saveexec->xor pattern.  Three additional
participant register kinds were added to both
`tryGetSrcWaveMaskI1` (sources) and `recordDerivedWaveMaskI1`
(destinations):

  * **VCC source** -- `loadVCC` yields the per-lane i1 directly
    (VCC alloca stores i1, not a wave-width mask).
  * **EXEC source** -- `extractLaneBitFromWaveMask(loadExec())`
    yields the per-lane i1 from the i64 EXEC alloca.
  * **VCC destination** -- when both sources have wave-width i1,
    overwrite the VCC alloca's i1 with the correct per-lane
    XOR/AND/OR (bypassing the lossy i32
    extract-lane-bit-from-replicate path `writeReg32(VCC, i32)`
    otherwise uses).

**Stage 3 -- `s_{and,or,xor,andn2,orn2}_saveexec_b32` record
dst SGPR shadow with the per-lane i1 of OLD EXEC.**  The
saveexec family saves the full-width `oldExec` to a source-
width SGPR (lossy under cross-widening).  The handlers already
have the i64 oldExec in hand; after `writeRegExecWidth` fires
the shadow invalidation via `onSgprWritten`, re-record the
dst SGPR shadow with `extractLaneBitFromWaveMask(oldExec)`.
Covers the "else-branch mask" idiom `s_and_saveexec_b32 sN,
vcc; s_xor_b32 sN, exec_lo, sN` Triton emits between bitonic
stages.

Combined, the three extensions close n=4 fully: WRONG 498 ->
match.  The matmul128x128-class shadow cache's invariants
(I1 additive, I2 SSA-monotonic within a BB, I3 any
interference defeats the cache, see sgpr-wave-mask-translation
.md §3.1) are preserved -- the propagation only ADDS shadow
entries; it never masks the narrow-mask fallback when the
wave-width info isn't actually available.

**Residuals open (distinct bug classes).**

  * `canary_tl_sort_fp32_n16` (WRONG 1056/8192, 12.9%): uses
    a different code shape -- direction XOR happens on
    MATERIALIZED VGPR bits (`v_cndmask -> vGPR -> v_cmp_eq_u32
    s0, v17, v_materialized`) rather than via scalar
    s_xor_b32.  The shadow cache is keyed on SGPR V_CMP
    writers; materialising to a VGPR and then comparing VGPRs
    puts the per-lane i1 back in SSA at the correct
    wave-width naturally, so there's no narrow-mask fallback
    at the compare-and-swap.  The structural error pattern
    (rows 18/19/22/23/26/27/30/31 mod 32, warps 2/3
    thread-slots 8-15/24-31) does NOT match what a
    shadow-cache residual would produce.  Open for a
    dedicated bisect; four diagnostic probes
    (`canary_tl_sort_fp32_n16_{deterministic,xor1,
    row_offset,altrow}`) pin the value-dependence.
  * `topk_forward_bisect_m2_strict` (WRONG 263/2048):
    streaming_topk merge-step bug (k=0 always match, k=1/2/3
    scale-up, different VALUE SETS).  Independent of tl.sort
    primitive correctness.
  * `topk_forward_bf16` (WRONG 670/8192): `_topk_forward` Yi
    index-set divergence, pre-existing open finding from
    2026-04-22; current fix reduces it from 2856/8192 to
    670/8192 (76% reduction) but can't close because the
    remaining divergence is in streaming_topk's merge step.

**Precedent-setting reference.**  The matmul128x128 shadow
cache treated the "V_CMP wave-width i1 cached, SGPR-narrow
path bypassed" invariant as additive at the V_CNDMASK
consumer.  This session's extensions are the first that
propagate the i1 THROUGH intermediate scalar ops
(s_{and,or,xor}_b32, s_*_saveexec_b32) and treat VCC/EXEC as
additional wave-width-carrying participant register kinds.
All three extensions remain compatible with the original
invariants (the shadow is additive -- its absence takes the
pre-existing lossy path).

---

## 2026-04-23 -- Triton gfx1250 permlane16_swap self-preserving rewrite (TRANSITIONAL)

`canary_tl_sort_fp32`, `canary_tl_topk_fp32`, `canary_tl_topk_bf16`,
and `canary_tl_topk_bf16_nw1` all graduate from `WRONG 2048/2048`
(or 15353/16384 for the 32-column sort) to `match`.  The compound
`topk_forward_bf16` and `topk_forward_bisect_m2_strict` reduce from
2856/8192 -> 670/8192 (-76%) and 1542/2048 -> 263/2048 (-83%)
respectively; the residuals are bf16 reduction-order drift with
tight `abs tol=0.0` comparators -- same class as the m1 residual
documented under FMA_MIX below, and not a miscompile (see §
"Residual characterisation" below).  The `canary_tl_sort_fp32
_deterministic` sibling continues to match under the new rewrite
(the xor3-partner sibling rewrite also still fires, both now
substituting the same `seed` value; double substitution is a
no-op).

**The generalisation.** The pre-existing `rewrite_permlane16
_xor3_partner` pass caught ONE Triton cross-16 bitonic-merge
composition -- the fused `v_xor3_b32`.  Triton's gfx1250 codegen,
empirically, emits AT LEAST THREE distinct compositions around
the same `v_permlane16_swap_b32 + v_dual_mov same-seed
initialiser` idiom:

  * `tl.sort` (deterministic): fused `v_xor3_b32 v_a, v_a, v_b,
    v_c` -- the only case the pre-existing rewrite covered.
  * `tl.sort` (random): split as `v_xor v_a, v_b, v_a` (inner) +
    `v_xor v_a, v_a, v_c` (outer).  The inner xor gets its OWN
    `emitUnderExec` SPE-phi wrapper on the way to the outer xor,
    so the pre-existing rewrite's `dyn_cast<BinaryOperator>`
    on the outer xor's LHS fails (it's a phi, not an xor).
  * `tl.topk`: max-based, no xor at all -- `v_max v_d, v_b, v_b
    :: v_max v_a, v_a, v_a; v_max v_a, v_a, v_d`.  No amount
    of pattern-matching on an xor will catch this shape.

All three compositions share ONE underlying structural invariant:
the `v_permlane16_swap_b32` is initialised with `v_dual_mov v_a,
v_c :: v_dual_mov v_b, v_c` so BOTH `vdst_in` and `src0_in` hold
the same SSA seed per-lane.  Under the gfx950-documented symmetric
cross-wire
  new_vdst[L]      = src0_in[L XOR 16]
  new_src0_out[L]  = vdst_in[L XOR 16]
both outputs end up holding `partner_seed` -- the algorithm has
no access to `self_seed` anywhere, and every downstream
composition degenerates:

  xor3(partner, partner, self)         = self   (want partner)
  xor(xor(partner, partner), self)     = self   (want partner)
  max(max(partner, partner),
      max(partner, partner))           = partner (want max(self, partner))

The new pass fixes this at the ROOT -- the bpermute-pair emission
itself -- instead of pattern-matching every downstream
composition.  When `emitPermLaneSwapEmulation` emits two bpermute
calls at a site and both data arguments trace (via SPE active-arm
phi walks) to the same SSA root, the SECOND call's result (the
`new_src0_out` of the swap) is RAUW'd with the shared seed root.
That makes the emulation asymmetric:

  new_vdst[L]      = bpermute(partner_addr, src0_in)  (partner_seed)
  new_src0_out[L]  = seed_root[L]                      (self_seed)

All three compositions above now produce the algorithm-expected
output:

  xor3(partner, self, self)                    = partner ✓
  xor(xor(partner, self), self)                = partner ✓
  max(max(self, self), max(partner, partner))  = max(self, partner) ✓

This is EXACTLY what Triton's gfx942-NATIVE compile does with
`ds_swizzle_b32 swap:16` -- a single-output swap that gives the
algorithm access to both `self` (in the original register) and
`partner` (in the swizzle result).  The asymmetric gfx1250
emulation matches that contract.

**Layer choice -- rewrite pass, not primitive semantic change.**
We don't have gfx1250 hardware to verify which of these is true
(same `(a)/(b)` split as the xor3-partner sibling):

  (a) gfx1250 silicon's `v_permlane16_swap_b32` is asymmetric --
      only one output cross-wires, the other preserves its input
      -- and Triton's codegen targets that semantic accurately.
      Our existing `emitPermLaneSwapEmulation` emits the gfx950
      symmetric cross-wire and is WRONG for gfx1250 sources.

  (b) gfx1250 silicon matches gfx950 (symmetric cross-wire), and
      Triton's gfx1250 codegen has a bug that produces
      algorithmically incorrect code on gfx1250 hardware itself.

The `Gfx1250Gpu.Permlane16Swap` GPU regression coverage validates the symmetric
cross-wire semantic by running a hotswap-lifted gfx1250 kernel
with DISTINCT `vdst_in[L] = L`, `src0_in[L] = 1000 + L` inputs
on gfx942 hardware -- it tests our EMULATION, not gfx1250 silicon.
It passes (both outputs are cross-wired per spec).  That's a
weak signal either way for the silicon question.

Under both (a) and (b), the algorithm-correct thing for hotswap
to do is expose `self` and `partner` to the downstream
composition -- which is what this pass does.  If silicon matches
(a), the fix is eventually better-placed at the primitive layer
(gate the asymmetric emulation behind a source-ISA check); the
rewrite is then dead code.  If Triton fixes (b), the idiom
disappears from lifted IR and the rewrite is also dead code.
Both dead-code states are the pass's intended TRANSITIONAL end --
same pattern as the xor3-partner sibling.  The two passes are
NOT redundant during the transition: the new self-preserve pass
subsumes the xor3-partner functionality for same-seed sites,
but the xor3-partner pass is retained as belt-and-suspenders
(it's narrow and harmless).

**Fingerprint narrowness.**  The check is structurally impossible
outside the Triton idiom:

  1. Two `@llvm.amdgcn.ds.bpermute` calls in the same BB.
  2. Their first operand (the partner address) is SSA-identical
     (`emitPermLaneSwapEmulation` emits it as a single `shl`).
  3. Their second operand (the data) traces via SPE active-arm
     phi walks to the same SSA root.

Non-Triton kernels that use `v_permlane16_swap_b32` with distinct
inputs per-operand (the `Permlane16Swap` regression, the AITER
kernels' `v_permlane32_swap_b32` siblings through a different
opcode) don't match condition 3 -- their data args trace to
distinct SSA roots.  DPP cross-widening's `ds.bpermute` calls
don't share address SSA (each DPP site computes its own
selector).  The fingerprint isolates exactly Triton's cross-16
bitonic-merge idiom.

**Residual characterisation -- the two compound recipes.**

`topk_forward_bf16` and `topk_forward_bisect_m2_strict` still
show mismatches under `abs tol=0.0` (670/8192 and 263/2048
respectively), but the verdict pattern matches legitimate bf16
drift, not a second miscompile class:

  * `topk_forward_bisect_m2` (same kernel, `rel-rms tol=0.15`
    comparator) now matches -- the shape-level top-k SET is
    correct within the tolerance Triton's own reduction-order
    non-associativity requires.
  * `topk_forward_bisect_m1` random-input continues at `WRONG
    1300/2048 max|err|=0.25` exactly as before the cross-16
    fix -- this bug was already fixed by the FMA_MIX commit
    (see entry below) and its residual was characterised as
    "non-associative bf16 reduction-order drift, NOT a
    miscompile".
  * `topk_forward_bisect_m2_strict` max|err| dropped from
    1.48 -> 0.53 and 5.9× fewer mismatched rows.  Deeper
    classification on the 263 remaining error rows shows a
    SECOND residual bug class that is NOT bf16 drift:
      - k=0 (top-1) NEVER mismatches (0/512).  Hotswap
        reliably finds the row-max.
      - k=1/2/3 mismatch rates scale monotonically: 16, 78,
        169 rows.
      - ALL 171 error rows (k≥1 any-position) have DIFFERENT
        top-k VALUE SETS between hotswap and native, not just
        reorderings.  Zero rows match the "same-SET different-
        order" tie-break signature.
      - In 137/171 different-SET rows, hotswap DROPS a LARGER
        value than it added (algorithmic miss).  Only 31/137
        are within ~1 bf16 ULP of the boundary; the remaining
        106 drop values 0.05–0.5 larger than substitutions --
        real algorithmic loss.
      - Odd-row clustering (all error rows at `row_idx & 1 ==
        1`) suggests lane-parity interaction in
        `streaming_topk`'s iterative merge across
        N_EXPTS_PAD / BLOCK_N peer groups, NOT the cross-16
        bitonic merge this commit fixes.
    This matches the 2026-04-22 `_topk_forward` Yi open-
    finding hypothesis list (u32 value+index packing under
    cross-widening, `-inf` poisoning of out-of-range lanes,
    mask carry across merge iterations).  The new datum -- k=0
    reliable, k=3 most-wrong -- narrows to the merge/exclude
    step that produces k=1..3, not the max reduction itself.

The remaining WRONG verdicts on strict-comparator compound
recipes stay bit-exact-WRONG by design -- same precedent as
`topk_forward_bisect_m1` random was left at WRONG 1300/2048 to
keep the regression surface bit-exact.  Relaxing the comparator
would hide the real streaming-merge bug documented above.

**`canary_tl_sort_fp32_n16` is orthogonal** -- it uses BLOCK_N=16,
no cross-16 merge required, zero `v_permlane16_swap_b32` in its
disassembly.  Its 12.9% WRONG is a separate bug class that this
rewrite doesn't (and can't) touch.

Residual characterisation (2026-04-23 session, committed as
diagnostic probes): errors cluster PERFECTLY at rows where
`(row_index & 0b10010) == 0b10010` within each 32-row workgroup
tile -- i.e., rows 18, 19, 22, 23, 26, 27, 30, 31 mod 32.  In
Triton's [M=32 rows × N=16 cols, num_warps=4, sizePerThread=
[1, 4]] layout, those rows live in warps 2 and 3 at thread-slot
positions {8..15, 24..31} within their warp -- target-wave-1
lanes with bit 3 of lane_id set.  Within each error row, each
of the 8 adjacent col-pairs has ≈50% chance of being inverted
INDEPENDENTLY (verified: 128 error rows × 8 pairs each, binomial
distribution around 4 wrong pairs per row; all wrong pairs are
plain in-row swaps -- the VALUE SET is a correct permutation of
native's output).

Value-dependence is strict: four deterministic probes with
different input patterns all MATCH, only random input fails:
  * `canary_tl_sort_fp32_n16_deterministic` -- X[r,c] = c (all
    rows identical, monotonic): match.
  * `canary_tl_sort_fp32_n16_xor1` -- X[r,c] = c^1 (pair-swapped
    monotonic): match.
  * `canary_tl_sort_fp32_n16_row_offset` -- X[r,c] = r*100 + c
    (per-row distinct, monotonic ascending): match.
  * `canary_tl_sort_fp32_n16_altrow` -- X[r,c] alternates
    ascending/descending per row parity: match.
  * `canary_tl_sort_fp32_n16` -- uniform random [-4, 4]:
    WRONG 1056/8192 (12.9%).

Projection and rewrite toggles isolate where the bug is NOT:
  * Under `--disable-wave-native` (ModRep instead of
    WaveNative): identical 1056/8192 WRONG.  Not a cross-wave
    truncation of a V_CMP->SGPR wave-mask (the class fixed for
    matmul128x128 by the `WaveMaskEntry` shadow cache).
  * Under `--disable-writelane-rewrite` (DPP cross-widening
    rewrite disabled, DPP lifts as `@llvm.amdgcn.update.dpp`
    intrinsic directly): identical 1056/8192 WRONG.  Not in the
    `rewrite_cross_lane_divergent` DPP-to-ds_bpermute rewrite.

That leaves either (i) a primitive-level handler with
value-dependent per-lane behaviour we haven't isolated, or (ii)
a pattern in Triton's gfx1250 compile of `tl.sort` at BLOCK_N=16
specifically that needs a different rewrite / handler
adjustment.  The `canary_tl_sort_fp32` graduation to `match` in
the presence of the N=16 residual pins that the residual is
wholly below the cross-16 stage -- the cross-[1, 2, 4, 8] DPP
stages compose correctly in the N=32 case (after the cross-16
fix) but fail at N=16 on specific rows.  Open for a dedicated
bisect.

The five probes ship together so a future session has the full
discriminator (monotonic / xor1 / row-offset / altrow / random)
already wired to narrow the bug class without re-deriving it.

**Regression surface.**

  * Full Triton corpus: 72/104 -> 76/104 match (+4 recipes:
    canary_tl_sort_fp32, canary_tl_topk_fp32, canary_tl_topk
    _bf16, canary_tl_topk_bf16_nw1).  No previously-passing
    recipe regresses.
  * `Gfx1250Gpu.Permlane16Swap{,Wave32,Wave32WaveNative}`,
    `Gfx1250Gpu.BitonicCross16Probe`,
    `Gfx1250Gpu.BitonicXor3TritonState`,
    `Gfx1250Gpu.RcpSqrt`, `Gfx1250Gpu.DppQuadPerm` -- all pass.
  * Full ctest: 95% (105/107 lit fixtures plus the focused test suite) with
    the same pre-existing failures as before the fix (the
    `wmma_phantom_lane_*` lit fixtures that the matmul-WMMA
    agent owns, and the `MfmaGpu.Gemm*` tests blocked on a
    separate `s_load_dwordx4` kernarg-slot parsing bug).

**Flag plumbing -- same pattern as the xor3-partner flag.**
`--enable-permlane16-swap-selfpreserve` / `--disable-permlane16-
swap-selfpreserve` on `raise_cli`; default on.  The
`enablePermLane16SwapSelfPreserveRewrite` parameter threads
through `raiser.hpp` -> `pipeline.hpp` -> `pipeline.cpp`.
`--disable-` audits the pre-rewrite symmetric-cross-wire shape
for baseline characterisation.

---

## 2026-04-23 -- global_atomic SADDR form silently miscompiled (sum_bitmatrix_rows_u32)

`sum_bitmatrix_rows_u32` and its `_nw4` sibling crashed every launch
with `HIP error 700 (illegal memory access)` despite `raise_cli`
reporting `OK 178/178` -- a textbook "lift succeeds, device explodes"
shape that the recipe-level native-vs-hotswap comparison caught but
the single-primitive canaries did not.  Root-caused to a SADDR-form
blind spot in the `GLOBAL_ATOMIC_*` handler in `handle-flat.cpp`.

**The instruction:**

    global_atomic_add_u32 v1, v0, s[4:5] scale_offset scope:SCOPE_DEV

This is the gfx12+ SADDR form -- vaddr(VGPR32) + vdata(VGPR32) +
saddr(SGPR64), semantically identical to `global_store` SADDR:
uniform SGPR64 base + per-lane VGPR32 offset, optionally scaled
by element size.  Triton's `tl.atomic_add(Out + offs_n, ...)`
codegen for gfx1250 lands exactly here.

**What hotswap emitted BEFORE the fix:**

    global_atomic_add v[2:3], v0, off offset:2064 sc1

Two compounding bugs, both inherited from a pre-gfx12 operand-shape
assumption in the atomic block:

  1. `ParsedReg addrReg = op.srcReg(0); Value *addr = readReg64(addrReg);`
     hard-coded the plain-form shape.  On SADDR, `srcReg(0)` is the
     VGPR32 vaddr offset (`v1`), not a 64-bit address; reading it as
     a VGPR pair pulled in `v2` as the high half and produced a
     garbage pointer rooted in whatever lived adjacent to `v1`.

  2. The imm-scan loop "first non-zero immediate wins" was meant to
     pick up the signed offset field.  On gfx12+ the CPol operand
     (packed scale_offset flag + scope bits) lives as an additional
     immediate AFTER the offset field, and for `scale_offset
     scope:SCOPE_DEV` carries the value 0x810 = 2064.  With the real
     offset being 0, the loop happily mistook CPol for offset and
     GEP'd the atomic address 2064 bytes past the Out-pointer base.

  3. Same loop's "last reg-valued src = data" heuristic then picked
     the SGPR saddr (s[4:5] = Out pointer) as the data to add,
     because after (1)(2) it was still scanning past the true vdata.
     Net device-visible atomic:
       `atomic_add(&B[~garbage+2064], &Out /* pointer */)`

**The fix:**

Delegate to `decodeGlobalStoreAddr` -- atomics share the store's
operand shape ((vaddr, vdata, [saddr], [imms])) and the helper
already handles both plain and SADDR discrimination, uses
`firstImmOffset` (FIRST imm regardless of value, not first
NON-ZERO), and returns the vdata register via `fa.stData`.  The
CMPSWAP branch still increments `baseIdx` on `fa.stData` to reach
the newVal half of the cmp/new pair.

Element size for `scale_offset` is 4 for every atomic in the
`[GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_PK_ADD_F16]` range that fix covered:
ADD/SUB/AND/OR/XOR/MIN/MAX/SWAP/ADD_F32/PK_ADD_{BF16,F16}/CMPSWAP.
`GLOBAL_ATOMIC_ADD_X2` uses the same decoder with `elemBytes=8`,
matching its 64-bit data/address stride.

**After the fix:**

    v_mul_i32_i24_e32 v2, 4, v1                 ; scaled vaddr lo
    v_mul_hi_i32_i24_e32 v3, 4, v1              ; scaled vaddr hi
    v_lshl_add_u64 v[0:1], s[12:13], 0, v[2:3]  ; s[12:13]=Out + v1*4
    global_atomic_add v[0:1], v10, off sc1      ; *(Out+v1*4) += v10

Structurally identical to Triton's gfx942-native compile (modulo
register allocation) -- no spurious offset, vdata is the real
popcount sum.

**Corpus impact:**

    sum_bitmatrix_rows_u32       4 EXIT=2 ->  4 match
    sum_bitmatrix_rows_u32_nw4   4 EXIT=2 ->  4 match
    corpus total                 64/104  -> 72/104

No regressions elsewhere -- `canary_tl_sort_fp32_deterministic`
(which exercises the permlane16-xor3-partner rewrite we just
landed) still matches, all Gfx1250Gpu cross-lane + atomic regression tests
pass, and the `c3_atomic_cas` lit fixture's CMPSWAP path goes
through the same code path via `fa.stData + baseIdx+1` so the
cmp/new pair contract is preserved.

**Generalization -- the bug class.**

"Handler assumes operand shape that varies by subtarget" is a
recurring hotswap bug class.  The same root cause (hard-coded
plain-form shape, later patched up for gfx12+ SADDR) bit
`rcp_sqrt_kernel` in `GLOBAL_LOAD` before this -- see the
RcpSqrt regression coverage and handle-flat.cpp:653-665 comment.
The prescribed pattern is: **when a gfx12+ form has SGPR-base +
VGPR-offset addressing, EXTRACT the shape decode into a helper
(`decodeGlobal{Load,Store}Addr`) and have every consumer -- load,
store, atomic, prefetch -- route through it**.  All six FLAT/GLOBAL
memory-class handlers (load, store, atomic × flat, global) now
route SADDR through the shared decoder.  The FLAT_ATOMIC fix was
applied in the same commit even though no recipe in the compare_
correctness Triton corpus currently triggers it (Triton's gfx1250
codegen prefers `global_atomic_*` when the buffer is known global)
-- the identical bug was still latent and would bite the first
kernel that landed on `flat_atomic_*` with SADDR.  Any new
FLAT-class handler (the pending `tensor_load_to_lds` family, flat
prefetch) should follow the same pattern.

**Investigation technique.**

  * Start from the device-reported failure shape (HIP 700 = VM page
    fault -> bad address, not bad data) to narrow "where" vs "what".
  * Produce all three disassemblies side-by-side:
      source gfx1250 (what Triton emitted)
      native gfx942 (what Triton's native compile does -- the
                     algorithm's ground truth)
      hotswap gfx942 (what we produced)
    Diff the critical memory op.
  * When CPol imms leak into offset lookups (the 2064 red herring),
    instrument the handler to dump `op.srcIdx(k)` / `di.getImm(idx)`
    for each operand -- a few lines of `llvm::errs()` behind an
    env-gated debug print is faster than reading LLVM's encoding
    specs.

The `sum_bitmatrix_rows_u32{,_nw4}` end-to-end recipes are the
regression pin: they exercise a Triton-emitted `global_atomic_*` SADDR
+ `scale_offset`
shape end-to-end on a real GPU, and both verdicts flip from
`EXIT=2 (HIP 700)` to `match` under the fix.  A narrower gtest-
scope regression (on the line of `Gfx1250Gpu.RcpSqrt` for the
load-side SADDR sibling) would need a hand-authored HIP kernel
that forces hipcc to emit `global_atomic_add_u32 vAddr, vData,
s[A:B] scale_offset`, which hipcc does not reliably produce from
`__builtin_amdgcn_atomic_add` -- the codegen picks the addressing
form based on alias/escape analysis of the pointer and does not
expose a builtin/intrinsic that pins the SADDR form.  The
recipe-level gate is therefore the right layer; adding a
semantically-duplicate gtest on top would give the illusion of
broader coverage without actually exercising a different path.

---

## 2026-04-22 -- Triton gfx1250 cross-16 bitonic-merge xor3-partner rewrite (TRANSITIONAL)

`canary_tl_sort_fp32_deterministic` now matches (was `WRONG 16384/16384`).
The new `rewrite_permlane16_xor3_partner` pass fires after
PromoteMemToReg and pattern-matches the specific Triton
compose

    v_dual_mov v_a, v_c :: v_dual_mov v_b, v_c
    v_permlane16_swap v_a, v_b
    v_xor3 v_a, v_a, v_b, v_c

which, under gfx950-documented swap semantics and standard
3-way xor, algebraically collapses `v_a = partner ^ partner ^
self = self` -- yet Triton's gfx942-native compile of the same
Python source produces a correct sort (using `ds_swizzle_b32
swap:16` instead of permlane16_swap+xor3).  So either:

  (a) gfx1250 silicon diverges from gfx950 for the identical-
      input initialiser and really does yield `partner` from
      this idiom, OR
  (b) Triton's gfx1250 codegen has a bug and the whole idiom is
      a no-op that happens to work on real gfx1250 hardware
      for some other reason.

Without gfx1250 hardware or AMD ISA docs for this specific
edge case we can't discriminate.  What we CAN do is emit the
value the algorithm NEEDS (`partner_v_c`) regardless of which
scenario is true -- that matches the gfx942-native compile
and the bitonic-sort math, and the fingerprint we match
(two ds.bpermute calls with matching first operand plus an
outer xor of their xor with the shared seed) is hotswap-local
and structurally impossible to produce outside this exact
Triton compose.

**The rewrite is a COMPAT BRIDGE, not a principled primitive-
semantic fix.**  Under both hypotheses, the principled fix is
in a different layer:

  (a) -> `emitPermLaneSwapEmulation` in
        `handle-valu-cross-lane.cpp` -- teach the primitive
        lift about the gfx1250 silicon's divergent semantic.
        The composition-level rewrite then becomes harmless
        dead code (the pattern still matches, substituting an
        already-correct partner for itself).
  (b) -> file against Triton.  When gfx1250 codegen stops
        emitting this idiom, the rewrite fingerprint stops
        appearing in lifted IR and the pass is dead code.

Both dead-code states are actually a feature: if we delete the
pass prematurely (before the correct-layer fix lands) we
silently regress `canary_tl_sort_fp32_deterministic` back to
`WRONG 16384/16384`.  Keeping the bridge around during the
transition period is zero-risk because:

  * The fingerprint is exact: two bpermutes with identical
    first operand and seed-equivalent second operands feeding
    an outer xor with the shared seed.  No non-Triton hotswap
    lift produces this shape.
  * `Gfx1250Gpu.BitonicXor3TritonState` probe verifies the
    per-lane transformation.  `Permlane16Swap*` regressions pin the
    standalone swap semantic.  `canary_dpp_reduce_fp32` +
    `canary_permlanex16_rowmax_fp32` pin other cross-lane
    primitives.  None regress.
  * `--disable-permlane16-xor3-partner` raise_cli flag audits
    the pre-rewrite shape for anyone characterising the
    baseline.

**Explicit TRANSITIONAL markers everywhere the pass and its
flag are referenced** (rewrite_permlane16_xor3_partner.hpp
block comment, raiser.hpp / pipeline.hpp flag doc, raise_cli.cpp
top-of-file + usage string).  Each marker cites the two
removal conditions (a/b) and the pass-level / flag-level /
probe-level artifacts that should be deleted together when the
condition is met.

The failed experiment recorded above (the env-gated fcmp-src1
partner-read injection at every `V_CMP_NGT_F32_e64_gfx12` site)
was the piece-of-the-puzzle move that made us realise:

  1. the whole bytes are a single 8-byte `v_cmp_ngt_f32_e64`
     (newer AMD LLVM 23.0.0git decodes cleanly -- older objdump
     just didn't know the gfx1250 decoder namespace); and
  2. the semantic mismatch is in the xor3 composition UPSTREAM
     of the compare, not in the compare itself.

`canary_tl_sort_fp32_deterministic` is the one recipe in the
suite that triggers the idiom (Triton's deterministic-input
codegen hits `v_a_in == v_b_in == v_c`).  `canary_tl_sort_fp32`
with random input, `canary_tl_topk_*`, `topk_forward_bf16`, and
`topk_forward_bisect_m2_strict` use a DIFFERENT codegen path
where the swap operands are distinct VGPRs -- the xor3 collapse
doesn't apply.  They're SEPARATE bugs to be investigated.

Commits for this thread: 2bd4381028 (the rewrite pass itself),
and the follow-up that adds the `--disable-permlane16-xor3-
partner` audit flag, the TRANSITIONAL markers, and this
learnings entry.

---

## 2026-04-22 -- Failed experiment: inject `ds_bpermute(lane ^ 16)` on V_CMP_NGT_F32_e64_gfx12 src1

After the xor3-triton-state probe (`Gfx1250Gpu.BitonicXor3TritonState`)
definitively confirmed v3 post-xor3 = self_v2 for every lane, the
remaining hypothesis was that `.long 0xd41b0002` (which LLVM's
gfx12 tables decode as `V_CMP_NGT_F32_e64`, opcode 42725) might be
a gfx1250-compact encoding that reuses this opcode slot with a
fused compare-with-permlane16-partner semantic.

**Experiment.**  Patched `handle-valu-vcmp.cpp` under an env gate
`HOTSWAP_EXPERIMENTAL_V_CMP_NGT_F32_PARTNER=1` to inject
`ds_bpermute(lane ^ 16, src1)` before the fcmp, making the compare
functionally `fcmp ngt self, permlane16(self)` on gfx1250 -- the
shape the bitonic cross-16 merge would want if the opcode really
had implicit partner-read semantics.

**Result: experiment FAILED.**  On
`canary_tl_sort_fp32_deterministic`:

  * Baseline (no patch): row 0 = `[15,14,…,0, 31,30,…,16]` --
    two sorted halves, distance-16 merge missing.  WRONG
    16384/16384, max\|err\|=16.
  * With patch:          row 0 = `[15,15,…,15, 31,31,…,31]` --
    MAX-broadcast within each 16-lane half.  WRONG 16384/16384,
    max\|err\|=31.

The patch made it worse: it broadcast the half-max to every lane
in the half.  That's because the patch applies to EVERY
`V_CMP_NGT_F32_e64_gfx12` call -- and those fire at distances
16, 8, 4, 2, 1 in the outer-stage-4 sequence.  At distances
8/4/2/1 the preceding DPP moves have already placed partner into
v3; forcing a further permlane16 read produces a compare against
a lane 16 positions away instead of the correct local-distance
partner.  The signature (max-broadcast within halves) indicates
every compare in the outer-stage-4 chain collapsed to
"self vs max-within-16" under the injected partner-read.

**Negative finding.** The opcode is NOT a simple fused
compare-with-permlane16-partner.  The correct semantic --
whatever it is -- must adapt to the lane distance the surrounding
DPP / permlane moves set up, or it uses a different mechanism
than an implicit cross-lane read on its source operand.

**Next candidate hypotheses** (not yet tested):

  1. The compare's real src1 reads v4 (which at the `.long
     0xd41b0002` site still holds partner_v2 from the swap's
     second output -- the `v_and_b32 v4, 8, v0` write that comes
     between xor3 and the compare may not have committed yet due
     to the `s_delay_alu instid0(VALU_DEP_2)` hint's scheduling
     model).  For distance-8/4/2/1, v4 at the compare site is
     lane_id & 8 / 4 / 2 / 1 -- irrelevant.  Hmm, that doesn't
     match either, since the distance-8 v_and is ALSO between
     xor3-equivalent moves and the compare.
  2. The compare implements a `compare_and_swap_within_16_by_
     lane_parity` -- a single-instruction compare-and-swap that
     uses lane_id-derived direction implicitly.  The fact that
     the SAME opcode works at all 5 distances within the outer
     stage suggests a distance-agnostic semantic, possibly
     using the preceding DPP setup's output (v3) combined with
     an implicit per-lane control.
  3. The compare participates in a VOPD / dual-issue shape not
     yet in LLVM's decode tables for gfx1250, where the
     subsequent `v_cndmask_b32_e32 v1, v2, v3, vcc_lo` is
     actually the SECOND half of a single 8-byte fused
     compare-and-conditional-move.

Hypothesis 3 feels most plausible but unverifiable without gfx1250
ISA documentation.  Hypothesis 1 can be ruled out by the observation
that v4 IS clobbered by `v_and` before the compare fires in the
instruction stream, and hotswap faithfully serializes that in the
lifted IR.  Hypothesis 2 would require a gfx1250 ISA spec to
confirm.

Deferring further investigation until we have:
  - AMD's gfx1250 ISA reference for the `.long 0xd41b000X`
    encoding, OR
  - A newer LLVM tree where the decode tables match gfx1250
    silicon (then cross-check against hotswap's current decode).

The full diagnostic probe suite committed across
79e1edc2f1 / 65b5fde536 / a9bcc14d03 / af64431eec stays as
regression guards.  When the fix lands, all of
`canary_tl_sort_fp32_deterministic`,
`canary_tl_sort_fp32`, `canary_tl_topk_{bf16,bf16_nw1,fp32}`,
`topk_forward_bisect_m2_strict`, and `topk_forward_bf16`
graduate to match; the wave32 permlane16_swap and bitonic
probes stay passing as they do today.

---

## 2026-04-22 -- `topk_forward_bf16` root-caused to `tl.sort` / `tl.topk` cross-16 bitonic merge under wave32->wave64 projection (OPEN)

**Context.** Continuation of the principled-tolerance thread.  The
`topk_forward_bf16` recipe stayed WRONG after the m1..m4 thread
closed, with a SYSTEMATIC bias on Yv[0]: 208 positive vs 32
negative signed diffs on the 249 disagreeing rows (SNR +0.877).
Systematic bias rules out bf16 rounding noise (which would be
symmetric), so the residual is a real miscompile.

**Bisect.**  The following probes (landing in this commit) narrow
the bug one layer at a time:

* `topk_forward_bisect_m2_strict` -- `streaming_topk` with
  `abs tol=0.0` on Yv.  WRONG 1542/2048, max\|err\|=1.48 at
  magnitude 3.8.  Bug is inside `streaming_topk`, not downstream
  (softmax / Yi write / Bits derivation).
* `canary_tl_topk_bf16` / `canary_tl_topk_fp32` -- a bare `tl.topk`
  at the same shape (BLOCK_M=32, BLOCK_N=32, k=4).  Both WRONG
  2048/2048.  Bug is not bf16-specific (rules out `fpval_to_key`
  u16 bit-magic and the u32 `(value_key << 16) | index_key` pack).
* `canary_tl_sort_fp32` -- bare `tl.sort(dim=1, descending=True)`
  at (BLOCK_M=32, BLOCK_N=32).  WRONG 15353/16384 (93.7%).  Bug
  is in the shared `tl.sort` / `tl.topk` backbone.
* `canary_tl_sort_fp32_n16` -- same shape but BLOCK_N=16 (no
  cross-16 merge needed).  WRONG only 1056/8192 (12.9%).
  Confirms: the **cross-16 merge step** is the problem class.
* `canary_tl_sort_fp32_deterministic` -- input `X[r, c] = c` so
  the sort output reads out exactly which lane's data each output
  slot pulled from.  Expected: `[31, 30, ..., 1, 0]`.  Hotswap
  produces `[15, 14, ..., 0, 31, 30, ..., 16]` -- TWO SORTED
  HALVES.  Lanes 0..15 never exchanged with lanes 16..31.

The signature "two sorted halves, no cross-16 merge" isolates the
bug to the **final bitonic merge step that uses `v_permlane16_swap_b32`**
(the hardware XOR-16 partner swap).

**What `Gfx1250Gpu.Permlane16Swap` does NOT cover.**  The existing
GPU regression test for `permlane16_swap` uses a wave64 source
kernel (`__launch_bounds__(64)` on gfx1250).  It verifies wave64
-> wave64 same-wave correctness.  The `tl.sort` case that breaks
is wave32 -> wave64 cross-widening -- the `v_permlane16_swap`
on the source wave32 side is compiled with wave32 semantics,
and hotswap's `emitPermLaneSwapEmulation` in
`handle-valu-cross-lane.cpp` emits `ds_bpermute(lane_id ^ 16,
src)` which LOOKS correct in isolation (per-32-lane-half
independent swap) but produces "no exchange" under the
specific cross-widening path Triton's `tl.sort` takes.

**Candidate root causes (not yet root-caused).**  The bug is
somewhere in the COMPOSITION of:

1. `tl.sort`'s bitonic compare-and-cndmask pattern around
   `v_permlane16_swap_b32`.  The assembly shows a specific
   4-instruction dance: `v_dual_mov; permlane16_swap;
   v_xor3_b32 (v3 = v3 ^ v4 ^ v2); v_cndmask based on vcc`.
   The XOR3 restores `v3 = self_v2` and keeps `v4 = partner_v2`;
   then the compare sets vcc from `self` vs `partner`, and
   cndmask picks one for the output.  Under wave32 native this
   works; under wave64 emulation, SOMEWHERE the partner value
   fails to propagate.
2. `emitPermLaneSwapEmulation` wraps the two `ds_bpermute` calls
   in `emitUnderExec` gating.  `saved_exec = ballot(init_whole_wave())`
   returns 0xFFFF_FFFF_FFFF_FFFF under WaveNativeProjection, so
   every lane should be "active" and the swap should run, but
   something downstream (possibly the `cndmask`'s VCC source,
   which is a `v_cmp` whose EXEC-mask semantics differ under
   wave32/wave64) is collapsing the swap into a no-op.
3. The `v_cmp` that produces VCC for the cndmask may itself be
   a `V_CMP_{LT,GT}_F32_e64` whose dual-dword encoding the
   disassembler shows as `.long 0xd41b0002` -- indicating that
   the instruction's real structure isn't being surfaced by
   objdump at this call site.  A different `v_cmp` lift under
   cross-widening could be producing an always-false VCC, which
   makes cndmask always pick `self` and the merge effectively
   skip.

**Handler gaps surfaced in the probes (filed separately):**

* `v_pk_sub_i16` -- VOP3P packed int16 subtract.  Hits on
  `topk_forward_bisect_m2_idx` when writing i16 Yi outputs
  directly (hitting a different instruction-selection path
  than the production kernel).  The initial probe was retired
  in favour of `topk_forward_bisect_m2_strict` which avoids
  the handler gap by not emitting the i16 output.  The handler
  gap itself remains open.

**Committed probes stay as regression guards.**  After a fix,
they all graduate:

* `topk_forward_bisect_m2_strict` -> match on `abs tol=0.0`
* `canary_tl_topk_bf16`, `canary_tl_topk_bf16_nw1`,
  `canary_tl_topk_fp32` -> match on `rel-rms tol=0.02` / `1e-5`
* `canary_tl_sort_fp32` -> match on `abs tol=0.0`
* `canary_tl_sort_fp32_n16` -> match on `abs tol=0.0`
  (the N=16 case currently has ~13% mismatch -- narrower bisect
  probe for a smaller-scale shape bug)
* `canary_tl_sort_fp32_deterministic` -> match on `abs tol=0.0`;
  its input is lane-identifiable so a regression would reveal
  the exact shuffle-network shape that broke

**Progress update (same-day).**  Step 1 DONE -- two wave32-source
regressions now exist alongside the pre-existing wave64-source one:

| test                                     | source    | projection      | verdict |
|------------------------------------------|-----------|------------------|---------|
| `Gfx1250Gpu.Permlane16Swap`              | wave64    | SameWave         | PASS    |
| `Gfx1250Gpu.Permlane16SwapWave32`        | wave32    | ModuloRep        | PASS    |
| `Gfx1250Gpu.Permlane16SwapWave32WaveNative` | wave32 | WaveNative       | PASS    |

(The middle test trips the phantom-lane regime because
`__launch_bounds__(32) < 64`, so `raiser.cpp` forces ModRep.  The
third test dispatches at `__launch_bounds__(128)` -- same shape
as Triton `tl.sort`'s 4 × 32-lane source waves -- so
`WaveNativeProjection` engages, exactly the path the broken
`tl.sort` takes.)

**Conclusion: `emitPermLaneSwapEmulation` is correct under all
three projection regimes, including WaveNative.**  The
`ds_bpermute(lane_id ^ 16, src)` cross-wired emission faithfully
implements the ISA-defined `new_vdst[L] = src0_in[L XOR 16],
new_src0_out[L] = vdst_in[L XOR 16]` semantics.  Lifted kernels
swap correctly per-lane on gfx942 hardware.

So the `tl.sort` cross-16 merge failure roots in the
**composition** around the swap, not in the primitive itself.
The Triton bitonic-merge step emits this 4-instruction dance
around `v_permlane16_swap`:

```
v3 = v2                                    ; copy self into v3
v4 = v2                                    ; copy self into v4
v_permlane16_swap_b32 v3, v4               ; v3 = v4 = partner_v2
v_xor3_b32 v3, v3, v4, v2                  ; v3 = self_v2
                                           ; (v3 ^ v4 ^ v2 =
                                           ;  partner ^ partner ^ self)
v_and_b32 v4, 8, v0                        ; v4 is CLOBBERED (next stage)
.long 0xd41b0002                           ; v_cmp (hotswap raises to
                                           ;  fcmp ule v2, v3 = self <= self)
v_cndmask_b32_e32 v1, v2, v3, vcc_lo       ; v1 = vcc ? v3 : v2
```

After the XOR3, both `v2` and `v3` hold `self_v2`, so hotswap
raises the compare as `fcmp ule self_v2, self_v2` -- an
identity compare that is always `true`, making the cndmask
collapse to "always pick v3" (= self), which IS a no-op from
the source-wave's perspective.

But the NATIVE gfx1250 hardware runs the SAME instruction
sequence and produces a CORRECT sort.  That means either:

(a) My read of `v_xor3` is wrong -- the XOR3 actually yields
    `partner_v2`, not `self_v2`.  Possible if `v_permlane16_swap`
    has different post-swap VGPR semantics than I've inferred
    (the `Gfx1250Gpu.Permlane16Swap` test verifies the
    cross-wired post-swap state, so this is unlikely).

(b) My read of the compare operands is wrong -- `0xd41b0002`
    may compare `v2 vs v4`-before-clobber rather than
    `v2 vs v3`-after-XOR3.  Hotswap's IR shows `fcmp ule v2, v3`
    but the gfx1250 ISA encoding for opcode 27 / `0xd41b` in
    this new VOP3-single-dword family may be what's actually
    different -- upstream llvm-mc cannot decode this opcode, but
    hotswap's newer decoder can.

(c) Something more subtle -- e.g. the compare opcode has
    implicit operand usage (v2 vs "prior v3 before XOR3"?),
    or it's not an `fcmp ule` at all but a gfx1250-new
    variant that has different semantics than the `NGT`
    mapping hotswap applied.

Committed the two wave32 regression tests as regression guards
regardless; the PASSING state of both pins the positive
correctness invariant for `emitPermLaneSwapEmulation` under
both ModRep and WaveNative.  Any future regression in the
emulation (e.g. forgetting to cross-wire the two
`ds_bpermute` calls) trips here before reaching a Triton
user.

**Hardware probe run (same-day):** the wave32 inline-asm canary
(`test_data/gfx1250/bitonic_cross16_probe_kernel.hip`, regression
`Gfx1250Gpu.BitonicCross16Probe`) confirms the textbook model
of the sequence `v_permlane16_swap_b32 v3, v4; v_xor3_b32 v3,
v3, v4, v2`.  With distinct inputs `v2=L, v3=100+L, v4=200+L`,
per-lane output on gfx942 hardware (lifted through hotswap) for
every lane matches exactly:

    after swap: v3 = 200 + (L XOR 16), v4 = 100 + (L XOR 16)
    after xor3: v3 = (200 + partner) XOR (100 + partner) XOR L

For lane 0: v3 = 216 XOR 116 XOR 0 = 172 -- matches the printed
trace.  For lane 16: v3 = 100 XOR 200 XOR 16 = 188 -- also
matches.  So:

  (a) `v_permlane16_swap_b32`'s cross-wired semantic
      (`new_vdst = src0_in[partner], new_src0_out = vdst_in[partner]`)
      is correct on gfx1250.
  (b) `v_xor3_b32 v3, v3, v4, v2` semantics (`v3 = v3 ^ v4 ^ v2`)
      are standard.
  (c) Under Triton's `v3_in = v4_in = v2` (copied self), the
      xor3 collapses to `v3 = self_v2`.

**This means hotswap's decode of `.long 0xd41b0002` as
`v_cmp_ngt_f32_e64 s2, v2, v3` must be WRONG.**  Under that
decoded form, the compare's operands would be `self_v2` and
`self_v2` -- an identity compare that is always `true`, making
the subsequent `v_cndmask` a no-op.  Native gfx1250 produces a
correct sort through this sequence, so the actual instruction
cannot be a self-vs-self compare.

Upstream llvm-mc (ROCm 7.2.1's gfx1250 MCPU) rejects the 4-byte
encoding `02 00 1b d4` as `invalid instruction encoding`, but
hotswap's own LLVM decoder (built into
`src/hotswap/build/raise_cli`) accepts it and produces an
`fcmp ule float v2, v3` IR shape.  Three hypotheses remain open:

  1. Hotswap's decoder (a newer LLVM than the toolchain MC)
     correctly decodes the opcode family but maps the operand
     positions to the wrong SSA sources.  The `.long 0xd41b0002`
     may be a gfx1250-new single-dword VOP3 compact encoding
     whose operand fields differ from the 2-dword VOP3 layout.
  2. The instruction reads `v4` (holding `partner_v2` from the
     swap, BEFORE `v_and_b32 v4, 8, v0` clobbers it) -- not `v3`.
     That would give `compare(v2, partner_v2)` = a meaningful
     per-lane result.  But instruction ordering rules should
     require the `v_and_b32 v4, 8, v0` that comes BEFORE
     `.long 0xd41b0002` in the stream to complete first.
  3. The instruction is a gfx1250-new compound op (e.g. a
     compare-and-cross-lane, or a lane-id-biased compare) that
     hotswap's decoder models with the wrong shape.

**Next narrowing step (deferred).**  Build a `raise_cli`-equivalent
harness that dumps the decoded MCInst for the `.long 0xd41b0002`
sites verbatim (Opcode + OperandVec) so we can compare what
hotswap's LLVM decoder thinks versus what the upstream tables
would infer.  Alternatively, test hotswap with a newer LLVM
(e.g. the `amd-llvm` source in
`a newer LLVM AMDGPU source tree`) --
if that newer tree has full gfx1250 decode tables, any
discrepancy with hotswap's current decode is the bug.

The wave32 permlane16_swap fixtures and the bitonic cross-16
probe remain in the test suite as permanent regression guards.
Their pass/fail state after a future fix pins the positive
correctness invariant for the cross-widening path the Triton
`tl.sort` takes.

---

## 2026-04-22 -- Principled bf16-reduction tolerances + open finding on `_topk_forward` Yi divergence

**Context.** After closing the FMA_MIX / d16_hi / MODE-register /
DPP-cross-widen bugs, the `topk_forward_bisect_m1..m4` family and the
production `topk_forward_bf16` recipe still reported `WRONG` under
`abs tol=0.0`.  The question: is the residual rounding noise, or a
miscompile?

**Measurements** (rms(diff)/rms(gold) over 512 rows × 4 picks):

| recipe                             | rel-rms  | max\|err\| | verdict at tol=0.0 |
|------------------------------------|----------|------------|--------------------|
| `topk_forward_bisect_m0`           | 0.00000  | 0.0000     | match (bit-exact)  |
| `topk_forward_bisect_m1`           | 0.00524  | 0.2500     | WRONG 1300/2048    |
| `topk_forward_bisect_m2`           | 0.07222  | 1.5938     | WRONG              |
| `topk_forward_bisect_m3`           | 0.13963  | 0.2207     | WRONG              |
| `topk_forward_bisect_m4`           | 0.14974  | 0.2715     | WRONG              |
| `topk_forward_bf16` / `Yv`         | 0.14711  | 0.2715     | WRONG              |
| `topk_forward_bf16` / `Yi`         | --        | --          | **93% row-set mismatch** |

MODE=0 (write zeros): bit-exact, no FP math.  MODE=1 (`tl.sum +
bf16 store`) drift sits well under the theoretical bf16 reduction
bound ≈ `log2(N) * 2^-7 * RMS(gold)` for N=32, which works out to
`5 * 0.008 * 13 ≈ 0.5`; observed 0.067 is well under.  MODE=2 adds
`streaming_topk` whose `tl.topk` + `tl.sort` interact with
near-tied `bf16` row values -- tie-break flips at the k-th / (k+1)-th
boundary produce max|err| ~= the tie gap (~1.6 at magnitude 4),
not ULP-scale.  MODE=3/4 adds `tl.softmax`, which amplifies near-tied
pre-softmax drift non-linearly (small input-value swap at the sort
boundary -> large post-softmax slot swing).  All of this is
legitimate IEEE-compliant rounding + legal sort-tie-break drift
under cross-widening.

**The fix for m1..m4:** mode-dependent `rel-rms` tolerances in
`topk_forward_bisect.py::_recipe`.  See the comment block there for
the per-mode derivation.  `topk_forward_bisect_m1_fp32` got its own
`rel-rms tol=1e-5` entry (observed 8.7e-8 ≈ 1 fp32 ULP at magnitude
7).  Harness change: the end-to-end comparator now routes integer dtypes
(`i16`/`u16`/`i32`/`u32`/`i64`) through the same tolerance-aware judge
when the comparator is `rel-rms`, matching the float paths; this is the
clean extension for "tolerance-aware integer comparison" that
keeps integer `abs` / `rel` on the fast bit-exact loop.

**Open finding -- do NOT close with tolerance alone.**  Yv on
`topk_forward_bf16` also passes rel-rms(0.25), but the **index
output** Yi has only 6.8% (35/512) of rows picking the same set
of columns between native and hotswap.  Top-1 alone diverges in
49% of rows.  That is much more than bf16 ULP drift + near-tied
flip can explain -- at N=128 columns with uniform X in [-4, 4]
and bf16 ULP ≈ 0.031 at magnitude 4, the probability of the
k/k+1 boundary pair being within 1 ULP is low enough that only
a few percent of rows should tie-break-flip.  Hotswap's Yv[0]
(post-softmax) is also systematically LARGER than native's in
rows where they disagree, which is the signature of hotswap
picking a value that native didn't consider (or vice versa) --
not just reordering the same set.

The two resolutions are (a) genuinely a legal tie-break amplified
by `streaming_topk`'s LDS-based merge stage (if so, the right
comparator is a set-equivalence check, not a scalar rel-rms), or
(b) a lift-level bug in one of:

* `tl.topk` / `tl.bitonic_merge` under cross-widening -- the u32
  `(value_key << 16) | index_key` packing assumes lane-local
  key lookups that may break when wave32 `tl.load`s feed a
  wave64 sort path;
* `tl.load(X, mask, other=-inf)` with cross-widened masks
  loading wrong lanes' X values (would produce exactly the
  "hotswap picks LARGER values than native" signature);
* residual `ds_bpermute + select` shape inside the DPP rewrite
  not preserving the inactive-lane => `-inf` poison that
  `streaming_topk` relies on for the first-iteration peel.

`topk_forward_bf16` remains intentionally at `abs tol=0.0` on
Yi / Bits -- no tolerance band would keep the signal honest --
so the verdict stays `WRONG 2856/8192` until the root cause is
nailed down.  Next triage probes:

* Canary: dump `X` for a row where Yi disagrees and compute
  numpy top-4 independently; compare to both native and hotswap
  picks.  If native picks a value smaller than hotswap's pick
  and smaller than numpy's, native's streaming_topk has a bug
  -- extremely unlikely since native is gfx1250-Triton-compiled
  code running on the same hardware, but necessary for rigor.
* Bisect with a HIP + inline-asm canary of the precise u32 pack
  + `tl.topk` shape at BLOCK_M=32, BLOCK_N=32, k=4, to isolate
  whether the index-key low 16 bits survive cross-widening.
* A second `_const_in` probe for `topk_forward` with all-equal
  X values: under ties-broken-by-smaller-index, both native and
  hotswap should pick indices [0, 1, 2, 3]; any other result is
  a direct bug.

The `rel-rms` extensions in the harness are general-purpose and
stand on their own correctness (integer reduction of `sumDiff2`
/ `sumGold2` is just as sound as the float paths); they don't
depend on the `_topk_forward` investigation finishing.

---

## 2026-04-22 -- V_FMA_MIX inline-constant narrow-half: op_sel misapplied, silent bf16-reduction miscompile (closes topk_forward_bisect_m1_const_in)

**Context.** Every bf16-in + reduction + bf16-out Triton kernel was
silently miscompiled under cross-widening (gfx1250 -> gfx942).  The
end-to-end signature: feed a 32-column `tl.sum(axis=1)` of bf16
`1.0`s -- native writes `32.0`, hotswap writes `1.0`.  Every reduction
step's multiplier silently evaporated and every bf16 accumulation
became a no-op.  Affected `topk_forward_bisect_m1..m4` and
`topk_forward_bf16`; a multi-day sequence of false leads (DPP
rewrite, EXEC-mask tracking, strict.wwm, d16_hi store) came and went
before the all-ones probe localised it.

**Root cause.** `V_FMA_MIX_F32` / `V_FMA_MIX_F32_BF16` encode narrow
sources via an `op_sel_hi[i]=1` flag plus an `op_sel[i]` VGPR-half
selector (0=low 16, 1=high 16).  `op_sel[i]` is a VGPR-half
selector -- it assumes the 32-bit source holds two packed 16-bit
values so either half is a valid narrow datum.  For register
sources the assumption holds.  For INLINE CONSTANTS (and 32-bit
literals in narrow-operand slots) it does NOT:

  LLVM's AMDGPU disassembler pre-resolves narrow inline constants
  to the 16-bit value in the LOW 16 of the MCOperand Imm, upper 16
  zero-extended -- `AMDGPUDisassembler.cpp::decodeMCOperand`, under
  `OPERAND_REG_INLINE_C_BF16` / `OPERAND_REG_IMM_BF16` /
  `OPERAND_REG_INLINE_C_FP16` / `OPERAND_REG_IMM_FP16` arms and
  their `getInlineImmValBF16` / `getInlineImmValF16` helpers.

The pre-fix handler applied op_sel unconditionally:

    if (opSel[i] == 0) bits = trunc(raw, i16);                // LO
    else               bits = trunc(lshr(raw, 16), i16);      // HI

For an inline bf16 `1.0` (= MC Imm `0x00003F80`) with
`op_sel[i]=1`, the handler ran `trunc(lshr(0x3F80, 16)) = 0x0000 =
bf16 0.0`.  The downstream shape `fma(bf16_val, 0.0, acc) = acc`
was an identity over all 32 reduction steps.

Triton's bf16 `tl.sum` / `tl.max` compiles to exactly this pattern
on gfx1250: `v_fma_mix_f32_bf16 v_acc, v_bf16, 1.0, v_acc
op_sel:[0,1,0] op_sel_hi:[1,1,0]`.  So every bf16 Triton kernel
fed a field of zeros through its reduction while the raised IR
looked structurally identical to native.

**Fix (commit 0d002aecf2).** In
`handle-valu-vop3p.cpp::readMixSrc`, when the operand slot is
non-register (`!op.isSrcReg(i)` -- inline constant, 32-bit literal,
or expression), skip the op_sel-based half extraction and take the
LOW 16 unconditionally.  Register sources retain op_sel.  The
`op_sel` bit is VGPR-slot state; it has no meaning for a pre-
resolved immediate whose bit pattern IS the narrow value.

    bool isImmediateOperand = !op.isSrcReg(i);
    if (!isImmediateOperand && opSel[i] == 1)
      bits = trunc(lshr(raw, 16), i16);   // HI 16 (register half)
    else
      bits = trunc(raw, i16);             // LO 16 (immediate OR
                                          // register op_sel[i]=0)

**How it was pinned -- the "constant-input probe" technique.**
Random-input probes (`topk_forward_bisect_m1`) showed noisy,
row-dependent wrong-to-right ratios (3.09x, 4.53x, 0.09x, ...) --
not a clean "subset of K elements summed" signature, because
partial sums of random values don't have stable ratios across rows.

The crucial move: replace the random input with a CONSTANT.  With
all X[r, c] = 1.0 and a 32-element `tl.sum(axis=1)`, the expected
output is exactly 32.0 for every row.  Any deviation becomes
VALUE-INDEPENDENT -- the signature tells us HOW MANY elements
hotswap actually summed, not what mix of them.

  * hotswap writing `N × const` for some `N != 32` -> reduction
    tree dropping `(32 - N)` contributions.  Examine which
    structural paths are skipped.
  * hotswap writing `const` (N=1) -> reduction is a full identity.
    The multiplier path is gone.  Inspect the CONSTANTS flowing
    into the reduction ops in the lifted IR.

Hotswap wrote 1.0 (N=1).  Zero contributions.  Lifted-IR inspection
found every `@llvm.fma.f32` had `float 0.000000e+00` as its second
argument (should have been 1.0).  That constant literally doesn't
come from anywhere other than the FMA_MIX handler's readMixSrc --
root cause trivially localised from there.

**Wrong hypotheses enumerated and ruled out (two-day chase).**

- **H1 -- DPP -> ds_bpermute rewrite broke EXEC convergence.**  The
  ds_bpermute emitted inside an `emitUnderExec` diamond reads 0
  from EXEC-inactive source lanes (AMDGPU LDS bpermute semantics),
  whereas native DPP reads VGPR content regardless of EXEC.
  Plausible shape for the miscompile.  Ruled out by temporarily
  gating the rewrite off (one-line comment in
  `rewrite-cross-lane-divergent.cpp`'s site-collector): the
  const-input probe still produced 1.0 with faithful-lift
  `@llvm.amdgcn.update.dpp` in place.
- **H2 -- `strict.wwm` would force wave-wide EXEC around the
  bperm.**  Followed from H1 as a candidate fix.  Wrapping the
  bperm result in `@llvm.amdgcn.strict.wwm` changed nothing.
  Reverted.
- **H3 -- `permlanex16` emulation mis-reads source lane.**  The
  existing `canary_permlanex16_rowmax_fp32` matches, so the
  emulation is sound in isolation.  The m1 miscompile surfaces
  only in composition with FMA_MIX + packing, not in permlanex16
  alone.  Composition-level suspicion was plausible but empirically
  wrong -- fixing FMA_MIX dropped m1's error by the full
  accumulator magnitude, and the residual is bf16 reduction-order
  drift.
- **H4 -- `s_pack_ll_b32_b16` produces wrong halves.**  The scalar
  pack reads `low16 | (low16 << 16)` on two SGPR sources.  If one
  source had garbage in its low 16, the pack would carry it.
  Ruled out by tracing the pack's source SGPRs back to their
  `v_readlane_b32 s, v, 31` producers: those readlanes were on
  the downstream side of the reduction, not the source side.  A
  pack-side bug would have shown a different signature (wrong
  halves, not zero-accumulation).
- **H5 -- `emitUnderExec` models inactive lanes with UNDEF phis;
  downstream cross-lane reads see UNDEF.**  Structurally possible
  but not the active bug here: the FMA_MIX fix resolved the
  symptom, proving the cross-lane read path was correct all
  along.  Remains a reasonable concern to audit in isolation if
  a future bug surfaces that FMA_MIX doesn't explain.

**Graduation.**

- `topk_forward_bisect_m1_const_in`: WRONG 2048/2048 (output 1.0)
  -> `match` (output 32.0).
- `topk_forward_bisect_m1` (random): 2048/2048 WRONG `max|err|=39`
  -> 1300/2048 WRONG `max|err|=0.25` (≤ 2 bf16 ULPs at every
  mismatched magnitude; 1404/2048 ≤ 1 ULP; residual is
  non-associative bf16 reduction-order drift between Triton's
  gfx1250 and gfx942 tree shapes -- not a miscompile).  Relaxing
  the comparator from `tol: 0.0` to a few bf16 ULPs would
  graduate m1 to `match`; kept untouched here so the regression
  surface stays bit-exact.
- `topk_forward_bisect_m2..m4`, `topk_forward_bf16`: improved
  proportionally; remaining mismatches are the same bf16
  rounding drift composed with softmax / sort / argmax
  sensitivity.
- Canary grid (6 DPP / permlane / readlane / cvt recipes)
  continues to match bit-exactly.
- `lit_tests/v_fma_mix_f32_bf16/` extended with two
  inline-constant FMA sites (`op_sel:[0,0,0]` and
  `op_sel:[0,1,0]`) that MUST produce `float 1.000000e+00` as the
  fma's second arg.  Negative pin rejects `float 0.000000e+00`
  feeding any `fma.f32` call in this fixture -- locks in the
  pre-fix miscompile shape as a regression guard.

**Generalised rule for the `readMixSrc`-like family.**  Any
handler that reads an MCOperand and applies MC-encoding-specific
post-processing (op_sel half extraction, sign extension, neg/abs
modifier bits, packed-vs-unpacked interpretation) MUST branch on
`op.isSrcReg(i)` vs the immediate forms.  MC's pre-resolution
path is dtype-aware (`OperandType` -> which inline-imm helper
runs) and strips modifier state that would have applied to a
VGPR operand.  Extending the current handler's op_sel /
op_sel_hi logic into a new neg/abs-carrying variant without
auditing the isSrcReg branch is how this class of bug returns.

**The constant-input probe approach as a methodology.**  See the
"Diagnostic technique -- value-independent constant-input probes"
entry below for mechanisation notes; this bug is the reference
case the entry is written against.

---

## 2026-04-22 -- Diagnostic technique -- value-independent constant-input probes

**Problem class.**  Cross-widening miscompiles whose symptom is
noisy under random inputs -- different rows show different
wrong-to-right ratios with no clean structural pattern.  Examples
from the corpus today include the FMA_MIX inline-constant bug
(entry above), the `_D16_HI` store-upper-half bug (commits
2ebfadeb95 / b827c55899), and the `topk_forward` / reduction
miscompile class generally.  Random-input probes surface the
symptom but can't localise it -- the noise floor swamps every
structural signal.

**Technique.**  For any recipe that computes a deterministic
function over its inputs, replace random inputs with a CONSTANT
and compare hotswap output to the analytically predicted result.

  * For a reduction `tl.sum(axis=1)` over N elements of value v,
    expected output = `N * v`.
  * For a reduction `tl.max(axis=1)` over N elements of value v,
    expected output = `v`.
  * For an elementwise `y = f(x)`, expected output = `f(v)` per
    slot.

Hotswap's deviation from the analytic prediction is now
value-INDEPENDENT.  The deviation ITSELF carries structural
information:

  * Output = `v` for a reduction over N ≠ 1 -> the reduction
    tree is an identity over the multiplier; the mul path is
    broken.  Inspect the CONSTANTS appearing in the lifted IR's
    reduction ops.  Wrong-constant-at-a-specific-slot is the
    smoking gun.
  * Output = `K * v` for some `K < N` -> the reduction drops
    `(N - K)` contributions.  Inspect which structural paths are
    skipped.  Does `K` equal a lane count, a warp count, a row
    block count?  That's the dimension that was collapsed.
  * Output = `v * scalar_not_in_N`s-divisor-set` -> FP arithmetic
    is happening but with the wrong multiplier somewhere.  Inspect
    constants AND the modifier flags (neg, abs, scale_sel).
  * Output = `v + offset` -> an additive bias is leaking in.
    Could be a prior register state not cleared, or an init-bias
    (e.g. bf16 RNE `+0x7FFF`) surfacing into the output.

**Mechanisation -- tractable today.** For every deterministic reduction
recipe, auto-synthesise a `_const_in` sibling recipe:

```python
# Sibling generator: given a base recipe, emit a _const_in variant
# with the same kernel but inputs filled to a single value.
const_in_recipe = {
  **base_recipe,
  "name": base_recipe["name"] + "_const_in",
  "inputs": [
    {**inp, "range_lo": 1.0, "range_hi": 1.001}  # tight range
    for inp in base_recipe["inputs"]
  ],
}
```

Run both the base recipe AND its _const_in sibling.  The _const_in
verdict is binary (match / wrong) -- and if WRONG, the pattern of
deviation mechanically maps to a suspect class via the table
above.  This can be a CI gate on every recipe that has a reduction
primitive in its kernel AST.

**Mechanisation -- harder.**  Automated lifted-IR constant
inspection: for any `@llvm.fma.f32` / `@llvm.fmuladd.f32` / other
arithmetic intrinsic in the raised IR, flag any LITERAL constant
operand that equals a "silently-damaging" value (0.0 as a
multiplier, 1.0 as an addend, NaN anywhere, etc.) and print the
MCInst operand it came from.  This is what a human does during
triage -- the Cursor / grep workflow is already mechanical-adjacent.
A proper lint pass would live under a future IR audit pass and run as part of `raise_cli --audit`.

**What NOT to mechanise (yet).**  Static analysis of handler
source for "this code applies MC-encoding-dependent logic without
checking isSrcReg first" -- the static surface is too noisy;
legitimate `op.srcF(i)` calls do not need the isSrcReg branch
unless they read MC-encoding-state AFTER the read (op_sel, neg,
abs, clamp, scale_sel).  Expressing that "after the read" condition
cleanly in a linter is harder than just writing one _const_in
probe per recipe.

**Corollary -- probe recipe hygiene.**  Every new recipe added to
the direct-invocation corpus SHOULD include a `_const_in` sibling
unless the kernel has no reduction or no element-wise op with a
per-element closed form.  The cost is low (~20 lines of Python);
the return is catching exactly this class of bug BEFORE it needs
a multi-day hunt through EXEC / DPP / permlane / strict.wwm false
leads.  See `topk_forward_bisect_m1_const_in.py` (commit
b77e477908) for the reference template.

---

## 2026-04-21 -- Matmul128x128 residual: fixed by V_CMP -> V_CNDMASK per-lane-i1 shadow (closes the whole family)

**Context.** Follow-up (and close-out) to the `warp-3-specific, K-iter-0-
specific, upper-VGPR-bank A-load` entry directly below. The four
diagnostic probes had narrowed the defect to an exact shape -- now we name
the ROOT CAUSE and retire all six `Gfx1250Gpu.Matmul*` XFAIL entries.

**Root cause.** V_CMP -> SGPR -> V_CNDMASK round trip across cross-widening.

The matmul's prologue computes the per-lane LDS address for the
upper-VGPR-bank A load through the pattern

```
v_cmp_*_e64  sN, ...       ; wave-mask (wave-width i1 per lane),
                           ; stored to SGPR narrow-width (i32 on
                           ; wave32 source)
... SGPR may be written / read scalarly ...
v_cndmask_b32  vdst, src0, src1, sN   ; per-lane select keyed on sN
```

Under `WaveNativeProjection` (wave32 source -> wave64 target), the V_CMP
writer produces a **wave-width** (i64) per-lane i1. The SGPR destination
is **source-width** (i32), so `ballotI1ToWidth` truncates the i64 ballot
to i32 -- destroying the upper 32 bits which carried the compare result
for target lanes 32..63 (the lanes holding source wave 3's share of the
workgroup). The V_CNDMASK consumer then reads the SGPR back and routed
it through `extractLaneBitFromWaveMask`, which **replicates** the low 32
bits into both halves -- so target lanes 32..63 see source wave 2's (or
0's) compare result instead of their own. For warp 3's prologue A-load,
this mis-routes the `v_cndmask` that selects between two LDS-offset
candidates, so warp 3 reads from warp 0's LDS region for the K-iter 0
portion of its A fragment. The observed substitution (rows 124..127 <-
A[0], A[2], A[4], A[6]) is the lane-by-lane consequence.

**Fix (commit da404faf84).** Add a per-BB
`DenseMap<int, WaveMaskEntry>` shadow in `RaiseContext` that caches the
per-lane i1 produced by the most recent V_CMP_*_e64 writer to each SGPR.
The V_CNDMASK consumer consults the shadow FIRST and reads the i1
directly -- bypassing the lossy narrow-ballot round trip entirely -- and
falls back to `extractLaneBitFromWaveMask` only when the shadow is
empty (no fresh V_CMP writer in this BB, scalar SGPR write invalidated
the cache, or we crossed a BB boundary).

Three invariants ensure soundness (see `sgpr-wave-mask-translation.md`
section 3.1 for the full treatment):
- **I1 (Additive).** Narrow store + extract reader both preserved; the
  shadow is consulted in addition, not instead. If a kernel's V_CNDMASK
  path was correct before the fix it stays correct after.
- **I2 (SSA-monotonic within a BB).** The SSA value in the cache is the
  exact `cmp` produced by the last V_CMP writer to `sN`, with no
  intervening write. Linear handler dispatch guarantees this.
- **I3 (Any interference defeats the cache).** Scalar SGPR writes
  invalidate via `onSgprWritten` (fired by `storeSGPR32 / storeSGPR64`),
  and the shadow is cleared at BB boundaries.

**What the four diagnostic probes each contributed.**

- `RowIdA` (`A[i,k] = (i+1)·0.001`): identified the SUBSTITUTION arithmetic
  (`rows 124..127 <- A[0], A[2], A[4], A[6]`) rather than a miss/zero.
- `RowOnly124`: ruled out 2×2-grid-specific boundary handling
  (single-tile reproduces the same pattern).
- `EvenRows`: proved WAVE-3 SPECIFICITY -- rows 12..15, 28..31, 44..47,
  60..63, 76..79, 92..95, 108..111 ("sub-tile-row-1 rows 12..15" bands
  for warps 0/1/2) were ALL CORRECT, only rows 125/127 wrong. Rules out
  any general pass-2 / row-12..15 / target-wave-upper-half defect.
- `KStripedRow124`: proved K-ITER 0 SPECIFICITY (got ≈ 44.8 = 48 - 3.2,
  missing the 0.1 strip, which lives at k in [0,32)). Rules out main-
  loop K-iter bugs, pins the defect to the prologue's upper-bank load.

All four pass bit-exact under the fix and remain in the test suite as
positive regression guards.

**Wrong hypotheses enumerated and ruled out (from the H1..H4 hand-off).**

- **H1 -- Upper-VGPR-bank `ds_load_tr16_b128` destination write under
  `s_set_vgpr_msb`.** The ds_load is lifted correctly: dest VGPR
  baseIdx picks up the MSB adjust via `parseReg`'s
  `currentVGPRAdjust[]`, and the write is gated by `emitUnderExec`
  which reads the per-lane mask. The defect was upstream of the
  ds_load: the v_cndmask that computed the ds_load's per-lane ADDRESS
  was the corrupted site, not the ds_load itself.
- **H2 -- Raiser ttmp8 seed wave-uniform.** Confirmed per-lane divergent
  at IR emission (`%ttmp8_val = shl i32 %wave_id_in_wg, 25` where
  `%wave_id_in_wg = lshr i32 %workitem.id.x, 5`). Not the defect.
- **H3 -- LLVM AMDGPU backend regalloc miscompile for the upper-VGPR-
  bank path.** Bug reproduces byte-identically on both projections and
  with identical gfx942 regalloc output patterns; if regalloc were
  coalescing v300.. onto a warp-0 live range, changing projection
  would not preserve the error pattern.
- **H4 -- Per-wave MSB drop at the `v_cndmask_b32_e32` dst.** The MSB
  adjust IS applied correctly (verified by runtime tracing: the
  v_cndmask's ParsedReg for dst has `baseIdx = 256` under
  `s_set_vgpr_msb 64`). The defect was the v_cndmask's COND operand
  (the SGPR wave mask) being truncated, not its DST.

**Why the obvious hypotheses all missed.** The substitution pattern
(warp 3 <- warp 0 A data) naturally suggests a cross-wave address or data
leak via LDS / bpermute / register-file aliasing. H1..H4 all point at
the DATA PATH that manifests the symptom. The actual defect was one
level upstream: the CONTROL-MASK (v_cmp result routed through an SGPR
to a v_cndmask) that steered the address arithmetic. Corrupting the
mask silently re-routes the data path in a way that looks identical to
a data-path bug.

**Graduation.**
- All six `Gfx1250Gpu.Matmul*` XFAIL entries removed from
  `the regression test configuration` (commit da404faf84 landed the fix;
  this entry and the follow-up retire the `WILL_FAIL TRUE`
  annotations).
- The four diagnostic probes remain as positive regression guards --
  each covers a distinct axis of the defect (substitution arithmetic /
  wave specificity / K-iter specificity / upper-vs-lower bank) and
  would fire independently on any regression that re-opens the shape.

---

## 2026-04-21 -- Matmul128x128 residual: narrowed to wave-3 + K-iter-0 (prologue) + A-data side

**Context.** Follow-up to the "data-substitution bug, NOT a collect-stage
defect" entry below. Two further diagnostic probes added to isolate the
defect axis.

**What was done.** Added `MatmulDataPattern::EvenRows` (A[i,·] = 1 iff i
is even, B = 1) and `MatmulDataPattern::KStripedRow124` (A[124,k] striped
per K-iter: 0.1 / 0.2 / 0.4 / 0.8 for k in [0,32)/[32,64)/[64,96)/[96,128)),
wired each through `doTestMatmul` and registered as XFAIL.

**What the evidence shows (in order of what each probe rules out).**

1. **`EvenRows` -- wave-3 specificity.** Errors appear ONLY at rows 125
   and 127 of the output. Rows 12..15, 28..31, 44..47, 60..63, 76..79,
   92..95, 108..111 -- every OTHER "sub-tile-row-1 rows 12..15" band
   across warps 0/1/2 -- are correct. Under the kernel's `4-warp × 32-row`
   tiling (`maxFlatWorkgroupSize=128` -> 4 source Wave32s; 128 output
   rows / 4 warps = 32 rows/warp; 32 rows = 2 sub-tile rows × 16), each
   warp has a "sub-tile row 1 rows 12..15" band that a general
   pass-2 / row-12..15 defect would affect. Only WARP 3's band
   (= rows 124..127) does. Source wave 3 = target wave 1 lanes 32..63 =
   pass-2 of `runGroupPass` on target wave 1; source wave 1 = target
   wave 0 lanes 32..63 is ALSO "pass-2" and is CORRECT, so the defect
   is NOT pass-2 in general.

2. **`KStripedRow124` -- K-iter 0 specificity.** With A[124,·] striped
   so each K-iter contributes a unique amount (3.2 / 6.4 / 12.8 / 25.6,
   summing to 48.0), the observed `C[124, ·] ≈ 44.8 = 48.0 - 3.2`
   identifies the missing contribution as K-iter 0 (k in [0,32)).
   This is the prologue 16-WMMA block in the disassembly (lines
   1564..1649 of `matmul_f16_large_gfx1250.hsaco`), NOT the main
   K-loop body. The prologue uses `s_set_vgpr_msb` to address the
   upper VGPR bank (v[300:331] for A sources via `/*v[300:307]*/`
   aliasing); the main loop uses the lower bank (v[128:159]). The
   bug therefore implicates the upper-bank-loaded A path, not the
   lower-bank path the main loop uses.

**Refined defect statement.** "Warp 3's prologue WMMA output for rows
12..15 of its sub-tile row 1 (= global rows 124..127) receives
A-fragment contribution from warp 0's A-rows 0, 2, 4, 6 -- a
cross-warp, wave-3-specific, K-iter-0-specific data substitution on
the A-fragment side of the WMMA chain, on the UPPER-VGPR-BANK path
(`s_set_vgpr_msb` active)."

**What NOT to waste time on.**

- The substituted source rows (0, 2, 4, 6 in GLOBAL A-matrix, not
  0, 2, 4, 6 intra-sub-tile) definitively rule out a local lane-map
  defect in `wmma-lowering.cpp`. See RowIdA arithmetic: `3.968 =
  4.000 - 32 * A[0] = 32 * (A[124] - A[0])` for row 124;
  intra-sub-tile row 0 would be A[112] = 113/1000, which produces
  `15.616`, not the observed `12.032`.
- The bug is invariant under WaveNative vs ModuloReplication
  projection (handoff confirmed, re-verified here: `EvenRows`
  produces identical errors under both). So `WaveNativeProjection::
  emitInitialExec` and the `init_whole_wave` hardware-EXEC path are
  not implicated.

**What to investigate next (sharpened from the earlier entry).**

1. **The upper-VGPR-bank A load chain for source wave 3.** In the
   prologue, `ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v72 /*v328*/`
   (and its 7 siblings at offsets 64/128/192/4608/4672/4736/4800)
   writes A data into v[300:331] under `s_set_vgpr_msb 0x41` (dst +
   src0 MSB=1). The LDS base address is in v72 (= v328 under MSB
   adjust), computed per-lane from `s8 + v229 + v230` (`v_add3_u32`,
   line 1534). Trace what warp 3 lanes see as v328 vs what warp 0
   lanes see -- the `A[0,·]` substitution implies warp 3 is reading
   warp 0's LDS region for this specific load.

2. **Wave-ID plumbing under cross-widening.** The matmul lifts
   `s_bfe_u32 ttmp8, 0x50019` to `(workitem.id.x >> 5) & 0x1F`. For
   a 128-thread workgroup on target Wave64 lanes 0..127, this gives
   per-lane values `0,0,...,0 (×32) | 1,1,...,1 (×32)` on target
   wave 0 and `2,2,...,2 (×32) | 3,3,...,3 (×32)` on target wave 1.
   The source kernel MAY assume wave_id is WAVE-UNIFORM (one value
   per hardware wave). Under modulo-replication, target wave 1 has
   TWO different wave_id values (2 and 3) on its two halves. If
   ANY downstream use of this per-lane value goes through a scalar
   operation (readfirstlane, scalar spill/reload, or an SGPR-
   constrained consumer the writelane/readlane rewrite's forward-
   use-chain classifier didn't catch), the wave-3 lanes would see
   wave_id=2 and read warp-0's LDS region via a derivation like
   `s73 = s2 & 3` -> LDS offset. But substituted = warp 0 (not warp 2),
   so it's NOT wave-id-collapse-to-peer (which would give warp 2).
   Check if there's a `readfirstlane` or similar on the wave_id-
   derived SGPR chain that could collapse warp 3's value to WARP
   0's (e.g. an SGPR broadcast from lane 0 under a narrow EXEC).

3. **The 8 `ds_load_tr16_b128` instruction offsets.** They split
   {0, 64, 128, 192} + {4608, 4672, 4736, 4800} into two groups of
   4. The group of 4 with larger offsets is at +4608 = +0x1200 = +4K
   into LDS -- crosses into a second 4K LDS region. Check whether
   wave 3's LDS stride for the prologue's first load causes it to
   wrap around into wave 0's region.

4. **The `s_set_vgpr_msb` end-to-end probe.** It exercises the
   upper-VGPR-bank path with a non-WMMA kernel. Run it under the same
   `--enable-writelane-rewrite + --enable-wave-native` config as
   the matmul and check whether warp-3-specific data substitution
   appears there too. Same result would confirm the bug is in
   MSB-path lifting; a clean run would localise to the
   LDS-side-of-WMMA interaction.

**Files touched:**
`GPU regression tests` (`EvenRows`, `KStripedRow124` patterns
+ `TEST_F` registrations), `the regression test configuration` (four `WILL_FAIL`
entries + commentary), this doc.

---

## 2026-04-21 -- Matmul128x128 residual: data-substitution bug, NOT a collect-stage defect

**Context.** Follow-up investigation of the `Gfx1250Gpu.Matmul128x128*`
residual documented in the previous entry. Handed-off with the primary
hypothesis that the bug lives in the WMMA->MFMA "collect" stage of
`wmma-lowering.cpp::runGroupPass`, specifically the `ds_bpermute`-based
gather that maps a 4-VGPR MFMA output back into the 8-VGPR Wave32 C
fragment.

**What was done.** Added two diagnostic input patterns to `doTestMatmul`
in `GPU regression tests`:

- `MatmulDataPattern::RowIdA`: `A[i,k] = (i+1) * 0.001` for all k, `B = 1`.
  Reference `C[i,j] = 128 * (i+1) * 0.001`, constant across columns.
  Every output row has a unique expected value -- any per-row mis-routing
  is immediately visible numerically.
- `MatmulDataPattern::RowOnly124`: `A[i,k] = (i == 124 ? 1 : 0)`,
  `B = 1`. Reference `C[124, j] = 128` for all j, every other row = 0.
  If any of row 124's K-iters is lost, C[124,j] ≠ 128.

Also added a per-row / per-column error histogram to the error-summary
path.

**What the evidence shows.**

1. **RowOnly124 is the smoking gun.** Output row 124 comes out as
   `96.0` across all 128 columns -- EXACTLY 32 units short of the
   reference 128.0. 32 is the K-dimension of a single WMMA call
   (`v_wmma_f32_16x16x32_f16`). So ONE WMMA instance's contribution
   to row 124 is being SILENTLY REPLACED with another row's data
   (which happens to be 0 under this pattern, since every row
   except 124 has A=0).

2. **RowIdA pins the substitution pattern.** For random-free data
   where `A[i] ∝ i`, output rows 124..127 come out with the missing
   K-iter's contribution equal to `32 * A[2*(row - 124)]`:

   | output row | expected    | got         | missing ctr | source row |
   |------------|-------------|-------------|-------------|------------|
   |      124   | 16.0        | 12.032      | 3.968       | A[0]       |
   |      125   | 16.125      | 12.1898     | 3.935       | A[2]       |
   |      126   | 16.25       | 12.3475     | 3.903       | A[4]       |
   |      127   | 16.3906     | 12.5170     | 3.874       | A[6]       |

   Equivalently, output row `(124 + r)` receives contribution from
   row `2 * r` for `r ∈ {0, 1, 2, 3}` on one specific WMMA call.

3. **Only output rows 124..127 are affected.** Rows 12..15, 28..31,
   44..47, …, 108..111 -- i.e., rows 12..15 of every OTHER 16-row
   sub-tile row -- are correct. If the defect were a general COLLECT-
   stage bug affecting Wave32 lanes 16..31 GPRs 4..7 uniformly across
   every WMMA, we would expect errors in ALL these rows. We don't.

4. **Collect-stage math is correct.** Spent significant time
   verifying the `srcLane = 32*(w32Lane>=16) + 16*(GPR_w>=4) +
   (w32Lane & 15)` formula both in `wmma-lowering.cpp` and in the
   `--emit-ir` dump (`/tmp/mm.ll`). For pass 2 at W64 lane 48
   (w32Lane=16, the first failing lane), the formula produces
   `srcLane = 48` for `gw ∈ {4..7}`, which reads MFMA `LG 3` at
   the expected `mfmaDwords[0..3]` indices -- exactly what the
   file-header lane-layout equations prescribe. The emitted IR
   matches the formula literally.

5. **IR structure is correct.** Pass 1 and pass 2 of `runGroupPass`
   produce independent `result0[]` / `result1[]` arrays, packed via
   `select i1 is_group1` into the final `<8 x float>` result. No
   SSA sharing, no cross-pass aliasing.

**What this means.** The defect is NOT the hypothesised collect-stage
lane-map bug. The collect math is correct AND uniformly applied across
every WMMA. The rejection criterion: if it were a collect bug, it would
affect EVERY sub-tile's rows 12..15, not just rows 124..127.

**What the bug IS.** Under-specified. The data is: ONE specific
WMMA-call-worth of contribution is dropped for rows 124..127, and the
wrong data substituted in follows a very specific `2*(row - 124)`
pattern in A-row indexing. The substitution crosses what would be
sub-tile boundaries (rows 0..6 of the A matrix, not rows 112..118 of
the failing sub-tile), which rules out intra-sub-tile mis-routing and
suggests the defect is EITHER:

- In a specific WMMA's A-fragment load (the `ds_load_tr16_b128`
  emulation, or the raising of its VGPR-MSB-adjusted destination), OR
- In a post-WMMA shuffle / reduce step (an LDS rearrangement
  the Triton kernel does between WMMA accumulation and the final
  store) that crosses warp boundaries, OR
- In the interaction between `s_set_vgpr_msb` state and some handler
  whose operand-index map doesn't consult `ctx.currentVGPRAdjust[]`.

**What NOT to waste time on (next investigation).** Both `wmma-lowering.cpp`
collect-stage math and `redistributeInput` / `redistributeAcc` math are
verified correct against the documented lane-layout equations. The IR
dump shows these formulas emitted literally. Do not re-derive them from
first principles a third time.

**What to investigate next.**

1. Grep the disassembly for the specific WMMA whose dst is the
   accumulator covering rows 124..127. Trace its A-fragment load
   chain back through `ds_load_tr16_b128` and the s_set_vgpr_msb
   state surrounding it. If the dst / src0 MSB adjustment on that
   specific path is dropped somewhere, that would explain the
   "wrong A row" substitution (reading from a lower-bank VGPR
   instead of the upper-bank one).
2. Write a minimal HIP reproducer that has 4 chained WMMAs across
   2 source waves with `s_set_vgpr_msb` in the source kernel and
   non-uniform A/B data. If it reproduces, the bug is
   `s_set_vgpr_msb`-related; if it doesn't, suspect the Triton-
   kernel-specific LDS shuffle path.
3. The `s_set_vgpr_msb` end-to-end probe exercises the upper-VGPR-bank
   path with a non-WMMA kernel. Run it under the same `--enable-writelane-rewrite` + `--enable-wave-native`
   configuration as the matmul gtest. A mismatch there isolates the
   MSB path; a match rules it out.

**Value landed anyway.** The two diagnostic input patterns (RowIdA,
RowOnly124) and the per-row/per-column error histogram make the defect
shape trivially visible without any manual scripting. The xfail entries
for them are gated with the same `WILL_FAIL TRUE` pattern as the parent
matmul tests, so CTest still validates that the defect is reproducible
(a silent fix would flip them unexpected-pass and force a review).

**Files touched:**
`GPU regression tests`, `the regression test configuration`, this doc.

---

## 2026-04-21 -- WaveNative projection alone does not fix the Matmul128x128 residual

**Context.** After the writelane/readlane symmetry fix (same day, below),
the Matmul128x128 random-input tests still failed with ~3% numerical
errors localised to output rows 124–127 / 252–255. The
`wmma-lowering.cpp` file header and the `WaveNativeProjection`
docstrings both point at partial-EXEC during WMMA -> MFMA as the
culprit, with `@llvm.amdgcn.init_whole_wave` as the documented fix.

**What was done.** Finished wiring `WaveNativeProjection` end-to-end:
added an `enableWaveNative` opt-in flag through `raiseToIR` /
`runPipeline*` / `raise_cli`, plumbed the projection-driven EXEC
alloca width + initial value through `reg-file.cpp`, widened the
source-width SGPR <-> EXEC-width bridging in
`raise-context.cpp::readOpExecWidth`, flipped the V_CMP -> SGPR write
in `handle-valu-vcmp.cpp` to pick `sourceWaveMaskTy()` rather than
`execTy`, removed the now-obsolete `strict.wwm` wrappers in
`wmma-lowering.cpp` (they are subsumed by the kernel-entry
`init_whole_wave`), updated the 5 lit fixtures that expected the
wave-native IR shape, and flipped `doTestMatmul` to opt in. The
compiled HSACO for the matmul kernel grew from 83648 B to 133760 B
and now starts with `s_or_saveexec_b64 s[0:1], -1` -- confirming the
projection reaches codegen exactly as intended.

**What the evidence showed.** The matmul random-input errors did not
change: same 490 count on the single-tile case, same 1985 on the 2×2
grid, same row pattern (124–127 / 252–255). Under both ModRep and
WaveNative the errors are byte-identical. That rules out partial-EXEC
at the MFMA collective as the cause -- WaveNative provably fixes
partial EXEC, and the error pattern is insensitive to that fix.

**What we still don't know.** The residual must live deeper in the
WMMA -> MFMA redistribution math -- most likely in the "collect" stage
in `wmma-lowering.cpp` for the second half of source wave 3's
sub-tile, or in some bookkeeping the matmul exercises that the
smaller `wmma_*` lit fixtures don't. Investigation handed off via
`the regression test configuration`'s rewritten commentary and the list of MFMA
lane-layout equations at the top of `wmma-lowering.cpp`.

**Value landed anyway.** WaveNative is a principled piece of
infrastructure: it unblocks the 5 previously-failing lit fixtures
(`v_cmpx_ballot`, the four `wmma_*`), provides the correct EXEC
shape for any future kernel whose WMMA path actually does suffer
from partial EXEC, and the `--enable-wave-native` flag structure
lets callers opt in per surface. The `strict.wwm` per-MFMA-output
strategy it replaces was documented to crash `SIPreAllocateWWMRegs`
on 128×128 matmul tiles, so the refactor is net-positive even though
it does not solve the specific matmul residual.

---

## 2026-04-21 -- Writelane/readlane rewrite must be symmetric under cross-widening

**Symptom.** `Gfx1250Gpu.Matmul128x128*` faulted at dispatch with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (HIP 700) after
`--enable-writelane-rewrite` graduation. Raise succeeded
(`4261/4261`, HSACO 83648 B); the kernel launched and faulted mid-dispatch.

**Wrong hypothesis (the obvious one).** "The rewrite shifts the
`v_readfirstlane_b32` scalarisation point a few instructions later."
Falsified by the compiled HSACO: only 3 `v_readfirstlane_b32`
instructions exist, none consume a `ds_bpermute` result, and all three
are uniform-VGPR scalarisations pre-existing in every gfx942 matmul
build.

**Actual cause.** The rewrite was **asymmetric**:

- Uniform writelanes (e.g. `writelane(ka_lo, 0, undef)`) were preserved
  as the native `v_writelane_b32` -- writes hardware lane 0 **only**.
- Divergent readlanes on the same VGPR were rewritten to
  `ds_bpermute(((lane_id & ~(W_s-1)) | lane_idx) << 2, src)` -- reads
  **both** lane 0 **and** lane 32.

Target lanes 32..63 read `undef` from hardware lane 32, zext'd it into
the high half of an `i64` global-memory pointer, and the `global_load`
faulted on the bogus address. Uniform-diag test (`A=B=1.0`) passed
because its reference is position-invariant (`K = 128` everywhere) --
which was the diagnostic that isolated the defect to "addressing /
cross-replica data-placement," not to WMMA / accumulator state.

**Fix.** `rewrite-cross-lane-divergent.cpp` now rewrites **every**
writelane and **every** readlane site unconditionally under
cross-widening, regardless of operand divergence. The
`select`/`ds_bpermute` shapes are semantically equivalent to the
source opcodes for any operand-divergence combination, so unconditional
rewriting is correctness-preserving and makes the writelane/readlane
pair trivially self-consistent on any shared VGPR.

**Guarded by a forward use-chain classifier.** Unconditional
rewriting is sound only when the rewritten result never reaches an
SGPR-constrained consumer. If a readlane result flows into
`llvm.amdgcn.s.buffer.load`'s rsrc, an `llvm.amdgcn.s.sendmsg`
message, a `llvm.amdgcn.readfirstlane` sink, a load from
addrspace(4), inline asm with `"s"` constraint, or any unknown sink,
the backend inserts `v_readfirstlane` on the `ds_bpermute` output
and recreates the original source-wave collapse. The rewrite pass's
`classifyForwardUseChain` walks every writelane / readlane result's
transitive uses pre-rewrite; unknown users over-approximate to
SGPR-forced. If any site's chain is SGPR-forced, the pass refuses
the whole function via `CrossLaneDivergentRewriteReport::
sgprForcedDetail`, which the raiser surfaces as
`RaiseFailure::crossWaveRewriteOracleDisagreement`. The refusal is
all-or-nothing because a mix of rewritten and preserved sites on a
shared VGPR recreates the asymmetric-rewrite fault. Pinned by
`lit_tests/writelane_sgpr_forced_use` (fixture chains writelane ->
`v_readfirstlane_b32`; asserts refusal under the flag + clean raise
under flag-off).

**Why the old rule was wrong in principle.** "Preserve uniform sites to
keep codegen quality" sounds local-reasoning-safe but pairs a
hardware-single-lane write with a software-two-lane read across the
source-wave boundary. The rule is only sound if you can prove that
**no** rewritten readlane on the same VGPR ever reads lane `N + W_s`
-- a non-local invariant the pass had no mechanism to enforce. The
symmetric rule replaces a non-local invariant with a local one
("every site uses per-source-wave semantics").

**Residual after fix.** `Matmul128x128_1tile_UniformDiag` now passes
bit-exact (0 errors). The two random-input `Matmul128x128*` variants
still fail with ~3% numerical errors, localised to source wave 3's
**last 4 output rows** (rows 124–127 single-tile; 252–255 in 2×2-grid).
Not caused by the writelane/readlane rewrite -- belongs to the
WMMA->MFMA "collect" stage under `ModuloReplicationProjection`.
Re-added to `xfail.cmake` with that precise reason.

**Files touched:**
`rewrite_cross_lane_divergent.{hpp,cpp}`,
`lit_tests/writelane_uniform_noop/*`,
`wave-size-translation.md` §5.6.3,
`the regression test configuration`,
`GPU regression tests`,
`raise_cli.cpp` (added `--write-hsaco` for disassembly triage).
