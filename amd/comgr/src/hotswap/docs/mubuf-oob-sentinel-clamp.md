# MUBUF out-of-bounds sentinel offset clamp (gfx1250 -> gfx950 loads)

## Problem

Triton's masked buffer-access idiom forms a per-lane buffer offset as

    offset = select(in_bounds, real_offset, 0x80000000)

where `0x80000000` (>= the descriptor's NUM_RECORDS) is a deliberate out-of-bounds sentinel for
lanes the kernel masks off. On the source gfx1250 a raw `buffer_load` whose offset exceeds
NUM_RECORDS is bounds-suppressed by hardware -- it returns 0 and does not fault -- and the kernel
relies on that: the sentinel-lane result is discarded downstream.

On gfx950 the same out-of-bounds `offen` offset (with XNACK) raises `EXCP.MEM_VIOL` instead of
being suppressed. This is a gfx1250 -> gfx950 hardware OOB-suppression semantic gap, and it lives
entirely inside a single source wave, so it is independent of the wave projection (it faults under
both WaveNative and the scaled/modulo-replication projections -- there is no partner wave to
remove).

Register-pinned on the gemma-3-4b `_fwd_kernel` BN=32 attention load:

    v_bfrev_b32_e32  v0, 1                                   ; v0 = 0x80000000 sentinel
    v_cndmask_b32_e32 v6, v0, v6, vcc                        ; select(mask, real, sentinel)
    buffer_load_dwordx4 v[152:155], v6, s[44:47], 0 offen   ; FAULT: MEM_VIOL + XNACK

with descriptor NUM_RECORDS = 0x7ffffffe and sentinel offset 0x80000000 = NUM_RECORDS + 2.

## Fix

Restore the source hardware's suppression at the offset value layer. In `decodeMubufAddr`
(`mubuf-addr.cpp`), after both the per-lane byte offset `Voffset` and the reconstructed byte
extent `NumRecords` are available, redirect any load offset that is `>= NumRecords` to 0:

    Voffset = (zext(Voffset) uge NumRecords) ? 0 : Voffset

gfx950 then reads in-bounds element 0 on those lanes instead of faulting -- matching gfx1250's
return-0 behaviour. The clamp is guarded on `>= NumRecords`, so in-bounds traffic (the common
path) is bit-identical.

### Load vs store

The clamp is applied to **loads only** (`!IsStore`). Stores and atomics are deliberately excluded:

- Redirecting a *store* offset to 0 would corrupt element 0 (write a masked lane's data over a
  real element). The store-side OOB sentinel is a separate, unfinished problem: live testing on
  gfx950 shows the OOB store faults (MEM_VIOL) too, so it cannot be left unclamped-and-unhandled,
  but the fix must SUPPRESS the sentinel lane's store (per-lane active information) rather than
  redirect its offset. That is store-mask territory, not an offset value clamp.
- An *atomic* must hit its real address; redirecting it would both corrupt element 0 and change
  the read-modify-write semantics.

## Known limitation (load side)

The clamp only covers offsets that are numerically `>= NUM_RECORDS`. Under a wave projection that
DEFEATS the source's per-lane sentinel select (e.g. WaveNative reactivating the full wave via
`init_whole_wave`), a source-inactive partial-tail lane can instead carry a wild/uninitialized
offset -- observed live as a qNaN bit pattern `0x7fc00000`, which is numerically *below*
NUM_RECORDS and so escapes this clamp. Making that case safe needs the per-lane sentinel select
to be honored (a projection that preserves per-lane masking), on top of which this clamp is a
correct backstop for the target-HW OOB-suppression gap. So the clamp is necessary but not
sufficient standalone; it must ride a projection that keeps the source mask.

`decodeMubufAddr` already carries `IsStore`; loads and buffer-load-to-LDS pass `false`, stores and
atomics pass `true`.

## Why not a per-op EXEC mask

This is a value clamp on the offset, not a per-lane EXEC / lane-active predicate. It does not
re-impose masking to defeat a back-end optimization, and it never touches the store mask, so it
cannot reintroduce the store over-masking regression that a lane-active store guard would. Because
the compare is against a runtime descriptor field (not a lane-position tautology such as
`lane < wave_size`), the optimizer cannot prove it constant and fold it away, so it survives -O2.
It is projection-independent and composes with the scaled dispatch projection rather than
competing with it.
