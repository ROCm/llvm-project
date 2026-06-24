; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=v_cmp_cndmask_sgpr_scalar_clobber_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; `--disable-wave-native` pins the MODREP projection path. The
; CHECK sequence below asserts the
; `ModuloReplicationProjection::extractLaneBitFromWaveMask` shape
; (mask_widen_shl / mask_widen_replicate), which is MODREP-specific:
; WaveNative stores per-lane EXEC at target width and emits a
; different extract chain, so the positive CHECKs below would not
; match post-graduation default. See
; hotswap/docs/modrep-predicate-chain.md §6 for the graduation
; rationale and wave-size-translation.md §5.6.1 for the WaveNative
; vs MODREP EXEC-storage contract.
;
; Fallback-path fixture for the V_CMP -> SGPR -> V_CNDMASK_B32_e64
; idiom. A scalar `s_mov_b32 s4, imm` sits between the V_CMP
; producer and the V_CNDMASK consumer (see companion .hip). That
; scalar write fires `AllocaRegFile::onSgprWritten`, which
; invalidates the per-lane i1 shadow entry in
; `RaiseContext::lastSgprWaveMaskI1`. The consumer then takes the
; fallback extract chain — exactly the "any interference defeats
; the cache" invariant from `sgpr-wave-mask-translation.md`
; section 3.1 / I3.
;
; Why this fixture exists. Without it, a future change that forgets
; the invalidation hook (wires `onSgprWritten` but doesn't call
; `invalidateSgprWaveMaskI1`, drops the hook from `storeSGPR32`,
; etc.) would silently read a stale per-lane `i1` that no longer
; corresponds to the scalar `s4` at the consumer site. The
; fused-path fixture in the sibling `v_cmp_cndmask_sgpr/` directory
; would still pass in that case. Only this fixture catches that
; specific regression mode.
;
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_cndmask_sgpr_scalar_clobber_kernel(

; The producer's compare still emits the wave-mask store (ballot
; + trunc). Shared with every V_CMP -> SGPR fixture; the trunc is
; the information-loss boundary documented in handle_valu_vcmp.cpp
; and in the design doc's section 2.
; CHECK: fcmp oge float %{{[^,]+}}, 5.000000e-01
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %{{[^)]+}})
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; Invalidation is not directly observable in the raised IR (it is a
; raise-time C++ DenseMap::erase call that leaves no IR artefact).
; What IS observable is its consequence: the cndmask's mask source
; goes through the full `extractLaneBitFromWaveMask` chain and the
; current shadow-selection guard. Pin the extract chain
; identifiers (`mask_lane_idx` / `mask_at_lane` / `mask_lane_bit` /
; `mask_lane_i1`) — these are the names emitted by
; `ModuloReplicationProjection::extractLaneBitFromWaveMask` in
; wave_projection.cpp.
;
; Under the wave32 source → wave64 target cross-widening this
; fixture drives (`--isa=gfx1250 --target-isa=gfx942`), the narrow
; mask is replicated into the upper half of the wide target mask
; BEFORE the per-lane `lshr`.  Replicate-widening is the fix for
; the `corpus_swiglu_fp32` miscompile: a plain `zext i32 → i64`
; left the upper 32 bits zero, and target-wave lanes 32..63 then
; shifted into the zeroed region — always reading a FALSE
; predicate and unconditionally taking the OOB-sentinel branch of
; every downstream `v_cndmask_b32_e64` that used the narrow mask
; as its selector (the `offset = s[m]?tid*8:0x80000000` shape
; Triton's AMDGPU backend emits for masked-out buffer accesses).
; The `mask_widen_shl` / `mask_widen_replicate` pair below is the
; replicate-by-left-shift-then-OR pattern that keeps lane `L` and
; lane `L + W_src` reading the same bit, matching MODREP's
; per-source-wave-width-replicated contract.
; CHECK: %mask_widen_shl{{[0-9]*}} = shl i64 %{{[^,]+}}, 32
; CHECK: %mask_widen_replicate{{[0-9]*}} = or i64 %{{[^,]+}}, %mask_widen_shl{{[0-9]*}}
; CHECK: %mask_lane_idx{{[0-9]*}} = zext i32 %{{[^ ]+}} to i64
; CHECK: %mask_at_lane{{[0-9]*}} = lshr i64 %mask_widen_replicate{{[0-9]*}}, %mask_lane_idx{{[0-9]*}}
; CHECK: %mask_lane_bit{{[0-9]*}} = and i64 %mask_at_lane{{[0-9]*}}, 1
; CHECK: %mask_lane_i1{{[0-9]*}} = icmp ne i64 %mask_lane_bit{{[0-9]*}}, 0

; NEGATIVE: the pre-fix "plain zext, then shift by full lane_id"
; shape must NOT be emitted.  If a future refactor drops the
; replicate-widening, target-wave lanes 32..63 get a zero'd mask
; upper half and the `corpus_swiglu_fp32` miscompile re-emerges.
; The specific pre-fix IR shape was `%mask_at_lane = lshr i64
; <zext_source_mask>, %mask_lane_idx` — i.e. the `lshr`'s first
; operand was a direct `zext i32 ... to i64` result rather than a
; `%mask_widen_replicate`.  This CHECK-NOT forbids that pattern.
;
; The CHECK-NOT is scoped to "a `lshr i64 %<zext-result>, ...` with
; NO intervening `%mask_widen_replicate`" — we spell it out as
; "lshr i64 where the shiftee is NOT named `mask_widen_replicate`"
; via the exact SSA-name prefix requirement in the positive CHECK
; above.  Defence-in-depth against a future refactor that keeps
; `mask_widen_replicate` emitted elsewhere but passes a different
; value to `lshr` — the positive CHECK's tight identifier binding
; above would catch it, this CHECK-NOT is a redundant belt:
; CHECK-NOT: %mask_at_lane{{[0-9]*}} = lshr i64 %{{[0-9]+}}, %mask_lane_idx

; The cndmask's condition MUST be the extract chain's `i1`, NOT the
; compare's `%vcmpf`. Under the shadow-path shortcut, `%vcmpf`
; would flow directly into the select; the invalidation correctly
; prevents that here because by the time of the cndmask, `s4` no
; longer carries the original compare's wave mask.
; CHECK: %sgpr_mask_shadow_sel = select i1 false, i1 %mask_lane_i1{{[0-9]*}}, i1 %mask_lane_i1{{[0-9]*}}
; CHECK: %cndmask = select i1 %sgpr_mask_shadow_sel, i32 1065353216, i32 -1082130432

; NEGATIVE: the fused / direct-i1 shape must NOT be taken here. The
; absence of `select i1 %vcmpf, ...` in the clobber kernel is the
; core invariant this fixture exists to enforce.
; CHECK-NOT: %cndmask = select i1 %vcmpf{{[0-9]*}},

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cmp_cndmask_sgpr_scalar_clobber_kernel
	.p2align	8
	.type	v_cmp_cndmask_sgpr_scalar_clobber_kernel,@function
v_cmp_cndmask_sgpr_scalar_clobber_kernel: ; @v_cmp_cndmask_sgpr_scalar_clobber_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_xcnt 0x0
	s_mov_b32 s2, 0x5a5a5a5a
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_cmp_ge_f32_e64 s4, |v1|, 0.5
	s_mov_b32 s4, s2
	v_mov_b32_e32 v5, s4
	v_cndmask_b32_e64 v1, -1.0, 1.0, s4
	
	;;#ASMEND
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cmp_cndmask_sgpr_scalar_clobber_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 5
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_cmp_cndmask_sgpr_scalar_clobber_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         v_cmp_cndmask_sgpr_scalar_clobber_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
