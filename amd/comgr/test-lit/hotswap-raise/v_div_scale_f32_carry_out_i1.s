; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=v_div_scale_f32_carry_out_i1_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression guard for `v_div_scale_f32`'s per-lane i1 carry-out under
; wave32-source -> wave64-target WaveNative cross-widening, when the
; SDST of the scale is a plain SGPR (not `vcc_lo`) and a downstream
; `s_mov_b32 vcc_lo, sN` restores it before `v_div_fmas_f32`.
;
; Pre-fix `V_DIV_SCALE_F32` handler stored the carry via an ad-hoc
; `zext i1 -> i32` + `storeSGPR32` when the SDST was a plain SGPR,
; leaving per-lane 0/1 in each lane's alloca slot. The matching
; `s_mov_b32 vcc_lo, sN` restore reads the SGPR back through
; `extractLaneBitFromWaveMask`, which interprets the per-lane value
; as a wave mask -- recovering bit 0 only for target lanes 0 and
; `W_src` (replication-aliased) and silently dropping the carry on
; every other target lane. The downstream `v_div_fmas_f32` then
; consumes the wrong predicate on lanes 32..63 of the widened wave,
; producing wrong divide results for those lanes whenever the carry
; matters (denormal-near or sign-flipping divides).
;
; The fix routes the carry through the shared `writeCarryOutI1`
; helper (already used by v_add_co / v_sub_co): ballot the per-lane
; i1 to a source-width wave mask via `ballotI1ToWidth` (so the SGPR
; alloca holds a proper wave mask, not per-lane 0/1), and cache the
; original i1 via `recordSgprWaveMaskI1` so same-BB consumers
; resolve to the SSA i1 directly without the lossy extract round-
; trip.
;
; The kernel below is the canonical IEEE-divide chain
; (v_div_scale_f32 + v_rcp_f32 + Newton fma's + v_div_fmas_f32 +
; v_div_fixup_f32) that hipcc emits for a register-only fdiv
; `a/b`. Crucially the scale-denominator and scale-numerator carries
; are routed through PLAIN SGPRS (`s2`, `s3`) rather than `vcc_lo`,
; and `v_div_fmas_f32`'s implicit-VCC input is restored via
; `s_mov_b32 vcc_lo, s3` -- the exact SGPR-bounce shape the audit
; targets.

; The scale-numerator divscale produces { float, i1 } whose i1 is
; the per-lane carry-out destined for v_div_fmas_f32.  Anchor the
; (1.0 / true)-flag scale call and capture its extractvalue.  Under
; the pre-fix handler this i1 was zext'd to i32 and stored into the
; SDST SGPR's alloca per lane; under the fix it is balloted to a
; source-width wave mask AND cached as i1 via recordSgprWaveMaskI1
; for the same-BB consumer below.

; The lifted v_div_fmas_f32 must consume the captured carry directly -- no
; extractLaneBitFromWaveMask / ballot / wn_mask_lane_i1 round-trip
; on the path from divscale to divfmas.  Pre-fix the cache wasn't
; populated, the `s_mov_b32 vcc_lo, s3` restore hit the lossy
; SGPR-alloca extract path, and the i1 reaching divfmas was
; structurally different from the carry (correct only on target
; lanes 0 and W_src by replication-aliasing).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_div_scale_f32_carry_out_i1_kernel
	.p2align	8
	.type	v_div_scale_f32_carry_out_i1_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_div_scale_f32_carry_out_i1_kernel(
v_div_scale_f32_carry_out_i1_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	flat_load_b32 v1, v0, s[6:7] scope:SCOPE_SYS
	flat_load_b32 v2, v0, s[6:7] offset:64 scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	; Scale-denominator: carry-out to plain SGPR s2.
	v_div_scale_f32 v3, s2, v2, v2, v1
	; Scale-numerator: carry-out to plain SGPR s3 (this carry is
	; the one consumed by v_div_fmas_f32 below via the SGPR-bounce
	; through `s_mov_b32 vcc_lo, s3`).
	; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{[^,]+}}, float %{{[^,]+}}, i1 true)
	; CHECK: %[[CARRY:[^ ]+]] = extractvalue { float, i1 } %{{[^,]+}}, 1
	v_div_scale_f32 v4, s3, v1, v2, v1
	v_rcp_f32_e32 v5, v3
	v_nop
	v_fma_f32 v6, -v3, v5, 1.0
	v_fmac_f32_e32 v5, v6, v5
	v_mul_f32_e32 v7, v4, v5
	v_fma_f32 v8, -v3, v7, v4
	v_fmac_f32_e32 v7, v8, v5
	v_fma_f32 v3, -v3, v7, v4
	; Restore scale-numerator carry from SGPR s3 to vcc_lo so
	; v_div_fmas_f32 picks it up as its implicit VCC input.
	s_mov_b32 vcc_lo, s3
	; CHECK: call float @llvm.amdgcn.div.fmas.f32(float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, i1 %[[CARRY]])
	v_div_fmas_f32 v3, v3, v5, v7
	v_div_fixup_f32 v1, v3, v2, v1
	global_store_b32 v0, v1, s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_div_scale_f32_carry_out_i1_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 3
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
    .name:           v_div_scale_f32_carry_out_i1_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_div_scale_f32_carry_out_i1_kernel.kd
    .vgpr_count:     10
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
