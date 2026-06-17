; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=div_carry_sgpr_kernel | %FileCheck %s
; RUN: raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=div_carry_sgpr_xbb_kernel | %FileCheck %s --check-prefix=XBB
;
; Wave32-source -> wave64-target lift of the IEEE fdiv carry chain
; v_div_scale_f32 (flag -> SGPR) ; s_mov_b32 vcc_lo, sN ; v_div_fmas_f32.
; The carry is per-lane, so the s_mov must commit the producer's per-lane shadow
; into VCC rather than re-widening the truncated SGPR. div_carry_sgpr_kernel has
; the producer and consumer in the same basic block (the fresh per-lane i1 is
; consumed directly); div_carry_sgpr_xbb_kernel separates them with a branch so
; the same-BB cache is cleared and the consumer must read the memory-backed
; shadow. See hotswap/docs/sgpr-wave-mask-translation.md.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	div_carry_sgpr_kernel
	.p2align	8
	.type	div_carry_sgpr_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @div_carry_sgpr_kernel(
div_carry_sgpr_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	flat_load_b32 v1, v0, s[4:5] scale_offset
	flat_load_b32 v2, v0, s[6:7] scale_offset
	s_wait_loadcnt_dscnt 0x0
; CHECK: %divscale = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{[^,]+}}, float %{{[^,]+}}, i1 false)
; CHECK: [[CARRY:%[0-9]+]] = extractvalue { float, i1 } %divscale, 1
	v_div_scale_f32 v3, s2, v2, v2, v1
	v_rcp_f32_e32 v4, v3
	v_div_scale_f32 v5, vcc_lo, v1, v2, v1
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_mul_f32_e32 v7, v5, v4
	v_fma_f32 v8, -v3, v7, v5
	v_fmac_f32_e32 v7, v8, v4
	v_fma_f32 v3, -v3, v7, v5
	s_mov_b32 vcc_lo, s2
; The fmas consumes the per-lane carry directly, not a re-widened SGPR mask:
; CHECK-NOT: wn_mask_widen
; CHECK-NOT: sgpr_mask_shadow_sel
; CHECK: call float @llvm.amdgcn.div.fmas.f32(float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, i1 [[CARRY]])
	v_div_fmas_f32 v3, v3, v4, v7
	v_div_fixup_f32 v1, v3, v2, v1
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm

	.globl	div_carry_sgpr_xbb_kernel
	.p2align	8
	.type	div_carry_sgpr_xbb_kernel,@function
; XBB-LABEL: define amdgpu_kernel void @div_carry_sgpr_xbb_kernel(
div_carry_sgpr_xbb_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	flat_load_b32 v1, v0, s[4:5] scale_offset
	flat_load_b32 v2, v0, s[6:7] scale_offset
	s_wait_loadcnt_dscnt 0x0
	v_div_scale_f32 v3, s2, v2, v2, v1
	v_rcp_f32_e32 v4, v3
	v_div_scale_f32 v5, vcc_lo, v1, v2, v1
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_mul_f32_e32 v7, v5, v4
	v_fma_f32 v8, -v3, v7, v5
	v_fmac_f32_e32 v7, v8, v4
	v_fma_f32 v3, -v3, v7, v5
	s_branch .Lnext
.Lnext:
; The consumer is in a separate block, so the same-BB i1 cache is gone and it
; must read the memory-backed shadow; the fmas consumes that selected carry:
; XBB: %sgpr_mask_shadow_sel = select i1
; XBB: call float @llvm.amdgcn.div.fmas.f32(float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, i1 %sgpr_mask_shadow_sel)
; The producer stored the per-lane carry to the SGPR as a wave-mask ballot, not
; a zext of the current lane's bit:
; XBB: %div_scale_flag_ballot = call i64 @llvm.amdgcn.ballot.i64
	s_mov_b32 vcc_lo, s2
	v_div_fmas_f32 v3, v3, v4, v7
	v_div_fixup_f32 v1, v3, v2, v1
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel div_carry_sgpr_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 3
	.end_amdhsa_kernel
	.amdhsa_kernel div_carry_sgpr_xbb_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 3
	.end_amdhsa_kernel
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
    .name:           div_carry_sgpr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         div_carry_sgpr_kernel.kd
    .vgpr_count:     9
    .wavefront_size: 32
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
    .name:           div_carry_sgpr_xbb_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         div_carry_sgpr_xbb_kernel.kd
    .vgpr_count:     9
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
