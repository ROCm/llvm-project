; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_andn2_mask_shadow_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_andn2/orn2/nand/nor/xnor wave-mask boolean lowering.
; CHECK-LABEL: define amdgpu_kernel void @s_andn2_mask_shadow_kernel(
; CHECK: %wave_mask_andn2 = and i1
; CHECK: %cndmask = select i1 %wave_mask_andn2, i32 1, i32 0
; CHECK: %wave_mask_orn2 = or i1
; CHECK: %cndmask{{[0-9]*}} = select i1 %wave_mask_orn2, i32 1, i32 0
; CHECK: %wave_mask_nand = xor i1
; CHECK: %cndmask{{[0-9]*}} = select i1 %wave_mask_nand, i32 1, i32 0
; CHECK: %wave_mask_nor = xor i1
; CHECK: %cndmask{{[0-9]*}} = select i1 %wave_mask_nor, i32 1, i32 0
; CHECK: %wave_mask_xnor = xor i1
; CHECK: %cndmask{{[0-9]*}} = select i1 %wave_mask_xnor, i32 1, i32 0
; CHECK: [[ALIAS:%wave_mask_andn2[0-9]*]] = and i1
; CHECK: %cndmask{{[0-9]*}} = select i1 [[ALIAS]], i32 1, i32 0

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_andn2_mask_shadow_kernel
	.p2align	8
	.type	s_andn2_mask_shadow_kernel,@function
s_andn2_mask_shadow_kernel:
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_lt_u32_e64 s3, v0, 8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 1
	s_andn2_b32 vcc_lo, exec_lo, s2
	v_cndmask_b32_e32 v1, v2, v3
	s_orn2_b32 vcc_lo, s2, s3
	v_cndmask_b32_e32 v1, v2, v3
	s_nand_b32 vcc_lo, s2, s3
	v_cndmask_b32_e32 v1, v2, v3
	s_nor_b32 vcc_lo, s2, s3
	v_cndmask_b32_e32 v1, v2, v3
	s_xnor_b32 vcc_lo, s2, s3
	v_cndmask_b32_e32 v1, v2, v3
	s_andn2_b32 s2, s2, s3
	v_cndmask_b32_e64 v1, v2, v3, s2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_andn2_mask_shadow_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 64
    .name: s_andn2_mask_shadow_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 4
    .symbol: s_andn2_mask_shadow_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
