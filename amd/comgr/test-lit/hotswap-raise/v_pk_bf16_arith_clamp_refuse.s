; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_pk_bf16_add_clamp_refuse_kernel,v_pk_bf16_mul_clamp_refuse_kernel,v_pk_bf16_fma_clamp_refuse_kernel 2>&1 | %FileCheck %s

; v_pk_add/mul/fma_bf16 clamp-bit refusal.
; CHECK: v_pk_add_bf16 has a nonzero clamp bit
; CHECK-SAME: overflow-mode semantics are not modelled
; CHECK: v_pk_mul_bf16 has a nonzero clamp bit
; CHECK-SAME: overflow-mode semantics are not modelled
; CHECK: v_pk_fma_bf16 has a nonzero clamp bit
; CHECK-SAME: overflow-mode semantics are not modelled

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_pk_bf16_add_clamp_refuse_kernel
	.p2align	8
	.type	v_pk_bf16_add_clamp_refuse_kernel,@function
v_pk_bf16_add_clamp_refuse_kernel:
	v_mov_b32_e32 v1, 0x3f803f80
	v_mov_b32_e32 v2, 0x40004000
	v_pk_add_bf16 v3, v1, v2 clamp
	s_endpgm

	.globl	v_pk_bf16_mul_clamp_refuse_kernel
	.p2align	8
	.type	v_pk_bf16_mul_clamp_refuse_kernel,@function
v_pk_bf16_mul_clamp_refuse_kernel:
	v_mov_b32_e32 v1, 0x3f803f80
	v_mov_b32_e32 v2, 0x40004000
	v_pk_mul_bf16 v3, v1, v2 clamp
	s_endpgm

	.globl	v_pk_bf16_fma_clamp_refuse_kernel
	.p2align	8
	.type	v_pk_bf16_fma_clamp_refuse_kernel,@function
v_pk_bf16_fma_clamp_refuse_kernel:
	v_mov_b32_e32 v1, 0x3f803f80
	v_mov_b32_e32 v2, 0x40004000
	v_mov_b32_e32 v3, 0x40404040
	v_pk_fma_bf16 v4, v1, v2, v3 clamp
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_bf16_add_clamp_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_bf16_mul_clamp_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_bf16_fma_clamp_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 0
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
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: v_pk_bf16_add_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: v_pk_bf16_add_clamp_refuse_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: v_pk_bf16_mul_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: v_pk_bf16_mul_clamp_refuse_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: v_pk_bf16_fma_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: v_pk_bf16_fma_clamp_refuse_kernel.kd
    .vgpr_count: 5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
