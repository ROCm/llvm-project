; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=v_pk_mad_u16_kernel | %FileCheck %s

; v_pk_mad_u16 packed u16 multiply-add lift.
; CHECK-LABEL: define amdgpu_kernel void @v_pk_mad_u16_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_pk_mad_u16_kernel
	.p2align	8
	.type	v_pk_mad_u16_kernel,@function
v_pk_mad_u16_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x00020001
	v_mov_b32_e32 v1, 0x00040003
	v_mov_b32_e32 v2, 0x00060005
	v_mov_b32_e32 v4, 0
; CHECK: %pk_mad_u16_mul{{[0-9]*}} = mul <2 x i16>
; CHECK: %pk_mad_u16{{[0-9]*}} = add <2 x i16> %pk_mad_u16_mul
; CHECK: bitcast <2 x i16>
; CHECK-NOT: pk_mad_u16_clamp
	v_pk_mad_u16 v3, v0, v1, v2
	s_wait_loadcnt 0x0
	global_store_b32 v4, v3, s[0:1] scale_offset
; CHECK: %pk_mad_u16_a_wide{{[0-9]*}} = zext <2 x i16>
; CHECK: %pk_mad_u16_mul_wide{{[0-9]*}} = mul <2 x i32>
; CHECK: %pk_mad_u16_wide{{[0-9]*}} = add <2 x i32>
; CHECK: %pk_mad_u16_clamp{{[0-9]*}} = call <2 x i32> @llvm.umin.v2i32(
; CHECK: %pk_mad_u16{{[0-9]*}} = trunc <2 x i32>
	v_pk_mad_u16 v3, v0, v1, v2 clamp
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_mad_u16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: v_pk_mad_u16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_pk_mad_u16_kernel.kd
    .vgpr_count: 5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
