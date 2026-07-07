; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_mad_u16_kernel | %FileCheck %s

; v_mad_u16 16-bit multiply-add lift.
; CHECK-LABEL: define amdgpu_kernel void @v_mad_u16_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_mad_u16_kernel
	.p2align	8
	.type	v_mad_u16_kernel,@function
v_mad_u16_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v0, 0x00020001
	v_mov_b32_e32 v1, 0x00040003
	v_mov_b32_e32 v2, 0x00060005
	v_mov_b32_e32 v6, 0
; CHECK: %mad_u16_mul{{[0-9]*}} = mul i16
; CHECK: %mad_u16{{[0-9]*}} = add i16 %mad_u16_mul{{[0-9]*}},
; CHECK: %mad_u16_merge_lo{{[0-9]*}} = or i32
; CHECK-NOT: mad_u16_clamp
	v_mad_u16 v3, v0, v1, v2
	global_store_b32 v6, v3, s[0:1] scale_offset
; CHECK: %mad_u16_a_wide{{[0-9]*}} = zext i16
; CHECK: %mad_u16_b_wide{{[0-9]*}} = zext i16
; CHECK: %mad_u16_c_wide{{[0-9]*}} = zext i16
; CHECK: %mad_u16_mul_wide{{[0-9]*}} = mul i32
; CHECK: %mad_u16_wide{{[0-9]*}} = add i32
; CHECK: %mad_u16_clamp{{[0-9]*}} = call i32 @llvm.umin.i32(i32 %mad_u16_wide{{[0-9]*}}, i32 65535)
; CHECK: trunc i32 %mad_u16_clamp{{[0-9]*}} to i16
	v_mad_u16 v4, v0, v1, v2 clamp
	global_store_b32 v6, v4, s[0:1] scale_offset
; CHECK: lshr i32 %{{[^,]+}}, 16
; CHECK: %mad_u16_mul{{[0-9]*}} = mul i16
; CHECK: %mad_u16{{[0-9]*}} = add i16 %mad_u16_mul{{[0-9]*}},
; CHECK: shl i32 %{{[^,]+}}, 16
; CHECK: %mad_u16_merge_hi{{[0-9]*}} = or i32
	v_mad_u16 v5, v0, v1, v2 op_sel:[1,1,1,1]
	global_store_b32 v6, v5, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_mad_u16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: v_mad_u16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_mad_u16_kernel.kd
    .vgpr_count: 7
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
