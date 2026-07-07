; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_dot4_i32_iu8_signed_kernel,v_dot4_i32_iu8_unsigned_kernel,v_dot4_i32_iu8_clamp_kernel 2>/dev/null | %FileCheck %s

; v_dot4_i32_iu8 signed/unsigned/clamp dot-product lift.
; CHECK-LABEL: define amdgpu_kernel void @v_dot4_i32_iu8_signed_kernel(
; CHECK: sext i8 %{{[^ ]+}} to i64
; CHECK: %dot4_mul{{[0-9]*}} = mul i64
; CHECK: %dot4_acc{{[0-9]*}} = add i64
; CHECK: trunc i64 %dot4_acc{{[0-9]*}} to i32
; CHECK-NOT: @llvm.amdgcn.sudot4
; CHECK-LABEL: define amdgpu_kernel void @v_dot4_i32_iu8_unsigned_kernel(
; CHECK: zext i8 %{{[^ ]+}} to i64
; CHECK-NOT: sext i8
; CHECK: trunc i64 %dot4_acc{{[0-9]*}} to i32
; CHECK-NOT: @llvm.amdgcn.sudot4
; CHECK-LABEL: define amdgpu_kernel void @v_dot4_i32_iu8_clamp_kernel(
; CHECK: icmp slt i64 %{{[^,]+}}, -2147483648
; CHECK: select i1 %{{[^,]+}}, i64 -2147483648
; CHECK: icmp sgt i64 %{{[^,]+}}, 2147483647
; CHECK: select i1 %{{[^,]+}}, i64 2147483647
; CHECK: trunc i64 %dot4_clamp{{[0-9]*}} to i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_dot4_i32_iu8_signed_kernel
	.p2align	8
	.type	v_dot4_i32_iu8_signed_kernel,@function
v_dot4_i32_iu8_signed_kernel:
	s_mov_b64 s[6:7], s[0:1]
	s_load_b128 s[0:3], s[6:7], 0x0
	s_load_b32 s4, s[6:7], 0x10
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s2
	v_mov_b32_e32 v1, s3
	v_mov_b32_e32 v2, s4
	v_dot4_i32_iu8 v3, v0, v1, v2 neg_lo:[1,1,0]
	v_mov_b32_e32 v4, 0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.globl	v_dot4_i32_iu8_unsigned_kernel
	.p2align	8
	.type	v_dot4_i32_iu8_unsigned_kernel,@function
v_dot4_i32_iu8_unsigned_kernel:
	s_mov_b64 s[6:7], s[0:1]
	s_load_b128 s[0:3], s[6:7], 0x0
	s_load_b32 s4, s[6:7], 0x10
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s2
	v_mov_b32_e32 v1, s3
	v_mov_b32_e32 v2, s4
	v_dot4_i32_iu8 v3, v0, v1, v2
	v_mov_b32_e32 v4, 0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.globl	v_dot4_i32_iu8_clamp_kernel
	.p2align	8
	.type	v_dot4_i32_iu8_clamp_kernel,@function
v_dot4_i32_iu8_clamp_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_dot4_i32_iu8 v3, v0, v1, v2 clamp
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_dot4_i32_iu8_signed_kernel
		.amdhsa_kernarg_size 20
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_dot4_i32_iu8_unsigned_kernel
		.amdhsa_kernarg_size 20
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_dot4_i32_iu8_clamp_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:        12, .size:           4, .value_kind:     by_value }
      - { .offset:        16, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .max_flat_workgroup_size: 1024
    .name:           v_dot4_i32_iu8_signed_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_dot4_i32_iu8_signed_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:        12, .size:           4, .value_kind:     by_value }
      - { .offset:        16, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .max_flat_workgroup_size: 1024
    .name:           v_dot4_i32_iu8_unsigned_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_dot4_i32_iu8_unsigned_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_dot4_i32_iu8_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_dot4_i32_iu8_clamp_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
