; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir=s_bfe_i64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_bfe_i64 signed bitfield-extract (shl/ashr) lift.
; CHECK-LABEL: define amdgpu_kernel void @s_bfe_i64_kernel(
; CHECK-DAG: [[SHL:%[^ ,]+]] = shl i64 %{{[^,]+}}, 48
; CHECK-DAG: %sbfe_i64 = ashr i64 [[SHL]], 56
; CHECK-NOT: lshr i64 [[SHL]], 56

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	s_bfe_i64_kernel
	.p2align	8
	.type	s_bfe_i64_kernel,@function
s_bfe_i64_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	s_bfe_i64 s[2:3], s[0:1], 0x80008
	v_mov_b32_e32 v0, s2
	v_mov_b32_e32 v1, s3
	v_mov_b32_e32 v2, 0
	global_store_dwordx2 v2, v[0:1], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_bfe_i64_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           s_bfe_i64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_bfe_i64_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
