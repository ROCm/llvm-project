; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>/dev/null | %FileCheck %s

; v_ceil_f64 (with neg/abs mods) ceil.f64 lift.
; CHECK-LABEL: define amdgpu_kernel void @v_ceil_f64_kernel(
; CHECK: %ceil = call double @llvm.ceil.f64(double %{{[^,]+}})
; CHECK: %neg = fneg double %{{[^,]+}}
; CHECK: %ceil{{[0-9]*}} = call double @llvm.ceil.f64(double %neg)
; CHECK: %abs = call double @llvm.fabs.f64(double %{{[^,]+}})
; CHECK: %ceil{{[0-9]*}} = call double @llvm.ceil.f64(double %abs)
; CHECK-NOT: call float @llvm.ceil.f32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	v_ceil_f64_kernel
	.p2align	8
	.type	v_ceil_f64_kernel,@function
v_ceil_f64_kernel:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, s3
	v_ceil_f64 v[0:1], v[2:3]
	v_ceil_f64_e64 v[6:7], -v[2:3]
	v_ceil_f64_e64 v[8:9], |v[2:3]|
	v_mov_b32_e32 v4, 0
	global_store_dwordx2 v4, v[0:1], s[0:1]
	global_store_dwordx2 v4, v[6:7], s[0:1]
	global_store_dwordx2 v4, v[8:9], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_ceil_f64_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           8, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_ceil_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_ceil_f64_kernel.kd
    .vgpr_count:     10
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
