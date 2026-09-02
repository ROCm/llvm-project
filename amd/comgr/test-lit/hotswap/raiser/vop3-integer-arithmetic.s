; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=vop3_integer_arithmetic \
; RUN:   | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	vop3_integer_arithmetic
	.p2align	8
	.type	vop3_integer_arithmetic,@function
; CHECK-LABEL: define amdgpu_kernel void @vop3_integer_arithmetic(
vop3_integer_arithmetic:
; CHECK: call i32 @llvm.sadd.sat.i32
	v_add_i32_e64 v4, v0, v1 clamp
; CHECK: call i32 @llvm.ssub.sat.i32
	v_sub_i32_e64 v5, v0, v1 clamp
; CHECK: = mul i64
; CHECK: call i64 @llvm.smax.i64
; CHECK: call i64 @llvm.smin.i64
	v_mad_i32_i24 v6, v0, v1, v2 clamp
; CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32
	v_add_co_u32_e64 v7, vcc, v0, v1
; CHECK: call { i64, i1 } @llvm.uadd.with.overflow.i64
	v_mad_u64_u32 v[10:11], vcc, v0, v1, v[2:3]
; CHECK: call { i64, i1 } @llvm.uadd.with.overflow.i64
; CHECK: icmp slt i64
; CHECK: xor i1
	v_mad_i64_i32 v[12:13], vcc, v0, v1, v[2:3]
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop3_integer_arithmetic
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 14
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 16
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop3_integer_arithmetic
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vop3_integer_arithmetic.kd
    .vgpr_count:     14
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
