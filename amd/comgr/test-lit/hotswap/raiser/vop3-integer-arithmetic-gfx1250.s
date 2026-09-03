; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vop3_integer_arithmetic | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vop3_integer_arithmetic
	.p2align	8
	.type	vop3_integer_arithmetic,@function
; CHECK-LABEL: define amdgpu_kernel void @vop3_integer_arithmetic(
vop3_integer_arithmetic:
	v_mov_b32_e64 v36, v0
	v_mov_b64_e32 v[38:39], v[0:1]
	v_mov_b64_e64 v[40:41], v[2:3]
; CHECK: = add i32
	v_add_nc_i32 v4, v0, v1
; CHECK: call i32 @llvm.ssub.sat.i32
	v_sub_nc_i32 v5, v0, v1 clamp
; CHECK: = mul i32
	v_mul_lo_u32 v6, v0, v1
; CHECK: = ashr i64
	v_mul_hi_i32 v7, v0, v1
; CHECK: = lshr i64
	v_mul_hi_u32 v8, v0, v1
; CHECK: = add i32
	v_mad_u32 v9, v0, v1, v2
; CHECK: = add i32
	v_add3_u32 v10, v0, v1, v2
; CHECK: call i32 @llvm.sadd.sat.i32
; CHECK: call i32 @llvm.smin.i32
; CHECK: call i32 @llvm.smax.i32({{.*}}, i32 0)
	v_add_min_i32 v11, v0, v1, v2 clamp
; CHECK: call i32 @llvm.umax.i32
	v_max3_u32 v12, v0, v1, v2
; CHECK: call i32 @llvm.smin.i32
; CHECK: call i32 @llvm.smax.i32
	v_med3_i32 v13, v0, v1, v2
; CHECK: call i32 @llvm.umin.i32
; CHECK: call i32 @llvm.umax.i32
	v_minmax_u32 v14, v0, v1, v2
; CHECK: call i64 @llvm.smin.i64
	v_min_i64 v[16:17], v[0:1], v[2:3]
; CHECK: = sext i32
; CHECK: = mul i64
; CHECK: = add i64
	v_mad_nc_i64_i32 v[18:19], v0, v1, v[2:3]
; CHECK: = zext i32
; CHECK: call i64 @llvm.uadd.sat.i64
	v_mad_nc_u64_u32 v[20:21], v0, v1, v[2:3] clamp
; CHECK: = mul i64
; CHECK: call i64 @llvm.smax.i64
; CHECK: call i64 @llvm.smin.i64
	v_mul_i32_i24 v22, v0, v1 clamp
; CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32
; CHECK: call i32 @llvm.uadd.sat.i32
	v_add_co_u32 v24, s2, v0, v1 clamp
; CHECK: = zext i32 {{.*}} to i33
; CHECK: = add i33
; CHECK: = add i33
; CHECK: = lshr i33
	v_add_co_ci_u32 v25, s4, v0, v1, s2
; CHECK: = zext i32 {{.*}} to i33
; CHECK: = sub i33
; CHECK: = sub i33
; CHECK: = lshr i33
; CHECK: call i32 @llvm.usub.sat.i32
	v_sub_co_ci_u32 v26, vcc_lo, v0, v1, s4 clamp
; CHECK: call { i64, i1 } @llvm.uadd.with.overflow.i64
	v_mad_co_u64_u32 v[28:29], s6, v0, v1, v[2:3]
; CHECK: call { i64, i1 } @llvm.uadd.with.overflow.i64
; CHECK: icmp slt i64
; CHECK: xor i1
; CHECK: call i64 @llvm.sadd.sat.i64
	v_mad_co_i64_i32 v[30:31], vcc_lo, v0, v1, v[2:3] clamp
; CHECK: call i64 @llvm.uadd.sat.i64
	v_add_nc_u64 v[32:33], v[0:1], v[2:3] clamp
; CHECK: call i64 @llvm.usub.sat.i64
	v_sub_nc_u64 v[34:35], v[0:1], v[2:3] clamp
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop3_integer_arithmetic
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 42
		.amdhsa_next_free_sgpr 7
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
    .sgpr_count:     7
    .symbol:         vop3_integer_arithmetic.kd
    .vgpr_count:     42
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
