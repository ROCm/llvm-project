; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vopd_dual_issue | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_dual_issue
	.p2align	8
	.type	vopd_dual_issue,@function
; CHECK-LABEL: define amdgpu_kernel void @vopd_dual_issue(
vopd_dual_issue:
; CHECK: fadd float
; CHECK: fadd float
	v_dual_add_f32 v4, v0, v1 :: v_dual_add_f32 v5, v2, v3
; CHECK: fmul float
	v_dual_mul_f32 v2, v0, v1 :: v_dual_mov_b32 v3, v0
; CHECK: fsub float
	v_dual_sub_f32 v2, v0, v1 :: v_dual_mov_b32 v3, v0
; CHECK: call float @llvm.maximumnum.f32
; CHECK: call float @llvm.minimumnum.f32
	v_dual_max_num_f32 v2, v0, v1 :: v_dual_min_num_f32 v3, v2, v3
; CHECK: call float @llvm.fma.f32
	v_dual_fma_f32 v6, v4, v2, v1 :: v_dual_mov_b32 v7, v5
; CHECK: call float @llvm.fma.f32
	v_dual_fmac_f32 v2, v0, v1 :: v_dual_mov_b32 v3, v0
; CHECK: call float @llvm.fma.f32
	v_dual_mov_b32 v3, v0 :: v_dual_fmaak_f32 v2, v1, v2, 0x3f800000
; CHECK: call float @llvm.fma.f32
	v_dual_mov_b32 v3, v0 :: v_dual_fmamk_f32 v2, v1, 0x3f800000, v2
; CHECK: shl i32
; CHECK: shl i32
	v_dual_lshlrev_b32 v2, 4, v0 :: v_dual_lshlrev_b32 v3, 2, v0
; CHECK: lshr i32
	v_dual_mov_b32 v4, v2 :: v_dual_lshrrev_b32 v5, 4, v3
; CHECK: sub i32
	v_dual_mov_b32 v4, v2 :: v_dual_sub_nc_u32 v5, 4, v3
; CHECK: ashr i32
	v_dual_mov_b32 v2, 1.0 :: v_dual_ashrrev_i32 v3, 31, v0
; CHECK: call i32 @llvm.smax.i32
	v_dual_mov_b32 v5, v5 :: v_dual_max_i32 v4, 0, v6
; CHECK: add i32
; CHECK: add i32
	v_dual_add_nc_u32 v2, 1, v0 :: v_dual_add_nc_u32 v3, 2, v0
; CHECK: xor i32
	v_dual_lshlrev_b32 v2, 1, v0 :: v_dual_bitop2_b32 v3, v0, v1 bitop3:0x14
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_dual_issue
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 1
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
    .name:           vopd_dual_issue
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vopd_dual_issue.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
