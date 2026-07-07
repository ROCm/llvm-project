; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s

; v_cvt_f16_i16/i16_f16 sitofp/fptosi lo/hi-half packing.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_f16_i16_kernel
	.p2align	8
	.type	v_cvt_f16_i16_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_cvt_f16_i16_kernel(
v_cvt_f16_i16_kernel:
; CHECK: trunc i32 {{.*}} to i16
; CHECK: sitofp i16 {{.*}} to half
; CHECK: %cvt_f16_i16_merge_lo{{[0-9]*}} = or {{(disjoint )?}}i32
	v_cvt_f16_i16 v0.l, v1.l
; CHECK: lshr i32 {{.*}}, 16
; CHECK: trunc i32 {{.*}} to i16
; CHECK: sitofp i16 {{.*}} to half
; CHECK: shl i32 %{{[^,]+}}, 16
; CHECK: %cvt_f16_i16_merge_hi{{[0-9]*}} = or {{(disjoint )?}}i32
; CHECK-NOT: uitofp
	v_cvt_f16_i16_e64 v0.h, v1.h
	s_endpgm

	.globl	v_cvt_i16_f16_kernel
	.p2align	8
	.type	v_cvt_i16_f16_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_cvt_i16_f16_kernel(
v_cvt_i16_f16_kernel:
; CHECK: trunc i32 {{.*}} to i16
; CHECK: bitcast i16 {{.*}} to half
; CHECK: call i16 @llvm.fptosi.sat.i16.f16(half %{{[^)]+}})
; CHECK: %cvt_i16_f16_merge_lo{{[0-9]*}} = or {{(disjoint )?}}i32
	v_cvt_i16_f16 v2.l, v3.l
; CHECK: lshr i32 {{.*}}, 16
; CHECK: trunc i32 {{.*}} to i16
; CHECK: bitcast i16 {{.*}} to half
; CHECK: call i16 @llvm.fptosi.sat.i16.f16(half %{{[^)]+}})
; CHECK: shl i32 %{{[^,]+}}, 16
; CHECK: %cvt_i16_f16_merge_hi{{[0-9]*}} = or {{(disjoint )?}}i32
; CHECK-NOT: @llvm.fptoui.sat
	v_cvt_i16_f16_e64 v2.h, v3.h
	s_endpgm

	.globl	v_cvt_f16_u16_kernel
	.p2align	8
	.type	v_cvt_f16_u16_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_cvt_f16_u16_kernel(
v_cvt_f16_u16_kernel:
; CHECK: trunc i32 {{.*}} to i16
; CHECK: uitofp i16 {{.*}} to half
; CHECK: %cvt_f16_u16_merge_lo{{[0-9]*}} = or {{(disjoint )?}}i32
	v_cvt_f16_u16 v0.l, v1.l
; CHECK: lshr i32 {{.*}}, 16
; CHECK: trunc i32 {{.*}} to i16
; CHECK: uitofp i16 {{.*}} to half
; CHECK: shl i32 %{{[^,]+}}, 16
; CHECK: %cvt_f16_u16_merge_hi{{[0-9]*}} = or {{(disjoint )?}}i32
; CHECK-NOT: sitofp
	v_cvt_f16_u16_e64 v0.h, v1.h
	s_endpgm

	.globl	v_cvt_u16_f16_kernel
	.p2align	8
	.type	v_cvt_u16_f16_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_cvt_u16_f16_kernel(
v_cvt_u16_f16_kernel:
; CHECK: trunc i32 {{.*}} to i16
; CHECK: bitcast i16 {{.*}} to half
; CHECK: call i16 @llvm.fptoui.sat.i16.f16(half %{{[^)]+}})
; CHECK: %cvt_u16_f16_merge_lo{{[0-9]*}} = or {{(disjoint )?}}i32
	v_cvt_u16_f16 v2.l, v3.l
; CHECK: lshr i32 {{.*}}, 16
; CHECK: trunc i32 {{.*}} to i16
; CHECK: bitcast i16 {{.*}} to half
; CHECK: call i16 @llvm.fptoui.sat.i16.f16(half %{{[^)]+}})
; CHECK: shl i32 %{{[^,]+}}, 16
; CHECK: %cvt_u16_f16_merge_hi{{[0-9]*}} = or {{(disjoint )?}}i32
; CHECK-NOT: @llvm.fptosi.sat
	v_cvt_u16_f16_e64 v2.h, v3.h
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_f16_i16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_i16_f16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_f16_u16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_u16_f16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
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
    .name:           v_cvt_f16_i16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_f16_i16_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_i16_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_i16_f16_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_f16_u16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_f16_u16_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_u16_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_u16_f16_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
