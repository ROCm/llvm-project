; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_f16_basic_kernel,v_fma_f16_opsel_kernel,v_fma_f16_dsthi_kernel,v_fma_f16_neg_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_fma_f16 op_sel/dsthi/neg-abs packed half FMA lift.
; CHECK-LABEL: define amdgpu_kernel void @v_fma_f16_basic_kernel(
; CHECK-DAG: trunc i32 {{.*}} to i16
; CHECK-DAG: bitcast i16 {{.*}} to half
; CHECK: %fma_f16 = call half @llvm.fma.f16(
; CHECK: bitcast half %fma_f16 to i16
; CHECK: zext i16 {{.*}} to i32
; CHECK: and i32 {{.*}}, -65536
; CHECK: %f16_merge_lo = or i32
; CHECK-NOT: unsupported instruction
; CHECK-LABEL: define amdgpu_kernel void @v_fma_f16_opsel_kernel(
; CHECK-DAG: %f16_src_hi = lshr i32 {{.*}}, 16
; CHECK-DAG: %f16_src_hi{{[0-9]+}} = lshr i32 {{.*}}, 16
; CHECK: %fma_f16 = call half @llvm.fma.f16(
; CHECK: %f16_merge_lo = or i32
; CHECK-NOT: unsupported instruction
; CHECK-LABEL: define amdgpu_kernel void @v_fma_f16_dsthi_kernel(
; CHECK: %fma_f16 = call half @llvm.fma.f16(
; CHECK: bitcast half %fma_f16 to i16
; CHECK: zext i16 {{.*}} to i32
; CHECK: and i32 {{.*}}, 65535
; CHECK: shl i32 {{.*}}, 16
; CHECK: %f16_merge_hi = or i32
; CHECK-NOT: unsupported instruction
; CHECK-LABEL: define amdgpu_kernel void @v_fma_f16_neg_kernel(
; CHECK: %neg_f16 = fneg half
; CHECK: %abs_f16 = call half @llvm.fabs.f16(half
; CHECK: %fma_f16 = call half @llvm.fma.f16(half %neg_f16, half {{.*}}, half %abs_f16)
; CHECK-NOT: unsupported instruction

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	v_fma_f16_basic_kernel
	.p2align	8
	.type	v_fma_f16_basic_kernel,@function
v_fma_f16_basic_kernel:
	v_fma_f16 v0, v1, v2, v3
	s_endpgm

	.globl	v_fma_f16_opsel_kernel
	.p2align	8
	.type	v_fma_f16_opsel_kernel,@function
v_fma_f16_opsel_kernel:
	v_fma_f16 v0, v1, v2, v3 op_sel:[1,0,1,0]
	s_endpgm

	.globl	v_fma_f16_dsthi_kernel
	.p2align	8
	.type	v_fma_f16_dsthi_kernel,@function
v_fma_f16_dsthi_kernel:
	v_fma_f16 v0, v1, v2, v3 op_sel:[0,0,0,1]
	s_endpgm

	.globl	v_fma_f16_neg_kernel
	.p2align	8
	.type	v_fma_f16_neg_kernel,@function
v_fma_f16_neg_kernel:
	v_fma_f16 v0, -v1, v2, |v3|
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_basic_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_opsel_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_dsthi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_neg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_basic_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_basic_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_opsel_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_opsel_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_dsthi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_dsthi_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_neg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_neg_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
