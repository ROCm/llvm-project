; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_scalar_f32_rounding_kernel 2>/dev/null | %FileCheck %s

; s_ceil/floor/trunc/rndne_f32 scalar rounding lifted to llvm rounding intrinsics.
; CHECK-LABEL: define amdgpu_kernel void @s_scalar_f32_rounding_kernel(
; CHECK-DAG: call float @llvm.ceil.f32(float {{.*}})
; CHECK-DAG: call float @llvm.floor.f32(float {{.*}})
; CHECK-DAG: call float @llvm.trunc.f32(float {{.*}})
; CHECK-DAG: call float @llvm.roundeven.f32(float {{.*}})
; CHECK-DAG: declare {{.*}}float @llvm.ceil.f32(float)
; CHECK-DAG: declare {{.*}}float @llvm.floor.f32(float)
; CHECK-DAG: declare {{.*}}float @llvm.trunc.f32(float)
; CHECK-DAG: declare {{.*}}float @llvm.roundeven.f32(float)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_scalar_f32_rounding_kernel
	.p2align	8
	.type	s_scalar_f32_rounding_kernel,@function
s_scalar_f32_rounding_kernel:
	s_ceil_f32 s0, s0
	s_floor_f32 s1, s0
	s_trunc_f32 s2, s0
	s_rndne_f32 s3, s0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_scalar_f32_rounding_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
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
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           s_scalar_f32_rounding_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_scalar_f32_rounding_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
