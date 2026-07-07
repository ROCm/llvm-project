; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_bcnt1_i32_b32_kernel 2>/dev/null | %FileCheck %s

; s_bcnt1_i32_b32 population-count lift to llvm.ctpop.i32.
; CHECK-LABEL: define amdgpu_kernel void @s_bcnt1_i32_b32_kernel(
; CHECK: call i32 @llvm.ctpop.i32(i32 {{.*}})
; CHECK: declare {{.*}}i32 @llvm.ctpop.i32(i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_bcnt1_i32_b32_kernel
	.p2align	8
	.type	s_bcnt1_i32_b32_kernel,@function
s_bcnt1_i32_b32_kernel:
	s_bcnt1_i32_b32 s1, s0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_bcnt1_i32_b32_kernel
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
    .name:           s_bcnt1_i32_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_bcnt1_i32_b32_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
