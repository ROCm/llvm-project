; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; The denormal and rounding modes the kernel descriptor asks for reach the
; lifted function as attributes.
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=fp_mode_kernel | %FileCheck %s --check-prefix=MODE

; A mode the raise cannot carry over and an encoding outside the dispatched
; VOP2 form are both refused rather than mislowered.
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=rounding_kernel,e64_kernel,dpp_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
; REFUSE:      unsupported-floating-point-mode: v_add_f32 [VOP2]
; REFUSE-SAME: f32 rounding mode 1 is unsupported
; REFUSE:      unsupported-instruction-form: v_add_f32 [VOP3]
; REFUSE:      unsupported-instruction-form: v_add_f32 [DPP]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	fp_mode_kernel
	.p2align	8
	.type	fp_mode_kernel,@function
; MODE-LABEL: define amdgpu_kernel void @fp_mode_kernel()
; MODE-SAME: #[[ATTR:[0-9]+]] {
fp_mode_kernel:
; MODE: fadd float
	v_add_f32_e32 v0, v1, v2
	s_endpgm

	.globl	rounding_kernel
	.p2align	8
	.type	rounding_kernel,@function
rounding_kernel:
	v_add_f32_e32 v0, v1, v2
	s_endpgm

	.globl	e64_kernel
	.p2align	8
	.type	e64_kernel,@function
e64_kernel:
	v_add_f32_e64 v0, v1, v2
	s_endpgm

	.globl	dpp_kernel
	.p2align	8
	.type	dpp_kernel,@function
dpp_kernel:
	v_add_f32_dpp v0, v1, v2 row_shr:1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
; MODE: attributes #[[ATTR]] = {
; MODE-SAME: {{.*}}denormal_fpenv(preservesign|ieee,
; MODE-SAME: float: ieee|preservesign)
	.amdhsa_kernel fp_mode_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_denorm_mode_32 2
		.amdhsa_float_denorm_mode_16_64 1
	.end_amdhsa_kernel
	.amdhsa_kernel rounding_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_32 1
	.end_amdhsa_kernel
	.amdhsa_kernel e64_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel dpp_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
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
    .name:           fp_mode_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         fp_mode_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           rounding_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         rounding_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           e64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         e64_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dpp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         dpp_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
