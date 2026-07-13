; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_and_b16_kernel,v_or_b16_kernel,v_xor_b16_kernel,v_not_b16_kernel \
; RUN:     2>/dev/null | %FileCheck %s

; Lift test for gfx1250 true16 16-bit bitwise ops. AND/OR/XOR are two-source
; with op_sel half select on src0/src1/dst; NOT is single-source. Each op works
; on the selected 16-bit half and merges back into the selected dst half,
; preserving the other half (RDNA3+ true16). This fixture pins the lo/lo shape.

; CHECK-LABEL: define amdgpu_kernel void @v_and_b16_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_and_b16_kernel
	.p2align	8
	.type	v_and_b16_kernel,@function
v_and_b16_kernel:
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: %and_b16 = and i16
	; CHECK: zext i16 %and_b16 to i32
	; CHECK: and i32 {{.+}}, -65536
	; CHECK: %logic_b16_merge_lo = or i32
	v_and_b16 v0.l, v1.l, v2.l
	s_endpgm

; CHECK-LABEL: define amdgpu_kernel void @v_or_b16_kernel(
	.globl	v_or_b16_kernel
	.p2align	8
	.type	v_or_b16_kernel,@function
v_or_b16_kernel:
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: %or_b16 = or i16
	; CHECK: zext i16 %or_b16 to i32
	; CHECK: and i32 {{.+}}, -65536
	; CHECK: %logic_b16_merge_lo = or i32
	v_or_b16 v0.l, v1.l, v2.l
	s_endpgm

; CHECK-LABEL: define amdgpu_kernel void @v_xor_b16_kernel(
	.globl	v_xor_b16_kernel
	.p2align	8
	.type	v_xor_b16_kernel,@function
v_xor_b16_kernel:
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: %xor_b16 = xor i16
	; CHECK: zext i16 %xor_b16 to i32
	; CHECK: and i32 {{.+}}, -65536
	; CHECK: %logic_b16_merge_lo = or i32
	v_xor_b16 v0.l, v1.l, v2.l
	s_endpgm

; CHECK-LABEL: define amdgpu_kernel void @v_not_b16_kernel(
	.globl	v_not_b16_kernel
	.p2align	8
	.type	v_not_b16_kernel,@function
v_not_b16_kernel:
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: %not_b16 = xor i16 {{.+}}, -1
	; CHECK: zext i16 %not_b16 to i32
	; CHECK: and i32 {{.+}}, -65536
	; CHECK: %not_b16_merge = or i32
	v_not_b16 v0.l, v1.l
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_and_b16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_or_b16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_xor_b16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_not_b16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
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
    .name:           v_and_b16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_and_b16_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_or_b16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_or_b16_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_xor_b16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_xor_b16_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_not_b16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_not_b16_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
