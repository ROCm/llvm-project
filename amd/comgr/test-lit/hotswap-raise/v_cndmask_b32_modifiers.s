; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_cndmask_b32_modifiers_kernel \
; RUN:   | %FileCheck %s
;
; Lift test for VOP3 v_cndmask_b32 source modifiers.

; CHECK-LABEL: define amdgpu_kernel void @v_cndmask_b32_modifiers_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cndmask_b32_modifiers_kernel
	.p2align	8
	.type	v_cndmask_b32_modifiers_kernel,@function
v_cndmask_b32_modifiers_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, s0
	v_mov_b32_e32 v2, s1
	; CHECK: [[CMP:%[[:alnum:]_.]+]] = icmp ult i32 0,
	v_cmp_lt_u32_e64 s0, 0, v0
	; CHECK: [[NEG:%[[:alnum:]_.]+]] = fneg float %{{[^,]+}}
	; CHECK: [[NEG_BITS:%[[:alnum:]_.]+]] = bitcast float [[NEG]] to i32
	; CHECK: [[ABS:%[[:alnum:]_.]+]] = call float @llvm.fabs.f32(float %{{[^)]+}})
	; CHECK: [[ABS_BITS:%[[:alnum:]_.]+]] = bitcast float [[ABS]] to i32
	; CHECK: %cndmask = select i1 [[CMP]], i32 [[ABS_BITS]], i32 [[NEG_BITS]]
	v_cndmask_b32_e64 v1, -v1, |v2|, s0
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cndmask_b32_modifiers_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_cndmask_b32_modifiers_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_cndmask_b32_modifiers_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
