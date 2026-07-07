; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=s_xor_imm_mask_shadow_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_xor_b32 by -1 mask lowered to i1 predicate inversion (lane-mask shadow).
; CHECK-LABEL: define amdgpu_kernel void @s_xor_imm_mask_shadow_kernel(
; CHECK: [[CMP:%[[:alnum:]_.]+]] = fcmp oge float %{{[^,]+}}, 5.000000e-01
; CHECK: [[IMM_MASK:%[[:alnum:]_.]+]] = icmp ne i64 %mask_lane_bit, 0
; CHECK: [[INV:%[[:alnum:]_.]+]] = xor i1 [[CMP]], [[IMM_MASK]]
; CHECK: %cndmask = select i1 [[INV]], i32 1065353216, i32 -1082130432
; CHECK-NOT: %mask_lane_i1

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_xor_imm_mask_shadow_kernel
	.p2align	8
	.type	s_xor_imm_mask_shadow_kernel,@function
s_xor_imm_mask_shadow_kernel:           ; @s_xor_imm_mask_shadow_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_ge_f32_e64 s2, |v1|, 0.5
	s_xor_b32 s2, s2, -1
	v_cndmask_b32_e64 v1, -1.0, 1.0, s2
	
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_xor_imm_mask_shadow_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
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
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           s_xor_imm_mask_shadow_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         s_xor_imm_mask_shadow_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
