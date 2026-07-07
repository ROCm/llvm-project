; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=s_and_imm_high_bit_mask_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_and_b32 high-bit wave-mask lowering under --disable-wave-native.
; CHECK-LABEL: define amdgpu_kernel void @s_and_imm_high_bit_mask_kernel(
; CHECK: %vcmpf = fcmp oge float %{{[^,]+}}, 5.000000e-01
; CHECK: %mask_at_lane = lshr i64 -281470681808896, %mask_lane_idx
; CHECK: %mask_lane_bit = and i64 %mask_at_lane, 1
; CHECK: %mask_lane_i1 = icmp ne i64 %mask_lane_bit, 0
; CHECK: %and = and i32 %{{[^,]+}}, -65536
; CHECK: %wave_mask_and = and i1 %vcmpf, %mask_lane_i1
; CHECK: %cndmask = select i1 %wave_mask_and, i32 1, i32 0

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_and_imm_high_bit_mask_kernel
	.p2align	8
	.type	s_and_imm_high_bit_mask_kernel,@function
s_and_imm_high_bit_mask_kernel:         ; @s_and_imm_high_bit_mask_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_ge_f32_e64 s2, |v1|, 0.5
	s_and_b32 s2, s2, 0xFFFF0000
	v_cndmask_b32_e64 v1, 0, 1, s2
	
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_and_imm_high_bit_mask_kernel
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
    .name:           s_and_imm_high_bit_mask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         s_and_imm_high_bit_mask_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
