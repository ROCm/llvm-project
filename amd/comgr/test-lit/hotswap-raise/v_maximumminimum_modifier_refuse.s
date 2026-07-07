; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_maximumminimum_f32_clamp_kernel,v_maximumminimum_f32_omod_kernel,v_maximum_f32_clamp_kernel,v_maximum_f32_omod_kernel,v_maximum3_f32_clamp_kernel,v_maximum3_f32_omod_kernel,v_maximumminimum_f16_clamp_kernel,v_maximumminimum_f16_omod_kernel 2>&1 | %FileCheck %s

; clamp/omod output-modifier refusal for maximum/minimum family.
; CHECK: failed to raise
; CHECK-SAME: v_maximumminimum_f32 has clamp=1
; CHECK: failed to raise
; CHECK-SAME: v_maximumminimum_f32 has nonzero omod
; CHECK: failed to raise
; CHECK-SAME: v_maximum_f32 has clamp=1
; CHECK: failed to raise
; CHECK-SAME: v_maximum_f32 has nonzero omod
; CHECK: failed to raise
; CHECK-SAME: v_maximum3_f32 has clamp=1
; CHECK: failed to raise
; CHECK-SAME: v_maximum3_f32 has nonzero omod
; CHECK: failed to raise
; CHECK-SAME: v_maximumminimum_f16 has clamp=1
; CHECK: failed to raise
; CHECK-SAME: v_maximumminimum_f16 has nonzero omod

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_maximumminimum_f32_clamp_kernel
	.p2align	8
	.type	v_maximumminimum_f32_clamp_kernel,@function
v_maximumminimum_f32_clamp_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_maximumminimum_f32 v0, v0, v1, v2 clamp
	s_endpgm

	.globl	v_maximumminimum_f32_omod_kernel
	.p2align	8
	.type	v_maximumminimum_f32_omod_kernel,@function
v_maximumminimum_f32_omod_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_maximumminimum_f32 v0, v0, v1, v2 mul:2
	s_endpgm

	.globl	v_maximum_f32_clamp_kernel
	.p2align	8
	.type	v_maximum_f32_clamp_kernel,@function
v_maximum_f32_clamp_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_maximum_f32 v0, v0, v1 clamp
	s_endpgm

	.globl	v_maximum_f32_omod_kernel
	.p2align	8
	.type	v_maximum_f32_omod_kernel,@function
v_maximum_f32_omod_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_maximum_f32 v0, v0, v1 mul:2
	s_endpgm

	.globl	v_maximum3_f32_clamp_kernel
	.p2align	8
	.type	v_maximum3_f32_clamp_kernel,@function
v_maximum3_f32_clamp_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_maximum3_f32 v0, v0, v1, v2 clamp
	s_endpgm

	.globl	v_maximum3_f32_omod_kernel
	.p2align	8
	.type	v_maximum3_f32_omod_kernel,@function
v_maximum3_f32_omod_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_maximum3_f32 v0, v0, v1, v2 mul:2
	s_endpgm

	.globl	v_maximumminimum_f16_clamp_kernel
	.p2align	8
	.type	v_maximumminimum_f16_clamp_kernel,@function
v_maximumminimum_f16_clamp_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_maximumminimum_f16 v0, v0, v1, v2 clamp
	s_endpgm

	.globl	v_maximumminimum_f16_omod_kernel
	.p2align	8
	.type	v_maximumminimum_f16_omod_kernel,@function
v_maximumminimum_f16_omod_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_maximumminimum_f16 v0, v0, v1, v2 mul:2
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_maximumminimum_f32_clamp_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_maximumminimum_f32_omod_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_maximum_f32_clamp_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_maximum_f32_omod_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_maximum3_f32_clamp_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_maximum3_f32_omod_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_maximumminimum_f16_clamp_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_maximumminimum_f16_omod_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_16_64 3
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
    .name:           v_maximumminimum_f32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximumminimum_f32_clamp_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maximumminimum_f32_omod_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximumminimum_f32_omod_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maximum_f32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximum_f32_clamp_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maximum_f32_omod_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximum_f32_omod_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maximum3_f32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximum3_f32_clamp_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maximum3_f32_omod_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximum3_f32_omod_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maximumminimum_f16_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximumminimum_f16_clamp_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maximumminimum_f16_omod_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maximumminimum_f16_omod_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
