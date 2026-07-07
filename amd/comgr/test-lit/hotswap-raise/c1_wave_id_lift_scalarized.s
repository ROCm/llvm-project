; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --disable-writelane-rewrite \
; RUN:     --emit-ir=c1_wave_id_lift_scalarized_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=c1_wave_id_lift_scalarized_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITTEN

; Refuse vs rewrite scalarized wave-id v_writelane lane leak.
; REFUSE: transpiler: pre-translation abort:
; REFUSE-SAME: cross-wave-lane-id-leak
; REFUSE-SAME: v_writelane_b32
; REFUSE-SAME: wave-size-translation.md
; REFUSE: WaveIdLiftScalarized
; REFUSE-SAME: Class 1
; REFUSE-SAME: v_writelane/v_readlane
; REFUSE-SAME: WMMA
; REFUSE: outcome: (c) refuse
; REFUSE: raise_cli: kernel 'c1_wave_id_lift_scalarized_kernel' failed to raise:
; REFUSE-SAME: v_writelane_b32
; REWRITTEN-LABEL: define amdgpu_kernel void @c1_wave_id_lift_scalarized_kernel(
; REWRITTEN: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITTEN: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITTEN: %cwd_wl_mask = icmp eq
; REWRITTEN: %cwd_writelane_rewritten = select i1
; REWRITTEN-NOT: call i32 @llvm.amdgcn.writelane

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c1_wave_id_lift_scalarized_kernel
	.p2align	8
	.type	c1_wave_id_lift_scalarized_kernel,@function
c1_wave_id_lift_scalarized_kernel:      ; @c1_wave_id_lift_scalarized_kernel
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_wait_xcnt 0x0
	s_load_b32 s0, s[0:1], 0x2c
	s_wait_xcnt 0x0
	s_bfe_u32 s1, ttmp8, 0x50019
	
	v_writelane_b32 v26, s1, 0
	
	v_dual_mov_b32 v1, 0 :: v_dual_bitop2_b32 v26, s1, v26 bitop3:0x14
	s_add_co_i32 s2, s2, 1
	s_and_b32 s3, ttmp6, 15
	s_mul_i32 s2, ttmp9, s2
	s_wait_kmcnt 0x0
	s_clause 0x5
	global_load_b128 v[2:5], v1, s[4:5]
	global_load_b128 v[10:13], v1, s[6:7]
	global_load_b128 v[14:17], v1, s[6:7] offset:16
	global_load_b128 v[22:25], v1, s[8:9] offset:16
	global_load_b128 v[6:9], v1, s[4:5] offset:16
	global_load_b128 v[18:21], v1, s[8:9]
	s_wait_xcnt 0x1
	s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s3, s3, s2
	s_and_b32 s0, s0, 0xffff
	s_cmp_eq_u32 s4, 0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[2:9], v[10:17], v[18:25]
	s_cselect_b32 s2, ttmp9, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v0, s2, s0, v0
	s_clause 0x1
	global_store_b128 v1, v[18:21], s[8:9]
	global_store_b128 v1, v[22:25], s[8:9] offset:16
	global_store_b32 v0, v26, s[10:11] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c1_wave_id_lift_scalarized_kernel
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 27
		.amdhsa_next_free_sgpr 12
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .max_flat_workgroup_size: 1024
    .name:           c1_wave_id_lift_scalarized_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         c1_wave_id_lift_scalarized_kernel.kd
    .vgpr_count:     27
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
