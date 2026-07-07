; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c2_permlane_swap_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Lower v_permlane16_swap_b32 to paired ds_bpermute lane swap.
; CHECK-LABEL: define amdgpu_kernel void @c2_permlane_swap_kernel(
; CHECK: %pls16_partner{{[0-9]*}} = xor i32 %{{[^,]+}}, 16
; CHECK: %pls16_addr{{[0-9]*}} = shl i32 %pls16_partner{{[0-9]*}}, 2
; CHECK: %pls16_bperm_src0{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls16_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: %pls16_bperm_vdst{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls16_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: %pls16_half_bit{{[0-9]*}} = and i32 %{{[^,]+}}, 16
; CHECK: %pls16_is_lane_low{{[0-9]*}} = icmp eq i32 %pls16_half_bit{{[0-9]*}}, 0
; CHECK: %pls16_new_vdst{{[0-9]*}} = select i1 %pls16_is_lane_low{{[0-9]*}}, i32 %{{[^,]+}}, i32 %pls16_bperm_src0{{[0-9]*}}
; CHECK: %pls16_new_src0_out{{[0-9]*}} = select i1 %pls16_is_lane_low{{[0-9]*}}, i32 %pls16_bperm_vdst{{[0-9]*}}, i32 %{{[^,]+}}
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_permlane_swap_kernel
	.p2align	8
	.type	c2_permlane_swap_kernel,@function
c2_permlane_swap_kernel:                ; @c2_permlane_swap_kernel
; %bb.0:
	s_load_b32 s4, s[0:1], 0x1c
	s_bfe_u32 s5, ttmp6, 0x4000c
	s_wait_xcnt 0x0
	s_load_b128 s[0:3], s[0:1], 0x0
	s_add_co_i32 s5, s5, 1
	s_and_b32 s6, ttmp6, 15
	s_mul_i32 s5, ttmp9, s5
	s_getreg_b32 s7, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s6, s6, s5
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s7, 0
	s_cselect_b32 s5, ttmp9, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v0, s5, s4, v0
	s_clause 0x1
	global_load_b32 v1, v0, s[0:1] scale_offset
	global_load_b32 v2, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	v_permlane16_swap_b32 v1, v2
	
	s_clause 0x1
	global_store_b32 v0, v1, s[0:1] scale_offset
	global_store_b32 v0, v2, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_permlane_swap_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
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
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           c2_permlane_swap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         c2_permlane_swap_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
