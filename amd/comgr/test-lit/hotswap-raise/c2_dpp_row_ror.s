; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c2_dpp_row_ror_kernel \
; RUN:   | %FileCheck %s

; Lower DPP row_ror:N to a ds_bpermute selector. row_ror:1 reads the
; within-row source lane (W - 1) mod 16, computed as (W + 15) & 15;
; the rotation wraps inside the 16-lane row, so every target lane is
; always in-range and the mapping is identical on wave32 and wave64.
; CHECK-LABEL: define amdgpu_kernel void @c2_dpp_row_ror_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_dpp_row_ror_kernel
	.p2align	8
	.type	c2_dpp_row_ror_kernel,@function
c2_dpp_row_ror_kernel:           ; @c2_dpp_row_ror_kernel
; %bb.0:
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x14
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v0, s0, s4, v0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
; The row_ror:1 update.dpp lowers to a ds_bpermute selector (no update.dpp left).
; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
; CHECK-DAG: %cwd_dpp_within_row = and i32 %{{.+}}, 15
; CHECK-DAG: %cwd_dpp_row_base = and i32 %{{.+}}, -16
; CHECK-DAG: %cwd_dpp_ror_sum = add i32 %cwd_dpp_within_row, 15
; CHECK-DAG: %cwd_dpp_ror_src = and i32 %cwd_dpp_ror_sum, 15
; CHECK-DAG: %cwd_dpp_src_safe = select i1 true, i32 %cwd_dpp_ror_src, i32 0
; CHECK-DAG: %cwd_dpp_src_abs = or i32 %cwd_dpp_row_base, %cwd_dpp_src_safe
; CHECK-DAG: %cwd_dpp_selector = shl i32 %cwd_dpp_src_abs, 2
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector, i32 %{{[^,]+}})
; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
; CHECK: declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)
	v_mov_b32_dpp v1, v1 row_ror:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	
	global_store_b32 v0, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_dpp_row_ror_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
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
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           c2_dpp_row_ror_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c2_dpp_row_ror_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
