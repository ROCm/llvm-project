; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=flat_store_byte_d16_hi_kernel 2>&1 \
; RUN:   | %FileCheck %s

; flat_store_d16_hi_b8: D16-hi byte store lowering (lshr 16 / trunc to i8),
; the byte-store sibling of flat_store_short_d16_hi. Surfaces bits [23:16].

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	flat_store_byte_d16_hi_kernel
	.p2align	8
	.type	flat_store_byte_d16_hi_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @flat_store_byte_d16_hi_kernel(
flat_store_byte_d16_hi_kernel:          ; @flat_store_byte_d16_hi_kernel
; %bb.0:
	s_load_b32 s3, s[0:1], 0x1c
	s_bfe_u32 s4, ttmp6, 0x4000c
	s_and_b32 s5, ttmp6, 15
	s_add_co_i32 s4, s4, 1
	s_getreg_b32 s6, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s4, ttmp9, s4
	s_wait_xcnt 0x0
	s_load_b96 s[0:2], s[0:1], 0x0
	s_add_co_i32 s5, s5, s4
	s_wait_kmcnt 0x0
	s_and_b32 s3, s3, 0xffff
	s_cmp_eq_u32 s6, 0
	s_cselect_b32 s4, ttmp9, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u32 v0, s4, s3, v0
	v_dual_mov_b32 v2, s2 :: v_dual_ashrrev_i32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[0:1]
; CHECK: [[SHIFT:%.+]] = lshr i32 %{{.+}}, 16
; CHECK: [[TRUNC:%.+]] = trunc i32 [[SHIFT]] to i8
; CHECK: store i8 [[TRUNC]], ptr %{{.+}}
; CHECK-NOT: store i32 [[TRUNC]], ptr %
; CHECK-NOT: store i16 [[TRUNC]], ptr %
	flat_store_d16_hi_b8 v[0:1], v2

	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flat_store_byte_d16_hi_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 7
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
        .value_kind:     by_value
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
    .name:           flat_store_byte_d16_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         flat_store_byte_d16_hi_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
