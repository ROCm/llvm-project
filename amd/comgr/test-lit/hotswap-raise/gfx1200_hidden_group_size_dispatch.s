; RUN: %llvm_mc -mcpu=gfx1200 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vector_add 2>/dev/null \
; RUN:   | %FileCheck %s

; hidden_group_size seeded from dispatch.ptr, not implicitarg.ptr.
; CHECK-LABEL: define amdgpu_kernel void @vector_add(
; CHECK: call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK: getelementptr inbounds i8, ptr addrspace(4) %{{[^,]+}}, i32 4
; CHECK: load i16, ptr addrspace(4) %{{[^,]+}}, align 2
; CHECK-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
	.amdhsa_code_object_version 6
	.text
	.protected	vector_add
	.globl	vector_add
	.p2align	8
	.type	vector_add,@function
vector_add:
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x2c
	s_load_b32 s3, s[0:1], 0x18
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e64 s3, v0
	s_cbranch_execz .Ldone
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x10
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_wait_kmcnt 0x0
	v_add_co_u32 v2, vcc_lo, s6, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s7, v1, vcc_lo
	v_add_co_u32 v4, vcc_lo, s0, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s1, v1, vcc_lo
	global_load_b32 v2, v[2:3], off
	global_load_b32 v3, v[4:5], off
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v2, v3
	global_store_b32 v[0:1], v2, off
.Ldone:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vector_add
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
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
      - .offset:         24
        .size:           4
        .value_kind:     by_value
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
    .name:           vector_add
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         vector_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1200
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
