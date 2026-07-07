; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_sub_nc_u64_kernel 2>/dev/null | %FileCheck %s

; v_sub_nc_u64 lowered as plain i64 sub (no usub.with.overflow, no i32 split) subtract.
; CHECK-LABEL: define amdgpu_kernel void @v_sub_nc_u64_kernel(
; CHECK: sub {{.*}}i64 %{{[^,]+}}, %{{[^,]+}}
; CHECK-NOT: @llvm.usub.with.overflow
; CHECK-NOT: sub {{.*}}i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_sub_nc_u64_kernel
	.p2align	8
	.type	v_sub_nc_u64_kernel,@function
v_sub_nc_u64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s8, s[0:1], 0x24
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s9, ttmp6, 15
	s_add_co_i32 s10, s2, 1
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_xcnt 0x0
	s_mul_i32 s0, ttmp9, s10
	s_getreg_b32 s1, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s9, s9, s0
	s_wait_kmcnt 0x0
	s_and_b32 s0, s8, 0xffff
	s_cmp_eq_u32 s1, 0
	s_cselect_b32 s1, ttmp9, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v4, s1, s0, v0
	s_clause 0x1
	global_load_b64 v[0:1], v4, s[6:7] scale_offset
	global_load_b64 v[2:3], v4, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	v_sub_nc_u64 v[0:1], v[0:1], v[2:3]
	
	global_store_b64 v4, v[0:1], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_sub_nc_u64_kernel
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 11
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
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 1024
    .name:           v_sub_nc_u64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .symbol:         v_sub_nc_u64_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
