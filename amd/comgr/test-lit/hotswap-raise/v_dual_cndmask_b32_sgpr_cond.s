; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_dual_cndmask_b32_sgpr_cond_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_dual_cndmask_b32 SGPR/VCC-cond select lift.
; CHECK-LABEL: define amdgpu_kernel void @v_dual_cndmask_b32_sgpr_cond_kernel(
; CHECK: [[SGPR_CMP:%[[:alnum:]_.]+]] = icmp ult i32 8,
; CHECK: [[VCC_CMP:%[[:alnum:]_.]+]] = icmp ult i32 3,
; CHECK: %vopd_cndmask = select i1 [[SGPR_CMP]],
; CHECK: %vopd_cndmask{{[0-9]+}} = select i1 [[VCC_CMP]],
; CHECK-NOT: call i32 @llvm.amdgcn.cndmask

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_dual_cndmask_b32_sgpr_cond_kernel
	.p2align	8
	.type	v_dual_cndmask_b32_sgpr_cond_kernel,@function
v_dual_cndmask_b32_sgpr_cond_kernel:    ; @v_dual_cndmask_b32_sgpr_cond_kernel
; %bb.0:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_add_nc_u32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v3, 3, v0 :: v_dual_bitop2_b32 v1, 31, v1 bitop3:0x40
	v_and_b32_e32 v2, 31, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 31, v3
	s_wait_kmcnt 0x0
	s_clause 0x3
	global_load_b32 v4, v0, s[6:7] scale_offset
	global_load_b32 v5, v1, s[6:7] scale_offset
	global_load_b32 v6, v2, s[6:7] scale_offset
	global_load_b32 v7, v3, s[6:7] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_u32_e64 s0, 8, v0
	v_cmp_lt_u32_e32 vcc_lo, 3, v0
	v_dual_cndmask_b32 v2, v4, v5, s0 :: v_dual_cndmask_b32 v3, v6, v7, vcc_lo
	
	global_store_b64 v0, v[2:3], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_dual_cndmask_b32_sgpr_cond_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
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
    .name:           v_dual_cndmask_b32_sgpr_cond_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_dual_cndmask_b32_sgpr_cond_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
