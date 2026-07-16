; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --emit-ir=ds_permute_b32_kernel \
; RUN:   | %FileCheck %s

; Lift ds_permute_b32 (forward/PUSH permute) to llvm.amdgcn.ds.permute.
; Mirror of ds_bpermute_b32.s -- distinct intrinsic (push not pull),
; same operand shape.
; CHECK-LABEL: define amdgpu_kernel void @ds_permute_b32_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_permute_b32_kernel
	.p2align	8
	.type	ds_permute_b32_kernel,@function
ds_permute_b32_kernel:
	s_load_dword s3, s[0:1], 0x24
	s_load_dword s4, s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s2, s2, s3
	v_add_u32_e32 v0, s2, v0
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[0:1]
	global_load_dword v2, v[2:3], off
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_and_b32_e32 v5, 64, v3
	v_xor_b32_e32 v4, 1, v3
	v_add_u32_e32 v5, 64, v5
	v_cmp_lt_i32_e32 vcc, v4, v5
	v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	s_nop 0
	v_cndmask_b32_e32 v3, v3, v4, vcc
	v_lshlrev_b32_e32 v3, 2, v3
	s_waitcnt vmcnt(0)
; CHECK:      %perm = call i32 @llvm.amdgcn.ds.permute(i32 %{{[^,]+}}, i32 %{{[^,]+}})
; CHECK: declare {{.*}}i32 @llvm.amdgcn.ds.permute(i32, i32)
	ds_permute_b32 v3, v3, v2
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	global_store_dword v[0:1], v2, off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_permute_b32_kernel
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 5
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
      - { .offset:         16, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 1024
    .name:           ds_permute_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .symbol:         ds_permute_b32_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
