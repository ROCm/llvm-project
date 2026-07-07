; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_frexp_exp_i32_f64_dpp_kernel 2>&1 \
; RUN:   | %FileCheck %s

; v_frexp_exp_i32_f64_dpp mixed source/dest width refusal.
; CHECK: kernel 'v_frexp_exp_i32_f64_dpp_kernel' failed to raise:
; CHECK-SAME: v_frexp_exp_i32_f64
; CHECK-SAME: VOP1
; CHECK-SAME: mixed source/destination widths

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	v_frexp_exp_i32_f64_dpp_kernel
	.p2align	8
	.type	v_frexp_exp_i32_f64_dpp_kernel,@function
v_frexp_exp_i32_f64_dpp_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_frexp_exp_i32_f64_dpp v0, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, 0
	global_store_dword v1, v0, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_frexp_exp_i32_f64_dpp_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_frexp_exp_i32_f64_dpp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_frexp_exp_i32_f64_dpp_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
