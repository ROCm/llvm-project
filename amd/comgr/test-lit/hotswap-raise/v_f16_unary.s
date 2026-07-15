; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=f16_unary_kernel \
; RUN:   | %FileCheck %s

; f16 unary rounding + reciprocal (true16 VOP3). ceil/trunc/rndne lower to the
; matching llvm.* intrinsic; rcp to the native llvm.amdgcn.rcp on f16.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	f16_unary_kernel
	.p2align	8
	.type	f16_unary_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @f16_unary_kernel(
f16_unary_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
; CHECK: call half @llvm.ceil.f16(
	v_ceil_f16 v1, v1
; CHECK: call half @llvm.trunc.f16(
	v_trunc_f16 v2, v2
; CHECK: call half @llvm.roundeven.f16(
	v_rndne_f16 v3, v3
; CHECK: call half @llvm.amdgcn.rcp.f16(
	v_rcp_f16 v4, v4
	global_store_b16 v0, v1, s[0:1]
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel f16_unary_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 2
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
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           f16_unary_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         f16_unary_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
