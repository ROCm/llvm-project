; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=frexp_exp_f32_kernel \
; RUN:   | %FileCheck %s

; v_frexp_exp_i32_f32 (VOP1): f32 source, i32 destination. The f64-exp and
; f32-mant variants already existed; this is the missing f32-exp sibling,
; lifting to llvm.amdgcn.frexp.exp overloaded on {i32,f32}. On torch's pow path.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	frexp_exp_f32_kernel
	.p2align	8
	.type	frexp_exp_f32_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @frexp_exp_f32_kernel(
frexp_exp_f32_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_cvt_f32_u32_e32 v1, v0
; CHECK: call i32 @llvm.amdgcn.frexp.exp.i32.f32(
	v_frexp_exp_i32_f32_e32 v2, v1
	global_store_b32 v0, v2, s[0:1]
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel frexp_exp_f32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
    .name:           frexp_exp_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         frexp_exp_f32_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
