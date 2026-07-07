; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_cmpk_eq_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_cmpk_eq_i32 scalar compare-with-immediate lift.
; CHECK-LABEL: define amdgpu_kernel void @s_cmpk_eq_i32_kernel(
; CHECK: %scmpk = icmp eq i32 %{{[^,]+}}, 1024
; CHECK: %csel = select i1 %scmpk, i32 1, i32 0
; CHECK-NOT: %scmpk = icmp eq i32 %{{[^,]+}}, %{{[^,]+}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	s_cmpk_eq_i32_kernel
	.p2align	8
	.type	s_cmpk_eq_i32_kernel,@function
s_cmpk_eq_i32_kernel:                   ; @s_cmpk_eq_i32_kernel
; %bb.0:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_cmpk_eq_i32 s2, 0x400
	s_cselect_b32 s2, 1, 0
	
	v_lshlrev_b32_e32 v0, 2, v0
	v_mov_b32_e32 v1, s2
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_cmpk_eq_i32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 3
		.amdhsa_accum_offset 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_tg_split 0
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           s_cmpk_eq_i32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .symbol:         s_cmpk_eq_i32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
