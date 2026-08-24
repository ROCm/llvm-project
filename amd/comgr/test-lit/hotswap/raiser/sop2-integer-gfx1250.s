; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=sop2_integer_gfx1250 \
; RUN:   | %FileCheck %s --check-prefix=DECODE
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=sop2_integer_gfx1250 | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sop2_integer_gfx1250
	.p2align	8
	.type	sop2_integer_gfx1250,@function
; IR-LABEL: define amdgpu_kernel void @sop2_integer_gfx1250(
sop2_integer_gfx1250:
	; DECODE: S_MUL_U64{{.+}}s_mul_u64
	; IR: mul i64
	s_mul_u64 s[2:3], s[0:1], s[4:5]
	; DECODE: S_ADD_NC_U64{{.+}}s_add_nc_u64
	; IR: add i64
	s_add_nc_u64 s[2:3], s[0:1], s[4:5]
	; DECODE: S_SUB_NC_U64{{.+}}s_sub_nc_u64
	; IR: sub i64
	s_sub_nc_u64 s[2:3], s[0:1], s[4:5]
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sop2_integer_gfx1250
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sop2_integer_gfx1250
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         sop2_integer_gfx1250.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
