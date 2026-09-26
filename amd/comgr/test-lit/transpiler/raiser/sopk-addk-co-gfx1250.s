; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=sopk_addk_co \
; RUN:   | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sopk_addk_co
	.p2align	8
	.type	sopk_addk_co,@function
; IR-LABEL: define amdgpu_kernel void @sopk_addk_co(
sopk_addk_co:
	s_movk_i32 s0, 7
	; IR: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 7, i32 -12288)
	; IR: extractvalue { i32, i1 } {{.+}}, 1
	s_addk_co_i32 s0, 0xd000
	v_nop
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sopk_addk_co
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
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
    .name:           sopk_addk_co
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sopk_addk_co.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
