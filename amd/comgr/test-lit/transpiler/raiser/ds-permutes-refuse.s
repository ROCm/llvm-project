; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu9.0a-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx950 --emit-ir 2>&1 \
; RUN:   | %FileCheck %s

	.amdhsa_code_object_version 6
	.text
	.globl swizzle_agpr
	.p2align 8
	.type swizzle_agpr,@function
swizzle_agpr:
; CHECK: unsupported-instruction-form: ds_swizzle_b32 [DS]
; CHECK-SAME: in kernel 'swizzle_agpr'
; CHECK-SAME: DS destination must be a VGPR
	ds_swizzle_b32 a1, v0 offset:swizzle(SWAP,1)
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel swizzle_agpr
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: swizzle_agpr
    .symbol: swizzle_agpr.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
