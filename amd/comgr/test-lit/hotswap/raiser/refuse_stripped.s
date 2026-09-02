; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %llvm-objcopy --strip-all %t.hsaco %t.stripped
; RUN: not %hotswap_transpile_cli %t.stripped --dump-meta 2>&1 | %FileCheck %s

; Symbol lookup walks .symtab, which stripping removes. Rather than degrading
; into a misleading missing-descriptor result, the load refuses stripped
; objects explicitly.
; CHECK: stripped code objects are not supported

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	meta_kernel
	.p2align	8
	.type	meta_kernel,@function
meta_kernel:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel meta_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 256
    .name:           meta_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         meta_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
