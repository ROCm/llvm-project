; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --dump-meta | %FileCheck %s

; The AMDHSA `.symbol` need not be `<name>.kd`. Here the descriptor symbol is
; other.kd while the kernel name is meta_kernel, so the load succeeds only if
; the descriptor is looked up through the metadata `.symbol` rather than a name
; synthesized from `.name`.
; CHECK: kernel: meta_kernel {{.+}} has_kd=1

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	meta_kernel
	.p2align	8
	.type	meta_kernel,@function
meta_kernel:
	s_endpgm
	.globl	other
	.p2align	8
	.type	other,@function
other:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel other
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
    .symbol:         other.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
