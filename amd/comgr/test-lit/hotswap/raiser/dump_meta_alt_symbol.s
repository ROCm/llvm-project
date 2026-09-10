; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --dump-meta | %FileCheck %s

; The AMDHSA `.name` and `.symbol` are permitted to differ. Here the source name
; is display_name while the descriptor symbol is real_entry.kd, and no symbol
; named display_name exists. The load therefore succeeds only because the code
; extent is resolved through the descriptor's kernel_code_entry_byte_offset
; (which points at real_entry), not through a symbol lookup of `.name`.
; extent_size=4 is real_entry's single s_endpgm, proving the descriptor-selected
; entry is used.
; CHECK: kernel: display_name {{.+}} has_kd=1 {{.+}} extent_size=4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	real_entry
	.p2align	8
	.type	real_entry,@function
real_entry:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel real_entry
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
    .name:           display_name
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         real_entry.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
