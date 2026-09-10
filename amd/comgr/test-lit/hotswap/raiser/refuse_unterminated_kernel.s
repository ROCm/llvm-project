; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; Execution running off the end of a kernel extent means the code is truncated
; or the extent is misbounded. Returning there would hand back a kernel that
; reads as having run to completion, so the raise refuses instead.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=trunc_kernel 2>&1 \
; RUN:   | %FileCheck %s
; CHECK: unterminated-kernel-extent in kernel 'trunc_kernel'

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	trunc_kernel
	.p2align	8
	.type	trunc_kernel,@function
trunc_kernel:
	s_mov_b32 s0, 0

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel trunc_kernel
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
    .max_flat_workgroup_size: 1024
    .name:           trunc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         trunc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
