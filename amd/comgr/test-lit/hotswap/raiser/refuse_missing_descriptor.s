; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %hotswap_transpile_cli %t.hsaco --dump-meta 2>&1 | %FileCheck %s

; The metadata names meta_kernel.kd as the descriptor symbol, but the object
; emits no descriptor (no .amdhsa_kernel block, hence no .rodata). Translation
; requires the descriptor, so the load fails fast rather than defaulting.
; CHECK: readKernelDescriptor

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	meta_kernel
	.p2align	8
	.type	meta_kernel,@function
meta_kernel:
	s_endpgm
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
