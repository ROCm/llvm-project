; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; The raise declares a kernel from its metadata alone, so what the module says
; it targets and what signature the lifted function takes are fixed before any
; instruction is lifted. Both kernels here run to an s_endpgm and lift to the
; terminator alone, leaving the declaration as the only thing under test.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir | %FileCheck %s

; The layout comes from a machine built for the GPU being raised onto, so
; private is the 32-bit address space it is on AMDGPU rather than the 64-bit one
; a module carrying no layout of its own would report.
; CHECK: target datalayout = {{.+}}p5:32:32
; CHECK: target triple = "amdgcn-amd-amdhsa"

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	karg_kernel
	.p2align	8
	.type	karg_kernel,@function
; The source kernarg segment is passed whole and by reference, so the raised
; kernel reads the segment the source ABI laid out instead of one rebuilt from
; individual arguments.
; CHECK-LABEL: define amdgpu_kernel void @karg_kernel(ptr addrspace(4) byref([40 x i8]) align 16
; CHECK: ret void
karg_kernel:
	s_mov_b32 s0, 0
	s_endpgm

	.globl	noarg_kernel
	.p2align	8
	.type	noarg_kernel,@function
; An empty kernarg segment is no segment: the function takes no parameter
; rather than an empty one.
; CHECK-LABEL: define amdgpu_kernel void @noarg_kernel()
; CHECK: ret void
noarg_kernel:
	s_endpgm

; The target's hidden arguments are suppressed. The source segment is passed as
; it stands, so there is nothing for the target ABI to append to it.
; CHECK: attributes {{.+}} "amdgpu-no-implicitarg-ptr"

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel karg_kernel
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel noarg_kernel
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
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 1024
    .name:           karg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         karg_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           noarg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         noarg_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
