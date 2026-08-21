; REQUIRES: comgr-has-hotswap-transpile

; The two export-counter waits are gfx11/gfx12 opcodes that gfx1250 dropped, so
; they need a source ISA of their own; the rest of the SOPP coverage is in
; sopp_waits.s.

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1200 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=export_wait_kernel \
; RUN:   | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
	.text
	.globl	export_wait_kernel
	.p2align	8
	.type	export_wait_kernel,@function
export_wait_kernel:
; CHECK-LABEL: define amdgpu_kernel void @export_wait_kernel(
; CHECK-NEXT: entry:
; CHECK-COUNT-2: fence syncscope("agent") seq_cst
; CHECK-NOT: fence
; CHECK: ret void
; CHECK-NEXT: }
	s_wait_event 0
	s_wait_expcnt 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel export_wait_kernel
		.amdhsa_kernarg_size 0
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
    .name:           export_wait_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         export_wait_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
