; REQUIRES: comgr-has-transpiler

; The source splits the workgroup barrier into an arrival in SOP1 and a wait in
; SOPP. An arrival that always reaches its wait is raised with it as the one
; barrier the two amount to; the shapes where it does not keep the refusals
; that barriers.s pins.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=adjacent_kernel \
; RUN:   | %FileCheck %s --check-prefix=ADJACENT
; RUN: %transpile_cli %t.hsaco --emit-ir=apart_kernel \
; RUN:   | %FileCheck %s --check-prefix=APART
; RUN: not %transpile_cli %t.hsaco --emit-ir=across_branch_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ACROSS-BRANCH
; RUN: not %transpile_cli %t.hsaco --emit-ir=named_barrier_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=NAMED

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	adjacent_kernel
	.p2align	8
	.type	adjacent_kernel,@function
adjacent_kernel:
; ADJACENT-LABEL: define amdgpu_kernel void @adjacent_kernel(
; ADJACENT-COUNT-1: call void @llvm.amdgcn.s.barrier()
; ADJACENT-NOT: call void @llvm.amdgcn.s.barrier()
; ADJACENT: ret void
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_endpgm

; Whatever lies between the halves, the arrival still reaches the wait, so the
; two still stand for one barrier.
	.globl	apart_kernel
	.p2align	8
	.type	apart_kernel,@function
apart_kernel:
; APART-LABEL: define amdgpu_kernel void @apart_kernel(
; APART-COUNT-1: call void @llvm.amdgcn.s.barrier()
; APART-NOT: call void @llvm.amdgcn.s.barrier()
; APART: ret void
	s_barrier_signal -1
	s_mov_b32 s0, 1
	s_add_co_i32 s0, s0, 2
	s_barrier_wait 0xffff
	s_endpgm

; A branch between the halves leaves a path that reaches the wait without the
; arrival, so neither half stands for a whole barrier.
	.globl	across_branch_kernel
	.p2align	8
	.type	across_branch_kernel,@function
across_branch_kernel:
; ACROSS-BRANCH: unsupported-instruction-form: s_barrier_signal [SOP1]
; ACROSS-BRANCH-SAME: arrives at a barrier without waiting there
	s_barrier_signal -1
	s_branch across_branch_tail
across_branch_tail:
	s_barrier_wait 0xffff
	s_endpgm

; Barrier 1 is a named barrier object rather than the workgroup barrier, and
; only the waves that joined it arrive there.
	.globl	named_barrier_kernel
	.p2align	8
	.type	named_barrier_kernel,@function
named_barrier_kernel:
; NAMED: unsupported-instruction-form: s_barrier_signal [SOP1]
; NAMED-SAME: arrives at a barrier without waiting there
	s_barrier_signal 1
	s_barrier_wait 1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel adjacent_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel apart_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel across_branch_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel named_barrier_kernel
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
    .name:           adjacent_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         adjacent_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           apart_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         apart_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           across_branch_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         across_branch_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           named_barrier_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         named_barrier_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
