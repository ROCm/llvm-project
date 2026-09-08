; REQUIRES: comgr-has-hotswap-transpile

; The event, image and BVH wait counters, which only a gfx12 source encodes,
; and the sleep that ends on a wakeup rather than after a bounded stall.

; RUN: %llvm-mc -triple=amdgpu12-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=wait_kernel \
; RUN:   | %FileCheck %s
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=sleep_forever_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=FOREVER

	.amdgcn_target "amdgcn-amd-amdhsa--gfx12-generic"
	.text
	.globl	wait_kernel
	.p2align	8
	.type	wait_kernel,@function
wait_kernel:
; CHECK-LABEL: define amdgpu_kernel void @wait_kernel(
; CHECK-NEXT: entry:
; CHECK-COUNT-5: fence syncscope("agent") seq_cst
; CHECK-NOT: fence
; CHECK: ret void
; CHECK-NEXT: }
	s_wait_event 0
	s_wait_expcnt 0
	s_wait_samplecnt 0
	s_wait_bvhcnt 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_endpgm

	.globl	sleep_forever_kernel
	.p2align	8
	.type	sleep_forever_kernel,@function
sleep_forever_kernel:
; FOREVER: unsupported-sleep-forever: s_sleep [SOPP]
	s_sleep 0x8000
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wait_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sleep_forever_kernel
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
    .name:           wait_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         wait_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sleep_forever_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sleep_forever_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
