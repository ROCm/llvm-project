; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; s_monitor_sleep is gfx12 and later only, so it needs a gfx1250 source.
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=monitor_sleep_kernel 2>&1 | %FileCheck %s
; CHECK: unsupported-opcode: s_monitor_sleep [SOPP]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	monitor_sleep_kernel
	.p2align	8
	.type	monitor_sleep_kernel,@function
monitor_sleep_kernel:
	s_monitor_sleep 1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel monitor_sleep_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
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
    .name:           monitor_sleep_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         monitor_sleep_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
