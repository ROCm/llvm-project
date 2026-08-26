; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=sleep_kernel 2>&1 | %FileCheck %s --check-prefix=SLEEP
; SLEEP: unsupported-opcode: s_sleep [SOPP]

; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=wakeup_kernel 2>&1 | %FileCheck %s --check-prefix=WAKEUP
; WAKEUP: unsupported-opcode: s_wakeup [SOPP]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	sleep_kernel
	.p2align	8
	.type	sleep_kernel,@function
sleep_kernel:
	s_sleep 1
	s_endpgm

	.globl	wakeup_kernel
	.p2align	8
	.type	wakeup_kernel,@function
wakeup_kernel:
	s_wakeup
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sleep_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel wakeup_kernel
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
    .name:           sleep_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sleep_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wakeup_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         wakeup_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
