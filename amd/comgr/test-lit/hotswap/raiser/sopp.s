; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=waits_kernel,setprio_kernel,sleep_kernel,wakeup_kernel \
; RUN:   --target-isa=gfx942 \
; RUN:   | %FileCheck %s --check-prefixes=WAIT-GFX9,PRIO-GFX9,SLEEP,WAKEUP

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	waits_kernel
	.p2align	8
	.type	waits_kernel,@function

; WAIT-GFX9-LABEL: define amdgpu_kernel void @waits_kernel(
waits_kernel:
; A partial wait fences as widely as a full one.
; WAIT-GFX9: fence syncscope("agent") seq_cst
	s_waitcnt vmcnt(1) expcnt(2) lgkmcnt(3)
	s_waitcnt lgkmcnt(0)
; WAIT-GFX9-NEXT: fence syncscope("agent") seq_cst
	s_nop 0
	s_incperflevel 0
	s_decperflevel 0
	s_ttracedata
	s_icache_inv
; WAIT-GFX9-NEXT: ret void
	s_endpgm

	.globl	setprio_kernel
	.p2align	8
	.type	setprio_kernel,@function

; PRIO-GFX9-LABEL: define amdgpu_kernel void @setprio_kernel(
setprio_kernel:
; PRIO-GFX9: call void @llvm.amdgcn.s.setprio(i16 3)
	s_setprio 3
; PRIO-GFX9-NEXT: ret void
	s_endpgm

	.globl	sleep_kernel
	.p2align	8
	.type	sleep_kernel,@function

; SLEEP-LABEL: define amdgpu_kernel void @sleep_kernel(
sleep_kernel:
	s_sleep 0
; SLEEP: ret void
; SLEEP-NEXT: }
	s_endpgm

	.globl	wakeup_kernel
	.p2align	8
	.type	wakeup_kernel,@function

; WAKEUP-LABEL: define amdgpu_kernel void @wakeup_kernel(
wakeup_kernel:
	s_wakeup
; WAKEUP: ret void
; WAKEUP-NEXT: }
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel waits_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel setprio_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel sleep_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
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
    .name:           waits_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         waits_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setprio_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         setprio_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
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
