; REQUIRES: comgr-has-hotswap-transpile

; A gfx9 source encodes the same opcodes differently and packs every counter
; into one s_waitcnt; the gfx12 coverage is in sopp_waits.s and
; sopp_waits_gfx1200.s.

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=wait_kernel \
; RUN:   | %FileCheck %s --check-prefix=WAIT
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=inert_kernel \
; RUN:   | %FileCheck %s --check-prefix=INERT

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.text
	.globl	wait_kernel
	.p2align	8
	.type	wait_kernel,@function
wait_kernel:
; A partial wait fences as widely as a full one: the counter it does not name
; is one the raise cannot reproduce on a target that counts differently.
; WAIT-LABEL: define amdgpu_kernel void @wait_kernel(
; WAIT: %workgroup_id_x
; WAIT-COUNT-2: fence syncscope("agent") seq_cst
; WAIT-NOT: fence
; WAIT: ret void
; WAIT-NEXT: }
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_waitcnt lgkmcnt(0)
	s_endpgm

	.globl	inert_kernel
	.p2align	8
	.type	inert_kernel,@function
inert_kernel:
; A wave64 source opens with an entry prologue, so that is what the terminator
; follows once the inert opcodes have emitted nothing.
; INERT-LABEL: define amdgpu_kernel void @inert_kernel(
; INERT: %workgroup_id_x
; INERT-NEXT: ret void
; INERT-NEXT: }
	s_nop 0
	s_sleep 1
	s_wakeup
	s_setprio 1
	s_incperflevel 1
	s_decperflevel 1
	s_ttracedata
	s_icache_inv
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wait_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel inert_kernel
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
    .name:           wait_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         wait_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 256
    .name:           inert_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         inert_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
