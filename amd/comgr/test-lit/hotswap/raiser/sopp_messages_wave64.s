; REQUIRES: comgr-has-hotswap-transpile

; The message opcodes on a source old enough to spell message id 3 as the
; geometry-shader completion rather than the VGPR deallocation hint, so this
; fixture is a wave64 gfx942 kernel. Halting and the interrupt mean the same
; thing here as they do on gfx1250; the id the two generations disagree on does
; not.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=interrupt_kernel \
; RUN:   | %FileCheck %s --check-prefix=INTERRUPT

; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=gs_done_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=GS-DONE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	interrupt_kernel
	.p2align	8
	.type	interrupt_kernel,@function
interrupt_kernel:
; INTERRUPT-LABEL: define amdgpu_kernel void @interrupt_kernel(
; INTERRUPT: call void @llvm.amdgcn.s.sethalt(i32 2)
	s_sethalt 2
	s_mov_b32 m0, 42
; INTERRUPT-NEXT: call void @llvm.amdgcn.s.sendmsg(i32 1, i32 42)
	s_sendmsg sendmsg(MSG_INTERRUPT)
; INTERRUPT-NEXT: call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 42)
	s_sendmsghalt sendmsg(MSG_INTERRUPT)
; INTERRUPT-NEXT: ret void
	s_endpgm

	.globl	gs_done_kernel
	.p2align	8
	.type	gs_done_kernel,@function
gs_done_kernel:
; Message id 3 here says a geometry shader finished, not that the wave is done
; with its VGPRs, so it is refused rather than dropped as the deallocation hint
; a newer source would have meant by the same bits.
; GS-DONE: unsupported-instruction-form: s_sendmsg [SOPP]
; GS-DONE-SAME: sends message 0x3, and the interrupt is the only message that
; GS-DONE-SAME: means the same thing on every target
	s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel interrupt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_accum_offset 4
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel gs_done_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_accum_offset 4
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
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
    .name:           interrupt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         interrupt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           gs_done_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         gs_done_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
