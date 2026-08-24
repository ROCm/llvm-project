; REQUIRES: comgr-has-hotswap-transpile

; llvm-mc warns that this kernel lacks the gfx1250 entry prologue. That is a
; warning, not an error, and the object it produces decodes normally; raising
; the prologue needs instruction families this patch does not add.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	skip_kernel
	.p2align	8
	.type	skip_kernel,@function

; gfx1250 spells the branch the same way gfx9 does and displaces it the same
; way, so the recovered CFG has the same shape as the gfx9 fixture's. What
; differs is the mask the condition reads: this source is wave32, so EXEC is
; 32 bits wide.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=skip_kernel | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @skip_kernel(
; CHECK: br label %bb_0x0
; CHECK: bb_0x0:
; CHECK: [[EXECZ:%.+]] = icmp eq i32 -1, 0
; CHECK-NEXT: br i1 [[EXECZ]], label %bb_0x8, label %bb_0x4
; CHECK: bb_0x4:
; CHECK-NEXT: br label %bb_0x8
; CHECK: bb_0x8:
; CHECK-NEXT: ret void

; Raising onto a wave64 target leaves the compare 32 bits wide: what the branch
; tests is the source wave's EXEC, and how many lanes the target runs does not
; change how many the source had. The gfx9 fixture pins the same invariant from
; the other side, where a 64-bit source mask survives a wave32 target.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=skip_kernel \
; RUN:   --target-isa=gfx942 | %FileCheck %s --check-prefix=TOGFX942
; TOGFX942-LABEL: define amdgpu_kernel void @skip_kernel(
; TOGFX942: [[EXECZ:%.+]] = icmp eq i32 -1, 0
; TOGFX942-NEXT: br i1 [[EXECZ]]

; The gfx12 encoding of the branch reaches the same canonical op as the gfx9
; one, which is what lets a single handler arm serve both.
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=skip_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE
skip_kernel:
; DECODE: S_CBRANCH_EXECZ{{.+}}s_cbranch_execz
	s_cbranch_execz .Lskip_end
	s_mov_b32 s0, 1
.Lskip_end:
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel skip_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_wavefront_size32 1
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
    .name:           skip_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         skip_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
