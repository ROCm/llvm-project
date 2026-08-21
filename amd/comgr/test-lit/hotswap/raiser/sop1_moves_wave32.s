; REQUIRES: comgr-has-hotswap-transpile

; The moves whose lifting depends on the source being wave32, or on the wide
; SOP1 literal form. Both need a target that has them.
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=literal64_kernel \
; RUN:   | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @literal64_kernel(

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=exec_pair_kernel \
; RUN:   | %FileCheck %s --check-prefix=EXECPAIR
; EXECPAIR-LABEL: define amdgpu_kernel void @exec_pair_kernel(

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=vcc_pair_kernel \
; RUN:   | %FileCheck %s --check-prefix=VCCPAIR
; VCCPAIR-LABEL: define amdgpu_kernel void @vcc_pair_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	literal64_kernel
	.p2align	8
	.type	literal64_kernel,@function
literal64_kernel:
	s_mov_b64 s[0:1], 0x123456789abcdef
; Both halves of the literal reach the destination pair: 0x89abcdef read as a
; signed dword is -1985229329, and 0x01234567 is 19088743.
; CHECK: [[LO:%.+]] = zext i32 -1985229329 to i64
; CHECK: [[HI:%.+]] = zext i32 19088743 to i64
; CHECK: [[SHL:%.+]] = shl i64 [[HI]], 32
; CHECK: [[JOIN:%.+]] = or i64 [[LO]], [[SHL]]
; CHECK: call i64 @llvm.bitreverse.i64(i64 [[JOIN]])
	s_brev_b64 s[2:3], s[0:1]
; CHECK: ret void
	s_endpgm

; A wave32 source has lanes only in the low half of EXEC, so a 64-bit write
; lands there and the high half goes nowhere.
	.globl	exec_pair_kernel
	.p2align	8
	.type	exec_pair_kernel,@function
exec_pair_kernel:
	s_mov_b32 s2, 0x1234
	s_mov_b32 s3, -1
; EXECPAIR: [[LO:%.+]] = zext i32 4660 to i64
; EXECPAIR: [[HI:%.+]] = zext i32 -1 to i64
; EXECPAIR: [[SHL:%.+]] = shl i64 [[HI]], 32
; EXECPAIR: [[JOIN:%.+]] = or i64 [[LO]], [[SHL]]
; EXECPAIR: [[MASK:%.+]] = trunc i64 [[JOIN]] to i32
	s_mov_b64 exec, s[2:3]
; EXECPAIR: call i32 @llvm.bitreverse.i32(i32 [[MASK]])
	s_brev_b32 s0, exec_lo
; EXECPAIR: ret void
	s_endpgm

; VCC is the other wave mask a pair can name, and it reads back the same way:
; the ballot is as wide as the source wave, and the dword above it holds no
; lanes to report.
	.globl	vcc_pair_kernel
	.p2align	8
	.type	vcc_pair_kernel,@function
vcc_pair_kernel:
	s_mov_b32 vcc_lo, 0x1234
; VCCPAIR: [[BALLOT:%.+]] = call i32 @llvm.amdgcn.ballot.i32(
; VCCPAIR: [[EXT:%.+]] = zext i32 [[BALLOT]] to i64
	s_mov_b64 s[0:1], vcc
	s_brev_b64 s[2:3], s[0:1]
; VCCPAIR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel literal64_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel exec_pair_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel vcc_pair_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
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
    .name:           literal64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         literal64_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           exec_pair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         exec_pair_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vcc_pair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         vcc_pair_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
