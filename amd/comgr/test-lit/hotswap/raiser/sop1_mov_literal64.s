; REQUIRES: comgr-has-hotswap-transpile

; A literal that needs all 64 bits only encodes on a target with the wide SOP1
; form, so this kernel is built for one.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=literal64_kernel \
; RUN:   | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @literal64_kernel(

; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=literal64_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	literal64_kernel
	.p2align	8
	.type	literal64_kernel,@function
literal64_kernel:
; The wide form spans three dwords, so the instruction after it sits at 0xc.
; DECODE: 0x0{{.+}}S_MOV_B64{{.+}}s_mov_b64 s[0:1], 0x123456789abcdef
	s_mov_b64 s[0:1], 0x123456789abcdef
; Both halves of the literal reach the destination pair: 0x89abcdef read as a
; signed dword is -1985229329, and 0x01234567 is 19088743.
; DECODE: 0xc{{.+}}S_BREV_B64{{.+}}s_brev_b64 s[2:3], s[0:1]
; CHECK: [[LO:%.+]] = zext i32 -1985229329 to i64
; CHECK: [[HI:%.+]] = zext i32 19088743 to i64
; CHECK: [[SHL:%.+]] = shl i64 [[HI]], 32
; CHECK: [[JOIN:%.+]] = or i64 [[LO]], [[SHL]]
; CHECK: call i64 @llvm.bitreverse.i64(i64 [[JOIN]])
	s_brev_b64 s[2:3], s[0:1]
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel literal64_kernel
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
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
