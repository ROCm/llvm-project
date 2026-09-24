; REQUIRES: comgr-has-transpiler

; gfx1250 is the ISA the rest of the raiser fixtures assemble for, and the one
; that spells both the program-counter capture and the 64-bit scalar arithmetic
; the chains below are built from.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; The raised IR is fed back to the assembly parser, which verifies it. A jump
; into the entry block, or a block left without a terminator, is caught there
; rather than by a pattern below.
; RUN: %transpile_cli %t.hsaco --emit-ir=outlined_call_kernel \
; RUN:   | %llvm-as -o /dev/null

; RUN: %transpile_cli %t.hsaco --emit-ir=outlined_call_kernel \
; RUN:   | %FileCheck %s --check-prefix=OUTLINED
; RUN: not %transpile_cli %t.hsaco --emit-ir=unowned_call_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNOWNED
; RUN: not %transpile_cli %t.hsaco --emit-ir=interior_call_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=INTERIOR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; A call may leave the kernel altogether for a helper the compiler outlined,
; which is code no decode of the kernel's own extent ever reads. The raise
; follows it into whichever function symbol covers the offset it reaches and
; lifts that helper alongside its caller.

; The helper sits below the kernel, so the entry block reaches the kernel by
; name rather than by being the block the lowest-addressed instruction leads.
; OUTLINED-LABEL: define amdgpu_kernel void @outlined_call_kernel(
; OUTLINED: entry:
; OUTLINED-NEXT: br label %[[CALLER:bb_.+]]

	.globl	outlined_helper
	.hidden	outlined_helper
	.p2align	8
	.type	outlined_helper,@function
; OUTLINED: [[HELPER:bb_.+]]: {{.*}}preds = %[[CALLER]]
outlined_helper:
	s_mov_b32 s2, 33
; OUTLINED: br label %[[RETURN:bb_.+]]
	s_set_pc_i64 s[12:13]
	.size	outlined_helper, .-outlined_helper

	.globl	outlined_call_kernel
	.p2align	8
	.type	outlined_call_kernel,@function
; OUTLINED: [[CALLER]]:
outlined_call_kernel:
; The helper lies below the capture, so the displacement borrows and both
; halves of the pair take part.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, outlined_helper@rel32@lo+4
	s_addc_u32 s11, s11, outlined_helper@rel32@hi+12
; OUTLINED: br label %[[HELPER]]
	s_swap_pc_i64 s[12:13], s[10:11]
outlined_return:
; OUTLINED: [[RETURN]]:
; OUTLINED: uitofp i32 33 to float
	s_cvt_f32_u32 s3, s2
	s_endpgm
	.size	outlined_call_kernel, .-outlined_call_kernel

	.globl	unowned_call_kernel
	.p2align	8
	.type	unowned_call_kernel,@function
unowned_call_kernel:
; The offset this reaches lies outside the kernel and inside no function
; symbol, so there is nothing to follow the call into and it is refused.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, unowned_gap-.
; UNOWNED: unsupported-instruction-form: s_swap_pc_i64 {{.+}} :: reaches source offset 0x{{.+}}, which no decoded instruction starts at
	s_swap_pc_i64 s[12:13], s[10:11]
	s_endpgm
	.size	unowned_call_kernel, .-unowned_call_kernel
unowned_gap:
	s_mov_b32 s2, 55
	s_endpgm

	.globl	interior_call_kernel
	.p2align	8
	.type	interior_call_kernel,@function
interior_call_kernel:
; The offset this reaches lies inside the kernel's own function symbol but two
; bytes past an instruction boundary, so it points into the middle of an
; instruction the decode already read. Reading those bytes again would decode
; them the same way, so the call is refused rather than followed.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, interior_target-.+2
; INTERIOR: unsupported-instruction-form: s_swap_pc_i64 {{.+}} :: reaches source offset 0x{{.+}}, which no decoded instruction starts at
	s_swap_pc_i64 s[12:13], s[10:11]
interior_target:
	s_mov_b32 s2, 77
	s_endpgm
	.size	interior_call_kernel, .-interior_call_kernel

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel outlined_call_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel unowned_call_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel interior_call_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
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
    .name:           outlined_call_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         outlined_call_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unowned_call_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         unowned_call_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           interior_call_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         interior_call_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
