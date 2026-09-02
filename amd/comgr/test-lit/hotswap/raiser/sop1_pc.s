; REQUIRES: comgr-has-hotswap-transpile

; gfx1250 is the ISA the rest of the raiser fixtures assemble for, and it is
; the one that carries both s_add_pc_i64 and the 96-bit SOP1 literal form the
; wide-displacement kernel below needs.

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=refuse_all_pc_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE

; The raised IR is fed back to the assembly parser, which verifies it. A jump
; into the entry block, or a block left without a terminator, is caught there
; rather than by a pattern below.
; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=addpc_forward_kernel,addpc_wide_kernel,addpc_backward_kernel \
; RUN:   | %llvm-as -o /dev/null

; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=addpc_forward_kernel,addpc_wide_kernel,addpc_backward_kernel \
; RUN:   | %FileCheck %s

; RUN: not %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=getpc_kernel,setpc_kernel,swappc_kernel,addpc_register_kernel \
; RUN:   --emit-ir=rfe_kernel,addpc_past_end_kernel,addpc_inside_wide_kernel \
; RUN:   --emit-ir=addpc_wrap_kernel 2>&1 | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; s_add_pc_i64 displaces the address of the instruction after it by a count of
; bytes. Every kernel here seeds s2 on one path and converts it where the paths
; meet, so the value the conversion reads names where control went.

	.globl	addpc_forward_kernel
	.p2align	8
	.type	addpc_forward_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @addpc_forward_kernel(
addpc_forward_kernel:
	s_mov_b32 s2, 11
; The jump sits four bytes into the kernel and is four bytes wide, so a
; displacement of four reaches twelve bytes in and steps over the seed below.
; CHECK: br label %[[FW_TARGET:.+]]
	s_add_pc_i64 4
	s_mov_b32 s2, 22
; CHECK: [[FW_TARGET]]:
; CHECK: uitofp i32 11 to float
	s_cvt_f32_u32 s3, s2
; CHECK: ret void
	s_endpgm

	.globl	addpc_wide_kernel
	.p2align	8
	.type	addpc_wide_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @addpc_wide_kernel(
addpc_wide_kernel:
	s_mov_b32 s2, 11
; The same jump written with a 64-bit literal, which makes the instruction
; twelve bytes rather than four. The displacement is unchanged, so where it
; reaches moves with the width of the instruction carrying it: the jump ends
; sixteen bytes into the kernel and a displacement of four reaches twenty.
; CHECK: br label %[[WIDE_TARGET:.+]]
	s_add_pc_i64 lit64(4)
	s_mov_b32 s2, 22
; CHECK: [[WIDE_TARGET]]:
; CHECK: uitofp i32 11 to float
	s_cvt_f32_u32 s3, s2
; CHECK: ret void
	s_endpgm

	.globl	addpc_backward_kernel
	.p2align	8
	.type	addpc_backward_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @addpc_backward_kernel(
addpc_backward_kernel:
	s_mov_b32 s2, 0
; CHECK: entry:
; CHECK: br label %[[HEAD:.+]]
; CHECK: [[HEAD]]:
backward_head:
; CHECK: [[NOTTED:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s2, s2
; CHECK: [[SCC:%.+]] = icmp ne i32 [[NOTTED]], 0
; CHECK: [[EXIT_COND:%.+]] = xor i1 [[SCC]], true
; CHECK: br i1 [[EXIT_COND]], label %[[EXIT:.+]], label %[[LATCH:.+]]
	s_cbranch_scc0 backward_exit
; CHECK: [[LATCH]]:
; The jump sits twelve bytes into the kernel and ends at sixteen, so minus
; twelve reaches four, the head of the loop. Read as an unsigned count it
; would leave the kernel, and measured from the jump rather than from its end
; it would reach the entry block, which no block may branch to.
; CHECK: br label %[[HEAD]]
	s_add_pc_i64 -12
; CHECK: [[EXIT]]:
backward_exit:
; CHECK: uitofp i32 [[NOTTED]] to float
	s_cvt_f32_u32 s3, s2
; CHECK: ret void
	s_endpgm

; Every other way of writing the program counter is refused, and each carries
; the reason it cannot be stated in the raised kernel.

	.globl	getpc_kernel
	.p2align	8
	.type	getpc_kernel,@function
getpc_kernel:
; REFUSE: unsupported-instruction-form: s_get_pc_i64 {{.+}} :: captures a source address
	s_get_pc_i64 s[10:11]
; The captured address escaping into arithmetic changes nothing: the refusal
; is on the capture, so no consumer of it is ever reached.
	s_add_pc_i64 s[10:11]
	s_endpgm

	.globl	setpc_kernel
	.p2align	8
	.type	setpc_kernel,@function
setpc_kernel:
; REFUSE: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: jumps to a register value
	s_set_pc_i64 s[10:11]
	s_endpgm

	.globl	swappc_kernel
	.p2align	8
	.type	swappc_kernel,@function
swappc_kernel:
; A call whose target the raise cannot resolve to a block, and whose return
; address it has nowhere to put.
; REFUSE: unsupported-instruction-form: s_swap_pc_i64 {{.+}} :: calls through a register value
	s_swap_pc_i64 s[12:13], s[10:11]
	s_endpgm

	.globl	addpc_register_kernel
	.p2align	8
	.type	addpc_register_kernel,@function
addpc_register_kernel:
; REFUSE: unsupported-instruction-form: s_add_pc_i64 {{.+}} :: displacement is not a constant
	s_add_pc_i64 s[10:11]
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
; REFUSE: unsupported-instruction-form: s_rfe_i64 {{.+}} :: returns from an exception handler
	s_rfe_i64 s[10:11]
	s_endpgm

	.globl	refuse_all_pc_kernel
	.p2align	8
	.type	refuse_all_pc_kernel,@function
refuse_all_pc_kernel:
; The opcodes reach the raiser under the names the map gives them, whichever
; mnemonic the source ISA spells them with.
; DECODE: S_GETPC_B64  s_get_pc_i64 s[10:11]
	s_get_pc_i64 s[10:11]
; DECODE-NEXT: S_SETPC_B64  s_set_pc_i64 s[10:11]
	s_set_pc_i64 s[10:11]
; DECODE-NEXT: S_SWAPPC_B64  s_swap_pc_i64 s[12:13], s[10:11]
	s_swap_pc_i64 s[12:13], s[10:11]
; DECODE-NEXT: S_ADD_PC_I64  s_add_pc_i64 s[10:11]
	s_add_pc_i64 s[10:11]
; DECODE-NEXT: S_RFE_B64  s_rfe_i64 s[10:11]
	s_rfe_i64 s[10:11]
	s_endpgm

	.globl	addpc_past_end_kernel
	.p2align	8
	.type	addpc_past_end_kernel,@function
addpc_past_end_kernel:
; REFUSE: decodeKernel: branch at .text offset 0x{{.+}} targets 0x{{.+}}, outside the kernel extent
	s_add_pc_i64 0x1000
	s_endpgm

	.globl	addpc_inside_wide_kernel
	.p2align	8
	.type	addpc_inside_wide_kernel,@function
addpc_inside_wide_kernel:
; The move below spans three dwords, so a displacement of four bytes from the
; end of the jump lands in the middle of it.
; REFUSE: decodeKernel: branch target 0x{{.+}} is not the first byte of a decoded instruction
	s_add_pc_i64 4
	s_mov_b64 s[0:1], 0x123456789abcdef
	s_endpgm

	.globl	addpc_wrap_kernel
	.p2align	8
	.type	addpc_wrap_kernel,@function
addpc_wrap_kernel:
; A displacement further back than the jump stands from zero. Text offsets do
; not wrap, so the sum names nothing and is reported as such rather than as the
; enormous offset the arithmetic produces.
; REFUSE: staticBranchTarget: s_add_pc_i64 at .text offset 0x{{.+}} targets an offset outside the address space
	s_add_pc_i64 lit64(0xffffffffffff0000)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel addpc_forward_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel addpc_wide_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel addpc_backward_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel getpc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel swappc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel addpc_register_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_all_pc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel addpc_past_end_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel addpc_inside_wide_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel addpc_wrap_kernel
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
    .name:           addpc_forward_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         addpc_forward_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           addpc_wide_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         addpc_wide_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           addpc_backward_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         addpc_backward_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           getpc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         getpc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           swappc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         swappc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           addpc_register_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         addpc_register_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           rfe_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         rfe_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_all_pc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_all_pc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           addpc_past_end_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         addpc_past_end_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           addpc_inside_wide_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         addpc_inside_wide_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           addpc_wrap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         addpc_wrap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
