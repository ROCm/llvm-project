; REQUIRES: comgr-has-hotswap-transpile

; gfx1250 is what the rest of the raiser fixtures assemble for, and it is the
; ISA that carries the 96-bit SOP1 literal form the wide-target refusal needs.

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=branch_backward_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE-BACKWARD
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=branch_next_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE-ADJACENT

; The raised IR is fed back to the assembly parser, which verifies it. A block
; left without a terminator, or a branch to a block of another function, is
; caught there rather than by a pattern below.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=branch_forward_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=unreachable_tail_kernel \
; RUN:   | %llvm-as -o /dev/null

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=branch_forward_kernel \
; RUN:   | %FileCheck %s --check-prefix=FORWARD
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=branch_next_kernel \
; RUN:   | %FileCheck %s --check-prefix=ADJACENT
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=branch_backward_kernel \
; RUN:   | %FileCheck %s --check-prefix=BACKWARD
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_scc1_kernel \
; RUN:   | %FileCheck %s --check-prefix=SCC1
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_scc0_kernel \
; RUN:   | %FileCheck %s --check-prefix=SCC0
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_execz_kernel \
; RUN:   | %FileCheck %s --check-prefix=EXECZ
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_execnz_kernel \
; RUN:   | %FileCheck %s --check-prefix=EXECNZ
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_vccz_kernel \
; RUN:   | %FileCheck %s --check-prefix=VCCZ
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_vccnz_kernel \
; RUN:   | %FileCheck %s --check-prefix=VCCNZ
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=unreachable_tail_kernel \
; RUN:   | %FileCheck %s --check-prefix=TAIL

; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=m0_across_block_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=M0-ACROSS
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=past_kernel_end_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=PAST-END
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=inside_wide_inst_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=INSIDE-WIDE
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=unhandled_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; Each kernel seeds s2 with a value per path and converts it at the join, so
; the phi the conversion reads names the path control took. s_not_b32 is the
; SCC source: notting 0 gives a non-zero result and sets SCC, notting -1 gives
; zero and clears it.

	.globl	branch_forward_kernel
	.p2align	8
	.type	branch_forward_kernel,@function
; FORWARD-LABEL: define amdgpu_kernel void @branch_forward_kernel(
branch_forward_kernel:
	s_not_b32 s4, 0
; FORWARD: [[FW_COND:%scc0.*]] = xor i1 true, true
; FORWARD: br i1 [[FW_COND]], label %[[FW_ELSE:.+]], label %[[FW_THEN:.+]]
	s_cbranch_scc0 forward_else
; FORWARD: [[FW_THEN]]:
	s_mov_b32 s2, 11
; The forward branch leaves the taken arm, skipping the arm below it.
; FORWARD: br label %[[FW_JOIN:.+]]
	s_branch forward_join
; FORWARD: [[FW_ELSE]]:
forward_else:
	s_mov_b32 s2, 22
; FORWARD: br label %[[FW_JOIN]]
; FORWARD: [[FW_JOIN]]:
forward_join:
; FORWARD: [[FW_VAL:%.+]] = phi i32 [ 22, %[[FW_ELSE]] ], [ 11, %[[FW_THEN]] ]
; FORWARD: uitofp i32 [[FW_VAL]] to float
	s_cvt_f32_u32 s3, s2
; FORWARD: ret void
	s_endpgm

	.globl	branch_next_kernel
	.p2align	8
	.type	branch_next_kernel,@function
; ADJACENT-LABEL: define amdgpu_kernel void @branch_next_kernel(
branch_next_kernel:
	s_mov_b32 s2, 7
; A displacement of zero names the instruction right after the branch, which
; still leads a block of its own.
; DECODE-ADJACENT: S_MOV_B32{{.+}}s_mov_b32 s2, 7
; DECODE-ADJACENT: S_BRANCH{{.+}}s_branch 0
; ADJACENT: br label %[[ADJ_BB:.+]]
	s_branch next_target
; ADJACENT: [[ADJ_BB]]:
next_target:
; ADJACENT: uitofp i32 7 to float
	s_cvt_f32_u32 s3, s2
; ADJACENT: ret void
	s_endpgm

	.globl	branch_backward_kernel
	.p2align	8
	.type	branch_backward_kernel,@function
; BACKWARD-LABEL: define amdgpu_kernel void @branch_backward_kernel(
branch_backward_kernel:
	s_mov_b32 s2, 0
; BACKWARD: entry:
; BACKWARD: br label %[[LOOP:.+]]
; BACKWARD: [[LOOP]]:
; A backward displacement is the sign-extended one: the raw field reads 65534,
; which is -2 dwords from the instruction after the branch.
; DECODE-BACKWARD: 0x{{.+}}4  S_NOT_B32  s_not_b32 s2, s2
; DECODE-BACKWARD-NEXT: 0x{{.+}}8  S_CBRANCH_SCC1  s_cbranch_scc1 65534
backward_head:
; BACKWARD: [[NOTTED:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s2, s2
; BACKWARD: [[LOOP_COND:%.+]] = icmp ne i32 [[NOTTED]], 0
; BACKWARD: br i1 [[LOOP_COND]], label %[[LOOP]], label %[[AFTER:.+]]
	s_cbranch_scc1 backward_head
; BACKWARD: [[AFTER]]:
; BACKWARD: uitofp i32 [[NOTTED]] to float
	s_cvt_f32_u32 s3, s2
; BACKWARD: ret void
	s_endpgm

; Both directions of each condition, so a polarity flip shows up as the wrong
; test on one of the two branches.

	.globl	cbranch_scc1_kernel
	.p2align	8
	.type	cbranch_scc1_kernel,@function
; SCC1-LABEL: define amdgpu_kernel void @cbranch_scc1_kernel(
cbranch_scc1_kernel:
	s_not_b32 s4, 0
; SCC1: br i1 true, label
	s_cbranch_scc1 scc1_second
	s_mov_b32 s2, 11
scc1_second:
	s_not_b32 s4, -1
; SCC1: br i1 false, label
	s_cbranch_scc1 scc1_join
	s_mov_b32 s2, 22
scc1_join:
	s_cvt_f32_u32 s3, s2
; SCC1: ret void
	s_endpgm

	.globl	cbranch_scc0_kernel
	.p2align	8
	.type	cbranch_scc0_kernel,@function
; SCC0-LABEL: define amdgpu_kernel void @cbranch_scc0_kernel(
cbranch_scc0_kernel:
	s_not_b32 s4, -1
; SCC0: [[SCC0_SET:%scc0.*]] = xor i1 false, true
; SCC0: br i1 [[SCC0_SET]], label
	s_cbranch_scc0 scc0_second
	s_mov_b32 s2, 11
scc0_second:
	s_not_b32 s4, 0
; SCC0: [[SCC0_CLEAR:%scc0.*]] = xor i1 true, true
; SCC0: br i1 [[SCC0_CLEAR]], label
	s_cbranch_scc0 scc0_join
	s_mov_b32 s2, 22
scc0_join:
	s_cvt_f32_u32 s3, s2
; SCC0: ret void
	s_endpgm

	.globl	cbranch_execz_kernel
	.p2align	8
	.type	cbranch_execz_kernel,@function
; EXECZ-LABEL: define amdgpu_kernel void @cbranch_execz_kernel(
cbranch_execz_kernel:
	s_mov_b32 exec_lo, 0
; EXECZ: [[EXECZ_EMPTY:%execz.*]] = icmp eq i32 0, 0
; EXECZ: br i1 [[EXECZ_EMPTY]], label
	s_cbranch_execz execz_second
	s_mov_b32 s2, 11
execz_second:
	s_mov_b32 exec_lo, -1
; EXECZ: [[EXECZ_FULL:%execz.*]] = icmp eq i32 -1, 0
; EXECZ: br i1 [[EXECZ_FULL]], label
	s_cbranch_execz execz_join
	s_mov_b32 s2, 22
execz_join:
	s_cvt_f32_u32 s3, s2
; EXECZ: ret void
	s_endpgm

	.globl	cbranch_execnz_kernel
	.p2align	8
	.type	cbranch_execnz_kernel,@function
; EXECNZ-LABEL: define amdgpu_kernel void @cbranch_execnz_kernel(
cbranch_execnz_kernel:
	s_mov_b32 exec_lo, -1
; EXECNZ: [[NZ_FULL:%execz.*]] = icmp eq i32 -1, 0
; EXECNZ: [[NZ_FULL_COND:%execnz.*]] = xor i1 [[NZ_FULL]], true
; EXECNZ: br i1 [[NZ_FULL_COND]], label
	s_cbranch_execnz execnz_second
	s_mov_b32 s2, 11
execnz_second:
	s_mov_b32 exec_lo, 0
; EXECNZ: [[NZ_EMPTY:%execz.*]] = icmp eq i32 0, 0
; EXECNZ: [[NZ_EMPTY_COND:%execnz.*]] = xor i1 [[NZ_EMPTY]], true
; EXECNZ: br i1 [[NZ_EMPTY_COND]], label
	s_cbranch_execnz execnz_join
	s_mov_b32 s2, 22
execnz_join:
	s_cvt_f32_u32 s3, s2
; EXECNZ: ret void
	s_endpgm

; VCC is read as the mask the source wave holding this target lane sees, so the
; condition tests a ballot of that lane's bit rather than the raw register.

	.globl	cbranch_vccz_kernel
	.p2align	8
	.type	cbranch_vccz_kernel,@function
; VCCZ-LABEL: define amdgpu_kernel void @cbranch_vccz_kernel(
cbranch_vccz_kernel:
	s_mov_b32 vcc_lo, 0
; VCCZ: lshr i32 0, %lane_lo
; VCCZ: [[Z_BALLOT:%.+]] = call i32 @llvm.amdgcn.ballot.i32(
; VCCZ: [[Z_EMPTY:%vccz.*]] = icmp eq i32 [[Z_BALLOT]], 0
; VCCZ: br i1 [[Z_EMPTY]], label
	s_cbranch_vccz vccz_second
	s_mov_b32 s2, 11
vccz_second:
	s_mov_b32 vcc_lo, -1
; VCCZ: lshr i32 -1, %lane_lo
; VCCZ: [[Z_BALLOT_SET:%.+]] = call i32 @llvm.amdgcn.ballot.i32(
; VCCZ: [[Z_SET:%vccz.*]] = icmp eq i32 [[Z_BALLOT_SET]], 0
; VCCZ: br i1 [[Z_SET]], label
	s_cbranch_vccz vccz_join
	s_mov_b32 s2, 22
vccz_join:
	s_cvt_f32_u32 s3, s2
; VCCZ: ret void
	s_endpgm

	.globl	cbranch_vccnz_kernel
	.p2align	8
	.type	cbranch_vccnz_kernel,@function
; VCCNZ-LABEL: define amdgpu_kernel void @cbranch_vccnz_kernel(
cbranch_vccnz_kernel:
	s_mov_b32 vcc_lo, -1
; VCCNZ: lshr i32 -1, %lane_lo
; VCCNZ: [[NZ_SET:%vccz.*]] = icmp eq i32 {{.+}}, 0
; VCCNZ: [[NZ_SET_COND:%vccnz.*]] = xor i1 [[NZ_SET]], true
; VCCNZ: br i1 [[NZ_SET_COND]], label
	s_cbranch_vccnz vccnz_second
	s_mov_b32 s2, 11
vccnz_second:
	s_mov_b32 vcc_lo, 0
; VCCNZ: lshr i32 0, %lane_lo
; VCCNZ: [[NZ_ZERO:%vccz.*]] = icmp eq i32 {{.+}}, 0
; VCCNZ: [[NZ_ZERO_COND:%vccnz.*]] = xor i1 [[NZ_ZERO]], true
; VCCNZ: br i1 [[NZ_ZERO_COND]], label
	s_cbranch_vccnz vccnz_join
	s_mov_b32 s2, 22
vccnz_join:
	s_cvt_f32_u32 s3, s2
; VCCNZ: ret void
	s_endpgm

	.globl	unreachable_tail_kernel
	.p2align	8
	.type	unreachable_tail_kernel,@function
; TAIL-LABEL: define amdgpu_kernel void @unreachable_tail_kernel(
unreachable_tail_kernel:
	s_mov_b32 s2, 5
	s_not_b32 s4, 0
; TAIL: br i1 true, label %[[TAIL_JOIN:.+]], label %[[TAIL_EXIT:.+]]
	s_cbranch_scc1 tail_join
; TAIL: [[TAIL_EXIT]]:
; TAIL-NEXT: ret void
	s_endpgm
; The scan runs past that terminator to reach the branch target, so what sits
; between the two leads a block nothing reaches, and only the seed the branch
; carried arrives at the join.
	s_mov_b32 s2, 33
; TAIL: [[TAIL_JOIN]]:
tail_join:
; TAIL: uitofp i32 5 to float
	s_cvt_f32_u32 s3, s2
; TAIL: unreached{{.+}}:
; TAIL: br label %[[TAIL_JOIN]]
	s_endpgm

	.globl	m0_across_block_kernel
	.p2align	8
	.type	m0_across_block_kernel,@function
m0_across_block_kernel:
; M0 is a fact of the block that wrote it. The relative move sits in the block
; the branch leads to, so the constant does not reach it and the move is
; refused instead of resolving against a register M0 no longer names.
; M0-ACROSS: unsupported-instruction-form: s_movrels_b32{{.+}}movrel: M0 does not hold a constant here
	s_mov_b32 m0, 10
	s_mov_b32 s17, 17
	s_branch m0_across_target
m0_across_target:
	s_movrels_b32 s5, s7
	s_endpgm

	.globl	past_kernel_end_kernel
	.p2align	8
	.type	past_kernel_end_kernel,@function
past_kernel_end_kernel:
; PAST-END: decodeKernel: branch at .text offset 0x{{.+}} targets 0x{{.+}}, outside the kernel extent
	s_branch 10000
	s_endpgm

	.globl	inside_wide_inst_kernel
	.p2align	8
	.type	inside_wide_inst_kernel,@function
inside_wide_inst_kernel:
; The move below spans three dwords, so a displacement of one dword from the
; instruction after the branch lands in the middle of it.
; INSIDE-WIDE: decodeKernel: branch target 0x{{.+}} is not the first byte of a decoded instruction
	s_branch 1
	s_mov_b64 s[0:1], 0x123456789abcdef
	s_endpgm

	.globl	unhandled_kernel
	.p2align	8
	.type	unhandled_kernel,@function
unhandled_kernel:
; UNHANDLED: unsupported-instruction-form: s_rfe_i64
	s_rfe_i64 s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel branch_forward_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel branch_next_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel branch_backward_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_scc1_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_scc0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_execz_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_execnz_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_vccz_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_vccnz_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel unreachable_tail_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel m0_across_block_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel past_kernel_end_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel inside_wide_inst_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel unhandled_kernel
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
    .name:           branch_forward_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         branch_forward_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           branch_next_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         branch_next_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           branch_backward_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         branch_backward_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_scc1_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         cbranch_scc1_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_scc0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         cbranch_scc0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_execz_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         cbranch_execz_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_execnz_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         cbranch_execnz_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_vccz_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         cbranch_vccz_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_vccnz_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         cbranch_vccnz_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unreachable_tail_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         unreachable_tail_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           m0_across_block_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         m0_across_block_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           past_kernel_end_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         past_kernel_end_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           inside_wide_inst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         inside_wide_inst_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unhandled_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         unhandled_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
