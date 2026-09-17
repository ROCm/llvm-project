; REQUIRES: comgr-has-transpiler

; gfx1250 is the ISA the rest of the raiser fixtures assemble for, and the one
; that spells both the program-counter capture and the 64-bit scalar arithmetic
; the chains below are built from.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; The raised IR is fed back to the assembly parser, which verifies it. A jump
; into the entry block, or a block left without a terminator, is caught there
; rather than by a pattern below.
; RUN: %transpile_cli %t.hsaco --emit-ir=dispatch_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=call_return_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=two_call_sites_kernel \
; RUN:   | %llvm-as -o /dev/null

; RUN: %transpile_cli %t.hsaco --emit-ir=dispatch_kernel \
; RUN:   | %FileCheck %s --check-prefix=DISPATCH
; RUN: %transpile_cli %t.hsaco --emit-ir=call_return_kernel \
; RUN:   | %FileCheck %s --check-prefix=RETURN
; RUN: %transpile_cli %t.hsaco --emit-ir=two_call_sites_kernel \
; RUN:   | %FileCheck %s --check-prefix=TWOSITES

; RUN: not %transpile_cli %t.hsaco --emit-ir=dispatch_partial_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PARTIAL
; RUN: not %transpile_cli %t.hsaco --emit-ir=dispatch_overcap_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=OVERCAP

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

	.globl	dispatch_kernel
	.p2align	8
	.type	dispatch_kernel,@function
; Three paths each leave a different source offset in the pair, so the jump
; goes to one of three blocks. None of them can be named by a branch, so the
; raise dispatches on the offset itself, which is what a program-counter
; capture and a call both leave behind. Falling out of the table cannot happen
; and traps rather than picking a block.
; DISPATCH-LABEL: define amdgpu_kernel void @dispatch_kernel(
dispatch_kernel:
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc0 disp_second
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, disp_first-.
	s_branch disp_join
disp_second:
	s_cmp_eq_u32 s1, 0
	s_cbranch_scc0 disp_third
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, disp_second_target-.
	s_branch disp_join
disp_third:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, disp_third_target-.
disp_join:
; DISPATCH: switch i64 %{{.+}}, label %[[TRAP:.+]] [
; DISPATCH-NEXT: i64 {{.+}}, label %[[FIRST:bb_.+]]
; DISPATCH-NEXT: i64 {{.+}}, label %[[SECOND:bb_.+]]
; DISPATCH-NEXT: i64 {{.+}}, label %[[THIRD:bb_.+]]
; DISPATCH-NEXT: ]
	s_set_pc_i64 s[10:11]
; The first target is where the jump falls through to, which leads a block
; because the jump ends one. The other two sit in the middle of what the decode
; read as straight-line code, and lead blocks only because the jump reaches
; them.
disp_first:
; DISPATCH: [[FIRST]]:
	s_mov_b32 s2, 11
disp_second_target:
; DISPATCH: [[SECOND]]:
	s_mov_b32 s2, 22
disp_third_target:
; DISPATCH: [[THIRD]]:
	s_cvt_f32_u32 s3, s2
	s_endpgm
; DISPATCH: [[TRAP]]:
; DISPATCH-NEXT: call void @llvm.trap()
; DISPATCH-NEXT: unreachable

	.globl	call_return_kernel
	.p2align	8
	.type	call_return_kernel,@function
; The call leaves the offset it returns to in its destination pair, which is a
; source offset like any a capture computes. The callee jumps through that pair
; without writing it, so what reaches the callee is what the call left, and the
; return is a branch back.
; RETURN-LABEL: define amdgpu_kernel void @call_return_kernel(
call_return_kernel:
; RETURN: entry:
; RETURN: br label %[[CALLER:bb_.+]]
; RETURN: [[CALLER]]:
	s_mov_b32 s2, 11
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, ret_callee-.
; RETURN: br label %[[CALLEE:bb_.+]]
	s_swap_pc_i64 s[12:13], s[10:11]
ret_after_call:
; RETURN: [[AFTER:bb_.+]]: {{.*}}preds = %[[CALLEE]]
; RETURN: uitofp i32 33 to float
	s_cvt_f32_u32 s3, s2
	s_branch ret_done
ret_callee:
; RETURN: [[CALLEE]]:
	s_mov_b32 s2, 33
; RETURN: br label %[[AFTER]]
	s_set_pc_i64 s[12:13]
ret_done:
	s_endpgm

	.globl	two_call_sites_kernel
	.p2align	8
	.type	two_call_sites_kernel,@function
; Two calls to one callee leave two different return offsets in the same pair,
; so the return jumps to one of two blocks and dispatches like any other jump
; the analysis cannot narrow to a single offset.
; TWOSITES-LABEL: define amdgpu_kernel void @two_call_sites_kernel(
two_call_sites_kernel:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, two_callee-.
	s_swap_pc_i64 s[12:13], s[10:11]
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, two_callee-.
	s_swap_pc_i64 s[12:13], s[10:11]
	s_cvt_f32_u32 s3, s2
	s_branch two_done
two_callee:
	s_mov_b32 s2, 44
; TWOSITES: switch i64 %{{.+}}, label %{{.+}} [
; TWOSITES-NEXT: i64 {{.+}}, label %bb_
; TWOSITES-NEXT: i64 {{.+}}, label %bb_
; TWOSITES-NEXT: ]
	s_set_pc_i64 s[12:13]
two_done:
	s_endpgm

	.globl	dispatch_partial_kernel
	.p2align	8
	.type	dispatch_partial_kernel,@function
dispatch_partial_kernel:
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc0 partial_else
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, partial_target-.
	s_branch partial_join
partial_else:
; One path leaves an offset in the pair and the other leaves something the
; analysis cannot name. A jump reading that is refused rather than narrowed to
; the path that did name an offset.
	s_mov_b64 s[10:11], 0
partial_join:
; PARTIAL: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[10:11], which some path reaching its block leaves without a source offset
	s_set_pc_i64 s[10:11]
partial_target:
	s_cvt_f32_u32 s3, s2
	s_endpgm

	.globl	dispatch_overcap_kernel
	.p2align	8
	.type	dispatch_overcap_kernel,@function
dispatch_overcap_kernel:
; Every one of these blocks leaves a different source offset in the pair, which
; is one more offset than the jump below may enumerate. Dispatching on the ones
; that fit would send the paths that did not to a block they never reach, so
; the jump is refused instead.
	.set overcap_index, 0
	.rept 17
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, overcap_join + 4 * overcap_index - .
	s_cbranch_scc0 overcap_join
	.set overcap_index, overcap_index + 1
	.endr
overcap_join:
; OVERCAP: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[10:11], which more than 16 source offsets reach
	s_set_pc_i64 s[10:11]
	.rept 17
	s_nop 0
	.endr
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel dispatch_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel call_return_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel two_call_sites_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel dispatch_partial_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel dispatch_overcap_kernel
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
    .name:           dispatch_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         dispatch_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           call_return_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         call_return_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           two_call_sites_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         two_call_sites_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dispatch_partial_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         dispatch_partial_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dispatch_overcap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         dispatch_overcap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
