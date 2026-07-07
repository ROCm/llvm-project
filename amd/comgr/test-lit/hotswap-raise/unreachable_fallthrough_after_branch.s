; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=unreachable_fallthrough_after_branch 2>/dev/null | %FileCheck %s

; Unreachable fall-through after an unconditional branch dropped by CFG terminator handling in the raiser.
; CHECK-LABEL: define amdgpu_kernel void @unreachable_fallthrough_after_branch(
; CHECK: br label %bb_0x8
; CHECK: bb_0x8:
; CHECK-NEXT: ret void
; CHECK-NOT: store i32 1

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	unreachable_fallthrough_after_branch
	.p2align	8
	.type	unreachable_fallthrough_after_branch,@function
unreachable_fallthrough_after_branch:
	s_branch 1
	s_mov_b32 s0, 1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel unreachable_fallthrough_after_branch
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unreachable_fallthrough_after_branch
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         unreachable_fallthrough_after_branch.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
