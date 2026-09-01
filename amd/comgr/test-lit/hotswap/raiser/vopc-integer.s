; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=vopc_exec_write \
; RUN:   | %FileCheck %s --check-prefix=EXEC

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	vopc_exec_write
	.p2align	8
	.type	vopc_exec_write,@function
; EXEC-LABEL: define amdgpu_kernel void @vopc_exec_write(
vopc_exec_write:
; A wave64 source holds EXEC at the same width as the wave64 target ballot, so
; the mask reaches the AND without being narrowed. This opcode writes the scalar
; destination it names as well as EXEC.
; EXEC: [[CMP:%.+]] = icmp ugt i32 {{.+}}, {{.+}}
; The bit that reaches the scalar destination is the comparison cleared where
; the lane is inactive, since a masked-off lane reads its bit back as zero.
; EXEC: [[WASACTIVE:%.+]] = icmp ne i64 {{.+}}, 0
; EXEC: [[VCC:%.+]] = and i1 [[CMP]], [[WASACTIVE]]
; EXEC: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; EXEC: [[NARROWED:%.+]] = and i64 -1, [[BALLOT]]
	v_cmpx_gt_u32_e32 vcc, v0, v1
; Reading the scalar destination back is what shows the comparison reached it:
; VCC is held as a per-lane bit and a scalar read of it ballots that bit.
; EXEC: call i64 @llvm.amdgcn.ballot.i64(i1 [[VCC]])
	s_mov_b32 s0, vcc_lo
; The vector write that follows is predicated on the narrowed EXEC, which is
; what makes the store observable: the lane-active bit is recomputed from it
; rather than from the EXEC in force before the comparison.
; EXEC: [[LANE:%.+]] = lshr i64 [[NARROWED]], {{.+}}
; EXEC: [[BIT:%.+]] = and i64 [[LANE]], 1
; EXEC: [[ACTIVE:%.+]] = icmp ne i64 [[BIT]], 0
; EXEC: br i1 [[ACTIVE]]
	v_add_f32_e32 v2, v0, v1
; EXEC: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopc_exec_write
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_denorm_mode_32 1
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vopc_exec_write
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vopc_exec_write.kd
    .vgpr_count:     3
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
