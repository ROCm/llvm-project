; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=vopc_exec_write,vop3_mask_writes \
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
; EXEC: [[VCC:%.+]] = select i1 [[WASACTIVE]], i1 [[CMP]], i1 false
; EXEC: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[VCC]])
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

	.globl	vop3_mask_writes
	.p2align	8
	.type	vop3_mask_writes,@function
; EXEC-LABEL: define amdgpu_kernel void @vop3_mask_writes(
vop3_mask_writes:
	s_mov_b64 s[4:5], -1
	s_mov_b32 exec_lo, 1
; EXEC: [[OLDEXEC:%.+]] = or i64 {{.+}}, 4294967296
	s_mov_b32 exec_hi, 1
; EXEC: [[CMP:%.+]] = icmp ult i32 -1, {{.+}}
; EXEC: [[LANE:%.+]] = lshr i64 [[OLDEXEC]], {{.+}}
; EXEC: [[LOWBIT:%.+]] = and i64 [[LANE]], 1
; EXEC: [[ACTIVE:%.+]] = icmp ne i64 [[LOWBIT]], 0
; EXEC: [[PRED:%.+]] = select i1 [[ACTIVE]], i1 [[CMP]], i1 false
; EXEC: [[MASK:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED]])
; EXEC: [[LOW:%.+]] = trunc i64 [[MASK]] to i32
; EXEC: [[SHIFT:%.+]] = lshr i64 [[MASK]], 32
; EXEC: [[HIGH:%.+]] = trunc i64 [[SHIFT]] to i32
	v_cmp_lt_u32_e64 s[4:5], -1, v0
; EXEC: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[LOW]], i32 [[HIGH]])
	s_add_u32 s6, s4, s5
; EXEC: select i1 [[PRED]], i32 2, i32 1
	v_cndmask_b32_e64 v2, 1, 2, s[4:5]
; EXEC: [[CMPX:%.+]] = icmp sge i32 {{.+}}, -1
; EXEC: [[PREDX:%.+]] = select i1 {{.+}}, i1 [[CMPX]], i1 false
; EXEC: [[MASKX:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PREDX]])
; EXEC: [[NARROWED:%.+]] = and i64 [[OLDEXEC]], [[MASKX]]
	v_cmpx_ge_i32_e64 vcc, v0, -1
; EXEC: call i64 @llvm.amdgcn.ballot.i64(i1 [[PREDX]])
	s_mov_b64 s[6:7], vcc
; EXEC: [[NEWLANE:%.+]] = lshr i64 [[NARROWED]], {{.+}}
; EXEC: [[NEWBIT:%.+]] = and i64 [[NEWLANE]], 1
; EXEC: [[NEWACTIVE:%.+]] = icmp ne i64 [[NEWBIT]], 0
; EXEC: br i1 [[NEWACTIVE]]
	v_mov_b32_e32 v2, 17
; EXEC: [[CMP2:%.+]] = icmp ne i32 {{.+}}, 0
; EXEC: [[PRED2:%.+]] = select i1 {{.+}}, i1 [[CMP2]], i1 false
; EXEC: [[MASK2:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED2]])
; EXEC: [[LOW2:%.+]] = trunc i64 [[MASK2]] to i32
; EXEC: [[SHIFT2:%.+]] = lshr i64 [[MASK2]], 32
; EXEC: [[HIGH2:%.+]] = trunc i64 [[SHIFT2]] to i32
	v_cmpx_ne_u32_e64 s[4:5], v0, 0
; EXEC: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[LOW2]], i32 [[HIGH2]])
	s_add_u32 s6, s4, s5
; EXEC: call i64 @llvm.amdgcn.ballot.i64(i1 [[PREDX]])
	s_mov_b64 s[6:7], vcc
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
	.amdhsa_kernel vop3_mask_writes
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
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
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop3_mask_writes
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         vop3_mask_writes.kd
    .vgpr_count:     3
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
