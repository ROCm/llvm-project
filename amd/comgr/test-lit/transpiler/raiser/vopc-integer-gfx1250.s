; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vopc_predicates,vopc_exec_write,vop3_mask_writes,vop3_cmpx_predicates \
; RUN:   | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopc_predicates
	.p2align	8
	.type	vopc_predicates,@function
; CHECK-LABEL: define amdgpu_kernel void @vopc_predicates(
vopc_predicates:
; CHECK: icmp slt i32
	v_cmp_lt_i32_e32 vcc_lo, v0, v1
; CHECK: icmp slt i32
	v_cmp_lt_i32_e64 s4, v0, v1
; CHECK: icmp eq i32
	v_cmp_eq_i32_e32 vcc_lo, v0, v1
; CHECK: icmp eq i32
	v_cmp_eq_i32_e64 s4, v0, v1
; CHECK: icmp sle i32
	v_cmp_le_i32_e32 vcc_lo, v0, v1
; CHECK: icmp sle i32
	v_cmp_le_i32_e64 s4, v0, v1
; CHECK: icmp sgt i32
	v_cmp_gt_i32_e32 vcc_lo, v0, v1
; CHECK: icmp sgt i32
	v_cmp_gt_i32_e64 s4, v0, v1
; CHECK: icmp ne i32
	v_cmp_ne_i32_e32 vcc_lo, v0, v1
; CHECK: icmp ne i32
	v_cmp_ne_i32_e64 s4, v0, v1
; CHECK: icmp sge i32
	v_cmp_ge_i32_e32 vcc_lo, v0, v1
; CHECK: icmp sge i32
	v_cmp_ge_i32_e64 s4, v0, v1
; CHECK: icmp ult i32
	v_cmp_lt_u32_e32 vcc_lo, v0, v1
; CHECK: icmp ult i32
	v_cmp_lt_u32_e64 s4, v0, v1
; CHECK: icmp eq i32
	v_cmp_eq_u32_e32 vcc_lo, v0, v1
; CHECK: icmp eq i32
	v_cmp_eq_u32_e64 s4, v0, v1
; CHECK: icmp ule i32
	v_cmp_le_u32_e32 vcc_lo, v0, v1
; CHECK: icmp ule i32
	v_cmp_le_u32_e64 s4, v0, v1
; CHECK: icmp ugt i32
	v_cmp_gt_u32_e32 vcc_lo, v0, v1
; CHECK: icmp ugt i32
	v_cmp_gt_u32_e64 s4, v0, v1
; CHECK: icmp ne i32
	v_cmp_ne_u32_e32 vcc_lo, v0, v1
; CHECK: icmp ne i32
	v_cmp_ne_u32_e64 s4, v0, v1
; CHECK: icmp uge i32
	v_cmp_ge_u32_e32 vcc_lo, v0, v1
; CHECK: icmp uge i32
	v_cmp_ge_u32_e64 s4, v0, v1
; CHECK: ret void
	s_endpgm

	.globl	vopc_exec_write
	.p2align	8
	.type	vopc_exec_write,@function
; CHECK-LABEL: define amdgpu_kernel void @vopc_exec_write(
; CHECK: [[HWLANE:%.+]] = call i32 @llvm.amdgcn.mbcnt.hi(
; CHECK: [[ENTRY_ACTIVE:%.+]] = call i1 @llvm.amdgcn.init.whole.wave()
; CHECK: [[ENTRY_MASK:%.+]] = trunc i64 {{%.+}} to i32
vopc_exec_write:
; CMPX intersects the source wave's ballot slice with its current EXEC.
; CHECK: [[CMP:%.+]] = icmp sgt i32 {{.+}}, {{.+}}
; CHECK: [[PRED:%.+]] = select i1 {{.+}}, i1 [[CMP]], i1 false
; CHECK: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED]])
; CHECK: [[BASE:%.+]] = and i32 [[HWLANE]], -32
; CHECK: [[SHIFT:%.+]] = zext i32 [[BASE]] to i64
; CHECK: [[SLICE:%.+]] = lshr i64 [[BALLOT]], [[SHIFT]]
; CHECK: [[MASK:%.+]] = trunc i64 [[SLICE]] to i32
; CHECK: [[NARROWED:%.+]] = and i32 [[ENTRY_MASK]], [[MASK]]
	v_cmpx_gt_i32_e32 v0, v1
; The following vector write uses the narrowed EXEC.
; CHECK: [[LANE:%.+]] = lshr i32 [[NARROWED]], {{.+}}
; CHECK: [[BIT:%.+]] = and i32 [[LANE]], 1
; CHECK: [[ACTIVE:%.+]] = icmp ne i32 [[BIT]], 0
; CHECK: [[DISPATCHED:%.+]] = select i1 [[ENTRY_ACTIVE]], i1 [[ACTIVE]], i1 false
; CHECK: br i1 [[DISPATCHED]]
	v_add_f32_e32 v2, v0, v1
; CHECK: ret void
	s_endpgm

	.globl	vop3_mask_writes
	.p2align	8
	.type	vop3_mask_writes,@function
; CHECK-LABEL: define amdgpu_kernel void @vop3_mask_writes(
; CHECK: [[HWLANE:%.+]] = call i32 @llvm.amdgcn.mbcnt.hi(
; CHECK: [[ENTRY_ACTIVE:%.+]] = call i1 @llvm.amdgcn.init.whole.wave()
; CHECK: [[ENTRY_MASK:%.+]] = trunc i64 {{%.+}} to i32
vop3_mask_writes:
	s_mov_b32 s4, -1
	s_mov_b32 s5, 42
	s_mov_b32 vcc_hi, 7
	s_mov_b32 exec_hi, 9
	s_and_b32 exec_lo, exec_lo, 5
; CHECK: [[INITIAL_EXEC:%.+]] = and i32 [[ENTRY_MASK]], 5
; CHECK: [[SIGNED:%.+]] = icmp sgt i32 {{.+}}, -1
; CHECK: [[LANE:%.+]] = lshr i32 [[INITIAL_EXEC]], {{.+}}
; CHECK: [[LOWBIT:%.+]] = and i32 [[LANE]], 1
; CHECK: [[ACTIVE:%.+]] = icmp ne i32 [[LOWBIT]], 0
; CHECK: [[DISPATCHED:%.+]] = select i1 [[ENTRY_ACTIVE]], i1 [[ACTIVE]], i1 false
; CHECK: [[PRED:%.+]] = select i1 [[DISPATCHED]], i1 [[SIGNED]], i1 false
; CHECK: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED]])
; CHECK: [[BASE:%.+]] = and i32 [[HWLANE]], -32
; CHECK: [[SHIFT:%.+]] = zext i32 [[BASE]] to i64
; CHECK: [[SLICE:%.+]] = lshr i64 [[BALLOT]], [[SHIFT]]
; CHECK: [[MASK:%.+]] = trunc i64 [[SLICE]] to i32
	v_cmp_gt_i32_e64 s4, v0, s4
; CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[MASK]], i32 42)
	s_add_u32 s6, s4, s5
; CHECK: select i1 [[PRED]], i32 2, i32 1
	v_cndmask_b32_e64 v2, 1, 2, s4
; CHECK: [[UNSIGNED:%.+]] = icmp ugt i32 {{.+}}, -1
; CHECK: [[VCC:%.+]] = select i1 {{.+}}, i1 [[UNSIGNED]], i1 false
	v_cmp_gt_u32_e64 vcc_lo, v0, 0xffffffff
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[VCC]])
	s_mov_b32 s7, vcc_lo
; CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 7, i32 9)
	s_add_u32 s6, vcc_hi, exec_hi
; CHECK: icmp eq i32
	v_cmp_eq_u32_e32 vcc_lo, s6, v0
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[SAVED:%.+]])
	s_mov_b32 s8, vcc_lo
; CHECK: [[CMPX:%.+]] = icmp slt i32 -1, {{.+}}
; CHECK: [[PREDX:%.+]] = select i1 {{.+}}, i1 [[CMPX]], i1 false
; CHECK: [[BALLOTX:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PREDX]])
; CHECK: [[BASEX:%.+]] = and i32 [[HWLANE]], -32
; CHECK: [[SHIFTX:%.+]] = zext i32 [[BASEX]] to i64
; CHECK: [[SLICEX:%.+]] = lshr i64 [[BALLOTX]], [[SHIFTX]]
; CHECK: [[MASKX:%.+]] = trunc i64 [[SLICEX]] to i32
; CHECK: [[EXEC:%.+]] = and i32 [[INITIAL_EXEC]], [[MASKX]]
	v_cmpx_lt_i32_e64 -1, v0
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[SAVED]])
	s_mov_b32 s9, vcc_lo
; CHECK: [[NEWLANE:%.+]] = lshr i32 [[EXEC]], {{.+}}
; CHECK: [[NEWBIT:%.+]] = and i32 [[NEWLANE]], 1
; CHECK: [[NEWACTIVE:%.+]] = icmp ne i32 [[NEWBIT]], 0
; CHECK: [[NEWDISPATCHED:%.+]] = select i1 [[ENTRY_ACTIVE]], i1 [[NEWACTIVE]], i1 false
; CHECK: br i1 [[NEWDISPATCHED]]
	v_mov_b32_e32 v2, 17
; CHECK: icmp eq i32
	v_cmp_eq_i32_e64 null, v0, v1
; CHECK: icmp ne i32 305419896, {{.+}}
	v_cmp_ne_u32_e64 null, 0x12345678, v0
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[SAVED]])
	s_mov_b32 s9, vcc_lo
; CHECK: ret void
	s_endpgm

	.globl	vop3_cmpx_predicates
	.p2align	8
	.type	vop3_cmpx_predicates,@function
; CHECK-LABEL: define amdgpu_kernel void @vop3_cmpx_predicates(
vop3_cmpx_predicates:
; CHECK: icmp slt i32
	v_cmpx_lt_i32_e32 v0, v1
; CHECK: icmp slt i32
	v_cmpx_lt_i32_e64 v0, v1
; CHECK: icmp eq i32
	v_cmpx_eq_i32_e32 v0, v1
; CHECK: icmp eq i32
	v_cmpx_eq_i32_e64 v0, v1
; CHECK: icmp sle i32
	v_cmpx_le_i32_e32 v0, v1
; CHECK: icmp sle i32
	v_cmpx_le_i32_e64 v0, v1
; CHECK: icmp sgt i32
	v_cmpx_gt_i32_e32 v0, v1
; CHECK: icmp sgt i32
	v_cmpx_gt_i32_e64 v0, v1
; CHECK: icmp ne i32
	v_cmpx_ne_i32_e32 v0, v1
; CHECK: icmp ne i32
	v_cmpx_ne_i32_e64 v0, v1
; CHECK: icmp sge i32
	v_cmpx_ge_i32_e32 v0, v1
; CHECK: icmp sge i32
	v_cmpx_ge_i32_e64 v0, v1
; CHECK: icmp ult i32
	v_cmpx_lt_u32_e32 v0, v1
; CHECK: icmp ult i32
	v_cmpx_lt_u32_e64 v0, v1
; CHECK: icmp eq i32
	v_cmpx_eq_u32_e32 v0, v1
; CHECK: icmp eq i32
	v_cmpx_eq_u32_e64 v0, v1
; CHECK: icmp ule i32
	v_cmpx_le_u32_e32 v0, v1
; CHECK: icmp ule i32
	v_cmpx_le_u32_e64 v0, v1
; CHECK: icmp ugt i32
	v_cmpx_gt_u32_e32 v0, v1
; CHECK: icmp ugt i32
	v_cmpx_gt_u32_e64 v0, v1
; CHECK: icmp ne i32
	v_cmpx_ne_u32_e32 v0, v1
; CHECK: icmp ne i32
	v_cmpx_ne_u32_e64 v0, v1
; CHECK: icmp uge i32
	v_cmpx_ge_u32_e32 v0, v1
; CHECK: icmp uge i32
	v_cmpx_ge_u32_e64 v0, v1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopc_predicates
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 10
	.end_amdhsa_kernel
	.amdhsa_kernel vopc_exec_write
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 10
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_denorm_mode_32 1
		.amdhsa_float_denorm_mode_16_64 3
	.end_amdhsa_kernel
	.amdhsa_kernel vop3_mask_writes
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 10
	.end_amdhsa_kernel
	.amdhsa_kernel vop3_cmpx_predicates
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 10
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vopc_predicates
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         vopc_predicates.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vopc_exec_write
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         vopc_exec_write.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop3_mask_writes
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         vop3_mask_writes.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop3_cmpx_predicates
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         vop3_cmpx_predicates.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
