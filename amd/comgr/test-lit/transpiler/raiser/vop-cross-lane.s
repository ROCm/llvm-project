; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --isa=gfx1250 \
; RUN:   --emit-ir=cross_lane,readfirstlane,readfirstlane_different_exec \
; RUN:   | %FileCheck %s --check-prefix=SAME
; RUN: %transpile_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:   --emit-ir=cross_lane | %FileCheck %s --check-prefix=WIDEN
; RUN: not %transpile_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:   --emit-ir=readfirstlane 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %transpile_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:   --emit-ir=readfirstlane_different_exec > %t.different.ll
; RUN: %FileCheck %s --check-prefix=DIFFERENT < %t.different.ll
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.different.ll -o %t.gfx942.o

; REFUSE: unsupported-wave-projection: s_mov_b32 [SOP1]
; REFUSE-SAME: WaveNative cannot prove that EXEC preserves the kernel entry mask
; DIFFERENT-LABEL: define amdgpu_kernel void @readfirstlane_different_exec(
; DIFFERENT: [[WORKITEM:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; DIFFERENT: [[LANE_LO:%.*]] = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; DIFFERENT-NEXT: [[LANE:%.*]] = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 [[LANE_LO]])
; DIFFERENT: [[PRED:%.*]] = icmp eq i32 5, [[WORKITEM]]
; DIFFERENT-NEXT: [[ACTIVE:%.*]] = select i1 {{%.*}}, i1 [[PRED]], i1 false
; DIFFERENT-NEXT: [[BALLOT:%.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[ACTIVE]])
; DIFFERENT-NEXT: [[MASK_BASE:%.*]] = and i32 [[LANE]], -32
; DIFFERENT-NEXT: [[SHIFT:%.*]] = zext i32 [[MASK_BASE]] to i64
; DIFFERENT-NEXT: [[SLICE:%.*]] = lshr i64 [[BALLOT]], [[SHIFT]]
; DIFFERENT-NEXT: [[MASK:%.*]] = trunc i64 [[SLICE]] to i32
; DIFFERENT-NEXT: [[EXEC:%.*]] = and i32 {{%.*}}, [[MASK]]
; DIFFERENT-NEXT: [[FIRST:%.*]] = call i32 @llvm.cttz.i32(i32 [[EXEC]], i1 false)
; DIFFERENT-NEXT: [[ZERO:%.*]] = icmp eq i32 [[EXEC]], 0
; DIFFERENT-NEXT: [[SOURCE_LANE:%.*]] = select i1 [[ZERO]], i32 0, i32 [[FIRST]]
; DIFFERENT-NEXT: [[BASE:%.*]] = and i32 [[LANE]], -32
; DIFFERENT-NEXT: [[TARGET_LANE:%.*]] = or i32 [[BASE]], [[SOURCE_LANE]]
; DIFFERENT-NEXT: [[ADDRESS:%.*]] = shl i32 [[TARGET_LANE]], 2
; DIFFERENT-NEXT: [[RESULT:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[ADDRESS]], i32 [[WORKITEM]])
; DIFFERENT: [[STORED:%.*]] = phi i32 [ [[RESULT]], %{{.*}} ], [ undef, %{{.*}} ]
; DIFFERENT: store i32 [[STORED]], ptr addrspace(1)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cross_lane
	.p2align	8
	.type	cross_lane,@function
cross_lane:
; SAME-LABEL: define amdgpu_kernel void @cross_lane(
	v_mov_b32_e32 v0, s0
; SAME: call i32 @llvm.amdgcn.readlane.i32
; WIDEN: [[READ_ADDR:%.+]] = shl i32 {{.+}}, 2
; WIDEN: call i32 @llvm.amdgcn.ds.bpermute(i32 [[READ_ADDR]], i32 {{.+}})
; WIDEN-NOT: call i32 @llvm.amdgcn.strict.wwm.i32
	v_readlane_b32 s5, v0, 7
	s_mov_b32 s6, 123
	v_mov_b32_e32 v1, 0
	s_mov_b32 exec_lo, 0
; SAME: call i32 @llvm.amdgcn.writelane.i32
; WIDEN: [[SOURCE_LANE:%.+]] = and i32 {{.+}}, 31
; WIDEN: [[IS_SELECTED:%.+]] = icmp eq i32 [[SOURCE_LANE]], 9
; WIDEN: select i1 [[IS_SELECTED]], i32 123, i32 {{.+}}
	v_writelane_b32 v1, s6, 9
; SAME: ret void
; WIDEN: ret void
	s_endpgm

	.globl readfirstlane
	.p2align 8
	.type readfirstlane,@function
readfirstlane:
; SAME-LABEL: define amdgpu_kernel void @readfirstlane(
; SAME: [[FIRST:%.+]] = call i32 @llvm.cttz.i32(i32 -1, i1 false)
; SAME: [[ZERO:%.+]] = icmp eq i32 -1, 0
; SAME: [[LANE:%.+]] = select i1 [[ZERO]], i32 0, i32 [[FIRST]]
; SAME: call i32 @llvm.amdgcn.readlane.i32(i32 {{.+}}, i32 [[LANE]])
	v_readfirstlane_b32 s4, v0
	s_mov_b32 exec_lo, 32
; SAME: [[FIRST:%.+]] = call i32 @llvm.cttz.i32(i32 32, i1 false)
; SAME: [[ZERO:%.+]] = icmp eq i32 32, 0
; SAME: [[LANE:%.+]] = select i1 [[ZERO]], i32 0, i32 [[FIRST]]
; SAME: call i32 @llvm.amdgcn.readlane.i32(i32 {{.+}}, i32 [[LANE]])
	v_readfirstlane_b32 s5, v0
	s_mov_b32 exec_lo, 0
; SAME: [[FIRST:%.+]] = call i32 @llvm.cttz.i32(i32 0, i1 false)
; SAME: [[ZERO:%.+]] = icmp eq i32 0, 0
; SAME: [[LANE:%.+]] = select i1 [[ZERO]], i32 0, i32 [[FIRST]]
; SAME: call i32 @llvm.amdgcn.readlane.i32(i32 {{.+}}, i32 [[LANE]])
	v_readfirstlane_b32 s5, v0
	s_endpgm

	.globl readfirstlane_different_exec
	.p2align 8
	.type readfirstlane_different_exec,@function
readfirstlane_different_exec:
; SAME-LABEL: define amdgpu_kernel void @readfirstlane_different_exec(
; A 64-thread group gives source EXEC masks 0x20 and 0. The reads must
; return workitem IDs 5 and 32, respectively.
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	s_mov_b32 s6, exec_lo
	v_lshlrev_b32 v1, 2, v0
	v_cmpx_eq_u32_e32 5, v0
; SAME: [[EXEC:%.+]] = and i32 -1, {{%.+}}
; SAME: [[FIRST:%.+]] = call i32 @llvm.cttz.i32(i32 [[EXEC]], i1 false)
; SAME: [[ZERO:%.+]] = icmp eq i32 [[EXEC]], 0
; SAME: [[LANE:%.+]] = select i1 [[ZERO]], i32 0, i32 [[FIRST]]
; SAME: call i32 @llvm.amdgcn.readlane.i32(i32 {{.+}}, i32 [[LANE]])
	v_readfirstlane_b32 s4, v0
	s_mov_b32 exec_lo, s6
	v_mov_b32 v2, s4
	global_store_b32 v1, v2, s[2:3]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cross_lane
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 7
	.end_amdhsa_kernel
	.amdhsa_kernel readfirstlane
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.amdhsa_kernel readfirstlane_different_exec
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 7
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
    .name:           cross_lane
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         cross_lane.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 32
    .name: readfirstlane
    .private_segment_fixed_size: 0
    .sgpr_count: 6
    .symbol: readfirstlane.kd
    .vgpr_count: 1
    .wavefront_size: 32
  - .args:
      - .address_space: global
        .name: output
        .offset: 0
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 64
    .reqd_workgroup_size: [64, 1, 1]
    .name: readfirstlane_different_exec
    .private_segment_fixed_size: 0
    .sgpr_count: 7
    .symbol: readfirstlane_different_exec.kd
    .vgpr_count: 3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
