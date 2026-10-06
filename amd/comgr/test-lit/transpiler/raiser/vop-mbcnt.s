; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --isa=gfx1250 \
; RUN:   --emit-ir=mbcnt_lane_id,mbcnt_mask_sources \
; RUN:   | %FileCheck %s --check-prefix=SAME
; RUN: %transpile_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:   --emit-ir=mbcnt_lane_id,mbcnt_mask_sources \
; RUN:   | %FileCheck %s --check-prefix=WIDEN

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	mbcnt_lane_id
	.p2align	8
	.type	mbcnt_lane_id,@function
mbcnt_lane_id:
; SAME-LABEL: define amdgpu_kernel void @mbcnt_lane_id(
; WIDEN-LABEL: define amdgpu_kernel void @mbcnt_lane_id(
; The lane-id idiom: an all-ones mask counts every lane below this one. On a
; wider target the count must stay within the source wave.
; WIDEN: [[LANE:%.+]] = and i32 {{.+}}, 31
; WIDEN: [[BIT:%.+]] = shl i32 1, [[LANE]]
; WIDEN: [[BELOW:%.+]] = sub i32 [[BIT]], 1
; WIDEN: [[SELECTED:%.+]] = and i32 -1, [[BELOW]]
; WIDEN: [[COUNT:%.+]] = call i32 @llvm.ctpop.i32(i32 [[SELECTED]])
; WIDEN: add i32 [[COUNT]], 0
	v_mbcnt_lo_u32_b32 v1, -1, 0
; A wave32 lane never reaches the high half of the wave mask, so widening
; forwards src1 unchanged.
; SAME: call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 {{.+}})
; WIDEN-NOT: call i32 @llvm.amdgcn.mbcnt.hi
	v_mbcnt_hi_u32_b32 v1, -1, v1
; WIDEN: ret void
	s_endpgm

	.globl	mbcnt_mask_sources
	.p2align	8
	.type	mbcnt_mask_sources,@function
mbcnt_mask_sources:
; SAME-LABEL: define amdgpu_kernel void @mbcnt_mask_sources(
; WIDEN-LABEL: define amdgpu_kernel void @mbcnt_mask_sources(
; WIDEN: [[TID:%.+]] = call i32 @llvm.amdgcn.workitem.id.x()
; WIDEN: [[ENTRY:%.+]] = trunc i64 {{%.+}} to i32
; SAME: [[EXEC:%.+]] = and i32 -1, {{.+}}
; WIDEN: [[EXEC:%.+]] = and i32 [[ENTRY]], {{%.+}}
	v_cmpx_gt_u32_e64 16, v0
; Counting the active lanes below this one. The mask operand is EXEC, which
; the widening path reads as the current source wave's slice.
; SAME: call i32 @llvm.amdgcn.mbcnt.lo(i32 [[EXEC]], i32 0)
; WIDEN: [[BIT:%.+]] = shl i32 1, {{.+}}
; WIDEN: [[BELOW:%.+]] = sub i32 [[BIT]], 1
; WIDEN: [[SELECTED:%.+]] = and i32 [[EXEC]], [[BELOW]]
; WIDEN: [[COUNT:%.+]] = call i32 @llvm.ctpop.i32(i32 [[SELECTED]])
; WIDEN: add i32 [[COUNT]], 0
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	v_cmp_gt_u32_e64 s2, 16, v0
; A mask held in an SGPR reaches the count through the same source-wave
; projection as EXEC, by way of the register's wave-mask shadow.
; WIDEN: [[SGPR_MASK:%.+]] = select i1 {{.+}}, i32 {{.+}}, i32 {{.+}}
; WIDEN: [[SGPR_BIT:%.+]] = shl i32 1, {{.+}}
; WIDEN: [[SGPR_BELOW:%.+]] = sub i32 [[SGPR_BIT]], 1
; WIDEN: and i32 [[SGPR_MASK]], [[SGPR_BELOW]]
	v_mbcnt_lo_u32_b32 v1, s2, 0
; A mask held in a VGPR is per-lane data rather than a wave mask, so it is
; read as it stands and only the lane the count runs against is projected.
; WIDEN: [[VGPR_BIT:%.+]] = shl i32 1, {{.+}}
; WIDEN: [[VGPR_BELOW:%.+]] = sub i32 [[VGPR_BIT]], 1
; WIDEN: and i32 [[TID]], [[VGPR_BELOW]]
	v_mbcnt_lo_u32_b32 v2, v0, 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_lane_id
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.amdhsa_kernel mbcnt_mask_sources
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 3
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
    .name: mbcnt_lane_id
    .private_segment_fixed_size: 0
    .sgpr_count: 3
    .symbol: mbcnt_lane_id.kd
    .vgpr_count: 3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: mbcnt_mask_sources
    .private_segment_fixed_size: 0
    .sgpr_count: 3
    .symbol: mbcnt_mask_sources.kd
    .vgpr_count: 3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
