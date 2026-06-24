; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=workitem_id_y_kernel \
; RUN:   | %FileCheck %s
;
; Exercises the kernel-entry v0 seed (WaveProjection::emitPackedWorkitemId,
; driven from raiser.cpp): the packed workitem id must reconstruct the Y field
; so a source threadIdx.y read survives instead of folding to 0.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	workitem_id_y_kernel
	.p2align	8
	.type	workitem_id_y_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @workitem_id_y_kernel(
workitem_id_y_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	v_dual_mov_b32 v1, 1 :: v_dual_lshrrev_b32 v0, 8, v0
	v_and_b32_e32 v0, 0xffc, v0
	s_wait_kmcnt 0x0
; The seed packs the native per-lane Y field into v0's [10:19] bits, so the
; source's shift/mask recovers threadIdx.y instead of a constant 0.
; CHECK: %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: %tid_y = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK: %tid_y_shl = shl i32 %tid_y, 10
; CHECK: %tid_xy = or i32 %tid, %tid_y_shl
; CHECK: lshr i32 %tid_xy, 8
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel workitem_id_y_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 64
    .name:           workitem_id_y_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         workitem_id_y_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
