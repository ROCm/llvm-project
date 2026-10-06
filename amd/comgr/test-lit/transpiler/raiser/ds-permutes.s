; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir > %t.ll
; RUN: %FileCheck %s --input-file=%t.ll
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir > %t.same.ll
; RUN: %FileCheck %s --input-file=%t.same.ll --check-prefix=SAME

	.amdhsa_code_object_version 6
	.text

	.globl bpermute
	.p2align 8
	.type bpermute,@function
; CHECK-LABEL: define amdgpu_kernel void @bpermute(
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()
; SAME-LABEL: define amdgpu_kernel void @bpermute(
bpermute:
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	v_lshlrev_b32 v1, 2, v0
	v_lshlrev_b32 v2, 3, v0
; The offset field biases the selector, which is then clipped into the source
; wave the reading lane belongs to. An inactive lane hands the gather a zero,
; which is what the source returns to a lane that reads an inactive one.
; CHECK: [[SELECTOR:%.+]] = add i32 {{.+}}, 8
; CHECK-NEXT: [[IN_WAVE:%.+]] = and i32 [[SELECTOR]], 127
; CHECK-NEXT: [[BASE:%.+]] = and i32 [[LANE:%.+]], -32
; CHECK-NEXT: [[BYTES:%.+]] = shl i32 [[BASE]], 2
; CHECK-NEXT: [[REBASED:%.+]] = or i32 [[IN_WAVE]], [[BYTES]]
; CHECK-NEXT: [[DATA:%.+]] = select i1 [[ACTIVE:%.+]], i32 [[RAW:%.+]], i32 0
; CHECK-NEXT: [[GATHER:%.+]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[REBASED]], i32 [[DATA]])
; CHECK-NEXT: br i1 [[ACTIVE]], label %[[DO:.+]], label %[[SKIP:.+]]
; CHECK: [[SKIP]]:
; CHECK-NEXT: phi i32 [ [[GATHER]], %[[DO]] ], [ undef, {{.+}} ]
; A target wave that holds one source wave addresses the lanes the source
; named, so the selector reaches the gather as the source computed it.
; SAME: [[SELECTOR:%.+]] = add i32 {{.+}}, 8
; SAME-NEXT: [[DATA:%.+]] = select i1 {{.+}}, i32 {{.+}}, i32 0
; SAME-NEXT: call i32 @llvm.amdgcn.ds.bpermute(i32 [[SELECTOR]], i32 [[DATA]])
	ds_bpermute_b32 v3, v1, v2 offset:8
	s_wait_dscnt 0
	global_store_b32 v0, v3, s[2:3]
	s_endpgm

	.globl bpermute_fi
	.p2align 8
	.type bpermute_fi,@function
; CHECK-LABEL: define amdgpu_kernel void @bpermute_fi(
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()
bpermute_fi:
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	v_lshlrev_b32 v1, 2, v0
	v_lshlrev_b32 v2, 3, v0
; This form reads the lanes the source left inactive, so the data reaches the
; gather as the source register holds it.
; CHECK: [[RAW:%.+]] = shl i32 {{.+}}, 3
; CHECK: [[IN_WAVE:%.+]] = and i32 {{.+}}, 127
; CHECK: [[REBASED:%.+]] = or i32 [[IN_WAVE]], {{.+}}
; CHECK-NEXT: [[GATHER:%.+]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[REBASED]], i32 [[RAW]])
; CHECK-NEXT: br i1 {{%.+}}, label %[[DO:.+]], label %[[SKIP:.+]]
; CHECK: [[SKIP]]:
; CHECK-NEXT: phi i32 [ [[GATHER]], %[[DO]] ], [ undef, {{.+}} ]
	ds_bpermute_fi_b32 v3, v1, v2 offset:8
	s_wait_dscnt 0
	global_store_b32 v0, v3, s[2:3]
	s_endpgm

	.globl permute
	.p2align 8
	.type permute,@function
; CHECK-LABEL: define amdgpu_kernel void @permute(
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()
; SAME-LABEL: define amdgpu_kernel void @permute(
permute:
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	v_lshlrev_b32 v1, 2, v0
	v_mov_b32 v2, 7
; A lane the source left inactive scatters to itself, so it cannot overwrite
; the result an active lane collects.
; CHECK: [[SELECTOR:%.+]] = add i32 {{.+}}, 8
; CHECK-NEXT: [[IN_WAVE:%.+]] = and i32 [[SELECTOR]], 127
; CHECK-NEXT: [[BASE:%.+]] = and i32 [[LANE:%.+]], -32
; CHECK-NEXT: [[BYTES:%.+]] = shl i32 [[BASE]], 2
; CHECK-NEXT: [[REBASED:%.+]] = or i32 [[IN_WAVE]], [[BYTES]]
; CHECK-NEXT: [[OWN:%.+]] = shl i32 [[LANE]], 2
; CHECK-NEXT: [[TARGET:%.+]] = select i1 [[ACTIVE:%.+]], i32 [[REBASED]], i32 [[OWN]]
; CHECK-NEXT: [[SCATTER:%.+]] = call i32 @llvm.amdgcn.ds.permute(i32 [[TARGET]], i32 7)
; CHECK-NEXT: br i1 [[ACTIVE]], label %[[DO:.+]], label %[[SKIP:.+]]
; CHECK: [[SKIP]]:
; CHECK-NEXT: phi i32 [ [[SCATTER]], %[[DO]] ], [ undef, {{.+}} ]
; SAME: [[SELECTOR:%.+]] = add i32 {{.+}}, 8
; SAME-NEXT: [[OWN:%.+]] = shl i32 {{.+}}, 2
; SAME-NEXT: [[TARGET:%.+]] = select i1 {{.+}}, i32 [[SELECTOR]], i32 [[OWN]]
; SAME-NEXT: call i32 @llvm.amdgcn.ds.permute(i32 [[TARGET]], i32 7)
	ds_permute_b32 v3, v1, v2 offset:8
	s_wait_dscnt 0
	global_store_b32 v0, v3, s[2:3]
	s_endpgm

	.globl swizzle
	.p2align 8
	.type swizzle,@function
; CHECK-LABEL: define amdgpu_kernel void @swizzle(
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()
; SAME-LABEL: define amdgpu_kernel void @swizzle(
swizzle:
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	v_mov_b32 v2, 7
; The pattern permutes within 32 lanes either way, so it carries over to the
; wider target wave unchanged and no lane index is rebased.
; CHECK: [[DATA:%.+]] = select i1 [[ACTIVE:%.+]], i32 7, i32 0
; CHECK-NEXT: [[PERMUTED:%.+]] = call i32 @llvm.amdgcn.ds.swizzle(i32 [[DATA]], i32 1055)
; CHECK-NEXT: br i1 [[ACTIVE]], label %[[DO:.+]], label %[[SKIP:.+]]
; CHECK: [[SKIP]]:
; CHECK-NEXT: phi i32 [ [[PERMUTED]], %[[DO]] ], [ undef, {{.+}} ]
; SAME: [[DATA:%.+]] = select i1 {{.+}}, i32 7, i32 0
; SAME-NEXT: call i32 @llvm.amdgcn.ds.swizzle(i32 [[DATA]], i32 1055)
	ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,1)
	s_wait_dscnt 0
	global_store_b32 v0, v3, s[2:3]
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel bpermute
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel bpermute_fi
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel permute
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel swizzle
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: bpermute
    .symbol: bpermute.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: bpermute_fi
    .symbol: bpermute_fi.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: permute
    .symbol: permute.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: swizzle
    .symbol: swizzle.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
