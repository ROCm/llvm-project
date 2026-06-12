; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=ttmp6_cluster_workgroup_id_init_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; gfx12+ source kernels may read TTMP6 for workgroup-cluster fields. When
; source metadata disables clusters, the source-visible cluster is the
; singleton cluster: workgroup-in-cluster IDs and max IDs are all zero.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ttmp6_cluster_workgroup_id_init_kernel
	.p2align	8
	.type	ttmp6_cluster_workgroup_id_init_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @ttmp6_cluster_workgroup_id_init_kernel(
ttmp6_cluster_workgroup_id_init_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
; CHECK: [[CLUSTER_WG_X:%[a-zA-Z0-9_.]+]] = and i32 0, 15
	s_and_b32 s2, ttmp6, 15
; CHECK: [[CLUSTER_MAX_X_SHIFT:%[a-zA-Z0-9_.]+]] = lshr i32 0, 12
; CHECK: [[CLUSTER_MAX_X:%[a-zA-Z0-9_.]+]] = and i32 [[CLUSTER_MAX_X_SHIFT]], 15
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v1, s2
; CHECK: store i32 [[CLUSTER_WG_X]],
	global_store_b32 v0, v1, s[0:1] scale_offset
	v_mov_b32_e32 v1, s3
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ttmp6_cluster_workgroup_id_init_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
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
    .cluster_dims: [0, 0, 0]
    .max_flat_workgroup_size: 1024
    .name: ttmp6_cluster_workgroup_id_init_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 4
    .symbol: ttmp6_cluster_workgroup_id_init_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
