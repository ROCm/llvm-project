; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmp_cndmask_sgpr_class_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; fused v_cmp_class+v_cndmask SGPR-condition rewrite.
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_cndmask_sgpr_class_kernel(
; CHECK: [[CMP:%vclass[0-9]*]] = call i1 @llvm.amdgcn.class.f32(float %{{[^,]+}}, i32 512)
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32
; CHECK: %cndmask = select i1 [[CMP]], i32 1065353216, i32 -1082130432
; CHECK-NOT: %mask_lane_idx{{[0-9]*}} = zext i32 %{{[^ ]+}} to i64
; CHECK-NOT: %mask_at_lane{{[0-9]*}} = lshr i64 %{{[^,]+}},
; CHECK-NOT: %mask_lane_i1{{[0-9]*}} = icmp ne i64
; CHECK-NOT: select i32 %{{.*}}, i32 {{.*}}, i32 {{.*}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cmp_cndmask_sgpr_class_kernel
	.p2align	8
	.type	v_cmp_cndmask_sgpr_class_kernel,@function
v_cmp_cndmask_sgpr_class_kernel:        ; @v_cmp_cndmask_sgpr_class_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_class_f32_e64 s4, v1, 0x200
	v_cndmask_b32_e64 v1, -1.0, 1.0, s4
	
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cmp_cndmask_sgpr_class_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 5
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
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_cmp_cndmask_sgpr_class_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         v_cmp_cndmask_sgpr_class_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
