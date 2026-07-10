; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_cndmask_b16_kernel,v_cndmask_b16_neg_kernel,v_cndmask_b16_abs_kernel \
; RUN:     2>/dev/null | %FileCheck %s

; CHECK-LABEL: define amdgpu_kernel void @v_cndmask_b16_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cndmask_b16_kernel
	.p2align	8
	.type	v_cndmask_b16_kernel,@function
v_cndmask_b16_kernel:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, s0
	v_mov_b32_e32 v2, s1
	; CHECK: [[CMP:%.+]] = icmp ult i32 0,
	v_cmp_lt_u32_e64 s0, 0, v0
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: %cndmask_b16 = select i1 [[CMP]], i16 {{.+}}, i16
	; CHECK: zext i16 %cndmask_b16 to i32
	; CHECK: and i32 {{.+}}, -65536
	; CHECK: %cndmask_b16_merge_lo = or i32
	v_cndmask_b16 v1.l, v1.l, v2.l, s0
	global_store_b32 v0, v1, s[2:3]
	s_endpgm

; v_cndmask_b16 with neg on src0: the selected half is negated as FP16 before
; the select. The lifted IR bitcasts to half, applies fneg, bitcasts back.
; CHECK-LABEL: define amdgpu_kernel void @v_cndmask_b16_neg_kernel(
	.globl	v_cndmask_b16_neg_kernel
	.p2align	8
	.type	v_cndmask_b16_neg_kernel,@function
v_cndmask_b16_neg_kernel:
	; CHECK: [[CMP_NEG:%.+]] = icmp ult i32 0,
	v_cmp_lt_u32_e64 s0, 0, v0
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: bitcast i16 {{.+}} to half
	; CHECK: %neg_b16_src0 = fneg half
	; CHECK: bitcast half %neg_b16_src0 to i16
	; CHECK: %cndmask_b16 = select i1 [[CMP_NEG]], i16 {{.+}}, i16
	; CHECK: zext i16 %cndmask_b16 to i32
	; CHECK: and i32 {{.+}}, -65536
	; CHECK: %cndmask_b16_merge_lo = or i32
	v_cndmask_b16 v0.l, -v1.l, v2.l, s0
	s_endpgm

; v_cndmask_b16 with abs on src1: the selected half is passed through fabs as
; FP16 before the select. The lifted IR bitcasts to half, calls llvm.fabs,
; bitcasts back.
; CHECK-LABEL: define amdgpu_kernel void @v_cndmask_b16_abs_kernel(
	.globl	v_cndmask_b16_abs_kernel
	.p2align	8
	.type	v_cndmask_b16_abs_kernel,@function
v_cndmask_b16_abs_kernel:
	; CHECK: [[CMP_ABS:%.+]] = icmp ult i32 0,
	v_cmp_lt_u32_e64 s0, 0, v0
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: trunc i32 {{.+}} to i16
	; CHECK: bitcast i16 {{.+}} to half
	; CHECK: %abs_b16_src1 = call half @llvm.fabs.f16(half
	; CHECK: bitcast half %abs_b16_src1 to i16
	; CHECK: %cndmask_b16 = select i1 [[CMP_ABS]], i16 {{.+}}, i16
	; CHECK: zext i16 %cndmask_b16 to i32
	; CHECK: and i32 {{.+}}, -65536
	; CHECK: %cndmask_b16_merge_lo = or i32
	v_cndmask_b16 v0.l, v1.l, |v2.l|, s0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cndmask_b16_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_cndmask_b16_neg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_cndmask_b16_abs_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
    .name:           v_cndmask_b16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_cndmask_b16_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cndmask_b16_neg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         v_cndmask_b16_neg_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cndmask_b16_abs_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         v_cndmask_b16_abs_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
