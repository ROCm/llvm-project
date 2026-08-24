; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=sop2_integer_gfx1250 | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sop2_integer_gfx1250
	.p2align	8
	.type	sop2_integer_gfx1250,@function
; IR-LABEL: define amdgpu_kernel void @sop2_integer_gfx1250(
sop2_integer_gfx1250:
	; IR: [[MUL_LO0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[MUL_HI0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[MUL_SHL0:%.*]] = shl i64 [[MUL_HI0]], 32
	; IR-NEXT: [[MUL_SRC0:%.*]] = or i64 [[MUL_LO0]], [[MUL_SHL0]]
	; IR-NEXT: [[MUL_LO1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[MUL_HI1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[MUL_SHL1:%.*]] = shl i64 [[MUL_HI1]], 32
	; IR-NEXT: [[MUL_SRC1:%.*]] = or i64 [[MUL_LO1]], [[MUL_SHL1]]
	; IR-NEXT: [[MUL:%.*]] = mul i64 [[MUL_SRC0]], [[MUL_SRC1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[MUL]] to i32
	; IR-NEXT: [[MUL_SHIFT:%.*]] = lshr i64 [[MUL]], 32
	; IR-NEXT: {{%.*}} = trunc i64 [[MUL_SHIFT]] to i32
	s_mul_u64 s[2:3], s[0:1], s[4:5]
	; IR: [[ADD_LO0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[ADD_HI0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[ADD_SHL0:%.*]] = shl i64 [[ADD_HI0]], 32
	; IR-NEXT: [[ADD_SRC0:%.*]] = or i64 [[ADD_LO0]], [[ADD_SHL0]]
	; IR-NEXT: [[ADD_LO1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[ADD_HI1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[ADD_SHL1:%.*]] = shl i64 [[ADD_HI1]], 32
	; IR-NEXT: [[ADD_SRC1:%.*]] = or i64 [[ADD_LO1]], [[ADD_SHL1]]
	; IR-NEXT: [[ADD:%.*]] = add i64 [[ADD_SRC0]], [[ADD_SRC1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[ADD]] to i32
	; IR-NEXT: [[ADD_SHIFT:%.*]] = lshr i64 [[ADD]], 32
	; IR-NEXT: {{%.*}} = trunc i64 [[ADD_SHIFT]] to i32
	s_add_nc_u64 s[2:3], s[0:1], s[4:5]
	; IR: [[SUB_LO0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[SUB_HI0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[SUB_SHL0:%.*]] = shl i64 [[SUB_HI0]], 32
	; IR-NEXT: [[SUB_SRC0:%.*]] = or i64 [[SUB_LO0]], [[SUB_SHL0]]
	; IR-NEXT: [[SUB_LO1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[SUB_HI1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[SUB_SHL1:%.*]] = shl i64 [[SUB_HI1]], 32
	; IR-NEXT: [[SUB_SRC1:%.*]] = or i64 [[SUB_LO1]], [[SUB_SHL1]]
	; IR-NEXT: [[SUB:%.*]] = sub i64 [[SUB_SRC0]], [[SUB_SRC1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[SUB]] to i32
	; IR-NEXT: [[SUB_SHIFT:%.*]] = lshr i64 [[SUB]], 32
	; IR-NEXT: {{%.*}} = trunc i64 [[SUB_SHIFT]] to i32
	s_sub_nc_u64 s[2:3], s[0:1], s[4:5]
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sop2_integer_gfx1250
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 6
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
    .name:           sop2_integer_gfx1250
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         sop2_integer_gfx1250.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
