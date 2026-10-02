; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=int16_gfx9 | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	int16_gfx9
	.p2align	8
	.type	int16_gfx9,@function
; IR-LABEL: define amdgpu_kernel void @int16_gfx9(
int16_gfx9:
	; The encodings with no op_sel field cannot name a destination half, and
	; zero the half they do not write rather than keeping it.
	; IR: [[ADD:%.+]] = add i16 {{.+}}, {{.+}}
	; IR-NEXT: {{.+}} = zext i16 [[ADD]] to i32
	v_add_u16 v0, v1, v2
	; IR: {{.+}} = call i16 @llvm.uadd.sat.i16(
	v_add_u16 v3, v4, v5 clamp
	; IR: [[SHL_AMOUNT:%.+]] = and i16 {{.+}}, 15
	; IR-NEXT: {{.+}} = shl i16 {{.+}}, [[SHL_AMOUNT]]
	v_lshlrev_b16 v6, v7, v8
	; IR: {{.+}} = call i16 @llvm.umax.i16(
	v_max_u16 v9, v10, v11
	; An op_sel field redirects the result to the high half, and the low half
	; the instruction leaves unwritten keeps its value.
	; IR: [[MAD_MUL:%.+]] = mul i32 {{.+}}, {{.+}}
	; IR-NEXT: [[MAD_C:%.+]] = zext i16 {{.+}} to i32
	; IR-NEXT: [[MAD_SUM:%.+]] = add i32 [[MAD_MUL]], [[MAD_C]]
	; IR-NEXT: [[MAD:%.+]] = trunc i32 [[MAD_SUM]] to i16
	; IR-NEXT: [[MAD_BITS:%.+]] = zext i16 [[MAD]] to i32
	; IR-NEXT: [[MAD_KEEP:%.+]] = and i32 {{.+}}, 65535
	; IR-NEXT: [[MAD_SHIFTED:%.+]] = shl i32 [[MAD_BITS]], 16
	; IR-NEXT: {{.+}} = or i32 [[MAD_KEEP]], [[MAD_SHIFTED]]
	v_mad_u16 v12, v13, v14, v15 op_sel:[0,0,0,1]
	; IR: [[MED3_LOW:%.+]] = call i16 @llvm.umin.i16(i16 [[MED3_A:.+]], i16 [[MED3_B:.+]])
	; IR-NEXT: [[MED3_HIGH:%.+]] = call i16 @llvm.umax.i16(i16 [[MED3_A]], i16 [[MED3_B]])
	; IR-NEXT: [[MED3_UPPER:%.+]] = call i16 @llvm.umin.i16(i16 [[MED3_HIGH]],
	; IR-NEXT: {{.+}} = call i16 @llvm.umax.i16(i16 [[MED3_LOW]], i16 [[MED3_UPPER]])
	v_med3_u16 v16, v17, v18, v19
	; Truth table 0x96 is the three-way exclusive or, so every minterm with an
	; odd number of set inputs contributes.
	; IR: [[XOR3_OR1:%.+]] = or i16 {{.+}}, {{.+}}
	; IR: [[XOR3_OR2:%.+]] = or i16 [[XOR3_OR1]], {{.+}}
	; IR: {{.+}} = or i16 [[XOR3_OR2]], {{.+}}
	v_bitop3_b16 v20, v21, v22, v23 bitop3:0x96
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel int16_gfx9
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 24
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           int16_gfx9
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         int16_gfx9.kd
    .vgpr_count:     24
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
