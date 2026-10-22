; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=int16_gfx1250 | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_clamp 2>&1 | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	int16_gfx1250
	.p2align	8
	.type	int16_gfx1250,@function
; IR-LABEL: define amdgpu_kernel void @int16_gfx1250(
int16_gfx1250:
	; A low-half destination keeps the high half of the register it writes.
	; IR: [[SHL_AMOUNT:%.+]] = and i16 {{.+}}, 15
	; IR-NEXT: [[SHL:%.+]] = shl i16 {{.+}}, [[SHL_AMOUNT]]
	; IR-NEXT: [[SHL_BITS:%.+]] = zext i16 [[SHL]] to i32
	; IR-NEXT: [[SHL_KEEP:%.+]] = and i32 {{.+}}, -65536
	; IR-NEXT: {{.+}} = or i32 [[SHL_KEEP]], [[SHL_BITS]]
	v_lshlrev_b16 v0.l, v1.l, v2.l
	; A high-half source is read from the top of its register, and a high-half
	; destination keeps the low half.
	; IR: [[SHR_SRC:%.+]] = lshr i32 {{.+}}, 16
	; IR-NEXT: [[SHR_VALUE:%.+]] = trunc i32 [[SHR_SRC]] to i16
	; IR-NEXT: [[SHR_AMOUNT:%.+]] = and i16 {{.+}}, 15
	; IR-NEXT: [[SHR:%.+]] = lshr i16 [[SHR_VALUE]], [[SHR_AMOUNT]]
	; IR-NEXT: [[SHR_BITS:%.+]] = zext i16 [[SHR]] to i32
	; IR-NEXT: [[SHR_KEEP:%.+]] = and i32 {{.+}}, 65535
	; IR-NEXT: [[SHR_SHIFTED:%.+]] = shl i32 [[SHR_BITS]], 16
	; IR-NEXT: {{.+}} = or i32 [[SHR_KEEP]], [[SHR_SHIFTED]]
	v_lshrrev_b16 v0.h, v1.l, v2.h
	; IR: [[ASHR_AMOUNT:%.+]] = and i16 {{.+}}, 15
	; IR-NEXT: {{.+}} = ashr i16 {{.+}}, [[ASHR_AMOUNT]]
	v_ashrrev_i16 v3.l, v4.l, v5.l
	; IR: {{.+}} = and i16 {{.+}}, {{.+}}
	v_and_b16 v6.l, v7.h, v8.l
	; IR: {{.+}} = or i16 {{.+}}, {{.+}}
	v_or_b16 v6.l, v7.l, v8.l
	; IR: {{.+}} = xor i16 {{.+}}, {{.+}}
	v_xor_b16 v6.l, v7.l, v8.l
	; IR: {{.+}} = xor i16 {{.+}}, -1
	v_not_b16 v9, v10
	; The truth table selects the minterms that make up the result, 0x40 being
	; the bitwise and of the first two sources.
	; IR: {{.+}} = xor i16 {{.+}}, -1
	; IR-NEXT: {{.+}} = xor i16 {{.+}}, -1
	; IR-NEXT: [[BITOP_NOT2:%.+]] = xor i16 {{.+}}, -1
	; IR-NEXT: [[BITOP_AND:%.+]] = and i16 {{.+}}, {{.+}}
	; IR-NEXT: [[BITOP_MINTERM:%.+]] = and i16 [[BITOP_AND]], [[BITOP_NOT2]]
	; IR-NEXT: {{.+}} = zext i16 [[BITOP_MINTERM]] to i32
	v_bitop3_b16 v11.l, v12.l, v13.l, v14.l bitop3:0x40
	; IR: {{.+}} = mul i16 {{.+}}, {{.+}}
	v_mul_lo_u16 v15, v16, v17
	; The sum is formed at 32 bits, which is what a set clamp bit saturates.
	; IR: [[MAD_MUL:%.+]] = mul i32 {{.+}}, {{.+}}
	; IR-NEXT: [[MAD_C:%.+]] = zext i16 {{.+}} to i32
	; IR-NEXT: [[MAD_SUM:%.+]] = add i32 [[MAD_MUL]], [[MAD_C]]
	; IR-NEXT: [[MAD_SAT:%.+]] = call i32 @llvm.umin.i32(i32 [[MAD_SUM]], i32 65535)
	; IR-NEXT: {{.+}} = trunc i32 [[MAD_SAT]] to i16
	v_mad_u16 v18, v19, v20, v21 clamp
	; IR: [[MIN3_INNER:%.+]] = call i16 @llvm.smin.i16(
	; IR-NEXT: {{.+}} = call i16 @llvm.smin.i16(i16 [[MIN3_INNER]],
	v_min3_i16 v22, v23, v24, v25
	; IR: [[MAX3_INNER:%.+]] = call i16 @llvm.umax.i16(
	; IR-NEXT: {{.+}} = call i16 @llvm.umax.i16(i16 [[MAX3_INNER]],
	v_max3_u16 v22, v23, v24, v25
	; The median clamps the third source into the range the other two span.
	; IR: [[MED3_LOW:%.+]] = call i16 @llvm.smin.i16(i16 [[MED3_A:.+]], i16 [[MED3_B:.+]])
	; IR-NEXT: [[MED3_HIGH:%.+]] = call i16 @llvm.smax.i16(i16 [[MED3_A]], i16 [[MED3_B]])
	; IR-NEXT: [[MED3_UPPER:%.+]] = call i16 @llvm.smin.i16(i16 [[MED3_HIGH]],
	; IR-NEXT: {{.+}} = call i16 @llvm.smax.i16(i16 [[MED3_LOW]], i16 [[MED3_UPPER]])
	v_med3_i16 v26, v27, v28, v29
	; IR: {{.+}} = call i16 @llvm.uadd.sat.i16(
	v_add_nc_u16 v30, v31, v32 clamp
	; IR: {{.+}} = call i16 @llvm.ssub.sat.i16(
	v_sub_nc_i16 v30, v31, v32 clamp
	; IR: {{.+}} = add i16 {{.+}}, {{.+}}
	v_add_nc_u16 v30, v31, v32
	; A moved half lands in the destination half the instruction names.
	; IR: [[MOV_KEEP:%.+]] = and i32 {{.+}}, 65535
	; IR-NEXT: [[MOV_SHIFTED:%.+]] = shl i32 {{.+}}, 16
	; IR-NEXT: {{.+}} = or i32 [[MOV_KEEP]], [[MOV_SHIFTED]]
	v_mov_b16 v33.h, v34.l
	v_cmp_lt_u32 vcc_lo, v35, v36
	; IR: {{.+}} = select i1 {{.+}}, i16 {{.+}}, i16 {{.+}}
	v_cndmask_b16 v37.l, v38.l, v39.l, vcc_lo
	; The select moves bits, so the f16 sign modifiers it accepts reduce to
	; clearing and flipping the sign bit of the half each source reads.
	; IR: [[NEG:%.+]] = xor i16 {{.+}}, -32768
	; IR: [[ABS:%.+]] = and i16 {{.+}}, 32767
	; IR-NEXT: {{.+}} = select i1 {{.+}}, i16 [[ABS]], i16 [[NEG]]
	v_cndmask_b16 v37.l, -v38.l, |v39.h|, vcc_lo
	; IR: [[NEGABS:%.+]] = and i16 {{.+}}, 32767
	; IR-NEXT: [[NEGABS_NEG:%.+]] = xor i16 [[NEGABS]], -32768
	; IR-NEXT: {{.+}} = select i1 {{.+}}, i16 [[NEGABS_NEG]], i16 {{.+}}
	v_cndmask_b16 v37.h, v38.l, -|v39.l|, vcc_lo
	; IR: ret void
	s_endpgm

	.globl	refuse_clamp
	.p2align	8
	.type	refuse_clamp,@function
; REFUSE: unsupported-instruction-form: v_med3_i16 [VOP3]
; REFUSE-SAME: 16-bit integer operation does not define clamp
refuse_clamp:
	v_med3_i16 v0, v1, v2, v3 clamp
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel int16_gfx1250
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_clamp
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
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
    .name:           int16_gfx1250
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         int16_gfx1250.kd
    .vgpr_count:     40
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_clamp
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         refuse_clamp.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
