; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=vop2_integer,vcc_exec_mask \
; RUN:   --target-isa=gfx942 \
; RUN:   | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	vop2_integer
	.p2align	8
	.type	vop2_integer,@function
; IR-LABEL: define amdgpu_kernel void @vop2_integer(
vop2_integer:
; IR: = add i32 1, {{.+}}
	v_add_u32_e32 v2, 1, v1
; IR: = sub i32 2, {{.+}}
	v_sub_u32_e32 v3, 2, v1
; The subrev opcodes compute src1 - src0, so the constant is the right operand.
; IR: = sub i32 {{.+}}, 3
	v_subrev_u32_e32 v4, 3, v1
; IR: = and i32 4, {{.+}}
	v_and_b32_e32 v5, 4, v1
; IR: = or i32 5, {{.+}}
	v_or_b32_e32 v6, 5, v1
; IR: = xor i32 6, {{.+}}
	v_xor_b32_e32 v7, 6, v1
; IR: [[XNOR_XOR:%.+]] = xor i32 7, {{.+}}
; IR-NEXT: = xor i32 [[XNOR_XOR]], -1
	v_xnor_b32_e32 v8, 7, v1

; The shift amount is masked to the width the hardware reads, because an
; unmasked LLVM shift is poison once the amount reaches the operand width.
; IR: [[SHL_AMT:%.+]] = and i32 {{.+}}, 31
; IR-NEXT: = shl i32 {{.+}}, [[SHL_AMT]]
	v_lshlrev_b32_e32 v9, v0, v1
; IR: [[SHR_AMT:%.+]] = and i32 {{.+}}, 31
; IR-NEXT: = lshr i32 {{.+}}, [[SHR_AMT]]
	v_lshrrev_b32_e32 v10, v0, v1
; IR: [[ASHR_AMT:%.+]] = and i32 {{.+}}, 31
; IR-NEXT: = ashr i32 {{.+}}, [[ASHR_AMT]]
	v_ashrrev_i32_e32 v11, v0, v1

; IR: = call i32 @llvm.smin.i32(i32 8, i32 {{.+}})
	v_min_i32_e32 v12, 8, v1
; IR: = call i32 @llvm.smax.i32(i32 9, i32 {{.+}})
	v_max_i32_e32 v13, 9, v1
; IR: = call i32 @llvm.umin.i32(i32 10, i32 {{.+}})
	v_min_u32_e32 v14, 10, v1
; IR: = call i32 @llvm.umax.i32(i32 11, i32 {{.+}})
	v_max_u32_e32 v15, 11, v1

; The 24-bit multiplies read only the low 24 bits of each source, with the
; signedness the opcode names.
; IR: = trunc i32 {{.+}} to i24
; IR-NEXT: = sext i24 {{.+}} to i32
; IR: = mul i32
	v_mul_i32_i24_e32 v16, v0, v1
; IR: = trunc i32 {{.+}} to i24
; IR-NEXT: = zext i24 {{.+}} to i32
; IR: = mul i32
	v_mul_u32_u24_e32 v17, v0, v1
; The high variants return bits [47:32] of the 48-bit product, sign- or
; zero-extended to 32 bits.
; IR: = sext i24 {{.+}} to i64
; IR: [[HI_WIDE:%.+]] = mul i64
; IR-NEXT: [[HI:%.+]] = ashr i64 [[HI_WIDE]], 32
; IR-NEXT: = trunc i64 [[HI]] to i32
	v_mul_hi_i32_i24_e32 v18, v0, v1
; IR: = zext i24 {{.+}} to i64
; IR: [[HIU_WIDE:%.+]] = mul i64
; IR-NEXT: [[HIU:%.+]] = lshr i64 [[HIU_WIDE]], 32
; IR-NEXT: = trunc i64 [[HIU]] to i32
	v_mul_hi_u32_u24_e32 v19, v0, v1

; IR: [[ADD_PAIR:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 1, i32 {{.+}})
; IR: [[ADD_RESULT:%.+]] = extractvalue { i32, i1 } [[ADD_PAIR]], 0
; IR: [[ADD_CARRY:%.+]] = extractvalue { i32, i1 } [[ADD_PAIR]], 1
; IR: [[VCC_AFTER_ADD:%.+]] = and i1 {{.+}}, [[ADD_CARRY]]
	v_add_co_u32_e32 v20, vcc, 1, v1
; IR: [[ADDC_IN:%.+]] = zext i1 [[VCC_AFTER_ADD]] to i32
; IR: [[ADDC_FIRST:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 2, i32 {{.+}})
; IR: [[ADDC_SUM:%.+]] = extractvalue { i32, i1 } [[ADDC_FIRST]], 0
; IR: [[ADDC_SECOND:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[ADDC_SUM]], i32 [[ADDC_IN]])
; IR: [[ADDC_RESULT:%.+]] = extractvalue { i32, i1 } [[ADDC_SECOND]], 0
	v_addc_co_u32_e32 v21, vcc, 2, v1, vcc
; IR: [[SUB_PAIR:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 3, i32 {{.+}})
; IR: [[SUB_RESULT:%.+]] = extractvalue { i32, i1 } [[SUB_PAIR]], 0
; IR: [[SUB_BORROW:%.+]] = extractvalue { i32, i1 } [[SUB_PAIR]], 1
; IR: [[VCC_AFTER_SUB:%.+]] = and i1 {{.+}}, [[SUB_BORROW]]
	v_sub_co_u32_e32 v22, vcc, 3, v1
; IR: [[SUBB_IN:%.+]] = zext i1 [[VCC_AFTER_SUB]] to i32
; IR: [[SUBB_FIRST:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 4, i32 {{.+}})
; IR: [[SUBB_DIFF:%.+]] = extractvalue { i32, i1 } [[SUBB_FIRST]], 0
; IR: [[SUBB_SECOND:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[SUBB_DIFF]], i32 [[SUBB_IN]])
	v_subb_co_u32_e32 v23, vcc, 4, v1, vcc
; IR: [[SUBREV_PAIR:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 {{.+}}, i32 5)
; IR: [[SUBREV_RESULT:%.+]] = extractvalue { i32, i1 } [[SUBREV_PAIR]], 0
; IR: [[SUBREV_BORROW:%.+]] = extractvalue { i32, i1 } [[SUBREV_PAIR]], 1
; IR: [[VCC_AFTER_SUBREV:%.+]] = and i1 {{.+}}, [[SUBREV_BORROW]]
	v_subrev_co_u32_e32 v24, vcc, 5, v1
; IR: [[SUBBREV_IN:%.+]] = zext i1 [[VCC_AFTER_SUBREV]] to i32
; IR: [[SUBBREV_FIRST:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 {{.+}}, i32 6)
; IR: [[SUBBREV_DIFF:%.+]] = extractvalue { i32, i1 } [[SUBBREV_FIRST]], 0
; IR: [[SUBBREV_SECOND:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[SUBBREV_DIFF]], i32 [[SUBBREV_IN]])
; IR: [[SUBBREV_BORROW:%.+]] = extractvalue { i32, i1 } [[SUBBREV_SECOND]], 1
; IR: [[SUBBREV_BORROW_OUT:%.+]] = or i1 {{.+}}, [[SUBBREV_BORROW]]
; IR: [[VCC_AFTER_SUBBREV:%.+]] = and i1 {{.+}}, [[SUBBREV_BORROW_OUT]]
	v_subbrev_co_u32_e32 v25, vcc, 6, v1, vcc
; IR: = select i1 [[VCC_AFTER_SUBBREV]], i32 {{.+}}, i32 7
	v_cndmask_b32_e32 v26, 7, v1, vcc

; IR: [[ADD16_LHS:%.+]] = trunc i32 {{.+}} to i16
; IR: [[ADD16_RHS:%.+]] = trunc i32 {{.+}} to i16
; IR: [[ADD16:%.+]] = add i16 [[ADD16_LHS]], [[ADD16_RHS]]
; IR: = zext i16 [[ADD16]] to i32
	v_add_u16_e32 v27, v0, v1
; IR: [[SUB16_LHS:%.+]] = trunc i32 {{.+}} to i16
; IR: [[SUB16_RHS:%.+]] = trunc i32 {{.+}} to i16
; IR: = sub i16 [[SUB16_LHS]], [[SUB16_RHS]]
	v_sub_u16_e32 v28, v0, v1
; IR: [[SUBREV16_LHS:%.+]] = trunc i32 {{.+}} to i16
; IR: [[SUBREV16_RHS:%.+]] = trunc i32 {{.+}} to i16
; IR: = sub i16 [[SUBREV16_RHS]], [[SUBREV16_LHS]]
	v_subrev_u16_e32 v29, v0, v1
; IR: [[MUL16_LHS:%.+]] = trunc i32 {{.+}} to i16
; IR: [[MUL16_RHS:%.+]] = trunc i32 {{.+}} to i16
; IR: = mul i16 [[MUL16_LHS]], [[MUL16_RHS]]
	v_mul_lo_u16_e32 v30, v0, v1
; IR: [[SHL16_AMOUNT:%.+]] = and i16 {{.+}}, 15
; IR: = shl i16 {{.+}}, [[SHL16_AMOUNT]]
	v_lshlrev_b16_e32 v31, v0, v1
; IR: [[SHR16_AMOUNT:%.+]] = and i16 {{.+}}, 15
; IR: = lshr i16 {{.+}}, [[SHR16_AMOUNT]]
	v_lshrrev_b16_e32 v32, v0, v1
; IR: [[ASHR16_AMOUNT:%.+]] = and i16 {{.+}}, 15
; IR: = ashr i16 {{.+}}, [[ASHR16_AMOUNT]]
	v_ashrrev_i16_e32 v33, v0, v1
; IR: = call i16 @llvm.smin.i16
	v_min_i16_e32 v34, v0, v1
; IR: = call i16 @llvm.smax.i16
	v_max_i16_e32 v35, v0, v1
; IR: = call i16 @llvm.umin.i16
	v_min_u16_e32 v36, v0, v1
; IR: = call i16 @llvm.umax.i16
	v_max_u16_e32 v37, v0, v1

; IR: [[DOT2_A:%.+]] = sext i16 {{.+}} to i32
; IR: [[DOT2_B:%.+]] = sext i16 {{.+}} to i32
; IR: [[DOT2_PRODUCT:%.+]] = mul i32 [[DOT2_A]], [[DOT2_B]]
; IR: = add i32 {{.+}}, [[DOT2_PRODUCT]]
; IR: [[DOT2_HIGH_SHIFT:%.+]] = lshr i32 {{.+}}, 16
; IR: [[DOT2_HIGH_BITS:%.+]] = trunc i32 [[DOT2_HIGH_SHIFT]] to i16
; IR: [[DOT2_HIGH_A:%.+]] = sext i16 [[DOT2_HIGH_BITS]] to i32
; IR: [[DOT2_HIGH_PRODUCT:%.+]] = mul i32 [[DOT2_HIGH_A]], {{.+}}
; IR: [[DOT2_RESULT:%.+]] = add i32 {{.+}}, [[DOT2_HIGH_PRODUCT]]
; IR: [[DOT2_DST:%.+]] = phi i32 [ [[DOT2_RESULT]], %{{.+}} ], [ {{.+}}, %{{.+}} ]
	v_dot2c_i32_i16_e32 v38, v0, v1
; IR: [[DOT4_A:%.+]] = sext i8 {{.+}} to i32
; IR: [[DOT4_B:%.+]] = sext i8 {{.+}} to i32
; IR: [[DOT4_PRODUCT:%.+]] = mul i32 [[DOT4_A]], [[DOT4_B]]
; IR: = add i32 [[DOT2_DST]], [[DOT4_PRODUCT]]
; IR: [[DOT4_HIGH_SHIFT:%.+]] = lshr i32 {{.+}}, 24
; IR: [[DOT4_HIGH_BITS:%.+]] = trunc i32 [[DOT4_HIGH_SHIFT]] to i8
; IR: [[DOT4_HIGH_A:%.+]] = sext i8 [[DOT4_HIGH_BITS]] to i32
; IR: [[DOT4_HIGH_PRODUCT:%.+]] = mul i32 [[DOT4_HIGH_A]], {{.+}}
; IR: [[DOT4_RESULT:%.+]] = add i32 {{.+}}, [[DOT4_HIGH_PRODUCT]]
; IR: [[DOT4_DST:%.+]] = phi i32 [ [[DOT4_RESULT]], %{{.+}} ], [ [[DOT2_DST]], %{{.+}} ]
	v_dot4c_i32_i8_e32 v38, v0, v1
; IR: [[DOT8_A:%.+]] = sext i4 {{.+}} to i32
; IR: [[DOT8_B:%.+]] = sext i4 {{.+}} to i32
; IR: [[DOT8_PRODUCT:%.+]] = mul i32 [[DOT8_A]], [[DOT8_B]]
; IR: = add i32 [[DOT4_DST]], [[DOT8_PRODUCT]]
; IR: [[DOT8_HIGH_SHIFT:%.+]] = lshr i32 {{.+}}, 28
; IR: [[DOT8_HIGH_BITS:%.+]] = trunc i32 [[DOT8_HIGH_SHIFT]] to i4
; IR: [[DOT8_HIGH_A:%.+]] = sext i4 [[DOT8_HIGH_BITS]] to i32
; IR: [[DOT8_HIGH_PRODUCT:%.+]] = mul i32 [[DOT8_HIGH_A]], {{.+}}
; IR: = add i32 {{.+}}, [[DOT8_HIGH_PRODUCT]]
	v_dot8c_i32_i4_e32 v38, v0, v1
; IR: ret void
	s_endpgm

	.globl	vcc_exec_mask
	.p2align	8
	.type	vcc_exec_mask,@function
; IR-LABEL: define amdgpu_kernel void @vcc_exec_mask(
vcc_exec_mask:
	s_mov_b32 vcc_lo, -1
	s_mov_b32 vcc_hi, -1
	s_mov_b32 exec_lo, 1
	s_mov_b32 exec_hi, 0
; IR: [[MASKED_ADD_PAIR:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 0, i32 {{.+}})
; IR: [[MASKED_ADD_CARRY:%.+]] = extractvalue { i32, i1 } [[MASKED_ADD_PAIR]], 1
; IR: [[MASKED_VCC:%.+]] = and i1 {{.+}}, [[MASKED_ADD_CARRY]]
	v_add_co_u32_e32 v0, vcc, 0, v2
; IR: = select i1 [[MASKED_VCC]], i32 {{.+}}, i32 11
	v_cndmask_b32_e32 v1, 11, v2, vcc
; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop2_integer
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 1
		.amdhsa_reserve_vcc 1
		.amdhsa_accum_offset 40
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel vcc_exec_mask
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
		.amdhsa_reserve_vcc 1
		.amdhsa_accum_offset 4
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
    .name:           vop2_integer
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vop2_integer.kd
    .vgpr_count:     40
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vcc_exec_mask
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vcc_exec_mask.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
