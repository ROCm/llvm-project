; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vop2_integer_gfx1250 | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vop2_integer_gfx1250
	.p2align	8
	.type	vop2_integer_gfx1250,@function
; IR-LABEL: define amdgpu_kernel void @vop2_integer_gfx1250(
vop2_integer_gfx1250:
; IR: = add i32 1, {{.+}}
	v_add_nc_u32_e32 v2, 1, v1
; IR: = sub i32 2, {{.+}}
	v_sub_nc_u32_e32 v3, 2, v1
; The subrev opcodes compute src1 - src0, so the constant is the right operand.
; IR: = sub i32 {{.+}}, 3
	v_subrev_nc_u32_e32 v4, 3, v1

; IR: = call i32 @llvm.smin.i32(i32 8, i32 {{.+}})
	v_min_i32_e32 v26, 8, v1
; IR: = call i32 @llvm.smax.i32(i32 9, i32 {{.+}})
	v_max_i32_e32 v27, 9, v1
; IR: = call i32 @llvm.umin.i32(i32 10, i32 {{.+}})
	v_min_u32_e32 v28, 10, v1
; IR: = call i32 @llvm.umax.i32(i32 11, i32 {{.+}})
	v_max_u32_e32 v29, 11, v1

; IR: = and i32 12, {{.+}}
	v_and_b32_e32 v30, 12, v1
; IR: = or i32 13, {{.+}}
	v_or_b32_e32 v31, 13, v1
; IR: = xor i32 14, {{.+}}
	v_xor_b32_e32 v32, 14, v1
; IR: [[XNOR_XOR:%.+]] = xor i32 15, {{.+}}
; IR-NEXT: = xor i32 [[XNOR_XOR]], -1
	v_xnor_b32_e32 v33, 15, v1

; IR: = shl i32 {{.+}}, 1
	v_lshlrev_b32_e32 v34, 33, v1
; IR: = lshr i32 {{.+}}, 2
	v_lshrrev_b32_e32 v35, 34, v1
; IR: = ashr i32 {{.+}}, 3
	v_ashrrev_i32_e32 v36, 35, v1

; IR: = add i64
	v_add_nc_u64_e32 v[10:11], v[6:7], v[8:9]
; IR: = sub i64
	v_sub_nc_u64_e32 v[12:13], v[6:7], v[8:9]
; IR: = mul i64
	v_mul_u64_e32 v[14:15], v[6:7], v[8:9]

; IR: [[SMUL_LHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[SMUL_LHS:%.+]] = sext i24 [[SMUL_LHS_NARROW]] to i32
; IR: [[SMUL_RHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[SMUL_RHS:%.+]] = sext i24 [[SMUL_RHS_NARROW]] to i32
; IR-NEXT: = mul i32 [[SMUL_LHS]], [[SMUL_RHS]]
	v_mul_i32_i24_e32 v18, v0, v1
; IR: [[UMUL_LHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[UMUL_LHS:%.+]] = zext i24 [[UMUL_LHS_NARROW]] to i32
; IR: [[UMUL_RHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[UMUL_RHS:%.+]] = zext i24 [[UMUL_RHS_NARROW]] to i32
; IR-NEXT: = mul i32 [[UMUL_LHS]], [[UMUL_RHS]]
	v_mul_u32_u24_e32 v19, v0, v1
; IR: [[SMUL_HI_LHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[SMUL_HI_LHS:%.+]] = sext i24 [[SMUL_HI_LHS_NARROW]] to i64
; IR: [[SMUL_HI_RHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[SMUL_HI_RHS:%.+]] = sext i24 [[SMUL_HI_RHS_NARROW]] to i64
; IR-NEXT: [[SMUL_HI_WIDE:%.+]] = mul i64 [[SMUL_HI_LHS]], [[SMUL_HI_RHS]]
; IR-NEXT: [[SMUL_HI:%.+]] = ashr i64 [[SMUL_HI_WIDE]], 32
; IR-NEXT: = trunc i64 [[SMUL_HI]] to i32
	v_mul_hi_i32_i24_e32 v20, v0, v1
; IR: [[UMUL_HI_LHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[UMUL_HI_LHS:%.+]] = zext i24 [[UMUL_HI_LHS_NARROW]] to i64
; IR: [[UMUL_HI_RHS_NARROW:%.+]] = trunc i32 {{.+}} to i24
; IR-NEXT: [[UMUL_HI_RHS:%.+]] = zext i24 [[UMUL_HI_RHS_NARROW]] to i64
; IR-NEXT: [[UMUL_HI_WIDE:%.+]] = mul i64 [[UMUL_HI_LHS]], [[UMUL_HI_RHS]]
; IR-NEXT: [[UMUL_HI:%.+]] = lshr i64 [[UMUL_HI_WIDE]], 32
; IR-NEXT: = trunc i64 [[UMUL_HI]] to i32
	v_mul_hi_u32_u24_e32 v21, v0, v1

; v_lshlrev_b64 is the one 64-bit VOP2 opcode with a 32-bit source: src0 is the
; shift amount, masked to the six bits the hardware reads.
; IR: [[AMOUNT:%.+]] = and i32 {{.+}}, 63
; IR-NEXT: [[AMOUNT64:%.+]] = zext i32 [[AMOUNT]] to i64
; IR-NEXT: = shl i64 {{.+}}, [[AMOUNT64]]
	v_lshlrev_b64_e32 v[16:17], v5, v[8:9]

; IR: [[ADDC_IN:%.+]] = zext i1 {{.+}} to i32
; IR: [[ADDC_FIRST:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32
; IR: [[ADDC_SUM:%.+]] = extractvalue { i32, i1 } [[ADDC_FIRST]], 0
; IR: [[ADDC_SECOND:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[ADDC_SUM]], i32 [[ADDC_IN]])
	v_add_co_ci_u32_e32 v22, vcc_lo, v0, v1, vcc_lo
; IR: [[SUBB_IN:%.+]] = zext i1 {{.+}} to i32
; IR: [[SUBB_FIRST:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32
; IR: [[SUBB_DIFF:%.+]] = extractvalue { i32, i1 } [[SUBB_FIRST]], 0
; IR: [[SUBB_SECOND:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[SUBB_DIFF]], i32 [[SUBB_IN]])
	v_sub_co_ci_u32_e32 v23, vcc_lo, v0, v1, vcc_lo
; IR: [[SUBREV_BORROW_IN:%.+]] = zext i1 {{.+}} to i32
; IR: [[SUBREV_FIRST:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32
; IR: [[SUBREV_DIFF:%.+]] = extractvalue { i32, i1 } [[SUBREV_FIRST]], 0
; IR: [[SUBREV_SECOND:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[SUBREV_DIFF]], i32 [[SUBREV_BORROW_IN]])
; IR: [[SUBREV_SECOND_BORROW:%.+]] = extractvalue { i32, i1 } [[SUBREV_SECOND]], 1
; IR: [[SUBREV_BORROW_OUT:%.+]] = or i1 {{.+}}, [[SUBREV_SECOND_BORROW]]
; IR: [[VCC_AFTER_SUBREV:%.+]] = and i1 {{.+}}, [[SUBREV_BORROW_OUT]]
	v_subrev_co_ci_u32_e32 v24, vcc_lo, v0, v1, vcc_lo
; IR: = select i1 [[VCC_AFTER_SUBREV]], i32 {{.+}}, i32 {{.+}}
	v_cndmask_b32_e32 v25, v0, v1, vcc_lo
; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop2_integer_gfx1250
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 37
		.amdhsa_next_free_sgpr 1
		.amdhsa_reserve_vcc 1
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
    .name:           vop2_integer_gfx1250
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vop2_integer_gfx1250.kd
    .vgpr_count:     37
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
