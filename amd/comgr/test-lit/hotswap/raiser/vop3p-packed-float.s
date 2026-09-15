; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=packed_float \
; RUN:   | %FileCheck %s --check-prefix=DECODE
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=packed_float | %FileCheck %s
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_fp16_overflow,refuse_fp16_rounding 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	packed_float
	.p2align	8
	.type	packed_float,@function
; CHECK-LABEL: define amdgpu_kernel void @packed_float(
packed_float:
; DECODE: V_PK_ADD_F16{{.+}}v_pk_add_f16
; CHECK: [[F16_SRC0_LO:%.+]] = extractelement <2 x half> [[F16_SRC0:%.+]], i64 0
; CHECK: [[F16_SRC0_HI:%.+]] = extractelement <2 x half> [[F16_SRC0]], i64 1
; CHECK: [[F16_NEG_LO:%.+]] = fneg half [[F16_SRC0_HI]]
; CHECK: [[F16_SRC0_LOW_LANE:%.+]] = insertelement <2 x half> poison, half [[F16_NEG_LO]], i64 0
; CHECK: [[F16_OPERAND0:%.+]] = insertelement <2 x half> [[F16_SRC0_LOW_LANE]], half [[F16_SRC0_LO]], i64 1
; CHECK: [[F16_SRC1_HI:%.+]] = extractelement <2 x half> [[F16_SRC1:%.+]], i64 1
; CHECK: [[F16_NEG_HI:%.+]] = fneg half [[F16_SRC1_HI]]
; CHECK: [[F16_OPERAND1:%.+]] = insertelement <2 x half> [[F16_SRC1_LOW_LANE:%.+]], half [[F16_NEG_HI]], i64 1
; CHECK: fadd <2 x half> [[F16_OPERAND0]], [[F16_OPERAND1]]
	v_pk_add_f16 v0, v1, v2 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1]
; DECODE: V_PK_MUL_F16{{.+}}v_pk_mul_f16
; CHECK: [[F16_MUL:%.+]] = fmul <2 x half>
; CHECK: call <2 x half> @llvm.maxnum.v2f16(<2 x half> [[F16_MUL]]
; CHECK: call <2 x half> @llvm.minnum.v2f16
	v_pk_mul_f16 v3, v4, v5 clamp
; DECODE: V_PK_ADD_F32{{.+}}v_pk_add_f32
; CHECK: [[F32_SRC0_LO:%.+]] = extractelement <2 x float> [[F32_SRC0:%.+]], i64 0
; CHECK: [[F32_SRC0_HI:%.+]] = extractelement <2 x float> [[F32_SRC0]], i64 1
; CHECK: [[F32_NEG_LO:%.+]] = fneg float [[F32_SRC0_HI]]
; CHECK: [[F32_SRC0_LOW_LANE:%.+]] = insertelement <2 x float> poison, float [[F32_NEG_LO]], i64 0
; CHECK: [[F32_OPERAND0:%.+]] = insertelement <2 x float> [[F32_SRC0_LOW_LANE]], float [[F32_SRC0_LO]], i64 1
; CHECK: [[F32_SRC1_HI:%.+]] = extractelement <2 x float> [[F32_SRC1:%.+]], i64 1
; CHECK: [[F32_NEG_HI:%.+]] = fneg float [[F32_SRC1_HI]]
; CHECK: [[F32_OPERAND1:%.+]] = insertelement <2 x float> [[F32_SRC1_LOW_LANE:%.+]], float [[F32_NEG_HI]], i64 1
; CHECK: fadd <2 x float> [[F32_OPERAND0]], [[F32_OPERAND1]]
	v_pk_add_f32 v[6:7], v[8:9], v[10:11] op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1]
; DECODE: V_PK_MUL_F32{{.+}}v_pk_mul_f32
; CHECK: [[F32_MUL:%.+]] = fmul <2 x float>
; CHECK: call <2 x float> @llvm.maxnum.v2f32(<2 x float> [[F32_MUL]]
; CHECK: call <2 x float> @llvm.minnum.v2f32
	v_pk_mul_f32 v[12:13], v[14:15], 1.0 clamp
; CHECK: ret void
	s_endpgm

	.globl	refuse_fp16_overflow
	.p2align	8
	.type	refuse_fp16_overflow,@function
; REFUSE: unsupported-floating-point-mode: v_pk_add_f16 [VOP3P]
; REFUSE-SAME: FP16 overflow saturation is unsupported
refuse_fp16_overflow:
	v_pk_add_f16 v0, v1, v2
	s_endpgm

	.globl	refuse_fp16_rounding
	.p2align	8
	.type	refuse_fp16_rounding,@function
; REFUSE: unsupported-floating-point-mode: v_pk_mul_f16 [VOP3P]
; REFUSE-SAME: f16 rounding mode 1 is unsupported
refuse_fp16_rounding:
	v_pk_mul_f16 v0, v1, v2
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel packed_float
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 16
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_fp16_overflow
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_fp16_overflow 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_fp16_rounding
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_float_round_mode_16_64 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           packed_float
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         packed_float.kd
    .vgpr_count:     16
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_fp16_overflow
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_fp16_overflow.kd
    .vgpr_count:     3
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_fp16_rounding
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_fp16_rounding.kd
    .vgpr_count:     3
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
