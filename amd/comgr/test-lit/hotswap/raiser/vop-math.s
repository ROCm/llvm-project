; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.5-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=vop_math,vop3_math \
; RUN:   | %FileCheck %s --check-prefix=DECODE
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=vop_math,vop3_math | %FileCheck %s
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=refuse_output_modifier 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	vop_math
	.p2align	8
	.type	vop_math,@function
; CHECK-LABEL: define amdgpu_kernel void @vop_math(
vop_math:
; DECODE: V_CVT_F32_I32{{.+}}v_cvt_f32_i32_e32
; CHECK: sitofp i32
	v_cvt_f32_i32_e32 v0, v1
; DECODE: V_CVT_F32_U32{{.+}}v_cvt_f32_u32_e32
; CHECK: uitofp i32
	v_cvt_f32_u32_e32 v2, v3
; DECODE: V_CVT_I32_F32{{.+}}v_cvt_i32_f32_e32
; CHECK: call i32 @llvm.fptosi.sat.i32.f32
	v_cvt_i32_f32_e32 v4, v5
; DECODE: V_CVT_U32_F32{{.+}}v_cvt_u32_f32_e32
; CHECK: call i32 @llvm.fptoui.sat.i32.f32
	v_cvt_u32_f32_e32 v6, v7
; DECODE: V_CVT_F16_F32{{.+}}v_cvt_f16_f32_e32
; CHECK: fptrunc float
	v_cvt_f16_f32_e32 v8, v9
; DECODE: V_CVT_F32_F16{{.+}}v_cvt_f32_f16_e32
; CHECK: fpext half
	v_cvt_f32_f16_e32 v10, v11
; DECODE: V_CVT_F32_UBYTE0{{.+}}v_cvt_f32_ubyte0_e32
; CHECK: uitofp i32
	v_cvt_f32_ubyte0_e32 v40, v41
; DECODE: V_CVT_F32_UBYTE1{{.+}}v_cvt_f32_ubyte1_e32
; CHECK: lshr i32 {{.+}}, 8
; CHECK: uitofp i32
	v_cvt_f32_ubyte1_e32 v42, v43
; DECODE: V_CVT_F32_UBYTE2{{.+}}v_cvt_f32_ubyte2_e32
; CHECK: lshr i32 {{.+}}, 16
; CHECK: uitofp i32
	v_cvt_f32_ubyte2_e32 v44, v45
; DECODE: V_CVT_F32_UBYTE3{{.+}}v_cvt_f32_ubyte3_e32
; CHECK: lshr i32 {{.+}}, 24
; CHECK: uitofp i32
	v_cvt_f32_ubyte3_e32 v46, v47
; DECODE: V_FRACT_F32{{.+}}v_fract_f32_e32
; CHECK: call float @llvm.amdgcn.fract.f32
	v_fract_f32_e32 v12, v13
; DECODE: V_TRUNC_F32{{.+}}v_trunc_f32_e32
; CHECK: call float @llvm.trunc.f32
	v_trunc_f32_e32 v14, v15
; DECODE: V_CEIL_F32{{.+}}v_ceil_f32_e32
; CHECK: call float @llvm.ceil.f32
	v_ceil_f32_e32 v16, v17
; DECODE: V_RNDNE_F32{{.+}}v_rndne_f32_e32
; CHECK: call float @llvm.roundeven.f32
	v_rndne_f32_e32 v18, v19
; DECODE: V_FLOOR_F32{{.+}}v_floor_f32_e32
; CHECK: call float @llvm.floor.f32
	v_floor_f32_e32 v20, v21
; DECODE: V_EXP_F32{{.+}}v_exp_f32_e32
; CHECK: call float @llvm.amdgcn.exp2.f32
	v_exp_f32_e32 v22, v23
; DECODE: V_LOG_F32{{.+}}v_log_f32_e32
; CHECK: call float @llvm.amdgcn.log.f32
	v_log_f32_e32 v24, v25
; DECODE: V_RCP_F32{{.+}}v_rcp_f32_e32
; CHECK: call float @llvm.amdgcn.rcp.f32
	v_rcp_f32_e32 v26, v27
; DECODE: V_RSQ_F32{{.+}}v_rsq_f32_e32
; CHECK: call float @llvm.amdgcn.rsq.f32
	v_rsq_f32_e32 v28, v29
; DECODE: V_SQRT_F32{{.+}}v_sqrt_f32_e32
; CHECK: call float @llvm.amdgcn.sqrt.f32
	v_sqrt_f32_e32 v30, v31
; DECODE: V_SIN_F32{{.+}}v_sin_f32_e32
; CHECK: call float @llvm.amdgcn.sin.f32
	v_sin_f32_e32 v32, v33
; DECODE: V_COS_F32{{.+}}v_cos_f32_e32
; CHECK: call float @llvm.amdgcn.cos.f32
	v_cos_f32_e32 v34, v35
; DECODE: V_FREXP_EXP_I32_F32{{.+}}v_frexp_exp_i32_f32_e32
; CHECK: call i32 @llvm.amdgcn.frexp.exp.i32.f32
	v_frexp_exp_i32_f32_e32 v36, v37
; DECODE: V_FREXP_MANT_F32{{.+}}v_frexp_mant_f32_e32
; CHECK: call float @llvm.amdgcn.frexp.mant.f32
	v_frexp_mant_f32_e32 v38, v39
; CHECK: ret void
	s_endpgm

	.globl	vop3_math
	.p2align	8
	.type	vop3_math,@function
; CHECK-LABEL: define amdgpu_kernel void @vop3_math(
vop3_math:
; DECODE: V_EXP_F32{{.+}}v_exp_f32_e64 v0, -v1
; CHECK: [[NEG:%.+]] = fneg float
; CHECK: call float @llvm.amdgcn.exp2.f32(float [[NEG]])
	v_exp_f32_e64 v0, -v1
; DECODE: V_CVT_I32_F32{{.+}}v_cvt_i32_f32_e64 v8, |v9|
; CHECK: [[ABS:%.+]] = call float @llvm.fabs.f32
; CHECK: call i32 @llvm.fptosi.sat.i32.f32(float [[ABS]])
	v_cvt_i32_f32_e64 v8, abs(v9)
; DECODE: V_LDEXP_F32{{.+}}v_ldexp_f32
; CHECK: call float @llvm.ldexp.f32.i32
	v_ldexp_f32 v2, v3, v4
	s_mov_b64 s[4:5], -1
; DECODE: V_CNDMASK_B32{{.+}}v_cndmask_b32_e64
; CHECK: [[COND:%.+]] = trunc i64 {{.+}} to i1
; CHECK: select i1 [[COND]], i32
	v_cndmask_b32_e64 v5, v6, v7, s[4:5]
; CHECK: ret void
	s_endpgm

	.globl	refuse_output_modifier
	.p2align	8
	.type	refuse_output_modifier,@function
; REFUSE: unsupported-instruction-form: v_exp_f32 [VOP3]
; REFUSE-SAME: floating-point output clamp is not supported
refuse_output_modifier:
	v_exp_f32_e64 v0, v1 clamp
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop_math
		.amdhsa_next_free_vgpr 48
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 48
	.end_amdhsa_kernel
	.amdhsa_kernel vop3_math
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 12
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_output_modifier
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop_math
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vop_math.kd
    .vgpr_count:     48
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop3_math
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         vop3_math.kd
    .vgpr_count:     10
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_output_modifier
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_output_modifier.kd
    .vgpr_count:     2
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
