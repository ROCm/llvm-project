; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vop_math,vop3_math | %FileCheck %s
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_clamp 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-CLAMP
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_omod 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-OMOD
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_true16_destination 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-TRUE16

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vop_math
	.p2align	8
	.type	vop_math,@function
; CHECK-LABEL: define amdgpu_kernel void @vop_math(
vop_math:
; CHECK: sitofp i32
	v_cvt_f32_i32_e32 v0, v1
; CHECK: uitofp i32
	v_cvt_f32_u32_e32 v2, v3
; CHECK: call i32 @llvm.fptosi.sat.i32.f32
	v_cvt_i32_f32_e32 v4, v5
; CHECK: call i32 @llvm.fptoui.sat.i32.f32
	v_cvt_u32_f32_e32 v6, v7
; CHECK: fpext half
	v_cvt_f32_f16_e32 v10, v11.l
; CHECK: uitofp i32
	v_cvt_f32_ubyte0_e32 v40, v41
; CHECK: lshr i32 {{.+}}, 8
; CHECK: uitofp i32
	v_cvt_f32_ubyte1_e32 v42, v43
; CHECK: lshr i32 {{.+}}, 16
; CHECK: uitofp i32
	v_cvt_f32_ubyte2_e32 v44, v45
; CHECK: lshr i32 {{.+}}, 24
; CHECK: uitofp i32
	v_cvt_f32_ubyte3_e32 v46, v47
; CHECK: call float @llvm.amdgcn.fract.f32
	v_fract_f32_e32 v12, v13
; CHECK: call float @llvm.trunc.f32
	v_trunc_f32_e32 v14, v15
; CHECK: call float @llvm.ceil.f32
	v_ceil_f32_e32 v16, v17
; CHECK: call float @llvm.roundeven.f32
	v_rndne_f32_e32 v18, v19
; CHECK: call float @llvm.floor.f32
	v_floor_f32_e32 v20, v21
; CHECK: call float @llvm.amdgcn.exp2.f32
	v_exp_f32_e32 v22, v23
; CHECK: call float @llvm.amdgcn.log.f32
	v_log_f32_e32 v24, v25
; CHECK: call float @llvm.amdgcn.rcp.f32
	v_rcp_f32_e32 v26, v27
; CHECK: call float @llvm.amdgcn.rsq.f32
	v_rsq_f32_e32 v28, v29
; CHECK: call float @llvm.amdgcn.sqrt.f32
	v_sqrt_f32_e32 v30, v31
; CHECK: call float @llvm.amdgcn.sin.f32
	v_sin_f32_e32 v32, v33
; CHECK: call float @llvm.amdgcn.cos.f32
	v_cos_f32_e32 v34, v35
; CHECK: call i32 @llvm.amdgcn.frexp.exp.i32.f32
	v_frexp_exp_i32_f32_e32 v36, v37
; CHECK: call float @llvm.amdgcn.frexp.mant.f32
	v_frexp_mant_f32_e32 v38, v39
; CHECK: ret void
	s_endpgm

	.globl	vop3_math
	.p2align	8
	.type	vop3_math,@function
; CHECK-LABEL: define amdgpu_kernel void @vop3_math(
vop3_math:
; CHECK: [[NEG:%.+]] = fneg float
; CHECK: call float @llvm.amdgcn.exp2.f32(float [[NEG]])
	v_exp_f32_e64 v0, -v1
; CHECK: [[ABS:%.+]] = call float @llvm.fabs.f32
; CHECK: call i32 @llvm.fptosi.sat.i32.f32(float [[ABS]])
	v_cvt_i32_f32_e64 v8, abs(v9)
; CHECK: [[HIGH:%.+]] = lshr i32 {{.+}}, 16
; CHECK: [[LOW:%.+]] = trunc i32 [[HIGH]] to i16
; CHECK: [[HALF:%.+]] = bitcast i16 [[LOW]] to half
; CHECK: fpext half [[HALF]] to float
	v_cvt_f32_f16_e64 v10, v11.h
; CHECK: call float @llvm.ldexp.f32.i32
	v_ldexp_f32 v2, v3, v4
	s_mov_b32 s4, -1
; CHECK: [[COND:%.+]] = trunc i64 {{.+}} to i1
; CHECK: select i1 [[COND]], i32
	v_cndmask_b32_e64 v5, v6, v7, s4
; CHECK: ret void
	s_endpgm

	.globl	refuse_clamp
	.p2align	8
	.type	refuse_clamp,@function
; REFUSE-CLAMP: unsupported-instruction-form: v_exp_f32 [VOP3]
; REFUSE-CLAMP-SAME: floating-point output clamp is not supported
refuse_clamp:
	v_exp_f32_e64 v0, v1 clamp
	s_endpgm

	.globl	refuse_omod
	.p2align	8
	.type	refuse_omod,@function
; REFUSE-OMOD: unsupported-instruction-form: v_exp_f32 [VOP3]
; REFUSE-OMOD-SAME: floating-point output multiplier is not supported
refuse_omod:
	v_exp_f32_e64 v0, v1 mul:2
	s_endpgm

	.globl	refuse_true16_destination
	.p2align	8
	.type	refuse_true16_destination,@function
; REFUSE-TRUE16: unsupported-instruction-form: v_cvt_f16_f32 [VOP1]
; REFUSE-TRUE16-SAME: true16 destination preservation is not supported
refuse_true16_destination:
	v_cvt_f16_f32_e32 v0.l, v1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop_math
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 48
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel vop3_math
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_clamp
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_omod
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_true16_destination
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
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
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop3_math
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         vop3_math.kd
    .vgpr_count:     12
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_clamp
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_clamp.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_omod
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_omod.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_true16_destination
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_true16_destination.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
