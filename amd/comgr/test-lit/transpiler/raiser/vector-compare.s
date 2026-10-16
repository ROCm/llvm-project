; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --dump-decoded=vector_comparisons | %FileCheck %s --check-prefix=DECODE
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=vector_comparisons | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl vector_comparisons
	.p2align 8
	.type vector_comparisons,@function
; CHECK-LABEL: define amdgpu_kernel void @vector_comparisons(
vector_comparisons:
; DECODE: V_CMP_CLASS_F32 v_cmp_class_f32_e64
; CHECK: call i1 @llvm.amdgcn.class.f32(float {{.+}}, i32 {{.+}})
	v_cmp_class_f32_e64 s4, v0, v1
; DECODE: V_CMP_CLASS_F64 v_cmp_class_f64_e64
; CHECK: call i1 @llvm.amdgcn.class.f64(double {{.+}}, i32 {{.+}})
	v_cmp_class_f64_e64 s4, v[0:1], v2
; DECODE: V_CMP_EQ_F32 v_cmp_eq_f32_e32
; CHECK: fcmp oeq float
	v_cmp_eq_f32_e32 vcc_lo, v0, v1
; DECODE: V_CMP_EQ_F32 v_cmp_eq_f32_e64
; CHECK: fcmp oeq float
	v_cmp_eq_f32_e64 s4, v0, v1
; DECODE: V_CMP_EQ_U16 v_cmp_eq_u16_e32
; CHECK: icmp eq i16
	v_cmp_eq_u16_e32 vcc_lo, v0, v1
; DECODE: V_CMP_EQ_U16 v_cmp_eq_u16_e64
; CHECK: icmp eq i16
	v_cmp_eq_u16_e64 s4, v0, v1
; DECODE: V_CMP_EQ_U64 v_cmp_eq_u64_e32
; CHECK: icmp eq i64
	v_cmp_eq_u64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_EQ_U64 v_cmp_eq_u64_e64
; CHECK: icmp eq i64
	v_cmp_eq_u64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_GE_I64 v_cmp_ge_i64_e32
; CHECK: icmp sge i64
	v_cmp_ge_i64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_GE_I64 v_cmp_ge_i64_e64
; CHECK: icmp sge i64
	v_cmp_ge_i64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_GE_U64 v_cmp_ge_u64_e64
; CHECK: icmp uge i64
	v_cmp_ge_u64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_GT_F32 v_cmp_gt_f32_e32
; CHECK: fcmp ogt float
	v_cmp_gt_f32_e32 vcc_lo, v0, v1
; DECODE: V_CMP_GT_F32 v_cmp_gt_f32_e64
; CHECK: fcmp ogt float
	v_cmp_gt_f32_e64 s4, v0, v1
; DECODE: V_CMP_GT_I64 v_cmp_gt_i64_e32
; CHECK: icmp sgt i64
	v_cmp_gt_i64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_GT_I64 v_cmp_gt_i64_e64
; CHECK: icmp sgt i64
	v_cmp_gt_i64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_LE_I64 v_cmp_le_i64_e32
; CHECK: icmp sle i64
	v_cmp_le_i64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_LE_I64 v_cmp_le_i64_e64
; CHECK: icmp sle i64
	v_cmp_le_i64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_LE_U64 v_cmp_le_u64_e32
; CHECK: icmp ule i64
	v_cmp_le_u64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_LG_F32 v_cmp_lg_f32_e64
; CHECK: fcmp one float
	v_cmp_lg_f32_e64 s4, v0, v1
; DECODE: V_CMP_LT_F32 v_cmp_lt_f32_e32
; CHECK: fcmp olt float
	v_cmp_lt_f32_e32 vcc_lo, v0, v1
; DECODE: V_CMP_LT_F32 v_cmp_lt_f32_e64
; CHECK: fcmp olt float
	v_cmp_lt_f32_e64 s4, v0, v1
; DECODE: V_CMP_LT_I64 v_cmp_lt_i64_e32
; CHECK: icmp slt i64
	v_cmp_lt_i64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_LT_I64 v_cmp_lt_i64_e64
; CHECK: icmp slt i64
	v_cmp_lt_i64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_LT_U64 v_cmp_lt_u64_e64
; CHECK: icmp ult i64
	v_cmp_lt_u64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_NE_U16 v_cmp_ne_u16_e32
; CHECK: icmp ne i16
	v_cmp_ne_u16_e32 vcc_lo, v0, v1
; DECODE: V_CMP_NE_U16 v_cmp_ne_u16_e64
; CHECK: icmp ne i16
	v_cmp_ne_u16_e64 s4, v0, v1
; DECODE: V_CMP_NE_U64 v_cmp_ne_u64_e32
; CHECK: icmp ne i64
	v_cmp_ne_u64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_NE_U64 v_cmp_ne_u64_e64
; CHECK: icmp ne i64
	v_cmp_ne_u64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_NEQ_F32 v_cmp_neq_f32_e32
; CHECK: fcmp une float
	v_cmp_neq_f32_e32 vcc_lo, v0, v1
; DECODE: V_CMP_NEQ_F32 v_cmp_neq_f32_e64
; CHECK: fcmp une float
	v_cmp_neq_f32_e64 s4, v0, v1
; DECODE: V_CMP_NEQ_F64 v_cmp_neq_f64_e32
; CHECK: fcmp une double
	v_cmp_neq_f64_e32 vcc_lo, v[0:1], v[2:3]
; DECODE: V_CMP_NEQ_F64 v_cmp_neq_f64_e64
; CHECK: fcmp une double
	v_cmp_neq_f64_e64 s4, v[0:1], v[2:3]
; DECODE: V_CMP_NGT_F32 v_cmp_ngt_f32_e32
; CHECK: fcmp ule float
	v_cmp_ngt_f32_e32 vcc_lo, v0, v1
; DECODE: V_CMP_NGT_F32 v_cmp_ngt_f32_e64
; CHECK: fcmp ule float
	v_cmp_ngt_f32_e64 s4, v0, v1
; DECODE: V_CMP_NLT_F32 v_cmp_nlt_f32_e32
; CHECK: fcmp uge float
	v_cmp_nlt_f32_e32 vcc_lo, v0, v1
; DECODE: V_CMP_NLT_F32 v_cmp_nlt_f32_e64
; CHECK: fcmp uge float
	v_cmp_nlt_f32_e64 s4, v0, v1
; DECODE: V_CMP_O_F32 v_cmp_o_f32_e32
; CHECK: fcmp ord float
	v_cmp_o_f32_e32 vcc_lo, v0, v1
; DECODE: V_CMP_O_F32 v_cmp_o_f32_e64
; CHECK: fcmp ord float
	v_cmp_o_f32_e64 s4, v0, v1
; DECODE: V_CMP_U_F32 v_cmp_u_f32_e64
; CHECK: fcmp uno float
	v_cmp_u_f32_e64 s4, v0, v1
; DECODE: V_CMP_EQ_U16 v_cmpx_eq_u16_e32
; CHECK: icmp eq i16
	v_cmpx_eq_u16_e32 v0, v1
; DECODE: V_CMP_EQ_U64 v_cmpx_eq_u64_e32
; CHECK: icmp eq i64
	v_cmpx_eq_u64_e32 v[0:1], v[2:3]
; DECODE: V_CMP_GT_I64 v_cmpx_gt_i64_e64
; CHECK: icmp sgt i64
	v_cmpx_gt_i64_e64 v[0:1], v[2:3]
; DECODE: V_CMP_GT_U64 v_cmpx_gt_u64_e64
; CHECK: icmp ugt i64
	v_cmpx_gt_u64_e64 v[0:1], v[2:3]
; DECODE: V_CMP_LT_I16 v_cmpx_lt_i16_e32
; CHECK: icmp slt i16
	v_cmpx_lt_i16_e32 v0, v1
; DECODE: V_CMP_NE_U64 v_cmpx_ne_u64_e32
; CHECK: icmp ne i64
	v_cmpx_ne_u64_e32 v[0:1], v[2:3]
; DECODE: V_CMP_NGT_F32 v_cmpx_ngt_f32_e32
; CHECK: fcmp ule float
	v_cmpx_ngt_f32_e32 v0, v1
; DECODE: V_CMP_NGT_F32 v_cmpx_ngt_f32_e64
; CHECK: fcmp ule float
	v_cmpx_ngt_f32_e64 v0, v1
; DECODE: V_CMP_O_F32 v_cmpx_o_f32_e32
; CHECK: fcmp ord float
	v_cmpx_o_f32_e32 v0, v1
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel vector_comparisons
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: vector_comparisons
    .symbol: vector_comparisons.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
