; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=sopc_float | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sopc_float
	.p2align	8
	.type	sopc_float,@function
; IR-LABEL: define amdgpu_kernel void @sopc_float(
sopc_float:
	; IR: [[F32_SRC0:%.*]] = bitcast i32 {{.*}} to float
	; IR-NEXT: [[F32_SRC1:%.*]] = bitcast i32 {{.*}} to float
	; IR-NEXT: fcmp oeq float [[F32_SRC0]], [[F32_SRC1]]
	s_cmp_eq_f32 s0, s1
	; IR: fcmp one float {{.*}}
	s_cmp_lg_f32 s0, s1
	; IR: fcmp ogt float {{.*}}
	s_cmp_gt_f32 s0, s1
	; IR: fcmp oge float {{.*}}
	s_cmp_ge_f32 s0, s1
	; IR: fcmp olt float {{.*}}
	s_cmp_lt_f32 s0, s1
	; IR: fcmp ole float {{.*}}
	s_cmp_le_f32 s0, s1
	; IR: fcmp une float {{.*}}
	s_cmp_neq_f32 s0, s1
	; IR: fcmp ule float {{.*}}
	s_cmp_ngt_f32 s0, s1
	; IR: fcmp ult float {{.*}}
	s_cmp_nge_f32 s0, s1
	; IR: fcmp uge float {{.*}}
	s_cmp_nlt_f32 s0, s1
	; IR: fcmp ugt float {{.*}}
	s_cmp_nle_f32 s0, s1
	; IR: fcmp ueq float {{.*}}
	s_cmp_nlg_f32 s0, s1
	; IR: fcmp ord float {{.*}}
	s_cmp_o_f32 s0, s1
	; IR: fcmp uno float {{.*}}
	s_cmp_u_f32 s0, s1
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s2, 1, 0
	; IR: [[F16_BITS0:%.*]] = trunc i32 {{.*}} to i16
	; IR-NEXT: [[F16_BITS1:%.*]] = trunc i32 {{.*}} to i16
	; IR-NEXT: [[F16_SRC0:%.*]] = bitcast i16 [[F16_BITS0]] to half
	; IR-NEXT: [[F16_SRC1:%.*]] = bitcast i16 [[F16_BITS1]] to half
	; IR-NEXT: fcmp oeq half [[F16_SRC0]], [[F16_SRC1]]
	s_cmp_eq_f16 s0, s1
	; IR: fcmp one half {{.*}}
	s_cmp_lg_f16 s0, s1
	; IR: fcmp ogt half {{.*}}
	s_cmp_gt_f16 s0, s1
	; IR: fcmp oge half {{.*}}
	s_cmp_ge_f16 s0, s1
	; IR: fcmp olt half {{.*}}
	s_cmp_lt_f16 s0, s1
	; IR: fcmp ole half {{.*}}
	s_cmp_le_f16 s0, s1
	; IR: fcmp une half {{.*}}
	s_cmp_neq_f16 s0, s1
	; IR: fcmp ule half {{.*}}
	s_cmp_ngt_f16 s0, s1
	; IR: fcmp ult half {{.*}}
	s_cmp_nge_f16 s0, s1
	; IR: fcmp uge half {{.*}}
	s_cmp_nlt_f16 s0, s1
	; IR: fcmp ugt half {{.*}}
	s_cmp_nle_f16 s0, s1
	; IR: fcmp ueq half {{.*}}
	s_cmp_nlg_f16 s0, s1
	; IR: fcmp ord half {{.*}}
	s_cmp_o_f16 s0, s1
	; IR: fcmp uno half {{.*}}
	s_cmp_u_f16 s0, s1
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s2, 1, 0
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sopc_float
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
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
    .name:           sopc_float
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         sopc_float.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
