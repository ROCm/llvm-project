; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>&1 | %FileCheck %s

; VOP3 neg/abs source modifiers on f64 VALU ops. src64() returns the raw
; operand (unlike srcF()), so these handlers must apply the modifiers
; explicitly -- see applyMods in raise-context.h. Grouped: dropping the
; negate miscompiles the f64 rcp/div Newton-Raphson refinement.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	f64_src_modifiers_kernel
	.p2align	8
	.type	f64_src_modifiers_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @f64_src_modifiers_kernel(
f64_src_modifiers_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0x3ff00000
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0x40000000
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0x40080000
; CHECK: [[ADDNEG:%.+]] = fneg double
; CHECK: fadd double [[ADDNEG]],
	v_add_f64 v[6:7], -v[0:1], v[2:3]
; CHECK: [[MULABS:%.+]] = call double @llvm.fabs.f64(double
; CHECK: fmul double [[MULABS]],
	v_mul_f64 v[8:9], |v[0:1]|, v[2:3]
; CHECK: [[RCPNEG:%.+]] = fneg double
; CHECK: call double @llvm.amdgcn.rcp.f64(double [[RCPNEG]])
	v_rcp_f64 v[10:11], -v[0:1]
; CHECK: [[FMANEG:%.+]] = fneg double
; CHECK: call double @llvm.fma.f64(double [[FMANEG]],
	v_fma_f64 v[12:13], -v[0:1], v[2:3], v[4:5]
; CHECK: [[CVTNEG:%.+]] = fneg double
; CHECK: call i32 @llvm.fptoui.sat.i32.f64(double [[CVTNEG]])
	v_cvt_u32_f64 v14, -v[0:1]
; CHECK: [[CMPNEG:%.+]] = fneg double
; CHECK: fcmp olt double [[CMPNEG]],
	v_cmp_lt_f64 vcc, -v[0:1], v[2:3]
; CHECK: [[CLSABS:%.+]] = call double @llvm.fabs.f64(double
; CHECK: call i1 @llvm.amdgcn.class.f64(double [[CLSABS]],
	v_cmp_class_f64 vcc, |v[0:1]|, v2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel f64_src_modifiers_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
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
    .name:           f64_src_modifiers_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         f64_src_modifiers_kernel.kd
    .vgpr_count:     16
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
