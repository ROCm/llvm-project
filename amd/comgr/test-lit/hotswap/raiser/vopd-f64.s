; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vopd_f64 | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_f64
	.p2align	8
	.type	vopd_f64,@function
; CHECK-LABEL: define amdgpu_kernel void @vopd_f64(
vopd_f64:
; CHECK: fneg double
; CHECK: call double @llvm.fma.f64
	v_dual_fma_f64 v[32:33], -v[48:49], v[54:55], v[32:33] :: v_dual_mov_b32 v48, v43
; CHECK: fmul double
	v_dual_mul_f64 v[6:7], s[20:21], v[18:19] :: v_dual_mov_b32 v3, 0
; CHECK: fadd double
	v_dual_add_f64 v[8:9], v[16:17], v[20:21] :: v_dual_mov_b32 v4, 0
; CHECK: call double @llvm.maximumnum.f64
	v_dual_max_num_f64 v[10:11], v[22:23], v[24:25] :: v_dual_mov_b32 v5, 0
; CHECK: call double @llvm.minimumnum.f64
	v_dual_min_num_f64 v[12:13], v[26:27], v[28:29] :: v_dual_mov_b32 v6, 0
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_f64
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 56
		.amdhsa_next_free_sgpr 24
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vopd_f64
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         vopd_f64.kd
    .vgpr_count:     56
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
