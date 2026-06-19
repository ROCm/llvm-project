; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=vopd_f64_kernel 2>/dev/null | %FileCheck %s
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=vopd_f64_kernel 2>/dev/null | %FileCheck %s --check-prefix=CROSS
;
; Pins gfx1250 VOPD3 FP64 components. These packets use 64-bit sources and
; destinations even though older VOPD support in the raiser only committed
; 32-bit component results. The fixture mirrors the real rocBLAS blockers:
; `v_dual_fma_f64 ... :: v_dual_mov_b32 ...` and
; `v_dual_mul_f64 ... :: v_dual_mov_b32 ...`, and covers the full FP64
; VOPD3 component family LLVM exposes on gfx1250.

; CHECK-LABEL: define amdgpu_kernel void @vopd_f64_kernel(

; The FMA component must preserve fused semantics and apply the VOPD3 source
; modifier as f64 negation, not as a 32-bit f32 modifier.
; CHECK: %vopd_neg = fneg double
; CHECK: %vopd_fma_f64 = call double @llvm.fma.f64(double %{{[^,]+}}, double %{{[^,]+}}, double %{{[^)]+}})

; The MUL component is an ordinary f64 multiply. The companion MOV_B32 halves
; in both packets ensure mixed-width VOPD commits still happen after both
; components have read their pre-instruction inputs.
; CHECK: %vopd_fmul_f64 = fmul double

; The remaining gfx1250 FP64 VOPD3 components are the two-source add and
; maximumNumber/minimumNumber operations.
; CHECK: %vopd_fadd_f64 = fadd double
; CHECK: %vopd_fmaxnum_f64 = call double @llvm.maximumnum.f64
; CHECK: %vopd_fminnum_f64 = call double @llvm.minimumnum.f64

; CHECK-NOT: unhandled structural VOPD component SemOp
; CROSS-LABEL: define amdgpu_kernel void @vopd_f64_kernel(
; CROSS: call double @llvm.fma.f64
; CROSS: fmul double
; CROSS: fadd double
; CROSS: call double @llvm.maximumnum.f64
; CROSS: call double @llvm.minimumnum.f64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_f64_kernel
	.p2align	8
	.type	vopd_f64_kernel,@function
vopd_f64_kernel:
	v_dual_fma_f64 v[32:33], -v[48:49], v[54:55], v[32:33] :: v_dual_mov_b32 v48, v43
	v_dual_mul_f64 v[6:7], s[20:21], v[18:19] :: v_dual_mov_b32 v3, 0
	v_dual_add_f64 v[8:9], v[16:17], v[20:21] :: v_dual_mov_b32 v4, 0
	v_dual_max_num_f64 v[10:11], v[22:23], v[24:25] :: v_dual_mov_b32 v5, 0
	v_dual_min_num_f64 v[12:13], v[26:27], v[28:29] :: v_dual_mov_b32 v6, 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_f64_kernel
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
    .name: vopd_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .symbol: vopd_f64_kernel.kd
    .vgpr_count: 56
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
