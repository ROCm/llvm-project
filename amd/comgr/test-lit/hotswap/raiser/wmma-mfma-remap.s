; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --isa=gfx1250 \
; RUN:   --dump-decoded=wmma_remap | %FileCheck %s --check-prefix=DECODE
; RUN: %hotswap_transpile_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:   --emit-ir=wmma_remap | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_remap
	.p2align	8
	.type	wmma_remap,@function
; IR-LABEL: define amdgpu_kernel void @wmma_remap(
wmma_remap:
; DECODE: V_WMMA_F32_16x16x32_F16{{.+}}v_wmma_f32_16x16x32_f16
; IR: call i32 @llvm.amdgcn.ds.bpermute
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16
	v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
; DECODE: V_WMMA_F32_16x16x32_BF16{{.+}}v_wmma_f32_16x16x32_bf16
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
; DECODE: V_WMMA_I32_16x16x64_IU8{{.+}}v_wmma_i32_16x16x64_iu8
; IR: call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8
	v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_remap
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 1
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
    .name:           wmma_remap
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         wmma_remap.kd
    .vgpr_count:     24
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
