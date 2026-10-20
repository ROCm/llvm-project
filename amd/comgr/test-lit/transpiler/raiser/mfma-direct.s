; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=mfma_direct | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx908 \
; RUN:   --emit-ir=mfma_direct 2>&1 | %FileCheck %s --check-prefix=GFX908
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx90a \
; RUN:   --emit-ir=mfma_direct 2>&1 | %FileCheck %s --check-prefix=GFX90A

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	mfma_direct
	.p2align	8
	.type	mfma_direct,@function
; IR-LABEL: define amdgpu_kernel void @mfma_direct(
mfma_direct:
	s_mov_b64 exec, 1
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16
; IR: br i1 {{.*}}, label %spe_do, label %spe_skip
; IR: spe_do:
	v_mfma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7]
	s_mov_b64 exec, -1
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k
	v_mfma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7]
; IR: call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8({{.*}}zeroinitializer
	v_mfma_i32_16x16x32_i8 v[4:7], v[0:1], v[2:3], 0
; IR: ret void
	s_endpgm

; GFX908: unsupported-instruction-form: v_mfma_f32_16x16x16_bf16
; GFX908-SAME: target ISA does not support MFMA operation
; GFX90A: unsupported-instruction-form: v_mfma_i32_16x16x32_i8
; GFX90A-SAME: target ISA does not support MFMA operation

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mfma_direct
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 8
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
    .name:           mfma_direct
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         mfma_direct.kd
    .vgpr_count:     8
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
