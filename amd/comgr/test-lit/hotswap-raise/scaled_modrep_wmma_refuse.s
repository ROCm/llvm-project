; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && not raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=scaled_modrep_wmma_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s

; A matrix fragment spans all target lanes and cannot be fed from replicas, so
; the scaled route refuses wmma/mfma. max_flat_workgroup_size=256 keeps it under
; the size gate so the matrix refusal is what fires.

; CHECK: ScaledModuloReplicationProjection cannot lower wmma/mfma
; CHECK-SAME: matrix fragment spans all target lanes
; CHECK-NOT: define amdgpu_kernel void @scaled_modrep_wmma_refuse_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	scaled_modrep_wmma_refuse_kernel
	.p2align	8
	.type	scaled_modrep_wmma_refuse_kernel,@function
scaled_modrep_wmma_refuse_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
	global_store_b128 v24, v[16:19], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scaled_modrep_wmma_refuse_kernel
		.amdhsa_next_free_vgpr 25
		.amdhsa_next_free_sgpr 2
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 256
    .name:           scaled_modrep_wmma_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         scaled_modrep_wmma_refuse_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
