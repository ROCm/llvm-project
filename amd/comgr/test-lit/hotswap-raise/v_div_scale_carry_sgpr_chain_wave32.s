; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=div_carry_sgpr_chain_kernel 2>/dev/null | %FileCheck %s

; v_div_scale_f32 carry-out threaded via vcc to v_div_fmas_f32.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	div_carry_sgpr_chain_kernel
	.p2align	8
	.type	div_carry_sgpr_chain_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @div_carry_sgpr_chain_kernel(
div_carry_sgpr_chain_kernel:
	v_div_scale_f32 v3, s2, v1, v1, v0
	; CHECK: %divscale = call { float, i1 } @llvm.amdgcn.div.scale.f32(
	; CHECK: [[CARRY:%[0-9]+]] = extractvalue { float, i1 } %divscale, 1
	s_mov_b32 s5, s2
	s_mov_b32 vcc_lo, s5
	v_div_fmas_f32 v3, v3, v1, v0
	; CHECK-NOT: sgpr_mask_shadow_sel
	; CHECK-NOT: mask_widen
	; CHECK: call float @llvm.amdgcn.div.fmas.f32(float %{{.+}}, float %{{.+}}, float %{{.+}}, i1 [[CARRY]])
	ds_store_b32 v4, v3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel div_carry_sgpr_chain_kernel
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           div_carry_sgpr_chain_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         div_carry_sgpr_chain_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
