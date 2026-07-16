; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>&1 | %FileCheck %s

; Lift v_bcnt_u32_b32 dst, s0, s1 to ctpop(s0) + s1.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	v_bcnt_u32_b32_kernel
	.p2align	8
	.type	v_bcnt_u32_b32_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_bcnt_u32_b32_kernel(
v_bcnt_u32_b32_kernel:
	v_mov_b32_e32 v0, 0xff
	v_mov_b32_e32 v1, 3
; CHECK: %[[POP:.+]] = call i32 @llvm.ctpop.i32(i32 %{{.+}})
; CHECK: add i32 %[[POP]], {{.+}}
	v_bcnt_u32_b32 v2, v0, v1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_bcnt_u32_b32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
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
    .name:           v_bcnt_u32_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_bcnt_u32_b32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
