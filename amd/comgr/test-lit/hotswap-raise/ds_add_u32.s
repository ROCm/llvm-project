; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>&1 | %FileCheck %s

; Lift ds_add_u32/ds_add_rtn_u32 to i32 atomicrmw add on addrspace(3).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_add_u32_kernel
	.p2align	8
	.type	ds_add_u32_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @ds_add_u32_kernel(
ds_add_u32_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, 1
; CHECK: atomicrmw add ptr addrspace(3) %{{.+}}, i32 {{.+}} seq_cst
	ds_add_u32 v0, v2
; CHECK: atomicrmw add ptr addrspace(3) %{{.+}}, i32 {{.+}} seq_cst
	ds_add_rtn_u32 v4, v0, v2
	s_endpgm
; CHECK-NOT: atomicrmw fadd
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_add_u32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_group_segment_fixed_size 16
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 16
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           ds_add_u32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         ds_add_u32_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
