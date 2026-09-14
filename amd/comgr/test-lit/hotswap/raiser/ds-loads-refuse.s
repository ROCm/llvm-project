; REQUIRES: comgr-has-hotswap-transpile
; RUN: %llvm-mc -triple=amdgpu9-amd-amdhsa -filetype=obj %s -o %t.gfx9.o
; RUN: %ld.lld -shared %t.gfx9.o -o %t.gfx9.hsaco
; RUN: not %hotswap_transpile_cli %t.gfx9.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=ds_gds 2>&1 | %FileCheck %s --check-prefix=GDS
; RUN: %llvm-mc -triple=amdgpu8.03-amd-amdhsa -filetype=obj %s -o %t.gfx803.o
; RUN: %ld.lld -shared %t.gfx803.o -o %t.gfx803.hsaco
; RUN: not %hotswap_transpile_cli %t.gfx803.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=ds_m0 2>&1 | %FileCheck %s --check-prefix=M0
; RUN: %llvm-mc -triple=amdgpu10.10-amd-amdhsa -filetype=obj %s -o %t.gfx1010.o
; RUN: %ld.lld -shared %t.gfx1010.o -o %t.gfx1010.hsaco
; RUN: not %hotswap_transpile_cli %t.gfx1010.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=ds_wide 2>&1 | %FileCheck %s --check-prefix=WGP

	.amdhsa_code_object_version 6
	.text
	.globl ds_gds
	.p2align 8
	.type ds_gds,@function
ds_gds:
; GDS: in kernel 'ds_gds'
; GDS-SAME: GDS loads are not modeled
	ds_read_b32 v1, v0 gds
	s_endpgm

	.globl ds_m0
	.p2align 8
	.type ds_m0,@function
ds_m0:
; M0: in kernel 'ds_m0'
; M0-SAME: M0-bounded LDS loads are not modeled
	ds_read_b32 v1, v0
	s_endpgm

	.globl ds_wide
	.p2align 8
	.type ds_wide,@function
ds_wide:
; WGP: unsupported-instruction-form: ds_read_b64 [DS]
; WGP-SAME: in kernel 'ds_wide'
; WGP-SAME: wide LDS loads with the WGP misalignment bug are not modeled
	ds_read_b64 v[0:1], v0
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel ds_gds
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel
	.amdhsa_kernel ds_m0
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel
	.amdhsa_kernel ds_wide
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: ds_gds
    .symbol: ds_gds.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 2
    .wavefront_size: 64
  - .name: ds_m0
    .symbol: ds_m0.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 2
    .wavefront_size: 64
  - .name: ds_wide
    .symbol: ds_wide.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 2
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
