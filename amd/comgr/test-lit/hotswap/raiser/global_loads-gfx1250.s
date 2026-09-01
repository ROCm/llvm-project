; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_loads_gfx1250 | %FileCheck %s --check-prefix=IR
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_load_scaled_offset 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=SCALED-OFFSET

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	global_loads_gfx1250
	.p2align	8
	.type	global_loads_gfx1250,@function
; IR-LABEL: define amdgpu_kernel void @global_loads_gfx1250(
global_loads_gfx1250:
; The immediate offset occupies 24 signed bits here, so a value that does not
; fit the narrower field of earlier targets still reads back unchanged.
; Widening a wave32 source onto a wave64 target pins the address of an inactive
; lane, which would otherwise be free to take a different value per use.
; IR: [[FROZEN0:%.+]] = freeze i64 {{%.+}}
; IR: [[POINTER0:%.+]] = inttoptr i64 [[FROZEN0]] to ptr addrspace(1)
; IR: [[OFFSET0:%.+]] = getelementptr i8, ptr addrspace(1) [[POINTER0]], i64 65536
; IR: load i32, ptr addrspace(1) [[OFFSET0]], align 4
	global_load_b32 v1, v[2:3], off offset:65536

; The per-lane offset added to the scalar base is signed on this target.
; IR: [[BASE1:%.+]] = or i64 {{%.+}}, {{%.+}}
; IR: [[LANE1:%.+]] = sext i32 {{.+}} to i64
; IR: [[ADDRESS1:%.+]] = add i64 [[BASE1]], [[LANE1]]
; IR: [[FROZEN1:%.+]] = freeze i64 [[ADDRESS1]]
; IR: [[POINTER1:%.+]] = inttoptr i64 [[FROZEN1]] to ptr addrspace(1)
; IR: load i32, ptr addrspace(1) [[POINTER1]], align 4
	global_load_b32 v1, v0, s[0:1]
; IR: ret void
	s_endpgm

	.globl	global_load_scaled_offset
	.p2align	8
	.type	global_load_scaled_offset,@function
global_load_scaled_offset:
; SCALED-OFFSET: scaling the per-lane offset by the access size is not modeled
	global_load_b32 v1, v0, s[0:1] offset:32 scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_loads_gfx1250
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel global_load_scaled_offset
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
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
    .name:           global_loads_gfx1250
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_loads_gfx1250.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           global_load_scaled_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_scaled_offset.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
