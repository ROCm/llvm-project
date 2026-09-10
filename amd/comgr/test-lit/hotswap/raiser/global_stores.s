; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_stores | %FileCheck %s --check-prefix=IR
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_store_pair 2>&1 | %FileCheck %s --check-prefix=PAIR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text

	.globl	global_stores
	.p2align	8
	.type	global_stores,@function
; IR-LABEL: define amdgpu_kernel void @global_stores(
global_stores:
; A lane that EXEC masks off must not reach memory, so the store is guarded.
; IR: [[POINTER0:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: br i1 {{%.+}}, label {{%.+}}, label {{%.+}}
; IR: store i32 {{.+}}, ptr addrspace(1) [[POINTER0]], align 4
	global_store_dword v[2:3], v1, off

; IR: [[BASE1:%.+]] = or i64 {{%.+}}, {{%.+}}
; IR: [[LANE1:%.+]] = zext i32 {{.+}} to i64
; IR: [[ADDRESS1:%.+]] = add i64 [[BASE1]], [[LANE1]]
; IR: [[POINTER1:%.+]] = inttoptr i64 [[ADDRESS1]] to ptr addrspace(1)
; IR: [[OFFSET1:%.+]] = getelementptr i8, ptr addrspace(1) [[POINTER1]], i64 -16
; IR: store i32 {{.+}}, ptr addrspace(1) [[OFFSET1]], align 4
	global_store_dword v0, v1, s[0:1] offset:-16
; IR: ret void
	s_endpgm

	.globl	global_store_pair
	.p2align	8
	.type	global_store_pair,@function
global_store_pair:
; PAIR: unsupported flat memory operation
	global_store_dwordx2 v[2:3], v[0:1], off
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_stores
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel global_store_pair
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_stores
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_stores.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_store_pair
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_store_pair.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
