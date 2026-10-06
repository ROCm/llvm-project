; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=ds_atomic_add | %FileCheck %s

; The LDS integer atomic add lifts to an atomicrmw at its natural alignment.
; The returning form publishes the pre-add memory value to its destination.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl ds_atomic_add
	.p2align 8
	.type ds_atomic_add,@function
; CHECK-LABEL: define amdgpu_kernel void @ds_atomic_add(
ds_atomic_add:
	v_mov_b32 v0, 4
	v_mov_b32 v1, 7
; CHECK: [[BASE:%.+]] = phi i32 [ 4, {{.+}}
; CHECK: add i32 [[BASE]], 8
; CHECK: [[PTR:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: atomicrmw add ptr addrspace(3) [[PTR]], i32 {{.+}} seq_cst, align 4
	ds_add_u32 v0, v1 offset:8
; The returning form keeps the value the memory held before the add.
; CHECK: add i32 [[BASE]], 12
; CHECK: [[RPTR:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: [[OLD:%.+]] = atomicrmw add ptr addrspace(3) [[RPTR]], i32 {{.+}} seq_cst, align 4
	ds_add_rtn_u32 v2, v0, v1 offset:12
	s_wait_dscnt 0
; The pre-add value reaches the destination register and can be stored back.
; CHECK: [[DEST:%.+]] = phi i32 [ [[OLD]], {{.+}}
; CHECK: store i32 [[DEST]], ptr addrspace(3)
	ds_store_b32 v0, v2 offset:16
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel ds_atomic_add
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: ds_atomic_add
    .symbol: ds_atomic_add.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
