; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=ds_subdword_loads | %FileCheck %s

; A sub-dword DS load narrows the access to the encoded width and extends the
; result into the whole destination register, signed for the i8 and i16 forms.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl ds_subdword_loads
	.p2align 8
	.type ds_subdword_loads,@function
; CHECK-LABEL: define amdgpu_kernel void @ds_subdword_loads(
ds_subdword_loads:
	v_mov_b32 v0, 4
; CHECK: [[BASE:%.+]] = phi i32 [ 4, {{.+}}
; CHECK: [[U8A:%.+]] = add i32 [[BASE]], 1
; CHECK: [[U8P:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: [[U8V:%.+]] = load i8, ptr addrspace(3) [[U8P]], align 1
; CHECK-NEXT: zext i8 [[U8V]] to i32
	ds_load_u8 v1, v0 offset:1
; CHECK: add i32 [[BASE]], 2
; CHECK: [[I8P:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: [[I8V:%.+]] = load i8, ptr addrspace(3) [[I8P]], align 1
; CHECK-NEXT: sext i8 [[I8V]] to i32
	ds_load_i8 v2, v0 offset:2
; CHECK: add i32 [[BASE]], 4
; CHECK: [[U16P:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: [[U16V:%.+]] = load i16, ptr addrspace(3) [[U16P]], align 1
; CHECK-NEXT: zext i16 [[U16V]] to i32
	ds_load_u16 v3, v0 offset:4
; CHECK: add i32 [[BASE]], 6
; CHECK: [[I16P:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: [[I16V:%.+]] = load i16, ptr addrspace(3) [[I16P]], align 1
; CHECK-NEXT: sext i16 [[I16V]] to i32
	ds_load_i16 v4, v0 offset:6
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel ds_subdword_loads
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: ds_subdword_loads
    .symbol: ds_subdword_loads.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
