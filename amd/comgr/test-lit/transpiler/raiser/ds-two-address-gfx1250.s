; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=ds_two_address_loads,ds_two_address_stores | %FileCheck %s

; Each immediate offset of a two-address DS access scales by the access width,
; and by a further 64 elements for the stride64 forms. The two accesses are
; independent, so the gap between the offsets has to survive the lift.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl ds_two_address_loads
	.p2align 8
	.type ds_two_address_loads,@function
; CHECK-LABEL: define amdgpu_kernel void @ds_two_address_loads(
ds_two_address_loads:
	v_mov_b32 v0, 4
; CHECK: [[BASE:%.+]] = phi i32 [ 4, {{.+}}
; CHECK: [[B32A:%.+]] = add i32 [[BASE]], 4
; CHECK-NEXT: [[B32FA:%.+]] = freeze i32 [[B32A]]
; CHECK: [[B32B:%.+]] = add i32 [[BASE]], 12
; CHECK-NEXT: [[B32FB:%.+]] = freeze i32 [[B32B]]
; CHECK: [[B32PA:%.+]] = inttoptr i32 [[B32FA]] to ptr addrspace(3)
; CHECK-NEXT: load i32, ptr addrspace(3) [[B32PA]], align 1
; CHECK: [[B32PB:%.+]] = inttoptr i32 [[B32FB]] to ptr addrspace(3)
; CHECK-NEXT: load i32, ptr addrspace(3) [[B32PB]], align 1
	ds_load_2addr_b32 v[2:3], v0 offset0:1 offset1:3
; CHECK: [[B64A:%.+]] = add i32 [[BASE]], 8
; CHECK-NEXT: [[B64FA:%.+]] = freeze i32 [[B64A]]
; CHECK: [[B64B:%.+]] = add i32 [[BASE]], 24
; CHECK-NEXT: [[B64FB:%.+]] = freeze i32 [[B64B]]
; CHECK: [[B64PA:%.+]] = inttoptr i32 [[B64FA]] to ptr addrspace(3)
; CHECK-NEXT: load i64, ptr addrspace(3) [[B64PA]], align 1
; CHECK: [[B64PB:%.+]] = inttoptr i32 [[B64FB]] to ptr addrspace(3)
; CHECK-NEXT: load i64, ptr addrspace(3) [[B64PB]], align 1
	ds_load_2addr_b64 v[4:7], v0 offset0:1 offset1:3
; CHECK: [[S32A:%.+]] = add i32 [[BASE]], 256
; CHECK-NEXT: [[S32FA:%.+]] = freeze i32 [[S32A]]
; CHECK: [[S32B:%.+]] = add i32 [[BASE]], 768
; CHECK-NEXT: [[S32FB:%.+]] = freeze i32 [[S32B]]
; CHECK: [[S32PA:%.+]] = inttoptr i32 [[S32FA]] to ptr addrspace(3)
; CHECK-NEXT: load i32, ptr addrspace(3) [[S32PA]], align 1
; CHECK: [[S32PB:%.+]] = inttoptr i32 [[S32FB]] to ptr addrspace(3)
; CHECK-NEXT: load i32, ptr addrspace(3) [[S32PB]], align 1
	ds_load_2addr_stride64_b32 v[8:9], v0 offset0:1 offset1:3
; CHECK: [[S64A:%.+]] = add i32 [[BASE]], 512
; CHECK-NEXT: [[S64FA:%.+]] = freeze i32 [[S64A]]
; CHECK: [[S64B:%.+]] = add i32 [[BASE]], 1536
; CHECK-NEXT: [[S64FB:%.+]] = freeze i32 [[S64B]]
; CHECK: [[S64PA:%.+]] = inttoptr i32 [[S64FA]] to ptr addrspace(3)
; CHECK-NEXT: load i64, ptr addrspace(3) [[S64PA]], align 1
; CHECK: [[S64PB:%.+]] = inttoptr i32 [[S64FB]] to ptr addrspace(3)
; CHECK-NEXT: load i64, ptr addrspace(3) [[S64PB]], align 1
	ds_load_2addr_stride64_b64 v[10:13], v0 offset0:1 offset1:3
	s_endpgm

	.globl ds_two_address_stores
	.p2align 8
	.type ds_two_address_stores,@function
; CHECK-LABEL: define amdgpu_kernel void @ds_two_address_stores(
ds_two_address_stores:
	v_mov_b32 v0, 4
	v_mov_b32 v1, 11
	v_mov_b32 v2, 22
; CHECK: [[SBASE:%.+]] = phi i32 [ 4, {{.+}}
; The data registers reach the two offsets in operand order.
; CHECK: [[WA:%.+]] = add i32 [[SBASE]], 4
; CHECK-NEXT: [[WFA:%.+]] = freeze i32 [[WA]]
; CHECK: [[WB:%.+]] = add i32 [[SBASE]], 12
; CHECK-NEXT: [[WFB:%.+]] = freeze i32 [[WB]]
; CHECK: [[WPA:%.+]] = inttoptr i32 [[WFA]] to ptr addrspace(3)
; CHECK-NEXT: store i32 11, ptr addrspace(3) [[WPA]], align 1
; CHECK: [[WPB:%.+]] = inttoptr i32 [[WFB]] to ptr addrspace(3)
; CHECK-NEXT: store i32 22, ptr addrspace(3) [[WPB]], align 1
	ds_store_2addr_b32 v0, v1, v2 offset0:1 offset1:3
; CHECK: [[XA:%.+]] = add i32 [[SBASE]], 256
; CHECK-NEXT: [[XFA:%.+]] = freeze i32 [[XA]]
; CHECK: [[XB:%.+]] = add i32 [[SBASE]], 768
; CHECK-NEXT: [[XFB:%.+]] = freeze i32 [[XB]]
; CHECK: [[XPA:%.+]] = inttoptr i32 [[XFA]] to ptr addrspace(3)
; CHECK-NEXT: store i32 11, ptr addrspace(3) [[XPA]], align 1
; CHECK: [[XPB:%.+]] = inttoptr i32 [[XFB]] to ptr addrspace(3)
; CHECK-NEXT: store i32 22, ptr addrspace(3) [[XPB]], align 1
	ds_store_2addr_stride64_b32 v0, v1, v2 offset0:1 offset1:3
; CHECK: [[YA:%.+]] = add i32 [[SBASE]], 8
; CHECK-NEXT: [[YFA:%.+]] = freeze i32 [[YA]]
; CHECK: [[YB:%.+]] = add i32 [[SBASE]], 24
; CHECK-NEXT: [[YFB:%.+]] = freeze i32 [[YB]]
; CHECK: [[YPA:%.+]] = inttoptr i32 [[YFA]] to ptr addrspace(3)
; CHECK-NEXT: store i64 {{.+}}, ptr addrspace(3) [[YPA]], align 1
; CHECK: [[YPB:%.+]] = inttoptr i32 [[YFB]] to ptr addrspace(3)
; CHECK-NEXT: store i64 {{.+}}, ptr addrspace(3) [[YPB]], align 1
	ds_store_2addr_b64 v0, v[4:5], v[6:7] offset0:1 offset1:3
; CHECK: [[ZA:%.+]] = add i32 [[SBASE]], 512
; CHECK-NEXT: [[ZFA:%.+]] = freeze i32 [[ZA]]
; CHECK: [[ZB:%.+]] = add i32 [[SBASE]], 1536
; CHECK-NEXT: [[ZFB:%.+]] = freeze i32 [[ZB]]
; CHECK: [[ZPA:%.+]] = inttoptr i32 [[ZFA]] to ptr addrspace(3)
; CHECK-NEXT: store i64 {{.+}}, ptr addrspace(3) [[ZPA]], align 1
; CHECK: [[ZPB:%.+]] = inttoptr i32 [[ZFB]] to ptr addrspace(3)
; CHECK-NEXT: store i64 {{.+}}, ptr addrspace(3) [[ZPB]], align 1
	ds_store_2addr_stride64_b64 v0, v[4:5], v[6:7] offset0:1 offset1:3
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel ds_two_address_loads
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel ds_two_address_stores
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: ds_two_address_loads
    .symbol: ds_two_address_loads.kd
    .group_segment_fixed_size: 65536
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 16
    .wavefront_size: 32
  - .name: ds_two_address_stores
    .symbol: ds_two_address_stores.kd
    .group_segment_fixed_size: 65536
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
