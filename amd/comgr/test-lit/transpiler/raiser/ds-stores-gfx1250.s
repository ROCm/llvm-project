; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=ds_stores,ds_store_exec | %FileCheck %s --check-prefixes=CHECK,EXEC

; A DS store narrows the source register to the encoded width and widens to a
; vector for the multi-register forms.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl ds_stores
	.p2align 8
	.type ds_stores,@function
; CHECK-LABEL: define amdgpu_kernel void @ds_stores(
ds_stores:
	v_mov_b32 v0, 4
	v_mov_b32 v1, 0x11223344
; CHECK: [[BASE:%.+]] = phi i32 [ 4, {{.+}}
; CHECK: add i32 [[BASE]], 1
; CHECK: [[V8:%.+]] = trunc i32 {{.+}} to i8
; CHECK: [[P8:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store i8 [[V8]], ptr addrspace(3) [[P8]], align 1
	ds_store_b8 v0, v1 offset:1
; CHECK: add i32 [[BASE]], 2
; CHECK: [[V16:%.+]] = trunc i32 {{.+}} to i16
; CHECK: [[P16:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store i16 [[V16]], ptr addrspace(3) [[P16]], align 1
	ds_store_b16 v0, v1 offset:2
; CHECK: add i32 [[BASE]], 4
; CHECK: [[P32:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store i32 287454020, ptr addrspace(3) [[P32]], align 1
	ds_store_b32 v0, v1 offset:4
; CHECK: add i32 [[BASE]], 8
; CHECK: [[P64:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store i64 {{.+}}, ptr addrspace(3) [[P64]], align 1
	ds_store_b64 v0, v[2:3] offset:8
; CHECK: add i32 [[BASE]], 16
; CHECK: [[P96:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store <3 x i32> {{.+}}, ptr addrspace(3) [[P96]], align 1
	ds_store_b96 v0, v[4:6] offset:16
; CHECK: add i32 [[BASE]], 32
; CHECK: [[P128:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store <4 x i32> {{.+}}, ptr addrspace(3) [[P128]], align 1
	ds_store_b128 v0, v[8:11] offset:32
; The D16_HI forms transfer bits 31:16 of the data register, so the byte an
; 8-bit store writes is 23:16 and not the low byte.
; CHECK: add i32 [[BASE]], 48
; CHECK: [[HI16:%.+]] = lshr i32 {{.+}}, 16
; CHECK-NEXT: [[V16HI:%.+]] = trunc i32 [[HI16]] to i16
; CHECK: [[P16HI:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store i16 [[V16HI]], ptr addrspace(3) [[P16HI]], align 1
	ds_store_b16_d16_hi v0, v1 offset:48
; CHECK: add i32 [[BASE]], 50
; CHECK: [[HI8:%.+]] = lshr i32 {{.+}}, 16
; CHECK-NEXT: [[V8HI:%.+]] = trunc i32 [[HI8]] to i8
; CHECK: [[P8HI:%.+]] = inttoptr i32 {{.+}} to ptr addrspace(3)
; CHECK-NEXT: store i8 [[V8HI]], ptr addrspace(3) [[P8HI]], align 1
	ds_store_b8_d16_hi v0, v1 offset:50
	s_endpgm

	.globl ds_store_exec
	.p2align 8
	.type ds_store_exec,@function
; EXEC-LABEL: define amdgpu_kernel void @ds_store_exec(
; EXEC: [[ENTRY_ACTIVE:%.+]] = call i1 @llvm.amdgcn.init.whole.wave()
ds_store_exec:
	v_mov_b32 v0, 4
	v_mov_b32 v1, 7
; An inactive lane must not reach memory, so the store sits inside the diamond
; while the value it writes is read outside.
	s_and_b32 exec_lo, exec_lo, 1
; EXEC: [[FROZEN:%.+]] = freeze i32 {{.+}}
; EXEC: [[MASK_ACTIVE:%.+]] = icmp ne i32 {{.+}}, 0
; EXEC-NEXT: [[ACTIVE:%.+]] = select i1 [[ENTRY_ACTIVE]], i1 [[MASK_ACTIVE]], i1 false
; EXEC-NEXT: br i1 [[ACTIVE]], label %[[DO:.+]], label %[[SKIP:.+]]
; EXEC: [[DO]]:
; EXEC-NEXT: [[PTR:%.+]] = inttoptr i32 [[FROZEN]] to ptr addrspace(3)
; EXEC-NEXT: store i32 {{.+}}, ptr addrspace(3) [[PTR]], align 1
; EXEC-NEXT: br label %[[SKIP]]
	ds_store_b32 v0, v1
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel ds_stores
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel ds_store_exec
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: ds_stores
    .symbol: ds_stores.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 12
    .wavefront_size: 32
  - .name: ds_store_exec
    .symbol: ds_store_exec.kd
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
