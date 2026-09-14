; REQUIRES: comgr-has-hotswap-transpile
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=ds_widths,ds_exec_overlap > %t.ll
; RUN: %FileCheck %s --check-prefixes=IR,EXEC --input-file=%t.ll \
; RUN:   --implicit-check-not="load {{.+}}, ptr addrspace(3)"
; RUN: %clang --target=amdgpu9.50-amd-amdhsa -nogpulib \
; RUN:   -x ir -O2 -S -emit-llvm %t.ll -o %t.opt.ll
; RUN: %FileCheck %s --check-prefix=OPT --input-file=%t.opt.ll
; RUN: %clang --target=amdgpu9.50-amd-amdhsa -nogpulib \
; RUN:   -x ir -O2 -c %t.opt.ll -o %t.target.o
; RUN: %llvm-readelf --notes %t.target.o | %FileCheck %s --check-prefix=META
; META: .group_segment_fixed_size: 65568
; META: .name:           ds_widths
; META: .group_segment_fixed_size: 256
; META: .name:           ds_exec_overlap
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=ds_exec_overlap | %FileCheck %s --check-prefix=EXEC \
; RUN:   --implicit-check-not="load {{.+}}, ptr addrspace(3)"
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=ds_high_address | %FileCheck %s --check-prefix=HIGH
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=ds_transposed,ds_two_addresses 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl ds_widths
	.p2align 8
	.type ds_widths,@function
; OPT-LABEL: define amdgpu_kernel void @ds_widths(
; IR-LABEL: define amdgpu_kernel void @ds_widths(
ds_widths:
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	v_mov_b32 v20, s2
	v_mov_b32 v21, s3
	v_mov_b32 v0, 4
; IR: [[CARRY32_ADDR:%.+]] = add i32 4, 65532
; IR-NEXT: [[CARRY32_FROZEN:%.+]] = freeze i32 [[CARRY32_ADDR]]
; IR: [[CARRY32_PTR:%.+]] = inttoptr i32 [[CARRY32_FROZEN]] to ptr addrspace(3)
; IR-NEXT: load i32, ptr addrspace(3) [[CARRY32_PTR]], align 1
; OPT: load i32, ptr addrspace(3) inttoptr (i32 65536 to ptr addrspace(3))
	ds_load_b32 v1, v0 offset:65532
; IR: [[CARRY64_ADDR:%.+]] = add i32 4, 65532
; IR-NEXT: [[CARRY64_FROZEN:%.+]] = freeze i32 [[CARRY64_ADDR]]
; IR: [[CARRY64_PTR:%.+]] = inttoptr i32 [[CARRY64_FROZEN]] to ptr addrspace(3)
; IR-NEXT: load i64, ptr addrspace(3) [[CARRY64_PTR]], align 1
; OPT: load i64, ptr addrspace(3) inttoptr (i32 65536 to ptr addrspace(3))
	ds_load_b64 v[2:3], v0 offset:65532
; IR: [[BYTE128_ADDR:%.+]] = add i32 4, 65533
; IR-NEXT: [[BYTE128_FROZEN:%.+]] = freeze i32 [[BYTE128_ADDR]]
; IR: [[BYTE128_PTR:%.+]] = inttoptr i32 [[BYTE128_FROZEN]] to ptr addrspace(3)
; IR-NEXT: load <4 x i32>, ptr addrspace(3) [[BYTE128_PTR]], align 1
; OPT: load i128, ptr addrspace(3) inttoptr (i32 65537 to ptr addrspace(3))
	ds_load_b128 v[4:7], v0 offset:65533
; IR: [[MAXOFFSET_ADDR:%.+]] = add i32 4, 65535
; IR-NEXT: [[MAXOFFSET_FROZEN:%.+]] = freeze i32 [[MAXOFFSET_ADDR]]
; IR: [[MAXOFFSET_PTR:%.+]] = inttoptr i32 [[MAXOFFSET_FROZEN]] to ptr addrspace(3)
; IR-NEXT: load i32, ptr addrspace(3) [[MAXOFFSET_PTR]], align 1
; OPT: load i32, ptr addrspace(3) inttoptr (i32 65539 to ptr addrspace(3))
	ds_load_b32 v11, v0 offset:65535
	s_wait_dscnt 0
	global_store_b32 v[20:21], v1, off offset:0
	global_store_b32 v[20:21], v2, off offset:4
	global_store_b32 v[20:21], v3, off offset:8
	global_store_b32 v[20:21], v4, off offset:12
	global_store_b32 v[20:21], v5, off offset:16
	global_store_b32 v[20:21], v6, off offset:20
	global_store_b32 v[20:21], v7, off offset:24
	global_store_b32 v[20:21], v11, off offset:28
	s_endpgm

	.globl ds_exec_overlap
	.p2align 8
	.type ds_exec_overlap,@function
; EXEC-LABEL: define amdgpu_kernel void @ds_exec_overlap(
ds_exec_overlap:
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	v_mov_b32 v20, s2
	v_mov_b32 v21, s3
	v_mov_b32 v4, -1
	v_mov_b32 v5, 22
	v_mov_b32 v6, 33
	v_mov_b32 v7, 44
	v_mov_b32 v8, 55
	v_mov_b32 v9, -1
	v_mov_b32 v10, -1
; Inactive lanes retain invalid addresses and distinct destination values.
; EXEC: [[EXEC:%.+]] = lshr i32 1, {{%.+}}
; EXEC-NEXT: [[BIT:%.+]] = and i32 [[EXEC]], 1
; EXEC-NEXT: [[ACTIVE:%.+]] = icmp ne i32 [[BIT]], 0
	s_mov_b32 exec_lo, 1
	v_mov_b32 v4, 32
; EXEC: [[OLD128:%.+]] = phi i32 [ 32, %{{.+}} ], [ -1, %{{.+}} ]
; EXEC-NEXT: [[ADDR128:%.+]] = add i32 [[OLD128]], 0
; EXEC-NEXT: [[FROZEN128:%.+]] = freeze i32 [[ADDR128]]
; EXEC-NEXT: br i1 [[ACTIVE]], label %[[DO128:.+]], label %[[SKIP128:.+]]
; EXEC: [[DO128]]:
; EXEC-NEXT: [[PTR128:%.+]] = inttoptr i32 [[FROZEN128]] to ptr addrspace(3)
; EXEC-NEXT: [[LOAD128:%.+]] = load <4 x i32>, ptr addrspace(3) [[PTR128]], align 1
; EXEC-NEXT: [[BITS128:%.+]] = bitcast <4 x i32> [[LOAD128]] to i128
; EXEC-NEXT: [[WORD128_0:%.+]] = trunc i128 [[BITS128]] to i32
; EXEC-NEXT: [[SHIFT128_1:%.+]] = lshr i128 [[BITS128]], 32
; EXEC-NEXT: [[WORD128_1:%.+]] = trunc i128 [[SHIFT128_1]] to i32
; EXEC-NEXT: [[SHIFT128_2:%.+]] = lshr i128 [[BITS128]], 64
; EXEC-NEXT: [[WORD128_2:%.+]] = trunc i128 [[SHIFT128_2]] to i32
; EXEC-NEXT: [[SHIFT128_3:%.+]] = lshr i128 [[BITS128]], 96
; EXEC-NEXT: [[WORD128_3:%.+]] = trunc i128 [[SHIFT128_3]] to i32
; EXEC-NEXT: br label %[[SKIP128]]
; EXEC: [[SKIP128]]:
; EXEC-NEXT: [[DEST128_0:%.+]] = phi i32 [ [[WORD128_0]], %[[DO128]] ], [ [[OLD128]], %{{.+}} ]
; EXEC-NEXT: [[DEST128_1:%.+]] = phi i32 [ [[WORD128_1]], %[[DO128]] ], [ 22, %{{.+}} ]
; EXEC-NEXT: [[DEST128_2:%.+]] = phi i32 [ [[WORD128_2]], %[[DO128]] ], [ 33, %{{.+}} ]
; EXEC-NEXT: [[DEST128_3:%.+]] = phi i32 [ [[WORD128_3]], %[[DO128]] ], [ 44, %{{.+}} ]
	ds_load_b128 v[4:7], v4
	v_mov_b32 v9, 48
; EXEC: [[OLD64:%.+]] = phi i32 [ 48, %{{.+}} ], [ -1, %{{.+}} ]
; EXEC-NEXT: [[ADDR64:%.+]] = add i32 [[OLD64]], 0
; EXEC-NEXT: [[FROZEN64:%.+]] = freeze i32 [[ADDR64]]
; EXEC-NEXT: br i1 [[ACTIVE]], label %[[DO64:.+]], label %[[SKIP64:.+]]
; EXEC: [[DO64]]:
; EXEC-NEXT: [[PTR64:%.+]] = inttoptr i32 [[FROZEN64]] to ptr addrspace(3)
; EXEC-NEXT: [[LOAD64:%.+]] = load i64, ptr addrspace(3) [[PTR64]], align 1
; EXEC-NEXT: [[WORD64_0:%.+]] = trunc i64 [[LOAD64]] to i32
; EXEC-NEXT: [[SHIFT64_1:%.+]] = lshr i64 [[LOAD64]], 32
; EXEC-NEXT: [[WORD64_1:%.+]] = trunc i64 [[SHIFT64_1]] to i32
; EXEC-NEXT: br label %[[SKIP64]]
; EXEC: [[SKIP64]]:
; EXEC-NEXT: [[DEST64_0:%.+]] = phi i32 [ [[WORD64_0]], %[[DO64]] ], [ 55, %{{.+}} ]
; EXEC-NEXT: [[DEST64_1:%.+]] = phi i32 [ [[WORD64_1]], %[[DO64]] ], [ [[OLD64]], %{{.+}} ]
	ds_load_b64 v[8:9], v9
	v_mov_b32 v10, 64
; EXEC: [[OLD32:%.+]] = phi i32 [ 64, %{{.+}} ], [ -1, %{{.+}} ]
; EXEC-NEXT: [[ADDR32:%.+]] = add i32 [[OLD32]], 0
; EXEC-NEXT: [[FROZEN32:%.+]] = freeze i32 [[ADDR32]]
; EXEC-NEXT: br i1 [[ACTIVE]], label %[[DO32:.+]], label %[[SKIP32:.+]]
; EXEC: [[DO32]]:
; EXEC-NEXT: [[PTR32:%.+]] = inttoptr i32 [[FROZEN32]] to ptr addrspace(3)
; EXEC-NEXT: [[LOAD32:%.+]] = load i32, ptr addrspace(3) [[PTR32]], align 1
; EXEC-NEXT: br label %[[SKIP32]]
; EXEC: [[SKIP32]]:
; EXEC-NEXT: [[DEST32_0:%.+]] = phi i32 [ [[LOAD32]], %[[DO32]] ], [ [[OLD32]], %{{.+}} ]
	ds_load_b32 v10, v10
	s_wait_dscnt 0
	s_mov_b32 exec_lo, -1
; EXEC: store i32 [[DEST128_0]], ptr addrspace(1)
	global_store_b32 v[20:21], v4, off offset:0
; EXEC: store i32 [[DEST128_1]], ptr addrspace(1)
	global_store_b32 v[20:21], v5, off offset:4
; EXEC: store i32 [[DEST128_2]], ptr addrspace(1)
	global_store_b32 v[20:21], v6, off offset:8
; EXEC: store i32 [[DEST128_3]], ptr addrspace(1)
	global_store_b32 v[20:21], v7, off offset:12
; EXEC: store i32 [[DEST64_0]], ptr addrspace(1)
	global_store_b32 v[20:21], v8, off offset:16
; EXEC: store i32 [[DEST64_1]], ptr addrspace(1)
	global_store_b32 v[20:21], v9, off offset:20
; EXEC: store i32 [[DEST32_0]], ptr addrspace(1)
	global_store_b32 v[20:21], v10, off offset:24
	s_endpgm

	.globl ds_high_address
	.p2align 8
	.type ds_high_address,@function
; HIGH-LABEL: define amdgpu_kernel void @ds_high_address(
ds_high_address:
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	v_mov_b32 v20, s2
	v_mov_b32 v21, s3
	v_mov_b32 v0, 0x40000
; HIGH: [[ADDR:%.+]] = add i32 262144, 0
; HIGH: [[PTR:%.+]] = inttoptr i32 [[ADDR]] to ptr addrspace(3)
; HIGH-NEXT: [[VALUE:%.+]] = load i32, ptr addrspace(3) [[PTR]], align 1
; HIGH: [[DEST:%.+]] = phi i32 [ [[VALUE]], %{{.+}} ], [ undef, %{{.+}} ]
	ds_load_b32 v1, v0
	s_wait_dscnt 0
; HIGH: store i32 [[DEST]], ptr addrspace(1)
	global_store_b32 v[20:21], v1, off offset:0
	s_endpgm

	.globl ds_transposed
	.p2align 8
	.type ds_transposed,@function
ds_transposed:
; REFUSE: unsupported-instruction-form: ds_load_tr8_b64 [DS]
; REFUSE-SAME: in kernel 'ds_transposed'
; REFUSE-SAME: unsupported DS operation
	ds_load_tr8_b64 v[2:3], v0
	s_endpgm

	.globl ds_two_addresses
	.p2align 8
	.type ds_two_addresses,@function
ds_two_addresses:
; REFUSE: unsupported-instruction-form: ds_load_2addr_b32 [DS]
; REFUSE-SAME: in kernel 'ds_two_addresses'
; REFUSE-SAME: unsupported DS operation
	ds_load_2addr_b32 v[2:3], v0 offset0:1 offset1:2
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel ds_widths
		.amdhsa_group_segment_fixed_size 65568
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 4
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel ds_exec_overlap
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 4
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel ds_high_address
		.amdhsa_group_segment_fixed_size 327680
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 4
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel ds_transposed
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel ds_two_addresses
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: ds_widths
    .symbol: ds_widths.kd
    .group_segment_fixed_size: 65568
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 24
    .wavefront_size: 32
  - .name: ds_exec_overlap
    .symbol: ds_exec_overlap.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 24
    .wavefront_size: 32
  - .name: ds_high_address
    .symbol: ds_high_address.kd
    .group_segment_fixed_size: 327680
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 24
    .wavefront_size: 32
  - .name: ds_transposed
    .symbol: ds_transposed.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 0
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: ds_two_addresses
    .symbol: ds_two_addresses.kd
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
