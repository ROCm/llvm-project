; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx950 --emit-ir > %t.ll
; RUN: %FileCheck %s --input-file=%t.ll \
; RUN:   --implicit-check-not="load {{.+}}, ptr addrspace(3)"
; RUN: %clang --target=amdgpu9.50-amd-amdhsa -nogpulib \
; RUN:   -x ir -O2 -S -emit-llvm %t.ll -o %t.opt.ll
; RUN: %clang --target=amdgpu9.50-amd-amdhsa -nogpulib \
; RUN:   -x ir -O2 -c %t.opt.ll -o %t.target.o
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir \
; RUN:   | %FileCheck %s
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir \
; RUN:   > %t.same.ll
; RUN: %clang --target=amdgpu12.50-amd-amdhsa -nogpulib \
; RUN:   -x ir -O2 -c %t.same.ll -o %t.same.o
; RUN: %llvm-mc -triple=amdgpu13.10-amd-amdhsa -filetype=obj %s -o %t.gfx13.o
; RUN: %ld.lld -shared %t.gfx13.o -o %t.gfx13.hsaco
; RUN: not %transpile_cli %t.gfx13.hsaco --target-isa=gfx950 --emit-ir 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
; REFUSE: unsupported-instruction-form: ds_load_tr8_b64 [DS]
; REFUSE-SAME: in kernel 'tr8'
; REFUSE-SAME: DS transpose loads require a gfx1250 wave32 source

	.amdhsa_code_object_version 6
	.text

	.globl tr8
	.p2align 8
	.type tr8,@function
; CHECK-LABEL: define amdgpu_kernel void @tr8(
tr8:
	s_load_b64 s[2:3], s[0:1], 0
; CHECK: [[MASK:%.+]] = load i32, ptr addrspace(1)
	s_load_b32 s4, s[0:1], 8
	s_wait_kmcnt 0
	s_mov_b32 s5, 0
	s_mov_b32 s6, 1
.Linit8:
	s_mov_b32 exec_lo, s6
	v_mov_b32 v12, s5
	s_add_u32 s5, s5, 8
	s_lshl_b32 s6, s6, 1
	s_cbranch_scc1 .Linit8
	s_mov_b32 exec_lo, -1
	v_mov_b32 v0, 11
; Zero EXEC leaves every address outside allocated LDS.
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s5, 0x80000000, 0
	v_add_nc_u32 v1, s5, v12
	s_mov_b32 exec_lo, s4
; CHECK: [[ADDR:%.+]] = add i32 [[OLD:%.+]], 7
; CHECK-NEXT: [[FROZEN:%.+]] = freeze i32 [[ADDR]]
; CHECK-NEXT: [[ELEMENT:%.+]] = and i32 [[LANE:%.+]], 7
; CHECK-NEXT: [[OFFSET:%.+]] = mul i32 [[ELEMENT]], 1
; CHECK-NEXT: [[GROUP:%.+]] = and i32 [[LANE]], -16
; CHECK-NEXT: [[HALF:%.+]] = lshr i32 [[LANE]], 1
; CHECK-NEXT: [[START:%.+]] = and i32 [[HALF]], 4
; CHECK-NEXT: [[BASE:%.+]] = or i32 [[GROUP]], [[START]]
; CHECK-NEXT: [[ZERO:%.+]] = icmp eq i32 [[MASK]], 0
; CHECK-NEXT: [[NONZERO:%.+]] = xor i1 [[ZERO]], true
; CHECK-NEXT: br i1 [[NONZERO]], label %[[DO:.+]], label %[[SKIP:.+]]
; CHECK: [[DO]]:
; CHECK-NEXT: [[SOURCE:%.+]] = add i32 [[BASE]], 0
; CHECK-NEXT: [[INDEX:%.+]] = mul i32 [[SOURCE]], 4
; CHECK-NEXT: [[GATHER:%.+]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[INDEX]], i32 [[FROZEN]])
; CHECK-NEXT: [[BYTE:%.+]] = add i32 [[GATHER]], [[OFFSET]]
; CHECK-NEXT: [[PTR:%.+]] = inttoptr i32 [[BYTE]] to ptr addrspace(3)
; CHECK-NEXT: load i8, ptr addrspace(3) [[PTR]], align 1
; CHECK: add i32 [[BASE]], 1
; CHECK: load i8, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 2
; CHECK: load i8, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 3
; CHECK: load i8, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 8
; CHECK: load i8, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 9
; CHECK: load i8, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 10
; CHECK: load i8, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 11
; CHECK: load i8, ptr addrspace(3) {{%.+}}, align 1
; CHECK: [[BITS:%.+]] = bitcast <2 x i32> {{%.+}} to i64
; CHECK-NEXT: [[WORD0:%.+]] = trunc i64 [[BITS]] to i32
; CHECK-NEXT: [[SHIFT1:%.+]] = lshr i64 [[BITS]], 32
; CHECK-NEXT: [[WORD1:%.+]] = trunc i64 [[SHIFT1]] to i32
; CHECK-NEXT: br label %[[SKIP]]
; CHECK: [[SKIP]]:
; CHECK-NEXT: [[DEST0:%.+]] = phi i32 [ [[WORD0]], %[[DO]] ], [ 11, %{{.+}} ]
; CHECK-NEXT: [[DEST1:%.+]] = phi i32 [ [[WORD1]], %[[DO]] ], [ [[OLD]], %{{.+}} ]
	ds_load_tr8_b64 v[0:1], v1 offset:7
	s_wait_dscnt 0
	s_mov_b32 exec_lo, -1
; CHECK: store i32 [[DEST0]], ptr addrspace(1)
	global_store_b32 v12, v0, s[2:3] offset:0
; CHECK: store i32 [[DEST1]], ptr addrspace(1)
	global_store_b32 v12, v1, s[2:3] offset:4
	s_endpgm

	.globl tr16
	.p2align 8
	.type tr16,@function
; CHECK-LABEL: define amdgpu_kernel void @tr16(
tr16:
	s_load_b64 s[2:3], s[0:1], 0
; CHECK: [[MASK:%.+]] = load i32, ptr addrspace(1)
	s_load_b32 s4, s[0:1], 8
	s_wait_kmcnt 0
	s_mov_b32 s5, 0
	s_mov_b32 s6, 1
.Linit16:
	s_mov_b32 exec_lo, s6
	v_mov_b32 v12, s5
	s_add_u32 s5, s5, 16
	s_lshl_b32 s6, s6, 1
	s_cbranch_scc1 .Linit16
	s_mov_b32 exec_lo, -1
	v_mov_b32 v0, 11
; Zero EXEC leaves every address outside allocated LDS.
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s5, 0x80000000, 0
	v_add_nc_u32 v1, s5, v12
	v_mov_b32 v2, 33
	v_mov_b32 v3, 44
	s_mov_b32 exec_lo, s4
; CHECK: [[ADDR:%.+]] = add i32 [[OLD:%.+]], 7
; CHECK-NEXT: [[FROZEN:%.+]] = freeze i32 [[ADDR]]
; CHECK-NEXT: [[ELEMENT:%.+]] = and i32 [[LANE:%.+]], 7
; CHECK-NEXT: [[OFFSET:%.+]] = mul i32 [[ELEMENT]], 2
; CHECK-NEXT: [[BASE:%.+]] = and i32 [[LANE]], -8
; CHECK-NEXT: [[ZERO:%.+]] = icmp eq i32 [[MASK]], 0
; CHECK-NEXT: [[NONZERO:%.+]] = xor i1 [[ZERO]], true
; CHECK-NEXT: br i1 [[NONZERO]], label %[[DO:.+]], label %[[SKIP:.+]]
; CHECK: [[DO]]:
; CHECK-NEXT: [[SOURCE:%.+]] = add i32 [[BASE]], 0
; CHECK-NEXT: [[INDEX:%.+]] = mul i32 [[SOURCE]], 4
; CHECK-NEXT: [[GATHER:%.+]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[INDEX]], i32 [[FROZEN]])
; CHECK-NEXT: [[BYTE:%.+]] = add i32 [[GATHER]], [[OFFSET]]
; CHECK-NEXT: [[PTR:%.+]] = inttoptr i32 [[BYTE]] to ptr addrspace(3)
; CHECK-NEXT: load i16, ptr addrspace(3) [[PTR]], align 1
; CHECK: add i32 [[BASE]], 1
; CHECK: load i16, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 2
; CHECK: load i16, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 3
; CHECK: load i16, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 4
; CHECK: load i16, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 5
; CHECK: load i16, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 6
; CHECK: load i16, ptr addrspace(3) {{%.+}}, align 1
; CHECK: add i32 [[BASE]], 7
; CHECK: load i16, ptr addrspace(3) {{%.+}}, align 1
; CHECK: [[BITS:%.+]] = bitcast <4 x i32> {{%.+}} to i128
; CHECK-NEXT: [[WORD0:%.+]] = trunc i128 [[BITS]] to i32
; CHECK-NEXT: [[SHIFT1:%.+]] = lshr i128 [[BITS]], 32
; CHECK-NEXT: [[WORD1:%.+]] = trunc i128 [[SHIFT1]] to i32
; CHECK-NEXT: [[SHIFT2:%.+]] = lshr i128 [[BITS]], 64
; CHECK-NEXT: [[WORD2:%.+]] = trunc i128 [[SHIFT2]] to i32
; CHECK-NEXT: [[SHIFT3:%.+]] = lshr i128 [[BITS]], 96
; CHECK-NEXT: [[WORD3:%.+]] = trunc i128 [[SHIFT3]] to i32
; CHECK-NEXT: br label %[[SKIP]]
; CHECK: [[SKIP]]:
; CHECK-NEXT: [[DEST0:%.+]] = phi i32 [ [[WORD0]], %[[DO]] ], [ 11, %{{.+}} ]
; CHECK-NEXT: [[DEST1:%.+]] = phi i32 [ [[WORD1]], %[[DO]] ], [ [[OLD]], %{{.+}} ]
; CHECK-NEXT: [[DEST2:%.+]] = phi i32 [ [[WORD2]], %[[DO]] ], [ 33, %{{.+}} ]
; CHECK-NEXT: [[DEST3:%.+]] = phi i32 [ [[WORD3]], %[[DO]] ], [ 44, %{{.+}} ]
	ds_load_tr16_b128 v[0:3], v1 offset:7
	s_wait_dscnt 0
	s_mov_b32 exec_lo, -1
; CHECK: store i32 [[DEST0]], ptr addrspace(1)
	global_store_b32 v12, v0, s[2:3] offset:0
; CHECK: store i32 [[DEST1]], ptr addrspace(1)
	global_store_b32 v12, v1, s[2:3] offset:4
; CHECK: store i32 [[DEST2]], ptr addrspace(1)
	global_store_b32 v12, v2, s[2:3] offset:8
; CHECK: store i32 [[DEST3]], ptr addrspace(1)
	global_store_b32 v12, v3, s[2:3] offset:12
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel tr8
		.amdhsa_group_segment_fixed_size 528
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 13
		.amdhsa_next_free_sgpr 7
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdhsa_kernel tr16
		.amdhsa_group_segment_fixed_size 528
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 13
		.amdhsa_next_free_sgpr 7
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: tr8
    .symbol: tr8.kd
    .group_segment_fixed_size: 528
    .kernarg_segment_size: 12
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .sgpr_count: 7
    .vgpr_count: 13
    .wavefront_size: 32
  - .name: tr16
    .symbol: tr16.kd
    .group_segment_fixed_size: 528
    .kernarg_segment_size: 12
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .sgpr_count: 7
    .vgpr_count: 13
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
