; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=global_atomic_add | %FileCheck %s

; A pre-gfx12 source spells the returning form with the GLC bit and the cache
; hints with SLC and SCC, all of which a sequentially consistent atomic already
; subsumes. The returning form also comes in an AGPR flavour on this target.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text

	.globl global_atomic_add
	.p2align 8
	.type global_atomic_add,@function
; CHECK-LABEL: define amdgpu_kernel void @global_atomic_add(
global_atomic_add:
; CHECK: [[POINTER0:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; CHECK: atomicrmw add ptr addrspace(1) [[POINTER0]], i32 {{.+}} seq_cst, align 4
	global_atomic_add v[0:1], v2, off

; The temporal and system-coherence hints do not change what is emitted.
; CHECK: atomicrmw add ptr addrspace(1) {{%.+}}, i32 {{.+}} seq_cst, align 4
	global_atomic_add v[0:1], v2, off nt sc1

; The returning form publishes the value the memory held before the add.
; CHECK: [[OLD:%.+]] = atomicrmw add ptr addrspace(1) {{%.+}}, i32 {{.+}} seq_cst, align 4
	global_atomic_add v3, v[0:1], v2, off sc0
	s_waitcnt vmcnt(0)
; CHECK: [[DEST:%.+]] = phi i32 [ [[OLD]], {{.+}}
; CHECK: store i32 [[DEST]], ptr addrspace(1)
	global_store_dword v[0:1], v3, off

; The AGPR flavour of the returning form reaches the same lowering.
; CHECK: atomicrmw add ptr addrspace(1) {{%.+}}, i32 {{.+}} seq_cst, align 4
	global_atomic_add a0, v[0:1], a1, off sc0

; The SADDR form zero-extends the per-lane offset on this target and folds the
; immediate byte offset on top of it.
; CHECK: [[BASE:%.+]] = or i64 {{%.+}}, {{%.+}}
; CHECK: [[LANE:%.+]] = zext i32 {{.+}} to i64
; CHECK-NEXT: [[ADDRESS:%.+]] = add i64 [[BASE]], [[LANE]]
; CHECK-NEXT: [[POINTER:%.+]] = inttoptr i64 [[ADDRESS]] to ptr addrspace(1)
; CHECK-NEXT: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) [[POINTER]], i64 16
; CHECK: atomicrmw add ptr addrspace(1) [[OFFSET]], i32 {{.+}} seq_cst, align 4
	global_atomic_add v4, v2, v3, s[0:1] offset:16 sc0
; CHECK: ret void
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel global_atomic_add
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 8
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: global_atomic_add
    .symbol: global_atomic_add.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 2
    .vgpr_count: 8
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
