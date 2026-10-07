; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_atomic_add | %FileCheck %s

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=UNMODELED_POLICY=1 %s -o %t.nv.o
; RUN: %ld.lld -shared %t.nv.o -o %t.nv.hsaco
; RUN: not %transpile_cli %t.nv.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_atomic_add 2>&1 | %FileCheck %s --check-prefix=POLICY
; POLICY: in kernel 'global_atomic_add'
; POLICY-SAME: non-default cache policy is not modeled

; The global integer atomic add lifts to an atomicrmw at its natural alignment.
; The temporal and scope hints the source carries only relax what a
; sequentially consistent atomic already guarantees, so they are dropped rather
; than refused.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	global_atomic_add
	.p2align	8
	.type	global_atomic_add,@function
; CHECK-LABEL: define amdgpu_kernel void @global_atomic_add(
global_atomic_add:
; The non-returning form takes a per-lane 64-bit address and reaches memory
; only for lanes active in EXEC.
; CHECK: [[FROZEN0:%.+]] = freeze i64 {{%.+}}
; CHECK: [[POINTER0:%.+]] = inttoptr i64 [[FROZEN0]] to ptr addrspace(1)
; CHECK: br i1 {{%.+}}, label %[[DO0:.+]], label %[[SKIP0:.+]]
; CHECK: [[DO0]]:
; CHECK: atomicrmw add ptr addrspace(1) [[POINTER0]], i32 {{.+}} seq_cst, align 4
; CHECK: br label %[[SKIP0]]
	global_atomic_add_u32 v[2:3], v1, off scope:SCOPE_DEV

; The SADDR form adds a signed per-lane offset to the scalar base, and folds
; the immediate byte offset on top of it.
; CHECK: [[BASE1:%.+]] = or i64 {{%.+}}, {{%.+}}
; CHECK: [[LANE1:%.+]] = sext i32 {{.+}} to i64
; CHECK-NEXT: [[ADDRESS1:%.+]] = add i64 [[BASE1]], [[LANE1]]
; CHECK-NEXT: [[FROZEN1:%.+]] = freeze i64 [[ADDRESS1]]
; CHECK-NEXT: [[POINTER1:%.+]] = inttoptr i64 [[FROZEN1]] to ptr addrspace(1)
; CHECK-NEXT: [[OFFSET1:%.+]] = getelementptr i8, ptr addrspace(1) [[POINTER1]], i64 64
; CHECK: [[OLD:%.+]] = atomicrmw add ptr addrspace(1) [[OFFSET1]], i32 {{.+}} seq_cst, align 4
	global_atomic_add_u32 v4, v0, v1, s[0:1] offset:64 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
	s_wait_loadcnt 0

; The returning form publishes the value the memory held before the add, which
; reaches the destination register and can be stored back.
; CHECK: [[DEST:%.+]] = phi i32 [ [[OLD]], {{.+}}
; CHECK: store i32 [[DEST]], ptr addrspace(1)
	global_store_b32 v[2:3], v4, off

; scale_offset multiplies the per-lane offset by the four bytes the atomic
; transfers.
; CHECK: [[BASE2:%.+]] = or i64 {{%.+}}, {{%.+}}
; CHECK: [[LANE2:%.+]] = sext i32 {{.+}} to i64
; CHECK-NEXT: [[SCALE2:%.+]] = mul i64 [[LANE2]], 4
; CHECK-NEXT: [[ADDRESS2:%.+]] = add i64 [[BASE2]], [[SCALE2]]
; CHECK-NEXT: [[FROZEN2:%.+]] = freeze i64 [[ADDRESS2]]
; CHECK-NEXT: [[POINTER2:%.+]] = inttoptr i64 [[FROZEN2]] to ptr addrspace(1)
; CHECK: atomicrmw add ptr addrspace(1) [[POINTER2]], i32 {{.+}} seq_cst, align 4
	global_atomic_add_u32 v5, v0, v1, s[0:1] scale_offset th:TH_ATOMIC_RETURN scope:SCOPE_DEV

; A system-scope hint on the non-returning SADDR form raises the same way.
; CHECK: atomicrmw add ptr addrspace(1) {{%.+}}, i32 {{.+}} seq_cst, align 4
	global_atomic_add_u32 v0, v1, s[0:1] scope:SCOPE_SYS

; CHECK: [[LANE3:%.+]] = sext i32 {{.+}} to i64
; CHECK-NEXT: [[SCALE3:%.+]] = mul i64 [[LANE3]], 4
; CHECK: atomicrmw add ptr addrspace(1) {{%.+}}, i32 {{.+}} seq_cst, align 4
	global_atomic_add_u32 v0, v1, s[0:1] scale_offset scope:SCOPE_DEV

.ifdef UNMODELED_POLICY
; The non-volatile bit is not a hint a sequentially consistent atomic subsumes.
	global_atomic_add_u32 v0, v1, s[0:1] nv
.endif
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_atomic_add
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 8
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
    .name:           global_atomic_add
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_atomic_add.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
