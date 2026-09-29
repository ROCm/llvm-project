; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_loads | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_load_cache_policy,global_load_unaligned_offset \
; RUN:   --emit-ir=global_load_flat,global_load_scratch 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text

	.globl	global_loads
	.p2align	8
	.type	global_loads,@function
; IR-LABEL: define amdgpu_kernel void @global_loads(
global_loads:
; The address register pair holds the whole per-lane address.
; IR: [[POINTER0:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i32, ptr addrspace(1) [[POINTER0]], align 4
	global_load_dword v1, v[2:3], off

; gfx942 encodes the immediate offset in 13 signed bits.
; IR: [[POINTER1:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: [[OFFSET1:%.+]] = getelementptr i8, ptr addrspace(1) [[POINTER1]], i64 -16
; IR: load i32, ptr addrspace(1) [[OFFSET1]], align 4
	global_load_dword v1, v[2:3], off offset:-16

; The scalar base form adds a per-lane offset, unsigned before gfx1250.
; IR: [[BASE2:%.+]] = or i64 {{%.+}}, {{%.+}}
; IR: [[LANE2:%.+]] = zext i32 {{.+}} to i64
; IR: [[ADDRESS2:%.+]] = add i64 [[BASE2]], [[LANE2]]
; IR: [[POINTER2:%.+]] = inttoptr i64 [[ADDRESS2]] to ptr addrspace(1)
; IR: [[OFFSET2:%.+]] = getelementptr i8, ptr addrspace(1) [[POINTER2]], i64 32
; IR: load i32, ptr addrspace(1) [[OFFSET2]], align 4
	global_load_dword v1, v0, s[0:1] offset:32

; An accumulation register is as much a per-lane destination as a vector one.
; IR: [[POINTER3:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i32, ptr addrspace(1) [[POINTER3]], align 4
	global_load_dword a1, v[2:3], off

; IR: [[POINTER4:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: br i1 {{%.+}}, label %[[DO4:.+]], label %[[SKIP4:.+]]
; IR: [[DO4]]:
; IR: [[LOAD4:%.+]] = load i64, ptr addrspace(1) [[POINTER4]], align 4
; IR: lshr i64 [[LOAD4]], 32
; IR: br label %[[SKIP4]]
	global_load_dwordx2 v[4:5], v[2:3], off

; IR: [[BASE5:%.+]] = or i64 {{%.+}}, {{%.+}}
; IR: [[LANE5:%.+]] = zext i32 {{.+}} to i64
; IR: [[ADDRESS5:%.+]] = add i64 [[BASE5]], [[LANE5]]
; IR: [[POINTER5:%.+]] = inttoptr i64 [[ADDRESS5]] to ptr addrspace(1)
; IR: br i1 {{%.+}}, label {{%.+}}, label {{%.+}}
; IR: [[LOAD5:%.+]] = load <3 x i32>, ptr addrspace(1) [[POINTER5]], align 4
; IR: [[PACK5:%.+]] = bitcast <3 x i32> [[LOAD5]] to i96
; IR: lshr i96 [[PACK5]], 64
	global_load_dwordx3 v[8:10], v6, s[0:1]

; A four-byte offset does not strengthen the dword alignment guarantee.
; IR: [[POINTER6:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: [[OFFSET6:%.+]] = getelementptr i8, ptr addrspace(1) [[POINTER6]], i64 4
; IR: br i1 {{%.+}}, label {{%.+}}, label {{%.+}}
; IR: [[LOAD6:%.+]] = load <4 x i32>, ptr addrspace(1) [[OFFSET6]], align 4
; IR: [[PACK6:%.+]] = bitcast <4 x i32> [[LOAD6]] to i128
; IR: lshr i128 [[PACK6]], 96
	global_load_dwordx4 v[12:15], v[2:3], off offset:4
; IR: ret void
	s_endpgm

	.globl	global_load_cache_policy
	.p2align	8
	.type	global_load_cache_policy,@function
global_load_cache_policy:
; REFUSE:      in kernel 'global_load_cache_policy'
; REFUSE-SAME: non-default cache policy is not modeled
	global_load_dwordx3 v[4:6], v[2:3], off sc0
	s_endpgm

	.globl	global_load_unaligned_offset
	.p2align	8
	.type	global_load_unaligned_offset,@function
global_load_unaligned_offset:
; REFUSE:      in kernel 'global_load_unaligned_offset'
; REFUSE-SAME: immediate offset does not preserve the alignment of the access
	global_load_dword v1, v[2:3], off offset:1
	s_endpgm

	.globl	global_load_flat
	.p2align	8
	.type	global_load_flat,@function
global_load_flat:
; REFUSE:      in kernel 'global_load_flat'
; REFUSE-SAME: unsupported flat memory operation
	flat_load_dword v1, v[2:3]
	s_endpgm

	.globl	global_load_scratch
	.p2align	8
	.type	global_load_scratch,@function
global_load_scratch:
; REFUSE:      in kernel 'global_load_scratch'
; REFUSE-SAME: unsupported flat memory operation
	scratch_load_dword v1, v0, off
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_loads
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel global_load_cache_policy
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel global_load_unaligned_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel global_load_flat
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel global_load_scratch
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
    .name:           global_loads
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_loads.kd
    .vgpr_count:     16
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_cache_policy
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_cache_policy.kd
    .vgpr_count:     8
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_unaligned_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_unaligned_offset.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_flat
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_flat.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_scratch
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_scratch.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
