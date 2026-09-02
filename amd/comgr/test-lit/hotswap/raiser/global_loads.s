; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_loads | %FileCheck %s --check-prefix=IR
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_load_cache_policy 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=CACHE-POLICY
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_load_unaligned_offset 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=UNALIGNED-OFFSET
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_load_pair 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=PAIR
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_load_flat 2>&1 | %FileCheck %s --check-prefix=FLAT
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_load_scratch 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=SCRATCH

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
; IR: ret void
	s_endpgm

	.globl	global_load_cache_policy
	.p2align	8
	.type	global_load_cache_policy,@function
global_load_cache_policy:
; CACHE-POLICY: non-default cache policy is not modeled
	global_load_dword v1, v[2:3], off sc0
	s_endpgm

	.globl	global_load_unaligned_offset
	.p2align	8
	.type	global_load_unaligned_offset,@function
global_load_unaligned_offset:
; UNALIGNED-OFFSET: immediate offset does not preserve the alignment of the access
	global_load_dword v1, v[2:3], off offset:1
	s_endpgm

	.globl	global_load_pair
	.p2align	8
	.type	global_load_pair,@function
global_load_pair:
; PAIR: unsupported flat memory operation
	global_load_dwordx2 v[2:3], v[2:3], off
	s_endpgm

	.globl	global_load_flat
	.p2align	8
	.type	global_load_flat,@function
global_load_flat:
; FLAT: unsupported flat memory operation
	flat_load_dword v1, v[2:3]
	s_endpgm

	.globl	global_load_scratch
	.p2align	8
	.type	global_load_scratch,@function
global_load_scratch:
; SCRATCH: unsupported flat memory operation
	scratch_load_dword v1, v0, off
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_loads
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel global_load_cache_policy
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
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
	.amdhsa_kernel global_load_pair
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
    .vgpr_count:     8
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_cache_policy
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_cache_policy.kd
    .vgpr_count:     4
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
    .name:           global_load_pair
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_pair.kd
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
