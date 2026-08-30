; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s \
; RUN:   -o %t.gfx1250.o
; RUN: %ld.lld -shared %t.gfx1250.o -o %t.gfx1250.hsaco

; RUN: %hotswap_transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=smem_loads | %FileCheck %s --check-prefix=IR
; RUN: not %hotswap_transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=smem_register_offset 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=REGISTER-OFFSET
; RUN: not %hotswap_transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=smem_cache_policy 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=CACHE-POLICY
; RUN: not %hotswap_transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=smem_buffer_load 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=BUFFER
; RUN: not %hotswap_transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=smem_negative_offset 2>&1 | \
; RUN:   %FileCheck %s --check-prefix=NEGATIVE-OFFSET

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	smem_loads
	.p2align	8
	.type	smem_loads,@function
; IR-LABEL: define amdgpu_kernel void @smem_loads(
smem_loads:
; The i64 -4 mask clears the two low address bits.
; IR: [[BASE0:%.+]] = and i64 {{%.+}}, -4
; IR: [[ADDRESS0:%.+]] = add i64 [[BASE0]], 0
; IR: [[POINTER0:%.+]] = inttoptr i64 [[ADDRESS0]] to ptr addrspace(1)
; IR: [[LOAD128:%.+]] = load <4 x i32>, ptr addrspace(1) [[POINTER0]], align 4
	s_load_b128 s[4:7], s[0:1], 0x3
; IR: [[LOAD128_BITS:%.+]] = bitcast <4 x i32> [[LOAD128]] to i128
; IR: [[LOAD128_WORD2_SHIFTED:%.+]] = lshr i128 [[LOAD128_BITS]], 64
; IR: [[LOAD128_WORD2:%.+]] = trunc i128 [[LOAD128_WORD2_SHIFTED]] to i32
; IR: [[LOAD128_WORD3_SHIFTED:%.+]] = lshr i128 [[LOAD128_BITS]], 96
; IR: [[LOAD128_WORD3:%.+]] = trunc i128 [[LOAD128_WORD3_SHIFTED]] to i32
; IR: [[LOAD128_WORD2_EXT:%.+]] = zext i32 [[LOAD128_WORD2]] to i64
; IR: [[LOAD128_WORD3_EXT:%.+]] = zext i32 [[LOAD128_WORD3]] to i64
; IR: [[LOAD128_WORD3_BITS:%.+]] = shl i64 [[LOAD128_WORD3_EXT]], 32
; IR: [[BASE1_BITS:%.+]] = or i64 [[LOAD128_WORD2_EXT]], [[LOAD128_WORD3_BITS]]

; IR: [[BASE1:%.+]] = and i64 [[BASE1_BITS]], -4
; IR: [[ADDRESS1:%.+]] = add i64 [[BASE1]], 4
; IR: [[POINTER1:%.+]] = inttoptr i64 [[ADDRESS1]] to ptr addrspace(1)
; IR: [[LOAD64:%.+]] = load i64, ptr addrspace(1) [[POINTER1]], align 4
	s_load_b64 s[2:3], s[6:7], 0x7
; IR: [[LOAD64_LO:%.+]] = trunc i64 [[LOAD64]] to i32
; IR: [[LOAD64_SHIFTED:%.+]] = lshr i64 [[LOAD64]], 32
; IR: [[LOAD64_HI:%.+]] = trunc i64 [[LOAD64_SHIFTED]] to i32
; IR: [[LOAD64_LO_EXT:%.+]] = zext i32 [[LOAD64_LO]] to i64
; IR: [[LOAD64_HI_EXT:%.+]] = zext i32 [[LOAD64_HI]] to i64
; IR: [[LOAD64_HI_BITS:%.+]] = shl i64 [[LOAD64_HI_EXT]], 32
; IR: [[BASE2_BITS:%.+]] = or i64 [[LOAD64_LO_EXT]], [[LOAD64_HI_BITS]]

; IR: [[BASE2:%.+]] = and i64 [[BASE2_BITS]], -4
; IR: [[ADDRESS2:%.+]] = add i64 [[BASE2]], 8
; IR: [[POINTER2:%.+]] = inttoptr i64 [[ADDRESS2]] to ptr addrspace(1)
; IR: [[LOAD32:%.+]] = load i32, ptr addrspace(1) [[POINTER2]], align 4
	s_load_b32 s4, s[2:3], 0xb
; IR: [[LOAD32_EXT:%.+]] = zext i32 [[LOAD32]] to i64
; IR: [[BASE3_BITS:%.+]] = or i64 [[LOAD32_EXT]], {{%.+}}

; IR: [[BASE3:%.+]] = and i64 [[BASE3_BITS]], -4
; IR: [[ADDRESS3:%.+]] = add i64 [[BASE3]], 12
; IR: [[POINTER3:%.+]] = inttoptr i64 [[ADDRESS3]] to ptr addrspace(1)
; IR: load i32, ptr addrspace(1) [[POINTER3]], align 4
	s_load_b32 s8, s[4:5], 0xf
; IR: ret void
	s_endpgm

	.globl	smem_register_offset
	.p2align	8
	.type	smem_register_offset,@function
smem_register_offset:
; REGISTER-OFFSET: only immediate scalar load offsets are supported
	s_load_b32 s2, s[0:1], s4
	s_endpgm

	.globl	smem_cache_policy
	.p2align	8
	.type	smem_cache_policy,@function
smem_cache_policy:
; CACHE-POLICY: non-default scalar load modifiers are not supported
	s_load_b32 s2, s[0:1], 0x0 scope:SCOPE_SYS
	s_endpgm

	.globl	smem_buffer_load
	.p2align	8
	.type	smem_buffer_load,@function
smem_buffer_load:
; BUFFER: unsupported scalar memory operation
	s_buffer_load_b32 s4, s[0:3], 0x0
	s_endpgm

	.globl	smem_negative_offset
	.p2align	8
	.type	smem_negative_offset,@function
smem_negative_offset:
; NEGATIVE-OFFSET: negative scalar load offsets are not supported
	s_load_b32 s2, s[0:1], -4
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_loads
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 9
	.end_amdhsa_kernel
	.amdhsa_kernel smem_register_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel smem_cache_policy
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.amdhsa_kernel smem_buffer_load
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel smem_negative_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_loads
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .symbol:         smem_loads.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_register_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         smem_register_offset.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_cache_policy
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         smem_cache_policy.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_buffer_load
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         smem_buffer_load.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_negative_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         smem_negative_offset.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
