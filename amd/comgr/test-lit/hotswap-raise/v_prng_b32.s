; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir=v_prng_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefixes=CHECK,SAME
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_prng_b32_kernel \
; RUN:     2>/dev/null \
; RUN:   | %FileCheck %s --check-prefixes=CHECK,CROSS

; v_prng_b32 intrinsic keep (same-target) vs xorshift lowering (cross-target).
; CHECK-LABEL: define amdgpu_kernel void @v_prng_b32_kernel(
; SAME: %prng_b32{{[0-9]*}} = call i32 @llvm.amdgcn.prng.b32(i32 %{{[^)]+}})
; SAME-NOT: select i1
; CROSS: %prng_shl{{[0-9]*}} = shl i32 %{{[^,]+}}, 1
; CROSS: %prng_neg{{[0-9]*}} = icmp slt i32 %{{[^,]+}}, 0
; CROSS: %prng_tap{{[0-9]*}} = select i1 %prng_neg{{[0-9]*}}, i32 197, i32 0
; CROSS: %prng_b32{{[0-9]*}} = xor i32 %prng_shl{{[0-9]*}}, %prng_tap{{[0-9]*}}
; CROSS-NOT: call i32 @llvm.amdgcn.prng.b32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	v_prng_b32_kernel
	.p2align	8
	.type	v_prng_b32_kernel,@function
v_prng_b32_kernel:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s2
	v_prng_b32 v0, v1
	v_prng_b32_e64 v2, v1
	v_mov_b32_e32 v3, 0
	global_store_dword v3, v0, s[0:1]
	global_store_dword v3, v2, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_prng_b32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           8, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_prng_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_prng_b32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
