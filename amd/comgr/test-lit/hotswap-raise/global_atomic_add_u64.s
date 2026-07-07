; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s
;
; gfx12 prints LLVM's GLOBAL_ATOMIC_ADD_X2 family as
; `global_atomic_add_u64`.  The SADDR form below is the no-return stream
; counter shape: VGPR32 byte/element offset, VGPR64 data, SGPR64 global base.
; The plain form additionally pins the RTN path by storing the returned old
; value.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_atomic_add_u64_kernel
	.p2align	8
	.type	global_atomic_add_u64_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @global_atomic_add_u64_kernel(
global_atomic_add_u64_kernel:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, 1
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, s3
; CHECK: %st_scaled_voff = mul i64 %{{[^,]+}}, 8
; CHECK: atomicrmw add ptr addrspace(1) %{{[^,]+}}, i64 %{{[^ ]+}} monotonic
	global_atomic_add_u64 v0, v[2:3], s[0:1] scale_offset scope:SCOPE_DEV
	v_mov_b32_e32 v4, s0
	v_mov_b32_e32 v5, s1
; CHECK: atomicrmw add ptr addrspace(1) %{{[^,]+}}, i64 %{{[^ ]+}} monotonic
	global_atomic_add_u64 v[6:7], v[4:5], v[2:3], off th:TH_ATOMIC_RETURN
	global_store_b64 v[4:5], v[6:7], off offset:8
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_atomic_add_u64_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
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
    .name:           global_atomic_add_u64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         global_atomic_add_u64_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
