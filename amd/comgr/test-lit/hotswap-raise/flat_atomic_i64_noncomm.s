; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=flat_atomic_i64_noncomm_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
; RUN: %raise_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=flat_atomic_i64_noncomm_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME
;
; 64-bit FLAT swap/cmpxchg are valid same-wave translations, but remain
; non-commutative cross-wave hazards under the existing Class-3 obstruction
; policy.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	flat_atomic_i64_noncomm_kernel
	.p2align	8
	.type	flat_atomic_i64_noncomm_kernel,@function
; REFUSE: cross-wave-replica-race: flat_atomic_swap_b64
; SAME-LABEL: define amdgpu_kernel void @flat_atomic_i64_noncomm_kernel(
flat_atomic_i64_noncomm_kernel:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, s3
	v_mov_b32_e32 v4, s0
	v_mov_b32_e32 v5, s1
	v_mov_b32_e32 v6, s2
	v_mov_b32_e32 v7, s3
; SAME: atomicrmw xchg ptr {{.*}}, i64 %{{[^ ]+}} seq_cst
	flat_atomic_swap_b64 v[8:9], v[0:1], v[2:3] th:TH_ATOMIC_RETURN
; SAME: cmpxchg ptr {{.*}} i64 %{{[^,]+}}, i64 %{{[^ ]+}} seq_cst seq_cst
	flat_atomic_cmpswap_b64 v[10:11], v[0:1], v[4:7] th:TH_ATOMIC_RETURN
	global_store_b64 v[0:1], v[8:9], off offset:8
	global_store_b64 v[0:1], v[10:11], off offset:16
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flat_atomic_i64_noncomm_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 12
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
    .name:           flat_atomic_i64_noncomm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         flat_atomic_i64_noncomm_kernel.kd
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
