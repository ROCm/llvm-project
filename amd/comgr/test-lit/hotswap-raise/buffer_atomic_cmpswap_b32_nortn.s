; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_cmpswap_b32_nortn_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Raw-buffer atomic cmpswap b32 (no-return) lift.
; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_cmpswap_b32_nortn_kernel(
; CHECK: call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap
; CHECK-NOT: cmpxchg
; CHECK-NOT: atomicrmw xchg

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_cmpswap_b32_nortn_kernel
	.p2align	8
	.type	buffer_atomic_cmpswap_b32_nortn_kernel,@function
buffer_atomic_cmpswap_b32_nortn_kernel: ; @buffer_atomic_cmpswap_b32_nortn_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[2:3]
	s_mov_b32 s4, s0
	s_mov_b32 s5, s1
	buffer_atomic_cmpswap_b32 v[0:1], v2, s[4:7], null offen scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_cmpswap_b32_nortn_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         12
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_cmpswap_b32_nortn_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         buffer_atomic_cmpswap_b32_nortn_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
