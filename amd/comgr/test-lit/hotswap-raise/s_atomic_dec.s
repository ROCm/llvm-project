; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_atomic_dec_kernel 2>/dev/null | %FileCheck %s

; s_atomic_dec scalar atomic lowered to atomicrmw udec_wrap.
; CHECK-LABEL: define amdgpu_kernel void @s_atomic_dec_kernel(
; CHECK: atomicrmw udec_wrap ptr addrspace(1) {{.*}} monotonic, align 4
; CHECK-NOT: atomicrmw sub
; CHECK-NOT: atomicrmw add
; CHECK-NOT: call {{.*}}@llvm.amdgcn.s.atomic

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	s_atomic_dec_kernel
	.p2align	8
	.type	s_atomic_dec_kernel,@function
s_atomic_dec_kernel:                    ; @s_atomic_dec_kernel
; %bb.0:
	v_or_b32_e32 v0, s2, v0
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x10
	v_mov_b32_e32 v0, 0
	s_waitcnt lgkmcnt(0)
	s_atomic_dec s6, s[4:5], s7
	s_waitcnt lgkmcnt(0)
	
	s_cmp_eq_u32 s6, 1
	s_cselect_b64 s[0:1], -1, 0
	v_cndmask_b32_e64 v1, 0, 1, s[0:1]
	global_store_dword v0, v1, s[2:3]
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_atomic_dec_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:         12, .size:           4, .value_kind:     by_value }
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           s_atomic_dec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .symbol:         s_atomic_dec_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
