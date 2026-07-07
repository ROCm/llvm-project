; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_mov_exec_lo_shadow_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_mov/s_or_b32 exec_lo scalar EXEC-writer projected to wave-native ballot mask.
; CHECK-LABEL: define amdgpu_kernel void @s_mov_exec_lo_shadow_kernel(
; CHECK: %[[SAVED_I1:.*]] = icmp ne i64 %{{.*}}, 0
; CHECK: %wm_shadow_exec{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(i1 %[[SAVED_I1]])
; CHECK: %wave_mask_or{{[0-9]*}} = or i1 %{{.*}}, %[[SAVED_I1]]
; CHECK: %wave_mask_exec{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(i1 %wave_mask_or{{[0-9]*}})

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_mov_exec_lo_shadow_kernel
	.p2align	8
	.type	s_mov_exec_lo_shadow_kernel,@function
s_mov_exec_lo_shadow_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_cmpx_lt_u32_e64 v0, 48
	s_mov_b32 s2, exec_lo
	v_cmpx_lt_u32_e64 v0, 16
	s_or_b32 exec_lo, exec_lo, s2
	v_mov_b32_e32 v1, 1
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_mov_exec_lo_shadow_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 3
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           s_mov_exec_lo_shadow_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         s_mov_exec_lo_shadow_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
