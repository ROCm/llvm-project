; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir=s_bfe_i64_exec_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_bfe_i64 writing exec drives per-lane predication.
; CHECK-LABEL: define amdgpu_kernel void @s_bfe_i64_exec_kernel(
; CHECK: %sbfe_i64 = ashr i64 %{{[^,]+}}, 63
; CHECK: [[BFE_SEL:%[0-9]+]] = select i1 {{.*}}, i64 0, i64 %{{[^ ]+}}
; CHECK: %spe_exec_at_lane{{[0-9]*}} = lshr i64 [[BFE_SEL]], %spe_lane_mod{{[0-9]*}}
; CHECK: br i1 %spe_lane_active{{[0-9]*}}, label %[[DO:[^ ,]+]], label %{{[^ ,]+}}
; CHECK: [[DO]]:
; CHECK-NEXT: store i32 170, ptr addrspace(1) %{{[^ ]+}}, align 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	s_bfe_i64_exec_kernel
	.p2align	8
	.type	s_bfe_i64_exec_kernel,@function
s_bfe_i64_exec_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v1, 0xaa
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[4:5], s[0:1], 0, v[2:3]
	s_bfe_i64 exec, s[0:1], 0x10000
	global_store_dword v[4:5], v1, off
	s_waitcnt vmcnt(0)
	s_mov_b64 exec, -1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_bfe_i64_exec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 8
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
    .name:           s_bfe_i64_exec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_bfe_i64_exec_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
