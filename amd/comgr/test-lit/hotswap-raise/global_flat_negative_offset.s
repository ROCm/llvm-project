; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_load_negative_offset_kernel,global_store_negative_offset_kernel,flat_load_saddr_negative_offset_kernel 2>/dev/null | %FileCheck %s

; global/flat load+store negative immediate offset: signed i64 GEP, no wraparound.
; CHECK-LABEL: define amdgpu_kernel void @global_load_negative_offset_kernel
; CHECK: getelementptr i8, ptr addrspace(1) %{{.*}}, i64 -19200
; CHECK-NOT: getelementptr i8, ptr addrspace(1) %{{.*}}, i64 16758016
; CHECK-LABEL: define amdgpu_kernel void @global_store_negative_offset_kernel
; CHECK: getelementptr i8, ptr addrspace(1) %{{.*}}, i64 -19200
; CHECK-NOT: getelementptr i8, ptr addrspace(1) %{{.*}}, i64 16758016
; CHECK-LABEL: define amdgpu_kernel void @flat_load_saddr_negative_offset_kernel
; CHECK: getelementptr i8, ptr addrspace(1) %{{.*}}, i64 -19200
; CHECK-NOT: getelementptr i8, ptr addrspace(1) %{{.*}}, i64 16758016

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_negative_offset_kernel
	.p2align	8
	.type	global_load_negative_offset_kernel,@function
global_load_negative_offset_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[2:3], s[0:1], 0x0
	s_load_b64 s[4:5], s[0:1], 0x10
	v_mov_b32_e32 v0, 0x5000
	s_wait_kmcnt 0x0
	global_load_b64 v[2:3], v0, s[2:3] offset:-19200
	s_wait_loadcnt 0x0
	global_store_b64 v0, v[2:3], s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_negative_offset_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

	.globl	global_store_negative_offset_kernel
	.p2align	8
	.type	global_store_negative_offset_kernel,@function
global_store_negative_offset_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[2:3], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x5000
	v_mov_b32_e32 v2, 0x3f800000
	s_wait_kmcnt 0x0
	global_store_b32 v0, v2, s[2:3] offset:-19200
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_store_negative_offset_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

	.globl	flat_load_saddr_negative_offset_kernel
	.p2align	8
	.type	flat_load_saddr_negative_offset_kernel,@function
flat_load_saddr_negative_offset_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[2:3], s[0:1], 0x0
	s_load_b64 s[4:5], s[0:1], 0x10
	v_mov_b32_e32 v0, 0x5000
	s_wait_kmcnt 0x0
	flat_load_b32 v2, v0, s[2:3] offset:-19200 scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	global_store_b32 v0, v2, s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flat_load_saddr_negative_offset_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .actual_access: read_only, .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .offset: 8, .size: 8, .value_kind: by_value }
      - { .actual_access: write_only, .address_space: global, .offset: 16, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name: global_load_negative_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 6
    .symbol: global_load_negative_offset_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
  - .args:
      - { .actual_access: write_only, .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: global_store_negative_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 4
    .symbol: global_store_negative_offset_kernel.kd
    .vgpr_count: 3
    .wavefront_size: 32
  - .args:
      - { .actual_access: read_only, .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .offset: 8, .size: 8, .value_kind: by_value }
      - { .actual_access: write_only, .address_space: global, .offset: 16, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name: flat_load_saddr_negative_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 6
    .symbol: flat_load_saddr_negative_offset_kernel.kd
    .vgpr_count: 3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
