; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=vcc_saddr_load_kernel,vcc_saddr_store_kernel 2>&1 | %FileCheck %s

; VCC used as a general-purpose 64-bit scalar SADDR base: the allocator computes
; a global address into the VCC pair (s_add_nc_u64 vcc, ...) and feeds it as the
; SADDR base of global_load/store. The raiser must read the raw 64-bit VCC value
; as a plain address, not the wave-mask ballot (a vcc_ballot would mean the
; address was fabricated from the i1 mask view -- a silent miscompile).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vcc_saddr_load_kernel
	.p2align	8
	.type	vcc_saddr_load_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @vcc_saddr_load_kernel
vcc_saddr_load_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[2:3], s[0:1], 0x0
	s_load_b64 s[4:5], s[0:1], 0x10
	v_mov_b32_e32 v1, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 vcc, s[2:3], 16
	; global_load with vcc as SADDR base -> raw 64-bit VCC value, not a ballot.
	; CHECK-NOT: vcc_ballot
	; CHECK: inttoptr i64 %{{.*}} to ptr addrspace(1)
	global_load_u16 v2, v1, vcc
	s_wait_loadcnt 0x0
	global_store_b16 v1, v2, s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vcc_saddr_load_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel

	.text
	.globl	vcc_saddr_store_kernel
	.p2align	8
	.type	vcc_saddr_store_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @vcc_saddr_store_kernel
vcc_saddr_store_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[2:3], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0x3c00
	s_wait_kmcnt 0x0
	s_add_nc_u64 vcc, s[2:3], 16
	; global_store with vcc as SADDR base -> raw 64-bit VCC value, not a ballot.
	; CHECK-NOT: vcc_ballot
	; CHECK: inttoptr i64 %{{.*}} to ptr addrspace(1)
	global_store_b16 v1, v2, vcc
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vcc_saddr_store_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .actual_access: read_only, .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .actual_access: write_only, .address_space: global, .offset: 16, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name: vcc_saddr_load_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 6
    .symbol: vcc_saddr_load_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
  - .args:
      - { .actual_access: write_only, .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: vcc_saddr_store_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 6
    .symbol: vcc_saddr_store_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
