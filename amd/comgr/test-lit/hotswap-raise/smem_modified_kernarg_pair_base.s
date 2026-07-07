; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=smem_modified_kernarg_pair_base_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Modified kernarg-pointer pair base still resolves to kernarg.segment.ptr loads.
; CHECK-LABEL: define amdgpu_kernel void @smem_modified_kernarg_pair_base_kernel(
; CHECK-SAME: ptr addrspace(4) byref([16 x i8]) align 16 %kargs
; CHECK: call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
; CHECK: %smem_load = load i32, ptr addrspace(1) %{{[^,]+}}, align 4
; CHECK: %smem_load{{[0-9]*}} = load i32, ptr addrspace(1) %{{[^,]+}}, align 4
; CHECK: %smem_load{{[0-9]*}} = load i32, ptr addrspace(1) %{{[^,]+}}, align 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	smem_modified_kernarg_pair_base_kernel
	.p2align	8
	.type	smem_modified_kernarg_pair_base_kernel,@function
smem_modified_kernarg_pair_base_kernel:
; %bb.0:
	s_mov_b64 s[6:7], 0
	s_add_nc_u64 s[0:1], s[2:3], s[6:7]
	s_load_b32 s8, s[0:1], 0x0
	s_load_b64 s[10:11], s[0:1], 0x8
	s_wait_kmcnt 0x0
	s_add_u32 s8, s8, s10
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, s8
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_modified_kernarg_pair_base_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_kernarg_preload_length 4
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .actual_access: write_only, .address_space: global, .offset: 8, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name: smem_modified_kernarg_pair_base_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 12
    .symbol: smem_modified_kernarg_pair_base_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
