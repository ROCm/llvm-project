; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=smem_pc_relative_literal_kernel \
; RUN:   | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.p2align	8
.Lliteral:
	.long	0x12345678
	.long	0

	.globl	smem_pc_relative_literal_kernel
	.p2align	8
	.type	smem_pc_relative_literal_kernel,@function
smem_pc_relative_literal_kernel:
; CHECK-LABEL: define amdgpu_kernel void @smem_pc_relative_literal_kernel(
; CHECK-NOT: smem_load
; CHECK-NOT: inttoptr i64 {{.*}} to ptr addrspace(1)
	s_get_pc_i64 s[4:5]
.Lafter_getpc:
	s_add_nc_u64 s[4:5], s[4:5], .Lliteral-.Lafter_getpc
	s_load_b32 s6, s[4:5], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, 0
	global_store_b32 v[0:1], v0, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_pc_relative_literal_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 256
    .name:           smem_pc_relative_literal_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         smem_pc_relative_literal_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
