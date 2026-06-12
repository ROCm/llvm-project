; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=cluster_dims_overflow_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=OVERFLOW
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=cluster_dims_negative_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=NEGATIVE
;
; Malformed cluster dimensions must not be narrowed into valid-looking values.
; In particular, UINT32_MAX+1 would otherwise truncate to the disabled-cluster
; sentinel zero.
;
; OVERFLOW: raise_cli: kernel 'cluster_dims_overflow_refuse_kernel' metadata: hotswap: extractKernelMeta: kernel 'cluster_dims_overflow_refuse_kernel' has malformed .cluster_dims metadata
; NEGATIVE: raise_cli: kernel 'cluster_dims_negative_refuse_kernel' metadata: hotswap: extractKernelMeta: kernel 'cluster_dims_negative_refuse_kernel' has malformed .cluster_dims metadata

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cluster_dims_overflow_refuse_kernel
	.p2align	8
	.type	cluster_dims_overflow_refuse_kernel,@function
cluster_dims_overflow_refuse_kernel:
	s_endpgm

	.globl	cluster_dims_negative_refuse_kernel
	.p2align	8
	.type	cluster_dims_negative_refuse_kernel,@function
cluster_dims_negative_refuse_kernel:
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cluster_dims_overflow_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel cluster_dims_negative_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .cluster_dims: [4294967296, 0, 0]
    .max_flat_workgroup_size: 1024
    .name: cluster_dims_overflow_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: cluster_dims_overflow_refuse_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .cluster_dims: [-1, 0, 0]
    .max_flat_workgroup_size: 1024
    .name: cluster_dims_negative_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: cluster_dims_negative_refuse_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
