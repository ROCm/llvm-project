; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=cluster_dims_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ERR
;
; Explicit non-disabled source clusters require real TTMP6 cluster workgroup
; state. The current HotSwap ABI model only supports disabled source clusters,
; so the lift must refuse instead of seeding TTMP6 as singleton state.
;
; ERR: raise_cli: kernel 'cluster_dims_refuse_kernel' failed to raise: unsupported-source-cluster-dims: <source-cluster-dims> [unsupported-source-cluster-dims]
; ERR-SAME: .cluster_dims=[2,1,1] requires real TTMP6 cluster workgroup state

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cluster_dims_refuse_kernel
	.p2align	8
	.type	cluster_dims_refuse_kernel,@function
cluster_dims_refuse_kernel:
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cluster_dims_refuse_kernel
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
    .cluster_dims: [2, 1, 1]
    .max_flat_workgroup_size: 1024
    .name: cluster_dims_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: cluster_dims_refuse_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
