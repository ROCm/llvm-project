; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=clusters_disabled | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=clusters_enabled 2>&1 | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	clusters_disabled
	.p2align	8
	.type	clusters_disabled,@function
; IR-LABEL: define amdgpu_kernel void @clusters_disabled(
clusters_disabled:
	s_endpgm

	.globl	clusters_enabled
	.p2align	8
	.type	clusters_enabled,@function
; REFUSE: unsupported-source-cluster-dims
; REFUSE-SAME: in kernel 'clusters_enabled'
; REFUSE-SAME: declares cluster dimensions
clusters_enabled:
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel clusters_disabled
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel clusters_enabled
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .cluster_dims:   [0, 0, 0]
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           clusters_disabled
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         clusters_disabled.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .cluster_dims:   [2, 1, 1]
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           clusters_enabled
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         clusters_enabled.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
