; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=flat_atomic_f64_raw_refuse_kernel 2>&1 | %FileCheck %s --check-prefix=FLAT
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_atomic_f64_raw_refuse_kernel 2>&1 | %FileCheck %s --check-prefix=GLOBAL
;
; FLAT-DAG: kernel 'flat_atomic_f64_raw_refuse_kernel'
; FLAT-DAG: f64 atomic min/max from a pre-gfx12 source uses raw compare semantics
;
; GLOBAL-DAG: kernel 'global_atomic_f64_raw_refuse_kernel'
; GLOBAL-DAG: f64 atomic min/max from a pre-gfx12 source uses raw compare semantics

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	flat_atomic_f64_raw_refuse_kernel
	.p2align	8
	.type	flat_atomic_f64_raw_refuse_kernel,@function
flat_atomic_f64_raw_refuse_kernel:
	flat_atomic_max_f64 v[0:1], v[2:3]
	s_endpgm

	.globl	global_atomic_f64_raw_refuse_kernel
	.p2align	8
	.type	global_atomic_f64_raw_refuse_kernel,@function
global_atomic_f64_raw_refuse_kernel:
	global_atomic_min_f64 v[0:1], v[2:3], off
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flat_atomic_f64_raw_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_accum_offset 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel

	.amdhsa_kernel global_atomic_f64_raw_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_accum_offset 4
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
    .max_flat_workgroup_size: 1024
    .name:           flat_atomic_f64_raw_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         flat_atomic_f64_raw_refuse_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           global_atomic_f64_raw_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         global_atomic_f64_raw_refuse_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
