; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=global_plain | %FileCheck %s

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=global_coherent 2>&1 | %FileCheck %s --check-prefix=REFUSE
; REFUSE: in kernel 'global_coherent'
; REFUSE-SAME: non-default cache policy is not modeled

; Only a source that names its memory scope in a field of its own gets the
; volatile treatment. A pre-gfx12 source spells coherence through the cache
; bits instead, whose meaning this raiser does not establish, so an access
; carrying one is refused rather than raised at the wrong strength.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text

	.globl global_plain
	.p2align 8
	.type global_plain,@function
; CHECK-LABEL: define amdgpu_kernel void @global_plain(
global_plain:
; CHECK: load i32, ptr addrspace(1) {{%.+}}, align 4
	global_load_dword v1, v[2:3], off
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{%.+}}, align 4
	global_store_dword v[2:3], v1, off
; CHECK: ret void
	s_endpgm

	.globl global_coherent
	.p2align 8
	.type global_coherent,@function
global_coherent:
	global_load_dword v1, v[2:3], off sc0
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel global_plain
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel global_coherent
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: global_plain
    .symbol: global_plain.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 2
    .vgpr_count: 4
    .wavefront_size: 64
  - .name: global_coherent
    .symbol: global_coherent.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 2
    .vgpr_count: 4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
