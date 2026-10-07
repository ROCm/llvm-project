; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --emit-ir=trap_kernel,debugtrap_kernel \
; RUN:   --target-isa=gfx1250 | %FileCheck %s --check-prefixes=TRAP,DEBUGTRAP
; RUN: %transpile_cli %t.hsaco --emit-ir=trap_kernel,debugtrap_kernel \
; RUN:   --target-isa=gfx1250 | %llc -mtriple=amdgpu12.50-amd-amdhsa \
; RUN:   | %FileCheck %s --check-prefix=ASM
; RUN: not %transpile_cli %t.hsaco --emit-ir=trap_kernel \
; RUN:   --target-isa=gfx942 2>&1 | %FileCheck %s --check-prefix=GFX942-TRAP
; RUN: not %transpile_cli %t.hsaco --emit-ir=debugtrap_kernel \
; RUN:   --target-isa=gfx942 2>&1 | %FileCheck %s --check-prefix=GFX942-DEBUGTRAP

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	trap_kernel
	.p2align	8
	.type	trap_kernel,@function
trap_kernel:
; TRAP-LABEL: define amdgpu_kernel void @trap_kernel(
; TRAP: call void @llvm.trap()
; TRAP-NEXT: unreachable
; ASM-LABEL: trap_kernel:
; ASM: s_trap 2
; GFX942-TRAP: unsupported-wave-projection: s_trap [SOPP]
; GFX942-TRAP-SAME: WaveNative does not support per-wave hardware side effects
	s_trap 0x12
	s_endpgm

	.globl	debugtrap_kernel
	.p2align	8
	.type	debugtrap_kernel,@function
debugtrap_kernel:
; DEBUGTRAP-LABEL: define amdgpu_kernel void @debugtrap_kernel(
; DEBUGTRAP: call void @llvm.debugtrap()
; DEBUGTRAP-NEXT: ret void
; ASM-LABEL: debugtrap_kernel:
; ASM: s_trap 3
; GFX942-DEBUGTRAP: unsupported-wave-projection: s_trap [SOPP]
; GFX942-DEBUGTRAP-SAME: WaveNative does not support per-wave hardware side effects
	s_trap 0x13
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel trap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel debugtrap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
    .name:           trap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         trap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           debugtrap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         debugtrap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
