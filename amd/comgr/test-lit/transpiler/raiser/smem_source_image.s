; REQUIRES: comgr-has-transpiler

; gfx1250 is the ISA the rest of the raiser fixtures assemble for, and the one
; that spells the 64-bit scalar arithmetic the PC-relative chains below use.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=pcrel_dword_kernel \
; RUN:   | %FileCheck %s --check-prefix=DWORD
; RUN: %transpile_cli %t.hsaco --emit-ir=pcrel_pair_kernel \
; RUN:   | %FileCheck %s --check-prefix=PAIR
; RUN: %transpile_cli %t.hsaco --emit-ir=pcrel_quad_kernel \
; RUN:   | %FileCheck %s --check-prefix=QUAD
; RUN: %transpile_cli %t.hsaco --emit-ir=pcrel_add_kernel \
; RUN:   | %FileCheck %s --check-prefix=ADD
; RUN: %transpile_cli %t.hsaco --emit-ir=pcrel_sub_kernel \
; RUN:   | %FileCheck %s --check-prefix=SUB

; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_dynamic_offset_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=DYNAMIC
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_outside_image_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=OUTSIDE
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_below_image_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=BELOW

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; s_get_pc_i64 names the address the next instruction sits at in the source
; code object, and a scalar load off it reads a literal the source compiled in
; rather than anything the raised kernel has mapped. Each kernel below folds
; one such load and hands the result to a vector add, which keeps the literal
; in the IR where a constant-folded scalar consumer would have absorbed it.

	.globl	pcrel_dword_kernel
	.p2align	8
	.type	pcrel_dword_kernel,@function
; DWORD-LABEL: define amdgpu_kernel void @pcrel_dword_kernel(
pcrel_dword_kernel:
	s_get_pc_i64 s[0:1]
; The capture sits four bytes into the kernel, and the table twenty bytes past
; that.
	s_load_b32 s2, s[0:1], 0x14
	s_wait_kmcnt 0x0
; DWORD: add i32 -559038737,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0xdeadbeef

	.globl	pcrel_pair_kernel
	.p2align	8
	.type	pcrel_pair_kernel,@function
; PAIR-LABEL: define amdgpu_kernel void @pcrel_pair_kernel(
pcrel_pair_kernel:
	s_get_pc_i64 s[0:1]
	s_load_b64 s[2:3], s[0:1], 0x18
	s_wait_kmcnt 0x0
; A pair is one 64-bit constant, which the register file splits back into the
; dwords the source wrote.
; PAIR: add i32 286331153,
	v_add_nc_u32 v0, s2, v0
; PAIR: add i32 572662306,
	v_add_nc_u32 v0, s3, v0
	s_endpgm
	.long	0x11111111
	.long	0x22222222

	.globl	pcrel_quad_kernel
	.p2align	8
	.type	pcrel_quad_kernel,@function
; QUAD-LABEL: define amdgpu_kernel void @pcrel_quad_kernel(
pcrel_quad_kernel:
	s_get_pc_i64 s[0:1]
; A load wider than a pair is one vector constant, which the register file
; distributes across the tuple.
	s_load_b128 s[4:7], s[0:1], 0x14
	s_wait_kmcnt 0x0
; QUAD: <4 x i32> <i32 1, i32 2, i32 3, i32 4>
	v_add_nc_u32 v0, s4, v0
	s_endpgm
	.long	1
	.long	2
	.long	3
	.long	4

	.globl	pcrel_add_kernel
	.p2align	8
	.type	pcrel_add_kernel,@function
; ADD-LABEL: define amdgpu_kernel void @pcrel_add_kernel(
pcrel_add_kernel:
	s_get_pc_i64 s[0:1]
; The captured address stays one wherever the constant sits, so the load still
; reads the table rather than target memory.
	s_add_nc_u64 s[0:1], 8, s[0:1]
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
; ADD: add i32 287454020,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0x11223344

	.globl	pcrel_sub_kernel
	.p2align	8
	.type	pcrel_sub_kernel,@function
; SUB-LABEL: define amdgpu_kernel void @pcrel_sub_kernel(
pcrel_sub_kernel:
	s_get_pc_i64 s[0:1]
; Subtracting a negative constant moves the captured address forward by eight.
	s_sub_nc_u64 s[0:1], s[0:1], -8
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
; SUB: add i32 -1430532899,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0xaabbccdd

; A source address the raise cannot resolve to a literal is refused, since
; letting the load through would read target memory at a source address.

	.globl	refuse_dynamic_offset_kernel
	.p2align	8
	.type	refuse_dynamic_offset_kernel,@function
refuse_dynamic_offset_kernel:
	s_get_pc_i64 s[0:1]
; DYNAMIC: unsupported-instruction-form: s_load_b32 {{.+}} :: reads the source code object at an offset only the running kernel knows
	s_load_b32 s2, s[0:1], s3 offset:0x0
	s_endpgm

	.globl	refuse_outside_image_kernel
	.p2align	8
	.type	refuse_outside_image_kernel,@function
refuse_outside_image_kernel:
	s_get_pc_i64 s[0:1]
	s_add_nc_u64 s[0:1], s[0:1], 0x100000
; OUTSIDE: unsupported-instruction-form: s_load_b32 {{.+}} :: reads a source address that no section of the source code object covers
	s_load_b32 s2, s[0:1], 0x0
	s_endpgm

	.globl	refuse_below_image_kernel
	.p2align	8
	.type	refuse_below_image_kernel,@function
refuse_below_image_kernel:
	s_get_pc_i64 s[0:1]
; BELOW: unsupported-instruction-form: s_sub_nc_u64 {{.+}} :: moves a source address before the start of the address space
	s_sub_nc_u64 s[0:1], s[0:1], 0x10000000
	s_load_b32 s2, s[0:1], 0x0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pcrel_dword_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_pair_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_quad_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_add_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_sub_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_dynamic_offset_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_outside_image_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_below_image_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
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
    .name:           pcrel_dword_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_dword_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_pair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_pair_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_quad_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_quad_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_add_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_add_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_sub_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_sub_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_dynamic_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_dynamic_offset_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_outside_image_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_outside_image_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_below_image_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_below_image_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
