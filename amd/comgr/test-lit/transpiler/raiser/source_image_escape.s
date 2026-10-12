; REQUIRES: comgr-has-transpiler

; The raiser fixtures assemble for gfx1250, which is also the ISA that spells
; the 64-bit scalar arithmetic these PC-relative chains use.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=overwrite_pair_kernel \
; RUN:   | %FileCheck %s

; A kernel that reads a source address is refused one kernel at a time,
; because the first refusal ends the run.

; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_overwrite_low_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=OVERWRITELOW
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_overwrite_high_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=OVERWRITEHIGH
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_movrels_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=MOVRELS
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_movrels_pair_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=MOVRELSPAIR
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_movrels_pair_high_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=MOVRELSPAIRHIGH
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_movrelsd_2_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=MOVRELSD2
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_buffer_descriptor_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=DESCRIPTOR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; s_get_pc_i64 marks a pair as holding a place in the source code object, which
; the running kernel has nothing mapped at. Every read of either half is
; refused unless the block has put an ordinary value there first, and a read
; that reaches the register file through a displaced index or through a buffer
; resource descriptor is refused on the same terms.

	.globl	refuse_overwrite_low_kernel
	.p2align	8
	.type	refuse_overwrite_low_kernel,@function
refuse_overwrite_low_kernel:
; Overwriting one half says nothing about the other, which still holds the half
; of the address the capture put there.
	s_get_pc_i64 s[0:1]
	s_mov_b32 s0, 0
; OVERWRITELOW: unsupported-instruction-form: v_add_nc_u32 {{.+}} :: operand-read: 's1' may hold a source code-object address
	v_add_nc_u32 v0, s1, v0
	s_endpgm

	.globl	refuse_overwrite_high_kernel
	.p2align	8
	.type	refuse_overwrite_high_kernel,@function
refuse_overwrite_high_kernel:
	s_get_pc_i64 s[0:1]
	s_mov_b32 s1, 0
; OVERWRITEHIGH: unsupported-instruction-form: v_add_nc_u32 {{.+}} :: operand-read: 's0' may hold a source code-object address
	v_add_nc_u32 v0, s0, v0
	s_endpgm

	.globl	overwrite_pair_kernel
	.p2align	8
	.type	overwrite_pair_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @overwrite_pair_kernel(
overwrite_pair_kernel:
; A block that has written both halves itself holds what it wrote there, so
; either half reads as the ordinary value it now carries.
	s_get_pc_i64 s[0:1]
	s_mov_b32 s0, 11
	s_mov_b32 s1, 22
; CHECK: add i32 11,
	v_add_nc_u32 v0, s0, v0
; CHECK: add i32 22,
	v_add_nc_u32 v0, s1, v0
	s_endpgm

	.globl	refuse_movrels_kernel
	.p2align	8
	.type	refuse_movrels_kernel,@function
refuse_movrels_kernel:
; s_movrels displaces its source index, so s7 with M0 = 10 reads s17, the half
; the capture wrote.
	s_get_pc_i64 s[16:17]
	s_mov_b32 m0, 10
; MOVRELS: unsupported-instruction-form: s_movrels_b32 {{.+}} :: operand-read: 's17' may hold a source code-object address
	s_movrels_b32 s5, s7
	s_endpgm

	.globl	refuse_movrels_pair_kernel
	.p2align	8
	.type	refuse_movrels_pair_kernel,@function
refuse_movrels_pair_kernel:
	s_get_pc_i64 s[16:17]
	s_mov_b32 m0, 8
; MOVRELSPAIR: unsupported-instruction-form: s_movrels_b64 {{.+}} :: operand-read: 's16' may hold a source code-object address
	s_movrels_b64 s[2:3], s[8:9]
	s_endpgm

	.globl	refuse_movrels_pair_high_kernel
	.p2align	8
	.type	refuse_movrels_pair_high_kernel,@function
refuse_movrels_pair_high_kernel:
; The low half holds an ordinary value this block wrote, so the 64-bit read is
; refused on the high half, which still holds what the capture put there.
	s_get_pc_i64 s[16:17]
	s_mov_b32 s16, 0
	s_mov_b32 m0, 8
; MOVRELSPAIRHIGH: unsupported-instruction-form: s_movrels_b64 {{.+}} :: operand-read: 's17' may hold a source code-object address
	s_movrels_b64 s[2:3], s[8:9]
	s_endpgm

	.globl	refuse_movrelsd_2_kernel
	.p2align	8
	.type	refuse_movrelsd_2_kernel,@function
refuse_movrelsd_2_kernel:
; s_movrelsd_2 displaces its source by M0[9:0], so s7 again reads s17.
	s_get_pc_i64 s[16:17]
	s_mov_b32 m0, 0x3000a
; MOVRELSD2: unsupported-instruction-form: s_movrelsd_2_b32 {{.+}} :: operand-read: 's17' may hold a source code-object address
	s_movrelsd_2_b32 s5, s7
	s_endpgm

	.globl	refuse_buffer_descriptor_kernel
	.p2align	8
	.type	refuse_buffer_descriptor_kernel,@function
refuse_buffer_descriptor_kernel:
; The descriptor is well formed apart from its base, which holds an address
; into the source image rather than a buffer the raised kernel can reach.
	s_get_pc_i64 s[4:5]
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0
	s_mov_b32 s10, 0
; DESCRIPTOR: unsupported-instruction-form: buffer_load_b32 {{.+}} :: operand-read: 's4' may hold a source code-object address
	buffer_load_b32 v2, v1, s[4:7], s10 offen
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6
	.amdhsa_kernel refuse_overwrite_low_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_overwrite_high_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel overwrite_pair_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_movrels_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_movrels_pair_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_movrels_pair_high_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_movrelsd_2_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_buffer_descriptor_kernel
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
    .name:           refuse_overwrite_low_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_overwrite_low_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_overwrite_high_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_overwrite_high_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           overwrite_pair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         overwrite_pair_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_movrels_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_movrels_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_movrels_pair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_movrels_pair_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_movrels_pair_high_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_movrels_pair_high_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_movrelsd_2_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_movrelsd_2_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_buffer_descriptor_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_buffer_descriptor_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
