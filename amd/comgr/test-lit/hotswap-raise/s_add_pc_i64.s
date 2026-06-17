; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=add_pc_i64_short_kernel 2>/dev/null | %FileCheck %s --check-prefix=SHORT
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=add_pc_i64_long_kernel 2>/dev/null | %FileCheck %s --check-prefix=LONG
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=add_pc_i64_lit64_kernel | %FileCheck %s --check-prefix=LIT64
;
; s_add_pc_i64 (gfx1250 SOP1 PC-relative direct branch). Both forms must
; resolve the target via Di.Size read from the actual encoding.
;
; Short form: operand fits inline, SOP1 is 4 bytes.
;   target = 0x08 (site) + 0x04 (size) + 8 (imm) = 0x14
; Long form: operand needs a 32-bit literal, SOP1 is 8 bytes; pins that
; Di.Size is read from the encoding, not assumed to be 4.
;   target = 0x08 (site) + 0x08 (size) + 0x80 (imm) = 0x90

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	add_pc_i64_short_kernel
	.p2align	8
	.type	add_pc_i64_short_kernel,@function
; SHORT-LABEL: define amdgpu_kernel void @add_pc_i64_short_kernel(
; SHORT: br label %bb_0x14
; SHORT-NOT: indirectbr
; SHORT-NOT: unreachable
; SHORT: bb_0x14:
; The skipped v_mov literal must not surface in the lifted IR.
; SHORT-NOT: 3735618049
; SHORT-NOT: 0xDEAD0001
add_pc_i64_short_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_add_pc_i64 8
	v_mov_b32 v1, 0xDEAD0001
	;;#ASMEND
	s_wait_kmcnt 0x0
	v_mov_b32 v1, 0xCAFE0002
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.globl	add_pc_i64_long_kernel
	.p2align	8
	.type	add_pc_i64_long_kernel,@function
; LONG-LABEL: define amdgpu_kernel void @add_pc_i64_long_kernel(
; LONG: br label %bb_0x90
; LONG-NOT: indirectbr
; LONG-NOT: unreachable
; LONG: bb_0x90:
add_pc_i64_long_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_add_pc_i64 0x80
	.rept 32
	s_nop 0
	.endr
	;;#ASMEND
	s_wait_kmcnt 0x0
	v_mov_b32 v1, 0xCAFE0002
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.globl	add_pc_i64_lit64_kernel
	.p2align	8
	.type	add_pc_i64_lit64_kernel,@function
; LIT64-LABEL: define amdgpu_kernel void @add_pc_i64_lit64_kernel(
; lit64 form is 12 bytes total (4-byte SOP1 header + 8-byte literal).
; target = 0x08 (site) + 0x0c (size) + 8 (imm) = 0x1c
; LIT64: br label %bb_0x1C
; LIT64-NOT: indirectbr
; LIT64-NOT: unreachable
; LIT64: bb_0x1C:
add_pc_i64_lit64_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_add_pc_i64 lit64(8)
	s_nop 0
	s_wait_kmcnt 0x0
	v_mov_b32 v1, 0xCAFE0002
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel add_pc_i64_short_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel add_pc_i64_long_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel add_pc_i64_lit64_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           add_pc_i64_short_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         add_pc_i64_short_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           add_pc_i64_long_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         add_pc_i64_long_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           add_pc_i64_lit64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         add_pc_i64_lit64_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
