; RUN: %llvm_mc -mcpu=gfx1200 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=unsupported_hidden_global_offset 2>&1 \
; RUN:   | %FileCheck %s
;
; Known source hidden args must not silently fall back to the target runtime's
; implicit-arg layout.  If we cannot synthesize a source hidden arg from the
; dispatch packet, refuse the translation.

; CHECK: unsupported-source-hidden-arg: s_load_b64 [SMEM]
; CHECK-SAME: unsupported source hidden argument kind 'hidden_global_offset_x'

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
	.amdhsa_code_object_version 6
	.text
	.globl	unsupported_hidden_global_offset
	.p2align	8
	.type	unsupported_hidden_global_offset,@function
unsupported_hidden_global_offset:
	s_load_b64 s[2:3], s[0:1], 0x48
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel unsupported_hidden_global_offset
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .max_flat_workgroup_size: 1024
    .name:           unsupported_hidden_global_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .sgpr_spill_count: 0
    .symbol:         unsupported_hidden_global_offset.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1200
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
