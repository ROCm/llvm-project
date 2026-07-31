; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: %raise_cli %t.hsaco --target-isa=gfx1151 --write-hsaco=%t.gfx1151.hsaco
; RUN: %llvm-readelf --notes %t.gfx1151.hsaco | %FileCheck %s
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --write-hsaco=%t.gfx942.hsaco
; RUN: %llvm-readelf --notes %t.gfx942.hsaco | %FileCheck %s

; The lifted segment stays at the source's 264 bytes on both a same-wave
; (gfx1151) and a wave-widening (gfx942) target, with no implicit-argument
; block appended after it.
; CHECK: .kernarg_segment_size: 264
; CHECK-NOT: hidden_

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	lifted_kernarg_segment
	.p2align	8
	.type	lifted_kernarg_segment,@function
lifted_kernarg_segment:
	s_load_b32 s4, s[0:1], 0x8
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s4
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lifted_kernarg_segment
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .offset: 8, .size: 4, .value_kind: hidden_block_count_x }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           lifted_kernarg_segment
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         lifted_kernarg_segment.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
