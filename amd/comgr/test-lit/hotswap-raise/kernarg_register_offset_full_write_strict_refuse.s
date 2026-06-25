; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && env HSA_HOTSWAP_STRICT=1 %not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=kernarg_register_offset_full_write_strict_refuse 2>&1 \
; RUN:   | %FileCheck %s
;
; A generic full-pair SGPR write kills the entry-pointer reaching definition but
; does not prove a non-entry value. This no-op add preserves the pointer value,
; so strict mode must still refuse dynamic source implicit-arg offsets.

; CHECK: dynamic source kernarg offsets may reach the source implicit-arg range

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kernarg_register_offset_full_write_strict_refuse
	.p2align	8
	.type	kernarg_register_offset_full_write_strict_refuse,@function
kernarg_register_offset_full_write_strict_refuse:
	s_mov_b32 s6, 0
	s_mov_b32 s7, 0
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]
	s_mov_b32 s4, 0xc
	s_load_b32 s2, s[0:1], s4
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_register_offset_full_write_strict_refuse
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           kernarg_register_offset_full_write_strict_refuse
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         kernarg_register_offset_full_write_strict_refuse.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
