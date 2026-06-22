; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && env HSA_HOTSWAP_STRICT=1 %not raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --emit-ir=narrow_implicitarg_unmatched 2>&1 \
; RUN:   | %FileCheck %s
;
; Narrow scalar loads share the same strict hidden-arg contract as dword SMEM:
; an implicit-range offset that does not map to source hidden metadata must not
; fall through to a raw kernarg-segment load in strict mode.

; CHECK: implicit-arg offsets are being applied to the target runtime hidden-arg block

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	narrow_implicitarg_unmatched
	.p2align	8
	.type	narrow_implicitarg_unmatched,@function
narrow_implicitarg_unmatched:
	s_load_u8 s2, s[0:1], 0xc
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel narrow_implicitarg_unmatched
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
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
    .name:           narrow_implicitarg_unmatched
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         narrow_implicitarg_unmatched.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
