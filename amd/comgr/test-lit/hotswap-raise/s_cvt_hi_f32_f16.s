; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_cvt_hi_f32_f16_kernel 2>/dev/null | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --write-hsaco=%t.out --kernel=s_cvt_hi_f32_f16_kernel 2>&1 | %FileCheck %s --check-prefix=PIPE

; s_cvt_hi_f32_f16 high-half scalar fpext-to-float lift.
; IR-LABEL: define amdgpu_kernel void @s_cvt_hi_f32_f16_kernel(
; IR: [[HI:%[^ ]+]] = lshr i32 {{%[^,]+}}, 16
; IR-NEXT: [[BITS:%[^ ]+]] = trunc i32 [[HI]] to i16
; IR-NEXT: [[HALF:%[^ ]+]] = bitcast i16 [[BITS]] to half
; IR-NEXT: [[F32:%[^ ]+]] = fpext half [[HALF]] to float
; IR-NEXT: bitcast float [[F32]] to i32
; IR-NOT: unsupported instruction
; PIPE: raise_cli: wrote
; PIPE-SAME: s_cvt_hi_f32_f16_kernel

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_cvt_hi_f32_f16_kernel
	.p2align	8
	.type	s_cvt_hi_f32_f16_kernel,@function
s_cvt_hi_f32_f16_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, 0
	s_cvt_hi_f32_f16 s0, s6
	v_mov_b32_e32 v1, s0
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_cvt_hi_f32_f16_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
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
      - { .offset:         8, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           s_cvt_hi_f32_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_cvt_hi_f32_f16_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
