; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=divscale_f32_vcc_hi 2>&1 | %FileCheck %s --check-prefix=F32-VCC
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=divscale_f32_exec_hi 2>&1 | %FileCheck %s --check-prefix=F32-EXEC
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=divscale_f64_vcc_hi 2>&1 | %FileCheck %s --check-prefix=F64-VCC
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=divscale_f64_exec_hi 2>&1 | %FileCheck %s --check-prefix=F64-EXEC
;
; On wave32, vcc_hi / exec_hi are scratch scalars, not the wave mask; the ISA
; does not allow them as a v_div_scale flag destination. Every such encoding
; (f32/f64 x vcc_hi/exec_hi) must be refused rather than silently lowered.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

; F32-VCC: kernel 'divscale_f32_vcc_hi'
; F32-VCC-SAME: v_div_scale_f32 [VALU]
; F32-VCC-SAME: v_div_scale flag destination is wave32 vcc_hi/exec_hi scratch
	.globl	divscale_f32_vcc_hi
	.p2align	8
	.type	divscale_f32_vcc_hi,@function
divscale_f32_vcc_hi:
	v_div_scale_f32 v5, vcc_hi, v0, v1, v0
	s_endpgm

; F32-EXEC: kernel 'divscale_f32_exec_hi'
; F32-EXEC-SAME: v_div_scale_f32 [VALU]
; F32-EXEC-SAME: v_div_scale flag destination is wave32 vcc_hi/exec_hi scratch
	.globl	divscale_f32_exec_hi
	.p2align	8
	.type	divscale_f32_exec_hi,@function
divscale_f32_exec_hi:
	v_div_scale_f32 v5, exec_hi, v0, v1, v0
	s_endpgm

; F64-VCC: kernel 'divscale_f64_vcc_hi'
; F64-VCC-SAME: v_div_scale_f64 [VALU]
; F64-VCC-SAME: v_div_scale flag destination is wave32 vcc_hi/exec_hi scratch
	.globl	divscale_f64_vcc_hi
	.p2align	8
	.type	divscale_f64_vcc_hi,@function
divscale_f64_vcc_hi:
	v_div_scale_f64 v[4:5], vcc_hi, v[0:1], v[2:3], v[0:1]
	s_endpgm

; F64-EXEC: kernel 'divscale_f64_exec_hi'
; F64-EXEC-SAME: v_div_scale_f64 [VALU]
; F64-EXEC-SAME: v_div_scale flag destination is wave32 vcc_hi/exec_hi scratch
	.globl	divscale_f64_exec_hi
	.p2align	8
	.type	divscale_f64_exec_hi,@function
divscale_f64_exec_hi:
	v_div_scale_f64 v[4:5], exec_hi, v[0:1], v[2:3], v[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel divscale_f32_vcc_hi
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.amdhsa_kernel divscale_f32_exec_hi
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.amdhsa_kernel divscale_f64_vcc_hi
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.amdhsa_kernel divscale_f64_exec_hi
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
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
    .name:           divscale_f32_vcc_hi
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         divscale_f32_vcc_hi.kd
    .vgpr_count:     8
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           divscale_f32_exec_hi
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         divscale_f32_exec_hi.kd
    .vgpr_count:     8
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           divscale_f64_vcc_hi
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         divscale_f64_vcc_hi.kd
    .vgpr_count:     8
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           divscale_f64_exec_hi
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         divscale_f64_exec_hi.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
