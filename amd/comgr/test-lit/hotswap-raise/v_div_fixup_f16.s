; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=div_fixup_f16_kernel \
; RUN:   | %FileCheck %s

; v_div_fixup_f16 (VOP3, true16) lifts to llvm.amdgcn.div.fixup on f16 -- the
; f16 sibling of the f32/f64 div-fixup handlers. Used by torch's f16 divide.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	div_fixup_f16_kernel
	.p2align	8
	.type	div_fixup_f16_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @div_fixup_f16_kernel(
div_fixup_f16_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
; CHECK: call half @llvm.amdgcn.div.fixup.f16(
	v_div_fixup_f16 v1, v2, v3, v4
	global_store_b16 v0, v1, s[0:1]
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel div_fixup_f16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 2
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
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           div_fixup_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         div_fixup_f16_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
