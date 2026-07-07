; REQUIRES: tdm-runtime
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=tensor_store_from_lds_kernel 2>&1 | %FileCheck %s --check-prefix=IR

; Tensor store-from-LDS TDM descriptor lift.
; IR: @llvm.compiler.used
; IR-SAME: @hotswap_tdm_store_from_lds
; IR-LABEL: define amdgpu_kernel void @tensor_store_from_lds_kernel
; IR-NOT: call void @hotswap_tdm_store_from_lds(
; IR: hotswap_tdm_store_from_lds.exit:

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	tensor_store_from_lds_kernel
	.p2align	8
	.type	tensor_store_from_lds_kernel,@function
tensor_store_from_lds_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	.long 0xd0714001
	.long 0x7c000000
	.long 0x7c7c0428
	s_wait_kmcnt 0x0
	global_store_b32 v0, v0, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel tensor_store_from_lds_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 44
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
    .name:           tensor_store_from_lds_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .symbol:         tensor_store_from_lds_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
