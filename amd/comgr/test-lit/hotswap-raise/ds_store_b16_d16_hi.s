; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=ds_store_b16_d16_hi_kernel 2>/dev/null | %FileCheck %s

; Lift ds_store_b16_d16_hi to a high-half i16 store (shr 16 + trunc) on addrspace(3).
; CHECK-LABEL: define amdgpu_kernel void @ds_store_b16_d16_hi_kernel(
; CHECK: %ds_st_hi16_shr = lshr i32 %{{[^,]+}}, 16
; CHECK: %ds_st_d16_hi = trunc i32 %ds_st_hi16_shr to i16
; CHECK: store i16 %ds_st_d16_hi, ptr addrspace(3) %{{[^,]+}}
; CHECK-NOT: store i32 %ds_st_d16_hi, ptr addrspace(3)
; CHECK-NOT: store i8 %ds_st_d16_hi, ptr addrspace(3)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_store_b16_d16_hi_kernel
	.p2align	8
	.type	ds_store_b16_d16_hi_kernel,@function
ds_store_b16_d16_hi_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s0, s[0:1], 0x0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, s0
	v_mov_b32 v0, v1
	
	ds_store_b16_d16_hi v0, v2
	
	s_and_saveexec_b32 s0, vcc_lo
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_store_b16_d16_hi_kernel
		.amdhsa_group_segment_fixed_size 4
		.amdhsa_kernarg_size 4
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 2
		.amdhsa_reserve_vcc 1
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
      - { .offset:         0, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 4
    .kernarg_segment_align: 4
    .kernarg_segment_size: 4
    .max_flat_workgroup_size: 1024
    .name:           ds_store_b16_d16_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         ds_store_b16_d16_hi_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
