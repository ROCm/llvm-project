; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_max3_i32_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for v_max3_i32. The hardware semantics are:
;   v_max_i32(v_max_i32(S0, S1), S2)
; LLVM's AMDGPU .td names the same semantic AMDGPUsmax3, so the
; target-independent lift is the two-step llvm.smax.i32 chain.

; CHECK-LABEL: define amdgpu_kernel void @v_max3_i32_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_max3_i32_kernel
	.p2align	8
	.type	v_max3_i32_kernel,@function
v_max3_i32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s2, s[0:1], 0x1c
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_and_b32 s4, ttmp6, 15
	s_add_co_i32 s3, s3, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s3, ttmp9, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s4, s3
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s3, ttmp9, s4
	v_mad_u32 v3, s3, s2, v0
	s_load_b128 s[0:3], s[0:1], 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v0, v3, 1, v3
	v_ashrrev_i32_e32 v1, 31, v0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[2:3]
	global_load_b96 v[0:2], v[0:1], off
	s_wait_loadcnt 0x0
; CHECK: %vmax3_i32_lo{{[0-9]*}} = call i32 @llvm.smax.i32(i32 %{{[^,]+}}, i32 %{{[^)]+}})
; CHECK: %vmax3_i32{{[0-9]*}} = call i32 @llvm.smax.i32(i32 %vmax3_i32_lo{{[0-9]*}}, i32 %{{[^)]+}})
; Negative checks: this is signed integer max, not unsigned max or FP maxnum.
; CHECK-NOT: call {{.*}}@llvm.umax
; CHECK-NOT: call {{.*}}@llvm.maxnum
	v_max3_i32 v0, v0, v1, v2
	global_store_b32 v3, v0, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_max3_i32_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           v_max3_i32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         v_max3_i32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
