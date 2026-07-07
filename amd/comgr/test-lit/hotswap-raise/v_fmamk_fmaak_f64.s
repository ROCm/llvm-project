; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_fmamk_fmaak_f64_kernel 2>/dev/null | %FileCheck %s

; v_fmamk_f64/fmaak_f64 literal-operand FMA lift.
; CHECK-LABEL: define amdgpu_kernel void @v_fmamk_fmaak_f64_kernel(
; CHECK: call {{.*}}double @llvm.fma.f64(double {{.*}}, double {{f?0x400921FB54442D18}}, double {{.*}})
; CHECK: call {{.*}}double @llvm.fma.f64(double {{.*}}, double {{.*}}, double {{f?0x4005BF0A8B145769}})
; CHECK: declare {{.*}}double @llvm.fma.f64(double, double, double)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_fmamk_fmaak_f64_kernel
	.p2align	8
	.type	v_fmamk_fmaak_f64_kernel,@function
v_fmamk_fmaak_f64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x1c
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s3, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v4, s0, s2, v0
	global_load_b64 v[0:1], v4, s[6:7] scale_offset
	global_load_b64 v[2:3], v4, s[6:7] offset:8 scale_offset
	s_wait_loadcnt 0x0
	v_fmamk_f64 v[0:1], v[0:1], 0x400921fb54442d18, v[2:3]
	v_fmaak_f64 v[2:3], v[0:1], v[2:3], 0x4005bf0a8b145769

	global_store_b64 v4, v[2:3], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_fmamk_fmaak_f64_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           v_fmamk_fmaak_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_fmamk_fmaak_f64_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
