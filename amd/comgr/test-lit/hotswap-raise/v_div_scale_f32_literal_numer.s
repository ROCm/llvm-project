; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_div_scale_f32_literal_numer_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_div_scale f32 literal-numerator lift.
; CHECK-LABEL: define amdgpu_kernel void @v_div_scale_f32_literal_numer_kernel(
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float 1.000000e+00, float %{{[^,]+}}, i1 false)
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float 1.000000e+00, float %{{[^,]+}}, i1 true)
; CHECK: call float @llvm.amdgcn.div.fixup.f32(float %{{[^,]+}}, float %{{[^,]+}}, float 1.000000e+00)
; CHECK-NOT: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{[^,)]+}},

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_div_scale_f32_literal_numer_kernel
	.p2align	8
	.type	v_div_scale_f32_literal_numer_kernel,@function
v_div_scale_f32_literal_numer_kernel:   ; @v_div_scale_f32_literal_numer_kernel
; %bb.0:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	flat_load_b32 v1, v0, s[6:7] scale_offset scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v2, 0x4f800000, v1
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_sqrt_f32_e32 v2, v1
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_nc_u32 v3, -1, v2 :: v_dual_add_nc_u32 v4, 1, v2
	v_fma_f32 v5, -v3, v2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ge_f32_e64 s0, 0, v5
	v_dual_fma_f32 v6, -v4, v2, v1 :: v_dual_cndmask_b32 v2, v2, v3, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e64 s0, 0, v6
	v_cndmask_b32_e64 v2, v2, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, 0x37800000, v2
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo
	v_cmp_class_f32_e64 vcc_lo, v1, 0x260
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, v2, v1, vcc_lo
	v_div_scale_f32 v2, null, v1, v1, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v3, v2
	v_nop
	v_fma_f32 v4, -v2, v3, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, vcc_lo, 1.0, v1, 1.0
	v_mul_f32_e32 v5, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v2, v5, v4
	v_fmac_f32_e32 v5, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v2, v5, v4
	v_div_fmas_f32 v2, v2, v3, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v1, v2, v1, 1.0
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_div_scale_f32_literal_numer_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_div_scale_f32_literal_numer_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_div_scale_f32_literal_numer_kernel.kd
    .vgpr_count:     7
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
