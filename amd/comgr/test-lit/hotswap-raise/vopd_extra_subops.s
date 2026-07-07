; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vopd_extra_subops_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; extra VOPD component subops (ashr/cndmask/smax/add/fmamk/fmaak) lift per-component.
; CHECK-LABEL: define amdgpu_kernel void @vopd_extra_subops_kernel(
; CHECK: %vopd_ashr = ashr i32 %{{[^,]+}}, 31
; CHECK: phi i32 [ 1065353216,
; CHECK: %vopd_smax = call i32 @llvm.smax.i32(i32 0, i32 %{{[^)]+}})
; CHECK: phi i32 [ %ttmp9_wg_id,
; CHECK: call i64 @llvm.amdgcn.ballot.i64
; CHECK: %vopd_add{{[0-9]*}} = add i32 -8, %{{[^,]+}}
; CHECK: %vopd_fmamk = call float @llvm.fma.f32
; CHECK: %vopd_fmaak = call float @llvm.fma.f32
; CHECK-NOT: %vopd_lshr =
; CHECK-NOT: @llvm.umax.i32
; CHECK-NOT: -2147483640

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_extra_subops_kernel
	.p2align	8
	.type	vopd_extra_subops_kernel,@function
vopd_extra_subops_kernel:               ; @vopd_extra_subops_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_add_nc_u32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 31, v1
	v_and_b32_e32 v4, 31, v2
	s_wait_kmcnt 0x0
	global_load_b32 v8, v0, s[0:1] scale_offset
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v2, 1.0 :: v_dual_ashrrev_i32 v3, 31, v8
	s_clause 0x1
	global_load_b32 v5, v1, s[0:1] scale_offset
	global_load_b32 v6, v4, s[0:1] scale_offset
	s_wait_xcnt 0x1
	v_dual_add_nc_u32 v1, 3, v0 :: v_dual_lshlrev_b32 v0, 5, v0
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v5, v5 :: v_dual_max_i32 v4, 0, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 31, v1
	v_dual_mov_b32 v6, ttmp9 :: v_dual_mov_b32 v7, s0
	global_load_b32 v1, v1, s[0:1] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_eq_u32_e64 vcc_lo, v8, 0
	v_dual_mov_b32 v8, vcc_lo :: v_dual_mov_b32 v9, v1
	v_dual_add_nc_u32 v10, -8, v0 :: v_dual_mov_b32 v11, v0
	v_dual_mov_b32 v10, v8 :: v_dual_fmamk_f32 v11, v8, 0xcf800000, v11
	v_dual_mov_b32 v10, v8 :: v_dual_fmaak_f32 v11, v8, v11, 0x3f800000
	s_clause 0x1
	global_store_b128 v0, v[2:5], s[2:3]
	global_store_b128 v0, v[6:9], s[2:3] offset:16
	global_store_dword v0, v10, s[2:3] offset:32
	global_store_dword v0, v11, s[2:3] offset:36
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_extra_subops_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 4
		.amdhsa_reserve_vcc 1
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
    .name:           vopd_extra_subops_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         vopd_extra_subops_kernel.kd
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
