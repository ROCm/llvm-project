; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_div_scale_f32_self_divide_kernel \
; RUN:   | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_div_scale_f32_self_divide_kernel
	.p2align	8
	.type	v_div_scale_f32_self_divide_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_div_scale_f32_self_divide_kernel(
v_div_scale_f32_self_divide_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	flat_load_b32 v2, v0, s[6:7] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_div_scale_f32 v3, s2, v2, v2, v2
	; CHECK: %[[N:[0-9]+]] = bitcast i32 %[[V:[A-Za-z0-9_.]+]] to float
	; CHECK: %[[D:[0-9]+]] = bitcast i32 %[[V]] to float
	; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %[[N]], float %[[D]], i1 false)
	v_div_scale_f32 v4, s3, v2, v2, v2
	v_rcp_f32_e32 v5, v3
	v_nop
	v_fma_f32 v6, -v3, v5, 1.0
	v_fmac_f32_e32 v5, v6, v5
	v_mul_f32_e32 v7, v4, v5
	v_fma_f32 v8, -v3, v7, v4
	v_fmac_f32_e32 v7, v8, v5
	v_fma_f32 v3, -v3, v7, v4
	s_mov_b32 vcc_lo, s3
	v_div_fmas_f32 v3, v3, v5, v7
	v_div_fixup_f32 v1, v3, v2, v2
	; CHECK: call float @llvm.amdgcn.div.fixup.f32(
	global_store_b32 v0, v1, s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_div_scale_f32_self_divide_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 10
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
    .name:           v_div_scale_f32_self_divide_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_div_scale_f32_self_divide_kernel.kd
    .vgpr_count:     10
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
