; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vopd_same_opcode_dual_issue_kernel \
; RUN:   | %FileCheck %s

; identical-opcode v_dual pairs lift as two scalar ops, not a <2 x> vector.
; CHECK-LABEL: define amdgpu_kernel void @vopd_same_opcode_dual_issue_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_same_opcode_dual_issue_kernel
	.p2align	8
	.type	vopd_same_opcode_dual_issue_kernel,@function
vopd_same_opcode_dual_issue_kernel:     ; @vopd_same_opcode_dual_issue_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v8, v0
	global_load_b128 v[0:3], v8, s[0:1] scale_offset
	s_wait_loadcnt 0x0

; CHECK: %vopd_fadd = fadd float
; CHECK: %vopd_fadd{{[0-9]+}} = fadd float
; CHECK-NOT: fadd <2 x float>
	v_dual_add_f32 v4, v0, v1 :: v_dual_add_f32 v5, v2, v3

; CHECK: %vopd_fmul = fmul float
; CHECK: %vopd_fmul{{[0-9]+}} = fmul float
; CHECK-NOT: fmul <2 x float>
	v_dual_mul_f32 v6, v0, v1 :: v_dual_mul_f32 v7, v2, v3

; CHECK: %vopd_fsub = fsub float
; CHECK: %vopd_fsub{{[0-9]+}} = fsub float
; CHECK-NOT: %vopd_fsubrev = fsub float
	v_dual_sub_f32 v10, v0, v1 :: v_dual_sub_f32 v11, v2, v3

; CHECK: %vopd_sub = sub i32
; CHECK: %vopd_sub{{[0-9]+}} = sub i32
	v_dual_sub_nc_u32 v12, v0, v1 :: v_dual_sub_nc_u32 v13, v2, v3

; CHECK: %vopd_fma = call float @llvm.fma.f32(float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}})
; CHECK: %vopd_fma{{[0-9]+}} = call float @llvm.fma.f32(float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}})
	v_dual_fma_f32 v14, v0, v1, v2 :: v_dual_fma_f32 v15, v2, v3, v0

; CHECK: %vopd_fmaak = call float @llvm.fma.f32(float %{{[^,]+}}, float %{{[^,]+}}, float 1.000000e+00)
; CHECK: %vopd_fmaak{{[0-9]+}} = call float @llvm.fma.f32(float %{{[^,]+}}, float %{{[^,]+}}, float 1.000000e+00)
	v_dual_fmaak_f32 v16, v0, v1, 0x3f800000 :: v_dual_fmaak_f32 v17, v2, v3, 0x3f800000

; CHECK-DAG: %vopd_shl = shl i32 %{{[^,]+}}, 4
; CHECK-DAG: %vopd_shl{{[0-9]+}} = shl i32 %{{[^,]+}}, 2
; CHECK-NOT: %vopd_lshr =
; CHECK-NOT: %vopd_ashr =
; CHECK-NOT: shl i32 4, %
; CHECK-NOT: shl i32 2, %
	v_dual_lshlrev_b32 v18, 4, v0 :: v_dual_lshlrev_b32 v19, 2, v0

; CHECK: declare float @llvm.fma.f32(float, float, float)
; CHECK-NOT: unhandled structural VOPD component
; CHECK-NOT: unsupported VOPD
	global_store_b32 v8, v4, s[2:3] scale_offset
	global_store_b32 v8, v5, s[2:3] offset:4
	global_store_b32 v8, v6, s[2:3] offset:8
	global_store_b32 v8, v7, s[2:3] offset:12
	global_store_b32 v8, v10, s[2:3] offset:16
	global_store_b32 v8, v11, s[2:3] offset:20
	global_store_b32 v8, v12, s[2:3] offset:24
	global_store_b32 v8, v13, s[2:3] offset:28
	global_store_b32 v8, v14, s[2:3] offset:32
	global_store_b32 v8, v15, s[2:3] offset:36
	global_store_b32 v8, v16, s[2:3] offset:40
	global_store_b32 v8, v17, s[2:3] offset:44
	global_store_b32 v8, v18, s[2:3] offset:48
	global_store_b32 v8, v19, s[2:3] offset:52
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_same_opcode_dual_issue_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 4
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
    .name:           vopd_same_opcode_dual_issue_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         vopd_same_opcode_dual_issue_kernel.kd
    .vgpr_count:     20
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
