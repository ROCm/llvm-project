; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_mix_f32_bf16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_fma_mix_f32_bf16 mixed-precision FMA lift.
; CHECK-LABEL: define amdgpu_kernel void @v_fma_mix_f32_bf16_kernel(
; CHECK-DAG: trunc i32 %{{.*}} to i16
; CHECK-DAG: bitcast i16 %{{.*}} to bfloat
; CHECK-DAG: %mix_cvt_bf16 = fpext bfloat %{{.*}} to float
; CHECK-DAG: lshr i32 %{{.*}}, 16
; CHECK-DAG: %mix_cvt_bf16{{[0-9]+}} = fpext bfloat %{{.*}} to float
; CHECK-DAG: bitcast i16 %{{.*}} to half
; CHECK-DAG: %mix_cvt = fpext half %{{.*}} to float
; CHECK-DAG: %mix_cvt{{[0-9]+}} = fpext half %{{.*}} to float
; CHECK-DAG: %fma_mix = call float @llvm.fma.f32(float %mix_cvt_bf16, float %mix_cvt_bf16{{[0-9]+}}, float %{{[0-9]+}})
; CHECK-DAG: %fma_mix{{[0-9]+}} = call float @llvm.fma.f32(float %mix_cvt, float %mix_cvt{{[0-9]+}}, float %{{[0-9]+}})
; CHECK-DAG: %fma_mix{{[0-9]+}} = call float @llvm.fma.f32(float %{{[^,]+}}, float 1.000000e+00, float %{{[^)]+}})
; CHECK-DAG: %fma_mix{{[0-9]+}} = call float @llvm.fma.f32(float %{{[^,]+}}, float 1.000000e+00, float %{{[^)]+}})
; CHECK-DAG: %mix_cvt_bf16_neg = fneg float %mix_cvt_bf16{{[0-9]+}}
; CHECK-DAG: %mix_full_abs = call float @llvm.fabs.f32(float %{{[^)]+}})
; CHECK-DAG: call float @llvm.fma.f32(float %mix_cvt_bf16{{[0-9]+}}, float 1.000000e+00, float %mix_full_abs)
; CHECK-NOT: call float @llvm.fma.f32(float %{{[^,]+}}, float 0.000000e+00,
; CHECK-NOT: unsupportedOpcode
; CHECK-NOT: llvm.amdgcn.cvt.pk
; CHECK-NOT: llvm.amdgcn.mfma

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_fma_mix_f32_bf16_kernel
	.p2align	8
	.type	v_fma_mix_f32_bf16_kernel,@function
v_fma_mix_f32_bf16_kernel:              ; @v_fma_mix_f32_bf16_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_add_nc_u32_e64 v3, s0, 4
	v_add_nc_u32_e64 v5, s0, 8
	v_dual_mov_b32 v1, s0 :: v_dual_mov_b32 v6, s0
	v_fma_mix_f32_bf16 v2, v1, v3, v5 op_sel:[0,1,0] op_sel_hi:[1,1,0]

	v_fma_mix_f32 v3, v6, v3, v5 op_sel:[0,1,0] op_sel_hi:[1,1,0]

	v_fma_mix_f32_bf16 v4, -v6, 1.0, v5 op_sel:[0,0,0] op_sel_hi:[1,1,0]

	v_fma_mix_f32_bf16 v5, v6, 1.0, v5 op_sel:[0,1,0] op_sel_hi:[1,1,0]

	v_fma_mix_f32_bf16 v6, v6, 1.0, |v5| op_sel:[0,0,0] op_sel_hi:[1,1,0]

	global_store_b128 v0, v[2:5], s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_mix_f32_bf16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 7
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_fma_mix_f32_bf16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         v_fma_mix_f32_bf16_kernel.kd
    .vgpr_count:     7
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
