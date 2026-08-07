; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 \
; RUN:     --emit-ir=v_cvt_scalef32_pk8_fp8_f32_kernel \
; RUN:   | %FileCheck %s --check-prefix=NATIVE
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_scalef32_pk8_fp8_f32_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS \
; RUN:       --implicit-check-not=e4m3_ocp \
; RUN:       --implicit-check-not=@llvm.amdgcn.cvt.scalef32.pk8.fp8.f32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_scalef32_pk8_fp8_f32_kernel
	.p2align	8
	.type	v_cvt_scalef32_pk8_fp8_f32_kernel,@function
v_cvt_scalef32_pk8_fp8_f32_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
; NATIVE-LABEL: define amdgpu_kernel void @v_cvt_scalef32_pk8_fp8_f32_kernel(
; NATIVE: %cvt_scalef32_pk8_fp8{{[0-9]*}} = call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.f32(<8 x float> %{{.*}}, float %{{.*}})
; NATIVE-NOT: call {{.*}}@llvm.amdgcn.cvt.pk.fp8.f32
; CROSS-LABEL: define amdgpu_kernel void @v_cvt_scalef32_pk8_fp8_f32_kernel(
; Hardware scales by the E8M0 exponent field of src1 alone, not by the whole
; f32: mask bits[30:23] to get 2^(e-127), with e==0 -> 2^-127 and e==255 -> NaN.
; CROSS-DAG: %e8m0_exp = and i32 %{{.+}}, 2139095040
; CROSS-DAG: select i1 %{{.+}}, float f0x00400000, float %{{.+}}
; CROSS-DAG: %e8m0_scale = select i1 %{{.+}}, float +qnan, float %{{.+}}
; CROSS-DAG: insertelement <8 x float> poison, float %e8m0_scale, i64 0
; CROSS-DAG: shufflevector <8 x float> %{{.+}}, <8 x float> poison, <8 x i32> zeroinitializer
; CROSS-DAG: %scaled = fmul <8 x float>
; Four hw encodes as before, but each now feeds an exact OCP pair-encode
; (%pk_fp8_ocp) instead of a byte-level FNUZ->OCP re-encode. Pin which scaled
; lanes each pair reads and which half of which result dword it lands in: an
; off-by-one in the pair/dword indexing mis-pairs lanes or drops half the
; output, and an un-counted DAG match would not notice.
; CROSS: %[[L0:[^ ]+]] = extractelement <8 x float> %scaled, i64 0
; CROSS: %[[L1:[^ ]+]] = extractelement <8 x float> %scaled, i64 1
; CROSS: %[[H0:[^ ]+]] = fmul float %[[L0]], 5.000000e-01
; CROSS: %[[H1:[^ ]+]] = fmul float %[[L1]], 5.000000e-01
; CROSS: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %[[H0]], float %[[H1]], i32 0, i1 false)
; CROSS: %[[P0:pk_fp8_ocp[0-9]*]] = or i32
; CROSS: %[[L2:[^ ]+]] = extractelement <8 x float> %scaled, i64 2
; CROSS: %[[L3:[^ ]+]] = extractelement <8 x float> %scaled, i64 3
; CROSS: %[[H2:[^ ]+]] = fmul float %[[L2]], 5.000000e-01
; CROSS: %[[H3:[^ ]+]] = fmul float %[[L3]], 5.000000e-01
; CROSS: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %[[H2]], float %[[H3]], i32 0, i1 false)
; CROSS: %[[P1:pk_fp8_ocp[0-9]*]] = or i32
; CROSS: %[[S1:[^ ]+]] = shl i32 %[[P1]], 16
; CROSS: %[[DW0:[^ ]+]] = or i32 %[[P0]], %[[S1]]
; CROSS: %[[L4:[^ ]+]] = extractelement <8 x float> %scaled, i64 4
; CROSS: %[[L5:[^ ]+]] = extractelement <8 x float> %scaled, i64 5
; CROSS: %[[H4:[^ ]+]] = fmul float %[[L4]], 5.000000e-01
; CROSS: %[[H5:[^ ]+]] = fmul float %[[L5]], 5.000000e-01
; CROSS: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %[[H4]], float %[[H5]], i32 0, i1 false)
; CROSS: %[[P2:pk_fp8_ocp[0-9]*]] = or i32
; CROSS: %[[L6:[^ ]+]] = extractelement <8 x float> %scaled, i64 6
; CROSS: %[[L7:[^ ]+]] = extractelement <8 x float> %scaled, i64 7
; CROSS: %[[H6:[^ ]+]] = fmul float %[[L6]], 5.000000e-01
; CROSS: %[[H7:[^ ]+]] = fmul float %[[L7]], 5.000000e-01
; CROSS: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %[[H6]], float %[[H7]], i32 0, i1 false)
; CROSS: %[[P3:pk_fp8_ocp[0-9]*]] = or i32
; CROSS: %[[S3:[^ ]+]] = shl i32 %[[P3]], 16
; CROSS: %[[DW1:[^ ]+]] = or i32 %[[P2]], %[[S3]]
; CROSS: %[[V0:[^ ]+]] = insertelement <2 x i32> poison, i32 %[[DW0]], i64 0
; CROSS: insertelement <2 x i32> %[[V0]], i32 %[[DW1]], i64 1
	v_cvt_scalef32_pk8_fp8_f32 v[0:1], v[2:9], v10
	global_store_b64 v11, v[0:1], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_scalef32_pk8_fp8_f32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 4
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_scalef32_pk8_fp8_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_cvt_scalef32_pk8_fp8_f32_kernel.kd
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
