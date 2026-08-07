; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_fp8_fp4_kernel | %FileCheck %s --check-prefix=IR_GFX942

; Scaled f8f6f4 WMMA fp8xfp4 format re-encode (fp8 to gfx942 FNUZ).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_scale_f32_16x16x128_fp8_fp4_kernel
	.p2align	8
	.type	wmma_scale_f32_16x16x128_fp8_fp4_kernel,@function
wmma_scale_f32_16x16x128_fp8_fp4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b256 s[36:43], s[0:1], 0x0
	v_mov_b32_e32 v40, 0
	s_wait_kmcnt 0x0
	s_load_b512 s[0:15], s[36:37], 0x0
	s_load_b256 s[16:23], s[38:39], 0x0
	s_load_b256 s[44:51], s[40:41], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	v_mov_b64_e32 v[24:25], s[44:45]
	v_mov_b64_e32 v[26:27], s[46:47]
	v_mov_b64_e32 v[28:29], s[48:49]
	v_mov_b64_e32 v[30:31], s[50:51]
	s_delay_alu instid0(VALU_DEP_1)

; IR_GFX942-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_fp8_fp4_kernel(
; IR_GFX942: trunc <4 x i32> %{{[^ ]+}} to <4 x i8>
; IR_GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942-COUNT-7: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; Mixed E8M0 (A) x UE4M3 (B): the E8M0 side keeps its ldexp shortcut, the
; UE4M3 side goes through the shared OCP fp8 decoder.
; IR_GFX942-DAG: sub i32 %{{[^,]+}}, 127
; IR_GFX942-DAG: %fp8_dec_ocp{{[0-9]*}} = select
; IR_GFX942-DAG: %scale_factor{{[0-9]*}} = fmul float
; IR_GFX942-NOT: @llvm.amdgcn.cvt.f32.fp8
; IR_GFX942-NOT: @__const.
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[0:15], v[16:23], v[24:31], s42, s43 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP4 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
	s_clause 0x1
	global_store_b128 v40, v[28:31], s[40:41] offset:16
	global_store_b128 v40, v[24:27], s[40:41]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_scale_f32_16x16x128_fp8_fp4_kernel
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 41
		.amdhsa_next_free_sgpr 52
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
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
      - { .offset:         24, .size:           4, .value_kind:     by_value }
      - { .offset:         28, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           wmma_scale_f32_16x16x128_fp8_fp4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_scale_f32_16x16x128_fp8_fp4_kernel.kd
    .vgpr_count:     41
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
