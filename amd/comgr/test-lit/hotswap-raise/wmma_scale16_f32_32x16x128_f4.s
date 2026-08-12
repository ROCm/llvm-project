; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale16_f32_32x16x128_f4_kernel | %FileCheck %s --check-prefix=IR_GFX942

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native --emit-ir=wmma_scale16_f32_32x16x128_f4_kernel | %FileCheck %s --check-prefix=IR_GFX942_MODREP

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx90a --emit-ir=wmma_scale16_f32_32x16x128_f4_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR_GFX90A

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=wmma_scale16_f32_32x16x128_f4_kernel | %FileCheck %s --check-prefix=IR

; Cross-target lift (gfx1250 -> gfx942) for M=32 FP4 WMMA-scale16. Lowering
; M-splits into two 16x16 emitWMMAScale16F8F6F4toMFMA chains (lo/hi A and C
; halves) with FP4 -> FP8 widening and 8 K=16 scale blocks each.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_scale16_f32_32x16x128_f4_kernel
	.p2align	8
	.type	wmma_scale16_f32_32x16x128_f4_kernel,@function
wmma_scale16_f32_32x16x128_f4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b256 s[36:43], s[0:1], 0x0
	v_mov_b32_e32 v56, 0
	s_wait_kmcnt 0x0
	s_load_b512 s[0:15], s[36:37], 0x0
	s_load_b256 s[16:23], s[38:39], 0x0
	s_load_b512 s[24:39], s[40:41], 0x0
	s_load_b256 s[44:51], s[42:43], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[8:9], s[0:1]
	v_mov_b64_e32 v[10:11], s[2:3]
	v_mov_b64_e32 v[12:13], s[4:5]
	v_mov_b64_e32 v[14:15], s[6:7]
	v_mov_b64_e32 v[16:17], s[8:9]
	v_mov_b64_e32 v[18:19], s[10:11]
	v_mov_b64_e32 v[20:21], s[12:13]
	v_mov_b64_e32 v[22:23], s[14:15]
	v_mov_b64_e32 v[24:25], s[16:17]
	v_mov_b64_e32 v[26:27], s[18:19]
	v_mov_b64_e32 v[28:29], s[20:21]
	v_mov_b64_e32 v[30:31], s[22:23]
	v_mov_b64_e32 v[32:33], s[24:25]
	v_mov_b64_e32 v[34:35], s[26:27]
	v_mov_b64_e32 v[36:37], s[28:29]
	v_mov_b64_e32 v[38:39], s[30:31]
	v_mov_b64_e32 v[40:41], s[32:33]
	v_mov_b64_e32 v[42:43], s[34:35]
	v_mov_b64_e32 v[44:45], s[36:37]
	v_mov_b64_e32 v[46:47], s[38:39]
	v_mov_b64_e32 v[48:49], s[44:45]
	v_mov_b64_e32 v[50:51], s[46:47]
	v_mov_b64_e32 v[52:53], s[48:49]
	v_mov_b64_e32 v[54:55], s[50:51]
	s_delay_alu instid0(VALU_DEP_1)

; IR_GFX942-LABEL: define amdgpu_kernel void @wmma_scale16_f32_32x16x128_f4_kernel(

; FP4 -> fp8 widening (vectorized per dword).
; IR_GFX942-DAG: select <8 x i1>
; IR_GFX942-DAG: shl <8 x i32>

; M-split: two 16x16 chains, 8 K=16 blocks each under WaveNative (16 MFMA/chain).
; IR_GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942-COUNT-31: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; Scale-on-output fmuladd per K-block.
; IR_GFX942-DAG: call <4 x float> @llvm.fmuladd.v4f32(

; Negatives: no LUT/load, no gfx1250/gfx950 scaled dispatch, no K=32 scale form.
; IR_GFX942-NOT: @__const.
; IR_GFX942-NOT: load i8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale16.f32.32x16x128.f4
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4

; IR_GFX942_MODREP-LABEL: define amdgpu_kernel void @wmma_scale16_f32_32x16x128_f4_kernel(
; IR_GFX942_MODREP-COUNT-16: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942_MODREP-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(
; IR_GFX942_MODREP-NOT: @llvm.amdgcn.wmma.scale16.f32.32x16x128.f4

; STDERR_GFX90A: raise_cli: kernel 'wmma_scale16_f32_32x16x128_f4_kernel' failed to raise:
; STDERR_GFX90A-SAME: v_wmma_scale16_f32_32x16x128_f4
; STDERR_GFX90A-SAME: hasFP8Insts

; IR-LABEL: define amdgpu_kernel void @wmma_scale16_f32_32x16x128_f4_kernel(
; IR: %wmma_scale16_32x16{{[0-9]*}} = call <16 x float> @llvm.amdgcn.wmma.scale16.f32.32x16x128.f4.v16f32.v16i32.v8i32(
; IR-SAME: <16 x i32> %{{[^,]+}},
; IR-SAME: <8 x i32> %{{[^,]+}},
; IR-SAME: i16 0, <16 x float> %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i64 %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i64 %{{[^,]+}},
; IR-SAME: i1 false, i1 false)
; IR-NOT: @llvm.amdgcn.mfma.scale.
; IR-NOT: @llvm.amdgcn.wmma.scale.f32.
; IR-NOT: @llvm.amdgcn.wmma.f32.32x16x

	v_wmma_scale16_f32_32x16x128_f4 v[32:47], v[8:23], v[24:31], v[32:47], s[44:45], s[46:47]
	s_clause 0x1
	global_store_b128 v56, v[44:47], s[40:41] offset:48
	global_store_b128 v56, v[40:43], s[40:41] offset:32
	global_store_b128 v56, v[36:39], s[40:41] offset:16
	global_store_b128 v56, v[32:35], s[40:41]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_scale16_f32_32x16x128_f4_kernel
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 57
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
    .name:           wmma_scale16_f32_32x16x128_f4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_scale16_f32_32x16x128_f4_kernel.kd
    .vgpr_count:     57
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
