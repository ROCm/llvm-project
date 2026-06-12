; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel | %FileCheck %s --check-prefix=IR_GFX942
;
; Cross-target lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4
; (gfx1250 -> gfx942), pinning the emitWMMAScaleF8F6F4toMFMA path.
;
; gfx942 lacks both the scaled-WMMA (gfx1250) and scaled-MFMA F8F6F4
; (gfx950) families, so the lowering decomposes K=128 into 4 K=32 unscaled
; bf8.fp8 MFMA calls and applies the per-K-block UE8M0 scale 2^(sA+sB-254)
; on each <4 x f32> partial via ldexp + fmuladd. The fixture kernel uses
; matrix_a_fmt:MATRIX_FMT_BF8 with default matrix_b_fmt:MATRIX_FMT_FP8.
;
; Default WaveNative cross-widen runs two passes (8 MFMA calls total); the
; MODREP RUN line below pins the single-pass path (4 calls).

; MODREP fallback: single-pass path (4 MFMA calls, no select diamond).
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel | %FileCheck %s --check-prefix=IR_GFX942_MODREP

; Refusal pin: gfx90a has MAI but no FP8 MFMA, so the gate must reject.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx90a --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR_GFX90A

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel | %FileCheck %s --check-prefix=IR
;
; Same-target (gfx1250 -> gfx1250) intrinsic-emit path, taken when
; hasTensorOps is true. The kernel compiles to the f8_f8 shape with
; matrix_a_fmt:MATRIX_FMT_BF8 and default matrix_b_fmt:MATRIX_FMT_FP8.
; The native intrinsic takes 14 args and is overloaded on D/A/B element
; types, so the f8_f8 form mangles to .v8f32.v16i32.v16i32.

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel | %FileCheck %s --check-prefix=IR_GFX950
;
; Cross-target lift fixture (gfx1250 -> gfx950), pinning the
; emitWMMAScaleF8F6F4toScaledMFMA path (gated on hasGfx950Insts, not
; hasMFMA, since gfx942 has MFMA but no scaled F8F6F4 family). The lift
; runs Wave32 -> Wave64 redistribution, applies C_mod, and emits the
; gfx950 native scaled MFMA intrinsic.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_scale_f32_16x16x128_f8f6f4_kernel
	.p2align	8
	.type	wmma_scale_f32_16x16x128_f8f6f4_kernel,@function
wmma_scale_f32_16x16x128_f8f6f4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b256 s[36:43], s[0:1], 0x0
	v_mov_b32_e32 v40, 0
	s_wait_kmcnt 0x0
	s_load_b512 s[0:15], s[36:37], 0x0
	s_load_b512 s[16:31], s[38:39], 0x0
	s_load_b256 s[44:51], s[40:41], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[32:33], s[44:45]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	v_mov_b64_e32 v[24:25], s[24:25]
	v_mov_b64_e32 v[26:27], s[26:27]
	v_mov_b64_e32 v[28:29], s[28:29]
	v_mov_b64_e32 v[30:31], s[30:31]
	v_mov_b64_e32 v[34:35], s[46:47]
	v_mov_b64_e32 v[36:37], s[48:49]
	v_mov_b64_e32 v[38:39], s[50:51]
	s_delay_alu instid0(VALU_DEP_1)
; IR_GFX942-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(

; First K-block of pass 0 pins the per-iteration emission order
; (MFMA partial with zero accumulator, then ldexp scale, then fmuladd):
; IR_GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942: sub i32 %{{[^,]+}}, 254
; IR_GFX942: call float @llvm.ldexp.f32.i32(float 1.000000e+00, i32 %{{[^)]+}})
; IR_GFX942: call <4 x float> @llvm.fmuladd.v4f32(
;
; Remaining 7 K-blocks (3 in pass 0, 4 in pass 1):
; IR_GFX942-COUNT-7: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; UE8M0 0xFF NaN-sentinel guard: scale byte == 255 selects qNaN.
; IR_GFX942-DAG: icmp eq i32 %{{[^,]+}}, 255
; IR_GFX942-DAG: select i1 %{{[^,]+}}, float +qnan, float %{{[^,]+}}

; Scale-and-accumulate must stay fused as one fmuladd, not fmul + fadd.
; IR_GFX942-NOT: fmul <4 x float>
; IR_GFX942-NOT: fadd <4 x float>

; WaveNative final per-lane select between pass 0 and pass 1.
; IR_GFX942-DAG: icmp uge i32 %{{[^,]+}}, 32
; IR_GFX942-DAG: select i1 %{{[^,]+}}, i32 %{{[^,]+}}, i32 %{{[^,]+}}

; Lane redistribution marker (wave32 -> wave64).
; IR_GFX942-DAG: call i32 @llvm.amdgcn.ds.bpermute(

; No fall-through to the gfx1250 or gfx950 arms, no wrong fp8/bf8 combo.
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8

; IR_GFX942_MODREP-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(
; IR_GFX942_MODREP-COUNT-4: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942_MODREP-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(
; IR_GFX942_MODREP-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942_MODREP-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4

; STDERR_GFX90A: raise_cli: kernel 'wmma_scale_f32_16x16x128_f8f6f4_kernel' failed to raise:
; STDERR_GFX90A-SAME: v_wmma_scale_f32_16x16x128_f8f6f4
; STDERR_GFX90A-SAME: hasFP8Insts

; IR-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(

; Native scaled-WMMA: matrix_a_fmt=1 (BF8), matrix_b_fmt=0 (FP8), C_mod=0,
; scales=0, scale_src0/1 are runtime VGPRs, reuse a/b = false.
; IR: %wmma_scale{{[0-9]*}} = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(
; IR-SAME: i32 1, <16 x i32> %{{[^,]+}},
; IR-SAME: i32 0, <16 x i32> %{{[^,]+}},
; IR-SAME: i16 0, <8 x float> %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i1 false, i1 false)

; Negative: no MFMA fallback, no non-scaled or other-K WMMA dispatch.
; IR-NOT: @llvm.amdgcn.mfma.scale.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4(
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x32.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x64.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x4.

; IR_GFX950-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(
; IR_GFX950: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.

; Negative: no gfx1250 scaled-WMMA or non-scaled WMMA in the gfx950 IR.
; IR_GFX950-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX950-NOT: @llvm.amdgcn.wmma.f32.16x16x
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[0:15], v[16:31], v[32:39], s42, s43 matrix_a_fmt:MATRIX_FMT_BF8
	s_clause 0x1
	global_store_b128 v40, v[36:39], s[40:41] offset:16
	global_store_b128 v40, v[32:35], s[40:41]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_scale_f32_16x16x128_f8f6f4_kernel
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
    .name:           wmma_scale_f32_16x16x128_f8f6f4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_scale_f32_16x16x128_f8f6f4_kernel.kd
    .vgpr_count:     41
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
