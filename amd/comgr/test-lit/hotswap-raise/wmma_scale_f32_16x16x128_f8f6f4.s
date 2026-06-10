; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for v_wmma_scale_f32_16x16x128_f8f6f4 (gfx1250
; RDNA4 VOP3PX2 opcode 0x033, ScaledWMMA family). Pins the contractual
; cross-target loud-failure behaviour of
; transpiler/handle_valu_vop3p.cpp under
; CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4. Companion fixture to
; `wmma_scale_f32_16x16x128_f8f6f4_same_target.ll`, which pins the
; same-target (gfx1250 -> gfx1250) intrinsic-emit path.
;
; gfx942 has no scaled-WMMA hardware AND no scaled-F8F6F4 MFMA family
; (gfx942's MFMA stops at the non-scaled `mfma_f32_16x16x32_*` shapes;
; the scaled `int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`, IntrinsicsAMDGPU.td:3694,
; is gated on the `FeatureGFX950Insts` subtarget feature and is gfx950+
; only). The dispatch in `handle_valu_vop3p.cpp` therefore admits two
; targets and refuses the rest:
;
;   - gfx1250 (`hasTensorOps`): emit the native
;     `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4` intrinsic.
;   - gfx950  (`hasGfx950Insts`): rewrite to the gfx950 scaled MFMA via
;     `wmma_lowering.cpp::emitWMMAScaleF8F6F4toMFMA` (Wave32→Wave64
;     redistribution + intrinsic swap; no software scale-exponent emul).
;   - Everything else (gfx942 with `hasMFMA == true` but
;     `hasGfx950Insts == false`, gfx900, gfx10/11 base, ...): refuse
;     loudly via `RaiseFailure::unsupportedInstructionForm`.
;
; A WMMA-scale → multi-MFMA decomposition with software scale-exponent
; application is *possible* on gfx942 but not implemented; per the
; user-rules (no silent fallbacks), the handler surfaces the capability
; gap rather than guessing.  The companion gfx950-acceptance RUN line
; at the bottom of this fixture pins the gfx950 cross-target path; the
; same-target gfx1250 → gfx1250 RUN line below pins the native-emit
; path.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code —
;      the test passes only when raise_cli actually failed).
;   2. The stderr diagnostic names the offending mnemonic
;      (`v_wmma_scale_f32_16x16x128_f8f6f4`), the encoding format
;      (`VOP3P`), and the architectural-mismatch detail. The handler
;      routes the failure through
;      `RaiseFailure::unsupportedInstructionForm(di, "VOP3P", detail)`, which
;      raise_cli then formats as `raise_cli: kernel '<name>' failed
;      to raise: <mnemonic> [<format>] @offset=0x... :: <detail>`
;      (raise_cli.cpp). Both the format bucket and the detail text
;      are pinned so that drift in either surfaces a meaningful
;      FileCheck failure pointing at the exact change.

; STDERR: raise_cli: kernel 'wmma_scale_f32_16x16x128_f8f6f4_kernel' failed to raise:
; STDERR-SAME: v_wmma_scale_f32_16x16x128_f8f6f4
; STDERR-SAME: [VOP3P]
; STDERR-SAME: hasTensorOps
; STDERR-SAME: int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4
; STDERR-SAME: hasGfx950Insts
; STDERR-SAME: int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
; STDERR-SAME: gfx942

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=IR
;
; Lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4 (gfx1250 RDNA4
; VOP3PX2 opcode 0x033, ScaledWMMA family) — same-target
; (gfx1250 -> gfx1250) intrinsic-emit path. Pins the principled lift
; in transpiler/handle_valu_vop3p.cpp under
; CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4 when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `wmma_scale_f32_16x16x128_f8f6f4.ll`, which pins the cross-target
; (gfx942) loud refusal.
;
; 18 MC pseudos collapse onto this single CanonicalOp (9 mantissa pairs
; `{f4,f6,f8} A × {f4,f6,f8} B` × `_twoaddr`/`_threeaddr`), per
; `WMMA_F8F6F4_Profiles` in VOP3PInstructions.td:1908. The per-matrix
; dword count is encoded by the opcode's `_fA_fB_w32_*` suffix
; (f8 → 16 dwords, f6 → 12, f4 → 8) and the in-family element
; distinction (BF8 vs FP8 within f8; BF6 vs FP6 within f6) lives in
; the `matrix_a_fmt` / `matrix_b_fmt` named-immediate operands
; (`enum MatrixFMT { FP8=0, BF8=1, FP6=2, BF6=3, FP4=4 }`,
; SIDefines.h:1052-1058). The HIP fixture compiles to the
; `_f8_f8_w32_threeaddr` MC pseudo with `matrix_a_fmt:MATRIX_FMT_BF8`
; and `matrix_b_fmt:MATRIX_FMT_FP8` (default) — the same shape as the
; failing kerneldex GEMMs (B8F8 / F8B8 ID73f0 contractions).
;
; The native intrinsic `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4`
; (IntrinsicsAMDGPU.td:4138, class
; `AMDGPUWmmaScaleIntrinsicModsC<llvm_i32_ty>`) takes 14 args:
;
;   <8 x float> llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4(
;       i32 matrix_a_fmt, <NA x i32> A,
;       i32 matrix_b_fmt, <NB x i32> B,
;       i16 c_mod, <8 x float> C,
;       i32 matrix_a_scale, i32 matrix_a_scale_fmt, i32 scale_src0,
;       i32 matrix_b_scale, i32 matrix_b_scale_fmt, i32 scale_src1,
;       i1 matrix_a_reuse, i1 matrix_b_reuse)
;
; Overloaded on D, A and B element vector types, so the f8_f8 form
; mangles to `.v8f32.v16i32.v16i32`. The handler decodes named
; operands via `AMDGPU::getNamedOperandIdx` (`matrix_a_fmt`,
; `matrix_b_fmt`, `matrix_a_scale`, `matrix_b_scale`,
; `matrix_a_scale_fmt`, `matrix_b_scale_fmt`, `scale_src0`,
; `scale_src1`, `matrix_a_reuse`, `matrix_b_reuse`,
; `src2_modifiers`) so any future TableGen reshuffle of the scaled-
; WMMA Ins64 layout flows in for free.
;
; INVARIANTS PINNED:
;
;   1. The native gfx1250 scaled-WMMA intrinsic is emitted (NOT a
;      fallback to MFMA / non-scaled WMMA / a different K-width).
;      The defining marker is the intrinsic name
;      `llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4` with mangled
;      types `.v8f32.v16i32.v16i32` reflecting the f8_f8 fragment
;      shape from the HIP fixture.
;
;   2. The `matrix_a_fmt` arg is `i32 1` (MATRIX_FMT_BF8) and
;      `matrix_b_fmt` is `i32 0` (MATRIX_FMT_FP8 default) — exactly
;      what the disassembled HIP fixture shows
;      (`matrix_a_fmt:MATRIX_FMT_BF8`, matrix_b_fmt omitted at default
;      0). The accumulator type is `<8 x float>` and the A/B fragment
;      types are `<16 x i32>` (the f8 family width).
;
;   3. The `scale_src0` and `scale_src1` slots carry the kernel's
;      runtime VGPR-loaded scale-source values, NOT immediates —
;      pinned via `i32 %{{.+}}` so any regression that hard-codes
;      scales to 0 surfaces immediately.
;
;   4. The reuse args use the canonical defaults: `i1 false` for
;      `matrix_a_reuse` / `matrix_b_reuse` (matches what the HIP
;      builtin emits when `_Constant bool` reuse args are passed
;      `false`).
;
; NEGATIVE PINS:
;
;   * NO call to `llvm.amdgcn.mfma.scale.*` — the cross-target gfx942
;     decomposition path is unimplemented and would mean the
;     same-target lift silently mis-dispatched.
;   * NO call to the non-scaled `llvm.amdgcn.wmma.f32.16x16x128.*`
;     intrinsic — a regression that drops the scale-source operands
;     would land here.
;   * NO call to a different K-width WMMA intrinsic
;     (`16x16x32`, `16x16x64`, `16x16x4`) — would indicate cross-K
;     dispatch confusion.

; IR-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(

; The native gfx1250 scaled-WMMA intrinsic, with the f8_f8 fragment
; shape reflected in the mangled types `.v8f32.v16i32.v16i32`.
; matrix_a_fmt = MATRIX_FMT_BF8 (1), matrix_b_fmt = MATRIX_FMT_FP8
; (0), C_mod = 0, scale {a,b}_scale = 0, scale {a,b}_scale_fmt = 0,
; scale_src0 / scale_src1 are runtime VGPR values, reuse a/b = false.
; IR: %wmma_scale{{[0-9]*}} = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(
; IR-SAME: i32 1, <16 x i32> %{{[^,]+}},
; IR-SAME: i32 0, <16 x i32> %{{[^,]+}},
; IR-SAME: i16 0, <8 x float> %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i1 false, i1 false)

; Negative: no MFMA scale fallback (K=128 scaled-WMMA → MFMA
; decomposition is unimplemented in wmma_lowering.cpp).
; IR-NOT: @llvm.amdgcn.mfma.scale.

; Negative: no non-scaled K=128 WMMA dispatch (would drop the scale
; operands).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4(

; Negative: no other-K WMMA dispatch (cross-K dispatch confusion).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x32.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x64.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x4.

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=IR_GFX950
;
; Cross-target lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4
; (gfx1250 RDNA4 source) → gfx950 (CDNA4 target). Pins the
; `emitWMMAScaleF8F6F4toMFMA` path in `wmma_lowering.cpp` dispatched
; by `ctx.targetIsa.hasGfx950Insts` in `handle_valu_vop3p.cpp` under
; `CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4`. The lift runs Wave32 →
; Wave64 lane redistribution, applies C_mod via IR fneg/fabs on the
; redistributed accumulator, and emits one or two calls (depending
; on `numSourceWavesPerTarget()`) to the gfx950 native scaled MFMA
; intrinsic `int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`
; (IntrinsicsAMDGPU.td:3694), which covers the same K=128 F8/F6/F4
; matmul shape on gfx950's MAI pipe.
;
; The discriminator predicate is `hasGfx950Insts`, NOT `hasMFMA` —
; gfx942 also has `hasMFMA == true` but lacks the scaled F8F6F4 MFMA
; family, so the dispatch must gate on `FeatureGFX950Insts`
; (`isa_profile.hpp::ISAProfile::hasGfx950Insts`). The companion
; gfx942-refusal RUN line at the top of this fixture pins that
; gfx942 still refuses; this RUN line pins that gfx950 accepts.
;
; IR_GFX950-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(
;
; The gfx950 cross-target MFMA-scaled intrinsic. Mangled types
; reflect the `<8 x i32>` A/B fragment width chosen by
; `emitWMMAScaleF8F6F4toMFMA` (widest case; `cbsz`/`blgp` narrow
; the active subset for f6/f4) and the `<4 x f32>` accumulator.
; IR_GFX950: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.

; Negative: no native gfx1250 scaled-WMMA intrinsic in the gfx950
; IR — would mean the cross-target dispatch silently mis-fired.
; IR_GFX950-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4

; Negative: no non-scaled WMMA family in the gfx950 IR — would
; indicate dropped scale operands or cross-K dispatch confusion.
; IR_GFX950-NOT: @llvm.amdgcn.wmma.f32.16x16x

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
