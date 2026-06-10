; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=wmma_f32_16x16x4_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Cross-target lift fixture for v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4
; VOP3P opcode 0x05D) lowered to gfx942 (CDNA3) via
; `emitWMMAtoMFMA_F32_16x16x4` (transpiler/wmma_lowering.cpp).
; Companion fixture to `wmma_f32_16x16x4_f32_same_target.ll`, which
; pins the same-target (gfx1250 -> gfx1250) native-intrinsic-emit
; path.
;
; This SemOp stands ALONE from the K=32 (16-bit) and K=64 (8-bit)
; WMMA family (`SemOp::V_WMMA_F32_16x16x32_*`,
; `SemOp::V_WMMA_F32_16x16x64_*`, `SemOp::V_WMMA_I32_16x16x64_IU8`)
; because:
;
;   * The per-Wave32-lane A/B fragment is `<2 x f32>` (2 VGPRs, not
;     the 8-VGPR <16 x t> / <8 x i32> fragment used by the K=32 /
;     K=64 family).
;   * The target MFMA is `mfma_f32_16x16x4f32` with signature
;     `(float A, float B, <4 x float> C)` — so the per-lane MFMA
;     input is ONE f32 dword, not a `<4 x t>` / i64 pack.
;   * K=4 matches the MFMA K dimension exactly, so each Wave32
;     group dispatches ONE MFMA call (not 2 chained like the
;     K=32/K=64 path where each group splits into 2× K=16 calls).
;
; The rest of the lowering is shared: ds.bpermute-driven lane
; redistribution via the same `redistributeAcc` / `collectResult`
; helpers that drive the K=32/K=64 decomposition.
;
; LAYOUT CONTRACT (corpus-verifiable, not just IR-pinned).
; The per-lane (i, k) layout assumed for the gfx1250 V_WMMA_F32_16x16x4_F32
; A/B inputs is the natural extrapolation of the documented K=32/K=64
; pattern (see `wmma_lowering.cpp`):
;
;   i = lane % 16
;   k = 2*floor(lane/16) + GPR
;
; Validated against the hipBLASLt/Tensile `SS_SS_HA_Bias_SAV_UA`
; family of f32 GEMM kernels (macro-tile `MT32x32x16`, WMMA shape
; `MI16x16x1`, wave32) — the corpus kernels that surfaced this
; gap in cross-target validation. A lifted gfx942 output comparison
; against a gfx1250 gold reference would catch layout regressions, not
; a silent wrong answer.
;
; INVARIANTS PINNED:
;
;   1. Cross-target lift dispatches to mfma_f32_16x16x4f32 (NOT
;      the native gfx1250 WMMA intrinsic, which would fail at
;      gfx942 codegen). Defining marker: the per-group MFMA
;      intrinsic name `mfma.f32.16x16x4f32`.
;
;   2. The MFMA signature is `(float, float, <4 x float>)` — NOT
;      the K=16 / K=32 shapes (which would use `<4 x t>` / i64 A/B
;      and would indicate cross-K dispatch confusion).
;
;   3. ONE MFMA call per Wave32 virtual group (2 total across
;      both groups). The K=32/K=64 family produces 2 MFMAs per
;      group (4 total); any extra `mfma.f32.16x16x4f32` call here
;      would indicate an incorrect K-decomposition.
;
;   4. The accumulator is packed into `<4 x float>` (standard
;      gfx942 MFMA C/D layout) from the WMMA's `<8 x float>`
;      per-Wave32-lane fragment via the shared `redistributeAcc`
;      path.
;
;   5. There is NO in-file `@llvm.amdgcn.strict.wwm*` marker around
;      the redistribute -> MFMA -> collect chain. The partial-wave
;      correctness guarantee (all 64 W64 lanes execute the MFMA
;      pipeline even on a `blockDim == 32` launch) is provided by
;      a single kernel-entry `@llvm.amdgcn.init_whole_wave` call
;      emitted by `WaveNativeProjection::emitInitialExec` — that
;      intrinsic sets hardware EXEC = -1 for the remainder of the
;      function, which the redistribute / MFMA / collect chain
;      relies on as a kernel-wide ambient. Side-effect gating
;      against the *logical* wave32 active mask stays on
;      `emitUnderExec` diamonds at the consuming VGPR stores,
;      completely decoupled from hardware EXEC. See
;      `hotswap/docs/wave-size-translation.md` §5.6.1 for the
;      register-allocator rationale behind moving the EXEC = -1
;      guarantee to kernel entry instead of wrapping per-MFMA
;      WWM brackets.
;
; NEGATIVE PINS:
;
;   * NO call to the native gfx1250 WMMA intrinsic
;     `llvm.amdgcn.wmma.f32.16x16x4.f32` — target is gfx942 which
;     does NOT have hasTensorOps, so any native emit here would
;     fail at codegen (and silently mis-lower if the codegen
;     somehow accepted it).
;   * NO `mfma.f32.16x16x16f16` / `mfma.f32.16x16x16bf16*` /
;     `mfma.f32.16x16x32_*` — K=16 / K=32 MFMA calls would mean
;     the K=4 SemOp fell through to the K=32/K=64 path.
;   * Exactly 2 `mfma.f32.16x16x4f32` calls (not 4) — the K=4
;     decomposition is 1 MFMA per Wave32 virtual group.
;   * Zero `@llvm.amdgcn.strict.wwm*` calls anywhere in the kernel
;     (the init_whole_wave entry ambient replaced them; leaving a
;     stray WWM marker would revert the K=4 path to the old
;     regalloc-bottlenecked shape).
;   * Exactly ONE `@llvm.amdgcn.init_whole_wave` call at function
;     entry (the EXEC alloca seed emitted by
;     `WaveNativeProjection::emitInitialExec`). More than one
;     would indicate a wave-projection bug; zero would indicate
;     the projection hook is not wired into `AllocaRegFile::init`.

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x4_f32_kernel(

; Kernel-entry EXEC virtualisation: exactly one init_whole_wave
; call, emitted by `WaveNativeProjection::emitInitialExec`. This
; replaces the prior per-MFMA `strict.wwm` wrap.
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()

; First group pass (Wave32 group 0: W64 lanes 0-31).
; The MFMA call takes scalar `float` for A and B, and `<4 x float>`
; for the accumulator. Pin that shape explicitly.
; CHECK: %mfma = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; Second group pass (Wave32 group 1: W64 lanes 32-63). LLVM
; uniquifies the value name because `%mfma` is in use, so we
; allow any integer suffix.
; CHECK: %mfma{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; Exactly 2 MFMA calls (one per Wave32 virtual group) — NOT 4.
; Anchored AFTER the per-group positive checks above, so any extra
; call would surface here as an unexpected match.
; CHECK-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(

; No in-file WWM markers around the redistribute / MFMA / collect
; chain; the kernel-entry init_whole_wave above is the sole
; hardware-EXEC-virtualisation signal.
; CHECK-NOT: call i32 @llvm.amdgcn.strict.wwm.i32(
; CHECK-NOT: call {{.*}} @llvm.amdgcn.strict.wwm
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave

; Negative: no native gfx1250 WMMA intrinsic (we are on gfx942).
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x4.f32

; Negative: no cross-K MFMA intrinsics (would indicate dispatch
; confusion with the K=32/K=64 WMMA family).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32_

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=wmma_f32_16x16x4_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=MODREP
;
; K=4 WMMA cross-target lift — MODULOREPLICATION regression pin.
; Runs the same `.co` as `wmma_f32_16x16x4_f32.ll` but with
; `--disable-wave-native` so `ModuloReplicationProjection`
; engages instead of `WaveNativeProjection`.
;
; Why this sibling exists
; =======================
;
; Pre-Session-8 (2026-04-23) `handle_valu_vop3p.cpp`'s K=4 arm
; refused the MODREP lift with `RaiseFailure::unsupportedInstructionForm`
; whenever `kernelHasPermlane16Swap` was true — a conservative
; sidecar refusal added during the matmul_fp16 multi-WMMA
; investigation.  The root cause turned out to be one layer up
; (the symmetric `v_permlane16_swap_b32` lift; see
; `handle_valu_cross_lane.cpp::emitPermLaneSwapEmulation` and
; matrix-translation.md §12.4.7), so the refusal was dropped and
; the K=4 MFMA redistribution runs in both projections.  No
; corpus kernel today exercises K=4 under MODREP — the gap
; between "K=4 MODREP drops through" and "K=4 MODREP is tested"
; is what this fixture closes.
;
; Invariants pinned (deliberately a subset of the WaveNative
; sibling's contract — the projection-specific fixture asserts
; the shared `emitWMMAtoMFMA_F32_16x16x4` structure, not the
; projection-specific EXEC-virtualisation story):
;
;   1. The lift SUCCEEDS (pre-Session-8 it refused loudly; a
;      regression to the refusal gate would fail the CHECK-LABEL
;      because the kernel wouldn't be emitted).
;
;   2. Exactly ONE `mfma.f32.16x16x4f32` call.  Under MODREP
;      `numSourceWavesPerTarget() == 1` (phantom-lane regime
;      single-source-wave projection), so
;      `emitWMMAtoMFMA_F32_16x16x4` emits ONE group pass — the
;      WaveNative sibling's two-pass layout is projection-
;      specific and must NOT surface under MODREP.
;
;   3. The MFMA signature is `(float, float, <4 x float>)` — NOT
;      the K=16 / K=32 packed shapes.  A dispatch misroute into
;      the K=32/K=64 WMMA arm would surface as a differently-
;      typed MFMA call.
;
;   4. MODREP-specific: NO `@llvm.amdgcn.init.whole.wave` call.
;      `WaveNativeProjection::emitInitialExec` emits exactly one
;      at kernel entry to make HW EXEC=-1 a kernel-wide ambient;
;      `ModuloReplicationProjection` does NOT (HW EXEC remains
;      the source-active mask).  Per-MFMA `strict.wwm` brackets
;      (emitted by `wrapAsWWMValue` under MODREP) are the
;      wave-scoped EXEC=-1 mechanism this projection uses.
;
;   5. At least one `@llvm.amdgcn.strict.wwm.*` call — the
;      MODREP-specific WWM bracket around the lane
;      redistribution / MFMA / collect chain that substitutes for
;      the WaveNative init_whole_wave ambient.  A regression that
;      loses both would ship a silently miscompiled kernel
;      (the MFMA would execute under the source-active EXEC
;      mask, leaving the phantom lanes' MFMA inputs undef).

; MODREP-LABEL: define amdgpu_kernel void @wmma_f32_16x16x4_f32_kernel(

; Exactly one MFMA call — MODREP is single-source-wave.
; MODREP: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; MODREP-specific strict.wwm wrap around the MFMA result.  The
; projection-specific WWM bracket (`wrapAsWWMValue`) substitutes
; for the WaveNative init_whole_wave kernel-entry ambient.  A
; regression that loses both would ship a silently miscompiled
; kernel (the MFMA would execute under the source-active EXEC
; mask, leaving the phantom lanes' MFMA inputs undef).
; MODREP: call {{.*}} @llvm.amdgcn.strict.wwm

; Exactly one MFMA call (anchored AFTER the positive check).
; MODREP-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(

; MODREP-specific: NO kernel-entry init_whole_wave (that's the
; WaveNative sibling's signature).
; MODREP-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave

; Negative: no native gfx1250 WMMA intrinsic (gfx942 target).
; MODREP-NOT: @llvm.amdgcn.wmma.f32.16x16x4.f32

; Negative: no cross-K MFMA intrinsics.
; MODREP-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16
; MODREP-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16
; MODREP-NOT: @llvm.amdgcn.mfma.f32.16x16x32_

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx1250 --emit-ir=wmma_f32_16x16x4_f32_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4 VOP3P
; opcode 0x05D) — same-target (gfx1250 -> gfx1250) intrinsic-emit
; path. Pins the principled lift in transpiler/handle_valu_vop3p.cpp
; under SemOp::V_WMMA_F32_16x16x4_F32 when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `wmma_f32_16x16x4_f32.ll`, which pins the cross-target (gfx942)
; loud refusal.
;
; This SemOp stands ALONE from the K=32 (16-bit) and K=64 (8-bit)
; WMMA family because (a) the per-lane A/B fragment is `<2 x f32>`
; (only 2 dwords) instead of `<16 x t>` (16-bit) or `<8 x i32>`
; (8-bit), and (b) `emitWMMAtoMFMA` in transpiler/wmma_lowering.cpp
; is parameterised on 16-/8-bit element packing and has NO K=4 f32
; codepath. The refusal sibling test pins the gfx942 contract; this
; fixture pins the same-target intrinsic-emit shape.
;
; The native intrinsic `int_amdgcn_wmma_f32_16x16x4_f32` is declared
; inside `AMDGPUWMMAIntrinsicsGFX1250` (gated by `isGFX125xOnly` in
; IntrinsicsAMDGPU.td:4113-4114). The matching call shape is
; `AMDGPUWmmaIntrinsicModsC` (6 args — this K=4 f32 variant has NO
; per-element A_mod / B_mod slots, unlike the 16-/8-bit
; ModsAllReuse / ModsABClamp classes used by the K=32 / K=64 WMMA
; family):
;
;   <8 x float> llvm.amdgcn.wmma.f32.16x16x4.f32(
;       <2 x float> a,
;       <2 x float> b,
;       i16 c_mod,
;       <8 x float> c,
;       i1 a_reuse, i1 b_reuse)
;
; The handler emits the modifier args as `i16 0` / `i1 false` to
; match what the disassembler surfaces for the failing kerneldex
; kernels (clang's `_Constant` builtin args constrain modifiers to
; constants, and the failing GEMMs always emit them at default).
;
; INVARIANTS PINNED:
;
;   1. The native gfx1250 intrinsic is emitted (NOT a fallback to
;      MFMA). The defining marker is the intrinsic name
;      `llvm.amdgcn.wmma.f32.16x16x4.f32` with mangled types
;      `.v8f32.v2f32` reflecting the K=4 fragment shape.
;
;   2. The accumulator type is `<8 x float>` and the A/B fragment
;      type is `<2 x float>`. A regression that misroutes K=4 to a
;      K=32 / K=64 dispatch would emit `<16 x t>` or `<8 x i32>`
;      fragments instead.
;
;   3. The modifier args use the canonical defaults: `i16 0` for
;      c_mod and `i1 false` for matrix_a_reuse / matrix_b_reuse.
;
;   4. The call is 6-args, NOT 8 — pinning against a regression
;      that dispatches the K=4 variant through the 16-bit
;      ModsAllReuse shape (8 args: A_mod, A, B_mod, B, C_mod, C,
;      a_reuse, b_reuse) or the 8-bit-iu8 ModsABClamp shape
;      (8 args incl. clamp).
;
; NEGATIVE PINS:
;
;   * NO call to `llvm.amdgcn.mfma.*` — the MFMA fallback path
;     does not cover K=4 f32 and any such call here would mean the
;     handler silently mis-dispatched.
;   * NO call to a different K-width WMMA intrinsic
;     (`16x16x32` or `16x16x64`) — would indicate cross-K
;     dispatch confusion.

; IR-LABEL: define amdgpu_kernel void @wmma_f32_16x16x4_f32_kernel(

; The native gfx1250 WMMA intrinsic, with the K=4 f32 fragment
; shape reflected in the mangled types `.v8f32.v2f32`. Modifier
; args are defaulted (i16 0 / i1 false) to match what clang's
; `_Constant` builtin args produce for the failing kerneldex GEMMs.
; The call is 6-args matching AMDGPUWmmaIntrinsicModsC (no per-
; element A_mod / B_mod slots).
; IR: %wmma{{[0-9]*}} = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(<2 x float> %{{[^,]+}}, <2 x float> %{{[^,]+}}, i16 0, <8 x float> %{{[^,]+}}, i1 false, i1 false)

; Negative: no MFMA fallback (K=4 f32 has no decomposition path).
; IR-NOT: @llvm.amdgcn.mfma.

; Negative: no other-K WMMA dispatch (cross-K dispatch confusion).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x32.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x64.
; IR-NOT: @llvm.amdgcn.wmma.i32.16x16x64.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_f32_16x16x4_f32_kernel
	.p2align	8
	.type	wmma_f32_16x16x4_f32_kernel,@function
wmma_f32_16x16x4_f32_kernel:            ; @wmma_f32_16x16x4_f32_kernel
; %bb.0:
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[12:13], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_load_b64 s[14:15], s[8:9], 0x0
	s_load_b64 s[16:17], s[10:11], 0x0
	s_load_b256 s[0:7], s[12:13], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[8:9], s[14:15]
	v_mov_b64_e32 v[10:11], s[16:17]
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[10:11], v[0:7]
	v_mov_b32_e32 v8, 0
	s_clause 0x1
	global_store_b128 v8, v[4:7], s[12:13] offset:16
	global_store_b128 v8, v[0:3], s[12:13]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_f32_16x16x4_f32_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 18
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
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           wmma_f32_16x16x4_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .symbol:         wmma_f32_16x16x4_f32_kernel.kd
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
