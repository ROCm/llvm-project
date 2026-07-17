//===- wmma-lowering.h - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_WMMA_LOWERING_H
#define HOTSWAP_TRANSPILER_WMMA_LOWERING_H

#include "llvm/Support/Error.h"

namespace llvm {
class Value;
} // namespace llvm

namespace COMGR::hotswap {

struct RaiseContext;

/// Element type / AB combination of the A/B input fragments to a
/// 16x16x{32,64} WMMA that we are lowering to gfx942 MFMA. Selects
/// the per-MFMA intrinsic AND the per-MFMA bitcast shape; the lane-
/// redistribution math itself is BYTE-IDENTICAL across all variants
/// because every supported WMMA input keeps the same per-Wave32-lane
/// fragment size -- 8 VGPRs (= 32 bytes) on the A side, 8 VGPRs on the
/// B side, and 8 VGPRs of f32 on the C/D side. We factor on the
/// operation, not the opcode.
///
/// Per-variant divergence inside `runGroupPass`:
///
///   F16    -> 2x mfma_f32_16x16x16f16,        AB pack `<4 x half>`,  acc f32
///   BF16   -> 2x mfma_f32_16x16x16bf16_1k,    AB pack `<4 x i16>`,   acc f32
///   FP8_*  -> 2x mfma_f32_16x16x32_<a>_<b>,   AB pack `i64`,         acc f32
///   BF8_*  -> 2x mfma_f32_16x16x32_<a>_<b>,   AB pack `i64`,         acc f32
///   IU8    -> 2x mfma_i32_16x16x32_i8,        AB pack `i64`,         acc i32
///
/// The 2x MFMA decomposition is invariant across variants: a WMMA
/// computes a 16x16 result for K=32 (16-bit) or K=64 (8-bit), and we
/// always split that K dimension in half so each MFMA covers K=16
/// (16-bit MFMA) or K=32 (8-bit MFMA) per call. Per-Wave64-lane MFMA
/// fragment is exactly 2 dwords either way.
///
/// The bf8/fp8 16-bit->i16 / fp8/i8->i64 bitcasts are principled: the
/// CDNA MFMA intrinsics for these element types were defined before
/// the matching first-class LLVM types existed (or, for i8, before
/// AMDGCN had a packed i8 vector type at all), and the storage
/// containers (i16 for bf16, i64 for 8 packed fp8/bf8/i8 bytes) are
/// bit-for-bit identical to the per-lane source dwords.
///
/// The IU8 variant is the only one with a non-f32 accumulator --
/// `runGroupPass` carries the accumulator IR pack type through the
/// dispatch alongside the AB pack type so the MFMA call signature
/// matches the i32-accumulator MFMA intrinsic exactly.
enum class WMMAInputType {
  F16,
  BF16,
  FP8_FP8,
  FP8_BF8,
  BF8_FP8,
  BF8_BF8,
  IU8,
};

/// Lower a Wave32 WMMA to Wave64 MFMA using ds_bpermute lane
/// redistribution.
///
/// This ports the proven approach from the original hotswap prototype
/// (hotswap/transpiler.cpp, WMMA->MFMA translation block) to principled
/// LLVM IR using three intrinsic families:
///
///   @llvm.amdgcn.ds.bpermute  -- cross-lane data reads
///   @llvm.amdgcn.mfma.*       -- hardware matrix multiply (reads all 64 lanes)
///   @llvm.amdgcn.strict.wwm   -- run the redistribute / MFMA / collect
///                               chain with EXEC = -1 so partial-wave
///                               launches (e.g. blockDim == 32 on gfx942)
///                               still update the upper-half lanes' VGPRs
///                               that MFMA and the collect-time bpermute
///                               read back. See wmma-lowering.cpp,
///                               file-header "Whole-wave mode".
///
/// `ds_bpermute` and `v_mfma_*` both READ from all 64 source lanes
/// regardless of EXEC, but their per-lane WRITE is EXEC-gated -- hence
/// the whole-wave-mode wrapper, which forces EXEC = -1 across the
/// collective so every lane's MFMA input / MFMA output / collected
/// result VGPR is written and subsequent cross-lane reads see fresh
/// data rather than stale / poison bits. We emit the WWM marker as
/// eight `strict.wwm.i32` calls (one per result dword), NOT one
/// `strict.wwm.v8i32` call on the packed vector: the backend's
/// `SIPreAllocateWWMRegs` pass reserves an aligned physical VGPR
/// per WWM virtual register, and a `<8 x i32>` WWM operand cannot
/// always be satisfied (`physreg not found for WWM expression`
/// llvm_unreachable) in WMMA-heavy kernels. Per-dword wrapping
/// only needs single-VGPR slots. See the "Whole-wave mode" header
/// in wmma-lowering.cpp for the full argument.
///
/// Source fragment shapes (Wave32, 8 VGPRs per side either way):
///   16-bit (F16 / BF16):  A/B = <16 x {half|bfloat}> per lane
///   8-bit  (FP8 / BF8 ):  A/B = <8 x i32> per lane (32 packed fp8/bf8 bytes)
///   8-bit  (IU8        ):  A/B = <8 x i32> per lane (32 packed i8/u8 bytes)
/// C/D fragment shape:
///   F32-accumulator variants (everything except IU8): <8 x float>
///   IU8 variant:                                       <8 x i32>
///
/// The C and result IR types must match the accumulator side of the
/// dispatched MFMA intrinsic (CDNA distinguishes mfma_f32_* and
/// mfma_i32_* by accumulator element type). Caller is expected to
/// pre-load `c` with the correct vector type for the inputType -- the
/// helper does NOT bitcast; the redistribution and pack code paths
/// operate on dword-extracted scalars and only the final vector
/// reassembly cares about the element type.
///
/// \param a  WMMA source A fragment (8 VGPRs in Wave32, layout per
///           inputType)
/// \param b  WMMA source B fragment (8 VGPRs in Wave32, layout per
///           inputType)
/// \param c  WMMA accumulator fragment (8 VGPRs in Wave32, vector
///           element type matches the dispatched MFMA accumulator --
///           f32 for everything except IU8 which uses i32)
/// \param inputType  selects MFMA intrinsic, per-MFMA AB pack type,
///                   AND the accumulator pack/result element type.
/// \returns  <8 x t> -- result in Wave32 layout, where t = float for
///           every variant except IU8 which returns <8 x i32>. Returns an
///           error for an unsupported source-wave projection.
llvm::Expected<llvm::Value *> emitWMMAtoMFMA(RaiseContext &Ctx, llvm::Value *A,
                                             llvm::Value *B, llvm::Value *C,
                                             WMMAInputType InputType);

/// Lower a Wave32 v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4 VOP3P opcode
/// 0x05D) to Wave64 mfma_f32_16x16x4f32 (gfx942 CDNA3) using
/// ds_bpermute lane redistribution.
///
/// This K=4 f32 variant is STRUCTURALLY DISTINCT from the K=32 / K=64
/// family covered by `emitWMMAtoMFMA` above:
///
///   * A/B fragment per Wave32 lane is `<2 x f32>` (2 VGPRs) -- not
///     the <16 x t> (16-bit element) or <8 x i32> (8-bit element)
///     8-VGPR fragments used by the K=32 / K=64 variants.
///   * Only ONE MFMA call per Wave32 group -- not 2 chained. The
///     source WMMA is already K=4, which exactly matches the target
///     `mfma_f32_16x16x4f32` K dimension; there is no K-tiling to do
///     inside a group.
///   * The MFMA A/B input is a single `float` per Wave64 lane -- not
///     `<4 x t>` (16-bit) or `i64` (8-bit).
///
/// The C/D fragment shape (<8 x f32> per Wave32 lane, Wave32 C-layout
/// equation `i = 8*floor(lane/16) + GPR`) is IDENTICAL to the K=32 /
/// K=64 variants, so the accumulator redistribution and the final
/// result collection reuse the same internal helpers as `emitWMMAtoMFMA`.
///
/// LAYOUT NOTE. The gfx1250 V_WMMA_F32_16x16x4_F32 per-lane A/B
/// (i, k) layout is extrapolated from the documented K=32 / K=64
/// pattern -- the AMD Matrix Instruction Calculator does not yet list
/// the K=4 f32 variant. We assume:
///
///   A/B input (<2 x f32> per lane):
///     i = lane % 16
///     k = 2*floor(lane/16) + GPR
///   Per-lane:
///     Lanes 0-15, GPR 0 -> k=0    GPR 1 -> k=1
///     Lanes 16-31, GPR 0 -> k=2   GPR 1 -> k=3
///
/// This is the unique natural extension of the K=32/K=64 layout
/// documented in wmma-lowering.cpp (lower half holds lower k-range,
/// upper half holds upper k-range; GPR indexes along k within a
/// lane-half). The layout is validated out-of-band by the hipBLASLt
/// Tensile `SS_SS_HA_Bias_SAV_UA` f32 GEMM kernels (macro-tile
/// `MT32x32x16`, WMMA shape `MI16x16x1`, wave32) -- see
/// a cross-target numerical comparison against the gfx1250 reference. A layout
/// mismatch would surface as a numerical regression there, not a
/// silent wrong answer.
///
/// \param a  WMMA source A fragment (<2 x f32> in Wave32)
/// \param b  WMMA source B fragment (<2 x f32> in Wave32)
/// \param c  WMMA accumulator fragment (<8 x f32> in Wave32)
/// \returns  `<8 x float>` -- result in Wave32 C-layout. Returns an error for
///           an unsupported source-wave projection.
llvm::Expected<llvm::Value *> emitWmmAtoMfmaF3216x16x4(RaiseContext &Ctx,
                                                       llvm::Value *A,
                                                       llvm::Value *B,
                                                       llvm::Value *C);

/// Lower a Wave32 v_wmma_scale_f32_16x16x128_f8f6f4 (gfx1250 RDNA4 VOP3PX2)
/// to a Wave64 v_mfma_scale_f32_16x16x128_f8f6f4 (gfx950 CDNA4 VOP3PX) via
/// ds_bpermute lane redistribution + the gfx950 scaled-MFMA intrinsic.
///
/// Structurally simpler than the K=32/K=64 emitWMMAtoMFMA path because the
/// gfx950 MFMA scaled F8F6F4 already covers the full K=128 in one call, so
/// there is no K-decomposition (no chained-MFMA-pair). The lane redistribution
/// is the "halve dwords-per-lane to spread Wave32's data across Wave64"
/// pattern, extended to the variable-width A/B fragment (16/12/8 dwords on the
/// WMMA side for f8/f6/f4; 8/6/4 dwords on the MFMA side after redistribution).
///
/// Layout assumption (extrapolated from the K=32/K=64 documented pattern,
/// pending hardware validation for K=128 F8F6F4):
///
///   WMMA Wave32 lane indexing (per source wave32 group of 32 lanes):
///     - Lanes 0-15 hold the lower-K half of the A/B matrix
///     - Lanes 16-31 hold the upper-K half
///     - Within a lane, the wmmaDwords[0..N-1] array stores K-elements in
///       linear order (no GPR-pair-K interleaving for K=128 -- extension of
///       the simpler-half / second-half decomposition that K=64 uses)
///
///   MFMA Wave64 lane indexing:
///     - Lane group LG_g (g=0..3, 16 lanes each) holds K-quarter g
///     - Each LG's mfmaDwords[i] reads from WMMA's
///         lower-W32-half if g is even, upper-W32-half if g is odd
///         first-half wmmaDwords[0..mfmaDw-1] if g/2==0
///         second-half wmmaDwords[mfmaDw..2*mfmaDw-1] if g/2==1
///
/// Operand mapping (gfx1250 WMMA -> gfx950 MFMA):
///   matrix_a_fmt (i32: F8/F6/F4 enum)   -> cbsz
///   matrix_b_fmt                        -> blgp
///   matrix_a_scale + matrix_a_scale_fmt -> op_sel_A (2-bit; first cut: 0)
///   scale_src0 (i32 VGPR)               -> scale src A (direct)
///   matrix_b_scale + matrix_b_scale_fmt -> op_sel_B
///   scale_src1                          -> scale src B
///   C_mod (i16: 0=none, 1=neg, 2=abs, 3=neg(abs)) -> apply via fneg/fabs
///                                                   on C in IR before MFMA
///   matrix_a_reuse, matrix_b_reuse      -> dropped (perf hint, not
///   correctness)
///
/// Returns a null Value if the helper cannot lower the call (e.g. unsupported
/// fragment width) -- caller produces an unsupportedInstructionForm refusal --
/// or an error for an unsupported source-wave projection.
///
/// \param a       WMMA A fragment (`<aDwords x i32>` per Wave32 lane;
///                aDwords = 16/12/8 for f8/f6/f4)
/// \param b       WMMA B fragment (`<bDwords x i32>` per Wave32 lane)
/// \param c       WMMA accumulator (`<8 x f32>` per Wave32 lane, same as
///                the K=32/K=64 family)
/// \param matrixAFmt, matrixBFmt   per-matrix format selectors (i32 const)
/// \param cMod    accumulator modifier (i16 const: 0/1/2/3 =
/// none/neg/abs/neg(abs))
/// \param matrixAScale, matrixAScaleFmt, scaleSrc0   A-side scale operands
/// \param matrixBScale, matrixBScaleFmt, scaleSrc1   B-side scale operands
/// \param aDwords, bDwords   per-Wave32-lane WMMA dword counts (16/12/8)
/// \returns       `<8 x float>` in Wave32 D-layout (caller writes back via
///                ctx.writeRegVec); null Value on unrecoverable shape error;
///                an error for an unsupported source-wave projection.
llvm::Expected<llvm::Value *> emitWMMAScaleF8F6F4toScaledMFMA(
    RaiseContext &ctx, llvm::Value *a, llvm::Value *b, llvm::Value *c,
    llvm::Value *matrixAFmt, llvm::Value *matrixBFmt, llvm::Value *cMod,
    llvm::Value *matrixAScale, llvm::Value *matrixAScaleFmt,
    llvm::Value *scaleSrc0, llvm::Value *matrixBScale,
    llvm::Value *matrixBScaleFmt, llvm::Value *scaleSrc1, unsigned aDwords,
    unsigned bDwords);

/// Cross-target gfx1250 -> gfx942 lowering for
/// `v_wmma_scale_f32_16x16x128_f8f6f4`. gfx942 has only the unscaled K=32
/// fp8/bf8 MFMA family, so this decomposes K=128 into 4 K=32 MFMAs, widens
/// FP6/BF6/FP4 to FP8/BF8 in-line, and applies the per-K-block scale via
/// fmuladd. Covers {FP8,BF8,FP6,BF6,FP4}^2, scale formats E8M0 / E4M3
/// (E5M3 deferred). Returns `<8 x float>` in wave32 D-layout or an error for
/// unsupported configurations / invariant violations.
llvm::Expected<llvm::Value *> emitWMMAScaleF8F6F4toMFMA(
    RaiseContext &ctx, llvm::Value *a, llvm::Value *b, llvm::Value *c,
    llvm::Value *matrixAFmt, llvm::Value *matrixBFmt, llvm::Value *cMod,
    llvm::Value *matrixAScale, llvm::Value *matrixAScaleFmt,
    llvm::Value *scaleSrc0, llvm::Value *matrixBScale,
    llvm::Value *matrixBScaleFmt, llvm::Value *scaleSrc1, unsigned aDwords,
    unsigned bDwords);

} // namespace COMGR::hotswap

#endif
