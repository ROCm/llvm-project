//===- rewrite-cross-lane-divergent.cpp - Hotswap transpiler --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rewrite-cross-lane-divergent.h"

#include "SIDefines.h" // llvm::AMDGPU::DPP::DppCtrl
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// ============================================================================
// Use-chain classifier
// ============================================================================
//
// Forward-walk the transitive uses of a cross-lane primitive's result.
// If any use reaches an SGPR-constrained operand, the backend will
// re-introduce `v_readfirstlane_b32` on the rewrite's `ds_bpermute`
// output and collapse the per-source-wave distinction the rewrite
// exists to preserve. Such kernels must be refused at raise time
// rather than emitted with a silently-broken rewrite.
//
// Over-approximation policy: unknown users (unrecognised intrinsics,
// inline asm with unrecognised constraints, non-instruction users
// like constant expressions) are classified as SGPR-forced. The
// project's no-silent-miscompile rule trades false positives for
// soundness. Each time the classifier refuses a kernel that would
// actually be VGPR-safe, the right fix is to extend one of the
// `is*Intrinsic*` tables below, not to relax the default.

enum class UseChainVerdict {
  VGPRSafe,
  SGPRForced,
};

// Per-instruction result: an intrinsic call either
//   (a) forces SGPR on the input at this operand index (refuse),
//   (b) is a VGPR-safe sink -- the call consumes our value as a VGPR
//       input and its result is a fresh value we do not need to
//       track further (e.g. a memory-store intrinsic),
//   (c) is a VGPR-safe propagator -- the call consumes our value as
//       a VGPR input and its result may still carry the per-source-
//       wave state into downstream uses (e.g. a VALU arithmetic
//       intrinsic), so we continue the walk through the result's
//       users.
enum class IntrinsicRole {
  SGPRForced,
  VGPRSafeSink,
  VGPRSafePropagator,
  Unknown,
};

// amdgcn.* intrinsics whose corresponding operand position *must*
// receive an SGPR. When our tracked value reaches any of these at the
// flagged operand index, the backend inserts `v_readfirstlane` on the
// input and the rewrite loses its per-source-wave state. Entries are
// keyed on the intrinsic ID; the operand-index predicate is checked
// per-call-site in `classifyIntrinsicUse`.
//
// This list is deliberately conservative -- it only covers intrinsics
// the raiser is *known* to emit today (see the `git grep
// Intrinsic::amdgcn_` audit in hotswap/docs/learnings.md). An
// unrecognised amdgcn intrinsic falls through the switch below and is
// treated as Unknown (-> SGPRForced).
bool operandForcesSGPR(Intrinsic::ID Id, unsigned OperandIdx) {
  switch (Id) {
  // `amdgcn.readfirstlane(src)` -- the entire point of the intrinsic
  // is scalarising a VGPR into an SGPR. If our value reaches this,
  // it is the explicit scalarisation marker the rewrite was designed
  // to avoid; refuse unconditionally (operandIdx is ignored -- every
  // operand position is an SGPR boundary in practice for this
  // intrinsic's intent).
  case Intrinsic::amdgcn_readfirstlane:
    return true;
  // `s_sendmsg(msg, val)` -- the message index must be a compile-time
  // immediate, but the payload `val` is required to be in an SGPR.
  case Intrinsic::amdgcn_s_sendmsg:
    return OperandIdx <= 1;
  // `s_barrier()` has no operands; including here as a scalar-context
  // sink would be noise. Omitted intentionally.
  default:
    return false;
  }
}

// amdgcn.* intrinsics whose *every* operand accepts a VGPR and whose
// result we need to continue tracking because it may still carry the
// per-source-wave state downstream (arithmetic, rotation, per-lane
// data motion). Must not overlap `operandForcesSGPR` for the same
// (intrinsic, operand) pair.
bool isIntrinsicVGPRSafePropagator(Intrinsic::ID Id) {
  switch (Id) {
  case Intrinsic::amdgcn_ds_bpermute:
  case Intrinsic::amdgcn_ds_swizzle:
  case Intrinsic::amdgcn_update_dpp:
  case Intrinsic::amdgcn_make_buffer_rsrc:
  case Intrinsic::amdgcn_mbcnt_lo:
  case Intrinsic::amdgcn_mbcnt_hi:
  case Intrinsic::amdgcn_perm:
  case Intrinsic::amdgcn_cvt_f32_bf8:
  case Intrinsic::amdgcn_cvt_f32_fp8:
  case Intrinsic::amdgcn_cvt_pk_bf8_f32:
  case Intrinsic::amdgcn_cvt_pk_f32_bf8:
  case Intrinsic::amdgcn_cvt_pk_f32_fp8:
  case Intrinsic::amdgcn_cvt_pk_fp8_f32:
  case Intrinsic::amdgcn_cvt_pkrtz:
  case Intrinsic::amdgcn_cvt_scalef32_pk_fp4_f32:
  case Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp4:
  case Intrinsic::amdgcn_class:
  case Intrinsic::amdgcn_rcp:
  case Intrinsic::amdgcn_rsq:
  case Intrinsic::amdgcn_fract:
  case Intrinsic::amdgcn_sqrt:
  case Intrinsic::amdgcn_trig_preop:
  case Intrinsic::amdgcn_frexp_mant:
  case Intrinsic::amdgcn_frexp_exp:
  case Intrinsic::amdgcn_sffbh:
  case Intrinsic::amdgcn_div_fixup:
  case Intrinsic::amdgcn_div_fmas:
  case Intrinsic::amdgcn_div_scale:
  case Intrinsic::amdgcn_exp2:
  case Intrinsic::amdgcn_log:
  // Whole-wave / whole-quad execution-mode markers: value-copy ops (VGPR
  // in, same-type VGPR out, no SGPR-forced operand) that only constrain the
  // EXEC mask under which the source is computed, so they are VGPR-safe.
  // This pass emits strict.wwm itself (see forceWholeWaveGather /
  // wrapAsWWMValue), so the classifier must walk through it.
  case Intrinsic::amdgcn_strict_wwm:
  case Intrinsic::amdgcn_wwm:
  case Intrinsic::amdgcn_strict_wqm:
  case Intrinsic::amdgcn_wqm:
  case Intrinsic::amdgcn_softwqm:
  // Cross-lane primitives the rewrite also rewrites in this same
  // pass: their post-rewrite shape is `select` / `ds_bpermute`, both
  // VGPR-safe. We can treat them as VGPR-safe propagators pre-
  // rewrite because the symmetry rule guarantees they will be
  // rewritten together.
  case Intrinsic::amdgcn_writelane:
  case Intrinsic::amdgcn_readlane:
    return true;
  // Generic LLVM intrinsics that the AMDGPU backend lowers to per-lane
  // VALU opcodes with no SGPR-forced operand position in any codegen
  // path.  The audit here is tight: only add intrinsics whose AMDGPU
  // lowering is a single VALU instruction (or a VALU-only expansion)
  // accepting all-VGPR operands -- same bar as the `amdgcn_*` cases
  // above.  Unknown generic intrinsics stay SGPR-forced via the default
  // arm below, consistent with the "refuse when uncertain" rule in
  // hotswap/docs/wave-size-translation.md sec. 5.6.3.
  //
  // Why these specifically, and why now: Triton's AMD backend emits
  // these in the fast-reciprocal / rsqrt Newton-iteration expansion
  // surrounding `@llvm.amdgcn.div_fixup` / `div_fmas` / `div_scale`
  // (already whitelisted above) and in the reduction post-processing
  // of layer-norm / softmax (rstd = `1 / sqrt(var + eps)`, softmax
  // normaliser `x / sum`).  All appear on every reduction-bearing
  // Triton kernel's readlane-result use chain.  Pre-audit, the
  // classifier over-approximated them as SGPR-forced, which disabled
  // the rewrite pass on the entire function (all-or-nothing per
  // sec. 5.6.3's "mix of rewritten and preserved sites recreates the
  // Matmul128x128 asymmetric-rewrite fault" rule).  The AMDGPU
  // lowerings are:
  //
  //   * `@llvm.fma.f32`     -> `v_fma_f32` (VOP3, three VGPR sources,
  //     VGPR destination; no SGPR-forced operand).  VALU, per-lane.
  //   * `@llvm.fmuladd.f32` -> `v_fma_f32` / `v_mac_f32` (relaxed-
  //     precision fused-or-split multiply-add; backend chooses per
  //     target and `contract` metadata).  Operand shapes identical to
  //     `fma` -- all VGPR, per-lane.
  //   * `@llvm.sqrt.f32`    -> `v_sqrt_f32` (VOP1, one VGPR source,
  //     VGPR destination).  VALU, per-lane.
  //   * `@llvm.maxnum.f32`  -> `v_max_f32` (VOP2, two VGPR sources,
  //     VGPR destination; IEEE max-num with NaN-propagation rules
  //     handled in the VALU expansion).  VALU, per-lane.  Appears on
  //     every softmax reduction (`m_i = max(m_i-1, x)`) -- the same
  //     position `fma` occupies in layer-norm.
  //   * `@llvm.minnum.f32`  -> `v_min_f32`.  Symmetric with maxnum;
  //     audited for parity so any future min-reducing kernel isn't
  //     blocked on a one-intrinsic gap.
  //   * `@llvm.fabs.f32`    -> `v_and_b32` with a `0x7fffffff` mask
  //     (the backend's preferred `fabs` lowering on modern AMDGPU;
  //     see AMDGPUCombinerHelper.cpp).  Per-lane, all-VGPR.  Shows
  //     up in Triton's reduction prologues when the source is the
  //     absolute-value form of a norm.
  //   * `@llvm.exp2.f32` / `@llvm.log2.f32` and the hardware-specialized
  //     `@llvm.amdgcn.exp2.f32` / `@llvm.amdgcn.log.f32` -> `v_exp_f32` /
  //     `v_log_f32` (VOP1).  Softmax's exponentiation and the `pow` / `log`
  //     decomposition both route here.  Per-lane, all-VGPR.
  //   * `@llvm.floor.f32` / `@llvm.ceil.f32` / `@llvm.trunc.f32` /
  //     `@llvm.roundeven.f32` / `@llvm.rint.f32` / `@llvm.round.f32` /
  //     `@llvm.nearbyint.f32`
  //     -> `v_floor_f32` / `v_ceil_f32` / `v_trunc_f32` /
  //     `v_rndne_f32` (VOP1).  Per-lane, all-VGPR.  Triton emits
  //     these from integer-float conversions and `tl.cdiv`-style
  //     ceiling division.
  //   * `@llvm.copysign.f32` -> `v_bfi_b32` with a sign-bit selector
  //     (backend-canonical).  Per-lane, all-VGPR.
  //   * `@llvm.smin.i32` / `@llvm.smax.i32` / `@llvm.umin.i32` /
  //     `@llvm.umax.i32` -> `v_min_i32` / `v_max_i32` / `v_min_u32` /
  //     `v_max_u32`.  Integer lane-parallel min/max; per-lane,
  //     all-VGPR.
  //   * `@llvm.abs.i32` -> `v_sub_i32` / `v_max_i32` pair (backend
  //     expansion).  Per-lane, all-VGPR.
  //   * `@llvm.ctpop.i32` / `@llvm.ctlz.i32` / `@llvm.cttz.i32` /
  //     `@llvm.bitreverse.i32` -> per-lane bit-counting / bit-reverse
  //     VALU instructions.  All-VGPR.
  //   * `@llvm.fshl.i32` / `@llvm.fshr.i32` -> `v_alignbit_b32`
  //     (funnel shift).  Per-lane, all-VGPR.
  //
  // All carry per-source-wave state through unchanged (SIMT per-lane
  // math), so the forward walk must continue past them -- hence
  // `VGPRSafePropagator` rather than `VGPRSafeSink`.  When LLVM ever
  // routes one of these operands through an SGPR-constrained form
  // (none exists today), extending `operandForcesSGPR` above would
  // shadow the per-operand entry back to SGPR-forced without having
  // to remove the intrinsic from this list.
  //
  // Intrinsics deliberately NOT whitelisted (require additional
  // audit / may decompose through an SGPR-forced helper): trig
  // functions (`sin` / `cos`), vector-reduction intrinsics
  // (`vector.reduce.*`), `experimental.constrained.*` variants,
  // anything that lowers to a library call.  Add here only after
  // confirming the AMDGPU lowering is a single per-lane VALU
  // instruction (or a VALU-only expansion) with no SGPR-forced
  // operand in any codegen path.
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::maxnum:
  case Intrinsic::minnum:
  case Intrinsic::maximumnum:
  case Intrinsic::minimumnum:
  // IEEE-754 2019 NaN-propagating max/min.  Same per-lane VGPR-only
  // operand shape as maxnum/minnum (the only semantic delta is NaN
  // propagation, which is handled inside the VALU expansion on every
  // arch we lower to -- gfx9 family expands to a v_max + NaN-fixup
  // VALU sequence, gfx12 has native v_maximum/v_minimum).  No
  // SGPR-forced operand in any codegen path; safe to whitelist as a
  // VGPRSafePropagator pass-through, identical to maxnum/minnum.
  // Used by the v_maximum3_f32 / v_minimum3_f32 handlers in
  // handle-valu.cpp (gfx12 ternary IEEE max/min lowered as chained
  // 2-source maximum/minimum).
  case Intrinsic::maximum:
  case Intrinsic::minimum:
  case Intrinsic::fabs:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  // `@llvm.ldexp.f32.i32` -> `v_ldexp_f32` (VOP2; `x * 2^n` with an
  // integer exponent).  Softmax's normalisation routes through ldexp
  // when the backend's exp2/pow decomposition picks it (LLVM r202+
  // converts `exp(x) * 2^k` patterns there).  Per-lane, all-VGPR.
  case Intrinsic::ldexp:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::roundeven:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::nearbyint:
  case Intrinsic::copysign:
  case Intrinsic::smin:
  case Intrinsic::smax:
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::umul_with_overflow:
  case Intrinsic::abs:
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::bitreverse:
  case Intrinsic::fshl:
  case Intrinsic::fshr:
    return true;
  default:
    return false;
  }
}

// amdgcn.* intrinsics that consume our value as a VGPR input and
// produce a result (or no result) we do not need to track further --
// the result either is unrelated to our value's per-source-wave
// identity (memory-store result, WMMA / MFMA accumulator -- which
// contains our value mixed with many other lanes' values, so
// "refuse downstream if WMMA output itself is SGPR-forced" is too
// conservative) or has no users the classifier would care about.
bool isIntrinsicVGPRSafeSink(Intrinsic::ID Id) {
  switch (Id) {
  // Stores: value consumed, no IR result to propagate.
  case Intrinsic::amdgcn_raw_buffer_store:
  case Intrinsic::amdgcn_raw_ptr_buffer_load:
  case Intrinsic::amdgcn_raw_ptr_buffer_store:
  case Intrinsic::amdgcn_tensor_store_from_lds:
  case Intrinsic::amdgcn_global_load_async_to_lds_b8:
  case Intrinsic::amdgcn_global_load_async_to_lds_b32:
  case Intrinsic::amdgcn_global_load_async_to_lds_b64:
  case Intrinsic::amdgcn_global_load_async_to_lds_b128:
  case Intrinsic::amdgcn_tensor_load_to_lds:
  case Intrinsic::amdgcn_global_prefetch:
  case Intrinsic::amdgcn_flat_prefetch:
  case Intrinsic::amdgcn_s_barrier:
    return true;
  // Ballot: `ballot(i1 pred)` is a proper cross-lane REDUCTION that
  // packs every active lane's predicate bit into the result mask
  // without inserting `v_readfirstlane` on the input. The mask is
  // wave-uniform by construction (same bit pattern on every lane),
  // and downstream uses consume it as EXEC manipulation / flow
  // control / mask arithmetic -- none of which re-introduces a per-
  // source-wave collapse. Classifying `ballot` as a sink (rather
  // than propagating through its result) matches the AMDGPU
  // divergence analysis' own treatment and keeps the classifier
  // from over-refusing kernels that ballot an `icmp` on a rewritten
  // writelane/readlane result (common Triton reduction shape).
  case Intrinsic::amdgcn_ballot:
    return true;
  // MFMA / WMMA: accumulator inputs are VGPR. The result is a VGPR
  // fragment distributed across lanes; downstream uses of the
  // accumulator flow through `select` reconstruction or memory
  // stores -- both VGPR-safe paths. Treating MFMA/WMMA as a sink
  // (rather than a propagator) keeps the classifier linear in the
  // kernel size; a theoretical "MFMA output lane-swapped into an
  // SGPR-forced consumer" shape would require re-classification on
  // the result, which no corpus kernel is known to need today.
  case Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k:
  case Intrinsic::amdgcn_mfma_f32_16x16x16f16:
  case Intrinsic::amdgcn_mfma_f32_16x16x1f32:
  case Intrinsic::amdgcn_mfma_f32_16x16x2bf16:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_bf16:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_f16:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8:
  case Intrinsic::amdgcn_mfma_f32_16x16x4f16:
  case Intrinsic::amdgcn_mfma_f32_16x16x4f32:
  case Intrinsic::amdgcn_mfma_f32_16x16x8_xf32:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_bf16:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8:
  case Intrinsic::amdgcn_mfma_f32_32x32x1f32:
  case Intrinsic::amdgcn_mfma_f32_32x32x2bf16:
  case Intrinsic::amdgcn_mfma_f32_32x32x2f32:
  case Intrinsic::amdgcn_mfma_f32_32x32x4f16:
  case Intrinsic::amdgcn_mfma_f32_32x32x4_xf32:
  case Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k:
  case Intrinsic::amdgcn_mfma_f32_32x32x8f16:
  case Intrinsic::amdgcn_mfma_f32_4x4x1f32:
  case Intrinsic::amdgcn_mfma_f32_4x4x2bf16:
  case Intrinsic::amdgcn_mfma_f32_4x4x4f16:
  case Intrinsic::amdgcn_mfma_i32_16x16x32_i8:
  case Intrinsic::amdgcn_mfma_i32_16x16x4i8:
  case Intrinsic::amdgcn_mfma_i32_32x32x16_i8:
  case Intrinsic::amdgcn_mfma_i32_32x32x4i8:
  case Intrinsic::amdgcn_mfma_i32_4x4x4i8:
  case Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4:
  case Intrinsic::amdgcn_wmma_f32_16x16x32_bf16:
  case Intrinsic::amdgcn_wmma_f32_16x16x32_f16:
  case Intrinsic::amdgcn_wmma_f32_16x16x4_f32:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_bf8:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_fp8:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_bf8:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_fp8:
  case Intrinsic::amdgcn_wmma_i32_16x16x64_iu8:
  case Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
    return true;
  default:
    return false;
  }
}

bool isRawBufferAtomic(Intrinsic::ID Id) {
  switch (Id) {
  case Intrinsic::amdgcn_raw_buffer_atomic_swap:
  case Intrinsic::amdgcn_raw_buffer_atomic_add:
  case Intrinsic::amdgcn_raw_buffer_atomic_sub:
  case Intrinsic::amdgcn_raw_buffer_atomic_smin:
  case Intrinsic::amdgcn_raw_buffer_atomic_umin:
  case Intrinsic::amdgcn_raw_buffer_atomic_fmin:
  case Intrinsic::amdgcn_raw_buffer_atomic_smax:
  case Intrinsic::amdgcn_raw_buffer_atomic_umax:
  case Intrinsic::amdgcn_raw_buffer_atomic_fmax:
  case Intrinsic::amdgcn_raw_buffer_atomic_and:
  case Intrinsic::amdgcn_raw_buffer_atomic_or:
  case Intrinsic::amdgcn_raw_buffer_atomic_xor:
  case Intrinsic::amdgcn_raw_buffer_atomic_inc:
  case Intrinsic::amdgcn_raw_buffer_atomic_dec:
  case Intrinsic::amdgcn_raw_buffer_atomic_cond_sub_u32:
  case Intrinsic::amdgcn_raw_buffer_atomic_sub_clamp_u32:
  case Intrinsic::amdgcn_raw_buffer_atomic_cmpswap:
  case Intrinsic::amdgcn_raw_buffer_atomic_fadd:
    return true;
  default:
    return false;
  }
}

// Classify how `V` (our tracked value) is used by the call
// instruction `CB` at operand index `operandIdx`. Returns the call
// site's role for the purposes of the forward walk.
IntrinsicRole classifyIntrinsicUse(CallBase *CB, Value *V,
                                   unsigned OperandIdx) {
  // Inline asm: we cannot audit the constraint letters cheaply here
  // (would need to walk InlineAsm::ParseConstraints and map the
  // physical arg index to the constraint tuple). Refuse -- this is
  // the conservative direction and inline asm with SGPR constraints
  // is exactly the shape the rewrite needs to avoid.
  if (CB->isInlineAsm())
    return IntrinsicRole::SGPRForced;

  Function *Callee = CB->getCalledFunction();
  if (!Callee)
    return IntrinsicRole::SGPRForced; // indirect / unresolved

  Intrinsic::ID Id = Callee->getIntrinsicID();
  if (Id == Intrinsic::not_intrinsic) {
    // Salmon TDM emulation helpers are terminal sinks: they consume the
    // four 20-SGPR descriptor groups (passed as <4 x i32> / <8 x i32> /
    // <4 x i32> / <4 x i32>) plus the source wave size, then write to
    // LDS / global memory and return void. The descriptor groups are
    // uniform per workgroup by construction (the gfx1250 TDM HW reads
    // them as SGPRs); the helpers do not produce a result that
    // continues a cross-lane value's per-source-wave identity, so
    // classifying these calls as VGPRSafeSink is correct -- the use
    // chain terminates here and there is no SGPR-forced operand at the
    // emulation level.  See `tdm-runtime.hpp` for the canonical
    // helper-symbol names; matching by exact name avoids whitelisting
    // arbitrary user-defined functions that happen to take these
    // vector types.
    StringRef name = Callee->getName();
    if (name == "salmon_tdm_load_to_lds" || name == "salmon_tdm_store_from_lds")
      return IntrinsicRole::VGPRSafeSink;
    return IntrinsicRole::SGPRForced; // other ordinary call -- unknown
  }

  if (operandForcesSGPR(Id, OperandIdx))
    return IntrinsicRole::SGPRForced;

  // Raw-buffer intrinsics produce fresh memory data, so their results do not
  // continue a cross-lane value's per-source-wave identity.  Cross-widening
  // MUBUF loads are emitted through the addrspace(8) raw-pointer form so LLVM
  // can preserve the source resource abstraction; the legacy <4 x i32> raw
  // buffer form remains a same-wave path where the descriptor is already
  // target-wave scalar.
  if (Id == Intrinsic::amdgcn_raw_buffer_load) {
    return IntrinsicRole::VGPRSafeSink;
  }
  if (Id == Intrinsic::amdgcn_raw_buffer_store) {
    return IntrinsicRole::VGPRSafeSink;
  }
  if (isRawBufferAtomic(Id)) {
    // Raw-buffer atomics consume vdata and vaddr as VGPR operands, then
    // terminate this value's use chain at memory. The descriptor and soffset
    // operands are SGPR-constrained; keep the safety net loud if a rewritten
    // cross-lane value ever reaches those positions.
    const bool IsCmpSwap = Id == Intrinsic::amdgcn_raw_buffer_atomic_cmpswap;
    const unsigned RsrcIdx = IsCmpSwap ? 2 : 1;
    const unsigned OffsetIdx = IsCmpSwap ? 3 : 2;
    const unsigned SoffsetIdx = IsCmpSwap ? 4 : 3;
    if (OperandIdx == RsrcIdx || OperandIdx == SoffsetIdx)
      return IntrinsicRole::SGPRForced;
    if (OperandIdx < RsrcIdx || OperandIdx == OffsetIdx)
      return IntrinsicRole::VGPRSafeSink;
  }

  if (isIntrinsicVGPRSafePropagator(Id))
    return IntrinsicRole::VGPRSafePropagator;
  if (isIntrinsicVGPRSafeSink(Id))
    return IntrinsicRole::VGPRSafeSink;

  (void)V; // reserved for future per-operand predicates
  return IntrinsicRole::Unknown;
}

SgprForcedConsumerKind classifySgprForcedIntrinsicUse(CallBase *CB,
                                                      unsigned OperandIdx) {
  if (CB->isInlineAsm())
    return SgprForcedConsumerKind::InlineAsm;

  Function *Callee = CB->getCalledFunction();
  if (!Callee)
    return SgprForcedConsumerKind::IndirectCall;

  Intrinsic::ID Id = Callee->getIntrinsicID();
  if (Id == Intrinsic::not_intrinsic)
    return SgprForcedConsumerKind::OrdinaryCall;

  if (Id == Intrinsic::amdgcn_readfirstlane)
    return SgprForcedConsumerKind::ExplicitReadFirstLane;
  if (Id == Intrinsic::amdgcn_s_sendmsg && OperandIdx <= 1)
    return SgprForcedConsumerKind::ScalarSendMsg;
  return SgprForcedConsumerKind::Unknown;
}

// Address spaces whose pointer operand accepts a divergent VGPR. If
// our tracked value feeds a load/store through one of these, the
// backend emits a VGPR-addressed memory op and the per-source-wave
// identity is preserved through the memory operation (the loaded
// value is a fresh VGPR the classifier no longer tracks).
//
// addrspace(4) (constant) and addrspace(6) (constant32bit) require
// uniform SGPR pointers; a divergent pointer there triggers backend
// scalarisation.
bool isVGPRAddressablePointerAS(unsigned AS) {
  switch (AS) {
  case 0: // flat
  case 1: // global
  case 3: // LDS
  case 5: // private / scratch
  case 7: // buffer fat pointer
    return true;
  default:
    // 2 (region) / 4 (constant) / 6 (constant32bit) / anything we
    // have not audited -> refuse.
    return false;
  }
}

// Build a stable one-line description of the blocking use, for the
// refusal diagnostic.
std::string describeUser(const Instruction *I) {
  std::string S;
  raw_string_ostream Os(S);
  if (const auto *CB = dyn_cast<CallBase>(I)) {
    if (CB->isInlineAsm())
      Os << "inline asm";
    else if (Function *F = CB->getCalledFunction())
      Os << "call @" << F->getName();
    else
      Os << "indirect call";
  } else if (const auto *LI = dyn_cast<LoadInst>(I)) {
    Os << "load (addrspace " << LI->getPointerAddressSpace() << ")";
  } else if (const auto *SI = dyn_cast<StoreInst>(I)) {
    Os << "store (addrspace " << SI->getPointerAddressSpace() << ")";
  } else {
    Os << I->getOpcodeName();
  }
  return S;
}

// Forward-classify the use chain of `root`. Returns VGPRSafe iff
// every transitive user is a proven-safe consumer; otherwise writes
// a single-line description of the first blocking user to
// `blockingDetail` and returns SGPRForced.
UseChainVerdict classifyForwardUseChain(
    Value *Root, std::string &BlockingDetail,
    SgprForcedConsumerKind &BlockingKind,
    SmallPtrSetImpl<CallInst *> *SourceWaveReadFirstLaneSites = nullptr) {
  SmallPtrSet<Value *, 32> Visited;
  SmallVector<Value *, 16> Worklist;
  Worklist.push_back(Root);

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    for (Use &U : V->uses()) {
      User *UserObj = U.getUser();
      auto *I = dyn_cast<Instruction>(UserObj);
      if (!I) {
        // ConstantExpr or similar non-instruction use. Refuse --
        // the rewrite pass runs post-mem2reg on lifted kernels
        // and the raiser does not emit constant expressions, so
        // this would indicate an unexpected user we cannot
        // prove safe.
        BlockingDetail = "non-instruction user (ConstantExpr?)";
        BlockingKind = SgprForcedConsumerKind::Unknown;
        return UseChainVerdict::SGPRForced;
      }

      if (I->isTerminator())
        continue; // br/switch/ret consume as i1/i32; AMDGPU handles via EXEC

      // Pure propagators: forward-walk the instruction's result.
      if (isa<CastInst>(I) || isa<BinaryOperator>(I) || isa<UnaryOperator>(I) ||
          isa<ICmpInst>(I) || isa<FCmpInst>(I) || isa<SelectInst>(I) ||
          isa<PHINode>(I) || isa<GetElementPtrInst>(I) || isa<FreezeInst>(I) ||
          isa<ExtractElementInst>(I) || isa<InsertElementInst>(I) ||
          isa<ShuffleVectorInst>(I) || isa<ExtractValueInst>(I) ||
          isa<InsertValueInst>(I)) {
        Worklist.push_back(I);
        continue;
      }

      if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (!isVGPRAddressablePointerAS(LI->getPointerAddressSpace())) {
          BlockingDetail = describeUser(I);
          BlockingKind = SgprForcedConsumerKind::ConstantAddressSpaceMemory;
          return UseChainVerdict::SGPRForced;
        }
        // Loaded value is a fresh result disconnected from our
        // per-source-wave chain; stop walking.
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        // Value operand: always VGPR-safe (hardware write port
        // accepts VGPR regardless of pointer addrspace).
        if (SI->getValueOperand() == V)
          continue;
        // Pointer operand: same addrspace rules as load.
        if (!isVGPRAddressablePointerAS(SI->getPointerAddressSpace())) {
          BlockingDetail = describeUser(I);
          BlockingKind = SgprForcedConsumerKind::ConstantAddressSpaceMemory;
          return UseChainVerdict::SGPRForced;
        }
        continue;
      }
      if (auto *AI = dyn_cast<AtomicRMWInst>(I)) {
        if (!isVGPRAddressablePointerAS(AI->getPointerAddressSpace())) {
          BlockingDetail = describeUser(I);
          BlockingKind = SgprForcedConsumerKind::ConstantAddressSpaceMemory;
          return UseChainVerdict::SGPRForced;
        }
        continue;
      }
      if (auto *CmpX = dyn_cast<AtomicCmpXchgInst>(I)) {
        if (!isVGPRAddressablePointerAS(CmpX->getPointerAddressSpace())) {
          BlockingDetail = describeUser(I);
          BlockingKind = SgprForcedConsumerKind::ConstantAddressSpaceMemory;
          return UseChainVerdict::SGPRForced;
        }
        continue;
      }

      if (auto *CB = dyn_cast<CallBase>(I)) {
        unsigned OperandIdx = U.getOperandNo();
        if (Function *Callee = CB->getCalledFunction();
            Callee &&
            Callee->getIntrinsicID() == Intrinsic::amdgcn_readfirstlane) {
          if (auto *CI = dyn_cast<CallInst>(CB)) {
            if (SourceWaveReadFirstLaneSites)
              SourceWaveReadFirstLaneSites->insert(CI);
            Worklist.push_back(CI);
            continue;
          }
          BlockingDetail = describeUser(I);
          BlockingKind = SgprForcedConsumerKind::ExplicitReadFirstLane;
          return UseChainVerdict::SGPRForced;
        }
        switch (classifyIntrinsicUse(CB, V, OperandIdx)) {
        case IntrinsicRole::SGPRForced:
          BlockingDetail = describeUser(I);
          BlockingKind = classifySgprForcedIntrinsicUse(CB, OperandIdx);
          return UseChainVerdict::SGPRForced;
        case IntrinsicRole::VGPRSafeSink:
          continue;
        case IntrinsicRole::VGPRSafePropagator:
          Worklist.push_back(CB);
          continue;
        case IntrinsicRole::Unknown:
          BlockingDetail = describeUser(I) + " (unaudited)";
          BlockingKind = SgprForcedConsumerKind::Unknown;
          return UseChainVerdict::SGPRForced;
        }
      }

      // Any other instruction kind we did not enumerate -- refuse.
      BlockingDetail = describeUser(I) + " (unaudited)";
      BlockingKind = SgprForcedConsumerKind::Unknown;
      return UseChainVerdict::SGPRForced;
    }
  }
  return UseChainVerdict::VGPRSafe;
}

// ============================================================================
// Rewrite primitives
// ============================================================================

// Build (once per function) the target-wave absolute lane Id as the
// standard two-step mbcnt idiom. Returned value dominates every use
// site because it is emitted at the head of the function's entry
// block, immediately after the terminator of `allocas-and-setup`
// prelude (we insert at the entry's first insertion point).
Value *buildTargetLaneId(Function &F) {
  Module *M = F.getParent();
  LLVMContext &C = F.getContext();
  Type *I32Ty = Type::getInt32Ty(C);
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  Function *MbcntLo =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_lo);
  Function *MbcntHi =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_hi);
  // `ConstantInt::get(IntegerType*, uint64_t V, bool IsSigned=false)`
  // asserts `V < 2^BitWidth` when `!IsSigned`; implicit (int64_t)-1 ->
  // uint64_t produces `0xFFFF'FFFF'FFFF'FFFF` which blows that assert
  // for a 32-bit type. Use the unsigned 32-bit all-ones bit pattern
  // (2^32 - 1), which mbcnt hardware interprets as the wave-wide
  // exec-all mask -- the standard idiom behind the two-step lane_id
  // construction.
  Value *MinusOne = ConstantInt::get(I32Ty, 0xFFFFFFFFu);
  Value *Zero = ConstantInt::get(I32Ty, 0);
  Value *LaneLo = B.CreateCall(MbcntLo, {MinusOne, Zero}, "cwd_lane_id_lo");
  Value *LaneId = B.CreateCall(MbcntHi, {MinusOne, LaneLo}, "cwd_lane_id");
  return LaneId;
}

// Rewrite one `amdgcn.writelane(val, lane, old)` call to
// `select ((lane_id & (W_s-1)) == lane), val, old` in-place.
void rewriteWritelaneCall(CallInst *CI, Value *LaneId,
                          unsigned SourceWaveSize) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Type *I32Ty = B.getInt32Ty();
  Value *Val = CI->getArgOperand(0);
  Value *LaneIdx = CI->getArgOperand(1);
  Value *OldVal = CI->getArgOperand(2);
  Value *ModMask = ConstantInt::get(I32Ty, SourceWaveSize - 1);
  Value *LaneMod = B.CreateAnd(LaneId, ModMask, "cwd_wl_lane_mod");
  Value *SelMask = B.CreateICmpEQ(LaneMod, LaneIdx, "cwd_wl_mask");
  Value *NewVal =
      B.CreateSelect(SelMask, Val, OldVal, "cwd_writelane_rewritten");
  CI->replaceAllUsesWith(NewVal);
  CI->eraseFromParent();
}

// Force a rewritten `ds_bpermute` gather to run whole-wave unless the
// projection already guarantees hardware EXEC = -1 kernel-wide.
Value *forceWholeWaveGather(IRBuilder<> &B, Value *Gather,
                            bool ProvidesFullWaveExecInvariant,
                            const Twine &Name) {
  if (ProvidesFullWaveExecInvariant)
    return Gather;
  Module *M = B.GetInsertBlock()->getModule();
  Function *WwmFn = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_strict_wwm, {Gather->getType()});
  return B.CreateCall(WwmFn, {Gather}, Name);
}

// Rewrite one `amdgcn.readlane(src, lane)` call to
// `ds_bpermute(((lane_id & ~(W_s-1)) | lane) << 2, src)`.
void rewriteReadlaneCall(CallInst *CI, Value *LaneId, unsigned SourceWaveSize,
                         bool ProvidesFullWaveExecInvariant) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Module *M = CI->getModule();
  Type *I32Ty = B.getInt32Ty();
  Value *Src = CI->getArgOperand(0);
  Value *LaneIdx = CI->getArgOperand(1);

  uint32_t BaseMaskImm = ~(static_cast<uint32_t>(SourceWaveSize) - 1u);
  Value *BaseMask = ConstantInt::get(I32Ty, BaseMaskImm);
  Value *SrcWaveBase = B.CreateAnd(LaneId, BaseMask, "cwd_rl_src_wave_base");
  Value *BcastLane = B.CreateOr(SrcWaveBase, LaneIdx, "cwd_rl_bcast_lane");
  Value *Selector =
      B.CreateShl(BcastLane, ConstantInt::get(I32Ty, 2), "cwd_rl_selector");
  Function *Bpermute =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_ds_bpermute);
  Value *Broadcast =
      B.CreateCall(Bpermute, {Selector, Src}, "cwd_readlane_rewritten");
  // `v_readlane_b32` ignores EXEC; force the gather whole-wave so an
  // EXEC-inactive source lane still contributes its real VGPR value
  // instead of `ds_bpermute`'s inactive-lane 0.
  Broadcast = forceWholeWaveGather(B, Broadcast, ProvidesFullWaveExecInvariant,
                                   "cwd_readlane_wwm");
  CI->replaceAllUsesWith(Broadcast);
  CI->eraseFromParent();
}

// Rewrite a reachable explicit `amdgcn.readfirstlane(src)` to a source-wave
// broadcast.  This preserves two independent wave32 scalar values inside one
// wave64 target wave instead of collapsing both halves through a single
// hardware SGPR.
void rewriteReadfirstlaneCall(CallInst *CI, Value *LaneId,
                              unsigned SourceWaveSize,
                              bool ProvidesFullWaveExecInvariant) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Module *M = CI->getModule();
  Type *I32Ty = B.getInt32Ty();
  Value *Src = CI->getArgOperand(0);

  uint32_t BaseMaskImm = ~(static_cast<uint32_t>(SourceWaveSize) - 1u);
  Value *BaseMask = ConstantInt::get(I32Ty, BaseMaskImm);
  Value *SrcWaveBase = B.CreateAnd(LaneId, BaseMask, "cwd_rfl_src_wave_base");
  Value *Selector =
      B.CreateShl(SrcWaveBase, ConstantInt::get(I32Ty, 2), "cwd_rfl_selector");
  Function *Bpermute =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_ds_bpermute);
  Value *Broadcast =
      B.CreateCall(Bpermute, {Selector, Src}, "cwd_readfirstlane_rewritten");
  // `v_readfirstlane_b32` ignores EXEC; force the gather whole-wave so
  // the broadcast source lane contributes its real VGPR value even when
  // the modeled EXEC marks it inactive.
  Broadcast = forceWholeWaveGather(B, Broadcast, ProvidesFullWaveExecInvariant,
                                   "cwd_readfirstlane_wwm");
  CI->replaceAllUsesWith(Broadcast);
  CI->eraseFromParent();
}

// ============================================================================
// DPP rewrite helpers
// ============================================================================

// Pure predicate: is `dpp_ctrl` in the supported family (QUAD_PERM /
// ROW_SHL / ROW_SHR / ROW_XMASK)?  Split out from `buildDppLaneMap` so the pre-
// flight phase can answer "is this ctrl rewritable?" without running
// an IRBuilder and without relying on IRBuilder's implicit constant-
// folding to make the dummy-inputs decode into pure constants.
//
// Invariant: this must be the single source of truth for the
// supported-ctrl set.  `buildDppLaneMap` handles the exact same
// families and no others; adding a new ctrl requires updating both.
//
// Uses `AMDGPU::DPP::DppCtrl` enum constants from
// `llvm/lib/Target/AMDGPU/SIDefines.h` so the per-family ranges are
// self-documenting and track any future ISA extension without
// silently drifting from raw hex.
bool isDppCtrlRewritable(unsigned Ctrl) {
  using namespace llvm::AMDGPU::DPP;
  // QUAD_PERM: every 8-bit value in [0, 0xFF] is a valid 4-nibble
  // quad permutation -- no "unused" encodings in this range.
  if (Ctrl <= QUAD_PERM_LAST)
    return true;
  // ROW_SHL:N with N in [1, 15].  Note ROW_SHL0 (0x100) is a
  // shift-by-zero identity that the ISA marks as "unused" (same
  // encoding as DPP_UNUSED1); we include it here for decode
  // completeness and `buildDppLaneMap` handles it correctly as
  // N=0 (identity: srcWithinRow == withinRow, always in-range).
  if (Ctrl >= ROW_SHL_FIRST && Ctrl <= ROW_SHL_LAST)
    return true;
  // ROW_SHR:N with N in [1, 15].  Same identity-at-N=0 note as
  // ROW_SHL above.
  if (Ctrl >= ROW_SHR_FIRST && Ctrl <= ROW_SHR_LAST)
    return true;
  // ROW_XMASK:N with N in [0, 15].  The ISA defines this as
  // source lane `(within-row lane) XOR N`, so it never leaves the
  // current 16-lane row.
  if (Ctrl >= ROW_XMASK_FIRST && Ctrl <= ROW_XMASK_LAST)
    return true;
  // ROW_ROR:N with N in [1, 15].  Row rotate right: source within-row
  // lane `(withinRow - N) mod 16`.  Rotation is modular within the
  // 16-lane row, so it never crosses a row boundary and is always
  // in-range -- the same wave-size-obliviousness property that makes
  // ROW_SHL / ROW_SHR / ROW_XMASK safe.  ROW_ROR0 (0x120, the N=0
  // identity the ISA marks "unused") is below ROW_ROR_FIRST, so it is
  // not accepted here -- like any unhandled ctrl it takes the refusal
  // path rather than being silently treated as identity.
  if (Ctrl >= ROW_ROR_FIRST && Ctrl <= ROW_ROR_LAST)
    return true;
  // Every other family (WAVE_*, ROW_MIRROR /
  // ROW_HALF_MIRROR, BCAST15 / BCAST31, ROW_NEWBCAST / ROW_SHARE)
  // either crosses 16-lane row boundaries in a wave-size-dependent
  // way OR has a correctness argument this rewrite has not yet
  // codified.  Refusing loudly via this predicate surfaces new-
  // corpus demand concretely -- each "unsupported" refusal points at
  // a specific ctrl to extend.
  return false;
}

// Per-target-lane source-lane mapping for a supported `dpp_ctrl`
// value.  Returned by `buildDppLaneMap` when the ctrl is in the
// supported family; unsupported ctrls are filtered out upstream via
// `isDppCtrlRewritable` before this is called.
struct DppLaneMap {
  // i32: within-row source-lane index (0..15) -- the ds_bpermute
  // selector's intra-row bits.
  Value *SrcWithinRow = nullptr;
  // i1: whether the mapping is valid for this target lane (false on
  // out-of-row references per the DPP semantics).
  Value *InRange = nullptr;
};

// Build the per-lane source-lane + in-range mapping for `dpp_ctrl`.
//
// Supported families (covers the observed Triton reduction corpus):
//
//   * QUAD_PERM          (0x000..0x0FF)  -- per-quad 4-lane permutation.
//     ctrl encodes four 2-bit selectors; lane L within its 4-lane
//     quad reads source-lane-in-quad = selector[L & 3].  Always
//     in-range.
//
//   * ROW_SL:N           (0x101..0x10F)  -- row shift left by N.
//     Target-lane L (within-row W) reads source within-row W + N.
//     Out-of-range iff W + N >= 16.
//
//   * ROW_SR:N           (0x111..0x11F)  -- row shift right by N.
//     Target-lane L (within-row W) reads source within-row W - N.
//     Out-of-range iff W < N.
//
//   * ROW_XMASK:N        (0x160..0x16F)  -- row XOR-mask by N.
//     Target-lane L (within-row W) reads source within-row W ^ N.
//     Always in-range: W ^ N stays within the 16-lane row.
//
//   * ROW_ROR:N          (0x121..0x12F)  -- row rotate right by N.
//     Target-lane L (within-row W) reads source within-row
//     (W - N) mod 16.  Always in-range: the rotation wraps within
//     the 16-lane row.
//
// All five families keep the source lane within the same 16-lane
// row as the target lane.  Since a 16-lane row is a topology
// invariant of every AMDGPU wave size >= 16, the `rowBase(L) |
// srcWithinRow` computation produces identical source-lane indices
// on wave32 and wave64, which is precisely the wave-size-
// obliviousness property the rewrite relies on.
//
// Unsupported families (filtered upstream via
// `isDppCtrlRewritable`; reaching this function with one returns
// an error at the trailing default case):
//
//   * WAVE_SHL1 / WAVE_ROL1 / WAVE_SHR1 / WAVE_ROR1 (wave-wide
//     shifts).  These cross 16-lane row boundaries within the source
//     wave; under cross-widening the "wave" meaning diverges
//     (wave32 = 32 lanes, wave64 = 64 lanes) and the translation is
//     NOT the identity ctrl.  A future rewrite would compute the
//     source-wave boundary via `(L & ~(W_s-1))` and clamp shifts
//     accordingly, but today every Triton reduction we have
//     expresses wave-width reductions via `row_shl/shr` +
//     `permlane16` rather than the wave-wide DPP ctrls, so this
//     family has no corpus demand.
//
//   * ROW_MIRROR / ROW_HALF_MIRROR.  Within a 16-lane row, so
//     expressible here -- no corpus demand yet.
//
//   * BCAST15 / BCAST31 (gfx9-only).  Cross 16- and 32-lane row
//     boundaries respectively.  gfx1250 source cannot emit them
//     (removed in RDNA), so refusing is no regression.
//
//   * ROW_SHARE:N (gfx10+).  Broadcasts lane N of each row to all
//     other lanes in that row.  Expressible via srcWithinRow = N,
//     inRange = true -- no corpus demand yet.
//
// When extending this table, prefer a one-case-per-ctrl-family
// layout and document the in-range predicate and source-lane
// formula alongside each case -- the correctness argument is local
// per ctrl value.  Caller MUST have verified `isDppCtrlRewritable`;
// this function re-checks the invariant and returns an error
// otherwise to turn an internal-invariant violation into a surfaced
// failure rather than a silent "miscompile with default values".
Expected<DppLaneMap> buildDppLaneMap(IRBuilder<> &B, Value *WithinRow,
                                     unsigned Ctrl) {
  using namespace llvm::AMDGPU::DPP;
  DppLaneMap Out;
  Type *I32Ty = B.getInt32Ty();

  if (Ctrl <= QUAD_PERM_LAST) {
    // QUAD_PERM family. Decode the 4 two-bit selectors on-the-fly
    // so the rewrite works for every ctrl value in [0, 0x100)
    // without a 256-way switch.  The 4-lane quad the target lane
    // sits in is `withinRow & ~3`; the target-lane's position
    // within that quad is `withinRow & 3`; the 2-bit selector
    // lives at bits `[2 * (withinRow & 3) .. 2 * (withinRow & 3) + 1]`
    // of `ctrl`.
    Value *QuadBase = B.CreateAnd(WithinRow, ConstantInt::get(I32Ty, ~3u),
                                  "cwd_dpp_quad_base");
    Value *QuadWithin = B.CreateAnd(WithinRow, ConstantInt::get(I32Ty, 3),
                                    "cwd_dpp_quad_within");
    Value *Shift = B.CreateShl(QuadWithin, ConstantInt::get(I32Ty, 1),
                               "cwd_dpp_quad_shift");
    Value *CtrlVal = ConstantInt::get(I32Ty, Ctrl);
    Value *Selector =
        B.CreateAnd(B.CreateLShr(CtrlVal, Shift), ConstantInt::get(I32Ty, 3),
                    "cwd_dpp_quad_sel");
    Out.SrcWithinRow = B.CreateOr(QuadBase, Selector, "cwd_dpp_quad_src");
    Out.InRange = ConstantInt::getTrue(B.getContext());
    return Out;
  }

  if (Ctrl >= ROW_SHL_FIRST && Ctrl <= ROW_SHL_LAST) {
    // ROW_SL:N.  Source within-row = withinRow + N; OOB iff the sum
    // falls outside [0, 16).  Use unsigned comparison -- withinRow
    // is already masked to [0, 16) by the caller's `laneId & 0xF`,
    // so the addition cannot wrap.
    unsigned N = Ctrl - ROW_SHL0;
    Value *NVal = ConstantInt::get(I32Ty, N);
    Out.SrcWithinRow = B.CreateAdd(WithinRow, NVal, "cwd_dpp_sl_src");
    Out.InRange = B.CreateICmpULT(Out.SrcWithinRow, ConstantInt::get(I32Ty, 16),
                                  "cwd_dpp_sl_inrange");
    return Out;
  }

  if (Ctrl >= ROW_SHR_FIRST && Ctrl <= ROW_SHR_LAST) {
    // ROW_SR:N.  Source within-row = withinRow - N; OOB iff
    // withinRow < N.  Compute srcWithinRow as a plain i32 subtract
    // -- the select on `inRange` at the caller clamps the bogus
    // wrap-around result before it feeds the ds_bpermute selector.
    unsigned N = Ctrl - ROW_SHR0;
    Value *NVal = ConstantInt::get(I32Ty, N);
    Out.InRange = B.CreateICmpUGE(WithinRow, NVal, "cwd_dpp_sr_inrange");
    Out.SrcWithinRow = B.CreateSub(WithinRow, NVal, "cwd_dpp_sr_src");
    return Out;
  }

  if (Ctrl >= ROW_XMASK_FIRST && Ctrl <= ROW_XMASK_LAST) {
    // ROW_XMASK:N.  Source within-row = withinRow XOR N.  The ISA
    // manual spells this as `lane[(n & 0x30) + ((n & 0xf) ^ mask)]`;
    // `rowBase | srcWithinRow` supplies the `(n & 0x30)` component
    // for the target wave, while the XOR stays entirely in the
    // 4-bit row index.  Hence every row_xmask:[0..15] is always
    // in-range and wave-size-oblivious under cross-widening.
    unsigned N = Ctrl - ROW_XMASK0;
    Value *NVal = ConstantInt::get(I32Ty, N);
    Out.SrcWithinRow = B.CreateXor(WithinRow, NVal, "cwd_dpp_xmask_src");
    Out.InRange = ConstantInt::getTrue(B.getContext());
    return Out;
  }

  if (Ctrl >= ROW_ROR_FIRST && Ctrl <= ROW_ROR_LAST) {
    // ROW_ROR:N.  Row rotate right by N within the 16-lane row:
    // source within-row = (withinRow - N) mod 16.  Computed as
    // `(withinRow + (16 - N)) & 15` so the result stays in [0, 16)
    // without a negative intermediate.  The rotation wraps within the
    // row, so every target lane maps to a valid source lane in the
    // same row -- always in-range and wave-size-oblivious under
    // cross-widening (the 16-lane row is a topology invariant of every
    // AMDGPU wave size, so `rowBase | srcWithinRow` yields identical
    // source lanes on wave32 and wave64).
    unsigned N = Ctrl - ROW_ROR0;
    Value *AddVal = ConstantInt::get(I32Ty, (16u - N) & 15u);
    Value *Sum = B.CreateAdd(WithinRow, AddVal, "cwd_dpp_ror_sum");
    Out.SrcWithinRow =
        B.CreateAnd(Sum, ConstantInt::get(I32Ty, 15), "cwd_dpp_ror_src");
    Out.InRange = ConstantInt::getTrue(B.getContext());
    return Out;
  }

  // Caller contract violation: `isDppCtrlRewritable` returned true
  // for this ctrl but the decode here has no matching case.  That
  // means the predicate and the decoder have drifted apart -- a
  // miscompile-by-omission shape.  Fail loudly rather than produce
  // a zero-initialised DppLaneMap that would silently short-circuit
  // the rewrite's correctness invariant.
  return createStringError(
      "buildDppLaneMap invariant: isDppCtrlRewritable said supported "
      "but the decoder has no matching case. Extend both together.");
}

// Format a `dpp_ctrl` value as a human-readable name for the refusal
// diagnostic.  Keeps the refusal message grep-able per control family
// so triage doesn't need an ISA reference open.  Range bounds use
// the `AMDGPU::DPP::DppCtrl` enum from
// `llvm/lib/Target/AMDGPU/SIDefines.h` so the classification tracks
// any future ISA addition without silent drift.
std::string describeDppCtrl(unsigned Ctrl) {
  using namespace llvm::AMDGPU::DPP;
  std::string S;
  raw_string_ostream Os(S);
  if (Ctrl <= QUAD_PERM_LAST) {
    Os << "quad_perm:[" << (Ctrl & 3) << "," << ((Ctrl >> 2) & 3) << ","
       << ((Ctrl >> 4) & 3) << "," << ((Ctrl >> 6) & 3) << "]";
  } else if (Ctrl >= ROW_SHL_FIRST && Ctrl <= ROW_SHL_LAST) {
    Os << "row_shl:" << (Ctrl - ROW_SHL0);
  } else if (Ctrl >= ROW_SHR_FIRST && Ctrl <= ROW_SHR_LAST) {
    Os << "row_shr:" << (Ctrl - ROW_SHR0);
  } else if (Ctrl >= ROW_ROR_FIRST && Ctrl <= ROW_ROR_LAST) {
    Os << "row_ror:" << (Ctrl - ROW_ROR0);
  } else if (Ctrl == WAVE_SHL1) {
    Os << "wave_shl:1";
  } else if (Ctrl == WAVE_ROL1) {
    Os << "wave_rol:1";
  } else if (Ctrl == WAVE_SHR1) {
    Os << "wave_shr:1";
  } else if (Ctrl == WAVE_ROR1) {
    Os << "wave_ror:1";
  } else if (Ctrl == ROW_MIRROR) {
    Os << "row_mirror";
  } else if (Ctrl == ROW_HALF_MIRROR) {
    Os << "row_half_mirror";
  } else if (Ctrl == BCAST15) {
    Os << "row_bcast15 (gfx9-only)";
  } else if (Ctrl == BCAST31) {
    Os << "row_bcast31 (gfx9-only)";
  } else if (Ctrl >= ROW_SHARE_FIRST && Ctrl <= ROW_SHARE_LAST) {
    Os << "row_share:" << (Ctrl - ROW_SHARE0);
  } else if (Ctrl >= ROW_XMASK_FIRST && Ctrl <= ROW_XMASK_LAST) {
    Os << "row_xmask:" << (Ctrl - ROW_XMASK0);
  } else {
    Os << "dpp_ctrl=0x" << utohexstr(Ctrl);
  }
  return S;
}

// Rewrite one `amdgcn.update.dpp.i32(old, src, dpp_ctrl, row_mask,
// bank_mask, bound_ctrl)` call.  CALLER CONTRACT: `isDppCtrlRewritable(
// ctrl)` MUST be true -- the pre-flight pass enforces this, and this
// function returns an error if the invariant is broken at the
// call site.  This is stronger than an assert (which would no-op in
// release builds) because a silently-half-rewritten function is
// exactly the "silent-fallback" shape the project rule forbids.
//
// Only called for i32-overloaded DPP.  i64 DPP sites are left to
// the backend's native lowering (see the header's "@llvm.amdgcn.
// update.dpp" paragraph for the i32-only scope rationale).
Error rewriteUpdateDppI32Call(CallInst *CI, Value *LaneId,
                              unsigned SourceWaveSize) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Module *M = CI->getModule();
  Type *I32Ty = B.getInt32Ty();

  // The intrinsic's TableGen declaration marks args 2..5 as `ImmArg`,
  // so the raiser (and any well-formed caller) always passes them as
  // `ConstantInt`. Cast assertively; a non-ConstantInt here indicates
  // a caller invariant violation worth asserting loudly.
  Value *OldVal = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);
  auto *CtrlC = cast<ConstantInt>(CI->getArgOperand(2));
  auto *RowMaskC = cast<ConstantInt>(CI->getArgOperand(3));
  auto *BankMaskC = cast<ConstantInt>(CI->getArgOperand(4));
  auto *BoundCtrlC = cast<ConstantInt>(CI->getArgOperand(5));
  unsigned Ctrl = CtrlC->getZExtValue();
  unsigned RowMaskImm = RowMaskC->getZExtValue();
  unsigned BankMaskImm = BankMaskC->getZExtValue();
  bool BoundCtrl = BoundCtrlC->getZExtValue() != 0;

  // Caller-contract enforcement BEFORE any IR emission.  If the pre-
  // flight gate was somehow bypassed and we reach here with an
  // unsupported ctrl, refuse to emit anything -- an incomplete rewrite
  // on one DPP site would violate the all-or-nothing symmetry across
  // the function's cross-lane primitives.
  if (!isDppCtrlRewritable(Ctrl))
    return createStringError(Twine("rewriteUpdateDppI32Call invariant: "
                                   "pre-flight missed unsupported dpp_ctrl ") +
                             describeDppCtrl(Ctrl));

  // Lane-topology values for the source-fetch path.  Both are
  // derived from the target-wave physical `LaneId` because DPP
  // source fetch is per-physical-row by ISA definition (the source
  // selector indexes a lane within the target wave's row).  `LaneId`
  // itself is memoised at the function level by the caller
  // (`buildTargetLaneId`), so the only duplication across DPP sites
  // is the and/lshr chain, which instcombine folds post-pass.
  Value *WithinRow =
      B.CreateAnd(LaneId, ConstantInt::get(I32Ty, 0xF), "cwd_dpp_within_row");
  Value *RowBase =
      B.CreateAnd(LaneId, ConstantInt::get(I32Ty, ~0xFu), "cwd_dpp_row_base");

  // Per-ctrl source mapping.  `isDppCtrlRewritable` gated the call
  // site -- `buildDppLaneMap` is guaranteed to return a valid map.
  Expected<DppLaneMap> LaneMapOrErr = buildDppLaneMap(B, WithinRow, Ctrl);
  if (!LaneMapOrErr)
    return LaneMapOrErr.takeError();
  DppLaneMap LaneMap = *LaneMapOrErr;

  // Clamp the bogus wrap-around result on OOB so the ds_bpermute
  // selector always references a deterministic intra-row lane.  The
  // `inRange` select below discards the bpermuted value for OOB
  // lanes, so the clamp is strictly for IR clarity -- lane 0's
  // selector reads row[0] instead of row[0xFFFF_FFF8 & 0x3F].
  Value *SrcWithinRowSafe =
      B.CreateSelect(LaneMap.InRange, LaneMap.SrcWithinRow,
                     ConstantInt::get(I32Ty, 0), "cwd_dpp_src_safe");
  Value *SrcLaneAbs = B.CreateOr(RowBase, SrcWithinRowSafe, "cwd_dpp_src_abs");
  Value *ByteAddr =
      B.CreateShl(SrcLaneAbs, ConstantInt::get(I32Ty, 2), "cwd_dpp_selector");

  Function *Bpermute =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_ds_bpermute);
  Value *Bperm = B.CreateCall(Bpermute, {ByteAddr, Src}, "cwd_dpp_bperm");

  // Out-of-range disposition.  Per the AMDGPU ISA DPP spec: an active
  // target lane whose source lane is OOB receives `0` under
  // `bound_ctrl=1` or retains `old` under `bound_ctrl=0`.
  Value *OobVal =
      BoundCtrl ? static_cast<Value *>(ConstantInt::get(I32Ty, 0)) : OldVal;
  Value *DppVal =
      B.CreateSelect(LaneMap.InRange, Bperm, OobVal, "cwd_dpp_inrange");

  // row_mask / bank_mask gating.  Fold the select away when both
  // masks are 0xF (the common "every lane participates" case) --
  // keeps the rewritten IR minimal for the overwhelmingly common
  // reduction-tree shape the corpus emits, and keeps lit-test
  // FileCheck patterns simple.
  //
  // row_mask and bank_mask gate writes by source-wave row/bank index,
  // not by target-wave physical row/bank.  Clip the target-wave lane
  // to the source-wave-local range (`LaneId & (SourceWaveSize - 1)`)
  // before deriving the row/bank index so each source wave within the
  // wider target wave is gated independently.
  Value *Result;
  if (RowMaskImm == 0xF && BankMaskImm == 0xF) {
    Result = DppVal;
  } else {
    Value *SourceLaneMask = ConstantInt::get(I32Ty, SourceWaveSize - 1);
    Value *SourceLane =
        B.CreateAnd(LaneId, SourceLaneMask, "cwd_dpp_source_lane");
    Value *SourceRow =
        B.CreateAnd(B.CreateLShr(SourceLane, ConstantInt::get(I32Ty, 4)),
                    ConstantInt::get(I32Ty, 3), "cwd_dpp_source_row");
    Value *SourceBank =
        B.CreateAnd(B.CreateLShr(SourceLane, ConstantInt::get(I32Ty, 2)),
                    ConstantInt::get(I32Ty, 3), "cwd_dpp_source_bank");
    Value *RowMaskVal = ConstantInt::get(I32Ty, RowMaskImm);
    Value *BankMaskVal = ConstantInt::get(I32Ty, BankMaskImm);
    Value *RowActive =
        B.CreateICmpNE(B.CreateAnd(B.CreateLShr(RowMaskVal, SourceRow),
                                   ConstantInt::get(I32Ty, 1)),
                       ConstantInt::get(I32Ty, 0), "cwd_dpp_row_active");
    Value *BankActive =
        B.CreateICmpNE(B.CreateAnd(B.CreateLShr(BankMaskVal, SourceBank),
                                   ConstantInt::get(I32Ty, 1)),
                       ConstantInt::get(I32Ty, 0), "cwd_dpp_bank_active");
    Value *LaneActive =
        B.CreateAnd(RowActive, BankActive, "cwd_dpp_lane_active");
    Result = B.CreateSelect(LaneActive, DppVal, OldVal, "cwd_dpp_gated");
  }

  CI->replaceAllUsesWith(Result);
  CI->eraseFromParent();
  return Error::success();
}

} // namespace

Expected<CrossLaneDivergentRewriteReport> rewriteCrossLaneDivergent(
    Function &F, unsigned SourceWaveSize, unsigned TargetWaveSize,
    bool ProvidesFullWaveExecInvariant, TargetMachine *TM) {
  CrossLaneDivergentRewriteReport Report;
  // `TM` is reserved for a future UA-backed classifier refinement.
  // See the header doc block for the soundness analysis of why the
  // naive "UA allow-gate on pre-rewrite IR" is insufficient.  Accept
  // null silently -- no current path uses it, and accepting nullptr
  // is part of the documented contract.
  (void)TM;

  // Direction gate. Same-wave / narrowing skip the rewrite entirely:
  // the backend's implicit readfirstlane would not collapse any per-
  // source-wave state that the target wave does not also hold.
  if (TargetWaveSize <= SourceWaveSize)
    return Report;

  // Pre-collect candidate call sites. Iterating the function while
  // rewriting would mutate the CFG under the iterator; the two-phase
  // shape keeps the walk O(n) and the rewrite-phase linear in the
  // number of matched sites.
  //
  // DPP collection is i32-only: the rewrite's `ds_bpermute` path is
  // i32-typed and i64 DPP sites keep their native `@llvm.amdgcn.
  // update.dpp.i64` lowering via the backend's implicit split
  // (correct but not wave-size-aware; see the header for rationale
  // of the i32-only rollout).  Any future widening of the rewrite
  // to i64 would add a split/recombine shim here alongside this
  // walk and need to update the symmetry invariant downstream.
  SmallVector<CallInst *, 16> WritelaneSites;
  SmallVector<CallInst *, 16> ReadlaneSites;
  SmallVector<CallInst *, 16> DppI32Sites;
  SmallPtrSet<CallInst *, 16> ReadfirstlaneSites;
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    Function *Callee = CI->getCalledFunction();
    if (!Callee)
      continue;
    switch (Callee->getIntrinsicID()) {
    case Intrinsic::amdgcn_writelane:
      WritelaneSites.push_back(CI);
      break;
    case Intrinsic::amdgcn_readlane:
      ReadlaneSites.push_back(CI);
      break;
    case Intrinsic::amdgcn_update_dpp:
      if (CI->getType() == Type::getInt32Ty(F.getContext()))
        DppI32Sites.push_back(CI);
      // i64 DPP: intentionally left unrewritten (see walk comment).
      break;
    default:
      break;
    }
  }

  if (WritelaneSites.empty() && ReadlaneSites.empty() && DppI32Sites.empty())
    return Report;

  // ==== Phase A: use-chain classification =================================
  //
  // Forward-walk the uses of every writelane result and every
  // readlane result. If any site's chain reaches an SGPR-constrained
  // consumer (s_buffer_load rsrc, s_sendmsg message, readfirstlane,
  // addrspace(4) load, inline asm `"s"`, or any unaudited sink),
  // refuse the whole function rather than produce an asymmetric
  // rewrite or a backend-re-scalarised rewrite. The refusal is all-
  // or-nothing to preserve the writelane/readlane symmetry invariant
  // on shared VGPRs -- a mix of rewritten and preserved sites on one
  // VGPR recreates the Matmul128x128 aperture-violation pattern
  // (hotswap/docs/learnings.md 2026-04-21 entry).
  auto ClassifySite = [&](CallInst *CI, const char *Kind) -> bool {
    std::string Detail;
    SgprForcedConsumerKind ConsumerKind = SgprForcedConsumerKind::None;
    if (classifyForwardUseChain(CI, Detail, ConsumerKind,
                                &ReadfirstlaneSites) ==
        UseChainVerdict::VGPRSafe)
      return true;
    std::string Msg;
    raw_string_ostream Os(Msg);
    Os << "function '" << F.getName() << "' has a " << Kind
       << " whose use chain reaches an SGPR-forced consumer (" << Detail
       << "). Rewriting to `ds_bpermute` here would re-introduce "
          "`v_readfirstlane` at the SGPR boundary -- refusing rather "
          "than silently miscompiling. See "
          "hotswap/docs/wave-size-translation.md sec. 5.6.3 (use-"
          "chain constraint).";
    Report.SgprForcedDetail = Os.str();
    Report.SgprForcedKind = ConsumerKind;
    Report.SgprForcedThreadLoopEligible =
        ConsumerKind == SgprForcedConsumerKind::ExplicitReadFirstLane &&
        (StringRef(Kind) == "writelane" || StringRef(Kind) == "readlane");
    return false;
  };

  for (CallInst *CI : WritelaneSites)
    if (!ClassifySite(CI, "writelane"))
      return Report;
  for (CallInst *CI : ReadlaneSites)
    if (!ClassifySite(CI, "readlane"))
      return Report;
  for (CallInst *CI : DppI32Sites)
    if (!ClassifySite(CI, "update.dpp"))
      return Report;

  // ==== Phase B: per-DPP ctrl pre-flight ==================================
  //
  // Refuse the whole function BEFORE any rewrite if ANY DPP ctrl is
  // outside the supported family.  Extends the symmetry invariant to
  // DPP: if we rewrote some DPPs to ds_bpermute while leaving others
  // as native `@llvm.amdgcn.update.dpp`, the mixed state would hit
  // the same source-wave asymmetry trap that writelane/readlane
  // symmetry exists to prevent (a shared VGPR written by one form
  // and read by the other produces divergent data).  All-or-nothing.
  //
  // Pre-flight via a non-mutating decode: walk the collected DPP
  // sites, call `buildDppLaneMap` with a dummy IRBuilder, check
  // `supported`.  Any failure populates the report and returns zero
  // rewrites across all three primitive families.
  //
  // Pre-flight is a pure predicate (`isDppCtrlRewritable`) on the
  // immediate dpp_ctrl operand -- no IRBuilder state, no dummy IR
  // insertion.  Keeps the check cheap, decouples it from any
  // implicit-constant-folding assumption about IRBuilder, and makes
  // the "supported families" contract unmistakably single-sourced.
  for (CallInst *CI : DppI32Sites) {
    unsigned Ctrl = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
    if (!isDppCtrlRewritable(Ctrl)) {
      std::string Msg;
      raw_string_ostream Os(Msg);
      Os << "function '" << F.getName()
         << "' has an update.dpp site with unsupported "
         << describeDppCtrl(Ctrl)
         << ". The cross-widen rewrite only covers quad_perm, "
            "row_shl:N, row_shr:N, row_xmask:N and row_ror:N today "
            "(all stay within a single 16-lane row, hence "
            "wave-size-oblivious). "
            "Extending the supported set requires a per-ctrl "
            "correctness argument in buildDppLaneMap and a new "
            "lit fixture; refusing rather than silently miscompiling. "
            "See hotswap/docs/wave-size-translation.md sec. 5.3.";
      Report.UnsupportedDppDetail = Os.str();
      return Report;
    }
  }

  // ==== Phase C: unconditional symmetric rewrite ==========================
  //
  // Every writelane, every readlane, and every i32 DPP under cross-
  // widening is rewritten -- uniform operands included -- because an
  // asymmetric mix (native `v_writelane_b32` + rewritten `ds_bpermute`
  // on the same VGPR, or native `v_mov_b32_dpp` + rewritten
  // `ds_bpermute + select` reading each other's outputs) is silently
  // unsound (see the header comment in
  // rewrite-cross-lane-divergent.h, "WRITELANE / READLANE
  // SYMMETRY" and the `@llvm.amdgcn.update.dpp` paragraph). The
  // `select` / `ds_bpermute` forms are semantically equivalent to
  // the source opcodes for every (val, old, src) divergence triple
  // so unconditional rewriting is correctness-preserving.
  Value *LaneIdCached = nullptr;
  auto GetLaneId = [&]() -> Value * {
    if (!LaneIdCached)
      LaneIdCached = buildTargetLaneId(F);
    return LaneIdCached;
  };

  for (CallInst *CI : WritelaneSites) {
    rewriteWritelaneCall(CI, GetLaneId(), SourceWaveSize);
    ++Report.WritelaneRewritten;
  }
  for (CallInst *CI : ReadlaneSites) {
    rewriteReadlaneCall(CI, GetLaneId(), SourceWaveSize,
                        ProvidesFullWaveExecInvariant);
    ++Report.ReadlaneRewritten;
  }
  for (CallInst *CI : ReadfirstlaneSites)
    rewriteReadfirstlaneCall(CI, GetLaneId(), SourceWaveSize,
                             ProvidesFullWaveExecInvariant);
  for (CallInst *CI : DppI32Sites) {
    // Phase B above guaranteed `isDppCtrlRewritable(ctrl)`, and
    // `rewriteUpdateDppI32Call` re-checks and returns an error
    // on violation -- defence in depth, release-build-safe (unlike
    // `assert`, which no-ops under NDEBUG and would let a silent
    // half-rewrite through).  The counter increments only AFTER the
    // rewriter successfully returns; a returned error propagates out
    // first, so it cannot leave the report lying.
    if (Error E = rewriteUpdateDppI32Call(CI, GetLaneId(), SourceWaveSize))
      return E;
    ++Report.DppRewritten;
  }

  return Report;
}

} // namespace COMGR::hotswap
