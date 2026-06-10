//===- handle-valu.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-internal.h"
#include "handle-valu-f16-utils.h"
#include "handle-valu-output-mods.h"
#include "handlers.h"
#include "opcode-map.h"

#include "canonical-op.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

std::optional<int64_t> readNamedImmOperand(const DecodedInst &Di,
                                           AMDGPU::OpName Name) {
  int Idx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Idx < 0 || static_cast<unsigned>(Idx) >= Di.Inst.getNumOperands())
    return std::nullopt;
  unsigned OperandIdx = static_cast<unsigned>(Idx);
  if (!Di.isImm(OperandIdx))
    return std::nullopt;
  return Di.getImm(OperandIdx);
}

std::optional<bool> readVOP3Clamp(const DecodedInst &Di, HandlerResult &Hr,
                                  StringRef OpName) {
  std::optional<int64_t> Clamp =
      readNamedImmOperand(Di, AMDGPU::OpName::clamp);
  if (!Clamp) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3",
        (Twine(OpName) +
         " missing immediate clamp operand; operand table layout does not "
         "match the expected VOP3 profile")
            .str());
    return std::nullopt;
  }
  return *Clamp != 0;
}

// True16 VOP3 half-select helpers. For code-object disassembly, LLVM decodes
// these op_sel bits into src*_modifiers; the non-DPP path does not reliably
// synthesize a standalone OpName::op_sel operand. The destination selector is
// carried as DST_OP_SEL on src0_modifiers. Keep the accepted modifier mask
// narrow so future neg/abs/other modifier forms fail loudly instead of being
// silently dropped.
struct True16OpSel {
  bool Src0Hi = false;
  bool Src1Hi = false;
  bool DstHi = false;
};

std::optional<True16OpSel> readTrue16OpSel(const DecodedInst &Di,
                                           OpResolver &Op, HandlerResult &Hr,
                                           StringRef OpName) {
  if (Op.nSrcs() < 2) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3",
        (Twine(OpName) +
         " has too few source operands; expected src0/src1")
            .str());
    return std::nullopt;
  }

  unsigned Src0Mods = Op.srcMod(0);
  unsigned Src1Mods = Op.srcMod(1);
  constexpr unsigned AllowedSrc0Mods =
      SISrcMods::OP_SEL_0 | SISrcMods::DST_OP_SEL;
  constexpr unsigned AllowedSrc1Mods = SISrcMods::OP_SEL_0;
  if ((Src0Mods & ~AllowedSrc0Mods) != 0 ||
      (Src1Mods & ~AllowedSrc1Mods) != 0) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3",
        (Twine(OpName) +
         " has unsupported source modifiers; only op_sel bits are modeled")
            .str());
    return std::nullopt;
  }

  True16OpSel Sel;
  Sel.Src0Hi = (Src0Mods & SISrcMods::OP_SEL_0) != 0;
  Sel.Src1Hi = (Src1Mods & SISrcMods::OP_SEL_0) != 0;
  Sel.DstHi = (Src0Mods & SISrcMods::DST_OP_SEL) != 0;
  return Sel;
}

// Extract the selected true16 source half from the containing 32-bit value.
Value *extractU16Half(RaiseContext &Ctx, Value *Bits, bool HighHalf) {
  if (HighHalf)
    Bits = Ctx.B.CreateLShr(Bits, 16);
  return Ctx.B.CreateTrunc(Bits, Type::getInt16Ty(Ctx.C));
}

// Merge the 16-bit result into the selected half, preserving the other half.
void writeSelectedU16Half(RaiseContext &Ctx, ParsedReg Dst, Value *Result,
                          bool HighHalf, StringRef MergeName) {
  Value *ResultZ = Ctx.B.CreateZExt(Result, Ctx.I32Ty);
  Value *Old = Ctx.Regs.readReg32(Ctx.B, Dst);
  if (!HighHalf) {
    Value *High =
        Ctx.B.CreateAnd(Old, ConstantInt::get(Ctx.I32Ty, 0xFFFF0000u));
    Ctx.writeReg32(Dst, Ctx.B.CreateOr(High, ResultZ, MergeName));
    return;
  }

  Value *Low = Ctx.B.CreateAnd(Old, ConstantInt::get(Ctx.I32Ty, 0x0000FFFFu));
  Value *Shifted = Ctx.B.CreateShl(ResultZ, 16);
  Ctx.writeReg32(Dst, Ctx.B.CreateOr(Low, Shifted, MergeName));
}

const char *true16AddSubOpName(bool IsSub, bool IsSigned) {
  if (IsSub)
    return IsSigned ? "v_sub_nc_i16" : "v_sub_nc_u16";
  return IsSigned ? "v_add_nc_i16" : "v_add_nc_u16";
}

const char *true16AddSubResultName(bool IsSub, bool IsSigned) {
  if (IsSub)
    return IsSigned ? "vsub_nc_i16" : "vsub_nc_u16";
  return IsSigned ? "vadd_nc_i16" : "vadd_nc_u16";
}

const char *true16AddSubMergeName(bool IsSub, bool IsSigned, bool DstHi) {
  if (IsSub) {
    if (DstHi)
      return IsSigned ? "vsub_i16_merge_hi" : "vsub_u16_merge_hi";
    return IsSigned ? "vsub_i16_merge_lo" : "vsub_u16_merge_lo";
  }
  if (DstHi)
    return IsSigned ? "vadd_i16_merge_hi" : "vadd_u16_merge_hi";
  return IsSigned ? "vadd_i16_merge_lo" : "vadd_u16_merge_lo";
}

Intrinsic::ID true16AddSubSatIntrinsic(bool IsSub, bool IsSigned) {
  if (IsSub)
    return IsSigned ? Intrinsic::ssub_sat : Intrinsic::usub_sat;
  return IsSigned ? Intrinsic::sadd_sat : Intrinsic::uadd_sat;
}

// ============================================================================
// Carry-chain scalar-operand routing for VOP2-CI / VOP3B-CI instructions.
//
// The V_{ADD,SUB,SUBREV}_CO_(CI_)U32 family exists in two encodings:
//
//   * e32: implicit VCC for both carry-in (ci variants) and carry-out.
//     The MC operand table has no scalar operand; `op.nSrcs() == 2`
//     (vsrc0, vsrc1) and `di.numDefs == 1` (vdst only).
//
//   * e64 / VOP3B: EXPLICIT scalar operand for both carry-in (ci
//     variants, MC src index 2) and carry-out (MC def index 1). The
//     scalar can be `vcc_lo` / `vcc` OR an arbitrary `sN` -- the
//     compiler picks based on SGPR pressure. `op.nSrcs() == 3` on ci
//     variants and `di.numDefs == 2` on every co variant (ci or not).
//
// Pre-2026-04-22 the six carry-chain handlers in this file hardcoded
// `ctx.Regs.loadVCC` / `ctx.Regs.storeVCC` for both endpoints,
// silently ignoring the explicit scalar operand on e64 forms. That
// matches the VOPD `v_dual_cndmask_b32` SGPR-condition bug that
// miscompiled `canary_bpermute_scan_fp32` and `corpus_layernorm_fp32`
// (hotswap/docs/modrep-predicate-chain.md §6.4). The current corpus
// (Triton on gfx1250 / gfx942, AITER TensileLite) does not exercise
// the non-VCC SGPR form of these instructions -- Triton emits
// `v_add_nc_u32` / `v_add_nc_u64` (no-carry) on gfx1250 and the
// fused `v_lshl_add_u64` on gfx942, AITER emits `v_add_co_u32 ...,
// vcc_lo, ...` exclusively. But the latent silent-miscompile is
// strictly worse than the VOPD bug it mirrors, because it would
// miscompile address arithmetic rather than a single predicate, and
// the principled project rule is "never do silent fallbacks". The
// helpers below mirror `V_CNDMASK_B32`'s SGPR-aware routing in
// `handle-valu-vop3p.cpp` so these six handlers now share the
// exact same scalar-operand semantics.
// ============================================================================

// Read the per-lane i1 carry-in for a carry-chain instruction.
//
// For e64 forms whose MC src at `srcIndex` is an explicit scalar register:
//
//   * `vcc_lo` / `vcc` -> `loadVCC` (the same path e32 would take).
//   * `sN` -> prefer `lookupSgprWaveMaskI1(N)`'s fresh per-BB V_CMP
//     shadow `i1` (populated by V_CMP_*_e64 writers in the same BB);
//     fall back to `projection.extractLaneBitFromWaveMask` on the
//     raw SGPR alloca (lossy under wave32 -> wave64 cross-widening
//     if the producer truncated to source width -- same residual as
//     the non-VOPD V_CNDMASK_B32 handler, see
//     hotswap/docs/sgpr-wave-mask-translation.md §3.1).
//   * NOREG (null ssrc2) -> zero carry-in (hardware semantics for
//     null scalar source; defensive -- AMDGPU backends don't emit
//     this in practice, but an i1 zero is the least-surprising
//     interpretation if it ever appears).
//
// For e32 forms (no explicit scalar operand -- `op.nSrcs() <= srcIndex`
// or the operand is not a register) -> `loadVCC` (the e32 implicit
// VCC semantics).
Value *readCarryInI1(RaiseContext &Ctx, const DecodedInst &Di,
                      OpResolver &Op, unsigned SrcIndex) {
  if (Op.nSrcs() > SrcIndex && Di.isReg(Op.srcIdx(SrcIndex))) {
    ParsedReg CarryReg =
        Ctx.parseReg(Di.getReg(Op.srcIdx(SrcIndex)), Op.srcIdx(SrcIndex));
    switch (CarryReg.RegKind) {
    case ParsedReg::VCC:
      return Ctx.Regs.loadVCC(Ctx.B);
    case ParsedReg::SGPR:
      if (CarryReg.BaseIdx >= 0) {
        if (Value *FreshCmp = Ctx.lookupSgprWaveMaskI1(CarryReg.BaseIdx))
          return FreshCmp;
        Value *CondVal = Ctx.Isa.isWave32()
                             ? Ctx.Regs.loadSGPR32(Ctx.B, CarryReg.BaseIdx)
                             : Ctx.Regs.loadSGPR64(Ctx.B, CarryReg.BaseIdx);
        return Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, CondVal);
      }
      break;
    case ParsedReg::NOREG:
      return ConstantInt::getFalse(Ctx.B.getInt1Ty());
    default:
      break;
    }
  }
  return Ctx.Regs.loadVCC(Ctx.B);
}

// Write the per-lane i1 carry-out for a carry-chain instruction.
//
// For e64 forms whose MC def at index 1 is an explicit scalar register:
//
//   * `vcc_lo` / `vcc` -> `storeVCC` (the same path e32 would take).
//   * `sN` -> ballot the per-lane i1 up to source-wave-mask width via
//     `projection.ballotI1ToWidth`, store the narrow mask to the
//     SGPR via `writeRegExecWidth` (wave32-source single SGPR;
//     wave64-source SGPR pair), AND record the fresh per-lane `i1`
//     shadow via `recordSgprWaveMaskI1` so a same-BB consumer (e.g.
//     a following V_CNDMASK_B32 or V_ADD_CO_CI_U32) can look it up
//     without the lossy extract round-trip. Mirrors the V_CMP_*_e64
//     SGPR-write path in `handle-valu-vcmp.cpp`.
//   * NOREG (null sdst) -> discard the carry-out (hardware semantics
//     for null scalar destination).
//
// For e32 forms (no explicit destination -- `di.numDefs < 2` or the
// def is not a register) -> `storeVCC` (the e32 implicit VCC
// semantics).
void writeCarryOutI1(RaiseContext &Ctx, const DecodedInst &Di,
                      OpResolver &Op, Value *CarryI1) {
  if (Di.NumDefs >= 2 && Di.isReg(1)) {
    ParsedReg CarryDst = Op.dst(1);
    switch (CarryDst.RegKind) {
    case ParsedReg::VCC:
      Ctx.Regs.storeVCC(Ctx.B, CarryI1);
      return;
    case ParsedReg::SGPR:
      if (CarryDst.BaseIdx >= 0) {
        Type *SourceWidth =
            (Ctx.Projection.sourceWaveScopedLaneOps() && CarryDst.Width >= 2)
                ? Ctx.I64Ty
                : Ctx.Projection.sourceWaveMaskTy();
        Value *Mask = Ctx.Projection.ballotI1ToWidth(
            Ctx.B, CarryI1, SourceWidth, "carry_ballot");
        Ctx.writeRegExecWidth(CarryDst, Mask);
        Ctx.recordSgprWaveMaskI1(CarryDst.BaseIdx, CarryI1,
                                  /*isPair=*/CarryDst.Width >= 2);
      }
      return;
    case ParsedReg::NOREG:
      return;
    default:
      break;
    }
  }
  Ctx.Regs.storeVCC(Ctx.B, CarryI1);
}

// Emit the cross-target (gfx1250 -> gfx94x) dequantisation expansion
// of `v_cvt_scale_pk8_bf16_fp4` for `scale_sel == 0` as pure IR.
// Returns a `<8 x bfloat>` Value; the caller hands it to
// `writeRegVec` exactly like the same-target intrinsic path does.
//
// see hotswap/docs/matrix-translation.md §7.4 -- MXFP4 dequant primitive.
//
// Algorithm (per lane i in 0..7; matches
// `mxfp4::mxfp4BitAlgebraBf16Bits` in mxfp4-dequant.cpp step-for-step
// so the cpp-level unit test pins the algorithm and the canary on
// gfx1250 pins it end-to-end against the hardware primitive):
//
//   1. Extract 4-bit nibble:   %n_i  = (%src >> (i*4)) & 0xF
//   2. Decompose FP4 E2M1:     sign=n_i[3], exp_fp4=n_i[2:1], mant_fp4=n_i[0]
//   3. FP4 -> BF16 fields:
//        // Normal FP4 (exp_fp4 >= 1): bf16_exp = exp_fp4 + 126,
//        // bf16_mant = mant_fp4 ? 0x40 : 0.
//        // Subnormal FP4 (exp_fp4 == 0): bf16_exp = mant_fp4 ? 126 : 0,
//        // bf16_mant = 0.  (±0 stays ±0; ±0.5 becomes normal BF16 exp=126.)
//   4. Scale byte:             scale_byte = %scale & 0xFF    (scale_sel==0 only)
//   5. Apply scale via exp add:
//        new_exp = (signed i32) bf16_exp + scale_byte - 127
//        result =
//          (scale_byte == 0xFF)         ? 0x7FC0                         // qNaN
//          : (bf16_magnitude == 0)       ? (sign << 15)                   // ±0
//          : (new_exp >= 0xFF)           ? (sign << 15) | 0x7F80          // ±Inf
//          : (new_exp >= 1)              ? (sign<<15) | (new_exp<<7) | bf16_mant   // normal
//          : subnormal_shift(new_exp, sign, 0x80 | bf16_mant)             // subnormal / ±0
//   6. Insert i16 bits -> bfloat -> <8 x bfloat> lane i.
//
// Corner-case summary (all bit-exact against the OCP MXFP spec + what
// the hardware primitive emits on bit-valid inputs; see
// `tests/mxfp4_dequant_test.cpp` for the full 4096-point sweep):
//   * FP4 ±0 × NaN scale  -> NaN (IEEE 0 × NaN = NaN).  The NaN-scale
//     branch short-circuits before the magnitude-zero check.
//   * FP4 ±0 × finite scale -> ±0 preserving sign.
//   * Overflow (new_exp >= 0xFF): saturate to BF16 ±Inf.  BF16
//     supports Inf even though FP4 does not -- destination-format
//     semantics apply after the scale add.
//   * Underflow (new_exp <= 0): compute BF16 subnormal via right-shift
//     of the (implicit-1).mant field; zero when the shift drops all
//     bits past count 7 (BF16 subnormal range floor is 2^-133).
//   * Rounding mode: N/A.  The multiplication by 2^(scale_byte - 127)
//     is exact in floating-point for any power-of-2 scale; we emit
//     integer field manipulation instead of an fmul so the lowering
//     is bit-exact regardless of the target's float-mode register
//     state (FTZ / DAZ bits are irrelevant because no fmul actually
//     runs).
static llvm::Value *emitCvtScalePk8Bf16Fp4CrossTargetExpansion(
    RaiseContext &Ctx, llvm::Value *SrcI32, llvm::Value *ScaleI32) {
  llvm::IRBuilder<> &B = Ctx.B;
  llvm::Type *I32Ty = Ctx.I32Ty;
  llvm::Type *I16Ty = llvm::Type::getInt16Ty(Ctx.C);
  llvm::Type *Bf16Ty = llvm::Type::getBFloatTy(Ctx.C);

  // Constants used across all 8 lanes.  Factored out so the emitted
  // IR reads cleanly in lit / FileCheck output.
  llvm::Constant *C0xF   = llvm::ConstantInt::get(I32Ty, 0xF);
  llvm::Constant *C0xFf  = llvm::ConstantInt::get(I32Ty, 0xFF);
  llvm::Constant *C127   = llvm::ConstantInt::get(I32Ty, 127);
  llvm::Constant *C126   = llvm::ConstantInt::get(I32Ty, 126);
  llvm::Constant *C1     = llvm::ConstantInt::get(I32Ty, 1);
  llvm::Constant *C3     = llvm::ConstantInt::get(I32Ty, 3);
  llvm::Constant *C7     = llvm::ConstantInt::get(I32Ty, 7);
  llvm::Constant *C8     = llvm::ConstantInt::get(I32Ty, 8);
  llvm::Constant *C0x40  = llvm::ConstantInt::get(I32Ty, 0x40);
  llvm::Constant *C0x80  = llvm::ConstantInt::get(I32Ty, 0x80);
  llvm::Constant *C0x7F80 = llvm::ConstantInt::get(I32Ty, 0x7F80);

  // Scale-byte extraction: low byte of the i32 scale register.  All 8
  // lanes share the same scale byte for scale_sel == 0 per the
  // declared support set.
  llvm::Value *ScaleByte = B.CreateAnd(ScaleI32, C0xFf, "mxfp4_scale_byte");
  llvm::Value *IsScaleNaN =
      B.CreateICmpEQ(ScaleByte, C0xFf, "mxfp4_is_scale_nan");

  // BF16 canonical qNaN (0x7FC0), used when scale_byte == 0xFF; stored
  // as i32 so it merges with the select chain's other i32 branches.
  llvm::Constant *Bf16NaN = llvm::ConstantInt::get(I32Ty, 0x7FC0);

  // Per-lane result accumulator: <8 x bfloat>, starts as undef (none
  // of the 8 lanes are fully-defined until every insertelement has
  // fired).  Mirrors the shape `writeRegVec` expects from the
  // same-target intrinsic arm.
  llvm::Type *V8bf16Ty = llvm::FixedVectorType::get(Bf16Ty, 8);
  llvm::Value *Vec = llvm::UndefValue::get(V8bf16Ty);

  llvm::Constant *C0 = llvm::ConstantInt::get(I32Ty, 0);

  for (unsigned Lane = 0; Lane < 8; ++Lane) {
    // Nibble extraction.  Low nibble (lane 0) is in src bits [3:0],
    // matching hardware's "nibble 0 = lane 0" contract (documented on
    // the same-target arm above).
    llvm::Value *Shamt = llvm::ConstantInt::get(I32Ty, Lane * 4);
    llvm::Value *Nibble = B.CreateAnd(
        B.CreateLShr(SrcI32, Shamt, "mxfp4_src_shr"),
        C0xF, "mxfp4_nibble");

    // FP4 E2M1 field decomposition.
    llvm::Value *SignBit =
        B.CreateAnd(B.CreateLShr(Nibble, C3), C1, "mxfp4_sign");
    llvm::Value *ExpFp4 =
        B.CreateAnd(B.CreateLShr(Nibble, C1), C3, "mxfp4_exp_fp4");
    llvm::Value *MantFp4 =
        B.CreateAnd(Nibble, C1, "mxfp4_mant_fp4");
    llvm::Value *SignField =
        B.CreateShl(SignBit, llvm::ConstantInt::get(I32Ty, 15),
                    "mxfp4_sign_field");

    // Normal-FP4 BF16 fields: exp_fp4 + 126 and (mant_fp4 ? 0x40 : 0).
    llvm::Value *Bf16ExpNorm =
        B.CreateAdd(ExpFp4, C126, "mxfp4_bf16_exp_norm");
    llvm::Value *MantFp4Nz =
        B.CreateICmpNE(MantFp4, C0, "mxfp4_mant_fp4_nz");
    llvm::Value *Bf16MantNorm =
        B.CreateSelect(MantFp4Nz, C0x40, C0, "mxfp4_bf16_mant_norm");

    // Subnormal-FP4 BF16 fields: if mant_fp4 = 1 (FP4 ±0.5) use
    // bf16_exp = 126; otherwise (FP4 ±0) bf16_exp = 0.  bf16_mant is
    // always 0 in this branch.
    llvm::Value *Bf16ExpSub =
        B.CreateSelect(MantFp4Nz, C126, C0, "mxfp4_bf16_exp_sub");
    llvm::Value *IsFp4Sub =
        B.CreateICmpEQ(ExpFp4, C0, "mxfp4_is_fp4_sub");
    llvm::Value *Bf16Exp = B.CreateSelect(IsFp4Sub, Bf16ExpSub, Bf16ExpNorm,
                                           "mxfp4_bf16_exp");
    llvm::Value *Bf16Mant = B.CreateSelect(IsFp4Sub, C0, Bf16MantNorm,
                                            "mxfp4_bf16_mant");

    // Magnitude (exp || mant in low 15 bits).  Used only to detect
    // the FP4-±0 shortcut; no rounding implication.
    llvm::Value *Magnitude = B.CreateOr(
        B.CreateShl(Bf16Exp, C7), Bf16Mant, "mxfp4_magnitude");
    llvm::Value *IsFp4Zero =
        B.CreateICmpEQ(Magnitude, C0, "mxfp4_is_fp4_zero");

    // Scaled exponent: bf16_exp + scale_byte - 127.  Signed i32 so
    // subnormal / zero decay is captured by new_exp < 1 rather than
    // by unsigned wrap.
    llvm::Value *ExpPlusScale =
        B.CreateAdd(Bf16Exp, ScaleByte, "mxfp4_exp_plus_scale");
    llvm::Value *NewExp =
        B.CreateSub(ExpPlusScale, C127, "mxfp4_new_exp");

    // Overflow branch: new_exp >= 0xFF -> BF16 ±Inf.  Comparison is
    // signed because new_exp may underflow negative; anything >=
    // 0xFF is overflow regardless.
    llvm::Value *IsOverflow =
        B.CreateICmpSGE(NewExp, C0xFf, "mxfp4_is_overflow");
    llvm::Value *InfBits =
        B.CreateOr(SignField, C0x7F80, "mxfp4_inf_bits");

    // Normal branch: new_exp in [1, 0xFE] -> (sign<<15) | (new_exp<<7)
    // | bf16_mant.  We mask new_exp to 8 bits to keep the field
    // width correct when the branch is dead (the select's other arm
    // handles that case, but we still want a clean IR shape).
    llvm::Value *NewExpMasked =
        B.CreateAnd(NewExp, C0xFf, "mxfp4_new_exp_masked");
    llvm::Value *NormalBits = B.CreateOr(
        B.CreateOr(SignField,
                   B.CreateShl(NewExpMasked, C7, "mxfp4_new_exp_shl"),
                   "mxfp4_sign_or_exp"),
        Bf16Mant, "mxfp4_normal_bits");

    // Subnormal branch: new_exp <= 0 -> shift (implicit-1).mant right
    // by (1 - new_exp).  If shift >= 8 the BF16 representation loses
    // every bit and we flush to ±0.  This defensive clamp is
    // unreachable today -- for FP4 exp >= 1 + scale_byte = 0 the
    // minimum new_exp is 127 + 0 - 127 = 0, giving shift_amt = 1; we
    // keep the clamp so widening the declared support set (e.g. a
    // future scale_sel handling that exposes smaller FP4 exponents)
    // doesn't silently miscompile.
    llvm::Value *Implicit1Mant =
        B.CreateOr(C0x80, Bf16Mant, "mxfp4_implicit_1_mant");
    llvm::Value *ShiftAmt =
        B.CreateSub(C1, NewExp, "mxfp4_shift_amt");
    llvm::Value *ShiftedMant =
        B.CreateLShr(Implicit1Mant, ShiftAmt, "mxfp4_shifted_mant");
    llvm::Value *ShiftTooBig =
        B.CreateICmpSGE(ShiftAmt, C8, "mxfp4_shift_too_big");
    llvm::Value *SubMant = B.CreateSelect(ShiftTooBig, C0, ShiftedMant,
                                           "mxfp4_sub_mant");
    llvm::Value *SubBits =
        B.CreateOr(SignField, SubMant, "mxfp4_sub_bits");

    // new_exp >= 1 selects the normal bits; otherwise subnormal.
    llvm::Value *NewExpGe1 =
        B.CreateICmpSGE(NewExp, C1, "mxfp4_new_exp_ge_1");
    llvm::Value *NormalOrSub =
        B.CreateSelect(NewExpGe1, NormalBits, SubBits,
                       "mxfp4_normal_or_sub");

    // Priority-ordered merge, matching the C++ reference's control flow:
    //   result = is_scale_nan ? qNaN
    //          : is_fp4_zero  ? sign_field
    //          : is_overflow  ? inf_bits
    //          :                normal_or_sub
    //
    // LLVM's `select` is bottom-up (inner selects evaluated last), so
    // build the chain from the default case outward.
    llvm::Value *AfterOverflow = B.CreateSelect(
        IsOverflow, InfBits, NormalOrSub, "mxfp4_after_overflow");
    llvm::Value *AfterZero = B.CreateSelect(
        IsFp4Zero, SignField, AfterOverflow, "mxfp4_after_zero");
    llvm::Value *LaneI32 = B.CreateSelect(
        IsScaleNaN, Bf16NaN, AfterZero, "mxfp4_lane_i32");

    // i32 -> i16 -> bfloat insertion.  Trunc drops the zero-padded
    // upper bits; every result branch above produces a value in
    // [0, 0xFFFF] (qNaN=0x7FC0, Inf|sign ≤ 0xFF80, normal|sign|mant
    // ≤ 0xFFC0, sub|sign ≤ 0x80C0), so trunc is information-
    // preserving.
    llvm::Value *LaneI16 = B.CreateTrunc(LaneI32, I16Ty, "mxfp4_lane_i16");
    llvm::Value *LaneBf = B.CreateBitCast(LaneI16, Bf16Ty, "mxfp4_lane_bf16");
    Vec = B.CreateInsertElement(Vec, LaneBf,
                                 llvm::ConstantInt::get(I32Ty, Lane),
                                 "mxfp4_vec_insert");
  }
  return Vec;
}

} // namespace

HandlerResult handleVALU(RaiseContext &Ctx, const DecodedInst &Di,
                        OpResolver &Op) {
  HandlerResult Hr;
  // `mn` is retained for diagnostic messages only; dispatch is driven entirely
  // by `sop`, which the OpcodeMap canonicalizer resolves from the DPP/SDWA/e32
  // encoding to the base pseudo before the handler sees it.
  StringRef Mn(Di.Mnemonic);
  CanonicalOp Sop = Di.CanonOp;
  if (Sop == CanonicalOp::V_NOP) {
    Hr.Handled = true;
    return Hr;
  }
  // ---- v_mov_b32 ----
  if (Sop == CanonicalOp::V_MOV_B32) {
    Ctx.writeReg32(Op.dst(), Op.src(0));
    Hr.Handled = true;
    return Hr;
  }
  // ---- v_mov_b16 ----
  //
  // gfx11+ true16 16-bit register move. The half selection lives in one of
  // two places depending on encoding form:
  //   * `_e32` (the form LLVM currently emits for true16 targets): the
  //     MCInst's vdst / src0 slots reference the parent VGPR's `_LO16` or
  //     `_HI16` subregister directly. The half is queried via
  //     `AMDGPU::isHi16Reg`; there is no modifier operand.
  //   * `_e64` (and its DPP variants): the register operands stay on the
  //     parent VGPR_32 and the half is carried by `src0_modifiers` --
  //     `OP_SEL_0` (bit 2) selects the src0 half and `DST_OP_SEL` (bit 3)
  //     selects the dst half. Both signals coexist on the same _e64 MCInst
  //     (LLVM mirrors the subreg into the register slot and the op_sel into
  //     the modifier), so reading both and OR-ing is safe across forms.
  // The other half of the dst dword must be preserved (RDNA3+ ISA), so the
  // result goes through writeSelectedU16Half. Source neg/abs are not
  // meaningful for a bit-pattern move and are refused loudly.
  if (Sop == CanonicalOp::V_MOV_B16) {
    if (Op.nSrcs() < 1) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1", "v_mov_b16 missing src0 operand");
      return Hr;
    }
    unsigned Src0Mods = Op.srcMod(0);
    constexpr unsigned AllowedSrc0Mods =
        SISrcMods::OP_SEL_0 | SISrcMods::DST_OP_SEL;
    if ((Src0Mods & ~AllowedSrc0Mods) != 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "v_mov_b16 has unsupported src0 modifiers; only op_sel/dst_op_sel "
          "are modeled");
      return Hr;
    }
    const MCRegisterInfo &MRI = *Ctx.Mc.RegInfo;
    bool DstHi = (Src0Mods & SISrcMods::DST_OP_SEL) != 0;
    if (Di.isReg(0) && AMDGPU::isHi16Reg(Di.getReg(0), MRI))
      DstHi = true;
    unsigned Src0Idx = Di.SrcMap[0];
    bool Src0Hi = (Src0Mods & SISrcMods::OP_SEL_0) != 0;
    if (Di.isReg(Src0Idx) && AMDGPU::isHi16Reg(Di.getReg(Src0Idx), MRI))
      Src0Hi = true;
    Value *Half = extractU16Half(Ctx, Op.src(0), Src0Hi);
    writeSelectedU16Half(Ctx, Op.dst(), Half, DstHi, "v_mov_b16_merge");
    Hr.Handled = true;
    return Hr;
  }
  // ---- Cross-lane primitives (readlane/writelane/permlane/mbcnt/
  //      readfirstlane) -- extracted to handle-valu-cross-lane.cpp ----
  {
    HandlerResult Sub = handleValuCrossLane(Ctx, Di, Op);
    if (Sub.Handled || Sub.Failure.hasFailed())
      return Sub;
  }

  // ---- Small ops (conversions, F16 arith, single-src F32 transcendentals,
  //      16-bit shifts, V_BFREV_B32 / V_NOT_B32, byte pack) ----
  // Extracted to handle-valu-small-ops.cpp.
  {
    HandlerResult Sub = handleValuSmallOps(Ctx, Di, Op);
    if (Sub.Handled || Sub.Failure.hasFailed())
      return Sub;
  }

  // ---- Simple 2-src integer ALU ----
  if (Sop == CanonicalOp::V_ADD_NC_U32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateAdd(Op.src(0), Op.src(1), "vadd"));
    Hr.Handled = true;
    return Hr;
  }
  // GFX9 VOP3-only v_add_i32 / v_sub_i32: plain add/sub when clamp=0,
  // signed saturation (saddsat/ssubsat) when clamp=1.
  if (Sop == CanonicalOp::V_ADD_I32 || Sop == CanonicalOp::V_SUB_I32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    int ClampIdx = AMDGPU::getNamedOperandIdx(
        Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
    bool Clamped = ClampIdx >= 0 && Di.isImm(ClampIdx) &&
                   Di.getImm(ClampIdx) != 0;
    if (Clamped) {
      Intrinsic::ID SatId = (Sop == CanonicalOp::V_ADD_I32)
                                ? Intrinsic::sadd_sat
                                : Intrinsic::ssub_sat;
      Value *Res = Ctx.B.CreateBinaryIntrinsic(SatId, S0, S1);
      Ctx.writeReg32(Op.dst(), Res);
    } else {
      Value *Res = (Sop == CanonicalOp::V_ADD_I32)
                       ? Ctx.B.CreateAdd(S0, S1, "vadd_i32")
                       : Ctx.B.CreateSub(S0, S1, "vsub_i32");
      Ctx.writeReg32(Op.dst(), Res);
    }
    Hr.Handled = true;
    return Hr;
  }
  // Vector add with carry-out (GFX12: v_add_co_u32; VCC or sN = carry).
  // See `writeCarryOutI1` above for the SGPR-vs-VCC routing rationale.
  if (Sop == CanonicalOp::V_ADD_CO_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Value *Res = Ctx.B.CreateAdd(S0, S1, "vadd_co");
    Ctx.writeReg32(Op.dst(), Res);
    auto *Ov = Ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {Ctx.I32Ty}, {S0, S1});
    writeCarryOutI1(Ctx, Di, Op, Ctx.B.CreateExtractValue(Ov, 1));
    Hr.Handled = true;
    return Hr;
  }
  // Vector sub with carry-out (GFX9: v_sub_u32; GFX10+: v_sub_co_u32).
  if (Sop == CanonicalOp::V_SUB_CO_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Value *Res = Ctx.B.CreateSub(S0, S1, "vsub_co");
    Ctx.writeReg32(Op.dst(), Res);
    writeCarryOutI1(Ctx, Di, Op, Ctx.B.CreateICmpULT(S0, S1));
    Hr.Handled = true;
    return Hr;
  }
  // Vector reversed sub with carry-out (GFX9: v_subrev_u32; GFX10+: v_subrev_co_u32).
  if (Sop == CanonicalOp::V_SUBREV_CO_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Value *Res = Ctx.B.CreateSub(S1, S0, "vsubrev_co");
    Ctx.writeReg32(Op.dst(), Res);
    writeCarryOutI1(Ctx, Di, Op, Ctx.B.CreateICmpULT(S1, S0));
    Hr.Handled = true;
    return Hr;
  }
  // Vector sub with borrow-in/borrow-out (GFX9: v_subb_u32; GFX10+:
  // v_sub_co_ci_u32). See `readCarryInI1` / `writeCarryOutI1` above for
  // the SGPR-vs-VCC routing on both endpoints; the e64 form can bind
  // either (or both!) of ssrc2 and sdst to an arbitrary `sN`.
  if (Sop == CanonicalOp::V_SUB_CO_CI_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Value *Bin = Ctx.B.CreateZExt(readCarryInI1(Ctx, Di, Op, /*srcIndex=*/2),
                                   Ctx.I32Ty);
    Value *Diff1 = Ctx.B.CreateSub(S0, S1);
    Value *Diff2 = Ctx.B.CreateSub(Diff1, Bin, "vsub_ci");
    Value *B1 = Ctx.B.CreateICmpULT(S0, S1);
    Value *B2 = Ctx.B.CreateICmpULT(Diff1, Bin);
    Ctx.writeReg32(Op.dst(), Diff2);
    writeCarryOutI1(Ctx, Di, Op, Ctx.B.CreateOr(B1, B2));
    Hr.Handled = true;
    return Hr;
  }
  // Vector reversed sub with borrow-in/borrow-out (v_subbrev_co_u32).
  if (Sop == CanonicalOp::V_SUBREV_CO_CI_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Value *Bin = Ctx.B.CreateZExt(readCarryInI1(Ctx, Di, Op, /*srcIndex=*/2),
                                   Ctx.I32Ty);
    Value *Diff1 = Ctx.B.CreateSub(S1, S0);
    Value *Diff2 = Ctx.B.CreateSub(Diff1, Bin, "vsubrev_ci");
    Value *B1 = Ctx.B.CreateICmpULT(S1, S0);
    Value *B2 = Ctx.B.CreateICmpULT(Diff1, Bin);
    Ctx.writeReg32(Op.dst(), Diff2);
    writeCarryOutI1(Ctx, Di, Op, Ctx.B.CreateOr(B1, B2));
    Hr.Handled = true;
    return Hr;
  }
  // Vector add with carry-in/carry-out (GFX12: v_add_co_ci_u32).
  if (Sop == CanonicalOp::V_ADD_CO_CI_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Value *Cin = Ctx.B.CreateZExt(readCarryInI1(Ctx, Di, Op, /*srcIndex=*/2),
                                   Ctx.I32Ty);
    Function *UaddOv = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::uadd_with_overflow, {Ctx.I32Ty});
    Value *Step1 = Ctx.B.CreateCall(UaddOv, {S0, S1});
    Value *Sum1 = Ctx.B.CreateExtractValue(Step1, 0);
    Value *C1   = Ctx.B.CreateExtractValue(Step1, 1);
    Value *Step2 = Ctx.B.CreateCall(UaddOv, {Sum1, Cin});
    Value *Res   = Ctx.B.CreateExtractValue(Step2, 0, "vadd_ci");
    Value *C2    = Ctx.B.CreateExtractValue(Step2, 1);
    Ctx.writeReg32(Op.dst(), Res);
    writeCarryOutI1(Ctx, Di, Op, Ctx.B.CreateOr(C1, C2));
    Hr.Handled = true;
    return Hr;
  }
  // v_mad_co_u64_u32: D.u64 = S0.u32 * S1.u32 + S2.u64, VCC = carry
  if (Sop == CanonicalOp::V_MAD_CO_U64_U32) {
    Value *A = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty), *B = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty);
    Value *Res = Ctx.B.CreateAdd(Ctx.B.CreateMul(A, B), Op.src64(2), "vmad_co64");
    Ctx.writeReg64(Op.dst(0), Res);
    Hr.Handled = true;
    return Hr;
  }
  // v_mad_nc_{u64_u32,i64_i32} (gfx1250, VOP3Only_Realtriple_gfx1250
  // @ 0x2fa / 0x2fb):
  //   U form: D.u64 = zext(S0.u32) * zext(S1.u32) + S2.u64
  //   I form: D.i64 = sext(S0.i32) * sext(S1.i32) + S2.i64
  // No carry output (hence the "nc" suffix), single dst.  Shared
  // clamp / widening dispatch below.
  //
  // The backend's `SelectMad64_32()` pattern matcher
  // (AMDGPUISelDAGToDAG.cpp:1220) matches the canonical
  // `add(mul(widen s0, widen s1), s2_i64)` and re-emits
  // V_MAD_NC_{U,I}64_{U,I}32 on gfx1250 targets or the legacy
  // V_MAD_{CO_,}U64_U32 / V_MAD_I64_I32 (with VCC allocated to a
  // scratch SGPR and discarded) on gfx942 -- so the same IR here
  // is correct for both same-target and cross-target lift paths.
  //
  // Clamp handling: the `VOP_I32_I32_I64_DPP` profile sets
  // `HasClamp = 1` (VOP3Instructions.td:196 profile body), so the
  // hardware instruction CAN saturate the 64-bit sum (to
  // INT64_{MIN,MAX} for the signed form, to UINT64_MAX for the
  // unsigned form) when the encoding's clamp bit is set.  The
  // corpus producers we've seen so far (`downcast_to_mxfp_*`,
  // which use the signed MAD as part of pointer/offset widening
  // arithmetic) all emit `clamp = 0`, relying on natural
  // wraparound.  If a future corpus kernel surfaces with
  // `clamp = 1`, the principled fix is to extend this handler to
  // emit `llvm.{s,u}add.sat.i64` for the final accumulator add
  // (the widening product is exact -- `i32 * i32` fits in `i64`
  // without overflow -- so saturation reduces to the sum step
  // only).  Until such a producer exists, refuse loudly rather
  // than silently emit a wraparound that the source kernel's
  // `clamp = 1` intent would not tolerate.  The refusal mirrors
  // the `V_ADD_I32` / `V_SUB_I32` GFX9 handler up-file which
  // already toggles between plain-add and `sadd_sat.i32` /
  // `ssub_sat.i32` on the clamp bit -- treating the clamp bit as
  // raise-time-authoritative, not "observably ignorable".
  if (Sop == CanonicalOp::V_MAD_NC_U64_U32 || Sop == CanonicalOp::V_MAD_NC_I64_I32) {
    const bool IsSigned = (Sop == CanonicalOp::V_MAD_NC_I64_I32);
    const int ClampIdx = AMDGPU::getNamedOperandIdx(
        Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
    const bool Clamped = ClampIdx >= 0 && Di.isImm(ClampIdx) &&
                         Di.getImm(ClampIdx) != 0;
    if (Clamped) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          IsSigned
              ? "v_mad_nc_i64_i32 with clamp=1 (signed 64-bit saturating MAD) "
                "is not yet lifted: no corpus producer exercises this encoding, "
                "and emitting the plain `add i64` form would silently drop the "
                "saturation semantics the source kernel's clamp bit requests.  "
                "Principled upgrade path when a producer surfaces: wrap the "
                "accumulator add in `llvm.sadd.sat.i64` (the widening product "
                "is exact in i64 so saturation reduces to the sum step only). "
                "See the block comment above this refusal for the full audit."
              : "v_mad_nc_u64_u32 with clamp=1 (unsigned 64-bit saturating "
                "MAD) is not yet lifted: same rationale as the signed sibling "
                "above -- no corpus producer, and the upgrade path is "
                "`llvm.uadd.sat.i64`.  See the V_MAD_NC_* block comment.");
      return Hr;
    }
    Value *A = IsSigned
                   ? Ctx.B.CreateSExt(Op.src(0), Ctx.I64Ty)
                   : Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty);
    Value *B = IsSigned
                   ? Ctx.B.CreateSExt(Op.src(1), Ctx.I64Ty)
                   : Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty);
    Value *Res = Ctx.B.CreateAdd(Ctx.B.CreateMul(A, B), Op.src64(2),
                                 IsSigned ? "vmad_nc_i64" : "vmad_nc_u64");
    Ctx.writeReg64(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  // v_mad_u32: D.u32 = S0.u32 * S1.u32 + S2.u32 (no carry)
  if (Sop == CanonicalOp::V_MAD_U32) {
    Value *Res = Ctx.B.CreateAdd(Ctx.B.CreateMul(Op.src(0), Op.src(1)), Op.src(2), "vmad_u32");
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_OR_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Op.src(0), Op.src(1), "vor"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_AND_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateAnd(Op.src(0), Op.src(1), "vand"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_LO_U32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateMul(Op.src(0), Op.src(1), "vmul"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_SUB_NC_U32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateSub(Op.src(0), Op.src(1), "vsub"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_SUBREV_NC_U32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateSub(Op.src(1), Op.src(0), "vsubrev"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_XOR_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateXor(Op.src(0), Op.src(1), "vxor"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_XNOR_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateNot(Ctx.B.CreateXor(Op.src(0), Op.src(1)), "vxnor"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAX_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateSelect(Ctx.B.CreateICmpUGT(S0, S1), S0, S1, "vmax"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MIN_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateSelect(Ctx.B.CreateICmpULT(S0, S1), S0, S1, "vmin"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAX_I32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateSelect(Ctx.B.CreateICmpSGT(S0, S1), S0, S1, "vmax"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MIN_I32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateSelect(Ctx.B.CreateICmpSLT(S0, S1), S0, S1, "vmin"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_HI_U32) {
    Value *A = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty), *B = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateTrunc(Ctx.B.CreateLShr(Ctx.B.CreateMul(A, B), 32), Ctx.I32Ty, "vmulhi"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_HI_I32) {
    Value *A = Ctx.B.CreateSExt(Op.src(0), Ctx.I64Ty), *B = Ctx.B.CreateSExt(Op.src(1), Ctx.I64Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateTrunc(Ctx.B.CreateAShr(Ctx.B.CreateMul(A, B), 32), Ctx.I32Ty, "vmulhi"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_U32_U24) {
    Value *A = Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0xFFFFFF));
    Value *B = Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0xFFFFFF));
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateMul(A, B, "mul24"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_I32_I24) {
    Value *A = Ctx.B.CreateShl(Op.src(0), 8);
    A = Ctx.B.CreateAShr(A, 8);
    Value *B = Ctx.B.CreateShl(Op.src(1), 8);
    B = Ctx.B.CreateAShr(B, 8);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateMul(A, B, "mul24i"));
    Hr.Handled = true;
    return Hr;
  }
  // v_dot8c_i32_i4: dst += sum of 8 signed 4-bit lane products
  if (Sop == CanonicalOp::V_DOT8C_I32_I4) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Value *Acc = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    for (int I = 0; I < 8; I++) {
      Value *Shift = ConstantInt::get(Ctx.I32Ty, I * 4);
      Value *A = Ctx.B.CreateAnd(Ctx.B.CreateLShr(S0, Shift), ConstantInt::get(Ctx.I32Ty, 0xF));
      Value *B = Ctx.B.CreateAnd(Ctx.B.CreateLShr(S1, Shift), ConstantInt::get(Ctx.I32Ty, 0xF));
      // Sign-extend 4-bit: shift left 28, arithmetic shift right 28
      A = Ctx.B.CreateAShr(Ctx.B.CreateShl(A, 28), 28);
      B = Ctx.B.CreateAShr(Ctx.B.CreateShl(B, 28), 28);
      Acc = Ctx.B.CreateAdd(Acc, Ctx.B.CreateMul(A, B));
    }
    Ctx.writeReg32(Op.dst(), Acc);
    Hr.Handled = true;
    return Hr;
  }
  // v_dot4c_i32_i8: dst += sum of 4 signed 8-bit lane products
  if (Sop == CanonicalOp::V_DOT4C_I32_I8) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Type *I8Ty = Type::getInt8Ty(Ctx.C);
    Value *Acc = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    for (int I = 0; I < 4; I++) {
      Value *Shift = ConstantInt::get(Ctx.I32Ty, I * 8);
      Value *A = Ctx.B.CreateSExt(Ctx.B.CreateTrunc(Ctx.B.CreateLShr(S0, Shift), I8Ty), Ctx.I32Ty);
      Value *B = Ctx.B.CreateSExt(Ctx.B.CreateTrunc(Ctx.B.CreateLShr(S1, Shift), I8Ty), Ctx.I32Ty);
      Acc = Ctx.B.CreateAdd(Acc, Ctx.B.CreateMul(A, B));
    }
    Ctx.writeReg32(Op.dst(), Acc);
    Hr.Handled = true;
    return Hr;
  }
  // v_dot2c_i32_i16: dst += src0.lo16 * src1.lo16 + src0.hi16 * src1.hi16
  if (Sop == CanonicalOp::V_DOT2C_I32_I16) {
    Value *S0 = Op.src(0), *S1 = Op.src(1);
    Type *I16Ty = Type::getInt16Ty(Ctx.C);
    Value *ALo = Ctx.B.CreateSExt(Ctx.B.CreateTrunc(S0, I16Ty), Ctx.I32Ty);
    Value *AHi = Ctx.B.CreateSExt(Ctx.B.CreateTrunc(Ctx.B.CreateLShr(S0, 16), I16Ty), Ctx.I32Ty);
    Value *BLo = Ctx.B.CreateSExt(Ctx.B.CreateTrunc(S1, I16Ty), Ctx.I32Ty);
    Value *BHi = Ctx.B.CreateSExt(Ctx.B.CreateTrunc(Ctx.B.CreateLShr(S1, 16), I16Ty), Ctx.I32Ty);
    Value *Dot = Ctx.B.CreateAdd(Ctx.B.CreateMul(ALo, BLo), Ctx.B.CreateMul(AHi, BHi));
    Value *Acc = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateAdd(Acc, Dot, "dot2c"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_HI_U32_U24) {
    Value *A = Ctx.B.CreateZExt(Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0xFFFFFF)), Ctx.I64Ty);
    Value *B = Ctx.B.CreateZExt(Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0xFFFFFF)), Ctx.I64Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateTrunc(Ctx.B.CreateLShr(Ctx.B.CreateMul(A, B), 32), Ctx.I32Ty, "mulhi24"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_HI_I32_I24) {
    Value *A = Ctx.B.CreateAShr(Ctx.B.CreateShl(Op.src(0), 8), 8);
    Value *B = Ctx.B.CreateAShr(Ctx.B.CreateShl(Op.src(1), 8), 8);
    Value *A64 = Ctx.B.CreateSExt(A, Ctx.I64Ty), *B64 = Ctx.B.CreateSExt(B, Ctx.I64Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateTrunc(Ctx.B.CreateAShr(Ctx.B.CreateMul(A64, B64), 32), Ctx.I32Ty, "mulhi24i"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAD_I32_I24) {
    // Signed 24-bit MAD: multiply sign-extended low-24 src0/src1 and add the
    // full i32 src2. The unclamped form intentionally stays in i32 IR so it
    // preserves the hardware's low-32-bit wraparound. The clamped form widens
    // only to detect overflow before saturating to signed i32 bounds.
    // V_MAD_U32_U24 clamp is a separate sibling contract; add it when a corpus
    // kernel actually surfaces that shape.
    Value *A = Ctx.B.CreateAShr(Ctx.B.CreateShl(Op.src(0), 8), 8);
    Value *B = Ctx.B.CreateAShr(Ctx.B.CreateShl(Op.src(1), 8), 8);
    bool Clamp = false;
    int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    if (ClampIdx >= 0) {
      if (!Di.isImm(static_cast<unsigned>(ClampIdx))) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3", "v_mad_i32_i24 clamp operand is not an immediate");
        return Hr;
      }
      Clamp = Di.getImm(static_cast<unsigned>(ClampIdx)) != 0;
    }
    Value *Acc = nullptr;
    if (Clamp) {
      Value *WideA = Ctx.B.CreateSExt(A, Ctx.I64Ty, "mad_i24_a_wide");
      Value *WideB = Ctx.B.CreateSExt(B, Ctx.I64Ty, "mad_i24_b_wide");
      Value *WideC = Ctx.B.CreateSExt(Op.src(2), Ctx.I64Ty, "mad_i24_c_wide");
      Value *Wide = Ctx.B.CreateAdd(
          Ctx.B.CreateMul(WideA, WideB, "mad_i24_mul_wide"), WideC,
          "mad_i24_wide");
      Value *Lo = ConstantInt::get(Ctx.I64Ty, INT32_MIN);
      Value *Hi = ConstantInt::get(Ctx.I64Ty, INT32_MAX);
      Wide = Ctx.B.CreateSelect(Ctx.B.CreateICmpSLT(Wide, Lo), Lo, Wide,
                                "mad_i24_clamp_lo");
      Wide = Ctx.B.CreateSelect(Ctx.B.CreateICmpSGT(Wide, Hi), Hi, Wide,
                                "mad_i24_clamp");
      Acc = Ctx.B.CreateTrunc(Wide, Ctx.I32Ty, "mad_i24_clamp_i32");
    } else {
      Acc = Ctx.B.CreateAdd(Ctx.B.CreateMul(A, B, "mad_i24_mul"), Op.src(2),
                            "mad_i24");
    }
    Ctx.writeReg32(Op.dst(), Acc);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAD_U32_U24) {
    Value *A = Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0xFFFFFF));
    Value *B = Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0xFFFFFF));
    bool Clamp = false;
    int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    if (ClampIdx >= 0) {
      if (!Di.isImm(static_cast<unsigned>(ClampIdx))) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3", "v_mad_u32_u24 clamp operand is not an immediate");
        return Hr;
      }
      Clamp = Di.getImm(static_cast<unsigned>(ClampIdx)) != 0;
    }
    Value *Res = nullptr;
    if (Clamp) {
      Value *WideA = Ctx.B.CreateZExt(A, Ctx.I64Ty, "mad_u24_a_wide");
      Value *WideB = Ctx.B.CreateZExt(B, Ctx.I64Ty, "mad_u24_b_wide");
      Value *WideC = Ctx.B.CreateZExt(Op.src(2), Ctx.I64Ty, "mad_u24_c_wide");
      Value *Wide = Ctx.B.CreateAdd(
          Ctx.B.CreateMul(WideA, WideB, "mad_u24_mul_wide"), WideC,
          "mad_u24_wide");
      Value *Hi = ConstantInt::get(Ctx.I64Ty, UINT32_MAX);
      Wide = Ctx.B.CreateSelect(Ctx.B.CreateICmpUGT(Wide, Hi), Hi, Wide,
                                "mad_u24_clamp");
      Res = Ctx.B.CreateTrunc(Wide, Ctx.I32Ty, "mad_u24_clamp_i32");
    } else {
      Res = Ctx.B.CreateAdd(Ctx.B.CreateMul(A, B, "mad24_mul"), Op.src(2),
                            "mad24");
    }
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  // v_writelane_b32 / v_readlane_b32 are handled in handle-valu-cross-lane.cpp.

  // v_bfe_u32: Bit Field Extract Unsigned
  // D.u = (S0.u >> S1.u[4:0]) & ((1 << S2.u[4:0]) - 1)
  if (Sop == CanonicalOp::V_BFE_U32) {
    Value *Base = Op.src(0), *Offset = Op.src(1), *Width = Op.src(2);
    Offset = Ctx.B.CreateAnd(Offset, ConstantInt::get(Ctx.I32Ty, 31));
    Width = Ctx.B.CreateAnd(Width, ConstantInt::get(Ctx.I32Ty, 31));
    Value *Shifted = Ctx.B.CreateLShr(Base, Offset);
    // width is 0-31 after masking, so shl i32 1, width is always valid
    Value *Mask = Ctx.B.CreateSub(Ctx.B.CreateShl(ConstantInt::get(Ctx.I32Ty, 1), Width),
                              ConstantInt::get(Ctx.I32Ty, 1));
    Value *IsZeroWidth = Ctx.B.CreateICmpEQ(Width, ConstantInt::get(Ctx.I32Ty, 0));
    Value *Result = Ctx.B.CreateAnd(Shifted, Mask, "bfe");
    Result = Ctx.B.CreateSelect(IsZeroWidth, ConstantInt::get(Ctx.I32Ty, 0), Result);
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  // v_bfe_i32: signed Bit Field Extract.
  //   D.i = signext(bits [off+width-1 : off] of src), treating src as if
  //         it had been sign-extended past bit 31 first.
  // Implementation: arithmetic right shift by off (fills high bits with
  // src's sign), then mask to `width` low bits and sign-extend from bit
  // (width-1).  Using LShr instead of AShr here would give mask-and-sx
  // only when off+width <= 32; hardware diverges in the wraparound case,
  // so we must use AShr to stay bit-identical to native v_bfe_i32.
  //
  // Note: this is NOT the same formula as s_bfe_i32 -- the scalar form
  // uses a shift-trick (`(src << (32-off-w)) >> (32-w)`), the vector
  // form uses mask-and-sign-extend.  The two hardware blocks differ on
  // the wraparound case; do not "unify" them.
  if (Sop == CanonicalOp::V_BFE_I32) {
    Value *Base = Op.src(0), *Offset = Op.src(1), *Width = Op.src(2);
    Value *C31 = ConstantInt::get(Ctx.I32Ty, 0x1F);
    Offset = Ctx.B.CreateAnd(Offset, C31);
    Width = Ctx.B.CreateAnd(Width, C31);
    Value *Shifted = Ctx.B.CreateAShr(Base, Offset);
    // Build a mask of `width` low bits.  For width == 0 the result is 0
    // (nothing to extract), so we special-case that below and use a
    // safe shift amount (1) here to avoid UB in the mask computation.
    Value *WidthNonZero = Ctx.B.CreateICmpNE(Width,
                                             ConstantInt::get(Ctx.I32Ty, 0));
    Value *MaskShift = Ctx.B.CreateSelect(WidthNonZero, Width,
                                          ConstantInt::get(Ctx.I32Ty, 1));
    Value *Mask = Ctx.B.CreateSub(
        Ctx.B.CreateShl(ConstantInt::get(Ctx.I32Ty, 1), MaskShift),
        ConstantInt::get(Ctx.I32Ty, 1));
    Value *Field = Ctx.B.CreateAnd(Shifted, Mask);
    Value *WidthMinus1 = Ctx.B.CreateSub(MaskShift,
                                         ConstantInt::get(Ctx.I32Ty, 1));
    Value *SignBit = Ctx.B.CreateShl(ConstantInt::get(Ctx.I32Ty, 1),
                                     WidthMinus1);
    Value *Sx = Ctx.B.CreateSub(Ctx.B.CreateXor(Field, SignBit), SignBit,
                                "vbfe_i");
    Value *Result = Ctx.B.CreateSelect(WidthNonZero, Sx,
                                       ConstantInt::get(Ctx.I32Ty, 0));
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  // v_bfi_b32: VOP3 bit-field insert.
  //   D.u = (S0.u & S1.u) | (~S0.u & S2.u)
  // S0 is the mask: bits set in S0 take the corresponding bit from
  // S1, bits clear in S0 take the corresponding bit from S2.
  // VOP3Instructions.td emits `AMDGPUbfiPattern` which is exactly
  // the mask-and-merge formula; no hardware masking of any operand,
  // no modifiers beyond the standard B32 set. gfx942 has the same
  // opcode, so the AMDGPU backend will isel this pair of and/or
  // back to a single v_bfi_b32 on the way down.
  if (Sop == CanonicalOp::V_BFI_B32) {
    Value *Mask = Op.src(0), *One = Op.src(1), *Zero = Op.src(2);
    Value *PickedOne  = Ctx.B.CreateAnd(Mask, One);
    Value *PickedZero = Ctx.B.CreateAnd(Ctx.B.CreateNot(Mask), Zero);
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateOr(PickedOne, PickedZero, "vbfi"));
    Hr.Handled = true;
    return Hr;
  }
  // v_mbcnt_{lo,hi}_u32_b32 are handled in handle-valu-cross-lane.cpp.
  // ---- 64-bit float ops ----
  if (Sop == CanonicalOp::V_ADD_F64) {
    Value *S0 = Op.src64(0), *S1 = Op.src64(1);
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    S0 = Ctx.B.CreateBitCast(S0, F64Ty); S1 = Ctx.B.CreateBitCast(S1, F64Ty);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateFAdd(S0, S1, "vadd_f64"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_F64) {
    Value *S0 = Op.src64(0), *S1 = Op.src64(1);
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    S0 = Ctx.B.CreateBitCast(S0, F64Ty); S1 = Ctx.B.CreateBitCast(S1, F64Ty);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateFMul(S0, S1, "vmul_f64"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAX_NUM_F64 || Sop == CanonicalOp::V_MIN_NUM_F64) {
    Value *S0 = Op.src64(0), *S1 = Op.src64(1);
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    S0 = Ctx.B.CreateBitCast(S0, F64Ty); S1 = Ctx.B.CreateBitCast(S1, F64Ty);
    Intrinsic::ID Id =
        Sop == CanonicalOp::V_MAX_NUM_F64 ? Intrinsic::maximumnum : Intrinsic::minimumnum;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Id, {F64Ty});
    const char *Name =
        Sop == CanonicalOp::V_MAX_NUM_F64 ? "vmaxnum_f64" : "vminnum_f64";
    Ctx.writeReg64(Op.dst(),
                   Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fn, {S0, S1}, Name),
                                       Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_rcp_f64: VOP1 transcendental, single F64 source -> F64 result.
  // The hardware op is a ~26-bit accurate reciprocal approximation
  // (TRANS-class, WriteTrans64). Lift to `llvm.amdgcn.rcp.f64` so
  // the AMDGPU backend isels straight back to v_rcp_f64 on gfx942
  // (no Newton-Raphson refinement is added). A generic `fdiv 1.0,
  // x` would lower to a software divide sequence here unless `arcp`
  // / fast-math flags are present, which would be a silent
  // semantics change versus the source op. See the V_RCP_F64
  // CanonicalOp comment in canonical-op.h for the rationale.
  if (Sop == CanonicalOp::V_RCP_F64) {
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Value *S = Ctx.B.CreateBitCast(Op.src64(0), F64Ty);
    Function *Rcp = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_rcp, {F64Ty});
    Value *R = Ctx.B.CreateCall(Rcp, {S}, "vrcp_f64");
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_ldexp_f64: VOP3-only F64 ldexp. src0 is F64 (with abs/neg
  // modifiers), src1 is the I32 exponent (no modifiers). Lift to the
  // generic `llvm.ldexp.f64.i32` intrinsic; the AMDGPU backend isels it
  // back to v_ldexp_f64 on targets that have the op.
  if (Sop == CanonicalOp::V_LDEXP_F64) {
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, "v_ldexp_f64"))
      return Hr;
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Value *S0 = Ctx.B.CreateBitCast(Op.src64(0), F64Ty);
    unsigned Src0Mods = Op.srcMod(0);
    if (Src0Mods & SISrcMods::ABS)
      S0 = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, S0, nullptr, "abs");
    if (Src0Mods & SISrcMods::NEG)
      S0 = Ctx.B.CreateFNeg(S0, "neg");
    Value *S1 = Op.src(1);
    Function *LdexpFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ldexp, {F64Ty, Ctx.I32Ty});
    Value *R = Ctx.B.CreateCall(LdexpFn, {S0, S1}, "vldexp_f64");
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_FMA_F64 || Sop == CanonicalOp::V_FMAC_F64) {
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Value *S0, *S1, *S2;
    if (Sop == CanonicalOp::V_FMA_F64) {
      S0 = Op.src64(0); S1 = Op.src64(1); S2 = Op.src64(2);
    } else {
      S0 = Op.src64(0); S1 = Op.src64(1);
      S2 = Ctx.B.CreateBitCast(Ctx.Regs.readReg64(Ctx.B, Op.dst()), F64Ty);
    }
    S0 = Ctx.B.CreateBitCast(S0, F64Ty); S1 = Ctx.B.CreateBitCast(S1, F64Ty);
    if (S2->getType() != F64Ty) S2 = Ctx.B.CreateBitCast(S2, F64Ty);
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {F64Ty});
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, S1, S2}, "vfma_f64"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_fmamk_f64 vd, src0, K, src1: vd = fma(src0, K, src1)
  // v_fmaak_f64 vd, src0, src1, K: vd = fma(src0, src1, K)
  // F64 mirror of V_FMAMK_F32 / V_FMAAK_F32. The 64-bit K immediate is the
  // KImmFP64 operand: the disassembler materialises it as an i64 Imm when the
  // upper 32 bits are non-zero, or as an MCExpr (`lit64(...)`) when the upper
  // 32 bits are zero. `readOp64` (via `src64`) handles both forms.
  // Per `decode.cpp` MADMK exception, V_FMAMK_F64's `(src0, K, src1)` layout
  // is recognised by the generic IsMadmk detector (imm strictly between src0
  // and src1) so the strict srcN-position drift check skips k=1 without
  // needing a per-opcode allowlist.
  if (Sop == CanonicalOp::V_FMAMK_F64) {
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Value *S0 = Ctx.B.CreateBitCast(Op.src64(0), F64Ty);
    Value *K  = Ctx.B.CreateBitCast(Op.src64(1), F64Ty);
    Value *S2 = Ctx.B.CreateBitCast(Op.src64(2), F64Ty);
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {F64Ty});
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, K, S2}, "vfmamk_f64"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_FMAAK_F64) {
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Value *S0 = Ctx.B.CreateBitCast(Op.src64(0), F64Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.src64(1), F64Ty);
    Value *K  = Ctx.B.CreateBitCast(Op.src64(2), F64Ty);
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {F64Ty});
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, S1, K}, "vfmaak_f64"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_CVT_F64_U32) {
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateUIToFP(Op.src(0), F64Ty, "cvt_f64_u32"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_CVT_F64_I32) {
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateSIToFP(Op.src(0), F64Ty, "cvt_f64_i32"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_CVT_U32_F64) {
    // Saturates out-of-range f64 to 0/UINT_MAX and maps NaN to 0, so lower to
    // fptoui.sat rather than plain fptoui (which is UB on overflow).
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Value *V = Ctx.B.CreateBitCast(Op.src64(0), F64Ty);
    Function *SatFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fptoui_sat, {Ctx.I32Ty, F64Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(SatFn, {V}, "cvt_u32_f64"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- Reversed-operand shifts ----
  if (Sop == CanonicalOp::V_LSHRREV_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateLShr(Op.src(1), Op.src(0), "vlshr"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_LSHLREV_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateShl(Op.src(1), Op.src(0), "vlshl"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_ASHRREV_I32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateAShr(Op.src(1), Op.src(0), "vashr"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- FP ALU (srcF applies VOP3 neg/abs modifiers) ----
  if (Sop == CanonicalOp::V_ADD_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateFAdd(S0, S1, "fadd"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MUL_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateFMul(S0, S1, "fmul"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_SUB_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateFSub(S0, S1, "fsub"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_SUBREV_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateFSub(S1, S0, "fsubrev"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAX_NUM_F32 || Sop == CanonicalOp::V_MIN_NUM_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Intrinsic::ID Id =
        Sop == CanonicalOp::V_MAX_NUM_F32 ? Intrinsic::maximumnum : Intrinsic::minimumnum;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Id, {Ctx.F32Ty});
    const char *Name = Sop == CanonicalOp::V_MAX_NUM_F32 ? "vmaxnum" : "vminnum";
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fn, {S0, S1}, Name), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAXIMUM_F32) {
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, "v_maximum_f32"))
      return Hr;
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Function *MaxFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::maximum, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(MaxFn, {S0, S1}, "fmaximum"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MINIMUM_F32) {
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, "v_minimum_f32"))
      return Hr;
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Function *MinFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::minimum, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(MinFn, {S0, S1}, "fminimum"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_FMA_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, S1, S2}, "fma"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_FMAC_F32) {
    ParsedReg DstReg = Op.dst();
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *Dv = Ctx.Regs.readReg32(Ctx.B, DstReg);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (Dv->getType() != Ctx.F32Ty) Dv = Ctx.B.CreateBitCast(Dv, Ctx.F32Ty);
    // llvm.fma (not llvm.fmuladd) -- v_fmac_f32 is hardware-guaranteed fused; fmuladd may be split by middle-end passes.
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Ctx.writeReg32(DstReg, Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, S1, Dv}, "fmac"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_fmamk_f32 dst, src0, K, src2: dst = src0 * K + src2
  // Operand order from MC disassembler: srcF(0)=src0, srcF(1)=K (literal),
  // srcF(2)=src2. Same ordering applies to v_fmaak_f32 below.
  if (Sop == CanonicalOp::V_FMAMK_F32) {
    Value *S0 = Op.srcF(0), *K = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (K->getType() != Ctx.F32Ty) K = Ctx.B.CreateBitCast(K, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, K, S2}, "fmamk"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_fmaak_f32 dst, src0, src1, K: dst = src0 * src1 + K
  if (Sop == CanonicalOp::V_FMAAK_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *K = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (K->getType() != Ctx.F32Ty) K = Ctx.B.CreateBitCast(K, Ctx.F32Ty);
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fma, {S0, S1, K}, "fmaak"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_madmk_f16 dst, src0, K, src2: dst = src0 * K + src2
  // v_madak_f16 dst, src0, src1, K: dst = src0 * src1 + K
  // F16 mirror of V_FMAMK_F32 / V_FMAAK_F32. Same operand ordering
  // convention: srcF(0..2) follow the disassembler's order, and the
  // 16-bit literal K lives in the slot named in the mnemonic. Both
  // lower to llvm.fma.f16 (no rounding of the intermediate product),
  // matching VOP2Instructions.td:1206-1210.
  if (Sop == CanonicalOp::V_MADMK_F16 || Sop == CanonicalOp::V_MADAK_F16) {
    Type *F16Ty = Type::getHalfTy(Ctx.C);
    Type *I16Ty = Type::getInt16Ty(Ctx.C);
    auto ToF16 = [&](Value *V) -> Value * {
      Value *Truncated = Ctx.B.CreateTrunc(V, I16Ty);
      return Ctx.B.CreateBitCast(Truncated, F16Ty);
    };
    Value *S0 = ToF16(Op.srcF(0));
    Value *S1 = ToF16(Op.srcF(1));
    Value *S2 = ToF16(Op.srcF(2));
    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma,
                                                     {F16Ty});
    Value *Res = Ctx.B.CreateCall(
        Fma, {S0, S1, S2},
        Sop == CanonicalOp::V_MADMK_F16 ? "madmk_f16" : "madak_f16");
    Value *Bits = Ctx.B.CreateBitCast(Res, I16Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Bits, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // v_fma_f16 dst, src0, src1, src2: dst = fma(src0, src1, src2).
  // VOP3 explicit-source F16 fused multiply-add. The gfx9+ pseudo
  // (V_FMA_F16_gfx9_e64 and its t16/fake16 collapses) carries per-source
  // op_sel (low/high half of the 32-bit VGPR), the usual VOP3 neg/abs
  // source modifiers, and a destination op_sel for half-write placement.
  // Lowers to llvm.fma.f16 to keep the fused-multiply-add semantics.
  if (Sop == CanonicalOp::V_FMA_F16) {
    StringRef OpName = "v_fma_f16";
    bool DstHigh = false;
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, OpName) ||
        !readVOP3F16DstHigh(Di, Hr, OpName, DstHigh))
      return Hr;

    SmallVector<Value *, 3> Srcs;
    for (unsigned I = 0; I < 3; ++I) {
      Value *Src = readOpSelF16(Ctx, Di, Op, Hr, I, OpName);
      if (!Src)
        return Hr;
      Srcs.push_back(Src);
    }

    Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma,
                                                     {Ctx.F16Ty});
    Value *R = Ctx.B.CreateCall(Fma, {Srcs[0], Srcs[1], Srcs[2]}, "fma_f16");
    writeOpSelF16(Ctx, Op, R, DstHigh);
    Hr.Handled = true;
    return Hr;
  }

  // ---- Division helpers (VOP3) ----
  if (Sop == CanonicalOp::V_DIV_SCALE_F32) {
    // `v_div_scale_f32 dst, vcc, src0, src1, src2` scales one operand
    // of a numerator/denominator pair for a subsequent IEEE-conformant
    // divide (rcp + Newton + div_fixup).  The hardware encodes the
    // divide via operand-identity equality in the (src0, src1, src2)
    // triple -- src2 duplicates either src0 or src1 to name which
    // operand is being scaled:
    //
    //   (n, d, n)  -- src0 == src2, both carry the numerator       -> scale numerator.
    //   (d, d, n)  -- src0 == src1, both carry the denominator     -> scale denominator.
    //
    // The LLVM intrinsic `@llvm.amdgcn.div.scale.f32(numer, denom, flag)`
    // takes canonical (numer, denom) and an i1 flag whose convention is
    // documented in `include/llvm/IR/IntrinsicsAMDGPU.td`:
    //   `0 = Denominator, 1 = Numerator`
    // -- the flag selects which of (numer, denom) is the scaling target,
    // and the corresponding bit-pattern is what the hardware backend
    // re-emits as src2 of the lowered instruction.
    //
    // Identity is at the operand level (same register slot, or same
    // literal bit pattern), not at the runtime-value level -- Triton's
    // AMDGPU codegen emits the literal-numerator variant of `1.0 / x`
    // as `(x, x, 1.0) + (1.0, x, 1.0)`, two scale calls whose
    // numerator is a `1.0` inline constant in src2.  Before this
    // handler knew about literal equality, both scale calls fell
    // through the register-only `isSrcReg(2) && isSrcReg(0)` check,
    // decoded as `selectNumerator = false`, and the backend's
    // re-lowering chain collapsed every `1.0 / sqrt(...)` in the
    // translated HSACO to a constant 1.0 via div_fixup's undefined-
    // scale-flag special-case (observable as layer-norm's rstd
    // deterministically reading `0x3f800000` regardless of input).
    // Detecting the literal-matching shape closes that gap without
    // changing the flag convention for the all-register case.
    //
    // Canonical `(numer, denom)` extraction from the hardware triple.
    // The denom always lives in src1.  The numer lives wherever the
    // matched shape identifies it: src0 in scale-numer (where
    // src0 == src2), src2 in scale-denom (where src0 == src1 and
    // src2 is the lone numer-bearing slot).  We pull the numer from
    // the slot that lexically exists in the matched shape -- src0
    // for scale-numer to keep IR identity with the pre-audit all-
    // register handler, src2 for scale-denom to route the numer
    // correctly rather than silently duplicating the denom through
    // s0/s1 (the pre-audit shape).  Modifier symmetry on the
    // matched-identity pair is asserted below before either pick
    // becomes observable: asymmetric modifiers on the duplicated
    // slot are an emitter-ambiguity shape the lifted IR cannot
    // represent faithfully, and we refuse rather than guess.
    auto SameOperand = [&](unsigned A, unsigned B) -> bool {
      bool AIsReg = Op.isSrcReg(A), BIsReg = Op.isSrcReg(B);
      if (AIsReg != BIsReg) return false;
      if (AIsReg) {
        ParsedReg Ra = Op.srcReg(A), Rb = Op.srcReg(B);
        return Ra.RegKind == Rb.RegKind && Ra.BaseIdx == Rb.BaseIdx;
      }
      // Both are non-register operands.  Only compare when both
      // are plain immediates -- other non-register kinds (special
      // encodings that parseReg would map to VCC / EXEC / SRC_*)
      // are not carried through the OpResolver as literals today and
      // the `isSrcReg` check above would have returned true for
      // them, so reaching here guarantees the isImm check is safe.
      //
      // Literal identity is compared via `MCOperand::getImm`, which
      // returns AMDGPU's raw encoded bit pattern -- inline constants
      // come through their special-index encoding (246 for `1.0`,
      // etc.) and 32-bit literals come through their IEEE bit
      // pattern.  Two representations of the same value (e.g. `1.0`
      // as inline-const vs as a 32-bit literal `0x3f800000`) would
      // compare as UNEQUAL at this layer.  That is a pre-condition
      // refuse, not a silent miscompile -- the handler falls through
      // to the three-arm match's `else` arm below and surfaces a
      // diagnostic.  In practice every corpus emitter (Triton's
      // AMDGPU backend, hipcc, libdevice) uses the canonical
      // inline-const encoding for the `1.0 / x` fdiv expansion, so
      // this representation assumption doesn't trip anywhere today;
      // tightening to semantic-value equality is the follow-up if a
      // future emitter surfaces the long-literal form.
      unsigned Ai = Op.srcIdx(A), Bi = Op.srcIdx(B);
      if (!Op.Di.isImm(Ai) || !Op.Di.isImm(Bi)) return false;
      return Op.Di.getImm(Ai) == Op.Di.getImm(Bi);
    };

    bool Src0EqSrc2 = SameOperand(0, 2);
    bool Src0EqSrc1 = SameOperand(0, 1);
    bool ScaleNumerator;
    if (Src0EqSrc2 && !Src0EqSrc1) {
      ScaleNumerator = true;    // (n, d, n)
    } else if (Src0EqSrc1 && !Src0EqSrc2) {
      ScaleNumerator = false;   // (d, d, n)
    } else {
      // All three sources matching is the degenerate `x/x` shape
      // (ambiguous between scale-numer and scale-denom); src2 not
      // matching either of src0/src1 would break the hardware's own
      // divide-protocol and is unreachable from any known codegen
      // emitter.  Refuse loudly rather than guess -- consistent with
      // the "refuse when uncertain" rule in
      // hotswap/docs/wave-size-translation.md.
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          "v_div_scale_f32 operand triple does not match a known "
          "divide-scaling shape: expected (numer, denom, numer) with "
          "src0 == src2 for scale-numerator, or (denom, denom, numer) "
          "with src0 == src1 for scale-denominator.  See handle-valu.cpp "
          "for the decode rule.");
      return Hr;
    }

    // FP-modifier symmetry check on the matched-identity pair.  The
    // hardware's operand-identity protocol makes `src0 == src<M>`
    // (for M = 1 in scale-denom, M = 2 in scale-numer) tell the
    // scale unit "these two slots carry the same operand value."
    // But VOP3 modifiers (abs/neg bits in the `modMap` entry per
    // source index) can be set independently on each slot, which
    // would make the two slots semantically different operands (one
    // `v`, the other `-v` or `abs(v)`).  The hardware's behaviour
    // in that case is undocumented / effectively undefined for the
    // divide protocol -- no known codegen emitter (Triton AMD
    // backend, hipcc, libdevice) produces asymmetric modifiers on
    // the duplicated slot -- and the lifted IR would silently drop
    // one modifier set because we can only thread a single
    // `(numer, denom)` pair through `@llvm.amdgcn.div.scale.f32`.
    // Refuse loudly rather than guess.
    unsigned Peer = ScaleNumerator ? 2u : 1u;
    if (Op.srcMod(0) != Op.srcMod(Peer)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          ScaleNumerator
              ? "v_div_scale_f32 scale-numerator shape (src0 == src2) "
                "has asymmetric FP modifiers on src0 and src2; the "
                "hardware's operand-identity protocol treats both as "
                "the same numerator operand, but the lifted IR can "
                "only carry one modifier set.  No known codegen "
                "emitter produces this shape; refusing rather than "
                "dropping a modifier silently."
              : "v_div_scale_f32 scale-denominator shape (src0 == src1) "
                "has asymmetric FP modifiers on src0 and src1; the "
                "hardware's operand-identity protocol treats both as "
                "the same denominator operand, but the lifted IR can "
                "only carry one modifier set.  No known codegen "
                "emitter produces this shape; refusing rather than "
                "dropping a modifier silently.");
      return Hr;
    }

    // Canonical (numer, denom) sourced from the operand slot that
    // holds each value in the matched shape.  Modifier symmetry was
    // just asserted above, so for scale-numer picking src0 or src2
    // is equivalent -- we take src0 to keep the all-register scale-
    // numer case IR-identical to the pre-fix handler (which also
    // used srcF(0) for numer); for scale-denom, src2 is the only
    // slot that carries the numer at all, so there is no choice.
    // The denom always lives in src1 (src0 aliases it in the
    // scale-denom shape, and src1 is the natural anchor in both).
    Value *Numer = Ctx.B.CreateBitCast(
        ScaleNumerator ? Op.srcF(0) : Op.srcF(2), Ctx.F32Ty);
    Value *Denom = Ctx.B.CreateBitCast(Op.srcF(1), Ctx.F32Ty);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_div_scale,
                                                     {Ctx.F32Ty});
    Value *R = Ctx.B.CreateCall(Fn, {Numer, Denom,
                 ScaleNumerator ? Ctx.B.getTrue() : Ctx.B.getFalse()}, "divscale");
    Ctx.writeReg32(Op.dst(0), Ctx.B.CreateBitCast(Ctx.B.CreateExtractValue(R, 0), Ctx.I32Ty));
    // Write the boolean flag to the actual SDST destination (operand 1):
    // vcc_lo, sN, or null. The kernel saves flags to SGPRs and later
    // restores them to VCC via s_mov_b32 before each v_div_fmas_f32.
    Value *Flag = Ctx.B.CreateExtractValue(R, 1);
    if (Di.NumDefs >= 2 && Di.isReg(1)) {
      ParsedReg FlagDst = Op.dst(1);
      if (FlagDst.RegKind == ParsedReg::VCC)
        Ctx.Regs.storeVCC(Ctx.B, Flag);
      else if (FlagDst.RegKind == ParsedReg::SGPR && FlagDst.BaseIdx >= 0)
        Ctx.Regs.storeSGPR32(Ctx.B, FlagDst.BaseIdx, Ctx.B.CreateZExt(Flag, Ctx.I32Ty));
      // NOREG (null) or unrecognized -> discard the flag
    } else {
      Ctx.Regs.storeVCC(Ctx.B, Flag);
    }
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_DIV_FIXUP_F32) {
    Value *S0 = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.srcF(1), Ctx.F32Ty);
    Value *S2 = Ctx.B.CreateBitCast(Op.srcF(2), Ctx.F32Ty);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_div_fixup,
                                                     {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fn, {S0, S1, S2}, "divfixup"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_DIV_FMAS_F32) {
    Value *S0 = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(Op.srcF(1), Ctx.F32Ty);
    Value *S2 = Ctx.B.CreateBitCast(Op.srcF(2), Ctx.F32Ty);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_div_fmas,
                                                     {Ctx.F32Ty});
    Value *Vcc = Ctx.Regs.loadVCC(Ctx.B);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fn, {S0, S1, S2, Vcc}, "divfmas"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- 3-source integer VOP3 ----
  if (Sop == CanonicalOp::V_ADD3_U32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateAdd(Ctx.B.CreateAdd(Op.src(0), Op.src(1)), Op.src(2), "vadd3"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_LSHL_ADD_U32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateAdd(Ctx.B.CreateShl(Op.src(0), Op.src(1)), Op.src(2), "vlshl_add"));
    Hr.Handled = true;
    return Hr;
  }
  // v_add_lshl_u32: fused three-input "add then shift".
  //   D.u = (S0.u + S1.u) << S2.u[4:0]
  // Unsigned wrap on the add is well-defined (CreateAdd defaults to
  // "may wrap"), matching hardware.  The shift amount must be masked to
  // 5 bits up front -- AMDGPU shifts only consume S2[4:0], but LLVM's
  // `shl` with a shift >= bit-width is poison, so an un-masked `op.src(2)`
  // containing any high bits would silently corrupt the IR.  V_ADD_LSHL
  // has no carry-out and writes no SCC/VCC, so this is the whole op.
  if (Sop == CanonicalOp::V_ADD_LSHL_U32) {
    Value *Sum = Ctx.B.CreateAdd(Op.src(0), Op.src(1));
    Value *Shamt = Ctx.B.CreateAnd(Op.src(2),
                                   ConstantInt::get(Ctx.I32Ty, 0x1F));
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateShl(Sum, Shamt, "vadd_lshl"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_LSHL_OR_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Ctx.B.CreateShl(Op.src(0), Op.src(1)), Op.src(2), "vlshlor"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_AND_OR_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Ctx.B.CreateAnd(Op.src(0), Op.src(1)), Op.src(2), "vandor"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_OR3_B32) {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Ctx.B.CreateOr(Op.src(0), Op.src(1)), Op.src(2), "vor3"));
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 v_xad_u32: dst = (src0 ^ src1) + src2. The .td iselect
  // pattern (VOP3Instructions.td:831) is
  // `ThreeOp_i32_Pats<xor, add, V_XAD_U32_e64>`. Same skeleton
  // as V_AND_OR_B32 / V_LSHL_OR_B32 above with xor+add in place
  // of the inner+outer ops.
  if (Sop == CanonicalOp::V_XAD_U32) {
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateAdd(Ctx.B.CreateXor(Op.src(0), Op.src(1)),
                                   Op.src(2), "vxad"));
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 v_alignbit_b32: funnel-shift right.
  //   dst = ((src0 << 32) | src1) >> (src2 & 0x1F))[31:0]
  // The .td uses the SDAG `fshr` node directly
  // (VOP3Instructions.td:222), so the lift maps 1:1 to
  // `llvm.fshr.i32`. The shift amount is masked to 5 bits in
  // hardware before dispatch -- we mirror that explicit mask
  // here (although LLVM's fshr semantics already implement
  // modulo-bitwidth shifts, the explicit AND keeps the IR
  // shape pinnable and makes the bit-width assumption local).
  if (Sop == CanonicalOp::V_ALIGNBIT_B32) {
    Value *Hi = Op.src(0);
    Value *Lo = Op.src(1);
    Value *Shamt = Ctx.B.CreateAnd(Op.src(2),
        ConstantInt::get(Ctx.I32Ty, 0x1F), "valign_shamt");
    Function *Fshr = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fshr, {Ctx.I32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(Fshr, {Hi, Lo, Shamt}, "valignbit"));
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 v_xor3_b32: 3-way xor. Direct mirror of V_OR3_B32 above
  // -- the .td iselect pattern is `(xor (xor a, b), c)` (see
  // VOP3Instructions.td:1350); both nested and outer xor lift to
  // plain CreateXor with no source modifiers (B32 ops carry only
  // ABS/NEG-style modifiers on the FP forms, not the bitwise
  // ones).
  //
  // Triton-on-gfx1250 cross-16 bitonic-merge IDIOM (special-case below):
  //
  //   v_dual_mov_b32 v_a, v_c :: v_dual_mov_b32 v_b, v_c
  //   v_permlane16_swap_b32 v_a, v_b
  //   v_xor3_b32           v_a, v_a, v_b, v_c
  //
  // After the swap (cross-wired per the ISA), v_a = v_b =
  // partner_v_c.  Then xor3 collapses to v_a = partner ^ partner ^
  // self = self_v_c -- which DEFEATS the algorithmic intent.
  //
  // Triton's `_compare_and_swap` algebraically wants
  //   `iy = ix ^ xor_sum(ix, axis, keep_dims=True) = ix_partner`
  // i.e. v_a should hold `partner_v_c` after this dance, so the
  // following `v_cmp_*` against v_c does a meaningful self-vs-
  // partner compare.  On native gfx942 Triton uses
  // `ds_swizzle_b32 swap:16` (clean: v3 directly gets partner).
  // The gfx1250 codegen swaps + xor3s -- with a literal evaluation
  // the result is `self`, but the algorithmic INTENT is `partner`.
  // Either Triton's gfx1250 codegen relies on a gfx1250-silicon
  // semantic that diverges from the gfx950-style ISA the swap is
  // documented for, or the gfx1250 codegen has a bug -- we can't
  // verify without gfx1250 hardware.  Either way, what we CAN do
  // is emit the algorithmically-intended value (`partner_v_c`)
  // when we recognise this idiom.  The cross-16 bitonic merge
  // then composes correctly with the surrounding `v_cmp` /
  // `cndmask` pair, and the gfx942-lifted kernel produces the
  // same sorted result the gfx942-NATIVE Triton compile already
  // produces (same Python source, different codegen path).
  //
  // Detection (`emitPermLaneSwapXor3PartnerIdiom`):
  //   * src(0) and src(1) both trace through phi-noop wrappers
  //     to `@llvm.amdgcn.ds.bpermute` calls.
  //   * Both bpermutes share the same byte-address argument
  //     (i.e. the same partner-lane selector -- emitted by
  //     `emitPermLaneSwapEmulation`).
  //   * The DATA argument of each bpermute traces (through phis)
  //     to the same SSA value as src(2) -- confirming the
  //     `vdst_in == src0_in == seed` precondition.
  //
  // Substitution: write src(0)'s bpermute result into op.dst().
  // That's `partner_v_c` (the bpermute reads v4_in = v_c from
  // partner lane).
  //
  // Falsely-matched programs would have to (a) feed two
  // bpermutes with identical address arguments, (b) source the
  // same SSA into both bpermutes' data and the third xor3
  // operand, and (c) intend the literal triple-XOR result.  The
  // probability of all three together appearing outside this
  // specific Triton compose is effectively zero -- the existing
  // `BitonicXor3TritonState` GTest pins the rewrite, and the
  // `Permlane16Swap` / `Permlane16SwapWave32` /
  // `Permlane16SwapWave32WaveNative` GTests pin that the swap's
  // standalone semantic is unaffected (they do NOT trigger the
  // idiom because their inputs are distinct).
  if (Sop == CanonicalOp::V_XOR3_B32) {
    // The Triton-on-gfx1250 cross-16 bitonic-merge idiom rewrite
    // (described in detail in the comment block above this `if`)
    // is implemented as a post-PromoteMemToReg LLVM pass in
    // `raiser.cpp` Phase 6.04 -- it cannot match here because
    // hotswap's per-instruction lift uses alloca-backed VGPR
    // storage and `op.src(i)` returns a `load i32, ptr addrspace(5)`
    // rather than the underlying SSA value at this point.  By the
    // time mem2reg has folded the alloca round-trips, the SSA
    // definitions of the bpermute results flow directly into the
    // xor3, and the post-mem2reg pass can pattern-match cleanly.
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateXor(Ctx.B.CreateXor(Op.src(0), Op.src(1)),
                                   Op.src(2), "vxor3"));
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 true16 16-bit no-carry add/sub with op_sel half selection on
  // src0/src1/dst. Unsigned forms use V_{ADD,SUB}_NC_U16 pseudos; signed
  // forms use LLVM's older V_{ADD,SUB}_I16 pseudos even when gfx10+ real
  // mnemonics print the no-carry spelling v_{add,sub}_nc_i16.
  //
  // op_sel routes 16-bit halves of src0/src1 (lo or hi) and selects which
  // half of the 32-bit dst register receives the result; the unselected
  // half of dst is preserved per the RDNA3+ ISA -- that is the *only*
  // reason this handler reads the prior dst value and merges,
  // distinguishing it from the existing V_MAX_U16 / V_MIN_U16 family
  // which assume default op_sel and zero-extend.
  if (Sop == CanonicalOp::V_ADD_NC_U16 ||
      Sop == CanonicalOp::V_SUB_NC_U16 ||
      Sop == CanonicalOp::V_ADD_NC_I16 ||
      Sop == CanonicalOp::V_SUB_NC_I16) {
    bool IsSub = Sop == CanonicalOp::V_SUB_NC_U16 ||
                 Sop == CanonicalOp::V_SUB_NC_I16;
    bool IsSigned = Sop == CanonicalOp::V_ADD_NC_I16 ||
                    Sop == CanonicalOp::V_SUB_NC_I16;
    StringRef OpName = true16AddSubOpName(IsSub, IsSigned);
    std::optional<bool> Clamp = readVOP3Clamp(Di, Hr, OpName);
    if (!Clamp)
      return Hr;

    std::optional<True16OpSel> Sel = readTrue16OpSel(Di, Op, Hr, OpName);
    if (!Sel)
      return Hr;

    Type *I16Ty = Type::getInt16Ty(Ctx.C);
    Value *LHS = extractU16Half(Ctx, Op.src(0), Sel->Src0Hi);
    Value *RHS = extractU16Half(Ctx, Op.src(1), Sel->Src1Hi);
    Value *Result = nullptr;
    StringRef ResultName = true16AddSubResultName(IsSub, IsSigned);
    if (*Clamp) {
      Function *SatFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, true16AddSubSatIntrinsic(IsSub, IsSigned), {I16Ty});
      Result = Ctx.B.CreateCall(SatFn, {LHS, RHS}, ResultName);
    } else {
      Result = IsSub ? Ctx.B.CreateSub(LHS, RHS, ResultName)
                     : Ctx.B.CreateAdd(LHS, RHS, ResultName);
    }

    StringRef MergeName = true16AddSubMergeName(IsSub, IsSigned, Sel->DstHi);
    writeSelectedU16Half(Ctx, Op.dst(), Result, Sel->DstHi, MergeName);
    Hr.Handled = true;
    return Hr;
  }
  // gfx1250 v_add_min/max_s/u32: dst = (s/u)(min/max)((s/u)addsat(src0, src1), src2).
  //
  // LLVM also exposes llvm.amdgcn.add.(min/max).(i/u)32 with an immediate clamp bit, but
  // that target intrinsic cannot be selected for non-gfx1250 targets such as
  // gfx942. Use the generic LLVM form instead: the AMDGPU backend has an
  // explicit selection pattern for it and it remains
  // target-independent when the destination ISA lacks this opcode.
  //
  // Clamp handling: LLVM's AMDGPU modifier docs define integer clamp as
  // clamping to the operation type's representable range. The `(s/u)add.sat.i32`
  // already produces a value in the reperesentable range, and
  // the unsigned min with src2 stays in that same range. The clamp bit is
  // therefore semantically redundant for this CanonicalOp, but we still require the
  // generated immediate operand to be present so an unexpected operand-table
  // shape refuses loudly instead of being guessed.
  if (Sop == CanonicalOp::V_ADD_MIN_U32 || Sop == CanonicalOp::V_ADD_MAX_U32 ||
      Sop == CanonicalOp::V_ADD_MIN_I32 || Sop == CanonicalOp::V_ADD_MAX_I32) {
    std::optional<bool> Clamp = readVOP3Clamp(Di, Hr, Di.Mnemonic);
    if (!Clamp)
      return Hr;

    bool IsSinged = Sop == CanonicalOp::V_ADD_MIN_I32 || Sop == CanonicalOp::V_ADD_MAX_I32;
    bool IsMax =  Sop == CanonicalOp::V_ADD_MAX_U32 || Sop == CanonicalOp::V_ADD_MAX_I32;

    const Intrinsic::ID AddSatId = IsSinged ? Intrinsic::sadd_sat : Intrinsic::uadd_sat;
    const Intrinsic::ID MinId = IsSinged ? Intrinsic::smin : Intrinsic::umin;
    const Intrinsic::ID MaxId = IsSinged ? Intrinsic::smax : Intrinsic::umax;
    const Intrinsic::ID MinMaxId = IsMax ? MaxId : MinId;

    Value *Sum = Ctx.B.CreateIntrinsic(AddSatId, {Ctx.I32Ty}, {Op.src(0), Op.src(1)}, {}, Di.Mnemonic + "_sum");
    Value *Result = Ctx.B.CreateIntrinsic(MinMaxId, {Ctx.I32Ty}, {Sum, Op.src(2)}, {}, Di.Mnemonic);
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_BITOP3_B32 || Sop == CanonicalOp::V_BITOP3_B16) {
    // v_bitop3 dst, src0, src1, src2, imm8
    // For each bit position i, dst[i] = LUT[4*src0[i] + 2*src1[i] + src2[i]]
    // Expand as: result = OR of (minterm_i AND expand(LUT[i])) for i in 0..7
    Value *A = Op.src(0), *B = Op.src(1), *C = Op.src(2);
    Value *Imm = Op.src(3);
    uint64_t LutConst = 0;
    bool LutIsConst = false;
    if (auto *CI = dyn_cast<ConstantInt>(Imm)) {
      LutConst = CI->getZExtValue() & 0xFF;
      LutIsConst = true;
    }
    Value *Na = Ctx.B.CreateNot(A), *Nb = Ctx.B.CreateNot(B), *Nc = Ctx.B.CreateNot(C);
    Value *Result = ConstantInt::get(Ctx.I32Ty, 0);
    Value *Minterms[8] = {
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, Nb), Nc),  // 0: ~a & ~b & ~c
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, Nb), C),   // 1: ~a & ~b &  c
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, B), Nc),   // 2: ~a &  b & ~c
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, B), C),    // 3: ~a &  b &  c
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, Nb), Nc),   // 4:  a & ~b & ~c
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, Nb), C),    // 5:  a & ~b &  c
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, B), Nc),    // 6:  a &  b & ~c
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, B), C),     // 7:  a &  b &  c
    };
    if (LutIsConst) {
      for (int I = 0; I < 8; I++)
        if (LutConst & (1 << I))
          Result = Ctx.B.CreateOr(Result, Minterms[I]);
    } else {
      for (int I = 0; I < 8; I++) {
        Value *Bit = Ctx.B.CreateAnd(Ctx.B.CreateLShr(Imm, ConstantInt::get(Ctx.I32Ty, I)),
                                 ConstantInt::get(Ctx.I32Ty, 1));
        Value *Mask = Ctx.B.CreateSub(ConstantInt::get(Ctx.I32Ty, 0), Bit);
        Result = Ctx.B.CreateOr(Result, Ctx.B.CreateAnd(Minterms[I], Mask));
      }
    }
    if (Sop == CanonicalOp::V_BITOP3_B16)
      Result = Ctx.B.CreateAnd(Result, ConstantInt::get(Ctx.I32Ty, 0xFFFF));
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 v_max3_u32: 3-way unsigned max, ternary. The .td pattern
  // is `AMDGPUumax3` = `umax(umax(a,b), c)`; `llvm.umax.i32`
  // is the canonical target-independent IR spelling of that semantic.
  if (Sop == CanonicalOp::V_MAX3_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1), *S2 = Op.src(2);
    Function *UmaxFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::umax,
                                          {Ctx.I32Ty});
    Value *M01 = Ctx.B.CreateCall(UmaxFn, {S0, S1}, "vmax3_lo");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(UmaxFn, {M01, S2}, "vmax3"));
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 v_min3_u32: 3-way unsigned min, ternary. Symmetric sibling of
  // V_MAX3_U32 above; the .td pattern is `AMDGPUumin3` =
  // `umin(umin(a,b), c)`.
  if (Sop == CanonicalOp::V_MIN3_U32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1), *S2 = Op.src(2);
    Function *UminFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::umin,
                                          {Ctx.I32Ty});
    Value *M01 = Ctx.B.CreateCall(UminFn, {S0, S1}, "vmin3_lo");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(UminFn, {M01, S2}, "vmin3"));
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 v_med3_i32: signed-integer median-of-three.
  // Hardware semantic (VOP3Instructions.td:1796 via AMDGPUsmed3
  // SDAG node) is the standard sort-and-pick-middle for three i32
  // values:
  //   med3_i32(a, b, c) = smax(smin(a, b), smin(smax(a, b), c))
  // We emit it as a pair of `llvm.smin.i32` + `llvm.smax.i32`
  // intrinsics -- these are the canonical IR forms, and the AMDGPU
  // backend pattern-matches the exact `smax(smin(...),
  // smin(smax(...), ...))` shape back to V_MED3_I32 via
  // AMDGPUInstructions.td so the round-trip is structure-preserving
  // (no codegen quality loss). We deliberately do not depend on
  // `llvm.amdgcn.smed3` because: (a) the LLVM IR-level intrinsic
  // is already the most compact lowering for the same pattern;
  // (b) the smin/smax form composes with peephole IR optimisations
  // that smed3 does not (e.g. constant folding when one source is
  // a known bound).
  if (Sop == CanonicalOp::V_MED3_I32) {
    Value *S0 = Op.src(0), *S1 = Op.src(1), *S2 = Op.src(2);
    Function *SminFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::smin, {Ctx.I32Ty});
    Function *SmaxFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::smax, {Ctx.I32Ty});
    Value *Lo = Ctx.B.CreateCall(SminFn, {S0, S1}, "vmed3_lo");
    Value *Hi = Ctx.B.CreateCall(SmaxFn, {S0, S1}, "vmed3_hi");
    Value *Clamped = Ctx.B.CreateCall(SminFn, {Hi, S2}, "vmed3_clamp");
    Ctx.writeReg32(
        Op.dst(),
        Ctx.B.CreateCall(SmaxFn, {Lo, Clamped}, "vmed3"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAX3_NUM_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);
    Function *MaxFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::maximumnum, {Ctx.F32Ty});
    Value *M01 = Ctx.B.CreateCall(MaxFn, {S0, S1});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(MaxFn, {M01, S2}, "max3"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // IEEE-754 2019 ternary maximum: NaN-propagating 3-source reduction.
  // Same shape as V_MAX3_NUM_F32 above but uses Intrinsic::maximum (NaN-
  // propagating) instead of Intrinsic::maximumnum (numeric operand preferred over NaN), matching
  // the gfx12 v_maximum3_f32 / v_minimum3_f32 hardware semantics.
  if (Sop == CanonicalOp::V_MAXIMUM3_F32 ||
      Sop == CanonicalOp::V_MINIMUM3_F32) {
    StringRef OpName = (Sop == CanonicalOp::V_MAXIMUM3_F32) ? "v_maximum3_f32"
                                                            : "v_minimum3_f32";
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, OpName))
      return Hr;
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);
    Intrinsic::ID IntrId = (Sop == CanonicalOp::V_MAXIMUM3_F32)
                               ? Intrinsic::maximum
                               : Intrinsic::minimum;
    const char *OutName =
        (Sop == CanonicalOp::V_MAXIMUM3_F32) ? "fmaximum3" : "fminimum3";
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, IntrId,
                                                     {Ctx.F32Ty});
    Value *R01 = Ctx.B.CreateCall(Fn, {S0, S1});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(
                                  Ctx.B.CreateCall(Fn, {R01, S2}, OutName),
                                  Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // IEEE-754 2019 ternary clamp pair:
  //   v_maximumminimum_f32: minimum(maximum(S0, S1), S2)
  //   v_minimummaximum_f32: maximum(minimum(S0, S1), S2)
  // These are the NaN-propagating non-.NUM forms, so they must use
  // llvm.maximum / llvm.minimum rather than maxnum / minnum.
  if (Sop == CanonicalOp::V_MAXIMUMMINIMUM_F32 ||
      Sop == CanonicalOp::V_MINIMUMMAXIMUM_F32) {
    const bool MaxThenMin = Sop == CanonicalOp::V_MAXIMUMMINIMUM_F32;
    StringRef OpName = MaxThenMin ? "v_maximumminimum_f32"
                                  : "v_minimummaximum_f32";
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, OpName))
      return Hr;

    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);

    Intrinsic::ID InnerId =
        MaxThenMin ? Intrinsic::maximum : Intrinsic::minimum;
    Intrinsic::ID OuterId =
        MaxThenMin ? Intrinsic::minimum : Intrinsic::maximum;
    Function *InnerFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, InnerId, {Ctx.F32Ty});
    Function *OuterFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, OuterId, {Ctx.F32Ty});
    const char *InnerName =
        MaxThenMin ? "vmaximumminimum_inner" : "vminimummaximum_inner";
    const char *OutName =
        MaxThenMin ? "vmaximumminimum" : "vminimummaximum";
    Value *R01 = Ctx.B.CreateCall(InnerFn, {S0, S1}, InnerName);
    Value *R = Ctx.B.CreateCall(OuterFn, {R01, S2}, OutName);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // IEEE-754 2019 f16 maximum/minimum family. Source and destination op_sel
  // select the low/high half; writes merge with the unselected destination
  // half preserved.
  if (Sop == CanonicalOp::V_MAXIMUM_F16 ||
      Sop == CanonicalOp::V_MINIMUM_F16 ||
      Sop == CanonicalOp::V_MAXIMUM3_F16 ||
      Sop == CanonicalOp::V_MINIMUM3_F16 ||
      Sop == CanonicalOp::V_MAXIMUMMINIMUM_F16 ||
      Sop == CanonicalOp::V_MINIMUMMAXIMUM_F16) {
    const bool IsBinary = Sop == CanonicalOp::V_MAXIMUM_F16 ||
                          Sop == CanonicalOp::V_MINIMUM_F16;
    const bool IsThreeSame = Sop == CanonicalOp::V_MAXIMUM3_F16 ||
                             Sop == CanonicalOp::V_MINIMUM3_F16;
    const bool MaxThenMin = Sop == CanonicalOp::V_MAXIMUMMINIMUM_F16;
    StringRef OpName;
    switch (Sop) {
    case CanonicalOp::V_MAXIMUM_F16:
      OpName = "v_maximum_f16";
      break;
    case CanonicalOp::V_MINIMUM_F16:
      OpName = "v_minimum_f16";
      break;
    case CanonicalOp::V_MAXIMUM3_F16:
      OpName = "v_maximum3_f16";
      break;
    case CanonicalOp::V_MINIMUM3_F16:
      OpName = "v_minimum3_f16";
      break;
    case CanonicalOp::V_MAXIMUMMINIMUM_F16:
      OpName = "v_maximumminimum_f16";
      break;
    case CanonicalOp::V_MINIMUMMAXIMUM_F16:
      OpName = "v_minimummaximum_f16";
      break;
    default:
      llvm_unreachable("filtered by outer f16 IEEE switch");
    }

    bool DstHigh = false;
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, OpName) ||
        !readVOP3F16DstHigh(Di, Hr, OpName, DstHigh))
      return Hr;

    const unsigned NumSrcs = IsBinary ? 2 : 3;
    SmallVector<Value *, 3> Srcs;
    for (unsigned I = 0; I < NumSrcs; ++I) {
      Value *Src = readOpSelF16(Ctx, Di, Op, Hr, I, OpName);
      if (!Src)
        return Hr;
      Srcs.push_back(Src);
    }

    Intrinsic::ID InnerId;
    Intrinsic::ID OuterId;
    if (IsBinary || IsThreeSame) {
      InnerId = (Sop == CanonicalOp::V_MAXIMUM_F16 ||
                 Sop == CanonicalOp::V_MAXIMUM3_F16)
                    ? Intrinsic::maximum
                    : Intrinsic::minimum;
      OuterId = InnerId;
    } else {
      InnerId = MaxThenMin ? Intrinsic::maximum : Intrinsic::minimum;
      OuterId = MaxThenMin ? Intrinsic::minimum : Intrinsic::maximum;
    }

    Function *InnerFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, InnerId, {Ctx.F16Ty});
    Value *R = Ctx.B.CreateCall(InnerFn, {Srcs[0], Srcs[1]},
                                Twine(OpName) + "_inner");
    if (!IsBinary) {
      Function *OuterFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, OuterId, {Ctx.F16Ty});
      R = Ctx.B.CreateCall(OuterFn, {R, Srcs[2]}, OpName);
    }
    writeOpSelF16(Ctx, Op, R, DstHigh);
    Hr.Handled = true;
    return Hr;
  }

  // F16 .NUM clamp pair: IEEE minimumNumber/maximumNumber with full
  // source/destination op_sel handling.
  if (Sop == CanonicalOp::V_MINMAX_NUM_F16 ||
      Sop == CanonicalOp::V_MAXMIN_NUM_F16) {
    const bool MinThenMax = Sop == CanonicalOp::V_MINMAX_NUM_F16;
    StringRef OpName = MinThenMax ? "v_minmax_num_f16" : "v_maxmin_num_f16";
    bool DstHigh = false;
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, OpName) ||
        !readVOP3F16DstHigh(Di, Hr, OpName, DstHigh))
      return Hr;

    SmallVector<Value *, 3> Srcs;
    for (unsigned I = 0; I < 3; ++I) {
      Value *Src = readOpSelF16(Ctx, Di, Op, Hr, I, OpName);
      if (!Src)
        return Hr;
      Srcs.push_back(Src);
    }

    Function *MaxFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::maximumnum, {Ctx.F16Ty});
    Function *MinFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::minimumnum, {Ctx.F16Ty});
    Function *InnerFn = MinThenMax ? MinFn : MaxFn;
    Function *OuterFn = MinThenMax ? MaxFn : MinFn;
    Value *Inner = Ctx.B.CreateCall(InnerFn, {Srcs[0], Srcs[1]},
                                    MinThenMax ? "vminmax_num_f16_inner"
                                               : "vmaxmin_num_f16_inner");
    Value *R = Ctx.B.CreateCall(OuterFn, {Inner, Srcs[2]},
                                MinThenMax ? "vminmax_num_f16"
                                           : "vmaxmin_num_f16");
    writeOpSelF16(Ctx, Op, R, DstHigh);
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 .NUM clamp pair: IEEE minimumNumber/maximumNumber semantics.
  //   v_minmax_num_f32: maximumnum(minimumnum(S0, S1), S2)
  //   v_maxmin_num_f32: minimumnum(maximumnum(S0, S1), S2)
  if (Sop == CanonicalOp::V_MINMAX_NUM_F32 ||
      Sop == CanonicalOp::V_MAXMIN_NUM_F32) {
    const bool MinThenMax = Sop == CanonicalOp::V_MINMAX_NUM_F32;
    StringRef OpName = MinThenMax ? "v_minmax_num_f32" : "v_maxmin_num_f32";
    if (!requireDefaultVOP3FpValuOutputMods(Di, Hr, OpName))
      return Hr;

    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);
    Function *MaxFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::maximumnum, {Ctx.F32Ty});
    Function *MinFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::minimumnum, {Ctx.F32Ty});
    Function *InnerFn = MinThenMax ? MinFn : MaxFn;
    Function *OuterFn = MinThenMax ? MaxFn : MinFn;
    const char *InnerName = MinThenMax ? "vminmax_inner" : "vmaxmin_inner";
    const char *OutName = MinThenMax ? "vminmax_num" : "vmaxmin_num";
    Value *Inner = Ctx.B.CreateCall(InnerFn, {S0, S1}, InnerName);
    Value *R = Ctx.B.CreateCall(OuterFn, {Inner, S2}, OutName);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MIN3_NUM_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);
    Function *MinFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::minimumnum, {Ctx.F32Ty});
    Value *M01 = Ctx.B.CreateCall(MinFn, {S0, S1});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(MinFn, {M01, S2}, "min3"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MED3_NUM_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *S2 = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (S2->getType() != Ctx.F32Ty) S2 = Ctx.B.CreateBitCast(S2, Ctx.F32Ty);
    Function *MaxFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::maximumnum, {Ctx.F32Ty});
    Function *MinFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::minimumnum, {Ctx.F32Ty});
    Value *Mn01 = Ctx.B.CreateCall(MinFn, {S0, S1});
    Value *Mx01 = Ctx.B.CreateCall(MaxFn, {S0, S1});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Ctx.B.CreateCall(MaxFn, {Mn01, Ctx.B.CreateCall(MinFn, {Mx01, S2})}, "med3"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_cvt_pkrtz_f16_f32: pack two f32 into <2 x f16> with round-to-zero.
  // Maps directly onto the dedicated hardware intrinsic so the backend
  // keeps the RTZ rounding mode (a plain FPTrunc uses round-to-nearest).
  if (Sop == CanonicalOp::V_CVT_PKRTZ_F16_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_cvt_pkrtz);
    Value *V2h = Ctx.B.CreateCall(Fn, {S0, S1}, "pkrtz");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(V2h, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_cvt_pk_f16_f32: pack two f32 into <2 x f16> with round-to-nearest-even
  // (the default IEEE rounding). No dedicated intrinsic exists; a pair of
  // FPTrunc operations followed by a packed i32 assembly is the canonical
  // lowering and the backend recognises the pattern.
  if (Sop == CanonicalOp::V_CVT_PK_F16_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Type *HalfTy = Type::getHalfTy(Ctx.C);
    Type *I16Ty = Type::getInt16Ty(Ctx.C);
    Value *H0 = Ctx.B.CreateFPTrunc(S0, HalfTy, "pk_h0");
    Value *H1 = Ctx.B.CreateFPTrunc(S1, HalfTy, "pk_h1");
    Value *B0 = Ctx.B.CreateZExt(Ctx.B.CreateBitCast(H0, I16Ty), Ctx.I32Ty);
    Value *B1 = Ctx.B.CreateZExt(Ctx.B.CreateBitCast(H1, I16Ty), Ctx.I32Ty);
    Ctx.writeReg32(Op.dst(),
        Ctx.B.CreateOr(B0, Ctx.B.CreateShl(B1, 16), "pk_f16"));
    Hr.Handled = true;
    return Hr;
  }
  // v_cvt_scalef32_pk_fp4_f32 vdst, src0_f32, src1_f32, scale_f32 op_sel:[..]
  //
  // Converts two f32 sources to FP4 and packs them into one of the four 8-bit
  // slots of vdst (selected by op_sel bits 0..3), using a scalar f32 scale.
  // The remaining slots of the old vdst value are preserved -- this is
  // captured by the intrinsic's tied `old_vdst` argument.
  if (Sop == CanonicalOp::V_CVT_SCALEF32_PK_FP4_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1), *Scale = Op.srcF(2);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    if (Scale->getType() != Ctx.F32Ty)
      Scale = Ctx.B.CreateBitCast(Scale, Ctx.F32Ty);
    // Extract the destination nibble index from op_sel. LLVM disasm prints
    // op_sel:[0,0,0,0] with the 4th entry being the slot selector.
    int OpSel[4] = {0, 0, 0, 0};
    StringRef Text(Di.FullText);
    auto Pos = Text.find("op_sel:");
    if (Pos != StringRef::npos) {
      auto Brk = Text.find('[', Pos);
      auto End = Text.find(']', Brk);
      if (Brk != StringRef::npos && End != StringRef::npos) {
        StringRef Inner = Text.slice(Brk + 1, End);
        SmallVector<StringRef, 4> Parts;
        Inner.split(Parts, ',');
        for (unsigned I = 0; I < Parts.size() && I < 4; I++) {
          int Val = 0;
          if (!Parts[I].trim().getAsInteger(10, Val))
            OpSel[I] = Val;
        }
      }
    }
    // Dest-slot index is packed as bits[3:2]+bit[0] per the HW op_sel
    // layout (see LLVM's SIInstrInfo::lowerScaleCvt for reference); for the
    // common `op_sel:[0,0,0,0]` form the selector is simply 0.
    unsigned DstSel = static_cast<unsigned>(OpSel[3]);
    Value *OldVdst = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_cvt_scalef32_pk_fp4_f32);
    Value *R = Ctx.B.CreateCall(
        Fn, {OldVdst, S0, S1, Scale, ConstantInt::get(Ctx.I32Ty, DstSel)},
        "scalef32_pk_fp4");
    Ctx.writeReg32(Op.dst(), R);
    Hr.Handled = true;
    return Hr;
  }
  // v_mov_b64 vdst:64, src:64
  if (Sop == CanonicalOp::V_MOV_B64) {
    Ctx.writeReg64(Op.dst(), Op.src64(0));
    Hr.Handled = true;
    return Hr;
  }
  // v_swap_b32 vdstA, vdstB / uses vdstA, vdstB - exchange two VGPRs.
  // MC encoding has two defs (vdst, vdst_in) and two uses (src0, src0_in).
  // The old values of both registers swap: src0 -> vdst and old-vdst ->
  // vdst_in.
  if (Sop == CanonicalOp::V_SWAP_B32) {
    // vdst = old src0; vdst_in(== src0's slot) = old vdst.
    ParsedReg DstA = Op.dst(0);
    ParsedReg DstB = (Di.NumDefs >= 2) ? Op.dst(1) : Op.srcReg(0);
    Value *VA = Ctx.Regs.readReg32(Ctx.B, DstA);
    Value *VB = Ctx.Regs.readReg32(Ctx.B, DstB);
    Ctx.writeReg32(DstA, VB);
    Ctx.writeReg32(DstB, VA);
    Hr.Handled = true;
    return Hr;
  }
  // v_cvt_f32_bf16: low 16 bits of src are interpreted as bfloat16.
  if (Sop == CanonicalOp::V_CVT_F32_BF16) {
    Type *BfTy = Type::getBFloatTy(Ctx.C);
    Type *I16Ty = Type::getInt16Ty(Ctx.C);
    Value *Bits = Ctx.B.CreateTrunc(Op.src(0), I16Ty);
    Value *Bf = Ctx.B.CreateBitCast(Bits, BfTy);
    Value *F = Ctx.B.CreateFPExt(Bf, Ctx.F32Ty, "cvt_bf16");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(F, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // v_bfm_b32: D = ((1 << src0[4:0]) - 1) << src1[4:0]
  if (Sop == CanonicalOp::V_BFM_B32) {
    Value *Width  = Ctx.B.CreateAnd(Op.src(0),
        ConstantInt::get(Ctx.I32Ty, 0x1F));
    Value *Offset = Ctx.B.CreateAnd(Op.src(1),
        ConstantInt::get(Ctx.I32Ty, 0x1F));
    Value *Ones   = Ctx.B.CreateSub(
        Ctx.B.CreateShl(ConstantInt::get(Ctx.I32Ty, 1), Width),
        ConstantInt::get(Ctx.I32Ty, 1));
    // width==0 must yield 0 -- the 1<<0 base case would otherwise leave a
    // single bit set. Mask it out explicitly rather than relying on the
    // subtraction underflow.
    Value *IsZero = Ctx.B.CreateICmpEQ(Width, ConstantInt::get(Ctx.I32Ty, 0));
    Ones = Ctx.B.CreateSelect(IsZero, ConstantInt::get(Ctx.I32Ty, 0), Ones);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateShl(Ones, Offset, "bfm"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_CVT_PK_BF16_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    auto *BfTy = Type::getBFloatTy(Ctx.C);
    Value *Bf0 = Ctx.B.CreateFPTrunc(S0, BfTy, "tobf16_0");
    Value *Bf1 = Ctx.B.CreateFPTrunc(S1, BfTy, "tobf16_1");
    Value *Bits0 = Ctx.B.CreateZExt(Ctx.B.CreateBitCast(Bf0, Type::getInt16Ty(Ctx.C)), Ctx.I32Ty);
    Value *Bits1 = Ctx.B.CreateZExt(Ctx.B.CreateBitCast(Bf1, Type::getInt16Ty(Ctx.C)), Ctx.I32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Bits0, Ctx.B.CreateShl(Bits1, 16), "pk_bf16"));
    Hr.Handled = true;
    return Hr;
  }
  // gfx1250 v_cvt_scalef32_pk8_fp8_f32 vdst:64, src0:256 (<8 x f32>), src1:32 (Scale)
  // Profile VOP_V2I32_V8F32_F32 (VOP3Instructions.td:1883):
  //   dst  = <2 x i32>           (8 Packed FP8 bytes)
  //   src0 = <8 x f32>            (8 consecutive VGPRs holding the f32 inputs)
  //   src1 = f32                  (broadcast Scale multiplier)
  // The intrinsic semantics: each output FP8[i] = cvt_fp8(src0[i] * src1).
  //
  // Same-target gfx1250: emit `int_amdgcn_cvt_scalef32_pk8_fp8_f32` directly.
  // Cross-target targets with FP8 conversion support: software-emulate via:
  //    Scaled = src0 * splat(src1)
  //    dword0 = pk_fp8(Scaled[0..1]) | (pk_fp8(Scaled[2..3]) << 16)
  //    dword1 = pk_fp8(Scaled[4..5]) | (pk_fp8(Scaled[6..7]) << 16)
  // using `int_amdgcn_cvt_pk_fp8_f32`. The numeric differences vs the
  // gfx1250 hardware path: pk_fp8 uses the same
  // round-to-nearest-even mantissa rounding and same exponent saturation,
  // so the structural delta is that the gfx1250 op fuses the f32 Scale
  // multiply with the pack while this expansion materialises the multiply
  // explicitly before packing. This opcode profile exposes src1 as the f32
  // Scale operand; Scale-format variants use different opcodes/profiles and
  // should get their own lowering/refusal rather than sharing this path.
  if (Sop == CanonicalOp::V_CVT_SCALEF32_PK8_FP8_F32) {
    ParsedReg SrcReg0 = Op.srcReg(0);
    Value *Scale = Op.srcF(1);
    if (Scale->getType() != Ctx.F32Ty)
      Scale = Ctx.B.CreateBitCast(Scale, Ctx.F32Ty);

    auto *V8F32Ty = FixedVectorType::get(Ctx.F32Ty, 8);
    Value *Src8 = Ctx.Regs.readRegVec(Ctx.B, SrcReg0, V8F32Ty);

    Value *Result = nullptr;
    if (Ctx.TargetIsa.HasTensorOps) {
      Function *CvtFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_cvt_scalef32_pk8_fp8_f32);
      Result = Ctx.B.CreateCall(CvtFn, {Src8, Scale}, "cvt_scalef32_pk8_fp8");
    } else if (Ctx.TargetIsa.HasFP8ConversionInsts) {
      // FP8 conversion emulation for targets such as gfx942/gfx950.
      Value *ScaleSplat = Ctx.B.CreateVectorSplat(8, Scale, "scale_splat");
      Value *Scaled = Ctx.B.CreateFMul(Src8, ScaleSplat, "scaled");
      Value *ZeroI32 = ConstantInt::get(Ctx.I32Ty, 0);
      Function *PkFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_cvt_pk_fp8_f32);
      auto ExtractF = [&](unsigned i) {
        return Ctx.B.CreateExtractElement(Scaled, i);
      };
      Value *Dw0Lo = Ctx.B.CreateCall(
          PkFn,
          {ExtractF(0), ExtractF(1), ZeroI32,
           ConstantInt::get(Ctx.I1Ty, 0)},
          "pk_fp8_01");
      Value *Dw0 = Ctx.B.CreateCall(
          PkFn,
          {ExtractF(2), ExtractF(3), Dw0Lo,
           ConstantInt::get(Ctx.I1Ty, 1)},
          "pk_fp8_23");
      Value *Dw1Lo = Ctx.B.CreateCall(
          PkFn,
          {ExtractF(4), ExtractF(5), ZeroI32,
           ConstantInt::get(Ctx.I1Ty, 0)},
          "pk_fp8_45");
      Value *Dw1 = Ctx.B.CreateCall(
          PkFn,
          {ExtractF(6), ExtractF(7), Dw1Lo,
           ConstantInt::get(Ctx.I1Ty, 1)},
          "pk_fp8_67");
      auto *V2I32Ty = FixedVectorType::get(Ctx.I32Ty, 2);
      Value *Packed = PoisonValue::get(V2I32Ty);
      Packed = Ctx.B.CreateInsertElement(Packed, Dw0, (uint64_t)0);
      Packed = Ctx.B.CreateInsertElement(Packed, Dw1, (uint64_t)1);
      Result = Packed;
    } else {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          "v_cvt_scalef32_pk8_fp8_f32 requires either HasTensorOps "
          "(gfx1250 native) or HasFP8ConversionInsts "
          "(int_amdgcn_cvt_pk_fp8_f32); this target has neither.");
      return Hr;
    }
    Ctx.writeRegVec(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_CVT_PK_FP8_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    // v_cvt_pk_fp8_f32 packs two f32 into two fp8 values in the low 16 bits.
    // The "old" value and word_sel determine where in the dest the result goes.
    // src2 = old value, src3 (imm) = word_sel.
    // Use the LLVM intrinsic which handles this correctly.
    Value *OldVal = (Op.nSrcs() >= 3) ? Op.src(2) : ConstantInt::get(Ctx.I32Ty, 0);
    bool WordSel = (Op.nSrcs() >= 4 && Di.isImm(Op.srcIdx(3))) ? (Op.srcImm(3) != 0) : false;
    Function *CvtFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_cvt_pk_fp8_f32);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(CvtFn,
        {S0, S1, OldVal, ConstantInt::get(Ctx.I1Ty, WordSel)}, "pk_fp8"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_CVT_PK_BF8_F32) {
    Value *S0 = Op.srcF(0), *S1 = Op.srcF(1);
    if (S0->getType() != Ctx.F32Ty) S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
    if (S1->getType() != Ctx.F32Ty) S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    Value *OldVal = (Op.nSrcs() >= 3) ? Op.src(2) : ConstantInt::get(Ctx.I32Ty, 0);
    bool WordSel = (Op.nSrcs() >= 4 && Di.isImm(Op.srcIdx(3))) ? (Op.srcImm(3) != 0) : false;
    Function *CvtFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_cvt_pk_bf8_f32);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(CvtFn,
        {S0, S1, OldVal, ConstantInt::get(Ctx.I1Ty, WordSel)}, "pk_bf8"));
    Hr.Handled = true;
    return Hr;
  }
  // VOP1 read-side companions: v_cvt_pk_f32_{fp8,bf8} expand 16 bits
  // of the i32 src into a v2f32 written to the dst VGPR pair. The
  // word selector (which 16-bit half of src to decode) lives in
  // op_sel:[0] for the e64 / VOP3 form and is parsed from di.fullText
  // -- we do not have a first-class modifier channel in OperandView.
  // The dst op_sel slot (op_sel:[1]) is irrelevant: the destination
  // is a v2f32 pair, not a half-register, so the assembler always
  // prints `0` there. We refuse loudly if op_sel parsing produces a
  // value outside {0,1} so corpus drift surfaces immediately rather
  // than silently flipping the word selector. Lowering selects the
  // matching `llvm.amdgcn.cvt.pk.f32.{fp8,bf8}` intrinsic and
  // bitcasts its v2f32 result to i64 before writeReg64.
  if (Sop == CanonicalOp::V_CVT_PK_F32_FP8 || Sop == CanonicalOp::V_CVT_PK_F32_BF8) {
    int WordSelInt = 0;
    StringRef Text(Di.FullText);
    auto Pos = Text.find("op_sel:");
    if (Pos != StringRef::npos) {
      auto Brk = Text.find('[', Pos);
      auto End = Text.find(']', Brk);
      if (Brk != StringRef::npos && End != StringRef::npos) {
        StringRef Inner = Text.slice(Brk + 1, End);
        SmallVector<StringRef, 4> Parts;
        Inner.split(Parts, ',');
        if (!Parts.empty()) {
          int Parsed = 0;
          if (Parts[0].trim().getAsInteger(10, Parsed) ||
              (Parsed != 0 && Parsed != 1)) {
            Hr.Failure = RaiseFailure::unsupportedInstructionForm(
                Di, "VOP3",
                "unparseable or out-of-range op_sel[0] (expected 0 or 1)");
            return Hr;
          }
          WordSelInt = Parsed;
        }
      }
    }
    Value *Src = Op.src(0);
    if (Src->getType() != Ctx.I32Ty)
      Src = Ctx.B.CreateBitOrPointerCast(Src, Ctx.I32Ty);
    Intrinsic::ID Iid = (Sop == CanonicalOp::V_CVT_PK_F32_FP8)
                            ? Intrinsic::amdgcn_cvt_pk_f32_fp8
                            : Intrinsic::amdgcn_cvt_pk_f32_bf8;
    Function *CvtFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Iid);
    Value *V2 = Ctx.B.CreateCall(CvtFn,
        {Src, ConstantInt::get(Ctx.I1Ty, WordSelInt != 0)},
        Sop == CanonicalOp::V_CVT_PK_F32_FP8 ? "cvt_pk_f32_fp8"
                                       : "cvt_pk_f32_bf8");
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(V2, Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  // VOP1 single-lane v_cvt_f32_{fp8,bf8}: decode one 8-bit lane of
  // src into f32. The corpus only ever emits the e64 form with no
  // op_sel (byte_sel=0) -- the SDWA / op_sel-bearing encodings, which
  // would let LLVM's isel pick byte 1/2/3, are not present in any
  // gfx1250 kernel today. We refuse loudly if disassembly carries an
  // op_sel: marker so corpus drift surfaces instead of a silent
  // byte-0 collapse.
  if (Sop == CanonicalOp::V_CVT_F32_FP8 || Sop == CanonicalOp::V_CVT_F32_BF8) {
    StringRef Text(Di.FullText);
    if (Text.contains("op_sel:") || Text.contains("_sdwa")) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "non-default op_sel/sdwa byte_sel on v_cvt_f32_{fp8,bf8} "
          "(only the byte_sel=0 e64 form is wired today)");
      return Hr;
    }
    Value *Src = Op.src(0);
    if (Src->getType() != Ctx.I32Ty)
      Src = Ctx.B.CreateBitOrPointerCast(Src, Ctx.I32Ty);
    Intrinsic::ID Iid = (Sop == CanonicalOp::V_CVT_F32_FP8)
                            ? Intrinsic::amdgcn_cvt_f32_fp8
                            : Intrinsic::amdgcn_cvt_f32_bf8;
    Function *CvtFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Iid);
    Value *F = Ctx.B.CreateCall(CvtFn,
        {Src, ConstantInt::get(Ctx.I32Ty, 0)},
        Sop == CanonicalOp::V_CVT_F32_FP8 ? "cvt_f32_fp8" : "cvt_f32_bf8");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(F, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // VOP3 gfx1250-only scaled packed-8 FP4 -> BF16 convert.
  //
  // Hardware shape (AMDGPUGenInstrInfo.inc / VOP3Instructions.td:1873,
  // opcode V_CVT_SCALE_PK8_BF16_FP4_e64, VOP3 opcode 0x2a0):
  //     0: vdst        (VReg_128 aligned -- 4 consecutive VGPRs,
  //                     written as <8 x bfloat> / 128 bits)
  //     1: src0        (VGPR_32 -- 1 VGPR, packed 8xFP4 in the i32
  //                     bits, nibble 0 = lane 0, nibble 7 = lane 7)
  //     2: src1        (VSrc_b32 -- scale, E8M0 encoded in an i32)
  //     3: scale_sel   (4-bit ImmArg, range 0..15 per
  //                     `AMDGPUCvtScaleIntrinsic` in
  //                     IntrinsicsAMDGPU.td:686.  The AMD ISA spec
  //                     definition of scale_sel's 4-bit semantics
  //                     for the packed-8 FP4 shape is not currently
  //                     reproduced in this tree; the captured gfx1250
  //                     corpus (`scope_discovery/kernels/
  //                     _matmul_ogs_{06d912ce88af,0af655e6ea2b}.hsaco`)
  //                     uses only `scale_sel == 0` across 128
  //                     instances combined, which has the unambiguous
  //                     reading "the scale byte is the low byte of
  //                     the 32-bit scale register".  Both handler
  //                     arms REFUSE scale_sel != 0 loudly until the
  //                     spec is pinned.).
  //
  // LLVM lowering (IntrinsicsAMDGPU.td:688):
  //   declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(
  //       i32 %src, i32 %scale, i32 immarg %scale_sel)
  //
  // Dispatch
  // --------
  //
  //   * `ctx.targetIsa.hasTensorOps` (gfx1250 or any future target
  //     that ships the same VOP3 family): emit the native intrinsic
  //     directly and let the backend select the hardware instruction.
  //
  //   * Otherwise (cross-target: gfx942 / gfx950): emit the per-nibble
  //     bit-algebra dequantisation expansion from
  //     `emitCvtScalePk8Bf16Fp4CrossTargetExpansion` above.  Bit-exact
  //     against the hardware primitive on bit-valid inputs within the
  //     declared support set (see `hotswap/docs/matrix-translation.md
  //     §7.4`).
  //
  // Both arms share the same operand-shape validation (src/scale i32,
  // `scale_sel` immediate, `scale_sel == 0` or refuse) so a corpus
  // drift surfaces on both paths rather than only on whichever one
  // happened to run.
  if (Sop == CanonicalOp::V_CVT_SCALE_PK8_BF16_FP4) {
    unsigned Opc = Di.Inst.getOpcode();
    int SelIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::scale_sel);
    if (SelIdx < 0 || !Di.isImm(SelIdx)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          "v_cvt_scale_pk8_bf16_fp4 missing OpName::scale_sel "
          "immediate operand -- operand table mismatch");
      return Hr;
    }
    int64_t ScaleSel = Di.getImm(SelIdx);

    // Declared support set: scale_sel == 0 only.  The 4-bit
    // scale_sel field's semantics for the packed-8 FP4 shape aren't
    // pinned in any doc in-tree (see comment block above); the
    // captured corpus uses only scale_sel == 0 across both blobs
    // that emit this primitive.  Refusing other values is the
    // "fail loud on declared-support-set boundary" discipline --
    // same shape as the refusal-of-non-default-op_sel check on
    // V_CVT_F32_{FP8,BF8} higher in this file.
    if (ScaleSel != 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          "v_cvt_scale_pk8_bf16_fp4 scale_sel != 0 is outside the "
          "declared support set (AMD ISA spec semantics for the "
          "4-bit scale_sel field on packed-8 FP4 are not pinned "
          "in-tree today; captured corpus uses only scale_sel=0 "
          "across every instance) -- see hotswap/docs/"
          "matrix-translation.md §7.4");
      return Hr;
    }

    Value *Src = Op.src(0);
    if (Src->getType() != Ctx.I32Ty)
      Src = Ctx.B.CreateBitOrPointerCast(Src, Ctx.I32Ty);
    Value *Scale = Op.src(1);
    if (Scale->getType() != Ctx.I32Ty)
      Scale = Ctx.B.CreateBitOrPointerCast(Scale, Ctx.I32Ty);

    Value *Result;
    if (Ctx.TargetIsa.HasTensorOps) {
      // Same-target arm: emit the LLVM intrinsic that lowers 1:1 to
      // the hardware opcode.  The write-back path bitcasts the
      // <8 x bfloat> to i128 before handing to writeRegVec.
      Function *CvtFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp4);
      Result = Ctx.B.CreateCall(
          CvtFn,
          {Src, Scale, ConstantInt::get(Ctx.I32Ty, ScaleSel)},
          "cvt_scale_pk8_bf16_fp4");
    } else {
      // Cross-target arm: bit-algebra per-nibble dequantisation,
      // bit-exact against the hardware primitive's output for
      // scale_sel == 0 on every (packed_fp4, scale) input in the
      // declared support set.  See
      // `emitCvtScalePk8Bf16Fp4CrossTargetExpansion` above.
      Result = emitCvtScalePk8Bf16Fp4CrossTargetExpansion(Ctx, Src, Scale);
    }
    Ctx.writeRegVec(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }

  if (Sop == CanonicalOp::V_PERM_B32) {
    Function *PermFn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_perm);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(PermFn, {Op.src(0), Op.src(1), Op.src(2)}, "perm"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- 64-bit vector ops ----
  //
  // V_LSHLREV_B64 / V_LSHRREV_B64 / V_ASHRREV_I64 / V_LSHL_ADD_U64 take
  // a 32-bit shift count; AMDGPU masks it to the low 6 bits, so we mask
  // the zext'd count before shifting -- an unmasked count >= 64 makes
  // the LLVM shift poison.
  const uint64_t ShiftCountMask = (1u << 6) - 1;
  if (Sop == CanonicalOp::V_LSHLREV_B64) {
    Value *ShiftAmount = Op.src(0);
    Value *Src = Op.src64(1);
    if (Src->getType() != Ctx.I64Ty) Src = Ctx.B.CreateBitOrPointerCast(Src, Ctx.I64Ty);
    Value *ShiftAmountExt = Ctx.B.CreateZExt(ShiftAmount, Ctx.I64Ty);
    ShiftAmountExt = Ctx.B.CreateAnd(ShiftAmountExt, Ctx.B.getInt64(ShiftCountMask),
                                     "shift_amount_masked");
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateShl(Src, ShiftAmountExt, "shl"));
    Hr.Handled = true;
    return Hr;
  }
  // gfx8+ V_LSHRREV_B64 / V_ASHRREV_I64 -- same operand shape as
  // V_LSHLREV_B64: `dst = src1 >> src0`. Logical right shift fills with
  // zero (lshr) and arithmetic right shift fills with the sign bit (ashr).
  if (Sop == CanonicalOp::V_LSHRREV_B64 || Sop == CanonicalOp::V_ASHRREV_I64) {
    Value *ShiftAmount = Op.src(0);
    Value *Src = Op.src64(1);
    if (Src->getType() != Ctx.I64Ty) Src = Ctx.B.CreateBitOrPointerCast(Src, Ctx.I64Ty);
    Value *ShiftAmountExt = Ctx.B.CreateZExt(ShiftAmount, Ctx.I64Ty);
    ShiftAmountExt = Ctx.B.CreateAnd(ShiftAmountExt, Ctx.B.getInt64(ShiftCountMask),
                                     "shift_amount_masked");
    Value *Res = (Sop == CanonicalOp::V_LSHRREV_B64)
                     ? Ctx.B.CreateLShr(Src, ShiftAmountExt, "lshr")
                     : Ctx.B.CreateAShr(Src, ShiftAmountExt, "ashr");
    Ctx.writeReg64(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_LSHL_ADD_U64) {
    Value *Src0 = Op.src64(0);
    Value *ShiftAmount = Op.src(1);
    Value *Src2 = Op.src64(2);
    if (Src0->getType()->isPointerTy()) Src0 = Ctx.B.CreatePtrToInt(Src0, Ctx.I64Ty);
    if (Src0->getType() != Ctx.I64Ty) Src0 = Ctx.B.CreateBitOrPointerCast(Src0, Ctx.I64Ty);
    if (Src2->getType() != Ctx.I64Ty) Src2 = Ctx.B.CreateBitOrPointerCast(Src2, Ctx.I64Ty);
    Value *ShiftAmountExt = Ctx.B.CreateZExt(ShiftAmount, Ctx.I64Ty);
    ShiftAmountExt = Ctx.B.CreateAnd(ShiftAmountExt, Ctx.B.getInt64(ShiftCountMask),
                                     "shift_amount_masked");
    Value *Shifted = Ctx.B.CreateShl(Src0, ShiftAmountExt);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateAdd(Shifted, Src2, "lshl_add"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_ADD_NC_U64) {
    Value *S0 = Op.src64(0), *S1 = Op.src64(1);
    if (S0->getType() != Ctx.I64Ty) S0 = Ctx.B.CreateBitOrPointerCast(S0, Ctx.I64Ty);
    if (S1->getType() != Ctx.I64Ty) S1 = Ctx.B.CreateBitOrPointerCast(S1, Ctx.I64Ty);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateAdd(S0, S1, "vadd64"));
    Hr.Handled = true;
    return Hr;
  }
  // gfx1250 V_SUB_NC_U64: per-lane i64 `src0 - src1`. The real instruction
  // exposes no carry/borrow result for Hotswap to model. LLVM's
  // VOP2Instructions.td names the opcode `v_sub_nc_u64` and SIInstructions.td
  // lowers the pseudo through `DivergentBinFrag<sub>` with operands preserved
  // in this order.
  if (Sop == CanonicalOp::V_SUB_NC_U64) {
    Value *S0 = Op.src64(0), *S1 = Op.src64(1);
    if (S0->getType() != Ctx.I64Ty)
      S0 = Ctx.B.CreateBitOrPointerCast(S0, Ctx.I64Ty);
    if (S1->getType() != Ctx.I64Ty)
      S1 = Ctx.B.CreateBitOrPointerCast(S1, Ctx.I64Ty);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateSub(S0, S1, "vsub64"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::V_MAX_I64 || Sop == CanonicalOp::V_MAX_U64 ||
      Sop == CanonicalOp::V_MIN_I64 || Sop == CanonicalOp::V_MIN_U64) {
    Value *S0 = Op.src64(0), *S1 = Op.src64(1);
    if (S0->getType() != Ctx.I64Ty)
      S0 = Ctx.B.CreateBitOrPointerCast(S0, Ctx.I64Ty);
    if (S1->getType() != Ctx.I64Ty)
      S1 = Ctx.B.CreateBitOrPointerCast(S1, Ctx.I64Ty);
    Value *Cmp = nullptr;
    switch (Sop) {
    case CanonicalOp::V_MAX_I64:
      Cmp = Ctx.B.CreateICmpSGT(S0, S1);
      break;
    case CanonicalOp::V_MAX_U64:
      Cmp = Ctx.B.CreateICmpUGT(S0, S1);
      break;
    case CanonicalOp::V_MIN_I64:
      Cmp = Ctx.B.CreateICmpSLT(S0, S1);
      break;
    case CanonicalOp::V_MIN_U64:
      Cmp = Ctx.B.CreateICmpULT(S0, S1);
      break;
    default:
      llvm_unreachable("not a 64-bit integer min/max CanonicalOp");
    }
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateSelect(Cmp, S0, S1, "vminmax64"));
    Hr.Handled = true;
    return Hr;
  }
  // gfx1250 V_MUL_U64: VOP2 64-bit unsigned multiply producing the low
  // 64 bits of (s0 * s1). Mirrors the V_ADD_NC_U64 shape.
  if (Sop == CanonicalOp::V_MUL_U64) {
    Value *S0 = Op.src64(0), *S1 = Op.src64(1);
    if (S0->getType() != Ctx.I64Ty) S0 = Ctx.B.CreateBitOrPointerCast(S0, Ctx.I64Ty);
    if (S1->getType() != Ctx.I64Ty) S1 = Ctx.B.CreateBitOrPointerCast(S1, Ctx.I64Ty);
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateMul(S0, S1, "vmul64"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_mad_u64_u32 (2 defs: VDST + SDST, firstSrcIdx=2) ----
  if (Sop == CanonicalOp::V_MAD_U64_U32) {
    Value *A = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty), *B = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty);
    Value *Res = Ctx.B.CreateAdd(Ctx.B.CreateMul(A, B), Op.src64(2), "vmad64");
    Ctx.writeReg64(Op.dst(0), Res);
    Ctx.writeReg64(Op.dst(1), ConstantInt::get(Ctx.I64Ty, 0));
    Hr.Handled = true;
    return Hr;
  }

  // ---- Vector compares (V_CMP / V_CMPX) ----
  // Extracted to handle-valu-vcmp.cpp.
  {
    HandlerResult Sub = handleValuVcmp(Ctx, Di, Op);
    if (Sub.Handled || Sub.Failure.hasFailed())
      return Sub;
  }

  // ---- VOP3P / WMMA / v_fma_mix_f32 / v_cndmask_b32 ----
  // Extracted to handle-valu-vop3p.cpp.
  {
    HandlerResult Sub = handleValuVoP3P(Ctx, Di, Op);
    if (Sub.Handled || Sub.Failure.hasFailed())
      return Sub;
  }

  return Hr;
}

} // namespace COMGR::hotswap
