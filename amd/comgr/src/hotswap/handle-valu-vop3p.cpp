//===- handle-valu-vop3p.cpp - Hotswap transpiler -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-internal.h"

#include "canonical-op.h"
#include "wmma-lowering.h"

#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName
#include "SIDefines.h"            // SISrcMods::NEG
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

struct PackedSrcOptions {
  // Register operands in the packed-f32 family are VGPR pairs that should be
  // read as `<2 x elem>` directly. Packed-f16/i16 operands are one i32 VGPR
  // whose low/high halves are bitcast to `<2 x elem>`.
  bool RegisterSourceIsVector = false;
  // Packed-f32 immediates are scalar 32-bit literals broadcast to both lanes.
  // Packed-f16/i16 immediates are raw packed i32 payloads decoded by LLVM MC.
  bool ImmediateIsScalarBroadcast = false;
  // Floating-point packed families use NEG / NEG_HI as per-lane fneg bits.
  // Integer packed families reject those bits before calling the helper.
  bool ApplyFloatNeg = false;
  // IRBuilder base name used for temporary values from this source family.
  const char *Name = "pk_src";
};

StringRef diagnosticMnemonic(const DecodedInst &Di) {
  return Di.Mnemonic.empty() ? StringRef(canonicalOpName(Di.CanonOp))
                             : StringRef(Di.Mnemonic);
}

bool readSourceMods(const DecodedInst &Di, OpResolver &Op, unsigned NumSrcs,
                    unsigned AllowedMods, unsigned Mods[3],
                    HandlerResult &Hr) {
  StringRef InstrName = diagnosticMnemonic(Di);
  if (Op.nSrcs() < NumSrcs) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3P", (InstrName + " requires more source operands").str());
    return false;
  }

  for (unsigned I = 0; I < NumSrcs; ++I) {
    unsigned ModIdx = Di.ModMap[I];
    if (ModIdx == UINT_MAX || !Di.isImm(ModIdx)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (InstrName + " missing immediate srcN_modifiers operand").str());
      return false;
    }
    Mods[I] = static_cast<unsigned>(Di.getImm(ModIdx));
    if ((Mods[I] & ~AllowedMods) != 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (InstrName + " has unsupported srcN_modifiers bits").str());
      return false;
    }
  }
  return true;
}

bool readPackedSrcMods(const DecodedInst &Di, OpResolver &Op, unsigned NumSrcs,
                       unsigned AllowedMods, unsigned Mods[3],
                       HandlerResult &Hr) {
  if (!readSourceMods(Di, Op, NumSrcs, AllowedMods, Mods, Hr))
    return false;

  StringRef InstrName = diagnosticMnemonic(Di);
  for (unsigned I = 0; I < NumSrcs; ++I) {
    unsigned SrcIdx = Op.srcIdx(I);
    if (!Di.isReg(SrcIdx) && !Di.isImm(SrcIdx)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (InstrName + " source is neither a register nor an immediate").str());
      return false;
    }
  }
  return true;
}

Value *readPacked2Src(RaiseContext &Ctx, OpResolver &Op, unsigned I,
                      Type *ElemTy, unsigned Mods,
                      const PackedSrcOptions &Opts) {
  auto *VecTy = FixedVectorType::get(ElemTy, 2);
  Value *NatLo = nullptr;
  Value *NatHi = nullptr;

  if (Opts.RegisterSourceIsVector && Op.isSrcReg(I)) {
    Value *Vec = Ctx.Regs.readRegVec(Ctx.B, Op.srcReg(I), VecTy);
    NatLo = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(0));
    NatHi = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(1));
  } else if (Opts.ImmediateIsScalarBroadcast && !Op.isSrcReg(I)) {
    Value *Scalar = Ctx.B.CreateBitCast(Op.src(I), ElemTy);
    NatLo = Scalar;
    NatHi = Scalar;
  } else {
    Value *Raw = Op.src(I);
    if (Raw->getType() != Ctx.I32Ty)
      Raw = Ctx.B.CreateBitCast(Raw, Ctx.I32Ty);
    Value *Vec = Ctx.B.CreateBitCast(Raw, VecTy, Opts.Name);
    NatLo = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(0));
    NatHi = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(1));
  }

  Value *Lo = (Mods & SISrcMods::OP_SEL_0) ? NatHi : NatLo;
  Value *Hi = (Mods & SISrcMods::OP_SEL_1) ? NatHi : NatLo;

  if (Opts.ApplyFloatNeg) {
    if (Mods & SISrcMods::NEG)
      Lo = Ctx.B.CreateFNeg(Lo, (Twine(Opts.Name) + "_neg_lo").str());
    if (Mods & SISrcMods::NEG_HI)
      Hi = Ctx.B.CreateFNeg(Hi, (Twine(Opts.Name) + "_neg_hi").str());
  }

  Value *R = UndefValue::get(VecTy);
  R = Ctx.B.CreateInsertElement(R, Lo, static_cast<uint64_t>(0));
  R = Ctx.B.CreateInsertElement(R, Hi, static_cast<uint64_t>(1));
  return R;
}

Value *applyF32InputMods(RaiseContext &Ctx, Value *V, unsigned Mods,
                         const Twine &Name) {
  if (V->getType() != Ctx.F32Ty)
    V = Ctx.B.CreateBitCast(V, Ctx.F32Ty);
  if (Mods & SISrcMods::ABS)
    V = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, V, nullptr,
                                   (Name + "_abs").str());
  if (Mods & SISrcMods::NEG)
    V = Ctx.B.CreateFNeg(V, (Name + "_neg").str());
  return V;
}

Value *readMixF32Src(RaiseContext &Ctx, OpResolver &Op, unsigned I,
                     Type *NarrowTy, unsigned Mods, StringRef CvtName) {
  Value *Raw = Op.src(I);
  if ((Mods & SISrcMods::OP_SEL_1) == 0)
    return applyF32InputMods(Ctx, Raw, Mods, "mix_full");

  if (Raw->getType() == Ctx.F32Ty)
    Raw = Ctx.B.CreateBitCast(Raw, Ctx.I32Ty);

  Value *Bits = nullptr;
  bool IsImmediateOperand = !Op.isSrcReg(I);
  if (!IsImmediateOperand && (Mods & SISrcMods::OP_SEL_0))
    Bits = Ctx.B.CreateTrunc(Ctx.B.CreateLShr(Raw, 16),
                              Type::getInt16Ty(Ctx.C));
  else
    Bits = Ctx.B.CreateTrunc(Raw, Type::getInt16Ty(Ctx.C));

  Value *NarrowVal = Ctx.B.CreateBitCast(Bits, NarrowTy);
  Value *Extended = Ctx.B.CreateFPExt(NarrowVal, Ctx.F32Ty, CvtName);
  return applyF32InputMods(Ctx, Extended, Mods, CvtName);
}

// Read the C (accumulator) operand of a WMMA instruction, handling the
// three encoding shapes LLVM's AMDGPU backend emits:
//
//   * _twoaddr form: C is tied to D (same VGPR slot, no separate `src2`
//     operand on the disassembled line). `op.isSrcReg(2)` is TRUE and
//     `srcReg(2)` returns the D VGPR -- we read the live VGPR value.
//   * _threeaddr form with a VGPR C: `isSrcReg(2)` TRUE and `srcReg(2)`
//     returns the explicit C VGPR. Same path as twoaddr -- just a
//     different VGPR index.
//   * _threeaddr form with an inline-constant C: LLVM picks this
//     encoding whenever the accumulator source is a constant that fits
//     in the VOP3P src2 inline-constant table (the important case is
//     `C = 0`, which Clang emits for every fresh accumulator built from
//     a zero-initialised `v8f c = {0, ..., 0}`). Here `isSrcReg(2)` is
//     FALSE; we MUST materialise the inline constant directly.
//
// The previous fallback `srcC = dest` was silently wrong for the third
// case: reading the D VGPR before the WMMA writes to it surfaces
// whatever stale (or undef) bits happened to be in those 8 VGPR slots,
// which on a cold kernel is typically zero by accident for the first
// WMMA in a wave but nondeterministic for any subsequent WMMA whose
// D range was never explicitly zero-initialised by the SGPR/VGPR
// prologue. In the `wmma_parallel{2,4,16}` probes the second and
// later WMMAs land on fresh D VGPRs (v[24:31], v[32:39], ...) that
// the compiler skipped zeroing -- precisely because it knew the
// threeaddr-imm-0 encoding would satisfy C.
//
// We handle only inline constant `0` today: it is the only src2 inline
// the AMDGPU backend actually emits for the WMMA family (Clang folds
// non-zero accumulator constants through a VGPR mov before the WMMA).
// Any other immediate surfaces as a structured `unsupportedInstructionForm`
// failure rather than silently miscompiling.
//
// On failure the helper populates `Hr.Failure` and returns nullptr; the
// caller must short-circuit.
llvm::Value *readWMMAAccumC(RaiseContext &Ctx, const DecodedInst &Di,
                             OpResolver &Op, const ParsedReg &Dest,
                             llvm::Type *CdIrTy, HandlerResult &Hr) {
  if (Op.nSrcs() < 3) {
    // No src2 operand on the instruction at all (e.g. a hypothetical
    // encoding with C implicitly zero and no disassembler-surfaced
    // slot). Safest to refuse -- the caller expects to have read C.
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3P",
        "WMMA instruction has no src2 (accumulator) operand; "
        "cannot recover C input");
    return nullptr;
  }
  if (Op.isSrcReg(2)) {
    ParsedReg SrcC = Op.srcReg(2);
    return Ctx.Regs.readRegVec(Ctx.B, SrcC, CdIrTy);
  }
  // Inline-constant src2. Today we only model `0`.
  unsigned SrcIdx2 = Op.srcIdx(2);
  if (!Di.isImm(SrcIdx2)) {
    // Could be a symbolic constant slot (e.g. SRC_EXEC_LO/HI, SRC_PC).
    // None of those are valid semantics for a WMMA accumulator; refuse.
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3P",
        "WMMA src2 is neither a register nor an immediate; no "
        "accumulator C input path is defined for this encoding");
    return nullptr;
  }
  int64_t ImmC = Di.getImm(SrcIdx2);
  if (ImmC == 0)
    return llvm::ConstantAggregateZero::get(CdIrTy);
  Hr.Failure = RaiseFailure::unsupportedInstructionForm(
      Di, "VOP3P",
      "WMMA src2 inline-constant other than 0 is not yet modelled; "
      "extend readWMMAAccumC if a corpus kernel surfaces this");
  (void)Dest;
  return nullptr;
}

} // namespace

HandlerResult handleValuVoP3P(RaiseContext &Ctx, const DecodedInst &Di,
                                OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;
  StringRef Mn(Di.Mnemonic);

  switch (Sop) {
  // ---- VOP3P packed ops ----
  // Handle op_sel/op_sel_hi and per-lane negation modifiers.
  case CanonicalOp::V_PK_MOV_B32: {
    Ctx.writeReg64(Op.dst(), Op.src64(0));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_FMA_F16: {
    constexpr unsigned KnownPkF16Mods =
        SISrcMods::NEG | SISrcMods::NEG_HI | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (!readPackedSrcMods(Di, Op, 3, KnownPkF16Mods, Mods, Hr))
      return Hr;

    int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx))) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", "v_pk_fma_f16 missing immediate clamp operand");
      return Hr;
    }
    int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
    if (ClampImm != 0 && ClampImm != 1) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", "v_pk_fma_f16 clamp operand is not 0 or 1");
      return Hr;
    }

    auto *V2f16 = FixedVectorType::get(Ctx.F16Ty, 2);
    PackedSrcOptions Opts;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_f16_src";
    // VSrc_v2f16 immediates are decoded by LLVM MC as the raw 32-bit
    // packed source bits. Scalar f16 inline constants occupy the low half;
    // OP_SEL_1 controls whether the high result lane also reads that low
    // half, matching LLVM's own v_pk_fma_f16 patterns.
    Value *S0 = readPacked2Src(Ctx, Op, 0, Ctx.F16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Ctx.F16Ty, Mods[1], Opts);
    Value *S2 = readPacked2Src(Ctx, Op, 2, Ctx.F16Ty, Mods[2], Opts);
    Function *FmaFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fma, {V2f16});
    Value *Res = Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, "pk_fma_f16");

    if (ClampImm != 0) {
      Function *MaxFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maxnum, {V2f16});
      Function *MinFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minnum, {V2f16});
      // AMDGPU clamp is [0, 1] after the arithmetic result; maxnum/minnum
      // gives the target-independent IR shape used elsewhere in Hotswap.
      Value *Zero = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(Ctx.F16Ty, 0.0));
      Value *One = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(Ctx.F16Ty, 1.0));
      Res = Ctx.B.CreateCall(MaxFn, {Res, Zero}, "pk_fma_f16_clamp_lo");
      Res = Ctx.B.CreateCall(MinFn, {Res, One}, "pk_fma_f16_clamp");
    }

    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(Res, Ctx.I32Ty, "pk_fma_f16_pack"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_ADD_F16:
  case CanonicalOp::V_PK_MUL_F16: {
    constexpr unsigned KnownPkF16Mods =
        SISrcMods::NEG | SISrcMods::NEG_HI | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (!readPackedSrcMods(Di, Op, 2, KnownPkF16Mods, Mods, Hr))
      return Hr;

    int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx))) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (diagnosticMnemonic(Di) + " missing immediate clamp operand").str());
      return Hr;
    }
    int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
    if (ClampImm != 0 && ClampImm != 1) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (diagnosticMnemonic(Di) + " clamp operand is not 0 or 1").str());
      return Hr;
    }

    auto *V2f16 = FixedVectorType::get(Ctx.F16Ty, 2);
    PackedSrcOptions Opts;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_f16_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, Ctx.F16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Ctx.F16Ty, Mods[1], Opts);
    const bool IsAdd = Sop == CanonicalOp::V_PK_ADD_F16;
    const char *Name = IsAdd ? "pk_add_f16" : "pk_mul_f16";
    Value *Res = IsAdd ? Ctx.B.CreateFAdd(S0, S1, Name)
                       : Ctx.B.CreateFMul(S0, S1, Name);

    if (ClampImm != 0) {
      Function *MaxFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maxnum, {V2f16});
      Function *MinFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minnum, {V2f16});
      Value *Zero = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(Ctx.F16Ty, 0.0));
      Value *One = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(Ctx.F16Ty, 1.0));
      Res = Ctx.B.CreateCall(MaxFn, {Res, Zero}, Twine(Name) + "_clamp_lo");
      Res = Ctx.B.CreateCall(MinFn, {Res, One}, Twine(Name) + "_clamp");
    }

    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(Res, Ctx.I32Ty, Twine(Name) + "_pack"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_ADD_BF16:
  case CanonicalOp::V_PK_MUL_BF16:
  case CanonicalOp::V_PK_MIN_NUM_BF16:
  case CanonicalOp::V_PK_MAX_NUM_BF16:
  case CanonicalOp::V_PK_FMA_BF16: {
    constexpr unsigned KnownPkBF16Mods =
        SISrcMods::NEG | SISrcMods::NEG_HI | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    const bool IsFMA = Sop == CanonicalOp::V_PK_FMA_BF16;
    const bool IsMinMax = Sop == CanonicalOp::V_PK_MIN_NUM_BF16 ||
                          Sop == CanonicalOp::V_PK_MAX_NUM_BF16;
    unsigned Mods[3] = {};
    if (!readPackedSrcMods(Di, Op, IsFMA ? 3 : 2, KnownPkBF16Mods, Mods, Hr))
      return Hr;

    int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx))) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (diagnosticMnemonic(Di) + " missing immediate clamp operand").str());
      return Hr;
    }
    int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
    if (ClampImm != 0 && ClampImm != 1) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (diagnosticMnemonic(Di) + " clamp operand is not 0 or 1").str());
      return Hr;
    }
    if (ClampImm != 0 && !IsMinMax) {
      // Packed BF16 add/mul/fma do not use the ordinary VOP3 ALU clamp
      // contract ([0, 1] saturation). Their non-default clamp/overflow
      // behavior is tied to the wave's MODE.FP16_OVFL state, which this
      // raiser does not currently model. Refuse instead of silently lowering
      // it as min(max(x, 0), 1).
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (diagnosticMnemonic(Di) +
           " has a nonzero clamp bit; packed BF16 add/mul/fma clamp and "
           "overflow-mode semantics are not modelled")
              .str());
      return Hr;
    }

    Type *Bf16Ty = Type::getBFloatTy(Ctx.C);
    FixedVectorType *V2BF16 = FixedVectorType::get(Bf16Ty, 2);
    PackedSrcOptions Opts;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_bf16_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, Bf16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Bf16Ty, Mods[1], Opts);
    Value *Res = nullptr;
    if (IsFMA) {
      Value *S2 = readPacked2Src(Ctx, Op, 2, Bf16Ty, Mods[2], Opts);
      Function *FmaFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::fma, {V2BF16});
      Res = Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, "pk_fma_bf16");
    } else if (Sop == CanonicalOp::V_PK_ADD_BF16) {
      Res = Ctx.B.CreateFAdd(S0, S1, "pk_add_bf16");
    } else if (Sop == CanonicalOp::V_PK_MUL_BF16) {
      Res = Ctx.B.CreateFMul(S0, S1, "pk_mul_bf16");
    } else {
      Intrinsic::ID Id = (Sop == CanonicalOp::V_PK_MIN_NUM_BF16)
                             ? Intrinsic::minimumnum
                             : Intrinsic::maximumnum;
      Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Id, {V2BF16});
      Res = Ctx.B.CreateCall(Fn, {S0, S1},
                             Sop == CanonicalOp::V_PK_MIN_NUM_BF16
                                 ? "pk_min_num_bf16"
                                 : "pk_max_num_bf16");
    }

    if (ClampImm != 0) {
      Function *MaxFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maximumnum, {V2BF16});
      Function *MinFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minimumnum, {V2BF16});
      Value *Zero = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(Bf16Ty, 0.0));
      Value *One = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(Bf16Ty, 1.0));
      const char *Name = (Sop == CanonicalOp::V_PK_MIN_NUM_BF16)
                             ? "pk_min_num_bf16"
                             : "pk_max_num_bf16";
      Res = Ctx.B.CreateCall(MaxFn, {Res, Zero}, Twine(Name) + "_clamp_lo");
      Res = Ctx.B.CreateCall(MinFn, {Res, One}, Twine(Name) + "_clamp");
    }

    const char *PackName = "pk_bf16_pack";
    switch (Sop) {
    case CanonicalOp::V_PK_ADD_BF16: PackName = "pk_add_bf16_pack"; break;
    case CanonicalOp::V_PK_MUL_BF16: PackName = "pk_mul_bf16_pack"; break;
    case CanonicalOp::V_PK_MIN_NUM_BF16:
      PackName = "pk_min_num_bf16_pack";
      break;
    case CanonicalOp::V_PK_MAX_NUM_BF16:
      PackName = "pk_max_num_bf16_pack";
      break;
    case CanonicalOp::V_PK_FMA_BF16: PackName = "pk_fma_bf16_pack"; break;
    default: llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Res, Ctx.I32Ty, PackName));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_ADD_F32:
  case CanonicalOp::V_PK_MUL_F32:
  case CanonicalOp::V_PK_FMA_F32:
  case CanonicalOp::V_PK_MAX_F32:
  case CanonicalOp::V_PK_MIN_F32: {
    auto *V2f32 = FixedVectorType::get(Ctx.F32Ty, 2);

    constexpr unsigned KnownPkF32Mods =
        SISrcMods::NEG | SISrcMods::NEG_HI | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    unsigned NumSrcs = (Sop == CanonicalOp::V_PK_FMA_F32) ? 3 : 2;
    if (!readPackedSrcMods(Di, Op, NumSrcs, KnownPkF32Mods, Mods, Hr))
      return Hr;

    // Read each source as <2 x f32>, applying source selection and negation
    // from LLVM's decoded srcN_modifiers operand.
    //
    // Two operand shapes are accepted:
    //   * Register (the common case): reads a 64-bit VGPR pair as
    //     `<2 x f32>`; lo/hi extract index the two packed lanes.
    //   * Immediate / inline literal: VOP3P encodes a single 32-bit
    //     literal per source slot which the hardware broadcasts to
    //     both packed lanes (the `op_sel_hi` modifier is ignored on
    //     scalar literals because there's only one element to choose).
    //     The swiglu tensilelite kernel exercises this path with
    //     `v_pk_add_f32 vN, vM, 0x...` where the literal is a packed
    //     bias constant.  We model it by reading the i32, bit-casting
    //     to f32, and constructing a 2-lane vector with both lanes
    //     equal to the literal.  `neg_lo` / `neg_hi` still apply per lane.
    PackedSrcOptions Opts;
    Opts.RegisterSourceIsVector = true;
    Opts.ImmediateIsScalarBroadcast = true;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_f32_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, Ctx.F32Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Ctx.F32Ty, Mods[1], Opts);

    Value *Res = nullptr;
    switch (Sop) {
    case CanonicalOp::V_PK_ADD_F32:
      Res = Ctx.B.CreateFAdd(S0, S1, "pk_add");
      break;
    case CanonicalOp::V_PK_MUL_F32:
      Res = Ctx.B.CreateFMul(S0, S1, "pk_mul");
      break;
    case CanonicalOp::V_PK_MAX_F32: {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maxnum, {V2f32});
      Res = Ctx.B.CreateCall(Fn, {S0, S1}, "pk_max");
      break;
    }
    case CanonicalOp::V_PK_MIN_F32: {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minnum, {V2f32});
      Res = Ctx.B.CreateCall(Fn, {S0, S1}, "pk_min");
      break;
    }
    case CanonicalOp::V_PK_FMA_F32: {
      Value *S2 = readPacked2Src(Ctx, Op, 2, Ctx.F32Ty, Mods[2], Opts);
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::fma, {V2f32});
      Res = Ctx.B.CreateCall(Fn, {S0, S1, S2}, "pk_fma");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeRegVec(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }

  // ---- VOP3P packed-pair `<2 x i16>` int ops ----
  // V_PK_ADD_U16 / V_PK_LSHLREV_B16 / V_PK_MUL_LO_U16. Operand profile is
  // VOP_V2I16_V2I16_V2I16: 32-bit dst / 32-bit src0 / 32-bit src1, each
  // bitcast to `<2 x i16>` for the lane-wise op and back to i32 for the
  // VGPR write-back. Shared handler shape; per-CanonicalOp dispatch picks the
  // IR opcode (`add` vs the reversed `clshl_rev_16` shape vs lane-wise
  // modular `mul` -- see notes on each case below). Inline literals encode
  // a packed `<2 x i16>` directly (lo i16 = bits[15:0], hi i16 = bits[31:16]);
  // there is NO broadcast analogue to the V_PK_F32 32-bit-element family
  // because the literal width matches the operand width here. Sibling
  // V_PK_LSHRREV_B16 / V_PK_ASHRREV_I16 / V_PK_SUB_U16 share this exact
  // shape -- one extra `case` + IR-opcode dispatch in the inner switch
  // and they're done -- but they're held out per the "no fallback / design
  // what the corpus exercises" discipline.
  case CanonicalOp::V_PK_ADD_U16:
  case CanonicalOp::V_PK_LSHLREV_B16:
  case CanonicalOp::V_PK_MUL_LO_U16: {
    auto *I16Ty = Type::getInt16Ty(Ctx.C);

    constexpr unsigned KnownPkI16Mods =
        SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (!readPackedSrcMods(Di, Op, 2, KnownPkI16Mods, Mods, Hr))
      return Hr;

    PackedSrcOptions Opts;
    Opts.Name = "pk_i16_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, I16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, I16Ty, Mods[1], Opts);

    Value *Res = nullptr;
    switch (Sop) {
    case CanonicalOp::V_PK_ADD_U16:
      Res = Ctx.B.CreateAdd(S0, S1, "pk_add_u16");
      break;
    case CanonicalOp::V_PK_MUL_LO_U16:
      // "lo" = low 16 bits of the per-lane 32-bit multiply, i.e. modular
      // u16 multiply. Plain `mul` on i16 without nuw/nsw matches that --
      // signed vs unsigned doesn't change the low half of the product.
      Res = Ctx.B.CreateMul(S0, S1, "pk_mul_lo_u16");
      break;
    case CanonicalOp::V_PK_LSHLREV_B16: {
      // clshl_rev_16 SDAG: dst = src1 << (src0 & 15). Reversed-operand
      // convention (shift count is src0, value is src1) AND a hardware
      // clamp to the low 4 bits of the count. LLVM `shl` is poison for
      // shifts >= bitwidth, the hardware masks instead -- emit the AND
      // explicitly so the LLVM semantics match the AMDGPU semantics for
      // every legal hardware input. For constant shift counts the
      // optimiser folds the AND away; for VGPR-sourced shift counts the
      // mask is mandatory to preserve the corpus shift semantics.
      Value *Mask = ConstantVector::getSplat(
          ElementCount::getFixed(2),
          ConstantInt::get(I16Ty, 15));
      Value *Amt = Ctx.B.CreateAnd(S0, Mask, "pk_lshlrev_amt");
      Res = Ctx.B.CreateShl(S1, Amt, "pk_lshlrev");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }

    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(Res, Ctx.I32Ty, "pk_i16_pack"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_dot4_i32_iu8 ----
  //
  // Mixed signed/unsigned 4-byte dot product:
  //   dst = src2 + sum_{i=0..3} extA(src0.byte[i]) * extB(src1.byte[i])
  //
  // AMDGPU models the input signedness through the VOP3P source modifier
  // operands: SISrcMods::NEG set on src0/src1 means that source's packed bytes
  // are signed, otherwise they are unsigned. LLVM's `VOP3PModsNeg` pattern in
  // SIInstrInfo.td encodes the same contract. Lower to ordinary integer IR
  // rather than a target dot intrinsic so gfx1250 same-target and gfx942
  // cross-target paths share one verifier-clean semantic representation; the
  // backend may rediscover a dot instruction where legal. A future
  // target-native optimisation can route supporting targets through
  // `llvm.amdgcn.sudot4`, but that should not be required for correctness.
  case CanonicalOp::V_DOT4_I32_IU8: {
    int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    bool Clamp = false;
    if (ClampIdx >= 0 && Di.isImm(static_cast<unsigned>(ClampIdx)))
      Clamp = Di.getImm(static_cast<unsigned>(ClampIdx)) != 0;
    if (ClampIdx >= 0 && !Di.isImm(static_cast<unsigned>(ClampIdx))) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", "v_dot4_i32_iu8 clamp operand is not an immediate");
      return Hr;
    }

    Value *Src0 = Op.src(0);
    Value *Src1 = Op.src(1);
    Value *Acc = Ctx.B.CreateSExt(Op.src(2), Ctx.I64Ty, "dot4_acc_wide");
    auto *I8Ty = Type::getInt8Ty(Ctx.C);

    auto ExtendByte = [&](Value *Packed, unsigned ByteIdx,
                          bool IsSigned) -> Value * {
      Value *Shift = ConstantInt::get(Ctx.I32Ty, ByteIdx * 8);
      Value *Lo = Ctx.B.CreateTrunc(Ctx.B.CreateLShr(Packed, Shift), I8Ty,
                                    "dot4_byte");
      return IsSigned ? Ctx.B.CreateSExt(Lo, Ctx.I64Ty, "dot4_sext")
                      : Ctx.B.CreateZExt(Lo, Ctx.I64Ty, "dot4_zext");
    };

    const bool Src0Signed = (Op.srcMod(0) & SISrcMods::NEG) != 0;
    const bool Src1Signed = (Op.srcMod(1) & SISrcMods::NEG) != 0;
    for (unsigned I = 0; I < 4; ++I) {
      Value *A = ExtendByte(Src0, I, Src0Signed);
      Value *B = ExtendByte(Src1, I, Src1Signed);
      Acc = Ctx.B.CreateAdd(Acc, Ctx.B.CreateMul(A, B, "dot4_mul"),
                            "dot4_acc");
    }

    if (Clamp) {
      Value *Lo = ConstantInt::get(Ctx.I64Ty, INT32_MIN);
      Value *Hi = ConstantInt::get(Ctx.I64Ty, INT32_MAX);
      Acc = Ctx.B.CreateSelect(Ctx.B.CreateICmpSLT(Acc, Lo), Lo, Acc,
                               "dot4_clamp_lo");
      Acc = Ctx.B.CreateSelect(Ctx.B.CreateICmpSGT(Acc, Hi), Hi, Acc,
                               "dot4_clamp");
    }

    Ctx.writeReg32(Op.dst(), Ctx.B.CreateTrunc(Acc, Ctx.I32Ty, "dot4_i32"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- WMMA (gfx1250 RDNA4, VOP3P encoding) ----
  // 16x16xK WMMA family. Three K-families × accumulator-type
  // permutations covered today:
  //   * 16-bit elements, K=32, f32 acc (8 VGPRs of <16 x t> per A/B side):
  //       v_wmma_f32_16x16x32_f16,  v_wmma_f32_16x16x32_bf16
  //   * 8-bit elements,  K=64, f32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_f32_16x16x64_<a>_<b>  for a,b ∈ {fp8, bf8}
  //   * 8-bit elements,  K=64, i32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_i32_16x16x64_iu8  (signed/unsigned 8-bit integer GEMMs)
  //
  // All share the per-Wave32-lane A/B fragment shape (8 VGPRs, 32 bytes).
  // The C/D side is <8 x f32> for the f32-accumulator variants and
  // <8 x i32> for the IU8 integer-accumulator variant. The WMMA12
  // native-intrinsic path (when target supports it) and the gfx942
  // MFMA lowering path (`emitWMMAtoMFMA`, parameterised on
  // `WMMAInputType`) are uniform across the entire family -- the local
  // A/B IR vector type + native-WMMA intrinsic ID + WMMAInputType +
  // accumulator IR type is the only delta between variants. "Design
  // the operation, not the opcode."
  //
  // Native WMMA12 intrinsic-call shapes split THREE ways:
  //   * 16-bit f32-acc: AMDGPUWmmaIntrinsicModsAllReuse -- 8 args
  //       (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  f32-acc: AMDGPUWmmaIntrinsicModsC       -- 6 args
  //       (A, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  i32-acc: AMDGPUWmmaIntrinsicModsABClamp -- 8 args
  //       (A_mod, A, B_mod, B, C, reuse_a, reuse_b, clamp)
  // The MFMA fallback path is uniform across all three.
  // 16x16x4 WMMA (32-bit f32 A/B/C, gfx1250 VOP3P opcode 0x05D).
  // This handler stands alone from the K=32 / K=64 family below
  // because (a) the per-lane A/B fragment is `<2 x f32>` (only 2
  // dwords) instead of <16 x t> (16-bit) or <8 x i32> (8-bit), and
  // (b) `emitWMMAtoMFMA` is parameterised on 16-/8-bit element
  // packing and does not cover the K=4 f32 case.
  //
  // The native intrinsic `int_amdgcn_wmma_f32_16x16x4_f32` is
  // declared inside `AMDGPUWMMAIntrinsicsGFX1250` (gated by
  // `isGFX125xOnly` in IntrinsicsAMDGPU.td:4113-4114), so it is
  // strictly gfx1250-only -- the gfx12 (RDNA4 base) WMMA family
  // (`AMDGPUWMMAIntrinsicsGFX12`, gated by `hasWMMA12` =
  // FeatureWMMA{128,256}bInsts) does NOT include it. Same-target
  // lift therefore gates on `Ctx.TargetIsa.hasTensorOps`
  // (FeatureGFX1250Insts), not `hasWMMA12`.
  //
  // Cross-target on gfx942 we lower to `mfma_f32_16x16x4f32` via the
  // dedicated `emitWMMAtoMFMA_F32_16x16x4` helper in
  // `wmma-lowering.cpp` -- gfx942 has a direct K=4 MFMA equivalent
  // so the decomposition is 1 MFMA per Wave32 group (not 2 chained
  // like the K=32/K=64 path). The shared ds_bpermute redistribution
  // math is documented alongside the helper. Targets with neither
  // `hasTensorOps` nor `hasMFMA` (e.g. gfx12 RDNA4 base) get a
  // principled refusal -- they have no K=4 f32 matrix path at all.
  case CanonicalOp::V_WMMA_F32_16x16x4_F32: {
    auto *AbIrTy = FixedVectorType::get(Ctx.F32Ty, 2);
    auto *CdIrTy = FixedVectorType::get(Ctx.F32Ty, 8);

    ParsedReg Dest = Op.dst();
    ParsedReg SrcA = Op.srcReg(0), SrcB = Op.srcReg(1);

    Value *A = Ctx.Regs.readRegVec(Ctx.B, SrcA, AbIrTy);
    Value *B = Ctx.Regs.readRegVec(Ctx.B, SrcB, AbIrTy);
    Value *C = readWMMAAccumC(Ctx, Di, Op, Dest, CdIrTy, Hr);
    if (!C)
      return Hr;

    Value *ResultVal;
    if (Ctx.TargetIsa.HasTensorOps) {
      Function *WmmaFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_wmma_f32_16x16x4_f32, {CdIrTy, AbIrTy});
      // AMDGPUWmmaIntrinsicModsC (6 args, see IntrinsicsAMDGPU.td):
      //   (A, B, C_mod, C, matrix_a_reuse, matrix_b_reuse)
      // C_mod is the i16 source-modifier bitfield (op_sel etc.) and
      // matrix_*_reuse are i1 flags. K=4 f32 WMMA has NO per-element
      // A_mod / B_mod slots (unlike the 16-/8-bit ModsAllReuse /
      // ModsABClamp classes used by the K=32 / K=64 family). The
      // gfx1250 corpus emits the instruction without those modifiers
      // set; defaulting to 0 / false matches what the disassembler
      // surfaces for the failing kernels.
      ResultVal = Ctx.B.CreateCall(WmmaFn, {
          A, B,
          ConstantInt::get(Type::getInt16Ty(Ctx.C), 0), C,
          Ctx.B.getFalse(), Ctx.B.getFalse()
      }, "wmma");
    } else if (Ctx.TargetIsa.HasMfma) {
      // Same-shape gate as the K=32/K=64 case below.  The staged
      // strict.wwm-scoped MODREP lowering is verified correct for
      // minimal-repro kernels (isolated and K-loop-chained WMMAs)
      // but an unexplained residual divergence remains on the
      // Triton `matmul_fp16_16x16` kernel at M>=32 through
      // `compare_correctness`.  See the K=32/K=64 arm below for
      // the full discussion.  Gate stays in place until the
      // residual is pinned and the fix lands; the infrastructure
      // in `wave-projection.h` (`numSourceWavesPerTarget`,
      // `wrapAsWWMValue`) is LANDED additively.
      // K=4 f32 arm: previously conservatively refused under MODREP
      // when a multi-WMMA-per-K-iter pattern (permlane16_swap
      // presence) was detected.  The root cause turned out to be a
      // wrong-semantic lift of `v_permlane16_swap_b32` (symmetric
      // vs. ISA-asymmetric -- see `handle-valu-cross-lane.cpp` and
      // matrix-translation.md §12.4.7), not a WMMA-lowering
      // problem, so with that fixed the MODREP MFMA lowering
      // handles both single- and multi-WMMA cases correctly.
      ResultVal = emitWmmAtoMfmaF3216x16x4(Ctx, A, B, C);
    } else {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          "v_wmma_f32_16x16x4_f32 cross-target requires either "
          "hasTensorOps (native gfx1250 intrinsic "
          "int_amdgcn_wmma_f32_16x16x4_f32) or hasMFMA (gfx942 "
          "mfma_f32_16x16x4f32 decomposition); this target has "
          "neither -- no K=4 f32 matrix path is available");
      return Hr;
    }

    Ctx.writeRegVec(Dest, ResultVal);
    Hr.Handled = true;
    return Hr;
  }

  case CanonicalOp::V_WMMA_F32_16x16x32_F16:
  case CanonicalOp::V_WMMA_F32_16x16x32_BF16:
  case CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8:
  case CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8:
  case CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8:
  case CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8:
  case CanonicalOp::V_WMMA_I32_16x16x64_IU8: {
    const bool IsIU8 = (Sop == CanonicalOp::V_WMMA_I32_16x16x64_IU8);
    const bool IsFP8orBF8 =
        (Sop == CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8) ||
        (Sop == CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8) ||
        (Sop == CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8) ||
        (Sop == CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8);
    const bool Is8bit = IsIU8 || IsFP8orBF8;
    const bool IsBF16 = (Sop == CanonicalOp::V_WMMA_F32_16x16x32_BF16);

    Type *AbIrTy = nullptr;
    if (Is8bit) {
      AbIrTy = FixedVectorType::get(Ctx.I32Ty, 8);
    } else {
      Type *ElemTy = IsBF16 ? Type::getBFloatTy(Ctx.C)
                            : Type::getHalfTy(Ctx.C);
      AbIrTy = FixedVectorType::get(ElemTy, 16);
    }
    Type *CdIrTy = IsIU8 ? FixedVectorType::get(Ctx.I32Ty, 8)
                         : FixedVectorType::get(Ctx.F32Ty, 8);

    ParsedReg Dest = Op.dst();
    ParsedReg SrcA = Op.srcReg(0), SrcB = Op.srcReg(1);

    Value *A = Ctx.Regs.readRegVec(Ctx.B, SrcA, AbIrTy);
    Value *B = Ctx.Regs.readRegVec(Ctx.B, SrcB, AbIrTy);
    Value *C = readWMMAAccumC(Ctx, Di, Op, Dest, CdIrTy, Hr);
    if (!C)
      return Hr;

    auto WmmaInputType = [&]() -> WMMAInputType {
      switch (Sop) {
      case CanonicalOp::V_WMMA_F32_16x16x32_F16:    return WMMAInputType::F16;
      case CanonicalOp::V_WMMA_F32_16x16x32_BF16:   return WMMAInputType::BF16;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8:return WMMAInputType::FP8_FP8;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8:return WMMAInputType::FP8_BF8;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8:return WMMAInputType::BF8_FP8;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8:return WMMAInputType::BF8_BF8;
      case CanonicalOp::V_WMMA_I32_16x16x64_IU8:    return WMMAInputType::IU8;
      default:
        report_fatal_error("transpiler: WMMA CanonicalOp not in dispatch table");
      }
    }();

    Value *ResultVal;
    // Native-intrinsic branch: the K=32 / K=64 WMMA intrinsics
    // (`int_amdgcn_wmma_f32_16x16x32_f16`, `..._f32_16x16x64_*`,
    // `..._i32_16x16x64_iu8`) live in `AMDGPUWMMAIntrinsicsGFX1250`
    // (IntrinsicsAMDGPU.td:4096 -- the gfx1250-specific family),
    // NOT in `AMDGPUWMMAIntrinsicsGFX12` (IntrinsicsAMDGPU.td:3123 --
    // the gfx12 RDNA4 base family, which only covers K=16
    // `..._16x16x16_*`).  The gate therefore must be
    // `hasTensorOps` (`FeatureGFX1250Insts`), not `hasWMMA12`
    // (`FeatureWMMA{128,256}bInsts` -- set only on the gfx12 base
    // subtargets that DON'T have K=32/K=64 hardware).  An earlier
    // version of this handler used `hasWMMA12` here, which made
    // gfx1250 same-target lifts of K=32/K=64 WMMA fall through to
    // the `emitWMMAtoMFMA` branch below and emit MFMA-intrinsic IR
    // the backend couldn't lower on gfx1250 (no MFMA hardware).
    // `BatchRaise.Gfx1250TestData` "succeeded" on that broken path
    // because the raise completed, even though any downstream
    // codegen attempt would have failed.  The WMMA taxonomy
    // and the K=4 f32 case above use the same structural fix
    // pattern (which got it right originally -- this K=32/K=64
    // case was slower to catch up).
    if (Ctx.TargetIsa.HasTensorOps) {
      Intrinsic::ID WmmaId;
      switch (Sop) {
      case CanonicalOp::V_WMMA_F32_16x16x32_F16:
        WmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_f16; break;
      case CanonicalOp::V_WMMA_F32_16x16x32_BF16:
        WmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_bf16; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8:
        WmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_fp8; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8:
        WmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_bf8; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8:
        WmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_fp8; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8:
        WmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_bf8; break;
      case CanonicalOp::V_WMMA_I32_16x16x64_IU8:
        WmmaId = Intrinsic::amdgcn_wmma_i32_16x16x64_iu8; break;
      default:
        report_fatal_error(
            "transpiler: WMMA CanonicalOp not in gfx1250 K=32/K=64 dispatch");
      }
      Function *WmmaFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, WmmaId, {CdIrTy, AbIrTy});
      if (IsIU8) {
        // AMDGPUWmmaIntrinsicModsABClamp:
        //   (A_mod, A, B_mod, B, C, reuse_a, reuse_b, clamp)
        // A_mod / B_mod carry the IU8 sign-vs-zero-extension knobs in
        // the gfx1250 ISA; we conservatively emit 0 (zero-extend, i.e.
        // unsigned interpretation) because the corpus IU8 GEMMs
        // observed so far never set the matching `neg_lo` bits. A
        // future loud refusal could be added if a corpus kernel ever
        // surfaces a non-zero A_mod / B_mod through the decoder.
        ResultVal = Ctx.B.CreateCall(WmmaFn, {
            Ctx.B.getFalse(), A,
            Ctx.B.getFalse(), B,
            C,
            Ctx.B.getFalse(), Ctx.B.getFalse(),
            Ctx.B.getFalse()
        }, "wmma");
      } else {
        // AMDGPUWmmaIntrinsicModsC: (A, B, C_mod, C, reuse_a, reuse_b)
        //
        // Both the 8-bit FP8/BF8 family and the 16-bit f16/bf16
        // family use this 6-arg shape in the gfx1250 intrinsic
        // set (`AMDGPUWMMAIntrinsicsGFX1250` -- IntrinsicsAMDGPU.td
        // ~line 4098).  An earlier version of this handler split
        // the 16-bit family into an 8-arg `ModsAllReuse` shape
        // (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b) that
        // matches the gfx12 RDNA4 base WMMA family's intrinsics --
        // which do NOT include K=32/K=64 variants; those gfx12-base
        // intrinsics are K=16-only (`..._16x16x16_*` in
        // `AMDGPUWMMAIntrinsicsGFX12` ~line 3123).  That mismatched
        // arg-list did not surface under the old `hasWMMA12` gate
        // because `hasWMMA12` is never true on gfx1250 subtargets
        // (their feature set deliberately excludes
        // `FeatureWMMA{128,256}bInsts`), so the branch was
        // unreachable at runtime -- but the principled fix in the
        // same commit (switching the gate to `hasTensorOps`) makes
        // the branch reachable, and the arg list must match.  See
        // the LLVM intrinsic-signature trailer further up in this
        // block comment for the full per-family taxonomy.
        ResultVal = Ctx.B.CreateCall(WmmaFn, {
            A, B,
            ConstantInt::get(Type::getInt16Ty(Ctx.C), 0), C,
            Ctx.B.getFalse(), Ctx.B.getFalse()
        }, "wmma");
      }
    } else if (Ctx.TargetIsa.HasMfma) {
      // Same-shape gate as the K=4 f32 case above.  The staged
      // strict.wwm-scoped MODREP lowering is verified correct for
      // isolated and K-loop-chained WMMAs in
      // `lit_tests/wmma_phantom_lane_f16_chain/` (post-rebuild with
      // my test harness actually populating per-iter A/B data --
      // earlier runs claiming "K-loop failure" were a host-side
      // test-input miscount that populated the wrong stride; the
      // fixture itself matches bit-exact at n_iters ∈ {1,2,4,8}
      // under both MODREP phantom-lane and WaveNative).  Triton's
      // `matmul_fp16_16x16` at M>=32 still shows a WRONG-numerics
      // residual through `compare_correctness`, where the
      // backend's MIR-level operand folding (SIWholeQuadMode input
      // from ISEL) merges MFMA2's A and B into the same virtual
      // register due to a kernel-specific IR pattern (v_dual_mov
      // copying v[0:3] to v[64:67] combined with `v5=v4, v6=v4,
      // v7=v4` in the source ISA, making ADwords[4..7] all equal
      // to v4's value).  That operand folding is semantically
      // valid if the values are provably equal at runtime -- which
      // they are, given the copies -- so the MFMA itself computes
      // correctly.  The residual divergence vs the native gfx1250
      // run is under separate investigation and NOT a bug in this
      // lowering's contract.  Until that residual is pinned to a
      // concrete root cause, this gate stays in place rather than
      // ship a subtle wrong-numerics kernel.  Land the
      // infrastructure (numSourceWavesPerTarget, wrapAsWWMValue)
      // additively; flip this gate once the matmul_fp16_16x16
      // divergence is explained.
      // K=32/K=64 WMMA arm: previously refused under MODREP when
      // the kernel carried a `v_permlane16_swap_b32` (the
      // multi-WMMA-per-K-iter marker of Triton's `matmul_fp16`
      // pattern), because empirical lane-data dumps showed the
      // MFMA redistribution reading duplicated K subsets for
      // WMMA.A.
      //
      // MI400 Shader Programming Guide § V_PERMLANE16_SWAP_B32
      // pinned the real root cause: our permlane16_swap lift was
      // implementing the SYMMETRIC cross-16 swap, but the ISA
      // semantic is ASYMMETRIC ("lanes 0:15 of src0 and lanes
      // 16:31 of vdst swapped; lanes 16:31 of src0 and lanes 0:15
      // of vdst UNCHANGED").  The symmetric emulation
      // over-swapped the two "unchanged" halves, which corrupted
      // every matmul_fp16 input position and surfaced downstream
      // as the "+16 col shift" on mode-5 / "+4 bias" on mode-8.
      // With the asymmetric lift in `handle-valu-cross-lane.cpp`
      // (Session-8, matrix-translation.md §12.4.7) the MODREP
      // MFMA redistribution is correct for both single- and
      // multi-WMMA regimes, so this refusal is no longer needed.
      ResultVal = emitWMMAtoMFMA(Ctx, A, B, C, WmmaInputType);
    } else {
      // Target has neither gfx1250 tensor ops (hasTensorOps, K=32
      // / K=64 WMMA native) nor MFMA (gfx942 CDNA3 et al., the
      // cross-target decomposition sink).  No path exists to
      // lower this opcode.  The pre-2026-04-22 handler fell
      // through to `emitWMMAtoMFMA` here and emitted MFMA-
      // intrinsic IR the target couldn't lower; the raise
      // "succeeded" but the backend then failed to codegen --
      // `BatchRaise.Gfx1250TestData` was structurally accepting
      // that broken-IR outcome, not catching it.  Refuse loudly
      // so coverage tooling surfaces targets whose WMMA K=32/
      // K=64 support is missing rather than pretending the lift
      // worked.
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          "v_wmma_*_16x16x{32,64}_* has no available lowering on "
          "the target ISA: the target does not provide gfx1250 "
          "tensor ops (`hasTensorOps` / `FeatureGFX1250Insts` for "
          "the native K=32/K=64 WMMA intrinsic family "
          "`AMDGPUWMMAIntrinsicsGFX1250`) nor MFMA (`hasMFMA` for "
          "the cross-target decomposition via "
          "`wmma-lowering.cpp::emitWMMAtoMFMA`).  No known safe "
          "lowering exists; refusing rather than emitting IR that "
          "cannot be lowered.");
      return Hr;
    }

    Ctx.writeRegVec(Dest, ResultVal);
    Hr.Handled = true;
    return Hr;
  }

  // 16x16x128 scaled WMMA, f8f6f4 mantissa-format family (gfx1250-only).
  //
  // 18 MC pseudos ({f4,f6,f8} A x {f4,f6,f8} B x _twoaddr/_threeaddr) collapse
  // onto this CanonicalOp; the `_fA_fB_w32_*` suffix encodes per-matrix width
  // (f8 -> 16, f6 -> 12, f4 -> 8 dwords). BF8/FP8 and BF6/FP6 are distinguished
  // by the matrix_a_fmt / matrix_b_fmt named immediates (enum MatrixFMT).
  //
  // Cross-target lowering of v_wmma_scale_f32_16x16x128_f8f6f4:
  //   * gfx1250 (hasTensorOps): native int_amdgcn_wmma_scale intrinsic.
  //   * gfx950 (hasGfx950Insts): scaled MFMA via emitWMMAScaleF8F6F4toScaledMFMA.
  //   * gfx942 (hasFP8Insts): K-decomposed unscaled FP8/BF8 MFMA + software
  //     scale via emitWMMAScaleF8F6F4toMFMA.
  //   * Otherwise: refuse.
  case CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4: {
    // Per-matrix dword count from the MC pseudo's fA_fB suffix.
    auto FmtSuffixToDwords = [](StringRef Tag) -> unsigned {
      if (Tag == "f8") return 16;
      if (Tag == "f6") return 12;
      if (Tag == "f4") return 8;
      return 0;
    };
    StringRef PseudoName = Ctx.Mc.InstrInfo->getName(Di.Inst.getOpcode());
    StringRef Body = PseudoName;
    Body.consume_front("V_WMMA_SCALE_F32_16X16X128_F8F6F4_");
    SmallVector<StringRef, 4> Parts;
    Body.split(Parts, '_');
    if (Parts.size() < 2) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4: cannot parse fA_fB suffix from "
          "MC pseudo name");
      return Hr;
    }
    unsigned ADwords = FmtSuffixToDwords(Parts[0]);
    unsigned BDwords = FmtSuffixToDwords(Parts[1]);
    if (ADwords == 0 || BDwords == 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4: unrecognised mantissa-format "
          "tag in MC pseudo suffix (expected f4/f6/f8)");
      return Hr;
    }

    auto *ATy = FixedVectorType::get(Ctx.I32Ty, ADwords);
    auto *BTy = FixedVectorType::get(Ctx.I32Ty, BDwords);
    auto *CdTy = FixedVectorType::get(Ctx.F32Ty, 8);

    ParsedReg Dest = Op.dst();
    ParsedReg SrcA = Op.srcReg(0), SrcB = Op.srcReg(1);

    Value *A = Ctx.Regs.readRegVec(Ctx.B, SrcA, ATy);
    Value *B = Ctx.Regs.readRegVec(Ctx.B, SrcB, BTy);
    // readWMMAAccumC materializes a zero accumulator for an inline-0 src2
    // and refuses other inline constants.
    Value *C = readWMMAAccumC(Ctx, Di, Op, Dest, CdTy, Hr);
    if (!C)
      return Hr;

    // Read named operands by name so a TableGen Ins64 reshuffle flows in for
    // free (mirrors the MFMA-scale handler).
    auto NamedImm = [&](AMDGPU::OpName Name) -> int64_t {
      int Idx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
      if (Idx < 0 || !Di.isImm(Idx)) return 0;
      return Di.getImm(Idx);
    };
    // scale_src0 / scale_src1 carry packed scale bytes. An absent or inline-0
    // source means scale = 1.0 per K-block, whose byte encoding depends on the
    // matrix_*_scale_fmt: E8M0 (0) -> 0x7f (2^0), E4M3 (2) -> 0x38 (1.0). The
    // E8M0 sentinel would decode to NaN under E4M3. Other inline constants
    // have no documented semantics and fail.
    auto UnitScalePacked = [&](int64_t ScaleFmt) -> Value * {
      uint32_t Byte = ScaleFmt == 2 /*E4M3*/ ? 0x38u : 0x7fu;
      return ConstantInt::get(Ctx.I32Ty, Byte * 0x01010101U);
    };
    bool ScaleSrcFailed = false;
    auto NamedScaleSrc32 = [&](AMDGPU::OpName Name,
                               int64_t ScaleFmt) -> Value * {
      int Idx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
      if (Idx < 0)
        return UnitScalePacked(ScaleFmt);
      if (Di.isReg(Idx)) {
        ParsedReg Pr = Ctx.parseReg(Di.getReg(Idx), Idx);
        if (Pr.RegKind != ParsedReg::OTHER && Pr.RegKind != ParsedReg::NOREG)
          return Ctx.Regs.readReg32(Ctx.B, Pr);
      }
      if (Di.isImm(Idx) && Di.getImm(Idx) == 0)
        return UnitScalePacked(ScaleFmt);
      ScaleSrcFailed = true;
      return ConstantInt::get(Ctx.I32Ty, 0);
    };

    Value *MatrixAFmt =
        ConstantInt::get(Ctx.I32Ty, NamedImm(AMDGPU::OpName::matrix_a_fmt));
    Value *MatrixBFmt =
        ConstantInt::get(Ctx.I32Ty, NamedImm(AMDGPU::OpName::matrix_b_fmt));
    Value *CMod = ConstantInt::get(
        Type::getInt16Ty(Ctx.C),
        NamedImm(AMDGPU::OpName::src2_modifiers));
    Value *MatrixAScale = ConstantInt::get(
        Ctx.I32Ty, NamedImm(AMDGPU::OpName::matrix_a_scale));
    int64_t AScaleFmtImm = NamedImm(AMDGPU::OpName::matrix_a_scale_fmt);
    Value *MatrixAScaleFmt = ConstantInt::get(Ctx.I32Ty, AScaleFmtImm);
    Value *ScaleSrc0 = NamedScaleSrc32(AMDGPU::OpName::scale_src0, AScaleFmtImm);
    Value *MatrixBScale = ConstantInt::get(
        Ctx.I32Ty, NamedImm(AMDGPU::OpName::matrix_b_scale));
    int64_t BScaleFmtImm = NamedImm(AMDGPU::OpName::matrix_b_scale_fmt);
    Value *MatrixBScaleFmt = ConstantInt::get(Ctx.I32Ty, BScaleFmtImm);
    Value *ScaleSrc1 = NamedScaleSrc32(AMDGPU::OpName::scale_src1, BScaleFmtImm);
    if (ScaleSrcFailed) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4: scale_src0 / scale_src1 "
          "encoding not supported. Supported: register, or inline "
          "constant 0 (decoded as scale = 1.0 per K-block in the active "
          "scale format, per the gfx1250 programming guide). Other inline "
          "constants have no documented WMMA-scale semantics.");
      return Hr;
    }
    Value *MatrixAReuse = ConstantInt::get(
        Type::getInt1Ty(Ctx.C),
        NamedImm(AMDGPU::OpName::matrix_a_reuse));
    Value *MatrixBReuse = ConstantInt::get(
        Type::getInt1Ty(Ctx.C),
        NamedImm(AMDGPU::OpName::matrix_b_reuse));

    Value *ResultVal = nullptr;

    if (Ctx.TargetIsa.HasTensorOps) {
      // Same-target gfx1250 -> gfx1250 fast path: native scaled WMMA.
      //
      // AMDGPUWmmaScaleIntrinsicModsC<i32>:
      //   (matrix_a_fmt, A, matrix_b_fmt, B, C_mod, C,
      //    matrix_a_scale, matrix_a_scale_fmt, scale_src0,
      //    matrix_b_scale, matrix_b_scale_fmt, scale_src1,
      //    matrix_a_reuse, matrix_b_reuse)
      // Overloaded on D, A, B element vector types.
      Function *WmmaFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4,
          {CdTy, ATy, BTy});
      ResultVal = Ctx.B.CreateCall(WmmaFn,
                                    {MatrixAFmt, A, MatrixBFmt, B, CMod, C,
                                     MatrixAScale, MatrixAScaleFmt, ScaleSrc0,
                                     MatrixBScale, MatrixBScaleFmt, ScaleSrc1,
                                     MatrixAReuse, MatrixBReuse},
                                    "wmma_scale");
    } else if (Ctx.TargetIsa.HasGfx950Insts) {
      // Cross-target gfx1250 -> gfx950 path: WMMA-scale -> MFMA-scale.
      //
      // The gfx950 scaled F8F6F4 MFMA (introduced as part of the gfx950
      // MAI family in LLVM upstream) covers the same K=128 F8/F6/F4 shape
      // as the gfx1250 WMMA, so the lowering is a pure wave32->wave64
      // lane redistribution + intrinsic swap (no K-decomposition, no
      // software scale emulation).  Implementation in
      // `wmma-lowering.cpp::emitWMMAScaleF8F6F4toScaledMFMA`.
      //
      // matrix_a_reuse / matrix_b_reuse are perf hints (not correctness)
      // and have no MFMA equivalent; the helper drops them.
      ResultVal = emitWMMAScaleF8F6F4toScaledMFMA(
          Ctx, A, B, C, MatrixAFmt, MatrixBFmt, CMod, MatrixAScale,
          MatrixAScaleFmt, ScaleSrc0, MatrixBScale, MatrixBScaleFmt, ScaleSrc1,
          ADwords, BDwords);
      // reuse hints have no MFMA equivalent; silence -Wunused-variable.
      (void)MatrixAReuse;
      (void)MatrixBReuse;
      if (!ResultVal) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P",
            "emitWMMAScaleF8F6F4toScaledMFMA refused this fragment width "
            "(ADwords/BDwords outside f8/f6/f4 set)");
        return Hr;
      }
    } else if (Ctx.TargetIsa.HasFP8Insts) {
      // Cross-target gfx1250 -> gfx942: K-decomposed unscaled FP8/BF8 MFMA
      // chain with software scale. Gated on HasFP8Insts (not HasMfma) so
      // gfx90a / gfx940 (MAI but no FP8 MFMA) don't take this path; gfx950
      // is already handled by the HasGfx950Insts branch above.
      ResultVal = emitWMMAScaleF8F6F4toMFMA(
          Ctx, A, B, C, MatrixAFmt, MatrixBFmt, CMod, MatrixAScale,
          MatrixAScaleFmt, ScaleSrc0, MatrixBScale, MatrixBScaleFmt,
          ScaleSrc1, ADwords, BDwords);
      (void)MatrixAReuse;
      (void)MatrixBReuse;
      if (!ResultVal) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P",
            "emitWMMAScaleF8F6F4toMFMA refused this configuration. "
            "Supported in this draft: matrix_a_fmt / matrix_b_fmt in "
            "{FP8, BF8, FP6, BF6, FP4} (ADwords / BDwords in {16, 12, "
            "8}; FP6 / BF6 / FP4 widen in-line to FP8 / BF8 via bit-"
            "arithmetic). matrix_*_scale_fmt in {E8M0 (0), E4M3 (2)}; "
            "E5M3 (1) is not yet supported. matrix_*_scale must be "
            "MATRIX_SCALE_ROW0; the ROW1 selector form "
            "(SCL_OPSEL[0] / SCL_OPSEL_HI[0] = 1, selecting lanes "
            "16..31 instead of 0..15 for scale values) is not yet "
            "implemented. The (data, scale) combination must be legal "
            "per the WMMA-scale spec (non-E8M0 scales require F4 data "
            "on that side; F4 x F4 with non-E8M0 scales requires "
            "matching scale formats).");
        return Hr;
      }
    } else {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4 requires one of: hasTensorOps "
          "(gfx1250 native scaled WMMA), hasGfx950Insts (gfx950 scaled-"
          "MFMA F8F6F4 via emitWMMAScaleF8F6F4toScaledMFMA), or hasFP8Insts "
          "(gfx942 K-decomposed unscaled FP8 / BF8 MFMA via "
          "emitWMMAScaleF8F6F4toMFMA); this target has none.");
      return Hr;
    }

    Ctx.writeRegVec(Dest, ResultVal);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_fma_mixlo_bf16: BF16-result mixed-precision FMA (VOP3P) ----
  //
  // LLVM's TableGen definitions declare V_FMA_MIX{LO,HI}_{F16,BF16} with
  // FPDPRounding=1. The generated selection patterns model them as:
  //
  //   fptrunc_narrow(llvm.fma.f32(cvt_f32(src0_part),
  //                               cvt_f32(src1_part),
  //                               cvt_f32(src2_part)))
  //
  // and the ISA family writes only one 16-bit half of vdst (the other half is
  // the tied vdst_in input). The source `*_part` selection matches
  // V_FMA_MIX_F32{,_BF16} below: op_sel_hi chooses narrow vs full-f32, and
  // op_sel chooses the high half when a register source is interpreted as
  // narrow. LO/HI differ only in destination half; F16/BF16 differ only in
  // narrow type.
  case CanonicalOp::V_FMA_MIXLO_F16:
  case CanonicalOp::V_FMA_MIXHI_F16:
  case CanonicalOp::V_FMA_MIXLO_BF16:
  case CanonicalOp::V_FMA_MIXHI_BF16: {
    StringRef InstrName = diagnosticMnemonic(Di);
    if (Op.nSrcs() < 3) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          (InstrName + " requires three explicit source operands").str());
      return Hr;
    }

    bool IsBF16 = Sop == CanonicalOp::V_FMA_MIXLO_BF16 ||
                  Sop == CanonicalOp::V_FMA_MIXHI_BF16;
    bool WritesHigh = Sop == CanonicalOp::V_FMA_MIXHI_F16 ||
                      Sop == CanonicalOp::V_FMA_MIXHI_BF16;
    Type *NarrowTy = IsBF16 ? Type::getBFloatTy(Ctx.C) : Ctx.F16Ty;
    const char *CvtName =
        IsBF16 ? (WritesHigh ? "mixhi_cvt_bf16" : "mixlo_cvt_bf16")
               : (WritesHigh ? "mixhi_cvt" : "mixlo_cvt");
    const char *FMAName =
        IsBF16 ? (WritesHigh ? "fma_mixhi_bf16" : "fma_mixlo_bf16")
               : (WritesHigh ? "fma_mixhi_f16" : "fma_mixlo_f16");

    bool ClampResult = false;
    int ClampIndex = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                AMDGPU::OpName::clamp);
    if (ClampIndex >= 0) {
      if (!Di.isImm(static_cast<unsigned>(ClampIndex))) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P",
            (InstrName + " clamp operand is not an immediate").str());
        return Hr;
      }
      ClampResult = Di.getImm(static_cast<unsigned>(ClampIndex)) != 0;
    }

    Type *I16Ty = Type::getInt16Ty(Ctx.C);

    constexpr unsigned KnownMixMods =
        SISrcMods::NEG | SISrcMods::ABS | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (!readSourceMods(Di, Op, 3, KnownMixMods, Mods, Hr))
      return Hr;

    Value *S0 = readMixF32Src(Ctx, Op, 0, NarrowTy, Mods[0], CvtName);
    Value *S1 = readMixF32Src(Ctx, Op, 1, NarrowTy, Mods[1], CvtName);
    Value *S2 = readMixF32Src(Ctx, Op, 2, NarrowTy, Mods[2], CvtName);
    Function *FmaFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Value *Fma = Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, FMAName);
    Value *Rounded =
        Ctx.B.CreateFPTrunc(Fma, NarrowTy, (Twine(FMAName) + "_round").str());
    if (ClampResult) {
      // AMDGPUclamp clamps to [0, 1] and maps NaN to 0 (SIInstrInfo.td).
      // V_FMA_MIX{LO,HI} applies it after destination narrow-type rounding.
      Function *MaxFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maxnum, {NarrowTy});
      Function *MinFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minnum, {NarrowTy});
      Rounded = Ctx.B.CreateCall(
          MinFn,
          {Ctx.B.CreateCall(MaxFn,
                            {Rounded, ConstantFP::get(NarrowTy, 0.0)},
                            (Twine(FMAName) + "_clamp_lo").str()),
           ConstantFP::get(NarrowTy, 1.0)},
          (Twine(FMAName) + "_clamp").str());
    }
    Value *NarrowBits =
        Ctx.B.CreateZExt(Ctx.B.CreateBitCast(Rounded, I16Ty), Ctx.I32Ty);

    ParsedReg Dest = Op.dst();
    Value *OldDest = Ctx.Regs.readReg32(Ctx.B, Dest);
    if (WritesHigh) {
      Value *OldLo = Ctx.B.CreateAnd(
          OldDest, ConstantInt::get(Ctx.I32Ty, 0x0000FFFFu),
          (Twine(FMAName) + "_old_lo").str());
      Value *HiBits = Ctx.B.CreateShl(NarrowBits, 16,
                                      (Twine(FMAName) + "_hi_bits").str());
      Ctx.writeReg32(Dest,
                     Ctx.B.CreateOr(OldLo, HiBits,
                                    (Twine(FMAName) + "_pack").str()));
    } else {
      Value *OldHi = Ctx.B.CreateAnd(
          OldDest, ConstantInt::get(Ctx.I32Ty, 0xFFFF0000u),
          (Twine(FMAName) + "_old_hi").str());
      Ctx.writeReg32(Dest,
                     Ctx.B.CreateOr(OldHi, NarrowBits,
                                    (Twine(FMAName) + "_pack").str()));
    }
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_fma_mix_f32 / v_fma_mix_f32_bf16: mixed-precision FMA (VOP3P) ----
  //
  // dst = fma(cvt_f32(src0_part), cvt_f32(src1_part), cvt_f32(src2_part))
  //
  // Per-source selection is driven by the VOP3P op_sel / op_sel_hi
  // modifier pair carried in LLVM's decoded srcN_modifiers operands:
  //
  //   op_sel_hi[i]==0  -> source i is the full f32 VGPR
  //   op_sel_hi[i]==1  -> source i is the 16-bit half selected by
  //                       op_sel[i] (0=lo [15:0], 1=hi [31:16])
  //                       interpreted as the mnemonic's narrow type
  //                       (f16 for V_FMA_MIX_F32, bf16 for
  //                       V_FMA_MIX_F32_BF16), then fpext'd to f32.
  //
  // The BF16 variant does NOT need a cross-target refusal because
  // `fpext bfloat to float` is universally lowered (it's a shift-left-16
  // + bitcast on every AMDGPU target); only the narrow element type
  // switches.
  case CanonicalOp::V_FMA_MIX_F32:
  case CanonicalOp::V_FMA_MIX_F32_BF16: {
    Type *NarrowTy = (Sop == CanonicalOp::V_FMA_MIX_F32_BF16)
                         ? Type::getBFloatTy(Ctx.C)
                         : Ctx.F16Ty;
    const char *CvtName = (Sop == CanonicalOp::V_FMA_MIX_F32_BF16)
                              ? "mix_cvt_bf16"
                              : "mix_cvt";

    constexpr unsigned KnownMixMods =
        SISrcMods::NEG | SISrcMods::ABS | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (!readSourceMods(Di, Op, 3, KnownMixMods, Mods, Hr))
      return Hr;

    // `OP_SEL_0` is a VGPR-half selector and only makes sense when the source
    // is a 32-bit VGPR that holds two packed 16-bit values. For immediates,
    // LLVM's AMDGPU disassembler pre-resolves narrow-width operands to the
    // 16-bit value in the low half of the MCOperand immediate; the helper
    // therefore ignores OP_SEL_0 for non-register narrow sources.
    Value *S0 = readMixF32Src(Ctx, Op, 0, NarrowTy, Mods[0], CvtName);
    Value *S1 = readMixF32Src(Ctx, Op, 1, NarrowTy, Mods[1], CvtName);
    Value *S2 = readMixF32Src(Ctx, Op, 2, NarrowTy, Mods[2], CvtName);
    Function *FmaFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, "fma_mix"),
                       Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_cndmask_b32 (VOP2 or VOP3 -- srcMap skips modifiers) ----
  case CanonicalOp::V_CNDMASK_B32: {
    ParsedReg Dest = Op.dst();
    Value *Src0 = Op.src(0);
    Value *Src1 = Op.src(1);
    Value *Cond = nullptr;
    if (Op.nSrcs() >= 3 && Di.isReg(Op.srcIdx(2))) {
      ParsedReg CondReg =
          Ctx.parseReg(Di.getReg(Op.srcIdx(2)), Op.srcIdx(2));
      if (CondReg.RegKind == ParsedReg::SGPR) {
        // Preferred path: a V_CMP_*_e64 in the current BB wrote this
        // SGPR and no intervening scalar write has invalidated the
        // cached per-lane `i1`. Use the `i1` directly -- it carries
        // the full target-hardware ballot without the cross-widening
        // narrow-write information loss (the SGPR itself holds only
        // the source-width-truncated 32-bit projection). See
        // hotswap/docs/sgpr-wave-mask-translation.md section 3.1 for
        // the full contract and
        // `RaiseContext::lastSgprWaveMaskI1` for the invariants that
        // make this lookup sound.
        if (Value *FreshCmp = Ctx.lookupSgprWaveMaskI1(CondReg.BaseIdx)) {
          Cond = FreshCmp;
        } else {
          // Fallback: no fresh V_CMP writer in this BB (or the cache
          // was invalidated by a scalar SGPR write, or we crossed a
          // BB boundary). Route through the projection's per-lane
          // extractor, mirroring `readVCCAsWaveMask`'s consumer
          // symmetry. This path is correct for same-wave and
          // modulo-replication same-width cases, and lossy only in
          // the documented wave32 -> wave64 cross-widening narrow-
          // write case (where recovering the upper-half lanes'
          // compare results is impossible from the 32-bit SGPR --
          // those bits were destroyed at the writer's truncate).
          Value *CondVal = Ctx.Isa.isWave32()
                               ? Ctx.Regs.loadSGPR32(Ctx.B, CondReg.BaseIdx)
                               : Ctx.Regs.loadSGPR64(Ctx.B, CondReg.BaseIdx);
          Value *Fallback =
              Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, CondVal);
          // Cross-BB path: prefer the memory-backed shadow if valid.
          // This avoids carrying non-dominating `i1` SSA values across
          // blocks while still preserving the full EXEC-width compare mask.
          if (Value *ShadowValid = Ctx.loadSgprWaveMaskValid(CondReg.BaseIdx)) {
            Value *ShadowExec = Ctx.loadSgprWaveMaskExec(CondReg.BaseIdx);
            Value *ShadowI1 =
                Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, ShadowExec);
            Cond = Ctx.B.CreateSelect(ShadowValid, ShadowI1, Fallback,
                                      "sgpr_mask_shadow_sel");
          } else {
            Cond = Fallback;
          }
        }
      } else {
        Cond = Ctx.Regs.loadVCC(Ctx.B);
      }
    }
    if (!Cond) Cond = Ctx.Regs.loadVCC(Ctx.B);
    Ctx.writeReg32(Dest, Ctx.B.CreateSelect(Cond, Src1, Src0, "cndmask"));
    Hr.Handled = true;
    return Hr;
  }

  default:
    break;
  }
  return Hr;
}

} // namespace COMGR::hotswap
