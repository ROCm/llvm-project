//===- handle-vopd.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

#include "canonical-op.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

struct PendingVopdWrite {
  ParsedReg Dst;
  Value *Val = nullptr;
};

ParsedReg applyVopdVGPRMsb(const RaiseContext &Ctx, ParsedReg Pr,
                           unsigned Slot) {
  if (Pr.RegKind != ParsedReg::VGPR && Pr.RegKind != ParsedReg::AGPR)
    return Pr;

  Pr.BaseIdx += ((Ctx.VgprMsBs >> (Slot * 2)) & 0x3) * 256;
  return Pr;
}

Value *readVopdVCCAsSource(RaiseContext &Ctx) {
  if (Ctx.Projection.sourceWaveScopedLaneOps()) {
    Value *Mask = Ctx.Regs.readVCCAsWaveMask(Ctx.B, Ctx.Regs.ExecTy);
    Value *Lo = Ctx.B.CreateTrunc(Mask, Ctx.I32Ty, "vopd_vcc_lo_src");
    Value *Hi = Ctx.B.CreateTrunc(
        Ctx.B.CreateLShr(Mask, Ctx.Isa.WaveSize), Ctx.I32Ty,
        "vopd_vcc_hi_src");
    Value *Lane = Ctx.Projection.emitLaneIdx(Ctx.B);
    Value *Upper = Ctx.B.CreateICmpUGE(
        Lane, ConstantInt::get(Ctx.I32Ty, Ctx.Isa.WaveSize),
        "vopd_vcc_upper_src_wave");
    return Ctx.B.CreateSelect(Upper, Hi, Lo, "vopd_vcc_src_wave_mask");
  }
  return Ctx.Regs.readVCCAsWaveMask(Ctx.B, Ctx.I32Ty);
}

Value *applyVopdSourceModifiers(RaiseContext &Ctx, Value *V,
                                uint8_t Modifiers) {
  if (Modifiers == 0)
    return V;
  bool IsI32 = V->getType() == Ctx.I32Ty;
  bool IsI64 = V->getType() == Ctx.I64Ty;
  if (IsI32)
    V = Ctx.B.CreateBitCast(V, Ctx.F32Ty);
  if (IsI64)
    V = Ctx.B.CreateBitCast(V, Type::getDoubleTy(Ctx.C));
  if (Modifiers & 2)
    V = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, V, nullptr, "vopd_abs");
  if (Modifiers & 1)
    V = Ctx.B.CreateFNeg(V, "vopd_neg");
  if (IsI32)
    V = Ctx.B.CreateBitCast(V, Ctx.I32Ty);
  if (IsI64)
    V = Ctx.B.CreateBitCast(V, Ctx.I64Ty);
  return V;
}

Value *readVopdSource(RaiseContext &Ctx, const DecodedInst::VopdSource &Src,
                      unsigned SrcSlot) {
  Value *V = nullptr;
  auto Parsed = [&](ParsedReg::Kind Kind) {
    ParsedReg Pr;
    Pr.RegKind = Kind;
    Pr.BaseIdx = Src.BaseIdx;
    Pr.Width = Src.Width;
    return applyVopdVGPRMsb(Ctx, Pr, SrcSlot);
  };
  switch (Src.SrcKind) {
  case DecodedInst::VopdSource::Kind::None:
    return nullptr;
  case DecodedInst::VopdSource::Kind::Imm:
    V = ConstantInt::get(Ctx.I32Ty,
                         static_cast<uint32_t>(Src.Imm & 0xffffffffu));
    break;
  case DecodedInst::VopdSource::Kind::VGPR:
    V = Ctx.Regs.readReg32(Ctx.B, Parsed(ParsedReg::VGPR));
    break;
  case DecodedInst::VopdSource::Kind::AGPR:
    V = Ctx.Regs.readReg32(Ctx.B, Parsed(ParsedReg::AGPR));
    break;
  case DecodedInst::VopdSource::Kind::SGPR:
    V = Ctx.Regs.loadSGPR32(Ctx.B, Src.BaseIdx);
    break;
  case DecodedInst::VopdSource::Kind::TTMP:
    V = Ctx.B.CreateLoad(Ctx.I32Ty, Ctx.Regs.Ttmp[Src.BaseIdx], "vopd_ttmp");
    break;
  case DecodedInst::VopdSource::Kind::VCC:
    V = readVopdVCCAsSource(Ctx);
    break;
  case DecodedInst::VopdSource::Kind::EXEC: {
    ParsedReg Pr;
    Pr.RegKind = ParsedReg::EXEC;
    Pr.BaseIdx = Src.BaseIdx;
    Pr.Width = Src.Width;
    V = Ctx.Regs.readReg32(Ctx.B, Pr);
    break;
  }
  case DecodedInst::VopdSource::Kind::SCC:
    V = Ctx.B.CreateZExt(Ctx.Regs.loadSCC(Ctx.B), Ctx.I32Ty);
    break;
  case DecodedInst::VopdSource::Kind::M0:
    V = Ctx.B.CreateLoad(Ctx.I32Ty, Ctx.Regs.M0, "vopd_m0");
    break;
  }
  return applyVopdSourceModifiers(Ctx, V, Src.Modifiers);
}

Value *readVopdSource64(RaiseContext &Ctx, const DecodedInst::VopdSource &Src,
                        unsigned SrcSlot, const DecodedInst &Di,
                        HandlerResult &Hr) {
  Value *V = nullptr;
  auto Parsed = [&](ParsedReg::Kind Kind) {
    ParsedReg Pr;
    Pr.RegKind = Kind;
    Pr.BaseIdx = Src.BaseIdx;
    Pr.Width = Src.Width;
    return applyVopdVGPRMsb(Ctx, Pr, SrcSlot);
  };

  switch (Src.SrcKind) {
  case DecodedInst::VopdSource::Kind::None:
    return nullptr;
  case DecodedInst::VopdSource::Kind::Imm:
    V = ConstantInt::get(Ctx.I64Ty, static_cast<uint64_t>(Src.Imm));
    break;
  case DecodedInst::VopdSource::Kind::VGPR:
    V = Ctx.Regs.readReg64(Ctx.B, Parsed(ParsedReg::VGPR));
    break;
  case DecodedInst::VopdSource::Kind::AGPR:
    V = Ctx.Regs.readReg64(Ctx.B, Parsed(ParsedReg::AGPR));
    break;
  case DecodedInst::VopdSource::Kind::SGPR:
    V = Ctx.Regs.loadSGPR64(Ctx.B, Src.BaseIdx);
    break;
  default:
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOPD", "VOPD f64 component source is not a 64-bit scalar/vector source");
    return nullptr;
  }

  return applyVopdSourceModifiers(Ctx, V, Src.Modifiers);
}

Value *readVopdCond(RaiseContext &Ctx, const DecodedInst &Di,
                    const DecodedInst::VopdSource &Src, HandlerResult &Hr) {
  if (Src.SrcKind != DecodedInst::VopdSource::Kind::VCC &&
      Src.SrcKind != DecodedInst::VopdSource::Kind::SGPR) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOPD", "VOPD cndmask explicit condition is neither VCC nor SGPR");
    return nullptr;
  }
  if (Src.SrcKind == DecodedInst::VopdSource::Kind::VCC)
    return Ctx.Regs.loadVCC(Ctx.B);

  if (Value *FreshCmp = Ctx.lookupSgprWaveMaskI1(Src.BaseIdx))
    return FreshCmp;

  Value *CondVal = Ctx.Isa.isWave32()
                       ? Ctx.Regs.loadSGPR32(Ctx.B, Src.BaseIdx)
                       : Ctx.Regs.loadSGPR64(Ctx.B, Src.BaseIdx);
  Value *Fallback = Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, CondVal);
  if (Value *ShadowValid = Ctx.loadSgprWaveMaskValid(Src.BaseIdx)) {
    Value *ShadowExec = Ctx.loadSgprWaveMaskExec(Src.BaseIdx);
    Value *ShadowI1 = Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B,
                                                                 ShadowExec);
    return Ctx.B.CreateSelect(ShadowValid, ShadowI1, Fallback,
                              "vopd_sgpr_mask_shadow_sel");
  }
  return Fallback;
}

bool requireVopdSources(const DecodedInst::VopdHalf &Half, unsigned N,
                        const DecodedInst &Di, HandlerResult &Hr) {
  if (Half.NumSrcs >= N)
    return true;
  Hr.Failure = RaiseFailure::unsupportedInstructionForm(
      Di, "VOPD", "VOPD component has too few decoded sources");
  return false;
}

bool requireVopdRegWidth(const DecodedInst &Di, const char *What,
                         unsigned Width, unsigned MinWidth,
                         HandlerResult &Hr) {
  if (Width >= MinWidth)
    return true;
  Hr.Failure = RaiseFailure::unsupportedInstructionForm(
      Di, "VOPD", (Twine("VOPD ") + What + " is narrower than " +
                   Twine(MinWidth) + " dwords")
                      .str());
  return false;
}

bool lowerVopdHalf(RaiseContext &Ctx, const DecodedInst &Di,
                   const DecodedInst::VopdHalf &Half,
                   SmallVectorImpl<PendingVopdWrite> &Writes,
                   HandlerResult &Hr) {
  ParsedReg Dst = applyVopdVGPRMsb(
      Ctx, Ctx.parseReg(Half.DstReg, /*mciOpIdx=*/-1),
      /*slot=*/3);
  auto Queue = [&](Value *V) {
    // VOPD destination operands name the low VGPR slot even for 64-bit
    // components. A 64-bit commit writes [baseIdx, baseIdx+1] through the
    // register file helper below, so dst.width is not a reliable arity check.
    Writes.push_back(PendingVopdWrite{Dst, V});
    return true;
  };
  auto LowerBitOp3 = [&]() {
    if (!requireVopdSources(Half, 2, Di, Hr)) return false;
    Value *A = readVopdSource(Ctx, Half.Src[0], 0);
    Value *B = readVopdSource(Ctx, Half.Src[1], 1);
    Value *C = ConstantInt::get(Ctx.I32Ty, 0);
    Value *Na = Ctx.B.CreateNot(A);
    Value *Nb = Ctx.B.CreateNot(B);
    Value *Nc = Ctx.B.CreateNot(C);
    Value *Minterms[8] = {
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, Nb), Nc),
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, Nb), C),
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, B), Nc),
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(Na, B), C),
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, Nb), Nc),
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, Nb), C),
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, B), Nc),
        Ctx.B.CreateAnd(Ctx.B.CreateAnd(A, B), C),
    };
    Value *Result = ConstantInt::get(Ctx.I32Ty, 0);
    for (int I = 0; I < 8; ++I)
      if (Half.BitOp3 & (1u << I))
        Result = Ctx.B.CreateOr(Result, Minterms[I]);
    return Queue(Result);
  };

  // `v_dual_bitop2_b32` components carry an 8-bit `bitop3` truth table even
  // though LLVM's canonical component opcode may look like a simple V_AND /
  // V_OR / V_XOR. The immediate is semantic, not decoration: Triton's sort
  // uses values such as 0x14 (xor) and 0x40 (and) in this encoding.
  if (Half.HasBitOp3)
    return LowerBitOp3();

  switch (Half.CanonOp) {
  case CanonicalOp::V_MOV_B32: {
    if (!requireVopdSources(Half, 1, Di, Hr)) return false;
    return Queue(readVopdSource(Ctx, Half.Src[0], 0));
  }
  case CanonicalOp::V_CNDMASK_B32: {
    if (!requireVopdSources(Half, 2, Di, Hr)) return false;
    Value *S0 = readVopdSource(Ctx, Half.Src[0], 0);
    Value *S1 = readVopdSource(Ctx, Half.Src[1], 1);
    Value *Cond = Half.NumSrcs >= 3 ? readVopdCond(Ctx, Di, Half.Src[2], Hr)
                                    : Ctx.Regs.loadVCC(Ctx.B);
    if (!Cond) return false;
    return Queue(Ctx.B.CreateSelect(Cond, S1, S0, "vopd_cndmask"));
  }
  case CanonicalOp::V_ADD_F32:
  case CanonicalOp::V_MUL_F32:
  case CanonicalOp::V_SUB_F32:
  case CanonicalOp::V_SUBREV_F32: {
    if (!requireVopdSources(Half, 2, Di, Hr)) return false;
    Value *S0 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[0], 0),
                                    Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[1], 1),
                                    Ctx.F32Ty);
    Value *Res = nullptr;
    if (Half.CanonOp == CanonicalOp::V_ADD_F32)
      Res = Ctx.B.CreateFAdd(S0, S1, "vopd_fadd");
    else if (Half.CanonOp == CanonicalOp::V_MUL_F32)
      Res = Ctx.B.CreateFMul(S0, S1, "vopd_fmul");
    else if (Half.CanonOp == CanonicalOp::V_SUBREV_F32)
      Res = Ctx.B.CreateFSub(S1, S0, "vopd_fsubrev");
    else
      Res = Ctx.B.CreateFSub(S0, S1, "vopd_fsub");
    return Queue(Ctx.B.CreateBitCast(Res, Ctx.I32Ty));
  }
  case CanonicalOp::V_FMAC_F32: {
    if (!requireVopdSources(Half, 2, Di, Hr)) return false;
    Value *S0 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[0], 0),
                                    Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[1], 1),
                                    Ctx.F32Ty);
    Value *Acc = Ctx.B.CreateBitCast(Ctx.Regs.readReg32(Ctx.B, Dst),
                                     Ctx.F32Ty);
    // llvm.fma (not llvm.fmuladd) -- v_dual_fmac_f32 is hardware-guaranteed fused; fmuladd may be split by middle-end passes.
    Function *Fma =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma,
                                          {Ctx.F32Ty});
    return Queue(Ctx.B.CreateBitCast(
        Ctx.B.CreateCall(Fma, {S0, S1, Acc}, "vopd_fmac"), Ctx.I32Ty));
  }
  case CanonicalOp::V_FMA_F32: {
    if (!requireVopdSources(Half, 3, Di, Hr)) return false;
    Value *S0 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[0], 0),
                                    Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[1], 1),
                                    Ctx.F32Ty);
    Value *S2 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[2], 2),
                                    Ctx.F32Ty);
    Function *Fma =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma,
                                          {Ctx.F32Ty});
    return Queue(Ctx.B.CreateBitCast(
        Ctx.B.CreateCall(Fma, {S0, S1, S2}, "vopd_fma"), Ctx.I32Ty));
  }
  case CanonicalOp::V_MUL_F64:
  case CanonicalOp::V_ADD_F64:
  case CanonicalOp::V_MAX_NUM_F64:
  case CanonicalOp::V_MIN_NUM_F64:
  case CanonicalOp::V_FMA_F64: {
    unsigned NumSrcs = Half.CanonOp == CanonicalOp::V_FMA_F64 ? 3 : 2;
    if (!requireVopdSources(Half, NumSrcs, Di, Hr)) return false;
    for (unsigned I = 0; I < NumSrcs; ++I) {
      if (Half.Src[I].SrcKind != DecodedInst::VopdSource::Kind::Imm &&
          !requireVopdRegWidth(Di, "f64 source", Half.Src[I].Width, 2, Hr))
        return false;
    }
    auto *F64Ty = Type::getDoubleTy(Ctx.C);
    Value *S0 = readVopdSource64(Ctx, Half.Src[0], 0, Di, Hr);
    if (!S0) return false;
    Value *S1 = readVopdSource64(Ctx, Half.Src[1], 1, Di, Hr);
    if (!S1) return false;
    S0 = Ctx.B.CreateBitCast(S0, F64Ty);
    S1 = Ctx.B.CreateBitCast(S1, F64Ty);

    Value *Res = nullptr;
    if (Half.CanonOp == CanonicalOp::V_MUL_F64) {
      Res = Ctx.B.CreateFMul(S0, S1, "vopd_fmul_f64");
    } else if (Half.CanonOp == CanonicalOp::V_ADD_F64) {
      Res = Ctx.B.CreateFAdd(S0, S1, "vopd_fadd_f64");
    } else if (Half.CanonOp == CanonicalOp::V_MAX_NUM_F64 ||
               Half.CanonOp == CanonicalOp::V_MIN_NUM_F64) {
      Intrinsic::ID Id = Half.CanonOp == CanonicalOp::V_MAX_NUM_F64
                             ? Intrinsic::maximumnum
                             : Intrinsic::minimumnum;
      Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Id, {F64Ty});
      const char *Name = Half.CanonOp == CanonicalOp::V_MAX_NUM_F64
                             ? "vopd_fmaxnum_f64"
                             : "vopd_fminnum_f64";
      Res = Ctx.B.CreateCall(Fn, {S0, S1}, Name);
    } else {
      Value *S2 = readVopdSource64(Ctx, Half.Src[2], 2, Di, Hr);
      if (!S2) return false;
      S2 = Ctx.B.CreateBitCast(S2, F64Ty);
      Function *Fma = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma,
                                                        {F64Ty});
      Res = Ctx.B.CreateCall(Fma, {S0, S1, S2}, "vopd_fma_f64");
    }
    return Queue(Ctx.B.CreateBitCast(Res, Ctx.I64Ty));
  }
  case CanonicalOp::V_FMAMK_F32:
  case CanonicalOp::V_FMAAK_F32: {
    if (!requireVopdSources(Half, 3, Di, Hr)) return false;
    Value *S0 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[0], 0),
                                    Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[1], 1),
                                    Ctx.F32Ty);
    // MADK VOPD encodings have only src0/vsrc1 register fields; the mandatory
    // literal occupies a logical source slot but not a VGPR-MSB slot.
    unsigned S2Slot = Half.CanonOp == CanonicalOp::V_FMAMK_F32 ? 1 : 2;
    Value *S2 =
        Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[2], S2Slot),
                            Ctx.F32Ty);
    Function *Fma =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma,
                                          {Ctx.F32Ty});
    const char *Name =
        Half.CanonOp == CanonicalOp::V_FMAMK_F32 ? "vopd_fmamk" : "vopd_fmaak";
    return Queue(Ctx.B.CreateBitCast(
        Ctx.B.CreateCall(Fma, {S0, S1, S2}, Name), Ctx.I32Ty));
  }
  case CanonicalOp::V_ADD_NC_U32:
  case CanonicalOp::V_SUB_NC_U32:
  case CanonicalOp::V_SUBREV_NC_U32:
  case CanonicalOp::V_LSHLREV_B32:
  case CanonicalOp::V_LSHRREV_B32:
  case CanonicalOp::V_ASHRREV_I32:
  case CanonicalOp::V_AND_B32:
  case CanonicalOp::V_OR_B32:
  case CanonicalOp::V_XOR_B32: {
    if (!requireVopdSources(Half, 2, Di, Hr)) return false;
    Value *S0 = readVopdSource(Ctx, Half.Src[0], 0);
    Value *S1 = readVopdSource(Ctx, Half.Src[1], 1);
    Value *Res = nullptr;
    switch (Half.CanonOp) {
    case CanonicalOp::V_ADD_NC_U32:    Res = Ctx.B.CreateAdd(S0, S1, "vopd_add"); break;
    case CanonicalOp::V_SUB_NC_U32:    Res = Ctx.B.CreateSub(S0, S1, "vopd_sub"); break;
    case CanonicalOp::V_SUBREV_NC_U32: Res = Ctx.B.CreateSub(S1, S0, "vopd_subrev"); break;
    case CanonicalOp::V_LSHLREV_B32:   Res = Ctx.B.CreateShl(S1, S0, "vopd_shl"); break;
    case CanonicalOp::V_LSHRREV_B32:   Res = Ctx.B.CreateLShr(S1, S0, "vopd_lshr"); break;
    case CanonicalOp::V_ASHRREV_I32:   Res = Ctx.B.CreateAShr(S1, S0, "vopd_ashr"); break;
    case CanonicalOp::V_AND_B32:       Res = Ctx.B.CreateAnd(S0, S1, "vopd_and"); break;
    case CanonicalOp::V_OR_B32:        Res = Ctx.B.CreateOr(S0, S1, "vopd_or"); break;
    case CanonicalOp::V_XOR_B32:       Res = Ctx.B.CreateXor(S0, S1, "vopd_xor"); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    return Queue(Res);
  }
  case CanonicalOp::V_MAX_I32:
  case CanonicalOp::V_MIN_I32:
  case CanonicalOp::V_MAX_U32:
  case CanonicalOp::V_MIN_U32: {
    if (!requireVopdSources(Half, 2, Di, Hr)) return false;
    Value *S0 = readVopdSource(Ctx, Half.Src[0], 0);
    Value *S1 = readVopdSource(Ctx, Half.Src[1], 1);
    Intrinsic::ID Id = Intrinsic::smax;
    if (Half.CanonOp == CanonicalOp::V_MIN_I32) Id = Intrinsic::smin;
    if (Half.CanonOp == CanonicalOp::V_MAX_U32) Id = Intrinsic::umax;
    if (Half.CanonOp == CanonicalOp::V_MIN_U32) Id = Intrinsic::umin;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Id, {Ctx.I32Ty});
    const char *Name = "vopd_smax";
    if (Half.CanonOp == CanonicalOp::V_MIN_I32) Name = "vopd_smin";
    if (Half.CanonOp == CanonicalOp::V_MAX_U32) Name = "vopd_umax";
    if (Half.CanonOp == CanonicalOp::V_MIN_U32) Name = "vopd_umin";
    return Queue(Ctx.B.CreateCall(Fn, {S0, S1}, Name));
  }
  case CanonicalOp::V_MAX_NUM_F32:
  case CanonicalOp::V_MIN_NUM_F32: {
    if (!requireVopdSources(Half, 2, Di, Hr)) return false;
    Value *S0 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[0], 0),
                                    Ctx.F32Ty);
    Value *S1 = Ctx.B.CreateBitCast(readVopdSource(Ctx, Half.Src[1], 1),
                                    Ctx.F32Ty);
    Intrinsic::ID Id =
        Half.CanonOp == CanonicalOp::V_MAX_NUM_F32 ? Intrinsic::maximumnum
                                                     : Intrinsic::minimumnum;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Id, {Ctx.F32Ty});
    const char *Name =
        Half.CanonOp == CanonicalOp::V_MAX_NUM_F32 ? "vopd_fmax" : "vopd_fmin";
    return Queue(Ctx.B.CreateBitCast(
        Ctx.B.CreateCall(Fn, {S0, S1}, Name), Ctx.I32Ty));
  }
  case CanonicalOp::V_BITOP3_B32: {
    if (!Half.HasBitOp3) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOPD", "VOPD bitop component missing bitop3 immediate");
      return false;
    }
    return LowerBitOp3();
  }
  default:
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOPD", "unhandled structural VOPD component CanonicalOp");
    return false;
  }
}

} // namespace

HandlerResult handleVOPD(RaiseContext &Ctx, const DecodedInst &Di,
                        OpResolver &Op) {
  HandlerResult Hr;
  (void)Op;
  if (!Di.HasVopd) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOPD", "VOPD instruction reached handler without sidecar");
    return Hr;
  }

  SmallVector<PendingVopdWrite, 4> PendingVgprWrites;
  bool XOk = lowerVopdHalf(Ctx, Di, Di.Vopd[AMDGPU::VOPD::ComponentIndex::X],
                           PendingVgprWrites, Hr);
  bool YOk = XOk && lowerVopdHalf(
                        Ctx, Di, Di.Vopd[AMDGPU::VOPD::ComponentIndex::Y],
                        PendingVgprWrites, Hr);
  if (!XOk || !YOk)
    return Hr;

  // VOPD executes as a paired issue packet: both halves read pre-instruction
  // register state. Commit writes only after both halves are decoded/lifted.
  for (const auto &W : PendingVgprWrites) {
    if (W.Val->getType()->getPrimitiveSizeInBits() == 64)
      Ctx.writeReg64(W.Dst, W.Val);
    else
      Ctx.writeReg32(W.Dst, W.Val);
  }
  Hr.Handled = true;
  return Hr;
}

} // namespace COMGR::hotswap
