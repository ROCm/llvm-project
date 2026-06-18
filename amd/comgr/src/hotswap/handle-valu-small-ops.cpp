//===- handle-valu-small-ops.cpp - Hotswap transpiler ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-internal.h"
#include "handle-valu-f16-utils.h"
#include "handle-valu-output-mods.h"

#include "canonical-op.h"
#include "ocml-runtime.h"

#include "SIDefines.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/LogicalResult.h"
#include "Utils/AMDGPUBaseInfo.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Lift a bf16 unary transcendental through an f32 callee, wrapping with
// bf16<->f32 fpext/fptrunc and merging into the dst half. Half-select honors
// both op_sel modifiers and _HI16 subreg naming. Returns failure (with
// Hr.Failure set) when a present src0 modifier operand is malformed.
LogicalResult emitBF16UnaryViaF32Callee(RaiseContext &Ctx, OpResolver &Op,
                                        HandlerResult &Hr,
                                        FunctionCallee F32Callee,
                                        StringRef Name) {
  const DecodedInst &Di = Op.Di;
  const MCRegisterInfo &MRI = *Ctx.Mc.RegInfo;
  unsigned Mods = 0;
  if (!readOptionalVOP3F16SrcMods(Di, Hr, 0, Name, Mods))
    return failure();
  Type *BfTy = Type::getBFloatTy(Ctx.C);
  Type *I16Ty = Type::getInt16Ty(Ctx.C);

  unsigned SrcSlot = Di.SrcMap[0];
  bool SrcHi = (Mods & SISrcMods::OP_SEL_0) != 0 ||
               (Di.isReg(SrcSlot) &&
                AMDGPU::isHi16Reg(Di.getReg(SrcSlot), MRI));
  bool DstHi = (Mods & SISrcMods::DST_OP_SEL) != 0 ||
               (Di.isReg(0) && AMDGPU::isHi16Reg(Di.getReg(0), MRI));

  Value *Raw = Op.src(0);
  if (SrcHi)
    Raw = Ctx.B.CreateLShr(Raw, 16, (Name + "_src_hi").str());
  Value *Bits = Ctx.B.CreateTrunc(Raw, I16Ty);
  Value *Bf = Op.applyMods(0, Ctx.B.CreateBitCast(Bits, BfTy));

  Value *F32 = Ctx.B.CreateFPExt(Bf, Ctx.F32Ty, (Name + "_ext").str());
  Value *Res32 = Ctx.B.CreateCall(F32Callee, {F32}, Name);
  Value *ResBf = Ctx.B.CreateFPTrunc(Res32, BfTy, (Name + "_tr").str());

  writeOpSelF16(Ctx, Op, ResBf, DstHi, "bf16_merge_lo", "bf16_merge_hi");
  return success();
}

FunctionCallee getF32Intrinsic(RaiseContext &Ctx, Intrinsic::ID IID) {
  return Intrinsic::getOrInsertDeclaration(&Ctx.M, IID, {Ctx.F32Ty});
}

} // namespace

// "Small ops": conversions (F32<->{U,I}32, F16<->F32, F16<->{U,I}16, byte
// extract), F16 two-src arith (add/sub/mul/min/max/mac/fmac), packed
// F16 fmac, 16-bit min/max and reverse-operand shifts, byte pack,
// V_BFREV_B32 / V_NOT_B32, and F32 single-src transcendentals
// (rcp/exp/log/ldexp/sqrt/rsq/floor/ceil/trunc/rndne/fract).
//
// Grouped here because each case is 1-5 lines of IR emission and they
// would bloat the arithmetic / 3-src sub-handlers if interleaved.
// Structured as a switch on CanonicalOp: cases are mutually exclusive and
// ordering is not load-bearing.
HandlerResult handleValuSmallOps(RaiseContext &Ctx, const DecodedInst &Di,
                                   OpResolver &Op) {
  HandlerResult Hr;
  Type *I16Ty = Type::getInt16Ty(Ctx.C);
  Type *HalfTy = Type::getHalfTy(Ctx.C);

  switch (Di.CanonOp) {
  // ---- F32 <-> integer conversions ----
  case CanonicalOp::V_CVT_F32_U32: {
    Value *R = Ctx.B.CreateUIToFP(Op.src(0), Ctx.F32Ty, "cvt");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_I32: {
    Value *R = Ctx.B.CreateSIToFP(Op.src(0), Ctx.F32Ty, "cvt");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_U32_F32: {
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateFPToUI(S, Ctx.I32Ty, "cvt"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_I32_F32: {
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateFPToSI(S, Ctx.I32Ty, "cvt"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F16_F32: {
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Value *H = Ctx.B.CreateFPTrunc(S, HalfTy, "cvt");
    Value *Bits = Ctx.B.CreateBitCast(H, I16Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Bits, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_F16: {
    Value *Bits = Ctx.B.CreateTrunc(Op.src(0), I16Ty);
    Value *H = Ctx.B.CreateBitCast(Bits, HalfTy);
    Value *F = Ctx.B.CreateFPExt(H, Ctx.F32Ty, "cvt");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(F, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F16_U16: {
    Value *S = Ctx.B.CreateTrunc(Op.src(0), I16Ty);
    Value *Res = Ctx.B.CreateUIToFP(S, Ctx.F16Ty, "cvt_f16_u16");
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateZExt(Ctx.B.CreateBitCast(Res, I16Ty),
                                    Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_U16_F16: {
    Value *S = Ctx.B.CreateBitCast(Ctx.B.CreateTrunc(Op.srcF(0), I16Ty),
                                    Ctx.F16Ty);
    Value *Res = Ctx.B.CreateFPToUI(S, I16Ty, "cvt_u16_f16");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  // gfx11+ true16/fake16 u16 -> u32 zero-extend. The 16-bit source half
  // selection lives in one of two places depending on the encoding form:
  //   * `_e32` (fake16 today): the MCInst's src0 slot holds a `_LO16` /
  //     `_HI16` subreg of the parent VGPR; there is no modifier operand.
  //   * `_e64`: src0_modifiers carries the OP_SEL_0 bit; the register
  //     operand is the base 32-bit VGPR.
  // The destination is a full 32-bit VGPR receiving the zero-extended u16,
  // so DST_OP_SEL does not apply and we refuse it. `Op.src(0)` returns the
  // parent VGPR's i32 value regardless of which subreg the MCInst slot
  // named, so the lift is `trunc(lshr_if_hi(src0, 16), i16)` zero-extended
  // back to i32.
  case CanonicalOp::V_CVT_U32_U16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    unsigned Mods = Op.srcMod(0);
    constexpr unsigned AllowedMods = SISrcMods::OP_SEL_0;
    if ((Mods & ~AllowedMods) != 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "v_cvt_u32_u16 has unsupported source modifiers; only src0 "
          "op_sel (half select) is modeled");
      return Hr;
    }
    const MCRegisterInfo &MRI = *Ctx.Mc.RegInfo;
    unsigned SrcSlot = Di.SrcMap[0];
    bool Src0SubHi = Di.isReg(SrcSlot) &&
                     AMDGPU::isHi16Reg(Di.getReg(SrcSlot), MRI);
    bool Src0Hi = Src0SubHi || (Mods & SISrcMods::OP_SEL_0) != 0;
    Value *Raw = Op.src(0);
    if (Src0Hi)
      Raw = Ctx.B.CreateLShr(Raw, 16, "cvt_u32_u16_hi");
    Value *Half = Ctx.B.CreateTrunc(Raw, I16Ty);
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateZExt(Half, Ctx.I32Ty, "cvt_u32_u16"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_F64: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (Di.HasDpp) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "V_CVT_F32_F64 DPP has mixed source/destination widths; inactive "
          "lane preservation must be modeled as old-destination semantics, "
          "not the generic same-width DPP source wrapper");
      return Hr;
    }
    Value *Src = Ctx.B.CreateBitCast(Op.src64(0), Ctx.F64Ty);
    Src = Op.applyMods(0, Src);
    Value *Result = Ctx.B.CreateFPTrunc(Src, Ctx.F32Ty, "cvt_f32_f64");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Result, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F64_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (Di.HasDpp) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "V_CVT_F64_F32 DPP has mixed source/destination widths; inactive "
          "lane preservation must be modeled as old-destination semantics, "
          "not the generic same-width DPP source wrapper");
      return Hr;
    }
    Value *Src = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Value *Result = Ctx.B.CreateFPExt(Src, Ctx.F64Ty, "cvt_f64_f32");
    Ctx.writeReg64(Op.dst(), Ctx.B.CreateBitCast(Result, Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_UBYTE0: {
    Value *Byte = Ctx.B.CreateAnd(Op.src(0),
                                   ConstantInt::get(Ctx.I32Ty, 0xFF));
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateUIToFP(Byte, Ctx.F32Ty), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_UBYTE1: {
    Value *Byte = Ctx.B.CreateAnd(Ctx.B.CreateLShr(Op.src(0), 8),
                                   ConstantInt::get(Ctx.I32Ty, 0xFF));
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateUIToFP(Byte, Ctx.F32Ty), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_UBYTE2: {
    Value *Byte = Ctx.B.CreateAnd(Ctx.B.CreateLShr(Op.src(0), 16),
                                   ConstantInt::get(Ctx.I32Ty, 0xFF));
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateUIToFP(Byte, Ctx.F32Ty), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_UBYTE3: {
    Value *Byte = Ctx.B.CreateLShr(Op.src(0), 24);
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateUIToFP(Byte, Ctx.F32Ty), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- F16 two-src arith (reused i16 bitcast idiom) ----
  case CanonicalOp::V_MUL_F16:
  case CanonicalOp::V_ADD_F16:
  case CanonicalOp::V_SUB_F16:
  case CanonicalOp::V_SUBREV_F16:
  case CanonicalOp::V_MAX_NUM_F16:
  case CanonicalOp::V_MIN_NUM_F16: {
    Value *A = Ctx.B.CreateBitCast(Ctx.B.CreateTrunc(Op.srcF(0), I16Ty),
                                    Ctx.F16Ty);
    Value *B = Ctx.B.CreateBitCast(Ctx.B.CreateTrunc(Op.srcF(1), I16Ty),
                                    Ctx.F16Ty);
    Value *Res = nullptr;
    switch (Di.CanonOp) {
    case CanonicalOp::V_MUL_F16:    Res = Ctx.B.CreateFMul(A, B, "mul_f16"); break;
    case CanonicalOp::V_ADD_F16:    Res = Ctx.B.CreateFAdd(A, B, "add_f16"); break;
    case CanonicalOp::V_SUB_F16:    Res = Ctx.B.CreateFSub(A, B, "sub_f16"); break;
    case CanonicalOp::V_SUBREV_F16: Res = Ctx.B.CreateFSub(B, A, "subrev_f16"); break;
    case CanonicalOp::V_MAX_NUM_F16: {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maximumnum, {Ctx.F16Ty});
      Res = Ctx.B.CreateCall(Fn, {A, B}, "max_f16");
      break;
    }
    case CanonicalOp::V_MIN_NUM_F16: {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minimumnum, {Ctx.F16Ty});
      Res = Ctx.B.CreateCall(Fn, {A, B}, "min_f16");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateZExt(Ctx.B.CreateBitCast(Res, I16Ty),
                                    Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- Packed <2xf16> FMA into dst ----
  case CanonicalOp::V_PK_FMAC_F16: {
    auto *V2f16 = FixedVectorType::get(Ctx.F16Ty, 2);
    Value *S0 = Ctx.B.CreateBitCast(Op.src(0), V2f16);
    Value *S1 = Ctx.B.CreateBitCast(Op.src(1), V2f16);
    Value *Acc = Ctx.B.CreateBitCast(Ctx.Regs.readReg32(Ctx.B, Op.dst()),
                                      V2f16);
    Function *Fma = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fma, {V2f16});
    Value *Res = Ctx.B.CreateCall(Fma, {S0, S1, Acc}, "pk_fmac");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- F16 MAC: dst = src0 * src1 + dst ----
  case CanonicalOp::V_MAC_F16:
  case CanonicalOp::V_FMAC_F16: {
    Value *S0 = Ctx.B.CreateBitCast(Ctx.B.CreateTrunc(Op.srcF(0), I16Ty),
                                     Ctx.F16Ty);
    Value *S1 = Ctx.B.CreateBitCast(Ctx.B.CreateTrunc(Op.srcF(1), I16Ty),
                                     Ctx.F16Ty);
    Value *Acc = Ctx.B.CreateBitCast(
        Ctx.B.CreateTrunc(Ctx.Regs.readReg32(Ctx.B, Op.dst()), I16Ty),
        Ctx.F16Ty);
    Function *Fma = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fma, {Ctx.F16Ty});
    Value *Res = Ctx.B.CreateBitCast(
        Ctx.B.CreateCall(Fma, {S0, S1, Acc}, "mac_f16"), I16Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  case CanonicalOp::V_FLOOR_F16: {
    Value *S = Ctx.B.CreateBitCast(Ctx.B.CreateTrunc(Op.srcF(0), I16Ty),
                                    Ctx.F16Ty);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::floor, {Ctx.F16Ty});
    Value *Res = Ctx.B.CreateBitCast(
        Ctx.B.CreateCall(Fn, {S}, "floor_f16"), I16Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_LDEXP_F16: {
    Value *S0 = Ctx.B.CreateBitCast(Ctx.B.CreateTrunc(Op.srcF(0), I16Ty),
                                     Ctx.F16Ty);
    Value *S1 = Ctx.B.CreateTrunc(Op.src(1), I16Ty);
    Function *LdexpFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ldexp, {Ctx.F16Ty, I16Ty});
    Value *Res = Ctx.B.CreateBitCast(
        Ctx.B.CreateCall(LdexpFn, {S0, S1}, "ldexp_f16"), I16Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_TANH_F16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;

    bool DstHigh = false;
    unsigned Mods = 0;
    if (!readOptionalVOP3F16SrcMods(Di, Hr, 0, "v_tanh_f16", Mods))
      return Hr;
    DstHigh = (Mods & SISrcMods::DST_OP_SEL) != 0;

    Value *Src = readOptionalOpSelF16(Ctx, Di, Op, Hr, 0, "v_tanh_f16");
    if (!Src)
      return Hr;

    Value *Result = nullptr;
    if (Ctx.TargetIsa.HasTanhInsts) {
      Function *TanhFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_tanh, {Ctx.F16Ty});
      Result = Ctx.B.CreateCall(TanhFn, {Src}, "tanh_f16");
    } else {
      // Targets without a native f16 tanh instruction lower through OCML
      // rather than widening through f32 or inventing a local approximation.
      FunctionCallee TanhFn = declareOCMLTanhF16(Ctx.M);
      Result = Ctx.B.CreateCall(TanhFn, {Src}, "ocml.tanh_f16");
    }

    writeOpSelF16(Ctx, Op, Result, DstHigh, "tanh_f16_merge_lo",
                  "tanh_f16_merge_hi");
    Hr.Handled = true;
    return Hr;
  }

  // ---- 16-bit integer min/max ----
  case CanonicalOp::V_MAX_U16:
  case CanonicalOp::V_MIN_U16:
  case CanonicalOp::V_MAX_I16:
  case CanonicalOp::V_MIN_I16: {
    Value *A = Ctx.B.CreateTrunc(Op.src(0), I16Ty);
    Value *B = Ctx.B.CreateTrunc(Op.src(1), I16Ty);
    Value *Cmp = nullptr;
    switch (Di.CanonOp) {
    case CanonicalOp::V_MAX_U16: Cmp = Ctx.B.CreateICmpUGT(A, B); break;
    case CanonicalOp::V_MIN_U16: Cmp = Ctx.B.CreateICmpULT(A, B); break;
    case CanonicalOp::V_MAX_I16: Cmp = Ctx.B.CreateICmpSGT(A, B); break;
    case CanonicalOp::V_MIN_I16: Cmp = Ctx.B.CreateICmpSLT(A, B); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    Value *Res = Ctx.B.CreateSelect(Cmp, A, B, "i16sel");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- 16-bit integer arith (no carry) ----
  // Plain wrapping i16 add/sub/subrev/mul. v_mul_lo_u16 returns the
  // low 16 bits, naturally produced by `mul i16` without an explicit
  // truncate. The sign-agnostic v_*_u16 family uses `add`/`sub`/`mul`
  // directly (per VOP2Instructions.td:add/sub/mul ARITH PatFrag).
  case CanonicalOp::V_ADD_U16:
  case CanonicalOp::V_SUB_U16:
  case CanonicalOp::V_SUBREV_U16:
  case CanonicalOp::V_MUL_LO_U16: {
    Value *A = Ctx.B.CreateTrunc(Op.src(0), I16Ty);
    Value *B = Ctx.B.CreateTrunc(Op.src(1), I16Ty);
    Value *Res = nullptr;
    switch (Di.CanonOp) {
    case CanonicalOp::V_ADD_U16:    Res = Ctx.B.CreateAdd(A, B, "vadd16"); break;
    case CanonicalOp::V_SUB_U16:    Res = Ctx.B.CreateSub(A, B, "vsub16"); break;
    case CanonicalOp::V_SUBREV_U16: Res = Ctx.B.CreateSub(B, A, "vsubrev16"); break;
    case CanonicalOp::V_MUL_LO_U16: Res = Ctx.B.CreateMul(A, B, "vmullo16"); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- 16-bit reverse-operand shifts (HW uses src0[3:0]) ----
  case CanonicalOp::V_ASHRREV_I16:
  case CanonicalOp::V_LSHRREV_B16:
  case CanonicalOp::V_LSHLREV_B16: {
    Value *Shamt = Ctx.B.CreateAnd(Ctx.B.CreateTrunc(Op.src(0), I16Ty),
                                    ConstantInt::get(I16Ty, 0xF));
    Value *Base = Ctx.B.CreateTrunc(Op.src(1), I16Ty);
    Value *Res = nullptr;
    switch (Di.CanonOp) {
    case CanonicalOp::V_ASHRREV_I16: Res = Ctx.B.CreateAShr(Base, Shamt, "vashr16"); break;
    case CanonicalOp::V_LSHRREV_B16: Res = Ctx.B.CreateLShr(Base, Shamt, "vlshr16"); break;
    case CanonicalOp::V_LSHLREV_B16: Res = Ctx.B.CreateShl(Base, Shamt, "vlshl16"); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  case CanonicalOp::V_PACK_B32_F16: {
    Value *Lo = Ctx.B.CreateAnd(Op.src(0),
                                 ConstantInt::get(Ctx.I32Ty, 0xFFFF));
    Value *Hi = Ctx.B.CreateShl(
        Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0xFFFF)), 16);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Lo, Hi, "pack_f16"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- Simple bit-twiddle single-src ----
  case CanonicalOp::V_BFREV_B32: {
    Function *Brev = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::bitreverse, {Ctx.I32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateCall(Brev, {Op.src(0)}, "bfrev"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_NOT_B32: {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateNot(Op.src(0), "vnot"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- find-first-bit (VOP1, gfx7+) ----
  // V_FFBH_U32 / V_FFBL_B32 use llvm.ctlz / llvm.cttz with
  // is_zero_undef=false so LLVM returns the bitwidth (32) for input 0.
  // Hardware instead returns -1 for input 0, so we explicitly cmov to
  // -1 on the zero-input path. V_FFBH_I32 uses the dedicated
  // llvm.amdgcn.sffbh intrinsic which selects directly back to
  // v_ffbh_i32_e32 (no fixup needed -- the intrinsic and the hardware
  // share the "-1 on uniform-sign input" convention).
  case CanonicalOp::V_FFBH_U32: {
    Function *Ctlz = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ctlz, {Ctx.I32Ty});
    Value *Src = Op.src(0);
    Value *Raw = Ctx.B.CreateCall(
        Ctlz, {Src, ConstantInt::getFalse(Ctx.I1Ty)}, "ffbh_u32_raw");
    Value *IsZero = Ctx.B.CreateICmpEQ(Src, Ctx.B.getInt32(0), "ffbh_u32_zero");
    Value *Res = Ctx.B.CreateSelect(IsZero, Ctx.B.getInt32(-1), Raw,
                                    "ffbh_u32");
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_FFBL_B32: {
    Function *Cttz = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::cttz, {Ctx.I32Ty});
    Value *Src = Op.src(0);
    Value *Raw = Ctx.B.CreateCall(
        Cttz, {Src, ConstantInt::getFalse(Ctx.I1Ty)}, "ffbl_b32_raw");
    Value *IsZero = Ctx.B.CreateICmpEQ(Src, Ctx.B.getInt32(0), "ffbl_b32_zero");
    Value *Res = Ctx.B.CreateSelect(IsZero, Ctx.B.getInt32(-1), Raw,
                                    "ffbl_b32");
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_FFBH_I32: {
    Function *Sffbh = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_sffbh, {Ctx.I32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateCall(Sffbh, {Op.src(0)}, "ffbh_i32"));
    Hr.Handled = true;
    return Hr;
  }

  // out = (in << 1) ^ (in[31] ? 197 : 0). Use the intrinsic where it
  // selects (HasPrngInst); expand in IR for targets without a pattern.
  case CanonicalOp::V_PRNG_B32: {
    Value *Src = Op.src(0);
    Value *Res;
    if (Ctx.TargetIsa.HasPrngInst) {
      Function *PrngFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_prng_b32);
      Res = Ctx.B.CreateCall(PrngFn, {Src}, "prng_b32");
    } else {
      Value *Shl = Ctx.B.CreateShl(Src, ConstantInt::get(Ctx.I32Ty, 1),
                                   "prng_shl");
      Value *Neg = Ctx.B.CreateICmpSLT(Src, ConstantInt::get(Ctx.I32Ty, 0),
                                       "prng_neg");
      Value *Tap = Ctx.B.CreateSelect(Neg, ConstantInt::get(Ctx.I32Ty, 197),
                                      ConstantInt::get(Ctx.I32Ty, 0), "prng_tap");
      Res = Ctx.B.CreateXor(Shl, Tap, "prng_b32");
    }
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }

  // ---- F32 scalar math / rounding ----
  case CanonicalOp::V_RCP_IFLAG_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Value *R = Ctx.B.CreateFDiv(ConstantFP::get(Ctx.F32Ty, 1.0), S, "rcp");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_RCP_F32:
  case CanonicalOp::V_S_RCP_F32: {
    if (Di.CanonOp == CanonicalOp::V_S_RCP_F32 &&
        !requireDefaultPseudoScalarOutputMods(Di, Hr))
      return Hr;
    if (Di.CanonOp == CanonicalOp::V_RCP_F32 &&
        !requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *RcpFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_rcp, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(RcpFn, {S}, "rcp"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_EXP_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *Exp2Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_exp2, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(Exp2Fn, {S}, "exp"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_S_EXP_F32: {
    if (!requireDefaultPseudoScalarOutputMods(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *Exp2Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_exp2, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(Exp2Fn, {S}, "s_exp"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_LOG_F32:
  case CanonicalOp::V_S_LOG_F32: {
    if (Di.CanonOp == CanonicalOp::V_S_LOG_F32 &&
        !requireDefaultPseudoScalarOutputMods(Di, Hr))
      return Hr;
    if (Di.CanonOp == CanonicalOp::V_LOG_F32 &&
        !requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *Log2Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_log, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(Log2Fn, {S}, "log"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_TANH_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    if (Ctx.TargetIsa.HasTanhInsts) {
      Function *TanhFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_tanh, {Ctx.F32Ty});
      Ctx.writeReg32(Op.dst(),
                     Ctx.B.CreateBitCast(
                         Ctx.B.CreateCall(TanhFn, {S}, "tanh"),
                         Ctx.I32Ty));
      Hr.Handled = true;
      return Hr;
    }

    // Targets without a native tanh instruction lower through OCML rather than
    // a local arithmetic approximation. Native-capable targets keep the
    // intrinsic path above.
    FunctionCallee TanhFn = declareOCMLTanhF32(Ctx.M);
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(TanhFn, {S}, "ocml.tanh"),
                       Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_LDEXP_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S0 = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Value *S1 = Op.src(1);
    Function *LdexpFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ldexp, {Ctx.F32Ty, Ctx.I32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(LdexpFn, {S0, S1}, "ldexp"),
                       Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_SQRT_F32:
  case CanonicalOp::V_S_SQRT_F32: {
    if (Di.CanonOp == CanonicalOp::V_S_SQRT_F32 &&
        !requireDefaultPseudoScalarOutputMods(Di, Hr))
      return Hr;
    if (Di.CanonOp == CanonicalOp::V_SQRT_F32 &&
        !requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *SqrtFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_sqrt, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(SqrtFn, {S}, "sqrt"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_RSQ_F32:
  case CanonicalOp::V_S_RSQ_F32: {
    if (Di.CanonOp == CanonicalOp::V_S_RSQ_F32 &&
        !requireDefaultPseudoScalarOutputMods(Di, Hr))
      return Hr;
    if (Di.CanonOp == CanonicalOp::V_RSQ_F32 &&
        !requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *RsqFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_rsq, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(RsqFn, {S}, "rsq"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_FREXP_EXP_I32_F64: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (Di.HasDpp) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "V_FREXP_EXP_I32_F64 DPP has mixed source/destination widths "
          "(f64 source, i32 destination); inactive lane preservation must "
          "be modeled as 32-bit old-destination semantics, not the generic "
          "same-width DPP source wrapper");
      return Hr;
    }
    Value *S = Ctx.B.CreateBitCast(Op.src64(0), Ctx.F64Ty);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_frexp_exp, {Ctx.I32Ty, Ctx.F64Ty});
    Value *Exp = Ctx.B.CreateCall(Fn, {S}, "frexp_exp");
    Ctx.writeReg32(Op.dst(), Exp);
    Hr.Handled = true;
    return Hr;
  }

  // bf16 transcendentals: widen to f32, dispatch the f32 intrinsic, narrow
  // back. v_tanh_bf16 has no f32 hardware, so it routes through OCML.
  case CanonicalOp::V_RCP_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(
            Ctx, Op, Hr, getF32Intrinsic(Ctx, Intrinsic::amdgcn_rcp),
            "rcp_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_RSQ_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(
            Ctx, Op, Hr, getF32Intrinsic(Ctx, Intrinsic::amdgcn_rsq),
            "rsq_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_SQRT_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(
            Ctx, Op, Hr, getF32Intrinsic(Ctx, Intrinsic::amdgcn_sqrt),
            "sqrt_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_LOG_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(
            Ctx, Op, Hr, getF32Intrinsic(Ctx, Intrinsic::amdgcn_log),
            "log_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_EXP_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(
            Ctx, Op, Hr, getF32Intrinsic(Ctx, Intrinsic::amdgcn_exp2),
            "exp_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_COS_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(
            Ctx, Op, Hr, getF32Intrinsic(Ctx, Intrinsic::amdgcn_cos),
            "cos_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_SIN_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(
            Ctx, Op, Hr, getF32Intrinsic(Ctx, Intrinsic::amdgcn_sin),
            "sin_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_TANH_BF16: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    if (failed(emitBF16UnaryViaF32Callee(Ctx, Op, Hr, declareOCMLTanhF32(Ctx.M),
                                   "tanh_bf16")))
      return Hr;
    Hr.Handled = true;
    return Hr;
  }

  case CanonicalOp::V_FLOOR_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *FloorFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::floor, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(FloorFn, {S}, "floor"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CEIL_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *CeilFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ceil, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(CeilFn, {S}, "ceil"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CEIL_F64: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.src64(0), Ctx.F64Ty);
    S = Op.applyMods(0, S);
    Function *CeilFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::ceil, {Ctx.F64Ty});
    Ctx.writeReg64(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(CeilFn, {S}, "ceil"), Ctx.I64Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_I32_F64: {
    // Saturates out-of-range f64 to INT_MIN/INT_MAX and maps NaN to 0, so
    // lower to fptosi.sat rather than plain fptosi (which is UB on overflow).
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.src64(0), Ctx.F64Ty);
    S = Op.applyMods(0, S);
    Function *SatFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::fptosi_sat, {Ctx.I32Ty, Ctx.F64Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(SatFn, {S}, "cvt_i32_f64"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_TRUNC_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *TruncFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::trunc, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(TruncFn, {S}, "trunc"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_RNDNE_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *RoundEvenFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::roundeven, {Ctx.F32Ty});
    Ctx.writeReg32(
        Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(RoundEvenFn, {S}, "rndne"),
                            Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_FREXP_MANT_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_frexp_mant, {Ctx.F32Ty});
    Ctx.writeReg32(
        Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(Fn, {S}, "frexp_mant"),
                            Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_FRACT_F32: {
    if (!requireDefaultOutputModsIfPresent(Di, Hr))
      return Hr;
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *FloorFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::floor, {Ctx.F32Ty});
    Value *Fl = Ctx.B.CreateCall(FloorFn, {S}, "floor");
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateFSub(S, Fl, "fract"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  default:
    break;
  }
  return Hr;
}

} // namespace COMGR::hotswap
