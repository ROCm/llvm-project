//===- fp8-convert.cpp - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fp8-convert.h"
#include "isa-profile.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"

#include <cmath>

using namespace llvm;

namespace COMGR::hotswap {

Fp8Format fp8FormatOf(const ISAProfile &P) { return P.Fp8Fmt; }

Expected<Fp8Reencode> fp8Reencode(const ISAProfile &Src, const ISAProfile &Tgt,
                                  Fp8Dir Dir) {
  Fp8Format S = fp8FormatOf(Src), T = fp8FormatOf(Tgt);
  auto Refuse = [](StringRef Side, Fp8Format F) {
    return createStringError(
        Twine(Side) + " ISA " +
        (F == Fp8Format::None ? "has no fp8/bf8 hardware"
                              : "has fp8/bf8 hardware of an unclassified "
                                "format") +
        ", so an fp8/bf8 value cannot cross this boundary");
  };
  if (S != Fp8Format::OCP && S != Fp8Format::FNUZ)
    return Refuse("source", S);
  if (T != Fp8Format::OCP && T != Fp8Format::FNUZ)
    return Refuse("target", T);
  if (S == T)
    return Fp8Reencode::None;
  return (Dir == Fp8Dir::SrcToTgt ? T : S) == Fp8Format::FNUZ
             ? Fp8Reencode::ToFnuz
             : Fp8Reencode::ToOcp;
}

namespace {

// Helper to build an N-lane i32 splat constant matching `Bytes`'s vector type.
struct ByteVecHelper {
  HotswapIRBuilder &B;
  unsigned N;
  IntegerType *I32Ty;
  ByteVecHelper(HotswapIRBuilder &B, Value *Bytes) : B(B) {
    auto *VecTy = cast<FixedVectorType>(Bytes->getType());
    N = VecTy->getNumElements();
    I32Ty = cast<IntegerType>(VecTy->getElementType());
  }
  Constant *splat(uint64_t V) const {
    return ConstantVector::getSplat(ElementCount::getFixed(N),
                                    ConstantInt::get(I32Ty, V));
  }
};

// OCP -> FNUZ, mantissa width \p M (3 = E4M3, 2 = E5M2).
//   normals: stored exponent +1 (mantissa identical, no rounding);
//   subnormals: byte = sign | (mant << 1) (exact); +/-0 -> +0.
// The top exponent is the one class the two widths do not share: OCP E4M3FN's
// is finite above FNUZ's 240 max (saturate) apart from its lone NaN, while OCP
// E5M2's is Inf or NaN and FNUZ has neither, so all of it becomes 0x80. That
// is what APFloat's Float8E5M2 -> Float8E5M2FNUZ yields, and what gfx942's own
// f32->bf8 encode yields for Inf under the default MODE.FP16_OVFL=0. No finite
// OCP E5M2 value overflows FNUZ E5M2 (both max at 57344).
Value *ocpToFnuz(HotswapIRBuilder &B, Value *Bytes, unsigned M,
                 const Twine &Name) {
  ByteVecHelper H(B, Bytes);
  const uint64_t EMask = (1u << (7 - M)) - 1, MMask = (1u << M) - 1;
  Value *Sign = B.CreateShl(
      B.CreateAnd(B.CreateLShr(Bytes, H.splat(7)), H.splat(1)), H.splat(7));
  Value *Exp = B.CreateAnd(B.CreateLShr(Bytes, H.splat(M)), H.splat(EMask));
  Value *Mant = B.CreateAnd(Bytes, H.splat(MMask));
  Value *Norm = B.CreateOr(
      Sign,
      B.CreateOr(B.CreateShl(B.CreateAdd(Exp, H.splat(1)), H.splat(M)), Mant));
  Value *Sub = B.CreateSelect(B.CreateICmpEQ(Mant, H.splat(0)), H.splat(0),
                              B.CreateOr(Sign, B.CreateShl(Mant, H.splat(1))));
  Value *Top = H.splat(0x80);
  if (M == 3)
    Top = B.CreateSelect(B.CreateICmpEQ(Mant, H.splat(MMask)), H.splat(0x80),
                         B.CreateOr(Sign, H.splat(0x7F)));
  Value *R = B.CreateSelect(B.CreateICmpEQ(Exp, H.splat(EMask)), Top, Norm);
  return B.CreateSelect(B.CreateICmpEQ(Exp, H.splat(0)), Sub, R, Name);
}

// FNUZ -> OCP, mantissa width \p M.
//   normals (exp>=2): stored exponent -1 (exact);
//   exp<=1: OCP subnormal, mant' = round-half-to-even(N/2) with
//           N = (exp==1 ? 1<<M : 0) + mant; 0x80 (NaN) -> 0x7F.
// FNUZ range is a subset of OCP except its finer subnormals, which round.
Value *fnuzToOcp(HotswapIRBuilder &B, Value *Bytes, unsigned M,
                 const Twine &Name) {
  ByteVecHelper H(B, Bytes);
  const uint64_t EMask = (1u << (7 - M)) - 1, MMask = (1u << M) - 1;
  Value *Sign = B.CreateShl(
      B.CreateAnd(B.CreateLShr(Bytes, H.splat(7)), H.splat(1)), H.splat(7));
  Value *Exp = B.CreateAnd(B.CreateLShr(Bytes, H.splat(M)), H.splat(EMask));
  Value *Mant = B.CreateAnd(Bytes, H.splat(MMask));
  Value *Norm = B.CreateOr(
      Sign,
      B.CreateOr(B.CreateShl(B.CreateSub(Exp, H.splat(1)), H.splat(M)), Mant));
  Value *NVal = B.CreateAdd(B.CreateSelect(B.CreateICmpEQ(Exp, H.splat(1)),
                                           H.splat(1u << M), H.splat(0)),
                            Mant);
  Value *Rne = B.CreateAdd(
      B.CreateLShr(NVal, H.splat(1)),
      B.CreateSelect(B.CreateICmpEQ(B.CreateAnd(NVal, H.splat(3)), H.splat(3)),
                     H.splat(1), H.splat(0)));
  Value *Sub = B.CreateOr(Sign, Rne);
  Value *R = B.CreateSelect(B.CreateICmpUGE(Exp, H.splat(2)), Norm, Sub);
  return B.CreateSelect(B.CreateICmpEQ(Bytes, H.splat(0x80)), H.splat(0x7F), R,
                        Name);
}

} // namespace

// OCP E4M3FN (bias 7) -> FNUZ E4M3 (bias 8).
Value *convertOcpE4M3ToFnuz(HotswapIRBuilder &B, Value *Bytes) {
  return ocpToFnuz(B, Bytes, /*M=*/3, "e4m3_fnuz");
}

// OCP E5M2 (bias 15) -> FNUZ E5M2 (bias 16).
Value *convertOcpE5M2ToFnuz(HotswapIRBuilder &B, Value *Bytes) {
  return ocpToFnuz(B, Bytes, /*M=*/2, "e5m2_fnuz");
}

// FNUZ E4M3 (bias 8) -> OCP E4M3FN (bias 7).
Value *convertFnuzE4M3ToOcp(HotswapIRBuilder &B, Value *Bytes) {
  return fnuzToOcp(B, Bytes, /*M=*/3, "e4m3_ocp");
}

// FNUZ E5M2 (bias 16) -> OCP E5M2 (bias 15).
Value *convertFnuzE5M2ToOcp(HotswapIRBuilder &B, Value *Bytes) {
  return fnuzToOcp(B, Bytes, /*M=*/2, "e5m2_ocp");
}

Value *convertFp8Dword(HotswapIRBuilder &B, Value *Dword, bool IsBf8,
                       bool ToFnuz) {
  auto *I32Ty = B.getInt32Ty();
  auto *Vec4I8 = FixedVectorType::get(B.getInt8Ty(), 4);
  Constant *Shifts = ConstantVector::get(
      {ConstantInt::get(I32Ty, 0), ConstantInt::get(I32Ty, 8),
       ConstantInt::get(I32Ty, 16), ConstantInt::get(I32Ty, 24)});
  Constant *ByteMask = ConstantVector::getSplat(ElementCount::getFixed(4),
                                                ConstantInt::get(I32Ty, 0xFF));
  Value *Splat = B.CreateVectorSplat(4, Dword, "fp8_splat");
  Value *Bytes =
      B.CreateAnd(B.CreateLShr(Splat, Shifts), ByteMask, "fp8_bytes");
  Value *Conv = ToFnuz ? (IsBf8 ? convertOcpE5M2ToFnuz(B, Bytes)
                                : convertOcpE4M3ToFnuz(B, Bytes))
                       : (IsBf8 ? convertFnuzE5M2ToOcp(B, Bytes)
                                : convertFnuzE4M3ToOcp(B, Bytes));
  Value *ConvBytes = B.CreateTrunc(Conv, Vec4I8, "fp8_conv_bytes");
  return B.CreateBitCast(ConvBytes, I32Ty, "fp8_conv_dw");
}

void convertFp8DwordsInPlace(HotswapIRBuilder &B,
                             SmallVectorImpl<Value *> &Dwords, bool IsBf8,
                             bool ToFnuz) {
  for (Value *&Dw : Dwords)
    Dw = convertFp8Dword(B, Dw, IsBf8, ToFnuz);
}

Value *decodeFp8ByteToF32(HotswapIRBuilder &B, Value *Byte, bool IsBf8,
                          Fp8Format Fmt) {
  if (Fmt != Fp8Format::OCP && Fmt != Fp8Format::FNUZ)
    return nullptr;
  const bool IsFnuz = Fmt == Fp8Format::FNUZ;
  Type *F32Ty = B.getFloatTy();
  IntegerType *I32Ty = B.getInt32Ty();
  const unsigned M = IsBf8 ? 2 : 3;
  const unsigned EMask = IsBf8 ? 0x1F : 0xF;
  const unsigned MMask = (1u << M) - 1;
  const int Bias = (IsBf8 ? 15 : 7) + (IsFnuz ? 1 : 0);
  auto I32 = [&](uint64_t V) { return ConstantInt::get(I32Ty, V); };

  Value *Sign =
      B.CreateShl(B.CreateAnd(B.CreateLShr(Byte, I32(7)), I32(1)), I32(31));
  Value *Exp = B.CreateAnd(B.CreateLShr(Byte, I32(M)), I32(EMask));
  Value *Mant = B.CreateAnd(Byte, I32(MMask));

  // Normal: the fp8 fields map onto f32's directly, only the bias and the
  // mantissa's left-alignment change.
  Value *NormBits =
      B.CreateOr(B.CreateShl(B.CreateAdd(Exp, I32(127 - Bias)), I32(23)),
                 B.CreateShl(Mant, I32(23 - M)));
  // Subnormal: mant * 2^(1 - Bias - M), exact as a single f32 multiply.  Even
  // the smallest fp8 subnormal is far above f32's subnormal range, so this
  // never depends on the denormal mode.
  Value *Sub =
      B.CreateFMul(B.CreateUIToFP(Mant, F32Ty),
                   ConstantFP::get(F32Ty, std::ldexp(1.0, 1 - Bias - int(M))));
  Value *Mag = B.CreateSelect(B.CreateICmpEQ(Exp, I32(0)), Sub,
                              B.CreateBitCast(NormBits, F32Ty));
  Value *Val =
      B.CreateBitCast(B.CreateOr(B.CreateBitCast(Mag, I32Ty), Sign), F32Ty);

  // Which encodings are non-finite is the one thing the three formats do not
  // agree on: FNUZ has the single NaN 0x80 and an ordinary top exponent; OCP
  // E4M3FN reserves only S.1111.111, so exp==15 still reaches 448; OCP E5M2
  // is IEEE-shaped, so its top exponent is Inf or NaN.
  Value *TopExp = B.CreateICmpEQ(Exp, I32(EMask));
  Value *IsNaN;
  if (IsFnuz)
    IsNaN = B.CreateICmpEQ(Byte, I32(0x80));
  else if (IsBf8)
    IsNaN = B.CreateAnd(TopExp, B.CreateICmpNE(Mant, I32(0)));
  else
    IsNaN = B.CreateAnd(TopExp, B.CreateICmpEQ(Mant, I32(MMask)));
  if (!IsFnuz && IsBf8)
    Val = B.CreateSelect(
        TopExp, B.CreateBitCast(B.CreateOr(I32(0x7F800000), Sign), F32Ty), Val);
  return B.CreateSelect(IsNaN, ConstantFP::getQNaN(F32Ty), Val,
                        Twine(IsBf8 ? "bf8" : "fp8") + "_dec_" +
                            (IsFnuz ? "fnuz" : "ocp"));
}

Value *encodeF32PairToOcpFp8(HotswapIRBuilder &B, Function *CvtFn, Value *S0,
                             Value *S1, bool IsBf8) {
  Type *F32Ty = B.getFloatTy();
  IntegerType *I32Ty = B.getInt32Ty();
  auto I32 = [&](uint64_t V) { return ConstantInt::get(I32Ty, V); };

  // Past Thresh the value leaves the OCP grid.  E5M2 ties to Inf (0x7C) at
  // 61440, sign preserved.  E4M3FN's 464 is the round-half-to-even tie and
  // still lands on 448, but anything strictly above it -- Inf included -- is
  // NaN (0x7F) under the default MODE.FP16_OVFL=0; the docs write that NaN
  // unsigned, unlike the signed FP16_OVFL=1 clamp.  Inside Thresh the halved
  // value is always within the target's FNUZ range, so the raw encoder never
  // overflows and never consults MODE.FP16_OVFL itself.
  const uint32_t TopByte = IsBf8 ? 0x7Cu : 0x7Fu;
  const uint32_t NaNByte = IsBf8 ? 0x7Eu : 0x7Fu;
  Constant *Thresh = ConstantFP::get(F32Ty, IsBf8 ? 61440.0 : 464.0);

  struct Prepped {
    Value *Scaled;
    Value *SignByte;
    Value *IsNaN;
    Value *IsTop;
  };
  auto Prep = [&](Value *X) {
    Prepped P;
    P.SignByte = B.CreateAnd(B.CreateLShr(B.CreateBitCast(X, I32Ty), I32(24)),
                             I32(0x80));
    P.IsNaN = B.CreateFCmpUNO(X, X);
    Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
    P.IsTop =
        IsBf8 ? B.CreateFCmpOGE(Abs, Thresh) : B.CreateFCmpOGT(Abs, Thresh);
    // X needs no sanitising first: whenever IsNaN or IsTop holds, Fixup
    // discards the encoder's byte outright, and the encode cannot trap.
    P.Scaled = B.CreateFMul(X, ConstantFP::get(F32Ty, 0.5));
    return P;
  };
  Prepped P0 = Prep(S0), P1 = Prep(S1);

  Value *Raw = B.CreateCall(
      CvtFn,
      {P0.Scaled, P1.Scaled, I32(0), ConstantInt::getFalse(B.getContext())},
      "pk_fp8_raw");

  auto Fixup = [&](const Prepped &P, unsigned Shift) {
    Value *Byte = B.CreateAnd(B.CreateLShr(Raw, I32(Shift)), I32(0xFF));
    // FNUZ has no -0, so the raw encoder drops the sign of a zero or
    // underflowed result where OCP keeps it. For every other in-range byte
    // the sign bit is already set, so re-applying it is idempotent.
    Byte = B.CreateOr(Byte, P.SignByte);
    Value *Top = IsBf8 ? B.CreateOr(P.SignByte, I32(TopByte))
                       : static_cast<Value *>(I32(TopByte));
    Byte = B.CreateSelect(P.IsTop, Top, Byte);
    return B.CreateSelect(P.IsNaN, I32(NaNByte), Byte);
  };
  return B.CreateOr(Fixup(P0, 0), B.CreateShl(Fixup(P1, 8), I32(8)),
                    "pk_fp8_ocp");
}

} // namespace COMGR::hotswap
