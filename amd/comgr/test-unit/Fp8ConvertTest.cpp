//===- Fp8ConvertTest.cpp - fp8 OCP<->FNUZ byte converter unit tests ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Unit tests for the per-byte-lane fp8/bf8 OCP<->FNUZ re-encoders in
// `fp8-convert.{h,cpp}`.  gfx940/941/942 (CDNA3) fp8 hardware reads bytes as
// FNUZ (E4M3FNUZ bias 8, E5M2FNUZ bias 16); gfx950/gfx12 use OCP (E4M3FN bias
// 7, E5M2 bias 15).  The raiser re-encodes at every gfx942 fp8 boundary, so a
// bit error in these converters silently corrupts every fp8 operand.
//
// The converters are IR emitters (`Value *convertX(HotswapIRBuilder&, Value*)`
// over a `<N x i32>` where each lane holds a byte).  HotswapIRBuilder carries a
// ConstantFolder, so a ConstantVector input folds the whole converter to a
// ConstantVector output with no target/codegen.  Each test feeds all 256 byte
// values as one `<256 x i32>` splat-free constant, runs the real emitter, and
// reads each folded lane back -- this exercises the shipping IR, not a C++
// mirror that could drift from it.
//
// Each converted byte is checked against an oracle built on llvm::APFloat,
// which shares no code with the converters: APFloat does the format conversion
// and its round-half-to-even, and the oracle layers on only the three policy
// classes fp8-convert.h documents (NaN/Inf -> canonical target NaN, finite
// overflow -> saturate, FNUZ has no -0).  A hand-written oracle would have to
// restate the rounding rules the converters implement, and a subtle
// misunderstanding would then be baked into both sides; APFloat cannot drift
// with them.  Edge classes are additionally asserted explicitly: NaN, Inf
// (E5M2), subnormals, E4M3 OCP (240,448] saturation, and FNUZ->OCP
// round-half-to-even.
//
// Two callers of those byte converters are covered here too, because lit
// cannot reach either:
//   * `convertFp8Dword` -- lit fixtures only count its truncs, so byte ORDER
//     (the shift vector) is pinned here instead.
//   * `encodeF32PairToOcpFp8` -- it calls the target's `cvt.pk.{fp8,bf8}.f32`,
//     which does not constant-fold, so its thresholds, sign re-application and
//     Top/NaN bytes are only textually echoed by lit CHECKs.  The E4M3 464 tie
//     (rounds DOWN to 448) versus E5M2's inclusive 61440 tie (rounds to Inf)
//     is the asymmetry that makes `ogt` vs `oge` deliberate rather than a typo.

#include "hotswap/fp8-convert.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <limits>

using namespace llvm;
using namespace COMGR::hotswap;

namespace {

// Run `Conv` over all 256 byte values at once and return the 256 folded
// results.  Builds a `<256 x i32>` constant [0, 1, ..., 255], calls the real
// emitter (which the ConstantFolder collapses to a constant), and extracts each
// lane.  Fails the test if any lane did not fold to a constant.
// A throwaway module to emit into. The converters are IR emitters, so a test
// needs somewhere to put the instructions even though every one of them folds
// away before it is inserted.
struct FoldHarness {
  LLVMContext Ctx;
  Module M{"fp8convtest", Ctx};
  HotswapIRBuilder B;

  FoldHarness()
      : B(BasicBlock::Create(
            Ctx, "entry",
            Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                             GlobalValue::ExternalLinkage, "f", &M))) {
    // Little-endian, as AMDGPU is: the dword path's `<4 x i8>` <-> i32
    // bitcasts only have a defined byte order under one.
    M.setDataLayout("e");
  }
};

// Reduce \p V to a plain constant. Two things the IRBuilder's ConstantFolder
// leaves behind need a DataLayout it does not carry: instructions whose
// operands only became constant later, and the `<4 x i8>` -> i32 bitcast at
// the end of the dword path, which it folds to a ConstantExpr rather than a
// ConstantInt.
Value *foldBlockTo(FoldHarness &H, Value *V) {
  const DataLayout &DL = H.M.getDataLayout();
  WeakTrackingVH Out(V);
  BasicBlock *BB = H.B.GetInsertBlock();
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (Instruction &I : make_early_inc_range(*BB))
      if (Constant *C = ConstantFoldInstruction(&I, DL)) {
        I.replaceAllUsesWith(C);
        I.eraseFromParent();
        Changed = true;
      }
  }
  if (auto *C = dyn_cast_or_null<Constant>(static_cast<Value *>(Out)))
    return ConstantFoldConstant(C, DL);
  return Out;
}

std::array<uint8_t, 256>
runConverter(llvm::function_ref<Value *(HotswapIRBuilder &, Value *)> Conv) {
  FoldHarness H;
  HotswapIRBuilder &B = H.B;

  Type *I32Ty = B.getInt32Ty();
  SmallVector<Constant *, 256> Lanes;
  for (unsigned I = 0; I < 256; ++I)
    Lanes.push_back(ConstantInt::get(I32Ty, I));
  Value *In = ConstantVector::get(Lanes);

  Value *Out = Conv(B, In);
  auto *OutC = dyn_cast<Constant>(Out);
  EXPECT_TRUE(OutC != nullptr) << "converter did not constant-fold";

  std::array<uint8_t, 256> Result{};
  for (unsigned I = 0; I < 256; ++I) {
    Constant *Lane = OutC ? OutC->getAggregateElement(I) : nullptr;
    auto *CI = dyn_cast_or_null<ConstantInt>(Lane);
    EXPECT_TRUE(CI != nullptr) << "lane " << I << " did not fold";
    Result[I] = CI ? static_cast<uint8_t>(CI->getZExtValue() & 0xFF) : 0;
  }
  return Result;
}

// --- APFloat-based oracle (shares no code with the converters) ---

// Convert one byte from \p From to \p To the way fp8-convert.h specifies.
// APFloat performs the format conversion and the round-half-to-even; only the
// documented policy classes are applied on top of it.
uint8_t oracleConvert(uint8_t Byte, const fltSemantics &From,
                      const fltSemantics &To) {
  const bool ToFnuz =
      &To == &APFloat::Float8E4M3FNUZ() || &To == &APFloat::Float8E5M2FNUZ();
  const uint8_t TargetNaN = ToFnuz ? 0x80 : 0x7F;

  APFloat V(From, APInt(8, Byte));
  // Neither target format has Inf, so Inf joins NaN in mapping to the target's
  // canonical NaN rather than saturating.
  if (V.isNaN() || V.isInfinity())
    return TargetNaN;

  const bool Neg = V.isNegative();
  bool LosesInfo = false;
  V.convert(To, APFloat::rmNearestTiesToEven, &LosesInfo);
  // A finite input that overflows the target comes back as NaN (no Inf to
  // round to); the converters saturate to the target max instead.
  if (V.isNaN() || V.isInfinity())
    V = APFloat::getLargest(To, Neg);
  if (ToFnuz && V.isZero())
    return 0x00; // FNUZ has no -0
  return static_cast<uint8_t>(V.bitcastToAPInt().getZExtValue());
}

// Run every byte through `Conv` and compare against the APFloat oracle.
// `std::hex` does not survive gtest's Message stream, so format explicitly --
// a converter bug reports bit patterns, and decimal would be unreadable.
void checkExhaustive(
    const char *Label,
    llvm::function_ref<Value *(HotswapIRBuilder &, Value *)> Conv,
    const fltSemantics &From, const fltSemantics &To) {
  auto Hex = [](unsigned V) {
    return "0x" + utohexstr(V, /*LowerCase=*/false, /*Width=*/2);
  };
  auto Got = runConverter(Conv);
  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    uint8_t Want = oracleConvert(static_cast<uint8_t>(Byte), From, To);
    EXPECT_EQ(Got[Byte], Want)
        << Label << " mismatch at byte " << Hex(Byte) << ": got "
        << Hex(Got[Byte]) << " want " << Hex(Want);
  }
}

// --- convertFp8Dword: the splat / shift / repack around the byte lanes ---

uint32_t runDwordConverter(uint32_t Dword, bool IsBf8, bool ToFnuz) {
  FoldHarness H;
  Value *Out = foldBlockTo(
      H, convertFp8Dword(H.B, H.B.getInt32(Dword), IsBf8, ToFnuz));
  auto *CI = dyn_cast_or_null<ConstantInt>(Out);
  EXPECT_TRUE(CI != nullptr) << "convertFp8Dword did not constant-fold";
  return CI ? static_cast<uint32_t>(CI->getZExtValue()) : 0;
}

// Byte i of the result must be byte i of the input, converted. Byte ORDER is
// the only thing the dword path adds over the byte converters, and the chosen
// inputs convert to four distinct bytes in every direction, so any lane
// permutation (e.g. a reversed shift vector) shows up here.
void checkDwordRepack(const char *Label, bool IsBf8, bool ToFnuz,
                      const fltSemantics &From, const fltSemantics &To) {
  for (uint32_t In : {0x01407FC8u, 0x7C03F080u}) {
    uint32_t Got = runDwordConverter(In, IsBf8, ToFnuz);
    for (unsigned I = 0; I < 4; ++I) {
      uint32_t Want = oracleConvert((In >> (8 * I)) & 0xFF, From, To);
      EXPECT_EQ((Got >> (8 * I)) & 0xFFu, Want)
          << Label << " dword 0x" << utohexstr(In, false, 8) << " byte " << I
          << ": whole result 0x" << utohexstr(Got, false, 8);
    }
  }
}

// --- encodeF32PairToOcpFp8 ---

// Fold the encoder down to its packed byte pair.  It calls the target's
// `llvm.amdgcn.cvt.pk.{fp8,bf8}.f32`, which has no constant folding -- the
// reason these numeric edges cannot be pinned in lit.  Its operands are
// constants by construction, so the call is replaced by an APFloat model of
// the FNUZ encode the hardware performs, and the surrounding IR (the
// thresholds, the sign re-application, the Top/NaN bytes -- everything under
// test) is then folded away for real.
uint32_t foldEncodePair(double S0, double S1, bool IsBf8) {
  FoldHarness H;
  HotswapIRBuilder &B = H.B;
  Type *F32Ty = B.getFloatTy();
  Function *CvtFn = Intrinsic::getOrInsertDeclaration(
      &H.M, IsBf8 ? Intrinsic::amdgcn_cvt_pk_bf8_f32
                  : Intrinsic::amdgcn_cvt_pk_fp8_f32);
  WeakTrackingVH Out(encodeF32PairToOcpFp8(
      B, CvtFn, ConstantFP::get(F32Ty, S0), ConstantFP::get(F32Ty, S1), IsBf8));

  BasicBlock *BB = B.GetInsertBlock();
  CallInst *Raw = nullptr;
  for (Instruction &I : *BB)
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (CI->getCalledFunction() == CvtFn)
        Raw = CI;
  EXPECT_TRUE(Raw != nullptr) << "no hardware encode call was emitted";
  if (!Raw)
    return 0;

  auto Fnuz = [&](Value *V) -> uint32_t {
    APFloat X = cast<ConstantFP>(V)->getValueAPF();
    bool Lost = false;
    X.convert(IsBf8 ? APFloat::Float8E5M2FNUZ() : APFloat::Float8E4M3FNUZ(),
              APFloat::rmNearestTiesToEven, &Lost);
    return X.bitcastToAPInt().getZExtValue() & 0xFF;
  };
  Raw->replaceAllUsesWith(ConstantInt::get(
      B.getInt32Ty(),
      Fnuz(Raw->getArgOperand(0)) | (Fnuz(Raw->getArgOperand(1)) << 8)));

  auto *CI = dyn_cast_or_null<ConstantInt>(foldBlockTo(H, Out));
  EXPECT_TRUE(CI != nullptr) << "encoder did not fold to a constant";
  return CI ? static_cast<uint32_t>(CI->getZExtValue()) : 0;
}

uint32_t loByte(uint32_t Pk) { return Pk & 0xFF; }
uint32_t hiByte(uint32_t Pk) { return (Pk >> 8) & 0xFF; }

const double PosInf = std::numeric_limits<double>::infinity();
const double QNaN = std::numeric_limits<double>::quiet_NaN();

} // namespace

TEST(Fp8Convert, OcpE4M3ToFnuzExhaustive) {
  checkExhaustive("OCP E4M3->FNUZ", convertOcpE4M3ToFnuz,
                  APFloat::Float8E4M3FN(), APFloat::Float8E4M3FNUZ());
}

TEST(Fp8Convert, OcpE5M2ToFnuzExhaustive) {
  checkExhaustive("OCP E5M2->FNUZ", convertOcpE5M2ToFnuz, APFloat::Float8E5M2(),
                  APFloat::Float8E5M2FNUZ());
}

TEST(Fp8Convert, FnuzE4M3ToOcpExhaustive) {
  checkExhaustive("FNUZ E4M3->OCP", convertFnuzE4M3ToOcp,
                  APFloat::Float8E4M3FNUZ(), APFloat::Float8E4M3FN());
}

TEST(Fp8Convert, FnuzE5M2ToOcpExhaustive) {
  checkExhaustive("FNUZ E5M2->OCP", convertFnuzE5M2ToOcp,
                  APFloat::Float8E5M2FNUZ(), APFloat::Float8E5M2());
}

// Decode every byte through the real emitter and compare against APFloat's
// own fp8 -> f32 conversion, which is exact (widening) and shares no code with
// the decoder.  NaN is compared by class, since the payload is unspecified.
void checkDecode(const char *Label, const fltSemantics &From, bool IsBf8,
                 Fp8Format Fmt) {
  FoldHarness H;
  HotswapIRBuilder &B = H.B;

  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    Value *Out = decodeFp8ByteToF32(B, B.getInt32(Byte), IsBf8, Fmt);
    auto *CF = dyn_cast<ConstantFP>(Out);
    ASSERT_TRUE(CF != nullptr)
        << Label << " byte " << Byte << " did not constant-fold";
    APFloat Got = CF->getValueAPF();

    APFloat Want(From, APInt(8, Byte));
    bool LosesInfo = false;
    Want.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                 &LosesInfo);
    EXPECT_FALSE(LosesInfo) << Label << ": fp8 -> f32 must be exact";

    if (Want.isNaN()) {
      EXPECT_TRUE(Got.isNaN()) << Label << " byte 0x"
                               << utohexstr(Byte, false, 2) << " should be NaN";
      continue;
    }
    // bitwiseIsEqual distinguishes +0 from -0, which is the point for OCP.
    EXPECT_TRUE(Got.bitwiseIsEqual(Want))
        << Label << " byte 0x" << utohexstr(Byte, false, 2) << ": got "
        << Got.convertToFloat() << " want " << Want.convertToFloat();
  }
}

TEST(Fp8Convert, DecodeOcpE4M3ToF32Exhaustive) {
  checkDecode("OCP E4M3 decode", APFloat::Float8E4M3FN(), /*IsBf8=*/false,
              Fp8Format::OCP);
}

TEST(Fp8Convert, DecodeOcpE5M2ToF32Exhaustive) {
  checkDecode("OCP E5M2 decode", APFloat::Float8E5M2(), /*IsBf8=*/true,
              Fp8Format::OCP);
}

TEST(Fp8Convert, DecodeFnuzE4M3ToF32Exhaustive) {
  checkDecode("FNUZ E4M3 decode", APFloat::Float8E4M3FNUZ(), /*IsBf8=*/false,
              Fp8Format::FNUZ);
}

TEST(Fp8Convert, DecodeFnuzE5M2ToF32Exhaustive) {
  checkDecode("FNUZ E5M2 decode", APFloat::Float8E5M2FNUZ(), /*IsBf8=*/true,
              Fp8Format::FNUZ);
}

// Explicit edge-class pins (independent of the oracle loop above) so a
// regression names the exact special case it broke.
TEST(Fp8Convert, EdgeClasses) {
  auto E4ToFnuz = runConverter(convertOcpE4M3ToFnuz);
  auto E5ToFnuz = runConverter(convertOcpE5M2ToFnuz);
  auto E4ToOcp = runConverter(convertFnuzE4M3ToOcp);

  // OCP E4M3 NaN (S.1111.111) -> FNUZ NaN 0x80.
  EXPECT_EQ(E4ToFnuz[0x7F], 0x80u);
  EXPECT_EQ(E4ToFnuz[0xFF], 0x80u);
  // OCP E4M3 (240,448] saturates to FNUZ max finite 240 (0x7F), sign kept.
  EXPECT_EQ(E4ToFnuz[0x78], 0x7Fu); // exp=15,mant=0 -> saturate
  EXPECT_EQ(E4ToFnuz[0xF8], 0xFFu); // negative saturate
  // +/-0 both map to FNUZ +0.
  EXPECT_EQ(E4ToFnuz[0x00], 0x00u);
  EXPECT_EQ(E4ToFnuz[0x80], 0x00u);

  // FNUZ has no Inf, so OCP E5M2 +/-Inf joins NaN at 0x80 rather than
  // saturating -- matching APFloat and the gfx942 f32->bf8 encode.
  EXPECT_EQ(E5ToFnuz[0x7C], 0x80u); // +Inf
  EXPECT_EQ(E5ToFnuz[0xFC], 0x80u); // -Inf
  EXPECT_EQ(E5ToFnuz[0x7D], 0x80u); // E5M2 NaN

  // FNUZ E4M3 NaN 0x80 -> OCP +0x7F.
  EXPECT_EQ(E4ToOcp[0x80], 0x7Fu);
  // FNUZ +0 -> OCP +0.
  EXPECT_EQ(E4ToOcp[0x00], 0x00u);

  auto E5ToOcp = runConverter(convertFnuzE5M2ToOcp);
  EXPECT_EQ(E5ToOcp[0x80], 0x7Fu); // FNUZ NaN -> OCP NaN
  EXPECT_EQ(E5ToOcp[0x00], 0x00u);
  EXPECT_EQ(E5ToOcp[0x7F], 0x7Bu); // FNUZ max finite -> OCP 57344
  EXPECT_EQ(E5ToOcp[0xFF], 0xFBu);
  EXPECT_EQ(E5ToOcp[0x52], 0x4Eu); // ordinary normal: stored exponent -1

  // FNUZ -> OCP subnormals round half to even. FNUZ's grid below OCP's
  // smallest normal is twice as fine, so every odd step is an exact tie.
  EXPECT_EQ(E4ToOcp[0x01], 0x00u); // 0.5 -> 0
  EXPECT_EQ(E4ToOcp[0x03], 0x02u); // 1.5 -> 2
  EXPECT_EQ(E4ToOcp[0x05], 0x02u); // 2.5 -> 2
  EXPECT_EQ(E4ToOcp[0x07], 0x04u); // 3.5 -> 4
  EXPECT_EQ(E5ToOcp[0x01], 0x00u);
  EXPECT_EQ(E5ToOcp[0x03], 0x02u);
  EXPECT_EQ(E5ToOcp[0x07], 0x04u);
}

// The dword path only splats, shifts, converts and repacks; byte order is the
// one thing it can get wrong that the byte converters cannot.
TEST(Fp8Convert, ConvertFp8DwordRepack) {
  checkDwordRepack("OCP E4M3->FNUZ dword", /*IsBf8=*/false, /*ToFnuz=*/true,
                   APFloat::Float8E4M3FN(), APFloat::Float8E4M3FNUZ());
  checkDwordRepack("OCP E5M2->FNUZ dword", /*IsBf8=*/true, /*ToFnuz=*/true,
                   APFloat::Float8E5M2(), APFloat::Float8E5M2FNUZ());
  checkDwordRepack("FNUZ E4M3->OCP dword", /*IsBf8=*/false, /*ToFnuz=*/false,
                   APFloat::Float8E4M3FNUZ(), APFloat::Float8E4M3FN());
  checkDwordRepack("FNUZ E5M2->OCP dword", /*IsBf8=*/true, /*ToFnuz=*/false,
                   APFloat::Float8E5M2FNUZ(), APFloat::Float8E5M2());
}

// E4M3FN: 464 is the exact round-half-to-even tie and still reaches 448
// (0x7E), so the overflow test is strictly-above (`ogt`); past it -- +/-Inf
// included -- the result is the NaN 0x7F, which E4M3FN writes unsigned.
TEST(Fp8Convert, EncodeF32PairToOcpFp8E4M3Edges) {
  EXPECT_EQ(loByte(foldEncodePair(464.0, 0.0, /*IsBf8=*/false)), 0x7Eu);
  EXPECT_EQ(hiByte(foldEncodePair(0.0, 464.0, false)), 0x7Eu);
  EXPECT_EQ(loByte(foldEncodePair(-464.0, 0.0, false)), 0xFEu);
  EXPECT_EQ(loByte(foldEncodePair(448.0, 0.0, false)), 0x7Eu);

  EXPECT_EQ(loByte(foldEncodePair(480.0, 0.0, false)), 0x7Fu);
  EXPECT_EQ(loByte(foldEncodePair(-480.0, 0.0, false)), 0x7Fu);
  EXPECT_EQ(loByte(foldEncodePair(PosInf, 0.0, false)), 0x7Fu);
  EXPECT_EQ(loByte(foldEncodePair(-PosInf, 0.0, false)), 0x7Fu);
  EXPECT_EQ(loByte(foldEncodePair(QNaN, 0.0, false)), 0x7Fu);
  EXPECT_EQ(hiByte(foldEncodePair(0.0, QNaN, false)), 0x7Fu);

  // FNUZ has no -0, so the sign is re-applied after the hardware encode.
  EXPECT_EQ(loByte(foldEncodePair(-0.0, 0.0, false)), 0x80u);
  EXPECT_EQ(loByte(foldEncodePair(1.0, 0.0, false)), 0x38u);
}

// E5M2 is IEEE-shaped: 61440 is its round-half-to-even tie between 57344 and
// Inf and rounds TO Inf (0x7C), so unlike fp8 the threshold is inclusive
// (`oge`). The asymmetry between the two sides is deliberate, not a typo.
TEST(Fp8Convert, EncodeF32PairToOcpFp8E5M2Edges) {
  EXPECT_EQ(loByte(foldEncodePair(61440.0, 0.0, /*IsBf8=*/true)), 0x7Cu);
  EXPECT_EQ(loByte(foldEncodePair(-61440.0, 0.0, true)), 0xFCu);
  EXPECT_EQ(loByte(foldEncodePair(PosInf, 0.0, true)), 0x7Cu);
  EXPECT_EQ(loByte(foldEncodePair(-PosInf, 0.0, true)), 0xFCu);
  EXPECT_EQ(hiByte(foldEncodePair(0.0, PosInf, true)), 0x7Cu);
  // Unlike E4M3FN's, E5M2's NaN is a distinct encoding from its Top byte.
  EXPECT_EQ(loByte(foldEncodePair(QNaN, 0.0, true)), 0x7Eu);

  // Below the tie the hardware encode still decides: 57344 is E5M2's max
  // finite (0x7B).
  EXPECT_EQ(loByte(foldEncodePair(57344.0, 0.0, true)), 0x7Bu);
  EXPECT_EQ(loByte(foldEncodePair(-0.0, 0.0, true)), 0x80u);
  EXPECT_EQ(loByte(foldEncodePair(1.0, 0.0, true)), 0x3Cu);
}
