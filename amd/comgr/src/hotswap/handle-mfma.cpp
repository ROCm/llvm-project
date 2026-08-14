//===- handle-mfma.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName
#include "canonical-op.h"
#include "fp8-convert.h"
#include "hotswap/raise-failure.h"
#include "opcode-map.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR::hotswap {

// =========================================================================
// CanonicalOp -> Intrinsic::ID is the one piece of MFMA metadata LLVM does not
// expose as a public MC-level table. The reverse direction (intrinsic ->
// selected instruction) is encoded in the selector patterns but not
// published. Everything else a handler needs -- source element types,
// accumulator vector type, argument count for scaled vs. non-scaled --
// is recoverable from the intrinsic's declared signature via
// `Intrinsic::getType`, so this map carries ID alone.
// =========================================================================
static const DenseMap<CanonicalOp, Intrinsic::ID> &mfmaIntrinsicTable() {
  static const auto *Table = new DenseMap<CanonicalOp, Intrinsic::ID>({
      {CanonicalOp::V_MFMA_F32_16x16x16_F16,
       Intrinsic::amdgcn_mfma_f32_16x16x16f16},
      {CanonicalOp::V_MFMA_F32_32x32x8_F16,
       Intrinsic::amdgcn_mfma_f32_32x32x8f16},
      {CanonicalOp::V_MFMA_F32_16x16x4_F32,
       Intrinsic::amdgcn_mfma_f32_16x16x4f32},
      {CanonicalOp::V_MFMA_F32_32x32x1_F32,
       Intrinsic::amdgcn_mfma_f32_32x32x1f32},
      {CanonicalOp::V_MFMA_F32_32x32x2_F32,
       Intrinsic::amdgcn_mfma_f32_32x32x2f32},
      {CanonicalOp::V_MFMA_F32_4x4x1_F32, Intrinsic::amdgcn_mfma_f32_4x4x1f32},
      {CanonicalOp::V_MFMA_F32_16x16x1_F32,
       Intrinsic::amdgcn_mfma_f32_16x16x1f32},
      {CanonicalOp::V_MFMA_F32_32x32x4_F16,
       Intrinsic::amdgcn_mfma_f32_32x32x4f16},
      {CanonicalOp::V_MFMA_F32_16x16x4_F16,
       Intrinsic::amdgcn_mfma_f32_16x16x4f16},
      {CanonicalOp::V_MFMA_F32_4x4x4_F16, Intrinsic::amdgcn_mfma_f32_4x4x4f16},
      {CanonicalOp::V_MFMA_I32_16x16x32_I8,
       Intrinsic::amdgcn_mfma_i32_16x16x32_i8},
      {CanonicalOp::V_MFMA_I32_32x32x16_I8,
       Intrinsic::amdgcn_mfma_i32_32x32x16_i8},
      {CanonicalOp::V_MFMA_F32_16x16x8_XF32,
       Intrinsic::amdgcn_mfma_f32_16x16x8_xf32},
      {CanonicalOp::V_MFMA_F32_32x32x4_XF32,
       Intrinsic::amdgcn_mfma_f32_32x32x4_xf32},
      {CanonicalOp::V_MFMA_I32_32x32x4_I8,
       Intrinsic::amdgcn_mfma_i32_32x32x4i8},
      {CanonicalOp::V_MFMA_I32_16x16x4_I8,
       Intrinsic::amdgcn_mfma_i32_16x16x4i8},
      {CanonicalOp::V_MFMA_I32_4x4x4_I8, Intrinsic::amdgcn_mfma_i32_4x4x4i8},
      {CanonicalOp::V_MFMA_F32_32x32x2_BF16,
       Intrinsic::amdgcn_mfma_f32_32x32x2bf16},
      {CanonicalOp::V_MFMA_F32_16x16x2_BF16,
       Intrinsic::amdgcn_mfma_f32_16x16x2bf16},
      {CanonicalOp::V_MFMA_F32_4x4x2_BF16,
       Intrinsic::amdgcn_mfma_f32_4x4x2bf16},
      {CanonicalOp::V_MFMA_F32_16x16x16_BF16_1K,
       Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k},
      {CanonicalOp::V_MFMA_F32_32x32x8_BF16_1K,
       Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k},
      {CanonicalOp::V_MFMA_F32_16x16x32_BF16,
       Intrinsic::amdgcn_mfma_f32_16x16x32_bf16},
      {CanonicalOp::V_MFMA_F32_32x32x16_BF16,
       Intrinsic::amdgcn_mfma_f32_32x32x16_bf16},
      {CanonicalOp::V_MFMA_F32_16x16x32_F16,
       Intrinsic::amdgcn_mfma_f32_16x16x32_f16},
      {CanonicalOp::V_MFMA_F32_16x16x32_FP8_FP8,
       Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8},
      {CanonicalOp::V_MFMA_F32_16x16x32_FP8_BF8,
       Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8},
      {CanonicalOp::V_MFMA_F32_16x16x32_BF8_FP8,
       Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8},
      {CanonicalOp::V_MFMA_F32_16x16x32_BF8_BF8,
       Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8},
      {CanonicalOp::V_MFMA_F32_32x32x16_FP8_FP8,
       Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8},
      {CanonicalOp::V_MFMA_F32_32x32x16_FP8_BF8,
       Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8},
      {CanonicalOp::V_MFMA_F32_32x32x16_BF8_FP8,
       Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8},
      {CanonicalOp::V_MFMA_F32_32x32x16_BF8_BF8,
       Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8},
      // gfx950 F8F6F4 scaled MFMAs. Intrinsic signature differs from the
      // non-scaled family (9 params instead of 6) and is overloaded on the
      // src AB type; the handler detects this via `FT->getNumParams() > 6`.
      {CanonicalOp::V_MFMA_F32_16x16x128_F8F6F4,
       Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4},
      {CanonicalOp::V_MFMA_SCALE_F32_16x16x128_F8F6F4,
       Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4},
      {CanonicalOp::V_MFMA_F32_32x32x64_F8F6F4,
       Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4},
      {CanonicalOp::V_MFMA_SCALE_F32_32x32x64_F8F6F4,
       Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4},
  });
  return *Table;
}

// Read a named immediate operand, or return `fallback` if the opcode does
// not expose that name. Using `getNamedOperandIdx` instead of positional
// scanning of the trailing source list means any future operand reshuffle
// in AMDGPU TableGen flows in for free.
static int64_t readNamedImm(const DecodedInst &Di, AMDGPU::OpName Name,
                            int64_t Fallback = 0) {
  int Idx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Idx < 0 || !Di.isImm(Idx))
    return Fallback;
  return Di.getImm(Idx);
}

// Read a named register operand as a 32-bit value. Returns `fallback`
// when the named operand is absent or not a register (e.g. when the
// encoding carries an immediate in the same slot).
static Value *readNamedReg32(RaiseContext &Ctx, const DecodedInst &Di,
                             AMDGPU::OpName Name, Value *Fallback) {
  int Idx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Idx < 0 || !Di.isReg(Idx))
    return Fallback;
  ParsedReg Pr = Ctx.parseReg(Di.getReg(Idx), Idx);
  if (Pr.RegKind == ParsedReg::OTHER || Pr.RegKind == ParsedReg::NOREG)
    return Fallback;
  return Ctx.Regs.readReg32(Ctx.B, Pr);
}

Expected<HandlerResult> handleMFMA(RaiseContext &Ctx, const DecodedInst &Di,
                                   OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  // AGPR moves travel through the MFMA format bit but are not MFMA ops.
  if (Sop == CanonicalOp::V_ACCVGPR_WRITE_B32 ||
      Sop == CanonicalOp::V_ACCVGPR_READ_B32) {
    Ctx.writeReg32(Op.dst(), Op.src(0));
    Hr.Handled = true;
    return Hr;
  }

  const auto &Table = mfmaIntrinsicTable();
  auto It = Table.find(Sop);
  if (It == Table.end()) {
    return RaiseFailure::unsupportedInstructionForm(
        Di, "MFMA", "no intrinsic mapping for this MFMA CanonicalOp");
  }
  const Intrinsic::ID IntrId = It->second;

  // Derive the src / accum IR types from the intrinsic signature.
  //
  // Non-scaled MFMA intrinsics (`AMDGPUMfmaIntrinsic<DestTy, SrcABTy>`)
  // are not overloaded: six fixed parameters
  //   (SrcABTy, SrcABTy, DestTy, i32 cbsz, i32 abid, i32 blgp)
  // and `Intrinsic::getType(Ctx, ID)` returns the whole signature.
  //
  // Scaled F8F6F4 MFMAs (`AMDGPUMfmaScaleIntrinsic<DestTy>`) are overloaded
  // on the AB vector type: nine parameters
  //   (anyvec A, anyvec B, DestTy C, i32 cbsz, i32 blgp,
  //    i32 opSelA, i32 scaleA, i32 opSelB, i32 scaleB)
  // and we must supply the overload types. AMDGPU kernels uniformly use
  // a v8i32 A/B layout (the widest F8 case) and select the active format
  // via `cbsz` / `blgp`, so we pass `{v8i32, v8i32}` here.
  auto *V8i32Ty = FixedVectorType::get(Ctx.I32Ty, 8);
  SmallVector<Type *, 2> Overloads;
  if (Intrinsic::isOverloaded(IntrId))
    Overloads = {V8i32Ty, V8i32Ty};

  FunctionType *FT = Intrinsic::getType(Ctx.C, IntrId, Overloads);
  const bool IsScaled = FT->getNumParams() == 9;
  if (!IsScaled && FT->getNumParams() != 6)
    return RaiseFailure::unsupportedInstructionForm(
        Di, "MFMA",
        "unexpected MFMA intrinsic arity " + Twine(FT->getNumParams()) +
            " for " + Di.Mnemonic);

  Type *SrcTy = FT->getParamType(0);
  Type *AccumTy = FT->getReturnType();

  ParsedReg Dest = Op.dst();
  ParsedReg SrcA = Op.srcReg(0), SrcB = Op.srcReg(1);
  // The accumulator (src2) may be tied to the destination in some encodings.
  ParsedReg SrcC = Op.isSrcReg(2) ? Op.srcReg(2) : Dest;
  if (SrcA.RegKind == ParsedReg::OTHER || SrcB.RegKind == ParsedReg::OTHER) {
    return RaiseFailure::unsupportedInstructionForm(
        Di, "MFMA", "cannot classify MFMA source registers");
  }

  Value *A = Ctx.Regs.readRegVec(Ctx.B, SrcA, SrcTy);
  Value *B = Ctx.Regs.readRegVec(Ctx.B, SrcB, SrcTy);
  Value *C = Ctx.Regs.readRegVec(Ctx.B, SrcC, AccumTy);

  // Which fp8/bf8 element formats the opcode gives its two operands; nullopt
  // for any non-fp8 MFMA, i.e. nothing here to re-encode.
  std::optional<Fp8AbFormat> AbFmt = [&]() -> std::optional<Fp8AbFormat> {
    switch (Sop) {
    case CanonicalOp::V_MFMA_F32_16x16x32_FP8_FP8:
    case CanonicalOp::V_MFMA_F32_32x32x16_FP8_FP8:
      return Fp8AbFormat::Fp8Fp8;
    case CanonicalOp::V_MFMA_F32_16x16x32_FP8_BF8:
    case CanonicalOp::V_MFMA_F32_32x32x16_FP8_BF8:
      return Fp8AbFormat::Fp8Bf8;
    case CanonicalOp::V_MFMA_F32_16x16x32_BF8_FP8:
    case CanonicalOp::V_MFMA_F32_32x32x16_BF8_FP8:
      return Fp8AbFormat::Bf8Fp8;
    case CanonicalOp::V_MFMA_F32_16x16x32_BF8_BF8:
    case CanonicalOp::V_MFMA_F32_32x32x16_BF8_BF8:
      return Fp8AbFormat::Bf8Bf8;
    default:
      return std::nullopt;
    }
  }();
  // The scaled F8F6F4 family carries fp8/bf8 bytes too, but which of its
  // bytes are fp8/bf8 and which are fp6/fp4 is chosen per instruction by
  // cbsz / blgp, so the table above cannot classify it.
  Fp8Reencode Reenc = Fp8Reencode::None;
  if (AbFmt || IsScaled) {
    Expected<Fp8Reencode> R =
        fp8Reencode(Ctx.Isa, Ctx.TargetIsa, Fp8Dir::SrcToTgt);
    if (!R)
      return RaiseFailure::unsupportedInstructionForm(Di, "MFMA",
                                                      toString(R.takeError()));
    Reenc = *R;
  }
  const bool ToFnuz = Reenc == Fp8Reencode::ToFnuz;
  // Decoding cbsz / blgp and re-encoding only the fp8/bf8 fragments is the
  // same work emitWMMAScaleF8F6F4toMFMA already does for the WMMA
  // equivalent, but it is not wired up here -- so refuse rather than hand
  // source-format bytes to hardware that reads them differently.
  if (IsScaled && Reenc != Fp8Reencode::None)
    return RaiseFailure::unsupportedInstructionForm(
        Di, "MFMA",
        "scaled F8F6F4 MFMA crosses an fp8/bf8 OCP<->FNUZ boundary; "
        "per-operand re-encode driven by cbsz/blgp is not implemented");
  if (AbFmt && Reenc != Fp8Reencode::None) {
    const Fp8Sides Sides = fp8SidesOf(*AbFmt);
    if (!SrcTy->isIntegerTy(64))
      return RaiseFailure::unsupportedInstructionForm(
          Di, "MFMA",
          Twine("fp8 MFMA operand re-encode expects an i64 A/B operand for ") +
              Di.Mnemonic);
    auto Conv = [&](Value *V, bool IsBf8) -> Value * {
      Value *Lo = Ctx.B.CreateTrunc(V, Ctx.I32Ty);
      Value *Hi = Ctx.B.CreateTrunc(
          Ctx.B.CreateLShr(V, ConstantInt::get(SrcTy, 32)), Ctx.I32Ty);
      Lo = convertFp8Dword(Ctx.B, Lo, IsBf8, ToFnuz);
      Hi = convertFp8Dword(Ctx.B, Hi, IsBf8, ToFnuz);
      return Ctx.B.CreateOr(Ctx.B.CreateZExt(Lo, SrcTy),
                            Ctx.B.CreateShl(Ctx.B.CreateZExt(Hi, SrcTy),
                                            ConstantInt::get(SrcTy, 32)));
    };
    A = Conv(A, Sides.AIsBf8);
    B = Conv(B, Sides.BIsBf8);
  }

  // Immediate modifiers keyed off the authoritative named-operand table.
  // `cbsz` is common to both families; `abid` is non-scaled only; scaled
  // instead carries `blgp` + four scale control operands.
  Value *Cbsz =
      ConstantInt::get(Ctx.I32Ty, readNamedImm(Di, AMDGPU::OpName::cbsz));
  Value *Blgp =
      ConstantInt::get(Ctx.I32Ty, readNamedImm(Di, AMDGPU::OpName::blgp));

  Function *MfmaFn =
      Intrinsic::getOrInsertDeclaration(&Ctx.M, IntrId, Overloads);

  Value *CallRet;
  if (IsScaled) {
    // The scale operand layout mirrors `ScaledMAIInst` in
    // `VOP3PInstructions.td`: the two scale VGPRs come in as
    // `scale_src0` / `scale_src1`, and the op_sel bits are carried in the
    // repurposed `src0_modifiers` / `src1_modifiers` immediate slots.
    Value *Zero = ConstantInt::get(Ctx.I32Ty, 0);
    Value *OpSelA = ConstantInt::get(
        Ctx.I32Ty, readNamedImm(Di, AMDGPU::OpName::src0_modifiers));
    Value *OpSelB = ConstantInt::get(
        Ctx.I32Ty, readNamedImm(Di, AMDGPU::OpName::src1_modifiers));
    Value *ScaleA = readNamedReg32(Ctx, Di, AMDGPU::OpName::scale_src0, Zero);
    Value *ScaleB = readNamedReg32(Ctx, Di, AMDGPU::OpName::scale_src1, Zero);
    CallRet = Ctx.B.CreateCall(
        MfmaFn, {A, B, C, Cbsz, Blgp, OpSelA, ScaleA, OpSelB, ScaleB},
        "mfma_scale");
  } else {
    Value *Abid =
        ConstantInt::get(Ctx.I32Ty, readNamedImm(Di, AMDGPU::OpName::abid));
    CallRet = Ctx.B.CreateCall(MfmaFn, {A, B, C, Cbsz, Abid, Blgp}, "mfma");
  }

  Ctx.writeRegVec(Dest, CallRet);
  Hr.Handled = true;
  return Hr;
}

// Drift-detection for the one column of MFMA metadata that stays
// hand-rolled: `CanonicalOp -> Intrinsic::ID`. Every MFMA-format MC opcode
// the decoder can reach at runtime must either (a) have a CanonicalOp entry
// in `mfmaIntrinsicTable()`, or (b) be one of the two AGPR-move
// pseudos that `handleMFMA` short-circuits. Anything else is a silent
// coverage gap: the raiser would hit the "Unknown MFMA" path at
// per-kernel lift time instead of telling us at startup that the
// canon table and the handler table disagree. This is the same
// discipline `initMCState`'s `KMaxSrcs` check uses -- run once, fail
// loudly, no per-kernel surprises.
llvm::Error verifyMFMACoverage(const MCInstrInfo &MCII) {
  const auto &Table = mfmaIntrinsicTable();
  for (unsigned Opc = 0, End = MCII.getNumOpcodes(); Opc < End; ++Opc) {
    const MCInstrDesc &Desc = MCII.get(Opc);
    if (!(Desc.TSFlags & llvm::SIInstrFlags::IsMAI))
      continue;
    CanonicalOp Sop = canonicalOpFor(Opc);
    if (Sop == CanonicalOp::Unknown)
      continue; // Opcode not modelled in CanonicalOpcodes.td -- fails at
                // raise time with "Unknown MFMA"; not a drift we own here.
    if (Sop == CanonicalOp::V_ACCVGPR_WRITE_B32 ||
        Sop == CanonicalOp::V_ACCVGPR_READ_B32)
      continue; // Handled specially above; no intrinsic entry needed.
    if (Table.find(Sop) == Table.end())
      return RaiseFailure::internalFailure(
          "transpiler: MFMA-format opcode #" + Twine(Opc) +
          " maps to CanonicalOp " + Twine(static_cast<int>(Sop)) +
          " but `mfmaIntrinsicTable` has no entry for it. Either add the "
          "Intrinsic::ID row or remove the CanonicalOp from "
          "`CanonicalOpcodes.td`.");
  }

  return llvm::Error::success();
}

} // namespace COMGR::hotswap
