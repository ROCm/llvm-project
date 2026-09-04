//===- handle-vopd.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/raiser/raise-context.h"

#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {
namespace {

// One component result awaiting the packet's shared commit point.
struct PendingWrite {
  ParsedReg Dst;
  Value *Val;
};

// Report a malformed VOPD packet as a structured raise failure.
Error malformedVOPD(RaiseContext &Ctx, const DecodedInst &Di,
                    const Twine &Detail) {
  return unsupported(Ctx, Di, Twine("malformed VOPD packet: ") + Detail);
}

// Apply the component's floating-point source modifiers to its bit value.
Value *applySourceMods(RaiseContext &Ctx, Value *V, uint8_t Mods) {
  if (Mods == 0)
    return V;
  bool IsI64 = V->getType()->isIntegerTy(64);
  V = Ctx.B.CreateBitCast(V, IsI64 ? Ctx.B.getDoubleTy() : Ctx.B.getFloatTy());
  if (Mods & 2)
    V = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, V, nullptr, "vopd.abs");
  if (Mods & 1)
    V = Ctx.B.CreateFNeg(V, "vopd.neg");
  return Ctx.B.CreateBitCast(V,
                             IsI64 ? Ctx.B.getInt64Ty() : Ctx.B.getInt32Ty());
}

// Read and modify a 64-bit component source.
Expected<Value *> readSource64(RaiseContext &Ctx, const DecodedInst &Di,
                               const DecodedInst::VOPDHalf &Half, unsigned I) {
  if (I >= Half.NumSrcs)
    return malformedVOPD(Ctx, Di, "component has too few sources");
  Expected<Value *> V = Ctx.registers().readOp64(Di, Half.SrcIdx[I]);
  if (!V)
    return V.takeError();
  return applySourceMods(Ctx, *V, Half.SrcMods[I]);
}

// Read and modify a 32-bit component source.
Expected<Value *> readSource(RaiseContext &Ctx, const DecodedInst &Di,
                             const DecodedInst::VOPDHalf &Half, unsigned I) {
  if (I >= Half.NumSrcs)
    return malformedVOPD(Ctx, Di, "component has too few sources");
  Expected<Value *> V = Ctx.registers().readOp32(Di, Half.SrcIdx[I]);
  if (!V)
    return V.takeError();
  return applySourceMods(Ctx, *V, Half.SrcMods[I]);
}

// Decode a component destination register.
Expected<ParsedReg> readDestination(RaiseContext &Ctx, const DecodedInst &Di,
                                    const DecodedInst::VOPDHalf &Half) {
  return Ctx.registers().parseReg(Di, Half.DstIdx);
}

// Read the per-lane condition from an explicit VOPD3 wave-mask operand.
Expected<Value *> readCondition(RaiseContext &Ctx, const DecodedInst &Di,
                                unsigned OperandIdx) {
  Expected<Value *> Known = Ctx.registers().readOpWaveMaskI1(Di, OperandIdx);
  if (!Known)
    return Known.takeError();
  if (*Known)
    return *Known;

  Expected<Value *> Mask = Ctx.registers().readOpExecWidth(Di, OperandIdx);
  if (!Mask)
    return Mask.takeError();
  return Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, *Mask);
}

// Restrict a 32-bit shift amount to the hardware range.
Value *maskShiftAmount(IRBuilder<> &B, Value *Amount) {
  return B.CreateAnd(Amount, B.getInt32(31), "vopd.shift.amount");
}

// Lower a two-input bitop using the encoded three-input truth table.
Expected<Value *> lowerBitOp3(RaiseContext &Ctx, const DecodedInst &Di,
                              const DecodedInst::VOPDHalf &Half) {
  Expected<Value *> A = readSource(Ctx, Di, Half, 0);
  if (!A)
    return A.takeError();
  Expected<Value *> B = readSource(Ctx, Di, Half, 1);
  if (!B)
    return B.takeError();

  Value *C = Ctx.B.getInt32(0);
  Value *NA = Ctx.B.CreateNot(*A);
  Value *NB = Ctx.B.CreateNot(*B);
  Value *NC = Ctx.B.CreateNot(C);
  Value *Minterms[8] = {
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(NA, NB), NC),
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(NA, NB), C),
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(NA, *B), NC),
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(NA, *B), C),
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(*A, NB), NC),
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(*A, NB), C),
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(*A, *B), NC),
      Ctx.B.CreateAnd(Ctx.B.CreateAnd(*A, *B), C),
  };
  Value *Result = Ctx.B.getInt32(0);
  for (unsigned I = 0; I != 8; ++I)
    if (Half.BitOp3 & (1u << I))
      Result = Ctx.B.CreateOr(Result, Minterms[I], "vopd.bitop");
  return Result;
}

// Lower one VOPD component without committing its destination.
Expected<Value *> lowerHalf(RaiseContext &Ctx, const DecodedInst &Di,
                            const DecodedInst::VOPDHalf &Half, ParsedReg Dst) {
  if (Half.HasBitOp3)
    return lowerBitOp3(Ctx, Di, Half);

  auto Read = [&](unsigned I) { return readSource(Ctx, Di, Half, I); };
  auto ReadBinary = [&]() -> Expected<std::pair<Value *, Value *>> {
    Expected<Value *> S0 = Read(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Read(1);
    if (!S1)
      return S1.takeError();
    return std::pair<Value *, Value *>{*S0, *S1};
  };
  auto ReadFloat = [&](unsigned I) -> Expected<Value *> {
    Expected<Value *> V = Read(I);
    if (!V)
      return V.takeError();
    return Ctx.B.CreateBitCast(*V, Ctx.B.getFloatTy());
  };

  switch (Half.CanonOp) {
  case CanonicalOp::V_MOV_B32:
    return Read(0);

  case CanonicalOp::V_CNDMASK_B32: {
    Expected<std::pair<Value *, Value *>> Srcs = ReadBinary();
    if (!Srcs)
      return Srcs.takeError();
    Value *Cond = nullptr;
    if (Half.NumSrcs == 2) {
      Cond = Ctx.registers().regFile().loadVCC(Ctx.B);
    } else {
      Expected<Value *> C = readCondition(Ctx, Di, Half.SrcIdx[2]);
      if (!C)
        return C.takeError();
      Cond = *C;
    }
    return Ctx.B.CreateSelect(Cond, Srcs->second, Srcs->first, "vopd.cndmask");
  }

  case CanonicalOp::V_ADD_F32:
  case CanonicalOp::V_MUL_F32:
  case CanonicalOp::V_SUB_F32:
  case CanonicalOp::V_SUBREV_F32: {
    if (Error Err = Ctx.validateF32Environment(Di))
      return std::move(Err);
    Expected<Value *> S0 = ReadFloat(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = ReadFloat(1);
    if (!S1)
      return S1.takeError();
    Value *Result = nullptr;
    if (Half.CanonOp == CanonicalOp::V_ADD_F32)
      Result = Ctx.B.CreateFAdd(*S0, *S1, "vopd.fadd");
    else if (Half.CanonOp == CanonicalOp::V_MUL_F32)
      Result = Ctx.B.CreateFMul(*S0, *S1, "vopd.fmul");
    else if (Half.CanonOp == CanonicalOp::V_SUBREV_F32)
      Result = Ctx.B.CreateFSub(*S1, *S0, "vopd.fsubrev");
    else
      Result = Ctx.B.CreateFSub(*S0, *S1, "vopd.fsub");
    return Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty());
  }

  case CanonicalOp::V_FMAC_F32:
  case CanonicalOp::V_FMA_F32:
  case CanonicalOp::V_FMAMK_F32:
  case CanonicalOp::V_FMAAK_F32: {
    if (Error Err = Ctx.validateF32Environment(Di))
      return std::move(Err);
    Expected<Value *> S0 = ReadFloat(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = ReadFloat(1);
    if (!S1)
      return S1.takeError();
    Value *S2 = nullptr;
    if (Half.CanonOp == CanonicalOp::V_FMAC_F32) {
      Value *Acc = Ctx.registers().regFile().readReg32(Ctx.B, Dst);
      if (!Acc)
        return malformedVOPD(Ctx, Di, "cannot read fmac accumulator");
      S2 = Ctx.B.CreateBitCast(Acc, Ctx.B.getFloatTy());
    } else {
      Expected<Value *> Third = ReadFloat(2);
      if (!Third)
        return Third.takeError();
      S2 = *Third;
    }
    Function *Fma =
        Intrinsic::getOrInsertDeclaration(Ctx.B.GetInsertBlock()->getModule(),
                                          Intrinsic::fma, {Ctx.B.getFloatTy()});
    Value *Result = Ctx.B.CreateCall(Fma, {*S0, *S1, S2}, "vopd.fma");
    return Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty());
  }

  case CanonicalOp::V_MAX_NUM_F32:
  case CanonicalOp::V_MIN_NUM_F32: {
    if (Error Err = Ctx.validateF32Environment(Di))
      return std::move(Err);
    Expected<Value *> S0 = ReadFloat(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = ReadFloat(1);
    if (!S1)
      return S1.takeError();
    Intrinsic::ID ID = Half.CanonOp == CanonicalOp::V_MAX_NUM_F32
                           ? Intrinsic::maximumnum
                           : Intrinsic::minimumnum;
    Value *Result =
        Ctx.B.CreateBinaryIntrinsic(ID, *S0, *S1, {}, "vopd.fminmax");
    return Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty());
  }

  case CanonicalOp::V_ADD_F64:
  case CanonicalOp::V_MUL_F64:
  case CanonicalOp::V_FMA_F64:
    if (Error Err = Ctx.validateF64Environment(Di))
      return std::move(Err);
    [[fallthrough]];
  case CanonicalOp::V_MAX_NUM_F64:
  case CanonicalOp::V_MIN_NUM_F64: {
    Expected<Value *> S0Bits = readSource64(Ctx, Di, Half, 0);
    if (!S0Bits)
      return S0Bits.takeError();
    Expected<Value *> S1Bits = readSource64(Ctx, Di, Half, 1);
    if (!S1Bits)
      return S1Bits.takeError();
    Value *S0 = Ctx.B.CreateBitCast(*S0Bits, Ctx.B.getDoubleTy());
    Value *S1 = Ctx.B.CreateBitCast(*S1Bits, Ctx.B.getDoubleTy());
    Value *Result = nullptr;
    if (Half.CanonOp == CanonicalOp::V_ADD_F64)
      Result = Ctx.B.CreateFAdd(S0, S1, "vopd.fadd64");
    else if (Half.CanonOp == CanonicalOp::V_MUL_F64)
      Result = Ctx.B.CreateFMul(S0, S1, "vopd.fmul64");
    else if (Half.CanonOp == CanonicalOp::V_FMA_F64) {
      Expected<Value *> S2Bits = readSource64(Ctx, Di, Half, 2);
      if (!S2Bits)
        return S2Bits.takeError();
      Value *S2 = Ctx.B.CreateBitCast(*S2Bits, Ctx.B.getDoubleTy());
      Function *Fma = Intrinsic::getOrInsertDeclaration(
          Ctx.B.GetInsertBlock()->getModule(), Intrinsic::fma,
          {Ctx.B.getDoubleTy()});
      Result = Ctx.B.CreateCall(Fma, {S0, S1, S2}, "vopd.fma64");
    } else {
      Intrinsic::ID ID = Half.CanonOp == CanonicalOp::V_MAX_NUM_F64
                             ? Intrinsic::maximumnum
                             : Intrinsic::minimumnum;
      Result = Ctx.B.CreateBinaryIntrinsic(ID, S0, S1, {}, "vopd.fminmax64");
    }
    return Ctx.B.CreateBitCast(Result, Ctx.B.getInt64Ty());
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
    Expected<std::pair<Value *, Value *>> Srcs = ReadBinary();
    if (!Srcs)
      return Srcs.takeError();
    Value *S0 = Srcs->first;
    Value *S1 = Srcs->second;
    switch (Half.CanonOp) {
    case CanonicalOp::V_ADD_NC_U32:
      return Ctx.B.CreateAdd(S0, S1, "vopd.add");
    case CanonicalOp::V_SUB_NC_U32:
      return Ctx.B.CreateSub(S0, S1, "vopd.sub");
    case CanonicalOp::V_SUBREV_NC_U32:
      return Ctx.B.CreateSub(S1, S0, "vopd.subrev");
    case CanonicalOp::V_LSHLREV_B32:
      return Ctx.B.CreateShl(S1, maskShiftAmount(Ctx.B, S0), "vopd.shl");
    case CanonicalOp::V_LSHRREV_B32:
      return Ctx.B.CreateLShr(S1, maskShiftAmount(Ctx.B, S0), "vopd.lshr");
    case CanonicalOp::V_ASHRREV_I32:
      return Ctx.B.CreateAShr(S1, maskShiftAmount(Ctx.B, S0), "vopd.ashr");
    case CanonicalOp::V_AND_B32:
      return Ctx.B.CreateAnd(S0, S1, "vopd.and");
    case CanonicalOp::V_OR_B32:
      return Ctx.B.CreateOr(S0, S1, "vopd.or");
    case CanonicalOp::V_XOR_B32:
      return Ctx.B.CreateXor(S0, S1, "vopd.xor");
    default:
      llvm_unreachable("filtered VOPD binary operation");
    }
  }

  case CanonicalOp::V_MIN_I32:
  case CanonicalOp::V_MAX_I32:
  case CanonicalOp::V_MIN_U32:
  case CanonicalOp::V_MAX_U32: {
    Expected<std::pair<Value *, Value *>> Srcs = ReadBinary();
    if (!Srcs)
      return Srcs.takeError();
    Intrinsic::ID ID = Intrinsic::smin;
    if (Half.CanonOp == CanonicalOp::V_MAX_I32)
      ID = Intrinsic::smax;
    else if (Half.CanonOp == CanonicalOp::V_MIN_U32)
      ID = Intrinsic::umin;
    else if (Half.CanonOp == CanonicalOp::V_MAX_U32)
      ID = Intrinsic::umax;
    return Ctx.B.CreateBinaryIntrinsic(ID, Srcs->first, Srcs->second, {},
                                       "vopd.minmax");
  }

  case CanonicalOp::V_BITOP3_B32:
    return malformedVOPD(Ctx, Di, "bitop3 component has no truth table");

  default:
    return unsupported(Ctx, Di, "unsupported VOPD component operation");
  }
}

} // namespace

Error handleVOPD(RaiseContext &Ctx, const DecodedInst &Di) {
  if (!Di.HasVOPD)
    return malformedVOPD(Ctx, Di, "missing structural decode");

  SmallVector<PendingWrite, 2> Writes;
  for (unsigned Component : AMDGPU::VOPD::COMPONENTS) {
    const DecodedInst::VOPDHalf &Half = Di.VOPD[Component];
    Expected<ParsedReg> Dst = readDestination(Ctx, Di, Half);
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Result = lowerHalf(Ctx, Di, Half, *Dst);
    if (!Result)
      return Result.takeError();
    Writes.push_back({*Dst, *Result});
  }

  // Both halves observe the pre-packet register state, including when one
  // half reads the other half's destination. Commit only after both results
  // have been formed.
  for (const PendingWrite &Write : Writes) {
    if (Write.Val->getType()->isIntegerTy(64))
      Ctx.registers().writeReg64(Write.Dst, Write.Val);
    else
      Ctx.registers().writeReg32(Write.Dst, Write.Val);
  }
  return Error::success();
}

} // namespace COMGR::hotswap
