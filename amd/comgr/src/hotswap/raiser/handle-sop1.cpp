//===- handle-sop1.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace COMGR::hotswap {

// Read source 0 at the width the opcode operates on.
static Expected<Value *> readSrc0(OpResolver &Op, bool Is64) {
  return Is64 ? Op.src64(0) : Op.src(0);
}

// Write V to Dst at the width the opcode operates on.
static void writeDst(RegisterState &Registers, ParsedReg Dst, Value *V,
                     bool Is64) {
  if (Is64)
    Registers.writeReg64(Dst, V);
  else
    Registers.writeReg32(Dst, V);
}

// Whether CanonOp is one of the scalar float opcodes, each of which takes one
// 32-bit source SGPR and writes one 32-bit destination SGPR.
static bool isScalarFloat(CanonicalOp CanonOp) {
  switch (CanonOp) {
  case CanonicalOp::S_CEIL_F32:
  case CanonicalOp::S_FLOOR_F32:
  case CanonicalOp::S_TRUNC_F32:
  case CanonicalOp::S_RNDNE_F32:
  case CanonicalOp::S_CVT_F32_I32:
  case CanonicalOp::S_CVT_F32_U32:
  case CanonicalOp::S_CVT_I32_F32:
  case CanonicalOp::S_CVT_U32_F32:
  case CanonicalOp::S_CVT_F16_F32:
  case CanonicalOp::S_CVT_F32_F16:
  case CanonicalOp::S_CVT_HI_F32_F16:
  case CanonicalOp::S_CEIL_F16:
  case CanonicalOp::S_FLOOR_F16:
  case CanonicalOp::S_TRUNC_F16:
  case CanonicalOp::S_RNDNE_F16:
    return true;
  default:
    return false;
  }
}

// Round the f32 in Src with intrinsic ID. The result stays in floating-point
// form, so it goes back as an f32 bit pattern rather than as an integer.
static Value *roundF32(IRBuilder<> &B, Value *Src, Intrinsic::ID ID) {
  Value *Rounded =
      B.CreateUnaryIntrinsic(ID, B.CreateBitCast(Src, B.getFloatTy()));
  return B.CreateBitCast(Rounded, B.getInt32Ty());
}

// Round the f16 in the low half of Src with intrinsic ID. A scalar f16 opcode
// reads the low half of its source SGPR and zeroes the high half of its
// destination.
static Value *roundF16(IRBuilder<> &B, Value *Src, Intrinsic::ID ID) {
  Value *Half =
      B.CreateBitCast(B.CreateTrunc(Src, B.getInt16Ty()), B.getHalfTy());
  Value *Rounded = B.CreateUnaryIntrinsic(ID, Half);
  return B.CreateZExt(B.CreateBitCast(Rounded, B.getInt16Ty()), B.getInt32Ty());
}

// The 32 bits the scalar float opcode CanonOp writes to its destination SGPR,
// given the 32 bits Src of its source. CanonOp must satisfy isScalarFloat.
static Value *emitScalarFloat(IRBuilder<> &B, CanonicalOp CanonOp, Value *Src) {
  switch (CanonOp) {
  case CanonicalOp::S_CEIL_F32:
    return roundF32(B, Src, Intrinsic::ceil);
  case CanonicalOp::S_FLOOR_F32:
    return roundF32(B, Src, Intrinsic::floor);
  case CanonicalOp::S_TRUNC_F32:
    return roundF32(B, Src, Intrinsic::trunc);
  case CanonicalOp::S_RNDNE_F32:
    return roundF32(B, Src, Intrinsic::roundeven);
  case CanonicalOp::S_CEIL_F16:
    return roundF16(B, Src, Intrinsic::ceil);
  case CanonicalOp::S_FLOOR_F16:
    return roundF16(B, Src, Intrinsic::floor);
  case CanonicalOp::S_TRUNC_F16:
    return roundF16(B, Src, Intrinsic::trunc);
  case CanonicalOp::S_RNDNE_F16:
    return roundF16(B, Src, Intrinsic::roundeven);
  case CanonicalOp::S_CVT_F32_I32:
    return B.CreateBitCast(B.CreateSIToFP(Src, B.getFloatTy()), B.getInt32Ty());
  case CanonicalOp::S_CVT_F32_U32:
    return B.CreateBitCast(B.CreateUIToFP(Src, B.getFloatTy()), B.getInt32Ty());
  // The hardware saturates an out-of-range input and converts NaN to zero.
  // Plain fptosi and fptoui make both of those poison.
  case CanonicalOp::S_CVT_I32_F32:
    return B.CreateIntrinsic(Intrinsic::fptosi_sat,
                             {B.getInt32Ty(), B.getFloatTy()},
                             {B.CreateBitCast(Src, B.getFloatTy())});
  case CanonicalOp::S_CVT_U32_F32:
    return B.CreateIntrinsic(Intrinsic::fptoui_sat,
                             {B.getInt32Ty(), B.getFloatTy()},
                             {B.CreateBitCast(Src, B.getFloatTy())});
  case CanonicalOp::S_CVT_F16_F32: {
    Value *Half =
        B.CreateFPTrunc(B.CreateBitCast(Src, B.getFloatTy()), B.getHalfTy());
    return B.CreateZExt(B.CreateBitCast(Half, B.getInt16Ty()), B.getInt32Ty());
  }
  case CanonicalOp::S_CVT_F32_F16:
  case CanonicalOp::S_CVT_HI_F32_F16: {
    Value *Bits =
        CanonOp == CanonicalOp::S_CVT_HI_F32_F16 ? B.CreateLShr(Src, 16) : Src;
    Value *Half =
        B.CreateBitCast(B.CreateTrunc(Bits, B.getInt16Ty()), B.getHalfTy());
    return B.CreateBitCast(B.CreateFPExt(Half, B.getFloatTy()), B.getInt32Ty());
  }
  default:
    llvm_unreachable("not a scalar float opcode");
  }
}

Error handleSOP1(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  if (Di.CanonOp == CanonicalOp::S_MOV_B32 ||
      Di.CanonOp == CanonicalOp::S_MOV_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_MOV_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = readSrc0(Op, Is64);
    if (!Src)
      return Src.takeError();
    writeDst(Ctx.registers(), *Dst, *Src, Is64);
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_BREV_B32 ||
      Di.CanonOp == CanonicalOp::S_BREV_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_BREV_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = readSrc0(Op, Is64);
    if (!Src)
      return Src.takeError();
    Value *Reversed = Ctx.B.CreateUnaryIntrinsic(Intrinsic::bitreverse, *Src,
                                                 /*FMFSource=*/{}, "s_brev");
    writeDst(Ctx.registers(), *Dst, Reversed, Is64);
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_NOT_B32 ||
      Di.CanonOp == CanonicalOp::S_NOT_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_NOT_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = readSrc0(Op, Is64);
    if (!Src)
      return Src.takeError();
    Value *Result = Ctx.B.CreateNot(*Src, "s_not");
    writeDst(Ctx.registers(), *Dst, Result, Is64);
    Ctx.registers().writeScc(Ctx.B.CreateIsNotNull(Result, "s_not_scc"));
    return Error::success();
  }

  // A clear SCC leaves the destination alone. The MC form carries no tied
  // operand for that preserved value, so it is read back off the destination.
  if (Di.CanonOp == CanonicalOp::S_CMOV_B32 ||
      Di.CanonOp == CanonicalOp::S_CMOV_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_CMOV_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = readSrc0(Op, Is64);
    if (!Src)
      return Src.takeError();
    Expected<Value *> Old = Is64 ? Op.dstValue64() : Op.dstValue();
    if (!Old)
      return Old.takeError();
    Value *Moved =
        Ctx.B.CreateSelect(Ctx.registers().readScc(), *Src, *Old, "s_cmov");
    writeDst(Ctx.registers(), *Dst, Moved, Is64);
    return Error::success();
  }

  if (isScalarFloat(Di.CanonOp)) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();
    Ctx.registers().writeReg32(*Dst, emitScalarFloat(Ctx.B, Di.CanonOp, *Src));
    return Error::success();
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

} // namespace COMGR::hotswap
