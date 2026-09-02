//===- handle-vop1.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {

Error handleVOP1(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() < 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();

  switch (Di.CanonOp) {
  case CanonicalOp::V_MOV_B32: {
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();
    Ctx.registers().writeReg32(*Dst, *Src);
    return Error::success();
  }
  case CanonicalOp::V_MOV_B64: {
    Expected<Value *> Src = Op.src64(0);
    if (!Src)
      return Src.takeError();
    Ctx.registers().writeReg64(*Dst, *Src);
    return Error::success();
  }
  case CanonicalOp::V_NOT_B32:
  case CanonicalOp::V_BFREV_B32:
  case CanonicalOp::V_FFBH_U32:
  case CanonicalOp::V_FFBL_B32:
  case CanonicalOp::V_FFBH_I32: {
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();

    Value *Result;
    switch (Di.CanonOp) {
    case CanonicalOp::V_NOT_B32:
      Result = Ctx.B.CreateNot(*Src, "not");
      break;
    case CanonicalOp::V_BFREV_B32:
      Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::bitreverse, *Src, nullptr,
                                          "bfrev");
      break;
    case CanonicalOp::V_FFBH_U32:
    case CanonicalOp::V_FFBL_B32: {
      Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_FFBH_U32
                             ? Intrinsic::ctlz
                             : Intrinsic::cttz;
      Value *Count = Ctx.B.CreateIntrinsic(
          ID, {Ctx.B.getInt32Ty()}, {*Src, Ctx.B.getFalse()}, nullptr, "ffb");
      Value *IsZero = Ctx.B.CreateICmpEQ(*Src, Ctx.B.getInt32(0), "ffb.zero");
      Result = Ctx.B.CreateSelect(IsZero, Ctx.B.getInt32(-1), Count, "ffb");
      break;
    }
    case CanonicalOp::V_FFBH_I32:
      Result =
          Ctx.B.CreateIntrinsic(Intrinsic::amdgcn_sffbh, {Ctx.B.getInt32Ty()},
                                {*Src}, nullptr, "ffbh.i32");
      break;
    default:
      llvm_unreachable("not a VOP1 integer bit operation");
    }
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }
  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
