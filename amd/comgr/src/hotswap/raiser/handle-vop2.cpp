//===- handle-vop2.cpp - Hotswap transpiler -------------------------------===//
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
#include "hotswap/raiser/op-resolver.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {

Error handleVOP2(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  Instruction::BinaryOps Opcode;
  bool ReverseOperands = false;
  switch (Di.CanonOp) {
  case CanonicalOp::V_ADD_F32:
    Opcode = Instruction::FAdd;
    break;
  case CanonicalOp::V_MUL_F32:
    Opcode = Instruction::FMul;
    break;
  case CanonicalOp::V_SUB_F32:
    Opcode = Instruction::FSub;
    break;
  case CanonicalOp::V_SUBREV_F32:
    Opcode = Instruction::FSub;
    ReverseOperands = true;
    break;
  default:
    return unsupported(Ctx, Di);
  }

  if (Di.NumDefs != 1 || Di.numOperands() == 0 || !Di.isReg(0) ||
      Op.nSrcs() != 2) {
    return unsupported(Ctx, Di,
                       "expected one register destination and two sources");
  }

  if (Error Err = Ctx.validateF32Environment(Di)) {
    return Err;
  }

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst) {
    return Dst.takeError();
  }
  Expected<Value *> Src0Bits = Op.src(0);
  if (!Src0Bits) {
    return Src0Bits.takeError();
  }
  Expected<Value *> Src1Bits = Op.src(1);
  if (!Src1Bits) {
    return Src1Bits.takeError();
  }

  Value *Src0 = Ctx.B.CreateBitCast(*Src0Bits, Ctx.B.getFloatTy());
  Value *Src1 = Ctx.B.CreateBitCast(*Src1Bits, Ctx.B.getFloatTy());
  Value *Lhs = ReverseOperands ? Src1 : Src0;
  Value *Rhs = ReverseOperands ? Src0 : Src1;
  Value *Result = Ctx.B.CreateBinOp(Opcode, Lhs, Rhs);
  Ctx.registers().writeReg32(*Dst,
                             Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty()));
  return Error::success();
}

} // namespace COMGR::hotswap
