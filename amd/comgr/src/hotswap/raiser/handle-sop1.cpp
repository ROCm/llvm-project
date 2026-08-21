//===- handle-sop1.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace COMGR::hotswap {

// Read source 0 at the width the opcode operates on.
static Expected<Value *> readSrc0(OperandResolver &Op, bool Is64) {
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

Error handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
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
    // SCC is the result compared against zero, which storeSCC does for a
    // value wider than the single bit SCC holds.
    Ctx.registers().regFile().storeSCC(Ctx.B, Result);
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
    Value *Moved = Ctx.B.CreateSelect(Ctx.registers().regFile().loadSCC(Ctx.B),
                                      *Src, *Old, "s_cmov");
    writeDst(Ctx.registers(), *Dst, Moved, Is64);
    return Error::success();
  }

  return unsupported(Ctx, Di);
}

} // namespace COMGR::hotswap
