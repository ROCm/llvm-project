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
  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::hotswap
