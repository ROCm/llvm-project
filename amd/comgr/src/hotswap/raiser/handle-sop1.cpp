//===- handle-sop1.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

using namespace llvm;

namespace COMGR::hotswap {

Expected<HandlerResult> handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                                   OpResolver &Op) {
  HandlerResult Hr;

  if (Di.CanonOp == CanonicalOp::S_MOV_B32) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();
    Ctx.writeReg32(*Dst, *Src);
    Hr.Handled = true;
    return Hr;
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset, "SOP1");
}

} // namespace COMGR::hotswap
