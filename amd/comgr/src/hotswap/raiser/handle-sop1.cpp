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
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Op.src(0));
    Hr.Handled = true;
    return Hr;
  }

  return RaiseFailure::unsupportedInstructionForm(
      strippedMnemonic(Ctx.Mc, Di.Inst), Di.Offset, "SOP1");
}

} // namespace COMGR::hotswap
