//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

using namespace llvm;

namespace COMGR::hotswap {

Expected<HandlerResult> handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                                   OpResolver &Op) {
  (void)Op;
  HandlerResult Hr;

  if (Di.CanonOp == CanonicalOp::S_ENDPGM) {
    Ctx.B.CreateRetVoid();
    Hr.Handled = true;
    return Hr;
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset, "SOPP");
}

} // namespace COMGR::hotswap
