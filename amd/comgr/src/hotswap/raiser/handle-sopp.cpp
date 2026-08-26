//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

using namespace llvm;

namespace COMGR::hotswap {

Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &) {
  if (Di.CanonOp == CanonicalOp::S_ENDPGM) {
    Ctx.B.CreateRetVoid();
    return Error::success();
  }

  return unsupported(Ctx, Di);
}

} // namespace COMGR::hotswap
