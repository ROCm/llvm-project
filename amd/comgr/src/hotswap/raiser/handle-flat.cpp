//===- handle-flat.cpp - Hotswap transpiler -------------------------------===//
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
#include "hotswap/raiser/flat-addr.h"
#include "hotswap/raiser/op-resolver.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {

// A dword global access is aligned to its own width.
static constexpr Align DwordGlobalAccessAlignment = Align::Constant<4>();

Error handleFLAT(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  if (Di.CanonOp != CanonicalOp::GLOBAL_LOAD_B32)
    return unsupported(Ctx, Di, "unsupported flat memory operation");

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();

  Expected<Value *> Address =
      emitGlobalAddress(Ctx, Di, DwordGlobalAccessAlignment);
  if (!Address)
    return Address.takeError();

  // An inactive lane holds an unconstrained address, so the load itself is
  // predicated and not only the register write it feeds.
  Ctx.registers().writeReg32UnderExec(*Destination, [&] {
    return Ctx.B.CreateAlignedLoad(Ctx.B.getInt32Ty(), *Address,
                                   DwordGlobalAccessAlignment, "global_load");
  });
  return Error::success();
}

} // namespace COMGR::hotswap
