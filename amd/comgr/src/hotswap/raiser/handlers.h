//===- handlers.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLERS_H
#define HOTSWAP_TRANSPILER_HANDLERS_H

#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/raiser/op-resolver.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

// Lower one instruction of the format the handler is named for, emitting into
// `Ctx`'s builder and reading its operands through `Op`. The raiser runs the
// first handler whose format bit matches and only that one, so a handler that
// does not recognize the opcode returns a `RaiseFailure` rather than declining:
// no later handler gets the chance to claim it.
llvm::Error handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
llvm::Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);

} // namespace COMGR::hotswap

#endif
