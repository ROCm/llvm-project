//===- handlers.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLERS_H
#define HOTSWAP_TRANSPILER_HANDLERS_H

#include "raise-context.h"

#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

// Lower a scalar ALU (SOP1) instruction. The HandlerResult is marked handled
// when the instruction was lifted; an unhandled result means this handler does
// not claim the instruction; a RaiseFailure means it recognises but refuses the
// form.
llvm::Expected<HandlerResult> handleSOP1(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);

// Lower a scalar program (SOPP) instruction; see handleSOP1 for the result
// convention.
llvm::Expected<HandlerResult> handleSOPP(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);

} // namespace COMGR::hotswap

#endif
