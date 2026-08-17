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

// Return value from every format handler, carried inside an
// `llvm::Expected<HandlerResult>`.
//
// Handlers communicate back in three ways:
//   * `Handled = true` -> the handler fully lowered the instruction.
//   * `Handled = false` (no Error) -> this handler does not claim the
//     instruction; the main loop falls through to the generic
//     `UnsupportedOpcode` diagnostic.
//   * an `llvm::Error` (a `RaiseFailure`) -> the handler recognised the
//     instruction but refuses to lower it (e.g. operand shape
//     unsupported); the main loop records the structured failure and
//     aborts without consulting other handlers.
//
// A handler that computes SCC as a side effect hands the value back in
// `SccResult` for the dispatch loop to commit, or sets `SccHandled` when it
// has already stored it itself.
struct HandlerResult {
  bool Handled = false;
  llvm::Value *SccResult = nullptr;
  bool SccHandled = false;
};

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
