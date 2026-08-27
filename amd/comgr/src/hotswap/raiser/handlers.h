//===- handlers.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLERS_H
#define HOTSWAP_TRANSPILER_HANDLERS_H

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/op-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

// Return a structured refusal for an unsupported instruction form.
inline llvm::Error unsupported(const RaiseContext &Ctx, const DecodedInst &Di,
                               const llvm::Twine &Detail = {}) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags), Detail);
}

// Lower one instruction of the format the handler is named for, emitting into
// `Ctx`'s builder and reading its operands through `Op`. The raiser runs the
// first handler whose format bit matches and only that one, so a handler that
// does not recognize the opcode returns a `RaiseFailure` rather than declining:
// no later handler gets the chance to claim it.
llvm::Error handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
llvm::Error handleSOP2(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
llvm::Error handleSOPC(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
llvm::Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
// Translate supported SMEM loads or return a structured refusal.
llvm::Error handleSMEM(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
// Translate supported GLOBAL memory accesses, or return a structured refusal.
// The format covers flat, global and scratch addressing; only the global forms
// are recognized and the rest are refused.
llvm::Error handleFLAT(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
/// Translate a supported plain VOP2 instruction, or return a structured
/// refusal.
llvm::Error handleVOP2(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
/// Translate a supported plain VOPC comparison into the condition registers
/// the opcode writes, or return a structured refusal.
llvm::Error handleVOPC(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);

} // namespace COMGR::hotswap

#endif
