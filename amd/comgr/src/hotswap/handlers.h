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

namespace llvm {
class MCInstrInfo;
} // namespace llvm

namespace COMGR::hotswap {

// Asserts every MFMA-format opcode the disassembler can decode has a
// CanonicalOp handler entry. See `handle-mfma.cpp` for details.
llvm::Error verifyMFMACoverage(const llvm::MCInstrInfo &MCII);

llvm::Expected<HandlerResult> handleSOPP(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleSMEM(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleSOPC(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleSOP1(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleSOPK(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleSOP2(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleVALU(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleFLAT(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleDS(RaiseContext &Ctx, const DecodedInst &Di,
                                       OpResolver &Op);
llvm::Expected<HandlerResult>
handleMUBUF(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleMFMA(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult> handleVOPD(RaiseContext &Ctx,
                                         const DecodedInst &Di, OpResolver &Op);
llvm::Expected<HandlerResult>
handleVIMAGE(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op);

} // namespace COMGR::hotswap

#endif
