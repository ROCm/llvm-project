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

namespace llvm {
class MCInstrInfo;
} // namespace llvm

namespace COMGR::hotswap {

class OpcodeMap;

// Asserts every MFMA-format opcode the disassembler can decode has a CanonicalOp
// handler entry. See `handle_mfma.cpp` for details.
void verifyMFMACoverage(const llvm::MCInstrInfo &MCII, const OpcodeMap &OpcMap);

HandlerResult handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleSMEM(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleSOPC(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleSOPK(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleSOP2(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleVALU(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleFLAT(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleDS(RaiseContext &Ctx, const DecodedInst &Di,
                       OpResolver &Op);
HandlerResult handleMUBUF(RaiseContext &Ctx, const DecodedInst &Di,
                          OpResolver &Op);
HandlerResult handleMFMA(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleVOPD(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op);
HandlerResult handleVIMAGE(RaiseContext &Ctx, const DecodedInst &Di,
                           OpResolver &Op);

} // namespace COMGR::hotswap

#endif
