//===- handle-vop-cross-lane.h - Cross-lane VOP helpers -------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_HANDLE_VOP_CROSS_LANE_H
#define TRANSPILER_HANDLE_VOP_CROSS_LANE_H

#include "llvm/Support/Error.h"

namespace COMGR::transpiler {

struct OperandResolver;
struct DecodedInst;
class RaiseContext;

/// Raise a DPP16 V_MOV_B32 row_shr with full write masks and BC = FI = 0.
llvm::Error raiseDPPMove32(RaiseContext &Ctx, const DecodedInst &Di,
                           OperandResolver &Op);

/// Raise V_MBCNT_LO_U32_B32.
llvm::Error raiseMaskedBitCountLow32(RaiseContext &Ctx, const DecodedInst &Di,
                                     OperandResolver &Op);

/// Raise V_MBCNT_HI_U32_B32.
llvm::Error raiseMaskedBitCountHigh32(RaiseContext &Ctx, const DecodedInst &Di,
                                      OperandResolver &Op);

/// Raise V_READFIRSTLANE_B32.
llvm::Error raiseReadFirstLane32(RaiseContext &Ctx, const DecodedInst &Di,
                                 OperandResolver &Op);

/// Raise V_READLANE_B32.
llvm::Error raiseReadLane32(RaiseContext &Ctx, const DecodedInst &Di,
                            OperandResolver &Op);

/// Raise V_WRITELANE_B32.
llvm::Error raiseWriteLane32(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op);

} // namespace COMGR::transpiler

#endif // TRANSPILER_HANDLE_VOP_CROSS_LANE_H
