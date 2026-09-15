//===- handle-vop-cross-lane.h - Cross-lane VOP helpers -------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLE_VOP_CROSS_LANE_H
#define HOTSWAP_TRANSPILER_HANDLE_VOP_CROSS_LANE_H

#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

class OperandResolver;
struct DecodedInst;
class RaiseContext;

/// Raise V_READFIRSTLANE_B32.
llvm::Error raiseReadFirstLane32(RaiseContext &Ctx, const DecodedInst &Di,
                                 OperandResolver &Op);

/// Raise V_READLANE_B32.
llvm::Error raiseReadLane32(RaiseContext &Ctx, const DecodedInst &Di,
                            OperandResolver &Op);

/// Raise V_WRITELANE_B32.
llvm::Error raiseWriteLane32(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_HANDLE_VOP_CROSS_LANE_H
