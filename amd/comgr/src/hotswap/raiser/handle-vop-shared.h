//===- handle-vop-shared.h - Shared VOP lowering helpers --------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLE_VOP_SHARED_H
#define HOTSWAP_TRANSPILER_HANDLE_VOP_SHARED_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

class OperandResolver;
struct DecodedInst;
struct RaiseContext;

/// Builds the result of a two-source instruction from its already-read sources.
using BinaryBuilder = llvm::function_ref<llvm::Value *(
    llvm::IRBuilder<> &, llvm::Value *, llvm::Value *)>;

/// Raise a 32-bit move shared by VOP1 and VOP3 encodings.
llvm::Error raiseMove32(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op);

/// Raise a 64-bit move shared by VOP1 and VOP3 encodings.
llvm::Error raiseMove64(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op);

/// Raise a unary 32-bit bit operation shared by VOP1 and VOP3 encodings.
llvm::Error raiseUnaryBit32(RaiseContext &Ctx, const DecodedInst &Di,
                            OperandResolver &Op);

/// Raise a binary 32-bit operation and write its result.
llvm::Error raiseBinary32(RaiseContext &Ctx, OperandResolver &Op,
                          BinaryBuilder Build);

/// Raise a binary 64-bit operation and write its result.
llvm::Error raiseBinary64(RaiseContext &Ctx, OperandResolver &Op,
                          BinaryBuilder Build);

/// Mask a shift amount to the low bits read by hardware.
llvm::Value *maskShiftAmount(llvm::IRBuilder<> &B, llvm::Value *Amount,
                             unsigned Width);

/// Raise V_BFM_B32 using its five-bit width and offset operands.
llvm::Error raiseBitMask(RaiseContext &Ctx, OperandResolver &Op);

/// Raise V_BCNT_U32_B32 by adding popcount(src0) to src1.
llvm::Error raiseBitCount(RaiseContext &Ctx, OperandResolver &Op);

/// Raise V_LSHLREV_B64 with a 32-bit shift amount and 64-bit value.
llvm::Error raiseShiftLeft64(RaiseContext &Ctx, OperandResolver &Op);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_HANDLE_VOP_SHARED_H
