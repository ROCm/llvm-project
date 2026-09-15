//===- wmma-lowering.h - Hotswap matrix remapping --------------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_WMMA_LOWERING_H
#define HOTSWAP_TRANSPILER_WMMA_LOWERING_H

#include "llvm/Support/Error.h"

namespace llvm {
class Value;
}

namespace COMGR::hotswap {

class RaiseContext;

enum class WMMAInputType {
  F16,
  BF16,
  IU8,
};

/// Remap one wave32 16x16 WMMA fragment to wave64 MFMA layout.
llvm::Expected<llvm::Value *> emitWMMAtoMFMA(RaiseContext &Ctx, llvm::Value *A,
                                             llvm::Value *B, llvm::Value *C,
                                             WMMAInputType InputType);

} // namespace COMGR::hotswap

#endif
