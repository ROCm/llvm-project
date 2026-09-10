//===- canonical-op.cpp - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/decoder/canonical-op.h"

#include "llvm/Support/ErrorHandling.h"

namespace COMGR::hotswap {

llvm::StringRef canonicalOpName(CanonicalOp Op) {
  switch (Op) {
#define CANONICAL_OP(Name)                                                     \
  case CanonicalOp::Name:                                                      \
    return #Name;
#include "hotswap/decoder/canonical-op.def"
#undef CANONICAL_OP
  case CanonicalOp::CanonicalOp_COUNT:
    break;
  }
  llvm_unreachable("canonicalOpName: invalid CanonicalOp");
}

} // namespace COMGR::hotswap
