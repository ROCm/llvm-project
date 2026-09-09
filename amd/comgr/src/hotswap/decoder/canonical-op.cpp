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
#define CANONICAL_OP(Name, Type, ElementType)                                  \
  case CanonicalOp::Name:                                                      \
    return #Name;
#include "hotswap/decoder/canonical-op.def"
#undef CANONICAL_OP
  case CanonicalOp::CanonicalOp_COUNT:
    break;
  }
  llvm_unreachable("canonicalOpName: invalid CanonicalOp");
}

llvm::StringRef canonicalTypeName(CanonicalType Type) {
  switch (Type) {
#define CANONICAL_TYPE(Name)                                                   \
  case CanonicalType::Name:                                                    \
    return #Name;
    CANONICAL_TYPE(None)
    CANONICAL_TYPE(B16)
    CANONICAL_TYPE(B32)
    CANONICAL_TYPE(B64)
    CANONICAL_TYPE(B96)
    CANONICAL_TYPE(B128)
    CANONICAL_TYPE(B256)
    CANONICAL_TYPE(B512)
    CANONICAL_TYPE(I4)
    CANONICAL_TYPE(I8)
    CANONICAL_TYPE(I16)
    CANONICAL_TYPE(I24)
    CANONICAL_TYPE(I32)
    CANONICAL_TYPE(I64)
    CANONICAL_TYPE(U16)
    CANONICAL_TYPE(U24)
    CANONICAL_TYPE(U32)
    CANONICAL_TYPE(U64)
    CANONICAL_TYPE(F16)
    CANONICAL_TYPE(F32)
    CANONICAL_TYPE(F64)
#undef CANONICAL_TYPE
  }
  llvm_unreachable("canonicalTypeName: invalid CanonicalType");
}

CanonicalInst canonicalInst(CanonicalOp Op) {
  switch (Op) {
#define CANONICAL_OP(Name, Type, ElementType)                                  \
  case CanonicalOp::Name:                                                      \
    return {Op, CanonicalType::Type, CanonicalType::ElementType};
#include "hotswap/decoder/canonical-op.def"
#undef CANONICAL_OP
  case CanonicalOp::CanonicalOp_COUNT:
    break;
  }
  llvm_unreachable("canonicalInst: invalid CanonicalOp");
}

unsigned canonicalTypeBitWidth(CanonicalType Type) {
  switch (Type) {
  case CanonicalType::None:
    return 0;
  case CanonicalType::I4:
    return 4;
  case CanonicalType::I8:
    return 8;
  case CanonicalType::B16:
  case CanonicalType::I16:
  case CanonicalType::U16:
  case CanonicalType::F16:
    return 16;
  case CanonicalType::I24:
  case CanonicalType::U24:
    return 24;
  case CanonicalType::B32:
  case CanonicalType::I32:
  case CanonicalType::U32:
  case CanonicalType::F32:
    return 32;
  case CanonicalType::B64:
  case CanonicalType::I64:
  case CanonicalType::U64:
  case CanonicalType::F64:
    return 64;
  case CanonicalType::B96:
    return 96;
  case CanonicalType::B128:
    return 128;
  case CanonicalType::B256:
    return 256;
  case CanonicalType::B512:
    return 512;
  }
  llvm_unreachable("canonicalTypeBitWidth: invalid CanonicalType");
}

} // namespace COMGR::hotswap
