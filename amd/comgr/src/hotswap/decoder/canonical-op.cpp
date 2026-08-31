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

llvm::StringRef canonicalTypeName(CanonicalType Type) {
  switch (Type) {
  case CanonicalType::None:
    return "";
  case CanonicalType::B16:
    return "B16";
  case CanonicalType::B32:
    return "B32";
  case CanonicalType::B64:
    return "B64";
  case CanonicalType::B128:
    return "B128";
  case CanonicalType::I4:
    return "I4";
  case CanonicalType::I8:
    return "I8";
  case CanonicalType::I16:
    return "I16";
  case CanonicalType::I24:
    return "I24";
  case CanonicalType::I32:
    return "I32";
  case CanonicalType::I64:
    return "I64";
  case CanonicalType::U16:
    return "U16";
  case CanonicalType::U24:
    return "U24";
  case CanonicalType::U32:
    return "U32";
  case CanonicalType::U64:
    return "U64";
  case CanonicalType::F16:
    return "F16";
  case CanonicalType::F32:
    return "F32";
  }
  llvm_unreachable("canonicalTypeName: invalid CanonicalType");
}

std::string canonicalInstName(CanonicalInst Inst) {
  std::string Name = canonicalOpName(Inst.Op).str();
  if (Inst.Type == CanonicalType::None)
    return Name;
  Name += '<';
  Name += canonicalTypeName(Inst.Type);
  if (Inst.ElementType != CanonicalType::None) {
    Name += ", ";
    Name += canonicalTypeName(Inst.ElementType);
  }
  Name += '>';
  return Name;
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
    return 64;
  case CanonicalType::B128:
    return 128;
  }
  llvm_unreachable("canonicalTypeBitWidth: invalid CanonicalType");
}

} // namespace COMGR::hotswap
