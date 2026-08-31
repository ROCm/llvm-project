//===- canonical-op.h - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_CANONICAL_OP_H
#define HOTSWAP_TRANSPILER_CANONICAL_OP_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace COMGR::hotswap {

// Architecture-neutral instruction operation used for dispatch in the raiser.
enum class CanonicalOp : uint16_t {
#define CANONICAL_OP(Name) Name,
#include "hotswap/decoder/canonical-op.def"
#undef CANONICAL_OP
  CanonicalOp_COUNT
};

enum class CanonicalType : uint8_t {
  None,
  B16,
  B32,
  B64,
  B128,
  I4,
  I8,
  I16,
  I24,
  I32,
  I64,
  U16,
  U24,
  U32,
  U64,
  F16,
  F32,
};

// A canonical operation and the types needed to interpret its operands.
struct CanonicalInst {
  CanonicalOp Op = CanonicalOp::Unknown;
  CanonicalType Type = CanonicalType::None;
  CanonicalType ElementType = CanonicalType::None;

  friend bool operator==(CanonicalInst Lhs, CanonicalInst Rhs) {
    return Lhs.Op == Rhs.Op && Lhs.Type == Rhs.Type &&
           Lhs.ElementType == Rhs.ElementType;
  }
  friend bool operator!=(CanonicalInst Lhs, CanonicalInst Rhs) {
    return !(Lhs == Rhs);
  }
};

llvm::StringRef canonicalOpName(CanonicalOp Op);
llvm::StringRef canonicalTypeName(CanonicalType Type);
std::string canonicalInstName(CanonicalInst Inst);
unsigned canonicalTypeBitWidth(CanonicalType Type);

} // namespace COMGR::hotswap

#endif
