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

namespace COMGR::hotswap {

// Architecture-neutral opcode used for dispatch in the raiser.
enum class CanonicalOp : uint16_t {
#define CANONICAL_OP(Name, Type, ElementType) Name,
#include "hotswap/decoder/canonical-op.def"
#undef CANONICAL_OP
  CanonicalOp_COUNT
};

enum class CanonicalType : uint8_t {
  None,
  B16,
  B32,
  B64,
  B96,
  B128,
  B256,
  B512,
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
  F64,
};

// A canonical opcode and its value types.
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

// The enum's spelling for `Op` (e.g. `"S_MOV_B32"` for
// `CanonicalOp::S_MOV_B32`), for use in diagnostics that name the instruction
// class rather than a raw enum position.
llvm::StringRef canonicalOpName(CanonicalOp Op);
llvm::StringRef canonicalTypeName(CanonicalType Type);
CanonicalInst canonicalInst(CanonicalOp Op);
unsigned canonicalTypeBitWidth(CanonicalType Type);

} // namespace COMGR::hotswap

#endif
