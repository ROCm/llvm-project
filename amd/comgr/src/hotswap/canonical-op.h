//===- canonical-op.h - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_CANONICAL_OP_H
#define HOTSWAP_TRANSPILER_CANONICAL_OP_H

#include <cstdint>

namespace COMGR::hotswap {

// Architecture-neutral instruction identity used for dispatch in the raiser.
// Each entry maps to one or more MC opcodes; `canonicalOpFor` in opcode-map.h
// is the lookup.
//
// The enumerators, their documentation, and the MC opcodes that canonicalize
// onto them all come from `CanonicalOpcodes.td`. Add a canonical opcode
// there, not here.
enum class CanonicalOp : uint16_t {
#define GET_CANONICAL_OP_ENUM
#include "CanonicalOpcodes.inc"
};

inline bool isMatrixCanonicalOp(CanonicalOp Op) {
  const uint16_t V = static_cast<uint16_t>(Op);
  return V >= static_cast<uint16_t>(CanonicalOp::MATRIX_OP_BEGIN_SENTINEL) &&
         V <= static_cast<uint16_t>(CanonicalOp::MATRIX_OP_END_SENTINEL);
}

// Stable human-readable identifier for a CanonicalOp (the enum's spelling,
// e.g. `"V_CMPX"` for `CanonicalOp::V_CMPX`). Used in diagnostics -- prefer
// this over `(int)sop` so errors name the instruction class rather
// than a raw enum position that drifts with enum edits.
const char *canonicalOpName(CanonicalOp Op);

} // namespace COMGR::hotswap

#endif
