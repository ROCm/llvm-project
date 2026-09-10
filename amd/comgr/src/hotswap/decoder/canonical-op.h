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

// Architecture-neutral instruction identity used for dispatch in the raiser.
// Each value maps to one or more MC opcodes via OpcodeMap; an MC opcode with no
// mapping is `Unknown` and is refused. The values come from canonical-op.def.
enum class CanonicalOp : uint16_t {
#define CANONICAL_OP(Name) Name,
#include "hotswap/decoder/canonical-op.def"
#undef CANONICAL_OP
  CanonicalOp_COUNT
};

// The enum's spelling for `Op` (e.g. `"S_MOV_B32"` for
// `CanonicalOp::S_MOV_B32`), for use in diagnostics that name the instruction
// class rather than a raw enum position.
llvm::StringRef canonicalOpName(CanonicalOp Op);

} // namespace COMGR::hotswap

#endif
