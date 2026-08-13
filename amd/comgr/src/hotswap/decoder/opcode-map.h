//===- opcode-map.h - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_OPCODE_MAP_H
#define HOTSWAP_TRANSPILER_OPCODE_MAP_H

#include "canonical-op.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCInstrInfo.h"

#include <cstdint>

namespace COMGR::hotswap {

// Maps raw AMDGPU MC opcodes to architecture-neutral CanonicalOp tags.
// Encoding variants (e32/e64/DPP/SDWA/subtarget-specific real encodings) fold
// onto a single canonical pseudo, so they share one tag; an opcode with no tag
// maps to Unknown.
class OpcodeMap {
public:
  // Tag for `Opcode`, or Unknown when it has none.
  CanonicalOp lookup(unsigned Opcode) const;

  // Populate the map from `MCII`. Must run before any lookup.
  void build(const llvm::MCInstrInfo &MCII);

private:
  llvm::DenseMap<unsigned, CanonicalOp> Map;
};

} // namespace COMGR::hotswap

#endif
