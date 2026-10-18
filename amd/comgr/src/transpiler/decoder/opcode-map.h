//===- opcode-map.h - Transpiler ------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_OPCODE_MAP_H
#define TRANSPILER_OPCODE_MAP_H

#include "canonical-op.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCInstrInfo.h"

#include <cstdint>

namespace COMGR::transpiler {

// Maps AMDGPU MC opcodes to CanonicalOp values. Opcodes without a mapping
// return CanonicalOp::Unknown.
class OpcodeMap {
public:
  // CanonicalOp for `Opcode`, or Unknown when it has no mapping.
  CanonicalOp lookup(unsigned Opcode) const;

  // Populate the map from `MCII`. Must run before any lookup.
  void build(const llvm::MCInstrInfo &MCII);

private:
  llvm::DenseMap<unsigned, CanonicalOp> Map;
};

} // namespace COMGR::transpiler

#endif
