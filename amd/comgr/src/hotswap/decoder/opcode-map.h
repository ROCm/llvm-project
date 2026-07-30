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

// Maps raw LLVM MC opcodes emitted by the AMDGPU disassembler to
// architecture-neutral CanonicalOp tags that the raiser dispatches on.
//
// The map is built once at raiser-initialization time from MCInstrInfo and
// TableGen-generated AMDGPU instruction tables. Almost no string parsing is
// involved: encoding variants (e32/e64/DPP/SDWA/subtarget-specific real
// encodings) are folded onto their canonical pseudo via the
// AMDGPU::InstrMapping helpers, and the resulting canonical pseudo is matched
// against a compile-time table of AMDGPU::<opcode> enum constants that grows
// one row per landed handler. Every other opcode maps to Unknown.
class OpcodeMap {
public:
  // Lookup is hot-path: called once per decoded instruction.
  CanonicalOp lookup(unsigned Opcode) const;

  // Build is one-shot: called during raiser initialization.
  void build(const llvm::MCInstrInfo &MCII);

private:
  llvm::DenseMap<unsigned, CanonicalOp> Map;
};

} // namespace COMGR::hotswap

#endif
