//===- decode.h - Hotswap transpiler --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_DECODE_H
#define HOTSWAP_TRANSPILER_DECODE_H

#include "decoded-inst.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <set>

namespace COMGR::hotswap {

struct MCState;
class OpcodeMap;

// Output of the decode phase: the decoded instructions in program order and
// the block-start offsets discovered while decoding. BlockStarts is ordered so
// it can be iterated in offset order and queried with upper_bound.
struct DecodeResult {
  llvm::SmallVector<DecodedInst, 0> Insts;
  std::set<uint64_t> BlockStarts;
};

// Decode the kernel in `TextBytes` at `[KernelOffset, KernelEndOffset)` using
// the caller-owned MC and OpcodeMap, producing a DecodedInst per instruction
// and the set of block-start offsets. A nullopt `KernelEndOffset` scans to the
// end of `TextBytes`; `KernelStartOffset` defaults to `KernelOffset`. Returns
// an error if any bytes in that range do not decode to an instruction.
llvm::Expected<DecodeResult>
decodeKernel(const MCState &Mc, const OpcodeMap &OpcMap,
             llvm::ArrayRef<uint8_t> TextBytes, uint64_t KernelOffset,
             std::optional<uint64_t> KernelEndOffset = std::nullopt,
             std::optional<uint64_t> KernelStartOffset = std::nullopt);

// Compute the CFG successors of a block whose last instruction is `LastInst`.
// `NextBlockOffset` is the fall-through successor, or nullopt when no block
// follows. A block ending in s_endpgm has no successors; any other block falls
// through.
llvm::Expected<llvm::SmallVector<uint64_t>>
computeDecodedBlockSuccessors(const DecodedInst &LastInst,
                              std::optional<uint64_t> NextBlockOffset);

// True when `LastInst` terminates its block.
bool decodedInstEndsBlock(const DecodedInst &LastInst);

} // namespace COMGR::hotswap

#endif
