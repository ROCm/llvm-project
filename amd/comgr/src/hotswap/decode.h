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

// Output of the decode phase: a linearised stream of decoded instructions
// plus the set of basic-block start offsets the CFG recovery discovered.
//
// `blockStarts` uses `std::set` because Phase 3 iterates it in ascending
// order to assign deterministic BB labels and because the decode loop
// relies on `upper_bound` to decide whether an `s_endpgm` terminates
// the scan or is an early-return that still has later reachable code.
//
// `insts` uses `SmallVector<T, 0>` -- the LLVM-native escape hatch for
// "I want the SmallVector API but no inline storage because
// `sizeof(T)` is too large for the default inline budget."
// `DecodedInst` contains three `std::string`s plus an inline MCInst
// operand array, which exceeds LLVM's 256-byte default cap; the zero
// inline capacity explicitly opts out of inline storage while still
// getting SmallVector's API and move-only guarantees.
struct DecodeResult {
  llvm::SmallVector<DecodedInst, 0> Insts;
  std::set<uint64_t> BlockStarts;
};

// Decode `textBytes` starting at `kernelOffset` using the caller-owned
// MC + OpcodeMap. Produces a fully populated `DecodedInst` per MCInst
// (canonOp, tsFlags, srcMap/modMap, implicit-def classification, branch
// targets) and the set of block-start offsets.
//
// Returns an error for invalid decode extents (offset/start/end outside the
// .text contents or inconsistent with one another) so callers can surface a
// diagnostic instead of aborting.
//
// Also returns an error on any MC/TableGen invariant violation (unknown
// tied-to-def OpName, srcMap vs OpName::srcN drift, KMaxSrcs overflow). This is
// the LLVM-version-drift guard surface -- every check here catches an upstream
// LLVM change before it can silently corrupt a handler's view of an
// instruction.
llvm::Expected<DecodeResult>
decodeKernel(const MCState &Mc, const OpcodeMap &OpcMap,
             llvm::ArrayRef<uint8_t> TextBytes, uint64_t KernelOffset,
             uint64_t KernelEndOffset = 0,
             std::optional<uint64_t> KernelStartOffset = std::nullopt);

// Compute the decoded CFG successors for a block ending in LastInst.
// NextBlockOffset is the linear fallthrough block start, or std::nullopt when
// no decoded linear fallthrough block exists. The model is intentionally
// conservative and is the shared successor model for block-level analyses.
llvm::Expected<llvm::SmallVector<uint64_t>>
computeDecodedBlockSuccessors(const DecodedInst &LastInst,
                              std::optional<uint64_t> NextBlockOffset);

// Compute the absolute byte offset of a SOPP branch target. SOPP branch
// immediates are signed 16-bit dword offsets from the next instruction
// (`PC + 4`). Invalid source-offset arithmetic returns an error rather than
// dropping the edge from CFG recovery.
llvm::Expected<uint64_t> computeSoppBranchTarget(uint64_t Off, int64_t RawImm);

// True when LastInst terminates the recovered source block. This is separate
// from successor count: s_swap_pc_i64 ends its block while still having a
// modeled fallthrough return site.
bool decodedInstEndsBlock(const DecodedInst &LastInst);

} // namespace COMGR::hotswap

#endif
