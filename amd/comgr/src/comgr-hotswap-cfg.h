//===- comgr-hotswap-cfg.h - HotSwap register-liveness CFG ---------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Control-flow graph construction and reverse-post-order traversal for the
/// HotSwap register-liveness port. This is the third stage: it partitions a
/// decoded instruction stream into single-entry basic blocks, wires up
/// successor/predecessor edges, and provides an RPO ordering that the backward
/// dataflow solver (a later stage) will iterate.
///
/// This is a dedicated \c reglive::Cfg, deliberately separate from the
/// production \c COMGR::hotswap::CFG (and its weak \c buildCfg stub): the
/// register-liveness port is grown in isolation and is not wired into any
/// production rewrite path yet.
///
/// Branch / terminator classification and direct-target resolution reuse the
/// shared HotSwap MC helpers (\c LLVMState::MIA and
/// \c evaluateDirectControlFlowTarget), so this file does not re-derive AMDGPU
/// branch encodings.
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_CFG_H
#define COMGR_HOTSWAP_CFG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace COMGR {
namespace hotswap {

struct InternalDecodedInst;
struct LLVMState;

namespace reglive {

/// A maximal single-entry run of decoded instructions.
///
/// \c InstIndices holds positions into the flat decoded vector the CFG was
/// built from; \c Successors / \c Predecessors are indices into
/// \c Cfg::Blocks. Offsets are .text byte offsets: \c StartOffset is the first
/// instruction's offset and \c EndOffset is one past the block's last
/// instruction (the fall-through address).
struct BasicBlock {
  uint64_t StartOffset = 0;
  uint64_t EndOffset = 0;
  llvm::SmallVector<size_t> InstIndices;
  llvm::SmallVector<unsigned> Successors;
  llvm::SmallVector<unsigned> Predecessors;
};

/// Control-flow graph over one decoded instruction range. \c OffsetToBlock
/// maps a block's start .text offset to its index in \c Blocks.
struct Cfg {
  std::vector<BasicBlock> Blocks;
  llvm::DenseMap<uint64_t, unsigned> OffsetToBlock;
};

/// Partition \p Decoded into a \c Cfg.
///
/// Block leaders are the first instruction, the instruction after every
/// terminator, every resolvable direct-branch target, and any \p ExtraLeaders
/// that fall inside the range (e.g. externally-known kernel entry offsets).
/// Edges are added from each block's last instruction: a direct branch adds
/// its resolved target (plus a fall-through for conditional branches); a
/// program terminator, return, or indirect branch adds none; anything else
/// falls through to the next block. Unresolved direct targets add no edge
/// (fail-open) rather than aborting, since this analysis is advisory.
///
/// \p LS supplies the MCInstrAnalysis and cached opcodes used for
/// classification; \p Decoded must be ordered by ascending offset.
Cfg buildCfg(llvm::ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
             llvm::ArrayRef<uint64_t> ExtraLeaders = {});

/// Reverse-post-order block indices for \p Graph.
///
/// When \p Scope is empty every block is included and used as a DFS root (so
/// unreachable blocks still appear). Otherwise traversal is restricted to the
/// blocks in \p Scope and rooted at them, letting a caller order one kernel's
/// blocks without walking into unrelated decoded code. Edges leaving the scope
/// are ignored.
std::vector<unsigned> reversePostOrder(const Cfg &Graph,
                                       llvm::ArrayRef<unsigned> Scope = {});

} // namespace reglive
} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_CFG_H
