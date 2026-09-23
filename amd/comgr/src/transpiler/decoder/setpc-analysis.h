//===- setpc-analysis.h - Transpiler --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_SETPC_ANALYSIS_H
#define TRANSPILER_SETPC_ANALYSIS_H

#include "transpiler/decoder/decoded-inst.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <set>
#include <string>

namespace COMGR::transpiler {

struct MCState;

// Where one indirect jump goes. An indirect jump is an s_set_pc_i64 or an
// s_swap_pc_i64: it moves the program counter to whatever a scalar register
// pair holds. s_swap_pc_i64 also writes the return address into a second
// register pair, so it is a call.
struct SetPcSite {
  // Source offsets the jump reaches, ascending and distinct. The analysis
  // resolved the jump when this is not empty.
  llvm::SmallVector<uint64_t> Targets;
  // Why the analysis could not say where the jump goes. Set when `Targets` is
  // empty. The text follows the mnemonic in a refusal message.
  std::string RefusalReason;

  bool isResolved() const { return !Targets.empty(); }
};

// Where the indirect jumps of one decoded kernel go.
struct SetPcAnalysis {
  // One entry per s_set_pc_i64 and s_swap_pc_i64, keyed by its source offset.
  llvm::DenseMap<uint64_t, SetPcSite> Sites;
  // Block starts the analysis found on top of the ones the decode gave it.
  // Every one is the offset of a decoded instruction. The caller must start a
  // block at each, because the analysis split its own walk there and its
  // answers only hold for that shape of control flow.
  llvm::DenseSet<uint64_t> ExtraBlockStarts;
};

// Work out where every indirect jump in `Insts` goes. `Insts` must be in
// source order.
//
// Within a block the analysis follows the value a jump reads: a program-counter
// capture starts a chain, adding a constant to it carries the chain along, a
// call writes the offset it returns to, and any other write ends the chain. A
// jump that reads a displaced chain goes to the offset that chain names.
//
// When a block does not write the register pair its jump reads, the jump reads
// what the
// paths into the block left there. A forward dataflow over the recovered blocks
// collects those offsets. One offset is a plain branch and several are a
// dispatch. If any path leaves the pair holding something the analysis cannot
// name, it refuses the jump instead of narrowing it to the paths that did name
// an offset.
//
// `BlockStarts` is the block-start set of the same decode. The analysis reads
// it and reports the offsets it adds separately, so the caller decides when to
// merge them into its own decode.
llvm::Expected<SetPcAnalysis>
analyzeSetPc(llvm::ArrayRef<DecodedInst> Insts,
             const std::set<uint64_t> &BlockStarts, const MCState &Mc);

} // namespace COMGR::transpiler

#endif
