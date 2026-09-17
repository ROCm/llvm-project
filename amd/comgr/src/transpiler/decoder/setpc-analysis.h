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

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

namespace COMGR::transpiler {

struct MCState;

// Most source offsets one register-indirect transfer is allowed to reach
// before the analysis stops enumerating and refuses it. A dispatcher written
// by a compiler stays far below this, so a wider fan-out is more likely a
// value the analysis is mistaken about than a table it can state.
constexpr size_t kMaxSetPcTargets = 16;

// What one register-indirect control transfer was found to do.
struct SetPcSite {
  enum class Kind {
    // Control reaches exactly one source offset, which the analysis computed
    // from the program-counter capture and displacement feeding the transfer.
    Direct,
    // Control reaches one of several source offsets, each written into the
    // pair by a different path leading to the transfer.
    Enumerated,
    // Nothing the analysis models says where control goes.
    Unresolvable,
  };

  Kind SiteKind = Kind::Unresolvable;
  // Source offset control reaches. Meaningful for Direct.
  uint64_t DirectTarget = 0;
  // Source offsets control reaches, in ascending order. Meaningful for
  // Enumerated, where it holds at least two.
  llvm::SmallVector<uint64_t, 4> Targets;
  // Why the site could not be resolved, phrased to follow the mnemonic in a
  // refusal. Meaningful for Unresolvable.
  std::string RefusalReason;
};

// Where the register-indirect control transfers of one decoded kernel lead.
struct SetPcAnalysis {
  // One entry per s_set_pc_i64 and s_swap_pc_i64, keyed by its source offset.
  llvm::DenseMap<uint64_t, SetPcSite> Sites;
  // Source offsets that lead a block because of a transfer classified above,
  // over and above the block starts the decode already found. Every one of
  // them is the offset of a decoded instruction.
  llvm::DenseSet<uint64_t> ExtraBlockStarts;
  // Source offsets a transfer reaches that no decoded instruction starts at,
  // which refused the transfers reaching them. A caller that can widen its
  // decode to cover one of these and ask again turns that refusal into a
  // classified transfer.
  llvm::DenseSet<uint64_t> UndecodedTargets;
};

// Classify every register-indirect control transfer in `Insts`, which must be
// in source order, and report the block starts those transfers imply.
//
// The value a transfer reads is tracked within the block that makes it: a
// program-counter capture starts a chain, a constant displacement added to it
// carries the chain along, a call writes the offset it returns to, and any
// other write to the pair ends it. A transfer reading a completed chain
// reaches the offset that chain names.
//
// A transfer whose block leaves the pair alone reads what the paths reaching
// that block left in it, which a forward dataflow over the recovered blocks
// collects: one offset over every path is a plain branch, several are a
// dispatch over all of them, and a path that leaves no offset in the pair at
// all refuses the site rather than narrowing it to the paths that did.
//
// `BlockStarts` is the block-start set of the same decode. It is read, not
// written: the offsets the transfers add are reported separately so the caller
// can order the merge against the rest of its decode. `EntryOffset` is where
// control enters, which need not be the lowest offset in `Insts`: a callee
// followed into the decode may sit below its caller.
llvm::Expected<SetPcAnalysis>
analyzeSetPc(llvm::ArrayRef<DecodedInst> Insts,
             const std::set<uint64_t> &BlockStarts, uint64_t EntryOffset,
             const MCState &Mc);

} // namespace COMGR::transpiler

#endif
