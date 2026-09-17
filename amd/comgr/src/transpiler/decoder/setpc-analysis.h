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

#include <cstdint>
#include <set>
#include <string>

namespace COMGR::transpiler {

struct MCState;

// What one register-indirect control transfer was found to do.
struct SetPcSite {
  enum class Kind {
    // Control reaches exactly one source offset, which the analysis computed
    // from the program-counter capture and displacement feeding the transfer.
    Direct,
    // Nothing the analysis models says where control goes.
    Unresolvable,
  };

  Kind SiteKind = Kind::Unresolvable;
  // Source offset control reaches. Meaningful for Direct.
  uint64_t DirectTarget = 0;
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
};

// Classify every register-indirect control transfer in `Insts`, which must be
// in source order, and report the block starts those transfers imply.
//
// The value a transfer reads is tracked within the block that makes it: a
// program-counter capture starts a chain, a constant displacement added to it
// carries the chain along, and any other write to the pair ends it. A transfer
// reading a completed chain reaches the offset the chain names, and one
// reading anything else is left unresolvable with the reason why.
//
// `BlockStarts` is the block-start set of the same decode. It is read, not
// written: the offsets the transfers add are reported separately so the caller
// can order the merge against the rest of its decode.
SetPcAnalysis analyzeSetPc(llvm::ArrayRef<DecodedInst> Insts,
                           const std::set<uint64_t> &BlockStarts,
                           const MCState &Mc);

} // namespace COMGR::transpiler

#endif
