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
#include <variant>

namespace COMGR::transpiler {

struct MCState;

// Why the analysis cannot say where a register-indirect control transfer
// leads. The raiser turns each of these into the refusal it reports.
enum class SetPcRefusal {
  // The transfer reads its target from something other than a scalar register
  // pair.
  NotARegisterPair,
  // The block wrote the pair, but never computed a source offset in it.
  PairWrittenWithoutOffset,
  // Nothing in the block ever gave the pair a source offset.
  PairNeverHeldOffset,
  // The pair names an offset that starts no decoded instruction.
  TargetNotAnInstruction,
};

// Control reaches exactly one source offset, which the analysis computed from
// the program-counter capture feeding the transfer plus whatever displacement
// was added to it.
struct SetPcDirect {
  uint64_t Target;
};

// Nothing the analysis models says where control goes.
struct SetPcUnresolvable {
  SetPcRefusal Why;
  // What the refusal speaks about: the low register of the source pair for
  // the two refusals that name a pair, and the computed source offset for
  // TargetNotAnInstruction. NotARegisterPair names nothing and leaves it zero.
  uint64_t Detail;
};

// What one register-indirect control transfer was found to do. A site is one
// or the other, never a half-filled mix of both.
using SetPcSite = std::variant<SetPcDirect, SetPcUnresolvable>;

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
// program-counter capture makes the pair name an offset, a constant
// displacement added to it makes the pair name another, and any other write to
// the pair drops what it named. A transfer that reads a pair naming an offset
// reaches that offset; one that reads anything else stays unresolvable, along
// with the reason why.
//
// `BlockStarts` is the block-start set of the same decode. It is read, not
// written: the offsets contributed by the transfers are reported separately so
// that the caller can order the merge against the rest of its decode.
SetPcAnalysis analyzeSetPc(llvm::ArrayRef<DecodedInst> Insts,
                           const std::set<uint64_t> &BlockStarts,
                           const MCState &Mc);

} // namespace COMGR::transpiler

#endif
