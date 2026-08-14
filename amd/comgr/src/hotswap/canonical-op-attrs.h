//===- canonical-op-attrs.h - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_SEM_OP_ATTRS_H
#define HOTSWAP_TRANSPILER_SEM_OP_ATTRS_H

#include "canonical-op.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class MCInstrInfo;
} // namespace llvm

namespace COMGR::hotswap {

// Per-CanonicalOp metadata that the raiser cross-references at its pre-
// translation gates. Grows incrementally as new per-writer predicates
// are needed (today only `routesExecThroughStoreExec`; the wave-size-
// obliviousness predicate from hotswap/docs/wave-size-translation.md
// sec. 7's decision procedure will attach here too).
//
// Default-constructed attrs mean "no declared guarantees" -- the gate
// aborts for any EXEC-writer encountered that does not have its
// `routesExecThroughStoreExec` bit set, so the default is fail-closed.
struct CanonicalOpAttrs {
  // The CanonicalOp's handler routes any write to ParsedReg::EXEC through
  // `regs.storeExec` -- either directly or via `writeReg{32,64,ExecWidth}`
  // which dispatch EXEC->storeExec. Required for any CanonicalOp whose
  // MCInstrDesc declares EXEC as a def (the startup invariant in
  // `verifyExecAttrCoverage`); also required to pass the per-kernel
  // Phase 1.5 SPE gate when an explicit `exec_lo`/`exec_hi` operand
  // makes the instruction a runtime EXEC-writer.
  bool RoutesExecThroughStoreExec = false;
};

// Pair of (CanonicalOp, attrs) registered by a handler file. The
// per-handler-file registrations below aggregate into the central
// lookup table on first use.
struct CanonicalOpAttrSpec {
  CanonicalOp Op;
  CanonicalOpAttrs Attrs;
};

// Per-handler-TU registrations. Each handler that claims SemOps with
// non-default attrs exposes a getter returning its static array of
// specs. Adding a new EXEC-writing CanonicalOp therefore requires declaring
// the attrs in the handler file that implements it -- the attribute
// registration moves with the handler.
//
// `theTable()` in `canonical-op-attrs.cpp` calls every getter declared here
// at first-use to materialise the combined table; adding a new getter
// is a two-line change (function definition + one line in
// `canonical-op-attrs.cpp` aggregating it).
llvm::ArrayRef<CanonicalOpAttrSpec> getHandlerSOP1Attrs();
llvm::ArrayRef<CanonicalOpAttrSpec> getHandlerSOP2Attrs();
llvm::ArrayRef<CanonicalOpAttrSpec> getHandlerValuVcmpAttrs();

// O(1) lookup keyed on CanonicalOp. Returns a default-constructed
// CanonicalOpAttrs for SemOps that have no declared attrs (fail-closed).
const CanonicalOpAttrs &getCanonicalOpAttrs(CanonicalOp Op);

// Startup invariant: for every MC opcode whose MCInstrDesc declares
// EXEC (or EXEC_LO/EXEC_HI) as an implicit def, the CanonicalOp it maps to
// must have `routesExecThroughStoreExec = true`. Catches the case where a
// new EXEC-writing opcode lands in LLVM and gets mapped to a CanonicalOp
// whose handler we have not yet audited for SPE.
// Explicit-operand EXEC writes (e.g. `s_mov_b32 exec_lo, s2`) remain
// the Phase 1.5 per-kernel gate's responsibility since they depend on
// runtime operand values.
//
// Called once at raiser init from the same slot that runs
// `verifyMFMACoverage`.
llvm::Error verifyExecAttrCoverage(const llvm::MCInstrInfo &MCII);

} // namespace COMGR::hotswap

#endif
