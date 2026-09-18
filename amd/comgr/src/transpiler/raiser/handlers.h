//===- handlers.h - Transpiler --------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_HANDLERS_H
#define TRANSPILER_HANDLERS_H

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/raise_failure.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Error.h"

namespace COMGR::transpiler {

// Return a structured refusal for an unsupported instruction form.
inline llvm::Error unsupported(const RaiseContext &Ctx, const DecodedInst &Di,
                               const llvm::Twine &Detail = {}) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags), Detail);
}

// Wait for every memory counter the target tracks, as one sequentially
// consistent agent-scope fence.
//
// Counter identities do not correspond across ISA families and no wait
// intrinsic exists on all of them, so the fence stands in for whichever
// counter the source named and the backend expands it for the target. The
// source's count is dropped along with the identity, a count naming a position
// in an issue order that raising does not preserve. Agent is the weakest scope
// that still expands to a wait everywhere: a narrower scope drops the wait on a
// target whose caches already order that scope, which suits a fence pairing
// with another thread but not a counter, which only has to have retired.
inline void emitMemoryWaitAll(RaiseContext &Ctx) {
  llvm::IRBuilder<> &B = Ctx.B;
  B.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,
                B.getContext().getOrInsertSyncScopeID("agent"));
}

// Lower one instruction of the format the handler is named for, emitting into
// `Ctx`'s builder and reading its operands through `Op`. The raiser runs the
// first handler whose format bit matches and only that one, so a handler that
// does not recognize the opcode returns a `RaiseFailure` rather than declining:
// no later handler gets the chance to claim it.
llvm::Error handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
llvm::Error handleSOP2(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
llvm::Error handleSOPC(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
llvm::Error handleSOPK(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
llvm::Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
// Translate supported SMEM loads or return a structured refusal.
llvm::Error handleSMEM(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
/// Translate supported VGLOBAL memory accesses, or return a structured refusal.
llvm::Error handleVGLOBAL(RaiseContext &Ctx, const DecodedInst &Di,
                          OperandResolver &Op);
/// Raise raw unformatted buffer loads and stores, or return a refusal.
llvm::Error handleMUBUF(RaiseContext &Context, const DecodedInst &Instruction);
/// Raise direct VGPR LDS loads using AMDHSA's unaligned access mode.
/// Active accesses must lie wholly within the workgroup's LDS allocation.
llvm::Error handleDS(RaiseContext &Context, const DecodedInst &Instruction);
/// Translate a supported plain VOP1 instruction, or return a structured
/// refusal.
llvm::Error handleVOP1(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
/// Translate a supported plain VOP2 instruction, or return a structured
/// refusal.
llvm::Error handleVOP2(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
/// Translate a supported plain VOP3 instruction, or return a structured
/// refusal.
llvm::Error handleVOP3(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);
/// Translate a supported packed VOP3 instruction, or return a structured
/// refusal.
llvm::Error handleVOP3P(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op);
/// Translate both components of a VOPD packet. Both halves read the register
/// state that preceded the packet; their writes commit together afterwards.
llvm::Error handleVOPD(RaiseContext &Ctx, const DecodedInst &Di);
/// Translate a supported plain VOPC comparison into the condition registers
/// the opcode writes, or return a structured refusal.
llvm::Error handleVOPC(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op);

} // namespace COMGR::transpiler

#endif
