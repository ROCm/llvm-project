//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Bits of the `s_sleep` immediate that hold the sleep duration, per the CDNA3
// ISA guide's 64..8128 clock range.
constexpr int64_t KSleepDurationMask = 0x7f;

void emitWait(RaiseContext &Ctx, Intrinsic::ID Wait, Value *Count) {
  IRBuilder<> &B = Ctx.B;
  Module *M = B.GetInsertBlock()->getModule();
  B.CreateCall(Intrinsic::getOrInsertDeclaration(M, Wait), {Count});
}

// Wait for every memory counter the target tracks.
//
// Counter identities do not correspond across ISA families, so a memory wait
// takes the strongest memory wait the target offers rather than a per-counter
// translation, and the source's count is discarded with the identity: a count
// names a position in the source's issue order, which re-scheduling
// invalidates. Waiting for more than the source asked cannot break it; waiting
// for less can.
void emitMemoryWaitAll(RaiseContext &Ctx) {
  IRBuilder<> &B = Ctx.B;
  if (Ctx.Projection.targetIsa().hasCombinedWaitcnt()) {
    emitWait(Ctx, Intrinsic::amdgcn_s_waitcnt, B.getInt32(0));
    return;
  }

  // No EXPcnt wait: this path is reached only for gfx1250, which has no export
  // instructions, so nothing can be pending in the counter and S_WAIT_EXPCNT is
  // gated off the target.
  static constexpr Intrinsic::ID KSplitWaits[] = {
      Intrinsic::amdgcn_s_wait_loadcnt, Intrinsic::amdgcn_s_wait_storecnt,
      Intrinsic::amdgcn_s_wait_dscnt, Intrinsic::amdgcn_s_wait_kmcnt};
  for (Intrinsic::ID Wait : KSplitWaits)
    emitWait(Ctx, Wait, B.getInt16(0));
}

} // namespace

Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &) {
  switch (Di.CanonOp) {
  case CanonicalOp::S_ENDPGM:
    Ctx.B.CreateRetVoid();
    return Error::success();

  case CanonicalOp::S_WAITCNT:
  case CanonicalOp::S_WAIT_LOADCNT:
  case CanonicalOp::S_WAIT_STORECNT:
  case CanonicalOp::S_WAIT_DSCNT:
  case CanonicalOp::S_WAIT_KMCNT:
  case CanonicalOp::S_WAIT_EXPCNT:
  case CanonicalOp::S_WAIT_LOADCNT_DSCNT:
  case CanonicalOp::S_WAIT_STORECNT_DSCNT:
  case CanonicalOp::S_WAIT_IDLE:
    emitMemoryWaitAll(Ctx);
    return Error::success();

  // A target without asynchronous transfer or tensor units cannot have that
  // work in flight. A target that has them only receives such work from the
  // backend, which pairs its own wait with each operation it issues.
  case CanonicalOp::S_WAIT_ASYNCCNT:
  case CanonicalOp::S_WAIT_TENSORCNT:
    return Error::success();

  // XCNT counts memory operations awaiting address translation; the ALU
  // counters count register hazards. Both waits stop a later instruction from
  // overwriting a register an earlier one still needs, so where they belong
  // depends on the register assignment -- which raising discards and the
  // backend remakes.
  case CanonicalOp::S_WAIT_XCNT:
  case CanonicalOp::S_WAIT_ALU:
    return Error::success();

  // Scheduling hints carry no architectural state, and the backend emits its
  // own for whichever target it lowers to, so the source's must not survive.
  case CanonicalOp::S_NOP:
  case CanonicalOp::S_CLAUSE:
  case CanonicalOp::S_DELAY_ALU:
  case CanonicalOp::S_SETPRIO:
    return Error::success();

  // A bounded sleep is a hint like the rest: it self-terminates, so no program
  // can rest on the wave staying parked. Wider immediates select modes that do
  // not, and LLVM decodes none of the field -- a plain `i32imm` with no operand
  // class -- so the raiser refuses instead of assuming.
  case CanonicalOp::S_SLEEP: {
    std::optional<int64_t> Imm = evalOperandAsConst(Di.Inst, 0);
    if (Imm && (*Imm & ~KSleepDurationMask) == 0)
      return Error::success();
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "s_sleep immediate selects more than a sleep duration");
  }

  default:
    break;
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

} // namespace COMGR::hotswap
