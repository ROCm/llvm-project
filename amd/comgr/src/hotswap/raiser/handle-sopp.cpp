//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-mc-tables.h"

#include "llvm/IR/IntrinsicsAMDGPU.h"

#include <cassert>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Emit Wait with an immediate encoding its counter thresholds.
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

  static constexpr Intrinsic::ID KSplitWaits[] = {
      Intrinsic::amdgcn_s_wait_loadcnt, Intrinsic::amdgcn_s_wait_storecnt,
      Intrinsic::amdgcn_s_wait_dscnt, Intrinsic::amdgcn_s_wait_kmcnt};
  for (Intrinsic::ID Wait : KSplitWaits)
    emitWait(Ctx, Wait, B.getInt16(0));
}

// Raise an instruction that changes wave priority only when the source and
// target use the same priority model. The dispatch-time system priority is
// unavailable, so different models cannot be proven to preserve wave ordering.
Error raiseWavePriority(RaiseContext &Ctx, const DecodedInst &Di) {
  if (Ctx.Projection.sourceIsa().wavePriorityModel() !=
      Ctx.Projection.targetIsa().wavePriorityModel())
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedWavePriority,
        strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "source wave priority is not representable on a target that composes "
        "it with the system priority differently");

  int16_t ImmIdx = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                      AMDGPU::OpName::simm16);
  assert(ImmIdx >= 0 && "every priority write encodes simm16");
  std::optional<int64_t> Imm = evalOperandAsConst(Di.Inst, ImmIdx);
  assert(Imm && "simm16 of a priority write is always an immediate");

  IRBuilder<> &B = Ctx.B;
  Intrinsic::ID Id = Di.CanonOp == CanonicalOp::S_SETPRIO
                         ? Intrinsic::amdgcn_s_setprio
                         : Intrinsic::amdgcn_s_setprio_inc_wg;
  B.CreateIntrinsic(Id, {}, {B.getInt16(static_cast<uint16_t>(*Imm))});
  return Error::success();
}

} // namespace

Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di, OperandResolver &) {
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

  case CanonicalOp::S_SETPRIO:
  case CanonicalOp::S_SETPRIO_INC_WG:
    return raiseWavePriority(Ctx, Di);

  // The sleep instructions may rely on s_wakeup for release, but s_wakeup has
  // no corresponding LLVM intrinsic. Refuse all three rather than emit a sleep
  // that may never end.
  case CanonicalOp::S_SLEEP:
  case CanonicalOp::S_MONITOR_SLEEP:
  case CanonicalOp::S_WAKEUP:
    return RaiseFailure::atInstruction(RaiseFailureReason::UnsupportedOpcode,
                                       strippedMnemonic(Ctx.MC, Di.Inst),
                                       Di.Offset,
                                       formatName(Di.TargetSpecificFlags));

  // Drop target-specific scheduling, instrumentation, cache-maintenance, and
  // padding instructions; none changes program state represented in raised IR.
  case CanonicalOp::S_NOP:
  case CanonicalOp::S_CLAUSE:
  case CanonicalOp::S_DELAY_ALU:
  case CanonicalOp::S_CODE_END:
  case CanonicalOp::S_INCPERFLEVEL:
  case CanonicalOp::S_DECPERFLEVEL:
  case CanonicalOp::S_TTRACEDATA:
  case CanonicalOp::S_TTRACEDATA_IMM:
  case CanonicalOp::S_ICACHE_INV:
    return Error::success();

  default:
    break;
  }

  return unsupported(Ctx, Di);
}

} // namespace COMGR::hotswap
