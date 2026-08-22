//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-mc-tables.h"
#include "hotswap/decoder/decode.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/MC/MCSubtargetInfo.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

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
void emitMemoryWaitAll(RaiseContext &Ctx) {
  IRBuilder<> &B = Ctx.B;
  B.CreateFence(AtomicOrdering::SequentiallyConsistent,
                B.getContext().getOrInsertSyncScopeID("agent"));
}

// Raise a wave priority write to the matching intrinsic. Refuse a source that
// composes the priority with a dispatch-time system priority, which is not
// available to the raise, leaving the resulting wave ordering unreproducible.
Error raiseWavePriority(RaiseContext &Ctx, const DecodedInst &Di) {
  if (Ctx.Projection.SourceSTI.hasFeature(AMDGPU::FeatureGFX1250Insts))
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedWavePriority,
        strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "source wave priority composes with a dispatch-time system priority "
        "that is not available to the raise");

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

// Drop a sleep that ends after a bounded number of clocks, and refuse one that
// ends on an external event. A bounded sleep leaves a wave that stalled for
// zero cycles, a schedule the source already had to tolerate. An unbounded one
// runs until another wave wakes it, so it is the wave's forward progress that
// depends on it, not just the wave's timing.
Error raiseSleep(RaiseContext &Ctx, const DecodedInst &Di) {
  int16_t ImmIdx = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                      AMDGPU::OpName::simm16);
  assert(ImmIdx >= 0 && "every sleep encodes simm16");
  std::optional<int64_t> Imm = evalOperandAsConst(Di.Inst, ImmIdx);
  assert(Imm && "simm16 of a sleep is always an immediate");

  // SIMM16[15] selects sleep-forever; SIMM16[6:0] holds the clock count.
  constexpr int64_t SleepForever = 0x8000;
  if (*Imm & SleepForever)
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedSleepForever,
        strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "sleep ends on a wakeup, trap or kill that the raise does not "
        "reproduce");

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
  case CanonicalOp::S_WAIT_SAMPLECNT:
  case CanonicalOp::S_WAIT_BVHCNT:
  case CanonicalOp::S_WAIT_EVENT:
  case CanonicalOp::S_WAIT_LOADCNT_DSCNT:
  case CanonicalOp::S_WAIT_STORECNT_DSCNT:
  case CanonicalOp::S_WAIT_IDLE:
    emitMemoryWaitAll(Ctx);
    return Error::success();

  // No asynchronous transfer or tensor operation raises, so a kernel that
  // raises has none of that work in flight for these to wait on.
  case CanonicalOp::S_WAIT_ASYNCCNT:
  case CanonicalOp::S_WAIT_TENSORCNT:
    return Error::success();

  // XCNT tracks address translation and the ALU counters track register
  // hazards; both stop a later instruction from overwriting a register an
  // earlier one still needs. Where such a wait belongs depends on the register
  // assignment, which raising discards and the backend remakes.
  case CanonicalOp::S_WAIT_XCNT:
  case CanonicalOp::S_WAIT_ALU:
    return Error::success();

  case CanonicalOp::S_SETPRIO:
  case CanonicalOp::S_SETPRIO_INC_WG:
    return raiseWavePriority(Ctx, Di);

  case CanonicalOp::S_SLEEP:
  case CanonicalOp::S_MONITOR_SLEEP:
    return raiseSleep(Ctx, Di);

  // None of these changes program state the raised IR represents. A wakeup
  // releases a wave sleeping until an external event, and such a sleep does
  // not raise, so no wave of a raised kernel is waiting for one.
  case CanonicalOp::S_NOP:
  case CanonicalOp::S_WAKEUP:
  case CanonicalOp::S_CLAUSE:
  case CanonicalOp::S_DELAY_ALU:
  case CanonicalOp::S_CODE_END:
  case CanonicalOp::S_INCPERFLEVEL:
  case CanonicalOp::S_DECPERFLEVEL:
  case CanonicalOp::S_TTRACEDATA:
  case CanonicalOp::S_TTRACEDATA_IMM:
  case CanonicalOp::S_ICACHE_INV:
    return Error::success();

  // The branches. A conditional one falls through to the block the instruction
  // after it leads. Its condition is wave-level: SCC as written, and execz and
  // vccz as the emptiness of the mask the source wave holding this target lane
  // sees.
  case CanonicalOp::S_BRANCH:
  case CanonicalOp::S_CBRANCH_SCC0:
  case CanonicalOp::S_CBRANCH_SCC1:
  case CanonicalOp::S_CBRANCH_VCCZ:
  case CanonicalOp::S_CBRANCH_VCCNZ:
  case CanonicalOp::S_CBRANCH_EXECZ:
  case CanonicalOp::S_CBRANCH_EXECNZ: {
    Expected<uint64_t> Target = soppBranchTarget(Di);
    if (!Target)
      return Target.takeError();
    BasicBlock *TakenBb = Ctx.lookupBB(*Target);
    if (Di.CanonOp == CanonicalOp::S_BRANCH) {
      Ctx.B.CreateBr(TakenBb);
      return Error::success();
    }

    RegisterState &Regs = Ctx.registers();
    Value *Taken = nullptr;
    if (Di.CanonOp == CanonicalOp::S_CBRANCH_SCC0)
      Taken = Ctx.B.CreateNot(Regs.regFile().loadSCC(Ctx.B), "scc0");
    else if (Di.CanonOp == CanonicalOp::S_CBRANCH_SCC1)
      Taken = Regs.regFile().loadSCC(Ctx.B);
    else if (Di.CanonOp == CanonicalOp::S_CBRANCH_VCCZ)
      Taken = Regs.emitVccIsZero();
    else if (Di.CanonOp == CanonicalOp::S_CBRANCH_VCCNZ)
      Taken = Ctx.B.CreateNot(Regs.emitVccIsZero(), "vccnz");
    else if (Di.CanonOp == CanonicalOp::S_CBRANCH_EXECZ)
      Taken = Regs.emitExecIsZero();
    else {
      assert(Di.CanonOp == CanonicalOp::S_CBRANCH_EXECNZ &&
             "unhandled SOPP conditional branch");
      Taken = Ctx.B.CreateNot(Regs.emitExecIsZero(), "execnz");
    }

    Ctx.B.CreateCondBr(Taken, TakenBb,
                       Ctx.lookupBB(Di.Offset + Di.sizeInBytes()));
    return Error::success();
  }

  default:
    break;
  }

  return unsupported(Ctx, Di);
}

} // namespace COMGR::hotswap
