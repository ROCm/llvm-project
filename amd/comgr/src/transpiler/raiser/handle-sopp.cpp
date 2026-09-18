//===- handle-sopp.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/decode.h"

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

namespace COMGR::transpiler {

namespace {

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

  int16_t ImmIdx = COMGR::transpiler::getNamedOperandIdx(
      Di.Inst.getOpcode(), AMDGPU::OpName::simm16);
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
  int16_t ImmIdx = COMGR::transpiler::getNamedOperandIdx(
      Di.Inst.getOpcode(), AMDGPU::OpName::simm16);
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

// The rounding and denormal modes the raised kernel computes in, in the field
// layout each mode-setting opcode takes its immediate in. Nothing the raiser
// emits moves either of them.
constexpr int64_t KRaisedRoundMode =
    FP_ROUND_MODE_SP(FP_ROUND_ROUND_TO_NEAREST) |
    FP_ROUND_MODE_DP(FP_ROUND_ROUND_TO_NEAREST);
constexpr int64_t KRaisedDenormMode =
    FP_DENORM_FLUSH_NONE | (FP_DENORM_FLUSH_NONE << 2);

} // namespace

Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
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

  case CanonicalOp::S_SET_VGPR_MSB: {
    int16_t ImmediateIndex = COMGR::transpiler::getNamedOperandIdx(
        Di.Inst.getOpcode(), AMDGPU::OpName::simm16);
    assert(ImmediateIndex >= 0 && "s_set_vgpr_msb encodes simm16");
    std::optional<int64_t> Immediate =
        evalOperandAsConst(Di.Inst, ImmediateIndex);
    assert(Immediate && "s_set_vgpr_msb simm16 is always immediate");
    // SIMM16[15:8] records the preceding mode; only the low-byte fields select
    // the VGPR banks used by subsequent instructions.
    Ctx.registers().setVgprMsBs(static_cast<uint8_t>(*Immediate & UINT8_MAX));
    return Error::success();
  }

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

  // The release half of the barrier the source splits; the arrival is in SOP1.
  // The raise has one barrier, and it arrives and waits at once, so standing it
  // in here makes the wave arrive a second time, at a barrier every other wave
  // has to reach as well. Waves the source left free to run on are held there
  // instead, which can deadlock, so the release is refused too.
  case CanonicalOp::S_BARRIER_WAIT:
    return unsupported(Ctx, Di,
                       "waits on a barrier it does not arrive at here, and "
                       "the raise has only a barrier that also arrives");

  // Leaving takes the wave out of a named barrier's membership and reports in
  // SCC whether it was the last member out. The raise keeps no membership to
  // leave and so has nothing truthful to write to SCC.
  case CanonicalOp::S_BARRIER_LEAVE:
    return unsupported(Ctx, Di, "leaves a named barrier");

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

  // Halting stops the wave until a debugger resumes it and says nothing about
  // any register, and the intrinsic exists on every AMDGPU target, so the
  // immediate goes through as it stands.
  case CanonicalOp::S_SETHALT:
    Ctx.B.CreateIntrinsic(Ctx.B.getVoidTy(), Intrinsic::amdgcn_s_sethalt,
                          {Ctx.B.getInt32(Op.srcImm(0))});
    return Error::success();

  // Entering the trap handler means running code the source queue installed,
  // at an address the source wave holds, against state it set up. None of that
  // is reachable from the raised kernel, and a wave that traps and never
  // returns is not a kernel that ran.
  case CanonicalOp::S_TRAP:
    return unsupported(Ctx, Di,
                       "enters trap handler " + Twine(Op.srcImm(0)) +
                           ", which the raised kernel does not have");

  // Ends the wave expecting the context-save hardware to have taken its state
  // and something to restore it later. Raising this to a plain return would
  // claim the kernel finished when the source only paused it.
  case CanonicalOp::S_ENDPGM_SAVED:
    return unsupported(Ctx, Di,
                       "ends the wave for a context save nothing here resumes");

  // Accept only the immediate naming the mode the raised kernel is already in,
  // so that the float instructions after it compute under the mode the source
  // asked for. Any other immediate would leave the raised arithmetic rounding
  // or flushing differently from the source, which is a wrong answer rather
  // than a missing one.
  //
  // TODO: carry the requested mode instead of refusing it, by recording it and
  // applying it to the float instructions it reaches. That needs control- and
  // data-flow analysis to find those instructions, and block duplication where
  // one is reachable under two different modes.
  case CanonicalOp::S_ROUND_MODE:
    if (Op.srcImm(0) != KRaisedRoundMode)
      return unsupported(Ctx, Di,
                         "selects rounding mode " + Twine(Op.srcImm(0)) +
                             " rather than round-to-nearest-even, which is the "
                             "mode the raised kernel computes in");
    return Error::success();
  case CanonicalOp::S_DENORM_MODE:
    if (Op.srcImm(0) != KRaisedDenormMode)
      return unsupported(Ctx, Di,
                         "selects denormal mode " + Twine(Op.srcImm(0)) +
                             " rather than keeping denormals, which is what "
                             "the raised kernel computes with");
    return Error::success();

  // The same SIMM16 names different messages on different targets, and most of
  // them are a conversation between the source wave and hardware that is not
  // there to answer. Only the interrupt means the same thing everywhere.
  case CanonicalOp::S_SENDMSG:
  case CanonicalOp::S_SENDMSGHALT: {
    unsigned Simm16 = static_cast<unsigned>(Op.srcImm(0)) & 0xFFFF;
    bool IsHalt = Di.CanonOp == CanonicalOp::S_SENDMSGHALT;

    // The deallocation hint claims the wave is done with its VGPRs, which is
    // not true of the raised kernel where the source said it, and where it
    // does become true is for the target backend to settle. Dropping the send
    // is therefore the faithful reading; dropping the halt the halting
    // spelling also performs would not be. The id only means the deallocation
    // hint on a source that spells it that way -- an older one gives the same
    // bits to the geometry-shader completion message, which falls through to
    // the refusal below.
    if (Simm16 == AMDGPU::SendMsg::ID_DEALLOC_VGPRS_GFX11Plus &&
        Ctx.Projection.SourceSTI.hasFeature(AMDGPU::FeatureGFX11Insts)) {
      if (IsHalt)
        return unsupported(Ctx, Di,
                           "halts the wave alongside a VGPR deallocation the "
                           "raised kernel must not claim");
      return Error::success();
    }

    if (Simm16 != AMDGPU::SendMsg::ID_INTERRUPT)
      return unsupported(Ctx, Di,
                         "sends message 0x" + Twine::utohexstr(Simm16) +
                             ", and the interrupt is the only message that "
                             "means the same thing on every target");

    Ctx.B.CreateIntrinsic(Ctx.B.getVoidTy(),
                          IsHalt ? Intrinsic::amdgcn_s_sendmsghalt
                                 : Intrinsic::amdgcn_s_sendmsg,
                          {Ctx.B.getInt32(Simm16), Ctx.registers().readM0()});
    return Error::success();
  }

  default:
    break;
  }

  return unsupported(Ctx, Di);
}

} // namespace COMGR::transpiler
