//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/decode.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise_failure.h"

// AMDGPU target-private headers.
#include "SIDefines.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdint>

using namespace llvm;

namespace COMGR::hotswap {

// Refuse Di as a barrier the raised kernel has no barrier to state it with.
static Error refuseBarrier(RaiseContext &Ctx, const DecodedInst &Di,
                           const Twine &Detail) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags), Detail);
}

// Refuse Di as disposing of the source wave in a way nothing picks up again.
static Error refuseWaveDisposal(RaiseContext &Ctx, const DecodedInst &Di,
                                const Twine &Detail) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags), Detail);
}

// Refuse Di as asking for a floating-point mode the raised kernel does not
// run in.
static Error refuseFpMode(RaiseContext &Ctx, const DecodedInst &Di,
                          const Twine &Detail) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags), Detail);
}

// Refuse Di as sending a message whose meaning the raise does not carry.
static Error refuseMessage(RaiseContext &Ctx, const DecodedInst &Di,
                           const Twine &Detail) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags), Detail);
}

// The rounding and denormal modes the raised kernel computes in, in the field
// layout each mode-setting opcode takes its immediate in. Nothing the raiser
// emits moves either of them.
static constexpr int64_t KRaisedRoundMode =
    FP_ROUND_MODE_SP(FP_ROUND_ROUND_TO_NEAREST) |
    FP_ROUND_MODE_DP(FP_ROUND_ROUND_TO_NEAREST);
static constexpr int64_t KRaisedDenormMode =
    FP_DENORM_FLUSH_NONE | (FP_DENORM_FLUSH_NONE << 2);

Error handleSOPP(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::S_ENDPGM:
    Ctx.B.CreateRetVoid();
    return Error::success();

  // Wait counters carry ordering the raised kernel still needs: drop the
  // `s_wait_dscnt 0` a source kernel puts between an LDS store and a barrier
  // and the target reaches the barrier before the store retires. Which
  // hardware unit each counter tracks does not survive a change of target, and
  // no AMDGPU wait intrinsic is available on every one, so they all lower to
  // one agent-scope fence, which subsumes any of them and which the backend
  // expands into the target's own wait sequence.
  case CanonicalOp::S_WAITCNT_DEPCTR:
  case CanonicalOp::S_WAIT_IDLE:
  case CanonicalOp::S_WAIT_EVENT:
  case CanonicalOp::S_WAIT_LOADCNT:
  case CanonicalOp::S_WAIT_STORECNT:
  case CanonicalOp::S_WAIT_EXPCNT:
  case CanonicalOp::S_WAIT_XCNT:
  case CanonicalOp::S_WAIT_DSCNT:
  case CanonicalOp::S_WAIT_KMCNT:
  case CanonicalOp::S_WAIT_LOADCNT_DSCNT:
  case CanonicalOp::S_WAIT_STORECNT_DSCNT:
  case CanonicalOp::S_WAIT_ASYNCCNT:
  case CanonicalOp::S_WAIT_TENSORCNT:
    Ctx.B.CreateFence(AtomicOrdering::SequentiallyConsistent,
                      Ctx.B.getContext().getOrInsertSyncScopeID("agent"));
    return Error::success();

  // Hints and side channels no computed value can depend on: instruction issue
  // timing (nop, sleep, the ping that wakes a sleeper, clause, delay_alu),
  // wave priority, performance counters, the thread-trace stream, and an
  // instruction cache the raised kernel does not run out of. s_code_end is the
  // padding a shader buffer carries past its terminator. What the target wants
  // in their place is the backend's to insert.
  case CanonicalOp::S_NOP:
  case CanonicalOp::S_SLEEP:
  case CanonicalOp::S_MONITOR_SLEEP:
  case CanonicalOp::S_CLAUSE:
  case CanonicalOp::S_DELAY_ALU:
  case CanonicalOp::S_CODE_END:
  case CanonicalOp::S_WAKEUP:
  case CanonicalOp::S_SETPRIO:
  case CanonicalOp::S_SETPRIO_INC_WG:
  case CanonicalOp::S_INCPERFLEVEL:
  case CanonicalOp::S_DECPERFLEVEL:
  case CanonicalOp::S_TTRACEDATA:
  case CanonicalOp::S_TTRACEDATA_IMM:
  case CanonicalOp::S_ICACHE_INV:
    return Error::success();

  // The release half of the barrier the source splits into an arrival, in
  // SOP1, and a release, here. Waiting for the workgroup barrier is what
  // `llvm.amdgcn.s.barrier` does, along with arriving at it, and the extra
  // arrival costs the wave nothing: it has already arrived, at the source
  // instruction that told it to.
  case CanonicalOp::S_BARRIER_WAIT: {
    // The field is 16-bit signed, and a non-negative value does not select a
    // barrier by number: it waits on whichever one the wave joined last.
    int64_t BarrierId = SignExtend64<16>(Op.srcImm(0));
    if (BarrierId >= 0)
      return refuseBarrier(Ctx, Di,
                           "waits on the named barrier this wave joined last, "
                           "and the target has only the workgroup barrier");
    if (BarrierId != AMDGPU::Barrier::WORKGROUP)
      return refuseBarrier(Ctx, Di,
                           "names barrier " + Twine(BarrierId) +
                               " rather than the workgroup barrier, which is "
                               "the only barrier the target has");
    Ctx.B.CreateIntrinsic(Ctx.B.getVoidTy(), Intrinsic::amdgcn_s_barrier, {});
    return Error::success();
  }

  // Leaving takes the wave out of a named barrier's membership and reports in
  // SCC whether it was the last member out. The target keeps no membership to
  // leave and so has nothing truthful to write to SCC.
  case CanonicalOp::S_BARRIER_LEAVE:
    return refuseBarrier(Ctx, Di, "leaves a named barrier");

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
      Taken = Ctx.B.CreateNot(Regs.readScc(), "scc0");
    else if (Di.CanonOp == CanonicalOp::S_CBRANCH_SCC1)
      Taken = Regs.readScc();
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

  // Two bits per source operand class, each adding 256 to the VGPR numbers the
  // instructions after it name. Applying that renumbering is the raiser's job,
  // since it holds the register file those numbers index, so latch the
  // immediate rather than emit anything. Only the low byte selects; the high
  // byte records the selection in force before, which the hardware ignores.
  case CanonicalOp::S_SET_VGPR_MSB:
    Ctx.registers().setVgprMsBs(static_cast<uint8_t>(Op.srcImm(0)));
    return Error::success();

  // Entering the trap handler means running code the source queue installed,
  // at an address the source wave holds, against state it set up. None of that
  // is reachable from the raised kernel, and a wave that traps and never
  // returns is not a kernel that ran.
  case CanonicalOp::S_TRAP:
    return refuseWaveDisposal(Ctx, Di,
                              "enters trap handler " + Twine(Op.srcImm(0)) +
                                  ", which the raised kernel does not have");

  // Ends the wave expecting the context-save hardware to have taken its state
  // and something to restore it later. Raising this to a plain return would
  // claim the kernel finished when the source only paused it.
  case CanonicalOp::S_ENDPGM_SAVED:
    return refuseWaveDisposal(
        Ctx, Di, "ends the wave for a context save nothing here resumes");

  // Accept only the immediate naming the mode the raised kernel is already in,
  // so that the float instructions after it compute under the mode the source
  // asked for. Any other immediate would leave the raised arithmetic rounding
  // or flushing differently from the source, which is a wrong answer rather
  // than a missing one.
  case CanonicalOp::S_ROUND_MODE:
    if (Op.srcImm(0) != KRaisedRoundMode)
      return refuseFpMode(Ctx, Di,
                          "selects rounding mode " + Twine(Op.srcImm(0)) +
                              " rather than round-to-nearest-even, which is "
                              "the mode the raised kernel computes in");
    return Error::success();
  case CanonicalOp::S_DENORM_MODE:
    if (Op.srcImm(0) != KRaisedDenormMode)
      return refuseFpMode(Ctx, Di,
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
        Ctx.Projection.sourceIsa().hasVgprDeallocMessage()) {
      if (IsHalt)
        return refuseMessage(Ctx, Di,
                             "halts the wave alongside a VGPR deallocation "
                             "the raised kernel must not claim");
      return Error::success();
    }

    if (Simm16 != AMDGPU::SendMsg::ID_INTERRUPT)
      return refuseMessage(Ctx, Di,
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

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

} // namespace COMGR::hotswap
