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

  default:
    break;
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

} // namespace COMGR::hotswap
