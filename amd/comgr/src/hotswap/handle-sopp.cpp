//===- handle-sopp.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

#include "SIDefines.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/MathExtras.h"

#include <climits>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

bool computeSoppTarget(const DecodedInst &Di, uint64_t &Target) {
  int64_t Raw = Di.getImm(0);
  int64_t BrOff = SignExtend64<16>(static_cast<uint64_t>(Raw));
  if (Di.Offset > UINT64_MAX - 4)
    return false;
  uint64_t Base = Di.Offset + 4;
  if (BrOff < 0) {
    uint64_t Back = static_cast<uint64_t>(-BrOff) * 4;
    if (Back > Base)
      return false;
    Target = Base - Back;
    return true;
  }
  uint64_t Forward = static_cast<uint64_t>(BrOff) * 4;
  if (Forward > UINT64_MAX - Base)
    return false;
  Target = Base + Forward;
  return true;
}

BasicBlock *lookupDecodedBB(RaiseContext &Ctx, const DecodedInst &Di,
                            uint64_t Addr, const llvm::Twine &Role,
                            HandlerResult &Hr) {
  auto It = Ctx.OffsetToBb.find(Addr);
  if (It != Ctx.OffsetToBb.end())
    return It->second;
  // Branch targets should already have been admitted by decode/CFG recovery and
  // materialised in OffsetToBb. Do not call RaiseContext::lookupBB here: that
  // helper creates fallback blocks for historical recovery paths, but a missing
  // SOPP target now means either control left the selected kernel extent or CFG
  // recovery failed to decode an in-extent target.
  if (Addr < Ctx.KernelStartOffset ||
      (Ctx.KernelEndOffset != 0 && Addr >= Ctx.KernelEndOffset)) {
    Hr.Failure = RaiseFailure::kernelBoundaryViolation(
        Ctx.Kernel->getName(), Addr,
        Twine(Role) + " target is outside the selected kernel extent");
    return nullptr;
  }
  Hr.Failure = RaiseFailure::unsupportedInstructionForm(
      Di, "SOPP",
      Twine(Role) + " target 0x" + Twine::utohexstr(Addr) +
          " is inside the selected kernel extent but was not decoded");
  return nullptr;
}

BasicBlock *lookupFallthroughBB(RaiseContext &Ctx, const DecodedInst &Di,
                                llvm::StringRef Role, HandlerResult &Hr) {
  if (Di.Size > UINT64_MAX - Di.Offset) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "SOPP", Twine(Role) + " fallthrough overflows source offset");
    return nullptr;
  }
  return lookupDecodedBB(Ctx, Di, Di.Offset + Di.Size,
                         Twine(Role) + " fallthrough", Hr);
}

} // namespace

HandlerResult handleSOPP(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op) {
  (void)Op;
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  if (Sop == CanonicalOp::S_ENDPGM) {
    if (Ctx.ThreadLoopLatch)
      Ctx.B.CreateBr(Ctx.ThreadLoopLatch);
    else
      Ctx.B.CreateRetVoid();
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BRANCH) {
    uint64_t Target = 0;
    if (!computeSoppTarget(Di, Target)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOPP", "s_branch target overflows source offset arithmetic");
      return Hr;
    }
    BasicBlock *TargetBb = lookupDecodedBB(Ctx, Di, Target, "s_branch", Hr);
    if (!TargetBb)
      return Hr;
    Ctx.B.CreateBr(TargetBb);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CBRANCH_EXECZ || Sop == CanonicalOp::S_CBRANCH_EXECNZ) {
    uint64_t Target = 0;
    if (!computeSoppTarget(Di, Target)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOPP", "s_cbranch_exec target overflows source offset arithmetic");
      return Hr;
    }
    BasicBlock *TargetBb = lookupDecodedBB(Ctx, Di, Target,
                                           "s_cbranch_exec", Hr);
    if (!TargetBb)
      return Hr;
    BasicBlock *FallthroughBb = lookupFallthroughBB(
        Ctx, Di, "s_cbranch_exec", Hr);
    if (!FallthroughBb)
      return Hr;
    Value *ExecVal = Ctx.Regs.loadExec(Ctx.B);
    Value *IsZero = Ctx.B.CreateICmpEQ(
        ExecVal, Constant::getNullValue(Ctx.Regs.ExecTy), "exec_is_zero");
    if (Sop == CanonicalOp::S_CBRANCH_EXECZ)
      Ctx.B.CreateCondBr(IsZero, TargetBb, FallthroughBb);
    else
      Ctx.B.CreateCondBr(Ctx.B.CreateNot(IsZero, "exec_nz"), TargetBb,
                         FallthroughBb);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CBRANCH_SCC0 || Sop == CanonicalOp::S_CBRANCH_SCC1) {
    uint64_t Target = 0;
    if (!computeSoppTarget(Di, Target)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOPP", "s_cbranch_scc target overflows source offset arithmetic");
      return Hr;
    }
    BasicBlock *TargetBb =
        lookupDecodedBB(Ctx, Di, Target, "s_cbranch_scc", Hr);
    if (!TargetBb)
      return Hr;
    BasicBlock *FallthroughBb = lookupFallthroughBB(
        Ctx, Di, "s_cbranch_scc", Hr);
    if (!FallthroughBb)
      return Hr;
    Value *SccV = Ctx.Regs.loadSCC(Ctx.B);
    if (Sop == CanonicalOp::S_CBRANCH_SCC0)
      SccV = Ctx.B.CreateNot(SccV, "not_scc");
    Ctx.B.CreateCondBr(SccV, TargetBb, FallthroughBb);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_CBRANCH_VCCNZ || Sop == CanonicalOp::S_CBRANCH_VCCZ) {
    uint64_t Target = 0;
    if (!computeSoppTarget(Di, Target)) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOPP", "s_cbranch_vcc target overflows source offset arithmetic");
      return Hr;
    }
    BasicBlock *TargetBb =
        lookupDecodedBB(Ctx, Di, Target, "s_cbranch_vcc", Hr);
    if (!TargetBb)
      return Hr;
    BasicBlock *FallthroughBb = lookupFallthroughBB(
        Ctx, Di, "s_cbranch_vcc", Hr);
    if (!FallthroughBb)
      return Hr;
    Value *VccV = Ctx.Regs.loadVCC(Ctx.B);
    if (Sop == CanonicalOp::S_CBRANCH_VCCZ)
      VccV = Ctx.B.CreateNot(VccV, "not_vcc");
    Ctx.B.CreateCondBr(VccV, TargetBb, FallthroughBb);
    Hr.Handled = true;
    return Hr;
  }
  // Barriers. GFX<12 uses a single `s_barrier`; GFX12+ splits it into a
  // separate signal and wait (both SOPP in this format). We model signal as
  // a no-op (the cross-wave rendezvous happens at the wait) and wait (or the
  // legacy unified barrier) as a full LLVM `amdgcn.s.barrier` call.
  if (Sop == CanonicalOp::S_BARRIER || Sop == CanonicalOp::S_BARRIER_WAIT) {
    Function *BarrierFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_s_barrier);
    Ctx.B.CreateCall(BarrierFn, {});
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_BARRIER_SIGNAL) {
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_SET_VGPR_MSB) {
    // Only the low 8 bits of the immediate carry runtime meaning; the high
    // 8 bits record the previous mode for compiler bookkeeping (see
    // AMDGPULowerVGPREncoding::setMode in LLVM).  The hardware ignores them.
    int64_t Imm = Di.getImm(0);
    Ctx.VgprMsBs = static_cast<uint8_t>(Imm & 0xFF);
    Hr.Handled = true;
    return Hr;
  }

  // Source wait counters are ordering operations, not decorative no-ops.  The
  // TensorDescriptor MXFP upcast emits `s_wait_dscnt 0` between LDS stores /
  // loads and split barriers; dropping it lets gfx942 reach `s_barrier` before
  // the prior DS operation is complete, producing sparse nondeterministic sign
  // flips after the LDS reshape.  Cross-target counter names do not map 1:1, so
  // use the conservative gfx942-compatible wait-all form.
  if (Sop == CanonicalOp::S_WAITCNT || Sop == CanonicalOp::S_WAIT_LOADCNT ||
      Sop == CanonicalOp::S_WAIT_STORECNT ||
      Sop == CanonicalOp::S_WAIT_KMCNT || Sop == CanonicalOp::S_WAIT_DSCNT ||
      Sop == CanonicalOp::S_WAIT_XCNT || Sop == CanonicalOp::S_WAIT_LOADCNT_DSCNT) {
    Function *WaitFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_s_waitcnt);
    Ctx.B.CreateCall(WaitFn, {Ctx.B.getInt32(0)});
    Hr.Handled = true;
    return Hr;
  }

  // gfx1250 async-memory wait counters. Explicit arm (rather than
  // falling through to the generic SOPP no-op catch-all below) so
  // this handler's surface documents the async/tensor cross-target
  // correctness argument alongside the other SOPP branches.
  //
  // Both counters track work in dedicated gfx1250 hardware units
  // (`ASYNCcnt`, `TENSORcnt`; programming_manual.pdf §4.9.9 and
  // §6 respectively) that do not exist on gfx942.  The source DMAs
  // they gate are emulated as synchronous `load`+`store` chains on
  // the cross-target arm (see `handle-flat.cpp`'s
  // `GLOBAL_LOAD_ASYNC_TO_LDS_B*` handler and `handle-vimage.cpp`'s
  // refusal -> future emulation for TENSOR ops), so by the time the
  // wait site is reached the underlying memory transfer has
  // already completed at the IR level.  IR dataflow from the
  // emulated `store` through subsequent LDS reads carries the
  // happens-before the native counter was enforcing; the backend
  // re-inserts the target-appropriate `s_waitcnt lgkmcnt(0)` on
  // gfx942 from that ordering constraint.
  //
  // On the same-target arm (gfx1250 -> gfx1250) this branch remains
  // emission-light for now: the async intrinsic's
  // `IntrInaccessibleMemOrArgMemOnly` annotation prevents reorder across the
  // wait site, while the asynchronous operation itself is what carries the
  // relevant memory dependency.  Do not merge this arm with the ordinary
  // wait-counter branch above unless the async/tensor counter semantics have a
  // target-independent wait-all lowering too.
  if (Sop == CanonicalOp::S_WAIT_ASYNCCNT || Sop == CanonicalOp::S_WAIT_TENSORCNT) {
    Hr.Handled = true;
    return Hr;
  }

  // s_sendmsg / s_sendmsghalt. Only INTERRUPT and DEALLOC_VGPRS are lifted;
  // every other ID refuses because the same SIMM16 aliases different messages
  // across generations and a blind pass-through would misencode on cross-target
  // lifts.
  if (Sop == CanonicalOp::S_SENDMSG || Sop == CanonicalOp::S_SENDMSGHALT) {
    int64_t Imm = Di.getImm(0);
    unsigned Simm16 = static_cast<unsigned>(Imm & 0xFFFF);
    bool IsInterrupt = Simm16 == AMDGPU::SendMsg::ID_INTERRUPT;
    bool IsDealloc = Simm16 == AMDGPU::SendMsg::ID_DEALLOC_VGPRS_GFX11Plus;

    if (!IsInterrupt && !IsDealloc) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SOPP",
          Twine("unsupported s_sendmsg SIMM16=0x") + Twine::utohexstr(Simm16) +
              "; only MSG_INTERRUPT (1) and MSG_DEALLOC_VGPRS (3) are lifted");
      return Hr;
    }

    // DEALLOC_VGPRS is a gfx11+ early-free hint; drop it where unsupported
    // (gfx942 frees implicitly at s_endpgm). INTERRUPT is portable everywhere.
    if (IsDealloc && !Ctx.TargetIsa.SupportsDeallocVgprs) {
      Hr.Handled = true;
      return Hr;
    }

    Intrinsic::ID IID = (Sop == CanonicalOp::S_SENDMSG)
                            ? Intrinsic::amdgcn_s_sendmsg
                            : Intrinsic::amdgcn_s_sendmsghalt;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, IID);
    ParsedReg M0Reg;
    M0Reg.RegKind = ParsedReg::M0;
    M0Reg.BaseIdx = 0;
    Value *M0Val = Ctx.Regs.readReg32(Ctx.B, M0Reg);
    Ctx.B.CreateCall(Fn, {Ctx.B.getInt32(Simm16), M0Val});
    Hr.Handled = true;
    return Hr;
  }

  // All other SOPP instructions (nop, scheduling hints, etc.) are no-ops.
  Hr.Handled = true;
  return Hr;
}

} // namespace COMGR::hotswap
