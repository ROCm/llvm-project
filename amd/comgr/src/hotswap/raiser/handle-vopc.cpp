//===- handle-vopc.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/register-state.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>

using namespace llvm;

namespace COMGR::hotswap {

Error handleVOPC(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  ICmpInst::Predicate Pred;
  switch (Di.CanonOp) {
  case CanonicalOp::V_CMP_LT_I32:
    Pred = ICmpInst::ICMP_SLT;
    break;
  case CanonicalOp::V_CMP_EQ_I32:
    Pred = ICmpInst::ICMP_EQ;
    break;
  case CanonicalOp::V_CMP_LE_I32:
    Pred = ICmpInst::ICMP_SLE;
    break;
  case CanonicalOp::V_CMP_GT_I32:
    Pred = ICmpInst::ICMP_SGT;
    break;
  case CanonicalOp::V_CMP_NE_I32:
    Pred = ICmpInst::ICMP_NE;
    break;
  case CanonicalOp::V_CMP_GE_I32:
    Pred = ICmpInst::ICMP_SGE;
    break;
  case CanonicalOp::V_CMP_LT_U32:
    Pred = ICmpInst::ICMP_ULT;
    break;
  case CanonicalOp::V_CMP_EQ_U32:
    Pred = ICmpInst::ICMP_EQ;
    break;
  case CanonicalOp::V_CMP_LE_U32:
    Pred = ICmpInst::ICMP_ULE;
    break;
  case CanonicalOp::V_CMP_GT_U32:
    Pred = ICmpInst::ICMP_UGT;
    break;
  case CanonicalOp::V_CMP_NE_U32:
    Pred = ICmpInst::ICMP_NE;
    break;
  case CanonicalOp::V_CMP_GE_U32:
    Pred = ICmpInst::ICMP_UGE;
    break;
  default:
    return unsupported(Ctx, Di);
  }

  // A comparison in the plain VOPC encoding names no destination operand at
  // all: the lane mask goes to the condition registers the opcode implicitly
  // defines.
  if (Di.NumDefs != 0 || Op.nSrcs() != 2) {
    return unsupported(Ctx, Di, "expected two sources and no destination");
  }
  assert((Di.defsVcc() || Di.defsExec()) &&
         "a comparison that writes neither VCC nor EXEC has no result");

  Expected<Value *> Src0 = Op.src(0);
  if (!Src0) {
    return Src0.takeError();
  }
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1) {
    return Src1.takeError();
  }

  Value *Cmp = Ctx.B.CreateICmp(Pred, *Src0, *Src1);
  RegisterState &Regs = Ctx.registers();

  // Which condition registers a comparison writes is a property of the opcode
  // and of the source generation, and the decoder has already read it off the
  // implicit definitions: `v_cmp` writes VCC, `v_cmpx` writes EXEC, and before
  // GFX10 `v_cmpx` wrote both.
  if (Di.defsVcc()) {
    // VCC is held as the bit belonging to the lane doing the writing, so the
    // per-lane comparison is what it wants, cleared where the lane is inactive:
    // a comparison writes the whole mask whatever EXEC holds, and the bits of
    // the lanes EXEC masks off read back as zero. Masking rather than skipping
    // the write, since an inactive lane does not keep the bit it had.
    Value *Bit = Ctx.B.CreateAnd(Cmp, Regs.emitLaneActiveBit(), "vcc_bit");
    Regs.regFile().storeVCC(Ctx.B, Bit);
  }

  if (Di.defsExec()) {
    // Narrowing EXEC needs the answer for the whole wave, not for the writing
    // lane: EXEC is one mask that every lane reads, and a sign-extended
    // per-lane bit would give each lane a private EXEC that disagrees with its
    // neighbors. The ballot is what turns the per-lane comparison into that one
    // mask, and taking it at the EXEC storage width keeps the AND and the store
    // in a single type.
    Value *Mask = Ctx.Projection.ballotI1ToWidth(
        Ctx.B, Cmp, Ctx.Projection.execStorageTy(), "cmpx_ballot");
    Value *CurrentExec = Regs.regFile().loadExec(Ctx.B);
    // The AND is also what clears the bits of the lanes that were inactive,
    // which the ballot took the comparison of regardless.
    Value *NarrowedExec = Ctx.B.CreateAnd(CurrentExec, Mask, "cmpx_exec");
    // Through RegisterState rather than the register file: the lane-active bit
    // every following per-lane write is predicated on was derived from the
    // EXEC being replaced and has to go with it.
    Regs.storeExec(NarrowedExec);
  }

  return Error::success();
}

} // namespace COMGR::hotswap
