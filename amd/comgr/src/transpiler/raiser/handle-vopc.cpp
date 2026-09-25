//===- handle-vopc.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/register-state.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <optional>

using namespace llvm;

namespace COMGR::transpiler {

std::optional<ICmpInst::Predicate>
getIntegerComparePredicate(CanonicalOp Opcode) {
  switch (Opcode) {
  case CanonicalOp::V_CMP_LT_I32:
    return ICmpInst::ICMP_SLT;
  case CanonicalOp::V_CMP_EQ_I32:
    return ICmpInst::ICMP_EQ;
  case CanonicalOp::V_CMP_LE_I32:
    return ICmpInst::ICMP_SLE;
  case CanonicalOp::V_CMP_GT_I32:
    return ICmpInst::ICMP_SGT;
  case CanonicalOp::V_CMP_NE_I32:
    return ICmpInst::ICMP_NE;
  case CanonicalOp::V_CMP_GE_I32:
    return ICmpInst::ICMP_SGE;
  case CanonicalOp::V_CMP_LT_U32:
    return ICmpInst::ICMP_ULT;
  case CanonicalOp::V_CMP_EQ_U32:
    return ICmpInst::ICMP_EQ;
  case CanonicalOp::V_CMP_LE_U32:
    return ICmpInst::ICMP_ULE;
  case CanonicalOp::V_CMP_GT_U32:
    return ICmpInst::ICMP_UGT;
  case CanonicalOp::V_CMP_NE_U32:
    return ICmpInst::ICMP_NE;
  case CanonicalOp::V_CMP_GE_U32:
    return ICmpInst::ICMP_UGE;
  default:
    return std::nullopt;
  }
}

Error raiseIntegerCompare32(RaiseContext &Ctx, const DecodedInst &Di,
                            OperandResolver &Op, ICmpInst::Predicate Predicate,
                            std::optional<ParsedReg> Destination) {
  if (Op.nSrcs() != 2 || Op.srcIdx(0) >= Di.numOperands() ||
      Op.srcIdx(1) >= Di.numOperands()) {
    return unsupported(Ctx, Di, "expected two comparison sources");
  }

  Expected<Value *> Src0 = Op.src(0);
  if (!Src0) {
    return Src0.takeError();
  }
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1) {
    return Src1.takeError();
  }

  Value *Cmp = Ctx.B.CreateICmp(Predicate, *Src0, *Src1);
  RegisterState &Regs = Ctx.registers();

  // Inactive lanes contribute zero even when their source values are poison.
  Value *Bit = Ctx.B.CreateSelect(Regs.emitLaneActiveBit(), Cmp,
                                  Ctx.B.getFalse(), "cmp_active");
  if (Destination && Destination->RegKind == ParsedReg::SGPR) {
    Value *Mask = Ctx.Projection.ballotI1ToWidth(
        Ctx.B, Bit, Ctx.Projection.sourceWaveMaskTy(), "cmp_mask");
    Regs.writeRegExecWidth(*Destination, Mask);
    Regs.recordWaveMaskI1(*Destination, Bit);
  }
  if (Di.defsVcc() || (Destination && Destination->RegKind == ParsedReg::VCC)) {
    Regs.regFile().storeVCC(Ctx.B, Bit);
  }
  if (Di.defsExec()) {
    Value *Mask = Ctx.Projection.ballotI1ToWidth(
        Ctx.B, Bit, Ctx.Projection.execStorageTy(), "cmpx_ballot");
    Value *CurrentExec = Regs.regFile().loadExec(Ctx.B);
    Value *NarrowedExec = Ctx.B.CreateAnd(CurrentExec, Mask, "cmpx_exec");
    Regs.storeExec(NarrowedExec);
  }
  return Error::success();
}

Error handleVOPC(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  if (Di.NumDefs != 0 || (!Di.defsVcc() && !Di.defsExec())) {
    return unsupported(Ctx, Di, "expected an implicit comparison destination");
  }
  std::optional<ICmpInst::Predicate> Predicate =
      getIntegerComparePredicate(Di.CanonOp);
  if (!Predicate) {
    return unsupported(Ctx, Di);
  }
  return raiseIntegerCompare32(Ctx, Di, Op, *Predicate, std::nullopt);
}

} // namespace COMGR::transpiler
