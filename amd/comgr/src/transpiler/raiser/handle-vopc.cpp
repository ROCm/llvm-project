//===- handle-vopc.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/register-state.h"

#include "MCTargetDesc/AMDGPUMCExpr.h"
#include "SIDefines.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <optional>

using namespace llvm;

namespace COMGR::transpiler {

std::optional<VectorCompareInfo> getVectorCompareInfo(CanonicalOp Opcode) {
  switch (Opcode) {
#define VECTOR_COMPARE(Name, Predicate, BitWidth, SignExtendLiteral)           \
  case CanonicalOp::V_CMP_##Name:                                              \
    return VectorCompareInfo{CmpInst::Predicate, BitWidth, SignExtendLiteral};
#define VECTOR_CLASS(Name, BitWidth)                                           \
  case CanonicalOp::V_CMP_CLASS_##Name:                                        \
    return VectorCompareInfo{std::nullopt, BitWidth};
#include "transpiler/decoder/vector-compare.def"
  default:
    return std::nullopt;
  }
}

static Expected<Value *> readCompareSource(OperandResolver &Op, unsigned Index,
                                           VectorCompareInfo Info) {
  if (!Info.Predicate && Index == 1) {
    assert(!Op.srcMod(Index) && "class mask cannot have source modifiers");
    return Op.src(Index);
  }
  if (Info.isFloat())
    return Info.BitWidth == 64 ? Op.srcF64(Index) : Op.srcF32(Index);

  unsigned AllowedModifiers = Info.BitWidth == 16 ? SISrcMods::OP_SEL_0 : 0;
  if (Op.srcMod(Index) & ~AllowedModifiers)
    return unsupported(Op.Ctx, Op.Di,
                       "integer source modifiers are not supported");
  if (Info.BitWidth == 16)
    return Op.src16(Index);
  Expected<Value *> Source = Op.src(Index, Info.BitWidth == 64);
  if (!Source)
    return Source.takeError();
  if (!Info.SignExtendLiteral || Op.isSrcReg(Index))
    return *Source;

  const MCOperand &Operand = Op.Di.Inst.getOperand(Op.srcIdx(Index));
  if (Operand.isExpr()) {
    const auto *Literal = dyn_cast<AMDGPUMCExpr>(Operand.getExpr());
    if (Literal && Literal->getKind() == AMDGPUMCExpr::AGVK_Lit64)
      return *Source;
  }
  const APInt &Bits = cast<ConstantInt>(*Source)->getValue();
  if (Bits.getActiveBits() <= 32)
    return Op.Ctx.B.getInt64(Bits.trunc(32).getSExtValue());
  return *Source;
}

Error raiseVectorCompare(RaiseContext &Ctx, const DecodedInst &Di,
                         OperandResolver &Op, VectorCompareInfo Info) {
  assert(Op.nSrcs() == 2 && "comparison must have two sources");
  assert(COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
                                               AMDGPU::OpName::omod) < 0 &&
         "comparison cannot have an output multiplier");
  int ClampIndex = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                         AMDGPU::OpName::clamp);
  if (ClampIndex >= 0) {
    assert(Di.isImm(ClampIndex) && "comparison clamp must be immediate");
    if (Di.getImm(ClampIndex))
      return unsupported(Ctx, Di, "comparison clamp is not supported");
  }

  std::optional<ParsedReg> Destination;
  if (Di.NumDefs) {
    assert(Di.NumDefs == 1 && "comparison must have one explicit destination");
    if (!Di.isReg(0))
      return unsupported(Ctx, Di, "expected a comparison mask destination");
    Expected<ParsedReg> Register = Op.dst();
    if (!Register)
      return Register.takeError();
    if (Register->RegKind != ParsedReg::SGPR &&
        Register->RegKind != ParsedReg::VCC &&
        Register->RegKind != ParsedReg::NOREG)
      return unsupported(Ctx, Di, "unsupported comparison mask destination");
    Destination = *Register;
  } else {
    assert((Di.defsVcc() || Di.defsExec()) &&
           "comparison must have an implicit destination");
  }

  Expected<Value *> Src0 = readCompareSource(Op, 0, Info);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = readCompareSource(Op, 1, Info);
  if (!Src1)
    return Src1.takeError();

  Value *Cmp;
  if (!Info.Predicate)
    Cmp = Ctx.B.CreateIntrinsic(Intrinsic::amdgcn_class, {(*Src0)->getType()},
                                {*Src0, *Src1});
  else if (Info.isFloat())
    Cmp = Ctx.B.CreateFCmp(*Info.Predicate, *Src0, *Src1);
  else
    Cmp = Ctx.B.CreateICmp(*Info.Predicate, *Src0, *Src1);
  RegisterState &Regs = Ctx.registers();

  // Inactive lanes contribute zero even when their source values are poison.
  Value *Bit = Ctx.B.CreateSelect(Regs.emitLaneActiveBit(), Cmp,
                                  Ctx.B.getFalse(), "cmp_active");
  if (Destination && Destination->RegKind == ParsedReg::SGPR) {
    Value *Mask = Ctx.Projection.ballotI1ToWidth(
        Ctx.B, Bit, Ctx.Projection.sourceWaveMaskTy(), "cmp_mask");
    Regs.writeRegExecWidth(*Destination, Mask);
  }
  if (Destination)
    Regs.recordWaveMaskI1(*Destination, Bit);
  if (Di.defsVcc())
    Regs.recordWaveMaskI1(ParsedReg{ParsedReg::VCC}, Bit);
  if (Di.defsExec()) {
    Value *Mask = Ctx.Projection.ballotI1ToWidth(
        Ctx.B, Bit, Ctx.Projection.execStorageTy(), "cmpx_ballot");
    Value *CurrentExec = Regs.readExec();
    Value *NarrowedExec = Ctx.B.CreateAnd(CurrentExec, Mask, "cmpx_exec");
    Regs.storeExec(NarrowedExec);
  }
  return Error::success();
}

Error handleVOPC(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  assert(Di.NumDefs == 0 && (Di.defsVcc() || Di.defsExec()) &&
         "VOPC comparison must have an implicit destination");
  std::optional<VectorCompareInfo> Info = getVectorCompareInfo(Di.CanonOp);
  if (!Info)
    return unsupported(Ctx, Di);
  return raiseVectorCompare(Ctx, Di, Op, *Info);
}

} // namespace COMGR::transpiler
