//===- operand-resolver.cpp - Transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/operand-resolver.h"

#include "transpiler/raiser/raise_failure.h"

#include "SIDefines.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/MC/MCRegisterInfo.h"

#include <climits>

using namespace llvm;

namespace COMGR::transpiler {

unsigned OperandResolver::srcMod(unsigned I) const {
  assert(I < Di.ModMap.size() && "source modifier index out of range");
  unsigned ModIdx = Di.ModMap[I];
  if (ModIdx == UINT_MAX)
    return 0;
  assert(Di.isImm(ModIdx) && "source modifier must be an immediate");
  return static_cast<unsigned>(Di.getImm(ModIdx) & 0xF);
}

Value *OperandResolver::applyMods(unsigned I, Value *V) {
  unsigned Mods = srcMod(I);
  if (Mods == 0)
    return V;
  bool IsI32 = (V->getType() == Ctx.B.getInt32Ty());
  if (IsI32)
    V = Ctx.B.CreateBitCast(V, Ctx.B.getFloatTy());
  if (Mods & 2)
    V = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, V, nullptr, "abs");
  if (Mods & 1)
    V = Ctx.B.CreateFNeg(V, "neg");
  if (IsI32)
    V = Ctx.B.CreateBitCast(V, Ctx.B.getInt32Ty());
  return V;
}

Expected<Value *> OperandResolver::srcF(unsigned I) {
  Expected<Value *> V = Ctx.registers().readOp32(Di, srcIdx(I));
  if (!V)
    return V.takeError();
  if (srcMod(I) & ~(SISrcMods::NEG | SISrcMods::ABS))
    return unsupportedInstruction(Ctx, Di, "unsupported f32 source modifier");
  Value *Float = Ctx.B.CreateBitCast(*V, Ctx.B.getFloatTy());
  return applyMods(I, Float);
}

Expected<Value *> OperandResolver::srcF16(unsigned I) {
  Expected<Value *> V = Ctx.registers().readOp32(Di, srcIdx(I));
  if (!V)
    return V.takeError();

  unsigned Modifiers = srcMod(I);
  constexpr unsigned AllowedModifiers =
      SISrcMods::NEG | SISrcMods::ABS | SISrcMods::OP_SEL_0;
  if (Modifiers & ~AllowedModifiers)
    return unsupportedInstruction(Ctx, Di, "unsupported f16 source modifier");

  unsigned SourceIndex = srcIdx(I);
  bool SourceIsHigh =
      (Modifiers & SISrcMods::OP_SEL_0) != 0 ||
      (Di.isReg(SourceIndex) &&
       (Ctx.MC.RegInfo->getEncodingValue(Di.getReg(SourceIndex)) &
        AMDGPU::HWEncoding::IS_HI16));
  Value *SelectedBits = *V;
  if (SourceIsHigh)
    SelectedBits = Ctx.B.CreateLShr(SelectedBits, 16, "src.hi");
  Value *LowBits = Ctx.B.CreateTrunc(SelectedBits, Ctx.B.getInt16Ty());
  Value *Half = Ctx.B.CreateBitCast(LowBits, Ctx.B.getHalfTy());
  return applyMods(I, Half);
}

Expected<std::optional<ParsedReg>> OperandResolver::srcReg(unsigned I) {
  unsigned Index = srcIdx(I);
  if (!Di.isReg(Index))
    return std::optional<ParsedReg>();
  Expected<ParsedReg> Reg = Ctx.registers().parseReg(Di, Index);
  if (!Reg)
    return Reg.takeError();
  return std::optional<ParsedReg>(*Reg);
}

Expected<BinaryOperands> OperandResolver::readBinary32() {
  Expected<ParsedReg> Dst = dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = src(1);
  if (!Src1)
    return Src1.takeError();
  return BinaryOperands{*Dst, *Src0, *Src1};
}

Expected<BinaryOperands> OperandResolver::readBinary64() {
  Expected<ParsedReg> Dst = dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = src64(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = src64(1);
  if (!Src1)
    return Src1.takeError();
  return BinaryOperands{*Dst, *Src0, *Src1};
}

Expected<TernaryOperands> OperandResolver::readTernary32() {
  assert(nSrcs() >= 3 && "ternary instruction must have three sources");
  Expected<ParsedReg> Dst = dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = src(1);
  if (!Src1)
    return Src1.takeError();
  Expected<Value *> Src2 = src(2);
  if (!Src2)
    return Src2.takeError();
  return TernaryOperands{*Dst, *Src0, *Src1, *Src2};
}

} // namespace COMGR::transpiler
