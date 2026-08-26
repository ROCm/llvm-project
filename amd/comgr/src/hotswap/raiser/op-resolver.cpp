//===- op-resolver.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/op-resolver.h"

#include "llvm/IR/Intrinsics.h"

#include <climits>

using namespace llvm;

namespace COMGR::hotswap {

unsigned OpResolver::srcMod(unsigned I) const {
  assert(I < Di.ModMap.size() && "source modifier index out of range");
  unsigned ModIdx = Di.ModMap[I];
  if (ModIdx == UINT_MAX)
    return 0;
  assert(Di.isImm(ModIdx) && "source modifier must be an immediate");
  return static_cast<unsigned>(Di.getImm(ModIdx) & 0xF);
}

Value *OpResolver::applyMods(unsigned I, Value *V) {
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

Expected<Value *> OpResolver::srcF(unsigned I) {
  Expected<Value *> V = Ctx.registers().readOp32(Di, srcIdx(I));
  if (!V)
    return V.takeError();
  return applyMods(I, *V);
}

Expected<std::optional<ParsedReg>> OpResolver::srcReg(unsigned I) {
  unsigned Index = srcIdx(I);
  if (!Di.isReg(Index))
    return std::optional<ParsedReg>();
  Expected<ParsedReg> Reg = Ctx.registers().parseReg(Di, Index);
  if (!Reg)
    return Reg.takeError();
  return std::optional<ParsedReg>(*Reg);
}

} // namespace COMGR::hotswap
