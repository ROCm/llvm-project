//===- raise-context.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise-context.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/ErrorHandling.h"

#include <climits>
#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

Expected<RaiseContext> RaiseContext::create(
    IRBuilder<> &B, const WaveProjection &Projection, const MCState &MC,
    const KernelMeta &Meta, DenseMap<uint64_t, BasicBlock *> OffsetToBb,
    ArrayRef<uint8_t> SourceTextBytes, uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset) {
  Expected<RegisterState> Registers =
      RegisterState::create(B, Projection, MC, Meta);
  if (!Registers)
    return Registers.takeError();
  return RaiseContext(B, Projection, MC, std::move(*Registers),
                      std::move(OffsetToBb), SourceTextBytes,
                      SourceTextBaseAddress, SourceImageSections,
                      KernelStartOffset, KernelEndOffset);
}

RaiseContext::RaiseContext(
    IRBuilder<> &B, const WaveProjection &Projection, const MCState &MC,
    RegisterState Registers, DenseMap<uint64_t, BasicBlock *> OffsetToBb,
    ArrayRef<uint8_t> SourceTextBytes, uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset)
    : B(B), Projection(Projection), MC(MC), Registers(std::move(Registers)),
      OffsetToBb(std::move(OffsetToBb)), SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset) {}

BasicBlock *RaiseContext::lookupBB(uint64_t Addr) {
  DenseMap<uint64_t, BasicBlock *>::iterator It = OffsetToBb.find(Addr);
  if (It != OffsetToBb.end())
    return It->second;
  // Every branch target is a block leader recorded during CFG layout, so a
  // miss is a raiser bug, not a recoverable case.
  report_fatal_error(Twine("transpiler: missing basic block for offset 0x") +
                     utohexstr(Addr));
}

Value *RaiseContext::emitLaneIdx() { return Projection.emitLaneIdx(B); }

Value *RaiseContext::freezeMemAddr(Value *Addr) {
  if (!Projection.sourceIsa().isWave32() || Projection.targetIsa().isWave32())
    return Addr;
  return B.CreateFreeze(Addr, "mem_addr_frozen");
}

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
