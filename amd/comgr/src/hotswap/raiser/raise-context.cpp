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
#include "llvm/Support/ErrorHandling.h"

#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

Expected<RaiseContext>
RaiseContext::create(IRBuilder<> &B, const WaveProjection &Projection,
                     const MCState &MC, const KernelMeta &Meta,
                     ArrayRef<uint8_t> SourceTextBytes,
                     uint64_t SourceTextBaseAddress,
                     ArrayRef<TextSection::ImageSection> SourceImageSections,
                     uint64_t KernelStartOffset, uint64_t KernelEndOffset) {
  Expected<RegisterState> Registers =
      RegisterState::create(B, Projection, MC, Meta);
  if (!Registers)
    return Registers.takeError();
  return RaiseContext(B, Projection, MC, std::move(*Registers), SourceTextBytes,
                      SourceTextBaseAddress, SourceImageSections,
                      KernelStartOffset, KernelEndOffset);
}

RaiseContext::RaiseContext(
    IRBuilder<> &B, const WaveProjection &Projection, const MCState &MC,
    RegisterState Registers, ArrayRef<uint8_t> SourceTextBytes,
    uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset)
    : B(B), Projection(Projection), MC(MC), Registers(std::move(Registers)),
      SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset) {
  // The builder is positioned in the entry block, which is what the source
  // kernel's first instruction raised into.
  OffsetToBb[KernelStartOffset] = B.GetInsertBlock();
}

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

} // namespace COMGR::hotswap
