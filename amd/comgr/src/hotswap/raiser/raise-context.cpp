//===- raise-context.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise-context.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>

#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

Expected<RaiseContext>
RaiseContext::create(IRBuilder<> &B, const WaveProjection &Projection,
                     const MCState &MC, const KernelMeta &Meta,
                     ArrayRef<uint8_t> SourceTextBytes,
                     uint64_t SourceTextBaseAddress,
                     ArrayRef<TextSection::ImageSection> SourceImageSections,
                     uint64_t KernelStartOffset, uint64_t KernelEndOffset,
                     const std::set<uint64_t> &BlockStarts) {
  Expected<RegisterState> Registers =
      RegisterState::create(B, Projection, MC, Meta);
  if (!Registers)
    return Registers.takeError();
  return RaiseContext(B, Projection, MC, std::move(*Registers), SourceTextBytes,
                      SourceTextBaseAddress, SourceImageSections,
                      KernelStartOffset, KernelEndOffset, BlockStarts);
}

RaiseContext::RaiseContext(
    IRBuilder<> &B, const WaveProjection &Projection, const MCState &MC,
    RegisterState Registers, ArrayRef<uint8_t> SourceTextBytes,
    uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset,
    const std::set<uint64_t> &BlockStarts)
    : B(B), Projection(Projection), MC(MC), Registers(std::move(Registers)),
      SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset) {
  assert(BlockStarts.count(KernelStartOffset) &&
         "the source kernel's first instruction starts a block");
  // The builder is positioned in the entry block, which holds the register
  // file and stays out of the block map: the source kernel may branch back to
  // its own first instruction, and an LLVM entry block may have no
  // predecessors. Ascending offsets give the blocks deterministic names and
  // lay them out in source order.
  Function *F = B.GetInsertBlock()->getParent();
  Blocks.reserve(BlockStarts.size());
  for (uint64_t Addr : BlockStarts) {
    BasicBlock *Bb = BasicBlock::Create(
        F->getContext(), "bb_0x" + utohexstr(Addr - KernelStartOffset), F);
    Blocks.push_back({Addr, Bb});
  }
}

BasicBlock *RaiseContext::findBB(uint64_t Addr) const {
  SmallVectorImpl<SourceBlock>::const_iterator It =
      llvm::lower_bound(Blocks, Addr, [](const SourceBlock &B, uint64_t Addr) {
        return B.Offset < Addr;
      });
  return It != Blocks.end() && It->Offset == Addr ? It->Bb : nullptr;
}

BasicBlock *RaiseContext::lookupBB(uint64_t Addr) {
  if (BasicBlock *Bb = findBB(Addr))
    return Bb;
  // Every in-extent branch target is a block leader recorded during CFG
  // recovery, so a miss is a raiser bug, not a recoverable case.
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
