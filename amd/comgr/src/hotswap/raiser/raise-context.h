//===- raise-context.h - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_CONTEXT_H
#define HOTSWAP_TRANSPILER_RAISE_CONTEXT_H

#include "hotswap/common/kernel-meta.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/loader/code-object-utils.h"
#include "hotswap/raiser/register-state.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace COMGR::hotswap {

// Shared state threaded through every format handler.
class RaiseContext {
public:
  // Build the context for the source kernel described by Meta. B must be
  // positioned in the entry block: the register file and the cross-block
  // shadow storage are allocated there. Fails when the kernel descriptor and
  // the metadata disagree on the user-SGPR layout.
  static llvm::Expected<RaiseContext>
  create(llvm::IRBuilder<> &B, const WaveProjection &Projection,
         const MCState &MC, const KernelMeta &Meta,
         llvm::DenseMap<uint64_t, llvm::BasicBlock *> OffsetToBb,
         llvm::ArrayRef<uint8_t> SourceTextBytes,
         uint64_t SourceTextBaseAddress,
         llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
         uint64_t KernelStartOffset, uint64_t KernelEndOffset);

  // Builder every handler emits into. Its insertion point moves as raising
  // progresses.
  llvm::IRBuilder<> &B;
  // Translation between the source and target wave sizes.
  const WaveProjection &Projection;
  // MC layer for the source ISA, shared by every kernel in the code object.
  const MCState &MC;

  // Source architectural registers and the operand reads and writes that
  // resolve through them.
  RegisterState &registers() { return Registers; }

  // Source text section, and the address the source code object loads it at.
  // PC-relative literals are materialized by reading out of these.
  llvm::ArrayRef<uint8_t> sourceTextBytes() const { return SourceTextBytes; }
  uint64_t sourceTextBaseAddress() const { return SourceTextBaseAddress; }
  // Source code-object sections a proven PC-relative address can land in.
  llvm::ArrayRef<TextSection::ImageSection> sourceImageSections() const {
    return SourceImageSections;
  }

  // Offset of the source kernel's first byte within the source text section.
  uint64_t kernelStartOffset() const { return KernelStartOffset; }
  // Offset one past the source kernel's last byte, or 0 when the kernel runs
  // to the end of the source text section.
  uint64_t kernelEndOffset() const { return KernelEndOffset; }

  // Source scratch allocation, disjoint from target spills. Null until a
  // handler needs source scratch.
  llvm::AllocaInst *scratchPrivateSegmentAlloca() const {
    return ScratchPrivateSegmentAlloca;
  }
  void setScratchPrivateSegmentAlloca(llvm::AllocaInst *Alloca) {
    ScratchPrivateSegmentAlloca = Alloca;
  }

  // Return the block raised from the source instruction at Addr. A missing
  // block is a raiser bug and aborts.
  llvm::BasicBlock *lookupBB(uint64_t Addr);

  // Target-hardware lane id (i32), emitted once per kernel and reused.
  llvm::Value *emitLaneIdx();

  // Freeze per-lane addresses when widening wave32 to wave64. New target lanes
  // may hold poison from an earlier inactive definition, which would make even
  // an EXEC-predicated memory operation undefined. Other wave-size directions
  // return the address unchanged.
  llvm::Value *freezeMemAddr(llvm::Value *Addr);

private:
  RaiseContext(llvm::IRBuilder<> &B, const WaveProjection &Projection,
               const MCState &MC, RegisterState Registers,
               llvm::DenseMap<uint64_t, llvm::BasicBlock *> OffsetToBb,
               llvm::ArrayRef<uint8_t> SourceTextBytes,
               uint64_t SourceTextBaseAddress,
               llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
               uint64_t KernelStartOffset, uint64_t KernelEndOffset);

  // Source architectural registers, allocated in the entry block.
  RegisterState Registers;
  // Block raised from each source instruction offset that starts one.
  llvm::DenseMap<uint64_t, llvm::BasicBlock *> OffsetToBb;

  // Source code object, read to materialize proven PC-relative literals.
  llvm::ArrayRef<uint8_t> SourceTextBytes;
  uint64_t SourceTextBaseAddress = 0;
  llvm::ArrayRef<TextSection::ImageSection> SourceImageSections;

  // Extent of the source kernel within the source text section.
  uint64_t KernelStartOffset = 0;
  uint64_t KernelEndOffset = 0;

  // Allocation backing the source private segment, made on first use.
  llvm::AllocaInst *ScratchPrivateSegmentAlloca = nullptr;
};

// Reads a handler's source operands via the decoded srcMap, applying VOP3
// neg/abs modifiers on the float paths.
struct OpResolver {
  // Context the operands are read through.
  RaiseContext &Ctx;
  // Instruction whose operands are being read.
  const DecodedInst &Di;

  // MC operand index of the I-th source.
  unsigned srcIdx(unsigned I) const {
    assert(I < Di.SrcMap.size() && "source index out of range");
    return Di.SrcMap[I];
  }
  // Number of sources the instruction takes.
  unsigned nSrcs() const { return static_cast<unsigned>(Di.SrcMap.size()); }

  // Modifier bits attached to the I-th source, 0 when it carries none. Bit 0
  // negates and bit 1 takes the absolute value.
  unsigned srcMod(unsigned I) const;

  // Apply the I-th source's modifiers to V, which the caller has already read.
  // An integer-typed V round-trips through float, since the modifiers are
  // defined on the float interpretation of the bits.
  llvm::Value *applyMods(unsigned I, llvm::Value *V);

  // Read the I-th source as a 32-bit value.
  llvm::Expected<llvm::Value *> src(unsigned I) {
    return Ctx.registers().readOp32(Di, srcIdx(I));
  }
  // Read the I-th source as a 32-bit value with its modifiers applied.
  llvm::Expected<llvm::Value *> srcF(unsigned I);
  // Read the I-th source as a 64-bit value.
  llvm::Expected<llvm::Value *> src64(unsigned I) {
    return Ctx.registers().readOp64(Di, srcIdx(I));
  }
  // Read the I-th source as a wave mask at target EXEC width.
  llvm::Expected<llvm::Value *> srcExecWidth(unsigned I) {
    return Ctx.registers().readOpExecWidth(Di, srcIdx(I));
  }
  // Value of the I-th source, which must be an immediate.
  int64_t srcImm(unsigned I) {
    unsigned Index = srcIdx(I);
    assert(Di.isImm(Index) && "source operand must be an immediate");
    return Di.getImm(Index);
  }

  // Register the I-th destination names.
  llvm::Expected<ParsedReg> dst(unsigned I = 0) {
    assert(Di.isReg(I) && "destination operand must be a register");
    return Ctx.registers().parseReg(Di, I);
  }
  // Whether the I-th source is a register rather than an immediate.
  bool isSrcReg(unsigned I) { return Di.isReg(srcIdx(I)); }
  // Register the I-th source names, or no value when it is an immediate.
  llvm::Expected<std::optional<ParsedReg>> srcReg(unsigned I);
};

} // namespace COMGR::hotswap

#endif
