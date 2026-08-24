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
#include "hotswap/decoder/mc-state.h"
#include "hotswap/loader/code-object-utils.h"
#include "hotswap/raiser/register-state.h"
#include "hotswap/raiser/wave-projection.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <set>

namespace COMGR::hotswap {

// Shared state threaded through every format handler.
class RaiseContext {
public:
  // Build the context for the source kernel described by Meta. B must be
  // positioned in the entry block: the register file and the cross-block
  // shadow storage are allocated there. One block is created per offset in
  // BlockStarts, which must hold every leader the decode recovered; the entry
  // block is not one of them, so it keeps no predecessors and stays a legal
  // home for the allocas. Fails when the kernel descriptor and the metadata
  // disagree on the user-SGPR layout.
  static llvm::Expected<RaiseContext>
  create(llvm::IRBuilder<> &B, const WaveProjection &Projection,
         const MCState &MC, const KernelMeta &Meta,
         llvm::ArrayRef<uint8_t> SourceTextBytes,
         uint64_t SourceTextBaseAddress,
         llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
         uint64_t KernelStartOffset, uint64_t KernelEndOffset,
         const std::set<uint64_t> &BlockStarts);

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

  // A source block leader and the LLVM block raised from it.
  struct SourceBlock {
    uint64_t Offset;
    llvm::BasicBlock *Bb;
  };

  // Return the block raised from the source instruction at Addr. Every leader
  // the decode recovered has one, so a missing block is a raiser bug and
  // aborts. Callers resolving a branch must reject an out-of-extent target
  // before asking.
  llvm::BasicBlock *lookupBB(uint64_t Addr);

  // Return the block starting at Addr, or null when Addr does not start one.
  llvm::BasicBlock *findBB(uint64_t Addr) const;

  // Whether Addr is inside the source kernel's extent. A branch leaving it
  // would raise instructions belonging to a neighboring symbol. A kernel that
  // runs to the end of the text section is still bounded by that section, so a
  // branch displacement large enough to address outside it is rejected here
  // rather than reaching a block lookup that has nothing to find.
  bool isInKernelExtent(uint64_t Addr) const {
    uint64_t End =
        KernelEndOffset == 0 ? SourceTextBytes.size() : KernelEndOffset;
    return Addr >= KernelStartOffset && Addr < End;
  }

  // Every source block, in ascending source-offset order.
  llvm::ArrayRef<SourceBlock> blocks() const { return Blocks; }

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
               llvm::ArrayRef<uint8_t> SourceTextBytes,
               uint64_t SourceTextBaseAddress,
               llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
               uint64_t KernelStartOffset, uint64_t KernelEndOffset,
               const std::set<uint64_t> &BlockStarts);

  // Source architectural registers, allocated in the entry block.
  RegisterState Registers;
  // Block raised from each source offset that starts one, ascending by offset.
  llvm::SmallVector<SourceBlock> Blocks;

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

} // namespace COMGR::hotswap

#endif
