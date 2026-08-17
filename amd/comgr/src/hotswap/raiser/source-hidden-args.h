//===- source-hidden-args.h - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_H
#define HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_H

#include "hotswap/common/kernel-meta.h"
#include "kernarg-layout.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace llvm {
class Function;
class IRBuilderBase;
class LLVMContext;
class Module;
class Type;
class Value;
} // namespace llvm

namespace COMGR::hotswap {

// Inputs needed to synthesize source-ABI hidden argument values in IR.
struct SourceHiddenArgContext {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::IRBuilderBase &B;
  llvm::Function &Fn;
  // The i8/i32/i64 types are cached here rather than re-fetched from the
  // builder at every use: the emit helpers below reference them densely and
  // the cached handles keep those expressions terse.
  llvm::Type *I8Ty;
  llvm::Type *I32Ty;
  llvm::Type *I64Ty;
  llvm::ArrayRef<KernelArgMeta> Args;
  bool AssumeHipGlobalOffsetZero = false;
  unsigned TargetCodeObjectVersion = 6;
  // Scaled replication factor along the x dimension. The runtime
  // launches such a block with an x extent scaled up by this factor, so the
  // source kernel's loops and reduction bounds must still observe the
  // un-scaled block size: the synthesized `hidden_group_size_x` and x grid-size
  // reads are divided by the factor. Derived hidden args (block_count =
  // grid/group, remainder = grid%group) stay correct because the factor
  // cancels in the ratio. A factor of 1 disables the adjustment.
  unsigned ScaledReplicationFactor = 1;
};

// Synthesize a 32-bit source hidden argument value at ByteOffset. Returns a
// null Value when the offset is not a source hidden argument (the caller then
// falls back to a plain kernarg load); returns an Error when the offset is a
// hidden argument with no supported source-side synthesis.
llvm::Expected<llvm::Value *> emitSourceHiddenDword(SourceHiddenArgContext &Ctx,
                                                    uint64_t ByteOffset);
// Synthesize a 1-, 2-, or 4-byte source hidden integer at ByteOffset, with the
// same null-Value / Error contract as emitSourceHiddenDword.
llvm::Expected<llvm::Value *>
emitSourceHiddenInteger(SourceHiddenArgContext &Ctx, uint64_t ByteOffset,
                        unsigned ByteWidth, bool IsSigned);

} // namespace COMGR::hotswap

#endif
