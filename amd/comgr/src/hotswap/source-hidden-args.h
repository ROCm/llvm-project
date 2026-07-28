//===- source-hidden-args.h - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_H
#define HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_H

#include "code-object-utils.h"
#include "kernarg-layout.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace llvm {
class Function;
class IRBuilderBase;
class LLVMContext;
class Module;
class Type;
class Value;
} // namespace llvm

namespace COMGR::hotswap {

class WaveProjection;

// Inputs needed to synthesize source-ABI hidden argument values in IR.
struct SourceHiddenArgContext {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::IRBuilderBase &B;
  llvm::Type *I8Ty;
  llvm::Type *I32Ty;
  llvm::Type *I64Ty;
  llvm::ArrayRef<KernelArgMeta> Args;
  bool AssumeHipGlobalOffsetZero = false;
  unsigned TargetCodeObjectVersion = 6;
  // Scaled-dispatch virtualization (ScaledModuloReplicationProjection). When
  // > 1, the runtime launches this block with a `ScaledDispatchFactor`-scaled x
  // extent (x is always the scaled dimension). The source kernel's loops and
  // reduction bounds must still observe the un-scaled block size, so the
  // synthesized `hidden_group_size_x` and grid-size-x reads are divided by the
  // factor. All derived hidden args (block_count = grid/group,
  // remainder = grid%group) stay correct because each halved read reproduces
  // the exact source size. 1 disables the adjustment.
  unsigned ScaledDispatchFactor = 1;
};

// Result of attempting to synthesize a source hidden argument.
struct SourceHiddenArgValue {
  // True when ByteOffset maps to a source metadata hidden_* field.
  bool Matched = false;
  // Non-null when a matched hidden field was lowered successfully.
  llvm::Value *Value = nullptr;
  // Non-empty when Matched is true and Value is null.
  std::string FailureDetail;
};

// Copy `Projection`'s scaled-dispatch factor onto `Ctx` when the projection
// uses a scaled dispatch, so the x-dimension size reads are virtualized back to
// the source block size. No-op otherwise.
void populateScaledDispatch(SourceHiddenArgContext &Ctx,
                            const WaveProjection &Projection);

// Synthesize a 32-bit source hidden argument value at ByteOffset.
SourceHiddenArgValue emitSourceHiddenDword(SourceHiddenArgContext &Ctx,
                                           int64_t ByteOffset);
// Synthesize a 1-, 2-, or 4-byte source hidden integer at ByteOffset.
SourceHiddenArgValue emitSourceHiddenInteger(SourceHiddenArgContext &Ctx,
                                             int64_t ByteOffset,
                                             unsigned ByteWidth, bool IsSigned);

} // namespace COMGR::hotswap

#endif
