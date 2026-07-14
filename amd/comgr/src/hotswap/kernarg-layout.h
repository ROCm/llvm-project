//===- kernarg-layout.h - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_KERNARG_LAYOUT_H
#define HOTSWAP_TRANSPILER_KERNARG_LAYOUT_H

#include "code-object-utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace COMGR::hotswap {

// Source-kernel kernarg-segment metadata needed without reading the segment.
struct KernargLayout {
  // Source ABI byte offset where the implicit-arg block begins. Loads at or
  // above this offset are rebased through `amdgcn_implicitarg_ptr`.
  int ImplicitArgsBase = 0;
  // Source metadata argument layout, including hidden_* entries.
  llvm::ArrayRef<KernelArgMeta> Args;
  // Total kernarg segment size in bytes, copied from the kernel
  // descriptor's `.kernarg_segment_size`. Informational; the lifted
  // kernel's `Function` parameter list drives the backend's
  // `kernarg_segment_size` calculation in the output KD.
  int KernargSegmentSize = 0;
};

// Source metadata hidden_* argument kinds with source-ABI synthesis support.
enum class SourceHiddenArgKind {
  None,
  HiddenBlockCountX,
  HiddenBlockCountY,
  HiddenBlockCountZ,
  HiddenGroupSizeX,
  HiddenGroupSizeY,
  HiddenGroupSizeZ,
  HiddenRemainderX,
  HiddenRemainderY,
  HiddenRemainderZ,
  HiddenGridDims,
  HiddenGlobalOffsetX,
  HiddenGlobalOffsetY,
  HiddenGlobalOffsetZ,
  HiddenPrivateBase,
  HiddenSharedBase,
  HiddenDefaultQueue,
  HiddenCompletionAction,
  HiddenMultigridSyncArg,
  HiddenHostcallBuffer,
  HiddenHeapV1,
  UnsupportedHidden,
};

// Metadata match for one byte in a source hidden_* argument.
struct SourceHiddenArgByte {
  SourceHiddenArgKind Kind = SourceHiddenArgKind::None;
  llvm::StringRef ValueKind;
  int64_t ArgOffset = 0;
  int64_t ByteOffset = 0;

  unsigned byteIndexInArg() const {
    return static_cast<unsigned>(ByteOffset - ArgOffset);
  }
};

// Resolve a byte offset in the source ABI's flat kernarg/hidden-arg metadata
// view. Unsupported hidden args are reported as matched-but-unsupported.
std::optional<SourceHiddenArgByte>
classifySourceHiddenArgByte(llvm::ArrayRef<KernelArgMeta> Args,
                            int64_t ByteOffset);

} // namespace COMGR::hotswap

#endif
