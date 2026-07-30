//===- kernarg-layout.h - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_KERNARG_LAYOUT_H
#define HOTSWAP_TRANSPILER_KERNARG_LAYOUT_H

#include "hotswap/common/kernel-meta.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>

namespace COMGR::hotswap {

// Source-kernel kernarg-segment metadata needed without reading the segment.
struct KernargLayout {
  // Source ABI byte offset where the implicit-arg block begins. Loads at or
  // above this offset are rebased through `amdgcn_implicitarg_ptr`.
  uint64_t ImplicitArgsBase = 0;
  // Source metadata argument layout, including hidden_* entries.
  llvm::ArrayRef<KernelArgMeta> Args;
  // Total kernarg segment size in bytes, copied from the kernel
  // descriptor's `.kernarg_segment_size`. Informational; the lifted
  // kernel's `Function` parameter list drives the backend's
  // `kernarg_segment_size` calculation in the output kernel descriptor.
  uint64_t KernargSegmentSize = 0;
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
  uint64_t ArgOffset = 0;
  uint64_t ByteOffset = 0;

  uint64_t byteIndexInArg() const { return ByteOffset - ArgOffset; }
};

// Resolve a byte offset in the source ABI's flat kernarg/hidden-arg metadata
// view. Returns nullopt when the offset does not land in a hidden_* argument;
// unsupported hidden args are reported with Kind == UnsupportedHidden.
std::optional<SourceHiddenArgByte>
classifySourceHiddenArgByte(llvm::ArrayRef<KernelArgMeta> Args,
                            uint64_t ByteOffset);

} // namespace COMGR::hotswap

#endif
