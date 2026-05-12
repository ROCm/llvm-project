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

namespace COMGR::hotswap {

// Source-kernel kernarg-segment metadata shared by the raiser and any
// helper that needs to reason about the segment without dereferencing
// it (e.g. the kernel-entry preloaded-SGPR seeding loop).
//
// The raiser itself no longer indexes the segment slot-by-slot:
// kernarg loads lift to GEP+load against `amdgcn_kernarg_segment_ptr`
// and the AMDGPU backend handles the ABI lowering. This struct keeps
// only the two pieces of metadata that survive that move.
struct KernargLayout {
  // Byte offset (within the source ABI's flat kernarg-segment view)
  // where the implicit-arg block begins. `handle_smem.cpp` consults
  // this to reroute SMEM loads at offsets >= implicitArgsBase through
  // `amdgcn_implicitarg_ptr` instead of the kernarg-segment pointer:
  // the source kernel's flat view is layout-correct for the source
  // ABI, but the lifted target kernel reaches implicit args via a
  // separate runtime pointer, so the offset must be rebased to
  // `byteOffset - implicitArgsBase`.
  int ImplicitArgsBase = 0;
  // Source metadata argument layout, including hidden_* entries. Used to
  // synthesize source-ABI hidden values without depending on target-runtime
  // implicit-arg layout.
  llvm::ArrayRef<KernelArgMeta> Args;
  // Total kernarg segment size in bytes, copied from the kernel
  // descriptor's `.kernarg_segment_size`. Informational; the lifted
  // kernel's `Function` parameter list drives the backend's
  // `kernarg_segment_size` calculation in the output KD.
  int KernargSegmentSize = 0;
};

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
  UnsupportedHidden,
};

struct SourceHiddenArgByte {
  SourceHiddenArgKind Kind = SourceHiddenArgKind::None;
  llvm::StringRef ValueKind;
  int ArgOffset = 0;
  int ByteOffset = 0;

  bool matched() const { return Kind != SourceHiddenArgKind::None; }
  unsigned byteIndexInArg() const {
    return static_cast<unsigned>(ByteOffset - ArgOffset);
  }
};

// Resolve a byte offset in the source ABI's flat kernarg/hidden-arg metadata
// view.  Known source hidden args are later synthesized from dispatch state;
// unsupported hidden args must refuse instead of falling back to target
// implicitarg layout.
SourceHiddenArgByte classifySourceHiddenArgByte(
    llvm::ArrayRef<KernelArgMeta> Args, int ByteOffset);

} // namespace COMGR::hotswap

#endif
