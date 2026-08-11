//===- kernarg-layout.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernarg-layout.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>

using namespace llvm;

namespace COMGR::hotswap {

SourceHiddenArgKind classifySourceHiddenArgKind(StringRef ValueKind) {
  return StringSwitch<SourceHiddenArgKind>(ValueKind)
      .Case("hidden_block_count_x", SourceHiddenArgKind::HiddenBlockCountX)
      .Case("hidden_block_count_y", SourceHiddenArgKind::HiddenBlockCountY)
      .Case("hidden_block_count_z", SourceHiddenArgKind::HiddenBlockCountZ)
      .Case("hidden_group_size_x", SourceHiddenArgKind::HiddenGroupSizeX)
      .Case("hidden_group_size_y", SourceHiddenArgKind::HiddenGroupSizeY)
      .Case("hidden_group_size_z", SourceHiddenArgKind::HiddenGroupSizeZ)
      .Case("hidden_remainder_x", SourceHiddenArgKind::HiddenRemainderX)
      .Case("hidden_remainder_y", SourceHiddenArgKind::HiddenRemainderY)
      .Case("hidden_remainder_z", SourceHiddenArgKind::HiddenRemainderZ)
      .Case("hidden_grid_dims", SourceHiddenArgKind::HiddenGridDims)
      .Case("hidden_global_offset_x", SourceHiddenArgKind::HiddenGlobalOffsetX)
      .Case("hidden_global_offset_y", SourceHiddenArgKind::HiddenGlobalOffsetY)
      .Case("hidden_global_offset_z", SourceHiddenArgKind::HiddenGlobalOffsetZ)
      .Case("hidden_private_base", SourceHiddenArgKind::HiddenPrivateBase)
      .Case("hidden_shared_base", SourceHiddenArgKind::HiddenSharedBase)
      .Case("hidden_default_queue", SourceHiddenArgKind::HiddenDefaultQueue)
      .Case("hidden_completion_action",
            SourceHiddenArgKind::HiddenCompletionAction)
      .Case("hidden_multigrid_sync_arg",
            SourceHiddenArgKind::HiddenMultigridSyncArg)
      .Case("hidden_hostcall_buffer", SourceHiddenArgKind::HiddenHostcallBuffer)
      .Case("hidden_heap_v1", SourceHiddenArgKind::HiddenHeapV1)
      .Default(ValueKind.starts_with("hidden_")
                   ? SourceHiddenArgKind::UnsupportedHidden
                   : SourceHiddenArgKind::None);
}

std::optional<SourceHiddenArgByte>
classifySourceHiddenArgByte(ArrayRef<KernelArgMeta> Args, uint64_t ByteOffset) {
  for (const KernelArgMeta &Arg : Args) {
    uint64_t ArgEnd = static_cast<uint64_t>(Arg.Offset) + Arg.Size;
    if (ByteOffset < Arg.Offset || ByteOffset >= ArgEnd)
      continue;

    SourceHiddenArgByte Result;
    Result.ArgOffset = Arg.Offset;
    Result.ByteOffset = ByteOffset;
    Result.Kind = classifySourceHiddenArgKind(Arg.ValueKind);
    // A non-hidden kernarg occupying this byte is not a hidden-arg match.
    if (Result.Kind == SourceHiddenArgKind::None)
      return std::nullopt;
    return Result;
  }
  return std::nullopt;
}

} // namespace COMGR::hotswap
