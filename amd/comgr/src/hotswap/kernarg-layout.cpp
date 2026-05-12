//===- kernarg-layout.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernarg-layout.h"

#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace COMGR::hotswap {

SourceHiddenArgByte classifySourceHiddenArgByte(ArrayRef<KernelArgMeta> Args,
                                                int ByteOffset) {
  for (const KernelArgMeta &Arg : Args) {
    if (ByteOffset < Arg.Offset || ByteOffset >= Arg.Offset + Arg.Size)
      continue;

    StringRef Kind(Arg.ValueKind);
    if (!Kind.starts_with("hidden_"))
      return {};

    SourceHiddenArgByte Result;
    Result.ValueKind = Kind;
    Result.ArgOffset = Arg.Offset;
    Result.ByteOffset = ByteOffset;
    if (Kind == "hidden_block_count_x")
      Result.Kind = SourceHiddenArgKind::HiddenBlockCountX;
    else if (Kind == "hidden_block_count_y")
      Result.Kind = SourceHiddenArgKind::HiddenBlockCountY;
    else if (Kind == "hidden_block_count_z")
      Result.Kind = SourceHiddenArgKind::HiddenBlockCountZ;
    else if (Kind == "hidden_group_size_x")
      Result.Kind = SourceHiddenArgKind::HiddenGroupSizeX;
    else if (Kind == "hidden_group_size_y")
      Result.Kind = SourceHiddenArgKind::HiddenGroupSizeY;
    else if (Kind == "hidden_group_size_z")
      Result.Kind = SourceHiddenArgKind::HiddenGroupSizeZ;
    else if (Kind == "hidden_remainder_x")
      Result.Kind = SourceHiddenArgKind::HiddenRemainderX;
    else if (Kind == "hidden_remainder_y")
      Result.Kind = SourceHiddenArgKind::HiddenRemainderY;
    else if (Kind == "hidden_remainder_z")
      Result.Kind = SourceHiddenArgKind::HiddenRemainderZ;
    else if (Kind == "hidden_grid_dims")
      Result.Kind = SourceHiddenArgKind::HiddenGridDims;
    else
      Result.Kind = SourceHiddenArgKind::UnsupportedHidden;
    return Result;
  }
  return {};
}

} // namespace COMGR::hotswap
