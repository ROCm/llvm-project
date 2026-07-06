//===- kernarg_layout_test.cpp - kernarg_layout unit tests ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/kernarg-layout.h"

#include "gtest/gtest.h"

#include <vector>

using COMGR::hotswap::classifySourceHiddenArgByte;
using COMGR::hotswap::KernelArgMeta;
using COMGR::hotswap::SourceHiddenArgByte;
using COMGR::hotswap::SourceHiddenArgKind;

namespace {
KernelArgMeta makeArg(const char *Name, int Offset, int Size,
                      const char *ValueKind) {
  KernelArgMeta Arg;
  Arg.Name = Name;
  Arg.Offset = Offset;
  Arg.Size = Size;
  Arg.ValueKind = ValueKind;
  return Arg;
}
} // namespace

TEST(KernargLayout, ClassifiesHiddenBlockCountsByByteContainment) {
  std::vector<KernelArgMeta> Args = {
      makeArg("out", 0, 8, "global_buffer"),
      makeArg("grid_x", 48, 4, "hidden_block_count_x"),
      makeArg("grid_y", 52, 4, "hidden_block_count_y"),
      makeArg("grid_z", 56, 4, "hidden_block_count_z"),
  };

  std::optional<SourceHiddenArgByte> X0 = classifySourceHiddenArgByte(Args, 48);
  std::optional<SourceHiddenArgByte> X3 = classifySourceHiddenArgByte(Args, 51);
  std::optional<SourceHiddenArgByte> Y0 = classifySourceHiddenArgByte(Args, 52);
  std::optional<SourceHiddenArgByte> Z0 = classifySourceHiddenArgByte(Args, 56);

  ASSERT_TRUE(X0.has_value());
  ASSERT_TRUE(X3.has_value());
  ASSERT_TRUE(Y0.has_value());
  ASSERT_TRUE(Z0.has_value());

  EXPECT_EQ(X0->Kind, SourceHiddenArgKind::HiddenBlockCountX);
  EXPECT_EQ(X0->byteIndexInArg(), 0u);
  EXPECT_EQ(X3->Kind, SourceHiddenArgKind::HiddenBlockCountX);
  EXPECT_EQ(X3->byteIndexInArg(), 3u);
  EXPECT_EQ(Y0->Kind, SourceHiddenArgKind::HiddenBlockCountY);
  EXPECT_EQ(Z0->Kind, SourceHiddenArgKind::HiddenBlockCountZ);
}

TEST(KernargLayout, ClassifiesGroupSizeRemainderAndGridDims) {
  std::vector<KernelArgMeta> Args = {
      makeArg("group_x", 44, 2, "hidden_group_size_x"),
      makeArg("rem_x", 50, 2, "hidden_remainder_x"),
      makeArg("grid_dims", 96, 2, "hidden_grid_dims"),
  };

  std::optional<SourceHiddenArgByte> GSX =
      classifySourceHiddenArgByte(Args, 44);
  std::optional<SourceHiddenArgByte> RemX =
      classifySourceHiddenArgByte(Args, 50);
  std::optional<SourceHiddenArgByte> GD = classifySourceHiddenArgByte(Args, 96);
  ASSERT_TRUE(GSX.has_value());
  ASSERT_TRUE(RemX.has_value());
  ASSERT_TRUE(GD.has_value());

  EXPECT_EQ(GSX->Kind, SourceHiddenArgKind::HiddenGroupSizeX);
  EXPECT_EQ(RemX->Kind, SourceHiddenArgKind::HiddenRemainderX);
  EXPECT_EQ(GD->Kind, SourceHiddenArgKind::HiddenGridDims);
}

TEST(KernargLayout, ClassifiesUnsupportedHiddenKinds) {
  std::vector<KernelArgMeta> Args = {
      makeArg("printf", 64, 8, "hidden_printf_buffer"),
  };

  std::optional<SourceHiddenArgByte> Unsupported =
      classifySourceHiddenArgByte(Args, 64);
  ASSERT_TRUE(Unsupported.has_value());

  EXPECT_EQ(Unsupported->Kind, SourceHiddenArgKind::UnsupportedHidden);
}

TEST(KernargLayout, ClassifiesIdentityAndDerivedHiddenKinds) {
  std::vector<KernelArgMeta> Args = {
      makeArg("private_base", 64, 4, "hidden_private_base"),
      makeArg("shared_base", 68, 4, "hidden_shared_base"),
      makeArg("default_queue", 72, 8, "hidden_default_queue"),
      makeArg("completion_action", 80, 8, "hidden_completion_action"),
      makeArg("multigrid", 88, 8, "hidden_multigrid_sync_arg"),
      makeArg("hostcall", 96, 8, "hidden_hostcall_buffer"),
      makeArg("heap", 104, 8, "hidden_heap_v1"),
  };

  EXPECT_EQ(classifySourceHiddenArgByte(Args, 64)->Kind,
            SourceHiddenArgKind::HiddenPrivateBase);
  EXPECT_EQ(classifySourceHiddenArgByte(Args, 68)->Kind,
            SourceHiddenArgKind::HiddenSharedBase);
  EXPECT_EQ(classifySourceHiddenArgByte(Args, 72)->Kind,
            SourceHiddenArgKind::HiddenDefaultQueue);
  EXPECT_EQ(classifySourceHiddenArgByte(Args, 80)->Kind,
            SourceHiddenArgKind::HiddenCompletionAction);
  EXPECT_EQ(classifySourceHiddenArgByte(Args, 88)->Kind,
            SourceHiddenArgKind::HiddenMultigridSyncArg);
  EXPECT_EQ(classifySourceHiddenArgByte(Args, 96)->Kind,
            SourceHiddenArgKind::HiddenHostcallBuffer);
  EXPECT_EQ(classifySourceHiddenArgByte(Args, 104)->Kind,
            SourceHiddenArgKind::HiddenHeapV1);
}

TEST(KernargLayout, NonHiddenAndMissingOffsetsAreNotHidden) {
  std::vector<KernelArgMeta> Args = {
      makeArg("n", 24, 4, "by_value"),
  };

  EXPECT_FALSE(classifySourceHiddenArgByte(Args, 24).has_value());
  EXPECT_FALSE(classifySourceHiddenArgByte(Args, 28).has_value());
}
