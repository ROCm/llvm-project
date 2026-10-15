//===- IncludeMaterializationTest.cpp - Header materialization tests ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void setEnv(const char *Name, const char *Value) {
#ifdef _WIN32
  _putenv_s(Name, Value ? Value : "");
#else
  if (Value)
    setenv(Name, Value, /* Overwrite */ 1);
  else
    unsetenv(Name);
#endif
}

// AMD_COMGR_USE_VFS and AMD_COMGR_SAVE_TEMPS both take precedence over the
// per-action setting these tests pin, and the command cache would let the
// second mode's compile be served from the first mode's result. Comgr reads
// each of these once into a static, so clear them before any action runs.
class EnvironmentSetup : public ::testing::Environment {
public:
  void SetUp() override {
    setEnv("AMD_COMGR_USE_VFS", nullptr);
    setEnv("AMD_COMGR_SAVE_TEMPS", nullptr);
    setEnv("AMD_COMGR_CACHE", "0");
  }
};

[[maybe_unused]] const auto *Registered =
    ::testing::AddGlobalTestEnvironment(new EnvironmentSetup);

struct Include {
  const char *Name;
  const char *Contents;
};

// Two distinct subdirectory shapes, since the include path is built by
// appending the data name to the include directory.
const Include Includes[] = {
    {"subdir/header1.h", "int x = 1;"},
    {"sub/dir/header2.h", "int y = 1;"},
    {"sub/dir/header3.h", "int z = 1;"},
};

const char *const Source = "#include \"subdir/header1.h\"\n"
                           "#include \"sub/dir/header2.h\"\n"
                           "#include \"sub/dir/header3.h\"\n";

std::string preprocess(bool UseVFS) {
  amd_comgr_data_set_t In, Out;
  amd_comgr_action_info_t Action;
  amd_comgr_data_t Data;

  EXPECT_COMGR(create_data_set(&In));

  EXPECT_COMGR(create_data(AMD_COMGR_DATA_KIND_SOURCE, &Data));
  EXPECT_COMGR(set_data(Data, strlen(Source), Source));
  EXPECT_COMGR(set_data_name(Data, "source.cl"));
  EXPECT_COMGR(data_set_add(In, Data));
  EXPECT_COMGR(release_data(Data));

  for (const Include &I : Includes) {
    EXPECT_COMGR(create_data(AMD_COMGR_DATA_KIND_INCLUDE, &Data));
    EXPECT_COMGR(set_data(Data, strlen(I.Contents), I.Contents));
    EXPECT_COMGR(set_data_name(Data, I.Name));
    EXPECT_COMGR(data_set_add(In, Data));
    EXPECT_COMGR(release_data(Data));
  }

  EXPECT_COMGR(create_action_info(&Action));
  EXPECT_COMGR(action_info_set_language(Action, AMD_COMGR_LANGUAGE_OPENCL_1_2));
  EXPECT_COMGR(action_info_set_isa_name(Action, "amdgcn-amd-amdhsa--gfx900"));
  EXPECT_COMGR(action_info_set_vfs(Action, UseVFS));

  EXPECT_COMGR(create_data_set(&Out));
  EXPECT_COMGR(
      do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR, Action, In, Out));

  size_t Count = 0;
  EXPECT_COMGR(action_data_count(Out, AMD_COMGR_DATA_KIND_SOURCE, &Count));
  EXPECT_EQ(size_t(1), Count);

  std::string Result;
  if (Count == 1) {
    EXPECT_COMGR(
        action_data_get_data(Out, AMD_COMGR_DATA_KIND_SOURCE, 0, &Data));

    size_t Size = 0;
    EXPECT_COMGR(get_data(Data, &Size, nullptr));
    Result.resize(Size);
    EXPECT_COMGR(get_data(Data, &Size, Result.data()));

    EXPECT_COMGR(release_data(Data));
  }

  EXPECT_COMGR(destroy_data_set(Out));
  EXPECT_COMGR(destroy_action_info(Action));
  EXPECT_COMGR(destroy_data_set(In));
  return Result;
}

class IncludeMaterializationTest : public ::testing::TestWithParam<bool> {};

// Includes are written into the in-memory VFS rather than to real temp files,
// so cover both filesystems: the contents have to reach the preprocessor
// either way, including from nested subdirectories.
TEST_P(IncludeMaterializationTest, SubdirectoryIncludesAreVisible) {
  const std::string Preprocessed = preprocess(GetParam());
  ASSERT_FALSE(Preprocessed.empty());

  for (const Include &I : Includes)
    EXPECT_NE(std::string::npos, Preprocessed.find(I.Contents))
        << "missing contents of " << I.Name;
}

INSTANTIATE_TEST_SUITE_P(FileSystem, IncludeMaterializationTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &Info) {
                           return Info.param ? "VFS" : "RealFS";
                         });

} // namespace
