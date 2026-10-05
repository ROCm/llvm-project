//===- LateActionTest.cpp - Comgr use during process exit -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(Call)                                                            \
  do {                                                                         \
    amd_comgr_status_t Status = (Call);                                        \
    if (Status != AMD_COMGR_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "%s failed: %d\n", #Call, (int)Status);             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int preprocess() {
  static const char Source[] = "int x;\n";
  amd_comgr_data_set_t Input, Output;
  amd_comgr_action_info_t Action;
  amd_comgr_data_t Data;
  size_t Count;

  CHECK(amd_comgr_create_data_set(&Input));
  CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &Data));
  CHECK(amd_comgr_set_data(Data, sizeof(Source) - 1, Source));
  CHECK(amd_comgr_set_data_name(Data, "source.cl"));
  CHECK(amd_comgr_data_set_add(Input, Data));
  CHECK(amd_comgr_release_data(Data));
  CHECK(amd_comgr_create_action_info(&Action));
  CHECK(amd_comgr_action_info_set_language(Action,
                                           AMD_COMGR_LANGUAGE_OPENCL_1_2));
  CHECK(
      amd_comgr_action_info_set_isa_name(Action, "amdgcn-amd-amdhsa--gfx900"));
  CHECK(amd_comgr_action_info_set_vfs(Action, true));
  CHECK(amd_comgr_create_data_set(&Output));
  CHECK(amd_comgr_do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR, Action,
                            Input, Output));
  CHECK(
      amd_comgr_action_data_count(Output, AMD_COMGR_DATA_KIND_SOURCE, &Count));
  if (Count != 1) {
    std::fprintf(stderr, "expected one preprocessed source, got %zu\n", Count);
    return 1;
  }
  CHECK(amd_comgr_destroy_data_set(Output));
  CHECK(amd_comgr_destroy_action_info(Action));
  CHECK(amd_comgr_destroy_data_set(Input));
  return 0;
}

static void lateAction() {
  if (preprocess())
    std::_Exit(1);
}

// Avoid LLVM's shared test main, which initializes its real filesystem.
int main(int argc, char **argv) {
#ifdef _WIN32
  _putenv_s("AMD_COMGR_CACHE", "0");
  _putenv_s("AMD_COMGR_USE_VFS", "");
  _putenv_s("AMD_COMGR_SAVE_TEMPS", "");
#else
  setenv("AMD_COMGR_CACHE", "0", 1);
  unsetenv("AMD_COMGR_USE_VFS");
  unsetenv("AMD_COMGR_SAVE_TEMPS");
#endif
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(LateActionDeathTest, PreprocessDuringExit) {
  // Re-execute so other tests cannot change the singleton initialization order.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        // Keep the ISA cache alive until after the exit callback.
        size_t IsaCount;
        if (amd_comgr_get_isa_count(&IsaCount) != AMD_COMGR_STATUS_SUCCESS ||
            std::atexit(lateAction) || preprocess())
          std::_Exit(1);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}
