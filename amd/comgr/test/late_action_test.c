//===- late_action_test.c - Comgr use during process exit -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(Call)                                                            \
  do {                                                                         \
    amd_comgr_status_t Status = (Call);                                        \
    if (Status != AMD_COMGR_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "%s failed: %d\n", #Call, (int)Status);                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int preprocess(void) {
  static const char Source[] = "int x;\n";
  amd_comgr_data_set_t Input, Output;
  amd_comgr_action_info_t Action;
  amd_comgr_data_t Data;
  size_t Count;

  CHECK(amd_comgr_create_data_set(&Input));
  CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &Data));
  CHECK(amd_comgr_set_data(Data, strlen(Source), Source));
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
    fprintf(stderr, "expected one preprocessed source, got %zu\n", Count);
    return 1;
  }
  CHECK(amd_comgr_destroy_data_set(Output));
  CHECK(amd_comgr_destroy_action_info(Action));
  CHECK(amd_comgr_destroy_data_set(Input));
  return 0;
}

static void late_action(void) {
  if (preprocess())
    _Exit(1);
}

int main(void) {
  // The first action constructs LLVM's lazy real-filesystem singleton. Since
  // this handler was registered first, it runs after that singleton's cleanup.
  if (atexit(late_action))
    return 1;
  if (preprocess())
    _Exit(1);
  return 0;
}
