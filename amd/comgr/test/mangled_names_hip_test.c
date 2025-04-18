//===- mangled_names_hip_test.c -------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char *BufSource;
  size_t SizeSource;
  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;
  const char *CompileOptions[] = {"-nogpulib", "-nogpuinc"};
  size_t CompileOptionsCount =
      sizeof(CompileOptions) / sizeof(CompileOptions[0]);

  SizeSource = setBuf(TEST_OBJ_DIR "/source1.hip", &BufSource);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "source1.hip");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_option_list(DataAction, CompileOptions,
                                                 CompileOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, DataAction,
      DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  Status =
      amd_comgr_action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  // Get bitcode mangled names
  amd_comgr_data_t DataBc;

  Status = amd_comgr_action_data_get_data(DataSetBc, AMD_COMGR_DATA_KIND_BC, 0,
                                          &DataBc);
  checkError(Status, "amd_comgr_action_data_get_data");

#if 1
  // write bitcode
  {
    size_t BytesSize = 0;
    char *Bytes = NULL;

    Status = amd_comgr_get_data(DataBc, &BytesSize, Bytes);
    checkError(Status, "amd_comgr_get_data");

    Bytes = (char *)malloc(BytesSize);

    Status = amd_comgr_get_data(DataBc, &BytesSize, Bytes);
    checkError(Status, "amd_comgr_get_data");

    const char *BitcodeFile = "comgr_mangled.bc";
    FILE *File = fopen(BitcodeFile, "wb");

    if (File)
      fwrite(Bytes, BytesSize, 1, File);
    else
      return AMD_COMGR_STATUS_ERROR;

    fclose(File);
    free(Bytes);
  }
#endif

  size_t NumNames;
  Status = amd_comgr_populate_mangled_names(DataBc, &NumNames);
  checkError(Status, "amd_comgr_populate_mangled_names");

  char *MangledSubstr = "__hip_cuid_";
  bool BcFound = false;

  for (size_t I = 0; I < NumNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *MName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, MName);
    checkError(Status, "amd_comgr_get_mangled_name");

    if (strstr(MName, MangledSubstr)) {
      BcFound = true;
    }

    free(MName);
  }

  if (!BcFound) {
    printf("amd_get_mangled_name from bc Failed: "
           "(expected '%s*')\n",
           MangledSubstr);
    exit(1);
  }

  Status = amd_comgr_create_data_set(&DataSetLinked);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, DataAction,
                               DataSetBc, DataSetLinked);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetLinked, AMD_COMGR_DATA_KIND_BC,
                                       &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_BC_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               DataAction, DataSetLinked, DataSetReloc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetReloc,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu source objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu executable objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Get Mangled Names
  amd_comgr_data_t DataExec;

  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec);

  Status = amd_comgr_populate_mangled_names(DataExec, &NumNames);

  bool ExecFound = false;

  for (size_t I = 0; I < NumNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *MName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, MName);
    checkError(Status, "amd_comgr_get_mangled_name");

    if (strstr(MName, MangledSubstr)) {
      ExecFound = true;
    }

    free(MName);
  }

  if (!ExecFound) {
    printf("amd_get_mangled_name from exec Failed: "
           "(expected '%s*')\n",
           MangledSubstr);
    exit(1);
  }

  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataBc);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinked);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(BufSource);
}
