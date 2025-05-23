//===- mangled_names_test.c -----------------------------------------------===//
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
  char *BufSource1, *BufSource2, *BufInclude;
  size_t SizeSource1, SizeSource2, SizeInclude;
  amd_comgr_data_t DataSource1, DataSource2, DataInclude;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;

  SizeSource1 = setBuf(TEST_OBJ_DIR "/source1.cl", &BufSource1);
  SizeSource2 = setBuf(TEST_OBJ_DIR "/source2.cl", &BufSource2);
  SizeInclude = setBuf(TEST_OBJ_DIR "/include-macro.h", &BufInclude);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource1);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource1, SizeSource1, BufSource1);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource1, "source1.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource1);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource2);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource2, SizeSource2, BufSource2);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource2, "source2.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource2);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &DataInclude);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataInclude, SizeInclude, BufInclude);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataInclude, "include-macro.h");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataInclude);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  Status =
      amd_comgr_action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 2) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 2)\n",
           Count);
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

  // Get bitcode mangled names
  amd_comgr_data_t DataBc;

  Status = amd_comgr_action_data_get_data(DataSetLinked, AMD_COMGR_DATA_KIND_BC,
                                          0, &DataBc);
  checkError(Status, "amd_comgr_action_data_get_data");

#if 0
  // write bitcode
  {
    size_t bytes_size = 0;
    char *bytes = NULL;

    Status = amd_comgr_get_data(DataBc, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    bytes = (char *) malloc(bytes_size);

    Status = amd_comgr_get_data(DataBc, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    const char *bitcode_file = "comgr_mangled.bc";
    FILE *file = fopen(bitcode_file, "wb");

    if (file)
      fwrite(bytes, bytes_size, 1, file);
    else
      return AMD_COMGR_STATUS_ERROR;

    fclose(file);
    free(bytes);
  }
#endif

  size_t NumNames;
  Status = amd_comgr_populate_mangled_names(DataBc, &NumNames);
  checkError(Status, "amd_comgr_populate_mangled_names");

  if (NumNames != 4) {
    printf("amd_populate_mangled_names Failed: "
           "produced %zu bitcode names (expected 4)\n",
           NumNames);
    exit(1);
  }

  const char *BcNames[] = {"source1", "__clang_ocl_kern_imp_source1", "source2", "__clang_ocl_kern_imp_source2"};

  for (size_t I = 0; I < NumNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *MName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, MName);
    checkError(Status, "amd_comgr_get_mangled_name");

    if (!BcNames[I]) {
      printf("Failed, bcNames[%ld] NULL\n", I);
      return 1;
    }

    if (strcmp(MName, BcNames[I])) {
      printf("amd_get_mangled_name from bc Failed: "
             "produced '%s' (expected '%s')\n",
             MName, BcNames[I]);
      exit(1);
    }

    free(MName);
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
  checkError(Status, "amd_comgr_action_data_get_data");

  Status = amd_comgr_populate_mangled_names(DataExec, &NumNames);
  checkError(Status, "amd_comgr_populate_mangled_names");

  if (NumNames != 6) {
    printf("amd_populate_mangled_names Failed: "
           "produced %zu executable names (expected 6)\n",
           NumNames);
    exit(1);
  }

  const char *ExecNames[] = {"source1", "source1.kd", "__clang_ocl_kern_imp_source1", "source2", "source2.kd", "__clang_ocl_kern_imp_source2"};

  for (size_t I = 0; I < NumNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *MName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, MName);
    checkError(Status, "amd_comgr_get_mangled_name");

    if (!ExecNames[I]) {
      printf("Failed, execNames[%ld] NULL\n", I);
      return 1;
    }

    if (strcmp(MName, ExecNames[I])) {
      printf("amd_get_mangled_name from executable Failed: "
             "produced '%s' (expected '%s')\n",
             MName, ExecNames[I]);
      exit(1);
    }

    free(MName);
  }

  Status = amd_comgr_release_data(DataSource1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataSource2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataInclude);
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
  free(BufSource1);
  free(BufSource2);
  free(BufInclude);
}
