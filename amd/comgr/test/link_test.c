//===- link_test.c --------------------------------------------------------===//
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
  size_t Size1, Size2;
  char *Buf1, *Buf2;
  size_t Count;
  amd_comgr_data_t DataIn1, DataIn2;
  amd_comgr_data_set_t DataSetIn, DataSetOutReloc, DataSetOutExec;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  // Read input file
  Size1 = setBuf(TEST_OBJ_DIR "/reloc1.o", &Buf1);
  Size2 = setBuf(TEST_OBJ_DIR "/reloc2.o", &Buf2);

  // Create data object
  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_cogmr_create_data_set");

  // File 1
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataIn1);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataIn1, Size1, Buf1);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataIn1, "DO_IN1");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataIn1);
  checkError(Status, "amd_cogmr_data_set_add");

  // File 2
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataIn2);
  checkError(Status, "amd_comgr_create_data_2");
  Status = amd_comgr_set_data(DataIn2, Size2, Buf2);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataIn2, "DO_IN2");
  checkError(Status, "amd_comgr_set_data_name_2");
  Status = amd_comgr_data_set_add(DataSetIn, DataIn2);
  checkError(Status, "amd_cogmr_data_set_add_2");

  Status = amd_comgr_create_data_set(&DataSetOutReloc);
  checkError(Status, "amd_cogmr_create_data_set");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  amd_comgr_action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_language");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE,
                               DataAction, DataSetIn, DataSetOutReloc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetOutReloc,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");
  if (Count != 1) {
    printf("Failed, output %zd relocatable objects (should output 1)\n", Count);
    exit(1);
  }

  Status = amd_comgr_create_data_set(&DataSetOutExec);
  checkError(Status, "amd_cogmr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetIn, DataSetOutExec);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetOutExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");
  if (Count != 1) {
    printf("Failed, output %zd executable objects (should output 1)\n", Count);
    exit(1);
  }

  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetOutReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetOutExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_release_data(DataIn1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataIn2);
  checkError(Status, "amd_comgr_release_data");
  free(Buf1);
  free(Buf2);

  return 0;
}
