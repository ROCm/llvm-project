//===- disasm_llvm_so_test.c ----------------------------------------------===//
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
  amd_comgr_data_t DataIn1, DataIn2, DataOut1, DataOut2;
  amd_comgr_data_set_t DataSetIn, DataSetOut;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  // Read input file
  Size1 = setBuf(TEST_OBJ_DIR "/shared.so", &Buf1);
  Size2 = setBuf(TEST_OBJ_DIR "/shared.so", &Buf2);

  // Create data object
  {
    printf("Test create input data set\n");

    Status = amd_comgr_create_data_set(&DataSetIn);
    checkError(Status, "amd_cogmr_create_data_set");

    // File 1
    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataIn1);
    checkError(Status, "amd_comgr_create_data");
    Status = amd_comgr_set_data(DataIn1, Size1, Buf1);
    checkError(Status, "amd_comgr_set_data");
    Status = amd_comgr_set_data_name(DataIn1, "DO_IN1");
    checkError(Status, "amd_comgr_set_data_name");
    Status = amd_comgr_data_set_add(DataSetIn, DataIn1);
    checkError(Status, "amd_cogmr_data_set_add");

    // File 2
    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataIn2);
    checkError(Status, "amd_comgr_create_data_2");
    Status = amd_comgr_set_data(DataIn2, Size2, Buf2);
    checkError(Status, "amd_comgr_set_data");
    Status = amd_comgr_set_data_name(DataIn2, "DO_IN2");
    checkError(Status, "amd_comgr_set_data_name_2");
    Status = amd_comgr_data_set_add(DataSetIn, DataIn2);
    checkError(Status, "amd_cogmr_data_set_add_2");
  }

  {
    printf("Test create empty output data set\n");

    Status = amd_comgr_create_data_set(&DataSetOut);
    checkError(Status, "amd_cogmr_create_data_set");
  }

  {
    printf("Test create action info\n");

    Status = amd_comgr_create_action_info(&DataAction);
    checkError(Status, "amd_comgr_create_action_info");
    Status = amd_comgr_action_info_set_isa_name(DataAction,
                                                "amdgcn-amd-amdhsa--gfx900");
    checkError(Status, "amd_comgr_action_info_set_isa_name");

    // Do disassembly action
    Status =
        amd_comgr_do_action(AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,
                            DataAction, DataSetIn, DataSetOut);
    checkError(Status, "amd_comgr_do_action");

    Status = amd_comgr_destroy_data_set(DataSetIn);
    checkError(Status, "amd_comgr_destroy_data_set");
  }

  {
    printf("Test action outputs\n");
    // There should be two output data object
    size_t Count;
    Status = amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_SOURCE,
                                         &Count);
    checkError(Status, "amd_comgr_action_data_count");
    if (Count == 2) {
      printf("Passed, output data object returned = 2\n");
    } else {
      printf("Failed, Output data object returned = %zd\n", Count);
    }

    // Retrieve the result data object 1 from dataSetOut
    Status = amd_comgr_action_data_get_data(
        DataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 0, &DataOut1);
    checkError(Status, "amd_comgr_action_data_get_data");
    Status = amd_comgr_get_data(DataOut1, &Count, NULL);
    checkError(Status, "amd_comgr_get_data");
    char *Bytes = (char *)malloc(Count);
    Status = amd_comgr_get_data(DataOut1, &Count, Bytes);
    checkError(Status, "amd_comgr_get_data");

    printf("Output = \n");
    for (size_t I = 0; I < Count; I++) {
      if (!Bytes[I]) {
        printf("Failed, NULL Bytes[%ld]\n", I);
        return 1;
      }
      printf("%c", Bytes[I]);
    }
    free(Bytes);

    // Retrieve the result data object 2 from dataSetOut
    Status = amd_comgr_action_data_get_data(
        DataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 1, &DataOut2);
    checkError(Status, "amd_comgr_action_data_get_data");
    Status = amd_comgr_get_data(DataOut2, &Count, NULL);
    checkError(Status, "amd_comgr_get_data");
    char *Bytes2 = (char *)malloc(Count);
    Status = amd_comgr_get_data(DataOut2, &Count, Bytes2);
    checkError(Status, "amd_comgr_get_data");

    printf("Output = \n");
    for (size_t I = 0; I < Count; I++) {
      if (!Bytes2[I]) {
        printf("Failed, NULL Bytes2[%ld]\n", I);
        return 1;
      }
      printf("%c", Bytes2[I]);
    }
    free(Bytes2);

    Status = amd_comgr_destroy_data_set(DataSetOut);
    checkError(Status, "amd_comgr_destroy_data_set");
  }

  {
    printf("Cleanup ...\n");
    Status = amd_comgr_destroy_action_info(DataAction);
    checkError(Status, "amd_comgr_destroy_action_info");
    Status = amd_comgr_release_data(DataIn1);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_release_data(DataIn2);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_release_data(DataOut1);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_release_data(DataOut2);
    checkError(Status, "amd_comgr_release_data");
    free(Buf1);
    free(Buf2);
  }

  return 0;
}
