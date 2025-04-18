//===- disasm_options_test.c ----------------------------------------------===//
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

const char *ExpectedOut = "\n"
                          ":\tfile format elf64-amdgpu\n"
                          "\n"
                          "Disassembly of section .text:\n"
                          "foo:\n"
                          "\ts_load_dwordx2 s[0:1], s[4:5], 0x0               "
                          "          // 000000000000: C0060002 00000000 \n"
                          "\tv_mov_b32_e32 v2, 42                             "
                          "          // 000000000008: 7E0402AA \n"
                          "\ts_waitcnt lgkmcnt(0)                             "
                          "          // 00000000000C: BF8CC07F \n"
                          "\tv_mov_b32_e32 v0, s0                             "
                          "          // 000000000010: 7E000200 \n"
                          "\tv_mov_b32_e32 v1, s1                             "
                          "          // 000000000014: 7E020201 \n"
                          "\tflat_store_dword v[0:1], v2                      "
                          "          // 000000000018: DC700000 00000200 \n"
                          "\ts_endpgm                                         "
                          "          // 000000000020: BF810000 \n";

void printChars(const char *Bytes, size_t Count) {
  for (size_t I = 0; I < Count; I++) {
    printf("%c", Bytes[I]);
  }
}

void expect(const char *Expected, const char *Actual, size_t Count) {
  if (strlen(Expected) != Count || strncmp(Expected, Actual, Count)) {
    printf("FAILED: unexpected output\n");
    printf("expected:\n");
    printf("%s", Expected);
    printf("actual:\n");
    printChars(Actual, Count);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  size_t Size;
  char *Buf;
  char *Bytes;
  amd_comgr_data_t DataIn, DataOut;
  amd_comgr_data_set_t DataSetIn, DataSetOut;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  const char *DisAsmOptions[] = {"-file-headers"};
  size_t DisAsmOptionsCount = sizeof(DisAsmOptions) / sizeof(DisAsmOptions[0]);

  // Read input file
  Size = setBuf(TEST_OBJ_DIR "/reloc-asm.o", &Buf);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_cogmr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataIn);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataIn, Size, Buf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataIn, "DO_IN");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataIn);
  checkError(Status, "amd_cogmr_data_set_add");

  Status = amd_comgr_create_data_set(&DataSetOut);
  checkError(Status, "amd_cogmr_create_data_set");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_logging(DataAction, true);
  checkError(Status, "amd_comgr_action_info_set_logging");
  Status = amd_comgr_action_info_set_option_list(DataAction, DisAsmOptions,
                                                 DisAsmOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status =
      amd_comgr_do_action(AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE,
                          DataAction, DataSetIn, DataSetOut);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");

  size_t Count;
  Status = amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_SOURCE,
                                       &Count);
  checkError(Status, "amd_comgr_action_data_count");
  if (Count != 1) {
    printf("wrong number of source data objects (%zd returned, expected 1)\n",
           Count);
    exit(1);
  }

  Status =
      amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_LOG, &Count);
  checkError(Status, "amd_comgr_action_data_count");
  if (Count != 1) {
    printf("wrong number of log data objects (%zd returned, expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_action_data_get_data(
      DataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 0, &DataOut);
  checkError(Status, "amd_comgr_action_data_get_data");
  Status = amd_comgr_get_data(DataOut, &Count, NULL);
  checkError(Status, "amd_comgr_get_data");
  Bytes = (char *)calloc(Count, sizeof(char));
  Status = amd_comgr_get_data(DataOut, &Count, Bytes);
  checkError(Status, "amd_comgr_get_data");
  if (!Bytes) {
    printf("Failed, NULL Bytes\n");
    return 1;
  }
  expect(ExpectedOut, Bytes, Count);
  free(Bytes);
  Status = amd_comgr_release_data(DataOut);
  checkError(Status, "amd_comgr_release_data");

  Status = amd_comgr_action_data_get_data(DataSetOut, AMD_COMGR_DATA_KIND_LOG,
                                          0, &DataOut);
  checkError(Status, "amd_comgr_action_data_get_data");
  Status = amd_comgr_get_data(DataOut, &Count, NULL);
  checkError(Status, "amd_comgr_get_data");
  Bytes = (char *)calloc(Count, sizeof(char));
  Status = amd_comgr_get_data(DataOut, &Count, Bytes);
  checkError(Status, "amd_comgr_get_data");
  free(Bytes);
  Status = amd_comgr_release_data(DataOut);
  checkError(Status, "amd_comgr_release_data");

  Status = amd_comgr_destroy_data_set(DataSetOut);
  checkError(Status, "amd_comgr_destroy_data_set");

  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_release_data(DataIn);
  checkError(Status, "amd_comgr_release_data");
  free(Buf);

  return 0;
}
