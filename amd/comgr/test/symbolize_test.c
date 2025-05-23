//===- symbolize_test.c ---------------------------------------------------===//
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

#define CHECK(ptr, ...)                                                        \
  do {                                                                         \
    if ((ptr) == NULL) {                                                       \
      fprintf(stderr, "Error: ");                                              \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, " at %s:%d\n", __FILE__, __LINE__);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

typedef struct Container {
  char *Data;
  int Sz;
} container_t;

void collectSymbolizedString(const char *Input, void *Data) {
  int Sz = strlen(Input);
  container_t *Ptr = (container_t *)Data;
  Ptr->Data = (char *)malloc(Sz + 1);
  Ptr->Data[Sz] = '\0';
  Ptr->Sz = Sz;
  memcpy(Ptr->Data, Input, Sz);
}

void testSymbolizedString(container_t *SymbolContainer) {

  char *SymbolStr = SymbolContainer->Data;
  CHECK(SymbolStr, "Failed, symbol_str is NULL.\n");

  char *SpacePos = strchr(SymbolStr, ' ');
  CHECK(SpacePos, "Expected spaces in %s\n", SymbolStr);

  char *LineColPos = strchr(SymbolStr, ':');
  CHECK(LineColPos, "Expected line:column information in %s\n", SymbolStr);

  char *NewlinePos = strchr(SymbolStr, '\n');
  CHECK(NewlinePos, "Expected '\\n' in %s", SymbolStr);

  size_t FuncNameSize = SpacePos - SymbolStr;
  char *FuncName = (char *)malloc(sizeof(char) * (FuncNameSize + 1));

  strncpy(FuncName, SymbolStr, FuncNameSize);
  FuncName[FuncNameSize] = '\0';

  size_t LineColSize = NewlinePos - LineColPos;
  char *LineCol = (char *)malloc(sizeof(char) * (LineColSize));

  strncpy(LineCol, LineColPos + 1, LineColSize);
  LineCol[LineColSize - 1] = '\0';

  if (strcmp(FuncName,
             "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz") &&
      strcmp(LineCol, "46:7 (approximate)")) {
    printf("mismatch:\n");
    printf("expected symbolized function name: "
           "'bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz'\n");
    printf("actual symbolized function name: '%s'\n", FuncName);
    printf("expected symbolized line:column output: '46:7 (approximate)'\n");
    printf("actual symbolized line:column output: '%s'\n", LineCol);
    exit(0);
  }

  printf("symbolized string is %s", SymbolStr);
  free(FuncName);
  free(LineCol);
  free(SymbolStr);

  return;
}

int main(int argc, char *argv[]) {
  size_t Size;
  char *Buf;
  amd_comgr_data_t DataIn;
  amd_comgr_status_t Status;
  amd_comgr_symbolizer_info_t Symbolizer;
  container_t UserData;

  // Read input file
  Size = setBuf(TEST_OBJ_DIR "/symbolize-debug.so", &Buf);

  // Create data object
  {
    printf("Test create input data set\n");
    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataIn);
    checkError(Status, "amd_comgr_create_data");
    Status = amd_comgr_set_data(DataIn, Size, Buf);
    checkError(Status, "amd_comgr_set_data");
    Status = amd_comgr_set_data_name(DataIn, "symbolize-debug.so");
    checkError(Status, "amd_comgr_set_data_name");
  }

  // Create symbolizer info and symbolize
  {
    printf("Test create symbolizer info\n");
    Status = amd_comgr_create_symbolizer_info(DataIn, &collectSymbolizedString,
                                              &Symbolizer);
    checkError(Status, "amd_comgr_create_symbolizer_info");
    // Use this command to get valid address
    // llvm-objdump --triple=amdgcn-amd-amdhsa -l --mcpu=gfx900 --disassemble
    // --source symbolize-debug.so
    uint64_t Address = 0x128;
    Status = amd_comgr_symbolize(Symbolizer, Address, 1, (void *)&UserData);
    checkError(Status, "amd_comgr_symbolize");

    testSymbolizedString(&UserData);
  }

  // Destroy symbolizer info
  {
    printf("Test destroy symbolizer info\n");
    Status = amd_comgr_destroy_symbolizer_info(Symbolizer);
    checkError(Status, "amd_comgr_destroy_symbolizer_info");
    Status = amd_comgr_release_data(DataIn);
    checkError(Status, "amd_comgr_release_data");
    free(Buf);
  }

  return 0;
}
