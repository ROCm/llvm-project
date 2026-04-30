//===- hotswap-rewrite.c - Test HotSwap rewrite API ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    amd_comgr_data_t dummy_output;
    amd_comgr_data_t dummy_input = {0};
    amd_comgr_status_t Status =
        amd_comgr_hotswap_rewrite(dummy_input, NULL, NULL, &dummy_output);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
      fail("rewrite with NULL args: expected INVALID_ARGUMENT");
    printf("NULL_ARGS: INVALID_ARGUMENT\n");
    return 0;
  }

  if (argc < 4)
    fail("usage: hotswap-rewrite <elf_file> <source_isa> <target_isa> "
         "[--zero-size | --output <path> [--allow-grow]]");

  const char *ElfFile = argv[1];
  const char *SourceISA = argv[2];
  const char *TargetISA = argv[3];

  // Optional trailing flags. --zero-size is used by the negative tests that
  // exercise INVALID_ARGUMENT paths; --output <path> is used by tests that
  // save the rewrite output for inspection by llvm-readelf / llvm-objdump.
  // --allow-grow is an opt-in modifier for --output that lets the output
  // ELF be larger than the input -- used by trampoline-based rewrites
  // (e.g. WMMA splitter) which append to .text. By default the size check
  // remains active so in-place tests catch accidental .text growth.
  int ZeroSize = 0;
  int AllowGrow = 0;
  const char *OutputPath = NULL;
  int Pos = 4;
  while (Pos < argc) {
    if (strcmp(argv[Pos], "--zero-size") == 0) {
      ZeroSize = 1;
      Pos += 1;
    } else if (strcmp(argv[Pos], "--output") == 0) {
      if (Pos + 1 >= argc)
        fail("--output requires a path argument");
      OutputPath = argv[Pos + 1];
      Pos += 2;
    } else if (strcmp(argv[Pos], "--allow-grow") == 0) {
      AllowGrow = 1;
      Pos += 1;
    } else {
      fail("unknown flag '%s'", argv[Pos]);
    }
  }
  if (AllowGrow && !OutputPath)
    fail("--allow-grow requires --output");

  char *ElfBuf;
  size_t ElfSize = (size_t)setBuf(ElfFile, &ElfBuf);

  amd_comgr_data_t InputData;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &InputData));
  if (!ZeroSize) {
    amd_comgr_(set_data(InputData, ElfSize, ElfBuf));
  }

  amd_comgr_data_t OutputData;
  amd_comgr_status_t Status =
      amd_comgr_hotswap_rewrite(InputData, SourceISA, TargetISA, &OutputData);

  if (Status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("RESULT: INVALID_ARGUMENT\n");
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("unexpected error status %d", (int)Status);

  if (OutputPath) {
    size_t OutSize = 0;
    amd_comgr_(get_data(OutputData, &OutSize, NULL));
    if (!AllowGrow && OutSize != ElfSize)
      fail("output size %zu != input size %zu (pass --allow-grow if the "
           "rewrite is expected to append trampolines)",
           OutSize, ElfSize);
    dumpData(OutputData, OutputPath);
  } else {
    size_t OutSize = 0;
    amd_comgr_(get_data(OutputData, &OutSize, NULL));

    if (OutSize != ElfSize)
      fail("output size %zu != input size %zu", OutSize, ElfSize);

    char *OutBuf = (char *)malloc(OutSize);
    if (!OutBuf)
      fail("malloc failed");
    amd_comgr_(get_data(OutputData, &OutSize, OutBuf));

    if (memcmp(OutBuf, ElfBuf, ElfSize) != 0)
      fail("output content differs from input");

    free(OutBuf);
  }
  amd_comgr_(release_data(OutputData));
  amd_comgr_(release_data(InputData));
  free(ElfBuf);

  printf("RESULT: SUCCESS\n");
  return 0;
}
