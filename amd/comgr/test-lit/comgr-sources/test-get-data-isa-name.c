//===- test-get-data-isa-name.c ------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

// Each path must be rejected with AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT.
// Asserting the exact status keeps this from passing on an unrelated failure.
static int expectFail(int argc, char *argv[]) {
  if (argc < 1)
    fail("Usage: test-get-data-isa-name --expect-fail <code-object-path>...");

  for (int I = 0; I < argc; ++I) {
    char *Buf;
    size_t Size = setBuf(argv[I], &Buf);
    size_t SizeIsa = 0;
    amd_comgr_data_t Data;

    amd_comgr_(create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &Data));
    amd_comgr_(set_data(Data, Size, Buf));

    amd_comgr_status_t Status =
        amd_comgr_get_data_isa_name(Data, &SizeIsa, NULL);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
      const char *Reason = "";
      amd_comgr_status_string(Status, &Reason);
      free(Buf);
      fail("get_data_isa_name(%s): expected "
           "AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, saw %s",
           argv[I], Reason);
    }

    amd_comgr_(release_data(Data));
    free(Buf);
  }

  return 0;
}

int main(int argc, char *argv[]) {
  char *BufReloc, *BufExec;
  size_t SizeReloc, SizeExec;
  size_t SizeIsa = 0;
  char *IsaName;
  amd_comgr_data_t DataReloc, DataExec;

  if (argc > 1 && strncmp(argv[1], "--", 2) == 0) {
    if (strcmp(argv[1], "--expect-fail") == 0)
      return expectFail(argc - 2, argv + 2);
    fprintf(stderr, "error: unknown argument: %s\n", argv[1]);
    return 1;
  }

  if (argc != 4)
    fail("Usage: test-get-data-isa-name <code-object-path> "
         "<code-shared-object-path> <expected-isa-name>\n"
         "       test-get-data-isa-name --expect-fail <code-object-path>...");

  SizeReloc = setBuf(argv[1], &BufReloc);
  SizeExec = setBuf(argv[2], &BufExec);

  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataReloc));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataExec));
  amd_comgr_(set_data(DataReloc, SizeReloc, BufReloc));
  amd_comgr_(set_data(DataExec, SizeExec, BufExec));

  // Size the buffer from the API (call with NULL to get the length, then
  // allocate), which is how a real client is expected to use
  // amd_comgr_get_data_isa_name().
  amd_comgr_(get_data_isa_name(DataReloc, &SizeIsa, NULL));
  IsaName = (char *)malloc(SizeIsa);
  if (!IsaName)
    fail("malloc");
  amd_comgr_(get_data_isa_name(DataReloc, &SizeIsa, IsaName));
  if (strcmp(IsaName, argv[3]))
    fail("incorrect isa name: expected %s, saw %s", argv[3], IsaName);
  free(IsaName);

  amd_comgr_(get_data_isa_name(DataExec, &SizeIsa, NULL));
  IsaName = (char *)malloc(SizeIsa);
  if (!IsaName)
    fail("malloc");
  amd_comgr_(get_data_isa_name(DataExec, &SizeIsa, IsaName));
  if (strcmp(IsaName, argv[3]))
    fail("incorrect isa name: expected %s, saw %s", argv[3], IsaName);
  free(IsaName);

  amd_comgr_(release_data(DataReloc));
  amd_comgr_(release_data(DataExec));
  free(BufReloc);
  free(BufExec);
  return 0;
}
