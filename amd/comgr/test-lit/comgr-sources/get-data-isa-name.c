//===- get-data-isa-name.c -------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  char *BufReloc, *BufExec;
  size_t SizeReloc, SizeExec;
  size_t SizeIsa = MAX_ISA_NAME_SIZE;
  char IsaName[MAX_ISA_NAME_SIZE];
  amd_comgr_data_t DataReloc, DataExec;

  if (argc != 4)
    fail("Usage: get-data-isa-name <code-object-path> <code-shared-object-path <expected-isa-name>");

  SizeReloc = setBuf(argv[1], &BufReloc);
  SizeExec = setBuf(argv[2], &BufExec);

  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataReloc));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataExec));
  amd_comgr_(set_data(DataReloc, SizeReloc, BufReloc));
  amd_comgr_(set_data(DataExec, SizeExec, BufExec));
  amd_comgr_(get_data_isa_name(DataReloc, &SizeIsa, IsaName));
  if (strcmp(IsaName, argv[3]))
    fail("incorrect isa name: expected %s, saw %s", argv[3], IsaName);
  amd_comgr_(get_data_isa_name(DataExec, &SizeIsa, IsaName));
  if (strcmp(IsaName, argv[3]))
    fail("incorrect isa name: expected %s, saw %s", argv[3], IsaName);
  amd_comgr_(release_data(DataReloc));
  amd_comgr_(release_data(DataExec));
  free(BufReloc);
  free(BufExec);
  return 0;
}
