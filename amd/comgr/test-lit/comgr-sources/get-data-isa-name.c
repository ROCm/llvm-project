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
  char *Buf;
  size_t Size;
  char IsaName[MAX_ISA_NAME_SIZE];
  amd_comgr_data_t Data;

  if (argc != 3)
    fail("Usage: get-data-isa-name <code-object-path> <expected-isa-name>");
  
  Size = setBuf(argv[1], &Buf);

  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &Data));
  amd_comgr_(set_data(Data, Size, Buf));
  amd_comgr_(get_data_isa_name(Data, &Size, IsaName));
  if (strcmp(IsaName, argv[2]))
    fail("incorrect isa name: expected %s, saw %s", argv[2], IsaName);
  amd_comgr_(release_data(Data));
  free(Buf);
  return 0;
}
