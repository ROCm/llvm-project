//===- metadata_tp_test.c -------------------------------------------------===//
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

static void checkMetadataString(amd_comgr_metadata_node_t Meta, const char *Key,
                                const char *Expected) {
  amd_comgr_metadata_node_t Value;
  amd_comgr_status_t Status = amd_comgr_metadata_lookup(Meta, Key, &Value);
  checkError(Status, "amd_comgr_metadata_lookup");

  size_t Size;
  Status = amd_comgr_get_metadata_string(Value, &Size, NULL);
  checkError(Status, "amd_comgr_get_metadata_string");

  char *Actual = (char *)malloc(Size);
  if (!Actual)
    fail("malloc");
  Status = amd_comgr_get_metadata_string(Value, &Size, Actual);
  checkError(Status, "amd_comgr_get_metadata_string");

  if (strcmp(Actual, Expected) != 0)
    fail("%s: expected %s, got %s", Key, Expected, Actual);

  free(Actual);
  Status = amd_comgr_destroy_metadata(Value);
  checkError(Status, "amd_comgr_destroy_metadata");
}

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;

  amd_comgr_metadata_node_t Gfx950Meta;
  Status = amd_comgr_get_isa_metadata("amdgcn-amd-amdhsa--gfx950", &Gfx950Meta);
  checkError(Status, "amd_comgr_get_isa_metadata");
  checkMetadataString(Gfx950Meta, "LocalMemorySize", "163840");
  checkMetadataString(Gfx950Meta, "LDSBankCount", "64");
  Status = amd_comgr_destroy_metadata(Gfx950Meta);
  checkError(Status, "amd_comgr_destroy_metadata");

  // how many isa_names do we support?
  size_t IsaCounts;
  Status = amd_comgr_get_isa_count(&IsaCounts);
  checkError(Status, "amd_comgr_get_isa_count");
  printf("isa count = %zu\n\n", IsaCounts);

  // print the list
  printf("*** List of ISA names supported:\n");
  for (size_t I = 0; I < IsaCounts; I++) {
    const char *Name;
    Status = amd_comgr_get_isa_name(I, &Name);
    checkError(Status, "amd_comgr_get_isa_name");
    printf("%zu: %s\n", I, Name);
    amd_comgr_metadata_node_t Meta;
    Status = amd_comgr_get_isa_metadata(Name, &Meta);
    checkError(Status, "amd_comgr_get_isa_metadata");
    int Indent = 1;
    Status = amd_comgr_iterate_map_metadata(Meta, printEntry, (void *)&Indent);
    checkError(Status, "amd_comgr_iterate_map_metadata");
    Status = amd_comgr_destroy_metadata(Meta);
    checkError(Status, "amd_comgr_destroy_metadata");
  }

  return 0;
}
