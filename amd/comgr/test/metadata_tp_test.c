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
  checkMetadataString(Gfx950Meta, "TotalNumVGPRs", "512");
  checkMetadataString(Gfx950Meta, "AddressableNumVGPRs", "512");
  // gfx950 has no image instructions.
  checkMetadataString(Gfx950Meta, "ImageSupport", "0");
  Status = amd_comgr_destroy_metadata(Gfx950Meta);
  checkError(Status, "amd_comgr_destroy_metadata");

  // gfx6 addresses 32 KiB of LDS, not the 64 KiB of gfx7 onwards.
  amd_comgr_metadata_node_t Gfx600Meta;
  Status = amd_comgr_get_isa_metadata("amdgcn-amd-amdhsa--gfx600", &Gfx600Meta);
  checkError(Status, "amd_comgr_get_isa_metadata");
  checkMetadataString(Gfx600Meta, "LocalMemorySize", "32768");
  checkMetadataString(Gfx600Meta, "ImageSupport", "1");
  checkMetadataString(Gfx600Meta, "TotalNumVGPRs", "256");
  checkMetadataString(Gfx600Meta, "AddressableNumVGPRs", "256");
  Status = amd_comgr_destroy_metadata(Gfx600Meta);
  checkError(Status, "amd_comgr_destroy_metadata");

  // VGPR counts use wave32 where supported, including generic and strict ISAs.
  const struct {
    const char *IsaName;
    const char *Total;
    const char *Addressable;
  } VGPRCounts[] = {
      {"amdgcn-amd-amdhsa--gfx900", "256", "256"},
      {"amdgcn-amd-amdhsa--gfx90a", "512", "512"},
      {"amdgcn-amd-amdhsa--gfx1030", "1024", "256"},
      {"amdgcn-amd-amdhsa--gfx1100", "1536", "256"},
      {"amdgcn-amd-amdhsa--gfx1102", "1024", "256"},
      {"amdgcn-amd-amdhsa--gfx1250", "1024", "1024"},
      {"amdgcn-amd-amdhsa--gfx1250-strict", "1024", "1024"},
      {"amdgcn-amd-amdhsa--gfx9-4-generic", "512", "512"},
      {"amdgcn-amd-amdhsa--gfx11-generic", "1024", "256"},
      {"amdgcn-amd-amdhsa--gfx12-generic", "1536", "256"},
      {"amdgcn-amd-amdhsa--gfx12-5-generic", "1024", "1024"},
  };
  for (size_t I = 0; I < sizeof(VGPRCounts) / sizeof(VGPRCounts[0]); ++I) {
    amd_comgr_metadata_node_t Meta;
    Status = amd_comgr_get_isa_metadata(VGPRCounts[I].IsaName, &Meta);
    checkError(Status, "amd_comgr_get_isa_metadata");
    checkMetadataString(Meta, "TotalNumVGPRs", VGPRCounts[I].Total);
    checkMetadataString(Meta, "AddressableNumVGPRs", VGPRCounts[I].Addressable);
    Status = amd_comgr_destroy_metadata(Meta);
    checkError(Status, "amd_comgr_destroy_metadata");
  }

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
    checkMetadataString(Meta, "MaxFlatWorkGroupSize", "1024");
    int Indent = 1;
    Status = amd_comgr_iterate_map_metadata(Meta, printEntry, (void *)&Indent);
    checkError(Status, "amd_comgr_iterate_map_metadata");
    Status = amd_comgr_destroy_metadata(Meta);
    checkError(Status, "amd_comgr_destroy_metadata");
  }

  return 0;
}
