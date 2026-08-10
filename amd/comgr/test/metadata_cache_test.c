//===- metadata_cache_test.c ----------------------------------------------===//
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

// Reads "amdhsa.version" as "<major>.<minor>". Out must hold 32 bytes.
static void readVersion(amd_comgr_metadata_node_t Root, char *Out) {
  amd_comgr_status_t Status;
  amd_comgr_metadata_node_t Version, Element;
  size_t Size, Count;
  char Buf[16];

  Status = amd_comgr_metadata_lookup(Root, "amdhsa.version", &Version);
  checkError(Status, "amd_comgr_metadata_lookup");

  Status = amd_comgr_get_metadata_list_size(Version, &Count);
  checkError(Status, "amd_comgr_get_metadata_list_size");
  if (Count != 2)
    fail("expected amdhsa.version to have 2 elements, saw %zu", Count);

  Out[0] = '\0';
  for (size_t I = 0; I < Count; ++I) {
    Status = amd_comgr_index_list_metadata(Version, I, &Element);
    checkError(Status, "amd_comgr_index_list_metadata");

    Size = sizeof(Buf);
    Status = amd_comgr_get_metadata_string(Element, &Size, Buf);
    checkError(Status, "amd_comgr_get_metadata_string");

    if (I)
      strcat(Out, ".");
    strcat(Out, Buf);

    Status = amd_comgr_destroy_metadata(Element);
    checkError(Status, "amd_comgr_destroy_metadata");
  }

  Status = amd_comgr_destroy_metadata(Version);
  checkError(Status, "amd_comgr_destroy_metadata");
}

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;
  amd_comgr_data_t DataObject;
  amd_comgr_metadata_node_t First, Second, Third;
  char VersionFirst[32], VersionSecond[32], VersionThird[32];
  char *Buf;
  long Size;

  Size = setBuf(TEST_OBJ_DIR "/shared-v3.so", &Buf);

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataObject);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataObject, Size, Buf);
  checkError(Status, "amd_comgr_set_data");

  Status = amd_comgr_get_data_metadata(DataObject, &First);
  checkError(Status, "amd_comgr_get_data_metadata");
  readVersion(First, VersionFirst);

  Status = amd_comgr_get_data_metadata(DataObject, &Second);
  checkError(Status, "amd_comgr_get_data_metadata");
  readVersion(Second, VersionSecond);

  if (strcmp(VersionFirst, VersionSecond))
    fail("cached metadata disagrees: first \"%s\", second \"%s\"", VersionFirst,
         VersionSecond);

  // Destroying the first handle must not invalidate the shared document.
  Status = amd_comgr_destroy_metadata(First);
  checkError(Status, "amd_comgr_destroy_metadata");
  readVersion(Second, VersionSecond);
  if (strcmp(VersionFirst, VersionSecond))
    fail("metadata invalidated by destroying a sibling handle");

  Status = amd_comgr_destroy_metadata(Second);
  checkError(Status, "amd_comgr_destroy_metadata");

  // Re-setting the data must invalidate the cache and re-parse.
  Status = amd_comgr_set_data(DataObject, Size, Buf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_get_data_metadata(DataObject, &Third);
  checkError(Status, "amd_comgr_get_data_metadata");
  readVersion(Third, VersionThird);
  if (strcmp(VersionFirst, VersionThird))
    fail("metadata after re-set disagrees: \"%s\" vs \"%s\"", VersionFirst,
         VersionThird);
  Status = amd_comgr_destroy_metadata(Third);
  checkError(Status, "amd_comgr_destroy_metadata");

  Status = amd_comgr_release_data(DataObject);
  checkError(Status, "amd_comgr_release_data");
  free(Buf);
  return 0;
}
