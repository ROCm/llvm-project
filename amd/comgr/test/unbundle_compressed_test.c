//===- unbundle_compressed_test.c -----------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Comgr unbundles a compressed bundle in memory instead of writing every
// unbundled code object to a temporary file and reading it back. The two
// bundles used here hold the same payload, one compressed and one not, so that
// the in-memory path and the file-based path it replaces can be checked
// against each other and against the file that was bundled.
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *BundleEntryIDs[] = {"host-x86_64-unknown-linux-gnu",
                                       "hip-amdgcn-amd-amdhsa-unknown-gfx900"};

/// Check that Data holds exactly the contents of PayloadPath.
static void checkContents(amd_comgr_data_t Data, const char *PayloadPath) {
  amd_comgr_status_t Status;
  char *PayloadBuf, *Bytes;
  size_t PayloadSize, Size;

  PayloadSize = setBuf(PayloadPath, &PayloadBuf);

  Status = amd_comgr_get_data(Data, &Size, NULL);
  checkError(Status, "checkContents:amd_comgr_get_data");
  if (Size != PayloadSize)
    fail("%s: unbundled %zu bytes (expected %zu)", PayloadPath, Size,
         PayloadSize);

  Bytes = (char *)malloc(Size);
  if (!Bytes)
    fail("malloc");
  Status = amd_comgr_get_data(Data, &Size, Bytes);
  checkError(Status, "checkContents:amd_comgr_get_data");

  if (memcmp(Bytes, PayloadBuf, Size))
    fail("%s: unbundled contents differ from the bundled file", PayloadPath);

  free(Bytes);
  free(PayloadBuf);
}

/// Check that Data is named ExpectedName.
static void checkName(amd_comgr_data_t Data, const char *ExpectedName) {
  amd_comgr_status_t Status;
  char Name[128];
  size_t NameSize;

  Status = amd_comgr_get_data_name(Data, &NameSize, NULL);
  checkError(Status, "checkName:amd_comgr_get_data_name");
  if (NameSize > sizeof(Name))
    fail("unbundled name is %zu bytes (expected at most %zu)", NameSize,
         sizeof(Name));

  Status = amd_comgr_get_data_name(Data, &NameSize, Name);
  checkError(Status, "checkName:amd_comgr_get_data_name");
  if (strcmp(Name, ExpectedName))
    fail("unbundled name is \"%s\" (expected \"%s\")", Name, ExpectedName);
}

/// Unbundle BundlePath and check the outputs against PayloadPath, which is the
/// file that was bundled.
static void unbundleAndCheck(const char *BundlePath, const char *PayloadPath) {
  amd_comgr_status_t Status;
  amd_comgr_data_t DataBundle, DataElement;
  amd_comgr_data_set_t DataSetBundled, DataSetUnbundled;
  amd_comgr_action_info_t ActionInfo;
  char *BundleBuf;
  size_t BundleSize, Size;

  BundleSize = setBuf(BundlePath, &BundleBuf);

  Status = amd_comgr_create_data_set(&DataSetBundled);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC_BUNDLE, &DataBundle);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataBundle, BundleSize, BundleBuf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataBundle, "square");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetBundled, DataBundle);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(ActionInfo, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "set_language");
  Status = amd_comgr_action_info_set_bundle_entry_ids(ActionInfo,
                                                      BundleEntryIDs, 2);
  checkError(Status, "set_bundle_entry_ids");

  Status = amd_comgr_create_data_set(&DataSetUnbundled);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_UNBUNDLE, ActionInfo,
                               DataSetBundled, DataSetUnbundled);
  checkError(Status, "amd_comgr_do_action");

  checkCount(BundlePath, DataSetUnbundled, AMD_COMGR_DATA_KIND_BC, 2);

  // The bundle holds no host bundle, so the host output is empty.
  Status = amd_comgr_action_data_get_data(DataSetUnbundled,
                                          AMD_COMGR_DATA_KIND_BC, 0,
                                          &DataElement);
  checkError(Status, "host:amd_comgr_action_data_get_data");
  Status = amd_comgr_get_data(DataElement, &Size, NULL);
  checkError(Status, "host:amd_comgr_get_data");
  if (Size)
    fail("%s: host output is %zu bytes (expected 0)", BundlePath, Size);
  checkName(DataElement, "square-host-x86_64-unknown-linux-gnu.bc");
  Status = amd_comgr_release_data(DataElement);
  checkError(Status, "host:amd_comgr_release_data");

  // The gfx900 output holds the file that was bundled.
  Status = amd_comgr_action_data_get_data(DataSetUnbundled,
                                          AMD_COMGR_DATA_KIND_BC, 1,
                                          &DataElement);
  checkError(Status, "device:amd_comgr_action_data_get_data");
  checkName(DataElement, "square-hip-amdgcn-amd-amdhsa-unknown-gfx900.bc");
  checkContents(DataElement, PayloadPath);
  Status = amd_comgr_release_data(DataElement);
  checkError(Status, "device:amd_comgr_release_data");

  Status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_destroy_data_set(DataSetBundled);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetUnbundled);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_release_data(DataBundle);
  checkError(Status, "amd_comgr_release_data");

  free(BundleBuf);
}

int main(int argc, char *argv[]) {
  // The in-memory path unbundles a compressed bundle.
  unbundleAndCheck(TEST_OBJ_DIR "/square_cc.bc",
                   TEST_OBJ_DIR "/square_cc_payload.bc");

  // The file-based path still unbundles an uncompressed bundle that holds the
  // same payload.
  unbundleAndCheck(TEST_OBJ_DIR "/square_unc.bc",
                   TEST_OBJ_DIR "/square_cc_payload.bc");

  return 0;
}
