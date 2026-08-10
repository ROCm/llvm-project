//===- api_contract_test.c ------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Exercises the argument-validation contract of the public API: every entry
/// point below must reject a null handle, a null output pointer, an
/// out-of-range enum, or a handle of the wrong kind with
/// AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT rather than proceeding.
///
/// Some entry points dereference an argument before validating it, so passing
/// null to them crashes instead of returning a status. They are deliberately
/// absent here, and each is a robustness gap rather than a test gap:
///   - amd_comgr_get_version, amd_comgr_create_symbolizer_info and
///     amd_comgr_destroy_metadata validate nothing at all.
///   - amd_comgr_get_data_kind writes through its output pointer before
///     testing it.
///   - amd_comgr_action_info_get_logging validates only its action_info handle,
///     not its output pointer.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// checkStatus already prints the stringified call when it fails, so this needs
// no logging of its own.
#define EXPECT_INVALID(call)                                                   \
  checkStatus((call), AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, #call)

// A handle whose numeric value is zero: what a caller gets from a
// zero-initialized struct. Comgr converts handles by casting, so this reaches
// the null checks in the entry points.
static amd_comgr_data_t nullData(void) {
  amd_comgr_data_t D;
  D.handle = 0;
  return D;
}

static amd_comgr_data_set_t nullDataSet(void) {
  amd_comgr_data_set_t S;
  S.handle = 0;
  return S;
}

static amd_comgr_action_info_t nullActionInfo(void) {
  amd_comgr_action_info_t A;
  A.handle = 0;
  return A;
}

static amd_comgr_disassembly_info_t nullDisassemblyInfo(void) {
  amd_comgr_disassembly_info_t D;
  D.handle = 0;
  return D;
}

static amd_comgr_symbolizer_info_t nullSymbolizerInfo(void) {
  amd_comgr_symbolizer_info_t S;
  S.handle = 0;
  return S;
}

static void testStatusString(void) {
  printf("-- testStatusString\n");
  fflush(stdout);
  const char *Str;
  EXPECT_INVALID(amd_comgr_status_string(AMD_COMGR_STATUS_SUCCESS, NULL));
  EXPECT_INVALID(amd_comgr_status_string((amd_comgr_status_t)0x7fffffff, &Str));
}

static void testIsa(void) {
  printf("-- testIsa\n");
  fflush(stdout);
  size_t Count;
  const char *Name;
  amd_comgr_metadata_node_t Node;

  EXPECT_INVALID(amd_comgr_get_isa_count(NULL));

  checkError(amd_comgr_get_isa_count(&Count), "amd_comgr_get_isa_count");
  EXPECT_INVALID(amd_comgr_get_isa_name(0, NULL));
  EXPECT_INVALID(amd_comgr_get_isa_name(Count, &Name));

  EXPECT_INVALID(amd_comgr_get_isa_metadata(NULL, &Node));
  EXPECT_INVALID(amd_comgr_get_isa_metadata("amdgcn-amd-amdhsa--gfx900", NULL));
  EXPECT_INVALID(amd_comgr_get_isa_metadata("not-a-real-isa", &Node));
}

static void testData(void) {
  printf("-- testData\n");
  fflush(stdout);
  amd_comgr_data_t Data;
  amd_comgr_data_kind_t Kind;
  size_t Size = 0;
  char Bytes[4] = {0};

  EXPECT_INVALID(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, NULL));
  EXPECT_INVALID(amd_comgr_create_data(AMD_COMGR_DATA_KIND_UNDEF, &Data));
  EXPECT_INVALID(
      amd_comgr_create_data((amd_comgr_data_kind_t)0x1000, &Data));

  EXPECT_INVALID(amd_comgr_release_data(nullData()));
  // Valid output pointer, invalid handle: get_data_kind writes through the
  // pointer before validating, so it must not be passed null.
  EXPECT_INVALID(amd_comgr_get_data_kind(nullData(), &Kind));
  EXPECT_INVALID(amd_comgr_set_data(nullData(), 1, Bytes));
  EXPECT_INVALID(amd_comgr_set_data_name(nullData(), "x"));
  EXPECT_INVALID(amd_comgr_get_data_name(nullData(), &Size, NULL));
  EXPECT_INVALID(amd_comgr_get_data(nullData(), &Size, NULL));
  EXPECT_INVALID(amd_comgr_get_data_isa_name(nullData(), &Size, NULL));
  EXPECT_INVALID(amd_comgr_set_data_from_file_slice(nullData(), 0, 0, 1));

  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &Data),
             "amd_comgr_create_data");

  // Zero size and null bytes are both rejected.
  EXPECT_INVALID(amd_comgr_set_data(Data, 0, Bytes));
  EXPECT_INVALID(amd_comgr_set_data(Data, 1, NULL));
  // No data set yet, so get_data has nothing to return.
  EXPECT_INVALID(amd_comgr_get_data(Data, &Size, NULL));
  EXPECT_INVALID(amd_comgr_get_data_name(Data, NULL, NULL));
  // SOURCE is neither RELOCATABLE nor EXECUTABLE.
  EXPECT_INVALID(amd_comgr_get_data_isa_name(Data, &Size, NULL));
  EXPECT_INVALID(amd_comgr_get_data_metadata(Data, NULL));

  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
}

static void testDataSet(void) {
  printf("-- testDataSet\n");
  fflush(stdout);
  amd_comgr_data_set_t Set;
  amd_comgr_data_t Data;
  size_t Count;

  EXPECT_INVALID(amd_comgr_create_data_set(NULL));
  EXPECT_INVALID(amd_comgr_destroy_data_set(nullDataSet()));
  EXPECT_INVALID(amd_comgr_data_set_add(nullDataSet(), nullData()));
  EXPECT_INVALID(
      amd_comgr_data_set_remove(nullDataSet(), AMD_COMGR_DATA_KIND_SOURCE));
  EXPECT_INVALID(amd_comgr_action_data_count(
      nullDataSet(), AMD_COMGR_DATA_KIND_SOURCE, &Count));
  EXPECT_INVALID(amd_comgr_action_data_get_data(
      nullDataSet(), AMD_COMGR_DATA_KIND_SOURCE, 0, &Data));

  checkError(amd_comgr_create_data_set(&Set), "amd_comgr_create_data_set");
  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &Data),
             "amd_comgr_create_data");

  // An unnamed data object cannot be added to a set.
  EXPECT_INVALID(amd_comgr_data_set_add(Set, Data));

  EXPECT_INVALID(
      amd_comgr_data_set_remove(Set, (amd_comgr_data_kind_t)0x1000));
  EXPECT_INVALID(amd_comgr_action_data_count(
      Set, (amd_comgr_data_kind_t)0x1000, &Count));
  EXPECT_INVALID(
      amd_comgr_action_data_count(Set, AMD_COMGR_DATA_KIND_SOURCE, NULL));
  EXPECT_INVALID(amd_comgr_action_data_get_data(
      Set, AMD_COMGR_DATA_KIND_SOURCE, 0, NULL));

  // Index past the end of an empty set.
  checkError(amd_comgr_set_data_name(Data, "contract.cl"),
             "amd_comgr_set_data_name");
  checkError(amd_comgr_data_set_add(Set, Data), "amd_comgr_data_set_add");
  EXPECT_INVALID(amd_comgr_action_data_get_data(
      Set, AMD_COMGR_DATA_KIND_SOURCE, 1, &Data));

  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
  checkError(amd_comgr_destroy_data_set(Set), "amd_comgr_destroy_data_set");
}

static void testActionInfo(void) {
  printf("-- testActionInfo\n");
  fflush(stdout);
  amd_comgr_action_info_t Info;
  amd_comgr_language_t Language;
  size_t Size, Count;
  bool Logging;
  const char *EntryID = "hipv4-amdgcn-amd-amdhsa--gfx900";
  size_t BlockSize = 64;
  amd_comgr_target_id_t TargetID;

  TargetID.triple = "amdgcn-amd-amdhsa";
  TargetID.arch = "gfx900";

  EXPECT_INVALID(amd_comgr_create_action_info(NULL));
  EXPECT_INVALID(amd_comgr_destroy_action_info(nullActionInfo()));

  EXPECT_INVALID(amd_comgr_action_info_set_isa_name(nullActionInfo(), "x"));
  EXPECT_INVALID(
      amd_comgr_action_info_get_isa_name(nullActionInfo(), &Size, NULL));
  EXPECT_INVALID(amd_comgr_action_info_set_language(
      nullActionInfo(), AMD_COMGR_LANGUAGE_HIP));
  EXPECT_INVALID(
      amd_comgr_action_info_get_language(nullActionInfo(), &Language));
  EXPECT_INVALID(
      amd_comgr_action_info_set_option_list(nullActionInfo(), NULL, 0));
  EXPECT_INVALID(
      amd_comgr_action_info_get_option_list_count(nullActionInfo(), &Count));
  EXPECT_INVALID(amd_comgr_action_info_get_option_list_item(
      nullActionInfo(), 0, &Size, NULL));
  EXPECT_INVALID(
      amd_comgr_action_info_set_bundle_entry_ids(nullActionInfo(), NULL, 0));
  EXPECT_INVALID(amd_comgr_action_info_get_bundle_entry_id_count(
      nullActionInfo(), &Count));
  EXPECT_INVALID(amd_comgr_action_info_get_bundle_entry_id(
      nullActionInfo(), 0, &Size, NULL));
  EXPECT_INVALID(
      amd_comgr_action_info_set_package_entry_ids(nullActionInfo(), NULL, 0));
  EXPECT_INVALID(amd_comgr_action_info_get_package_entry_id_count(
      nullActionInfo(), &Count));
  EXPECT_INVALID(amd_comgr_action_info_get_package_entry_id(
      nullActionInfo(), 0, &Size, NULL, &Size, NULL));
  EXPECT_INVALID(
      amd_comgr_action_info_set_block_sizes(nullActionInfo(), NULL, 0));
  EXPECT_INVALID(
      amd_comgr_action_info_get_block_sizes_count(nullActionInfo(), &Count));
  EXPECT_INVALID(
      amd_comgr_action_info_get_block_sizes(nullActionInfo(), 1, &BlockSize));
  EXPECT_INVALID(
      amd_comgr_action_info_set_working_directory_path(nullActionInfo(), "x"));
  EXPECT_INVALID(amd_comgr_action_info_get_working_directory_path(
      nullActionInfo(), &Size, NULL));
  EXPECT_INVALID(amd_comgr_action_info_set_logging(nullActionInfo(), true));
  EXPECT_INVALID(amd_comgr_action_info_get_logging(nullActionInfo(), &Logging));
  EXPECT_INVALID(amd_comgr_action_info_set_vfs(nullActionInfo(), true));
  EXPECT_INVALID(
      amd_comgr_action_info_set_device_lib_linking(nullActionInfo(), true));

  checkError(amd_comgr_create_action_info(&Info),
             "amd_comgr_create_action_info");

  EXPECT_INVALID(amd_comgr_action_info_set_language(
      Info, (amd_comgr_language_t)0x1000));
  EXPECT_INVALID(amd_comgr_action_info_get_language(Info, NULL));
  EXPECT_INVALID(amd_comgr_action_info_get_isa_name(Info, NULL, NULL));
  EXPECT_INVALID(amd_comgr_action_info_get_option_list_count(Info, NULL));
  EXPECT_INVALID(amd_comgr_action_info_get_option_list_item(Info, 0, NULL, NULL));
  EXPECT_INVALID(amd_comgr_action_info_get_block_sizes_count(Info, NULL));
  EXPECT_INVALID(amd_comgr_action_info_get_block_sizes(Info, 1, NULL));
  EXPECT_INVALID(
      amd_comgr_action_info_get_working_directory_path(Info, NULL, NULL));
  // A non-zero count with no array is rejected for each list setter.
  EXPECT_INVALID(amd_comgr_action_info_set_option_list(Info, NULL, 1));
  EXPECT_INVALID(amd_comgr_action_info_set_bundle_entry_ids(Info, NULL, 1));
  EXPECT_INVALID(amd_comgr_action_info_set_package_entry_ids(Info, NULL, 1));
  EXPECT_INVALID(amd_comgr_action_info_set_block_sizes(Info, NULL, 1));

  // Index past the end of each list.
  checkError(amd_comgr_action_info_set_option_list(Info, &EntryID, 1),
             "amd_comgr_action_info_set_option_list");
  EXPECT_INVALID(
      amd_comgr_action_info_get_option_list_item(Info, 1, &Size, NULL));
  checkError(amd_comgr_action_info_set_bundle_entry_ids(Info, &EntryID, 1),
             "amd_comgr_action_info_set_bundle_entry_ids");
  EXPECT_INVALID(
      amd_comgr_action_info_get_bundle_entry_id(Info, 1, &Size, NULL));
  checkError(amd_comgr_action_info_set_package_entry_ids(Info, &TargetID, 1),
             "amd_comgr_action_info_set_package_entry_ids");
  EXPECT_INVALID(amd_comgr_action_info_get_package_entry_id(
      Info, 1, &Size, NULL, &Size, NULL));
  checkError(amd_comgr_action_info_set_block_sizes(Info, &BlockSize, 1),
             "amd_comgr_action_info_set_block_sizes");

  checkError(amd_comgr_destroy_action_info(Info),
             "amd_comgr_destroy_action_info");
}

static void testDoAction(void) {
  printf("-- testDoAction\n");
  fflush(stdout);
  amd_comgr_action_info_t Info;
  amd_comgr_data_set_t In, Out;

  checkError(amd_comgr_create_action_info(&Info),
             "amd_comgr_create_action_info");
  checkError(amd_comgr_create_data_set(&In), "amd_comgr_create_data_set");
  checkError(amd_comgr_create_data_set(&Out), "amd_comgr_create_data_set");

  EXPECT_INVALID(amd_comgr_do_action((amd_comgr_action_kind_t)0x1000, Info, In,
                                     Out));
  EXPECT_INVALID(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                                    Info, nullDataSet(), Out));
  EXPECT_INVALID(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                                    Info, In, nullDataSet()));

  checkError(amd_comgr_destroy_data_set(Out), "amd_comgr_destroy_data_set");
  checkError(amd_comgr_destroy_data_set(In), "amd_comgr_destroy_data_set");
  checkError(amd_comgr_destroy_action_info(Info),
             "amd_comgr_destroy_action_info");
}

// The metadata entry points dereference their node handle before validating it,
// so they are driven with a real node of the wrong kind instead of a null one.
static void testMetadataKindMismatch(void) {
  printf("-- testMetadataKindMismatch\n");
  fflush(stdout);
  amd_comgr_data_t Data;
  amd_comgr_metadata_node_t Root, Node;
  amd_comgr_metadata_kind_t Kind;
  size_t Size;
  char *Buf;
  long FileSize;

  FileSize = setBuf(TEST_OBJ_DIR "/shared-v3.so", &Buf);

  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data),
             "amd_comgr_create_data");
  checkError(amd_comgr_set_data(Data, FileSize, Buf), "amd_comgr_set_data");
  checkError(amd_comgr_get_data_metadata(Data, &Root),
             "amd_comgr_get_data_metadata");

  checkError(amd_comgr_get_metadata_kind(Root, &Kind),
             "amd_comgr_get_metadata_kind");
  if (Kind != AMD_COMGR_METADATA_KIND_MAP)
    fail("expected the metadata root of shared-v3.so to be a map");

  EXPECT_INVALID(amd_comgr_get_metadata_kind(Root, NULL));

  // Root is a map, so the string and list accessors must reject it.
  EXPECT_INVALID(amd_comgr_get_metadata_string(Root, &Size, NULL));
  EXPECT_INVALID(amd_comgr_get_metadata_list_size(Root, &Size));
  EXPECT_INVALID(amd_comgr_index_list_metadata(Root, 0, &Node));

  // Map accessors reject a null size, callback or key.
  EXPECT_INVALID(amd_comgr_get_metadata_map_size(Root, NULL));
  EXPECT_INVALID(amd_comgr_iterate_map_metadata(Root, NULL, NULL));
  EXPECT_INVALID(amd_comgr_metadata_lookup(Root, NULL, &Node));
  EXPECT_INVALID(amd_comgr_metadata_lookup(Root, "amdhsa.version", NULL));
  // A key that is not in the map is an error, not a null node.
  checkStatus(amd_comgr_metadata_lookup(Root, "no.such.key", &Node),
              AMD_COMGR_STATUS_ERROR, "amd_comgr_metadata_lookup missing key");

  // amdhsa.version is a list, so the map and string accessors must reject it.
  checkError(amd_comgr_metadata_lookup(Root, "amdhsa.version", &Node),
             "amd_comgr_metadata_lookup");
  EXPECT_INVALID(amd_comgr_get_metadata_map_size(Node, &Size));
  EXPECT_INVALID(amd_comgr_get_metadata_string(Node, &Size, NULL));
  EXPECT_INVALID(amd_comgr_get_metadata_list_size(Node, NULL));
  checkError(amd_comgr_destroy_metadata(Node), "amd_comgr_destroy_metadata");

  checkError(amd_comgr_destroy_metadata(Root), "amd_comgr_destroy_metadata");
  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
  free(Buf);
}

static void testSymbolsAndDisassembly(void) {
  printf("-- testSymbolsAndDisassembly\n");
  fflush(stdout);
  amd_comgr_data_t Data;
  amd_comgr_symbol_t Symbol;
  amd_comgr_disassembly_info_t DisasmInfo;
  size_t Size = 4;

  EXPECT_INVALID(amd_comgr_symbol_lookup(nullData(), "foo", &Symbol));
  EXPECT_INVALID(amd_comgr_iterate_symbols(nullData(), NULL, NULL));

  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &Data),
             "amd_comgr_create_data");
  // SOURCE is neither RELOCATABLE nor EXECUTABLE.
  EXPECT_INVALID(amd_comgr_symbol_lookup(Data, "foo", &Symbol));
  EXPECT_INVALID(amd_comgr_iterate_symbols(Data, NULL, NULL));
  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");

  EXPECT_INVALID(amd_comgr_create_disassembly_info(NULL, NULL, NULL, NULL,
                                                   &DisasmInfo));
  EXPECT_INVALID(amd_comgr_create_disassembly_info("not-a-real-isa", NULL, NULL,
                                                   NULL, &DisasmInfo));
  EXPECT_INVALID(amd_comgr_destroy_disassembly_info(nullDisassemblyInfo()));
  EXPECT_INVALID(amd_comgr_disassemble_instruction(nullDisassemblyInfo(), 0,
                                                   NULL, &Size));
}

static void testSymbolizer(void) {
  printf("-- testSymbolizer\n");
  fflush(stdout);
  EXPECT_INVALID(amd_comgr_destroy_symbolizer_info(nullSymbolizerInfo()));
  EXPECT_INVALID(amd_comgr_symbolize(nullSymbolizerInfo(), 0, true, NULL));
}

int main(int argc, char *argv[]) {
  testStatusString();
  testIsa();
  testData();
  testDataSet();
  testActionInfo();
  testDoAction();
  testMetadataKindMismatch();
  testSymbolsAndDisassembly();
  testSymbolizer();
  printf("api_contract_test passed\n");
  return 0;
}
