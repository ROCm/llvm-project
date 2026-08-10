//===- action_info_roundtrip_test.c ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Round-trips every settable field of an action info object: set a value, read
/// it back with the matching getter, and check it survived. The two-call
/// size-then-buffer protocol the string and list getters use is exercised both
/// ways, since querying the length is a separate path from copying the value.
///
/// Every getter is therefore called only after its field has been set.
/// amd_comgr_action_info_get_isa_name calls strlen on the stored pointer
/// without a null check, so querying it on a freshly created action info, or
/// after clearing the name, crashes instead of reporting an empty string. That
/// is a robustness gap to fix separately, not something to assert here.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read a string-valued getter using the size-then-buffer protocol and compare.
// Getter is called twice on purpose: once with a NULL buffer to learn the
// length, once to copy.
#define CHECK_STR_FIELD(getter, info, expected)                                \
  do {                                                                         \
    size_t Size = 0;                                                           \
    char *Buf;                                                                 \
    checkError(getter(info, &Size, NULL), #getter " size");                    \
    if (Size != strlen(expected) + 1)                                          \
      fail(#getter " reported %zu bytes, expected %zu", Size,                  \
           strlen(expected) + 1);                                              \
    Buf = (char *)calloc(Size, 1);                                             \
    if (!Buf)                                                                  \
      fail("calloc");                                                          \
    checkError(getter(info, &Size, Buf), #getter " value");                    \
    if (strcmp(Buf, expected))                                                 \
      fail(#getter " returned \"%s\", expected \"%s\"", Buf, expected);         \
    free(Buf);                                                                 \
  } while (false)

static void testIsaName(amd_comgr_action_info_t Info) {
  const char *IsaName = "amdgcn-amd-amdhsa--gfx906";

  checkError(amd_comgr_action_info_set_isa_name(Info, IsaName),
             "amd_comgr_action_info_set_isa_name");
  CHECK_STR_FIELD(amd_comgr_action_info_get_isa_name, Info, IsaName);

  // Setting an empty name is accepted and is the documented way to clear the
  // field. The value is not read back here: get_isa_name calls strlen on the
  // stored pointer, which set_isa_name has just set to null, so querying a
  // cleared (or freshly created) action info crashes. See the header comment.
  checkError(amd_comgr_action_info_set_isa_name(Info, ""),
             "amd_comgr_action_info_set_isa_name(empty)");

  checkError(amd_comgr_action_info_set_isa_name(Info, IsaName),
             "amd_comgr_action_info_set_isa_name");
  CHECK_STR_FIELD(amd_comgr_action_info_get_isa_name, Info, IsaName);
}

static void testLanguage(amd_comgr_action_info_t Info) {
  amd_comgr_language_t Language;
  const amd_comgr_language_t All[] = {
      AMD_COMGR_LANGUAGE_NONE,        AMD_COMGR_LANGUAGE_OPENCL_1_2,
      AMD_COMGR_LANGUAGE_OPENCL_2_0,  AMD_COMGR_LANGUAGE_HIP,
      AMD_COMGR_LANGUAGE_LLVM_IR,
  };

  for (size_t I = 0; I < sizeof(All) / sizeof(All[0]); ++I) {
    checkError(amd_comgr_action_info_set_language(Info, All[I]),
               "amd_comgr_action_info_set_language");
    checkError(amd_comgr_action_info_get_language(Info, &Language),
               "amd_comgr_action_info_get_language");
    if (Language != All[I])
      fail("language round-trip gave %d, expected %d", Language, All[I]);
  }
}

static void testLogging(amd_comgr_action_info_t Info) {
  bool Logging;

  checkError(amd_comgr_action_info_set_logging(Info, true),
             "amd_comgr_action_info_set_logging(true)");
  checkError(amd_comgr_action_info_get_logging(Info, &Logging),
             "amd_comgr_action_info_get_logging");
  if (!Logging)
    fail("logging round-trip lost the true value");

  checkError(amd_comgr_action_info_set_logging(Info, false),
             "amd_comgr_action_info_set_logging(false)");
  checkError(amd_comgr_action_info_get_logging(Info, &Logging),
             "amd_comgr_action_info_get_logging");
  if (Logging)
    fail("logging round-trip lost the false value");
}

static void testOptionList(amd_comgr_action_info_t Info) {
  const char *Options[] = {"-O3", "-mllvm", "-amdgpu-early-inline-all"};
  const size_t N = sizeof(Options) / sizeof(Options[0]);
  size_t Count = 0;

  checkError(amd_comgr_action_info_set_option_list(Info, Options, N),
             "amd_comgr_action_info_set_option_list");
  checkError(amd_comgr_action_info_get_option_list_count(Info, &Count),
             "amd_comgr_action_info_get_option_list_count");
  if (Count != N)
    fail("option list count is %zu, expected %zu", Count, N);

  for (size_t I = 0; I < N; ++I) {
    size_t Size = 0;
    char *Buf;
    checkError(amd_comgr_action_info_get_option_list_item(Info, I, &Size, NULL),
               "amd_comgr_action_info_get_option_list_item size");
    Buf = (char *)calloc(Size, 1);
    if (!Buf)
      fail("calloc");
    checkError(amd_comgr_action_info_get_option_list_item(Info, I, &Size, Buf),
               "amd_comgr_action_info_get_option_list_item value");
    if (strcmp(Buf, Options[I]))
      fail("option %zu is \"%s\", expected \"%s\"", I, Buf, Options[I]);
    free(Buf);
  }

  // An empty list is a valid state and must report zero.
  checkError(amd_comgr_action_info_set_option_list(Info, NULL, 0),
             "amd_comgr_action_info_set_option_list(empty)");
  checkError(amd_comgr_action_info_get_option_list_count(Info, &Count),
             "amd_comgr_action_info_get_option_list_count");
  if (Count != 0)
    fail("emptied option list reports %zu entries", Count);
}

static void testBundleEntryIDs(amd_comgr_action_info_t Info) {
  const char *EntryIDs[] = {"hipv4-amdgcn-amd-amdhsa--gfx900",
                            "hipv4-amdgcn-amd-amdhsa--gfx906"};
  const size_t N = sizeof(EntryIDs) / sizeof(EntryIDs[0]);
  size_t Count = 0;

  checkError(amd_comgr_action_info_set_bundle_entry_ids(Info, EntryIDs, N),
             "amd_comgr_action_info_set_bundle_entry_ids");
  checkError(amd_comgr_action_info_get_bundle_entry_id_count(Info, &Count),
             "amd_comgr_action_info_get_bundle_entry_id_count");
  if (Count != N)
    fail("bundle entry id count is %zu, expected %zu", Count, N);

  for (size_t I = 0; I < N; ++I) {
    size_t Size = 0;
    char *Buf;
    checkError(amd_comgr_action_info_get_bundle_entry_id(Info, I, &Size, NULL),
               "amd_comgr_action_info_get_bundle_entry_id size");
    Buf = (char *)calloc(Size, 1);
    if (!Buf)
      fail("calloc");
    checkError(amd_comgr_action_info_get_bundle_entry_id(Info, I, &Size, Buf),
               "amd_comgr_action_info_get_bundle_entry_id value");
    if (strcmp(Buf, EntryIDs[I]))
      fail("bundle entry %zu is \"%s\", expected \"%s\"", I, Buf, EntryIDs[I]);
    free(Buf);
  }
}

static void testPackageEntryIDs(amd_comgr_action_info_t Info) {
  amd_comgr_target_id_t TargetIDs[2];
  size_t Count = 0;

  TargetIDs[0].triple = "amdgcn-amd-amdhsa";
  TargetIDs[0].arch = "gfx900";
  TargetIDs[1].triple = "amdgcn-amd-amdhsa";
  TargetIDs[1].arch = "gfx906";

  checkError(amd_comgr_action_info_set_package_entry_ids(Info, TargetIDs, 2),
             "amd_comgr_action_info_set_package_entry_ids");
  checkError(amd_comgr_action_info_get_package_entry_id_count(Info, &Count),
             "amd_comgr_action_info_get_package_entry_id_count");
  if (Count != 2)
    fail("package entry id count is %zu, expected 2", Count);

  for (size_t I = 0; I < 2; ++I) {
    size_t TripleSize = 0, ArchSize = 0;
    char *Triple, *Arch;

    checkError(amd_comgr_action_info_get_package_entry_id(
                   Info, I, &TripleSize, NULL, &ArchSize, NULL),
               "amd_comgr_action_info_get_package_entry_id sizes");
    Triple = (char *)calloc(TripleSize, 1);
    Arch = (char *)calloc(ArchSize, 1);
    if (!Triple || !Arch)
      fail("calloc");
    checkError(amd_comgr_action_info_get_package_entry_id(
                   Info, I, &TripleSize, Triple, &ArchSize, Arch),
               "amd_comgr_action_info_get_package_entry_id values");
    if (strcmp(Triple, TargetIDs[I].triple))
      fail("package %zu triple is \"%s\", expected \"%s\"", I, Triple,
           TargetIDs[I].triple);
    if (strcmp(Arch, TargetIDs[I].arch))
      fail("package %zu arch is \"%s\", expected \"%s\"", I, Arch,
           TargetIDs[I].arch);
    free(Arch);
    free(Triple);
  }
}

static void testBlockSizes(amd_comgr_action_info_t Info) {
  const size_t Sizes[] = {64, 128, 256};
  const size_t N = sizeof(Sizes) / sizeof(Sizes[0]);
  size_t Count = 0;
  size_t ReadBack[3];

  checkError(amd_comgr_action_info_set_block_sizes(Info, Sizes, N),
             "amd_comgr_action_info_set_block_sizes");
  checkError(amd_comgr_action_info_get_block_sizes_count(Info, &Count),
             "amd_comgr_action_info_get_block_sizes_count");
  if (Count != N)
    fail("block size count is %zu, expected %zu", Count, N);

  checkError(amd_comgr_action_info_get_block_sizes(Info, Count, ReadBack),
             "amd_comgr_action_info_get_block_sizes");
  for (size_t I = 0; I < N; ++I) {
    if (ReadBack[I] != Sizes[I])
      fail("block size %zu is %zu, expected %zu", I, ReadBack[I], Sizes[I]);
  }

  // A buffer smaller than the stored list must be rejected rather than
  // partially filled.
  checkStatus(amd_comgr_action_info_get_block_sizes(Info, N - 1, ReadBack),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "amd_comgr_action_info_get_block_sizes with a short buffer");

  checkError(amd_comgr_action_info_set_block_sizes(Info, NULL, 0),
             "amd_comgr_action_info_set_block_sizes(empty)");
  checkError(amd_comgr_action_info_get_block_sizes_count(Info, &Count),
             "amd_comgr_action_info_get_block_sizes_count");
  if (Count != 0)
    fail("emptied block size list reports %zu entries", Count);
}

static void testWorkingDirectory(amd_comgr_action_info_t Info) {
  const char *Path = "/tmp/comgr-working-dir";

  checkError(amd_comgr_action_info_set_working_directory_path(Info, Path),
             "amd_comgr_action_info_set_working_directory_path");
  CHECK_STR_FIELD(amd_comgr_action_info_get_working_directory_path, Info, Path);
}

static void testFlags(amd_comgr_action_info_t Info) {
  // These two have no getter in the public API, so the round-trip that can be
  // checked is that both values are accepted.
  checkError(amd_comgr_action_info_set_vfs(Info, true),
             "amd_comgr_action_info_set_vfs(true)");
  checkError(amd_comgr_action_info_set_vfs(Info, false),
             "amd_comgr_action_info_set_vfs(false)");
  checkError(amd_comgr_action_info_set_device_lib_linking(Info, true),
             "amd_comgr_action_info_set_device_lib_linking(true)");
  checkError(amd_comgr_action_info_set_device_lib_linking(Info, false),
             "amd_comgr_action_info_set_device_lib_linking(false)");
}

int main(int argc, char *argv[]) {
  amd_comgr_action_info_t Info;

  checkError(amd_comgr_create_action_info(&Info),
             "amd_comgr_create_action_info");

  testIsaName(Info);
  testLanguage(Info);
  testLogging(Info);
  testOptionList(Info);
  testBundleEntryIDs(Info);
  testPackageEntryIDs(Info);
  testBlockSizes(Info);
  testWorkingDirectory(Info);
  testFlags(Info);

  checkError(amd_comgr_destroy_action_info(Info),
             "amd_comgr_destroy_action_info");

  printf("action_info_roundtrip_test passed\n");
  return 0;
}
