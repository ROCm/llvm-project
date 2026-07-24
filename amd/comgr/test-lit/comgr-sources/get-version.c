//===- get-version.c ------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

#include <ctype.h>
#include <string.h>

static void testVersion(void) {
  size_t Major;
  size_t Minor;

  amd_comgr_get_version(&Major, &Minor);

  if (Major != AMD_COMGR_INTERFACE_VERSION_MAJOR ||
      Minor != AMD_COMGR_INTERFACE_VERSION_MINOR)
    fail("incorrect version: expected %d.%d, saw %zu, %zu",
         AMD_COMGR_INTERFACE_VERSION_MAJOR, AMD_COMGR_INTERFACE_VERSION_MINOR,
         Major, Minor);
}

static void testCacheIdentifier(void) {
  if (amd_comgr_get_cache_identifier(NULL) !=
      AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    fail("NULL identifier was accepted");

  const char *Identifier = NULL;
  amd_comgr_(get_cache_identifier(&Identifier));
  if (!Identifier)
    fail("identifier is NULL");
  if (strlen(Identifier) != 64)
    fail("identifier has unexpected length: %zu", strlen(Identifier));

  for (size_t I = 0; Identifier[I]; ++I) {
    if (!isxdigit((unsigned char)Identifier[I]))
      fail("identifier contains a non-hex character");
  }

  const char *SecondIdentifier = NULL;
  amd_comgr_(get_cache_identifier(&SecondIdentifier));
  if (strcmp(Identifier, SecondIdentifier) != 0)
    fail("identifier changed between calls");
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    testVersion();
    return 0;
  }

  if (argc == 2 && strcmp(argv[1], "--cache-identifier") == 0) {
    testCacheIdentifier();
    return 0;
  }

  fail("unknown argument: %s", argv[1]);
  return 1;
}
