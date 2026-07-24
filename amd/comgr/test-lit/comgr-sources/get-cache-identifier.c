//===- get-cache-identifier.c ---------------------------------------------===//
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

int main(void) {
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

  return 0;
}
