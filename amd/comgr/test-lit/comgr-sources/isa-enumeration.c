//===- isa-enumeration.c -------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

#define MAX_ISA_NAME_SIZE 1024

int main(int argc, char *argv[]) {
  size_t IsaCount;
  amd_comgr_(get_isa_count(&IsaCount));
  if (IsaCount == 0)
    fail("ISA Count: %zu", IsaCount);
  for (size_t i = 0; i < IsaCount; i++) {
    const char *Name;
    char IsaName[MAX_ISA_NAME_SIZE];
    amd_comgr_(get_isa_name(i, &Name));
    strncpy(IsaName, Name, MAX_ISA_NAME_SIZE);
    printf("%s\n", IsaName);
  }
  return 0;
}
