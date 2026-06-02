//===- isa-enumeration.c -------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  size_t IsaCount;
  amd_comgr_(get_isa_count(&IsaCount));
  if (IsaCount == 0)
    fail("ISA Count: %zu", IsaCount);
  for (size_t i = 0; i < IsaCount; i++) {
    const char *Name;
    bool sramecc = false, xnack = false;
    amd_comgr_metadata_node_t Root, Features, Val;
    amd_comgr_(get_isa_name(i, &Name));
    amd_comgr_(get_isa_metadata(Name, &Root));
    amd_comgr_(metadata_lookup(Root, "Features", &Features));
    if (amd_comgr_metadata_lookup(Features, "sramecc", &Val) ==
      AMD_COMGR_STATUS_SUCCESS) {
      sramecc = true;
    }
    if (amd_comgr_metadata_lookup(Features, "xnack", &Val) ==
      AMD_COMGR_STATUS_SUCCESS) {
      xnack = true;
    }


    printf("%s\n", Name);

    if (sramecc) {
      printf("%s:sramecc+\n", Name);
      printf("%s:sramecc-\n", Name);
    }
    if (xnack) {
      printf("%s:xnack+\n", Name);
      printf("%s:xnack-\n", Name);
    }
    if (sramecc && xnack) {
      printf("%s:sramecc+:xnack+\n", Name);
      printf("%s:sramecc+:xnack-\n", Name);
      printf("%s:sramecc-:xnack+\n", Name);
      printf("%s:sramecc-:xnack-\n", Name);
      
    }
  }
  return 0;
}
