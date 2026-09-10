//===- isa-enumeration.c -------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

// Keep the supported targets visible in Comgr. This list deliberately does not
// come from TargetParser: adding or removing a target must update this test.
// clang-format off
static const char *const ExpectedIsaNames[] = {
    "amdgcn-amd-amdhsa--gfx600",
    "amdgcn-amd-amdhsa--gfx601",
    "amdgcn-amd-amdhsa--gfx602",
    "amdgcn-amd-amdhsa--gfx700",
    "amdgcn-amd-amdhsa--gfx701",
    "amdgcn-amd-amdhsa--gfx702",
    "amdgcn-amd-amdhsa--gfx703",
    "amdgcn-amd-amdhsa--gfx704",
    "amdgcn-amd-amdhsa--gfx705",
    "amdgcn-amd-amdhsa--gfx801",
    "amdgcn-amd-amdhsa--gfx802",
    "amdgcn-amd-amdhsa--gfx803",
    "amdgcn-amd-amdhsa--gfx805",
    "amdgcn-amd-amdhsa--gfx810",
    "amdgcn-amd-amdhsa--gfx900",
    "amdgcn-amd-amdhsa--gfx902",
    "amdgcn-amd-amdhsa--gfx904",
    "amdgcn-amd-amdhsa--gfx906",
    "amdgcn-amd-amdhsa--gfx908",
    "amdgcn-amd-amdhsa--gfx909",
    "amdgcn-amd-amdhsa--gfx90a",
    "amdgcn-amd-amdhsa--gfx90c",
    "amdgcn-amd-amdhsa--gfx942",
    "amdgcn-amd-amdhsa--gfx950",
    "amdgcn-amd-amdhsa--gfx1010",
    "amdgcn-amd-amdhsa--gfx1011",
    "amdgcn-amd-amdhsa--gfx1012",
    "amdgcn-amd-amdhsa--gfx1013",
    "amdgcn-amd-amdhsa--gfx1030",
    "amdgcn-amd-amdhsa--gfx1031",
    "amdgcn-amd-amdhsa--gfx1032",
    "amdgcn-amd-amdhsa--gfx1033",
    "amdgcn-amd-amdhsa--gfx1034",
    "amdgcn-amd-amdhsa--gfx1035",
    "amdgcn-amd-amdhsa--gfx1036",
    "amdgcn-amd-amdhsa--gfx1100",
    "amdgcn-amd-amdhsa--gfx1101",
    "amdgcn-amd-amdhsa--gfx1102",
    "amdgcn-amd-amdhsa--gfx1103",
    "amdgcn-amd-amdhsa--gfx1150",
    "amdgcn-amd-amdhsa--gfx1151",
    "amdgcn-amd-amdhsa--gfx1152",
    "amdgcn-amd-amdhsa--gfx1153",
    "amdgcn-amd-amdhsa--gfx1154",
    "amdgcn-amd-amdhsa--gfx1170",
    "amdgcn-amd-amdhsa--gfx1171",
    "amdgcn-amd-amdhsa--gfx1172",
    "amdgcn-amd-amdhsa--gfx1200",
    "amdgcn-amd-amdhsa--gfx1201",
    "amdgcn-amd-amdhsa--gfx1250",
    "amdgcn-amd-amdhsa--gfx1250-strict",
    "amdgcn-amd-amdhsa--gfx1251",
    "amdgcn-amd-amdhsa--gfx1310",
    "amdgcn-amd-amdhsa--gfx9-generic",
    "amdgcn-amd-amdhsa--gfx9-4-generic",
    "amdgcn-amd-amdhsa--gfx10-1-generic",
    "amdgcn-amd-amdhsa--gfx10-3-generic",
    "amdgcn-amd-amdhsa--gfx11-generic",
    "amdgcn-amd-amdhsa--gfx11-7-generic",
    "amdgcn-amd-amdhsa--gfx12-generic",
    "amdgcn-amd-amdhsa--gfx12-5-generic",
    "amdgcn-amd-amdhsa--gfx13-generic",
};
// clang-format on

enum {
  ExpectedIsaCount = sizeof(ExpectedIsaNames) / sizeof(ExpectedIsaNames[0])
};

int main(int argc, char *argv[]) {
  size_t IsaCount;
  bool Seen[ExpectedIsaCount] = {false};
  amd_comgr_(get_isa_count(&IsaCount));
  if (IsaCount != ExpectedIsaCount)
    fail("Expected %zu ISAs, got %zu", (size_t)ExpectedIsaCount, IsaCount);
  for (size_t i = 0; i < IsaCount; i++) {
    const char *Name;
    bool sramecc = false, xnack = false;
    amd_comgr_metadata_node_t Root, Features, Val;
    amd_comgr_(get_isa_name(i, &Name));

    // Compare sets rather than relying on TargetParser's enumeration order.
    size_t Index;
    for (Index = 0; Index < ExpectedIsaCount; ++Index)
      if (!strcmp(Name, ExpectedIsaNames[Index]))
        break;
    if (Index == ExpectedIsaCount)
      fail("Unexpected ISA: %s", Name);
    if (Seen[Index])
      fail("Duplicate ISA: %s", Name);
    Seen[Index] = true;

    amd_comgr_(get_isa_metadata(Name, &Root));
    amd_comgr_(metadata_lookup(Root, "Features", &Features));

    if (amd_comgr_metadata_lookup(Features, "sramecc", &Val) ==
        AMD_COMGR_STATUS_SUCCESS) {
      sramecc = true;
      amd_comgr_(destroy_metadata(Val));
    }
    if (amd_comgr_metadata_lookup(Features, "xnack", &Val) ==
        AMD_COMGR_STATUS_SUCCESS) {
      xnack = true;
      amd_comgr_(destroy_metadata(Val));
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
    amd_comgr_(destroy_metadata(Root));
    amd_comgr_(destroy_metadata(Features));
  }
  return 0;
}
