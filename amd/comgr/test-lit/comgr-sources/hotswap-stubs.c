//===- hotswap-stubs.c ----------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These deprecated APIs remain callable until Comgr v4.0.
#define AMD_COMGR_DEPRECATED(msg)
#include "amd_comgr.h"
#include "common.h"

int main(void) {
  amd_comgr_data_t Input = {0};
  amd_comgr_data_t Output = {42};
  amd_comgr_hotswap_rewrite_options_t Options = {sizeof(Options), 0};
  const char *Isa = "amdgcn-amd-amdhsa--gfx1250";

  amd_comgr_status_t Status =
      amd_comgr_hotswap_rewrite(Input, Isa, Isa, &Output);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    fail("rewrite returned %d instead of INVALID_ARGUMENT", Status);
  if (Output.handle != 42)
    fail("rewrite modified the output handle");

  Status = amd_comgr_hotswap_rewrite_with_options(Input, Isa, Isa, &Options,
                                                  &Output);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    fail("rewrite_with_options returned %d instead of INVALID_ARGUMENT",
         Status);
  if (Output.handle != 42)
    fail("rewrite_with_options modified the output handle");

  Status = amd_comgr_hotswap_rewrite(Input, NULL, NULL, NULL);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    fail("rewrite with null arguments returned %d", Status);
  Status =
      amd_comgr_hotswap_rewrite_with_options(Input, NULL, NULL, NULL, NULL);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    fail("rewrite_with_options with null arguments returned %d", Status);

  return 0;
}
