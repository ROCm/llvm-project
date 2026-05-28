//===- parse-isa-name.c ------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  const char *reason = "";

  if (argc != 3) {
    fprintf(stderr, "Usage: parse-isa-name <isa-name> <expected-status>\n");
    exit(1);
  }
  amd_comgr_(create_action_info(&DataAction));
  Status = amd_comgr_action_info_set_isa_name(DataAction, argv[1]);
  amd_comgr_status_string(Status, &reason);
  if (strcmp(reason, argv[2])) {
    fail("INVALID: %s\n", reason);
  }
  else
    printf("OK\n");
  return 0;
}
