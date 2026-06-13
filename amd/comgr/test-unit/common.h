//===- common.h -----------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TEST_UNIT_COMMON_H
#define COMGR_TEST_UNIT_COMMON_H

#include "amd_comgr.h"

#define ASSERT_COMGR(call)                                                \
  do {                                                                    \
    amd_comgr_status_t status = amd_comgr_##call;                         \
    ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS, status);                          \
  } while (false)
#endif // COMGR_TEST_COMMON_H
