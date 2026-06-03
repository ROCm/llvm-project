//===- common.h -----------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TEST_COMMON_H
#define COMGR_TEST_COMMON_H

#include "amd_comgr.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#else // Windows
#include <io.h>
#endif
#include <errno.h>
#include <fcntl.h>
#define MAX_ISA_NAME_SIZE 1024

#define ASSERT_COMGR(call)                                                      \
  do {                                                                         \
    amd_comgr_status_t status = amd_comgr_##call;                              \
    ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS, status);                              \
  } while (false)
#endif // COMGR_TEST_COMMON_H
