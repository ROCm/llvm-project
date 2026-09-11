//===- comgr-hotswap-stubs.cpp - Deprecated HotSwap rewrite entry points --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The gfx1250 B0-to-A0 byte rewriter these entry points drove has been removed.
// No source/target ISA pair selects an enabled transformation any more, so every
// call takes the documented "unsupported combination" path. These definitions
// exist only to keep the amd_comgr_3.2 and amd_comgr_3.4 symbol versions in
// src/exportmap.in intact; drop them, the declarations, and the version nodes at
// Comgr v4.0.
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite(
    amd_comgr_data_t /*input*/, const char * /*source_isa_name*/,
    const char * /*target_isa_name*/, amd_comgr_data_t * /*output*/) {
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite_with_options(
    amd_comgr_data_t /*input*/, const char * /*source_isa_name*/,
    const char * /*target_isa_name*/,
    const amd_comgr_hotswap_rewrite_options_t * /*rewrite_options*/,
    amd_comgr_data_t * /*output*/) {
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}
