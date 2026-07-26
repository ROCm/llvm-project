//===- comgr-cache-identifier.h - Cache compatibility identity -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_CACHE_IDENTIFIER_H
#define COMGR_CACHE_IDENTIFIER_H

#include "llvm/ADT/StringRef.h"

namespace COMGR {

/// Return the opaque public cache compatibility identifier.
llvm::StringRef getCacheIdentifier();

} // namespace COMGR

#endif // COMGR_CACHE_IDENTIFIER_H
