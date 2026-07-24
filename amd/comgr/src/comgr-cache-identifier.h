//===- comgr-cache-identifier.h - Cache compatibility identity -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_CACHE_IDENTIFIER_H
#define COMGR_CACHE_IDENTIFIER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <cstdint>

namespace COMGR {

/// Add an integer to @p Hash using a stable little-endian representation.
void addCacheHashUInt(llvm::SHA256 &Hash, uint64_t Value);

/// Add length-delimited bytes to @p Hash.
void addCacheHashBytes(llvm::SHA256 &Hash, llvm::ArrayRef<uint8_t> Value);

/// Add a length-delimited string to @p Hash.
void addCacheHashString(llvm::SHA256 &Hash, llvm::StringRef Value);

/// Return the raw digest identifying this Comgr implementation and its embedded
/// compilation inputs.
llvm::ArrayRef<uint8_t> getComgrImplementationIdentifier();

/// Return the public hexadecimal representation of the implementation digest.
llvm::StringRef getCacheIdentifier();

} // namespace COMGR

#endif // COMGR_CACHE_IDENTIFIER_H
