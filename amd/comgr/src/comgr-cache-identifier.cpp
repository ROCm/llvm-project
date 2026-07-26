//===- comgr-cache-identifier.cpp - Cache compatibility identity ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Computes the implementation identity returned by the public
/// cache-identifier API.
///
//===----------------------------------------------------------------------===//

#include "comgr-cache-identifier.h"
#include "comgr-device-libs.h"
#include "comgr-libcxx-headers.h"
#include "comgr-resource-directory.h"
#include "comgr.h"

#include "clang/Basic/Version.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <array>
#include <string>

using namespace llvm;

namespace COMGR {
namespace {

void addCacheHashUInt(SHA256 &Hash, uint64_t Value) {
  uint8_t Bytes[sizeof(Value)];
  for (size_t I = 0; I != sizeof(Bytes); ++I)
    Bytes[I] = static_cast<uint8_t>(Value >> (I * 8));
  Hash.update(Bytes);
}

void addCacheHashBytes(SHA256 &Hash, ArrayRef<uint8_t> Value) {
  addCacheHashUInt(Hash, Value.size());
  Hash.update(Value);
}

void addCacheHashString(SHA256 &Hash, StringRef Value) {
  addCacheHashBytes(
      Hash, ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Value.data()),
                              Value.size()));
}

void addCacheHashResources(SHA256 &Hash, StringRef Kind,
                           ArrayRef<ResourceDirResource> Resources) {
  addCacheHashString(Hash, Kind);
  addCacheHashUInt(Hash, Resources.size());
  for (const ResourceDirResource &Resource : Resources) {
    addCacheHashString(Hash, Resource.RelativePath);
    addCacheHashString(Hash, Resource.FileContent);
  }
}

std::array<uint8_t, 32> computeCacheIdentifier() {
  SHA256 Hash;
  addCacheHashString(Hash, "amd_comgr_cache_identifier_v1");
  addCacheHashString(Hash, clang::getClangFullVersion());
  addCacheHashString(Hash, getComgrHashIdentifier());
  // The interface version need not change when compiler implementation changes
  // alter output, so include the exact configured Comgr source revision.
  addCacheHashString(Hash, getComgrGitCommitIdentifier());

  addCacheHashBytes(Hash, getDeviceLibrariesIdentifier());
  addCacheHashString(Hash, getOpenCLCBaseHeaderContents());
  addCacheHashResources(Hash, "resource-directory",
                        getResourceDirectoryFiles());
  addCacheHashResources(Hash, "libcxx-headers", getLibcxxHeaderFiles());
  addCacheHashResources(Hash, "clang-builtin-headers",
                        getClangBuiltinHeaderFiles());

  addCacheHashString(Hash, "spirv-translator");
#ifdef COMGR_SPIRV_TRANSLATOR_AVAILABLE
  addCacheHashUInt(Hash, 1);
#else
  addCacheHashUInt(Hash, 0);
#endif

  addCacheHashString(Hash, "spirv-backend");
#ifdef COMGR_SPIRV_BACKEND_AVAILABLE
  addCacheHashUInt(Hash, 1);
#else
  addCacheHashUInt(Hash, 0);
#endif

  addCacheHashString(Hash, "hotswap-transpiler");
#ifdef COMGR_ENABLE_HOTSWAP_TRANSPILE
  addCacheHashUInt(Hash, 1);
#else
  addCacheHashUInt(Hash, 0);
#endif

  return Hash.final();
}

} // namespace

StringRef getCacheIdentifier() {
  static const std::string Identifier = []() {
    SmallString<64> Hex;
    toHex(computeCacheIdentifier(), true, Hex);
    return std::string(Hex.begin(), Hex.end());
  }();
  return Identifier;
}

} // namespace COMGR
