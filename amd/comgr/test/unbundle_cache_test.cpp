//===- unbundle_cache_test.cpp --------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compressed bundles that expand to a very large image are intentionally not
// persisted in the COMGR cache. The cache entry would contain a second copy of
// the decompressed image and costs more than regenerating it. This test checks
// that a normal bundle is cached, a bundle with an oversized header is not, and
// a malformed bundle is rejected without touching the cache.
//
//===----------------------------------------------------------------------===//

#include "unbundle_test.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace UnbundleTest;

namespace {
constexpr uint64_t MaxCachedUncompressedSize = 64 * 1024 * 1024;

std::string unbundle(StringRef Input, bool ExpectSuccess,
                     StringRef Target = Device) {
  std::string TargetString = Target.str();
  auto Data = unbundleData(Input, {TargetString.c_str()}, ExpectSuccess);

  std::string Result;
  if (ExpectSuccess) {
    size_t Size = 0;
    checkError(amd_comgr_get_data(Data.front(), &Size, nullptr), "get size");
    Result.resize(Size);
    checkError(amd_comgr_get_data(Data.front(), &Size, Result.data()),
               "get output");
    checkError(amd_comgr_release_data(Data.front()), "release output");
  }
  return Result;
}

size_t countCacheFiles(const std::filesystem::path &CacheDir) {
  std::error_code EC;
  if (!std::filesystem::exists(CacheDir, EC))
    return 0;

  size_t Count = 0;
  for (const auto &Entry :
       std::filesystem::recursive_directory_iterator(CacheDir, EC))
    if (Entry.is_regular_file(EC) &&
        Entry.path().filename().string().rfind("llvmcache-", 0) == 0)
      ++Count;
  return Count;
}

void checkCacheFraming(const std::filesystem::path &CacheDir) {
  std::error_code EC;
  for (const auto &Entry : std::filesystem::directory_iterator(CacheDir, EC)) {
    std::string Name = Entry.path().filename().string();
    if (Name.rfind("llvmcache-", 0) != 0 || Name == "llvmcache.timestamp")
      continue;

    std::ifstream File(Entry.path(), std::ios::binary);
    std::string Bytes((std::istreambuf_iterator<char>(File)),
                      std::istreambuf_iterator<char>());
    if (Bytes.size() < 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t))
      continue;

    uint64_t ContentsSize;
    memcpy(&ContentsSize, Bytes.data(), sizeof(ContentsSize));
    if (ContentsSize > Bytes.size() - sizeof(ContentsSize))
      fail("cache entry has an invalid contents size");

    StringRef Contents(Bytes.data() + sizeof(ContentsSize), ContentsSize);
    if (!Contents.consume_front("CUB1"))
      continue;

    uint32_t MemberSize, Padding;
    if (Contents.size() < 2 * sizeof(uint32_t))
      fail("cache entry has a truncated member header");
    memcpy(&MemberSize, Contents.data(), sizeof(MemberSize));
    memcpy(&Padding, Contents.data() + sizeof(MemberSize), sizeof(Padding));
    if (Padding > Contents.size() - 2 * sizeof(uint32_t))
      fail("cache entry has invalid member padding");
    if (MemberSize > Contents.size() - 2 * sizeof(uint32_t) - Padding)
      fail("cache entry has an invalid member size");

    // The command output follows an eight-byte size prefix in the cache entry.
    size_t MemberOffset = sizeof(uint64_t) + 4 + 2 * sizeof(uint32_t) + Padding;
    if (MemberOffset % alignof(std::max_align_t) != 0)
      fail("cached member is not aligned");
    return;
  }

  fail("no aligned cache entry found");
}

void setEnv(const char *Name, const char *Value) {
#if defined(_WIN32)
  if (_putenv_s(Name, Value) != 0)
    fail("_putenv_s(%s)", Name);
#else
  if (setenv(Name, Value, 1) != 0)
    fail("setenv(%s)", Name);
#endif
}
} // namespace

int main() {
  SmallString<128> CacheDirBuffer;
  if (std::error_code EC = sys::fs::createUniqueDirectory(
          "comgr-unbundle-cache", CacheDirBuffer))
    fail("create cache directory: %s", EC.message().c_str());
  std::filesystem::path CacheDir(
      std::string(CacheDirBuffer.begin(), CacheDirBuffer.end()));

  setEnv("AMD_COMGR_CACHE", "1");
  setEnv("AMD_COMGR_CACHE_DIR", CacheDir.string().c_str());

  auto Format = compression::Format::Zstd;
  if (compression::getReasonIfUnsupported(Format))
    Format = compression::Format::Zlib;
  if (compression::getReasonIfUnsupported(Format))
    fail("no supported compression format");

  std::string Small = compress(bundle({{Device, "first"}}), Format);
  if (unbundle(Small, true) != "first")
    fail("small bundle output differs");

  size_t CachedAfterSmall = countCacheFiles(CacheDir);
  if (CachedAfterSmall == 0)
    fail("small bundle was not cached");
  checkCacheFraming(CacheDir);

  // A host target returns the original compressed bundle. The cache key must
  // distinguish compressed representations, not just the decompressed bytes.
  std::string HostBundle = bundle({{Device, "host"}});
  std::string HostVersion2 = compress(HostBundle, Format, 2);
  std::string HostVersion3 = compress(HostBundle, Format, 3);
  if (unbundle(HostVersion2, true, Host) != HostVersion2)
    fail("version 2 host bundle output differs");
  if (unbundle(HostVersion3, true, Host) != HostVersion3)
    fail("version 3 host bundle reused the version 2 cache entry");

  if (!compression::getReasonIfUnsupported(compression::Format::Zlib)) {
    std::string Original = compress(HostBundle, compression::Format::Zlib);
    std::string Alternate = Original;
    // Change the zlib FLEVEL bits while leaving FDICT clear. FLEVEL is
    // informational, so the stream still decompresses to the same bytes. The
    // CCOB header and compressed length remain unchanged, which is exactly the
    // case the host cache key must distinguish.
    uint8_t CMF = static_cast<unsigned char>(Alternate[32]);
    uint8_t FLG = (static_cast<unsigned char>(Alternate[33]) ^ 0x80) & 0xc0;
    uint16_t Header = (static_cast<uint16_t>(CMF) << 8) | FLG;
    Header += (31 - Header % 31) % 31;
    Alternate[32] = Header >> 8;
    Alternate[33] = Header & 0xff;
    if (Original.substr(0, 32) != Alternate.substr(0, 32) ||
        Original.substr(32) == Alternate.substr(32))
      fail("zlib mutation changed the CCOB header or compressed bytes");
    if (unbundle(Original, true, Host) != Original ||
        unbundle(Alternate, true, Host) != Alternate)
      fail("host cache key ignored compressed bytes with identical headers");
    if (unbundle(Alternate, true) != "host")
      fail("alternate compressed representation is invalid");
  }

  // A cache key must cover every concatenated bundle, not only the first
  // header. Cache an empty result for a target absent from the first bundle,
  // then verify that adding a bundle containing the target produces a miss and
  // the expected member.
  if (!unbundle(Small, true, OtherDevice).empty())
    fail("unexpected member for missing target");
  std::string Concatenated =
      Small + compress(bundle({{OtherDevice, "second"}}), Format);
  if (unbundle(Concatenated, true, OtherDevice) != "second")
    fail("concatenated bundle reused the first bundle's cache entry");

  for (uint16_t Version : {1, 2, 3}) {
    std::string Boundary =
        compress(bundle({{OtherDevice, "second"}}), Format, Version);
    auto SetExpandedSize = [&](uint64_t Size) {
      if (Version == 3)
        support::endian::write64le(Boundary.data() + 16, Size);
      else
        support::endian::write32le(Boundary.data() + (Version == 1 ? 8 : 12),
                                   Size);
    };
    SetExpandedSize(MaxCachedUncompressedSize);
    size_t Before = countCacheFiles(CacheDir);
    if (unbundle(Boundary, true, OtherDevice) != "second" ||
        countCacheFiles(CacheDir) != Before + 1)
      fail("bundle at the cache size limit was not cached");

    SetExpandedSize(MaxCachedUncompressedSize + 1);
    Before = countCacheFiles(CacheDir);
    for (const std::string &Input :
         {Boundary, Small + Boundary, Boundary + Small}) {
      if (unbundle(Input, true, OtherDevice) != "second")
        fail("oversized bundle output differs");
      if (countCacheFiles(CacheDir) != Before)
        fail("oversized bundle was written to the cache");
    }
  }

  size_t CachedBeforeLarge = countCacheFiles(CacheDir);
  std::string Truncated = Small.substr(0, 24);
  unbundle(Truncated, false);
  if (countCacheFiles(CacheDir) != CachedBeforeLarge)
    fail("truncated bundle changed the cache");

  unbundle("CCOB", false);
  if (countCacheFiles(CacheDir) != CachedBeforeLarge)
    fail("malformed bundle changed the cache");

  for (const std::string &Input :
       {Small + "trailing bytes",
        Small + std::string("\0padding\0", 9) + Small}) {
    if (unbundle(Input, true) != "first")
      fail("padded bundle output differs");
    if (countCacheFiles(CacheDir) != CachedBeforeLarge)
      fail("padded bundle unexpectedly changed the cache");
  }

  // Keep a cache-hit result alive while pruning its backing entry. In
  // particular, Windows must copy cached members so the mapping does not
  // prevent removal of the cache file. Use enough data to require an mmap.
  std::string Payload(64 * 1024, 'x');
  std::string Prunable = compress(bundle({{Device, Payload}}), Format);
  if (unbundle(Prunable, true) != Payload)
    fail("prunable bundle output differs");
  auto Retained = unbundleData(Prunable, {Device});
  std::error_code EC;
  std::filesystem::remove_all(CacheDir, EC);
  if (EC)
    fail("could not prune cache with retained output: %s",
         EC.message().c_str());
  size_t Size = 0;
  checkError(amd_comgr_get_data(Retained.front(), &Size, nullptr), "get size");
  std::string Contents(Size, '\0');
  checkError(amd_comgr_get_data(Retained.front(), &Size, Contents.data()),
             "get pruned output");
  if (Contents != Payload)
    fail("pruning invalidated retained output");
  checkError(amd_comgr_release_data(Retained.front()), "release pruned output");
  return 0;
}
