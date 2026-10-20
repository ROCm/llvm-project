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

#include "amd_comgr.h"
#include "common.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/OffloadBundle.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

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

namespace {
constexpr char Magic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char Device[] = "hip-amdgcn-amd-amdhsa-unknown-gfx900";
constexpr char OtherDevice[] = "hip-amdgcn-amd-amdhsa-unknown-gfx906";
constexpr char Host[] = "host-x86_64-unknown-linux-gnu";
constexpr uint64_t MaxCachedUncompressedSize = 64 * 1024 * 1024;

void appendInteger(std::string &Buffer, uint64_t Value) {
  size_t Offset = Buffer.size();
  Buffer.resize(Offset + sizeof(Value));
  support::endian::write64le(Buffer.data() + Offset, Value);
}

std::string bundle(ArrayRef<std::pair<StringRef, StringRef>> Entries) {
  uint64_t Offset = sizeof(Magic) - 1 + sizeof(uint64_t);
  for (const auto &Entry : Entries)
    Offset += 3 * sizeof(uint64_t) + Entry.first.size();

  std::string Result = Magic;
  appendInteger(Result, Entries.size());
  for (const auto &Entry : Entries) {
    appendInteger(Result, Offset);
    appendInteger(Result, Entry.second.size());
    appendInteger(Result, Entry.first.size());
    Result += Entry.first;
    Offset += Entry.second.size();
  }
  for (const auto &Entry : Entries)
    Result += Entry.second;
  return Result;
}

std::string compress(StringRef Input, compression::Format Format,
                     uint16_t Version = 3) {
  auto Buffer = MemoryBuffer::getMemBuffer(Input, "", false);
  auto Compressed = object::CompressedOffloadBundle::compress(
      compression::Params(Format), *Buffer, Version);
  if (!Compressed)
    fail("compress: %s", toString(Compressed.takeError()).c_str());
  return (*Compressed)->getBuffer().str();
}

std::string unbundle(StringRef Input, bool ExpectSuccess,
                     StringRef Target = Device) {
  amd_comgr_data_set_t Inputs, Outputs;
  amd_comgr_data_t InputData, OutputData;
  amd_comgr_action_info_t Action;
  std::string TargetString = Target.str();
  const char *Targets[] = {TargetString.c_str()};

  checkError(amd_comgr_create_data_set(&Inputs), "create inputs");
  checkError(amd_comgr_create_data_set(&Outputs), "create outputs");
  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC_BUNDLE, &InputData),
             "create input");
  checkError(amd_comgr_set_data(InputData, Input.size(), Input.data()),
             "set input");
  checkError(amd_comgr_set_data_name(InputData, "bundle"), "set name");
  checkError(amd_comgr_data_set_add(Inputs, InputData), "add input");
  checkError(amd_comgr_release_data(InputData), "release input");
  checkError(amd_comgr_create_action_info(&Action), "create action");
  checkError(amd_comgr_action_info_set_language(Action, AMD_COMGR_LANGUAGE_HIP),
             "set language");
  checkError(amd_comgr_action_info_set_bundle_entry_ids(Action, Targets, 1),
             "set targets");

  amd_comgr_status_t Status =
      amd_comgr_do_action(AMD_COMGR_ACTION_UNBUNDLE, Action, Inputs, Outputs);
  if (ExpectSuccess)
    checkError(Status, "unbundle");
  else if (Status == AMD_COMGR_STATUS_SUCCESS)
    fail("malformed bundle was accepted");

  std::string Result;
  if (ExpectSuccess) {
    checkError(amd_comgr_action_data_get_data(Outputs, AMD_COMGR_DATA_KIND_BC, 0,
                                               &OutputData),
               "get output");
    size_t Size = 0;
    checkError(amd_comgr_get_data(OutputData, &Size, nullptr), "get size");
    Result.resize(Size);
    checkError(amd_comgr_get_data(OutputData, &Size, Result.data()),
               "get bytes");
    checkError(amd_comgr_release_data(OutputData), "release output");
  }

  checkError(amd_comgr_destroy_action_info(Action), "destroy action");
  checkError(amd_comgr_destroy_data_set(Inputs), "destroy inputs");
  checkError(amd_comgr_destroy_data_set(Outputs), "destroy outputs");
  return Result;
}

size_t countCacheFiles(const std::filesystem::path &CacheDir) {
  std::error_code EC;
  if (!std::filesystem::exists(CacheDir, EC))
    return 0;

  size_t Count = 0;
  for (const auto &Entry :
       std::filesystem::recursive_directory_iterator(CacheDir, EC))
    if (Entry.is_regular_file(EC))
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
    if (MemberSize >
        Contents.size() - 2 * sizeof(uint32_t) - Padding)
      fail("cache entry has an invalid member size");

    // The command output follows an eight-byte size prefix in the cache entry.
    size_t MemberOffset =
        sizeof(uint64_t) + 4 + 2 * sizeof(uint32_t) + Padding;
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
  if (std::error_code EC =
          sys::fs::createUniqueDirectory("comgr-unbundle-cache", CacheDirBuffer))
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

  size_t CachedBeforeLarge = countCacheFiles(CacheDir);
  std::string Large = Small;
  support::endian::write64le(Large.data() + 16,
                             MaxCachedUncompressedSize + 1);
  if (unbundle(Large, true) != "first")
    fail("large bundle output differs");
  if (countCacheFiles(CacheDir) != CachedBeforeLarge)
    fail("oversized bundle was written to the cache");

  std::string LargeConcatenated = Small + Large;
  if (unbundle(LargeConcatenated, true) != "first")
    fail("concatenated bundle output differs");
  if (countCacheFiles(CacheDir) != CachedBeforeLarge)
    fail("oversized concatenated bundle was written to the cache");

  std::string LargeSecond = compress(bundle({{OtherDevice, "second"}}), Format);
  support::endian::write64le(LargeSecond.data() + 16,
                             MaxCachedUncompressedSize + 1);
  if (unbundle(Small + LargeSecond, true, OtherDevice) != "second")
    fail("oversized concatenated bundle output differs");
  if (countCacheFiles(CacheDir) != CachedBeforeLarge)
    fail("oversized concatenated bundle was written to the cache");

  std::string Truncated = Small.substr(0, 24);
  unbundle(Truncated, false);
  if (countCacheFiles(CacheDir) != CachedBeforeLarge)
    fail("truncated bundle changed the cache");

  unbundle("CCOB", false);
  if (countCacheFiles(CacheDir) != CachedBeforeLarge)
    fail("malformed bundle changed the cache");

  std::error_code EC;
  std::filesystem::remove_all(CacheDir, EC);
  return 0;
}
