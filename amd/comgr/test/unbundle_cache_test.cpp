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
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;

namespace {
constexpr char Magic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char Device[] = "hip-amdgcn-amd-amdhsa-unknown-gfx900";
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

std::string compress(StringRef Input, compression::Format Format) {
  auto Buffer = MemoryBuffer::getMemBuffer(Input, "", false);
  auto Compressed = object::CompressedOffloadBundle::compress(
      compression::Params(Format), *Buffer, 3);
  if (!Compressed)
    fail("compress: %s", toString(Compressed.takeError()).c_str());
  return (*Compressed)->getBuffer().str();
}

std::string unbundle(StringRef Input, bool ExpectSuccess) {
  amd_comgr_data_set_t Inputs, Outputs;
  amd_comgr_data_t InputData, OutputData;
  amd_comgr_action_info_t Action;
  const char *Targets[] = {Device};

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

  std::string Large = Small;
  support::endian::write64le(Large.data() + 16,
                             MaxCachedUncompressedSize + 1);
  if (unbundle(Large, true) != "first")
    fail("large bundle output differs");
  if (countCacheFiles(CacheDir) != CachedAfterSmall)
    fail("oversized bundle was written to the cache");

  std::string Concatenated = Small + Large;
  if (unbundle(Concatenated, true) != "first")
    fail("concatenated bundle output differs");
  if (countCacheFiles(CacheDir) != CachedAfterSmall)
    fail("oversized concatenated bundle was written to the cache");

  std::string Truncated = Small.substr(0, 24);
  unbundle(Truncated, false);
  if (countCacheFiles(CacheDir) != CachedAfterSmall)
    fail("truncated bundle changed the cache");

  unbundle("CCOB", false);
  if (countCacheFiles(CacheDir) != CachedAfterSmall)
    fail("malformed bundle changed the cache");

  std::error_code EC;
  std::filesystem::remove_all(CacheDir, EC);
  return 0;
}
