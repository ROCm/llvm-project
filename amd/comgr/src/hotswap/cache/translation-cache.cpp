#include "translation-cache.h"

#include "code-object-utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <string>
#include <sys/stat.h>

#define DEBUG_TYPE "translation-cache"

namespace COMGR::hotswap {
namespace {

using TimingClock = std::chrono::steady_clock;

double secondsBetween(TimingClock::time_point start,
                      TimingClock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

TimingClock::time_point timingStart(bool CollectTimings) {
  return CollectTimings ? TimingClock::now() : TimingClock::time_point{};
}

double timingElapsed(bool CollectTimings, TimingClock::time_point start) {
  return CollectTimings ? secondsBetween(start, TimingClock::now()) : 0.0;
}

constexpr int kCacheSchemaVersion = 4;

struct FileIdentity {
  std::string path;
  bool present = false;
  uint64_t size = 0;
  int64_t mtimeSec = 0;
  int64_t mtimeNsec = 0;
  std::string sha256;
};

struct KeyData {
  std::string key;
  std::string sourceSha256;
  std::string rulesSha256;
  std::string buildIdentity;
  std::string deviceLibrariesIdentity;
  std::string elfMachineHex;
  std::string elfFlagsHex;
  std::vector<std::string> kernelNames;
};

std::string hexU32(uint32_t value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "0x" << llvm::format_hex_no_prefix(value, 0);
  return os.str();
}

struct ElfHeaderFields {
  uint16_t machine = 0;
  uint32_t flags = 0;
};

llvm::Expected<ElfHeaderFields>
readElfHeaderFields(llvm::MemoryBufferRef buffer) {
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(buffer);
  if (!objOrErr)
    return objOrErr.takeError();
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(objOrErr->get());
  if (!elf)
    return llvm::createStringError(
        "source code object is not a 64-bit little-endian ELF");
  return ElfHeaderFields{elf->getEMachine(), elf->getPlatformFlags()};
}

llvm::Expected<std::string> hashFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::errorCodeToError(buffer.getError());
  return sha256Hex((*buffer)->getMemBufferRef());
}

llvm::Expected<FileIdentity> statIdentity(llvm::StringRef path) {
  FileIdentity id;
  id.path = path.str();
  struct stat st;
  if (::stat(id.path.c_str(), &st) != 0)
    return id;
  id.present = true;
  id.size = static_cast<uint64_t>(st.st_size);
#if defined(__linux__)
  id.mtimeSec = static_cast<int64_t>(st.st_mtim.tv_sec);
  id.mtimeNsec = static_cast<int64_t>(st.st_mtim.tv_nsec);
#else
  id.mtimeSec = static_cast<int64_t>(st.st_mtime);
  id.mtimeNsec = 0;
#endif
  llvm::Expected<std::string> hash = hashFile(id.path);
  if (!hash)
    return llvm::createStringError(id.path + ": " +
                                   llvm::toString(hash.takeError()));
  id.sha256 = std::move(*hash);
  return id;
}

std::string identityString(const FileIdentity &id) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << id.path << "|present=" << (id.present ? "1" : "0")
     << "|size=" << id.size << "|mtime=" << id.mtimeSec << "." << id.mtimeNsec
     << "|sha256=" << id.sha256;
  return os.str();
}

std::string loadedImageIdentity() {
  static const std::string identity = [] {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << "llvm=" << LLVM_VERSION_STRING;
#ifndef _WIN32
    Dl_info info;
    if (::dladdr(reinterpret_cast<void *>(&loadedImageIdentity), &info) &&
        info.dli_fname) {
      if (auto id = statIdentity(info.dli_fname))
        os << "|image=" << identityString(*id);
      else
        os << "|image=<error=" << llvm::toString(id.takeError()) << ">";
    } else {
      os << "|image=<dladdr-unavailable>";
    }
#else
    os << "|image=<unavailable-on-windows>";
#endif
    return os.str();
  }();
  return identity;
}

void appendKeyField(std::string &material, llvm::StringRef name,
                    llvm::StringRef value) {
  material.append(name.data(), name.size());
  material.push_back('\0');
  material += std::to_string(value.size());
  material.push_back(':');
  if (!value.empty())
    material.append(value.data(), value.size());
  material.push_back('\0');
}

void appendKeyField(std::string &material, llvm::StringRef name, bool value) {
  appendKeyField(material, name, llvm::StringRef(value ? "true" : "false"));
}

void appendKeyField(std::string &material, llvm::StringRef name, int value) {
  appendKeyField(material, name, std::to_string(value));
}

llvm::Expected<KeyData> buildKeyData(const TranslationCacheRequest &request,
                                     bool CollectTimings,
                                     TranslationCacheKeyBuildTimings &Timings) {
  KeyData data;
  if (request.SourceObject.getBufferSize() == 0)
    return llvm::createStringError("empty source code object");
  if (request.SourceGfx.empty() || request.TargetGfx.empty())
    return llvm::createStringError("missing source or target gfx");

  auto sourceHashStart = timingStart(CollectTimings);
  data.sourceSha256 = sha256Hex(request.SourceObject);
  Timings.sourceHashSeconds = timingElapsed(CollectTimings, sourceHashStart);

  auto elfHeaderStart = timingStart(CollectTimings);
  llvm::Expected<ElfHeaderFields> elfHeader =
      readElfHeaderFields(request.SourceObject);
  Timings.elfHeaderSeconds = timingElapsed(CollectTimings, elfHeaderStart);
  if (!elfHeader)
    return elfHeader.takeError();
  data.elfMachineHex = hexU32(elfHeader->machine);
  data.elfFlagsHex = hexU32(elfHeader->flags);

  if (!request.HotswapRulesPath.empty()) {
    auto rulesHashStart = timingStart(CollectTimings);
    llvm::Expected<std::string> rulesHash = hashFile(request.HotswapRulesPath);
    Timings.rulesHashSeconds = timingElapsed(CollectTimings, rulesHashStart);
    if (!rulesHash)
      return llvm::createStringError(
          "failed to hash HSA_HOTSWAP_RULES '" + request.HotswapRulesPath +
          "': " + llvm::toString(rulesHash.takeError()));
    data.rulesSha256 = std::move(*rulesHash);
  }

  auto loadedImageIdentityStart = timingStart(CollectTimings);
  data.buildIdentity = loadedImageIdentity();
  Timings.loadedImageIdentitySeconds =
      timingElapsed(CollectTimings, loadedImageIdentityStart);
  data.deviceLibrariesIdentity = request.DeviceLibrariesIdentity;
  if (!request.KernelName.empty()) {
    data.kernelNames.push_back(request.KernelName);
  } else {
    auto kernelNamesStart = timingStart(CollectTimings);
    llvm::Expected<llvm::SmallVector<std::string>> NamesOrErr =
        listKernelNames(request.SourceObject);
    Timings.kernelNamesSeconds =
        timingElapsed(CollectTimings, kernelNamesStart);
    if (!NamesOrErr)
      return llvm::createStringError(
          "failed to list kernels for translation cache key: " +
          llvm::toString(NamesOrErr.takeError()));
    data.kernelNames.assign(NamesOrErr->begin(), NamesOrErr->end());
    if (data.kernelNames.empty())
      return llvm::createStringError(
          "source code object has no kernel metadata entries");
  }

  auto materialBuildStart = timingStart(CollectTimings);
  std::string material;
  appendKeyField(material, "schema", std::to_string(kCacheSchemaVersion));
  appendKeyField(material, "source_sha256", data.sourceSha256);
  appendKeyField(material, "source_gfx", request.SourceGfx);
  appendKeyField(material, "target_gfx", request.TargetGfx);
  appendKeyField(material, "source_isa", request.SourceIsa);
  appendKeyField(material, "target_isa", request.TargetIsa);
  appendKeyField(material, "code_isa", request.CodeIsa);
  appendKeyField(material, "elf_machine", data.elfMachineHex);
  appendKeyField(material, "elf_flags", data.elfFlagsHex);
  appendKeyField(material, "orig_mach", request.OrigMach);
  appendKeyField(material, "opt_level", static_cast<int>(request.OptLevel));
  appendKeyField(material, "rules_path", request.HotswapRulesPath);
  appendKeyField(material, "rules_sha256", data.rulesSha256);
  appendKeyField(material, "strict", request.StrictMode);
  appendKeyField(material, "enable_writelane_rewrite",
                 request.EnableWritelaneRewrite);
  appendKeyField(material, "enable_wave_native", request.EnableWaveNative);
  appendKeyField(material, "force_scaled_modrep", request.ForceScaledModrep);
  appendKeyField(material, "assume_hip_global_offset_zero",
                 request.AssumeHipGlobalOffsetZero);
  if (!request.KernelName.empty())
    appendKeyField(material, "kernel_name", request.KernelName);
  appendKeyField(material, "hotswap_build_identity", data.buildIdentity);
  appendKeyField(material, "device_libraries_identity",
                 data.deviceLibrariesIdentity);
  Timings.materialBuildSeconds =
      timingElapsed(CollectTimings, materialBuildStart);
  auto keyHashStart = timingStart(CollectTimings);
  data.key = sha256Hex(llvm::MemoryBufferRef(material, ""));
  Timings.keyHashSeconds = timingElapsed(CollectTimings, keyHashStart);
  return data;
}

std::string cacheRoot(const TranslationCacheRequest &request) {
  return request.CacheDirectory;
}

bool cacheDisabledByPolicy(const TranslationCacheRequest &request) {
  return request.CacheDisabled || cacheRoot(request).empty();
}

std::string cacheSubdir(const TranslationCacheRequest &request,
                        llvm::StringRef key) {
  llvm::SmallString<256> path(cacheRoot(request));
  llvm::sys::path::append(path, key.substr(0, 2));
  return std::string(path);
}

std::string cacheObjectPath(const TranslationCacheRequest &request,
                            llvm::StringRef key) {
  llvm::SmallString<256> path(cacheSubdir(request, key));
  llvm::sys::path::append(path, llvm::Twine(key) + ".Hsaco");
  return std::string(path);
}

std::string cacheMetadataPath(const TranslationCacheRequest &request,
                              llvm::StringRef key) {
  llvm::SmallString<256> path(cacheSubdir(request, key));
  llvm::sys::path::append(path, llvm::Twine(key) + ".json");
  return std::string(path);
}

bool exists(llvm::StringRef path) { return llvm::sys::fs::exists(path); }

std::string jsonToString(llvm::json::Value value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  value.print(os);
  os << "\n";
  return os.str();
}

llvm::Error writeFileAtomic(llvm::StringRef path, llvm::StringRef contents) {
  return llvm::writeToOutput(path, [&](llvm::raw_ostream &os) {
    os << contents;
    return llvm::Error::success();
  });
}

llvm::Error writeFileAtomic(llvm::StringRef path,
                            llvm::ArrayRef<uint8_t> data) {
  return writeFileAtomic(
      path, llvm::StringRef(reinterpret_cast<const char *>(data.data()),
                            data.size()));
}

llvm::Expected<std::string> requireString(const llvm::json::Object &obj,
                                          llvm::StringRef field) {
  auto value = obj.getString(field);
  if (!value)
    return llvm::createStringError("metadata field '" + field +
                                   "' missing or not a string");
  return value->str();
}

llvm::Expected<int64_t> requireInt(const llvm::json::Object &obj,
                                   llvm::StringRef field) {
  auto value = obj.getInteger(field);
  if (!value)
    return llvm::createStringError("metadata field '" + field +
                                   "' missing or not an integer");
  return *value;
}

llvm::Expected<bool> requireBool(const llvm::json::Object &obj,
                                 llvm::StringRef field) {
  auto value = obj.getBoolean(field);
  if (!value)
    return llvm::createStringError("metadata field '" + field +
                                   "' missing or not a boolean");
  return *value;
}

llvm::Error requireEqualString(const llvm::json::Object &obj,
                               llvm::StringRef field,
                               llvm::StringRef expected) {
  auto value = requireString(obj, field);
  if (!value)
    return value.takeError();
  if (*value != expected)
    return llvm::createStringError("metadata field '" + field + "' mismatch");
  return llvm::Error::success();
}

llvm::Error validateKernelNameField(const llvm::json::Object &obj,
                                    llvm::StringRef expected) {
  const llvm::json::Value *rawValue = obj.get("kernel_name");
  if (!rawValue) {
    if (expected.empty())
      return llvm::Error::success();
    return llvm::createStringError("metadata field 'kernel_name' missing");
  }
  auto value = rawValue->getAsString();
  if (!value)
    return llvm::createStringError(
        "metadata field 'kernel_name' is not a string");
  if (*value != expected)
    return llvm::createStringError("metadata field 'kernel_name' mismatch");
  return llvm::Error::success();
}

llvm::Error requireEqualInt(const llvm::json::Object &obj,
                            llvm::StringRef field, int64_t expected) {
  auto value = requireInt(obj, field);
  if (!value)
    return value.takeError();
  if (*value != expected)
    return llvm::createStringError("metadata field '" + field + "' mismatch");
  return llvm::Error::success();
}

llvm::Error requireEqualBool(const llvm::json::Object &obj,
                             llvm::StringRef field, bool expected) {
  auto value = requireBool(obj, field);
  if (!value)
    return value.takeError();
  if (*value != expected)
    return llvm::createStringError("metadata field '" + field + "' mismatch");
  return llvm::Error::success();
}

llvm::json::Array kernelArray(const std::vector<std::string> &kernelNames) {
  llvm::json::Array arr;
  for (llvm::StringRef name : kernelNames)
    arr.push_back(name);
  return arr;
}

llvm::Error validateKernelArray(const llvm::json::Object &obj,
                                const std::vector<std::string> &expected) {
  const llvm::json::Array *arr = obj.getArray("kernel_names");
  if (!arr)
    return llvm::createStringError(
        "metadata field 'kernel_names' missing or not an array");
  if (arr->size() != expected.size())
    return llvm::createStringError("metadata kernel_names size mismatch");
  for (size_t i = 0; i < expected.size(); ++i) {
    auto value = (*arr)[i].getAsString();
    if (!value || *value != expected[i])
      return llvm::createStringError("metadata kernel_names mismatch");
  }
  return llvm::Error::success();
}

llvm::json::Object metadataObject(const TranslationCacheRequest &request,
                                  const KeyData &keyData,
                                  const PipelineResult &Result,
                                  llvm::StringRef objectSha256) {
  llvm::json::Object Obj{
      {"schema_version", kCacheSchemaVersion},
      {"key", keyData.key},
      {"source_object_sha256", keyData.sourceSha256},
      {"source_gfx", request.SourceGfx},
      {"target_gfx", request.TargetGfx},
      {"source_isa", request.SourceIsa},
      {"target_isa", request.TargetIsa},
      {"code_isa", request.CodeIsa},
      {"elf_machine", keyData.elfMachineHex},
      {"elf_flags", keyData.elfFlagsHex},
      {"orig_mach", request.OrigMach},
      {"opt_level", static_cast<int64_t>(request.OptLevel)},
      {"hotswap_rules_path", request.HotswapRulesPath},
      {"hotswap_rules_sha256", keyData.rulesSha256},
      {"strict_mode", request.StrictMode},
      {"enable_writelane_rewrite", request.EnableWritelaneRewrite},
      {"enable_wave_native", request.EnableWaveNative},
      {"force_scaled_modrep", request.ForceScaledModrep},
      {"assume_hip_global_offset_zero", request.AssumeHipGlobalOffsetZero},
      {"hotswap_build_identity", keyData.buildIdentity},
      {"device_libraries_identity", keyData.deviceLibrariesIdentity},
      {"kernel_count", static_cast<int64_t>(keyData.kernelNames.size())},
      {"kernel_names", kernelArray(keyData.kernelNames)},
      {"cached_object_sha256", objectSha256.str()},
      {"cached_object_size",
       static_cast<int64_t>(Result.Hsaco ? Result.Hsaco->getBufferSize() : 0)},
      {"lifted_count", Result.LiftedCount},
      {"total_count", Result.TotalCount},
      {"scaled_dispatch_factor",
       static_cast<int64_t>(Result.ScaledDispatchFactor)},
      {"c5_suppressed_count", Result.C5SuppressedCount},
      {"c5_suppression_reason", Result.C5SuppressionReason},
      {"uses_scratch_private_segment", Result.UsesScratchPrivateSegment},
      {"source_private_segment_fixed_size",
       static_cast<int64_t>(Result.SourcePrivateSegmentFixedSize)},
      {"target_private_segment_fixed_size",
       static_cast<int64_t>(Result.TargetPrivateSegmentFixedSize)},
      {"target_enable_private_segment", Result.TargetEnablePrivateSegment},
  };
  if (!request.KernelName.empty())
    Obj["kernel_name"] = request.KernelName;
  return Obj;
}

llvm::Error validateMetadata(const TranslationCacheRequest &request,
                             const KeyData &keyData,
                             const llvm::json::Object &obj,
                             llvm::StringRef objectSha256, size_t objectSize,
                             PipelineResult &Result) {
  if (llvm::Error e =
          requireEqualInt(obj, "schema_version", kCacheSchemaVersion))
    return e;
  if (llvm::Error e = requireEqualString(obj, "key", keyData.key))
    return e;
  if (llvm::Error e =
          requireEqualString(obj, "source_object_sha256", keyData.sourceSha256))
    return e;
  if (llvm::Error e = requireEqualString(obj, "source_gfx", request.SourceGfx))
    return e;
  if (llvm::Error e = requireEqualString(obj, "target_gfx", request.TargetGfx))
    return e;
  if (llvm::Error e = requireEqualString(obj, "source_isa", request.SourceIsa))
    return e;
  if (llvm::Error e = requireEqualString(obj, "target_isa", request.TargetIsa))
    return e;
  if (llvm::Error e = requireEqualString(obj, "code_isa", request.CodeIsa))
    return e;
  if (llvm::Error e =
          requireEqualString(obj, "elf_machine", keyData.elfMachineHex))
    return e;
  if (llvm::Error e = requireEqualString(obj, "elf_flags", keyData.elfFlagsHex))
    return e;
  if (llvm::Error e = requireEqualInt(obj, "orig_mach", request.OrigMach))
    return e;
  if (llvm::Error e = requireEqualInt(obj, "opt_level",
                                      static_cast<int64_t>(request.OptLevel)))
    return e;
  if (llvm::Error e = requireEqualString(obj, "hotswap_rules_path",
                                         request.HotswapRulesPath))
    return e;
  if (llvm::Error e =
          requireEqualString(obj, "hotswap_rules_sha256", keyData.rulesSha256))
    return e;
  if (llvm::Error e = requireEqualBool(obj, "strict_mode", request.StrictMode))
    return e;
  if (llvm::Error e = requireEqualBool(obj, "enable_writelane_rewrite",
                                       request.EnableWritelaneRewrite))
    return e;
  if (llvm::Error e =
          requireEqualBool(obj, "enable_wave_native", request.EnableWaveNative))
    return e;
  if (llvm::Error e = requireEqualBool(obj, "force_scaled_modrep",
                                       request.ForceScaledModrep))
    return e;
  if (llvm::Error e = requireEqualBool(obj, "assume_hip_global_offset_zero",
                                       request.AssumeHipGlobalOffsetZero))
    return e;
  if (llvm::Error e = requireEqualString(obj, "hotswap_build_identity",
                                         keyData.buildIdentity))
    return e;
  if (llvm::Error e = requireEqualString(obj, "device_libraries_identity",
                                         keyData.deviceLibrariesIdentity))
    return e;
  if (llvm::Error e = validateKernelNameField(obj, request.KernelName))
    return e;
  if (llvm::Error e =
          requireEqualInt(obj, "kernel_count",
                          static_cast<int64_t>(keyData.kernelNames.size())))
    return e;
  if (llvm::Error e = validateKernelArray(obj, keyData.kernelNames))
    return e;
  if (llvm::Error e =
          requireEqualString(obj, "cached_object_sha256", objectSha256))
    return e;
  if (llvm::Error e = requireEqualInt(obj, "cached_object_size",
                                      static_cast<int64_t>(objectSize)))
    return e;

  llvm::Expected<int64_t> lifted = requireInt(obj, "lifted_count");
  if (!lifted)
    return lifted.takeError();
  llvm::Expected<int64_t> total = requireInt(obj, "total_count");
  if (!total)
    return total.takeError();
  llvm::Expected<int64_t> c5Count = requireInt(obj, "c5_suppressed_count");
  if (!c5Count)
    return c5Count.takeError();
  llvm::Expected<std::string> c5Reason =
      requireString(obj, "c5_suppression_reason");
  if (!c5Reason)
    return c5Reason.takeError();
  llvm::Expected<bool> usesScratch =
      requireBool(obj, "uses_scratch_private_segment");
  if (!usesScratch)
    return usesScratch.takeError();
  llvm::Expected<int64_t> sourceScratch =
      requireInt(obj, "source_private_segment_fixed_size");
  if (!sourceScratch)
    return sourceScratch.takeError();
  llvm::Expected<int64_t> targetScratch =
      requireInt(obj, "target_private_segment_fixed_size");
  if (!targetScratch)
    return targetScratch.takeError();
  llvm::Expected<bool> targetEnable =
      requireBool(obj, "target_enable_private_segment");
  if (!targetEnable)
    return targetEnable.takeError();

  Result.Success = true;
  Result.LiftedCount = static_cast<int>(*lifted);
  Result.TotalCount = static_cast<int>(*total);
  Result.C5SuppressedCount = static_cast<int>(*c5Count);
  Result.C5SuppressionReason = *c5Reason;
  Result.UsesScratchPrivateSegment = *usesScratch;
  Result.SourcePrivateSegmentFixedSize = static_cast<uint32_t>(*sourceScratch);
  Result.TargetPrivateSegmentFixedSize = static_cast<uint32_t>(*targetScratch);
  Result.TargetEnablePrivateSegment = *targetEnable;
  // Optional (default 1) so cache entries written before scaled-dispatch
  // support still load. The cache key includes the scaled-modrep flags, so a
  // scaled transpile never collides with a non-scaled entry.
  Result.ScaledDispatchFactor = static_cast<unsigned>(
      obj.getInteger("scaled_dispatch_factor").value_or(1));
  return llvm::Error::success();
}

} // namespace

const char *translationCacheStatusString(TranslationCacheStatus Status) {
  switch (Status) {
  case TranslationCacheStatus::Disabled:
    return "disabled";
  case TranslationCacheStatus::Bypassed:
    return "bypassed";
  case TranslationCacheStatus::Miss:
    return "miss";
  case TranslationCacheStatus::Hit:
    return "hit";
  case TranslationCacheStatus::Invalid:
    return "invalid";
  case TranslationCacheStatus::WriteSuccess:
    return "write_success";
  case TranslationCacheStatus::WriteFailed:
    return "write_failed";
  }
  return "invalid";
}

std::string sha256Hex(llvm::MemoryBufferRef buffer) {
  llvm::ArrayRef data(buffer.getBuffer().bytes_begin(),
                      buffer.getBuffer().bytes_end());
  auto digest = llvm::SHA256::hash(data);
  std::string out;
  llvm::raw_string_ostream os(out);
  for (uint8_t byte : digest)
    os << llvm::format_hex_no_prefix(byte, 2);
  return os.str();
}

std::string translationCacheKey(const TranslationCacheRequest &request) {
  TranslationCacheKeyBuildTimings timings;
  llvm::Expected<KeyData> keyData =
      buildKeyData(request, /*CollectTimings=*/false, timings);
  if (!keyData) {
    // Derivation failed (empty source, no kernels, unreadable rules, ...).
    // Swallow the error and report "uncacheable" via an empty key; the
    // producer still runs, we just don't cache. Must not throw (this TU is
    // built -fno-exceptions).
    llvm::consumeError(keyData.takeError());
    return std::string();
  }
  return std::move(keyData->key);
}

TranslationCacheLookup
lookupTranslationCache(const TranslationCacheRequest &request) {
  auto totalStart = timingStart(request.CollectTimings);
  TranslationCacheLookup lookup;
  auto finish = [&]() {
    lookup.Timings.totalSeconds =
        timingElapsed(request.CollectTimings, totalStart);
    return std::move(lookup);
  };
  if (cacheDisabledByPolicy(request))
    return finish();

  auto keyBuildStart = timingStart(request.CollectTimings);
  auto keyDataOrErr =
      buildKeyData(request, request.CollectTimings, lookup.Timings.keyBuild);
  lookup.Timings.keyBuildSeconds =
      timingElapsed(request.CollectTimings, keyBuildStart);
  if (!keyDataOrErr) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = llvm::toString(keyDataOrErr.takeError());
    return finish();
  }
  KeyData keyData = std::move(*keyDataOrErr);
  lookup.key = keyData.key;
  lookup.MetadataPath = cacheMetadataPath(request, keyData.key);
  lookup.ObjectPath = cacheObjectPath(request, keyData.key);

  auto metadataObjectStatStart = timingStart(request.CollectTimings);
  const bool metadataExists = exists(lookup.MetadataPath);
  const bool objectExists = exists(lookup.ObjectPath);
  lookup.Timings.metadataObjectStatSeconds =
      timingElapsed(request.CollectTimings, metadataObjectStatStart);
  if (!metadataExists && !objectExists) {
    lookup.Status = TranslationCacheStatus::Miss;
    lookup.Reason = "entry not present";
    return finish();
  }
  if (metadataExists != objectExists) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = metadataExists ? "metadata exists without object"
                                   : "object exists without metadata";
    return finish();
  }

  auto objectReadStart = timingStart(request.CollectTimings);
  auto objectBuffer = llvm::MemoryBuffer::getFile(lookup.ObjectPath);
  lookup.Timings.objectReadSeconds =
      timingElapsed(request.CollectTimings, objectReadStart);
  if (!objectBuffer) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason =
        "failed to read cached object: " + objectBuffer.getError().message();
    return finish();
  }
  auto objectHashStart = timingStart(request.CollectTimings);
  llvm::MemoryBufferRef objectBufRef = (*objectBuffer)->getMemBufferRef();
  std::string objectSha = sha256Hex(objectBufRef);
  lookup.Timings.objectHashSeconds =
      timingElapsed(request.CollectTimings, objectHashStart);

  auto metadataReadStart = timingStart(request.CollectTimings);
  auto metadataBuffer = llvm::MemoryBuffer::getFile(lookup.MetadataPath);
  lookup.Timings.metadataReadSeconds =
      timingElapsed(request.CollectTimings, metadataReadStart);
  if (!metadataBuffer) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason =
        "failed to read cache metadata: " + metadataBuffer.getError().message();
    return finish();
  }
  auto metadataParseStart = timingStart(request.CollectTimings);
  auto parsed = llvm::json::parse((*metadataBuffer)->getBuffer());
  lookup.Timings.metadataParseSeconds =
      timingElapsed(request.CollectTimings, metadataParseStart);
  if (!parsed) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason =
        "failed to parse cache metadata: " + llvm::toString(parsed.takeError());
    return finish();
  }
  const llvm::json::Object *obj = parsed->getAsObject();
  if (!obj) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = "cache metadata is not a JSON object";
    return finish();
  }
  auto metadataValidateStart = timingStart(request.CollectTimings);
  llvm::Error validateErr =
      validateMetadata(request, keyData, *obj, objectSha,
                       objectBufRef.getBufferSize(), lookup.Result);
  lookup.Timings.metadataValidateSeconds =
      timingElapsed(request.CollectTimings, metadataValidateStart);
  if (validateErr) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = llvm::toString(std::move(validateErr));
    return finish();
  }

  lookup.Result.Hsaco = std::move(*objectBuffer);
  lookup.Status = TranslationCacheStatus::Hit;
  lookup.Reason = "ok";
  return finish();
}

TranslationCacheWrite
writeTranslationCache(const TranslationCacheRequest &request,
                      const PipelineResult &Result) {
  auto totalStart = timingStart(request.CollectTimings);
  TranslationCacheWrite write;
  auto finish = [&]() {
    write.Timings.totalSeconds =
        timingElapsed(request.CollectTimings, totalStart);
    return write;
  };
  if (cacheDisabledByPolicy(request) || request.CacheReadonly)
    return finish();

  auto keyBuildStart = timingStart(request.CollectTimings);
  auto keyDataOrErr =
      buildKeyData(request, request.CollectTimings, write.Timings.keyBuild);
  write.Timings.keyBuildSeconds =
      timingElapsed(request.CollectTimings, keyBuildStart);
  if (!keyDataOrErr) {
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = llvm::toString(keyDataOrErr.takeError());
    return finish();
  }
  KeyData keyData = std::move(*keyDataOrErr);
  write.key = keyData.key;
  write.MetadataPath = cacheMetadataPath(request, keyData.key);
  write.ObjectPath = cacheObjectPath(request, keyData.key);

  if (!Result.Success || !Result.Hsaco || Result.Hsaco->getBufferSize() == 0) {
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = "refusing to cache unsuccessful or empty translation";
    return finish();
  }

  std::string dir = cacheSubdir(request, keyData.key);
  auto createDirectoryStart = timingStart(request.CollectTimings);
  if (auto ec = llvm::sys::fs::create_directories(dir)) {
    write.Timings.createDirectorySeconds =
        timingElapsed(request.CollectTimings, createDirectoryStart);
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason =
        "failed to create cache directory '" + dir + "': " + ec.message();
    return finish();
  }
  write.Timings.createDirectorySeconds =
      timingElapsed(request.CollectTimings, createDirectoryStart);

  auto objectHashStart = timingStart(request.CollectTimings);
  std::string objectSha = sha256Hex(Result.Hsaco->getMemBufferRef());
  write.Timings.objectHashSeconds =
      timingElapsed(request.CollectTimings, objectHashStart);
  auto objectWriteStart = timingStart(request.CollectTimings);
  if (auto err = writeFileAtomic(write.ObjectPath, Result.Hsaco->getBuffer())) {
    write.Timings.objectWriteSeconds =
        timingElapsed(request.CollectTimings, objectWriteStart);
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason =
        "failed to write cached object: " + llvm::toString(std::move(err));
    return finish();
  }
  write.Timings.objectWriteSeconds =
      timingElapsed(request.CollectTimings, objectWriteStart);

  auto metadataBuildStart = timingStart(request.CollectTimings);
  llvm::json::Object meta = metadataObject(request, keyData, Result, objectSha);
  write.Timings.metadataBuildSeconds =
      timingElapsed(request.CollectTimings, metadataBuildStart);
  auto metadataWriteStart = timingStart(request.CollectTimings);
  if (auto err =
          writeFileAtomic(write.MetadataPath,
                          jsonToString(llvm::json::Value(std::move(meta))))) {
    write.Timings.metadataWriteSeconds =
        timingElapsed(request.CollectTimings, metadataWriteStart);
    llvm::sys::fs::remove(write.ObjectPath);
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason =
        "failed to write cache metadata: " + llvm::toString(std::move(err));
    return finish();
  }
  write.Timings.metadataWriteSeconds =
      timingElapsed(request.CollectTimings, metadataWriteStart);

  write.Status = TranslationCacheStatus::WriteSuccess;
  write.Reason = "ok";
  return finish();
}

std::string
skippedKernelForTranslationCache(llvm::ArrayRef<std::string> kernelNames,
                                 llvm::StringRef skipList) {
  if (skipList.empty())
    return "";

  llvm::StringRef remaining(skipList);
  while (!remaining.empty()) {
    auto split = remaining.split(',');
    llvm::StringRef requested = split.first.trim();
    remaining = split.second;
    if (requested.empty())
      continue;
    for (llvm::StringRef kernelName : kernelNames) {
      if (requested == kernelName)
        return kernelName.str();
    }
  }
  return "";
}

} // namespace COMGR::hotswap
