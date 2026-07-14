#include "translation-cache.h"

#include "code-object-utils.h"
#include "comgr-device-libs.h"

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

#include <dlfcn.h>
#include <chrono>
#include <optional>
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
  std::string error;
};

struct KeyData {
  std::string key;
  std::string sourceSha256;
  std::string rulesSha256;
  std::string rulesError;
  std::string buildIdentity;
  std::string deviceLibrariesIdentity;
  std::string elfMachineHex;
  std::string elfFlagsHex;
  std::vector<std::string> kernelNames;
  std::string error;
  TranslationCacheKeyBuildTimings Timings;
};

std::string hexU32(uint32_t value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "0x" << llvm::format_hex_no_prefix(value, 0);
  return os.str();
}

bool readElfHeaderFields(llvm::MemoryBufferRef buffer, uint16_t &machine,
                         uint32_t &flags) {
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(buffer);
  if (!objOrErr) {
    llvm::consumeError(objOrErr.takeError());
    return false;
  }
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(objOrErr->get());
  if (!elf)
    return false;
  machine = elf->getEMachine();
  flags = elf->getPlatformFlags();
  return true;
}

std::string hashFile(llvm::StringRef path, std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = buffer.getError().message();
    return "";
  }
  return sha256Hex((*buffer)->getMemBufferRef());
}

FileIdentity statIdentity(llvm::StringRef path) {
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
  id.sha256 = hashFile(id.path, id.error);
  return id;
}

std::string identityString(const FileIdentity &id) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << id.path << "|present=" << (id.present ? "1" : "0")
     << "|size=" << id.size << "|mtime=" << id.mtimeSec << "."
     << id.mtimeNsec << "|sha256=" << id.sha256;
  if (!id.error.empty())
    os << "|error=" << id.error;
  return os.str();
}

std::string loadedImageIdentity() {
  static const std::string identity = [] {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << "llvm=" << LLVM_VERSION_STRING;
    Dl_info info;
    if (::dladdr(reinterpret_cast<void *>(&loadedImageIdentity), &info) &&
        info.dli_fname) {
      os << "|image=" << identityString(statIdentity(info.dli_fname));
    } else {
      os << "|image=<dladdr-unavailable>";
    }
    return os.str();
  }();
  return identity;
}

std::string hexBytes(llvm::ArrayRef<unsigned char> Bytes) {
  std::string Out;
  llvm::raw_string_ostream Os(Out);
  for (unsigned char Byte : Bytes)
    Os << llvm::format_hex_no_prefix(static_cast<uint8_t>(Byte), 2);
  return Os.str();
}

std::string deviceLibrariesIdentity() {
  return hexBytes(COMGR::getDeviceLibrariesIdentifier());
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

KeyData buildKeyData(const TranslationCacheRequest &request,
                     bool CollectTimings) {
  KeyData data;
  if (request.SourceObject.getBufferSize() == 0) {
    data.error = "empty source code object";
    return data;
  }
  if (request.SourceGfx.empty() || request.TargetGfx.empty()) {
    data.error = "missing source or target gfx";
    return data;
  }

  auto sourceHashStart = timingStart(CollectTimings);
  data.sourceSha256 = sha256Hex(request.SourceObject);
  data.Timings.sourceHashSeconds =
      timingElapsed(CollectTimings, sourceHashStart);
  uint16_t machine = 0;
  uint32_t flags = 0;
  auto elfHeaderStart = timingStart(CollectTimings);
  if (readElfHeaderFields(request.SourceObject, machine, flags)) {
    data.Timings.elfHeaderSeconds =
        timingElapsed(CollectTimings, elfHeaderStart);
    data.elfMachineHex = hexU32(machine);
    data.elfFlagsHex = hexU32(flags);
  } else {
    data.Timings.elfHeaderSeconds =
        timingElapsed(CollectTimings, elfHeaderStart);
    data.error = "source code object is not a 64-bit little-endian ELF";
    return data;
  }

  if (!request.HotswapRulesPath.empty()) {
    auto rulesHashStart = timingStart(CollectTimings);
    data.rulesSha256 = hashFile(request.HotswapRulesPath, data.rulesError);
    data.Timings.rulesHashSeconds =
        timingElapsed(CollectTimings, rulesHashStart);
    if (!data.rulesError.empty()) {
      data.error = "failed to hash HSA_HOTSWAP_RULES '" +
                   request.HotswapRulesPath + "': " + data.rulesError;
      return data;
    }
  }

  auto loadedImageIdentityStart = timingStart(CollectTimings);
  data.buildIdentity = loadedImageIdentity();
  data.Timings.loadedImageIdentitySeconds =
      timingElapsed(CollectTimings, loadedImageIdentityStart);
  data.deviceLibrariesIdentity = deviceLibrariesIdentity();
  if (!request.KernelName.empty()) {
    data.kernelNames.push_back(request.KernelName);
  } else {
    auto kernelNamesStart = timingStart(CollectTimings);
    llvm::Expected<llvm::SmallVector<std::string>> NamesOrErr =
        listKernelNames(request.SourceObject);
    data.Timings.kernelNamesSeconds =
        timingElapsed(CollectTimings, kernelNamesStart);
    if (!NamesOrErr) {
      data.error = "failed to list kernels for translation cache key: " +
                   llvm::toString(NamesOrErr.takeError());
      return data;
    }
    data.kernelNames.assign(NamesOrErr->begin(), NamesOrErr->end());
    if (data.kernelNames.empty()) {
      data.error = "source code object has no kernel metadata entries";
      return data;
    }
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
  appendKeyField(material, "assume_hip_global_offset_zero",
                 request.AssumeHipGlobalOffsetZero);
  if (!request.KernelName.empty())
    appendKeyField(material, "kernel_name", request.KernelName);
  appendKeyField(material, "hotswap_build_identity", data.buildIdentity);
  appendKeyField(material, "device_libraries_identity",
                 data.deviceLibrariesIdentity);
  data.Timings.materialBuildSeconds =
      timingElapsed(CollectTimings, materialBuildStart);
  auto keyHashStart = timingStart(CollectTimings);
  data.key = sha256Hex(llvm::MemoryBufferRef(material, ""));
  data.Timings.keyHashSeconds =
      timingElapsed(CollectTimings, keyHashStart);
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

bool exists(llvm::StringRef path) {
  return llvm::sys::fs::exists(path);
}

std::string jsonToString(llvm::json::Value value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  value.print(os);
  os << "\n";
  return os.str();
}

bool writeFileAtomic(llvm::StringRef path, llvm::StringRef contents,
                     std::string &error) {
  llvm::SmallString<256> model(path);
  model += ".tmp-%%%%%%";
  llvm::SmallString<256> tmpPath;
  int fd = -1;
  if (auto ec = llvm::sys::fs::createUniqueFile(model, fd, tmpPath)) {
    error = ec.message();
    return false;
  }
  {
    llvm::raw_fd_ostream os(fd, true);
    os << contents;
    if (os.has_error()) {
      error = os.error().message();
      os.clear_error();
      llvm::sys::fs::remove(tmpPath);
      return false;
    }
  }
  if (auto ec = llvm::sys::fs::rename(tmpPath, path)) {
    error = ec.message();
    llvm::sys::fs::remove(tmpPath);
    return false;
  }
  return true;
}

bool writeFileAtomic(llvm::StringRef path, llvm::ArrayRef<uint8_t> data,
                     std::string &error) {
  return writeFileAtomic(
      path, llvm::StringRef(reinterpret_cast<const char *>(data.data()),
                            data.size()),
      error);
}

std::optional<std::string> requireString(const llvm::json::Object &obj,
                                         llvm::StringRef field,
                                         std::string &Reason) {
  auto value = obj.getString(field);
  if (!value) {
    Reason = "metadata field '" + field.str() + "' missing or not a string";
    return std::nullopt;
  }
  return value->str();
}

std::optional<int64_t> requireInt(const llvm::json::Object &obj,
                                  llvm::StringRef field, std::string &Reason) {
  auto value = obj.getInteger(field);
  if (!value) {
    Reason = "metadata field '" + field.str() + "' missing or not an integer";
    return std::nullopt;
  }
  return *value;
}

std::optional<bool> requireBool(const llvm::json::Object &obj,
                                llvm::StringRef field, std::string &Reason) {
  auto value = obj.getBoolean(field);
  if (!value) {
    Reason = "metadata field '" + field.str() + "' missing or not a boolean";
    return std::nullopt;
  }
  return *value;
}

bool requireEqualString(const llvm::json::Object &obj, llvm::StringRef field,
                        llvm::StringRef expected, std::string &Reason) {
  auto value = requireString(obj, field, Reason);
  if (!value)
    return false;
  if (*value != expected) {
    Reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
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

bool requireEqualInt(const llvm::json::Object &obj, llvm::StringRef field,
                     int64_t expected, std::string &Reason) {
  auto value = requireInt(obj, field, Reason);
  if (!value)
    return false;
  if (*value != expected) {
    Reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

bool requireEqualBool(const llvm::json::Object &obj, llvm::StringRef field,
                      bool expected, std::string &Reason) {
  auto value = requireBool(obj, field, Reason);
  if (!value)
    return false;
  if (*value != expected) {
    Reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

llvm::json::Array kernelArray(const std::vector<std::string> &kernelNames) {
  llvm::json::Array arr;
  for (llvm::StringRef name : kernelNames)
    arr.push_back(name);
  return arr;
}

bool validateKernelArray(const llvm::json::Object &obj,
                         const std::vector<std::string> &expected,
                         std::string &Reason) {
  const llvm::json::Array *arr = obj.getArray("kernel_names");
  if (!arr) {
    Reason = "metadata field 'kernel_names' missing or not an array";
    return false;
  }
  if (arr->size() != expected.size()) {
    Reason = "metadata kernel_names size mismatch";
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    auto value = (*arr)[i].getAsString();
    if (!value || *value != expected[i]) {
      Reason = "metadata kernel_names mismatch";
      return false;
    }
  }
  return true;
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

bool validateMetadata(const TranslationCacheRequest &request,
                      const KeyData &keyData, const llvm::json::Object &obj,
                      llvm::StringRef objectSha256, size_t objectSize,
                      PipelineResult &Result, std::string &Reason) {
  if (!requireEqualInt(obj, "schema_version", kCacheSchemaVersion, Reason) ||
      !requireEqualString(obj, "key", keyData.key, Reason) ||
      !requireEqualString(obj, "source_object_sha256", keyData.sourceSha256,
                          Reason) ||
      !requireEqualString(obj, "source_gfx", request.SourceGfx, Reason) ||
      !requireEqualString(obj, "target_gfx", request.TargetGfx, Reason) ||
      !requireEqualString(obj, "source_isa", request.SourceIsa, Reason) ||
      !requireEqualString(obj, "target_isa", request.TargetIsa, Reason) ||
      !requireEqualString(obj, "code_isa", request.CodeIsa, Reason) ||
      !requireEqualString(obj, "elf_machine", keyData.elfMachineHex, Reason) ||
      !requireEqualString(obj, "elf_flags", keyData.elfFlagsHex, Reason) ||
      !requireEqualInt(obj, "orig_mach", request.OrigMach, Reason) ||
      !requireEqualInt(obj, "opt_level", static_cast<int64_t>(request.OptLevel),
                       Reason) ||
      !requireEqualString(obj, "hotswap_rules_path", request.HotswapRulesPath,
                          Reason) ||
      !requireEqualString(obj, "hotswap_rules_sha256", keyData.rulesSha256,
                          Reason) ||
      !requireEqualBool(obj, "strict_mode", request.StrictMode, Reason) ||
      !requireEqualBool(obj, "enable_writelane_rewrite",
                        request.EnableWritelaneRewrite, Reason) ||
      !requireEqualBool(obj, "enable_wave_native", request.EnableWaveNative,
                        Reason) ||
      !requireEqualBool(obj, "assume_hip_global_offset_zero",
                        request.AssumeHipGlobalOffsetZero, Reason) ||
      !requireEqualString(obj, "hotswap_build_identity",
                          keyData.buildIdentity, Reason) ||
      !requireEqualString(obj, "device_libraries_identity",
                          keyData.deviceLibrariesIdentity, Reason))
    return false;
  if (llvm::Error Err = validateKernelNameField(obj, request.KernelName)) {
    Reason = llvm::toString(std::move(Err));
    return false;
  }
  if (!requireEqualInt(obj, "kernel_count",
                       static_cast<int64_t>(keyData.kernelNames.size()),
                       Reason) ||
      !validateKernelArray(obj, keyData.kernelNames, Reason) ||
      !requireEqualString(obj, "cached_object_sha256", objectSha256, Reason) ||
      !requireEqualInt(obj, "cached_object_size",
                       static_cast<int64_t>(objectSize), Reason))
    return false;

  auto lifted = requireInt(obj, "lifted_count", Reason);
  auto total = requireInt(obj, "total_count", Reason);
  auto c5Count = requireInt(obj, "c5_suppressed_count", Reason);
  auto c5Reason = requireString(obj, "c5_suppression_reason", Reason);
  auto usesScratch = requireBool(obj, "uses_scratch_private_segment", Reason);
  auto sourceScratch =
      requireInt(obj, "source_private_segment_fixed_size", Reason);
  auto targetScratch =
      requireInt(obj, "target_private_segment_fixed_size", Reason);
  auto targetEnable =
      requireBool(obj, "target_enable_private_segment", Reason);
  if (!lifted || !total || !c5Count || !c5Reason || !usesScratch ||
      !sourceScratch || !targetScratch || !targetEnable)
    return false;

  Result.Success = true;
  Result.LiftedCount = static_cast<int>(*lifted);
  Result.TotalCount = static_cast<int>(*total);
  Result.C5SuppressedCount = static_cast<int>(*c5Count);
  Result.C5SuppressionReason = *c5Reason;
  Result.UsesScratchPrivateSegment = *usesScratch;
  Result.SourcePrivateSegmentFixedSize =
      static_cast<uint32_t>(*sourceScratch);
  Result.TargetPrivateSegmentFixedSize =
      static_cast<uint32_t>(*targetScratch);
  Result.TargetEnablePrivateSegment = *targetEnable;
  return true;
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
  llvm::ArrayRef data(buffer.getBuffer().bytes_begin(), buffer.getBuffer().bytes_end());
  auto digest = llvm::SHA256::hash(data);
  std::string out;
  llvm::raw_string_ostream os(out);
  for (uint8_t byte : digest)
    os << llvm::format_hex_no_prefix(byte, 2);
  return os.str();
}

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &request) {
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
  KeyData keyData = buildKeyData(request, request.CollectTimings);
  lookup.Timings.keyBuildSeconds =
      timingElapsed(request.CollectTimings, keyBuildStart);
  lookup.Timings.keyBuild = keyData.Timings;
  lookup.key = keyData.key;
  if (!keyData.error.empty()) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = keyData.error;
    return finish();
  }
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
    lookup.Reason = "failed to read cached object: " +
                    objectBuffer.getError().message();
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
    lookup.Reason = "failed to read cache metadata: " +
                    metadataBuffer.getError().message();
    return finish();
  }
  auto metadataParseStart = timingStart(request.CollectTimings);
  auto parsed = llvm::json::parse((*metadataBuffer)->getBuffer());
  lookup.Timings.metadataParseSeconds =
      timingElapsed(request.CollectTimings, metadataParseStart);
  if (!parsed) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = "failed to parse cache metadata: " +
                    llvm::toString(parsed.takeError());
    return finish();
  }
  const llvm::json::Object *obj = parsed->getAsObject();
  if (!obj) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = "cache metadata is not a JSON object";
    return finish();
  }
  auto metadataValidateStart = timingStart(request.CollectTimings);
  if (!validateMetadata(request, keyData, *obj, objectSha,
                        objectBufRef.getBufferSize(), lookup.Result,
                        lookup.Reason)) {
    lookup.Timings.metadataValidateSeconds =
        timingElapsed(request.CollectTimings, metadataValidateStart);
    lookup.Status = TranslationCacheStatus::Invalid;
    return finish();
  }
  lookup.Timings.metadataValidateSeconds =
      timingElapsed(request.CollectTimings, metadataValidateStart);

  lookup.Result.Hsaco = std::move(*objectBuffer);
  lookup.Status = TranslationCacheStatus::Hit;
  lookup.Reason = "ok";
  return finish();
}

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &request, const PipelineResult &Result) {
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
  KeyData keyData = buildKeyData(request, request.CollectTimings);
  write.Timings.keyBuildSeconds =
      timingElapsed(request.CollectTimings, keyBuildStart);
  write.Timings.keyBuild = keyData.Timings;
  write.key = keyData.key;
  if (!keyData.error.empty()) {
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = keyData.error;
    return finish();
  }
  write.MetadataPath = cacheMetadataPath(request, keyData.key);
  write.ObjectPath = cacheObjectPath(request, keyData.key);

  if (!Result.Success || !Result.Hsaco ||
      Result.Hsaco->getBufferSize() == 0) {
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
    write.Reason = "failed to create cache directory '" + dir + "': " +
                   ec.message();
    return finish();
  }
  write.Timings.createDirectorySeconds =
      timingElapsed(request.CollectTimings, createDirectoryStart);

  auto objectHashStart = timingStart(request.CollectTimings);
  std::string objectSha = sha256Hex(Result.Hsaco->getMemBufferRef());
  write.Timings.objectHashSeconds =
      timingElapsed(request.CollectTimings, objectHashStart);
  std::string error;
  auto objectWriteStart = timingStart(request.CollectTimings);
  if (!writeFileAtomic(write.ObjectPath, Result.Hsaco->getBuffer(), error)) {
    write.Timings.objectWriteSeconds =
        timingElapsed(request.CollectTimings, objectWriteStart);
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = "failed to write cached object: " + error;
    return finish();
  }
  write.Timings.objectWriteSeconds =
      timingElapsed(request.CollectTimings, objectWriteStart);

  auto metadataBuildStart = timingStart(request.CollectTimings);
  llvm::json::Object meta =
      metadataObject(request, keyData, Result, objectSha);
  write.Timings.metadataBuildSeconds =
      timingElapsed(request.CollectTimings, metadataBuildStart);
  auto metadataWriteStart = timingStart(request.CollectTimings);
  if (!writeFileAtomic(write.MetadataPath,
                       jsonToString(llvm::json::Value(std::move(meta))),
                       error)) {
    write.Timings.metadataWriteSeconds =
        timingElapsed(request.CollectTimings, metadataWriteStart);
    llvm::sys::fs::remove(write.ObjectPath);
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = "failed to write cache metadata: " + error;
    return finish();
  }
  write.Timings.metadataWriteSeconds =
      timingElapsed(request.CollectTimings, metadataWriteStart);

  write.Status = TranslationCacheStatus::WriteSuccess;
  write.Reason = "ok";
  return finish();
}

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> kernelNames, llvm::StringRef skipList) {
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
