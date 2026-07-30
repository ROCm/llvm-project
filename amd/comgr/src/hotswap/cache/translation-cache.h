#ifndef HOTSWAP_TRANSPILER_TRANSLATION_CACHE_H
#define HOTSWAP_TRANSPILER_TRANSLATION_CACHE_H

#include "pipeline.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <string>

namespace COMGR::hotswap {

struct TranslationCacheKeyBuildTimings {
  double sourceHashSeconds = 0.0;
  double elfHeaderSeconds = 0.0;
  double rulesHashSeconds = 0.0;
  double loadedImageIdentitySeconds = 0.0;
  double kernelNamesSeconds = 0.0;
  double materialBuildSeconds = 0.0;
  double keyHashSeconds = 0.0;
};

struct TranslationCacheLookupTimings {
  double totalSeconds = 0.0;
  double keyBuildSeconds = 0.0;
  TranslationCacheKeyBuildTimings keyBuild;
  double metadataObjectStatSeconds = 0.0;
  double objectReadSeconds = 0.0;
  double objectHashSeconds = 0.0;
  double metadataReadSeconds = 0.0;
  double metadataParseSeconds = 0.0;
  double metadataValidateSeconds = 0.0;
};

struct TranslationCacheWriteTimings {
  double totalSeconds = 0.0;
  double keyBuildSeconds = 0.0;
  TranslationCacheKeyBuildTimings keyBuild;
  double createDirectorySeconds = 0.0;
  double objectHashSeconds = 0.0;
  double objectWriteSeconds = 0.0;
  double metadataBuildSeconds = 0.0;
  double metadataWriteSeconds = 0.0;
};

struct TranslationCacheRequest {
  llvm::MemoryBufferRef SourceObject;
  std::string SourceGfx;
  std::string TargetGfx;
  std::string SourceIsa;
  std::string TargetIsa;
  std::string CodeIsa;
  std::string HotswapRulesPath;
  std::string CacheDirectory;
  std::string CacheSkipKernels;
  std::string KernelName;
  // Opaque identity of the bundled device libraries, salted into the cache
  // key so a device-library change invalidates prior entries. Supplied by
  // the caller (the transpiler passes COMGR::getDeviceLibrariesIdentifier())
  // to keep this module free of comgr metadata-layer coupling.
  std::string DeviceLibrariesIdentity;
  int OrigMach = -1;
  unsigned OptLevel = 0;
  bool EnableWritelaneRewrite = true;
  bool EnableWaveNative = true;
  // Unconditionally select ScaledModuloReplicationProjection for wave32->wave64
  // cross-widening (testing knob). The normal WaveNative y/z-refusal ->
  // scaled-dispatch upgrade is automatic and needs no flag.
  bool ForceScaledModrep = false;
  bool AssumeHipGlobalOffsetZero = false;
  bool StrictMode = false;
  bool CacheDisabled = true;
  bool CacheReadonly = false;
  bool CollectTimings = false;
};

enum class TranslationCacheStatus {
  Disabled,
  Bypassed,
  Miss,
  Hit,
  Invalid,
  WriteSuccess,
  WriteFailed,
};

struct TranslationCacheLookup {
  TranslationCacheStatus Status = TranslationCacheStatus::Disabled;
  std::string key;
  std::string MetadataPath;
  std::string ObjectPath;
  std::string Reason;
  TranslationCacheLookupTimings Timings;
  PipelineResult Result;
};

struct TranslationCacheWrite {
  TranslationCacheStatus Status = TranslationCacheStatus::Disabled;
  std::string key;
  std::string MetadataPath;
  std::string ObjectPath;
  std::string Reason;
  TranslationCacheWriteTimings Timings;
};

const char *translationCacheStatusString(TranslationCacheStatus Status);

TranslationCacheLookup
lookupTranslationCache(const TranslationCacheRequest &request);

TranslationCacheWrite
writeTranslationCache(const TranslationCacheRequest &request,
                      const PipelineResult &Result);

std::string
skippedKernelForTranslationCache(llvm::ArrayRef<std::string> kernelNames,
                                 llvm::StringRef skipList);

std::string sha256Hex(llvm::MemoryBufferRef buffer);

// Returns the content-addressed cache key for `request` -- the same SHA-256
// hex identity the disk tier stores under. Empty string if the key cannot be
// derived (e.g. empty source object, missing gfx, no kernel metadata, or an
// unreadable rules file); callers treat an empty key as "uncacheable" and
// bypass caching for that request. Shared by the in-memory tier so both tiers
// agree on identity by construction.
std::string translationCacheKey(const TranslationCacheRequest &request);

} // namespace COMGR::hotswap

#endif
