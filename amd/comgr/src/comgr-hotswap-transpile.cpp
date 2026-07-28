//===- comgr-hotswap-transpile.cpp - ISA transpilation via LLVM IR --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// COMgr entry point for the hotswap transpiler. Where the byte-level
/// `amd_comgr_hotswap_rewrite` path patches a small set of stepping-specific
/// instruction encodings in place, this entry point hands the whole code
/// object to the hotswap pipeline - every kernel is disassembled, raised to
/// LLVM IR, re-lowered through the stock AMDGPU backend for the target ISA,
/// and re-linked into a single merged HSACO via
/// `COMGR::hotswap::runPipelineAllKernels` (see amd/comgr/hotswap/pipeline.hpp and
/// amd/comgr/hotswap/raise_cli.cpp for the standalone driver this entry point
/// mirrors).
///
/// Failure is loud: any per-kernel raise failure surfaced by the hotswap
/// pipeline turns into `AMD_COMGR_STATUS_ERROR`. The hotswap library logs
/// the offending kernel and mnemonic on stderr (use hotswap's CLI with the
/// `--write-Hsaco` mode for the same output).
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"

#include "hotswap/code-object-utils.h"
#include "hotswap/pipeline.h"
#include "hotswap/translation-cache.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

using namespace COMGR;

namespace {

using TimingClock = std::chrono::steady_clock;

constexpr unsigned DefaultHotswapComgrOptLevel = 2;
constexpr unsigned MaxHotswapComgrOptLevel = 3;

double secondsBetween(TimingClock::time_point start, TimingClock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

TimingClock::time_point timingStart(bool CollectTimings) {
  return CollectTimings ? TimingClock::now() : TimingClock::time_point{};
}

double timingElapsed(bool CollectTimings, TimingClock::time_point start) {
  return CollectTimings ? secondsBetween(start, TimingClock::now()) : 0.0;
}

bool HotSwapTimingEnabled() {
  const char *value = std::getenv("HSA_HOTSWAP_TIMING");
  return value && value[0] && std::strcmp(value, "0") != 0;
}

struct HotswapComgrTimings {
  double totalSeconds = 0.0;
  double inputCopySeconds = 0.0;
  double listKernelsSeconds = 0.0;
  double cacheLookupTotalSeconds = 0.0;
  double cacheLookupKeyBuildSeconds = 0.0;
  double cacheLookupKeySourceHashSeconds = 0.0;
  double cacheLookupKeyElfHeaderSeconds = 0.0;
  double cacheLookupKeyRulesHashSeconds = 0.0;
  double cacheLookupKeyLoadedImageIdentitySeconds = 0.0;
  double cacheLookupKeyKernelNamesSeconds = 0.0;
  double cacheLookupKeyMaterialBuildSeconds = 0.0;
  double cacheLookupKeyHashSeconds = 0.0;
  double cacheLookupStatSeconds = 0.0;
  double cacheLookupObjectReadSeconds = 0.0;
  double cacheLookupObjectHashSeconds = 0.0;
  double cacheLookupMetadataReadSeconds = 0.0;
  double cacheLookupMetadataParseSeconds = 0.0;
  double cacheLookupMetadataValidateSeconds = 0.0;
  double pipelineTotalSeconds = 0.0;
  double pipelineListKernelsSeconds = 0.0;
  double pipelineExtractTextSeconds = 0.0;
  double pipelineCreateTempDirSeconds = 0.0;
  double pipelineRaiseSeconds = 0.0;
  double pipelineWriteIrSeconds = 0.0;
  double pipelineOptSeconds = 0.0;
  double pipelineLlcSeconds = 0.0;
  double pipelineLinkSeconds = 0.0;
  double pipelineReadHsacoSeconds = 0.0;
  double pipelineCollectMetadataSeconds = 0.0;
  double cacheWriteTotalSeconds = 0.0;
  double cacheWriteKeyBuildSeconds = 0.0;
  double cacheWriteKeySourceHashSeconds = 0.0;
  double cacheWriteKeyElfHeaderSeconds = 0.0;
  double cacheWriteKeyRulesHashSeconds = 0.0;
  double cacheWriteKeyLoadedImageIdentitySeconds = 0.0;
  double cacheWriteKeyKernelNamesSeconds = 0.0;
  double cacheWriteKeyMaterialBuildSeconds = 0.0;
  double cacheWriteKeyHashSeconds = 0.0;
  double cacheWriteCreateDirectorySeconds = 0.0;
  double cacheWriteObjectHashSeconds = 0.0;
  double cacheWriteObjectWriteSeconds = 0.0;
  double cacheWriteMetadataBuildSeconds = 0.0;
  double cacheWriteMetadataWriteSeconds = 0.0;
  double createOutputDataSeconds = 0.0;
};

std::string timingJson(const HotswapComgrTimings &Timings) {
  llvm::json::Object object{
      {"total_seconds", Timings.totalSeconds},
      {"input_copy_seconds", Timings.inputCopySeconds},
      {"list_kernels_seconds", Timings.listKernelsSeconds},
      {"cache_lookup_total_seconds", Timings.cacheLookupTotalSeconds},
      {"cache_lookup_key_build_seconds", Timings.cacheLookupKeyBuildSeconds},
      {"cache_lookup_key_source_hash_seconds",
       Timings.cacheLookupKeySourceHashSeconds},
      {"cache_lookup_key_elf_header_seconds",
       Timings.cacheLookupKeyElfHeaderSeconds},
      {"cache_lookup_key_rules_hash_seconds",
       Timings.cacheLookupKeyRulesHashSeconds},
      {"cache_lookup_key_loaded_image_identity_seconds",
       Timings.cacheLookupKeyLoadedImageIdentitySeconds},
      {"cache_lookup_key_kernel_names_seconds",
       Timings.cacheLookupKeyKernelNamesSeconds},
      {"cache_lookup_key_material_build_seconds",
       Timings.cacheLookupKeyMaterialBuildSeconds},
      {"cache_lookup_key_hash_seconds", Timings.cacheLookupKeyHashSeconds},
      {"cache_lookup_stat_seconds", Timings.cacheLookupStatSeconds},
      {"cache_lookup_object_read_seconds", Timings.cacheLookupObjectReadSeconds},
      {"cache_lookup_object_hash_seconds", Timings.cacheLookupObjectHashSeconds},
      {"cache_lookup_metadata_read_seconds",
       Timings.cacheLookupMetadataReadSeconds},
      {"cache_lookup_metadata_parse_seconds",
       Timings.cacheLookupMetadataParseSeconds},
      {"cache_lookup_metadata_validate_seconds",
       Timings.cacheLookupMetadataValidateSeconds},
      {"pipeline_total_seconds", Timings.pipelineTotalSeconds},
      {"pipeline_list_kernels_seconds", Timings.pipelineListKernelsSeconds},
      {"pipeline_extract_text_seconds", Timings.pipelineExtractTextSeconds},
      {"pipeline_create_temp_dir_seconds", Timings.pipelineCreateTempDirSeconds},
      {"pipeline_raise_seconds", Timings.pipelineRaiseSeconds},
      {"pipeline_write_ir_seconds", Timings.pipelineWriteIrSeconds},
      {"pipeline_opt_seconds", Timings.pipelineOptSeconds},
      {"pipeline_llc_seconds", Timings.pipelineLlcSeconds},
      {"pipeline_link_seconds", Timings.pipelineLinkSeconds},
      {"pipeline_read_hsaco_seconds", Timings.pipelineReadHsacoSeconds},
      {"pipeline_collect_metadata_seconds",
       Timings.pipelineCollectMetadataSeconds},
      {"cache_write_total_seconds", Timings.cacheWriteTotalSeconds},
      {"cache_write_key_build_seconds", Timings.cacheWriteKeyBuildSeconds},
      {"cache_write_key_source_hash_seconds",
       Timings.cacheWriteKeySourceHashSeconds},
      {"cache_write_key_elf_header_seconds",
       Timings.cacheWriteKeyElfHeaderSeconds},
      {"cache_write_key_rules_hash_seconds",
       Timings.cacheWriteKeyRulesHashSeconds},
      {"cache_write_key_loaded_image_identity_seconds",
       Timings.cacheWriteKeyLoadedImageIdentitySeconds},
      {"cache_write_key_kernel_names_seconds",
       Timings.cacheWriteKeyKernelNamesSeconds},
      {"cache_write_key_material_build_seconds",
       Timings.cacheWriteKeyMaterialBuildSeconds},
      {"cache_write_key_hash_seconds", Timings.cacheWriteKeyHashSeconds},
      {"cache_write_create_directory_seconds",
       Timings.cacheWriteCreateDirectorySeconds},
      {"cache_write_object_hash_seconds", Timings.cacheWriteObjectHashSeconds},
      {"cache_write_object_write_seconds", Timings.cacheWriteObjectWriteSeconds},
      {"cache_write_metadata_build_seconds",
       Timings.cacheWriteMetadataBuildSeconds},
      {"cache_write_metadata_write_seconds",
       Timings.cacheWriteMetadataWriteSeconds},
      {"create_output_data_seconds", Timings.createOutputDataSeconds},
  };
  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::json::Value(std::move(object)).print(os);
  return out;
}

void addLookupTimings(HotswapComgrTimings &Timings,
                      const COMGR::hotswap::TranslationCacheLookupTimings &lookup) {
  Timings.cacheLookupTotalSeconds += lookup.totalSeconds;
  Timings.cacheLookupKeyBuildSeconds += lookup.keyBuildSeconds;
  Timings.cacheLookupKeySourceHashSeconds += lookup.keyBuild.sourceHashSeconds;
  Timings.cacheLookupKeyElfHeaderSeconds += lookup.keyBuild.elfHeaderSeconds;
  Timings.cacheLookupKeyRulesHashSeconds += lookup.keyBuild.rulesHashSeconds;
  Timings.cacheLookupKeyLoadedImageIdentitySeconds +=
      lookup.keyBuild.loadedImageIdentitySeconds;
  Timings.cacheLookupKeyKernelNamesSeconds +=
      lookup.keyBuild.kernelNamesSeconds;
  Timings.cacheLookupKeyMaterialBuildSeconds +=
      lookup.keyBuild.materialBuildSeconds;
  Timings.cacheLookupKeyHashSeconds += lookup.keyBuild.keyHashSeconds;
  Timings.cacheLookupStatSeconds += lookup.metadataObjectStatSeconds;
  Timings.cacheLookupObjectReadSeconds += lookup.objectReadSeconds;
  Timings.cacheLookupObjectHashSeconds += lookup.objectHashSeconds;
  Timings.cacheLookupMetadataReadSeconds += lookup.metadataReadSeconds;
  Timings.cacheLookupMetadataParseSeconds += lookup.metadataParseSeconds;
  Timings.cacheLookupMetadataValidateSeconds += lookup.metadataValidateSeconds;
}

void addPipelineTimings(HotswapComgrTimings &Timings,
                        const COMGR::hotswap::PipelineTimings &pipeline) {
  Timings.pipelineTotalSeconds += pipeline.totalSeconds;
  Timings.pipelineListKernelsSeconds += pipeline.listKernelsSeconds;
  Timings.pipelineExtractTextSeconds += pipeline.extractTextSeconds;
  Timings.pipelineCreateTempDirSeconds += pipeline.createTempDirSeconds;
  Timings.pipelineRaiseSeconds += pipeline.raiseSeconds;
  Timings.pipelineWriteIrSeconds += pipeline.writeIrSeconds;
  Timings.pipelineOptSeconds += pipeline.optSeconds;
  Timings.pipelineLlcSeconds += pipeline.llcSeconds;
  Timings.pipelineLinkSeconds += pipeline.linkSeconds;
  Timings.pipelineReadHsacoSeconds += pipeline.readHsacoSeconds;
  Timings.pipelineCollectMetadataSeconds += pipeline.collectMetadataSeconds;
}

void addWriteTimings(HotswapComgrTimings &Timings,
                     const COMGR::hotswap::TranslationCacheWriteTimings &write) {
  Timings.cacheWriteTotalSeconds += write.totalSeconds;
  Timings.cacheWriteKeyBuildSeconds += write.keyBuildSeconds;
  Timings.cacheWriteKeySourceHashSeconds += write.keyBuild.sourceHashSeconds;
  Timings.cacheWriteKeyElfHeaderSeconds += write.keyBuild.elfHeaderSeconds;
  Timings.cacheWriteKeyRulesHashSeconds += write.keyBuild.rulesHashSeconds;
  Timings.cacheWriteKeyLoadedImageIdentitySeconds +=
      write.keyBuild.loadedImageIdentitySeconds;
  Timings.cacheWriteKeyKernelNamesSeconds += write.keyBuild.kernelNamesSeconds;
  Timings.cacheWriteKeyMaterialBuildSeconds +=
      write.keyBuild.materialBuildSeconds;
  Timings.cacheWriteKeyHashSeconds += write.keyBuild.keyHashSeconds;
  Timings.cacheWriteCreateDirectorySeconds += write.createDirectorySeconds;
  Timings.cacheWriteObjectHashSeconds += write.objectHashSeconds;
  Timings.cacheWriteObjectWriteSeconds += write.objectWriteSeconds;
  Timings.cacheWriteMetadataBuildSeconds += write.metadataBuildSeconds;
  Timings.cacheWriteMetadataWriteSeconds += write.metadataWriteSeconds;
}

struct HotswapTranspileResult {
  bool success = false;
  bool cacheHit = false;
  amd_comgr_hotswap_cache_lookup_status_t lookupStatus =
      AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  amd_comgr_hotswap_cache_write_status_t writeStatus =
      AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  int64_t LiftedCount = 0;
  int64_t TotalCount = 0;
  // ScaledModuloReplicationProjection requirement for the transpiled kernel:
  // the factor the launch runtime must scale the block's x extent by (always x,
  // the wave-carrying dimension). 1 means no scaling. Threaded to the loader
  // via get_info.
  int64_t ScaledDispatchFactor = 1;
  std::string backend = "comgr";
  std::string sourceGfx;
  std::string targetGfx;
  std::string cacheKey;
  std::string cacheDetail;
  std::string cacheMetadataPath;
  std::string cacheObjectPath;
  std::string FailReason;
  std::string FailDetail;
  std::string timingJson;
  std::string kernelName;

  static HotswapTranspileResult *convert(
      amd_comgr_hotswap_transpile_result_t result) {
    return reinterpret_cast<HotswapTranspileResult *>(
        static_cast<uintptr_t>(result.handle));
  }

  static amd_comgr_hotswap_transpile_result_t convert(
      HotswapTranspileResult *result) {
    amd_comgr_hotswap_transpile_result_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result))};
    return handle;
  }
};

amd_comgr_status_t createDataObject(amd_comgr_data_kind_t kind,
                                    llvm::StringRef data,
                                    amd_comgr_data_t *output) {
  DataObject *Object = DataObject::allocate(kind);
  if (!Object)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (amd_comgr_status_t Status = Object->setData(data)) {
    Object->release();
    return Status;
  }

  *output = DataObject::convert(Object);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t createExecutableData(llvm::StringRef Hsaco,
                                        amd_comgr_data_t *output) {
  return createDataObject(AMD_COMGR_DATA_KIND_EXECUTABLE, Hsaco, output);
}

struct ResolvedHotswapOptions {
  const char *CacheDirectory = nullptr;
  const char *CacheSkipKernels = nullptr;
  const char *HotswapRulesPath = nullptr;
  bool CacheDisable = false;
  bool CacheReadonly = false;
  bool StrictMode = false;
  bool AssumeHipGlobalOffsetZero = false;
  std::string KernelName;
  unsigned OptLevel = DefaultHotswapComgrOptLevel;
};

bool hasFlag(uint64_t flags, uint64_t flag) { return flags & flag; }

bool hasOptionsFlag(const amd_comgr_hotswap_transpile_options_t *options,
                    amd_comgr_hotswap_transpile_option_flags_t flag) {
  return options && hasFlag(options->flags, static_cast<uint64_t>(flag));
}

bool hasOptionsV2Flag(const amd_comgr_hotswap_transpile_options_v2_t *options,
                      amd_comgr_hotswap_transpile_options_v2_flags_t flag) {
  return options && hasFlag(options->flags, static_cast<uint64_t>(flag));
}

llvm::Expected<std::string>
getKernelNameOption(const amd_comgr_hotswap_transpile_options_v2_t *options) {
  if (!hasOptionsV2Flag(options,
                        AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_KERNEL_NAME))
    return std::string();
  if (!options->kernel_name || options->kernel_name[0] == '\0')
    return llvm::createStringError(
        "hotswap v2 options set USE_KERNEL_NAME without a kernel name");
  return std::string(options->kernel_name);
}

llvm::Expected<unsigned>
getOptLevelOption(const amd_comgr_hotswap_transpile_options_v2_t *options) {
  if (!hasOptionsV2Flag(options,
                        AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_OPT_LEVEL))
    return DefaultHotswapComgrOptLevel;
  if (options->opt_level > MaxHotswapComgrOptLevel)
    return llvm::createStringError(
        "hotswap v2 opt_level must be between 0 and 3");
  return options->opt_level;
}

llvm::Expected<ResolvedHotswapOptions> resolveOptions(
    const amd_comgr_hotswap_transpile_options_t *options) {
  ResolvedHotswapOptions Resolved;
  if (!options) {
    Resolved.CacheDisable = true;
    return Resolved;
  }
  if (options->size < sizeof(amd_comgr_hotswap_transpile_options_t))
    return llvm::createStringError(
        "hotswap options size " + std::to_string(options->size) +
        " is smaller than expected " +
        std::to_string(sizeof(amd_comgr_hotswap_transpile_options_t)));

  Resolved.CacheDirectory = options->cache_directory;
  Resolved.CacheSkipKernels = options->cache_skip_kernels;
  Resolved.HotswapRulesPath = options->hotswap_rules_path;
  Resolved.CacheDisable = hasOptionsFlag(
      options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_DISABLE);
  Resolved.CacheReadonly = hasOptionsFlag(
      options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_READONLY);
  Resolved.StrictMode =
      hasOptionsFlag(options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_STRICT);
  Resolved.AssumeHipGlobalOffsetZero = hasOptionsFlag(
      options,
      AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_ASSUME_HIP_GLOBAL_OFFSET_ZERO);
  return Resolved;
}

llvm::Expected<ResolvedHotswapOptions>
resolveOptionsV2(const amd_comgr_hotswap_transpile_options_v2_t *options) {
  ResolvedHotswapOptions Resolved;
  if (!options) {
    Resolved.CacheDisable = true;
    return Resolved;
  }
  if (options->version != AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_VERSION_2)
    return llvm::createStringError(
        "unsupported hotswap options version " +
        std::to_string(options->version) + " (expected " +
        std::to_string(AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_VERSION_2) + ")");

  llvm::Expected<std::string> KernelName = getKernelNameOption(options);
  if (!KernelName)
    return KernelName.takeError();
  llvm::Expected<unsigned> OptLevel = getOptLevelOption(options);
  if (!OptLevel)
    return OptLevel.takeError();

  Resolved.CacheDirectory = options->cache_directory;
  Resolved.CacheSkipKernels = options->cache_skip_kernels;
  Resolved.HotswapRulesPath = options->hotswap_rules_path;
  Resolved.CacheDisable = hasOptionsV2Flag(
      options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_DISABLE);
  Resolved.CacheReadonly = hasOptionsV2Flag(
      options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_READONLY);
  Resolved.StrictMode =
      hasOptionsV2Flag(options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_STRICT);
  Resolved.AssumeHipGlobalOffsetZero = hasOptionsV2Flag(
      options,
      AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_ASSUME_HIP_GLOBAL_OFFSET_ZERO);
  Resolved.KernelName = std::move(*KernelName);
  Resolved.OptLevel = *OptLevel;
  return Resolved;
}

amd_comgr_status_t invalidOptions(llvm::Error err,
                                  llvm::StringRef functionName) {
  llvm::errs() << functionName << ": " << llvm::toString(std::move(err))
               << "\n";
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

std::string pipelineFailReason(const COMGR::hotswap::PipelineResult &pipeline) {
  if (!pipeline.FailReason.empty())
    return pipeline.FailReason;
  if (!pipeline.Hsaco || pipeline.Hsaco->getBufferSize() == 0)
    return "empty_output";
  return "hotswap_pipeline_failed";
}

std::string pipelineFailDetail(const COMGR::hotswap::PipelineResult &pipeline) {
  if (!pipeline.FailDetail.empty())
    return pipeline.FailDetail;
  if (!pipeline.FailMnemonic.empty())
    return pipeline.FailMnemonic;
  if (!pipeline.FailKernel.empty())
    return pipeline.FailKernel;
  return "hotswap pipeline did not produce a loadable HSACO";
}

amd_comgr_hotswap_cache_lookup_status_t
lookupStatusFromCacheStatus(COMGR::hotswap::TranslationCacheStatus status) {
  switch (status) {
  case COMGR::hotswap::TranslationCacheStatus::Disabled:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  case COMGR::hotswap::TranslationCacheStatus::Bypassed:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_BYPASSED;
  case COMGR::hotswap::TranslationCacheStatus::Miss:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_MISS;
  case COMGR::hotswap::TranslationCacheStatus::Hit:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_HIT;
  case COMGR::hotswap::TranslationCacheStatus::Invalid:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID;
  case COMGR::hotswap::TranslationCacheStatus::WriteSuccess:
  case COMGR::hotswap::TranslationCacheStatus::WriteFailed:
    break;
  }
  return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID;
}

amd_comgr_hotswap_cache_write_status_t
writeStatusFromCacheStatus(COMGR::hotswap::TranslationCacheStatus status) {
  switch (status) {
  case COMGR::hotswap::TranslationCacheStatus::WriteSuccess:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_SUCCESS;
  case COMGR::hotswap::TranslationCacheStatus::WriteFailed:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_FAILED;
  case COMGR::hotswap::TranslationCacheStatus::Disabled:
  case COMGR::hotswap::TranslationCacheStatus::Bypassed:
  case COMGR::hotswap::TranslationCacheStatus::Miss:
  case COMGR::hotswap::TranslationCacheStatus::Hit:
  case COMGR::hotswap::TranslationCacheStatus::Invalid:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  }
  return AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
}

void fillResult(HotswapTranspileResult &result, llvm::StringRef sourceGfx,
                llvm::StringRef targetGfx, bool success, bool cacheHit,
                amd_comgr_hotswap_cache_lookup_status_t lookupStatus,
                amd_comgr_hotswap_cache_write_status_t writeStatus,
                llvm::StringRef cacheDetail,
                const COMGR::hotswap::PipelineResult *pipeline,
                llvm::StringRef cacheKey = "",
                llvm::StringRef cacheMetadataPath = "",
                llvm::StringRef cacheObjectPath = "",
                llvm::StringRef FailReason = "",
                llvm::StringRef FailDetail = "",
                llvm::StringRef timingJson = "",
                llvm::StringRef kernelName = "") {
  result.sourceGfx = sourceGfx.str();
  result.targetGfx = targetGfx.str();
  result.success = success;
  result.cacheHit = cacheHit;
  result.lookupStatus = lookupStatus;
  result.writeStatus = writeStatus;
  result.cacheDetail = cacheDetail.str();
  result.cacheKey = cacheKey.str();
  result.cacheMetadataPath = cacheMetadataPath.str();
  result.cacheObjectPath = cacheObjectPath.str();
  result.FailReason = FailReason.str();
  result.FailDetail = FailDetail.str();
  result.timingJson = timingJson.str();
  result.kernelName = kernelName.str();
  if (pipeline) {
    result.LiftedCount = pipeline->LiftedCount;
    result.TotalCount = pipeline->TotalCount;
    result.ScaledDispatchFactor =
        static_cast<int64_t>(pipeline->ScaledDispatchFactor);
  }
}

amd_comgr_status_t returnResult(HotswapTranspileResult &&value,
                                amd_comgr_hotswap_transpile_result_t *result) {
  if (!result)
    return AMD_COMGR_STATUS_SUCCESS;
  HotswapTranspileResult *owned =
      new (std::nothrow) HotswapTranspileResult(std::move(value));
  if (!owned)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  *result = HotswapTranspileResult::convert(owned);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t hotswapTranspileWithResolvedOptions(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name, const ResolvedHotswapOptions &Options,
    amd_comgr_data_t *output,
    amd_comgr_hotswap_transpile_result_t *result) {
  const bool CollectTimings = HotSwapTimingEnabled();
  auto totalStart = timingStart(CollectTimings);
  HotswapComgrTimings Timings;
  auto finalTimingJson = [&]() {
    if (!CollectTimings)
      return std::string();
    Timings.totalSeconds = timingElapsed(CollectTimings, totalStart);
    return timingJson(Timings);
  };
  DataObject *InputP = DataObject::convert(input);
  if (!InputP || !InputP->Data ||
      InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE || !source_isa_name ||
      !target_isa_name || !output)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Validate both ISA names through the same parser the byte-level
  // `amd_comgr_hotswap_rewrite` uses, so the public contract is identical:
  // malformed identifiers are rejected up-front and never reach the hotswap
  // pipeline. We do not gate on the processor name here — hotswap decides
  // per-kernel whether the source/target pair is supported, and surfaces
  // unsupported instructions as a pipeline failure (see
  // RaiseFailure::reason in amd/comgr/hotswap/raise-failure.hpp).
  TargetIdentifier SourceIdent, TargetIdent;
  if (parseTargetIdentifier(source_isa_name, SourceIdent) ||
      parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  llvm::MemoryBufferRef InputBuf(llvm::StringRef(InputP->Data, InputP->Size),
                                 "hotswap_input");

  COMGR::hotswap::TranslationCacheRequest CacheRequest;
  CacheRequest.SourceObject = InputBuf;
  CacheRequest.SourceGfx = SourceIdent.Processor.str();
  CacheRequest.TargetGfx = TargetIdent.Processor.str();
  CacheRequest.SourceIsa = source_isa_name;
  CacheRequest.TargetIsa = target_isa_name;
  CacheRequest.CodeIsa = source_isa_name;
  CacheRequest.HotswapRulesPath =
      Options.HotswapRulesPath ? Options.HotswapRulesPath : "";
  CacheRequest.CacheDirectory =
      Options.CacheDirectory ? Options.CacheDirectory : "";
  CacheRequest.CacheSkipKernels =
      Options.CacheSkipKernels ? Options.CacheSkipKernels : "";
  CacheRequest.KernelName = Options.KernelName;
  CacheRequest.StrictMode = Options.StrictMode;
  CacheRequest.AssumeHipGlobalOffsetZero = Options.AssumeHipGlobalOffsetZero;
  CacheRequest.CacheDisabled =
      Options.CacheDisable || CacheRequest.CacheDirectory.empty();
  CacheRequest.CacheReadonly = Options.CacheReadonly;
  CacheRequest.CollectTimings = CollectTimings;
  CacheRequest.OptLevel = Options.OptLevel;
  // No env gate: the WaveNative workitem.id.y()/.z()-derived C5 refusal
  // (reduce_kernel RMSNorm aperture class) is auto-upgraded to a scaled
  // dispatch (ScaledModuloReplicationProjection) by the raiser, and the launch
  // runtime honours it via the scaled dim/factor threaded through the
  // transpile result. See amd/comgr/src/hotswap/docs/modrep-predicate-chain.md
  // sec. 10.

  std::string SkippedKernel;
  if (!CacheRequest.KernelName.empty()) {
    llvm::SmallVector<std::string, 1> KernelNames{CacheRequest.KernelName};
    SkippedKernel = COMGR::hotswap::skippedKernelForTranslationCache(
        KernelNames, CacheRequest.CacheSkipKernels);
  } else {
    auto listKernelsStart = timingStart(CollectTimings);
    llvm::Expected<llvm::SmallVector<std::string>> KernelNamesOrErr =
        COMGR::hotswap::listKernelNames(InputBuf);
    Timings.listKernelsSeconds =
        timingElapsed(CollectTimings, listKernelsStart);
    if (!KernelNamesOrErr) {
      llvm::errs() << "amd_comgr_hotswap_transpile: listKernelNames failed: "
                   << llvm::toString(KernelNamesOrErr.takeError()) << "\n";
      return AMD_COMGR_STATUS_ERROR;
    }
    const llvm::SmallVector<std::string> KernelNames =
        std::move(*KernelNamesOrErr);
    SkippedKernel = COMGR::hotswap::skippedKernelForTranslationCache(
        KernelNames, CacheRequest.CacheSkipKernels);
  }

  COMGR::hotswap::TranslationCacheStatus CacheStatus =
      COMGR::hotswap::TranslationCacheStatus::Disabled;
  std::string CacheDetail;
  std::string CacheKey;
  std::string CacheMetadataPath;
  std::string CacheObjectPath;
  bool CacheHit = false;

  COMGR::hotswap::PipelineResult Pipeline;
  if (!SkippedKernel.empty()) {
    CacheStatus = COMGR::hotswap::TranslationCacheStatus::Bypassed;
    CacheDetail = "kernel listed in HSA_HOTSWAP_CACHE_SKIP_KERNELS: " +
                  SkippedKernel;
  } else {
    COMGR::hotswap::TranslationCacheLookup Lookup =
        COMGR::hotswap::lookupTranslationCache(CacheRequest);
    addLookupTimings(Timings, Lookup.Timings);
    CacheStatus = Lookup.Status;
    CacheDetail = Lookup.Reason;
    CacheKey = Lookup.key;
    CacheMetadataPath = Lookup.MetadataPath;
    CacheObjectPath = Lookup.ObjectPath;

    if (Lookup.Status == COMGR::hotswap::TranslationCacheStatus::Invalid) {
      HotswapTranspileResult Result;
      fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, false,
                 false, lookupStatusFromCacheStatus(Lookup.Status),
                 AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED, Lookup.Reason,
                 nullptr, Lookup.key, Lookup.MetadataPath, Lookup.ObjectPath,
                 "cache_invalid", Lookup.Reason, finalTimingJson(),
                 CacheRequest.KernelName);
      if (amd_comgr_status_t ResultStatus =
              returnResult(std::move(Result), result))
        return ResultStatus;
      return AMD_COMGR_STATUS_ERROR;
    }

    if (Lookup.Status == COMGR::hotswap::TranslationCacheStatus::Hit) {
      Pipeline = std::move(Lookup.Result);
      CacheHit = true;
    }
  }

  // A named-kernel request is independent of other metadata kernels in the
  // object; an unnamed request translates and relinks the complete code object.
  if (!CacheHit) {
    COMGR::hotswap::ScopedStrictMode StrictMode(CacheRequest.StrictMode);
    COMGR::hotswap::PipelineOptions PipelineOptions;
    PipelineOptions.EnableWritelaneRewrite =
        CacheRequest.EnableWritelaneRewrite;
    PipelineOptions.EnableWaveNative = CacheRequest.EnableWaveNative;
    PipelineOptions.ForceScaledModrep = CacheRequest.ForceScaledModrep;
    PipelineOptions.CollectTimings = CollectTimings;
    PipelineOptions.AssumeHipGlobalOffsetZero =
        CacheRequest.AssumeHipGlobalOffsetZero;
    PipelineOptions.OptLevel = CacheRequest.OptLevel;
    if (!CacheRequest.KernelName.empty()) {
      Pipeline = COMGR::hotswap::runPipeline(InputBuf,
                                             SourceIdent.Processor,
                                             TargetIdent.Processor,
                                             CacheRequest.KernelName,
                                             PipelineOptions);
    } else {
      Pipeline = COMGR::hotswap::runPipelineAllKernels(
          InputBuf, SourceIdent.Processor, TargetIdent.Processor,
          PipelineOptions);
    }
    addPipelineTimings(Timings, Pipeline.Timings);
  }

  if (!Pipeline.Success || !Pipeline.Hsaco ||
      Pipeline.Hsaco->getBufferSize() == 0) {
    HotswapTranspileResult Result;
    fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, false,
               CacheHit, lookupStatusFromCacheStatus(CacheStatus),
               AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED, CacheDetail,
               &Pipeline, CacheKey, CacheMetadataPath, CacheObjectPath,
               pipelineFailReason(Pipeline), pipelineFailDetail(Pipeline),
               finalTimingJson(), CacheRequest.KernelName);
    if (amd_comgr_status_t ResultStatus =
            returnResult(std::move(Result), result))
      return ResultStatus;
    return AMD_COMGR_STATUS_ERROR;
  }

  amd_comgr_hotswap_cache_write_status_t CacheWriteStatus =
      AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  if (!CacheHit && CacheStatus == COMGR::hotswap::TranslationCacheStatus::Miss) {
    COMGR::hotswap::TranslationCacheWrite Write =
        COMGR::hotswap::writeTranslationCache(CacheRequest, Pipeline);
    addWriteTimings(Timings, Write.Timings);
    CacheWriteStatus = writeStatusFromCacheStatus(Write.Status);
    if (!Write.key.empty())
      CacheKey = Write.key;
    if (!Write.MetadataPath.empty())
      CacheMetadataPath = Write.MetadataPath;
    if (!Write.ObjectPath.empty())
      CacheObjectPath = Write.ObjectPath;
    if (!Write.Reason.empty())
      CacheDetail = Write.Reason;
    if (Write.Status == COMGR::hotswap::TranslationCacheStatus::WriteFailed) {
      HotswapTranspileResult Result;
      fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, false,
                 false, lookupStatusFromCacheStatus(CacheStatus),
                 CacheWriteStatus, Write.Reason, &Pipeline, Write.key,
                 Write.MetadataPath, Write.ObjectPath, "cache_write_failed",
                 Write.Reason, finalTimingJson(), CacheRequest.KernelName);
      if (amd_comgr_status_t ResultStatus =
              returnResult(std::move(Result), result))
        return ResultStatus;
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  amd_comgr_data_t OutputData = {0};
  auto createOutputDataStart = timingStart(CollectTimings);
  if (auto Status =
          createExecutableData(Pipeline.Hsaco->getBuffer(), &OutputData))
    return Status;
  Timings.createOutputDataSeconds =
      timingElapsed(CollectTimings, createOutputDataStart);

  HotswapTranspileResult Result;
  fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, true,
             CacheHit, lookupStatusFromCacheStatus(CacheStatus),
             CacheWriteStatus, CacheDetail, &Pipeline, CacheKey,
             CacheMetadataPath, CacheObjectPath, "", "", finalTimingJson(),
             CacheRequest.KernelName);
  if (amd_comgr_status_t ResultStatus =
          returnResult(std::move(Result), result)) {
    amd_comgr_release_data(OutputData);
    return ResultStatus;
  }

  *output = OutputData;
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_with_options(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name,
    const amd_comgr_hotswap_transpile_options_t *options,
    amd_comgr_data_t *output,
    amd_comgr_hotswap_transpile_result_t *result) {
  llvm::Expected<ResolvedHotswapOptions> Resolved =
      resolveOptions(options);
  if (!Resolved)
    return invalidOptions(Resolved.takeError(),
                          "amd_comgr_hotswap_transpile_with_options");
  return hotswapTranspileWithResolvedOptions(
      input, source_isa_name, target_isa_name, *Resolved, output, result);
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_with_options_v2(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name,
    const amd_comgr_hotswap_transpile_options_v2_t *options,
    amd_comgr_data_t *output,
    amd_comgr_hotswap_transpile_result_t *result) {
  llvm::Expected<ResolvedHotswapOptions> Resolved = resolveOptionsV2(options);
  if (!Resolved)
    return invalidOptions(Resolved.takeError(),
                          "amd_comgr_hotswap_transpile_with_options_v2");
  return hotswapTranspileWithResolvedOptions(
      input, source_isa_name, target_isa_name, *Resolved, output, result);
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name, amd_comgr_data_t *output) {
  return amd_comgr_hotswap_transpile_with_options(
      input, source_isa_name, target_isa_name, nullptr, output, nullptr);
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_destroy_hotswap_transpile_result(
    amd_comgr_hotswap_transpile_result_t result) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  delete Result;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_result_get_info(
    amd_comgr_hotswap_transpile_result_t result,
    amd_comgr_hotswap_transpile_result_info_t info, void *value) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result || !value)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  switch (info) {
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SUCCESS:
    *static_cast<bool *>(value) = Result->success;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_HIT:
    *static_cast<bool *>(value) = Result->cacheHit;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_LOOKUP:
    *static_cast<amd_comgr_hotswap_cache_lookup_status_t *>(value) =
        Result->lookupStatus;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_WRITE:
    *static_cast<amd_comgr_hotswap_cache_write_status_t *>(value) =
        Result->writeStatus;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_LIFTED_COUNT:
    *static_cast<int64_t *>(value) = Result->LiftedCount;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TOTAL_COUNT:
    *static_cast<int64_t *>(value) = Result->TotalCount;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SCALED_DISPATCH_FACTOR:
    *static_cast<int64_t *>(value) = Result->ScaledDispatchFactor;
    return AMD_COMGR_STATUS_SUCCESS;
  }
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_result_get_string(
    amd_comgr_hotswap_transpile_result_t result,
    amd_comgr_hotswap_transpile_result_string_t field, size_t *size,
    char *value) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result || !size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  const std::string *Field = nullptr;
  switch (field) {
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_BACKEND:
    Field = &Result->backend;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SOURCE_GFX:
    Field = &Result->sourceGfx;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TARGET_GFX:
    Field = &Result->targetGfx;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_KEY:
    Field = &Result->cacheKey;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_DETAIL:
    Field = &Result->cacheDetail;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_METADATA_PATH:
    Field = &Result->cacheMetadataPath;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_OBJECT_PATH:
    Field = &Result->cacheObjectPath;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_REASON:
    Field = &Result->FailReason;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_DETAIL:
    Field = &Result->FailDetail;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TIMING_JSON:
    Field = &Result->timingJson;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_KERNEL_NAME:
    Field = &Result->kernelName;
    break;
  }
  if (!Field)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  const size_t Required = Field->size() + 1;
  if (!value) {
    *size = Required;
    return AMD_COMGR_STATUS_SUCCESS;
  }
  if (*size < Required)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  std::memcpy(value, Field->c_str(), Required);
  *size = Required;
  return AMD_COMGR_STATUS_SUCCESS;
}
