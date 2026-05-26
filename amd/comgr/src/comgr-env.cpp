//===- comgr-env.cpp - Comgr environment variables ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the management of Comgr's environment variables. See
/// amd/comgr/README.md for descriptions of these.
///
//===----------------------------------------------------------------------===//

#include "comgr-env.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;

namespace COMGR {
namespace env {

bool shouldSaveTemps() {
  static char *SaveTemps = getenv("AMD_COMGR_SAVE_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

bool shouldSaveLLVMTemps() {
  static char *SaveTemps = getenv("AMD_COMGR_SAVE_LLVM_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

std::optional<bool> shouldUseVFS() {
  if (shouldSaveTemps())
    return false;

  static char *UseVFS = getenv("AMD_COMGR_USE_VFS");
  if (UseVFS) {
    if (StringRef(UseVFS) == "0")
      return false;
    else if (StringRef(UseVFS) == "1")
      return true;
  }

  return std::nullopt;
}

std::optional<StringRef> getRedirectLogs() {
  static char *RedirectLogs = getenv("AMD_COMGR_REDIRECT_LOGS");
  if (!RedirectLogs || StringRef(RedirectLogs) == "0") {
    return std::nullopt;
  }
  return StringRef(RedirectLogs);
}

bool needTimeStatistics() {
  static char *TimeStatistics = getenv("AMD_COMGR_TIME_STATISTICS");
  return TimeStatistics && StringRef(TimeStatistics) != "0";
}

bool shouldEmitVerboseLogs() {
  static char *VerboseLogs = getenv("AMD_COMGR_EMIT_VERBOSE_LOGS");
  return VerboseLogs && StringRef(VerboseLogs) != "0";
}

llvm::StringRef getClangExecutablePath() {
  static const std::string Path = [] {
    if (const char *E = std::getenv("AMD_COMGR_CLANG_EXECUTABLE"))
      if (E[0] != '\0')
        return std::string(E);
    if (const char *E = std::getenv("LLVM_PATH"))
      if (E[0] != '\0') {
        SmallString<256> P(E);
        sys::path::append(P, "bin", "clang");
        return std::string(P);
      }
    return std::string();
  }();
  return Path;
}

llvm::StringRef getLLVMInstallDir() {
  static const std::string Dir =
      sys::path::parent_path(sys::path::parent_path(getClangExecutablePath())).str();
  return Dir;
}

StringRef getCachePolicy() {
  static const char *EnvCachePolicy = std::getenv("AMD_COMGR_CACHE_POLICY");
  return EnvCachePolicy;
}

StringRef getCacheDirectory() {
  // By default the cache is enabled
  static const char *Enable = std::getenv("AMD_COMGR_CACHE");
  bool CacheDisabled = StringRef(Enable) == "0";
  if (CacheDisabled)
    return "";

  StringRef EnvCacheDirectory = std::getenv("AMD_COMGR_CACHE_DIR");
  if (!EnvCacheDirectory.empty())
    return EnvCacheDirectory;

  // mark Result as static to keep it cached across calls
  static SmallString<256> Result;
  if (!Result.empty())
    return Result;

  if (sys::path::cache_directory(Result)) {
    sys::path::append(Result, "comgr");
    return Result;
  }

  return "";
}

StringRef getDriverOptionsAppend() {
  static const char *Options = std::getenv("AMD_COMGR_DRIVER_OPTIONS_APPEND");
  return Options ? Options : "";
}

} // namespace env
} // namespace COMGR
