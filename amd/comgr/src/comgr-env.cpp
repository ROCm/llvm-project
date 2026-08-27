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
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <cstdlib>

#ifndef _WIN32
#include <dlfcn.h>
#endif

using namespace llvm;

// Use secure_getenv() on glibc so env-controlled paths are ignored under
// AT_SECURE; no such concept elsewhere, so fall back to getenv().
#if defined(__GLIBC__)
#define COMGR_GETENV secure_getenv
#else
#define COMGR_GETENV getenv
#endif

namespace COMGR {
namespace env {

bool shouldSaveTemps() {
  static char *SaveTemps = COMGR_GETENV("AMD_COMGR_SAVE_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

bool shouldSaveLLVMTemps() {
  static char *SaveTemps = COMGR_GETENV("AMD_COMGR_SAVE_LLVM_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

bool shouldAddEntryTrampolineSymbols() {
  // Opt-in (exactly "1"): the B0->B0 fast path skips the debug-only stub
  // symbols by default on the load-time-critical path.
  static char *AddSyms = COMGR_GETENV("AMD_COMGR_HOTSWAP_ENTRY_STUB_SYMBOLS");
  return AddSyms && StringRef(AddSyms) == "1";
}

std::optional<bool> shouldUseVFS() {
  if (shouldSaveTemps())
    return false;

  static char *UseVFS = COMGR_GETENV("AMD_COMGR_USE_VFS");
  if (UseVFS) {
    if (StringRef(UseVFS) == "0")
      return false;
    else if (StringRef(UseVFS) == "1")
      return true;
  }

  return std::nullopt;
}

std::optional<StringRef> getRedirectLogs() {
  static char *RedirectLogs = COMGR_GETENV("AMD_COMGR_REDIRECT_LOGS");
  if (!RedirectLogs || StringRef(RedirectLogs) == "0") {
    return std::nullopt;
  }
  return StringRef(RedirectLogs);
}

bool needTimeStatistics() {
  static char *TimeStatistics = COMGR_GETENV("AMD_COMGR_TIME_STATISTICS");
  return TimeStatistics && StringRef(TimeStatistics) != "0";
}

uint32_t getGranularityUnitsPerSecond() {
  StringRef G = getTimeStatisticsGranularity();
  if (G == "us")
    return 1e6;
  else if (G == "ns")
    return 1e9;
  return 1e3;
}

llvm::StringRef getTimeStatisticsGranularity() {
  static const char *TimeStatisticsGranularity =
      COMGR_GETENV("AMD_COMGR_TIME_STATISTICS_GRANULARITY");
  if (!TimeStatisticsGranularity)
    return "ms";
  StringRef G(TimeStatisticsGranularity);
  if (G == "ms" || G == "us" || G == "ns")
    return G;
  return "ms";
}

bool shouldEmitVerboseLogs() {
  static char *VerboseLogs = COMGR_GETENV("AMD_COMGR_EMIT_VERBOSE_LOGS");
  return VerboseLogs && StringRef(VerboseLogs) != "0";
}

LogLevel parseLogLevel(StringRef Requested, bool VerboseFallback) {
  // Unset or non-integer: default to Debug when verbose logs are requested
  // (back-compat with AMD_COMGR_EMIT_VERBOSE_LOGS), else Error so errors show.
  unsigned Numeric;
  if (Requested.getAsInteger(10, Numeric))
    return VerboseFallback ? LogLevel::Debug : LogLevel::Error;

  unsigned Max = static_cast<unsigned>(LogLevel::Debug);
  return static_cast<LogLevel>(std::min(Numeric, Max));
}

LogLevel resolveLogLevel() {
  static const char *LogThreshold = getenv("AMD_COMGR_LOG_LEVEL");
  StringRef Requested = LogThreshold ? StringRef(LogThreshold) : StringRef();
  return parseLogLevel(Requested, shouldEmitVerboseLogs());
}

// Probe whether path P names a clang binary whose derived resource directory
// exists on disk. The binary itself need not exist; clang's Driver only uses
// the path to derive the resource dir.
static bool probeClangResourceDir(StringRef P) {
  SmallString<256> ResourceDir(
      sys::path::parent_path(sys::path::parent_path(P)));
  sys::path::append(ResourceDir, "lib", "clang");
  return sys::fs::is_directory(ResourceDir);
}

struct ClangInstallPaths {
  std::string LLVMPrefix;
  std::string ClangBinaryPath;
};

static ClangInstallPaths makeClangInstallPaths(StringRef LLVMPrefix) {
  SmallString<256> ClangBinaryPath(LLVMPrefix);
  sys::path::append(ClangBinaryPath, "bin", "clang");
  return {std::string(LLVMPrefix), std::string(ClangBinaryPath)};
}

// Keep the LLVM install prefix and clang binary path in one cached decision.
// The driver resource directory and VFS header locations are derived from
// these paths; computing them separately can make clang look in a different
// tree from where Comgr plants embedded headers.
static const ClangInstallPaths &getClangInstallPaths() {
  static const ClangInstallPaths Cached = []() -> ClangInstallPaths {
    const char *EnvLLVMPath = COMGR_GETENV("LLVM_PATH");
    if (EnvLLVMPath && StringRef(EnvLLVMPath) != "")
      return makeClangInstallPaths(EnvLLVMPath);

#ifndef _WIN32
    Dl_info Info;
    if (dladdr(reinterpret_cast<void *>(&getClangInstallPaths), &Info) &&
        Info.dli_fname) {
      StringRef SoDir = sys::path::parent_path(Info.dli_fname);

      // Anchor package-layout probing at the loaded Comgr library. The clang
      // path may be synthetic; the in-process driver only needs it to derive
      // the resource directory, so probe the resource tree instead.
      SmallString<256> RocmPrefix(sys::path::parent_path(SoDir));
      sys::path::append(RocmPrefix, "llvm");
      ClangInstallPaths RocmLayout = makeClangInstallPaths(RocmPrefix);
      if (probeClangResourceDir(RocmLayout.ClangBinaryPath))
        return RocmLayout;

      SmallString<256> RuntimeWheelPrefix(SoDir);
      sys::path::append(RuntimeWheelPrefix, "llvm");
      ClangInstallPaths RuntimeWheelLayout =
          makeClangInstallPaths(RuntimeWheelPrefix);
      if (probeClangResourceDir(RuntimeWheelLayout.ClangBinaryPath))
        return RuntimeWheelLayout;

      SmallString<256> StandardPrefix(sys::path::parent_path(SoDir));
      ClangInstallPaths StandardLayout = makeClangInstallPaths(StandardPrefix);
      if (probeClangResourceDir(StandardLayout.ClangBinaryPath))
        return StandardLayout;
    }
#endif

    // Keep fallback paths relative; this avoids assuming a host install layout
    // while still giving clang and Comgr matching VFS keys.
    return makeClangInstallPaths("");
  }();
  return Cached;
}

llvm::StringRef getLLVMPath() { return getClangInstallPaths().LLVMPrefix; }

llvm::StringRef getClangBinaryPath() {
  return getClangInstallPaths().ClangBinaryPath;
}

StringRef getCachePolicy() {
  static const char *EnvCachePolicy = COMGR_GETENV("AMD_COMGR_CACHE_POLICY");
  return EnvCachePolicy ? EnvCachePolicy : "";
}

StringRef getCacheDirectory() {
  // By default the cache is enabled
  static const char *Enable = COMGR_GETENV("AMD_COMGR_CACHE");
  bool CacheDisabled = StringRef(Enable) == "0";
  if (CacheDisabled)
    return "";

  StringRef EnvCacheDirectory = COMGR_GETENV("AMD_COMGR_CACHE_DIR");
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
  static const char *Options = COMGR_GETENV("AMD_COMGR_DRIVER_OPTIONS_APPEND");
  return Options ? Options : "";
}

EmbeddedLibcxxMode getEmbeddedLibcxxMode() {
  static const char *V = std::getenv("AMD_COMGR_USE_EMBEDDED_LIBCXX");
  if (!V)
    return EmbeddedLibcxxMode::Auto;
  StringRef S(V);
  if (S.equals_insensitive("force") || S == "1")
    return EmbeddedLibcxxMode::Force;
  if (S.equals_insensitive("disable") || S == "0")
    return EmbeddedLibcxxMode::Disable;
  return EmbeddedLibcxxMode::Auto;
}

} // namespace env
} // namespace COMGR
