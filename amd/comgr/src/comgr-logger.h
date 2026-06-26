//===- comgr-logger.h - Global Comgr logging facility ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares COMGR::Logger, a process-global, thread-safe logging
/// facility shared by every Comgr API. Any API can emit diagnostics at a
/// configurable severity through a single set of utilities (emitError,
/// emitWarning, emitInfo, emitDebug).
///
/// Output goes to two independent destinations:
///   - The global "sink": resolved once from AMD_COMGR_REDIRECT_LOGS (stdout,
///     stderr, or an appended file).
///   - An optional per-thread "capture" stream: installed via LogCaptureScope
///     so emitted messages are also collected into the AMD_COMGR_DATA_KIND_LOG
///     ("comgr.log") data object returned to the caller.
///
/// All writes are guarded by a mutex, so concurrent callers share the sink
/// safely. The severity threshold is configured via AMD_COMGR_LOG_LEVEL; see
/// COMGR::env::getLogLevel().
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_LOGGER_H
#define COMGR_LOGGER_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <mutex>

namespace COMGR {

/// Severity of a log message, ordered from least to most verbose. A message is
/// emitted only when its severity is enabled by the logger's configured level
/// (see Logger::isEnabled). Info corresponds to the "Basic" granularity and
/// Debug to the "Detailed" granularity.
enum class LogLevel {
  None = 0,
  Error,
  Warning,
  Info,
  Debug,
};

/// Process-global, thread-safe logging facility. Obtain the shared instance
/// through getLogger(); do not construct directly.
class Logger {
public:
  /// Construct a Logger configured from the environment (AMD_COMGR_LOG_LEVEL
  /// and AMD_COMGR_REDIRECT_LOGS). Used for the process-global instance.
  Logger();

  /// Construct a Logger with an explicit level and non-owning sink (which may
  /// be null). Intended for tests and embedding.
  Logger(LogLevel Level, llvm::raw_ostream *Sink);

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  /// Return whether a message of the given @p Severity would be emitted under
  /// the current level. Callers that build expensive messages can guard their
  /// formatting with this.
  bool isEnabled(LogLevel Severity) const {
    return Severity != LogLevel::None && Level != LogLevel::None &&
           Severity <= Level;
  }

  /// The currently configured maximum severity that will be emitted.
  LogLevel getLevel() const { return Level; }

  /// Return the global sink stream, resolved once from AMD_COMGR_REDIRECT_LOGS
  /// (stdout, stderr, or an appended file), or null when logs are not
  /// redirected. Callers that maintain their own per-action log stream can
  /// reuse this to avoid opening the redirect destination a second time.
  llvm::raw_ostream *getSink() const { return Sink; }

  /// Emit @p Message at @p Severity, prefixed and newline-terminated. Writes to
  /// the global sink and, when one is installed on the calling thread, the
  /// capture stream. Thread-safe.
  void emit(LogLevel Severity, const llvm::Twine &Message);

  void emitError(const llvm::Twine &Message) { emit(LogLevel::Error, Message); }
  void emitWarning(const llvm::Twine &Message) {
    emit(LogLevel::Warning, Message);
  }
  void emitInfo(const llvm::Twine &Message) { emit(LogLevel::Info, Message); }
  void emitDebug(const llvm::Twine &Message) { emit(LogLevel::Debug, Message); }

private:
  LogLevel Level;

  // The global sink, resolved once at construction. Null when logs are not
  // redirected (AMD_COMGR_REDIRECT_LOGS unset). When pointing at a file, the
  // stream is owned by SinkFile.
  llvm::raw_ostream *Sink;
  std::unique_ptr<llvm::raw_fd_ostream> SinkFile;

  // Guards all writes to Sink and to the active capture stream.
  std::mutex Mutex;
};

/// Return the process-global Logger instance.
Logger &getLogger();

/// Install a capture stream for the current thread for the duration of this
/// scope. While active, every Logger::emit on this thread also writes into
/// @p OS, in addition to the global sink. Nesting is supported: the previous
/// capture stream (if any) is restored on destruction.
class LogCaptureScope {
public:
  explicit LogCaptureScope(llvm::raw_ostream &OS);
  ~LogCaptureScope();

  LogCaptureScope(const LogCaptureScope &) = delete;
  LogCaptureScope &operator=(const LogCaptureScope &) = delete;

private:
  llvm::raw_ostream *Previous;
};

/// Return the capture stream installed on the calling thread, or null. Exposed
/// for Logger::emit; callers should use LogCaptureScope to manage it.
llvm::raw_ostream *getThreadCaptureStream();

} // namespace COMGR

#endif // COMGR_LOGGER_H
