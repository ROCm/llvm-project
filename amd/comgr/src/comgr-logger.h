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
/// configurable severity through Logger::emit, passing a LogLevel severity.
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
/// COMGR::env::resolveLevel().
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_LOGGER_H
#define COMGR_LOGGER_H

#include "comgr-env.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <mutex>
#include <string>

namespace COMGR {

/// The severity type shared with the logger; defined in comgr-env alongside the
/// AMD_COMGR_LOG_LEVEL parsing that produces it.
using env::LogLevel;

/// Process-global, thread-safe logging facility. Obtain the shared instance
/// through getLogger(); do not construct directly except for tests.
class Logger {
public:
  /// Construct a Logger configured from the environment (AMD_COMGR_LOG_LEVEL
  /// and AMD_COMGR_REDIRECT_LOGS). Used for the process-global instance.
  Logger();

  /// Construct a Logger with an explicit level and non-owning sink (which may
  /// be null). Intended for tests.
  Logger(LogLevel Level, llvm::raw_ostream *Sink);

  /// Construct a Logger with an explicit level, resolving the sink from
  /// @p RedirectTarget exactly as the default constructor resolves
  /// AMD_COMGR_REDIRECT_LOGS ("stdout"/"stderr"/"-" select a stream; any other
  /// value is opened as an append-mode file). Bypasses the process-global
  /// environment cache so redirect behavior can be exercised deterministically.
  /// Intended for tests.
  Logger(LogLevel Level, llvm::StringRef RedirectTarget);

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  /// Return whether a message of the given @p Severity would be emitted under
  /// the current level. A severity of None is never emitted, and emission is
  /// disabled entirely when the level is None. Callers that build expensive
  /// messages can guard their formatting with this.
  bool isEnabled(LogLevel Severity) const {
    return Severity != LogLevel::None && Level != LogLevel::None &&
           Severity <= Level;
  }

  /// The currently configured maximum severity that will be emitted.
  LogLevel getLevel() const { return Level; }

  /// Whether a global redirect sink (from AMD_COMGR_REDIRECT_LOGS) is active.
  /// The stream itself is not exposed; use writeToSink()/sinkFlush() so all
  /// access stays serialized under the logger's mutex.
  bool hasSink() const { return Sink != nullptr; }

  /// Return a diagnostic describing why the redirect sink could not be opened,
  /// or an empty string when redirection was not requested or succeeded. The
  /// Logger is constructed before any per-action log buffer exists, so the
  /// action layer surfaces this into the returned comgr.log when hasSink() is
  /// false despite AMD_COMGR_REDIRECT_LOGS being set.
  llvm::StringRef getSinkError() const { return SinkError; }

  /// Return the filename the redirect sink was opened on, or an empty string
  /// when the sink is a stream (stdout/stderr/"-") or redirection was not
  /// requested. Lets callers reuse the resolved destination (e.g. as the time-
  /// statistics output path) without re-classifying the raw AMD_COMGR_REDIRECT_
  /// LOGS value against the reserved stream names.
  llvm::StringRef getRedirectFilename() const { return SinkFilename; }

  /// Write @p Data verbatim to the global sink under the logger's mutex, so
  /// teed output (see TeeStream in comgr.cpp) does not race emit(). No prefix,
  /// newline, or flush is added; a no-op when there is no sink.
  void writeToSink(llvm::StringRef Data);

  /// Flush the global sink under the logger's mutex. A no-op when there is no
  /// sink.
  void sinkFlush();

  /// Emit @p Message at @p Severity, prefixed and newline-terminated. Writes to
  /// the global sink and, when one is installed on the calling thread, the
  /// capture stream. Thread-safe.
  void emit(LogLevel Severity, const llvm::Twine &Message);

private:
  // Resolve and install the redirect sink from @p RedirectLog, using the same
  // rules as the default constructor: "stdout"/"stderr"/"-" select a standard
  // stream; any other value is opened as an append-mode file, recording a
  // diagnostic in SinkError on failure. Shared by the constructors.
  void openSink(llvm::StringRef RedirectLog);

  LogLevel Level;

  // The global sink, resolved once at construction. Null when logs are not
  // redirected (AMD_COMGR_REDIRECT_LOGS unset). When pointing at a file, the
  // stream is owned by SinkFile.
  llvm::raw_ostream *Sink;
  std::unique_ptr<llvm::raw_fd_ostream> SinkFile;

  // The filename the sink was opened on, when AMD_COMGR_REDIRECT_LOGS named a
  // file that opened successfully. Empty for stream sinks (stdout/stderr/"-")
  // or when redirection was not requested. Surfaced via getRedirectFilename().
  std::string SinkFilename;

  // Diagnostic recorded when AMD_COMGR_REDIRECT_LOGS named a file that could
  // not be opened. Empty otherwise. Surfaced to the caller via getSinkError().
  std::string SinkError;

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
