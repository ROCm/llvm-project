//===- comgr-logger.cpp - Global Comgr logging facility -------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements COMGR::Logger. See comgr-logger.h for the design.
///
//===----------------------------------------------------------------------===//

#include "comgr-logger.h"
#include "comgr-env.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;

namespace COMGR {

namespace {

// The capture stream is per-thread so that a captured Action on one thread does
// not collect log output emitted by an unrelated API on another thread.
thread_local raw_ostream *ThreadCaptureStream = nullptr;

// Map AMD_COMGR_LOG_LEVEL (case-insensitive) to a LogLevel. When the variable
// is unset, default to Debug if verbose logs are requested for back-compat with
// AMD_COMGR_EMIT_VERBOSE_LOGS, otherwise to Error.
LogLevel resolveLevel() {
  StringRef Requested = env::getLogLevel();
  if (Requested.empty())
    return env::shouldEmitVerboseLogs() ? LogLevel::Debug : LogLevel::Error;

  if (Requested.equals_insensitive("none"))
    return LogLevel::None;
  if (Requested.equals_insensitive("error"))
    return LogLevel::Error;
  if (Requested.equals_insensitive("warning"))
    return LogLevel::Warning;
  if (Requested.equals_insensitive("info"))
    return LogLevel::Info;
  if (Requested.equals_insensitive("debug"))
    return LogLevel::Debug;

  // Unrecognized value: fall back to the verbose-logs back-compat default.
  return env::shouldEmitVerboseLogs() ? LogLevel::Debug : LogLevel::Error;
}

StringRef severityPrefix(LogLevel Severity) {
  switch (Severity) {
  case LogLevel::Error:
    return "comgr: error: ";
  case LogLevel::Warning:
    return "comgr: warning: ";
  case LogLevel::Info:
    return "comgr: info: ";
  case LogLevel::Debug:
    return "comgr: debug: ";
  case LogLevel::None:
    return "comgr: ";
  }
  return "comgr: ";
}

} // namespace

Logger::Logger() : Level(resolveLevel()), Sink(nullptr) {
  std::optional<StringRef> RedirectLogs = env::getRedirectLogs();
  if (!RedirectLogs)
    return;

  StringRef RedirectLog = *RedirectLogs;
  if (RedirectLog == "stdout" || RedirectLog == "-") {
    Sink = &outs();
  } else if (RedirectLog == "stderr") {
    Sink = &errs();
  } else {
    std::error_code EC;
    SinkFile = std::make_unique<raw_fd_ostream>(
        RedirectLog, EC, sys::fs::OF_Text | sys::fs::OF_Append);
    if (EC) {
      SinkFile.reset();
      errs() << "comgr: error: unable to redirect log to file '" << RedirectLog
             << "': " << EC.message() << "\n";
    } else {
      Sink = SinkFile.get();
    }
  }
}

Logger::Logger(LogLevel Level, raw_ostream *Sink) : Level(Level), Sink(Sink) {}

void Logger::emit(LogLevel Severity, const Twine &Message) {
  if (!isEnabled(Severity))
    return;

  SmallString<256> Buffer;
  StringRef Text = Message.toStringRef(Buffer);
  StringRef Prefix = severityPrefix(Severity);

  std::scoped_lock<std::mutex> Lock(Mutex);

  raw_ostream *Capture = ThreadCaptureStream;
  if (Sink) {
    *Sink << Prefix << Text << '\n';
    Sink->flush();
  }
  // Guard against double-emission if a capture stream happens to alias the
  // sink.
  if (Capture && Capture != Sink) {
    *Capture << Prefix << Text << '\n';
    Capture->flush();
  }
}

Logger &getLogger() {
  static Logger TheLogger;
  return TheLogger;
}

raw_ostream *getThreadCaptureStream() { return ThreadCaptureStream; }

LogCaptureScope::LogCaptureScope(raw_ostream &OS)
    : Previous(ThreadCaptureStream) {
  ThreadCaptureStream = &OS;
}

LogCaptureScope::~LogCaptureScope() { ThreadCaptureStream = Previous; }

} // namespace COMGR
