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

// Maximum value on the severity/level scale; values above this are clamped.
constexpr LogLevel MaxLogLevel = 4;

// The capture stream is per-thread so that a captured Action on one thread does
// not collect log output emitted by an unrelated API on another thread.
thread_local raw_ostream *ThreadCaptureStream = nullptr;

// Resolve the configured level from the environment. Delegates the mapping to
// the testable parseLogLevel(); kept separate so the Logger constructor reads
// the (cached) environment exactly once.
LogLevel resolveLevel() {
  return parseLogLevel(env::getLogLevel(), env::shouldEmitVerboseLogs());
}

} // namespace

LogLevel parseLogLevel(StringRef Requested, bool VerboseFallback) {
  // When the variable is unset or not a valid integer, default to the most
  // verbose level if verbose logs are requested for back-compat with
  // AMD_COMGR_EMIT_VERBOSE_LOGS, otherwise to a low level that still shows
  // errors.
  unsigned Numeric;
  if (Requested.getAsInteger(10, Numeric))
    return VerboseFallback ? MaxLogLevel : 1;

  return Numeric > MaxLogLevel ? MaxLogLevel : static_cast<LogLevel>(Numeric);
}

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
      // Record the failure rather than writing it to stderr here. The Logger is
      // constructed before any action's log buffer exists; the action layer
      // surfaces this message into the returned comgr.log via getSinkError(),
      // restoring the pre-Logger behavior of reporting it to the caller.
      SinkError = (Twine("unable to redirect log to file '") + RedirectLog +
                   "': " + EC.message())
                      .str();
    } else {
      Sink = SinkFile.get();
    }
  }
}

Logger::Logger(LogLevel Level, raw_ostream *Sink) : Level(Level), Sink(Sink) {}

void Logger::writeToSink(StringRef Data) {
  if (!Sink)
    return;

  std::scoped_lock<std::mutex> Lock(Mutex);
  *Sink << Data;
}

void Logger::sinkFlush() {
  if (!Sink)
    return;
  std::scoped_lock<std::mutex> Lock(Mutex);
  Sink->flush();
}

void Logger::emit(LogLevel Severity, const Twine &Message) {
  if (!isEnabled(Severity))
    return;

  SmallString<256> Buffer;
  StringRef Text = Message.toStringRef(Buffer);
  StringRef Prefix = "comgr: ";

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
