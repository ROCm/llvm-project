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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;

namespace COMGR {

namespace {

// The capture stream is per-thread so that a captured Action on one thread does
// not collect log output emitted by an unrelated API on another thread.
thread_local raw_ostream *ThreadCaptureStream = nullptr;

// Resolve the configured level from the environment. Delegates the mapping to
// the testable parseLogLevel(); kept separate so the Logger constructor reads
// the (cached) environment exactly once.
LogLevel resolveLevel() {
  return parseLogLevel(env::getLogLevel(), env::shouldEmitVerboseLogs());
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
  default:
    llvm_unreachable();
    return "";
  }
}

} // namespace

LogLevel parseLogLevel(StringRef Requested, bool VerboseFallback) {
  // When the variable is unset or unrecognized, default to Debug if verbose
  // logs are requested for back-compat with AMD_COMGR_EMIT_VERBOSE_LOGS,
  // otherwise to Error. An explicit, recognized value always wins (including
  // "none", which silences logging even when verbose logs are requested).
  LogLevel Fallback = VerboseFallback ? LogLevel::Debug : LogLevel::Error;
  return StringSwitch<LogLevel>(Requested)
    .CaseLower("none", LogLevel::None)
    .CaseLower("error", LogLevel::Error)
    .CaseLower("warning", LogLevel::Warning)
    .CaseLower("info", LogLevel::Info)
    .CaseLower("debug", LogLevel::Debug)
    .Default(Fallback);
  /* 
  if (Requested.empty())
    return Fallback;

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

  return Fallback;
  */
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

  // Share the same mutex as emit() so teed output and Logger messages never
  // interleave mid-write on the shared sink. The sink is intentionally left
  // unflushed here; the caller flushes once when the action completes.
  std::scoped_lock<std::mutex> Lock(Mutex);
  *Sink << Data;
}

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
