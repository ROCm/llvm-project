//===- LoggerTest.cpp - Unit tests for the global Comgr Logger ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-logger.h"
#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace COMGR;
using llvm::raw_string_ostream;
using llvm::StringRef;

// -- isEnabled / level filtering ---------------------------------------------

TEST(Logger, NoneLevelDisablesEverything) {
  Logger Log(LogLevel::None, nullptr);
  EXPECT_FALSE(Log.isEnabled(LogLevel::Error));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Warning));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Info));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Debug));
}

TEST(Logger, ErrorLevelEnablesOnlyError) {
  Logger Log(LogLevel::Error, nullptr);
  EXPECT_TRUE(Log.isEnabled(LogLevel::Error));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Warning));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Info));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Debug));
}

TEST(Logger, InfoLevelEnablesErrorWarningInfo) {
  Logger Log(LogLevel::Info, nullptr);
  EXPECT_TRUE(Log.isEnabled(LogLevel::Error));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Warning));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Info));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Debug));
}

TEST(Logger, DebugLevelEnablesEverything) {
  Logger Log(LogLevel::Debug, nullptr);
  EXPECT_TRUE(Log.isEnabled(LogLevel::Error));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Warning));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Info));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Debug));
}

TEST(Logger, NeverEmitsNoneSeverity) {
  Logger Log(LogLevel::Debug, nullptr);
  // None is not a real severity and must never pass the filter.
  EXPECT_FALSE(Log.isEnabled(LogLevel::None));
}

// -- parseLogLevel (AMD_COMGR_LOG_LEVEL mapping) -----------------------------

TEST(Logger, ParseLogLevelRecognizedValues) {
  EXPECT_EQ(parseLogLevel("none", false), LogLevel::None);
  EXPECT_EQ(parseLogLevel("error", false), LogLevel::Error);
  EXPECT_EQ(parseLogLevel("warning", false), LogLevel::Warning);
  EXPECT_EQ(parseLogLevel("info", false), LogLevel::Info);
  EXPECT_EQ(parseLogLevel("debug", false), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelIsCaseInsensitive) {
  EXPECT_EQ(parseLogLevel("NONE", false), LogLevel::None);
  EXPECT_EQ(parseLogLevel("Error", false), LogLevel::Error);
  EXPECT_EQ(parseLogLevel("WaRnInG", false), LogLevel::Warning);
  EXPECT_EQ(parseLogLevel("INFO", false), LogLevel::Info);
  EXPECT_EQ(parseLogLevel("Debug", false), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelEmptyUsesVerboseFallback) {
  // Unset variable: Error normally, Debug when verbose logs are requested.
  EXPECT_EQ(parseLogLevel("", false), LogLevel::Error);
  EXPECT_EQ(parseLogLevel("", true), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelUnrecognizedUsesVerboseFallback) {
  // A typo'd level falls back to the same default as an unset variable.
  EXPECT_EQ(parseLogLevel("verbose", false), LogLevel::Error);
  EXPECT_EQ(parseLogLevel("warn", false), LogLevel::Error);
  EXPECT_EQ(parseLogLevel("trace", true), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelExplicitWinsOverVerbose) {
  // An explicit, recognized value overrides the verbose-logs fallback, even
  // when it silences logging entirely.
  EXPECT_EQ(parseLogLevel("none", true), LogLevel::None);
  EXPECT_EQ(parseLogLevel("error", true), LogLevel::Error);
  EXPECT_EQ(parseLogLevel("warning", true), LogLevel::Warning);
}

// -- Sink output and prefixes ------------------------------------------------

TEST(Logger, EmitsPrefixedAndNewlineTerminated) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(LogLevel::Debug, &OS);

  Log.emitError("boom");
  Log.emitWarning("careful");
  Log.emitInfo("fyi");
  Log.emitDebug("trace");
  OS.flush();

  EXPECT_EQ(Out, "comgr: error: boom\n"
                 "comgr: warning: careful\n"
                 "comgr: info: fyi\n"
                 "comgr: debug: trace\n");
}

TEST(Logger, SuppressedSeverityWritesNothing) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(LogLevel::Error, &OS);

  Log.emitWarning("dropped");
  Log.emitInfo("dropped");
  Log.emitDebug("dropped");
  OS.flush();

  EXPECT_TRUE(Out.empty());
}

TEST(Logger, NullSinkDoesNotCrash) {
  Logger Log(LogLevel::Debug, nullptr);
  Log.emitError("no sink");
  SUCCEED();
}

// -- Capture scope -----------------------------------------------------------

TEST(Logger, CaptureScopeTeesEmittedMessages) {
  std::string SinkOut;
  raw_string_ostream SinkOS(SinkOut);
  Logger Log(LogLevel::Debug, &SinkOS);

  std::string CaptureOut;
  raw_string_ostream CaptureOS(CaptureOut);
  {
    LogCaptureScope Capture(CaptureOS);
    Log.emitError("captured");
  }
  // Outside the scope, the capture stream is detached.
  Log.emitError("not captured");

  SinkOS.flush();
  CaptureOS.flush();

  EXPECT_EQ(SinkOut, "comgr: error: captured\ncomgr: error: not captured\n");
  EXPECT_EQ(CaptureOut, "comgr: error: captured\n");
}

TEST(Logger, CaptureScopeRestoresPreviousOnExit) {
  Logger Log(LogLevel::Debug, nullptr);

  std::string OuterOut;
  raw_string_ostream OuterOS(OuterOut);
  std::string InnerOut;
  raw_string_ostream InnerOS(InnerOut);

  {
    LogCaptureScope Outer(OuterOS);
    {
      LogCaptureScope Inner(InnerOS);
      Log.emitInfo("inner");
    }
    Log.emitInfo("outer");
  }
  Log.emitInfo("none");

  OuterOS.flush();
  InnerOS.flush();

  EXPECT_EQ(InnerOut, "comgr: info: inner\n");
  EXPECT_EQ(OuterOut, "comgr: info: outer\n");
  EXPECT_EQ(getThreadCaptureStream(), nullptr);
}

TEST(Logger, CaptureRespectsLevelFilter) {
  Logger Log(LogLevel::Error, nullptr);
  std::string CaptureOut;
  raw_string_ostream CaptureOS(CaptureOut);
  {
    LogCaptureScope Capture(CaptureOS);
    Log.emitDebug("filtered");
    Log.emitError("kept");
  }
  CaptureOS.flush();
  EXPECT_EQ(CaptureOut, "comgr: error: kept\n");
}

// -- Thread safety -----------------------------------------------------------

TEST(Logger, ConcurrentEmitsAreNotInterleaved) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(LogLevel::Debug, &OS);

  const int NumThreads = 8;
  const int PerThread = 200;
  std::vector<std::thread> Threads;
  for (int T = 0; T < NumThreads; ++T) {
    Threads.emplace_back([&Log]() {
      for (int I = 0; I < PerThread; ++I)
        Log.emitError("line");
    });
  }
  for (std::thread &Th : Threads)
    Th.join();
  OS.flush();

  // Every write is atomic under the logger mutex, so each line must be the
  // exact, intact message and the total count must match.
  int Lines = 0;
  StringRef Remaining(Out);
  while (!Remaining.empty()) {
    std::pair<StringRef, StringRef> Split = Remaining.split('\n');
    if (Split.first.empty() && Split.second.empty())
      break;
    if (!Split.first.empty()) {
      EXPECT_EQ(Split.first, "comgr: error: line");
      ++Lines;
    }
    Remaining = Split.second;
  }
  EXPECT_EQ(Lines, NumThreads * PerThread);
}

// -- Capture streams are per-thread ------------------------------------------

TEST(Logger, CaptureStreamIsThreadLocal) {
  Logger Log(LogLevel::Debug, nullptr);

  std::string MainOut;
  raw_string_ostream MainOS(MainOut);
  LogCaptureScope MainCapture(MainOS);

  std::atomic<llvm::raw_ostream *> SeenOnOtherThread{&MainOS};
  std::thread Other([&]() {
    // No capture installed on this thread; it must not see the main thread's.
    SeenOnOtherThread = getThreadCaptureStream();
    Log.emitError("other-thread");
  });
  Other.join();

  Log.emitError("main-thread");
  MainOS.flush();

  EXPECT_EQ(SeenOnOtherThread.load(), nullptr);
  EXPECT_EQ(MainOut, "comgr: error: main-thread\n");
}

// -- Environment-configured constructor --------------------------------------
//
// comgr-env.cpp caches getenv() results in function-local statics, so the
// environment is read only once per process. This must therefore be the ONLY
// test that constructs an environment-configured Logger; a second one with
// different values would not observe them. setenv/unsetenv are POSIX-only.

#ifndef _WIN32
TEST(Logger, RedirectOpenFailureIsRecorded) {
  // Point the redirect at a path inside a directory that does not exist, so the
  // sink cannot be opened. The Logger must record the failure (for the action
  // layer to surface into comgr.log) rather than installing a sink.
  setenv("AMD_COMGR_REDIRECT_LOGS",
         "comgr_logger_nonexistent_dir_a1b2c3/redirect.log", /*overwrite=*/1);
  Logger Log;
  unsetenv("AMD_COMGR_REDIRECT_LOGS");

  EXPECT_EQ(Log.getSink(), nullptr);
  EXPECT_FALSE(Log.getSinkError().empty());
  EXPECT_NE(Log.getSinkError().find("unable to redirect log to file"),
            StringRef::npos);
}
#endif // _WIN32
