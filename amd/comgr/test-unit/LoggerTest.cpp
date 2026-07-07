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
//
// Severities and levels range over LogLevel (None..Debug) where higher is more
// verbose; a message is emitted when its severity is not None and does not
// exceed the configured level.

TEST(Logger, ZeroLevelDisablesEverything) {
  Logger Log(LogLevel::None, nullptr);
  EXPECT_FALSE(Log.isEnabled(LogLevel::Error));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Warning));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Debug));
}

TEST(Logger, LevelEnablesSeveritiesUpToItself) {
  Logger Log(LogLevel::Warning, nullptr);
  EXPECT_TRUE(Log.isEnabled(LogLevel::Error));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Warning));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Info));
  EXPECT_FALSE(Log.isEnabled(LogLevel::Debug));
}

TEST(Logger, MaxLevelEnablesEverything) {
  Logger Log(LogLevel::Debug, nullptr);
  EXPECT_TRUE(Log.isEnabled(LogLevel::Error));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Warning));
  EXPECT_TRUE(Log.isEnabled(LogLevel::Debug));
}

TEST(Logger, NeverEmitsZeroSeverity) {
  Logger Log(LogLevel::Debug, nullptr);
  // A severity of None must never pass the filter.
  EXPECT_FALSE(Log.isEnabled(LogLevel::None));
}

// -- parseLogLevel (AMD_COMGR_LOG_LEVEL mapping) -----------------------------

TEST(Logger, ParseLogLevelNumericValues) {
  EXPECT_EQ(env::parseLogLevel("0", false), LogLevel::None);
  EXPECT_EQ(env::parseLogLevel("2", false), LogLevel::Warning);
  EXPECT_EQ(env::parseLogLevel("4", false), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelClampsAboveMax) {
  EXPECT_EQ(env::parseLogLevel("5", false), LogLevel::Debug);
  EXPECT_EQ(env::parseLogLevel("1000", false), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelEmptyUsesVerboseFallback) {
  // Unset variable: low level normally, max level when verbose logs requested.
  EXPECT_EQ(env::parseLogLevel("", false), LogLevel::Error);
  EXPECT_EQ(env::parseLogLevel("", true), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelNonNumericUsesVerboseFallback) {
  // A non-integer value falls back to the same default as an unset variable.
  EXPECT_EQ(env::parseLogLevel("foo", false), LogLevel::Error);
  EXPECT_EQ(env::parseLogLevel("bar", true), LogLevel::Debug);
}

TEST(Logger, ParseLogLevelExplicitWinsOverVerbose) {
  EXPECT_EQ(env::parseLogLevel("0", true), LogLevel::None);
  EXPECT_EQ(env::parseLogLevel("2", true), LogLevel::Warning);
  EXPECT_EQ(env::parseLogLevel("4", true), LogLevel::Debug);
}

// -- Sink output and prefixes ------------------------------------------------

TEST(Logger, EmitsPrefixedAndNewlineTerminated) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(LogLevel::Debug, &OS);

  Log.emit(LogLevel::Error, "boom");
  Log.emit(LogLevel::Warning, "careful");
  Log.emit(LogLevel::Info, "fyi");
  Log.emit(LogLevel::Debug, "trace");
  OS.flush();

  EXPECT_EQ(Out, "comgr: boom\n"
                 "comgr: careful\n"
                 "comgr: fyi\n"
                 "comgr: trace\n");
}

TEST(Logger, SuppressedSeverityWritesNothing) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(LogLevel::Error, &OS);

  Log.emit(LogLevel::Warning, "dropped");
  Log.emit(LogLevel::Info, "dropped");
  Log.emit(LogLevel::Debug, "dropped");
  OS.flush();

  EXPECT_TRUE(Out.empty());
}

TEST(Logger, NullSinkDoesNotCrash) {
  Logger Log(LogLevel::Debug, nullptr);
  Log.emit(LogLevel::Error, "no sink");
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
    Log.emit(LogLevel::Error, "captured");
  }
  // Outside the scope, the capture stream is detached.
  Log.emit(LogLevel::Error, "not captured");

  SinkOS.flush();
  CaptureOS.flush();

  EXPECT_EQ(SinkOut, "comgr: captured\ncomgr: not captured\n");
  EXPECT_EQ(CaptureOut, "comgr: captured\n");
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
      Log.emit(LogLevel::Info, "inner");
    }
    Log.emit(LogLevel::Info, "outer");
  }
  Log.emit(LogLevel::Info, "none");

  OuterOS.flush();
  InnerOS.flush();

  EXPECT_EQ(InnerOut, "comgr: inner\n");
  EXPECT_EQ(OuterOut, "comgr: outer\n");
  EXPECT_EQ(getThreadCaptureStream(), nullptr);
}

TEST(Logger, CaptureRespectsLevelFilter) {
  Logger Log(LogLevel::Error, nullptr);
  std::string CaptureOut;
  raw_string_ostream CaptureOS(CaptureOut);
  {
    LogCaptureScope Capture(CaptureOS);
    Log.emit(LogLevel::Debug, "filtered");
    Log.emit(LogLevel::Error, "kept");
  }
  CaptureOS.flush();
  EXPECT_EQ(CaptureOut, "comgr: kept\n");
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
    Threads.emplace_back([&Log, PerThread]() {
      for (int I = 0; I < PerThread; ++I)
        Log.emit(LogLevel::Error, "line");
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
      EXPECT_EQ(Split.first, "comgr: line");
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
    Log.emit(LogLevel::Error, "other-thread");
  });
  Other.join();

  Log.emit(LogLevel::Error, "main-thread");
  MainOS.flush();

  EXPECT_EQ(SeenOnOtherThread.load(), nullptr);
  EXPECT_EQ(MainOut, "comgr: main-thread\n");
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

  EXPECT_FALSE(Log.hasSink());
  EXPECT_FALSE(Log.getSinkError().empty());
  EXPECT_NE(Log.getSinkError().find("unable to redirect log to file"),
            StringRef::npos);
}
#endif // _WIN32
