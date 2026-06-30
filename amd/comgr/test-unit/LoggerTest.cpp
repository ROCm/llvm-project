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
// Severities and levels are on a 0-to-20 scale where higher is more verbose; a
// message is emitted when its severity is non-zero and does not exceed the
// configured level.

TEST(Logger, ZeroLevelDisablesEverything) {
  Logger Log(0, nullptr);
  EXPECT_FALSE(Log.isEnabled(5));
  EXPECT_FALSE(Log.isEnabled(10));
  EXPECT_FALSE(Log.isEnabled(20));
}

TEST(Logger, LevelEnablesSeveritiesUpToItself) {
  Logger Log(10, nullptr);
  EXPECT_TRUE(Log.isEnabled(5));
  EXPECT_TRUE(Log.isEnabled(10));
  EXPECT_FALSE(Log.isEnabled(11));
  EXPECT_FALSE(Log.isEnabled(20));
}

TEST(Logger, MaxLevelEnablesEverything) {
  Logger Log(20, nullptr);
  EXPECT_TRUE(Log.isEnabled(5));
  EXPECT_TRUE(Log.isEnabled(10));
  EXPECT_TRUE(Log.isEnabled(20));
}

TEST(Logger, NeverEmitsZeroSeverity) {
  Logger Log(20, nullptr);
  // A severity of 0 must never pass the filter.
  EXPECT_FALSE(Log.isEnabled(0));
}

// -- parseLogLevel (AMD_COMGR_LOG_LEVEL mapping) -----------------------------

TEST(Logger, ParseLogLevelNumericValues) {
  EXPECT_EQ(int{parseLogLevel("0", false)}, 0);
  EXPECT_EQ(int{parseLogLevel("5", false)}, 5);
  EXPECT_EQ(int{parseLogLevel("20", false)}, 20);
}

TEST(Logger, ParseLogLevelClampsAboveMax) {
  EXPECT_EQ(int{parseLogLevel("21", false)}, 20);
  EXPECT_EQ(int{parseLogLevel("1000", false)}, 20);
}

TEST(Logger, ParseLogLevelEmptyUsesVerboseFallback) {
  // Unset variable: low level normally, max level when verbose logs requested.
  EXPECT_EQ(int{parseLogLevel("", false)}, 5);
  EXPECT_EQ(int{parseLogLevel("", true)}, 20);
}

TEST(Logger, ParseLogLevelNonNumericUsesVerboseFallback) {
  // A non-integer value falls back to the same default as an unset variable.
  EXPECT_EQ(int{parseLogLevel("foo", false)}, 5);
  EXPECT_EQ(int{parseLogLevel("bar", true)}, 20);
}

// -- Sink output and prefixes ------------------------------------------------

TEST(Logger, EmitsPrefixedAndNewlineTerminated) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(20, &OS);

  Log.emit(5, "boom");
  Log.emit(10, "careful");
  Log.emit(15, "fyi");
  Log.emit(20, "trace");
  OS.flush();

  EXPECT_EQ(Out, "comgr: boom\n"
                 "comgr: careful\n"
                 "comgr: fyi\n"
                 "comgr: trace\n");
}

TEST(Logger, SuppressedSeverityWritesNothing) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(5, &OS);

  Log.emit(10, "dropped");
  Log.emit(15, "dropped");
  Log.emit(20, "dropped");
  OS.flush();

  EXPECT_TRUE(Out.empty());
}

TEST(Logger, NullSinkDoesNotCrash) {
  Logger Log(20, nullptr);
  Log.emit(5, "no sink");
  SUCCEED();
}

// -- Capture scope -----------------------------------------------------------

TEST(Logger, CaptureScopeTeesEmittedMessages) {
  std::string SinkOut;
  raw_string_ostream SinkOS(SinkOut);
  Logger Log(20, &SinkOS);

  std::string CaptureOut;
  raw_string_ostream CaptureOS(CaptureOut);
  {
    LogCaptureScope Capture(CaptureOS);
    Log.emit(5, "captured");
  }
  // Outside the scope, the capture stream is detached.
  Log.emit(5, "not captured");

  SinkOS.flush();
  CaptureOS.flush();

  EXPECT_EQ(SinkOut, "comgr: captured\ncomgr: not captured\n");
  EXPECT_EQ(CaptureOut, "comgr: captured\n");
}

TEST(Logger, CaptureScopeRestoresPreviousOnExit) {
  Logger Log(20, nullptr);

  std::string OuterOut;
  raw_string_ostream OuterOS(OuterOut);
  std::string InnerOut;
  raw_string_ostream InnerOS(InnerOut);

  {
    LogCaptureScope Outer(OuterOS);
    {
      LogCaptureScope Inner(InnerOS);
      Log.emit(15, "inner");
    }
    Log.emit(15, "outer");
  }
  Log.emit(15, "none");

  OuterOS.flush();
  InnerOS.flush();

  EXPECT_EQ(InnerOut, "comgr: inner\n");
  EXPECT_EQ(OuterOut, "comgr: outer\n");
  EXPECT_EQ(getThreadCaptureStream(), nullptr);
}

TEST(Logger, CaptureRespectsLevelFilter) {
  Logger Log(5, nullptr);
  std::string CaptureOut;
  raw_string_ostream CaptureOS(CaptureOut);
  {
    LogCaptureScope Capture(CaptureOS);
    Log.emit(20, "filtered");
    Log.emit(5, "kept");
  }
  CaptureOS.flush();
  EXPECT_EQ(CaptureOut, "comgr: kept\n");
}

// -- Thread safety -----------------------------------------------------------

TEST(Logger, ConcurrentEmitsAreNotInterleaved) {
  std::string Out;
  raw_string_ostream OS(Out);
  Logger Log(20, &OS);

  const int NumThreads = 8;
  const int PerThread = 200;
  std::vector<std::thread> Threads;
  for (int T = 0; T < NumThreads; ++T) {
    Threads.emplace_back([&Log]() {
      for (int I = 0; I < PerThread; ++I)
        Log.emit(5, "line");
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
  Logger Log(20, nullptr);

  std::string MainOut;
  raw_string_ostream MainOS(MainOut);
  LogCaptureScope MainCapture(MainOS);

  std::atomic<llvm::raw_ostream *> SeenOnOtherThread{&MainOS};
  std::thread Other([&]() {
    // No capture installed on this thread; it must not see the main thread's.
    SeenOnOtherThread = getThreadCaptureStream();
    Log.emit(5, "other-thread");
  });
  Other.join();

  Log.emit(5, "main-thread");
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

  EXPECT_EQ(Log.getSink(), nullptr);
  EXPECT_FALSE(Log.getSinkError().empty());
  EXPECT_NE(Log.getSinkError().find("unable to redirect log to file"),
            StringRef::npos);
}
#endif // _WIN32
