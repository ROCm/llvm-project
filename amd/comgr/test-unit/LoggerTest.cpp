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
