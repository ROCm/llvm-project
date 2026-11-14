//===- perf-timer.h - Timing statistics -----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMD_COMGR_PERF_TIMER_H
#define AMD_COMGR_PERF_TIMER_H

#include <cstdint>
#include <memory>

namespace COMGR {
namespace TimeStatistics {

// Timer abstract interface
class PerfTimerImpl {
protected:
  long long CounterStart;
  double PCFreq;
  uint32_t GranularityPerSecond;

public:
  PerfTimerImpl() : CounterStart(0), PCFreq(0.0) {};
  virtual ~PerfTimerImpl() = default;
  virtual bool init() = 0;
  virtual double getCurrentTime() = 0;
};

// Timer client interface class
class PerfTimer {
  std::unique_ptr<PerfTimerImpl> Impl;

public:
  bool init();
  double getCurrentTime() { return Impl->getCurrentTime(); }
};
} // namespace TimeStatistics
} // namespace COMGR

#endif // AMD_COMGR_PERF_TIMER_H
