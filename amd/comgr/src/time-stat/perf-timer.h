//===- perf-timer.h - Timing statistics -----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMD_COMGR_PERF_TIMER_H
#define AMD_COMGR_PERF_TIMER_H

namespace COMGR {
namespace TimeStatistics {

// Timer abstract interface
class PerfTimerImpl {
protected:
  long long CounterStart;
  double PCFreq;

public:
  PerfTimerImpl() : CounterStart(0), PCFreq(0.0) {};
  virtual ~PerfTimerImpl() = default;
  virtual bool Init() = 0;
  virtual double getCurrentTime() = 0;
};

// Timer client interface class
class PerfTimer {
  std::unique_ptr<PerfTimerImpl> pImpl;

public:
  bool Init();
  double getCurrentTime() { return pImpl->getCurrentTime(); }
};
} // namespace TimeStatistics
} // namespace COMGR

#endif // AMD_COMGR_PERF_TIMER_H
