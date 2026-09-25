//===- OmptProfiler.cpp - OMPT impl of GenericProfilerTy --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OmptProfilerTy
//
//===----------------------------------------------------------------------===//

#include "OmptProfiler.h"
#include "OpenMP/OMPT/Interface.h"
#include "PluginInterface.h"
#include "Shared/Debug.h"

using namespace llvm::omp::target;

// NOTE: the device_initialize / device_finalize / device_load callbacks are
// deliberately *not* implemented here. They are dispatched by DeviceTy in
// libompaccsupport (init(), deinit() and loadBinary()), which is the layer that
// knows the device number OpenMP reports to the user. Dispatching them from
// here as well would deliver each callback twice; because the two sites pass
// different 'lookup' functions, the second dispatch would also hand the tool a
// lookup that cannot resolve the device tracing entry points, nulling out the
// handles it saved from the first.

void ompt::OmptProfilerTy::handleDataAlloc(uint64_t StartNanos,
                                           uint64_t EndNanos, void *HostPtr,
                                           uint64_t Size, void *Data) {
  ompt::setOmptTimestamp(StartNanos, EndNanos);
}

void ompt::OmptProfilerTy::handleDataDelete(uint64_t StartNanos,
                                            uint64_t EndNanos, void *TgtPtr,
                                            void *Data) {
  ompt::setOmptTimestamp(StartNanos, EndNanos);
}

void ompt::OmptProfilerTy::handlePreKernelLaunch(
    plugin::GenericDeviceTy *Device, uint32_t NumBlocks[3],
    __tgt_async_info *AI) {
  if (!ompt::isTracedDevice(getDeviceId(Device)))
    return;

  if (AI->ProfilerData == nullptr)
    return;

  auto ProfilerSpecificData =
      reinterpret_cast<ompt::OmptEventInfoTy *>(AI->ProfilerData);
  assert(ProfilerSpecificData && "Invalid ProfilerSpecificData");
  // Set number of granted teams for OMPT
  setOmptGrantedNumTeams(NumBlocks[0]);
  ProfilerSpecificData->NumTeams = NumBlocks[0];
}

void ompt::OmptProfilerTy::handleKernelCompletion(uint64_t StartNanos,
                                                  uint64_t EndNanos,
                                                  void *Data) {

  if (!isProfilingEnabled())
    return;

  /// Empty data means no tracing in OMPT
  /// offload/include/OpenMP/OMPT/Interface.h line 492
  if (!Data)
    return;

  ODBG(ODT_Tool) << "OMPT-Async: Time kernel for asynchronous execution: Start "
                 << StartNanos << " End " << EndNanos;

  auto OmptEventInfo = reinterpret_cast<ompt::OmptEventInfoTy *>(Data);
  assert(OmptEventInfo && "Invalid OmptEventInfo");
  assert(OmptEventInfo->TraceRecord && "Invalid TraceRecord");

  ompt::RegionInterface.stopTargetSubmitTraceAsync(OmptEventInfo->TraceRecord,
                                                   OmptEventInfo->NumTeams,
                                                   StartNanos, EndNanos);

  // Done processing, our responsibility to free the memory
  freeProfilerDataEntry(OmptEventInfo);
}

void ompt::OmptProfilerTy::handleDataTransfer(uint64_t StartNanos,
                                              uint64_t EndNanos, void *Data) {

  if (!isProfilingEnabled())
    return;

  /// Empty data means no tracing in OMPT
  /// offload/include/OpenMP/OMPT/Interface.h line 492
  if (!Data)
    return;

  ODBG(ODT_Tool) << "OMPT-Async: Time data for asynchronous execution: Start "
                 << StartNanos << " End " << EndNanos;

  auto OmptEventInfo = reinterpret_cast<ompt::OmptEventInfoTy *>(Data);
  assert(OmptEventInfo && "Invalid OmptEventInfo");
  assert(OmptEventInfo->TraceRecord && "Invalid TraceRecord");

  ompt::RegionInterface.stopTargetDataMovementTraceAsync(
      OmptEventInfo->TraceRecord, StartNanos, EndNanos);

  // Done processing, our responsibility to free the memory
  freeProfilerDataEntry(OmptEventInfo);
}

bool ompt::OmptProfilerTy::isProfilingEnabled() { return ompt::TracingActive; }

void ompt::OmptProfilerTy::setTimeConversionFactorsImpl(double Slope,
                                                        double Offset) {
  ODBG(ODT_Tool) << "Using Time Slope: " << Slope << " and Offset: " << Offset;
  setOmptHostToDeviceRate(Slope, Offset);
}
