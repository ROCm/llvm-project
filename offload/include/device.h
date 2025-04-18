//===----------- device.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_DEVICE_H
#define _OMPTARGET_DEVICE_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>

#include "ExclusiveAccess.h"
#include "OffloadEntry.h"
#include "omptarget.h"
#include "rtl.h"

#include "OpenMP/Mapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "PluginInterface.h"

using GenericPluginTy = llvm::omp::target::plugin::GenericPluginTy;

// Forward declarations.
struct __tgt_bin_desc;
struct __tgt_target_table;

struct DeviceTy {
  int32_t DeviceID;
  GenericPluginTy *RTL;
  int32_t RTLDeviceID;
  /// The physical number of processors that may concurrently execute a team
  /// For cuda, this is number of SMs, for amdgcn, this is number of CUs.
  /// This field is used by ompx_get_team_procs(devid).
  int32_t TeamProcs;


  /// Flag to force synchronous data transfers
  /// Controlled via environment flag OMPX_FORCE_SYNC_REGIONS
  bool ForceSynchronousTargetRegions = false;


  DeviceTy(GenericPluginTy *RTL, int32_t DeviceID, int32_t RTLDeviceID);
  // DeviceTy is not copyable
  DeviceTy(const DeviceTy &D) = delete;
  DeviceTy &operator=(const DeviceTy &D) = delete;

  ~DeviceTy();

  /// Try to initialize the device and return any failure.
  llvm::Error init();

  /// Provide access to the mapping handler.
  MappingInfoTy &getMappingInfo() { return MappingInfo; }

  llvm::Expected<__tgt_device_binary> loadBinary(__tgt_device_image *Img);

  // device memory allocation/deallocation routines
  /// Allocates \p Size bytes on the device, host or shared memory space
  /// (depending on \p Kind) and returns the address/nullptr when
  /// succeeds/fails. \p HstPtr is an address of the host data which the
  /// allocated target data will be associated with. If it is unknown, the
  /// default value of \p HstPtr is nullptr. Note: this function doesn't do
  /// pointer association. Actually, all the __tgt_rtl_data_alloc
  /// implementations ignore \p HstPtr. \p Kind dictates what allocator should
  /// be used (host, shared, device).
  void *allocData(int64_t Size, void *HstPtr = nullptr,
                  int32_t Kind = TARGET_ALLOC_DEFAULT);

  /// Deallocates memory which \p TgtPtrBegin points at and returns
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails. p Kind dictates what
  /// allocator should be used (host, shared, device).
  int32_t deleteData(void *TgtPtrBegin, int32_t Kind = TARGET_ALLOC_DEFAULT);

  // Data transfer. When AsyncInfo is nullptr, the transfer will be
  // synchronous.
  // Copy data from host to device
  int32_t submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                     AsyncInfoTy &AsyncInfo,
                     HostDataToTargetTy *Entry = nullptr,
                     MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr = nullptr);

  // Copy data from device back to host
  int32_t retrieveData(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size,
                       AsyncInfoTy &AsyncInfo,
                       HostDataToTargetTy *Entry = nullptr,
                       MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr = nullptr);

  // Return true if data can be copied to DstDevice directly
  bool isDataExchangable(const DeviceTy &DstDevice);

  // Copy data from current device to destination device directly
  int32_t dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                       int64_t Size, AsyncInfoTy &AsyncInfo);

  /// Notify the plugin about a new mapping starting at the host address
  /// \p HstPtr and \p Size bytes.
  int32_t notifyDataMapped(void *HstPtr, int64_t Size);

  /// Notify the plugin about an existing mapping being unmapped starting at
  /// the host address \p HstPtr.
  int32_t notifyDataUnmapped(void *HstPtr);

  // Launch the kernel identified by \p TgtEntryPtr with the given arguments.
  int32_t launchKernel(void *TgtEntryPtr, void **TgtVarsPtr,
                       ptrdiff_t *TgtOffsets, KernelArgsTy &KernelArgs,
                       AsyncInfoTy &AsyncInfo);

  /// Synchronize device/queue/event based on \p AsyncInfo and return
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails.
  int32_t synchronize(AsyncInfoTy &AsyncInfo);

  /// Query for device/queue/event based completion on \p AsyncInfo in a
  /// non-blocking manner and return OFFLOAD_SUCCESS/OFFLOAD_FAIL when
  /// succeeds/fails. Must be called multiple times until AsyncInfo is
  /// completed and AsyncInfo.isDone() returns true.
  int32_t queryAsync(AsyncInfoTy &AsyncInfo);

  /// Calls the corresponding print device info function in the plugin.
  bool printDeviceInfo();

  /// Event related interfaces.
  /// {
  /// Create an event.
  int32_t createEvent(void **Event);

  /// Record the event based on status in AsyncInfo->Queue at the moment the
  /// function is called.
  int32_t recordEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Wait for an event. This function can be blocking or non-blocking,
  /// depending on the implementation. It is expected to set a dependence on the
  /// event such that corresponding operations shall only start once the event
  /// is fulfilled.
  int32_t waitEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Synchronize the event. It is expected to block the thread.
  int32_t syncEvent(void *Event);

  /// Destroy the event.
  int32_t destroyEvent(void *Event);

  void setTeamProcs(int32_t num_team_procs) { TeamProcs = num_team_procs; }
  int32_t getTeamProcs() { return TeamProcs; }
  /// }

  /// Print all offload entries to stderr.
  void dumpOffloadEntries();

  /// Ask the device whether the runtime should use auto zero-copy.
  bool useAutoZeroCopy();

  /// Ask the device whether it is an APU.
  bool checkIfAPU();

  bool checkIfGFX90a();

  bool checkIfMI300x();

  /// Ask the device whether it supports unified memory.
  bool supportsUnifiedMemory();

  /// Ask the device to perform sanity checks for zero-copy configurations.
  void zeroCopySanityChecksAndDiag(bool isUnifiedSharedMemory,
                                   bool isAutoZeroCopy, bool isEagerMaps);

  /// Check if there are pending images for this device.
  bool hasPendingImages() const { return HasPendingImages; }

  /// Indicate that there are pending images for this device or not.
  void setHasPendingImages(bool V) { HasPendingImages = V; }

  /// Get number of devices used for multi-device kernels
  uint32_t getNumMultiDevices() const;

  /// Check if the kernel is multi device
  bool isMultiDeviceKernel(void *TgtEntryPtr);

private:
  /// Deinitialize the device (and plugin).
  void deinit();

  /// All offload entries available on this device.
  using DeviceOffloadEntriesMapTy =
      llvm::DenseMap<llvm::StringRef, OffloadEntryTy>;
  ProtectedObj<DeviceOffloadEntriesMapTy> DeviceOffloadEntries;

  /// Handler to collect and organize host-2-device mapping information.
  MappingInfoTy MappingInfo;

  /// Flag to indicate pending images (true after construction).
  bool HasPendingImages = true;
};

#endif
