//===------ omptarget.cpp - Target independent OpenMP target RTL -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include "omptarget.h"
#include "OffloadPolicy.h"
#include "OpenMP/OMPT/Callback.h"
#include "OpenMP/OMPT/Interface.h"
#include "PluginManager.h"
#include "Shared/Debug.h"
#include "Shared/EnvironmentVar.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

#include "Shared/APITypes.h"
#include "Shared/Profile.h"

#include "OpenMP/Mapping.h"
#include "OpenMP/omp.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Object/ObjectFile.h"

#include <cassert>
#include <cstdint>
#include <vector>

using llvm::SmallVector;

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif

int AsyncInfoTy::synchronize() {
  int Result = OFFLOAD_SUCCESS;
  if (!isQueueEmpty()) {
    switch (SyncType) {
    case SyncTy::BLOCKING:
      // If we have a queue we need to synchronize it now.
      Result = Device.synchronize(*this);
      assert(AsyncInfo.Queue == nullptr &&
             "The device plugin should have nulled the queue to indicate there "
             "are no outstanding actions!");
      break;
    case SyncTy::NON_BLOCKING:
      Result = Device.queryAsync(*this);
      break;
    }
  }

  // Run any pending post-processing function registered on this async object.
  if (Result == OFFLOAD_SUCCESS && isQueueEmpty())
    Result = runPostProcessing();

  return Result;
}

void *&AsyncInfoTy::getVoidPtrLocation() {
  BufferLocations.push_back(nullptr);
  return BufferLocations.back();
}

bool AsyncInfoTy::isDone() const { return isQueueEmpty(); }

int32_t AsyncInfoTy::runPostProcessing() {
  size_t Size = PostProcessingFunctions.size();
  for (size_t I = 0; I < Size; ++I) {
    const int Result = PostProcessingFunctions[I]();
    if (Result != OFFLOAD_SUCCESS)
      return Result;
  }

  // Clear the vector up until the last known function, since post-processing
  // procedures might add new procedures themselves.
  const auto *PrevBegin = PostProcessingFunctions.begin();
  PostProcessingFunctions.erase(PrevBegin, PrevBegin + Size);

  return OFFLOAD_SUCCESS;
}

bool AsyncInfoTy::isQueueEmpty() const { return AsyncInfo.Queue == nullptr; }

/* All begin addresses for partially mapped structs must be aligned, up to 16,
 * in order to ensure proper alignment of members. E.g.
 *
 * struct S {
 *   int a;   // 4-aligned
 *   int b;   // 4-aligned
 *   int *p;  // 8-aligned
 * } s1;
 * ...
 * #pragma omp target map(tofrom: s1.b, s1.p[0:N])
 * {
 *   s1.b = 5;
 *   for (int i...) s1.p[i] = ...;
 * }
 *
 * Here we are mapping s1 starting from member b, so BaseAddress=&s1=&s1.a and
 * BeginAddress=&s1.b. Let's assume that the struct begins at address 0x100,
 * then &s1.a=0x100, &s1.b=0x104, &s1.p=0x108. Each member obeys the alignment
 * requirements for its type. Now, when we allocate memory on the device, in
 * CUDA's case cuMemAlloc() returns an address which is at least 256-aligned.
 * This means that the chunk of the struct on the device will start at a
 * 256-aligned address, let's say 0x200. Then the address of b will be 0x200 and
 * address of p will be a misaligned 0x204 (on the host there was no need to add
 * padding between b and p, so p comes exactly 4 bytes after b). If the device
 * kernel tries to access s1.p, a misaligned address error occurs (as reported
 * by the CUDA plugin). By padding the begin address down to a multiple of 8 and
 * extending the size of the allocated chuck accordingly, the chuck on the
 * device will start at 0x200 with the padding (4 bytes), then &s1.b=0x204 and
 * &s1.p=0x208, as they should be to satisfy the alignment requirements.
 */
static const int64_t MaxAlignment = 16;

/// Return the alignment requirement of partially mapped structs, see
/// MaxAlignment above.
static uint64_t getPartialStructRequiredAlignment(void *HstPtrBase) {
  int LowestOneBit = __builtin_ffsl(reinterpret_cast<uintptr_t>(HstPtrBase));
  uint64_t BaseAlignment = 1 << (LowestOneBit - 1);
  return MaxAlignment < BaseAlignment ? MaxAlignment : BaseAlignment;
}

void handleTargetOutcome(bool Success, ident_t *Loc) {
  switch (OffloadPolicy::get(*PM).Kind) {
  case OffloadPolicy::DISABLED:
    if (Success) {
      FATAL_MESSAGE0(1, "expected no offloading while offloading is disabled");
    }
    break;
  case OffloadPolicy::MANDATORY:
    if (!Success) {
      if (getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE) {
        auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
        for (auto &Device : PM->devices(ExclusiveDevicesAccessor))
          dumpTargetPointerMappings(Loc, Device);
      } else
        FAILURE_MESSAGE("Consult https://openmp.llvm.org/design/Runtimes.html "
                        "for debugging options.\n");

      if (!PM->getNumActivePlugins())
        FAILURE_MESSAGE(
            "No images found compatible with the installed hardware. ");

      SourceInfo Info(Loc);
      if (Info.isAvailible())
        fprintf(stderr, "%s:%d:%d: ", Info.getFilename(), Info.getLine(),
                Info.getColumn());
      else
        FAILURE_MESSAGE("Source location information not present. Compile with "
                        "-g or -gline-tables-only.\n");
      FATAL_MESSAGE0(
          1, "failure of target construct while offloading is mandatory");
    } else {
      if (getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE) {
        auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
        for (auto &Device : PM->devices(ExclusiveDevicesAccessor))
          dumpTargetPointerMappings(Loc, Device);
      }
    }
    break;
  }
}

static int32_t getParentIndex(int64_t Type) {
  return ((Type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

void *targetAllocExplicit(size_t Size, int DeviceNum, int Kind,
                          const char *Name) {
  DP("Call to %s for device %d requesting %zu bytes\n", Name, DeviceNum, Size);

  if (Size <= 0) {
    DP("Call to %s with non-positive length\n", Name);
    return NULL;
  }

  void *Rc = NULL;

  if (DeviceNum == omp_get_initial_device()) {
    Rc = malloc(Size);
    DP("%s returns host ptr " DPxMOD "\n", Name, DPxPTR(Rc));
    return Rc;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  Rc = DeviceOrErr->allocData(Size, nullptr, Kind);
  DP("%s returns device ptr " DPxMOD "\n", Name, DPxPTR(Rc));
  return Rc;
}

void targetFreeExplicit(void *DevicePtr, int DeviceNum, int Kind,
                        const char *Name) {
  DP("Call to %s for device %d and address " DPxMOD "\n", Name, DeviceNum,
     DPxPTR(DevicePtr));

  if (!DevicePtr) {
    DP("Call to %s with NULL ptr\n", Name);
    return;
  }

  if (DeviceNum == omp_get_initial_device()) {
    free(DevicePtr);
    DP("%s deallocated host ptr\n", Name);
    return;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  if (DeviceOrErr->deleteData(DevicePtr, Kind) == OFFLOAD_FAIL)
    FATAL_MESSAGE(DeviceNum, "%s",
                  "Failed to deallocate device ptr. Set "
                  "OFFLOAD_TRACK_ALLOCATION_TRACES=1 to track allocations.");

  DP("omp_target_free deallocated device ptr\n");
}

void *targetLockExplicit(void *HostPtr, size_t Size, int DeviceNum,
                         const char *Name) {
  DP("Call to %s for device %d locking %zu bytes\n", Name, DeviceNum, Size);

  if (Size <= 0) {
    DP("Call to %s with non-positive length\n", Name);
    return NULL;
  }

  void *RC = NULL;

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  int32_t Err = 0;
  Err = DeviceOrErr->RTL->data_lock(DeviceNum, HostPtr, Size, &RC);
  if (Err) {
    DP("Could not lock ptr %p\n", HostPtr);
    return nullptr;
  }
  DP("%s returns device ptr " DPxMOD "\n", Name, DPxPTR(RC));
  return RC;
}

void targetUnlockExplicit(void *HostPtr, int DeviceNum, const char *Name) {
  DP("Call to %s for device %d unlocking\n", Name, DeviceNum);

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  DeviceOrErr->RTL->data_unlock(DeviceNum, HostPtr);
  DP("%s returns\n", Name);
}

/// Call the user-defined mapper function followed by the appropriate
// targetData* function (targetData{Begin,End,Update}).
int targetDataMapper(ident_t *Loc, DeviceTy &Device, void *ArgBase, void *Arg,
                     int64_t ArgSize, int64_t ArgType, map_var_info_t ArgNames,
                     void *ArgMapper, AsyncInfoTy &AsyncInfo,
                     TargetDataFuncPtrTy TargetDataFunction) {
  DP("Calling the mapper function " DPxMOD "\n", DPxPTR(ArgMapper));

  // The mapper function fills up Components.
  MapperComponentsTy MapperComponents;
  MapperFuncPtrTy MapperFuncPtr = (MapperFuncPtrTy)(ArgMapper);
  (*MapperFuncPtr)((void *)&MapperComponents, ArgBase, Arg, ArgSize, ArgType,
                   ArgNames);

  // Construct new arrays for args_base, args, arg_sizes and arg_types
  // using the information in MapperComponents and call the corresponding
  // targetData* function using these new arrays.
  std::vector<void *> MapperArgsBase(MapperComponents.Components.size());
  std::vector<void *> MapperArgs(MapperComponents.Components.size());
  std::vector<int64_t> MapperArgSizes(MapperComponents.Components.size());
  std::vector<int64_t> MapperArgTypes(MapperComponents.Components.size());
  std::vector<void *> MapperArgNames(MapperComponents.Components.size());

  for (unsigned I = 0, E = MapperComponents.Components.size(); I < E; ++I) {
    auto &C = MapperComponents.Components[I];
    MapperArgsBase[I] = C.Base;
    MapperArgs[I] = C.Begin;
    MapperArgSizes[I] = C.Size;
    MapperArgTypes[I] = C.Type;
    MapperArgNames[I] = C.Name;
  }

  int Rc = TargetDataFunction(Loc, Device, MapperComponents.Components.size(),
                              MapperArgsBase.data(), MapperArgs.data(),
                              MapperArgSizes.data(), MapperArgTypes.data(),
                              MapperArgNames.data(), /*arg_mappers*/ nullptr,
                              AsyncInfo, /*FromMapper=*/true);

  return Rc;
}

static int prepareAndSubmitData(DeviceTy &Device, void *HstPtrBegin,
                                void *HstPtrBase, void *LocalTgtPtrBegin,
                                TargetPointerResultTy &PointerTpr,
                                void *PointerHstPtrBegin,
                                void *PointerTgtPtrBegin,
                                AsyncInfoTy &AsyncInfo, bool IsDescBaseAddr) {
  uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
  void *ExpectedTgtPtrBase = (void *)((uint64_t)LocalTgtPtrBegin - Delta);

  if (PointerTpr.getEntry()->addShadowPointer(ShadowPtrInfoTy{
          (void **)PointerHstPtrBegin, HstPtrBase, (void **)PointerTgtPtrBegin,
          ExpectedTgtPtrBase, IsDescBaseAddr})) {
    DP("USM_SPECIAL: Update pointer (" DPxMOD ") -> [" DPxMOD "]\n",
       DPxPTR(PointerTgtPtrBegin), DPxPTR(LocalTgtPtrBegin));

    void *&LocalTgtPtrBase = AsyncInfo.getVoidPtrLocation();
    LocalTgtPtrBase = ExpectedTgtPtrBase;

    int Ret =
        Device.submitData(PointerTgtPtrBegin, &LocalTgtPtrBase, sizeof(void *),
                          AsyncInfo, PointerTpr.getEntry());
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data to device failed.\n");
      return OFFLOAD_FAIL;
    }
    if (PointerTpr.getEntry()->addEventIfNecessary(Device, AsyncInfo) !=
        OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

/// Internal function to do the mapping and transfer the data to the device
int targetDataBegin(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                    void **ArgsBase, void **Args, int64_t *ArgSizes,
                    int64_t *ArgTypes, map_var_info_t *ArgNames,
                    void **ArgMappers, AsyncInfoTy &AsyncInfo,
                    bool FromMapper) {
  // process each input.
  for (int32_t I = 0; I < ArgNum; ++I) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;
    TIMESCOPE_WITH_DETAILS_AND_IDENT(
        "HostToDev", "Size=" + std::to_string(ArgSizes[I]) + "B", Loc);
    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataBegin, call the
      // targetDataMapper variant which will call targetDataBegin again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", I);

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      int Rc = targetDataMapper(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                                targetDataBegin);

      if (Rc != OFFLOAD_SUCCESS) {
        REPORT("Call to targetDataBegin via targetDataMapper for custom mapper"
               " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = Args[I];
    void *HstPtrBase = ArgsBase[I];
    int64_t DataSize = ArgSizes[I];
    map_var_info_t HstPtrName = (!ArgNames) ? nullptr : ArgNames[I];

    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    int64_t TgtPadding = 0;
    const int NextI = I + 1;
    if (getParentIndex(ArgTypes[I]) < 0 && NextI < ArgNum &&
        getParentIndex(ArgTypes[NextI]) == I) {
      int64_t Alignment = getPartialStructRequiredAlignment(HstPtrBase);
      TgtPadding = (int64_t)HstPtrBegin % Alignment;
      if (TgtPadding) {
        DP("Using a padding of %" PRId64 " bytes for begin address " DPxMOD
           "\n",
           TgtPadding, DPxPTR(HstPtrBegin));
      }
    }

    // Address of pointer on the host and device, respectively.
    void *PointerHstPtrBegin, *PointerTgtPtrBegin;
    TargetPointerResultTy PointerTpr;
    bool IsHostPtr = false;
    bool IsImplicit = ArgTypes[I] & OMP_TGT_MAPTYPE_IMPLICIT;
    // Force the creation of a device side copy of the data when:
    // a close map modifier was associated with a map that contained a to.
    bool HasCloseModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_CLOSE;
    bool HasPresentModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_PRESENT;
    bool HasHoldModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_OMPX_HOLD;
    // UpdateRef is based on MEMBER_OF instead of TARGET_PARAM because if we
    // have reached this point via __tgt_target_data_begin and not __tgt_target
    // then no argument is marked as TARGET_PARAM ("omp target data map" is not
    // associated with a target region, so there are no target parameters). This
    // may be considered a hack, we could revise the scheme in the future.
    bool UpdateRef =
        !(ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) && !(FromMapper && I == 0);

    MappingInfoTy::HDTTMapAccessorTy HDTTMap =
        Device.getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Has a pointer entry: \n");
      // Base is address of pointer.
      //
      // Usually, the pointer is already allocated by this time.  For example:
      //
      //   #pragma omp target map(s.p[0:N])
      //
      // The map entry for s comes first, and the PTR_AND_OBJ entry comes
      // afterward, so the pointer is already allocated by the time the
      // PTR_AND_OBJ entry is handled below, and PointerTgtPtrBegin is thus
      // non-null.  However, "declare target link" can produce a PTR_AND_OBJ
      // entry for a global that might not already be allocated by the time the
      // PTR_AND_OBJ entry is handled below, and so the allocation might fail
      // when HasPresentModifier.
      PointerTpr = Device.getMappingInfo().getTargetPointer(
          HDTTMap, HstPtrBase, HstPtrBase, /*TgtPadding=*/0, sizeof(void *),
          ArgTypes[I], /*HstPtrName=*/nullptr, /*HasFlagTo=*/false,
          /*HasFlagAlways=*/false, IsImplicit, UpdateRef, HasCloseModifier,
          HasPresentModifier, HasHoldModifier, AsyncInfo, /*OwnedTPR=*/nullptr,
          /*ReleaseHDTTMap=*/false);
      PointerTgtPtrBegin = PointerTpr.TargetPointer;
      IsHostPtr = PointerTpr.Flags.IsHostPointer;
      if (!PointerTgtPtrBegin) {
        REPORT("Call to getTargetPointer returned null pointer (%s).\n",
               HasPresentModifier ? "'present' map type modifier"
                                  : "device failure or illegal mapping");
        return OFFLOAD_FAIL;
      }
      DP("There are %zu bytes allocated at target address " DPxMOD " - is%s new"
         "\n",
         sizeof(void *), DPxPTR(PointerTgtPtrBegin),
         (PointerTpr.Flags.IsNewEntry ? "" : " not"));
      PointerHstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      // No need to update pointee ref count for the first element of the
      // subelement that comes from mapper.
      // subsequently update ref count of pointee
      UpdateRef = (!FromMapper || I != 0);
    }

    const bool HasFlagTo = ArgTypes[I] & OMP_TGT_MAPTYPE_TO;
    const bool HasFlagAlways = ArgTypes[I] & OMP_TGT_MAPTYPE_ALWAYS;
    // Note that HDTTMap will be released in getTargetPointer.
    auto TPR = Device.getMappingInfo().getTargetPointer(
        HDTTMap, HstPtrBegin, HstPtrBase, TgtPadding, DataSize, ArgTypes[I],
        HstPtrName, HasFlagTo, HasFlagAlways, IsImplicit, UpdateRef,
        HasCloseModifier, HasPresentModifier, HasHoldModifier, AsyncInfo,
        PointerTpr.getEntry());
    void *TgtPtrBegin = TPR.TargetPointer;
    IsHostPtr = TPR.Flags.IsHostPointer;
    // If data_size==0, then the argument could be a zero-length pointer to
    // NULL, so getOrAlloc() returning NULL is not an error.
    if (!TgtPtrBegin && (DataSize || HasPresentModifier)) {
      REPORT("Call to getTargetPointer returned null pointer (%s).\n",
             HasPresentModifier ? "'present' map type modifier"
                                : "device failure or illegal mapping");
      return OFFLOAD_FAIL;
    }
    DP("There are %" PRId64 " bytes allocated at %s address " DPxMOD
       " - is%s new\n",
       DataSize, (IsHostPtr ? "host" : "target"), DPxPTR(TgtPtrBegin),
       (TPR.Flags.IsNewEntry ? "" : " not"));

    if (ArgTypes[I] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      uintptr_t Delta = (uintptr_t)HstPtrBegin - (uintptr_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uintptr_t)TgtPtrBegin - Delta);
      DP("Returning device pointer " DPxMOD "\n", DPxPTR(TgtPtrBase));
      ArgsBase[I] = TgtPtrBase;
    }

    // The || part of the if condition covers flang dope vectors that
    // have different host and target addresses when USM is enabled. The
    // pointer to the array is IsHostPtr but the dope vector is not.
    // This happens  with dope vectors in Fortran modules.
    // The pointer has to be copied into the
    // target dope vector.
    // Perhaps OMP_TGT_MAPTYPE_DESCRIPTOR would help here, not sure.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) &&
        (!IsHostPtr || (PointerTpr.getEntry() != nullptr &&
                        PointerHstPtrBegin != PointerTgtPtrBegin)))
      if (prepareAndSubmitData(
              Device, HstPtrBegin, HstPtrBase, TgtPtrBegin, PointerTpr,
              PointerHstPtrBegin, PointerTgtPtrBegin, AsyncInfo,
              ArgTypes[I] & OMP_MAP_DESCRIPTOR_BASE_ADDR) != OFFLOAD_SUCCESS)
        return OFFLOAD_FAIL;

    // Check if variable can be used on the device:
    bool IsStructMember = ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF;
    if (getInfoLevel() & OMP_INFOTYPE_EMPTY_MAPPING && ArgTypes[I] != 0 &&
        !IsStructMember && !IsImplicit && !TPR.isPresent() &&
        !TPR.isContained() && !TPR.isHostPointer())
      INFO(OMP_INFOTYPE_EMPTY_MAPPING, Device.DeviceID,
           "variable %s does not have a valid device counterpart\n",
           (HstPtrName) ? getNameFromMapping(HstPtrName).c_str() : "unknown");
  }

  return OFFLOAD_SUCCESS;
}

namespace {
/// This structure contains information to deallocate a target pointer, aka.
/// used to fix up the shadow map and potentially delete the entry from the
/// mapping table via \p DeviceTy::deallocTgtPtr.
struct PostProcessingInfo {
  /// Host pointer used to look up into the map table
  void *HstPtrBegin;

  /// Size of the data
  int64_t DataSize;

  /// The mapping type (bitfield).
  int64_t ArgType;

  /// The target pointer information.
  TargetPointerResultTy TPR;

  PostProcessingInfo(void *HstPtr, int64_t Size, int64_t ArgType,
                     TargetPointerResultTy &&TPR)
      : HstPtrBegin(HstPtr), DataSize(Size), ArgType(ArgType),
        TPR(std::move(TPR)) {}
};

} // namespace

/// Applies the necessary post-processing procedures to entries listed in \p
/// EntriesInfo after the execution of all device side operations from a target
/// data end. This includes the update of pointers at the host and removal of
/// device buffer when needed. It returns OFFLOAD_FAIL or OFFLOAD_SUCCESS
/// according to the successfulness of the operations.
[[nodiscard]] static int
postProcessingTargetDataEnd(DeviceTy *Device,
                            SmallVector<PostProcessingInfo> &EntriesInfo) {
  int Ret = OFFLOAD_SUCCESS;

  for (auto &[HstPtrBegin, DataSize, ArgType, TPR] : EntriesInfo) {
    bool DelEntry = !TPR.isHostPointer();

    // If the last element from the mapper (for end transfer args comes in
    // reverse order), do not remove the partial entry, the parent struct still
    // exists.
    if ((ArgType & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(ArgType & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    // If we marked the entry to be deleted we need to verify no other
    // thread reused it by now. If deletion is still supposed to happen by
    // this thread LR will be set and exclusive access to the HDTT map
    // will avoid another thread reusing the entry now. Note that we do
    // not request (exclusive) access to the HDTT map if DelEntry is
    // not set.
    MappingInfoTy::HDTTMapAccessorTy HDTTMap =
        Device->getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();

    // We cannot use a lock guard because we may end up delete the mutex.
    // We also explicitly unlocked the entry after it was put in the EntriesInfo
    // so it can be reused.
    TPR.getEntry()->lock();
    auto *Entry = TPR.getEntry();

    const bool IsNotLastUser = Entry->decDataEndThreadCount() != 0;
    if (DelEntry && (Entry->getTotalRefCount() != 0 || IsNotLastUser)) {
      // The thread is not in charge of deletion anymore. Give up access
      // to the HDTT map and unset the deletion flag.
      HDTTMap.destroy();
      DelEntry = false;
    }

    // If we copied back to the host a struct/array containing pointers,
    // we need to restore the original host pointer values from their
    // shadow copies. If the struct is going to be deallocated, remove any
    // remaining shadow pointer entries for this struct.
    const bool HasFrom = ArgType & OMP_TGT_MAPTYPE_FROM;
    if (HasFrom) {
      Entry->foreachShadowPointerInfo([&](const ShadowPtrInfoTy &ShadowPtr) {
        // For Fortran descriptors/dope vectors, it is possible, we have
        // deallocated the data on host and the descriptor persists as it is
        // a separate entity, and we do not want to map back the data to host
        // in these cases when releasing the dope vector.
        // TODO/FIXME: Look into a better longterm solution, such as a different
        // mapping combination for descriptors or performing a similar base
        // address skip that we've done elsewhere in the omptarget runtime.
        const bool isZeroCopy = PM->getRequirements() & OMPX_REQ_AUTO_ZERO_COPY;
        const bool isUSMMode =
            PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY;
        if ((*ShadowPtr.HstPtrAddr == nullptr || isZeroCopy || isUSMMode) &&
            ShadowPtr.IsDescriptorBaseAddr)
          return OFFLOAD_SUCCESS;
        *ShadowPtr.HstPtrAddr = ShadowPtr.HstPtrVal;
        DP("Restoring original host pointer value " DPxMOD " for host "
           "pointer " DPxMOD "\n",
           DPxPTR(ShadowPtr.HstPtrVal), DPxPTR(ShadowPtr.HstPtrAddr));
        return OFFLOAD_SUCCESS;
      });
    }

    // Give up the lock as we either don't need it anymore (e.g., done with
    // TPR), or erase TPR.
    TPR.setEntry(nullptr);

    if (!DelEntry)
      continue;

    Ret = Device->getMappingInfo().eraseMapEntry(HDTTMap, Entry, DataSize);
    // Entry is already remove from the map, we can unlock it now.
    HDTTMap.destroy();
    Ret |= Device->getMappingInfo().deallocTgtPtrAndEntry(Entry, DataSize);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Deallocating data from device failed.\n");
      break;
    }
  }

  delete &EntriesInfo;
  return Ret;
}

/// Internal function to undo the mapping and retrieve the data from the device.
int targetDataEnd(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                  void **ArgBases, void **Args, int64_t *ArgSizes,
                  int64_t *ArgTypes, map_var_info_t *ArgNames,
                  void **ArgMappers, AsyncInfoTy &AsyncInfo, bool FromMapper) {
  int Ret = OFFLOAD_SUCCESS;
  auto *PostProcessingPtrs = new SmallVector<PostProcessingInfo>();
  // process each input.
  for (int32_t I = ArgNum - 1; I >= 0; --I) {
    // Ignore private variables and arrays - there is no mapping for them.
    // Also, ignore the use_device_ptr directive, it has no effect here.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataEnd, call the
      // targetDataMapper variant which will call targetDataEnd again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", I);

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      Ret = targetDataMapper(Loc, Device, ArgBases[I], Args[I], ArgSizes[I],
                             ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                             targetDataEnd);

      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Call to targetDataEnd via targetDataMapper for custom mapper"
               " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = Args[I];
    int64_t DataSize = ArgSizes[I];
    bool IsImplicit = ArgTypes[I] & OMP_TGT_MAPTYPE_IMPLICIT;
    bool UpdateRef = (!(ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
                      (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) &&
                     !(FromMapper && I == 0);
    bool ForceDelete = ArgTypes[I] & OMP_TGT_MAPTYPE_DELETE;
    bool HasPresentModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_PRESENT;
    bool HasHoldModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_OMPX_HOLD;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
        HstPtrBegin, DataSize, UpdateRef, HasHoldModifier, !IsImplicit,
        ForceDelete, /*FromDataEnd=*/true);
    void *TgtPtrBegin = TPR.TargetPointer;
    if (!TPR.isPresent() && !TPR.isHostPointer() &&
        (DataSize || HasPresentModifier)) {
      DP("Mapping does not exist (%s)\n",
         (HasPresentModifier ? "'present' map type modifier" : "ignored"));
      if (HasPresentModifier) {
        // OpenMP 5.1, sec. 2.21.7.1 "map Clause", p. 350 L10-13:
        // "If a map clause appears on a target, target data, target enter data
        // or target exit data construct with a present map-type-modifier then
        // on entry to the region if the corresponding list item does not appear
        // in the device data environment then an error occurs and the program
        // terminates."
        //
        // This should be an error upon entering an "omp target exit data".  It
        // should not be an error upon exiting an "omp target data" or "omp
        // target".  For "omp target data", Clang thus doesn't include present
        // modifiers for end calls.  For "omp target", we have not found a valid
        // OpenMP program for which the error matters: it appears that, if a
        // program can guarantee that data is present at the beginning of an
        // "omp target" region so that there's no error there, that data is also
        // guaranteed to be present at the end.
        MESSAGE("device mapping required by 'present' map type modifier does "
                "not exist for host address " DPxMOD " (%" PRId64 " bytes)",
                DPxPTR(HstPtrBegin), DataSize);
        return OFFLOAD_FAIL;
      }
    } else {
      DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
         " - is%s last\n",
         DataSize, DPxPTR(TgtPtrBegin), (TPR.Flags.IsLast ? "" : " not"));
    }

    // OpenMP 5.1, sec. 2.21.7.1 "map Clause", p. 351 L14-16:
    // "If the map clause appears on a target, target data, or target exit data
    // construct and a corresponding list item of the original list item is not
    // present in the device data environment on exit from the region then the
    // list item is ignored."
    if (!TPR.isPresent())
      continue;

    // Move data back to the host
    const bool HasAlways = ArgTypes[I] & OMP_TGT_MAPTYPE_ALWAYS;
    const bool HasFrom = ArgTypes[I] & OMP_TGT_MAPTYPE_FROM;
    if (HasFrom && (HasAlways || TPR.Flags.IsLast) &&
        !TPR.Flags.IsHostPointer && DataSize != 0) {
      DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
         DataSize, DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
      TIMESCOPE_WITH_DETAILS_AND_IDENT(
          "DevToHost", "Size=" + std::to_string(DataSize) + "B", Loc);
      // Wait for any previous transfer if an event is present.
      if (void *Event = TPR.getEntry()->getEvent()) {
        if (Device.waitEvent(Event, AsyncInfo) != OFFLOAD_SUCCESS) {
          REPORT("Failed to wait for event " DPxMOD ".\n", DPxPTR(Event));
          return OFFLOAD_FAIL;
        }
      }

      Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, DataSize, AsyncInfo,
                                TPR.getEntry());
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Copying data from device failed.\n");
        return OFFLOAD_FAIL;
      }

      // As we are expecting to delete the entry the d2h copy might race
      // with another one that also tries to delete the entry. This happens
      // as the entry can be reused and the reuse might happen after the
      // copy-back was issued but before it completed. Since the reuse might
      // also copy-back a value we would race.
      if (TPR.Flags.IsLast) {
        if (TPR.getEntry()->addEventIfNecessary(Device, AsyncInfo) !=
            OFFLOAD_SUCCESS)
          return OFFLOAD_FAIL;
      }
    }

    // Add pointer to the buffer for post-synchronize processing.
    PostProcessingPtrs->emplace_back(HstPtrBegin, DataSize, ArgTypes[I],
                                     std::move(TPR));
    PostProcessingPtrs->back().TPR.getEntry()->unlock();
  }

  // Add post-processing functions
  // TODO: We might want to remove `mutable` in the future by not changing the
  // captured variables somehow.
  AsyncInfo.addPostProcessingFunction([=, Device = &Device]() mutable -> int {
    return postProcessingTargetDataEnd(Device, *PostProcessingPtrs);
  });

  return Ret;
}

static int targetDataContiguous(ident_t *Loc, DeviceTy &Device, void *ArgsBase,
                                void *HstPtrBegin, int64_t ArgSize,
                                int64_t ArgType, AsyncInfoTy &AsyncInfo) {
  TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
      HstPtrBegin, ArgSize, /*UpdateRefCount=*/false,
      /*UseHoldRefCount=*/false, /*MustContain=*/true);
  void *TgtPtrBegin = TPR.TargetPointer;
  if (!TPR.isPresent()) {
    DP("hst data:" DPxMOD " not found, becomes a noop\n", DPxPTR(HstPtrBegin));
    if (ArgType & OMP_TGT_MAPTYPE_PRESENT) {
      MESSAGE("device mapping required by 'present' motion modifier does not "
              "exist for host address " DPxMOD " (%" PRId64 " bytes)",
              DPxPTR(HstPtrBegin), ArgSize);
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  if (TPR.Flags.IsHostPointer) {
    DP("hst data:" DPxMOD " unified and shared, becomes a noop\n",
       DPxPTR(HstPtrBegin));
    return OFFLOAD_SUCCESS;
  }

  if (ArgType & OMP_TGT_MAPTYPE_TO) {
    DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
       ArgSize, DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
    int Ret = Device.submitData(TgtPtrBegin, HstPtrBegin, ArgSize, AsyncInfo,
                                TPR.getEntry());
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data to device failed.\n");
      return OFFLOAD_FAIL;
    }
    if (TPR.getEntry()) {
      int Ret = TPR.getEntry()->foreachShadowPointerInfo(
          [&](ShadowPtrInfoTy &ShadowPtr) {
            DP("Restoring original target pointer value " DPxMOD " for target "
               "pointer " DPxMOD "\n",
               DPxPTR(ShadowPtr.TgtPtrVal), DPxPTR(ShadowPtr.TgtPtrAddr));
            Ret = Device.submitData(ShadowPtr.TgtPtrAddr,
                                    (void *)&ShadowPtr.TgtPtrVal,
                                    sizeof(void *), AsyncInfo);
            if (Ret != OFFLOAD_SUCCESS) {
              REPORT("Copying data to device failed.\n");
              return OFFLOAD_FAIL;
            }
            return OFFLOAD_SUCCESS;
          });
      if (Ret != OFFLOAD_SUCCESS) {
        DP("Updating shadow map failed\n");
        return Ret;
      }
    }
  }

  if (ArgType & OMP_TGT_MAPTYPE_FROM) {
    DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
       ArgSize, DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
    int Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, ArgSize, AsyncInfo,
                                  TPR.getEntry());
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data from device failed.\n");
      return OFFLOAD_FAIL;
    }

    // Wait for device-to-host memcopies for whole struct to complete,
    // before restoring the correct host pointer.
    if (auto *Entry = TPR.getEntry()) {
      AsyncInfo.addPostProcessingFunction([=]() -> int {
        int Ret = Entry->foreachShadowPointerInfo(
            [&](const ShadowPtrInfoTy &ShadowPtr) {
              // For Fortran descriptors/dope vectors, it is possible, we have
              // deallocated the data on host and the descriptor persists as it
              // is a separate entity, and we do not want to map back the data
              // to host in these cases when releasing the dope vector.
              // TODO/FIXME: Look into a better longterm solution, such as a
              // different mapping combination for descriptors or performing a
              // similar base address skip that we've done elsewhere in the
              // omptarget runtime.
              const bool isZeroCopy =
                  PM->getRequirements() & OMPX_REQ_AUTO_ZERO_COPY;
              const bool isUSMMode =
                  PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY;
              if ((*ShadowPtr.HstPtrAddr == nullptr || isZeroCopy ||
                   isUSMMode) &&
                  ShadowPtr.IsDescriptorBaseAddr)
                return OFFLOAD_SUCCESS;
              *ShadowPtr.HstPtrAddr = ShadowPtr.HstPtrVal;
              DP("Restoring original host pointer value " DPxMOD
                 " for host pointer " DPxMOD "\n",
                 DPxPTR(ShadowPtr.HstPtrVal), DPxPTR(ShadowPtr.HstPtrAddr));
              return OFFLOAD_SUCCESS;
            });
        Entry->unlock();
        if (Ret != OFFLOAD_SUCCESS) {
          DP("Updating shadow map failed\n");
          return Ret;
        }
        return OFFLOAD_SUCCESS;
      });
    }
  }

  return OFFLOAD_SUCCESS;
}

static int targetDataNonContiguous(ident_t *Loc, DeviceTy &Device,
                                   void *ArgsBase,
                                   __tgt_target_non_contig *NonContig,
                                   uint64_t Size, int64_t ArgType,
                                   int CurrentDim, int DimSize, uint64_t Offset,
                                   AsyncInfoTy &AsyncInfo) {
  int Ret = OFFLOAD_SUCCESS;
  if (CurrentDim < DimSize) {
    for (unsigned int I = 0; I < NonContig[CurrentDim].Count; ++I) {
      uint64_t CurOffset =
          (NonContig[CurrentDim].Offset + I) * NonContig[CurrentDim].Stride;
      // we only need to transfer the first element for the last dimension
      // since we've already got a contiguous piece.
      if (CurrentDim != DimSize - 1 || I == 0) {
        Ret = targetDataNonContiguous(Loc, Device, ArgsBase, NonContig, Size,
                                      ArgType, CurrentDim + 1, DimSize,
                                      Offset + CurOffset, AsyncInfo);
        // Stop the whole process if any contiguous piece returns anything
        // other than OFFLOAD_SUCCESS.
        if (Ret != OFFLOAD_SUCCESS)
          return Ret;
      }
    }
  } else {
    char *Ptr = (char *)ArgsBase + Offset;
    DP("Transfer of non-contiguous : host ptr " DPxMOD " offset %" PRIu64
       " len %" PRIu64 "\n",
       DPxPTR(Ptr), Offset, Size);
    Ret = targetDataContiguous(Loc, Device, ArgsBase, Ptr, Size, ArgType,
                               AsyncInfo);
  }
  return Ret;
}

static int getNonContigMergedDimension(__tgt_target_non_contig *NonContig,
                                       int32_t DimSize) {
  int RemovedDim = 0;
  for (int I = DimSize - 1; I > 0; --I) {
    if (NonContig[I].Count * NonContig[I].Stride == NonContig[I - 1].Stride)
      RemovedDim++;
  }
  return RemovedDim;
}

/// Internal function to pass data to/from the target.
int targetDataUpdate(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                     void **ArgsBase, void **Args, int64_t *ArgSizes,
                     int64_t *ArgTypes, map_var_info_t *ArgNames,
                     void **ArgMappers, AsyncInfoTy &AsyncInfo, bool) {
  // process each input.
  for (int32_t I = 0; I < ArgNum; ++I) {
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataUpdate, call the
      // targetDataMapper variant which will call targetDataUpdate again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", I);

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      int Ret = targetDataMapper(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                 ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                                 targetDataUpdate);

      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Call to targetDataUpdate via targetDataMapper for custom mapper"
               " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    int Ret = OFFLOAD_SUCCESS;

    if (ArgTypes[I] & OMP_TGT_MAPTYPE_NON_CONTIG) {
      __tgt_target_non_contig *NonContig = (__tgt_target_non_contig *)Args[I];
      int32_t DimSize = ArgSizes[I];
      uint64_t Size =
          NonContig[DimSize - 1].Count * NonContig[DimSize - 1].Stride;
      int32_t MergedDim = getNonContigMergedDimension(NonContig, DimSize);
      Ret = targetDataNonContiguous(
          Loc, Device, ArgsBase[I], NonContig, Size, ArgTypes[I],
          /*current_dim=*/0, DimSize - MergedDim, /*offset=*/0, AsyncInfo);
    } else {
      Ret = targetDataContiguous(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                 ArgTypes[I], AsyncInfo);
    }
    if (Ret == OFFLOAD_FAIL)
      return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

static const unsigned LambdaMapping = OMP_TGT_MAPTYPE_PTR_AND_OBJ |
                                      OMP_TGT_MAPTYPE_LITERAL |
                                      OMP_TGT_MAPTYPE_IMPLICIT;
static bool isLambdaMapping(int64_t Mapping) {
  return (Mapping & LambdaMapping) == LambdaMapping;
}

namespace {
/// Find the table information in the map or look it up in the translation
/// tables.
TableMap *getTableMap(void *HostPtr) {
  std::lock_guard<std::mutex> TblMapLock(PM->TblMapMtx);
  HostPtrToTableMapTy::iterator TableMapIt =
      PM->HostPtrToTableMap.find(HostPtr);

  if (TableMapIt != PM->HostPtrToTableMap.end())
    return &TableMapIt->second;

  // We don't have a map. So search all the registered libraries.
  TableMap *TM = nullptr;
  std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
  for (HostEntriesBeginToTransTableTy::iterator Itr =
           PM->HostEntriesBeginToTransTable.begin();
       Itr != PM->HostEntriesBeginToTransTable.end(); ++Itr) {
    // get the translation table (which contains all the good info).
    TranslationTable *TransTable = &Itr->second;
    // iterate over all the host table entries to see if we can locate the
    // host_ptr.
    llvm::offloading::EntryTy *Cur = TransTable->HostTable.EntriesBegin;
    for (uint32_t I = 0; Cur < TransTable->HostTable.EntriesEnd; ++Cur, ++I) {
      if (Cur->Address != HostPtr)
        continue;
      // we got a match, now fill the HostPtrToTableMap so that we
      // may avoid this search next time.
      TM = &(PM->HostPtrToTableMap)[HostPtr];
      TM->Table = TransTable;
      TM->Index = I;
      return TM;
    }
  }

  return nullptr;
}

/// A class manages private arguments in a target region.
class PrivateArgumentManagerTy {
  /// A data structure for the information of first-private arguments. We can
  /// use this information to optimize data transfer by packing all
  /// first-private arguments and transfer them all at once.
  struct FirstPrivateArgInfoTy {
    /// Host pointer begin
    char *HstPtrBegin;
    /// Host pointer end
    char *HstPtrEnd;
    /// The index of the element in \p TgtArgs corresponding to the argument
    int Index;
    /// Alignment of the entry (base of the entry, not after the entry).
    uint32_t Alignment;
    /// Size (without alignment, see padding)
    uint32_t Size;
    /// Padding used to align this argument entry, if necessary.
    uint32_t Padding;
    /// Host pointer name
    map_var_info_t HstPtrName = nullptr;

    FirstPrivateArgInfoTy(int Index, void *HstPtr, uint32_t Size,
                          uint32_t Alignment, uint32_t Padding,
                          map_var_info_t HstPtrName = nullptr)
        : HstPtrBegin(reinterpret_cast<char *>(HstPtr)),
          HstPtrEnd(HstPtrBegin + Size), Index(Index), Alignment(Alignment),
          Size(Size), Padding(Padding), HstPtrName(HstPtrName) {}
  };

  /// A vector of target pointers for all private arguments
  std::vector<void *> TgtPtrs;

  /// A vector of information of all first-private arguments to be packed
  std::vector<FirstPrivateArgInfoTy> FirstPrivateArgInfo;
  /// Host buffer for all arguments to be packed
  std::vector<char> FirstPrivateArgBuffer;
  /// The total size of all arguments to be packed
  int64_t FirstPrivateArgSize = 0;

  /// A reference to the \p DeviceTy object
  DeviceTy &Device;
  /// A pointer to a \p AsyncInfoTy object
  AsyncInfoTy &AsyncInfo;

  // TODO: What would be the best value here? Should we make it configurable?
  // If the size is larger than this threshold, we will allocate and transfer it
  // immediately instead of packing it.
  static constexpr const int64_t FirstPrivateArgSizeThreshold = 1024;

public:
  /// Constructor
  PrivateArgumentManagerTy(DeviceTy &Dev, AsyncInfoTy &AsyncInfo)
      : Device(Dev), AsyncInfo(AsyncInfo) {}

  /// Add a private argument
  int addArg(void *HstPtr, int64_t ArgSize, int64_t ArgOffset,
             bool IsFirstPrivate, void *&TgtPtr, int TgtArgsIndex,
             map_var_info_t HstPtrName = nullptr,
             const bool AllocImmediately = false) {
    // If the argument is not first-private, or its size is greater than a
    // predefined threshold, we will allocate memory and issue the transfer
    // immediately.
    if (ArgSize > FirstPrivateArgSizeThreshold || !IsFirstPrivate ||
        AllocImmediately) {
      TgtPtr = Device.allocData(ArgSize, HstPtr);
      if (!TgtPtr) {
        DP("Data allocation for %sprivate array " DPxMOD " failed.\n",
           (IsFirstPrivate ? "first-" : ""), DPxPTR(HstPtr));
        return OFFLOAD_FAIL;
      }
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtr + ArgOffset);
      DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD
         " for %sprivate array " DPxMOD " - pushing target argument " DPxMOD
         "\n",
         ArgSize, DPxPTR(TgtPtr), (IsFirstPrivate ? "first-" : ""),
         DPxPTR(HstPtr), DPxPTR(TgtPtrBase));
#endif
      // If first-private, copy data from host
      if (IsFirstPrivate) {
        DP("Submitting firstprivate data to the device.\n");
        int Ret = Device.submitData(TgtPtr, HstPtr, ArgSize, AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed, failed.\n");
          return OFFLOAD_FAIL;
        }
      }
      TgtPtrs.push_back(TgtPtr);
    } else {
      DP("Firstprivate array " DPxMOD " of size %" PRId64 " will be packed\n",
         DPxPTR(HstPtr), ArgSize);
      // When reach this point, the argument must meet all following
      // requirements:
      // 1. Its size does not exceed the threshold (see the comment for
      // FirstPrivateArgSizeThreshold);
      // 2. It must be first-private (needs to be mapped to target device).
      // We will pack all this kind of arguments to transfer them all at once
      // to reduce the number of data transfer. We will not take
      // non-first-private arguments, aka. private arguments that doesn't need
      // to be mapped to target device, into account because data allocation
      // can be very efficient with memory manager.

      // Placeholder value
      TgtPtr = nullptr;
      auto *LastFPArgInfo =
          FirstPrivateArgInfo.empty() ? nullptr : &FirstPrivateArgInfo.back();

      // Compute the start alignment of this entry, add padding if necessary.
      // TODO: Consider sorting instead.
      uint32_t Padding = 0;
      uint32_t StartAlignment =
          LastFPArgInfo ? LastFPArgInfo->Alignment : MaxAlignment;
      if (LastFPArgInfo) {
        // Check if we keep the start alignment or if it is shrunk due to the
        // size of the last element.
        uint32_t Offset = LastFPArgInfo->Size % StartAlignment;
        if (Offset)
          StartAlignment = Offset;
        // We only need as much alignment as the host pointer had (since we
        // don't know the alignment information from the source we might end up
        // overaligning accesses but not too much).
        uint32_t RequiredAlignment =
            llvm::bit_floor(getPartialStructRequiredAlignment(HstPtr));
        if (RequiredAlignment > StartAlignment) {
          Padding = RequiredAlignment - StartAlignment;
          StartAlignment = RequiredAlignment;
        }
      }

      FirstPrivateArgInfo.emplace_back(TgtArgsIndex, HstPtr, ArgSize,
                                       StartAlignment, Padding, HstPtrName);
      FirstPrivateArgSize += Padding + ArgSize;
    }

    return OFFLOAD_SUCCESS;
  }

  /// Pack first-private arguments, replace place holder pointers in \p TgtArgs,
  /// and start the transfer.
  int packAndTransfer(SmallVector<void *> &TgtArgs) {
    if (!FirstPrivateArgInfo.empty()) {
      assert(FirstPrivateArgSize != 0 &&
             "FirstPrivateArgSize is 0 but FirstPrivateArgInfo is empty");
      FirstPrivateArgBuffer.resize(FirstPrivateArgSize, 0);
      auto Itr = FirstPrivateArgBuffer.begin();
      // Copy all host data to this buffer
      for (FirstPrivateArgInfoTy &Info : FirstPrivateArgInfo) {
        // First pad the pointer as we (have to) pad it on the device too.
        Itr = std::next(Itr, Info.Padding);
        std::copy(Info.HstPtrBegin, Info.HstPtrEnd, Itr);
        Itr = std::next(Itr, Info.Size);
      }
      // Allocate target memory
      void *TgtPtr =
          Device.allocData(FirstPrivateArgSize, FirstPrivateArgBuffer.data());
      if (TgtPtr == nullptr) {
        DP("Failed to allocate target memory for private arguments.\n");
        return OFFLOAD_FAIL;
      }
      TgtPtrs.push_back(TgtPtr);
      DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD "\n",
         FirstPrivateArgSize, DPxPTR(TgtPtr));
      // Transfer data to target device
      int Ret = Device.submitData(TgtPtr, FirstPrivateArgBuffer.data(),
                                  FirstPrivateArgSize, AsyncInfo);
      if (Ret != OFFLOAD_SUCCESS) {
        DP("Failed to submit data of private arguments.\n");
        return OFFLOAD_FAIL;
      }
      // Fill in all placeholder pointers
      auto TP = reinterpret_cast<uintptr_t>(TgtPtr);
      for (FirstPrivateArgInfoTy &Info : FirstPrivateArgInfo) {
        void *&Ptr = TgtArgs[Info.Index];
        assert(Ptr == nullptr && "Target pointer is already set by mistaken");
        // Pad the device pointer to get the right alignment.
        TP += Info.Padding;
        Ptr = reinterpret_cast<void *>(TP);
        TP += Info.Size;
        DP("Firstprivate array " DPxMOD " of size %" PRId64 " mapped to " DPxMOD
           "\n",
           DPxPTR(Info.HstPtrBegin), Info.HstPtrEnd - Info.HstPtrBegin,
           DPxPTR(Ptr));
      }
    }

    return OFFLOAD_SUCCESS;
  }

  /// Free all target memory allocated for private arguments
  int free() {
    for (void *P : TgtPtrs) {
      int Ret = Device.deleteData(P);
      if (Ret != OFFLOAD_SUCCESS) {
        DP("Deallocation of (first-)private arrays failed.\n");
        return OFFLOAD_FAIL;
      }
    }

    TgtPtrs.clear();

    return OFFLOAD_SUCCESS;
  }
};

/// Process data before launching the kernel, including calling targetDataBegin
/// to map and transfer data to target device, transferring (first-)private
/// variables.
static int processDataBefore(ident_t *Loc, int64_t DeviceId, void *HostPtr,
                             int32_t ArgNum, void **ArgBases, void **Args,
                             int64_t *ArgSizes, int64_t *ArgTypes,
                             map_var_info_t *ArgNames, void **ArgMappers,
                             SmallVector<void *> &TgtArgs,
                             SmallVector<ptrdiff_t> &TgtOffsets,
                             PrivateArgumentManagerTy &PrivateArgumentManager,
                             AsyncInfoTy &AsyncInfo) {

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  int Ret = targetDataBegin(Loc, *DeviceOrErr, ArgNum, ArgBases, Args, ArgSizes,
                            ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Call to targetDataBegin failed, abort target.\n");
    return OFFLOAD_FAIL;
  }

  // List of (first-)private arrays allocated for this target region
  std::vector<int> TgtArgsPositions(ArgNum, -1);

  for (int32_t I = 0; I < ArgNum; ++I) {
    if (!(ArgTypes[I] & OMP_TGT_MAPTYPE_TARGET_PARAM)) {
      // This is not a target parameter, do not push it into TgtArgs.
      // Check for lambda mapping.
      if (isLambdaMapping(ArgTypes[I])) {
        assert((ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
               "PTR_AND_OBJ must be also MEMBER_OF.");
        unsigned Idx = getParentIndex(ArgTypes[I]);
        int TgtIdx = TgtArgsPositions[Idx];
        assert(TgtIdx != -1 && "Base address must be translated already.");
        // The parent lambda must be processed already and it must be the last
        // in TgtArgs and TgtOffsets arrays.
        void *HstPtrVal = Args[I];
        void *HstPtrBegin = ArgBases[I];
        void *HstPtrBase = Args[Idx];
        void *TgtPtrBase =
            (void *)((intptr_t)TgtArgs[TgtIdx] + TgtOffsets[TgtIdx]);
        DP("Parent lambda base " DPxMOD "\n", DPxPTR(TgtPtrBase));
        uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
        void *TgtPtrBegin = (void *)((uintptr_t)TgtPtrBase + Delta);
        void *&PointerTgtPtrBegin = AsyncInfo.getVoidPtrLocation();
        TargetPointerResultTy TPR =
            DeviceOrErr->getMappingInfo().getTgtPtrBegin(
                HstPtrVal, ArgSizes[I], /*UpdateRefCount=*/false,
                /*UseHoldRefCount=*/false);
        PointerTgtPtrBegin = TPR.TargetPointer;
        if (!TPR.isPresent()) {
          DP("No lambda captured variable mapped (" DPxMOD ") - ignored\n",
             DPxPTR(HstPtrVal));
          continue;
        }
        if (TPR.Flags.IsHostPointer) {
          DP("Unified memory is active, no need to map lambda captured"
             "variable (" DPxMOD ")\n",
             DPxPTR(HstPtrVal));
          continue;
        }
        DP("Update lambda reference (" DPxMOD ") -> [" DPxMOD "]\n",
           DPxPTR(PointerTgtPtrBegin), DPxPTR(TgtPtrBegin));
        Ret =
            DeviceOrErr->submitData(TgtPtrBegin, &PointerTgtPtrBegin,
                                    sizeof(void *), AsyncInfo, TPR.getEntry());
        if (Ret != OFFLOAD_SUCCESS) {
          REPORT("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
      continue;
    }
    void *HstPtrBegin = Args[I];
    void *HstPtrBase = ArgBases[I];
    void *TgtPtrBegin;
    map_var_info_t HstPtrName = (!ArgNames) ? nullptr : ArgNames[I];
    ptrdiff_t TgtBaseOffset;
    TargetPointerResultTy TPR;
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) {
      DP("Forwarding first-private value " DPxMOD " to the target construct\n",
         DPxPTR(HstPtrBase));
      TgtPtrBegin = HstPtrBase;
      TgtBaseOffset = 0;
    } else if (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE) {
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
      const bool IsFirstPrivate = (ArgTypes[I] & OMP_TGT_MAPTYPE_TO);
      // If there is a next argument and it depends on the current one, we need
      // to allocate the private memory immediately. If this is not the case,
      // then the argument can be marked for optimization and packed with the
      // other privates.
      const bool AllocImmediately =
          (I < ArgNum - 1 && (ArgTypes[I + 1] & OMP_TGT_MAPTYPE_MEMBER_OF));
      Ret = PrivateArgumentManager.addArg(
          HstPtrBegin, ArgSizes[I], TgtBaseOffset, IsFirstPrivate, TgtPtrBegin,
          TgtArgs.size(), HstPtrName, AllocImmediately);
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Failed to process %sprivate argument " DPxMOD "\n",
               (IsFirstPrivate ? "first-" : ""), DPxPTR(HstPtrBegin));
        return OFFLOAD_FAIL;
      }
    } else {
      if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
        HstPtrBase = *reinterpret_cast<void **>(HstPtrBase);
      TPR = DeviceOrErr->getMappingInfo().getTgtPtrBegin(
          HstPtrBegin, ArgSizes[I],
          /*UpdateRefCount=*/false,
          /*UseHoldRefCount=*/false);
      TgtPtrBegin = TPR.TargetPointer;
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD "\n",
         DPxPTR(TgtPtrBase), DPxPTR(HstPtrBegin));
#endif
    }
    TgtArgsPositions[I] = TgtArgs.size();
    TgtArgs.push_back(TgtPtrBegin);
    TgtOffsets.push_back(TgtBaseOffset);
  }

  assert(TgtArgs.size() == TgtOffsets.size() &&
         "Size mismatch in arguments and offsets");

  // Pack and transfer first-private arguments
  Ret = PrivateArgumentManager.packAndTransfer(TgtArgs);
  if (Ret != OFFLOAD_SUCCESS) {
    DP("Failed to pack and transfer first private arguments\n");
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

/// Process data after launching the kernel, including transferring data back to
/// host if needed and deallocating target memory of (first-)private variables.
static int processDataAfter(ident_t *Loc, int64_t DeviceId, void *HostPtr,
                            int32_t ArgNum, void **ArgBases, void **Args,
                            int64_t *ArgSizes, int64_t *ArgTypes,
                            map_var_info_t *ArgNames, void **ArgMappers,
                            PrivateArgumentManagerTy &PrivateArgumentManager,
                            AsyncInfoTy &AsyncInfo) {

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  // Move data from device.
  int Ret = targetDataEnd(Loc, *DeviceOrErr, ArgNum, ArgBases, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Call to targetDataEnd failed, abort target.\n");
    return OFFLOAD_FAIL;
  }

  // Free target memory for private arguments after synchronization.
  // TODO: We might want to remove `mutable` in the future by not changing the
  // captured variables somehow.
  AsyncInfo.addPostProcessingFunction(
      [PrivateArgumentManager =
           std::move(PrivateArgumentManager)]() mutable -> int {
        int Ret = PrivateArgumentManager.free();
        if (Ret != OFFLOAD_SUCCESS) {
          REPORT("Failed to deallocate target memory for private args\n");
          return OFFLOAD_FAIL;
        }
        return Ret;
      });

  return OFFLOAD_SUCCESS;
}
} // namespace

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// returns 0 if it was able to transfer the execution to a target and an
/// integer different from zero otherwise.
int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
           KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo,
           bool InMultiDeviceMode, bool &IsMultiDeviceKernel) {
  int32_t DeviceId = Device.DeviceID;
  TableMap *TM = getTableMap(HostPtr);
  // No map for this host pointer found!
  if (!TM) {
    REPORT("Host ptr " DPxMOD " does not have a matching target pointer.\n",
           DPxPTR(HostPtr));
    return OFFLOAD_FAIL;
  }

  // get target table.
  __tgt_target_table *TargetTable = nullptr;
  {
    std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
    assert(TM->Table->TargetsTable.size() > (size_t)DeviceId &&
           "Not expecting a device ID outside the table's bounds!");
    TargetTable = TM->Table->TargetsTable[DeviceId];
  }
  assert(TargetTable && "Global data has not been mapped\n");

  DP("loop trip count is %" PRIu64 ".\n", KernelArgs.Tripcount);

  // We need to keep bases and offsets separate. Sometimes (e.g. in OpenCL) we
  // need to manifest base pointers prior to launching a kernel. Even if we have
  // mapped an object only partially, e.g. A[N:M], although the kernel is
  // expected to access elements starting at address &A[N] and beyond, we still
  // need to manifest the base of the array &A[0]. In other cases, e.g. the COI
  // API, we need the begin address itself, i.e. &A[N], as the API operates on
  // begin addresses, not bases. That's why we pass args and offsets as two
  // separate entities so that each plugin can do what it needs. This behavior
  // was introduced via https://reviews.llvm.org/D33028 and commit 1546d319244c.
  SmallVector<void *> TgtArgs;
  SmallVector<ptrdiff_t> TgtOffsets;

  PrivateArgumentManagerTy PrivateArgumentManager(Device, AsyncInfo);

  int NumClangLaunchArgs = KernelArgs.NumArgs;
  int Ret = OFFLOAD_SUCCESS;
  if (NumClangLaunchArgs) {
    // Process data, such as data mapping, before launching the kernel
    Ret = processDataBefore(Loc, DeviceId, HostPtr, NumClangLaunchArgs,
                            KernelArgs.ArgBasePtrs, KernelArgs.ArgPtrs,
                            KernelArgs.ArgSizes, KernelArgs.ArgTypes,
                            KernelArgs.ArgNames, KernelArgs.ArgMappers, TgtArgs,
                            TgtOffsets, PrivateArgumentManager, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Failed to process data before launching the kernel.\n");
      return OFFLOAD_FAIL;
    }

    // Clang might pass more values via the ArgPtrs to the runtime that we pass
    // on to the kernel.
    // TODO: Next time we adjust the KernelArgsTy we should introduce a new
    // NumKernelArgs field.
    KernelArgs.NumArgs = TgtArgs.size();
  }

  // Launch device execution.
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].Address;
  DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
     TargetTable->EntriesBegin[TM->Index].SymbolName, DPxPTR(TgtEntryPtr),
     TM->Index);

  {
    assert(KernelArgs.NumArgs == TgtArgs.size() && "Argument count mismatch!");
    TIMESCOPE_WITH_DETAILS_AND_IDENT(
        "Kernel Target",
        "NumArguments=" + std::to_string(KernelArgs.NumArgs) +
            ";NumTeams=" + std::to_string(KernelArgs.NumTeams[0]) +
            ";TripCount=" + std::to_string(KernelArgs.Tripcount),
        Loc);

#ifdef OMPT_SUPPORT
    /// RAII to establish tool anchors before and after kernel launch
    int32_t NumTeams = KernelArgs.NumTeams[0];
    // No need to guard this with OMPT_IF_BUILT
    InterfaceRAII TargetSubmitRAII(
        RegionInterface.getCallbacks<ompt_callback_target_submit>(), NumTeams);

    // Calls "begin" for the OMPT trace record and let the plugin
    // enqueue the stop operation for after the kernel is done. The stop
    // operation completes the trace record entry with the information from
    // within the plugin, eg., kernel timing info.
    // Only if 'TracedDeviceId' is actually traced, AsyncInfo->OmptEventInfo is
    // set and a trace record generated. Otherwise: No OMPT device tracing.
    TracerInterfaceRAII TargetTraceRAII(
        RegionInterface.getTraceGenerators<ompt_callback_target_submit>(),
        AsyncInfo, /*TracedDeviceId=*/DeviceId,
        /*EventType=*/ompt_callback_target_submit, DeviceId, NumTeams);
#endif
    Ret = Device.launchKernel(TgtEntryPtr, TgtArgs.data(), TgtOffsets.data(),
                              KernelArgs, AsyncInfo);

    // If we are in multidevice mode the check the value of the global variable
    // for this kernel to see if the kernel is indeed a multi device kernel.
    if (InMultiDeviceMode)
      IsMultiDeviceKernel = Device.isMultiDeviceKernel(TgtEntryPtr);
  }

  // Reset number of arguments just in case the kernel launch changed it.
  KernelArgs.NumArgs = NumClangLaunchArgs;

  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Executing target region abort target.\n");
    return OFFLOAD_FAIL;
  }

  if (NumClangLaunchArgs) {
    // Transfer data back and deallocate target memory for (first-)private
    // variables
    Ret = processDataAfter(Loc, DeviceId, HostPtr, NumClangLaunchArgs,
                           KernelArgs.ArgBasePtrs, KernelArgs.ArgPtrs,
                           KernelArgs.ArgSizes, KernelArgs.ArgTypes,
                           KernelArgs.ArgNames, KernelArgs.ArgMappers,
                           PrivateArgumentManager, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Failed to process data after launching the kernel.\n");
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}

/// Enables the record replay mechanism by pre-allocating MemorySize
/// and informing the record-replayer of whether to store the output
/// in some file.
int target_activate_rr(DeviceTy &Device, uint64_t MemorySize, void *VAddr,
                       bool IsRecord, bool SaveOutput,
                       uint64_t &ReqPtrArgOffset) {
  return Device.RTL->initialize_record_replay(Device.DeviceID, MemorySize,
                                              VAddr, IsRecord, SaveOutput,
                                              ReqPtrArgOffset);
}

/// Executes a kernel using pre-recorded information for loading to
/// device memory to launch the target kernel with the pre-recorded
/// configuration.
int target_replay(ident_t *Loc, DeviceTy &Device, void *HostPtr,
                  void *DeviceMemory, int64_t DeviceMemorySize, void **TgtArgs,
                  ptrdiff_t *TgtOffsets, int32_t NumArgs, int32_t NumTeams,
                  int32_t ThreadLimit, uint64_t LoopTripCount,
                  AsyncInfoTy &AsyncInfo) {
  int32_t DeviceId = Device.DeviceID;
  TableMap *TM = getTableMap(HostPtr);
  // Fail if the table map fails to find the target kernel pointer for the
  // provided host pointer.
  if (!TM) {
    REPORT("Host ptr " DPxMOD " does not have a matching target pointer.\n",
           DPxPTR(HostPtr));
    return OFFLOAD_FAIL;
  }

  // Retrieve the target table of offloading entries.
  __tgt_target_table *TargetTable = nullptr;
  {
    std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
    assert(TM->Table->TargetsTable.size() > (size_t)DeviceId &&
           "Not expecting a device ID outside the table's bounds!");
    TargetTable = TM->Table->TargetsTable[DeviceId];
  }
  assert(TargetTable && "Global data has not been mapped\n");

  // Retrieve the target kernel pointer, allocate and store the recorded device
  // memory data, and launch device execution.
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].Address;
  DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
     TargetTable->EntriesBegin[TM->Index].SymbolName, DPxPTR(TgtEntryPtr),
     TM->Index);

  void *TgtPtr = Device.allocData(DeviceMemorySize, /*HstPtr=*/nullptr,
                                  TARGET_ALLOC_DEFAULT);
  Device.submitData(TgtPtr, DeviceMemory, DeviceMemorySize, AsyncInfo);

  KernelArgsTy KernelArgs{};
  KernelArgs.Version = OMP_KERNEL_ARG_VERSION;
  KernelArgs.NumArgs = NumArgs;
  KernelArgs.Tripcount = LoopTripCount;
  KernelArgs.NumTeams[0] = NumTeams;
  KernelArgs.ThreadLimit[0] = ThreadLimit;

  int Ret = Device.launchKernel(TgtEntryPtr, TgtArgs, TgtOffsets, KernelArgs,
                                AsyncInfo);

  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Executing target region abort target.\n");
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}
