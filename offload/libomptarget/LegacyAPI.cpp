//===-------- LegacyAPI.cpp - Target independent OpenMP target RTL --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legacy interfaces for libomptarget used to maintain backwards-compatibility.
//
//===----------------------------------------------------------------------===//

#include "OpenMP/OMPT/Interface.h"
#include "OpenMP/OMPT/OmptCommonDefs.h"
#include "omptarget.h"
#include "private.h"

#include "Shared/Profile.h"

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif

EXTERN void __tgt_target_data_begin(int64_t DeviceId, int32_t ArgNum,
                                    void **ArgsBase, void **Args,
                                    int64_t *ArgSizes, int64_t *ArgTypes) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  __tgt_target_data_begin_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                 ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_begin_nowait(int64_t DeviceId, int32_t ArgNum,
                                           void **ArgsBase, void **Args,
                                           int64_t *ArgSizes, int64_t *ArgTypes,
                                           int32_t DepNum, void *DepList,
                                           int32_t NoAliasDepNum,
                                           void *NoAliasDepList) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  __tgt_target_data_begin_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                 ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end(int64_t DeviceId, int32_t ArgNum,
                                  void **ArgsBase, void **Args,
                                  int64_t *ArgSizes, int64_t *ArgTypes) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  __tgt_target_data_end_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                               ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_update(int64_t DeviceId, int32_t ArgNum,
                                     void **ArgsBase, void **Args,
                                     int64_t *ArgSizes, int64_t *ArgTypes) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  __tgt_target_data_update_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                  ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_update_nowait(
    int64_t DeviceId, int32_t ArgNum, void **ArgsBase, void **Args,
    int64_t *ArgSizes, int64_t *ArgTypes, int32_t DepNum, void *DepList,
    int32_t NoAliasDepNum, void *NoAliasDepList) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  __tgt_target_data_update_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                  ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end_nowait(int64_t DeviceId, int32_t ArgNum,
                                         void **ArgsBase, void **Args,
                                         int64_t *ArgSizes, int64_t *ArgTypes,
                                         int32_t DepNum, void *DepList,
                                         int32_t NoAliasDepNum,
                                         void *NoAliasDepList) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  __tgt_target_data_end_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                               ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN int __tgt_target_mapper(ident_t *Loc, int64_t DeviceId, void *HostPtr,
                               uint32_t ArgNum, void **ArgsBase, void **Args,
                               int64_t *ArgSizes, int64_t *ArgTypes,
                               map_var_info_t *ArgNames, void **ArgMappers) {
  TIMESCOPE_WITH_IDENT(Loc);
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  KernelArgsTy KernelArgs{1,        ArgNum,   ArgsBase,   Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, 0,    {},
                          {},       {},       0};
  return __tgt_target_kernel(Loc, DeviceId, -1, -1, HostPtr, &KernelArgs);
}

EXTERN int __tgt_target(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                        void **ArgsBase, void **Args, int64_t *ArgSizes,
                        int64_t *ArgTypes) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return __tgt_target_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase, Args,
                             ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN int __tgt_target_nowait(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                               void **ArgsBase, void **Args, int64_t *ArgSizes,
                               int64_t *ArgTypes, int32_t DepNum, void *DepList,
                               int32_t NoAliasDepNum, void *NoAliasDepList) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return __tgt_target_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase, Args,
                             ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN int __tgt_target_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, void *HostPtr, int32_t ArgNum,
    void **ArgsBase, void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
    map_var_info_t *ArgNames, void **ArgMappers, int32_t DepNum, void *DepList,
    int32_t NoAliasDepNum, void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return __tgt_target_mapper(Loc, DeviceId, HostPtr, ArgNum, ArgsBase, Args,
                             ArgSizes, ArgTypes, ArgNames, ArgMappers);
}

EXTERN int __tgt_target_teams_mapper(ident_t *Loc, int64_t DeviceId,
                                     void *HostPtr, uint32_t ArgNum,
                                     void **ArgsBase, void **Args,
                                     int64_t *ArgSizes, int64_t *ArgTypes,
                                     map_var_info_t *ArgNames,
                                     void **ArgMappers, int32_t NumTeams,
                                     int32_t ThreadLimit) {
  TIMESCOPE_WITH_IDENT(Loc);
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  KernelArgsTy KernelArgs{1,        ArgNum,   ArgsBase,   Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, 0,    {},
                          {},       {},       0};
  return __tgt_target_kernel(Loc, DeviceId, NumTeams, ThreadLimit, HostPtr,
                             &KernelArgs);
}

EXTERN int __tgt_target_teams(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                              void **ArgsBase, void **Args, int64_t *ArgSizes,
                              int64_t *ArgTypes, int32_t NumTeams,
                              int32_t ThreadLimit) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return __tgt_target_teams_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase,
                                   Args, ArgSizes, ArgTypes, nullptr, nullptr,
                                   NumTeams, ThreadLimit);
}

EXTERN int __tgt_target_teams_nowait(int64_t DeviceId, void *HostPtr,
                                     int32_t ArgNum, void **ArgsBase,
                                     void **Args, int64_t *ArgSizes,
                                     int64_t *ArgTypes, int32_t NumTeams,
                                     int32_t ThreadLimit, int32_t DepNum,
                                     void *DepList, int32_t NoAliasDepNum,
                                     void *NoAliasDepList) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return __tgt_target_teams_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase,
                                   Args, ArgSizes, ArgTypes, nullptr, nullptr,
                                   NumTeams, ThreadLimit);
}

EXTERN int __tgt_target_teams_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, void *HostPtr, int32_t ArgNum,
    void **ArgsBase, void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
    map_var_info_t *ArgNames, void **ArgMappers, int32_t NumTeams,
    int32_t ThreadLimit, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return __tgt_target_teams_mapper(Loc, DeviceId, HostPtr, ArgNum, ArgsBase,
                                   Args, ArgSizes, ArgTypes, ArgNames,
                                   ArgMappers, NumTeams, ThreadLimit);
}

EXTERN void __kmpc_push_target_tripcount_mapper(ident_t *Loc, int64_t DeviceId,
                                                uint64_t LoopTripcount) {
  TIMESCOPE_WITH_IDENT(Loc);
  if (checkDevice(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return;
  }
  DP("WARNING: __kmpc_push_target_tripcount has been deprecated and is a noop");
}

EXTERN void __kmpc_push_target_tripcount(int64_t DeviceId,
                                         uint64_t LoopTripcount) {
  __kmpc_push_target_tripcount_mapper(nullptr, DeviceId, LoopTripcount);
}

EXTERN int __tgt_target_kernel_nowait(ident_t *Loc, int64_t DeviceId,
                                      int32_t NumTeams, int32_t ThreadLimit,
                                      void *HostPtr, KernelArgsTy *KernelArgs,
                                      int32_t DepNum, void *DepList,
                                      int32_t NoAliasDepNum,
                                      void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return __tgt_target_kernel(Loc, DeviceId, NumTeams, ThreadLimit, HostPtr,
                             KernelArgs);
}
