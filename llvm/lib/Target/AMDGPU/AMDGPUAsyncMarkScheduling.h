//===--- AMDGPUAsyncMarkScheduling.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Adds the minimal scheduling-DAG dependencies required to keep
/// AMDGPU::ASYNCMARK / AMDGPU::WAIT_ASYNCMARK semantics intact while
/// allowing the scheduler to freely reorder unrelated work around them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUASYNCMARKSCHEDULING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUASYNCMARKSCHEDULING_H

#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include <memory>

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createAMDGPUAsyncMarkSchedDAGMutation();

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUASYNCMARKSCHEDULING_H
