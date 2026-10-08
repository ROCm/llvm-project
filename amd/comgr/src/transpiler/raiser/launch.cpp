//===- launch.cpp - Transpiler launch requirements ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/launch.h"

#include "transpiler/raiser/raise_failure.h"

#include <cassert>

using namespace llvm;

namespace COMGR::transpiler {

Expected<LaunchDimensions>
KernelLaunchRequirements::project(StringRef KernelName,
                                  const LaunchDimensions &Source) const {
  assert(!KernelName.empty() && "launch requirements need a kernel name");
  auto Refuse = [&](const Twine &Detail) {
    return RaiseFailure::inKernel(RaiseFailureReason::UnsupportedLaunch,
                                  KernelName, Detail);
  };
  if (RequiredWorkgroupSize && Source.Workgroup != *RequiredWorkgroupSize)
    return Refuse(
        "workgroup does not match the source kernel's required dimensions");
  uint64_t Workitems = 1;
  for (size_t I = 0; I != Source.Grid.size(); ++I) {
    if (Source.Grid[I] == 0 || Source.Workgroup[I] == 0)
      return Refuse("launch dimensions must be nonzero");
    // Bound each factor before multiplying, including untrusted dimensions.
    if (Source.Workgroup[I] > MaxWorkgroupSize)
      return Refuse("workgroup exceeds the kernel's supported launch size");
    Workitems *= Source.Workgroup[I];
    if (Workitems > MaxWorkgroupSize)
      return Refuse("workgroup exceeds the kernel's supported launch size");
  }
  if (Mapping == Kind::Unchanged)
    return Source;

  assert(SourceWaveSize > 0 && ReplicationFactor > 1 &&
         "invalid replicated launch requirements");
  if (Source.Workgroup[1] != 1 || Source.Workgroup[2] != 1 ||
      Source.Grid[1] != 1 || Source.Grid[2] != 1)
    return Refuse("replicated dispatch requires a one-dimensional launch");
  if (Source.Workgroup[0] % SourceWaveSize != 0)
    return Refuse("replicated dispatch requires whole source waves");
  if (Source.Grid[0] % Source.Workgroup[0] != 0)
    return Refuse("replicated dispatch requires complete workgroups");
  if (Source.Grid[0] > UINT32_MAX / ReplicationFactor)
    return Refuse("replicated grid size overflows the dispatch packet");

  LaunchDimensions Target = Source;
  Target.Grid[0] *= ReplicationFactor;
  Target.Workgroup[0] *= ReplicationFactor;
  return Target;
}

} // namespace COMGR::transpiler
