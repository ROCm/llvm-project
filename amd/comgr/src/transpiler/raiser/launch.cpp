//===- launch.cpp - Transpiler launch requirements ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/launch.h"

#include "transpiler/raiser/raise_failure.h"

using namespace llvm;

namespace COMGR::transpiler {

Expected<LaunchDimensions>
KernelLaunchRequirements::project(const LaunchDimensions &Source) const {
  auto Refuse = [](const Twine &Detail) {
    return RaiseFailure::general(RaiseFailureReason::UnsupportedLaunch, Detail);
  };
  if (RequiredWorkgroupSize && Source.Workgroup != *RequiredWorkgroupSize)
    return Refuse(
        "workgroup does not match the source kernel's required dimensions");
  uint64_t Workitems = 1;
  for (unsigned I = 0; I != 3; ++I) {
    if (!Source.Grid[I] || !Source.Workgroup[I])
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

  if (Source.Workgroup[1] != 1 || Source.Workgroup[2] != 1 ||
      Source.Grid[1] != 1 || Source.Grid[2] != 1)
    return Refuse("replicated dispatch requires a one-dimensional launch");
  if (Source.Workgroup[0] % 32)
    return Refuse("replicated dispatch requires whole source waves");
  if (Source.Grid[0] % Source.Workgroup[0])
    return Refuse("replicated dispatch requires complete workgroups");
  if (Source.Grid[0] > UINT32_MAX / 2)
    return Refuse("replicated grid size overflows the dispatch packet");

  LaunchDimensions Target = Source;
  Target.Grid[0] *= 2;
  Target.Workgroup[0] *= 2;
  return Target;
}

} // namespace COMGR::transpiler
