//===- launch.h - Transpiler launch requirements -----------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_LAUNCH_H
#define TRANSPILER_LAUNCH_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>

namespace COMGR::transpiler {

/// Source or target dispatch dimensions, all measured in workitems.
struct LaunchDimensions {
  /// Number of workitems in the complete grid along X, Y, and Z.
  std::array<uint32_t, 3> Grid;
  /// Number of workitems in each workgroup along X, Y, and Z.
  std::array<uint32_t, 3> Workgroup;
};

/// Per-kernel launch constraints. Kernarg bytes retain their source values,
/// including hidden geometry arguments; only the dispatch extents change.
struct KernelLaunchRequirements {
  enum class Kind { Unchanged, Replicated1D };
  Kind Mapping = Kind::Unchanged;
  /// Largest supported logical workgroup, in workitems.
  unsigned MaxWorkgroupSize;
  /// Exact logical dimensions, if required by the source kernel.
  std::optional<std::array<uint32_t, 3>> RequiredWorkgroupSize;
  /// Number of lanes in a logical source wave.
  unsigned SourceWaveSize = 1;
  /// Number of physical workitems launched for each logical workitem.
  unsigned ReplicationFactor = 1;

  /// Validate a source launch for KernelName and return its target dimensions.
  llvm::Expected<LaunchDimensions>
  project(llvm::StringRef KernelName, const LaunchDimensions &Source) const;
};

/// Callers allowing changed geometry must carry and enforce every kernel's
/// launch requirements when executing the resulting code.
enum class LaunchPolicy { PreserveGeometry, AllowReplication };

} // namespace COMGR::transpiler

#endif
