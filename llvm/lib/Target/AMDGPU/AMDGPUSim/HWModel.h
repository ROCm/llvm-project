//===- AMDGPUSim/HWModel.h - Hardware Model Parameters ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines hardware model parameters for the AMDGPU static simulator,
/// including latency defaults, WMMA co-execution rules, and memory limits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_HWMODEL_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_HWMODEL_H

#include "SimInst.h"
#include <array>
#include <optional>
#include <string_view>

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// GPU Targets
//===----------------------------------------------------------------------===//

/// Supported GPU targets for the simulator.
enum class GPUTarget : uint8_t {
  GFX1250 = 0, // gfx1250 (default)
};

//===----------------------------------------------------------------------===//
// Latency Constants (gfx1250 defaults)
//===----------------------------------------------------------------------===//

namespace DefaultLatency {
constexpr unsigned VALU = 5;
constexpr unsigned SALU = 2;
constexpr unsigned TRANS = 8;
constexpr unsigned DS_READ = 50;
constexpr unsigned DS_WRITE = 8;
constexpr unsigned VMEM = 300;
constexpr unsigned SMEM = 20;
constexpr unsigned BARRIER = 32;
} // namespace DefaultLatency

/// Get the default latency for an instruction class.
unsigned getLatencyForClass(InstClass IC);

//===----------------------------------------------------------------------===//
// WMMA Co-execution Masks
//===----------------------------------------------------------------------===//

namespace CoExecMask {
constexpr uint8_t None = 0;
constexpr uint8_t CTRL = 1 << 0;
constexpr uint8_t VALU = 1 << 1;
constexpr uint8_t TRANS = 1 << 2;
constexpr uint8_t SALU = 1 << 3;
constexpr uint8_t DS = 1 << 4;
constexpr uint8_t VMEM = 1 << 5;
constexpr uint8_t SMEM = 1 << 6;
constexpr uint8_t WMMA = 1 << 7;
constexpr uint8_t All = 0xFF;

constexpr uint8_t MEM = DS | VMEM | SMEM;
constexpr uint8_t StageE0 = CTRL;
constexpr uint8_t StageE = CTRL | SALU | MEM;
constexpr uint8_t StageI = CTRL | SALU | MEM | VALU | TRANS;
constexpr uint8_t StageV = CTRL | SALU | MEM | WMMA;

/// Get the WMMA stage type for a co-execution mask.
WMMAStageType getStageType(uint8_t Mask);

/// Get the co-execution mask for an instruction class.
uint8_t getMaskForIC(InstClass IC);
} // namespace CoExecMask

//===----------------------------------------------------------------------===//
// WMMA Co-execution Info
//===----------------------------------------------------------------------===//

constexpr unsigned MaxWMMAStages = 20;

struct WMMACoExecInfo {
  unsigned Occupancy = 0;
  unsigned TotalWindow = 0;
  std::array<uint8_t, MaxWMMAStages> StageMask = {};
  unsigned LastIStage = 0;
  bool HasScaling = false;
  std::string_view Pattern;

  WMMACoExecInfo() { StageMask.fill(CoExecMask::All); }

  bool canCoExec(InstClass IC, unsigned Stage) const;
  std::optional<unsigned> findNextAllowedStage(InstClass IC,
                                               unsigned CurrentStage) const;

  bool isBackToBack(unsigned NextWMMAStage) const {
    return NextWMMAStage >= Occupancy && NextWMMAStage < TotalWindow;
  }
};

/// Get WMMA co-execution info for a given variant and scaling mode.
WMMACoExecInfo getWMMACoExecInfo(WMMAVariant Variant, bool HasScaling);

//===----------------------------------------------------------------------===//
// Hardware Model
//===----------------------------------------------------------------------===//

struct HWModel {
  // Memory FIFO limits
  unsigned MaxDSInFlight = 10;
  unsigned MaxVMEMInFlight = 16;
  unsigned MaxSMEMInFlight = 10;
  unsigned MaxTDMInFlight = 4;

  // VGPR source cache parameters
  unsigned VGPRCacheBanks = 8;
  unsigned VGPRCachePorts = 3;
  unsigned VGPRCacheDepth = 4;

  // SGPR bank count
  unsigned SGPRBanks = 4;

  // IS cache parameters
  unsigned ISCacheNumLines = 4;
  unsigned ISCacheLineSize = 64; // bytes per line (16 DW * 4 bytes)
  unsigned SQCToISLatency = 26;

  // VA_VDST multiplier
  unsigned VaVdstMultiplier = 4;

  static HWModel gfx1250() { return HWModel(); }
};

//===----------------------------------------------------------------------===//
// Memory Limits
//===----------------------------------------------------------------------===//

namespace MemLimits {
constexpr unsigned MaxDSInFlight = 10;
constexpr unsigned MaxVMEMInFlight = 16;
constexpr unsigned MaxSMEMInFlight = 10;
constexpr unsigned MaxTDMInFlight = 4;
} // namespace MemLimits

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_HWMODEL_H
