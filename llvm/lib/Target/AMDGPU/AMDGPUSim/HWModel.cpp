//===- AMDGPUSim/HWModel.cpp - Hardware Model Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HWModel.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Default Latency Lookup
//===----------------------------------------------------------------------===//

unsigned getLatencyForClass(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:
    return DefaultLatency::VALU;
  case InstClass::SALU:
    return DefaultLatency::SALU;
  case InstClass::TRANS:
    return DefaultLatency::TRANS;
  case InstClass::WMMA:
    return DefaultLatency::TRANS;
  case InstClass::DS_READ:
    return DefaultLatency::DS_READ;
  case InstClass::DS_WRITE:
    return DefaultLatency::DS_WRITE;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return DefaultLatency::VMEM;
  case InstClass::SMEM:
    return DefaultLatency::SMEM;
  case InstClass::TDM:
    return DefaultLatency::DS_READ;
  case InstClass::BARRIER:
    return DefaultLatency::BARRIER;
  default:
    return 1;
  }
}

//===----------------------------------------------------------------------===//
// CoExecMask Functions
//===----------------------------------------------------------------------===//

namespace CoExecMask {

WMMAStageType getStageType(uint8_t Mask) {
  if (Mask == StageE0)
    return WMMAStageType::E0;
  if (Mask == StageE)
    return WMMAStageType::E;
  if (Mask == StageI)
    return WMMAStageType::I;
  if (Mask == StageV)
    return WMMAStageType::V;
  if (Mask & VALU)
    return WMMAStageType::I;
  if (Mask & WMMA)
    return WMMAStageType::V;
  return WMMAStageType::E;
}

uint8_t getMaskForIC(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:
    return VALU;
  case InstClass::TRANS:
    return TRANS;
  case InstClass::SALU:
  case InstClass::BARRIER:
  case InstClass::WAITCNT:
  case InstClass::BRANCH:
    return SALU;
  case InstClass::DELAY_ALU:
  case InstClass::MSB_SET:
    return CTRL;
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::TDM:
    return DS;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return VMEM;
  case InstClass::SMEM:
    return SMEM;
  case InstClass::WMMA:
    return WMMA;
  case InstClass::NOP:
    return VALU;
  default:
    return 0;
  }
}

} // namespace CoExecMask

//===----------------------------------------------------------------------===//
// WMMACoExecInfo Methods
//===----------------------------------------------------------------------===//

bool WMMACoExecInfo::canCoExec(InstClass IC, unsigned Stage) const {
  if (Stage >= TotalWindow)
    return false;
  return StageMask[Stage] & CoExecMask::getMaskForIC(IC);
}

std::optional<unsigned>
WMMACoExecInfo::findNextAllowedStage(InstClass IC,
                                     unsigned CurrentStage) const {
  uint8_t Needed = CoExecMask::getMaskForIC(IC);
  if (Needed == 0)
    return std::nullopt;
  for (unsigned S = CurrentStage; S < TotalWindow; ++S) {
    if (StageMask[S] & Needed)
      return S;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// WMMA Co-execution Info Factory
//===----------------------------------------------------------------------===//

static WMMACoExecInfo makeWMMACoExecInfo(std::string_view Pattern,
                                         bool HasScaling) {
  WMMACoExecInfo Info;
  Info.Occupancy = Pattern.find_last_not_of('V') + 1;
  Info.TotalWindow = Pattern.size();
  Info.LastIStage = Pattern.find_last_not_of('I');
  Info.HasScaling = HasScaling;
  Info.Pattern = Pattern;

  for (unsigned i = 0; i < Info.TotalWindow; ++i) {
    switch (Pattern[i]) {
    case '0':
      Info.StageMask[i] = CoExecMask::StageE0;
      break;
    case 'E':
      Info.StageMask[i] = CoExecMask::StageE;
      break;
    case 'I':
      Info.StageMask[i] = CoExecMask::StageI;
      break;
    case 'V':
      Info.StageMask[i] = CoExecMask::StageV;
      break;
    case 'A':
      Info.StageMask[i] = CoExecMask::All;
      break;
    default:
      llvm_unreachable("Invalid character in pattern");
      break;
    }
  }
  return Info;
}

WMMACoExecInfo getWMMACoExecInfo(WMMAVariant Variant, bool HasScaling) {
  switch (Variant) {
  case WMMAVariant::IU8_16x16x64:
    return makeWMMACoExecInfo("0EIIEEIIEEIIEEIIV", HasScaling);
  case WMMAVariant::F8F6F4_16x16x128:
    return makeWMMACoExecInfo("0EEIEEIIVV", HasScaling);
  case WMMAVariant::F8F6F4_16x16x128_BothF4:
    return makeWMMACoExecInfo("0EEIVV", HasScaling);
  case WMMAVariant::FP8_16x16x64:
  case WMMAVariant::BF8_16x16x64:
    return makeWMMACoExecInfo("0EEIVV", HasScaling);
  case WMMAVariant::F16_16x16x32:
  case WMMAVariant::BF16_16x16x32:
    return makeWMMACoExecInfo("0EIIEEIIV", HasScaling);
  case WMMAVariant::FP8_16x16x128:
  case WMMAVariant::BF8_16x16x128:
    return makeWMMACoExecInfo("0EEIEEIIVV", HasScaling);
  case WMMAVariant::F4_32x16x128:
    return makeWMMACoExecInfo("0EEIVV", HasScaling);
  case WMMAVariant::Default:
    return makeWMMACoExecInfo("AAAAAAAIV", HasScaling);
  }
  llvm_unreachable("Unknown WMMAVariant");
}

} // namespace AMDGPUSim
} // namespace llvm
