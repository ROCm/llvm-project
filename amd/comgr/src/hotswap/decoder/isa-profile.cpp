//===- isa-profile.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The ISAProfile queries live here rather than in the header so only this
// translation unit pulls the AMDGPU target-private subtarget-feature enum (via
// its TableGen-generated includes); consumers include a clean isa-profile.h.
// The queries read feature bits through MCSubtargetInfo directly rather than
// the AMDGPU:: helper functions, whose out-of-line symbols an
// LLVM_LINK_LLVM_DYLIB build does not export.
//
//===----------------------------------------------------------------------===//

#include "hotswap/decoder/isa-profile.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace COMGR::hotswap {

unsigned ISAProfile::waveSize() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureWavefrontSize32) ? 32 : 64;
}

bool ISAProfile::isWave32() const { return waveSize() == 32; }

bool ISAProfile::hasValidWaveSize() const {
  return waveSize() == 32 || waveSize() == 64;
}

bool ISAProfile::hasAgpr() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureMAIInsts);
}

bool ISAProfile::hasGfx125UserSgprCountField() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureGFX1250Insts);
}

unsigned ISAProfile::maxUserSgprs() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureGFX1250Insts) ? 32 : 16;
}

unsigned ISAProfile::flatOffsetBits() const {
  if (STI->hasFeature(llvm::AMDGPU::FeatureFlatOffsetBits12))
    return 12;
  if (STI->hasFeature(llvm::AMDGPU::FeatureFlatOffsetBits24))
    return 24;
  return 13;
}

bool ISAProfile::hasSignedGlobalLaneOffset() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureGFX1250Insts);
}

bool ISAProfile::hasKernargPreload() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureKernargPreload);
}

bool ISAProfile::hasArchitectedSgprs() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureArchitectedSGPRs);
}

bool ISAProfile::hasDx10ClampAndIeeeMode() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureDX10ClampAndIEEEMode);
}

bool ISAProfile::hasCombinedWaitcnt() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureGFX9);
}

ISAProfile::WavePriorityModel ISAProfile::wavePriorityModel() const {
  return STI->hasFeature(llvm::AMDGPU::FeatureGFX1250Insts)
             ? WavePriorityModel::Gfx125
             : WavePriorityModel::Gfx9;
}

} // namespace COMGR::hotswap
