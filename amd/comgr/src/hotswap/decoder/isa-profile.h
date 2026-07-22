//===- isa-profile.h - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_ISA_PROFILE_H
#define HOTSWAP_TRANSPILER_ISA_PROFILE_H

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace COMGR::hotswap {

// The subset of AMDGPU subtarget capabilities the raiser branches on, queried
// on demand from the MCSubtargetInfo rather than cached. Construct via
// `fromSubtarget`; the referenced subtarget must outlive the profile.
class ISAProfile {
public:
  static ISAProfile fromSubtarget(const llvm::MCSubtargetInfo &STI) {
    return ISAProfile(STI);
  }

  // Wavefront width in lanes (32 or 64).
  unsigned waveSize() const {
    return STI->hasFeature(llvm::AMDGPU::FeatureWavefrontSize32) ? 32 : 64;
  }
  bool isWave32() const { return waveSize() == 32; }
  bool hasValidWaveSize() const { return waveSize() == 32 || waveSize() == 64; }

  // Whether the target has AGPRs / the MAI (matrix) instruction set.
  bool hasAgpr() const { return llvm::AMDGPU::hasMAIInsts(*STI); }

  // Whether compute_pgm_rsrc2.USER_SGPR_COUNT is the wider gfx1250 6-bit field
  // rather than the older 5-bit field.
  bool hasGfx125UserSgprCountField() const {
    return llvm::AMDGPU::isGFX1250Plus(*STI);
  }

private:
  explicit ISAProfile(const llvm::MCSubtargetInfo &STI) : STI(&STI) {}

  const llvm::MCSubtargetInfo *STI;
};

} // namespace COMGR::hotswap

#endif
