//===- isa-profile.h - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_ISA_PROFILE_H
#define HOTSWAP_TRANSPILER_ISA_PROFILE_H

namespace llvm {
class MCSubtargetInfo;
} // namespace llvm

namespace COMGR::hotswap {

// The subset of AMDGPU subtarget capabilities the raiser branches on, queried
// on demand from the MCSubtargetInfo rather than cached. Construct via
// `fromSubtarget`; the referenced subtarget must outlive the profile. The
// queries are defined in isa-profile.cpp so this header stays free of the
// AMDGPU target-private headers they need.
class ISAProfile {
public:
  static ISAProfile fromSubtarget(const llvm::MCSubtargetInfo &STI) {
    return ISAProfile(STI);
  }

  // Wavefront width in lanes (32 or 64).
  unsigned waveSize() const;
  bool isWave32() const;
  bool hasValidWaveSize() const;

  // Whether the target has AGPRs / the MAI (matrix) instruction set.
  bool hasAgpr() const;

  // Whether compute_pgm_rsrc2.USER_SGPR_COUNT is the wider gfx1250 6-bit field
  // rather than the older 5-bit field.
  bool hasGfx125UserSgprCountField() const;

  // Maximum USER_SGPR_COUNT supported by the source ISA.
  unsigned maxUserSgprs() const;

  // Width in bits of the signed immediate byte offset of a FLAT, GLOBAL or
  // SCRATCH instruction.
  unsigned flatOffsetBits() const;

  // Whether the per-lane offset a GLOBAL instruction adds to its SGPR-pair
  // base is signed. Targets without it read that offset as unsigned.
  bool hasSignedGlobalLaneOffset() const;

  // Whether the source ISA supports kernarg preloading.
  bool hasKernargPreload() const;

  // Whether the source ISA uses architected SGPRs.
  bool hasArchitectedSgprs() const;

  /// Return whether kernel descriptors for this ISA encode DX10_CLAMP and
  /// IEEE_MODE.
  bool hasDx10ClampAndIeeeMode() const;

  // Whether the ISA has the combined `s_waitcnt` covering every wait counter.
  bool hasCombinedWaitcnt() const;

  enum class WavePriorityModel { Gfx9, Gfx125 };

  /// Return the model used to combine system and user wave priorities.
  WavePriorityModel wavePriorityModel() const;

private:
  explicit ISAProfile(const llvm::MCSubtargetInfo &STI) : STI(&STI) {}

  const llvm::MCSubtargetInfo *STI;
};

} // namespace COMGR::hotswap

#endif
