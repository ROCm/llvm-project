//===-- amdgpu-mode-hwreg.h - SQ wave MODE register helpers ---------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Named bit positions and decode helpers for the per-wave MODE register
// (HW_REG_MODE / HW_REG_WAVE_MODE on gfx12+). Field layout matches
// s_setreg/s_getreg simm16 encoding:
//   id[5:0] | offset[10:6] | (size-1)[15:11]
//
// Canonical gfx1250 kernel prologue (before the first SMEM/VMEM op):
//   s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, ModeReg::ReplayModeBit, 1), 1
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_AMDGPU_MODE_HWREG_H
#define HOTSWAP_TRANSPILER_AMDGPU_MODE_HWREG_H

#include <cstdint>

namespace COMGR::hotswap::amdgpu {

/// HWREG id for the wave MODE register (`HW_REG_MODE` / `HW_REG_WAVE_MODE`).
static constexpr unsigned HwregIdMode = 1;

/// Per-wave MODE register bit fields.
struct ModeReg {
  /// FP16_OVFL -- overflowed f16 VALU results clamp to +/-MAX_FP16 instead of
  /// +/-inf (true infinities are preserved).
  static constexpr unsigned Fp16OvflBit = 23;

  /// REPLAY_MODE -- 0 = single-VMEM-group replay (hardware XCNT waits); 1 =
  /// multi-VMEM-group replay (software inserts s_wait_xcnt). Must be
  /// programmed before the first SMEM/VMEM instruction on the wave.
  static constexpr unsigned ReplayModeBit = 25;
  static constexpr unsigned ReplayModeFieldSizeBits = 1;
  static constexpr unsigned ReplayModeMultiGroup = 1;

  /// VGPR_MSB (gfx1250+ / 1024-addressable-VGPRs targets only) -- four 2-bit
  /// high-bit (most-significant-bit) bank selectors for VGPR operands, packed
  /// as (dst, src0, src1, src2) into a MODE setreg's imm bits [12:19].
  /// Captured by any MODE setreg on these targets regardless of the field
  /// selector; on older targets bits [12:19] are ordinary FP-mode fields.
  /// \ref VgprMsbModeToSetregRotate converts this byte to the S_SET_VGPR_MSB
  /// layout (src0, src1, src2, dst).
  static constexpr unsigned VgprMsbLowBit = 12;
  static constexpr uint8_t VgprMsbByteMask = 0xffu;
  static constexpr unsigned VgprMsbModeToSetregRotate = 2;
};

/// Decoded s_setreg / s_getreg field selector from the simm16 immediate.
struct SetregField {
  unsigned HwregId;
  unsigned Offset;
  unsigned SizeBits;
};

inline SetregField decodeSetregSimm16(int64_t Simm16) {
  const uint32_t Enc = static_cast<uint32_t>(Simm16) & 0xffffu;
  return {Enc & 0x3fu, (Enc >> 6) & 0x1fu, ((Enc >> 11) & 0x1fu) + 1u};
}

/// True when \p HwregId / \p Simm16 / \p Imm encode
/// `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, ModeReg::ReplayModeBit, 1), 1`.
inline bool isModeReplayMultiGroupWrite(unsigned HwregId, int64_t Simm16,
                                        int64_t Imm) {
  const SetregField Field = decodeSetregSimm16(Simm16);
  return HwregId == HwregIdMode &&
         Field.Offset == ModeReg::ReplayModeBit &&
         Field.SizeBits == ModeReg::ReplayModeFieldSizeBits &&
         Imm == ModeReg::ReplayModeMultiGroup;
}

} // namespace COMGR::hotswap::amdgpu

#endif // HOTSWAP_TRANSPILER_AMDGPU_MODE_HWREG_H
