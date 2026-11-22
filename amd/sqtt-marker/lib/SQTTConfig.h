//===- SQTTConfig.h - SQTT marker configuration ---------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines marker encoding constants and compile-time configuration.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_AMD_SQTT_MARKER_LIB_SQTTCONFIG_H
#define LLVM_AMD_SQTT_MARKER_LIB_SQTTCONFIG_H

#include <cstdint>

// getRegisterImmediate encoding: (size_minus_1 << 11) | (offset << 6) |
// register_id
constexpr uint32_t getRegisterImmediate(uint32_t SzM1, uint32_t Off,
                                        uint32_t Reg) {
  return (SzM1 << 11) | (Off << 6) | Reg;
}

// GCN/CDNA (gfx9): HW_ID register = 4
constexpr uint32_t Gfx9HwregWave =
    getRegisterImmediate(3, 0, 4); // WAVE_ID [3:0], 4 bits
constexpr uint32_t Gfx9HwregSimd =
    getRegisterImmediate(1, 4, 4); // SIMD_ID [5:4], 2 bits
constexpr uint32_t Gfx9HwregCu =
    getRegisterImmediate(3, 8, 4); // CU_ID [11:8], 4 bits
constexpr uint32_t Gfx9HwregWg =
    getRegisterImmediate(3, 16, 4); // TG_ID [19:16], 4 bits

// RDNA (gfx10/11/12): HW_ID1=23, HW_ID2=24
constexpr uint32_t RdnaHwregWave =
    getRegisterImmediate(4, 0, 23); // WAVE_ID [4:0], 5 bits
constexpr uint32_t RdnaHwregSimd =
    getRegisterImmediate(1, 8, 23); // SIMD_ID [9:8], 2 bits
constexpr uint32_t RdnaHwregCu =
    getRegisterImmediate(3, 10, 23); // WGP_ID [13:10], 4 bits
constexpr uint32_t RdnaHwregWg =
    getRegisterImmediate(4, 16, 24); // WG_ID [20:16], 5 bits

// Maximum useful mask per HW field (covers all valid IDs)
constexpr uint32_t FullWaveMask = 0xFFFFFFFF; // up to 32 waves
constexpr uint32_t FullSimdMask = 0xF;        // up to 4 SIMDs
constexpr uint32_t FullCuMask = 0xFFFF;       // up to 16 CUs/WGPs
constexpr uint32_t FullWgMask = 0xFFFFFFFF;   // up to 32 WGs

// Bit flags for marker encoding (low 2 bits)
//
//   Bit  0:      exit previous scope (pop top)
//   Bit  1:      enter scope (push)
//   Bits [7:2]:  6-bit ID   (s_ttracedata_imm, IDs 0-63)
//   Bits [31:2]: 30-bit ID  (s_ttracedata, IDs 0-1G)
//
// The marker type (function, user, barrier, memory) is determined by
// looking up the ID in the .sqtt_funcmap section, not from encoding bits.
constexpr uint32_t FlagExitPrev = 1u;   // bit 0: exit previous scope
constexpr uint32_t FlagEnter = 1u << 1; // bit 1: entering scope
constexpr uint32_t FlagMask = 0x3;      // all flag bits

// Encode a marker value for s_ttracedata / s_ttracedata_imm
inline uint32_t encodeMarker(uint32_t Id, bool Enter, bool ExitPrev) {
  uint32_t Val = (Id << 2);
  if (ExitPrev)
    Val |= FlagExitPrev;
  if (Enter)
    Val |= FlagEnter;
  return Val;
}

// Can this encoded marker value fit in s_ttracedata_imm (8-bit)?
inline bool canUseImm(uint32_t Encoded) { return Encoded <= 0xFF; }

enum class CostMode { InstructionCount, WeightedCost };

// SQTT_MEM_BARRIER selects the strength of the reordering boundary planted
// around every trace marker.
//
//   None:       no fence/clobber. Only the cheap sched_barrier(0) hints
//               survive. Fastest kernel; markers may drift in LDS-pipelined
//               regions.
//   AsmClobber: empty inline asm with "~{memory}" -- IR/MIR-level memory
//               reorder constraint, no machine code.
//   Fence:      fence syncscope("workgroup") acq_rel before AND after the
//               marker, tagged as AMDGPU local/LDS synchronization. Preserves
//               the compiler-visible marker boundary while avoiding global
//               cache invalidation for marker-only fences. Default.
enum class MemBarrierMode { None, AsmClobber, Fence };

struct SQTTConfig {
  bool InstrumentBarriers = false;
  CostMode Mode = CostMode::InstructionCount;
  unsigned FunctionThreshold = 0; // 0 = disabled
  unsigned MemoryChunkSize = 0;   // 0 = disabled; otherwise N ops per marker
  unsigned MemoryMaxGap = 0;      // M: max non-memory instructions between ops
  uint32_t WaveMask = 0xFFFFFFFF; // default: all waves (0-31)
  uint32_t SimdMask = 0xF;        // default: all 4 SIMDs
  uint32_t CuMask = 0x3;          // default: CU 0-1
  uint32_t WgMask = 0xFFFFFFFF;   // default: all WGs (0-31)
  MemBarrierMode MemBarrier = MemBarrierMode::Fence;
  bool TraceMemoryAddrs = false; // trace global/buffer/flat addresses
  bool TraceLDSAddrs = false;    // trace LDS addresses
  unsigned ShaderClockBits = 0;  // opt in to clock packing explicitly
  unsigned ShaderClockShift = 4;

  bool hasAddressTracing() const { return TraceMemoryAddrs || TraceLDSAddrs; }

  bool needsScopeCheck() const {
    return (WaveMask & FullWaveMask) != FullWaveMask ||
           (SimdMask & FullSimdMask) != FullSimdMask ||
           (CuMask & FullCuMask) != FullCuMask ||
           (WgMask & FullWgMask) != FullWgMask;
  }

  /// Reads only the SQTT_* environment variables.
  static SQTTConfig fromEnvironment();

  /// Reads plugin command-line options, falling back to SQTT_* environment
  /// variables for options not explicitly provided.
  static SQTTConfig fromCommandLine();
};

#endif // LLVM_AMD_SQTT_MARKER_LIB_SQTTCONFIG_H
