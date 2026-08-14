//===- SQTTConfig.h - SQTT marker configuration ---------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines marker encoding constants and compile-time environment
/// configuration.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_AMD_SQTT_MARKER_LIB_SQTTCONFIG_H
#define LLVM_AMD_SQTT_MARKER_LIB_SQTTCONFIG_H

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

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

  static uint32_t parseEnvMask(const char *Name, uint32_t Def = 0xFFFFFFFF) {
    const char *V = std::getenv(Name);
    if (!V || V[0] == '\0')
      return Def;
    if (std::strcmp(V, "-1") == 0)
      return 0xFFFFFFFF;
    char *End = nullptr;
    unsigned long Val = std::strtoul(V, &End, 0);
    if (End == V || *End != '\0') {
      llvm::errs() << "sqtt: warning: invalid value for " << Name << "='" << V
                   << "', using default\n";
      return Def;
    }
    return static_cast<uint32_t>(Val);
  }

  static bool parseEnvBool(const char *Name, bool Def) {
    const char *V = std::getenv(Name);
    if (!V || V[0] == '\0')
      return Def;
    llvm::StringRef S(V);
    return S.equals_insensitive("1") || S.equals_insensitive("y") ||
           S.equals_insensitive("yes") || S.equals_insensitive("true") ||
           S.equals_insensitive("on");
  }

  static MemBarrierMode parseEnvMemBarrier(const char *Name,
                                           MemBarrierMode Def) {
    const char *V = std::getenv(Name);
    if (!V || V[0] == '\0')
      return Def;
    llvm::StringRef S(V);
    // Numeric: 0=None, 1=AsmClobber, 2=Fence
    if (S == "0")
      return MemBarrierMode::None;
    if (S == "1")
      return MemBarrierMode::AsmClobber;
    if (S == "2")
      return MemBarrierMode::Fence;
    // Named (case-insensitive)
    if (S.equals_insensitive("none") || S.equals_insensitive("off"))
      return MemBarrierMode::None;
    if (S.equals_insensitive("asm") || S.equals_insensitive("compiler") ||
        S.equals_insensitive("clobber"))
      return MemBarrierMode::AsmClobber;
    if (S.equals_insensitive("fence") || S.equals_insensitive("on") ||
        S.equals_insensitive("hw"))
      return MemBarrierMode::Fence;
    llvm::errs() << "sqtt: warning: invalid value for " << Name << "='" << V
                 << "', expected one of "
                 << "{none|asm|fence|0|1|2}, using default\n";
    return Def;
  }

  static unsigned parseEnvUnsigned(const char *Name, unsigned Def) {
    const char *V = std::getenv(Name);
    if (!V || V[0] == '\0')
      return Def;
    llvm::StringRef S(V);
    unsigned Out = 0;
    if (S.getAsInteger(10, Out)) {
      llvm::errs() << "sqtt: warning: invalid value for " << Name << "='" << V
                   << "', using default\n";
      return Def;
    }
    return Out;
  }

  static SQTTConfig fromEnvironment() {
    SQTTConfig C;
    C.InstrumentBarriers = parseEnvBool("SQTT_INSTRUMENT_BARRIERS", false);
    C.MemBarrier =
        parseEnvMemBarrier("SQTT_MEM_BARRIER", MemBarrierMode::Fence);
    C.WaveMask = parseEnvMask("SQTT_SCOPE_WAVE", 0xFFFFFFFF);
    C.SimdMask = parseEnvMask("SQTT_SCOPE_SIMD", 0xF);
    C.CuMask = parseEnvMask("SQTT_SCOPE_CU", 0x3);
    C.WgMask = parseEnvMask("SQTT_SCOPE_WG", 0xFFFFFFFF);
    C.ShaderClockBits = parseEnvUnsigned("SQTT_SHADER_CLOCK_BITS", 0);
    C.ShaderClockShift = parseEnvUnsigned("SQTT_SHADER_CLOCK_SHIFT", 4);

    const char *FuncEnv = std::getenv("SQTT_INSTRUMENT_FUNCTIONS");
    if (FuncEnv && FuncEnv[0] != '\0') {
      llvm::StringRef S(FuncEnv);
      if (S.consume_front("cost:"))
        C.Mode = CostMode::WeightedCost;
      S.getAsInteger(10, C.FunctionThreshold);
    }

    // SQTT_INSTRUMENT_MEMORY=N:M  (N=chunk size, M=max gap)
    const char *MemEnv = std::getenv("SQTT_INSTRUMENT_MEMORY");
    if (MemEnv && MemEnv[0] != '\0') {
      llvm::StringRef S(MemEnv);
      llvm::StringRef NStr, MStr;
      std::tie(NStr, MStr) = S.split(':');
      unsigned N = 0, M = 0;
      if (!NStr.getAsInteger(10, N) && !MStr.empty() &&
          !MStr.getAsInteger(10, M) && N > 0) {
        C.MemoryChunkSize = N;
        C.MemoryMaxGap = M;
      } else {
        llvm::errs() << "sqtt: warning: invalid SQTT_INSTRUMENT_MEMORY "
                        "format '"
                     << MemEnv << "', expected N:M\n";
      }
    }

    // SQTT_TRACE_ADDRESSES=memory|lds|memory,lds
    const char *AddrEnv = std::getenv("SQTT_TRACE_ADDRESSES");
    if (AddrEnv && AddrEnv[0] != '\0') {
      llvm::StringRef S(AddrEnv);
      llvm::SmallVector<llvm::StringRef, 2> Parts;
      S.split(Parts, ',');
      for (llvm::StringRef &P : Parts) {
        llvm::StringRef T = P.trim();
        if (T == "memory")
          C.TraceMemoryAddrs = true;
        else if (T == "lds")
          C.TraceLDSAddrs = true;
        else
          llvm::errs() << "sqtt: warning: unknown SQTT_TRACE_ADDRESSES "
                          "category '"
                       << T << "'\n";
      }
      if (C.hasAddressTracing() && C.MemoryChunkSize) {
        llvm::errs() << "sqtt: error: SQTT_TRACE_ADDRESSES and "
                        "SQTT_INSTRUMENT_MEMORY are mutually exclusive\n";
        C.TraceMemoryAddrs = C.TraceLDSAddrs = false;
      }
    }

    return C;
  }
};

#endif // LLVM_AMD_SQTT_MARKER_LIB_SQTTCONFIG_H
