//===- comgr-hotswap-patch-f32-to-e5m3.cpp - E5M3 CLAMP-bit emulation ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Scratch-patch pass for Case 1 of the B0-to-A0 scratch-patch pipeline:
/// E5M3 CLAMP-bit emulation for FP8 conversion instructions.
///
/// GFX1250 B0 added a CLAMP bit to three FP8 conversion instructions that
/// selects UE5M3 format (CLAMP=1) instead of E4M3 (CLAMP=0). On A0 the
/// CLAMP bit is non-functional — CLAMP=1 silently produces E4M3. This file
/// provides the strong override of applyScratchPatches that detects CLAMP=1
/// FP8 conversions and emits software emulation sequences.
///
/// Covered instructions (implementation order):
///   1. v_cvt_pk_fp8_f32  — F32 pack to FP8 (this file, done)
///   2. v_cvt_sr_fp8_f32  — F32 stochastic-round to FP8 (future)
///   3. v_cvt_f32_fp8     — FP8 unpack to F32 (future)
///
/// Design document: docs/scratch-patches/1_f32_to_e5m3/v_cvt_pk_fp8_f32.md
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Emit a deferred trampoline: queues \p Replacement bytes (plus a
/// MinInstSize placeholder for the branch-back) into \p Ctx.OutTrampolines.
/// The branch encoding is resolved later by fixupTrampolineBranches in
/// comgr-hotswap-b0a0.cpp.
static bool queueTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                            uint32_t InstSize, ArrayRef<uint8_t> Replacement) {
  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
  T.Bytes.insert(T.Bytes.end(), Replacement.begin(), Replacement.end());
  T.Bytes.insert(T.Bytes.end(), MinInstSize, uint8_t{0});
  Ctx.OutTrampolines.emplace_back(std::move(T));
  return true;
}

/// Format a VGPR number as an assembly operand string ("v0", "v1", ...).
static std::string vgprName(unsigned N) {
  return ("v" + Twine(N)).str();
}

/// Parse the printed operands of a 3-operand instruction.  Expects MCInst-
/// Printer output shaped like "  mnemonic dst, src0, src1 [modifiers...]".
/// Returns true on success and fills \p Dst, \p Src0, \p Src1 with the
/// trimmed operand strings (including any neg/abs wrappers).
static bool parseThreeOperands(const MCInst &Inst, const LLVMState &LS,
                               std::string &Dst, std::string &Src0,
                               std::string &Src1) {
  std::string Buf;
  raw_string_ostream OS(Buf);
  LS.MCIP->printInst(&Inst, 0, "", *LS.STI, OS);

  StringRef S = StringRef(Buf).ltrim();
  // Skip mnemonic.
  auto [Mnem, Rest] = S.split(' ');
  Rest = Rest.ltrim();

  SmallVector<StringRef, 4> Parts;
  Rest.split(Parts, ',');
  if (Parts.size() < 3)
    return false;

  Dst = Parts[0].trim().str();
  Src0 = Parts[1].trim().str();
  // Third operand may have trailing modifiers ("v2 clamp"); take first word.
  Src1 = Parts[2].trim().split(' ').first.str();
  return !Dst.empty() && !Src0.empty() && !Src1.empty();
}

// ---------------------------------------------------------------------------
// v_cvt_pk_fp8_f32 patch  (Case 1, instruction 1)
// ---------------------------------------------------------------------------

/// Patch a single v_cvt_pk_fp8_f32 with CLAMP=1 (E5M3 mode).
///
/// Base conversion: F32 -> clamp-to-positive -> F16 -> shift >> 7 -> UE5M3.
/// UE5M3 shares F16's 5-bit exponent (bias 15), so v_cvt_f16_f32 handles
/// exponent re-biasing and denormal flush; the shift extracts the top 8 bits
/// of the unsigned F16 representation (sign bit excluded, since UE5M3 is
/// unsigned and negatives are clamped to zero).
///
/// Edge cases NOT covered by this base path (documented in
/// docs/scratch-patches/1_f32_to_e5m3/v_cvt_pk_fp8_f32.md §4):
///   - NaN:    base path maps NaN→0 instead of 0xFF
///   - Overflow/Inf: base path produces 0xF8 instead of 0xFE
///   - RTE rounding: base path truncates F16 mantissa bits [6:0]
///   - Source modifiers: not propagated from original instruction
static uint32_t patchCvtPkFp8F32(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  if (DI.Size != 8) {
    log() << "hotswap: error: cvt_pk_fp8_f32: unexpected inst size "
          << DI.Size << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  const uint8_t *Raw = Ctx.Text + DI.Offset;
  bool Clamp = (Raw[1] >> 7) & 1;
  if (!Clamp)
    return 0;

  bool WriteHigh = (Raw[1] >> 6) & 1;

  // Extract operand names from the disassembled instruction.
  std::string VdstStr, Src0Str, Src1Str;
  if (!parseThreeOperands(DI.Inst, Ctx.LS, VdstStr, Src0Str, Src1Str)) {
    log() << "hotswap: error: cvt_pk_fp8_f32: failed to parse operands at "
          << "offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // Allocate two scratch VGPRs for the conversion temporaries.
  // findKernelAtOffset compares against symbol st_value, which is a virtual
  // address in linked ELFs; DI.Offset is section-relative.
  std::string KernelName =
      Ctx.Elf.findKernelAtOffset(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs =
      Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize);
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  ScratchAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                         Ctx.Config.MaxVgprs);

  std::optional<unsigned> T0 = Alloc.alloc();
  std::optional<unsigned> T1 = Alloc.alloc();
  if (!T0 || !T1) {
    log() << "hotswap: error: cvt_pk_fp8_f32: unable to allocate 2 scratch "
          << "VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string T0Name = vgprName(*T0);
  std::string T1Name = vgprName(*T1);

  // Build the replacement assembly.
  //
  // For each source: clamp negative → F16 → shift to get UE5M3 byte.
  // Then pack and merge into the correct half of vdst.
  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  // --- src0 → byte0 in T0 ---
  AsmOS << "v_max_f32 " << T0Name << ", 0, " << Src0Str << "\n";
  AsmOS << "v_cvt_f16_f32 " << T0Name << ", " << T0Name << "\n";
  AsmOS << "v_lshrrev_b32 " << T0Name << ", 7, " << T0Name << "\n";

  // --- src1 → byte1 in T1 ---
  AsmOS << "v_max_f32 " << T1Name << ", 0, " << Src1Str << "\n";
  AsmOS << "v_cvt_f16_f32 " << T1Name << ", " << T1Name << "\n";
  AsmOS << "v_lshrrev_b32 " << T1Name << ", 7, " << T1Name << "\n";

  // --- pack: T0[15:0] = { byte1, byte0 } ---
  AsmOS << "v_lshl_or_b32 " << T0Name << ", " << T1Name << ", 8, " << T0Name
        << "\n";

  // --- merge into the correct 16-bit half of vdst ---
  if (!WriteHigh) {
    // op_sel[3]==0: write low half, preserve high half.
    AsmOS << "v_bfi_b32 " << VdstStr << ", 0xFFFF, " << T0Name << ", "
          << VdstStr << "\n";
  } else {
    // op_sel[3]==1: shift packed result to [31:16], write high half.
    AsmOS << "v_lshlrev_b32 " << T0Name << ", 16, " << T0Name << "\n";
    AsmOS << "v_bfi_b32 " << VdstStr << ", 0xFFFF0000, " << T0Name << ", "
          << VdstStr << "\n";
  }

  SmallVector<uint8_t> ReplacementBytes = assembleSingleInst(Asm, Ctx.LS);
  if (ReplacementBytes.empty()) {
    log() << "hotswap: error: cvt_pk_fp8_f32: assembly failed for "
          << "replacement at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  if (!queueTrampoline(Ctx, DI.Offset, DI.Size, ReplacementBytes))
    return 0;

  // Update kernel stats.
  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  unsigned Extra = Alloc.extraVgprsNeeded();
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchReused += 2 - Alloc.extraVgprsNeeded();
  Stats.ScratchAboveKd += Alloc.extraVgprsNeeded();

  // Record scratch usage for post-patch verification.
  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: cvt_pk_fp8_f32: patched CLAMP=1 (E5M3) at offset 0x"
        << utohexstr(DI.Offset) << " (" << ReplacementBytes.size()
        << " bytes, scratch v" << *T0 << "/v" << *T1
        << ", half=" << (WriteHigh ? "high" : "low") << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// applyScratchPatches — strong override
// ---------------------------------------------------------------------------

/// Dispatch scratch-patch pass over the decoded instruction stream.
/// Called once per instruction from the per-instruction loop in
/// applyGfx1250B0toA0Rules (comgr-hotswap-b0a0.cpp). Returns the number
/// of patches applied at \p Idx (0 or 1).
uint32_t applyScratchPatches(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_cvt_pk_fp8_f32")
    return patchCvtPkFp8F32(Ctx, Idx);

  return 0;
}

} // namespace hotswap
} // namespace COMGR
