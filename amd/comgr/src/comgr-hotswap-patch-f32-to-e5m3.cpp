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
///   2. v_cvt_sr_fp8_f32  — F32 stochastic-round to FP8 (done)
///   3. v_cvt_f32_fp8     — FP8 unpack to F32 (done)
///
/// Design documents:
///   docs/scratch-patches/1_f32_to_e5m3/v_cvt_pk_fp8_f32.md
///   docs/scratch-patches/1_f32_to_e5m3/v_cvt_sr_fp8_f32.md
///   docs/scratch-patches/1_f32_to_e5m3/v_cvt_f32_fp8.md
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

static std::string vgprName(unsigned N) {
  return ("v" + Twine(N)).str();
}

/// Strip True16 sub-register suffixes (`.l`, `.h`) from a register name.
/// Newer LLVM MCInstPrinter builds emit these for instructions like
/// v_cvt_pk_fp8_f32 that write to a 16-bit half, but the 32-bit ALU
/// instructions in our replacement sequences don't accept them.
static std::string stripTrue16Suffix(StringRef S) {
  if (S.ends_with(".l") || S.ends_with(".h"))
    return S.drop_back(2).str();
  return S.str();
}

/// Strip neg/abs modifier wrappers from a printed operand, returning the
/// bare register or literal name.  Handles `-v1`, `neg(v1)`, `|v1|`,
/// `abs(v1)`, and combinations like `-|v1|` or `neg(abs(v1))`.
/// Also strips True16 `.l`/`.h` suffixes from the resulting bare name.
static std::string stripModifiers(StringRef Op) {
  Op = Op.trim();
  if (Op.starts_with("-"))
    Op = Op.drop_front(1);
  if (Op.starts_with("neg(") && Op.ends_with(")"))
    Op = Op.drop_front(4).drop_back(1);
  if (Op.starts_with("|") && Op.ends_with("|"))
    Op = Op.drop_front(1).drop_back(1);
  if (Op.starts_with("abs(") && Op.ends_with(")"))
    Op = Op.drop_front(4).drop_back(1);
  if (Op.starts_with("-"))
    Op = Op.drop_front(1);
  if (Op.starts_with("|") && Op.ends_with("|"))
    Op = Op.drop_front(1).drop_back(1);
  return stripTrue16Suffix(Op);
}

/// Parse the printed operands of a 2-operand instruction.  Expects MCInst-
/// Printer output shaped like "  mnemonic dst, src0 [modifiers...]".
/// Returns true on success and fills \p Dst, \p Src0 with the trimmed
/// operand strings.  Trailing modifiers (byte_sel, clamp, etc.) are stripped
/// from Src0.
static bool parseTwoOperands(const MCInst &Inst, const LLVMState &LS,
                             std::string &Dst, std::string &Src0) {
  std::string Buf;
  raw_string_ostream OS(Buf);
  LS.MCIP->printInst(&Inst, 0, "", *LS.STI, OS);

  StringRef S = StringRef(Buf).ltrim();
  auto [Mnem, Rest] = S.split(' ');
  Rest = Rest.ltrim();

  SmallVector<StringRef, 4> Parts;
  Rest.split(Parts, ',');
  if (Parts.size() < 2)
    return false;

  Dst = stripTrue16Suffix(Parts[0].trim());
  Src0 = Parts[1].trim().split(' ').first.str();
  return !Dst.empty() && !Src0.empty();
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
  auto [Mnem, Rest] = S.split(' ');
  Rest = Rest.ltrim();

  SmallVector<StringRef, 4> Parts;
  Rest.split(Parts, ',');
  if (Parts.size() < 3)
    return false;

  Dst = stripTrue16Suffix(Parts[0].trim());
  Src0 = Parts[1].trim().str();
  Src1 = Parts[2].trim().split(' ').first.str();
  return !Dst.empty() && !Src0.empty() && !Src1.empty();
}

// ---------------------------------------------------------------------------
// Per-source F32 → UE5M3 conversion with full fixups
// ---------------------------------------------------------------------------

/// Emit assembly for converting one F32 source to a UE5M3 byte in \p Out.
///
/// Handles NaN (→ 0xFF), overflow/Inf (→ 0xFE), RTE rounding of the 7
/// discarded F16 mantissa bits, and source modifiers (neg/abs forwarded
/// from the original instruction via \p Src).
///
/// Register contract:
///   \p Out   — output VGPR, receives UE5M3 byte in bits [7:0]
///   \p Tmp   — scratch VGPR, clobbered
///   \p NanSgpr — SGPR name (e.g. "s0") for saving/restoring the NaN flag
///   \p Src   — full operand with modifiers (used in v_max_f32)
///   \p BareSrc — bare register name (used in v_and_b32 for NaN detect)
///   VCC is clobbered.
///
/// RTE rounding shortcut: rather than extracting round_bit, sticky, and lsb
/// into separate registers (which would require a 3rd VGPR per source), we
/// use the identity: round_up = (guard_bits * 2 + lsb) > 128, where
/// guard_bits = F16[6:0].  This collapses the entire RTE decision into a
/// single integer comparison.  See design doc §4.3 for derivation.
static void emitF32ToUE5M3(raw_string_ostream &OS, StringRef Src,
                            StringRef BareSrc, StringRef Out, StringRef Tmp,
                            StringRef NanSgpr) {
  // NaN detection: (|src| > 0x7F800000) ⇒ NaN.
  // v_and_b32 strips the sign, so neg/abs modifiers don't affect this test.
  // VOPC form: literal in src0, VGPR in src1 (implicit VCC write).
  OS << "v_and_b32 " << Tmp << ", 0x7FFFFFFF, " << BareSrc << "\n";
  OS << "v_cmp_lt_u32 0x7F800000, " << Tmp << "\n";
  OS << "s_mov_b32 " << NanSgpr << ", vcc_lo\n";

  // Clamp to non-negative → F16.  Source modifiers are applied by v_max_f32.
  OS << "v_max_f32 " << Out << ", 0, " << Src << "\n";
  OS << "v_cvt_f16_f32 " << Out << ", " << Out << "\n";

  // RTE rounding: extract guard_bits = F16[6:0], shift to get preliminary
  // byte, then compute round_up = (guard_bits*2 + lsb) > 128.
  OS << "v_and_b32 " << Tmp << ", 0x7F, " << Out << "\n";
  OS << "v_lshrrev_b32 " << Out << ", 7, " << Out << "\n";
  OS << "v_lshlrev_b32 " << Tmp << ", 1, " << Tmp << "\n";
  // v_bfi_b32: dst = (mask & insert) | (~mask & background)
  // With mask=0xFFFFFFFE: copies Tmp[31:1] and Out[0] → guard*2 + lsb
  OS << "v_bfi_b32 " << Tmp << ", 0xFFFFFFFE, " << Tmp << ", " << Out << "\n";
  // v_add_co_ci_u32 adds VCC as carry-in, collapsing the conditional
  // increment into one instruction: Out += (guard*2 + lsb > 128) ? 1 : 0.
  OS << "v_cmp_lt_u32 0x80, " << Tmp << "\n";
  OS << "v_add_co_ci_u32 " << Out << ", 0, " << Out << "\n";

  // Safety clamp: cap at UE5M3 max 0xFE so NaN override ordering works.
  // NOTE: this is effectively a no-op for overflow/Inf — F16 +Inf yields
  // 0xF8 after >>7 which is below 0xFE.  The full UE5M3 exponent-31
  // octave (0xF8–0xFE) is unreachable via F16 intermediate; see design doc
  // §4.3 "Overflow / Inf handling" for accepted limitation details.
  OS << "v_min_u32 " << Out << ", 0xFE, " << Out << "\n";

  // NaN override: if original F32 was NaN, force 0xFF.
  // Load the NaN byte into Tmp (avoids literal in v_cndmask src1).
  OS << "s_mov_b32 vcc_lo, " << NanSgpr << "\n";
  OS << "v_mov_b32 " << Tmp << ", 0xFF\n";
  OS << "v_cndmask_b32 " << Out << ", " << Out << ", " << Tmp << "\n";
}

// ---------------------------------------------------------------------------
// v_cvt_pk_fp8_f32 patch  (Case 1, instruction 1)
// ---------------------------------------------------------------------------

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

  std::string VdstStr, Src0Str, Src1Str;
  if (!parseThreeOperands(DI.Inst, Ctx.LS, VdstStr, Src0Str, Src1Str)) {
    log() << "hotswap: error: cvt_pk_fp8_f32: failed to parse operands at "
          << "offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string Src0Bare = stripModifiers(Src0Str);
  std::string Src1Bare = stripModifiers(Src1Str);

  // findKernelAtOffset takes a virtual address.
  std::string KernelName =
      Ctx.Elf.findKernelAtOffset(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs =
      Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize);
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  ScratchAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                         Ctx.Config.MaxVgprs);

  // 3 scratch VGPRs: T0 (src0 byte), T1 (src1 byte), T2 (shared scratch
  // for NaN detection and RTE rounding intermediates within each source).
  std::optional<unsigned> T0 = Alloc.alloc();
  std::optional<unsigned> T1 = Alloc.alloc();
  std::optional<unsigned> T2 = Alloc.alloc();
  if (!T0 || !T1 || !T2) {
    log() << "hotswap: error: cvt_pk_fp8_f32: unable to allocate 3 scratch "
          << "VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string T0Name = vgprName(*T0);
  std::string T1Name = vgprName(*T1);
  std::string T2Name = vgprName(*T2);

  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  // --- src0 → byte in T0 (scratch: T2) ---
  emitF32ToUE5M3(AsmOS, Src0Str, Src0Bare, T0Name, T2Name, "s0");

  // --- src1 → byte in T1 (scratch: T2) ---
  emitF32ToUE5M3(AsmOS, Src1Str, Src1Bare, T1Name, T2Name, "s1");

  // Pack: T0[15:0] = { byte1, byte0 }
  AsmOS << "v_lshl_or_b32 " << T0Name << ", " << T1Name << ", 8, " << T0Name
        << "\n";

  // Merge into the correct 16-bit half of vdst.
  if (!WriteHigh) {
    AsmOS << "v_bfi_b32 " << VdstStr << ", 0xFFFF, " << T0Name << ", "
          << VdstStr << "\n";
  } else {
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

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  unsigned Extra = Alloc.extraVgprsNeeded();
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchReused += 3 - Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: cvt_pk_fp8_f32: patched CLAMP=1 (E5M3) at offset 0x"
        << utohexstr(DI.Offset) << " (" << ReplacementBytes.size()
        << " bytes, scratch v" << *T0 << "/v" << *T1 << "/v" << *T2
        << ", half=" << (WriteHigh ? "high" : "low") << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// v_cvt_sr_fp8_f32 patch  (Case 1, instruction 2)
// ---------------------------------------------------------------------------

/// Patch a CLAMP=1 `v_cvt_sr_fp8_f32` (stochastic-round F32 → UE5M3).
///
/// The SR path injects stochastic noise into the F32 mantissa before the
/// F16 intermediate conversion, replicating the ISA pseudocode (§17.6.94).
/// Unlike the PK path, no explicit RTE rounding block is needed — the noise
/// makes simple truncation statistically equivalent to unbiased rounding.
/// See design doc v_cvt_sr_fp8_f32.md §3 for the full rationale.
///
/// Scratch: 2 VGPRs (Out + Tmp), 1 SGPR (s0 for NaN flag).
static uint32_t patchCvtSrFp8F32(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  if (DI.Size != 8) {
    log() << "hotswap: error: cvt_sr_fp8_f32: unexpected inst size "
          << DI.Size << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  const uint8_t *Raw = Ctx.Text + DI.Offset;
  bool Clamp = (Raw[1] >> 7) & 1;
  if (!Clamp)
    return 0;

  // OPSEL[3:2] selects which byte of vdst to write (0–3).
  unsigned ByteSel = (Raw[1] >> 5) & 0x3;

  std::string VdstStr, Src0Str, Src1Str;
  if (!parseThreeOperands(DI.Inst, Ctx.LS, VdstStr, Src0Str, Src1Str)) {
    log() << "hotswap: error: cvt_sr_fp8_f32: failed to parse operands at "
          << "offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // Only src0 supports neg/abs modifiers; src1 is U32 with no modifiers.
  std::string Src0Bare = stripModifiers(Src0Str);

  std::string KernelName =
      Ctx.Elf.findKernelAtOffset(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs =
      Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize);
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  ScratchAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                         Ctx.Config.MaxVgprs);

  // 2 scratch VGPRs: Out (conversion result + noise intermediate), Tmp (NaN
  // flag save + noise computation).
  std::optional<unsigned> Out = Alloc.alloc();
  std::optional<unsigned> Tmp = Alloc.alloc();
  if (!Out || !Tmp) {
    log() << "hotswap: error: cvt_sr_fp8_f32: unable to allocate 2 scratch "
          << "VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string OutName = vgprName(*Out);
  std::string TmpName = vgprName(*Tmp);

  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  // --- NaN detection (before max destroys NaN) ---
  // v_and_b32 strips the sign, making this modifier-agnostic.
  AsmOS << "v_and_b32 " << TmpName << ", 0x7FFFFFFF, " << Src0Bare << "\n";
  AsmOS << "v_cmp_lt_u32 0x7F800000, " << TmpName << "\n";
  AsmOS << "s_mov_b32 s0, vcc_lo\n";

  // --- Clamp negative (UE5M3 is unsigned) ---
  // Source modifiers on src0 are applied natively by v_max_f32 VOP3.
  AsmOS << "v_max_f32 " << OutName << ", 0, " << Src0Str << "\n";

  // --- Stochastic noise injection ---
  // Replicate ISA pseudocode: add S1[31:12] to F32 mantissa, truncate back
  // to 23 bits, then reconstruct the perturbed F32 via v_bfi_b32.
  AsmOS << "v_and_b32 " << TmpName << ", 0x007FFFFF, " << OutName << "\n";
  AsmOS << "v_lshrrev_b32 " << OutName << ", 12, " << Src1Str << "\n";
  AsmOS << "v_add_u32 " << TmpName << ", " << TmpName << ", " << OutName
        << "\n";
  AsmOS << "v_and_b32 " << TmpName << ", 0x007FFFFF, " << TmpName << "\n";
  AsmOS << "v_max_f32 " << OutName << ", 0, " << Src0Str << "\n";
  AsmOS << "v_bfi_b32 " << OutName << ", 0x007FFFFF, " << TmpName << ", "
        << OutName << "\n";

  // --- F32 → F16 → UE5M3 (truncation, not RTE — SR noise handles rounding) ---
  AsmOS << "v_cvt_f16_f32 " << OutName << ", " << OutName << "\n";
  AsmOS << "v_lshrrev_b32 " << OutName << ", 7, " << OutName << "\n";

  // --- Overflow clamp (safety) ---
  AsmOS << "v_min_u32 " << OutName << ", 0xFE, " << OutName << "\n";

  // --- NaN override ---
  AsmOS << "s_mov_b32 vcc_lo, s0\n";
  AsmOS << "v_mov_b32 " << TmpName << ", 0xFF\n";
  AsmOS << "v_cndmask_b32 " << OutName << ", " << OutName << ", " << TmpName
        << "\n";

  // --- Byte merge (byte_sel known at patch time) ---
  if (ByteSel == 0) {
    AsmOS << "v_bfi_b32 " << VdstStr << ", 0xFF, " << OutName << ", "
          << VdstStr << "\n";
  } else {
    unsigned Shift = ByteSel * 8;
    static const char *const Masks[] = {nullptr, "0xFF00", "0xFF0000",
                                        "0xFF000000"};
    AsmOS << "v_lshlrev_b32 " << OutName << ", " << Shift << ", " << OutName
          << "\n";
    AsmOS << "v_bfi_b32 " << VdstStr << ", " << Masks[ByteSel] << ", "
          << OutName << ", " << VdstStr << "\n";
  }

  SmallVector<uint8_t> ReplacementBytes = assembleSingleInst(Asm, Ctx.LS);
  if (ReplacementBytes.empty()) {
    log() << "hotswap: error: cvt_sr_fp8_f32: assembly failed for "
          << "replacement at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  if (!queueTrampoline(Ctx, DI.Offset, DI.Size, ReplacementBytes))
    return 0;

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  unsigned Extra = Alloc.extraVgprsNeeded();
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchReused += 2 - Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: cvt_sr_fp8_f32: patched CLAMP=1 (E5M3) at offset 0x"
        << utohexstr(DI.Offset) << " (" << ReplacementBytes.size()
        << " bytes, scratch v" << *Out << "/v" << *Tmp
        << ", byte_sel=" << ByteSel << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// v_cvt_f32_fp8 patch  (Case 1, instruction 3)
// ---------------------------------------------------------------------------

/// Patch a CLAMP=1 `v_cvt_f32_fp8` (UE5M3 → F32 unpack).
///
/// The unpack path extracts a UE5M3 byte from the source VGPR (position
/// selected by OPSEL[1:0]), converts it to F32 via a left-shift-7 → F16 →
/// F32 pipeline, and applies fixups for the exponent-31 octave (bytes
/// 0xF8–0xFE) and UE5M3 NaN (byte 0xFF).  See design doc v_cvt_f32_fp8.md
/// §3–§5 for the full rationale.
///
/// Only VOP3 (_e64) encoding can carry CLAMP=1; VOP1 has no CLAMP bit and
/// is skipped.  No source modifiers exist on this instruction (OPF_NOABS,
/// OPF_NONEG) so no modifier forwarding is needed.
///
/// Scratch: 2 VGPRs (Out + Tmp), 2 SGPRs (s0 for NaN flag, s1 for exp-31).
static uint32_t patchCvtF32Fp8(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  // VOP1 (4 bytes) has no CLAMP bit; only VOP3 (8 bytes) needs patching.
  if (DI.Size != 8)
    return 0;

  const uint8_t *Raw = Ctx.Text + DI.Offset;
  bool Clamp = (Raw[1] >> 7) & 1;
  if (!Clamp)
    return 0;

  // OPSEL[1:0] at dword 0 bits [12:11] (byte 1 bits [4:3]).
  // Reversed mapping: byte_sel = OPSEL[0]*2 + OPSEL[1].
  unsigned Opsel1 = (Raw[1] >> 4) & 1;
  unsigned Opsel0 = (Raw[1] >> 3) & 1;
  unsigned ByteSel = Opsel0 * 2 + Opsel1;

  std::string VdstStr, Src0Str;
  if (!parseTwoOperands(DI.Inst, Ctx.LS, VdstStr, Src0Str)) {
    log() << "hotswap: error: cvt_f32_fp8: failed to parse operands at "
          << "offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string KernelName =
      Ctx.Elf.findKernelAtOffset(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs =
      Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize);
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  ScratchAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                         Ctx.Config.MaxVgprs);

  // 2 scratch VGPRs: Out (byte extraction → F16 path → result),
  // Tmp (exp-31 direct construction + NaN constant).
  std::optional<unsigned> Out = Alloc.alloc();
  std::optional<unsigned> Tmp = Alloc.alloc();
  if (!Out || !Tmp) {
    log() << "hotswap: error: cvt_f32_fp8: unable to allocate 2 scratch "
          << "VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string OutName = vgprName(*Out);
  std::string TmpName = vgprName(*Tmp);

  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  // --- Byte extraction (byte_sel known at patch time) ---
  switch (ByteSel) {
  case 0:
    AsmOS << "v_and_b32 " << OutName << ", 0xFF, " << Src0Str << "\n";
    break;
  case 1:
    AsmOS << "v_bfe_u32 " << OutName << ", " << Src0Str << ", 8, 8\n";
    break;
  case 2:
    AsmOS << "v_bfe_u32 " << OutName << ", " << Src0Str << ", 16, 8\n";
    break;
  case 3:
    AsmOS << "v_lshrrev_b32 " << OutName << ", 24, " << Src0Str << "\n";
    break;
  }

  // --- NaN detection (byte == 0xFF) ---
  AsmOS << "v_cmp_eq_u32 0xFF, " << OutName << "\n";
  AsmOS << "s_mov_b32 s0, vcc_lo\n";

  // --- Exp-31 detection (byte >= 0xF8) ---
  AsmOS << "v_cmp_lt_u32 0xF7, " << OutName << "\n";
  AsmOS << "s_mov_b32 s1, vcc_lo\n";

  // --- Exp-31 direct F32 construction ---
  AsmOS << "v_and_b32 " << TmpName << ", 0x07, " << OutName << "\n";
  AsmOS << "v_lshlrev_b32 " << TmpName << ", 20, " << TmpName << "\n";
  AsmOS << "v_or_b32 " << TmpName << ", 0x47800000, " << TmpName << "\n";

  // --- F16 base path (handles bytes 0x00–0xF7 correctly) ---
  AsmOS << "v_lshlrev_b32 " << OutName << ", 7, " << OutName << "\n";
  AsmOS << "v_cvt_f32_f16 " << OutName << ", " << OutName << "\n";

  // --- Select exp-31 fixup ---
  AsmOS << "s_mov_b32 vcc_lo, s1\n";
  AsmOS << "v_cndmask_b32 " << OutName << ", " << OutName << ", " << TmpName
        << "\n";

  // --- NaN override (byte 0xFF → hardware qNaN 0x7FA3D000) ---
  AsmOS << "s_mov_b32 vcc_lo, s0\n";
  AsmOS << "v_mov_b32 " << TmpName << ", 0x7FA3D000\n";
  AsmOS << "v_cndmask_b32 " << VdstStr << ", " << OutName << ", " << TmpName
        << "\n";

  SmallVector<uint8_t> ReplacementBytes = assembleSingleInst(Asm, Ctx.LS);
  if (ReplacementBytes.empty()) {
    log() << "hotswap: error: cvt_f32_fp8: assembly failed for "
          << "replacement at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  if (!queueTrampoline(Ctx, DI.Offset, DI.Size, ReplacementBytes))
    return 0;

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  unsigned Extra = Alloc.extraVgprsNeeded();
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchReused += 2 - Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: cvt_f32_fp8: patched CLAMP=1 (E5M3) at offset 0x"
        << utohexstr(DI.Offset) << " (" << ReplacementBytes.size()
        << " bytes, scratch v" << *Out << "/v" << *Tmp
        << ", byte_sel=" << ByteSel << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// applyScratchPatches — strong override
// ---------------------------------------------------------------------------

uint32_t applyScratchPatches(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_cvt_pk_fp8_f32")
    return patchCvtPkFp8F32(Ctx, Idx);

  if (Mnem == "v_cvt_sr_fp8_f32")
    return patchCvtSrFp8F32(Ctx, Idx);

  // VOP1 mnemonic is "v_cvt_f32_fp8"; VOP3 may append "_e64" or other
  // suffixes depending on the LLVM build.  Use starts_with to match all
  // encoding variants; the Size and CLAMP checks inside patchCvtF32Fp8
  // filter out non-VOP3 and non-CLAMP forms.
  if (Mnem.starts_with("v_cvt_f32_fp8"))
    return patchCvtF32Fp8(Ctx, Idx);

  return 0;
}

} // namespace hotswap
} // namespace COMGR
