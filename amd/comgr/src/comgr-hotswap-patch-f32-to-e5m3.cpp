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

/// Strip neg/abs modifier wrappers from a printed operand, returning the
/// bare register or literal name.  Handles `-v1`, `neg(v1)`, `|v1|`,
/// `abs(v1)`, and combinations like `-|v1|` or `neg(abs(v1))`.
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
  return Op.str();
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

  Dst = Parts[0].trim().str();
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

  // Overflow / Inf clamp: max representable UE5M3 = 0xFE (no infinity).
  // Must precede NaN override so 0xFF is not clamped.
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
// applyScratchPatches — strong override
// ---------------------------------------------------------------------------

uint32_t applyScratchPatches(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_cvt_pk_fp8_f32")
    return patchCvtPkFp8F32(Ctx, Idx);

  return 0;
}

} // namespace hotswap
} // namespace COMGR
