//===- comgr-hotswap-patch-wmma-scale16.cpp - WMMA Scale16 decomposition --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Scratch-patch pass for Case 3 of the B0-to-A0 scratch-patch pipeline:
/// WMMA Scale16 (block-16) to regular Scale (block-32) decomposition.
///
/// GFX1250 B0 uses VOP3PX3-encoded `v_wmma_scale16_f32_*` instructions with
/// 64-bit (B64) scale operands carrying block-16 scale granularity. A0 only
/// supports VOP3PX2-encoded `v_wmma_scale_f32_*` with 32-bit (B32) scale
/// operands at block-32 granularity. This file reduces block-16 scales to
/// block-32 via byte-pair max, then rewrites the encoding from VOP3PX3 to
/// VOP3PX2.
///
/// Covered instructions:
///   1. v_wmma_scale16_f32_16x16x128_f8f6f4 — decompose to regular Scale
///   2. v_wmma_scale16_f32_32x16x128_f4     — B0-only, detection + logging
///
/// Design document: docs/scratch-patches/3_wmma_scale16_decomp/README.md
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

static std::string vgprName(unsigned N) { return ("v" + Twine(N)).str(); }

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

// ---------------------------------------------------------------------------
// VOP3PX3 encoding field accessors
// ---------------------------------------------------------------------------
//
// VOP3PX3 and VOP3PX2 are both 128-bit (16-byte) fused instructions: an
// 8-byte LD_SCALE uop followed by an 8-byte base WMMA uop.  The encoding
// differences between VOP3PX3 (Scale16) and VOP3PX2 (regular Scale) are:
//   - Byte[2]: LD_SCALE opcode (0x3A for Scale16, 0x35 for regular Scale)
//   - SCALE_SRC field interpretation (B64 vs B32)
//
// Field positions within the LD_SCALE uop (first 8 bytes):
//   SCALE_SRC0: bits [40:32] = byte[4] bits [7:0] + byte[5] bit[0]
//   SCALE_SRC1: bits [49:41] = byte[5] bits [7:1] + byte[6] bits [1:0]

static unsigned extractScaleSrc0(const uint8_t *Raw) {
  return Raw[4] | ((Raw[5] & 0x01) << 8);
}

static unsigned extractScaleSrc1(const uint8_t *Raw) {
  return ((Raw[5] >> 1) & 0x7F) | ((Raw[6] & 0x03) << 7);
}

// Write a 9-bit SCALE_SRC0 value into bits [40:32], preserving all
// other bits.  Caller must ensure Raw points at 16+ writable bytes.
static void writeScaleSrc0(uint8_t *Raw, unsigned Enc) {
  Raw[4] = Enc & 0xFF;
  Raw[5] = (Raw[5] & 0xFE) | ((Enc >> 8) & 0x01);
}

// Write a 9-bit SCALE_SRC1 value into bits [49:41], preserving all
// other bits.  Must be called AFTER writeScaleSrc0 to avoid clobbering the
// shared byte [5].
static void writeScaleSrc1(uint8_t *Raw, unsigned Enc) {
  Raw[5] = (Raw[5] & 0x01) | ((Enc & 0x7F) << 1);
  Raw[6] = (Raw[6] & 0xFC) | ((Enc >> 7) & 0x03);
}

// AMDGPU SRC operand encoding: VGPRs are encoded as 256 + N.
static constexpr unsigned VgprEncBase = 256;

static bool isVgprEncoding(unsigned Enc) { return Enc >= VgprEncBase; }

// ---------------------------------------------------------------------------
// Block-16 → block-32 scale reduction via VALU preamble
// ---------------------------------------------------------------------------
//
// Each B64 scale operand holds 8 × 8-bit block-16 scales across two VGPRs
// (Vn and Vn+1).  The reduction computes max(even, odd) for each adjacent
// byte pair, producing 4 × 8-bit block-32 scales in one VGPR:
//
//   max(Vn[7:0],     Vn[15:8])    → Vs[7:0]
//   max(Vn[23:16],   Vn[31:24])   → Vs[15:8]
//   max(Vn+1[7:0],   Vn+1[15:8])  → Vs[23:16]
//   max(Vn+1[23:16], Vn+1[31:24]) → Vs[31:24]
//
// Max-exponent is the recommended strategy for E8M0 scales (pure exponent,
// bias 127): taking the larger exponent ensures no element overflows the
// scale range.  For E5M3/E4M3 scales (only valid with FP4 matrices), max
// still provides a safe upper bound.

static void emitScaleReduction(raw_string_ostream &OS, StringRef SrcLo,
                               StringRef SrcHi, StringRef Dst, StringRef T1,
                               StringRef T2) {
  // Byte pair 0: max(SrcLo[7:0], SrcLo[15:8]) → Dst[7:0]
  OS << "v_and_b32 " << T1 << ", 0xFF, " << SrcLo << "\n";
  OS << "v_bfe_u32 " << T2 << ", " << SrcLo << ", 8, 8\n";
  OS << "v_max_u32 " << Dst << ", " << T1 << ", " << T2 << "\n";

  // Byte pair 1: max(SrcLo[23:16], SrcLo[31:24]) → Dst[15:8]
  OS << "v_bfe_u32 " << T1 << ", " << SrcLo << ", 16, 8\n";
  OS << "v_lshrrev_b32 " << T2 << ", 24, " << SrcLo << "\n";
  OS << "v_max_u32 " << T1 << ", " << T1 << ", " << T2 << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T1 << ", 8, " << Dst << "\n";

  // Byte pair 2: max(SrcHi[7:0], SrcHi[15:8]) → Dst[23:16]
  OS << "v_and_b32 " << T1 << ", 0xFF, " << SrcHi << "\n";
  OS << "v_bfe_u32 " << T2 << ", " << SrcHi << ", 8, 8\n";
  OS << "v_max_u32 " << T1 << ", " << T1 << ", " << T2 << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T1 << ", 16, " << Dst << "\n";

  // Byte pair 3: max(SrcHi[23:16], SrcHi[31:24]) → Dst[31:24]
  OS << "v_bfe_u32 " << T1 << ", " << SrcHi << ", 16, 8\n";
  OS << "v_lshrrev_b32 " << T2 << ", 24, " << SrcHi << "\n";
  OS << "v_max_u32 " << T1 << ", " << T1 << ", " << T2 << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T1 << ", 24, " << Dst << "\n";
}

// ---------------------------------------------------------------------------
// VOP3PX3 → VOP3PX2 encoding rewrite
// ---------------------------------------------------------------------------
//
// Both encodings are 128-bit (16-byte) fused instructions.  The rewrite:
//   1. Replaces byte[2] (LD_SCALE opcode: 0x3A → 0x35)
//   2. Replaces SCALE_SRC0/SCALE_SRC1 with scratch VGPR encodings
//
// The opcode constant is obtained by assembling a template VOP3PX2
// instruction, keeping the code free of hardcoded encoding bits.

static constexpr unsigned VOP3PXSize = 16;

static SmallVector<uint8_t> rewriteScale16ToScale(const uint8_t *OrigRaw,
                                                  unsigned OrigSize,
                                                  unsigned NewScaleSrc0Enc,
                                                  unsigned NewScaleSrc1Enc,
                                                  const LLVMState &LS) {
  SmallVector<uint8_t> Template = assembleSingleInst(
      "v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], "
      "v[40:47], v48, v50",
      LS);
  if (Template.size() != VOP3PXSize) {
    log() << "hotswap: error: wmma_scale16: VOP3PX2 template assembly "
          << "produced " << Template.size() << " bytes (expected " << VOP3PXSize
          << ")\n";
    return {};
  }

  SmallVector<uint8_t> Rewritten(OrigRaw, OrigRaw + OrigSize);

  // The LD_SCALE opcode lives at byte[2]: 0x3A for Scale16 (VOP3PX3) and
  // 0x35 for regular Scale (VOP3PX2).  All other base WMMA encoding bytes
  // are identical between the two variants.
  Rewritten[2] = Template[2];

  writeScaleSrc0(Rewritten.data(), NewScaleSrc0Enc);

  // Must be called after writeScaleSrc0 because both share byte [5].
  writeScaleSrc1(Rewritten.data(), NewScaleSrc1Enc);

  return Rewritten;
}

// ---------------------------------------------------------------------------
// v_wmma_scale16_f32_16x16x128_f8f6f4 → v_wmma_scale_f32_16x16x128_f8f6f4
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16_16x16(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];

  if (DI.Size != VOP3PXSize) {
    log() << "hotswap: error: wmma_scale16: unexpected inst size " << DI.Size
          << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // Idempotency: if the preceding instruction is s_branch, this site was
  // already patched (the branch targets the trampoline we previously emitted).
  if (Idx > 0 && StringRef(Ctx.Decoded[Idx - 1].Mnemonic) == "s_branch")
    return 0;

  const uint8_t *Raw = Ctx.Text + DI.Offset;

  unsigned ScaleSrc0Enc = extractScaleSrc0(Raw);
  unsigned ScaleSrc1Enc = extractScaleSrc1(Raw);

  bool NeedReductionA = isVgprEncoding(ScaleSrc0Enc);
  bool NeedReductionB = isVgprEncoding(ScaleSrc1Enc);

  unsigned Src0Lo = 0, Src0Hi = 0, Src1Lo = 0, Src1Hi = 0;
  if (NeedReductionA) {
    Src0Lo = ScaleSrc0Enc - VgprEncBase;
    Src0Hi = Src0Lo + 1;
  }
  if (NeedReductionB) {
    Src1Lo = ScaleSrc1Enc - VgprEncBase;
    Src1Hi = Src1Lo + 1;
  }

  std::string KernelName =
      Ctx.Elf.findKernelAtOffset(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs =
      Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize);
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  ScratchAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                         Ctx.Config.MaxVgprs);

  // Scratch allocation: 1 reduced-scale VGPR per operand needing reduction,
  // plus 2 shared temporaries for byte extraction/max.
  std::optional<unsigned> ScratchA, ScratchB, T1, T2;
  unsigned ScratchCount = 0;

  if (NeedReductionA) {
    ScratchA = Alloc.alloc();
    ++ScratchCount;
  }
  if (NeedReductionB) {
    ScratchB = Alloc.alloc();
    ++ScratchCount;
  }
  if (NeedReductionA || NeedReductionB) {
    T1 = Alloc.alloc();
    T2 = Alloc.alloc();
    ScratchCount += 2;
  }

  if ((NeedReductionA && !ScratchA) || (NeedReductionB && !ScratchB) ||
      ((NeedReductionA || NeedReductionB) && (!T1 || !T2))) {
    log() << "hotswap: error: wmma_scale16: unable to allocate " << ScratchCount
          << " scratch VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // --- Build VALU preamble for scale reduction ---
  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  if (NeedReductionA)
    emitScaleReduction(AsmOS, vgprName(Src0Lo), vgprName(Src0Hi),
                       vgprName(*ScratchA), vgprName(*T1), vgprName(*T2));
  if (NeedReductionB)
    emitScaleReduction(AsmOS, vgprName(Src1Lo), vgprName(Src1Hi),
                       vgprName(*ScratchB), vgprName(*T1), vgprName(*T2));

  SmallVector<uint8_t> PreambleBytes;
  if (NeedReductionA || NeedReductionB) {
    PreambleBytes = assembleSingleInst(Asm, Ctx.LS);
    if (PreambleBytes.empty()) {
      log() << "hotswap: error: wmma_scale16: preamble assembly failed at "
            << "offset 0x" << utohexstr(DI.Offset) << "\n";
      return 0;
    }
  }

  // --- Rewrite the WMMA encoding from VOP3PX3 → VOP3PX2 ---
  unsigned NewSrc0Enc =
      NeedReductionA ? (VgprEncBase + *ScratchA) : ScaleSrc0Enc;
  unsigned NewSrc1Enc =
      NeedReductionB ? (VgprEncBase + *ScratchB) : ScaleSrc1Enc;

  SmallVector<uint8_t> WmmaBytes =
      rewriteScale16ToScale(Raw, DI.Size, NewSrc0Enc, NewSrc1Enc, Ctx.LS);
  if (WmmaBytes.empty())
    return 0;

  // --- Concatenate: VALU preamble + rewritten WMMA → replacement ---
  SmallVector<uint8_t> Replacement;
  Replacement.insert(Replacement.end(), PreambleBytes.begin(),
                     PreambleBytes.end());
  Replacement.insert(Replacement.end(), WmmaBytes.begin(), WmmaBytes.end());

  if (!queueTrampoline(Ctx, DI.Offset, DI.Size, Replacement))
    return 0;

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  unsigned Extra = Alloc.extraVgprsNeeded();
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchReused += ScratchCount - Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: wmma_scale16: patched 16x16 Scale16→Scale at offset 0x"
        << utohexstr(DI.Offset) << " (" << Replacement.size()
        << " bytes, reductionA=" << NeedReductionA
        << ", reductionB=" << NeedReductionB << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// v_wmma_scale16_f32_32x16x128_f4 — B0-only, no A0 counterpart
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16_32x16(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];

  log() << "hotswap: error: wmma_scale16: "
        << "v_wmma_scale16_f32_32x16x128_f4 at offset 0x"
        << utohexstr(DI.Offset)
        << " is B0-only with no A0 counterpart; cannot patch\n";

  return 0;
}

// ---------------------------------------------------------------------------
// patchWmmaScale16 — dispatch for WMMA Scale16 variants
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_wmma_scale16_f32_16x16x128_f8f6f4")
    return patchWmmaScale16_16x16(Ctx, Idx);

  if (Mnem == "v_wmma_scale16_f32_32x16x128_f4")
    return patchWmmaScale16_32x16(Ctx, Idx);

  return 0;
}

// ---------------------------------------------------------------------------
// applyScratchPatches — strong symbol override
// ---------------------------------------------------------------------------
//
// Overrides the weak stub in comgr-hotswap-b0a0.cpp.  Called once per decoded
// instruction during the rewrite loop.  Returns the number of patches applied
// (0 or 1).

static uint32_t applyScratchPatchesImpl(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);
  if (Mnem.starts_with("v_wmma_scale16_f32_"))
    return patchWmmaScale16(Ctx, Idx);
  return 0;
}

void registerWmmaScale16Patch(HotswapPatchVTable &VT) {
  VT.applyScratchPatches = &applyScratchPatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
