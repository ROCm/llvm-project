//===- comgr-hotswap-patch-vop3px-wrap.cpp - VOP3PX wrap patch ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applyVop3pxWrapPatch. VOP3PX2 wrapping for
/// V_WMMA_F32_16X16X128_F8F6F4 on GFX1250 A0 silicon.
///
/// On A0, an async trap fired between LD_SCALE and the WMMA half of a
/// VOP3PX2 pair is unrecoverable. The trap handler can rewind the PC for
/// known-paired forms (see ROCm/rocm-systems commit 74c647e6605, "rocr:
/// GFX12.5 : VOP3PX instruction split in trap handler"), but a standalone
/// v_wmma_f32_16x16x128_f8f6f4 (no preceding LD_SCALE) cannot be safely
/// rewound at trap time.
///
/// Workaround: prepend an inline-zero LD_SCALE prefix to every standalone
/// V_WMMA_F32_16X16X128_F8F6F4, turning it into a fused VOP3PX2 with
/// scale=1.0 (a no-op). The trap handler's rewind path then handles it.
///
/// The replacement is byte-level: an 8-byte LD_SCALE prefix is prepended
/// to the original 8-byte WMMA, leaving the WMMA portion bit-identical.
/// This avoids re-encoding modifier-rich operand layouts (matrix_a_fmt,
/// matrix_b_fmt, neg_lo, neg_hi, matrix_a_reuse, matrix_b_reuse, ...).
///
/// Two-pass operation:
///   1. Decoded[] scan -- wraps user-written standalone WMMAs.
///   2. Trampoline scan -- wraps splitter-emitted WMMAs sitting in
///      trampoline bytes (the K=128 32x16x128_f4 splitter emits f8f6f4
///      into trampolines).
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

// LD_SCALE prefix encoding for inline-0 scale (= scale 1.0):
//   DWORD 0: 0xCC35_0000 (ENCODING=0xCC3, SCLOP=0x5)
//   DWORD 1: 0x0401_0080 = (0x080) | (0x080 << 9) | (0x100 << 18)
//     SCL_SRC0[40:32] = 0x80 (= inline 0)
//     SCL_SRC1[49:41] = 0x80 (= inline 0)
//     Constant[58:50] = 0x100 (= VGPR0; the VOP3PX2 scale_src2 field is
//     architecturally unused, but if left at 0 the SQ mis-decodes it as
//     an SGPR reference and stalls the SALU for 3 cycles; setting it to
//     a VGPR encoding eliminates the false dependency. Same workaround
//     the in-place vop3px2-src2 patch applies to user-emitted VOP3PX2
//     instructions; baking it into the wrap pass's prefix bytes keeps
//     wrap-emitted trampolines stall-free at creation.)
constexpr uint8_t LdScalePrefix[8] = {
    0x00, 0x00, 0x35, 0xCC, // DWORD 0
    0x80, 0x00, 0x01, 0x04, // DWORD 1
};
constexpr size_t LdScalePrefixSize = sizeof(LdScalePrefix);
constexpr size_t WmmaInstSize = 8;

// 9 type combinations of f8f6f4 share the same printed mnemonic; the
// matrix_a_fmt / matrix_b_fmt modifiers distinguish them at the encoding
// level. We don't care about the variant distinction for wrapping -- the
// WMMA bytes are preserved verbatim and only the LD_SCALE prefix is
// prepended.
constexpr StringLiteral StandaloneWmma = "v_wmma_f32_16x16x128_f8f6f4";

// Already-wrapped detection: per ISA doc page 158, the SCALE prefix MUST
// be immediately preceding the WMMA -- no intervening instructions
// allowed. If the previous decoded instruction is a SCALE form, the WMMA
// is the trailing half of an existing VOP3PX2 and must NOT be wrapped.
constexpr StringLiteral AlreadyWrappedScale =
    "v_wmma_scale_f32_16x16x128_f8f6f4";
constexpr StringLiteral AlreadyWrappedScale16 =
    "v_wmma_scale16_f32_16x16x128_f8f6f4";

// Defensive: 32x16x128_f4 should be eliminated by the K=128 splitter
// before the wrap pass runs. V_WMMA_SCALE_F32_32X16X128_F4 doesn't exist
// on A0 and a leftover would cause the trap-handler rewind to misdecode
// garbage.
constexpr StringLiteral F4Mnemonic = "v_wmma_f32_32x16x128_f4";

// Byte-level WMMA detection (high two bytes of DWORD 0): 0xCC33.
constexpr uint8_t WmmaOpcodeByte2 = 0x33;
constexpr uint8_t WmmaOpcodeByte3 = 0xCC;
// LD_SCALE prefix detection (high two bytes of DWORD 0): 0xCC35.
constexpr uint8_t ScalePrefixByte2 = 0x35;
constexpr uint8_t ScalePrefixByte3 = 0xCC;

bool isWmmaBytes(const uint8_t *P) {
  return P[2] == WmmaOpcodeByte2 && P[3] == WmmaOpcodeByte3;
}

bool isLdScaleBytes(const uint8_t *P) {
  return P[2] == ScalePrefixByte2 && P[3] == ScalePrefixByte3;
}

// Per-instruction patches (e.g. the K=128 splitter) record their rewrites
// by appending a Trampoline whose `OriginalOffset` points at the source
// instruction. The actual text-byte overwrite (s_branch over the original)
// only happens later in fixupTrampolineBranches, so within the dispatch
// pipeline the canonical "was this offset patched?" signal is "appears as
// a Trampoline.OriginalOffset", NOT a text-byte check.
bool offsetIsPatched(const std::vector<Trampoline> &Trampolines,
                     uint64_t Offset) {
  for (const Trampoline &T : Trampolines)
    if (T.OriginalOffset == Offset)
      return true;
  return false;
}

// Build a trampoline carrying the wrapped form. Replacement is a fixed
// 16 bytes (8-byte LD_SCALE prefix + 8-byte WMMA copied verbatim from
// text). The branch-back goes at the tail; fixupTrampolineBranches
// re-encodes it once the final layout is known.
Trampoline buildWrappedTrampoline(const uint8_t *OriginalWmmaBytes,
                                  uint64_t OriginalOffset,
                                  uint32_t OriginalSize,
                                  uint64_t TrampTextOffset,
                                  const LLVMState &LS) {
  Trampoline T;
  T.OriginalOffset = OriginalOffset;
  T.OriginalSize = OriginalSize;
  T.Bytes.reserve(LdScalePrefixSize + WmmaInstSize + MinInstSize);
  T.Bytes.insert(T.Bytes.end(), LdScalePrefix,
                 LdScalePrefix + LdScalePrefixSize);
  T.Bytes.insert(T.Bytes.end(), OriginalWmmaBytes,
                 OriginalWmmaBytes + WmmaInstSize);

  SmallVector<uint8_t> Branch =
      LS.encodeSBranch(TrampTextOffset + T.Bytes.size(),
                       OriginalOffset + OriginalSize);
  if (Branch.empty()) {
    T.Bytes.clear();
    return T;
  }
  T.Bytes.insert(T.Bytes.end(), Branch.begin(), Branch.end());
  return T;
}

// Pass 1: wrap user-written standalone WMMAs found in Decoded[] whose
// bytes still match the WMMA encoding (i.e., not already replaced by
// another patch's s_branch).
uint32_t wrapDecodedInstructions(PatchContext &Ctx) {
  uint32_t Patched = 0;
  for (size_t I = 0, E = Ctx.Decoded.size(); I < E; ++I) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Mnemonic != StandaloneWmma)
      continue;
    if (DI.Size != WmmaInstSize) {
      log() << "hotswap: error: VOP3PX wrap: " << DI.Mnemonic
            << " at offset 0x" << utohexstr(DI.Offset)
            << " has unexpected size " << DI.Size << "\n";
      continue;
    }
    if (DI.Offset + DI.Size > Ctx.TextSize)
      continue;
    if (offsetIsPatched(Ctx.OutTrampolines, DI.Offset))
      continue; // Another patch already claimed this offset.
    if (I > 0) {
      const InternalDecodedInst &Prev = Ctx.Decoded[I - 1];
      if (Prev.Mnemonic == AlreadyWrappedScale ||
          Prev.Mnemonic == AlreadyWrappedScale16)
        continue;
    }

    uint64_t TrampTextOffset = Ctx.TextSize;
    for (const Trampoline &T : Ctx.OutTrampolines)
      TrampTextOffset += T.Bytes.size();

    Trampoline T = buildWrappedTrampoline(Ctx.Text + DI.Offset, DI.Offset,
                                          DI.Size, TrampTextOffset, Ctx.LS);
    if (T.Bytes.empty()) {
      log() << "hotswap: error: VOP3PX wrap: trampoline encoding failed at 0x"
            << utohexstr(DI.Offset) << "\n";
      continue;
    }
    Ctx.OutTrampolines.push_back(std::move(T));

    log() << "hotswap: VOP3PX wrap: patched " << DI.Mnemonic << " at offset 0x"
          << utohexstr(DI.Offset) << "\n";
    ++Patched;
  }
  return Patched;
}

// Pass 2: scan trampoline bodies for splitter-emitted standalone WMMAs
// and prepend the LD_SCALE prefix in-place. Trampoline layout (per
// buildTrampoline / buildWrappedTrampoline):
//   [replacement bytes ... ][branch-back 4 bytes]
// We only walk the body, not the branch-back placeholder. Each insert
// grows T.Bytes by LdScalePrefixSize; fixupTrampolineBranches re-encodes
// the branch-back later with the correct trampoline-end offset.
uint32_t wrapTrampolineInstructions(PatchContext &Ctx) {
  uint32_t Patched = 0;
  for (Trampoline &T : Ctx.OutTrampolines) {
    if (T.Bytes.size() < MinInstSize)
      continue;
    size_t BodyEnd = T.Bytes.size() - MinInstSize;
    size_t Pos = 0;
    while (Pos + WmmaInstSize <= BodyEnd) {
      const uint8_t *P = T.Bytes.data() + Pos;
      if (!isWmmaBytes(P)) {
        Pos += MinInstSize;
        continue;
      }
      if (Pos >= LdScalePrefixSize &&
          isLdScaleBytes(T.Bytes.data() + Pos - LdScalePrefixSize)) {
        Pos += WmmaInstSize;
        continue;
      }
      T.Bytes.insert(T.Bytes.begin() + Pos, LdScalePrefix,
                     LdScalePrefix + LdScalePrefixSize);
      BodyEnd += LdScalePrefixSize;
      Pos += LdScalePrefixSize + WmmaInstSize;
      ++Patched;
      log() << "hotswap: VOP3PX wrap: patched in-trampoline WMMA (orig at 0x"
            << utohexstr(T.OriginalOffset) << ")\n";
    }
  }
  return Patched;
}

// Defensive: refuse to retarget if an unsupported 32x16x128_f4 leftover
// exists in Decoded[] -- the K=128 splitter should have eliminated all of
// these. A leftover would cause the trap-handler rewind to misdecode
// garbage, since V_WMMA_SCALE_F32_32X16X128_F4 doesn't exist on A0.
void checkNoF4Leftovers(PatchContext &Ctx) {
  for (const InternalDecodedInst &DI : Ctx.Decoded) {
    if (DI.Mnemonic != F4Mnemonic)
      continue;
    if (DI.Offset + DI.Size > Ctx.TextSize)
      continue;
    if (offsetIsPatched(Ctx.OutTrampolines, DI.Offset))
      continue; // K=128 splitter handled it.
    log() << "hotswap: error: VOP3PX wrap: unsplit " << F4Mnemonic
          << " at 0x" << utohexstr(DI.Offset)
          << " -- K=128 splitter must run before VOP3PX wrap\n";
  }
}

uint32_t applyVop3pxWrapPatchImpl(PatchContext &Ctx) {
  checkNoF4Leftovers(Ctx);
  uint32_t Patched = wrapDecodedInstructions(Ctx);
  Patched += wrapTrampolineInstructions(Ctx);
  return Patched;
}

} // namespace

void registerVop3pxWrapPatch(HotswapPatchVTable &VT) {
  VT.applyVop3pxWrapPatch = &applyVop3pxWrapPatchImpl;
}

} // namespace hotswap
} // namespace COMGR
