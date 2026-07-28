//===- patch-wmma-scale16.cpp - WMMA Scale16 decomposition ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Lowers block-16 scaled WMMA (v_wmma_scale16_f32_*) for gfx1250 hardware that
/// only has block-32 scaled WMMA (v_wmma_scale_f32_*). Done exactly, or failing
/// closed when it cannot be applied.
///
/// A block-32 op applies one (scaleA, scaleB) pair across all 32 K-elements of
/// a block, so it cannot honor both block-16 sub-scales of that block at once.
/// The earlier approach collapsed each sub-scale pair with a byte-pair max,
/// which scaled the smaller half by a power of two and silently miscompiled
/// scaled kernels.
///
/// Exact lowering (K-split): the scale is applied per block after the dot and
/// before the accumulate, so we split each block-16 WMMA into two block-32
/// WMMAs chained through the accumulator, each seeing one 16-wide K-subblock:
///
///   pass-low : A' = low-16 K-subblock of A, rest zeroed; even scale bytes;
///              write D (src2 = original C).
///   pass-high: A' = high-16 K-subblock of A, rest zeroed; odd scale bytes;
///              accumulate (src2 = D).
///
/// Masking A alone suffices since A==0 => A*B==0. How a 16-K subblock maps to
/// lanes or VGPRs depends on the matrix-A format:
///   * FP8/BF8: subblocks split by wave lane, so a lane mask isolates one.
///   * FP4/FP6/BF6: a whole 32-block sits in one lane group and the split runs
///     along the VGPR index, so we null the opposite subblock's VGPRs (a lane
///     mask would wrongly zero whole 32-blocks).
/// Each pass's block-32 scale is a byte-gather of the block-16 scale bytes:
/// even bytes feed the low subblocks, odd bytes the high ones.
///
/// The replacement is assembled from textual register names, for which the
<<<<<<< HEAD
/// AMDGPU parser accepts v0-v255. Scale-prefix operands ignore VGPR-MSB, so
/// their generated scale and temporary VGPRs must stay in bank zero. Masked A
/// shares one contiguous low-bank block with those operands. Live values
/// borrowed for that block are saved in above-KD scratch and restored after
/// the final WMMA. Both passes read the same matrix B, so B is copied into the
/// above-KD scratch bank only when its incoming SRC1 bank differs from that
/// bank; a same-bank B is consumed in place. The copy costs B-width moves and
/// B-width above-KD registers, which can flip an occupancy-safe rewrite past
/// its required wave count, so it is not taken unconditionally.
=======
/// AMDGPU parser accepts v0-v255. Scratch may nevertheless live above v255:
/// the generated assembly encodes each scratch VGPR's low byte and brackets
/// mixed-bank instructions with role-specific s_set_vgpr_msb transitions.
/// Matrix B is copied into the scratch bank so the lowered WMMA can use one
/// src1 bank for both its gathered scale and matrix-B operands.
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
///
/// Fail-closed fallback: when the scratch budget (one low-bank A-width-plus-5
/// block, matching save slots, B-width VGPRs when B must be copied, and one
/// scratch SGPR) is unavailable, the pass marks the patch failed so the rewrite
/// returns an error instead of a miscompile. A loud failure beats silent wrong
/// results.
///
/// The 32x16x128_f4 (M=32) variant also needs an M-split; it is not lowered
/// exactly yet and fails closed.
///
//===----------------------------------------------------------------------===//

#include "internal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

<<<<<<< HEAD
#include <algorithm>
=======
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
#include <initializer_list>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// Both Scale16 (VOP3PX3) and regular Scale (VOP3PX2) are 128-bit (16-byte)
// fused instructions: an 8-byte LD_SCALE uop followed by an 8-byte base WMMA
// uop.
static constexpr unsigned VOP3PXSize = 16;

// AMDGPU SRC operand encoding: VGPRs are 256 + N. VgprBankSize comes from
// internal.h so the DS and Scale16 rewrites share one bank definition.
static constexpr unsigned VgprEncBase = 256;
static constexpr unsigned VgprBankSize = 256;

static std::string vgprName(unsigned N) { return ("v" + Twine(N)).str(); }

static std::string encodedVgprName(unsigned Physical) {
  return vgprName(Physical % VgprBankSize);
}

static bool isVgprEncoding(unsigned Enc) { return Enc >= VgprEncBase; }

static std::optional<unsigned> decodeVgprEncoding(unsigned Enc) {
  if (!isVgprEncoding(Enc))
    return std::nullopt;
  return Enc - VgprEncBase;
}

struct LowBankScratchBlock {
  unsigned Base = 0;
  BitVector Preserve;
};

// Allocate one contiguous bank-zero block. Prefer dead registers, then extend
// a small kernel within bank zero, and finally borrow a non-architectural block
// while recording the live values that need save/restore.
static std::optional<LowBankScratchBlock>
allocLowBankScratchBlock(VgprAllocator &Alloc, const BitVector &Forbidden,
                         unsigned Count, unsigned Align) {
  unsigned LowBankLimit =
      std::min({VgprBankSize, Alloc.MaxVgprs,
                static_cast<unsigned>(Alloc.LiveAtPoint.size())});
  if (Count == 0 || Count > LowBankLimit)
    return std::nullopt;

  unsigned ExistingLimit = std::min(LowBankLimit, Alloc.KdAllocatedVgprs);
  for (unsigned Base = 0; Base + Count <= ExistingLimit; ++Base) {
    if (Align > 1 && Base % Align != 0)
      continue;
    bool Available = true;
    for (unsigned I = 0; I < Count; ++I) {
      if (Alloc.LiveAtPoint.test(Base + I) || Forbidden.test(Base + I)) {
        Available = false;
        break;
      }
    }
    if (Available) {
      Alloc.LiveAtPoint.set(Base, Base + Count);
      LowBankScratchBlock Result;
      Result.Base = Base;
      Result.Preserve.resize(Count);
      return Result;
    }
  }

  unsigned Base = Alloc.NextAboveKd;
  if (Align > 1 && Base % Align != 0)
    Base += Align - Base % Align;
  unsigned Step = std::max(Align, 1u);
  for (; Base + Count <= LowBankLimit; Base += Step) {
    bool Available = true;
    for (unsigned I = 0; I < Count; ++I) {
      if (Forbidden.test(Base + I)) {
        Available = false;
        break;
      }
    }
    if (Available) {
      Alloc.ExtraAllocated += Base + Count - Alloc.NextAboveKd;
      Alloc.NextAboveKd = Base + Count;
      Alloc.LiveAtPoint.set(Base, Base + Count);
      LowBankScratchBlock Result;
      Result.Base = Base;
      Result.Preserve.resize(Count);
      return Result;
    }
  }

  for (Base = 0; Base + Count <= LowBankLimit; ++Base) {
    if (Align > 1 && Base % Align != 0)
      continue;
    bool Available = true;
    for (unsigned I = 0; I < Count; ++I) {
      if (Forbidden.test(Base + I)) {
        Available = false;
        break;
      }
    }
    if (Available) {
      LowBankScratchBlock Result;
      Result.Base = Base;
      Result.Preserve.resize(Count);
      for (unsigned I = 0; I < Count; ++I)
        if (Alloc.LiveAtPoint.test(Base + I))
          Result.Preserve.set(I);
      Alloc.LiveAtPoint.set(Base, Base + Count);
      return Result;
    }
  }

  return std::nullopt;
}

// -- LD_SCALE uop field accessors (bytes 0-7) --------------------------------
//   SCALE_SRC0: bits [40:32] = byte[4] + byte[5] bit[0]
//   SCALE_SRC1: bits [49:41] = byte[5] bits[7:1] + byte[6] bits[1:0]

static unsigned extractScaleSrc0(const uint8_t *Raw) {
  return Raw[4] | ((Raw[5] & 0x01) << 8);
}

static unsigned extractScaleSrc1(const uint8_t *Raw) {
  return ((Raw[5] >> 1) & 0x7F) | ((Raw[6] & 0x03) << 7);
}

static void writeScaleSrc0(uint8_t *Raw, unsigned Enc) {
  Raw[4] = Enc & 0xFF;
  Raw[5] = (Raw[5] & 0xFE) | ((Enc >> 8) & 0x01);
}

// Must be called after writeScaleSrc0 (both share byte[5]).
static void writeScaleSrc1(uint8_t *Raw, unsigned Enc) {
  Raw[5] = (Raw[5] & 0x01) | ((Enc & 0x7F) << 1);
  Raw[6] = (Raw[6] & 0xFC) | ((Enc >> 7) & 0x03);
}

// -- Base WMMA uop field accessors (bytes 8-15) ------------------------------
//   VDST: byte[8] (8-bit raw VGPR number, no +256)
//   SRC0: byte[12] + byte[13] bit[0] (9-bit; matrix A)
//   SRC1: byte[13] bits[7:1] + byte[14] bits[1:0] (9-bit; matrix B)
//   SRC2: byte[14] bits[7:2] + byte[15] bits[2:0] (9-bit; accumulator C)
//
// Field positions are the VOP3P operand layout of the base WMMA uop, which is
// the second 8-byte half of the fused encoding. Confirm them against MC rather
// than by inspection, varying one operand at a time:
//
//   echo 'v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], \
//         v[40:47], v2, v3' \
//     | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -show-encoding
//
// gives ...,0x33,0xcc,0x08,0x31,0xa2,0x04; moving SRC1 to v[64:79] changes only
// byte[13], 0x31 -> 0x81. That is ((64 & 0x7f) << 1) with byte[13] bit[0]
// holding SRC0's bit[8], matching the SRC0/SRC1 split described above.

static unsigned extractVdst(const uint8_t *Raw) { return Raw[8]; }

static unsigned extractSrc2(const uint8_t *Raw) {
  return ((Raw[14] >> 2) & 0x3F) | ((Raw[15] & 0x07) << 6);
}

static void writeSrc0(uint8_t *Raw, unsigned Enc) {
  Raw[12] = Enc & 0xFF;
  Raw[13] = (Raw[13] & 0xFE) | ((Enc >> 8) & 0x01);
}

static void writeSrc1(uint8_t *Raw, unsigned Enc) {
  Raw[13] = (Raw[13] & 0x01) | ((Enc & 0x7F) << 1);
  Raw[14] = (Raw[14] & 0xFC) | ((Enc >> 7) & 0x03);
}

static void writeSrc2(uint8_t *Raw, unsigned Enc) {
  Raw[14] = (Raw[14] & 0x03) | ((Enc & 0x3F) << 2);
  Raw[15] = (Raw[15] & 0xF8) | ((Enc >> 6) & 0x07);
}

// -- VOP3PX3 -> VOP3PX2 encoding rewrite -------------------------------------
//
// Turns a block-16 (VOP3PX3) scaled WMMA into a block-32 (VOP3PX2) one: copies
// the 16-byte instruction, swaps the LD_SCALE opcode byte (taken from a
// template assembly so no opcode bits are hardcoded), writes the new block-32
// scale sources, and bakes scale_src2 = VGPR0. scale_src2 is unused on
// VOP3PX2, but leaving it 0 makes the SQ mis-decode it as an SGPR and stall;
// baking it also keeps the bytes idempotent across passes. Matrix reuse bits
// are cleared because both replacement passes substitute matrix operands. All
// other base-WMMA bytes (VDST, SRC0/1/2, matrix formats, neg modifiers) survive
// the byte copy and are patched by the caller.
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
  Rewritten[2] = Template[2];
  constexpr unsigned MatrixAReuseBit = 13;
  constexpr unsigned MatrixBReuseBit = 14;
  static_assert(MatrixAReuseBit / 8 == MatrixBReuseBit / 8);
  constexpr uint8_t MatrixReuseMask =
      (1u << (MatrixAReuseBit % 8)) | (1u << (MatrixBReuseBit % 8));
  Rewritten[MatrixAReuseBit / 8] &= static_cast<uint8_t>(~MatrixReuseMask);
  writeScaleSrc0(Rewritten.data(), NewScaleSrc0Enc);
  writeScaleSrc1(Rewritten.data(), NewScaleSrc1Enc);
  Rewritten[6] &= 0x03;                        // clear scale_src2[5:0]
  Rewritten[7] = (Rewritten[7] & 0xF8) | 0x04; // scale_src2[8]=1, clear [7:6]
  return Rewritten;
}

// -- Block-16 scale byte-gather (deinterleave) -------------------------------
//
// Each B64 scale operand holds 8 8-bit block-16 scales across Vn (bytes 0-3)
// and Vn+1 (bytes 4-7). The block-32 scale for K-block j (j=0..3) is the
// low-subblock scale (even byte 2j) for pass-low and the high-subblock scale
// (odd byte 2j+1) for pass-high, packed into one VGPR as
// [byte0..3] = k-block 0..3.

using VgprBankRequirement = std::pair<VgprMsbOperand, unsigned>;

static void
emitModeForOperands(raw_string_ostream &OS, unsigned &CurrentMode,
                    std::initializer_list<VgprBankRequirement> Requirements) {
  unsigned NewMode = CurrentMode;
  for (const VgprBankRequirement &Requirement : Requirements)
    setVgprMsbBank(NewMode, Requirement.first, Requirement.second);
  if (NewMode == CurrentMode)
    return;
<<<<<<< HEAD
  // Drain outstanding XNACK-replayable memory operations before changing the
  // physical VGPR mapping they were issued under.
  //
  // Hardware already guarantees this: MI400 Shader Programming Guide §6.9.7.2
  // ("VMEM Multi-group Replay Operation and Programming", p. 275) lists
  // S_SET_VGPR_MSB among the events before which "hardware stalls and waits for
  // XCNT==0 and completes any rewind/replay actions". The explicit wait is
  // therefore redundant and kept only as a defensive barrier; the WMMA split
  // pass emits its S_SET_VGPR_MSB transitions without one and relies on the
  // documented hardware stall.
  OS << "s_wait_xcnt 0\n";
=======
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
  OS << "s_set_vgpr_msb " << (NewMode | (CurrentMode << 8)) << "\n";
  CurrentMode = NewMode;
}

static void emitGatherEven(raw_string_ostream &OS, unsigned Lo, unsigned Hi,
                           unsigned Dst, unsigned T, unsigned ScratchBank,
                           unsigned &CurrentMode) {
  std::string LoName = encodedVgprName(Lo);
  std::string HiName = encodedVgprName(Hi);
  std::string DstName = encodedVgprName(Dst);
  std::string TName = encodedVgprName(T);

  // Dst = { Lo[7:0], Lo[23:16], Hi[7:0], Hi[23:16] } (bytes 0,2,4,6)
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src1, Lo / VgprBankSize}});
  OS << "v_and_b32 " << DstName << ", 0xff, " << LoName << "\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, Lo / VgprBankSize}});
  OS << "v_bfe_u32 " << TName << ", " << LoName << ", 16, 8\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, ScratchBank},
                       {VgprMsbOperand::Src2, ScratchBank}});
  OS << "v_lshl_or_b32 " << DstName << ", " << TName << ", 8, " << DstName
     << "\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src1, Hi / VgprBankSize}});
  OS << "v_and_b32 " << TName << ", 0xff, " << HiName << "\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, ScratchBank},
                       {VgprMsbOperand::Src2, ScratchBank}});
  OS << "v_lshl_or_b32 " << DstName << ", " << TName << ", 16, " << DstName
     << "\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, Hi / VgprBankSize}});
  OS << "v_bfe_u32 " << TName << ", " << HiName << ", 16, 8\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, ScratchBank},
                       {VgprMsbOperand::Src2, ScratchBank}});
  OS << "v_lshl_or_b32 " << DstName << ", " << TName << ", 24, " << DstName
     << "\n";
}

static void emitGatherOdd(raw_string_ostream &OS, unsigned Lo, unsigned Hi,
                          unsigned Dst, unsigned T, unsigned ScratchBank,
                          unsigned &CurrentMode) {
  std::string LoName = encodedVgprName(Lo);
  std::string HiName = encodedVgprName(Hi);
  std::string DstName = encodedVgprName(Dst);
  std::string TName = encodedVgprName(T);

  // Dst = { Lo[15:8], Lo[31:24], Hi[15:8], Hi[31:24] } (bytes 1,3,5,7)
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, Lo / VgprBankSize}});
  OS << "v_bfe_u32 " << DstName << ", " << LoName << ", 8, 8\n";
  OS << "v_bfe_u32 " << TName << ", " << LoName << ", 24, 8\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, ScratchBank},
                       {VgprMsbOperand::Src2, ScratchBank}});
  OS << "v_lshl_or_b32 " << DstName << ", " << TName << ", 8, " << DstName
     << "\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, Hi / VgprBankSize}});
  OS << "v_bfe_u32 " << TName << ", " << HiName << ", 8, 8\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, ScratchBank},
                       {VgprMsbOperand::Src2, ScratchBank}});
  OS << "v_lshl_or_b32 " << DstName << ", " << TName << ", 16, " << DstName
     << "\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src1, Hi / VgprBankSize}});
  OS << "v_lshrrev_b32 " << TName << ", 24, " << HiName << "\n";
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, ScratchBank},
                       {VgprMsbOperand::Src0, ScratchBank},
                       {VgprMsbOperand::Src2, ScratchBank}});
  OS << "v_lshl_or_b32 " << DstName << ", " << TName << ", 24, " << DstName
     << "\n";
}

// A' = mask ? A : 0, per lane, for W consecutive VGPRs from ABase into SBase.
// MaskImm selects the wave lanes to keep (0x0000FFFF = lanes 0-15).
//
// FP8/BF8 only: a K=32 block's low-16 K-subblock lives in lanes 0-15 and the
// high-16 in lanes 16-31, so a lane mask isolates a subblock.
static void emitLaneMaskCopy(raw_string_ostream &OS, StringRef MaskSgpr,
                             uint32_t MaskImm, unsigned SBase, unsigned ABase,
                             unsigned W, unsigned ScratchBank,
                             unsigned &CurrentMode) {
  OS << "s_mov_b32 " << MaskSgpr << ", 0x" << utohexstr(MaskImm) << "\n";
  for (unsigned I = 0; I < W; ++I) {
    emitModeForOperands(OS, CurrentMode,
                        {{VgprMsbOperand::Dst, ScratchBank},
                         {VgprMsbOperand::Src1, (ABase + I) / VgprBankSize}});
    OS << "v_cndmask_b32_e64 " << encodedVgprName(SBase + I) << ", 0, "
       << encodedVgprName(ABase + I) << ", " << MaskSgpr << "\n";
  }
}

// A' keeps the VGPRs of the low (KeepLow=true) or high 16-K subblocks and zeros
// the rest, copying W consecutive VGPRs from ABase into SBase.
//
// FP4/FP6/BF6: a whole K=32 block sits in one lane group and the low-16/high-16
// split runs along the VGPR index. Subblocks are SubW consecutive VGPRs (FP4=2,
// FP6=3); even-indexed ones are the low halves, odd-indexed the high. A lane
// mask would wrongly zero whole 32-blocks here, so we null the opposite
// subblock's VGPRs instead.
static void emitVgprSelectCopy(raw_string_ostream &OS, bool KeepLow,
                               unsigned SBase, unsigned ABase, unsigned W,
                               unsigned SubW, unsigned ScratchBank,
                               unsigned &CurrentMode) {
  for (unsigned I = 0; I < W; ++I) {
    bool IsLow = ((I / SubW) % 2) == 0;
    if (IsLow == KeepLow) {
      emitModeForOperands(OS, CurrentMode,
                          {{VgprMsbOperand::Dst, ScratchBank},
                           {VgprMsbOperand::Src0, (ABase + I) / VgprBankSize}});
      OS << "v_mov_b32 " << encodedVgprName(SBase + I) << ", "
         << encodedVgprName(ABase + I) << "\n";
    } else {
      emitModeForOperands(OS, CurrentMode,
                          {{VgprMsbOperand::Dst, ScratchBank}});
      OS << "v_mov_b32 " << encodedVgprName(SBase + I) << ", 0\n";
    }
  }
}

<<<<<<< HEAD
static void emitVgprMove(raw_string_ostream &OS, unsigned Dst, unsigned Src,
                         unsigned &CurrentMode) {
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, Dst / VgprBankSize},
                       {VgprMsbOperand::Src0, Src / VgprBankSize}});
  OS << "v_mov_b32 " << encodedVgprName(Dst) << ", " << encodedVgprName(Src)
     << "\n";
}

static void emitVgprCopy(raw_string_ostream &OS, unsigned DstBase,
                         unsigned SrcBase, unsigned W, unsigned &CurrentMode) {
  for (unsigned I = 0; I < W; ++I)
    emitVgprMove(OS, DstBase + I, SrcBase + I, CurrentMode);
}

// Parse a matrix VGPR range from the printer's canonical form.
//
// The operands are read positionally from the printed form rather than through
// getNamedOperandIdx because the fused VOP3PX3 encoding presents one MCInst
// whose matrix operands do not carry the base WMMA's operand names; the printer
// is the layer that resolves a register tuple to its canonical "v[lo:hi]" text,
// including the width implied by the selected matrix format. Commas are stable
// separators in that canonical form, so position is well defined here even
// though it would not be on hand-written assembly.
=======
static void emitVgprCopy(raw_string_ostream &OS, unsigned DstBase,
                         unsigned SrcBase, unsigned W, unsigned ScratchBank,
                         unsigned &CurrentMode) {
  for (unsigned I = 0; I < W; ++I) {
    emitModeForOperands(OS, CurrentMode,
                        {{VgprMsbOperand::Dst, ScratchBank},
                         {VgprMsbOperand::Src0, (SrcBase + I) / VgprBankSize}});
    OS << "v_mov_b32 " << encodedVgprName(DstBase + I) << ", "
       << encodedVgprName(SrcBase + I) << "\n";
  }
}

// Parse a matrix VGPR range from the printer's canonical form.
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
struct VgprRange {
  unsigned Base;
  unsigned Width;
};

static std::optional<VgprRange>
matrixOperandRange(PatchContext &Ctx, const InternalDecodedInst &DI,
                   unsigned OperandIndex) {
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  Ctx.LS.MCIP->printInst(&DI.Inst, /*Address=*/0, /*Annot=*/"", *Ctx.LS.STI,
                         OS);
  StringRef S = StringRef(Buf).trim();
  size_t MnemEnd = S.find_first_of(" \t");
  if (MnemEnd == StringRef::npos)
    return std::nullopt;
  StringRef Rest = S.substr(MnemEnd).ltrim();
  for (unsigned I = 0; I < OperandIndex; ++I) {
    size_t Comma = Rest.find(',');
    if (Comma == StringRef::npos)
      return std::nullopt;
    Rest = Rest.substr(Comma + 1).ltrim();
  }
  size_t End = Rest.find(',');
  StringRef Operand = (End == StringRef::npos) ? Rest : Rest.substr(0, End);
  Operand = Operand.trim();
  if (!Operand.starts_with("v[") || !Operand.ends_with("]"))
    return std::nullopt;
  StringRef Inside = Operand.drop_front(2).drop_back(1);
  StringRef LoS, HiS;
  std::tie(LoS, HiS) = Inside.split(':');
  unsigned Lo = 0, Hi = 0;
  if (LoS.getAsInteger(10, Lo) || HiS.getAsInteger(10, Hi) || Hi < Lo)
    return std::nullopt;
  return VgprRange{Lo, Hi - Lo + 1};
}

// Matrix-A K-subblock masking scheme, chosen by the matrix-A data format.
// The K-split must isolate each 16-K subblock, and how a subblock maps to
// lanes/VGPRs is format-dependent:
//   * FP8/BF8: subblocks split by wave lane  -> Lane mask.
//   * FP6/BF6: subblocks split by VGPR index -> Vgpr select, 3 VGPRs/subblock.
//   * FP4    : subblocks split by VGPR index -> Vgpr select, 2 VGPRs/subblock.
enum class AMaskScheme { Lane, Vgpr };
struct AMaskPlan {
  AMaskScheme Scheme;
  unsigned SubW; // VGPRs per 16-K subblock (Vgpr scheme only)
};

// Parse "matrix_a_fmt:MATRIX_FMT_<fmt>" from the printer's canonical form and
// map it to a masking plan. FP8 is the default when the modifier is omitted.
static std::optional<AMaskPlan> matrixAMaskPlan(PatchContext &Ctx,
                                                const InternalDecodedInst &DI) {
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  Ctx.LS.MCIP->printInst(&DI.Inst, /*Address=*/0, /*Annot=*/"", *Ctx.LS.STI,
                         OS);
  StringRef S(Buf);
  StringRef Key = "matrix_a_fmt:MATRIX_FMT_";
  StringRef Fmt = "FP8"; // omitted modifier => default FP8
  size_t P = S.find(Key);
  if (P != StringRef::npos) {
    StringRef R = S.substr(P + Key.size());
    size_t E = R.find_first_of(" \t\r\n");
    Fmt = (E == StringRef::npos) ? R : R.substr(0, E);
  }
  if (Fmt == "FP8" || Fmt == "BF8")
    return AMaskPlan{AMaskScheme::Lane, /*SubW=*/4};
  if (Fmt == "FP6" || Fmt == "BF6")
    return AMaskPlan{AMaskScheme::Vgpr, /*SubW=*/3};
  if (Fmt == "FP4")
    return AMaskPlan{AMaskScheme::Vgpr, /*SubW=*/2};
  return std::nullopt; // unknown format -> caller fails closed
}

// Fail the whole rewrite closed rather than emit a miscompile.
static uint32_t failClosed(PatchContext &Ctx, const InternalDecodedInst &DI,
                           const Twine &Why) {
  log() << "hotswap: error: wmma_scale16: " << DI.Mnemonic << " at offset 0x"
        << utohexstr(DI.Offset) << ": " << Why
        << "; refusing to return a miscompiled code object.\n";
  Ctx.RequiredPatchFailed = true;
  return 0;
}

// ---------------------------------------------------------------------------
// v_wmma_scale16_f32_16x16x128_f8f6f4 -> exact K-split
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16_16x16(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];

  if (DI.Size != VOP3PXSize)
    return failClosed(Ctx, DI, "unexpected instruction size " + Twine(DI.Size));

  // Skip offsets a prior pass/rewrite already claimed (idempotency).
  for (const Trampoline &T : Ctx.OutTrampolines)
    if (T.OriginalOffset == DI.Offset)
      return 0;

  const uint8_t *Raw = Ctx.Text + DI.Offset;

  std::optional<unsigned> ScaleABase =
      decodeVgprEncoding(extractScaleSrc0(Raw));
  std::optional<unsigned> ScaleBBase =
      decodeVgprEncoding(extractScaleSrc1(Raw));
  if (!ScaleABase || !ScaleBBase)
    return failClosed(Ctx, DI, "non-VGPR block-16 scale operand");

  std::optional<unsigned> ActiveMode = getActiveVgprMsbMode(Ctx, Idx);
  // A compiler-emitted scale16 whose immediately preceding instruction sets
  // the mode already depends on that setter for the original fused operands.
  // Preserve that local contract when unrelated opaque control flow prevents
  // object-wide mode recovery.
  if (!ActiveMode)
    ActiveMode = getLocallyEstablishedVgprMsbMode(Ctx, Idx);
  if (!ActiveMode)
    return failClosed(Ctx, DI, "cannot determine active VGPR-MSB mode");

  unsigned OrigSrc0Bank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src0);
  unsigned OrigSrc1Bank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src1);
<<<<<<< HEAD
  unsigned OrigSrc2Bank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src2);
  unsigned OrigDstBank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Dst);

  // Scale operands are always addressed in bank zero. VGPR-MSB applies to
  // the matrix operands, but not to the Scale16 prefix operands.
  unsigned ScaleALo = *ScaleABase;
  unsigned ScaleAHi = ScaleALo + 1;
  unsigned ScaleBLo = *ScaleBBase;
  unsigned ScaleBHi = ScaleBLo + 1;
  if (ScaleAHi >= VgprBankSize || ScaleBHi >= VgprBankSize)
    return failClosed(Ctx, DI,
                      "block-16 scale tuple crosses the low VGPR bank");
=======
  unsigned OrigDstBank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Dst);

  unsigned ScaleALo = *ScaleABase + OrigSrc0Bank * VgprBankSize;
  unsigned ScaleAHi = ScaleALo + 1;
  unsigned ScaleBLo = *ScaleBBase + OrigSrc1Bank * VgprBankSize;
  unsigned ScaleBHi = ScaleBLo + 1;
  if (ScaleAHi >= Ctx.Config.MaxVgprs || ScaleBHi >= Ctx.Config.MaxVgprs)
    return failClosed(Ctx, DI, "block-16 scale tuple exceeds VGPR capacity");
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)

  std::optional<VgprRange> ARange =
      matrixOperandRange(Ctx, DI, /*OperandIndex=*/1);
  std::optional<VgprRange> BRange =
      matrixOperandRange(Ctx, DI, /*OperandIndex=*/2);
  if (!ARange || !BRange)
    return failClosed(Ctx, DI, "could not determine matrix-A/B VGPR ranges");
  unsigned ABase = ARange->Base + OrigSrc0Bank * VgprBankSize;
  unsigned AWidth = ARange->Width;
  unsigned BBase = BRange->Base + OrigSrc1Bank * VgprBankSize;
  unsigned BWidth = BRange->Width;
  if (ABase + AWidth > Ctx.Config.MaxVgprs ||
      BBase + BWidth > Ctx.Config.MaxVgprs)
    return failClosed(Ctx, DI, "matrix operand exceeds VGPR capacity");

  // The masking scheme depends on the matrix-A data format.
  std::optional<AMaskPlan> Plan = matrixAMaskPlan(Ctx, DI);
  if (!Plan)
    return failClosed(Ctx, DI,
                      "unrecognized matrix_a_fmt for K-subblock split");
  // For the VGPR-select scheme the 16-K subblocks must pair up (low/high)
  // across the matrix-A VGPRs; a partial trailing subblock would be malformed
  // input.
  if (Plan->Scheme == AMaskScheme::Vgpr &&
      (Plan->SubW == 0 || AWidth % (2 * Plan->SubW) != 0))
    return failClosed(Ctx, DI,
                      "matrix-A width " + Twine(AWidth) +
                          " not a multiple of subblock pair " +
                          Twine(2 * Plan->SubW));

  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs = Ctx.Elf.getKernelVgprCount(
      KernelName, getKernelVgprGranuleSize(Ctx, KernelName));
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  VgprAllocator Alloc(Ctx.Liveness.liveBefore(Idx), KdCount,
                      Ctx.Config.MaxVgprs);

<<<<<<< HEAD
  // Low-bank scratch must not overwrite any architectural operand. Matrix B
  // is copied before the scratch is clobbered, but keeping every original
  // input forbidden makes the save/restore contract explicit.
  constexpr unsigned DstWidth = 8;
  unsigned DstBase = extractVdst(Raw) + OrigDstBank * VgprBankSize;
  if (DstBase + DstWidth > Ctx.Config.MaxVgprs)
    return failClosed(Ctx, DI, "destination exceeds VGPR capacity");

  BitVector Forbidden(Ctx.Config.MaxVgprs);
  Forbidden.set(ScaleALo, ScaleAHi + 1);
  Forbidden.set(ScaleBLo, ScaleBHi + 1);
  Forbidden.set(ABase, ABase + AWidth);
  Forbidden.set(BBase, BBase + BWidth);
  Forbidden.set(DstBase, DstBase + DstWidth);
  std::optional<unsigned> Src2Base = decodeVgprEncoding(extractSrc2(Raw));
  if (Src2Base) {
    unsigned Src2Physical = *Src2Base + OrigSrc2Bank * VgprBankSize;
    if (Src2Physical + DstWidth > Ctx.Config.MaxVgprs)
      return failClosed(Ctx, DI, "accumulator exceeds VGPR capacity");
    Forbidden.set(Src2Physical, Src2Physical + DstWidth);
  }

  constexpr unsigned ScalarScratchCount = 5;
  unsigned LowScratchCount = AWidth + ScalarScratchCount;
  std::optional<LowBankScratchBlock> LowScratch =
      allocLowBankScratchBlock(Alloc, Forbidden, LowScratchCount, /*Align=*/2);
  if (!LowScratch)
    return failClosed(Ctx, DI,
                      "no usable bank-zero block for masked A and scales");

  unsigned SBase = LowScratch->Base;
  unsigned ScaleAloReg = SBase + AWidth;
  unsigned ScaleBloReg = ScaleAloReg + 1;
  unsigned ScaleAhiReg = ScaleAloReg + 2;
  unsigned ScaleBhiReg = ScaleAloReg + 3;
  unsigned TmpReg = ScaleAloReg + 4;

  // Every above-KD block lands in the bank the allocator is about to use, so
  // the scratch bank is known before reserving anything in it.
  unsigned ScratchBank = Alloc.NextAboveKd / VgprBankSize;

  // Save slots are only written for low-bank registers that were borrowed while
  // live. A dead or freshly extended block preserves nothing, so reserving the
  // slots anyway would charge the kernel a full A-width block it never touches.
  unsigned SaveBase = 0;
  if (LowScratch->Preserve.any()) {
    unsigned SaveCount = (LowScratchCount + 1) & ~1u;
    std::optional<unsigned> Save = Alloc.allocContiguousAboveKdInBank(
        SaveCount, /*Align=*/2, VgprBankSize);
    if (!Save)
      return failClosed(Ctx, DI,
                        "no single-bank above-KD VGPR block for exact K-split");
    SaveBase = *Save;
  }

  // Both replacement WMMAs read the same matrix B, so a B already addressed by
  // the scratch bank can stay where it is: SRC1 needs one bank across both
  // passes, not a private copy. Copying a same-bank B would add BWidth moves
  // and BWidth above-KD registers, which is enough to push an otherwise
  // occupancy-safe rewrite past its required wave count.
  bool CopyB = OrigSrc1Bank != ScratchBank;
  unsigned BCopyBase = BBase;
  if (CopyB) {
    std::optional<unsigned> BCopy =
        Alloc.allocContiguousAboveKdInBank(BWidth, /*Align=*/2, VgprBankSize);
    if (!BCopy)
      return failClosed(Ctx, DI,
                        "no single-bank above-KD VGPR block for matrix-B copy");
    BCopyBase = *BCopy;
  }
  // The copy may land in a later bank than the save area, so SRC1 follows the
  // block that actually holds B rather than the save-area bank.
  unsigned Src1Bank = BCopyBase / VgprBankSize;
=======
  // Keep every generated operand in one physical VGPR bank. Five scalar
  // temporaries precede an even-aligned masked-A block and a matrix-B copy.
  // Copying B is necessary because both scale-B and matrix-B use the WMMA
  // src1 VGPR-MSB field.
  constexpr unsigned ScalarScratchCount = 5;
  unsigned AOffset = (ScalarScratchCount + 1) & ~1u;
  unsigned BOffset = AOffset + AWidth;
  unsigned ScratchCount = BOffset + BWidth;
  std::optional<unsigned> ScratchBase = Alloc.allocContiguousAboveKdInBank(
      ScratchCount, /*Align=*/2, VgprBankSize);
  if (!ScratchBase)
    return failClosed(Ctx, DI,
                      "no single-bank above-KD VGPR block for exact K-split");

  unsigned ScaleAloReg = *ScratchBase;
  unsigned ScaleBloReg = *ScratchBase + 1;
  unsigned ScaleAhiReg = *ScratchBase + 2;
  unsigned ScaleBhiReg = *ScratchBase + 3;
  unsigned TmpReg = *ScratchBase + 4;
  unsigned SBase = *ScratchBase + AOffset;
  unsigned BCopyBase = *ScratchBase + BOffset;
  unsigned ScratchBank = *ScratchBase / VgprBankSize;
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)

  // The lane-mask scheme (FP8/BF8) needs one scratch SGPR for the wave-lane
  // bitmask; the VGPR-select scheme (FP4/FP6) uses plain v_mov and needs none.
  std::optional<SafeSgprScratchBlock> MaskSgpr;
  std::string MaskS;
  if (Plan->Scheme == AMaskScheme::Lane) {
    MaskSgpr =
        findSafeSgprScratchBlock(Ctx, DI.Offset, /*Count=*/1,
                                 /*Alignment=*/1, "wmma_scale16 lane mask");
    if (!MaskSgpr)
      return failClosed(Ctx, DI, "no scratch SGPR for lane mask");
    MaskS = ("s" + Twine(MaskSgpr->Base)).str();
  }

  // Preamble + pass-low masked copy (assembled together), then pass-high copy.
  std::string PreAsm, HiAsm, PostAsm;
  raw_string_ostream PreOS(PreAsm), HiOS(HiAsm), PostOS(PostAsm);
  unsigned PreMode = *ActiveMode;

<<<<<<< HEAD
  for (unsigned I = 0; I < LowScratchCount; ++I)
    if (LowScratch->Preserve.test(I))
      emitVgprMove(PreOS, SaveBase + I, SBase + I, PreMode);

  if (CopyB)
    emitVgprCopy(PreOS, BCopyBase, BBase, BWidth, PreMode);
  if (Plan->Scheme == AMaskScheme::Lane) {
    // pass-low keeps lanes 0-15 (low-16 subblocks); pass-high lanes 16-31.
    emitLaneMaskCopy(PreOS, MaskS, 0x0000FFFFu, SBase, ABase, AWidth,
                     /*ScratchBank=*/0, PreMode);
  } else {
    // pass-low keeps the low-16 subblock VGPRs; pass-high the high-16 ones.
    emitVgprSelectCopy(PreOS, /*KeepLow=*/true, SBase, ABase, AWidth,
                       Plan->SubW, /*ScratchBank=*/0, PreMode);
  }

  emitGatherEven(PreOS, ScaleALo, ScaleAHi, ScaleAloReg, TmpReg,
                 /*ScratchBank=*/0, PreMode);
  emitGatherEven(PreOS, ScaleBLo, ScaleBHi, ScaleBloReg, TmpReg,
                 /*ScratchBank=*/0, PreMode);
  emitGatherOdd(PreOS, ScaleALo, ScaleAHi, ScaleAhiReg, TmpReg,
                /*ScratchBank=*/0, PreMode);
  emitGatherOdd(PreOS, ScaleBLo, ScaleBHi, ScaleBhiReg, TmpReg,
                /*ScratchBank=*/0, PreMode);

  unsigned WmmaLoMode = *ActiveMode;
  setVgprMsbBank(WmmaLoMode, VgprMsbOperand::Src0, 0);
  setVgprMsbBank(WmmaLoMode, VgprMsbOperand::Src1, Src1Bank);
  emitModeForOperands(
      PreOS, PreMode,
      {{VgprMsbOperand::Src0, 0},
       {VgprMsbOperand::Src1, Src1Bank},
=======
  emitGatherEven(PreOS, ScaleALo, ScaleAHi, ScaleAloReg, TmpReg, ScratchBank,
                 PreMode);
  emitGatherEven(PreOS, ScaleBLo, ScaleBHi, ScaleBloReg, TmpReg, ScratchBank,
                 PreMode);
  emitGatherOdd(PreOS, ScaleALo, ScaleAHi, ScaleAhiReg, TmpReg, ScratchBank,
                PreMode);
  emitGatherOdd(PreOS, ScaleBLo, ScaleBHi, ScaleBhiReg, TmpReg, ScratchBank,
                PreMode);
  emitVgprCopy(PreOS, BCopyBase, BBase, BWidth, ScratchBank, PreMode);
  if (Plan->Scheme == AMaskScheme::Lane) {
    // pass-low keeps lanes 0-15 (low-16 subblocks); pass-high lanes 16-31.
    emitLaneMaskCopy(PreOS, MaskS, 0x0000FFFFu, SBase, ABase, AWidth,
                     ScratchBank, PreMode);
  } else {
    // pass-low keeps the low-16 subblock VGPRs; pass-high the high-16 ones.
    emitVgprSelectCopy(PreOS, /*KeepLow=*/true, SBase, ABase, AWidth,
                       Plan->SubW, ScratchBank, PreMode);
  }

  unsigned WmmaLoMode = *ActiveMode;
  setVgprMsbBank(WmmaLoMode, VgprMsbOperand::Src0, ScratchBank);
  setVgprMsbBank(WmmaLoMode, VgprMsbOperand::Src1, ScratchBank);
  emitModeForOperands(
      PreOS, PreMode,
      {{VgprMsbOperand::Src0, ScratchBank},
       {VgprMsbOperand::Src1, ScratchBank},
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
       {VgprMsbOperand::Src2, getVgprMsbBank(WmmaLoMode, VgprMsbOperand::Src2)},
       {VgprMsbOperand::Dst, getVgprMsbBank(WmmaLoMode, VgprMsbOperand::Dst)}});

  // pass-low WMMA: matrix A = masked copy, scales = even-byte gathers, src2 =
  // original C (preserved by the byte copy).
<<<<<<< HEAD
  SmallVector<uint8_t> WmmaLo =
      rewriteScale16ToScale(Raw, DI.Size, VgprEncBase + ScaleAloReg,
                            VgprEncBase + ScaleBloReg, Ctx.LS);
=======
  SmallVector<uint8_t> WmmaLo = rewriteScale16ToScale(
      Raw, DI.Size, VgprEncBase + (ScaleAloReg % VgprBankSize),
      VgprEncBase + (ScaleBloReg % VgprBankSize), Ctx.LS);
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
  if (WmmaLo.empty())
    return failClosed(Ctx, DI, "pass-low WMMA rewrite failed");
  writeSrc0(WmmaLo.data(), VgprEncBase + (SBase % VgprBankSize));
  writeSrc1(WmmaLo.data(), VgprEncBase + (BCopyBase % VgprBankSize));

  // pass-high WMMA: odd-byte gathers, and src2 = D so it accumulates onto the
  // pass-low result.
<<<<<<< HEAD
  SmallVector<uint8_t> WmmaHi =
      rewriteScale16ToScale(Raw, DI.Size, VgprEncBase + ScaleAhiReg,
                            VgprEncBase + ScaleBhiReg, Ctx.LS);
=======
  SmallVector<uint8_t> WmmaHi = rewriteScale16ToScale(
      Raw, DI.Size, VgprEncBase + (ScaleAhiReg % VgprBankSize),
      VgprEncBase + (ScaleBhiReg % VgprBankSize), Ctx.LS);
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
  if (WmmaHi.empty())
    return failClosed(Ctx, DI, "pass-high WMMA rewrite failed");
  writeSrc0(WmmaHi.data(), VgprEncBase + (SBase % VgprBankSize));
  writeSrc1(WmmaHi.data(), VgprEncBase + (BCopyBase % VgprBankSize));
  writeSrc2(WmmaHi.data(), VgprEncBase + extractVdst(Raw));

  unsigned HiMode = WmmaLoMode;
  if (Plan->Scheme == AMaskScheme::Lane) {
    emitLaneMaskCopy(HiOS, MaskS, 0xFFFF0000u, SBase, ABase, AWidth,
<<<<<<< HEAD
                     /*ScratchBank=*/0, HiMode);
  } else {
    emitVgprSelectCopy(HiOS, /*KeepLow=*/false, SBase, ABase, AWidth,
                       Plan->SubW, /*ScratchBank=*/0, HiMode);
=======
                     ScratchBank, HiMode);
  } else {
    emitVgprSelectCopy(HiOS, /*KeepLow=*/false, SBase, ABase, AWidth,
                       Plan->SubW, ScratchBank, HiMode);
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
  }
  unsigned WmmaHiMode = WmmaLoMode;
  setVgprMsbBank(WmmaHiMode, VgprMsbOperand::Src2, OrigDstBank);
  emitModeForOperands(
      HiOS, HiMode,
<<<<<<< HEAD
      {{VgprMsbOperand::Src0, 0},
       {VgprMsbOperand::Src1, Src1Bank},
       {VgprMsbOperand::Src2, OrigDstBank},
       {VgprMsbOperand::Dst, getVgprMsbBank(WmmaHiMode, VgprMsbOperand::Dst)}});

  int A0Nops = classifyWmmaNops(DI.Mnemonic).A0Nops;
  unsigned PostMode = WmmaHiMode;
  bool RestoreLowScratch = LowScratch->Preserve.any();
  if (RestoreLowScratch) {
    for (int I = 0; I < A0Nops; ++I)
      PostOS << "v_nop\n";
    for (unsigned I = 0; I < LowScratchCount; ++I)
      if (LowScratch->Preserve.test(I))
        emitVgprMove(PostOS, SBase + I, SaveBase + I, PostMode);
  }

=======
      {{VgprMsbOperand::Src0, ScratchBank},
       {VgprMsbOperand::Src1, ScratchBank},
       {VgprMsbOperand::Src2, OrigDstBank},
       {VgprMsbOperand::Dst, getVgprMsbBank(WmmaHiMode, VgprMsbOperand::Dst)}});

  unsigned PostMode = WmmaHiMode;
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
  unsigned ActiveSrc0 = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src0);
  unsigned ActiveSrc1 = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src1);
  unsigned ActiveSrc2 = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src2);
  unsigned ActiveDst = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Dst);
  emitModeForOperands(PostOS, PostMode,
                      {{VgprMsbOperand::Src0, ActiveSrc0},
                       {VgprMsbOperand::Src1, ActiveSrc1},
                       {VgprMsbOperand::Src2, ActiveSrc2},
                       {VgprMsbOperand::Dst, ActiveDst}});

  SmallVector<uint8_t> PreBytes = assembleInstructions(PreAsm, Ctx.LS);
  SmallVector<uint8_t> HiBytes = assembleInstructions(HiAsm, Ctx.LS);
  SmallVector<uint8_t> PostBytes;
  if (!PostAsm.empty())
    PostBytes = assembleInstructions(PostAsm, Ctx.LS);
  if (PreBytes.empty() || HiBytes.empty() ||
      (!PostAsm.empty() && PostBytes.empty()))
    return failClosed(Ctx, DI, "mode-aware preamble assembly failed");

  // gfx1250 WMMA co-exec hazard: the pass-high copy (VALU) overwrites the
  // masked-A block the pass-low WMMA still reads, so it must not co-execute
  // with the in-flight WMMA. Insert the full required v_nop separation between
  // them (trampoline bytes carry none of the compiler's own spacing). The
  // hazard pass re-validates each trampoline against this count as a safety
  // net.
  SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", Ctx.LS);
  if (VNop.empty())
    return failClosed(Ctx, DI, "v_nop assembly failed");

  SmallVector<uint8_t> Replacement;
  Replacement.append(PreBytes.begin(), PreBytes.end());
  Replacement.append(WmmaLo.begin(), WmmaLo.end());
  for (int I = 0; I < A0Nops; ++I)
    Replacement.append(VNop.begin(), VNop.end());
  Replacement.append(HiBytes.begin(), HiBytes.end());
  Replacement.append(WmmaHi.begin(), WmmaHi.end());
  Replacement.append(PostBytes.begin(), PostBytes.end());

  unsigned Extra = Alloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, KernelName, Extra, PatchRequirement::Required) !=
      VgprBumpDecision::Apply)
    return 0; // checkKernelVgprBump set RequiredPatchFailed on the Fail path.

  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement))
    return failClosed(Ctx, DI, "trampoline emission failed");

  if (MaskSgpr && !commitSafeSgprScratchBlock(Ctx, DI.Offset, *MaskSgpr,
                                              "wmma_scale16 lane mask"))
    return failClosed(Ctx, DI, "scratch SGPR commit failed");

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: wmma_scale16: exact K-split at offset 0x"
        << utohexstr(DI.Offset) << " ("
        << (Plan->Scheme == AMaskScheme::Lane ? "lane-mask" : "vgpr-select")
        << ", A=v" << ABase << ":" << (ABase + AWidth - 1) << " -> masked v"
<<<<<<< HEAD
        << SBase << (CopyB ? ", B copy=v" : ", B in place=v") << BCopyBase
        << ":" << (BCopyBase + BWidth - 1) << ", scales=v" << ScaleAloReg
        << ",v" << ScaleBloReg << ",v" << ScaleAhiReg << ",v" << ScaleBhiReg
=======
        << SBase << ", B copy=v" << BCopyBase << ":" << (BCopyBase + BWidth - 1)
>>>>>>> fb2143bd91 ([comgr][hotswap] upstream review snapshot: finite indirect control flow)
        << ", scratch bank " << ScratchBank << ", +" << Extra << " vgpr, "
        << A0Nops << " hazard v_nop, " << Replacement.size() << " bytes)\n";
  return 1;
}

// ---------------------------------------------------------------------------
// patchWmmaScale16 -- dispatch
// ---------------------------------------------------------------------------

static uint32_t applyWmmaScale16PatchesImpl(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_wmma_scale16_f32_16x16x128_f8f6f4")
    return patchWmmaScale16_16x16(Ctx, Idx);

  // The M=32 FP4 form needs an M-split in addition to the K-split; not yet
  // lowered exactly, so fail closed rather than miscompile.
  if (Mnem.starts_with("v_wmma_scale16_f32_"))
    return failClosed(Ctx, Ctx.Decoded[Idx],
                      "block-16 scaled variant has no exact lowering yet");

  return 0;
}

void registerWmmaScale16Patch(HotswapPatchVTable &VT) {
  VT.applyWmmaScale16Patches = &applyWmmaScale16PatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
