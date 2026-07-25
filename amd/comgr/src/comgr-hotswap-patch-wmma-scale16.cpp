//===- comgr-hotswap-patch-wmma-scale16.cpp - WMMA Scale16 decomposition --===//
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
/// AMDGPU parser accepts v0-v255. Scratch may nevertheless live above v255:
/// the generated assembly encodes each scratch VGPR's low byte and brackets
/// mixed-bank instructions with role-specific s_set_vgpr_msb transitions.
/// Matrix B is copied into the scratch bank so the lowered WMMA can use one
/// src1 bank for both its gathered scale and matrix-B operands.
///
/// Fail-closed fallback: when the scratch budget (A-width VGPRs plus a few
/// scale/temp VGPRs and one scratch SGPR) is unavailable, the pass marks the
/// patch failed so the rewrite returns an error instead of a miscompile. A loud
/// failure beats silent wrong results.
///
/// The 32x16x128_f4 (M=32) variant is split into two M=16 halves, and each
/// resulting half is K-split as above, for four exact block-32 WMMAs total.
/// Scratch reuse inside a fully allocated kernel is allowed only when exact
/// all-path physical-register liveness proves each of its four scratch values
/// dead.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <initializer_list>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// Both Scale16 (VOP3PX3) and regular Scale (VOP3PX2) are 128-bit (16-byte)
// fused instructions: an 8-byte LD_SCALE uop followed by an 8-byte base WMMA
// uop.
static constexpr unsigned VOP3PXSize = 16;

// AMDGPU SRC operand encoding: VGPRs are 256 + N.
static constexpr unsigned VgprEncBase = 256;
static constexpr unsigned VgprBankSize = 256;

bool physicalVgprRangeFitsOneBank(unsigned Base, unsigned Width,
                                  unsigned MaxVgprs) {
  return Width != 0 && Base < MaxVgprs && Width <= MaxVgprs - Base &&
         Base / VgprBankSize == (Base + Width - 1) / VgprBankSize;
}

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

static unsigned extractVdst(const uint8_t *Raw) { return Raw[8]; }

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
// baking it also keeps the bytes idempotent across passes. All other base-WMMA
// bytes (VDST, SRC0/1/2, matrix formats, neg modifiers) survive the byte copy
// and are patched by the caller.
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

static void emitModeForOperands(raw_string_ostream &OS, unsigned &CurrentMode,
                                ArrayRef<VgprBankRequirement> Requirements) {
  unsigned NewMode = CurrentMode;
  for (const VgprBankRequirement &Requirement : Requirements)
    setVgprMsbBank(NewMode, Requirement.first, Requirement.second);
  if (NewMode == CurrentMode)
    return;
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

static void emitScalePairSplitInPlace(raw_string_ostream &OS, unsigned Lo,
                                      unsigned Hi, unsigned Tmp,
                                      unsigned &CurrentMode) {
  unsigned PairBank = Lo / VgprBankSize;
  unsigned TmpBank = Tmp / VgprBankSize;
  emitVgprCopy(OS, Tmp, Lo, /*W=*/1, TmpBank, CurrentMode);
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, PairBank},
                       {VgprMsbOperand::Src0, TmpBank},
                       {VgprMsbOperand::Src1, PairBank}});
  OS << "v_perm_b32 " << encodedVgprName(Lo) << ", " << encodedVgprName(Tmp)
     << ", " << encodedVgprName(Hi) << ", 0x06040200\n";
  OS << "v_perm_b32 " << encodedVgprName(Hi) << ", " << encodedVgprName(Tmp)
     << ", " << encodedVgprName(Hi) << ", 0x07050301\n";
}

static void emitScalePairRestoreInPlace(raw_string_ostream &OS, unsigned Lo,
                                        unsigned Hi, unsigned Tmp,
                                        unsigned &CurrentMode) {
  unsigned PairBank = Lo / VgprBankSize;
  unsigned TmpBank = Tmp / VgprBankSize;
  emitVgprCopy(OS, Tmp, Lo, /*W=*/1, TmpBank, CurrentMode);
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Dst, PairBank},
                       {VgprMsbOperand::Src0, TmpBank},
                       {VgprMsbOperand::Src1, PairBank}});
  OS << "v_perm_b32 " << encodedVgprName(Lo) << ", " << encodedVgprName(Tmp)
     << ", " << encodedVgprName(Hi) << ", 0x05010400\n";
  OS << "v_perm_b32 " << encodedVgprName(Hi) << ", " << encodedVgprName(Tmp)
     << ", " << encodedVgprName(Hi) << ", 0x07030602\n";
}

static void emitMaskVgprsInPlace(raw_string_ostream &OS, bool KeepLow,
                                 unsigned Base, unsigned W, unsigned SubW,
                                 unsigned &CurrentMode) {
  for (unsigned I = 0; I != W; ++I) {
    bool IsLow = ((I / SubW) % 2) == 0;
    if (IsLow == KeepLow)
      continue;
    unsigned Physical = Base + I;
    emitModeForOperands(OS, CurrentMode,
                        {{VgprMsbOperand::Dst, Physical / VgprBankSize}});
    OS << "v_mov_b32 " << encodedVgprName(Physical) << ", 0\n";
  }
}

// Parse a matrix VGPR range from the printer's canonical form.
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

struct Scale32PrintedAsm {
  std::string Operands[6]; // dst, matrix A/B, src2, scale A/B
  std::string ModifierSuffix;
};

// Parse the six positional operands of the M=32 scaled WMMA while preserving
// the printer's exact inline-immediate spelling and modifier ordering.
static std::optional<Scale32PrintedAsm>
parseScale32PrintedAsm(PatchContext &Ctx, const InternalDecodedInst &DI) {
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  Ctx.LS.MCIP->printInst(&DI.Inst, /*Address=*/0, /*Annot=*/"", *Ctx.LS.STI,
                         OS);
  StringRef S = StringRef(Buf).trim();
  size_t MnemEnd = S.find_first_of(" \t");
  if (MnemEnd == StringRef::npos)
    return std::nullopt;

  Scale32PrintedAsm Result;
  StringRef Rest = S.substr(MnemEnd).ltrim();
  for (unsigned I = 0; I != 5; ++I) {
    size_t Comma = Rest.find(',');
    if (Comma == StringRef::npos)
      return std::nullopt;
    Result.Operands[I] = Rest.substr(0, Comma).trim().str();
    Rest = Rest.substr(Comma + 1).ltrim();
  }
  size_t ModBegin = Rest.find_first_of(" \t");
  if (ModBegin == StringRef::npos) {
    Result.Operands[5] = Rest.str();
  } else {
    Result.Operands[5] = Rest.substr(0, ModBegin).str();
    Result.ModifierSuffix = Rest.substr(ModBegin).str();
  }
  return Result;
}

static SmallVector<StringRef, 8> tokenizeScaleModifiers(StringRef Suffix) {
  SmallVector<StringRef, 8> Result;
  StringRef Rest = Suffix.ltrim();
  while (!Rest.empty()) {
    size_t Space = Rest.find_first_of(" \t");
    if (Space == StringRef::npos) {
      Result.push_back(Rest);
      break;
    }
    Result.push_back(Rest.substr(0, Space));
    Rest = Rest.substr(Space + 1).ltrim();
  }
  return Result;
}

static bool parsePackedScaleModifier(StringRef Token, StringRef Name,
                                     std::array<StringRef, 3> &Bits) {
  if (!Token.starts_with(Name) || !Token.ends_with("]"))
    return false;
  Token = Token.drop_front(Name.size());
  if (!Token.starts_with(":["))
    return false;
  SmallVector<StringRef, 3> Parts;
  Token.drop_front(2).drop_back(1).split(Parts, ",");
  if (Parts.size() != 3)
    return false;
  for (unsigned I = 0; I != 3; ++I) {
    Bits[I] = Parts[I].trim();
    if (Bits[I] != "0" && Bits[I] != "1")
      return false;
  }
  return true;
}

static bool isKnownScale32Modifier(StringRef Token) {
  if (Token == "matrix_a_reuse" || Token == "matrix_b_reuse" ||
      Token == "matrix_a_scale:MATRIX_SCALE_ROW1" ||
      Token == "matrix_b_scale:MATRIX_SCALE_ROW1" ||
      Token == "matrix_a_scale_fmt:MATRIX_SCALE_FMT_E8" ||
      Token == "matrix_a_scale_fmt:MATRIX_SCALE_FMT_E5M3" ||
      Token == "matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3" ||
      Token == "matrix_b_scale_fmt:MATRIX_SCALE_FMT_E8" ||
      Token == "matrix_b_scale_fmt:MATRIX_SCALE_FMT_E5M3" ||
      Token == "matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3")
    return true;
  std::array<StringRef, 3> Bits;
  return parsePackedScaleModifier(Token, "neg_lo", Bits) ||
         parsePackedScaleModifier(Token, "neg_hi", Bits);
}

// The split changes both matrix register layout and the K-pass accumulator.
// Reuse promises therefore no longer describe the generated sequence and are
// removed. On each high-K pass src2 is the low-pass result, not the original
// C, so clear the src2 neg/abs bit instead of applying C's modifier twice.
static std::optional<std::string>
transformScale32ModifierSuffix(StringRef Suffix, bool HighKPass) {
  std::string Result;
  for (StringRef Token : tokenizeScaleModifiers(Suffix)) {
    if (!isKnownScale32Modifier(Token)) {
      log() << "hotswap: error: wmma_scale16: unsupported M=32 modifier token "
               "\""
            << Token << "\"\n";
      return std::nullopt;
    }
    if (Token == "matrix_a_reuse" || Token == "matrix_b_reuse")
      continue;

    std::array<StringRef, 3> Bits;
    if (HighKPass && (parsePackedScaleModifier(Token, "neg_lo", Bits) ||
                      parsePackedScaleModifier(Token, "neg_hi", Bits))) {
      if (Bits[0] == "0" && Bits[1] == "0")
        continue;
      StringRef Name = Token.take_front(Token.find(':'));
      Result += (" " + Name + ":[" + Bits[0] + "," + Bits[1] + ",0]").str();
      continue;
    }
    Result += ' ';
    Result += Token.str();
  }
  return Result;
}

static std::string encodedVgprRange(unsigned PhysicalBase, unsigned Width) {
  assert(Width > 0);
  unsigned EncodedBase = PhysicalBase % VgprBankSize;
  return formatv("v[{0}:{1}]", EncodedBase, EncodedBase + Width - 1).str();
}

static void emitScale32Half(raw_string_ostream &OS, unsigned DstBase,
                            unsigned MatrixABase, unsigned MatrixBBase,
                            StringRef Src2, unsigned ScaleAReg,
                            unsigned ScaleBReg, StringRef ModifierSuffix) {
  OS << "v_wmma_scale_f32_16x16x128_f8f6f4 " << encodedVgprRange(DstBase, 8)
     << ", " << encodedVgprRange(MatrixABase, 8) << ", "
     << encodedVgprRange(MatrixBBase, 8) << ", " << Src2 << ", "
     << encodedVgprName(ScaleAReg) << ", " << encodedVgprName(ScaleBReg)
     << " matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4"
     << ModifierSuffix << "\n";
}

// Prefer one exact-liveness-proven dead block inside the kernel's declared
// allocation. Fully allocated production kernels often have no above-KD
// headroom even though a particular WMMA site has a large dead interval.
// Search high-to-low within each physical bank so every generated range keeps
// one VGPR-MSB setting. Fall back to the allocator's normal above-KD growth.
static std::optional<unsigned>
allocContiguousDeadOrAboveInBank(VgprAllocator &Alloc, unsigned Count,
                                 unsigned Align, unsigned BankSize,
                                 bool AllowDeadReuse) {
  if (Count == 0 || Count > BankSize || Align == 0)
    return std::nullopt;

  if (AllowDeadReuse) {
    unsigned BankCount = (Alloc.KdAllocatedVgprs + BankSize - 1) / BankSize;
    for (unsigned ReverseBank = BankCount; ReverseBank != 0; --ReverseBank) {
      unsigned Bank = ReverseBank - 1;
      unsigned BankBegin = Bank * BankSize;
      unsigned BankEnd = std::min(Alloc.KdAllocatedVgprs, BankBegin + BankSize);
      if (BankEnd < BankBegin + Count)
        continue;

      unsigned Base = BankEnd - Count;
      Base -= Base % Align;
      while (Base >= BankBegin) {
        bool AllDead = true;
        for (unsigned V = Base; V != Base + Count; ++V) {
          if (V >= static_cast<unsigned>(Alloc.LiveAtPoint.size()) ||
              Alloc.LiveAtPoint.test(V)) {
            AllDead = false;
            break;
          }
        }
        if (AllDead) {
          Alloc.LiveAtPoint.set(Base, Base + Count);
          return Base;
        }
        if (Base < BankBegin + Align)
          break;
        Base -= Align;
      }
    }
  }
  return Alloc.allocContiguousAboveKdInBank(Count, Align, BankSize);
}

static bool rangesOverlap(unsigned ABase, unsigned AWidth, unsigned BBase,
                          unsigned BWidth) {
  return ABase < BBase + BWidth && BBase < ABase + AWidth;
}

static bool sameRange(unsigned ABase, unsigned AWidth, unsigned BBase,
                      unsigned BWidth) {
  return ABase == BBase && AWidth == BWidth;
}

static void reserveVgprRange(VgprAllocator &Alloc, unsigned Base,
                             unsigned Width) {
  assert(Width > 0 && Base <= Alloc.LiveAtPoint.size() &&
         Width <= Alloc.LiveAtPoint.size() - Base);
  Alloc.LiveAtPoint.set(Base, Base + Width);
}

struct EncodedVgprRange {
  unsigned Base = 0;
  unsigned Width = 0;
  bool FullDwords = false;
};

static std::optional<unsigned> parseScalarVgprName(StringRef Name,
                                                   bool &IsPartial) {
  IsPartial = false;
  if (!Name.consume_front("VGPR"))
    return std::nullopt;
  size_t Digits = Name.find_first_not_of("0123456789");
  StringRef Number = Digits == StringRef::npos ? Name : Name.take_front(Digits);
  unsigned Index = 0;
  if (Number.empty() || Number.getAsInteger(10, Index))
    return std::nullopt;
  if (Digits == StringRef::npos)
    return Index;
  StringRef Suffix = Name.drop_front(Digits);
  if (Suffix == "_LO16" || Suffix == "_HI16") {
    IsPartial = true;
    return Index;
  }
  return std::nullopt;
}

// Convert an explicit MC VGPR or VGPR tuple to its encoded v0..v255 interval.
// True16 operands are identified but never accepted as a full-value kill.
static std::optional<EncodedVgprRange>
getEncodedVgprRange(MCRegister Reg, const MCRegisterInfo &MRI) {
  if (!Reg)
    return std::nullopt;

  bool IsPartial = false;
  if (std::optional<unsigned> Scalar =
          parseScalarVgprName(MRI.getName(Reg), IsPartial))
    return EncodedVgprRange{*Scalar, 1, !IsPartial};

  SmallVector<unsigned, 16> Scalars;
  for (MCPhysReg Sub : MRI.subregs(Reg)) {
    bool SubPartial = false;
    std::optional<unsigned> Index =
        parseScalarVgprName(MRI.getName(Sub), SubPartial);
    if (Index && !SubPartial)
      Scalars.push_back(*Index);
  }
  if (Scalars.empty())
    return std::nullopt;
  llvm::sort(Scalars);
  Scalars.erase(std::unique(Scalars.begin(), Scalars.end()), Scalars.end());
  for (unsigned I = 1; I != Scalars.size(); ++I)
    if (Scalars[I] != Scalars.front() + I)
      return std::nullopt;
  return EncodedVgprRange{Scalars.front(),
                          static_cast<unsigned>(Scalars.size()), true};
}

bool isVectorRegisterOrAlias(MCRegister Reg, const MCRegisterInfo &MRI) {
  if (!Reg)
    return false;
  for (MCRegAliasIterator Alias(Reg, &MRI, /*IncludeSelf=*/true);
       Alias.isValid(); ++Alias) {
    StringRef Name = MRI.getName(*Alias);
    if (Name.contains("VGPR") || Name.contains("AGPR"))
      return true;
  }
  return false;
}

static bool setPhysicalVgprRange(BitVector &Out, const EncodedVgprRange &Range,
                                 unsigned Bank, unsigned MaxVgprs) {
  if (Range.Width == 0 || Range.Base >= VgprBankSize ||
      Range.Width > VgprBankSize - Range.Base)
    return false;
  unsigned Base = Range.Base + Bank * VgprBankSize;
  if (Base >= MaxVgprs || Range.Width > MaxVgprs - Base)
    return false;
  Out.set(Base, Base + Range.Width);
  return true;
}

struct PhysicalVgprAccess {
  BitVector Uses;
  BitVector FullDefs;
  bool Valid = true;

  explicit PhysicalVgprAccess(unsigned MaxVgprs)
      : Uses(MaxVgprs), FullDefs(MaxVgprs) {}
};

// The M=32 Scale16 MC layout is mirrored and runtime-validated by the
// lowering below. Its two scale operands reuse the matrix source banks:
// matrix-A/scale-A use src0, matrix-B/scale-B use src1, and the accumulator
// uses src2. Other instruction layouts remain conservative-all-banks unless
// their role is structurally unambiguous.
static std::optional<VgprMsbOperand>
getExactSourceRole(const InternalDecodedInst &DI, unsigned OperandIndex,
                   unsigned NumDefs) {
  if (DI.Mnemonic == "v_wmma_scale16_f32_32x16x128_f4") {
    switch (OperandIndex) {
    case 1:
    case 5:
      return VgprMsbOperand::Src0;
    case 2:
    case 6:
      return VgprMsbOperand::Src1;
    case 4:
      return VgprMsbOperand::Src2;
    default:
      return std::nullopt;
    }
  }
  if (StringRef(DI.Mnemonic).starts_with("ds_") && OperandIndex == NumDefs)
    return VgprMsbOperand::Src0;
  return std::nullopt;
}

// Resolve every explicit access through the exact persistent VGPR-MSB mode.
// Sources with a validated architectural role use that role's bank. Every
// other source maps through the union of src0/src1/src2 banks, which can only
// add uses, never hide one. Explicit full-width definitions use the
// architectural dst bank. Tied definitions are reads of their incoming
// destination. Implicit/partial VGPR operands cannot prove a kill and
// conservatively block the encoded range in every bank.
static PhysicalVgprAccess getPhysicalVgprAccess(const InternalDecodedInst &DI,
                                                const LLVMState &LS,
                                                unsigned Mode,
                                                unsigned MaxVgprs) {
  PhysicalVgprAccess Result(MaxVgprs);
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  const MCRegisterInfo &MRI = *LS.MRI;
  unsigned DstBank = getVgprMsbBank(Mode, VgprMsbOperand::Dst);
  SmallVector<unsigned, 3> SrcBanks = {
      getVgprMsbBank(Mode, VgprMsbOperand::Src0),
      getVgprMsbBank(Mode, VgprMsbOperand::Src1),
      getVgprMsbBank(Mode, VgprMsbOperand::Src2)};
  llvm::sort(SrcBanks);
  SrcBanks.erase(std::unique(SrcBanks.begin(), SrcBanks.end()), SrcBanks.end());

  auto AddEveryBankUse = [&](const EncodedVgprRange &Range) {
    for (unsigned Bank = 0; Bank * VgprBankSize < MaxVgprs; ++Bank)
      if (!setPhysicalVgprRange(Result.Uses, Range, Bank, MaxVgprs))
        Result.Valid = false;
  };

  unsigned NumDefs = Desc.getNumDefs();
  for (unsigned I = 0, E = DI.Inst.getNumOperands(); I != E; ++I) {
    const MCOperand &Op = DI.Inst.getOperand(I);
    if (!Op.isReg() || !Op.getReg())
      continue;
    std::optional<EncodedVgprRange> Range =
        getEncodedVgprRange(MCRegister(Op.getReg()), MRI);
    if (!Range) {
      if (isVectorRegisterOrAlias(MCRegister(Op.getReg()), MRI))
        Result.Valid = false;
      continue;
    }

    bool IsDef = I < NumDefs;
    if (!IsDef) {
      if (std::optional<VgprMsbOperand> Role =
              getExactSourceRole(DI, I, NumDefs)) {
        unsigned Bank = getVgprMsbBank(Mode, *Role);
        if (!setPhysicalVgprRange(Result.Uses, *Range, Bank, MaxVgprs))
          Result.Valid = false;
      } else {
        for (unsigned Bank : SrcBanks)
          if (!setPhysicalVgprRange(Result.Uses, *Range, Bank, MaxVgprs))
            Result.Valid = false;
      }
      continue;
    }

    int TiedTo = Desc.getOperandConstraint(I, MCOI::TIED_TO);
    if (TiedTo >= 0) {
      if (!setPhysicalVgprRange(Result.Uses, *Range, DstBank, MaxVgprs))
        Result.Valid = false;
    }
    if (!Range->FullDwords) {
      AddEveryBankUse(*Range);
      continue;
    }
    if (!setPhysicalVgprRange(Result.FullDefs, *Range, DstBank, MaxVgprs))
      Result.Valid = false;
  }

  for (MCPhysReg Implicit : Desc.implicit_uses()) {
    std::optional<EncodedVgprRange> Range =
        getEncodedVgprRange(MCRegister(Implicit), MRI);
    if (Range)
      AddEveryBankUse(*Range);
    else if (isVectorRegisterOrAlias(MCRegister(Implicit), MRI))
      Result.Valid = false;
  }
  for (MCPhysReg Implicit : Desc.implicit_defs()) {
    std::optional<EncodedVgprRange> Range =
        getEncodedVgprRange(MCRegister(Implicit), MRI);
    if (Range)
      AddEveryBankUse(*Range);
    else if (isVectorRegisterOrAlias(MCRegister(Implicit), MRI))
      Result.Valid = false;
  }
  return Result;
}

static bool hasDynamicVgprAddressing(ArrayRef<InternalDecodedInst> Decoded,
                                     size_t Begin, size_t End) {
  for (size_t I = Begin; I != End; ++I) {
    StringRef Mnemonic = Decoded[I].Mnemonic;
    if (Mnemonic.contains("movrel") || Mnemonic.contains("gpr_idx") ||
        Mnemonic.starts_with("s_setreg"))
      return true;
  }
  return false;
}

std::optional<BitVector>
computeForwardDeadVgprs(ArrayRef<ForwardVgprProofNode> Nodes, size_t EntryNode,
                        unsigned MaxVgprs) {
  if (Nodes.empty() || EntryNode >= Nodes.size() || MaxVgprs == 0)
    return std::nullopt;
  for (const ForwardVgprProofNode &Node : Nodes) {
    if (Node.Uses.size() != MaxVgprs || Node.FullDefs.size() != MaxVgprs)
      return std::nullopt;
    for (size_t Successor : Node.Successors)
      if (Successor >= Nodes.size())
        return std::nullopt;
  }

  std::vector<BitVector> AliveAt(Nodes.size(), BitVector(MaxVgprs));
  AliveAt[EntryNode].set();
  BitVector Unsafe(MaxVgprs);
  SmallVector<size_t, 64> Worklist;
  Worklist.push_back(EntryNode);

  while (!Worklist.empty()) {
    size_t Index = Worklist.pop_back_val();
    BitVector Alive = AliveAt[Index];
    if (Alive.none())
      continue;

    const ForwardVgprProofNode &Node = Nodes[Index];
    if (Node.Opaque) {
      Unsafe |= Alive;
      continue;
    }

    BitVector UsedAlive = Alive;
    UsedAlive &= Node.Uses;
    Unsafe |= UsedAlive;
    Alive.reset(Node.Uses);
    Alive.reset(Node.FullDefs);

    if (Node.HasUnsafeExit)
      Unsafe |= Alive;
    if (Node.Successors.empty()) {
      if (!Node.SafeTerminal)
        Unsafe |= Alive;
      continue;
    }

    for (size_t Successor : Node.Successors) {
      BitVector NewBits = Alive;
      NewBits.reset(AliveAt[Successor]);
      if (NewBits.none())
        continue;
      AliveAt[Successor] |= Alive;
      Worklist.push_back(Successor);
    }
  }

  // A value that can circulate around a cycle without a use or full kill is
  // not accepted as scratch. Detect such cycles in the per-value subgraph:
  // Kahn removal leaves exactly the nodes belonging to, or fed only by, a
  // surviving cycle. This is intentionally conservative for non-terminating
  // paths and makes loop handling independent of worklist visitation order.
  SmallVector<unsigned, 64> InDegree(Nodes.size());
  SmallVector<size_t, 64> Queue;
  BitVector Included(Nodes.size());
  for (unsigned V = 0; V != MaxVgprs; ++V) {
    if (Unsafe.test(V))
      continue;
    Included.reset();
    unsigned IncludedCount = 0;
    for (size_t I = 0; I != Nodes.size(); ++I) {
      const ForwardVgprProofNode &Node = Nodes[I];
      if (AliveAt[I].test(V) && !Node.Opaque && !Node.Uses.test(V) &&
          !Node.FullDefs.test(V)) {
        Included.set(I);
        ++IncludedCount;
      }
    }
    if (IncludedCount == 0)
      continue;

    llvm::fill(InDegree, 0);
    for (int I = Included.find_first(); I >= 0; I = Included.find_next(I))
      for (size_t Successor : Nodes[static_cast<size_t>(I)].Successors)
        if (Included.test(Successor))
          ++InDegree[Successor];
    Queue.clear();
    for (int I = Included.find_first(); I >= 0; I = Included.find_next(I))
      if (InDegree[static_cast<size_t>(I)] == 0)
        Queue.push_back(static_cast<size_t>(I));

    unsigned Removed = 0;
    while (!Queue.empty()) {
      size_t I = Queue.pop_back_val();
      ++Removed;
      for (size_t Successor : Nodes[I].Successors)
        if (Included.test(Successor) && --InDegree[Successor] == 0)
          Queue.push_back(Successor);
    }
    if (Removed != IncludedCount)
      Unsafe.set(V);
  }

  BitVector Safe(MaxVgprs);
  Safe.set();
  Safe.reset(Unsafe);
  return Safe;
}

struct ScaleForwardGraph {
  std::vector<ForwardVgprProofNode> Nodes;
  std::vector<size_t> GlobalIndices;
  std::vector<int16_t> ModeBefore;
  BitVector UndecodedVop3Heads;
  size_t EntryNode = 0;
};

// The B0 f4gemm object contains a legacy VOP3 opcode that the A0 gfx1250 MC
// decoder intentionally does not recognize. The failed decode advances only
// one dword, so its second dword appears as a separate unknown instruction.
//
// Recognize only the exact observed encoding:
//   * legacy VOP3 major 0x34 and opcode 0x31;
//   * vdst encoding zero, with only the observed bit-14 modifier variation;
//   * scalar source encodings, in either observed second-dword spelling.
//
// We do not assign the unknown opcode semantics. Instead, conservatively model
// encoded v0 as a use in every physical bank (so it cannot be scratch), model
// no kill, and skip the split continuation dword. This is enough to traverse
// the instruction without relying on whether opcode 0x31 has a vector or
// scalar destination on B0.
static bool recognizeUndecodedB0Vop3ScalarSources(PatchContext &Ctx,
                                                  size_t Global,
                                                  unsigned MaxVgprs,
                                                  BitVector &ConservativeUses) {
  if (Global + 1 >= Ctx.Decoded.size())
    return false;
  const InternalDecodedInst &Head = Ctx.Decoded[Global];
  const InternalDecodedInst &Tail = Ctx.Decoded[Global + 1];
  if (Head.DecodeSucceeded || Head.Size != MinInstSize ||
      Tail.Offset != Head.Offset + MinInstSize || Head.Offset > Ctx.TextSize ||
      2 * MinInstSize > Ctx.TextSize - Head.Offset)
    return false;

  const uint8_t *Raw = Ctx.Text + Head.Offset;
  uint32_t Word0 = support::endian::read32le(Raw);
  uint32_t Word1 = support::endian::read32le(Raw + MinInstSize);
  constexpr uint32_t Bit14 = 1u << 14;
  if ((Word0 & ~Bit14) != 0xd0310000u || (Word1 != 0 && Word1 != 0x00100000u))
    return false;

  // A direct/declared entry into the decoder-created continuation slot would
  // make treating the pair as one instruction unsound.
  if (Ctx.DirectControlFlow.Targets.contains(Tail.Offset) ||
      llvm::is_contained(Ctx.DeclaredEntries, Tail.Offset))
    return false;

  for (unsigned Physical = 0; Physical < MaxVgprs; Physical += VgprBankSize)
    ConservativeUses.set(Physical);
  return true;
}

static std::optional<ScaleForwardGraph>
buildScaleForwardGraph(PatchContext &Ctx, size_t SiteIdx, unsigned EntryMode,
                       unsigned MaxVgprs) {
  if (!Ctx.LS.MIA || !Ctx.LS.MCII || !Ctx.LS.MRI ||
      SiteIdx >= Ctx.Decoded.size())
    return std::nullopt;

  std::optional<ElfView::FunctionTextRange> Owner =
      Ctx.Elf.findFunctionTextRangeAtOffset(Ctx.Decoded[SiteIdx].Offset);
  if (!Owner || SiteIdx + 1 >= Ctx.Decoded.size())
    return std::nullopt;

  size_t BeginIndex = SiteIdx;
  while (BeginIndex > 0 && Ctx.Decoded[BeginIndex - 1].Offset >= Owner->Begin)
    --BeginIndex;
  size_t EndIndex = SiteIdx + 1;
  while (EndIndex < Ctx.Decoded.size() &&
         Ctx.Decoded[EndIndex].Offset < Owner->End)
    ++EndIndex;
  if (BeginIndex == EndIndex || SiteIdx + 1 >= EndIndex)
    return std::nullopt;

  ScaleForwardGraph Graph;
  size_t Count = EndIndex - BeginIndex;
  Graph.Nodes.reserve(Count);
  Graph.GlobalIndices.reserve(Count);
  for (size_t I = BeginIndex; I != EndIndex; ++I) {
    Graph.Nodes.emplace_back(MaxVgprs);
    Graph.GlobalIndices.push_back(I);
  }
  Graph.UndecodedVop3Heads.resize(Count);
  Graph.EntryNode = SiteIdx + 1 - BeginIndex;

  DenseMap<uint64_t, size_t> IndexAtOffset;
  for (size_t Local = 0; Local != Count; ++Local)
    IndexAtOffset[Ctx.Decoded[Graph.GlobalIndices[Local]].Offset] = Local;

  auto AddFallthrough = [&](size_t Local) {
    if (Local + 1 < Count)
      Graph.Nodes[Local].Successors.push_back(Local + 1);
    else
      Graph.Nodes[Local].HasUnsafeExit = true;
  };

  for (size_t Local = 0; Local != Count; ++Local) {
    size_t Global = Graph.GlobalIndices[Local];
    const InternalDecodedInst &DI = Ctx.Decoded[Global];
    ForwardVgprProofNode &Node = Graph.Nodes[Local];

    // A later loop iteration reaches the replacement itself. Scratch excludes
    // every original operand, and the replacement defines its scratch before
    // reading it, so no incoming scratch value can be observed there.
    if (Global == SiteIdx) {
      Node.SafeTerminal = true;
      continue;
    }
    if (!DI.DecodeSucceeded && recognizeUndecodedB0Vop3ScalarSources(
                                   Ctx, Global, MaxVgprs, Node.Uses)) {
      if (Local + 2 >= Count)
        Node.HasUnsafeExit = true;
      else
        Node.Successors.push_back(Local + 2);
      Graph.UndecodedVop3Heads.set(Local);
      continue;
    }
    if (!DI.DecodeSucceeded ||
        hasDynamicVgprAddressing(Ctx.Decoded, Global, Global + 1)) {
      Node.Opaque = true;
      continue;
    }
    if (DI.Inst.getOpcode() == Ctx.LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == Ctx.LS.SEndPgmSavedOpcode) {
      Node.SafeTerminal = true;
      continue;
    }
    if (Ctx.LS.MIA->isCall(DI.Inst) || Ctx.LS.MIA->isIndirectBranch(DI.Inst) ||
        Ctx.LS.MIA->isReturn(DI.Inst)) {
      Node.Opaque = true;
      continue;
    }
    if (Ctx.LS.MIA->isBranch(DI.Inst)) {
      uint64_t Target = 0;
      if (!Ctx.LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target)) {
        Node.Opaque = true;
        continue;
      }
      DenseMap<uint64_t, size_t>::const_iterator TargetIt =
          IndexAtOffset.find(Target);
      if (TargetIt == IndexAtOffset.end())
        Node.HasUnsafeExit = true;
      else
        Node.Successors.push_back(TargetIt->second);
      if (Ctx.LS.MIA->isConditionalBranch(DI.Inst))
        AddFallthrough(Local);
      else if (!Ctx.LS.MIA->isUnconditionalBranch(DI.Inst))
        Node.Opaque = true;
      continue;
    }
    if (Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI)) {
      Node.Opaque = true;
      continue;
    }
    AddFallthrough(Local);
  }

  Graph.ModeBefore.assign(Count, VgprMsbUnreachable);
  Graph.ModeBefore[Graph.EntryNode] = static_cast<int16_t>(EntryMode & 0xff);
  SmallVector<size_t, 64> Worklist;
  Worklist.push_back(Graph.EntryNode);
  for (size_t Next = 0; Next != Worklist.size(); ++Next) {
    size_t Local = Worklist[Next];
    const ForwardVgprProofNode &Node = Graph.Nodes[Local];
    if (Node.Opaque || Node.SafeTerminal)
      continue;
    int16_t Out = Graph.UndecodedVop3Heads.test(Local)
                      ? Graph.ModeBefore[Local]
                      : transferExactVgprMsbMode(
                            Graph.ModeBefore[Local],
                            Ctx.Decoded[Graph.GlobalIndices[Local]], Ctx.LS);
    for (size_t Successor : Node.Successors) {
      int16_t Old = Graph.ModeBefore[Successor];
      int16_t Merged = Old == VgprMsbUnreachable ? Out
                       : Old == Out              ? Old
                                                 : VgprMsbUnknown;
      if (Merged != Old) {
        Graph.ModeBefore[Successor] = Merged;
        Worklist.push_back(Successor);
      }
    }
  }
  return Graph;
}

// Return physical VGPR values whose incoming contents cannot be observed on
// any continuation path after SiteIdx. This deliberately does not consume the
// generic LivenessInfo: its weak in-tree implementation is encoded-v0..v255
// conservative liveness, not a proof over gfx1250's four physical banks.
static std::optional<BitVector>
computeForwardDeadPhysicalVgprs(PatchContext &Ctx, size_t SiteIdx,
                                unsigned EntryMode, unsigned MaxVgprs) {
  if (Ctx.DirectControlFlow.HasUnresolvedTargets ||
      Ctx.DirectControlFlow.HasUnboundedIndirectEntries || !Ctx.LS.MIA ||
      !Ctx.LS.MCII || !Ctx.LS.MRI || SiteIdx >= Ctx.Decoded.size())
    return std::nullopt;

  std::optional<ScaleForwardGraph> Graph =
      buildScaleForwardGraph(Ctx, SiteIdx, EntryMode, MaxVgprs);
  if (!Graph)
    return std::nullopt;

  for (size_t Local = 0; Local != Graph->Nodes.size(); ++Local) {
    ForwardVgprProofNode &Node = Graph->Nodes[Local];
    if (Node.Opaque || Node.SafeTerminal)
      continue;
    if (Graph->UndecodedVop3Heads.test(Local))
      continue;

    int16_t Mode = Graph->ModeBefore[Local];
    if (Mode == VgprMsbUnreachable)
      continue;
    unsigned AccessMode = Mode >= 0 ? static_cast<unsigned>(Mode) : 0;
    PhysicalVgprAccess Access = getPhysicalVgprAccess(
        Ctx.Decoded[Graph->GlobalIndices[Local]], Ctx.LS, AccessMode, MaxVgprs);
    if (!Access.Valid)
      return std::nullopt;
    if (Mode < 0 && (Access.Uses.any() || Access.FullDefs.any()))
      return std::nullopt;
    Node.Uses = std::move(Access.Uses);
    Node.FullDefs = std::move(Access.FullDefs);
  }
  return computeForwardDeadVgprs(Graph->Nodes, Graph->EntryNode, MaxVgprs);
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
  unsigned OrigDstBank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Dst);

  unsigned ScaleALo = *ScaleABase + OrigSrc0Bank * VgprBankSize;
  unsigned ScaleAHi = ScaleALo + 1;
  unsigned ScaleBLo = *ScaleBBase + OrigSrc1Bank * VgprBankSize;
  unsigned ScaleBHi = ScaleBLo + 1;
  if (ScaleAHi >= Ctx.Config.MaxVgprs || ScaleBHi >= Ctx.Config.MaxVgprs)
    return failClosed(Ctx, DI, "block-16 scale tuple exceeds VGPR capacity");

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
       {VgprMsbOperand::Src2, getVgprMsbBank(WmmaLoMode, VgprMsbOperand::Src2)},
       {VgprMsbOperand::Dst, getVgprMsbBank(WmmaLoMode, VgprMsbOperand::Dst)}});

  // pass-low WMMA: matrix A = masked copy, scales = even-byte gathers, src2 =
  // original C (preserved by the byte copy).
  SmallVector<uint8_t> WmmaLo = rewriteScale16ToScale(
      Raw, DI.Size, VgprEncBase + (ScaleAloReg % VgprBankSize),
      VgprEncBase + (ScaleBloReg % VgprBankSize), Ctx.LS);
  if (WmmaLo.empty())
    return failClosed(Ctx, DI, "pass-low WMMA rewrite failed");
  writeSrc0(WmmaLo.data(), VgprEncBase + (SBase % VgprBankSize));
  writeSrc1(WmmaLo.data(), VgprEncBase + (BCopyBase % VgprBankSize));

  // pass-high WMMA: odd-byte gathers, and src2 = D so it accumulates onto the
  // pass-low result.
  SmallVector<uint8_t> WmmaHi = rewriteScale16ToScale(
      Raw, DI.Size, VgprEncBase + (ScaleAhiReg % VgprBankSize),
      VgprEncBase + (ScaleBhiReg % VgprBankSize), Ctx.LS);
  if (WmmaHi.empty())
    return failClosed(Ctx, DI, "pass-high WMMA rewrite failed");
  writeSrc0(WmmaHi.data(), VgprEncBase + (SBase % VgprBankSize));
  writeSrc1(WmmaHi.data(), VgprEncBase + (BCopyBase % VgprBankSize));
  writeSrc2(WmmaHi.data(), VgprEncBase + extractVdst(Raw));

  unsigned HiMode = WmmaLoMode;
  if (Plan->Scheme == AMaskScheme::Lane) {
    emitLaneMaskCopy(HiOS, MaskS, 0xFFFF0000u, SBase, ABase, AWidth,
                     ScratchBank, HiMode);
  } else {
    emitVgprSelectCopy(HiOS, /*KeepLow=*/false, SBase, ABase, AWidth,
                       Plan->SubW, ScratchBank, HiMode);
  }
  unsigned WmmaHiMode = WmmaLoMode;
  setVgprMsbBank(WmmaHiMode, VgprMsbOperand::Src2, OrigDstBank);
  emitModeForOperands(
      HiOS, HiMode,
      {{VgprMsbOperand::Src0, ScratchBank},
       {VgprMsbOperand::Src1, ScratchBank},
       {VgprMsbOperand::Src2, OrigDstBank},
       {VgprMsbOperand::Dst, getVgprMsbBank(WmmaHiMode, VgprMsbOperand::Dst)}});

  unsigned PostMode = WmmaHiMode;
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
  int A0Nops = classifyWmmaNops(DI.Mnemonic).A0Nops;
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
  Info.ScratchRegs.resize(Ctx.Config.MaxVgprs);
  Info.ScratchRegs.set(*ScratchBase, *ScratchBase + ScratchCount);
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: wmma_scale16: exact K-split at offset 0x"
        << utohexstr(DI.Offset) << " ("
        << (Plan->Scheme == AMaskScheme::Lane ? "lane-mask" : "vgpr-select")
        << ", A=v" << ABase << ":" << (ABase + AWidth - 1) << " -> masked v"
        << SBase << ", B copy=v" << BCopyBase << ":" << (BCopyBase + BWidth - 1)
        << ", scratch bank " << ScratchBank << ", +" << Extra << " vgpr, "
        << A0Nops << " hazard v_nop, " << Replacement.size() << " bytes)\n";
  return 1;
}

// ---------------------------------------------------------------------------
// v_wmma_scale16_f32_32x16x128_f4 -> exact M+K split
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16_32x16(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];

  if (DI.Size != VOP3PXSize)
    return failClosed(Ctx, DI, "unexpected instruction size " + Twine(DI.Size));
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
  if (!ActiveMode)
    ActiveMode = getLocallyEstablishedVgprMsbMode(Ctx, Idx);
  if (!ActiveMode) {
    std::string Detail = "cannot determine active VGPR-MSB mode";
    if (Ctx.DirectControlFlow.HasUnresolvedTargets)
      Detail += " (unresolved control-flow target)";
    if (Ctx.DirectControlFlow.HasUnboundedIndirectEntries)
      Detail += " (unbounded indirect entry)";
    return failClosed(Ctx, DI, Detail);
  }

  std::optional<Scale32PrintedAsm> Printed = parseScale32PrintedAsm(Ctx, DI);
  if (!Printed)
    return failClosed(Ctx, DI, "could not parse canonical instruction");
  std::optional<std::string> LowSuffix =
      transformScale32ModifierSuffix(Printed->ModifierSuffix,
                                     /*HighKPass=*/false);
  std::optional<std::string> HighSuffix =
      transformScale32ModifierSuffix(Printed->ModifierSuffix,
                                     /*HighKPass=*/true);
  if (!LowSuffix || !HighSuffix)
    return failClosed(Ctx, DI, "unsupported modifier combination");

  std::optional<VgprRange> DRange =
      matrixOperandRange(Ctx, DI, /*OperandIndex=*/0);
  std::optional<VgprRange> ARange =
      matrixOperandRange(Ctx, DI, /*OperandIndex=*/1);
  std::optional<VgprRange> BRange =
      matrixOperandRange(Ctx, DI, /*OperandIndex=*/2);
  if (!DRange || !ARange || !BRange || DRange->Width != 16 ||
      ARange->Width != 16 || BRange->Width != 8)
    return failClosed(Ctx, DI, "unexpected M=32 matrix operand widths/layout");

  // The M=32 profile mirrors the common VOP3P layout in its first five MC
  // operands: vdst, src0, src1, src2 modifiers, src2. Validate that mirror at
  // runtime so a TableGen layout change fails closed.
  if (DI.Inst.getNumOperands() < 5)
    return failClosed(Ctx, DI, "truncated M=32 MC operand layout");
  const MCOperand &Src2Op = DI.Inst.getOperand(4);
  bool Src2IsImm = Src2Op.isImm();
  std::optional<VgprRange> CRange;
  if (Src2Op.isReg())
    CRange = matrixOperandRange(Ctx, DI, /*OperandIndex=*/3);
  else if (!Src2IsImm)
    return failClosed(Ctx, DI, "unsupported non-VGPR/non-immediate src2");
  if (CRange && CRange->Width != 16)
    return failClosed(Ctx, DI, "src2 and destination widths differ");
  if (!Src2IsImm && !CRange)
    return failClosed(Ctx, DI, "could not determine src2 VGPR range");

  unsigned Src0Bank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src0);
  unsigned Src1Bank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src1);
  unsigned Src2Bank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src2);
  unsigned DstBank = getVgprMsbBank(*ActiveMode, VgprMsbOperand::Dst);

  unsigned DBase = DRange->Base + DstBank * VgprBankSize;
  unsigned ABase = ARange->Base + Src0Bank * VgprBankSize;
  unsigned BBase = BRange->Base + Src1Bank * VgprBankSize;
  unsigned CBase = CRange ? CRange->Base + Src2Bank * VgprBankSize : 0;
  unsigned ScaleALo = *ScaleABase + Src0Bank * VgprBankSize;
  unsigned ScaleAHi = ScaleALo + 1;
  unsigned ScaleBLo = *ScaleBBase + Src1Bank * VgprBankSize;
  unsigned ScaleBHi = ScaleBLo + 1;

  if (!physicalVgprRangeFitsOneBank(DBase, 16, Ctx.Config.MaxVgprs) ||
      !physicalVgprRangeFitsOneBank(ABase, 16, Ctx.Config.MaxVgprs) ||
      !physicalVgprRangeFitsOneBank(BBase, 8, Ctx.Config.MaxVgprs) ||
      (CRange &&
       !physicalVgprRangeFitsOneBank(CBase, 16, Ctx.Config.MaxVgprs)) ||
      !physicalVgprRangeFitsOneBank(ScaleALo, 2, Ctx.Config.MaxVgprs) ||
      !physicalVgprRangeFitsOneBank(ScaleBLo, 2, Ctx.Config.MaxVgprs))
    return failClosed(Ctx, DI,
                      "M=32 operand exceeds or crosses a physical VGPR bank");

  // The split reads A in stages after its first D-half write. The fused source
  // instruction reads all of A before writing D, so any D/A overlap would
  // otherwise let the replacement destroy a later A read.
  if (rangesOverlap(DBase, 16, ABase, 16))
    return failClosed(Ctx, DI,
                      "destination overlaps matrix A across staged reads");
  if (rangesOverlap(DBase, 16, BBase, 8))
    return failClosed(Ctx, DI,
                      "destination overlaps matrix B across staged reads");
  if (rangesOverlap(ABase, 16, BBase, 8))
    return failClosed(Ctx, DI,
                      "matrix A overlaps matrix B during in-place masking");

  // Exact D==C is the ordinary in-place accumulator form. A disjoint C is
  // likewise safe. Reject partial/cross-half overlap: writing DLo could
  // otherwise destroy CHi before the second low-K pass consumes it.
  if (CRange && rangesOverlap(DBase, 16, CBase, 16) &&
      !sameRange(DBase, 16, CBase, 16))
    return failClosed(Ctx, DI,
                      "partial destination/src2 overlap across staged reads");

  auto ScaleOverlaps = [&](unsigned Base, unsigned Width) {
    return rangesOverlap(ScaleALo, 2, Base, Width) ||
           rangesOverlap(ScaleBLo, 2, Base, Width);
  };
  if (ScaleOverlaps(DBase, 16) || ScaleOverlaps(ABase, 16) ||
      ScaleOverlaps(BBase, 8) || (CRange && ScaleOverlaps(CBase, 16)) ||
      rangesOverlap(ScaleALo, 2, ScaleBLo, 2))
    return failClosed(
        Ctx, DI,
        "scale pair overlaps a staged matrix operand or the other scale");
  if (CRange && rangesOverlap(ABase, 16, CBase, 16))
    return failClosed(Ctx, DI,
                      "matrix A overlaps src2 during in-place masking");

  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs = Ctx.Elf.getKernelVgprCount(
      KernelName, getKernelVgprGranuleSize(Ctx, KernelName));
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);
  VgprAllocator Alloc(Ctx.Liveness.liveBefore(Idx), KdCount,
                      Ctx.Config.MaxVgprs);

  // Replace generic encoded-register liveness with a physical-bank all-path
  // proof when available. An unset bit is the only state the in-KD allocator
  // accepts; failure leaves the allocator conservative-all-live and therefore
  // permits only ordinary above-KD growth.
  std::optional<BitVector> ForwardDead = computeForwardDeadPhysicalVgprs(
      Ctx, Idx, *ActiveMode, Ctx.Config.MaxVgprs);
  if (ForwardDead) {
    for (int V = ForwardDead->find_first(); V >= 0;
         V = ForwardDead->find_next(V))
      Alloc.LiveAtPoint.reset(static_cast<unsigned>(V));
    unsigned BestBase = 0;
    unsigned BestWidth = 0;
    for (unsigned BankBase = 0; BankBase < Ctx.Config.MaxVgprs;
         BankBase += VgprBankSize) {
      unsigned BankEnd = std::min(Ctx.Config.MaxVgprs, BankBase + VgprBankSize);
      for (unsigned V = BankBase; V != BankEnd;) {
        if (!ForwardDead->test(V)) {
          ++V;
          continue;
        }
        unsigned Begin = V;
        while (V != BankEnd && ForwardDead->test(V))
          ++V;
        if (V - Begin > BestWidth) {
          BestBase = Begin;
          BestWidth = V - Begin;
        }
      }
    }
    log() << "hotswap: wmma_scale16: physical forward-dead proof at offset 0x"
          << utohexstr(DI.Offset) << " found " << ForwardDead->count()
          << " VGPRs; longest single-bank run ";
    if (BestWidth)
      log() << "v" << BestBase << ":" << (BestBase + BestWidth - 1) << " ("
            << BestWidth << ")\n";
    else
      log() << "<none>\n";
  } else {
    log() << "hotswap: wmma_scale16: physical forward-dead proof unavailable "
             "at offset 0x"
          << utohexstr(DI.Offset) << "\n";
  }

  // Liveness describes values entering the original instruction. Its
  // destination can therefore appear dead even though every replacement
  // writes it, and tied/overlapping inputs need the same protection. Reserve
  // every physical VGPR range decoded from the original instruction before
  // considering an in-KD scratch block.
  reserveVgprRange(Alloc, DBase, 16);
  reserveVgprRange(Alloc, ABase, 16);
  reserveVgprRange(Alloc, BBase, 8);
  if (CRange)
    reserveVgprRange(Alloc, CBase, 16);
  reserveVgprRange(Alloc, ScaleALo, 2);
  reserveVgprRange(Alloc, ScaleBLo, 2);

  // Masking one eight-register FP4 A half overwrites exactly four registers.
  // Preserve only those four values, not the four already-retained values.
  // The first save slot doubles as the reversible scale-pair permutation
  // temporary before the first A save and after the final A restore. Low-K
  // runs for both M halves before high-K, so the same four slots serve all four
  // WMMAs. Matrix B stays in place.
  constexpr unsigned MatrixHalfWidth = 8;
  constexpr unsigned SavedARegCount = MatrixHalfWidth / 2;
  constexpr unsigned ScratchCount = SavedARegCount;
  std::array<unsigned, SavedARegCount> SavedARegs;
  for (unsigned &Reg : SavedARegs) {
    std::optional<unsigned> Allocated = allocContiguousDeadOrAboveInBank(
        Alloc, /*Count=*/1, /*Align=*/1, VgprBankSize, ForwardDead.has_value());
    if (!Allocated)
      return failClosed(Ctx, DI,
                        "fewer than four dead/above-KD VGPRs for exact M+K "
                        "split");
    Reg = *Allocated;
  }
  unsigned TmpReg = SavedARegs.front();

  std::string ReplacementAsm;
  raw_string_ostream OS(ReplacementAsm);
  unsigned CurrentMode = *ActiveMode;
  emitScalePairSplitInPlace(OS, ScaleALo, ScaleAHi, TmpReg, CurrentMode);
  emitScalePairSplitInPlace(OS, ScaleBLo, ScaleBHi, TmpReg, CurrentMode);

  int HazardNops = classifyWmmaNops("v_wmma_scale_f32_16x16x128_f8f6f4").A0Nops;
  auto EmitHazardNops = [&] {
    for (int I = 0; I != HazardNops; ++I)
      OS << "v_nop\n";
  };

  for (bool HighK : {false, true}) {
    for (unsigned MHalf = 0; MHalf != 2; ++MHalf) {
      unsigned OriginalAHalf = ABase + MHalf * MatrixHalfWidth;
      unsigned DstHalf = DBase + MHalf * MatrixHalfWidth;
      unsigned ABank = OriginalAHalf / VgprBankSize;
      unsigned BBank = BBase / VgprBankSize;

      unsigned SavedIndex = 0;
      for (unsigned I = 0; I != MatrixHalfWidth; ++I) {
        bool IsLow = ((I / 2) % 2) == 0;
        if (IsLow == !HighK)
          continue;
        unsigned Saved = SavedARegs[SavedIndex++];
        emitVgprCopy(OS, Saved, OriginalAHalf + I, /*W=*/1,
                     Saved / VgprBankSize, CurrentMode);
      }
      assert(SavedIndex == SavedARegCount);
      emitMaskVgprsInPlace(OS, /*KeepLow=*/!HighK, OriginalAHalf,
                           MatrixHalfWidth, /*SubW=*/2, CurrentMode);

      SmallVector<VgprBankRequirement, 4> WmmaMode = {
          {VgprMsbOperand::Dst, DstHalf / VgprBankSize},
          {VgprMsbOperand::Src0, ABank},
          {VgprMsbOperand::Src1, BBank}};
      std::string Src2;
      if (HighK) {
        WmmaMode.push_back({VgprMsbOperand::Src2, DstHalf / VgprBankSize});
        Src2 = encodedVgprRange(DstHalf, MatrixHalfWidth);
      } else if (CRange) {
        unsigned CHalf = CBase + MHalf * MatrixHalfWidth;
        WmmaMode.push_back({VgprMsbOperand::Src2, CHalf / VgprBankSize});
        Src2 = encodedVgprRange(CHalf, MatrixHalfWidth);
      } else {
        Src2 = Printed->Operands[3];
      }
      emitModeForOperands(OS, CurrentMode, WmmaMode);
      emitScale32Half(OS, DstHalf, OriginalAHalf, BBase, Src2,
                      HighK ? ScaleAHi : ScaleALo, HighK ? ScaleBHi : ScaleBLo,
                      HighK ? *HighSuffix : *LowSuffix);

      EmitHazardNops();
      SavedIndex = 0;
      for (unsigned I = 0; I != MatrixHalfWidth; ++I) {
        bool IsLow = ((I / 2) % 2) == 0;
        if (IsLow == !HighK)
          continue;
        unsigned Saved = SavedARegs[SavedIndex++];
        emitVgprCopy(OS, OriginalAHalf + I, Saved, /*W=*/1, ABank, CurrentMode);
      }
      assert(SavedIndex == SavedARegCount);
    }
  }

  emitScalePairRestoreInPlace(OS, ScaleALo, ScaleAHi, TmpReg, CurrentMode);
  emitScalePairRestoreInPlace(OS, ScaleBLo, ScaleBHi, TmpReg, CurrentMode);
  emitModeForOperands(OS, CurrentMode,
                      {{VgprMsbOperand::Src0,
                        getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src0)},
                       {VgprMsbOperand::Src1,
                        getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src1)},
                       {VgprMsbOperand::Src2,
                        getVgprMsbBank(*ActiveMode, VgprMsbOperand::Src2)},
                       {VgprMsbOperand::Dst,
                        getVgprMsbBank(*ActiveMode, VgprMsbOperand::Dst)}});

  SmallVector<uint8_t> Replacement =
      assembleInstructions(ReplacementAsm, Ctx.LS);
  if (Replacement.empty())
    return failClosed(Ctx, DI, "M+K split assembly failed");

  unsigned Extra = Alloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, KernelName, Extra, PatchRequirement::Required) !=
      VgprBumpDecision::Apply)
    return 0;
  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement))
    return failClosed(Ctx, DI, "M+K split trampoline emission failed");

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  Stats.ExtraVgprs = std::max(Stats.ExtraVgprs, Extra);
  if (Extra == 0)
    Stats.ScratchReused += ScratchCount;
  Stats.ScratchAboveKd += Extra;
  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs.resize(Ctx.Config.MaxVgprs);
  for (unsigned Reg : SavedARegs)
    Info.ScratchRegs.set(Reg);
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: wmma_scale16: exact M+K split at offset 0x"
        << utohexstr(DI.Offset) << " (D=v" << DBase << ":" << (DBase + 15)
        << ", A=v" << ABase << ":" << (ABase + 15) << ", B=v" << BBase << ":"
        << (BBase + 7) << ", four saved-A VGPRs, tmp=v" << TmpReg << ", +"
        << Extra << " vgpr, 4 WMMAs, " << Replacement.size() << " bytes)\n";
  return 1;
}

// ---------------------------------------------------------------------------
// patchWmmaScale16 -- dispatch
// ---------------------------------------------------------------------------

static uint32_t applyWmmaScale16PatchesImpl(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_wmma_scale16_f32_16x16x128_f8f6f4")
    return patchWmmaScale16_16x16(Ctx, Idx);
  if (Mnem == "v_wmma_scale16_f32_32x16x128_f4")
    return patchWmmaScale16_32x16(Ctx, Idx);

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
