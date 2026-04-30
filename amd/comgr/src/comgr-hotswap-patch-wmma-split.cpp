//===- comgr-hotswap-patch-wmma-split.cpp - WMMA split patches -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applyWmmaSplitPatches. Decomposes WMMA
/// variants present on GFX1250 B0 but not on A0 into pairs of narrower WMMAs
/// that exist on both steppings, emitted as trampolines appended to .text:
///
///   - v_wmma_*_16x16x128_{fp8,bf8}_{fp8,bf8} -> two 16x16x64 halves
///     (K dimension split, accumulator threads through)
///   - v_wmma_f32_32x16x128_f4 -> two 16x16x128_f8f6f4 halves
///     (M dimension split, both halves use MATRIX_FMT_FP4 modifiers)
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

// MSVC does not support weak symbols; LLVM_ATTRIBUTE_WEAK expands to nothing,
// so the stub in comgr-hotswap-b0a0.cpp becomes a regular definition and
// this file would produce a duplicate-symbol link error (LNK2005). Guard
// the strong override until a proper registration mechanism replaces the
// weak-symbol pattern on Windows (tracked in #2294 / #2285).
#if !defined(_MSC_VER)

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

// -- Split family table ------------------------------------------------------
//
// The set of splittable WMMA variants is small (9 opcodes) and closed: there
// is no parametric family we need to match against. Exact mnemonic match is
// the simplest form that cannot false-match SWMMAC (v_swmmac_*) instructions,
// which share textual substrings with WMMA but carry a different operand
// layout.

enum class SplitKind {
  // 16x16x128 {fp8|bf8}_{fp8|bf8} -> two 16x16x64 WMMAs of the same variant.
  // K dimension (src0 / src1) is split in half; dst is unchanged; src2 = dst
  // for the second half so the accumulator threads through.
  Split128to64FP8BF8,
  // 32x16x128_f4 -> two 16x16x128_f8f6f4 WMMAs, each with both matrix formats
  // forced to MATRIX_FMT_FP4 to match the original data layout. M dimension
  // (dst / src2) is split in half and A (src0) is split in half; B (src1) is
  // shared across both halves (broadcast across M).
  Split32x16to16x16F4,
};

struct SplitRow {
  StringLiteral Mnemonic;
  SplitKind Kind;
  StringLiteral Replacement;
};

// Sole source of truth for what can be split and what it becomes; the
// dispatcher in applyWmmaSplitPatches selects the emitter from SplitKind only.
constexpr SplitRow SplitTable[] = {
    {"v_wmma_f16_16x16x128_fp8_fp8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f16_16x16x64_fp8_fp8"},
    {"v_wmma_f16_16x16x128_fp8_bf8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f16_16x16x64_fp8_bf8"},
    {"v_wmma_f16_16x16x128_bf8_fp8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f16_16x16x64_bf8_fp8"},
    {"v_wmma_f16_16x16x128_bf8_bf8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f16_16x16x64_bf8_bf8"},
    {"v_wmma_f32_16x16x128_fp8_fp8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f32_16x16x64_fp8_fp8"},
    {"v_wmma_f32_16x16x128_fp8_bf8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f32_16x16x64_fp8_bf8"},
    {"v_wmma_f32_16x16x128_bf8_fp8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f32_16x16x64_bf8_fp8"},
    {"v_wmma_f32_16x16x128_bf8_bf8", SplitKind::Split128to64FP8BF8,
     "v_wmma_f32_16x16x64_bf8_bf8"},
    {"v_wmma_f32_32x16x128_f4", SplitKind::Split32x16to16x16F4,
     "v_wmma_f32_16x16x128_f8f6f4"},
};

struct SplitMatch {
  SplitKind Kind;
  StringRef Replacement;
  bool Matched = false;
};

SplitMatch lookupSplitRule(StringRef Mnemonic) {
  for (const SplitRow &Row : SplitTable)
    if (Mnemonic == Row.Mnemonic)
      return {Row.Kind, Row.Replacement, true};
  return {};
}

// -- VGPR range extraction --------------------------------------------------
//
// Each WMMA reg operand is a VGPR tuple (e.g. v[16:23]); the splitter needs
// the base index and count to slice the range.
//   - Base index: the low 10 bits of MRI.getEncodingValue(Reg). The high
//     bits encode hardware flags (e.g. IS_VGPR = 1 << 10 on GFX). The mask
//     matches AMDGPU::HWEncoding::REG_IDX_MASK from the AMDGPU target
//     backend (SIDefines.h), which is target-internal and not exposed to
//     comgr; the value 0x3ff is a stable part of the AMDGPU HW encoding.
//   - Count: bit width of the register's smallest enclosing class divided
//     by 32 (same pattern as AMDGPUDisassembler::CheckVGPROverflow).
//     Using regunits() directly is wrong on GFX12+ where each VGPR_32 has
//     two regunits (lo16 + hi16 sub-register units).

constexpr unsigned VgprRegIdxMask = 0x3ff;

// Cache the smallest-enclosing MCRegisterClass per MCRegister. Without it,
// every operand lookup walks all ~100 AMDGPU register classes; with it, a
// kernel that repeatedly uses the same VGPR ranges (the common shape for
// tile-multiply loops) hits the cache after the first WMMA. The cache is
// thread_local because MCRegisterClass pointers are owned by the
// MCRegisterInfo passed in, which lives for the duration of one rewrite
// call -- using a static cache across calls would leave dangling pointers
// when the next rewrite constructs a fresh MCRegisterInfo. Invalidation is
// keyed on the MRI pointer: a different MRI from the previous call clears
// the cache before any lookup. Single-threaded access by construction
// (thread_local), so no mutex is needed.
const MCRegisterClass *
findSmallestEnclosingClass(MCRegister Reg, const MCRegisterInfo &MRI) {
  thread_local const MCRegisterInfo *CachedMRI = nullptr;
  thread_local DenseMap<unsigned, const MCRegisterClass *> Cache;

  if (CachedMRI != &MRI) {
    Cache.clear();
    CachedMRI = &MRI;
  }

  auto It = Cache.find(Reg.id());
  if (It != Cache.end())
    return It->second;

  const MCRegisterClass *Smallest = nullptr;
  for (unsigned I = 0, E = MRI.getNumRegClasses(); I < E; ++I) {
    const MCRegisterClass &RC = MRI.getRegClass(I);
    if (RC.contains(Reg) &&
        (!Smallest || RC.getSizeInBits() < Smallest->getSizeInBits()))
      Smallest = &RC;
  }
  Cache[Reg.id()] = Smallest;
  return Smallest;
}

std::pair<int, int> getVgprRange(MCRegister Reg, const MCRegisterInfo &MRI) {
  if (!Reg)
    return {-1, 0};
  const MCRegisterClass *RC = findSmallestEnclosingClass(Reg, MRI);
  if (!RC || RC->getSizeInBits() < 32)
    return {-1, 0};
  int Base = static_cast<int>(MRI.getEncodingValue(Reg) & VgprRegIdxMask);
  int Count = static_cast<int>(RC->getSizeInBits() / 32);
  return {Base, Count};
}

struct WmmaRegOperands {
  std::pair<int, int> Dst{-1, 0};
  std::pair<int, int> Src0{-1, 0};
  std::pair<int, int> Src1{-1, 0};
  std::pair<int, int> Src2{-1, 0};
  // For VOP3P WMMAs, $src2 (the accumulator input) accepts inline constants
  // as well as VGPRs. Clang folds a statically-zero accumulator into an
  // immediate, which is the common shape for `C = wmma(A, B, 0)`-style code.
  bool Src2IsImm = false;
  int64_t Src2Imm = 0;
  bool Valid = false;
};

// VOP3P WMMAs have modifier immediates interleaved between the register
// operands (see VOP3PInstructions.td's `InsVOP3P`). The only stable way to
// find vdst / src0 / src1 / src2 from the MCInst is to walk operands and
// pick the first four registers in order. This is equivalent to the TableGen
// contract for non-SWMMAC WMMAs (vdst, src0, src1, src2).
WmmaRegOperands extractWmmaRegOperands(const MCInst &Inst,
                                       const MCRegisterInfo &MRI) {
  WmmaRegOperands R;
  std::pair<int, int> *Targets[] = {&R.Dst, &R.Src0, &R.Src1, &R.Src2};
  unsigned Found = 0;
  unsigned LastRegIdx = 0;
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E && Found < 4; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (!Op.isReg())
      continue;
    std::pair<int, int> Rng = getVgprRange(Op.getReg(), MRI);
    if (Rng.first < 0)
      return R;
    *Targets[Found++] = Rng;
    LastRegIdx = I;
  }
  if (Found == 4) {
    R.Valid = true;
    return R;
  }
  // Found vdst, src0, src1 but not a register $src2. For VOP3P WMMA the
  // disassembler-built MCInst lays out [vdst, src0, src1, src2, ...trailing
  // modifier imms]. When clang folds a zero-initialized accumulator, $src2 is
  // the inline immediate at index (LastRegIdx + 1) instead of a VGPR.
  if (Found == 3) {
    unsigned Src2Idx = LastRegIdx + 1;
    if (Src2Idx < Inst.getNumOperands()) {
      const MCOperand &Op = Inst.getOperand(Src2Idx);
      if (Op.isImm()) {
        R.Src2IsImm = true;
        R.Src2Imm = Op.getImm();
        R.Valid = true;
      }
    }
  }
  return R;
}

std::string formatVgprRange(int Base, int Count) {
  assert(Count > 0 && Base >= 0);
  return formatv("v[{0}:{1}]", Base, Base + Count - 1).str();
}

// Validate operand shapes before emitting replacement asm. The Build*Asm
// helpers assume specific invariants (dst and src2 share width, halved
// dimensions are even). Those hold for well-formed compiler output, but
// validation guards against handwritten kernels, corrupted ELFs, or future
// operand layouts producing nonsense that the trampoline assembler would
// then reject.
bool validateSplitOperands(SplitKind Kind, const WmmaRegOperands &R,
                           StringRef Mnemonic) {
  auto LogError = [&](StringRef Reason) {
    log() << "hotswap: error: WMMA split: invalid operands for " << Mnemonic
          << ": " << Reason << "\n";
  };
  if (R.Dst.second <= 0 || R.Src0.second <= 0 || R.Src1.second <= 0) {
    LogError("non-positive VGPR range width");
    return false;
  }
  if (!R.Src2IsImm) {
    if (R.Src2.second <= 0) {
      LogError("non-positive VGPR range width");
      return false;
    }
    if (R.Dst.second != R.Src2.second) {
      LogError("dst and src2 VGPR widths differ");
      return false;
    }
  }
  switch (Kind) {
  case SplitKind::Split128to64FP8BF8:
    if (R.Src0.second % 2 != 0 || R.Src1.second % 2 != 0) {
      LogError("src0/src1 VGPR widths must be even to split K in half");
      return false;
    }
    return true;
  case SplitKind::Split32x16to16x16F4:
    if (R.Dst.second % 2 != 0) {
      LogError("dst VGPR width must be even to split M in half");
      return false;
    }
    if (R.Src0.second % 2 != 0) {
      LogError("src0 VGPR width must be even to split A in half");
      return false;
    }
    return true;
  }
  return false;
}

// Split K in half; dst and src2 are unchanged. For the second half, src2 =
// dst so the accumulator threads through from the first half.
std::vector<std::string> buildSplit128to64Asm(StringRef Replacement,
                                              const WmmaRegOperands &R) {
  assert(R.Dst.second > 0 && (R.Src2IsImm || R.Src2.second == R.Dst.second));
  assert(R.Src0.second > 0 && R.Src0.second % 2 == 0);
  assert(R.Src1.second > 0 && R.Src1.second % 2 == 0);

  int AHalf = R.Src0.second / 2;
  int BHalf = R.Src1.second / 2;
  std::string Dst = formatVgprRange(R.Dst.first, R.Dst.second);
  std::string C = R.Src2IsImm ? itostr(R.Src2Imm)
                              : formatVgprRange(R.Src2.first, R.Src2.second);

  std::vector<std::string> Out;
  Out.reserve(2);
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}", Replacement, Dst,
                        formatVgprRange(R.Src0.first, AHalf),
                        formatVgprRange(R.Src1.first, BHalf), C)
                    .str());
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}", Replacement, Dst,
                        formatVgprRange(R.Src0.first + AHalf, AHalf),
                        formatVgprRange(R.Src1.first + BHalf, BHalf), Dst)
                    .str());
  return Out;
}

// Split M in half. A (src0) is split in half (A's VGPRs carry the FP4 data
// for all 32 M rows; each half needs the A rows it accumulates over). B
// (src1) is broadcast: each half reads the same N x K data. dst / src2 are
// split in half by M. The replacement uses the f8f6f4 WMMA with both matrix
// format modifiers forced to MATRIX_FMT_FP4 so the data layout matches the
// original f4 instruction.
std::vector<std::string> buildSplit32x16Asm(StringRef Replacement,
                                            const WmmaRegOperands &R) {
  assert(R.Dst.second > 0 && R.Dst.second % 2 == 0);
  assert(R.Src2IsImm || R.Src2.second == R.Dst.second);
  assert(R.Src0.second > 0 && R.Src0.second % 2 == 0);
  assert(R.Src1.second > 0);

  int DstHalf = R.Dst.second / 2;
  int AHalf = R.Src0.second / 2;
  std::string B = formatVgprRange(R.Src1.first, R.Src1.second);
  // When src2 is an inline immediate (compiler-folded zero accumulator), both
  // halves use the same immediate. When it's a VGPR range, each half takes
  // its own slice (lower / upper half of M).
  std::string CLo = R.Src2IsImm ? itostr(R.Src2Imm)
                                : formatVgprRange(R.Src2.first, DstHalf);
  std::string CHi = R.Src2IsImm
                        ? itostr(R.Src2Imm)
                        : formatVgprRange(R.Src2.first + DstHalf, DstHalf);
  constexpr StringLiteral FmtSuffix =
      "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4";

  std::vector<std::string> Out;
  Out.reserve(2);
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4} {5}", Replacement,
                        formatVgprRange(R.Dst.first, DstHalf),
                        formatVgprRange(R.Src0.first, AHalf), B, CLo, FmtSuffix)
                    .str());
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4} {5}", Replacement,
                        formatVgprRange(R.Dst.first + DstHalf, DstHalf),
                        formatVgprRange(R.Src0.first + AHalf, AHalf), B, CHi,
                        FmtSuffix)
                    .str());
  return Out;
}

} // anonymous namespace

uint32_t applyWmmaSplitPatches(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];

  SplitMatch Match = lookupSplitRule(DI.Mnemonic);
  if (!Match.Matched)
    return 0;

  // Structural sanity check against the opcode side. Every WMMA variant this
  // patch handles has exactly one destination operand at the MCInstrDesc
  // level; a differing def count means the operand layout is not what
  // extractWmmaRegOperands expects, so refuse to emit rather than produce
  // silently-wrong asm.
  const MCInstrDesc &MCID = Ctx.LS.MCII->get(DI.Inst.getOpcode());
  if (MCID.getNumDefs() != 1) {
    log() << "hotswap: error: WMMA split: " << DI.Mnemonic << " has "
          << MCID.getNumDefs() << " defs, expected 1\n";
    return 0;
  }

  WmmaRegOperands Regs = extractWmmaRegOperands(DI.Inst, *Ctx.LS.MRI);
  if (!Regs.Valid) {
    log() << "hotswap: error: WMMA split: could not extract 4 VGPR operands "
          << "from " << DI.Mnemonic << "\n";
    return 0;
  }

  if (!validateSplitOperands(Match.Kind, Regs, DI.Mnemonic))
    return 0;

  std::vector<std::string> AsmLines;
  switch (Match.Kind) {
  case SplitKind::Split128to64FP8BF8:
    AsmLines = buildSplit128to64Asm(Match.Replacement, Regs);
    break;
  case SplitKind::Split32x16to16x16F4:
    AsmLines = buildSplit32x16Asm(Match.Replacement, Regs);
    break;
  }
  if (AsmLines.empty())
    return 0;

  // Compute the trampoline's eventual .text offset so buildTrampoline can
  // emit relative jumps. Same accumulation pattern as emitToTrampoline in
  // b0a0.cpp.
  uint64_t TrampTextOffset = Ctx.TextSize;
  for (const Trampoline &T : Ctx.OutTrampolines)
    TrampTextOffset += T.Bytes.size();

  Trampoline T = buildTrampoline(AsmLines, DI.Offset, DI.Size, TrampTextOffset,
                                 Ctx.LS);
  if (T.Bytes.empty()) {
    log() << "hotswap: error: WMMA split: trampoline assembly failed for "
          << DI.Mnemonic << "\n";
    return 0;
  }
  Ctx.OutTrampolines.emplace_back(std::move(T));

  log() << "hotswap: WMMA split: patched " << DI.Mnemonic << " at offset 0x"
        << utohexstr(DI.Offset) << "\n";
  return 1;
}

} // namespace hotswap
} // namespace COMGR

#endif // !defined(_MSC_VER)
