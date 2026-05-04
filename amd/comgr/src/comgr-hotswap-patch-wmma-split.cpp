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
/// Operand identification uses a per-SplitKind VOP3PWmmaLayout table that
/// names each MCInst slot (vdst, src0, src1, src2_modifiers, src2, plus
/// any trailing modifier slots present in the profile). AMDGPU's
/// getNamedOperandIdx() and OpName enum live in
/// lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h, which is a backend-private
/// header (not installed in the LLVM dist), so we follow the same
/// mirror-and-document pattern that comgr-hotswap-patch-wmma-hazard.cpp
/// uses for SIInstrFlags. The slot positions below match the VOP3P
/// InsVOP3P dag in llvm/lib/Target/AMDGPU/VOP3PInstructions.td;
/// validated at runtime by checking the MCInst operand count and
/// per-slot operand kinds.
///
/// The K=128 fp8/bf8 family and the f4 form do not accept any modifier
/// asm syntax (op_sel, neg_lo, neg_hi, clamp, matrix_*_reuse) per the
/// AMDGPU asm parser, even though the TableGen profile reserves slots
/// for some of them. extractWmmaOps() defensively refuses the rewrite
/// if any trailing modifier slot is non-default, so future opcode
/// evolution that exposes modifier syntax will fail loudly here rather
/// than silently miscompile.
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

// -- VOP3P WMMA operand layout ----------------------------------------------
//
// Mirrors the per-opcode MCInst layout produced by the AMDGPU disassembler
// for the splittable WMMA opcodes. The AMDGPU backend's
// getNamedOperandIdx() and OpName enum (in
// llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h) provide the canonical
// per-opcode operand positions, but that header is backend-private and
// not exposed to comgr -- comgr-hotswap-patch-wmma-hazard.cpp's local
// mirror of SIInstrFlags is the established pattern for this kind of
// cross-layer lookup. The two layouts below cover all 9 splittable
// opcodes; runtime validation in extractWmmaOps() (operand count +
// operand-kind check at each named position) catches drift if upstream
// reorders the dag, in which case the table here needs an update.
//
// Layout source: llvm/lib/Target/AMDGPU/VOP3PInstructions.td InsVOP3P dag
// for IsWMMA=1 with HasModifiers=0 on src0/src1 (the K=128 fp8/bf8 family
// and the f4 form both omit Src0Mods / Src1Mods because their packed
// integer/fp formats do not admit per-source NEG/ABS modifiers).

struct VOP3PWmmaLayout {
  unsigned NumOperands; // expected MCInst operand count for structural check
  unsigned VDst;
  unsigned Src0;
  unsigned Src1;
  unsigned Src2Mods;
  unsigned Src2;
  // -1 (treated as "absent") for opcodes whose profile has HasNeg=0 or
  // HasClamp=0; the splitter then knows to skip emitting these modifiers.
  int NegLo;
  int NegHi;
  int Clamp;
};

// K=128 fp8/bf8 WMMAs: vdst, src0, src1, src2_modifiers, src2, neg_lo,
// neg_hi (7 operands, no clamp on these variants).
constexpr VOP3PWmmaLayout LayoutK128Fp8Bf8 = {
    /*NumOperands=*/7, /*VDst=*/0, /*Src0=*/1, /*Src1=*/2,
    /*Src2Mods=*/3, /*Src2=*/4, /*NegLo=*/5, /*NegHi=*/6, /*Clamp=*/-1};

// 32x16x128 f4: vdst, src0, src1, src2_modifiers, src2 (5 operands; no
// neg/clamp -- VOP3PWMMA_F4_Profile sets HasNeg=0).
constexpr VOP3PWmmaLayout Layout32x16F4 = {
    /*NumOperands=*/5, /*VDst=*/0, /*Src0=*/1, /*Src1=*/2,
    /*Src2Mods=*/3, /*Src2=*/4, /*NegLo=*/-1, /*NegHi=*/-1, /*Clamp=*/-1};

const VOP3PWmmaLayout &layoutFor(SplitKind Kind) {
  switch (Kind) {
  case SplitKind::Split128to64FP8BF8:
    return LayoutK128Fp8Bf8;
  case SplitKind::Split32x16to16x16F4:
    return Layout32x16F4;
  }
  llvm_unreachable("unknown SplitKind");
}

// -- VGPR range extraction --------------------------------------------------
//
// For a tuple register the VGPR base index is the low 10 bits of the
// encoding value (high bits encode HW flags like IS_VGPR = 1 << 10 per
// SIDefines.h's AMDGPU::HWEncoding). The mask 0x3ff is the stable AMDGPU
// HW encoding -- target-internal, named locally because SIDefines.h is
// not exposed to comgr.
//
// The VGPR count is the smallest enclosing MCRegisterClass's bit width
// divided by 32, the same pattern AMDGPUDisassembler::CheckVGPROverflow
// uses. MCRegisterInfo::regunits() is NOT the right primitive: on GFX12+
// each VGPR_32 has two regunits (lo16 + hi16 sub-register slots), so a
// VReg_256 (8 VGPRs, 256 bits) yields 16 regunits, not 8. The
// smallest-class lookup is memoized per MCRegister in a thread_local
// DenseMap, invalidated when the MCRegisterInfo pointer changes between
// calls (each retargetCodeObjectB0A0 constructs a fresh MCRegisterInfo
// via initLLVM, so a static cross-call cache would dangle).

constexpr unsigned VgprRegIdxMask = 0x3ff;

const MCRegisterClass *
findSmallestEnclosingClass(MCRegister Reg, const MCRegisterInfo &MRI) {
  thread_local const MCRegisterInfo *CachedMRI = nullptr;
  thread_local DenseMap<unsigned, const MCRegisterClass *> Cache;

  if (CachedMRI != &MRI) {
    Cache.clear();
    CachedMRI = &MRI;
  }

  DenseMap<unsigned, const MCRegisterClass *>::iterator It =
      Cache.find(Reg.id());
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

// -- Operand extraction -----------------------------------------------------

struct WmmaOps {
  // Register ranges (base, count). count == 0 means "not present".
  std::pair<int, int> Dst{-1, 0};
  std::pair<int, int> Src0{-1, 0};
  std::pair<int, int> Src1{-1, 0};
  std::pair<int, int> Src2{-1, 0};

  // src2 may be an inline immediate. Captured as the raw imm value; only
  // imm 0 (the compiler-folded zero accumulator -- the common shape for
  // `C = wmma(A, B, 0)`-style code) is currently safe to emit verbatim.
  // Non-zero / FP inline constants would need MCInstPrinter-mediated
  // formatting, which has no public single-operand entry point, so we
  // refuse the rewrite on those rather than risk a wrong encoding.
  bool Src2IsImm = false;
  int64_t Src2Imm = 0;

  bool Valid = false;
};

// Extract operands from a decoded WMMA MCInst by per-SplitKind layout.
// Validates structurally (operand count + reg/imm kind at each named
// position) before reading values, so a TableGen reorder upstream is
// caught loudly rather than silently miscompiling.
//
// Modifier-bearing variants are NOT supported: per the AMDGPU asm parser,
// the K=128 fp8/bf8 family and the f4 form do not accept any modifier
// syntax (op_sel, neg_lo, neg_hi, clamp, matrix_*_reuse) at this opcode
// level. The TableGen profile defines slots for some of these (e.g.
// HasMatrixReuse=1 on F32_FP8BF8X128_WMMA_w32) but the asm grammar does
// not expose them, so a well-formed disassembled MCInst always has the
// trailing modifier slots set to 0. As a defensive check, we refuse the
// split if any trailing slot (between Src2Mods and the end) is non-zero,
// which would mean either (a) a manually-constructed MCInst the splitter
// shouldn't trust or (b) upstream having added modifier support that
// the splitter does not yet know how to preserve.
WmmaOps extractWmmaOps(const MCInst &Inst, const MCRegisterInfo &MRI,
                       SplitKind Kind, StringRef Mnemonic) {
  WmmaOps R;
  const VOP3PWmmaLayout &L = layoutFor(Kind);

  if (Inst.getNumOperands() != L.NumOperands) {
    log() << "hotswap: error: WMMA split: operand count mismatch for "
          << Mnemonic << ": expected " << L.NumOperands << ", got "
          << Inst.getNumOperands() << " (VOP3P layout drift -- update the "
          << "VOP3PWmmaLayout table in comgr-hotswap-patch-wmma-split.cpp)\n";
    return R;
  }

  const MCOperand &VDstOp = Inst.getOperand(L.VDst);
  const MCOperand &Src0Op = Inst.getOperand(L.Src0);
  const MCOperand &Src1Op = Inst.getOperand(L.Src1);
  const MCOperand &Src2ModsOp = Inst.getOperand(L.Src2Mods);
  const MCOperand &Src2Op = Inst.getOperand(L.Src2);

  if (!VDstOp.isReg() || !Src0Op.isReg() || !Src1Op.isReg() ||
      !Src2ModsOp.isImm()) {
    log() << "hotswap: error: WMMA split: operand kind mismatch for "
          << Mnemonic << " (VOP3P layout drift -- update the table)\n";
    return R;
  }

  // Defensive modifier check. src2_modifiers and any optional trailing
  // slots (e.g. matrix_a_reuse / matrix_b_reuse) must all be the default
  // 0 for the splitter's correctness contract: the replacement asm we
  // emit carries no modifier suffix, so we must not silently drop a
  // non-default value.
  if (Src2ModsOp.getImm() != 0) {
    log() << "hotswap: error: WMMA split: " << Mnemonic
          << " has non-zero src2_modifiers; refusing to split (modifier "
          << "preservation is not implemented for this opcode family)\n";
    return R;
  }
  for (unsigned I = L.Src2 + 1, E = L.NumOperands; I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isImm() && Op.getImm() != 0) {
      log() << "hotswap: error: WMMA split: " << Mnemonic
            << " has non-zero modifier at operand " << I << "; refusing "
            << "to split (modifier preservation not implemented)\n";
      return R;
    }
  }

  R.Dst = getVgprRange(VDstOp.getReg(), MRI);
  R.Src0 = getVgprRange(Src0Op.getReg(), MRI);
  R.Src1 = getVgprRange(Src1Op.getReg(), MRI);
  if (R.Dst.first < 0 || R.Src0.first < 0 || R.Src1.first < 0)
    return R;

  if (Src2Op.isReg()) {
    R.Src2 = getVgprRange(Src2Op.getReg(), MRI);
    if (R.Src2.first < 0)
      return R;
  } else if (Src2Op.isImm()) {
    R.Src2IsImm = true;
    R.Src2Imm = Src2Op.getImm();
  } else {
    return R;
  }

  R.Valid = true;
  return R;
}

// Format a VGPR range as `v[lo:hi]`.
std::string formatVgprRange(int Base, int Count) {
  assert(Count > 0 && Base >= 0);
  return formatv("v[{0}:{1}]", Base, Base + Count - 1).str();
}

// -- Operand validation -----------------------------------------------------

bool validateSplitOperands(SplitKind Kind, const WmmaOps &R,
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
  } else {
    // Inline-immediate src2 path: only integer 0 (the compiler-folded zero
    // accumulator) is currently safe to emit. Non-zero integer literals
    // and FP inline constants encode through different VOP3P slots than
    // the integer-literal asm form would land in (see the asm parser's
    // VSrc handling), so re-emitting them as `itostr(getImm())` would
    // either fail to encode or change the instruction. Refuse and let
    // the caller leave the original A0-incompatible opcode in place;
    // the runtime will report an error rather than silently miscompile.
    if (R.Src2Imm != 0) {
      LogError("non-zero src2 inline immediate not supported (only zero "
               "accumulator is recognized for inline-imm splits)");
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

// -- Replacement asm builders -----------------------------------------------
//
// No modifier suffix is emitted: extractWmmaOps refuses any input with a
// non-zero modifier slot for these opcode families (see comment there).
// MATRIX_FMT_FP4 in buildSplit32x16Asm is added by the splitter itself --
// not preserved from the source -- because the f4 source opcode does not
// carry matrix_*_fmt operands but the f8f6f4 destination opcode requires
// them to interpret the data correctly.

// K-dimension split: dst and src2 are unchanged on the first half. For the
// second half, src2 = dst (the carry from the first half).
std::vector<std::string> buildSplit128to64Asm(StringRef Replacement,
                                              const WmmaOps &R) {
  assert(R.Dst.second > 0 && (R.Src2IsImm || R.Src2.second == R.Dst.second));
  assert(R.Src0.second > 0 && R.Src0.second % 2 == 0);
  assert(R.Src1.second > 0 && R.Src1.second % 2 == 0);

  int AHalf = R.Src0.second / 2;
  int BHalf = R.Src1.second / 2;
  std::string Dst = formatVgprRange(R.Dst.first, R.Dst.second);
  std::string CFirst = R.Src2IsImm ? itostr(R.Src2Imm)
                                   : formatVgprRange(R.Src2.first,
                                                     R.Src2.second);

  std::vector<std::string> Out;
  Out.reserve(2);
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}", Replacement, Dst,
                        formatVgprRange(R.Src0.first, AHalf),
                        formatVgprRange(R.Src1.first, BHalf), CFirst)
                    .str());
  // Second half: src2 = dst (the carry).
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}", Replacement, Dst,
                        formatVgprRange(R.Src0.first + AHalf, AHalf),
                        formatVgprRange(R.Src1.first + BHalf, BHalf), Dst)
                    .str());
  return Out;
}

// M-dimension split: A (src0) is split in half; B (src1) is broadcast; dst /
// src2 are split in half by M. The replacement uses the f8f6f4 WMMA with
// both matrix format modifiers forced to MATRIX_FMT_FP4 so the data layout
// matches the original f4 instruction.
std::vector<std::string> buildSplit32x16Asm(StringRef Replacement,
                                            const WmmaOps &R) {
  assert(R.Dst.second > 0 && R.Dst.second % 2 == 0);
  assert(R.Src2IsImm || R.Src2.second == R.Dst.second);
  assert(R.Src0.second > 0 && R.Src0.second % 2 == 0);
  assert(R.Src1.second > 0);

  int DstHalf = R.Dst.second / 2;
  int AHalf = R.Src0.second / 2;
  std::string B = formatVgprRange(R.Src1.first, R.Src1.second);
  // When src2 is an inline immediate, both halves use the same value. When
  // it's a VGPR range, each half takes its own slice (lower / upper half
  // of M).
  std::string CLo = R.Src2IsImm ? itostr(R.Src2Imm)
                                : formatVgprRange(R.Src2.first, DstHalf);
  std::string CHi = R.Src2IsImm
                        ? itostr(R.Src2Imm)
                        : formatVgprRange(R.Src2.first + DstHalf, DstHalf);
  constexpr StringLiteral FmtSuffix =
      " matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4";

  std::vector<std::string> Out;
  Out.reserve(2);
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}", Replacement,
                        formatVgprRange(R.Dst.first, DstHalf),
                        formatVgprRange(R.Src0.first, AHalf), B, CLo, FmtSuffix)
                    .str());
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}", Replacement,
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
  // extractWmmaOps expects, so refuse to emit rather than produce
  // silently-wrong asm.
  const MCInstrDesc &MCID = Ctx.LS.MCII->get(DI.Inst.getOpcode());
  if (MCID.getNumDefs() != 1) {
    log() << "hotswap: error: WMMA split: " << DI.Mnemonic << " has "
          << MCID.getNumDefs() << " defs, expected 1\n";
    return 0;
  }

  WmmaOps Ops = extractWmmaOps(DI.Inst, *Ctx.LS.MRI, Match.Kind, DI.Mnemonic);
  if (!Ops.Valid) {
    log() << "hotswap: error: WMMA split: could not extract operands from "
          << DI.Mnemonic << "\n";
    return 0;
  }

  if (!validateSplitOperands(Match.Kind, Ops, DI.Mnemonic))
    return 0;

  std::vector<std::string> AsmLines;
  switch (Match.Kind) {
  case SplitKind::Split128to64FP8BF8:
    AsmLines = buildSplit128to64Asm(Match.Replacement, Ops);
    break;
  case SplitKind::Split32x16to16x16F4:
    AsmLines = buildSplit32x16Asm(Match.Replacement, Ops);
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
