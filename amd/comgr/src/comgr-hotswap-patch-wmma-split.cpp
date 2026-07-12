//===- comgr-hotswap-patch-wmma-split.cpp - WMMA split patches -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Patch module bound to HotswapPatchVTable::applyWmmaSplitPatches via
/// registerWmmaSplitPatch (see comgr-hotswap-patches.def). Decomposes WMMA
/// variants present on GFX1250 B0 but not on A0 into pairs of narrower WMMAs
/// that exist on both steppings, emitted as trampolines appended to .text:
///
///   - v_wmma_*_16x16x128_{fp8,bf8}_{fp8,bf8} -> two 16x16x64 halves
///     (K dimension split, accumulator threads through)
///   - v_wmma_f32_32x16x128_f4 -> two 16x16x128_f8f6f4 halves
///     (M dimension split, both halves use MATRIX_FMT_FP4 modifiers)
///
/// Modifier and src2-inline-immediate handling is delegated to the LLVM
/// MCInstPrinter via printInst(): the splitter prints the original
/// instruction once, then performs textual surgery on the result to
/// produce each split half. This way the splitter never has to reproduce
/// the printer's per-operand formatting decisions (FP inline constants
/// like 1.0 vs 1, modifier suffix ordering and bracket syntax, etc.) --
/// any input the printer accepts is preserved verbatim modulo the
/// per-half transformations described below. The supported asm surface
/// for these 9 opcodes is documented by upstream LLVM's MC test
/// llvm/test/MC/AMDGPU/gfx1250_asm_wmma_w32.s; the test cases for
/// this patch in test-lit/hotswap-wmma-split*.s exercise each form.
///
/// Per-half transformations:
///   - K-split first half: original operand list with src0/src1 sliced
///     to the lower halves; src2 and modifier suffix preserved verbatim.
///   - K-split second half: src0/src1 sliced to the upper halves; src2
///     replaced with the dst register (the accumulator carry from the
///     first half); modifier suffix has the src2-bit cleared in
///     neg_lo:[X,Y,Z] and neg_hi:[X,Y,Z] (because the operand at the
///     src2 slot is no longer the original src2), and matrix_a_reuse /
///     matrix_b_reuse stripped (they refer to data layout that no
///     longer applies after a split).
///   - M-split halves: dst, src0, src2 (when VGPR) sliced to lower /
///     upper halves; src1 broadcast; modifier suffix preserved on both
///     halves with matrix_a_reuse / matrix_b_reuse stripped; the
///     destination opcode (16x16x128_f8f6f4) requires matrix_a_fmt and
///     matrix_b_fmt operands which the source opcode (32x16x128_f4)
///     does not carry, so the splitter appends them with the literal
///     value MATRIX_FMT_FP4 to coerce the f8f6f4 form to interpret the
///     data as the original f4 layout.
///
/// Operand identification uses a per-SplitKind VOP3PWmmaLayout table
/// that names each MCInst slot (vdst, src0, src1, src2_modifiers, src2,
/// plus any trailing modifier slots present in the profile). AMDGPU's
/// getNamedOperandIdx() and OpName enum live in
/// llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h, which is a
/// backend-private header (not installed in the LLVM dist), so we
/// follow the same mirror-and-document pattern that
/// comgr-hotswap-patch-wmma-hazard.cpp uses for SIInstrFlags. The slot
/// positions below match the VOP3P InsVOP3P dag in
/// llvm/lib/Target/AMDGPU/VOP3PInstructions.td; validated at runtime
/// by checking the MCInst operand count and per-slot operand kinds.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

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

struct SplitRule {
  SplitKind Kind;
  StringRef Replacement;
};

// Sole source of truth for what can be split and what it becomes; the
// dispatcher in applyWmmaSplitPatches selects the emitter from SplitKind
// only. Function-local static so the StringMap is built exactly once per
// process (StringMap is not constexpr-initializable; the per-process build
// cost is tiny -- 9 inserts).
const StringMap<SplitRule> &getSplitTable() {
  static const StringMap<SplitRule> Table = {
      {"v_wmma_f16_16x16x128_fp8_fp8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f16_16x16x64_fp8_fp8"}},
      {"v_wmma_f16_16x16x128_fp8_bf8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f16_16x16x64_fp8_bf8"}},
      {"v_wmma_f16_16x16x128_bf8_fp8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f16_16x16x64_bf8_fp8"}},
      {"v_wmma_f16_16x16x128_bf8_bf8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f16_16x16x64_bf8_bf8"}},
      {"v_wmma_f32_16x16x128_fp8_fp8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f32_16x16x64_fp8_fp8"}},
      {"v_wmma_f32_16x16x128_fp8_bf8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f32_16x16x64_fp8_bf8"}},
      {"v_wmma_f32_16x16x128_bf8_fp8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f32_16x16x64_bf8_fp8"}},
      {"v_wmma_f32_16x16x128_bf8_bf8",
       {SplitKind::Split128to64FP8BF8, "v_wmma_f32_16x16x64_bf8_bf8"}},
      {"v_wmma_f32_32x16x128_f4",
       {SplitKind::Split32x16to16x16F4, "v_wmma_f32_16x16x128_f8f6f4"}},
  };
  return Table;
}

std::optional<SplitRule> lookupSplitRule(StringRef Mnemonic) {
  const StringMap<SplitRule> &Table = getSplitTable();
  StringMap<SplitRule>::const_iterator It = Table.find(Mnemonic);
  if (It == Table.end())
    return std::nullopt;
  return It->second;
}

// -- VOP3P WMMA operand layout ----------------------------------------------
//
// Mirrors the per-opcode MCInst layout produced by the AMDGPU disassembler
// for the splittable WMMA opcodes. The two layouts below cover all 9
// splittable opcodes; runtime validation in extractWmmaOps() catches drift.

struct VOP3PWmmaLayout {
  unsigned NumOperands; // expected MCInst operand count for structural check
  unsigned VDst;
  unsigned Src0;
  unsigned Src1;
  unsigned Src2Mods;
  unsigned Src2;
};

// K=128 fp8/bf8 WMMAs: vdst, src0, src1, src2_modifiers, src2, then two
// trailing imm slots (matrix_a_reuse, matrix_b_reuse per the
// HasMatrixReuse=1 profile).
constexpr VOP3PWmmaLayout LayoutK128Fp8Bf8 = {
    /*NumOperands=*/7, /*VDst=*/0, /*Src0=*/1, /*Src1=*/2,
    /*Src2Mods=*/3, /*Src2=*/4};

// 32x16x128 f4: vdst, src0, src1, src2_modifiers, src2 (5 operands; no
// matrix_*_reuse -- HasMatrixReuse=0 on the F4 profile).
constexpr VOP3PWmmaLayout Layout32x16F4 = {
    /*NumOperands=*/5, /*VDst=*/0, /*Src0=*/1, /*Src1=*/2,
    /*Src2Mods=*/3, /*Src2=*/4};

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
//
// extractWmmaOps captures only the structural information the splitter
// needs for register slicing: dst / src0 / src1 widths and base indices,
// and whether src2 is a register or an immediate. Modifier values and the
// canonical src2 textual form come from the printer (see
// transformPrintedAsm below).

struct WmmaOps {
  std::pair<int, int> Dst{-1, 0};
  std::pair<int, int> Src0{-1, 0};
  std::pair<int, int> Src1{-1, 0};
  std::pair<int, int> Src2{-1, 0}; // valid only when Src2IsImm == false
  bool Src2IsImm = false;
};

std::optional<WmmaOps> extractWmmaOps(const MCInst &Inst,
                                      const MCRegisterInfo &MRI,
                                      SplitKind Kind, StringRef Mnemonic) {
  WmmaOps R;
  const VOP3PWmmaLayout &L = layoutFor(Kind);

  if (Inst.getNumOperands() != L.NumOperands) {
    log() << "hotswap: error: WMMA split: operand count mismatch for "
          << Mnemonic << ": expected " << L.NumOperands << ", got "
          << Inst.getNumOperands() << " (VOP3P layout drift -- update the "
          << "VOP3PWmmaLayout table in comgr-hotswap-patch-wmma-split.cpp)\n";
    return std::nullopt;
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
    return std::nullopt;
  }

  R.Dst = getVgprRange(VDstOp.getReg(), MRI);
  R.Src0 = getVgprRange(Src0Op.getReg(), MRI);
  R.Src1 = getVgprRange(Src1Op.getReg(), MRI);
  if (R.Dst.first < 0 || R.Src0.first < 0 || R.Src1.first < 0)
    return std::nullopt;

  if (Src2Op.isReg()) {
    R.Src2 = getVgprRange(Src2Op.getReg(), MRI);
    if (R.Src2.first < 0)
      return std::nullopt;
  } else if (Src2Op.isImm()) {
    R.Src2IsImm = true;
  } else {
    return std::nullopt;
  }

  return R;
}

// -- Printed-asm parsing and transformation ---------------------------------

struct PrintedAsm {
  StringRef Mnemonic;
  StringRef Operands[4]; // vdst, src0, src1, src2 (printer-canonical form)
  StringRef ModifierSuffix; // includes leading space if non-empty
};

// Parse the printer's output for a VOP3P WMMA instruction:
//   `\t<mnemonic> <op0>, <op1>, <op2>, <op3>[ <modifier> ...]`
// Returns std::nullopt if the structure does not match the expected shape
// (e.g. fewer than 4 comma-separated operands).
std::optional<PrintedAsm> parsePrintedAsm(StringRef S) {
  PrintedAsm R;
  S = S.trim();
  size_t MnemEnd = S.find_first_of(" \t");
  if (MnemEnd == StringRef::npos)
    return std::nullopt;
  R.Mnemonic = S.substr(0, MnemEnd);
  StringRef Rest = S.substr(MnemEnd).ltrim();

  // First three operands end at a comma.
  for (int I = 0; I < 3; ++I) {
    size_t Comma = Rest.find(',');
    if (Comma == StringRef::npos)
      return std::nullopt;
    R.Operands[I] = Rest.substr(0, Comma).trim();
    Rest = Rest.substr(Comma + 1).ltrim();
  }
  // Fourth operand ends at the first whitespace (modifier suffix start) or
  // end-of-string. Modifier syntax never contains spaces inside a single
  // modifier token (e.g. `neg_lo:[0,0,1]` has no space) so this split is
  // unambiguous for the supported asm surface (see file header).
  size_t ModBegin = Rest.find_first_of(" \t");
  if (ModBegin == StringRef::npos) {
    R.Operands[3] = Rest;
    R.ModifierSuffix = StringRef();
  } else {
    R.Operands[3] = Rest.substr(0, ModBegin);
    R.ModifierSuffix = Rest.substr(ModBegin); // includes leading space
  }
  return R;
}

// Tokenize a modifier suffix into individual modifier tokens. Tokens are
// whitespace-separated; the suffix may have a leading space.
SmallVector<StringRef, 8> tokenizeModifiers(StringRef Suffix) {
  SmallVector<StringRef, 8> Out;
  StringRef S = Suffix.ltrim();
  while (!S.empty()) {
    size_t Sp = S.find_first_of(" \t");
    if (Sp == StringRef::npos) {
      Out.push_back(S);
      break;
    }
    Out.push_back(S.substr(0, Sp));
    S = S.substr(Sp + 1).ltrim();
  }
  return Out;
}

// Returns true if `T` is a `<Name>:[X,Y,Z]` packed-modifier token; on success,
// fills in `Bits` with three-character views of X, Y, Z (which may be 0 or 1).
// `Name` is checked piecewise so we never have to materialize `<Name>:[` on
// the heap for every token (this runs once per modifier per split half).
bool parsePackedModifier(StringRef T, StringRef Name,
                         std::array<StringRef, 3> &Bits) {
  if (!T.starts_with(Name) || !T.ends_with("]"))
    return false;
  T = T.drop_front(Name.size());
  if (!T.starts_with(":["))
    return false;
  StringRef Inside = T.drop_front(2).drop_back(1);
  SmallVector<StringRef, 3> Parts;
  Inside.split(Parts, ",");
  if (Parts.size() != 3)
    return false;
  Bits[0] = Parts[0].trim();
  Bits[1] = Parts[1].trim();
  Bits[2] = Parts[2].trim();
  return true;
}

// Build a modifier suffix for a split half. `KSplitSecondHalf` is true for
// the K-split's second half: in that case the operand at the src2 position
// is the dst register (the accumulator carry), so any neg_lo / neg_hi bit
// targeting src2 must be cleared. `StripMatrixReuse` is always true for the
// splitter's output: matrix_a_reuse / matrix_b_reuse refer to data layout
// that no longer applies after a split (the original data lives in a
// different VGPR set in each half), so preserving them would assert a
// guarantee the splitter cannot make.
// Closed set of modifier tokens the splitter knows how to handle on its
// source surface (K=128 fp8/bf8 WMMAs and the 32x16x128_f4 WMMA). Anything
// outside this set means the source mnemonic acquired a modifier the
// splitter has not been audited for -- failing fast (returning nullopt) is
// safer than silently carrying it through both halves, where it could
// double-apply or apply to the wrong half. Update this set in lockstep with
// any new K=128/M=32 source mnemonic the splitter table grows to cover.
bool isKnownSplitterModifier(StringRef T) {
  if (T == "matrix_a_reuse" || T == "matrix_b_reuse")
    return true;
  std::array<StringRef, 3> Bits;
  return parsePackedModifier(T, "neg_lo", Bits) ||
         parsePackedModifier(T, "neg_hi", Bits);
}

std::optional<std::string> transformModifierSuffix(StringRef Suffix,
                                                   bool KSplitSecondHalf) {
  std::string Out;
  for (StringRef T : tokenizeModifiers(Suffix)) {
    if (!isKnownSplitterModifier(T)) {
      log() << "hotswap: error: WMMA split: unsupported modifier token \""
            << T << "\" -- splitter modifier set must be updated\n";
      return std::nullopt;
    }
    if (T == "matrix_a_reuse" || T == "matrix_b_reuse")
      continue;
    std::array<StringRef, 3> Bits;
    if (KSplitSecondHalf &&
        (parsePackedModifier(T, "neg_lo", Bits) ||
         parsePackedModifier(T, "neg_hi", Bits))) {
      // Clear the src2 bit (third element of the [X,Y,Z] tuple). If the
      // remaining bits are all zero, drop the modifier entirely (matches
      // the printer's behavior of omitting an all-zero packed modifier).
      bool X = Bits[0] != "0";
      bool Y = Bits[1] != "0";
      if (!X && !Y)
        continue;
      StringRef Name = T.substr(0, T.find(':'));
      Out += ' ';
      Out += Name.str();
      Out += ":[";
      Out += Bits[0].str();
      Out += ',';
      Out += Bits[1].str();
      Out += ",0]";
      continue;
    }
    Out += ' ';
    Out += T.str();
  }
  return Out;
}

// Format a VGPR range as `v[lo:hi]`.
std::string formatVgprRange(int Base, int Count) {
  assert(Count > 0 && Base >= 0);
  return formatv("v[{0}:{1}]", Base, Base + Count - 1).str();
}

// Return the active VGPR-MSB mode when it is locally unambiguous. The late
// VGPR lowering pass makes the mode persistent, so a trampoline that changes
// it must restore the value that was active at the source instruction. Stop at
// control flow rather than guessing across a predecessor merge; compiler-
// generated straight-line blocks place the relevant s_set_vgpr_msb before the
// first high-register instruction.
std::optional<unsigned> findActiveVgprMsbMode(const PatchContext &Ctx,
                                              size_t Idx) {
  std::optional<ElfView::FunctionTextRange> Function =
      Ctx.Elf.findFunctionTextRangeAtOffset(Ctx.Decoded[Idx].Offset);
  if (!Function)
    return std::nullopt;

  for (size_t I = Idx; I-- > 0;) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Offset < Function->Begin)
      break;
    if (DI.Mnemonic == "s_set_vgpr_msb") {
      if (DI.Inst.getNumOperands() != 1 ||
          !DI.Inst.getOperand(0).isImm())
        return std::nullopt;
      return static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) & 0xff;
    }
    if (DI.Mnemonic == "<unknown>")
      return std::nullopt;

    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    if (Desc.isTerminator() || Desc.isBranch() || Desc.isCall() ||
        Desc.isReturn() ||
        (Ctx.LS.MIA &&
         Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI)))
      return std::nullopt;
  }

  // The gfx1250 ABI and late VGPR-lowering convention require zero mode at a
  // function entry.
  return 0;
}

bool kSplitNeedsVgprMsbTransition(const WmmaOps &R) {
  return R.Src0.first + R.Src0.second / 2 > 255 ||
         R.Src1.first + R.Src1.second / 2 > 255;
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

// K-dimension split: dst and src2 are unchanged on the first half. For the
// second half, src2 = dst (the carry from the first half).
std::vector<std::string> buildSplit128to64Asm(StringRef Replacement,
                                              const PrintedAsm &P,
                                              const WmmaOps &R,
                                              std::optional<unsigned>
                                                  ActiveVgprMsbMode) {
  assert(R.Dst.second > 0 && (R.Src2IsImm || R.Src2.second == R.Dst.second));
  assert(R.Src0.second > 0 && R.Src0.second % 2 == 0);
  assert(R.Src1.second > 0 && R.Src1.second % 2 == 0);

  int AHalf = R.Src0.second / 2;
  int BHalf = R.Src1.second / 2;
  StringRef Dst = P.Operands[0]; // verbatim from printer (e.g. "v[16:23]")
  StringRef Src2Printed = P.Operands[3];
  std::optional<std::string> ModFirst =
      transformModifierSuffix(P.ModifierSuffix, /*KSplitSecondHalf=*/false);
  std::optional<std::string> ModSecond =
      transformModifierSuffix(P.ModifierSuffix, /*KSplitSecondHalf=*/true);
  if (!ModFirst || !ModSecond)
    return {};

  std::vector<std::string> Out;
  Out.reserve(5);
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}", Replacement, Dst,
                        formatVgprRange(R.Src0.first, AHalf),
                        formatVgprRange(R.Src1.first, BHalf), Src2Printed,
                        *ModFirst)
                    .str());

  int Src0HiBase = R.Src0.first + AHalf;
  int Src1HiBase = R.Src1.first + BHalf;
  if (Src0HiBase > 255 || Src1HiBase > 255) {
    if (!ActiveVgprMsbMode || R.Src0.first > 255 || R.Src1.first > 255)
      return {};

    unsigned OldMode = *ActiveVgprMsbMode;
    unsigned NewMode = OldMode;
    auto AdvanceOperandMode = [&](int &Base, unsigned Shift) {
      unsigned OldMsbs = (OldMode >> Shift) & 0x3;
      unsigned PhysicalBase = (OldMsbs << 8) + static_cast<unsigned>(Base);
      unsigned NewMsbs = PhysicalBase >> 8;
      if (NewMsbs > 3)
        return false;
      Base = static_cast<int>(PhysicalBase & 0xff);
      NewMode = (NewMode & ~(0x3u << Shift)) | (NewMsbs << Shift);
      return true;
    };
    if (!AdvanceOperandMode(Src0HiBase, /*src0=*/0) ||
        !AdvanceOperandMode(Src1HiBase, /*src1=*/2))
      return {};

    // The second half threads the destination through the src2 slot. Its
    // addressing mode must therefore match vdst rather than the original
    // src2 operand.
    unsigned DstMsbs = (OldMode >> 6) & 0x3;
    NewMode = (NewMode & ~(0x3u << 4)) | (DstMsbs << 4);

    // s_set_vgpr_msb encodes the new mode in bits [7:0] and the prior mode in
    // bits [15:8]. Restore the incoming mode before the trampoline returns.
    unsigned SetUpperMode = NewMode | (OldMode << 8);
    unsigned RestoreMode = OldMode | (NewMode << 8);
    Out.push_back(formatv("s_set_vgpr_msb {0}", SetUpperMode).str());
    Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}", Replacement, Dst,
                          formatVgprRange(Src0HiBase, AHalf),
                          formatVgprRange(Src1HiBase, BHalf), Dst, *ModSecond)
                      .str());
    Out.push_back(formatv("s_set_vgpr_msb {0}", RestoreMode).str());
    return Out;
  }

  // Second half: src2 = dst (the carry).
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}", Replacement, Dst,
                        formatVgprRange(Src0HiBase, AHalf),
                        formatVgprRange(Src1HiBase, BHalf), Dst, *ModSecond)
                    .str());
  return Out;
}

// M-dimension split: A (src0) is split in half; B (src1) is broadcast; dst /
// src2 are split in half by M. The replacement uses the f8f6f4 WMMA with
// both matrix format modifiers forced to MATRIX_FMT_FP4 so the data layout
// matches the original f4 instruction.
std::vector<std::string> buildSplit32x16Asm(StringRef Replacement,
                                            const PrintedAsm &P,
                                            const WmmaOps &R) {
  assert(R.Dst.second > 0 && R.Dst.second % 2 == 0);
  assert(R.Src2IsImm || R.Src2.second == R.Dst.second);
  assert(R.Src0.second > 0 && R.Src0.second % 2 == 0);
  assert(R.Src1.second > 0);

  int DstHalf = R.Dst.second / 2;
  int AHalf = R.Src0.second / 2;
  StringRef B = P.Operands[2]; // broadcast: same printer-canonical form
  // src2 is preserved on both halves when imm; sliced when VGPR.
  std::string CLo = R.Src2IsImm ? P.Operands[3].str()
                                : formatVgprRange(R.Src2.first, DstHalf);
  std::string CHi = R.Src2IsImm
                        ? P.Operands[3].str()
                        : formatVgprRange(R.Src2.first + DstHalf, DstHalf);
  // Matrix format modifiers are required by the f8f6f4 destination opcode
  // and not present on the f4 source opcode, so the splitter appends them
  // explicitly. Modifier suffix from the source is preserved on both halves
  // (with matrix_a_reuse / matrix_b_reuse stripped, same as K-split).
  constexpr StringLiteral FmtSuffix =
      " matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4";
  std::optional<std::string> Mod =
      transformModifierSuffix(P.ModifierSuffix, /*KSplitSecondHalf=*/false);
  if (!Mod)
    return {};

  std::vector<std::string> Out;
  Out.reserve(2);
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}{6}", Replacement,
                        formatVgprRange(R.Dst.first, DstHalf),
                        formatVgprRange(R.Src0.first, AHalf), B, CLo,
                        FmtSuffix, *Mod)
                    .str());
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}{6}", Replacement,
                        formatVgprRange(R.Dst.first + DstHalf, DstHalf),
                        formatVgprRange(R.Src0.first + AHalf, AHalf), B, CHi,
                        FmtSuffix, *Mod)
                    .str());
  return Out;
}

} // anonymous namespace

// The shared pass ABI uses zero for both "no match" and "failed". A matched
// WMMA split is mandatory on A0, so failure paths also set the context flag
// that runPerInstPass checks before it permits dispatcher fall-through.
static uint32_t applyWmmaSplitPatchesImpl(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];

  std::optional<SplitRule> Match = lookupSplitRule(DI.Mnemonic);
  if (!Match)
    return 0; // Did NOT match -- correct dispatcher fall-through.

  auto FailRequiredPatch = [&]() -> uint32_t {
    Ctx.RequiredPatchFailed = true;
    return 0;
  };

  // Structural sanity check against the opcode side. Every WMMA variant this
  // patch handles has exactly one destination operand at the MCInstrDesc
  // level; a differing def count means the operand layout is not what
  // extractWmmaOps expects, so refuse to emit rather than produce
  // silently-wrong asm.
  const MCInstrDesc &MCID = Ctx.LS.MCII->get(DI.Inst.getOpcode());
  if (MCID.getNumDefs() != 1) {
    log() << "hotswap: error: WMMA split: " << DI.Mnemonic << " has "
          << MCID.getNumDefs() << " defs, expected 1\n";
    return FailRequiredPatch();
  }

  std::optional<WmmaOps> Ops =
      extractWmmaOps(DI.Inst, *Ctx.LS.MRI, Match->Kind, DI.Mnemonic);
  if (!Ops) {
    log() << "hotswap: error: WMMA split: could not extract operands from "
          << DI.Mnemonic << "\n";
    return FailRequiredPatch();
  }

  if (!validateSplitOperands(Match->Kind, *Ops, DI.Mnemonic))
    return FailRequiredPatch();

  // Print the source instruction in canonical asm form. The printer is the
  // authoritative source for src2 inline-immediate formatting (FP inline
  // constants like 1.0 vs integer 1 encode differently) and for the
  // modifier suffix (op_sel / neg_lo / neg_hi / matrix_a_reuse /
  // matrix_b_reuse, in whatever order the printer chose).
  SmallString<256> PrintedBuf;
  raw_svector_ostream PrintOS(PrintedBuf);
  Ctx.LS.MCIP->printInst(&DI.Inst, /*Address=*/0, /*Annot=*/"", *Ctx.LS.STI,
                         PrintOS);
  std::optional<PrintedAsm> P = parsePrintedAsm(StringRef(PrintedBuf));
  if (!P) {
    log() << "hotswap: error: WMMA split: could not parse printed form of "
          << DI.Mnemonic << ": " << StringRef(PrintedBuf).trim() << "\n";
    return FailRequiredPatch();
  }

  std::vector<std::string> AsmLines;
  switch (Match->Kind) {
  case SplitKind::Split128to64FP8BF8:
    AsmLines = buildSplit128to64Asm(
        Match->Replacement, *P, *Ops,
        kSplitNeedsVgprMsbTransition(*Ops)
            ? findActiveVgprMsbMode(Ctx, Idx)
            : std::optional<unsigned>());
    break;
  case SplitKind::Split32x16to16x16F4:
    AsmLines = buildSplit32x16Asm(Match->Replacement, *P, *Ops);
    break;
  }
  if (AsmLines.empty())
    return FailRequiredPatch();

  // Assemble the split sequence and defer trampoline emission to
  // emitToTrampoline, which picks a short s_branch or a set-PC long branch
  // based on the site's distance from the appended pool.
  SmallVector<uint8_t> Replacement =
      assembleSingleInst(joinAsmLines(AsmLines), Ctx.LS);
  if (Replacement.empty()) {
    log() << "hotswap: error: WMMA split: trampoline assembly failed for "
          << DI.Mnemonic << "\n";
    return FailRequiredPatch();
  }
  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement,
                           /*AllowSafeFarReturn=*/true)) {
    log() << "hotswap: error: WMMA split: could not emit trampoline for "
          << DI.Mnemonic << "\n";
    return FailRequiredPatch();
  }

  log() << "hotswap: WMMA split: patched " << DI.Mnemonic << " at offset 0x"
        << utohexstr(DI.Offset) << "\n";
  return 1;
}

void registerWmmaSplitPatch(HotswapPatchVTable &VT) {
  VT.applyWmmaSplitPatches = &applyWmmaSplitPatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
