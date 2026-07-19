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
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <limits>
#include <optional>

using namespace llvm;

namespace COMGR {
namespace hotswap {

bool hasDirectControlFlowTargetInWindowInterior(
    const std::optional<DenseSet<uint64_t>> &Targets, uint64_t Begin,
    uint64_t End) {
  if (!Targets || Begin >= End)
    return true;
  return llvm::any_of(*Targets, [=](uint64_t Target) {
    return Target > Begin && Target < End;
  });
}

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
    /*Src2Mods=*/3,    /*Src2=*/4};

// 32x16x128 f4: vdst, src0, src1, src2_modifiers, src2 (5 operands; no
// matrix_*_reuse -- HasMatrixReuse=0 on the F4 profile).
constexpr VOP3PWmmaLayout Layout32x16F4 = {
    /*NumOperands=*/5, /*VDst=*/0, /*Src0=*/1, /*Src1=*/2,
    /*Src2Mods=*/3,    /*Src2=*/4};

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

const MCRegisterClass *findSmallestEnclosingClass(MCRegister Reg,
                                                  const MCRegisterInfo &MRI) {
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
                                      const MCRegisterInfo &MRI, SplitKind Kind,
                                      StringRef Mnemonic) {
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
  StringRef Operands[4];    // vdst, src0, src1, src2 (printer-canonical form)
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
      log() << "hotswap: error: WMMA split: unsupported modifier token \"" << T
            << "\" -- splitter modifier set must be updated\n";
      return std::nullopt;
    }
    if (T == "matrix_a_reuse" || T == "matrix_b_reuse")
      continue;
    std::array<StringRef, 3> Bits;
    if (KSplitSecondHalf && (parsePackedModifier(T, "neg_lo", Bits) ||
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

// Recover the persistent VGPR-MSB mode from the immutable whole-function CFG
// fixed point. Equal predecessor states, including loop backedges, retain an
// exact mode; conflicting paths, opaque calls, and unknown MODE writes remain
// unknown and fail closed.
std::optional<unsigned> findActiveVgprMsbMode(const PatchContext &Ctx,
                                              size_t Idx) {
  if (Idx >= Ctx.VgprMsbModeBefore.size())
    return std::nullopt;
  // A mandatory A0-incompatible WMMA must still be rewritten in a block that
  // a fully validated CFG proves unreachable. Its MODE is semantically
  // unobservable, so use the ABI entry value. Unanalyzed functions and
  // reachable paths with conflicting MODE values continue to fail closed.
  if (Ctx.VgprMsbModeBefore[Idx] == VgprMsbUnreachable)
    return 0;
  if (Ctx.VgprMsbModeBefore[Idx] < 0)
    return std::nullopt;
  return static_cast<unsigned>(Ctx.VgprMsbModeBefore[Idx]);
}

enum class VgprMsbOperand : unsigned {
  Src0 = 0,
  Src1 = 2,
  Src2 = 4,
  Dst = 6,
};

unsigned getVgprMsbs(unsigned Mode, VgprMsbOperand Operand) {
  return (Mode >> static_cast<unsigned>(Operand)) & 0x3;
}

void setVgprMsbs(unsigned &Mode, VgprMsbOperand Operand, unsigned Msbs) {
  const unsigned Shift = static_cast<unsigned>(Operand);
  Mode = (Mode & ~(0x3u << Shift)) | (Msbs << Shift);
}

bool advanceVgprMsbMode(int &Base, VgprMsbOperand Operand, unsigned OldMode,
                        unsigned &NewMode) {
  unsigned OldMsbs = getVgprMsbs(OldMode, Operand);
  unsigned PhysicalBase = (OldMsbs << 8) + static_cast<unsigned>(Base);
  unsigned NewMsbs = PhysicalBase >> 8;
  if (NewMsbs > 3)
    return false;
  Base = static_cast<int>(PhysicalBase & 0xff);
  setVgprMsbs(NewMode, Operand, NewMsbs);
  return true;
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
                                              unsigned ActiveVgprMsbMode,
                                              bool &UsesVgprMsbTransition) {
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
  UsesVgprMsbTransition = false;
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}", Replacement, Dst,
                        formatVgprRange(R.Src0.first, AHalf),
                        formatVgprRange(R.Src1.first, BHalf), Src2Printed,
                        *ModFirst)
                    .str());

  int Src0HiBase = R.Src0.first + AHalf;
  int Src1HiBase = R.Src1.first + BHalf;
  unsigned OldMode = ActiveVgprMsbMode;
  unsigned NewMode = OldMode;
  if (!advanceVgprMsbMode(Src0HiBase, VgprMsbOperand::Src0, OldMode, NewMode) ||
      !advanceVgprMsbMode(Src1HiBase, VgprMsbOperand::Src1, OldMode, NewMode))
    return {};

  // The upper half uses dst as src2, so src2 must select the incoming
  // destination bank even when neither source slice crossed v255.
  setVgprMsbs(NewMode, VgprMsbOperand::Src2,
              getVgprMsbs(OldMode, VgprMsbOperand::Dst));

  if (NewMode != OldMode) {
    UsesVgprMsbTransition = true;
    // Immediate bits [15:8] record the previous mode. Restore the exact
    // incoming mode before returning from the split trampoline.
    unsigned SetUpperMode = NewMode | (OldMode << 8);
    Out.push_back(formatv("s_set_vgpr_msb {0}", SetUpperMode).str());
  }

  // Second half: src2 = dst (the carry).
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}", Replacement, Dst,
                        formatVgprRange(Src0HiBase, AHalf),
                        formatVgprRange(Src1HiBase, BHalf), Dst, *ModSecond)
                    .str());
  if (UsesVgprMsbTransition) {
    unsigned RestoreMode = OldMode | (NewMode << 8);
    Out.push_back(formatv("s_set_vgpr_msb {0}", RestoreMode).str());
  }
  return Out;
}

// M-dimension split: A (src0) is split in half; B (src1) is broadcast; dst /
// src2 are split in half by M. The replacement uses the f8f6f4 WMMA with
// both matrix format modifiers forced to MATRIX_FMT_FP4 so the data layout
// matches the original f4 instruction.
std::vector<std::string>
buildSplit32x16Asm(StringRef Replacement, const PrintedAsm &P, const WmmaOps &R,
                   unsigned ActiveVgprMsbMode, bool &UsesVgprMsbTransition) {
  assert(R.Dst.second > 0 && R.Dst.second % 2 == 0);
  assert(R.Src2IsImm || R.Src2.second == R.Dst.second);
  assert(R.Src0.second > 0 && R.Src0.second % 2 == 0);
  assert(R.Src1.second > 0);

  int DstHalf = R.Dst.second / 2;
  int AHalf = R.Src0.second / 2;
  StringRef B = P.Operands[2]; // broadcast: same printer-canonical form
  int DstHiBase = R.Dst.first + DstHalf;
  int Src0HiBase = R.Src0.first + AHalf;
  int Src2HiBase = R.Src2IsImm ? 0 : R.Src2.first + DstHalf;
  unsigned OldMode = ActiveVgprMsbMode;
  unsigned NewMode = OldMode;
  if (!advanceVgprMsbMode(DstHiBase, VgprMsbOperand::Dst, OldMode, NewMode) ||
      !advanceVgprMsbMode(Src0HiBase, VgprMsbOperand::Src0, OldMode, NewMode) ||
      (!R.Src2IsImm &&
       !advanceVgprMsbMode(Src2HiBase, VgprMsbOperand::Src2, OldMode, NewMode)))
    return {};

  // src2 is preserved on both halves when imm; sliced when VGPR.
  std::string CLo = R.Src2IsImm ? P.Operands[3].str()
                                : formatVgprRange(R.Src2.first, DstHalf);
  std::string CHi =
      R.Src2IsImm ? P.Operands[3].str() : formatVgprRange(Src2HiBase, DstHalf);
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
  Out.reserve(4);
  UsesVgprMsbTransition = NewMode != OldMode;
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}{6}", Replacement,
                        formatVgprRange(R.Dst.first, DstHalf),
                        formatVgprRange(R.Src0.first, AHalf), B, CLo, FmtSuffix,
                        *Mod)
                    .str());
  if (UsesVgprMsbTransition) {
    unsigned SetUpperMode = NewMode | (OldMode << 8);
    Out.push_back(formatv("s_set_vgpr_msb {0}", SetUpperMode).str());
  }
  Out.push_back(formatv("{0} {1}, {2}, {3}, {4}{5}{6}", Replacement,
                        formatVgprRange(DstHiBase, DstHalf),
                        formatVgprRange(Src0HiBase, AHalf), B, CHi, FmtSuffix,
                        *Mod)
                    .str());
  if (UsesVgprMsbTransition) {
    unsigned RestoreMode = OldMode | (NewMode << 8);
    Out.push_back(formatv("s_set_vgpr_msb {0}", RestoreMode).str());
  }
  return Out;
}

struct ProtectedWmmaSourceWindow {
  uint64_t Offset = 0;
  uint32_t Size = 0;
  size_t DelayIndex = 0;
  size_t LastMovedIndex = 0;
  size_t SecondTargetIndex = 0;
  unsigned FirstInstId = 0;
  unsigned SecondInstId = 0;
  bool RetainsSecondTarget = false;

  bool demergesCombinedDelay() const { return SecondInstId != 0; }
};

struct CombinedDelayFields {
  unsigned FirstInstId = 0;
  unsigned Skip = 0;
  unsigned SecondInstId = 0;
};

std::optional<CombinedDelayFields>
decodeDelayFields(const InternalDecodedInst &Delay) {
  if (Delay.Mnemonic != "s_delay_alu" || Delay.Inst.getNumOperands() != 1 ||
      !Delay.Inst.getOperand(0).isImm())
    return std::nullopt;

  uint64_t Imm = static_cast<uint64_t>(Delay.Inst.getOperand(0).getImm());
  if ((Imm & ~uint64_t{0x7FF}) != 0)
    return std::nullopt;

  CombinedDelayFields Fields{static_cast<unsigned>(Imm & 0xF),
                             static_cast<unsigned>((Imm >> 4) & 0x7),
                             static_cast<unsigned>((Imm >> 7) & 0xF)};
  if (Fields.FirstInstId >= 12 || Fields.SecondInstId >= 12 ||
      Fields.Skip > 5 || (Fields.SecondInstId == 0 && Fields.Skip != 0))
    return std::nullopt;
  return Fields;
}

std::optional<InternalDecodedInst>
decodeCurrentWindowMember(const PatchContext &Ctx,
                          const InternalDecodedInst &Original) {
  if (Original.Mnemonic == "<unknown>" || Original.Mnemonic == "<replaced>" ||
      Original.Offset > Ctx.TextSize ||
      Original.Size > Ctx.TextSize - Original.Offset)
    return std::nullopt;
  std::vector<InternalDecodedInst> Current;
  if (!decodeTextSection(Ctx.Text + Original.Offset, Original.Size, Ctx.LS,
                         Current) ||
      Current.size() != 1 || Current.front().Offset != 0 ||
      Current.front().Size != Original.Size)
    return std::nullopt;
  Current.front().Offset = Original.Offset;
  return std::move(Current.front());
}

namespace AmdgpuDelayTSFlags {
static constexpr uint64_t TRANS = UINT64_C(1) << 16;
static constexpr uint64_t IsWMMA = UINT64_C(1) << 59;
static constexpr uint64_t IsSWMMAC = UINT64_C(1) << 63;
} // namespace AmdgpuDelayTSFlags

struct DelayPipelineResources {
  bool HasValu = false;
  bool HasTransValu = false;
  bool HasXdl = false;
};

std::optional<DelayPipelineResources>
getDelayPipelineResources(const InternalDecodedInst &DI,
                          const PatchContext &Ctx) {
  if (!Ctx.LS.STI || !Ctx.LS.MCII)
    return std::nullopt;
  const MCSchedModel &Model = Ctx.LS.STI->getSchedModel();
  if (!Model.hasInstrSchedModel())
    return std::nullopt;

  unsigned SchedClass = Ctx.LS.MCII->get(DI.Inst.getOpcode()).getSchedClass();
  for (unsigned Depth = 0; Depth != 8; ++Depth) {
    if (SchedClass >= Model.NumSchedClasses)
      return std::nullopt;
    const MCSchedClassDesc *Desc = Model.getSchedClassDesc(SchedClass);
    if (!Desc->isValid())
      return std::nullopt;
    if (!Desc->isVariant()) {
      DelayPipelineResources Resources;
      for (const MCWriteProcResEntry *
               It = Ctx.LS.STI->getWriteProcResBegin(Desc),
              *End = Ctx.LS.STI->getWriteProcResEnd(Desc);
           It != End; ++It) {
        unsigned ResourceIndex = It->ProcResourceIdx;
        for (unsigned ResourceDepth = 0;
             ResourceIndex != 0 &&
             ResourceDepth != Model.getNumProcResourceKinds();
             ++ResourceDepth) {
          if (ResourceIndex >= Model.getNumProcResourceKinds())
            return std::nullopt;
          const MCProcResourceDesc *Resource =
              Model.getProcResource(ResourceIndex);
          StringRef Name = Resource->Name ? Resource->Name : "";
          Resources.HasValu |= Name == "HWVALU";
          Resources.HasTransValu |= Name == "HWTransVALU";
          Resources.HasXdl |= Name == "HWXDL";
          ResourceIndex = Resource->SuperIdx;
        }
      }
      if (Desc->NumWriteProcResEntries == 0)
        return std::nullopt;
      return Resources;
    }

    unsigned Resolved = Ctx.LS.STI->resolveVariantSchedClass(
        SchedClass, &DI.Inst, Ctx.LS.MCII.get(), Model.getProcessorID());
    if (Resolved == 0 || Resolved == SchedClass)
      return std::nullopt;
    SchedClass = Resolved;
  }
  return std::nullopt;
}

// Match AMDGPUInsertDelayAlu's dependency class without duplicating an opcode
// list. TRANS32 uses the transcendental pipeline without the ordinary VALU
// pipeline, while gfx1250 XDL WMMA uses HWXDL. In particular, F64 DPMACC and
// non-XDL WMMA must not advance the TRANS ordinal. An incomplete scheduling
// model is unknown rather than silently selecting the wrong producer.
std::optional<bool> isDelayTransInstruction(const InternalDecodedInst &DI,
                                            const PatchContext &Ctx) {
  uint64_t Flags = Ctx.LS.MCII->get(DI.Inst.getOpcode()).TSFlags;
  const bool HasTransFlag = (Flags & AmdgpuDelayTSFlags::TRANS) != 0;
  const bool IsWmmaLike =
      (Flags & (AmdgpuDelayTSFlags::IsWMMA | AmdgpuDelayTSFlags::IsSWMMAC)) !=
      0;
  if (!HasTransFlag && !IsWmmaLike)
    return false;

  std::optional<DelayPipelineResources> Resources =
      getDelayPipelineResources(DI, Ctx);
  if (!Resources)
    return std::nullopt;
  if (IsWmmaLike)
    return Resources->HasXdl;
  if (!Resources->HasTransValu)
    return std::nullopt;
  return !Resources->HasValu;
}

bool isPcIndependentDelayWindowMember(const InternalDecodedInst &DI,
                                      const PatchContext &Ctx) {
  if (!Ctx.LS.MIA || DI.Mnemonic == "<unknown>" ||
      DI.Mnemonic == "<replaced>" || DI.Mnemonic == "s_delay_alu" ||
      DI.Mnemonic == "s_clause" || DI.Mnemonic == "s_set_vgpr_msb" ||
      // Tensor DMA instructions are explicitly linked-PC-sensitive on A0.
      DI.Mnemonic == "tensor_load_to_lds" ||
      StringRef(DI.Mnemonic).contains("_pc_"))
    return false;
  const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
  return !Desc.isTerminator() && !Desc.isBranch() && !Desc.isCall() &&
         !Desc.isReturn() &&
         !Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI);
}

bool sourceRangeOverlapsQueuedReplacement(const PatchContext &Ctx,
                                          uint64_t Begin, uint64_t End) {
  for (const Trampoline &T : Ctx.OutTrampolines) {
    std::optional<uint64_t> TEnd = checkedAddUint64(
        T.OriginalOffset, T.OriginalSize, "queued replacement source end");
    if (!TEnd || (Begin < *TEnd && T.OriginalOffset < End))
      return true;
  }
  return false;
}

bool sourceRangeOverlapsTextSymbolExtent(const PatchContext &Ctx,
                                         uint64_t Begin, uint64_t End) {
  if (!Ctx.TextSymbolExtents)
    return true;
  auto It =
      llvm::lower_bound(*Ctx.TextSymbolExtents, Begin,
                        [](const ElfView::TextOffsetRange &Extent,
                           uint64_t Offset) { return Extent.End <= Offset; });
  return It != Ctx.TextSymbolExtents->end() && It->Begin < End;
}

// A delay-protected WMMA cannot be replaced at its linked address alone: the
// surviving s_delay_alu would describe the source branch rather than the
// replacement. Recover the unique owner from the encoded instruction span.
// For a combined delay, move every position-independent member before the
// second target, split the WMMA in that ordered stream, and use the last moved
// dword for the reconstructed second delay. No instruction mnemonic or kernel
// schedule determines the geometry.
std::optional<ProtectedWmmaSourceWindow>
findProtectedWmmaSourceWindow(const PatchContext &Ctx, size_t WmmaIndex) {
  const InternalDecodedInst &Wmma = Ctx.Decoded[WmmaIndex];
  auto Fail =
      [&](StringRef Reason) -> std::optional<ProtectedWmmaSourceWindow> {
    log() << "hotswap: error: WMMA split: protected site at 0x"
          << utohexstr(Wmma.Offset) << " " << Reason << "\n";
    return std::nullopt;
  };

  if (WmmaIndex == 0 || !Ctx.LS.MIA || !Ctx.DirectControlFlowTargets)
    return Fail("has no supported preceding delay window");
  if (Ctx.HasUnknownArbitraryIndirectTarget)
    return Fail("cannot be relocated with an unresolved indirect entry");

  constexpr size_t MaxClauseSpan = 64;
  const size_t FirstPossibleOwner =
      WmmaIndex > MaxClauseSpan ? WmmaIndex - MaxClauseSpan : 0;
  size_t DelayIndex = 0;
  unsigned DelayOwners = 0;
  std::optional<CombinedDelayFields> Fields;
  for (size_t I = FirstPossibleOwner; I != WmmaIndex; ++I) {
    const InternalDecodedInst &Candidate = Ctx.Decoded[I];
    const size_t Distance = WmmaIndex - I;
    if (Candidate.Mnemonic == "s_delay_alu" &&
        getDelayProtectedSpan(Candidate) >= Distance) {
      ++DelayOwners;
      std::optional<InternalDecodedInst> Current =
          decodeCurrentWindowMember(Ctx, Candidate);
      std::optional<CombinedDelayFields> CandidateFields =
          Current ? decodeDelayFields(*Current) : std::nullopt;
      if (!CandidateFields || getDelayProtectedSpan(*Current) < Distance)
        continue;
      DelayIndex = I;
      Fields = *CandidateFields;
      continue;
    }
    if (Candidate.Mnemonic == "s_clause" &&
        Candidate.Inst.getNumOperands() == 1 &&
        Candidate.Inst.getOperand(0).isImm()) {
      const unsigned ClauseSpan =
          (static_cast<unsigned>(Candidate.Inst.getOperand(0).getImm()) & 63u) +
          1;
      if (ClauseSpan >= Distance)
        return Fail("overlaps another delay or hard clause");
    }
  }
  if (DelayOwners != 1 || !Fields)
    return Fail(DelayOwners > 1 ? "has ambiguous delay ownership"
                                : "has no supported preceding delay window");

  const InternalDecodedInst *Delay = &Ctx.Decoded[DelayIndex];
  const unsigned Span = getDelayProtectedSpan(*Delay);
  if (Span == 0 || Span > Ctx.Decoded.size() - DelayIndex - 1)
    return Fail("has an invalid delay target layout");
  const size_t SecondTargetIndex = DelayIndex + Span;
  if (WmmaIndex > SecondTargetIndex)
    return Fail("has an invalid delay target layout");

  const bool DemergeCombinedDelay = Span > 1;
  if (DemergeCombinedDelay &&
      (Fields->FirstInstId == 0 || Fields->SecondInstId == 0 ||
       Span != Fields->Skip + 1))
    return Fail("has an unsupported combined dependency graph");
  const bool RetainsSecondTarget =
      DemergeCombinedDelay && WmmaIndex < SecondTargetIndex;
  const size_t LastMovedIndex =
      RetainsSecondTarget ? SecondTargetIndex - 1 : WmmaIndex;
  if (RetainsSecondTarget && Ctx.Decoded[LastMovedIndex].Size != MinInstSize)
    return Fail("has no dword slot for the reconstructed second delay");

  // collectRelocationProtectedOffsets does not mark the directive itself.
  // Therefore a protected Delay proves that an earlier clause or another
  // delay overlaps this two-instruction window.
  if (Ctx.RelocationProtectedOffsets.contains(Delay->Offset))
    return Fail("overlaps another delay or hard clause");

  std::optional<ElfView::FunctionTextRange> DelayFunction =
      Ctx.Elf.findFunctionTextRangeAtOffset(Delay->Offset);
  std::optional<ElfView::FunctionTextRange> WmmaFunction =
      Ctx.Elf.findFunctionTextRangeAtOffset(Wmma.Offset);
  if (!DelayFunction || !WmmaFunction ||
      DelayFunction->Begin != WmmaFunction->Begin ||
      DelayFunction->End != WmmaFunction->End)
    return Fail("does not share a proven function with its delay");
  if (Ctx.IndirectControlFlowFunctions.contains(WmmaFunction->Begin))
    return Fail("is in a function with an ambiguous indirect entry");

  std::optional<uint64_t> End =
      RetainsSecondTarget
          ? std::optional<uint64_t>(Ctx.Decoded[LastMovedIndex].Offset)
          : checkedAddUint64(Wmma.Offset, Wmma.Size,
                             "delay-protected WMMA window end");
  if (!End || *End <= Delay->Offset ||
      *End - Delay->Offset > std::numeric_limits<uint32_t>::max())
    return Fail("has an invalid source-window extent");
  std::optional<uint64_t> ClaimEnd = checkedAddUint64(
      Ctx.Decoded[LastMovedIndex].Offset, Ctx.Decoded[LastMovedIndex].Size,
      "delay-protected WMMA claimed-window end");
  const uint64_t EntryCheckEnd =
      RetainsSecondTarget ? Ctx.Decoded[SecondTargetIndex].Offset : *End;
  if (!ClaimEnd ||
      sourceRangeOverlapsQueuedReplacement(Ctx, Delay->Offset, *ClaimEnd))
    return Fail("overlaps an existing replacement source");
  if (sourceRangeOverlapsTextSymbolExtent(Ctx, Delay->Offset, *ClaimEnd))
    return Fail("overlaps a sized non-callable text symbol");
  if (hasDirectControlFlowTargetInWindowInterior(Ctx.DirectControlFlowTargets,
                                                 Delay->Offset, EntryCheckEnd))
    return Fail("has a direct entry into the source-window interior "
                "(including symbol entries)");
  if (Delay->Offset > Ctx.TextSize || EntryCheckEnd > Ctx.TextSize)
    return Fail("has a source window outside .text");

  uint64_t ExpectedOffset = Delay->Offset;
  for (size_t I = DelayIndex; I <= SecondTargetIndex; ++I) {
    const InternalDecodedInst &Original = Ctx.Decoded[I];
    std::optional<InternalDecodedInst> Current =
        decodeCurrentWindowMember(Ctx, Original);
    std::optional<ElfView::FunctionTextRange> Function =
        Ctx.Elf.findFunctionTextRangeAtOffset(Original.Offset);
    if (!Current || !Function || Function->Begin != DelayFunction->Begin ||
        Function->End != DelayFunction->End ||
        Original.Offset != ExpectedOffset)
      return Fail("has a non-contiguous current instruction window");
    ExpectedOffset += Original.Size;

    if (I == WmmaIndex && Current->Inst.getOpcode() != Wmma.Inst.getOpcode())
      return Fail("has a modified WMMA source instruction");

    if (I <= LastMovedIndex &&
        Ctx.ClaimedReplacementOffsets.contains(Original.Offset))
      return Fail("overlaps an existing atomic replacement window");
    if (I != DelayIndex && I != WmmaIndex && I <= LastMovedIndex &&
        !isPcIndependentDelayWindowMember(*Current, Ctx))
      return Fail("has a non-relocatable instruction in its delay window");
    if (I != DelayIndex && (Current->Mnemonic == "s_delay_alu" ||
                            Current->Mnemonic == "s_clause" ||
                            Current->Mnemonic == "s_set_vgpr_msb"))
      return Fail("has an unsupported nested dependency graph");
    // Every moved position is claimed after this atomic relocation. A later
    // per-instruction patch there would be silently skipped by the outer
    // dispatcher, so reject before mutating bytes. The retained second target
    // is deliberately excluded: it remains at its linked address and can
    // compose with the reconstructed delay (tensor masking relies on this).
    if (I != DelayIndex && I != WmmaIndex && I <= LastMovedIndex &&
        requiresIndependentInstructionRewrite(Ctx, I)) {
      log() << "hotswap: error: WMMA split: delay-window member at 0x"
            << utohexstr(Original.Offset)
            << " requires a separate HotSwap patch\n";
      return Fail("would suppress another required HotSwap patch");
    }
  }

  return ProtectedWmmaSourceWindow{
      Delay->Offset,
      static_cast<uint32_t>(*End - Delay->Offset),
      DelayIndex,
      LastMovedIndex,
      SecondTargetIndex,
      DemergeCombinedDelay ? Fields->FirstInstId : 0,
      DemergeCombinedDelay ? Fields->SecondInstId : 0,
      RetainsSecondTarget};
}

std::optional<unsigned> remapSecondDelayDependency(
    const PatchContext &Ctx, const ProtectedWmmaSourceWindow &Window,
    size_t WmmaIndex, ArrayRef<uint8_t> WmmaReplacement) {
  unsigned Dependency = Window.SecondInstId;
  if (!Window.RetainsSecondTarget || Dependency < 5 || Dependency > 7)
    return Dependency;

  std::optional<InternalDecodedInst> Source =
      decodeCurrentWindowMember(Ctx, Ctx.Decoded[WmmaIndex]);
  std::optional<bool> SourceIsTrans =
      Source ? isDelayTransInstruction(*Source, Ctx) : std::nullopt;
  if (!SourceIsTrans || !*SourceIsTrans)
    return std::nullopt;

  std::vector<InternalDecodedInst> Split;
  if (!decodeTextSection(WmmaReplacement.data(), WmmaReplacement.size(), Ctx.LS,
                         Split))
    return std::nullopt;
  unsigned ReplacementTrans = 0;
  for (const InternalDecodedInst &DI : Split) {
    std::optional<bool> IsTrans = isDelayTransInstruction(DI, Ctx);
    if (!IsTrans)
      return std::nullopt;
    ReplacementTrans += *IsTrans;
  }
  if (ReplacementTrans == 0)
    return std::nullopt;

  // Instid counts prior operations in its dependency class. Instructions
  // after the source WMMA retain their ordinal. The original WMMA maps to the
  // last TRANS in its replacement and therefore also retains its ordinal.
  // Only a producer older than the source shifts by the net extra TRANS count.
  unsigned LaterTrans = 0;
  for (size_t I = WmmaIndex + 1; I <= Window.LastMovedIndex; ++I) {
    std::optional<InternalDecodedInst> Current =
        decodeCurrentWindowMember(Ctx, Ctx.Decoded[I]);
    if (!Current)
      return std::nullopt;
    std::optional<bool> IsTrans = isDelayTransInstruction(*Current, Ctx);
    if (!IsTrans)
      return std::nullopt;
    LaterTrans += *IsTrans;
  }

  unsigned Ordinal = Dependency - 4;
  if (Ordinal <= LaterTrans + 1)
    return Dependency;
  const unsigned ExtraTrans = ReplacementTrans - 1;
  if (ExtraTrans > 3 || Ordinal > 3 - ExtraTrans)
    return std::nullopt;
  return 4 + Ordinal + ExtraTrans;
}

} // anonymous namespace

bool isWmmaSplitPatchCandidate(StringRef Mnemonic) {
  return lookupSplitRule(Mnemonic).has_value();
}

// The dispatcher uses zero for both "no match" and "failed". Once a split
// rule matches, set RequiredPatchFailed on every failure so the original
// A0-incompatible WMMA can never be returned as a successful rewrite.
static uint32_t failWmmaSplit(PatchContext &Ctx) {
  Ctx.RequiredPatchFailed = true;
  return 0;
}

static uint32_t applyWmmaSplitPatchesImpl(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];

  std::optional<SplitRule> Match = lookupSplitRule(DI.Mnemonic);
  if (!Match)
    return 0; // Did NOT match -- correct dispatcher fall-through.

  // Snapshot the original relocation constraint before VGPR-MSB handling adds
  // the split site to the set to protect its generated mode transitions.
  const bool WasRelocationProtected =
      Ctx.RelocationProtectedOffsets.contains(DI.Offset);
  std::optional<ProtectedWmmaSourceWindow> ProtectedWindow;
  if (WasRelocationProtected) {
    ProtectedWindow = findProtectedWmmaSourceWindow(Ctx, Idx);
    if (!ProtectedWindow)
      return failWmmaSplit(Ctx);
  }

  // Structural sanity check against the opcode side. Every WMMA variant this
  // patch handles has exactly one destination operand at the MCInstrDesc
  // level; a differing def count means the operand layout is not what
  // extractWmmaOps expects, so refuse to emit rather than produce
  // silently-wrong asm.
  const MCInstrDesc &MCID = Ctx.LS.MCII->get(DI.Inst.getOpcode());
  if (MCID.getNumDefs() != 1) {
    log() << "hotswap: error: WMMA split: " << DI.Mnemonic << " has "
          << MCID.getNumDefs() << " defs, expected 1\n";
    return failWmmaSplit(Ctx);
  }

  std::optional<WmmaOps> Ops =
      extractWmmaOps(DI.Inst, *Ctx.LS.MRI, Match->Kind, DI.Mnemonic);
  if (!Ops) {
    log() << "hotswap: error: WMMA split: could not extract operands from "
          << DI.Mnemonic << "\n";
    return failWmmaSplit(Ctx);
  }

  if (!validateSplitOperands(Match->Kind, *Ops, DI.Mnemonic))
    return failWmmaSplit(Ctx);

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
    return failWmmaSplit(Ctx);
  }

  bool UsesVgprMsbTransition = false;
  bool NeedsKnownVgprMsbMode = Match->Kind == SplitKind::Split128to64FP8BF8;
  if (Match->Kind == SplitKind::Split32x16to16x16F4) {
    NeedsKnownVgprMsbMode =
        Ops->Dst.first + Ops->Dst.second / 2 > 255 ||
        Ops->Src0.first + Ops->Src0.second / 2 > 255 ||
        (!Ops->Src2IsImm && Ops->Src2.first + Ops->Dst.second / 2 > 255);
  }

  unsigned ActiveVgprMsbMode = 0;
  if (NeedsKnownVgprMsbMode) {
    std::optional<unsigned> Mode = findActiveVgprMsbMode(Ctx, Idx);
    if (!Mode) {
      log() << "hotswap: error: WMMA split: cannot determine VGPR-MSB mode "
               "for "
            << DI.Mnemonic << " at offset 0x" << utohexstr(DI.Offset) << "\n";
      return failWmmaSplit(Ctx);
    }
    ActiveVgprMsbMode = *Mode;
  }

  std::vector<std::string> AsmLines;
  switch (Match->Kind) {
  case SplitKind::Split128to64FP8BF8:
    AsmLines = buildSplit128to64Asm(Match->Replacement, *P, *Ops,
                                    ActiveVgprMsbMode, UsesVgprMsbTransition);
    break;
  case SplitKind::Split32x16to16x16F4:
    AsmLines = buildSplit32x16Asm(Match->Replacement, *P, *Ops,
                                  ActiveVgprMsbMode, UsesVgprMsbTransition);
    break;
  }
  if (AsmLines.empty()) {
    log() << "hotswap: error: WMMA split: could not build replacement for "
          << DI.Mnemonic << "\n";
    return failWmmaSplit(Ctx);
  }
  if (UsesVgprMsbTransition)
    protectNonClauseRelocationOffset(Ctx, DI.Offset);

  // Assemble the split sequence and defer trampoline emission to
  // emitToTrampoline, which picks a short s_branch or an SGPR-backed set-PC
  // gateway based on the site's distance from the appended pool.
  SmallVector<uint8_t> Replacement =
      assembleInstructions(joinAsmLines(AsmLines), Ctx.LS);
  if (Replacement.empty()) {
    log() << "hotswap: error: WMMA split: trampoline assembly failed for "
          << DI.Mnemonic << "\n";
    return failWmmaSplit(Ctx);
  }

  uint64_t SourceOffset = DI.Offset;
  uint32_t SourceSize = DI.Size;
  SmallVector<uint8_t> DeferredDelayBytes;
  if (ProtectedWindow) {
    const InternalDecodedInst &Delay = Ctx.Decoded[ProtectedWindow->DelayIndex];
    SmallVector<uint8_t> WithDelay;
    if (ProtectedWindow->demergesCombinedDelay()) {
      std::optional<unsigned> SecondInstId =
          remapSecondDelayDependency(Ctx, *ProtectedWindow, Idx, Replacement);
      if (!SecondInstId) {
        log() << "hotswap: error: WMMA split: combined s_delay_alu at 0x"
              << utohexstr(Delay.Offset)
              << " has an unrepresentable TRANS dependency after split\n";
        return failWmmaSplit(Ctx);
      }
      SmallVector<uint8_t> FirstDelayBytes = assembleSingleInst(
          "s_delay_alu " + std::to_string(ProtectedWindow->FirstInstId),
          Ctx.LS);
      DeferredDelayBytes = assembleSingleInst(
          "s_delay_alu " + std::to_string(*SecondInstId), Ctx.LS);
      if (FirstDelayBytes.size() != MinInstSize ||
          DeferredDelayBytes.size() != MinInstSize) {
        log() << "hotswap: error: WMMA split: could not demerge combined "
                 "s_delay_alu at 0x"
              << utohexstr(Delay.Offset) << "\n";
        return failWmmaSplit(Ctx);
      }

      WithDelay.append(FirstDelayBytes.begin(), FirstDelayBytes.end());
      for (size_t I = ProtectedWindow->DelayIndex + 1;
           I <= ProtectedWindow->LastMovedIndex; ++I) {
        if (!ProtectedWindow->RetainsSecondTarget &&
            I == ProtectedWindow->SecondTargetIndex)
          WithDelay.append(DeferredDelayBytes.begin(),
                           DeferredDelayBytes.end());
        if (I == Idx) {
          WithDelay.append(Replacement.begin(), Replacement.end());
          continue;
        }
        const InternalDecodedInst &Member = Ctx.Decoded[I];
        WithDelay.append(Ctx.Text + Member.Offset,
                         Ctx.Text + Member.Offset + Member.Size);
      }
    } else {
      WithDelay.append(Ctx.Text + Delay.Offset,
                       Ctx.Text + Delay.Offset + Delay.Size);
      WithDelay.append(Replacement.begin(), Replacement.end());
    }
    Replacement = std::move(WithDelay);
    SourceOffset = ProtectedWindow->Offset;
    SourceSize = ProtectedWindow->Size;
  }

  if (ProtectedWindow && ProtectedWindow->RetainsSecondTarget &&
      !canEmitShortTrampoline(Ctx, SourceOffset, SourceSize,
                              Replacement.size())) {
    log() << "hotswap: error: WMMA split: combined-delay demerge at 0x"
          << utohexstr(SourceOffset) << " requires a short trampoline\n";
    return failWmmaSplit(Ctx);
  }

  if (!emitToTrampoline(Ctx, SourceOffset, SourceSize, Replacement)) {
    log() << "hotswap: error: WMMA split: could not emit trampoline for "
          << DI.Mnemonic << "\n";
    return failWmmaSplit(Ctx);
  }

  if (ProtectedWindow && ProtectedWindow->RetainsSecondTarget) {
    const InternalDecodedInst &LastMoved =
        Ctx.Decoded[ProtectedWindow->LastMovedIndex];
    std::memcpy(Ctx.Text + LastMoved.Offset, DeferredDelayBytes.data(),
                MinInstSize);
  }
  if (ProtectedWindow) {
    for (size_t I = ProtectedWindow->DelayIndex;
         I <= ProtectedWindow->LastMovedIndex; ++I) {
      const uint64_t Offset = Ctx.Decoded[I].Offset;
      Ctx.ClaimedReplacementOffsets.insert(Offset);
      protectNonClauseRelocationOffset(Ctx, Offset);
    }
  }

  Ctx.RequiredPatchApplied = true;
  log() << "hotswap: WMMA split: patched " << DI.Mnemonic << " at offset 0x"
        << utohexstr(DI.Offset);
  if (ProtectedWindow && ProtectedWindow->demergesCombinedDelay())
    log() << " by demerging combined delay in source window at 0x"
          << utohexstr(SourceOffset);
  else if (ProtectedWindow)
    log() << " with preceding delay in source window at 0x"
          << utohexstr(SourceOffset);
  log() << "\n";
  return 1;
}

void registerWmmaSplitPatch(HotswapPatchVTable &VT) {
  VT.applyWmmaSplitPatches = &applyWmmaSplitPatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
