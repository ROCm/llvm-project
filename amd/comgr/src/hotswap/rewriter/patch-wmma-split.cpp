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

#include "internal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>
#include <utility>
#include <vector>

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

// -- VGPR-MSB mode recovery -------------------------------------------------
//
// A K=128 (or M=32) WMMA is split into two narrower WMMAs. When a split
// operand's upper half crosses v255 the upper WMMA must execute under a
// different persistent VGPR-MSB bank, bracketed by s_set_vgpr_msb, and the
// incoming bank restored afterwards. Emitting that bracket correctly requires
// knowing the persistent VGPR-MSB mode live at the WMMA. This is a
// self-contained forward-CFG fixed point over the immutable decoded stream:
// ambiguous, conflicting, or unanalyzable sites stay unknown so a required
// split fails closed rather than guessing a mode.
//
// The four two-bit fields are packed exactly as s_set_vgpr_msb encodes them:
//   bits [1:0]=src0, [3:2]=src1, [5:4]=src2, [7:6]=dst.

struct VgprMsbState {
  int8_t Dst = VgprMsbUnreachable;
  int8_t Src0 = VgprMsbUnreachable;
  int8_t Src1 = VgprMsbUnreachable;
  int8_t Src2 = VgprMsbUnreachable;
};

VgprMsbState vgprMsbStateFromMode(unsigned Mode) {
  Mode &= 0xff;
  return {static_cast<int8_t>((Mode >> 6) & 0x3),
          static_cast<int8_t>(Mode & 0x3),
          static_cast<int8_t>((Mode >> 2) & 0x3),
          static_cast<int8_t>((Mode >> 4) & 0x3)};
}

VgprMsbState unknownVgprMsbState() {
  return {VgprMsbUnknown, VgprMsbUnknown, VgprMsbUnknown, VgprMsbUnknown};
}

[[nodiscard]] int16_t exactVgprMsbMode(VgprMsbState State) {
  if (State.Dst < 0 || State.Src0 < 0 || State.Src1 < 0 || State.Src2 < 0)
    return VgprMsbUnknown;
  return static_cast<int16_t>(State.Src0 | (State.Src1 << 2) |
                              (State.Src2 << 4) | (State.Dst << 6));
}

// True for either gfx1250 s_setreg form (register or immediate) that can write
// the HW_REG_WAVE_MODE bank fields. Matched by cached opcode rather than
// disassembled mnemonic string.
bool isSetregOpcode(const InternalDecodedInst &DI, const LLVMState &LS) {
  unsigned Opcode = DI.Inst.getOpcode();
  return Opcode == LS.SSetregImm32Opcode || Opcode == LS.SSetregB32Opcode;
}

// gfx1250 reaches the persistent VGPR-MSB bank mode through an immediate
// HW_REG_WAVE_MODE (ID_MODE) write. The exact decode is implemented upstream by
// llvm::AMDGPU::convertSetRegImmToVgprMSBs (the MCInst overload at
// llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h:1790), but comgr cannot call
// it: it lives in the target-internal LLVMAMDGPUUtils component, which is not
// reachable when comgr links libLLVM.so (LLVM_LINK_LLVM_DYLIB, used by the
// multi-arch build) -- doing so fails to link with an undefined reference. So
// mirror the setreg-VGPR-MSB-fixup decode here. The constants mirror the
// HW_REG_WAVE_MODE encoding in llvm/lib/Target/AMDGPU/SIDefines.h; keep them in
// sync with that header.
// TODO(https://github.com/ROCm/llvm-project/issues/3516): replace this mirror
// with a direct call to llvm::AMDGPU::convertSetRegImmToVgprMSBs once that
// helper is exported for comgr to link.
constexpr unsigned HwregIdMask = 0x3f; // Hwreg::HwregEncoding ID field [5:0]
constexpr unsigned HwregIdMode = 1;    // AMDGPU::Hwreg::ID_MODE
constexpr unsigned VgprMsbShift = 12;  // countr_zero(Hwreg::DST_VGPR_MSB)
constexpr unsigned VgprMsbFieldMask = 0xff; // VGPR_MSB_MASK >> VgprMsbShift:
                                            // four packed 2-bit fields
constexpr unsigned VgprMsbRotate = 2; // rotr into s_set_vgpr_msb field order

std::optional<unsigned>
decodeSetregImmVgprMsbMode(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (DI.Inst.getOpcode() != LS.SSetregImm32Opcode ||
      DI.Inst.getNumOperands() != 2 || !DI.Inst.getOperand(0).isImm() ||
      !DI.Inst.getOperand(1).isImm())
    return std::nullopt;
  unsigned Simm16 = static_cast<unsigned>(DI.Inst.getOperand(1).getImm());
  if ((Simm16 & HwregIdMask) != HwregIdMode)
    return std::nullopt;
  unsigned Raw =
      (static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) >> VgprMsbShift) &
      VgprMsbFieldMask;
  // Equivalent to llvm::rotr<uint8_t>(Raw, VgprMsbRotate).
  return ((Raw >> VgprMsbRotate) | (Raw << (8 - VgprMsbRotate))) &
         VgprMsbFieldMask;
}

std::optional<unsigned> getSetregHwregId(const InternalDecodedInst &DI,
                                         const LLVMState &LS) {
  if (!isSetregOpcode(DI, LS) || DI.Inst.getNumOperands() == 0 ||
      !DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm())
    return std::nullopt;
  return static_cast<unsigned>(
             DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm()) &
         HwregIdMask;
}

bool instructionDefinesNamedRegister(const InternalDecodedInst &DI,
                                     StringRef Name, const LLVMState &LS) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != NumDefs; ++I) {
    const MCOperand &Op = DI.Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() &&
        StringRef(LS.MRI->getName(Op.getReg())) == Name)
      return true;
  }
  return llvm::any_of(Desc.implicit_defs(), [&](MCPhysReg Reg) {
    return StringRef(LS.MRI->getName(Reg)) == Name;
  });
}

[[nodiscard]] std::optional<unsigned>
getExactVgprMsbModeWritten(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (DI.Inst.getOpcode() == LS.SSetVgprMsbOpcode) {
    if (DI.Inst.getNumOperands() != 1 || !DI.Inst.getOperand(0).isImm())
      return std::nullopt;
    return static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) & 0xff;
  }
  return decodeSetregImmVgprMsbMode(DI, LS);
}

bool isStandardLinkReturn(const InternalDecodedInst &DI, const LLVMState &LS) {
  return DI.Inst.getOpcode() == LS.SSetPcI64Opcode &&
         DI.Inst.getNumOperands() == 1 && DI.Inst.getOperand(0).isReg() &&
         DI.Inst.getOperand(0).getReg() &&
         StringRef(LS.MRI->getName(DI.Inst.getOperand(0).getReg())) ==
             "SGPR30_SGPR31";
}

// Forward transfer function: how an instruction changes the incoming mode.
// Exact MODE writes install a known mode; unknown MODE writes, opaque calls,
// and undecoded slots poison it to unknown.
VgprMsbState transferVgprMsbState(VgprMsbState In,
                                  const InternalDecodedInst &DI,
                                  const LLVMState &LS) {
  if (In.Dst == VgprMsbUnreachable)
    return In;
  if (std::optional<unsigned> Mode = getExactVgprMsbModeWritten(DI, LS))
    return vgprMsbStateFromMode(*Mode);
  if (DI.Inst.getOpcode() == LS.SSetVgprMsbOpcode)
    return unknownVgprMsbState();
  if (isSetregOpcode(DI, LS)) {
    std::optional<unsigned> HwregId = getSetregHwregId(DI, LS);
    if (!HwregId)
      return unknownVgprMsbState();
    if (*HwregId != HwregIdMode)
      return In;
    return unknownVgprMsbState();
  }
  if (DI.Mnemonic == "<unknown>" ||
      instructionDefinesNamedRegister(DI, "MODE", LS) ||
      (LS.MIA && LS.MIA->isCall(DI.Inst)))
    return unknownVgprMsbState();
  return In;
}

int8_t mergeVgprMsbValue(int8_t Old, int8_t Incoming) {
  if (Old == VgprMsbUnreachable)
    return Incoming;
  if (Incoming == VgprMsbUnreachable || Old == Incoming)
    return Old;
  return VgprMsbUnknown;
}

VgprMsbState mergeVgprMsbState(VgprMsbState Old, VgprMsbState Incoming) {
  return {mergeVgprMsbValue(Old.Dst, Incoming.Dst),
          mergeVgprMsbValue(Old.Src0, Incoming.Src0),
          mergeVgprMsbValue(Old.Src1, Incoming.Src1),
          mergeVgprMsbValue(Old.Src2, Incoming.Src2)};
}

// The B0 f4gemm corpus uses one legacy VOP3 encoding that the A0 gfx1250
// decoder splits into two unknown dwords. It cannot write MODE: recognize only
// the exact observed opcode/operand spellings, and only when no known entry
// reaches the decoder-created continuation dword.
bool isUndecodedB0Vop3Pair(const PatchContext &Ctx, size_t HeadIndex) {
  if (HeadIndex + 1 >= Ctx.Decoded.size())
    return false;
  const InternalDecodedInst &Head = Ctx.Decoded[HeadIndex];
  const InternalDecodedInst &Tail = Ctx.Decoded[HeadIndex + 1];
  if (Head.DecodeSucceeded || Head.Size != MinInstSize ||
      Tail.Offset != Head.Offset + MinInstSize || Head.Offset > Ctx.TextSize ||
      2 * MinInstSize > Ctx.TextSize - Head.Offset)
    return false;

  uint32_t Word0 = support::endian::read32le(Ctx.Text + Head.Offset);
  uint32_t Word1 =
      support::endian::read32le(Ctx.Text + Head.Offset + MinInstSize);
  constexpr uint32_t Bit14 = 1u << 14;
  if ((Word0 & ~Bit14) != 0xd0310000u || (Word1 != 0 && Word1 != 0x00100000u))
    return false;
  if (Ctx.DirectControlFlow.Targets.contains(Tail.Offset) ||
      llvm::is_contained(Ctx.DeclaredEntries, Tail.Offset))
    return false;
  return true;
}

// Populate Ctx.VgprMsbModeBefore with a per-instruction packed VGPR-MSB mode
// via a forward CFG fixed point over each analyzable function. Fail-closed:
// only exact, consistent modes are recorded; any conflict, unknown MODE write,
// opaque call, unresolved/indirect branch, or non-start cross-function entry
// leaves the affected sites unknown/unanalyzed so a required split declines.
//
// Pass-ordering invariant: this analysis reads Ctx.Decoded as an immutable
// snapshot (per HOTSWAP_CONVENTIONS) and recovers the mode from the bytes of
// the VGPR-MSB mode instructions (s_set_vgpr_msb and s_setreg writes to
// HW_REG_WAVE_MODE). It must therefore run before, or be re-decoded after, any
// pass that rewrites those specific instructions' bytes. This holds today: the
// in-place pass only touches cluster loads / s_barrier_signal_isfirst and the
// trampoline pass preserves instruction sizes and mode-instruction bytes.
void computeVgprMsbModes(PatchContext &Ctx) {
  const std::vector<InternalDecodedInst> &Decoded = Ctx.Decoded;
  const LLVMState &LS = Ctx.LS;
  ElfView &Elf = Ctx.Elf;
  ArrayRef<uint8_t> Text(Ctx.Text, Ctx.TextSize);
  std::vector<int16_t> &ModeBefore = Ctx.VgprMsbModeBefore;
  ModeBefore.assign(Decoded.size(), VgprMsbUnanalyzed);
  if (!LS.MIA || !LS.MCII || !LS.MRI)
    return;

  // An unresolved call target (collectDirectBranchTargets could not prove its
  // destination) may enter any function at an interior offset, bypassing that
  // function's s_set_vgpr_msb prefix. The incoming mode is then unprovable
  // object-wide, so decline the whole analysis and let each required split fail
  // closed rather than risk seeding a wrong mode.
  if (Ctx.DirectControlFlow.HasUnresolvedTargets ||
      Ctx.DirectControlFlow.HasUnboundedIndirectEntries)
    return;

  // Functions entered at a non-start offset by a cross-function branch or call
  // cannot be seeded at their entry with the ABI mode; skip them (fail closed).
  // Resolve targets over the same control-flow surface the rewrite uses:
  // direct branches/calls plus absolute-immediate and PC-materialized
  // s_swap_pc_i64 / s_set_pc_i64 transfers, so an interior cross-function
  // s_swap_pc_i64 entry is caught instead of being analyzed from the wrong
  // incoming mode.
  auto resolveInteriorEntryTarget =
      [&](const InternalDecodedInst &DI,
          size_t Index) -> std::optional<uint64_t> {
    if (std::optional<uint64_t> Direct =
            evaluateDirectControlFlowTarget(DI, LS))
      return Direct;
    unsigned Opcode = DI.Inst.getOpcode();
    if ((Opcode != LS.SSwapPcI64Opcode && Opcode != LS.SSetPcI64Opcode) ||
        DI.Inst.getNumOperands() == 0)
      return std::nullopt;
    const MCOperand &TargetOp =
        DI.Inst.getOperand(DI.Inst.getNumOperands() - 1);
    if (TargetOp.isImm()) {
      uint64_t Absolute = static_cast<uint64_t>(TargetOp.getImm());
      if (Absolute < Elf.textAddr())
        return std::nullopt;
      return Absolute - Elf.textAddr();
    }
    if (TargetOp.isReg() && TargetOp.getReg()) {
      std::optional<MaterializedPcSequence> Sequence =
          resolveMaterializedPcTarget(Decoded, Index,
                                      MCRegister(TargetOp.getReg()), LS,
                                      Elf.textAddr());
      if (Sequence && Sequence->Target >= Elf.textAddr())
        return Sequence->Target - Elf.textAddr();
    }
    return std::nullopt;
  };

  DenseSet<uint64_t> CrossFunctionInteriorEntries;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    unsigned Opcode = DI.Inst.getOpcode();
    bool IsMaterializablePcTransfer =
        Opcode == LS.SSwapPcI64Opcode || Opcode == LS.SSetPcI64Opcode;
    // s_set_pc_i64 is an indirect transfer the generic MC layer classifies as a
    // return rather than a branch, so the branch/call/return filter would skip
    // it. Always attempt to resolve a materialized/absolute PC transfer target
    // so a cross-function interior jump is caught; a genuine register return
    // has no provable target and resolves to nullopt below.
    if (!IsMaterializablePcTransfer &&
        ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
         LS.MIA->isReturn(DI.Inst)))
      continue;
    std::optional<uint64_t> Target = resolveInteriorEntryTarget(DI, I);
    if (!Target || *Target >= Text.size())
      continue;
    std::optional<ElfView::FunctionTextRange> SourceOwner =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    std::optional<ElfView::FunctionTextRange> TargetOwner =
        Elf.findFunctionTextRangeAtOffset(*Target);
    if (TargetOwner && *Target != TargetOwner->Begin &&
        (!SourceOwner || SourceOwner->Begin != TargetOwner->Begin ||
         SourceOwner->End != TargetOwner->End))
      CrossFunctionInteriorEntries.insert(TargetOwner->Begin);
  }

  DenseSet<std::pair<uint64_t, uint64_t>> SeenRanges;
  std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Elf.functionTextRanges();
  for (size_t RangeIndex = 0; RangeIndex != FunctionRanges.size();
       ++RangeIndex) {
    const ElfView::FunctionTextRange &VirtualRange = FunctionRanges[RangeIndex];
    if (VirtualRange.Begin < Elf.textAddr() ||
        VirtualRange.End < VirtualRange.Begin ||
        VirtualRange.End > Elf.textAddr() + Text.size())
      continue;
    uint64_t Begin = VirtualRange.Begin - Elf.textAddr();
    uint64_t End = VirtualRange.End - Elf.textAddr();
    if (Begin >= End || !SeenRanges.insert({Begin, End}).second ||
        CrossFunctionInteriorEntries.contains(Begin))
      continue;

    // An outer symbol must not donate facts to instructions owned by a nested
    // function symbol; skip overlapping / alias ranges.
    size_t NextRange = RangeIndex + 1;
    while (NextRange != FunctionRanges.size() &&
           FunctionRanges[NextRange].Begin == VirtualRange.Begin)
      ++NextRange;
    if (NextRange != FunctionRanges.size() &&
        FunctionRanges[NextRange].Begin < VirtualRange.End)
      continue;
    std::optional<ElfView::FunctionTextRange> Owner =
        Elf.findFunctionTextRangeAtOffset(Begin);
    if (!Owner || Owner->Begin != Begin || Owner->End != End)
      continue;

    std::vector<InternalDecodedInst>::const_iterator First = llvm::lower_bound(
        Decoded, Begin, [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    std::vector<InternalDecodedInst>::const_iterator After = llvm::lower_bound(
        Decoded, End, [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    if (First == After || First->Offset != Begin)
      continue;

    const size_t GlobalFirst = static_cast<size_t>(First - Decoded.begin());
    const size_t Count = static_cast<size_t>(After - First);
    DenseMap<uint64_t, unsigned> OffsetToLocalIndex;
    OffsetToLocalIndex.reserve(Count);
    BitVector UndecodedVop3Heads(Count);
    BitVector UndecodedVop3Tails(Count);
    bool Valid = true;
    for (unsigned I = 0; I != Count; ++I) {
      OffsetToLocalIndex.try_emplace(First[I].Offset, I);
      if (First[I].Mnemonic != "<unknown>")
        continue;
      if (isUndecodedB0Vop3Pair(Ctx, GlobalFirst + I) && I + 1 < Count) {
        UndecodedVop3Heads.set(I);
        UndecodedVop3Tails.set(I + 1);
        OffsetToLocalIndex.erase(First[I + 1].Offset);
        ++I;
      } else {
        Valid = false;
      }
    }
    if (!Valid)
      continue;

    std::vector<SmallVector<unsigned, 2>> Successors(Count);
    BitVector CallableEntries(Count);
    auto AddTarget = [&](SmallVectorImpl<unsigned> &Out, uint64_t Target) {
      if (Target < Begin || Target >= End)
        return true;
      DenseMap<uint64_t, unsigned>::iterator It =
          OffsetToLocalIndex.find(Target);
      if (It == OffsetToLocalIndex.end())
        return false;
      Out.push_back(It->second);
      return true;
    };
    auto AddFallthrough = [&](SmallVectorImpl<unsigned> &Out, unsigned I) {
      if (I + 1 < Count)
        Out.push_back(I + 1);
    };

    for (unsigned I = 0; I != Count && Valid; ++I) {
      const InternalDecodedInst &DI = First[I];
      SmallVectorImpl<unsigned> &Out = Successors[I];
      if (UndecodedVop3Tails.test(I))
        continue;
      if (UndecodedVop3Heads.test(I)) {
        if (I + 2 < Count)
          Out.push_back(I + 2);
        continue;
      }
      if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
          DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
          LS.MIA->isReturn(DI.Inst) || isStandardLinkReturn(DI, LS))
        continue;
      // A materialized-PC jump (s_get_pc_i64 / s_add_nc_u64 / s_set_pc_i64)
      // with a provable target is not an unanalyzable indirect branch: a
      // target inside this function is a CFG edge; a target outside exits the
      // function (no fallthrough), so the following code is unreachable and its
      // mode is unobservable. An unresolved s_set_pc_i64 stays fail-closed.
      if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
        std::optional<MaterializedPcSequence> Sequence;
        if (DI.Inst.getNumOperands() >= 1) {
          const MCOperand &TargetOp =
              DI.Inst.getOperand(DI.Inst.getNumOperands() - 1);
          if (TargetOp.isReg() && TargetOp.getReg())
            Sequence = resolveMaterializedPcTarget(
                Decoded, GlobalFirst + I, MCRegister(TargetOp.getReg()), LS,
                Elf.textAddr());
        }
        if (!Sequence || Sequence->Target < Elf.textAddr()) {
          Valid = false;
          break;
        }
        Valid &= AddTarget(Out, Sequence->Target - Elf.textAddr());
        continue;
      }
      if (LS.MIA->isCall(DI.Inst)) {
        // A call target inside this function is a fresh ABI (mode-0) entry and
        // must be seeded like the function start. Resolve it over the same
        // control-flow surface the interior-entry pre-pass uses (direct,
        // absolute-immediate, and PC-materialized) rather than direct targets
        // only: a same-function materialized/absolute call whose block is also
        // reached by a differing-mode fallthrough/branch would otherwise drop
        // the mode-0 contribution and let the join converge on a more specific
        // mode than is provable. An unresolved call target could enter this
        // function anywhere, so fail closed rather than seed too few entries.
        std::optional<uint64_t> Target =
            resolveInteriorEntryTarget(DI, GlobalFirst + I);
        if (!Target) {
          Valid = false;
          break;
        }
        if (*Target >= Begin && *Target < End) {
          DenseMap<uint64_t, unsigned>::iterator It =
              OffsetToLocalIndex.find(*Target);
          if (It != OffsetToLocalIndex.end())
            CallableEntries.set(It->second);
        }
        AddFallthrough(Out, I);
        continue;
      }
      if (LS.MIA->isBranch(DI.Inst)) {
        if (LS.MIA->isIndirectBranch(DI.Inst)) {
          Valid = false;
          break;
        }
        std::optional<uint64_t> Target =
            evaluateDirectControlFlowTarget(DI, LS);
        if (!Target) {
          Valid = false;
          break;
        }
        Valid &= AddTarget(Out, *Target);
        if (LS.MIA->isConditionalBranch(DI.Inst))
          AddFallthrough(Out, I);
        else if (!LS.MIA->isUnconditionalBranch(DI.Inst))
          Valid = false;
        continue;
      }
      if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI)) {
        Valid = false;
        break;
      }
      AddFallthrough(Out, I);
    }
    if (!Valid)
      continue;

    // A validated CFG distinguishes dead instructions (unreachable; their mode
    // is semantically unobservable, so treat as the ABI entry value) from
    // sites we could not analyze at all (left VgprMsbUnanalyzed above).
    for (size_t I = 0; I != Count; ++I)
      ModeBefore[GlobalFirst + I] = VgprMsbUnreachable;

    std::vector<VgprMsbState> In(Count);
    SmallVector<unsigned, 64> Worklist;
    // The gfx1250 VGPR-lowering ABI requires all four MSB fields to be zero on
    // function entry. Calls transfer to unknown (this object-level proof does
    // not inspect a callee's return), so callable entries reseed the ABI mode.
    auto SeedAbiEntry = [&](unsigned I) {
      In[I] = mergeVgprMsbState(In[I], vgprMsbStateFromMode(0));
      Worklist.push_back(I);
    };
    SeedAbiEntry(0);
    for (int I = CallableEntries.find_first(); I >= 0;
         I = CallableEntries.find_next(I))
      if (I != 0)
        SeedAbiEntry(static_cast<unsigned>(I));
    // Declared text entries (kernel-descriptor entry points) are real ABI
    // entries even when no intra-object transfer targets them. Seeding any that
    // land inside this function keeps an interior KD-entry block from being
    // classified unreachable (then read as mode 0) when the real entry path
    // sets the mode before reaching a WMMA there.
    for (uint64_t Entry : Ctx.DeclaredEntries) {
      if (Entry < Begin || Entry >= End)
        continue;
      DenseMap<uint64_t, unsigned>::iterator It =
          OffsetToLocalIndex.find(Entry);
      if (It != OffsetToLocalIndex.end() && It->second != 0)
        SeedAbiEntry(It->second);
    }

    for (size_t Next = 0; Next != Worklist.size(); ++Next) {
      unsigned I = Worklist[Next];
      ModeBefore[GlobalFirst + I] = exactVgprMsbMode(In[I]);
      VgprMsbState Out = UndecodedVop3Heads.test(I)
                             ? In[I]
                             : transferVgprMsbState(In[I], First[I], LS);
      for (unsigned Succ : Successors[I]) {
        VgprMsbState Merged = mergeVgprMsbState(In[Succ], Out);
        if (Merged.Dst != In[Succ].Dst || Merged.Src0 != In[Succ].Src0 ||
            Merged.Src1 != In[Succ].Src1 || Merged.Src2 != In[Succ].Src2) {
          In[Succ] = Merged;
          Worklist.push_back(Succ);
        }
      }
    }
  }
}

// -- Mode-aware split emission ----------------------------------------------

// Recover the persistent VGPR-MSB mode at Idx from the CFG fixed point. Equal
// predecessor states (including loop backedges) retain an exact mode;
// conflicting paths, opaque calls, and unknown MODE writes fail closed. A
// mandatory WMMA in a validated-unreachable block uses the ABI entry mode.
[[nodiscard]] std::optional<unsigned>
findActiveVgprMsbMode(const PatchContext &Ctx, size_t Idx) {
  if (Idx >= Ctx.VgprMsbModeBefore.size())
    return std::nullopt;
  if (Ctx.VgprMsbModeBefore[Idx] == VgprMsbUnreachable)
    return 0;
  if (Ctx.VgprMsbModeBefore[Idx] < 0)
    return std::nullopt;
  return static_cast<unsigned>(Ctx.VgprMsbModeBefore[Idx]);
}

unsigned getVgprMsbs(unsigned Mode, VgprMsbOperand Operand) {
  return (Mode >> static_cast<unsigned>(Operand)) & 0x3;
}

void setVgprMsbs(unsigned &Mode, VgprMsbOperand Operand, unsigned Msbs) {
  const unsigned Shift = static_cast<unsigned>(Operand);
  Mode = (Mode & ~(0x3u << Shift)) | (Msbs << Shift);
}

// Rebase an operand's upper-half index into the [0,255] encoding field and
// record the bank it now selects. Returns false when the physical index needs
// a bank > 3 (unrepresentable), so the caller fails closed.
[[nodiscard]] bool advanceVgprMsbMode(int &Base, VgprMsbOperand Operand,
                                      unsigned OldMode, unsigned &NewMode) {
  unsigned OldMsbs = getVgprMsbs(OldMode, Operand);
  unsigned PhysicalBase = (OldMsbs << 8) + static_cast<unsigned>(Base);
  unsigned NewMsbs = PhysicalBase >> 8;
  if (NewMsbs > 3)
    return false;
  Base = static_cast<int>(PhysicalBase & 0xff);
  setVgprMsbs(NewMode, Operand, NewMsbs);
  return true;
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

} // anonymous namespace

void ensureVgprMsbModes(PatchContext &Ctx) {
  if (Ctx.VgprMsbModeBefore.empty())
    computeVgprMsbModes(Ctx);
}

std::optional<unsigned> getActiveVgprMsbMode(PatchContext &Ctx, size_t Idx) {
  ensureVgprMsbModes(Ctx);
  return findActiveVgprMsbMode(Ctx, Idx);
}

std::optional<unsigned> getLocallyEstablishedVgprMsbMode(PatchContext &Ctx,
                                                         size_t Idx) {
  // An unresolved transfer can enter at any instruction in the object, so an
  // apparently adjacent setter is not a dominance proof in that case.
  if (Ctx.DirectControlFlow.HasUnresolvedTargets ||
      Ctx.DirectControlFlow.HasUnboundedIndirectEntries)
    return std::nullopt;

  while (Idx > 0) {
    const InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
    const InternalDecodedInst &Current = Ctx.Decoded[Idx];
    if (Prev.Offset + Prev.Size != Current.Offset)
      return std::nullopt;

    if (Ctx.DirectControlFlow.Targets.contains(Current.Offset))
      return std::nullopt;
    for (uint64_t Entry : Ctx.DeclaredEntries)
      if (Entry == Current.Offset)
        return std::nullopt;

    // The A0 decoder represents the exact B0 legacy-VOP3 pair as two unknown
    // dwords. It cannot change MODE, so a straight-line local dominance scan
    // may step over the complete pair. isUndecodedB0Vop3Pair also rejects a
    // known entry into its continuation dword.
    if (Idx >= 2 && isUndecodedB0Vop3Pair(Ctx, Idx - 2)) {
      Idx -= 2;
      continue;
    }

    if (std::optional<unsigned> Mode = getExactVgprMsbModeWritten(Prev, Ctx.LS))
      return Mode;

    if (Prev.Mnemonic == "<unknown>" ||
        Prev.Inst.getOpcode() == Ctx.LS.SSetVgprMsbOpcode ||
        instructionDefinesNamedRegister(Prev, "MODE", Ctx.LS) ||
        (Ctx.LS.MIA &&
         Ctx.LS.MIA->mayAffectControlFlow(Prev.Inst, *Ctx.LS.MRI)))
      return std::nullopt;
    --Idx;
  }
  return std::nullopt;
}

int16_t transferExactVgprMsbMode(int16_t Incoming,
                                 const InternalDecodedInst &DI,
                                 const LLVMState &LS) {
  VgprMsbState State =
      Incoming >= 0 ? vgprMsbStateFromMode(static_cast<unsigned>(Incoming))
                    : unknownVgprMsbState();
  return exactVgprMsbMode(transferVgprMsbState(State, DI, LS));
}

unsigned getVgprMsbBank(unsigned Mode, VgprMsbOperand Operand) {
  return getVgprMsbs(Mode, Operand);
}

void setVgprMsbBank(unsigned &Mode, VgprMsbOperand Operand, unsigned Bank) {
  setVgprMsbs(Mode, Operand, Bank);
}

// Return-value semantics (current shared dispatcher API in b0a0.cpp):
//   0  = either "this patch did not match the instruction" OR "matched
//        but failed to apply" -- the dispatcher cannot distinguish the
//        two and will fall through to the next patch class. For WMMA
//        split mnemonics no other patch class will match, so a
//        matched-but-failed case results in the rewriter returning
//        SUCCESS at the API level with the original A0-incompatible
//        opcode left in .text. The runtime will then fail to load (or
//        worse, mis-execute) the kernel with no clear error attribution.
//   N>0 = "matched, applied N patches" (this splitter only ever returns
//        1 since it splits one source WMMA into one trampoline).
//
// chinmaydd flagged this on PR #2379 as a cross-cutting concern across
// every patch in the hotswap subsystem: the shared `uint32_t (*)(
// PatchContext&, size_t)` signature in b0a0.cpp's weak-stub dispatcher
// has the same ambiguity for in-place patches (#2222), the WMMA hazard
// patch (#2265), and any future patch. A proper fix is a separate
// follow-up that changes the dispatcher's return type to an enum
// (NoMatch / Patched / Failed) or threads a `bool *Aborted` through
// PatchContext, with the dispatcher checking the failure flag and
// short-circuiting the rewrite with AMD_COMGR_STATUS_ERROR rather than
// silently leaving the original opcode in .text.
//
// For now: every "matched but failed" path below logs an error via
// log() (so the failure is at least visible when AMD_COMGR_EMIT_VERBOSE_LOGS
// is set) and returns 0. The early "did not match" path returns 0
// without logging.
// Fail closed: a recognized A0-incompatible WMMA that cannot be safely split
// must abort the whole rewrite (the dispatcher checks RequiredPatchFailed)
// rather than leave the original B0 opcode in .text and return SUCCESS.
static uint32_t failWmmaSplit(PatchContext &Ctx) {
  Ctx.RequiredPatchFailed = true;
  return 0;
}

static uint32_t applyWmmaSplitPatchesImpl(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];

  std::optional<SplitRule> Match = lookupSplitRule(DI.Mnemonic);
  if (!Match)
    return 0; // Did NOT match -- correct dispatcher fall-through.

  // ----- All failure paths below fail closed via failWmmaSplit -----
  // A recognized WMMA opcode that is invalid on A0 must never be left in
  // .text with the rewrite still reporting SUCCESS.

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
    return failWmmaSplit(Ctx); // validateSplitOperands logs the reason

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

  // Recover the persistent VGPR-MSB mode once per rewrite (lazy; WMMA-only).
  // A split whose upper half crosses v255 needs the incoming mode to bracket
  // the transition and restore it. K-splits always consult the mode (the
  // upper half reuses dst as src2). M-splits consult it only when a half
  // actually crosses v255.
  ensureVgprMsbModes(Ctx);

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
    std::optional<unsigned> Mode = getActiveVgprMsbMode(Ctx, Idx);
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
  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement)) {
    log() << "hotswap: error: WMMA split: could not emit trampoline for "
          << DI.Mnemonic << "\n";
    return failWmmaSplit(Ctx);
  }

  Ctx.RequiredPatchApplied = true;
  log() << "hotswap: WMMA split: patched " << DI.Mnemonic << " at offset 0x"
        << utohexstr(DI.Offset) << "\n";
  return 1;
}

void registerWmmaSplitPatch(HotswapPatchVTable &VT) {
  VT.applyWmmaSplitPatches = &applyWmmaSplitPatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
