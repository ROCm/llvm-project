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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <mutex>
#include <optional>
#include <string>

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

// -- Direct MCInst construction (no print / re-parse round-trip) ------------
//
// The splitter builds each replacement half by cloning the decoded source
// MCInst and adjusting operands at the MC layer, then encodes via
// MCCodeEmitter. This avoids the per-patch asm-parser spin-up that dominated
// the pass (~89% of wmma_split time; see B0A0_TRANSPILE_HOTSPOTS.md). Cloning
// preserves the source operands verbatim -- including an inline-immediate
// src2 (e.g. 1.0 vs 1) -- so the byte output is identical to the text path.
//
// The disassembler-visible operand layouts (validated at runtime by
// extractWmmaOps' NumOperands check and by the exact-operand lit tests in
// test-lit/hotswap-wmma-split*.s) are:
//   K=128 fp8/bf8 (source) and K=64 (replacement), 7 operands:
//     [0] vdst  [1] src0  [2] src1  [3] src2_modifiers  [4] src2
//     [5] matrix_a_reuse  [6] matrix_b_reuse
//   32x16x128_f4 (source), 5 operands: [0..4] as above, no trailing modifiers.
//   16x16x128_f8f6f4 (replacement), 7 operands: [0..4] as above,
//     [5] matrix_a_fmt  [6] matrix_b_fmt.
// For these opcodes src2_modifiers packs only the src2 neg_lo (bit0) / neg_hi
// (bit1) flags; src0/src1 negation is not part of the splitter's supported
// surface (a source carrying extra modifier operands changes the operand
// count and is rejected by extractWmmaOps before reaching here).

// MATRIX_FMT_FP4 enum value for the f8f6f4 matrix-format operands, verified
// against the gfx1250 MC layer (printed `matrix_a_fmt:MATRIX_FMT_FP4` <-> Imm:4).
constexpr int64_t MatrixFmtFP4 = 4;

// K=128/K=64 and f8f6f4 operand slots (disassembler layout, see above).
enum : unsigned {
  OpVDst = 0,
  OpSrc0 = 1,
  OpSrc1 = 2,
  OpSrc2Mods = 3,
  OpSrc2 = 4,
  OpTrailA = 5, // matrix_a_reuse (K) / matrix_a_fmt (f8f6f4)
  OpTrailB = 6, // matrix_b_reuse (K) / matrix_b_fmt (f8f6f4)
};

// Return the sub-register of \p Super covering \p Count consecutive 32-bit
// lanes starting \p LaneOffset lanes in, or a null MCRegister if none exists.
// Uses only public MCRegisterInfo APIs: it walks the target's SubRegIndex
// table and matches on the resulting register's VGPR width and base encoding,
// so no backend-private sub-register index constants are hardcoded. Because
// every sliced operand is a sub-range of a register already present in the
// decoded source, the result is guaranteed to be a real VGPR tuple.
MCRegister subVgprTuple(MCRegister Super, int LaneOffset, int Count,
                        const MCRegisterInfo &MRI) {
  std::pair<int, int> SR = getVgprRange(Super, MRI);
  if (SR.first < 0)
    return MCRegister();
  if (LaneOffset == 0 && Count == SR.second)
    return Super;
  int WantBase = SR.first + LaneOffset;
  for (unsigned Idx = 1, E = MRI.getNumSubRegIndices(); Idx < E; ++Idx) {
    MCRegister Sub = MRI.getSubReg(Super, Idx);
    if (!Sub)
      continue;
    std::pair<int, int> R = getVgprRange(Sub, MRI);
    if (R.first == WantBase && R.second == Count)
      return Sub;
  }
  return MCRegister();
}

// Encode an MCInst to raw bytes via MCCodeEmitter (mirrors the helper in
// comgr-hotswap-patch-inplace.cpp).
SmallVector<uint8_t> encodeMCInst(const MCInst &Inst, const LLVMState &LS) {
  SmallVector<char, 16> Code;
  SmallVector<MCFixup, 4> Fixups;
  LS.MCE->encodeInstruction(Inst, Code, Fixups, *LS.STI);
  return SmallVector<uint8_t>(Code.begin(), Code.end());
}

// Resolve the MC opcode for a replacement mnemonic and cache it. The opcode
// is a pure function of (target, mnemonic + operand kinds), so we assemble a
// single canonical instance per (target, mnemonic) via the asm parser and
// cache the decoded opcode. This keeps the parser off the per-patch path
// (at most one parse per replacement mnemonic per process) while remaining
// correct-by-construction: the cached opcode is exactly what the assembler
// would have produced.
std::optional<unsigned> cachedReplacementOpcode(StringRef CanonicalAsm,
                                                StringRef Mnemonic,
                                                const LLVMState &LS) {
  static std::mutex Mu;
  static std::map<std::pair<const void *, std::string>, unsigned> Cache;
  std::pair<const void *, std::string> Key{
      static_cast<const void *>(LS.Target), Mnemonic.str()};
  {
    std::lock_guard<std::mutex> Lock(Mu);
    auto It = Cache.find(Key);
    if (It != Cache.end())
      return It->second;
  }
  SmallVector<uint8_t> Bytes = assembleSingleInst(CanonicalAsm, LS);
  if (Bytes.empty())
    return std::nullopt;
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes.data(), Bytes.size(), LS, Decoded) ||
      Decoded.empty())
    return std::nullopt;
  unsigned Opc = Decoded[0].Inst.getOpcode();
  {
    std::lock_guard<std::mutex> Lock(Mu);
    Cache.emplace(std::move(Key), Opc);
  }
  return Opc;
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

// -- Replacement MCInst builders --------------------------------------------

// Build a canonical one-line asm string for a replacement mnemonic using only
// register operands, sufficient to resolve (and cache) the MC opcode. The
// f8f6f4 form requires its matrix-format modifiers to parse, so they are
// included; the opcode does not depend on the src2 operand kind, so the dst
// register is used as a stand-in src2.
std::string canonicalKSplitAsm(StringRef Repl, const WmmaOps &R) {
  int AHalf = R.Src0.second / 2;
  int BHalf = R.Src1.second / 2;
  return formatv("{0} {1}, {2}, {3}, {1}", Repl,
                 formatVgprRange(R.Dst.first, R.Dst.second),
                 formatVgprRange(R.Src0.first, AHalf),
                 formatVgprRange(R.Src1.first, BHalf))
      .str();
}

std::string canonicalMSplitAsm(StringRef Repl, const WmmaOps &R) {
  int DstHalf = R.Dst.second / 2;
  int AHalf = R.Src0.second / 2;
  return formatv("{0} {1}, {2}, {3}, {1} matrix_a_fmt:MATRIX_FMT_FP4 "
                 "matrix_b_fmt:MATRIX_FMT_FP4",
                 Repl, formatVgprRange(R.Dst.first, DstHalf),
                 formatVgprRange(R.Src0.first, AHalf),
                 formatVgprRange(R.Src1.first, R.Src1.second))
      .str();
}

// K-dimension split at the MCInst level. Both halves are clones of the decoded
// source with the opcode retargeted to the K=64 replacement; src0/src1 are
// sliced to their lower/upper halves and matrix_a/b_reuse are stripped. The
// first half preserves src2 and its modifiers verbatim (including an inline
// immediate); the second half sets src2 to the dst accumulator carry and
// clears the src2 neg_lo/neg_hi modifiers (src2_modifiers -> 0), matching the
// text path's behavior.
std::optional<SmallVector<MCInst, 2>>
buildSplit128to64Insts(unsigned ReplOpcode, const MCInst &Src, const WmmaOps &R,
                       const MCRegisterInfo &MRI) {
  int AHalf = R.Src0.second / 2;
  int BHalf = R.Src1.second / 2;
  MCRegister Dst = Src.getOperand(OpVDst).getReg();
  MCRegister Src0 = Src.getOperand(OpSrc0).getReg();
  MCRegister Src1 = Src.getOperand(OpSrc1).getReg();

  MCRegister A0 = subVgprTuple(Src0, 0, AHalf, MRI);
  MCRegister A1 = subVgprTuple(Src0, AHalf, AHalf, MRI);
  MCRegister B0 = subVgprTuple(Src1, 0, BHalf, MRI);
  MCRegister B1 = subVgprTuple(Src1, BHalf, BHalf, MRI);
  if (!A0 || !A1 || !B0 || !B1) {
    log() << "hotswap: error: WMMA split: could not slice K-split VGPR "
             "tuples\n";
    return std::nullopt;
  }

  MCInst H1 = Src;
  H1.setOpcode(ReplOpcode);
  H1.getOperand(OpSrc0) = MCOperand::createReg(A0);
  H1.getOperand(OpSrc1) = MCOperand::createReg(B0);
  H1.getOperand(OpTrailA) = MCOperand::createImm(0); // strip matrix_a_reuse
  H1.getOperand(OpTrailB) = MCOperand::createImm(0); // strip matrix_b_reuse

  MCInst H2 = Src;
  H2.setOpcode(ReplOpcode);
  H2.getOperand(OpSrc0) = MCOperand::createReg(A1);
  H2.getOperand(OpSrc1) = MCOperand::createReg(B1);
  H2.getOperand(OpSrc2Mods) = MCOperand::createImm(0); // clear src2 neg bits
  H2.getOperand(OpSrc2) = MCOperand::createReg(Dst);   // carry from first half
  H2.getOperand(OpTrailA) = MCOperand::createImm(0);
  H2.getOperand(OpTrailB) = MCOperand::createImm(0);

  SmallVector<MCInst, 2> Out;
  Out.push_back(std::move(H1));
  Out.push_back(std::move(H2));
  return Out;
}

// M-dimension split at the MCInst level. A (src0) and dst/src2 are sliced by
// M; B (src1) is broadcast. The 5-operand f4 source becomes a 7-operand
// f8f6f4 instruction with both matrix-format operands forced to
// MATRIX_FMT_FP4. src2_modifiers (neg) are preserved on both halves.
std::optional<SmallVector<MCInst, 2>>
buildSplit32x16Insts(unsigned ReplOpcode, const MCInst &Src, const WmmaOps &R,
                     const MCRegisterInfo &MRI) {
  int DstHalf = R.Dst.second / 2;
  int AHalf = R.Src0.second / 2;
  MCRegister Dst = Src.getOperand(OpVDst).getReg();
  MCRegister Src0 = Src.getOperand(OpSrc0).getReg();
  MCOperand Src1 = Src.getOperand(OpSrc1);     // broadcast, verbatim
  MCOperand Src2Mods = Src.getOperand(OpSrc2Mods);
  MCOperand Src2 = Src.getOperand(OpSrc2);     // reg (sliced) or imm (preserved)

  MCRegister D0 = subVgprTuple(Dst, 0, DstHalf, MRI);
  MCRegister D1 = subVgprTuple(Dst, DstHalf, DstHalf, MRI);
  MCRegister A0 = subVgprTuple(Src0, 0, AHalf, MRI);
  MCRegister A1 = subVgprTuple(Src0, AHalf, AHalf, MRI);
  MCRegister C0, C1;
  if (!R.Src2IsImm) {
    MCRegister Src2Reg = Src2.getReg();
    C0 = subVgprTuple(Src2Reg, 0, DstHalf, MRI);
    C1 = subVgprTuple(Src2Reg, DstHalf, DstHalf, MRI);
  }
  if (!D0 || !D1 || !A0 || !A1 || (!R.Src2IsImm && (!C0 || !C1))) {
    log() << "hotswap: error: WMMA split: could not slice M-split VGPR "
             "tuples\n";
    return std::nullopt;
  }

  auto Build = [&](MCRegister D, MCRegister A, MCRegister C) {
    MCInst I;
    I.setOpcode(ReplOpcode);
    I.addOperand(MCOperand::createReg(D)); // vdst
    I.addOperand(MCOperand::createReg(A)); // src0
    I.addOperand(Src1);                    // src1 (broadcast)
    I.addOperand(Src2Mods);                // src2_modifiers (neg preserved)
    I.addOperand(R.Src2IsImm ? Src2 : MCOperand::createReg(C)); // src2
    I.addOperand(MCOperand::createImm(MatrixFmtFP4)); // matrix_a_fmt
    I.addOperand(MCOperand::createImm(MatrixFmtFP4)); // matrix_b_fmt
    return I;
  };

  SmallVector<MCInst, 2> Out;
  Out.push_back(Build(D0, A0, C0));
  Out.push_back(Build(D1, A1, C1));
  return Out;
}

} // anonymous namespace

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
static uint32_t applyWmmaSplitPatchesImpl(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];

  std::optional<SplitRule> Match = lookupSplitRule(DI.Mnemonic);
  if (!Match)
    return 0; // Did NOT match -- correct dispatcher fall-through.

  // ----- All return-0 paths below are MATCHED-BUT-FAILED -----
  // Until the dispatcher API is refactored to distinguish these cleanly,
  // each of these is a silent miscompile risk for the runtime; the log()
  // line is the only signal the user gets that a recognized opcode was
  // left in .text.

  // Structural sanity check against the opcode side. Every WMMA variant this
  // patch handles has exactly one destination operand at the MCInstrDesc
  // level; a differing def count means the operand layout is not what
  // extractWmmaOps expects, so refuse to emit rather than produce
  // silently-wrong asm.
  const MCInstrDesc &MCID = Ctx.LS.MCII->get(DI.Inst.getOpcode());
  if (MCID.getNumDefs() != 1) {
    log() << "hotswap: error: WMMA split: " << DI.Mnemonic << " has "
          << MCID.getNumDefs() << " defs, expected 1\n";
    return 0; // matched-but-failed
  }

  std::optional<WmmaOps> Ops =
      extractWmmaOps(DI.Inst, *Ctx.LS.MRI, Match->Kind, DI.Mnemonic);
  if (!Ops) {
    log() << "hotswap: error: WMMA split: could not extract operands from "
          << DI.Mnemonic << "\n";
    return 0; // matched-but-failed
  }

  if (!validateSplitOperands(Match->Kind, *Ops, DI.Mnemonic))
    return 0; // matched-but-failed (validateSplitOperands logs the reason)

  // Build the two split-half instructions directly at the MCInst level (clone
  // + operand surgery + MCCodeEmitter) rather than round-tripping through the
  // asm printer and a per-patch MCAsmParser. The replacement MC opcode is
  // resolved once per mnemonic via a cached canonical parse; everything else
  // is pure MCInst manipulation. Cloning preserves the source operands
  // verbatim, so an inline-immediate src2 encodes identically to the text path.
  std::optional<unsigned> ReplOpcode =
      Match->Kind == SplitKind::Split128to64FP8BF8
          ? cachedReplacementOpcode(canonicalKSplitAsm(Match->Replacement, *Ops),
                                     Match->Replacement, Ctx.LS)
          : cachedReplacementOpcode(canonicalMSplitAsm(Match->Replacement, *Ops),
                                     Match->Replacement, Ctx.LS);
  if (!ReplOpcode) {
    log() << "hotswap: error: WMMA split: could not resolve replacement opcode "
          << "for " << Match->Replacement << "\n";
    return 0; // matched-but-failed
  }

  std::optional<SmallVector<MCInst, 2>> Insts;
  switch (Match->Kind) {
  case SplitKind::Split128to64FP8BF8:
    Insts = buildSplit128to64Insts(*ReplOpcode, DI.Inst, *Ops, *Ctx.LS.MRI);
    break;
  case SplitKind::Split32x16to16x16F4:
    Insts = buildSplit32x16Insts(*ReplOpcode, DI.Inst, *Ops, *Ctx.LS.MRI);
    break;
  }
  if (!Insts)
    return 0; // matched-but-failed (builder logged the reason)

  // Encode the split instructions and defer trampoline emission to
  // emitToTrampoline, which picks a short s_branch or an SGPR-backed set-PC
  // gateway based on the site's distance from the appended pool.
  SmallVector<uint8_t> Replacement;
  for (const MCInst &I : *Insts) {
    SmallVector<uint8_t> Bytes = encodeMCInst(I, Ctx.LS);
    if (Bytes.empty()) {
      log() << "hotswap: error: WMMA split: MCCodeEmitter produced no bytes "
            << "for " << DI.Mnemonic << "\n";
      return 0; // matched-but-failed
    }
    Replacement.append(Bytes.begin(), Bytes.end());
  }
  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement)) {
    log() << "hotswap: error: WMMA split: could not emit trampoline for "
          << DI.Mnemonic << "\n";
    return 0; // matched-but-failed
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
