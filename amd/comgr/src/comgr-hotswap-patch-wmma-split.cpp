//===- comgr-hotswap-patch-wmma-split.cpp - WMMA split patch -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Per-instruction decomposition of unsupported WMMA variants:
///   - Split 16x16x128 FP8/BF8 WMMA into two 16x16x64 halves
///   - Split 32x16x128_f4 WMMA into two 16x16x128_f8f6f4 WMMAs
///
/// This patch targets GFX1250 B0-to-A0 silicon stepping compatibility.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

namespace {

// ── WMMA split families ─────────────────────────────────────────────────────
//
// Splittable WMMA variants fall into two families, enumerated exhaustively
// below.  The full set is small (9 opcodes) and closed: there is no
// parametric family we need to match against today.  Exact whole-string
// matching on the printed mnemonic is used because it is the simplest form
// that cannot false-match SWMMAC (v_swmmac_*) instructions, which share
// textual substrings with WMMA but carry a different operand layout.

enum class SplitKind {
  // 16x16x128 {fp8|bf8}_{fp8|bf8} → two 16x16x64 WMMAs of the same variant.
  // K dimension (src0 / src1) is split in half; dst is unchanged; C = dst for
  // the second half so the accumulator threads through.
  Split128to64_FP8BF8,

  // 32x16x128_f4 → two 16x16x128_f8f6f4 WMMAs, each with both matrix formats
  // forced to MATRIX_FMT_FP4 to match the original data layout.  M dimension
  // (dst / src2) is split in half and A (src0) is split in half; B (src1) is
  // shared across both halves (broadcast across M).
  Split32x16_to_16x16_F4,
};

struct WmmaSplitMatch {
  SplitKind kind;
  llvm::StringRef replacement; // canonical mnemonic for the replacement WMMA
  bool matched = false;
};

struct SplitRow {
  llvm::StringLiteral mnemonic;
  SplitKind kind;
  llvm::StringLiteral replacement;
};

// Exhaustive table of splittable WMMA mnemonics.  Keep this table the sole
// source of truth for what can be split and what it becomes; the dispatcher
// in ApplyWmmaSplitPatches selects the emitter from SplitKind only.
static constexpr SplitRow kSplitTable[] = {
    {"v_wmma_f16_16x16x128_fp8_fp8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f16_16x16x64_fp8_fp8"},
    {"v_wmma_f16_16x16x128_fp8_bf8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f16_16x16x64_fp8_bf8"},
    {"v_wmma_f16_16x16x128_bf8_fp8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f16_16x16x64_bf8_fp8"},
    {"v_wmma_f16_16x16x128_bf8_bf8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f16_16x16x64_bf8_bf8"},
    {"v_wmma_f32_16x16x128_fp8_fp8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f32_16x16x64_fp8_fp8"},
    {"v_wmma_f32_16x16x128_fp8_bf8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f32_16x16x64_fp8_bf8"},
    {"v_wmma_f32_16x16x128_bf8_fp8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f32_16x16x64_bf8_fp8"},
    {"v_wmma_f32_16x16x128_bf8_bf8", SplitKind::Split128to64_FP8BF8,
     "v_wmma_f32_16x16x64_bf8_bf8"},
    {"v_wmma_f32_32x16x128_f4", SplitKind::Split32x16_to_16x16_F4,
     "v_wmma_f32_16x16x128_f8f6f4"},
};

static WmmaSplitMatch LookupSplitRule(llvm::StringRef mnemonic) {
  for (const auto &row : kSplitTable) {
    if (mnemonic == row.mnemonic)
      return {row.kind, row.replacement, true};
  }
  return {};
}

// ── Helper: walk MCInst register operands ───────────────────────────────────
//
// VOP3P WMMAs have modifier immediates interleaved between the register
// operands (see VOP3PInstructions.td's `InsVOP3P`).  The only stable way to
// find vdst / src0 / src1 / src2 from the MCInst is to walk operands and pick
// the first four that are registers, in order.  This is equivalent to the
// TableGen contract for non-SWMMAC WMMAs (vdst, src0, src1, src2).

struct WmmaRegOperands {
  std::pair<int, int> dst{-1, 0};
  std::pair<int, int> src0{-1, 0};
  std::pair<int, int> src1{-1, 0};
  std::pair<int, int> src2{-1, 0};
  bool valid = false;
};

static WmmaRegOperands ExtractWmmaRegOperands(const llvm::MCInst &inst,
                                              const llvm::MCRegisterInfo &MRI) {
  WmmaRegOperands r;
  std::pair<int, int> *targets[] = {&r.dst, &r.src0, &r.src1, &r.src2};
  unsigned found = 0;
  for (unsigned i = 0, e = inst.getNumOperands(); i < e && found < 4; ++i) {
    const auto &op = inst.getOperand(i);
    if (!op.isReg())
      continue;
    auto rng = GetVgprRange(op.getReg(), MRI);
    if (rng.first < 0)
      return r;
    *targets[found++] = rng;
  }
  if (found == 4)
    r.valid = true;
  return r;
}

// ── Helper: format a VGPR range as `v[lo:hi]` ───────────────────────────────

static std::string FormatVgprRange(int base, int count) {
  assert(count > 0 && base >= 0);
  return llvm::formatv("v[{0}:{1}]", base, base + count - 1).str();
}

// ── Helper: validate operand shapes before emitting replacement asm ─────────
//
// The Build*Asm helpers assume specific invariants on the WmmaRegOperands
// (e.g. dst and src2 span the same number of VGPRs, halved dimensions are
// even).  Those invariants hold for well-formed compiler output, but if the
// instruction came from a handwritten kernel, a corrupted ELF, or a future
// operand layout, violating them would either trip an assert (debug) or
// silently emit nonsense that the trampoline assembler would then reject.
// Verify them explicitly and log a specific error so the caller can mark the
// patch as a hard failure (see PatchContext::patch_failure).

static bool ValidateSplitOperands(SplitKind kind, const WmmaRegOperands &r,
                                  llvm::StringRef mnemonic) {
  auto logError = [&](llvm::StringRef reason) {
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WMMA split: invalid operands for " << mnemonic << ": "
        << reason << "\n";
  };

  // Shared across both families: all four operands must be present as
  // positive-width VGPR ranges, and C (src2) must share the destination's
  // shape because the accumulator aliases the destination after emission.
  if (r.dst.second <= 0 || r.src0.second <= 0 || r.src1.second <= 0 ||
      r.src2.second <= 0) {
    logError("non-positive VGPR range width");
    return false;
  }
  if (r.dst.second != r.src2.second) {
    logError("dst and src2 VGPR widths differ");
    return false;
  }

  switch (kind) {
  case SplitKind::Split128to64_FP8BF8:
    // K dimension (src0, src1) is halved; the resulting halves must each be
    // a whole number of VGPRs.
    if (r.src0.second % 2 != 0 || r.src1.second % 2 != 0) {
      logError("src0/src1 VGPR widths must be even to split K in half");
      return false;
    }
    return true;
  case SplitKind::Split32x16_to_16x16_F4:
    // M dimension (dst, src2) and A (src0) are halved.  B (src1) is
    // broadcast, so only its positivity matters (already checked above).
    if (r.dst.second % 2 != 0) {
      logError("dst VGPR width must be even to split M in half");
      return false;
    }
    if (r.src0.second % 2 != 0) {
      logError("src0 VGPR width must be even to split A in half");
      return false;
    }
    return true;
  }
  return false;
}

// ── Helper: emit replacement assembly lines ─────────────────────────────────

static std::vector<std::string>
BuildSplit128to64Asm(llvm::StringRef replacement, const WmmaRegOperands &r) {
  // Split K in half; dst and C are unchanged.  For the second half, C = dst
  // so the accumulator threads through from the first half.  Preconditions
  // are enforced by ValidateSplitOperands at the call site.
  assert(r.dst.second > 0 && r.src2.second == r.dst.second);
  assert(r.src0.second > 0 && r.src0.second % 2 == 0);
  assert(r.src1.second > 0 && r.src1.second % 2 == 0);

  int a_half = r.src0.second / 2;
  int b_half = r.src1.second / 2;

  const std::string dst = FormatVgprRange(r.dst.first, r.dst.second);
  const std::string c = FormatVgprRange(r.src2.first, r.src2.second);

  std::vector<std::string> out;
  out.reserve(2);
  out.push_back(llvm::formatv("{0} {1}, {2}, {3}, {4}", replacement, dst,
                              FormatVgprRange(r.src0.first, a_half),
                              FormatVgprRange(r.src1.first, b_half), c)
                    .str());
  out.push_back(llvm::formatv("{0} {1}, {2}, {3}, {4}", replacement, dst,
                              FormatVgprRange(r.src0.first + a_half, a_half),
                              FormatVgprRange(r.src1.first + b_half, b_half),
                              dst)
                    .str());
  return out;
}

static std::vector<std::string> BuildSplit32x16Asm(llvm::StringRef replacement,
                                                   const WmmaRegOperands &r) {
  // Split M in half.  A (src0) is split in half (A's 16 VGPRs carry the FP4
  // data for all 32 M rows; each half needs the A rows it accumulates over).
  // B (src1) is broadcast — each half reads the same N×K data.  dst / src2
  // are split in half by M.  The replacement uses the f8f6f4 WMMA with both
  // matrix format modifiers forced to MATRIX_FMT_FP4 so the data layout
  // matches the original f4 instruction.  Preconditions are enforced by
  // ValidateSplitOperands at the call site.
  assert(r.dst.second > 0 && r.dst.second % 2 == 0);
  assert(r.src2.second == r.dst.second);
  assert(r.src0.second > 0 && r.src0.second % 2 == 0);
  assert(r.src1.second > 0);

  int dst_half = r.dst.second / 2;
  int a_half = r.src0.second / 2;

  const std::string b = FormatVgprRange(r.src1.first, r.src1.second);
  constexpr llvm::StringLiteral kFmtSuffix =
      "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4";

  std::vector<std::string> out;
  out.reserve(2);
  out.push_back(llvm::formatv("{0} {1}, {2}, {3}, {4} {5}", replacement,
                              FormatVgprRange(r.dst.first, dst_half),
                              FormatVgprRange(r.src0.first, a_half), b,
                              FormatVgprRange(r.src2.first, dst_half),
                              kFmtSuffix)
                    .str());
  out.push_back(
      llvm::formatv("{0} {1}, {2}, {3}, {4} {5}", replacement,
                    FormatVgprRange(r.dst.first + dst_half, dst_half),
                    FormatVgprRange(r.src0.first + a_half, a_half), b,
                    FormatVgprRange(r.src2.first + dst_half, dst_half),
                    kFmtSuffix)
          .str());
  return out;
}

} // namespace

// ── Main patch entry point ──────────────────────────────────────────────────

uint32_t ApplyWmmaSplitPatches(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];

  WmmaSplitMatch match = LookupSplitRule(di.mnemonic);
  if (!match.matched)
    return 0;

  // Structural sanity check against the opcode side.  Exact-match against
  // kSplitTable already excludes SWMMAC and unrelated opcodes, but the
  // printed mnemonic is an MCInstPrinter artifact that is not guaranteed to
  // stay in lock-step with the TableGen operand layout across LLVM versions.
  // Every WMMA variant this patch handles has exactly one destination
  // operand at the MCInstrDesc level; a differing def count means the
  // operand layout is not what ExtractWmmaRegOperands expects, so refuse to
  // emit rather than produce silently-wrong asm.
  const auto &mcid = ctx.llvm_state.MCII->get(di.inst.getOpcode());
  if (mcid.getNumDefs() != 1) {
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WMMA split: " << di.mnemonic << " has "
        << mcid.getNumDefs() << " defs, expected 1\n";
    ctx.patch_failure = true;
    return 0;
  }

  HotswapLog(HotswapLogLevel::Debug)
      << "hotswap: WMMA split: " << di.mnemonic << " at offset 0x"
      << llvm::utohexstr(di.offset) << "\n";

  WmmaRegOperands regs = ExtractWmmaRegOperands(di.inst, *ctx.llvm_state.MRI);
  if (!regs.valid) {
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WMMA split: could not extract 4 VGPR operands from "
        << di.mnemonic << "\n";
    ctx.patch_failure = true;
    return 0;
  }

  if (!ValidateSplitOperands(match.kind, regs, di.mnemonic)) {
    ctx.patch_failure = true;
    return 0;
  }

  std::vector<std::string> asm_lines;
  switch (match.kind) {
  case SplitKind::Split128to64_FP8BF8:
    asm_lines = BuildSplit128to64Asm(match.replacement, regs);
    break;
  case SplitKind::Split32x16_to_16x16_F4:
    asm_lines = BuildSplit32x16Asm(match.replacement, regs);
    break;
  }

  if (asm_lines.empty()) {
    ctx.patch_failure = true;
    return 0;
  }

  // Emit via trampoline.  The back-branch is patched by
  // FixupTrampolineBranches once all trampolines have been laid out, so we
  // only need to reserve 4 bytes at the tail here.
  uint64_t tramp_text_offset = ctx.text_size;
  for (auto &t : ctx.out_trampolines)
    tramp_text_offset += t.bytes.size();

  Trampoline t = BuildTrampoline(asm_lines, di.offset, di.size,
                                 tramp_text_offset, ctx.config, ctx.llvm_state);
  if (t.bytes.empty()) {
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WMMA split: trampoline assembly failed for " << di.mnemonic
        << "\n";
    ctx.patch_failure = true;
    return 0;
  }

  ctx.out_trampolines.push_back(std::move(t));

  HotswapLog(HotswapLogLevel::Info)
      << "hotswap: WMMA split: patched " << di.mnemonic << " at offset 0x"
      << llvm::utohexstr(di.offset) << "\n";

  return 1;
}
