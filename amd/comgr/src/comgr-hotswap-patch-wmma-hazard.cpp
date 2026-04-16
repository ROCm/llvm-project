//===- comgr-hotswap-patch-wmma-hazard.cpp - WMMA hazard patch -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Whole-kernel patch for the GFX1250 A0 WMMA/SWMMAC co-execution hazard.
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"

namespace {

struct WmmaHazard {
  size_t valu_idx;
  int deficit;
};

namespace AmdgpuTSFlags {
// Mirrors the AMDGPU MC instruction flags in llvm/lib/Target/AMDGPU/SIDefines.h
// so the hotswap pass can classify decoded MC instructions without depending on
// backend-private MachineInstr helpers.
static constexpr uint64_t VALU = UINT64_C(1) << 1;
static constexpr uint64_t IsWMMA = UINT64_C(1) << 59;
static constexpr uint64_t IsSWMMAC = UINT64_C(1) << 63;
} // namespace AmdgpuTSFlags

static uint64_t GetTSFlags(const llvm::MCInst &inst,
                           const llvm::MCInstrInfo &MCII) {
  return MCII.get(inst.getOpcode()).TSFlags;
}

static bool HasTSFlags(const llvm::MCInst &inst, const llvm::MCInstrInfo &MCII,
                       uint64_t mask) {
  return (GetTSFlags(inst, MCII) & mask) != 0;
}

static bool IsWmmaLike(const llvm::MCInst &inst,
                       const llvm::MCInstrInfo &MCII) {
  return HasTSFlags(inst, MCII,
                    AmdgpuTSFlags::IsWMMA | AmdgpuTSFlags::IsSWMMAC);
}

static bool IsCoexecutableVALUInst(const InternalDecodedInst &inst,
                                   const llvm::MCInstrInfo &MCII) {
  if (inst.mnemonic == "v_nop")
    return false;
  if (!HasTSFlags(inst.inst, MCII, AmdgpuTSFlags::VALU))
    return false;
  return !IsWmmaLike(inst.inst, MCII);
}

static bool IsTerminatingSalu(const std::string &mnemonic) {
  llvm::StringRef mnemonic_ref(mnemonic);
  return mnemonic_ref.starts_with("s_branch") ||
         mnemonic_ref.starts_with("s_cbranch") || mnemonic_ref == "s_endpgm" ||
         mnemonic_ref == "s_setpc" || mnemonic_ref == "s_swappc" ||
         mnemonic_ref == "s_call";
}

static std::vector<WmmaHazard>
ValidateWmmaCoexecHazards(const PatchContext &ctx) {
  const llvm::MCInstrInfo &MCII = *ctx.llvm_state.MCII;
  const llvm::MCRegisterInfo &MRI = *ctx.llvm_state.MRI;
  std::vector<WmmaHazard> hazards;
  int wmma_scanned = 0;

  for (size_t wmma_idx = 0; wmma_idx < ctx.decoded.size(); ++wmma_idx) {
    const auto &wmma = ctx.decoded[wmma_idx];
    if (!IsWmmaLike(wmma.inst, MCII))
      continue;

    ++wmma_scanned;
    WmmaNopReq requirement = ClassifyWmmaNops(wmma.mnemonic);
    if (requirement.a0_nops <= requirement.b0_nops)
      continue;

    int safe_slots = 0;
    for (size_t valu_idx = wmma_idx + 1; valu_idx < ctx.decoded.size();
         ++valu_idx) {
      const auto &candidate = ctx.decoded[valu_idx];

      if (candidate.mnemonic == "v_nop") {
        ++safe_slots;
        if (safe_slots >= requirement.a0_nops)
          break;
        continue;
      }

      if (llvm::StringRef(candidate.mnemonic).starts_with("s_")) {
        if (IsTerminatingSalu(candidate.mnemonic))
          break;
        continue;
      }

      if (IsCoexecutableVALUInst(candidate, MCII)) {
        if (!CheckVgprOverlap(wmma.inst, candidate.inst, MCII, MRI)) {
          ++safe_slots;
          if (safe_slots >= requirement.a0_nops)
            break;
          continue;
        }

        if (safe_slots < requirement.a0_nops) {
          hazards.push_back({valu_idx, requirement.a0_nops - safe_slots});
          HotswapLog(HotswapLogLevel::Debug)
              << "hotswap: WMMA co-exec hazard @0x"
              << llvm::utohexstr(wmma.offset) << ": " << wmma.mnemonic
              << " needs " << requirement.a0_nops << " V_NOPs for A0, only "
              << safe_slots << " found before " << candidate.mnemonic << " @0x"
              << llvm::utohexstr(candidate.offset) << "\n";
        }
        break;
      }

      break;
    }
  }

  HotswapLog(HotswapLogLevel::Info)
      << "hotswap: WMMA co-exec validation: " << hazards.size() << " hazards ("
      << wmma_scanned << " WMMA instructions scanned)\n";
  return hazards;
}

} // namespace

WmmaNopReq ClassifyWmmaNops(const std::string &mnemonic) {
  llvm::StringRef mnemonic_ref(mnemonic);
  bool is_wmma = mnemonic_ref.starts_with("v_wmma");
  bool is_swmmac = mnemonic_ref.starts_with("v_swmmac");
  if (!is_wmma && !is_swmmac)
    return {4, 4};

  if (mnemonic_ref.contains("_iu8") || mnemonic_ref.contains("_iu4"))
    return {8, 4};

  if (mnemonic_ref.contains("f8f6f4"))
    return {1, 4};

  bool has_f8 = mnemonic_ref.contains("_fp8") || mnemonic_ref.contains("_f8") ||
                mnemonic_ref.contains("_bf8");
  if (has_f8) {
    if (mnemonic_ref.contains("16x16x128"))
      return {3, 4};
    return {1, 4};
  }

  if (mnemonic_ref.contains("_f16") || mnemonic_ref.contains("_bf16"))
    return {4, 4};

  return {4, 4};
}

uint32_t ApplyWmmaHazardPatch(PatchContext &ctx) {
  std::vector<WmmaHazard> hazards = ValidateWmmaCoexecHazards(ctx);
  if (hazards.empty())
    return 0;

  auto vnop_bytes = AssembleSingleInst("v_nop", ctx.llvm_state);
  if (vnop_bytes.size() != kMinInstSize) {
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WMMA hazard: v_nop assembly failed\n";
    return 0;
  }

  uint32_t patched = 0;
  for (const auto &hazard : hazards) {
    const auto &valu = ctx.decoded[hazard.valu_idx];
    std::vector<uint8_t> replacement;
    replacement.reserve(static_cast<size_t>(hazard.deficit) * kMinInstSize +
                        valu.size);

    for (int nop_count = 0; nop_count < hazard.deficit; ++nop_count)
      replacement.insert(replacement.end(), vnop_bytes.begin(),
                         vnop_bytes.end());

    replacement.insert(replacement.end(), ctx.text + valu.offset,
                       ctx.text + valu.offset + valu.size);

    if (!EmitReplacementCode(ctx, valu.offset, valu.size, replacement)) {
      HotswapLog(HotswapLogLevel::Error)
          << "hotswap: WMMA hazard @0x" << llvm::utohexstr(valu.offset)
          << ": failed to emit " << hazard.deficit << " v_nop(s)\n";
      continue;
    }

    HotswapLog(HotswapLogLevel::Info)
        << "hotswap: WMMA hazard fix @0x" << llvm::utohexstr(valu.offset)
        << ": inserted " << hazard.deficit << " v_nop(s)\n";
    ++patched;
  }

  return patched;
}
