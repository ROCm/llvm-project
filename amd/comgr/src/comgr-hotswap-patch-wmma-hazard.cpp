//===- comgr-hotswap-patch-wmma-hazard.cpp - WMMA hazard patch -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Whole-kernel patch for the GFX1250 A0 WMMA/SWMMAC co-execution hazard.
/// Detects WMMA/SWMMAC instructions that lack sufficient v_nop separation
/// before the first overlapping co-executable VALU, and inserts the required
/// v_nop padding.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

struct WmmaHazard {
  size_t ValuIdx;
  int Deficit;
};

// Mirrors SIInstrFlags from llvm/lib/Target/AMDGPU/SIDefines.h.
// SIDefines.h is a backend-private header (not installed), so we
// duplicate the bit positions here. These must stay in sync with
// the AMDGPU backend; verify against SIDefines.h if TSFlags layout
// changes upstream.
namespace AmdgpuTSFlags {
static constexpr uint64_t VALU = UINT64_C(1) << 1;
static constexpr uint64_t IsWMMA = UINT64_C(1) << 59;
static constexpr uint64_t IsSWMMAC = UINT64_C(1) << 63;
} // namespace AmdgpuTSFlags

uint64_t getTSFlags(const MCInst &Inst, const MCInstrInfo &MCII) {
  return MCII.get(Inst.getOpcode()).TSFlags;
}

bool hasTSFlags(const MCInst &Inst, const MCInstrInfo &MCII, uint64_t Mask) {
  return (getTSFlags(Inst, MCII) & Mask) != 0;
}

bool isWmmaLike(const MCInst &Inst, const MCInstrInfo &MCII) {
  return hasTSFlags(Inst, MCII,
                    AmdgpuTSFlags::IsWMMA | AmdgpuTSFlags::IsSWMMAC);
}

bool isVNop(const InternalDecodedInst &DI) { return DI.Mnemonic == "v_nop"; }

bool isCoexecutableVALU(const InternalDecodedInst &DI,
                        const MCInstrInfo &MCII) {
  if (isVNop(DI))
    return false;
  if (!hasTSFlags(DI.Inst, MCII, AmdgpuTSFlags::VALU))
    return false;
  return !isWmmaLike(DI.Inst, MCII);
}

bool isTerminatingSalu(const MCInst &Inst, const MCInstrInfo &MCII) {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  return Desc.isTerminator() || Desc.isBranch() || Desc.isCall() ||
         Desc.isReturn();
}

} // anonymous namespace

// Checks are ordered most-restrictive-first. If a mnemonic matches
// multiple substrings (e.g. contains both "_iu8" and "_f16"), the
// first match wins. Do not reorder without verifying A0 nop counts.
WmmaNopReq classifyWmmaNops(StringRef Mnemonic) {
  // Redundant in production (caller filters via isWmmaLike), but kept
  // as a defensive guard since classifyWmmaNops is a public function
  // also exercised directly by unit tests with non-WMMA mnemonics.
  bool IsWmma = Mnemonic.starts_with("v_wmma");
  bool IsSwmmac = Mnemonic.starts_with("v_swmmac");
  if (!IsWmma && !IsSwmmac)
    return {4, 4};

  // The backend's sparse SWMMAC category needs four VALU slots on A0; dense
  // SWMMAC needs only two. A valid B0 object already has four, so neither
  // category requires HotSwap padding.
  if (IsSwmmac)
    return Mnemonic.contains("_iu8") || Mnemonic.contains("_iu4")
               ? WmmaNopReq{4, 4}
               : WmmaNopReq{2, 4};

  if (Mnemonic.contains("_iu8") || Mnemonic.contains("_iu4"))
    return {8, 4};

  if (Mnemonic.contains("f8f6f4"))
    return {1, 4};

  if (Mnemonic.contains("_fp8") || Mnemonic.contains("_f8") ||
      Mnemonic.contains("_bf8")) {
    if (Mnemonic.contains("16x16x128"))
      return {3, 4};
    return {1, 4};
  }

  if (Mnemonic.contains("_f16") || Mnemonic.contains("_bf16"))
    return {4, 4};

  return {4, 4};
}

namespace {

std::vector<WmmaHazard> findWmmaCoexecHazards(const PatchContext &Ctx) {
  const MCInstrInfo &MCII = *Ctx.LS.MCII;
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  std::vector<WmmaHazard> Hazards;
  DenseSet<size_t> PatchedValuIndices;
  int WmmaScanned = 0;

  for (size_t WmmaIdx = 0, E = Ctx.Decoded.size(); WmmaIdx < E; ++WmmaIdx) {
    const InternalDecodedInst &WmmaDI = Ctx.Decoded[WmmaIdx];
    if (!isWmmaLike(WmmaDI.Inst, MCII))
      continue;

    ++WmmaScanned;
    WmmaNopReq Req = classifyWmmaNops(WmmaDI.Mnemonic);
    if (Req.A0Nops <= Req.B0Nops)
      continue;

    int SafeSlots = 0;
    for (size_t ValuIdx = WmmaIdx + 1; ValuIdx < E; ++ValuIdx) {
      const InternalDecodedInst &Candidate = Ctx.Decoded[ValuIdx];

      if (isVNop(Candidate)) {
        ++SafeSlots;
        if (SafeSlots >= Req.A0Nops)
          break;
        continue;
      }

      if (!hasTSFlags(Candidate.Inst, MCII, AmdgpuTSFlags::VALU)) {
        if (isTerminatingSalu(Candidate.Inst, MCII))
          break;
        continue;
      }

      if (isCoexecutableVALU(Candidate, MCII)) {
        if (!checkVgprOverlap(WmmaDI.Inst, Candidate.Inst, MCII, MRI)) {
          ++SafeSlots;
          if (SafeSlots >= Req.A0Nops)
            break;
          continue;
        }

        if (SafeSlots < Req.A0Nops &&
            PatchedValuIndices.insert(ValuIdx).second) {
          Hazards.push_back({ValuIdx, Req.A0Nops - SafeSlots});
          log() << "hotswap: WMMA co-exec hazard at 0x"
                << utohexstr(WmmaDI.Offset) << ": " << WmmaDI.Mnemonic
                << " needs " << Req.A0Nops << " v_nops, only " << SafeSlots
                << " found before " << Candidate.Mnemonic << " at 0x"
                << utohexstr(Candidate.Offset) << "\n";
        }
        break;
      }

      break;
    }
  }

  log() << "hotswap: WMMA co-exec validation: " << Hazards.size()
        << " hazards (" << WmmaScanned << " WMMA instructions scanned)\n";
  return Hazards;
}

} // anonymous namespace

static uint32_t applyWmmaHazardPatchImpl(PatchContext &Ctx) {
  std::vector<WmmaHazard> Hazards = findWmmaCoexecHazards(Ctx);
  if (Hazards.empty())
    return 0;

  uint32_t Patched = 0;
  for (const WmmaHazard &H : Hazards) {
    const InternalDecodedInst &ValuDI = Ctx.Decoded[H.ValuIdx];

    SmallVector<uint8_t> Padding;
    for (int I = 0; I < H.Deficit; ++I)
      Padding.append(encodeInstructionBytes(Ctx.LS.VNopInst, Ctx.LS));
    if (Padding.size() != static_cast<size_t>(H.Deficit) * MinInstSize) {
      log() << "hotswap: error: WMMA hazard: v_nop encoding failed at 0x"
            << utohexstr(ValuDI.Offset) << "\n";
      Ctx.RequiredPatchFailed = true;
      continue;
    }

    auto Existing = llvm::find_if(Ctx.OutTrampolines, [&](const Trampoline &T) {
      return T.OriginalOffset == ValuDI.Offset;
    });
    if (Existing != Ctx.OutTrampolines.end()) {
      Existing->Bytes.insert(Existing->Bytes.begin(), Padding.begin(),
                             Padding.end());
      auto Recorded = Ctx.ReplacementCodeBySite.find(ValuDI.Offset);
      if (Recorded != Ctx.ReplacementCodeBySite.end())
        Recorded->second.insert(Recorded->second.begin(), Padding.begin(),
                                Padding.end());
    } else {
      SmallVector<uint8_t> Replacement = Padding;
      if (Ctx.MutatedOffsets.contains(ValuDI.Offset)) {
        auto Recorded = Ctx.ReplacementCodeBySite.find(ValuDI.Offset);
        if (Recorded == Ctx.ReplacementCodeBySite.end()) {
          log() << "hotswap: error: WMMA hazard: site 0x"
                << utohexstr(ValuDI.Offset)
                << " was mutated without a semantic replacement; refusing "
                   "to relocate raw control-flow bytes\n";
          Ctx.RequiredPatchFailed = true;
          continue;
        }
        Replacement.append(Recorded->second.begin(), Recorded->second.end());
      } else {
        SmallVector<uint8_t> Original =
            encodeInstructionBytes(ValuDI.Inst, Ctx.LS);
        if (Original.size() != ValuDI.Size) {
          log() << "hotswap: error: WMMA hazard: VALU encoding failed at 0x"
                << utohexstr(ValuDI.Offset) << "\n";
          Ctx.RequiredPatchFailed = true;
          continue;
        }
        Replacement.append(Original.begin(), Original.end());
      }
      if (!emitReplacementCode(Ctx, ValuDI.Offset, ValuDI.Size, Replacement,
                               /*AllowSafeFarReturn=*/true)) {
        log() << "hotswap: error: WMMA hazard: emission failed at 0x"
              << utohexstr(ValuDI.Offset) << "\n";
        Ctx.RequiredPatchFailed = true;
        continue;
      }
    }
    Ctx.MutatedOffsets.insert(ValuDI.Offset);

    log() << "hotswap: WMMA hazard fix at 0x" << utohexstr(ValuDI.Offset)
          << ": inserted " << H.Deficit << " v_nop(s)\n";
    ++Patched;
  }

  return Patched;
}

void registerWmmaHazardPatch(HotswapPatchVTable &VT) {
  VT.applyWmmaHazardPatch = &applyWmmaHazardPatchImpl;
}

} // namespace hotswap
} // namespace COMGR
