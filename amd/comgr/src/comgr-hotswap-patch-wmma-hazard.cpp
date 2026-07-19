//===- comgr-hotswap-patch-wmma-hazard.cpp - WMMA hazard patch -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Whole-kernel patch for the gfx1250 WMMA/SWMMAC co-execution hazard.
/// Detects WMMA/SWMMAC instructions that lack sufficient v_nop separation
/// before the first overlapping co-executable VALU, and inserts the required
/// v_nop padding.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/ArrayRef.h"
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
// first match wins. Do not reorder without verifying the required nop counts.
WmmaNopReq classifyWmmaNops(StringRef Mnemonic) {
  // Redundant in production (caller filters via isWmmaLike), but kept
  // as a defensive guard since classifyWmmaNops is a public function
  // also exercised directly by unit tests with non-WMMA mnemonics.
  bool IsWmma = Mnemonic.starts_with("v_wmma");
  bool IsSwmmac = Mnemonic.starts_with("v_swmmac");
  if (!IsWmma && !IsSwmmac)
    return {4, 4};

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

int updateWmmaHazardDeficit(DenseMap<size_t, int> &MaxDeficits,
                            size_t ValuIndex, int Deficit) {
  int &Maximum = MaxDeficits[ValuIndex];
  Maximum = std::max(Maximum, Deficit);
  return Maximum;
}

namespace {

// Scan one decoded stream for WMMA/SWMMAC -> overlapping co-executable VALU
// hazards. Original code uses its compiler-provided baseline as the scan gate;
// freshly emitted trampoline bodies must satisfy the absolute target budget.
// Multiple WMMAs may converge on one VALU, so retain the maximum deficit.
static std::optional<std::vector<WmmaHazard>> scanCoexecHazards(
    ArrayRef<InternalDecodedInst> Stream, const MCInstrInfo &MCII,
    const MCRegisterInfo &MRI, bool RequireAbsolute,
    int *WmmaScannedOut = nullptr) {
  DenseMap<size_t, int> MaxDeficitByValu;
  int WmmaScanned = 0;

  for (size_t WmmaIdx = 0, E = Stream.size(); WmmaIdx < E; ++WmmaIdx) {
    const InternalDecodedInst &WmmaDI = Stream[WmmaIdx];
    if (!isWmmaLike(WmmaDI.Inst, MCII))
      continue;

    ++WmmaScanned;
    const WmmaNopReq Req = classifyWmmaNops(WmmaDI.Mnemonic);
    if (!RequireAbsolute && Req.A0Nops <= Req.B0Nops)
      continue;
    const int Target = Req.A0Nops;

    int SafeSlots = 0;
    for (size_t ValuIdx = WmmaIdx + 1; ValuIdx < E; ++ValuIdx) {
      const InternalDecodedInst &Candidate = Stream[ValuIdx];
      if (Candidate.Mnemonic == "<unknown>" ||
          Candidate.Mnemonic == "<replaced>") {
        log() << "hotswap: error: cannot prove WMMA co-exec spacing from 0x"
              << utohexstr(WmmaDI.Offset) << " across instruction at 0x"
              << utohexstr(Candidate.Offset) << " (" << Candidate.Mnemonic
              << ")\n";
        return std::nullopt;
      }

      if (isVNop(Candidate)) {
        ++SafeSlots;
        if (SafeSlots >= Target)
          break;
        continue;
      }

      if (!hasTSFlags(Candidate.Inst, MCII, AmdgpuTSFlags::VALU)) {
        if (isTerminatingSalu(Candidate.Inst, MCII))
          break;
        continue;
      }

      if (isCoexecutableVALU(Candidate, MCII)) {
        if (!checkVgprOverlap(WmmaDI.Inst, Candidate.Inst, MRI)) {
          ++SafeSlots;
          if (SafeSlots >= Target)
            break;
          continue;
        }

        if (SafeSlots < Target) {
          const int Deficit = Target - SafeSlots;
          const int MaxDeficit =
              updateWmmaHazardDeficit(MaxDeficitByValu, ValuIdx, Deficit);
          log() << "hotswap: WMMA co-exec hazard at 0x"
                << utohexstr(WmmaDI.Offset) << ": " << WmmaDI.Mnemonic
                << " needs " << Target << " v_nops, only " << SafeSlots
                << " found before " << Candidate.Mnemonic << " at 0x"
                << utohexstr(Candidate.Offset) << " (candidate max deficit "
                << MaxDeficit << ")\n";
        }
        break;
      }

      break;
    }
  }

  std::vector<WmmaHazard> Hazards;
  Hazards.reserve(MaxDeficitByValu.size());
  for (size_t I = 0, E = Stream.size(); I != E; ++I)
    if (auto It = MaxDeficitByValu.find(I); It != MaxDeficitByValu.end())
      Hazards.push_back({I, It->second});
  if (WmmaScannedOut)
    *WmmaScannedOut = WmmaScanned;
  return Hazards;
}

static std::optional<std::vector<WmmaHazard>>
findWmmaCoexecHazards(const PatchContext &Ctx) {
  int WmmaScanned = 0;
  std::optional<std::vector<WmmaHazard>> Hazards = scanCoexecHazards(
      Ctx.Decoded, *Ctx.LS.MCII, *Ctx.LS.MRI,
      /*RequireAbsolute=*/false, &WmmaScanned);
  if (!Hazards)
    return std::nullopt;
  log() << "hotswap: WMMA co-exec validation: " << Hazards->size()
        << " hazards (" << WmmaScanned << " WMMA instructions scanned)\n";
  return Hazards;
}

// Emitted trampoline bodies were not present in Ctx.Decoded and carry no
// compiler spacing baseline. Validate their absolute WMMA/VALU separation
// before branch fixup and coalescing; malformed bodies fail the rewrite closed.
static void validateTrampolineCoexec(PatchContext &Ctx) {
  const MCInstrInfo &MCII = *Ctx.LS.MCII;
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  for (const Trampoline &T : Ctx.OutTrampolines) {
    const unsigned Reserve = T.Long ? SetPcReturnReserveBytes : MinInstSize;
    if (T.Bytes.size() <= Reserve)
      continue;
    const size_t BodySize = T.Bytes.size() - Reserve;

    std::vector<InternalDecodedInst> Body;
    if (!decodeTextSection(T.Bytes.data(), BodySize, Ctx.LS, Body)) {
      log() << "hotswap: error: could not decode trampoline body for WMMA "
               "co-exec validation at site 0x"
            << utohexstr(T.OriginalOffset) << "\n";
      Ctx.RequiredPatchFailed = true;
      continue;
    }

    std::optional<std::vector<WmmaHazard>> Hazards = scanCoexecHazards(
        Body, MCII, MRI, /*RequireAbsolute=*/true);
    if (!Hazards) {
      log() << "hotswap: error: indeterminate WMMA co-exec spacing in "
               "trampoline for site 0x"
            << utohexstr(T.OriginalOffset) << "\n";
      Ctx.RequiredPatchFailed = true;
      continue;
    }
    if (!Hazards->empty()) {
      log() << "hotswap: error: WMMA co-exec hazard unmitigated in trampoline "
               "for site 0x"
            << utohexstr(T.OriginalOffset) << " (" << Hazards->size()
            << " site(s)); failing closed\n";
      Ctx.RequiredPatchFailed = true;
    }
  }
}

} // anonymous namespace

static bool precomputeWmmaHazardsImpl(PatchContext &Ctx) {
  std::optional<std::vector<WmmaHazard>> Hazards =
      findWmmaCoexecHazards(Ctx);
  if (!Hazards) {
    Ctx.RequiredPatchFailed = true;
    return false;
  }
  for (const WmmaHazard &H : *Hazards) {
    if (H.ValuIdx >= Ctx.Decoded.size() || H.Deficit <= 0) {
      log() << "hotswap: error: invalid precomputed WMMA hazard candidate\n";
      Ctx.RequiredPatchFailed = true;
      return false;
    }
    const InternalDecodedInst &DI = Ctx.Decoded[H.ValuIdx];
    SiteReplacementState &State = Ctx.SiteReplacements[DI.Offset];
    if (State.Committed ||
        (State.OriginalSize != 0 && State.OriginalSize != DI.Size)) {
      log() << "hotswap: error: WMMA hazard candidate at 0x"
            << utohexstr(DI.Offset)
            << " conflicts with pre-existing replacement ownership\n";
      Ctx.RequiredPatchFailed = true;
      return false;
    }
    if (State.RequiredLeadingVNops == 0)
      Ctx.WmmaHazardSites.push_back(DI.Offset);
    State.OriginalSize = DI.Size;
    State.RequiredLeadingVNops =
        std::max(State.RequiredLeadingVNops, static_cast<unsigned>(H.Deficit));
  }
  return true;
}

static uint32_t applyWmmaHazardPatchImpl(PatchContext &Ctx) {
  for (uint64_t Offset : Ctx.WmmaHazardSites) {
    auto StateIt = Ctx.SiteReplacements.find(Offset);
    if (StateIt == Ctx.SiteReplacements.end() ||
        StateIt->second.RequiredLeadingVNops == 0) {
      log() << "hotswap: error: lost precomputed WMMA hazard state at 0x"
            << utohexstr(Offset) << "\n";
      Ctx.RequiredPatchFailed = true;
      return 0;
    }
    if (StateIt->second.WmmaHazardComposed)
      continue;
    if (StateIt->second.Committed) {
      log() << "hotswap: error: replacement at 0x" << utohexstr(Offset)
            << " committed without its WMMA hazard requirement\n";
      Ctx.RequiredPatchFailed = true;
      return 0;
    }

    auto DI = llvm::lower_bound(
        Ctx.Decoded, Offset,
        [](const InternalDecodedInst &Inst, uint64_t CandidateOffset) {
          return Inst.Offset < CandidateOffset;
        });
    if (DI == Ctx.Decoded.end() || DI->Offset != Offset ||
        DI->Size != StateIt->second.OriginalSize || Offset > Ctx.TextSize ||
        DI->Size > Ctx.TextSize - Offset) {
      log() << "hotswap: error: WMMA hazard source at 0x"
            << utohexstr(Offset) << " is not a valid current instruction\n";
      Ctx.RequiredPatchFailed = true;
      return 0;
    }

    // No per-instruction pass owned this site. Preserve the bytes after all
    // same-size in-place corrections, then let the central emission helper
    // prepend the precomputed v_nops and commit the single site owner.
    SmallVector<uint8_t> Current(Ctx.Text + Offset,
                                 Ctx.Text + Offset + DI->Size);
    if (!emitReplacementCode(Ctx, Offset, DI->Size, Current)) {
      log() << "hotswap: error: could not emit required WMMA hazard fix at 0x"
            << utohexstr(Offset) << "\n";
      Ctx.RequiredPatchFailed = true;
      return 0;
    }

    StateIt = Ctx.SiteReplacements.find(Offset);
    if (StateIt == Ctx.SiteReplacements.end() ||
        !StateIt->second.WmmaHazardComposed) {
      log() << "hotswap: error: WMMA hazard requirement at 0x"
            << utohexstr(Offset) << " was not composed by emission\n";
      Ctx.RequiredPatchFailed = true;
      return 0;
    }
  }

  if (Ctx.WmmaHazardsComposed != Ctx.WmmaHazardSites.size()) {
    log() << "hotswap: error: composed " << Ctx.WmmaHazardsComposed << " of "
          << Ctx.WmmaHazardSites.size()
          << " precomputed WMMA hazard requirements\n";
    Ctx.RequiredPatchFailed = true;
    return 0;
  }

  // Scale16 and other patch families can emit WMMA-bearing bodies that were
  // absent from the immutable source stream. Enforce their full target spacing.
  validateTrampolineCoexec(Ctx);
  if (Ctx.RequiredPatchFailed)
    return 0;
  return Ctx.WmmaHazardsComposed;
}

void registerWmmaHazardPatch(HotswapPatchVTable &VT) {
  VT.precomputeWmmaHazards = &precomputeWmmaHazardsImpl;
  VT.applyWmmaHazardPatch = &applyWmmaHazardPatchImpl;
}

} // namespace hotswap
} // namespace COMGR
