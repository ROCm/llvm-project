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

#if !defined(_MSC_VER)

#include "llvm/ADT/StringExtras.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

struct WmmaHazard {
  size_t ValuIdx;
  int Deficit;
};

struct WmmaNopReq {
  int A0Nops;
  int B0Nops;
};

namespace AmdgpuTSFlags {
static constexpr uint64_t VALU = UINT64_C(1) << 1;
static constexpr uint64_t IsWMMA = UINT64_C(1) << 59;
static constexpr uint64_t IsSWMMAC = UINT64_C(1) << 63;
} // namespace AmdgpuTSFlags

static uint64_t getTSFlags(const MCInst &Inst, const MCInstrInfo &MCII) {
  return MCII.get(Inst.getOpcode()).TSFlags;
}

static bool hasTSFlags(const MCInst &Inst, const MCInstrInfo &MCII,
                       uint64_t Mask) {
  return (getTSFlags(Inst, MCII) & Mask) != 0;
}

static bool isWmmaLike(const MCInst &Inst, const MCInstrInfo &MCII) {
  return hasTSFlags(Inst, MCII,
                    AmdgpuTSFlags::IsWMMA | AmdgpuTSFlags::IsSWMMAC);
}

static bool isVNop(const MCInst &Inst, const MCInstrInfo &MCII) {
  return MCII.getName(Inst.getOpcode()) == "V_NOP_e32";
}

static bool isCoexecutableVALU(const InternalDecodedInst &DI,
                               const MCInstrInfo &MCII) {
  if (isVNop(DI.Inst, MCII))
    return false;
  if (!hasTSFlags(DI.Inst, MCII, AmdgpuTSFlags::VALU))
    return false;
  return !isWmmaLike(DI.Inst, MCII);
}

static bool isTerminatingSalu(const MCInst &Inst, const MCInstrInfo &MCII) {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  return Desc.isTerminator() || Desc.isBranch() || Desc.isCall() ||
         Desc.isReturn();
}

static WmmaNopReq classifyWmmaNops(StringRef Mnemonic) {
  bool IsWmma = Mnemonic.starts_with("v_wmma");
  bool IsSwmmac = Mnemonic.starts_with("v_swmmac");
  if (!IsWmma && !IsSwmmac)
    return {4, 4};

  if (Mnemonic.contains("_iu8") || Mnemonic.contains("_iu4"))
    return {8, 4};

  if (Mnemonic.contains("f8f6f4"))
    return {1, 4};

  bool HasF8 = Mnemonic.contains("_fp8") || Mnemonic.contains("_f8") ||
               Mnemonic.contains("_bf8");
  if (HasF8) {
    if (Mnemonic.contains("16x16x128"))
      return {3, 4};
    return {1, 4};
  }

  if (Mnemonic.contains("_f16") || Mnemonic.contains("_bf16"))
    return {4, 4};

  return {4, 4};
}

static std::vector<WmmaHazard>
validateWmmaCoexecHazards(const PatchContext &Ctx) {
  const MCInstrInfo &MCII = *Ctx.LS.MCII;
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  std::vector<WmmaHazard> Hazards;
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

      if (isVNop(Candidate.Inst, MCII)) {
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
        if (!checkVgprOverlap(WmmaDI.Inst, Candidate.Inst, MRI)) {
          ++SafeSlots;
          if (SafeSlots >= Req.A0Nops)
            break;
          continue;
        }

        if (SafeSlots < Req.A0Nops) {
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

uint32_t applyWmmaHazardPatch(PatchContext &Ctx) {
  std::vector<WmmaHazard> Hazards = validateWmmaCoexecHazards(Ctx);
  if (Hazards.empty())
    return 0;

  SmallVector<uint8_t> VnopBytes = assembleSingleInst("v_nop", Ctx.LS);
  if (VnopBytes.size() != 4) {
    log() << "hotswap: error: WMMA hazard: v_nop assembly failed\n";
    return 0;
  }

  uint32_t Patched = 0;
  for (const WmmaHazard &H : Hazards) {
    const InternalDecodedInst &ValuDI = Ctx.Decoded[H.ValuIdx];

    uint64_t TrampolineTextOffset = Ctx.TextSize;
    for (const Trampoline &T : Ctx.OutTrampolines)
      TrampolineTextOffset += T.Bytes.size();

    SmallVector<std::string> AsmLines;
    for (int I = 0; I < H.Deficit; ++I)
      AsmLines.push_back("v_nop");

    std::string PrintedInst;
    raw_string_ostream OS(PrintedInst);
    Ctx.LS.MCIP->printInst(&ValuDI.Inst, 0, "", *Ctx.LS.STI, OS);
    AsmLines.push_back(StringRef(PrintedInst).trim().str());

    Trampoline T = buildTrampoline(AsmLines, ValuDI.Offset, ValuDI.Size,
                                   TrampolineTextOffset, Ctx.LS);
    Ctx.OutTrampolines.push_back(std::move(T));

    log() << "hotswap: WMMA hazard fix at 0x" << utohexstr(ValuDI.Offset)
          << ": inserted " << H.Deficit << " v_nop(s)\n";
    ++Patched;
  }

  return Patched;
}

} // namespace hotswap
} // namespace COMGR

#endif // !defined(_MSC_VER)
