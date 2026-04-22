//===- comgr-hotswap-patch-trampoline.cpp - B0-to-A0 trampoline patches ---===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applyTrampolinePatches. Handles B0 errata
/// whose fix is larger than the original instruction:
///   - ds_*_2addr_stride64_*  : one 8B DS instruction -> two single-address DS
///   - tensor_load_to_lds     : prepend s_pack_hh_b32_b16 to clear multicast
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <charconv>
#include <string>
#include <utility>
#include <vector>

namespace COMGR {
namespace hotswap {

static std::string printInst(const InternalDecodedInst &DI,
                             const LLVMState &LS) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  LS.MCIP->printInst(&DI.Inst, 0, "", *LS.STI, OS);
  return S;
}

// -- DS stride64 swap table ---------------------------------------------------

static const std::pair<llvm::StringRef, llvm::StringRef>
    kDs2AddrStride64Swaps[] = {
        {"ds_load_2addr_stride64_b32", "ds_load_b32"},
        {"ds_load_2addr_stride64_b64", "ds_load_b64"},
        {"ds_store_2addr_stride64_b32", "ds_store_b32"},
        {"ds_store_2addr_stride64_b64", "ds_store_b64"},
        {"ds_storexchg_2addr_stride64_rtn_b32", "ds_storexchg_rtn_b32"},
        {"ds_storexchg_2addr_stride64_rtn_b64", "ds_storexchg_rtn_b64"},
};

static bool isDs2AddrStride64(llvm::StringRef Mnem) {
  return Mnem.contains("_2addr_stride64_");
}

static std::pair<llvm::StringRef, llvm::StringRef>
lookupDs2AddrSwap(llvm::StringRef Mnem) {
  for (const auto &P : kDs2AddrStride64Swaps) {
    if (Mnem == P.first)
      return P;
  }
  return {"", ""};
}

// -- expandDs2AddrAsm ---------------------------------------------------------
//
// Ported from ROCR's hotswap_core.cpp. Parses the printed assembly of a
// ds_*_2addr_stride64_* instruction, scales stride64 offsets, splits register
// pairs, and produces two single-address DS instructions.

static std::string extractOffsetVal(const std::string &S,
                                    const std::string &Key) {
  size_t Pos = S.find(Key);
  if (Pos == std::string::npos)
    return "0";
  size_t VStart = Pos + Key.size();
  size_t VEnd = VStart;
  while (VEnd < S.size() && S[VEnd] != ' ' && S[VEnd] != '\t' && S[VEnd] != ',')
    VEnd++;
  return S.substr(VStart, VEnd - VStart);
}

static void removeToken(std::string &S, const std::string &Prefix) {
  size_t Pos = S.find(Prefix);
  if (Pos == std::string::npos)
    return;
  size_t End = Pos + Prefix.size();
  while (End < S.size() && S[End] != ' ' && S[End] != '\t' && S[End] != ',')
    End++;
  while (End < S.size() && (S[End] == ' ' || S[End] == '\t' || S[End] == ','))
    End++;
  S.erase(Pos, End - Pos);
}

static std::string extractFirstOfPair(const std::string &Reg) {
  size_t Bracket = Reg.find('[');
  if (Bracket == std::string::npos)
    return Reg;
  size_t Colon = Reg.find(':', Bracket);
  if (Colon == std::string::npos)
    return Reg;
  return Reg.substr(0, Bracket) +
         Reg.substr(Bracket + 1, Colon - Bracket - 1);
}

static std::string extractSecondOfPair(const std::string &Reg) {
  size_t Colon = Reg.find(':');
  size_t Close = Reg.find(']');
  if (Colon == std::string::npos || Close == std::string::npos)
    return Reg;
  return Reg.substr(0, Reg.find('[')) +
         Reg.substr(Colon + 1, Close - Colon - 1);
}

static std::string withOffset(const std::string &Base, const std::string &Off) {
  if (Off == "0" || Off.empty())
    return Base;
  return Base + " offset:" + Off;
}

static std::vector<std::string> splitOperands(const std::string &OperandStr) {
  std::vector<std::string> Ops;
  std::string Rest = OperandStr;
  size_t S = Rest.find_first_not_of(" \t");
  if (S != std::string::npos)
    Rest = Rest.substr(S);
  while (!Rest.empty()) {
    size_t Comma = Rest.find(',');
    if (Comma == std::string::npos) {
      std::string Tok = Rest;
      S = Tok.find_first_not_of(" \t");
      size_t E = Tok.find_last_not_of(" \t");
      if (S != std::string::npos && E != std::string::npos)
        Tok = Tok.substr(S, E - S + 1);
      if (!Tok.empty())
        Ops.push_back(Tok);
      break;
    }
    std::string Tok = Rest.substr(0, Comma);
    S = Tok.find_first_not_of(" \t");
    size_t E = Tok.find_last_not_of(" \t");
    if (S != std::string::npos && E != std::string::npos)
      Tok = Tok.substr(S, E - S + 1);
    if (!Tok.empty())
      Ops.push_back(Tok);
    Rest = Rest.substr(Comma + 1);
    S = Rest.find_first_not_of(" \t");
    if (S != std::string::npos)
      Rest = Rest.substr(S);
  }
  return Ops;
}

std::vector<std::string> expandDs2AddrAsm(const std::string &PrintedAsm,
                                          const std::string &FromMnemonic,
                                          const std::string &ToMnemonic) {
  size_t Start = PrintedAsm.find_first_not_of(" \t");
  if (Start == std::string::npos)
    return {};
  size_t MnemEnd = PrintedAsm.find_first_of(" \t", Start);
  if (MnemEnd == std::string::npos)
    return {};

  std::string OperandStr = PrintedAsm.substr(MnemEnd);

  std::string Off0Val = extractOffsetVal(OperandStr, "offset0:");
  std::string Off1Val = extractOffsetVal(OperandStr, "offset1:");

  if (FromMnemonic.find("stride64") != std::string::npos) {
    uint32_t ElemBytes =
        (FromMnemonic.find("_b64") != std::string::npos) ? 8 : 4;
    uint32_t Scale = 64 * ElemBytes;
    auto ScaleVal = [Scale](std::string &Val) {
      if (Val == "0" || Val.empty())
        return;
      uint32_t V = 0;
      auto Result = std::from_chars(Val.data(), Val.data() + Val.size(), V);
      if (Result.ec != std::errc{})
        return;
      Val = std::to_string(V * Scale);
    };
    ScaleVal(Off0Val);
    ScaleVal(Off1Val);
  }

  removeToken(OperandStr, "offset0:");
  removeToken(OperandStr, "offset1:");

  std::vector<std::string> AllOps = splitOperands(OperandStr);
  std::vector<std::string> Ops;
  for (auto &Op : AllOps) {
    if (Op.find("offset") == std::string::npos)
      Ops.push_back(Op);
  }

  bool IsLoad = (FromMnemonic.find("ds_load") == 0);
  bool IsStore = (FromMnemonic.find("ds_store_") == 0 ||
                  FromMnemonic.find("ds_store_2addr") == 0) &&
                 FromMnemonic.find("xchg") == std::string::npos;
  bool IsXchg = (FromMnemonic.find("ds_storexchg") == 0);

  if (IsLoad && Ops.size() >= 2) {
    std::string D0 = extractFirstOfPair(Ops[0]);
    std::string D1 = extractSecondOfPair(Ops[0]);
    std::string Addr = Ops[1];
    return {
        withOffset(ToMnemonic + " " + D0 + ", " + Addr, Off0Val),
        withOffset(ToMnemonic + " " + D1 + ", " + Addr, Off1Val),
    };
  }

  if (IsStore && Ops.size() >= 3) {
    return {
        withOffset(ToMnemonic + " " + Ops[0] + ", " + Ops[1], Off0Val),
        withOffset(ToMnemonic + " " + Ops[0] + ", " + Ops[2], Off1Val),
    };
  }

  if (IsXchg && Ops.size() >= 4) {
    std::string D0 = extractFirstOfPair(Ops[0]);
    std::string D1 = extractSecondOfPair(Ops[0]);
    return {
        withOffset(ToMnemonic + " " + D0 + ", " + Ops[1] + ", " + Ops[2],
                   Off0Val),
        withOffset(ToMnemonic + " " + D1 + ", " + Ops[1] + ", " + Ops[3],
                   Off1Val),
    };
  }

  return {};
}

// -- bumpNextWaitDscnt --------------------------------------------------------
//
// After splitting one DS 2-addr instruction into two, the next s_wait_dscnt
// in the stream must be incremented by 1 to account for the extra outstanding
// DS operation.

static void bumpNextWaitDscnt(PatchContext &Ctx, size_t Idx) {
  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    if (Ctx.Decoded[I].Mnemonic == "s_wait_dscnt") {
      uint64_t Off = Ctx.Decoded[I].Offset;
      uint32_t Dw;
      std::memcpy(&Dw, Ctx.Text + Off, sizeof(Dw));
      uint32_t Imm = Dw & 0xFFFFu;
      Imm += 1;
      Dw = (Dw & 0xFFFF0000u) | (Imm & 0xFFFFu);
      std::memcpy(Ctx.Text + Off, &Dw, sizeof(Dw));
      return;
    }
  }
}

// -- extractDescriptorBaseReg -------------------------------------------------
//
// Extract the base scalar register from the second operand of a
// tensor_load_to_lds instruction. The second operand is an 8-SGPR group
// descriptor (e.g., s[4:11]); the multicast routing bits live in its first
// word, so we need the base register name (e.g., "s4") for s_pack_hh_b32_b16.
//
// Format: tensor_load_to_lds <op0>, s[N:N+7], ...

static std::string extractDescriptorBaseReg(const InternalDecodedInst &DI,
                                              const LLVMState &LS) {
  std::string Printed = printInst(DI, LS);

  size_t FirstComma = Printed.find(',');
  if (FirstComma == std::string::npos)
    return "";
  std::string After = Printed.substr(FirstComma + 1);

  size_t SPos = After.find("s[");
  if (SPos == std::string::npos) {
    SPos = After.find('s');
    if (SPos == std::string::npos)
      return "";
    size_t NumStart = SPos + 1;
    size_t NumEnd = NumStart;
    while (NumEnd < After.size() && After[NumEnd] >= '0' &&
           After[NumEnd] <= '9')
      NumEnd++;
    return (NumEnd > NumStart)
               ? "s" + After.substr(NumStart, NumEnd - NumStart)
               : "";
  }

  size_t Bracket = SPos + 1;
  size_t Colon = After.find(':', Bracket + 1);
  if (Colon == std::string::npos)
    return "";
  std::string Num = After.substr(Bracket + 1, Colon - Bracket - 1);
  return "s" + Num;
}

// -- isSgprLiveAfter ----------------------------------------------------------
//
// Conservative forward-scan heuristic. Returns true if the given SGPR
// (identified by its MCRegister) is used before being redefined in the
// instruction stream following Idx. Conservatively returns true on branches,
// s_endpgm, or end of stream.

bool isSgprLiveAfter(const PatchContext &Ctx, size_t Idx, unsigned SgprMCReg) {
  if (SgprMCReg == 0)
    return true;

  const llvm::MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const llvm::MCInstrInfo &MCII = *Ctx.LS.MCII;

  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    const auto &DI = Ctx.Decoded[I];
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>")
      continue;

    const llvm::MCInst &Inst = DI.Inst;
    const llvm::MCInstrDesc &Desc = MCII.get(Inst.getOpcode());

    if (DI.Mnemonic == "s_endpgm")
      return false;

    if (Desc.isBranch() || Desc.isReturn() || DI.Mnemonic == "s_setpc_b64")
      return true;

    unsigned NumDefs = Desc.getNumDefs();
    bool FoundUse = false;
    bool FoundDef = false;

    for (unsigned OpI = 0; OpI < Inst.getNumOperands(); ++OpI) {
      const auto &Op = Inst.getOperand(OpI);
      if (!Op.isReg() || Op.getReg() == 0)
        continue;
      if (!MRI.regsOverlap(Op.getReg(), SgprMCReg))
        continue;

      if (OpI < NumDefs)
        FoundDef = true;
      else
        FoundUse = true;
    }

    if (FoundUse)
      return true;
    if (FoundDef && !FoundUse)
      return false;
  }

  return true;
}

// -- parseSgprNum -------------------------------------------------------------
//
// Parse an SGPR register name like "s4" and return its number (4).
// Returns -1 on failure.

static int parseSgprNum(const std::string &SregName) {
  if (SregName.size() < 2 || SregName[0] != 's')
    return -1;
  int Num = -1;
  auto Result = std::from_chars(
      SregName.data() + 1, SregName.data() + SregName.size(), Num);
  return (Result.ec == std::errc()) ? Num : -1;
}

// -- allocScratchVgpr -----------------------------------------------------------
//
// Allocate a scratch VGPR for temporary use at instruction Idx.
// Returns the VGPR number, or -1 if none available.
// Updates Ctx.KernelStats with any extra VGPR allocation.

static int allocScratchVgpr(PatchContext &Ctx, size_t Idx) {
  auto &DI = Ctx.Decoded[Idx];
  std::string KernelName = Ctx.Elf.findKernelAtOffset(DI.Offset);
  unsigned KdVgprs = 0;
  if (auto Opt =
          Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize))
    KdVgprs = *Opt;

  ScratchAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdVgprs,
                         Ctx.Config.MaxVgprs);
  auto ScratchOpt = Alloc.alloc();
  if (!ScratchOpt)
    return -1;

  if (Alloc.extraVgprsNeeded() > 0 && !KernelName.empty()) {
    auto &Stats = Ctx.KernelStats[KernelName];
    Stats.ExtraVgprs =
        std::max(Stats.ExtraVgprs, Alloc.extraVgprsNeeded());
    Stats.ScratchAboveKd += Alloc.extraVgprsNeeded();
  }

  return static_cast<int>(*ScratchOpt);
}

// -- assembleOrFail -----------------------------------------------------------
//
// Assemble a single instruction and return its bytes. If assembly fails,
// log an error and return an empty vector.

static llvm::SmallVector<uint8_t, 16>
assembleOrFail(const std::string &AsmStr, const LLVMState &LS,
               const char *Context) {
  auto Bytes = assembleSingleInst(AsmStr, LS);
  if (Bytes.empty())
    log() << "hotswap: " << Context << ": assembly failed: " << AsmStr << "\n";
  return Bytes;
}

// -- patchDs2AddrStride64 -----------------------------------------------------
//
// Expand one ds_*_2addr_stride64_* instruction into two single-address DS
// instructions. The split doubles the outstanding DS operation count, so
// bumpNextWaitDscnt adjusts the next s_wait_dscnt accordingly.

static uint32_t patchDs2AddrStride64(PatchContext &Ctx, size_t Idx) {
  auto &DI = Ctx.Decoded[Idx];
  auto [From, To] = lookupDs2AddrSwap(DI.Mnemonic);
  if (From.empty())
    return 0;

  std::string Printed = printInst(DI, Ctx.LS);
  auto Expanded = expandDs2AddrAsm(Printed, From.str(), To.str());
  if (Expanded.empty()) {
    log() << "hotswap: error: ds_2addr_stride64 expansion failed for: "
          << Printed << "\n";
    return 0;
  }

  std::string Combined;
  for (auto &Line : Expanded)
    Combined += Line + "\n";
  auto Bytes = assembleOrFail(Combined, Ctx.LS, "ds_2addr_stride64");
  if (Bytes.empty())
    return 0;

  std::vector<uint8_t> Replacement(Bytes.begin(), Bytes.end());
  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return 0;

  bumpNextWaitDscnt(Ctx, Idx);
  DI.Mnemonic = "<replaced>";
  return 1;
}

// -- patchTensorLoadToLds -----------------------------------------------------
//
// Prepend s_pack_hh_b32_b16 to clear multicast routing bits in the group
// descriptor's base SGPR. If the SGPR is live after the tensor_load, bracket
// the sequence with v_writelane/v_readlane to save and restore its value
// through a scratch VGPR lane.

static uint32_t patchTensorLoadToLds(PatchContext &Ctx, size_t Idx) {
  auto &DI = Ctx.Decoded[Idx];

  if (Idx > 0) {
    llvm::StringRef Prev = Ctx.Decoded[Idx - 1].Mnemonic;
    if (Prev == "s_pack_hh_b32_b16" || Prev == "v_writelane_b32")
      return 0;
  }

  std::string BaseSreg = extractDescriptorBaseReg(DI, Ctx.LS);
  if (BaseSreg.empty()) {
    log() << "hotswap: error: tensor_load_to_lds: could not extract descriptor "
             "base register\n";
    return 0;
  }

  auto PackBytes = assembleOrFail(
      "s_pack_hh_b32_b16 " + BaseSreg + ", 0, " + BaseSreg, Ctx.LS,
      "tensor_load_to_lds pack");
  if (PackBytes.empty())
    return 0;

  int SgprNum = parseSgprNum(BaseSreg);
  unsigned SgprMCReg =
      (SgprNum >= 0) ? lookupSgprMCReg(SgprNum, *Ctx.LS.MRI) : 0;
  bool SgprLive =
      (SgprMCReg == 0) || isSgprLiveAfter(Ctx, Idx, SgprMCReg);

  const uint8_t *OrigInst = Ctx.Text + DI.Offset;

  if (SgprLive) {
    int ScratchVgpr = allocScratchVgpr(Ctx, Idx);
    if (ScratchVgpr < 0) {
      log() << "hotswap: error: tensor_load_to_lds: no scratch VGPR "
               "available\n";
      return 0;
    }

    std::string V = "v" + std::to_string(ScratchVgpr);
    auto Save = assembleOrFail(
        "v_writelane_b32 " + V + ", " + BaseSreg + ", 0", Ctx.LS,
        "tensor_load_to_lds save");
    auto Restore = assembleOrFail(
        "v_readlane_b32 " + BaseSreg + ", " + V + ", 0", Ctx.LS,
        "tensor_load_to_lds restore");
    if (Save.empty() || Restore.empty())
      return 0;

    std::vector<uint8_t> Replacement;
    Replacement.insert(Replacement.end(), Save.begin(), Save.end());
    Replacement.insert(Replacement.end(), PackBytes.begin(), PackBytes.end());
    Replacement.insert(Replacement.end(), OrigInst, OrigInst + DI.Size);
    Replacement.insert(Replacement.end(), Restore.begin(), Restore.end());

    if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
      return 0;

    log() << "hotswap: tensor_load_to_lds: " << BaseSreg
          << " live, save/restore via " << V << "\n";
  } else {
    std::vector<uint8_t> Replacement;
    Replacement.insert(Replacement.end(), PackBytes.begin(), PackBytes.end());
    Replacement.insert(Replacement.end(), OrigInst, OrigInst + DI.Size);

    if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
      return 0;

    log() << "hotswap: tensor_load_to_lds: " << BaseSreg
          << " dead, no save/restore needed\n";
  }

  DI.Mnemonic = "<replaced>";
  return 1;
}

// -- applyTrampolinePatches ---------------------------------------------------
//
// Strong-symbol override. Handles two B0 errata that produce replacement code
// larger than the original instruction slot:
//
//   ds_*_2addr_stride64_*  → split into two single-address DS ops
//   tensor_load_to_lds     → prepend s_pack_hh (+ save/restore if SGPR live)

uint32_t applyTrampolinePatches(PatchContext &Ctx, size_t Idx) {
  llvm::StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (isDs2AddrStride64(Mnem))
    return patchDs2AddrStride64(Ctx, Idx);

  if (Mnem == "tensor_load_to_lds")
    return patchTensorLoadToLds(Ctx, Idx);

  return 0;
}

} // namespace hotswap
} // namespace COMGR
