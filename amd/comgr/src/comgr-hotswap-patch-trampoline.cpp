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

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// -- MC-layer register helpers ------------------------------------------------
//
// MCRegisterInfo::getName() returns internal LLVM names (e.g. "VGPR0",
// "SGPR4"). These stable TableGen identifiers are converted to assembly
// syntax ("v0", "s4") for instruction building. MCSubRegIterator returns
// ALL sub-registers including nested lo16/hi16 fragments; we filter to
// keep only the direct 32-bit components.

static std::string toAsmRegName(const MCRegisterInfo &MRI, MCRegister Reg) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return {};
  StringRef Name(N);
  if (Name.starts_with("VGPR"))
    return ("v" + Name.drop_front(4)).str();
  if (Name.starts_with("SGPR"))
    return ("s" + Name.drop_front(4)).str();
  return Name.str();
}

static SmallVector<MCRegister, 2>
getDirectSubRegs(MCRegister Reg, const MCRegisterInfo &MRI) {
  SmallVector<MCRegister, 2> Result;
  for (MCPhysReg Sub : MRI.subregs(Reg)) {
    StringRef Name = MRI.getName(Sub);
    if ((Name.starts_with("VGPR") || Name.starts_with("SGPR")) &&
        !Name.contains("LO") && !Name.contains("HI"))
      Result.push_back(MCRegister(Sub));
  }
  return Result;
}

// -- DS stride64 swap table (StringMap) ---------------------------------------

static const StringMap<StringRef> &getDs2AddrSwapMap() {
  static const StringMap<StringRef> Map({
      {"ds_load_2addr_stride64_b32", "ds_load_b32"},
      {"ds_load_2addr_stride64_b64", "ds_load_b64"},
      {"ds_store_2addr_stride64_b32", "ds_store_b32"},
      {"ds_store_2addr_stride64_b64", "ds_store_b64"},
      {"ds_storexchg_2addr_stride64_rtn_b32", "ds_storexchg_rtn_b32"},
      {"ds_storexchg_2addr_stride64_rtn_b64", "ds_storexchg_rtn_b64"},
  });
  return Map;
}

// -- expandDs2Addr (MC-layer) -------------------------------------------------
//
// Reads operands directly from the decoded MCInst to build two single-address
// DS assembly strings. DS_READ2ST64/DS_WRITE2ST64 operand layout (TableGen
// DS_1A_Off8_RET / DS_1A1D_Off8_NORET / DS_1A2D_Off8_NORET):
//   Op 0: $vdst (64b pair, load/xchg) or $addr (store)
//   Op 1: $addr (load/xchg) or $data0 (store) or $vdst (xchg)
//   Op 2: $offset0 (8b imm) or $data0/$data1
//   Op 3: $offset1 (8b imm) or ...
//
// The operand order varies across load/store/xchg, but the key insight is that
// register operands come first and the two 8-bit offset immediates always
// follow. We scan the operand list for them.

static std::vector<std::string>
expandDs2Addr(const MCInst &Inst, StringRef FromMnem, StringRef ToMnem,
              const LLVMState &LS) {
  const MCRegisterInfo &MRI = *LS.MRI;

  // Collect register operands and locate the two offset immediates.
  SmallVector<MCRegister, 4> Regs;
  int64_t Off0 = 0, Off1 = 0;
  unsigned ImmsSeen = 0;
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() != 0)
      Regs.push_back(MCRegister(Op.getReg()));
    else if (Op.isImm()) {
      if (ImmsSeen == 0)
        Off0 = Op.getImm();
      else if (ImmsSeen == 1)
        Off1 = Op.getImm();
      ++ImmsSeen;
    }
  }

  uint32_t ElemBytes = FromMnem.contains("_b64") ? 8 : 4;
  uint32_t Scale = 64 * ElemBytes;
  uint32_t ScaledOff0 = static_cast<uint32_t>(Off0) * Scale;
  uint32_t ScaledOff1 = static_cast<uint32_t>(Off1) * Scale;

  auto FmtOff = [](uint32_t V) -> std::string {
    return V ? " offset:" + std::to_string(V) : "";
  };

  bool IsLoad = FromMnem.starts_with("ds_load");
  bool IsXchg = FromMnem.starts_with("ds_storexchg");

  if (IsLoad && Regs.size() >= 2) {
    // Load: Inst = ds_load_2addr_stride64 vdst_pair, addr
    auto Subs = getDirectSubRegs(Regs[0], MRI);
    if (Subs.size() < 2)
      return {};
    std::string D0 = toAsmRegName(MRI, Subs[0]);
    std::string D1 = toAsmRegName(MRI, Subs[1]);
    std::string Addr = toAsmRegName(MRI, Regs[1]);
    return {
        ToMnem.str() + " " + D0 + ", " + Addr + FmtOff(ScaledOff0),
        ToMnem.str() + " " + D1 + ", " + Addr + FmtOff(ScaledOff1),
    };
  }

  if (IsXchg && Regs.size() >= 4) {
    // Xchg: Inst = ds_storexchg_2addr_stride64_rtn vdst_pair, addr, data0, data1
    auto Subs = getDirectSubRegs(Regs[0], MRI);
    if (Subs.size() < 2)
      return {};
    std::string D0 = toAsmRegName(MRI, Subs[0]);
    std::string D1 = toAsmRegName(MRI, Subs[1]);
    std::string Addr = toAsmRegName(MRI, Regs[1]);
    std::string Data0 = toAsmRegName(MRI, Regs[2]);
    std::string Data1 = toAsmRegName(MRI, Regs[3]);
    return {
        ToMnem.str() + " " + D0 + ", " + Addr + ", " + Data0 +
            FmtOff(ScaledOff0),
        ToMnem.str() + " " + D1 + ", " + Addr + ", " + Data1 +
            FmtOff(ScaledOff1),
    };
  }

  // Store: Inst = ds_store_2addr_stride64 addr, data0, data1
  if (Regs.size() >= 3) {
    std::string Addr = toAsmRegName(MRI, Regs[0]);
    std::string Data0 = toAsmRegName(MRI, Regs[1]);
    std::string Data1 = toAsmRegName(MRI, Regs[2]);
    return {
        ToMnem.str() + " " + Addr + ", " + Data0 + FmtOff(ScaledOff0),
        ToMnem.str() + " " + Addr + ", " + Data1 + FmtOff(ScaledOff1),
    };
  }

  return {};
}

// -- bumpNextWaitDscnt --------------------------------------------------------
//
// After splitting one DS 2-addr instruction into two, the next s_wait_dscnt
// in the stream must be incremented by 1 to account for the extra outstanding
// DS operation. Uses the already-decoded MCInst to read the immediate, then
// re-encodes the modified instruction via MCCodeEmitter.

static void bumpNextWaitDscnt(PatchContext &Ctx, size_t Idx) {
  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    if (Ctx.Decoded[I].Mnemonic != "s_wait_dscnt")
      continue;

    MCInst NewInst = Ctx.Decoded[I].Inst;
    for (unsigned OpI = 0, OpE = NewInst.getNumOperands(); OpI < OpE; ++OpI) {
      MCOperand &Op = NewInst.getOperand(OpI);
      if (!Op.isImm())
        continue;
      Op.setImm(Op.getImm() + 1);
      break;
    }

    SmallVector<char, 8> Bytes;
    SmallVector<MCFixup, 2> Fixups;
    Ctx.LS.MCE->encodeInstruction(NewInst, Bytes, Fixups, *Ctx.LS.STI);

    uint64_t Off = Ctx.Decoded[I].Offset;
    std::memcpy(Ctx.Text + Off, Bytes.data(), Bytes.size());
    return;
  }
}

// -- getDescriptorBaseSgpr ----------------------------------------------------
//
// Extract the base SGPR MCRegister from the second operand of a
// tensor_load_to_lds instruction. The second operand is an 8-SGPR group
// descriptor (SReg_256); we need its first sub-register for the
// s_pack_hh_b32_b16 fix.

static MCRegister getDescriptorBaseSgpr(const MCInst &Inst,
                                        const MCRegisterInfo &MRI) {
  if (Inst.getNumOperands() < 2 || !Inst.getOperand(1).isReg())
    return MCRegister();
  MCRegister Tuple = MCRegister(Inst.getOperand(1).getReg());
  auto Subs = getDirectSubRegs(Tuple, MRI);
  return Subs.empty() ? MCRegister() : Subs[0];
}

// -- isSgprLiveAfter ----------------------------------------------------------
//
// Conservative forward-scan heuristic. Returns true if the given SGPR
// (identified by its MCRegister) is used before being redefined in the
// instruction stream following Idx. Conservatively returns true on
// control-flow-affecting instructions or end of stream.

bool isSgprLiveAfter(const PatchContext &Ctx, size_t Idx, unsigned SgprMCReg) {
  if (SgprMCReg == 0)
    return true;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const MCInstrInfo &MCII = *Ctx.LS.MCII;

  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    const auto &DI = Ctx.Decoded[I];
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>")
      continue;

    const MCInst &Inst = DI.Inst;
    const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());

    if (DI.Mnemonic == "s_endpgm")
      return false;

    if (Desc.mayAffectControlFlow(Inst, MRI))
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

// -- allocScratchVgpr -----------------------------------------------------------

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

static SmallVector<uint8_t, 16>
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

static bool patchDs2AddrStride64(PatchContext &Ctx, size_t Idx) {
  auto &DI = Ctx.Decoded[Idx];
  const auto &Map = getDs2AddrSwapMap();
  auto It = Map.find(DI.Mnemonic);
  if (It == Map.end())
    return false;

  StringRef ToMnem = It->second;
  auto Expanded = expandDs2Addr(DI.Inst, DI.Mnemonic, ToMnem, Ctx.LS);
  if (Expanded.empty()) {
    log() << "hotswap: error: ds_2addr_stride64 expansion failed for: "
          << DI.Mnemonic << "\n";
    return false;
  }

  std::string Combined;
  for (auto &Line : Expanded)
    Combined += Line + "\n";
  auto Bytes = assembleOrFail(Combined, Ctx.LS, "ds_2addr_stride64");
  if (Bytes.empty())
    return false;

  std::vector<uint8_t> Replacement(Bytes.begin(), Bytes.end());
  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return false;

  bumpNextWaitDscnt(Ctx, Idx);
  DI.Mnemonic = "<replaced>";
  return true;
}

// -- patchTensorLoadToLds -----------------------------------------------------
//
// Prepend s_pack_hh_b32_b16 to clear multicast routing bits in the group
// descriptor's base SGPR. If the SGPR is live after the tensor_load, bracket
// the sequence with v_writelane/v_readlane to save and restore its value
// through a scratch VGPR lane.

static bool patchTensorLoadToLds(PatchContext &Ctx, size_t Idx) {
  auto &DI = Ctx.Decoded[Idx];
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  MCRegister BaseMCReg = getDescriptorBaseSgpr(DI.Inst, MRI);
  if (!BaseMCReg.isValid()) {
    log() << "hotswap: error: tensor_load_to_lds: could not extract descriptor "
             "base register\n";
    return false;
  }

  // Idempotency guard: verify the previous instruction is part of an earlier
  // patch for the *same* descriptor SGPR, not just any s_pack_hh / v_writelane.
  if (Idx > 0) {
    const auto &Prev = Ctx.Decoded[Idx - 1];
    if (Prev.Mnemonic == "s_pack_hh_b32_b16" ||
        Prev.Mnemonic == "v_writelane_b32") {
      for (unsigned OpI = 0; OpI < Prev.Inst.getNumOperands(); ++OpI) {
        const MCOperand &Op = Prev.Inst.getOperand(OpI);
        if (Op.isReg() && MRI.regsOverlap(Op.getReg(), BaseMCReg))
          return false;
      }
    }
  }

  std::string BaseSreg = toAsmRegName(MRI, BaseMCReg);

  auto PackBytes = assembleOrFail(
      "s_pack_hh_b32_b16 " + BaseSreg + ", 0, " + BaseSreg, Ctx.LS,
      "tensor_load_to_lds pack");
  if (PackBytes.empty())
    return false;

  bool SgprLive = isSgprLiveAfter(Ctx, Idx, BaseMCReg.id());

  const uint8_t *OrigInst = Ctx.Text + DI.Offset;

  if (SgprLive) {
    int ScratchVgpr = allocScratchVgpr(Ctx, Idx);
    if (ScratchVgpr < 0) {
      log() << "hotswap: error: tensor_load_to_lds: no scratch VGPR "
               "available\n";
      return false;
    }

    ScratchPatchInfo SPI;
    SPI.Offset = DI.Offset;
    SPI.ScratchRegs.resize(Ctx.Config.MaxVgprs);
    SPI.ScratchRegs.set(ScratchVgpr);
    Ctx.OutScratchPatches.push_back(std::move(SPI));

    std::string V = "v" + std::to_string(ScratchVgpr);
    auto Save = assembleOrFail(
        "v_writelane_b32 " + V + ", " + BaseSreg + ", 0", Ctx.LS,
        "tensor_load_to_lds save");
    auto Restore = assembleOrFail(
        "v_readlane_b32 " + BaseSreg + ", " + V + ", 0", Ctx.LS,
        "tensor_load_to_lds restore");
    if (Save.empty() || Restore.empty())
      return false;

    std::vector<uint8_t> Replacement;
    Replacement.insert(Replacement.end(), Save.begin(), Save.end());
    Replacement.insert(Replacement.end(), PackBytes.begin(), PackBytes.end());
    Replacement.insert(Replacement.end(), OrigInst, OrigInst + DI.Size);
    Replacement.insert(Replacement.end(), Restore.begin(), Restore.end());

    if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
      return false;

    log() << "hotswap: tensor_load_to_lds: " << BaseSreg
          << " live, save/restore via " << V << "\n";
  } else {
    std::vector<uint8_t> Replacement;
    Replacement.insert(Replacement.end(), PackBytes.begin(), PackBytes.end());
    Replacement.insert(Replacement.end(), OrigInst, OrigInst + DI.Size);

    if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
      return false;

    log() << "hotswap: tensor_load_to_lds: " << BaseSreg
          << " dead, no save/restore needed\n";
  }

  DI.Mnemonic = "<replaced>";
  return true;
}

// -- applyTrampolinePatches ---------------------------------------------------
//
// Strong-symbol override. Handles two B0 errata that produce replacement code
// larger than the original instruction slot:
//
//   ds_*_2addr_stride64_*  -> split into two single-address DS ops
//   tensor_load_to_lds     -> prepend s_pack_hh (+ save/restore if SGPR live)

uint32_t applyTrampolinePatches(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (getDs2AddrSwapMap().count(Mnem))
    return patchDs2AddrStride64(Ctx, Idx) ? 1 : 0;

  if (Mnem == "tensor_load_to_lds")
    return patchTensorLoadToLds(Ctx, Idx) ? 1 : 0;

  return 0;
}

} // namespace hotswap
} // namespace COMGR
