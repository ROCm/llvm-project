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
///   - ds_*_2addr_stride64_*  : one 8B DS instruction -> two single-address
///     DS instructions
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

// MSVC does not support weak symbols; LLVM_ATTRIBUTE_WEAK expands to nothing,
// so the stub in comgr-hotswap-b0a0.cpp becomes a regular definition and
// this file would produce a duplicate-symbol link error (LNK2005). Guard
// the strong override until a proper registration mechanism replaces the
// weak-symbol pattern on Windows (tracked in #2294 / #2285).
#if !defined(_MSC_VER)

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// -- MC-layer register helpers ----------------------------------------------
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

static SmallVector<MCRegister, 2> getDirectSubRegs(MCRegister Reg,
                                                   const MCRegisterInfo &MRI) {
  SmallVector<MCRegister, 2> Result;
  for (MCPhysReg Sub : MRI.subregs(Reg)) {
    StringRef Name = MRI.getName(Sub);
    if ((Name.starts_with("VGPR") || Name.starts_with("SGPR")) &&
        !Name.contains("LO") && !Name.contains("HI"))
      Result.push_back(MCRegister(Sub));
  }
  return Result;
}

// -- DS stride64 swap table (StringMap) -------------------------------------

static StringRef getDs2AddrReplacement(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("ds_load_2addr_stride64_b32", "ds_load_b32")
      .Case("ds_load_2addr_stride64_b64", "ds_load_b64")
      .Case("ds_store_2addr_stride64_b32", "ds_store_b32")
      .Case("ds_store_2addr_stride64_b64", "ds_store_b64")
      .Case("ds_storexchg_2addr_stride64_rtn_b32", "ds_storexchg_rtn_b32")
      .Case("ds_storexchg_2addr_stride64_rtn_b64", "ds_storexchg_rtn_b64")
      .Default("");
}

// -- expandDs2Addr (MC-layer) -----------------------------------------------
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

static std::vector<std::string> expandDs2Addr(const MCInst &Inst,
                                              StringRef FromMnem,
                                              StringRef ToMnem,
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
    // Xchg: Inst = ds_storexchg_2addr_stride64_rtn vdst_pair, addr, data0,
    // data1
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

// -- bumpNextWaitDscnt ------------------------------------------------------
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

    Ctx.Decoded[I].Inst = NewInst;
    return;
  }
}

// -- assembleOrFail ---------------------------------------------------------

static SmallVector<uint8_t> assembleOrFail(const std::string &AsmStr,
                                           const LLVMState &LS,
                                           const char *Context) {
  auto Bytes = assembleSingleInst(AsmStr, LS);
  if (Bytes.empty())
    log() << "hotswap: " << Context << ": assembly failed: " << AsmStr << "\n";
  return Bytes;
}

// -- patchDs2AddrStride64 ---------------------------------------------------
//
// Expand one ds_*_2addr_stride64_* instruction into two single-address DS
// instructions. The split doubles the outstanding DS operation count, so
// bumpNextWaitDscnt adjusts the next s_wait_dscnt accordingly.

static bool patchDs2AddrStride64(PatchContext &Ctx, size_t Idx) {
  auto &DI = Ctx.Decoded[Idx];
  StringRef ToMnem = getDs2AddrReplacement(DI.Mnemonic);
  if (ToMnem.empty())
    return false;
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

// -- applyTrampolinePatches -------------------------------------------------
//
// Strong-symbol override. Handles B0 errata that produce replacement code
// larger than the original instruction slot:
//
//   ds_*_2addr_stride64_*  -> split into two single-address DS ops

uint32_t applyTrampolinePatches(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (!getDs2AddrReplacement(Mnem).empty())
    return patchDs2AddrStride64(Ctx, Idx) ? 1 : 0;

  return 0;
}

} // namespace hotswap
} // namespace COMGR

#endif // !defined(_MSC_VER)
