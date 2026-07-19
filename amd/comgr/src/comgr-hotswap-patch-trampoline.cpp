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
///   - ds_*_2addr_*           : one 8B DS instruction -> two single-address
///     DS instructions. Covers both the stride64 and non-stride64 encodings:
///     A0 requires DS2 addresses to be aligned to the payload size, while
///     B0 dropped that restriction, so a B0-compiled binary may emit a
///     2-address DS instruction with unaligned offsets that silently
///     corrupts LDS on A0. The expansion uses two single-address ops with
///     byte offsets scaled appropriately for each encoding.
///   - tensor_load_to_lds     : clear multicast routing bits in the group
///     descriptor's base SGPR. A0 clears unconditionally; B0 clears only when
///     runtime cluster state reports a non-cluster wave.
///   - cluster_load*          : for cluster-load forms that remain cluster
///     loads after in-place demotion on A0, save M0, clear wg_mask bits
///     [15:0], issue the original load, then restore M0
///   - ds_*_addtid_b32        : compute the LDS address through the ALU and
///     issue a regular ds_*_b32, bypassing the gfx1250 A0 16-bit M0
///     truncation. On B0 the DS unit reads 20 bits of M0; on A0 it reads only
///     16, silently dropping bits [19:16].
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace COMGR {
namespace hotswap {

namespace {

bool failRequiredPatch(PatchContext &Ctx) {
  Ctx.RequiredPatchFailed = true;
  return false;
}

// -- DS 2-address swap table (StringSwitch) ---------------------------------
//
// Maps each 2-address DS mnemonic to its single-address replacement. Covers
// both encodings -- the stride64 variants pack the index*64*ElemBytes
// stride into each per-operand offset field, while the non-stride64
// variants encode raw index*ElemBytes byte offsets. The single-address
// replacement is the same regardless of encoding; only the offset scale
// differs (see extractDsOperands).

StringRef getDs2AddrReplacement(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("ds_load_2addr_b32", "ds_load_b32")
      .Case("ds_load_2addr_b64", "ds_load_b64")
      .Case("ds_load_2addr_stride64_b32", "ds_load_b32")
      .Case("ds_load_2addr_stride64_b64", "ds_load_b64")
      .Case("ds_store_2addr_b32", "ds_store_b32")
      .Case("ds_store_2addr_b64", "ds_store_b64")
      .Case("ds_store_2addr_stride64_b32", "ds_store_b32")
      .Case("ds_store_2addr_stride64_b64", "ds_store_b64")
      .Case("ds_storexchg_2addr_rtn_b32", "ds_storexchg_rtn_b32")
      .Case("ds_storexchg_2addr_rtn_b64", "ds_storexchg_rtn_b64")
      .Case("ds_storexchg_2addr_stride64_rtn_b32", "ds_storexchg_rtn_b32")
      .Case("ds_storexchg_2addr_stride64_rtn_b64", "ds_storexchg_rtn_b64")
      .Default("");
}

// -- MC-layer register helpers ----------------------------------------------
//
// MCRegisterInfo::getName() returns internal LLVM names (e.g. "VGPR0",
// "SGPR4"). We convert these to assembly syntax ("v0", "s4") for instruction
// building. Sub-register iteration returns ALL fragments (including lo16/hi16);
// getDirectSubRegs filters to only scalar 32-bit components.

std::string toAsmRegName(const MCRegisterInfo &MRI, MCRegister Reg) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return {};
  StringRef Name(N);
  if (Name.starts_with("VGPR") && !Name.contains('_'))
    return ("v" + Name.drop_front(4)).str();
  if (Name.starts_with("SGPR") && !Name.contains('_'))
    return ("s" + Name.drop_front(4)).str();
  return Name.str();
}

bool isM0Reg(MCRegister Reg, const MCRegisterInfo &MRI) {
  const char *N = MRI.getName(Reg);
  return N && StringRef(N).starts_with("M0");
}

SmallVector<MCRegister, 4> getDirectSubRegs(MCRegister Reg,
                                            const MCRegisterInfo &MRI) {
  SmallVector<MCRegister, 4> Result;
  for (MCPhysReg Sub : MRI.subregs(Reg)) {
    StringRef Name = MRI.getName(Sub);
    if ((Name.starts_with("VGPR") || Name.starts_with("SGPR")) &&
        !Name.contains("LO") && !Name.contains("HI") && !Name.contains('_'))
      Result.push_back(MCRegister(Sub));
  }
  return Result;
}

// Format a VGPR pair as a range expression: (VGPR0, VGPR1) -> "v[0:1]".
std::string fmtRegPair(const MCRegisterInfo &MRI, MCRegister Lo,
                       MCRegister Hi) {
  std::string LoName = toAsmRegName(MRI, Lo);
  std::string HiName = toAsmRegName(MRI, Hi);
  char Prefix = LoName[0];
  StringRef LoIdx = StringRef(LoName).drop_front(1);
  StringRef HiIdx = StringRef(HiName).drop_front(1);
  return std::string(1, Prefix) + "[" + LoIdx.str() + ":" + HiIdx.str() + "]";
}

// Format a register operand for assembly. Single registers (VGPR0) produce
// "v0"; register tuples (VGPR0_VGPR1) produce "v[0:1]" by decomposing into
// their scalar sub-registers.
std::string fmtRegOperand(const MCRegisterInfo &MRI, MCRegister Reg) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return {};
  StringRef Name(N);
  if (!Name.contains('_'))
    return toAsmRegName(MRI, Reg);
  SmallVector<MCRegister, 4> Subs = getDirectSubRegs(Reg, MRI);
  if (Subs.size() < 2)
    return toAsmRegName(MRI, Reg);
  return fmtRegPair(MRI, Subs.front(), Subs.back());
}

// Format an optional byte offset as " offset:N" (empty string when zero).
std::string fmtOffset(uint32_t Offset) {
  return Offset ? " offset:" + std::to_string(Offset) : "";
}

// -- DS expansion -----------------------------------------------------------
//
// Expands one DS 2-address instruction into two single-address assembly
// strings. The three operation types have different operand layouts (the
// stride64 and non-stride64 encodings share identical operand layouts;
// only the offset scale differs):
//   Load:  ds_load_2addr[_stride64]  vdst_pair, addr, off0, off1
//   Store: ds_store_2addr[_stride64] addr, data0, data1, off0, off1
//   Xchg:  ds_storexchg_2addr[_stride64]_rtn vdst_pair, addr, data0, data1, ...
//
// For b32 operations, destinations are split into individual VGPRs.
// For b64 operations, destinations are split into VGPR pairs (v[X:Y]).

// Maximum byte offset encodable in a single-address DS instruction's
// 16-bit immediate offset field on gfx1250. The replacement we emit uses
// this field directly, so any scaled byte offset that exceeds it cannot
// be represented and the patch must be skipped.
constexpr uint32_t Ds1AddrOffsetMax = 0xFFFF;

struct DsOperands {
  SmallVector<MCRegister, 4> Regs;
  uint32_t Off0 = 0;
  uint32_t Off1 = 0;
  bool IsB64 = false;
  const MCRegisterInfo *MRI = nullptr;
};

// Extract register operands and scaled offsets from a DS 2-address MCInst.
// The per-operand immediate fields hold dword indices that the hardware
// scales differently for the two encodings: the non-stride64 forms encode
// (index * ElemBytes) byte offsets, while the stride64 forms encode
// (index * 64 * ElemBytes) byte offsets. The replacement single-address
// instructions take byte offsets directly, so we materialise the scaled
// value here once and let the layout-specific helpers consume it.
//
// Range check: the stride64 b64 encoding can scale a raw 8-bit index up to
// 255 * 64 * 8 = 130560 bytes, which overflows the single-address 16-bit
// offset field (max 0xFFFF = 65535). When that happens the patch is not
// representable in this expansion shape; std::nullopt signals the failure
// to the caller, which leaves the original (broken-on-A0) instruction in
// place rather than emitting a silently-truncated replacement.
std::optional<DsOperands>
extractDsOperands(const MCInst &Inst, StringRef FromMnem, const LLVMState &LS) {
  DsOperands Ops;
  Ops.MRI = LS.MRI.get();

  int64_t RawOff0 = 0, RawOff1 = 0;
  unsigned ImmsSeen = 0;
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg())
      Ops.Regs.push_back(MCRegister(Op.getReg()));
    else if (Op.isImm()) {
      if (ImmsSeen == 0)
        RawOff0 = Op.getImm();
      else if (ImmsSeen == 1)
        RawOff1 = Op.getImm();
      ++ImmsSeen;
    }
  }

  uint32_t ElemBytes = FromMnem.contains("_b64") ? 8 : 4;
  uint32_t Scale = FromMnem.contains("_stride64_") ? 64 * ElemBytes : ElemBytes;
  // Compute scaled offsets in 64-bit so an oversize stride64_b64 index
  // does not silently wrap when assigned to Off*.
  uint64_t Scaled0 = static_cast<uint64_t>(RawOff0) * Scale;
  uint64_t Scaled1 = static_cast<uint64_t>(RawOff1) * Scale;
  if (Scaled0 > Ds1AddrOffsetMax || Scaled1 > Ds1AddrOffsetMax) {
    log() << "hotswap: error: " << FromMnem
          << " scaled offsets exceed the single-address DS 16-bit field "
             "(off0=raw "
          << RawOff0 << " * scale " << Scale << " = " << Scaled0
          << ", off1=raw " << RawOff1 << " * scale " << Scale << " = "
          << Scaled1 << ", max " << Ds1AddrOffsetMax
          << "); required A0 rewrite cannot continue\n";
    return std::nullopt;
  }
  Ops.Off0 = static_cast<uint32_t>(Scaled0);
  Ops.Off1 = static_cast<uint32_t>(Scaled1);
  Ops.IsB64 = (ElemBytes == 8);
  return Ops;
}

// Split a compound destination register into two formatted destination strings.
// b32: VReg_64 -> ("v0", "v1"); b64: VReg_128 -> ("v[0:1]", "v[2:3]")
std::pair<std::string, std::string>
splitDstPair(MCRegister CompoundReg, bool IsB64, const MCRegisterInfo &MRI) {
  SmallVector<MCRegister, 4> Subs = getDirectSubRegs(CompoundReg, MRI);
  if (IsB64) {
    if (Subs.size() < 4)
      return {};
    return {fmtRegPair(MRI, Subs[0], Subs[1]),
            fmtRegPair(MRI, Subs[2], Subs[3])};
  }
  if (Subs.size() < 2)
    return {};
  return {toAsmRegName(MRI, Subs[0]), toAsmRegName(MRI, Subs[1])};
}

// Expand a DS 2-address load into two single-address loads (dst, addr).
std::vector<std::string> expandDs2AddrLoad(const DsOperands &Ops,
                                           StringRef ToMnem) {
  if (Ops.Regs.size() < 2)
    return {};
  std::pair<std::string, std::string> Dst =
      splitDstPair(Ops.Regs[0], Ops.IsB64, *Ops.MRI);
  if (Dst.first.empty())
    return {};
  std::string Addr = toAsmRegName(*Ops.MRI, Ops.Regs[1]);
  std::string First =
      ToMnem.str() + " " + Dst.first + ", " + Addr + fmtOffset(Ops.Off0);
  std::string Second =
      ToMnem.str() + " " + Dst.second + ", " + Addr + fmtOffset(Ops.Off1);

  // A compound DS load reads its address once before writing either half of
  // the destination. After splitting, the first single-address load must not
  // overwrite the address needed by the second. If the address overlaps the
  // first destination half, issue the independent second half first and put
  // the self-overlapping load last. (If it overlaps the second half, the
  // natural order is already safe.)
  SmallVector<MCRegister, 4> DstSubs = getDirectSubRegs(Ops.Regs[0], *Ops.MRI);
  const unsigned FirstHalfWidth = Ops.IsB64 ? 2 : 1;
  bool AddrOverlapsFirst = llvm::any_of(
      ArrayRef(DstSubs).take_front(FirstHalfWidth),
      [&](MCRegister Reg) { return Ops.MRI->regsOverlap(Reg, Ops.Regs[1]); });
  if (AddrOverlapsFirst)
    return {std::move(Second), std::move(First)};
  return {std::move(First), std::move(Second)};
}

// Expand a DS 2-address store into two single-address stores (addr, data).
std::vector<std::string> expandDs2AddrStore(const DsOperands &Ops,
                                            StringRef ToMnem) {
  if (Ops.Regs.size() < 3)
    return {};
  const MCRegisterInfo &MRI = *Ops.MRI;
  std::string Addr = toAsmRegName(MRI, Ops.Regs[0]);
  std::string Data0 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[1])
                                : toAsmRegName(MRI, Ops.Regs[1]);
  std::string Data1 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[2])
                                : toAsmRegName(MRI, Ops.Regs[2]);
  return {
      ToMnem.str() + " " + Addr + ", " + Data0 + fmtOffset(Ops.Off0),
      ToMnem.str() + " " + Addr + ", " + Data1 + fmtOffset(Ops.Off1),
  };
}

// Expand a DS 2-address exchange into two single-address exchanges
// (dst, addr, data).
bool halfOverlaps(ArrayRef<MCRegister> DstSubs, unsigned Begin, unsigned Width,
                  MCRegister Reg, const MCRegisterInfo &MRI) {
  return llvm::any_of(DstSubs.slice(Begin, Width), [&](MCRegister DstReg) {
    return MRI.regsOverlap(DstReg, Reg);
  });
}

std::vector<std::string> expandDs2AddrXchg(const DsOperands &Ops,
                                           StringRef ToMnem) {
  if (Ops.Regs.size() < 4)
    return {};
  const MCRegisterInfo &MRI = *Ops.MRI;
  std::pair<std::string, std::string> Dst =
      splitDstPair(Ops.Regs[0], Ops.IsB64, MRI);
  if (Dst.first.empty())
    return {};
  std::string Addr = toAsmRegName(MRI, Ops.Regs[1]);
  std::string Data0 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[2])
                                : toAsmRegName(MRI, Ops.Regs[2]);
  std::string Data1 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[3])
                                : toAsmRegName(MRI, Ops.Regs[3]);
  std::string First = ToMnem.str() + " " + Dst.first + ", " + Addr + ", " +
                      Data0 + fmtOffset(Ops.Off0);
  std::string Second = ToMnem.str() + " " + Dst.second + ", " + Addr + ", " +
                       Data1 + fmtOffset(Ops.Off1);

  SmallVector<MCRegister, 4> DstSubs = getDirectSubRegs(Ops.Regs[0], MRI);
  const unsigned HalfWidth = Ops.IsB64 ? 2 : 1;

  // Op0 writes the first destination half and op1 still needs addr + data1;
  // op1 writes the second half and op0 still needs addr + data0. Pick the safe
  // order when only one direction has a dependency. If both directions do,
  // neither ordering preserves the compound instruction's read-before-write
  // semantics without a scratch VGPR, so decline the rewrite.
  const bool FirstClobbersSecond =
      halfOverlaps(DstSubs, 0, HalfWidth, Ops.Regs[1], MRI) ||
      halfOverlaps(DstSubs, 0, HalfWidth, Ops.Regs[3], MRI);
  const bool SecondClobbersFirst =
      halfOverlaps(DstSubs, HalfWidth, HalfWidth, Ops.Regs[1], MRI) ||
      halfOverlaps(DstSubs, HalfWidth, HalfWidth, Ops.Regs[2], MRI);
  if (FirstClobbersSecond && SecondClobbersFirst) {
    log() << "hotswap: error: ds_storexchg_2addr has cyclic "
             "destination/source overlap and cannot be split without scratch "
             "VGPRs\n";
    return {};
  }
  if (FirstClobbersSecond)
    return {std::move(Second), std::move(First)};
  return {std::move(First), std::move(Second)};
}

// -- expandDs2Addr ----------------------------------------------------------
//
// Top-level expansion: extracts operands from the decoded MCInst, computes
// scaled offsets, then dispatches to the appropriate layout-specific helper.

std::vector<std::string> expandDs2Addr(const MCInst &Inst, StringRef FromMnem,
                                       StringRef ToMnem, const LLVMState &LS) {
  std::optional<DsOperands> Ops = extractDsOperands(Inst, FromMnem, LS);
  if (!Ops)
    return {};

  // Use the trailing underscore so the three prefixes are disjoint
  // ("ds_load_", "ds_store_", "ds_storexchg_"); without it "ds_store" is a
  // prefix of "ds_storexchg" and the dispatch order would matter.
  if (FromMnem.starts_with("ds_load_"))
    return expandDs2AddrLoad(*Ops, ToMnem);
  if (FromMnem.starts_with("ds_storexchg_"))
    return expandDs2AddrXchg(*Ops, ToMnem);
  if (FromMnem.starts_with("ds_store_"))
    return expandDs2AddrStore(*Ops, ToMnem);

  log() << "hotswap: error: unrecognized DS mnemonic: " << FromMnem << "\n";
  return {};
}

bool definesRegister(const InternalDecodedInst &DI, MCRegister Reg,
                     const LLVMState &LS) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != NumDefs; ++I) {
    const MCOperand &Op = DI.Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() && LS.MRI->regsOverlap(Op.getReg(), Reg.id()))
      return true;
  }
  return llvm::any_of(Desc.implicit_defs(), [&](MCPhysReg Implicit) {
    return LS.MRI->regsOverlap(Implicit, Reg.id());
  });
}

bool definesModeOrExec(const InternalDecodedInst &DI, const LLVMState &LS) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  auto IsModeOrExec = [&](MCRegister Reg) {
    StringRef Name = LS.MRI->getName(Reg.id());
    return Name == "MODE" || Name == "EXEC" || Name == "EXEC_LO" ||
           Name == "EXEC_HI";
  };
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != NumDefs; ++I) {
    const MCOperand &Op = DI.Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() && IsModeOrExec(MCRegister(Op.getReg())))
      return true;
  }
  return llvm::any_of(Desc.implicit_defs(), [&](MCPhysReg Implicit) {
    return IsModeOrExec(MCRegister(Implicit));
  });
}

bool isAlignedConstantAddressDef(const InternalDecodedInst &DI,
                                 MCRegister Address, unsigned Alignment,
                                 const PatchContext &Ctx, size_t Idx,
                                 bool RequireEqualMode = true) {
  if (RequireEqualMode && (Idx >= Ctx.VgprMsbDstSrc0EqualBefore.size() ||
                           !Ctx.VgprMsbDstSrc0EqualBefore.test(Idx)))
    return false;
  if (DI.Mnemonic == "v_add_nc_u32" && DI.Inst.getNumOperands() == 4) {
    const MCOperand &Dst = DI.Inst.getOperand(0);
    const MCOperand &Src0 = DI.Inst.getOperand(1);
    const MCOperand &Src1 = DI.Inst.getOperand(2);
    const MCOperand &Modifiers = DI.Inst.getOperand(3);
    if (!Dst.isReg() || Dst.getReg() != Address.id() || !Src0.isImm() ||
        !Src1.isImm() || !Modifiers.isImm() || Modifiers.getImm() != 0)
      return false;
    uint32_t Value = static_cast<uint32_t>(Src0.getImm()) +
                     static_cast<uint32_t>(Src1.getImm());
    return Alignment != 0 && Value % Alignment == 0;
  }

  if (DI.Mnemonic == "v_mov_b32" && DI.Inst.getNumOperands() == 2) {
    const MCOperand &Dst = DI.Inst.getOperand(0);
    if (!Dst.isReg() || Dst.getReg() != Address.id() || Alignment == 0)
      return false;
    std::optional<int64_t> Value = getAbsoluteOperandValue(
        DI.Inst.getOperand(1), DI, ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize));
    return Value && static_cast<uint32_t>(*Value) % Alignment == 0;
  }

  if (DI.Mnemonic == "v_mul_lo_u32" &&
      (DI.Inst.getNumOperands() == 3 || DI.Inst.getNumOperands() == 4)) {
    const MCOperand &Dst = DI.Inst.getOperand(0);
    const MCOperand &Src0 = DI.Inst.getOperand(1);
    const MCOperand &Src1 = DI.Inst.getOperand(2);
    if (!Dst.isReg() || Dst.getReg() != Address.id() || Alignment == 0 ||
        (DI.Inst.getNumOperands() == 4 &&
         (!DI.Inst.getOperand(3).isImm() ||
          DI.Inst.getOperand(3).getImm() != 0)))
      return false;
    return (Src0.isImm() &&
            static_cast<uint32_t>(Src0.getImm()) % Alignment == 0) ||
           (Src1.isImm() &&
            static_cast<uint32_t>(Src1.getImm()) % Alignment == 0);
  }

  if (DI.Mnemonic != "v_dual_mov_b32" || DI.Inst.getNumOperands() != 4)
    return false;
  for (unsigned Def = 0; Def != 2; ++Def) {
    const MCOperand &Dst = DI.Inst.getOperand(Def);
    const MCOperand &Src = DI.Inst.getOperand(Def + 2);
    if (!Dst.isReg() || Dst.getReg() != Address.id())
      continue;
    return Alignment != 0 && Src.isImm() &&
           static_cast<uint32_t>(Src.getImm()) % Alignment == 0;
  }
  return false;
}

static bool isAlignedSgprCopyAddressDef(const InternalDecodedInst &DI,
                                        MCRegister Address, unsigned Alignment,
                                        const PatchContext &Ctx, size_t Idx) {
  if (Alignment == 0 || Alignment > 8 ||
      Idx >= Ctx.VgprMsbDstSrc0EqualBefore.size() ||
      !Ctx.VgprMsbDstSrc0EqualBefore.test(Idx) ||
      Idx >= Ctx.VgprDef0AlignedTo8.size() ||
      Idx >= Ctx.VgprDef1AlignedTo8.size())
    return false;

  if (DI.Mnemonic == "v_mov_b32")
    return DI.Inst.getNumOperands() == 2 && DI.Inst.getOperand(0).isReg() &&
           DI.Inst.getOperand(0).getReg() == Address.id() &&
           Ctx.VgprDef0AlignedTo8.test(Idx);

  if (DI.Mnemonic != "v_dual_mov_b32" || DI.Inst.getNumOperands() != 4)
    return false;
  for (unsigned Def = 0; Def != 2; ++Def) {
    const MCOperand &Dst = DI.Inst.getOperand(Def);
    if (Dst.isReg() && Dst.getReg() == Address.id())
      return (Def == 0 ? Ctx.VgprDef0AlignedTo8 : Ctx.VgprDef1AlignedTo8)
          .test(Idx);
  }
  return false;
}

static bool isAlignedAddressDef(const InternalDecodedInst &DI,
                                MCRegister Address, unsigned Alignment,
                                const PatchContext &Ctx, size_t Idx) {
  return isAlignedConstantAddressDef(DI, Address, Alignment, Ctx, Idx) ||
         isAlignedSgprCopyAddressDef(DI, Address, Alignment, Ctx, Idx);
}

/// A0 only needs the DS2 split when the effective address may be unaligned.
/// Preserve the compact original instruction when a same-block constant or a
/// proven aligned SGPR copy defines the address. Equality of MODE's destination
/// and src0 bank selectors is a must-dataflow fact; unchanged MODE and EXEC
/// then guarantee that the VALU definition and DS address name the same
/// physical VGPRs for the same active lanes.
bool computeDs2AddressProvenAligned(PatchContext &Ctx, size_t Idx) {
  if (Ctx.HasUnknownArbitraryIndirectTarget || !Ctx.DirectControlFlowTargets ||
      Idx >= Ctx.VgprMsbDstSrc0EqualBefore.size())
    return false;
  const InternalDecodedInst &DS = Ctx.Decoded[Idx];
  StringRef Mnem = DS.Mnemonic;
  if (Mnem != "ds_load_2addr_b32" && Mnem != "ds_load_2addr_b64" &&
      Mnem != "ds_store_2addr_b32" && Mnem != "ds_store_2addr_b64")
    return false;
  if (DS.Inst.getNumOperands() == 0 ||
      !DS.Inst.getOperand(DS.Inst.getNumOperands() - 1).isImm() ||
      DS.Inst.getOperand(DS.Inst.getNumOperands() - 1).getImm() != 0)
    return false; // GDS and malformed operand layouts are never exempt.

  std::optional<DsOperands> Ops =
      extractDsOperands(DS.Inst, DS.Mnemonic, Ctx.LS);
  if (!Ops)
    return false;
  unsigned AddrIndex = StringRef(DS.Mnemonic).starts_with("ds_store_") ? 0 : 1;
  if (Ops->Regs.size() <= AddrIndex)
    return false;
  MCRegister Address = Ops->Regs[AddrIndex];
  unsigned Alignment = Ops->IsB64 ? 8 : 4;
  if (Ops->Off0 % Alignment != 0 || Ops->Off1 % Alignment != 0)
    return false;

  std::optional<ElfView::FunctionTextRange> Function =
      Ctx.Elf.findFunctionTextRangeAtOffset(DS.Offset);
  if (!Function || Ctx.IndirectControlFlowFunctions.contains(Function->Begin) ||
      Ctx.DirectControlFlowTargets->contains(DS.Offset))
    return false;

  for (size_t I = Idx; I-- > 0;) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Offset < Function->Begin)
      break;
    if (Ctx.DirectControlFlowTargets->contains(DI.Offset) ||
        definesModeOrExec(DI, Ctx.LS) ||
        StringRef(DI.Mnemonic).starts_with("s_setreg") ||
        DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>" ||
        (Ctx.LS.MIA && Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI)))
      break;
    if (definesRegister(DI, Address, Ctx.LS))
      return isAlignedAddressDef(DI, Address, Alignment, Ctx, I);
  }
  return false;
}

bool isDs2AddressProvenAligned(PatchContext &Ctx, size_t Idx) {
  if (Ctx.Ds2AddressProvenAligned.size() == Ctx.Decoded.size())
    return Ctx.Ds2AddressProvenAligned.test(Idx);
  return computeDs2AddressProvenAligned(Ctx, Idx);
}

struct Ds2FlowEdge {
  unsigned Successor = 0;
  bool IsBranchTarget = false;
};

struct Ds2FunctionFlow {
  size_t GlobalFirst = 0;
  std::vector<SmallVector<Ds2FlowEdge, 2>> Successors;
  bool Valid = false;
};

static Ds2FunctionFlow
buildDs2FunctionFlow(PatchContext &Ctx,
                     const ElfView::FunctionTextRange &Function) {
  Ds2FunctionFlow Flow;
  auto First =
      llvm::lower_bound(Ctx.Decoded, Function.Begin,
                        [](const InternalDecodedInst &DI, uint64_t Offset) {
                          return DI.Offset < Offset;
                        });
  auto After =
      llvm::lower_bound(Ctx.Decoded, Function.End,
                        [](const InternalDecodedInst &DI, uint64_t Offset) {
                          return DI.Offset < Offset;
                        });
  if (First == After || First->Offset != Function.Begin || !Ctx.LS.MIA)
    return Flow;

  Flow.GlobalFirst = static_cast<size_t>(First - Ctx.Decoded.begin());
  const unsigned Count = static_cast<unsigned>(After - First);
  Flow.Successors.resize(Count);
  DenseMap<uint64_t, unsigned> OffsetToIndex;
  OffsetToIndex.reserve(Count);
  for (unsigned I = 0; I != Count; ++I) {
    if (First[I].Mnemonic == "<unknown>")
      return Flow;
    OffsetToIndex.try_emplace(First[I].Offset, I);
  }

  auto AddFallthrough = [&](unsigned I) {
    if (I + 1 < Count)
      Flow.Successors[I].push_back({I + 1, false});
  };
  for (unsigned I = 0; I != Count; ++I) {
    const InternalDecodedInst &DI = First[I];
    if (DI.Mnemonic == "s_endpgm" || DI.Mnemonic == "s_endpgm_saved" ||
        Ctx.LS.MIA->isReturn(DI.Inst) || isStandardLinkReturn(DI, Ctx.LS))
      continue;
    if (Ctx.LS.MIA->isCall(DI.Inst)) {
      AddFallthrough(I);
      continue;
    }
    if (Ctx.LS.MIA->isBranch(DI.Inst)) {
      uint64_t Target = 0;
      if (Ctx.LS.MIA->isIndirectBranch(DI.Inst)) {
        std::optional<MaterializedPcTransfer> Transfer;
        if (Ctx.DirectControlFlowTargets)
          Transfer = evaluateMaterializedPcTransfer(
              Ctx.Decoded, Flow.GlobalFirst + I, *Ctx.DirectControlFlowTargets,
              ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize), Ctx.LS, Ctx.Elf);
        if (!Transfer)
          return Flow;
        Target = Transfer->Target;
      } else if (!Ctx.LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size,
                                             Target))
        return Flow;
      auto It = OffsetToIndex.find(Target);
      if (It == OffsetToIndex.end() && Target >= Function.Begin &&
          Target < Function.End)
        return Flow;
      if (It != OffsetToIndex.end())
        Flow.Successors[I].push_back({It->second, true});
      if (Ctx.LS.MIA->isConditionalBranch(DI.Inst))
        AddFallthrough(I);
      else if (!Ctx.LS.MIA->isUnconditionalBranch(DI.Inst))
        return Flow;
      continue;
    }
    if (Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI))
      return Flow;
    AddFallthrough(I);
  }
  Flow.Valid = true;
  return Flow;
}

static std::optional<unsigned> ds2NumberedSgprIndex(const MCRegisterInfo &MRI,
                                                    MCRegister Reg,
                                                    unsigned MaxSgprs) {
  if (!Reg.isValid())
    return std::nullopt;
  StringRef Name(MRI.getName(Reg));
  if (!Name.consume_front("SGPR") || Name.empty() || Name.contains('_'))
    return std::nullopt;
  unsigned Index = 0;
  if (Name.getAsInteger(10, Index) || Index >= MaxSgprs)
    return std::nullopt;
  return Index;
}

static SmallVector<MCRegister, 128>
getDs2NumberedSgprs(const PatchContext &Ctx) {
  SmallVector<MCRegister, 128> Registers(Ctx.Config.MaxSgprs);
  for (unsigned Reg = 1, End = Ctx.LS.MRI->getNumRegs(); Reg != End; ++Reg) {
    std::optional<unsigned> Index =
        ds2NumberedSgprIndex(*Ctx.LS.MRI, MCRegister(Reg), Ctx.Config.MaxSgprs);
    if (Index)
      Registers[*Index] = MCRegister(Reg);
  }
  return Registers;
}

struct Ds2MaskTaint {
  bool Reachable = false;
  bool ExecUnsafe = true;
  BitVector UnsafeMasks;

  explicit Ds2MaskTaint(unsigned MaskCount = 0) : UnsafeMasks(MaskCount) {}
};

static bool getDs2MaskUnsafe(const MCOperand &Op, const Ds2MaskTaint &State,
                             const PatchContext &Ctx,
                             ArrayRef<MCRegister> NumberedSgprs) {
  if (Op.isImm())
    return Op.getImm() != 0;
  if (!Op.isReg() || !Op.getReg())
    return true;
  MCRegister Reg(Op.getReg());
  StringRef Name(Ctx.LS.MRI->getName(Reg));
  if (Name.starts_with("EXEC"))
    return State.ExecUnsafe;
  if (Name.starts_with("VCC"))
    return State.UnsafeMasks.test(Ctx.Config.MaxSgprs);
  bool Found = false;
  bool Unsafe = false;
  for (unsigned I = 0; I != NumberedSgprs.size(); ++I) {
    if (!NumberedSgprs[I].isValid() ||
        !Ctx.LS.MRI->regsOverlap(Reg.id(), NumberedSgprs[I].id()))
      continue;
    Found = true;
    Unsafe |= State.UnsafeMasks.test(I);
  }
  return !Found || Unsafe;
}

static bool setDs2MaskUnsafe(MCRegister Reg, bool Unsafe, Ds2MaskTaint &State,
                             const PatchContext &Ctx,
                             ArrayRef<MCRegister> NumberedSgprs) {
  if (!Reg.isValid())
    return false;
  StringRef Name(Ctx.LS.MRI->getName(Reg));
  if (Name.starts_with("EXEC")) {
    State.ExecUnsafe = Unsafe;
    return true;
  }
  if (Name.starts_with("VCC")) {
    if (Unsafe)
      State.UnsafeMasks.set(Ctx.Config.MaxSgprs);
    else
      State.UnsafeMasks.reset(Ctx.Config.MaxSgprs);
    return true;
  }
  bool Found = false;
  for (unsigned I = 0; I != NumberedSgprs.size(); ++I) {
    if (!NumberedSgprs[I].isValid() ||
        !Ctx.LS.MRI->regsOverlap(Reg.id(), NumberedSgprs[I].id()))
      continue;
    if (Unsafe)
      State.UnsafeMasks.set(I);
    else
      State.UnsafeMasks.reset(I);
    Found = true;
  }
  return Found;
}

static void setDs2MaskDefsUnsafe(const InternalDecodedInst &DI, bool Unsafe,
                                 Ds2MaskTaint &State, const PatchContext &Ctx,
                                 ArrayRef<MCRegister> NumberedSgprs) {
  const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != NumDefs; ++I) {
    const MCOperand &Op = DI.Inst.getOperand(I);
    if (Op.isReg() && Op.getReg())
      setDs2MaskUnsafe(MCRegister(Op.getReg()), Unsafe, State, Ctx,
                       NumberedSgprs);
  }
  for (MCPhysReg Reg : Desc.implicit_defs())
    setDs2MaskUnsafe(MCRegister(Reg), Unsafe, State, Ctx, NumberedSgprs);
}

static Ds2MaskTaint transferDs2MaskTaint(PatchContext &Ctx,
                                         const Ds2FunctionFlow &Flow,
                                         unsigned I, unsigned Candidate,
                                         ArrayRef<MCRegister> NumberedSgprs,
                                         Ds2MaskTaint State) {
  const size_t GlobalI = Flow.GlobalFirst + I;
  const InternalDecodedInst &DI = Ctx.Decoded[GlobalI];
  if (I == Candidate)
    State.ExecUnsafe = false;

  if (Ctx.LS.MIA->isCall(DI.Inst)) {
    State.ExecUnsafe = true;
    State.UnsafeMasks.set();
    return State;
  }

  StringRef Mnem(DI.Mnemonic);
  if ((Mnem == "s_and_saveexec_b32" || Mnem == "s_and_not1_saveexec_b32" ||
       Mnem == "s_or_saveexec_b32" || Mnem == "s_xor_saveexec_b32") &&
      DI.Inst.getNumOperands() >= 2 && DI.Inst.getOperand(0).isReg()) {
    const bool OldExecUnsafe = State.ExecUnsafe;
    const bool SourceUnsafe =
        getDs2MaskUnsafe(DI.Inst.getOperand(1), State, Ctx, NumberedSgprs);
    setDs2MaskUnsafe(MCRegister(DI.Inst.getOperand(0).getReg()), OldExecUnsafe,
                     State, Ctx, NumberedSgprs);
    if (Mnem == "s_and_saveexec_b32")
      State.ExecUnsafe = OldExecUnsafe && SourceUnsafe;
    else if (Mnem == "s_and_not1_saveexec_b32")
      State.ExecUnsafe = OldExecUnsafe;
    else
      State.ExecUnsafe = OldExecUnsafe || SourceUnsafe;
    return State;
  }

  if (Mnem.starts_with("v_cmpx")) {
    setDs2MaskDefsUnsafe(DI, State.ExecUnsafe, State, Ctx, NumberedSgprs);
    return State; // EXEC only narrows.
  }
  if (Mnem.starts_with("v_cmp")) {
    setDs2MaskDefsUnsafe(DI, State.ExecUnsafe, State, Ctx, NumberedSgprs);
    return State;
  }

  auto TransferBinary = [&](StringRef Opcode) {
    if (Mnem != Opcode || DI.Inst.getNumOperands() < 3 ||
        !DI.Inst.getOperand(0).isReg())
      return false;
    const bool Left =
        getDs2MaskUnsafe(DI.Inst.getOperand(1), State, Ctx, NumberedSgprs);
    const bool Right =
        getDs2MaskUnsafe(DI.Inst.getOperand(2), State, Ctx, NumberedSgprs);
    bool Unsafe = true;
    if (Opcode == "s_and_b32")
      Unsafe = Left && Right;
    else if (Opcode == "s_and_not1_b32")
      Unsafe = Left;
    else if (Opcode == "s_or_b32" || Opcode == "s_xor_b32")
      Unsafe = Left || Right;
    setDs2MaskUnsafe(MCRegister(DI.Inst.getOperand(0).getReg()), Unsafe, State,
                     Ctx, NumberedSgprs);
    return true;
  };
  if (TransferBinary("s_and_b32") || TransferBinary("s_and_not1_b32") ||
      TransferBinary("s_or_b32") || TransferBinary("s_xor_b32"))
    return State;

  if (Mnem == "s_mov_b32" && DI.Inst.getNumOperands() >= 2 &&
      DI.Inst.getOperand(0).isReg()) {
    setDs2MaskUnsafe(
        MCRegister(DI.Inst.getOperand(0).getReg()),
        getDs2MaskUnsafe(DI.Inst.getOperand(1), State, Ctx, NumberedSgprs),
        State, Ctx, NumberedSgprs);
    return State;
  }

  setDs2MaskDefsUnsafe(DI, true, State, Ctx, NumberedSgprs);
  return State;
}

static bool mergeDs2MaskTaint(Ds2MaskTaint &Into,
                              const Ds2MaskTaint &Incoming) {
  if (!Incoming.Reachable)
    return false;
  if (!Into.Reachable) {
    Into = Incoming;
    return true;
  }
  const bool OldExecUnsafe = Into.ExecUnsafe;
  BitVector OldMasks = Into.UnsafeMasks;
  Into.ExecUnsafe |= Incoming.ExecUnsafe;
  Into.UnsafeMasks |= Incoming.UnsafeMasks;
  return OldExecUnsafe != Into.ExecUnsafe || OldMasks != Into.UnsafeMasks;
}

static void proveLongRangeDs2Alignment(PatchContext &Ctx, size_t CandidateIdx,
                                       MCRegister Address,
                                       ArrayRef<size_t> Uses,
                                       BitVector &Proven) {
  std::optional<ElfView::FunctionTextRange> Function =
      Ctx.Elf.findFunctionTextRangeAtOffset(Ctx.Decoded[CandidateIdx].Offset);
  if (!Function || Ctx.IndirectControlFlowFunctions.contains(Function->Begin) ||
      Ctx.CrossFunctionInteriorEntryFunctions.contains(Function->Begin) ||
      CandidateIdx >= Ctx.VgprMsbDstBefore.size())
    return;
  const int8_t CandidateDst = Ctx.VgprMsbDstBefore[CandidateIdx];
  if (CandidateDst < 0)
    return;

  Ds2FunctionFlow Flow = buildDs2FunctionFlow(Ctx, *Function);
  if (!Flow.Valid || CandidateIdx < Flow.GlobalFirst)
    return;
  const unsigned Candidate =
      static_cast<unsigned>(CandidateIdx - Flow.GlobalFirst);
  if (Candidate >= Flow.Successors.size())
    return;

  // Address redefinitions are independent of lane-mask coverage. Propagate a
  // monotone may-clobber fact from Candidate so a loop cannot hide a clobber
  // by executing Candidate again for a disjoint set of lanes.
  std::vector<uint8_t> AddressFlow(Flow.Successors.size());
  SmallVector<unsigned, 128> AddressWorklist;
  AddressFlow[Candidate] = 1; // bit 0: reachable, bit 1: clobbered
  AddressWorklist.push_back(Candidate);
  for (size_t Next = 0; Next != AddressWorklist.size(); ++Next) {
    const unsigned I = AddressWorklist[Next];
    uint8_t Out = AddressFlow[I];
    if (I != Candidate &&
        (definesRegister(Ctx.Decoded[Flow.GlobalFirst + I], Address, Ctx.LS) ||
         Ctx.LS.MIA->isCall(Ctx.Decoded[Flow.GlobalFirst + I].Inst)))
      Out |= 2;
    for (const Ds2FlowEdge &Edge : Flow.Successors[I]) {
      uint8_t Merged = AddressFlow[Edge.Successor] | Out;
      if (Merged != AddressFlow[Edge.Successor]) {
        AddressFlow[Edge.Successor] = Merged;
        AddressWorklist.push_back(Edge.Successor);
      }
    }
  }

  const unsigned MaskCount = Ctx.Config.MaxSgprs + 1;
  SmallVector<MCRegister, 128> NumberedSgprs = getDs2NumberedSgprs(Ctx);
  std::vector<Ds2MaskTaint> In;
  In.reserve(Flow.Successors.size());
  for (size_t I = 0; I != Flow.Successors.size(); ++I)
    In.emplace_back(MaskCount);
  In[0].Reachable = true;
  In[0].ExecUnsafe = true;
  In[0].UnsafeMasks.set();
  SmallVector<unsigned, 128> Worklist;
  Worklist.push_back(0);
  for (size_t Next = 0; Next != Worklist.size(); ++Next) {
    const unsigned I = Worklist[Next];
    Ds2MaskTaint Out =
        transferDs2MaskTaint(Ctx, Flow, I, Candidate, NumberedSgprs, In[I]);
    for (const Ds2FlowEdge &Edge : Flow.Successors[I]) {
      Ds2MaskTaint EdgeOut = Out;
      StringRef Mnem(Ctx.Decoded[Flow.GlobalFirst + I].Mnemonic);
      if ((Mnem == "s_cbranch_execz" && Edge.IsBranchTarget) ||
          (Mnem == "s_cbranch_execnz" && !Edge.IsBranchTarget))
        EdgeOut.ExecUnsafe = false; // This edge proves EXEC is empty.
      if (mergeDs2MaskTaint(In[Edge.Successor], EdgeOut))
        Worklist.push_back(Edge.Successor);
    }
  }

  for (size_t UseIdx : Uses) {
    if (UseIdx < Flow.GlobalFirst ||
        UseIdx - Flow.GlobalFirst >= Flow.Successors.size() ||
        UseIdx >= Ctx.VgprMsbSrc0Before.size())
      continue;
    const unsigned Use = static_cast<unsigned>(UseIdx - Flow.GlobalFirst);
    if (!In[Use].Reachable || In[Use].ExecUnsafe ||
        (AddressFlow[Use] & 2) != 0 ||
        Ctx.VgprMsbSrc0Before[UseIdx] != CandidateDst)
      continue;
    Proven.set(UseIdx);
  }
}

// -- patchDs2Addr -----------------------------------------------------------
//
// Expand one ds_*_2addr_* instruction (stride64 or non-stride64) into two
// single-address DS instructions, followed by an s_wait_dscnt 0 drain so both
// halves are guaranteed complete before any downstream DS consumer. Splitting
// one DS instruction into two perturbs the outstanding-DS instruction count
// that later s_wait_dscnt immediates encode; the local drain sidesteps that
// entirely (see the rationale in the body below).

SmallVector<uint8_t> buildDs2AddrReplacement(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  StringRef ToMnem = getDs2AddrReplacement(DI.Mnemonic);
  if (ToMnem.empty())
    return {};
  std::vector<std::string> Expanded =
      expandDs2Addr(DI.Inst, DI.Mnemonic, ToMnem, Ctx.LS);
  if (Expanded.empty()) {
    log() << "hotswap: error: ds_2addr expansion failed for: " << DI.Mnemonic
          << "\n";
    return {};
  }

  std::string Combined;
  for (const std::string &Line : Expanded)
    Combined += Line + "\n";
  // Drain the DS counter right after the split pair so both halves are
  // guaranteed complete before any downstream consumer. The original code
  // tracked completion of the single 2-addr instruction via a later
  // s_wait_dscnt whose immediate counts outstanding DS *instructions*;
  // splitting one instruction into two perturbs that count. Adjusting the
  // downstream wait by +1 (the previous bumpNextWaitDscnt approach) relaxes
  // the wait (s_wait_dscnt K stalls until outstanding <= K, so a larger K
  // waits for FEWER ops), which lets a consumer read the second half's LDS
  // slot before it lands -- observed as NaN in MIOpen layernormbfp16. A
  // local drain is unconditionally correct; a precise per-wait dataflow
  // recomputation is the eventual optimization (tracked separately).
  Combined += "s_wait_dscnt 0\n";
  SmallVector<uint8_t> Bytes = assembleInstructions(Combined, Ctx.LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: ds_2addr: assembly failed: " << Combined << "\n";
    return {};
  }
  return Bytes;
}

struct ProtectedDs2DelayWindow {
  size_t DelayIndex = 0;
  size_t LastIndex = 0;
  uint32_t SourceSize = 0;
  unsigned FirstDependency = 0;
  unsigned SecondDependency = 0;
  SmallVector<size_t, 4> Ds2Indices;
};

/// A combined s_delay_alu names the immediately following instruction and a
/// later instruction selected by instskip. LLVM forms it by folding a second
/// standalone delay into the first. A DS2 split between those two targets
/// changes the instruction count, so restore the two standalone delays and
/// relocate the complete protected window as one unit.
static std::optional<ProtectedDs2DelayWindow>
findProtectedDs2DelayWindow(PatchContext &Ctx, size_t Ds2Index) {
  if (Ds2Index == 0 || !Ctx.DirectControlFlowTargets || !Ctx.LS.MIA)
    return std::nullopt;

  // Locate the owner from encoded delay geometry rather than assuming the DS2
  // immediately follows the first protected target. Demerging preserves both
  // dependencies only when DS2 is strictly between those targets: replacing a
  // target itself would change the operation named by that dependency.
  std::optional<ProtectedDs2DelayWindow> Window;
  constexpr size_t MaxClauseSpan = 64;
  const size_t FirstPossibleOwner =
      Ds2Index > MaxClauseSpan ? Ds2Index - MaxClauseSpan : 0;
  unsigned DelayOwners = 0;
  for (size_t I = FirstPossibleOwner; I != Ds2Index; ++I) {
    const InternalDecodedInst &Candidate = Ctx.Decoded[I];
    if (Candidate.Mnemonic == "s_delay_alu") {
      const unsigned CandidateSpan = getDelayProtectedSpan(Candidate);
      if (CandidateSpan < Ds2Index - I)
        continue;
      ++DelayOwners;

      if (Candidate.Inst.getNumOperands() != 1 ||
          !Candidate.Inst.getOperand(0).isImm())
        continue;
      const uint64_t Imm =
          static_cast<uint64_t>(Candidate.Inst.getOperand(0).getImm());
      if ((Imm & ~uint64_t{0x7FF}) != 0)
        continue;
      const unsigned FirstDependency = Imm & 0xF;
      const unsigned Skip = (Imm >> 4) & 0x7;
      const unsigned SecondDependency = (Imm >> 7) & 0xF;
      if (FirstDependency == 0 || FirstDependency >= 12 || Skip > 5 ||
          SecondDependency == 0 || SecondDependency >= 12)
        continue;

      const unsigned Span = Skip + 1;
      if (getDelayProtectedSpan(Candidate) != Span ||
          Span > Ctx.Decoded.size() - I - 1)
        continue;
      const size_t FirstTargetIndex = I + 1;
      const size_t SecondTargetIndex = I + Span;
      if (Ds2Index <= FirstTargetIndex || Ds2Index >= SecondTargetIndex)
        continue;
      if (Window)
        return std::nullopt;
      Window = ProtectedDs2DelayWindow{
          I, SecondTargetIndex, 0, FirstDependency, SecondDependency, {}};
      continue;
    }
    if (Candidate.Mnemonic != "s_clause" ||
        Candidate.Inst.getNumOperands() != 1 ||
        !Candidate.Inst.getOperand(0).isImm())
      continue;
    const unsigned ClauseSpan =
        (static_cast<unsigned>(Candidate.Inst.getOperand(0).getImm()) & 63u) +
        1;
    if (ClauseSpan >= Ds2Index - I)
      return std::nullopt;
  }
  if (DelayOwners != 1 || !Window)
    return std::nullopt;

  const size_t DelayIndex = Window->DelayIndex;
  const size_t LastIndex = Window->LastIndex;
  const InternalDecodedInst &Delay = Ctx.Decoded[DelayIndex];

  // Reject an owner whose directive is itself covered by an earlier delay or
  // hard clause. Such nested geometry cannot be preserved by this demerge.
  if (Ctx.RelocationProtectedOffsets.contains(Delay.Offset))
    return std::nullopt;

  std::optional<ElfView::FunctionTextRange> Function =
      Ctx.Elf.findFunctionTextRangeAtOffset(Delay.Offset);
  if (!Function || Ctx.IndirectControlFlowFunctions.contains(Function->Begin))
    return std::nullopt;

  uint64_t ExpectedOffset = Delay.Offset;
  uint64_t WindowEnd = Delay.Offset;
  for (size_t I = DelayIndex; I <= LastIndex; ++I) {
    const InternalDecodedInst &Member = Ctx.Decoded[I];
    const bool IsDs2 = !getDs2AddrReplacement(Member.Mnemonic).empty();
    const bool NeedsDs2Rewrite = IsDs2 && !isDs2AddressProvenAligned(Ctx, I);
    std::optional<ElfView::FunctionTextRange> MemberFunction =
        Ctx.Elf.findFunctionTextRangeAtOffset(Member.Offset);
    if (!MemberFunction || MemberFunction->Begin != Function->Begin ||
        MemberFunction->End != Function->End ||
        Member.Offset != ExpectedOffset || Member.Offset > Ctx.TextSize ||
        Member.Size > Ctx.TextSize - Member.Offset ||
        Member.Mnemonic == "<unknown>" || Member.Mnemonic == "<replaced>")
      return std::nullopt;
    if (I != DelayIndex &&
        (Member.Mnemonic == "s_delay_alu" || Member.Mnemonic == "s_clause" ||
         Member.Mnemonic == "s_set_vgpr_msb" ||
         Member.Mnemonic == "tensor_load_to_lds" ||
         StringRef(Member.Mnemonic).contains("_pc_") ||
         Ctx.LS.MIA->mayAffectControlFlow(Member.Inst, *Ctx.LS.MRI)))
      return std::nullopt;
    if (NeedsDs2Rewrite) {
      // Demerging keeps the original dependency targets as single
      // instructions. Required DS2 rewrites are therefore composable only
      // at interior positions, where every replacement can move as part of
      // the same atomic window.
      if (I == DelayIndex + 1 || I == LastIndex ||
          Ctx.ClaimedReplacementOffsets.contains(Member.Offset) ||
          hasSiteReplacementReservation(Ctx, Member.Offset))
        return std::nullopt;
      Window->Ds2Indices.push_back(I);
    }
    // The outer dispatcher skips every member after this atomic relocation.
    // Reject any as-yet-unvisited member that would need its own patch rather
    // than silently copying the incompatible instruction into the body.
    if (I > Ds2Index && !NeedsDs2Rewrite &&
        requiresIndependentInstructionRewrite(Ctx, I)) {
      log() << "hotswap: error: ds_2addr: combined-delay window member at 0x"
            << utohexstr(Member.Offset)
            << " requires a separate HotSwap patch\n";
      return std::nullopt;
    }
    ExpectedOffset += Member.Size;
    WindowEnd = ExpectedOffset;
  }

  if (WindowEnd < Delay.Offset ||
      WindowEnd - Delay.Offset > std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  for (uint64_t Offset = Delay.Offset + MinInstSize; Offset < WindowEnd;
       Offset += MinInstSize)
    if (Ctx.DirectControlFlowTargets->contains(Offset))
      return std::nullopt;

  Window->SourceSize = static_cast<uint32_t>(WindowEnd - Delay.Offset);
  if (Window->Ds2Indices.empty() ||
      !llvm::is_contained(Window->Ds2Indices, Ds2Index))
    return std::nullopt;
  return Window;
}

static bool
patchProtectedDs2DelayWindow(PatchContext &Ctx, size_t Ds2Index,
                             ArrayRef<uint8_t> Ds2Replacement,
                             const ProtectedDs2DelayWindow &Window) {
  SmallVector<uint8_t> FirstDelay = assembleSingleInst(
      "s_delay_alu " + std::to_string(Window.FirstDependency), Ctx.LS);
  SmallVector<uint8_t> SecondDelay = assembleSingleInst(
      "s_delay_alu " + std::to_string(Window.SecondDependency), Ctx.LS);
  if (FirstDelay.size() != MinInstSize || SecondDelay.size() != MinInstSize)
    return false;

  SmallVector<uint8_t> Replacement;
  Replacement.append(FirstDelay.begin(), FirstDelay.end());
  for (size_t I = Window.DelayIndex + 1; I <= Window.LastIndex; ++I) {
    if (I == Window.LastIndex)
      Replacement.append(SecondDelay.begin(), SecondDelay.end());
    if (llvm::is_contained(Window.Ds2Indices, I)) {
      if (I == Ds2Index) {
        Replacement.append(Ds2Replacement.begin(), Ds2Replacement.end());
        continue;
      }
      SmallVector<uint8_t> MemberReplacement = buildDs2AddrReplacement(Ctx, I);
      if (MemberReplacement.empty())
        return false;
      Replacement.append(MemberReplacement.begin(), MemberReplacement.end());
      continue;
    }
    const InternalDecodedInst &Member = Ctx.Decoded[I];
    Replacement.append(Ctx.Text + Member.Offset,
                       Ctx.Text + Member.Offset + Member.Size);
  }

  const uint64_t SourceOffset = Ctx.Decoded[Window.DelayIndex].Offset;
  if (!emitReplacementCode(Ctx, SourceOffset, Window.SourceSize, Replacement,
                           ReplacementPlacement::ProtectedCombinedDelay))
    return false;

  const uint64_t Ds2Offset = Ctx.Decoded[Ds2Index].Offset;
  for (size_t I = Window.DelayIndex; I <= Window.LastIndex; ++I)
    Ctx.Decoded[I].Mnemonic = "<replaced>";
  Ctx.RequiredPatchApplied = true;
  log() << "hotswap: ds_2addr: demerged combined s_delay_alu at 0x"
        << utohexstr(SourceOffset) << " around protected site 0x"
        << utohexstr(Ds2Offset) << " with " << Window.Ds2Indices.size()
        << " DS2 rewrite(s)\n";
  return true;
}

uint32_t patchDs2Addr(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  SmallVector<uint8_t> Replacement = buildDs2AddrReplacement(Ctx, Idx);
  if (Replacement.empty()) {
    failRequiredPatch(Ctx);
    return 0;
  }

  if (Ctx.RelocationProtectedOffsets.contains(DI.Offset)) {
    std::optional<ProtectedDs2DelayWindow> Window =
        findProtectedDs2DelayWindow(Ctx, Idx);
    if (!Window) {
      log() << "hotswap: error: ds_2addr: protected source at 0x"
            << utohexstr(DI.Offset)
            << " has no supported combined-delay window\n";
      failRequiredPatch(Ctx);
      return 0;
    }
    if (!patchProtectedDs2DelayWindow(Ctx, Idx, Replacement, *Window)) {
      failRequiredPatch(Ctx);
      return 0;
    }
    return Window->Ds2Indices.size();
  }

  std::optional<ElfView::FunctionTextRange> Function =
      Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset);

  // Adjacent required DS2 rewrites need one branch-back, not one per original
  // instruction. This preserves replacement order and every local DS drain
  // while reducing padding pressure. Never erase a possible interior entry.
  if (!Ctx.HasUnknownArbitraryIndirectTarget && Ctx.DirectControlFlowTargets &&
      Function && !Ctx.IndirectControlFlowFunctions.contains(Function->Begin) &&
      Idx + 1 < Ctx.Decoded.size()) {
    size_t Next = Idx + 1;
    std::optional<uint64_t> ExpectedOffset =
        checkedAddUint64(DI.Offset, DI.Size, "adjacent ds_2addr source window");
    if (ExpectedOffset) {
      InternalDecodedInst &NextDI = Ctx.Decoded[Next];
      std::optional<uint64_t> NextEnd = checkedAddUint64(
          NextDI.Offset, NextDI.Size, "adjacent ds_2addr source end");
      std::optional<ElfView::FunctionTextRange> NextFunction =
          Ctx.Elf.findFunctionTextRangeAtOffset(NextDI.Offset);
      if (NextDI.Offset == *ExpectedOffset && NextEnd &&
          *NextEnd <= Function->End &&
          !Ctx.DirectControlFlowTargets->contains(NextDI.Offset) &&
          !Ctx.RelocationProtectedOffsets.contains(NextDI.Offset) &&
          !getDs2AddrReplacement(NextDI.Mnemonic).empty() &&
          !isDs2AddressProvenAligned(Ctx, Next) && NextFunction &&
          NextFunction->Begin == Function->Begin &&
          NextFunction->End == Function->End &&
          NextDI.Size <= std::numeric_limits<uint32_t>::max() - DI.Size) {
        SmallVector<uint8_t> NextReplacement =
            buildDs2AddrReplacement(Ctx, Next);
        if (!NextReplacement.empty()) {
          SmallVector<uint8_t> Combined = Replacement;
          Combined.append(NextReplacement.begin(), NextReplacement.end());
          uint32_t CombinedOriginalSize = DI.Size + NextDI.Size;
          if (emitReplacementCode(Ctx, DI.Offset, CombinedOriginalSize,
                                  Combined, ReplacementPlacement::Ds2SourceTail,
                                  /*DiagnoseFailure=*/false)) {
            Ctx.RequiredPatchApplied = true;
            DI.Mnemonic = "<replaced>";
            NextDI.Mnemonic = "<replaced>";
            log() << "hotswap: coalesced 2 adjacent ds_2addr rewrites at 0x"
                  << utohexstr(DI.Offset) << "\n";
            return 2;
          }
        }
      }
    }
  }

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement,
                           ReplacementPlacement::Ds2SourceTail)) {
    failRequiredPatch(Ctx);
    return 0;
  }

  Ctx.RequiredPatchApplied = true;
  DI.Mnemonic = "<replaced>";
  return 1;
}

// -- getDescriptorBaseSgpr --------------------------------------------------
//
// Extract the base SGPR MCRegister from the second operand of a
// tensor_load_to_lds instruction. The second operand is an 8-SGPR group
// descriptor (SReg_256); we need its first sub-register for the
// s_pack_hh_b32_b16 fix.

MCRegister getDescriptorBaseSgpr(const MCInst &Inst,
                                 const MCRegisterInfo &MRI) {
  if (Inst.getNumOperands() < 2 || !Inst.getOperand(1).isReg())
    return MCRegister();
  MCRegister Tuple = MCRegister(Inst.getOperand(1).getReg());
  SmallVector<MCRegister, 4> Subs = getDirectSubRegs(Tuple, MRI);
  return Subs.empty() ? MCRegister() : Subs[0];
}

std::optional<unsigned> getSgprIndex(MCRegister Reg,
                                     const MCRegisterInfo &MRI) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return std::nullopt;
  StringRef Name(N);
  if (!Name.starts_with("SGPR") || Name.contains('_'))
    return std::nullopt;
  unsigned Index = 0;
  if (Name.drop_front(4).getAsInteger(10, Index))
    return std::nullopt;
  return Index;
}

SmallVector<unsigned, 8> getDescriptorSgprIndices(const MCInst &Inst,
                                                  const MCRegisterInfo &MRI) {
  SmallVector<unsigned, 8> Result;
  if (Inst.getNumOperands() < 2 || !Inst.getOperand(1).isReg())
    return Result;

  MCRegister Tuple = MCRegister(Inst.getOperand(1).getReg());
  for (MCRegister Sub : getDirectSubRegs(Tuple, MRI)) {
    if (std::optional<unsigned> Index = getSgprIndex(Sub, MRI))
      Result.push_back(*Index);
  }
  return Result;
}

SmallVector<unsigned, 8> getSgprOperandIndices(const MCInst &Inst,
                                               const MCRegisterInfo &MRI) {
  SmallVector<unsigned, 8> Result;
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (!Op.isReg() || !Op.getReg())
      continue;

    MCRegister Reg = MCRegister(Op.getReg());
    if (std::optional<unsigned> Index = getSgprIndex(Reg, MRI)) {
      Result.push_back(*Index);
      continue;
    }

    for (MCRegister Sub : getDirectSubRegs(Reg, MRI)) {
      if (std::optional<unsigned> Index = getSgprIndex(Sub, MRI))
        Result.push_back(*Index);
    }
  }
  return Result;
}

bool isTensorDescriptorMask(const InternalDecodedInst &DI, MCRegister BaseMCReg,
                            const MCRegisterInfo &MRI) {
  const MCInst &Inst = DI.Inst;
  if (DI.Mnemonic != "s_pack_hh_b32_b16" || Inst.getNumOperands() < 3 ||
      !Inst.getOperand(0).isReg() ||
      !MRI.regsOverlap(Inst.getOperand(0).getReg(), BaseMCReg.id()) ||
      !Inst.getOperand(1).isImm() || Inst.getOperand(1).getImm() != 0 ||
      !Inst.getOperand(2).isReg())
    return false;
  return MRI.regsOverlap(Inst.getOperand(2).getReg(), BaseMCReg.id());
}

bool isAlreadyTensorMaskPatched(const PatchContext &Ctx, size_t Idx,
                                MCRegister BaseMCReg) {
  if (Idx == 0)
    return false;
  const InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
  return Prev.Offset + Prev.Size == Ctx.Decoded[Idx].Offset &&
         isTensorDescriptorMask(Prev, BaseMCReg, *Ctx.LS.MRI);
}

bool instructionTouchesRegister(const InternalDecodedInst &DI, MCRegister Reg,
                                const LLVMState &LS) {
  const MCRegisterInfo &MRI = *LS.MRI;
  for (const MCOperand &Op : DI.Inst)
    if (Op.isReg() && Op.getReg() && MRI.regsOverlap(Op.getReg(), Reg.id()))
      return true;

  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  for (MCPhysReg Implicit : Desc.implicit_uses())
    if (MRI.regsOverlap(Implicit, Reg.id()))
      return true;
  for (MCPhysReg Implicit : Desc.implicit_defs())
    if (MRI.regsOverlap(Implicit, Reg.id()))
      return true;
  return false;
}

struct LocalTensorMaskDefinition {
  size_t Index = 0;
  bool AlreadyMasked = false;
};

bool isDirectControlFlowTarget(const PatchContext &Ctx, uint64_t Offset) {
  return !Ctx.DirectControlFlowTargets ||
         Ctx.DirectControlFlowTargets->contains(Offset);
}

bool functionHasIndirectControlFlow(const PatchContext &Ctx, uint64_t Offset) {
  std::optional<ElfView::FunctionTextRange> Range =
      Ctx.Elf.findFunctionTextRangeAtOffset(Offset);
  return !Range || Ctx.IndirectControlFlowFunctions.contains(Range->Begin);
}

bool isRelocatedTensorMaskDefinition(const PatchContext &Ctx,
                                     const InternalDecodedInst &Branch,
                                     MCRegister BaseMCReg) {
  if (!Ctx.LS.MIA || Branch.Mnemonic != "s_branch" ||
      !Ctx.LS.MIA->isUnconditionalBranch(Branch.Inst) ||
      Ctx.LS.MIA->isIndirectBranch(Branch.Inst))
    return false;

  uint64_t Target = 0;
  if (!Ctx.LS.MIA->evaluateBranch(Branch.Inst, Branch.Offset, Branch.Size,
                                  Target))
    return false;

  constexpr uint64_t SequenceBytes = 3 * MinInstSize;
  std::optional<uint64_t> TargetVAddr = checkedAddUint64(
      Ctx.Elf.textAddr(), Target, "tensor mask trampoline target");
  if (!TargetVAddr)
    return false;
  const uint8_t *Bytes = Ctx.Elf.dataAtVAddr(*TargetVAddr, SequenceBytes);
  if (!Bytes)
    return false;
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes, SequenceBytes, Ctx.LS, Decoded) ||
      Decoded.size() != 3)
    return false;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const InternalDecodedInst &Def = Decoded[0];
  const MCInstrDesc &DefDesc = Ctx.LS.MCII->get(Def.Inst.getOpcode());
  if (Def.Mnemonic != "v_readfirstlane_b32" || DefDesc.getNumDefs() != 1 ||
      Def.Inst.getNumOperands() < 2 || !Def.Inst.getOperand(0).isReg() ||
      Def.Inst.getOperand(0).getReg() != BaseMCReg.id() ||
      !isTensorDescriptorMask(Decoded[1], BaseMCReg, MRI) ||
      Decoded[2].Mnemonic != "s_branch")
    return false;

  uint64_t Resume = 0;
  if (!Ctx.LS.MIA->evaluateBranch(Decoded[2].Inst, Target + Decoded[2].Offset,
                                  Decoded[2].Size, Resume))
    return false;
  return Resume == Branch.Offset + Branch.Size;
}

bool isTransparentShortTrampoline(const PatchContext &Ctx,
                                  const InternalDecodedInst &Branch,
                                  MCRegister BaseMCReg) {
  if (!Ctx.LS.MIA || Branch.Mnemonic != "s_branch" ||
      !Ctx.LS.MIA->isUnconditionalBranch(Branch.Inst) ||
      Ctx.LS.MIA->isIndirectBranch(Branch.Inst))
    return false;

  uint64_t Target = 0;
  if (!Ctx.LS.MIA->evaluateBranch(Branch.Inst, Branch.Offset, Branch.Size,
                                  Target) ||
      Target < Ctx.TextSize)
    return false;
  std::optional<uint64_t> TargetVAddr =
      checkedAddUint64(Ctx.Elf.textAddr(), Target, "tensor trampoline target");
  if (!TargetVAddr)
    return false;

  constexpr uint64_t MaxSequenceBytes = 256;
  for (uint64_t Size = 2 * MinInstSize; Size <= MaxSequenceBytes;
       Size += MinInstSize) {
    const uint8_t *Bytes = Ctx.Elf.dataAtVAddr(*TargetVAddr, Size);
    if (!Bytes)
      return false;
    std::vector<InternalDecodedInst> Decoded;
    if (!decodeTextSection(Bytes, Size, Ctx.LS, Decoded) || Decoded.empty() ||
        Decoded.back().Offset + Decoded.back().Size != Size)
      continue;

    const InternalDecodedInst &Back = Decoded.back();
    uint64_t Resume = 0;
    if (Back.Mnemonic != "s_branch" ||
        !Ctx.LS.MIA->evaluateBranch(Back.Inst, Target + Back.Offset, Back.Size,
                                    Resume))
      continue;
    const uint64_t Fallthrough = Branch.Offset + Branch.Size;
    if (Resume < Fallthrough || Resume - Fallthrough > 4 * MinInstSize ||
        (Resume - Fallthrough) % MinInstSize != 0 || Resume > Ctx.TextSize)
      continue;
    bool PaddingIsNop = true;
    for (uint64_t Offset = Fallthrough; Offset < Resume; Offset += MinInstSize)
      PaddingIsNop &= ArrayRef<uint8_t>(Ctx.Text + Offset, MinInstSize) ==
                      ArrayRef<uint8_t>(Ctx.LS.SNopBytes);
    if (!PaddingIsNop)
      continue;

    for (const InternalDecodedInst &DI : ArrayRef(Decoded).drop_back()) {
      const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
      if (DI.Mnemonic == "<unknown>" ||
          instructionTouchesRegister(DI, BaseMCReg, Ctx.LS) ||
          Desc.mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI))
        return false;
    }
    return true;
  }
  return false;
}

// Find a descriptor definition that unconditionally precedes the tensor in
// the same straight-line basic block. Relocating that definition and appending
// the mask is safe only when no entry edge can skip it and no intervening
// instruction observes or changes the descriptor base.
std::optional<LocalTensorMaskDefinition>
findLocalTensorMaskDefinition(const PatchContext &Ctx, size_t TensorIdx,
                              MCRegister BaseMCReg) {
  if (!Ctx.LS.MIA || TensorIdx == 0 ||
      isDirectControlFlowTarget(Ctx, Ctx.Decoded[TensorIdx].Offset) ||
      functionHasIndirectControlFlow(Ctx, Ctx.Decoded[TensorIdx].Offset))
    return std::nullopt;

  std::optional<ElfView::FunctionTextRange> Range =
      Ctx.Elf.findFunctionTextRangeAtOffset(Ctx.Decoded[TensorIdx].Offset);
  if (!Range)
    return std::nullopt;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  uint64_t NextOffset = Ctx.Decoded[TensorIdx].Offset;
  for (size_t I = TensorIdx; I-- > 0;) {
    const InternalDecodedInst &Candidate = Ctx.Decoded[I];
    if (Candidate.Offset < Range->Begin ||
        Candidate.Offset + Candidate.Size != NextOffset ||
        Candidate.Mnemonic == "<unknown>" || Candidate.Mnemonic == "<replaced>")
      return std::nullopt;

    if (isTensorDescriptorMask(Candidate, BaseMCReg, MRI))
      return LocalTensorMaskDefinition{I, true};
    if (isRelocatedTensorMaskDefinition(Ctx, Candidate, BaseMCReg))
      return LocalTensorMaskDefinition{I, true};

    const MCInstrDesc &Desc = Ctx.LS.MCII->get(Candidate.Inst.getOpcode());
    const bool IsMatchingReadFirstLane =
        Candidate.Mnemonic == "v_readfirstlane_b32" && Desc.getNumDefs() == 1 &&
        Candidate.Inst.getNumOperands() >= 2 &&
        Candidate.Inst.getOperand(0).isReg() &&
        Candidate.Inst.getOperand(0).getReg() == BaseMCReg.id();
    if (IsMatchingReadFirstLane)
      return LocalTensorMaskDefinition{I, false};

    if (instructionTouchesRegister(Candidate, BaseMCReg, Ctx.LS) ||
        isDirectControlFlowTarget(Ctx, Candidate.Offset))
      return std::nullopt;
    if (Desc.mayAffectControlFlow(Candidate.Inst, MRI) &&
        !isTransparentShortTrampoline(Ctx, Candidate, BaseMCReg))
      return std::nullopt;
    NextOffset = Candidate.Offset;
  }
  return std::nullopt;
}

bool isSccReg(MCRegister Reg, const MCRegisterInfo &MRI) {
  const char *N = MRI.getName(Reg);
  return N && StringRef(N) == "SCC";
}

bool hasSccReg(ArrayRef<MCPhysReg> Regs, const MCRegisterInfo &MRI) {
  for (MCPhysReg Reg : Regs) {
    if (isSccReg(MCRegister(Reg), MRI))
      return true;
  }
  return false;
}

bool explicitDefsScc(const MCInst &Inst, const MCInstrDesc &Desc,
                     const MCRegisterInfo &MRI) {
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = 0; I < NumDefs; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() && isSccReg(MCRegister(Op.getReg()), MRI))
      return true;
  }
  return false;
}

bool explicitUsesScc(const MCInst &Inst, const MCInstrDesc &Desc,
                     const MCRegisterInfo &MRI) {
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = NumDefs, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() && isSccReg(MCRegister(Op.getReg()), MRI))
      return true;
  }
  return false;
}

bool instReadsScc(const MCInst &Inst, const MCInstrDesc &Desc,
                  const MCRegisterInfo &MRI) {
  return explicitUsesScc(Inst, Desc, MRI) ||
         hasSccReg(Desc.implicit_uses(), MRI);
}

bool instWritesScc(const MCInst &Inst, const MCInstrDesc &Desc,
                   const MCRegisterInfo &MRI) {
  return explicitDefsScc(Inst, Desc, MRI) ||
         hasSccReg(Desc.implicit_defs(), MRI);
}

// -- isSgprLiveAfter --------------------------------------------------------
//
// Conservative forward-scan heuristic. Returns true if the given SGPR
// (identified by its MCRegister) is used before being redefined in the
// instruction stream following Idx. Conservatively returns true on
// control-flow-affecting instructions or end of stream.

bool isSgprLiveAfter(const PatchContext &Ctx, size_t Idx,
                     MCRegister SgprMCReg) {
  if (!SgprMCReg.isValid())
    return true;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const MCInstrInfo &MCII = *Ctx.LS.MCII;

  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>")
      continue;

    const MCInst &Inst = DI.Inst;
    const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());

    if (DI.Mnemonic == "s_endpgm")
      return false;

    if (Desc.mayAffectControlFlow(Inst, MRI))
      return true;

    unsigned NumDefs = Desc.getNumDefs();
    auto RegInRange = [&](ArrayRef<MCOperand> Ops) {
      for (const MCOperand &Op : Ops) {
        if (!Op.isReg() || !Op.getReg())
          continue;
        if (MRI.regsOverlap(Op.getReg(), SgprMCReg.id()))
          return true;
      }
      return false;
    };
    ArrayRef<MCOperand> Operands = Inst.getOperands();
    ArrayRef<MCOperand> Defs = Operands.slice(0, NumDefs);
    ArrayRef<MCOperand> Uses = Operands.slice(NumDefs);
    if (RegInRange(Uses))
      return true;
    if (RegInRange(Defs))
      return false;
  }

  return true;
}

// -- isSccLiveAfter ---------------------------------------------------------
//
// Conservative forward-scan heuristic for the scalar condition code. Returns
// true if SCC is read before the next instruction that defines SCC. Returns
// true at control-flow boundaries because the linear stream alone cannot prove
// the branch target does not consume the incoming SCC value.

bool isSccLiveAfter(const PatchContext &Ctx, size_t Idx) {
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const MCInstrInfo &MCII = *Ctx.LS.MCII;

  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>")
      continue;

    const MCInst &Inst = DI.Inst;
    const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());

    if (DI.Mnemonic == "s_endpgm")
      return false;

    if (instReadsScc(Inst, Desc, MRI))
      return true;
    if (instWritesScc(Inst, Desc, MRI))
      return false;
    if (Desc.mayAffectControlFlow(Inst, MRI))
      return true;
  }

  return true;
}

// -- scratch-VGPR allocation ------------------------------------------------
//
// Allocation is split into a pure try-step and a commit-step so callers can
// decide a scratch VGPR before assembling/emitting the patch and then only
// charge the kernel descriptor for the extra VGPRs once the patch is known
// to have landed. Bumping KernelPatchStats inside the try-step would leave
// orphan VGPR reservations in the kernel descriptor whenever assembly or
// emission failed downstream.

struct ScratchAlloc {
  unsigned Vgpr = 0;
  std::string KernelName;
  unsigned ExtraVgprsNeeded = 0;
};

std::optional<ScratchAlloc> tryAllocScratchVgpr(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  // findKernelAtAddress matches against symbol virtual addresses, so bias the
  // .text-relative DI.Offset by textAddr() (matching the other patches). A
  // bare offset misses when .text has a non-zero sh_addr, leaving KdVgprs ==
  // 0 and handing the allocator a live register.
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  unsigned KdVgprs = 0;
  if (std::optional<unsigned> Opt = Ctx.Elf.getKernelVgprCount(
          KernelName, getKernelVgprGranuleSize(Ctx, KernelName)))
    KdVgprs = *Opt;

  VgprAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdVgprs,
                      Ctx.Config.MaxVgprs);
  std::optional<unsigned> ScratchOpt = Alloc.alloc();
  if (!ScratchOpt)
    return std::nullopt;

  ScratchAlloc Out;
  Out.Vgpr = *ScratchOpt;
  Out.KernelName = std::move(KernelName);
  Out.ExtraVgprsNeeded = Alloc.extraVgprsNeeded();
  return Out;
}

// Apply the kernel-descriptor accounting for a scratch VGPR. Must be called
// only after the corresponding patch has been emitted successfully.
void commitScratchVgpr(PatchContext &Ctx, const ScratchAlloc &Alloc) {
  if (Alloc.ExtraVgprsNeeded == 0 || Alloc.KernelName.empty())
    return;
  KernelPatchStats &Stats = Ctx.KernelStats[Alloc.KernelName];
  Stats.ExtraVgprs = std::max(Stats.ExtraVgprs, Alloc.ExtraVgprsNeeded);
  Stats.ScratchAboveKd += Alloc.ExtraVgprsNeeded;
}

// -- scratch-SGPR allocation ------------------------------------------------
//
// Allocate a scratch SGPR above the kernel's .sgpr_count. Those SGPRs are
// never used by the kernel, and GFX10+ waves always have the full SGPR file
// (no KD bump needed), so unlike VGPRs this needs no liveness. Same strategy
// the E5M3 patch uses.
//
struct SgprScratchAlloc {
  unsigned Sgpr = 0;
  std::string KernelName;
  unsigned ExtraSgprsNeeded = 0;
};

std::optional<SgprScratchAlloc>
tryAllocScratchSgpr(PatchContext &Ctx, size_t Idx,
                    ArrayRef<unsigned> ExcludedSgprs = {}) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdSgprs = Ctx.Elf.getKernelSgprCount(KernelName);
  unsigned SgprKdCount = KdSgprs.value_or(Ctx.Config.MaxSgprs);

  SgprAllocator Alloc(SgprKdCount, Ctx.Config.MaxSgprs);
  while (std::optional<unsigned> S = Alloc.alloc()) {
    if (llvm::is_contained(ExcludedSgprs, *S))
      continue;

    SgprScratchAlloc Out;
    Out.Sgpr = *S;
    Out.KernelName = std::move(KernelName);
    Out.ExtraSgprsNeeded = Alloc.extraSgprsNeeded();
    return Out;
  }

  return std::nullopt;
}

void commitScratchSgpr(PatchContext &Ctx, const SgprScratchAlloc &Alloc) {
  if (Alloc.ExtraSgprsNeeded == 0 || Alloc.KernelName.empty())
    return;
  KernelPatchStats &Stats = Ctx.KernelStats[Alloc.KernelName];
  Stats.ExtraSgprs = std::max(Stats.ExtraSgprs, Alloc.ExtraSgprsNeeded);
}

// -- tensor descriptor must analysis ---------------------------------------

std::optional<uint64_t> applyTensorSignedPcDelta(uint64_t CapturedPc,
                                                 int64_t Delta) {
  if (Delta >= 0) {
    uint64_t Magnitude = static_cast<uint64_t>(Delta);
    if (CapturedPc > std::numeric_limits<uint64_t>::max() - Magnitude)
      return std::nullopt;
    return CapturedPc + Magnitude;
  }

  uint64_t Magnitude = Delta == std::numeric_limits<int64_t>::min()
                           ? uint64_t{1} << 63
                           : static_cast<uint64_t>(-Delta);
  if (CapturedPc < Magnitude)
    return std::nullopt;
  return CapturedPc - Magnitude;
}

std::optional<uint64_t>
evaluateTensorDirectControlFlowTarget(const InternalDecodedInst &DI,
                                      const LLVMState &LS) {
  uint64_t Target = 0;
  if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target))
    return Target;
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  if (DI.Inst.getOpcode() != LS.SCallI64Opcode || !Desc.isCall() ||
      DI.Inst.getNumOperands() != 2 || !DI.Inst.getOperand(0).isReg() ||
      !DI.Inst.getOperand(0).getReg() || !DI.Inst.getOperand(1).isImm())
    return std::nullopt;

  const uint64_t Encoded =
      static_cast<uint64_t>(DI.Inst.getOperand(1).getImm()) & 0xffffu;
  const int64_t DwordDelta = Encoded < 0x8000u
                                 ? static_cast<int64_t>(Encoded)
                                 : static_cast<int64_t>(Encoded) - 0x10000;
  std::optional<uint64_t> PcBase = checkedAddUint64(
      DI.Offset, DI.Size, "tensor direct control-flow PC base");
  if (!PcBase)
    return std::nullopt;
  if (DwordDelta >= 0)
    return checkedAddUint64(*PcBase,
                            static_cast<uint64_t>(DwordDelta) * MinInstSize,
                            "tensor direct control-flow target");
  return checkedSubUint64(*PcBase,
                          static_cast<uint64_t>(-DwordDelta) * MinInstSize,
                          "tensor direct control-flow target");
}

std::optional<std::pair<MCRegister, int64_t>>
getTensorHotswapAddSetPc(ArrayRef<InternalDecodedInst> Decoded,
                         size_t SetPcIndex) {
  if (SetPcIndex == 0 || SetPcIndex >= Decoded.size())
    return std::nullopt;

  const InternalDecodedInst &Add = Decoded[SetPcIndex - 1];
  const InternalDecodedInst &SetPc = Decoded[SetPcIndex];
  if (Add.Mnemonic != "s_add_nc_u64" || SetPc.Mnemonic != "s_set_pc_i64" ||
      Add.Offset > std::numeric_limits<uint64_t>::max() - Add.Size ||
      Add.Offset + Add.Size != SetPc.Offset || Add.Inst.getNumOperands() != 3 ||
      SetPc.Inst.getNumOperands() != 1 || !Add.Inst.getOperand(0).isReg() ||
      !Add.Inst.getOperand(1).isReg() || !Add.Inst.getOperand(2).isImm() ||
      !SetPc.Inst.getOperand(0).isReg())
    return std::nullopt;

  MCRegister Pair = Add.Inst.getOperand(0).getReg();
  if (!Pair.isValid() || Add.Inst.getOperand(1).getReg() != Pair.id() ||
      SetPc.Inst.getOperand(0).getReg() != Pair.id())
    return std::nullopt;
  return std::pair<MCRegister, int64_t>{Pair, Add.Inst.getOperand(2).getImm()};
}

std::optional<size_t>
findTensorDecodedIndex(ArrayRef<InternalDecodedInst> Decoded, uint64_t Offset) {
  auto It = llvm::lower_bound(Decoded, Offset,
                              [](const InternalDecodedInst &DI,
                                 uint64_t Value) { return DI.Offset < Value; });
  if (It == Decoded.end() || It->Offset != Offset)
    return std::nullopt;
  return It - Decoded.begin();
}

std::optional<uint64_t>
resolveTensorContiguousSetPc(ArrayRef<InternalDecodedInst> Decoded,
                             size_t SetPcIndex,
                             const DenseSet<uint64_t> &DirectTargets) {
  std::optional<std::pair<MCRegister, int64_t>> AddSet =
      getTensorHotswapAddSetPc(Decoded, SetPcIndex);
  if (!AddSet || SetPcIndex < 2)
    return std::nullopt;

  const InternalDecodedInst &GetPc = Decoded[SetPcIndex - 2];
  const InternalDecodedInst &Add = Decoded[SetPcIndex - 1];
  const InternalDecodedInst &SetPc = Decoded[SetPcIndex];
  if (GetPc.Mnemonic != "s_get_pc_i64" ||
      GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size ||
      GetPc.Offset + GetPc.Size != Add.Offset ||
      GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != AddSet->first.id())
    return std::nullopt;
  if (SetPc.Offset > std::numeric_limits<uint64_t>::max() - SetPc.Size)
    return std::nullopt;
  const uint64_t SequenceEnd = SetPc.Offset + SetPc.Size;
  if (llvm::any_of(DirectTargets, [&](uint64_t Target) {
        return Target > GetPc.Offset && Target < SequenceEnd;
      }))
    return std::nullopt;
  return applyTensorSignedPcDelta(GetPc.Offset + GetPc.Size, AddSet->second);
}

std::optional<uint64_t>
resolveTensorSetPcTarget(ArrayRef<InternalDecodedInst> AllDecoded,
                         uint64_t SetPcOffset,
                         const DenseSet<uint64_t> &DirectTargets) {
  std::optional<size_t> Index = findTensorDecodedIndex(AllDecoded, SetPcOffset);
  if (!Index)
    return std::nullopt;
  return resolveTensorContiguousSetPc(AllDecoded, *Index, DirectTargets);
}

struct TensorTrampolinePath {
  uint64_t ResumeOffset = 0;
  SmallVector<size_t, 16> Instructions;
};

std::optional<TensorTrampolinePath>
findTensorTrampolinePath(ArrayRef<InternalDecodedInst> AllDecoded,
                         uint64_t Target, const TensorAnalysisRange &Range,
                         size_t ExternalBegin, const LLVMState &LS,
                         const DenseSet<uint64_t> &DirectTargets) {
  if (!LS.MIA)
    return std::nullopt;

  TensorTrampolinePath Result;
  // The split-relay resolver can produce a destination that already lies in
  // the candidate range.  Re-enter it directly: walking from that address as
  // if it were external would incorrectly compress candidate instructions
  // (including a tensor load) into the trampoline edge.
  if (Target >= Range.Begin && Target < Range.End) {
    Result.ResumeOffset = Target;
    return Result;
  }
  size_t RemainingInstructions = AllDecoded.size();
  DenseSet<uint64_t> VisitedOffsets;
  auto SegmentHasInteriorEntry = [&](uint64_t Begin, uint64_t End) {
    return llvm::any_of(
               DirectTargets,
               [&](uint64_t Entry) { return Entry > Begin && Entry < End; }) ||
           llvm::any_of(Range.ForeignExternalEntries, [&](uint64_t Entry) {
             return Entry > Begin && Entry < End;
           });
  };
  while (RemainingInstructions != 0) {
    if (llvm::is_contained(Range.ForeignExternalEntries, Target))
      return std::nullopt;
    std::optional<size_t> Start = findTensorDecodedIndex(AllDecoded, Target);
    // Instructions in another original-.text function may have independent
    // callable roots and predecessors carrying arbitrary descriptor state.
    // Only candidate instructions (handled by the immediate resume above) or
    // appended executable trampoline code can be candidate-owned here.
    if (!Start || *Start < ExternalBegin)
      return std::nullopt;

    const uint64_t SegmentStart = Target;
    uint64_t ExpectedOffset = Target;
    bool FollowedHop = false;
    for (size_t I = *Start; I < AllDecoded.size(); ++I) {
      const InternalDecodedInst &DI = AllDecoded[I];
      if (RemainingInstructions == 0)
        return std::nullopt;
      --RemainingInstructions;
      if (!VisitedOffsets.insert(DI.Offset).second ||
          DI.Offset != ExpectedOffset || DI.Size == 0 ||
          DI.Offset > std::numeric_limits<uint64_t>::max() - DI.Size)
        return std::nullopt;
      ExpectedOffset = DI.Offset + DI.Size;
      Result.Instructions.push_back(I);
      if (DI.Mnemonic == "<unknown>")
        return std::nullopt;

      if (DI.Mnemonic == "s_set_pc_i64") {
        if (I < 2 || AllDecoded[I - 2].Offset < SegmentStart)
          return std::nullopt;
        std::optional<uint64_t> Next =
            resolveTensorContiguousSetPc(AllDecoded, I, DirectTargets);
        if (!Next)
          return std::nullopt;
        if (*Next >= Range.Begin && *Next < Range.End) {
          if (SegmentHasInteriorEntry(SegmentStart, ExpectedOffset))
            return std::nullopt;
          Result.ResumeOffset = *Next;
          return Result;
        }
        if (SegmentHasInteriorEntry(SegmentStart, ExpectedOffset))
          return std::nullopt;
        Target = *Next;
        FollowedHop = true;
        break;
      }

      const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
      const bool IsCall = LS.MIA->isCall(DI.Inst);
      const bool IsReturn = LS.MIA->isReturn(DI.Inst);
      const bool IsBranch = LS.MIA->isBranch(DI.Inst);
      if (IsCall || IsReturn || Desc.isTrap())
        return std::nullopt;
      if (IsBranch) {
        uint64_t Next = 0;
        if (!LS.MIA->isUnconditionalBranch(DI.Inst) ||
            LS.MIA->isIndirectBranch(DI.Inst) ||
            !LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Next))
          return std::nullopt;
        if (Next >= Range.Begin && Next < Range.End) {
          if (SegmentHasInteriorEntry(SegmentStart, ExpectedOffset))
            return std::nullopt;
          Result.ResumeOffset = Next;
          return Result;
        }
        if (SegmentHasInteriorEntry(SegmentStart, ExpectedOffset))
          return std::nullopt;
        Target = Next;
        FollowedHop = true;
        break;
      }
      if (Desc.isTerminator() || LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))
        return std::nullopt;
    }
    if (!FollowedHop)
      return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::pair<uint64_t, SmallVector<size_t, 2>>>
resolveTensorRelayTarget(ArrayRef<InternalDecodedInst> AllDecoded,
                         const InternalDecodedInst &GetPc, uint64_t RelayOffset,
                         const DenseSet<uint64_t> &DirectTargets) {
  std::optional<size_t> AddIndex =
      findTensorDecodedIndex(AllDecoded, RelayOffset);
  if (!AddIndex || *AddIndex + 1 >= AllDecoded.size())
    return std::nullopt;

  const size_t SetPcIndex = *AddIndex + 1;
  std::optional<std::pair<MCRegister, int64_t>> AddSet =
      getTensorHotswapAddSetPc(AllDecoded, SetPcIndex);
  if (!AddSet || GetPc.Mnemonic != "s_get_pc_i64" ||
      GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != AddSet->first.id() ||
      GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size)
    return std::nullopt;
  const InternalDecodedInst &SetPc = AllDecoded[SetPcIndex];
  if (SetPc.Offset > std::numeric_limits<uint64_t>::max() - SetPc.Size)
    return std::nullopt;
  const uint64_t SequenceEnd = SetPc.Offset + SetPc.Size;
  if (llvm::any_of(DirectTargets, [&](uint64_t Target) {
        return Target > RelayOffset && Target < SequenceEnd;
      }))
    return std::nullopt;

  std::optional<uint64_t> Target =
      applyTensorSignedPcDelta(GetPc.Offset + GetPc.Size, AddSet->second);
  if (!Target)
    return std::nullopt;
  return std::pair<uint64_t, SmallVector<size_t, 2>>{
      *Target, SmallVector<size_t, 2>{*AddIndex, SetPcIndex}};
}

std::optional<uint64_t> resolveTensorCarrySetPc(
    ArrayRef<InternalDecodedInst> Decoded, size_t SetPcIndex,
    const DenseSet<uint64_t> &DirectTargets, const LLVMState &LS) {
  if (!LS.MRI || SetPcIndex < 3 || SetPcIndex >= Decoded.size())
    return std::nullopt;

  const InternalDecodedInst &GetPc = Decoded[SetPcIndex - 3];
  const InternalDecodedInst &AddLo = Decoded[SetPcIndex - 2];
  const InternalDecodedInst &AddHi = Decoded[SetPcIndex - 1];
  const InternalDecodedInst &SetPc = Decoded[SetPcIndex];
  if (GetPc.Mnemonic != "s_get_pc_i64" || AddLo.Mnemonic != "s_add_u32" ||
      AddHi.Mnemonic != "s_addc_u32" || SetPc.Mnemonic != "s_set_pc_i64" ||
      GetPc.Inst.getNumOperands() != 1 || AddLo.Inst.getNumOperands() != 3 ||
      AddHi.Inst.getNumOperands() != 3 || SetPc.Inst.getNumOperands() != 1 ||
      !GetPc.Inst.getOperand(0).isReg() || !AddLo.Inst.getOperand(0).isReg() ||
      !AddLo.Inst.getOperand(1).isReg() || !AddLo.Inst.getOperand(2).isImm() ||
      !AddHi.Inst.getOperand(0).isReg() || !AddHi.Inst.getOperand(1).isReg() ||
      !AddHi.Inst.getOperand(2).isImm() || !SetPc.Inst.getOperand(0).isReg())
    return std::nullopt;

  auto IsContiguous = [](const InternalDecodedInst &L,
                         const InternalDecodedInst &R) {
    return L.Offset <= std::numeric_limits<uint64_t>::max() - L.Size &&
           L.Offset + L.Size == R.Offset;
  };
  if (!IsContiguous(GetPc, AddLo) || !IsContiguous(AddLo, AddHi) ||
      !IsContiguous(AddHi, SetPc))
    return std::nullopt;

  MCRegister Pair = GetPc.Inst.getOperand(0).getReg();
  MCRegister Lo = AddLo.Inst.getOperand(0).getReg();
  MCRegister Hi = AddHi.Inst.getOperand(0).getReg();
  if (!Pair.isValid() || !Lo.isValid() || !Hi.isValid() ||
      SetPc.Inst.getOperand(0).getReg() != Pair.id() ||
      AddLo.Inst.getOperand(1).getReg() != Lo.id() ||
      AddHi.Inst.getOperand(1).getReg() != Hi.id())
    return std::nullopt;
  const unsigned LoSubReg = LS.MRI->getSubRegIndex(Pair, Lo);
  const unsigned HiSubReg = LS.MRI->getSubRegIndex(Pair, Hi);
  if (LoSubReg == 0 || HiSubReg == 0 || LoSubReg >= HiSubReg)
    return std::nullopt;

  std::optional<uint64_t> SequenceEnd = checkedAddUint64(
      SetPc.Offset, SetPc.Size, "tensor carry set-PC sequence end");
  std::optional<uint64_t> CapturedPc = checkedAddUint64(
      GetPc.Offset, GetPc.Size, "tensor carry set-PC captured PC");
  if (!SequenceEnd || !CapturedPc ||
      llvm::any_of(DirectTargets, [&](uint64_t Target) {
        return Target > GetPc.Offset && Target < *SequenceEnd;
      }))
    return std::nullopt;

  const uint64_t LoImm =
      static_cast<uint32_t>(AddLo.Inst.getOperand(2).getImm());
  const uint64_t HiImm =
      static_cast<uint32_t>(AddHi.Inst.getOperand(2).getImm());
  return *CapturedPc + (LoImm | (HiImm << 32));
}

struct TensorExternalCfgEdge {
  size_t SourceIndex = 0;
  uint64_t Target = 0;
  std::optional<unsigned> DispatchStubIndex;
};

struct TensorExternalCfg {
  bool Valid = false;
  size_t ExternalBegin = 0;
  std::vector<TensorExternalCfgEdge> Edges;
  BitVector UnresolvedSources;
};

struct TensorPcSequence {
  uint64_t Begin = 0;
  uint64_t End = 0;
  bool RequiresRelaySource = false;
  SmallVector<size_t, 2> RelaySources;
};

TensorExternalCfg buildTensorExternalCfg(
    ArrayRef<InternalDecodedInst> Decoded,
    ArrayRef<InternalDecodedInst> AllDecoded,
    ArrayRef<TensorDispatchStub> DispatchStubs,
    ArrayRef<uint64_t> VirtualExternalEntries,
    ArrayRef<std::pair<uint64_t, uint64_t>> OriginalControlFlowEdges,
    ArrayRef<uint64_t> OriginalCodeEntries, const LLVMState &LS,
    const DenseSet<uint64_t> &DirectTargets) {
  TensorExternalCfg Result;
  Result.UnresolvedSources = BitVector(AllDecoded.size());
  if (!LS.MIA || !LS.MCII || !LS.MRI || AllDecoded.size() < Decoded.size())
    return Result;
  for (size_t I = 0; I < Decoded.size(); ++I)
    if (AllDecoded[I].Offset != Decoded[I].Offset ||
        AllDecoded[I].Size != Decoded[I].Size)
      return Result;
  Result.ExternalBegin = Decoded.size();

  for (uint64_t Root : VirtualExternalEntries) {
    std::optional<size_t> RootIndex = findTensorDecodedIndex(AllDecoded, Root);
    if (!RootIndex || *RootIndex < Result.ExternalBegin)
      return Result;
    if (llvm::any_of(DispatchStubs, [&](const TensorDispatchStub &Stub) {
          return Root >= Stub.Begin && Root < Stub.End;
        }))
      return Result;
  }

  DenseMap<size_t, unsigned> DispatchTerminals;
  for (unsigned StubIndex = 0; StubIndex < DispatchStubs.size(); ++StubIndex) {
    const TensorDispatchStub &Stub = DispatchStubs[StubIndex];
    std::optional<size_t> Terminal =
        findTensorDecodedIndex(AllDecoded, Stub.Terminal);
    if (!Terminal || *Terminal < Result.ExternalBegin ||
        !DispatchTerminals.try_emplace(*Terminal, StubIndex).second)
      return Result;
  }

  DenseMap<size_t, uint64_t> RelayTargets;
  DenseMap<size_t, SmallVector<size_t, 2>> RelaySources;
  DenseSet<size_t> ConflictingRelays;
  for (size_t I = 1; I < AllDecoded.size(); ++I) {
    const InternalDecodedInst &Branch = AllDecoded[I];
    const InternalDecodedInst &GetPc = AllDecoded[I - 1];
    if (Branch.Mnemonic != "s_branch" || GetPc.Mnemonic != "s_get_pc_i64" ||
        GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size ||
        GetPc.Offset + GetPc.Size != Branch.Offset)
      continue;
    std::optional<uint64_t> RelayOffset =
        evaluateTensorDirectControlFlowTarget(Branch, LS);
    if (!RelayOffset)
      continue;
    auto Relay = resolveTensorRelayTarget(AllDecoded, GetPc, *RelayOffset,
                                          DirectTargets);
    if (!Relay || Relay->second.empty())
      continue;
    const size_t Terminal = Relay->second.back();
    auto [It, Inserted] = RelayTargets.try_emplace(Terminal, Relay->first);
    if (!Inserted && It->second != Relay->first)
      ConflictingRelays.insert(Terminal);
    RelaySources[Terminal].push_back(I);
  }
  if (!ConflictingRelays.empty())
    return Result;

  std::vector<TensorPcSequence> PcSequences;
  for (size_t I = 0; I < AllDecoded.size(); ++I) {
    if (AllDecoded[I].Mnemonic != "s_set_pc_i64" ||
        DispatchTerminals.contains(I))
      continue;
    std::optional<uint64_t> Target =
        resolveTensorContiguousSetPc(AllDecoded, I, DirectTargets);
    size_t BeginIndex = 0;
    bool RequiresRelaySource = false;
    if (Target) {
      BeginIndex = I - 2;
    } else {
      Target = resolveTensorCarrySetPc(AllDecoded, I, DirectTargets, LS);
      if (Target) {
        BeginIndex = I - 3;
      } else {
        auto Relay = RelayTargets.find(I);
        if (Relay == RelayTargets.end())
          continue;
        Target = Relay->second;
        BeginIndex = I - 1;
        RequiresRelaySource = true;
      }
    }
    std::optional<uint64_t> End =
        checkedAddUint64(AllDecoded[I].Offset, AllDecoded[I].Size,
                         "tensor computed-PC sequence end");
    if (!End || BeginIndex >= I || AllDecoded[BeginIndex].Offset >= *End)
      return Result;
    TensorPcSequence Sequence{AllDecoded[BeginIndex].Offset, *End,
                              RequiresRelaySource};
    if (RequiresRelaySource)
      Sequence.RelaySources = RelaySources[I];
    PcSequences.push_back(std::move(Sequence));
  }
  for (const auto &[Terminal, Sources] : RelaySources)
    for (size_t Source : Sources) {
      if (Source == 0 || Source >= AllDecoded.size())
        return Result;
      std::optional<uint64_t> End =
          checkedAddUint64(AllDecoded[Source].Offset, AllDecoded[Source].Size,
                           "tensor relay source sequence end");
      if (!End)
        return Result;
      PcSequences.push_back({AllDecoded[Source - 1].Offset, *End, false, {}});
    }

  auto AddEdge = [&](size_t Source, uint64_t Target,
                     std::optional<unsigned> DispatchStubIndex = std::nullopt) {
    Result.Edges.push_back({Source, Target, DispatchStubIndex});
  };
  auto AddFallthrough = [&](size_t I) {
    if (I + 1 >= AllDecoded.size() ||
        AllDecoded[I].Offset >
            std::numeric_limits<uint64_t>::max() - AllDecoded[I].Size ||
        AllDecoded[I].Offset + AllDecoded[I].Size != AllDecoded[I + 1].Offset)
      return false;
    AddEdge(I, AllDecoded[I + 1].Offset);
    return true;
  };

  for (size_t I = Result.ExternalBegin; I < AllDecoded.size(); ++I) {
    const InternalDecodedInst &DI = AllDecoded[I];
    auto Dispatch = DispatchTerminals.find(I);
    if (Dispatch != DispatchTerminals.end()) {
      AddEdge(I, DispatchStubs[Dispatch->second].Target, Dispatch->second);
      continue;
    }

    if (DI.Mnemonic == "<unknown>") {
      Result.UnresolvedSources.set(I);
      continue;
    }
    if (StringRef(DI.Mnemonic).starts_with("s_rfe")) {
      Result.UnresolvedSources.set(I);
      continue;
    }
    if (DI.Mnemonic == "s_set_pc_i64") {
      std::optional<uint64_t> Target =
          resolveTensorContiguousSetPc(AllDecoded, I, DirectTargets);
      if (!Target)
        Target = resolveTensorCarrySetPc(AllDecoded, I, DirectTargets, LS);
      if (!Target) {
        auto Relay = RelayTargets.find(I);
        if (Relay != RelayTargets.end() && !ConflictingRelays.contains(I))
          Target = Relay->second;
      }
      if (Target)
        AddEdge(I, *Target);
      else
        Result.UnresolvedSources.set(I);
      continue;
    }
    if (DI.Mnemonic == "s_code_end" || DI.Mnemonic == "s_endpgm" ||
        DI.Mnemonic == "s_endpgm_saved")
      continue;

    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    const bool IsCall = LS.MIA->isCall(DI.Inst) || Desc.isCall();
    const bool IsReturn = LS.MIA->isReturn(DI.Inst) || Desc.isReturn();
    const bool IsBranch = LS.MIA->isBranch(DI.Inst) || Desc.isBranch();
    if (IsReturn) {
      Result.UnresolvedSources.set(I);
      continue;
    }
    if (IsBranch || IsCall) {
      if (LS.MIA->isIndirectBranch(DI.Inst) || Desc.isIndirectBranch()) {
        Result.UnresolvedSources.set(I);
        continue;
      }
      std::optional<uint64_t> Target =
          evaluateTensorDirectControlFlowTarget(DI, LS);
      if (!Target) {
        Result.UnresolvedSources.set(I);
        continue;
      }
      AddEdge(I, *Target);
      const bool IsConditional =
          LS.MIA->isConditionalBranch(DI.Inst) || Desc.isConditionalBranch();
      if ((IsCall || IsConditional) && !AddFallthrough(I))
        Result.UnresolvedSources.set(I);
      continue;
    }
    if (Desc.isTrap() || Desc.isTerminator()) {
      Result.UnresolvedSources.set(I);
      continue;
    }
    if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI)) {
      Result.UnresolvedSources.set(I);
      continue;
    }
    if (!AddFallthrough(I))
      Result.UnresolvedSources.set(I);
  }

  // A recognized get-PC/add/set-PC sequence has a statically known target
  // only when execution enters through all of its prerequisites.  An edge or
  // callable root into the middle can reuse an arbitrary pre-existing SGPR
  // value and turn the nominally direct set-PC into an indirect transfer.
  auto HasValidSequenceIngress = [&](std::optional<size_t> SourceIndex,
                                     uint64_t Target) {
    for (const TensorPcSequence &Sequence : PcSequences) {
      if (Target < Sequence.Begin || Target >= Sequence.End)
        continue;

      if (SourceIndex) {
        if (*SourceIndex >= AllDecoded.size())
          return false;
        const InternalDecodedInst &Source = AllDecoded[*SourceIndex];
        const bool IsSequentialPrerequisite =
            Source.Offset >= Sequence.Begin && Source.Offset < Sequence.End &&
            Source.Offset <=
                std::numeric_limits<uint64_t>::max() - Source.Size &&
            Source.Offset + Source.Size == Target;
        if (IsSequentialPrerequisite)
          continue;
        if (Target == Sequence.Begin &&
            (!Sequence.RequiresRelaySource ||
             llvm::is_contained(Sequence.RelaySources, *SourceIndex)))
          continue;
      } else if (Target == Sequence.Begin && !Sequence.RequiresRelaySource) {
        continue;
      }
      return false;
    }
    return true;
  };

  // Every resolved edge must land on an exact decoded instruction boundary.
  // Otherwise the target is outside the analyzed graph (or in the middle of
  // an instruction), so treating it as a closed direct edge would be unsound.
  for (const TensorExternalCfgEdge &Edge : Result.Edges) {
    if (Edge.SourceIndex >= AllDecoded.size() ||
        !findTensorDecodedIndex(AllDecoded, Edge.Target) ||
        !HasValidSequenceIngress(Edge.SourceIndex, Edge.Target))
      return TensorExternalCfg{};
  }
  for (const auto &[Source, Target] : OriginalControlFlowEdges) {
    std::optional<size_t> SourceIndex =
        findTensorDecodedIndex(AllDecoded, Source);
    if (!SourceIndex || *SourceIndex >= Result.ExternalBegin ||
        !findTensorDecodedIndex(AllDecoded, Target) ||
        !HasValidSequenceIngress(*SourceIndex, Target))
      return TensorExternalCfg{};
  }
  for (uint64_t Root : OriginalCodeEntries) {
    std::optional<size_t> RootIndex = findTensorDecodedIndex(AllDecoded, Root);
    if (!RootIndex || *RootIndex >= Result.ExternalBegin ||
        !HasValidSequenceIngress(std::nullopt, Root))
      return TensorExternalCfg{};
  }
  for (uint64_t Root : VirtualExternalEntries)
    if (!HasValidSequenceIngress(std::nullopt, Root))
      return TensorExternalCfg{};

  for (const TensorExternalCfgEdge &Edge : Result.Edges)
    for (const TensorDispatchStub &Stub : DispatchStubs)
      if (Edge.Target >= Stub.Begin && Edge.Target < Stub.End) {
        const uint64_t Source = AllDecoded[Edge.SourceIndex].Offset;
        if (Source < Stub.Begin || Source >= Stub.End)
          return TensorExternalCfg{};
      }
  Result.Valid = true;
  return Result;
}

std::optional<unsigned> getNumberedRegFactIndex(MCRegister Reg,
                                                const MCRegisterInfo &MRI,
                                                unsigned MaxSgprs,
                                                unsigned MaxVgprs) {
  const char *RawName = MRI.getName(Reg);
  if (!RawName)
    return std::nullopt;
  StringRef Name(RawName);
  if (Name.contains('_'))
    return std::nullopt;

  unsigned Index = 0;
  if (Name.consume_front("SGPR")) {
    if (Name.getAsInteger(10, Index) || Index >= MaxSgprs)
      return std::nullopt;
    return Index;
  }
  if (Name.consume_front("VGPR")) {
    if (Name.getAsInteger(10, Index) || Index >= MaxVgprs)
      return std::nullopt;
    return MaxSgprs + Index;
  }
  return std::nullopt;
}

std::optional<unsigned> getTrue16ParentFactIndex(MCRegister Reg,
                                                 const MCRegisterInfo &MRI,
                                                 unsigned MaxSgprs,
                                                 unsigned MaxVgprs) {
  StringRef Name = MRI.getName(Reg);
  const bool IsLowHalf = Name.ends_with("_LO16");
  const bool IsHighHalf = Name.ends_with("_HI16");
  if (!IsLowHalf && !IsHighHalf)
    return std::nullopt;

  unsigned Index = 0;
  if (Name.consume_front("SGPR")) {
    Name = Name.take_until([](char C) { return C == '_'; });
    if (!Name.getAsInteger(10, Index) && Index < MaxSgprs)
      return Index;
    return std::nullopt;
  }
  if (Name.consume_front("VGPR")) {
    Name = Name.take_until([](char C) { return C == '_'; });
    if (!Name.getAsInteger(10, Index) && Index < MaxVgprs)
      return MaxSgprs + Index;
  }
  return std::nullopt;
}

SmallVector<MCRegister, 4> getNumberedRegLeaves(MCRegister Reg,
                                                const MCRegisterInfo &MRI,
                                                unsigned MaxSgprs,
                                                unsigned MaxVgprs) {
  if (getNumberedRegFactIndex(Reg, MRI, MaxSgprs, MaxVgprs))
    return {Reg};

  SmallVector<MCRegister, 4> Leaves;
  for (MCRegister Sub : getDirectSubRegs(Reg, MRI))
    if (getNumberedRegFactIndex(Sub, MRI, MaxSgprs, MaxVgprs))
      Leaves.push_back(Sub);
  return Leaves;
}

bool operandLow16KnownZero(const MCOperand &Op, const BitVector &State,
                           const MCRegisterInfo &MRI, unsigned MaxSgprs,
                           unsigned MaxVgprs) {
  if (Op.isImm())
    return (static_cast<uint64_t>(Op.getImm()) & 0xffff) == 0;
  if (!Op.isReg() || !Op.getReg())
    return false;
  std::optional<unsigned> Fact =
      getNumberedRegFactIndex(MCRegister(Op.getReg()), MRI, MaxSgprs, MaxVgprs);
  return Fact && State.test(*Fact);
}

void setRegLow16Fact(BitVector &State, MCRegister Reg, bool IsKnownZero,
                     const MCRegisterInfo &MRI, unsigned MaxSgprs,
                     unsigned MaxVgprs) {
  if (std::optional<unsigned> Fact =
          getNumberedRegFactIndex(Reg, MRI, MaxSgprs, MaxVgprs)) {
    if (IsKnownZero)
      State.set(*Fact);
    else
      State.reset(*Fact);
    return;
  }

  // True16 defs do not enumerate their 32-bit parent as a direct subregister.
  // Conservatively kill the parent's fact for either half. This includes
  // partial SGPR defs: retaining an SGPR descriptor fact across an overlapping
  // subregister write would make the tensor proof unsound.
  if (std::optional<unsigned> Parent =
          getTrue16ParentFactIndex(Reg, MRI, MaxSgprs, MaxVgprs)) {
    State.reset(*Parent);
    return;
  }

  for (MCRegister Sub : getDirectSubRegs(Reg, MRI))
    setRegLow16Fact(State, Sub, IsKnownZero, MRI, MaxSgprs, MaxVgprs);
}

bool tensorInstructionDefinesExec(const MCInst &Inst, const MCInstrDesc &Desc,
                                  const MCRegisterInfo &MRI) {
  auto IsExec = [&](MCRegister Reg) {
    StringRef Name = MRI.getName(Reg);
    return Name == "EXEC" || Name == "EXEC_LO" || Name == "EXEC_HI";
  };
  const unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = 0; I < NumDefs; ++I)
    if (Inst.getOperand(I).isReg() && Inst.getOperand(I).getReg() &&
        IsExec(MCRegister(Inst.getOperand(I).getReg())))
      return true;
  return llvm::any_of(Desc.implicit_defs(),
                      [&](MCPhysReg Reg) { return IsExec(MCRegister(Reg)); });
}

BitVector transferTensorDescriptorFacts(const InternalDecodedInst &DI,
                                        const BitVector &Input,
                                        const LLVMState &LS, unsigned MaxSgprs,
                                        unsigned MaxVgprs) {
  const MCInst &Inst = DI.Inst;
  const MCInstrDesc &Desc = LS.MCII->get(Inst.getOpcode());
  const MCRegisterInfo &MRI = *LS.MRI;
  const unsigned ExecNonemptyFact = MaxSgprs + MaxVgprs;
  BitVector EffectiveInput = Input;
  if (tensorInstructionDefinesExec(Inst, Desc, MRI)) {
    EffectiveInput.reset(MaxSgprs, MaxSgprs + MaxVgprs);
    EffectiveInput.reset(ExecNonemptyFact);
  }
  BitVector Output = EffectiveInput;

  const unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = 0; I < NumDefs; ++I) {
    const MCOperand &Def = Inst.getOperand(I);
    if (Def.isReg() && Def.getReg())
      setRegLow16Fact(Output, MCRegister(Def.getReg()), false, MRI, MaxSgprs,
                      MaxVgprs);
  }
  for (MCPhysReg Def : Desc.implicit_defs())
    setRegLow16Fact(Output, MCRegister(Def), false, MRI, MaxSgprs, MaxVgprs);

  auto CopyOne = [&](unsigned DefOp, unsigned SourceOp) {
    if (DefOp >= Inst.getNumOperands() || SourceOp >= Inst.getNumOperands() ||
        !Inst.getOperand(DefOp).isReg() || !Inst.getOperand(DefOp).getReg())
      return;
    MCRegister Dst = MCRegister(Inst.getOperand(DefOp).getReg());
    std::optional<unsigned> DstFact =
        getNumberedRegFactIndex(Dst, MRI, MaxSgprs, MaxVgprs);
    const bool FullNumberedDestination = DstFact.has_value();
    const bool VgprDestination = DstFact && *DstFact >= MaxSgprs;
    bool Known = FullNumberedDestination &&
                 (!VgprDestination || EffectiveInput.test(ExecNonemptyFact)) &&
                 operandLow16KnownZero(Inst.getOperand(SourceOp),
                                       EffectiveInput, MRI, MaxSgprs, MaxVgprs);
    setRegLow16Fact(Output, Dst, Known, MRI, MaxSgprs, MaxVgprs);
  };

  if ((DI.Mnemonic == "s_mov_b32" || DI.Mnemonic == "v_mov_b32") &&
      Inst.getNumOperands() == 2) {
    CopyOne(0, 1);
  } else if (DI.Mnemonic == "s_mov_b64" && Inst.getNumOperands() >= 2 &&
             Inst.getOperand(0).isReg() && Inst.getOperand(0).getReg()) {
    SmallVector<MCRegister, 4> Dst = getNumberedRegLeaves(
        MCRegister(Inst.getOperand(0).getReg()), MRI, MaxSgprs, MaxVgprs);
    if (Inst.getOperand(1).isReg() && Inst.getOperand(1).getReg()) {
      SmallVector<MCRegister, 4> Src = getNumberedRegLeaves(
          MCRegister(Inst.getOperand(1).getReg()), MRI, MaxSgprs, MaxVgprs);
      if (Dst.size() == Src.size()) {
        for (unsigned I = 0; I < Dst.size(); ++I) {
          std::optional<unsigned> SourceFact =
              getNumberedRegFactIndex(Src[I], MRI, MaxSgprs, MaxVgprs);
          setRegLow16Fact(Output, Dst[I],
                          SourceFact && EffectiveInput.test(*SourceFact), MRI,
                          MaxSgprs, MaxVgprs);
        }
      }
    }
  } else if (DI.Mnemonic == "v_dual_mov_b32" && NumDefs == 2 &&
             Inst.getNumOperands() == 4) {
    CopyOne(0, 2);
    CopyOne(1, 3);
  } else if (DI.Mnemonic == "v_readfirstlane_b32") {
    if (EffectiveInput.test(ExecNonemptyFact))
      CopyOne(0, 1);
  } else if (DI.Mnemonic == "s_pack_hh_b32_b16" && Inst.getNumOperands() >= 2 &&
             Inst.getOperand(0).isReg() && Inst.getOperand(0).getReg() &&
             Inst.getOperand(1).isImm() && Inst.getOperand(1).getImm() == 0) {
    setRegLow16Fact(Output, MCRegister(Inst.getOperand(0).getReg()), true, MRI,
                    MaxSgprs, MaxVgprs);
  }

  return Output;
}

static constexpr uint64_t TensorMaskDefTop =
    std::numeric_limits<uint64_t>::max();
static constexpr uint64_t TensorMaskDefUnknown = TensorMaskDefTop - 1;
using TensorMaskDefState = SmallVector<uint64_t, 8>;

TensorMaskDefState transferTensorMaskDefinitions(
    const InternalDecodedInst &DI, const TensorMaskDefState &Input,
    const DenseMap<unsigned, unsigned> &TrackedSgprs, const LLVMState &LS,
    unsigned MaxSgprs, unsigned MaxVgprs) {
  TensorMaskDefState Output = Input;
  const MCInst &Inst = DI.Inst;
  const MCInstrDesc &Desc = LS.MCII->get(Inst.getOpcode());
  const MCRegisterInfo &MRI = *LS.MRI;

  auto KillReg = [&](MCRegister Reg) {
    if (std::optional<unsigned> Parent =
            getTrue16ParentFactIndex(Reg, MRI, MaxSgprs, MaxVgprs)) {
      auto Slot = TrackedSgprs.find(*Parent);
      if (Slot != TrackedSgprs.end())
        Output[Slot->second] = TensorMaskDefUnknown;
      return;
    }
    for (MCRegister Leaf : getNumberedRegLeaves(Reg, MRI, MaxSgprs, MaxVgprs)) {
      std::optional<unsigned> Fact =
          getNumberedRegFactIndex(Leaf, MRI, MaxSgprs, MaxVgprs);
      if (!Fact)
        continue;
      auto Slot = TrackedSgprs.find(*Fact);
      if (Slot != TrackedSgprs.end()) {
        Output[Slot->second] = TensorMaskDefUnknown;
      }
    }
  };

  const unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = 0; I < NumDefs; ++I) {
    const MCOperand &Def = Inst.getOperand(I);
    if (Def.isReg() && Def.getReg())
      KillReg(MCRegister(Def.getReg()));
  }
  for (MCPhysReg Def : Desc.implicit_defs())
    KillReg(MCRegister(Def));

  if (DI.Mnemonic == "v_readfirstlane_b32" && Inst.getNumOperands() >= 1 &&
      Inst.getOperand(0).isReg() && Inst.getOperand(0).getReg()) {
    std::optional<unsigned> Fact = getNumberedRegFactIndex(
        MCRegister(Inst.getOperand(0).getReg()), MRI, MaxSgprs, MaxVgprs);
    if (Fact) {
      auto Slot = TrackedSgprs.find(*Fact);
      if (Slot != TrackedSgprs.end())
        Output[Slot->second] = DI.Offset;
    }
  }
  return Output;
}

void meetTensorMaskDefinitions(TensorMaskDefState &Accumulator,
                               const TensorMaskDefState &Candidate) {
  assert(Accumulator.size() == Candidate.size());
  for (unsigned I = 0; I < Accumulator.size(); ++I) {
    if (Accumulator[I] == TensorMaskDefTop)
      Accumulator[I] = Candidate[I];
    else if (Accumulator[I] != Candidate[I])
      Accumulator[I] = TensorMaskDefUnknown;
  }
}

struct TensorCfgPredecessor {
  unsigned From = 0;
  SmallVector<size_t, 16> ExternalInstructions;
};

void addTensorCfgEdge(
    unsigned From, unsigned To,
    std::vector<SmallVector<unsigned, 2>> &Successors,
    std::vector<SmallVector<TensorCfgPredecessor, 2>> &Predecessors,
    ArrayRef<size_t> ExternalInstructions = {}) {
  if (!llvm::is_contained(Successors[From], To))
    Successors[From].push_back(To);
  if (llvm::any_of(Predecessors[To], [&](const TensorCfgPredecessor &Pred) {
        return Pred.From == From &&
               llvm::equal(Pred.ExternalInstructions, ExternalInstructions);
      }))
    return;
  TensorCfgPredecessor Pred;
  Pred.From = From;
  Pred.ExternalInstructions.append(ExternalInstructions.begin(),
                                   ExternalInstructions.end());
  Predecessors[To].push_back(std::move(Pred));
}

void analyzeTensorDescriptorRange(ArrayRef<InternalDecodedInst> Decoded,
                                  ArrayRef<InternalDecodedInst> AllDecoded,
                                  const TensorExternalCfg &ExternalCfg,
                                  const TensorAnalysisRange &Range,
                                  const LLVMState &LS, unsigned MaxSgprs,
                                  unsigned MaxVgprs,
                                  const DenseSet<uint64_t> &DirectTargets,
                                  TensorDescriptorMustAnalysis &Result,
                                  BitVector &Seen) {
  SmallVector<size_t> GlobalIndices;
  auto First = llvm::lower_bound(
      Decoded, Range.Begin, [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  auto Last = llvm::lower_bound(
      Decoded, Range.End, [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  for (auto It = First; It != Last; ++It)
    GlobalIndices.push_back(It - Decoded.begin());
  if (GlobalIndices.empty() ||
      Decoded[GlobalIndices.front()].Offset != Range.Begin ||
      !llvm::any_of(GlobalIndices, [&](size_t I) {
        return Decoded[I].Mnemonic == "tensor_load_to_lds";
      }))
    return;
  uint64_t ExpectedOffset = Range.Begin;
  for (size_t GlobalIdx : GlobalIndices) {
    const InternalDecodedInst &DI = Decoded[GlobalIdx];
    if (DI.Offset != ExpectedOffset || DI.Size == 0 ||
        DI.Offset > std::numeric_limits<uint64_t>::max() - DI.Size)
      return;
    ExpectedOffset = DI.Offset + DI.Size;
  }
  if (ExpectedOffset != Range.End)
    return;

  const unsigned Count = GlobalIndices.size();
  DenseMap<uint64_t, unsigned> OffsetToLocal;
  for (unsigned I = 0; I < Count; ++I)
    OffsetToLocal.try_emplace(Decoded[GlobalIndices[I]].Offset, I);

  DenseMap<unsigned, unsigned> TrackedSgprs;
  for (size_t GlobalIdx : GlobalIndices) {
    if (Decoded[GlobalIdx].Mnemonic != "tensor_load_to_lds")
      continue;
    MCRegister Base = getDescriptorBaseSgpr(Decoded[GlobalIdx].Inst, *LS.MRI);
    std::optional<unsigned> Fact =
        getNumberedRegFactIndex(Base, *LS.MRI, MaxSgprs, MaxVgprs);
    if (Fact && *Fact < MaxSgprs && !TrackedSgprs.contains(*Fact))
      TrackedSgprs.try_emplace(*Fact, TrackedSgprs.size());
  }

  std::vector<SmallVector<unsigned, 2>> Successors(Count);
  std::vector<SmallVector<TensorCfgPredecessor, 2>> Predecessors(Count);
  BitVector UnknownSuccessors(Count);
  BitVector ModeledExternalInstructions(AllDecoded.size());
  auto RecordModeledExternal = [&](const TensorTrampolinePath &Path) {
    for (size_t ExternalIdx : Path.Instructions)
      if (ExternalIdx < ModeledExternalInstructions.size())
        ModeledExternalInstructions.set(ExternalIdx);
  };
  SmallVector<unsigned> HotswapSetPcCandidates;
  for (unsigned I = 0; I < Count; ++I) {
    const InternalDecodedInst &DI = Decoded[GlobalIndices[I]];
    const bool HasFallthrough = I + 1 < Count;
    if (DI.Mnemonic == "<unknown>" || !LS.MIA) {
      UnknownSuccessors.set(I);
      continue;
    }

    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    const bool IsCall = LS.MIA->isCall(DI.Inst) || Desc.isCall();
    const bool IsReturn = LS.MIA->isReturn(DI.Inst) || Desc.isReturn();
    const bool IsBranch = LS.MIA->isBranch(DI.Inst) || Desc.isBranch();
    if (StringRef(DI.Mnemonic).starts_with("s_rfe")) {
      UnknownSuccessors.set(I);
      continue;
    }
    if (IsReturn)
      continue;
    if (IsCall) {
      UnknownSuccessors.set(I);
      if (!Desc.isTerminator() && HasFallthrough)
        addTensorCfgEdge(I, I + 1, Successors, Predecessors);
      continue;
    }
    if (DI.Mnemonic == "s_set_pc_i64") {
      HotswapSetPcCandidates.push_back(I);
      continue;
    }
    if (Desc.isTrap()) {
      UnknownSuccessors.set(I);
      continue;
    }

    if (IsBranch) {
      const bool IsIndirect =
          LS.MIA->isIndirectBranch(DI.Inst) || Desc.isIndirectBranch();
      const bool IsConditional =
          LS.MIA->isConditionalBranch(DI.Inst) || Desc.isConditionalBranch();
      const bool IsUnconditional = LS.MIA->isUnconditionalBranch(DI.Inst) ||
                                   Desc.isUnconditionalBranch();
      bool TargetKnown = false;
      if (!IsIndirect) {
        uint64_t Target = 0;
        if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target)) {
          TargetKnown = true;
          if (Target >= Range.Begin && Target < Range.End) {
            auto TargetIt = OffsetToLocal.find(Target);
            if (TargetIt == OffsetToLocal.end()) {
              UnknownSuccessors.set(I);
              TargetKnown = false;
            } else {
              addTensorCfgEdge(I, TargetIt->second, Successors, Predecessors);
            }
          } else {
            // An out-of-range destination is not an ordinary function exit:
            // generated HotSwap bodies may re-enter this range. Only a fully
            // decoded unconditional trampoline path can preserve facts.
            TargetKnown = false;
          }
          if (!TargetKnown && IsUnconditional) {
            std::optional<TensorTrampolinePath> Path = findTensorTrampolinePath(
                AllDecoded, Target, Range, ExternalCfg.ExternalBegin, LS,
                DirectTargets);
            if (!Path && I != 0 && DI.Mnemonic == "s_branch" &&
                !DirectTargets.contains(DI.Offset)) {
              const InternalDecodedInst &GetPc = Decoded[GlobalIndices[I - 1]];
              if (GetPc.Offset <=
                      std::numeric_limits<uint64_t>::max() - GetPc.Size &&
                  GetPc.Offset + GetPc.Size == DI.Offset &&
                  GetPc.Mnemonic == "s_get_pc_i64") {
                auto Relay = resolveTensorRelayTarget(AllDecoded, GetPc, Target,
                                                      DirectTargets);
                if (Relay) {
                  Path = findTensorTrampolinePath(
                      AllDecoded, Relay->first, Range,
                      ExternalCfg.ExternalBegin, LS, DirectTargets);
                  if (Path)
                    Path->Instructions.insert(Path->Instructions.begin(),
                                              Relay->second.begin(),
                                              Relay->second.end());
                }
              }
            }
            if (Path) {
              RecordModeledExternal(*Path);
              auto ResumeIt = OffsetToLocal.find(Path->ResumeOffset);
              if (ResumeIt == OffsetToLocal.end()) {
                UnknownSuccessors.set(I);
                TargetKnown = false;
              } else {
                addTensorCfgEdge(I, ResumeIt->second, Successors, Predecessors,
                                 Path->Instructions);
                TargetKnown = true;
              }
            }
          }
        }
      }
      if (!TargetKnown)
        UnknownSuccessors.set(I);
      if (IsConditional) {
        if (HasFallthrough)
          addTensorCfgEdge(I, I + 1, Successors, Predecessors);
        else
          UnknownSuccessors.set(I);
      } else if (!IsUnconditional) {
        UnknownSuccessors.set(I);
      }
      continue;
    }

    if (Desc.isTerminator())
      continue;
    if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI)) {
      UnknownSuccessors.set(I);
      continue;
    }
    if (HasFallthrough)
      addTensorCfgEdge(I, I + 1, Successors, Predecessors);
    else
      UnknownSuccessors.set(I);
  }

  for (unsigned I : HotswapSetPcCandidates) {
    const InternalDecodedInst &SetPc = Decoded[GlobalIndices[I]];
    std::optional<uint64_t> Target =
        resolveTensorSetPcTarget(AllDecoded, SetPc.Offset, DirectTargets);
    if (!Target) {
      UnknownSuccessors.set(I);
      continue;
    }
    if (*Target >= Range.Begin && *Target < Range.End) {
      auto TargetIt = OffsetToLocal.find(*Target);
      if (TargetIt == OffsetToLocal.end())
        UnknownSuccessors.set(I);
      else
        addTensorCfgEdge(I, TargetIt->second, Successors, Predecessors);
      continue;
    }

    std::optional<TensorTrampolinePath> Path =
        findTensorTrampolinePath(AllDecoded, *Target, Range,
                                 ExternalCfg.ExternalBegin, LS, DirectTargets);
    if (!Path) {
      UnknownSuccessors.set(I);
      continue;
    }
    auto ResumeIt = OffsetToLocal.find(Path->ResumeOffset);
    if (ResumeIt == OffsetToLocal.end()) {
      UnknownSuccessors.set(I);
      continue;
    }
    RecordModeledExternal(*Path);
    addTensorCfgEdge(I, ResumeIt->second, Successors, Predecessors,
                     Path->Instructions);
  }

  // A modeled external subgraph is candidate-owned only if every decoded
  // predecessor is candidate-owned too. Virtual KD dispatch is the sole
  // sourceless entry and is admitted only through an exact validated stub.
  auto OffsetTouchesModeledExternal = [&](uint64_t Offset) {
    auto It = llvm::upper_bound(
        AllDecoded, Offset, [](uint64_t Value, const InternalDecodedInst &DI) {
          return Value < DI.Offset;
        });
    if (It == AllDecoded.begin())
      return false;
    --It;
    const size_t Index = It - AllDecoded.begin();
    return Index >= ExternalCfg.ExternalBegin &&
           ModeledExternalInstructions.test(Index) &&
           It->Offset <= std::numeric_limits<uint64_t>::max() - It->Size &&
           Offset >= It->Offset && Offset < It->Offset + It->Size;
  };
  if (llvm::any_of(Range.ForeignExternalEntries, OffsetTouchesModeledExternal))
    return;

  auto CandidateDispatchStubContaining =
      [&](uint64_t Offset) -> const TensorDispatchStub * {
    auto It = llvm::find_if(Range.DispatchStubs, [&](const auto &Stub) {
      return Stub.Target == Range.Begin && Offset >= Stub.Begin &&
             Offset < Stub.End;
    });
    return It == Range.DispatchStubs.end() ? nullptr : &*It;
  };
  if (llvm::any_of(Range.VirtualExternalEntries, [&](uint64_t Root) {
        return OffsetTouchesModeledExternal(Root) ||
               CandidateDispatchStubContaining(Root) != nullptr;
      }))
    return;
  if (ExternalCfg.UnresolvedSources.any())
    return;

  for (const TensorExternalCfgEdge &Edge : ExternalCfg.Edges) {
    const bool SourceModeled =
        ModeledExternalInstructions.test(Edge.SourceIndex);
    const bool TargetsCandidate =
        Edge.Target >= Range.Begin && Edge.Target < Range.End;
    const bool IsCandidateDispatch =
        Edge.DispatchStubIndex &&
        *Edge.DispatchStubIndex < Range.DispatchStubs.size() &&
        Range.DispatchStubs[*Edge.DispatchStubIndex].Target == Range.Begin &&
        Edge.Target == Range.Begin;
    if (TargetsCandidate && !SourceModeled && !IsCandidateDispatch)
      return;
    if (OffsetTouchesModeledExternal(Edge.Target) && !SourceModeled)
      return;

    if (const TensorDispatchStub *Stub =
            CandidateDispatchStubContaining(Edge.Target)) {
      const uint64_t Source = AllDecoded[Edge.SourceIndex].Offset;
      if (Source < Stub->Begin || Source >= Stub->End)
        return;
    }
  }

  BitVector Reachable(Count);
  SmallVector<unsigned> Worklist{0};
  Reachable.set(0);
  for (size_t Next = 0; Next < Worklist.size(); ++Next) {
    unsigned I = Worklist[Next];
    if (UnknownSuccessors.test(I))
      return;
    for (unsigned Succ : Successors[I])
      if (!Reachable.test(Succ)) {
        Reachable.set(Succ);
        Worklist.push_back(Succ);
      }
  }

  const unsigned FactCount = MaxSgprs + MaxVgprs + 1;
  const unsigned ExecNonemptyFact = MaxSgprs + MaxVgprs;
  BitVector Top(FactCount, true);
  BitVector Bottom(FactCount);
  BitVector EntryState = Bottom;
  EntryState.set(ExecNonemptyFact);
  std::vector<BitVector> MustIn(Count, Top);
  std::vector<BitVector> MustOut(Count, Top);
  MustIn[0] = EntryState;
  MustOut[0] = transferTensorDescriptorFacts(
      Decoded[GlobalIndices[0]], EntryState, LS, MaxSgprs, MaxVgprs);

  TensorMaskDefState DefTop(TrackedSgprs.size(), TensorMaskDefTop);
  TensorMaskDefState DefUnknown(TrackedSgprs.size(), TensorMaskDefUnknown);
  std::vector<TensorMaskDefState> DefIn(Count, DefTop);
  std::vector<TensorMaskDefState> DefOut(Count, DefTop);
  DefIn[0] = DefUnknown;
  DefOut[0] =
      transferTensorMaskDefinitions(Decoded[GlobalIndices[0]], DefUnknown,
                                    TrackedSgprs, LS, MaxSgprs, MaxVgprs);

  bool Changed = true;
  unsigned Iterations = 0;
  const unsigned IterationLimit = std::max(Count + 1, FactCount + 1);
  while (Changed && Iterations++ < IterationLimit) {
    Changed = false;
    for (unsigned I = 0; I < Count; ++I) {
      if (!Reachable.test(I))
        continue;
      BitVector NewIn = I == 0 ? EntryState : Top;
      TensorMaskDefState NewDefIn = I == 0 ? DefUnknown : DefTop;
      bool SawPredecessor = I == 0;
      // Node zero also has a virtual dispatch predecessor carrying EntryState.
      // Meet real backedges with that seed instead of treating every loop to
      // the kernel entry as a fresh dispatch with nonempty EXEC.
      for (const TensorCfgPredecessor &Pred : Predecessors[I]) {
        if (!Reachable.test(Pred.From))
          continue;
        SawPredecessor = true;
        BitVector EdgeOut = MustOut[Pred.From];
        TensorMaskDefState EdgeDefOut = DefOut[Pred.From];
        for (size_t ExternalIdx : Pred.ExternalInstructions) {
          EdgeOut = transferTensorDescriptorFacts(
              AllDecoded[ExternalIdx], EdgeOut, LS, MaxSgprs, MaxVgprs);
          EdgeDefOut = transferTensorMaskDefinitions(AllDecoded[ExternalIdx],
                                                     EdgeDefOut, TrackedSgprs,
                                                     LS, MaxSgprs, MaxVgprs);
        }
        NewIn &= EdgeOut;
        meetTensorMaskDefinitions(NewDefIn, EdgeDefOut);
      }
      if (!SawPredecessor) {
        NewIn.reset();
        NewDefIn = DefUnknown;
      }
      BitVector NewOut = transferTensorDescriptorFacts(
          Decoded[GlobalIndices[I]], NewIn, LS, MaxSgprs, MaxVgprs);
      TensorMaskDefState NewDefOut =
          transferTensorMaskDefinitions(Decoded[GlobalIndices[I]], NewDefIn,
                                        TrackedSgprs, LS, MaxSgprs, MaxVgprs);
      if (NewIn != MustIn[I] || NewOut != MustOut[I] || NewDefIn != DefIn[I] ||
          NewDefOut != DefOut[I]) {
        MustIn[I] = std::move(NewIn);
        MustOut[I] = std::move(NewOut);
        DefIn[I] = std::move(NewDefIn);
        DefOut[I] = std::move(NewDefOut);
        Changed = true;
      }
    }
  }
  if (Changed)
    return;

  for (unsigned I = 0; I < Count; ++I) {
    const size_t GlobalIdx = GlobalIndices[I];
    if (!Reachable.test(I) ||
        Decoded[GlobalIdx].Mnemonic != "tensor_load_to_lds")
      continue;
    MCRegister Base = getDescriptorBaseSgpr(Decoded[GlobalIdx].Inst, *LS.MRI);
    std::optional<unsigned> Fact =
        getNumberedRegFactIndex(Base, *LS.MRI, MaxSgprs, MaxVgprs);
    const bool KnownZero = Fact && MustIn[I].test(*Fact);
    uint64_t MaskDef = TensorMaskDefUnknown;
    if (Fact) {
      auto Slot = TrackedSgprs.find(*Fact);
      if (Slot != TrackedSgprs.end())
        MaskDef = DefIn[I][Slot->second];
    }
    if (MaskDef < Range.Begin || MaskDef >= Range.End)
      MaskDef = TensorMaskDefUnknown;
    if (Seen.test(GlobalIdx)) {
      if (!KnownZero)
        Result.Low16KnownZero.reset(GlobalIdx);
      if (Result.MaskDefinitionOffsets[GlobalIdx] != MaskDef)
        Result.MaskDefinitionOffsets[GlobalIdx] = TensorMaskDefUnknown;
    } else {
      Seen.set(GlobalIdx);
      if (KnownZero)
        Result.Low16KnownZero.set(GlobalIdx);
      Result.MaskDefinitionOffsets[GlobalIdx] = MaskDef;
    }
  }
}

TensorDescriptorMustAnalysis computeTensorDescriptorMustAnalysisImpl(
    ArrayRef<InternalDecodedInst> Decoded,
    ArrayRef<InternalDecodedInst> AllDecoded,
    ArrayRef<TensorAnalysisRange> KernelRanges, const LLVMState &LS,
    const DenseSet<uint64_t> &DirectTargets, unsigned MaxSgprs,
    unsigned MaxVgprs) {
  TensorDescriptorMustAnalysis Result{
      BitVector(Decoded.size()),
      std::vector<uint64_t>(Decoded.size(), TensorMaskDefUnknown)};
  BitVector Seen(Decoded.size());
  if (!LS.MCII || !LS.MRI || MaxSgprs == 0 || MaxVgprs == 0)
    return Result;
  if (KernelRanges.empty())
    return Result;

  ArrayRef<TensorDispatchStub> DispatchStubs =
      KernelRanges.front().DispatchStubs;
  auto SameDispatchStub = [](const TensorDispatchStub &L,
                             const TensorDispatchStub &R) {
    return L.Begin == R.Begin && L.End == R.End && L.Terminal == R.Terminal &&
           L.Target == R.Target;
  };
  if (llvm::any_of(KernelRanges, [&](const TensorAnalysisRange &Range) {
        return Range.DispatchStubs.size() != DispatchStubs.size() ||
               !llvm::equal(Range.DispatchStubs, DispatchStubs,
                            SameDispatchStub);
      }))
    return Result;
  ArrayRef<uint64_t> VirtualExternalEntries =
      KernelRanges.front().VirtualExternalEntries;
  if (llvm::any_of(KernelRanges, [&](const TensorAnalysisRange &Range) {
        return !llvm::equal(Range.VirtualExternalEntries,
                            VirtualExternalEntries);
      }))
    return Result;
  ArrayRef<std::pair<uint64_t, uint64_t>> OriginalControlFlowEdges =
      KernelRanges.front().OriginalControlFlowEdges;
  if (llvm::any_of(KernelRanges, [&](const TensorAnalysisRange &Range) {
        return !llvm::equal(Range.OriginalControlFlowEdges,
                            OriginalControlFlowEdges);
      }))
    return Result;
  ArrayRef<uint64_t> OriginalCodeEntries =
      KernelRanges.front().OriginalCodeEntries;
  if (llvm::any_of(KernelRanges, [&](const TensorAnalysisRange &Range) {
        return !llvm::equal(Range.OriginalCodeEntries, OriginalCodeEntries);
      }))
    return Result;
  TensorExternalCfg ExternalCfg = buildTensorExternalCfg(
      Decoded, AllDecoded, DispatchStubs, VirtualExternalEntries,
      OriginalControlFlowEdges, OriginalCodeEntries, LS, DirectTargets);
  if (!ExternalCfg.Valid)
    return Result;
  for (const TensorAnalysisRange &Range : KernelRanges)
    analyzeTensorDescriptorRange(Decoded, AllDecoded, ExternalCfg, Range, LS,
                                 MaxSgprs, MaxVgprs, DirectTargets, Result,
                                 Seen);
  return Result;
}

// -- patchTensorLoadToLdsA0 -------------------------------------------------
//
// Replace the canonical one-cycle scalar delay immediately before the tensor
// load with s_pack_hh_b32_b16. Tensor loads are PC-sensitive on gfx1250 A0, so
// they must remain at their linked address instead of executing in a sled or
// appended trampoline. Reuse the exact canonical delay slot to clear the
// descriptor's multicast bits without allocating a scratch register.

bool patchTensorLoadToLdsA0(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  MCRegister BaseMCReg = getDescriptorBaseSgpr(DI.Inst, MRI);
  if (!BaseMCReg.isValid()) {
    log() << "hotswap: error: tensor_load_to_lds: could not extract descriptor "
             "base register\n";
    return failRequiredPatch(Ctx);
  }

  // The tensor instruction is PC-sensitive on gfx1250 A0. It must never be
  // borrowed by generic far source-window expansion, even when its mask is
  // placed at an earlier local descriptor definition.
  protectNonClauseRelocationOffset(Ctx, DI.Offset);

  if (Ctx.TensorDescriptorAnalysis &&
      Idx < Ctx.TensorDescriptorAnalysis->Low16KnownZero.size() &&
      Ctx.TensorDescriptorAnalysis->Low16KnownZero.test(Idx)) {
    log() << "hotswap: tensor_load_to_lds: descriptor low16 already zero at 0x"
          << utohexstr(DI.Offset) << "; tensor remains unchanged\n";
    DI.Mnemonic = "<replaced>";
    return false;
  }

  if (Ctx.TensorDescriptorAnalysis &&
      Idx < Ctx.TensorDescriptorAnalysis->MaskDefinitionOffsets.size()) {
    const uint64_t DefOffset =
        Ctx.TensorDescriptorAnalysis->MaskDefinitionOffsets[Idx];
    if (Ctx.TensorMaskedDefinitionOffsets.contains(DefOffset)) {
      log() << "hotswap: tensor_load_to_lds: reusing masked descriptor "
               "definition at 0x"
            << utohexstr(DefOffset) << "; tensor remains at 0x"
            << utohexstr(DI.Offset) << "\n";
      DI.Mnemonic = "<replaced>";
      return false;
    }
  }

  // Every A0 mask path executes before the PC-sensitive tensor instruction.
  // A direct edge to the tensor or any unresolved indirect entry in its
  // function can bypass that mask, including one already present in the input.
  if (isDirectControlFlowTarget(Ctx, DI.Offset) ||
      functionHasIndirectControlFlow(Ctx, DI.Offset)) {
    log() << "hotswap: error: tensor_load_to_lds at 0x" << utohexstr(DI.Offset)
          << " may be entered without executing its descriptor mask\n";
    return failRequiredPatch(Ctx);
  }

  if (isAlreadyTensorMaskPatched(Ctx, Idx, BaseMCReg))
    return false;

  std::string BaseSreg = toAsmRegName(MRI, BaseMCReg);

  std::string PackAsm = "s_pack_hh_b32_b16 " + BaseSreg + ", 0, " + BaseSreg;
  SmallVector<uint8_t> PackBytes = assembleSingleInst(PackAsm, Ctx.LS);
  if (PackBytes.empty()) {
    log() << "hotswap: tensor_load_to_lds pack: assembly failed: " << PackAsm
          << "\n";
    return failRequiredPatch(Ctx);
  }

  std::optional<LocalTensorMaskDefinition> LocalDef =
      findLocalTensorMaskDefinition(Ctx, Idx, BaseMCReg);
  const bool ClaimedLocalDefinition = LocalDef && !LocalDef->AlreadyMasked &&
                                      Ctx.ClaimedReplacementOffsets.contains(
                                          Ctx.Decoded[LocalDef->Index].Offset);
  if (ClaimedLocalDefinition) {
    // The definition still executes in the atomic relocated stream, but its
    // linked address is no longer an independent patch source. Prefer the
    // reconstructed canonical delay slot immediately before the tensor.
    log() << "hotswap: tensor_load_to_lds: descriptor definition at 0x"
          << utohexstr(Ctx.Decoded[LocalDef->Index].Offset)
          << " is relocated; using linked delay slot for the mask\n";
  } else if (LocalDef) {
    if (LocalDef->AlreadyMasked) {
      protectNonClauseRelocationOffset(Ctx,
                                       Ctx.Decoded[LocalDef->Index].Offset);
      Ctx.TensorMaskedDefinitionOffsets.insert(
          Ctx.Decoded[LocalDef->Index].Offset);
      log() << "hotswap: tensor_load_to_lds: descriptor already masked at 0x"
            << utohexstr(Ctx.Decoded[LocalDef->Index].Offset)
            << "; tensor remains at 0x" << utohexstr(DI.Offset) << "\n";
      DI.Mnemonic = "<replaced>";
      return false;
    }

    InternalDecodedInst &Def = Ctx.Decoded[LocalDef->Index];
    if (Def.Offset > Ctx.TextSize || Def.Size > Ctx.TextSize - Def.Offset)
      return failRequiredPatch(Ctx);
    SmallVector<uint8_t> Replacement(Ctx.Text + Def.Offset,
                                     Ctx.Text + Def.Offset + Def.Size);
    Replacement.append(PackBytes.begin(), PackBytes.end());
    if (!emitReplacementCode(Ctx, Def.Offset, Def.Size, Replacement))
      return failRequiredPatch(Ctx);
    protectNonClauseRelocationOffset(Ctx, Def.Offset);
    Ctx.TensorMaskedDefinitionOffsets.insert(Def.Offset);

    log() << "hotswap: tensor_load_to_lds: masked local descriptor definition "
             "at 0x"
          << utohexstr(Def.Offset) << "; tensor remains at 0x"
          << utohexstr(DI.Offset) << "\n";
    Ctx.RequiredPatchApplied = true;
    Def.Mnemonic = "<replaced>";
    DI.Mnemonic = "<replaced>";
    return true;
  }

  SmallVector<uint8_t> DelayBytes =
      assembleSingleInst("s_delay_alu instid0(SALU_CYCLE_1)", Ctx.LS);
  if (DelayBytes.empty()) {
    log() << "hotswap: tensor_load_to_lds delay assembly failed\n";
    return failRequiredPatch(Ctx);
  }

  if (Idx == 0) {
    log() << "hotswap: error: tensor_load_to_lds at 0x" << utohexstr(DI.Offset)
          << " has no preceding delay slot\n";
    return failRequiredPatch(Ctx);
  }

  InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
  ArrayRef<uint8_t> PrevBytes(Ctx.Text + Prev.Offset, Prev.Size);
  const bool ReconstructedDelay =
      Ctx.ClaimedReplacementOffsets.contains(Prev.Offset);
  if ((!ReconstructedDelay && Prev.Mnemonic != "s_delay_alu") ||
      Prev.Offset + Prev.Size != DI.Offset ||
      PrevBytes != ArrayRef<uint8_t>(DelayBytes) ||
      Prev.Size != PackBytes.size()) {
    log() << "hotswap: error: tensor_load_to_lds at 0x" << utohexstr(DI.Offset)
          << " is not preceded by the canonical scalar delay\n";
    return failRequiredPatch(Ctx);
  }

  protectNonClauseRelocationOffset(Ctx, Prev.Offset);

  std::memcpy(Ctx.Text + Prev.Offset, PackBytes.data(), PackBytes.size());
  log() << "hotswap: tensor_load_to_lds: in-place descriptor mask at 0x"
        << utohexstr(Prev.Offset) << "; tensor remains at 0x"
        << utohexstr(DI.Offset) << "\n";

  Ctx.RequiredPatchApplied = true;
  DI.Mnemonic = "<replaced>";
  return true;
}

// -- Cluster/TDM mask helpers ------------------------------------------------
//
// In-place patching demotes off-form cluster_load* instructions to
// global_load* first. Any cluster_load* that reaches this trampoline pass is
// still a real cluster load on A0 and must see M0.wg_mask[15:0] cleared. B0
// does not need the cluster-load M0 workaround; its hotswap mask rule applies
// only to tensor_load_to_lds when the wave is effectively non-cluster.

// MI400 SPG section 3.4: SQ_WAVE_IB_STS2.CLUSTER_ID is bits [9:6].
constexpr unsigned IbSts2ClusterIdOffset = 6;
constexpr unsigned IbSts2ClusterIdWidth = 4;

bool isClusterLoad(StringRef Mnemonic) {
  return StringSwitch<bool>(Mnemonic)
      .Case("cluster_load_b32", true)
      .Case("cluster_load_b64", true)
      .Case("cluster_load_b128", true)
      .Case("cluster_load_async_to_lds_b8", true)
      .Case("cluster_load_async_to_lds_b32", true)
      .Case("cluster_load_async_to_lds_b64", true)
      .Case("cluster_load_async_to_lds_b128", true)
      .Default(false);
}

bool operandIsM0(const MCInst &Inst, const MCRegisterInfo &MRI,
                 unsigned OperandIdx) {
  if (OperandIdx >= Inst.getNumOperands())
    return false;
  const MCOperand &Op = Inst.getOperand(OperandIdx);
  return Op.isReg() && isM0Reg(MCRegister(Op.getReg()), MRI);
}

bool isAlreadyClusterMaskPatched(const PatchContext &Ctx, size_t Idx) {
  if (Idx == 0)
    return false;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
  const MCInst &PI = Prev.Inst;

  if (Prev.Mnemonic == "s_pack_hh_b32_b16") {
    if (PI.getNumOperands() < 3 || !operandIsM0(PI, MRI, 0))
      return false;
    if (!PI.getOperand(1).isImm() || PI.getOperand(1).getImm() != 0)
      return false;
    return operandIsM0(PI, MRI, 2);
  }

  if (Prev.Mnemonic != "s_and_b32" || PI.getNumOperands() < 3 ||
      !operandIsM0(PI, MRI, 0))
    return false;

  for (unsigned OpIdx = 1; OpIdx < PI.getNumOperands(); ++OpIdx) {
    if (operandIsM0(PI, MRI, OpIdx))
      return true;
  }
  return false;
}

std::optional<uint64_t> getFlatClusterSize(const KernelClusterDims &Dims,
                                           StringRef KernelName) {
  if (Dims.X == 0 && Dims.Y == 0 && Dims.Z == 0)
    return 0;

  if (Dims.X == 0 || Dims.Y == 0 || Dims.Z == 0) {
    log() << "hotswap: error: .cluster_dims for '" << KernelName
          << "' contains a zero dimension in a nonzero fixed cluster ("
          << Dims.X << ", " << Dims.Y << ", " << Dims.Z
          << "); falling back to dynamic cluster-id check\n";
    return std::nullopt;
  }

  uint64_t Flat = Dims.X;
  if (Dims.Y > std::numeric_limits<uint64_t>::max() / Flat) {
    log() << "hotswap: error: .cluster_dims for '" << KernelName
          << "' overflows uint64_t; falling back to dynamic cluster-id check\n";
    return std::nullopt;
  }
  Flat *= Dims.Y;
  if (Dims.Z > std::numeric_limits<uint64_t>::max() / Flat) {
    log() << "hotswap: error: .cluster_dims for '" << KernelName
          << "' overflows uint64_t; falling back to dynamic cluster-id check\n";
    return std::nullopt;
  }
  Flat *= Dims.Z;
  return Flat;
}

bool appendAsmBytes(SmallVectorImpl<uint8_t> &Out, StringRef Asm,
                    const LLVMState &LS, StringRef Context) {
  SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: " << Context << ": assembly failed: " << Asm
          << "\n";
    return false;
  }
  Out.append(Bytes.begin(), Bytes.end());
  return true;
}

bool appendRequiredAsm(PatchContext &Ctx, SmallVectorImpl<uint8_t> &Out,
                       StringRef Asm, StringRef Context) {
  if (appendAsmBytes(Out, Asm, Ctx.LS, Context))
    return true;
  return failRequiredPatch(Ctx);
}

bool hasKnownNonClusterDispatch(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<KernelClusterDims> ClusterDims =
      Ctx.Elf.getKernelClusterDims(KernelName);
  if (!ClusterDims)
    return false;

  std::optional<uint64_t> Flat = getFlatClusterSize(*ClusterDims, KernelName);
  return Flat && *Flat <= 1;
}

bool patchTensorLoadToLdsB0(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  MCRegister BaseMCReg = getDescriptorBaseSgpr(DI.Inst, MRI);
  if (!BaseMCReg.isValid()) {
    log() << "hotswap: error: tensor_load_to_lds: could not extract descriptor "
             "base register\n";
    return failRequiredPatch(Ctx);
  }

  if (isAlreadyTensorMaskPatched(Ctx, Idx, BaseMCReg))
    return false;

  if (hasKnownNonClusterDispatch(Ctx, Idx))
    return patchTensorLoadToLdsA0(Ctx, Idx);

  SmallVector<unsigned, 8> DescriptorSgprs =
      getDescriptorSgprIndices(DI.Inst, MRI);
  std::optional<SgprScratchAlloc> ScratchSgpr =
      tryAllocScratchSgpr(Ctx, Idx, DescriptorSgprs);
  if (!ScratchSgpr) {
    log() << "hotswap: error: tensor_load_to_lds: no scratch SGPR available "
             "for B0 cluster-id check\n";
    return failRequiredPatch(Ctx);
  }

  bool SccLive = isSccLiveAfter(Ctx, Idx);
  std::optional<SgprScratchAlloc> SccScratchSgpr;
  SmallVector<unsigned, 9> SccExcludedSgprs;
  SccExcludedSgprs.append(DescriptorSgprs.begin(), DescriptorSgprs.end());
  SccExcludedSgprs.push_back(ScratchSgpr->Sgpr);
  if (SccLive) {
    SccScratchSgpr = tryAllocScratchSgpr(Ctx, Idx, SccExcludedSgprs);
    if (!SccScratchSgpr) {
      log() << "hotswap: error: tensor_load_to_lds: no scratch SGPR available "
               "to preserve SCC for B0 cluster-id check\n";
      return failRequiredPatch(Ctx);
    }
  }

  std::string BaseSreg = toAsmRegName(MRI, BaseMCReg);
  std::string S = "s" + std::to_string(ScratchSgpr->Sgpr);
  std::string SccS =
      SccScratchSgpr ? "s" + std::to_string(SccScratchSgpr->Sgpr) : "";
  std::string Context =
      "tensor_load_to_lds B0 mask at 0x" + utohexstr(DI.Offset);

  SmallVector<uint8_t> Prefix;
  std::string ReadClusterIdAsm = "s_getreg_b32 " + BaseSreg +
                                 ", hwreg(HW_REG_IB_STS2, " +
                                 std::to_string(IbSts2ClusterIdOffset) + ", " +
                                 std::to_string(IbSts2ClusterIdWidth) + ")";
  if (!appendRequiredAsm(Ctx, Prefix, "s_mov_b32 " + S + ", " + BaseSreg,
                         Context))
    return false;

  if (SccLive) {
    if (!appendRequiredAsm(Ctx, Prefix, "s_cselect_b32 " + SccS + ", 1, 0",
                           Context))
      return false;
  }

  if (!appendRequiredAsm(Ctx, Prefix, ReadClusterIdAsm, Context))
    return false;
  if (!appendRequiredAsm(Ctx, Prefix, "s_cmp_eq_u32 " + BaseSreg + ", 0",
                         Context))
    return false;
  if (!appendRequiredAsm(
          Ctx, Prefix, "s_pack_hh_b32_b16 " + BaseSreg + ", 0, " + S, Context))
    return false;
  if (!appendRequiredAsm(
          Ctx, Prefix, "s_cselect_b32 " + BaseSreg + ", " + BaseSreg + ", " + S,
          Context))
    return false;

  if (SccLive) {
    if (!appendRequiredAsm(Ctx, Prefix, "s_cmp_lg_u32 " + SccS + ", 0",
                           Context))
      return false;
  }

  const uint8_t *OrigInst = Ctx.Text + DI.Offset;
  SmallVector<uint8_t> Replacement;
  Replacement.append(Prefix.begin(), Prefix.end());
  Replacement.append(OrigInst, OrigInst + DI.Size);

  bool SgprLive = isSgprLiveAfter(Ctx, Idx, BaseMCReg);
  if (SgprLive) {
    SmallVector<uint8_t> Restore =
        assembleSingleInst("s_mov_b32 " + BaseSreg + ", " + S, Ctx.LS);
    if (Restore.empty()) {
      log() << "hotswap: error: tensor_load_to_lds: B0 restore assembly "
               "failed\n";
      return failRequiredPatch(Ctx);
    }
    Replacement.append(Restore.begin(), Restore.end());
  }

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return failRequiredPatch(Ctx);

  commitScratchSgpr(Ctx, *ScratchSgpr);
  if (SccScratchSgpr)
    commitScratchSgpr(Ctx, *SccScratchSgpr);
  Ctx.RequiredPatchApplied = true;

  log() << "hotswap: tensor_load_to_lds: B0 cluster-id conditional mask for "
        << BaseSreg << ", save/restore via " << S << " at 0x"
        << utohexstr(DI.Offset) << "\n";
  DI.Mnemonic = "<replaced>";
  return true;
}

std::optional<SmallVector<uint8_t>>
buildClusterLoadA0MaskPrefix(PatchContext &Ctx, StringRef ScratchSgpr,
                             StringRef Context) {
  SmallVector<uint8_t> Prefix;
  std::string SaveAsm = "s_mov_b32 ";
  SaveAsm += ScratchSgpr;
  SaveAsm += ", m0";
  std::string MaskAsm = "s_pack_hh_b32_b16 m0, 0, m0";
  if (!appendAsmBytes(Prefix, SaveAsm, Ctx.LS, Context))
    return std::nullopt;
  if (!appendAsmBytes(Prefix, MaskAsm, Ctx.LS, Context))
    return std::nullopt;
  return Prefix;
}

bool patchClusterLoadMaskA0(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  if (isAlreadyClusterMaskPatched(Ctx, Idx))
    return false;

  SmallVector<unsigned, 8> ClusterLoadSgprs =
      getSgprOperandIndices(DI.Inst, MRI);
  std::optional<SgprScratchAlloc> ScratchSgpr =
      tryAllocScratchSgpr(Ctx, Idx, ClusterLoadSgprs);
  if (!ScratchSgpr) {
    log() << "hotswap: error: " << DI.Mnemonic
          << ": no scratch SGPR available for M0 mask save/restore at 0x"
          << utohexstr(DI.Offset) << "\n";
    return failRequiredPatch(Ctx);
  }

  std::string S = "s" + std::to_string(ScratchSgpr->Sgpr);
  std::string Context = DI.Mnemonic + " M0 mask at 0x" + utohexstr(DI.Offset);
  std::optional<SmallVector<uint8_t>> Prefix =
      buildClusterLoadA0MaskPrefix(Ctx, S, Context);
  std::string RestoreAsm = "s_mov_b32 m0, " + S;
  SmallVector<uint8_t> Restore = assembleSingleInst(RestoreAsm, Ctx.LS);
  if (!Prefix || Restore.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic
          << ": M0 mask save/restore assembly failed at 0x"
          << utohexstr(DI.Offset) << "\n";
    return failRequiredPatch(Ctx);
  }

  const uint8_t *OrigInst = Ctx.Text + DI.Offset;
  SmallVector<uint8_t> Replacement;
  Replacement.append(Prefix->begin(), Prefix->end());
  Replacement.append(OrigInst, OrigInst + DI.Size);
  Replacement.append(Restore.begin(), Restore.end());

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return failRequiredPatch(Ctx);

  commitScratchSgpr(Ctx, *ScratchSgpr);
  Ctx.RequiredPatchApplied = true;

  log() << "hotswap: cluster_load M0 mask: " << DI.Mnemonic
        << " clears A0 wg_mask bits, save/restore via " << S << " at 0x"
        << utohexstr(DI.Offset) << "\n";
  DI.Mnemonic = "<replaced>";
  return true;
}

// -- ADDTID swap table (StringSwitch) ---------------------------------------
//
// Maps each ADDTID DS mnemonic to its plain DS replacement. The lane-id
// expression that ADDTID encodes implicitly is materialised in the ALU by
// the trampoline body, then a regular DS op consumes the computed address.

StringRef getAddtidReplacement(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("ds_load_addtid_b32", "ds_load_b32")
      .Case("ds_store_addtid_b32", "ds_store_b32")
      .Default("");
}

// Predicate that pins the load/store dispatch alongside getAddtidReplacement
// so the two stay in sync if the table grows. Avoids a string compare in
// patchDsAddtid that would silently diverge from the StringSwitch above.
bool isAddtidLoad(StringRef Mnemonic) {
  return Mnemonic == "ds_load_addtid_b32";
}

// LDS allocations strictly above this threshold are unreachable through
// ADDTID once hotswapped to A0, because A0 truncates M0 to 16 bits. The
// patch itself is still applied (the lane-id math runs through the ALU);
// this constant only gates a diagnostic so users with oversized LDS
// allocations are warned that values may still be silently wrong.
// Derived from the M0 bit-width on A0 so the magic number stays out of
// the source: 1 << 16 = 65536 bytes addressable per ADDTID encoding.
constexpr uint32_t AddtidLdsLimitA0 = 1u << 16;

// ADDTID MCInst operand layout (AddtidOpReg / AddtidOpOffset / AddtidOpGds)
// lives in comgr-hotswap-internal.h so the layout pin is shared with the unit
// tests in HotswapMCTest.cpp.

// GDS=1 ADDTID is not reachable through the gfx12 assembler -- the asm
// parser rejects the `gds` modifier on this subtarget, so any MCInst
// produced by clang/llvm-mc has GDS=0. This predicate stays as
// defense-in-depth for hand-crafted byte input or future subtargets that
// re-enable the encoding through the same MCInst slot. Because the path
// is unreachable on gfx12 it is not exercised by lit; coverage exists via
// AddTid.{Load,Store}AddTidDecodesWithExpectedLayout pinning the operand
// shape that this predicate consumes.
bool isAddtidGds(const MCInst &Inst) {
  if (Inst.getNumOperands() <= AddtidOpGds)
    return false;
  const MCOperand &Op = Inst.getOperand(AddtidOpGds);
  return Op.isImm() && Op.getImm() != 0;
}

// The DS offset field is a 16-bit immediate per the gfx12 ISA encoding;
// returning uint16_t keeps the field width visible at the type level and
// lets callers widen explicitly when needed.
std::optional<uint16_t> getAddtidOffset(const MCInst &Inst) {
  if (Inst.getNumOperands() <= AddtidOpOffset)
    return std::nullopt;
  const MCOperand &Op = Inst.getOperand(AddtidOpOffset);
  if (!Op.isImm())
    return std::nullopt;
  return static_cast<uint16_t>(Op.getImm());
}

// Build the trampoline asm for a ds_load_addtid_b32 site. The destination
// VGPR is reused as the address-compute scratch because the load overwrites
// it, so no extra VGPR allocation is needed for the load path. Reusing the
// destination as both source operands of ds_load_b32 (`ds_load_b32 vN, vN`)
// is well-defined on gfx12: the DS unit reads vaddr from the operand file
// before vdst is written, so the same VGPR can serve both roles.
//
// The replacement reproduces the ADDTID address computation in the ALU:
//   lane_id = mbcnt_lo(-1, 0)    ; lanes 0-31 contribute via exec_lo
//             mbcnt_hi(-1, V)    ;   lanes 32-63 (wave64) extend through
//                                ;   exec_hi; in wave32 exec_hi is zero so
//                                ;   the hi step is a no-op (the sequence
//                                ;   is identical for both wave sizes)
//   addr    = m0 + lane_id * 4   ; + offset (folded into the DS encoding by
//                                ;   the assembler when ToMnem is emitted)
//
// Address mask: B0 hardware reads only 20 bits of M0 at the DS unit, so any
// junk in M0[31:20] (e.g. left over from s_sendmsg or other M0 producers) is
// ignored. v_add_nc_u32 reads M0 as a full 32-bit scalar source, so we mask
// the post-add result to the same 20 bits to stay bit-exact with B0 across
// the entire reachable LDS range (gfx1250 LDS <= 320 KiB and lane_id*4 <=
// 0xFC, so the sum fits comfortably below 1 MiB and the mask is a no-op for
// any conforming M0 -- the mask only fires defensively when M0[31:20] is
// non-zero on entry).
SmallVector<std::string> buildAddtidLoadAsm(StringRef VName, uint16_t Offset,
                                            StringRef ToMnem) {
  std::string V(VName);
  SmallVector<std::string> Lines;
  Lines.push_back("v_mbcnt_lo_u32_b32 " + V + ", -1, 0");
  Lines.push_back("v_mbcnt_hi_u32_b32 " + V + ", -1, " + V);
  Lines.push_back("v_lshlrev_b32 " + V + ", 2, " + V);
  Lines.push_back("v_add_nc_u32 " + V + ", m0, " + V);
  Lines.push_back("v_and_b32 " + V + ", 0xfffff, " + V);
  Lines.push_back(ToMnem.str() + " " + V + ", " + V + fmtOffset(Offset));
  return Lines;
}

// Build the trampoline asm for a ds_store_addtid_b32 site. \p VTmpName is a
// scratch VGPR holding the computed address; \p VDataName is the original
// data VGPR. Operand order for ds_store_b32 is (addr, data).
//
// Same mbcnt_lo/mbcnt_hi pair and 20-bit M0 mask as the load path; see
// buildAddtidLoadAsm above for the full rationale.
SmallVector<std::string> buildAddtidStoreAsm(StringRef VTmpName,
                                             StringRef VDataName,
                                             uint16_t Offset,
                                             StringRef ToMnem) {
  std::string VTmp(VTmpName);
  std::string VData(VDataName);
  SmallVector<std::string> Lines;
  Lines.push_back("v_mbcnt_lo_u32_b32 " + VTmp + ", -1, 0");
  Lines.push_back("v_mbcnt_hi_u32_b32 " + VTmp + ", -1, " + VTmp);
  Lines.push_back("v_lshlrev_b32 " + VTmp + ", 2, " + VTmp);
  Lines.push_back("v_add_nc_u32 " + VTmp + ", m0, " + VTmp);
  Lines.push_back("v_and_b32 " + VTmp + ", 0xfffff, " + VTmp);
  Lines.push_back(ToMnem.str() + " " + VTmp + ", " + VData + fmtOffset(Offset));
  return Lines;
}

// -- patchDsAddtid ----------------------------------------------------------
//
// Trampoline expansion for ds_load_addtid_b32 / ds_store_addtid_b32 on
// A0. The replacement materialises the ADDTID address through the ALU
// (so the full 32-bit M0 is used) and issues a regular ds_*_b32. GDS=1
// is rejected. Once an ADDTID mnemonic is recognized, every failure must be
// fatal so the rewrite cannot report success with an A0-incompatible
// instruction still present.

bool patchDsAddtid(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  // The dispatcher in applyTrampolinePatchesImpl already gates on
  // !getAddtidReplacement(Mnem).empty(), so by contract we only see
  // ds_load_addtid_b32 / ds_store_addtid_b32 here.
  StringRef ToMnem = getAddtidReplacement(DI.Mnemonic);
  assert(!ToMnem.empty() &&
         "patchDsAddtid called for non-ADDTID mnemonic; caller must filter");

  if (isAddtidGds(DI.Inst)) {
    log() << "hotswap: error: " << DI.Mnemonic << " with GDS=1 at 0x"
          << utohexstr(DI.Offset)
          << " is not supported; leaving original instruction in place\n";
    return failRequiredPatch(Ctx);
  }

  std::optional<uint16_t> OffsetOpt = getAddtidOffset(DI.Inst);
  if (!OffsetOpt) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": missing/non-immediate offset\n";
    return failRequiredPatch(Ctx);
  }
  uint16_t Offset = *OffsetOpt;

  if (DI.Inst.getNumOperands() <= AddtidOpReg ||
      !DI.Inst.getOperand(AddtidOpReg).isReg() ||
      !DI.Inst.getOperand(AddtidOpReg).getReg()) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": missing register operand\n";
    return failRequiredPatch(Ctx);
  }

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  MCRegister Reg = MCRegister(DI.Inst.getOperand(AddtidOpReg).getReg());
  std::string RegName = toAsmRegName(MRI, Reg);
  if (RegName.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": cannot resolve register name\n";
    return failRequiredPatch(Ctx);
  }

  bool IsLoad = isAddtidLoad(DI.Mnemonic);
  SmallVector<std::string> AsmLines;
  std::optional<ScratchAlloc> StoreScratch;

  if (IsLoad) {
    AsmLines = buildAddtidLoadAsm(RegName, Offset, ToMnem);
  } else {
    // Store path needs a scratch VGPR for the address-compute temporary
    // because the original data VGPR must be preserved as the store source.
    StoreScratch = tryAllocScratchVgpr(Ctx, Idx);
    if (!StoreScratch) {
      std::string KernelName =
          Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
      StringRef KernelDisplay =
          KernelName.empty() ? StringRef("<unknown>") : StringRef(KernelName);
      std::optional<uint32_t> LdsSize =
          Ctx.Elf.getKernelStaticLdsSize(KernelName);
      // Trampoline could not be applied: the original ds_*_addtid_b32 stays
      // in the code object and will silently truncate M0 to 16 bits on gfx1250
      // A0 whenever the runtime LDS layout exceeds 64 KiB.
      // Static LDS is visible in the kernel descriptor; dynamic LDS added
      // by the host at dispatch (hidden_dynamic_lds_size kernarg or a
      // dynamic_shared_pointer user arg) is not. The warning therefore
      // fires unconditionally rather than gating on the visible lower
      // bound -- a follow-up will use ElfView::kernelUsesDynamicLds to
      // tighten the condition to (static>64KiB || dynamicUsed).
      log() << "hotswap: warning: kernel '" << KernelDisplay << "' uses "
            << DI.Mnemonic
            << "; trampoline could not be applied, so A0 16-bit M0"
               " truncation may produce silently wrong results when runtime"
               " LDS (static + dynamic) exceeds "
            << AddtidLdsLimitA0 << " bytes";
      if (LdsSize)
        log() << " (static LDS = " << *LdsSize << " bytes)";
      log() << " at 0x" << utohexstr(DI.Offset) << "\n";
      log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
            << utohexstr(DI.Offset) << ": no scratch VGPR available\n";
      return failRequiredPatch(Ctx);
    }

    std::string TmpName = ("v" + Twine(StoreScratch->Vgpr)).str();
    AsmLines = buildAddtidStoreAsm(TmpName, RegName, Offset, ToMnem);
  }

  if (StoreScratch && checkKernelVgprBump(Ctx, StoreScratch->KernelName,
                                          StoreScratch->ExtraVgprsNeeded,
                                          PatchRequirement::Optional) !=
                          VgprBumpDecision::Apply)
    return false;

  std::string Combined;
  for (const std::string &Line : AsmLines)
    Combined += Line + "\n";
  SmallVector<uint8_t> Bytes = assembleInstructions(Combined, Ctx.LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic
          << " trampoline assembly failed at 0x" << utohexstr(DI.Offset)
          << "\n";
    return failRequiredPatch(Ctx);
  }

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Bytes))
    return failRequiredPatch(Ctx);

  // Commit the scratch-VGPR reservation only after the patch is in place:
  // any earlier failure (assembly, sled/trampoline emission) leaves no
  // bytes at DI.Offset to back the reservation, so neither the descriptor
  // accounting nor OutScratchPatches must advertise a slot for it.
  if (StoreScratch) {
    ScratchPatchInfo SPI;
    SPI.Offset = DI.Offset;
    SPI.ScratchRegs.resize(Ctx.Config.MaxVgprs);
    SPI.ScratchRegs.set(StoreScratch->Vgpr);
    Ctx.OutScratchPatches.push_back(std::move(SPI));
    commitScratchVgpr(Ctx, *StoreScratch);
  }

  log() << "hotswap: trampoline: " << DI.Mnemonic << " -> " << ToMnem
        << " at 0x" << utohexstr(DI.Offset) << " (offset=" << Offset << ", "
        << RegName << ")\n";
  DI.Mnemonic = "<replaced>";
  return true;
}

} // anonymous namespace

TensorDescriptorMustAnalysis
computeTensorDescriptorMustAnalysis(ArrayRef<InternalDecodedInst> Decoded,
                                    ArrayRef<InternalDecodedInst> AllDecoded,
                                    ArrayRef<TensorAnalysisRange> KernelRanges,
                                    const LLVMState &LS,
                                    const DenseSet<uint64_t> &DirectTargets,
                                    unsigned MaxSgprs, unsigned MaxVgprs) {
  return computeTensorDescriptorMustAnalysisImpl(
      Decoded, AllDecoded, KernelRanges, LS, DirectTargets, MaxSgprs, MaxVgprs);
}

// Keep atomic relocation eligibility aligned with both precomputed whole-pass
// ownership and the per-instruction dispatcher. Callers may copy an earlier
// same-size correction from current text, but must not claim a site that owns
// an independent rewrite. Conditional families mirror their dispatch gates so
// valid no-op variants remain relocatable.
bool requiresIndependentInstructionRewrite(const PatchContext &Ctx,
                                           size_t Idx) {
  if (Idx >= Ctx.Decoded.size())
    return true;

  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  StringRef Mnemonic(DI.Mnemonic);
  if (Mnemonic == "<unknown>" || Mnemonic == "<replaced>")
    return true;

  if (hasSiteReplacementReservation(Ctx, DI.Offset))
    return true;

  if (Mnemonic == "tensor_load_to_lds")
    return Ctx.Config.MaskPolicy != MaskWorkaroundPolicy::None;

  if (!Ctx.Config.RunB0A0Patches)
    return false;

  if (!getDs2AddrReplacement(Mnemonic).empty()) {
    const bool ProvenAligned =
        Ctx.Ds2AddressProvenAligned.size() == Ctx.Decoded.size() &&
        Ctx.Ds2AddressProvenAligned.test(Idx);
    return !ProvenAligned;
  }

  if (isClusterLoad(Mnemonic)) {
    // The off form is demoted in-place. The SGPR-relative form survives that
    // pass and needs the trampoline mask only for the A0 policy.
    bool HasSgprOperand = !Ctx.LS.MRI;
    if (Ctx.LS.MRI)
      for (const MCOperand &Operand : DI.Inst)
        HasSgprOperand |= Operand.isReg() && Operand.getReg() &&
                          StringRef(Ctx.LS.MRI->getName(Operand.getReg()))
                              .starts_with("SGPR");
    return !HasSgprOperand || Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::A0;
  }

  if (!getAddtidReplacement(Mnemonic).empty() ||
      isWmmaSplitPatchCandidate(Mnemonic) ||
      Mnemonic == "s_barrier_signal_isfirst" ||
      Mnemonic == "v_wmma_scale16_f32_16x16x128_f8f6f4" ||
      Mnemonic == "v_wmma_scale16_f32_32x16x128_f4")
    return true;

  // Mirror only the dispatcher's semantic CLAMP gate. Non-CLAMP FP8
  // conversions are valid on A0 and may move with a protected window.
  auto NeedsClampPatchAt = [&](unsigned OperandIndex) {
    // A missing or non-immediate CLAMP field is malformed, not a proven no-op;
    // retain ownership so the scratch pass can reject it explicitly.
    return OperandIndex >= DI.Inst.getNumOperands() ||
           !DI.Inst.getOperand(OperandIndex).isImm() ||
           DI.Inst.getOperand(OperandIndex).getImm() != 0;
  };
  if (Mnemonic == "v_cvt_pk_fp8_f32")
    return (DI.Size != 8 && DI.Size != 12) || NeedsClampPatchAt(5);
  if (Mnemonic == "v_cvt_sr_fp8_f32")
    return DI.Size != 8 || NeedsClampPatchAt(5);
  if (Mnemonic.starts_with("v_cvt_f32_fp8")) {
    if (DI.Size == 4)
      return false;
    return DI.Size != 8 || NeedsClampPatchAt(2);
  }

  return false;
}

void precomputeDs2AddressAlignment(PatchContext &Ctx) {
  Ctx.Ds2AddressProvenAligned.resize(Ctx.Decoded.size());
  Ctx.Ds2AddressProvenAligned.reset();
  if (Ctx.HasUnknownArbitraryIndirectTarget)
    return;

  DenseMap<std::pair<size_t, unsigned>, SmallVector<size_t, 4>> CandidateUses;
  for (size_t I = 0; I != Ctx.Decoded.size(); ++I) {
    if (getDs2AddrReplacement(Ctx.Decoded[I].Mnemonic).empty())
      continue;
    if (computeDs2AddressProvenAligned(Ctx, I)) {
      Ctx.Ds2AddressProvenAligned.set(I);
      continue;
    }

    const InternalDecodedInst &DS = Ctx.Decoded[I];
    StringRef Mnem(DS.Mnemonic);
    if (Mnem != "ds_load_2addr_b32" && Mnem != "ds_load_2addr_b64" &&
        Mnem != "ds_store_2addr_b32" && Mnem != "ds_store_2addr_b64")
      continue;
    if (DS.Inst.getNumOperands() == 0 ||
        !DS.Inst.getOperand(DS.Inst.getNumOperands() - 1).isImm() ||
        DS.Inst.getOperand(DS.Inst.getNumOperands() - 1).getImm() != 0)
      continue;
    std::optional<DsOperands> Ops =
        extractDsOperands(DS.Inst, DS.Mnemonic, Ctx.LS);
    if (!Ops)
      continue;
    const unsigned AddrIndex = Mnem.starts_with("ds_store_") ? 0 : 1;
    if (Ops->Regs.size() <= AddrIndex)
      continue;
    MCRegister Address = Ops->Regs[AddrIndex];
    const unsigned Alignment = Ops->IsB64 ? 8 : 4;
    if (Ops->Off0 % Alignment != 0 || Ops->Off1 % Alignment != 0)
      continue;
    std::optional<ElfView::FunctionTextRange> Function =
        Ctx.Elf.findFunctionTextRangeAtOffset(DS.Offset);
    if (!Function ||
        Ctx.IndirectControlFlowFunctions.contains(Function->Begin) ||
        Ctx.CrossFunctionInteriorEntryFunctions.contains(Function->Begin) ||
        !Ctx.DirectControlFlowTargets ||
        Ctx.DirectControlFlowTargets->contains(DS.Offset))
      continue;

    for (size_t DefI = I; DefI-- > 0;) {
      const InternalDecodedInst &Def = Ctx.Decoded[DefI];
      if (Def.Offset < Function->Begin)
        break;
      if (!definesRegister(Def, Address, Ctx.LS))
        continue;
      if ((isAlignedConstantAddressDef(Def, Address, Alignment, Ctx, DefI,
                                       /*RequireEqualMode=*/false) ||
           isAlignedSgprCopyAddressDef(Def, Address, Alignment, Ctx, DefI)) &&
          DefI < Ctx.VgprMsbDstBefore.size() && Ctx.VgprMsbDstBefore[DefI] >= 0)
        CandidateUses[{DefI, Address.id()}].push_back(I);
      break;
    }
  }

  for (const auto &Entry : CandidateUses)
    proveLongRangeDs2Alignment(Ctx, Entry.first.first,
                               MCRegister(Entry.first.second), Entry.second,
                               Ctx.Ds2AddressProvenAligned);
}

// -- applyTrampolinePatches -------------------------------------------------
//
// Strong-symbol override. Handles B0 errata that produce replacement code
// larger than the original instruction slot:
//
//   ds_*_2addr_*           -> split into two single-address DS ops
//     (covers both the stride64 and non-stride64 encodings)
//   tensor_load_to_lds     -> apply the selected target stepping's multicast
//                             mask rule
//   cluster_load*          -> in A0 mask mode, save/clear/restore M0 for
//                             remaining cluster ops
//   ds_*_addtid_b32        -> materialise lane-id math in ALU, then ds_*_b32

static uint32_t applyTrampolinePatchesImpl(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Ctx.Config.RunB0A0Patches && !getDs2AddrReplacement(Mnem).empty()) {
    HotswapProfile::Scope S =
        Ctx.Profile.time(HotswapMetric::TrampolineDs2Addr);
    if (isDs2AddressProvenAligned(Ctx, Idx)) {
      log() << "hotswap: ds_2addr: preserved proven-aligned " << Mnem
            << " at 0x" << utohexstr(Ctx.Decoded[Idx].Offset) << "\n";
      return 0;
    }
    const uint32_t P = patchDs2Addr(Ctx, Idx);
    S.addPatches(P);
    return P;
  }

  if (Mnem == "tensor_load_to_lds") {
    if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::A0) {
      HotswapProfile::Scope S =
          Ctx.Profile.time(HotswapMetric::TrampolineTensorTdm);
      const uint32_t P = patchTensorLoadToLdsA0(Ctx, Idx) ? 1 : 0;
      S.addPatches(P);
      return P;
    }
    if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::B0) {
      HotswapProfile::Scope S =
          Ctx.Profile.time(HotswapMetric::TrampolineTensorTdm);
      const uint32_t P = patchTensorLoadToLdsB0(Ctx, Idx) ? 1 : 0;
      S.addPatches(P);
      return P;
    }
  }

  if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::A0 &&
      isClusterLoad(Mnem)) {
    HotswapProfile::Scope S =
        Ctx.Profile.time(HotswapMetric::TrampolineClusterLoad);
    const uint32_t P = patchClusterLoadMaskA0(Ctx, Idx) ? 1 : 0;
    S.addPatches(P);
    return P;
  }

  if (Ctx.Config.RunB0A0Patches && !getAddtidReplacement(Mnem).empty()) {
    HotswapProfile::Scope S = Ctx.Profile.time(HotswapMetric::TrampolineAddtid);
    const uint32_t P = patchDsAddtid(Ctx, Idx) ? 1 : 0;
    S.addPatches(P);
    return P;
  }

  return 0;
}

void registerTrampolinePatch(HotswapPatchVTable &VT) {
  VT.applyTrampolinePatches = &applyTrampolinePatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
