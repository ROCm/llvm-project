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

#include "llvm/ADT/DenseSet.h"
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
          << "); required rewrite is not representable\n";
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
  const MCRegisterInfo &MRI = *Ops.MRI;
  std::pair<std::string, std::string> Dst =
      splitDstPair(Ops.Regs[0], Ops.IsB64, MRI);
  if (Dst.first.empty())
    return {};
  std::string Addr = toAsmRegName(MRI, Ops.Regs[1]);
  std::string First =
      ToMnem.str() + " " + Dst.first + ", " + Addr + fmtOffset(Ops.Off0);
  std::string Second =
      ToMnem.str() + " " + Dst.second + ", " + Addr + fmtOffset(Ops.Off1);

  // A 2-address load samples Addr once, before either destination is written.
  // Preserve that lifetime after splitting: if the first destination half
  // contains Addr, emitting it first can replace Addr before the second load
  // samples it. Emit the aliasing half last instead. If Addr belongs to the
  // second half, the natural order already has the required property.
  SmallVector<MCRegister, 4> DstRegs = getDirectSubRegs(Ops.Regs[0], MRI);
  unsigned FirstHalfWidth = Ops.IsB64 ? 2 : 1;
  bool AddrOverlapsFirst = false;
  for (unsigned I = 0; I < FirstHalfWidth && I < DstRegs.size(); ++I) {
    // regsOverlap does not report every scalar VGPR alias carried by the
    // gfx1250 DS2 MC operands (notably b32 tuple components). The canonical
    // assembler names identify those physical aliases without depending on
    // the register-class wrapper used by the decoded operand.
    AddrOverlapsFirst |=
        MRI.regsOverlap(Ops.Regs[1], DstRegs[I]) ||
        toAsmRegName(MRI, Ops.Regs[1]) == toAsmRegName(MRI, DstRegs[I]);
  }

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
  return {
      ToMnem.str() + " " + Dst.first + ", " + Addr + ", " + Data0 +
          fmtOffset(Ops.Off0),
      ToMnem.str() + " " + Dst.second + ", " + Addr + ", " + Data1 +
          fmtOffset(Ops.Off1),
  };
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

// A non-stride store with adjacent offsets and consecutive data VGPRs is one
// store of twice the width. Keeping it as one DS instruction preserves the DS
// counter and lets the rewrite stay in place, which is important when the
// source site is a branch target with no safe trampoline window.
std::optional<std::pair<std::string, bool>>
getContiguousDs2AddrStore(const LLVMState &LS,
                          const InternalDecodedInst &DI) {
  if (DI.Mnemonic != "ds_store_2addr_b32" &&
      DI.Mnemonic != "ds_store_2addr_b64")
    return std::nullopt;

  std::optional<DsOperands> Ops =
      extractDsOperands(DI.Inst, DI.Mnemonic, LS);
  const uint32_t ElementBytes = Ops && Ops->IsB64 ? 8 : 4;
  if (!Ops || Ops->Regs.size() < 3 ||
      Ops->Off1 != Ops->Off0 + ElementBytes)
    return std::nullopt;

  const MCRegisterInfo &MRI = *Ops->MRI;
  auto VgprIndex = [&](MCRegister Reg) -> std::optional<unsigned> {
    std::string NameStorage = toAsmRegName(MRI, Reg);
    StringRef Name(NameStorage);
    if (!Name.consume_front("v") || Name.contains('['))
      return std::nullopt;
    unsigned Index = 0;
    if (Name.getAsInteger(10, Index))
      return std::nullopt;
    return Index;
  };
  SmallVector<unsigned, 4> DataIndices;
  auto AppendData = [&](MCRegister Reg) {
    SmallVector<MCRegister, 4> Components =
        Ops->IsB64 ? getDirectSubRegs(Reg, MRI)
                   : SmallVector<MCRegister, 4>{Reg};
    const unsigned Width = Ops->IsB64 ? 2 : 1;
    if (Components.size() < Width)
      return false;
    for (unsigned I = 0; I < Width; ++I) {
      std::optional<unsigned> Index = VgprIndex(Components[I]);
      if (!Index)
        return false;
      DataIndices.push_back(*Index);
    }
    return true;
  };
  if (!AppendData(Ops->Regs[1]) || !AppendData(Ops->Regs[2]))
    return std::nullopt;
  for (unsigned I = 1; I < DataIndices.size(); ++I)
    if (DataIndices[I] != DataIndices[0] + I)
      return std::nullopt;

  // gfx1250 A0's single-address B64/B128 DS operands use the aligned
  // load/store register class, while DS2 data operands do not. An odd DS2
  // base such as v19 is therefore legal in the input but cannot be folded to
  // v[19:20]. Leave it on the split path, which emits scalar B32 stores.
  if (DataIndices.front() & 1)
    return std::nullopt;

  std::string Asm = (Ops->IsB64 ? "ds_store_b128 " : "ds_store_b64 ") +
                    toAsmRegName(MRI, Ops->Regs[0]) + ", v[" +
                    std::to_string(DataIndices.front()) + ":" +
                    std::to_string(DataIndices.back()) + "]" +
                    fmtOffset(Ops->Off0);
  return std::pair<std::string, bool>{std::move(Asm), Ops->IsB64};
}

bool patchContiguousDs2AddrStore(PatchContext &Ctx,
                                 const InternalDecodedInst &DI,
                                 std::pair<std::string, bool> Fold) {
  SmallVector<uint8_t> Bytes = assembleSingleInst(Fold.first, Ctx.LS);
  if (Bytes.size() != DI.Size)
    return false;

  RewriteRule Rule;
  Rule.ReplaceBytes = Bytes;
  if (!applyByteReplace(Rule, DI.Offset, DI.Size, Ctx.Text, Ctx.TextSize,
                        Ctx.LS))
    return false;

  Ctx.MutatedOffsets.insert(DI.Offset);
  Ctx.ReplacementCodeBySite.insert_or_assign(
      DI.Offset, SmallVector<uint8_t>(Bytes.begin(), Bytes.end()));
  Ctx.RequiredPatchApplied = true;
  log() << "hotswap: ds_2addr: folded contiguous "
        << (Fold.second ? "b64 store to b128" : "b32 store to b64")
        << " at 0x" << utohexstr(DI.Offset) << "\n";
  return true;
}

// -- patchDs2Addr -----------------------------------------------------------
//
// Expand one ds_*_2addr_* instruction (stride64 or non-stride64) into two
// single-address DS instructions, followed by an s_wait_dscnt 0 drain so both
// halves are guaranteed complete before any downstream DS consumer. Splitting
// one DS instruction into two perturbs the outstanding-DS instruction count
// that later s_wait_dscnt immediates encode; the local drain sidesteps that
// entirely (see the rationale in the body below).

bool patchDs2Addr(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  StringRef ToMnem = getDs2AddrReplacement(DI.Mnemonic);
  if (ToMnem.empty())
    return false;
  std::optional<std::pair<std::string, bool>> Fold =
      getContiguousDs2AddrStore(Ctx.LS, DI);
  bool HasAdjacentDeferredSite = llvm::any_of(
      Ctx.OutTrampolines, [&](const Trampoline &T) {
        return T.OriginalOffset + T.OriginalSize == DI.Offset;
      });
  bool NextSiteRequiresTrampoline = false;
  if (Idx + 1 < Ctx.Decoded.size()) {
    const InternalDecodedInst &Next = Ctx.Decoded[Idx + 1];
    if (Next.Offset == DI.Offset + DI.Size &&
        !getDs2AddrReplacement(Next.Mnemonic).empty()) {
      std::optional<std::pair<std::string, bool>> NextFold =
          getContiguousDs2AddrStore(Ctx.LS, Next);
      // Let a non-foldable current site use a local cave when one exists. If
      // it instead queues a trampoline, the next iteration observes that via
      // HasAdjacentDeferredSite. A non-foldable next site must be deferred
      // early so the current site cannot consume storage needed by the pair.
      NextSiteRequiresTrampoline = !NextFold;
    }
  }
  const bool ForceDeferredTrampoline =
      HasAdjacentDeferredSite || NextSiteRequiresTrampoline;
  if (Fold && !ForceDeferredTrampoline &&
      patchContiguousDs2AddrStore(Ctx, DI, std::move(*Fold)))
    return true;
  SmallVector<uint8_t> Replacement;
  if (Fold && ForceDeferredTrampoline) {
    SmallVector<uint8_t> Bytes = assembleSingleInst(Fold->first, Ctx.LS);
    if (Bytes.size() != DI.Size)
      return failRequiredPatch(Ctx);
    Replacement.assign(Bytes.begin(), Bytes.end());
    log() << "hotswap: ds_2addr: retained contiguous "
          << (Fold->second ? "b128" : "b64")
          << " fold in the deferred trampoline set at 0x"
          << utohexstr(DI.Offset) << "\n";
  } else {
    std::vector<std::string> Expanded =
        expandDs2Addr(DI.Inst, DI.Mnemonic, ToMnem, Ctx.LS);
    if (Expanded.empty()) {
      log() << "hotswap: error: ds_2addr expansion failed for: " << DI.Mnemonic
            << "\n";
      return failRequiredPatch(Ctx);
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
    SmallVector<uint8_t> Bytes = assembleSingleInst(Combined, Ctx.LS);
    if (Bytes.empty()) {
      log() << "hotswap: error: ds_2addr: assembly failed: " << Combined
            << "\n";
      return failRequiredPatch(Ctx);
    }
    Replacement.assign(Bytes.begin(), Bytes.end());
  }
  // DS2 encodings require the B0-to-A0 split even when the appended pool is
  // outside s_branch reach. Use the SCC-neutral SGPR-pair return rather than
  // leaving the original B0 instruction executable on A0.
  bool Emitted = false;
  if (ForceDeferredTrampoline) {
    Emitted = emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement,
                               /*AllowSafeFarReturn=*/true);
    if (Emitted) {
      Ctx.ReplacementCodeBySite.insert_or_assign(
          DI.Offset,
          SmallVector<uint8_t>(Replacement.begin(), Replacement.end()));
      log() << "hotswap: ds_2addr: kept adjacent mixed-width site 0x"
            << utohexstr(DI.Offset) << " in the deferred trampoline set\n";
    }
  }
  if (!Emitted)
    Emitted = emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement,
                                  /*AllowSafeFarReturn=*/true);
  if (!Emitted)
    return failRequiredPatch(Ctx);

  Ctx.RequiredPatchApplied = true;
  DI.Mnemonic = "<replaced>";
  return true;
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

bool isAlreadyTensorMaskPatched(const PatchContext &Ctx, size_t Idx,
                                MCRegister BaseMCReg) {
  if (Idx == 0)
    return false;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
  const MCInst &PI = Prev.Inst;
  if (Prev.Offset + Prev.Size != Ctx.Decoded[Idx].Offset ||
      Prev.Mnemonic != "s_pack_hh_b32_b16" || PI.getNumOperands() < 3)
    return false;
  if (!PI.getOperand(0).isReg() ||
      !MRI.regsOverlap(PI.getOperand(0).getReg(), BaseMCReg.id()))
    return false;
  if (!PI.getOperand(1).isImm() || PI.getOperand(1).getImm() != 0)
    return false;
  return PI.getOperand(2).isReg() &&
         MRI.regsOverlap(PI.getOperand(2).getReg(), BaseMCReg.id());
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
  if (std::optional<unsigned> Opt =
          Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize))
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
// TODO: the E5M3 patch open-codes this same scratch-SGPR reservation. Hoist
// SgprScratchAlloc / tryAllocScratchSgpr / commitScratchSgpr into shared
// infrastructure both patches call, rather than duplicating it.

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

  uint64_t Magnitude =
      Delta == std::numeric_limits<int64_t>::min()
          ? uint64_t{1} << 63
          : static_cast<uint64_t>(-Delta);
  if (CapturedPc < Magnitude)
    return std::nullopt;
  return CapturedPc - Magnitude;
}

std::optional<std::pair<MCRegister, int64_t>>
getTensorHotswapAddSetPc(ArrayRef<InternalDecodedInst> Decoded,
                         size_t SetPcIndex) {
  if (SetPcIndex == 0 || SetPcIndex >= Decoded.size())
    return std::nullopt;

  const InternalDecodedInst &Add = Decoded[SetPcIndex - 1];
  const InternalDecodedInst &SetPc = Decoded[SetPcIndex];
  if (Add.Mnemonic != "s_add_nc_u64" ||
      SetPc.Mnemonic != "s_set_pc_i64" ||
      Add.Offset > std::numeric_limits<uint64_t>::max() - Add.Size ||
      Add.Offset + Add.Size != SetPc.Offset ||
      Add.Inst.getNumOperands() != 3 || SetPc.Inst.getNumOperands() != 1 ||
      !Add.Inst.getOperand(0).isReg() ||
      !Add.Inst.getOperand(1).isReg() ||
      !Add.Inst.getOperand(2).isImm() ||
      !SetPc.Inst.getOperand(0).isReg())
    return std::nullopt;

  MCRegister Pair = Add.Inst.getOperand(0).getReg();
  if (!Pair.isValid() || Add.Inst.getOperand(1).getReg() != Pair.id() ||
      SetPc.Inst.getOperand(0).getReg() != Pair.id())
    return std::nullopt;
  return std::pair<MCRegister, int64_t>{Pair,
                                        Add.Inst.getOperand(2).getImm()};
}

std::optional<size_t>
findTensorDecodedIndex(ArrayRef<InternalDecodedInst> Decoded,
                       uint64_t Offset) {
  auto It = llvm::lower_bound(
      Decoded, Offset, [](const InternalDecodedInst &DI, uint64_t Value) {
        return DI.Offset < Value;
      });
  if (It == Decoded.end() || It->Offset != Offset)
    return std::nullopt;
  return It - Decoded.begin();
}

std::optional<uint64_t>
resolveTensorContiguousSetPc(ArrayRef<InternalDecodedInst> Decoded,
                             size_t SetPcIndex) {
  std::optional<std::pair<MCRegister, int64_t>> AddSet =
      getTensorHotswapAddSetPc(Decoded, SetPcIndex);
  if (!AddSet || SetPcIndex < 2)
    return std::nullopt;

  const InternalDecodedInst &GetPc = Decoded[SetPcIndex - 2];
  const InternalDecodedInst &Add = Decoded[SetPcIndex - 1];
  if (GetPc.Mnemonic != "s_get_pc_i64" ||
      GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size ||
      GetPc.Offset + GetPc.Size != Add.Offset ||
      GetPc.Inst.getNumOperands() != 1 ||
      !GetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != AddSet->first.id())
    return std::nullopt;
  return applyTensorSignedPcDelta(GetPc.Offset + GetPc.Size, AddSet->second);
}

std::optional<uint64_t>
resolveTensorSetPcTarget(ArrayRef<InternalDecodedInst> AllDecoded,
                         uint64_t SetPcOffset) {
  std::optional<size_t> Index =
      findTensorDecodedIndex(AllDecoded, SetPcOffset);
  if (!Index)
    return std::nullopt;
  return resolveTensorContiguousSetPc(AllDecoded, *Index);
}

struct TensorTrampolinePath {
  uint64_t ResumeOffset = 0;
  SmallVector<size_t, 16> Instructions;
};

std::optional<TensorTrampolinePath> findTensorTrampolinePath(
    ArrayRef<InternalDecodedInst> AllDecoded, uint64_t Target,
    const KernelTextRange &Range, const LLVMState &LS) {
  if (!LS.MIA)
    return std::nullopt;

  TensorTrampolinePath Result;
  size_t RemainingInstructions = AllDecoded.size();
  DenseSet<uint64_t> VisitedOffsets;
  while (RemainingInstructions != 0) {
    std::optional<size_t> Start =
        findTensorDecodedIndex(AllDecoded, Target);
    if (!Start)
      return std::nullopt;

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
        std::optional<uint64_t> Next =
            resolveTensorContiguousSetPc(AllDecoded, I);
        if (!Next)
          return std::nullopt;
        if (*Next >= Range.Begin && *Next < Range.End) {
          Result.ResumeOffset = *Next;
          return Result;
        }
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
          Result.ResumeOffset = Next;
          return Result;
        }
        Target = Next;
        FollowedHop = true;
        break;
      }
      if (Desc.isTerminator() ||
          LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))
        return std::nullopt;
    }
    if (!FollowedHop)
      return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::pair<uint64_t, SmallVector<size_t, 2>>>
resolveTensorRelayTarget(ArrayRef<InternalDecodedInst> AllDecoded,
                         const InternalDecodedInst &GetPc,
                         uint64_t RelayOffset) {
  std::optional<size_t> AddIndex =
      findTensorDecodedIndex(AllDecoded, RelayOffset);
  if (!AddIndex || *AddIndex + 1 >= AllDecoded.size())
    return std::nullopt;

  const size_t SetPcIndex = *AddIndex + 1;
  std::optional<std::pair<MCRegister, int64_t>> AddSet =
      getTensorHotswapAddSetPc(AllDecoded, SetPcIndex);
  if (!AddSet || GetPc.Mnemonic != "s_get_pc_i64" ||
      GetPc.Inst.getNumOperands() != 1 ||
      !GetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != AddSet->first.id() ||
      GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size)
    return std::nullopt;

  std::optional<uint64_t> Target = applyTensorSignedPcDelta(
      GetPc.Offset + GetPc.Size, AddSet->second);
  if (!Target)
    return std::nullopt;
  return std::pair<uint64_t, SmallVector<size_t, 2>>{
      *Target, SmallVector<size_t, 2>{*AddIndex, SetPcIndex}};
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
  std::optional<unsigned> Fact = getNumberedRegFactIndex(
      MCRegister(Op.getReg()), MRI, MaxSgprs, MaxVgprs);
  return Fact && State.test(*Fact);
}

void setRegLow16Fact(BitVector &State, MCRegister Reg, bool IsKnownZero,
                     const MCRegisterInfo &MRI, unsigned MaxSgprs,
                     unsigned MaxVgprs) {
  for (MCRegister Leaf :
       getNumberedRegLeaves(Reg, MRI, MaxSgprs, MaxVgprs)) {
    std::optional<unsigned> Fact =
        getNumberedRegFactIndex(Leaf, MRI, MaxSgprs, MaxVgprs);
    if (!Fact)
      continue;
    if (IsKnownZero)
      State.set(*Fact);
    else
      State.reset(*Fact);
  }
}

BitVector transferTensorDescriptorFacts(const InternalDecodedInst &DI,
                                        const BitVector &Input,
                                        const LLVMState &LS,
                                        unsigned MaxSgprs,
                                        unsigned MaxVgprs) {
  const MCInst &Inst = DI.Inst;
  const MCInstrDesc &Desc = LS.MCII->get(Inst.getOpcode());
  const MCRegisterInfo &MRI = *LS.MRI;
  BitVector Output = Input;

  const unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = 0; I < NumDefs; ++I) {
    const MCOperand &Def = Inst.getOperand(I);
    if (Def.isReg() && Def.getReg())
      setRegLow16Fact(Output, MCRegister(Def.getReg()), false, MRI,
                      MaxSgprs, MaxVgprs);
  }
  for (MCPhysReg Def : Desc.implicit_defs())
    setRegLow16Fact(Output, MCRegister(Def), false, MRI, MaxSgprs,
                    MaxVgprs);

  auto CopyOne = [&](unsigned DefOp, unsigned SourceOp) {
    if (DefOp >= Inst.getNumOperands() || SourceOp >= Inst.getNumOperands() ||
        !Inst.getOperand(DefOp).isReg() ||
        !Inst.getOperand(DefOp).getReg())
      return;
    bool Known = operandLow16KnownZero(Inst.getOperand(SourceOp), Input, MRI,
                                      MaxSgprs, MaxVgprs);
    setRegLow16Fact(Output, MCRegister(Inst.getOperand(DefOp).getReg()), Known,
                    MRI, MaxSgprs, MaxVgprs);
  };

  if (DI.Mnemonic == "s_mov_b32" ||
      StringRef(DI.Mnemonic).starts_with("v_mov_b32")) {
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
          std::optional<unsigned> SourceFact = getNumberedRegFactIndex(
              Src[I], MRI, MaxSgprs, MaxVgprs);
          setRegLow16Fact(Output, Dst[I],
                          SourceFact && Input.test(*SourceFact), MRI,
                          MaxSgprs, MaxVgprs);
        }
      }
    }
  } else if (StringRef(DI.Mnemonic).starts_with("v_dual_mov_b32") &&
             NumDefs == 2 &&
             Inst.getNumOperands() >= 4) {
    CopyOne(0, 2);
    CopyOne(1, 3);
  } else if (DI.Mnemonic == "v_readfirstlane_b32") {
    CopyOne(0, 1);
  } else if (DI.Mnemonic == "s_pack_hh_b32_b16" &&
             Inst.getNumOperands() >= 2 && Inst.getOperand(0).isReg() &&
             Inst.getOperand(0).getReg() && Inst.getOperand(1).isImm() &&
             Inst.getOperand(1).getImm() == 0) {
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
    for (MCRegister Leaf :
         getNumberedRegLeaves(Reg, MRI, MaxSgprs, MaxVgprs)) {
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

  if (DI.Mnemonic == "v_readfirstlane_b32" &&
      Inst.getNumOperands() >= 1 && Inst.getOperand(0).isReg() &&
      Inst.getOperand(0).getReg()) {
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

void analyzeTensorDescriptorRange(
    ArrayRef<InternalDecodedInst> Decoded,
    ArrayRef<InternalDecodedInst> AllDecoded,
    const KernelTextRange &Range, const LLVMState &LS, unsigned MaxSgprs,
    unsigned MaxVgprs, TensorDescriptorMustAnalysis &Result, BitVector &Seen) {
  SmallVector<size_t> GlobalIndices;
  auto First = llvm::lower_bound(
      Decoded, Range.Begin,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
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

  const unsigned Count = GlobalIndices.size();
  DenseMap<uint64_t, unsigned> OffsetToLocal;
  for (unsigned I = 0; I < Count; ++I)
    OffsetToLocal.try_emplace(Decoded[GlobalIndices[I]].Offset, I);

  DenseMap<unsigned, unsigned> TrackedSgprs;
  for (size_t GlobalIdx : GlobalIndices) {
    if (Decoded[GlobalIdx].Mnemonic != "tensor_load_to_lds")
      continue;
    MCRegister Base = getDescriptorBaseSgpr(Decoded[GlobalIdx].Inst, *LS.MRI);
    std::optional<unsigned> Fact = getNumberedRegFactIndex(
        Base, *LS.MRI, MaxSgprs, MaxVgprs);
    if (Fact && *Fact < MaxSgprs && !TrackedSgprs.contains(*Fact))
      TrackedSgprs.try_emplace(*Fact, TrackedSgprs.size());
  }

  std::vector<SmallVector<unsigned, 2>> Successors(Count);
  std::vector<SmallVector<TensorCfgPredecessor, 2>> Predecessors(Count);
  BitVector UnknownSuccessors(Count);
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
      if (HasFallthrough)
        addTensorCfgEdge(I, I + 1, Successors, Predecessors);
      continue;
    }

    if (IsBranch) {
      const bool IsIndirect = LS.MIA->isIndirectBranch(DI.Inst) ||
                              Desc.isIndirectBranch();
      const bool IsConditional = LS.MIA->isConditionalBranch(DI.Inst) ||
                                 Desc.isConditionalBranch();
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
          } else if (IsUnconditional) {
            std::optional<TensorTrampolinePath> Path =
                findTensorTrampolinePath(AllDecoded, Target, Range, LS);
            bool RelayCandidate = false;
            if (!Path && I != 0 && DI.Mnemonic == "s_branch") {
              const InternalDecodedInst &GetPc =
                  Decoded[GlobalIndices[I - 1]];
              if (GetPc.Offset <=
                      std::numeric_limits<uint64_t>::max() - GetPc.Size &&
                  GetPc.Offset + GetPc.Size == DI.Offset &&
                  GetPc.Mnemonic == "s_get_pc_i64") {
                RelayCandidate = true;
                auto Relay =
                    resolveTensorRelayTarget(AllDecoded, GetPc, Target);
                if (Relay) {
                  Path = findTensorTrampolinePath(AllDecoded, Relay->first,
                                                  Range, LS);
                  if (Path)
                    Path->Instructions.insert(Path->Instructions.begin(),
                                              Relay->second.begin(),
                                              Relay->second.end());
                }
              }
            }
            if (Path) {
              auto ResumeIt = OffsetToLocal.find(Path->ResumeOffset);
              if (ResumeIt == OffsetToLocal.end()) {
                UnknownSuccessors.set(I);
                TargetKnown = false;
              } else {
                addTensorCfgEdge(I, ResumeIt->second, Successors,
                                 Predecessors, Path->Instructions);
              }
            } else if (RelayCandidate) {
              UnknownSuccessors.set(I);
              TargetKnown = false;
            }
          }
        }
      }
      if (!TargetKnown)
        UnknownSuccessors.set(I);
      if (IsConditional && HasFallthrough)
        addTensorCfgEdge(I, I + 1, Successors, Predecessors);
      else if (!IsConditional && !IsUnconditional)
        UnknownSuccessors.set(I);
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
  }

  for (unsigned I : HotswapSetPcCandidates) {
    const InternalDecodedInst &SetPc = Decoded[GlobalIndices[I]];
    std::optional<uint64_t> Target =
        resolveTensorSetPcTarget(AllDecoded, SetPc.Offset);
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
        findTensorTrampolinePath(AllDecoded, *Target, Range, LS);
    if (!Path) {
      UnknownSuccessors.set(I);
      continue;
    }
    auto ResumeIt = OffsetToLocal.find(Path->ResumeOffset);
    if (ResumeIt == OffsetToLocal.end()) {
      UnknownSuccessors.set(I);
      continue;
    }
    addTensorCfgEdge(I, ResumeIt->second, Successors, Predecessors,
                     Path->Instructions);
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

  const unsigned FactCount = MaxSgprs + MaxVgprs;
  BitVector Top(FactCount, true);
  BitVector Bottom(FactCount);
  std::vector<BitVector> MustIn(Count, Top);
  std::vector<BitVector> MustOut(Count, Top);
  MustIn[0] = Bottom;
  MustOut[0] = transferTensorDescriptorFacts(
      Decoded[GlobalIndices[0]], Bottom, LS, MaxSgprs, MaxVgprs);

  TensorMaskDefState DefTop(TrackedSgprs.size(), TensorMaskDefTop);
  TensorMaskDefState DefUnknown(TrackedSgprs.size(), TensorMaskDefUnknown);
  std::vector<TensorMaskDefState> DefIn(Count, DefTop);
  std::vector<TensorMaskDefState> DefOut(Count, DefTop);
  DefIn[0] = DefUnknown;
  DefOut[0] = transferTensorMaskDefinitions(
      Decoded[GlobalIndices[0]], DefUnknown, TrackedSgprs, LS, MaxSgprs,
      MaxVgprs);

  bool Changed = true;
  unsigned Iterations = 0;
  const unsigned IterationLimit = std::max(Count + 1, FactCount + 1);
  while (Changed && Iterations++ < IterationLimit) {
    Changed = false;
    for (unsigned I = 0; I < Count; ++I) {
      if (!Reachable.test(I))
        continue;
      BitVector NewIn = I == 0 ? Bottom : Top;
      TensorMaskDefState NewDefIn = I == 0 ? DefUnknown : DefTop;
      bool SawPredecessor = I == 0;
      if (I != 0) {
        for (const TensorCfgPredecessor &Pred : Predecessors[I]) {
          if (!Reachable.test(Pred.From))
            continue;
          SawPredecessor = true;
          BitVector EdgeOut = MustOut[Pred.From];
          TensorMaskDefState EdgeDefOut = DefOut[Pred.From];
          for (size_t ExternalIdx : Pred.ExternalInstructions) {
            EdgeOut = transferTensorDescriptorFacts(
                AllDecoded[ExternalIdx], EdgeOut, LS, MaxSgprs, MaxVgprs);
            EdgeDefOut = transferTensorMaskDefinitions(
                AllDecoded[ExternalIdx], EdgeDefOut, TrackedSgprs, LS,
                MaxSgprs, MaxVgprs);
          }
          NewIn &= EdgeOut;
          meetTensorMaskDefinitions(NewDefIn, EdgeDefOut);
        }
        if (!SawPredecessor) {
          NewIn.reset();
          NewDefIn = DefUnknown;
        }
      }
      BitVector NewOut = transferTensorDescriptorFacts(
          Decoded[GlobalIndices[I]], NewIn, LS, MaxSgprs, MaxVgprs);
      TensorMaskDefState NewDefOut = transferTensorMaskDefinitions(
          Decoded[GlobalIndices[I]], NewDefIn, TrackedSgprs, LS, MaxSgprs,
          MaxVgprs);
      if (NewIn != MustIn[I] || NewOut != MustOut[I] ||
          NewDefIn != DefIn[I] || NewDefOut != DefOut[I]) {
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
    std::optional<unsigned> Fact = getNumberedRegFactIndex(
        Base, *LS.MRI, MaxSgprs, MaxVgprs);
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
    ArrayRef<KernelTextRange> KernelRanges, const LLVMState &LS,
    unsigned MaxSgprs, unsigned MaxVgprs) {
  TensorDescriptorMustAnalysis Result{
      BitVector(Decoded.size()),
      std::vector<uint64_t>(Decoded.size(), TensorMaskDefUnknown)};
  BitVector Seen(Decoded.size());
  if (!LS.MCII || !LS.MRI || MaxSgprs == 0 || MaxVgprs == 0)
    return Result;
  for (const KernelTextRange &Range : KernelRanges)
    analyzeTensorDescriptorRange(Decoded, AllDecoded, Range, LS, MaxSgprs,
                                 MaxVgprs, Result, Seen);
  return Result;
}

// -- patchTensorLoadToLdsA0 -------------------------------------------------
//
// Replace the canonical one-cycle scalar delay immediately before the tensor
// load with s_pack_hh_b32_b16. Tensor loads are PC-sensitive on gfx1250 A0, so
// they must remain at their linked address instead of executing in a sled or
// appended trampoline. The replaced delay is a scheduling hint, not a
// correctness wait: the pack occupies its issue slot and the hardware
// interlock covers the new pack-to-tensor dependency. This clears the
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

  if (isAlreadyTensorMaskPatched(Ctx, Idx, BaseMCReg))
    return false;

  if (Idx < Ctx.TensorDescriptorAnalysis.Low16KnownZero.size() &&
      Ctx.TensorDescriptorAnalysis.Low16KnownZero.test(Idx)) {
    log() << "hotswap: tensor_load_to_lds: descriptor low16 already zero at 0x"
          << utohexstr(DI.Offset) << "; tensor remains unchanged\n";
    DI.Mnemonic = "<replaced>";
    return false;
  }

  std::string BaseSreg = toAsmRegName(MRI, BaseMCReg);

  std::string PackAsm = "s_pack_hh_b32_b16 " + BaseSreg + ", 0, " + BaseSreg;
  SmallVector<uint8_t> PackBytes = assembleSingleInst(PackAsm, Ctx.LS);
  if (PackBytes.empty()) {
    log() << "hotswap: tensor_load_to_lds pack: assembly failed: " << PackAsm
          << "\n";
    return failRequiredPatch(Ctx);
  }

  if (Idx < Ctx.TensorDescriptorAnalysis.MaskDefinitionOffsets.size()) {
    uint64_t DefOffset =
        Ctx.TensorDescriptorAnalysis.MaskDefinitionOffsets[Idx];
    std::optional<size_t> DefIdx =
        findTensorDecodedIndex(Ctx.Decoded, DefOffset);
    if (DefIdx) {
      InternalDecodedInst &Def = Ctx.Decoded[*DefIdx];
      const MCInst &DefInst = Def.Inst;
      const bool IsMatchingReadFirstLane =
          (Def.Mnemonic == "v_readfirstlane_b32" ||
           Def.Mnemonic == "<replaced>") &&
          DefInst.getNumOperands() >= 1 && DefInst.getOperand(0).isReg() &&
          DefInst.getOperand(0).getReg() == BaseMCReg.id();
      auto Existing = Ctx.ReplacementCodeBySite.find(DefOffset);
      const bool AlreadyMasked =
          IsMatchingReadFirstLane && Def.Mnemonic == "<replaced>" &&
          Existing != Ctx.ReplacementCodeBySite.end() &&
          Existing->second.size() >= PackBytes.size() &&
          ArrayRef<uint8_t>(Existing->second).take_back(PackBytes.size()) ==
              ArrayRef<uint8_t>(PackBytes);
      if (AlreadyMasked) {
        log() << "hotswap: tensor_load_to_lds: reusing masked descriptor "
                 "definition at 0x"
              << utohexstr(Def.Offset) << "; tensor remains at 0x"
              << utohexstr(DI.Offset) << "\n";
        DI.Mnemonic = "<replaced>";
        return false;
      }
      if (IsMatchingReadFirstLane && Def.Offset < DI.Offset &&
          Def.Offset <= Ctx.TextSize && Def.Size <= Ctx.TextSize - Def.Offset) {
        SmallVector<uint8_t> Replacement;
        Replacement.append(Ctx.Text + Def.Offset,
                           Ctx.Text + Def.Offset + Def.Size);
        Replacement.append(PackBytes.begin(), PackBytes.end());
        if (!emitReplacementCode(Ctx, Def.Offset, Def.Size, Replacement,
                                 /*AllowSafeFarReturn=*/true))
          return failRequiredPatch(Ctx);

        log() << "hotswap: tensor_load_to_lds: masked unique descriptor "
                 "definition at 0x"
              << utohexstr(Def.Offset) << "; tensor remains at 0x"
              << utohexstr(DI.Offset) << "\n";
        Ctx.RequiredPatchApplied = true;
        Def.Mnemonic = "<replaced>";
        DI.Mnemonic = "<replaced>";
        return true;
      }
    }
  }

  SmallVector<uint8_t> DelayBytes = assembleSingleInst(
      "s_delay_alu instid0(SALU_CYCLE_1)", Ctx.LS);
  if (DelayBytes.empty()) {
    log() << "hotswap: tensor_load_to_lds delay assembly failed\n";
    return failRequiredPatch(Ctx);
  }

  if (Idx == 0) {
    log() << "hotswap: error: tensor_load_to_lds at 0x"
          << utohexstr(DI.Offset) << " has no preceding delay slot\n";
    return failRequiredPatch(Ctx);
  }

  InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
  ArrayRef<uint8_t> PrevBytes(Ctx.Text + Prev.Offset, Prev.Size);
  if (Prev.Mnemonic != "s_delay_alu" ||
      Prev.Offset + Prev.Size != DI.Offset ||
      PrevBytes != ArrayRef<uint8_t>(DelayBytes) ||
      Prev.Size != PackBytes.size()) {
    log() << "hotswap: error: tensor_load_to_lds at 0x"
          << utohexstr(DI.Offset)
          << " is not preceded by the canonical scalar delay\n";
    return failRequiredPatch(Ctx);
  }

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
// is rejected: the rewrite stays a no-op so the original (broken on A0)
// instruction is preserved and the failure is loud in the verbose log.

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
    return false;
  }

  std::optional<uint16_t> OffsetOpt = getAddtidOffset(DI.Inst);
  if (!OffsetOpt) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": missing/non-immediate offset\n";
    return false;
  }
  uint16_t Offset = *OffsetOpt;

  if (DI.Inst.getNumOperands() <= AddtidOpReg ||
      !DI.Inst.getOperand(AddtidOpReg).isReg() ||
      !DI.Inst.getOperand(AddtidOpReg).getReg()) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": missing register operand\n";
    return false;
  }

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  MCRegister Reg = MCRegister(DI.Inst.getOperand(AddtidOpReg).getReg());
  std::string RegName = toAsmRegName(MRI, Reg);
  if (RegName.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": cannot resolve register name\n";
    return false;
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
      return false;
    }

    std::string TmpName = ("v" + Twine(StoreScratch->Vgpr)).str();
    AsmLines = buildAddtidStoreAsm(TmpName, RegName, Offset, ToMnem);
  }

  std::string Combined;
  for (const std::string &Line : AsmLines)
    Combined += Line + "\n";
  SmallVector<uint8_t> Bytes = assembleSingleInst(Combined, Ctx.LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic
          << " trampoline assembly failed at 0x" << utohexstr(DI.Offset)
          << "\n";
    return false;
  }

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Bytes))
    return false;

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

enum class PerInstTrampolineKind {
  None,
  Ds2Addr,
  TensorLoadToLds,
  ClusterLoad,
  DsAddtid,
};

PerInstTrampolineKind getPerInstTrampolineKind(const PatchContext &Ctx,
                                               size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);
  if (Ctx.Config.RunB0A0Patches && !getDs2AddrReplacement(Mnem).empty())
    return PerInstTrampolineKind::Ds2Addr;
  if (Mnem == "tensor_load_to_lds" &&
      Ctx.Config.MaskPolicy != MaskWorkaroundPolicy::None)
    return PerInstTrampolineKind::TensorLoadToLds;
  if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::A0 && isClusterLoad(Mnem))
    return PerInstTrampolineKind::ClusterLoad;
  if (Ctx.Config.RunB0A0Patches && !getAddtidReplacement(Mnem).empty())
    return PerInstTrampolineKind::DsAddtid;
  return PerInstTrampolineKind::None;
}

} // anonymous namespace

TensorDescriptorMustAnalysis computeTensorDescriptorMustAnalysis(
    ArrayRef<InternalDecodedInst> Decoded,
    ArrayRef<InternalDecodedInst> AllDecoded,
    ArrayRef<KernelTextRange> KernelRanges, const LLVMState &LS,
    unsigned MaxSgprs, unsigned MaxVgprs) {
  return computeTensorDescriptorMustAnalysisImpl(
      Decoded, AllDecoded, KernelRanges, LS, MaxSgprs, MaxVgprs);
}

bool requiresPerInstTrampoline(const PatchContext &Ctx, size_t Idx) {
  return getPerInstTrampolineKind(Ctx, Idx) != PerInstTrampolineKind::None;
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
  switch (getPerInstTrampolineKind(Ctx, Idx)) {
  case PerInstTrampolineKind::Ds2Addr:
    return patchDs2Addr(Ctx, Idx) ? 1 : 0;
  case PerInstTrampolineKind::TensorLoadToLds:
    if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::A0)
      return patchTensorLoadToLdsA0(Ctx, Idx) ? 1 : 0;
    return patchTensorLoadToLdsB0(Ctx, Idx) ? 1 : 0;
  case PerInstTrampolineKind::ClusterLoad:
    return patchClusterLoadMaskA0(Ctx, Idx) ? 1 : 0;
  case PerInstTrampolineKind::DsAddtid:
    return patchDsAddtid(Ctx, Idx) ? 1 : 0;
  case PerInstTrampolineKind::None:
    return 0;
  }
  return 0;
}

void registerTrampolinePatch(HotswapPatchVTable &VT) {
  VT.applyTrampolinePatches = &applyTrampolinePatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
