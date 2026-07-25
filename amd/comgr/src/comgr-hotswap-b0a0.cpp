//===- comgr-hotswap-b0a0.cpp - GFX1250 B0-to-A0 patch dispatcher --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Dispatcher for B0-to-A0 silicon stepping patches and the
/// retargetCodeObject orchestrator that drives the full pipeline:
/// decode -> patch -> trampoline growth -> DWARF update.
///
/// Patch passes are dispatched through HotswapPatchVTable. The membership
/// list lives in comgr-hotswap-patches.def; each entry corresponds to one
/// slot on the vtable and one register*Patch function in a sibling
/// comgr-hotswap-patch-*.cpp. installHotswapPatches() walks the .def to
/// bind every slot. The vtable is exposed through getHotswapPatchVTable(),
/// a Meyers singleton whose initializer eagerly runs installHotswapPatches
/// on its private storage; C++11 [stmt.dcl]/4 guarantees this happens
/// exactly once and is safe under concurrent first access, so the
/// dispatcher and the amd_comgr_hotswap_rewrite entry point can fetch the
/// fully-bound vtable with no explicit synchronization.
/// This replaces the prior LLVM_ATTRIBUTE_WEAK + `#if !defined(_MSC_VER)`
/// override pattern, which silently disabled hotswap on Windows because
/// PE/COFF does not honour weak the way ELF does
/// (issue ROCm/llvm-project#2479).
///
//===----------------------------------------------------------------------===//

#include "comgr-env.h"
#include "comgr-hotswap-internal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <tuple>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// HotSwap rewrite profiling lives in comgr-hotswap-internal.h so the sibling
// comgr-hotswap-patch-*.cpp TUs can record into the same per-rewrite session.

// -- GFX1250 B0-to-A0 constants -----------------------------------------------
//
// All instruction encoding lives in LLVMState (s_branch opcode + pre-encoded
// s_nop bytes, populated at initLLVM time via the MC asm parser). This policy
// layer only carries ISA identifiers and register granularity -- no
// target-specific opcode bits should land here.

static constexpr unsigned Gfx1250MaxVgprs = 1024;
// GFX1250 wave32 VGPR ENCODING granularity is 16 (per
// AMDGPUBaseInfo::getVGPREncodingGranule with Feature1024AddressableVGPRs),
// not the 8 used by earlier GFX10/11 wave32. Used by ElfView's KD
// decode/encode helpers (getKernelVgprCount /
// updateKernelDescriptorVgprCount) to
// interpret COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT.
// GFX12 wave32: 106 user-addressable SGPRs (s0-s105); s106-s107 are VCC.
static constexpr unsigned Gfx1250MaxSgprs = 106;
static constexpr unsigned Gfx1250VgprGranuleSize = 16;

/// Build the default RewriteConfig used for the GFX1250 B0-to-A0 rewrite:
/// fills in the identity source / target ISA (both gfx1250) and the
/// AMDGPU register granularity constants consumed by
/// ElfView::updateKernelDescriptorVgprCount. Instruction-encoding state is not
/// carried in RewriteConfig; see LLVMState for the s_branch opcode and
/// pre-encoded s_nop bytes.
static RewriteConfig makeGfx1250B0A0Config() {
  // `Config` / `Cfg` are reserved below: `Config` always names a
  // RewriteConfig; `Cfg` is only used for the CFG (control-flow graph)
  // local in applyGfx1250B0toA0Rules.
  RewriteConfig Config;
  Config.SourceIsa = "amdgcn-amd-amdhsa--gfx1250";
  Config.TargetIsa = "amdgcn-amd-amdhsa--gfx1250";
  Config.TargetCpu = "gfx1250";
  Config.MaxVgprs = Gfx1250MaxVgprs;
  Config.MaxSgprs = Gfx1250MaxSgprs;
  Config.VgprGranuleSize = Gfx1250VgprGranuleSize;
  return Config;
}

static bool appendCodeEndGuard(std::vector<Trampoline> &Growth,
                               uint64_t GuardBytes, const LLVMState &LS) {
  if (GuardBytes == 0)
    return true;

  SmallVector<uint8_t> CodeEnd = assembleSingleInst("s_code_end", LS);
  if (CodeEnd.empty()) {
    log() << "hotswap: error: failed to assemble s_code_end for trampoline "
          << "prefetch guard.\n";
    return false;
  }
  if (GuardBytes % CodeEnd.size() != 0) {
    log() << "hotswap: error: trampoline prefetch guard size " << GuardBytes
          << " is not a multiple of s_code_end size " << CodeEnd.size()
          << ".\n";
    return false;
  }

  Trampoline Guard;
  while (static_cast<uint64_t>(Guard.Bytes.size()) < GuardBytes)
    Guard.Bytes.append(CodeEnd.begin(), CodeEnd.end());
  Growth.push_back(std::move(Guard));
  return true;
}

static std::optional<uint32_t>
getMaxOriginalKernelInstPrefSize(const ElfView &Elf, const LLVMState &LS) {
  ArrayRef<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
  uint32_t MaxOriginalInstPrefLines = 0;
  for (const KernelDescriptorInfo &KD : Descriptors) {
    std::optional<uint32_t> OriginalInstPrefLines =
        Elf.getKernelDescriptorInstPrefSize(KD.KernelName, LS.Cpu);
    if (!OriginalInstPrefLines)
      return std::nullopt;
    MaxOriginalInstPrefLines =
        std::max(MaxOriginalInstPrefLines, *OriginalInstPrefLines);
  }
  return MaxOriginalInstPrefLines;
}

static bool
appendDeferredTrampolinePrefetchGuard(const ElfView &Elf, const LLVMState &LS,
                                      std::vector<Trampoline> &Growth) {
  // Deferred instruction-rewrite trampolines are reached from the original
  // kernel entries, so their trailing guard follows the original descriptor
  // prefetch size. Kernel-entry stubs clamp their own descriptor prefetch.
  std::optional<uint32_t> MaxOriginalInstPrefLines =
      getMaxOriginalKernelInstPrefSize(Elf, LS);
  if (!MaxOriginalInstPrefLines)
    return false;

  uint64_t GuardBytes = static_cast<uint64_t>(*MaxOriginalInstPrefLines) *
                        KernelEntryInstPrefUnitBytes;
  if (!appendCodeEndGuard(Growth, GuardBytes, LS))
    return false;

  log() << "hotswap: appended " << GuardBytes
        << " trampoline prefetch guard bytes\n";
  return true;
}

// -- Forward declarations for liveness/DWARF stubs ----------------------------
//
// These have weak default definitions below. The apply* patch families use
// HotswapPatchVTable dispatch; these lower-level helpers stay on weak stubs
// until a real implementation lands, at which point they should migrate to
// an explicit registration contract as well.

CFG buildCfg(ArrayRef<InternalDecodedInst> Decoded, const MCInstrInfo &);
LivenessInfo computeLiveness(ArrayRef<InternalDecodedInst> Decoded, const CFG &,
                             const MCInstrInfo &, const MCRegisterInfo &,
                             unsigned MaxVgprs);
RegDefUse getInstRegDefUse(const MCInst &, const MCInstrInfo &,
                           const MCRegisterInfo &);
int64_t getBranchImm(const MCInst &);
bool verifyPatchCorrectness(const uint8_t *, uint64_t, const LLVMState &,
                            ArrayRef<ScratchPatchInfo>, unsigned);
bool addTrampolineSymbols(WritableMemoryBuffer &ElfBuf,
                          ArrayRef<Trampoline> Trampolines,
                          uint64_t TextSizeBefore, unsigned TextSectionIdx);
bool patchDebugLine(WritableMemoryBuffer &ElfBuf,
                    ArrayRef<Trampoline> Trampolines, uint64_t TextSizeBefore,
                    uint64_t TextAddr);
void patchDebugRanges(uint8_t *Elf, size_t ElfSize, uint64_t TextAddr,
                      uint64_t TextSizeBefore, uint64_t TrampTotal);
void patchDebugInfo(uint8_t *Elf, size_t ElfSize, uint64_t TextAddr,
                    uint64_t TextSizeBefore, uint64_t TrampTotal);
void patchDebugFrame(uint8_t *Elf, size_t ElfSize, uint64_t TextAddr,
                     uint64_t TextSizeBefore, uint64_t TrampTotal);

// -- HotswapPatchVTable plumbing ----------------------------------------------
//
// Patch-module forward declarations live in comgr-hotswap-internal.h
// (driven off the same comgr-hotswap-patches.def), so libamd_comgr and
// the unit tests share one prototype source. Here we supply the
// singleton accessor and the installer that walks the .def to invoke
// each register*Patch. A .def entry without a matching register*Patch
// definition produces a link error at libamd_comgr link time.
//
// installHotswapPatches() is exposed in the header so unit tests can
// bind a local HotswapPatchVTable for fixture-style coverage. Production
// code never calls it directly: getHotswapPatchVTable()'s initializer
// invokes it eagerly on the singleton's private storage, which the C++11
// magic-static rule guarantees runs exactly once even under concurrent
// first access. That removes both the explicit std::call_once at the
// retargetCodeObject entry point and any inter-TU static-init order
// dependency on the patch modules.

void installHotswapPatches(HotswapPatchVTable &VT) {
#define HOTSWAP_PATCH(Name) register##Name##Patch(VT);
#include "comgr-hotswap-patches.def"
#undef HOTSWAP_PATCH
}

HotswapPatchVTable &getHotswapPatchVTable() {
  static HotswapPatchVTable VT = [] {
    HotswapPatchVTable Tmp;
    installHotswapPatches(Tmp);
    return Tmp;
  }();
  return VT;
}

// -- Weak-symbol liveness stubs -----------------------------------------------
//
// Conservative defaults: all VGPRs reported live. VgprAllocator will
// allocate above KD count (correct but suboptimal until the real liveness
// layer lands).

LLVM_ATTRIBUTE_WEAK CFG buildCfg(ArrayRef<InternalDecodedInst> Decoded,
                                 const MCInstrInfo &) {
  (void)Decoded;
  return CFG();
}

LLVM_ATTRIBUTE_WEAK LivenessInfo computeLiveness(
    ArrayRef<InternalDecodedInst> Decoded, const CFG &, const MCInstrInfo &,
    const MCRegisterInfo &, unsigned MaxVgprs) {
  (void)Decoded;
  LivenessInfo Info;
  Info.setConservativeAllLive(MaxVgprs);
  Info.Converged = true;
  return Info;
}

LLVM_ATTRIBUTE_WEAK RegDefUse getInstRegDefUse(const MCInst &,
                                               const MCInstrInfo &,
                                               const MCRegisterInfo &) {
  return {};
}

LLVM_ATTRIBUTE_WEAK int64_t getBranchImm(const MCInst &) { return 0; }

LLVM_ATTRIBUTE_WEAK bool verifyPatchCorrectness(const uint8_t *, uint64_t,
                                                const LLVMState &,
                                                ArrayRef<ScratchPatchInfo>,
                                                unsigned) {
  return true;
}

// -- Weak-symbol DWARF stubs --------------------------------------------------

LLVM_ATTRIBUTE_WEAK bool addTrampolineSymbols(WritableMemoryBuffer &,
                                              ArrayRef<Trampoline>, uint64_t,
                                              unsigned) {
  return true;
}
LLVM_ATTRIBUTE_WEAK bool patchDebugLine(WritableMemoryBuffer &,
                                        ArrayRef<Trampoline>, uint64_t,
                                        uint64_t) {
  return true;
}
LLVM_ATTRIBUTE_WEAK void patchDebugRanges(uint8_t *, size_t, uint64_t, uint64_t,
                                          uint64_t) {}
LLVM_ATTRIBUTE_WEAK void patchDebugInfo(uint8_t *, size_t, uint64_t, uint64_t,
                                        uint64_t) {}
LLVM_ATTRIBUTE_WEAK void patchDebugFrame(uint8_t *, size_t, uint64_t, uint64_t,
                                         uint64_t) {}

// -- NOP sled scanning --------------------------------------------------------

static void appendNopSledIfLarge(std::vector<NopSled> &Sleds, uint64_t Start,
                                 uint64_t End, uint64_t FunctionStart,
                                 uint64_t FunctionEnd) {
  if (End - Start >= MinNopSledSize)
    Sleds.push_back({Start, End, Start, FunctionStart, FunctionEnd});
}

static void appendNopSledIfLarge(std::vector<NopSled> &Sleds, uint64_t Start,
                                 uint64_t End,
                                 const ElfView::FunctionTextRange &Range) {
  appendNopSledIfLarge(Sleds, Start, End, Range.Begin, Range.End);
}

/// Scan \p Decoded for runs of consecutive `s_nop` instructions at least
/// MinNopSledSize bytes long and return the resulting NopSled list. Each sled
/// records its owning function range so emitReplacementCode can only borrow
/// padding from the same kernel as the instruction being patched. NOPs outside
/// any sized function symbol are ignored.
static std::vector<NopSled>
buildNopSledMap(ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
                const ElfView &Elf) {
  std::vector<NopSled> Sleds;
  bool HasActiveRange = false;
  ElfView::FunctionTextRange ActiveRange;
  uint64_t Start = 0;
  uint64_t End = 0;

  for (const InternalDecodedInst &DI : Decoded) {
    if (DI.Inst.getOpcode() != LS.SNopOpcode) {
      if (HasActiveRange)
        appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
      HasActiveRange = false;
      continue;
    }

    std::optional<ElfView::FunctionTextRange> Range =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    if (!Range || DI.Size > Range->End - DI.Offset) {
      if (HasActiveRange)
        appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
      HasActiveRange = false;
      continue;
    }

    if (!HasActiveRange || ActiveRange.Begin != Range->Begin ||
        ActiveRange.End != Range->End || DI.Offset != End) {
      if (HasActiveRange)
        appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
      ActiveRange = *Range;
      HasActiveRange = true;
      Start = DI.Offset;
    }
    End = DI.Offset + DI.Size;
  }

  if (HasActiveRange)
    appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
  return Sleds;
}

/// A direct branch/call target into a NOP run makes that offset and every
/// following byte in the run reachable by fallthrough, so only the prefix
/// before the first target remains available as scratch padding.
static void
truncateNopSledsAtDirectTargets(std::vector<NopSled> &Sleds,
                                const DenseSet<uint64_t> &DirectBranchTargets) {
  if (DirectBranchTargets.empty() || Sleds.empty())
    return;

  std::vector<NopSled> Filtered;
  Filtered.reserve(Sleds.size());
  uint64_t Truncated = 0;
  for (const NopSled &Sled : Sleds) {
    uint64_t End = Sled.End;
    for (uint64_t Target : DirectBranchTargets)
      if (Target >= Sled.Start && Target < End)
        End = Target;
    if (End != Sled.End)
      ++Truncated;
    appendNopSledIfLarge(Filtered, Sled.Start, End, Sled.FunctionStart,
                         Sled.FunctionEnd);
  }
  if (Truncated != 0)
    log() << "hotswap: protected " << Truncated
          << " NOP sled(s) containing direct branch/call target(s)\n";
  Sleds = std::move(Filtered);
}

// -- Sled-or-trampoline code emission -----------------------------------------

/// Emit the replacement code for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize) into a nearby NOP sled: writes \p Replacement
/// into the sled, appends a branch-back to the next instruction after the
/// original site, overwrites the original site with a branch-forward to the
/// sled, and pads the leftover bytes of the original slot with cached s_nop
/// bytes. Advances \c Sled.WritePos by the amount consumed. Returns false if
/// either branch encoding fails. Branches are encoded before any bytes are
/// written so a failure leaves \c Ctx.Text and \c Sled.WritePos unchanged.
[[nodiscard]] bool emitToNopSled(PatchContext &Ctx, NopSled &Sled,
                                 uint64_t InstOffset, uint32_t InstSize,
                                 ArrayRef<uint8_t> Replacement) {
  const LLVMState &LS = Ctx.LS;
  SmallVector<uint8_t> BrBack = LS.encodeSBranch(
      Sled.WritePos + Replacement.size(), InstOffset + InstSize);
  if (BrBack.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-back "
          << "at sled offset 0x"
          << utohexstr(Sled.WritePos + Replacement.size()) << " -> 0x"
          << utohexstr(InstOffset + InstSize) << " failed.\n";
    return false;
  }

  SmallVector<uint8_t> BrFwd = LS.encodeSBranch(InstOffset, Sled.WritePos);
  if (BrFwd.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-fwd "
          << "at original offset 0x" << utohexstr(InstOffset) << " -> sled 0x"
          << utohexstr(Sled.WritePos) << " failed.\n";
    return false;
  }

  std::memcpy(Ctx.Text + Sled.WritePos, Replacement.data(), Replacement.size());
  std::memcpy(Ctx.Text + Sled.WritePos + Replacement.size(), BrBack.data(),
              BrBack.size());
  std::memcpy(Ctx.Text + InstOffset, BrFwd.data(), BrFwd.size());

  // Pad the tail of the replaced instruction slot with cached s_nop bytes
  // (pre-encoded in LLVMState at initLLVM() time).
  for (uint32_t I = MinInstSize; I < InstSize; I += MinInstSize)
    std::memcpy(Ctx.Text + InstOffset + I, LS.SNopBytes.data(), MinInstSize);

  Sled.WritePos += Replacement.size() + MinInstSize;
  // Count-only row: patch placed in-line via a nearby NOP sled, no trampoline.
  Ctx.Profile.count(HotswapMetric::JumpNopSled);
  return true;
}

std::optional<SmallVector<uint8_t>>
encodeSetPCLongBranch(const LLVMState &LS, uint64_t FromOffset,
                      uint64_t TargetOffset, unsigned SgprBase, bool UseVcc) {
  if (!UseVcc && (SgprBase & 1u) != 0) {
    log() << "hotswap: error: set-PC long branch requires an aligned "
             "SGPR pair, got s"
          << SgprBase << "\n";
    return std::nullopt;
  }

  const std::string Pair = UseVcc ? "vcc"
                                  : "s[" + std::to_string(SgprBase) + ":" +
                                        std::to_string(SgprBase + 1) + "]";
  SmallVector<uint8_t> GetPc = assembleSingleInst("s_get_pc_i64 " + Pair, LS);
  if (GetPc.empty())
    return std::nullopt;
  std::optional<uint64_t> PcBase =
      checkedAddUint64(FromOffset, GetPc.size(), "set-PC long branch PC base");
  if (!PcBase)
    return std::nullopt;
  uint64_t Delta = TargetOffset - *PcBase;
  // AMDGPU/SOPInstructions.td defines S_ADD_U64 as an SOP2_64 outside the
  // Defs = [SCC] scope and maps its gfx12 encoding to s_add_nc_u64. It can
  // therefore add the complete PC displacement without saving or clobbering
  // SCC.
  SmallVector<std::string, 3> AsmLines;
  AsmLines.push_back("s_get_pc_i64 " + Pair);
  AsmLines.push_back("s_add_nc_u64 " + Pair + ", " + Pair + ", 0x" +
                     utohexstr(Delta));
  AsmLines.push_back("s_set_pc_i64 " + Pair);
  SmallVector<uint8_t> Bytes = assembleInstructions(joinAsmLines(AsmLines), LS);
  if (Bytes.empty() || Bytes.size() > SetPcReturnReserveBytes) {
    log() << "hotswap: error: failed to assemble SCC-neutral set-PC branch via "
          << Pair << "\n";
    return std::nullopt;
  }
  return Bytes;
}

static bool isSetPcDeltaInline(uint64_t Delta) {
  int64_t SignedDelta = static_cast<int64_t>(Delta);
  if (SignedDelta >= -16 && SignedDelta <= 64)
    return true;

  // AMDGPU::isInlinableLiteral64 is target-internal and unavailable to
  // standalone COMGR builds. Keep this mirror in sync with it. HotSwap only
  // invokes this gfx1250 path, whose subtarget includes the inv2pi inline
  // immediate.
  switch (Delta) {
  case 0x3ff0000000000000ULL: // 1.0
  case 0xbff0000000000000ULL: // -1.0
  case 0x3fe0000000000000ULL: // 0.5
  case 0xbfe0000000000000ULL: // -0.5
  case 0x4000000000000000ULL: // 2.0
  case 0xc000000000000000ULL: // -2.0
  case 0x4010000000000000ULL: // 4.0
  case 0xc010000000000000ULL: // -4.0
  case 0x3fc45f306dc9c882ULL: // 1 / (2 * pi)
    return true;
  default:
    return false;
  }
}

static std::optional<uint32_t>
getSetPcLongBranchLayoutSize(uint64_t FromOffset, uint64_t TargetOffset) {
  std::optional<uint64_t> PcBase = checkedAddUint64(
      FromOffset, MinInstSize, "set-PC long branch layout PC base");
  if (!PcBase)
    return std::nullopt;
  uint64_t Delta = TargetOffset - *PcBase;

  // This model is gfx1250-specific. s_get_pc_i64 and s_set_pc_i64 each occupy
  // one dword. The intervening s_add_nc_u64 occupies one dword for an inline
  // immediate, two for a non-negative signed-32-bit literal, and three for a
  // 64-bit literal.
  if (isSetPcDeltaInline(Delta))
    return 3 * MinInstSize;
  if (Delta <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
    return 4 * MinInstSize;
  return SetPcReturnReserveBytes;
}

static std::optional<SmallVector<uint8_t>>
encodeSetPcGateway(const LLVMState &LS, uint64_t FromOffset,
                   uint64_t TargetOffset, unsigned SgprBase, bool UseVcc,
                   bool PreserveVcc) {
  SmallVector<uint8_t> Bytes;
  uint64_t SetPcOffset = FromOffset;
  if (PreserveVcc) {
    if (!UseVcc) {
      log() << "hotswap: error: VCC-preserving gateway does not use VCC\n";
      return std::nullopt;
    }
    Bytes = assembleSingleInst(
        "s_mov_b32 s" + std::to_string(SgprBase) + ", vcc_lo", LS);
    if (Bytes.size() != VccMoveBytes)
      return std::nullopt;
    std::optional<uint64_t> Offset = checkedAddUint64(
        FromOffset, Bytes.size(), "VCC-preserving set-PC gateway offset");
    if (!Offset)
      return std::nullopt;
    SetPcOffset = *Offset;
  }

  std::optional<SmallVector<uint8_t>> SetPc =
      encodeSetPCLongBranch(LS, SetPcOffset, TargetOffset, SgprBase, UseVcc);
  if (!SetPc)
    return std::nullopt;
  Bytes.append(SetPc->begin(), SetPc->end());
  return Bytes;
}

static std::optional<uint32_t>
getSetPcGatewayLayoutSize(uint64_t FromOffset, uint64_t TargetOffset,
                          unsigned SgprBase, bool UseVcc, bool PreserveVcc) {
  if (PreserveVcc && !UseVcc)
    return std::nullopt;
  if (!UseVcc && (SgprBase & 1u) != 0)
    return std::nullopt;

  uint64_t SetPcOffset = FromOffset;
  uint32_t PrefixBytes = 0;
  if (PreserveVcc) {
    std::optional<uint64_t> Offset =
        checkedAddUint64(FromOffset, VccMoveBytes,
                         "VCC-preserving set-PC gateway layout offset");
    if (!Offset)
      return std::nullopt;
    SetPcOffset = *Offset;
    PrefixBytes = VccMoveBytes;
  }

  std::optional<uint32_t> SetPcBytes =
      getSetPcLongBranchLayoutSize(SetPcOffset, TargetOffset);
  if (!SetPcBytes)
    return std::nullopt;
  return PrefixBytes + *SetPcBytes;
}

static bool gatewayRangeIsOccupied(
    uint64_t Begin, uint64_t Size,
    const DenseSet<uint64_t> *Occupied) {
  if (!Occupied)
    return false;
  for (uint64_t Offset = 0; Offset < Size; Offset += MinInstSize)
    if (Occupied->contains(Begin + Offset))
      return true;
  return false;
}

Expected<std::optional<EncodedSetPcGateway>>
findNearestSetPcGateway(std::vector<NopSled> &Gateways, const LLVMState &LS,
                        uint64_t FromOffset, uint64_t TargetOffset,
                        unsigned SgprBase, bool UseVcc, bool PreserveVcc,
                        const DenseSet<uint64_t> *Occupied) {
  NopSled *Best = nullptr;
  uint32_t BestLayoutSize = 0;
  uint64_t BestUsableEnd = 0;
  uint64_t BestDistance = std::numeric_limits<uint64_t>::max();
  for (NopSled &Sled : Gateways) {
    if (FromOffset < Sled.FunctionStart || FromOffset >= Sled.FunctionEnd)
      continue;
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    if (Sled.WritePos > UsableEnd)
      continue;
    uint64_t Distance = Sled.WritePos > FromOffset ? Sled.WritePos - FromOffset
                                                   : FromOffset - Sled.WritePos;
    if (Distance >= MaxSledDistance || Distance >= BestDistance ||
        LS.encodeSBranch(FromOffset, Sled.WritePos).empty())
      continue;
    std::optional<uint32_t> LayoutSize = getSetPcGatewayLayoutSize(
        Sled.WritePos, TargetOffset, SgprBase, UseVcc, PreserveVcc);
    if (!LayoutSize)
      return createStringError(
          Twine("failed to encode set-PC gateway at candidate offset 0x") +
          utohexstr(Sled.WritePos));
    if (*LayoutSize > UsableEnd - Sled.WritePos ||
        gatewayRangeIsOccupied(Sled.WritePos, *LayoutSize, Occupied))
      continue;

    Best = &Sled;
    BestLayoutSize = *LayoutSize;
    BestUsableEnd = UsableEnd;
    BestDistance = Distance;
  }
  if (!Best)
    return std::nullopt;
  std::optional<SmallVector<uint8_t>> BestBytes = encodeSetPcGateway(
      LS, Best->WritePos, TargetOffset, SgprBase, UseVcc, PreserveVcc);
  if (!BestBytes)
    return createStringError(
        Twine("failed to encode set-PC gateway at candidate offset 0x") +
        utohexstr(Best->WritePos));
  if (BestBytes->size() != BestLayoutSize ||
      BestBytes->size() > BestUsableEnd - Best->WritePos)
    return createStringError(
        Twine("set-PC gateway layout mismatch at candidate offset 0x") +
        utohexstr(Best->WritePos) + ": predicted " + Twine(BestLayoutSize) +
        " bytes, encoded " + Twine(BestBytes->size()) + " bytes");
  return std::optional<EncodedSetPcGateway>(
      EncodedSetPcGateway{Best, std::move(*BestBytes)});
}

/// Split a live-VCC gateway across an eight-byte nearby save/branch segment
/// and a 16-byte get-PC/add/set-PC segment. This serves functions whose safe
/// padding is fragmented too finely for the ordinary contiguous 20-byte
/// sequence, while preserving SCC and saving VCC before it is used as the
/// target pair.
std::optional<EncodedSplitVccGateway>
findSplitVccGateway(std::vector<NopSled> &Gateways, const LLVMState &LS,
                    uint64_t FromOffset, uint64_t TargetOffset,
                    unsigned SaveSgpr,
                    const DenseSet<uint64_t> *Occupied) {
  constexpr uint64_t PrimaryBytes = 2 * MinInstSize;
  constexpr uint64_t SecondaryBytes = SetPcForwardSequenceBytes - MinInstSize;
  SmallVector<size_t, 32> PrimaryCandidates;
  for (size_t I = 0; I != Gateways.size(); ++I) {
    const NopSled &Sled = Gateways[I];
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    if (Sled.GatewayOnly || FromOffset < Sled.FunctionStart ||
        FromOffset >= Sled.FunctionEnd || Sled.WritePos > UsableEnd ||
        PrimaryBytes > UsableEnd - Sled.WritePos ||
        gatewayRangeIsOccupied(Sled.WritePos, PrimaryBytes, Occupied) ||
        !isSBranchReachable(FromOffset, Sled.WritePos))
      continue;
    PrimaryCandidates.push_back(I);
  }
  llvm::sort(PrimaryCandidates, [&](size_t LHS, size_t RHS) {
    uint64_t L = Gateways[LHS].WritePos;
    uint64_t R = Gateways[RHS].WritePos;
    uint64_t LDistance = FromOffset > L ? FromOffset - L : L - FromOffset;
    uint64_t RDistance = FromOffset > R ? FromOffset - R : R - FromOffset;
    return LDistance < RDistance;
  });

  for (size_t PrimaryIndex : PrimaryCandidates) {
    const NopSled &Primary = Gateways[PrimaryIndex];
    uint64_t PrimaryOffset = Primary.WritePos;
    for (size_t SecondaryIndex = 0; SecondaryIndex != Gateways.size();
         ++SecondaryIndex) {
      if (SecondaryIndex == PrimaryIndex)
        continue;
      const NopSled &Secondary = Gateways[SecondaryIndex];
      uint64_t UsableEnd = std::min(Secondary.End, Secondary.FunctionEnd);
      if (FromOffset < Secondary.FunctionStart ||
          FromOffset >= Secondary.FunctionEnd ||
          Secondary.WritePos > UsableEnd ||
          SecondaryBytes > UsableEnd - Secondary.WritePos ||
          gatewayRangeIsOccupied(Secondary.WritePos, SecondaryBytes,
                                 Occupied) ||
          (PrimaryOffset < Secondary.WritePos + SecondaryBytes &&
           Secondary.WritePos < PrimaryOffset + PrimaryBytes) ||
          !isSBranchReachable(PrimaryOffset + MinInstSize, Secondary.WritePos))
        continue;
      std::optional<SmallVector<uint8_t>> SetPc = encodeSetPCLongBranch(
          LS, Secondary.WritePos, TargetOffset, SaveSgpr, /*UseVcc=*/true);
      if (!SetPc || SetPc->size() > SecondaryBytes)
        continue;
      SmallVector<uint8_t> Save = assembleSingleInst(
          "s_mov_b32 s" + std::to_string(SaveSgpr) + ", vcc_lo", LS);
      SmallVector<uint8_t> Branch =
          LS.encodeSBranch(PrimaryOffset + MinInstSize, Secondary.WritePos);
      if (Save.size() != MinInstSize || Branch.size() != MinInstSize)
        continue;
      Save.append(Branch);
      return EncodedSplitVccGateway{PrimaryIndex, SecondaryIndex,
                                    std::move(Save), std::move(*SetPc)};
    }
  }
  return std::nullopt;
}

static std::optional<unsigned> numberedSgprIndex(const MCRegisterInfo &MRI,
                                                 MCRegister Reg) {
  // TODO(https://github.com/ROCm/llvm-project/issues/3350): Replace this
  // register-name fallback with a public AMDGPU MC hardware-index helper.
  if (!Reg.isValid())
    return std::nullopt;
  StringRef Name(MRI.getName(Reg));
  if (!Name.consume_front("SGPR") || Name.empty() || Name.contains('_'))
    return std::nullopt;
  unsigned Index = 0;
  if (Name.getAsInteger(10, Index))
    return std::nullopt;
  return Index;
}

static std::optional<unsigned>
numberedSgprPairLowIndex(const MCRegisterInfo &MRI, MCRegister Reg) {
  SmallVector<unsigned, 2> Indices;
  auto Add = [&](MCRegister Candidate) {
    std::optional<unsigned> Index = numberedSgprIndex(MRI, Candidate);
    if (Index && !llvm::is_contained(Indices, *Index))
      Indices.push_back(*Index);
  };
  Add(Reg);
  for (MCPhysReg Sub : MRI.subregs(Reg))
    Add(MCRegister(Sub));
  llvm::sort(Indices);
  if (Indices.size() != 2 || Indices[1] != Indices[0] + 1)
    return std::nullopt;
  return Indices[0];
}

static bool updateNumberedSgprHighWatermark(const MCRegisterInfo &MRI,
                                            MCRegister Reg, unsigned MaxSgprs,
                                            unsigned &HighWatermark,
                                            StringRef Context) {
  SmallVector<MCRegister, 8> Candidates;
  Candidates.push_back(Reg);
  for (MCPhysReg Sub : MRI.subregs(Reg))
    Candidates.push_back(MCRegister(Sub));

  for (MCRegister Candidate : Candidates) {
    std::optional<unsigned> Index = numberedSgprIndex(MRI, Candidate);
    if (!Index)
      continue;
    if (*Index >= MaxSgprs) {
      log() << "hotswap: error: " << Context << ": numbered SGPR s" << *Index
            << " exceeds the addressable limit s" << (MaxSgprs - 1) << "\n";
      return false;
    }
    HighWatermark = std::max(HighWatermark, *Index + 1);
  }
  return true;
}

static bool isVccRegister(const LLVMState &LS, MCRegister Reg) {
  return Reg.isValid() && LS.VCCRegister.isValid() &&
         LS.MRI->regsOverlap(Reg, LS.VCCRegister);
}

static bool instructionUsesVcc(const LLVMState &LS,
                               const InternalDecodedInst &DI) {
  for (const MCOperand &Op : DI.Inst)
    if (Op.isReg() && Op.getReg() && isVccRegister(LS, MCRegister(Op.getReg())))
      return true;

  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  for (MCPhysReg Reg : Desc.implicit_uses())
    if (isVccRegister(LS, MCRegister(Reg)))
      return true;
  for (MCPhysReg Reg : Desc.implicit_defs())
    if (isVccRegister(LS, MCRegister(Reg)))
      return true;
  return false;
}

static SafeSgprUsageSummary
summarizeSafeSgprUsage(PatchContext &Ctx,
                       ArrayRef<InternalDecodedInst> Instructions,
                       StringRef Context) {
  SafeSgprUsageSummary Summary;
  for (const InternalDecodedInst &DI : Instructions) {
    Summary.UsesVcc |= instructionUsesVcc(Ctx.LS, DI);
    Summary.HasCall |= Ctx.LS.MIA && Ctx.LS.MIA->isCall(DI.Inst);
    for (const MCOperand &Op : DI.Inst) {
      if (!Op.isReg() || !Op.getReg())
        continue;
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Op.getReg()),
                                           Ctx.Config.MaxSgprs,
                                           Summary.HighWatermark, Context)) {
        Summary.Valid = false;
        return Summary;
      }
    }

    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    for (MCPhysReg Reg : Desc.implicit_uses())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs,
                                           Summary.HighWatermark, Context)) {
        Summary.Valid = false;
        return Summary;
      }
    for (MCPhysReg Reg : Desc.implicit_defs())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs,
                                           Summary.HighWatermark, Context)) {
        Summary.Valid = false;
        return Summary;
      }
  }
  return Summary;
}

static std::string findKernelOwnerAtTextOffset(PatchContext &Ctx,
                                               uint64_t TextOffset) {
  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(TextOffset);
  if (!FunctionRange)
    return Ctx.Elf.findKernelAtAddress(TextOffset + Ctx.Elf.textAddr());

  std::pair<uint64_t, uint64_t> Key{FunctionRange->Begin, FunctionRange->End};
  auto Cached = Ctx.FunctionKernelOwner.find(Key);
  if (Cached != Ctx.FunctionKernelOwner.end())
    return Cached->second;

  std::string Owner =
      Ctx.Elf.findKernelAtAddress(TextOffset + Ctx.Elf.textAddr());
  Ctx.FunctionKernelOwner.try_emplace(Key, Owner);
  return Owner;
}

std::optional<SafeSgprScratchBlock>
findSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset, unsigned Count,
                         unsigned Alignment, StringRef Context,
                         bool ReportNoSpace) {
  if (Count == 0 || Alignment == 0 || (Alignment & (Alignment - 1)) != 0) {
    log() << "hotswap: error: " << Context
          << ": invalid global SGPR block request (count=" << Count
          << ", alignment=" << Alignment << ")\n";
    return std::nullopt;
  }

  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(TextOffset);
  std::string Owner = findKernelOwnerAtTextOffset(Ctx, TextOffset);
  bool ScanWholeObject = Owner.empty() || !FunctionRange;
  SafeSgprUsageSummary *Usage = nullptr;
  if (!ScanWholeObject) {
    using FunctionKey = std::pair<uint64_t, uint64_t>;
    FunctionKey Key{FunctionRange->Begin, FunctionRange->End};
    DenseMap<FunctionKey, SafeSgprUsageSummary>::iterator Cached =
        Ctx.FunctionSgprUsage.find(Key);
    if (Cached == Ctx.FunctionSgprUsage.end()) {
      std::vector<InternalDecodedInst>::const_iterator Begin = std::lower_bound(
          Ctx.Decoded.cbegin(), Ctx.Decoded.cend(), FunctionRange->Begin,
          [](const InternalDecodedInst &DI, uint64_t Offset) {
            return DI.Offset < Offset;
          });
      std::vector<InternalDecodedInst>::const_iterator End =
          std::lower_bound(Begin, Ctx.Decoded.cend(), FunctionRange->End,
                           [](const InternalDecodedInst &DI, uint64_t Offset) {
                             return DI.Offset < Offset;
                           });
      size_t BeginIndex = Begin - Ctx.Decoded.cbegin();
      size_t InstructionCount = End - Begin;
      SafeSgprUsageSummary Summary =
          summarizeSafeSgprUsage(Ctx,
                                 ArrayRef<InternalDecodedInst>(Ctx.Decoded)
                                     .slice(BeginIndex, InstructionCount),
                                 Context);
      Cached = Ctx.FunctionSgprUsage.try_emplace(Key, Summary).first;
    }
    Usage = &Cached->second;
    ScanWholeObject = Usage->HasCall;
  }

  if (ScanWholeObject) {
    if (!Ctx.WholeObjectSgprUsage)
      Ctx.WholeObjectSgprUsage = summarizeSafeSgprUsage(
          Ctx, ArrayRef<InternalDecodedInst>(Ctx.Decoded), Context);
    Usage = &*Ctx.WholeObjectSgprUsage;
  }
  if (!Usage || !Usage->Valid) {
    log() << "hotswap: error: " << Context
          << ": cached SGPR usage analysis failed\n";
    return std::nullopt;
  }

  bool UsesVcc = Usage->UsesVcc;
  unsigned HighWatermark = Usage->HighWatermark;

  constexpr unsigned VccSgprs = 2;
  if (!Owner.empty()) {
    std::optional<unsigned> Declared = Ctx.Elf.getKernelSgprCount(Owner);
    if (!Declared) {
      log() << "hotswap: error: " << Context
            << ": failed to read SGPR count for kernel " << Owner << "\n";
      return std::nullopt;
    }
    if (UsesVcc && *Declared < VccSgprs) {
      log() << "hotswap: error: " << Context << ": VCC-using kernel " << Owner
            << " has invalid SGPR count " << *Declared << "\n";
      return std::nullopt;
    }
    unsigned DeclaredNumbered = *Declared - (UsesVcc ? VccSgprs : 0);
    HighWatermark = std::max(HighWatermark, DeclaredNumbered);
  } else {
    // A device function can be reached from kernels with different declared
    // register footprints. Without a complete call graph, keep the block above
    // every declaration and charge every kernel in the commit step.
    for (const KernelDescriptorInfo &KD : Ctx.Elf.kernelDescriptors()) {
      std::optional<unsigned> Declared =
          Ctx.Elf.getKernelSgprCount(KD.KernelName);
      if (!Declared) {
        log() << "hotswap: error: " << Context
              << ": failed to read SGPR count for kernel " << KD.KernelName
              << "\n";
        return std::nullopt;
      }
      HighWatermark = std::max(HighWatermark, *Declared);
    }
  }

  if (HighWatermark > std::numeric_limits<unsigned>::max() - (Alignment - 1)) {
    log() << "hotswap: error: " << Context
          << ": SGPR alignment calculation overflows unsigned\n";
    return std::nullopt;
  }
  unsigned Base = (HighWatermark + Alignment - 1) & ~(Alignment - 1);
  if (Base > Ctx.Config.MaxSgprs || Count > Ctx.Config.MaxSgprs - Base) {
    if (ReportNoSpace)
      log() << "hotswap: error: " << Context << ": no aligned block of "
            << Count << " safe SGPRs fits below s" << Ctx.Config.MaxSgprs
            << "\n";
    return std::nullopt;
  }
  return SafeSgprScratchBlock{Base, Count};
}

bool commitSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset,
                                const SafeSgprScratchBlock &Block,
                                StringRef Context) {
  ArrayRef<KernelDescriptorInfo> Descriptors = Ctx.Elf.kernelDescriptors();
  if (Descriptors.empty()) {
    log() << "hotswap: error: " << Context
          << ": code object has no kernel descriptors to charge for scratch "
             "SGPRs\n";
    return false;
  }

  std::string Owner = findKernelOwnerAtTextOffset(Ctx, TextOffset);
  bool ChargedOwner = false;

  // llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp::getNumExtraSGPRs returns
  // two non-numbered VCC SGPRs on GFX1250. Always include them in the metadata
  // requirement. This may conservatively overstate a kernel that does not use
  // VCC, but never mistakes VCC for numbered s0-s105 registers.
  constexpr unsigned VccSgprs = 2;
  if (Block.Count > std::numeric_limits<unsigned>::max() - Block.Base ||
      VccSgprs >
          std::numeric_limits<unsigned>::max() - (Block.Base + Block.Count)) {
    log() << "hotswap: error: " << Context
          << ": SGPR descriptor requirement overflows unsigned\n";
    return false;
  }
  unsigned RequiredSgprs = Block.Base + Block.Count + VccSgprs;
  unsigned CoveredRequirement = Ctx.AllKernelSgprRequirement;
  if (!Owner.empty()) {
    auto Committed = Ctx.KernelSgprRequirements.find(Owner);
    if (Committed != Ctx.KernelSgprRequirements.end())
      CoveredRequirement = std::max(CoveredRequirement, Committed->second);
  }
  if (CoveredRequirement >= RequiredSgprs)
    return true;

  // Preflight every selected descriptor before mutating KernelStats. This
  // keeps a malformed later descriptor from leaving a partial commitment that
  // the monotone cache cannot describe.
  SmallVector<std::pair<StringRef, unsigned>, 8> Updates;
  for (const KernelDescriptorInfo &KD : Descriptors) {
    if (!Owner.empty() && KD.KernelName != Owner)
      continue;
    ChargedOwner = true;

    std::optional<unsigned> Current = Ctx.Elf.getKernelSgprCount(KD.KernelName);
    if (!Current) {
      log() << "hotswap: error: " << Context
            << ": failed to read SGPR count for kernel " << KD.KernelName
            << "\n";
      return false;
    }
    if (*Current >= RequiredSgprs)
      continue;
    unsigned Existing = 0;
    auto Stats = Ctx.KernelStats.find(KD.KernelName);
    if (Stats != Ctx.KernelStats.end())
      Existing = Stats->second.ExtraSgprs;
    Updates.push_back(
        {KD.KernelName, std::max(Existing, RequiredSgprs - *Current)});
  }

  if (!ChargedOwner) {
    log() << "hotswap: error: " << Context << ": kernel '" << Owner
          << "' has no descriptor\n";
    return false;
  }

  for (const auto &[KernelName, ExtraSgprs] : Updates)
    Ctx.KernelStats[KernelName].ExtraSgprs =
        std::max(Ctx.KernelStats[KernelName].ExtraSgprs, ExtraSgprs);
  ++Ctx.SgprDescriptorChargePasses;
  if (Owner.empty())
    Ctx.AllKernelSgprRequirement = RequiredSgprs;
  else
    Ctx.KernelSgprRequirements[Owner] = RequiredSgprs;
  return true;
}

bool instructionReadsRegister(const InternalDecodedInst &DI,
                              const LLVMState &LS, MCRegister Register) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned DefCount = std::min(Desc.getNumDefs(), DI.Inst.getNumOperands());
  // A tied use makes its corresponding explicit def a read/modify/write
  // operand. Some MCInst producers materialize a duplicate use operand while
  // others only retain the destination operand, so consult the descriptor
  // instead of relying on the decoded operand list to contain that duplicate.
  for (unsigned Def = 0; Def != DefCount; ++Def) {
    bool HasTiedUse = false;
    for (unsigned Use = Desc.getNumDefs(); Use != Desc.getNumOperands();
         ++Use) {
      if (Desc.getOperandConstraint(Use, MCOI::TIED_TO) ==
          static_cast<int>(Def)) {
        HasTiedUse = true;
        break;
      }
    }
    if (!HasTiedUse)
      continue;
    const MCOperand &Operand = DI.Inst.getOperand(Def);
    if (Operand.isReg() && Operand.getReg() &&
        LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
      return true;
  }
  for (unsigned I = DefCount; I != DI.Inst.getNumOperands(); ++I) {
    const MCOperand &Operand = DI.Inst.getOperand(I);
    if (Operand.isReg() && Operand.getReg() &&
        LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
      return true;
  }
  for (MCPhysReg ImplicitUse : Desc.implicit_uses())
    if (LS.MRI->regsOverlap(MCRegister(ImplicitUse), Register))
      return true;
  return false;
}

static bool instructionWritesRegister(const InternalDecodedInst &DI,
                                      const LLVMState &LS,
                                      MCRegister Register) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned DefCount = std::min(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != DefCount; ++I) {
    const MCOperand &Operand = DI.Inst.getOperand(I);
    if (Operand.isReg() && Operand.getReg() &&
        LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
      return true;
  }
  if (Desc.variadicOpsAreDefs()) {
    unsigned VariadicBegin =
        std::min(Desc.getNumOperands(), DI.Inst.getNumOperands());
    for (unsigned I = VariadicBegin; I != DI.Inst.getNumOperands(); ++I) {
      const MCOperand &Operand = DI.Inst.getOperand(I);
      if (Operand.isReg() && Operand.getReg() &&
          LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
        return true;
    }
  }
  for (MCPhysReg ImplicitDef : Desc.implicit_defs())
    if (LS.MRI->regsOverlap(MCRegister(ImplicitDef), Register))
      return true;
  return false;
}

bool instructionFullyWritesRegister(const InternalDecodedInst &DI,
                                    const LLVMState &LS,
                                    MCRegister Register) {
  auto DefinitionCoversRegister = [&](MCRegister Def) {
    return Def.isValid() && LS.MRI->isSubRegisterEq(Def, Register);
  };

  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned DefCount = std::min(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != DefCount; ++I) {
    const MCOperand &Operand = DI.Inst.getOperand(I);
    if (Operand.isReg() &&
        DefinitionCoversRegister(MCRegister(Operand.getReg())))
      return true;
  }
  if (Desc.variadicOpsAreDefs()) {
    unsigned VariadicBegin =
        std::min(Desc.getNumOperands(), DI.Inst.getNumOperands());
    for (unsigned I = VariadicBegin; I != DI.Inst.getNumOperands(); ++I) {
      const MCOperand &Operand = DI.Inst.getOperand(I);
      if (Operand.isReg() &&
          DefinitionCoversRegister(MCRegister(Operand.getReg())))
        return true;
    }
  }
  for (MCPhysReg ImplicitDef : Desc.implicit_defs())
    if (DefinitionCoversRegister(MCRegister(ImplicitDef)))
      return true;
  return false;
}

bool replacementNeedsIncomingRegister(ArrayRef<uint8_t> Replacement,
                                      const LLVMState &LS,
                                      MCRegister Register) {
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Replacement.data(), Replacement.size(), LS, Decoded))
    return true;

  for (const InternalDecodedInst &DI : Decoded) {
    if (!DI.DecodeSucceeded || !LS.MIA ||
        LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))
      return true;
    if (instructionReadsRegister(DI, LS, Register))
      return true;
    if (instructionFullyWritesRegister(DI, LS, Register))
      return false;
  }
  return false;
}

static bool isRegisterDefinitelyDeadAtContinuation(PatchContext &Ctx,
                                                   uint64_t InstOffset,
                                                   uint32_t InstSize,
                                                   MCRegister Register) {
  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset);
  if (!FunctionRange)
    return false;

  std::optional<uint64_t> Continuation = checkedAddUint64(
      InstOffset, InstSize, "far-return register liveness continuation");
  if (!Continuation)
    return false;
  std::vector<InternalDecodedInst>::const_iterator It =
      std::lower_bound(Ctx.Decoded.cbegin(), Ctx.Decoded.cend(), *Continuation,
                       [](const InternalDecodedInst &DI, uint64_t Offset) {
                         return DI.Offset < Offset;
                       });
  if (It == Ctx.Decoded.cend() || It->Offset != *Continuation)
    return false;

  SmallVector<size_t, 8> Worklist;
  DenseSet<size_t> Visited;
  Worklist.push_back(It - Ctx.Decoded.cbegin());
  while (!Worklist.empty()) {
    size_t Index = Worklist.pop_back_val();
    if (!Visited.insert(Index).second)
      continue;
    const InternalDecodedInst &DI = Ctx.Decoded[Index];
    if (!DI.DecodeSucceeded || !Ctx.LS.MIA ||
        DI.Offset < FunctionRange->Begin || DI.Offset >= FunctionRange->End)
      return false;
    if (instructionReadsRegister(DI, Ctx.LS, Register))
      return false;
    if (instructionFullyWritesRegister(DI, Ctx.LS, Register) ||
        DI.Inst.getOpcode() == Ctx.LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == Ctx.LS.SEndPgmSavedOpcode)
      continue;

    auto AddSuccessor = [&](uint64_t Offset) {
      if (Offset < FunctionRange->Begin || Offset >= FunctionRange->End)
        return false;
      std::vector<InternalDecodedInst>::const_iterator Successor =
          std::lower_bound(
              Ctx.Decoded.cbegin(), Ctx.Decoded.cend(), Offset,
              [](const InternalDecodedInst &Candidate, uint64_t Target) {
                return Candidate.Offset < Target;
              });
      if (Successor == Ctx.Decoded.cend() || Successor->Offset != Offset)
        return false;
      Worklist.push_back(Successor - Ctx.Decoded.cbegin());
      return true;
    };

    if (Ctx.LS.MIA->isCall(DI.Inst) || Ctx.LS.MIA->isIndirectBranch(DI.Inst) ||
        Ctx.LS.MIA->isReturn(DI.Inst))
      return false;
    if (Ctx.LS.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> Target =
          evaluateDirectControlFlowTarget(DI, Ctx.LS);
      if (!Target || !AddSuccessor(*Target))
        return false;
      if (Ctx.LS.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    } else if (Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI) &&
               !Ctx.LS.MIA->isBarrier(DI.Inst)) {
      return false;
    }

    std::optional<uint64_t> Fallthrough = checkedAddUint64(
        DI.Offset, DI.Size, "far-return register liveness fallthrough");
    if (!Fallthrough || !AddSuccessor(*Fallthrough))
      return false;
  }
  return true;
}

std::optional<DenseSet<uint64_t>> computeIncomingRegisterNeeds(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t FunctionBegin, uint64_t FunctionEnd, MCRegister Register) {
  if (!LS.MIA || FunctionBegin >= FunctionEnd)
    return std::nullopt;

  auto Begin = llvm::lower_bound(
      Decoded, FunctionBegin,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  auto End = llvm::lower_bound(
      Decoded, FunctionEnd,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  if (Begin == End)
    return std::nullopt;

  const size_t Count = End - Begin;
  std::vector<SmallVector<size_t, 2>> Predecessors(Count);
  BitVector NeedsIncoming(Count);
  BitVector FullDefinitions(Count);

  auto FindLocalIndex = [&](uint64_t Offset) -> std::optional<size_t> {
    if (Offset < FunctionBegin || Offset >= FunctionEnd)
      return std::nullopt;
    auto It = std::lower_bound(
        Begin, End, Offset,
        [](const InternalDecodedInst &DI, uint64_t Target) {
          return DI.Offset < Target;
        });
    if (It == End || It->Offset != Offset)
      return std::nullopt;
    return It - Begin;
  };

  for (size_t I = 0; I != Count; ++I) {
    const InternalDecodedInst &DI = Begin[I];
    if (!DI.DecodeSucceeded) {
      NeedsIncoming.set(I);
      continue;
    }
    if (instructionReadsRegister(DI, LS, Register)) {
      NeedsIncoming.set(I);
      continue;
    }
    if (instructionFullyWritesRegister(DI, LS, Register)) {
      FullDefinitions.set(I);
      continue;
    }
    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode)
      continue;
    if (LS.MIA->isCall(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
        LS.MIA->isReturn(DI.Inst)) {
      NeedsIncoming.set(I);
      continue;
    }

    auto AddSuccessor = [&](uint64_t Offset) {
      std::optional<size_t> Successor = FindLocalIndex(Offset);
      if (!Successor) {
        NeedsIncoming.set(I);
        return;
      }
      Predecessors[*Successor].push_back(I);
    };

    if (LS.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> Target =
          evaluateDirectControlFlowTarget(DI, LS);
      if (Target)
        AddSuccessor(*Target);
      else
        NeedsIncoming.set(I);
      if (LS.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    } else if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI) &&
               !LS.MIA->isBarrier(DI.Inst)) {
      NeedsIncoming.set(I);
      continue;
    }

    std::optional<uint64_t> Fallthrough = checkedAddUint64(
        DI.Offset, DI.Size, "batched register liveness fallthrough");
    if (Fallthrough)
      AddSuccessor(*Fallthrough);
    else
      NeedsIncoming.set(I);
  }

  SmallVector<size_t, 32> Worklist;
  for (int I = NeedsIncoming.find_first(); I >= 0;
       I = NeedsIncoming.find_next(I))
    Worklist.push_back(static_cast<size_t>(I));
  while (!Worklist.empty()) {
    size_t Successor = Worklist.pop_back_val();
    for (size_t Predecessor : Predecessors[Successor]) {
      if (NeedsIncoming.test(Predecessor) ||
          FullDefinitions.test(Predecessor))
        continue;
      NeedsIncoming.set(Predecessor);
      Worklist.push_back(Predecessor);
    }
  }

  DenseSet<uint64_t> Result;
  for (int I = NeedsIncoming.find_first(); I >= 0;
       I = NeedsIncoming.find_next(I))
    Result.insert(Begin[static_cast<size_t>(I)].Offset);
  return Result;
}

std::optional<SmallVector<MCRegister, 128>>
resolveNumberedSgprRegisters(const MCRegisterInfo &MRI, unsigned MaxSgprs) {
  SmallVector<MCRegister, 128> Registers(MaxSgprs);
  for (unsigned I = 1; I != MRI.getNumRegs(); ++I) {
    MCRegister Register(I);
    std::optional<unsigned> Index = numberedSgprIndex(MRI, Register);
    if (Index && *Index < MaxSgprs)
      Registers[*Index] = Register;
  }
  if (llvm::any_of(Registers,
                   [](MCRegister Register) { return !Register.isValid(); }))
    return std::nullopt;
  return Registers;
}

void getNumberedSgprUsesAndDefs(const InternalDecodedInst &DI,
                                const LLVMState &LS,
                                ArrayRef<MCRegister> NumberedSgprs,
                                BitVector &Uses, BitVector &Defs) {
  assert(Uses.size() == NumberedSgprs.size() &&
         Defs.size() == NumberedSgprs.size());
  for (unsigned I = 0; I != NumberedSgprs.size(); ++I) {
    if (instructionReadsRegister(DI, LS, NumberedSgprs[I]))
      Uses.set(I);
    if (instructionWritesRegister(DI, LS, NumberedSgprs[I]))
      Defs.set(I);
  }
}

/// Return the numbered SGPRs whose incoming values can be observed by the
/// replacement. A malformed or control-flow-bearing replacement conservatively
/// keeps every value that has not already been overwritten.
BitVector
unsafeIncomingNumberedSgprsInReplacement(ArrayRef<uint8_t> Replacement,
                                         const LLVMState &LS,
                                         ArrayRef<MCRegister> NumberedSgprs) {
  const unsigned MaxSgprs = NumberedSgprs.size();
  BitVector Unsafe(MaxSgprs);
  BitVector Incoming(MaxSgprs, true);
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Replacement.data(), Replacement.size(), LS, Decoded)) {
    Unsafe.set();
    return Unsafe;
  }

  for (const InternalDecodedInst &DI : Decoded) {
    if (!DI.DecodeSucceeded || !LS.MIA) {
      Unsafe |= Incoming;
      break;
    }
    BitVector Uses(MaxSgprs);
    BitVector Defs(MaxSgprs);
    getNumberedSgprUsesAndDefs(DI, LS, NumberedSgprs, Uses, Defs);
    Uses &= Incoming;
    Unsafe |= Uses;
    Incoming.reset(Defs);
    if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI)) {
      Unsafe |= Incoming;
      break;
    }
  }
  return Unsafe;
}

/// Analyze all numbered SGPR incoming values in one monotone CFG walk.
/// Unsafe contains a register when some path reads its incoming value before
/// overwriting it, or reaches control flow that cannot be bounded precisely.
std::optional<BitVector>
unsafeIncomingNumberedSgprsInRange(ArrayRef<InternalDecodedInst> Decoded,
                                   const LLVMState &LS, uint64_t FunctionBegin,
                                   uint64_t FunctionEnd, uint64_t Continuation,
                                   ArrayRef<MCRegister> NumberedSgprs) {
  const unsigned MaxSgprs = NumberedSgprs.size();
  if (!LS.MIA)
    return std::nullopt;

  auto FindInstruction = [&](uint64_t Offset) -> std::optional<size_t> {
    if (Offset < FunctionBegin || Offset >= FunctionEnd)
      return std::nullopt;
    auto It =
        std::lower_bound(Decoded.begin(), Decoded.end(), Offset,
                         [](const InternalDecodedInst &DI, uint64_t Target) {
                           return DI.Offset < Target;
                         });
    if (It == Decoded.end() || It->Offset != Offset)
      return std::nullopt;
    return It - Decoded.begin();
  };
  std::optional<size_t> ContinuationIndex = FindInstruction(Continuation);
  if (!ContinuationIndex)
    return std::nullopt;

  DenseMap<size_t, BitVector> IncomingAt;
  IncomingAt.try_emplace(*ContinuationIndex, MaxSgprs, true);
  SmallVector<size_t, 16> Worklist(1, *ContinuationIndex);
  BitVector Queued(Decoded.size());
  Queued.set(*ContinuationIndex);
  BitVector Unsafe(MaxSgprs);

  auto Propagate = [&](uint64_t Offset, const BitVector &Incoming) {
    std::optional<size_t> Successor = FindInstruction(Offset);
    if (!Successor) {
      Unsafe |= Incoming;
      return;
    }
    auto It = IncomingAt.try_emplace(*Successor, MaxSgprs).first;
    BitVector NewValues = Incoming;
    NewValues.reset(It->second);
    if (NewValues.none())
      return;
    It->second |= Incoming;
    if (!Queued.test(*Successor)) {
      Queued.set(*Successor);
      Worklist.push_back(*Successor);
    }
  };

  while (!Worklist.empty()) {
    size_t Index = Worklist.pop_back_val();
    Queued.reset(Index);
    const InternalDecodedInst &DI = Decoded[Index];
    BitVector Incoming = IncomingAt.find(Index)->second;
    if (!DI.DecodeSucceeded || DI.Offset < FunctionBegin ||
        DI.Offset >= FunctionEnd) {
      Unsafe |= Incoming;
      continue;
    }

    BitVector Uses(MaxSgprs);
    BitVector Defs(MaxSgprs);
    getNumberedSgprUsesAndDefs(DI, LS, NumberedSgprs, Uses, Defs);
    Uses &= Incoming;
    Unsafe |= Uses;
    Incoming.reset(Defs);
    if (Incoming.none())
      continue;

    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode)
      continue;
    if (LS.MIA->isCall(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
        LS.MIA->isReturn(DI.Inst)) {
      Unsafe |= Incoming;
      continue;
    }
    if (LS.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
      if (Target)
        Propagate(*Target, Incoming);
      else
        Unsafe |= Incoming;
      if (LS.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    } else if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI) &&
               !LS.MIA->isBarrier(DI.Inst)) {
      Unsafe |= Incoming;
      continue;
    }

    std::optional<uint64_t> Fallthrough = checkedAddUint64(
        DI.Offset, DI.Size, "far-return SGPR liveness fallthrough");
    if (Fallthrough)
      Propagate(*Fallthrough, Incoming);
    else
      Unsafe |= Incoming;
  }
  return Unsafe;
}

struct BatchedSgprContinuationAnalysis {
  uint64_t FunctionBegin = 0;
  uint64_t FunctionEnd = 0;
  size_t BeginIndex = 0;
  size_t InstructionCount = 0;
  unsigned RegisterCount = 0;
  unsigned WordsPerRow = 0;
  std::vector<uint64_t> UnsafeRows;

  std::optional<BitVector>
  query(ArrayRef<InternalDecodedInst> Decoded, uint64_t Continuation) const {
    if (Continuation < FunctionBegin || Continuation >= FunctionEnd)
      return std::nullopt;
    auto Begin = Decoded.begin() + BeginIndex;
    auto End = Begin + InstructionCount;
    auto It = llvm::lower_bound(
        ArrayRef<InternalDecodedInst>(Begin, End), Continuation,
        [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    if (It == ArrayRef<InternalDecodedInst>(Begin, End).end() ||
        It->Offset != Continuation)
      return std::nullopt;
    size_t LocalIndex = It - Begin;
    const uint64_t *Row =
        UnsafeRows.data() + LocalIndex * WordsPerRow;
    BitVector Result(RegisterCount);
    for (unsigned Register = 0; Register != RegisterCount; ++Register)
      if ((Row[Register / 64] >> (Register % 64)) & 1)
        Result.set(Register);
    return Result;
  }
};

static std::optional<BatchedSgprContinuationAnalysis>
computeBatchedSgprContinuationAnalysis(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t FunctionBegin, uint64_t FunctionEnd,
    ArrayRef<MCRegister> NumberedSgprs) {
  if (!LS.MIA || FunctionBegin >= FunctionEnd)
    return std::nullopt;
  auto Begin = llvm::lower_bound(
      Decoded, FunctionBegin,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  auto End = llvm::lower_bound(
      Decoded, FunctionEnd,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  if (Begin == End)
    return std::nullopt;

  BatchedSgprContinuationAnalysis Result;
  Result.FunctionBegin = FunctionBegin;
  Result.FunctionEnd = FunctionEnd;
  Result.BeginIndex = Begin - Decoded.begin();
  Result.InstructionCount = End - Begin;
  Result.RegisterCount = NumberedSgprs.size();
  Result.WordsPerRow = (Result.RegisterCount + 63) / 64;
  if (Result.WordsPerRow == 0)
    return Result;
  if (Result.InstructionCount >
      std::numeric_limits<size_t>::max() / Result.WordsPerRow)
    return std::nullopt;
  size_t WordCount = Result.InstructionCount * Result.WordsPerRow;

  std::vector<uint64_t> UsesRows(WordCount);
  std::vector<uint64_t> DefsRows(WordCount);
  Result.UnsafeRows.assign(WordCount, 0);
  std::vector<SmallVector<size_t, 2>> Successors(Result.InstructionCount);
  std::vector<SmallVector<size_t, 2>> Predecessors(Result.InstructionCount);
  BitVector OpaqueSuccessor(Result.InstructionCount);

  auto FindLocalIndex = [&](uint64_t Offset) -> std::optional<size_t> {
    if (Offset < FunctionBegin || Offset >= FunctionEnd)
      return std::nullopt;
    auto It = llvm::lower_bound(
        ArrayRef<InternalDecodedInst>(Begin, End), Offset,
        [](const InternalDecodedInst &DI, uint64_t Target) {
          return DI.Offset < Target;
        });
    if (It == ArrayRef<InternalDecodedInst>(Begin, End).end() ||
        It->Offset != Offset)
      return std::nullopt;
    return It - Begin;
  };
  auto AddSuccessor = [&](size_t From, uint64_t Offset) {
    std::optional<size_t> To = FindLocalIndex(Offset);
    if (!To) {
      OpaqueSuccessor.set(From);
      return;
    }
    if (!llvm::is_contained(Successors[From], *To)) {
      Successors[From].push_back(*To);
      Predecessors[*To].push_back(From);
    }
  };
  auto StoreBits = [&](uint64_t *Row, const BitVector &Bits) {
    for (int Bit = Bits.find_first(); Bit >= 0; Bit = Bits.find_next(Bit))
      Row[static_cast<unsigned>(Bit) / 64] |=
          uint64_t{1} << (static_cast<unsigned>(Bit) % 64);
  };

  for (size_t I = 0; I != Result.InstructionCount; ++I) {
    const InternalDecodedInst &DI = Begin[I];
    uint64_t *Uses = UsesRows.data() + I * Result.WordsPerRow;
    uint64_t *Defs = DefsRows.data() + I * Result.WordsPerRow;
    if (!DI.DecodeSucceeded || DI.Offset < FunctionBegin ||
        DI.Offset >= FunctionEnd) {
      OpaqueSuccessor.set(I);
      continue;
    }

    BitVector UsesBits(Result.RegisterCount);
    BitVector DefsBits(Result.RegisterCount);
    getNumberedSgprUsesAndDefs(DI, LS, NumberedSgprs, UsesBits, DefsBits);
    StoreBits(Uses, UsesBits);
    StoreBits(Defs, DefsBits);

    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode)
      continue;
    if (LS.MIA->isCall(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
        LS.MIA->isReturn(DI.Inst)) {
      OpaqueSuccessor.set(I);
      continue;
    }
    if (LS.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
      if (Target)
        AddSuccessor(I, *Target);
      else
        OpaqueSuccessor.set(I);
      if (LS.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    } else if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI) &&
               !LS.MIA->isBarrier(DI.Inst)) {
      OpaqueSuccessor.set(I);
      continue;
    }

    std::optional<uint64_t> Fallthrough = checkedAddUint64(
        DI.Offset, DI.Size, "batched SGPR liveness fallthrough");
    if (Fallthrough)
      AddSuccessor(I, *Fallthrough);
    else
      OpaqueSuccessor.set(I);
  }

  SmallVector<size_t, 32> Worklist;
  Worklist.reserve(Result.InstructionCount);
  BitVector Queued(Result.InstructionCount, true);
  for (size_t I = 0; I != Result.InstructionCount; ++I)
    Worklist.push_back(I);
  while (!Worklist.empty()) {
    size_t I = Worklist.pop_back_val();
    Queued.reset(I);
    const uint64_t *Uses = UsesRows.data() + I * Result.WordsPerRow;
    const uint64_t *Defs = DefsRows.data() + I * Result.WordsPerRow;
    uint64_t *Unsafe =
        Result.UnsafeRows.data() + I * Result.WordsPerRow;
    bool Changed = false;
    for (unsigned Word = 0; Word != Result.WordsPerRow; ++Word) {
      uint64_t SuccessorUnsafe =
          OpaqueSuccessor.test(I) ? std::numeric_limits<uint64_t>::max() : 0;
      for (size_t Successor : Successors[I])
        SuccessorUnsafe |=
            Result.UnsafeRows[Successor * Result.WordsPerRow + Word];
      uint64_t NewUnsafe =
          Uses[Word] | (SuccessorUnsafe & ~Defs[Word]);
      if (Word + 1 == Result.WordsPerRow &&
          Result.RegisterCount % 64 != 0)
        NewUnsafe &= (uint64_t{1} << (Result.RegisterCount % 64)) - 1;
      uint64_t Added = NewUnsafe & ~Unsafe[Word];
      if (Added != 0) {
        Unsafe[Word] |= Added;
        Changed = true;
      }
    }
    if (!Changed)
      continue;
    for (size_t Predecessor : Predecessors[I])
      if (!Queued.test(Predecessor)) {
        Queued.set(Predecessor);
        Worklist.push_back(Predecessor);
      }
  }
  return Result;
}

BatchedSgprContinuationTestResult
runBatchedSgprContinuationAnalysisForTest(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t FunctionBegin, uint64_t FunctionEnd,
    ArrayRef<uint64_t> Continuations,
    ArrayRef<MCRegister> NumberedSgprs) {
  BatchedSgprContinuationTestResult Result;
  std::optional<BatchedSgprContinuationAnalysis> Analysis =
      computeBatchedSgprContinuationAnalysis(
          Decoded, LS, FunctionBegin, FunctionEnd, NumberedSgprs);
  Result.Analyses = 1;
  for (uint64_t Continuation : Continuations)
    Result.Queries.push_back(
        Analysis ? Analysis->query(Decoded, Continuation) : std::nullopt);
  return Result;
}

static std::optional<BitVector> unsafeIncomingNumberedSgprsAtContinuation(
    PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
    ArrayRef<MCRegister> NumberedSgprs) {
  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset);
  if (!FunctionRange)
    return std::nullopt;
  std::optional<uint64_t> Continuation = checkedAddUint64(
      InstOffset, InstSize, "far-return SGPR liveness continuation");
  if (!Continuation)
    return std::nullopt;
  return unsafeIncomingNumberedSgprsInRange(
      Ctx.Decoded, Ctx.LS, FunctionRange->Begin, FunctionRange->End,
      *Continuation, NumberedSgprs);
}

static std::optional<unsigned>
selectLocallyDeadSgprPair(unsigned MaxSgprs, const BitVector &Unsafe) {
  if (MaxSgprs < 2 || Unsafe.size() < MaxSgprs)
    return std::nullopt;
  unsigned Base = (MaxSgprs - 2) & ~1u;
  for (;;) {
    if (!Unsafe.test(Base) && !Unsafe.test(Base + 1))
      return Base;
    if (Base == 0)
      break;
    Base -= 2;
  }
  return std::nullopt;
}

static std::optional<unsigned>
findLocallyDeadSgprPair(PatchContext &Ctx, uint64_t InstOffset,
                        uint32_t InstSize, ArrayRef<uint8_t> Replacement) {
  if (Ctx.Config.MaxSgprs < 2)
    return std::nullopt;
  std::optional<SmallVector<MCRegister, 128>> NumberedSgprs =
      resolveNumberedSgprRegisters(*Ctx.LS.MRI, Ctx.Config.MaxSgprs);
  if (!NumberedSgprs)
    return std::nullopt;
  std::optional<BitVector> ContinuationUnsafe =
      unsafeIncomingNumberedSgprsAtContinuation(Ctx, InstOffset, InstSize,
                                                *NumberedSgprs);
  if (!ContinuationUnsafe)
    return std::nullopt;
  BitVector Unsafe = unsafeIncomingNumberedSgprsInReplacement(
      Replacement, Ctx.LS, *NumberedSgprs);
  Unsafe |= *ContinuationUnsafe;
  return selectLocallyDeadSgprPair(Ctx.Config.MaxSgprs, Unsafe);
}

using BatchedSgprContinuationCache =
    DenseMap<std::pair<uint64_t, uint64_t>,
             std::optional<BatchedSgprContinuationAnalysis>>;

static std::optional<unsigned> findLocallyDeadSgprPairWithCache(
    PatchContext &Ctx, const ElfView::FunctionTextRange &FunctionRange,
    uint64_t InstOffset, uint32_t InstSize, ArrayRef<uint8_t> Replacement,
    ArrayRef<MCRegister> NumberedSgprs,
    BatchedSgprContinuationCache &Cache, uint64_t &AnalysisCount) {
  std::optional<uint64_t> Continuation = checkedAddUint64(
      InstOffset, InstSize, "cached far-return SGPR liveness continuation");
  if (!Continuation)
    return std::nullopt;

  std::pair<uint64_t, uint64_t> Key{FunctionRange.Begin, FunctionRange.End};
  auto It = Cache.find(Key);
  if (It == Cache.end()) {
    std::optional<BatchedSgprContinuationAnalysis> Analysis =
        computeBatchedSgprContinuationAnalysis(
            Ctx.Decoded, Ctx.LS, FunctionRange.Begin, FunctionRange.End,
            NumberedSgprs);
    It = Cache.try_emplace(Key, std::move(Analysis)).first;
    ++AnalysisCount;
  }
  if (!It->second)
    return std::nullopt;
  std::optional<BitVector> ContinuationUnsafe =
      It->second->query(Ctx.Decoded, *Continuation);
  if (!ContinuationUnsafe)
    return std::nullopt;

  BitVector Unsafe = unsafeIncomingNumberedSgprsInReplacement(
      Replacement, Ctx.LS, NumberedSgprs);
  Unsafe |= *ContinuationUnsafe;
  return selectLocallyDeadSgprPair(Ctx.Config.MaxSgprs, Unsafe);
}

struct FarReturnScratch {
  bool Available = false;
  unsigned SgprBase = 0;
  bool UseVcc = false;
  bool PreserveVcc = false;
};

static FarReturnScratch reserveSafeFarReturn(PatchContext &Ctx,
                                             uint64_t InstOffset,
                                             uint32_t InstSize,
                                             ArrayRef<uint8_t> Replacement) {
  std::optional<SafeSgprScratchBlock> Scratch = findSafeSgprScratchBlock(
      Ctx, InstOffset, /*Count=*/2, /*Alignment=*/2, "safe far return",
      /*ReportNoSpace=*/false);
  if (Scratch) {
    if (!commitSafeSgprScratchBlock(Ctx, InstOffset, *Scratch,
                                    "safe far return"))
      return {};
    return FarReturnScratch{/*Available=*/true, Scratch->Base,
                            /*UseVcc=*/false, /*PreserveVcc=*/false};
  }

  if (Ctx.LS.VCCRegister.isValid() &&
      !replacementNeedsIncomingRegister(Replacement, Ctx.LS,
                                        Ctx.LS.VCCRegister) &&
      isRegisterDefinitelyDeadAtContinuation(Ctx, InstOffset, InstSize,
                                             Ctx.LS.VCCRegister)) {
    log() << "hotswap: safe far return: reusing dead VCC at 0x"
          << utohexstr(InstOffset) << "\n";
    return FarReturnScratch{/*Available=*/true, /*SgprBase=*/0,
                            /*UseVcc=*/true, /*PreserveVcc=*/false};
  }

  if (std::optional<unsigned> LocalPair =
          findLocallyDeadSgprPair(Ctx, InstOffset, InstSize, Replacement)) {
    SafeSgprScratchBlock Scratch{*LocalPair, 2};
    if (!commitSafeSgprScratchBlock(Ctx, InstOffset, Scratch,
                                    "locally dead far-return SGPR pair"))
      return {};
    log() << "hotswap: safe far return: reusing locally dead s[" << *LocalPair
          << ":" << *LocalPair + 1 << "] at 0x" << utohexstr(InstOffset)
          << "\n";
    return FarReturnScratch{/*Available=*/true, *LocalPair,
                            /*UseVcc=*/false, /*PreserveVcc=*/false};
  }

  std::string Owner = findKernelOwnerAtTextOffset(Ctx, InstOffset);
  std::optional<unsigned> WavefrontSize =
      Owner.empty() ? std::nullopt : Ctx.Elf.getKernelWavefrontSize(Owner);
  if (WavefrontSize == 32 && InstSize >= 2 * MinInstSize) {
    std::optional<SafeSgprScratchBlock> Save = findSafeSgprScratchBlock(
        Ctx, InstOffset, /*Count=*/1, /*Alignment=*/1,
        "VCC-preserving far return", /*ReportNoSpace=*/false);
    if (Save) {
      log() << "hotswap: safe far return: deferring live wave32 VCC_LO "
               "preservation in s"
            << Save->Base << " at 0x" << utohexstr(InstOffset) << "\n";
      return FarReturnScratch{/*Available=*/true, Save->Base,
                              /*UseVcc=*/true, /*PreserveVcc=*/true};
    }
  }

  log() << "hotswap: safe far return: no register pair at 0x"
        << utohexstr(InstOffset)
        << "; deferring to the s_branch island planner\n";
  return {};
}

bool isSBranchReachable(uint64_t From, uint64_t To) {
  std::optional<uint64_t> PcBase =
      checkedAddUint64(From, MinInstSize, "short branch PC base");
  if (!PcBase)
    return false;
  uint64_t Delta = To >= *PcBase ? To - *PcBase : *PcBase - To;
  if (Delta % MinInstSize != 0)
    return false;
  uint64_t MaxDelta =
      To >= *PcBase ? static_cast<uint64_t>(BranchOffsetMax) * MinInstSize
                    : static_cast<uint64_t>(-BranchOffsetMin) * MinInstSize;
  return Delta <= MaxDelta;
}

/// Queue a deferred trampoline for [\p InstOffset, +\p InstSize) with
/// \p Replacement as its body; fixupTrampolineBranches fills in the edges once
/// the pool layout is known. A site beyond s_branch reach of the appended pool
/// uses either an SCC-neutral get-PC/add/set-PC sequence or a chain of
/// registerless s_branch islands on the backward edge.
/// Adjacent far sites are coalesced after patching to reduce gateway pressure.
/// Every far source edge uses a short branch to nearby safe padding; that
/// gateway either continues through s_branch islands or uses the gfx12
/// SGPR-backed set-PC sequence. No source or return edge executes gfx1250's
/// broken s_add_pc_i64 instruction.
[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    ArrayRef<uint8_t> Replacement) {
  // This trampoline lands at the appended pool base and after every trampoline
  // already queued -- later ones are appended behind it and cannot shift it,
  // and fixupTrampolineBranches walks the same list in the same order -- so its
  // final pool offset (relative to .text) is known exactly now.
  std::optional<uint64_t> PoolStart = checkedAddUint64(
      Ctx.PoolBaseOffset, Ctx.QueuedTrampolineBytes, "trampoline pool layout");
  if (!PoolStart)
    return false;

  // An s_branch encodes To - From as a signed simm16 dword field, in range iff
  // (To - From - MinInstSize) / MinInstSize fits [BranchOffsetMin,
  // BranchOffsetMax] (see LLVMState::encodeSBranch). Test both edges with the
  // short branch-back slot; the branch-back (pool tail -> site) is the farther
  // of the two. Go long only when a short branch cannot reach.
  std::optional<uint64_t> ShortBackFrom = checkedAddUint64(
      *PoolStart, Replacement.size(), "short trampoline return slot");
  std::optional<uint64_t> ReturnTo =
      checkedAddUint64(InstOffset, InstSize, "trampoline return target");
  if (!ShortBackFrom || !ReturnTo)
    return false;
  const bool Far = !(isSBranchReachable(InstOffset, *PoolStart) &&
                     isSBranchReachable(*ShortBackFrom, *ReturnTo));

  FarReturnScratch Scratch;
  if (Far)
    Scratch = reserveSafeFarReturn(Ctx, InstOffset, InstSize, Replacement);
  uint64_t ReturnReserve = MinInstSize;
  uint64_t BodyPrefix = 0;
  if (Far && Scratch.Available) {
    ReturnReserve = Scratch.PreserveVcc ? VccPreservingReturnReserveBytes
                                        : SetPcReturnReserveBytes;
    BodyPrefix = Scratch.PreserveVcc ? VccRestoreSequenceBytes : 0;
  }
  std::optional<uint64_t> TrampolineSize = checkedAddUint64(
      Replacement.size(), BodyPrefix, "queued trampoline body size");
  if (TrampolineSize)
    TrampolineSize = checkedAddUint64(*TrampolineSize, ReturnReserve,
                                      "queued trampoline size");
  if (!TrampolineSize)
    return false;
  std::optional<uint64_t> QueuedBytes =
      checkedAddUint64(Ctx.QueuedTrampolineBytes, *TrampolineSize,
                       "queued trampoline byte count");
  if (!QueuedBytes)
    return false;

  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
  if (Scratch.PreserveVcc) {
    SmallVector<std::string, 2> RestoreLines;
    RestoreLines.push_back("s_mov_b32 vcc_lo, s" +
                           std::to_string(Scratch.SgprBase));
    RestoreLines.push_back("s_delay_alu instid0(SALU_CYCLE_1)");
    SmallVector<uint8_t> Restore =
        assembleInstructions(joinAsmLines(RestoreLines), Ctx.LS);
    if (Restore.size() != VccRestoreSequenceBytes)
      return false;
    T.Bytes.append(Restore.begin(), Restore.end());
  }
  T.Bytes.insert(T.Bytes.end(), Replacement.begin(), Replacement.end());
  if (std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset)) {
    T.HasFunctionRange = true;
    T.FunctionStart = Range->Begin;
    T.FunctionEnd = Range->End;
  }

  if (Far) {
    // Every decline of a valid far site increments jump:declined_far (a
    // count-only row) so the metric reflects all placement failures, including
    // resource pressure, not just the size guard.
    auto declineFar = [&](const Twine &Reason) {
      Ctx.Profile.count(HotswapMetric::JumpDeclined);
      log() << "hotswap: far trampoline site 0x" << utohexstr(InstOffset)
            << " declined: " << Reason << "\n";
      return false;
    };
    if (InstSize < MinInstSize)
      return declineFar(Twine(InstSize) + " B, smaller than " +
                        Twine(MinInstSize) + " B forward branch");
    T.Bytes.insert(T.Bytes.end(), ReturnReserve, uint8_t{0});
    T.Long = true;
    T.UsesSetPCBack = Scratch.Available;
    T.LongBranchSgprBase = Scratch.SgprBase;
    T.LongBranchUsesVcc = Scratch.UseVcc;
    T.LongBranchPreservesVcc = Scratch.PreserveVcc;
    Ctx.Profile.count(HotswapMetric::JumpLong);
    Ctx.OutTrampolines.emplace_back(std::move(T));
    Ctx.QueuedTrampolineBytes = *QueuedBytes;
    return true;
  }
  {
    Ctx.Profile.count(HotswapMetric::JumpShort);
    // Reserve the short branch-back slot; fixupTrampolineBranches fills it in.
    T.Bytes.insert(T.Bytes.end(), MinInstSize, uint8_t{0});
  }
  Ctx.OutTrampolines.emplace_back(std::move(T));
  Ctx.QueuedTrampolineBytes = *QueuedBytes;
  return true;
}

std::optional<uint64_t>
evaluateDirectControlFlowTarget(const InternalDecodedInst &DI,
                                const LLVMState &LS) {
  uint64_t Target = 0;
  if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target))
    return Target;

  // TODO(https://github.com/ROCm/llvm-project/issues/3351): Remove this
  // fallback when AMDGPUMCInstrAnalysis::evaluateBranch locates the descriptor
  // operand marked MCOI::OPERAND_PCREL. Its current operand-zero restriction
  // is in llvm/lib/Target/AMDGPU/MCTargetDesc/AMDGPUMCTargetDesc.cpp.
  // GFX1250 s_call_i64 instead has its destination SGPR pair in slot zero and
  // its simm16 dword displacement in slot one; the operand layout and width
  // are pinned by llvm/test/MC/AMDGPU/gfx1250_asm_sopk.s.
  if (DI.Inst.getOpcode() != LS.SCallI64Opcode ||
      DI.Inst.getNumOperands() == 0 ||
      !DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm())
    return std::nullopt;

  uint64_t Encoded =
      static_cast<uint64_t>(
          DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm()) &
      0xFFFFu;
  int64_t DwordDelta = SignExtend64<16>(Encoded);
  std::optional<uint64_t> PcBase = checkedAddUint64(
      DI.Offset, DI.Size, "direct control-flow target PC base");
  if (!PcBase)
    return std::nullopt;
  if (DwordDelta >= 0)
    return checkedAddUint64(*PcBase,
                            static_cast<uint64_t>(DwordDelta) * MinInstSize,
                            "direct control-flow target");
  return checkedSubUint64(*PcBase,
                          static_cast<uint64_t>(-DwordDelta) * MinInstSize,
                          "direct control-flow target");
}

static bool definesOverlappingRegister(const InternalDecodedInst &DI,
                                       const LLVMState &LS,
                                       MCRegister Register) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned DefCount = std::min(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != DefCount; ++I) {
    const MCOperand &Operand = DI.Inst.getOperand(I);
    if (Operand.isReg() && Operand.getReg() &&
        LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
      return true;
  }
  if (Desc.variadicOpsAreDefs()) {
    unsigned VariadicBegin =
        std::min(Desc.getNumOperands(), DI.Inst.getNumOperands());
    for (unsigned I = VariadicBegin; I != DI.Inst.getNumOperands(); ++I) {
      const MCOperand &Operand = DI.Inst.getOperand(I);
      if (Operand.isReg() && Operand.getReg() &&
          LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
        return true;
    }
  }
  for (MCPhysReg ImplicitDef : Desc.implicit_defs())
    if (LS.MRI->regsOverlap(MCRegister(ImplicitDef), Register))
      return true;
  return false;
}

static bool isControlFlowBoundary(const InternalDecodedInst &DI,
                                  const LLVMState &LS) {
  return DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
         DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
         LS.MIA->isBranch(DI.Inst) || LS.MIA->isCall(DI.Inst) ||
         LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
         LS.MIA->isBarrier(DI.Inst);
}

struct DeclaredTextEntryInfo {
  SmallVector<uint64_t, 16> Entries;
  SmallVector<uint64_t, 16> ExternalEntries;
  SmallVector<uint64_t, 16> NonCallEntries;
};

static std::optional<DeclaredTextEntryInfo>
collectDeclaredTextEntries(const ElfView &Elf) {
  std::optional<uint64_t> TextEnd =
      checkedAddUint64(Elf.textAddr(), Elf.textSize(), "declared text end");
  if (!TextEnd)
    return std::nullopt;

  DeclaredTextEntryInfo Info;
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges())
    if (Range.Begin >= Elf.textAddr() && Range.Begin < *TextEnd) {
      uint64_t Entry = Range.Begin - Elf.textAddr();
      Info.Entries.push_back(Entry);
      if (Range.Symbol && Range.Symbol->getBinding() != ELF::STB_LOCAL)
        Info.ExternalEntries.push_back(Entry);
    }

  for (const KernelDescriptorInfo &Descriptor : Elf.kernelDescriptors()) {
    std::optional<uint64_t> EntryAddress;
    if (Descriptor.EntryOffset >= 0) {
      EntryAddress = checkedAddUint64(
          Descriptor.VAddr, static_cast<uint64_t>(Descriptor.EntryOffset),
          "kernel descriptor entry address");
    } else {
      uint64_t Magnitude =
          Descriptor.EntryOffset == std::numeric_limits<int64_t>::min()
              ? uint64_t{1} << 63
              : static_cast<uint64_t>(-Descriptor.EntryOffset);
      EntryAddress = checkedSubUint64(Descriptor.VAddr, Magnitude,
                                      "kernel descriptor entry address");
    }
    if (!EntryAddress)
      return std::nullopt;
    if (*EntryAddress >= Elf.textAddr() && *EntryAddress < *TextEnd) {
      uint64_t Entry = *EntryAddress - Elf.textAddr();
      Info.Entries.push_back(Entry);
      Info.ExternalEntries.push_back(Entry);
      Info.NonCallEntries.push_back(Entry);
    }
  }
  return Info;
}

struct PcMaterializedCallInfo {
  uint64_t Target = 0;
  uint64_t SequenceStart = 0;
  uint64_t SequenceEnd = 0;
  MCRegister ReturnRegister;
};

static std::optional<uint64_t>
evaluateAbsoluteUint64Operand(const MCOperand &Operand) {
  if (Operand.isImm())
    return static_cast<uint64_t>(Operand.getImm());
  if (!Operand.isExpr())
    return std::nullopt;
  int64_t Value = 0;
  if (!Operand.getExpr()->evaluateAsAbsolute(Value))
    return std::nullopt;
  return static_cast<uint64_t>(Value);
}

/// Resolve the compiler-emitted PC materialization used by the production
/// reproducer:
///
///   s_get_pc_i64 Target
///   ...                         // no Target definition or control flow
///   s_add_nc_u64 Target, Target, Immediate
///   ...                         // no Target definition or control flow
///   s_swap_pc_i64 Return, Target
///
/// The opcode and operand layout are defined by SOPInstructions.td and pinned
/// by llvm/test/MC/AMDGPU/gfx1250_asm_salu_lit64.s. Stop at the first
/// overlapping definition or control-flow boundary, so any variation remains
/// unresolved and follows the existing fail-closed policy.
// Shared get-PC/add resolver behind matchPcMaterializedCall (s_swap_pc_i64
// calls) and the WMMA split pass's s_set_pc_i64 jump handling. The backward
// scan and fail-closed policy are identical for both transfer kinds; only the
// terminating opcode and the return-register semantics differ, which the
// callers handle.
std::optional<MaterializedPcSequence>
resolveMaterializedPcTarget(ArrayRef<InternalDecodedInst> Decoded,
                            size_t TransferIndex, MCRegister TargetRegister,
                            const LLVMState &LS, uint64_t TextAddr) {
  std::optional<size_t> AddIndex;
  uint64_t AddImmediate = 0;
  for (size_t I = TransferIndex; I != 0;) {
    --I;
    const InternalDecodedInst &Candidate = Decoded[I];
    if (!Candidate.DecodeSucceeded || isControlFlowBoundary(Candidate, LS))
      return std::nullopt;
    if (!definesOverlappingRegister(Candidate, LS, TargetRegister))
      continue;
    if (Candidate.Inst.getOpcode() != LS.SAddNcU64Opcode ||
        Candidate.Inst.getNumOperands() != 3 ||
        !Candidate.Inst.getOperand(0).isReg() ||
        Candidate.Inst.getOperand(0).getReg() != TargetRegister ||
        !Candidate.Inst.getOperand(1).isReg() ||
        Candidate.Inst.getOperand(1).getReg() != TargetRegister)
      return std::nullopt;
    AddIndex = I;
    std::optional<uint64_t> Immediate =
        evaluateAbsoluteUint64Operand(Candidate.Inst.getOperand(2));
    if (!Immediate)
      return std::nullopt;
    AddImmediate = *Immediate;
    break;
  }
  if (!AddIndex)
    return std::nullopt;

  for (size_t I = *AddIndex; I != 0;) {
    --I;
    const InternalDecodedInst &Candidate = Decoded[I];
    if (!Candidate.DecodeSucceeded || isControlFlowBoundary(Candidate, LS))
      return std::nullopt;
    if (!definesOverlappingRegister(Candidate, LS, TargetRegister))
      continue;
    if (Candidate.Inst.getOpcode() != LS.SGetPcI64Opcode ||
        Candidate.Inst.getNumOperands() != 1 ||
        !Candidate.Inst.getOperand(0).isReg() ||
        Candidate.Inst.getOperand(0).getReg() != TargetRegister)
      return std::nullopt;

    std::optional<uint64_t> GetPcAddress = checkedAddUint64(
        TextAddr, Candidate.Offset, "PC-materialized transfer instruction");
    if (!GetPcAddress)
      return std::nullopt;
    std::optional<uint64_t> PcValue = checkedAddUint64(
        *GetPcAddress, Candidate.Size, "PC-materialized transfer PC value");
    if (!PcValue)
      return std::nullopt;
    // s_add_nc_u64 uses modulo-2^64 arithmetic. Casting the signed MC
    // immediate to uint64_t and adding it reproduces both positive and
    // negative literals, including INT64_MIN, without signed overflow.
    return MaterializedPcSequence{*PcValue + AddImmediate, Candidate.Offset};
  }
  return std::nullopt;
}

static std::optional<PcMaterializedCallInfo>
matchPcMaterializedCall(ArrayRef<InternalDecodedInst> Decoded, size_t CallIndex,
                        const LLVMState &LS, uint64_t TextAddr) {
  const InternalDecodedInst &Call = Decoded[CallIndex];
  if (!Call.DecodeSucceeded || Call.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
      Call.Inst.getNumOperands() < 2 || !Call.Inst.getOperand(0).isReg() ||
      !Call.Inst.getOperand(0).getReg())
    return std::nullopt;
  const MCOperand &TargetOperand =
      Call.Inst.getOperand(Call.Inst.getNumOperands() - 1);
  if (!TargetOperand.isReg() || !TargetOperand.getReg())
    return std::nullopt;
  std::optional<MaterializedPcSequence> Sequence = resolveMaterializedPcTarget(
      Decoded, CallIndex, MCRegister(TargetOperand.getReg()), LS, TextAddr);
  if (!Sequence)
    return std::nullopt;
  return PcMaterializedCallInfo{Sequence->Target, Sequence->SequenceStart,
                                Call.Offset,
                                MCRegister(Call.Inst.getOperand(0).getReg())};
}

using ReachingCallTargets = SmallVector<uint64_t, 8>;

struct ReachingPcState {
  bool Reached = false;
  bool HasUnknown = false;
  ReachingCallTargets Targets;
  SmallVector<size_t, 4> ActiveMaterializations;
};

static bool mergeReachingPcState(ReachingPcState &Into,
                                 const ReachingPcState &From) {
  if (!From.Reached)
    return false;
  ReachingPcState Before = Into;
  Into.Reached = true;
  Into.HasUnknown |= From.HasUnknown;
  for (uint64_t Target : From.Targets)
    if (!llvm::is_contained(Into.Targets, Target))
      Into.Targets.push_back(Target);
  for (size_t Completion : From.ActiveMaterializations)
    if (!llvm::is_contained(Into.ActiveMaterializations, Completion))
      Into.ActiveMaterializations.push_back(Completion);
  llvm::sort(Into.Targets);
  llvm::sort(Into.ActiveMaterializations);
  return Before.Reached != Into.Reached ||
         Before.HasUnknown != Into.HasUnknown ||
         Before.Targets != Into.Targets ||
         Before.ActiveMaterializations != Into.ActiveMaterializations;
}

static bool isExactRegisterOperand(const MCInst &Inst, unsigned Index,
                                   MCRegister Reg) {
  return Index < Inst.getNumOperands() && Inst.getOperand(Index).isReg() &&
         Inst.getOperand(Index).getReg() == Reg;
}

static std::optional<uint32_t>
evaluateAbsoluteUint32Operand(const MCOperand &Operand) {
  if (Operand.isImm())
    return static_cast<uint32_t>(Operand.getImm());
  if (!Operand.isExpr())
    return std::nullopt;
  int64_t Value = 0;
  if (!Operand.getExpr()->evaluateAsAbsolute(Value))
    return std::nullopt;
  return static_cast<uint32_t>(Value);
}

/// Recognize the two compiler-emitted ways a reusable register-call target is
/// materialized. The first is the canonical get-PC/add-nc pair. Tensile also
/// computes a 32-bit displacement in a temporary and propagates carry into the
/// high half:
///
///   s_get_pc_i64 Pair
///   s_add_co_i32 Tmp, Imm0, Imm1
///   s_add_co_u32 Pair.lo, Pair.lo, Tmp
///   s_add_co_ci_u32 Pair.hi, Pair.hi, 0
///
/// Return the completion instruction and absolute target. Intermediate
/// definitions deliberately remain "unknown" in the reaching-value solver.
static std::optional<std::pair<size_t, uint64_t>>
matchReusablePcMaterialization(ArrayRef<InternalDecodedInst> Decoded,
                               size_t GetPcIndex, size_t FunctionEndIndex,
                               MCRegister Pair, const LLVMState &LS,
                               uint64_t TextAddr) {
  const InternalDecodedInst &GetPc = Decoded[GetPcIndex];
  if (!GetPc.DecodeSucceeded || GetPc.Inst.getOpcode() != LS.SGetPcI64Opcode ||
      !isExactRegisterOperand(GetPc.Inst, 0, Pair))
    return std::nullopt;
  std::optional<uint64_t> Pc =
      checkedAddUint64(TextAddr, GetPc.Offset, "reusable get-PC address");
  if (!Pc)
    return std::nullopt;
  Pc = checkedAddUint64(*Pc, GetPc.Size, "reusable get-PC value");
  if (!Pc)
    return std::nullopt;

  for (size_t I = GetPcIndex + 1; I < FunctionEndIndex; ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (!DI.DecodeSucceeded || isControlFlowBoundary(DI, LS))
      break;
    if (!definesOverlappingRegister(DI, LS, Pair))
      continue;
    if (DI.Inst.getOpcode() != LS.SAddNcU64Opcode ||
        DI.Inst.getNumOperands() != 3 ||
        !isExactRegisterOperand(DI.Inst, 0, Pair) ||
        !isExactRegisterOperand(DI.Inst, 1, Pair))
      break;
    std::optional<uint64_t> Delta =
        evaluateAbsoluteUint64Operand(DI.Inst.getOperand(2));
    if (!Delta)
      break;
    return std::make_pair(I, *Pc + *Delta);
  }

  if (GetPcIndex + 3 >= FunctionEndIndex)
    return std::nullopt;
  const InternalDecodedInst &MakeDelta = Decoded[GetPcIndex + 1];
  const InternalDecodedInst &AddLow = Decoded[GetPcIndex + 2];
  const InternalDecodedInst &AddHigh = Decoded[GetPcIndex + 3];
  if (MakeDelta.Mnemonic != "s_add_co_i32" ||
      AddLow.Mnemonic != "s_add_co_u32" ||
      AddHigh.Mnemonic != "s_add_co_ci_u32" ||
      MakeDelta.Inst.getNumOperands() != 3 ||
      !MakeDelta.Inst.getOperand(0).isReg() ||
      !MakeDelta.Inst.getOperand(0).getReg() ||
      !MakeDelta.Inst.getOperand(2).isImm())
    return std::nullopt;
  MCRegister DeltaReg(MakeDelta.Inst.getOperand(0).getReg());
  if (LS.MRI->regsOverlap(DeltaReg, Pair))
    return std::nullopt;
  if (AddLow.Inst.getNumOperands() != 3 || !AddLow.Inst.getOperand(0).isReg() ||
      !AddLow.Inst.getOperand(1).isReg() ||
      !AddLow.Inst.getOperand(0).getReg() ||
      AddLow.Inst.getOperand(0).getReg() !=
          AddLow.Inst.getOperand(1).getReg() ||
      !isExactRegisterOperand(AddLow.Inst, 2, DeltaReg) ||
      AddHigh.Inst.getNumOperands() != 3 ||
      !AddHigh.Inst.getOperand(0).isReg() ||
      !AddHigh.Inst.getOperand(1).isReg() ||
      !AddHigh.Inst.getOperand(0).getReg() ||
      AddHigh.Inst.getOperand(0).getReg() !=
          AddHigh.Inst.getOperand(1).getReg() ||
      !AddHigh.Inst.getOperand(2).isImm() ||
      AddHigh.Inst.getOperand(2).getImm() != 0)
    return std::nullopt;
  MCRegister Low(AddLow.Inst.getOperand(0).getReg());
  MCRegister High(AddHigh.Inst.getOperand(0).getReg());
  std::optional<unsigned> LowIndex = numberedSgprIndex(*LS.MRI, Low);
  std::optional<unsigned> HighIndex = numberedSgprIndex(*LS.MRI, High);
  if (!LowIndex || !HighIndex || *HighIndex != *LowIndex + 1 ||
      !LS.MRI->regsOverlap(Low, Pair) || !LS.MRI->regsOverlap(High, Pair))
    return std::nullopt;

  std::optional<uint32_t> FirstAddend =
      evaluateAbsoluteUint32Operand(MakeDelta.Inst.getOperand(1));
  if (!FirstAddend)
    return std::nullopt;
  uint32_t Delta = *FirstAddend +
                   static_cast<uint32_t>(MakeDelta.Inst.getOperand(2).getImm());
  return std::make_pair(GetPcIndex + 3, *Pc + Delta);
}

struct ReachingCallGroup {
  uint64_t Begin = 0;
  uint64_t End = 0;
  MCRegister TargetRegister;
  SmallVector<size_t, 8> Calls;
};

struct ReachingPcUse {
  size_t InstIndex = 0;
  MCRegister Register;
};

struct FiniteSetPcTransfer {
  size_t InstIndex = 0;
  size_t SequenceBeginIndex = 0;
  size_t SequenceEndIndex = 0;
  uint64_t Target = 0;
  std::optional<size_t> LocalTargetIndex;
  uint64_t FunctionBegin = 0;
  uint64_t FunctionEnd = 0;
};

static const ElfView::FunctionTextRange *
findInnermostFunctionRange(uint64_t Address,
                           ArrayRef<ElfView::FunctionTextRange> Ranges) {
  const ElfView::FunctionTextRange *Best = nullptr;
  for (const ElfView::FunctionTextRange &Range : Ranges)
    if (Range.Begin <= Address && Address < Range.End &&
        (!Best || Range.Begin > Best->Begin ||
         (Range.Begin == Best->Begin && Range.End < Best->End)))
      Best = &Range;
  return Best;
}

/// Collect exact materialized s_set_pc_i64 transfers. These are candidates,
/// not proofs: a later closed-world audit must still rule out an alternate
/// entry into the materialization before the edge can authorize mutation.
static SmallVector<FiniteSetPcTransfer, 8> collectFiniteSetPcCandidates(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextEnd,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges) {
  SmallVector<FiniteSetPcTransfer, 8> Candidates;
  DenseMap<uint64_t, size_t> OffsetToIndex;
  for (size_t I = 0; I != Decoded.size(); ++I)
    OffsetToIndex.try_emplace(Decoded[I].Offset, I);
  std::optional<SmallVector<MCRegister, 128>> NumberedSgprs =
      resolveNumberedSgprRegisters(*LS.MRI, Gfx1250MaxSgprs);

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &GetPc = Decoded[I];
    if (!GetPc.DecodeSucceeded ||
        GetPc.Inst.getOpcode() != LS.SGetPcI64Opcode ||
        GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
        !GetPc.Inst.getOperand(0).getReg())
      continue;
    MCRegister Pair(GetPc.Inst.getOperand(0).getReg());
    std::optional<uint64_t> GetPcAddress = checkedAddUint64(
        TextAddr, GetPc.Offset, "finite set-PC get-PC address");
    if (!GetPcAddress)
      continue;
    const ElfView::FunctionTextRange *Range =
        findInnermostFunctionRange(*GetPcAddress, FunctionRanges);
    if (!Range || Range->Begin < TextAddr || Range->End > TextEnd)
      continue;
    ArrayRef<InternalDecodedInst>::const_iterator FunctionEnd =
        llvm::lower_bound(Decoded, Range->End - TextAddr,
                          [](const InternalDecodedInst &DI, uint64_t Offset) {
                            return DI.Offset < Offset;
                          });
    size_t FunctionEndIndex =
        static_cast<size_t>(FunctionEnd - Decoded.begin());
    auto appendCandidate = [&](size_t SetPcIndex, size_t SequenceEndIndex,
                               uint64_t Target) {
      for (size_t J = I; J <= SequenceEndIndex; ++J) {
        if (!Decoded[J].DecodeSucceeded)
          return;
        if (J == SequenceEndIndex)
          break;
        std::optional<uint64_t> End =
            checkedAddUint64(Decoded[J].Offset, Decoded[J].Size,
                             "finite set-PC materialization instruction end");
        if (!End || *End != Decoded[J + 1].Offset)
          return;
      }
      std::optional<size_t> LocalTargetIndex;
      if (Target >= TextAddr && Target < TextEnd) {
        DenseMap<uint64_t, size_t>::const_iterator LocalTarget =
            OffsetToIndex.find(Target - TextAddr);
        // A local transfer to the middle of an instruction is not finite for
        // purposes of safe rewriting.
        if (LocalTarget == OffsetToIndex.end())
          return;
        LocalTargetIndex = LocalTarget->second;
      }
      Candidates.push_back({SetPcIndex, I, SequenceEndIndex, Target,
                            LocalTargetIndex, Range->Begin - TextAddr,
                            Range->End - TextAddr});
    };

    std::optional<std::pair<size_t, uint64_t>> Match =
        matchReusablePcMaterialization(Decoded, I, FunctionEndIndex, Pair, LS,
                                       TextAddr);
    if (Match && Match->first + 1 < FunctionEndIndex) {
      size_t SetPcIndex = Match->first + 1;
      const InternalDecodedInst &SetPc = Decoded[SetPcIndex];
      if (SetPc.DecodeSucceeded &&
          SetPc.Inst.getOpcode() == LS.SSetPcI64Opcode &&
          SetPc.Inst.getNumOperands() == 1 &&
          isExactRegisterOperand(SetPc.Inst, 0, Pair))
        appendCandidate(SetPcIndex, SetPcIndex, Match->second);
    }

    // gfx1250 also materializes a local 64-bit PC delta directly into the
    // low/high halves of the get-PC pair:
    //
    //   s_get_pc_i64 Pair
    //   s_add_co_u32 Pair.lo, Pair.lo, Delta.lo
    //   s_add_co_ci_u32 Pair.hi, Pair.hi, Delta.hi
    //   s_set_pc_i64 Pair
    //
    // The two immediate halves are linker constants. Treat the four-op shape
    // as one candidate; appendCandidate and the later closed-world audit still
    // reject gaps, instruction-interior targets, and alternate entries.
    if (I + 3 < FunctionEndIndex) {
      const InternalDecodedInst &AddLow = Decoded[I + 1];
      const InternalDecodedInst &AddHigh = Decoded[I + 2];
      const InternalDecodedInst &SetPc = Decoded[I + 3];
      if (AddLow.Mnemonic == "s_add_co_u32" &&
          AddHigh.Mnemonic == "s_add_co_ci_u32" &&
          AddLow.Inst.getNumOperands() == 3 &&
          AddHigh.Inst.getNumOperands() == 3 &&
          SetPc.Inst.getOpcode() == LS.SSetPcI64Opcode &&
          SetPc.Inst.getNumOperands() == 1 &&
          isExactRegisterOperand(SetPc.Inst, 0, Pair)) {
        std::optional<unsigned> PairIndex =
            numberedSgprPairLowIndex(*LS.MRI, Pair);
        std::optional<uint32_t> LowDelta =
            evaluateAbsoluteUint32Operand(AddLow.Inst.getOperand(2));
        std::optional<uint32_t> HighDelta =
            evaluateAbsoluteUint32Operand(AddHigh.Inst.getOperand(2));
        if (PairIndex && LowDelta && HighDelta && NumberedSgprs &&
            *PairIndex + 1 < NumberedSgprs->size() &&
            isExactRegisterOperand(AddLow.Inst, 0,
                                   (*NumberedSgprs)[*PairIndex]) &&
            isExactRegisterOperand(AddLow.Inst, 1,
                                   AddLow.Inst.getOperand(0).getReg()) &&
            isExactRegisterOperand(AddHigh.Inst, 0,
                                   (*NumberedSgprs)[*PairIndex + 1]) &&
            isExactRegisterOperand(AddHigh.Inst, 1,
                                   AddHigh.Inst.getOperand(0).getReg())) {
          std::optional<uint64_t> PcValue = checkedAddUint64(
              *GetPcAddress, GetPc.Size, "split-immediate set-PC value");
          if (PcValue) {
            uint64_t Delta = static_cast<uint64_t>(*LowDelta) |
                             (static_cast<uint64_t>(*HighDelta) << 32);
            // s_add_co_u32 / s_add_co_ci_u32 form one modulo-2^64 add.
            // A negative linker delta therefore wraps in unsigned arithmetic;
            // appendCandidate still requires every local result to be an
            // exact decoded instruction boundary.
            appendCandidate(I + 3, I + 3, *PcValue + Delta);
          }
        }
      }
    }

    // Tensile also emits a signed-direction materialized jump. The signed
    // displacement is a link-time constant, but the sequence uses a generic
    // compare/abs/add-or-sub shape:
    //
    //   get_pc Pair
    //   add_co_i32 Delta, Literal, Imm
    //   cmp_ge_i32 Delta, 0
    //   cbranch_scc1 Positive
    //   abs_i32 Delta, Delta
    //   sub_co_u32/sub_co_ci_u32 Pair, Pair, Delta
    //   set_pc Pair
    // Positive:
    //   add_co_u32/add_co_ci_u32 Pair, Pair, Delta
    //   set_pc Pair
    //
    // Both set-PC instructions have the same finite target. Keep the entire
    // shape in one audited materialization interval so an alternate entry
    // into either arm invalidates both candidates.
    if (I + 10 >= FunctionEndIndex)
      continue;
    const InternalDecodedInst &MakeDelta = Decoded[I + 1];
    const InternalDecodedInst &Compare = Decoded[I + 2];
    const InternalDecodedInst &Branch = Decoded[I + 3];
    const InternalDecodedInst &Abs = Decoded[I + 4];
    const InternalDecodedInst &SubLow = Decoded[I + 5];
    const InternalDecodedInst &SubHigh = Decoded[I + 6];
    const InternalDecodedInst &NegativeSetPc = Decoded[I + 7];
    const InternalDecodedInst &AddLow = Decoded[I + 8];
    const InternalDecodedInst &AddHigh = Decoded[I + 9];
    const InternalDecodedInst &PositiveSetPc = Decoded[I + 10];
    if (!MakeDelta.DecodeSucceeded || !Compare.DecodeSucceeded ||
        !Branch.DecodeSucceeded || !Abs.DecodeSucceeded ||
        !SubLow.DecodeSucceeded || !SubHigh.DecodeSucceeded ||
        !NegativeSetPc.DecodeSucceeded || !AddLow.DecodeSucceeded ||
        !AddHigh.DecodeSucceeded || !PositiveSetPc.DecodeSucceeded ||
        MakeDelta.Mnemonic != "s_add_co_i32" ||
        MakeDelta.Inst.getNumOperands() != 3 ||
        !MakeDelta.Inst.getOperand(0).isReg() ||
        !MakeDelta.Inst.getOperand(0).getReg() ||
        !MakeDelta.Inst.getOperand(2).isImm())
      continue;
    MCRegister DeltaReg(MakeDelta.Inst.getOperand(0).getReg());
    if (LS.MRI->regsOverlap(DeltaReg, Pair))
      continue;
    if (Compare.Mnemonic != "s_cmp_ge_i32" ||
        Compare.Inst.getNumOperands() != 2 ||
        !isExactRegisterOperand(Compare.Inst, 0, DeltaReg) ||
        !Compare.Inst.getOperand(1).isImm() ||
        Compare.Inst.getOperand(1).getImm() != 0 ||
        Branch.Mnemonic != "s_cbranch_scc1" || Abs.Mnemonic != "s_abs_i32" ||
        Abs.Inst.getNumOperands() != 2 ||
        !isExactRegisterOperand(Abs.Inst, 0, DeltaReg) ||
        !isExactRegisterOperand(Abs.Inst, 1, DeltaReg) ||
        NegativeSetPc.Inst.getOpcode() != LS.SSetPcI64Opcode ||
        NegativeSetPc.Inst.getNumOperands() != 1 ||
        !isExactRegisterOperand(NegativeSetPc.Inst, 0, Pair) ||
        PositiveSetPc.Inst.getOpcode() != LS.SSetPcI64Opcode ||
        PositiveSetPc.Inst.getNumOperands() != 1 ||
        !isExactRegisterOperand(PositiveSetPc.Inst, 0, Pair))
      continue;
    std::optional<uint64_t> PositiveLabel =
        evaluateDirectControlFlowTarget(Branch, LS);
    if (!PositiveLabel || *PositiveLabel != AddLow.Offset)
      continue;

    auto matchesPairArithmetic = [&](const InternalDecodedInst &Low,
                                     const InternalDecodedInst &High,
                                     StringRef LowMnemonic,
                                     StringRef HighMnemonic) {
      if (Low.Mnemonic != LowMnemonic || High.Mnemonic != HighMnemonic ||
          Low.Inst.getNumOperands() != 3 || !Low.Inst.getOperand(0).isReg() ||
          !Low.Inst.getOperand(1).isReg() || !Low.Inst.getOperand(0).getReg() ||
          Low.Inst.getOperand(0).getReg() != Low.Inst.getOperand(1).getReg() ||
          !isExactRegisterOperand(Low.Inst, 2, DeltaReg) ||
          High.Inst.getNumOperands() != 3 || !High.Inst.getOperand(0).isReg() ||
          !High.Inst.getOperand(1).isReg() ||
          !High.Inst.getOperand(0).getReg() ||
          High.Inst.getOperand(0).getReg() !=
              High.Inst.getOperand(1).getReg() ||
          !High.Inst.getOperand(2).isImm() ||
          High.Inst.getOperand(2).getImm() != 0)
        return false;
      MCRegister LowReg(Low.Inst.getOperand(0).getReg());
      MCRegister HighReg(High.Inst.getOperand(0).getReg());
      std::optional<unsigned> LowIndex = numberedSgprIndex(*LS.MRI, LowReg);
      std::optional<unsigned> HighIndex = numberedSgprIndex(*LS.MRI, HighReg);
      return LowIndex && HighIndex && *HighIndex == *LowIndex + 1 &&
             LS.MRI->regsOverlap(LowReg, Pair) &&
             LS.MRI->regsOverlap(HighReg, Pair);
    };
    if (!matchesPairArithmetic(SubLow, SubHigh, "s_sub_co_u32",
                               "s_sub_co_ci_u32") ||
        !matchesPairArithmetic(AddLow, AddHigh, "s_add_co_u32",
                               "s_add_co_ci_u32"))
      continue;

    std::optional<uint32_t> FirstAddend =
        evaluateAbsoluteUint32Operand(MakeDelta.Inst.getOperand(1));
    if (!FirstAddend)
      continue;
    uint32_t DeltaBits =
        *FirstAddend +
        static_cast<uint32_t>(MakeDelta.Inst.getOperand(2).getImm());
    int64_t SignedDelta = static_cast<int32_t>(DeltaBits);
    std::optional<uint64_t> PcValue = checkedAddUint64(
        *GetPcAddress, GetPc.Size, "signed finite set-PC get-PC value");
    if (!PcValue)
      continue;
    uint64_t Target = *PcValue + static_cast<uint64_t>(SignedDelta);
    appendCandidate(I + 7, I + 10, Target);
    appendCandidate(I + 10, I + 10, Target);
  }

  // The one-shot get-PC/add-nc resolver permits unrelated register restores
  // between the add and transfer while stopping at any target-pair clobber or
  // control-flow boundary. Reuse it for set-PC as well as swap-PC so compiler
  // epilogues with delayed transfers remain finite candidates.
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &SetPc = Decoded[I];
    if (!SetPc.DecodeSucceeded ||
        SetPc.Inst.getOpcode() != LS.SSetPcI64Opcode ||
        SetPc.Inst.getNumOperands() != 1 || !SetPc.Inst.getOperand(0).isReg() ||
        !SetPc.Inst.getOperand(0).getReg())
      continue;
    MCRegister Pair(SetPc.Inst.getOperand(0).getReg());
    std::optional<MaterializedPcSequence> Sequence =
        resolveMaterializedPcTarget(Decoded, I, Pair, LS, TextAddr);
    if (!Sequence)
      continue;
    DenseMap<uint64_t, size_t>::const_iterator Begin =
        OffsetToIndex.find(Sequence->SequenceStart);
    if (Begin == OffsetToIndex.end() || Begin->second > I)
      continue;
    std::optional<uint64_t> SetPcAddress = checkedAddUint64(
        TextAddr, SetPc.Offset, "delayed finite set-PC address");
    if (!SetPcAddress)
      continue;
    const ElfView::FunctionTextRange *Range =
        findInnermostFunctionRange(*SetPcAddress, FunctionRanges);
    if (!Range || Range->Begin < TextAddr || Range->End > TextEnd ||
        Decoded[Begin->second].Offset < Range->Begin - TextAddr)
      continue;
    bool Contiguous = true;
    for (size_t J = Begin->second; J < I; ++J) {
      std::optional<uint64_t> End =
          checkedAddUint64(Decoded[J].Offset, Decoded[J].Size,
                           "delayed finite set-PC instruction end");
      if (!Decoded[J].DecodeSucceeded || !End ||
          *End != Decoded[J + 1].Offset) {
        Contiguous = false;
        break;
      }
    }
    if (!Contiguous)
      continue;
    std::optional<size_t> LocalTargetIndex;
    if (Sequence->Target >= TextAddr && Sequence->Target < TextEnd) {
      DenseMap<uint64_t, size_t>::const_iterator Target =
          OffsetToIndex.find(Sequence->Target - TextAddr);
      if (Target == OffsetToIndex.end())
        continue;
      LocalTargetIndex = Target->second;
    }
    Candidates.push_back({I, Begin->second, I, Sequence->Target,
                          LocalTargetIndex, Range->Begin - TextAddr,
                          Range->End - TextAddr});
  }
  llvm::sort(Candidates, [](const FiniteSetPcTransfer &LHS,
                            const FiniteSetPcTransfer &RHS) {
    return std::tie(LHS.InstIndex, LHS.Target, LHS.SequenceBeginIndex,
                    LHS.SequenceEndIndex) < std::tie(RHS.InstIndex, RHS.Target,
                                                     RHS.SequenceBeginIndex,
                                                     RHS.SequenceEndIndex);
  });
  SmallVector<FiniteSetPcTransfer, 8> Unambiguous;
  for (size_t I = 0; I != Candidates.size();) {
    size_t After = I + 1;
    while (After != Candidates.size() &&
           Candidates[After].InstIndex == Candidates[I].InstIndex)
      ++After;
    bool Same = llvm::all_of(
        ArrayRef<FiniteSetPcTransfer>(Candidates).slice(I, After - I),
        [&](const FiniteSetPcTransfer &Candidate) {
          return Candidate.Target == Candidates[I].Target &&
                 Candidate.SequenceBeginIndex ==
                     Candidates[I].SequenceBeginIndex &&
                 Candidate.SequenceEndIndex == Candidates[I].SequenceEndIndex;
        });
    if (Same)
      Unambiguous.push_back(Candidates[I]);
    else
      log() << "hotswap: rejected ambiguous exact set-PC materialization at "
               "0x"
            << utohexstr(Decoded[Candidates[I].InstIndex].Offset) << "\n";
    I = After;
  }
  return Unambiguous;
}

static BitVector computeStaticallyReachableInstructions(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    ArrayRef<uint64_t> DeclaredEntries, ArrayRef<uint64_t> ExternalEntries,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges, uint64_t TextAddr,
    ArrayRef<FiniteSetPcTransfer> FiniteSetPcTransfers) {
  BitVector Reachable(Decoded.size());
  DenseMap<uint64_t, size_t> OffsetToIndex;
  for (size_t I = 0; I != Decoded.size(); ++I)
    OffsetToIndex.try_emplace(Decoded[I].Offset, I);
  DenseMap<size_t, const FiniteSetPcTransfer *> TransferByInst;
  for (const FiniteSetPcTransfer &Transfer : FiniteSetPcTransfers)
    TransferByInst.try_emplace(Transfer.InstIndex, &Transfer);

  SmallVector<size_t, 32> Worklist;
  auto addRoot = [&](uint64_t Offset) {
    DenseMap<uint64_t, size_t>::const_iterator It = OffsetToIndex.find(Offset);
    if (It != OffsetToIndex.end())
      Worklist.push_back(It->second);
  };
  for (uint64_t Entry : DeclaredEntries)
    addRoot(Entry);
  for (uint64_t Entry : ExternalEntries)
    addRoot(Entry);
  for (const ElfView::FunctionTextRange &Range : FunctionRanges)
    if (Range.Begin >= TextAddr)
      addRoot(Range.Begin - TextAddr);
  if (Worklist.empty() && !Decoded.empty())
    Worklist.push_back(0);

  while (!Worklist.empty()) {
    size_t I = Worklist.pop_back_val();
    if (Reachable.test(I))
      continue;
    Reachable.set(I);
    const InternalDecodedInst &DI = Decoded[I];
    if (!DI.DecodeSucceeded)
      continue;
    auto addOffset = [&](uint64_t Offset) {
      DenseMap<uint64_t, size_t>::const_iterator It =
          OffsetToIndex.find(Offset);
      if (It != OffsetToIndex.end())
        Worklist.push_back(It->second);
    };
    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
        LS.MIA->isReturn(DI.Inst))
      continue;
    if (LS.MIA->isCall(DI.Inst)) {
      std::optional<uint64_t> Fallthrough = checkedAddUint64(
          DI.Offset, DI.Size, "finite set-PC call continuation");
      if (Fallthrough)
        addOffset(*Fallthrough);
      continue;
    }
    if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode ||
        LS.MIA->isIndirectBranch(DI.Inst)) {
      DenseMap<size_t, const FiniteSetPcTransfer *>::const_iterator Transfer =
          TransferByInst.find(I);
      if (Transfer != TransferByInst.end() &&
          Transfer->second->LocalTargetIndex)
        Worklist.push_back(*Transfer->second->LocalTargetIndex);
      continue;
    }
    if (LS.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
      if (Target)
        addOffset(*Target);
      if (LS.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    }
    std::optional<uint64_t> Fallthrough = checkedAddUint64(
        DI.Offset, DI.Size, "finite set-PC reachability fallthrough");
    if (Fallthrough)
      addOffset(*Fallthrough);
  }
  return Reachable;
}

static SmallVector<FiniteSetPcTransfer, 8> selectLeastReachableSetPcCandidates(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    ArrayRef<uint64_t> DeclaredEntries, ArrayRef<uint64_t> ExternalEntries,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges, uint64_t TextAddr,
    ArrayRef<FiniteSetPcTransfer> AllCandidates,
    const BitVector &ProvenCandidates, const BitVector &RejectedCandidates) {
  SmallVector<FiniteSetPcTransfer, 8> Selected;
  BitVector SelectedBits(AllCandidates.size());
  for (size_t I = 0; I != AllCandidates.size(); ++I)
    if (ProvenCandidates.test(I) && !RejectedCandidates.test(I)) {
      SelectedBits.set(I);
      Selected.push_back(AllCandidates[I]);
    }
  for (;;) {
    BitVector Reachable = computeStaticallyReachableInstructions(
        Decoded, LS, DeclaredEntries, ExternalEntries, FunctionRanges, TextAddr,
        Selected);
    bool Changed = false;
    for (size_t I = 0; I != AllCandidates.size(); ++I) {
      if (RejectedCandidates.test(I) || SelectedBits.test(I) ||
          !Reachable.test(AllCandidates[I].InstIndex))
        continue;
      SelectedBits.set(I);
      Selected.push_back(AllCandidates[I]);
      Changed = true;
    }
    if (!Changed)
      return Selected;
  }
}

/// Resolve uses of an SGPR pair whose value is selected once and reused across
/// control flow. A monotone intraprocedural solver propagates finite values
/// from proven get-PC materializations. Any unrecognized pair definition
/// introduces Unknown, so a bypass around the selector remains fail-closed.
static std::vector<ReachingCallTargets> resolveReusablePcValuesAtUses(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextEnd,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<ReachingPcUse> Uses,
    ArrayRef<FiniteSetPcTransfer> FiniteSetPcTransfers = {}) {
  std::vector<ReachingCallTargets> Resolved(Decoded.size());
  SmallVector<ReachingCallGroup, 8> Groups;
  DenseMap<size_t, const FiniteSetPcTransfer *> FiniteSetPcByInst;
  for (const FiniteSetPcTransfer &Transfer : FiniteSetPcTransfers)
    FiniteSetPcByInst.try_emplace(Transfer.InstIndex, &Transfer);

  for (const ReachingPcUse &Use : Uses) {
    size_t I = Use.InstIndex;
    if (I >= Decoded.size() || !Decoded[I].DecodeSucceeded || !Use.Register)
      continue;
    MCRegister TargetRegister = Use.Register;

    const ElfView::FunctionTextRange *Best = nullptr;
    uint64_t Address = TextAddr + Decoded[I].Offset;
    for (const ElfView::FunctionTextRange &Range : FunctionRanges)
      if (Range.Begin <= Address && Address < Range.End &&
          (!Best || Range.Begin > Best->Begin))
        Best = &Range;
    if (!Best || Best->Begin < TextAddr || Best->End > TextEnd)
      continue;
    uint64_t Begin = Best->Begin - TextAddr;
    uint64_t End = Best->End - TextAddr;
    ReachingCallGroup *Group = nullptr;
    for (ReachingCallGroup &Candidate : Groups)
      if (Candidate.Begin == Begin && Candidate.End == End &&
          Candidate.TargetRegister == TargetRegister) {
        Group = &Candidate;
        break;
      }
    if (!Group) {
      Groups.push_back({Begin, End, TargetRegister, {}});
      Group = &Groups.back();
    }
    Group->Calls.push_back(I);
  }

  DenseMap<uint64_t, size_t> OffsetToIndex;
  for (size_t I = 0; I != Decoded.size(); ++I)
    OffsetToIndex[Decoded[I].Offset] = I;

  for (const ReachingCallGroup &Group : Groups) {
    ArrayRef<InternalDecodedInst>::const_iterator Begin =
        llvm::lower_bound(Decoded, Group.Begin,
                          [](const InternalDecodedInst &DI, uint64_t Offset) {
                            return DI.Offset < Offset;
                          });
    ArrayRef<InternalDecodedInst>::const_iterator End =
        std::lower_bound(Begin, Decoded.end(), Group.End,
                         [](const InternalDecodedInst &DI, uint64_t Offset) {
                           return DI.Offset < Offset;
                         });
    size_t BeginIndex = static_cast<size_t>(Begin - Decoded.begin());
    size_t EndIndex = static_cast<size_t>(End - Decoded.begin());
    if (BeginIndex == EndIndex)
      continue;

    DenseMap<size_t, size_t> Starters;
    DenseMap<size_t, uint64_t> Completions;
    DenseMap<size_t, SmallVector<size_t, 2>> Intermediates;
    for (size_t I = BeginIndex; I != EndIndex; ++I) {
      std::optional<std::pair<size_t, uint64_t>> Match =
          matchReusablePcMaterialization(Decoded, I, EndIndex,
                                         Group.TargetRegister, LS, TextAddr);
      if (Match) {
        Starters[I] = Match->first;
        Completions[Match->first] = Match->second;
        for (size_t J = I + 1; J != Match->first; ++J)
          if (definesOverlappingRegister(Decoded[J], LS, Group.TargetRegister))
            Intermediates[J].push_back(Match->first);
      }
    }
    if (Completions.empty())
      continue;

    std::vector<ReachingPcState> Before(EndIndex - BeginIndex);
    Before.front().Reached = true;
    Before.front().HasUnknown = true;
    SmallVector<size_t, 32> Worklist;
    BitVector Queued(EndIndex - BeginIndex);
    Worklist.push_back(BeginIndex);
    Queued.set(0);

    while (!Worklist.empty()) {
      size_t I = Worklist.pop_back_val();
      Queued.reset(I - BeginIndex);
      ReachingPcState State = Before[I - BeginIndex];
      const InternalDecodedInst &DI = Decoded[I];

      DenseMap<size_t, size_t>::const_iterator Starter = Starters.find(I);
      DenseMap<size_t, uint64_t>::const_iterator Completion =
          Completions.find(I);
      if (Starter != Starters.end()) {
        // The get-PC instruction overwrites the complete target pair. Record a
        // token proving that this path entered the exact materialization; the
        // completion may only produce a known target from that token.
        State.HasUnknown = false;
        State.Targets.clear();
        State.ActiveMaterializations.assign(1, Starter->second);
      } else if (Completion != Completions.end()) {
        bool HasMatchingToken =
            llvm::is_contained(State.ActiveMaterializations, I);
        bool HasBypassPath =
            State.HasUnknown || !State.Targets.empty() ||
            llvm::any_of(State.ActiveMaterializations,
                         [I](size_t Active) { return Active != I; });
        State.HasUnknown = HasBypassPath || !HasMatchingToken;
        State.Targets.clear();
        State.ActiveMaterializations.clear();
        if (HasMatchingToken)
          State.Targets.push_back(Completion->second);
      } else if (definesOverlappingRegister(DI, LS, Group.TargetRegister)) {
        // Preserve only tokens for which this is a proven instruction inside
        // the exact matched sequence. All other reaching values are clobbered.
        SmallVector<size_t, 4> Preserved;
        DenseMap<size_t, SmallVector<size_t, 2>>::const_iterator Intermediate =
            Intermediates.find(I);
        if (Intermediate != Intermediates.end())
          for (size_t Active : State.ActiveMaterializations)
            if (llvm::is_contained(Intermediate->second, Active))
              Preserved.push_back(Active);
        if (State.HasUnknown || !State.Targets.empty() ||
            Preserved.size() != State.ActiveMaterializations.size())
          State.HasUnknown = true;
        State.Targets.clear();
        State.ActiveMaterializations = std::move(Preserved);
      }

      if (llvm::is_contained(Group.Calls, I) && !State.HasUnknown &&
          State.ActiveMaterializations.empty() && !State.Targets.empty())
        Resolved[I] = State.Targets;

      SmallVector<size_t, 2> Successors;
      auto appendFallthrough = [&]() {
        if (I + 1 < EndIndex)
          Successors.push_back(I + 1);
      };
      if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
          DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
          LS.MIA->isReturn(DI.Inst)) {
        // No successor.
      } else if (LS.MIA->isCall(DI.Inst)) {
        appendFallthrough();
      } else if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode ||
                 LS.MIA->isIndirectBranch(DI.Inst)) {
        DenseMap<size_t, const FiniteSetPcTransfer *>::const_iterator
            FiniteSetPc = FiniteSetPcByInst.find(I);
        if (FiniteSetPc != FiniteSetPcByInst.end() &&
            FiniteSetPc->second->LocalTargetIndex &&
            *FiniteSetPc->second->LocalTargetIndex >= BeginIndex &&
            *FiniteSetPc->second->LocalTargetIndex < EndIndex)
          Successors.push_back(*FiniteSetPc->second->LocalTargetIndex);
        // All other indirect jumps or bounded returns leave this
        // intraprocedural path.
      } else if (LS.MIA->isBranch(DI.Inst)) {
        std::optional<uint64_t> Target =
            evaluateDirectControlFlowTarget(DI, LS);
        if (Target) {
          DenseMap<uint64_t, size_t>::const_iterator TargetIndex =
              OffsetToIndex.find(*Target);
          if (TargetIndex != OffsetToIndex.end() &&
              TargetIndex->second >= BeginIndex &&
              TargetIndex->second < EndIndex)
            Successors.push_back(TargetIndex->second);
        }
        if (!LS.MIA->isUnconditionalBranch(DI.Inst))
          appendFallthrough();
      } else {
        appendFallthrough();
      }

      for (size_t Successor : Successors) {
        if (mergeReachingPcState(Before[Successor - BeginIndex], State) &&
            !Queued.test(Successor - BeginIndex)) {
          Worklist.push_back(Successor);
          Queued.set(Successor - BeginIndex);
        }
      }
    }
  }
  return Resolved;
}

/// Call-specific wrapper for the generic reusable-PC value solver.
static std::vector<ReachingCallTargets> resolveReusablePcCallTargets(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextEnd,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<FiniteSetPcTransfer> FiniteSetPcTransfers = {}) {
  SmallVector<ReachingPcUse, 8> Uses;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &Call = Decoded[I];
    if (!Call.DecodeSucceeded || Call.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
        Call.Inst.getNumOperands() < 2)
      continue;
    const MCOperand &TargetOp =
        Call.Inst.getOperand(Call.Inst.getNumOperands() - 1);
    if (TargetOp.isReg() && TargetOp.getReg())
      Uses.push_back({I, MCRegister(TargetOp.getReg())});
  }
  return resolveReusablePcValuesAtUses(Decoded, LS, TextAddr, TextEnd,
                                       FunctionRanges, Uses,
                                       FiniteSetPcTransfers);
}

struct KnownCallSite {
  size_t InstIndex = 0;
  uint64_t Target = 0;
  uint64_t Continuation = 0;
  MCRegister ReturnRegister;
};

struct BoundedSetPcReturn {
  size_t InstIndex = 0;
  SmallVector<uint64_t, 2> Targets;
};

struct DirectTargetSource {
  size_t InstIndex = 0;
  uint64_t Target = 0;
};

struct KnownCallEntry {
  uint64_t Entry = 0;
  size_t CallIndex = 0;
};

static bool compareKnownCallEntries(const KnownCallEntry &LHS,
                                    const KnownCallEntry &RHS) {
  return std::tie(LHS.Entry, LHS.CallIndex) <
         std::tie(RHS.Entry, RHS.CallIndex);
}

struct ExternalCallContinuation {
  size_t InstIndex = 0;
  uint64_t Continuation = 0;
};

struct CallContinuationSource {
  size_t InstIndex = 0;
  uint64_t Continuation = 0;
};

struct FallthroughEntryInfo {
  bool Proven = false;
  uint64_t ChainBegin = 0;
};

struct ControlFlowScanIndex {
  DenseMap<size_t, PcMaterializedCallInfo> MaterializedCalls;
  DenseMap<uint64_t, FallthroughEntryInfo> FallthroughEntries;
  SmallVector<KnownCallSite, 4> Calls;
  SmallVector<KnownCallEntry, 8> CallsByTarget;
  SmallVector<KnownCallEntry, 16> CallEntries;
  DenseMap<size_t, MCRegister> CallReturnRegistersBySource;
  SmallVector<CallContinuationSource, 4> CallContinuationsByOffset;
  SmallVector<ExternalCallContinuation, 4> ExternalCallContinuations;
  SmallVector<size_t, 16> SetPcIndices;
  SmallVector<size_t, 4> UnboundedIndirectIndices;
  SmallVector<size_t, 16> BranchOrCallIndices;
  SmallVector<DirectTargetSource, 16> DirectTargetsByTarget;
  bool HasUnboundedIndirectEntry = false;
};

static void indexKnownCalls(ControlFlowScanIndex &Index) {
  Index.CallsByTarget.clear();
  Index.CallEntries.clear();
  Index.CallReturnRegistersBySource.clear();
  Index.CallsByTarget.reserve(Index.Calls.size());
  Index.CallEntries.reserve(Index.Calls.size() * 2);
  for (size_t CallIndex = 0; CallIndex != Index.Calls.size(); ++CallIndex) {
    const KnownCallSite &Call = Index.Calls[CallIndex];
    Index.CallsByTarget.push_back({Call.Target, CallIndex});
    Index.CallEntries.push_back({Call.Target, CallIndex});
    Index.CallEntries.push_back({Call.Continuation, CallIndex});
    Index.CallReturnRegistersBySource.try_emplace(Call.InstIndex,
                                                  Call.ReturnRegister);
  }
  llvm::sort(Index.CallsByTarget, compareKnownCallEntries);
  llvm::sort(Index.CallEntries, compareKnownCallEntries);
}

static bool hasPcRelativeOperand(const InternalDecodedInst &DI,
                                 const LLVMState &LS) {
  for (const MCOperandInfo &Operand :
       LS.MCII->get(DI.Inst.getOpcode()).operands())
    if (Operand.OperandType == MCOI::OPERAND_PCREL)
      return true;
  return false;
}

static std::optional<MCRegister>
getCallReturnRegister(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (!DI.DecodeSucceeded || !LS.MIA->isCall(DI.Inst) ||
      DI.Inst.getNumOperands() == 0 || !DI.Inst.getOperand(0).isReg() ||
      !DI.Inst.getOperand(0).getReg())
    return std::nullopt;
  return MCRegister(DI.Inst.getOperand(0).getReg());
}

static std::optional<uint64_t>
getDirectTextTarget(const InternalDecodedInst &DI, const LLVMState &LS,
                    uint64_t TextAddr, uint64_t TextEnd) {
  if (!DI.DecodeSucceeded ||
      (!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
      LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst))
    return std::nullopt;

  if (hasPcRelativeOperand(DI, LS))
    return evaluateDirectControlFlowTarget(DI, LS);

  if (DI.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
      DI.Inst.getNumOperands() == 0 ||
      !DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm())
    return std::nullopt;
  uint64_t AbsoluteTarget = static_cast<uint64_t>(
      DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm());
  if (AbsoluteTarget < TextAddr || AbsoluteTarget >= TextEnd)
    return std::nullopt;
  return AbsoluteTarget - TextAddr;
}

static std::optional<ControlFlowScanIndex>
buildControlFlowScanIndex(ArrayRef<InternalDecodedInst> Decoded,
                          const LLVMState &LS, uint64_t TextAddr,
                          uint64_t TextEnd,
                          ArrayRef<ElfView::FunctionTextRange> FunctionRanges) {
  ControlFlowScanIndex Index;
  DenseSet<uint64_t> FunctionBegins;
  for (const ElfView::FunctionTextRange &Range : FunctionRanges)
    if (Range.Begin >= TextAddr && Range.Begin < TextEnd)
      FunctionBegins.insert(Range.Begin - TextAddr);

  bool FallthroughProven = true;
  uint64_t FallthroughChainBegin = 0;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (I == 0) {
      FallthroughChainBegin = DI.Offset;
    } else {
      const InternalDecodedInst &Predecessor = Decoded[I - 1];
      bool EndOverflows =
          Predecessor.Offset >
          std::numeric_limits<uint64_t>::max() - Predecessor.Size;
      if (EndOverflows || Predecessor.Offset + Predecessor.Size != DI.Offset ||
          !Predecessor.DecodeSucceeded) {
        FallthroughProven = false;
        FallthroughChainBegin = DI.Offset;
      } else if (LS.MIA->isBarrier(Predecessor.Inst)) {
        FallthroughProven = true;
        FallthroughChainBegin = DI.Offset;
      }
    }
    if (FunctionBegins.contains(DI.Offset))
      Index.FallthroughEntries.try_emplace(
          DI.Offset,
          FallthroughEntryInfo{FallthroughProven, FallthroughChainBegin});

    std::optional<PcMaterializedCallInfo> Materialized =
        matchPcMaterializedCall(Decoded, I, LS, TextAddr);
    if (Materialized)
      Index.MaterializedCalls.try_emplace(I, *Materialized);

    std::optional<MCRegister> ReturnRegister = getCallReturnRegister(DI, LS);
    if (ReturnRegister) {
      std::optional<uint64_t> Target;
      bool HasFiniteExternalTarget = false;
      if (Materialized) {
        uint64_t AbsoluteTarget = Materialized->Target;
        if (AbsoluteTarget >= TextAddr && AbsoluteTarget < TextEnd)
          Target = AbsoluteTarget - TextAddr;
        else
          HasFiniteExternalTarget = true;
      } else if (DI.Inst.getOpcode() == LS.SSwapPcI64Opcode &&
                 DI.Inst.getNumOperands() != 0 &&
                 DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm()) {
        uint64_t AbsoluteTarget = static_cast<uint64_t>(
            DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm());
        if (AbsoluteTarget >= TextAddr && AbsoluteTarget < TextEnd)
          Target = AbsoluteTarget - TextAddr;
        else
          HasFiniteExternalTarget = true;
      } else if (hasPcRelativeOperand(DI, LS)) {
        std::optional<uint64_t> RelativeTarget =
            evaluateDirectControlFlowTarget(DI, LS);
        if (RelativeTarget) {
          uint64_t TextSize = TextEnd - TextAddr;
          if (*RelativeTarget < TextSize)
            Target = *RelativeTarget;
          else
            HasFiniteExternalTarget = true;
        }
      } else {
        Target = getDirectTextTarget(DI, LS, TextAddr, TextEnd);
      }
      if (Target || HasFiniteExternalTarget) {
        std::optional<uint64_t> Continuation = checkedAddUint64(
            DI.Offset, DI.Size, "known call continuation address");
        if (!Continuation)
          return std::nullopt;
        if (Target)
          Index.Calls.push_back({I, *Target, *Continuation, *ReturnRegister});
        if (HasFiniteExternalTarget)
          Index.ExternalCallContinuations.push_back({I, *Continuation});
      }
    }

    if (DI.DecodeSucceeded && DI.Inst.getOpcode() == LS.SSetPcI64Opcode)
      Index.SetPcIndices.push_back(I);

    // Set-PC returns are checked separately against BoundedReturnPositions.
    // MC lowering erases their return pseudo identity, so including them in
    // this generic bucket would make even a proven bounded return look like
    // an arbitrary object-wide entry.
    if (DI.DecodeSucceeded && DI.Inst.getOpcode() != LS.SSetPcI64Opcode &&
        !LS.MIA->isReturn(DI.Inst) &&
        (LS.MIA->isIndirectBranch(DI.Inst) ||
         DI.Inst.getOpcode() == LS.SAddPcI64Opcode)) {
      Index.HasUnboundedIndirectEntry = true;
      Index.UnboundedIndirectIndices.push_back(I);
    }

    if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
        LS.MIA->isReturn(DI.Inst))
      continue;
    Index.BranchOrCallIndices.push_back(I);

    std::optional<uint64_t> DirectTarget =
        getDirectTextTarget(DI, LS, TextAddr, TextEnd);
    if (DirectTarget)
      Index.DirectTargetsByTarget.push_back({I, *DirectTarget});
  }
  llvm::sort(Index.DirectTargetsByTarget,
             [](const DirectTargetSource &LHS, const DirectTargetSource &RHS) {
               return std::tie(LHS.Target, LHS.InstIndex) <
                      std::tie(RHS.Target, RHS.InstIndex);
             });
  return Index;
}

static bool hasUnprovenFallthroughEntry(ArrayRef<InternalDecodedInst> Decoded,
                                        uint64_t FunctionBegin,
                                        uint64_t ReturnOffset,
                                        ArrayRef<uint64_t> DeclaredEntries,
                                        const ControlFlowScanIndex &Index) {
  if (FunctionBegin == 0)
    return false;

  DenseMap<uint64_t, FallthroughEntryInfo>::const_iterator Fallthrough =
      Index.FallthroughEntries.find(FunctionBegin);
  if (Fallthrough == Index.FallthroughEntries.end()) {
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: function entry at 0x"
          << utohexstr(FunctionBegin) << " is not an instruction boundary\n";
    return true;
  }

  if (!Fallthrough->second.Proven) {
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: fallthrough into function entry "
             "at 0x"
          << utohexstr(FunctionBegin) << " is unprovable\n";
    return true;
  }
  uint64_t ChainBegin = Fallthrough->second.ChainBegin;

  if (ChainBegin == FunctionBegin)
    return false;

  ArrayRef<uint64_t>::iterator DeclaredEntry = std::lower_bound(
      DeclaredEntries.begin(), DeclaredEntries.end(), ChainBegin);
  if (DeclaredEntry != DeclaredEntries.end() &&
      *DeclaredEntry < FunctionBegin) {
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: declared entry at 0x"
          << utohexstr(*DeclaredEntry) << " falls through to function entry 0x"
          << utohexstr(FunctionBegin) << "\n";
    return true;
  }

  SmallVector<KnownCallEntry, 16>::const_iterator CallEntry =
      llvm::lower_bound(Index.CallEntries, ChainBegin,
                        [](const KnownCallEntry &Indexed, uint64_t Offset) {
                          return Indexed.Entry < Offset;
                        });
  for (;
       CallEntry != Index.CallEntries.end() && CallEntry->Entry < FunctionBegin;
       ++CallEntry) {
    const KnownCallSite &Call = Index.Calls[CallEntry->CallIndex];
    uint64_t Source = Decoded[Call.InstIndex].Offset;
    if (Source >= ChainBegin && Source < FunctionBegin)
      continue;
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: call at 0x" << utohexstr(Source)
          << " enters the fallthrough chain at 0x"
          << utohexstr(CallEntry->Entry) << "\n";
    return true;
  }

  for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations) {
    uint64_t Source = Decoded[Call.InstIndex].Offset;
    if (Call.Continuation >= ChainBegin && Call.Continuation < FunctionBegin) {
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
            << " is not a bounded return: external call at 0x"
            << utohexstr(Source) << " returns into the fallthrough chain at 0x"
            << utohexstr(Call.Continuation) << "\n";
      return true;
    }
  }

  SmallVector<DirectTargetSource, 16>::const_iterator FirstTarget =
      llvm::lower_bound(Index.DirectTargetsByTarget, ChainBegin,
                        [](const DirectTargetSource &Source, uint64_t Target) {
                          return Source.Target < Target;
                        });
  size_t FirstSourceIndex = Decoded.size();
  uint64_t FirstSourceTarget = 0;
  for (SmallVector<DirectTargetSource, 16>::const_iterator It = FirstTarget;
       It != Index.DirectTargetsByTarget.end() && It->Target < FunctionBegin;
       ++It) {
    const InternalDecodedInst &Source = Decoded[It->InstIndex];
    if (Source.Offset >= ChainBegin && Source.Offset < FunctionBegin)
      continue;
    if (It->InstIndex < FirstSourceIndex) {
      FirstSourceIndex = It->InstIndex;
      FirstSourceTarget = It->Target;
    }
  }
  if (FirstSourceIndex != Decoded.size()) {
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: control flow at 0x"
          << utohexstr(Decoded[FirstSourceIndex].Offset)
          << " enters the fallthrough chain at 0x"
          << utohexstr(FirstSourceTarget) << "\n";
    return true;
  }
  return false;
}

static std::optional<SmallVector<BoundedSetPcReturn, 2>>
collectBoundedSetPcReturns(ArrayRef<InternalDecodedInst> Decoded,
                           const LLVMState &LS, uint64_t TextAddr,
                           uint64_t TextEnd, ArrayRef<uint64_t> DeclaredEntries,
                           ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
                           ArrayRef<uint64_t> ExternalEntries,
                           const ControlFlowScanIndex &Index) {
  SmallVector<BoundedSetPcReturn, 2> Returns;
  SmallVector<uint64_t, 16> SortedDeclaredEntries(DeclaredEntries);
  llvm::sort(SortedDeclaredEntries);
  SortedDeclaredEntries.erase(
      std::unique(SortedDeclaredEntries.begin(), SortedDeclaredEntries.end()),
      SortedDeclaredEntries.end());

  SmallVector<uint64_t, 16> SortedExternalEntries(ExternalEntries);
  llvm::sort(SortedExternalEntries);
  SortedExternalEntries.erase(
      std::unique(SortedExternalEntries.begin(), SortedExternalEntries.end()),
      SortedExternalEntries.end());

  SmallVector<SmallVector<size_t, 2>, 16> CandidateRanges(
      Index.SetPcIndices.size());
  for (size_t RangeIndex = 0; RangeIndex != FunctionRanges.size();
       ++RangeIndex) {
    const ElfView::FunctionTextRange &Range = FunctionRanges[RangeIndex];
    if (Range.End <= Range.Begin)
      continue;
    SmallVector<size_t, 16>::const_iterator First = llvm::lower_bound(
        Index.SetPcIndices, Range.Begin,
        [&](size_t InstIndex, uint64_t Address) {
          return TextAddr + Decoded[InstIndex].Offset < Address;
        });
    SmallVector<size_t, 16>::const_iterator After = std::lower_bound(
        First, Index.SetPcIndices.end(), Range.End,
        [&](size_t InstIndex, uint64_t Address) {
          return TextAddr + Decoded[InstIndex].Offset < Address;
        });
    for (SmallVector<size_t, 16>::const_iterator It = First; It != After;
         ++It) {
      size_t Position = static_cast<size_t>(It - Index.SetPcIndices.begin());
      CandidateRanges[Position].push_back(RangeIndex);
    }
  }

  for (size_t ReturnPosition = 0; ReturnPosition != Index.SetPcIndices.size();
       ++ReturnPosition) {
    size_t ReturnIndex = Index.SetPcIndices[ReturnPosition];
    const InternalDecodedInst &Return = Decoded[ReturnIndex];
    // AMDGPUMCInstLower lowers S_SETPC_B64_return to S_SETPC_B64, so the
    // decoded instruction no longer carries MIA::isReturn identity. Recover
    // only the bounded local-function form from its call/link dataflow.
    if (!Return.DecodeSucceeded ||
        Return.Inst.getOpcode() != LS.SSetPcI64Opcode ||
        Return.Inst.getNumOperands() != 1 ||
        !Return.Inst.getOperand(0).isReg() ||
        !Return.Inst.getOperand(0).getReg())
      continue;
    MCRegister ReturnRegister(Return.Inst.getOperand(0).getReg());

    for (size_t RangeIndex : CandidateRanges[ReturnPosition]) {
      const ElfView::FunctionTextRange &Range = FunctionRanges[RangeIndex];
      if (Range.Begin < TextAddr || Range.Begin >= TextEnd ||
          Range.End <= Range.Begin || Range.End > TextEnd ||
          (Range.Symbol && Range.Symbol->getBinding() != ELF::STB_LOCAL))
        continue;
      uint64_t FunctionBegin = Range.Begin - TextAddr;
      uint64_t FunctionEnd = Range.End - TextAddr;
      if (Return.Offset < FunctionBegin || Return.Offset >= FunctionEnd)
        continue;

      bool Safe = true;
      SmallVector<uint64_t, 16>::iterator ExternalEntry =
          std::lower_bound(SortedExternalEntries.begin(),
                           SortedExternalEntries.end(), FunctionBegin);
      if (ExternalEntry != SortedExternalEntries.end() &&
          *ExternalEntry < FunctionEnd) {
        log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
              << " is not a bounded return: externally reachable entry at 0x"
              << utohexstr(*ExternalEntry) << " overlaps the local function\n";
        continue;
      }

      for (size_t AliasIndex : CandidateRanges[ReturnPosition]) {
        const ElfView::FunctionTextRange &Alias = FunctionRanges[AliasIndex];
        if (Alias.Begin == Range.Begin)
          continue;
        log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
              << " is not a bounded return: overlapping function entry at "
                 "0x"
              << utohexstr(Alias.Begin - TextAddr)
              << " makes entry provenance ambiguous\n";
        Safe = false;
        break;
      }
      if (!Safe)
        continue;

      SmallVector<uint64_t, 16>::iterator InteriorEntry =
          std::upper_bound(SortedDeclaredEntries.begin(),
                           SortedDeclaredEntries.end(), FunctionBegin);
      if (InteriorEntry != SortedDeclaredEntries.end() &&
          *InteriorEntry < FunctionEnd) {
        log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
              << " is not a bounded return: declared entry at 0x"
              << utohexstr(*InteriorEntry) << " bypasses the function entry\n";
        continue;
      }

      if (hasUnprovenFallthroughEntry(Decoded, FunctionBegin, Return.Offset,
                                      SortedDeclaredEntries, Index))
        continue;

      // The link pair must retain the value written by the incoming call
      // throughout the function. This includes blocks laid out after the
      // return that may branch back into its epilogue.
      ArrayRef<InternalDecodedInst>::const_iterator FunctionFirst =
          llvm::lower_bound(Decoded, FunctionBegin,
                            [](const InternalDecodedInst &DI, uint64_t Offset) {
                              return DI.Offset < Offset;
                            });
      ArrayRef<InternalDecodedInst>::const_iterator FunctionAfter =
          std::lower_bound(FunctionFirst, Decoded.end(), FunctionEnd,
                           [](const InternalDecodedInst &DI, uint64_t Offset) {
                             return DI.Offset < Offset;
                           });
      for (ArrayRef<InternalDecodedInst>::const_iterator It = FunctionFirst;
           It != FunctionAfter; ++It) {
        const InternalDecodedInst &DI = *It;
        // MC call instructions carry no transitive callee-clobber information.
        // Without interprocedural proof, a nested callee may overwrite the
        // outer link pair even when the call defines a different return pair.
        if (DI.DecodeSucceeded && LS.MIA->isCall(DI.Inst)) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: nested call at 0x"
                << utohexstr(DI.Offset) << " may clobber the link register\n";
          Safe = false;
          break;
        }
        if (!DI.DecodeSucceeded ||
            definesOverlappingRegister(DI, LS, ReturnRegister)) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: link register is "
                   "unprovable at 0x"
                << utohexstr(DI.Offset) << "\n";
          Safe = false;
          break;
        }
      }
      if (!Safe)
        continue;

      // A call that returns into this function does not supply its link pair
      // at the function entry. Reject continuations at the exact entry as
      // well as in the interior; the earlier fallthrough-chain check only
      // covers bytes laid out before FunctionBegin.
      SmallVector<CallContinuationSource, 4>::const_iterator Continuation =
          llvm::lower_bound(
              Index.CallContinuationsByOffset, FunctionBegin,
              [](const CallContinuationSource &Source, uint64_t Offset) {
                return Source.Continuation < Offset;
              });
      if (Continuation != Index.CallContinuationsByOffset.end() &&
          Continuation->Continuation < FunctionEnd) {
        log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
              << " is not a bounded return: call at 0x"
              << utohexstr(Decoded[Continuation->InstIndex].Offset)
              << " returns into the function at 0x"
              << utohexstr(Continuation->Continuation) << "\n";
        Safe = false;
      }
      if (!Safe)
        continue;
      SmallVector<ExternalCallContinuation, 4>::const_iterator
          ExternalContinuation = llvm::lower_bound(
              Index.ExternalCallContinuations, FunctionBegin,
              [](const ExternalCallContinuation &Source, uint64_t Offset) {
                return Source.Continuation < Offset;
              });
      if (ExternalContinuation != Index.ExternalCallContinuations.end() &&
          ExternalContinuation->Continuation < FunctionEnd) {
        log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
              << " is not a bounded return: external call at 0x"
              << utohexstr(Decoded[ExternalContinuation->InstIndex].Offset)
              << " returns into the function at 0x"
              << utohexstr(ExternalContinuation->Continuation) << "\n";
        Safe = false;
      }
      if (!Safe)
        continue;

      SmallVector<uint64_t, 2> Targets;
      SmallVector<KnownCallEntry, 8>::const_iterator CallAtTarget =
          llvm::lower_bound(Index.CallsByTarget, FunctionBegin,
                            [](const KnownCallEntry &Indexed, uint64_t Offset) {
                              return Indexed.Entry < Offset;
                            });
      for (; CallAtTarget != Index.CallsByTarget.end() &&
             CallAtTarget->Entry < FunctionEnd;
           ++CallAtTarget) {
        const KnownCallSite &Call = Index.Calls[CallAtTarget->CallIndex];
        if (Call.Target != FunctionBegin) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: call at 0x"
                << utohexstr(Decoded[Call.InstIndex].Offset)
                << " enters the function interior at 0x"
                << utohexstr(Call.Target) << "\n";
          Safe = false;
          break;
        }
        if (Call.ReturnRegister != ReturnRegister) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: call at 0x"
                << utohexstr(Decoded[Call.InstIndex].Offset)
                << " uses a different link register\n";
          Safe = false;
          break;
        }
        Targets.push_back(Call.Continuation);
      }
      if (!Safe || Targets.empty())
        continue;

      // A branch from outside the function would bypass the call link
      // definition. Direct calls to the function entry are allowed only when
      // they were collected above with this exact return register.
      SmallVector<DirectTargetSource, 16>::const_iterator FirstTarget =
          llvm::lower_bound(
              Index.DirectTargetsByTarget, FunctionBegin,
              [](const DirectTargetSource &Source, uint64_t Target) {
                return Source.Target < Target;
              });
      size_t FirstUnsafeSourceIndex = Decoded.size();
      uint64_t FirstUnsafeTarget = 0;
      for (SmallVector<DirectTargetSource, 16>::const_iterator It = FirstTarget;
           It != Index.DirectTargetsByTarget.end() && It->Target < FunctionEnd;
           ++It) {
        size_t SourceIndex = It->InstIndex;
        const InternalDecodedInst &Source = Decoded[SourceIndex];
        if (Source.Offset >= FunctionBegin && Source.Offset < FunctionEnd)
          continue;

        bool IsKnownEntryCall = false;
        if (LS.MIA->isCall(Source.Inst) && It->Target == FunctionBegin) {
          DenseMap<size_t, MCRegister>::const_iterator KnownCall =
              Index.CallReturnRegistersBySource.find(SourceIndex);
          IsKnownEntryCall =
              KnownCall != Index.CallReturnRegistersBySource.end() &&
              KnownCall->second == ReturnRegister;
        }
        if (!IsKnownEntryCall && SourceIndex < FirstUnsafeSourceIndex) {
          FirstUnsafeSourceIndex = SourceIndex;
          FirstUnsafeTarget = It->Target;
        }
      }
      if (FirstUnsafeSourceIndex != Decoded.size()) {
        log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
              << " is not a bounded return: control flow at 0x"
              << utohexstr(Decoded[FirstUnsafeSourceIndex].Offset)
              << " enters at 0x" << utohexstr(FirstUnsafeTarget) << "\n";
        Safe = false;
      }
      if (!Safe)
        continue;

      llvm::sort(Targets);
      Targets.erase(std::unique(Targets.begin(), Targets.end()), Targets.end());
      Returns.push_back({ReturnIndex, std::move(Targets)});
      break;
    }
  }
  return Returns;
}

static BitVector computeFiniteControlFlowReachability(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize, ArrayRef<uint64_t> DeclaredEntries,
    ArrayRef<uint64_t> ExternalEntries,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    const ControlFlowScanIndex &Index,
    ArrayRef<FiniteSetPcTransfer> FiniteSetPcTransfers,
    ArrayRef<BoundedSetPcReturn> BoundedReturns) {
  BitVector Reachable(Decoded.size());
  DenseMap<uint64_t, size_t> OffsetToIndex;
  for (size_t I = 0; I != Decoded.size(); ++I)
    OffsetToIndex.try_emplace(Decoded[I].Offset, I);
  DenseMap<size_t, const FiniteSetPcTransfer *> TransferByInst;
  for (const FiniteSetPcTransfer &Transfer : FiniteSetPcTransfers)
    TransferByInst.try_emplace(Transfer.InstIndex, &Transfer);
  DenseMap<size_t, const BoundedSetPcReturn *> ReturnByInst;
  for (const BoundedSetPcReturn &Return : BoundedReturns)
    ReturnByInst.try_emplace(Return.InstIndex, &Return);
  DenseMap<size_t, SmallVector<uint64_t, 2>> CallTargetsByInst;
  for (const KnownCallSite &Call : Index.Calls) {
    SmallVector<uint64_t, 2> &Targets = CallTargetsByInst[Call.InstIndex];
    if (!llvm::is_contained(Targets, Call.Target))
      Targets.push_back(Call.Target);
  }

  SmallVector<size_t, 32> Worklist;
  auto addOffset = [&](uint64_t Offset) {
    DenseMap<uint64_t, size_t>::const_iterator It = OffsetToIndex.find(Offset);
    if (It != OffsetToIndex.end())
      Worklist.push_back(It->second);
  };
  for (uint64_t Entry : DeclaredEntries)
    addOffset(Entry);
  for (uint64_t Entry : ExternalEntries)
    addOffset(Entry);
  for (const ElfView::FunctionTextRange &Range : FunctionRanges)
    if (Range.Begin >= TextAddr)
      addOffset(Range.Begin - TextAddr);
  if (Worklist.empty() && !Decoded.empty())
    Worklist.push_back(0);

  while (!Worklist.empty()) {
    size_t I = Worklist.pop_back_val();
    if (Reachable.test(I))
      continue;
    Reachable.set(I);
    const InternalDecodedInst &DI = Decoded[I];
    if (!DI.DecodeSucceeded)
      continue;
    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
        LS.MIA->isReturn(DI.Inst))
      continue;

    if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
      DenseMap<size_t, const FiniteSetPcTransfer *>::const_iterator Transfer =
          TransferByInst.find(I);
      if (Transfer != TransferByInst.end() &&
          Transfer->second->LocalTargetIndex)
        Worklist.push_back(*Transfer->second->LocalTargetIndex);
      DenseMap<size_t, const BoundedSetPcReturn *>::const_iterator Return =
          ReturnByInst.find(I);
      if (Return != ReturnByInst.end())
        for (uint64_t Target : Return->second->Targets)
          if (Target < TextSize)
            addOffset(Target);
      continue;
    }
    if (LS.MIA->isCall(DI.Inst)) {
      DenseMap<size_t, SmallVector<uint64_t, 2>>::const_iterator Targets =
          CallTargetsByInst.find(I);
      if (Targets != CallTargetsByInst.end())
        for (uint64_t Target : Targets->second)
          addOffset(Target);
      std::optional<uint64_t> Fallthrough = checkedAddUint64(
          DI.Offset, DI.Size, "finite call continuation reachability");
      if (Fallthrough && *Fallthrough < TextSize)
        addOffset(*Fallthrough);
      continue;
    }
    if (LS.MIA->isIndirectBranch(DI.Inst) ||
        DI.Inst.getOpcode() == LS.SAddPcI64Opcode)
      continue;
    if (LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) {
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
      if (Target)
        addOffset(*Target);
      if (LS.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    }
    std::optional<uint64_t> Fallthrough = checkedAddUint64(
        DI.Offset, DI.Size, "finite control-flow reachability fallthrough");
    if (Fallthrough && *Fallthrough < TextSize)
      addOffset(*Fallthrough);
  }
  return Reachable;
}

struct SymbolLessReturnRegion {
  uint64_t Entry = 0;
  MCRegister LinkRegister;
  SmallVector<size_t, 16> Instructions;
  SmallVector<size_t, 2> Returns;
  SmallVector<uint64_t, 8> Continuations;
};

static bool instructionMayFallThrough(const InternalDecodedInst &DI,
                                      const LLVMState &LS) {
  if (!DI.DecodeSucceeded)
    return true;
  if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
      DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
      LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst))
    return false;
  return !LS.MIA->isBranch(DI.Inst) || !LS.MIA->isUnconditionalBranch(DI.Inst);
}

/// Prove symbol-less callable regions from finite call targets. This is
/// intentionally based on forward CFG reachability from a concrete call
/// target, rather than layout labels or source tails. Every entry into the
/// resulting region must be one of the calls that supplies the exact link
/// pair, and the pair must remain untouched until each s_set_pc_i64 return.
static SmallVector<SymbolLessReturnRegion, 8> collectSymbolLessReturnRegions(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<uint64_t> DeclaredEntries, ArrayRef<uint64_t> ExternalEntries,
    const ControlFlowScanIndex &Index,
    ArrayRef<FiniteSetPcTransfer> FiniteSetPcTransfers,
    ArrayRef<BoundedSetPcReturn> PreviouslyBoundedReturns,
    const BitVector &ReachableCallSources) {
  struct CallGroup {
    uint64_t Entry = 0;
    MCRegister LinkRegister;
    SmallVector<uint64_t, 8> Continuations;
  };
  SmallVector<CallGroup, 8> Groups;
  DenseMap<std::pair<uint64_t, unsigned>, size_t> GroupPositions;
  for (const KnownCallSite &Call : Index.Calls) {
    if (!ReachableCallSources.test(Call.InstIndex))
      continue;
    std::pair<uint64_t, unsigned> Key{Call.Target, Call.ReturnRegister.id()};
    auto Inserted = GroupPositions.try_emplace(Key, Groups.size());
    if (Inserted.second) {
      Groups.push_back({Call.Target, Call.ReturnRegister, {}});
    }
    Groups[Inserted.first->second].Continuations.push_back(Call.Continuation);
  }

  DenseSet<uint64_t> RejectedRegionEntries;
  RejectedRegionEntries.insert(DeclaredEntries.begin(), DeclaredEntries.end());
  RejectedRegionEntries.insert(ExternalEntries.begin(), ExternalEntries.end());
  size_t RetainedGroups = 0;
  for (size_t I = 0; I != Groups.size(); ++I) {
    if (RejectedRegionEntries.contains(Groups[I].Entry))
      continue;
    if (RetainedGroups != I)
      Groups[RetainedGroups] = std::move(Groups[I]);
    ++RetainedGroups;
  }
  Groups.resize(RetainedGroups);
  if (Groups.empty())
    return {};

  DenseMap<uint64_t, size_t> OffsetToIndex;
  for (size_t I = 0; I != Decoded.size(); ++I)
    OffsetToIndex.try_emplace(Decoded[I].Offset, I);

  SmallVector<SymbolLessReturnRegion, 8> Regions;
  // Track overlap components as regions are proven instead of retaining every
  // overlapping instruction vector until a final quadratic pass. A value of
  // -1 is unclaimed, -2 belongs to an already-invalid overlap component, and
  // every nonnegative value names the sole still-valid owning region.
  std::vector<int64_t> RegionOwner;
  for (CallGroup &Group : Groups) {
    DenseMap<uint64_t, size_t>::const_iterator Entry =
        OffsetToIndex.find(Group.Entry);
    if (Entry == OffsetToIndex.end())
      continue;

    SymbolLessReturnRegion Region;
    Region.Entry = Group.Entry;
    Region.LinkRegister = Group.LinkRegister;
    llvm::sort(Group.Continuations);
    Group.Continuations.erase(
        std::unique(Group.Continuations.begin(), Group.Continuations.end()),
        Group.Continuations.end());
    Region.Continuations = Group.Continuations;

    SmallVector<size_t, 32> Worklist{Entry->second};
    BitVector Visited(Decoded.size());
    bool Safe = true;
    while (!Worklist.empty() && Safe) {
      size_t I = Worklist.pop_back_val();
      if (Visited.test(I))
        continue;
      Visited.set(I);
      Region.Instructions.push_back(I);
      const InternalDecodedInst &DI = Decoded[I];
      if (!DI.DecodeSucceeded) {
        Safe = false;
        break;
      }
      if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
        if (DI.Inst.getNumOperands() != 1 ||
            !isExactRegisterOperand(DI.Inst, 0, Group.LinkRegister)) {
          Safe = false;
          break;
        }
        Region.Returns.push_back(I);
        continue;
      }
      if (LS.MIA->isCall(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
          DI.Inst.getOpcode() == LS.SAddPcI64Opcode ||
          LS.MIA->isReturn(DI.Inst) ||
          definesOverlappingRegister(DI, LS, Group.LinkRegister)) {
        Safe = false;
        break;
      }
      if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
          DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode)
        continue;

      auto addSuccessor = [&](uint64_t Offset) {
        DenseMap<uint64_t, size_t>::const_iterator It =
            OffsetToIndex.find(Offset);
        if (It == OffsetToIndex.end()) {
          Safe = false;
          return;
        }
        Worklist.push_back(It->second);
      };
      if (LS.MIA->isBranch(DI.Inst)) {
        std::optional<uint64_t> Target =
            evaluateDirectControlFlowTarget(DI, LS);
        if (!Target || *Target >= TextSize) {
          Safe = false;
          break;
        }
        addSuccessor(*Target);
        if (!Safe)
          break;
        if (LS.MIA->isUnconditionalBranch(DI.Inst))
          continue;
      }
      std::optional<uint64_t> Fallthrough = checkedAddUint64(
          DI.Offset, DI.Size, "symbol-less return fallthrough");
      if (!Fallthrough || *Fallthrough >= TextSize) {
        Safe = false;
        break;
      }
      addSuccessor(*Fallthrough);
    }
    if (!Safe || Region.Returns.empty())
      continue;
    llvm::sort(Region.Instructions);
    llvm::sort(Region.Returns);

    auto containsInstructionByte = [&](uint64_t Offset) {
      for (size_t InstIndex : Region.Instructions) {
        const InternalDecodedInst &DI = Decoded[InstIndex];
        std::optional<uint64_t> End = checkedAddUint64(
            DI.Offset, DI.Size, "symbol-less return instruction end");
        // Overflow is itself unprovable; conservatively treat the queried
        // byte as overlapping the claimed region.
        if (!End || (Offset >= DI.Offset && Offset < *End))
          return true;
      }
      return false;
    };
    auto sourceIsInside = [&](size_t InstIndex) {
      return llvm::is_contained(Region.Instructions, InstIndex);
    };

    for (uint64_t EntryOffset : DeclaredEntries)
      if (containsInstructionByte(EntryOffset)) {
        Safe = false;
        break;
      }
    if (!Safe)
      continue;
    for (const ElfView::FunctionTextRange &Range : FunctionRanges) {
      if (Range.Begin < TextAddr || Range.Begin - TextAddr >= TextSize)
        continue;
      uint64_t EntryOffset = Range.Begin - TextAddr;
      if (EntryOffset != Region.Entry && containsInstructionByte(EntryOffset)) {
        Safe = false;
        break;
      }
    }
    if (!Safe)
      continue;
    for (uint64_t EntryOffset : ExternalEntries)
      if (containsInstructionByte(EntryOffset)) {
        Safe = false;
        break;
      }
    if (!Safe)
      continue;

    // Reject layout fallthrough from outside the reachable region.
    for (size_t InstIndex : Region.Instructions) {
      if (InstIndex == 0)
        continue;
      const InternalDecodedInst &DI = Decoded[InstIndex];
      const InternalDecodedInst &Predecessor = Decoded[InstIndex - 1];
      std::optional<uint64_t> PredecessorEnd =
          checkedAddUint64(Predecessor.Offset, Predecessor.Size,
                           "symbol-less return predecessor end");
      if (sourceIsInside(InstIndex - 1))
        continue;
      if (!Predecessor.DecodeSucceeded || !PredecessorEnd ||
          *PredecessorEnd != DI.Offset ||
          instructionMayFallThrough(Predecessor, LS)) {
        Safe = false;
        break;
      }
    }
    if (!Safe)
      continue;

    for (const KnownCallSite &Call : Index.Calls) {
      bool TargetInside = containsInstructionByte(Call.Target);
      bool ContinuationInside = containsInstructionByte(Call.Continuation);
      if (!TargetInside && !ContinuationInside)
        continue;
      if (sourceIsInside(Call.InstIndex)) {
        Safe = false;
        break;
      }
      if (ContinuationInside || Call.Target != Region.Entry ||
          Call.ReturnRegister != Region.LinkRegister) {
        Safe = false;
        break;
      }
    }
    if (!Safe)
      continue;
    for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations)
      if (containsInstructionByte(Call.Continuation)) {
        Safe = false;
        break;
      }
    if (!Safe)
      continue;

    for (const DirectTargetSource &Source : Index.DirectTargetsByTarget) {
      if (!containsInstructionByte(Source.Target) ||
          sourceIsInside(Source.InstIndex))
        continue;
      bool IsEntryCall = false;
      for (const KnownCallSite &Call : Index.Calls)
        IsEntryCall |= Call.InstIndex == Source.InstIndex &&
                       Call.Target == Region.Entry &&
                       Call.ReturnRegister == Region.LinkRegister;
      if (!IsEntryCall) {
        Safe = false;
        break;
      }
    }
    if (!Safe)
      continue;

    for (const FiniteSetPcTransfer &Transfer : FiniteSetPcTransfers) {
      if (!Transfer.LocalTargetIndex ||
          !llvm::is_contained(Region.Instructions,
                              *Transfer.LocalTargetIndex) ||
          sourceIsInside(Transfer.InstIndex))
        continue;
      Safe = false;
      break;
    }
    if (!Safe)
      continue;
    for (const BoundedSetPcReturn &Return : PreviouslyBoundedReturns) {
      if (sourceIsInside(Return.InstIndex))
        continue;
      for (uint64_t Target : Return.Targets)
        if (containsInstructionByte(Target)) {
          Safe = false;
          break;
        }
      if (!Safe)
        break;
    }
    if (!Safe)
      continue;

    if (RegionOwner.empty())
      RegionOwner.assign(Decoded.size(), -1);
    bool Overlaps = false;
    SmallVector<size_t, 4> OverlappingOwners;
    for (size_t InstIndex : Region.Instructions) {
      int64_t Owner = RegionOwner[InstIndex];
      if (Owner == -1)
        continue;
      Overlaps = true;
      if (Owner >= 0 &&
          !llvm::is_contained(OverlappingOwners, static_cast<size_t>(Owner)))
        OverlappingOwners.push_back(static_cast<size_t>(Owner));
    }
    if (Overlaps) {
      // Preserve the transitive overlap component as a compact tombstone so a
      // later region touching either side is rejected without retaining any
      // of the invalid regions' instruction vectors.
      for (size_t Owner : OverlappingOwners) {
        for (size_t InstIndex : Regions[Owner].Instructions)
          RegionOwner[InstIndex] = -2;
        Regions[Owner] = SymbolLessReturnRegion{};
      }
      for (size_t InstIndex : Region.Instructions)
        RegionOwner[InstIndex] = -2;
      continue;
    }

    size_t Owner = Regions.size();
    for (size_t InstIndex : Region.Instructions)
      RegionOwner[InstIndex] = static_cast<int64_t>(Owner);
    Regions.push_back(std::move(Region));
  }

  // Empty slots are regions invalidated by a later overlap.
  SmallVector<SymbolLessReturnRegion, 8> Disjoint;
  for (SymbolLessReturnRegion &Region : Regions)
    if (!Region.Instructions.empty())
      Disjoint.push_back(std::move(Region));
  return Disjoint;
}

struct FiniteControlFlowAudit {
  BitVector InvalidSetPcCandidates;
  bool Closed = false;
  bool HasUnboundedIndirectEntries = false;
};

static FiniteControlFlowAudit auditFiniteIndirectControlFlow(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<uint64_t> DeclaredEntries, ArrayRef<uint64_t> ExternalEntries,
    const ControlFlowScanIndex &Index,
    ArrayRef<FiniteSetPcTransfer> FiniteSetPcTransfers,
    ArrayRef<BoundedSetPcReturn> BoundedReturns,
    ArrayRef<SymbolLessReturnRegion> SymbolLessRegions) {
  FiniteControlFlowAudit Audit{BitVector(FiniteSetPcTransfers.size()), true};
  auto markUnboundedIndirectEntry = [&]() {
    Audit.Closed = false;
    Audit.HasUnboundedIndirectEntries = true;
  };

  for (size_t CandidateIndex = 0; CandidateIndex != FiniteSetPcTransfers.size();
       ++CandidateIndex) {
    const FiniteSetPcTransfer &Candidate = FiniteSetPcTransfers[CandidateIndex];
    uint64_t SequenceStart = Decoded[Candidate.SequenceBeginIndex].Offset;
    std::optional<uint64_t> SequenceEnd =
        checkedAddUint64(Decoded[Candidate.SequenceEndIndex].Offset,
                         Decoded[Candidate.SequenceEndIndex].Size,
                         "finite set-PC materialization end");
    auto isInteriorByte = [&](uint64_t Offset) {
      return !SequenceEnd || (Offset > SequenceStart && Offset < *SequenceEnd);
    };
    auto sourceIsSequence = [&](size_t InstIndex) {
      return InstIndex >= Candidate.SequenceBeginIndex &&
             InstIndex <= Candidate.SequenceEndIndex;
    };
    bool Safe = true;
    for (uint64_t Entry : DeclaredEntries)
      if (isInteriorByte(Entry)) {
        Safe = false;
        break;
      }
    if (Safe)
      for (const ElfView::FunctionTextRange &Range : FunctionRanges)
        if (Range.Begin >= TextAddr && isInteriorByte(Range.Begin - TextAddr)) {
          Safe = false;
          break;
        }
    if (Safe)
      for (uint64_t Entry : ExternalEntries)
        if (isInteriorByte(Entry)) {
          Safe = false;
          break;
        }
    if (Safe)
      for (const DirectTargetSource &Source : Index.DirectTargetsByTarget)
        if (isInteriorByte(Source.Target) &&
            !sourceIsSequence(Source.InstIndex)) {
          Safe = false;
          break;
        }
    if (Safe)
      for (const KnownCallSite &Call : Index.Calls)
        if ((isInteriorByte(Call.Target) ||
             isInteriorByte(Call.Continuation)) &&
            !sourceIsSequence(Call.InstIndex)) {
          Safe = false;
          break;
        }
    if (Safe)
      for (const ExternalCallContinuation &Call :
           Index.ExternalCallContinuations)
        if (isInteriorByte(Call.Continuation) &&
            !sourceIsSequence(Call.InstIndex)) {
          Safe = false;
          break;
        }
    if (Safe)
      for (const FiniteSetPcTransfer &Transfer : FiniteSetPcTransfers)
        if (Transfer.LocalTargetIndex &&
            isInteriorByte(Decoded[*Transfer.LocalTargetIndex].Offset)) {
          Safe = false;
          break;
        }
    if (Safe)
      for (const BoundedSetPcReturn &Return : BoundedReturns) {
        for (uint64_t TargetOffset : Return.Targets) {
          if (isInteriorByte(TargetOffset)) {
            Safe = false;
            break;
          }
        }
        if (!Safe)
          break;
      }
    if (!Safe)
      Audit.InvalidSetPcCandidates.set(CandidateIndex);
  }

  if (Audit.InvalidSetPcCandidates.any()) {
    Audit.Closed = false;
    return Audit;
  }

  BitVector BoundedSetPc(Decoded.size());
  for (const FiniteSetPcTransfer &Transfer : FiniteSetPcTransfers)
    BoundedSetPc.set(Transfer.InstIndex);
  for (const BoundedSetPcReturn &Return : BoundedReturns)
    BoundedSetPc.set(Return.InstIndex);
  for (const SymbolLessReturnRegion &Region : SymbolLessRegions)
    for (size_t Return : Region.Returns)
      BoundedSetPc.set(Return);

  // Symbol-less regions are inferred together so large mutually independent
  // helper families can reach a fixed point. Validate that joint proof before
  // treating any provisional return as bounded: each provisional source must
  // have exactly one owning region, and a return owned by another region (or
  // by a symbol-backed function) may not enter any byte of this region.
  DenseMap<size_t, unsigned> ProvisionalOwners;
  DenseMap<size_t, unsigned> PublishedProvisionalReturns;
  for (const SymbolLessReturnRegion &Region : SymbolLessRegions)
    for (size_t Return : Region.Returns) {
      ++ProvisionalOwners[Return];
      if (!llvm::is_contained(Region.Instructions, Return))
        markUnboundedIndirectEntry();
    }
  for (const BoundedSetPcReturn &Return : BoundedReturns)
    if (ProvisionalOwners.contains(Return.InstIndex))
      ++PublishedProvisionalReturns[Return.InstIndex];
  for (const auto &Owner : ProvisionalOwners)
    if (Owner.second != 1 ||
        PublishedProvisionalReturns.lookup(Owner.first) != 1)
      markUnboundedIndirectEntry();

  for (const SymbolLessReturnRegion &Region : SymbolLessRegions) {
    auto containsInstructionByte = [&](uint64_t Offset) {
      for (size_t InstIndex : Region.Instructions) {
        const InternalDecodedInst &DI = Decoded[InstIndex];
        std::optional<uint64_t> End = checkedAddUint64(
            DI.Offset, DI.Size, "symbol-less joint audit instruction end");
        if (!End || (Offset >= DI.Offset && Offset < *End))
          return true;
      }
      return false;
    };
    for (const BoundedSetPcReturn &Return : BoundedReturns) {
      if (llvm::is_contained(Region.Instructions, Return.InstIndex)) {
        if (!llvm::is_contained(Region.Returns, Return.InstIndex))
          markUnboundedIndirectEntry();
        continue;
      }
      for (uint64_t Target : Return.Targets)
        if (containsInstructionByte(Target)) {
          markUnboundedIndirectEntry();
          break;
        }
    }
  }

  BitVector Reachable = computeFiniteControlFlowReachability(
      Decoded, LS, TextAddr, TextSize, DeclaredEntries, ExternalEntries,
      FunctionRanges, Index, FiniteSetPcTransfers, BoundedReturns);
  DenseSet<uint64_t> InstructionOffsets;
  for (const InternalDecodedInst &DI : Decoded)
    InstructionOffsets.insert(DI.Offset);
  for (uint64_t Entry : DeclaredEntries)
    if (Entry < TextSize && !InstructionOffsets.contains(Entry))
      markUnboundedIndirectEntry();
  for (uint64_t Entry : ExternalEntries)
    if (Entry < TextSize && !InstructionOffsets.contains(Entry))
      markUnboundedIndirectEntry();
  for (const ElfView::FunctionTextRange &Range : FunctionRanges)
    if (Range.Begin >= TextAddr && Range.Begin - TextAddr < TextSize &&
        !InstructionOffsets.contains(Range.Begin - TextAddr))
      markUnboundedIndirectEntry();
  for (const DirectTargetSource &Source : Index.DirectTargetsByTarget)
    if (Reachable.test(Source.InstIndex) && Source.Target < TextSize &&
        !InstructionOffsets.contains(Source.Target))
      markUnboundedIndirectEntry();
  for (const KnownCallSite &Call : Index.Calls)
    if (Reachable.test(Call.InstIndex) &&
        ((!InstructionOffsets.contains(Call.Target) &&
          Call.Target < TextSize) ||
         (!InstructionOffsets.contains(Call.Continuation) &&
          Call.Continuation < TextSize)))
      markUnboundedIndirectEntry();
  for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations)
    if (Reachable.test(Call.InstIndex) && Call.Continuation < TextSize &&
        !InstructionOffsets.contains(Call.Continuation))
      markUnboundedIndirectEntry();
  for (size_t SetPc : Index.SetPcIndices)
    if (Reachable.test(SetPc) && !BoundedSetPc.test(SetPc))
      markUnboundedIndirectEntry();

  // Every call is also an indirect entry source until either a finite local
  // target or a finite external target has been recorded for it.
  BitVector FiniteCalls(Decoded.size());
  for (const KnownCallSite &Call : Index.Calls)
    FiniteCalls.set(Call.InstIndex);
  for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations)
    FiniteCalls.set(Call.InstIndex);

  // MC may also classify a register call as an indirect branch. Do not let
  // that generic classification create a false unbounded self-edge after the
  // exact call target and continuation have been admitted to this same joint
  // audit. Unknown calls remain unbounded in the call-specific loop below.
  for (size_t InstIndex : Index.UnboundedIndirectIndices)
    if (Reachable.test(InstIndex) && !FiniteCalls.test(InstIndex))
      markUnboundedIndirectEntry();

  for (size_t InstIndex : Index.BranchOrCallIndices) {
    if (!Reachable.test(InstIndex) || !LS.MIA->isCall(Decoded[InstIndex].Inst))
      continue;
    if (!FiniteCalls.test(InstIndex))
      markUnboundedIndirectEntry();
  }
  for (const BoundedSetPcReturn &Return : BoundedReturns) {
    if (!Reachable.test(Return.InstIndex))
      continue;
    for (uint64_t Target : Return.Targets)
      if (Target < TextSize && !InstructionOffsets.contains(Target))
        markUnboundedIndirectEntry();
  }
  for (int I = Reachable.find_first(); I >= 0; I = Reachable.find_next(I))
    if (!Decoded[static_cast<size_t>(I)].DecodeSucceeded)
      markUnboundedIndirectEntry();
  return Audit;
}

static bool
hasKnownControlFlowEntry(ArrayRef<uint64_t> DeclaredEntries,
                         ArrayRef<BoundedSetPcReturn> BoundedReturns,
                         const DenseMap<size_t, size_t> &BoundedReturnPositions,
                         const ControlFlowScanIndex &Index,
                         bool HasUnboundedIndirectEntries,
                         uint64_t SequenceStart, uint64_t SequenceEnd) {
  for (uint64_t Entry : DeclaredEntries)
    if (Entry > SequenceStart && Entry <= SequenceEnd)
      return true;

  for (size_t InstIndex : Index.SetPcIndices) {
    DenseMap<size_t, size_t>::const_iterator It =
        BoundedReturnPositions.find(InstIndex);
    if (It == BoundedReturnPositions.end())
      return true;
    const BoundedSetPcReturn &Return = BoundedReturns[It->second];
    for (uint64_t Target : Return.Targets)
      if (Target > SequenceStart && Target <= SequenceEnd)
        return true;
  }

  // Without bounding an indirect target, it may enter at any instruction in
  // the materialization. Keep the call unresolved rather than relying on the
  // indirect transfer's containing function alone.
  if (HasUnboundedIndirectEntries)
    return true;

  auto EntersSequence = [&](uint64_t Target) {
    return Target > SequenceStart && Target <= SequenceEnd;
  };
  for (const KnownCallSite &Call : Index.Calls)
    if (EntersSequence(Call.Target) || EntersSequence(Call.Continuation))
      return true;
  for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations)
    if (EntersSequence(Call.Continuation))
      return true;

  SmallVector<DirectTargetSource, 16>::const_iterator First =
      llvm::upper_bound(Index.DirectTargetsByTarget, SequenceStart,
                        [](uint64_t Target, const DirectTargetSource &Source) {
                          return Target < Source.Target;
                        });
  if (First != Index.DirectTargetsByTarget.end() &&
      First->Target <= SequenceEnd)
    return true;
  return false;
}

struct WellFormedAbiEntrySet {
  DenseSet<size_t> Calls;
  DenseSet<size_t> SetPcs;
  DenseSet<uint64_t> Targets;
};

static std::optional<unsigned> numberedVgprIndex(const MCRegisterInfo &MRI,
                                                 MCRegister Reg) {
  if (!Reg)
    return std::nullopt;
  StringRef Name(MRI.getName(Reg));
  if (!Name.consume_front("VGPR") || Name.empty() || Name.contains('_'))
    return std::nullopt;
  unsigned Index = 0;
  if (Name.getAsInteger(10, Index))
    return std::nullopt;
  return Index;
}

static bool isAbiCalleeSavedVgpr(unsigned Vgpr) {
  // CSR_AMDGPU_VGPRs in AMDGPUCallingConv.td uses eight-register saved
  // stripes alternating with eight caller-saved registers, starting at v40.
  return Vgpr >= 40 && Vgpr <= 255 && ((Vgpr - 40) % 16) < 8;
}

static bool instructionDefinesRegisterNamed(const InternalDecodedInst &DI,
                                            StringRef Name,
                                            const LLVMState &LS) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != NumDefs; ++I)
    if (DI.Inst.getOperand(I).isReg() && DI.Inst.getOperand(I).getReg() &&
        StringRef(LS.MRI->getName(DI.Inst.getOperand(I).getReg())) == Name)
      return true;
  return llvm::any_of(Desc.implicit_defs(), [&](MCPhysReg Reg) {
    return StringRef(LS.MRI->getName(Reg)) == Name;
  });
}

/// Return the exact packed VGPR-MSB mode installed by this instruction, or
/// std::nullopt when it does not install one. The constants mirror the
/// documented HW_REG_WAVE_MODE decode used by the WMMA mode analysis.
static std::optional<unsigned>
getExactAbiVgprMode(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (DI.Inst.getOpcode() == LS.SSetVgprMsbOpcode) {
    if (DI.Inst.getNumOperands() != 1 || !DI.Inst.getOperand(0).isImm())
      return std::nullopt;
    return static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) & 0xff;
  }
  if (DI.Inst.getOpcode() != LS.SSetregImm32Opcode ||
      DI.Inst.getNumOperands() != 2 || !DI.Inst.getOperand(0).isImm() ||
      !DI.Inst.getOperand(1).isImm())
    return std::nullopt;
  constexpr unsigned HwregIdMask = 0x3f;
  constexpr unsigned HwregIdMode = 1;
  unsigned Simm16 = static_cast<unsigned>(DI.Inst.getOperand(1).getImm());
  if ((Simm16 & HwregIdMask) != HwregIdMode)
    return std::nullopt;
  unsigned Raw =
      (static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) >> 12) & 0xff;
  return ((Raw >> 2) | (Raw << 6)) & 0xff;
}

static bool mayWriteAbiVgprBankUnknown(const InternalDecodedInst &DI,
                                       const LLVMState &LS) {
  if (DI.Inst.getOpcode() == LS.SSetVgprMsbOpcode)
    return !getExactAbiVgprMode(DI, LS);
  if (DI.Inst.getOpcode() == LS.SSetregImm32Opcode ||
      DI.Inst.getOpcode() == LS.SSetregB32Opcode) {
    if (DI.Inst.getNumOperands() == 0 ||
        !DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm())
      return true;
    constexpr unsigned HwregIdMask = 0x3f;
    constexpr unsigned HwregIdMode = 1;
    unsigned Simm16 = static_cast<unsigned>(
        DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm());
    if ((Simm16 & HwregIdMask) != HwregIdMode)
      return false;
    return !getExactAbiVgprMode(DI, LS);
  }
  return instructionDefinesRegisterNamed(DI, "MODE", LS);
}

bool matchesCanonicalLaneTransfer(const InternalDecodedInst &DI,
                                  StringRef Mnemonic, MCRegister Dst,
                                  MCRegister Src, int64_t Lane) {
  unsigned ExpectedOperands = Mnemonic == "v_readlane_b32"    ? 3
                              : Mnemonic == "v_writelane_b32" ? 4
                                                              : 0;
  if (!ExpectedOperands || DI.Mnemonic != Mnemonic ||
      DI.Inst.getNumOperands() != ExpectedOperands ||
      !isExactRegisterOperand(DI.Inst, 0, Dst) ||
      !isExactRegisterOperand(DI.Inst, 1, Src) ||
      !DI.Inst.getOperand(2).isImm() || DI.Inst.getOperand(2).getImm() != Lane)
    return false;
  return Mnemonic == "v_readlane_b32" ||
         isExactRegisterOperand(DI.Inst, 3, Dst);
}

/// MachineInstr analysis historically classifies a few EXEC-mask operations as
/// branches even though they do not change the PC.  The canonical-frame proof
/// needs the narrower hardware notion: an explicit PC opcode, a return or
/// indirect branch, or a branch/call carrying a PC-relative target.
bool isTruePcTransfer(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (!DI.DecodeSucceeded)
    return false;
  if (DI.Inst.getOpcode() == LS.SSwapPcI64Opcode ||
      DI.Inst.getOpcode() == LS.SSetPcI64Opcode ||
      DI.Inst.getOpcode() == LS.SAddPcI64Opcode || LS.MIA->isReturn(DI.Inst) ||
      LS.MIA->isIndirectBranch(DI.Inst))
    return true;
  return (LS.MIA->isBranch(DI.Inst) || LS.MIA->isCall(DI.Inst)) &&
         hasPcRelativeOperand(DI, LS);
}

struct CanonicalSavedLane {
  MCRegister Vgpr;
  int64_t Lane = -1;

  bool operator==(const CanonicalSavedLane &Other) const {
    return Vgpr == Other.Vgpr && Lane == Other.Lane;
  }
};

struct CanonicalAbiReturn {
  size_t ReturnIndex = 0;
  size_t RestoreLowIndex = 0;
  size_t RestoreHighIndex = 0;
  CanonicalSavedLane SavedLow;
  CanonicalSavedLane SavedHigh;
};

struct AbiVgprModeState {
  int8_t Dst = -2;
  int8_t Src0 = -2;
  int8_t Src1 = -2;
  int8_t Src2 = -2;
};

static AbiVgprModeState abiVgprModeState(unsigned Mode) {
  return {static_cast<int8_t>((Mode >> 6) & 3), static_cast<int8_t>(Mode & 3),
          static_cast<int8_t>((Mode >> 2) & 3),
          static_cast<int8_t>((Mode >> 4) & 3)};
}

static bool isUnreachable(AbiVgprModeState State) { return State.Dst == -2; }

static AbiVgprModeState mergeAbiVgprMode(AbiVgprModeState Old,
                                         AbiVgprModeState Incoming) {
  if (isUnreachable(Old))
    return Incoming;
  if (isUnreachable(Incoming))
    return Old;
  auto Merge = [](int8_t LHS, int8_t RHS) {
    return LHS == RHS ? LHS : int8_t{-1};
  };
  return {Merge(Old.Dst, Incoming.Dst), Merge(Old.Src0, Incoming.Src0),
          Merge(Old.Src1, Incoming.Src1), Merge(Old.Src2, Incoming.Src2)};
}

static bool sameAbiVgprMode(AbiVgprModeState LHS, AbiVgprModeState RHS) {
  return std::tie(LHS.Dst, LHS.Src0, LHS.Src1, LHS.Src2) ==
         std::tie(RHS.Dst, RHS.Src0, RHS.Src1, RHS.Src2);
}

static bool isAbiVgprModeZero(AbiVgprModeState State) {
  return State.Dst == 0 && State.Src0 == 0 && State.Src1 == 0 &&
         State.Src2 == 0;
}

/// Prove full VGPR-MSB mode zero at every call and s30 ABI return accepted by
/// the linked-object fallback. Every defined function/declared entry is seeded
/// with the HSA ABI entry mode; cross-function direct edges and CFG joins are
/// then modeled explicitly so a non-ABI caller cannot be hidden by reseeding
/// its callee.
static bool validateAbiControlTransferModes(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextSize, ArrayRef<uint64_t> AbiRootEntries,
    const DenseMap<size_t, const FiniteSetPcTransfer *> &ExactSetPcs,
    MCRegister AbiLinkPair) {
  DenseMap<uint64_t, unsigned> OffsetToIndex;
  OffsetToIndex.reserve(Decoded.size());
  for (unsigned I = 0; I != Decoded.size(); ++I)
    OffsetToIndex.try_emplace(Decoded[I].Offset, I);

  std::vector<SmallVector<unsigned, 2>> Successors(Decoded.size());
  auto addTarget = [&](SmallVectorImpl<unsigned> &Out, uint64_t Target) {
    if (Target >= TextSize)
      return true;
    DenseMap<uint64_t, unsigned>::const_iterator It =
        OffsetToIndex.find(Target);
    if (It == OffsetToIndex.end())
      return false;
    Out.push_back(It->second);
    return true;
  };
  auto addFallthrough = [&](SmallVectorImpl<unsigned> &Out, unsigned I) {
    if (I + 1 == Decoded.size())
      return true;
    std::optional<uint64_t> End = checkedAddUint64(
        Decoded[I].Offset, Decoded[I].Size, "ABI mode fallthrough");
    return End && (*End >= TextSize || addTarget(Out, *End));
  };

  for (unsigned I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    SmallVectorImpl<unsigned> &Out = Successors[I];
    if (!DI.DecodeSucceeded)
      return false;
    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
        LS.MIA->isReturn(DI.Inst))
      continue;
    if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
      DenseMap<size_t, const FiniteSetPcTransfer *>::const_iterator Exact =
          ExactSetPcs.find(I);
      if (Exact != ExactSetPcs.end() && Exact->second->LocalTargetIndex &&
          !addTarget(Out, Decoded[*Exact->second->LocalTargetIndex].Offset))
        return false;
      continue;
    }
    if (LS.MIA->isCall(DI.Inst) && isTruePcTransfer(DI, LS)) {
      if (!addFallthrough(Out, I))
        return false;
      continue;
    }
    if (LS.MIA->isBranch(DI.Inst) && hasPcRelativeOperand(DI, LS)) {
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
      if (!Target || !addTarget(Out, *Target))
        return false;
      if (LS.MIA->isConditionalBranch(DI.Inst)) {
        if (!addFallthrough(Out, I))
          return false;
      } else if (!LS.MIA->isUnconditionalBranch(DI.Inst)) {
        return false;
      }
      continue;
    }
    if (isTruePcTransfer(DI, LS))
      continue;
    if (!addFallthrough(Out, I))
      return false;
  }

  SmallVector<AbiVgprModeState, 0> ModeBefore(Decoded.size());
  SmallVector<unsigned, 64> Worklist;
  for (uint64_t Root : AbiRootEntries) {
    DenseMap<uint64_t, unsigned>::const_iterator It = OffsetToIndex.find(Root);
    if (It == OffsetToIndex.end())
      return false;
    AbiVgprModeState Seeded =
        mergeAbiVgprMode(ModeBefore[It->second], abiVgprModeState(0));
    if (!sameAbiVgprMode(Seeded, ModeBefore[It->second])) {
      ModeBefore[It->second] = Seeded;
      Worklist.push_back(It->second);
    }
  }
  for (size_t Next = 0; Next != Worklist.size(); ++Next) {
    unsigned I = Worklist[Next];
    AbiVgprModeState ModeOut = ModeBefore[I];
    if (std::optional<unsigned> Mode = getExactAbiVgprMode(Decoded[I], LS))
      ModeOut = abiVgprModeState(*Mode);
    else if (mayWriteAbiVgprBankUnknown(Decoded[I], LS))
      ModeOut = {-1, -1, -1, -1};
    for (unsigned Successor : Successors[I]) {
      AbiVgprModeState Merged =
          mergeAbiVgprMode(ModeBefore[Successor], ModeOut);
      if (!sameAbiVgprMode(Merged, ModeBefore[Successor])) {
        ModeBefore[Successor] = Merged;
        Worklist.push_back(Successor);
      }
    }
  }

  for (unsigned I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    bool IsCall = LS.MIA->isCall(DI.Inst) && isTruePcTransfer(DI, LS);
    bool IsAbiReturn = DI.Inst.getOpcode() == LS.SSetPcI64Opcode &&
                       isExactRegisterOperand(DI.Inst, 0, AbiLinkPair) &&
                       !ExactSetPcs.contains(I);
    if (!IsCall && !IsAbiReturn)
      continue;
    if (isUnreachable(ModeBefore[I])) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "control transfer at 0x"
            << utohexstr(DI.Offset) << " is unreachable from an ABI root\n";
      return false;
    }
    if (!isAbiVgprModeZero(ModeBefore[I])) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "control transfer at 0x"
            << utohexstr(DI.Offset) << " is not in full VGPR-MSB mode 0\n";
      return false;
    }
  }
  return true;
}

/// Prove the compiler's canonical AMDGPU nested-call frame for a group of
/// s[30:31] returns in one local STT_FUNC:
///
///   v_writelane SavedLowVgpr,  s30, SavedLowLane
///   v_writelane SavedHighVgpr, s31, SavedHighLane
///     ... s_swap_pc_i64 s[30:31], Target ...
///   v_readlane s30, SavedLowVgpr,  SavedLowLane
///   v_readlane s31, SavedHighVgpr, SavedHighLane
///   s_set_pc_i64 s[30:31]
///
/// Both saved VGPRs must be in CSR_AMDGPU_VGPRs, so nested C/Fast/Cold callees
/// preserve them (SIRegisterInfo::getCallPreservedMask). The locations may
/// straddle a VGPR boundary when the compiler packs a large SGPR spill into
/// consecutive wave lanes. A function-local CFG fixed point additionally
/// proves every caller write to either saved physical VGPR addresses a
/// different persistent destination bank, writes a different lane, or is the
/// exact prologue save. Calls are permitted only in full VGPR-MSB mode zero,
/// the mode required at both call entry and return.
static bool validateCanonicalAbiFrameReturns(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, const ElfView::FunctionTextRange &Range,
    ArrayRef<size_t> ReturnIndices,
    const DenseMap<size_t, const FiniteSetPcTransfer *> &ExactSetPcs,
    ArrayRef<DirectTargetSource> ExactSetPcEntries,
    ArrayRef<CallContinuationSource> AllCallContinuations,
    const DenseSet<uint64_t> &PotentialEntries,
    ArrayRef<uint64_t> ActualRootEntries, ArrayRef<uint64_t> AbiModeRootEntries,
    const ControlFlowScanIndex &Index, MCRegister AbiLinkPair,
    ArrayRef<MCRegister> NumberedSgprs,
    const DenseSet<size_t> &CanonicalAbiTailTransfers,
    bool ForceFullCanonicalFrame, DenseSet<size_t> &SafeReturns) {
  if (!Range.Symbol || Range.Symbol->getType() != ELF::STT_FUNC ||
      Range.Symbol->getBinding() != ELF::STB_LOCAL || Range.Begin < TextAddr ||
      Range.End <= Range.Begin)
    return false;
  uint64_t BeginOffset = Range.Begin - TextAddr;
  uint64_t EndOffset = Range.End - TextAddr;
  auto reject = [&](const Twine &Reason) {
    log() << "hotswap: canonical s30 frame at 0x" << utohexstr(BeginOffset)
          << " rejected: " << Reason << "\n";
    return false;
  };
  auto BeginIt = llvm::lower_bound(
      Decoded, BeginOffset, [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  auto EndIt =
      std::lower_bound(BeginIt, Decoded.end(), EndOffset,
                       [](const InternalDecodedInst &DI, uint64_t Offset) {
                         return DI.Offset < Offset;
                       });
  if (BeginIt == EndIt || BeginIt->Offset != BeginOffset)
    return reject("function start is not a decoded instruction");
  size_t BeginIndex = static_cast<size_t>(BeginIt - Decoded.begin());
  size_t EndIndex = static_cast<size_t>(EndIt - Decoded.begin());
  if (hasUnprovenFallthroughEntry(Decoded, BeginOffset,
                                  Decoded[ReturnIndices.front()].Offset,
                                  AbiModeRootEntries, Index)) {
    // A compiler-emitted noreturn/trap tail may be laid out immediately
    // before the next local function. It is still a valid linked-call entry
    // when every path through that fallthrough suffix executes one exact ABI
    // call that defines s[30:31], and no root or transfer can bypass the call.
    auto Fallthrough = Index.FallthroughEntries.find(BeginOffset);
    std::optional<size_t> LinkCall;
    if (Fallthrough != Index.FallthroughEntries.end() &&
        Fallthrough->second.Proven) {
      uint64_t ChainBegin = Fallthrough->second.ChainBegin;
      for (size_t I = BeginIndex; I != 0;) {
        --I;
        if (Decoded[I].Offset < ChainBegin)
          break;
        if (!LS.MIA->isCall(Decoded[I].Inst) ||
            !isTruePcTransfer(Decoded[I], LS))
          continue;
        std::optional<MCRegister> Link = getCallReturnRegister(Decoded[I], LS);
        if (Link && *Link == AbiLinkPair)
          LinkCall = I;
        break;
      }
    }
    if (!LinkCall)
      return reject("function entry has an unproven layout fallthrough");
    std::optional<uint64_t> Continuation =
        checkedAddUint64(Decoded[*LinkCall].Offset, Decoded[*LinkCall].Size,
                         "canonical frame prefix call continuation");
    if (!Continuation || *Continuation >= BeginOffset)
      return reject("function entry has an invalid link-defining prefix call");
    ArrayRef<uint64_t>::iterator Root =
        llvm::lower_bound(AbiModeRootEntries, *Continuation);
    if (Root != AbiModeRootEntries.end() && *Root < BeginOffset)
      return reject(Twine("ABI root bypasses prefix link call at 0x") +
                    utohexstr(*Root));
    auto Direct = llvm::lower_bound(
        Index.DirectTargetsByTarget, *Continuation,
        [](const DirectTargetSource &Source, uint64_t Target) {
          return Source.Target < Target;
        });
    if (Direct != Index.DirectTargetsByTarget.end() &&
        Direct->Target < BeginOffset)
      return reject(Twine("control flow bypasses prefix link call at 0x") +
                    utohexstr(Direct->Target));
    auto Exact = llvm::lower_bound(
        ExactSetPcEntries, *Continuation,
        [](const DirectTargetSource &Source, uint64_t Target) {
          return Source.Target < Target;
        });
    if (Exact != ExactSetPcEntries.end() && Exact->Target < BeginOffset)
      return reject(Twine("exact set-PC bypasses prefix link call at 0x") +
                    utohexstr(Exact->Target));
    auto OtherCall = llvm::lower_bound(
        AllCallContinuations, *Continuation,
        [](const CallContinuationSource &Source, uint64_t Target) {
          return Source.Continuation < Target;
        });
    for (; OtherCall != AllCallContinuations.end() &&
           OtherCall->Continuation < BeginOffset;
         ++OtherCall)
      if (OtherCall->InstIndex != *LinkCall)
        return reject(Twine("foreign call continuation bypasses prefix link "
                            "call at 0x") +
                      utohexstr(OtherCall->Continuation));
    for (size_t I = *LinkCall + 1; I != BeginIndex; ++I)
      if (definesOverlappingRegister(Decoded[I], LS, NumberedSgprs[30]) ||
          definesOverlappingRegister(Decoded[I], LS, NumberedSgprs[31]) ||
          (isTruePcTransfer(Decoded[I], LS) && LS.MIA->isCall(Decoded[I].Inst)))
        return reject(Twine("prefix link is clobbered at 0x") +
                      utohexstr(Decoded[I].Offset));
  }
  ArrayRef<uint64_t>::iterator FunctionRoot =
      llvm::upper_bound(AbiModeRootEntries, BeginOffset);
  if (FunctionRoot != AbiModeRootEntries.end() && *FunctionRoot < EndOffset)
    return reject(Twine("ABI function root enters at 0x") +
                  utohexstr(*FunctionRoot));
  ArrayRef<uint64_t>::iterator ActualRoot =
      llvm::lower_bound(ActualRootEntries, BeginOffset);
  if (ActualRoot != ActualRootEntries.end() && *ActualRoot < EndOffset)
    return reject(Twine("declared/external root enters at 0x") +
                  utohexstr(*ActualRoot));
  SmallVector<size_t, 4> BeginReentrySources;
  auto Direct =
      llvm::lower_bound(Index.DirectTargetsByTarget, BeginOffset,
                        [](const DirectTargetSource &Source, uint64_t Target) {
                          return Source.Target < Target;
                        });
  for (; Direct != Index.DirectTargetsByTarget.end() &&
         Direct->Target < EndOffset;
       ++Direct) {
    const DirectTargetSource &Source = *Direct;
    uint64_t SourceOffset = Decoded[Source.InstIndex].Offset;
    bool SourceInside = SourceOffset >= BeginOffset && SourceOffset < EndOffset;
    if (SourceInside) {
      if (Source.Target == BeginOffset)
        BeginReentrySources.push_back(Source.InstIndex);
      continue;
    }
    if (Source.Target == BeginOffset) {
      if (CanonicalAbiTailTransfers.contains(Source.InstIndex))
        continue;
      std::optional<MCRegister> Link =
          getCallReturnRegister(Decoded[Source.InstIndex], LS);
      if (!Link || *Link != AbiLinkPair)
        return reject(Twine("non-ABI control transfer enters function at 0x") +
                      utohexstr(SourceOffset));
      continue;
    }
    return reject(Twine("external direct transfer enters at 0x") +
                  utohexstr(Source.Target));
  }
  auto Exact =
      llvm::lower_bound(ExactSetPcEntries, BeginOffset,
                        [](const DirectTargetSource &Source, uint64_t Target) {
                          return Source.Target < Target;
                        });
  for (; Exact != ExactSetPcEntries.end() && Exact->Target < EndOffset;
       ++Exact) {
    uint64_t Target = Exact->Target;
    uint64_t Source = Decoded[Exact->InstIndex].Offset;
    bool SourceInside = Source >= BeginOffset && Source < EndOffset;
    // A separately validated canonical tail thunk may enter exactly at Begin:
    // its source frame proves s[30:31] was restored to the incoming link
    // before this exact set-PC. Interior entries and all other external exact
    // transfers remain forbidden.
    if (Target == BeginOffset && !SourceInside &&
        CanonicalAbiTailTransfers.contains(Exact->InstIndex))
      continue;
    if (Target == BeginOffset || (Target > BeginOffset && !SourceInside))
      return reject(Twine("external exact set-PC enters at 0x") +
                    utohexstr(Target));
  }
  auto Continuation = llvm::lower_bound(
      AllCallContinuations, BeginOffset,
      [](const CallContinuationSource &Source, uint64_t Target) {
        return Source.Continuation < Target;
      });
  for (; Continuation != AllCallContinuations.end() &&
         Continuation->Continuation < EndOffset;
       ++Continuation)
    if (Decoded[Continuation->InstIndex].Offset < BeginOffset ||
        Decoded[Continuation->InstIndex].Offset >= EndOffset)
      return reject(Twine("external call continuation enters at 0x") +
                    utohexstr(Continuation->Continuation));

  bool HasNestedCall = false;
  for (size_t I = BeginIndex; I != EndIndex; ++I)
    HasNestedCall |=
        LS.MIA->isCall(Decoded[I].Inst) && isTruePcTransfer(Decoded[I], LS);
  if (!ForceFullCanonicalFrame && !HasNestedCall) {
    bool DefinesLink = false;
    for (size_t I = BeginIndex; I != EndIndex; ++I) {
      const InternalDecodedInst &DI = Decoded[I];
      DefinesLink |= definesOverlappingRegister(DI, LS, NumberedSgprs[30]) ||
                     definesOverlappingRegister(DI, LS, NumberedSgprs[31]);
    }
    if (!DefinesLink) {
      for (size_t Return : ReturnIndices)
        SafeReturns.insert(Return);
      return true;
    }
  }

  SmallVector<CanonicalAbiReturn, 4> Returns;
  std::optional<CanonicalSavedLane> SavedLow;
  std::optional<CanonicalSavedLane> SavedHigh;
  DenseSet<size_t> RestoreInstructions;
  DenseSet<size_t> PostRestoreTeardownInstructions;
  for (size_t ReturnIndex : ReturnIndices) {
    std::optional<size_t> Low;
    std::optional<size_t> High;
    std::optional<CanonicalSavedLane> CandidateLow;
    std::optional<CanonicalSavedLane> CandidateHigh;
    for (size_t I = ReturnIndex; I != BeginIndex;) {
      --I;
      const InternalDecodedInst &DI = Decoded[I];
      if (!DI.DecodeSucceeded)
        return reject("undecoded restore instruction");
      if (!Low && definesOverlappingRegister(DI, LS, NumberedSgprs[30])) {
        if (DI.Mnemonic != "v_readlane_b32" || DI.Inst.getNumOperands() != 3 ||
            !isExactRegisterOperand(DI.Inst, 0, NumberedSgprs[30]) ||
            !DI.Inst.getOperand(1).isReg() || !DI.Inst.getOperand(1).getReg() ||
            !DI.Inst.getOperand(2).isImm() ||
            DI.Inst.getOperand(2).getImm() < 0 ||
            DI.Inst.getOperand(2).getImm() >= 32)
          return reject(Twine("invalid s30 restore before return 0x") +
                        utohexstr(Decoded[ReturnIndex].Offset));
        Low = I;
        CandidateLow =
            CanonicalSavedLane{MCRegister(DI.Inst.getOperand(1).getReg()),
                               DI.Inst.getOperand(2).getImm()};
      }
      if (!High && definesOverlappingRegister(DI, LS, NumberedSgprs[31])) {
        if (DI.Mnemonic != "v_readlane_b32" || DI.Inst.getNumOperands() != 3 ||
            !isExactRegisterOperand(DI.Inst, 0, NumberedSgprs[31]) ||
            !DI.Inst.getOperand(1).isReg() || !DI.Inst.getOperand(1).getReg() ||
            !DI.Inst.getOperand(2).isImm() ||
            DI.Inst.getOperand(2).getImm() < 0 ||
            DI.Inst.getOperand(2).getImm() >= 32)
          return reject(Twine("invalid s31 restore before return 0x") +
                        utohexstr(Decoded[ReturnIndex].Offset));
        High = I;
        CandidateHigh =
            CanonicalSavedLane{MCRegister(DI.Inst.getOperand(1).getReg()),
                               DI.Inst.getOperand(2).getImm()};
      }
      if (Low && High)
        break;
      if (PotentialEntries.contains(DI.Offset))
        return reject(Twine("entry at 0x") + utohexstr(DI.Offset) +
                      " interrupts the restore search");
      if (isTruePcTransfer(DI, LS))
        return reject(Twine("control flow at 0x") + utohexstr(DI.Offset) +
                      " interrupts the restore search (" + DI.Mnemonic + ")");
    }
    std::optional<unsigned> SavedLowIndex =
        CandidateLow ? numberedVgprIndex(*LS.MRI, CandidateLow->Vgpr)
                     : std::nullopt;
    std::optional<unsigned> SavedHighIndex =
        CandidateHigh ? numberedVgprIndex(*LS.MRI, CandidateHigh->Vgpr)
                      : std::nullopt;
    if (!Low || !High || !CandidateLow || !CandidateHigh || !SavedLowIndex ||
        !SavedHighIndex || !isAbiCalleeSavedVgpr(*SavedLowIndex) ||
        !isAbiCalleeSavedVgpr(*SavedHighIndex) ||
        *CandidateLow == *CandidateHigh ||
        (SavedLow && !(*SavedLow == *CandidateLow)) ||
        (SavedHigh && !(*SavedHigh == *CandidateHigh)))
      return reject(Twine("return 0x") +
                    utohexstr(Decoded[ReturnIndex].Offset) +
                    " has no consistent callee-saved VGPR-lane restore");
    SavedLow = *CandidateLow;
    SavedHigh = *CandidateHigh;
    size_t RestoreBegin = std::min(*Low, *High);
    size_t RestoreEnd = std::max(*Low, *High);
    for (size_t I = RestoreBegin + 1; I <= ReturnIndex; ++I) {
      if (PotentialEntries.contains(Decoded[I].Offset) ||
          (I != ReturnIndex && isTruePcTransfer(Decoded[I], LS)))
        return reject(Twine("entry or control flow interrupts epilogue at 0x") +
                      utohexstr(Decoded[I].Offset));
      if (I > RestoreEnd && I < ReturnIndex)
        PostRestoreTeardownInstructions.insert(I);
    }
    Returns.push_back(
        {ReturnIndex, *Low, *High, *CandidateLow, *CandidateHigh});
    RestoreInstructions.insert(*Low);
    RestoreInstructions.insert(*High);
  }
  if (!SavedLow || !SavedHigh)
    return reject("no saved VGPR lanes recovered");

  std::optional<size_t> SaveLow;
  std::optional<size_t> SaveHigh;
  for (size_t I = BeginIndex; I != EndIndex; ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.Offset - BeginOffset > 512)
      break;
    if (matchesCanonicalLaneTransfer(DI, "v_writelane_b32", SavedLow->Vgpr,
                                     NumberedSgprs[30], SavedLow->Lane))
      SaveLow = I;
    if (matchesCanonicalLaneTransfer(DI, "v_writelane_b32", SavedHigh->Vgpr,
                                     NumberedSgprs[31], SavedHigh->Lane))
      SaveHigh = I;
    if (SaveLow && SaveHigh)
      break;
    if (isTruePcTransfer(DI, LS))
      return reject(Twine("control flow at 0x") + utohexstr(DI.Offset) +
                    " precedes the prologue saves (" + DI.Mnemonic + ")");
  }
  if (!SaveLow || !SaveHigh)
    return reject("matching prologue saves are absent");
  size_t SaveEnd = std::max(*SaveLow, *SaveHigh);
  for (size_t I = BeginIndex + 1; I <= SaveEnd; ++I)
    if (PotentialEntries.contains(Decoded[I].Offset))
      return reject(Twine("entry bypasses a prologue save at 0x") +
                    utohexstr(Decoded[I].Offset));

  const size_t Count = EndIndex - BeginIndex;
  DenseMap<uint64_t, unsigned> OffsetToLocal;
  OffsetToLocal.reserve(Count);
  for (unsigned I = 0; I != Count; ++I)
    OffsetToLocal.try_emplace(Decoded[BeginIndex + I].Offset, I);
  std::vector<SmallVector<unsigned, 2>> Successors(Count);
  bool Valid = true;
  auto addTarget = [&](SmallVectorImpl<unsigned> &Out, uint64_t Target) {
    if (Target < BeginOffset || Target >= EndOffset)
      return;
    DenseMap<uint64_t, unsigned>::const_iterator It =
        OffsetToLocal.find(Target);
    if (It == OffsetToLocal.end()) {
      Valid = false;
      return;
    }
    Out.push_back(It->second);
  };
  auto addFallthrough = [&](SmallVectorImpl<unsigned> &Out, unsigned I) {
    if (I + 1 != Count)
      Out.push_back(I + 1);
  };
  for (unsigned Local = 0; Local != Count && Valid; ++Local) {
    size_t Global = BeginIndex + Local;
    const InternalDecodedInst &DI = Decoded[Global];
    SmallVectorImpl<unsigned> &Out = Successors[Local];
    if (!DI.DecodeSucceeded) {
      Valid = false;
      break;
    }
    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode)
      continue;
    if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
      DenseMap<size_t, const FiniteSetPcTransfer *>::const_iterator Exact =
          ExactSetPcs.find(Global);
      if (Exact != ExactSetPcs.end() && Exact->second->LocalTargetIndex)
        addTarget(Out, Decoded[*Exact->second->LocalTargetIndex].Offset);
      else if (!llvm::is_contained(ReturnIndices, Global))
        Valid = false;
      continue;
    }
    if (LS.MIA->isReturn(DI.Inst))
      continue;
    if (LS.MIA->isCall(DI.Inst) && isTruePcTransfer(DI, LS)) {
      addFallthrough(Out, Local);
      continue;
    }
    if (LS.MIA->isBranch(DI.Inst) && hasPcRelativeOperand(DI, LS)) {
      if (LS.MIA->isIndirectBranch(DI.Inst)) {
        Valid = false;
        break;
      }
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
      if (!Target) {
        Valid = false;
        break;
      }
      addTarget(Out, *Target);
      if (LS.MIA->isConditionalBranch(DI.Inst))
        addFallthrough(Out, Local);
      else if (!LS.MIA->isUnconditionalBranch(DI.Inst))
        Valid = false;
      continue;
    }
    if (isTruePcTransfer(DI, LS)) {
      Valid = false;
      break;
    }
    addFallthrough(Out, Local);
  }
  if (!Valid)
    return reject("local CFG is open");

  // A control-flow edge back to the function entry may execute the prologue
  // again. It is safe only while s[30:31] still contains the function's
  // original incoming link. Compute that as a must fact: every path reaching
  // the edge source must be free of calls and s30/s31 definitions. Merging by
  // AND is important for cycles where a lexically early backedge can be
  // revisited from after a nested call.
  SmallVector<int8_t, 0> IncomingLinkIntactBefore(Count, int8_t{-1});
  IncomingLinkIntactBefore[0] = 1;
  SmallVector<unsigned, 64> LinkWorklist(1, 0);
  for (size_t Next = 0; Next != LinkWorklist.size(); ++Next) {
    unsigned Local = LinkWorklist[Next];
    size_t Global = BeginIndex + Local;
    const InternalDecodedInst &DI = Decoded[Global];
    int8_t IntactOut = IncomingLinkIntactBefore[Local];
    if ((LS.MIA->isCall(DI.Inst) && isTruePcTransfer(DI, LS)) ||
        definesOverlappingRegister(DI, LS, NumberedSgprs[30]) ||
        definesOverlappingRegister(DI, LS, NumberedSgprs[31]))
      IntactOut = 0;
    for (unsigned Succ : Successors[Local]) {
      int8_t Merged = IncomingLinkIntactBefore[Succ] == -1
                          ? IntactOut
                          : (IncomingLinkIntactBefore[Succ] & IntactOut);
      if (Merged != IncomingLinkIntactBefore[Succ]) {
        IncomingLinkIntactBefore[Succ] = Merged;
        LinkWorklist.push_back(Succ);
      }
    }
  }
  for (size_t Source : BeginReentrySources) {
    unsigned Local = static_cast<unsigned>(Source - BeginIndex);
    if (IncomingLinkIntactBefore[Local] == 0)
      return reject(Twine("internal transfer reenters the prologue with a "
                          "clobbered s30 link at 0x") +
                    utohexstr(Decoded[Source].Offset));
  }

  SmallVector<AbiVgprModeState, 0> ModeBefore(Count);
  ModeBefore[0] = abiVgprModeState(0);
  SmallVector<int8_t, 0> SavesBefore(Count, int8_t{-1});
  SavesBefore[0] = 0;
  SmallVector<unsigned, 64> Worklist(1, 0);
  for (size_t Next = 0; Next != Worklist.size(); ++Next) {
    unsigned Local = Worklist[Next];
    size_t Global = BeginIndex + Local;
    const InternalDecodedInst &DI = Decoded[BeginIndex + Local];
    AbiVgprModeState ModeOut = ModeBefore[Local];
    if (std::optional<unsigned> Mode = getExactAbiVgprMode(DI, LS))
      ModeOut = abiVgprModeState(*Mode);
    else if (mayWriteAbiVgprBankUnknown(DI, LS))
      ModeOut = {-1, -1, -1, -1};
    int8_t SavesOut = SavesBefore[Local];
    if (Global == *SaveLow)
      SavesOut |= 1;
    if (Global == *SaveHigh)
      SavesOut |= 2;
    for (unsigned Succ : Successors[Local]) {
      AbiVgprModeState MergedMode = mergeAbiVgprMode(ModeBefore[Succ], ModeOut);
      int8_t MergedSaves =
          SavesBefore[Succ] == -1 ? SavesOut : (SavesBefore[Succ] & SavesOut);
      if (!sameAbiVgprMode(MergedMode, ModeBefore[Succ]) ||
          MergedSaves != SavesBefore[Succ]) {
        ModeBefore[Succ] = MergedMode;
        SavesBefore[Succ] = MergedSaves;
        Worklist.push_back(Succ);
      }
    }
  }

  for (unsigned Local = 0; Local != Count; ++Local) {
    if (isUnreachable(ModeBefore[Local]))
      continue;
    size_t Global = BeginIndex + Local;
    const InternalDecodedInst &DI = Decoded[Global];
    if (LS.MIA->isCall(DI.Inst) && !isAbiVgprModeZero(ModeBefore[Local])) {
      log() << "hotswap: canonical s30 frame: call at 0x"
            << utohexstr(DI.Offset) << " is not in full VGPR-MSB mode 0\n";
      return false;
    }
    if (llvm::is_contained(ReturnIndices, Global) && SavesBefore[Local] != 3) {
      log() << "hotswap: canonical s30 frame: prologue saves do not dominate "
               "return at 0x"
            << utohexstr(DI.Offset) << "\n";
      return false;
    }
    if (llvm::is_contained(ReturnIndices, Global) &&
        !isAbiVgprModeZero(ModeBefore[Local])) {
      log() << "hotswap: canonical s30 frame: return at 0x"
            << utohexstr(DI.Offset) << " is not in full VGPR-MSB mode 0\n";
      return false;
    }
    if (RestoreInstructions.contains(Global) && ModeBefore[Local].Src0 != 0) {
      log() << "hotswap: canonical s30 frame: readlane at 0x"
            << utohexstr(DI.Offset) << " does not read VGPR bank 0\n";
      return false;
    }

    bool DefinesLink = definesOverlappingRegister(DI, LS, NumberedSgprs[30]) ||
                       definesOverlappingRegister(DI, LS, NumberedSgprs[31]);
    if (DefinesLink && !RestoreInstructions.contains(Global) &&
        SavesBefore[Local] != 3) {
      log() << "hotswap: canonical s30 frame: link definition before both "
               "prologue saves at 0x"
            << utohexstr(DI.Offset) << "\n";
      return false;
    }

    bool WritesSavedLow = instructionWritesRegister(DI, LS, SavedLow->Vgpr);
    bool WritesSavedHigh = instructionWritesRegister(DI, LS, SavedHigh->Vgpr);
    if (!WritesSavedLow && !WritesSavedHigh)
      continue;
    // The canonical epilogue may restore the caller's old callee-saved VGPR
    // from scratch after both link halves have already been copied into
    // s[30:31]. The entry/control-flow checks above make this a straight-line
    // teardown interval, so that write cannot affect the return address.
    if (PostRestoreTeardownInstructions.contains(Global))
      continue;
    if (Global == *SaveLow || Global == *SaveHigh) {
      if (ModeBefore[Local].Dst != 0) {
        log() << "hotswap: canonical s30 frame: writelane save at 0x"
              << utohexstr(DI.Offset) << " does not write VGPR bank 0\n";
        return false;
      }
      continue;
    }
    if (ModeBefore[Local].Dst > 0)
      continue;
    if (ModeBefore[Local].Dst < 0) {
      log() << "hotswap: canonical s30 frame: saved-VGPR write at 0x"
            << utohexstr(DI.Offset) << " has unknown destination bank\n";
      return false;
    }
    if (DI.Mnemonic == "v_writelane_b32" && DI.Inst.getNumOperands() == 4 &&
        DI.Inst.getOperand(0).isReg() && DI.Inst.getOperand(0).getReg() &&
        isExactRegisterOperand(DI.Inst, 3,
                               MCRegister(DI.Inst.getOperand(0).getReg())) &&
        DI.Inst.getOperand(2).isImm()) {
      CanonicalSavedLane Written{MCRegister(DI.Inst.getOperand(0).getReg()),
                                 DI.Inst.getOperand(2).getImm()};
      if (!(Written == *SavedLow) && !(Written == *SavedHigh))
        continue;
    }
    log() << "hotswap: canonical s30 frame: saved lower-bank VGPR clobber at "
             "0x"
          << utohexstr(DI.Offset) << "\n";
    return false;
  }
  for (const CanonicalAbiReturn &Return : Returns)
    SafeReturns.insert(Return.ReturnIndex);
  return true;
}

/// Validate the linked-code-object ABI entry set as a late, all-or-nothing
/// fallback after the ordinary machine-level closed-world proof remains open.
///
/// A valid compiler-produced, linked AMDGPU HSA code object may contain calls
/// whose selector is intentionally opaque to static analysis. This fallback
/// relies on the linked-code ABI contract that every such selector denotes a
/// defined STT_FUNC entry and that every callee obeys the AMDGPU calling
/// convention, including preservation of CSR_AMDGPU_VGPRs. It is not a proof
/// for arbitrary or adversarial machine code. Under that explicit contract,
/// the ABI gives those transfers a finite local entry set when all of the
/// following object-wide invariants hold:
///   * every opaque register call is exactly
///       s_swap_pc_i64 s[30:31], TargetPair;
///   * every s_set_pc_i64 s[30:31] is a return to one of the syntactic call
///     continuations;
///   * every other set-PC is an exact local PC materialization;
///   * no other indirect transfer exists; and
///   * every function start, continuation, and exact jump target is a decoded
///     instruction boundary outside the interior of a PC materialization.
///
/// This is deliberately not a replacement for auditFiniteIndirectControlFlow:
/// callers invoke it only when that stronger proof did not close. Any opcode,
/// register, symbol, boundary, or entry-set variation rejects the whole
/// fallback and preserves the existing fail-closed result.
static std::optional<WellFormedAbiEntrySet> validateWellFormedAbiEntrySet(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<uint64_t> DeclaredEntries, ArrayRef<uint64_t> ExternalEntries,
    ArrayRef<uint64_t> NonCallEntries, const ControlFlowScanIndex &Index,
    ArrayRef<FiniteSetPcTransfer> SetPcCandidates, const ElfView &Elf) {
  const auto Header = Elf.file().getHeader();
  std::optional<uint64_t> TextEnd =
      checkedAddUint64(TextAddr, TextSize, "well-formed ABI text end");
  if (Header.e_type != ELF::ET_DYN || Header.e_machine != ELF::EM_AMDGPU ||
      Header.e_ident[ELF::EI_OSABI] != ELF::ELFOSABI_AMDGPU_HSA ||
      TextAddr != Elf.textAddr() || TextSize != Elf.textSize() || !TextEnd ||
      Decoded.empty() || !Elf.functionTextRangesComplete()) {
    log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
             "object preconditions do not hold\n";
    return std::nullopt;
  }

  DenseMap<uint64_t, size_t> InstructionAt;
  InstructionAt.reserve(Decoded.size());
  uint64_t ExpectedOffset = 0;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (!DI.DecodeSucceeded || DI.Size == 0 || DI.Offset != ExpectedOffset ||
        !InstructionAt.try_emplace(DI.Offset, I).second) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "non-contiguous decode at 0x"
            << utohexstr(DI.Offset) << "\n";
      return std::nullopt;
    }
    std::optional<uint64_t> End =
        checkedAddUint64(DI.Offset, DI.Size, "well-formed ABI instruction end");
    if (!End || *End > TextSize) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "instruction end exceeds .text at 0x"
            << utohexstr(DI.Offset) << "\n";
      return std::nullopt;
    }
    ExpectedOffset = *End;
  }
  if (ExpectedOffset != TextSize) {
    log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
             "decoded instructions do not cover .text\n";
    return std::nullopt;
  }

  WellFormedAbiEntrySet Result;
  auto addBoundary = [&](uint64_t Offset) {
    if (Offset >= TextSize || !InstructionAt.contains(Offset))
      return false;
    Result.Targets.insert(Offset);
    return true;
  };

  bool SawFunctionEntry = false;
  SmallVector<uint64_t, 32> AbiRootEntries;
  DenseSet<uint64_t> DefinedFunctionBeginOffsets;
  for (const ElfView::FunctionTextRange &Range : FunctionRanges) {
    // Internal users can supply synthetic ranges to the stronger machine
    // audit. Only actual defined STT_FUNC symbols expand this ABI fallback's
    // function-pointer target set.
    if (!Range.Symbol || Range.Symbol->getType() != ELF::STT_FUNC)
      continue;
    bool SizedRangeValid =
        Range.Symbol->st_size == 0 ||
        (Range.Symbol->st_value <=
             std::numeric_limits<uint64_t>::max() - Range.Symbol->st_size &&
         Range.Symbol->st_value + Range.Symbol->st_size == Range.End &&
         Range.End <= *TextEnd);
    if (Range.Symbol->st_shndx != Elf.textSectionIndex() ||
        Range.Symbol->st_value != Range.Begin || !SizedRangeValid ||
        Range.Begin < TextAddr || Range.Begin >= *TextEnd ||
        Range.End <= Range.Begin || Range.End > *TextEnd ||
        !addBoundary(Range.Begin - TextAddr) ||
        (Range.End != *TextEnd &&
         !InstructionAt.contains(Range.End - TextAddr))) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "STT_FUNC range is not bounded by exact .text instructions\n";
      return std::nullopt;
    }
    SawFunctionEntry = true;
    uint64_t BeginOffset = Range.Begin - TextAddr;
    AbiRootEntries.push_back(BeginOffset);
    DefinedFunctionBeginOffsets.insert(BeginOffset);
  }
  if (!SawFunctionEntry) {
    log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
             "no defined STT_FUNC entry\n";
    return std::nullopt;
  }
  for (uint64_t Entry : DeclaredEntries)
    if (!addBoundary(Entry)) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "declared entry 0x"
            << utohexstr(Entry) << " is not an instruction boundary\n";
      return std::nullopt;
    }
  AbiRootEntries.append(DeclaredEntries.begin(), DeclaredEntries.end());
  for (uint64_t Entry : ExternalEntries)
    if (!addBoundary(Entry)) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "external entry 0x"
            << utohexstr(Entry) << " is not an instruction boundary\n";
      return std::nullopt;
    }
  AbiRootEntries.append(ExternalEntries.begin(), ExternalEntries.end());
  llvm::sort(AbiRootEntries);
  AbiRootEntries.erase(
      std::unique(AbiRootEntries.begin(), AbiRootEntries.end()),
      AbiRootEntries.end());
  SmallVector<uint64_t, 32> ActualRootEntries(NonCallEntries.begin(),
                                              NonCallEntries.end());
  llvm::sort(ActualRootEntries);
  ActualRootEntries.erase(
      std::unique(ActualRootEntries.begin(), ActualRootEntries.end()),
      ActualRootEntries.end());

  DenseSet<size_t> FiniteCalls;
  for (const KnownCallSite &Call : Index.Calls)
    FiniteCalls.insert(Call.InstIndex);
  for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations)
    FiniteCalls.insert(Call.InstIndex);

  std::optional<SmallVector<MCRegister, 128>> NumberedSgprs =
      resolveNumberedSgprRegisters(*LS.MRI, Gfx1250MaxSgprs);
  if (!NumberedSgprs || NumberedSgprs->size() <= 31) {
    log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
             "numbered SGPR map is unavailable\n";
    return std::nullopt;
  }
  MCRegister AbiLinkPair;
  for (const InternalDecodedInst &DI : Decoded)
    if (DI.DecodeSucceeded && DI.Inst.getOpcode() == LS.SSwapPcI64Opcode &&
        DI.Inst.getNumOperands() >= 1 && DI.Inst.getOperand(0).isReg()) {
      std::optional<unsigned> Index =
          numberedSgprPairLowIndex(*LS.MRI, DI.Inst.getOperand(0).getReg());
      if (Index && *Index == 30) {
        AbiLinkPair = DI.Inst.getOperand(0).getReg();
        break;
      }
    }
  if (!AbiLinkPair) {
    log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
             "no s30 ABI link pair\n";
    return std::nullopt;
  }

  DenseMap<size_t, const FiniteSetPcTransfer *> ExactSetPcs;
  for (const FiniteSetPcTransfer &Candidate : SetPcCandidates) {
    auto Inserted = ExactSetPcs.try_emplace(Candidate.InstIndex, &Candidate);
    if (!Inserted.second)
      return std::nullopt;
  }
  if (!validateAbiControlTransferModes(Decoded, LS, TextSize, AbiRootEntries,
                                       ExactSetPcs, AbiLinkPair))
    return std::nullopt;

  DenseSet<uint64_t> PotentialEntries(Result.Targets.begin(),
                                      Result.Targets.end());
  for (const DirectTargetSource &Source : Index.DirectTargetsByTarget)
    if (Source.Target < TextSize)
      PotentialEntries.insert(Source.Target);
  SmallVector<CallContinuationSource, 16> AllCallContinuations;
  for (size_t I = 0; I != Decoded.size(); ++I)
    if (LS.MIA->isCall(Decoded[I].Inst)) {
      std::optional<uint64_t> Continuation =
          checkedAddUint64(Decoded[I].Offset, Decoded[I].Size,
                           "canonical ABI call continuation");
      if (!Continuation || !addBoundary(*Continuation))
        return std::nullopt;
      PotentialEntries.insert(*Continuation);
      AllCallContinuations.push_back({I, *Continuation});
    }
  llvm::sort(AllCallContinuations, [](const CallContinuationSource &LHS,
                                      const CallContinuationSource &RHS) {
    return std::tie(LHS.Continuation, LHS.InstIndex) <
           std::tie(RHS.Continuation, RHS.InstIndex);
  });
  SmallVector<DirectTargetSource, 16> ExactSetPcEntries;
  for (const auto &Entry : ExactSetPcs)
    if (Entry.second->LocalTargetIndex) {
      uint64_t Target = Decoded[*Entry.second->LocalTargetIndex].Offset;
      if (!addBoundary(Target))
        return std::nullopt;
      PotentialEntries.insert(Target);
      ExactSetPcEntries.push_back({Entry.first, Target});
    }
  llvm::sort(ExactSetPcEntries,
             [](const DirectTargetSource &LHS, const DirectTargetSource &RHS) {
               return std::tie(LHS.Target, LHS.InstIndex) <
                      std::tie(RHS.Target, RHS.InstIndex);
             });

  DenseMap<std::pair<uint64_t, uint64_t>, SmallVector<size_t, 4>>
      ReturnsByFunction;
  DenseMap<std::pair<uint64_t, uint64_t>, const ElfView::FunctionTextRange *>
      ReturnFunctionRanges;
  SmallVector<const ElfView::FunctionTextRange *, 32> LocalFunctionRanges;
  for (const ElfView::FunctionTextRange &Range : FunctionRanges)
    if (Range.Symbol && Range.Symbol->getType() == ELF::STT_FUNC &&
        Range.Symbol->getBinding() == ELF::STB_LOCAL)
      LocalFunctionRanges.push_back(&Range);
  llvm::sort(LocalFunctionRanges, [](const ElfView::FunctionTextRange *LHS,
                                     const ElfView::FunctionTextRange *RHS) {
    if (LHS->Begin != RHS->Begin)
      return LHS->Begin < RHS->Begin;
    return LHS->End > RHS->End;
  });
  size_t RangeTreeBase = 1;
  while (RangeTreeBase < LocalFunctionRanges.size())
    RangeTreeBase *= 2;
  SmallVector<uint64_t, 0> MaxRangeEnd(2 * RangeTreeBase);
  for (size_t I = 0; I != LocalFunctionRanges.size(); ++I)
    MaxRangeEnd[RangeTreeBase + I] = LocalFunctionRanges[I]->End;
  for (size_t I = RangeTreeBase; I-- != 1;)
    MaxRangeEnd[I] = std::max(MaxRangeEnd[2 * I], MaxRangeEnd[2 * I + 1]);
  auto FindRightmostContaining =
      [&](auto &&Self, size_t Node, size_t Begin, size_t End, size_t Limit,
          uint64_t Address) -> std::optional<size_t> {
    if (Begin >= Limit || MaxRangeEnd[Node] <= Address)
      return std::nullopt;
    if (End - Begin == 1)
      return Begin;
    size_t Middle = Begin + (End - Begin) / 2;
    if (std::optional<size_t> Right =
            Self(Self, 2 * Node + 1, Middle, End, Limit, Address))
      return Right;
    return Self(Self, 2 * Node, Begin, Middle, Limit, Address);
  };
  auto FindLocalFunction = [&](uint64_t Address) {
    auto Candidate = llvm::upper_bound(
        LocalFunctionRanges, Address,
        [](uint64_t Value, const ElfView::FunctionTextRange *Range) {
          return Value < Range->Begin;
        });
    size_t Limit = static_cast<size_t>(Candidate - LocalFunctionRanges.begin());
    std::optional<size_t> BestIndex = FindRightmostContaining(
        FindRightmostContaining, 1, 0, RangeTreeBase, Limit, Address);
    return BestIndex ? LocalFunctionRanges[*BestIndex] : nullptr;
  };
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.Inst.getOpcode() != LS.SSetPcI64Opcode || ExactSetPcs.contains(I) ||
        DI.Inst.getNumOperands() != 1 ||
        !isExactRegisterOperand(DI.Inst, 0, AbiLinkPair))
      continue;
    uint64_t Address = TextAddr + DI.Offset;
    const ElfView::FunctionTextRange *Best = FindLocalFunction(Address);
    if (!Best)
      continue;
    std::pair<uint64_t, uint64_t> Key{Best->Begin, Best->End};
    ReturnsByFunction[Key].push_back(I);
    ReturnFunctionRanges.try_emplace(Key, Best);
  }

  // A compiler tail thunk restores its incoming s30 link and then performs an
  // exact set-PC to another defined function entry. Validate the source thunk
  // with the same canonical save/restore proof used for ordinary s30 returns;
  // only then may the target function accept that otherwise-external entry.
  DenseMap<std::pair<uint64_t, uint64_t>, SmallVector<size_t, 2>>
      TailExitsByFunction;
  DenseMap<std::pair<uint64_t, uint64_t>, const ElfView::FunctionTextRange *>
      TailFunctionRanges;
  DenseSet<uint64_t> ReturningFunctionBegins;
  for (const auto &Entry : ReturnsByFunction)
    ReturningFunctionBegins.insert(Entry.first.first - TextAddr);
  for (const auto &Exact : ExactSetPcs) {
    size_t I = Exact.first;
    if (!Exact.second->LocalTargetIndex)
      continue;
    uint64_t Target = Decoded[*Exact.second->LocalTargetIndex].Offset;
    if (!DefinedFunctionBeginOffsets.contains(Target) ||
        !ReturningFunctionBegins.contains(Target))
      continue;
    const ElfView::FunctionTextRange *Source =
        FindLocalFunction(TextAddr + Decoded[I].Offset);
    if (!Source || Source->Begin - TextAddr == Target)
      continue;
    std::pair<uint64_t, uint64_t> Key{Source->Begin, Source->End};
    TailExitsByFunction[Key].push_back(I);
    TailFunctionRanges.try_emplace(Key, Source);
  }

  // Phase 1 proves tail sources under the original strict entry policy. This
  // deliberately rejects tail chains: no candidate is authorized merely by
  // being present in the candidate set. A source function's ordinary s30
  // returns are included so its complete local CFG remains closed.
  const DenseSet<size_t> NoSafeTailTransfers;
  DenseSet<size_t> SafeAbiTailTransfers;
  for (const auto &Entry : TailExitsByFunction) {
    const ElfView::FunctionTextRange *Range =
        TailFunctionRanges.lookup(Entry.first);
    SmallVector<size_t, 4> SourceExits(Entry.second.begin(),
                                       Entry.second.end());
    auto Returns = ReturnsByFunction.find(Entry.first);
    if (Returns != ReturnsByFunction.end())
      SourceExits.append(Returns->second.begin(), Returns->second.end());
    llvm::sort(SourceExits);
    SourceExits.erase(std::unique(SourceExits.begin(), SourceExits.end()),
                      SourceExits.end());
    DenseSet<size_t> ValidatedSourceExits;
    if (!Range || !validateCanonicalAbiFrameReturns(
                      Decoded, LS, TextAddr, *Range, SourceExits, ExactSetPcs,
                      ExactSetPcEntries, AllCallContinuations, PotentialEntries,
                      ActualRootEntries, AbiRootEntries, Index, AbiLinkPair,
                      *NumberedSgprs, NoSafeTailTransfers,
                      /*ForceFullCanonicalFrame=*/true, ValidatedSourceExits)) {
      log() << "hotswap: tail source function at 0x"
            << utohexstr(Entry.first.first - TextAddr)
            << " was not certified to preserve the incoming s30 link\n";
      continue;
    }
    for (size_t Tail : Entry.second)
      if (ValidatedSourceExits.contains(Tail))
        SafeAbiTailTransfers.insert(Tail);
  }

  // Phase 2 may now authorize only the exact source indices proven above
  // while validating ordinary s30-return functions.
  DenseSet<size_t> CanonicalAbiReturns;
  DenseSet<uint64_t> CanonicalReturningFunctionBegins;
  for (const auto &Entry : ReturnsByFunction) {
    const ElfView::FunctionTextRange *Range =
        ReturnFunctionRanges.lookup(Entry.first);
    if (!Range || !validateCanonicalAbiFrameReturns(
                      Decoded, LS, TextAddr, *Range, Entry.second, ExactSetPcs,
                      ExactSetPcEntries, AllCallContinuations, PotentialEntries,
                      ActualRootEntries, AbiRootEntries, Index, AbiLinkPair,
                      *NumberedSgprs, SafeAbiTailTransfers,
                      /*ForceFullCanonicalFrame=*/false, CanonicalAbiReturns)) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "local function at 0x"
            << utohexstr(Entry.first.first - TextAddr)
            << " does not use a provable canonical s30 call frame\n";
      return std::nullopt;
    }
    for (size_t Return : Entry.second)
      if (CanonicalAbiReturns.contains(Return)) {
        CanonicalReturningFunctionBegins.insert(Entry.first.first - TextAddr);
        break;
      }
  }

  bool SawAbiCall = false;
  bool SawAbiReturn = false;
  SmallVector<std::pair<size_t, uint64_t>, 16> CallContinuations;
  SmallVector<const FiniteSetPcTransfer *, 16> SelectedSetPcs;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (LS.MIA->isCall(DI.Inst)) {
      std::optional<uint64_t> Continuation = checkedAddUint64(
          DI.Offset, DI.Size, "well-formed ABI call continuation");
      if (!Continuation || !addBoundary(*Continuation)) {
        log() << "hotswap: linked-code-object ABI entry-set fallback "
                 "rejected: call continuation at 0x"
              << utohexstr(DI.Offset) << " is not an instruction boundary\n";
        return std::nullopt;
      }
      CallContinuations.push_back({I, *Continuation});

      const MCOperand *Target =
          DI.Inst.getNumOperands() == 0
              ? nullptr
              : &DI.Inst.getOperand(DI.Inst.getNumOperands() - 1);
      if (Target && Target->isReg()) {
        if (DI.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
            DI.Inst.getNumOperands() != 2 ||
            !isExactRegisterOperand(DI.Inst, 0, AbiLinkPair) ||
            !Target->getReg() ||
            LS.MRI->regsOverlap(AbiLinkPair, Target->getReg()) ||
            !numberedSgprPairLowIndex(*LS.MRI, Target->getReg())) {
          log() << "hotswap: linked-code-object ABI entry-set fallback "
                   "rejected: register call at 0x"
                << utohexstr(DI.Offset)
                << " does not use the exact s30 swap-call shape\n";
          return std::nullopt;
        }
        Result.Calls.insert(I);
        if (!FiniteCalls.contains(I))
          SawAbiCall = true;
      } else if (!FiniteCalls.contains(I)) {
        log() << "hotswap: linked-code-object ABI entry-set fallback "
                 "rejected: non-register call at 0x"
              << utohexstr(DI.Offset) << " is not finite\n";
        return std::nullopt;
      }
    }

    if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
      if (DI.Inst.getNumOperands() != 1 || !DI.Inst.getOperand(0).isReg() ||
          !DI.Inst.getOperand(0).getReg()) {
        log() << "hotswap: linked-code-object ABI entry-set fallback "
                 "rejected: malformed set-PC at 0x"
              << utohexstr(DI.Offset) << "\n";
        return std::nullopt;
      }
      DenseMap<size_t, const FiniteSetPcTransfer *>::const_iterator It =
          ExactSetPcs.find(I);
      if (It != ExactSetPcs.end()) {
        if (!It->second->LocalTargetIndex ||
            *It->second->LocalTargetIndex >= Decoded.size() ||
            !addBoundary(Decoded[*It->second->LocalTargetIndex].Offset)) {
          log() << "hotswap: linked-code-object ABI entry-set fallback "
                   "rejected: exact set-PC at 0x"
                << utohexstr(DI.Offset)
                << " has an ambiguous or non-boundary target\n";
          return std::nullopt;
        }
        Result.SetPcs.insert(I);
        SelectedSetPcs.push_back(It->second);
      } else if (isExactRegisterOperand(DI.Inst, 0, AbiLinkPair) &&
                 CanonicalAbiReturns.contains(I)) {
        SawAbiReturn = true;
        Result.SetPcs.insert(I);
      } else {
        log() << "hotswap: linked-code-object ABI entry-set fallback "
                 "rejected: non-s30 set-PC at 0x"
              << utohexstr(DI.Offset)
              << " is not an exact local PC materialization\n";
        return std::nullopt;
      }
    }

    if (DI.Inst.getOpcode() == LS.SAddPcI64Opcode) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "s_add_pc_i64 at 0x"
            << utohexstr(DI.Offset) << "\n";
      return std::nullopt;
    }
    if (LS.MIA->isIndirectBranch(DI.Inst) &&
        DI.Inst.getOpcode() != LS.SSetPcI64Opcode &&
        !(LS.MIA->isCall(DI.Inst) &&
          DI.Inst.getOpcode() == LS.SSwapPcI64Opcode)) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "unsupported indirect transfer at 0x"
            << utohexstr(DI.Offset) << "\n";
      return std::nullopt;
    }
  }
  bool SawExactCanonicalCall = false;
  if (!SawAbiCall && SawAbiReturn)
    for (const KnownCallSite &Call : Index.Calls)
      if (Index.MaterializedCalls.contains(Call.InstIndex) &&
          Call.ReturnRegister == AbiLinkPair &&
          CanonicalReturningFunctionBegins.contains(Call.Target)) {
        SawExactCanonicalCall = true;
        break;
      }
  if (!SawAbiReturn || (!SawAbiCall && !SawExactCanonicalCall)) {
    log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
             "no opaque s30 call/return pair\n";
    return std::nullopt;
  }

  if (!SawAbiCall) {
    auto HasMaterializationEntry =
        [&](const PcMaterializedCallInfo &Materialized) {
          auto IsInterior = [&](uint64_t Offset) {
            return Offset > Materialized.SequenceStart &&
                   Offset <= Materialized.SequenceEnd;
          };
          for (uint64_t Entry : Result.Targets)
            if (IsInterior(Entry))
              return true;
          for (const DirectTargetSource &Source : Index.DirectTargetsByTarget)
            if (IsInterior(Source.Target))
              return true;
          for (const KnownCallSite &Call : Index.Calls)
            if (IsInterior(Call.Target) || IsInterior(Call.Continuation))
              return true;
          for (const auto &Call : CallContinuations)
            if (IsInterior(Call.second))
              return true;
          for (const FiniteSetPcTransfer *Transfer : SelectedSetPcs)
            if (Transfer->LocalTargetIndex &&
                IsInterior(Decoded[*Transfer->LocalTargetIndex].Offset))
              return true;
          return false;
        };
    for (const auto &Entry : Index.MaterializedCalls)
      if (Result.Calls.contains(Entry.first) &&
          HasMaterializationEntry(Entry.second)) {
        log() << "hotswap: exact materialized-call/canonical-return closure "
                 "rejected: alternate entry inside call materialization "
                 "ending at 0x"
              << utohexstr(Decoded[Entry.first].Offset) << "\n";
        return std::nullopt;
      }
  }

  auto hasInteriorEntry = [&](const FiniteSetPcTransfer &Candidate) {
    uint64_t Begin = Decoded[Candidate.SequenceBeginIndex].Offset;
    std::optional<uint64_t> End =
        checkedAddUint64(Decoded[Candidate.SequenceEndIndex].Offset,
                         Decoded[Candidate.SequenceEndIndex].Size,
                         "well-formed ABI materialization end");
    if (!End)
      return true;
    auto IsInterior = [&](uint64_t Offset) {
      return Offset > Begin && Offset < *End;
    };
    auto SourceIsSequence = [&](size_t InstIndex) {
      return InstIndex >= Candidate.SequenceBeginIndex &&
             InstIndex <= Candidate.SequenceEndIndex;
    };
    for (uint64_t Entry : Result.Targets)
      if (IsInterior(Entry))
        return true;
    for (const DirectTargetSource &Source : Index.DirectTargetsByTarget)
      if (!SourceIsSequence(Source.InstIndex) && IsInterior(Source.Target))
        return true;
    for (const auto &Call : CallContinuations)
      if (!SourceIsSequence(Call.first) && IsInterior(Call.second))
        return true;
    for (const FiniteSetPcTransfer *Transfer : SelectedSetPcs)
      if (Transfer->LocalTargetIndex &&
          IsInterior(Decoded[*Transfer->LocalTargetIndex].Offset))
        return true;
    return false;
  };
  for (const FiniteSetPcTransfer *Candidate : SelectedSetPcs)
    if (Candidate->SequenceBeginIndex > Candidate->SequenceEndIndex ||
        Candidate->SequenceEndIndex >= Decoded.size() ||
        hasInteriorEntry(*Candidate)) {
      log() << "hotswap: linked-code-object ABI entry-set fallback rejected: "
               "alternate entry inside set-PC materialization ending at 0x"
            << utohexstr(Decoded[Candidate->InstIndex].Offset) << "\n";
      return std::nullopt;
    }

  for (const auto &Call : CallContinuations)
    Result.Targets.insert(Call.second);
  log() << "hotswap: accepted "
        << (SawAbiCall ? "well-formed linked-code-object ABI entry set"
                       : "exact materialized-call/canonical-return closure")
        << " for " << Result.Calls.size() << " register call(s), "
        << Result.SetPcs.size() << " set-PC transfer(s), and "
        << Result.Targets.size() << " finite local entry point(s)\n";
  return Result;
}

static bool addReusableCallsToIndex(ArrayRef<InternalDecodedInst> Decoded,
                                    const LLVMState &LS, uint64_t TextAddr,
                                    uint64_t TextEnd,
                                    ArrayRef<ReachingCallTargets> ReusableCalls,
                                    ControlFlowScanIndex &Index) {
  for (size_t I = 0; I != ReusableCalls.size(); ++I) {
    if (ReusableCalls[I].empty() || Index.MaterializedCalls.contains(I))
      continue;
    std::optional<MCRegister> ReturnRegister =
        getCallReturnRegister(Decoded[I], LS);
    if (!ReturnRegister)
      continue;
    std::optional<uint64_t> Continuation =
        checkedAddUint64(Decoded[I].Offset, Decoded[I].Size,
                         "known reusable call continuation address");
    if (!Continuation)
      return false;
    bool HasExternalTarget = false;
    for (uint64_t Target : ReusableCalls[I])
      if (Target >= TextAddr && Target < TextEnd) {
        Index.Calls.push_back(
            {I, Target - TextAddr, *Continuation, *ReturnRegister});
      } else {
        HasExternalTarget = true;
      }
    if (HasExternalTarget)
      Index.ExternalCallContinuations.push_back({I, *Continuation});
  }
  return true;
}

static void finalizeCallContinuationIndex(ControlFlowScanIndex &Index) {
  Index.CallContinuationsByOffset.clear();
  for (const KnownCallSite &Call : Index.Calls)
    Index.CallContinuationsByOffset.push_back(
        {Call.InstIndex, Call.Continuation});
  llvm::sort(
      Index.CallContinuationsByOffset,
      [](const CallContinuationSource &LHS, const CallContinuationSource &RHS) {
        return std::tie(LHS.Continuation, LHS.InstIndex) <
               std::tie(RHS.Continuation, RHS.InstIndex);
      });
  llvm::sort(Index.ExternalCallContinuations,
             [](const ExternalCallContinuation &LHS,
                const ExternalCallContinuation &RHS) {
               return std::tie(LHS.Continuation, LHS.InstIndex) <
                      std::tie(RHS.Continuation, RHS.InstIndex);
             });
}

static void
addPotentialFiniteSetPcTransfersToIndex(ArrayRef<InternalDecodedInst> Decoded,
                                        ArrayRef<FiniteSetPcTransfer> Transfers,
                                        const BitVector &Reachable,
                                        ControlFlowScanIndex &Index) {
  for (const FiniteSetPcTransfer &Transfer : Transfers)
    if (Reachable.test(Transfer.InstIndex) && Transfer.LocalTargetIndex)
      Index.DirectTargetsByTarget.push_back(
          {Transfer.InstIndex, Decoded[*Transfer.LocalTargetIndex].Offset});
  llvm::sort(Index.DirectTargetsByTarget,
             [](const DirectTargetSource &LHS, const DirectTargetSource &RHS) {
               return std::tie(LHS.Target, LHS.InstIndex) <
                      std::tie(RHS.Target, RHS.InstIndex);
             });
}

/// Collect statically known direct branch and call destinations so an interior
/// entry point is never swallowed by coalescing.
std::optional<DirectControlFlowInfo> collectDirectBranchTargets(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize, ArrayRef<uint64_t> DeclaredEntries,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<uint64_t> ExternalEntries, ArrayRef<uint8_t> Text,
    const ElfView *Elf, ArrayRef<uint64_t> NonCallEntries) {
  if (!LS.MIA) {
    log() << "hotswap: MC branch analysis is unavailable; adjacent far "
             "trampolines will not be coalesced\n";
    return std::nullopt;
  }

  std::optional<uint64_t> TextEnd =
      checkedAddUint64(TextAddr, TextSize, "direct target text end");
  if (!TextEnd)
    return std::nullopt;

  SmallVector<FiniteSetPcTransfer, 8> AllSetPcCandidates =
      collectFiniteSetPcCandidates(Decoded, LS, TextAddr, *TextEnd,
                                   FunctionRanges);
  BitVector RejectedSetPcCandidates(AllSetPcCandidates.size());
  BitVector ProvenSetPcCandidates(AllSetPcCandidates.size());
  SmallVector<FiniteSetPcTransfer, 8> EnabledSetPcTransfers;
  std::vector<ReachingCallTargets> ReusableCalls;
  std::optional<ControlFlowScanIndex> Index;
  SmallVector<BoundedSetPcReturn, 2> BoundedReturns;
  SmallVector<SymbolLessReturnRegion, 8> SymbolLessRegions;
  bool IndirectControlFlowClosed = false;
  bool HasUnboundedIndirectEntries = false;

  // Exact set-PC edges are an optimistic over-approximation used only to
  // discover later finite call targets. Rebuild from scratch whenever the
  // closed-world audit removes an edge; never let a rejected edge leave
  // self-supporting call or return facts behind.
  for (;;) {
    EnabledSetPcTransfers = selectLeastReachableSetPcCandidates(
        Decoded, LS, DeclaredEntries, ExternalEntries, FunctionRanges, TextAddr,
        AllSetPcCandidates, ProvenSetPcCandidates, RejectedSetPcCandidates);
    ReusableCalls = resolveReusablePcCallTargets(
        Decoded, LS, TextAddr, *TextEnd, FunctionRanges, EnabledSetPcTransfers);
    Index = buildControlFlowScanIndex(Decoded, LS, TextAddr, *TextEnd,
                                      FunctionRanges);
    if (!Index || !addReusableCallsToIndex(Decoded, LS, TextAddr, *TextEnd,
                                           ReusableCalls, *Index))
      return std::nullopt;
    indexKnownCalls(*Index);
    finalizeCallContinuationIndex(*Index);
    BitVector PotentialSetPcSources = computeFiniteControlFlowReachability(
        Decoded, LS, TextAddr, TextSize, DeclaredEntries, ExternalEntries,
        FunctionRanges, *Index, EnabledSetPcTransfers,
        /*BoundedReturns=*/ArrayRef<BoundedSetPcReturn>{});
    addPotentialFiniteSetPcTransfersToIndex(Decoded, AllSetPcCandidates,
                                            PotentialSetPcSources, *Index);

    std::optional<SmallVector<BoundedSetPcReturn, 2>> FunctionReturns =
        collectBoundedSetPcReturns(Decoded, LS, TextAddr, *TextEnd,
                                   DeclaredEntries, FunctionRanges,
                                   ExternalEntries, *Index);
    if (!FunctionReturns)
      return std::nullopt;
    BoundedReturns = std::move(*FunctionReturns);
    BitVector CandidateReachability = computeFiniteControlFlowReachability(
        Decoded, LS, TextAddr, TextSize, DeclaredEntries, ExternalEntries,
        FunctionRanges, *Index, EnabledSetPcTransfers, BoundedReturns);
    bool AddedCandidate = false;
    for (size_t I = 0; I != AllSetPcCandidates.size(); ++I) {
      if (RejectedSetPcCandidates.test(I) || ProvenSetPcCandidates.test(I) ||
          !CandidateReachability.test(AllSetPcCandidates[I].InstIndex))
        continue;
      ProvenSetPcCandidates.set(I);
      AddedCandidate = true;
    }
    if (AddedCandidate)
      continue;
    BitVector ReachableCallSources = computeFiniteControlFlowReachability(
        Decoded, LS, TextAddr, TextSize, DeclaredEntries, ExternalEntries,
        FunctionRanges, *Index, EnabledSetPcTransfers, BoundedReturns);
    SymbolLessRegions = collectSymbolLessReturnRegions(
        Decoded, LS, TextAddr, TextSize, FunctionRanges, DeclaredEntries,
        ExternalEntries, *Index, EnabledSetPcTransfers, BoundedReturns,
        ReachableCallSources);
    SmallVector<BoundedSetPcReturn, 2> AllBoundedReturns = BoundedReturns;
    for (const SymbolLessReturnRegion &Region : SymbolLessRegions)
      for (size_t Return : Region.Returns) {
        SmallVector<uint64_t, 2> Targets(Region.Continuations.begin(),
                                         Region.Continuations.end());
        AllBoundedReturns.push_back({Return, std::move(Targets)});
      }
    FiniteControlFlowAudit Audit = auditFiniteIndirectControlFlow(
        Decoded, LS, TextAddr, TextSize, FunctionRanges, DeclaredEntries,
        ExternalEntries, *Index, EnabledSetPcTransfers, AllBoundedReturns,
        SymbolLessRegions);
    if (Audit.InvalidSetPcCandidates.any()) {
      for (size_t I = 0; I != EnabledSetPcTransfers.size(); ++I) {
        if (!Audit.InvalidSetPcCandidates.test(I))
          continue;
        for (size_t J = 0; J != AllSetPcCandidates.size(); ++J)
          if (AllSetPcCandidates[J].InstIndex ==
              EnabledSetPcTransfers[I].InstIndex) {
            RejectedSetPcCandidates.set(J);
          }
      }
      // Every dynamic proof is conditional on the complete edge set used to
      // reach it. A downstream candidate may have been reachable only through
      // a just-rejected candidate, so rediscover the least fixed point from
      // roots rather than retaining sticky proof bits.
      ProvenSetPcCandidates.reset();
      continue;
    }
    if (!Audit.Closed && !SymbolLessRegions.empty()) {
      // Symbol-less returns depend on a closed object-wide entry proof. If
      // any reachable entry source remains open, discard those inferred
      // regions before finalizing; unlike symbol-backed local returns, they
      // have no independent function boundary to constrain provenance.
      SymbolLessRegions.clear();
      AllBoundedReturns = BoundedReturns;
      Audit = auditFiniteIndirectControlFlow(
          Decoded, LS, TextAddr, TextSize, FunctionRanges, DeclaredEntries,
          ExternalEntries, *Index, EnabledSetPcTransfers, AllBoundedReturns,
          SymbolLessRegions);
    }
    if (!Audit.Closed && !EnabledSetPcTransfers.empty()) {
      for (const FiniteSetPcTransfer &Enabled : EnabledSetPcTransfers)
        for (size_t J = 0; J != AllSetPcCandidates.size(); ++J)
          if (AllSetPcCandidates[J].InstIndex == Enabled.InstIndex) {
            RejectedSetPcCandidates.set(J);
          }
      ProvenSetPcCandidates.reset();
      continue;
    }
    IndirectControlFlowClosed = Audit.Closed;
    HasUnboundedIndirectEntries = Audit.HasUnboundedIndirectEntries;
    if (!Audit.Closed)
      AllBoundedReturns.clear();
    BoundedReturns = std::move(AllBoundedReturns);
    break;
  }

  for (const FiniteSetPcTransfer &Transfer : EnabledSetPcTransfers) {
    SmallVector<uint64_t, 2> Targets;
    if (Transfer.LocalTargetIndex)
      Targets.push_back(Decoded[*Transfer.LocalTargetIndex].Offset);
    BoundedReturns.push_back({Transfer.InstIndex, std::move(Targets)});
  }
  indexKnownCalls(*Index);

  DenseMap<size_t, size_t> BoundedReturnPositions;
  for (size_t I = 0; I != BoundedReturns.size(); ++I)
    BoundedReturnPositions.try_emplace(BoundedReturns[I].InstIndex, I);

  // Canonical one-shot materializations also participate in the reusable
  // reaching-value solver so CFG joins can prove their exact path. Preserve
  // the established fail-closed entry proof once bounded returns are known:
  // an interior alias, fallthrough, or unbounded transfer may still bypass
  // the materialization even when its local dataflow token is exact.
  BitVector LocallyProvenMaterializedCalls(Decoded.size());
  for (const auto &Entry : Index->MaterializedCalls) {
    size_t I = Entry.first;
    if (ReusableCalls[I].empty())
      continue;
    if (hasKnownControlFlowEntry(
            DeclaredEntries, BoundedReturns, BoundedReturnPositions, *Index,
            HasUnboundedIndirectEntries, Entry.second.SequenceStart,
            Entry.second.SequenceEnd)) {
      ReusableCalls[I].clear();
      continue;
    }
    LocallyProvenMaterializedCalls.set(I);
  }

  std::optional<WellFormedAbiEntrySet> AbiEntrySet;
  if (!IndirectControlFlowClosed && Elf)
    AbiEntrySet = validateWellFormedAbiEntrySet(
        Decoded, LS, TextAddr, TextSize, FunctionRanges, DeclaredEntries,
        ExternalEntries, NonCallEntries, *Index, AllSetPcCandidates, *Elf);

  DirectControlFlowInfo Info;
  for (uint64_t Entry : DeclaredEntries)
    if (Entry < TextSize)
      Info.Targets.insert(Entry);
  for (uint64_t Entry : ExternalEntries)
    if (Entry < TextSize)
      Info.Targets.insert(Entry);
  for (const BoundedSetPcReturn &Return : BoundedReturns)
    for (uint64_t Target : Return.Targets)
      Info.Targets.insert(Target);
  for (const ExternalCallContinuation &Call : Index->ExternalCallContinuations)
    Info.Targets.insert(Call.Continuation);
  if (AbiEntrySet) {
    for (uint64_t Target : AbiEntrySet->Targets)
      Info.Targets.insert(Target);
    for (size_t InstIndex : AbiEntrySet->Calls)
      Info.BoundedIndirectTransfers.insert(Decoded[InstIndex].Offset);
    for (size_t InstIndex : AbiEntrySet->SetPcs)
      Info.BoundedIndirectTransfers.insert(Decoded[InstIndex].Offset);
  }
  for (size_t InstIndex : Index->BranchOrCallIndices) {
    const InternalDecodedInst &DI = Decoded[InstIndex];
    // Existing indirect branches are handled by
    // collectIndirectControlFlowFunctions(), which protects their containing
    // function from source relocation. Calls without a statically resolvable
    // target are handled below.
    if (LS.MIA->isIndirectBranch(DI.Inst))
      continue;

    bool HasPcRelativeOperand = false;
    for (const MCOperandInfo &Op : LS.MCII->get(DI.Inst.getOpcode()).operands())
      HasPcRelativeOperand |= Op.OperandType == MCOI::OPERAND_PCREL;
    if (!HasPcRelativeOperand) {
      // Preserve the established handling for non-call indirect transfers
      // such as s_set_pc_i64. collectIndirectControlFlowFunctions() prevents
      // source relocation in their containing function.
      if (!LS.MIA->isCall(DI.Inst))
        continue;
      std::optional<uint64_t> Target;
      if (DI.Inst.getOpcode() == LS.SSwapPcI64Opcode &&
          DI.Inst.getNumOperands() != 0 &&
          DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm()) {
        Target = static_cast<uint64_t>(
            DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm());
      } else {
        DenseMap<size_t, PcMaterializedCallInfo>::const_iterator Materialized =
            Index->MaterializedCalls.find(InstIndex);
        if (Materialized != Index->MaterializedCalls.end() &&
            !hasKnownControlFlowEntry(
                DeclaredEntries, BoundedReturns, BoundedReturnPositions, *Index,
                HasUnboundedIndirectEntries, Materialized->second.SequenceStart,
                Materialized->second.SequenceEnd))
          Target = Materialized->second.Target;
      }
      if (!ReusableCalls[InstIndex].empty()) {
        for (uint64_t ReusableTarget : ReusableCalls[InstIndex])
          if (ReusableTarget >= TextAddr && ReusableTarget < *TextEnd)
            Info.Targets.insert(ReusableTarget - TextAddr);
        Info.BoundedIndirectTransfers.insert(DI.Offset);
        if (LocallyProvenMaterializedCalls.test(InstIndex)) {
          log() << "hotswap: resolved PC-materialized call at 0x"
                << utohexstr(DI.Offset) << " to .text+0x"
                << utohexstr(ReusableCalls[InstIndex].front() - TextAddr)
                << "\n";
        } else {
          log() << "hotswap: resolved reusable PC-materialized call at 0x"
                << utohexstr(DI.Offset) << " to "
                << ReusableCalls[InstIndex].size() << " target(s)\n";
        }
        continue;
      }
      if (!Target) {
        if (AbiEntrySet && AbiEntrySet->Calls.contains(InstIndex))
          continue;
        log() << "hotswap: unresolved call target at 0x" << utohexstr(DI.Offset)
              << " (" << DI.Mnemonic << ")\n";
        Info.HasUnresolvedTargets = true;
        continue;
      }

      if (*Target >= TextAddr && *Target < *TextEnd) {
        uint64_t RelativeTarget = *Target - TextAddr;
        Info.Targets.insert(RelativeTarget);
        if (DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isReg())
          log() << "hotswap: resolved PC-materialized call at 0x"
                << utohexstr(DI.Offset) << " to .text+0x"
                << utohexstr(RelativeTarget) << "\n";
      }
      // A proven finite register target outside this object's .text cannot
      // enter a local instruction or synthetic source range. Keep that
      // control-flow proof separate from whether the target contributes a
      // local offset to the mutation-protection set.
      if (DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isReg()) {
        Info.BoundedIndirectTransfers.insert(DI.Offset);
      }
      continue;
    }

    std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
    if (!Target) {
      log() << "hotswap: MC analysis could not evaluate direct control-flow "
               "instruction at 0x"
            << utohexstr(DI.Offset)
            << "; adjacent far trampolines will not be coalesced\n";
      return std::nullopt;
    }
    if (*Target < TextSize) {
      Info.Targets.insert(*Target);
    } else if (LS.MIA->isCall(DI.Inst)) {
      std::optional<uint64_t> Continuation = checkedAddUint64(
          DI.Offset, DI.Size, "finite external direct call continuation");
      if (!Continuation)
        return std::nullopt;
      Info.Targets.insert(*Continuation);
    }
  }
  for (const BoundedSetPcReturn &Return : BoundedReturns)
    Info.BoundedIndirectTransfers.insert(Decoded[Return.InstIndex].Offset);
  if (!AbiEntrySet && !IndirectControlFlowClosed && HasUnboundedIndirectEntries)
    Info.HasUnboundedIndirectEntries = true;
  return Info;
}

/// Coalesce runs of adjacent far patch sites when the same SGPR scratch block
/// is safe at every site. Removing each interior return reservation preserves
/// replacement order and reduces the number of required forward gateways.
/// This deliberately never steals an unpatched neighboring instruction.
static void
mergeAdjacentLongTrampolines(std::vector<Trampoline> &Trampolines,
                             const DenseSet<uint64_t> &DirectBranchTargets) {
  std::vector<Trampoline> Merged;
  Merged.reserve(Trampolines.size());
  uint64_t MergeCount = 0;

  for (Trampoline &T : Trampolines) {
    bool Adjacent = false;
    if (!Merged.empty()) {
      Trampoline &Prev = Merged.back();
      std::optional<uint64_t> PrevEnd = checkedAddUint64(
          Prev.OriginalOffset, Prev.OriginalSize, "adjacent trampoline end");
      uint32_t BackReserve = Prev.LongBranchPreservesVcc
                                 ? VccPreservingReturnReserveBytes
                                 : SetPcReturnReserveBytes;
      uint32_t BodyPrefix =
          Prev.LongBranchPreservesVcc ? VccRestoreSequenceBytes : 0;
      Adjacent = PrevEnd && *PrevEnd == T.OriginalOffset && Prev.Long &&
                 T.Long && Prev.UsesSetPCBack && T.UsesSetPCBack &&
                 Prev.LongBranchPreservesVcc == T.LongBranchPreservesVcc &&
                 Prev.LongBranchSgprBase == T.LongBranchSgprBase &&
                 Prev.LongBranchUsesVcc == T.LongBranchUsesVcc &&
                 Prev.HasFunctionRange && T.HasFunctionRange &&
                 Prev.FunctionStart == T.FunctionStart &&
                 Prev.FunctionEnd == T.FunctionEnd &&
                 !DirectBranchTargets.contains(T.OriginalOffset) &&
                 Prev.Bytes.size() >= BackReserve &&
                 T.Bytes.size() >= BackReserve + BodyPrefix;
    }

    if (!Adjacent) {
      Merged.emplace_back(std::move(T));
      continue;
    }

    Trampoline &Prev = Merged.back();
    if (T.OriginalSize >
        std::numeric_limits<uint32_t>::max() - Prev.OriginalSize) {
      Merged.emplace_back(std::move(T));
      continue;
    }
    uint32_t BackReserve = Prev.LongBranchPreservesVcc
                               ? VccPreservingReturnReserveBytes
                               : SetPcReturnReserveBytes;
    size_t BodyPrefix =
        Prev.LongBranchPreservesVcc ? VccRestoreSequenceBytes : 0;
    Prev.Bytes.resize(Prev.Bytes.size() - BackReserve);
    Prev.Bytes.append(T.Bytes.begin() + BodyPrefix, T.Bytes.end());
    Prev.OriginalSize += T.OriginalSize;
    ++MergeCount;
  }

  Trampolines = std::move(Merged);
  if (MergeCount != 0)
    log() << "hotswap: coalesced " << MergeCount
          << " adjacent far trampoline edge(s)\n";
}

static void appendPoolBranchIslands(std::vector<Trampoline> &Trampolines) {
  for (Trampoline &T : Trampolines) {
    if (!T.Long)
      continue;
    T.Bytes.append(PoolBranchIslandBytes, uint8_t{0});
    T.HasPoolBranchIsland = true;
  }
}

/// Live-VCC preservation can be selected for an original eight-byte site so
/// adjacent patched sites can merge into a restore+delay landing. If
/// coalescing cannot prove at least that 12-byte source window, fall back to
/// the registerless route before gateway planning. Ordinary straight-line
/// expansion deliberately skips these candidates: growing their pool bodies
/// after initial layout could invalidate a later short-branch classification.
static bool finalizeDeferredVccPreservation(PatchContext &Ctx) {
  for (Trampoline &T : Ctx.OutTrampolines) {
    if (!T.LongBranchPreservesVcc)
      continue;
    if (T.Bytes.size() <
        VccRestoreSequenceBytes + VccPreservingReturnReserveBytes) {
      log() << "hotswap: error: deferred live-VCC trampoline at 0x"
            << utohexstr(T.OriginalOffset) << " is truncated\n";
      return false;
    }
    if (T.OriginalSize < VccPreservingSourceBytes) {
      T.Bytes.erase(T.Bytes.begin(), T.Bytes.begin() + VccRestoreSequenceBytes);
      T.Bytes.resize(T.Bytes.size() - VccPreservingReturnReserveBytes);
      T.Bytes.insert(T.Bytes.end(), MinInstSize, uint8_t{0});
      T.UsesSetPCBack = false;
      T.LongBranchSgprBase = 0;
      T.LongBranchUsesVcc = false;
      T.LongBranchPreservesVcc = false;
      log() << "hotswap: deferred live-VCC preservation at 0x"
            << utohexstr(T.OriginalOffset)
            << " fell back to a registerless far return\n";
      continue;
    }
    SafeSgprScratchBlock Save{T.LongBranchSgprBase, 1};
    if (!commitSafeSgprScratchBlock(Ctx, T.OriginalOffset, Save,
                                    "activated VCC-preserving far return"))
      return false;
  }
  return true;
}

static bool isEndProgram(const InternalDecodedInst &DI, const LLVMState &LS) {
  unsigned Opcode = DI.Inst.getOpcode();
  return Opcode == LS.SEndPgmOpcode || Opcode == LS.SEndPgmSavedOpcode;
}

static bool isPcSensitive(const InternalDecodedInst &DI, const LLVMState &LS) {
  unsigned Opcode = DI.Inst.getOpcode();
  return Opcode == LS.SAddPcI64Opcode || Opcode == LS.SGetPcI64Opcode ||
         Opcode == LS.SSetPcI64Opcode || Opcode == LS.SSwapPcI64Opcode ||
         Opcode == LS.SPrefetchInstPcRelOpcode ||
         Opcode == LS.SPrefetchDataPcRelOpcode;
}

static bool isSafeStraightLineRelocation(const InternalDecodedInst &DI,
                                         const LLVMState &LS,
                                         const DenseSet<uint64_t> &Protected) {
  if (!LS.MIA || LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))
    return false;
  unsigned Opcode = DI.Inst.getOpcode();
  return DI.DecodeSucceeded && !Protected.contains(DI.Offset) &&
         Opcode != LS.SClauseOpcode && Opcode != LS.SDelayAluOpcode &&
         !isPcSensitive(DI, LS);
}

/// Decode the bytes currently present at an original instruction site. Earlier
/// rewrite passes may have changed Ctx.Text after Ctx.Decoded was populated, so
/// relocation decisions must not classify the stale MCInst and then copy a
/// different instruction. A size change is conservatively non-relocatable.
static std::optional<InternalDecodedInst>
decodeCurrentInstruction(const PatchContext &Ctx,
                         const InternalDecodedInst &Original) {
  if (Original.Offset > Ctx.TextSize ||
      Original.Size > Ctx.TextSize - Original.Offset)
    return std::nullopt;

  std::vector<InternalDecodedInst> Current;
  if (!decodeTextSection(Ctx.Text + Original.Offset, Original.Size, Ctx.LS,
                         Current) ||
      Current.size() != 1 || Current[0].Size != Original.Size)
    return std::nullopt;
  Current[0].Offset = Original.Offset;
  return std::move(Current[0]);
}

/// Instructions covered by a hard clause or a delay directive must remain in
/// place relative to that directive. B0-to-A0 rewrites have already replaced
/// clauses with s_nop, so only preserve clause members when requested. Always
/// mark the maximum six-instruction forward span addressable by s_delay_alu.
static DenseSet<uint64_t>
collectRelocationProtectedOffsets(ArrayRef<InternalDecodedInst> Decoded,
                                  const LLVMState &LS,
                                  bool ProtectClauseMembers) {
  DenseSet<uint64_t> Protected;
  unsigned ClauseRemaining = 0;
  unsigned DelayRemaining = 0;

  for (const InternalDecodedInst &DI : Decoded) {
    if (ClauseRemaining != 0) {
      Protected.insert(DI.Offset);
      --ClauseRemaining;
    }
    if (DelayRemaining != 0) {
      Protected.insert(DI.Offset);
      --DelayRemaining;
    }

    if (ProtectClauseMembers && DI.Inst.getOpcode() == LS.SClauseOpcode &&
        DI.Inst.getNumOperands() == 1 && DI.Inst.getOperand(0).isImm())
      ClauseRemaining =
          (static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) & 63u) + 1;
    else if (DI.Inst.getOpcode() == LS.SDelayAluOpcode)
      DelayRemaining = 6;
  }
  return Protected;
}

/// Relocating an instruction changes its address. In a function containing a
/// register-based PC transfer, MC cannot prove that the instruction is not an
/// indirect destination, so leave the complete function in place.
static DenseSet<uint64_t>
collectIndirectControlFlowFunctions(ArrayRef<InternalDecodedInst> Decoded,
                                    const LLVMState &LS, const ElfView &Elf,
                                    const DenseSet<uint64_t> &Bounded) {
  DenseSet<uint64_t> Functions;
  if (!LS.MIA)
    return Functions;

  for (const InternalDecodedInst &DI : Decoded) {
    if (Bounded.contains(DI.Offset))
      continue;
    if (LS.MIA->isBarrier(DI.Inst) || isEndProgram(DI, LS))
      continue;
    if (!LS.MIA->isIndirectBranch(DI.Inst) &&
        !(LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI) &&
          isPcSensitive(DI, LS)))
      continue;
    std::optional<ElfView::FunctionTextRange> Range =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    if (Range && Functions.insert(Range->Begin).second)
      log() << "hotswap: source relocation disabled for function at 0x"
            << utohexstr(Range->Begin) << " by " << DI.Mnemonic << " at 0x"
            << utohexstr(DI.Offset) << "\n";
  }
  return Functions;
}

/// Grow undersized far-site windows only through proven straight-line code.
/// Patched neighbors are merged; ordinary instructions are copied verbatim
/// into the trampoline body and retain their original order. This is bounded
/// to the source bytes required by the selected gfx12 set-PC sequence and, for
/// a live wave32 VCC, its restore landing pad.
static void
expandStraightLineTrampolines(PatchContext &Ctx,
                              const DenseSet<uint64_t> &DirectBranchTargets) {
  DenseMap<uint64_t, size_t> DecodedAt;
  for (size_t I = 0; I != Ctx.Decoded.size(); ++I)
    DecodedAt[Ctx.Decoded[I].Offset] = I;
  DenseSet<uint64_t> Protected = collectRelocationProtectedOffsets(
      Ctx.Decoded, Ctx.LS, !Ctx.Config.RunB0A0Patches);
  DenseSet<uint64_t> IndirectControlFlowFunctions =
      collectIndirectControlFlowFunctions(
          Ctx.Decoded, Ctx.LS, Ctx.Elf,
          Ctx.DirectControlFlow.BoundedIndirectTransfers);

  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    if (Ctx.OutTrampolines[I].HasFunctionRange &&
        IndirectControlFlowFunctions.contains(
            Ctx.OutTrampolines[I].FunctionStart))
      continue;
    if (Ctx.OutTrampolines[I].LongBranchPreservesVcc)
      continue;
    while (Ctx.OutTrampolines[I].Long && Ctx.OutTrampolines[I].UsesSetPCBack &&
           Ctx.OutTrampolines[I].OriginalSize <
               (Ctx.OutTrampolines[I].LongBranchPreservesVcc
                    ? VccPreservingReturnReserveBytes + VccLandingPadBytes
                    : SetPcForwardSequenceBytes)) {
      Trampoline &T = Ctx.OutTrampolines[I];
      std::optional<uint64_t> End = checkedAddUint64(
          T.OriginalOffset, T.OriginalSize, "straight-line expansion end");
      if (!End || DirectBranchTargets.contains(*End))
        break;

      if (I + 1 < Ctx.OutTrampolines.size() &&
          Ctx.OutTrampolines[I + 1].OriginalOffset == *End) {
        if (T.LongBranchPreservesVcc)
          break;
        Trampoline &Next = Ctx.OutTrampolines[I + 1];
        if (!Next.Long || !Next.UsesSetPCBack ||
            Next.LongBranchSgprBase != T.LongBranchSgprBase ||
            Next.LongBranchUsesVcc != T.LongBranchUsesVcc ||
            Next.LongBranchPreservesVcc != T.LongBranchPreservesVcc ||
            !T.HasFunctionRange || !Next.HasFunctionRange ||
            T.FunctionStart != Next.FunctionStart ||
            T.FunctionEnd != Next.FunctionEnd ||
            Next.Bytes.size() < SetPcReturnReserveBytes)
          break;
        T.Bytes.resize(T.Bytes.size() - SetPcReturnReserveBytes);
        T.Bytes.append(Next.Bytes.begin(), Next.Bytes.end());
        T.OriginalSize += Next.OriginalSize;
        Ctx.OutTrampolines.erase(Ctx.OutTrampolines.begin() + I + 1);
        continue;
      }

      DenseMap<uint64_t, size_t>::const_iterator It = DecodedAt.find(*End);
      if (It == DecodedAt.end())
        break;
      const InternalDecodedInst &Original = Ctx.Decoded[It->second];
      std::optional<InternalDecodedInst> Current =
          decodeCurrentInstruction(Ctx, Original);
      if (!Current)
        break;
      const InternalDecodedInst &DI = *Current;
      uint32_t BackReserve = T.LongBranchPreservesVcc
                                 ? VccPreservingReturnReserveBytes
                                 : SetPcReturnReserveBytes;
      std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset);
      if (!Range || !T.HasFunctionRange || Range->Begin != T.FunctionStart ||
          Range->End != T.FunctionEnd ||
          !isSafeStraightLineRelocation(DI, Ctx.LS, Protected) ||
          T.Bytes.size() < BackReserve)
        break;

      T.Bytes.insert(T.Bytes.end() - BackReserve, Ctx.Text + DI.Offset,
                     Ctx.Text + DI.Offset + DI.Size);
      T.OriginalSize += DI.Size;
    }

    while (Ctx.OutTrampolines[I].Long && Ctx.OutTrampolines[I].UsesSetPCBack &&
           Ctx.OutTrampolines[I].OriginalSize <
               (Ctx.OutTrampolines[I].LongBranchPreservesVcc
                    ? VccPreservingReturnReserveBytes + VccLandingPadBytes
                    : SetPcForwardSequenceBytes)) {
      Trampoline &T = Ctx.OutTrampolines[I];
      if (DirectBranchTargets.contains(T.OriginalOffset))
        break;
      DenseMap<uint64_t, size_t>::const_iterator It =
          DecodedAt.find(T.OriginalOffset);
      if (It == DecodedAt.end() || It->second == 0)
        break;
      const InternalDecodedInst &Original = Ctx.Decoded[It->second - 1];
      std::optional<InternalDecodedInst> Current =
          decodeCurrentInstruction(Ctx, Original);
      if (!Current)
        break;
      const InternalDecodedInst &DI = *Current;
      if (DI.Offset + DI.Size != T.OriginalOffset ||
          !isSafeStraightLineRelocation(DI, Ctx.LS, Protected))
        break;
      if (I != 0) {
        const Trampoline &Previous = Ctx.OutTrampolines[I - 1];
        if (Previous.OriginalOffset + Previous.OriginalSize > DI.Offset)
          break;
      }
      std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset);
      if (!Range || !T.HasFunctionRange || Range->Begin != T.FunctionStart ||
          Range->End != T.FunctionEnd)
        break;
      size_t BodyPrefix =
          T.LongBranchPreservesVcc ? VccRestoreSequenceBytes : 0;
      T.Bytes.insert(T.Bytes.begin() + BodyPrefix, Ctx.Text + DI.Offset,
                     Ctx.Text + DI.Offset + DI.Size);
      T.OriginalOffset = DI.Offset;
      T.OriginalSize += DI.Size;
    }
  }
}

static bool hasNoFallthrough(const InternalDecodedInst &DI,
                             const LLVMState &LS) {
  return isEndProgram(DI, LS) ||
         (LS.MIA &&
          (LS.MIA->isUnconditionalBranch(DI.Inst) ||
           LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
           LS.MIA->isBarrier(DI.Inst)));
}

static void appendGatewaySled(std::vector<NopSled> &Sleds, uint64_t Start,
                              uint64_t End, uint64_t TextSize, bool Safe,
                              bool HasTarget) {
  if (Safe && !HasTarget && End - Start >= MinInstSize)
    Sleds.push_back({Start, End, Start, 0, TextSize});
}

/// Find zero-filled alignment holes, including holes covered by an oversized
/// function symbol, and s_nop padding outside every function. Such padding is
/// a safe branch gateway only when it follows a no-fallthrough instruction and
/// contains no direct branch/call target. In-function s_nop runs are added from
/// Ctx.NopSleds separately.
static std::vector<NopSled>
buildExternalGatewaySleds(ArrayRef<InternalDecodedInst> Decoded,
                          const LLVMState &LS, const ElfView &Elf,
                          ArrayRef<uint8_t> Text,
                          const DenseSet<uint64_t> &DirectBranchTargets) {
  std::vector<NopSled> Sleds;
  const InternalDecodedInst *Previous = nullptr;
  bool Active = false;
  bool Safe = false;
  bool HasTarget = false;
  uint64_t Start = 0;
  uint64_t End = 0;

  for (const InternalDecodedInst &DI : Decoded) {
    bool ZeroPadding =
        DI.Offset <= Text.size() && DI.Size <= Text.size() - DI.Offset;
    if (ZeroPadding)
      for (uint8_t Byte : Text.slice(DI.Offset, DI.Size))
        ZeroPadding &= Byte == 0;
    bool IsExternalNop = DI.Inst.getOpcode() == LS.SNopOpcode &&
                         !Elf.findFunctionTextRangeAtOffset(DI.Offset);
    bool GatewayPadding = ZeroPadding || IsExternalNop;
    if (!GatewayPadding || (Active && DI.Offset != End)) {
      if (Active)
        appendGatewaySled(Sleds, Start, End, Text.size(), Safe, HasTarget);
      Active = false;
    }
    if (!GatewayPadding) {
      Previous = &DI;
      continue;
    }
    if (!Active) {
      Active = true;
      Start = DI.Offset;
      Safe = Previous && hasNoFallthrough(*Previous, LS);
      HasTarget = false;
    }
    HasTarget |= DirectBranchTargets.contains(DI.Offset);
    End = DI.Offset + DI.Size;
  }
  if (Active)
    appendGatewaySled(Sleds, Start, End, Text.size(), Safe, HasTarget);
  return Sleds;
}

/// Gateway sleds are decoded before adjacent trampoline sites are coalesced.
/// Remove every final source interval from those earlier views, then callers
/// may add back only the explicitly unreachable source-tail subranges. This
/// prevents two independently-derived sleds from aliasing the same bytes.
static void subtractTrampolineSources(std::vector<NopSled> &Gateways,
                                      ArrayRef<Trampoline> Trampolines) {
  SmallVector<std::pair<uint64_t, uint64_t>, 64> Sources;
  for (const Trampoline &T : Trampolines) {
    std::optional<uint64_t> End = checkedAddUint64(
        T.OriginalOffset, T.OriginalSize, "gateway source interval");
    if (End)
      Sources.push_back({T.OriginalOffset, *End});
  }
  llvm::sort(Sources);

  std::vector<NopSled> Filtered;
  for (const NopSled &Sled : Gateways) {
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    if (Sled.WritePos >= UsableEnd)
      continue;
    uint64_t Cursor = Sled.WritePos;
    for (const auto &[SourceBegin, SourceEnd] : Sources) {
      if (SourceEnd <= Cursor)
        continue;
      if (SourceBegin >= UsableEnd)
        break;
      if (SourceBegin > Cursor)
        Filtered.push_back({Cursor, std::min(SourceBegin, UsableEnd), Cursor,
                            Sled.FunctionStart, Sled.FunctionEnd});
      Cursor = std::max(Cursor, SourceEnd);
      if (Cursor >= UsableEnd)
        break;
    }
    if (Cursor < UsableEnd)
      Filtered.push_back(
          {Cursor, UsableEnd, Cursor, Sled.FunctionStart, Sled.FunctionEnd});
  }
  Gateways = std::move(Filtered);
}

Expected<uint64_t>
countReachableSetPcGatewaySlots(ArrayRef<NopSled> Gateways, const LLVMState &LS,
                                uint64_t FromOffset, uint64_t TargetOffset,
                                unsigned SgprBase, uint64_t MaxSlots,
                                bool UseVcc, bool PreserveVcc) {
  uint64_t Slots = 0;
  for (const NopSled &Sled : Gateways) {
    if (FromOffset < Sled.FunctionStart || FromOffset >= Sled.FunctionEnd)
      continue;
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    uint64_t Candidate = Sled.WritePos;
    while (Candidate <= UsableEnd && Slots < MaxSlots) {
      uint64_t Distance = Candidate > FromOffset ? Candidate - FromOffset
                                                 : FromOffset - Candidate;
      if (Distance >= MaxSledDistance ||
          LS.encodeSBranch(FromOffset, Candidate).empty())
        break;
      std::optional<uint32_t> LayoutSize = getSetPcGatewayLayoutSize(
          Candidate, TargetOffset, SgprBase, UseVcc, PreserveVcc);
      if (!LayoutSize)
        return createStringError(
            Twine("invalid set-PC gateway while counting candidate "
                  "offset 0x") +
            utohexstr(Candidate));
      if (*LayoutSize > UsableEnd - Candidate)
        break;
      ++Slots;
      Candidate += *LayoutSize;
    }
    if (Slots == MaxSlots)
      break;
  }
  return Slots;
}

struct BranchIslandFailure {
  uint64_t CurrentOffset = 0;
  uint64_t TargetOffset = 0;
  uint64_t CorridorOffset = 0;
  bool Forward = false;
};

using BranchIslandPromoter =
    std::function<bool(const BranchIslandFailure &)>;

using BranchGatewayHead = std::pair<uint64_t, size_t>;
using BranchGatewayHeadSet = std::set<BranchGatewayHead>;

static bool hasFreeBranchGatewaySlot(const NopSled &Sled) {
  uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
  return !Sled.GatewayOnly && Sled.WritePos <= UsableEnd &&
         MinInstSize <= UsableEnd - Sled.WritePos;
}

static BranchGatewayHeadSet
buildBranchGatewayHeads(std::vector<NopSled> &Gateways,
                        const DenseSet<uint64_t> &Occupied) {
  BranchGatewayHeadSet Heads;
  for (size_t I = 0; I != Gateways.size(); ++I) {
    while (hasFreeBranchGatewaySlot(Gateways[I]) &&
           Occupied.contains(Gateways[I].WritePos))
      Gateways[I].WritePos += MinInstSize;
    if (hasFreeBranchGatewaySlot(Gateways[I]))
      Heads.insert({Gateways[I].WritePos, I});
  }
  return Heads;
}

static void
subtractOccupiedBranchGatewaySlots(std::vector<NopSled> &Gateways,
                                   const DenseSet<uint64_t> &Occupied) {
  SmallVector<uint64_t, 32> SortedOccupied(Occupied.begin(), Occupied.end());
  llvm::sort(SortedOccupied);
  std::vector<NopSled> Available;
  Available.reserve(Gateways.size());
  for (const NopSled &Sled : Gateways) {
    uint64_t Cursor = Sled.WritePos;
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    if (Cursor >= UsableEnd)
      continue;
    auto It = llvm::lower_bound(SortedOccupied, Cursor);
    while (It != SortedOccupied.end() && *It < UsableEnd) {
      if (Cursor < *It)
        Available.push_back({Cursor, *It, Cursor, Sled.FunctionStart,
                             Sled.FunctionEnd, Sled.GatewayOnly});
      Cursor = std::max(Cursor, *It + MinInstSize);
      ++It;
    }
    if (Cursor < UsableEnd)
      Available.push_back({Cursor, UsableEnd, Cursor, Sled.FunctionStart,
                           Sled.FunctionEnd, Sled.GatewayOnly});
  }
  Gateways = std::move(Available);
}

static std::optional<SmallVector<uint64_t, 4>>
allocateForwardBranchIslands(std::vector<NopSled> &Gateways,
                             uint64_t FromOffset, uint64_t TargetOffset,
                             BranchIslandFailure *Failure = nullptr,
                             BranchGatewayHeadSet *PersistentHeads = nullptr,
                             DenseSet<uint64_t> *PersistentOccupied = nullptr,
                             BranchIslandPromoter Promote = {}) {
  struct Allocation {
    size_t SledIndex = 0;
    uint64_t PreviousWritePos = 0;
  };
  BranchGatewayHeadSet LocalHeads;
  DenseSet<uint64_t> LocalOccupied;
  if (!PersistentOccupied)
    PersistentOccupied = &LocalOccupied;
  if (!PersistentHeads) {
    LocalHeads =
        buildBranchGatewayHeads(Gateways, *PersistentOccupied);
    PersistentHeads = &LocalHeads;
  }
  BranchGatewayHeadSet &Heads = *PersistentHeads;
  DenseSet<uint64_t> &Occupied = *PersistentOccupied;
  SmallVector<Allocation, 4> Allocations;
  SmallVector<uint64_t, 4> Islands;
  uint64_t Current = FromOffset;

  while (!isSBranchReachable(Current, TargetOffset)) {
    bool Forward = TargetOffset > Current;
    size_t BestIndex = Gateways.size();
    uint64_t BestOffset = 0;
    if (Forward) {
      uint64_t ReachEnd =
          Current > std::numeric_limits<uint64_t>::max() - MaxSledDistance
              ? std::numeric_limits<uint64_t>::max()
              : Current + MaxSledDistance;
      uint64_t Upper = std::min(TargetOffset, ReachEnd);
      auto It =
          TargetOffset <= ReachEnd
              ? Heads.lower_bound({Upper, 0})
              : Heads.upper_bound(
                    {Upper, std::numeric_limits<size_t>::max()});
      while (It != Heads.begin()) {
        --It;
        if (It->first <= Current)
          break;
        const NopSled &Sled = Gateways[It->second];
        if (FromOffset < Sled.FunctionStart ||
            FromOffset >= Sled.FunctionEnd ||
            Sled.WritePos != It->first ||
            !isSBranchReachable(Current, It->first))
          continue;
        BestIndex = It->second;
        BestOffset = It->first;
        break;
      }
    } else {
      uint64_t ReachBegin =
          Current > MaxSledDistance ? Current - MaxSledDistance : 0;
      uint64_t Lower =
          TargetOffset == std::numeric_limits<uint64_t>::max()
              ? TargetOffset
              : TargetOffset + 1;
      Lower = std::max(Lower, ReachBegin);
      for (auto It = Heads.lower_bound({Lower, 0});
           It != Heads.end() && It->first < Current; ++It) {
        const NopSled &Sled = Gateways[It->second];
        if (FromOffset < Sled.FunctionStart ||
            FromOffset >= Sled.FunctionEnd ||
            Sled.WritePos != It->first ||
            !isSBranchReachable(Current, It->first))
          continue;
        BestIndex = It->second;
        BestOffset = It->first;
        break;
      }
    }

    if (BestIndex == Gateways.size()) {
      uint64_t Corridor =
          Forward ? std::numeric_limits<uint64_t>::max() : 0;
      if (Forward) {
        uint64_t AfterCurrent =
            Current == std::numeric_limits<uint64_t>::max()
                ? Current
                : Current + 1;
        for (auto It = Heads.lower_bound({AfterCurrent, 0});
             It != Heads.end() && It->first < TargetOffset; ++It) {
          const NopSled &Sled = Gateways[It->second];
          if (FromOffset < Sled.FunctionStart ||
              FromOffset >= Sled.FunctionEnd)
            continue;
          Corridor = It->first;
          break;
        }
      } else {
        auto It = Heads.lower_bound({Current, 0});
        while (It != Heads.begin()) {
          --It;
          if (It->first <= TargetOffset)
            break;
          const NopSled &Sled = Gateways[It->second];
          if (FromOffset < Sled.FunctionStart ||
              FromOffset >= Sled.FunctionEnd)
            continue;
          Corridor = It->first;
          break;
        }
      }
      if (Forward && Corridor == std::numeric_limits<uint64_t>::max())
        Corridor = TargetOffset;
      if (!Forward && Corridor == 0)
        Corridor = TargetOffset;
      BranchIslandFailure ThisFailure{Current, TargetOffset, Corridor,
                                      Forward};
      if (Failure)
        *Failure = ThisFailure;
      // No Gateways reference or Heads iterator survives this call. The
      // promoter may grow both vectors/sets; successful promotion resumes the
      // same transaction with all prior heads still held.
      if (Promote && Promote(ThisFailure))
        continue;
      for (size_t I = Allocations.size(); I != 0; --I) {
        const Allocation &A = Allocations[I - 1];
        Gateways[A.SledIndex].WritePos = A.PreviousWritePos;
        Heads.insert({A.PreviousWritePos, A.SledIndex});
      }
      for (uint64_t Offset : Islands)
        Occupied.erase(Offset);
      return std::nullopt;
    }

    auto AliasBegin = Heads.lower_bound({BestOffset, 0});
    auto AliasEnd =
        Heads.upper_bound(
            {BestOffset, std::numeric_limits<size_t>::max()});
    for (auto It = AliasBegin; It != AliasEnd; ++It) {
      NopSled &Alias = Gateways[It->second];
      Allocations.push_back({It->second, Alias.WritePos});
      Alias.WritePos += MinInstSize;
    }
    Heads.erase(AliasBegin, AliasEnd);
    Occupied.insert(BestOffset);
    Islands.push_back(BestOffset);
    Current = BestOffset;
  }
  for (const Allocation &A : Allocations) {
    NopSled &Sled = Gateways[A.SledIndex];
    while (hasFreeBranchGatewaySlot(Sled) &&
           Occupied.contains(Sled.WritePos))
      Sled.WritePos += MinInstSize;
    if (hasFreeBranchGatewaySlot(Sled))
      Heads.insert({Gateways[A.SledIndex].WritePos, A.SledIndex});
  }
  return Islands;
}

static SmallVector<uint8_t> encodeScc1Branch(const LLVMState &LS,
                                             uint64_t FromOffset,
                                             uint64_t TargetOffset) {
  std::optional<uint64_t> PcBase =
      checkedAddUint64(FromOffset, MinInstSize, "conditional branch PC base");
  if (!PcBase || ((TargetOffset - *PcBase) & (MinInstSize - 1)) != 0)
    return {};
  int64_t DwordDelta =
      static_cast<int64_t>(TargetOffset - *PcBase) / MinInstSize;
  if (DwordDelta < BranchOffsetMin || DwordDelta > BranchOffsetMax)
    return {};
  return assembleSingleInst("s_cbranch_scc1 " + std::to_string(DwordDelta), LS);
}

bool sourceHasUniqueFunctionRange(
    const Trampoline &T, ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    uint64_t TextAddr) {
  if (!T.HasFunctionRange)
    return false;
  std::optional<uint64_t> SourceEnd = checkedAddUint64(
      T.OriginalOffset, T.OriginalSize, "source-tail unique function end");
  if (!SourceEnd)
    return false;

  SmallVector<std::pair<uint64_t, uint64_t>, 2> DistinctCoveringRanges;
  bool MatchesSelectedRange = false;
  for (const ElfView::FunctionTextRange &Range : FunctionRanges) {
    if (Range.Begin < TextAddr || Range.End < TextAddr)
      continue;
    uint64_t Begin = Range.Begin - TextAddr;
    uint64_t End = Range.End - TextAddr;
    if (Begin > T.OriginalOffset || End < *SourceEnd)
      continue;
    std::pair<uint64_t, uint64_t> Bounds{Begin, End};
    if (!llvm::is_contained(DistinctCoveringRanges, Bounds))
      DistinctCoveringRanges.push_back(Bounds);
    MatchesSelectedRange |= Begin == T.FunctionStart && End == T.FunctionEnd;
    if (DistinctCoveringRanges.size() > 1)
      return false;
  }
  return DistinctCoveringRanges.size() == 1 && MatchesSelectedRange;
}

class FunctionRangeUniquenessIndex {
  using Bounds = std::pair<uint64_t, uint64_t>;
  static constexpr size_t NoRange = std::numeric_limits<size_t>::max();

  struct PrefixMaximums {
    size_t First = NoRange;
    size_t Second = NoRange;
  };

  std::vector<Bounds> Ranges;
  std::vector<PrefixMaximums> Prefix;

public:
  FunctionRangeUniquenessIndex(
      ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
      uint64_t TextAddr) {
    Ranges.reserve(FunctionRanges.size());
    for (const ElfView::FunctionTextRange &Range : FunctionRanges) {
      if (Range.Begin < TextAddr || Range.End < TextAddr)
        continue;
      Bounds Relative{Range.Begin - TextAddr, Range.End - TextAddr};
      if (Relative.first < Relative.second)
        Ranges.push_back(Relative);
    }
    llvm::sort(Ranges);
    Ranges.erase(std::unique(Ranges.begin(), Ranges.end()), Ranges.end());

    Prefix.reserve(Ranges.size());
    PrefixMaximums Top;
    for (size_t I = 0; I != Ranges.size(); ++I) {
      if (Top.First == NoRange ||
          Ranges[I].second > Ranges[Top.First].second) {
        Top.Second = Top.First;
        Top.First = I;
      } else if (Top.Second == NoRange ||
                 Ranges[I].second > Ranges[Top.Second].second) {
        Top.Second = I;
      }
      Prefix.push_back(Top);
    }
  }

  bool hasUniqueFunctionRange(const Trampoline &T) const {
    if (!T.HasFunctionRange || T.FunctionStart >= T.FunctionEnd)
      return false;
    std::optional<uint64_t> SourceEnd = checkedAddUint64(
        T.OriginalOffset, T.OriginalSize,
        "indexed source-tail unique function end");
    if (!SourceEnd || T.FunctionStart > T.OriginalOffset ||
        T.FunctionEnd < *SourceEnd)
      return false;

    Bounds Selected{T.FunctionStart, T.FunctionEnd};
    auto SelectedIt = llvm::lower_bound(Ranges, Selected);
    if (SelectedIt == Ranges.end() || *SelectedIt != Selected)
      return false;

    auto PrefixEnd = std::upper_bound(
        Ranges.begin(), Ranges.end(), T.OriginalOffset,
        [](uint64_t Offset, const Bounds &Range) {
          return Offset < Range.first;
        });
    if (PrefixEnd == Ranges.begin())
      return false;
    const PrefixMaximums &Top = Prefix[PrefixEnd - Ranges.begin() - 1];
    size_t Other =
        Top.First != NoRange && Ranges[Top.First] == Selected
            ? Top.Second
            : Top.First;
    return Other == NoRange || Ranges[Other].second < *SourceEnd;
  }
};

bool sourceHasUniqueFunctionRangeIndexedForTest(
    const Trampoline &T, ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    uint64_t TextAddr) {
  return FunctionRangeUniquenessIndex(FunctionRanges, TextAddr)
      .hasUniqueFunctionRange(T);
}

bool isSafeSourceTailRange(const Trampoline &T,
                           const DirectControlFlowInfo &ControlFlow,
                           bool HasUniqueFunctionRange, uint64_t Begin,
                           uint64_t End) {
  if (ControlFlow.HasUnresolvedTargets ||
      ControlFlow.HasUnboundedIndirectEntries || !HasUniqueFunctionRange ||
      Begin >= End)
    return false;
  std::optional<uint64_t> FirstTailByte =
      checkedAddUint64(T.OriginalOffset, MinInstSize, "source-tail first byte");
  std::optional<uint64_t> SourceEnd = checkedAddUint64(
      T.OriginalOffset, T.OriginalSize, "source-tail source end");
  if (!FirstTailByte || !SourceEnd || Begin < *FirstTailByte ||
      End > *SourceEnd)
    return false;
  // These intervals are at most a handful of instruction dwords. Checking
  // every byte also catches an unusually-aligned declared alias rather than
  // assuming every protected entry is four-byte aligned.
  for (uint64_t Offset = Begin; Offset != End; ++Offset)
    if (ControlFlow.Targets.contains(Offset))
      return false;
  return true;
}

bool mustReserveSourceTailForRegisterlessReturn(const Trampoline &T) {
  return T.Long && !T.UsesSetPCBack && !T.LongBranchPreservesVcc &&
         !T.UsesSharedDispatcherForward && T.OriginalSize >= 2 * MinInstSize;
}

std::optional<std::pair<uint64_t, uint64_t>>
registerlessSourceAffineGatewayRange(const Trampoline &T) {
  constexpr uint64_t GatewayStart = 2 * MinInstSize;
  constexpr uint64_t GatewayBytes = 5 * MinInstSize;
  if (!mustReserveSourceTailForRegisterlessReturn(T) ||
      T.OriginalSize < GatewayStart + GatewayBytes)
    return std::nullopt;
  std::optional<uint64_t> Begin = checkedAddUint64(
      T.OriginalOffset, GatewayStart, "registerless affine gateway begin");
  std::optional<uint64_t> End =
      checkedAddUint64(T.OriginalOffset, GatewayStart + GatewayBytes,
                       "registerless affine gateway end");
  if (!Begin || !End)
    return std::nullopt;
  return std::pair<uint64_t, uint64_t>{*Begin, *End};
}

static SmallVector<uint8_t> encodeDirectCall(const LLVMState &LS,
                                             uint64_t FromOffset,
                                             uint64_t TargetOffset,
                                             StringRef ReturnPair) {
  std::optional<uint64_t> PcBase =
      checkedAddUint64(FromOffset, MinInstSize, "direct call PC base");
  if (!PcBase)
    return {};
  uint64_t ByteDistance =
      TargetOffset >= *PcBase ? TargetOffset - *PcBase : *PcBase - TargetOffset;
  if ((ByteDistance & (MinInstSize - 1)) != 0)
    return {};
  uint64_t DwordDistance = ByteDistance / MinInstSize;
  int64_t DwordDelta = 0;
  if (TargetOffset >= *PcBase) {
    if (DwordDistance > static_cast<uint64_t>(BranchOffsetMax))
      return {};
    DwordDelta = static_cast<int64_t>(DwordDistance);
  } else {
    const uint64_t MaxBackward =
        static_cast<uint64_t>(-static_cast<int64_t>(BranchOffsetMin));
    if (DwordDistance > MaxBackward)
      return {};
    DwordDelta = -static_cast<int64_t>(DwordDistance);
  }
  std::string Asm =
      "s_call_i64 " + ReturnPair.str() + ", " + std::to_string(DwordDelta);
  return assembleSingleInst(Asm, LS);
}

/// Plan common far gateways before the final pool layout. An 8-byte source
/// cannot hold the 20-byte SCC-neutral set-PC sequence, but it can preserve
/// its identity without touching SCC:
///
///   s_get_pc_i64 ScratchSource
///   s_branch CommonGateway
///
/// The common gateway reaches a dispatcher in the pool through a second
/// scratch pair. The dispatcher saves SCC, compares the recorded source PCs,
/// restores SCC in the selected stub, and branches to the matching trampoline
/// body. One 20-byte .text gateway can therefore serve hundreds of otherwise
/// independent 8-byte patch sites.
static bool planSharedDispatchGateways(PatchContext &Ctx,
                                       std::vector<NopSled> &TextGateways) {
  struct Candidate {
    size_t Index = 0;
    unsigned ScratchBase = 0;
  };
  SmallVector<Candidate, 64> Candidates;
  uint64_t MissingScratchCandidates = 0;
  uint64_t FirstMissingScratch = 0;
  uint64_t TP = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    uint64_t ThisTP = TP;
    std::optional<uint64_t> Next =
        checkedAddUint64(TP, T.Bytes.size(), "shared dispatcher pool layout");
    if (!Next)
      return false;
    TP = *Next;
    if (!T.Long || T.UsesMirroredStubForward ||
        T.OriginalSize < 2 * MinInstSize ||
        isSBranchReachable(T.OriginalOffset, ThisTP))
      continue;
    std::optional<SmallVector<uint8_t>> Direct = encodeSetPCLongBranch(
        Ctx.LS, T.OriginalOffset, ThisTP, T.LongBranchSgprBase);
    if (Direct && Direct->size() <= T.OriginalSize)
      continue;
    std::optional<SafeSgprScratchBlock> Scratch = findSafeSgprScratchBlock(
        Ctx, T.OriginalOffset, /*Count=*/4,
        /*Alignment=*/2, "shared far-dispatch gateway",
        /*ReportNoSpace=*/false);
    if (Scratch) {
      Candidates.push_back({I, Scratch->Base});
    } else {
      if (MissingScratchCandidates == 0)
        FirstMissingScratch = T.OriginalOffset;
      ++MissingScratchCandidates;
    }
  }
  if (MissingScratchCandidates != 0)
    log() << "hotswap: shared far-dispatch skipped " << MissingScratchCandidates
          << " site(s) without four safe SGPRs"
          << " (first at 0x" << utohexstr(FirstMissingScratch) << ")\n";

  // The ordinary planner is simpler and uses fewer scratch registers for
  // small objects. Shared dispatch is a capacity mechanism for dense far-site
  // workloads, not a replacement for individual gateways.
  log() << "hotswap: shared planner considering " << Candidates.size()
        << " source site(s) across " << TextGateways.size()
        << " gateway/relay window(s)\n";
  if (Candidates.size() < 8)
    return true;

  const std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Ctx.Elf.functionTextRanges();
  const FunctionRangeUniquenessIndex FunctionRangeIndex(FunctionRanges,
                                                          Ctx.Elf.textAddr());
  auto HasSafeTail = [&](const Trampoline &T, uint64_t Begin, uint64_t End) {
    bool HasUniqueFunctionRange =
        FunctionRangeIndex.hasUniqueFunctionRange(T);
    return isSafeSourceTailRange(T, Ctx.DirectControlFlow,
                                 HasUniqueFunctionRange, Begin, End);
  };

  BitVector Assigned(Ctx.OutTrampolines.size());
  SmallVector<SmallVector<size_t, 32>, 8> Groups;
  constexpr size_t MaxGroupSites = 1024;
  for (const Candidate &Seed : Candidates) {
    if (Assigned.test(Seed.Index))
      continue;
    const Trampoline &SeedT = Ctx.OutTrampolines[Seed.Index];

    size_t SledIndex = TextGateways.size();
    uint64_t BestDistance = std::numeric_limits<uint64_t>::max();
    for (size_t I = 0; I != TextGateways.size(); ++I) {
      const NopSled &Sled = TextGateways[I];
      uint64_t From = SeedT.OriginalOffset;
      if (SeedT.OriginalOffset < Sled.FunctionStart ||
          SeedT.OriginalOffset >= Sled.FunctionEnd)
        continue;
      uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
      if (Sled.WritePos > UsableEnd ||
          SetPcForwardSequenceBytes > UsableEnd - Sled.WritePos ||
          !isSBranchReachable(From, Sled.WritePos))
        continue;
      uint64_t Distance =
          From > Sled.WritePos ? From - Sled.WritePos : Sled.WritePos - From;
      if (Distance < BestDistance) {
        BestDistance = Distance;
        SledIndex = I;
      }
    }
    uint64_t GatewayOffset = 0;
    uint64_t SecondaryGatewayOffset = 0;
    uint64_t GatewayFunctionStart = 0;
    uint64_t GatewayFunctionEnd = std::numeric_limits<uint64_t>::max();
    std::vector<NopSled> WorkingGateways;
    SmallVector<uint64_t, 4> SeedIslands;
    if (SledIndex != TextGateways.size()) {
      GatewayOffset = TextGateways[SledIndex].WritePos;
      GatewayFunctionStart = TextGateways[SledIndex].FunctionStart;
      GatewayFunctionEnd = TextGateways[SledIndex].FunctionEnd;
      WorkingGateways = TextGateways;
      WorkingGateways[SledIndex].WritePos += SetPcForwardSequenceBytes;
    } else {
      size_t BestIslandCount = std::numeric_limits<size_t>::max();
      for (size_t I = 0; I != TextGateways.size(); ++I) {
        const NopSled &Sled = TextGateways[I];
        uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
        if (SeedT.OriginalOffset < Sled.FunctionStart ||
            SeedT.OriginalOffset >= Sled.FunctionEnd ||
            Sled.WritePos > UsableEnd ||
            SetPcForwardSequenceBytes > UsableEnd - Sled.WritePos)
          continue;
        std::vector<NopSled> Trial = TextGateways;
        uint64_t TrialGateway = Trial[I].WritePos;
        Trial[I].WritePos += SetPcForwardSequenceBytes;
        std::optional<SmallVector<uint64_t, 4>> Islands =
            allocateForwardBranchIslands(Trial, SeedT.OriginalOffset,
                                         TrialGateway);
        if (!Islands || Islands->empty() || Islands->size() >= BestIslandCount)
          continue;
        BestIslandCount = Islands->size();
        GatewayOffset = TrialGateway;
        GatewayFunctionStart = Sled.FunctionStart;
        GatewayFunctionEnd = Sled.FunctionEnd;
        WorkingGateways = std::move(Trial);
        SeedIslands = std::move(*Islands);
      }
      if (WorkingGateways.empty()) {
        // Split the SCC-neutral 20-byte sequence across an 8-byte get-PC
        // segment and a 16-byte add/set-PC segment. This admits functions
        // that have no single 20-byte padding window.
        for (size_t I = 0; I != TextGateways.size() && WorkingGateways.empty();
             ++I) {
          const NopSled &First = TextGateways[I];
          uint64_t FirstEnd = std::min(First.End, First.FunctionEnd);
          uint64_t SourceBranch = SeedT.OriginalOffset;
          if (SeedT.OriginalOffset < First.FunctionStart ||
              SeedT.OriginalOffset >= First.FunctionEnd ||
              First.WritePos > FirstEnd ||
              2 * MinInstSize > FirstEnd - First.WritePos ||
              !isSBranchReachable(SourceBranch, First.WritePos))
            continue;
          std::vector<NopSled> FirstReserved = TextGateways;
          GatewayOffset = FirstReserved[I].WritePos;
          FirstReserved[I].WritePos += 2 * MinInstSize;
          for (size_t J = 0; J != FirstReserved.size(); ++J) {
            const NopSled &Second = FirstReserved[J];
            uint64_t SecondEnd = std::min(Second.End, Second.FunctionEnd);
            if (SeedT.OriginalOffset < Second.FunctionStart ||
                SeedT.OriginalOffset >= Second.FunctionEnd ||
                Second.WritePos > SecondEnd ||
                4 * MinInstSize > SecondEnd - Second.WritePos ||
                !isSBranchReachable(GatewayOffset + MinInstSize,
                                    Second.WritePos))
              continue;
            WorkingGateways = FirstReserved;
            SecondaryGatewayOffset = WorkingGateways[J].WritePos;
            WorkingGateways[J].WritePos += 4 * MinInstSize;
            GatewayFunctionStart =
                std::max(First.FunctionStart, Second.FunctionStart);
            GatewayFunctionEnd =
                std::min(First.FunctionEnd, Second.FunctionEnd);
            break;
          }
        }
      }
      if (WorkingGateways.empty())
        continue;
    }

    SmallVector<size_t, 32> Members;
    DenseMap<size_t, SmallVector<uint64_t, 4>> MemberIslands;
    DenseMap<size_t, uint64_t> MemberRelays;
    uint64_t GroupBodyBytes = 0;
    constexpr uint64_t MaxDispatcherSpan = 120 * 1024;
    for (const Candidate &C : Candidates) {
      if (Members.size() == MaxGroupSites || Assigned.test(C.Index) ||
          C.ScratchBase != Seed.ScratchBase)
        continue;
      const Trampoline &T = Ctx.OutTrampolines[C.Index];
      if (T.OriginalOffset < GatewayFunctionStart ||
          T.OriginalOffset >= GatewayFunctionEnd)
        continue;
      uint64_t ProposedSpan =
          8 + 28 * (Members.size() + 1) + GroupBodyBytes + T.Bytes.size();
      if (ProposedSpan > MaxDispatcherSpan)
        continue;
      uint64_t From = T.OriginalOffset;
      SmallVector<uint64_t, 4> Islands;
      if (C.Index == Seed.Index && !SeedIslands.empty()) {
        Islands = SeedIslands;
      } else if (!isSBranchReachable(From, GatewayOffset)) {
        continue;
      }
      Members.push_back(C.Index);
      GroupBodyBytes += T.Bytes.size();
      if (!Islands.empty())
        MemberIslands[C.Index] = std::move(Islands);
    }
    DenseSet<size_t> LocalMembers;
    SmallVector<std::pair<uint64_t, size_t>, 32> RelayAnchors;
    for (size_t Index : Members) {
      LocalMembers.insert(Index);
      const Trampoline &T = Ctx.OutTrampolines[Index];
      uint64_t Tail = T.OriginalOffset + MinInstSize;
      if (HasSafeTail(T, Tail, Tail + MinInstSize))
        RelayAnchors.push_back({Tail, Index});
    }
    llvm::sort(RelayAnchors);
    bool AddedRelayMember;
    do {
      AddedRelayMember = false;
      for (const Candidate &C : Candidates) {
        if (Members.size() == MaxGroupSites || Assigned.test(C.Index) ||
            LocalMembers.contains(C.Index) || C.ScratchBase != Seed.ScratchBase)
          continue;
        const Trampoline &T = Ctx.OutTrampolines[C.Index];
        if (T.OriginalOffset < GatewayFunctionStart ||
            T.OriginalOffset >= GatewayFunctionEnd)
          continue;
        uint64_t ProposedSpan =
            8 + 28 * (Members.size() + 1) + GroupBodyBytes + T.Bytes.size();
        if (ProposedSpan > MaxDispatcherSpan)
          continue;
        uint64_t From = T.OriginalOffset;
        auto It =
            llvm::lower_bound(RelayAnchors, std::make_pair(From, size_t{0}));
        std::optional<uint64_t> Relay;
        if (It != RelayAnchors.end() && isSBranchReachable(From, It->first))
          Relay = It->first;
        if (It != RelayAnchors.begin()) {
          --It;
          if (isSBranchReachable(From, It->first))
            Relay = It->first;
        }
        if (!Relay)
          continue;
        Members.push_back(C.Index);
        LocalMembers.insert(C.Index);
        MemberRelays[C.Index] = *Relay;
        GroupBodyBytes += T.Bytes.size();
        uint64_t Tail = T.OriginalOffset + MinInstSize;
        if (HasSafeTail(T, Tail, Tail + MinInstSize))
          RelayAnchors.insert(
              llvm::lower_bound(RelayAnchors, std::make_pair(Tail, C.Index)),
              {Tail, C.Index});
        AddedRelayMember = true;
      }
    } while (AddedRelayMember && Members.size() != MaxGroupSites);
    // A single site with a normal 20-byte gateway gains nothing from the
    // dispatcher and would unnecessarily consume two extra SGPRs. Leave it to
    // the established direct planner. A split 8+16-byte gateway is retained
    // because the established planner cannot represent it.
    if (Members.size() == 1 && SecondaryGatewayOffset == 0)
      continue;
    if (Members.empty())
      continue;
    TextGateways = std::move(WorkingGateways);

    uint32_t Group = Groups.size() + 1;
    for (size_t Index : Members) {
      Trampoline &T = Ctx.OutTrampolines[Index];
      SafeSgprScratchBlock Scratch{Seed.ScratchBase, 4};
      if (!commitSafeSgprScratchBlock(Ctx, T.OriginalOffset, Scratch,
                                      "shared far-dispatch gateway"))
        return false;
      T.UsesSharedDispatcherForward = true;
      T.SharedDispatcherGroup = Group;
      T.SharedDispatcherSgprBase = Seed.ScratchBase;
      T.SharedDispatcherGatewayOffset = GatewayOffset;
      DenseMap<size_t, uint64_t>::const_iterator Relay =
          MemberRelays.find(Index);
      if (Relay != MemberRelays.end())
        T.SharedDispatcherRelayOffset = Relay->second;
      T.SharedDispatcherSecondaryGatewayOffset = SecondaryGatewayOffset;
      DenseMap<size_t, SmallVector<uint64_t, 4>>::iterator Islands =
          MemberIslands.find(Index);
      if (Islands != MemberIslands.end()) {
        T.ForwardBranchIslands = std::move(Islands->second);
        T.ForwardBranchTargetOffset = GatewayOffset;
      }
      Assigned.set(Index);
    }
    Groups.push_back(std::move(Members));
  }

  if (Groups.empty())
    return true;

  std::vector<Trampoline> Reordered;
  Reordered.reserve(Ctx.OutTrampolines.size());
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I)
    if (!Assigned.test(I))
      Reordered.push_back(std::move(Ctx.OutTrampolines[I]));
  for (const SmallVector<size_t, 32> &Group : Groups)
    for (size_t Index : Group)
      Reordered.push_back(std::move(Ctx.OutTrampolines[Index]));
  Ctx.OutTrampolines = std::move(Reordered);

  for (size_t I = 0; I != Ctx.OutTrampolines.size();) {
    Trampoline &First = Ctx.OutTrampolines[I];
    if (!First.UsesSharedDispatcherForward) {
      ++I;
      continue;
    }
    uint32_t Group = First.SharedDispatcherGroup;
    size_t Count = 0;
    while (I + Count != Ctx.OutTrampolines.size() &&
           Ctx.OutTrampolines[I + Count].SharedDispatcherGroup == Group)
      ++Count;
    uint64_t Prefix = 8 + 28 * Count;
    if (Prefix > std::numeric_limits<uint32_t>::max())
      return false;
    First.PoolEntryPrefixBytes = static_cast<uint32_t>(Prefix);
    First.Bytes.insert(First.Bytes.begin(), Prefix, uint8_t{0});
    I += Count;
  }

  log() << "hotswap: planned " << Groups.size()
        << " shared far-dispatch gateway group(s) for " << Assigned.count()
        << " source site(s)\n";
  return true;
}

/// Plan a relocation-neutral shared gateway for far sites that have only the
/// pair already reserved for their return edge. Each source records its own PC
/// and branches to a common SCC-neutral add/set-PC sequence:
///
///   s_call_i64 Pair, CommonGateway
///
/// The call records source S+4 without touching SCC. The common delta maps it
/// to a sparse pool stub at
/// StubBase + (S - MinSource). Thus the load bias cancels without a dispatcher
/// classifier or another scratch pair. The stub branches to S's trampoline
/// body. Sources are grouped only when their sparse prefix plus bodies stays
/// within short-branch range.
static bool planMirroredStubGateways(PatchContext &Ctx,
                                     std::vector<NopSled> &TextGateways) {
  const auto PlanStart = std::chrono::steady_clock::now();
  struct Candidate {
    size_t Index = 0;
    unsigned PairBase = 0;
    bool UsesVcc = false;
  };
  SmallVector<Candidate, 64> Candidates;
  uint64_t TP = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    uint64_t ThisTP = TP;
    std::optional<uint64_t> Next =
        checkedAddUint64(TP, T.Bytes.size(), "mirrored-stub pool layout");
    if (!Next)
      return false;
    TP = *Next;
    if (!T.Long || !T.UsesSetPCBack || T.LongBranchPreservesVcc ||
        T.UsesSharedDispatcherForward || T.UsesMirroredStubForward ||
        T.OriginalSize < 2 * MinInstSize ||
        isSBranchReachable(T.OriginalOffset, ThisTP))
      continue;
    std::optional<SmallVector<uint8_t>> Direct =
        encodeSetPCLongBranch(Ctx.LS, T.OriginalOffset, ThisTP,
                              T.LongBranchSgprBase, T.LongBranchUsesVcc);
    if (Direct && Direct->size() <= T.OriginalSize)
      continue;
    Candidates.push_back({I, T.LongBranchSgprBase, T.LongBranchUsesVcc});
  }
  if (Candidates.empty())
    return true;
  log() << "hotswap: affine planner considering " << Candidates.size()
        << " pair-only source site(s) across " << TextGateways.size()
        << " gateway/relay window(s)\n";
  // Two sources are the first point where one 12-byte text gateway replaces
  // multiple ordinary gateways and therefore adds real text-space capacity.
  constexpr size_t MinSharedSites = 2;
  if (Candidates.size() < MinSharedSites)
    return true;

  // The sparse stub is after .text and within 4 GiB, so its positive delta
  // uses the eight-byte literal32 form of s_add_nc_u64 plus four-byte set-PC.
  constexpr uint64_t GatewayReserve = 3 * MinInstSize;
  constexpr uint64_t MaxGroupSpan = 120 * 1024;
  constexpr size_t MaxGroupSites = 1024;
  BitVector Assigned(Ctx.OutTrampolines.size());
  SmallVector<SmallVector<size_t, 32>, 8> Groups;
  uint64_t DirectGatewayGroups = 0;
  uint64_t IslandGatewayGroups = 0;
  uint32_t GroupBase = 1;
  for (const Trampoline &T : Ctx.OutTrampolines)
    GroupBase = std::max(GroupBase, T.MirroredStubGroup + 1);

  // Prefer high-address seeds. Their gateway and pool are usually close, and
  // lower sources can reuse their unreachable second dword as a safe relay.
  llvm::stable_sort(Candidates,
                    [&](const Candidate &LHS, const Candidate &RHS) {
                      return Ctx.OutTrampolines[LHS.Index].OriginalOffset >
                             Ctx.OutTrampolines[RHS.Index].OriginalOffset;
                    });

  for (const Candidate &Seed : Candidates) {
    if (Assigned.test(Seed.Index))
      continue;
    const Trampoline &SeedT = Ctx.OutTrampolines[Seed.Index];

    uint64_t GatewayOffset = 0;
    uint64_t GatewayFunctionStart = 0;
    uint64_t GatewayFunctionEnd = std::numeric_limits<uint64_t>::max();
    std::vector<NopSled> WorkingGateways;
    bool GatewayReservedInText = false;
    SmallVector<uint64_t, 4> SeedIslands;
    uint64_t BestDistance = std::numeric_limits<uint64_t>::max();

    // Dense objects can expose tens of thousands of one-dword relays. Find a
    // directly reachable gateway without cloning that entire vector for every
    // candidate. A direct route is always preferable to an island chain, so
    // only use the copying fallback when no such gateway exists.
    size_t DirectSledIndex = TextGateways.size();
    uint64_t From = SeedT.OriginalOffset;
    for (size_t I = 0; I != TextGateways.size(); ++I) {
      const NopSled &Sled = TextGateways[I];
      uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
      if (From < Sled.FunctionStart || From >= Sled.FunctionEnd ||
          Sled.WritePos > UsableEnd ||
          GatewayReserve > UsableEnd - Sled.WritePos ||
          !isSBranchReachable(From, Sled.WritePos))
        continue;
      uint64_t Distance =
          From > Sled.WritePos ? From - Sled.WritePos : Sled.WritePos - From;
      if (Distance >= BestDistance)
        continue;
      DirectSledIndex = I;
      BestDistance = Distance;
    }
    if (DirectSledIndex != TextGateways.size()) {
      // The group always contains its seed once a direct gateway has been
      // found, so this reservation cannot need rollback. Commit it in place
      // instead of copying every relay window into a trial vector.
      GatewayOffset = TextGateways[DirectSledIndex].WritePos;
      GatewayFunctionStart = TextGateways[DirectSledIndex].FunctionStart;
      GatewayFunctionEnd = TextGateways[DirectSledIndex].FunctionEnd;
      TextGateways[DirectSledIndex].WritePos += GatewayReserve;
      GatewayReservedInText = true;
    }

    if (!GatewayReservedInText) {
      SmallVector<size_t, 32> GatewayCandidates;
      for (size_t I = 0; I != TextGateways.size(); ++I) {
        const NopSled &Sled = TextGateways[I];
        uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
        if (From < Sled.FunctionStart || From >= Sled.FunctionEnd ||
            Sled.WritePos > UsableEnd ||
            GatewayReserve > UsableEnd - Sled.WritePos)
          continue;
        GatewayCandidates.push_back(I);
      }
      llvm::sort(GatewayCandidates, [&](size_t LHS, size_t RHS) {
        uint64_t L = TextGateways[LHS].WritePos;
        uint64_t R = TextGateways[RHS].WritePos;
        uint64_t LDistance = From > L ? From - L : L - From;
        uint64_t RDistance = From > R ? From - R : R - From;
        return LDistance < RDistance;
      });

      // Failed island allocation rolls its own reservations back, so one
      // trial vector is enough for every possible gateway. This avoids the
      // quadratic full-vector cloning that dense corpus objects otherwise
      // trigger.
      std::vector<NopSled> Trial = TextGateways;
      for (size_t I : GatewayCandidates) {
        uint64_t TrialGateway = Trial[I].WritePos;
        Trial[I].WritePos += GatewayReserve;
        std::optional<SmallVector<uint64_t, 4>> Allocated =
            allocateForwardBranchIslands(Trial, From, TrialGateway);
        if (!Allocated) {
          Trial[I].WritePos -= GatewayReserve;
          continue;
        }
        GatewayOffset = TrialGateway;
        GatewayFunctionStart = TextGateways[I].FunctionStart;
        GatewayFunctionEnd = TextGateways[I].FunctionEnd;
        WorkingGateways = std::move(Trial);
        SeedIslands = std::move(*Allocated);
        break;
      }
    }
    if (!GatewayReservedInText && WorkingGateways.empty())
      continue;
    SmallVector<size_t, 32> Members;
    DenseMap<size_t, SmallVector<uint64_t, 4>> MemberIslands;
    uint64_t MinSource = SeedT.OriginalOffset;
    uint64_t MaxSource = SeedT.OriginalOffset;
    uint64_t GroupBodyBytes = 0;
    auto FitsGroup = [&](const Trampoline &T) {
      uint64_t NewMin = std::min(MinSource, T.OriginalOffset);
      uint64_t NewMax = std::max(MaxSource, T.OriginalOffset);
      uint64_t Prefix = NewMax - NewMin + MinInstSize;
      return Prefix <= MaxGroupSpan &&
             GroupBodyBytes <= MaxGroupSpan - Prefix &&
             T.Bytes.size() <= MaxGroupSpan - Prefix - GroupBodyBytes;
    };
    auto AddMember = [&](size_t Index) {
      const Trampoline &T = Ctx.OutTrampolines[Index];
      Members.push_back(Index);
      MinSource = std::min(MinSource, T.OriginalOffset);
      MaxSource = std::max(MaxSource, T.OriginalOffset);
      GroupBodyBytes += T.Bytes.size();
    };

    AddMember(Seed.Index);
    if (!SeedIslands.empty())
      MemberIslands[Seed.Index] = SeedIslands;
    for (const Candidate &C : Candidates) {
      if (C.Index == Seed.Index || Members.size() == MaxGroupSites ||
          Assigned.test(C.Index) || C.PairBase != Seed.PairBase ||
          C.UsesVcc != Seed.UsesVcc)
        continue;
      const Trampoline &T = Ctx.OutTrampolines[C.Index];
      if (T.OriginalOffset < GatewayFunctionStart ||
          T.OriginalOffset >= GatewayFunctionEnd || !FitsGroup(T))
        continue;
      uint64_t From = T.OriginalOffset;
      if (!isSBranchReachable(From, GatewayOffset))
        continue;
      AddMember(C.Index);
    }

    // Small affine groups only replace established ordinary gateways with a
    // larger sparse pool prefix. Preserve those simpler paths and reserve
    // mirrored stubs for dense cases where sharing materially adds capacity.
    if (Members.size() < MinSharedSites) {
      if (GatewayReservedInText)
        TextGateways[DirectSledIndex].WritePos -= GatewayReserve;
      continue;
    }

    if (GatewayReservedInText) {
      ++DirectGatewayGroups;
    } else {
      TextGateways = std::move(WorkingGateways);
      ++IslandGatewayGroups;
    }
    uint32_t Group = GroupBase + Groups.size();
    for (size_t Index : Members) {
      Trampoline &T = Ctx.OutTrampolines[Index];
      T.UsesMirroredStubForward = true;
      T.MirroredStubGroup = Group;
      T.MirroredStubGatewayOffset = GatewayOffset;
      DenseMap<size_t, SmallVector<uint64_t, 4>>::iterator Islands =
          MemberIslands.find(Index);
      if (Islands != MemberIslands.end()) {
        T.ForwardBranchIslands = std::move(Islands->second);
        T.ForwardBranchTargetOffset = GatewayOffset;
      }
      Assigned.set(Index);
    }
    Groups.push_back(std::move(Members));
  }

  if (Groups.empty())
    return true;

  // Preserve the pool offsets of unclaimed trampolines. The earlier planners
  // may have proved their ordinary short edges using those offsets; inserting
  // a sparse prefix ahead of them could invalidate that proof. Mirrored groups
  // can sit at the end because their common gateway uses a full 64-bit delta.
  std::vector<Trampoline> Reordered;
  Reordered.reserve(Ctx.OutTrampolines.size());
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I)
    if (!Assigned.test(I))
      Reordered.push_back(std::move(Ctx.OutTrampolines[I]));
  for (const SmallVector<size_t, 32> &Group : Groups)
    for (size_t Index : Group)
      Reordered.push_back(std::move(Ctx.OutTrampolines[Index]));
  Ctx.OutTrampolines = std::move(Reordered);

  for (size_t I = 0; I != Ctx.OutTrampolines.size();) {
    Trampoline &First = Ctx.OutTrampolines[I];
    if (!First.UsesMirroredStubForward) {
      ++I;
      continue;
    }
    uint32_t Group = First.MirroredStubGroup;
    size_t Count = 0;
    uint64_t MinSource = First.OriginalOffset;
    uint64_t MaxSource = First.OriginalOffset;
    while (I + Count != Ctx.OutTrampolines.size() &&
           Ctx.OutTrampolines[I + Count].UsesMirroredStubForward &&
           Ctx.OutTrampolines[I + Count].MirroredStubGroup == Group) {
      MinSource =
          std::min(MinSource, Ctx.OutTrampolines[I + Count].OriginalOffset);
      MaxSource =
          std::max(MaxSource, Ctx.OutTrampolines[I + Count].OriginalOffset);
      ++Count;
    }
    uint64_t Prefix = MaxSource - MinSource + MinInstSize;
    if (Prefix > std::numeric_limits<uint32_t>::max())
      return false;
    First.PoolEntryPrefixBytes = static_cast<uint32_t>(Prefix);
    First.Bytes.insert(First.Bytes.begin(), Prefix, uint8_t{0});
    I += Count;
  }

  log() << "hotswap: planned " << Groups.size()
        << " mirrored-stub gateway group(s) for " << Assigned.count()
        << " pair-only source site(s) (" << DirectGatewayGroups
        << " direct gateway group(s), " << IslandGatewayGroups
        << " island gateway group(s), "
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - PlanStart)
               .count()
        << " ms)\n";
  return true;
}

static bool emitSharedDispatchers(PatchContext &Ctx) {
  DenseMap<uint32_t, SmallVector<size_t, 32>> Groups;
  SmallVector<uint64_t, 64> PoolOffsets;
  uint64_t TP = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    PoolOffsets.push_back(TP);
    Trampoline &T = Ctx.OutTrampolines[I];
    if (T.UsesSharedDispatcherForward)
      Groups[T.SharedDispatcherGroup].push_back(I);
    std::optional<uint64_t> Next =
        checkedAddUint64(TP, T.Bytes.size(), "shared dispatcher final layout");
    if (!Next)
      return false;
    TP = *Next;
  }

  for (auto &KV : Groups) {
    ArrayRef<size_t> Members = KV.second;
    if (Members.empty())
      continue;
    Trampoline &Owner = Ctx.OutTrampolines[Members.front()];
    uint64_t DispatcherOffset = PoolOffsets[Members.front()];
    auto fail = [&](const Twine &Reason) {
      log() << "hotswap: error: shared dispatcher group " << KV.first
            << " at 0x" << utohexstr(DispatcherOffset) << ": " << Reason
            << "\n";
      return false;
    };
    unsigned Base = Owner.SharedDispatcherSgprBase;
    const std::string SourceLow = "s" + std::to_string(Base);
    const std::string CursorLow = "s" + std::to_string(Base + 2);
    // After the SCC-neutral gateway has transferred control, only the low
    // cursor half is needed: every source and pool address differs by less
    // than 4 GiB, so modulo-2^32 deltas remain exact across a load-address
    // wrap. Reuse the cursor high half to preserve SCC.
    const std::string Save = "s" + std::to_string(Base + 3);

    Owner.HasForwardGateway = true;
    Owner.ForwardGatewayOffset = Owner.SharedDispatcherGatewayOffset;
    if (Owner.SharedDispatcherSecondaryGatewayOffset == 0) {
      std::optional<SmallVector<uint8_t>> Gateway =
          encodeSetPCLongBranch(Ctx.LS, Owner.SharedDispatcherGatewayOffset,
                                DispatcherOffset, Base + 2);
      if (!Gateway || Gateway->size() > SetPcForwardSequenceBytes)
        return fail("single-segment gateway encoding failed");
      Owner.ForwardGatewayBytes = std::move(*Gateway);
    } else {
      const std::string GatewayPair = "s[" + std::to_string(Base + 2) + ":" +
                                      std::to_string(Base + 3) + "]";
      Owner.ForwardGatewayBytes =
          assembleSingleInst("s_get_pc_i64 " + GatewayPair, Ctx.LS);
      SmallVector<uint8_t> ToSecond =
          Ctx.LS.encodeSBranch(Owner.SharedDispatcherGatewayOffset +
                                   Owner.ForwardGatewayBytes.size(),
                               Owner.SharedDispatcherSecondaryGatewayOffset);
      if (Owner.ForwardGatewayBytes.size() != MinInstSize ||
          ToSecond.size() != MinInstSize)
        return fail("split gateway first segment encoding failed");
      Owner.ForwardGatewayBytes.append(ToSecond);

      uint64_t PcBase = Owner.SharedDispatcherGatewayOffset + MinInstSize;
      uint64_t Delta = DispatcherOffset - PcBase;
      SmallVector<std::string, 2> Lines;
      Lines.push_back("s_add_nc_u64 " + GatewayPair + ", " + GatewayPair +
                      ", 0x" + utohexstr(Delta));
      Lines.push_back("s_set_pc_i64 " + GatewayPair);
      Owner.SecondaryForwardGatewayBytes =
          assembleInstructions(joinAsmLines(Lines), Ctx.LS);
      if (Owner.SecondaryForwardGatewayBytes.empty() ||
          Owner.SecondaryForwardGatewayBytes.size() > 4 * MinInstSize)
        return fail("split gateway second segment encoding failed");
      while (Owner.SecondaryForwardGatewayBytes.size() < 4 * MinInstSize)
        Owner.SecondaryForwardGatewayBytes.append(Ctx.LS.SNopBytes);
    }

    SmallVector<uint64_t, 32> BodyOffsets;
    for (size_t Member : Members)
      BodyOffsets.push_back(PoolOffsets[Member] +
                            Ctx.OutTrampolines[Member].PoolEntryPrefixBytes);

    SmallVector<uint8_t> Bytes;
    auto appendInst = [&](StringRef Asm) {
      SmallVector<uint8_t> Encoded = assembleSingleInst(Asm, Ctx.LS);
      if (Encoded.empty())
        return false;
      Bytes.append(Encoded);
      return true;
    };
    if (!appendInst("s_cselect_b32 " + Save + ", 1, 0"))
      return fail("SCC save encoding failed");

    uint64_t CursorValue = DispatcherOffset;
    uint64_t StubBase =
        DispatcherOffset + 4 + 20 * Members.size() + MinInstSize;
    for (size_t J = 0; J != Members.size(); ++J) {
      const Trampoline &T = Ctx.OutTrampolines[Members[J]];
      uint64_t SourcePc = T.OriginalOffset + MinInstSize;
      uint64_t Distance = SourcePc > CursorValue ? SourcePc - CursorValue
                                                 : CursorValue - SourcePc;
      if (Distance >= (uint64_t{1} << 32))
        return fail("source-to-dispatcher span exceeds 32 bits");
      uint64_t Delta = SourcePc - CursorValue;
      SmallVector<uint8_t> Add = assembleSingleInst(
          "s_add_co_u32 " + CursorLow + ", " + CursorLow + ", 0x" +
              utohexstr(static_cast<uint32_t>(Delta)),
          Ctx.LS);
      if (Add.empty() || Add.size() > 3 * MinInstSize)
        return fail("source-PC cursor add encoding failed");
      Bytes.append(Add);
      while (Add.size() < 3 * MinInstSize) {
        Bytes.append(Ctx.LS.SNopBytes);
        Add.append(Ctx.LS.SNopBytes);
      }
      if (!appendInst("s_cmp_eq_u32 " + SourceLow + ", " + CursorLow))
        return fail("source-PC compare encoding failed");
      uint64_t BranchFrom = DispatcherOffset + Bytes.size();
      SmallVector<uint8_t> Branch =
          encodeScc1Branch(Ctx.LS, BranchFrom, StubBase + J * 2 * MinInstSize);
      if (Branch.size() != MinInstSize)
        return fail("source-PC conditional branch is out of range");
      Bytes.append(Branch);
      CursorValue = SourcePc;
    }
    if (!appendInst("s_trap 2"))
      return fail("unmatched-source trap encoding failed");
    for (size_t J = 0; J != Members.size(); ++J) {
      if (!appendInst("s_cmp_lg_u32 " + Save + ", 0"))
        return fail("SCC restore encoding failed");
      uint64_t BranchFrom = DispatcherOffset + Bytes.size();
      SmallVector<uint8_t> Branch =
          Ctx.LS.encodeSBranch(BranchFrom, BodyOffsets[J]);
      if (Branch.size() != MinInstSize)
        return fail("selected trampoline body is out of branch range");
      Bytes.append(Branch);
    }
    if (Bytes.size() != Owner.PoolEntryPrefixBytes)
      return fail("dispatcher size differs from reserved prefix");
    std::memcpy(Owner.Bytes.data(), Bytes.data(), Bytes.size());
  }
  return true;
}

static std::optional<SmallVector<uint64_t, 4>>
allocateBackwardBranchIslands(std::vector<NopSled> &Gateways,
                              uint64_t OwnerOffset, uint64_t FromOffset,
                              uint64_t TargetOffset,
                              BranchIslandFailure *Failure = nullptr,
                              BranchGatewayHeadSet *PersistentHeads = nullptr,
                              DenseSet<uint64_t> *PersistentOccupied = nullptr,
                              BranchIslandPromoter Promote = {}) {
  struct Allocation {
    size_t SledIndex = 0;
    uint64_t PreviousWritePos = 0;
  };
  BranchGatewayHeadSet LocalHeads;
  DenseSet<uint64_t> LocalOccupied;
  if (!PersistentOccupied)
    PersistentOccupied = &LocalOccupied;
  if (!PersistentHeads) {
    LocalHeads =
        buildBranchGatewayHeads(Gateways, *PersistentOccupied);
    PersistentHeads = &LocalHeads;
  }
  BranchGatewayHeadSet &Heads = *PersistentHeads;
  DenseSet<uint64_t> &Occupied = *PersistentOccupied;
  SmallVector<Allocation, 4> Allocations;
  SmallVector<uint64_t, 4> Islands;
  uint64_t Current = FromOffset;

  while (!isSBranchReachable(Current, TargetOffset)) {
    size_t BestIndex = Gateways.size();
    uint64_t BestOffset = std::numeric_limits<uint64_t>::max();
    uint64_t ReachBegin =
        Current > MaxSledDistance ? Current - MaxSledDistance : 0;
    uint64_t Lower =
        TargetOffset == std::numeric_limits<uint64_t>::max()
            ? TargetOffset
            : TargetOffset + 1;
    Lower = std::max(Lower, ReachBegin);
    for (auto It = Heads.lower_bound({Lower, 0});
         It != Heads.end() && It->first < Current; ++It) {
      const NopSled &Sled = Gateways[It->second];
      if (OwnerOffset < Sled.FunctionStart ||
          OwnerOffset >= Sled.FunctionEnd || Sled.WritePos != It->first ||
          !isSBranchReachable(Current, It->first))
        continue;
      BestIndex = It->second;
      BestOffset = It->first;
      break;
    }

    if (BestIndex == Gateways.size()) {
      uint64_t Corridor = TargetOffset;
      auto CorridorIt = Heads.lower_bound({Current, 0});
      while (CorridorIt != Heads.begin()) {
        --CorridorIt;
        if (CorridorIt->first <= TargetOffset)
          break;
        const NopSled &Sled = Gateways[CorridorIt->second];
        if (OwnerOffset < Sled.FunctionStart ||
            OwnerOffset >= Sled.FunctionEnd)
          continue;
        Corridor = CorridorIt->first;
        break;
      }
      BranchIslandFailure ThisFailure{Current, TargetOffset, Corridor,
                                      /*Forward=*/false};
      if (Failure)
        *Failure = ThisFailure;
      // Keep the partial chain transactional while recoverable capacity is
      // added. There are no live vector references or set iterators here.
      if (Promote && Promote(ThisFailure))
        continue;

      uint64_t EligibleOwnerSleds = 0;
      uint64_t FreeCorridorSleds = 0;
      uint64_t ReachableFromCurrent = 0;
      uint64_t ReachableToTarget = 0;
      uint64_t LowestCorridor = std::numeric_limits<uint64_t>::max();
      uint64_t HighestCorridor = 0;
      uint64_t LowestReachableFromCurrent =
          std::numeric_limits<uint64_t>::max();
      uint64_t HighestReachableToTarget = 0;
      for (const BranchGatewayHead &Head : Heads) {
        const NopSled &Sled = Gateways[Head.second];
        if (OwnerOffset < Sled.FunctionStart ||
            OwnerOffset >= Sled.FunctionEnd)
          continue;
        ++EligibleOwnerSleds;
        if (Head.first <= TargetOffset || Head.first >= Current)
          continue;
        ++FreeCorridorSleds;
        LowestCorridor = std::min(LowestCorridor, Head.first);
        HighestCorridor = std::max(HighestCorridor, Head.first);
        if (isSBranchReachable(Current, Head.first)) {
          ++ReachableFromCurrent;
          LowestReachableFromCurrent =
              std::min(LowestReachableFromCurrent, Head.first);
        }
        if (isSBranchReachable(Head.first, TargetOffset)) {
          ++ReachableToTarget;
          HighestReachableToTarget =
              std::max(HighestReachableToTarget, Head.first);
        }
      }
      if (Failure)
        *Failure = {Current, TargetOffset,
                    HighestCorridor ? HighestCorridor : TargetOffset,
                    /*Forward=*/false};
      log() << "hotswap: backward return allocator stranded owner 0x"
            << utohexstr(OwnerOffset) << ": from=0x"
            << utohexstr(FromOffset) << ", current=0x"
            << utohexstr(Current) << ", target=0x"
            << utohexstr(TargetOffset) << ", selected=" << Islands.size()
            << ", owner-sleds=" << EligibleOwnerSleds
            << ", free-corridor=" << FreeCorridorSleds
            << ", reachable-from-current=" << ReachableFromCurrent
            << ", reachable-to-target=" << ReachableToTarget;
      if (LowestCorridor != std::numeric_limits<uint64_t>::max())
        log() << ", corridor=[0x" << utohexstr(LowestCorridor) << ",0x"
              << utohexstr(HighestCorridor) << "]";
      if (LowestReachableFromCurrent != std::numeric_limits<uint64_t>::max())
        log() << ", lowest-reachable-from-current=0x"
              << utohexstr(LowestReachableFromCurrent);
      if (HighestReachableToTarget)
        log() << ", highest-reachable-to-target=0x"
              << utohexstr(HighestReachableToTarget);
      log() << "\n";
      for (size_t I = Allocations.size(); I != 0; --I) {
        const Allocation &A = Allocations[I - 1];
        Gateways[A.SledIndex].WritePos = A.PreviousWritePos;
        Heads.insert({A.PreviousWritePos, A.SledIndex});
      }
      for (uint64_t Offset : Islands)
        Occupied.erase(Offset);
      return std::nullopt;
    }

    auto AliasBegin = Heads.lower_bound({BestOffset, 0});
    auto AliasEnd =
        Heads.upper_bound(
            {BestOffset, std::numeric_limits<size_t>::max()});
    for (auto It = AliasBegin; It != AliasEnd; ++It) {
      NopSled &Alias = Gateways[It->second];
      Allocations.push_back({It->second, Alias.WritePos});
      Alias.WritePos += MinInstSize;
    }
    Heads.erase(AliasBegin, AliasEnd);
    Occupied.insert(BestOffset);
    Islands.push_back(BestOffset);
    Current = BestOffset;
  }
  for (const Allocation &A : Allocations) {
    NopSled &Sled = Gateways[A.SledIndex];
    while (hasFreeBranchGatewaySlot(Sled) &&
           Occupied.contains(Sled.WritePos))
      Sled.WritePos += MinInstSize;
    if (hasFreeBranchGatewaySlot(Sled))
      Heads.insert({Gateways[A.SledIndex].WritePos, A.SledIndex});
  }
  return Islands;
}

BranchIslandAllocatorTestResult runBranchIslandAllocatorForTest(
    std::vector<NopSled> Gateways, uint64_t OwnerOffset, uint64_t FromOffset,
    uint64_t TargetOffset, bool Backward, DenseSet<uint64_t> Occupied) {
  BranchGatewayHeadSet Heads =
      buildBranchGatewayHeads(Gateways, Occupied);
  std::optional<SmallVector<uint64_t, 4>> Islands =
      Backward
          ? allocateBackwardBranchIslands(
                Gateways, OwnerOffset, FromOffset, TargetOffset, nullptr,
                &Heads, &Occupied)
          : allocateForwardBranchIslands(
                Gateways, FromOffset, TargetOffset, nullptr, &Heads,
                &Occupied);
  BranchIslandAllocatorTestResult Result;
  Result.Success = Islands.has_value();
  if (Islands)
    Result.Islands = std::move(*Islands);
  Result.Gateways = std::move(Gateways);
  Result.Occupied = std::move(Occupied);
  return Result;
}

BranchIslandAllocatorTestResult
runBranchIslandAllocatorWithPromotionsForTest(
    std::vector<NopSled> Gateways, uint64_t OwnerOffset, uint64_t FromOffset,
    uint64_t TargetOffset, bool Backward, ArrayRef<NopSled> Promotions) {
  DenseSet<uint64_t> Occupied;
  BranchGatewayHeadSet Heads =
      buildBranchGatewayHeads(Gateways, Occupied);
  SmallVector<size_t, 4> HeldCounts;
  size_t NextPromotion = 0;
  BranchIslandPromoter Promote = [&](const BranchIslandFailure &) {
    if (NextPromotion == Promotions.size())
      return false;
    HeldCounts.push_back(Occupied.size());
    size_t Index = Gateways.size();
    Gateways.push_back(Promotions[NextPromotion++]);
    NopSled &Added = Gateways.back();
    while (hasFreeBranchGatewaySlot(Added) &&
           Occupied.contains(Added.WritePos))
      Added.WritePos += MinInstSize;
    if (hasFreeBranchGatewaySlot(Added))
      Heads.insert({Added.WritePos, Index});
    return true;
  };
  std::optional<SmallVector<uint64_t, 4>> Islands =
      Backward
          ? allocateBackwardBranchIslands(
                Gateways, OwnerOffset, FromOffset, TargetOffset, nullptr,
                &Heads, &Occupied, Promote)
          : allocateForwardBranchIslands(
                Gateways, FromOffset, TargetOffset, nullptr, &Heads,
                &Occupied, Promote);
  BranchIslandAllocatorTestResult Result;
  Result.Success = Islands.has_value();
  if (Islands)
    Result.Islands = std::move(*Islands);
  Result.HeldIslandCountsAtPromotion = std::move(HeldCounts);
  Result.Gateways = std::move(Gateways);
  Result.Occupied = std::move(Occupied);
  return Result;
}

std::vector<NopSled> subtractOccupiedBranchGatewaySlotsForTest(
    std::vector<NopSled> Gateways, const DenseSet<uint64_t> &Occupied) {
  subtractOccupiedBranchGatewaySlots(Gateways, Occupied);
  return Gateways;
}

std::pair<uint64_t, uint64_t>
branchPromotionSearchRangeForTest(uint64_t CurrentOffset,
                                  uint64_t CorridorOffset, bool Forward) {
  constexpr uint64_t MinSourceBytes =
      SetPcForwardSequenceBytes + MinInstSize;
  if (Forward) {
    uint64_t ReachEnd =
        CurrentOffset >
                std::numeric_limits<uint64_t>::max() - MaxSledDistance
            ? std::numeric_limits<uint64_t>::max()
            : CurrentOffset + MaxSledDistance;
    return {CurrentOffset, std::min(CorridorOffset, ReachEnd)};
  }

  uint64_t CorridorBegin =
      CorridorOffset > MinSourceBytes ? CorridorOffset - MinSourceBytes : 0;
  constexpr uint64_t ReachWithPrefix =
      MaxSledDistance + SetPcForwardSequenceBytes;
  uint64_t ReachBegin =
      CurrentOffset > ReachWithPrefix ? CurrentOffset - ReachWithPrefix : 0;
  return {std::max(CorridorBegin, ReachBegin), CurrentOffset};
}

static int findNextPromotionCandidateIndex(
    const BitVector &StillPossible, size_t BeginIndex, size_t EndIndex,
    bool Forward, std::optional<size_t> Previous = std::nullopt) {
  BeginIndex = std::min(BeginIndex, static_cast<size_t>(StillPossible.size()));
  EndIndex = std::min(EndIndex, static_cast<size_t>(StillPossible.size()));
  if (BeginIndex >= EndIndex)
    return -1;

  int Found = -1;
  if (Forward) {
    size_t Before = Previous.value_or(EndIndex);
    if (Before == 0)
      return -1;
    Found = StillPossible.find_prev(static_cast<unsigned>(Before));
    if (Found < 0 || static_cast<size_t>(Found) < BeginIndex)
      return -1;
  } else {
    if (Previous)
      Found = StillPossible.find_next(static_cast<unsigned>(*Previous));
    else if (BeginIndex == 0)
      Found = StillPossible.find_first();
    else
      Found =
          StillPossible.find_next(static_cast<unsigned>(BeginIndex - 1));
    if (Found < 0 || static_cast<size_t>(Found) >= EndIndex)
      return -1;
  }
  return Found;
}

SmallVector<size_t, 8> promotionCandidateOrderForTest(
    size_t CandidateCount, ArrayRef<size_t> PermanentlyRejected,
    size_t BeginIndex, size_t EndIndex, bool Forward) {
  BitVector StillPossible(CandidateCount, true);
  for (size_t Index : PermanentlyRejected)
    if (Index < CandidateCount)
      StillPossible.reset(Index);

  SmallVector<size_t, 8> Order;
  std::optional<size_t> Previous;
  while (true) {
    int Found = findNextPromotionCandidateIndex(
        StillPossible, BeginIndex, EndIndex, Forward, Previous);
    if (Found < 0)
      break;
    size_t Index = static_cast<size_t>(Found);
    Order.push_back(Index);
    Previous = Index;
  }
  return Order;
}

static bool emitMirroredStubGateways(PatchContext &Ctx) {
  DenseMap<uint32_t, SmallVector<size_t, 32>> Groups;
  SmallVector<uint64_t, 64> PoolOffsets;
  uint64_t TP = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    PoolOffsets.push_back(TP);
    Trampoline &T = Ctx.OutTrampolines[I];
    if (T.UsesMirroredStubForward)
      Groups[T.MirroredStubGroup].push_back(I);
    std::optional<uint64_t> Next =
        checkedAddUint64(TP, T.Bytes.size(), "mirrored-stub final layout");
    if (!Next)
      return false;
    TP = *Next;
  }

  for (auto &KV : Groups) {
    ArrayRef<size_t> Members = KV.second;
    if (Members.empty())
      continue;
    Trampoline &Owner = Ctx.OutTrampolines[Members.front()];
    uint64_t StubBase = PoolOffsets[Members.front()];
    auto fail = [&](const Twine &Reason) {
      log() << "hotswap: error: mirrored-stub group " << KV.first << " at 0x"
            << utohexstr(StubBase) << ": " << Reason << "\n";
      return false;
    };

    uint64_t MinSource = Owner.OriginalOffset;
    for (size_t Member : Members)
      MinSource =
          std::min(MinSource, Ctx.OutTrampolines[Member].OriginalOffset);
    std::optional<uint64_t> SourcePc = checkedAddUint64(
        MinSource, MinInstSize, "mirrored-stub minimum source PC");
    if (!SourcePc)
      return false;
    const std::string Pair =
        Owner.LongBranchUsesVcc
            ? "vcc"
            : "s[" + std::to_string(Owner.LongBranchSgprBase) + ":" +
                  std::to_string(Owner.LongBranchSgprBase + 1) + "]";
    if (StubBase < *SourcePc)
      return fail("common gateway target precedes its source-PC base");
    uint64_t Delta = StubBase - *SourcePc;
    if (Delta > std::numeric_limits<uint32_t>::max())
      return fail("common gateway delta does not fit literal32");
    SmallVector<std::string, 2> GatewayLines;
    GatewayLines.push_back("s_add_nc_u64 " + Pair + ", " + Pair + ", 0x" +
                           utohexstr(Delta));
    GatewayLines.push_back("s_set_pc_i64 " + Pair);
    Owner.ForwardGatewayBytes =
        assembleInstructions(joinAsmLines(GatewayLines), Ctx.LS);
    if (Owner.ForwardGatewayBytes.empty() ||
        Owner.ForwardGatewayBytes.size() > 3 * MinInstSize)
      return fail("common add/set-PC gateway encoding failed");
    Owner.HasForwardGateway = true;
    Owner.ForwardGatewayOffset = Owner.MirroredStubGatewayOffset;

    if (Owner.PoolEntryPrefixBytes < MinInstSize ||
        Owner.PoolEntryPrefixBytes > Owner.Bytes.size())
      return fail("sparse stub prefix reservation is invalid");
    for (uint64_t Offset = 0;
         Offset + MinInstSize <= Owner.PoolEntryPrefixBytes;
         Offset += MinInstSize)
      std::memcpy(Owner.Bytes.data() + Offset, Ctx.LS.SNopBytes.data(),
                  MinInstSize);

    for (size_t Member : Members) {
      const Trampoline &T = Ctx.OutTrampolines[Member];
      uint64_t StubDisplacement = T.OriginalOffset - MinSource;
      if (StubDisplacement > Owner.PoolEntryPrefixBytes - MinInstSize ||
          StubDisplacement + MinInstSize > Owner.Bytes.size())
        return fail("source-selected stub lies outside the sparse prefix");
      uint64_t BodyOffset = PoolOffsets[Member] + T.PoolEntryPrefixBytes;
      SmallVector<uint8_t> Branch =
          Ctx.LS.encodeSBranch(StubBase + StubDisplacement, BodyOffset);
      if (Branch.size() != MinInstSize)
        return fail("source-selected body branch is out of range");
      std::memcpy(Owner.Bytes.data() + StubDisplacement, Branch.data(),
                  Branch.size());
    }
  }
  return true;
}

static std::optional<uint64_t>
sourceTailBranchTarget(const Trampoline &T, uint64_t Offset) {
  for (const auto &[RelayOffset, TargetOffset] :
       T.SourceTailBranchIslands)
    if (RelayOffset == Offset)
      return TargetOffset;
  return std::nullopt;
}

static bool recordSourceTailBranchIsland(Trampoline &T, uint64_t Offset,
                                         uint64_t Target) {
  if (std::optional<uint64_t> Existing =
          sourceTailBranchTarget(T, Offset))
    return *Existing == Target;
  T.SourceTailBranchIslands.push_back({Offset, Target});
  return true;
}

static bool
assignLongBranchGateways(PatchContext &Ctx,
                         const DenseSet<uint64_t> &DirectBranchTargets,
                         bool AllowTextGateways) {
  struct PoolIslandOwner {
    size_t TrampolineIndex = 0;
    uint64_t RelativeOffset = 0;
  };
  std::vector<NopSled> Gateways;
  DenseMap<uint64_t, PoolIslandOwner> PoolIslandOwners;
  DenseMap<uint64_t, size_t> SourceTailIslandOwners;
  DenseMap<uint64_t, size_t> SourceTailGatewayOwners;
  DenseMap<uint64_t, uint64_t> EarlySourceTailOwnerOffsets;
  DenseMap<uint64_t, uint64_t> EarlySourceGatewayOwnerOffsets;
  DenseMap<uint64_t, uint64_t> ReservedBackboneOwnerOffsets;
  DenseMap<uint64_t, uint64_t> LateDirectSuffixOwnerOffsets;
  const std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Ctx.Elf.functionTextRanges();
  const FunctionRangeUniquenessIndex FunctionRangeIndex(FunctionRanges,
                                                          Ctx.Elf.textAddr());
  auto HasSafeTail = [&](const Trampoline &T, uint64_t Begin, uint64_t End) {
    bool HasUniqueFunctionRange =
        FunctionRangeIndex.hasUniqueFunctionRange(T);
    return isSafeSourceTailRange(T, Ctx.DirectControlFlow,
                                 HasUniqueFunctionRange, Begin, End);
  };
  if (AllowTextGateways) {
    Gateways = buildExternalGatewaySleds(
        Ctx.Decoded, Ctx.LS, Ctx.Elf, ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize),
        DirectBranchTargets);
    for (const NopSled &Sled : Ctx.NopSleds)
      Gateways.push_back(Sled);
    subtractTrampolineSources(Gateways, Ctx.OutTrampolines);

    // Keep one 12-byte tail from each safe padding window available for the
    // pair-only affine gateway. The four-register planner otherwise consumes
    // these scarce tails as one-dword branch islands before it discovers the
    // pair-only residuals. Its 20-byte gateways continue to use the prefix.
    bool HasPairOnlyAffineDemand = false;
    uint64_t CandidateTP = Ctx.PoolBaseOffset;
    for (const Trampoline &T : Ctx.OutTrampolines) {
      uint64_t ThisTP = CandidateTP;
      std::optional<uint64_t> Next = checkedAddUint64(
          CandidateTP, T.Bytes.size(), "affine reserve demand pool layout");
      if (!Next)
        return false;
      CandidateTP = *Next;
      if (!T.Long || !T.UsesSetPCBack || T.LongBranchPreservesVcc ||
          T.UsesSharedDispatcherForward || T.UsesMirroredStubForward ||
          T.OriginalSize < 2 * MinInstSize ||
          isSBranchReachable(T.OriginalOffset, ThisTP))
        continue;
      std::optional<SmallVector<uint8_t>> Direct =
          encodeSetPCLongBranch(Ctx.LS, T.OriginalOffset, ThisTP,
                                T.LongBranchSgprBase, T.LongBranchUsesVcc);
      if (Direct && Direct->size() <= T.OriginalSize)
        continue;
      if (!findSafeSgprScratchBlock(Ctx, T.OriginalOffset, /*Count=*/4,
                                    /*Alignment=*/2, "affine reserve demand",
                                    /*ReportNoSpace=*/false)) {
        HasPairOnlyAffineDemand = true;
        break;
      }
    }

    std::vector<NopSled> MirroredGatewayReserves;
    if (HasPairOnlyAffineDemand) {
      constexpr uint64_t ReserveBytes = 3 * MinInstSize;
      for (NopSled &Sled : Gateways) {
        uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
        if (Sled.WritePos > UsableEnd ||
            ReserveBytes > UsableEnd - Sled.WritePos)
          continue;
        uint64_t ReserveStart = UsableEnd - ReserveBytes;
        MirroredGatewayReserves.push_back({ReserveStart, UsableEnd,
                                           ReserveStart, Sled.FunctionStart,
                                           Sled.FunctionEnd});
        Sled.End = ReserveStart;
      }
    }

    const auto SharedPlanStart = std::chrono::steady_clock::now();
    if (!planSharedDispatchGateways(Ctx, Gateways))
      return false;
    log() << "hotswap: shared gateway planning took "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - SharedPlanStart)
                 .count()
          << " ms\n";
    Gateways.insert(Gateways.end(), MirroredGatewayReserves.begin(),
                    MirroredGatewayReserves.end());

    // Collect pair-backed source tails into one sparse backbone. These tails
    // are withheld from affine planning below, then reactivated after
    // forward-mode selection proves that their second dword is unreachable.
    SmallVector<std::pair<uint64_t, uint64_t>, 64>
        SourceTailBackboneCandidates;

    // Shared-dispatch sources can also use one-dword direct calls: their link
    // pair is exactly the source-PC identity consumed by the dispatcher. Keep
    // tails that the shared planner already selected as relays on that route,
    // collect independent tails for the sparse backbone, and
    // expose every other tail to the later affine planner.
    DenseSet<uint64_t> RequiredSharedRelayTails;
    for (const Trampoline &T : Ctx.OutTrampolines)
      if (T.UsesSharedDispatcherForward && T.SharedDispatcherRelayOffset != 0)
        RequiredSharedRelayTails.insert(T.SharedDispatcherRelayOffset);
    for (Trampoline &T : Ctx.OutTrampolines) {
      if (!T.UsesSharedDispatcherForward || T.OriginalSize < 2 * MinInstSize ||
          T.LongBranchPreservesVcc)
        continue;
      uint64_t Tail = T.OriginalOffset + MinInstSize;
      if (!HasSafeTail(T, Tail, Tail + MinInstSize))
        continue;
      uint64_t Route =
          T.SharedDispatcherRelayOffset    ? T.SharedDispatcherRelayOffset
          : T.ForwardBranchIslands.empty() ? T.SharedDispatcherGatewayOffset
                                           : T.ForwardBranchIslands.front();
      if (RequiredSharedRelayTails.contains(Tail)) {
        if (!recordSourceTailBranchIsland(T, Tail, Route))
          return false;
        continue;
      }
      SourceTailBackboneCandidates.push_back({Tail, T.OriginalOffset});
      continue;
    }

    // Shared-dispatch sources reserve one direct call in their first dword,
    // while their second dword remains available as an independent relay.
    // Expose up to five dwords after that relay: three hold an affine gateway,
    // while five can hold the save-VCC + set-PC sequence needed by live-VCC
    // sources. A five-dword window has only eight bytes left after an affine
    // allocation, so every source owner still hosts at most one gateway.
    for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
      const Trampoline &T = Ctx.OutTrampolines[I];
      constexpr uint64_t ForwardBytes = 2 * MinInstSize;
      constexpr uint64_t AffineGatewayBytes = 3 * MinInstSize;
      constexpr uint64_t VccGatewayBytes = 5 * MinInstSize;
      if (!T.UsesSharedDispatcherForward ||
          T.OriginalSize < ForwardBytes + AffineGatewayBytes)
        continue;
      uint64_t GatewayBytes = T.OriginalSize >= ForwardBytes + VccGatewayBytes
                                  ? VccGatewayBytes
                                  : AffineGatewayBytes;
      uint64_t Start = T.OriginalOffset + ForwardBytes;
      if (!HasSafeTail(T, Start, Start + GatewayBytes))
        continue;
      EarlySourceGatewayOwnerOffsets[Start] = T.OriginalOffset;
      Gateways.push_back({Start, Start + GatewayBytes, Start, 0,
                          std::numeric_limits<uint64_t>::max(),
                          /*GatewayOnly=*/true});
    }

    // s_call_i64 records source+4 in the reserved pair and transfers control
    // in one dword. Most now-unreachable second dwords are offered to affine
    // planning. Withhold a sparse 96-KiB backbone, however: affine routes can
    // consume every relay in a dense text interval and leave no path for a
    // later registerless return. Only pair-backed sources with a proven-safe
    // second dword participate, so a member that receives a mirrored
    // one-dword call has an independently-proven unreachable +4 dword.
    CandidateTP = Ctx.PoolBaseOffset;
    for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
      const Trampoline &T = Ctx.OutTrampolines[I];
      uint64_t ThisTP = CandidateTP;
      std::optional<uint64_t> Next = checkedAddUint64(
          CandidateTP, T.Bytes.size(), "affine candidate pool layout");
      if (!Next)
        return false;
      CandidateTP = *Next;
      if (!T.Long || !T.UsesSetPCBack || T.LongBranchPreservesVcc ||
          T.UsesSharedDispatcherForward || T.OriginalSize < 2 * MinInstSize ||
          isSBranchReachable(T.OriginalOffset, ThisTP))
        continue;
      std::optional<SmallVector<uint8_t>> Direct =
          encodeSetPCLongBranch(Ctx.LS, T.OriginalOffset, ThisTP,
                                T.LongBranchSgprBase, T.LongBranchUsesVcc);
      if (Direct && Direct->size() <= T.OriginalSize) {
        // A direct set-PC sequence consumes only its prefix. Once ordinary
        // mode selection commits that sequence, a safe dword in the remaining
        // source window is unreachable and can serve the same sparse
        // backbone. Keep at most one suffix candidate per owner.
        if (!T.LongBranchPreservesVcc &&
            Direct->size() + MinInstSize <= T.OriginalSize) {
          for (uint64_t RelativeOffset = Direct->size();
               RelativeOffset + MinInstSize <= T.OriginalSize;
               RelativeOffset += MinInstSize) {
            uint64_t Suffix = T.OriginalOffset + RelativeOffset;
            if (!HasSafeTail(T, Suffix, Suffix + MinInstSize))
              continue;
            LateDirectSuffixOwnerOffsets[Suffix] = T.OriginalOffset;
            break;
          }
        }
        continue;
      }
      uint64_t Tail = T.OriginalOffset + MinInstSize;
      if (!HasSafeTail(T, Tail, Tail + MinInstSize))
        continue;
      SourceTailBackboneCandidates.push_back({Tail, T.OriginalOffset});
      continue;
    }
    llvm::sort(SourceTailBackboneCandidates);
    SourceTailBackboneCandidates.erase(
        std::unique(SourceTailBackboneCandidates.begin(),
                    SourceTailBackboneCandidates.end()),
        SourceTailBackboneCandidates.end());
    constexpr uint64_t SourceTailBackboneSpacing = 96 * 1024;
    BitVector ReservedSourceTails(SourceTailBackboneCandidates.size());
    if (!SourceTailBackboneCandidates.empty()) {
      ReservedSourceTails.set(0);
      size_t LastReserved = 0;
      for (size_t I = 1; I != SourceTailBackboneCandidates.size(); ++I) {
        if (SourceTailBackboneCandidates[I].first -
                SourceTailBackboneCandidates[LastReserved].first <=
            SourceTailBackboneSpacing)
          continue;
        ReservedSourceTails.set(I - 1);
        LastReserved = I - 1;
        // A gap between adjacent candidates cannot be filled by this
        // backbone. Preserve both sides so other stable text/pool relays can
        // still connect to the nearest available endpoint.
        if (SourceTailBackboneCandidates[I].first -
                SourceTailBackboneCandidates[LastReserved].first >
            SourceTailBackboneSpacing) {
          ReservedSourceTails.set(I);
          LastReserved = I;
        }
      }
      ReservedSourceTails.set(SourceTailBackboneCandidates.size() - 1);
    }
    for (size_t I = 0; I != SourceTailBackboneCandidates.size(); ++I) {
      const auto &[Tail, Owner] = SourceTailBackboneCandidates[I];
      if (ReservedSourceTails.test(I)) {
        ReservedBackboneOwnerOffsets[Tail] = Owner;
      } else {
        EarlySourceTailOwnerOffsets[Tail] = Owner;
        Gateways.push_back({Tail, Tail + MinInstSize, Tail, 0,
                            std::numeric_limits<uint64_t>::max()});
      }
    }

    // A registerless far source needs its second dword for its own return
    // chain. Do not expose that owner tail to the affine planner: a greedy
    // forward chain could consume it and strand the source before return
    // allocation runs. Bytes after the reserved second dword remain eligible
    // as a source-local gateway when the original window is large enough.
    for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
      const Trampoline &T = Ctx.OutTrampolines[I];
      std::optional<std::pair<uint64_t, uint64_t>> Range =
          registerlessSourceAffineGatewayRange(T);
      if (!Range)
        continue;
      uint64_t Start = Range->first;
      uint64_t End = Range->second;
      if (!HasSafeTail(T, Start, End))
        continue;
      EarlySourceGatewayOwnerOffsets[Start] = T.OriginalOffset;
      Gateways.push_back({Start, End, Start, 0,
                          std::numeric_limits<uint64_t>::max(),
                          /*GatewayOnly=*/true});
    }

    if (!planMirroredStubGateways(Ctx, Gateways))
      return false;
    DenseMap<uint64_t, size_t> TrampolineAtOriginalOffset;
    for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I)
      TrampolineAtOriginalOffset[Ctx.OutTrampolines[I].OriginalOffset] = I;
    for (const auto &KV : EarlySourceTailOwnerOffsets) {
      DenseMap<uint64_t, size_t>::const_iterator Owner =
          TrampolineAtOriginalOffset.find(KV.second);
      if (Owner != TrampolineAtOriginalOffset.end())
        SourceTailIslandOwners[KV.first] = Owner->second;
    }
    for (const auto &KV : EarlySourceGatewayOwnerOffsets) {
      DenseMap<uint64_t, size_t>::const_iterator Owner =
          TrampolineAtOriginalOffset.find(KV.second);
      if (Owner != TrampolineAtOriginalOffset.end())
        SourceTailGatewayOwners[KV.first] = Owner->second;
    }
    for (const Trampoline &T : Ctx.OutTrampolines) {
      if (!T.UsesMirroredStubForward)
        continue;
      for (size_t I = 0; I != T.ForwardBranchIslands.size(); ++I) {
        uint64_t From = T.ForwardBranchIslands[I];
        DenseMap<uint64_t, size_t>::const_iterator Owner =
            SourceTailIslandOwners.find(From);
        if (Owner == SourceTailIslandOwners.end())
          continue;
        Trampoline &OwnerT = Ctx.OutTrampolines[Owner->second];
        if (OwnerT.UsesSetPCBack && !OwnerT.UsesMirroredStubForward &&
            !OwnerT.UsesSharedDispatcherForward) {
          log() << "hotswap: error: affine relay owner at 0x"
                << utohexstr(OwnerT.OriginalOffset)
                << " did not receive a mirrored route\n";
          return false;
        }
        uint64_t To = I + 1 == T.ForwardBranchIslands.size()
                          ? T.ForwardBranchTargetOffset
                          : T.ForwardBranchIslands[I + 1];
        if (!recordSourceTailBranchIsland(OwnerT, From, To)) {
          log() << "hotswap: error: affine source tail at 0x" << utohexstr(From)
                << " has conflicting relay targets\n";
          return false;
        }
      }
    }
    for (const Trampoline &T : Ctx.OutTrampolines) {
      if (!T.UsesMirroredStubForward)
        continue;
      DenseMap<uint64_t, size_t>::const_iterator Owner =
          SourceTailGatewayOwners.find(T.MirroredStubGatewayOffset);
      if (Owner == SourceTailGatewayOwners.end())
        continue;
      Trampoline &OwnerT = Ctx.OutTrampolines[Owner->second];
      OwnerT.HasSourceTailGateway = true;
      OwnerT.SourceTailGatewayOffset = T.MirroredStubGatewayOffset;
      OwnerT.SourceTailGatewayBytes = 3 * MinInstSize;
    }
    const auto DispatcherEmitStart = std::chrono::steady_clock::now();
    if (!emitSharedDispatchers(Ctx) || !emitMirroredStubGateways(Ctx))
      return false;
    log() << "hotswap: shared/affine gateway emission took "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - DispatcherEmitStart)
                 .count()
          << " ms\n";
  }

  uint64_t IslandLayoutOffset = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    std::optional<uint64_t> Next = checkedAddUint64(
        IslandLayoutOffset, T.Bytes.size(), "pool branch-island layout");
    if (!Next)
      return false;
    if (T.HasPoolBranchIsland) {
      T.PoolBranchIslandOffset = *Next - PoolBranchIslandBytes;
      PoolIslandOwners[T.PoolBranchIslandOffset] = {
          I, T.Bytes.size() - PoolBranchIslandBytes};
      Gateways.push_back({T.PoolBranchIslandOffset,
                          T.PoolBranchIslandOffset + PoolBranchIslandBytes,
                          T.PoolBranchIslandOffset, 0,
                          std::numeric_limits<uint64_t>::max()});
    }
    IslandLayoutOffset = *Next;
  }

  struct PendingGateway {
    size_t TrampolineIndex = 0;
    uint64_t TargetOffset = 0;
    uint64_t InitialCandidateSlots = 0;
  };
  std::vector<PendingGateway> Pending;
  const auto OrdinaryPlanStart = std::chrono::steady_clock::now();
  uint64_t ReturnBranchIslandChains = 0;
  uint64_t TrampOffset = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    uint64_t TP = TrampOffset;
    std::optional<uint64_t> Next = checkedAddUint64(
        TrampOffset, T.Bytes.size(), "gateway trampoline layout");
    if (!Next)
      return false;
    TrampOffset = *Next;
    if (!T.Long)
      continue;
    if (T.UsesSharedDispatcherForward || T.UsesMirroredStubForward)
      continue;

    if (isSBranchReachable(T.OriginalOffset, TP)) {
      T.UsesShortBranchForward = true;
      if (T.LongBranchPreservesVcc)
        std::memcpy(T.Bytes.data(), Ctx.LS.SNopBytes.data(), MinInstSize);
      continue;
    }
    if (T.UsesSetPCBack) {
      std::optional<SmallVector<uint8_t>> Direct =
          T.LongBranchPreservesVcc
              ? encodeSetPcGateway(Ctx.LS, T.OriginalOffset, TP,
                                   T.LongBranchSgprBase, T.LongBranchUsesVcc,
                                   /*PreserveVcc=*/true)
              : encodeSetPCLongBranch(Ctx.LS, T.OriginalOffset, TP,
                                      T.LongBranchSgprBase,
                                      T.LongBranchUsesVcc);
      uint64_t RequiredSourceBytes =
          Direct ? Direct->size() +
                       (T.LongBranchPreservesVcc ? VccLandingPadBytes : 0)
                 : 0;
      if (Direct && RequiredSourceBytes <= T.OriginalSize) {
        T.UsesDirectSetPCForward = true;
        T.DirectSetPCForwardBytes = std::move(*Direct);
        continue;
      }
    }
    Pending.push_back({I, TP, 0});
  }
  log() << "hotswap: ordinary gateway planner collected " << Pending.size()
        << " pending site(s) across " << Gateways.size()
        << " gateway/relay window(s) in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - OrdinaryPlanStart)
               .count()
        << " ms\n";
  uint64_t PoolEndOffset = TrampOffset;

  // Affine planning has finished and ordinary mode selection has now either
  // committed a direct set-PC source or left a pair-backed owner on its
  // one-dword branch path. Reactivate the withheld backbone only at this
  // point: its +4 dword is provably unreachable for both mirrored calls and
  // pending ordinary branches, while it could not be consumed by an earlier
  // affine route.
  DenseMap<uint64_t, size_t> FinalTrampolineAtOriginalOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I)
    FinalTrampolineAtOriginalOffset[Ctx.OutTrampolines[I].OriginalOffset] = I;
  uint64_t ActivatedBackboneRelays = 0;
  for (const auto &KV : ReservedBackboneOwnerOffsets) {
    DenseMap<uint64_t, size_t>::const_iterator Owner =
        FinalTrampolineAtOriginalOffset.find(KV.second);
    if (Owner == FinalTrampolineAtOriginalOffset.end())
      continue;
    const Trampoline &OwnerT = Ctx.OutTrampolines[Owner->second];
    if (KV.first < OwnerT.OriginalOffset)
      continue;
    uint64_t RelativeOffset = KV.first - OwnerT.OriginalOffset;
    bool HasOneDwordForward =
        RelativeOffset == MinInstSize &&
        (OwnerT.UsesSharedDispatcherForward ||
         (OwnerT.UsesSetPCBack && !OwnerT.UsesDirectSetPCForward));
    if (!OwnerT.Long || OwnerT.LongBranchPreservesVcc ||
        !HasOneDwordForward ||
        !HasSafeTail(OwnerT, KV.first, KV.first + MinInstSize) ||
        sourceTailBranchTarget(OwnerT, KV.first))
      continue;
    SourceTailIslandOwners[KV.first] = Owner->second;
    Gateways.push_back({KV.first, KV.first + MinInstSize, KV.first, 0,
                        std::numeric_limits<uint64_t>::max()});
    ++ActivatedBackboneRelays;
  }
  uint64_t ActivatedDirectSuffixRelays = 0;
  for (const auto &KV : LateDirectSuffixOwnerOffsets) {
    DenseMap<uint64_t, size_t>::const_iterator Owner =
        FinalTrampolineAtOriginalOffset.find(KV.second);
    if (Owner == FinalTrampolineAtOriginalOffset.end())
      continue;
    const Trampoline &OwnerT = Ctx.OutTrampolines[Owner->second];
    if (KV.first < OwnerT.OriginalOffset)
      continue;
    uint64_t RelativeOffset = KV.first - OwnerT.OriginalOffset;
    if (!OwnerT.Long || OwnerT.LongBranchPreservesVcc ||
        !OwnerT.UsesDirectSetPCForward ||
        RelativeOffset < OwnerT.DirectSetPCForwardBytes.size() ||
        RelativeOffset + MinInstSize > OwnerT.OriginalSize ||
        !HasSafeTail(OwnerT, KV.first, KV.first + MinInstSize) ||
        sourceTailBranchTarget(OwnerT, KV.first))
      continue;
    SourceTailIslandOwners[KV.first] = Owner->second;
    Gateways.push_back({KV.first, KV.first + MinInstSize, KV.first, 0,
                        std::numeric_limits<uint64_t>::max()});
    ++ActivatedDirectSuffixRelays;
  }
  if (ActivatedBackboneRelays != 0)
    log() << "hotswap: reserved " << ActivatedBackboneRelays
          << " source-tail backbone relay(s) from affine planning\n";
  if (ActivatedDirectSuffixRelays != 0)
    log() << "hotswap: exposed " << ActivatedDirectSuffixRelays
          << " direct set-PC source-suffix relay(s) after affine planning\n";

  // Once a source is replaced by a one-dword branch, the remainder of its
  // original instruction window is unreachable and can provide a safe relay.
  // Add these only after selecting direct set-PC sources, whose longer forward
  // sequence consumes the tail. Shared dispatch and VCC preservation likewise
  // reserve the second dword. Relays are object-wide: unlike an arbitrary NOP
  // sled they cannot be reached by the owning function's original fallthrough.
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    const Trampoline &T = Ctx.OutTrampolines[I];
    if (!AllowTextGateways ||
        Ctx.DirectControlFlow.HasUnboundedIndirectEntries ||
        T.OriginalSize < 2 * MinInstSize || T.UsesDirectSetPCForward ||
        T.UsesSharedDispatcherForward || T.UsesMirroredStubForward ||
        T.LongBranchPreservesVcc)
      continue;
    uint64_t Tail = T.OriginalOffset + MinInstSize;
    if (SourceTailIslandOwners.contains(Tail) ||
        !HasSafeTail(T, Tail, Tail + MinInstSize))
      continue;
    SourceTailIslandOwners[Tail] = I;
    Gateways.push_back({Tail, Tail + MinInstSize, Tail, 0,
                        std::numeric_limits<uint64_t>::max()});
  }

  // Forward-mode selection above exposes every source that will execute a
  // one-dword branch. Allocate registerless returns only now, so they can use
  // the complete set of unreachable source tails. Returns retain first access
  // to those relays before ordinary forward gateways and island chains.
  //
  // Reserve each registerless source's own second dword before allocating any
  // chain. That dword lies four bytes before ReturnTo, so the generic
  // monotonically-backward allocator cannot select it directly; without this
  // reservation an earlier return can consume it as an ordinary relay and
  // strand its owner. The owner's chain targets the reserved tail, whose final
  // branch then advances to ReturnTo.
  DenseMap<size_t, uint64_t> ReservedReturnTails;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    const Trampoline &T = Ctx.OutTrampolines[I];
    if (!mustReserveSourceTailForRegisterlessReturn(T))
      continue;
    uint64_t Tail = T.OriginalOffset + MinInstSize;
    DenseMap<uint64_t, size_t>::const_iterator Owner =
        SourceTailIslandOwners.find(Tail);
    if (Owner == SourceTailIslandOwners.end() || Owner->second != I)
      continue;
    for (NopSled &Gateway : Gateways) {
      if (Gateway.WritePos != Tail || Gateway.End < Tail + MinInstSize)
        continue;
      Gateway.WritePos += MinInstSize;
      ReservedReturnTails[I] = Tail;
      break;
    }
  }

  DenseSet<uint64_t> OccupiedBranchGatewaySlots;
  BranchGatewayHeadSet BranchGatewayHeads =
      buildBranchGatewayHeads(Gateways, OccupiedBranchGatewaySlots);
  uint64_t RemainingRelayDemand = 0;
  for (const Trampoline &T : Ctx.OutTrampolines)
    RemainingRelayDemand += T.Long && !T.UsesSetPCBack;
  RemainingRelayDemand += Pending.size();

  BitVector PromotionStillPossible(Ctx.Decoded.size(), true);
  DenseSet<uint64_t> PromotionProtectedOffsets =
      collectRelocationProtectedOffsets(Ctx.Decoded, Ctx.LS,
                                        /*ProtectClauseMembers=*/true);
  DenseSet<uint64_t> PromotionEntryOffsets = Ctx.DirectControlFlow.Targets;
  PromotionEntryOffsets.insert(DirectBranchTargets.begin(),
                               DirectBranchTargets.end());
  PromotionEntryOffsets.insert(Ctx.DeclaredEntries.begin(),
                               Ctx.DeclaredEntries.end());
  DenseSet<uint64_t> PromotionIndirectFunctions =
      collectIndirectControlFlowFunctions(
          Ctx.Decoded, Ctx.LS, Ctx.Elf,
          Ctx.DirectControlFlow.BoundedIndirectTransfers);
  DenseMap<std::pair<uint64_t, uint64_t>,
           std::optional<DenseSet<uint64_t>>>
      PromotionVccNeedsCache;
  std::optional<SmallVector<MCRegister, 128>> PromotionNumberedSgprs =
      resolveNumberedSgprRegisters(*Ctx.LS.MRI, Ctx.Config.MaxSgprs);
  BatchedSgprContinuationCache PromotionSgprContinuationCache;
  uint64_t PromotionSgprContinuationAnalyses = 0;
  using PromotionRange = std::pair<uint64_t, uint64_t>;
  std::set<PromotionRange> PromotionReservedRanges;
  auto ReservePromotionRange = [&](uint64_t Begin, uint64_t End) {
    if (Begin >= End)
      return;
    auto It = PromotionReservedRanges.lower_bound({Begin, 0});
    if (It != PromotionReservedRanges.begin()) {
      auto Previous = std::prev(It);
      if (Previous->second >= Begin)
        It = Previous;
    }
    while (It != PromotionReservedRanges.end() && It->first <= End) {
      Begin = std::min(Begin, It->first);
      End = std::max(End, It->second);
      It = PromotionReservedRanges.erase(It);
    }
    PromotionReservedRanges.insert(It, {Begin, End});
  };
  auto OverlapsPromotionRange = [&](uint64_t Begin, uint64_t End) {
    if (Begin >= End)
      return false;
    auto It = PromotionReservedRanges.lower_bound({End, 0});
    if (It == PromotionReservedRanges.begin())
      return false;
    --It;
    return It->second > Begin;
  };
  for (const Trampoline &T : Ctx.OutTrampolines)
    ReservePromotionRange(T.OriginalOffset,
                          T.OriginalOffset + T.OriginalSize);
  for (const NopSled &Gateway : Gateways)
    ReservePromotionRange(Gateway.Start, Gateway.End);
  auto IsVccDeadAtContinuation =
      [&](const ElfView::FunctionTextRange &Function, uint64_t Start,
          uint32_t Size) {
    std::optional<uint64_t> Continuation = checkedAddUint64(
        Start, Size, "promoted VCC liveness continuation");
    if (!Continuation || *Continuation < Function.Begin ||
        *Continuation >= Function.End)
      return false;
    auto ContinuationIt = llvm::lower_bound(
        Ctx.Decoded, *Continuation,
        [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    if (ContinuationIt == Ctx.Decoded.end() ||
        ContinuationIt->Offset != *Continuation)
      return false;

    std::pair<uint64_t, uint64_t> Key{Function.Begin, Function.End};
    auto CacheIt = PromotionVccNeedsCache.find(Key);
    if (CacheIt == PromotionVccNeedsCache.end()) {
      std::optional<DenseSet<uint64_t>> Needs =
          computeIncomingRegisterNeeds(Ctx.Decoded, Ctx.LS, Function.Begin,
                                       Function.End, Ctx.LS.VCCRegister);
      CacheIt =
          PromotionVccNeedsCache.try_emplace(Key, std::move(Needs)).first;
    }
    return CacheIt->second &&
           !CacheIt->second->contains(*Continuation);
  };
  uint64_t PromotedBranchCapacityRelays = 0;
  SmallVector<std::pair<size_t, uint64_t>, 8>
      ReturnPromotionBranchOnlyRanges;
  bool PlanningRegisterlessReturns = true;
  constexpr uint64_t MinBranchCapacitySourceBytes =
      SetPcForwardSequenceBytes + MinInstSize;
  auto PromoteBranchCapacityRelay =
      [&](const BranchIslandFailure &Failure) -> bool {
    auto [SearchBegin, SearchEnd] = branchPromotionSearchRangeForTest(
        Failure.CurrentOffset, Failure.CorridorOffset, Failure.Forward);
    if (SearchBegin >= SearchEnd)
      return false;
    auto TryPromote = [&](bool AllowLocalPair) -> bool {
      auto Begin = llvm::lower_bound(
          Ctx.Decoded, SearchBegin,
          [](const InternalDecodedInst &DI, uint64_t Offset) {
            return DI.Offset < Offset;
          });
      auto End = llvm::lower_bound(
          Ctx.Decoded, SearchEnd,
          [](const InternalDecodedInst &DI, uint64_t Offset) {
            return DI.Offset < Offset;
          });
      size_t BeginIndex = static_cast<size_t>(Begin - Ctx.Decoded.begin());
      size_t EndIndex = static_cast<size_t>(End - Ctx.Decoded.begin());
      std::optional<size_t> PreviousIndex;
      while (true) {
        int CandidateIndex = findNextPromotionCandidateIndex(
            PromotionStillPossible, BeginIndex, EndIndex, Failure.Forward,
            PreviousIndex);
        if (CandidateIndex < 0)
          break;
        size_t Index = static_cast<size_t>(CandidateIndex);
        PreviousIndex = Index;
        auto It = Ctx.Decoded.begin() + Index;
        uint64_t Start = It->Offset;

        // Scratch selection does not change the fixed set-PC width. Reject
        // offsets outside the depleted branch corridor before function,
        // overlap, or liveness analysis.
        std::optional<uint32_t> ProbeDirectSize =
            getSetPcLongBranchLayoutSize(Start, PoolEndOffset);
        if (!ProbeDirectSize) {
          PromotionStillPossible.reset(Index);
          continue;
        }
        uint64_t Tail = Start + *ProbeDirectSize;
        bool MakesReachableProgress =
            Failure.Forward
                ? Tail > Failure.CurrentOffset &&
                      Tail < Failure.CorridorOffset &&
                      Tail < Failure.TargetOffset &&
                      isSBranchReachable(Failure.CurrentOffset, Tail)
                : Tail < Failure.CurrentOffset &&
                      Tail > Failure.CorridorOffset &&
                      Tail > Failure.TargetOffset &&
                      isSBranchReachable(Failure.CurrentOffset, Tail);
        if (!MakesReachableProgress)
          continue;

        std::optional<ElfView::FunctionTextRange> Function =
            Ctx.Elf.findFunctionTextRangeAtOffset(Start);
        if (!Function ||
            PromotionIndirectFunctions.contains(Function->Begin)) {
          PromotionStillPossible.reset(Index);
          continue;
        }

        uint64_t DemandBytes =
            RemainingRelayDemand >
                    (std::numeric_limits<uint32_t>::max() -
                     *ProbeDirectSize) /
                        MinInstSize
                ? std::numeric_limits<uint32_t>::max() -
                      *ProbeDirectSize
                : RemainingRelayDemand * MinInstSize;
        uint64_t DesiredSourceBytes = *ProbeDirectSize + DemandBytes;
        DesiredSourceBytes =
            std::max(DesiredSourceBytes, MinBranchCapacitySourceBytes);
        DesiredSourceBytes =
            std::min(DesiredSourceBytes, Function->End - Start);

        SmallVector<uint8_t, 32> Replacement;
        uint64_t Cursor = Start;
        auto Member = It;
        uint32_t MinimumReplacementSize = 0;
        uint32_t BestVccReplacementSize = 0;
        bool NeedsIncomingVcc = false;
        bool FullyWritesVcc = false;
        while (Replacement.size() < DesiredSourceBytes) {
          if (Member == Ctx.Decoded.end() || Member->Offset != Cursor ||
              PromotionEntryOffsets.contains(Member->Offset))
            break;
          std::optional<ElfView::FunctionTextRange> MemberFunction =
              Ctx.Elf.findFunctionTextRangeAtOffset(Member->Offset);
          std::optional<InternalDecodedInst> Current =
              decodeCurrentInstruction(Ctx, *Member);
          if (!MemberFunction || MemberFunction->Begin != Function->Begin ||
              MemberFunction->End != Function->End || !Current ||
              !isSafeStraightLineRelocation(*Current, Ctx.LS,
                                            PromotionProtectedOffsets) ||
              Current->Inst.getOpcode() == Ctx.LS.SNopOpcode)
            break;
          std::optional<uint64_t> MemberEnd = checkedAddUint64(
              Member->Offset, Member->Size,
              "promoted branch-capacity member end");
          if (!MemberEnd || *MemberEnd > Function->End ||
              Replacement.size() + Member->Size > DesiredSourceBytes ||
              OverlapsPromotionRange(Member->Offset, *MemberEnd))
            break;

          if (!FullyWritesVcc &&
              instructionReadsRegister(*Current, Ctx.LS,
                                       Ctx.LS.VCCRegister))
            NeedsIncomingVcc = true;
          if (instructionFullyWritesRegister(*Current, Ctx.LS,
                                             Ctx.LS.VCCRegister))
            FullyWritesVcc = true;
          Replacement.append(Ctx.Text + Member->Offset,
                             Ctx.Text + Member->Offset + Member->Size);
          Cursor = *MemberEnd;
          ++Member;

          if (!MinimumReplacementSize &&
              Replacement.size() >= MinBranchCapacitySourceBytes)
            MinimumReplacementSize = Replacement.size();
          if (Replacement.size() >= MinBranchCapacitySourceBytes &&
              !NeedsIncomingVcc && Ctx.LS.VCCRegister.isValid() &&
              IsVccDeadAtContinuation(*Function, Start,
                                      Replacement.size()))
            BestVccReplacementSize = Replacement.size();
        }
        if (!MinimumReplacementSize ||
            Replacement.size() > std::numeric_limits<uint32_t>::max()) {
          PromotionStillPossible.reset(Index);
          continue;
        }
        bool UseVcc = BestVccReplacementSize != 0;
        if (!UseVcc && !AllowLocalPair)
          continue;
        Replacement.resize(UseVcc ? BestVccReplacementSize
                                  : MinimumReplacementSize);
        std::optional<unsigned> Scratch;
        if (!UseVcc && PromotionNumberedSgprs)
          Scratch = findLocallyDeadSgprPairWithCache(
              Ctx, *Function, Start, Replacement.size(), Replacement,
              *PromotionNumberedSgprs, PromotionSgprContinuationCache,
              PromotionSgprContinuationAnalyses);
        if (!Scratch && !UseVcc) {
          PromotionStillPossible.reset(Index);
          continue;
        }
        uint64_t End = Start + Replacement.size();
        Trampoline Promoted;
        Promoted.OriginalOffset = Start;
        Promoted.OriginalSize = Replacement.size();
        Promoted.HasFunctionRange = true;
        Promoted.FunctionStart = Function->Begin;
        Promoted.FunctionEnd = Function->End;
        if (!HasSafeTail(Promoted, Tail, End)) {
          PromotionStillPossible.reset(Index);
          continue;
        }
        std::optional<SmallVector<uint8_t>> Direct = encodeSetPCLongBranch(
            Ctx.LS, Start, PoolEndOffset, Scratch.value_or(0), UseVcc);
        if (!Direct || Direct->size() != *ProbeDirectSize) {
          PromotionStillPossible.reset(Index);
          continue;
        }
        if (Scratch) {
          SafeSgprScratchBlock ScratchBlock{*Scratch, 2};
          if (!commitSafeSgprScratchBlock(
                  Ctx, Start, ScratchBlock,
                  "promoted branch-capacity trampoline SGPR pair")) {
            PromotionStillPossible.reset(Index);
            continue;
          }
        }
        Promoted.Bytes.append(Replacement.begin(), Replacement.end());
        Promoted.Bytes.append(SetPcReturnReserveBytes, uint8_t{0});
        Promoted.Bytes.append(PoolBranchIslandBytes, uint8_t{0});
        Promoted.Long = true;
        Promoted.UsesSetPCBack = true;
        Promoted.LongBranchSgprBase = Scratch.value_or(0);
        Promoted.LongBranchUsesVcc = UseVcc;
        Promoted.UsesDirectSetPCForward = true;
        Promoted.DirectSetPCForwardBytes = std::move(*Direct);
        Promoted.HasPoolBranchIsland = true;

        std::optional<uint64_t> NewPoolEnd = checkedAddUint64(
            PoolEndOffset, Promoted.Bytes.size(),
            "promoted branch-capacity trampoline layout");
        if (!NewPoolEnd)
          return false;
        Promoted.PoolBranchIslandOffset =
            *NewPoolEnd - PoolBranchIslandBytes;
        size_t NewIndex = Ctx.OutTrampolines.size();
        Ctx.OutTrampolines.emplace_back(std::move(Promoted));
        PoolIslandOwners[*NewPoolEnd - PoolBranchIslandBytes] = {
            NewIndex, Ctx.OutTrampolines.back().Bytes.size() -
                          PoolBranchIslandBytes};
        size_t PoolGatewayIndex = Gateways.size();
        Gateways.push_back({*NewPoolEnd - PoolBranchIslandBytes, *NewPoolEnd,
                            *NewPoolEnd - PoolBranchIslandBytes, 0,
                            std::numeric_limits<uint64_t>::max()});
        BranchGatewayHeads.insert(
            {*NewPoolEnd - PoolBranchIslandBytes, PoolGatewayIndex});
        FinalTrampolineAtOriginalOffset[Start] = NewIndex;
        for (uint64_t Relay = Tail; Relay + MinInstSize <= End;
             Relay += MinInstSize)
          SourceTailIslandOwners[Relay] = NewIndex;
        size_t SourceGatewayIndex = Gateways.size();
        Gateways.push_back({Tail, End, Tail, 0,
                            std::numeric_limits<uint64_t>::max()});
        BranchGatewayHeads.insert({Tail, SourceGatewayIndex});
        if (PlanningRegisterlessReturns)
          ReturnPromotionBranchOnlyRanges.push_back(
              {SourceGatewayIndex, End});
        ReservePromotionRange(Start, End);

        PromotionStillPossible.reset(Index);
        PoolEndOffset = *NewPoolEnd;
        Ctx.QueuedTrampolineBytes = PoolEndOffset - Ctx.PoolBaseOffset;
        Ctx.Profile.count(HotswapMetric::JumpLong);
        ++PromotedBranchCapacityRelays;
        // Keep small fixtures fully observable while bounding verbose output
        // for giant objects with tens of thousands of promotions.
        if (PromotedBranchCapacityRelays <= 16 ||
            isPowerOf2_64(PromotedBranchCapacityRelays))
          log() << "hotswap: promoted safe straight-line source at 0x"
                << utohexstr(Start) << " (size=" << Replacement.size()
                << ") to provide "
                << (Failure.Forward ? "forward" : "return")
                << "-capacity relay window [0x" << utohexstr(Tail) << ",0x"
                << utohexstr(End) << ") with "
                << (End - Tail) / MinInstSize << " slot(s) using "
                << (UseVcc ? "VCC" : "a local SGPR pair")
                << " (promotion " << PromotedBranchCapacityRelays << ")\n";
        return true;
      }
      return false;
    };
    if (TryPromote(/*AllowLocalPair=*/false))
      return true;
    return TryPromote(/*AllowLocalPair=*/true);
  };

  const auto ReturnBranchPlanStart = std::chrono::steady_clock::now();
  TrampOffset = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    std::optional<uint64_t> Next = checkedAddUint64(
        TrampOffset, T.Bytes.size(), "return-island trampoline layout");
    if (!Next)
      return false;
    TrampOffset = *Next;
    if (!T.Long || T.UsesSetPCBack)
      continue;
    const uint64_t TrailingIsland =
        T.HasPoolBranchIsland ? PoolBranchIslandBytes : 0;
    if (T.Bytes.size() < TrailingIsland + MinInstSize) {
      log() << "hotswap: error: registerless return reservation is "
               "truncated at 0x"
            << utohexstr(T.OriginalOffset) << "\n";
      return false;
    }
    uint64_t BackSlot = *Next - TrailingIsland - MinInstSize;
    std::optional<uint64_t> ReturnTo =
        checkedAddUint64(T.OriginalOffset, T.OriginalSize,
                         "registerless trampoline return target");
    if (!ReturnTo)
      return false;
    uint64_t OwnerOffset = T.OriginalOffset;
    uint32_t OwnerSize = T.OriginalSize;
    bool OwnerPreservesVcc = T.LongBranchPreservesVcc;
    bool OwnerUsesShared = T.UsesSharedDispatcherForward;
    bool OwnerUsesMirrored = T.UsesMirroredStubForward;
    uint64_t ChainTarget = ReservedReturnTails.lookup(I);
    BranchIslandFailure Failure;
    std::optional<SmallVector<uint64_t, 4>> ReturnIslands =
        allocateBackwardBranchIslands(
            Gateways, OwnerOffset, BackSlot,
            ChainTarget ? ChainTarget : *ReturnTo, &Failure,
            &BranchGatewayHeads, &OccupiedBranchGatewaySlots,
            PromoteBranchCapacityRelay);
    if (!ReturnIslands) {
      log() << "hotswap: error: no safe return s_branch island chain for far "
               "site 0x"
            << utohexstr(OwnerOffset) << " (size=" << OwnerSize
            << ", pool-back-slot=0x" << utohexstr(BackSlot)
            << ", return-to=0x" << utohexstr(*ReturnTo)
            << ", reserved-tail=0x" << utohexstr(ChainTarget)
            << ", preserve-vcc=" << OwnerPreservesVcc
            << ", shared=" << OwnerUsesShared
            << ", mirrored=" << OwnerUsesMirrored << ")\n";
      return false;
    }
    if (ChainTarget)
      ReturnIslands->push_back(ChainTarget);
    Trampoline &Planned = Ctx.OutTrampolines[I];
    Planned.ReturnBranchIslands = std::move(*ReturnIslands);
    Planned.ReturnBranchTargetOffset = *ReturnTo;
    ReturnBranchIslandChains += !Planned.ReturnBranchIslands.empty();
    if (RemainingRelayDemand)
      --RemainingRelayDemand;
  }
  log() << "hotswap: return branch-island planning took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - ReturnBranchPlanStart)
               .count()
        << " ms\n";
  PlanningRegisterlessReturns = false;
  // A return-phase promotion reserves its source suffix specifically for
  // registerless branch relays. Hide its remaining slots from the ordinary
  // set-PC planners, which do not record source-tail gateway ownership and
  // would otherwise have their bytes overwritten by final source padding.
  SmallVector<std::pair<uint64_t, uint64_t>, 8>
      HiddenReturnPromotionRanges;
  for (const auto &[GatewayIndex, OriginalEnd] :
       ReturnPromotionBranchOnlyRanges) {
    if (Gateways[GatewayIndex].WritePos < OriginalEnd)
      HiddenReturnPromotionRanges.push_back(
          {Gateways[GatewayIndex].WritePos, OriginalEnd});
    Gateways[GatewayIndex].End = Gateways[GatewayIndex].WritePos;
  }
  // Remove every committed branch dword from all physical aliases before a
  // variable-width ordinary gateway can cross and overwrite it. Splitting a
  // range preserves the free prefix and suffix while making the occupied
  // dword a hard layout boundary.
  subtractOccupiedBranchGatewaySlots(Gateways,
                                     OccupiedBranchGatewaySlots);
  const auto CandidateCountStart = std::chrono::steady_clock::now();
  for (PendingGateway &P : Pending) {
    const Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    if (!T.UsesSetPCBack)
      continue;
    Expected<uint64_t> CandidateSlots = countReachableSetPcGatewaySlots(
        Gateways, Ctx.LS, T.OriginalOffset, P.TargetOffset,
        T.LongBranchSgprBase, Pending.size(), T.LongBranchUsesVcc,
        T.LongBranchPreservesVcc);
    if (!CandidateSlots) {
      log() << "hotswap: error: failed to count gateways for far site 0x"
            << utohexstr(T.OriginalOffset) << ": "
            << toString(CandidateSlots.takeError()) << "\n";
      return false;
    }
    P.InitialCandidateSlots = *CandidateSlots;
  }
  log() << "hotswap: ordinary gateway candidate counting took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - CandidateCountStart)
               .count()
        << " ms\n";

  std::stable_sort(Pending.begin(), Pending.end(),
                   [](const PendingGateway &LHS, const PendingGateway &RHS) {
                     return LHS.InitialCandidateSlots <
                            RHS.InitialCandidateSlots;
                   });

  std::vector<PendingGateway> StillPending;
  StillPending.reserve(Pending.size());
  uint64_t AssignedGateways = 0;
  uint64_t AssignedSplitVccGateways = 0;
  auto OccupyOrdinaryGateway = [&](uint64_t Begin, uint64_t Size) {
    for (uint64_t Offset = 0; Offset < Size; Offset += MinInstSize)
      OccupiedBranchGatewaySlots.insert(Begin + Offset);
  };
  for (const PendingGateway &P : Pending) {
    Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    if (!T.UsesSetPCBack) {
      StillPending.push_back(P);
      continue;
    }
    Expected<std::optional<EncodedSetPcGateway>> GatewayOrErr =
        findNearestSetPcGateway(Gateways, Ctx.LS, T.OriginalOffset,
                                P.TargetOffset, T.LongBranchSgprBase,
                                T.LongBranchUsesVcc, T.LongBranchPreservesVcc,
                                &OccupiedBranchGatewaySlots);
    if (!GatewayOrErr) {
      log() << "hotswap: error: failed to plan gateway for far site 0x"
            << utohexstr(T.OriginalOffset) << ": "
            << toString(GatewayOrErr.takeError()) << "\n";
      return false;
    }
    std::optional<EncodedSetPcGateway> Gateway = std::move(*GatewayOrErr);
    if (!Gateway) {
      if (T.LongBranchPreservesVcc) {
        std::optional<EncodedSplitVccGateway> Split =
            findSplitVccGateway(Gateways, Ctx.LS, T.OriginalOffset,
                                P.TargetOffset, T.LongBranchSgprBase,
                                &OccupiedBranchGatewaySlots);
        if (Split) {
          NopSled &Primary = Gateways[Split->PrimaryIndex];
          NopSled &Secondary = Gateways[Split->SecondaryIndex];
          T.HasForwardGateway = true;
          T.ForwardGatewayOffset = Primary.WritePos;
          T.ForwardGatewayBytes = std::move(Split->PrimaryBytes);
          T.SharedDispatcherSecondaryGatewayOffset = Secondary.WritePos;
          T.SecondaryForwardGatewayBytes = std::move(Split->SecondaryBytes);
          DenseMap<uint64_t, size_t>::const_iterator SourceOwner =
              SourceTailGatewayOwners.find(
                  T.SharedDispatcherSecondaryGatewayOffset);
          if (SourceOwner != SourceTailGatewayOwners.end()) {
            Trampoline &OwnerT = Ctx.OutTrampolines[SourceOwner->second];
            if (OwnerT.HasSourceTailGateway) {
              log() << "hotswap: error: split VCC source-tail gateway at 0x"
                    << utohexstr(T.SharedDispatcherSecondaryGatewayOffset)
                    << " has conflicting allocations\n";
              return false;
            }
            OwnerT.HasSourceTailGateway = true;
            OwnerT.SourceTailGatewayOffset =
                T.SharedDispatcherSecondaryGatewayOffset;
            OwnerT.SourceTailGatewayBytes =
                T.SecondaryForwardGatewayBytes.size();
          }
          Primary.WritePos += T.ForwardGatewayBytes.size();
          Secondary.WritePos += T.SecondaryForwardGatewayBytes.size();
          OccupyOrdinaryGateway(T.ForwardGatewayOffset,
                                T.ForwardGatewayBytes.size());
          OccupyOrdinaryGateway(T.SharedDispatcherSecondaryGatewayOffset,
                                T.SecondaryForwardGatewayBytes.size());
          ++AssignedGateways;
          ++AssignedSplitVccGateways;
          if (RemainingRelayDemand)
            --RemainingRelayDemand;
          continue;
        }
      }
      StillPending.push_back(P);
      continue;
    }
    T.HasForwardGateway = true;
    T.ForwardGatewayOffset = Gateway->Sled->WritePos;
    T.ForwardGatewayBytes = std::move(Gateway->Bytes);
    DenseMap<uint64_t, size_t>::const_iterator SourceOwner =
        SourceTailGatewayOwners.find(T.ForwardGatewayOffset);
    if (SourceOwner != SourceTailGatewayOwners.end()) {
      Trampoline &OwnerT = Ctx.OutTrampolines[SourceOwner->second];
      if (OwnerT.HasSourceTailGateway &&
          (OwnerT.SourceTailGatewayOffset != T.ForwardGatewayOffset ||
           OwnerT.SourceTailGatewayBytes != T.ForwardGatewayBytes.size())) {
        log() << "hotswap: error: source-tail gateway at 0x"
              << utohexstr(T.ForwardGatewayOffset)
              << " has conflicting allocations\n";
        return false;
      }
      OwnerT.HasSourceTailGateway = true;
      OwnerT.SourceTailGatewayOffset = T.ForwardGatewayOffset;
      OwnerT.SourceTailGatewayBytes = T.ForwardGatewayBytes.size();
    }
    Gateway->Sled->WritePos += T.ForwardGatewayBytes.size();
    OccupyOrdinaryGateway(T.ForwardGatewayOffset,
                          T.ForwardGatewayBytes.size());
    ++AssignedGateways;
    if (RemainingRelayDemand)
      --RemainingRelayDemand;
  }
  Pending = std::move(StillPending);

  for (const auto &[Begin, End] : HiddenReturnPromotionRanges)
    Gateways.push_back({Begin, End, Begin, 0,
                        std::numeric_limits<uint64_t>::max()});
  subtractOccupiedBranchGatewaySlots(Gateways,
                                     OccupiedBranchGatewaySlots);
  BranchGatewayHeads =
      buildBranchGatewayHeads(Gateways, OccupiedBranchGatewaySlots);
  uint64_t BranchIslandChains = 0;
  StillPending.clear();
  StillPending.reserve(Pending.size());
  const auto ForwardBranchPlanStart = std::chrono::steady_clock::now();
  for (const PendingGateway &P : Pending) {
    uint64_t OwnerOffset =
        Ctx.OutTrampolines[P.TrampolineIndex].OriginalOffset;
    BranchIslandFailure Failure;
    std::optional<SmallVector<uint64_t, 4>> Islands =
        allocateForwardBranchIslands(
            Gateways, OwnerOffset, P.TargetOffset, &Failure,
            &BranchGatewayHeads, &OccupiedBranchGatewaySlots,
            PromoteBranchCapacityRelay);
    if (!Islands || Islands->empty()) {
      StillPending.push_back(P);
      continue;
    }
    Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    T.ForwardBranchIslands = std::move(*Islands);
    T.ForwardBranchTargetOffset = P.TargetOffset;
    if (T.LongBranchPreservesVcc)
      std::memcpy(T.Bytes.data(), Ctx.LS.SNopBytes.data(), MinInstSize);
    ++BranchIslandChains;
    if (RemainingRelayDemand)
      --RemainingRelayDemand;
  }
  Pending = std::move(StillPending);
  log() << "hotswap: forward branch-island planning took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - ForwardBranchPlanStart)
               .count()
        << " ms\n";
  if (PromotionSgprContinuationAnalyses != 0)
    log() << "hotswap: batched promotion SGPR continuation analysis built "
          << PromotionSgprContinuationAnalyses << " function cache(s)\n";

  if (!Pending.empty()) {
    const PendingGateway &P = Pending.front();
    const Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    if (!T.UsesSetPCBack)
      log() << "hotswap: error: no safe forward s_branch island chain for "
               "registerless far site 0x"
            << utohexstr(T.OriginalOffset) << "\n";
    else
      log() << "hotswap: error: no safe short-branch gateway for far site 0x"
            << utohexstr(T.OriginalOffset) << " (" << P.InitialCandidateSlots
            << " initial candidate slot(s), source_bytes=" << T.OriginalSize
            << ", setpc_back=" << T.UsesSetPCBack
            << ", preserve_vcc=" << T.LongBranchPreservesVcc
            << ", use_vcc=" << T.LongBranchUsesVcc << ")\n";
    return false;
  }
  if (AssignedGateways != 0)
    log() << "hotswap: assigned " << AssignedGateways
          << " SCC-neutral forward gateway(s)\n";
  if (AssignedSplitVccGateways != 0)
    log() << "hotswap: assigned " << AssignedSplitVccGateways
          << " split VCC-preserving gateway(s)\n";
  if (BranchIslandChains != 0)
    log() << "hotswap: assigned " << BranchIslandChains
          << " forward s_branch island chain(s)\n";
  if (ReturnBranchIslandChains != 0)
    log() << "hotswap: assigned " << ReturnBranchIslandChains
          << " return s_branch island chain(s)\n";
  if (PromotedBranchCapacityRelays != 0)
    log() << "hotswap: promoted " << PromotedBranchCapacityRelays
          << " safe straight-line source(s) for branch-island capacity\n";

  for (Trampoline &T : Ctx.OutTrampolines) {
    if (T.HasForwardGateway) {
      if (T.ForwardGatewayOffset > Ctx.TextSize ||
          T.ForwardGatewayBytes.size() >
              Ctx.TextSize - T.ForwardGatewayOffset) {
        log() << "hotswap: error: forward gateway at 0x"
              << utohexstr(T.ForwardGatewayOffset) << " extends past .text.\n";
        return false;
      }
      std::memcpy(Ctx.Text + T.ForwardGatewayOffset,
                  T.ForwardGatewayBytes.data(), T.ForwardGatewayBytes.size());
      if (!T.SecondaryForwardGatewayBytes.empty()) {
        uint64_t Offset = T.SharedDispatcherSecondaryGatewayOffset;
        if (Offset > Ctx.TextSize ||
            T.SecondaryForwardGatewayBytes.size() > Ctx.TextSize - Offset) {
          log() << "hotswap: error: secondary forward gateway at 0x"
                << utohexstr(Offset) << " extends past .text.\n";
          return false;
        }
        std::memcpy(Ctx.Text + Offset, T.SecondaryForwardGatewayBytes.data(),
                    T.SecondaryForwardGatewayBytes.size());
      }
    }
    for (size_t I = 0; I != T.ForwardBranchIslands.size(); ++I) {
      uint64_t From = T.ForwardBranchIslands[I];
      uint64_t To = I + 1 == T.ForwardBranchIslands.size()
                        ? T.ForwardBranchTargetOffset
                        : T.ForwardBranchIslands[I + 1];
      SmallVector<uint8_t> Branch = Ctx.LS.encodeSBranch(From, To);
      if (Branch.size() != MinInstSize) {
        log() << "hotswap: error: failed to encode forward branch island at "
                 "0x"
              << utohexstr(From) << "\n";
        return false;
      }
      DenseMap<uint64_t, PoolIslandOwner>::const_iterator Owner =
          PoolIslandOwners.find(From);
      DenseMap<uint64_t, size_t>::const_iterator SourceOwner =
          SourceTailIslandOwners.find(From);
      if (SourceOwner != SourceTailIslandOwners.end()) {
        Trampoline &OwnerT = Ctx.OutTrampolines[SourceOwner->second];
        if (!recordSourceTailBranchIsland(OwnerT, From, To)) {
          log() << "hotswap: error: forward source-tail relay at 0x"
                << utohexstr(From) << " has conflicting targets\n";
          return false;
        }
      } else if (Owner != PoolIslandOwners.end()) {
        Trampoline &OwnerT =
            Ctx.OutTrampolines[Owner->second.TrampolineIndex];
        std::memcpy(OwnerT.Bytes.data() + Owner->second.RelativeOffset,
                    Branch.data(), Branch.size());
      } else {
        if (From > Ctx.TextSize || Branch.size() > Ctx.TextSize - From) {
          log() << "hotswap: error: forward branch island at 0x"
                << utohexstr(From) << " is outside .text and trampoline pool\n";
          return false;
        }
        std::memcpy(Ctx.Text + From, Branch.data(), Branch.size());
      }
    }
    for (size_t I = 0; I != T.ReturnBranchIslands.size(); ++I) {
      uint64_t From = T.ReturnBranchIslands[I];
      uint64_t To = I + 1 == T.ReturnBranchIslands.size()
                        ? T.ReturnBranchTargetOffset
                        : T.ReturnBranchIslands[I + 1];
      SmallVector<uint8_t> Branch = Ctx.LS.encodeSBranch(From, To);
      if (Branch.size() != MinInstSize) {
        log() << "hotswap: error: failed to encode return branch island at 0x"
              << utohexstr(From) << "\n";
        return false;
      }
      DenseMap<uint64_t, PoolIslandOwner>::const_iterator Owner =
          PoolIslandOwners.find(From);
      DenseMap<uint64_t, size_t>::const_iterator SourceOwner =
          SourceTailIslandOwners.find(From);
      if (SourceOwner != SourceTailIslandOwners.end()) {
        Trampoline &OwnerT = Ctx.OutTrampolines[SourceOwner->second];
        if (!recordSourceTailBranchIsland(OwnerT, From, To)) {
          log() << "hotswap: error: return source-tail relay at 0x"
                << utohexstr(From) << " has conflicting targets\n";
          return false;
        }
      } else if (Owner != PoolIslandOwners.end()) {
        Trampoline &OwnerT =
            Ctx.OutTrampolines[Owner->second.TrampolineIndex];
        std::memcpy(OwnerT.Bytes.data() + Owner->second.RelativeOffset,
                    Branch.data(), Branch.size());
      } else {
        if (From > Ctx.TextSize || Branch.size() > Ctx.TextSize - From) {
          log() << "hotswap: error: return branch island at 0x"
                << utohexstr(From) << " is outside .text and trampoline pool\n";
          return false;
        }
        std::memcpy(Ctx.Text + From, Branch.data(), Branch.size());
      }
    }
  }
  return true;
}

/// Emit \p Replacement for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize). Prefers an in-place NOP-sled rewrite when a
/// reachable sled with sufficient headroom exists; otherwise falls back to a
/// deferred trampoline.
[[nodiscard]] bool emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset,
                                       uint32_t InstSize,
                                       ArrayRef<uint8_t> Replacement) {
  std::optional<uint64_t> ReturnTo = checkedAddUint64(
      InstOffset, InstSize, "replacement trampoline return target");
  std::optional<uint64_t> PoolReturnFrom =
      checkedAddUint64(Ctx.PoolBaseOffset, Replacement.size(),
                       "replacement trampoline return slot");
  if (!ReturnTo || !PoolReturnFrom)
    return false;

  // When the pool base is already out of short-branch reach, defer every site
  // to the global trampoline pass. That pass can coalesce adjacent patches
  // before allocating gateways; consuming NOP padding greedily here can strand
  // a later small or clause/delay-constrained source window.
  bool PoolBaseFar = !isSBranchReachable(InstOffset, Ctx.PoolBaseOffset) ||
                     !isSBranchReachable(*PoolReturnFrom, *ReturnTo);
  if (!PoolBaseFar && !Ctx.DirectControlFlow.HasUnresolvedTargets) {
    // findNearestSled enforces sled headroom. emitToNopSled still validates
    // exact branch reachability because branch-back distance includes the
    // replacement size, not just the original instruction offset.
    uint64_t Needed = Replacement.size() + MinInstSize;
    if (NopSled *Sled = findNearestSled(Ctx.NopSleds, InstOffset, Needed)) {
      if (emitToNopSled(Ctx, *Sled, InstOffset, InstSize, Replacement))
        return true;
      log() << "hotswap: emitReplacementCode: NOP sled at offset 0x"
            << utohexstr(Sled->WritePos)
            << " is not branch-reachable after assembly; using trampoline.\n";
    }
  }
  return emitToTrampoline(Ctx, InstOffset, InstSize, Replacement);
}

// -- applyGfx1250B0toA0Rules --------------------------------------------------

/// Per-instruction patch-pass trampoline: invokes \p Fn with (\p Ctx,
/// \p Idx) if it is non-null, or returns 0 otherwise. nullptr means
/// the corresponding pass family has no implementation linked in,
/// which the dispatcher treats as a no-op slot. std::nullopt means the
/// pass found a required patch failure after logging a specific reason.
static std::optional<uint32_t> runPerInstPass(uint32_t (*Fn)(PatchContext &,
                                                             size_t),
                                              PatchContext &Ctx, size_t Idx) {
  if (!Fn)
    return 0;

  uint32_t PatchCount = Fn(Ctx, Idx);
  if (Ctx.RequiredPatchFailed)
    return std::nullopt;
  return PatchCount;
}

/// Main per-instruction dispatcher for the GFX1250 B0-to-A0 rewrite.
/// Builds the NOP sled map, CFG, and VGPR liveness for the decoded stream,
/// then walks each decoded instruction and runs the patch passes in order
/// (in-place -> trampoline -> WMMA split -> scratch). Each pass gets a
/// chance to claim the instruction; first non-zero return wins. Also runs
/// the whole-function WMMA-hazard pass after the per-instruction loop and
/// records per-kernel stats via ElfView::updateKernelDescriptorVgprCount.
/// Returns the total number of applied patches across all passes.
static std::optional<uint32_t> applyGfx1250B0toA0Rules(
    std::vector<InternalDecodedInst> &Decoded, uint8_t *Text, uint64_t TextSize,
    const LLVMState &LS, std::vector<Trampoline> &OutTrampolines, ElfView &Elf,
    std::vector<ScratchPatchInfo> &OutScratchPatches,
    const RewriteConfig &Config, bool &OutRequiredPatchApplied,
    HotswapProfile &Profile) {
  uint32_t Patched = 0;

  HotswapProfile::Scope SledScope = Profile.time(HotswapMetric::NopSledScan);
  std::vector<NopSled> Sleds = buildNopSledMap(Decoded, LS, Elf);
  SledScope.finish();

  std::optional<DeclaredTextEntryInfo> DeclaredEntries =
      collectDeclaredTextEntries(Elf);
  if (!DeclaredEntries)
    return std::nullopt;
  std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Elf.functionTextRanges();
  std::optional<DirectControlFlowInfo> ControlFlow = collectDirectBranchTargets(
      Decoded, LS, Elf.textAddr(), Elf.textSize(), DeclaredEntries->Entries,
      FunctionRanges, DeclaredEntries->ExternalEntries,
      ArrayRef<uint8_t>(Text, TextSize), &Elf, DeclaredEntries->NonCallEntries);
  if (!ControlFlow)
    return std::nullopt;
  if (ControlFlow->HasUnresolvedTargets) {
    log() << "hotswap: unresolved control-flow target disables NOP-sled "
             "emission, trampoline coalescing, source relocation, and .text "
             "gateways\n";
    Sleds.clear();
  } else {
    truncateNopSledsAtDirectTargets(Sleds, ControlFlow->Targets);
  }

  HotswapProfile::Scope CfgScope = Profile.time(HotswapMetric::CfgBuild);
  CFG Cfg = buildCfg(Decoded, *LS.MCII);
  CfgScope.finish();

  HotswapProfile::Scope LiveScope = Profile.time(HotswapMetric::Liveness);
  LivenessInfo Liveness =
      computeLiveness(Decoded, Cfg, *LS.MCII, *LS.MRI, Config.MaxVgprs);
  LiveScope.finish();

  if (!Liveness.Converged) {
    log() << "hotswap: error: liveness analysis did not converge, using "
          << "conservative all-VGPRs-live fallback\n";
    Liveness.setConservativeAllLive(Config.MaxVgprs);
  }

  StringMap<KernelPatchStats> KernelStats;
  // Pool base as a .text-relative offset for trampoline branch math. The pool
  // is always >= textAddr(); checkedSubUint64 guards a malformed object.
  std::optional<uint64_t> PoolVAddr = Elf.trampolinePoolVAddr();
  if (!PoolVAddr)
    return std::nullopt;
  std::optional<uint64_t> PoolBaseOffset = checkedSubUint64(
      *PoolVAddr, Elf.textAddr(), "trampoline pool base offset");
  if (!PoolBaseOffset)
    return std::nullopt;
  PatchContext Ctx{Config,         Decoded,         Text,
                   TextSize,       *PoolBaseOffset, LS,
                   OutTrampolines, Sleds,           Elf,
                   Liveness,       KernelStats,     OutScratchPatches,
                   *ControlFlow,   Profile,         DeclaredEntries->Entries};

  const HotswapPatchVTable &VT = getHotswapPatchVTable();

  // Skip undecoded slots produced by the decoder for bytes it could not
  // classify as a valid instruction; the dispatcher has nothing to match
  // against on these and we must not invoke the patch passes for them.
  constexpr StringLiteral UnknownMnemonic = "<unknown>";
  using PerInstPatchFn = uint32_t (*)(PatchContext &, size_t);
  // A pass plus its metric; time/patches are summed locally and flushed once
  // after the loop (see HotswapProfile::add).
  struct TimedPass {
    PerInstPatchFn Fn;
    HotswapMetric Metric;
    uint64_t Nanos = 0;
    uint64_t Patches = 0;
  };
  SmallVector<TimedPass, 5> Passes;
  if (Config.RunB0A0Patches) {
    Passes.push_back({VT.applyInPlacePatches, HotswapMetric::InPlace});
    Passes.push_back({VT.applyTrampolinePatches, HotswapMetric::Trampoline});
    Passes.push_back({VT.applyWmmaSplitPatches, HotswapMetric::WmmaSplit});
    Passes.push_back({VT.applyScratchPatches, HotswapMetric::ScratchFp8});
    Passes.push_back({VT.applyWmmaScale16Patches, HotswapMetric::WmmaScale16});
  } else {
    Passes.push_back({VT.applyTrampolinePatches, HotswapMetric::Trampoline});
  }

  const bool Prof = Ctx.Profile.enabled();

  for (size_t Idx = 0, E = Decoded.size(); Idx < E; ++Idx) {
    const InternalDecodedInst &DI = Decoded[Idx];
    if (DI.Mnemonic == UnknownMnemonic)
      continue;

    for (TimedPass &Pass : Passes) {
      const uint64_t T0 = Prof ? profNowNs() : 0;
      std::optional<uint32_t> P = runPerInstPass(Pass.Fn, Ctx, Idx);
      if (Prof) {
        Pass.Nanos += profNowNs() - T0;
        Pass.Patches += P.value_or(0);
      }
      if (!P)
        return std::nullopt;
      if (*P == 0)
        continue;
      Patched += *P;
      break;
    }
  }

  if (Prof)
    for (const TimedPass &Pass : Passes)
      Ctx.Profile.add(Pass.Metric, Pass.Nanos, Pass.Patches);

  // Whole-kernel passes below run after per-instruction patches. Earlier
  // passes may have modified Text bytes, but the Decoded stream still holds
  // the original MCInst/Mnemonic/Offset entries. This is safe because:
  //  - In-place patches only change opcodes within the same encoding size,
  //    preserving instruction boundaries and offsets.
  //  - Trampoline patches replace the original instruction with a branch
  //    (same size), so the Decoded entry's Offset still points at the
  //    branch site; the WMMA classifier and VOP3PX2 mnemonic match won't
  //    treat a branch as WMMA/VALU/VOP3PX2.
  // If a future patch family changes instruction boundaries, the Decoded
  // stream must be rebuilt before these passes run.
  if (Config.RunB0A0Patches && VT.applyWmmaHazardPatch) {
    HotswapProfile::Scope HazardScope =
        Ctx.Profile.time(HotswapMetric::WmmaHazard);
    const uint32_t P = VT.applyWmmaHazardPatch(Ctx);
    HazardScope.addPatches(P);
    HazardScope.finish();
    Patched += P;
  }
  if (Config.RunB0A0Patches && VT.applyVop3px2Src2Fix) {
    HotswapProfile::Scope Vop3Scope =
        Ctx.Profile.time(HotswapMetric::Vop3px2Src2);
    const uint32_t P = VT.applyVop3px2Src2Fix(Ctx);
    Vop3Scope.addPatches(P);
    Vop3Scope.finish();
    Patched += P;
  }

  if (!OutTrampolines.empty()) {
    if (!ControlFlow->HasUnresolvedTargets) {
      mergeAdjacentLongTrampolines(OutTrampolines, ControlFlow->Targets);
      expandStraightLineTrampolines(Ctx, ControlFlow->Targets);
      mergeAdjacentLongTrampolines(OutTrampolines, ControlFlow->Targets);
    }
    if (!finalizeDeferredVccPreservation(Ctx))
      return std::nullopt;
    appendPoolBranchIslands(OutTrampolines);
    bool AllowTextGateways = !ControlFlow->HasUnresolvedTargets &&
                             !ControlFlow->HasUnboundedIndirectEntries;
    if (!assignLongBranchGateways(Ctx, ControlFlow->Targets, AllowTextGateways))
      return std::nullopt;
  }

  struct ResourceCounts {
    unsigned Vgprs;
    unsigned Sgprs;
  };
  StringMap<ResourceCounts> CountsBefore;
  StringMap<unsigned> VgprGranules;
  StringMap<unsigned> RequiredVgprCounts;
  StringMap<unsigned> RequiredSgprCounts;
  for (const StringMapEntry<KernelPatchStats> &KV : KernelStats) {
    StringRef KName = KV.first();
    const KernelPatchStats &Stats = KV.second;
    if (KName.empty())
      continue;
    unsigned VgprGranule = getKernelVgprGranuleSize(Ctx, KName);
    VgprGranules.try_emplace(KName, VgprGranule);
    std::optional<unsigned> VgprsBefore =
        Elf.getKernelVgprCount(KName, VgprGranule);
    std::optional<unsigned> SgprsBefore = Elf.getKernelSgprCount(KName);
    CountsBefore.try_emplace(KName, ResourceCounts{VgprsBefore.value_or(0),
                                                   SgprsBefore.value_or(0)});
    if (Stats.ExtraVgprs > 0) {
      // Every current VGPR-growing patch preflights before emitting bytes.
      // Keep this required-policy check as a fail-safe so a future path cannot
      // silently emit a kernel that no longer admits one maximum workgroup.
      if (checkKernelVgprBump(Ctx, KName, Stats.ExtraVgprs,
                              PatchRequirement::Required) !=
          VgprBumpDecision::Apply)
        return std::nullopt;
      if (!VgprsBefore) {
        log() << "hotswap: error: failed to read VGPR count for kernel "
              << KName << "\n";
        return std::nullopt;
      }
      if (Stats.ExtraVgprs >
          std::numeric_limits<unsigned>::max() - *VgprsBefore) {
        log() << "hotswap: error: VGPR count for kernel " << KName
              << " overflows unsigned after hotswap scratch allocation\n";
        return std::nullopt;
      }
      RequiredVgprCounts.try_emplace(KName, *VgprsBefore + Stats.ExtraVgprs);
    }
    if (Stats.ExtraSgprs > 0) {
      if (!SgprsBefore) {
        log() << "hotswap: error: failed to read SGPR count for kernel "
              << KName << "\n";
        return std::nullopt;
      }
      if (Stats.ExtraSgprs >
          std::numeric_limits<unsigned>::max() - *SgprsBefore) {
        log() << "hotswap: error: SGPR count for kernel " << KName
              << " overflows unsigned after hotswap scratch allocation\n";
        return std::nullopt;
      }
      unsigned RequiredSgprs = *SgprsBefore + Stats.ExtraSgprs;
      RequiredSgprCounts.try_emplace(KName, RequiredSgprs);
    }
  }

  if (!Elf.updateKernelMetadataVgprCounts(RequiredVgprCounts)) {
    log() << "hotswap: error: failed to update kernel VGPR metadata\n";
    return std::nullopt;
  }
  if (!Elf.updateKernelMetadataSgprCounts(RequiredSgprCounts)) {
    log() << "hotswap: error: failed to update kernel SGPR metadata\n";
    return std::nullopt;
  }
  for (const StringMapEntry<unsigned> &Required : RequiredVgprCounts) {
    StringMap<unsigned>::const_iterator Granule =
        VgprGranules.find(Required.first());
    if (Granule == VgprGranules.end()) {
      log() << "hotswap: error: missing VGPR granule for kernel "
            << Required.first() << "\n";
      return std::nullopt;
    }
    if (!Elf.updateKernelDescriptorVgprCount(Required.first(), Required.second,
                                             Granule->second)) {
      log() << "hotswap: error: failed to update VGPR descriptor count for "
            << Required.first() << "\n";
      return std::nullopt;
    }
  }

  for (const StringMapEntry<KernelPatchStats> &KV : KernelStats) {
    StringRef KName = KV.first();
    const KernelPatchStats &Stats = KV.second;
    if (KName.empty())
      continue;
    StringMap<ResourceCounts>::const_iterator Before = CountsBefore.find(KName);
    if (Before == CountsBefore.end()) {
      log() << "hotswap: error: missing cached resource counts for kernel "
            << KName << "\n";
      return std::nullopt;
    }
    StringMap<unsigned>::const_iterator Granule = VgprGranules.find(KName);
    if (Granule == VgprGranules.end()) {
      log() << "hotswap: error: missing VGPR granule for kernel " << KName
            << "\n";
      return std::nullopt;
    }
    std::optional<unsigned> VgprsAfter =
        Elf.getKernelVgprCount(KName, Granule->second);
    std::optional<unsigned> SgprsAfter = Elf.getKernelSgprCount(KName);
    log() << "hotswap: liveness: kernel " << KName
          << ": vgprs_before=" << Before->second.Vgprs
          << ", vgprs_after=" << VgprsAfter.value_or(0)
          << ", sgprs_before=" << Before->second.Sgprs
          << ", sgprs_after=" << SgprsAfter.value_or(0)
          << ", scratch_reused=" << Stats.ScratchReused
          << ", scratch_above_kd=" << Stats.ScratchAboveKd << "\n";
  }
  OutRequiredPatchApplied = Ctx.RequiredPatchApplied;
  return Patched;
}

// -- retargetCodeObject helpers -------------------------------------------

/// Finalize the deferred trampolines produced by emitToTrampoline: resolves
/// the branch-back at the tail of each trampoline to land on the next
/// instruction after the original site, writes the branch-forward + s_nop
/// padding at the original .text slot, and reports per-trampoline encoding
/// failures through log(). Runs after all patch passes finish so the
/// post-.text layout of trampolines is known. Returns false if any
/// trampoline could not be fixed up.
[[nodiscard]] static bool
fixupTrampolineBranches(std::vector<Trampoline> &Trampolines, uint8_t *Text,
                        uint64_t PoolBaseOffset, const LLVMState &LS) {
  // Fail-fast on the first encoding error: the position of later
  // trampolines depends on earlier ones, so a single bad branch would
  // cascade into incorrect layout. A single failure invalidates the whole
  // rewrite, so there is nothing useful to recover beyond it.
  //
  // Offsets are .text-relative; the pool begins at PoolBaseOffset
  // (trampolinePoolVAddr() - textAddr()), which can be far past .text.
  uint64_t TrampOffset = PoolBaseOffset;
  for (Trampoline &T : Trampolines) {
    uint64_t TP = TrampOffset;
    std::optional<uint64_t> NextTrampOffset = checkedAddUint64(
        TrampOffset, T.Bytes.size(), "trampoline fixup layout");
    if (!NextTrampOffset)
      return false;
    TrampOffset = *NextTrampOffset;

    const uint32_t BackReserve =
        T.LongBranchPreservesVcc
            ? VccPreservingReturnReserveBytes
            : (T.UsesSetPCBack ? SetPcReturnReserveBytes : MinInstSize);
    const uint32_t TrailingIsland =
        T.HasPoolBranchIsland ? PoolBranchIslandBytes : 0;
    if (T.Bytes.size() < BackReserve + TrailingIsland) {
      log() << "hotswap: error: trampoline return reservation is truncated at "
               "0x"
            << utohexstr(T.OriginalOffset) << "\n";
      return false;
    }
    const uint64_t BackSlot = TrampOffset - TrailingIsland - BackReserve;
    const size_t BackOffset = T.Bytes.size() - TrailingIsland - BackReserve;
    std::optional<uint64_t> ReturnTo = checkedAddUint64(
        T.OriginalOffset, T.OriginalSize, "trampoline return target");
    if (!ReturnTo)
      return false;

    std::optional<SmallVector<uint8_t>> BrBack;
    if (T.LongBranchPreservesVcc) {
      SmallVector<uint8_t> Save = assembleSingleInst(
          "s_mov_b32 s" + std::to_string(T.LongBranchSgprBase) + ", vcc_lo",
          LS);
      std::optional<uint64_t> SetPcOffset = checkedAddUint64(
          BackSlot, Save.size(), "VCC-preserving return set-PC offset");
      uint64_t LandingDisplacement = T.UsesDirectSetPCForward
                                         ? T.DirectSetPCForwardBytes.size()
                                         : MinInstSize;
      std::optional<uint64_t> Landing =
          checkedAddUint64(T.OriginalOffset, LandingDisplacement,
                           "VCC-preserving return landing offset");
      if (Save.size() != VccMoveBytes || !SetPcOffset || !Landing)
        return false;
      std::optional<SmallVector<uint8_t>> SetPc = encodeSetPCLongBranch(
          LS, *SetPcOffset, *Landing, T.LongBranchSgprBase, /*UseVcc=*/true);
      if (SetPc) {
        Save.append(SetPc->begin(), SetPc->end());
        BrBack = std::move(Save);
      }
    } else if (T.UsesSetPCBack) {
      BrBack = encodeSetPCLongBranch(LS, BackSlot, *ReturnTo,
                                     T.LongBranchSgprBase, T.LongBranchUsesVcc);
    } else {
      uint64_t BranchTarget = T.ReturnBranchIslands.empty()
                                  ? *ReturnTo
                                  : T.ReturnBranchIslands.front();
      SmallVector<uint8_t> ShortBranch =
          LS.encodeSBranch(BackSlot, BranchTarget);
      if (!ShortBranch.empty())
        BrBack = std::move(ShortBranch);
    }
    if (!BrBack || BrBack->size() > BackReserve) {
      log() << "hotswap: error: trampoline branch-back encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << (T.Long ? " (long)\n" : "\n");
      return false;
    }
    std::memcpy(T.Bytes.data() + BackOffset, BrBack->data(), BrBack->size());
    for (uint32_t I = BrBack->size(); I + MinInstSize <= BackReserve;
         I += MinInstSize)
      std::memcpy(T.Bytes.data() + BackOffset + I, LS.SNopBytes.data(),
                  MinInstSize);

    SmallVector<uint8_t> BrFwd;
    if (T.Long) {
      if (T.UsesSharedDispatcherForward) {
        const std::string Pair =
            "s[" + std::to_string(T.SharedDispatcherSgprBase) + ":" +
            std::to_string(T.SharedDispatcherSgprBase + 1) + "]";
        uint64_t BranchTarget =
            T.SharedDispatcherRelayOffset    ? T.SharedDispatcherRelayOffset
            : T.ForwardBranchIslands.empty() ? T.SharedDispatcherGatewayOffset
                                             : T.ForwardBranchIslands.front();
        BrFwd = encodeDirectCall(LS, T.OriginalOffset, BranchTarget, Pair);
        if (BrFwd.size() != MinInstSize)
          return false;
      } else if (T.UsesMirroredStubForward) {
        const std::string Pair =
            T.LongBranchUsesVcc
                ? "vcc"
                : "s[" + std::to_string(T.LongBranchSgprBase) + ":" +
                      std::to_string(T.LongBranchSgprBase + 1) + "]";
        uint64_t BranchTarget = T.ForwardBranchIslands.empty()
                                    ? T.MirroredStubGatewayOffset
                                    : T.ForwardBranchIslands.front();
        BrFwd = encodeDirectCall(LS, T.OriginalOffset, BranchTarget, Pair);
        if (BrFwd.size() != MinInstSize)
          return false;
      } else if (T.UsesShortBranchForward) {
        BrFwd = LS.encodeSBranch(T.OriginalOffset, TP);
      } else if (!T.ForwardBranchIslands.empty()) {
        BrFwd =
            LS.encodeSBranch(T.OriginalOffset, T.ForwardBranchIslands.front());
      } else if (T.UsesDirectSetPCForward) {
        BrFwd = T.DirectSetPCForwardBytes;
      } else if (T.HasForwardGateway) {
        BrFwd = LS.encodeSBranch(T.OriginalOffset, T.ForwardGatewayOffset);
      } else {
        log() << "hotswap: error: far trampoline has no forward gateway at 0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
    } else {
      BrFwd = LS.encodeSBranch(T.OriginalOffset, TP);
    }
    if (BrFwd.empty() || BrFwd.size() > T.OriginalSize) {
      log() << "hotswap: error: trampoline branch-fwd encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << (T.Long ? " (long)\n" : "\n");
      return false;
    }
    std::memcpy(Text + T.OriginalOffset, BrFwd.data(), BrFwd.size());
    uint32_t PadStart = BrFwd.size();
    if (T.LongBranchPreservesVcc) {
      uint64_t LandingDisplacement =
          T.UsesDirectSetPCForward ? BrFwd.size() : MinInstSize;
      if ((!T.UsesDirectSetPCForward && BrFwd.size() != MinInstSize) ||
          LandingDisplacement > T.OriginalSize ||
          VccLandingPadBytes > T.OriginalSize - LandingDisplacement) {
        log() << "hotswap: error: VCC-preserving source window is invalid at "
                 "0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
      SmallVector<std::string, 2> RestoreLines;
      RestoreLines.push_back("s_mov_b32 vcc_lo, s" +
                             std::to_string(T.LongBranchSgprBase));
      RestoreLines.push_back("s_delay_alu instid0(SALU_CYCLE_1)");
      SmallVector<uint8_t> Restore =
          assembleInstructions(joinAsmLines(RestoreLines), LS);
      if (Restore.size() != VccRestoreSequenceBytes) {
        log() << "hotswap: error: failed to encode VCC restore landing at 0x"
              << utohexstr(T.OriginalOffset + LandingDisplacement) << "\n";
        return false;
      }
      std::memcpy(Text + T.OriginalOffset + LandingDisplacement, Restore.data(),
                  Restore.size());
      PadStart = LandingDisplacement + VccLandingPadBytes;
    }
    if (T.HasSourceTailGateway) {
      std::optional<uint64_t> GatewayEnd =
          checkedAddUint64(T.SourceTailGatewayOffset, T.SourceTailGatewayBytes,
                           "source-tail gateway end");
      std::optional<uint64_t> SourceEnd = checkedAddUint64(
          T.OriginalOffset, T.OriginalSize, "source-tail gateway source end");
      if (!GatewayEnd || !SourceEnd ||
          T.SourceTailGatewayOffset < T.OriginalOffset + PadStart ||
          *GatewayEnd > *SourceEnd) {
        log() << "hotswap: error: source-tail gateway overlaps the forward "
                 "sequence at 0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
    }
    // Pad the tail of the replaced slot with cached s_nop bytes.
    for (uint32_t I = PadStart; I + MinInstSize <= T.OriginalSize;
         I += MinInstSize) {
      uint64_t Offset = T.OriginalOffset + I;
      if (T.HasSourceTailGateway && Offset >= T.SourceTailGatewayOffset &&
          Offset - T.SourceTailGatewayOffset < T.SourceTailGatewayBytes)
        continue;
      std::memcpy(Text + T.OriginalOffset + I, LS.SNopBytes.data(),
                  MinInstSize);
    }
    for (const auto &[RelayOffset, RelayTarget] :
         T.SourceTailBranchIslands) {
      if (RelayOffset < T.OriginalOffset ||
          RelayOffset - T.OriginalOffset < PadStart ||
          RelayOffset - T.OriginalOffset > T.OriginalSize - MinInstSize ||
          (T.HasSourceTailGateway &&
           RelayOffset >= T.SourceTailGatewayOffset &&
           RelayOffset - T.SourceTailGatewayOffset <
               T.SourceTailGatewayBytes)) {
        log() << "hotswap: error: source-tail branch island overlaps the "
                 "forward sequence at 0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
      SmallVector<uint8_t> Relay =
          LS.encodeSBranch(RelayOffset, RelayTarget);
      if (Relay.size() != MinInstSize) {
        log() << "hotswap: error: source-tail branch island encoding failed "
                 "at 0x"
              << utohexstr(RelayOffset) << "\n";
        return false;
      }
      std::memcpy(Text + RelayOffset, Relay.data(), Relay.size());
    }
  }
  return true;
}

/// Fix up DWARF sections of the grown ELF after trampolines have been
/// appended: adds trampoline symbols to the symbol table, shifts
/// .debug_line / .debug_ranges / .debug_info / .debug_frame addresses by
/// the total trampoline footprint, and reports per-section failures via
/// log(). Individual patchDebug* helpers are weak stubs here; concrete
/// implementations land in separate PRs.
static void patchDebugSections(WritableMemoryBuffer &ElfBuf,
                               ArrayRef<Trampoline> Trampolines,
                               const ElfView &Elf, size_t GrowthTotal) {
  uint8_t *Data = reinterpret_cast<uint8_t *>(ElfBuf.getBufferStart());
  size_t Size = ElfBuf.getBufferSize();
  if (!addTrampolineSymbols(ElfBuf, Trampolines, Elf.textSize(),
                            Elf.textSectionIndex()))
    log() << "hotswap: error: addTrampolineSymbols failed\n";
  patchDebugRanges(Data, Size, Elf.textAddr(), Elf.textSize(), GrowthTotal);
  patchDebugInfo(Data, Size, Elf.textAddr(), Elf.textSize(), GrowthTotal);
  patchDebugFrame(Data, Size, Elf.textAddr(), Elf.textSize(), GrowthTotal);
  if (!patchDebugLine(ElfBuf, Trampolines, Elf.textSize(), Elf.textAddr()))
    log() << "hotswap: error: patchDebugLine failed\n";
}

/// Re-open the grown ELF and cross-check that no scratch-patched site
/// reads a VGPR still live at the patch point: builds a fresh ElfView over
/// the output buffer, hands the new .text to verifyPatchCorrectness, and
/// logs a diagnostic if the verifier detects a potential conflict. Runs
/// only when the scratch patch pass produced at least one ScratchPatchInfo
/// record.
static void runScratchVerification(WritableMemoryBuffer &OutBuf,
                                   const LLVMState &LS,
                                   ArrayRef<ScratchPatchInfo> ScratchPatches,
                                   unsigned MaxVgprs) {
  // Build a fresh ElfView over the grown buffer to find the new .text.
  // WritableMemoryBuffer::getBufferStart() returns char *, so no const_cast
  // is needed on the way to ElfView::create's uint8_t * contract.
  uint8_t *Data = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart());
  Expected<ElfView> ViewOrErr = ElfView::create(Data, OutBuf.getBufferSize());
  if (!ViewOrErr) {
    consumeError(ViewOrErr.takeError());
    return;
  }
  if (ViewOrErr->textSize() == 0)
    return;
  if (!verifyPatchCorrectness(ViewOrErr->textData(), ViewOrErr->textSize(), LS,
                              ScratchPatches, MaxVgprs))
    log() << "hotswap: error: post-patch verification detected possible "
          << "scratch conflicts\n";
}

static std::unique_ptr<WritableMemoryBuffer>
copyOutputBuffer(const void *Data, size_t Size, StringRef CopyKind) {
  std::unique_ptr<WritableMemoryBuffer> Result =
      WritableMemoryBuffer::getNewUninitMemBuffer(Size);
  if (!Result) {
    log() << "hotswap: error: retargetCodeObject: "
          << "getNewUninitMemBuffer(" << Size
          << ") failed (out of memory) for the " << CopyKind
          << " output copy.\n";
    return nullptr;
  }

  std::memcpy(Result->getBufferStart(), Data, Size);
  return Result;
}

// -- retargetCodeObject -------------------------------------------------------

static amd_comgr_status_t retargetCodeObjectImpl(
    const void *ElfData, size_t ElfSize, const TargetIdentifier &TargetIdent,
    const Gfx1250RewriteOptions &Options, std::unique_ptr<MemoryBuffer> &Out,
    bool AllowTextDisplacement, HotswapProfile &Profile) {
  // The dispatcher fetches the patch vtable lazily via
  // getHotswapPatchVTable() inside applyGfx1250B0toA0Rules; the singleton's
  // initializer binds every register*Patch slot on first access, so no
  // explicit install step is needed here.

  const bool RunInstructionPatches =
      Options.RunB0A0Patches ||
      Options.MaskPolicy != MaskWorkaroundPolicy::None;
  const bool Prof = Profile.enabled();

  // Take a working copy so the input is preserved and we have a mutable
  // buffer to parse / patch.
  uint64_t InputCopyT0 = Prof ? profNowNs() : 0;
  std::vector<uint8_t> Buf(static_cast<const uint8_t *>(ElfData),
                           static_cast<const uint8_t *>(ElfData) + ElfSize);
  if (Prof)
    Profile.add(HotswapMetric::InputCopy, profNowNs() - InputCopyT0, 0);

  uint64_t ParseT0 = Prof ? profNowNs() : 0;
  Expected<ElfView> ViewOrErr = ElfView::create(Buf.data(), Buf.size());
  if (!ViewOrErr) {
    log() << "hotswap: error: retargetCodeObject: input is not a "
          << "parseable ELF64 (" << toString(ViewOrErr.takeError()) << ").\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfView &Elf = *ViewOrErr;
  if (ViewOrErr->textSize() == 0) {
    if (!Elf.isValidDataOnlyObject()) {
      log() << "hotswap: error: retargetCodeObject: empty .text does not "
               "describe a valid data-only code object.\n";
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    std::unique_ptr<WritableMemoryBuffer> Result =
        copyOutputBuffer(ElfData, ElfSize, "data-only");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    Out = std::move(Result);
    log() << "hotswap: accepted data-only code object with empty .text; "
             "returning a byte-identical copy.\n";
    return AMD_COMGR_STATUS_SUCCESS;
  }
  if (Prof)
    Profile.add(HotswapMetric::ElfParse, profNowNs() - ParseT0, 0);

  // The CPU name and s_nop padding bytes are the only rewrite state the fast
  // path needs; both are also carried by LLVMState on the MC path. Holding them
  // as standalone locals lets the shared tail work off them regardless of which
  // path ran, so the fast path never builds an LLVMState.
  const StringRef TargetCpu = TargetIdent.Processor;
  static constexpr uint8_t SNop[4] = {0x00, 0x00, 0x80, 0xbf};
  SmallVector<uint8_t, 4> SNopBytes(SNop, SNop + sizeof(SNop));

  // B0->B0 entry-only fast path: no instruction patches means no .text decode,
  // so skip the whole LLVM MC layer and emit entry stubs from a pre-encoded
  // byte template. UseB0B0EntryFastPath is decided by the caller from the
  // source/target stepping; the template bytes and the HWSD workaround are
  // gfx1250-specific, which that flag already accounts for.
  const bool UseFastAppend = Options.RunEntryTrampolines &&
                             !RunInstructionPatches &&
                             Options.UseB0B0EntryFastPath;

  // The MC layer (disassembler, encoder, register info) is only initialized on
  // the non-fast path. The fast path leaves LS default-constructed and unused:
  // it works entirely off TargetCpu / SNopBytes above, and every LS access
  // below is guarded by a condition that is false on the fast path.
  LLVMState LS;
  if (UseFastAppend) {
    log() << "hotswap: entry trampolines: B0->B0 fast path (no MC/.text "
             "disassembly)\n";
  } else {
    uint64_t InitT0 = Prof ? profNowNs() : 0;
    LS = initLLVM(TargetIdent);
    if (Prof)
      Profile.add(HotswapMetric::InitLLVM, profNowNs() - InitT0, 0);
    if (!LS.Valid) {
      log() << "hotswap: error: retargetCodeObject: initLLVM failed "
            << "for CPU '" << TargetIdent.Processor << "'; aborting rewrite.\n";
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  // Direct displacement is an entry-workaround optimization only. Apply the
  // entry prefixes before ordinary instruction rewriting so the existing
  // NOP-sled/trampoline planner sees the final instruction offsets. If the ELF
  // cannot be displaced safely, continue from the pristine working copy and
  // append the established entry stubs below.
  if (Options.RunEntryTrampolines && AllowTextDisplacement && !UseFastAppend) {
    std::vector<DisplacementEdit> EntryDisplacements;
    std::optional<uint32_t> EntryCount =
        collectKernelEntryDisplacements(Elf, LS, EntryDisplacements);
    if (!EntryCount)
      return AMD_COMGR_STATUS_ERROR;

    if (!EntryDisplacements.empty()) {
      Expected<std::unique_ptr<WritableMemoryBuffer>> DisplacedOrErr =
          tryApplyTextDisplacementToNewBuffer(Elf, LS, EntryDisplacements);
      if (DisplacedOrErr) {
        std::unique_ptr<WritableMemoryBuffer> Displaced =
            std::move(*DisplacedOrErr);
        if (!RunInstructionPatches) {
          Out = std::move(Displaced);
          return AMD_COMGR_STATUS_SUCCESS;
        }

        Gfx1250RewriteOptions RemainingOptions = Options;
        RemainingOptions.RunEntryTrampolines = false;
        RemainingOptions.UseB0B0EntryFastPath = false;
        return retargetCodeObjectImpl(Displaced->getBufferStart(),
                                      Displaced->getBufferSize(), TargetIdent,
                                      RemainingOptions, Out,
                                      /*AllowTextDisplacement=*/false, Profile);
      }

      log() << "hotswap: entry displacement unavailable: "
            << toString(DisplacedOrErr.takeError())
            << "; using appended entry stubs\n";
    }
  }

  RewriteConfig Config = makeGfx1250B0A0Config();
  Config.RunB0A0Patches = Options.RunB0A0Patches;
  Config.MaskPolicy = Options.MaskPolicy;

  uint8_t *Text = Elf.textData();
  uint64_t Count = 0;
  std::vector<Trampoline> Deferred;
  std::vector<ScratchPatchInfo> ScratchPatches;
  bool RequiredPatchApplied = false;
  if (RunInstructionPatches) {
    std::vector<InternalDecodedInst> Decoded;
    uint64_t DecodeT0 = Prof ? profNowNs() : 0;
    bool DecodedOk = decodeTextSection(Text, Elf.textSize(), LS, Decoded);
    if (Prof)
      Profile.add(HotswapMetric::Decode, profNowNs() - DecodeT0, 0);
    if (!DecodedOk) {
      log() << "hotswap: error: retargetCodeObject: decodeTextSection "
            << "failed on .text (" << Elf.textSize() << " bytes).\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    uint64_t DispatchT0 = Prof ? profNowNs() : 0;
    std::optional<uint32_t> Patched = applyGfx1250B0toA0Rules(
        Decoded, Text, Elf.textSize(), LS, Deferred, Elf, ScratchPatches,
        Config, RequiredPatchApplied, Profile);
    if (Prof)
      Profile.add(HotswapMetric::B0A0Dispatch, profNowNs() - DispatchT0, 0);
    if (!Patched)
      return AMD_COMGR_STATUS_ERROR;
    Count = *Patched;
    log() << "hotswap: applied " << Count << " instruction patches\n";
  } else {
    log() << "hotswap: instruction patches disabled for this rewrite\n";
  }

  // gfx1250 revision is recorded per kernel in the AMDGPU metadata note.
  // Running a B0 object on A0 requires retagging that metadata even when no
  // machine instruction needed rewriting.
  if (Options.RunB0A0Patches && !Elf.updateGfx1250RevisionMetadata("A0"))
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<WritableMemoryBuffer> Result;
  uint64_t PoolT0 = Prof ? profNowNs() : 0;
  std::vector<Trampoline> Growth = Deferred;
  // The appended pool's fresh virtual address is the single reference point for
  // all trampoline branch/stub targets (growWithTrampolines places it there).
  std::optional<uint64_t> PoolVAddrOr = Elf.trampolinePoolVAddr();
  if (!PoolVAddrOr) {
    log() << "hotswap: error: retargetCodeObject: could not compute trampoline "
          << "pool virtual address.\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  const uint64_t PoolVAddr = *PoolVAddrOr;
  // Pool is always >= textAddr(); checkedSubUint64 guards a malformed object.
  std::optional<uint64_t> PoolBaseOffsetOr = checkedSubUint64(
      PoolVAddr, Elf.textAddr(), "trampoline pool base offset");
  if (!PoolBaseOffsetOr)
    return AMD_COMGR_STATUS_ERROR;
  const uint64_t PoolBaseOffset = *PoolBaseOffsetOr;
  if (Prof)
    Profile.add(HotswapMetric::PoolSetup, profNowNs() - PoolT0, 0);
  if (!Deferred.empty()) {
    uint64_t FixupT0 = Prof ? profNowNs() : 0;
    bool FixupOk = fixupTrampolineBranches(Deferred, Text, PoolBaseOffset, LS);
    if (Prof)
      Profile.add(HotswapMetric::FixupTrampolines, profNowNs() - FixupT0, 0);
    if (!FixupOk) {
      if (RequiredPatchApplied) {
        log() << "hotswap: error: required patch trampoline branch fixup "
                 "failed; refusing to return the original unsafe code "
                 "object\n";
        return AMD_COMGR_STATUS_ERROR;
      }
      // A trampoline branch could not be encoded, so the local `Buf` copy
      // is half-redirected; shipping it would run corrupted code. Fall back
      // to the pristine input object (`ElfData`, untouched) so the loader
      // runs the original unpatched code instead.
      log() << "hotswap: error: some trampolines could not be fixed up; "
            << "falling back to the original (unpatched) code object\n";
      std::unique_ptr<WritableMemoryBuffer> Orig =
          WritableMemoryBuffer::getNewUninitMemBuffer(ElfSize);
      if (!Orig) {
        log() << "hotswap: error: retargetCodeObject: "
              << "getNewUninitMemBuffer(" << ElfSize
              << ") failed (out of memory) for the fallback copy.\n";
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
      }
      std::memcpy(Orig->getBufferStart(), ElfData, ElfSize);
      Out = std::move(Orig);
      // SUCCESS here is misleading the returned buffer is the
      // *unpatched* original, so callers cannot tell "rewrote successfully"
      // from "declined and fell back". The status vocabulary needs a distinct
      // "no-op / not-applied" code.
      return AMD_COMGR_STATUS_SUCCESS;
    }
    Growth = Deferred;
  }

  std::vector<KernelEntryTrampolineFixup> EntryFixups;
  if (Options.RunEntryTrampolines) {
    uint64_t EntryT0 = Prof ? profNowNs() : 0;
    std::optional<uint32_t> EntryCount =
        UseFastAppend
            ? appendKernelEntryTrampolinesFast(Elf, TargetCpu, Config.MaxSgprs,
                                               Growth, EntryFixups)
            : appendKernelEntryTrampolines(Elf, LS, Config.MaxSgprs, Growth,
                                           EntryFixups);
    if (Prof)
      Profile.add(HotswapMetric::EntryTrampolines, profNowNs() - EntryT0,
                  EntryCount.value_or(0));
    if (!EntryCount)
      return AMD_COMGR_STATUS_ERROR;
    Count += *EntryCount;
  } else {
    log() << "hotswap: kernel-entry trampolines disabled for this rewrite\n";
  }

  if (!Deferred.empty()) {
    uint64_t GuardT0 = Prof ? profNowNs() : 0;
    bool GuardOk = appendDeferredTrampolinePrefetchGuard(Elf, LS, Growth);
    if (Prof)
      Profile.add(HotswapMetric::PrefetchGuard, profNowNs() - GuardT0, 0);
    if (!GuardOk)
      return AMD_COMGR_STATUS_ERROR;
  }

  if (!Growth.empty()) {
    uint64_t GrowT0 = Prof ? profNowNs() : 0;
    Result = Elf.growWithTrampolines(Growth, SNopBytes);
    if (Prof)
      Profile.add(HotswapMetric::GrowElf, profNowNs() - GrowT0, 0);
    if (!Result) {
      log() << "hotswap: error: retargetCodeObject: "
            << "ElfView::growWithTrampolines returned null with "
            << Growth.size() << " trampolines queued.\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    size_t GrowthTotal = 0;
    for (const Trampoline &T : Growth) {
      if (T.Bytes.size() > std::numeric_limits<size_t>::max() - GrowthTotal) {
        log() << "hotswap: error: retargetCodeObject: growth byte count "
              << "overflows size_t.\n";
        return AMD_COMGR_STATUS_ERROR;
      }
      GrowthTotal += T.Bytes.size();
    }
    uint64_t DbgT0 = Prof ? profNowNs() : 0;
    patchDebugSections(*Result, Deferred, Elf, GrowthTotal);
    if (Prof)
      Profile.add(HotswapMetric::DebugSections, profNowNs() - DbgT0, 0);

    uint64_t KdT0 = Prof ? profNowNs() : 0;
    bool KdOk = rewriteKernelEntryDescriptorOffsets(*Result, PoolVAddr,
                                                    TargetCpu, EntryFixups);
    if (Prof)
      Profile.add(HotswapMetric::KdRewrite, profNowNs() - KdT0, 0);
    if (!KdOk)
      return AMD_COMGR_STATUS_ERROR;

    // Give each appended entry stub a `<kernel>.stub` symbol so a dispatch
    // whose entry now points at the stub still resolves to a name (e.g. rocgdb
    // `info dispatches`). This grows only the non-alloc .symtab/.strtab and
    // returns a new buffer; failure is non-fatal (the rewritten code object is
    // still correct, just missing the debug-only symbol).
    //
    // FAST PATH: this .symtab/.strtab rebuild + full buffer copy scales with
    // kernel count and is pure overhead for a load-time-critical path (the ROCr
    // loader trampoline adds no such symbols). The symbols are only a debugging
    // aid, so the fast path skips them by default. Set
    // AMD_COMGR_HOTSWAP_ENTRY_STUB_SYMBOLS=1 to re-enable (e.g. for rocgdb).
    const bool AddStubSymbols =
        !UseFastAppend || env::shouldAddEntryTrampolineSymbols();
    if (!EntryFixups.empty() && AddStubSymbols) {
      uint64_t SymT0 = Prof ? profNowNs() : 0;
      std::unique_ptr<WritableMemoryBuffer> WithSyms =
          addKernelEntryTrampolineSymbols(*Result, PoolVAddr, EntryFixups);
      if (Prof)
        Profile.add(HotswapMetric::SymbolInsert, profNowNs() - SymT0, 0);
      if (WithSyms)
        Result = std::move(WithSyms);
    }
  } else {
    uint64_t OutCopyT0 = Prof ? profNowNs() : 0;
    Result = copyOutputBuffer(Buf.data(), ElfSize, "patched");
    if (Prof)
      Profile.add(HotswapMetric::OutputCopy, profNowNs() - OutCopyT0, 0);
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (!ScratchPatches.empty()) {
    uint64_t VerifyT0 = Prof ? profNowNs() : 0;
    runScratchVerification(*Result, LS, ScratchPatches, Config.MaxVgprs);
    if (Prof)
      Profile.add(HotswapMetric::ScratchVerify, profNowNs() - VerifyT0, 0);
  }

  Out = std::move(Result);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<MemoryBuffer> &Out) {
  const bool RunInstructionPatches =
      Options.RunB0A0Patches ||
      Options.MaskPolicy != MaskWorkaroundPolicy::None;
  if (!RunInstructionPatches && !Options.RunEntryTrampolines) {
    std::unique_ptr<WritableMemoryBuffer> Result =
        copyOutputBuffer(ElfData, ElfSize, "no-op");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    Out = std::move(Result);
    return AMD_COMGR_STATUS_SUCCESS;
  }

  // One profiling session per code object, merged into TimeStatistics when it
  // goes out of scope. Prof gates the manual per-phase clock reads.
  HotswapProfile Profile(hotswapProfilingEnabled());
  // RAII guard: records phase:rewrite_total on every return path.
  [[maybe_unused]] HotswapProfile::Scope TotalScope =
      Profile.time(HotswapMetric::RewriteTotal);

  return retargetCodeObjectImpl(ElfData, ElfSize, TargetIdent, Options, Out,
                                /*AllowTextDisplacement=*/true, Profile);
}

} // namespace hotswap
} // namespace COMGR
