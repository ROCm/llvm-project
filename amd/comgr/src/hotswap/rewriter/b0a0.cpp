//===- b0a0.cpp - GFX1250 B0-to-A0 patch dispatcher -----------------------===//
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
/// list lives in patches.def; each entry corresponds to one
/// slot on the vtable and one register*Patch function in a sibling
/// patch-*.cpp. installHotswapPatches() walks the .def to
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
#include "internal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <limits>
#include <mutex>
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
// (driven off the same patches.def), so libamd_comgr and
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
#include "patches.def"
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

bool writeCurrentText(PatchContext &Ctx, uint64_t Offset,
                      ArrayRef<uint8_t> Bytes, StringRef Context) {
  if (Offset > Ctx.TextSize || Bytes.size() > Ctx.TextSize - Offset) {
    log() << "hotswap: error: " << Context << ": current .text write [0x"
          << utohexstr(Offset) << ", +0x" << utohexstr(Bytes.size())
          << ") exceeds size 0x" << utohexstr(Ctx.TextSize) << "\n";
    return false;
  }
  if (Bytes.empty())
    return true;
  std::memcpy(Ctx.Text + Offset, Bytes.data(), Bytes.size());
  noteCurrentTextMutation(Ctx);
  return true;
}

void noteCurrentTextMutation(PatchContext &Ctx) {
  Ctx.CurrentFunctionSgprLivenessCache.clear();
  if (Ctx.TextMutationGeneration == std::numeric_limits<uint64_t>::max()) {
    Ctx.TextMutationGeneration = 0;
    return;
  }
  ++Ctx.TextMutationGeneration;
}

void notePendingTrampolineMutation(PatchContext &Ctx, const Trampoline &T) {
  if (!T.HasFunctionRange) {
    Ctx.HasUnresolvedPendingTrampoline = true;
    return;
  }
  Ctx.PendingTrampolineFunctions.insert({T.FunctionStart, T.FunctionEnd});
}

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
    if (Bytes.size() != VccSaveRestoreBytes)
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
                          unsigned SgprBase, bool UseVcc,
                          bool PreserveVcc) {
  if (PreserveVcc && !UseVcc)
    return std::nullopt;
  if (!UseVcc && (SgprBase & 1u) != 0)
    return std::nullopt;

  uint64_t SetPcOffset = FromOffset;
  uint32_t PrefixBytes = 0;
  if (PreserveVcc) {
    std::optional<uint64_t> Offset = checkedAddUint64(
        FromOffset, VccSaveRestoreBytes,
        "VCC-preserving set-PC gateway layout offset");
    if (!Offset)
      return std::nullopt;
    SetPcOffset = *Offset;
    PrefixBytes = VccSaveRestoreBytes;
  }

  std::optional<uint32_t> SetPcBytes =
      getSetPcLongBranchLayoutSize(SetPcOffset, TargetOffset);
  if (!SetPcBytes)
    return std::nullopt;
  return PrefixBytes + *SetPcBytes;
}

Expected<std::optional<EncodedSetPcGateway>>
findNearestSetPcGateway(std::vector<NopSled> &Gateways, const LLVMState &LS,
                        uint64_t FromOffset, uint64_t TargetOffset,
                        unsigned SgprBase, bool UseVcc, bool PreserveVcc) {
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
    std::optional<uint32_t> LayoutSize =
        getSetPcGatewayLayoutSize(Sled.WritePos, TargetOffset, SgprBase,
                                  UseVcc, PreserveVcc);
    if (!LayoutSize)
      return createStringError(
          Twine("failed to encode set-PC gateway at candidate offset 0x") +
          utohexstr(Sled.WritePos));
    if (*LayoutSize > UsableEnd - Sled.WritePos)
      continue;

    Best = &Sled;
    BestLayoutSize = *LayoutSize;
    BestUsableEnd = UsableEnd;
    BestDistance = Distance;
  }
  if (!Best)
    return std::nullopt;
  std::optional<SmallVector<uint8_t>> BestBytes =
      encodeSetPcGateway(LS, Best->WritePos, TargetOffset, SgprBase, UseVcc,
                         PreserveVcc);
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
  std::string Owner =
      Ctx.Elf.findKernelAtAddress(TextOffset + Ctx.Elf.textAddr());
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

  std::string Owner =
      Ctx.Elf.findKernelAtAddress(TextOffset + Ctx.Elf.textAddr());
  bool ChargedOwner = false;

  // llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp::getNumExtraSGPRs returns
  // two non-numbered VCC SGPRs on GFX1250. Always include them in the metadata
  // requirement. This may conservatively overstate a kernel that does not use
  // VCC, but never mistakes VCC for numbered s0-s105 registers.
  constexpr unsigned VccSgprs = 2;
  unsigned RequiredSgprs = Block.Base + Block.Count + VccSgprs;
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
    KernelPatchStats &Stats = Ctx.KernelStats[KD.KernelName];
    Stats.ExtraSgprs = std::max(Stats.ExtraSgprs, RequiredSgprs - *Current);
  }

  if (!ChargedOwner) {
    log() << "hotswap: error: " << Context << ": kernel '" << Owner
          << "' has no descriptor\n";
    return false;
  }
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
    if (instructionWritesRegister(DI, LS, Register))
      return false;
  }
  return false;
}

bool isRegisterDefinitelyDeadAtContinuation(PatchContext &Ctx,
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
    if (instructionWritesRegister(DI, Ctx.LS, Register) ||
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
    ArrayRef<InternalDecodedInst>::const_iterator It =
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
    DenseMap<size_t, BitVector>::iterator It =
        IncomingAt.try_emplace(*Successor, MaxSgprs).first;
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

  unsigned Base = (Ctx.Config.MaxSgprs - 2) & ~1u;
  for (;;) {
    if (!Unsafe.test(Base) && !Unsafe.test(Base + 1)) {
      SafeSgprScratchBlock Scratch{Base, 2};
      if (commitSafeSgprScratchBlock(Ctx, InstOffset, Scratch,
                                     "locally dead far-return SGPR pair"))
        return Base;
    }
    if (Base == 0)
      break;
    Base -= 2;
  }
  return std::nullopt;
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
    log() << "hotswap: safe far return: reusing locally dead s[" << *LocalPair
          << ":" << *LocalPair + 1 << "] at 0x" << utohexstr(InstOffset)
          << "\n";
    return FarReturnScratch{/*Available=*/true, *LocalPair,
                            /*UseVcc=*/false, /*PreserveVcc=*/false};
  }

  if (InstSize >= VccPreservingSourceBytes) {
    std::string Owner =
        Ctx.Elf.findKernelAtAddress(InstOffset + Ctx.Elf.textAddr());
    std::optional<unsigned> WavefrontSize =
        Owner.empty() ? std::nullopt : Ctx.Elf.getKernelWavefrontSize(Owner);
    if (WavefrontSize == 32) {
      std::optional<SafeSgprScratchBlock> Save = findSafeSgprScratchBlock(
          Ctx, InstOffset, /*Count=*/1, /*Alignment=*/1,
          "VCC-preserving far return", /*ReportNoSpace=*/false);
      if (Save && commitSafeSgprScratchBlock(Ctx, InstOffset, *Save,
                                             "VCC-preserving far return")) {
        log() << "hotswap: safe far return: preserving live wave32 VCC_LO in s"
              << Save->Base << " at 0x" << utohexstr(InstOffset) << "\n";
        return FarReturnScratch{/*Available=*/true, Save->Base,
                                /*UseVcc=*/true, /*PreserveVcc=*/true};
      }
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
    BodyPrefix = Scratch.PreserveVcc ? VccSaveRestoreBytes : 0;
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
    SmallVector<uint8_t> Restore = assembleSingleInst(
        "s_mov_b32 vcc_lo, s" + std::to_string(Scratch.SgprBase), Ctx.LS);
    if (Restore.size() != VccSaveRestoreBytes)
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
  int64_t AddImmediate = 0;
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
        Candidate.Inst.getOperand(1).getReg() != TargetRegister ||
        !Candidate.Inst.getOperand(2).isImm())
      return std::nullopt;
    AddIndex = I;
    AddImmediate = Candidate.Inst.getOperand(2).getImm();
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
    return MaterializedPcSequence{
        *PcValue + static_cast<uint64_t>(AddImmediate), Candidate.Offset};
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

/// Read a constant 32-bit source operand through MC. A decoded SOP2 literal is
/// an immediate; an assembled operand may be an absolute expression. Anything
/// else (a register source) is not a constant displacement.
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
        !isExactRegisterOperand(DI.Inst, 1, Pair) ||
        !DI.Inst.getOperand(2).isImm())
      break;
    uint64_t Delta = static_cast<uint64_t>(DI.Inst.getOperand(2).getImm());
    return std::make_pair(I, *Pc + Delta);
  }

  if (GetPcIndex + 3 >= FunctionEndIndex)
    return std::nullopt;
  const InternalDecodedInst &MakeDelta = Decoded[GetPcIndex + 1];
  const InternalDecodedInst &AddLow = Decoded[GetPcIndex + 2];
  const InternalDecodedInst &AddHigh = Decoded[GetPcIndex + 3];
  if (MakeDelta.Inst.getOpcode() != LS.SAddCoI32Opcode ||
      AddLow.Inst.getOpcode() != LS.SAddU32Opcode ||
      AddHigh.Inst.getOpcode() != LS.SAddcU32Opcode ||
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
    if (Compare.Inst.getOpcode() != LS.SCompareGeI32Opcode ||
        Compare.Inst.getNumOperands() != 2 ||
        !isExactRegisterOperand(Compare.Inst, 0, DeltaReg) ||
        !Compare.Inst.getOperand(1).isImm() ||
        Compare.Inst.getOperand(1).getImm() != 0 ||
        Branch.Inst.getOpcode() != LS.SBranchScc1Opcode ||
        Abs.Inst.getOpcode() != LS.SAbsI32Opcode ||
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
                                     unsigned LowOpcode, unsigned HighOpcode) {
      if (Low.Inst.getOpcode() != LowOpcode ||
          High.Inst.getOpcode() != HighOpcode ||
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
    if (!matchesPairArithmetic(SubLow, SubHigh, LS.SSubU32Opcode,
                                LS.SSubbU32Opcode) ||
        !matchesPairArithmetic(AddLow, AddHigh, LS.SAddU32Opcode,
                                LS.SAddcU32Opcode))
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
  llvm::stable_sort(Candidates, [](const FiniteSetPcTransfer &LHS,
                                   const FiniteSetPcTransfer &RHS) {
    return std::tie(LHS.InstIndex, LHS.Target) <
           std::tie(RHS.InstIndex, RHS.Target);
  });
  Candidates.erase(std::unique(Candidates.begin(), Candidates.end(),
                               [](const FiniteSetPcTransfer &LHS,
                                  const FiniteSetPcTransfer &RHS) {
                                 return LHS.InstIndex == RHS.InstIndex &&
                                        LHS.Target == RHS.Target;
                               }),
                   Candidates.end());
  return Candidates;
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

static std::optional<uint64_t>
getDirectTextTarget(const InternalDecodedInst &DI, const LLVMState &LS,
                    uint64_t TextAddr, uint64_t TextEnd);

/// A reusable target value remains valid after a call only when the exact local
/// callee is fully decoded, returns through the call's link pair, and cannot
/// transitively or directly clobber the target pair.
static bool calleePreservesReusableTarget(
    uint64_t Target, MCRegister TargetRegister, MCRegister ReturnRegister,
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextEnd,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges) {
  const ElfView::FunctionTextRange *Callee = nullptr;
  for (const ElfView::FunctionTextRange &Range : FunctionRanges)
    if (Range.Begin == Target && Range.Begin >= TextAddr &&
        Range.End > Range.Begin && Range.End <= TextEnd &&
        (!Callee || Range.End < Callee->End))
      Callee = &Range;
  if (!Callee)
    return false;

  uint64_t CalleeBegin = Callee->Begin - TextAddr;
  uint64_t CalleeEnd = Callee->End - TextAddr;
  bool SawInstruction = false;
  for (const InternalDecodedInst &DI : Decoded) {
    if (DI.Offset < CalleeBegin || DI.Offset >= CalleeEnd)
      continue;
    SawInstruction = true;
    if (!DI.DecodeSucceeded || LS.MIA->isCall(DI.Inst) ||
        definesOverlappingRegister(DI, LS, TargetRegister))
      return false;

    if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
      if (DI.Inst.getNumOperands() != 1 ||
          !isExactRegisterOperand(DI.Inst, 0, ReturnRegister))
        return false;
      continue;
    }
    if (DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
        DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
        LS.MIA->isReturn(DI.Inst))
      continue;
    if (LS.MIA->isIndirectBranch(DI.Inst) ||
        DI.Inst.getOpcode() == LS.SAddPcI64Opcode)
      return false;

    bool HasFallthrough = true;
    if (LS.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> BranchTarget =
          getDirectTextTarget(DI, LS, TextAddr, TextEnd);
      if (!BranchTarget || *BranchTarget < CalleeBegin ||
          *BranchTarget >= CalleeEnd)
        return false;
      HasFallthrough = !LS.MIA->isUnconditionalBranch(DI.Inst);
    }
    if (!HasFallthrough)
      continue;
    std::optional<uint64_t> Fallthrough =
        checkedAddUint64(DI.Offset, DI.Size, "reusable callee fallthrough");
    if (!Fallthrough || *Fallthrough >= CalleeEnd)
      return false;
  }
  return SawInstruction;
}

/// Resolve register calls whose target pair is selected once and reused across
/// control flow. A monotone intraprocedural solver propagates finite target
/// sets from proven get-PC materializations. Any unrecognized pair definition
/// introduces Unknown, so a bypass around the selector remains fail-closed.
static std::vector<ReachingCallTargets> resolveReusablePcCallTargets(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextEnd,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<std::optional<PcMaterializedCallInfo>> LocalCalls,
    ArrayRef<uint64_t> DeclaredEntries,
    ArrayRef<FiniteSetPcTransfer> FiniteSetPcTransfers = {}) {
  std::vector<ReachingCallTargets> Resolved(Decoded.size());
  SmallVector<ReachingCallGroup, 8> Groups;
  DenseMap<size_t, const FiniteSetPcTransfer *> FiniteSetPcByInst;
  for (const FiniteSetPcTransfer &Transfer : FiniteSetPcTransfers)
    FiniteSetPcByInst.try_emplace(Transfer.InstIndex, &Transfer);

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &Call = Decoded[I];
    if (LocalCalls[I] || !Call.DecodeSucceeded ||
        Call.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
        Call.Inst.getNumOperands() < 2)
      continue;
    const MCOperand &TargetOp =
        Call.Inst.getOperand(Call.Inst.getNumOperands() - 1);
    if (!TargetOp.isReg() || !TargetOp.getReg())
      continue;
    MCRegister TargetRegister(TargetOp.getReg());

    const ElfView::FunctionTextRange *Best = nullptr;
    uint64_t Address = TextAddr + Call.Offset;
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

  SmallVector<std::pair<size_t, uint64_t>, 16> DirectEntries;
  for (size_t SourceIndex = 0; SourceIndex != Decoded.size(); ++SourceIndex) {
    const InternalDecodedInst &Source = Decoded[SourceIndex];
    if (!Source.DecodeSucceeded ||
        (!LS.MIA->isBranch(Source.Inst) && !LS.MIA->isCall(Source.Inst)) ||
        LS.MIA->isReturn(Source.Inst))
      continue;
    std::optional<uint64_t> Target =
        getDirectTextTarget(Source, LS, TextAddr, TextEnd);
    if (Target)
      DirectEntries.emplace_back(SourceIndex, *Target);
  }
  DenseMap<std::pair<uint64_t, uint64_t>, bool> CalleePreservation;

  for (const ReachingCallGroup &Group : Groups) {
    ArrayRef<InternalDecodedInst>::const_iterator Begin =
        llvm::lower_bound(Decoded, Group.Begin,
                          [](const InternalDecodedInst &DI, uint64_t Offset) {
                            return DI.Offset < Offset;
                          });
    ArrayRef<InternalDecodedInst>::const_iterator End = std::lower_bound(
        Begin, Decoded.end(), Group.End,
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
    SmallVector<size_t, 32> Worklist;
    BitVector Queued(EndIndex - BeginIndex);
    auto seedUnknownEntry = [&](size_t Index) {
      ReachingPcState &Entry = Before[Index - BeginIndex];
      Entry.Reached = true;
      Entry.HasUnknown = true;
      if (!Queued.test(Index - BeginIndex)) {
        Worklist.push_back(Index);
        Queued.set(Index - BeginIndex);
      }
    };
    seedUnknownEntry(BeginIndex);

    // Every declared entry and direct cross-function target is an independent
    // root. An entry into a materialization interior must not inherit the
    // token established by the containing function's ordinary entry path.
    for (uint64_t Entry : DeclaredEntries) {
      DenseMap<uint64_t, size_t>::const_iterator EntryIndex =
          OffsetToIndex.find(Entry);
      if (EntryIndex != OffsetToIndex.end() &&
          EntryIndex->second >= BeginIndex && EntryIndex->second < EndIndex)
        seedUnknownEntry(EntryIndex->second);
    }
    for (const std::pair<size_t, uint64_t> &Entry : DirectEntries) {
      if (Entry.first >= BeginIndex && Entry.first < EndIndex)
        continue;
      DenseMap<uint64_t, size_t>::const_iterator TargetIndex =
          OffsetToIndex.find(Entry.second);
      if (TargetIndex != OffsetToIndex.end() &&
          TargetIndex->second >= BeginIndex &&
          TargetIndex->second < EndIndex)
        seedUnknownEntry(TargetIndex->second);
    }

    while (!Worklist.empty()) {
      size_t I = Worklist.pop_back_val();
      Queued.reset(I - BeginIndex);
      ReachingPcState State = Before[I - BeginIndex];
      const InternalDecodedInst &DI = Decoded[I];

      DenseMap<size_t, size_t>::const_iterator Starter = Starters.find(I);
      DenseMap<size_t, uint64_t>::const_iterator Completion =
          Completions.find(I);
      if (!DI.DecodeSucceeded) {
        // An undecoded slot has an unknown effect on the target pair and on
        // control flow, so any reaching value it carries forward is unproven.
        // Fail closed to Unknown; the finite-state test below then refuses
        // every reusable call this path reaches. Starters and Completions are
        // built only from decoded materializations, so neither can name I.
        State.HasUnknown = true;
        State.Targets.clear();
        State.ActiveMaterializations.clear();
      } else if (Starter != Starters.end()) {
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

      bool IsReusableCall = llvm::is_contained(Group.Calls, I);
      bool HasFiniteState = !State.HasUnknown &&
                            State.ActiveMaterializations.empty() &&
                            !State.Targets.empty();
      // Recompute Resolved[I] on every visit so a later reconvergent path that
      // makes this call Unknown erases an earlier finite result. Writing only
      // on finite state would leave a stale target set from the first visit.
      if (IsReusableCall)
        Resolved[I] = HasFiniteState ? State.Targets : ReachingCallTargets();

      // The first call after a straight-line materialization is also resolved
      // by the one-shot matcher. Let that bootstrap call preserve the reaching
      // value only when both analyses prove the same singleton target through
      // this exact target pair.
      bool IsMatchingBootstrapCall = false;
      if (LocalCalls[I] && HasFiniteState && State.Targets.size() == 1 &&
          State.Targets.front() == LocalCalls[I]->Target &&
          DI.Inst.getNumOperands() >= 2) {
        const MCOperand &TargetOp =
            DI.Inst.getOperand(DI.Inst.getNumOperands() - 1);
        IsMatchingBootstrapCall =
            TargetOp.isReg() && TargetOp.getReg() == Group.TargetRegister;
      }
      bool CalleesPreserve = (IsReusableCall || IsMatchingBootstrapCall) &&
                             HasFiniteState && DI.Inst.getNumOperands() != 0 &&
                             DI.Inst.getOperand(0).isReg() &&
                             DI.Inst.getOperand(0).getReg();
      if (CalleesPreserve) {
        MCRegister ReturnRegister(DI.Inst.getOperand(0).getReg());
        uint64_t RegisterKey = static_cast<uint64_t>(Group.TargetRegister.id())
                                   << 32 |
                               ReturnRegister.id();
        for (uint64_t Target : State.Targets) {
          std::pair<uint64_t, uint64_t> Key{Target, RegisterKey};
          DenseMap<std::pair<uint64_t, uint64_t>, bool>::iterator Cached =
              CalleePreservation.find(Key);
          if (Cached == CalleePreservation.end())
            Cached =
                CalleePreservation
                    .try_emplace(Key, calleePreservesReusableTarget(
                                          Target, Group.TargetRegister,
                                          ReturnRegister, Decoded, LS, TextAddr,
                                          TextEnd, FunctionRanges))
                    .first;
          CalleesPreserve &= Cached->second;
        }
      }

      if (LS.MIA->isCall(DI.Inst) && !CalleesPreserve) {
        // MC call operands do not describe transitive clobbers. Carry the
        // finite set past this call only after proving every exact local
        // target preserves the reusable target pair.
        State.HasUnknown = true;
        State.Targets.clear();
        State.ActiveMaterializations.clear();
      }

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
  uint64_t TextSize = TextEnd - TextAddr;
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
        if (*Continuation >= TextSize ||
            (*Continuation & (MinInstSize - 1)) != 0) {
          log() << "hotswap: call at 0x" << utohexstr(DI.Offset)
                << " has no aligned continuation inside .text\n";
          return std::nullopt;
        }
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

  for (const ExternalCallContinuation &Call :
       Index.ExternalCallContinuations) {
    uint64_t Source = Decoded[Call.InstIndex].Offset;
    if (Call.Continuation >= ChainBegin &&
        Call.Continuation < FunctionBegin) {
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

  DenseMap<uint64_t, size_t> OffsetToIndex;
  for (size_t I = 0; I != Decoded.size(); ++I)
    OffsetToIndex.try_emplace(Decoded[I].Offset, I);

  SmallVector<SymbolLessReturnRegion, 8> Regions;
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
    if (Safe)
      Regions.push_back(std::move(Region));
  }

  // Two independently inferred call entries may not claim a shared body.
  BitVector Claimed(Decoded.size());
  BitVector Overlap(Regions.size());
  for (size_t I = 0; I != Regions.size(); ++I)
    for (size_t InstIndex : Regions[I].Instructions) {
      if (Claimed.test(InstIndex)) {
        Overlap.set(I);
        for (size_t J = 0; J != I; ++J)
          if (llvm::is_contained(Regions[J].Instructions, InstIndex))
            Overlap.set(J);
      }
      Claimed.set(InstIndex);
    }
  SmallVector<SymbolLessReturnRegion, 8> Disjoint;
  for (size_t I = 0; I != Regions.size(); ++I)
    if (!Overlap.test(I))
      Disjoint.push_back(std::move(Regions[I]));
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
  auto markUnboundedIndirectEntry = [&](StringRef Reason,
                                        std::optional<uint64_t> Offset =
                                            std::nullopt) {
    Audit.Closed = false;
    Audit.HasUnboundedIndirectEntries = true;
    log() << "hotswap: finite control-flow audit: " << Reason;
    if (Offset)
      log() << " at offset 0x" << utohexstr(*Offset);
    log() << "\n";
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
    if (!Safe) {
      Audit.InvalidSetPcCandidates.set(CandidateIndex);
      log() << "hotswap: finite set-PC candidate rejected at offset 0x"
            << utohexstr(SequenceStart)
            << ": reachable control-flow entry overlaps its materialization\n";
    }
  }

  if (Audit.InvalidSetPcCandidates.any()) {
    log() << "hotswap: finite-control-flow audit rejected "
          << Audit.InvalidSetPcCandidates.count()
          << " set-PC candidate(s)\n";
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
        markUnboundedIndirectEntry(
            "symbol-less return is outside its inferred region",
            Decoded[Return].Offset);
    }
  for (const BoundedSetPcReturn &Return : BoundedReturns)
    if (ProvisionalOwners.contains(Return.InstIndex))
      ++PublishedProvisionalReturns[Return.InstIndex];
  for (const std::pair<size_t, unsigned> &Owner : ProvisionalOwners)
    if (Owner.second != 1 ||
        PublishedProvisionalReturns.lookup(Owner.first) != 1)
      markUnboundedIndirectEntry(
          "symbol-less return ownership is not unique",
          Decoded[Owner.first].Offset);

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
          markUnboundedIndirectEntry(
              "bounded return is not owned by its inferred region",
              Decoded[Return.InstIndex].Offset);
        continue;
      }
      for (uint64_t Target : Return.Targets)
        if (containsInstructionByte(Target)) {
          markUnboundedIndirectEntry(
              "return from another region enters an inferred region", Target);
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
      markUnboundedIndirectEntry(
          "declared entry is not an instruction boundary", Entry);
  for (uint64_t Entry : ExternalEntries)
    if (Entry < TextSize && !InstructionOffsets.contains(Entry))
      markUnboundedIndirectEntry(
          "external entry is not an instruction boundary", Entry);
  for (const ElfView::FunctionTextRange &Range : FunctionRanges)
    if (Range.Begin >= TextAddr && Range.Begin - TextAddr < TextSize &&
        !InstructionOffsets.contains(Range.Begin - TextAddr))
      markUnboundedIndirectEntry(
          "function entry is not an instruction boundary",
          Range.Begin - TextAddr);
  for (const DirectTargetSource &Source : Index.DirectTargetsByTarget)
    if (Reachable.test(Source.InstIndex) && Source.Target < TextSize &&
        !InstructionOffsets.contains(Source.Target))
      markUnboundedIndirectEntry(
          "direct target is not an instruction boundary", Source.Target);
  for (const KnownCallSite &Call : Index.Calls) {
    if (!Reachable.test(Call.InstIndex))
      continue;
    if (Call.Target < TextSize && !InstructionOffsets.contains(Call.Target))
      markUnboundedIndirectEntry(
          "finite call target is not an instruction boundary", Call.Target);
    if (Call.Continuation < TextSize &&
        !InstructionOffsets.contains(Call.Continuation))
      markUnboundedIndirectEntry(
          "finite call continuation is not an instruction boundary",
          Call.Continuation);
  }
  for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations)
    if (Reachable.test(Call.InstIndex) && Call.Continuation < TextSize &&
        !InstructionOffsets.contains(Call.Continuation))
      markUnboundedIndirectEntry(
          "external call continuation is not an instruction boundary",
          Call.Continuation);
  for (size_t SetPc : Index.SetPcIndices)
    if (Reachable.test(SetPc) && !BoundedSetPc.test(SetPc))
      markUnboundedIndirectEntry("reachable set-PC transfer is unbounded",
                                 Decoded[SetPc].Offset);

  for (size_t InstIndex : Index.UnboundedIndirectIndices)
    if (Reachable.test(InstIndex))
      markUnboundedIndirectEntry("reachable indirect transfer is unbounded",
                                 Decoded[InstIndex].Offset);

  // Every call is also an indirect entry source until either a finite local
  // target or a finite external target has been recorded for it.
  BitVector FiniteCalls(Decoded.size());
  for (const KnownCallSite &Call : Index.Calls)
    FiniteCalls.set(Call.InstIndex);
  for (const ExternalCallContinuation &Call : Index.ExternalCallContinuations)
    FiniteCalls.set(Call.InstIndex);
  for (size_t InstIndex : Index.BranchOrCallIndices) {
    if (!Reachable.test(InstIndex) || !LS.MIA->isCall(Decoded[InstIndex].Inst))
      continue;
    if (!FiniteCalls.test(InstIndex))
      markUnboundedIndirectEntry("reachable call target is unresolved",
                                 Decoded[InstIndex].Offset);
  }
  for (const BoundedSetPcReturn &Return : BoundedReturns) {
    if (!Reachable.test(Return.InstIndex))
      continue;
    for (uint64_t Target : Return.Targets)
      if (Target < TextSize && !InstructionOffsets.contains(Target))
        markUnboundedIndirectEntry(
            "bounded return target is not an instruction boundary", Target);
  }
  for (int I = Reachable.find_first(); I >= 0; I = Reachable.find_next(I))
    if (!Decoded[static_cast<size_t>(I)].DecodeSucceeded)
      markUnboundedIndirectEntry(
          "reachable instruction failed to decode",
          Decoded[static_cast<size_t>(I)].Offset);
  return Audit;
}

static bool
hasKnownControlFlowEntry(ArrayRef<uint64_t> DeclaredEntries,
                         ArrayRef<BoundedSetPcReturn> BoundedReturns,
                         const DenseMap<size_t, size_t> &BoundedReturnPositions,
                         const ControlFlowScanIndex &Index,
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
  if (Index.HasUnboundedIndirectEntry)
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
    ArrayRef<uint64_t> ExternalEntries, ArrayRef<uint8_t> Text) {
  if (!LS.MIA) {
    log() << "hotswap: MC branch analysis is unavailable; adjacent far "
             "trampolines will not be coalesced\n";
    return std::nullopt;
  }

  std::optional<uint64_t> TextEnd =
      checkedAddUint64(TextAddr, TextSize, "direct target text end");
  if (!TextEnd)
    return std::nullopt;

  SmallVector<std::optional<PcMaterializedCallInfo>, 16> MaterializedCalls(
      Decoded.size());
  for (size_t I = 0; I != Decoded.size(); ++I)
    MaterializedCalls[I] = matchPcMaterializedCall(Decoded, I, LS, TextAddr);

  SmallVector<FiniteSetPcTransfer, 8> AllSetPcCandidates =
      collectFiniteSetPcCandidates(Decoded, LS, TextAddr, *TextEnd,
                                   FunctionRanges);
  BitVector RejectedSetPcCandidates(AllSetPcCandidates.size());
  BitVector ProvenSetPcCandidates(AllSetPcCandidates.size());
  SmallVector<FiniteSetPcTransfer, 8> EnabledSetPcTransfers;
  std::vector<ReachingCallTargets> ReusableCalls;
  std::optional<ControlFlowScanIndex> Index;
  SmallVector<BoundedSetPcReturn, 2> BoundedReturns;
  SmallVector<BoundedSetPcReturn, 2> LocalFunctionReturns;
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
        Decoded, LS, TextAddr, *TextEnd, FunctionRanges, MaterializedCalls,
        DeclaredEntries, EnabledSetPcTransfers);
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
    LocalFunctionReturns = BoundedReturns;
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
      log() << "hotswap: finite-control-flow audit collapsed with "
            << EnabledSetPcTransfers.size()
            << " enabled set-PC transfer(s); rebuilding from roots\n";
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

  if (IndirectControlFlowClosed)
    for (const FiniteSetPcTransfer &Transfer : EnabledSetPcTransfers) {
      SmallVector<uint64_t, 2> Targets;
      if (Transfer.LocalTargetIndex)
        Targets.push_back(Decoded[*Transfer.LocalTargetIndex].Offset);
      BoundedReturns.push_back({Transfer.InstIndex, std::move(Targets)});
    }
  indexKnownCalls(*Index);

  // A non-closed object cannot publish bounded indirect transfers, but a
  // symbol-backed local return has an independent function-boundary proof.
  // Retain those local facts only for checking concrete alternate entries
  // into one-shot PC materializations.
  ArrayRef<BoundedSetPcReturn> EntryProofReturns =
      IndirectControlFlowClosed ? ArrayRef(BoundedReturns)
                                : ArrayRef(LocalFunctionReturns);
  DenseMap<size_t, size_t> EntryProofReturnPositions;
  for (size_t I = 0; I != EntryProofReturns.size(); ++I)
    EntryProofReturnPositions.try_emplace(EntryProofReturns[I].InstIndex, I);

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
            DeclaredEntries, EntryProofReturns, EntryProofReturnPositions,
            *Index, Entry.second.SequenceStart, Entry.second.SequenceEnd)) {
      ReusableCalls[I].clear();
      continue;
    }
    LocallyProvenMaterializedCalls.set(I);
  }

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
  for (const ExternalCallContinuation &Call :
       Index->ExternalCallContinuations)
    Info.Targets.insert(Call.Continuation);
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
            !hasKnownControlFlowEntry(DeclaredEntries, EntryProofReturns,
                                      EntryProofReturnPositions, *Index,
                                      Materialized->second.SequenceStart,
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
      } else if (DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isReg()) {
        log() << "hotswap: resolved PC-materialized call at 0x"
              << utohexstr(DI.Offset) << " to finite external target 0x"
              << utohexstr(*Target) << "\n";
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
  if (!IndirectControlFlowClosed && HasUnboundedIndirectEntries)
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
          Prev.LongBranchPreservesVcc ? VccSaveRestoreBytes : 0;
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
    size_t BodyPrefix = Prev.LongBranchPreservesVcc ? VccSaveRestoreBytes : 0;
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
      size_t BodyPrefix = T.LongBranchPreservesVcc ? VccSaveRestoreBytes : 0;
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
      std::optional<uint32_t> LayoutSize =
          getSetPcGatewayLayoutSize(Candidate, TargetOffset, SgprBase, UseVcc,
                                    PreserveVcc);
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

static std::optional<SmallVector<uint64_t, 4>>
allocateForwardBranchIslands(std::vector<NopSled> &Gateways,
                             uint64_t FromOffset, uint64_t TargetOffset) {
  struct Allocation {
    size_t SledIndex = 0;
    uint64_t PreviousWritePos = 0;
  };
  SmallVector<Allocation, 4> Allocations;
  SmallVector<uint64_t, 4> Islands;
  DenseSet<size_t> UsedSleds;
  uint64_t Current = FromOffset;

  while (!isSBranchReachable(Current, TargetOffset)) {
    bool Forward = TargetOffset > Current;
    size_t BestIndex = Gateways.size();
    uint64_t BestOffset = 0;
    for (size_t I = 0; I != Gateways.size(); ++I) {
      NopSled &Sled = Gateways[I];
      if (UsedSleds.contains(I) || FromOffset < Sled.FunctionStart ||
          FromOffset >= Sled.FunctionEnd)
        continue;
      uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
      bool MakesProgress =
          Forward ? Sled.WritePos > Current && Sled.WritePos < TargetOffset
                  : Sled.WritePos < Current && Sled.WritePos > TargetOffset;
      if (!MakesProgress || Sled.WritePos > UsableEnd ||
          MinInstSize > UsableEnd - Sled.WritePos ||
          !isSBranchReachable(Current, Sled.WritePos))
        continue;
      if (BestIndex == Gateways.size() ||
          (Forward ? Sled.WritePos > BestOffset : Sled.WritePos < BestOffset)) {
        BestIndex = I;
        BestOffset = Sled.WritePos;
      }
    }

    if (BestIndex == Gateways.size()) {
      for (size_t I = Allocations.size(); I != 0; --I) {
        const Allocation &A = Allocations[I - 1];
        Gateways[A.SledIndex].WritePos = A.PreviousWritePos;
      }
      return std::nullopt;
    }

    NopSled &Best = Gateways[BestIndex];
    Allocations.push_back({BestIndex, Best.WritePos});
    Islands.push_back(Best.WritePos);
    Current = Best.WritePos;
    Best.WritePos += MinInstSize;
    UsedSleds.insert(BestIndex);
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
    if (!T.Long || T.OriginalSize < 2 * MinInstSize ||
        isSBranchReachable(T.OriginalOffset, ThisTP))
      continue;
    std::optional<SmallVector<uint8_t>> Direct = encodeSetPCLongBranch(
        Ctx.LS, T.OriginalOffset, ThisTP, T.LongBranchSgprBase);
    if (Direct && Direct->size() <= T.OriginalSize)
      continue;
    std::optional<SafeSgprScratchBlock> Scratch = findSafeSgprScratchBlock(
        Ctx, T.OriginalOffset, /*Count=*/4,
        /*Alignment=*/2, "shared far-dispatch gateway");
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
  if (Candidates.size() < 8)
    return true;

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
      uint64_t From = SeedT.OriginalOffset + MinInstSize;
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
    std::vector<NopSled> WorkingGateways;
    SmallVector<uint64_t, 4> SeedIslands;
    if (SledIndex != TextGateways.size()) {
      GatewayOffset = TextGateways[SledIndex].WritePos;
      WorkingGateways = TextGateways;
      WorkingGateways[SledIndex].WritePos += SetPcForwardSequenceBytes;
    } else {
      size_t BestIslandCount = std::numeric_limits<size_t>::max();
      for (size_t I = 0; I != TextGateways.size(); ++I) {
        const NopSled &Sled = TextGateways[I];
        uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
        if (Sled.WritePos > UsableEnd ||
            SetPcForwardSequenceBytes > UsableEnd - Sled.WritePos)
          continue;
        std::vector<NopSled> Trial = TextGateways;
        uint64_t TrialGateway = Trial[I].WritePos;
        Trial[I].WritePos += SetPcForwardSequenceBytes;
        std::optional<SmallVector<uint64_t, 4>> Islands =
            allocateForwardBranchIslands(
                Trial, SeedT.OriginalOffset + MinInstSize, TrialGateway);
        if (!Islands || Islands->empty() || Islands->size() >= BestIslandCount)
          continue;
        BestIslandCount = Islands->size();
        GatewayOffset = TrialGateway;
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
          uint64_t SourceBranch = SeedT.OriginalOffset + MinInstSize;
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
      uint64_t ProposedSpan =
          8 + 28 * (Members.size() + 1) + GroupBodyBytes + T.Bytes.size();
      if (ProposedSpan > MaxDispatcherSpan)
        continue;
      uint64_t From = T.OriginalOffset + MinInstSize;
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
      RelayAnchors.push_back(
          {Ctx.OutTrampolines[Index].OriginalOffset + MinInstSize, Index});
    }
    llvm::sort(RelayAnchors);
    for (const Candidate &C : Candidates) {
      if (Members.size() == MaxGroupSites || Assigned.test(C.Index) ||
          LocalMembers.contains(C.Index) || C.ScratchBase != Seed.ScratchBase)
        continue;
      const Trampoline &T = Ctx.OutTrampolines[C.Index];
      uint64_t ProposedSpan =
          8 + 28 * (Members.size() + 1) + GroupBodyBytes + T.Bytes.size();
      if (ProposedSpan > MaxDispatcherSpan)
        continue;
      uint64_t From = T.OriginalOffset + MinInstSize;
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
      RelayAnchors.insert(
          llvm::lower_bound(RelayAnchors, std::make_pair(From, C.Index)),
          {From, C.Index});
    }
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
    if (Prefix > std::numeric_limits<uint32_t>::max()) {
      log() << "hotswap: shared far-dispatch group " << Group << " dispatcher "
            << "prefix (" << Prefix << " bytes for " << Count
            << " site(s)) exceeds the 32-bit pool-entry limit\n";
      return false;
    }
    First.PoolEntryPrefixBytes = static_cast<uint32_t>(Prefix);
    First.Bytes.insert(First.Bytes.begin(), Prefix, uint8_t{0});
    I += Count;
  }

  log() << "hotswap: planned " << Groups.size()
        << " shared far-dispatch gateway group(s) for " << Assigned.count()
        << " source site(s)\n";
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
                              uint64_t TargetOffset) {
  struct Allocation {
    size_t SledIndex = 0;
    uint64_t PreviousWritePos = 0;
  };
  SmallVector<Allocation, 4> Allocations;
  SmallVector<uint64_t, 4> Islands;
  DenseSet<size_t> UsedSleds;
  uint64_t Current = FromOffset;

  while (!isSBranchReachable(Current, TargetOffset)) {
    size_t BestIndex = Gateways.size();
    uint64_t BestOffset = std::numeric_limits<uint64_t>::max();
    for (size_t I = 0; I != Gateways.size(); ++I) {
      NopSled &Sled = Gateways[I];
      if (UsedSleds.contains(I) || OwnerOffset < Sled.FunctionStart ||
          OwnerOffset >= Sled.FunctionEnd)
        continue;
      uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
      if (Sled.WritePos <= TargetOffset || Sled.WritePos >= Current ||
          Sled.WritePos > UsableEnd ||
          MinInstSize > UsableEnd - Sled.WritePos ||
          !isSBranchReachable(Current, Sled.WritePos))
        continue;
      if (BestIndex == Gateways.size() || Sled.WritePos < BestOffset) {
        BestIndex = I;
        BestOffset = Sled.WritePos;
      }
    }

    if (BestIndex == Gateways.size()) {
      for (size_t I = Allocations.size(); I != 0; --I) {
        const Allocation &A = Allocations[I - 1];
        Gateways[A.SledIndex].WritePos = A.PreviousWritePos;
      }
      return std::nullopt;
    }

    NopSled &Best = Gateways[BestIndex];
    Allocations.push_back({BestIndex, Best.WritePos});
    Islands.push_back(Best.WritePos);
    Current = Best.WritePos;
    Best.WritePos += MinInstSize;
    UsedSleds.insert(BestIndex);
  }
  return Islands;
}

static bool
assignLongBranchGateways(PatchContext &Ctx,
                         const DenseSet<uint64_t> &DirectBranchTargets,
                         bool AllowTextGateways) {
  std::vector<NopSled> Gateways;
  if (AllowTextGateways) {
    Gateways = buildExternalGatewaySleds(
        Ctx.Decoded, Ctx.LS, Ctx.Elf, ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize),
        DirectBranchTargets);
    for (const NopSled &Sled : Ctx.NopSleds)
      Gateways.push_back(Sled);
    if (!planSharedDispatchGateways(Ctx, Gateways) ||
        !emitSharedDispatchers(Ctx))
      return false;
  }

  DenseMap<uint64_t, size_t> PoolIslandOwners;
  DenseMap<uint64_t, size_t> SourceTailIslandOwners;
  uint64_t IslandLayoutOffset = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    std::optional<uint64_t> Next = checkedAddUint64(
        IslandLayoutOffset, T.Bytes.size(), "pool branch-island layout");
    if (!Next)
      return false;
    if (T.HasPoolBranchIsland) {
      T.PoolBranchIslandOffset = *Next - PoolBranchIslandBytes;
      PoolIslandOwners[T.PoolBranchIslandOffset] = I;
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
    if (T.UsesSharedDispatcherForward)
      continue;

    if (!T.UsesSetPCBack) {
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
      std::optional<SmallVector<uint64_t, 4>> ReturnIslands =
          allocateBackwardBranchIslands(Gateways, T.OriginalOffset, BackSlot,
                                        *ReturnTo);
      if (!ReturnIslands) {
        log() << "hotswap: error: no safe return s_branch island chain for "
                 "far site 0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
      T.ReturnBranchIslands = std::move(*ReturnIslands);
      T.ReturnBranchTargetOffset = *ReturnTo;
      ReturnBranchIslandChains += !T.ReturnBranchIslands.empty();
    }

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

  // Once a source is replaced by a one-dword branch, the remainder of its
  // original instruction window is unreachable and can provide a safe relay.
  // Add these only after selecting direct set-PC sources, whose longer forward
  // sequence consumes the tail. Shared dispatch and VCC preservation likewise
  // reserve the second dword. Relays are object-wide: unlike an arbitrary NOP
  // sled they cannot be reached by the owning function's original fallthrough.
  if (!Ctx.DirectControlFlow.HasUnboundedIndirectEntries)
    for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
      const Trampoline &T = Ctx.OutTrampolines[I];
      if (T.OriginalSize < 2 * MinInstSize || T.UsesDirectSetPCForward ||
          T.UsesSharedDispatcherForward || T.LongBranchPreservesVcc)
        continue;
      uint64_t Tail = T.OriginalOffset + MinInstSize;
      SourceTailIslandOwners[Tail] = I;
      Gateways.push_back({Tail, Tail + MinInstSize, Tail, 0,
                          std::numeric_limits<uint64_t>::max()});
    }

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

  std::stable_sort(Pending.begin(), Pending.end(),
                   [](const PendingGateway &LHS, const PendingGateway &RHS) {
                     return LHS.InitialCandidateSlots <
                            RHS.InitialCandidateSlots;
                   });

  std::vector<PendingGateway> StillPending;
  StillPending.reserve(Pending.size());
  uint64_t AssignedGateways = 0;
  for (const PendingGateway &P : Pending) {
    Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    if (!T.UsesSetPCBack) {
      StillPending.push_back(P);
      continue;
    }
    Expected<std::optional<EncodedSetPcGateway>> GatewayOrErr =
        findNearestSetPcGateway(Gateways, Ctx.LS, T.OriginalOffset,
                                P.TargetOffset, T.LongBranchSgprBase,
                                T.LongBranchUsesVcc, T.LongBranchPreservesVcc);
    if (!GatewayOrErr) {
      log() << "hotswap: error: failed to plan gateway for far site 0x"
            << utohexstr(T.OriginalOffset) << ": "
            << toString(GatewayOrErr.takeError()) << "\n";
      return false;
    }
    std::optional<EncodedSetPcGateway> Gateway = std::move(*GatewayOrErr);
    if (!Gateway) {
      StillPending.push_back(P);
      continue;
    }
    T.HasForwardGateway = true;
    T.ForwardGatewayOffset = Gateway->Sled->WritePos;
    T.ForwardGatewayBytes = std::move(Gateway->Bytes);
    Gateway->Sled->WritePos += T.ForwardGatewayBytes.size();
    ++AssignedGateways;
  }
  Pending = std::move(StillPending);

  uint64_t BranchIslandChains = 0;
  StillPending.clear();
  StillPending.reserve(Pending.size());
  for (const PendingGateway &P : Pending) {
    Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    std::optional<SmallVector<uint64_t, 4>> Islands =
        allocateForwardBranchIslands(Gateways, T.OriginalOffset,
                                     P.TargetOffset);
    if (!Islands || Islands->empty()) {
      StillPending.push_back(P);
      continue;
    }
    T.ForwardBranchIslands = std::move(*Islands);
    T.ForwardBranchTargetOffset = P.TargetOffset;
    if (T.LongBranchPreservesVcc)
      std::memcpy(T.Bytes.data(), Ctx.LS.SNopBytes.data(), MinInstSize);
    ++BranchIslandChains;
  }
  Pending = std::move(StillPending);

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
            << " initial candidate slot(s))\n";
    return false;
  }
  if (AssignedGateways != 0)
    log() << "hotswap: assigned " << AssignedGateways
          << " SCC-neutral forward gateway(s)\n";
  if (BranchIslandChains != 0)
    log() << "hotswap: assigned " << BranchIslandChains
          << " forward s_branch island chain(s)\n";
  if (ReturnBranchIslandChains != 0)
    log() << "hotswap: assigned " << ReturnBranchIslandChains
          << " return s_branch island chain(s)\n";

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
      DenseMap<uint64_t, size_t>::const_iterator Owner =
          PoolIslandOwners.find(From);
      DenseMap<uint64_t, size_t>::const_iterator SourceOwner =
          SourceTailIslandOwners.find(From);
      if (SourceOwner != SourceTailIslandOwners.end()) {
        Trampoline &OwnerT = Ctx.OutTrampolines[SourceOwner->second];
        OwnerT.HasSourceTailBranchIsland = true;
        OwnerT.SourceTailBranchIslandOffset = From;
        OwnerT.SourceTailBranchTargetOffset = To;
      } else if (Owner != PoolIslandOwners.end()) {
        Trampoline &OwnerT = Ctx.OutTrampolines[Owner->second];
        std::memcpy(OwnerT.Bytes.data() + OwnerT.Bytes.size() -
                        PoolBranchIslandBytes,
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
      DenseMap<uint64_t, size_t>::const_iterator Owner =
          PoolIslandOwners.find(From);
      DenseMap<uint64_t, size_t>::const_iterator SourceOwner =
          SourceTailIslandOwners.find(From);
      if (SourceOwner != SourceTailIslandOwners.end()) {
        Trampoline &OwnerT = Ctx.OutTrampolines[SourceOwner->second];
        OwnerT.HasSourceTailBranchIsland = true;
        OwnerT.SourceTailBranchIslandOffset = From;
        OwnerT.SourceTailBranchTargetOffset = To;
      } else if (Owner != PoolIslandOwners.end()) {
        Trampoline &OwnerT = Ctx.OutTrampolines[Owner->second];
        std::memcpy(OwnerT.Bytes.data() + OwnerT.Bytes.size() -
                        PoolBranchIslandBytes,
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
      ArrayRef<uint8_t>(Text, TextSize));
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
    appendPoolBranchIslands(OutTrampolines);
    if (!assignLongBranchGateways(Ctx, ControlFlow->Targets,
                                  !ControlFlow->HasUnresolvedTargets))
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
      if (Save.size() != VccSaveRestoreBytes || !SetPcOffset || !Landing)
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
        BrFwd = assembleSingleInst("s_get_pc_i64 " + Pair, LS);
        if (BrFwd.size() != MinInstSize)
          return false;
        uint64_t BranchTarget =
            T.SharedDispatcherRelayOffset    ? T.SharedDispatcherRelayOffset
            : T.ForwardBranchIslands.empty() ? T.SharedDispatcherGatewayOffset
                                             : T.ForwardBranchIslands.front();
        SmallVector<uint8_t> Branch =
            LS.encodeSBranch(T.OriginalOffset + BrFwd.size(), BranchTarget);
        if (Branch.size() != MinInstSize)
          return false;
        BrFwd.append(Branch);
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
      SmallVector<uint8_t> Restore = assembleSingleInst(
          "s_mov_b32 vcc_lo, s" + std::to_string(T.LongBranchSgprBase), LS);
      if (Restore.size() != VccSaveRestoreBytes) {
        log() << "hotswap: error: failed to encode VCC restore landing at 0x"
              << utohexstr(T.OriginalOffset + LandingDisplacement) << "\n";
        return false;
      }
      std::memcpy(Text + T.OriginalOffset + LandingDisplacement, Restore.data(),
                  Restore.size());
      PadStart = LandingDisplacement + VccLandingPadBytes;
    }
    // Pad the tail of the replaced slot with cached s_nop bytes.
    for (uint32_t I = PadStart; I + MinInstSize <= T.OriginalSize;
         I += MinInstSize)
      std::memcpy(Text + T.OriginalOffset + I, LS.SNopBytes.data(),
                  MinInstSize);
    if (T.HasSourceTailBranchIsland) {
      if (T.SourceTailBranchIslandOffset < T.OriginalOffset ||
          T.SourceTailBranchIslandOffset - T.OriginalOffset < PadStart ||
          T.SourceTailBranchIslandOffset - T.OriginalOffset >
              T.OriginalSize - MinInstSize) {
        log() << "hotswap: error: source-tail branch island overlaps the "
                 "forward sequence at 0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
      SmallVector<uint8_t> Relay = LS.encodeSBranch(
          T.SourceTailBranchIslandOffset, T.SourceTailBranchTargetOffset);
      if (Relay.size() != MinInstSize) {
        log() << "hotswap: error: source-tail branch island encoding failed "
                 "at 0x"
              << utohexstr(T.SourceTailBranchIslandOffset) << "\n";
        return false;
      }
      std::memcpy(Text + T.SourceTailBranchIslandOffset, Relay.data(),
                  Relay.size());
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
  if (Prof)
    Profile.add(HotswapMetric::ElfParse, profNowNs() - ParseT0, 0);
  ElfView &Elf = *ViewOrErr;
  if (ViewOrErr->textSize() == 0) {
    if (!Elf.isValidDataOnlyObject()) {
      log() << "hotswap: error: retargetCodeObject: empty .text does not "
               "describe a valid data-only code object.\n";
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    uint64_t OutCopyT0 = Prof ? profNowNs() : 0;
    std::unique_ptr<WritableMemoryBuffer> Result =
        copyOutputBuffer(ElfData, ElfSize, "data-only");
    if (Prof)
      Profile.add(HotswapMetric::OutputCopy, profNowNs() - OutCopyT0, 0);
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    Out = std::move(Result);
    log() << "hotswap: accepted data-only code object with empty .text; "
             "returning a byte-identical copy.\n";
    return AMD_COMGR_STATUS_SUCCESS;
  }

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
