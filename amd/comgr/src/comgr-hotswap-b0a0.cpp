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

#include "comgr-hotswap-internal.h"

#include "comgr-env.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

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
// decode/encode helpers (getKernelVgprCount / updateKernelDescriptor) to
// interpret COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT.
// GFX12 wave32: 106 user-addressable SGPRs (s0-s105); s106-s107 are VCC.
static constexpr unsigned Gfx1250MaxSgprs = 106;
static constexpr unsigned Gfx1250VgprGranuleSize = 16;

/// Build the default RewriteConfig used for the GFX1250 B0-to-A0 rewrite:
/// fills in the identity source / target ISA (both gfx1250) and the
/// AMDGPU register granularity constants consumed by
/// ElfView::updateKernelDescriptor. Instruction-encoding state is not
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
  LivenessInfo Info;
  BitVector AllLive(MaxVgprs);
  AllLive.set(0, MaxVgprs);
  Info.LiveBefore.resize(Decoded.size(), AllLive);
  Info.LiveAfter.resize(Decoded.size(), AllLive);
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

static std::vector<ElfView::FunctionTextRange>
buildMergedFunctionTextRanges(const ElfView &Elf) {
  std::vector<ElfView::FunctionTextRange> Ranges = Elf.functionTextRanges();
  llvm::sort(Ranges, [](const ElfView::FunctionTextRange &L,
                        const ElfView::FunctionTextRange &R) {
    if (L.Begin != R.Begin)
      return L.Begin < R.Begin;
    return L.End < R.End;
  });

  std::vector<ElfView::FunctionTextRange> MergedRanges;
  for (const ElfView::FunctionTextRange &Range : Ranges) {
    if (Range.Begin >= Range.End)
      continue;
    if (MergedRanges.empty() || Range.Begin > MergedRanges.back().End) {
      MergedRanges.push_back(Range);
      continue;
    }
    MergedRanges.back().End = std::max(MergedRanges.back().End, Range.End);
  }
  return MergedRanges;
}

static bool isInFunctionTextRange(ArrayRef<ElfView::FunctionTextRange> Ranges,
                                  uint64_t TextAddress) {
  ArrayRef<ElfView::FunctionTextRange>::iterator It =
      std::upper_bound(Ranges.begin(), Ranges.end(), TextAddress,
                       [](uint64_t Value, const ElfView::FunctionTextRange &R) {
                         return Value < R.Begin;
                       });
  if (It == Ranges.begin())
    return false;
  --It;
  return TextAddress < It->End;
}

static bool
isZeroFillDword(const uint8_t *Text, uint64_t TextSize,
                const InternalDecodedInst &DI, uint64_t TextAddr,
                ArrayRef<ElfView::FunctionTextRange> FunctionRanges) {
  if (FunctionRanges.empty() || DI.Size != MinInstSize ||
      DI.Offset > TextSize || MinInstSize > TextSize - DI.Offset ||
      DI.Offset > std::numeric_limits<uint64_t>::max() - TextAddr ||
      isInFunctionTextRange(FunctionRanges, TextAddr + DI.Offset))
    return false;
  return std::all_of(Text + DI.Offset, Text + DI.Offset + MinInstSize,
                     [](uint8_t B) { return B == 0; });
}

/// Scan \p Decoded for runs of consecutive `s_nop` instructions or undecoded
/// zero-filled alignment padding at least MinNopSledSize bytes long and return
/// the resulting NopSled list (each sled records Start / End byte offsets in
/// .text and the initial WritePos at Start). These sleds are the landing zones
/// emitToNopSled targets for in-place rewrites. NOPs are identified by MC
/// opcode (cached on \p LS at initLLVM() time) rather than mnemonic string, so
/// the scanner is robust against printer aliasing / mnemonic formatting
/// variations. Zero-filled padding is accepted only when function-symbol ranges
/// prove it is between functions; zero bytes inside a function are executable
/// code and are not sled space.
static std::vector<NopSled>
buildNopSledMap(ArrayRef<InternalDecodedInst> Decoded, const uint8_t *Text,
                uint64_t TextSize, const LLVMState &LS, const ElfView &Elf) {
  std::vector<NopSled> Sleds;
  std::vector<ElfView::FunctionTextRange> FunctionRanges =
      buildMergedFunctionTextRanges(Elf);
  const size_t N = Decoded.size();
  size_t I = 0;
  while (I < N) {
    bool IsSledInst = Decoded[I].Inst.getOpcode() == LS.SNopOpcode;
    bool IsZeroFill = isZeroFillDword(Text, TextSize, Decoded[I],
                                      Elf.textAddr(), FunctionRanges);
    if (IsSledInst || IsZeroFill) {
      uint64_t Start = Decoded[I].Offset;
      uint64_t End = Start;
      while (I < N && (Decoded[I].Inst.getOpcode() == LS.SNopOpcode ||
                       isZeroFillDword(Text, TextSize, Decoded[I],
                                       Elf.textAddr(), FunctionRanges))) {
        End = Decoded[I].Offset + Decoded[I].Size;
        ++I;
      }
      if (End - Start >= MinNopSledSize)
        Sleds.push_back({Start, End, Start});
    } else {
      ++I;
    }
  }
  return Sleds;
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
  return true;
}

// s_add_pc_i64 long branch from \p FromOffset to \p TargetOffset (.text
// offsets). It adds a signed literal to the next instruction's PC, so the
// offset is TargetOffset - (FromOffset + size); size is 8 (forward, 32-bit
// literal) or 12 (backward, 64-bit literal), resolved by trying each. Encoded
// via the MC assembler. Returns empty on failure.
//
// Precondition: callers only long-branch *far* sites, so the target is far
// enough that s_add_pc_i64 always needs a real (>= 32-bit) literal. A tiny
// offset would instead assemble to the 4-byte inline-constant form, match
// neither candidate size, and yield empty. That is asserted here (assert
// liberally); the empty return is kept as a release-build safety net so a
// stray near target degrades to the unpatched-object fallback rather than
// crashing the loader.
SmallVector<uint8_t> encodeLongBranch(const LLVMState &LS, uint64_t FromOffset,
                                      uint64_t TargetOffset) {
  [[maybe_unused]] const int64_t Distance =
      static_cast<int64_t>(TargetOffset) - static_cast<int64_t>(FromOffset);
  [[maybe_unused]] const int64_t MinLongDistance = LongBranchMaxBytes;
  assert((Distance > MinLongDistance || Distance < -MinLongDistance) &&
         "encodeLongBranch: target is not a far site; the long-branch path is "
         "only valid for offsets beyond an s_add_pc_i64 instruction's reach");

  for (uint32_t Size : {LongBranchFwdBytes, LongBranchMaxBytes}) {
    int64_t Off = static_cast<int64_t>(TargetOffset) -
                  static_cast<int64_t>(FromOffset + Size);
    SmallVector<uint8_t> Bytes =
        assembleSingleInst("s_add_pc_i64 " + std::to_string(Off), LS);
    if (Bytes.size() == Size)
      return Bytes;
  }
  return {};
}

// Backward long branch via the legacy set-pc expansion. Used on gfx1250 A0
// where the backward (64-bit-literal) s_add_pc_i64 form triggers HSV-009. This
// mirrors the pre-s_add_pc_i64 sequence upstream emits in
// SIInstrInfo::insertIndirectBranch and the entry-stub materialization in
// buildKernelEntryTrampoline:
//
//   s_cselect_b32 sT, 1, 0        ; sT = caller SCC (SCC unchanged)
//   s_get_pc_i64  s[N:N+1]        ; pair = PC of the following s_add_u32
//   s_add_u32     sN,  sN,  lo32  ; pair += (target - post_getpc); clobbers SCC
//   s_addc_u32    sN1, sN1, hi32  ; carry from the add
//   s_cmp_lg_u32  sT, 0           ; SCC = (sT != 0), restoring caller SCC
//   s_set_pc_i64  s[N:N+1]        ; jump to the materialized target
//
// s_get_pc_i64 captures the address of the *next* instruction, so the delta is
// measured from the s_add_u32 slot (FromOffset + the 8-byte cselect+getpc
// prefix). The add/addc pair is the only SCC writer, so it is bracketed by the
// cselect (save) / cmp (restore) pair because a far trampoline sits in the
// middle of straight-line code where the caller's SCC may be live across the
// relocated site. Assembled via the MC layer; empty on failure.
SmallVector<uint8_t> encodeSetPCLongBranch(const LLVMState &LS,
                                           uint64_t FromOffset,
                                           uint64_t TargetOffset,
                                           unsigned SgprBase) {
  const std::string Lo = "s" + std::to_string(SgprBase);
  const std::string Hi = "s" + std::to_string(SgprBase + 1);
  const std::string Tmp = "s" + std::to_string(SgprBase + 2);
  const std::string Pair = "s[" + std::to_string(SgprBase) + ":" +
                           std::to_string(SgprBase + 1) + "]";

  // Fixed-size prefix ahead of the s_add_u32: s_cselect_b32 (4) +
  // s_get_pc_i64 (4). s_get_pc_i64 returns the PC of the s_add_u32, so the
  // PC-relative delta is measured from there.
  constexpr uint32_t PrefixBytes = 8;
  const int64_t Delta = static_cast<int64_t>(TargetOffset) -
                        static_cast<int64_t>(FromOffset + PrefixBytes);
  const uint32_t OffLo = static_cast<uint32_t>(static_cast<uint64_t>(Delta));
  const uint32_t OffHi =
      static_cast<uint32_t>(static_cast<uint64_t>(Delta) >> 32);

  std::string Asm = joinAsmLines({
      "s_cselect_b32 " + Tmp + ", 1, 0",
      "s_get_pc_i64 " + Pair,
      "s_add_u32 " + Lo + ", " + Lo + ", 0x" + utohexstr(OffLo),
      "s_addc_u32 " + Hi + ", " + Hi + ", 0x" + utohexstr(OffHi),
      "s_cmp_lg_u32 " + Tmp + ", 0",
      "s_set_pc_i64 " + Pair,
  });
  SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, LS);
  if (Bytes.empty() || Bytes.size() > LongBranchBackSeqBytes) {
    log() << "hotswap: error: set-pc long branch encoding failed (size "
          << Bytes.size() << ", max " << LongBranchBackSeqBytes << ")\n";
    return {};
  }
  return Bytes;
}

// Read once: HOTSWAP_BACK_ADDPC=1 forces the legacy backward s_add_pc_i64 path
// (pre-fix behavior) instead of the SCC-preserving set-pc expansion, so the two
// backward-edge encodings can be A/B compared on hardware without a rebuild.
static bool useLegacyBackAddPC() {
  static const bool V = [] {
    const char *E = getenv("HOTSWAP_BACK_ADDPC");
    return E && strtol(E, nullptr, 0) != 0;
  }();
  return V;
}

// Reserve an even-aligned 3-SGPR scratch block above the owning kernel's
// declared .sgpr_count for a far trampoline's backward set-pc long branch:
// s[Base:Base+1] (PC pair) + s[Base+2] (SCC save). SGPRs above .sgpr_count are
// never used by the kernel, and GFX10+ waves always carry the full SGPR file,
// so no liveness analysis or kernel-descriptor bump is required (the SGPR
// granule field is architecturally reserved on GFX10+). Records the reservation
// in per-kernel stats. Returns the aligned base, or nullopt if the kernel is
// too SGPR-saturated to fit the block below MaxSgprs.
static std::optional<unsigned>
tryReserveLongBranchSgprs(PatchContext &Ctx, uint64_t InstOffset) {
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(InstOffset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdSgprs = Ctx.Elf.getKernelSgprCount(KernelName);
  // Unknown count -> assume saturated so we decline rather than risk clobbering
  // a live SGPR.
  unsigned SgprCount = KdSgprs.value_or(Ctx.Config.MaxSgprs);

  const unsigned Base = (SgprCount + 1) & ~1u;
  constexpr unsigned NeededRegs = 3; // even-aligned PC pair + SCC temp
  if (Base > Ctx.Config.MaxSgprs || Ctx.Config.MaxSgprs - Base < NeededRegs)
    return std::nullopt;

  if (!KernelName.empty()) {
    KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
    Stats.ExtraSgprs = std::max(Stats.ExtraSgprs, (Base + NeededRegs) - SgprCount);
  }
  return Base;
}

/// Queue a deferred trampoline for [\p InstOffset, +\p InstSize) with
/// \p Replacement as its body; fixupTrampolineBranches fills in the edges once
/// the pool layout is known. A site beyond s_branch reach of the appended pool
/// uses a PC-relative long branch on both edges: an 8-byte forward
/// s_add_pc_i64 (32-bit literal) that overwrites the site in place -- so a
/// site smaller than that declines rather than clobbering the next instruction
/// -- and, on the backward edge, the SCC-preserving set-pc expansion (see
/// encodeSetPCLongBranch), which needs an even-aligned scratch SGPR block. A
/// far site whose owning kernel has no room for that block also declines.
[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    ArrayRef<uint8_t> Replacement) {
  // This trampoline lands right after .text and after every trampoline already
  // queued -- later ones are appended behind it and cannot shift it, and
  // fixupTrampolineBranches walks the same list in the same order -- so its
  // final pool offset is known exactly now.
  uint64_t PoolStart = Ctx.TextSize;
  for (const Trampoline &Prev : Ctx.OutTrampolines)
    PoolStart += Prev.Bytes.size();

  // An s_branch encodes To - From as a signed simm16 dword field, in range iff
  // (To - From - MinInstSize) / MinInstSize fits [BranchOffsetMin,
  // BranchOffsetMax] (see LLVMState::encodeSBranch). Test both edges with the
  // short branch-back slot; the branch-back (pool tail -> site) is the farther
  // of the two. Go long only when a short branch cannot reach.
  auto WithinSBranch = [](uint64_t From, uint64_t To) {
    int64_t Dword = (static_cast<int64_t>(To) - static_cast<int64_t>(From) -
                     static_cast<int64_t>(MinInstSize)) /
                    static_cast<int64_t>(MinInstSize);
    return Dword >= BranchOffsetMin && Dword <= BranchOffsetMax;
  };
  const uint64_t ShortBackFrom = PoolStart + Replacement.size();
  const bool Far = !(WithinSBranch(InstOffset, PoolStart) &&
                     WithinSBranch(ShortBackFrom, InstOffset + InstSize));

  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
  T.Bytes.insert(T.Bytes.end(), Replacement.begin(), Replacement.end());

  if (Far) {
    // HSV-009 bisection instrument. HOTSWAP_FAR_KEEP controls how many far
    // sites (in object scan order) still get a real long-branch trampoline;
    // the rest are DECLINED (original instruction left in place).
    //   unset / <0  -> keep all far sites (pre-fix behavior, reproduces fault)
    //   0           -> decline all far sites (the HSV-009 "fix")
    //   N > 0       -> keep the first N far sites, decline the rest
    // Lets us sweep N across GPUs to find whether a count threshold or a
    // specific site triggers the crash, without editing/rebuilding per point.
    static const long FarKeep = [] {
      const char *E = getenv("HOTSWAP_FAR_KEEP");
      return E ? strtol(E, nullptr, 0) : -1;
    }();
    static std::atomic<long> FarSeen{0};
    const long Idx = FarSeen.fetch_add(1);
    const bool Decline = (FarKeep >= 0 && Idx >= FarKeep);
    if (Decline) {
      log() << "hotswap: far trampoline #" << Idx << " at 0x"
            << utohexstr(InstOffset) << " declined (HOTSWAP_FAR_KEEP=" << FarKeep
            << ", HSV-009); site left unpatched\n";
      return false;
      
    }
    if (InstSize < LongBranchFwdBytes) {
      log() << "hotswap: long trampoline: site 0x" << utohexstr(InstOffset)
            << " is " << InstSize << " B < " << LongBranchFwdBytes
            << " B forward branch; declining (site left unpatched)\n";
      return false;
    }
    T.Long = true;
    if (useLegacyBackAddPC()) {
      // A/B path: legacy backward s_add_pc_i64 (reproduces the HSV-009 fault).
      T.Bytes.insert(T.Bytes.end(), LongBranchMaxBytes, uint8_t{0});
    } else {
      // Default fix: the backward edge uses the SCC-preserving set-pc
      // expansion, which needs an even-aligned scratch SGPR block above the
      // owning kernel's .sgpr_count. Decline the site if none fits.
      std::optional<unsigned> SgprBase =
          tryReserveLongBranchSgprs(Ctx, InstOffset);
      if (!SgprBase) {
        log() << "hotswap: far trampoline at 0x" << utohexstr(InstOffset)
              << ": no aligned scratch SGPR block for backward set-pc branch; "
              << "declining (site left unpatched)\n";
        return false;
      }
      T.UsesSetPCBack = true;
      T.LongBranchSgprBase = *SgprBase;
      T.Bytes.insert(T.Bytes.end(), LongBranchBackSeqBytes, uint8_t{0});
    }
    log() << "hotswap: far trampoline #" << Idx << " at 0x"
          << utohexstr(InstOffset) << " kept (HOTSWAP_FAR_KEEP=" << FarKeep
          << (T.UsesSetPCBack ? ", set-pc back)\n" : ", addpc back)\n");
  } else {
    // Reserve the short branch-back slot; fixupTrampolineBranches fills it in.
    T.Bytes.insert(T.Bytes.end(), MinInstSize, uint8_t{0});
  }
  Ctx.OutTrampolines.emplace_back(std::move(T));
  return true;
}

/// Emit \p Replacement for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize). Prefers an in-place NOP-sled rewrite when a
/// reachable sled with sufficient headroom exists; otherwise falls back to a
/// deferred trampoline.
[[nodiscard]] bool emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset,
                                       uint32_t InstSize,
                                       ArrayRef<uint8_t> Replacement) {
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
  return emitToTrampoline(Ctx, InstOffset, InstSize, Replacement);
}

// -- Forward-edge set-pc upgrade (instruction stealing) -----------------------

// Read once: HOTSWAP_FWD_ADDPC=1 keeps the legacy forward s_add_pc_i64 edge
// (pre-fix behavior) instead of upgrading it to the SCC-preserving set-pc
// expansion, so the forward-edge fix can be A/B compared without a rebuild.
static bool useLegacyFwdAddPC() {
  static const bool V = [] {
    const char *E = getenv("HOTSWAP_FWD_ADDPC");
    return E && strtol(E, nullptr, 0) != 0;
  }();
  return V;
}

// Mnemonics of B0-incompatible instructions handled by the trampoline patch
// family. If one still carries its original mnemonic in the decoded stream it
// was NOT patched (its trampoline was declined), so its .text bytes are the
// broken-on-A0 B0 encoding. Such an instruction must never be stolen verbatim
// into a trampoline body -- that would reintroduce the hazard the pool exists
// to remove. Instructions rewritten in place (e.g. s_clause -> s_nop) keep
// their original mnemonic too, but their .text bytes are already the A0-safe
// encoding, so this check is intentionally limited to the trampoline-family
// hazards (a strict superset check is harmless: it only declines a steal).
static bool looksLikeUnpatchedB0Hazard(StringRef Mnemonic) {
  if (Mnemonic == "tensor_load_to_lds")
    return true;
  if (Mnemonic.starts_with("ds_") &&
      (Mnemonic.contains("_2addr") || Mnemonic.contains("_addtid")))
    return true;
  return false;
}

/// Post-process the deferred far trampolines to replace their forward
/// s_add_pc_i64 edge (site -> pool) with the SCC-preserving set-pc expansion,
/// eliminating the HSV-009-triggering instruction on the forward edge as well
/// as the backward. The forward set-pc sequence (LongBranchFwdSeqBytes) is
/// larger than the original instruction it overwrites, so the site "steals"
/// the bytes of the instructions that follow it:
///
///   - A following instruction that is itself a far trampoline site is MERGED:
///     its already-built (A0-fixed) replacement body is relocated into this
///     trampoline and the separate trampoline is dropped, so consecutive
///     patch sites collapse into one trampoline with one forward + one
///     backward set-pc.
///   - A following normal instruction is relocated verbatim from the
///     (post-patch) .text bytes, provided it is safe to move.
///
/// The steal window grows one whole instruction at a time until it can hold
/// the forward set-pc sequence. A trampoline whose window cannot be filled
/// safely is left on the legacy forward s_add_pc_i64 path (correct, but still
/// hits the erratum for that one site) -- far below the site-count threshold
/// that triggers the erratum, so this is safe. Reuses each trampoline's
/// existing backward-edge SGPR block; no additional SGPRs are reserved.
///
/// Only instructions AFTER the site are stolen (forward). A backward-steal
/// fallback (stealing preceding instructions) exists behind
/// HOTSWAP_FWD_STEAL_MODE>=2 but is off by default: it tended to relocate an
/// s_delay_alu hazard hint out of its scheduling context and fault (see the
/// StealMode comment below).
///
/// Runs after every patch pass, so OutTrampolines holds every far site's
/// finished (A0-fixed) body and the decoded stream is complete. Only mutates
/// the trampoline list; the site .text bytes and pool layout are materialized
/// later by fixupTrampolineBranches.
static void expandForwardSetPc(PatchContext &Ctx) {
  // A/B gates: the legacy backward-addpc path keeps the whole pre-fix long
  // branch shape for comparison, and HOTSWAP_FWD_ADDPC keeps just the forward
  // edge legacy. In both cases leave the forward edge as s_add_pc_i64.
  if (useLegacyBackAddPC() || useLegacyFwdAddPC())
    return;

  // Forward-steal aggressiveness. Backward stealing (mode >= 2) is DISABLED by
  // default: it relocates the instructions PRECEDING a site into the trampoline,
  // and those commonly include an address computation guarded by an s_delay_alu
  // hazard hint whose meaning is tied to the preceding instruction stream. Moved
  // into the pool the hint applies the wrong stall and a dependent VALU reads a
  // stale operand -> garbage store/load address -> GPU page fault (root-caused
  // on rocsparse spgemm/csrgemm). Forward stealing does not hit this because the
  // s_delay_alu sits ahead of the site, not among the following instructions it
  // relocates. isUnsafeToRelocate() also refuses to move any s_delay_alu, so
  // even the gated-on backward path is hazard-safe; the gate stays for A/B.
  //   HOTSWAP_FWD_STEAL_MODE=0  merge adjacent patch sites only (never relocate
  //                             a non-site instruction, no backward steal)
  //   HOTSWAP_FWD_STEAL_MODE=1  merge + forward-steal normal instructions, no
  //                             backward steal (default)
  //   HOTSWAP_FWD_STEAL_MODE>=2 also enable the backward-steal fallback
  static const long StealMode = [] {
    const char *E = getenv("HOTSWAP_FWD_STEAL_MODE");
    return E ? strtol(E, nullptr, 0) : 1;
  }();

  std::vector<Trampoline> &Tramps = Ctx.OutTrampolines;
  const LLVMState &LS = Ctx.LS;
  if (Tramps.empty() || !LS.MCII || !LS.MRI)
    return;

  // Instruction-boundary index: a steal may only consume whole instructions,
  // so every candidate offset must start a decoded instruction.
  DenseMap<uint64_t, size_t> OffToInst;
  for (size_t I = 0, E = Ctx.Decoded.size(); I < E; ++I)
    OffToInst[Ctx.Decoded[I].Offset] = I;

  // Direct branch/call target offsets. Decode used .text-relative offsets as
  // the instruction address (see decodeTextSection), so evaluateBranch yields
  // targets in the same .text-offset space as OriginalOffset / the steal
  // cursor. An interior offset of a steal window must not be a branch target,
  // or a jump would land in the middle of the forward set-pc / nop padding.
  DenseSet<uint64_t> BranchTargets;
  if (LS.MIA) {
    for (const InternalDecodedInst &DI : Ctx.Decoded) {
      const MCInst &Inst = DI.Inst;
      if (LS.MIA->isCall(Inst) || LS.MIA->isUnconditionalBranch(Inst) ||
          LS.MIA->isConditionalBranch(Inst)) {
        uint64_t Target = 0;
        if (LS.MIA->evaluateBranch(Inst, DI.Offset, DI.Size, Target))
          BranchTargets.insert(Target);
      }
    }
  }

  // Site offset -> trampoline index, so a steal cursor landing on a patch site
  // can merge that site's trampoline instead of stealing broken B0 bytes.
  DenseMap<uint64_t, size_t> OffToTramp;
  for (size_t I = 0, E = Tramps.size(); I < E; ++I)
    OffToTramp[Tramps[I].OriginalOffset] = I;

  // Reserved back-edge slot width for a trampoline in its current state.
  auto backReserve = [](const Trampoline &T) -> uint32_t {
    if (T.UsesSetPCBack)
      return LongBranchBackSeqBytes;
    if (T.Long)
      return LongBranchMaxBytes;
    return MinInstSize;
  };

  // An instruction that cannot be relocated verbatim into a trampoline body.
  // Two classes:
  //   1. PC-reading/writing or control-flow instructions -- their encoded
  //      target/PC is position-dependent, so moving them changes where they go.
  //   2. Context-dependent hazard hints -- s_delay_alu encodes a VALU
  //      dependency/latency relative to the PRECEDING instruction stream. Once
  //      relocated into a trampoline the preceding instructions differ, so the
  //      hint silently applies the wrong stall and a later VALU can read a stale
  //      operand (observed as a GPU page fault when a stolen address-compute's
  //      s_delay_alu moved with it). Never relocate these.
  auto isUnsafeToRelocate = [&](const InternalDecodedInst &DI) -> bool {
    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    if (Desc.mayAffectControlFlow(DI.Inst, *LS.MRI))
      return true;
    StringRef M(DI.Mnemonic);
    return M.contains("get_pc") || M.contains("getpc") || M.contains("set_pc") ||
           M.contains("setpc") || M.contains("add_pc") || M.contains("addpc") ||
           M.starts_with("s_delay");
  };

  std::vector<bool> Removed(Tramps.size(), false);
  unsigned Upgraded = 0, Merged = 0, Kept = 0;
  // Decline-reason tally (only for sites that fell short of the required
  // footprint), to explain residual forward s_add_pc_i64 edges in the log.
  unsigned RsnBoundary = 0, RsnTarget = 0, RsnEndUnknown = 0, RsnPcRel = 0,
           RsnHazard = 0, RsnMergedOrder = 0, RsnBounds = 0, RsnBackExhausted = 0,
           RsnStealDisabled = 0;

  // Process sites in ascending .text offset. Forward stealing extends a site's
  // window into following instructions; backward stealing (the fallback below)
  // extends it into preceding ones. HighWater is the highest .text offset any
  // already-finalized trampoline's window occupies (its forward set-pc region
  // or its original slot). Backward stealing must not cross it -- those bytes
  // are owned by an earlier trampoline, and an earlier trampoline's (indirect,
  // set-pc) backward edge returns exactly to HighWater, so keeping the new
  // window at or above HighWater keeps that return landing on this site's
  // forward edge rather than inside its interior.
  std::vector<size_t> Order(Tramps.size());
  for (size_t I = 0, E = Tramps.size(); I < E; ++I)
    Order[I] = I;
  llvm::sort(Order, [&](size_t A, size_t B) {
    return Tramps[A].OriginalOffset < Tramps[B].OriginalOffset;
  });
  uint64_t HighWater = 0;

  for (size_t OI = 0, OE = Order.size(); OI < OE; ++OI) {
    const size_t I = Order[OI];
    if (Removed[I])
      continue;
    Trampoline &T = Tramps[I];
    // Only far trampolines carry the erratum instruction; near ones use
    // s_branch (no erratum) and are left untouched. The backward edge must
    // already be the set-pc form (its SGPR block is what the forward edge
    // reuses); the legacy backward-addpc path was excluded above.
    if (!T.Long || !T.UsesSetPCBack) {
      HighWater = std::max(HighWater, T.OriginalOffset + T.OriginalSize);
      continue;
    }

    // Strip T's own reserved back slot to recover its A0-fixed replacement.
    const uint32_t TBack = backReserve(T);
    if (T.Bytes.size() < TBack) {
      HighWater = std::max(HighWater, T.OriginalOffset + T.OriginalSize);
      continue;
    }
    SmallVector<uint8_t> Body(T.Bytes.begin(), T.Bytes.end() - TBack);

    uint64_t Footprint = T.OriginalSize;
    uint64_t StealAt = T.OriginalOffset + T.OriginalSize;
    uint64_t WindowStart = T.OriginalOffset;
    SmallVector<uint8_t> Stolen;   // bytes stolen AFTER the site
    SmallVector<uint8_t> Prefix;   // bytes stolen BEFORE the site
    SmallVector<size_t, 4> MergedIdx;
    unsigned *DeclineRsn = nullptr;

    // Forward steal: consume following instructions until the site can hold the
    // forward set-pc sequence. Keep going past that point while the next
    // instruction is itself a patch site, so a run of consecutive sites
    // collapses into one trampoline instead of stranding its tail.
    while (true) {
      DenseMap<uint64_t, size_t>::iterator InstIt = OffToInst.find(StealAt);
      if (InstIt == OffToInst.end()) {
        DeclineRsn = &RsnBoundary; // cursor not on an instruction boundary
        break;
      }
      DenseMap<uint64_t, size_t>::iterator TrampIt = OffToTramp.find(StealAt);
      const bool NextIsSite = TrampIt != OffToTramp.end();
      const bool HaveRoom = Footprint >= LongBranchFwdSeqBytes;
      // Enough room and the next instruction is not a strandable patch site:
      // stop (a successful upgrade).
      if (HaveRoom && !NextIsSite) {
        DeclineRsn = nullptr;
        break;
      }
      if (BranchTargets.count(StealAt)) {
        DeclineRsn = &RsnTarget; // would relocate the target of a branch
        break;
      }
      const InternalDecodedInst &DN = Ctx.Decoded[InstIt->second];
      if (DN.Mnemonic == "<unknown>" || DN.Mnemonic == "s_endpgm") {
        DeclineRsn = &RsnEndUnknown; // padding or a wave terminator: stop
        break;
      }

      if (NextIsSite) {
        // A following patch site: merge its fixed replacement (drop its edges).
        const size_t J = TrampIt->second;
        if (Removed[J] || Tramps[J].OriginalOffset <= T.OriginalOffset) {
          DeclineRsn = &RsnMergedOrder;
          break;
        }
        const Trampoline &T2 = Tramps[J];
        const uint32_t T2Back = backReserve(T2);
        if (T2.Bytes.size() < T2Back) {
          DeclineRsn = &RsnMergedOrder;
          break;
        }
        Stolen.append(T2.Bytes.begin(), T2.Bytes.end() - T2Back);
        Footprint += T2.OriginalSize;
        StealAt += T2.OriginalSize;
        MergedIdx.push_back(J);
      } else {
        // A normal instruction: relocate its (post-patch) .text bytes verbatim.
        if (StealMode < 1) {
          DeclineRsn = &RsnStealDisabled;
          break;
        }
        if (isUnsafeToRelocate(DN)) {
          DeclineRsn = &RsnPcRel;
          break;
        }
        if (looksLikeUnpatchedB0Hazard(DN.Mnemonic)) {
          DeclineRsn = &RsnHazard;
          break;
        }
        if (DN.Offset + DN.Size > Ctx.TextSize) {
          DeclineRsn = &RsnBounds;
          break;
        }
        log() << "hotswap: fwd-steal-normal: off=0x" << utohexstr(DN.Offset)
              << " mn=" << DN.Mnemonic << "\n";
        const uint8_t *P = Ctx.Text + DN.Offset;
        Stolen.append(P, P + DN.Size);
        Footprint += DN.Size;
        StealAt += DN.Size;
      }
    }

    // Backward steal fallback: when forward stealing was blocked short (most
    // commonly by a branch a few instructions after the site), extend the
    // window into the PRECEDING instructions instead. The forward set-pc then
    // starts at WindowStart (< the original site) and those preceding
    // instructions run -- in original order -- ahead of the site's replacement
    // in the trampoline body. Only normal, relocatable instructions above
    // HighWater are eligible, and the site itself must not be a branch target
    // (it becomes an interior offset once the window starts before it).
    if (Footprint < LongBranchFwdSeqBytes && StealMode >= 2) {
      DenseMap<uint64_t, size_t>::iterator SiteIt =
          OffToInst.find(T.OriginalOffset);
      size_t K =
          (SiteIt != OffToInst.end()) ? SiteIt->second : Ctx.Decoded.size();
      const std::string SiteKernel =
          Ctx.Elf.findKernelAtAddress(T.OriginalOffset + Ctx.Elf.textAddr());
      while (Footprint < LongBranchFwdSeqBytes && K > 0) {
        // WindowStart becomes an interior offset once we prepend before it, so
        // nothing may branch to it (the very first check covers the site).
        if (BranchTargets.count(WindowStart)) {
          DeclineRsn = &RsnBackExhausted;
          break;
        }
        const InternalDecodedInst &DP = Ctx.Decoded[K - 1];
        const uint64_t POff = DP.Offset;
        if (POff < HighWater || POff + DP.Size != WindowStart) {
          DeclineRsn = &RsnBackExhausted; // owned by an earlier tramp / gap
          break;
        }
        if (OffToTramp.count(POff)) {
          DeclineRsn = &RsnBackExhausted; // don't overlap a preceding site
          break;
        }
        if (DP.Mnemonic == "<unknown>" || DP.Mnemonic == "s_endpgm") {
          DeclineRsn = &RsnBackExhausted;
          break;
        }
        // Never relocate across a kernel boundary (the s_endpgm/PC-relative
        // barriers above usually stop first; this is defense-in-depth).
        if (Ctx.Elf.findKernelAtAddress(POff + Ctx.Elf.textAddr()) != SiteKernel) {
          DeclineRsn = &RsnBackExhausted;
          break;
        }
        if (isUnsafeToRelocate(DP)) {
          DeclineRsn = &RsnPcRel;
          break;
        }
        if (looksLikeUnpatchedB0Hazard(DP.Mnemonic)) {
          DeclineRsn = &RsnHazard;
          break;
        }
        if (POff + DP.Size > Ctx.TextSize) {
          DeclineRsn = &RsnBounds;
          break;
        }
        const uint8_t *P = Ctx.Text + POff;
        Prefix.insert(Prefix.begin(), P, P + DP.Size);
        Footprint += DP.Size;
        WindowStart = POff;
        --K;
      }
    }

    if (Footprint < LongBranchFwdSeqBytes) {
      if (DeclineRsn)
        ++*DeclineRsn;
      ++Kept; // leave T on the legacy forward s_add_pc_i64 edge
      HighWater = std::max(HighWater, T.OriginalOffset + T.OriginalSize);
      continue;
    }

    // Commit: rebuild T as [prefix...][replacement][stolen...][back slot] and
    // move the site to WindowStart (where the forward set-pc is written).
    SmallVector<uint8_t> NewBytes;
    NewBytes.reserve(Prefix.size() + Body.size() + Stolen.size() +
                     LongBranchBackSeqBytes);
    NewBytes.append(Prefix.begin(), Prefix.end());
    NewBytes.append(Body.begin(), Body.end());
    NewBytes.append(Stolen.begin(), Stolen.end());
    NewBytes.append(LongBranchBackSeqBytes, uint8_t{0});
    T.Bytes = std::move(NewBytes);
    T.OriginalOffset = WindowStart;
    T.UsesSetPCFwd = true;
    T.StolenBytes = static_cast<uint32_t>(Footprint);
    for (size_t J : MergedIdx)
      Removed[J] = true;
    ++Upgraded;
    Merged += MergedIdx.size();
    HighWater = std::max(HighWater, StealAt);
    if (!Prefix.empty())
      log() << "hotswap: fwd set-pc backward-stolen: window_start=0x"
            << utohexstr(WindowStart) << " site_end=0x" << utohexstr(StealAt)
            << " footprint=" << Footprint << " prefix_bytes=" << Prefix.size()
            << " merged=" << MergedIdx.size() << "\n";
  }

  if (Merged) {
    std::vector<Trampoline> Compact;
    Compact.reserve(Tramps.size() - Merged);
    for (size_t I = 0, E = Tramps.size(); I < E; ++I)
      if (!Removed[I])
        Compact.push_back(std::move(Tramps[I]));
    Tramps.swap(Compact);
  }

  log() << "hotswap: forward set-pc: upgraded " << Upgraded
        << " far trampoline(s) (merged " << Merged
        << " adjacent site(s)); " << Kept
        << " left on legacy forward s_add_pc_i64 (decline reasons:"
        << " branch_target=" << RsnTarget << " end/unknown=" << RsnEndUnknown
        << " pc_relative=" << RsnPcRel << " unpatched_hazard=" << RsnHazard
        << " boundary=" << RsnBoundary << " merge_order=" << RsnMergedOrder
        << " bounds=" << RsnBounds << " back_exhausted=" << RsnBackExhausted
        << " steal_disabled=" << RsnStealDisabled << ")\n";
}

// -- applyGfx1250B0toA0Rules --------------------------------------------------

/// Per-instruction patch-pass trampoline: invokes \p Fn with (\p Ctx,
/// \p Idx) if it is non-null, or returns 0 otherwise. nullptr means
/// the corresponding pass family has no implementation linked in
/// (e.g. scratch today), which the dispatcher treats as a no-op slot.
static uint32_t runPerInstPass(uint32_t (*Fn)(PatchContext &, size_t),
                               PatchContext &Ctx, size_t Idx) {
  return Fn ? Fn(Ctx, Idx) : 0;
}

/// Main per-instruction dispatcher for the GFX1250 B0-to-A0 rewrite.
/// Builds the NOP sled map, CFG, and VGPR liveness for the decoded stream,
/// then walks each decoded instruction and runs the patch passes in order
/// (in-place -> trampoline -> WMMA split -> scratch). Each pass gets a
/// chance to claim the instruction; first non-zero return wins. Also runs
/// the whole-function WMMA-hazard pass after the per-instruction loop and
/// records per-kernel stats via ElfView::updateKernelDescriptor.
/// Returns the total number of applied patches across all passes.
static std::optional<uint32_t>
applyGfx1250B0toA0Rules(std::vector<InternalDecodedInst> &Decoded,
                        uint8_t *Text, uint64_t TextSize, const LLVMState &LS,
                        std::vector<Trampoline> &OutTrampolines, ElfView &Elf,
                        std::vector<ScratchPatchInfo> &OutScratchPatches,
                        const RewriteConfig &Config) {
  uint32_t Patched = 0;
  std::vector<NopSled> Sleds =
      buildNopSledMap(Decoded, Text, TextSize, LS, Elf);

  CFG Cfg = buildCfg(Decoded, *LS.MCII);
  LivenessInfo Liveness =
      computeLiveness(Decoded, Cfg, *LS.MCII, *LS.MRI, Config.MaxVgprs);

  if (!Liveness.Converged) {
    log() << "hotswap: error: liveness analysis did not converge, using "
          << "conservative all-VGPRs-live fallback\n";
    BitVector AllVgprs(Config.MaxVgprs);
    AllVgprs.set(0, Config.MaxVgprs);
    for (size_t I = 0, LE = Liveness.LiveBefore.size(); I < LE; ++I) {
      Liveness.LiveBefore[I] = AllVgprs;
      Liveness.LiveAfter[I] = AllVgprs;
    }
  }

  StringMap<KernelPatchStats> KernelStats;
  PatchContext Ctx{Config,           Decoded, Text, TextSize, LS,
                   OutTrampolines,   Sleds,   Elf,  Liveness, KernelStats,
                   OutScratchPatches};

  const HotswapPatchVTable &VT = getHotswapPatchVTable();

  // Skip undecoded slots produced by the decoder for bytes it could not
  // classify as a valid instruction; the dispatcher has nothing to match
  // against on these and we must not invoke the patch passes for them.
  constexpr StringLiteral UnknownMnemonic = "<unknown>";

  for (size_t Idx = 0, E = Decoded.size(); Idx < E; ++Idx) {
    const InternalDecodedInst &DI = Decoded[Idx];
    if (DI.Mnemonic == UnknownMnemonic)
      continue;

    if (uint32_t P = runPerInstPass(VT.applyInPlacePatches, Ctx, Idx)) {
      Patched += P;
      continue;
    }
    if (uint32_t P = runPerInstPass(VT.applyTrampolinePatches, Ctx, Idx)) {
      Patched += P;
      continue;
    }
    if (uint32_t P = runPerInstPass(VT.applyWmmaSplitPatches, Ctx, Idx)) {
      Patched += P;
      continue;
    }
    if (uint32_t P = runPerInstPass(VT.applyScratchPatches, Ctx, Idx)) {
      Patched += P;
      continue;
    }
    if (uint32_t P = runPerInstPass(VT.applyWmmaScale16Patches, Ctx, Idx)) {
      Patched += P;
      continue;
    }
  }

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
  if (VT.applyWmmaHazardPatch)
    Patched += VT.applyWmmaHazardPatch(Ctx);
  if (VT.applyVop3px2Src2Fix)
    Patched += VT.applyVop3px2Src2Fix(Ctx);

  // Upgrade far trampolines' forward edge from s_add_pc_i64 to the
  // SCC-preserving set-pc expansion via instruction stealing. Runs after all
  // patch passes so every far site's finished body is present and adjacent
  // sites can be merged. Reuses the backward edge's SGPR block, so it must run
  // before the per-kernel SGPR accounting below (which is already charged for
  // that block at emit time); it never reserves new SGPRs.
  expandForwardSetPc(Ctx);

  for (const llvm::StringMapEntry<KernelPatchStats> &KV : KernelStats) {
    StringRef KName = KV.first();
    const KernelPatchStats &Stats = KV.second;
    if (KName.empty())
      continue;
    std::optional<unsigned> VgprsBefore =
        Elf.getKernelVgprCount(KName, Config.VgprGranuleSize);
    std::optional<unsigned> SgprsBefore = Elf.getKernelSgprCount(KName);
    if (Stats.ExtraVgprs > 0)
      Elf.updateKernelDescriptor(KName, Stats.ExtraVgprs,
                                 Config.VgprGranuleSize);
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
      if (!Elf.updateKernelDescriptorSgprCount(KName, RequiredSgprs)) {
        log() << "hotswap: error: failed to update SGPR count for kernel "
              << KName << "\n";
        return std::nullopt;
      }
    }
    std::optional<unsigned> VgprsAfter =
        Elf.getKernelVgprCount(KName, Config.VgprGranuleSize);
    std::optional<unsigned> SgprsAfter = Elf.getKernelSgprCount(KName);
    log() << "hotswap: liveness: kernel " << KName
          << ": vgprs_before=" << VgprsBefore.value_or(0)
          << ", vgprs_after=" << VgprsAfter.value_or(0)
          << ", sgprs_before=" << SgprsBefore.value_or(0)
          << ", sgprs_after=" << SgprsAfter.value_or(0)
          << ", scratch_reused=" << Stats.ScratchReused
          << ", scratch_above_kd=" << Stats.ScratchAboveKd << "\n";
  }
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
                        uint64_t TextSize, const LLVMState &LS) {
  // Fail-fast on the first encoding error: the position of later
  // trampolines depends on earlier ones, so a single bad branch would
  // cascade into incorrect layout. A single failure invalidates the whole
  // rewrite, so there is nothing useful to recover beyond it.
  uint64_t TrampOffset = TextSize;
  for (Trampoline &T : Trampolines) {
    uint64_t TP = TrampOffset;
    TrampOffset += T.Bytes.size();

    // Long trampolines reserve a wider branch-back slot; short ones use
    // s_branch. The backward edge of a long trampoline uses the SCC-preserving
    // set-pc expansion by default (UsesSetPCBack) and only the legacy backward
    // s_add_pc_i64 under HOTSWAP_BACK_ADDPC. The forward edge uses an 8-byte
    // forward s_add_pc_i64 unless expandForwardSetPc upgraded it to the set-pc
    // expansion (UsesSetPCFwd), in which case the site footprint is the stolen
    // window (StolenBytes) and the backward edge returns past it. Every slot is
    // s_nop-padded out to its reserved size after the branch is written.
    const uint32_t BackReserve =
        T.Long ? (T.UsesSetPCBack ? LongBranchBackSeqBytes : LongBranchMaxBytes)
               : MinInstSize;
    const uint64_t BackSlot = TP + T.Bytes.size() - BackReserve;
    // The site footprint is the original slot, or the full stolen window when
    // the forward edge was upgraded to set-pc.
    const uint32_t SiteFootprint =
        T.UsesSetPCFwd ? T.StolenBytes : T.OriginalSize;
    const uint64_t ReturnTo = T.OriginalOffset + SiteFootprint;

    SmallVector<uint8_t> BrBack =
        !T.Long ? LS.encodeSBranch(BackSlot, ReturnTo)
                : (T.UsesSetPCBack ? encodeSetPCLongBranch(LS, BackSlot, ReturnTo,
                                                           T.LongBranchSgprBase)
                                   : encodeLongBranch(LS, BackSlot, ReturnTo));
    if (BrBack.empty() || BrBack.size() > BackReserve) {
      log() << "hotswap: error: trampoline branch-back encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << (T.Long ? " (long)\n" : "\n");
      return false;
    }
    std::memcpy(T.Bytes.data() + T.Bytes.size() - BackReserve, BrBack.data(),
                BrBack.size());
    for (uint32_t I = BrBack.size(); I + MinInstSize <= BackReserve;
         I += MinInstSize)
      std::memcpy(T.Bytes.data() + T.Bytes.size() - BackReserve + I,
                  LS.SNopBytes.data(), MinInstSize);

    SmallVector<uint8_t> BrFwd =
        T.UsesSetPCFwd
            ? encodeSetPCLongBranch(LS, T.OriginalOffset, TP,
                                    T.LongBranchSgprBase)
            : (T.Long ? encodeLongBranch(LS, T.OriginalOffset, TP)
                      : LS.encodeSBranch(T.OriginalOffset, TP));
    if (BrFwd.empty() || BrFwd.size() > SiteFootprint) {
      log() << "hotswap: error: trampoline branch-fwd encoding failed at 0x"
            << utohexstr(T.OriginalOffset)
            << (T.UsesSetPCFwd ? " (long, set-pc fwd)\n"
                               : (T.Long ? " (long)\n" : "\n"));
      return false;
    }
    std::memcpy(Text + T.OriginalOffset, BrFwd.data(), BrFwd.size());
    // Pad the tail of the site footprint with cached s_nop bytes.
    for (uint32_t I = BrFwd.size(); I + MinInstSize <= SiteFootprint;
         I += MinInstSize)
      std::memcpy(Text + T.OriginalOffset + I, LS.SNopBytes.data(),
                  MinInstSize);
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
  if (COMGR::env::shouldEmitVerboseLogs()) {
    // Trampolines are appended contiguously right after the original .text,
    // in array order. Their virtual address is textAddr + original textSize +
    // cumulative body bytes. Emit a map line per trampoline so a fault PC in
    // the trampoline region can be traced to its origin kernel + site.
    uint64_t Pos = Elf.textSize();
    for (const Trampoline &T : Trampolines) {
      uint64_t TrampVA = Elf.textAddr() + Pos;
      uint64_t SiteVA = Elf.textAddr() + T.OriginalOffset;
      std::string K = Elf.findKernelAtAddress(SiteVA);
      if (K.empty())
        K = "<unknown>";
      log() << "hotswap-map: tramp kernel='" << K << "' tramp_vaddr=0x"
            << utohexstr(TrampVA) << " site_vaddr=0x" << utohexstr(SiteVA)
            << " orig_off=0x" << utohexstr(T.OriginalOffset)
            << " bytes=" << T.Bytes.size() << "\n";
      Pos += T.Bytes.size();
    }
  }
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

amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<MemoryBuffer> &Out) {
  // The dispatcher fetches the patch vtable lazily via
  // getHotswapPatchVTable() inside applyGfx1250B0toA0Rules; the singleton's
  // initializer binds every register*Patch slot on first access, so no
  // explicit install step is needed here.

  if (!Options.RunB0A0Patches && !Options.RunEntryTrampolines) {
    std::unique_ptr<WritableMemoryBuffer> Result =
        copyOutputBuffer(ElfData, ElfSize, "no-op");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    Out = std::move(Result);
    return AMD_COMGR_STATUS_SUCCESS;
  }

  // Take a working copy so the input is preserved and we have a mutable
  // buffer to parse / patch.
  std::vector<uint8_t> Buf(static_cast<const uint8_t *>(ElfData),
                           static_cast<const uint8_t *>(ElfData) + ElfSize);

  Expected<ElfView> ViewOrErr = ElfView::create(Buf.data(), Buf.size());
  if (!ViewOrErr) {
    log() << "hotswap: error: retargetCodeObject: input is not a "
          << "parseable ELF64 (" << toString(ViewOrErr.takeError()) << ").\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (ViewOrErr->textSize() == 0) {
    log() << "hotswap: error: retargetCodeObject: input ELF has empty "
          << ".text section; nothing to rewrite.\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfView &Elf = *ViewOrErr;

  LLVMState LS = initLLVM(TargetIdent);
  if (!LS.Valid) {
    log() << "hotswap: error: retargetCodeObject: initLLVM failed "
          << "for CPU '" << TargetIdent.Processor << "'; aborting rewrite.\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  RewriteConfig Config = makeGfx1250B0A0Config();

  uint8_t *Text = Elf.textData();
  uint64_t Count = 0;
  std::vector<Trampoline> Deferred;
  std::vector<ScratchPatchInfo> ScratchPatches;
  if (Options.RunB0A0Patches) {
    std::vector<InternalDecodedInst> Decoded;
    if (!decodeTextSection(Text, Elf.textSize(), LS, Decoded)) {
      log() << "hotswap: error: retargetCodeObject: decodeTextSection "
            << "failed on .text (" << Elf.textSize() << " bytes).\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    std::optional<uint32_t> Patched =
        applyGfx1250B0toA0Rules(Decoded, Text, Elf.textSize(), LS, Deferred,
                                Elf, ScratchPatches, Config);
    if (!Patched)
      return AMD_COMGR_STATUS_ERROR;
    Count = *Patched;
    log() << "hotswap: applied " << Count << " B0-to-A0 patches\n";
  } else {
    log() << "hotswap: B0-to-A0 patches disabled for this rewrite\n";
  }

  std::unique_ptr<WritableMemoryBuffer> Result;
  std::vector<Trampoline> Growth = Deferred;
  if (!Deferred.empty()) {
    if (!fixupTrampolineBranches(Deferred, Text, Elf.textSize(), LS)) {
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
    std::optional<uint32_t> EntryCount = appendKernelEntryTrampolines(
        Elf, LS, Config.MaxSgprs, Growth, EntryFixups);
    if (!EntryCount)
      return AMD_COMGR_STATUS_ERROR;
    Count += *EntryCount;
  } else {
    log() << "hotswap: kernel-entry trampolines disabled for this rewrite\n";
  }

  if (!Growth.empty()) {
    Result = Elf.growWithTrampolines(Growth, LS.SNopBytes);
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
    patchDebugSections(*Result, Deferred, Elf, GrowthTotal);
    if (!rewriteKernelEntryDescriptorOffsets(*Result, Elf.textSize(),
                                             EntryFixups, LS))
      return AMD_COMGR_STATUS_ERROR;
  } else {
    Result = copyOutputBuffer(Buf.data(), ElfSize, "patched");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (!ScratchPatches.empty())
    runScratchVerification(*Result, LS, ScratchPatches, Config.MaxVgprs);

  // Debug dump: when HOTSWAP_DUMP_DIR is set, write the input (pre) and patched
  // (post) code objects to disk so the exact bytes the loader ships can be
  // disassembled and diffed against a synthetic repro. Files are keyed by a
  // per-process index plus the input size so multiple transpiles are
  // distinguishable and matchable across runs.
  if (const char *DumpDir = getenv("HOTSWAP_DUMP_DIR")) {
    static std::atomic<unsigned> DumpIdx{0};
    unsigned Idx = DumpIdx.fetch_add(1);
    auto WriteFile = [&](const char *Tag, const void *Data, size_t Sz) {
      char Path[4096];
      snprintf(Path, sizeof(Path), "%s/co_%03u_in%zu_%s.elf", DumpDir, Idx,
               ElfSize, Tag);
      if (FILE *F = fopen(Path, "wb")) {
        fwrite(Data, 1, Sz, F);
        fclose(F);
        log() << "hotswap: dumped " << Tag << " -> " << Path << " (" << Sz
              << " B)\n";
      } else {
        log() << "hotswap: error: could not open dump file " << Path << "\n";
      }
    };
    WriteFile("pre", ElfData, ElfSize);
    WriteFile("post", Result->getBufferStart(), Result->getBufferSize());
  }

  Out = std::move(Result);
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace hotswap
} // namespace COMGR
