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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
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
static constexpr unsigned Gfx1250VccLoIndex = 106;
static constexpr unsigned Gfx1250SgprSlotsWithVcc = 108;
static constexpr unsigned Gfx1250VgprGranuleSize = 16;

static bool isSBranchReachable(uint64_t From, uint64_t To) {
  if (From > std::numeric_limits<uint64_t>::max() - MinInstSize)
    return false;
  const uint64_t Pc = From + MinInstSize;
  if (To >= Pc) {
    const uint64_t Delta = To - Pc;
    return Delta % MinInstSize == 0 &&
           Delta / MinInstSize <= static_cast<uint64_t>(BranchOffsetMax);
  }

  const uint64_t Delta = Pc - To;
  return Delta % MinInstSize == 0 &&
         Delta / MinInstSize <=
             static_cast<uint64_t>(-static_cast<int64_t>(BranchOffsetMin));
}

static std::optional<uint64_t>
kernelEntryVAddr(const KernelDescriptorInfo &KD) {
  if (KD.EntryOffset >= 0)
    return checkedAddUint64(KD.VAddr, static_cast<uint64_t>(KD.EntryOffset),
                            "kernel entry address");

  uint64_t Magnitude = KD.EntryOffset == std::numeric_limits<int64_t>::min()
                           ? uint64_t{1} << 63
                           : static_cast<uint64_t>(-KD.EntryOffset);
  if (KD.VAddr < Magnitude)
    return std::nullopt;
  return KD.VAddr - Magnitude;
}

static std::vector<KernelTextRange>
collectKernelTextRanges(ElfView &Elf, const LLVMState &LS) {
  DenseSet<uint64_t> EntryVAddrs;
  std::optional<uint64_t> TextEnd = checkedAddUint64(
      Elf.textAddr(), Elf.textSize(), "kernel range text end");
  if (!TextEnd)
    return {};

  for (const KernelDescriptorInfo &KD : Elf.kernelDescriptors()) {
    std::optional<uint64_t> Entry = kernelEntryVAddr(KD);
    if (!Entry)
      continue;
    if (*Entry >= Elf.textAddr() && *Entry < *TextEnd) {
      EntryVAddrs.insert(*Entry);
      continue;
    }

    const uint8_t *Stub = Elf.dataAtVAddr(*Entry, KernelEntryStubStride);
    if (!Stub)
      continue;
    std::optional<uint64_t> OriginalEntry =
        getKernelEntryTrampolineTargetVAddr(
            ArrayRef<uint8_t>(Stub, KernelEntryStubStride), *Entry, LS);
    if (OriginalEntry && *OriginalEntry >= Elf.textAddr() &&
        *OriginalEntry < *TextEnd)
      EntryVAddrs.insert(*OriginalEntry);
  }

  std::vector<KernelTextRange> Result;
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges()) {
    if (!EntryVAddrs.contains(Range.Begin) || Range.Begin < Elf.textAddr())
      continue;
    Result.push_back(
        {Range.Begin - Elf.textAddr(), Range.End - Elf.textAddr()});
  }
  return Result;
}

/// Decode every executable section into one .text-relative instruction view
/// for fixed-point control-flow analysis. The patch dispatcher still mutates
/// only the original .text stream; appended HotSwap pools are read-only input
/// here so generated source and return edges can be resolved on rewrite #2.
static bool buildHotswapAnalysisDecoded(
    const ElfView &Elf, const LLVMState &LS,
    ArrayRef<InternalDecodedInst> TextDecoded,
    std::vector<InternalDecodedInst> &Out) {
  Out.assign(TextDecoded.begin(), TextDecoded.end());
  std::optional<uint64_t> TextEnd = checkedAddUint64(
      Elf.textAddr(), Elf.textSize(), "analysis .text end");
  if (!TextEnd)
    return false;

  for (const ElfView::ELFT::Shdr &Shdr : Elf.sections()) {
    if (&Shdr == Elf.textSection() ||
        !(Shdr.sh_flags & ELF::SHF_EXECINSTR) ||
        Shdr.sh_type == ELF::SHT_NOBITS || Shdr.sh_size == 0 ||
        Shdr.sh_addr < *TextEnd)
      continue;
    if (Shdr.sh_offset > Elf.size() ||
        Shdr.sh_size > Elf.size() - Shdr.sh_offset)
      return false;

    std::vector<InternalDecodedInst> SectionDecoded;
    if (!decodeTextSection(Elf.data() + Shdr.sh_offset, Shdr.sh_size, LS,
                           SectionDecoded)) {
      log() << "hotswap: error: failed to decode executable analysis section "
               "at vaddr 0x"
            << utohexstr(Shdr.sh_addr) << "\n";
      return false;
    }

    const uint64_t Base = Shdr.sh_addr - Elf.textAddr();
    for (InternalDecodedInst &DI : SectionDecoded) {
      std::optional<uint64_t> Rebased = checkedAddUint64(
          Base, DI.Offset, "executable analysis instruction offset");
      if (!Rebased)
        return false;
      DI.Offset = *Rebased;
      Out.push_back(std::move(DI));
    }
  }

  llvm::sort(Out, [](const InternalDecodedInst &L,
                     const InternalDecodedInst &R) {
    return L.Offset < R.Offset;
  });
  for (size_t I = 1; I < Out.size(); ++I)
    if (Out[I - 1].Offset == Out[I].Offset) {
      log() << "hotswap: error: overlapping executable analysis instruction "
               "at 0x"
            << utohexstr(Out[I].Offset) << "\n";
      return false;
    }
  return true;
}

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
  std::vector<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
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

static void appendNopSledIfLarge(std::vector<NopSled> &Sleds, uint64_t Start,
                                 uint64_t End,
                                 const ElfView::FunctionTextRange &Range) {
  if (End - Start >= MinNopSledSize)
    Sleds.push_back({Start, End, Start, Range.Begin, Range.End});
}

struct TextSymbolRange {
  uint64_t Begin = 0;
  uint64_t End = 0;
};

/// Add anonymous alignment after explicitly-sized functions as local code
/// caves reserved for required rewrites. This is intentionally narrower
/// than treating arbitrary padding as executable: the gap must end at the next
/// 256-byte-aligned function, contain only zeroes or canonical s_nop encodings,
/// have no symbol or branch target, and follow a kernel exit or function
/// return. These constraints keep labeled data and adjacent code out of the
/// sled map while supporting internal device functions without descriptors.
static void appendFunctionAlignmentSleds(
    std::vector<NopSled> &Sleds, ArrayRef<InternalDecodedInst> Decoded,
    const LLVMState &LS, const ElfView &Elf) {
  constexpr uint64_t FunctionAlign = 256;
  const uint64_t TextAddr = Elf.textAddr();
  const uint64_t TextSize = Elf.textSize();

  for (const ElfView::ELFT::Shdr &Section : Elf.sections()) {
    if ((Section.sh_type == ELF::SHT_REL ||
         Section.sh_type == ELF::SHT_RELA) &&
        Section.sh_info == Elf.textSectionIndex())
      return;
  }

  DenseMap<uint64_t, uint64_t> FunctionEnds;
  DenseSet<uint64_t> ConflictingFunctionEnds;
  DenseSet<uint64_t> FunctionStarts;
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges()) {
    if (!Range.Symbol || Range.Symbol->st_size == 0 ||
        Range.Begin < TextAddr || Range.End < Range.Begin ||
        Range.End - TextAddr > TextSize)
      continue;
    const uint64_t Begin = Range.Begin - TextAddr;
    const uint64_t End = Range.End - TextAddr;
    FunctionStarts.insert(Begin);
    auto [It, Inserted] = FunctionEnds.try_emplace(Begin, End);
    if (!Inserted && It->second != End)
      ConflictingFunctionEnds.insert(Begin);
  }

  SmallVector<TextSymbolRange, 64> TextSymbols;
  for (const ElfView::ELFT::Shdr &Symtab : Elf.sections()) {
    if (Symtab.sh_type != ELF::SHT_SYMTAB &&
        Symtab.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ElfView::ELFT::SymRange> Symbols = Elf.file().symbols(&Symtab);
    if (!Symbols) {
      consumeError(Symbols.takeError());
      return;
    }
    for (const ElfView::ELFT::Sym &Sym : *Symbols) {
      if (Sym.st_shndx != Elf.textSectionIndex() || Sym.st_value < TextAddr ||
          Sym.st_value - TextAddr >= TextSize)
        continue;
      const uint64_t Begin = Sym.st_value - TextAddr;
      const uint64_t End =
          std::min(TextSize, Begin + std::min<uint64_t>(Sym.st_size,
                                                       TextSize - Begin));
      TextSymbols.push_back({Begin, End});
    }
  }

  SmallVector<uint64_t, 64> DirectBranchTargets;
  if (LS.MIA) {
    for (const InternalDecodedInst &DI : Decoded) {
      if (!LS.MIA->isBranch(DI.Inst) ||
          LS.MIA->isIndirectBranch(DI.Inst))
        continue;
      uint64_t Target = 0;
      if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target))
        DirectBranchTargets.push_back(Target);
    }
    llvm::sort(DirectBranchTargets);
    DirectBranchTargets.erase(
        std::unique(DirectBranchTargets.begin(), DirectBranchTargets.end()),
        DirectBranchTargets.end());
  }

  for (const auto &FunctionRange : FunctionEnds) {
    const uint64_t FunctionBegin = FunctionRange.first;
    const uint64_t FunctionEnd = FunctionRange.second;
    if (ConflictingFunctionEnds.contains(FunctionBegin) ||
        FunctionEnd >= TextSize)
      continue;

    const uint64_t PaddingBegin = alignTo(FunctionEnd, uint64_t{8});
    const uint64_t PaddingEnd = alignTo(FunctionEnd, FunctionAlign);
    if (PaddingEnd <= PaddingBegin || PaddingEnd > TextSize ||
        !FunctionStarts.contains(PaddingEnd) ||
        PaddingEnd - PaddingBegin < MinNopSledSize)
      continue;

    bool HasSymbol = llvm::any_of(TextSymbols, [&](const TextSymbolRange &S) {
      const bool StartsInGap = S.Begin >= FunctionEnd && S.Begin < PaddingEnd;
      const bool OverlapsGap =
          S.End > FunctionEnd && S.Begin < PaddingEnd;
      return StartsInGap || OverlapsGap;
    });
    if (HasSymbol)
      continue;

    ArrayRef<uint8_t> Padding(Elf.textData() + FunctionEnd,
                              PaddingEnd - FunctionEnd);
    bool IsAnonymousPadding = Padding.size() % MinInstSize == 0;
    for (size_t I = 0; IsAnonymousPadding && I < Padding.size();
         I += MinInstSize) {
      ArrayRef<uint8_t> Dword = Padding.slice(I, MinInstSize);
      bool IsZero = llvm::all_of(Dword, [](uint8_t Byte) { return Byte == 0; });
      IsAnonymousPadding =
          IsZero || Dword == ArrayRef<uint8_t>(LS.SNopBytes);
    }
    if (!IsAnonymousPadding)
      continue;

    auto First = llvm::lower_bound(
        Decoded, FunctionBegin,
        [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    auto After = llvm::lower_bound(
        Decoded, FunctionEnd,
        [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    if (First == After || First->Offset != FunctionBegin)
      continue;

    bool UnsafeControlFlow = false;
    auto GapTarget = llvm::lower_bound(DirectBranchTargets, FunctionEnd);
    if (GapTarget != DirectBranchTargets.end() && *GapTarget < PaddingEnd)
      UnsafeControlFlow = true;

    for (auto It = First; !UnsafeControlFlow && It != After; ++It) {
      const InternalDecodedInst &DI = *It;
      const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
      const bool IsBranch = LS.MIA ? LS.MIA->isBranch(DI.Inst)
                                   : Desc.isBranch();
      const bool IsCall = LS.MIA ? LS.MIA->isCall(DI.Inst) : Desc.isCall();
      const bool IsReturn =
          LS.MIA ? LS.MIA->isReturn(DI.Inst) : Desc.isReturn();
      const bool IsTerminalSetPc =
          DI.Offset + DI.Size == FunctionEnd &&
          DI.Mnemonic == "s_set_pc_i64";
      const bool IsIndirect =
          IsBranch && (LS.MIA ? LS.MIA->isIndirectBranch(DI.Inst)
                              : Desc.isIndirectBranch());

      // Calls may occur inside compiler-emitted device functions. Their
      // targets are function entries, while the candidate gap is proven to
      // contain no symbol. Reject only an unclassified indirect jump that
      // could target anonymous bytes.
      if (IsIndirect && !IsCall && !IsReturn && !IsTerminalSetPc)
        UnsafeControlFlow = true;
    }
    const InternalDecodedInst &Last = *std::prev(After);
    if (Last.Offset + Last.Size != FunctionEnd)
      continue;
    const MCInstrDesc &LastDesc = LS.MCII->get(Last.Inst.getOpcode());
    const bool LastIsReturn = LS.MIA ? LS.MIA->isReturn(Last.Inst)
                                     : LastDesc.isReturn();
    if (UnsafeControlFlow ||
        (Last.Mnemonic != "s_endpgm" &&
         Last.Mnemonic != "s_set_pc_i64" && !LastIsReturn))
      continue;

    // FunctionStart/FunctionEnd record the function used to prove that the
    // padding is unreachable. Unlike an interior NOP sled, this anonymous gap
    // may be borrowed by any required rewrite whose two short branches reach
    // it.
    Sleds.push_back({PaddingBegin, PaddingEnd, PaddingBegin, FunctionBegin,
                     FunctionEnd, true});
    log() << "hotswap: using anonymous function alignment [0x"
          << utohexstr(PaddingBegin) << ", 0x" << utohexstr(PaddingEnd)
          << ") as a local code cave\n";
  }
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

  DenseSet<uint64_t> InteriorTargets;
  DenseSet<uint64_t> IndirectFunctions;
  if (LS.MIA) {
    for (const InternalDecodedInst &DI : Decoded) {
      if (LS.MIA->isBranch(DI.Inst) &&
          !LS.MIA->isIndirectBranch(DI.Inst)) {
        uint64_t Target = 0;
        if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target))
          InteriorTargets.insert(Target);
      }
      if (LS.MIA->isBranch(DI.Inst) && LS.MIA->isIndirectBranch(DI.Inst) &&
          !LS.MIA->isCall(DI.Inst) && !LS.MIA->isReturn(DI.Inst)) {
        std::optional<ElfView::FunctionTextRange> Range =
            Elf.findFunctionTextRangeAtOffset(DI.Offset);
        if (Range)
          IndirectFunctions.insert(Range->Begin);
      }
    }
  }
  for (const ElfView::ELFT::Shdr &Symtab : Elf.sections()) {
    if (Symtab.sh_type != ELF::SHT_SYMTAB &&
        Symtab.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ElfView::ELFT::SymRange> Symbols = Elf.file().symbols(&Symtab);
    if (!Symbols) {
      consumeError(Symbols.takeError());
      continue;
    }
    for (const ElfView::ELFT::Sym &Sym : *Symbols)
      if (Sym.st_shndx == Elf.textSectionIndex() &&
          Sym.st_value >= Elf.textAddr() &&
          Sym.st_value - Elf.textAddr() < Elf.textSize())
        InteriorTargets.insert(Sym.st_value - Elf.textAddr());
  }
  llvm::erase_if(Sleds, [&](const NopSled &Sled) {
    if (Sled.IsTailPadding)
      return false;
    if (IndirectFunctions.contains(Sled.FunctionStart))
      return true;
    return llvm::any_of(InteriorTargets, [&](uint64_t Target) {
      return Target > Sled.Start && Target < Sled.End;
    });
  });

  appendFunctionAlignmentSleds(Sleds, Decoded, LS, Elf);
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
                                 ArrayRef<uint8_t> Replacement,
                                 ArrayRef<uint8_t> OriginalTail) {
  const LLVMState &LS = Ctx.LS;
  if (!OriginalTail.empty() &&
      (InstSize < 2 * MinInstSize ||
       OriginalTail.size() > InstSize - MinInstSize))
    return false;
  std::optional<uint64_t> ReturnTo = checkedAddUint64(
      InstOffset, OriginalTail.empty() ? InstSize : MinInstSize,
      "NOP-sled return offset");
  uint64_t GuardSize = Sled.IsTailPadding ? 0 : MinInstSize;
  std::optional<uint64_t> PayloadOffset = checkedAddUint64(
      Sled.WritePos, GuardSize, "NOP-sled payload offset");
  std::optional<uint64_t> BackFrom =
      PayloadOffset
          ? checkedAddUint64(*PayloadOffset, Replacement.size(),
                             "NOP-sled branch-back offset")
          : std::nullopt;
  std::optional<uint64_t> StorageEnd =
      BackFrom ? checkedAddUint64(*BackFrom, MinInstSize,
                                  "NOP-sled storage end")
               : std::nullopt;
  if (!ReturnTo || !PayloadOffset || !BackFrom || !StorageEnd ||
      *StorageEnd > Sled.End)
    return false;
  SmallVector<uint8_t> BrBack = LS.encodeSBranch(*BackFrom, *ReturnTo);
  if (BrBack.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-back "
          << "at sled offset 0x"
          << utohexstr(*BackFrom) << " -> 0x" << utohexstr(*ReturnTo)
          << " failed.\n";
    return false;
  }

  SmallVector<uint8_t> BrFwd = LS.encodeSBranch(InstOffset, *PayloadOffset);
  if (BrFwd.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-fwd "
          << "at original offset 0x" << utohexstr(InstOffset) << " -> sled 0x"
          << utohexstr(*PayloadOffset) << " failed.\n";
    return false;
  }

  SmallVector<uint8_t> Guard;
  if (GuardSize) {
    Guard = LS.encodeSBranch(Sled.WritePos, *StorageEnd);
    if (Guard.size() != MinInstSize)
      return false;
  }

  uint64_t SledWriteBegin = Sled.WritePos;
  if (GuardSize)
    std::memcpy(Ctx.Text + Sled.WritePos, Guard.data(), Guard.size());
  std::memcpy(Ctx.Text + *PayloadOffset, Replacement.data(), Replacement.size());
  std::memcpy(Ctx.Text + *BackFrom, BrBack.data(),
              BrBack.size());
  std::memcpy(Ctx.Text + InstOffset, BrFwd.data(), BrFwd.size());

  uint32_t PadStart = MinInstSize;
  if (!OriginalTail.empty()) {
    std::memcpy(Ctx.Text + InstOffset + MinInstSize, OriginalTail.data(),
                OriginalTail.size());
    PadStart += OriginalTail.size();
  }
  // Pad the unused tail of the original slot with cached s_nop bytes.
  for (uint32_t I = PadStart; I < InstSize; I += MinInstSize)
    std::memcpy(Ctx.Text + InstOffset + I, LS.SNopBytes.data(), MinInstSize);

  Sled.WritePos = *StorageEnd;
  for (uint64_t Offset = SledWriteBegin; Offset < Sled.WritePos;
       Offset += MinInstSize)
    Ctx.MutatedOffsets.insert(Offset);
  for (uint64_t Offset = InstOffset; Offset < InstOffset + InstSize;
       Offset += MinInstSize)
    Ctx.MutatedOffsets.insert(Offset);
  Ctx.ImmediateCaveResumeOffsets.insert(*ReturnTo);
  return true;
}

/// Route a compact required replacement through otherwise-unused function-tail
/// padding when no single body cave is directly reachable. Each intermediate
/// cave contributes two independent s_branch slots: one for the outbound path
/// and one for the return path. The path therefore needs no scratch register
/// and remains safe under every device-function calling convention.
static bool emitToRoutedTailCave(PatchContext &Ctx, uint64_t InstOffset,
                                 uint32_t InstSize,
                                 ArrayRef<uint8_t> Replacement,
                                 ArrayRef<uint8_t> OriginalTail) {
  if (!OriginalTail.empty() &&
      (InstSize < 2 * MinInstSize ||
       OriginalTail.size() > InstSize - MinInstSize))
    return false;

  std::optional<uint64_t> ReturnTo = checkedAddUint64(
      InstOffset, OriginalTail.empty() ? InstSize : MinInstSize,
      "routed cave return offset");
  std::optional<uint64_t> BodyStorage = checkedAddUint64(
      Replacement.size(), MinInstSize, "routed cave body storage");
  if (!ReturnTo || !BodyStorage)
    return false;

  struct Candidate {
    size_t SledIndex = 0;
    uint64_t Pos = 0;
    uint64_t Free = 0;
  };
  SmallVector<Candidate, 128> Candidates;
  for (size_t I = 0, E = Ctx.NopSleds.size(); I < E; ++I) {
    const NopSled &Sled = Ctx.NopSleds[I];
    if (!Sled.IsTailPadding || Sled.WritePos > Sled.End)
      continue;
    uint64_t Free = Sled.End - Sled.WritePos;
    if (Free >= MinInstSize)
      Candidates.push_back({I, Sled.WritePos, Free});
  }
  llvm::sort(Candidates, [](const Candidate &L, const Candidate &R) {
    return L.Pos < R.Pos;
  });
  if (Candidates.empty())
    return false;

  auto BodyEndAt = [&](const Candidate &C) -> std::optional<uint64_t> {
    if (C.Free < *BodyStorage)
      return std::nullopt;
    return checkedAddUint64(C.Pos, Replacement.size(),
                            "routed cave body end");
  };
  auto DirectBodyWorks = [&](const Candidate &C) {
    std::optional<uint64_t> BodyEnd = BodyEndAt(C);
    return BodyEnd && isSBranchReachable(InstOffset, C.Pos) &&
           isSBranchReachable(*BodyEnd, *ReturnTo);
  };

  int BodyCandidate = -1;
  SmallVector<int, 16> Route;
  for (size_t I = 0, E = Candidates.size(); I < E; ++I)
    if (DirectBodyWorks(Candidates[I])) {
      BodyCandidate = static_cast<int>(I);
      break;
    }

  if (BodyCandidate < 0) {
    SmallVector<int, 128> Parent(Candidates.size(), -2);
    SmallVector<unsigned, 128> Worklist;
    for (size_t I = 0, E = Candidates.size(); I < E; ++I) {
      const Candidate &C = Candidates[I];
      if (C.Free < 2 * MinInstSize)
        continue;
      if (isSBranchReachable(InstOffset, C.Pos) &&
          isSBranchReachable(C.Pos + MinInstSize, *ReturnTo)) {
        Parent[I] = -1;
        Worklist.push_back(I);
      }
    }

    auto PathContainsSled = [&](unsigned Last, size_t SledIndex) {
      for (int I = static_cast<int>(Last); I >= 0; I = Parent[I])
        if (Candidates[I].SledIndex == SledIndex)
          return true;
      return false;
    };
    auto CandidateRange = [&](uint64_t Pos) {
      const uint64_t Slack = MaxSledDistance + 2 * MinInstSize;
      const uint64_t Low = Pos > Slack ? Pos - Slack : 0;
      const uint64_t High =
          Pos > std::numeric_limits<uint64_t>::max() - Slack
              ? std::numeric_limits<uint64_t>::max()
              : Pos + Slack;
      auto First = llvm::lower_bound(
          Candidates, Low,
          [](const Candidate &C, uint64_t Offset) { return C.Pos < Offset; });
      auto After = llvm::upper_bound(
          Candidates, High,
          [](uint64_t Offset, const Candidate &C) { return Offset < C.Pos; });
      return std::make_pair(First, After);
    };

    int LastRelay = -1;
    for (size_t Next = 0; Next < Worklist.size() && BodyCandidate < 0;
         ++Next) {
      unsigned Current = Worklist[Next];
      const Candidate &From = Candidates[Current];
      auto [First, After] = CandidateRange(From.Pos);
      for (auto It = First; It != After; ++It) {
        unsigned J = It - Candidates.begin();
        const Candidate &To = *It;
        if (PathContainsSled(Current, To.SledIndex))
          continue;

        if (std::optional<uint64_t> BodyEnd = BodyEndAt(To)) {
          if (isSBranchReachable(From.Pos, To.Pos) &&
              isSBranchReachable(*BodyEnd, From.Pos + MinInstSize)) {
            BodyCandidate = static_cast<int>(J);
            LastRelay = static_cast<int>(Current);
            break;
          }
        }

        if (To.Free < 2 * MinInstSize || Parent[J] != -2 ||
            !isSBranchReachable(From.Pos, To.Pos) ||
            !isSBranchReachable(To.Pos + MinInstSize,
                                From.Pos + MinInstSize))
          continue;
        Parent[J] = static_cast<int>(Current);
        Worklist.push_back(J);
      }
    }
    if (BodyCandidate < 0)
      return false;
    for (int I = LastRelay; I >= 0; I = Parent[I])
      Route.push_back(I);
    std::reverse(Route.begin(), Route.end());
  }

  Candidate &Body = Candidates[BodyCandidate];
  std::optional<uint64_t> BodyEnd = BodyEndAt(Body);
  std::optional<uint64_t> BodyStorageEnd =
      BodyEnd ? checkedAddUint64(*BodyEnd, MinInstSize,
                                 "routed cave body storage end")
              : std::nullopt;
  if (!BodyEnd || !BodyStorageEnd)
    return false;

  SmallVector<SmallVector<uint8_t>, 16> ForwardBranches;
  SmallVector<SmallVector<uint8_t>, 16> ReturnBranches;
  uint64_t FirstTarget =
      Route.empty() ? Body.Pos : Candidates[Route.front()].Pos;
  SmallVector<uint8_t> SourceBranch =
      Ctx.LS.encodeSBranch(InstOffset, FirstTarget);
  if (SourceBranch.size() != MinInstSize)
    return false;
  for (size_t I = 0, E = Route.size(); I < E; ++I) {
    const Candidate &Node = Candidates[Route[I]];
    uint64_t Target = I + 1 < E ? Candidates[Route[I + 1]].Pos : Body.Pos;
    SmallVector<uint8_t> Branch = Ctx.LS.encodeSBranch(Node.Pos, Target);
    if (Branch.size() != MinInstSize)
      return false;
    ForwardBranches.push_back(std::move(Branch));

    uint64_t BackTarget =
        I == 0 ? *ReturnTo : Candidates[Route[I - 1]].Pos + MinInstSize;
    SmallVector<uint8_t> Back =
        Ctx.LS.encodeSBranch(Node.Pos + MinInstSize, BackTarget);
    if (Back.size() != MinInstSize)
      return false;
    ReturnBranches.push_back(std::move(Back));
  }
  uint64_t BodyBackTarget =
      Route.empty() ? *ReturnTo
                    : Candidates[Route.back()].Pos + MinInstSize;
  SmallVector<uint8_t> BodyBack =
      Ctx.LS.encodeSBranch(*BodyEnd, BodyBackTarget);
  if (BodyBack.size() != MinInstSize)
    return false;

  std::memcpy(Ctx.Text + InstOffset, SourceBranch.data(), MinInstSize);
  uint32_t PadStart = MinInstSize;
  if (!OriginalTail.empty()) {
    std::memcpy(Ctx.Text + InstOffset + MinInstSize, OriginalTail.data(),
                OriginalTail.size());
    PadStart += OriginalTail.size();
  }
  for (uint32_t I = PadStart; I < InstSize; I += MinInstSize)
    std::memcpy(Ctx.Text + InstOffset + I, Ctx.LS.SNopBytes.data(),
                MinInstSize);

  for (size_t I = 0, E = Route.size(); I < E; ++I) {
    Candidate &Node = Candidates[Route[I]];
    NopSled &Sled = Ctx.NopSleds[Node.SledIndex];
    std::memcpy(Ctx.Text + Node.Pos, ForwardBranches[I].data(), MinInstSize);
    std::memcpy(Ctx.Text + Node.Pos + MinInstSize,
                ReturnBranches[I].data(), MinInstSize);
    Sled.WritePos += 2 * MinInstSize;
    Ctx.MutatedOffsets.insert(Node.Pos);
    Ctx.MutatedOffsets.insert(Node.Pos + MinInstSize);
  }

  NopSled &BodySled = Ctx.NopSleds[Body.SledIndex];
  std::memcpy(Ctx.Text + Body.Pos, Replacement.data(), Replacement.size());
  std::memcpy(Ctx.Text + *BodyEnd, BodyBack.data(), MinInstSize);
  BodySled.WritePos = *BodyStorageEnd;
  for (uint64_t Offset = Body.Pos; Offset < *BodyStorageEnd;
       Offset += MinInstSize)
    Ctx.MutatedOffsets.insert(Offset);
  for (uint64_t Offset = InstOffset; Offset < InstOffset + InstSize;
       Offset += MinInstSize)
    Ctx.MutatedOffsets.insert(Offset);
  Ctx.ImmediateCaveResumeOffsets.insert(*ReturnTo);

  log() << "hotswap: routed required replacement at 0x"
        << utohexstr(InstOffset) << " through " << Route.size()
        << " tail-cave relay hop(s) to 0x" << utohexstr(Body.Pos) << "\n";
  return true;
}

// Encode an EXEC-independent, SCC-neutral PC-relative edge without
// s_add_pc_i64. HSV-009 makes s_add_pc_i64 unsafe on gfx1250 A0 regardless of
// direction. s_get_pc_i64 captures the following instruction, so the delta is
// measured from the add slot.
SmallVector<uint8_t> encodeSetPcLongBranch(const LLVMState &LS,
                                           uint64_t FromOffset,
                                           uint64_t TargetOffset,
                                           unsigned SgprBase) {
  std::string Pair =
      SgprBase == Gfx1250VccLoIndex
          ? "vcc"
          : "s[" + std::to_string(SgprBase) + ":" +
                std::to_string(SgprBase + 1) + "]";
  SmallVector<uint8_t> GetPc =
      assembleSingleInst("s_get_pc_i64 " + Pair, LS);
  if (GetPc.empty())
    return {};

  uint64_t PcBase = FromOffset + GetPc.size();
  uint64_t Delta = TargetOffset - PcBase;
  std::string Asm = "s_get_pc_i64 " + Pair + "\n" +
                    "s_add_nc_u64 " + Pair + ", " + Pair + ", 0x" +
                    utohexstr(Delta) + "\n" + "s_set_pc_i64 " + Pair +
                    "\n";
  SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, LS);
  if (Bytes.empty() || Bytes.size() > LongSetPcMaxBytes) {
    log() << "hotswap: error: set-PC long edge at 0x"
          << utohexstr(FromOffset) << " -> 0x" << utohexstr(TargetOffset)
          << " encoded to " << Bytes.size() << " bytes (max "
          << LongSetPcMaxBytes << ")\n";
    return {};
  }
  return Bytes;
}

static SmallVector<uint8_t>
encodeSetPcRelay(const LLVMState &LS, uint64_t CapturedPc,
                 uint64_t TargetOffset, unsigned SgprBase) {
  std::string Pair =
      SgprBase == Gfx1250VccLoIndex
          ? "vcc"
          : "s[" + std::to_string(SgprBase) + ":" +
                std::to_string(SgprBase + 1) + "]";
  uint64_t Delta = TargetOffset - CapturedPc;
  std::string Asm = "s_add_nc_u64 " + Pair + ", " + Pair + ", 0x" +
                    utohexstr(Delta) + "\n" + "s_set_pc_i64 " + Pair +
                    "\n";
  SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, LS);
  if (Bytes.empty() || Bytes.size() > LongSetPcRelayMaxBytes) {
    log() << "hotswap: error: set-PC relay to 0x" << utohexstr(TargetOffset)
          << " encoded to " << Bytes.size() << " bytes (max "
          << LongSetPcRelayMaxBytes << ")\n";
    return {};
  }
  return Bytes;
}

// AMDGPU C/Fast and SI shader/Gfx functions have different scalar return and
// callee-save sets. Only s44-s47 and s56-s63 are neither scalar return slots
// nor callee-saved under any of those conventions. Restrict ABI-agnostic
// post-link scratch reuse to that intersection.
static bool isHighCallerClobberedSgprPair(unsigned Pair) {
  constexpr std::pair<unsigned, unsigned> Ranges[] = {
      {44, 48}, {56, 64}};
  return llvm::any_of(Ranges, [&](const auto &Range) {
    return Pair >= Range.first && Pair + 1 < Range.second;
  });
}

static bool collectTouchedNumberedSgprs(ArrayRef<uint8_t> Bytes,
                                        const LLVMState &LS,
                                        unsigned NumberedSgprLimit,
                                        BitVector &Touched) {
  Touched = BitVector(NumberedSgprLimit);
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes.data(), Bytes.size(), LS, Decoded))
    return false;

  const MCRegisterInfo &MRI = *LS.MRI;
  SmallVector<MCRegister> NumberedSgprs(NumberedSgprLimit);
  for (unsigned Reg = 1, End = MRI.getNumRegs(); Reg < End; ++Reg) {
    StringRef Name = MRI.getName(Reg);
    if (!Name.consume_front("SGPR") || Name.contains('_'))
      if (Name == "VCC_LO" && Gfx1250VccLoIndex < NumberedSgprLimit)
        NumberedSgprs[Gfx1250VccLoIndex] = MCRegister(Reg);
      else if (Name == "VCC_HI" &&
               Gfx1250VccLoIndex + 1 < NumberedSgprLimit)
        NumberedSgprs[Gfx1250VccLoIndex + 1] = MCRegister(Reg);
      else
        continue;
    else {
      unsigned Index = 0;
      if (!Name.getAsInteger(10, Index) && Index < NumberedSgprLimit)
        NumberedSgprs[Index] = MCRegister(Reg);
    }
  }
  if (!llvm::all_of(NumberedSgprs,
                    [](MCRegister Reg) { return Reg.isValid(); }))
    return false;

  auto Mark = [&](MCRegister Reg) {
    if (!Reg.isValid())
      return;
    for (unsigned I = 0; I < NumberedSgprLimit; ++I)
      if (MRI.regsOverlap(Reg.id(), NumberedSgprs[I].id()))
        Touched.set(I);
  };
  for (const InternalDecodedInst &DI : Decoded) {
    if (DI.Mnemonic == "<unknown>")
      return false;
    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    for (const MCOperand &Op : DI.Inst)
      if (Op.isReg() && Op.getReg())
        Mark(MCRegister(Op.getReg()));
    for (MCPhysReg Reg : Desc.implicit_uses())
      Mark(MCRegister(Reg));
    for (MCPhysReg Reg : Desc.implicit_defs())
      Mark(MCRegister(Reg));
  }
  return true;
}

static std::optional<unsigned>
parseNumberedSgprName(StringRef Name) {
  if (!Name.consume_front("SGPR") || Name.empty() || Name.contains('_'))
    return std::nullopt;
  unsigned Index = 0;
  if (Name.getAsInteger(10, Index))
    return std::nullopt;
  return Index;
}

static std::optional<unsigned>
parseNumberedSgprPairBase(MCRegister Reg, const MCRegisterInfo &MRI) {
  if (!Reg.isValid())
    return std::nullopt;
  auto [LoName, HiName] = StringRef(MRI.getName(Reg)).split('_');
  std::optional<unsigned> Lo = parseNumberedSgprName(LoName);
  std::optional<unsigned> Hi = parseNumberedSgprName(HiName);
  if (!Lo || !Hi || *Hi != *Lo + 1)
    return std::nullopt;
  return Lo;
}

std::optional<uint64_t> resolveStaticSetPcFarBranchTarget(
    ArrayRef<InternalDecodedInst> Function, size_t SetPcIndex,
    ArrayRef<uint8_t> Text, const LLVMState &LS) {
  if (SetPcIndex < 3 || SetPcIndex >= Function.size() || !LS.MRI)
    return std::nullopt;

  const InternalDecodedInst &GetPc = Function[SetPcIndex - 3];
  const InternalDecodedInst &AddLo = Function[SetPcIndex - 2];
  const InternalDecodedInst &AddHi = Function[SetPcIndex - 1];
  const InternalDecodedInst &SetPc = Function[SetPcIndex];
  if (GetPc.Mnemonic != "s_get_pc_i64" ||
      AddLo.Mnemonic != "s_add_co_u32" ||
      AddHi.Mnemonic != "s_add_co_ci_u32" ||
      SetPc.Mnemonic != "s_set_pc_i64")
    return std::nullopt;

  auto IsContiguous = [](const InternalDecodedInst &L,
                         const InternalDecodedInst &R) {
    return L.Offset <= std::numeric_limits<uint64_t>::max() - L.Size &&
           L.Offset + L.Size == R.Offset;
  };
  if (!IsContiguous(GetPc, AddLo) || !IsContiguous(AddLo, AddHi) ||
      !IsContiguous(AddHi, SetPc))
    return std::nullopt;

  if (GetPc.Inst.getNumOperands() < 1 ||
      SetPc.Inst.getNumOperands() < 1 ||
      AddLo.Inst.getNumOperands() < 3 ||
      AddHi.Inst.getNumOperands() < 3)
    return std::nullopt;
  const MCOperand &GetPairOp = GetPc.Inst.getOperand(0);
  const MCOperand &SetPairOp = SetPc.Inst.getOperand(0);
  if (!GetPairOp.isReg() || !GetPairOp.getReg() || !SetPairOp.isReg() ||
      SetPairOp.getReg() != GetPairOp.getReg())
    return std::nullopt;

  std::optional<unsigned> PairBase = parseNumberedSgprPairBase(
      MCRegister(GetPairOp.getReg()), *LS.MRI);
  if (!PairBase)
    return std::nullopt;
  auto IsScalarReg = [&](const MCOperand &Op, unsigned Expected) {
    if (!Op.isReg() || !Op.getReg())
      return false;
    return parseNumberedSgprName(
               StringRef(LS.MRI->getName(Op.getReg()))) == Expected;
  };
  if (!IsScalarReg(AddLo.Inst.getOperand(0), *PairBase) ||
      !IsScalarReg(AddLo.Inst.getOperand(1), *PairBase) ||
      !IsScalarReg(AddHi.Inst.getOperand(0), *PairBase + 1) ||
      !IsScalarReg(AddHi.Inst.getOperand(1), *PairBase + 1))
    return std::nullopt;

  auto ReadLiteral = [&](const InternalDecodedInst &DI,
                         const MCOperand &Op) -> std::optional<uint32_t> {
    if (Op.isImm())
      return static_cast<uint32_t>(Op.getImm());
    // gfx1250 represents a decoded literal source as a target-specific
    // lit(...) MCExpr. Recover the already-resolved value from the exact
    // 8-byte scalar instruction instead of retaining/dereferencing that
    // target-private expression across a multi-million-instruction decode.
    if (!Op.isExpr() || DI.Size != 2 * MinInstSize ||
        DI.Offset > Text.size() || DI.Size > Text.size() - DI.Offset)
      return std::nullopt;
    return support::endian::read32le(Text.data() + DI.Offset + MinInstSize);
  };
  std::optional<uint32_t> LoLiteral =
      ReadLiteral(AddLo, AddLo.Inst.getOperand(2));
  std::optional<uint32_t> HiLiteral =
      ReadLiteral(AddHi, AddHi.Inst.getOperand(2));
  if (!LoLiteral || !HiLiteral)
    return std::nullopt;

  const uint64_t DeltaBits =
      *LoLiteral | (static_cast<uint64_t>(*HiLiteral) << 32);
  if (GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size)
    return std::nullopt;
  const uint64_t CapturedPc = GetPc.Offset + GetPc.Size;
  uint64_t Target = 0;
  if ((DeltaBits >> 63) == 0) {
    if (CapturedPc > std::numeric_limits<uint64_t>::max() - DeltaBits)
      return std::nullopt;
    Target = CapturedPc + DeltaBits;
  } else {
    const uint64_t Magnitude = ~DeltaBits + 1;
    if (CapturedPc < Magnitude)
      return std::nullopt;
    Target = CapturedPc - Magnitude;
  }

  auto TargetIt = llvm::lower_bound(
      Function, Target, [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  if (TargetIt == Function.end() || TargetIt->Offset != Target)
    return std::nullopt;
  return Target;
}

/// Find an aligned numbered SGPR pair whose current value is dead after
/// InstOffset on every path in the owning function. Calls and unknown control
/// flow consume every still-live candidate conservatively. This is only the
/// fallback for a function with no globally unused pair, so the per-candidate
/// DFS remains rare and function-local.
static std::optional<unsigned> findDeadSgprPairAfter(
    const PatchContext &Ctx, uint64_t InstOffset,
    std::vector<InternalDecodedInst>::const_iterator FunctionFirst,
    std::vector<InternalDecodedInst>::const_iterator FunctionAfter,
    unsigned NumberedSgprLimit, const BitVector &ExcludedSgprs,
    bool VccOnly) {
  if (FunctionFirst == FunctionAfter || NumberedSgprLimit < 2)
    return std::nullopt;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  SmallVector<MCRegister> NumberedSgprs(NumberedSgprLimit);
  MCRegister StandardLinkPair;
  for (unsigned Reg = 1, End = MRI.getNumRegs(); Reg < End; ++Reg) {
    StringRef Name = MRI.getName(Reg);
    if (Name == "SGPR30_SGPR31")
      StandardLinkPair = MCRegister(Reg);
    if (!Name.consume_front("SGPR") || Name.contains('_')) {
      if (Name == "VCC_LO" && Gfx1250VccLoIndex < NumberedSgprLimit)
        NumberedSgprs[Gfx1250VccLoIndex] = MCRegister(Reg);
      else if (Name == "VCC_HI" &&
               Gfx1250VccLoIndex + 1 < NumberedSgprLimit)
        NumberedSgprs[Gfx1250VccLoIndex + 1] = MCRegister(Reg);
      continue;
    }
    unsigned Index = 0;
    if (!Name.getAsInteger(10, Index) && Index < NumberedSgprLimit)
      NumberedSgprs[Index] = MCRegister(Reg);
  }
  if (!llvm::all_of(NumberedSgprs,
                    [](MCRegister Reg) { return Reg.isValid(); }))
    return std::nullopt;

  const unsigned Count = FunctionAfter - FunctionFirst;
  ArrayRef<InternalDecodedInst> Function(&*FunctionFirst, Count);
  DenseMap<uint64_t, unsigned> OffsetToLocal;
  for (unsigned I = 0; I < Count; ++I)
    OffsetToLocal.try_emplace(FunctionFirst[I].Offset, I);
  auto Site = OffsetToLocal.find(InstOffset);
  if (Site == OffsetToLocal.end() || Site->second + 1 >= Count)
    return std::nullopt;

  auto AddSuccessor = [&](SmallVectorImpl<std::pair<unsigned, uint8_t>> &Work,
                          unsigned Index, uint8_t LiveMask) {
    if (LiveMask)
      Work.emplace_back(Index, LiveMask);
  };

  unsigned FirstPair = (NumberedSgprLimit - 2) & ~1u;
  for (int Pair = static_cast<int>(FirstPair); Pair >= 0; Pair -= 2) {
    const bool Eligible = VccOnly ? Pair == Gfx1250VccLoIndex
                                  : isHighCallerClobberedSgprPair(Pair);
    if (!Eligible ||
        ExcludedSgprs.test(Pair) || ExcludedSgprs.test(Pair + 1))
      continue;
    MCRegister Lo = NumberedSgprs[Pair];
    MCRegister Hi = NumberedSgprs[Pair + 1];
    SmallVector<std::pair<unsigned, uint8_t>, 32> Worklist;
    std::vector<uint8_t> Seen(Count * 4);
    Worklist.emplace_back(Site->second + 1, uint8_t{3});
    bool Live = false;

    for (size_t Next = 0; Next < Worklist.size() && !Live; ++Next) {
      unsigned I = Worklist[Next].first;
      uint8_t LiveMask = Worklist[Next].second;
      if (I >= Count) {
        Live = true;
        break;
      }
      uint8_t &WasSeen = Seen[I * 4 + LiveMask];
      if (WasSeen)
        continue;
      WasSeen = 1;

      const InternalDecodedInst &DI = FunctionFirst[I];
      if (DI.Mnemonic == "<unknown>") {
        Live = true;
        break;
      }
      const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
      const bool IsCall = Ctx.LS.MIA ? Ctx.LS.MIA->isCall(DI.Inst)
                                     : Desc.isCall();

      auto RegMask = [&](MCRegister Reg, bool IsImplicit = false) {
        // GFX1250 is fixed wave32. Its e32 predicate/carry instructions use
        // the descriptor's implicit VCC register, even though the hardware
        // operand is VCC_LO. Keep explicit VCC operands 64-bit so scalar
        // instructions such as s_mov_b64 still protect both halves.
        if (VccOnly && IsImplicit && Reg.isValid() &&
            StringRef(MRI.getName(Reg)) == "VCC")
          Reg = Lo;
        uint8_t Mask = 0;
        if (Reg.isValid() && MRI.regsOverlap(Reg.id(), Lo.id()))
          Mask |= 1;
        if (Reg.isValid() && MRI.regsOverlap(Reg.id(), Hi.id()))
          Mask |= 2;
        return Mask;
      };

      uint8_t Uses = 0;
      uint8_t Defs = 0;
      unsigned NumDefs =
          std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
      for (unsigned OpIdx = 0, E = DI.Inst.getNumOperands(); OpIdx < E;
           ++OpIdx) {
        const MCOperand &Op = DI.Inst.getOperand(OpIdx);
        if (!Op.isReg() || !Op.getReg())
          continue;
        uint8_t Mask = RegMask(MCRegister(Op.getReg()));
        if (OpIdx < NumDefs)
          Defs |= Mask;
        else
          Uses |= Mask;
      }
      for (MCPhysReg Reg : Desc.implicit_uses())
        Uses |= RegMask(MCRegister(Reg), /*IsImplicit=*/true);
      for (MCPhysReg Reg : Desc.implicit_defs())
        Defs |= RegMask(MCRegister(Reg), /*IsImplicit=*/true);

      if (Uses & LiveMask) {
        Live = true;
        break;
      }
      LiveMask &= ~Defs;
      // High caller-clobbered SGPRs and VCC are outside the scalar
      // argument/return ranges, so a call kills them. Any explicit call-target
      // use was handled by Uses above.
      if (IsCall)
        LiveMask = 0;
      if (!LiveMask)
        continue;

      const bool IsReturn = Ctx.LS.MIA ? Ctx.LS.MIA->isReturn(DI.Inst)
                                       : Desc.isReturn();
      const bool IsStandardSetPcReturn =
          DI.Mnemonic == "s_set_pc_i64" && StandardLinkPair.isValid() &&
          DI.Inst.getNumOperands() != 0 &&
          DI.Inst.getOperand(0).isReg() &&
          DI.Inst.getOperand(0).getReg() == StandardLinkPair.id();
      if (IsReturn || IsStandardSetPcReturn || DI.Mnemonic == "s_endpgm")
        continue;
      if (DI.Mnemonic == "s_set_pc_i64") {
        std::optional<uint64_t> Target =
            resolveStaticSetPcFarBranchTarget(
                Function, I, ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize),
                Ctx.LS);
        auto TargetIt = Target ? OffsetToLocal.find(*Target)
                               : OffsetToLocal.end();
        if (TargetIt == OffsetToLocal.end()) {
          Live = true;
          break;
        }
        AddSuccessor(Worklist, TargetIt->second, LiveMask);
        continue;
      }

      const bool HasFallthrough = I + 1 < Count;
      const bool IsBranch = Ctx.LS.MIA ? Ctx.LS.MIA->isBranch(DI.Inst)
                                       : Desc.isBranch();
      if (IsBranch) {
        const bool IsIndirect =
            Ctx.LS.MIA ? Ctx.LS.MIA->isIndirectBranch(DI.Inst)
                        : Desc.isIndirectBranch();
        const bool IsConditional =
            Ctx.LS.MIA ? Ctx.LS.MIA->isConditionalBranch(DI.Inst)
                        : Desc.isConditionalBranch();
        const bool IsUnconditional =
            Ctx.LS.MIA ? Ctx.LS.MIA->isUnconditionalBranch(DI.Inst)
                        : Desc.isUnconditionalBranch();
        if (IsIndirect || !Ctx.LS.MIA) {
          Live = true;
          break;
        }

        uint64_t Target = 0;
        if (!Ctx.LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target)) {
          Live = true;
          break;
        }
        auto TargetIt = OffsetToLocal.find(Target);
        if (TargetIt != OffsetToLocal.end())
          AddSuccessor(Worklist, TargetIt->second, LiveMask);
        // A known direct target outside the function is an exit.
        if (IsConditional && HasFallthrough)
          AddSuccessor(Worklist, I + 1, LiveMask);
        else if (!IsConditional && !IsUnconditional) {
          Live = true;
          break;
        }
        continue;
      }

      if (Desc.isTerminator())
        continue;
      if (Ctx.LS.MIA &&
          Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, MRI)) {
        Live = true;
        break;
      }
      if (!HasFallthrough) {
        Live = true;
        break;
      }
      AddSuccessor(Worklist, I + 1, LiveMask);
    }

    if (!Live) {
      log() << "hotswap: safe far return: reusing site-dead "
            << (VccOnly ? "vcc" : "caller-clobbered s[" +
                                       std::to_string(Pair) + ":" +
                                       std::to_string(Pair + 1) + "]")
            << " after 0x" << utohexstr(InstOffset) << "\n";
      return static_cast<unsigned>(Pair);
    }
  }
  return std::nullopt;
}

static std::optional<SmallVector<uint8_t>>
buildSafeFarReturn(PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
                   uint64_t BackStart, ArrayRef<uint8_t> Replacement,
                   unsigned &OutScratchPair,
                   unsigned &OutRequiredSgprs,
                   std::string &OutKernelName) {
  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset);
  if (!FunctionRange) {
    log() << "hotswap: error: safe far return: no function owns site 0x"
          << utohexstr(InstOffset) << "\n";
    return std::nullopt;
  }

  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(InstOffset + Ctx.Elf.textAddr());
  const bool HasKernelDescriptor = !KernelName.empty();
  std::optional<unsigned> TotalSgprCount;
  if (HasKernelDescriptor)
    TotalSgprCount = Ctx.Elf.getKernelSgprCount(KernelName);
  if (HasKernelDescriptor && !TotalSgprCount) {
    log() << "hotswap: error: safe far return: invalid SGPR count for kernel "
          << KernelName << "\n";
    return std::nullopt;
  }
  if (!HasKernelDescriptor)
    KernelName = "device function at 0x" + utohexstr(FunctionRange->Begin);

  BitVector ReplacementTouchedSgprs;
  if (!collectTouchedNumberedSgprs(Replacement, Ctx.LS,
                                   Gfx1250SgprSlotsWithVcc,
                                   ReplacementTouchedSgprs)) {
    log() << "hotswap: error: safe far return: cannot prove replacement "
             "SGPR usage at 0x"
          << utohexstr(InstOffset) << "\n";
    return std::nullopt;
  }

  auto FunctionFirst = llvm::lower_bound(
      Ctx.Decoded, FunctionRange->Begin,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  auto FunctionAfter = llvm::lower_bound(
      Ctx.Decoded, FunctionRange->End,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });

  // LLVM metadata counts two implicit SGPRs when a kernel uses VCC, but those
  // are not part of the numbered s0-s105 register space. Inspect the owning
  // function so scratch allocation starts above numbered SGPRs rather than
  // above the metadata total. If VCC is only used by a callee, retaining the
  // total as the allocation base is conservative; the call flag below keeps
  // two implicit slots in the updated metadata.
  bool UsesVcc = false;
  bool HasCall = false;

  auto IsVcc = [&](MCRegister Reg) {
    return Reg.isValid() &&
           StringRef(Ctx.LS.MRI->getName(Reg)).starts_with("VCC");
  };
  auto InstUsesVcc = [&](const InternalDecodedInst &DI) {
    for (const MCOperand &Op : DI.Inst)
      if (Op.isReg() && Op.getReg() && IsVcc(MCRegister(Op.getReg())))
        return true;
    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    for (MCPhysReg Reg : Desc.implicit_uses())
      if (IsVcc(MCRegister(Reg)))
        return true;
    for (MCPhysReg Reg : Desc.implicit_defs())
      if (IsVcc(MCRegister(Reg)))
        return true;
    return false;
  };

  for (const InternalDecodedInst &DI :
       llvm::make_range(FunctionFirst, FunctionAfter)) {
    UsesVcc |= InstUsesVcc(DI);
    HasCall |= Ctx.LS.MIA && Ctx.LS.MIA->isCall(DI.Inst);
  }

  constexpr unsigned VccSgprs = 2;
  const bool KernelMetadataReservesVcc =
      HasKernelDescriptor && *TotalSgprCount > Gfx1250VccLoIndex;
  const bool HasVccAllocation =
      UsesVcc || KernelMetadataReservesVcc ||
      (!HasKernelDescriptor && Ctx.AllKernelsReserveVcc);
  if (HasKernelDescriptor && (UsesVcc || KernelMetadataReservesVcc) &&
      *TotalSgprCount < VccSgprs) {
    log() << "hotswap: error: safe far return: VCC-using kernel " << KernelName
          << " has invalid SGPR count " << *TotalSgprCount << "\n";
    return std::nullopt;
  }
  unsigned NumberedSgprCount =
      HasKernelDescriptor
          ? *TotalSgprCount -
                (KernelMetadataReservesVcc ? VccSgprs : 0)
                          : Ctx.Config.MaxSgprs;
  if (NumberedSgprCount > Ctx.Config.MaxSgprs) {
    log() << "hotswap: error: safe far return: numbered SGPR count for kernel "
          << KernelName << " exceeds " << Ctx.Config.MaxSgprs << "\n";
    return std::nullopt;
  }

  // s_get_pc_i64/s_set_pc_i64 require an aligned pair. A kernel can grow its
  // descriptor above the original numbered allocation. Internal functions
  // cannot: they may only reuse a high caller-clobbered pair that is absent
  // from the whole function or dead at this site. Callee-saved stripes are
  // never considered, so this post-link rewrite creates no save obligation.
  unsigned ScratchPair = HasKernelDescriptor
                             ? (NumberedSgprCount + 1) & ~1u
                             : Ctx.Config.MaxSgprs;
  while (ScratchPair + 1 < Ctx.Config.MaxSgprs &&
         (ReplacementTouchedSgprs.test(ScratchPair) ||
          ReplacementTouchedSgprs.test(ScratchPair + 1)))
    ScratchPair += 2;
  if (!HasKernelDescriptor || ScratchPair + 1 >= Ctx.Config.MaxSgprs) {
    const std::pair<uint64_t, uint64_t> FunctionKey{FunctionRange->Begin,
                                                    FunctionRange->End};
    auto CacheIt = Ctx.FunctionSgprUsageCache.find(FunctionKey);
    if (CacheIt == Ctx.FunctionSgprUsageCache.end()) {
      FunctionSgprUsage Usage;
      const MCRegisterInfo &MRI = *Ctx.LS.MRI;
      MCRegister StandardLinkPair;
      for (unsigned Reg = 1, End = MRI.getNumRegs(); Reg < End; ++Reg)
        if (StringRef(MRI.getName(Reg)) == "SGPR30_SGPR31") {
          StandardLinkPair = MCRegister(Reg);
          break;
        }
      bool SawStandardReturn = false;
      bool StandardReturningAbi = Ctx.LS.MIA && StandardLinkPair.isValid();
      SmallVector<MCRegister> NumberedSgprs(NumberedSgprCount);
      for (unsigned Reg = 1, End = MRI.getNumRegs(); Reg < End; ++Reg) {
        StringRef Name = MRI.getName(Reg);
        if (!Name.consume_front("SGPR") || Name.contains('_'))
          continue;
        unsigned Index = 0;
        if (!Name.getAsInteger(10, Index) && Index < NumberedSgprCount)
          NumberedSgprs[Index] = MCRegister(Reg);
      }

      Usage.Used = BitVector(NumberedSgprCount);
      bool CanProveUnused = llvm::all_of(NumberedSgprs, [](MCRegister Reg) {
        return Reg.isValid();
      });
      for (const InternalDecodedInst &DI :
           llvm::make_range(FunctionFirst, FunctionAfter)) {
        if (DI.Mnemonic == "<unknown>") {
          CanProveUnused = false;
          StandardReturningAbi = false;
          break;
        }

        const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
        const bool UsesStandardLink =
            DI.Inst.getNumOperands() != 0 &&
            DI.Inst.getOperand(0).isReg() &&
            DI.Inst.getOperand(0).getReg() == StandardLinkPair.id();
        const bool IsCall = Ctx.LS.MIA && Ctx.LS.MIA->isCall(DI.Inst);
        const bool IsReturn =
            (Ctx.LS.MIA && Ctx.LS.MIA->isReturn(DI.Inst)) ||
            (DI.Mnemonic == "s_set_pc_i64" && UsesStandardLink);
        if (IsCall || IsReturn) {
          StandardReturningAbi &= UsesStandardLink;
          SawStandardReturn |= IsReturn && UsesStandardLink;
        }

        auto MarkUsed = [&](MCRegister Reg) {
          if (!Reg.isValid())
            return;
          for (unsigned I = 0; I < NumberedSgprCount; ++I)
            if (MRI.regsOverlap(Reg.id(), NumberedSgprs[I].id())) {
              Usage.Used.set(I);
            }
        };
        for (const MCOperand &Op : DI.Inst)
          if (Op.isReg() && Op.getReg())
            MarkUsed(MCRegister(Op.getReg()));
        for (MCPhysReg Reg : Desc.implicit_uses())
          MarkUsed(MCRegister(Reg));
        for (MCPhysReg Reg : Desc.implicit_defs())
          MarkUsed(MCRegister(Reg));
      }

      if (!HasKernelDescriptor) {
        int HighestUsed = Usage.Used.find_last();
        Usage.ExistingLimit =
            HighestUsed < 0 ? 0 : static_cast<unsigned>(HighestUsed + 1);
      } else {
        Usage.ExistingLimit = NumberedSgprCount;
      }
      Usage.Complete = CanProveUnused;
      Usage.StandardReturningAbi =
          StandardReturningAbi && SawStandardReturn;
      if (CanProveUnused && Usage.ExistingLimit >= 2) {
        unsigned Pair = (Usage.ExistingLimit - 2) & ~1u;
        for (;;) {
          if (isHighCallerClobberedSgprPair(Pair) &&
              !Usage.Used.test(Pair) && !Usage.Used.test(Pair + 1)) {
            Usage.UnusedCallerClobberedPair = static_cast<int>(Pair);
            break;
          }
          if (Pair < 2)
            break;
          Pair -= 2;
        }
      }
      auto Inserted = Ctx.FunctionSgprUsageCache.try_emplace(
          FunctionKey, std::move(Usage));
      CacheIt = Inserted.first;
      if (CacheIt->second.UnusedCallerClobberedPair >= 0)
        log() << "hotswap: safe far return: reusing globally unused s["
              << CacheIt->second.UnusedCallerClobberedPair << ':'
              << CacheIt->second.UnusedCallerClobberedPair + 1 << "] in "
              << KernelName << "\n";
    }
    const FunctionSgprUsage &Usage = CacheIt->second;
    const int CachedPair = Usage.UnusedCallerClobberedPair;
    const bool MayReuseInternalScratch =
        HasKernelDescriptor || Usage.StandardReturningAbi;
    if (MayReuseInternalScratch && CachedPair >= 0 &&
        !ReplacementTouchedSgprs.test(CachedPair) &&
        !ReplacementTouchedSgprs.test(CachedPair + 1)) {
      ScratchPair = static_cast<unsigned>(CachedPair);
    } else {
      unsigned SiteLivenessLimit =
          HasKernelDescriptor
              ? NumberedSgprCount
              : (Usage.Complete
                     ? std::min(
                           Ctx.Config.MaxSgprs,
                           static_cast<unsigned>(alignTo(Usage.ExistingLimit,
                                                         uint64_t{2})))
                     : 0);
      std::optional<unsigned> DeadPair;
      if (MayReuseInternalScratch)
        DeadPair = findDeadSgprPairAfter(
            Ctx, InstOffset, FunctionFirst, FunctionAfter,
            SiteLivenessLimit, ReplacementTouchedSgprs,
            /*VccOnly=*/false);
      if (!DeadPair && MayReuseInternalScratch && HasVccAllocation)
        DeadPair = findDeadSgprPairAfter(
            Ctx, InstOffset, FunctionFirst, FunctionAfter,
            Gfx1250SgprSlotsWithVcc, ReplacementTouchedSgprs,
            /*VccOnly=*/true);
      if (!DeadPair) {
        log() << "hotswap: safe far return unavailable: " << KernelName
              << " has no allocatable/dead high caller-clobbered SGPR pair "
                 "or site-dead VCC below s"
              << Ctx.Config.MaxSgprs << "\n";
        return std::nullopt;
      }
      ScratchPair = *DeadPair;
    }
  }

  uint64_t ReturnTo = InstOffset + InstSize;
  SmallVector<uint8_t> Bytes =
      encodeSetPcLongBranch(Ctx.LS, BackStart, ReturnTo, ScratchPair);
  if (Bytes.empty())
    return std::nullopt;

  if (HasKernelDescriptor) {
    unsigned RequiredNumberedSgprs =
        ScratchPair == Gfx1250VccLoIndex
            ? NumberedSgprCount
            : std::max(NumberedSgprCount, ScratchPair + 2);
    unsigned PreservedImplicitSgprs =
        (KernelMetadataReservesVcc || UsesVcc || HasCall) ? VccSgprs : 0;
    unsigned RequiredSgprs = RequiredNumberedSgprs + PreservedImplicitSgprs;
    if (RequiredSgprs < *TotalSgprCount) {
      log() << "hotswap: error: safe far return: SGPR accounting underflow for "
            << KernelName << "\n";
      return std::nullopt;
    }
    OutRequiredSgprs = RequiredSgprs;
    OutKernelName = KernelName;
  } else {
    OutRequiredSgprs = 0;
    OutKernelName.clear();
  }
  OutScratchPair = ScratchPair;
  log() << "hotswap: safe far return at 0x" << utohexstr(InstOffset)
        << " via ";
  if (ScratchPair == Gfx1250VccLoIndex)
    log() << "vcc\n";
  else
    log() << "s[" << ScratchPair << ':' << ScratchPair + 1 << "]\n";
  return Bytes;
}

/// Queue a deferred trampoline for [\p InstOffset, +\p InstSize) with
/// \p Replacement as its body; fixupTrampolineBranches fills in the edges once
/// the pool layout is known. A site beyond s_branch reach of the appended pool
/// uses an SGPR pair with an SCC-neutral get-PC/add/set-PC sequence on both
/// edges. Finalization grows the forward site to fit that sequence by merging
/// adjacent rewritten sites or relocating verified-safe whole instructions.
/// Optional far rewrites decline rather than introduce s_add_pc_i64.
[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    ArrayRef<uint8_t> Replacement,
                                    bool AllowSafeFarReturn) {
  // This trampoline lands at the appended pool base and after every trampoline
  // already queued -- later ones are appended behind it and cannot shift it,
  // and fixupTrampolineBranches walks the same list in the same order -- so its
  // final pool offset (relative to .text) is known exactly now.
  uint64_t PoolStart = Ctx.PoolBaseOffset;
  for (const Trampoline &Prev : Ctx.OutTrampolines) {
    std::optional<uint64_t> Next = checkedAddUint64(
        PoolStart, Prev.Bytes.size(), "provisional trampoline pool cursor");
    if (!Next)
      return false;
    PoolStart = *Next;
  }

  // An s_branch encodes To - From as a signed simm16 dword field, in range iff
  // (To - From - MinInstSize) / MinInstSize fits [BranchOffsetMin,
  // BranchOffsetMax] (see LLVMState::encodeSBranch). Test both edges with the
  // short branch-back slot; the branch-back (pool tail -> site) is the farther
  // of the two. Go long only when a short branch cannot reach.
  std::optional<uint64_t> ShortBackFrom = checkedAddUint64(
      PoolStart, Replacement.size(), "provisional trampoline body end");
  std::optional<uint64_t> ReturnTo = checkedAddUint64(
      InstOffset, InstSize, "provisional trampoline return offset");
  if (!ShortBackFrom || !ReturnTo)
    return false;
  const bool Far = !(isSBranchReachable(InstOffset, PoolStart) &&
                     isSBranchReachable(*ShortBackFrom, *ReturnTo));

  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
  T.SiteFootprint = InstSize;
  T.AllowSafeFarReturn = AllowSafeFarReturn;
  T.Bytes.insert(T.Bytes.end(), Replacement.begin(), Replacement.end());

  if (Far) {
    // HSV-009 prohibits s_add_pc_i64 on gfx1250 A0 in either direction.
    // Optional transformations decline; required transformations reserve a
    // scratch pair for two set-PC edges.
    if (!AllowSafeFarReturn) {
      log() << "hotswap: far trampoline at 0x" << utohexstr(InstOffset)
            << " declined (s_add_pc_i64 prohibited on gfx1250 A0, HSV-009); "
               "site left unpatched\n";
      return false;
    }

    unsigned ScratchPair = 0;
    unsigned RequiredSgprs = 0;
    std::string ScratchKernelName;
    std::optional<SmallVector<uint8_t>> Back = buildSafeFarReturn(
        Ctx, InstOffset, InstSize, *ShortBackFrom, Replacement, ScratchPair,
        RequiredSgprs, ScratchKernelName);
    if (!Back)
      return false;
    T.BackReserve = LongSetPcMaxBytes;
    T.Bytes.append(T.BackReserve, uint8_t{0});
    T.Long = true;
    T.LongBranchSgprBase = ScratchPair;
    T.LongBranchRequiredSgprs = RequiredSgprs;
    T.LongBranchKernelName = std::move(ScratchKernelName);
    Ctx.OutTrampolines.emplace_back(std::move(T));
    return true;
  }
  {
    // Reserve the short branch-back slot; fixupTrampolineBranches fills it in.
    T.BackReserve = MinInstSize;
    T.Bytes.insert(T.Bytes.end(), T.BackReserve, uint8_t{0});
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
                                       ArrayRef<uint8_t> Replacement,
                                       bool AllowSafeFarReturn) {
  auto RememberReplacement = [&]() {
    Ctx.ReplacementCodeBySite.insert_or_assign(
        InstOffset,
        SmallVector<uint8_t>(Replacement.begin(), Replacement.end()));
    return true;
  };
  ArrayRef<uint8_t> SledReplacement = Replacement;
  ArrayRef<uint8_t> OriginalTail;
  SmallVector<uint8_t> WaitDscnt0;
  if (AllowSafeFarReturn && InstSize >= 2 * MinInstSize) {
    WaitDscnt0 = assembleSingleInst("s_wait_dscnt 0", Ctx.LS);
    if (WaitDscnt0.size() == MinInstSize &&
        Replacement.size() >= WaitDscnt0.size() &&
        Replacement.take_back(WaitDscnt0.size()) ==
            ArrayRef<uint8_t>(WaitDscnt0)) {
      SledReplacement = Replacement.drop_back(WaitDscnt0.size());
      OriginalTail = WaitDscnt0;
    }
  }

  // Interior sleds belong to this function and cannot deprive another
  // function of its only required-patch storage, so prefer them first.
  uint64_t Needed = SledReplacement.size() + MinInstSize;
  uint64_t ReturnTo = OriginalTail.empty() ? InstOffset + InstSize
                                            : InstOffset + MinInstSize;
  if (NopSled *Sled = findNearestSled(
          Ctx.NopSleds, InstOffset, Needed, SledReplacement.size(), ReturnTo,
          /*AllowTailPadding=*/false)) {
    if (emitToNopSled(Ctx, *Sled, InstOffset, InstSize, SledReplacement,
                      OriginalTail))
      return RememberReplacement();
    log() << "hotswap: emitReplacementCode: NOP sled at offset 0x"
          << utohexstr(Sled->WritePos)
          << " is not branch-reachable after assembly; using trampoline.\n";
  }
  // A proven scratch pair makes the appended set-PC trampoline independent of
  // scarce local padding. Preserve anonymous caves for saturated functions
  // that cannot prove such a pair.
  if (emitToTrampoline(Ctx, InstOffset, InstSize, Replacement,
                       AllowSafeFarReturn))
    return RememberReplacement();

  if (AllowSafeFarReturn) {
    if (NopSled *Sled = findNearestSled(
            Ctx.NopSleds, InstOffset, Needed, SledReplacement.size(), ReturnTo,
            /*AllowTailPadding=*/true)) {
      if (emitToNopSled(Ctx, *Sled, InstOffset, InstSize, SledReplacement,
                        OriginalTail))
        return RememberReplacement();
    }
    if (emitToRoutedTailCave(Ctx, InstOffset, InstSize, SledReplacement,
                             OriginalTail))
      return RememberReplacement();
    log() << "hotswap: error: required replacement at 0x"
          << utohexstr(InstOffset)
          << " has neither a safe set-PC scratch pair nor a reachable code "
             "cave\n";
  }
  return false;
}

static bool looksLikeUnpatchedB0Hazard(StringRef Mnemonic) {
  if (Mnemonic == "tensor_load_to_lds")
    return true;
  return Mnemonic.starts_with("ds_") &&
         (Mnemonic.contains("_2addr") || Mnemonic.contains("_addtid"));
}

/// Convert a required far trampoline back into a short code-cave rewrite when
/// its original site cannot safely grow to hold a set-PC forward edge. The
/// original text has not been changed yet, so this is a transactional fallback.
static bool emitFarSiteToCodeCave(PatchContext &Ctx, const Trampoline &T) {
  if (T.BackReserve > T.Bytes.size())
    return false;

  ArrayRef<uint8_t> Replacement(T.Bytes.data(),
                                T.Bytes.size() - T.BackReserve);
  ArrayRef<uint8_t> SledReplacement = Replacement;
  ArrayRef<uint8_t> OriginalTail;
  SmallVector<uint8_t> WaitDscnt0;
  if (T.OriginalSize >= 2 * MinInstSize) {
    WaitDscnt0 = assembleSingleInst("s_wait_dscnt 0", Ctx.LS);
    if (WaitDscnt0.size() == MinInstSize &&
        Replacement.size() >= WaitDscnt0.size() &&
        Replacement.take_back(WaitDscnt0.size()) ==
            ArrayRef<uint8_t>(WaitDscnt0)) {
      SledReplacement = Replacement.drop_back(WaitDscnt0.size());
      OriginalTail = WaitDscnt0;
    }
  }

  uint64_t Needed = SledReplacement.size() + MinInstSize;
  uint64_t ReturnTo = OriginalTail.empty()
                          ? T.OriginalOffset + T.OriginalSize
                          : T.OriginalOffset + MinInstSize;
  NopSled *Sled = findNearestSled(
      Ctx.NopSleds, T.OriginalOffset, Needed, SledReplacement.size(),
      ReturnTo, /*AllowTailPadding=*/true);
  if (!Sled || !emitToNopSled(Ctx, *Sled, T.OriginalOffset, T.OriginalSize,
                              SledReplacement, OriginalTail)) {
    if (!emitToRoutedTailCave(Ctx, T.OriginalOffset, T.OriginalSize,
                              SledReplacement, OriginalTail))
      return false;
  }

  log() << "hotswap: set-PC forward site 0x"
        << utohexstr(T.OriginalOffset)
        << " used a short code cave because its forward window is unsafe\n";
  return true;
}

/// Reserve a small local relay for a far site whose following instructions
/// cannot be relocated. The original 8-byte slot holds s_get_pc_i64 followed
/// by a short branch. Finalization fills the relay with add/set-PC once the
/// appended trampoline's exact offset is known.
static bool reserveSetPcForwardRelay(PatchContext &Ctx, Trampoline &T,
                                     uint64_t PoolTarget) {
  if (T.OriginalSize < 2 * MinInstSize)
    return false;

  std::optional<uint64_t> CapturedPc = checkedAddUint64(
      T.OriginalOffset, MinInstSize, "set-PC relay captured PC");
  if (!CapturedPc)
    return false;

  SmallVector<uint8_t> MaxRelay = encodeSetPcRelay(
      Ctx.LS, *CapturedPc, PoolTarget, T.LongBranchSgprBase);
  if (MaxRelay.empty())
    return false;
  uint32_t RelayReserve = MaxRelay.size();

  NopSled *Best = nullptr;
  uint64_t BestDist = std::numeric_limits<uint64_t>::max();
  bool BestIsOwner = false;
  for (NopSled &Sled : Ctx.NopSleds) {
    if (!Sled.IsTailPadding)
      continue;
    if (Sled.WritePos > Sled.End || RelayReserve > Sled.End - Sled.WritePos ||
        !isSBranchReachable(*CapturedPc, Sled.WritePos))
      continue;
    bool IsOwner = T.OriginalOffset >= Sled.FunctionStart &&
                   T.OriginalOffset < Sled.FunctionEnd;
    uint64_t Dist = Sled.WritePos > T.OriginalOffset
                        ? Sled.WritePos - T.OriginalOffset
                        : T.OriginalOffset - Sled.WritePos;
    if ((!BestIsOwner && IsOwner) ||
        (BestIsOwner == IsOwner && Dist < BestDist)) {
      Best = &Sled;
      BestDist = Dist;
      BestIsOwner = IsOwner;
    }
  }
  if (!Best)
    return false;

  T.ForwardRelayOffset = Best->WritePos;
  T.ForwardRelayReserve = RelayReserve;
  T.UsesSetPcForward = true;
  T.UsesSetPcRelay = true;
  std::optional<uint64_t> NewWritePos = checkedAddUint64(
      Best->WritePos, RelayReserve, "set-PC relay storage end");
  if (!NewWritePos || *NewWritePos > Best->End)
    return false;
  Best->WritePos = *NewWritePos;
  for (uint64_t Offset = T.ForwardRelayOffset; Offset < Best->WritePos;
       Offset += MinInstSize)
    Ctx.MutatedOffsets.insert(Offset);
  for (uint64_t Offset = T.OriginalOffset;
       Offset < T.OriginalOffset + T.OriginalSize; Offset += MinInstSize)
    Ctx.MutatedOffsets.insert(Offset);
  log() << "hotswap: set-PC forward site 0x" << utohexstr(T.OriginalOffset)
        << " reserved relay at 0x" << utohexstr(T.ForwardRelayOffset) << "\n";
  return true;
}

/// Grow every required far site's forward window until it can hold the
/// set-PC edge. Adjacent trampoline sites are merged; otherwise only whole,
/// unmodified, position-independent instructions are relocated into the body.
/// Any unsafe window is a required-patch failure: gfx1250 A0 must never fall
/// back to s_add_pc_i64.
static bool expandFarForwardSetPc(PatchContext &Ctx) {
  std::vector<Trampoline> &Tramps = Ctx.OutTrampolines;
  if (Tramps.empty())
    return true;

  std::stable_sort(Tramps.begin(), Tramps.end(),
                   [](const Trampoline &L, const Trampoline &R) {
                     return L.OriginalOffset < R.OriginalOffset;
                   });

  DenseMap<uint64_t, size_t> OffToInst;
  for (size_t I = 0, E = Ctx.Decoded.size(); I < E; ++I)
    OffToInst.try_emplace(Ctx.Decoded[I].Offset, I);

  DenseMap<uint64_t, size_t> OffToTramp;
  DenseMap<uint64_t, unsigned> ResumeCounts;
  for (size_t I = 0, E = Tramps.size(); I < E; ++I) {
    auto [It, Inserted] =
        OffToTramp.try_emplace(Tramps[I].OriginalOffset, I);
    if (!Inserted) {
      log() << "hotswap: error: duplicate trampoline site 0x"
            << utohexstr(Tramps[I].OriginalOffset) << " at indices "
            << It->second << " and " << I << "\n";
      return false;
    }
    ++ResumeCounts[Tramps[I].OriginalOffset + Tramps[I].OriginalSize];
  }
  auto ResumeCountAt = [&](uint64_t Offset) {
    return ResumeCounts.lookup(Offset) +
           unsigned(Ctx.ImmediateCaveResumeOffsets.contains(Offset));
  };

  DenseSet<uint64_t> BranchTargets;
  DenseSet<std::pair<uint64_t, uint64_t>> IndirectFunctionRanges;
  if (Ctx.LS.MIA) {
    for (const InternalDecodedInst &DI : Ctx.Decoded) {
      if (Ctx.LS.MIA->isBranch(DI.Inst) &&
          Ctx.LS.MIA->isIndirectBranch(DI.Inst) &&
          !Ctx.LS.MIA->isCall(DI.Inst) &&
          !Ctx.LS.MIA->isReturn(DI.Inst)) {
        if (std::optional<ElfView::FunctionTextRange> Range =
                Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset))
          IndirectFunctionRanges.insert({Range->Begin, Range->End});
      }
      if (!Ctx.LS.MIA->isCall(DI.Inst) &&
          !Ctx.LS.MIA->isUnconditionalBranch(DI.Inst) &&
          !Ctx.LS.MIA->isConditionalBranch(DI.Inst))
        continue;
      uint64_t Target = 0;
      if (Ctx.LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target))
        BranchTargets.insert(Target);
    }
  }

  DenseSet<uint64_t> SymbolTargets;
  for (const ElfView::ELFT::Shdr &Symtab : Ctx.Elf.sections()) {
    if (Symtab.sh_type != ELF::SHT_SYMTAB &&
        Symtab.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ElfView::ELFT::SymRange> Symbols =
        Ctx.Elf.file().symbols(&Symtab);
    if (!Symbols) {
      consumeError(Symbols.takeError());
      continue;
    }
    for (const ElfView::ELFT::Sym &Sym : *Symbols)
      if (Sym.st_shndx == Ctx.Elf.textSectionIndex() &&
          Sym.st_value >= Ctx.Elf.textAddr() &&
          Sym.st_value - Ctx.Elf.textAddr() < Ctx.Elf.textSize())
        SymbolTargets.insert(Sym.st_value - Ctx.Elf.textAddr());
  }

  auto UnsafeToRelocate = [&](const InternalDecodedInst &DI) {
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>" ||
        Ctx.MutatedOffsets.contains(DI.Offset) ||
        looksLikeUnpatchedB0Hazard(DI.Mnemonic))
      return true;
    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    if (Desc.isBranch() || Desc.isCall() || Desc.isReturn() ||
        Desc.isTerminator() || Desc.isTrap() ||
        Desc.mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI))
      return true;
    StringRef M(DI.Mnemonic);
    return M.starts_with("s_delay") || M == "s_clause" ||
           M.contains("get_pc") || M.contains("getpc") ||
           M.contains("set_pc") || M.contains("setpc") ||
           M.contains("add_pc") || M.contains("addpc");
  };
  SmallVector<uint8_t> CanonicalValuDelay = assembleSingleInst(
      "s_delay_alu instid0(VALU_DEP_1)", Ctx.LS);
  auto IsBackwardRelocatableDelay = [&](const InternalDecodedInst &DI) {
    if (DI.Mnemonic != "s_delay_alu" || CanonicalValuDelay.empty() ||
        DI.Size != CanonicalValuDelay.size())
      return false;
    return ArrayRef<uint8_t>(Ctx.Text + DI.Offset, DI.Size) ==
           ArrayRef<uint8_t>(CanonicalValuDelay);
  };

  auto TouchesScratchPair = [&](const InternalDecodedInst &DI,
                                unsigned PairBase) {
    MCRegister Lo;
    MCRegister Hi;
    for (unsigned Reg = 1, End = Ctx.LS.MRI->getNumRegs(); Reg < End; ++Reg) {
      StringRef Name = Ctx.LS.MRI->getName(Reg);
      if (PairBase == Gfx1250VccLoIndex) {
        if (Name == "VCC_LO")
          Lo = MCRegister(Reg);
        else if (Name == "VCC_HI")
          Hi = MCRegister(Reg);
        continue;
      }
      if (!Name.consume_front("SGPR") || Name.contains('_'))
        continue;
      unsigned Index = 0;
      if (Name.getAsInteger(10, Index))
        continue;
      if (Index == PairBase)
        Lo = MCRegister(Reg);
      else if (Index == PairBase + 1)
        Hi = MCRegister(Reg);
    }
    if (!Lo.isValid() || !Hi.isValid())
      return true;
    auto Overlaps = [&](MCRegister Reg) {
      return Reg.isValid() &&
             (Ctx.LS.MRI->regsOverlap(Reg.id(), Lo.id()) ||
              Ctx.LS.MRI->regsOverlap(Reg.id(), Hi.id()));
    };
    for (const MCOperand &Op : DI.Inst)
      if (Op.isReg() && Op.getReg() && Overlaps(MCRegister(Op.getReg())))
        return true;
    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    for (MCPhysReg Reg : Desc.implicit_uses())
      if (Overlaps(MCRegister(Reg)))
        return true;
    for (MCPhysReg Reg : Desc.implicit_defs())
      if (Overlaps(MCRegister(Reg)))
        return true;
    return false;
  };
  auto BytesTouchScratchPair = [&](ArrayRef<uint8_t> Bytes,
                                   unsigned PairBase) {
    BitVector Touched;
    return !collectTouchedNumberedSgprs(Bytes, Ctx.LS,
                                        Gfx1250SgprSlotsWithVcc, Touched) ||
           Touched.test(PairBase) || Touched.test(PairBase + 1);
  };

  std::vector<bool> Removed(Tramps.size());
  uint64_t HighWater = 0;
  uint64_t PoolCursor = Ctx.PoolBaseOffset;
  unsigned Expanded = 0;
  unsigned Merged = 0;

  for (size_t I = 0, E = Tramps.size(); I < E; ++I) {
    Trampoline &T = Tramps[I];
    if (Removed[I])
      continue;
    if (T.BackReserve > T.Bytes.size())
      return false;

    uint64_t TP = PoolCursor;
    uint64_t BodySize = T.Bytes.size() - T.BackReserve;
    if (!T.Long) {
      std::optional<uint64_t> ShortBackFrom = checkedAddUint64(
          TP, BodySize, "short trampoline branch-back offset");
      std::optional<uint64_t> ShortReturnTo = checkedAddUint64(
          T.OriginalOffset, T.OriginalSize,
          "short trampoline return offset");
      if (!ShortBackFrom || !ShortReturnTo)
        return false;
      bool ShortFits =
          isSBranchReachable(T.OriginalOffset, TP) &&
          isSBranchReachable(*ShortBackFrom, *ShortReturnTo);
      if (ShortFits) {
        std::optional<uint64_t> Next = checkedAddUint64(
            PoolCursor, T.Bytes.size(), "short trampoline pool cursor");
        if (!Next)
          return false;
        PoolCursor = *Next;
        continue;
      }

      if (emitFarSiteToCodeCave(Ctx, T)) {
        Removed[I] = true;
        continue;
      }
      if (!T.AllowSafeFarReturn) {
        log() << "hotswap: error: short trampoline at 0x"
              << utohexstr(T.OriginalOffset)
              << " moved out of range after final pool layout and has no "
                 "safe fallback\n";
        return false;
      }

      unsigned ScratchPair = 0;
      unsigned RequiredSgprs = 0;
      std::string ScratchKernelName;
      std::optional<uint64_t> BackStart = checkedAddUint64(
          TP, BodySize, "promoted trampoline body end");
      if (!BackStart)
        return false;
      ArrayRef<uint8_t> Replacement(T.Bytes.data(), BodySize);
      std::optional<SmallVector<uint8_t>> Back = buildSafeFarReturn(
          Ctx, T.OriginalOffset, T.OriginalSize, *BackStart, Replacement,
          ScratchPair, RequiredSgprs, ScratchKernelName);
      if (!Back)
        return false;
      T.Bytes.resize(BodySize);
      T.BackReserve = LongSetPcMaxBytes;
      T.Bytes.append(T.BackReserve, uint8_t{0});
      T.Long = true;
      T.LongBranchSgprBase = ScratchPair;
      T.LongBranchRequiredSgprs = RequiredSgprs;
      T.LongBranchKernelName = std::move(ScratchKernelName);
      log() << "hotswap: promoted final-layout trampoline at 0x"
            << utohexstr(T.OriginalOffset) << " to a set-PC edge\n";
    }

    std::optional<ElfView::FunctionTextRange> FunctionRange =
        Ctx.Elf.findFunctionTextRangeAtOffset(T.OriginalOffset);
    if (!FunctionRange) {
      log() << "hotswap: error: set-PC forward site 0x"
            << utohexstr(T.OriginalOffset) << " has no owning function\n";
      return false;
    }
    const bool HasUnknownIndirectEntry = IndirectFunctionRanges.contains(
        {FunctionRange->Begin, FunctionRange->End});

    SmallVector<uint8_t> OriginalBody(T.Bytes.begin(),
                                      T.Bytes.end() - T.BackReserve);
    SmallVector<uint8_t> Body = OriginalBody;
    SmallVector<uint8_t> MaxForward = encodeSetPcLongBranch(
        Ctx.LS, T.OriginalOffset, TP, T.LongBranchSgprBase);
    if (MaxForward.empty())
      return false;
    uint32_t RequiredForwardBytes = MaxForward.size();
    std::optional<uint64_t> OriginalEnd = checkedAddUint64(
        T.OriginalOffset, T.OriginalSize, "far forward source window end");
    if (!OriginalEnd)
      return false;
    uint64_t Cursor = *OriginalEnd;
    uint32_t Footprint = T.OriginalSize;
    unsigned AllowedResumeCount = 1;
    bool Unsafe = T.OriginalOffset < HighWater ||
                  (HasUnknownIndirectEntry &&
                   Footprint < RequiredForwardBytes);
    SmallVector<size_t, 4> PendingMerges;

    while (!Unsafe && Footprint < RequiredForwardBytes) {
      if (Cursor >= FunctionRange->End || BranchTargets.contains(Cursor) ||
          SymbolTargets.contains(Cursor) ||
          ResumeCountAt(Cursor) > AllowedResumeCount) {
        Unsafe = true;
        break;
      }

      auto TrampIt = OffToTramp.find(Cursor);
      if (TrampIt != OffToTramp.end()) {
        size_t J = TrampIt->second;
        if (J <= I || Removed[J]) {
          Unsafe = true;
          break;
        }
        Trampoline &Next = Tramps[J];
        std::optional<uint64_t> NextEnd = checkedAddUint64(
            Cursor, Next.OriginalSize, "merged trampoline source window end");
        if (Next.BackReserve > Next.Bytes.size() ||
            !NextEnd || *NextEnd > FunctionRange->End ||
            BytesTouchScratchPair(
                ArrayRef<uint8_t>(Next.Bytes.data(),
                                  Next.Bytes.size() - Next.BackReserve),
                T.LongBranchSgprBase)) {
          Unsafe = true;
          break;
        }
        Body.append(Next.Bytes.begin(),
                    Next.Bytes.end() - Next.BackReserve);
        Footprint += Next.OriginalSize;
        Cursor = *NextEnd;
        AllowedResumeCount = 1;
        PendingMerges.push_back(J);
        continue;
      }

      auto InstIt = OffToInst.find(Cursor);
      if (InstIt == OffToInst.end()) {
        Unsafe = true;
        break;
      }
      const InternalDecodedInst &DI = Ctx.Decoded[InstIt->second];
      std::optional<uint64_t> InstEnd = checkedAddUint64(
          DI.Offset, DI.Size, "relocated instruction end");
      if (!InstEnd || *InstEnd > FunctionRange->End ||
          UnsafeToRelocate(DI) ||
          TouchesScratchPair(DI, T.LongBranchSgprBase)) {
        Unsafe = true;
        break;
      }
      Body.append(Ctx.Text + DI.Offset, Ctx.Text + *InstEnd);
      Footprint += DI.Size;
      Cursor = *InstEnd;
      AllowedResumeCount = 0;
    }

    // Once the forward edge fits, keep coalescing immediately adjacent patch
    // sites. Leaving the third site in a 3-site run stranded in its own
    // 8-byte window can require a relay even though the existing source window
    // can execute all three semantic replacements and return after the run.
    while (!Unsafe) {
      if (Cursor >= FunctionRange->End || BranchTargets.contains(Cursor) ||
          SymbolTargets.contains(Cursor) ||
          ResumeCountAt(Cursor) > AllowedResumeCount)
        break;
      auto TrampIt = OffToTramp.find(Cursor);
      if (TrampIt == OffToTramp.end())
        break;
      size_t J = TrampIt->second;
      if (J <= I || Removed[J])
        break;
      Trampoline &Next = Tramps[J];
      std::optional<uint64_t> NextEnd = checkedAddUint64(
          Cursor, Next.OriginalSize, "coalesced trampoline source window end");
      if (Next.BackReserve > Next.Bytes.size() || !NextEnd ||
          *NextEnd > FunctionRange->End ||
          BytesTouchScratchPair(
              ArrayRef<uint8_t>(Next.Bytes.data(),
                                Next.Bytes.size() - Next.BackReserve),
              T.LongBranchSgprBase))
        break;
      Body.append(Next.Bytes.begin(), Next.Bytes.end() - Next.BackReserve);
      Footprint += Next.OriginalSize;
      Cursor = *NextEnd;
      AllowedResumeCount = 1;
      PendingMerges.push_back(J);
    }

    // A completed source window can otherwise strand the next required site
    // behind a few relocatable instructions. Bridge only a short, untargeted
    // run and commit it transactionally when it reaches another trampoline.
    // Unknown indirect entries forbid this because any instruction boundary
    // in the function may be externally reachable.
    if (!Unsafe && !HasUnknownIndirectEntry) {
      constexpr uint32_t MaxBridgeBytes = LongSetPcMaxBytes;
      while (true) {
        uint64_t BridgeCursor = Cursor;
        uint32_t BridgeBytes = 0;
        unsigned BridgeAllowedResumeCount = AllowedResumeCount;
        SmallVector<uint8_t> BridgeBody;
        bool Bridged = false;

        while (BridgeBytes <= MaxBridgeBytes) {
          if (BridgeCursor >= FunctionRange->End ||
              BranchTargets.contains(BridgeCursor) ||
              SymbolTargets.contains(BridgeCursor) ||
              ResumeCountAt(BridgeCursor) > BridgeAllowedResumeCount)
            break;

          auto TrampIt = OffToTramp.find(BridgeCursor);
          if (TrampIt != OffToTramp.end()) {
            size_t J = TrampIt->second;
            if (J <= I || Removed[J])
              break;
            Trampoline &Next = Tramps[J];
            std::optional<uint64_t> NextEnd = checkedAddUint64(
                BridgeCursor, Next.OriginalSize,
                "bridged trampoline source window end");
            if (Next.BackReserve > Next.Bytes.size() || !NextEnd ||
                *NextEnd > FunctionRange->End ||
                BytesTouchScratchPair(
                    ArrayRef<uint8_t>(
                        Next.Bytes.data(),
                        Next.Bytes.size() - Next.BackReserve),
                    T.LongBranchSgprBase))
              break;

            Body.append(BridgeBody.begin(), BridgeBody.end());
            Body.append(Next.Bytes.begin(),
                        Next.Bytes.end() - Next.BackReserve);
            Footprint += BridgeBytes + Next.OriginalSize;
            Cursor = *NextEnd;
            AllowedResumeCount = 1;
            PendingMerges.push_back(J);
            Bridged = true;
            break;
          }

          auto InstIt = OffToInst.find(BridgeCursor);
          if (InstIt == OffToInst.end())
            break;
          const InternalDecodedInst &DI = Ctx.Decoded[InstIt->second];
          std::optional<uint64_t> InstEnd = checkedAddUint64(
              DI.Offset, DI.Size, "bridged instruction end");
          if (!InstEnd || *InstEnd > FunctionRange->End ||
              DI.Size > MaxBridgeBytes - BridgeBytes ||
              UnsafeToRelocate(DI) ||
              TouchesScratchPair(DI, T.LongBranchSgprBase))
            break;
          BridgeBody.append(Ctx.Text + DI.Offset, Ctx.Text + *InstEnd);
          BridgeBytes += DI.Size;
          BridgeCursor = *InstEnd;
          BridgeAllowedResumeCount = 0;
        }

        if (!Bridged)
          break;
      }
    }

    if (Unsafe || Footprint < RequiredForwardBytes) {
      // A forward instruction may be unsafe to move (for example an EXEC
      // update). In that case, try a contiguous window ending at the original
      // site. Replayed instructions must be untargeted and must not observe or
      // modify the SGPR pair clobbered by the set-PC edge.
      uint64_t OriginalSite = T.OriginalOffset;
      uint32_t OriginalSize = T.OriginalSize;
      uint64_t WindowStart = OriginalSite;
      uint32_t BackwardFootprint = OriginalSize;
      uint32_t BackwardRequiredBytes = RequiredForwardBytes;
      SmallVector<uint8_t> BackwardBody = OriginalBody;
      bool BackwardUnsafe = OriginalSite < HighWater;
      auto SiteInst = OffToInst.find(OriginalSite);
      size_t PrevIndex = SiteInst == OffToInst.end()
                             ? 0
                             : SiteInst->second;
      while (!BackwardUnsafe &&
             BackwardFootprint < BackwardRequiredBytes) {
        if (PrevIndex == 0 || BranchTargets.contains(WindowStart) ||
            SymbolTargets.contains(WindowStart) ||
            ResumeCountAt(WindowStart) != 0) {
          BackwardUnsafe = true;
          break;
        }
        const InternalDecodedInst &Prev = Ctx.Decoded[--PrevIndex];
        std::optional<uint64_t> PrevEnd = checkedAddUint64(
            Prev.Offset, Prev.Size, "backward source window instruction end");
        if (!PrevEnd || *PrevEnd != WindowStart ||
            Prev.Offset < FunctionRange->Begin ||
            (UnsafeToRelocate(Prev) &&
             !IsBackwardRelocatableDelay(Prev)) ||
            OffToTramp.contains(Prev.Offset) ||
            ResumeCountAt(Prev.Offset) != 0 ||
            TouchesScratchPair(Prev, T.LongBranchSgprBase)) {
          BackwardUnsafe = true;
          break;
        }
        SmallVector<uint8_t> NewBody;
        NewBody.append(Ctx.Text + Prev.Offset,
                       Ctx.Text + Prev.Offset + Prev.Size);
        NewBody.append(BackwardBody.begin(), BackwardBody.end());
        BackwardBody = std::move(NewBody);
        WindowStart = Prev.Offset;
        BackwardFootprint += Prev.Size;
        SmallVector<uint8_t> CandidateForward = encodeSetPcLongBranch(
            Ctx.LS, WindowStart, TP, T.LongBranchSgprBase);
        if (CandidateForward.empty()) {
          BackwardUnsafe = true;
          break;
        }
        BackwardRequiredBytes = CandidateForward.size();
      }
      if (!BackwardUnsafe &&
          BackwardFootprint >= BackwardRequiredBytes) {
        T.OriginalOffset = WindowStart;
        T.OriginalSize = BackwardFootprint;
        Body = std::move(BackwardBody);
        Footprint = BackwardFootprint;
        std::optional<uint64_t> OriginalSiteEnd = checkedAddUint64(
            OriginalSite, OriginalSize, "backward source window end");
        if (!OriginalSiteEnd)
          return false;
        Cursor = *OriginalSiteEnd;
        Unsafe = false;
        PendingMerges.clear();
        log() << "hotswap: set-PC forward site 0x" << utohexstr(OriginalSite)
              << " expanded backward to 0x" << utohexstr(WindowStart)
              << " (" << BackwardFootprint << " bytes)\n";
      }
    }

    if (Unsafe || Footprint < RequiredForwardBytes) {
      if (reserveSetPcForwardRelay(Ctx, T, TP)) {
        ++Expanded;
        std::optional<uint64_t> Next = checkedAddUint64(
            PoolCursor, T.Bytes.size(), "relay trampoline pool cursor");
        if (!Next)
          return false;
        PoolCursor = *Next;
        continue;
      }
      if (emitFarSiteToCodeCave(Ctx, T)) {
        Removed[I] = true;
        continue;
      }
      log() << "hotswap: error: required far site 0x"
            << utohexstr(T.OriginalOffset)
            << " cannot safely form a set-PC forward window (footprint "
            << Footprint << ", need " << RequiredForwardBytes << ")\n";
      return false;
    }

    for (size_t J : PendingMerges)
      Removed[J] = true;
    Merged += PendingMerges.size();

    T.Bytes = std::move(Body);
    T.Bytes.append(T.BackReserve, uint8_t{0});
    T.SiteFootprint = Footprint;
    T.UsesSetPcForward = true;
    for (uint64_t Offset = T.OriginalOffset; Offset < Cursor;
         Offset += MinInstSize)
      Ctx.MutatedOffsets.insert(Offset);
    HighWater = Cursor;
    ++Expanded;
    std::optional<uint64_t> Next = checkedAddUint64(
        PoolCursor, T.Bytes.size(), "expanded trampoline pool cursor");
    if (!Next)
      return false;
    PoolCursor = *Next;
  }

  if (llvm::is_contained(Removed, true)) {
    std::vector<Trampoline> Compact;
    Compact.reserve(Tramps.size() - Merged);
    for (size_t I = 0, E = Tramps.size(); I < E; ++I)
      if (!Removed[I])
        Compact.push_back(std::move(Tramps[I]));
    Tramps.swap(Compact);
  }

  for (const Trampoline &T : Tramps) {
    if (!T.Long || T.LongBranchRequiredSgprs == 0 ||
        T.LongBranchKernelName.empty())
      continue;
    std::optional<unsigned> Before =
        Ctx.Elf.getKernelSgprCount(T.LongBranchKernelName);
    if (!Before || T.LongBranchRequiredSgprs < *Before) {
      log() << "hotswap: error: invalid surviving set-PC SGPR reservation for "
            << T.LongBranchKernelName << "\n";
      return false;
    }
    KernelPatchStats &Stats = Ctx.KernelStats[T.LongBranchKernelName];
    Stats.ExtraSgprs = std::max(
        Stats.ExtraSgprs, T.LongBranchRequiredSgprs - *Before);
  }

  log() << "hotswap: set-PC forward windows: expanded " << Expanded
        << " far site(s), merged " << Merged
        << " adjacent trampoline site(s), synthesized zero s_add_pc_i64\n";
  return true;
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
/// records per-kernel stats via ElfView::updateKernelDescriptor.
/// Returns the total number of applied patches across all passes.
static std::optional<uint32_t> applyGfx1250B0toA0Rules(
    std::vector<InternalDecodedInst> &Decoded, uint8_t *Text, uint64_t TextSize,
    const LLVMState &LS, std::vector<Trampoline> &OutTrampolines, ElfView &Elf,
    std::vector<ScratchPatchInfo> &OutScratchPatches,
    const RewriteConfig &Config, bool &OutRequiredPatchApplied) {
  uint32_t Patched = 0;
  std::vector<NopSled> Sleds = buildNopSledMap(Decoded, LS, Elf);

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
  std::vector<KernelTextRange> KernelRanges = collectKernelTextRanges(Elf, LS);
  std::vector<KernelDescriptorInfo> KernelDescriptors =
      Elf.kernelDescriptors();
  bool AllKernelsReserveVcc = !KernelDescriptors.empty();
  for (const KernelDescriptorInfo &KD : KernelDescriptors) {
    std::optional<unsigned> Count = Elf.getKernelSgprCount(KD.KernelName);
    if (!Count || *Count <= Gfx1250VccLoIndex) {
      AllKernelsReserveVcc = false;
      break;
    }
  }
  std::vector<InternalDecodedInst> AnalysisDecoded;
  if (!buildHotswapAnalysisDecoded(Elf, LS, Decoded, AnalysisDecoded))
    return std::nullopt;
  InitialVmemMustAnalysis InitialVmemAnalysis =
      computeInitialVmemMustAnalysis(Decoded, AnalysisDecoded, KernelRanges,
                                     LS);
  TensorDescriptorMustAnalysis TensorDescriptorAnalysis =
      computeTensorDescriptorMustAnalysis(Decoded, AnalysisDecoded,
                                          KernelRanges, LS,
                                          Config.MaxSgprs, Config.MaxVgprs);
  // Pool base as a .text-relative offset for trampoline branch math. The pool
  // is always >= textAddr(); checkedSubUint64 guards a malformed object.
  std::optional<uint64_t> PoolVAddr = Elf.trampolinePoolVAddr();
  if (!PoolVAddr)
    return std::nullopt;
  std::optional<uint64_t> PoolBaseOffset = checkedSubUint64(
      *PoolVAddr, Elf.textAddr(), "trampoline pool base offset");
  if (!PoolBaseOffset)
    return std::nullopt;
  PatchContext Ctx{Config,
                   Decoded,
                   Text,
                   TextSize,
                   *PoolBaseOffset,
                   LS,
                   OutTrampolines,
                   Sleds,
                   Elf,
                   InitialVmemAnalysis,
                   TensorDescriptorAnalysis,
                   Liveness,
                   KernelStats,
                   OutScratchPatches,
                   AllKernelsReserveVcc,
                   DenseMap<std::pair<uint64_t, uint64_t>,
                            FunctionSgprUsage>(),
                   DenseSet<uint64_t>(),
                   DenseMap<uint64_t, SmallVector<uint8_t>>()};

  const HotswapPatchVTable &VT = getHotswapPatchVTable();

  // Skip undecoded slots produced by the decoder for bytes it could not
  // classify as a valid instruction; the dispatcher has nothing to match
  // against on these and we must not invoke the patch passes for them.
  constexpr StringLiteral UnknownMnemonic = "<unknown>";
  using PerInstPatchFn = uint32_t (*)(PatchContext &, size_t);
  SmallVector<PerInstPatchFn, 5> PerInstPasses;
  if (Config.RunB0A0Patches) {
    PerInstPasses.push_back(VT.applyInPlacePatches);
    PerInstPasses.push_back(VT.applyTrampolinePatches);
    PerInstPasses.push_back(VT.applyWmmaSplitPatches);
    PerInstPasses.push_back(VT.applyScratchPatches);
    PerInstPasses.push_back(VT.applyWmmaScale16Patches);
  } else {
    PerInstPasses.push_back(VT.applyTrampolinePatches);
  }

  for (size_t Idx = 0, E = Decoded.size(); Idx < E; ++Idx) {
    const InternalDecodedInst &DI = Decoded[Idx];
    if (DI.Mnemonic == UnknownMnemonic)
      continue;

    for (PerInstPatchFn Fn : PerInstPasses) {
      std::optional<uint32_t> P = runPerInstPass(Fn, Ctx, Idx);
      if (!P)
        return std::nullopt;
      if (*P == 0)
        continue;
      Ctx.MutatedOffsets.insert(DI.Offset);
      Patched += *P;
      break;
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
  if (Config.RunB0A0Patches && VT.applyWmmaHazardPatch) {
    Patched += VT.applyWmmaHazardPatch(Ctx);
    if (Ctx.RequiredPatchFailed)
      return std::nullopt;
  }
  if (Config.RunB0A0Patches && VT.applyVop3px2Src2Fix) {
    Patched += VT.applyVop3px2Src2Fix(Ctx);
    if (Ctx.RequiredPatchFailed)
      return std::nullopt;
  }

  if (!expandFarForwardSetPc(Ctx)) {
    Ctx.RequiredPatchFailed = true;
    return std::nullopt;
  }

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
      bool UpdateDescriptor = !StringRef(Config.TargetCpu).starts_with("gfx1");
      if (!Elf.updateKernelDescriptorSgprCount(KName, RequiredSgprs,
                                               UpdateDescriptor)) {
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
        TrampOffset, T.Bytes.size(), "final trampoline pool cursor");
    if (!NextTrampOffset)
      return false;
    TrampOffset = *NextTrampOffset;

    if (T.BackReserve > T.Bytes.size())
      return false;
    std::optional<uint64_t> TrampEnd = checkedAddUint64(
        TP, T.Bytes.size(), "final trampoline end");
    std::optional<uint64_t> ReturnTo = checkedAddUint64(
        T.OriginalOffset, T.SiteFootprint, "final trampoline return offset");
    if (!TrampEnd || !ReturnTo || *TrampEnd < T.BackReserve)
      return false;
    const uint64_t BackSlot = *TrampEnd - T.BackReserve;
    SmallVector<uint8_t> BrBack =
        T.Long ? encodeSetPcLongBranch(LS, BackSlot, *ReturnTo,
                                       T.LongBranchSgprBase)
               : LS.encodeSBranch(BackSlot, *ReturnTo);
    if (BrBack.empty() || BrBack.size() > T.BackReserve) {
      log() << "hotswap: error: trampoline branch-back encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << (T.Long ? " (set-PC)\n" : "\n");
      return false;
    }
    std::memcpy(T.Bytes.data() + T.Bytes.size() - T.BackReserve, BrBack.data(),
                BrBack.size());
    for (uint32_t I = BrBack.size(); I + MinInstSize <= T.BackReserve;
         I += MinInstSize)
      std::memcpy(T.Bytes.data() + T.Bytes.size() - T.BackReserve + I,
                  LS.SNopBytes.data(), MinInstSize);

    if (T.Long && !T.UsesSetPcForward) {
      log() << "hotswap: error: far trampoline at 0x"
            << utohexstr(T.OriginalOffset)
            << " reached finalization without a set-PC forward window\n";
      return false;
    }
    if (T.UsesSetPcRelay) {
      std::string Pair =
          T.LongBranchSgprBase == Gfx1250VccLoIndex
              ? "vcc"
              : "s[" + std::to_string(T.LongBranchSgprBase) + ":" +
                    std::to_string(T.LongBranchSgprBase + 1) + "]";
      SmallVector<uint8_t> GetPc =
          assembleSingleInst("s_get_pc_i64 " + Pair, LS);
      std::optional<uint64_t> CapturedPc = checkedAddUint64(
          T.OriginalOffset, MinInstSize, "final set-PC relay captured PC");
      if (!CapturedPc)
        return false;
      SmallVector<uint8_t> ToRelay =
          LS.encodeSBranch(*CapturedPc, T.ForwardRelayOffset);
      SmallVector<uint8_t> Relay = encodeSetPcRelay(
          LS, *CapturedPc, TP, T.LongBranchSgprBase);
      if (GetPc.size() != MinInstSize || ToRelay.size() != MinInstSize ||
          Relay.empty() || Relay.size() > T.ForwardRelayReserve) {
        log() << "hotswap: error: trampoline relay encoding failed at 0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
      std::memcpy(Text + T.OriginalOffset, GetPc.data(), GetPc.size());
      std::memcpy(Text + T.OriginalOffset + MinInstSize, ToRelay.data(),
                  ToRelay.size());
      std::memcpy(Text + T.ForwardRelayOffset, Relay.data(), Relay.size());
      for (uint32_t I = Relay.size();
           I + MinInstSize <= T.ForwardRelayReserve; I += MinInstSize)
        std::memcpy(Text + T.ForwardRelayOffset + I, LS.SNopBytes.data(),
                    MinInstSize);
    } else {
      SmallVector<uint8_t> BrFwd =
          T.Long ? encodeSetPcLongBranch(LS, T.OriginalOffset, TP,
                                         T.LongBranchSgprBase)
                 : LS.encodeSBranch(T.OriginalOffset, TP);
      if (BrFwd.empty() || BrFwd.size() > T.SiteFootprint) {
        log() << "hotswap: error: trampoline branch-fwd encoding failed at 0x"
              << utohexstr(T.OriginalOffset)
              << (T.Long ? " (set-PC)\n" : "\n");
        return false;
      }
      std::memcpy(Text + T.OriginalOffset, BrFwd.data(), BrFwd.size());
      // Pad the tail of the replaced window with cached s_nop bytes.
      for (uint32_t I = BrFwd.size(); I + MinInstSize <= T.SiteFootprint;
           I += MinInstSize)
        std::memcpy(Text + T.OriginalOffset + I, LS.SNopBytes.data(),
                    MinInstSize);
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

amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<MemoryBuffer> &Out) {
  // The dispatcher fetches the patch vtable lazily via
  // getHotswapPatchVTable() inside applyGfx1250B0toA0Rules; the singleton's
  // initializer binds every register*Patch slot on first access, so no
  // explicit install step is needed here.

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
  Config.RunB0A0Patches = Options.RunB0A0Patches;
  Config.MaskPolicy = Options.MaskPolicy;

  uint8_t *Text = Elf.textData();
  uint64_t Count = 0;
  std::vector<Trampoline> Deferred;
  std::vector<ScratchPatchInfo> ScratchPatches;
  bool RequiredPatchApplied = false;
  if (RunInstructionPatches) {
    std::vector<InternalDecodedInst> Decoded;
    if (!decodeTextSection(Text, Elf.textSize(), LS, Decoded)) {
      log() << "hotswap: error: retargetCodeObject: decodeTextSection "
            << "failed on .text (" << Elf.textSize() << " bytes).\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    std::optional<uint32_t> Patched = applyGfx1250B0toA0Rules(
        Decoded, Text, Elf.textSize(), LS, Deferred, Elf, ScratchPatches,
        Config, RequiredPatchApplied);
    if (!Patched)
      return AMD_COMGR_STATUS_ERROR;
    Count = *Patched;
    log() << "hotswap: applied " << Count << " instruction patches\n";
  } else {
    log() << "hotswap: instruction patches disabled for this rewrite\n";
  }

  std::unique_ptr<WritableMemoryBuffer> Result;
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
  if (!Deferred.empty()) {
    if (!fixupTrampolineBranches(Deferred, Text, PoolBaseOffset, LS)) {
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
    std::optional<uint32_t> EntryCount = appendKernelEntryTrampolines(
        Elf, LS, Config.MaxSgprs, Growth, EntryFixups);
    if (!EntryCount)
      return AMD_COMGR_STATUS_ERROR;
    Count += *EntryCount;
  } else {
    log() << "hotswap: kernel-entry trampolines disabled for this rewrite\n";
  }

  if (!Deferred.empty() &&
      !appendDeferredTrampolinePrefetchGuard(Elf, LS, Growth))
    return AMD_COMGR_STATUS_ERROR;

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
    if (!rewriteKernelEntryDescriptorOffsets(*Result, PoolVAddr, LS.Cpu,
                                             EntryFixups))
      return AMD_COMGR_STATUS_ERROR;
  } else {
    Result = copyOutputBuffer(Buf.data(), ElfSize, "patched");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (!ScratchPatches.empty())
    runScratchVerification(*Result, LS, ScratchPatches, Config.MaxVgprs);

  Out = std::move(Result);
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace hotswap
} // namespace COMGR
