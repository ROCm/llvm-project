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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <limits>
#include <mutex>

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
static constexpr unsigned Gfx1250VgprGranuleSize = 16;

struct OriginalIngressInfo {
  DenseSet<uint64_t> CrossRangeEntryFunctions;
  std::vector<std::pair<uint64_t, uint64_t>> ExternalEntries;
  std::vector<std::pair<uint64_t, uint64_t>> ControlFlowEdges;
};

/// Proof facts produced by the later amd-staging control-flow analysis.  The
/// PR's whole-object analysis deliberately accepts only ABI-standard link calls;
/// these records carry the separately proven arbitrary-link calls and bounded
/// returns through the richer production pipeline instead of reducing them to
/// a target-only compatibility summary.
struct ResolvedCompatibilityCall {
  size_t InstIndex = 0;
  uint64_t Site = 0;
  uint64_t SequenceStart = 0;
  uint64_t SequenceEnd = 0;
  std::optional<uint64_t> TextTarget;
  uint64_t Continuation = 0;
  MCRegister ReturnRegister;
};

struct CompatibilityControlFlowFacts {
  DirectControlFlowInfo Summary;
  SmallVector<ResolvedCompatibilityCall, 4> ResolvedCalls;
  SmallVector<size_t, 4> UnresolvedCallIndices;
  DenseSet<uint64_t> BoundedReturnOffsets;
};

static const ResolvedCompatibilityCall *
findResolvedCompatibilityCall(const CompatibilityControlFlowFacts &Facts,
                              size_t InstIndex) {
  auto It = llvm::find_if(Facts.ResolvedCalls, [&](const auto &Call) {
    return Call.InstIndex == InstIndex;
  });
  return It == Facts.ResolvedCalls.end() ? nullptr : &*It;
}

static std::optional<MaterializedPcTransfer>
getResolvedCompatibilityTransfer(const CompatibilityControlFlowFacts &Facts,
                                 size_t InstIndex) {
  const ResolvedCompatibilityCall *Call =
      findResolvedCompatibilityCall(Facts, InstIndex);
  if (!Call || !Call->TextTarget)
    return std::nullopt;
  return MaterializedPcTransfer{Call->SequenceStart, Call->SequenceEnd,
                                *Call->TextTarget};
}

static std::optional<uint64_t>
kernelEntryVAddr(const KernelDescriptorInfo &KD) {
  if (KD.EntryOffset >= 0)
    return checkedAddUint64(KD.VAddr, static_cast<uint64_t>(KD.EntryOffset),
                            "kernel entry address");

  const uint64_t Magnitude =
      KD.EntryOffset == std::numeric_limits<int64_t>::min()
          ? uint64_t{1} << 63
          : static_cast<uint64_t>(-KD.EntryOffset);
  if (KD.VAddr < Magnitude)
    return std::nullopt;
  return KD.VAddr - Magnitude;
}

static std::optional<DenseSet<uint64_t>>
collectResolvedKernelEntryOffsets(ElfView &Elf, const LLVMState &LS) {
  if (!Elf.kernelDescriptorCacheIsComplete())
    return std::nullopt;

  std::optional<uint64_t> TextEnd =
      checkedAddUint64(Elf.textAddr(), Elf.textSize(), "kernel entry text end");
  if (!TextEnd)
    return std::nullopt;

  DenseSet<uint64_t> Result;
  for (const KernelDescriptorInfo &KD : Elf.kernelDescriptors()) {
    std::optional<uint64_t> Entry = kernelEntryVAddr(KD);
    if (!Entry)
      return std::nullopt;
    if (*Entry >= Elf.textAddr() && *Entry < *TextEnd) {
      Result.insert(*Entry - Elf.textAddr());
      continue;
    }

    const uint8_t *Stub = Elf.dataAtVAddr(*Entry, KernelEntryStubStride);
    if (!Stub)
      return std::nullopt;
    std::optional<uint64_t> OriginalEntry = getKernelEntryTrampolineTargetVAddr(
        ArrayRef<uint8_t>(Stub, KernelEntryStubStride), *Entry, LS);
    if (!OriginalEntry || *OriginalEntry < Elf.textAddr() ||
        *OriginalEntry >= *TextEnd)
      return std::nullopt;
    Result.insert(*OriginalEntry - Elf.textAddr());
  }
  return Result;
}

static bool
isExternallyVisibleCallable(const ElfView::FunctionTextRange &Range) {
  if (!Range.Symbol || Range.Symbol->getBinding() == ELF::STB_LOCAL)
    return false;
  const unsigned Visibility = Range.Symbol->getVisibility();
  return Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_PROTECTED;
}

static std::vector<KernelTextRange> collectKernelTextRanges(
    ElfView &Elf, const LLVMState &LS,
    ArrayRef<std::pair<uint64_t, uint64_t>> OriginalControlFlowEdges,
    bool HasUnknownArbitraryIndirectTarget,
    bool HasUnresolvedStandardLinkCall) {
  std::optional<DenseSet<uint64_t>> EntryOffsets =
      collectResolvedKernelEntryOffsets(Elf, LS);
  if (!EntryOffsets)
    return {};

  std::vector<KernelTextRange> Result;
  for (uint64_t Entry : *EntryOffsets) {
    std::optional<ElfView::FunctionTextRange> Owner =
        Elf.findFunctionTextRangeAtOffset(Entry);
    if (!Owner || Owner->End <= Owner->Begin)
      continue;
    auto Existing = llvm::find_if(Result, [&](const KernelTextRange &Range) {
      return Range.Begin == Owner->Begin && Range.End == Owner->End;
    });
    if (Existing == Result.end()) {
      Result.push_back(
          {Owner->Begin, Owner->End, {}, HasUnknownArbitraryIndirectTarget});
      Existing = std::prev(Result.end());
    }
    if (!llvm::is_contained(Existing->Entries, Entry))
      Existing->Entries.push_back(Entry);
  }

  for (KernelTextRange &Candidate : Result) {
    for (const auto &[Source, Target] : OriginalControlFlowEdges) {
      if (Target < Candidate.Begin || Target >= Candidate.End ||
          (Source >= Candidate.Begin && Source < Candidate.End))
        continue;
      if (!llvm::is_contained(Candidate.Entries, Target))
        Candidate.Entries.push_back(Target);
    }
    for (const ElfView::FunctionTextRange &Callable :
         Elf.functionTextRanges()) {
      if (Callable.Begin < Elf.textAddr() ||
          (!HasUnresolvedStandardLinkCall &&
           !isExternallyVisibleCallable(Callable)))
        continue;
      const uint64_t Entry = Callable.Begin - Elf.textAddr();
      if (Entry >= Candidate.Begin && Entry < Candidate.End &&
          !llvm::is_contained(Candidate.Entries, Entry))
        Candidate.Entries.push_back(Entry);
    }
    llvm::sort(Candidate.Entries);
  }
  return Result;
}

static std::vector<TensorAnalysisRange> collectTensorAnalysisRanges(
    ElfView &Elf, const LLVMState &LS, const DenseSet<uint64_t> &CodeEntries,
    const DenseSet<uint64_t> &IndirectControlFlowFunctions,
    const DenseSet<uint64_t> &CrossRangeEntryFunctions,
    ArrayRef<std::pair<uint64_t, uint64_t>> ExternalEntries,
    ArrayRef<std::pair<uint64_t, uint64_t>> OriginalControlFlowEdges,
    bool HasUnknownArbitraryIndirectTarget) {
  if (HasUnknownArbitraryIndirectTarget)
    return {};

  DenseSet<uint64_t> EntryVAddrs;
  std::vector<TensorDispatchStub> DispatchStubs;
  std::vector<uint64_t> VirtualExternalEntries;
  std::optional<uint64_t> TextEnd = checkedAddUint64(
      Elf.textAddr(), Elf.textSize(), "tensor analysis .text end");
  if (!TextEnd)
    return {};

  for (const KernelDescriptorInfo &KD : Elf.kernelDescriptors()) {
    std::optional<uint64_t> Entry = kernelEntryVAddr(KD);
    if (!Entry)
      return {};
    if (*Entry >= Elf.textAddr() && *Entry < *TextEnd) {
      EntryVAddrs.insert(*Entry);
      continue;
    }
    if (*Entry < Elf.textAddr())
      return {};
    const uint64_t EntryOffset = *Entry - Elf.textAddr();

    const uint8_t *Stub = Elf.dataAtVAddr(*Entry, KernelEntryStubStride);
    if (!Stub) {
      if (!llvm::is_contained(VirtualExternalEntries, EntryOffset))
        VirtualExternalEntries.push_back(EntryOffset);
      continue;
    }
    std::optional<KernelEntryTrampolineInfo> Info =
        getKernelEntryTrampolineInfo(
            ArrayRef<uint8_t>(Stub, KernelEntryStubStride), *Entry, LS);
    if (!Info) {
      if (!llvm::is_contained(VirtualExternalEntries, EntryOffset))
        VirtualExternalEntries.push_back(EntryOffset);
      continue;
    }
    if (Info->TargetVAddr < Elf.textAddr())
      return {};

    std::optional<uint64_t> StubOffset = checkedSubUint64(
        *Entry, Elf.textAddr(), "tensor kernel-entry stub offset");
    if (!StubOffset)
      return {};
    std::optional<uint64_t> StubEnd = checkedAddUint64(
        *StubOffset, KernelEntryStubStride, "tensor kernel-entry stub end");
    std::optional<uint64_t> TerminalOffset =
        checkedSubUint64(Info->TerminalVAddr, Elf.textAddr(),
                         "tensor kernel-entry terminal offset");
    if (!StubEnd || !TerminalOffset || *TerminalOffset < *StubOffset ||
        *TerminalOffset >= *StubEnd)
      return {};
    const uint64_t TargetOffset = Info->TargetVAddr - Elf.textAddr();
    if (Info->TargetVAddr < *TextEnd)
      EntryVAddrs.insert(Info->TargetVAddr);
    TensorDispatchStub Dispatch{*StubOffset, *StubEnd, *TerminalOffset,
                                TargetOffset};
    auto Existing = llvm::find_if(DispatchStubs, [&](const auto &Other) {
      return Other.Begin == Dispatch.Begin;
    });
    if (Existing == DispatchStubs.end()) {
      DispatchStubs.push_back(Dispatch);
    } else if (Existing->End != Dispatch.End ||
               Existing->Terminal != Dispatch.Terminal ||
               Existing->Target != Dispatch.Target) {
      return {};
    }
  }

  llvm::sort(DispatchStubs,
             [](const TensorDispatchStub &L, const TensorDispatchStub &R) {
               return L.Begin < R.Begin;
             });
  for (size_t I = 1; I < DispatchStubs.size(); ++I)
    if (DispatchStubs[I - 1].End > DispatchStubs[I].Begin)
      return {};
  if (llvm::any_of(ExternalEntries, [&](const auto &Edge) {
        return llvm::any_of(DispatchStubs, [&](const auto &Stub) {
          return Edge.second >= Stub.Begin && Edge.second < Stub.End;
        });
      }))
    return {};

  std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Elf.functionTextRanges();
  std::vector<uint64_t> OriginalCodeEntries(CodeEntries.begin(),
                                            CodeEntries.end());
  llvm::sort(OriginalCodeEntries);
  std::vector<TensorAnalysisRange> Result;
  for (const ElfView::FunctionTextRange &Range : FunctionRanges) {
    if (!EntryVAddrs.contains(Range.Begin) || Range.Begin < Elf.textAddr() ||
        Range.End <= Range.Begin || Range.End < Elf.textAddr())
      continue;
    TensorAnalysisRange Candidate{Range.Begin - Elf.textAddr(),
                                  Range.End - Elf.textAddr()};
    Candidate.VirtualExternalEntries = VirtualExternalEntries;
    Candidate.OriginalControlFlowEdges.assign(OriginalControlFlowEdges.begin(),
                                              OriginalControlFlowEdges.end());
    Candidate.OriginalCodeEntries = OriginalCodeEntries;
    Candidate.DispatchStubs = DispatchStubs;
    if (IndirectControlFlowFunctions.contains(Candidate.Begin))
      continue;
    if (llvm::any_of(CodeEntries, [&](uint64_t Entry) {
          return Entry > Candidate.Begin && Entry < Candidate.End;
        }))
      continue;
    if (llvm::any_of(EntryVAddrs, [&](uint64_t Entry) {
          return Entry > Range.Begin && Entry < Range.End;
        }))
      continue;
    if (CrossRangeEntryFunctions.contains(Candidate.Begin))
      continue;
    if (llvm::any_of(
            FunctionRanges, [&](const ElfView::FunctionTextRange &Other) {
              if (Other.Begin == Range.Begin && Other.End == Range.End)
                return false;
              return Other.Begin < Range.End && Range.Begin < Other.End;
            }))
      continue;
    if (!llvm::any_of(Result, [&](const TensorAnalysisRange &Existing) {
          return Existing.Begin == Candidate.Begin &&
                 Existing.End == Candidate.End;
        })) {
      for (const auto &[Source, Target] : ExternalEntries)
        if (Source < Candidate.Begin || Source >= Candidate.End)
          if (!llvm::is_contained(Candidate.ForeignExternalEntries, Target))
            Candidate.ForeignExternalEntries.push_back(Target);
      Result.push_back(Candidate);
    }
  }
  return Result;
}

/// Decode all executable sections into one .text-relative instruction view.
/// The original .text stream remains the only mutable input; appended HotSwap
/// pools are included only for conservative control-flow analyses.
static bool
buildHotswapAnalysisDecoded(const ElfView &Elf, const LLVMState &LS,
                            ArrayRef<InternalDecodedInst> TextDecoded,
                            std::vector<InternalDecodedInst> &Out) {
  Out.assign(TextDecoded.begin(), TextDecoded.end());
  std::optional<uint64_t> TextEnd = checkedAddUint64(
      Elf.textAddr(), Elf.textSize(), "HotSwap analysis .text end");
  if (!TextEnd)
    return false;

  for (const ElfView::ELFT::Shdr &Shdr : Elf.sections()) {
    if (&Shdr == Elf.textSection() || !(Shdr.sh_flags & ELF::SHF_EXECINSTR) ||
        Shdr.sh_type == ELF::SHT_NOBITS || Shdr.sh_size == 0)
      continue;
    std::optional<uint64_t> SectionEnd = checkedAddUint64(
        Shdr.sh_addr, Shdr.sh_size, "HotSwap analysis executable section end");
    if (!SectionEnd)
      return false;
    if (Shdr.sh_addr < *TextEnd) {
      if (*SectionEnd > Elf.textAddr()) {
        log() << "hotswap: error: executable analysis section at "
                 "vaddr 0x"
              << utohexstr(Shdr.sh_addr) << " overlaps .text\n";
        return false;
      }
      // Sections entirely below .text cannot be represented in the unsigned
      // .text-relative analysis view. Any path to them remains unknown and
      // therefore fails the range proof closed.
      continue;
    }
    if (Shdr.sh_offset > Elf.size() ||
        Shdr.sh_size > Elf.size() - Shdr.sh_offset)
      return false;

    std::vector<InternalDecodedInst> SectionDecoded;
    if (!decodeTextSection(Elf.data() + Shdr.sh_offset, Shdr.sh_size, LS,
                           SectionDecoded)) {
      log() << "hotswap: error: failed to decode executable analysis "
               "section at vaddr 0x"
            << utohexstr(Shdr.sh_addr) << "\n";
      return false;
    }

    const uint64_t Base = Shdr.sh_addr - Elf.textAddr();
    for (InternalDecodedInst &DI : SectionDecoded) {
      std::optional<uint64_t> Rebased = checkedAddUint64(
          Base, DI.Offset, "HotSwap analysis instruction offset");
      if (!Rebased)
        return false;
      DI.Offset = *Rebased;
      Out.push_back(std::move(DI));
    }
  }

  llvm::sort(Out,
             [](const InternalDecodedInst &L, const InternalDecodedInst &R) {
               return L.Offset < R.Offset;
             });
  if (!Out.empty() && Out.front().Size == 0)
    return false;
  for (size_t I = 1; I < Out.size(); ++I) {
    if (Out[I].Size == 0)
      return false;
    std::optional<uint64_t> PreviousEnd = checkedAddUint64(
        Out[I - 1].Offset, Out[I - 1].Size, "HotSwap analysis instruction end");
    if (!PreviousEnd || *PreviousEnd > Out[I].Offset) {
      log() << "hotswap: error: overlapping executable analysis "
               "instruction at 0x"
            << utohexstr(Out[I].Offset) << "\n";
      return false;
    }
  }
  return true;
}

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
                                 const ElfView::FunctionTextRange &Range,
                                 bool GlobalPostFunction = false) {
  if (End - Start >= MinNopSledSize)
    Sleds.push_back({Start, End, Start, Range.Begin, Range.End,
                     /*GatewayOnly=*/false,
                     /*GlobalGateway=*/GlobalPostFunction,
                     /*GlobalBody=*/GlobalPostFunction});
}

static bool hasNoFallthrough(const InternalDecodedInst &DI,
                             const LLVMState &LS);

static bool overlapsTextSymbolExtent(ArrayRef<ElfView::TextOffsetRange> Extents,
                                     const InternalDecodedInst &DI) {
  auto It =
      llvm::lower_bound(Extents, DI.Offset,
                        [](const ElfView::TextOffsetRange &Extent,
                           uint64_t Offset) { return Extent.End <= Offset; });
  return It != Extents.end() &&
         (It->Begin <= DI.Offset || It->Begin - DI.Offset < DI.Size);
}

/// Scan \p Decoded for runs of consecutive `s_nop` instructions at least
/// MinNopSledSize bytes long and return the resulting NopSled list. Each sled
/// records its owner range so ordinary replacement bodies use source-owned
/// padding. A strictly proven explicit-NOP run immediately after a sized
/// function remains owned by that function and is also globally usable for
/// branch gateways and explicitly opted-in PC-independent relocation bodies.
static std::vector<NopSled>
buildNopSledMap(ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
                const ElfView &Elf,
                const DenseSet<uint64_t> &DirectBranchTargets,
                const DenseSet<uint64_t> &IndirectControlFlowFunctions,
                ArrayRef<uint64_t> TextSymbolOffsets,
                ArrayRef<ElfView::TextOffsetRange> TextSymbolExtents) {
  enum class Ownership { None, InFunction, PostFunction };
  std::vector<NopSled> Sleds;
  Ownership ActiveOwnership = Ownership::None;
  bool ActiveSafe = false;
  bool ActiveHasTarget = false;
  ElfView::FunctionTextRange ActiveRange;
  uint64_t Start = 0;
  uint64_t End = 0;

  SmallVector<uint64_t, 16> SortedDirectTargets(DirectBranchTargets.begin(),
                                                DirectBranchTargets.end());
  llvm::sort(SortedDirectTargets);

  auto HasOffsetInInstruction = [](ArrayRef<uint64_t> Offsets,
                                   const InternalDecodedInst &DI) {
    auto It = llvm::lower_bound(Offsets, DI.Offset);
    return It != Offsets.end() && *It - DI.Offset < DI.Size;
  };

  auto Flush = [&] {
    if (ActiveOwnership != Ownership::None && ActiveSafe && !ActiveHasTarget &&
        (ActiveOwnership == Ownership::PostFunction || End == ActiveRange.End))
      appendNopSledIfLarge(Sleds, Start, End, ActiveRange,
                           /*GlobalPostFunction=*/ActiveOwnership ==
                               Ownership::PostFunction);
    ActiveOwnership = Ownership::None;
  };

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.Inst.getOpcode() != LS.SNopOpcode) {
      Flush();
      continue;
    }

    std::optional<ElfView::FunctionTextRange> Range =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    const bool InFunction =
        Range && DI.Offset < Range->End && DI.Size <= Range->End - DI.Offset;
    const bool HasSymbol = HasOffsetInInstruction(TextSymbolOffsets, DI);
    const bool HasSymbolExtent =
        overlapsTextSymbolExtent(TextSymbolExtents, DI);
    const bool HasDirectTarget =
        HasOffsetInInstruction(SortedDirectTargets, DI);
    const bool HasProtectedEntry =
        HasSymbol || HasSymbolExtent || HasDirectTarget;

    if (ActiveOwnership == Ownership::InFunction && InFunction &&
        ActiveRange.Begin == Range->Begin && ActiveRange.End == Range->End &&
        DI.Offset == End) {
      // Donor storage may not overwrite any symbol boundary. Exact-start
      // aliases are permitted only for the original replacement source in
      // emitReplacementCode, never for independently discovered storage.
      ActiveHasTarget |= HasProtectedEntry;
      End = DI.Offset + DI.Size;
      continue;
    }
    if (ActiveOwnership == Ownership::PostFunction && !InFunction &&
        DI.Offset == End) {
      ActiveHasTarget |= HasProtectedEntry;
      End = DI.Offset + DI.Size;
      continue;
    }
    Flush();

    if (InFunction) {
      ActiveRange = *Range;
      ActiveOwnership = Ownership::InFunction;
      Start = DI.Offset;
      ActiveSafe = I != 0 &&
                   Decoded[I - 1].Offset + Decoded[I - 1].Size == DI.Offset &&
                   hasNoFallthrough(Decoded[I - 1], LS) &&
                   !IndirectControlFlowFunctions.contains(Range->Begin);
      ActiveHasTarget = false;
    } else if (I != 0 &&
               Decoded[I - 1].Offset + Decoded[I - 1].Size == DI.Offset) {
      std::optional<ElfView::FunctionTextRange> PreviousRange =
          Elf.findFunctionTextRangeAtOffset(Decoded[I - 1].Offset);
      if (!PreviousRange || !PreviousRange->Symbol ||
          PreviousRange->Symbol->st_size == 0 ||
          PreviousRange->End != DI.Offset ||
          IndirectControlFlowFunctions.contains(PreviousRange->Begin) ||
          !hasNoFallthrough(Decoded[I - 1], LS))
        continue;
      ActiveRange = *PreviousRange;
      ActiveOwnership = Ownership::PostFunction;
      Start = DI.Offset;
      ActiveSafe = true;
      ActiveHasTarget = false;
    } else {
      continue;
    }
    ActiveHasTarget |= HasProtectedEntry;
    End = DI.Offset + DI.Size;
  }

  Flush();
  return Sleds;
}

// -- Sled-or-trampoline code emission -----------------------------------------

struct NopSledEmissionLayout {
  size_t BodySize = 0;
  size_t SourceTailOffset = 0;
  size_t SourceTailSize = 0;

  uint64_t sledBytes() const { return BodySize + MinInstSize; }
};

static bool isDs2SourceMnemonic(StringRef Mnemonic) {
  return Mnemonic == "ds_load_2addr_b32" || Mnemonic == "ds_load_2addr_b64" ||
         Mnemonic == "ds_load_2addr_stride64_b32" ||
         Mnemonic == "ds_load_2addr_stride64_b64" ||
         Mnemonic == "ds_store_2addr_b32" || Mnemonic == "ds_store_2addr_b64" ||
         Mnemonic == "ds_store_2addr_stride64_b32" ||
         Mnemonic == "ds_store_2addr_stride64_b64" ||
         Mnemonic == "ds_storexchg_2addr_rtn_b32" ||
         Mnemonic == "ds_storexchg_2addr_rtn_b64" ||
         Mnemonic == "ds_storexchg_2addr_stride64_rtn_b32" ||
         Mnemonic == "ds_storexchg_2addr_stride64_rtn_b64";
}

/// Keep a structurally validated replacement suffix in the bytes after the
/// source branch. An ordinary DS2 keeps its final DS-counter drain. A validated
/// combined-delay window keeps the reconstructed standalone delay together
/// with the instruction it protects. In both cases the relocated prefix
/// branches back to the retained suffix and execution falls through the padded
/// remainder of the original window.
static NopSledEmissionLayout
getNopSledEmissionLayout(const PatchContext &Ctx, uint64_t InstOffset,
                         uint32_t InstSize, ArrayRef<uint8_t> Replacement,
                         ReplacementPlacement Placement) {
  NopSledEmissionLayout Layout{Replacement.size(), 0, 0};
  if (Placement == ReplacementPlacement::Default ||
      InstSize < 2 * MinInstSize || Replacement.size() < MinInstSize ||
      Ctx.HasUnknownArbitraryIndirectTarget || !Ctx.DirectControlFlowTargets)
    return Layout;

  std::optional<ElfView::FunctionTextRange> Function =
      Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset);
  std::optional<uint64_t> InstEnd =
      checkedAddUint64(InstOffset, InstSize, "NOP sled source-tail end");
  if (!Function || !InstEnd || *InstEnd > Function->End)
    return Layout;
  for (uint64_t Offset = InstOffset + MinInstSize; Offset < *InstEnd;
       Offset += MinInstSize)
    if (Ctx.DirectControlFlowTargets->contains(Offset))
      return Layout;

  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Replacement.data(), Replacement.size(), Ctx.LS,
                         Decoded) ||
      Decoded.empty())
    return Layout;
  if (Placement == ReplacementPlacement::Ds2SourceTail) {
    auto Source =
        llvm::lower_bound(Ctx.Decoded, InstOffset,
                          [](const InternalDecodedInst &DI, uint64_t Offset) {
                            return DI.Offset < Offset;
                          });
    uint64_t ExpectedSourceOffset = InstOffset;
    while (ExpectedSourceOffset < *InstEnd) {
      if (Source == Ctx.Decoded.end() ||
          Source->Offset != ExpectedSourceOffset ||
          !isDs2SourceMnemonic(Source->Mnemonic))
        return Layout;
      ExpectedSourceOffset += Source->Size;
      ++Source;
    }
    if (ExpectedSourceOffset != *InstEnd)
      return Layout;
    for (uint64_t Offset = InstOffset + MinInstSize; Offset < *InstEnd;
         Offset += MinInstSize)
      if (Ctx.RelocationProtectedOffsets.contains(Offset))
        return Layout;

    const InternalDecodedInst &Wait = Decoded.back();
    if (Wait.Offset + Wait.Size != Replacement.size() ||
        Wait.Size != MinInstSize || Wait.Mnemonic != "s_wait_dscnt" ||
        Wait.Inst.getNumOperands() != 1 || !Wait.Inst.getOperand(0).isImm() ||
        Wait.Inst.getOperand(0).getImm() != 0 ||
        Replacement.size() == MinInstSize)
      return Layout;
    Layout.SourceTailOffset = MinInstSize;
    Layout.SourceTailSize = MinInstSize;
    Layout.BodySize -= MinInstSize;
    return Layout;
  }

  if (Placement != ReplacementPlacement::ProtectedCombinedDelay ||
      Decoded.size() < 2)
    return Layout;
  const InternalDecodedInst &Delay = Decoded[Decoded.size() - 2];
  const InternalDecodedInst &Protected = Decoded.back();
  if (Delay.Mnemonic != "s_delay_alu" || Delay.Size != MinInstSize ||
      getDelayProtectedSpan(Delay) != 1 ||
      Delay.Offset + Delay.Size != Protected.Offset ||
      Protected.Offset + Protected.Size != Replacement.size() ||
      Protected.Mnemonic == "<unknown>" || Protected.Mnemonic == "<replaced>" ||
      Protected.Mnemonic == "s_delay_alu" || Protected.Mnemonic == "s_clause" ||
      Protected.Mnemonic == "s_set_vgpr_msb" ||
      StringRef(Protected.Mnemonic).contains("_pc_") ||
      (Ctx.LS.MIA &&
       Ctx.LS.MIA->mayAffectControlFlow(Protected.Inst, *Ctx.LS.MRI)))
    return Layout;
  const uint64_t TailSize = Delay.Size + Protected.Size;
  if (TailSize > std::numeric_limits<uint32_t>::max() ||
      TailSize >= Replacement.size() || TailSize > InstSize - MinInstSize)
    return Layout;
  Layout.SourceTailSize = static_cast<uint32_t>(TailSize);
  Layout.SourceTailOffset = InstSize - TailSize;
  Layout.BodySize -= TailSize;
  return Layout;
}

uint64_t getNopSledBytesNeeded(const PatchContext &Ctx, uint64_t InstOffset,
                               uint32_t InstSize, ArrayRef<uint8_t> Replacement,
                               ReplacementPlacement Placement) {
  return getNopSledEmissionLayout(Ctx, InstOffset, InstSize, Replacement,
                                  Placement)
      .sledBytes();
}

/// Emit the replacement code for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize) into a nearby NOP sled: writes \p Replacement
/// into the sled, appends a branch-back to the next instruction after the
/// original site, overwrites the original site with a branch-forward to the
/// sled, and pads the leftover bytes of the original slot with cached s_nop
/// bytes. Advances \c Sled.WritePos by the amount consumed. Returns false if
/// either branch encoding fails. Branches are encoded before any bytes are
/// written so a failure leaves \c Ctx.Text and \c Sled.WritePos unchanged.
[[nodiscard]] static bool emitToNopSled(PatchContext &Ctx, NopSled &Sled,
                                        uint64_t InstOffset, uint32_t InstSize,
                                        ArrayRef<uint8_t> Replacement,
                                        ReplacementPlacement Placement) {
  const bool AllowGlobalBody = Placement != ReplacementPlacement::Default;
  if (Sled.GatewayOnly) {
    log() << "hotswap: error: emitToNopSled: gateway-only padding cannot "
             "hold a replacement body\n";
    return false;
  }
  if (!Sled.canHoldBodyFrom(InstOffset, AllowGlobalBody)) {
    log() << "hotswap: error: emitToNopSled: source does not own replacement "
             "body storage\n";
    return false;
  }
  const LLVMState &LS = Ctx.LS;
  const NopSledEmissionLayout Layout = getNopSledEmissionLayout(
      Ctx, InstOffset, InstSize, Replacement, Placement);
  const uint64_t UsableEnd = Sled.End;
  if (Sled.WritePos < Sled.Start || Sled.WritePos > UsableEnd ||
      UsableEnd > Ctx.TextSize || Layout.BodySize > UsableEnd - Sled.WritePos ||
      MinInstSize > UsableEnd - Sled.WritePos - Layout.BodySize) {
    log() << "hotswap: error: emitToNopSled: replacement exceeds owned sled "
             "capacity\n";
    return false;
  }
  if (InstOffset > Ctx.TextSize || InstSize > Ctx.TextSize - InstOffset) {
    log() << "hotswap: error: emitToNopSled: replacement site is outside "
             ".text\n";
    return false;
  }
  const uint64_t ResumeOffset = Layout.SourceTailSize != 0
                                    ? InstOffset + Layout.SourceTailOffset
                                    : InstOffset + InstSize;
  SmallVector<uint8_t> BrBack =
      LS.encodeSBranch(Sled.WritePos + Layout.BodySize, ResumeOffset);
  if (BrBack.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-back "
          << "at sled offset 0x" << utohexstr(Sled.WritePos + Layout.BodySize)
          << " -> 0x" << utohexstr(ResumeOffset) << " failed.\n";
    return false;
  }

  SmallVector<uint8_t> BrFwd = LS.encodeSBranch(InstOffset, Sled.WritePos);
  if (BrFwd.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-fwd "
          << "at original offset 0x" << utohexstr(InstOffset) << " -> sled 0x"
          << utohexstr(Sled.WritePos) << " failed.\n";
    return false;
  }

  std::memcpy(Ctx.Text + Sled.WritePos, Replacement.data(), Layout.BodySize);
  std::memcpy(Ctx.Text + Sled.WritePos + Layout.BodySize, BrBack.data(),
              BrBack.size());
  std::memcpy(Ctx.Text + InstOffset, BrFwd.data(), BrFwd.size());

  // Pad every unused source dword with the cached s_nop encoding. The retained
  // suffix, when present, is copied after this loop and therefore remains
  // contiguous even when it sits at the end of a larger source window.
  for (uint32_t I = MinInstSize; I < InstSize; I += MinInstSize)
    std::memcpy(Ctx.Text + InstOffset + I, LS.SNopBytes.data(), MinInstSize);
  if (Layout.SourceTailSize != 0)
    std::memcpy(Ctx.Text + InstOffset + Layout.SourceTailOffset,
                Replacement.data() + Layout.BodySize, Layout.SourceTailSize);

  Sled.WritePos += Layout.sledBytes();
  // Count-only row: patch placed in-line via a nearby NOP sled, no trampoline.
  Ctx.Profile.count(HotswapMetric::JumpNopSled);
  return true;
}

bool isSBranchReachable(uint64_t From, uint64_t To);

/// Place a validated, PC-independent DS2 replacement body through independently
/// routed short-branch entry and return paths. Gateway reservations are exact
/// instruction intervals, so non-overlapping roles may share certified padding
/// and either route may use zero or several hops. A validated suffix remains in
/// the source window. Every edge is encoded before bytes or cursors change.
static bool emitDs2ThroughNopSledGateways(PatchContext &Ctx,
                                          uint64_t InstOffset,
                                          uint32_t InstSize,
                                          ArrayRef<uint8_t> Replacement,
                                          ReplacementPlacement Placement) {
  if (Placement != ReplacementPlacement::Ds2SourceTail &&
      Placement != ReplacementPlacement::ProtectedCombinedDelay)
    return false;
  const bool AllowDs2SourceTailWait =
      Placement == ReplacementPlacement::Ds2SourceTail;
  const NopSledEmissionLayout Layout = getNopSledEmissionLayout(
      Ctx, InstOffset, InstSize, Replacement, Placement);
  constexpr uint64_t DsInstSize = 2 * MinInstSize;
  if (Layout.BodySize > std::numeric_limits<uint64_t>::max() - MinInstSize)
    return false;
  const uint64_t BodyBytes = Layout.sledBytes();
  const uint64_t ResumeOffset = Layout.SourceTailSize != 0
                                    ? InstOffset + Layout.SourceTailOffset
                                    : InstOffset + InstSize;
  if (!Ctx.LS.MIA || Ctx.RelocationProtectedOffsets.contains(InstOffset) ||
      InstOffset > Ctx.TextSize || InstSize > Ctx.TextSize - InstOffset ||
      InstSize < MinInstSize || InstSize % MinInstSize != 0 ||
      Replacement.size() != Layout.BodySize + Layout.SourceTailSize)
    return false;

  const size_t ValidatedSize =
      AllowDs2SourceTailWait ? Layout.BodySize : Replacement.size();
  std::vector<InternalDecodedInst> Body;
  if (!decodeTextSection(Replacement.data(), ValidatedSize, Ctx.LS, Body))
    return false;

  if (AllowDs2SourceTailWait) {
    const uint64_t SourceCount = InstSize / DsInstSize;
    if (InstSize % DsInstSize != 0 || SourceCount == 0 ||
        Layout.SourceTailSize != MinInstSize ||
        Layout.BodySize !=
            SourceCount * 2 * DsInstSize + (SourceCount - 1) * MinInstSize)
      return false;

    size_t BodyIndex = 0;
    uint64_t ExpectedOffset = 0;
    for (uint64_t Source = 0; Source != SourceCount; ++Source) {
      for (unsigned Half = 0; Half != 2; ++Half) {
        if (BodyIndex >= Body.size())
          return false;
        const InternalDecodedInst &Single = Body[BodyIndex++];
        if (Single.Offset != ExpectedOffset || Single.Size != DsInstSize ||
            !StringRef(Single.Mnemonic).starts_with("ds_") ||
            isDs2SourceMnemonic(Single.Mnemonic))
          return false;
        ExpectedOffset += DsInstSize;
      }
      if (Source + 1 == SourceCount)
        continue;
      if (BodyIndex >= Body.size())
        return false;
      const InternalDecodedInst &Wait = Body[BodyIndex++];
      if (Wait.Offset != ExpectedOffset || Wait.Size != MinInstSize ||
          Wait.Mnemonic != "s_wait_dscnt" || Wait.Inst.getNumOperands() != 1 ||
          !Wait.Inst.getOperand(0).isImm() ||
          Wait.Inst.getOperand(0).getImm() != 0)
        return false;
      ExpectedOffset += MinInstSize;
    }
    if (BodyIndex != Body.size() || ExpectedOffset != Layout.BodySize)
      return false;
  } else {
    // The only whole-window caller is the validated combined-delay DS2
    // demerge. Keep the relay defensive: its reconstructed body must remain
    // straight-line and position-independent, and both standalone delays must
    // protect only their immediately following instruction.
    unsigned DelayCount = 0;
    unsigned SplitDsCount = 0;
    for (const InternalDecodedInst &DI : Body) {
      if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>" ||
          DI.Mnemonic == "s_clause" || DI.Mnemonic == "s_set_vgpr_msb" ||
          StringRef(DI.Mnemonic).contains("_pc_") ||
          (Ctx.LS.MIA &&
           Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI)))
        return false;
      if (DI.Mnemonic == "s_delay_alu") {
        if (getDelayProtectedSpan(DI) != 1)
          return false;
        ++DelayCount;
      }
      if (StringRef(DI.Mnemonic).starts_with("ds_") &&
          !isDs2SourceMnemonic(DI.Mnemonic))
        ++SplitDsCount;
    }
    if (DelayCount != 2 || SplitDsCount < 2)
      return false;
  }

  struct ByteInterval {
    uint64_t Begin = 0;
    uint64_t End = 0;
  };
  struct GatewaySlot {
    size_t Sled = 0;
    uint64_t Offset = 0;
  };
  struct GatewayPlan {
    size_t BodySled = 0;
    uint64_t BodyOffset = 0;
    SmallVector<GatewaySlot, 4> EntryPath;
    SmallVector<GatewaySlot, 4> ReturnPath;
    uint64_t CursorBytes = 0;
    uint64_t MaxEdgeDistance = 0;
  };

  auto Distance = [](uint64_t A, uint64_t B) { return A > B ? A - B : B - A; };
  auto Overlaps = [](ByteInterval A, ByteInterval B) {
    return A.Begin < B.End && B.Begin < A.End;
  };
  auto IsReserved = [&](ByteInterval Candidate,
                        ArrayRef<ByteInterval> Reserved) {
    return llvm::any_of(Reserved, [&](ByteInterval Existing) {
      return Overlaps(Candidate, Existing);
    });
  };
  auto BetterPlan = [](const GatewayPlan &A, const GatewayPlan &B) {
    const size_t AGateways = A.EntryPath.size() + A.ReturnPath.size();
    const size_t BGateways = B.EntryPath.size() + B.ReturnPath.size();
    if (AGateways != BGateways)
      return AGateways < BGateways;
    if (A.CursorBytes != B.CursorBytes)
      return A.CursorBytes < B.CursorBytes;
    if (A.MaxEdgeDistance != B.MaxEdgeDistance)
      return A.MaxEdgeDistance < B.MaxEdgeDistance;
    return A.BodyOffset < B.BodyOffset;
  };

  // A placement has two independent gateway roles, entry and return, so expose
  // two leading branch slots from each certified sled. Moving the body past
  // zero, one, or two prefix slots also permits those disjoint roles and the
  // body to share one allocation cursor.
  constexpr unsigned MaxGatewaySlotsPerSled = 2;
  constexpr unsigned MaxNextAlternatives = 16;
  constexpr unsigned MaxCompletedPaths = 64;
  constexpr unsigned MaxSearchStates = 32768;
  std::optional<GatewayPlan> Best;

  auto MinimumGatewayCount = [&](uint64_t From, uint64_t To) {
    if (isSBranchReachable(From, To))
      return uint64_t{0};
    const uint64_t DistanceBytes = Distance(From, To);
    const uint64_t EdgeSpan =
        To > From ? (static_cast<uint64_t>(BranchOffsetMax) + 1) * MinInstSize
                  : static_cast<uint64_t>(-BranchOffsetMin) * MinInstSize -
                        MinInstSize;
    const uint64_t Edges =
        DistanceBytes / EdgeSpan + (DistanceBytes % EdgeSpan != 0);
    return Edges - 1;
  };
  auto BodyGatewayLowerBound = [&](size_t BI) {
    const uint64_t BodyOffset = Ctx.NopSleds[BI].WritePos;
    return MinimumGatewayCount(InstOffset, BodyOffset) +
           MinimumGatewayCount(BodyOffset + Layout.BodySize, ResumeOffset);
  };

  SmallVector<size_t, 32> BodyOrder;
  for (size_t BI = 0; BI != Ctx.NopSleds.size(); ++BI) {
    const NopSled &BodySled = Ctx.NopSleds[BI];
    if (BodySled.GatewayOnly ||
        !BodySled.canHoldBodyFrom(InstOffset, /*AllowGlobalBody=*/true) ||
        BodySled.End > Ctx.TextSize || BodySled.WritePos < BodySled.Start ||
        BodySled.WritePos > BodySled.End ||
        BodyBytes > BodySled.End - BodySled.WritePos)
      continue;
    BodyOrder.push_back(BI);
  }
  llvm::sort(BodyOrder, [&](size_t A, size_t B) {
    const uint64_t ALowerBound = BodyGatewayLowerBound(A);
    const uint64_t BLowerBound = BodyGatewayLowerBound(B);
    if (ALowerBound != BLowerBound)
      return ALowerBound < BLowerBound;
    const uint64_t AOffset = Ctx.NopSleds[A].WritePos;
    const uint64_t BOffset = Ctx.NopSleds[B].WritePos;
    const uint64_t ADistance =
        std::max(Distance(InstOffset, AOffset),
                 Distance(AOffset + Layout.BodySize, ResumeOffset));
    const uint64_t BDistance =
        std::max(Distance(InstOffset, BOffset),
                 Distance(BOffset + Layout.BodySize, ResumeOffset));
    return std::tie(ADistance, AOffset, A) < std::tie(BDistance, BOffset, B);
  });

  bool FoundResourceOptimalPlan = false;
  for (size_t BI : BodyOrder) {
    const NopSled &BodySled = Ctx.NopSleds[BI];
    for (unsigned PrefixSlots = 0; PrefixSlots <= MaxGatewaySlotsPerSled;
         ++PrefixSlots) {
      const uint64_t PrefixBytes = PrefixSlots * MinInstSize;
      if (PrefixBytes > BodySled.End - BodySled.WritePos ||
          BodyBytes > BodySled.End - BodySled.WritePos - PrefixBytes)
        continue;
      const uint64_t BodyOffset = BodySled.WritePos + PrefixBytes;
      const uint64_t BodyEnd = BodyOffset + BodyBytes;
      const uint64_t BodyReturn = BodyOffset + Layout.BodySize;
      const ByteInterval BodyInterval{BodyOffset, BodyEnd};
      const ByteInterval SourceInterval{InstOffset, InstOffset + InstSize};
      if (Overlaps(BodyInterval, SourceInterval))
        continue;

      SmallVector<GatewaySlot, 32> Slots;
      for (size_t SI = 0; SI != Ctx.NopSleds.size(); ++SI) {
        const NopSled &Sled = Ctx.NopSleds[SI];
        if (!Sled.canGatewayFrom(InstOffset) || Sled.End > Ctx.TextSize ||
            Sled.WritePos < Sled.Start || Sled.WritePos > Sled.End)
          continue;

        auto AppendSlots = [&](uint64_t Start, unsigned Count) {
          for (unsigned I = 0; I != Count; ++I) {
            if (Start > Sled.End || MinInstSize > Sled.End - Start)
              break;
            ByteInterval SlotInterval{Start, Start + MinInstSize};
            if (!Overlaps(SlotInterval, BodyInterval) &&
                !Overlaps(SlotInterval, SourceInterval))
              Slots.push_back({SI, Start});
            Start += MinInstSize;
          }
        };

        if (SI == BI) {
          AppendSlots(Sled.WritePos, PrefixSlots);
          AppendSlots(BodyEnd, MaxGatewaySlotsPerSled);
        } else {
          AppendSlots(Sled.WritePos, MaxGatewaySlotsPerSled);
        }
      }
      llvm::sort(Slots, [](const GatewaySlot &A, const GatewaySlot &B) {
        return std::tie(A.Offset, A.Sled) < std::tie(B.Offset, B.Sled);
      });

      SmallVector<ByteInterval, 4> PermanentReservations{BodyInterval,
                                                         SourceInterval};
      // Enumerate a bounded set of progressively closer routes. Backtracking
      // over several next-hop choices avoids the old nearest-island greediness;
      // running the joint search in both path orders avoids privileging entry
      // over return when the two routes contend for the same interval.
      auto SearchPath = [&](uint64_t Start, uint64_t Target,
                            ArrayRef<ByteInterval> Reservations,
                            unsigned &StatesRemaining, auto &&OnComplete) {
        SmallVector<size_t, 8> Path;
        unsigned CompletedPaths = 0;
        auto Visit = [&](auto &&Self, uint64_t Current) -> bool {
          if (StatesRemaining == 0 || CompletedPaths >= MaxCompletedPaths)
            return false;
          --StatesRemaining;

          if (isSBranchReachable(Current, Target)) {
            ++CompletedPaths;
            if (OnComplete(ArrayRef<size_t>(Path)))
              return true;
          }

          SmallVector<size_t, MaxNextAlternatives> Next;
          const uint64_t CurrentDistance = Distance(Current, Target);
          for (size_t I = 0; I != Slots.size(); ++I) {
            const GatewaySlot &Slot = Slots[I];
            const ByteInterval SlotInterval{Slot.Offset,
                                            Slot.Offset + MinInstSize};
            if (Distance(Slot.Offset, Target) >= CurrentDistance ||
                !isSBranchReachable(Current, Slot.Offset) ||
                IsReserved(SlotInterval, Reservations))
              continue;
            bool ConflictsWithPath = llvm::any_of(Path, [&](size_t PI) {
              const GatewaySlot &Used = Slots[PI];
              return Overlaps(SlotInterval,
                              {Used.Offset, Used.Offset + MinInstSize});
            });
            if (!ConflictsWithPath)
              Next.push_back(I);
          }
          llvm::sort(Next, [&](size_t A, size_t B) {
            const uint64_t ADistance = Distance(Slots[A].Offset, Target);
            const uint64_t BDistance = Distance(Slots[B].Offset, Target);
            return std::tie(ADistance, Slots[A].Offset, Slots[A].Sled) <
                   std::tie(BDistance, Slots[B].Offset, Slots[B].Sled);
          });
          if (Next.size() > MaxNextAlternatives)
            Next.resize(MaxNextAlternatives);
          for (size_t I : Next) {
            Path.push_back(I);
            if (Self(Self, Slots[I].Offset))
              return true;
            Path.pop_back();
          }
          return false;
        };
        return Visit(Visit, Start);
      };

      SmallVector<size_t, 8> EntryIndices;
      SmallVector<size_t, 8> ReturnIndices;
      auto FindJointPaths = [&](bool EntryFirst) {
        unsigned StatesRemaining = MaxSearchStates;
        if (EntryFirst) {
          return SearchPath(
              InstOffset, BodyOffset, PermanentReservations, StatesRemaining,
              [&](ArrayRef<size_t> Entry) {
                SmallVector<ByteInterval, 12> Reserved(
                    PermanentReservations.begin(), PermanentReservations.end());
                for (size_t I : Entry)
                  Reserved.push_back(
                      {Slots[I].Offset, Slots[I].Offset + MinInstSize});
                return SearchPath(
                    BodyReturn, ResumeOffset, Reserved, StatesRemaining,
                    [&](ArrayRef<size_t> Return) {
                      EntryIndices.assign(Entry.begin(), Entry.end());
                      ReturnIndices.assign(Return.begin(), Return.end());
                      return true;
                    });
              });
        }
        return SearchPath(
            BodyReturn, ResumeOffset, PermanentReservations, StatesRemaining,
            [&](ArrayRef<size_t> Return) {
              SmallVector<ByteInterval, 12> Reserved(
                  PermanentReservations.begin(), PermanentReservations.end());
              for (size_t I : Return)
                Reserved.push_back(
                    {Slots[I].Offset, Slots[I].Offset + MinInstSize});
              return SearchPath(
                  InstOffset, BodyOffset, Reserved, StatesRemaining,
                  [&](ArrayRef<size_t> Entry) {
                    EntryIndices.assign(Entry.begin(), Entry.end());
                    ReturnIndices.assign(Return.begin(), Return.end());
                    return true;
                  });
            });
      };
      if (!FindJointPaths(/*EntryFirst=*/true) &&
          !FindJointPaths(/*EntryFirst=*/false))
        continue;

      GatewayPlan Plan;
      Plan.BodySled = BI;
      Plan.BodyOffset = BodyOffset;
      for (size_t I : EntryIndices)
        Plan.EntryPath.push_back(Slots[I]);
      for (size_t I : ReturnIndices)
        Plan.ReturnPath.push_back(Slots[I]);

      SmallVector<uint64_t, 32> CommittedPositions(Ctx.NopSleds.size());
      for (size_t I = 0; I != Ctx.NopSleds.size(); ++I)
        CommittedPositions[I] = Ctx.NopSleds[I].WritePos;
      CommittedPositions[BI] = BodyEnd;
      for (const GatewaySlot &Slot : Plan.EntryPath)
        CommittedPositions[Slot.Sled] =
            std::max(CommittedPositions[Slot.Sled], Slot.Offset + MinInstSize);
      for (const GatewaySlot &Slot : Plan.ReturnPath)
        CommittedPositions[Slot.Sled] =
            std::max(CommittedPositions[Slot.Sled], Slot.Offset + MinInstSize);
      for (size_t I = 0; I != Ctx.NopSleds.size(); ++I)
        Plan.CursorBytes += CommittedPositions[I] - Ctx.NopSleds[I].WritePos;

      auto RecordEdgeDistance = [&](uint64_t From, uint64_t To) {
        Plan.MaxEdgeDistance =
            std::max(Plan.MaxEdgeDistance, Distance(From, To));
      };
      uint64_t From = InstOffset;
      for (const GatewaySlot &Slot : Plan.EntryPath) {
        RecordEdgeDistance(From, Slot.Offset);
        From = Slot.Offset;
      }
      RecordEdgeDistance(From, BodyOffset);
      From = BodyReturn;
      for (const GatewaySlot &Slot : Plan.ReturnPath) {
        RecordEdgeDistance(From, Slot.Offset);
        From = Slot.Offset;
      }
      RecordEdgeDistance(From, ResumeOffset);

      const size_t GatewayCount =
          Plan.EntryPath.size() + Plan.ReturnPath.size();
      const uint64_t GatewayLowerBound =
          MinimumGatewayCount(InstOffset, BodyOffset) +
          MinimumGatewayCount(BodyReturn, ResumeOffset);
      const uint64_t CursorLowerBound =
          BodyBytes + GatewayCount * static_cast<uint64_t>(MinInstSize);
      const bool PlanIsResourceOptimal = GatewayCount == GatewayLowerBound &&
                                         Plan.CursorBytes == CursorLowerBound;
      if (!Best || BetterPlan(Plan, *Best))
        Best = std::move(Plan);
      // Every body byte and gateway instruction occupies a disjoint interval,
      // so this is an absolute lower bound on aggregate cursor advancement.
      // Do not stop merely because the route uses the fewest gateways: a plan
      // with the same gateway count can still strand later rewrites by skipping
      // otherwise reusable bytes at one or more allocation cursors.
      if (PlanIsResourceOptimal) {
        FoundResourceOptimalPlan = true;
        break;
      }
    }
    // If no cursor-perfect route exists for this body sled, compare all of its
    // bounded prefix layouts but do not scan every sled with the same gateway
    // lower bound. That preserves large-object scaling while avoiding the
    // first-prefix cursor bias; later required patches still fail closed if the
    // selected valid route leaves insufficient storage.
    if (!FoundResourceOptimalPlan && Best) {
      const size_t BestGatewayCount =
          Best->EntryPath.size() + Best->ReturnPath.size();
      FoundResourceOptimalPlan = BestGatewayCount == BodyGatewayLowerBound(BI);
    }
    if (FoundResourceOptimalPlan)
      break;
  }
  if (!Best)
    return false;

  struct EncodedBranch {
    uint64_t Offset = 0;
    SmallVector<uint8_t, MinInstSize> Bytes;
  };
  SmallVector<EncodedBranch, 12> Branches;
  auto EncodeEdge = [&](uint64_t From, uint64_t To) {
    SmallVector<uint8_t> Bytes = Ctx.LS.encodeSBranch(From, To);
    if (Bytes.size() != MinInstSize)
      return false;
    Branches.push_back({From, {Bytes.begin(), Bytes.end()}});
    return true;
  };

  uint64_t From = InstOffset;
  for (const GatewaySlot &Slot : Best->EntryPath) {
    if (!EncodeEdge(From, Slot.Offset))
      return false;
    From = Slot.Offset;
  }
  if (!EncodeEdge(From, Best->BodyOffset))
    return false;
  From = Best->BodyOffset + Layout.BodySize;
  for (const GatewaySlot &Slot : Best->ReturnPath) {
    if (!EncodeEdge(From, Slot.Offset))
      return false;
    From = Slot.Offset;
  }
  if (!EncodeEdge(From, ResumeOffset))
    return false;

  // Branch encoding and interval planning are complete. From this point no
  // operation can fail, so bytes and then all participating cursors commit as
  // one placement transaction.
  std::memcpy(Ctx.Text + Best->BodyOffset, Replacement.data(), Layout.BodySize);
  for (const EncodedBranch &Branch :
       ArrayRef<EncodedBranch>(Branches).drop_front())
    std::memcpy(Ctx.Text + Branch.Offset, Branch.Bytes.data(), MinInstSize);
  for (uint32_t I = MinInstSize; I < InstSize; I += MinInstSize)
    std::memcpy(Ctx.Text + InstOffset + I, Ctx.LS.SNopBytes.data(),
                MinInstSize);
  if (Layout.SourceTailSize != 0)
    std::memcpy(Ctx.Text + InstOffset + Layout.SourceTailOffset,
                Replacement.data() + Layout.BodySize, Layout.SourceTailSize);
  // The source edge is the first encoded branch. Write it after source padding
  // and suffix retention so no source-window copy can overwrite it.
  std::memcpy(Ctx.Text + InstOffset, Branches.front().Bytes.data(),
              MinInstSize);

  SmallVector<uint64_t, 32> CommittedPositions(Ctx.NopSleds.size());
  for (size_t I = 0; I != Ctx.NopSleds.size(); ++I)
    CommittedPositions[I] = Ctx.NopSleds[I].WritePos;
  CommittedPositions[Best->BodySled] = Best->BodyOffset + BodyBytes;
  for (const GatewaySlot &Slot : Best->EntryPath)
    CommittedPositions[Slot.Sled] =
        std::max(CommittedPositions[Slot.Sled], Slot.Offset + MinInstSize);
  for (const GatewaySlot &Slot : Best->ReturnPath)
    CommittedPositions[Slot.Sled] =
        std::max(CommittedPositions[Slot.Sled], Slot.Offset + MinInstSize);
  for (size_t I = 0; I != Ctx.NopSleds.size(); ++I)
    Ctx.NopSleds[I].WritePos = CommittedPositions[I];

  log() << "hotswap: DS2 gateway plan: used " << Best->EntryPath.size()
        << " entry gateway(s), " << Best->ReturnPath.size()
        << " return gateway(s), and body sled at 0x"
        << utohexstr(Best->BodyOffset) << " for site 0x"
        << utohexstr(InstOffset) << "\n";
  Ctx.Profile.count(HotswapMetric::JumpNopSled);
  return true;
}

static std::optional<std::string>
getLongBranchPairName(unsigned SgprBase, StringRef Context) {
  if ((SgprBase & 1u) != 0 ||
      SgprBase > std::numeric_limits<unsigned>::max() - 2) {
    log() << "hotswap: error: " << Context
          << " requires an aligned SGPR pair, got s" << SgprBase << "\n";
    return std::nullopt;
  }

  return SgprBase == Gfx1250MaxSgprs
             ? "vcc"
             : "s[" + std::to_string(SgprBase) + ":" +
                   std::to_string(SgprBase + 1) + "]";
}

static SmallVector<uint8_t>
encodeLinkRelativeSetPCTail(const LLVMState &LS, uint64_t LinkOffset,
                            uint64_t TargetOffset, unsigned SgprBase) {
  std::optional<std::string> Pair =
      getLongBranchPairName(SgprBase, "set-PC long branch tail");
  if (!Pair)
    return {};

  const uint64_t Delta = TargetOffset - LinkOffset;
  std::string Asm = "s_add_nc_u64 " + *Pair + ", " + *Pair + ", 0x" +
                    utohexstr(Delta) + "\n" + "s_set_pc_i64 " + *Pair +
                    "\n";
  SmallVector<uint8_t> Tail = assembleInstructions(Asm, LS);
  if (Tail.empty() ||
      Tail.size() > SetPcReturnReserveBytes - MinInstSize) {
    log() << "hotswap: error: failed to assemble SCC-neutral set-PC tail via "
          << *Pair << "\n";
    return {};
  }
  return Tail;
}

SmallVector<uint8_t> encodeSetPCLongBranch(const LLVMState &LS,
                                           uint64_t FromOffset,
                                           uint64_t TargetOffset,
                                           unsigned SgprBase) {
  std::optional<std::string> Pair =
      getLongBranchPairName(SgprBase, "set-PC long branch");
  if (!Pair)
    return {};

  SmallVector<uint8_t> GetPc =
      assembleSingleInst("s_get_pc_i64 " + *Pair, LS);
  if (GetPc.size() != MinInstSize)
    return {};

  std::optional<uint64_t> PcBase =
      checkedAddUint64(FromOffset, GetPc.size(), "set-PC long branch PC base");
  if (!PcBase)
    return {};
  SmallVector<uint8_t> Tail =
      encodeLinkRelativeSetPCTail(LS, *PcBase, TargetOffset, SgprBase);
  if (Tail.empty())
    return {};
  SmallVector<uint8_t> Bytes = std::move(GetPc);
  llvm::append_range(Bytes, Tail);
  return Bytes;
}

SmallVector<uint8_t> encodeSCallI64(const LLVMState &LS, uint64_t FromOffset,
                                    uint64_t TargetOffset,
                                    unsigned SgprBase) {
  std::optional<std::string> Pair =
      getLongBranchPairName(SgprBase, "s_call_i64");
  if (!Pair)
    return {};

  std::optional<uint64_t> PcBase =
      checkedAddUint64(FromOffset, MinInstSize, "s_call_i64 PC base");
  if (!PcBase)
    return {};

  const bool Forward = TargetOffset >= *PcBase;
  const uint64_t ByteDistance =
      Forward ? TargetOffset - *PcBase : *PcBase - TargetOffset;
  if (ByteDistance % MinInstSize != 0) {
    log() << "hotswap: error: s_call_i64 has unaligned byte distance "
          << ByteDistance << " from 0x" << utohexstr(FromOffset) << " to 0x"
          << utohexstr(TargetOffset) << "\n";
    return {};
  }

  const uint64_t DwordDistance = ByteDistance / MinInstSize;
  const uint64_t MaxDwordDistance =
      Forward ? static_cast<uint64_t>(BranchOffsetMax)
              : static_cast<uint64_t>(-static_cast<int64_t>(BranchOffsetMin));
  if (DwordDistance > MaxDwordDistance) {
    log() << "hotswap: error: s_call_i64 target is out of simm16 range from 0x"
          << utohexstr(FromOffset) << " to 0x" << utohexstr(TargetOffset)
          << "\n";
    return {};
  }
  const int64_t DwordOffset =
      Forward ? static_cast<int64_t>(DwordDistance)
              : -static_cast<int64_t>(DwordDistance);

  // RDNA4 S_CALL_B64 writes the address after the call to SDST before making
  // the PC-relative transfer. It takes no SCC operand, so an add/set-PC tail
  // encoded from the same next-instruction address preserves SCC exactly.
  SmallVector<uint8_t> Bytes = assembleSingleInst(
      "s_call_i64 " + *Pair + ", " + std::to_string(DwordOffset), LS);
  if (Bytes.size() != MinInstSize) {
    log() << "hotswap: error: failed to assemble four-byte s_call_i64 via "
          << *Pair << "\n";
    return {};
  }
  return Bytes;
}

static std::optional<SmallVector<uint8_t>>
extractSetPCCallTail(ArrayRef<uint8_t> Direct, const LLVMState &LS) {
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Direct.data(), Direct.size(), LS, Decoded) ||
      Decoded.size() != 3 || Decoded[0].Offset != 0 ||
      Decoded[0].Size != MinInstSize ||
      Decoded[0].Inst.getOpcode() != LS.SGetPcI64Opcode ||
      Decoded[1].Offset != MinInstSize ||
      Decoded[1].Inst.getOpcode() != LS.SAddNcU64Opcode ||
      Decoded[2].Offset + Decoded[2].Size != Direct.size() ||
      Decoded[2].Size != MinInstSize ||
      Decoded[2].Inst.getOpcode() != LS.SSetPcI64Opcode ||
      Decoded[0].Inst.getNumOperands() != 1 ||
      Decoded[1].Inst.getNumOperands() != 3 ||
      Decoded[2].Inst.getNumOperands() != 1 ||
      !Decoded[0].Inst.getOperand(0).isReg() ||
      !Decoded[1].Inst.getOperand(0).isReg() ||
      !Decoded[1].Inst.getOperand(1).isReg() ||
      !Decoded[2].Inst.getOperand(0).isReg()) {
    log() << "hotswap: error: set-PC call tail has unexpected instruction "
             "structure\n";
    return std::nullopt;
  }

  const unsigned Pair = Decoded[0].Inst.getOperand(0).getReg();
  if (!Pair || Decoded[1].Inst.getOperand(0).getReg() != Pair ||
      Decoded[1].Inst.getOperand(1).getReg() != Pair ||
      Decoded[2].Inst.getOperand(0).getReg() != Pair) {
    log() << "hotswap: error: set-PC call tail does not preserve one SGPR pair\n";
    return std::nullopt;
  }

  SmallVector<uint8_t> Tail;
  llvm::append_range(Tail, Direct.drop_front(MinInstSize));
  if (Tail.size() != 12 && Tail.size() != 16) {
    log() << "hotswap: error: set-PC call tail has unsupported size "
          << Tail.size() << "\n";
    return std::nullopt;
  }
  return Tail;
}

Expected<std::optional<EncodedSetPcGateway>>
findNearestSetPcGateway(std::vector<NopSled> &Gateways, const LLVMState &LS,
                        uint64_t FromOffset, uint64_t TargetOffset,
                        unsigned SgprBase) {
  NopSled *Best = nullptr;
  SmallVector<uint8_t> BestBytes;
  uint64_t BestDistance = std::numeric_limits<uint64_t>::max();
  for (NopSled &Sled : Gateways) {
    if (!Sled.canGatewayFrom(FromOffset))
      continue;
    uint64_t UsableEnd = Sled.End;
    if (Sled.WritePos > UsableEnd)
      continue;
    uint64_t Distance = Sled.WritePos > FromOffset ? Sled.WritePos - FromOffset
                                                   : FromOffset - Sled.WritePos;
    if (Distance >= MaxSledDistance || Distance >= BestDistance ||
        LS.encodeSBranch(FromOffset, Sled.WritePos).empty())
      continue;
    SmallVector<uint8_t> Bytes =
        encodeSetPCLongBranch(LS, Sled.WritePos, TargetOffset, SgprBase);
    if (Bytes.empty())
      return createStringError(
          Twine("failed to encode set-PC gateway at candidate offset 0x") +
          utohexstr(Sled.WritePos));
    if (Bytes.size() > UsableEnd - Sled.WritePos)
      continue;
    Best = &Sled;
    BestBytes = std::move(Bytes);
    BestDistance = Distance;
  }
  if (!Best)
    return std::nullopt;
  return std::optional<EncodedSetPcGateway>(
      EncodedSetPcGateway{Best, std::move(BestBytes)});
}

Expected<uint64_t>
countReachableSetPcGatewaySlots(ArrayRef<NopSled> Gateways, const LLVMState &LS,
                                uint64_t FromOffset, uint64_t TargetOffset,
                                unsigned SgprBase, uint64_t MaxSlots) {
  uint64_t Slots = 0;
  for (const NopSled &Sled : Gateways) {
    if (!Sled.canGatewayFrom(FromOffset))
      continue;
    uint64_t Candidate = Sled.WritePos;
    while (Candidate <= Sled.End && Slots < MaxSlots) {
      uint64_t Distance = Candidate > FromOffset ? Candidate - FromOffset
                                                 : FromOffset - Candidate;
      if (Distance >= MaxSledDistance ||
          LS.encodeSBranch(FromOffset, Candidate).empty())
        break;
      SmallVector<uint8_t> Bytes =
          encodeSetPCLongBranch(LS, Candidate, TargetOffset, SgprBase);
      if (Bytes.empty())
        return createStringError(
            Twine("failed to encode set-PC gateway while counting candidate "
                  "offset 0x") +
            utohexstr(Candidate));
      if (Bytes.size() > Sled.End - Candidate)
        break;
      ++Slots;
      Candidate += Bytes.size();
    }
    if (Slots == MaxSlots)
      break;
  }
  return Slots;
}

static std::optional<unsigned> numberedSgprIndex(const MCRegisterInfo &MRI,
                                                 MCRegister Reg) {
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

static uint8_t registerPairMask(const MCRegisterInfo &MRI,
                                ArrayRef<MCRegister> PairRegs, MCRegister Reg) {
  uint8_t Mask = 0;
  for (unsigned Half = 0; Half < PairRegs.size(); ++Half)
    if (Reg.isValid() && MRI.regsOverlap(Reg.id(), PairRegs[Half].id()))
      Mask |= uint8_t{1} << Half;
  return Mask;
}

struct RegisterPairAccess {
  uint8_t Uses = 0;
  uint8_t Defs = 0;
};

/// Classify one decoded instruction's semantic accesses to a physical register
/// pair. Decoded MC operands do not always retain tied read-modify-write
/// inputs, so recover those uses from the descriptor and fail closed when an
/// omitted register operand cannot be accounted for.
static std::optional<RegisterPairAccess>
getRegisterPairAccess(const LLVMState &LS, const InternalDecodedInst &DI,
                      ArrayRef<MCRegister> PairRegs,
                      bool PairIsAbiVcc = false) {
  if (!LS.MCII || !LS.MRI || PairRegs.size() != 2 || !PairRegs[0].isValid() ||
      !PairRegs[1].isValid() || DI.Mnemonic == "<unknown>")
    return std::nullopt;

  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  auto PairMask = [&](MCRegister Reg, bool IsImplicit = false) {
    // GFX1250 is wave32. MC descriptors spell the implicit predicate as the
    // composite VCC register even though only VCC_LO participates.
    if (PairIsAbiVcc && IsImplicit && Reg.isValid() &&
        StringRef(LS.MRI->getName(Reg)) == "VCC")
      return uint8_t{1};
    if (PairIsAbiVcc && Reg.isValid() &&
        StringRef(LS.MRI->getName(Reg)).starts_with("SRC_VCCZ"))
      return uint8_t{1};
    return registerPairMask(*LS.MRI, PairRegs, Reg);
  };

  RegisterPairAccess Access;
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned OpIdx = 0; OpIdx < DI.Inst.getNumOperands(); ++OpIdx) {
    const MCOperand &Op = DI.Inst.getOperand(OpIdx);
    if (!Op.isReg() || !Op.getReg())
      continue;
    uint8_t Mask = PairMask(MCRegister(Op.getReg()));
    if (OpIdx < NumDefs)
      Access.Defs |= Mask;
    else
      Access.Uses |= Mask;
  }

  for (unsigned OpIdx = NumDefs; OpIdx < Desc.getNumOperands(); ++OpIdx) {
    int TiedTo = Desc.getOperandConstraint(OpIdx, MCOI::TIED_TO);
    if (TiedTo < 0)
      continue;
    if (static_cast<unsigned>(TiedTo) >= NumDefs ||
        static_cast<unsigned>(TiedTo) >= DI.Inst.getNumOperands())
      return std::nullopt;
    const MCOperand &Def = DI.Inst.getOperand(TiedTo);
    if (!Def.isReg() || !Def.getReg())
      return std::nullopt;
    Access.Uses |= PairMask(MCRegister(Def.getReg()));
  }

  // Some AMDGPU disassembly forms omit a tied input even when the MC
  // descriptor still declares it. Match such operands to an explicit
  // definition by register class; an unaccounted operand is not safe to infer.
  for (unsigned OpIdx = DI.Inst.getNumOperands(); OpIdx < Desc.getNumOperands();
       ++OpIdx) {
    const MCOperandInfo &Missing = Desc.operands()[OpIdx];
    if (Missing.RegClass < 0)
      continue;
    bool MatchedDefinition = false;
    for (unsigned DefIdx = 0; DefIdx < NumDefs; ++DefIdx) {
      if (Desc.operands()[DefIdx].RegClass != Missing.RegClass)
        continue;
      const MCOperand &Def = DI.Inst.getOperand(DefIdx);
      if (!Def.isReg() || !Def.getReg())
        return std::nullopt;
      Access.Uses |= PairMask(MCRegister(Def.getReg()));
      MatchedDefinition = true;
    }
    if (!MatchedDefinition)
      return std::nullopt;
  }

  for (MCPhysReg Reg : Desc.implicit_uses())
    Access.Uses |= PairMask(MCRegister(Reg), /*IsImplicit=*/true);
  for (MCPhysReg Reg : Desc.implicit_defs())
    Access.Defs |= PairMask(MCRegister(Reg), /*IsImplicit=*/true);
  return Access;
}

std::optional<uint64_t>
evaluateDirectControlFlowTarget(const InternalDecodedInst &DI,
                                const LLVMState &LS);

static bool isStandardLinkCall(const InternalDecodedInst &DI,
                               const LLVMState &LS);

bool isStandardLinkReturn(const InternalDecodedInst &DI, const LLVMState &LS);

std::optional<int64_t> getAbsoluteOperandValue(const MCOperand &Operand,
                                               const InternalDecodedInst &DI,
                                               ArrayRef<uint8_t> Text) {
  if (Operand.isImm())
    return Operand.getImm();
  if (!Operand.isExpr() ||
      (DI.Size != 2 * MinInstSize && DI.Size != 3 * MinInstSize))
    return std::nullopt;

  std::optional<uint64_t> End =
      checkedAddUint64(DI.Offset, DI.Size, "literal instruction end");
  if (!End || *End > Text.size())
    return std::nullopt;
  // AMDGPU's binary decoder represents trailing literal dwords as an MCExpr
  // whose storage is not guaranteed to outlive getInstruction(). Read the
  // encoded literal32 or literal64 instead of dereferencing that transient
  // expression. The callers only accept scalar add opcodes whose base
  // encoding is one dword.
  if (DI.Size == 2 * MinInstSize)
    return static_cast<int64_t>(
        support::endian::read32le(Text.data() + *End - MinInstSize));
  return static_cast<int64_t>(
      support::endian::read64le(Text.data() + *End - 2 * MinInstSize));
}

static std::optional<uint64_t> evaluateMaterializedSetPcTarget(
    ArrayRef<InternalDecodedInst> Function, unsigned SetPcIndex,
    const DenseSet<uint64_t> &DirectTargets, ArrayRef<uint8_t> Text,
    const LLVMState &LS, bool AllowOutsideText = false) {
  if (SetPcIndex < 3 || !LS.MRI)
    return std::nullopt;
  const InternalDecodedInst &GetPc = Function[SetPcIndex - 3];
  const InternalDecodedInst &AddLo = Function[SetPcIndex - 2];
  const InternalDecodedInst &AddHi = Function[SetPcIndex - 1];
  const InternalDecodedInst &SetPc = Function[SetPcIndex];
  if (GetPc.Offset + GetPc.Size != AddLo.Offset ||
      AddLo.Offset + AddLo.Size != AddHi.Offset ||
      AddHi.Offset + AddHi.Size != SetPc.Offset ||
      GetPc.Mnemonic != "s_get_pc_i64" || AddLo.Mnemonic != "s_add_co_u32" ||
      AddHi.Mnemonic != "s_add_co_ci_u32" || SetPc.Mnemonic != "s_set_pc_i64" ||
      GetPc.Inst.getNumOperands() != 1 || SetPc.Inst.getNumOperands() != 1 ||
      !GetPc.Inst.getOperand(0).isReg() || !SetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != SetPc.Inst.getOperand(0).getReg() ||
      AddLo.Inst.getNumOperands() != 3 || AddHi.Inst.getNumOperands() != 3 ||
      !AddLo.Inst.getOperand(0).isReg() || !AddLo.Inst.getOperand(1).isReg() ||
      !AddHi.Inst.getOperand(0).isReg() || !AddHi.Inst.getOperand(1).isReg() ||
      AddLo.Inst.getOperand(0).getReg() != AddLo.Inst.getOperand(1).getReg() ||
      AddHi.Inst.getOperand(0).getReg() != AddHi.Inst.getOperand(1).getReg())
    return std::nullopt;

  std::optional<int64_t> LoValue =
      getAbsoluteOperandValue(AddLo.Inst.getOperand(2), AddLo, Text);
  std::optional<int64_t> HiValue =
      getAbsoluteOperandValue(AddHi.Inst.getOperand(2), AddHi, Text);
  std::optional<uint64_t> SequenceEnd = checkedAddUint64(
      SetPc.Offset, SetPc.Size, "materialized set-PC sequence end");
  if (!LoValue || !HiValue || !SequenceEnd)
    return std::nullopt;
  for (uint64_t DirectTarget : DirectTargets)
    if (DirectTarget > GetPc.Offset && DirectTarget < *SequenceEnd)
      return std::nullopt;

  MCRegister Pair(GetPc.Inst.getOperand(0).getReg());
  std::optional<unsigned> Lo =
      numberedSgprIndex(*LS.MRI, MCRegister(AddLo.Inst.getOperand(0).getReg()));
  std::optional<unsigned> Hi =
      numberedSgprIndex(*LS.MRI, MCRegister(AddHi.Inst.getOperand(0).getReg()));
  if (!Lo || !Hi || *Hi != *Lo + 1 ||
      !LS.MRI->regsOverlap(Pair.id(), AddLo.Inst.getOperand(0).getReg()) ||
      !LS.MRI->regsOverlap(Pair.id(), AddHi.Inst.getOperand(0).getReg()))
    return std::nullopt;

  uint64_t Delta =
      static_cast<uint32_t>(*LoValue) |
      (static_cast<uint64_t>(static_cast<uint32_t>(*HiValue)) << 32);
  std::optional<uint64_t> PcBase =
      checkedAddUint64(GetPc.Offset, GetPc.Size, "materialized set-PC PC base");
  if (!PcBase)
    return std::nullopt;
  uint64_t Target = *PcBase + Delta;

  if (Target >= Text.size())
    return AllowOutsideText ? std::optional<uint64_t>(Target) : std::nullopt;

  // Only model a transfer to a decoded instruction boundary owned by this
  // function. The unsigned addition above intentionally has ISA wraparound
  // semantics; the range check rejects a wrapped target outside the function.
  if (Function.empty() || Target < Function.front().Offset)
    return std::nullopt;
  std::optional<uint64_t> FunctionEnd =
      checkedAddUint64(Function.back().Offset, Function.back().Size,
                       "materialized set-PC function end");
  if (!FunctionEnd || Target >= *FunctionEnd)
    return std::nullopt;
  ArrayRef<InternalDecodedInst>::iterator TargetIt = llvm::lower_bound(
      Function, Target, [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  if (TargetIt == Function.end() || TargetIt->Offset != Target)
    return std::nullopt;
  return Target;
}

static bool isDecodedInstructionBoundary(ArrayRef<InternalDecodedInst> Decoded,
                                         uint64_t Target) {
  ArrayRef<InternalDecodedInst>::iterator It = llvm::lower_bound(
      Decoded, Target, [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  return It != Decoded.end() && It->Offset == Target;
}

struct SgprPairHalves {
  std::array<MCRegister, 2> Regs;
  bool IsVcc = false;
};

static std::optional<SgprPairHalves> getSgprPairHalves(const LLVMState &LS,
                                                       MCRegister Pair) {
  if (!LS.MRI || !Pair.isValid())
    return std::nullopt;

  if (StringRef(LS.MRI->getName(Pair)) == "VCC") {
    std::array<MCRegister, 2> Regs;
    for (unsigned Reg = 1, End = LS.MRI->getNumRegs(); Reg < End; ++Reg) {
      StringRef Name(LS.MRI->getName(Reg));
      if (Name == "VCC_LO")
        Regs[0] = MCRegister(Reg);
      else if (Name == "VCC_HI")
        Regs[1] = MCRegister(Reg);
    }
    if (!Regs[0].isValid() || !Regs[1].isValid())
      return std::nullopt;
    return SgprPairHalves{Regs, /*IsVcc=*/true};
  }

  SmallVector<std::pair<unsigned, MCRegister>, 2> Halves;
  for (unsigned Reg = 1, End = LS.MRI->getNumRegs(); Reg < End; ++Reg) {
    MCRegister Candidate(Reg);
    std::optional<unsigned> Index = numberedSgprIndex(*LS.MRI, Candidate);
    if (Index && LS.MRI->regsOverlap(Pair.id(), Candidate.id()))
      Halves.emplace_back(*Index, Candidate);
  }
  llvm::sort(Halves, llvm::less_first());
  Halves.erase(llvm::unique(Halves,
                            [](const auto &Lhs, const auto &Rhs) {
                              return Lhs.first == Rhs.first;
                            }),
               Halves.end());
  if (Halves.size() != 2 || Halves[1].first != Halves[0].first + 1)
    return std::nullopt;
  return SgprPairHalves{{Halves[0].second, Halves[1].second},
                        /*IsVcc=*/false};
}

static bool hasOffsetInside(ArrayRef<uint64_t> SortedOffsets, uint64_t Begin,
                            uint64_t End) {
  auto It = llvm::upper_bound(SortedOffsets, Begin);
  return It != SortedOffsets.end() && *It < End;
}

static bool overlapsTextSymbolExtent(ArrayRef<ElfView::TextOffsetRange> Extents,
                                     uint64_t Begin, uint64_t End) {
  auto It =
      llvm::lower_bound(Extents, Begin,
                        [](const ElfView::TextOffsetRange &Extent,
                           uint64_t Offset) { return Extent.End <= Offset; });
  return It != Extents.end() && It->Begin < End;
}

/// Recognize the split gateway emitted for an undersized far source:
///
///   source: s_call_i64 pair, tail
///   tail:   s_add_nc_u64 pair, pair, target - (source + 4)
///           s_set_pc_i64 pair
///
/// The tail is not self-initializing, so accept it only in unowned gateway
/// padding with one exact compatible call ingress and a no-fallthrough
/// predecessor. These conditions distinguish generated tails from ordinary
/// ABI returns, including the s[30:31] case.
static std::optional<MaterializedPcTransfer> evaluateCallTailTransfer(
    ArrayRef<InternalDecodedInst> Decoded, size_t TransferIndex, size_t AddIndex,
    MCRegister TargetPair, const DenseSet<uint64_t> &DirectTargets,
    ArrayRef<uint8_t> Text, const LLVMState &LS, const ElfView &Elf,
    std::optional<ArrayRef<uint64_t>> TextSymbolOffsets,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents) {
  if (!TextSymbolOffsets || !TextSymbolExtents || AddIndex == 0 ||
      AddIndex + 1 != TransferIndex || TransferIndex >= Decoded.size())
    return std::nullopt;

  const InternalDecodedInst &Add = Decoded[AddIndex];
  const InternalDecodedInst &Transfer = Decoded[TransferIndex];
  const InternalDecodedInst &Previous = Decoded[AddIndex - 1];
  std::optional<uint64_t> SequenceEnd = checkedAddUint64(
      Transfer.Offset, Transfer.Size, "call-tail transfer end");
  if (!SequenceEnd || Elf.findFunctionTextRangeAtOffset(Add.Offset) ||
      Previous.Offset + Previous.Size != Add.Offset ||
      !hasNoFallthrough(Previous, LS) ||
      Add.Inst.getOpcode() != LS.SAddNcU64Opcode ||
      Add.Inst.getNumOperands() != 3 ||
      !Add.Inst.getOperand(0).isReg() ||
      !Add.Inst.getOperand(1).isReg() ||
      Add.Inst.getOperand(0).getReg() != TargetPair.id() ||
      Add.Inst.getOperand(1).getReg() != TargetPair.id() ||
      Transfer.Inst.getOpcode() != LS.SSetPcI64Opcode ||
      Transfer.Inst.getNumOperands() != 1 ||
      !Transfer.Inst.getOperand(0).isReg() ||
      Transfer.Inst.getOperand(0).getReg() != TargetPair.id() ||
      Add.Offset + Add.Size != Transfer.Offset ||
      !DirectTargets.contains(Add.Offset))
    return std::nullopt;

  for (uint64_t DirectTarget : DirectTargets)
    if (DirectTarget > Add.Offset && DirectTarget < *SequenceEnd)
      return std::nullopt;
  auto Symbol = llvm::lower_bound(*TextSymbolOffsets, Add.Offset);
  if ((Symbol != TextSymbolOffsets->end() && *Symbol < *SequenceEnd) ||
      overlapsTextSymbolExtent(*TextSymbolExtents, Add.Offset, *SequenceEnd))
    return std::nullopt;

  const uint64_t SourceLower =
      Add.Offset > MaxSledDistance ? Add.Offset - MaxSledDistance : 0;
  const uint64_t SourceUpper =
      Add.Offset > std::numeric_limits<uint64_t>::max() -
                       (MaxSledDistance - MinInstSize)
          ? std::numeric_limits<uint64_t>::max()
          : Add.Offset + MaxSledDistance - MinInstSize;
  auto Source = llvm::lower_bound(
      Decoded, SourceLower,
      [](const InternalDecodedInst &DI, uint64_t Offset) {
        return DI.Offset < Offset;
      });
  std::optional<uint64_t> LinkOffset;
  unsigned MatchingCalls = 0;
  for (; Source != Decoded.end() && Source->Offset <= SourceUpper; ++Source) {
    if ((!LS.MIA->isBranch(Source->Inst) && !LS.MIA->isCall(Source->Inst)) ||
        LS.MIA->isIndirectBranch(Source->Inst) ||
        LS.MIA->isReturn(Source->Inst))
      continue;
    std::optional<uint64_t> EdgeTarget =
        evaluateDirectControlFlowTarget(*Source, LS);
    if (!EdgeTarget || *EdgeTarget != Add.Offset)
      continue;
    if (Source->Inst.getOpcode() != LS.SCallI64Opcode ||
        Source->Inst.getNumOperands() != 2 ||
        !Source->Inst.getOperand(0).isReg() ||
        Source->Inst.getOperand(0).getReg() != TargetPair.id())
      return std::nullopt;
    LinkOffset = checkedAddUint64(Source->Offset, Source->Size,
                                  "call-tail link offset");
    if (!LinkOffset)
      return std::nullopt;
    ++MatchingCalls;
  }
  if (MatchingCalls != 1)
    return std::nullopt;

  std::optional<int64_t> Delta =
      getAbsoluteOperandValue(Add.Inst.getOperand(2), Add, Text);
  if (!Delta)
    return std::nullopt;
  const uint64_t Target = *LinkOffset + static_cast<uint64_t>(*Delta);
  std::optional<uint64_t> TargetVAddr = checkedAddUint64(
      Elf.textAddr(), Target, "call-tail gateway target vaddr");
  if (Target < Text.size() || !TargetVAddr ||
      (Target & (MinInstSize - 1)) != 0 ||
      !Elf.isExecutableVAddrRange(*TargetVAddr, MinInstSize))
    return std::nullopt;
  return MaterializedPcTransfer{Add.Offset, *SequenceEnd, Target};
}

/// Recognize a compiler-emitted get-PC/add-nc/straight-line/set-or-swap-PC
/// transfer. The exact register and layout checks make the computed
/// destination as trustworthy as a direct branch target; malformed
/// near-matches fail closed.
static std::optional<MaterializedPcTransfer> evaluateMaterializedAddNcTransfer(
    ArrayRef<InternalDecodedInst> Decoded, size_t TransferIndex,
    const DenseSet<uint64_t> &DirectTargets, ArrayRef<uint8_t> Text,
    const LLVMState &LS, const ElfView &Elf,
    std::optional<ArrayRef<uint64_t>> TextSymbolOffsets,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents) {
  if (!LS.MCII || !LS.MIA || !LS.MRI || TransferIndex >= Decoded.size())
    return std::nullopt;

  const InternalDecodedInst &Transfer = Decoded[TransferIndex];
  const bool IsSetPc = Transfer.Inst.getOpcode() == LS.SSetPcI64Opcode;
  const bool IsSwapPc = Transfer.Inst.getOpcode() == LS.SSwapPcI64Opcode;
  if (!IsSetPc && !IsSwapPc)
    return std::nullopt;

  MCRegister TargetPair;
  if (IsSetPc) {
    if (Transfer.Inst.getNumOperands() != 1 ||
        !Transfer.Inst.getOperand(0).isReg())
      return std::nullopt;
    TargetPair = MCRegister(Transfer.Inst.getOperand(0).getReg());
  } else {
    if (Transfer.Inst.getNumOperands() != 2 ||
        !Transfer.Inst.getOperand(0).isReg() ||
        !Transfer.Inst.getOperand(1).isReg() ||
        StringRef(LS.MRI->getName(Transfer.Inst.getOperand(0).getReg())) !=
            "SGPR30_SGPR31")
      return std::nullopt;
    TargetPair = MCRegister(Transfer.Inst.getOperand(1).getReg());
    if (LS.MRI->regsOverlap(Transfer.Inst.getOperand(0).getReg(),
                            TargetPair.id()))
      return std::nullopt;
  }

  std::optional<SgprPairHalves> PairRegs = getSgprPairHalves(LS, TargetPair);
  if (!PairRegs)
    return std::nullopt;

  std::optional<ElfView::FunctionTextRange> TransferOwner =
      Elf.findFunctionTextRangeAtOffset(Transfer.Offset);
  std::optional<uint64_t> SequenceEnd = checkedAddUint64(
      Transfer.Offset, Transfer.Size, "materialized PC transfer end");
  if (!SequenceEnd || (TransferOwner && *SequenceEnd > TransferOwner->End))
    return std::nullopt;

  size_t AddIndex = TransferIndex;
  uint64_t NextOffset = Transfer.Offset;
  for (size_t I = TransferIndex; I-- > 0;) {
    const InternalDecodedInst &Candidate = Decoded[I];
    std::optional<uint64_t> CandidateEnd = checkedAddUint64(
        Candidate.Offset, Candidate.Size, "materialized transfer member end");
    if (!CandidateEnd || *CandidateEnd != NextOffset ||
        (TransferOwner && Candidate.Offset < TransferOwner->Begin))
      return std::nullopt;

    if (Candidate.Inst.getOpcode() == LS.SAddNcU64Opcode &&
        Candidate.Inst.getNumOperands() == 3 &&
        Candidate.Inst.getOperand(0).isReg() &&
        Candidate.Inst.getOperand(1).isReg() &&
        Candidate.Inst.getOperand(0).getReg() == TargetPair.id() &&
        Candidate.Inst.getOperand(1).getReg() == TargetPair.id()) {
      AddIndex = I;
      break;
    }

    // An unowned sequence is accepted only for the exact gateway form below;
    // never search arbitrary section contents for a matching definition.
    if (!TransferOwner)
      return std::nullopt;

    const MCInstrDesc &Desc = LS.MCII->get(Candidate.Inst.getOpcode());
    std::optional<RegisterPairAccess> Access =
        getRegisterPairAccess(LS, Candidate, PairRegs->Regs, PairRegs->IsVcc);
    if (!Access || Candidate.Mnemonic == "<unknown>" ||
        Candidate.Mnemonic == "<replaced>" || Access->Uses != 0 ||
        Access->Defs != 0 || Desc.mayAffectControlFlow(Candidate.Inst, *LS.MRI))
      return std::nullopt;
    NextOffset = Candidate.Offset;
  }
  if (AddIndex == TransferIndex)
    return std::nullopt;

  if (std::optional<MaterializedPcTransfer> CallTail =
          evaluateCallTailTransfer(
              Decoded, TransferIndex, AddIndex, TargetPair, DirectTargets, Text,
              LS, Elf, TextSymbolOffsets, TextSymbolExtents))
    return CallTail;
  if (AddIndex == 0)
    return std::nullopt;

  const bool DelayOnly = AddIndex + 2 == TransferIndex &&
                         Decoded[AddIndex + 1].Mnemonic == "s_delay_alu";
  const bool RequiresTailEntry =
      IsSetPc && AddIndex + 1 != TransferIndex && !DelayOnly;

  const size_t GetPcIndex = AddIndex - 1;
  const InternalDecodedInst &GetPc = Decoded[GetPcIndex];
  const InternalDecodedInst &Add = Decoded[AddIndex];
  if (GetPc.Inst.getOpcode() != LS.SGetPcI64Opcode ||
      GetPc.Offset + GetPc.Size != Add.Offset ||
      GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != TargetPair.id() ||
      Add.Inst.getNumOperands() != 3 || !Add.Inst.getOperand(0).isReg() ||
      !Add.Inst.getOperand(1).isReg() ||
      Add.Inst.getOperand(0).getReg() != TargetPair.id() ||
      Add.Inst.getOperand(1).getReg() != TargetPair.id())
    return std::nullopt;

  std::optional<int64_t> Delta =
      getAbsoluteOperandValue(Add.Inst.getOperand(2), Add, Text);
  if (!Delta)
    return std::nullopt;
  for (uint64_t DirectTarget : DirectTargets)
    if (DirectTarget > GetPc.Offset && DirectTarget < *SequenceEnd)
      return std::nullopt;
  if ((TextSymbolOffsets &&
       hasOffsetInside(*TextSymbolOffsets, GetPc.Offset, *SequenceEnd)) ||
      (TextSymbolExtents &&
       overlapsTextSymbolExtent(*TextSymbolExtents, GetPc.Offset,
                                *SequenceEnd)))
    return std::nullopt;

  std::optional<ElfView::FunctionTextRange> Owner =
      Elf.findFunctionTextRangeAtOffset(GetPc.Offset);

  uint64_t Target = GetPc.Offset + GetPc.Size + static_cast<uint64_t>(*Delta);
  if ((Target > GetPc.Offset && Target < *SequenceEnd) ||
      (Target < Text.size() && !isDecodedInstructionBoundary(Decoded, Target)))
    return std::nullopt;

  const bool HasCommonOwner = Owner && TransferOwner &&
                              Owner->Begin == TransferOwner->Begin &&
                              Owner->End == TransferOwner->End;
  bool IsCertifiedExternalGateway = false;
  if (IsSetPc && !Owner && !TransferOwner && AddIndex + 1 == TransferIndex &&
      DirectTargets.contains(GetPc.Offset) && GetPcIndex != 0 &&
      TextSymbolOffsets) {
    const InternalDecodedInst &Previous = Decoded[GetPcIndex - 1];
    const bool PreviousDoesNotFallThrough =
        Previous.Offset + Previous.Size == GetPc.Offset &&
        (hasNoFallthrough(Previous, LS) ||
         Previous.Inst.getOpcode() == LS.SSetPcI64Opcode);
    auto InteriorSymbol = llvm::upper_bound(*TextSymbolOffsets, GetPc.Offset);
    const bool HasNoInteriorSymbol =
        InteriorSymbol == TextSymbolOffsets->end() ||
        *InteriorSymbol >= *SequenceEnd;
    std::optional<uint64_t> TargetVAddr = checkedAddUint64(
        Elf.textAddr(), Target, "materialized gateway target vaddr");
    IsCertifiedExternalGateway =
        PreviousDoesNotFallThrough && HasNoInteriorSymbol &&
        Target >= Text.size() && TargetVAddr &&
        (Target & (MinInstSize - 1)) == 0 &&
        Elf.isExecutableVAddrRange(*TargetVAddr, MinInstSize);
  }
  if (!HasCommonOwner && !IsCertifiedExternalGateway)
    return std::nullopt;

  if (Target < Text.size()) {
    std::optional<ElfView::FunctionTextRange> TargetOwner =
        Elf.findFunctionTextRangeAtOffset(Target);
    if (!HasCommonOwner || !TargetOwner)
      return std::nullopt;
    if (RequiresTailEntry &&
        (*SequenceEnd != Owner->End || TargetOwner->Begin != Target ||
         TargetOwner->Begin == Owner->Begin))
      return std::nullopt;
    if (!RequiresTailEntry && TargetOwner->Begin != Owner->Begin &&
        TargetOwner->Begin != Target)
      return std::nullopt;
  } else if (!IsCertifiedExternalGateway && RequiresTailEntry) {
    return std::nullopt;
  }

  return MaterializedPcTransfer{GetPc.Offset, *SequenceEnd, Target};
}

/// Recognize the older scalar 64-bit-add lowering:
///   s_get_pc_i64 pair
///   s_add_co_u32 lo, lo, delta_lo
///   s_add_co_ci_u32 hi, hi, delta_hi
///   s_set_pc_i64 pair
/// The low add's SCC carry is consumed immediately by the high add. Exact
/// pair, adjacency, target-boundary, and interior-entry checks keep this as
/// precise as a decoded direct branch.
static std::optional<MaterializedPcTransfer> evaluateMaterializedCarryTransfer(
    ArrayRef<InternalDecodedInst> Decoded, size_t TransferIndex,
    const DenseSet<uint64_t> &DirectTargets, ArrayRef<uint8_t> Text,
    const LLVMState &LS, const ElfView &Elf) {
  if (TransferIndex < 3 || TransferIndex >= Decoded.size())
    return std::nullopt;
  std::optional<uint64_t> Target = evaluateMaterializedSetPcTarget(
      Decoded, TransferIndex, DirectTargets, Text, LS,
      /*AllowOutsideText=*/true);
  if (!Target)
    return std::nullopt;

  const InternalDecodedInst &GetPc = Decoded[TransferIndex - 3];
  const InternalDecodedInst &SetPc = Decoded[TransferIndex];
  std::optional<uint64_t> SequenceEnd = checkedAddUint64(
      SetPc.Offset, SetPc.Size, "materialized carry transfer end");
  std::optional<ElfView::FunctionTextRange> Owner =
      Elf.findFunctionTextRangeAtOffset(GetPc.Offset);
  std::optional<ElfView::FunctionTextRange> TransferOwner =
      Elf.findFunctionTextRangeAtOffset(SetPc.Offset);
  if (!SequenceEnd || !Owner || !TransferOwner ||
      Owner->Begin != TransferOwner->Begin ||
      Owner->End != TransferOwner->End ||
      (*Target > GetPc.Offset && *Target < *SequenceEnd))
    return std::nullopt;
  if (*Target < Text.size()) {
    std::optional<ElfView::FunctionTextRange> TargetOwner =
        Elf.findFunctionTextRangeAtOffset(*Target);
    if (!TargetOwner ||
        (TargetOwner->Begin != Owner->Begin && TargetOwner->Begin != *Target))
      return std::nullopt;
  }
  return MaterializedPcTransfer{GetPc.Offset, *SequenceEnd, *Target};
}

std::optional<MaterializedPcTransfer> evaluateMaterializedPcTransfer(
    ArrayRef<InternalDecodedInst> Decoded, size_t TransferIndex,
    const DenseSet<uint64_t> &DirectTargets, ArrayRef<uint8_t> Text,
    const LLVMState &LS, const ElfView &Elf,
    std::optional<ArrayRef<uint64_t>> TextSymbolOffsets,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents) {
  if (std::optional<MaterializedPcTransfer> Transfer =
          evaluateMaterializedAddNcTransfer(
              Decoded, TransferIndex, DirectTargets, Text, LS, Elf,
              TextSymbolOffsets, TextSymbolExtents))
    return Transfer;
  return evaluateMaterializedCarryTransfer(Decoded, TransferIndex,
                                           DirectTargets, Text, LS, Elf);
}

/// Combine the PR's ELF-aware materialized-transfer proof with the later
/// amd-staging arbitrary-link call proof.  The latter is indexed by the actual
/// transfer instruction, so only the exact call accepted by its ingress and
/// bounded-return analysis gains direct-transfer semantics.
static std::optional<MaterializedPcTransfer> evaluateProvenPcTransfer(
    ArrayRef<InternalDecodedInst> Decoded, size_t TransferIndex,
    const DenseSet<uint64_t> &DirectTargets, ArrayRef<uint8_t> Text,
    const LLVMState &LS, const ElfView &Elf,
    const CompatibilityControlFlowFacts &CompatibilityFacts,
    std::optional<ArrayRef<uint64_t>> TextSymbolOffsets = std::nullopt,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents =
        std::nullopt) {
  if (std::optional<MaterializedPcTransfer> Transfer =
          getResolvedCompatibilityTransfer(CompatibilityFacts, TransferIndex))
    return Transfer;
  return evaluateMaterializedPcTransfer(
      Decoded, TransferIndex, DirectTargets, Text, LS, Elf, TextSymbolOffsets,
      TextSymbolExtents);
}

static OriginalIngressInfo collectOriginalIngress(
    ArrayRef<InternalDecodedInst> Decoded,
    const DenseSet<uint64_t> &DirectTargets, ArrayRef<uint8_t> Text,
    const LLVMState &LS, const ElfView &Elf,
    const CompatibilityControlFlowFacts &CompatibilityFacts,
    std::optional<ArrayRef<uint64_t>> TextSymbolOffsets,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents) {
  OriginalIngressInfo Result;
  auto Record = [&](uint64_t Source, uint64_t Target) {
    Result.ControlFlowEdges.emplace_back(Source, Target);
    if (Target >= Text.size()) {
      Result.ExternalEntries.emplace_back(Source, Target);
      return;
    }
    std::optional<ElfView::FunctionTextRange> SourceOwner =
        Elf.findFunctionTextRangeAtOffset(Source);
    std::optional<ElfView::FunctionTextRange> TargetOwner =
        Elf.findFunctionTextRangeAtOffset(Target);
    if (TargetOwner &&
        (!SourceOwner || SourceOwner->Begin != TargetOwner->Begin ||
         SourceOwner->End != TargetOwner->End))
      Result.CrossRangeEntryFunctions.insert(TargetOwner->Begin);
  };

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if ((LS.MIA->isBranch(DI.Inst) || LS.MIA->isCall(DI.Inst)) &&
        !LS.MIA->isIndirectBranch(DI.Inst) && !LS.MIA->isReturn(DI.Inst))
      if (std::optional<uint64_t> Target =
              evaluateDirectControlFlowTarget(DI, LS))
        Record(DI.Offset, *Target);

    // A call or ordinary fallthrough immediately into another function's
    // entry is not kernel dispatch and therefore cannot carry the nonempty
    // EXEC seed used by the tensor proof.
    if (!hasNoFallthrough(DI, LS) &&
        Elf.findFunctionTextRangeAtOffset(DI.Offset)) {
      std::optional<uint64_t> Next = checkedAddUint64(
          DI.Offset, DI.Size, "tensor cross-range fallthrough");
      if (Next)
        Record(DI.Offset, *Next);
    }
  }
  for (size_t I = 0; I != Decoded.size(); ++I)
    if (std::optional<MaterializedPcTransfer> Transfer =
            evaluateProvenPcTransfer(Decoded, I, DirectTargets, Text, LS, Elf,
                                     CompatibilityFacts, TextSymbolOffsets,
                                     TextSymbolExtents))
      Record(Transfer->Begin, Transfer->Target);
  return Result;
}

std::optional<BitVector> collectTouchedNumberedSgprs(ArrayRef<uint8_t> Bytes,
                                                     unsigned NumberedSgprLimit,
                                                     const LLVMState &LS) {
  if (!LS.MCII || !LS.MRI || NumberedSgprLimit == 0) {
    log() << "hotswap: error: cannot collect replacement SGPR usage with "
             "invalid LLVM state or register limit\n";
    return std::nullopt;
  }

  SmallVector<MCRegister> NumberedSgprs(NumberedSgprLimit);
  for (unsigned Reg = 1, End = LS.MRI->getNumRegs(); Reg < End; ++Reg) {
    std::optional<unsigned> Index = numberedSgprIndex(*LS.MRI, MCRegister(Reg));
    if (Index && *Index < NumberedSgprLimit)
      NumberedSgprs[*Index] = MCRegister(Reg);
  }
  if (!llvm::all_of(NumberedSgprs,
                    [](MCRegister Reg) { return Reg.isValid(); })) {
    log() << "hotswap: error: cannot map every numbered SGPR while checking "
             "replacement usage\n";
    return std::nullopt;
  }

  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes.data(), Bytes.size(), LS, Decoded)) {
    log() << "hotswap: error: cannot decode replacement while checking SGPR "
             "usage\n";
    return std::nullopt;
  }

  BitVector Touched(NumberedSgprLimit);
  for (const InternalDecodedInst &DI : Decoded) {
    if (DI.Mnemonic == "<unknown>") {
      log() << "hotswap: error: unknown replacement instruction prevents "
               "SGPR usage proof\n";
      return std::nullopt;
    }
    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    SmallVector<MCRegister, 16> Regs;
    for (const MCOperand &Op : DI.Inst)
      if (Op.isReg() && Op.getReg())
        Regs.push_back(MCRegister(Op.getReg()));
    for (MCPhysReg Reg : Desc.implicit_uses())
      Regs.push_back(MCRegister(Reg));
    for (MCPhysReg Reg : Desc.implicit_defs())
      Regs.push_back(MCRegister(Reg));

    for (MCRegister Reg : Regs)
      for (unsigned I = 0; I < NumberedSgprLimit; ++I)
        if (Reg.isValid() &&
            LS.MRI->regsOverlap(Reg.id(), NumberedSgprs[I].id()))
          Touched.set(I);
  }
  return Touched;
}

static bool isRegisterPairDeadFrom(ArrayRef<InternalDecodedInst> Function,
                                   size_t ResumeIndex,
                                   ArrayRef<MCRegister> PairRegs,
                                   ArrayRef<uint8_t> Text, const LLVMState &LS,
                                   bool PairIsAbiVcc = false) {
  if (!LS.MCII || !LS.MRI || !LS.MIA || PairRegs.size() != 2 ||
      !PairRegs[0].isValid() || !PairRegs[1].isValid() ||
      ResumeIndex >= Function.size())
    return false;

  DenseMap<uint64_t, unsigned> OffsetToIndex;
  for (unsigned I = 0; I < Function.size(); ++I)
    OffsetToIndex.try_emplace(Function[I].Offset, I);
  DenseSet<uint64_t> DirectTargets;
  DenseSet<uint64_t> NoTargets;
  for (unsigned I = 3; I < Function.size(); ++I) {
    std::optional<uint64_t> Target =
        evaluateMaterializedSetPcTarget(Function, I, NoTargets, Text, LS);
    if (!Target)
      continue;
    // Entering after getpc would use a stale address, so a replacement may
    // not resume in the middle of a sequence that this proof recognizes.
    if (ResumeIndex > I - 3 && ResumeIndex <= I)
      return false;
    DirectTargets.insert(*Target);
  }
  for (const InternalDecodedInst &DI : Function) {
    if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
        LS.MIA->isIndirectBranch(DI.Inst) || LS.MIA->isReturn(DI.Inst))
      continue;
    bool HasImmediate = false;
    for (const MCOperand &Operand : DI.Inst)
      HasImmediate |= Operand.isImm();
    // Register-target control flow such as s_swap_pc_i64 and s_set_pc_i64
    // has no direct target. Exact materialized set-PC sequences were handled
    // above; other indirect transfers make the path proof fail when reached.
    if (!HasImmediate)
      continue;
    std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
    if (!Target) {
      log() << "hotswap: site-dead pair proof cannot resolve direct "
               "control flow at 0x"
            << utohexstr(DI.Offset) << " (" << DI.Mnemonic << ")\n";
      return false;
    }
    DirectTargets.insert(*Target);
  }

  SmallVector<std::pair<unsigned, uint8_t>, 32> Worklist;
  std::vector<uint8_t> Seen(Function.size() * 4);
  Worklist.emplace_back(ResumeIndex, uint8_t{3});
  for (size_t Next = 0; Next < Worklist.size(); ++Next) {
    unsigned I = Worklist[Next].first;
    uint8_t LiveMask = Worklist[Next].second;
    if (I >= Function.size())
      return false;
    uint8_t &WasSeen = Seen[I * 4 + LiveMask];
    if (WasSeen)
      continue;
    WasSeen = 1;

    const InternalDecodedInst &DI = Function[I];
    if (DI.Mnemonic == "<unknown>") {
      if (PairIsAbiVcc)
        log() << "hotswap: VCC lifetime proof reached unknown instruction at 0x"
              << utohexstr(DI.Offset) << "\n";
      return false;
    }
    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    const bool IsKnownNonControlFlow =
        DI.Mnemonic == "s_set_vgpr_msb" ||
        (DI.Mnemonic == "s_set_pc_i64" && DI.Inst.getNumOperands() == 1 &&
         DI.Inst.getOperand(0).isImm());
    std::optional<RegisterPairAccess> Access =
        getRegisterPairAccess(LS, DI, PairRegs, PairIsAbiVcc);
    if (!Access)
      return false;

    if (Access->Uses & LiveMask) {
      if (PairIsAbiVcc)
        log() << "hotswap: VCC lifetime proof found use-before-def at 0x"
              << utohexstr(DI.Offset) << " (" << DI.Mnemonic << ")\n";
      return false;
    }
    LiveMask &= ~Access->Defs;
    if (!LiveMask)
      continue;
    if (PairIsAbiVcc && isStandardLinkReturn(DI, LS))
      continue;
    if (DI.Mnemonic == "s_set_pc_i64") {
      std::optional<uint64_t> Materialized =
          evaluateMaterializedSetPcTarget(Function, I, DirectTargets, Text, LS);
      if (Materialized) {
        DenseMap<uint64_t, unsigned>::const_iterator TargetIt =
            OffsetToIndex.find(*Materialized);
        if (TargetIt == OffsetToIndex.end()) {
          if (PairIsAbiVcc)
            log() << "hotswap: VCC lifetime proof reached outside materialized "
                     "target at 0x"
                  << utohexstr(DI.Offset) << "\n";
          return false;
        }
        Worklist.emplace_back(TargetIt->second, LiveMask);
        continue;
      }
    }

    if (LS.MIA->isCall(DI.Inst)) {
      if (PairIsAbiVcc && (evaluateDirectControlFlowTarget(DI, LS) ||
                           isStandardLinkCall(DI, LS)))
        continue;
      if (PairIsAbiVcc)
        log() << "hotswap: VCC lifetime proof reached unresolved call at 0x"
              << utohexstr(DI.Offset) << "\n";
      // MC classifies every s_swap_pc_i64 as a call, including hand-written
      // register transfers whose target and ABI provenance are unknowable.
      // Without a proven cross-function entry set, no call is a safe lifetime
      // boundary for this post-link proof.
      return false;
    }

    if (DI.Mnemonic == "s_endpgm" || LS.MIA->isReturn(DI.Inst)) {
      if (PairIsAbiVcc)
        continue;
      return false;
    }

    if (!IsKnownNonControlFlow && LS.MIA->isBranch(DI.Inst)) {
      uint64_t Target = 0;
      if (LS.MIA->isIndirectBranch(DI.Inst) ||
          !LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target)) {
        if (PairIsAbiVcc)
          log() << "hotswap: VCC lifetime proof reached unresolved branch at 0x"
                << utohexstr(DI.Offset) << " (" << DI.Mnemonic << ")\n";
        return false;
      }
      DenseMap<uint64_t, unsigned>::const_iterator TargetIt =
          OffsetToIndex.find(Target);
      if (TargetIt == OffsetToIndex.end()) {
        if (PairIsAbiVcc)
          log() << "hotswap: VCC lifetime proof reached outside branch target "
                   "at 0x"
                << utohexstr(DI.Offset) << "\n";
        return false;
      }
      Worklist.emplace_back(TargetIt->second, LiveMask);
      if (LS.MIA->isConditionalBranch(DI.Inst)) {
        if (I + 1 >= Function.size())
          return false;
        Worklist.emplace_back(I + 1, LiveMask);
      } else if (!LS.MIA->isUnconditionalBranch(DI.Inst)) {
        return false;
      }
      continue;
    }

    if ((!IsKnownNonControlFlow &&
         (Desc.isTerminator() ||
          LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))) ||
        I + 1 >= Function.size()) {
      if (PairIsAbiVcc)
        log() << "hotswap: VCC lifetime proof reached opaque control flow at 0x"
              << utohexstr(DI.Offset) << " (" << DI.Mnemonic << ")\n";
      return false;
    }
    Worklist.emplace_back(I + 1, LiveMask);
  }
  return true;
}

static std::optional<std::array<MCRegister, 2>>
findNumberedSgprPair(const LLVMState &LS, unsigned SgprBase) {
  if (!LS.MRI || (SgprBase & 1u) != 0 ||
      SgprBase == std::numeric_limits<unsigned>::max())
    return std::nullopt;

  std::array<MCRegister, 2> PairRegs;
  for (unsigned Reg = 1, End = LS.MRI->getNumRegs(); Reg < End; ++Reg) {
    std::optional<unsigned> Index = numberedSgprIndex(*LS.MRI, MCRegister(Reg));
    if (Index && *Index >= SgprBase && *Index <= SgprBase + 1)
      PairRegs[*Index - SgprBase] = MCRegister(Reg);
  }
  if (!PairRegs[0].isValid() || !PairRegs[1].isValid())
    return std::nullopt;
  return PairRegs;
}

bool isSgprPairDeadFrom(ArrayRef<InternalDecodedInst> Function,
                        size_t ResumeIndex, unsigned SgprBase,
                        const LLVMState &LS, ArrayRef<uint8_t> Text) {
  std::optional<std::array<MCRegister, 2>> PairRegs =
      findNumberedSgprPair(LS, SgprBase);
  if (!PairRegs)
    return false;
  return isRegisterPairDeadFrom(Function, ResumeIndex, *PairRegs, Text, LS);
}

using NumberedSgprMask = std::array<uint64_t, 2>;

static NumberedSgprMask allNumberedSgprs(unsigned Limit) {
  NumberedSgprMask Result{};
  for (unsigned I = 0; I < Limit; ++I)
    Result[I / 64] |= uint64_t{1} << (I % 64);
  return Result;
}

static void addOverlappingNumberedSgprs(NumberedSgprMask &Mask, MCRegister Reg,
                                        ArrayRef<MCRegister> NumberedSgprs,
                                        const MCRegisterInfo &MRI) {
  if (!Reg.isValid())
    return;
  for (unsigned I = 0; I != NumberedSgprs.size(); ++I)
    if (MRI.regsOverlap(Reg.id(), NumberedSgprs[I].id()))
      Mask[I / 64] |= uint64_t{1} << (I % 64);
}

static bool masksEqual(const NumberedSgprMask &A, const NumberedSgprMask &B) {
  return A[0] == B[0] && A[1] == B[1];
}

static std::optional<SiteDeadSgprFunctionFacts>
computeSiteDeadSgprFunctionFacts(PatchContext &Ctx,
                                 const ElfView::FunctionTextRange &Range) {
  if (!Ctx.LS.MCII || !Ctx.LS.MRI || !Ctx.LS.MIA || Ctx.Config.MaxSgprs == 0 ||
      Ctx.Config.MaxSgprs > 128)
    return std::nullopt;

  std::vector<InternalDecodedInst>::const_iterator First =
      llvm::lower_bound(Ctx.Decoded, Range.Begin,
                        [](const InternalDecodedInst &DI, uint64_t Offset) {
                          return DI.Offset < Offset;
                        });
  std::vector<InternalDecodedInst>::const_iterator After =
      llvm::lower_bound(Ctx.Decoded, Range.End,
                        [](const InternalDecodedInst &DI, uint64_t Offset) {
                          return DI.Offset < Offset;
                        });
  if (First == After)
    return std::nullopt;
  ArrayRef<InternalDecodedInst> Function(&*First,
                                         static_cast<size_t>(After - First));
  const unsigned Count = Function.size();

  SmallVector<MCRegister, 128> NumberedSgprs(Ctx.Config.MaxSgprs);
  for (unsigned Reg = 1, End = Ctx.LS.MRI->getNumRegs(); Reg < End; ++Reg) {
    std::optional<unsigned> Index =
        numberedSgprIndex(*Ctx.LS.MRI, MCRegister(Reg));
    if (Index && *Index < NumberedSgprs.size())
      NumberedSgprs[*Index] = MCRegister(Reg);
  }
  if (!llvm::all_of(NumberedSgprs,
                    [](MCRegister Reg) { return Reg.isValid(); }))
    return std::nullopt;

  DenseMap<uint64_t, unsigned> OffsetToIndex;
  for (unsigned I = 0; I != Count; ++I)
    OffsetToIndex.try_emplace(Function[I].Offset, I);

  DenseSet<uint64_t> DirectTargets;
  DenseSet<uint64_t> NoTargets;
  BitVector ForbiddenResume(Count);
  for (unsigned I = 3; I < Count; ++I)
    if (std::optional<uint64_t> Target = evaluateMaterializedSetPcTarget(
            Function, I, NoTargets, ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize),
            Ctx.LS)) {
      DirectTargets.insert(*Target);
      // A resume inside get-PC/add/set-PC is invalid even when a direct target
      // into that sequence prevents recognizing it as a CFG edge below.
      ForbiddenResume.set(I - 2, I + 1);
    }
  for (const InternalDecodedInst &DI : Function) {
    if ((!Ctx.LS.MIA->isBranch(DI.Inst) && !Ctx.LS.MIA->isCall(DI.Inst)) ||
        Ctx.LS.MIA->isIndirectBranch(DI.Inst) || Ctx.LS.MIA->isReturn(DI.Inst))
      continue;
    bool HasImmediate = false;
    for (const MCOperand &Operand : DI.Inst)
      HasImmediate |= Operand.isImm();
    if (!HasImmediate)
      continue;
    if (std::optional<uint64_t> Target =
            evaluateDirectControlFlowTarget(DI, Ctx.LS))
      DirectTargets.insert(*Target);
  }

  DenseMap<unsigned, unsigned> MaterializedSuccessor;
  for (unsigned I = 3; I < Count; ++I) {
    std::optional<uint64_t> Target = evaluateMaterializedSetPcTarget(
        Function, I, DirectTargets, ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize),
        Ctx.LS);
    if (!Target)
      continue;
    DenseMap<uint64_t, unsigned>::const_iterator TargetIt =
        OffsetToIndex.find(*Target);
    if (TargetIt == OffsetToIndex.end())
      continue;
    MaterializedSuccessor.try_emplace(I, TargetIt->second);
  }

  std::vector<SmallVector<unsigned, 2>> Successors(Count);
  std::vector<SmallVector<unsigned, 2>> Predecessors(Count);
  for (unsigned I = 0; I != Count; ++I) {
    const InternalDecodedInst &DI = Function[I];
    DenseMap<unsigned, unsigned>::const_iterator Materialized =
        MaterializedSuccessor.find(I);
    if (Materialized != MaterializedSuccessor.end()) {
      Successors[I].push_back(Materialized->second);
    } else if (DI.Mnemonic == "<unknown>" || Ctx.LS.MIA->isCall(DI.Inst) ||
               DI.Mnemonic == "s_endpgm" || Ctx.LS.MIA->isReturn(DI.Inst)) {
      // Calls and exits are fail-closed lifetime boundaries.
    } else if (Ctx.LS.MIA->isBranch(DI.Inst)) {
      uint64_t Target = 0;
      if (!Ctx.LS.MIA->isIndirectBranch(DI.Inst) &&
          Ctx.LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target)) {
        DenseMap<uint64_t, unsigned>::const_iterator TargetIt =
            OffsetToIndex.find(Target);
        if (TargetIt != OffsetToIndex.end()) {
          Successors[I].push_back(TargetIt->second);
          if (Ctx.LS.MIA->isConditionalBranch(DI.Inst) && I + 1 < Count)
            Successors[I].push_back(I + 1);
          else if (!Ctx.LS.MIA->isUnconditionalBranch(DI.Inst))
            Successors[I].clear();
        }
      }
    } else {
      const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
      if (!Desc.isTerminator() &&
          !Ctx.LS.MIA->mayAffectControlFlow(DI.Inst, *Ctx.LS.MRI) &&
          I + 1 < Count)
        Successors[I].push_back(I + 1);
    }
    for (unsigned Succ : Successors[I])
      Predecessors[Succ].push_back(I);
  }

  NumberedSgprMask All = allNumberedSgprs(Ctx.Config.MaxSgprs);
  std::vector<NumberedSgprMask> Defs(Count);
  std::vector<NumberedSgprMask> Uses(Count);
  for (unsigned I = 0; I != Count; ++I) {
    const InternalDecodedInst &DI = Function[I];
    if (DI.Mnemonic == "<unknown>") {
      Uses[I] = All;
      continue;
    }
    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    unsigned NumDefs =
        std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
    for (unsigned OpIdx = 0; OpIdx < DI.Inst.getNumOperands(); ++OpIdx) {
      const MCOperand &Op = DI.Inst.getOperand(OpIdx);
      if (!Op.isReg() || !Op.getReg())
        continue;
      addOverlappingNumberedSgprs(OpIdx < NumDefs ? Defs[I] : Uses[I],
                                  MCRegister(Op.getReg()), NumberedSgprs,
                                  *Ctx.LS.MRI);
    }

    bool Malformed = false;
    for (unsigned OpIdx = NumDefs; OpIdx < Desc.getNumOperands(); ++OpIdx) {
      int TiedTo = Desc.getOperandConstraint(OpIdx, MCOI::TIED_TO);
      if (TiedTo < 0)
        continue;
      if (static_cast<unsigned>(TiedTo) >= NumDefs ||
          static_cast<unsigned>(TiedTo) >= DI.Inst.getNumOperands()) {
        Malformed = true;
        break;
      }
      const MCOperand &Def = DI.Inst.getOperand(TiedTo);
      if (!Def.isReg() || !Def.getReg()) {
        Malformed = true;
        break;
      }
      addOverlappingNumberedSgprs(Uses[I], MCRegister(Def.getReg()),
                                  NumberedSgprs, *Ctx.LS.MRI);
    }
    for (unsigned OpIdx = DI.Inst.getNumOperands();
         !Malformed && OpIdx < Desc.getNumOperands(); ++OpIdx) {
      const MCOperandInfo &Missing = Desc.operands()[OpIdx];
      if (Missing.RegClass < 0)
        continue;
      bool MatchedDefinition = false;
      for (unsigned DefIdx = 0; DefIdx < NumDefs; ++DefIdx) {
        if (Desc.operands()[DefIdx].RegClass != Missing.RegClass)
          continue;
        const MCOperand &Def = DI.Inst.getOperand(DefIdx);
        if (!Def.isReg() || !Def.getReg()) {
          Malformed = true;
          break;
        }
        addOverlappingNumberedSgprs(Uses[I], MCRegister(Def.getReg()),
                                    NumberedSgprs, *Ctx.LS.MRI);
        MatchedDefinition = true;
      }
      if (!MatchedDefinition)
        Malformed = true;
    }
    if (Malformed) {
      Uses[I] = All;
      continue;
    }
    for (MCPhysReg Reg : Desc.implicit_uses())
      addOverlappingNumberedSgprs(Uses[I], MCRegister(Reg), NumberedSgprs,
                                  *Ctx.LS.MRI);
    for (MCPhysReg Reg : Desc.implicit_defs())
      addOverlappingNumberedSgprs(Defs[I], MCRegister(Reg), NumberedSgprs,
                                  *Ctx.LS.MRI);
  }

  std::vector<NumberedSgprMask> SafeBefore(Count, All);
  SmallVector<unsigned, 128> Worklist;
  BitVector Queued(Count, true);
  for (unsigned I = 0; I != Count; ++I)
    Worklist.push_back(I);
  while (!Worklist.empty()) {
    unsigned I = Worklist.pop_back_val();
    Queued.reset(I);
    NumberedSgprMask SafeOut{};
    if (!Successors[I].empty()) {
      SafeOut = All;
      for (unsigned Succ : Successors[I]) {
        SafeOut[0] &= SafeBefore[Succ][0];
        SafeOut[1] &= SafeBefore[Succ][1];
      }
    }
    NumberedSgprMask New{(SafeOut[0] | Defs[I][0]) & ~Uses[I][0],
                         (SafeOut[1] | Defs[I][1]) & ~Uses[I][1]};
    New[0] &= All[0];
    New[1] &= All[1];
    if (masksEqual(New, SafeBefore[I]))
      continue;
    SafeBefore[I] = New;
    for (unsigned Pred : Predecessors[I])
      if (!Queued.test(Pred)) {
        Queued.set(Pred);
        Worklist.push_back(Pred);
      }
  }

  SiteDeadSgprFunctionFacts Facts;
  Facts.Begin = Range.Begin;
  Facts.End = Range.End;
  Facts.GlobalFirst = First - Ctx.Decoded.cbegin();
  unsigned HighWatermark = 0;
  for (const InternalDecodedInst &DI : Function) {
    for (const MCOperand &Op : DI.Inst) {
      if (!Op.isReg() || !Op.getReg())
        continue;
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Op.getReg()),
                                           Ctx.Config.MaxSgprs, HighWatermark,
                                           "site-dead SGPR analysis"))
        return std::nullopt;
    }
    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    for (MCPhysReg Reg : Desc.implicit_uses())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs, HighWatermark,
                                           "site-dead SGPR analysis"))
        return std::nullopt;
    for (MCPhysReg Reg : Desc.implicit_defs())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs, HighWatermark,
                                           "site-dead SGPR analysis"))
        return std::nullopt;
  }
  Facts.NumberedLimit = HighWatermark;
  std::string Owner =
      Ctx.Elf.findKernelAtAddress(Range.Begin + Ctx.Elf.textAddr());
  if (!Owner.empty()) {
    std::optional<unsigned> Declared = Ctx.Elf.getKernelSgprCount(Owner);
    if (!Declared || *Declared < 2)
      return std::nullopt;
    // The descriptor count may include VCC. Subtracting it unconditionally
    // gives a conservative numbered-register limit.
    Facts.NumberedLimit = std::max(
        Facts.NumberedLimit, std::min(*Declared - 2, Ctx.Config.MaxSgprs));
  }
  Facts.SafeBefore = std::move(SafeBefore);
  Facts.ForbiddenResume = std::move(ForbiddenResume);
  return Facts;
}

void precomputeSiteDeadSgprFacts(PatchContext &Ctx) {
  const uint64_t TextAddr = Ctx.Elf.textAddr();
  for (const ElfView::FunctionTextRange &Absolute :
       Ctx.Elf.functionTextRanges()) {
    if (Absolute.Begin < TextAddr || Absolute.End <= Absolute.Begin)
      continue;
    uint64_t Begin = Absolute.Begin - TextAddr;
    uint64_t End = std::min(Absolute.End - TextAddr, Ctx.TextSize);
    if (Begin >= End)
      continue;
    std::pair<uint64_t, uint64_t> Key{Begin, End};
    if (Ctx.SiteDeadSgprFacts.find(Key) != Ctx.SiteDeadSgprFacts.end())
      continue;
    ElfView::FunctionTextRange Relative{Begin, End, Absolute.Symbol,
                                        Absolute.Symtab};
    std::optional<SiteDeadSgprFunctionFacts> Facts =
        computeSiteDeadSgprFunctionFacts(Ctx, Relative);
    if (Facts)
      Ctx.SiteDeadSgprFacts.try_emplace(Key, std::move(*Facts));
  }
}

std::optional<BitVector> getSiteDeadNumberedSgprs(PatchContext &Ctx,
                                                  uint64_t InstOffset,
                                                  uint32_t InstSize) {
  std::optional<ElfView::FunctionTextRange> Range =
      Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset);
  if (!Range)
    return std::nullopt;
  using FunctionKey = std::pair<uint64_t, uint64_t>;
  FunctionKey Key{Range->Begin, Range->End};
  auto It = Ctx.SiteDeadSgprFacts.find(Key);
  // Facts are deliberately computed from the immutable decoded stream before
  // any instruction is relabeled as replaced.
  if (It == Ctx.SiteDeadSgprFacts.end())
    return std::nullopt;

  std::optional<uint64_t> ResumeOffset =
      checkedAddUint64(InstOffset, InstSize, "site-dead SGPR resume offset");
  if (!ResumeOffset)
    return std::nullopt;
  const SiteDeadSgprFunctionFacts &Facts = It->second;
  std::vector<InternalDecodedInst>::const_iterator First =
      Ctx.Decoded.cbegin() + Facts.GlobalFirst;
  std::vector<InternalDecodedInst>::const_iterator After =
      First + Facts.SafeBefore.size();
  std::vector<InternalDecodedInst>::const_iterator Resume =
      std::lower_bound(First, After, *ResumeOffset,
                       [](const InternalDecodedInst &DI, uint64_t Offset) {
                         return DI.Offset < Offset;
                       });
  if (Resume == After || Resume->Offset != *ResumeOffset)
    return std::nullopt;
  size_t ResumeIndex = Resume - First;
  BitVector Result(Ctx.Config.MaxSgprs);
  if (Facts.ForbiddenResume.test(ResumeIndex))
    return Result;
  const NumberedSgprMask &Mask = Facts.SafeBefore[ResumeIndex];
  for (unsigned I = 0; I != Ctx.Config.MaxSgprs; ++I)
    if (Mask[I / 64] & (uint64_t{1} << (I % 64)))
      Result.set(I);
  return Result;
}

static std::optional<std::pair<ArrayRef<InternalDecodedInst>, size_t>>
findFunctionContinuation(const PatchContext &Ctx, uint64_t InstOffset,
                         uint32_t InstSize) {
  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset);
  if (!FunctionRange)
    return std::nullopt;

  std::vector<InternalDecodedInst>::const_iterator FunctionFirst =
      llvm::lower_bound(Ctx.Decoded, FunctionRange->Begin,
                        [](const InternalDecodedInst &DI, uint64_t Offset) {
                          return DI.Offset < Offset;
                        });
  std::vector<InternalDecodedInst>::const_iterator FunctionAfter =
      llvm::lower_bound(Ctx.Decoded, FunctionRange->End,
                        [](const InternalDecodedInst &DI, uint64_t Offset) {
                          return DI.Offset < Offset;
                        });
  if (FunctionFirst == FunctionAfter)
    return std::nullopt;

  std::optional<uint64_t> ResumeOffset =
      checkedAddUint64(InstOffset, InstSize, "site-dead pair resume offset");
  if (!ResumeOffset)
    return std::nullopt;
  std::vector<InternalDecodedInst>::const_iterator Resume =
      std::lower_bound(FunctionFirst, FunctionAfter, *ResumeOffset,
                       [](const InternalDecodedInst &DI, uint64_t Offset) {
                         return DI.Offset < Offset;
                       });
  if (Resume == FunctionAfter || Resume->Offset != *ResumeOffset)
    return std::nullopt;

  ArrayRef<InternalDecodedInst> Function(
      &*FunctionFirst, static_cast<size_t>(FunctionAfter - FunctionFirst));
  return std::pair{Function, static_cast<size_t>(Resume - FunctionFirst)};
}

static std::optional<std::array<MCRegister, 2>>
findVccLoHiPair(const LLVMState &LS) {
  if (!LS.MRI)
    return std::nullopt;
  std::array<MCRegister, 2> Pair;
  for (unsigned Reg = 1, End = LS.MRI->getNumRegs(); Reg < End; ++Reg) {
    StringRef Name(LS.MRI->getName(Reg));
    if (Name == "VCC_LO")
      Pair[0] = MCRegister(Reg);
    else if (Name == "VCC_HI")
      Pair[1] = MCRegister(Reg);
  }
  if (!Pair[0].isValid() || !Pair[1].isValid())
    return std::nullopt;
  return Pair;
}

bool isVccPairDeadFrom(ArrayRef<InternalDecodedInst> Function,
                       size_t ResumeIndex, const LLVMState &LS,
                       ArrayRef<uint8_t> Text) {
  std::optional<std::array<MCRegister, 2>> Pair = findVccLoHiPair(LS);
  return Pair && isRegisterPairDeadFrom(Function, ResumeIndex, *Pair, Text, LS,
                                        /*PairIsAbiVcc=*/true);
}

static std::optional<bool> replacementTouchesVcc(ArrayRef<uint8_t> Replacement,
                                                 const LLVMState &LS) {
  if (!LS.MCII || !LS.MRI)
    return std::nullopt;
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Replacement.data(), Replacement.size(), LS, Decoded))
    return std::nullopt;
  for (const InternalDecodedInst &DI : Decoded) {
    if (DI.Mnemonic == "<unknown>")
      return std::nullopt;
    for (const MCOperand &Op : DI.Inst)
      if (Op.isReg() && Op.getReg() &&
          isVccRegister(LS, MCRegister(Op.getReg())))
        return true;
    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    for (MCPhysReg Reg : Desc.implicit_uses())
      if (isVccRegister(LS, MCRegister(Reg)))
        return true;
    for (MCPhysReg Reg : Desc.implicit_defs())
      if (isVccRegister(LS, MCRegister(Reg)))
        return true;
  }
  return false;
}

static bool isVccPairDeadAfter(PatchContext &Ctx, uint64_t InstOffset,
                               uint32_t InstSize) {
  std::optional<std::pair<ArrayRef<InternalDecodedInst>, size_t>> Continuation =
      findFunctionContinuation(Ctx, InstOffset, InstSize);
  if (!Continuation ||
      !llvm::any_of(Continuation->first, [&](const InternalDecodedInst &DI) {
        return instructionUsesVcc(Ctx.LS, DI);
      }))
    return false;

  // Using VCC for the trampoline must not introduce a new special-register
  // allocation. An original VCC reference in the owning function ensures the
  // compiler already propagated the VCC resource requirement to its callers.
  return isVccPairDeadFrom(Continuation->first, Continuation->second, Ctx.LS,
                           ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize));
}

static bool findSiteDeadVccPair(PatchContext &Ctx, uint64_t InstOffset,
                                uint32_t InstSize,
                                ArrayRef<uint8_t> Replacement) {
  std::optional<bool> ReplacementTouches =
      replacementTouchesVcc(Replacement, Ctx.LS);
  if (!ReplacementTouches || *ReplacementTouches ||
      !isVccPairDeadAfter(Ctx, InstOffset, InstSize))
    return false;
  log() << "hotswap: safe far return: reusing site-dead vcc after 0x"
        << utohexstr(InstOffset) << "\n";
  return true;
}

static std::optional<unsigned>
findSiteDeadOriginalPair(PatchContext &Ctx, uint64_t InstOffset,
                         uint32_t InstSize, ArrayRef<uint8_t> Replacement) {
  std::optional<std::pair<ArrayRef<InternalDecodedInst>, size_t>> Continuation =
      findFunctionContinuation(Ctx, InstOffset, InstSize);
  if (!Continuation)
    return std::nullopt;

  std::optional<BitVector> ReplacementTouched =
      collectTouchedNumberedSgprs(Replacement, Ctx.Config.MaxSgprs, Ctx.LS);
  if (!ReplacementTouched)
    return std::nullopt;
  std::optional<BitVector> SiteDead =
      getSiteDeadNumberedSgprs(Ctx, InstOffset, InstSize);
  if (!SiteDead)
    return std::nullopt;
  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset);
  if (!FunctionRange)
    return std::nullopt;
  auto Facts = Ctx.SiteDeadSgprFacts.find(
      std::pair{FunctionRange->Begin, FunctionRange->End});
  if (Facts == Ctx.SiteDeadSgprFacts.end())
    return std::nullopt;
  unsigned NumberedLimit = Facts->second.NumberedLimit;

  if (NumberedLimit < 2)
    return std::nullopt;

  unsigned Pair = (NumberedLimit - 2) & ~1u;
  for (;;) {
    if (!ReplacementTouched->test(Pair) &&
        !ReplacementTouched->test(Pair + 1) && SiteDead->test(Pair) &&
        SiteDead->test(Pair + 1)) {
#ifndef NDEBUG
      assert(isSgprPairDeadFrom(
          Continuation->first, Continuation->second, Pair, Ctx.LS,
          ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize))));
#endif
      log() << "hotswap: safe far return: reusing original site-dead s[" << Pair
            << ':' << Pair + 1 << "] after 0x" << utohexstr(InstOffset)
            << " (defined before every exit)\n";
      return Pair;
    }
    if (Pair < 2)
      break;
    Pair -= 2;
  }
  return std::nullopt;
}

static SafeSgprUsageSummary
summarizeSafeSgprUsage(PatchContext &Ctx,
                       ArrayRef<InternalDecodedInst> Instructions,
                       StringRef Context) {
  SafeSgprUsageSummary Summary;
  for (const InternalDecodedInst &DI : Instructions) {
    if (DI.Mnemonic == "<unknown>") {
      log() << "hotswap: error: " << Context
            << ": undecoded instruction prevents SGPR usage proof at 0x"
            << utohexstr(DI.Offset) << "\n";
      Summary.Valid = false;
      return Summary;
    }
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
                         bool DiagnoseFailure) {
  if (Count == 0 || Alignment == 0 || (Alignment & (Alignment - 1)) != 0) {
    if (DiagnoseFailure)
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
    if (DiagnoseFailure)
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
      if (DiagnoseFailure)
        log() << "hotswap: error: " << Context
              << ": failed to read SGPR count for kernel " << Owner << "\n";
      return std::nullopt;
    }
    if (UsesVcc && *Declared < VccSgprs) {
      if (DiagnoseFailure)
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
    if (!Ctx.Elf.kernelDescriptorCacheIsComplete()) {
      if (DiagnoseFailure)
        log() << "hotswap: error: " << Context
              << ": kernel descriptor set is incomplete or ambiguous\n";
      return std::nullopt;
    }
    for (const KernelDescriptorInfo &KD : Ctx.Elf.kernelDescriptors()) {
      std::optional<unsigned> Declared =
          Ctx.Elf.getKernelSgprCount(KD.KernelName);
      if (!Declared) {
        if (DiagnoseFailure)
          log() << "hotswap: error: " << Context
                << ": failed to read SGPR count for kernel " << KD.KernelName
                << "\n";
        return std::nullopt;
      }
      HighWatermark = std::max(HighWatermark, *Declared);
    }
  }

  if (HighWatermark > std::numeric_limits<unsigned>::max() - (Alignment - 1)) {
    if (DiagnoseFailure)
      log() << "hotswap: error: " << Context
            << ": SGPR alignment calculation overflows unsigned\n";
    return std::nullopt;
  }
  unsigned Base = (HighWatermark + Alignment - 1) & ~(Alignment - 1);
  if (Base > Ctx.Config.MaxSgprs || Count > Ctx.Config.MaxSgprs - Base) {
    if (DiagnoseFailure)
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
  if (!Ctx.Elf.kernelDescriptorCacheIsComplete()) {
    log() << "hotswap: error: " << Context
          << ": kernel descriptor set is incomplete or ambiguous\n";
    return false;
  }
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

  // GFX1250 has two non-numbered VCC SGPRs (GCNSubtarget::getNumExtraSGPRs).
  // Always include them in the metadata requirement. This may conservatively
  // overstate a kernel that does not use VCC, but never mistakes VCC for
  // numbered s0-s105 registers.
  constexpr unsigned VccSgprs = 2;
  unsigned RequiredSgprs = Block.Base + Block.Count + VccSgprs;
  SmallVector<std::pair<StringRef, unsigned>, 4> PendingCharges;
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
    if (*Current < RequiredSgprs)
      PendingCharges.emplace_back(KD.KernelName, RequiredSgprs - *Current);
  }

  if (!ChargedOwner) {
    log() << "hotswap: error: " << Context << ": kernel '" << Owner
          << "' has no descriptor\n";
    return false;
  }
  for (const auto &[KernelName, ExtraSgprs] : PendingCharges) {
    KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
    Stats.ExtraSgprs = std::max(Stats.ExtraSgprs, ExtraSgprs);
  }
  return true;
}

static std::optional<SafeSgprScratchBlock>
reserveSafeFarReturn(PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
                     ArrayRef<uint8_t> Replacement,
                     bool DiagnoseFailure = true) {
  if (std::optional<unsigned> ReusedPair =
          findSiteDeadOriginalPair(Ctx, InstOffset, InstSize, Replacement))
    return SafeSgprScratchBlock{*ReusedPair, 2,
                                /*IsSiteProven=*/true};

  if (findSiteDeadVccPair(Ctx, InstOffset, InstSize, Replacement))
    return SafeSgprScratchBlock{Gfx1250MaxSgprs, 2,
                                /*IsSiteProven=*/true};

  std::optional<SafeSgprScratchBlock> Scratch =
      findSafeSgprScratchBlock(Ctx, InstOffset, /*Count=*/2, /*Alignment=*/2,
                               "safe far return", DiagnoseFailure);
  if (!Scratch)
    return std::nullopt;
  if (!commitSafeSgprScratchBlock(Ctx, InstOffset, *Scratch, "safe far return"))
    return std::nullopt;
  return Scratch;
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

bool canEmitShortTrampoline(const PatchContext &Ctx, uint64_t InstOffset,
                            uint32_t InstSize, uint64_t ReplacementSize) {
  std::optional<uint64_t> PoolStart =
      checkedAddUint64(Ctx.PoolBaseOffset, Ctx.QueuedTrampolineBytes,
                       "short trampoline pool position");
  std::optional<uint64_t> PoolReturn =
      PoolStart ? checkedAddUint64(*PoolStart, ReplacementSize,
                                   "short trampoline return position")
                : std::nullopt;
  std::optional<uint64_t> ReturnTo = checkedAddUint64(
      InstOffset, InstSize, "short trampoline source return target");
  return PoolStart && PoolReturn && ReturnTo &&
         isSBranchReachable(InstOffset, *PoolStart) &&
         isSBranchReachable(*PoolReturn, *ReturnTo);
}

/// Queue a deferred trampoline for [\p InstOffset, +\p InstSize) with
/// \p Replacement as its body; fixupTrampolineBranches fills in the edges once
/// the pool layout is known. A site beyond s_branch reach of the appended pool
/// uses an SCC-neutral get-PC/add/set-PC sequence on the backward edge.
/// Adjacent far sites are coalesced after patching to reduce gateway pressure.
/// Every far source edge then uses a short branch to nearby safe NOP padding;
/// that gateway uses the pre-Gen5 SGPR-backed set-PC sequence. No source or
/// return edge executes gfx1250's broken s_add_pc_i64 instruction.
static bool emitToTrampolineRaw(PatchContext &Ctx, uint64_t InstOffset,
                                uint32_t InstSize,
                                ArrayRef<uint8_t> Replacement,
                                ReplacementPlacement Placement,
                                bool DiagnoseFailure) {
  const bool AllowGlobalBody = Placement != ReplacementPlacement::Default;
  // This trampoline lands at the appended pool base after every trampoline
  // already queued. QueuedTrampolineBytes is a conservative upper bound: far
  // entries reserve the island appended during final layout and enough room
  // for straight-line source-window growth. The actual pool position can only
  // be earlier, which can only improve both branch directions.
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
  const bool Far =
      !canEmitShortTrampoline(Ctx, InstOffset, InstSize, Replacement.size());

  uint64_t ReturnReserve = Far ? SetPcReturnReserveBytes : MinInstSize;
  std::optional<uint64_t> TrampolineSize = checkedAddUint64(
      Replacement.size(), ReturnReserve, "queued trampoline size");
  if (!TrampolineSize)
    return false;
  uint64_t LayoutReserve = *TrampolineSize;
  if (Far) {
    std::optional<uint64_t> WithIsland =
        checkedAddUint64(LayoutReserve, PoolBranchIslandBytes,
                         "queued trampoline branch-island reserve");
    if (!WithIsland)
      return false;
    LayoutReserve = *WithIsland;

    // Expansion stops as soon as the source window reaches the set-PC forward
    // sequence size. Instruction sizes are dword multiples, so the last copied
    // instruction starts no later than one dword below that threshold.
    if (InstSize < SetPcForwardSequenceBytes) {
      std::optional<uint64_t> MaxExpandedSize = checkedAddUint64(
          SetPcForwardSequenceBytes - MinInstSize, Ctx.MaxDecodedInstSize,
          "queued trampoline source-growth bound");
      if (!MaxExpandedSize || *MaxExpandedSize < InstSize)
        return false;
      std::optional<uint64_t> WithGrowth =
          checkedAddUint64(LayoutReserve, *MaxExpandedSize - InstSize,
                           "queued trampoline source-growth reserve");
      if (!WithGrowth)
        return false;
      LayoutReserve = *WithGrowth;
    }
  }
  std::optional<uint64_t> QueuedBytes = checkedAddUint64(
      Ctx.QueuedTrampolineBytes, LayoutReserve, "queued trampoline byte count");
  if (!QueuedBytes)
    return false;

  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
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
    std::optional<SafeSgprScratchBlock> Scratch = reserveSafeFarReturn(
        Ctx, InstOffset, InstSize, Replacement, /*DiagnoseFailure=*/false);
    if (!Scratch) {
      uint64_t Needed = getNopSledBytesNeeded(Ctx, InstOffset, InstSize,
                                              Replacement, Placement);
      NopSledUse SledUse =
          AllowGlobalBody ? NopSledUse::RelocationBody : NopSledUse::OwnerBody;
      if (NopSled *Sled =
              findNearestSled(Ctx.NopSleds, InstOffset, Needed, SledUse)) {
        if (emitToNopSled(Ctx, *Sled, InstOffset, InstSize, Replacement,
                          Placement)) {
          log() << "hotswap: safe far return: used local NOP sled at 0x"
                << utohexstr(Sled->WritePos - Needed) << " for site 0x"
                << utohexstr(InstOffset) << "\n";
          return true;
        }
      }
      if (AllowGlobalBody &&
          emitDs2ThroughNopSledGateways(Ctx, InstOffset, InstSize, Replacement,
                                        Placement))
        return true;
      if (DiagnoseFailure)
        (void)reserveSafeFarReturn(Ctx, InstOffset, InstSize, Replacement,
                                   /*DiagnoseFailure=*/true);
      return declineFar("no safe SGPR pair or fallback placement");
    }
    T.Bytes.insert(T.Bytes.end(), SetPcReturnReserveBytes, uint8_t{0});
    T.Long = true;
    T.UsesSetPCBack = true;
    T.LongBranchSgprBase = Scratch->Base;
    T.LongBranchScratchIsSiteProven = Scratch->IsSiteProven;
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

struct PreparedSiteReplacement {
  SmallVector<uint8_t> Bytes;
  bool ComposesWmmaHazard = false;
};

static std::optional<PreparedSiteReplacement>
prepareSiteReplacement(const PatchContext &Ctx, uint64_t InstOffset,
                       uint32_t InstSize, ArrayRef<uint8_t> Replacement,
                       bool DiagnoseFailure) {
  std::optional<uint64_t> InstEnd = checkedAddUint64(
      InstOffset, InstSize, "site replacement source interval");
  if (!InstEnd)
    return std::nullopt;

  auto IsActive = [](const SiteReplacementState &State) {
    return State.Committed || State.RequiredLeadingVNops != 0;
  };
  auto DiagnoseOverlap = [&](uint64_t Offset,
                             const SiteReplacementState &State) {
    std::optional<uint64_t> End = checkedAddUint64(
        Offset, State.OriginalSize, "owned replacement source interval");
    if (!End)
      return false;
    if (DiagnoseFailure)
      log() << "hotswap: error: replacement source [0x" << utohexstr(InstOffset)
            << ", 0x" << utohexstr(*InstEnd) << ") overlaps reserved source [0x"
            << utohexstr(Offset) << ", 0x" << utohexstr(*End) << ")\n";
    return true;
  };

  // Replacement intervals are inserted only after a successful emission or
  // by whole-pass precomputation, and central ownership keeps them disjoint.
  // An ordered map therefore needs only the predecessor plus entries whose
  // starts fall inside the requested interval, instead of scanning every
  // patched instruction in a large code object.
  auto It = Ctx.SiteReplacements.lower_bound(InstOffset);
  if (It != Ctx.SiteReplacements.begin()) {
    auto Prev = std::prev(It);
    if (IsActive(Prev->second)) {
      std::optional<uint64_t> PrevEnd =
          checkedAddUint64(Prev->first, Prev->second.OriginalSize,
                           "owned replacement source interval");
      if (!PrevEnd)
        return std::nullopt;
      if (*PrevEnd > InstOffset) {
        DiagnoseOverlap(Prev->first, Prev->second);
        return std::nullopt;
      }
    }
  }

  const SiteReplacementState *Exact = nullptr;
  if (It != Ctx.SiteReplacements.end() && It->first == InstOffset) {
    Exact = &It->second;
    ++It;
  }
  for (; It != Ctx.SiteReplacements.end() && It->first < *InstEnd; ++It) {
    if (!IsActive(It->second))
      continue;
    DiagnoseOverlap(It->first, It->second);
    return std::nullopt;
  }

  if (Exact && Exact->Committed) {
    if (DiagnoseFailure)
      log() << "hotswap: error: replacement source at 0x"
            << utohexstr(InstOffset) << " already has a committed owner\n";
    return std::nullopt;
  }
  if (Exact && Exact->OriginalSize != InstSize) {
    if (DiagnoseFailure)
      log() << "hotswap: error: replacement source at 0x"
            << utohexstr(InstOffset) << " has reserved size "
            << Exact->OriginalSize << " but owner requested " << InstSize
            << " bytes\n";
    return std::nullopt;
  }

  PreparedSiteReplacement Prepared;
  const unsigned VNops = Exact ? Exact->RequiredLeadingVNops : 0;
  if (VNops != 0) {
    SmallVector<uint8_t> VNopBytes = assembleSingleInst("v_nop", Ctx.LS);
    if (VNopBytes.size() != MinInstSize) {
      if (DiagnoseFailure)
        log() << "hotswap: error: could not encode v_nop requirement for "
                 "replacement source at 0x"
              << utohexstr(InstOffset) << "\n";
      return std::nullopt;
    }
    for (unsigned I = 0; I != VNops; ++I)
      Prepared.Bytes.append(VNopBytes.begin(), VNopBytes.end());
    Prepared.ComposesWmmaHazard = true;
  }
  Prepared.Bytes.append(Replacement.begin(), Replacement.end());
  return Prepared;
}

static bool commitSiteReplacement(PatchContext &Ctx, uint64_t InstOffset,
                                  uint32_t InstSize,
                                  const PreparedSiteReplacement &Prepared) {
  SiteReplacementState &State = Ctx.SiteReplacements[InstOffset];
  if (State.Committed ||
      (State.OriginalSize != 0 && State.OriginalSize != InstSize) ||
      (Prepared.ComposesWmmaHazard && State.RequiredLeadingVNops == 0) ||
      (!Prepared.ComposesWmmaHazard && State.RequiredLeadingVNops != 0)) {
    log() << "hotswap: error: replacement ownership changed while committing "
             "source at 0x"
          << utohexstr(InstOffset) << "\n";
    Ctx.RequiredPatchFailed = true;
    return false;
  }

  State.OriginalSize = InstSize;
  State.Committed = true;
  if (Prepared.ComposesWmmaHazard) {
    State.WmmaHazardComposed = true;
    ++Ctx.WmmaHazardsComposed;
    log() << "hotswap: WMMA co-exec requirement composed into replacement at "
             "0x"
          << utohexstr(InstOffset) << " (" << State.RequiredLeadingVNops
          << " leading v_nop(s))\n";
  }
  Ctx.RequiredPatchApplied = true;
  return true;
}

bool hasSiteReplacementReservation(const PatchContext &Ctx, uint64_t Offset) {
  auto It = Ctx.SiteReplacements.find(Offset);
  return It != Ctx.SiteReplacements.end() &&
         (It->second.Committed || It->second.RequiredLeadingVNops != 0);
}

[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    ArrayRef<uint8_t> Replacement,
                                    ReplacementPlacement Placement,
                                    bool DiagnoseFailure) {
  std::optional<PreparedSiteReplacement> Prepared = prepareSiteReplacement(
      Ctx, InstOffset, InstSize, Replacement, DiagnoseFailure);
  if (!Prepared)
    return false;
  if (!emitToTrampolineRaw(Ctx, InstOffset, InstSize, Prepared->Bytes,
                           Placement, DiagnoseFailure))
    return false;
  return commitSiteReplacement(Ctx, InstOffset, InstSize, *Prepared);
}

std::optional<uint64_t>
evaluateDirectControlFlowTarget(const InternalDecodedInst &DI,
                                const LLVMState &LS) {
  uint64_t Target = 0;
  if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target))
    return Target;

  // AMDGPUMCInstrAnalysis currently only accepts a PC-relative operand in
  // slot zero. GFX1250 s_call_i64 instead has its destination SGPR pair in
  // slot zero and its simm16 dword displacement in slot one. Keep this narrow
  // workaround private to hotswap.
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  if (DI.Inst.getOpcode() != LS.SCallI64Opcode || !Desc.isCall() ||
      DI.Inst.getNumOperands() != 2 || !DI.Inst.getOperand(0).isReg() ||
      !DI.Inst.getOperand(0).getReg() || !DI.Inst.getOperand(1).isImm())
    return std::nullopt;

  uint64_t Encoded =
      static_cast<uint64_t>(DI.Inst.getOperand(1).getImm()) & 0xFFFFu;
  int64_t DwordDelta = Encoded < 0x8000u
                           ? static_cast<int64_t>(Encoded)
                           : static_cast<int64_t>(Encoded) - 0x10000;
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

/// Collect every declared in-text function and kernel entry.  The external
/// subset is kept separately because an externally reachable alias invalidates
/// a local helper's call-defined return-link proof even when both symbols begin
/// at the same address.
static std::optional<DeclaredTextEntryInfo>
collectDeclaredTextEntries(const ElfView &Elf) {
  std::optional<uint64_t> TextEnd =
      checkedAddUint64(Elf.textAddr(), Elf.textSize(), "declared text end");
  if (!TextEnd)
    return std::nullopt;

  DeclaredTextEntryInfo Info;
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges())
    if (Range.Begin >= Elf.textAddr() && Range.Begin < *TextEnd) {
      const uint64_t Entry = Range.Begin - Elf.textAddr();
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
      const uint64_t Magnitude =
          Descriptor.EntryOffset == std::numeric_limits<int64_t>::min()
              ? uint64_t{1} << 63
              : static_cast<uint64_t>(-Descriptor.EntryOffset);
      EntryAddress = checkedSubUint64(Descriptor.VAddr, Magnitude,
                                      "kernel descriptor entry address");
    }
    if (!EntryAddress)
      return std::nullopt;
    if (*EntryAddress >= Elf.textAddr() && *EntryAddress < *TextEnd) {
      const uint64_t Entry = *EntryAddress - Elf.textAddr();
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
  MCRegister TargetRegister(TargetOperand.getReg());

  std::optional<size_t> AddIndex;
  int64_t AddImmediate = 0;
  for (size_t I = CallIndex; I != 0;) {
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
        TextAddr, Candidate.Offset, "PC-materialized call instruction");
    if (!GetPcAddress)
      return std::nullopt;
    std::optional<uint64_t> PcValue = checkedAddUint64(
        *GetPcAddress, Candidate.Size, "PC-materialized call PC value");
    if (!PcValue)
      return std::nullopt;
    // s_add_nc_u64 uses modulo-2^64 arithmetic. Casting the signed MC
    // immediate to uint64_t and adding it reproduces both positive and
    // negative literals, including INT64_MIN, without signed overflow.
    return PcMaterializedCallInfo{
        *PcValue + static_cast<uint64_t>(AddImmediate), Candidate.Offset,
        Call.Offset, MCRegister(Call.Inst.getOperand(0).getReg())};
  }
  return std::nullopt;
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

static std::optional<SmallVector<KnownCallSite, 4>> collectKnownCallSites(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextEnd,
    ArrayRef<std::optional<PcMaterializedCallInfo>> MaterializedCalls) {
  SmallVector<KnownCallSite, 4> Calls;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    std::optional<MCRegister> ReturnRegister = getCallReturnRegister(DI, LS);
    if (!ReturnRegister)
      continue;

    std::optional<uint64_t> Target;
    if (MaterializedCalls[I]) {
      uint64_t AbsoluteTarget = MaterializedCalls[I]->Target;
      if (AbsoluteTarget >= TextAddr && AbsoluteTarget < TextEnd)
        Target = AbsoluteTarget - TextAddr;
    } else {
      Target = getDirectTextTarget(DI, LS, TextAddr, TextEnd);
    }
    if (!Target)
      continue;

    std::optional<uint64_t> Continuation =
        checkedAddUint64(DI.Offset, DI.Size, "known call continuation address");
    if (!Continuation)
      return std::nullopt;
    Calls.push_back({I, *Target, *Continuation, *ReturnRegister});
  }
  return Calls;
}

static bool hasUnprovenFallthroughEntry(ArrayRef<InternalDecodedInst> Decoded,
                                        const LLVMState &LS, uint64_t TextAddr,
                                        uint64_t TextEnd,
                                        uint64_t FunctionBegin,
                                        uint64_t ReturnOffset,
                                        ArrayRef<uint64_t> DeclaredEntries,
                                        ArrayRef<KnownCallSite> Calls) {
  if (FunctionBegin == 0)
    return false;

  size_t BeginIndex = 0;
  while (BeginIndex != Decoded.size() &&
         Decoded[BeginIndex].Offset < FunctionBegin)
    ++BeginIndex;
  if (BeginIndex == Decoded.size() ||
      Decoded[BeginIndex].Offset != FunctionBegin) {
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: function entry at 0x"
          << utohexstr(FunctionBegin) << " is not an instruction boundary\n";
    return true;
  }

  uint64_t ChainBegin = FunctionBegin;
  size_t PredecessorIndex = BeginIndex;
  while (PredecessorIndex != 0) {
    const InternalDecodedInst &Predecessor = Decoded[PredecessorIndex - 1];
    std::optional<uint64_t> PredecessorEnd = checkedAddUint64(
        Predecessor.Offset, Predecessor.Size, "fallthrough predecessor end");
    if (!PredecessorEnd || *PredecessorEnd != ChainBegin ||
        !Predecessor.DecodeSucceeded) {
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
            << " is not a bounded return: fallthrough into function entry "
               "at 0x"
            << utohexstr(FunctionBegin) << " is unprovable\n";
      return true;
    }
    if (LS.MIA->isBarrier(Predecessor.Inst))
      break;
    ChainBegin = Predecessor.Offset;
    --PredecessorIndex;
  }

  if (ChainBegin == FunctionBegin)
    return false;

  for (uint64_t Entry : DeclaredEntries)
    if (Entry >= ChainBegin && Entry < FunctionBegin) {
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
            << " is not a bounded return: declared entry at 0x"
            << utohexstr(Entry) << " falls through to function entry 0x"
            << utohexstr(FunctionBegin) << "\n";
      return true;
    }

  for (const KnownCallSite &Call : Calls) {
    uint64_t Source = Decoded[Call.InstIndex].Offset;
    if (Source >= ChainBegin && Source < FunctionBegin)
      continue;
    if ((Call.Target >= ChainBegin && Call.Target < FunctionBegin) ||
        (Call.Continuation >= ChainBegin &&
         Call.Continuation < FunctionBegin)) {
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
            << " is not a bounded return: call at 0x" << utohexstr(Source)
            << " enters the fallthrough chain at 0x"
            << utohexstr(Call.Target >= ChainBegin &&
                                 Call.Target < FunctionBegin
                             ? Call.Target
                             : Call.Continuation)
            << "\n";
      return true;
    }
  }

  for (const InternalDecodedInst &Source : Decoded) {
    std::optional<uint64_t> Target =
        getDirectTextTarget(Source, LS, TextAddr, TextEnd);
    if (!Target || *Target < ChainBegin || *Target >= FunctionBegin ||
        (Source.Offset >= ChainBegin && Source.Offset < FunctionBegin))
      continue;
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: control flow at 0x"
          << utohexstr(Source.Offset) << " enters the fallthrough chain at 0x"
          << utohexstr(*Target) << "\n";
    return true;
  }
  return false;
}

static std::optional<SmallVector<BoundedSetPcReturn, 2>>
collectBoundedSetPcReturns(ArrayRef<InternalDecodedInst> Decoded,
                           const LLVMState &LS, uint64_t TextAddr,
                           uint64_t TextEnd, ArrayRef<uint64_t> DeclaredEntries,
                           ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
                           ArrayRef<KnownCallSite> Calls,
                           ArrayRef<uint64_t> ExternalEntries) {
  SmallVector<BoundedSetPcReturn, 2> Returns;
  for (size_t ReturnIndex = 0; ReturnIndex != Decoded.size(); ++ReturnIndex) {
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

    bool IsBounded = false;
    for (const ElfView::FunctionTextRange &Range : FunctionRanges) {
      if (Range.Begin < TextAddr || Range.Begin >= TextEnd ||
          Range.End <= Range.Begin || Range.End > TextEnd ||
          (Range.Symbol && Range.Symbol->getBinding() != ELF::STB_LOCAL))
        continue;
      uint64_t FunctionBegin = Range.Begin - TextAddr;
      uint64_t FunctionEnd = Range.End - TextAddr;
      if (Return.Offset < FunctionBegin || Return.Offset >= FunctionEnd)
        continue;

      bool Safe = true;
      for (uint64_t Entry : ExternalEntries)
        if (Entry >= FunctionBegin && Entry < FunctionEnd) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: externally reachable entry at "
                   "0x"
                << utohexstr(Entry) << " overlaps the local function\n";
          Safe = false;
          break;
        }
      if (!Safe)
        continue;

      for (const ElfView::FunctionTextRange &Alias : FunctionRanges) {
        if (Return.Offset + TextAddr < Alias.Begin ||
            Return.Offset + TextAddr >= Alias.End || Alias.Begin == Range.Begin)
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

      for (uint64_t Entry : DeclaredEntries)
        if (Entry > FunctionBegin && Entry < FunctionEnd) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: declared entry at 0x"
                << utohexstr(Entry) << " bypasses the function entry\n";
          Safe = false;
          break;
        }
      if (!Safe)
        continue;

      if (hasUnprovenFallthroughEntry(Decoded, LS, TextAddr, TextEnd,
                                      FunctionBegin, Return.Offset,
                                      DeclaredEntries, Calls))
        continue;

      // The link pair must retain the value written by the incoming call
      // throughout the function. This includes blocks laid out after the
      // return that may branch back into its epilogue.
      for (const InternalDecodedInst &DI : Decoded) {
        if (DI.Offset < FunctionBegin || DI.Offset >= FunctionEnd)
          continue;
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

      SmallVector<uint64_t, 2> Targets;
      for (const KnownCallSite &Call : Calls) {
        if (Call.Target < FunctionBegin || Call.Target >= FunctionEnd)
          continue;
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
      for (size_t SourceIndex = 0; SourceIndex != Decoded.size();
           ++SourceIndex) {
        const InternalDecodedInst &Source = Decoded[SourceIndex];
        std::optional<uint64_t> Target =
            getDirectTextTarget(Source, LS, TextAddr, TextEnd);
        if (!Target || *Target < FunctionBegin || *Target >= FunctionEnd ||
            (Source.Offset >= FunctionBegin && Source.Offset < FunctionEnd))
          continue;

        bool IsKnownEntryCall = false;
        if (LS.MIA->isCall(Source.Inst) && *Target == FunctionBegin)
          for (const KnownCallSite &Call : Calls)
            IsKnownEntryCall |= Call.InstIndex == SourceIndex &&
                                Call.ReturnRegister == ReturnRegister;
        if (!IsKnownEntryCall) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: control flow at 0x"
                << utohexstr(Source.Offset) << " enters at 0x"
                << utohexstr(*Target) << "\n";
          Safe = false;
          break;
        }
      }
      if (!Safe)
        continue;

      llvm::sort(Targets);
      Targets.erase(std::unique(Targets.begin(), Targets.end()), Targets.end());
      Returns.push_back({ReturnIndex, std::move(Targets)});
      IsBounded = true;
      break;
    }
    if (!IsBounded)
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
            << " remains an unbounded indirect transfer\n";
  }
  return Returns;
}

static const BoundedSetPcReturn *
findBoundedSetPcReturn(ArrayRef<BoundedSetPcReturn> Returns, size_t InstIndex) {
  for (const BoundedSetPcReturn &Return : Returns)
    if (Return.InstIndex == InstIndex)
      return &Return;
  return nullptr;
}

static bool
hasKnownControlFlowEntry(ArrayRef<InternalDecodedInst> Decoded,
                         const LLVMState &LS, uint64_t TextAddr,
                         uint64_t TextEnd, ArrayRef<uint64_t> DeclaredEntries,
                         ArrayRef<BoundedSetPcReturn> BoundedReturns,
                         uint64_t SequenceStart, uint64_t SequenceEnd) {
  for (uint64_t Entry : DeclaredEntries)
    if (Entry > SequenceStart && Entry <= SequenceEnd)
      return true;

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.DecodeSucceeded && DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
      const BoundedSetPcReturn *Return =
          findBoundedSetPcReturn(BoundedReturns, I);
      if (!Return)
        return true;
      for (uint64_t Target : Return->Targets)
        if (Target > SequenceStart && Target <= SequenceEnd)
          return true;
      continue;
    }

    // Without bounding an indirect target, it may enter at any instruction in
    // the materialization. Keep the call unresolved rather than relying on
    // the indirect transfer's containing function alone.
    if (DI.DecodeSucceeded && !LS.MIA->isReturn(DI.Inst) &&
        (LS.MIA->isIndirectBranch(DI.Inst) ||
         DI.Inst.getOpcode() == LS.SAddPcI64Opcode))
      return true;
    if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
        LS.MIA->isReturn(DI.Inst))
      continue;

    std::optional<uint64_t> RelativeTarget =
        getDirectTextTarget(DI, LS, TextAddr, TextEnd);
    if (RelativeTarget && *RelativeTarget > SequenceStart &&
        *RelativeTarget <= SequenceEnd)
      return true;
  }
  return false;
}

/// Collect statically known direct branch and call destinations so an interior
/// entry point is never swallowed by coalescing.  This is the PR's richer
/// whole-object analysis used by production rewriting.
static std::optional<DenseSet<uint64_t>> collectDirectBranchTargets(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    ArrayRef<uint8_t> Text, const ElfView &Elf,
    const DenseSet<uint64_t> &CodeEntries, ArrayRef<uint64_t> TextSymbolOffsets,
    ArrayRef<ElfView::TextOffsetRange> TextSymbolExtents,
    const CompatibilityControlFlowFacts &CompatibilityFacts) {
  if (!LS.MIA) {
    log() << "hotswap: MC branch analysis is unavailable; adjacent far "
             "trampolines will not be coalesced\n";
    return std::nullopt;
  }

  DenseSet<uint64_t> Targets = CodeEntries;
  Targets.insert(CompatibilityFacts.Summary.Targets.begin(),
                 CompatibilityFacts.Summary.Targets.end());
  DenseSet<uint64_t> DecodedBoundaries;
  for (const InternalDecodedInst &DI : Decoded)
    DecodedBoundaries.insert(DI.Offset);
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (LS.MIA->isCall(DI.Inst)) {
      std::optional<uint64_t> Continuation = checkedAddUint64(
          DI.Offset, DI.Size, "call continuation control-flow target");
      if (!Continuation || *Continuation >= Text.size() ||
          !DecodedBoundaries.contains(*Continuation)) {
        log() << "hotswap: call at 0x" << utohexstr(DI.Offset)
              << " has no exact in-text continuation; source relocation and "
                 "donated sleds are disabled\n";
        return std::nullopt;
      }
      Targets.insert(*Continuation);
    }
    // Absolute and arbitrary-link materialized calls were resolved by the
    // later analysis. Their text targets are already in Targets; do not feed
    // their non-PC-relative immediates back through the PR evaluator.
    if (findResolvedCompatibilityCall(CompatibilityFacts, I))
      continue;
    if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
        LS.MIA->isIndirectBranch(DI.Inst) || LS.MIA->isReturn(DI.Inst))
      continue;
    bool HasImmediate = false;
    for (const MCOperand &Op : DI.Inst)
      HasImmediate |= Op.isImm();
    if (!HasImmediate)
      continue;
    std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
    if (!Target) {
      log() << "hotswap: MC analysis could not evaluate direct control-flow "
               "instruction at 0x"
            << utohexstr(DI.Offset)
            << "; source relocation and donated sleds are disabled\n";
      return std::nullopt;
    }
    Targets.insert(*Target);
  }

  SmallVector<MaterializedPcTransfer> Candidates;
  for (size_t I = 0; I != Decoded.size(); ++I)
    if (std::optional<MaterializedPcTransfer> Transfer =
            evaluateProvenPcTransfer(Decoded, I, Targets, Text, LS, Elf,
                                     CompatibilityFacts, TextSymbolOffsets,
                                     TextSymbolExtents))
      Candidates.push_back(*Transfer);

  DenseSet<uint64_t> CombinedTargets = Targets;
  for (const MaterializedPcTransfer &Transfer : Candidates)
    CombinedTargets.insert(Transfer.Target);
  return CombinedTargets;
}

/// Run the later amd-staging arbitrary-link call analysis and retain the proof
/// records needed by the PR's production pipeline.  Reducing this result to
/// DirectControlFlowInfo alone loses which indirect calls and returns were
/// proven, causing downstream analysis to classify them as arbitrary again.
static std::optional<CompatibilityControlFlowFacts>
collectCompatibilityControlFlowFacts(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize,
    ArrayRef<uint64_t> DeclaredEntries,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<uint64_t> ExternalEntries) {
  if (!LS.MIA)
    return std::nullopt;

  std::optional<uint64_t> TextEnd =
      checkedAddUint64(TextAddr, TextSize, "direct target text end");
  if (!TextEnd)
    return std::nullopt;

  SmallVector<std::optional<PcMaterializedCallInfo>, 16> MaterializedCalls(
      Decoded.size());
  for (size_t I = 0; I != Decoded.size(); ++I)
    MaterializedCalls[I] = matchPcMaterializedCall(Decoded, I, LS, TextAddr);

  std::optional<SmallVector<KnownCallSite, 4>> Calls =
      collectKnownCallSites(Decoded, LS, TextAddr, *TextEnd, MaterializedCalls);
  if (!Calls)
    return std::nullopt;
  std::optional<SmallVector<BoundedSetPcReturn, 2>> BoundedReturns =
      collectBoundedSetPcReturns(Decoded, LS, TextAddr, *TextEnd,
                                 DeclaredEntries, FunctionRanges, *Calls,
                                 ExternalEntries);
  if (!BoundedReturns)
    return std::nullopt;

  CompatibilityControlFlowFacts Facts;
  DirectControlFlowInfo &Info = Facts.Summary;
  for (const BoundedSetPcReturn &Return : *BoundedReturns) {
    if (Return.InstIndex >= Decoded.size())
      return std::nullopt;
    Facts.BoundedReturnOffsets.insert(Decoded[Return.InstIndex].Offset);
  }

  for (size_t InstIndex = 0; InstIndex != Decoded.size(); ++InstIndex) {
    const InternalDecodedInst &DI = Decoded[InstIndex];
    if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
        LS.MIA->isReturn(DI.Inst))
      continue;
    if (LS.MIA->isIndirectBranch(DI.Inst))
      continue;

    bool HasPcRelativeOperand = false;
    for (const MCOperandInfo &Op : LS.MCII->get(DI.Inst.getOpcode()).operands())
      HasPcRelativeOperand |= Op.OperandType == MCOI::OPERAND_PCREL;
    if (!HasPcRelativeOperand) {
      if (!LS.MIA->isCall(DI.Inst))
        continue;
      std::optional<uint64_t> Target;
      bool IsMaterialized = false;
      if (DI.Inst.getOpcode() == LS.SSwapPcI64Opcode &&
          DI.Inst.getNumOperands() != 0 &&
          DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm()) {
        Target = static_cast<uint64_t>(
            DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm());
      } else if (MaterializedCalls[InstIndex] &&
                 !hasKnownControlFlowEntry(
                     Decoded, LS, TextAddr, *TextEnd, DeclaredEntries,
                     *BoundedReturns,
                     MaterializedCalls[InstIndex]->SequenceStart,
                     MaterializedCalls[InstIndex]->SequenceEnd)) {
        Target = MaterializedCalls[InstIndex]->Target;
        IsMaterialized = true;
      }
      if (!Target) {
        Facts.UnresolvedCallIndices.push_back(InstIndex);
        Info.HasUnresolvedTargets = true;
        continue;
      }

      std::optional<uint64_t> Continuation = checkedAddUint64(
          DI.Offset, DI.Size, "resolved compatibility call continuation");
      std::optional<MCRegister> ReturnRegister = getCallReturnRegister(DI, LS);
      if (!Continuation || !ReturnRegister)
        return std::nullopt;
      ResolvedCompatibilityCall Call;
      Call.InstIndex = InstIndex;
      Call.Site = DI.Offset;
      Call.SequenceStart = IsMaterialized
                               ? MaterializedCalls[InstIndex]->SequenceStart
                               : DI.Offset;
      Call.SequenceEnd = *Continuation;
      Call.Continuation = *Continuation;
      Call.ReturnRegister = *ReturnRegister;
      if (*Target >= TextAddr && *Target < *TextEnd) {
        uint64_t RelativeTarget = *Target - TextAddr;
        Call.TextTarget = RelativeTarget;
        Info.Targets.insert(RelativeTarget);
        if (IsMaterialized)
          log() << "hotswap: resolved PC-materialized call at 0x"
                << utohexstr(DI.Offset) << " to .text+0x"
                << utohexstr(RelativeTarget) << "\n";
      }
      Facts.ResolvedCalls.push_back(std::move(Call));
      continue;
    }

    std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
    if (!Target)
      return std::nullopt;
    Info.Targets.insert(*Target);
  }
  return Facts;
}

/// Compatibility entry point retained for later amd-staging callers and the
/// focused unit tests.  Production consumes the full fact set above.
std::optional<DirectControlFlowInfo>
collectDirectBranchTargets(ArrayRef<InternalDecodedInst> Decoded,
                           const LLVMState &LS, uint64_t TextAddr,
                           uint64_t TextSize,
                           ArrayRef<uint64_t> DeclaredEntries,
                           ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
                           ArrayRef<uint64_t> ExternalEntries) {
  std::optional<CompatibilityControlFlowFacts> Facts =
      collectCompatibilityControlFlowFacts(
          Decoded, LS, TextAddr, TextSize, DeclaredEntries, FunctionRanges,
          ExternalEntries);
  if (!Facts)
    return std::nullopt;
  return std::move(Facts->Summary);
}

/// Coalesce runs of adjacent far patch sites when the same SGPR scratch block
/// is safe at every site. Removing each interior return reservation preserves
/// replacement order and reduces the number of required forward gateways.
/// This deliberately never steals an unpatched neighboring instruction.
static void
mergeAdjacentLongTrampolines(std::vector<Trampoline> &Trampolines,
                             const DenseSet<uint64_t> &DirectBranchTargets,
                             const DenseSet<uint64_t> &IncompleteFunctions) {
  std::vector<Trampoline> Merged;
  Merged.reserve(Trampolines.size());
  uint64_t MergeCount = 0;

  for (Trampoline &T : Trampolines) {
    bool Adjacent = false;
    if (!Merged.empty()) {
      Trampoline &Prev = Merged.back();
      std::optional<uint64_t> PrevEnd = checkedAddUint64(
          Prev.OriginalOffset, Prev.OriginalSize, "adjacent trampoline end");
      Adjacent = PrevEnd && *PrevEnd == T.OriginalOffset && Prev.Long &&
                 T.Long && Prev.UsesSetPCBack && T.UsesSetPCBack &&
                 Prev.LongBranchSgprBase == T.LongBranchSgprBase &&
                 Prev.LongBranchScratchIsSiteProven ==
                     T.LongBranchScratchIsSiteProven &&
                 Prev.HasFunctionRange && T.HasFunctionRange &&
                 Prev.FunctionStart == T.FunctionStart &&
                 Prev.FunctionEnd == T.FunctionEnd &&
                 !IncompleteFunctions.contains(T.FunctionStart) &&
                 !DirectBranchTargets.contains(T.OriginalOffset) &&
                 Prev.Bytes.size() >= SetPcReturnReserveBytes &&
                 T.Bytes.size() >= SetPcReturnReserveBytes;
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
    Prev.Bytes.resize(Prev.Bytes.size() - SetPcReturnReserveBytes);
    Prev.Bytes.append(T.Bytes.begin(), T.Bytes.end());
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

static bool isSafeStraightLineRelocation(const InternalDecodedInst &DI,
                                         const LLVMState &LS,
                                         const DenseSet<uint64_t> &Protected) {
  if (!LS.MIA || LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))
    return false;
  return !Protected.contains(DI.Offset) && DI.Mnemonic != "<unknown>" &&
         DI.Mnemonic != "<replaced>" && DI.Mnemonic != "s_clause" &&
         DI.Mnemonic != "s_delay_alu" &&
         !StringRef(DI.Mnemonic).contains("_pc_");
}

/// Re-decode the bytes that source growth would actually copy. Per-instruction
/// patching runs before source-window expansion, so the original Decoded entry
/// may no longer describe this interval. Requiring one same-size current
/// instruction prevents relocation of a newly installed branch or a
/// multi-instruction replacement under stale dataflow semantics.
static std::optional<InternalDecodedInst>
decodeCurrentRelocationCandidate(const PatchContext &Ctx,
                                 const InternalDecodedInst &Original) {
  if (Original.Offset > Ctx.TextSize ||
      Original.Size > Ctx.TextSize - Original.Offset)
    return std::nullopt;
  std::vector<InternalDecodedInst> Current;
  if (!decodeTextSection(Ctx.Text + Original.Offset, Original.Size, Ctx.LS,
                         Current) ||
      Current.size() != 1 || Current.front().Offset != 0 ||
      Current.front().Size != Original.Size)
    return std::nullopt;
  Current.front().Offset = Original.Offset;
  return std::move(Current.front());
}

/// Return the number of following instructions whose relative positions are
/// significant to an s_delay_alu encoding. Instid0 names the immediately
/// following instruction. Instid1 names the instruction selected by instskip,
/// so every instruction through that target must stay in place. Malformed or
/// undecodable immediates retain the conservative six-instruction bound.
unsigned getDelayProtectedSpan(const InternalDecodedInst &DI) {
  if (DI.Inst.getNumOperands() != 1 || !DI.Inst.getOperand(0).isImm())
    return 6;

  uint64_t Imm = static_cast<uint64_t>(DI.Inst.getOperand(0).getImm());
  if ((Imm & ~uint64_t{0x7FF}) != 0)
    return 6;

  unsigned InstId0 = Imm & 0xF;
  unsigned Skip = (Imm >> 4) & 0x7;
  unsigned InstId1 = (Imm >> 7) & 0xF;
  if (InstId0 >= 12 || InstId1 >= 12 || Skip > 5 || (InstId1 == 0 && Skip != 0))
    return 6;

  unsigned Span = InstId0 != 0 ? 1 : 0;
  if (InstId1 != 0)
    Span = std::max(Span, Skip + 1);
  return Span;
}

/// Instructions covered by a hard clause or a delay directive must remain in
/// place relative to that directive. Mark the complete encoded clause and the
/// exact s_delay_alu span when its immediate is well formed. The blanket
/// B0-to-A0 s_clause rewrite releases each clause's member protections only
/// after that directive has been replaced successfully.
static void
collectRelocationProtectedOffsets(ArrayRef<InternalDecodedInst> Decoded,
                                  PatchContext &Ctx) {
  unsigned DelayRemaining = 0;

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DelayRemaining != 0) {
      protectNonClauseRelocationOffset(Ctx, DI.Offset);
      --DelayRemaining;
    }

    if (DI.Mnemonic == "s_clause" && DI.Inst.getNumOperands() == 1 &&
        DI.Inst.getOperand(0).isImm()) {
      const unsigned MemberCount =
          (static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) & 63u) + 1;
      const size_t Available =
          std::min<size_t>(MemberCount, Decoded.size() - I - 1);
      for (size_t Member = I + 1, End = I + 1 + Available; Member != End;
           ++Member) {
        const uint64_t Offset = Decoded[Member].Offset;
        ++Ctx.ClauseRelocationProtectionCounts[Offset];
        Ctx.RelocationProtectedOffsets.insert(Offset);
      }
    } else if (DI.Mnemonic == "s_delay_alu") {
      DelayRemaining = std::max(DelayRemaining, getDelayProtectedSpan(DI));
    }
  }
}

/// Under the public HotSwap API's callable-control-flow precondition, this is
/// an ABI call and its register target is a callable function entry. This is a
/// control-flow geometry fact only: all liveness and value analyses continue
/// to treat the instruction as an opaque call boundary.
static bool isStandardLinkCall(const InternalDecodedInst &DI,
                               const LLVMState &LS) {
  return DI.Inst.getOpcode() == LS.SSwapPcI64Opcode &&
         DI.Inst.getNumOperands() == 2 && DI.Inst.getOperand(0).isReg() &&
         DI.Inst.getOperand(0).getReg() &&
         StringRef(LS.MRI->getName(DI.Inst.getOperand(0).getReg())) ==
             "SGPR30_SGPR31";
}

static std::optional<std::pair<unsigned, SgprPairHalves>>
getNumberedSgprPair(const LLVMState &LS, const MCOperand &Operand) {
  if (!Operand.isReg() || !Operand.getReg())
    return std::nullopt;
  std::optional<SgprPairHalves> Halves =
      getSgprPairHalves(LS, MCRegister(Operand.getReg()));
  if (!Halves || Halves->IsVcc)
    return std::nullopt;
  std::optional<unsigned> Lo = numberedSgprIndex(*LS.MRI, Halves->Regs[0]);
  std::optional<unsigned> Hi = numberedSgprIndex(*LS.MRI, Halves->Regs[1]);
  if (!Lo || !Hi || *Hi != *Lo + 1)
    return std::nullopt;
  return std::make_pair(*Lo, *Halves);
}

static bool hasUnresolvedStandardLinkCall(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    const ElfView &Elf, const DenseSet<uint64_t> &DirectControlFlowTargets,
    ArrayRef<uint8_t> Text, std::optional<ArrayRef<uint64_t>> TextSymbolOffsets,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents,
    const CompatibilityControlFlowFacts &CompatibilityFacts) {
  for (size_t I = 0; I != Decoded.size(); ++I) {
    if (!isStandardLinkCall(Decoded[I], LS))
      continue;
    if (!findResolvedCompatibilityCall(CompatibilityFacts, I) &&
        !evaluateMaterializedPcTransfer(Decoded, I, DirectControlFlowTargets,
                                        Text, LS, Elf, TextSymbolOffsets,
                                        TextSymbolExtents))
      return true;
  }
  return false;
}

/// Prove nonstandard s_set_pc_i64 returns only when every modeled ingress
/// carries the exact link written by an in-text s_call_i64 using the same
/// physical SGPR pair. Any alternate entry, clobber, or unmodeled indirect
/// transfer invalidates the proof globally.
static DenseSet<uint64_t> collectProvenDirectCallReturns(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS, ElfView &Elf,
    const DenseSet<uint64_t> &DirectControlFlowTargets, ArrayRef<uint8_t> Text,
    std::optional<ArrayRef<uint64_t>> TextSymbolOffsets,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents,
    bool HasUnresolvedStandardLinkCall,
    const CompatibilityControlFlowFacts &CompatibilityFacts) {
  // The compatibility analysis supplies candidates and teaches the PR-wide
  // dataflow about call forms it cannot otherwise resolve.  It is not an
  // independent fallback: an unmodeled standard-link ingress can carry an
  // arbitrary value in the candidate return pair.
  DenseSet<uint64_t> CompatibilityCandidates;
  if (!CompatibilityFacts.Summary.HasUnresolvedTargets &&
      !HasUnresolvedStandardLinkCall)
    CompatibilityCandidates = CompatibilityFacts.BoundedReturnOffsets;
  DenseSet<uint64_t> Proven;
  DenseSet<uint64_t> ValidatedCompatibilityProven;
  if (!LS.MIA || !LS.MCII || !LS.MRI || Decoded.empty())
    return {};

  // AMDGPU's instruction metadata does not classify RFE as a branch or a
  // terminator. It can nevertheless resume at an arbitrary PC with arbitrary
  // SGPR contents. An undecodable word may hide the same behavior, so no
  // nonstandard link-pair proof survives either condition.
  if (llvm::any_of(Decoded, [&](const InternalDecodedInst &DI) {
        return DI.Mnemonic == "<unknown>" ||
               StringRef(DI.Mnemonic).starts_with("s_rfe");
      }))
    return {};

  DenseMap<uint64_t, unsigned> OffsetToIndex;
  OffsetToIndex.reserve(Decoded.size());
  for (unsigned I = 0; I != Decoded.size(); ++I) {
    if (Decoded[I].Size == 0 ||
        !OffsetToIndex.try_emplace(Decoded[I].Offset, I).second)
      return {};
  }

  std::vector<std::optional<MaterializedPcTransfer>> Materialized(
      Decoded.size());
  for (size_t I = 0; I != Decoded.size(); ++I)
    Materialized[I] = evaluateProvenPcTransfer(
        Decoded, I, DirectControlFlowTargets, Text, LS, Elf,
        CompatibilityFacts, TextSymbolOffsets, TextSymbolExtents);

  std::vector<std::optional<unsigned>> ReturnPairs(Decoded.size());
  SmallVector<std::pair<unsigned, SgprPairHalves>, 4> Pairs;
  for (unsigned I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.Inst.getOpcode() != LS.SSetPcI64Opcode ||
        isStandardLinkReturn(DI, LS) || Materialized[I] ||
        DI.Inst.getNumOperands() != 1)
      continue;
    std::optional<std::pair<unsigned, SgprPairHalves>> Pair =
        getNumberedSgprPair(LS, DI.Inst.getOperand(0));
    if (!Pair)
      continue;
    ReturnPairs[I] = Pair->first;
    if (!llvm::any_of(Pairs, [&](const auto &Existing) {
          return Existing.first == Pair->first;
        }))
      Pairs.push_back(*Pair);
  }
  if (Pairs.empty())
    return {};

  std::optional<DenseSet<uint64_t>> DescriptorEntries =
      collectResolvedKernelEntryOffsets(Elf, LS);
  if (!DescriptorEntries)
    return {};
  SmallVector<unsigned, 8> DescriptorEntryIndices;
  for (uint64_t Entry : *DescriptorEntries) {
    auto It = OffsetToIndex.find(Entry);
    if (It == OffsetToIndex.end())
      return {};
    DescriptorEntryIndices.push_back(It->second);
  }

  SmallVector<unsigned, 16> CallableEntryIndices;
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges()) {
    if (Range.Begin < Elf.textAddr() ||
        (!HasUnresolvedStandardLinkCall && !isExternallyVisibleCallable(Range)))
      continue;
    const uint64_t Entry = Range.Begin - Elf.textAddr();
    if (Entry >= Text.size())
      continue;
    auto It = OffsetToIndex.find(Entry);
    if (It == OffsetToIndex.end())
      return {};
    if (!llvm::is_contained(CallableEntryIndices, It->second))
      CallableEntryIndices.push_back(It->second);
  }

  struct CallIngress {
    unsigned TargetIndex = 0;
    std::optional<unsigned> ExactLinkPair;
  };
  SmallVector<unsigned, 16> CallContinuationIndices;
  SmallVector<CallIngress, 16> CallIngresses;
  for (unsigned I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (!LS.MIA->isCall(DI.Inst))
      continue;

    std::optional<uint64_t> Continuation = checkedAddUint64(
        DI.Offset, DI.Size, "direct-call return proof continuation");
    if (!Continuation || *Continuation >= Text.size())
      return {};
    auto ContinuationIt = OffsetToIndex.find(*Continuation);
    if (ContinuationIt == OffsetToIndex.end())
      return {};
    if (!llvm::is_contained(CallContinuationIndices, ContinuationIt->second))
      CallContinuationIndices.push_back(ContinuationIt->second);

    std::optional<uint64_t> Target;
    std::optional<unsigned> ExactPair;
    if (DI.Inst.getOpcode() == LS.SCallI64Opcode) {
      Target = evaluateDirectControlFlowTarget(DI, LS);
      if (!Target)
        return {};
      if (DI.Inst.getNumOperands() == 2)
        if (std::optional<std::pair<unsigned, SgprPairHalves>> Pair =
                getNumberedSgprPair(LS, DI.Inst.getOperand(0)))
          ExactPair = Pair->first;
    } else if (Materialized[I]) {
      Target = Materialized[I]->Target;
      if (DI.Inst.getNumOperands() != 0)
        if (std::optional<std::pair<unsigned, SgprPairHalves>> Pair =
                getNumberedSgprPair(LS, DI.Inst.getOperand(0)))
          ExactPair = Pair->first;
    } else if (!LS.MIA->isIndirectBranch(DI.Inst)) {
      Target = evaluateDirectControlFlowTarget(DI, LS);
      if (!Target)
        return {};
    }
    if (!Target || *Target >= Text.size())
      continue;
    auto TargetIt = OffsetToIndex.find(*Target);
    if (TargetIt == OffsetToIndex.end())
      return {};
    CallIngresses.push_back({TargetIt->second, ExactPair});
  }

  std::vector<SmallVector<unsigned, 2>> Successors(Decoded.size());
  auto AddFallthrough = [&](unsigned I) {
    std::optional<uint64_t> Next =
        checkedAddUint64(Decoded[I].Offset, Decoded[I].Size,
                         "direct-call return proof fallthrough");
    if (!Next)
      return false;
    // A trailing padding instruction can end exactly at the .text boundary.
    // That is a terminal edge, not an incomplete in-text successor; accepting
    // it does not create a new ingress into any proven return.
    if (*Next == Text.size())
      return I + 1 == Decoded.size();
    if (*Next > Text.size() || I + 1 >= Decoded.size() ||
        Decoded[I + 1].Offset != *Next)
      return false;
    Successors[I].push_back(I + 1);
    return true;
  };
  auto AddTarget = [&](unsigned I, uint64_t Target) {
    if (Target >= Text.size())
      return true;
    auto It = OffsetToIndex.find(Target);
    if (It == OffsetToIndex.end())
      return false;
    Successors[I].push_back(It->second);
    return true;
  };

  for (unsigned I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    if (LS.MIA->isCall(DI.Inst) || DI.Mnemonic == "s_code_end" ||
        DI.Mnemonic == "s_endpgm" || DI.Mnemonic == "s_endpgm_saved" ||
        LS.MIA->isReturn(DI.Inst))
      continue;
    if (Materialized[I]) {
      if (!AddTarget(I, Materialized[I]->Target))
        return {};
      continue;
    }
    if (DI.Inst.getOpcode() == LS.SSetPcI64Opcode)
      continue;
    if (LS.MIA->isBranch(DI.Inst)) {
      if (LS.MIA->isIndirectBranch(DI.Inst))
        continue;
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
      if (!Target || !AddTarget(I, *Target))
        return {};
      if (LS.MIA->isConditionalBranch(DI.Inst)) {
        if (!AddFallthrough(I))
          return {};
      } else if (!LS.MIA->isUnconditionalBranch(DI.Inst)) {
        return {};
      }
      continue;
    }
    if (StringRef(DI.Mnemonic).starts_with("s_trap") || Desc.isTrap()) {
      if (!AddFallthrough(I))
        return {};
      continue;
    }
    if (Desc.isTerminator() || LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))
      return {};
    if (!AddFallthrough(I))
      return {};
  }

  constexpr uint8_t ExactDirectCallLink = 1;
  constexpr uint8_t UnknownLink = 2;
  for (const auto &[PairBase, PairHalves] : Pairs) {
    SmallVector<uint8_t> In(Decoded.size(), 0);
    SmallVector<unsigned, 64> Worklist;
    auto Merge = [&](unsigned Index, uint8_t State) {
      const uint8_t Merged = In[Index] | State;
      if (Merged == In[Index])
        return;
      In[Index] = Merged;
      Worklist.push_back(Index);
    };

    for (unsigned Entry : DescriptorEntryIndices)
      Merge(Entry, UnknownLink);
    for (unsigned Entry : CallableEntryIndices)
      Merge(Entry, UnknownLink);
    for (unsigned Continuation : CallContinuationIndices)
      Merge(Continuation, UnknownLink);
    for (const CallIngress &Ingress : CallIngresses)
      Merge(Ingress.TargetIndex,
            Ingress.ExactLinkPair && *Ingress.ExactLinkPair == PairBase
                ? ExactDirectCallLink
                : UnknownLink);

    for (size_t Next = 0; Next != Worklist.size(); ++Next) {
      const unsigned I = Worklist[Next];
      const InternalDecodedInst &DI = Decoded[I];
      if (ReturnPairs[I] && *ReturnPairs[I] == PairBase) {
        if (In[I] == ExactDirectCallLink) {
          if (CompatibilityCandidates.contains(DI.Offset))
            ValidatedCompatibilityProven.insert(DI.Offset);
          else
            Proven.insert(DI.Offset);
        } else {
          Proven.erase(DI.Offset);
          ValidatedCompatibilityProven.erase(DI.Offset);
        }
        continue;
      }

      uint8_t Out = In[I];
      const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
      if (DI.Mnemonic == "<unknown>" ||
          StringRef(DI.Mnemonic).starts_with("s_trap") || Desc.isTrap()) {
        Out = UnknownLink;
      } else if (Out != 0) {
        std::optional<RegisterPairAccess> Access = getRegisterPairAccess(
            LS, DI, PairHalves.Regs, /*PairIsAbiVcc=*/false);
        if (!Access || Access->Defs != 0)
          Out = UnknownLink;
      }
      for (unsigned Succ : Successors[I])
        Merge(Succ, Out);
    }
  }

  // A genuinely arbitrary transfer can enter any instruction with an unknown
  // link value, invalidating every otherwise-local direct-return proof.
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.Mnemonic == "s_endpgm" || DI.Mnemonic == "s_endpgm_saved")
      continue;
    const bool HasOpaqueControlFlow =
        DI.Mnemonic == "<unknown>" ||
        StringRef(DI.Mnemonic).starts_with("s_rfe");
    if (!HasOpaqueControlFlow && !LS.MIA->isIndirectBranch(DI.Inst) &&
        !(LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI) &&
          StringRef(DI.Mnemonic).contains("_pc_")))
      continue;
    if (isStandardLinkReturn(DI, LS) || Materialized[I] ||
        findResolvedCompatibilityCall(CompatibilityFacts, I) ||
        isStandardLinkCall(DI, LS) || Proven.contains(DI.Offset) ||
        ValidatedCompatibilityProven.contains(DI.Offset))
      continue;
    return {};
  }
  Proven.insert(ValidatedCompatibilityProven.begin(),
                ValidatedCompatibilityProven.end());
  return Proven;
}

/// Relocating an instruction changes its address. In a function containing a
/// register-based PC transfer, MC cannot prove that the instruction is not an
/// indirect destination, so leave the complete function in place.
static DenseSet<uint64_t> collectIndirectControlFlowFunctions(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    const ElfView &Elf, const DenseSet<uint64_t> &DirectControlFlowTargets,
    ArrayRef<uint8_t> Text, std::optional<ArrayRef<uint64_t>> TextSymbolOffsets,
    std::optional<ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents,
    const DenseSet<uint64_t> &ProvenDirectCallReturns,
    const CompatibilityControlFlowFacts &CompatibilityFacts,
    bool &HasUnknownArbitraryIndirectTarget) {
  DenseSet<uint64_t> Functions;
  HasUnknownArbitraryIndirectTarget = false;
  if (!LS.MIA)
    return Functions;

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.Mnemonic == "s_endpgm" || DI.Mnemonic == "s_endpgm_saved")
      continue;
    const bool HasOpaqueControlFlow =
        DI.Mnemonic == "<unknown>" ||
        StringRef(DI.Mnemonic).starts_with("s_rfe");
    if (!HasOpaqueControlFlow && !LS.MIA->isIndirectBranch(DI.Inst) &&
        !(LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI) &&
          StringRef(DI.Mnemonic).contains("_pc_")))
      continue;
    std::optional<ElfView::FunctionTextRange> Range =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    const bool IsSetPc = DI.Inst.getOpcode() == LS.SSetPcI64Opcode;
    const bool UsesStandardLinkPair =
        IsSetPc && DI.Inst.getNumOperands() == 1 &&
        DI.Inst.getOperand(0).isReg() && DI.Inst.getOperand(0).getReg() &&
        StringRef(LS.MRI->getName(DI.Inst.getOperand(0).getReg())) ==
            "SGPR30_SGPR31";
    if (IsSetPc && ProvenDirectCallReturns.contains(DI.Offset)) {
      log() << "hotswap: recognized proven direct-call return at 0x"
            << utohexstr(DI.Offset) << "\n";
      continue;
    }
    if (std::optional<MaterializedPcTransfer> Transfer =
            evaluateProvenPcTransfer(
                Decoded, I, DirectControlFlowTargets, Text, LS, Elf,
                CompatibilityFacts, TextSymbolOffsets, TextSymbolExtents)) {
      const bool CompatibilityResolved =
          findResolvedCompatibilityCall(CompatibilityFacts, I) != nullptr;
      // Arbitrary-link calls proven by the later analysis may intentionally
      // enter a protected instruction boundary. ABI-standard link calls remain
      // subject to the PR's public callable-entry contract even when the later
      // pattern matcher recovered an exact address.
      if (LS.MIA->isCall(DI.Inst) &&
          (!CompatibilityResolved || isStandardLinkCall(DI, LS))) {
        std::optional<ElfView::FunctionTextRange> TargetRange =
            Elf.findFunctionTextRangeAtOffset(Transfer->Target);
        if (!TargetRange || Transfer->Target != TargetRange->Begin) {
          HasUnknownArbitraryIndirectTarget = true;
          if (Range && Functions.insert(Range->Begin).second)
            log() << "hotswap: source relocation disabled for function at 0x"
                  << utohexstr(Range->Begin)
                  << " by materialized call to a non-entry target at 0x"
                  << utohexstr(DI.Offset) << "\n";
          continue;
        }
      }
      log() << "hotswap: recognized materialized PC transfer [0x"
            << utohexstr(Transfer->Begin) << ", 0x" << utohexstr(Transfer->End)
            << ") -> 0x" << utohexstr(Transfer->Target) << "\n";
      continue;
    }
    if (findResolvedCompatibilityCall(CompatibilityFacts, I)) {
      log() << "hotswap: recognized resolved direct call at 0x"
            << utohexstr(DI.Offset) << "\n";
      continue;
    }
    // A standard-link return can leave from any basic block, but a recognized
    // call-tail gateway using s[30:31] must be classified above first.
    if (IsSetPc && UsesStandardLinkPair)
      continue;
    if (isStandardLinkCall(DI, LS)) {
      log() << "hotswap: recognized ABI standard-link indirect call at 0x"
            << utohexstr(DI.Offset) << "\n";
      continue;
    }
    HasUnknownArbitraryIndirectTarget = true;
    if (Range && Functions.insert(Range->Begin).second)
      log() << "hotswap: source relocation disabled for function at 0x"
            << utohexstr(Range->Begin) << " by " << DI.Mnemonic << " at 0x"
            << utohexstr(DI.Offset) << "\n";
  }
  return Functions;
}

struct VgprMsbState {
  int8_t Dst = VgprMsbUnreachable;
  int8_t Src0 = VgprMsbUnreachable;
  int8_t Src1 = VgprMsbUnreachable;
  int8_t Src2 = VgprMsbUnreachable;
};

static VgprMsbState vgprMsbStateFromMode(unsigned Mode) {
  Mode &= 0xff;
  return {static_cast<int8_t>((Mode >> 6) & 0x3),
          static_cast<int8_t>(Mode & 0x3),
          static_cast<int8_t>((Mode >> 2) & 0x3),
          static_cast<int8_t>((Mode >> 4) & 0x3)};
}

static VgprMsbState unknownVgprMsbState() {
  return {VgprMsbUnknown, VgprMsbUnknown, VgprMsbUnknown, VgprMsbUnknown};
}

static int16_t exactVgprMsbMode(VgprMsbState State) {
  if (State.Dst < 0 || State.Src0 < 0 || State.Src1 < 0 || State.Src2 < 0)
    return VgprMsbUnknown;
  return static_cast<int16_t>(State.Src0 | (State.Src1 << 2) |
                              (State.Src2 << 4) | (State.Dst << 6));
}

// COMGR intentionally does not depend on AMDGPU's backend-private
// AMDGPUBaseInfo.h. Mirror the gfx1250 SETREG-to-VGPR-MSB conversion here:
// the simm16 low six bits select HW_REG_WAVE_MODE (ID 1), and gfx1250's
// setreg-vgpr-msb-fixup makes an immediate MODE write load all four two-bit
// fields from immediate bits [19:12], rotated into s_set_vgpr_msb order.
static std::optional<unsigned>
decodeSetregImmVgprMsbMode(const InternalDecodedInst &DI) {
  if (DI.Mnemonic != "s_setreg_imm32_b32" || DI.Inst.getNumOperands() != 2 ||
      !DI.Inst.getOperand(0).isImm() || !DI.Inst.getOperand(1).isImm())
    return std::nullopt;
  constexpr unsigned ModeHwregId = 1;
  unsigned Simm16 = static_cast<unsigned>(DI.Inst.getOperand(1).getImm());
  if ((Simm16 & 0x3f) != ModeHwregId)
    return std::nullopt;
  unsigned Raw =
      (static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) >> 12) & 0xff;
  return ((Raw >> 2) | (Raw << 6)) & 0xff;
}

static std::optional<unsigned> getSetregHwregId(const InternalDecodedInst &DI) {
  if (!StringRef(DI.Mnemonic).starts_with("s_setreg") ||
      DI.Inst.getNumOperands() == 0 ||
      !DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm())
    return std::nullopt;
  return static_cast<unsigned>(
             DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm()) &
         0x3f;
}

static bool instructionDefinesNamedRegister(const InternalDecodedInst &DI,
                                            StringRef Name,
                                            const LLVMState &LS) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != NumDefs; ++I) {
    const MCOperand &Op = DI.Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() &&
        StringRef(LS.MRI->getName(Op.getReg())) == Name)
      return true;
  }
  return llvm::any_of(Desc.implicit_defs(), [&](MCPhysReg Reg) {
    return StringRef(LS.MRI->getName(Reg)) == Name;
  });
}

static std::optional<unsigned>
getExactVgprMsbModeWritten(const InternalDecodedInst &DI) {
  if (DI.Mnemonic == "s_set_vgpr_msb") {
    if (DI.Inst.getNumOperands() != 1 || !DI.Inst.getOperand(0).isImm())
      return std::nullopt;
    return static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) & 0xff;
  }
  return decodeSetregImmVgprMsbMode(DI);
}

static VgprMsbState transferVgprMsbState(VgprMsbState In,
                                         const InternalDecodedInst &DI,
                                         const LLVMState &LS) {
  if (In.Dst == VgprMsbUnreachable)
    return In;
  if (std::optional<unsigned> Mode = getExactVgprMsbModeWritten(DI))
    return vgprMsbStateFromMode(*Mode);
  if (DI.Mnemonic == "s_set_vgpr_msb")
    return unknownVgprMsbState();
  if (StringRef(DI.Mnemonic).starts_with("s_setreg")) {
    std::optional<unsigned> HwregId = getSetregHwregId(DI);
    if (!HwregId)
      return unknownVgprMsbState();
    constexpr unsigned ModeHwregId = 1;
    if (*HwregId != ModeHwregId)
      return In;
    return unknownVgprMsbState();
  }
  if (DI.Mnemonic == "<unknown>" ||
      instructionDefinesNamedRegister(DI, "MODE", LS) ||
      (LS.MIA && LS.MIA->isCall(DI.Inst)))
    return unknownVgprMsbState();
  return In;
}

bool isStandardLinkReturn(const InternalDecodedInst &DI, const LLVMState &LS) {
  return DI.Inst.getOpcode() == LS.SSetPcI64Opcode &&
         DI.Inst.getNumOperands() == 1 && DI.Inst.getOperand(0).isReg() &&
         DI.Inst.getOperand(0).getReg() &&
         StringRef(LS.MRI->getName(DI.Inst.getOperand(0).getReg())) ==
             "SGPR30_SGPR31";
}

static int8_t mergeVgprMsbValue(int8_t Old, int8_t Incoming) {
  if (Old == VgprMsbUnreachable)
    return Incoming;
  if (Incoming == VgprMsbUnreachable || Old == Incoming)
    return Old;
  return VgprMsbUnknown;
}

static VgprMsbState mergeVgprMsbState(VgprMsbState Old, VgprMsbState Incoming) {
  return {mergeVgprMsbValue(Old.Dst, Incoming.Dst),
          mergeVgprMsbValue(Old.Src0, Incoming.Src0),
          mergeVgprMsbValue(Old.Src1, Incoming.Src1),
          mergeVgprMsbValue(Old.Src2, Incoming.Src2)};
}

static void clearNumberedSgprAliases(BitVector &Aligned, MCRegister Reg,
                                     const MCRegisterInfo &MRI) {
  auto Clear = [&](MCRegister Candidate) {
    if (std::optional<unsigned> Index = numberedSgprIndex(MRI, Candidate))
      if (*Index < Aligned.size())
        Aligned.reset(*Index);
  };
  Clear(Reg);
  for (MCPhysReg Sub : MRI.subregs(Reg))
    Clear(MCRegister(Sub));
  for (MCPhysReg Super : MRI.superregs(Reg))
    Clear(MCRegister(Super));
}

/// Track numbered SGPR values that are known to be divisible by eight. Calls
/// lose every fact because post-link code has no IPRA regmask; ordinary defs
/// kill aliases, and an exact s_mul_i32 by an aligned immediate regenerates
/// the destination fact independently of its other operand.
static BitVector transferSgprAlignment(BitVector In,
                                       const InternalDecodedInst &DI,
                                       ArrayRef<uint8_t> Text,
                                       const LLVMState &LS) {
  if (DI.Mnemonic == "<unknown>" || LS.MIA->isCall(DI.Inst)) {
    In.reset();
    return In;
  }

  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != NumDefs; ++I) {
    const MCOperand &Op = DI.Inst.getOperand(I);
    if (Op.isReg() && Op.getReg())
      clearNumberedSgprAliases(In, MCRegister(Op.getReg()), *LS.MRI);
  }
  for (MCPhysReg Reg : Desc.implicit_defs())
    clearNumberedSgprAliases(In, MCRegister(Reg), *LS.MRI);

  if (DI.Mnemonic != "s_mul_i32" || NumDefs != 1 ||
      !DI.Inst.getOperand(0).isReg())
    return In;
  std::optional<unsigned> Dst =
      numberedSgprIndex(*LS.MRI, MCRegister(DI.Inst.getOperand(0).getReg()));
  if (!Dst || *Dst >= In.size())
    return In;
  for (unsigned I = NumDefs; I != DI.Inst.getNumOperands(); ++I) {
    std::optional<int64_t> Value =
        getAbsoluteOperandValue(DI.Inst.getOperand(I), DI, Text);
    if (Value && (static_cast<uint32_t>(*Value) & 7u) == 0) {
      In.set(*Dst);
      break;
    }
  }
  return In;
}

static bool isAlignedSgprSource(const MCOperand &Op, const BitVector &Aligned,
                                const MCRegisterInfo &MRI) {
  if (!Op.isReg() || !Op.getReg())
    return false;
  std::optional<unsigned> Index =
      numberedSgprIndex(MRI, MCRegister(Op.getReg()));
  return Index && *Index < Aligned.size() && Aligned.test(*Index);
}

static void recordAlignedVgprCopies(const InternalDecodedInst &DI,
                                    size_t GlobalIndex,
                                    const BitVector &Aligned,
                                    const LLVMState &LS, BitVector &Def0,
                                    BitVector &Def1) {
  Def0.reset(GlobalIndex);
  Def1.reset(GlobalIndex);
  if (DI.Mnemonic == "v_mov_b32") {
    if (DI.Inst.getNumOperands() == 2 &&
        isAlignedSgprSource(DI.Inst.getOperand(1), Aligned, *LS.MRI))
      Def0.set(GlobalIndex);
    return;
  }
  if (DI.Mnemonic != "v_dual_mov_b32" || DI.Inst.getNumOperands() != 4)
    return;
  if (isAlignedSgprSource(DI.Inst.getOperand(2), Aligned, *LS.MRI))
    Def0.set(GlobalIndex);
  if (isAlignedSgprSource(DI.Inst.getOperand(3), Aligned, *LS.MRI))
    Def1.set(GlobalIndex);
}

/// Recover the persistent VGPR bank selectors on every path. The complete
/// four-field state feeds WMMA splitting, while the independently merged
/// dst/src0 fields also prove the equality needed by DS address rewrites.
/// Unknown MODE definitions and calls lose the affected proof, and functions
/// with arbitrary indirect entries are rejected.
static BitVector computeVgprMsbDstSrc0Equality(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    const ElfView &Elf, const DenseSet<uint64_t> &DirectControlFlowTargets,
    const DenseSet<uint64_t> &CodeEntries,
    const DenseSet<uint64_t> &IndirectControlFlowFunctions,
    ArrayRef<uint8_t> Text,
    const CompatibilityControlFlowFacts &CompatibilityFacts,
    BitVector &VgprDef0AlignedTo8,
    BitVector &VgprDef1AlignedTo8, std::vector<int8_t> &VgprMsbDstBefore,
    std::vector<int8_t> &VgprMsbSrc0Before,
    std::vector<int16_t> &VgprMsbModeBefore,
    DenseSet<uint64_t> &CrossFunctionInteriorEntryFunctions) {
  BitVector ProvenEqual(Decoded.size());
  VgprDef0AlignedTo8 = BitVector(Decoded.size());
  VgprDef1AlignedTo8 = BitVector(Decoded.size());
  VgprMsbDstBefore.assign(Decoded.size(), VgprMsbUnknown);
  VgprMsbSrc0Before.assign(Decoded.size(), VgprMsbUnknown);
  VgprMsbModeBefore.assign(Decoded.size(), VgprMsbUnanalyzed);
  if (!LS.MIA || !LS.MCII || !LS.MRI)
    return ProvenEqual;

  DenseSet<uint64_t> CrossFunctionInteriorEntries;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    std::optional<uint64_t> Target;
    if (std::optional<MaterializedPcTransfer> Transfer =
            getResolvedCompatibilityTransfer(CompatibilityFacts, I)) {
      Target = Transfer->Target;
    } else {
      if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
          LS.MIA->isIndirectBranch(DI.Inst) || LS.MIA->isReturn(DI.Inst))
        continue;
      Target = evaluateDirectControlFlowTarget(DI, LS);
    }
    if (!Target || *Target >= Text.size())
      continue;
    std::optional<ElfView::FunctionTextRange> SourceOwner =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    std::optional<ElfView::FunctionTextRange> TargetOwner =
        Elf.findFunctionTextRangeAtOffset(*Target);
    if (TargetOwner && *Target != TargetOwner->Begin &&
        (!SourceOwner || SourceOwner->Begin != TargetOwner->Begin ||
         SourceOwner->End != TargetOwner->End))
      CrossFunctionInteriorEntries.insert(TargetOwner->Begin);
  }
  CrossFunctionInteriorEntryFunctions = CrossFunctionInteriorEntries;

  DenseSet<std::pair<uint64_t, uint64_t>> SeenRanges;
  std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Elf.functionTextRanges();
  for (size_t RangeIndex = 0; RangeIndex != FunctionRanges.size();
       ++RangeIndex) {
    const ElfView::FunctionTextRange &VirtualRange = FunctionRanges[RangeIndex];
    if (VirtualRange.Begin < Elf.textAddr() ||
        VirtualRange.End < VirtualRange.Begin ||
        VirtualRange.End > Elf.textAddr() + Text.size())
      continue;
    uint64_t Begin = VirtualRange.Begin - Elf.textAddr();
    uint64_t End = VirtualRange.End - Elf.textAddr();
    if (Begin >= End || !SeenRanges.insert({Begin, End}).second ||
        IndirectControlFlowFunctions.contains(Begin) ||
        CrossFunctionInteriorEntries.contains(Begin))
      continue;

    // An outer symbol must not donate facts to instructions owned by a nested
    // function symbol. The ranges are sorted by start address, so the first
    // later distinct start is sufficient to detect an overlap.
    size_t NextRange = RangeIndex + 1;
    while (NextRange != FunctionRanges.size() &&
           FunctionRanges[NextRange].Begin == VirtualRange.Begin)
      ++NextRange;
    if (NextRange != FunctionRanges.size() &&
        FunctionRanges[NextRange].Begin < VirtualRange.End)
      continue;

    // Ignore an overlapping/alias range unless ElfView would select it as the
    // owner of its first instruction. This keeps the CFG and the later patch
    // site's function identity consistent.
    std::optional<ElfView::FunctionTextRange> Owner =
        Elf.findFunctionTextRangeAtOffset(Begin);
    if (!Owner || Owner->Begin != Begin || Owner->End != End)
      continue;

    auto First = llvm::lower_bound(
        Decoded, Begin, [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    auto After = llvm::lower_bound(
        Decoded, End, [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    if (First == After || First->Offset != Begin)
      continue;

    const size_t GlobalFirst = static_cast<size_t>(First - Decoded.begin());
    const size_t Count = static_cast<size_t>(After - First);
    DenseMap<uint64_t, unsigned> OffsetToLocalIndex;
    OffsetToLocalIndex.reserve(Count);
    bool Valid = true;
    for (unsigned I = 0; I != Count; ++I) {
      OffsetToLocalIndex.try_emplace(First[I].Offset, I);
      if (First[I].Mnemonic == "<unknown>")
        Valid = false;
    }
    if (!Valid)
      continue;

    std::vector<SmallVector<unsigned, 2>> Successors(Count);
    BitVector CallableEntries(Count);
    auto AddTarget = [&](SmallVectorImpl<unsigned> &Out, uint64_t Target) {
      if (Target < Begin || Target >= End)
        return true;
      auto It = OffsetToLocalIndex.find(Target);
      if (It == OffsetToLocalIndex.end())
        return false;
      Out.push_back(It->second);
      return true;
    };
    auto AddFallthrough = [&](SmallVectorImpl<unsigned> &Out, unsigned I) {
      if (I + 1 < Count)
        Out.push_back(I + 1);
    };

    for (unsigned I = 0; I != Count && Valid; ++I) {
      const InternalDecodedInst &DI = First[I];
      SmallVectorImpl<unsigned> &Out = Successors[I];
      if (DI.Mnemonic == "s_endpgm" || DI.Mnemonic == "s_endpgm_saved" ||
          LS.MIA->isReturn(DI.Inst) || isStandardLinkReturn(DI, LS))
        continue;
      if (LS.MIA->isCall(DI.Inst)) {
        std::optional<uint64_t> Target;
        if (std::optional<MaterializedPcTransfer> Transfer =
                getResolvedCompatibilityTransfer(CompatibilityFacts,
                                                 GlobalFirst + I))
          Target = Transfer->Target;
        else if (!LS.MIA->isIndirectBranch(DI.Inst))
          Target = evaluateDirectControlFlowTarget(DI, LS);
        if (Target && *Target >= Begin && *Target < End)
          if (auto It = OffsetToLocalIndex.find(*Target);
              It != OffsetToLocalIndex.end())
            CallableEntries.set(It->second);
        AddFallthrough(Out, I);
        continue;
      }
      if (LS.MIA->isBranch(DI.Inst)) {
        std::optional<uint64_t> Target;
        std::optional<MaterializedPcTransfer> Materialized =
            evaluateProvenPcTransfer(
                Decoded, GlobalFirst + I, DirectControlFlowTargets, Text, LS,
                Elf, CompatibilityFacts);
        if (Materialized) {
          Target = Materialized->Target;
        } else if (LS.MIA->isIndirectBranch(DI.Inst)) {
          Valid = false;
          break;
        } else {
          Target = evaluateDirectControlFlowTarget(DI, LS);
          if (!Target) {
            Valid = false;
            break;
          }
        }
        Valid &= AddTarget(Out, *Target);
        if (LS.MIA->isConditionalBranch(DI.Inst))
          AddFallthrough(Out, I);
        else if (!Materialized && !LS.MIA->isUnconditionalBranch(DI.Inst) &&
                 !LS.MIA->isIndirectBranch(DI.Inst))
          Valid = false;
        continue;
      }
      if (LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI)) {
        Valid = false;
        break;
      }
      AddFallthrough(Out, I);
    }
    if (!Valid)
      continue;

    // A validated function CFG distinguishes dead instructions from sites we
    // could not analyze at all. Mandatory rewrites may use that distinction to
    // transform semantically unreachable instructions without treating an
    // ambiguous reachable MODE value as the ABI default.
    for (size_t I = 0; I != Count; ++I)
      VgprMsbModeBefore[GlobalFirst + I] = VgprMsbUnreachable;

    std::vector<VgprMsbState> In(Count);
    std::vector<BitVector> AlignedIn(Count, BitVector(Gfx1250MaxSgprs));
    BitVector AlignmentReachable(Count);
    SmallVector<unsigned, 64> Worklist;
    // LLVM's gfx1250 VGPR-lowering ABI explicitly requires all four MSB fields
    // to be zero on function entry. Calls still transfer to Unknown above
    // because this object-level proof does not inspect the callee's return.
    auto SeedAbiEntry = [&](unsigned I) {
      VgprMsbState Seed = vgprMsbStateFromMode(0);
      In[I] = mergeVgprMsbState(In[I], Seed);
      AlignmentReachable.set(I);
      Worklist.push_back(I);
    };
    SeedAbiEntry(0);
    for (uint64_t Entry : CodeEntries)
      if (Entry >= Begin && Entry < End)
        if (auto It = OffsetToLocalIndex.find(Entry);
            It != OffsetToLocalIndex.end() && It->second != 0)
          SeedAbiEntry(It->second);
    for (int I = CallableEntries.find_first(); I >= 0;
         I = CallableEntries.find_next(I))
      if (I != 0)
        SeedAbiEntry(static_cast<unsigned>(I));
    for (size_t Next = 0; Next != Worklist.size(); ++Next) {
      unsigned I = Worklist[Next];
      size_t GlobalI = GlobalFirst + I;
      VgprMsbDstBefore[GlobalI] = In[I].Dst;
      VgprMsbSrc0Before[GlobalI] = In[I].Src0;
      VgprMsbModeBefore[GlobalI] = exactVgprMsbMode(In[I]);
      if (In[I].Dst >= 0 && In[I].Dst == In[I].Src0)
        ProvenEqual.set(GlobalI);
      else
        ProvenEqual.reset(GlobalI);
      recordAlignedVgprCopies(Decoded[GlobalI], GlobalI, AlignedIn[I], LS,
                              VgprDef0AlignedTo8, VgprDef1AlignedTo8);
      VgprMsbState Out = transferVgprMsbState(In[I], Decoded[GlobalI], LS);
      BitVector AlignedOut =
          transferSgprAlignment(AlignedIn[I], Decoded[GlobalI], Text, LS);
      for (unsigned Succ : Successors[I]) {
        VgprMsbState Merged = mergeVgprMsbState(In[Succ], Out);
        bool Changed =
            Merged.Dst != In[Succ].Dst || Merged.Src0 != In[Succ].Src0 ||
            Merged.Src1 != In[Succ].Src1 || Merged.Src2 != In[Succ].Src2;
        if (Changed)
          In[Succ] = Merged;
        if (!AlignmentReachable.test(Succ)) {
          AlignedIn[Succ] = AlignedOut;
          AlignmentReachable.set(Succ);
          Changed = true;
        } else {
          BitVector AlignedMerged = AlignedIn[Succ];
          AlignedMerged &= AlignedOut;
          if (AlignedMerged != AlignedIn[Succ]) {
            AlignedIn[Succ] = std::move(AlignedMerged);
            Changed = true;
          }
        }
        if (Changed)
          Worklist.push_back(Succ);
      }
    }
  }
  return ProvenEqual;
}

/// Grow undersized far-site windows only through proven straight-line code.
/// Patched neighbors are merged; ordinary instructions are copied verbatim
/// into the trampoline body and retain their original order. Growth is bounded
/// by the space required for the selected SCC-neutral set-PC sequence.
static void
expandStraightLineTrampolines(PatchContext &Ctx,
                              const DenseSet<uint64_t> &DirectBranchTargets) {
  DenseMap<uint64_t, size_t> DecodedAt;
  for (size_t I = 0; I != Ctx.Decoded.size(); ++I)
    DecodedAt[Ctx.Decoded[I].Offset] = I;

  SmallVector<uint64_t, 16> SortedDirectTargets(DirectBranchTargets.begin(),
                                                DirectBranchTargets.end());
  llvm::sort(SortedDirectTargets);
  auto HasTargetInRange = [&](uint64_t Begin, uint64_t End, bool IncludeBegin) {
    auto It = llvm::lower_bound(SortedDirectTargets, Begin);
    if (!IncludeBegin && It != SortedDirectTargets.end() && *It == Begin)
      ++It;
    return It != SortedDirectTargets.end() && *It < End;
  };

  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    if (Ctx.OutTrampolines[I].HasFunctionRange &&
        Ctx.IndirectControlFlowFunctions.contains(
            Ctx.OutTrampolines[I].FunctionStart))
      continue;
    while (Ctx.OutTrampolines[I].Long &&
           Ctx.OutTrampolines[I].OriginalSize < SetPcForwardSequenceBytes) {
      Trampoline &T = Ctx.OutTrampolines[I];
      std::optional<uint64_t> End = checkedAddUint64(
          T.OriginalOffset, T.OriginalSize, "straight-line expansion end");
      if (!End || DirectBranchTargets.contains(*End))
        break;

      if (I + 1 < Ctx.OutTrampolines.size() &&
          Ctx.OutTrampolines[I + 1].OriginalOffset == *End) {
        Trampoline &Next = Ctx.OutTrampolines[I + 1];
        if (!Next.Long || !Next.UsesSetPCBack ||
            Next.LongBranchSgprBase != T.LongBranchSgprBase ||
            Next.LongBranchScratchIsSiteProven !=
                T.LongBranchScratchIsSiteProven ||
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
      const InternalDecodedInst &OriginalDI = Ctx.Decoded[It->second];
      std::optional<InternalDecodedInst> CurrentDI =
          decodeCurrentRelocationCandidate(Ctx, OriginalDI);
      if (!CurrentDI)
        break;
      const InternalDecodedInst &DI = *CurrentDI;
      std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset);
      if (!Range || !T.HasFunctionRange || Range->Begin != T.FunctionStart ||
          Range->End != T.FunctionEnd ||
          hasSiteReplacementReservation(Ctx, DI.Offset) ||
          HasTargetInRange(DI.Offset, DI.Offset + DI.Size,
                           /*IncludeBegin=*/true) ||
          !isSafeStraightLineRelocation(DI, Ctx.LS,
                                        Ctx.RelocationProtectedOffsets) ||
          DI.Offset > Ctx.TextSize || DI.Size > Ctx.TextSize - DI.Offset ||
          T.Bytes.size() < SetPcReturnReserveBytes)
        break;

      if (T.LongBranchScratchIsSiteProven) {
        // The original proof is tied to the old resume offset. Re-prove the
        // chosen pair after every prospective moved instruction. The forward
        // set-PC edge executes before the moved instruction, so also reject a
        // semantic read of that pair. A pure definition is safe only when the
        // new resume-point proof shows that its value is dead.
        unsigned Pair = T.LongBranchSgprBase;
        if (Pair == Gfx1250MaxSgprs) {
          std::optional<std::array<MCRegister, 2>> PairRegs =
              findVccLoHiPair(Ctx.LS);
          std::optional<RegisterPairAccess> Access =
              PairRegs ? getRegisterPairAccess(Ctx.LS, DI, *PairRegs,
                                               /*PairIsAbiVcc=*/true)
                       : std::nullopt;
          if (!Access || Access->Uses ||
              !isVccPairDeadAfter(Ctx, DI.Offset, DI.Size))
            break;
        } else {
          std::optional<std::array<MCRegister, 2>> PairRegs =
              findNumberedSgprPair(Ctx.LS, Pair);
          std::optional<RegisterPairAccess> Access =
              PairRegs ? getRegisterPairAccess(Ctx.LS, DI, *PairRegs)
                       : std::nullopt;
          std::optional<BitVector> SiteDead =
              getSiteDeadNumberedSgprs(Ctx, DI.Offset, DI.Size);
          if (!Access || Access->Uses || !SiteDead ||
              Pair + 1 >= SiteDead->size() || !SiteDead->test(Pair) ||
              !SiteDead->test(Pair + 1))
            break;
        }
      }

      T.Bytes.insert(T.Bytes.end() - SetPcReturnReserveBytes,
                     Ctx.Text + DI.Offset, Ctx.Text + DI.Offset + DI.Size);
      T.OriginalSize += DI.Size;
    }

    while (Ctx.OutTrampolines[I].Long &&
           Ctx.OutTrampolines[I].OriginalSize < SetPcForwardSequenceBytes) {
      Trampoline &T = Ctx.OutTrampolines[I];
      // A target at the current window start may remain there, but prepending
      // another instruction would turn it into an interior entry. The selected
      // predecessor may itself be a target because it becomes the new start.
      if (DirectBranchTargets.contains(T.OriginalOffset))
        break;
      DenseMap<uint64_t, size_t>::const_iterator It =
          DecodedAt.find(T.OriginalOffset);
      if (It == DecodedAt.end() || It->second == 0)
        break;
      const InternalDecodedInst &OriginalDI = Ctx.Decoded[It->second - 1];
      if (OriginalDI.Offset + OriginalDI.Size != T.OriginalOffset)
        break;
      std::optional<InternalDecodedInst> CurrentDI =
          decodeCurrentRelocationCandidate(Ctx, OriginalDI);
      if (!CurrentDI)
        break;
      const InternalDecodedInst &DI = *CurrentDI;
      if (DI.Offset + DI.Size != T.OriginalOffset ||
          hasSiteReplacementReservation(Ctx, DI.Offset) ||
          HasTargetInRange(DI.Offset, DI.Offset + DI.Size,
                           /*IncludeBegin=*/false) ||
          !isSafeStraightLineRelocation(DI, Ctx.LS,
                                        Ctx.RelocationProtectedOffsets))
        break;
      if (I != 0) {
        const Trampoline &Previous = Ctx.OutTrampolines[I - 1];
        if (Previous.OriginalOffset + Previous.OriginalSize > DI.Offset)
          break;
      }
      std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset);
      if (!Range || !T.HasFunctionRange || Range->Begin != T.FunctionStart ||
          Range->End != T.FunctionEnd || DI.Offset > Ctx.TextSize ||
          DI.Size > Ctx.TextSize - DI.Offset)
        break;

      if (T.LongBranchScratchIsSiteProven) {
        // Growing backward leaves the resume point unchanged, so the original
        // dead-after-resume proof remains valid. The new forward set-PC edge
        // clobbers the pair before the prepended instruction executes, so the
        // instruction may define the pair but must not read its old value.
        unsigned Pair = T.LongBranchSgprBase;
        if (Pair == Gfx1250MaxSgprs) {
          std::optional<std::array<MCRegister, 2>> PairRegs =
              findVccLoHiPair(Ctx.LS);
          std::optional<RegisterPairAccess> Access =
              PairRegs ? getRegisterPairAccess(Ctx.LS, DI, *PairRegs,
                                               /*PairIsAbiVcc=*/true)
                       : std::nullopt;
          if (!Access || Access->Uses)
            break;
        } else {
          std::optional<std::array<MCRegister, 2>> PairRegs =
              findNumberedSgprPair(Ctx.LS, Pair);
          std::optional<RegisterPairAccess> Access =
              PairRegs ? getRegisterPairAccess(Ctx.LS, DI, *PairRegs)
                       : std::nullopt;
          if (!Access || Access->Uses)
            break;
        }
      }

      T.Bytes.insert(T.Bytes.begin(), Ctx.Text + DI.Offset,
                     Ctx.Text + DI.Offset + DI.Size);
      T.OriginalOffset = DI.Offset;
      T.OriginalSize += DI.Size;
    }
  }
}

static bool hasNoFallthrough(const InternalDecodedInst &DI,
                             const LLVMState &LS) {
  if (DI.Mnemonic == "s_code_end" || DI.Mnemonic == "s_endpgm" ||
      DI.Mnemonic == "s_endpgm_saved")
    return true;
  if (!LS.MIA || LS.MIA->isCall(DI.Inst))
    return false;
  if (LS.MIA->isUnconditionalBranch(DI.Inst) &&
      !LS.MIA->isIndirectBranch(DI.Inst))
    return true;
  return DI.Inst.getOpcode() == LS.SSetPcI64Opcode;
}

static void appendGatewaySled(std::vector<NopSled> &Sleds, uint64_t Start,
                              uint64_t End, uint64_t TextSize, bool Safe,
                              bool HasTarget) {
  if (Safe && !HasTarget && End - Start >= MinInstSize)
    Sleds.push_back({Start, End, Start, 0, TextSize,
                     /*GatewayOnly=*/true});
}

static std::optional<DenseSet<uint64_t>>
collectCodeObjectEntryOffsets(const ElfView &Elf, uint64_t TextSize) {
  if (!Elf.kernelDescriptorCacheIsComplete()) {
    log() << "hotswap: incomplete kernel descriptor set prevents complete "
             "code-entry discovery\n";
    return std::nullopt;
  }

  DenseSet<uint64_t> Entries;
  const uint64_t TextAddr = Elf.textAddr();

  // A zero-filled function body is indistinguishable from alignment padding
  // to the decoder. Preserve every callable symbol entry, including aliases
  // and zero-sized functions, independently of whether direct control flow in
  // this object names it.
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges()) {
    if (Range.Begin >= TextAddr && Range.Begin - TextAddr < TextSize)
      Entries.insert(Range.Begin - TextAddr);
  }

  // Kernel descriptors may name an entry that has no STT_FUNC symbol. The
  // signed entry offset is relative to the descriptor address.
  for (const KernelDescriptorInfo &KD : Elf.kernelDescriptors()) {
    uint64_t Entry = 0;
    if (KD.EntryOffset >= 0) {
      uint64_t Delta = static_cast<uint64_t>(KD.EntryOffset);
      if (Delta > std::numeric_limits<uint64_t>::max() - KD.VAddr)
        continue;
      Entry = KD.VAddr + Delta;
    } else {
      uint64_t Magnitude =
          KD.EntryOffset == std::numeric_limits<int64_t>::min()
              ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1
              : static_cast<uint64_t>(-KD.EntryOffset);
      if (Magnitude > KD.VAddr)
        continue;
      Entry = KD.VAddr - Magnitude;
    }
    if (Entry >= TextAddr && Entry - TextAddr < TextSize)
      Entries.insert(Entry - TextAddr);
  }
  return Entries;
}

/// Find zero-filled alignment holes, including holes covered by an oversized
/// function symbol, and s_nop padding outside every function. Such padding is
/// a safe branch gateway only when it follows a no-fallthrough instruction and
/// contains no known control-flow target or callable entry. Body-owned
/// s_nop runs are excluded so one physical range never has two allocation
/// cursors.
static std::vector<NopSled>
buildExternalGatewaySleds(ArrayRef<InternalDecodedInst> Decoded,
                          const LLVMState &LS, const ElfView &Elf,
                          ArrayRef<uint8_t> Text,
                          const DenseSet<uint64_t> &DirectBranchTargets,
                          ArrayRef<uint64_t> TextSymbolOffsets,
                          ArrayRef<ElfView::TextOffsetRange> TextSymbolExtents,
                          ArrayRef<NopSled> OwnedSleds) {
  std::vector<NopSled> Sleds;
  SmallVector<uint64_t, 16> ProtectedEntries(DirectBranchTargets.begin(),
                                             DirectBranchTargets.end());
  llvm::sort(ProtectedEntries);
  ProtectedEntries.erase(
      std::unique(ProtectedEntries.begin(), ProtectedEntries.end()),
      ProtectedEntries.end());

  const InternalDecodedInst *Previous = nullptr;
  bool Active = false;
  bool Safe = false;
  bool HasTarget = false;
  uint64_t Start = 0;
  uint64_t End = 0;
  size_t OwnedIndex = 0;
  for (const InternalDecodedInst &DI : Decoded) {
    while (OwnedIndex != OwnedSleds.size() &&
           OwnedSleds[OwnedIndex].End <= DI.Offset)
      ++OwnedIndex;
    const bool BodyOwned = OwnedIndex != OwnedSleds.size() &&
                           OwnedSleds[OwnedIndex].Start <= DI.Offset &&
                           DI.Offset < OwnedSleds[OwnedIndex].End;
    bool ZeroPadding =
        DI.Offset <= Text.size() && DI.Size <= Text.size() - DI.Offset;
    if (ZeroPadding)
      for (uint8_t Byte : Text.slice(DI.Offset, DI.Size))
        ZeroPadding &= Byte == 0;
    bool IsExternalNop = DI.Inst.getOpcode() == LS.SNopOpcode &&
                         !Elf.findFunctionTextRangeAtOffset(DI.Offset);
    bool GatewayPadding = !BodyOwned && (ZeroPadding || IsExternalNop);
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
    auto Entry = llvm::lower_bound(ProtectedEntries, DI.Offset);
    const bool HasHardEntry =
        Entry != ProtectedEntries.end() && *Entry - DI.Offset < DI.Size;
    auto Symbol = llvm::lower_bound(TextSymbolOffsets, DI.Offset);
    const bool HasSymbol =
        Symbol != TextSymbolOffsets.end() && *Symbol - DI.Offset < DI.Size;
    const bool HasSymbolExtent =
        overlapsTextSymbolExtent(TextSymbolExtents, DI);
    // External gateways are donor storage, so even an exact-start symbol
    // blocks the range. Only an original replacement source may preserve such
    // an alias while changing the bytes at that same address.
    HasTarget |= HasHardEntry || HasSymbol || HasSymbolExtent;
    End = DI.Offset + DI.Size;
  }
  if (Active)
    appendGatewaySled(Sleds, Start, End, Text.size(), Safe, HasTarget);
  return Sleds;
}

static uint64_t countReachableGatewaySlots(const NopSled &Sled, uint64_t Offset,
                                           uint64_t Needed) {
  if (!Sled.canGatewayFrom(Offset) || Sled.WritePos > Sled.End ||
      Needed > Sled.End - Sled.WritePos)
    return 0;
  uint64_t Distance =
      Sled.WritePos > Offset ? Sled.WritePos - Offset : Offset - Sled.WritePos;
  if (Distance >= MaxSledDistance)
    return 0;
  return (Sled.End - Sled.WritePos) / Needed;
}

bool canEverReachGatewaySlot(const NopSled &Sled, uint64_t Offset,
                             uint64_t Needed) {
  if (!Sled.canGatewayFrom(Offset) || Sled.WritePos > Sled.End ||
      Needed > Sled.End - Sled.WritePos)
    return false;

  uint64_t LastWritePos = Sled.End - Needed;
  uint64_t Reach = MaxSledDistance - 1;
  uint64_t ReachableStart = Offset > Reach ? Offset - Reach : 0;
  uint64_t ReachableEnd = Offset > std::numeric_limits<uint64_t>::max() - Reach
                              ? std::numeric_limits<uint64_t>::max()
                              : Offset + Reach;
  return Sled.WritePos <= ReachableEnd && ReachableStart <= LastWritePos;
}

struct GatewaySlotOwner {
  size_t OwnerIndex = 0;
  uint64_t SourceOffset = 0;
  uint64_t NeededBytes = 0;
};

using GatewaySlotOwnerList = SmallVector<GatewaySlotOwner, 2>;

static bool
wouldConsumeLastGatewaySlot(ArrayRef<NopSled> Gateways, size_t SledIndex,
                            uint64_t Bytes,
                            ArrayRef<GatewaySlotOwnerList> OwnersBySled,
                            ArrayRef<uint64_t> CurrentViableSleds,
                            ArrayRef<uint8_t> Done, size_t CurrentOwner) {
  if (SledIndex >= Gateways.size() || SledIndex >= OwnersBySled.size())
    return false;
  const NopSled &Sled = Gateways[SledIndex];
  if (Sled.WritePos > Sled.End || Bytes > Sled.End - Sled.WritePos)
    return false;

  NopSled Advanced = Sled;
  Advanced.WritePos += Bytes;
  for (const GatewaySlotOwner &Owner : OwnersBySled[SledIndex]) {
    if (Owner.OwnerIndex == CurrentOwner ||
        (Owner.OwnerIndex < Done.size() && Done[Owner.OwnerIndex]))
      continue;
    uint64_t TotalBefore = CurrentViableSleds[Owner.OwnerIndex];
    if (TotalBefore == 0)
      continue;
    uint64_t SledBefore =
        canEverReachGatewaySlot(Sled, Owner.SourceOffset, Owner.NeededBytes);
    uint64_t SledAfter = canEverReachGatewaySlot(Advanced, Owner.SourceOffset,
                                                 Owner.NeededBytes);
    assert(SledBefore <= TotalBefore && "gateway slot count is inconsistent");
    if (TotalBefore - SledBefore + SledAfter == 0)
      return true;
  }
  return false;
}

struct GatewaySlotStateChange {
  size_t OwnerIndex = 0;
  uint64_t PreviousCandidateSlots = 0;
  uint64_t PreviousViableSleds = 0;
  std::optional<size_t> PreviousLastDestroyer;
};

using GatewaySlotStateChangeList = SmallVector<GatewaySlotStateChange, 4>;

static GatewaySlotStateChangeList advanceGatewaySled(
    NopSled &Sled, uint64_t NewWritePos, ArrayRef<GatewaySlotOwner> SledOwners,
    MutableArrayRef<uint64_t> CurrentCandidateSlots, ArrayRef<uint8_t> Done,
    MutableArrayRef<uint64_t> CurrentViableSleds, size_t CurrentOwner,
    MutableArrayRef<std::optional<size_t>> LastDestroyer) {
  assert(Sled.WritePos <= NewWritePos && NewWritePos <= Sled.End &&
         "invalid gateway sled cursor advance");
  NopSled Advanced = Sled;
  Advanced.WritePos = NewWritePos;
  GatewaySlotStateChangeList Changes;
  for (const GatewaySlotOwner &Owner : SledOwners) {
    uint64_t SledBefore =
        countReachableGatewaySlots(Sled, Owner.SourceOffset, Owner.NeededBytes);
    uint64_t SledAfter = countReachableGatewaySlots(
        Advanced, Owner.SourceOffset, Owner.NeededBytes);
    uint64_t ViableBefore =
        canEverReachGatewaySlot(Sled, Owner.SourceOffset, Owner.NeededBytes);
    uint64_t ViableAfter = canEverReachGatewaySlot(Advanced, Owner.SourceOffset,
                                                   Owner.NeededBytes);
    if (SledBefore == SledAfter && ViableBefore == ViableAfter)
      continue;
    uint64_t TotalBefore = CurrentCandidateSlots[Owner.OwnerIndex];
    assert(SledBefore <= TotalBefore && "gateway slot count is inconsistent");
    uint64_t TotalViableBefore = CurrentViableSleds[Owner.OwnerIndex];
    assert(ViableBefore <= TotalViableBefore &&
           "gateway viable-sled count is inconsistent");
    Changes.push_back({Owner.OwnerIndex, TotalBefore, TotalViableBefore,
                       LastDestroyer[Owner.OwnerIndex]});
    uint64_t TotalAfter = TotalBefore - SledBefore + SledAfter;
    uint64_t TotalViableAfter = TotalViableBefore - ViableBefore + ViableAfter;
    CurrentCandidateSlots[Owner.OwnerIndex] = TotalAfter;
    CurrentViableSleds[Owner.OwnerIndex] = TotalViableAfter;
    if (TotalViableBefore == 0 && TotalViableAfter != 0)
      LastDestroyer[Owner.OwnerIndex].reset();
    else if (Owner.OwnerIndex != CurrentOwner && !Done[Owner.OwnerIndex] &&
             TotalViableBefore != 0 && TotalViableAfter == 0)
      LastDestroyer[Owner.OwnerIndex] = CurrentOwner;
  }
  Sled.WritePos = NewWritePos;
  return Changes;
}

static void
restoreGatewaySled(NopSled &Sled, uint64_t PreviousWritePos,
                   ArrayRef<GatewaySlotStateChange> Changes,
                   MutableArrayRef<uint64_t> CurrentCandidateSlots,
                   MutableArrayRef<uint64_t> CurrentViableSleds,
                   MutableArrayRef<std::optional<size_t>> LastDestroyer) {
  Sled.WritePos = PreviousWritePos;
  for (const GatewaySlotStateChange &Change : Changes) {
    CurrentCandidateSlots[Change.OwnerIndex] = Change.PreviousCandidateSlots;
    CurrentViableSleds[Change.OwnerIndex] = Change.PreviousViableSleds;
    LastDestroyer[Change.OwnerIndex] = Change.PreviousLastDestroyer;
  }
}

static std::optional<SmallVector<uint64_t, 4>> allocateForwardBranchIslands(
    std::vector<NopSled> &Gateways, uint64_t FromOffset, uint64_t TargetOffset,
    ArrayRef<GatewaySlotOwnerList> OwnersBySled,
    MutableArrayRef<uint64_t> CurrentCandidateSlots,
    MutableArrayRef<uint64_t> CurrentViableSleds, ArrayRef<uint8_t> Done,
    size_t CurrentOwner, MutableArrayRef<std::optional<size_t>> LastDestroyer,
    bool ProtectSlots, size_t MaxAlternateStates, bool LogSearchLimit) {
  static constexpr size_t MaxPathDepth = 2048;

  // Precompute the forward DAG nodes that have any suffix path to the target,
  // ignoring reservations. For a fixed source, the nearest reachable suffix
  // node dominates every farther one, so this reverse scan is linear after
  // sorting and rejects impossible relay routes before bounded backtracking.
  SmallVector<size_t, 64> ForwardSleds;
  for (size_t I = 0; I != Gateways.size(); ++I) {
    const NopSled &Sled = Gateways[I];
    if (!Sled.canGatewayFrom(FromOffset) || Sled.WritePos <= FromOffset ||
        Sled.WritePos >= TargetOffset || Sled.WritePos > Sled.End ||
        MinInstSize > Sled.End - Sled.WritePos)
      continue;
    ForwardSleds.push_back(I);
  }
  llvm::sort(ForwardSleds, [&](size_t LHS, size_t RHS) {
    if (Gateways[LHS].WritePos != Gateways[RHS].WritePos)
      return Gateways[LHS].WritePos < Gateways[RHS].WritePos;
    return LHS < RHS;
  });

  std::vector<uint8_t> CanReachTarget(Gateways.size(), 0);
  std::optional<size_t> NearestReachable;
  size_t GroupEnd = ForwardSleds.size();
  while (GroupEnd != 0) {
    size_t GroupBegin = GroupEnd - 1;
    uint64_t Position = Gateways[ForwardSleds[GroupBegin]].WritePos;
    while (GroupBegin != 0 &&
           Gateways[ForwardSleds[GroupBegin - 1]].WritePos == Position)
      --GroupBegin;
    bool HasSuffix = isSBranchReachable(Position, TargetOffset);
    if (!HasSuffix && NearestReachable)
      HasSuffix =
          isSBranchReachable(Position, Gateways[*NearestReachable].WritePos);
    if (HasSuffix) {
      for (size_t I = GroupBegin; I != GroupEnd; ++I)
        CanReachTarget[ForwardSleds[I]] = 1;
      NearestReachable = ForwardSleds[GroupBegin];
    }
    GroupEnd = GroupBegin;
  }

  bool HasRoute = isSBranchReachable(FromOffset, TargetOffset);
  if (!HasRoute)
    for (size_t I : ForwardSleds)
      if (CanReachTarget[I] &&
          isSBranchReachable(FromOffset, Gateways[I].WritePos)) {
        HasRoute = true;
        break;
      }
  if (!HasRoute)
    return std::nullopt;

  struct Allocation {
    size_t SledIndex = 0;
    uint64_t PreviousWritePos = 0;
    GatewaySlotStateChangeList SlotChanges;
  };
  SmallVector<uint64_t, 4> Islands;

  // Preserve the former farthest-first path as the common fast path. It
  // commits on success and restores every cursor and owner count on failure.
  auto TryGreedy = [&] {
    SmallVector<Allocation, 4> Allocations;
    DenseSet<size_t> UsedSleds;
    uint64_t Current = FromOffset;
    while (!isSBranchReachable(Current, TargetOffset)) {
      size_t BestIndex = Gateways.size();
      uint64_t BestOffset = 0;
      for (size_t I : ForwardSleds) {
        NopSled &Sled = Gateways[I];
        if (!CanReachTarget[I] || UsedSleds.contains(I) ||
            Sled.WritePos <= Current ||
            !isSBranchReachable(Current, Sled.WritePos))
          continue;
        if (ProtectSlots &&
            wouldConsumeLastGatewaySlot(Gateways, I, MinInstSize, OwnersBySled,
                                        CurrentViableSleds, Done, CurrentOwner))
          continue;
        if (BestIndex == Gateways.size() || Sled.WritePos > BestOffset ||
            (Sled.WritePos == BestOffset && I < BestIndex)) {
          BestIndex = I;
          BestOffset = Sled.WritePos;
        }
      }
      if (BestIndex == Gateways.size()) {
        for (size_t I = Allocations.size(); I != 0; --I) {
          const Allocation &A = Allocations[I - 1];
          restoreGatewaySled(Gateways[A.SledIndex], A.PreviousWritePos,
                             A.SlotChanges, CurrentCandidateSlots,
                             CurrentViableSleds, LastDestroyer);
        }
        Islands.clear();
        return false;
      }

      NopSled &Best = Gateways[BestIndex];
      uint64_t PreviousWritePos = Best.WritePos;
      GatewaySlotStateChangeList Changes = advanceGatewaySled(
          Best, Best.WritePos + MinInstSize, OwnersBySled[BestIndex],
          CurrentCandidateSlots, Done, CurrentViableSleds, CurrentOwner,
          LastDestroyer);
      Allocations.push_back({BestIndex, PreviousWritePos, std::move(Changes)});
      Islands.push_back(PreviousWritePos);
      UsedSleds.insert(BestIndex);
      Current = PreviousWritePos;
    }
    return true;
  };

  if (TryGreedy())
    return Islands;
  if (MaxAlternateStates == 0)
    return std::nullopt;

  DenseSet<size_t> UsedSleds;
  size_t ExploredStates = 0;
  bool HitSearchLimit = false;

  auto Search = [&](auto &&Self, uint64_t Current) -> bool {
    if (isSBranchReachable(Current, TargetOffset))
      return true;
    if (Islands.size() >= MaxPathDepth) {
      HitSearchLimit = true;
      return false;
    }

    SmallVector<size_t, 32> Candidates;
    for (size_t I : ForwardSleds) {
      NopSled &Sled = Gateways[I];
      if (!CanReachTarget[I] || UsedSleds.contains(I))
        continue;
      if (Sled.WritePos <= Current ||
          !isSBranchReachable(Current, Sled.WritePos))
        continue;
      if (ProtectSlots &&
          wouldConsumeLastGatewaySlot(Gateways, I, MinInstSize, OwnersBySled,
                                      CurrentViableSleds, Done, CurrentOwner))
        continue;
      Candidates.push_back(I);
    }
    llvm::sort(Candidates, [&](size_t LHS, size_t RHS) {
      if (Gateways[LHS].WritePos != Gateways[RHS].WritePos)
        return Gateways[LHS].WritePos > Gateways[RHS].WritePos;
      return LHS < RHS;
    });

    for (size_t SledIndex : Candidates) {
      if (ExploredStates >= MaxAlternateStates) {
        HitSearchLimit = true;
        return false;
      }
      ++ExploredStates;
      NopSled &Sled = Gateways[SledIndex];
      uint64_t PreviousWritePos = Sled.WritePos;
      GatewaySlotStateChangeList Changes = advanceGatewaySled(
          Sled, Sled.WritePos + MinInstSize, OwnersBySled[SledIndex],
          CurrentCandidateSlots, Done, CurrentViableSleds, CurrentOwner,
          LastDestroyer);
      Islands.push_back(PreviousWritePos);
      UsedSleds.insert(SledIndex);
      if (Self(Self, PreviousWritePos))
        return true;
      UsedSleds.erase(SledIndex);
      Islands.pop_back();
      restoreGatewaySled(Sled, PreviousWritePos, Changes, CurrentCandidateSlots,
                         CurrentViableSleds, LastDestroyer);
    }
    return false;
  };

  if (!Search(Search, FromOffset)) {
    if (HitSearchLimit && LogSearchLimit)
      log() << "hotswap: branch-island search limit reached for site 0x"
            << utohexstr(FromOffset) << " after " << ExploredStates
            << " explored state(s)\n";
    return std::nullopt;
  }
  return Islands;
}

static std::optional<size_t> findNearestGatewaySled(
    ArrayRef<NopSled> Gateways, uint64_t Offset, uint64_t Needed,
    ArrayRef<GatewaySlotOwnerList> OwnersBySled,
    ArrayRef<uint64_t> CurrentViableSleds, ArrayRef<uint8_t> Done,
    size_t CurrentOwner, bool ProtectSlots, bool RequireExternalGateway) {
  std::optional<size_t> Best;
  uint64_t BestDistance = std::numeric_limits<uint64_t>::max();
  for (size_t I = 0; I != Gateways.size(); ++I) {
    const NopSled &Sled = Gateways[I];
    if ((RequireExternalGateway && !Sled.GlobalGateway && !Sled.GatewayOnly) ||
        !Sled.canGatewayFrom(Offset) || Sled.WritePos > Sled.End ||
        Needed > Sled.End - Sled.WritePos)
      continue;
    uint64_t Distance = Sled.WritePos > Offset ? Sled.WritePos - Offset
                                               : Offset - Sled.WritePos;
    if (Distance >= MaxSledDistance || Distance >= BestDistance)
      continue;
    if (ProtectSlots &&
        wouldConsumeLastGatewaySlot(Gateways, I, Needed, OwnersBySled,
                                    CurrentViableSleds, Done, CurrentOwner))
      continue;
    Best = I;
    BestDistance = Distance;
  }
  return Best;
}

static bool assignLongBranchGateways(PatchContext &Ctx) {
  // This is a snapshot of the one padding map built before any rewrite. It
  // preserves every allocation already made by direct sled emission and never
  // reinterprets the stale decoded stream after Ctx.Text has been modified.
  std::vector<NopSled> Gateways = Ctx.NopSleds;

  DenseMap<uint64_t, size_t> PoolIslandOwners;
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
                          std::numeric_limits<uint64_t>::max(),
                          /*GatewayOnly=*/true});
    }
    IslandLayoutOffset = *Next;
  }

  struct PendingGateway {
    size_t TrampolineIndex = 0;
    uint64_t TargetOffset = 0;
    uint64_t NeededBytes = 0;
    uint64_t InitialCandidateSlots = 0;
    bool UsesCallTail = false;
    SmallVector<uint8_t> CallTailBytes;
    SmallVector<size_t, 4> CandidateSleds;
  };
  std::vector<PendingGateway> Pending;
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

    if (isSBranchReachable(T.OriginalOffset, TP)) {
      T.UsesShortBranchForward = true;
      continue;
    }
    SmallVector<uint8_t> Direct = encodeSetPCLongBranch(
        Ctx.LS, T.OriginalOffset, TP, T.LongBranchSgprBase);
    if (!Direct.empty() && Direct.size() <= T.OriginalSize) {
      T.UsesDirectSetPCForward = true;
      T.DirectSetPCForwardBytes = std::move(Direct);
      continue;
    }
    PendingGateway P;
    P.TrampolineIndex = I;
    P.TargetOffset = TP;
    // The full sequence's MC width depends on the gateway displacement.
    // Reserve the maximum here; only the owner-stable call tail below may use
    // its exact 12/16-byte size.
    P.NeededBytes = SetPcForwardSequenceBytes;

    // S_CALL_B64 writes the address immediately after the call to SDST. The
    // get-PC instruction at the head of Direct produces that same address, so
    // a source call followed by the source-relative add/set-PC tail at a nearby
    // sled is the same SCC-neutral transfer. Unlike re-encoding the full
    // sequence at each sled cursor, the 12/16-byte tail is owner-stable across
    // literal-width boundaries because its addend remains
    // Target - (OriginalOffset + 4).
    std::optional<uint64_t> SourceContinuation = checkedAddUint64(
        T.OriginalOffset, MinInstSize, "forward call-gateway continuation");
    std::optional<uint64_t> SourceEnd = checkedAddUint64(
        T.OriginalOffset, T.OriginalSize, "forward call-gateway source end");
    if (!SourceContinuation || !SourceEnd)
      return false;
    if (*SourceContinuation < *SourceEnd && Ctx.DirectControlFlowTargets &&
        Ctx.DirectControlFlowTargets->contains(*SourceContinuation)) {
      log() << "hotswap: error: forward call-gateway source [0x"
            << utohexstr(T.OriginalOffset) << ", 0x" << utohexstr(*SourceEnd)
            << ") contains protected call continuation 0x"
            << utohexstr(*SourceContinuation) << "\n";
      return false;
    }
    std::optional<SmallVector<uint8_t>> ExtractedTail =
        Direct.empty() ? std::nullopt : extractSetPCCallTail(Direct, Ctx.LS);
    SmallVector<uint8_t> EncodedTail = encodeLinkRelativeSetPCTail(
        Ctx.LS, *SourceContinuation, TP, T.LongBranchSgprBase);
    if (ExtractedTail && *ExtractedTail == EncodedTail &&
        T.OriginalSize >= MinInstSize &&
        encodeSCallI64(Ctx.LS, T.OriginalOffset, *SourceContinuation,
                       T.LongBranchSgprBase)
                .size() == MinInstSize) {
      P.UsesCallTail = true;
      P.CallTailBytes = std::move(EncodedTail);
      P.NeededBytes = P.CallTailBytes.size();
    }

    for (size_t SledIndex = 0; SledIndex != Gateways.size(); ++SledIndex) {
      if (P.UsesCallTail && !Gateways[SledIndex].GlobalGateway &&
          !Gateways[SledIndex].GatewayOnly)
        continue;
      uint64_t Slots = countReachableGatewaySlots(
          Gateways[SledIndex], T.OriginalOffset, P.NeededBytes);
      P.InitialCandidateSlots += Slots;
      if (canEverReachGatewaySlot(Gateways[SledIndex], T.OriginalOffset,
                                  P.NeededBytes))
        P.CandidateSleds.push_back(SledIndex);
    }
    Pending.push_back(std::move(P));
  }

  std::stable_sort(Pending.begin(), Pending.end(),
                   [](const PendingGateway &LHS, const PendingGateway &RHS) {
                     if (LHS.NeededBytes != RHS.NeededBytes)
                       return LHS.NeededBytes > RHS.NeededBytes;
                     return LHS.InitialCandidateSlots <
                            RHS.InitialCandidateSlots;
                   });

  const std::vector<NopSled> BaseGateways = Gateways;
  std::vector<GatewaySlotOwnerList> OwnersBySled(Gateways.size());
  std::vector<uint64_t> InitialCandidateSlots;
  std::vector<uint64_t> InitialViableSleds;
  InitialCandidateSlots.reserve(Pending.size());
  InitialViableSleds.reserve(Pending.size());
  for (size_t I = 0; I != Pending.size(); ++I) {
    const PendingGateway &P = Pending[I];
    const Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    InitialCandidateSlots.push_back(P.InitialCandidateSlots);
    InitialViableSleds.push_back(P.CandidateSleds.size());
    for (size_t SledIndex : P.CandidateSleds)
      OwnersBySled[SledIndex].push_back({I, T.OriginalOffset, P.NeededBytes});
  }

  std::vector<SmallVector<size_t, 2>> Prerequisites(Pending.size());
  std::vector<SmallVector<size_t, 2>> Successors(Pending.size());
  uint64_t BranchIslandChains = 0;
  uint64_t AssignedGateways = 0;
  uint64_t ProtectedPasses = 0;
  uint64_t AlternateRelayPasses = 0;
  uint64_t UnrestrictedCycleBreaks = 0;
  uint64_t DependencyRetries = 0;
  static constexpr size_t NormalAlternateStates = 256;
  static constexpr size_t StalledAlternateStates = 16384;
  const size_t MaxDependencyRetries = std::min(Pending.size(), size_t{64});

  auto HasDependencyPath = [&](size_t From, size_t To) {
    std::vector<uint8_t> Seen(Pending.size(), 0);
    SmallVector<size_t, 16> Worklist;
    Worklist.push_back(From);
    while (!Worklist.empty()) {
      size_t Current = Worklist.pop_back_val();
      if (Current == To)
        return true;
      if (Seen[Current])
        continue;
      Seen[Current] = 1;
      llvm::append_range(Worklist, Successors[Current]);
    }
    return false;
  };

  for (;;) {
    Gateways = BaseGateways;
    std::vector<uint64_t> CurrentCandidateSlots = InitialCandidateSlots;
    std::vector<uint64_t> CurrentViableSleds = InitialViableSleds;
    std::vector<uint8_t> Done(Pending.size(), 0);
    std::vector<std::optional<size_t>> LastDestroyer(Pending.size());
    for (const PendingGateway &P : Pending) {
      Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
      T.ForwardBranchIslands.clear();
      T.ForwardBranchTargetOffset = 0;
      T.HasForwardGateway = false;
      T.ForwardGatewayOffset = 0;
      T.ForwardGatewayCallBytes.clear();
      T.ForwardGatewayBytes.clear();
    }

    BranchIslandChains = 0;
    AssignedGateways = 0;
    ProtectedPasses = 0;
    AlternateRelayPasses = 0;
    UnrestrictedCycleBreaks = 0;
    bool FatalAllocationError = false;
    auto IsEligible = [&](size_t I) {
      for (size_t Prerequisite : Prerequisites[I])
        if (!Done[Prerequisite])
          return false;
      return true;
    };

    auto TryRelay = [&](size_t I, bool ProtectSlots, size_t MaxAlternateStates,
                        bool LogSearchLimit) {
      const PendingGateway &P = Pending[I];
      Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
      std::optional<SmallVector<uint64_t, 4>> Islands =
          allocateForwardBranchIslands(
              Gateways, T.OriginalOffset, P.TargetOffset, OwnersBySled,
              CurrentCandidateSlots, CurrentViableSleds, Done, I, LastDestroyer,
              ProtectSlots, MaxAlternateStates, LogSearchLimit);
      if (!Islands || Islands->empty())
        return false;
      T.ForwardBranchIslands = std::move(*Islands);
      T.ForwardBranchTargetOffset = P.TargetOffset;
      Done[I] = 1;
      ++BranchIslandChains;
      return true;
    };

    auto TryGateway = [&](size_t I, bool ProtectSlots) {
      const PendingGateway &P = Pending[I];
      Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
      std::optional<size_t> SledIndex = findNearestGatewaySled(
          Gateways, T.OriginalOffset, P.NeededBytes, OwnersBySled,
          CurrentViableSleds, Done, I, ProtectSlots,
          /*RequireExternalGateway=*/P.UsesCallTail);
      if (!SledIndex)
        return false;
      NopSled &Sled = Gateways[*SledIndex];
      uint64_t GatewayOffset = Sled.WritePos;
      SmallVector<uint8_t> Call;
      SmallVector<uint8_t> Gateway;
      if (P.UsesCallTail) {
        Call = encodeSCallI64(Ctx.LS, T.OriginalOffset, GatewayOffset,
                             T.LongBranchSgprBase);
        Gateway = P.CallTailBytes;
      } else {
        Call = Ctx.LS.encodeSBranch(T.OriginalOffset, GatewayOffset);
        Gateway = encodeSetPCLongBranch(Ctx.LS, GatewayOffset, P.TargetOffset,
                                        T.LongBranchSgprBase);
      }
      if (Call.size() != MinInstSize || Gateway.empty() ||
          Gateway.size() > P.NeededBytes ||
          (P.UsesCallTail && Gateway.size() != P.NeededBytes)) {
        log() << "hotswap: error: failed to encode far-site gateway at 0x"
              << utohexstr(GatewayOffset) << "\n";
        FatalAllocationError = true;
        return false;
      }
      T.HasForwardGateway = true;
      T.ForwardGatewayOffset = GatewayOffset;
      if (P.UsesCallTail)
        T.ForwardGatewayCallBytes = std::move(Call);
      T.ForwardGatewayBytes = std::move(Gateway);
      advanceGatewaySled(Sled, Sled.WritePos + T.ForwardGatewayBytes.size(),
                         OwnersBySled[*SledIndex], CurrentCandidateSlots, Done,
                         CurrentViableSleds, I, LastDestroyer);
      Done[I] = 1;
      ++AssignedGateways;
      return true;
    };

    // Drain allocations that preserve a full gateway for every unfinished
    // owner. If that reaches a dependency stall, permit one allocation to
    // break it. A later failure learns which owner must precede that breaker
    // and restarts from the pristine sled cursors.
    size_t Remaining = Pending.size();
    size_t FailedOwner = Pending.size();
    while (Remaining != 0) {
      ++ProtectedPasses;
      bool Progress = false;
      for (size_t I = 0; I != Pending.size(); ++I) {
        if (Done[I] || !IsEligible(I))
          continue;
        if (TryRelay(I, /*ProtectSlots=*/true, NormalAlternateStates,
                     /*LogSearchLimit=*/false) ||
            TryGateway(I, /*ProtectSlots=*/true)) {
          --Remaining;
          Progress = true;
        }
        if (FatalAllocationError)
          return false;
      }
      if (Progress)
        continue;

      ++AlternateRelayPasses;
      for (size_t I = 0; I != Pending.size(); ++I) {
        if (Done[I] || !IsEligible(I))
          continue;
        if (TryRelay(I, /*ProtectSlots=*/true, StalledAlternateStates,
                     /*LogSearchLimit=*/true)) {
          --Remaining;
          Progress = true;
        }
      }
      if (Progress)
        continue;

      for (size_t I = 0; I != Pending.size(); ++I) {
        if (Done[I] || !IsEligible(I))
          continue;
        if (TryRelay(I, /*ProtectSlots=*/false,
                     /*MaxAlternateStates=*/0,
                     /*LogSearchLimit=*/false) ||
            TryGateway(I, /*ProtectSlots=*/false)) {
          --Remaining;
          Progress = true;
          ++UnrestrictedCycleBreaks;
          log() << "hotswap: gateway reservation solver: unrestricted cycle "
                   "break at site 0x"
                << utohexstr(Ctx.OutTrampolines[Pending[I].TrampolineIndex]
                                 .OriginalOffset)
                << "\n";
          break;
        }
        if (FatalAllocationError)
          return false;
      }
      if (Progress)
        continue;

      size_t FirstEligible = Pending.size();
      for (size_t I = 0; I != Pending.size(); ++I) {
        if (!Done[I] && IsEligible(I)) {
          if (FirstEligible == Pending.size())
            FirstEligible = I;
          if (LastDestroyer[I]) {
            FailedOwner = I;
            break;
          }
        }
      }
      if (FailedOwner == Pending.size())
        FailedOwner = FirstEligible;
      break;
    }

    if (Remaining == 0)
      break;
    if (FailedOwner == Pending.size()) {
      log() << "hotswap: error: gateway allocation dependency cycle\n";
      return false;
    }

    const PendingGateway &Failed = Pending[FailedOwner];
    const Trampoline &FailedT = Ctx.OutTrampolines[Failed.TrampolineIndex];
    auto LogFailedOwner = [&] {
      log() << "hotswap: error: no safe short-branch gateway for far site 0x"
            << utohexstr(FailedT.OriginalOffset) << " ("
            << Failed.InitialCandidateSlots << " initial candidate slot(s))\n";
    };
    if (!LastDestroyer[FailedOwner]) {
      LogFailedOwner();
      return false;
    }
    size_t Destroyer = *LastDestroyer[FailedOwner];
    bool Duplicate = llvm::is_contained(Successors[FailedOwner], Destroyer);
    if (Destroyer == FailedOwner || Duplicate ||
        HasDependencyPath(Destroyer, FailedOwner)) {
      log() << "hotswap: error: gateway allocation dependency cycle between "
               "sites 0x"
            << utohexstr(FailedT.OriginalOffset) << " and 0x"
            << utohexstr(Ctx.OutTrampolines[Pending[Destroyer].TrampolineIndex]
                             .OriginalOffset)
            << "\n";
      LogFailedOwner();
      return false;
    }
    if (DependencyRetries >= MaxDependencyRetries) {
      log() << "hotswap: error: gateway allocation dependency retry limit "
               "reached\n";
      LogFailedOwner();
      return false;
    }

    Prerequisites[Destroyer].push_back(FailedOwner);
    Successors[FailedOwner].push_back(Destroyer);
    ++DependencyRetries;
    log() << "hotswap: gateway reservation solver: retrying with site 0x"
          << utohexstr(FailedT.OriginalOffset) << " before site 0x"
          << utohexstr(Ctx.OutTrampolines[Pending[Destroyer].TrampolineIndex]
                           .OriginalOffset)
          << "\n";
  }

  log() << "hotswap: gateway reservation solver: " << ProtectedPasses
        << " protected pass(es), " << AlternateRelayPasses
        << " alternate relay pass(es), " << UnrestrictedCycleBreaks
        << " unrestricted cycle-break allocation(s), " << DependencyRetries
        << " conflict-directed retry/retries\n";
  if (AssignedGateways != 0)
    log() << "hotswap: assigned " << AssignedGateways
          << " SCC-neutral forward gateway(s)\n";
  if (BranchIslandChains != 0)
    log() << "hotswap: assigned " << BranchIslandChains
          << " forward s_branch island chain(s)\n";

  for (Trampoline &T : Ctx.OutTrampolines) {
    if (T.HasForwardGateway)
      std::memcpy(Ctx.Text + T.ForwardGatewayOffset,
                  T.ForwardGatewayBytes.data(), T.ForwardGatewayBytes.size());
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
      if (Owner != PoolIslandOwners.end()) {
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
  }
  return true;
}

/// Emit \p Replacement for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize). Prefers an in-place NOP-sled rewrite when a
/// reachable sled with sufficient headroom exists; otherwise falls back to a
/// deferred trampoline.
static bool emitReplacementCodeRaw(PatchContext &Ctx, uint64_t InstOffset,
                                   uint32_t InstSize,
                                   ArrayRef<uint8_t> Replacement,
                                   ReplacementPlacement Placement,
                                   bool DiagnoseFailure) {
  const bool AllowGlobalBody = Placement != ReplacementPlacement::Default;
  if (Ctx.RelocationProtectedOffsets.contains(InstOffset)) {
    if (DiagnoseFailure)
      log() << "hotswap: error: replacement source at 0x"
            << utohexstr(InstOffset) << " is relocation-protected\n";
    return false;
  }

  std::optional<uint64_t> ReturnTo = checkedAddUint64(
      InstOffset, InstSize, "replacement trampoline return target");
  std::optional<uint64_t> PoolReturnFrom =
      checkedAddUint64(Ctx.PoolBaseOffset, Replacement.size(),
                       "replacement trampoline return slot");
  if (!ReturnTo || !PoolReturnFrom)
    return false;

  // The address at the start remains a valid alias for the replacement or
  // redirect, and the end is outside the rewritten half-open interval. Any
  // entry strictly inside would instead land in a moved instruction, branch
  // tail, or padding, so every patch family must reject the complete window.
  if (Ctx.DirectControlFlowTargets && *ReturnTo > InstOffset) {
    uint64_t Offset = InstOffset;
    while (++Offset < *ReturnTo)
      if (Ctx.DirectControlFlowTargets->contains(Offset)) {
        if (DiagnoseFailure)
          log() << "hotswap: error: replacement source [0x"
                << utohexstr(InstOffset) << ", 0x" << utohexstr(*ReturnTo)
                << ") contains protected interior entry 0x" << utohexstr(Offset)
                << "\n";
        return false;
      }
  }

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
    uint64_t Needed = getNopSledBytesNeeded(Ctx, InstOffset, InstSize,
                                            Replacement, Placement);
    NopSledUse SledUse =
        AllowGlobalBody ? NopSledUse::RelocationBody : NopSledUse::OwnerBody;
    if (NopSled *Sled =
            findNearestSled(Ctx.NopSleds, InstOffset, Needed, SledUse)) {
      if (emitToNopSled(Ctx, *Sled, InstOffset, InstSize, Replacement,
                        Placement))
        return true;
      log() << "hotswap: emitReplacementCode: NOP sled at offset 0x"
            << utohexstr(Sled->WritePos)
            << " is not branch-reachable after assembly; using trampoline.\n";
    }
  }
  return emitToTrampolineRaw(Ctx, InstOffset, InstSize, Replacement, Placement,
                             DiagnoseFailure);
}

[[nodiscard]] bool emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset,
                                       uint32_t InstSize,
                                       ArrayRef<uint8_t> Replacement,
                                       ReplacementPlacement Placement,
                                       bool DiagnoseFailure) {
  std::optional<PreparedSiteReplacement> Prepared = prepareSiteReplacement(
      Ctx, InstOffset, InstSize, Replacement, DiagnoseFailure);
  if (!Prepared)
    return false;
  if (!emitReplacementCodeRaw(Ctx, InstOffset, InstSize, Prepared->Bytes,
                              Placement, DiagnoseFailure))
    return false;
  return commitSiteReplacement(Ctx, InstOffset, InstSize, *Prepared);
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
  std::optional<DenseSet<uint64_t>> CodeEntries =
      collectCodeObjectEntryOffsets(Elf, TextSize);
  std::optional<std::vector<uint64_t>> TextSymbolOffsets =
      Elf.textSymbolOffsets();
  std::optional<std::vector<ElfView::TextOffsetRange>> TextSymbolExtents =
      Elf.textSymbolExtents();
  std::optional<DeclaredTextEntryInfo> DeclaredEntries =
      collectDeclaredTextEntries(Elf);
  if (!DeclaredEntries)
    return std::nullopt;
  std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Elf.functionTextRanges();
  std::optional<CompatibilityControlFlowFacts> CompatibilityFacts =
      collectCompatibilityControlFlowFacts(
          Decoded, LS, Elf.textAddr(), Elf.textSize(), DeclaredEntries->Entries,
          FunctionRanges, DeclaredEntries->ExternalEntries);
  if (!CompatibilityFacts)
    return std::nullopt;

  std::optional<DenseSet<uint64_t>> DirectControlFlowTargets;
  if (CodeEntries && TextSymbolOffsets && TextSymbolExtents)
    DirectControlFlowTargets = collectDirectBranchTargets(
        Decoded, LS, ArrayRef<uint8_t>(Text, TextSize), Elf, *CodeEntries,
        *TextSymbolOffsets, *TextSymbolExtents, *CompatibilityFacts);
  DenseSet<uint64_t> EmptyTargets;
  const DenseSet<uint64_t> &KnownDirectTargets =
      DirectControlFlowTargets ? *DirectControlFlowTargets : EmptyTargets;
  const std::optional<ArrayRef<uint64_t>> OptionalTextSymbolOffsets =
      TextSymbolOffsets ? std::optional<ArrayRef<uint64_t>>(*TextSymbolOffsets)
                        : std::nullopt;
  const std::optional<ArrayRef<ElfView::TextOffsetRange>>
      OptionalTextSymbolExtents =
          TextSymbolExtents ? std::optional<ArrayRef<ElfView::TextOffsetRange>>(
                                  *TextSymbolExtents)
                            : std::nullopt;
  const ArrayRef<uint8_t> TextBytes(Text, TextSize);

  // The later analyzer is deliberately stricter than the PR's public callable
  // contract and does not recognize PR call-tail/materialized forms. Reconcile
  // each unresolved call independently: only an exact PR materialization proof
  // or the ABI-standard link-call contract discharges that site. Every other
  // site retains the later global fail-closed policy.
  bool HasRemainingCompatibilityUnresolvedCall = false;
  for (size_t I : CompatibilityFacts->UnresolvedCallIndices) {
    if (I >= Decoded.size())
      return std::nullopt;
    const bool ResolvedByPr =
        isStandardLinkCall(Decoded[I], LS) ||
        (DirectControlFlowTargets &&
         evaluateMaterializedPcTransfer(
             Decoded, I, KnownDirectTargets, TextBytes, LS, Elf,
             OptionalTextSymbolOffsets, OptionalTextSymbolExtents)
             .has_value());
    if (ResolvedByPr)
      continue;
    log() << "hotswap: unresolved call target at 0x"
          << utohexstr(Decoded[I].Offset) << " (" << Decoded[I].Mnemonic
          << ")\n";
    HasRemainingCompatibilityUnresolvedCall = true;
  }
  CompatibilityFacts->Summary.HasUnresolvedTargets =
      HasRemainingCompatibilityUnresolvedCall;

  const bool HasUnresolvedStandardLinkCall =
      DirectControlFlowTargets
          ? hasUnresolvedStandardLinkCall(Decoded, LS, Elf, KnownDirectTargets,
                                          TextBytes, OptionalTextSymbolOffsets,
                                          OptionalTextSymbolExtents,
                                          *CompatibilityFacts)
          : llvm::any_of(Decoded, [&](const InternalDecodedInst &DI) {
              return isStandardLinkCall(DI, LS);
            });
  DenseSet<uint64_t> ProvenDirectCallReturns;
  if (DirectControlFlowTargets && TextSymbolOffsets && TextSymbolExtents)
    ProvenDirectCallReturns = collectProvenDirectCallReturns(
        Decoded, LS, Elf, KnownDirectTargets, TextBytes,
        OptionalTextSymbolOffsets, OptionalTextSymbolExtents,
        HasUnresolvedStandardLinkCall, *CompatibilityFacts);
  bool PrHasUnknownArbitraryIndirectTarget = false;
  DenseSet<uint64_t> IndirectControlFlowFunctions =
      collectIndirectControlFlowFunctions(
          Decoded, LS, Elf, KnownDirectTargets, TextBytes,
          OptionalTextSymbolOffsets, OptionalTextSymbolExtents,
          ProvenDirectCallReturns, *CompatibilityFacts,
          PrHasUnknownArbitraryIndirectTarget);
  const bool HasUnknownArbitraryIndirectTarget =
      PrHasUnknownArbitraryIndirectTarget ||
      CompatibilityFacts->Summary.HasUnresolvedTargets;
  const bool HasIncompleteControlFlow =
      !DirectControlFlowTargets || HasUnknownArbitraryIndirectTarget;
  if (HasIncompleteControlFlow)
    for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges())
      if (Range.Begin >= Elf.textAddr())
        IndirectControlFlowFunctions.insert(Range.Begin - Elf.textAddr());
  if (HasUnknownArbitraryIndirectTarget)
    log() << "hotswap: unresolved control-flow target disables NOP-sled "
             "emission, trampoline coalescing, source relocation, and .text "
             "gateways\n";
  HotswapProfile::Scope SledScope =
      Profile.time(HotswapMetric::NopSledScan);
  std::vector<NopSled> Sleds;
  if (DirectControlFlowTargets && TextSymbolOffsets && TextSymbolExtents &&
      !HasUnknownArbitraryIndirectTarget) {
    Sleds = buildNopSledMap(Decoded, LS, Elf, KnownDirectTargets,
                            IndirectControlFlowFunctions, *TextSymbolOffsets,
                            *TextSymbolExtents);
    std::vector<NopSled> ExternalSleds = buildExternalGatewaySleds(
        Decoded, LS, Elf, ArrayRef<uint8_t>(Text, TextSize), KnownDirectTargets,
        *TextSymbolOffsets, *TextSymbolExtents, Sleds);
    Sleds.insert(Sleds.end(), ExternalSleds.begin(), ExternalSleds.end());
  } else {
    log() << "hotswap: incomplete control-flow targets disable NOP padding "
             "donation\n";
  }
  SledScope.finish();

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
    BitVector AllVgprs(Config.MaxVgprs);
    AllVgprs.set(0, Config.MaxVgprs);
    for (size_t I = 0, LE = Liveness.LiveBefore.size(); I < LE; ++I) {
      Liveness.LiveBefore[I] = AllVgprs;
      Liveness.LiveAfter[I] = AllVgprs;
    }
  }

  TensorDescriptorMustAnalysis TensorDescriptorAnalysis;
  InitialVmemMustAnalysis InitialVmemAnalysis;
  const bool HasClause =
      Config.RunB0A0Patches &&
      llvm::any_of(Decoded, [](const InternalDecodedInst &DI) {
        return DI.Mnemonic == "s_clause";
      });
  const bool HasTensorLoad =
      llvm::any_of(Decoded, [](const InternalDecodedInst &DI) {
        return DI.Mnemonic == "tensor_load_to_lds";
      });
  std::vector<InternalDecodedInst> HotswapAnalysisDecoded;
  if ((HasClause || HasTensorLoad) &&
      !buildHotswapAnalysisDecoded(Elf, LS, Decoded, HotswapAnalysisDecoded))
    return std::nullopt;
  OriginalIngressInfo OriginalIngress;
  if (HasClause || HasTensorLoad)
    OriginalIngress = collectOriginalIngress(
        Decoded, KnownDirectTargets, ArrayRef<uint8_t>(Text, TextSize), LS, Elf,
        *CompatibilityFacts,
        TextSymbolOffsets
            ? std::optional<ArrayRef<uint64_t>>(*TextSymbolOffsets)
            : std::nullopt,
        TextSymbolExtents ? std::optional<ArrayRef<ElfView::TextOffsetRange>>(
                                *TextSymbolExtents)
                          : std::nullopt);
  if (HasClause) {
    std::vector<KernelTextRange> KernelRanges = collectKernelTextRanges(
        Elf, LS, OriginalIngress.ControlFlowEdges,
        HasUnknownArbitraryIndirectTarget, HasUnresolvedStandardLinkCall);
    InitialVmemAnalysis = computeInitialVmemMustAnalysis(
        Decoded, HotswapAnalysisDecoded, KernelRanges, LS);
  }
  if (HasTensorLoad) {
    const bool TensorHasUnknownOwnedInstruction =
        llvm::any_of(Decoded, [&](const InternalDecodedInst &DI) {
          return DI.Mnemonic == "<unknown>" &&
                 Elf.findFunctionTextRangeAtOffset(DI.Offset).has_value();
        });
    const bool TensorHasTrapOrRfe =
        llvm::any_of(Decoded, [&](const InternalDecodedInst &DI) {
          return StringRef(DI.Mnemonic).starts_with("s_rfe") ||
                 StringRef(DI.Mnemonic).starts_with("s_trap") ||
                 (LS.MCII && LS.MCII->get(DI.Inst.getOpcode()).isTrap());
        });
    std::vector<TensorAnalysisRange> TensorRanges;
    if (CodeEntries && DirectControlFlowTargets)
      TensorRanges = collectTensorAnalysisRanges(
          Elf, LS, *CodeEntries, IndirectControlFlowFunctions,
          OriginalIngress.CrossRangeEntryFunctions,
          OriginalIngress.ExternalEntries, OriginalIngress.ControlFlowEdges,
          HasUnknownArbitraryIndirectTarget ||
              TensorHasUnknownOwnedInstruction ||
              HasUnresolvedStandardLinkCall || TensorHasTrapOrRfe);
    TensorDescriptorAnalysis = computeTensorDescriptorMustAnalysis(
        Decoded, HotswapAnalysisDecoded, TensorRanges, LS, KnownDirectTargets,
        Config.MaxSgprs, Config.MaxVgprs);
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
  DirectControlFlowInfo CompatibilityControlFlow = CompatibilityFacts->Summary;
  CompatibilityControlFlow.Targets = KnownDirectTargets;
  CompatibilityControlFlow.HasUnresolvedTargets |=
      !DirectControlFlowTargets || HasUnknownArbitraryIndirectTarget ||
      HasUnresolvedStandardLinkCall;
  PatchContext Ctx{Config,         Decoded,         Text,
                   TextSize,       *PoolBaseOffset, LS,
                   OutTrampolines, Sleds,           Elf,
                   Liveness,       KernelStats,     OutScratchPatches,
                   CompatibilityControlFlow, Profile};
  Ctx.InitialVmemAnalysis = &InitialVmemAnalysis;
  Ctx.TensorDescriptorAnalysis = &TensorDescriptorAnalysis;
  for (const InternalDecodedInst &DI : Decoded)
    Ctx.MaxDecodedInstSize = std::max(Ctx.MaxDecodedInstSize, DI.Size);
  Ctx.DirectControlFlowTargets = DirectControlFlowTargets;
  if (TextSymbolExtents)
    Ctx.TextSymbolExtents = *TextSymbolExtents;
  Ctx.IndirectControlFlowFunctions = std::move(IndirectControlFlowFunctions);
  Ctx.HasUnknownArbitraryIndirectTarget = HasUnknownArbitraryIndirectTarget;
  if (Ctx.DirectControlFlowTargets && !Ctx.HasUnknownArbitraryIndirectTarget)
    Ctx.VgprMsbDstSrc0EqualBefore = computeVgprMsbDstSrc0Equality(
        Decoded, LS, Elf, *DirectControlFlowTargets, *CodeEntries,
        Ctx.IndirectControlFlowFunctions, ArrayRef<uint8_t>(Text, TextSize),
        *CompatibilityFacts,
        Ctx.VgprDef0AlignedTo8, Ctx.VgprDef1AlignedTo8, Ctx.VgprMsbDstBefore,
        Ctx.VgprMsbSrc0Before, Ctx.VgprMsbModeBefore,
        Ctx.CrossFunctionInteriorEntryFunctions);
  collectRelocationProtectedOffsets(Decoded, Ctx);

  // Patch placement needs a conservative entry-boundary set, not just the
  // edges used to build the CFG. Keep ordinary symbols out of control-flow
  // analysis, then include them here so moving a source window can never
  // swallow an addressable interior boundary.
  if (Ctx.DirectControlFlowTargets && TextSymbolOffsets)
    Ctx.DirectControlFlowTargets->insert(TextSymbolOffsets->begin(),
                                         TextSymbolOffsets->end());

  const HotswapPatchVTable &VT = getHotswapPatchVTable();
  // Whole-function requirements must be discovered from the immutable stream
  // before any same-size or relocating pass mutates the source bytes. Their
  // reserved sites then participate in every atomic-window ownership check.
  if (Config.RunB0A0Patches && VT.precomputeWmmaHazards &&
      !VT.precomputeWmmaHazards(Ctx))
    return std::nullopt;

  precomputeDs2AddressAlignment(Ctx);
  precomputeSiteDeadSgprFacts(Ctx);

  // VOP3PX2 is a same-size bit-field correction. Apply it before any atomic
  // source-window relocation so those windows copy already-correct current
  // bytes rather than forcing a later overlapping rewrite.
  if (Config.RunB0A0Patches && VT.applyVop3px2Src2Fix) {
    HotswapProfile::Scope Vop3Scope =
        Ctx.Profile.time(HotswapMetric::Vop3px2Src2);
    const uint32_t P = VT.applyVop3px2Src2Fix(Ctx);
    Vop3Scope.addPatches(P);
    Vop3Scope.finish();
    Patched += P;
  }
  if (Ctx.RequiredPatchFailed)
    return std::nullopt;

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
    if (DI.Mnemonic == UnknownMnemonic ||
        Ctx.ClaimedReplacementOffsets.contains(DI.Offset))
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

  // Finalize requirements that no ordinary per-instruction owner composed.
  // The hazard pass copies current post-in-place bytes for those sites; it
  // never rebuilds a replacement from stale MCInst state.
  if (Config.RunB0A0Patches && VT.applyWmmaHazardPatch) {
    HotswapProfile::Scope HazardScope =
        Ctx.Profile.time(HotswapMetric::WmmaHazard);
    const uint32_t P = VT.applyWmmaHazardPatch(Ctx);
    HazardScope.addPatches(P);
    HazardScope.finish();
    Patched += P;
  }
  if (Ctx.RequiredPatchFailed)
    return std::nullopt;

  if (!OutTrampolines.empty()) {
    if (!Ctx.DirectControlFlowTargets)
      return std::nullopt;
    // Whole-function passes append after the per-instruction walk and can
    // therefore add an earlier site at the end of this vector. Neighbor-based
    // coalescing and source growth require address order, independent of which
    // patch family discovered each site.
    llvm::stable_sort(OutTrampolines,
                      [](const Trampoline &LHS, const Trampoline &RHS) {
                        return LHS.OriginalOffset < RHS.OriginalOffset;
                      });
    if (!Ctx.HasUnknownArbitraryIndirectTarget) {
      mergeAdjacentLongTrampolines(
          OutTrampolines, *Ctx.DirectControlFlowTargets,
          Ctx.IndirectControlFlowFunctions);
      expandStraightLineTrampolines(Ctx, *Ctx.DirectControlFlowTargets);
      mergeAdjacentLongTrampolines(
          OutTrampolines, *Ctx.DirectControlFlowTargets,
          Ctx.IndirectControlFlowFunctions);
    }
    appendPoolBranchIslands(OutTrampolines);
    if (!assignLongBranchGateways(Ctx))
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

    if (T.Long && !T.UsesSetPCBack) {
      log() << "hotswap: error: far trampoline lacks safe set-PC return at 0x"
            << utohexstr(T.OriginalOffset) << "\n";
      return false;
    }
    const uint32_t BackReserve =
        T.UsesSetPCBack ? SetPcReturnReserveBytes : MinInstSize;
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

    SmallVector<uint8_t> BrBack =
        T.UsesSetPCBack ? encodeSetPCLongBranch(LS, BackSlot, *ReturnTo,
                                                T.LongBranchSgprBase)
                        : LS.encodeSBranch(BackSlot, *ReturnTo);
    if (BrBack.empty() || BrBack.size() > BackReserve) {
      log() << "hotswap: error: trampoline branch-back encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << (T.Long ? " (long)\n" : "\n");
      return false;
    }
    std::memcpy(T.Bytes.data() + BackOffset, BrBack.data(), BrBack.size());
    for (uint32_t I = BrBack.size(); I + MinInstSize <= BackReserve;
         I += MinInstSize)
      std::memcpy(T.Bytes.data() + BackOffset + I, LS.SNopBytes.data(),
                  MinInstSize);

    SmallVector<uint8_t> BrFwd;
    if (T.Long) {
      if (T.UsesShortBranchForward) {
        BrFwd = LS.encodeSBranch(T.OriginalOffset, TP);
      } else if (!T.ForwardBranchIslands.empty()) {
        BrFwd =
            LS.encodeSBranch(T.OriginalOffset, T.ForwardBranchIslands.front());
      } else if (T.UsesDirectSetPCForward) {
        BrFwd = T.DirectSetPCForwardBytes;
      } else if (T.HasForwardGateway) {
        BrFwd = T.ForwardGatewayCallBytes.empty()
                    ? LS.encodeSBranch(T.OriginalOffset,
                                       T.ForwardGatewayOffset)
                    : T.ForwardGatewayCallBytes;
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
    // Pad the tail of the replaced slot with cached s_nop bytes.
    for (uint32_t I = BrFwd.size(); I + MinInstSize <= T.OriginalSize;
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
    log() << "hotswap: error: retargetCodeObject: " << "getNewUninitMemBuffer("
          << Size << ") failed (out of memory) for the " << CopyKind
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
  if (ViewOrErr->textSize() == 0) {
    log() << "hotswap: error: retargetCodeObject: input ELF has empty "
          << ".text section; nothing to rewrite.\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfView &Elf = *ViewOrErr;
  if (Prof)
    Profile.add(HotswapMetric::ElfParse, profNowNs() - ParseT0, 0);

  // Validate every executable region covered by the requested B0-to-A0
  // operation before metadata can short-circuit instruction rewriting. The
  // metadata certificate must not allow an incompatible or forged external
  // pool to bypass structural provenance checks.
  if (Options.RunB0A0Patches) {
    std::optional<bool> CompatibleExternalCode =
        Elf.executableCodeOutsideTextIsCompatibleWith(
            ExecutablePoolTargetState::A0);
    if (!CompatibleExternalCode)
      return AMD_COMGR_STATUS_ERROR;
    if (!*CompatibleExternalCode) {
      log() << "hotswap: error: B0-to-A0 rewrite found unprovenanced or "
               "target-incompatible executable code outside .text; refusing "
               "to issue an incomplete A0 target-state certificate\n";
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  bool RunB0A0Patches = Options.RunB0A0Patches;
  MaskWorkaroundPolicy MaskPolicy = Options.MaskPolicy;
  if (RunB0A0Patches) {
    std::optional<bool> AlreadyA0 = Elf.allKernelsHaveGfx1250Revision("A0");
    if (!AlreadyA0)
      return AMD_COMGR_STATUS_ERROR;
    if (*AlreadyA0) {
      // The metadata retag is committed only on a successfully returned
      // rewrite. It is therefore an object-wide target-state certificate, not
      // a heuristic derived from instruction patterns in transformed code.
      log() << "hotswap: every kernel already reports gfx1250 revision A0; "
               "skipping B0-to-A0 instruction rewrites\n";
      RunB0A0Patches = false;
      if (MaskPolicy == MaskWorkaroundPolicy::A0)
        MaskPolicy = MaskWorkaroundPolicy::None;
    }
  }
  const bool RunInstructionPatches =
      RunB0A0Patches || MaskPolicy != MaskWorkaroundPolicy::None;
  if (!RunInstructionPatches && !Options.RunEntryTrampolines) {
    std::unique_ptr<WritableMemoryBuffer> Result =
        copyOutputBuffer(ElfData, ElfSize, "already-targeted");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    Out = std::move(Result);
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
  Config.RunB0A0Patches = RunB0A0Patches;
  Config.MaskPolicy = MaskPolicy;

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
  if (RunB0A0Patches && !Elf.updateGfx1250RevisionMetadata("A0"))
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
    ExecutablePoolTargetState PoolTargetState =
        ExecutablePoolTargetState::Neutral;
    if (!Deferred.empty()) {
      if (RunB0A0Patches || MaskPolicy == MaskWorkaroundPolicy::A0)
        PoolTargetState = ExecutablePoolTargetState::A0;
      else if (MaskPolicy == MaskWorkaroundPolicy::B0)
        PoolTargetState = ExecutablePoolTargetState::B0;
      if (PoolTargetState == ExecutablePoolTargetState::Neutral) {
        log() << "hotswap: error: stepping-dependent deferred trampolines "
                 "require an explicit A0 or B0 executable-pool target state.\n";
        return AMD_COMGR_STATUS_ERROR;
      }
    } else if (EntryFixups.empty()) {
      log() << "hotswap: error: a stepping-neutral executable pool must be "
               "proven to contain kernel-entry stubs.\n";
      return AMD_COMGR_STATUS_ERROR;
    }
    uint64_t GrowT0 = Prof ? profNowNs() : 0;
    Result =
        Elf.growWithTrampolines(Growth, SNopBytes, PoolTargetState);
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
