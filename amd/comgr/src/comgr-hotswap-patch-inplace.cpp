//===- comgr-hotswap-patch-inplace.cpp - In-place B0-to-A0 patches --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applyInPlacePatches.  Handles instruction
/// rewrites that fit in the same code size as the original:
///
///   - s_clause                 -> s_nop 0
///   - cluster_load             -> global_load    (opcode swap via MCInst +
///                                                 MCCodeEmitter)
///   - unsafe s_clause          -> s_nop          (byte-level overwrite)
///   - s_barrier_signal_isfirst -> s_barrier_signal
///                                                (opcode swap; same operand
///                                                 layout, drops SCC write)
///
/// No trampolines, ELF growth, or extra VGPRs are required.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

uint32_t failRequiredInPlacePatch(PatchContext &Ctx, StringRef Mnemonic,
                                  uint64_t Offset) {
  log() << "hotswap: error: required in-place patch for " << Mnemonic
        << " failed at 0x" << utohexstr(Offset) << "\n";
  Ctx.RequiredPatchFailed = true;
  return 0;
}

/// Map a B0-only cluster_load mnemonic to the assembly string of its
/// A0-compatible global_load equivalent (with a dummy operand to resolve
/// the opcode). Returns an empty StringRef if \p Mnemonic is not a
/// cluster_load variant.
StringRef getClusterLoadReplacementAsm(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("cluster_load_b32", "global_load_b32 v0, v[0:1], off")
      .Case("cluster_load_b64", "global_load_b64 v[0:1], v[2:3], off")
      .Case("cluster_load_b128", "global_load_b128 v[0:3], v[4:5], off")
      .Case("cluster_load_async_to_lds_b8",
            "global_load_async_to_lds_b8 v0, v[0:1], off")
      .Case("cluster_load_async_to_lds_b32",
            "global_load_async_to_lds_b32 v0, v[0:1], off")
      .Case("cluster_load_async_to_lds_b64",
            "global_load_async_to_lds_b64 v0, v[0:1], off")
      .Case("cluster_load_async_to_lds_b128",
            "global_load_async_to_lds_b128 v0, v[0:1], off")
      .Default("");
}

/// Detect the SGPR-relative (_SADDR) form of a cluster_load.
///
/// The off-form cluster_load carries its global address in a 64-bit VGPR pair
/// and encodes the saddr field as the "off" sentinel, so all of its register
/// operands are VGPRs. The _SADDR variant shares the same display mnemonic but
/// is a distinct MC opcode with an extra scalar saddr operand. The off-form
/// replacement templates in getClusterLoadReplacementAsm would mis-encode that
/// scalar operand, so the two forms must be told apart. getNamedOperandIdx() /
/// OpName are backend-private headers, so -- mirroring the operand-kind
/// inspection in comgr-hotswap-patch-wmma-split.cpp -- classify by operand
/// kind: the presence of any SGPR register operand marks the _SADDR form.
bool usesSgprBaseAddress(const MCInst &Inst, const MCRegisterInfo &MRI) {
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() &&
        StringRef(MRI.getName(Op.getReg())).starts_with("SGPR"))
      return true;
  }
  return false;
}

/// Resolve the MC opcode index for an assembly mnemonic by parsing a dummy
/// instruction through the asm parser.
std::optional<unsigned> resolveOpcode(StringRef AsmSnippet,
                                      const LLVMState &LS) {
  SmallVector<uint8_t> Bytes = assembleSingleInst(AsmSnippet, LS);
  if (Bytes.empty())
    return std::nullopt;
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes.data(), Bytes.size(), LS, Decoded) ||
      Decoded.empty())
    return std::nullopt;
  return Decoded[0].Inst.getOpcode();
}

/// Encode an MCInst to raw bytes via MCCodeEmitter.
SmallVector<uint8_t> encodeMCInst(const MCInst &Inst, const LLVMState &LS) {
  SmallVector<char, 16> Code;
  SmallVector<MCFixup, 4> Fixups;
  LS.MCE->encodeInstruction(Inst, Code, Fixups, *LS.STI);
  return SmallVector<uint8_t>(Code.begin(), Code.end());
}

/// Perform an opcode swap: clone the decoded MCInst, set the replacement
/// opcode, re-encode via MCCodeEmitter, and overwrite in place.
/// Returns true on success.
bool swapOpcode(InternalDecodedInst &DI, uint8_t *Text, const LLVMState &LS,
                unsigned NewOpcode) {
  MCInst NewInst = DI.Inst;
  NewInst.setOpcode(NewOpcode);
  SmallVector<uint8_t> Bytes = encodeMCInst(NewInst, LS);
  if (Bytes.empty() || Bytes.size() != DI.Size)
    return false;
  std::memcpy(Text + DI.Offset, Bytes.data(), DI.Size);
  return true;
}

enum class ClauseScope { CU, SE, Device, System };
enum class ClauseFamily { VMEM, Flat, SMEM };

std::optional<ClauseFamily> getClauseFamily(StringRef Mnemonic) {
  if (Mnemonic.starts_with("buffer_") || Mnemonic.starts_with("tbuffer_") ||
      Mnemonic.starts_with("global_") || Mnemonic.starts_with("scratch_") ||
      Mnemonic.starts_with("image_") || Mnemonic.starts_with("cluster_") ||
      Mnemonic.starts_with("tensor_"))
    return ClauseFamily::VMEM;
  if (Mnemonic.starts_with("flat_"))
    return ClauseFamily::Flat;
  if (Mnemonic.starts_with("s_load") || Mnemonic.starts_with("s_buffer_load"))
    return ClauseFamily::SMEM;
  return std::nullopt;
}

std::optional<ClauseFamily> getClauseFamily(const InternalDecodedInst &DI,
                                            const LLVMState &LS) {
  // Some GLOBAL_WB encodings do not have a stable printed mnemonic, but they
  // are unclaused VMEM operations and therefore establish the entry workaround.
  if (DI.Inst.getOpcode() == LS.GlobalWbOpcode)
    return ClauseFamily::VMEM;
  return getClauseFamily(DI.Mnemonic);
}

std::optional<ClauseScope> getClauseScope(const MCInst &Inst,
                                          const LLVMState &LS) {
  if (!LS.MCIP)
    return std::nullopt;

  SmallString<256> PrintedBuf;
  raw_svector_ostream OS(PrintedBuf);
  LS.MCIP->printInst(&Inst, /*Address=*/0, /*Annot=*/"", *LS.STI, OS);
  StringRef Printed(PrintedBuf);
  constexpr StringLiteral ScopePrefix("scope:");
  size_t ScopePos = Printed.find(ScopePrefix);
  if (ScopePos == StringRef::npos)
    return ClauseScope::CU;

  StringRef Scope = Printed.drop_front(ScopePos + ScopePrefix.size());
  Scope = Scope.take_while([](char C) { return llvm::isAlnum(C) || C == '_'; });
  return StringSwitch<std::optional<ClauseScope>>(Scope)
      .Case("SCOPE_CU", ClauseScope::CU)
      .Case("SCOPE_SE", ClauseScope::SE)
      .Case("SCOPE_DEV", ClauseScope::Device)
      .Case("SCOPE_SYS", ClauseScope::System)
      .Default(std::nullopt);
}

/// Retain a hard clause only when its complete encoded membership is known,
/// all non-NOP members are memory operations from one instruction family and
/// cache scope, and no member needs a relocating rewrite.
bool clauseHasUniformScope(const PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &Clause = Ctx.Decoded[Idx];
  if (Clause.Inst.getNumOperands() == 0 || !Clause.Inst.getOperand(0).isImm())
    return false;

  const uint64_t MemberCount =
      (static_cast<uint64_t>(Clause.Inst.getOperand(0).getImm()) & 63) + 1;
  if (MemberCount > 63 || MemberCount > Ctx.Decoded.size() - Idx - 1)
    return false;

  std::optional<ClauseScope> Scope;
  std::optional<ClauseFamily> Family;
  bool SawMemory = false;
  for (size_t I = Idx + 1; I <= Idx + MemberCount; ++I) {
    const InternalDecodedInst &Member = Ctx.Decoded[I];
    if (Member.Inst.getOpcode() == Ctx.LS.SNopOpcode)
      continue;

    if (requiresIndependentInstructionRewrite(Ctx, I)) {
      const StringRef Mnemonic(Member.Mnemonic);
      const bool WillBeRewrittenInPlace =
          !getClusterLoadReplacementAsm(Mnemonic).empty() &&
          !usesSgprBaseAddress(Member.Inst, *Ctx.LS.MRI);
      if (!WillBeRewrittenInPlace)
        return false;
    }

    const MCInstrDesc &Desc = Ctx.LS.MCII->get(Member.Inst.getOpcode());
    if (!Desc.mayLoad() && !Desc.mayStore())
      return false;

    std::optional<ClauseFamily> MemberFamily = getClauseFamily(Member, Ctx.LS);
    if (!MemberFamily || (Family && *Family != *MemberFamily))
      return false;
    Family = MemberFamily;

    std::optional<ClauseScope> MemberScope =
        getClauseScope(Member.Inst, Ctx.LS);
    if (!MemberScope || (Scope && *Scope != *MemberScope))
      return false;
    Scope = MemberScope;
    SawMemory = true;
  }
  return SawMemory;
}

bool clauseContainsVmem(ArrayRef<InternalDecodedInst> Decoded, size_t Idx,
                        const LLVMState &LS) {
  const InternalDecodedInst &Clause = Decoded[Idx];
  if (Clause.Inst.getNumOperands() == 0 || !Clause.Inst.getOperand(0).isImm())
    return false;

  const uint64_t MemberCount =
      (static_cast<uint64_t>(Clause.Inst.getOperand(0).getImm()) & 63) + 1;
  if (MemberCount > 63 || MemberCount > Decoded.size() - Idx - 1)
    return false;

  for (size_t I = Idx + 1; I <= Idx + MemberCount; ++I) {
    std::optional<ClauseFamily> Family = getClauseFamily(Decoded[I], LS);
    if (Family == ClauseFamily::VMEM || Family == ClauseFamily::Flat)
      return true;
  }
  return false;
}

bool isVmemOrFlat(const InternalDecodedInst &DI, const LLVMState &LS) {
  std::optional<ClauseFamily> Family = getClauseFamily(DI, LS);
  return Family == ClauseFamily::VMEM || Family == ClauseFamily::Flat;
}

void addCfgEdge(unsigned From, unsigned To,
                std::vector<SmallVector<unsigned, 2>> &Successors,
                std::vector<SmallVector<unsigned, 2>> &Predecessors) {
  if (llvm::is_contained(Successors[From], To))
    return;
  Successors[From].push_back(To);
  Predecessors[To].push_back(From);
}

std::optional<uint64_t> applySignedPcDelta(uint64_t CapturedPc, int64_t Delta) {
  if (Delta >= 0)
    return checkedAddUint64(CapturedPc, static_cast<uint64_t>(Delta),
                            "HotSwap set-PC target");
  const uint64_t Magnitude = Delta == std::numeric_limits<int64_t>::min()
                                 ? uint64_t{1} << 63
                                 : static_cast<uint64_t>(-Delta);
  if (CapturedPc < Magnitude)
    return std::nullopt;
  return CapturedPc - Magnitude;
}

std::optional<std::pair<MCRegister, int64_t>>
getHotswapAddSetPc(ArrayRef<InternalDecodedInst> Decoded,
                   size_t SetPcGlobalIndex) {
  if (SetPcGlobalIndex == 0 || SetPcGlobalIndex >= Decoded.size())
    return std::nullopt;

  const InternalDecodedInst &Add = Decoded[SetPcGlobalIndex - 1];
  const InternalDecodedInst &SetPc = Decoded[SetPcGlobalIndex];
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

std::optional<uint64_t>
resolveContiguousHotswapSetPcTarget(ArrayRef<InternalDecodedInst> Decoded,
                                    size_t SetPcGlobalIndex) {
  std::optional<std::pair<MCRegister, int64_t>> AddSet =
      getHotswapAddSetPc(Decoded, SetPcGlobalIndex);
  if (!AddSet || SetPcGlobalIndex < 2)
    return std::nullopt;

  const InternalDecodedInst &GetPc = Decoded[SetPcGlobalIndex - 2];
  const InternalDecodedInst &Add = Decoded[SetPcGlobalIndex - 1];
  if (GetPc.Mnemonic != "s_get_pc_i64" ||
      GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size ||
      GetPc.Offset + GetPc.Size != Add.Offset ||
      GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != AddSet->first.id())
    return std::nullopt;
  return applySignedPcDelta(GetPc.Offset + GetPc.Size, AddSet->second);
}

std::optional<size_t>
findDecodedIndexAtOffset(ArrayRef<InternalDecodedInst> Decoded,
                         uint64_t Offset) {
  auto It = llvm::lower_bound(Decoded, Offset,
                              [](const InternalDecodedInst &DI,
                                 uint64_t Value) { return DI.Offset < Value; });
  if (It == Decoded.end() || It->Offset != Offset)
    return std::nullopt;
  return It - Decoded.begin();
}

std::optional<uint64_t>
resolveHotswapSetPcTarget(ArrayRef<InternalDecodedInst> AllDecoded,
                          uint64_t SetPcOffset) {
  std::optional<size_t> Index =
      findDecodedIndexAtOffset(AllDecoded, SetPcOffset);
  if (!Index)
    return std::nullopt;
  return resolveContiguousHotswapSetPcTarget(AllDecoded, *Index);
}

/// Resolve a get-PC/s_branch relay whose add/set-PC pair is in another
/// executable section. The displacement still uses the PC captured at source.
std::optional<uint64_t>
resolveHotswapRelayTarget(ArrayRef<InternalDecodedInst> AllDecoded,
                          const InternalDecodedInst &GetPc,
                          uint64_t RelayOffset) {
  std::optional<size_t> AddIndex =
      findDecodedIndexAtOffset(AllDecoded, RelayOffset);
  if (!AddIndex || *AddIndex + 1 >= AllDecoded.size())
    return std::nullopt;
  const size_t SetPcIndex = *AddIndex + 1;
  std::optional<std::pair<MCRegister, int64_t>> AddSet =
      getHotswapAddSetPc(AllDecoded, SetPcIndex);
  if (!AddSet || GetPc.Mnemonic != "s_get_pc_i64" ||
      GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
      GetPc.Inst.getOperand(0).getReg() != AddSet->first.id() ||
      GetPc.Offset > std::numeric_limits<uint64_t>::max() - GetPc.Size)
    return std::nullopt;
  return applySignedPcDelta(GetPc.Offset + GetPc.Size, AddSet->second);
}

struct HotswapTrampolineResume {
  uint64_t Offset = 0;
  bool HasVmem = false;
};

std::optional<HotswapTrampolineResume>
findHotswapTrampolineResume(ArrayRef<InternalDecodedInst> Decoded,
                            uint64_t Target, const KernelTextRange &Range,
                            const LLVMState &LS) {
  if (!LS.MIA)
    return std::nullopt;

  size_t RemainingInstructions = Decoded.size();
  bool HasVmem = false;
  DenseSet<uint64_t> VisitedOffsets;
  while (RemainingInstructions != 0) {
    auto It = llvm::lower_bound(
        Decoded, Target, [](const InternalDecodedInst &DI, uint64_t Offset) {
          return DI.Offset < Offset;
        });
    if (It == Decoded.end() || It->Offset != Target)
      return std::nullopt;

    uint64_t ExpectedOffset = Target;
    bool FollowedHop = false;
    for (; It != Decoded.end(); ++It) {
      if (RemainingInstructions == 0)
        return std::nullopt;
      --RemainingInstructions;
      if (!VisitedOffsets.insert(It->Offset).second ||
          It->Offset != ExpectedOffset || It->Size == 0 ||
          It->Offset > std::numeric_limits<uint64_t>::max() - It->Size)
        return std::nullopt;
      ExpectedOffset = It->Offset + It->Size;
      const size_t GlobalIndex = It - Decoded.begin();
      if (It->Mnemonic == "<unknown>")
        return std::nullopt;
      HasVmem |= isVmemOrFlat(*It, LS);

      if (It->Mnemonic == "s_set_pc_i64") {
        std::optional<uint64_t> Next =
            resolveContiguousHotswapSetPcTarget(Decoded, GlobalIndex);
        if (!Next)
          return std::nullopt;
        if (*Next >= Range.Begin && *Next < Range.End)
          return HotswapTrampolineResume{*Next, HasVmem};
        Target = *Next;
        FollowedHop = true;
        break;
      }

      const MCInstrDesc &Desc = LS.MCII->get(It->Inst.getOpcode());
      const bool IsCall = LS.MIA->isCall(It->Inst);
      const bool IsReturn = LS.MIA->isReturn(It->Inst);
      const bool IsBranch = LS.MIA->isBranch(It->Inst);
      if (IsCall || IsReturn || Desc.isTrap())
        return std::nullopt;
      if (IsBranch) {
        uint64_t Next = 0;
        if (!LS.MIA->isUnconditionalBranch(It->Inst) ||
            LS.MIA->isIndirectBranch(It->Inst) ||
            !LS.MIA->evaluateBranch(It->Inst, It->Offset, It->Size, Next))
          return std::nullopt;
        if (Next >= Range.Begin && Next < Range.End)
          return HotswapTrampolineResume{Next, HasVmem};
        Target = Next;
        FollowedHop = true;
        break;
      }
      if (Desc.isTerminator() ||
          LS.MIA->mayAffectControlFlow(It->Inst, *LS.MRI))
        return std::nullopt;
    }
    if (!FollowedHop)
      return std::nullopt;
  }
  return std::nullopt;
}

/// Analyze one descriptor-backed kernel range. Unknown control flow is modeled
/// as reaching every instruction in the range, so a prior-VMEM fact is retained
/// only when every modeled path proves it.
void analyzeKernelRange(ArrayRef<InternalDecodedInst> Decoded,
                        ArrayRef<InternalDecodedInst> AllDecoded,
                        const KernelTextRange &Range, const LLVMState &LS,
                        InitialVmemMustAnalysis &Result) {
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
  if (GlobalIndices.empty())
    return;

  for (size_t GlobalIdx : GlobalIndices)
    Result.DescriptorCovered.set(GlobalIdx);

  auto MergeConservativeRange = [&] {
    for (size_t GlobalIdx : GlobalIndices) {
      Result.Reachable.set(GlobalIdx);
      Result.MustHavePriorVmem.reset(GlobalIdx);
    }
  };
  if (Decoded[GlobalIndices.front()].Offset != Range.Begin ||
      Range.HasArbitraryIndirectIngress) {
    MergeConservativeRange();
    return;
  }

  const unsigned Count = GlobalIndices.size();
  DenseMap<uint64_t, unsigned> OffsetToLocal;
  for (unsigned I = 0; I < Count; ++I)
    OffsetToLocal.try_emplace(Decoded[GlobalIndices[I]].Offset, I);

  BitVector EntryNodes(Count);
  for (uint64_t Entry : Range.Entries) {
    auto It = OffsetToLocal.find(Entry);
    if (It == OffsetToLocal.end()) {
      MergeConservativeRange();
      return;
    }
    EntryNodes.set(It->second);
  }
  if (EntryNodes.none()) {
    MergeConservativeRange();
    return;
  }

  std::vector<SmallVector<unsigned, 2>> Successors(Count);
  std::vector<SmallVector<unsigned, 2>> Predecessors(Count);
  BitVector UnknownSuccessors(Count);
  BitVector IsVmem(Count);
  SmallVector<unsigned> HotswapSetPcCandidates;

  for (unsigned I = 0; I < Count; ++I) {
    const InternalDecodedInst &DI = Decoded[GlobalIndices[I]];
    if (isVmemOrFlat(DI, LS))
      IsVmem.set(I);

    const bool HasFallthrough = I + 1 < Count;
    if (DI.Mnemonic == "<unknown>") {
      UnknownSuccessors.set(I);
      continue;
    }

    const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
    const bool IsCall = LS.MIA ? LS.MIA->isCall(DI.Inst) : Desc.isCall();
    const bool IsReturn = LS.MIA ? LS.MIA->isReturn(DI.Inst) : Desc.isReturn();
    const bool IsBranch = LS.MIA ? LS.MIA->isBranch(DI.Inst) : Desc.isBranch();
    if (IsReturn)
      continue;

    // A debug trap can resume at the following instruction.
    if (Desc.isTrap()) {
      if (HasFallthrough)
        addCfgEdge(I, I + 1, Successors, Predecessors);
      continue;
    }

    if (IsCall) {
      if (!Desc.isTerminator() && HasFallthrough)
        addCfgEdge(I, I + 1, Successors, Predecessors);
      UnknownSuccessors.set(I);
      continue;
    }

    if (DI.Mnemonic == "s_set_pc_i64") {
      HotswapSetPcCandidates.push_back(I);
      continue;
    }

    if (IsBranch) {
      const bool IsIndirect =
          LS.MIA ? LS.MIA->isIndirectBranch(DI.Inst) : Desc.isIndirectBranch();
      const bool IsConditional = LS.MIA ? LS.MIA->isConditionalBranch(DI.Inst)
                                        : Desc.isConditionalBranch();
      const bool IsUnconditional = LS.MIA
                                       ? LS.MIA->isUnconditionalBranch(DI.Inst)
                                       : Desc.isUnconditionalBranch();

      bool TargetKnown = false;
      if (!IsIndirect && LS.MIA) {
        uint64_t Target = 0;
        if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target)) {
          TargetKnown = true;
          if (Target >= Range.Begin && Target < Range.End) {
            auto TargetIt = OffsetToLocal.find(Target);
            if (TargetIt == OffsetToLocal.end()) {
              UnknownSuccessors.set(I);
              TargetKnown = false;
            } else {
              addCfgEdge(I, TargetIt->second, Successors, Predecessors);
            }
          } else if (IsUnconditional) {
            std::optional<HotswapTrampolineResume> Resume =
                findHotswapTrampolineResume(AllDecoded, Target, Range, LS);
            if (!Resume && I != 0 && DI.Mnemonic == "s_branch") {
              const InternalDecodedInst &GetPc = Decoded[GlobalIndices[I - 1]];
              if (GetPc.Offset <=
                      std::numeric_limits<uint64_t>::max() - GetPc.Size &&
                  GetPc.Offset + GetPc.Size == DI.Offset &&
                  GetPc.Mnemonic == "s_get_pc_i64") {
                std::optional<uint64_t> PoolTarget =
                    resolveHotswapRelayTarget(AllDecoded, GetPc, Target);
                if (PoolTarget)
                  Resume = findHotswapTrampolineResume(AllDecoded, *PoolTarget,
                                                       Range, LS);
              }
            }
            if (Resume) {
              auto ResumeIt = OffsetToLocal.find(Resume->Offset);
              if (ResumeIt == OffsetToLocal.end()) {
                UnknownSuccessors.set(I);
                TargetKnown = false;
              } else {
                addCfgEdge(I, ResumeIt->second, Successors, Predecessors);
                if (Resume->HasVmem)
                  IsVmem.set(I);
              }
            } else {
              UnknownSuccessors.set(I);
              TargetKnown = false;
            }
          }
        }
      }
      if (!TargetKnown)
        UnknownSuccessors.set(I);

      if (IsConditional && HasFallthrough)
        addCfgEdge(I, I + 1, Successors, Predecessors);
      else if (!IsConditional && !IsUnconditional)
        UnknownSuccessors.set(I);
      continue;
    }

    if (Desc.isTerminator())
      continue;
    if (LS.MIA && LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI)) {
      UnknownSuccessors.set(I);
      continue;
    }
    if (HasFallthrough)
      addCfgEdge(I, I + 1, Successors, Predecessors);
  }

  for (unsigned I : HotswapSetPcCandidates) {
    const InternalDecodedInst &SetPc = Decoded[GlobalIndices[I]];
    std::optional<uint64_t> Target =
        resolveHotswapSetPcTarget(AllDecoded, SetPc.Offset);
    if (!Target) {
      UnknownSuccessors.set(I);
      continue;
    }
    if (*Target >= Range.Begin && *Target < Range.End) {
      auto TargetIt = OffsetToLocal.find(*Target);
      if (TargetIt == OffsetToLocal.end())
        UnknownSuccessors.set(I);
      else
        addCfgEdge(I, TargetIt->second, Successors, Predecessors);
      continue;
    }

    std::optional<HotswapTrampolineResume> Resume =
        findHotswapTrampolineResume(AllDecoded, *Target, Range, LS);
    if (!Resume) {
      UnknownSuccessors.set(I);
      continue;
    }
    auto ResumeIt = OffsetToLocal.find(Resume->Offset);
    if (ResumeIt == OffsetToLocal.end()) {
      UnknownSuccessors.set(I);
      continue;
    }
    addCfgEdge(I, ResumeIt->second, Successors, Predecessors);
    if (Resume->HasVmem)
      IsVmem.set(I);
  }

  BitVector Reachable(Count);
  SmallVector<unsigned> Worklist;
  for (int Entry = EntryNodes.find_first(); Entry != -1;
       Entry = EntryNodes.find_next(Entry)) {
    Reachable.set(Entry);
    Worklist.push_back(Entry);
  }
  for (size_t Next = 0; Next < Worklist.size(); ++Next) {
    const unsigned I = Worklist[Next];
    if (UnknownSuccessors.test(I)) {
      Reachable.set();
      break;
    }
    for (unsigned Target : Successors[I]) {
      if (!Reachable.test(Target)) {
        Reachable.set(Target);
        Worklist.push_back(Target);
      }
    }
  }

  SmallVector<uint8_t> MustIn(Count, 1);
  SmallVector<uint8_t> MustOut(Count, 1);
  for (int Entry = EntryNodes.find_first(); Entry != -1;
       Entry = EntryNodes.find_next(Entry)) {
    MustIn[Entry] = 0;
    MustOut[Entry] = IsVmem.test(Entry);
  }

  bool Changed = true;
  while (Changed) {
    Changed = false;
    bool HasReachableUnknown = false;
    bool UnknownMustOut = true;
    for (int Pred = UnknownSuccessors.find_first(); Pred != -1;
         Pred = UnknownSuccessors.find_next(Pred)) {
      if (!Reachable.test(Pred))
        continue;
      HasReachableUnknown = true;
      UnknownMustOut &= MustOut[Pred] != 0;
    }

    for (unsigned I = 0; I < Count; ++I) {
      if (!Reachable.test(I))
        continue;
      bool NewIn = !EntryNodes.test(I);
      bool SawPredecessor = EntryNodes.test(I);
      if (!EntryNodes.test(I)) {
        for (unsigned Pred : Predecessors[I]) {
          if (!Reachable.test(Pred))
            continue;
          SawPredecessor = true;
          NewIn &= MustOut[Pred] != 0;
        }
        if (HasReachableUnknown) {
          SawPredecessor = true;
          NewIn &= UnknownMustOut;
        }
        if (!SawPredecessor)
          NewIn = false;
      }
      const bool NewOut = NewIn || IsVmem.test(I);
      if (MustIn[I] != NewIn || MustOut[I] != NewOut) {
        MustIn[I] = NewIn;
        MustOut[I] = NewOut;
        Changed = true;
      }
    }
  }

  for (unsigned I = 0; I < Count; ++I) {
    if (!Reachable.test(I))
      continue;
    const size_t GlobalIdx = GlobalIndices[I];
    if (Result.Reachable.test(GlobalIdx)) {
      if (!MustIn[I])
        Result.MustHavePriorVmem.reset(GlobalIdx);
    } else {
      Result.Reachable.set(GlobalIdx);
      if (MustIn[I])
        Result.MustHavePriorVmem.set(GlobalIdx);
      else
        Result.MustHavePriorVmem.reset(GlobalIdx);
    }
  }
}

bool isInitialVmemClause(const PatchContext &Ctx, size_t Idx) {
  if (!clauseContainsVmem(Ctx.Decoded, Idx, Ctx.LS))
    return false;
  if (!Ctx.InitialVmemAnalysis ||
      Idx >= Ctx.InitialVmemAnalysis->DescriptorCovered.size() ||
      !Ctx.InitialVmemAnalysis->DescriptorCovered.test(Idx))
    return true;
  if (!Ctx.InitialVmemAnalysis->Reachable.test(Idx))
    return false;
  return !Ctx.InitialVmemAnalysis->MustHavePriorVmem.test(Idx);
}

void releaseClauseMemberRelocationProtection(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &Clause = Ctx.Decoded[Idx];
  if (Clause.Inst.getNumOperands() == 0 || !Clause.Inst.getOperand(0).isImm())
    return;

  const size_t MemberCount =
      (static_cast<uint64_t>(Clause.Inst.getOperand(0).getImm()) & 63) + 1;
  const size_t Available = std::min(MemberCount, Ctx.Decoded.size() - Idx - 1);
  const size_t End = Idx + 1 + Available;
  for (size_t I = Idx + 1; I != End; ++I) {
    const uint64_t Offset = Ctx.Decoded[I].Offset;
    auto It = Ctx.ClauseRelocationProtectionCounts.find(Offset);
    if (It == Ctx.ClauseRelocationProtectionCounts.end())
      continue;
    if (--It->second != 0)
      continue;
    Ctx.ClauseRelocationProtectionCounts.erase(It);
    if (!Ctx.NonClauseRelocationProtectedOffsets.contains(Offset))
      Ctx.RelocationProtectedOffsets.erase(Offset);
  }
}

} // anonymous namespace

InitialVmemMustAnalysis
computeInitialVmemMustAnalysis(ArrayRef<InternalDecodedInst> Decoded,
                               ArrayRef<InternalDecodedInst> AllDecoded,
                               ArrayRef<KernelTextRange> KernelRanges,
                               const LLVMState &LS) {
  InitialVmemMustAnalysis Result{BitVector(Decoded.size()),
                                 BitVector(Decoded.size()),
                                 BitVector(Decoded.size())};
  for (const KernelTextRange &Range : KernelRanges)
    analyzeKernelRange(Decoded, AllDecoded, Range, LS, Result);
  return Result;
}

static uint32_t applyInPlacePatchesImpl(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  StringRef Mnemonic(DI.Mnemonic);

  StringRef ReplacementAsm = getClusterLoadReplacementAsm(Mnemonic);
  if (!ReplacementAsm.empty()) {
    HotswapProfile::Scope S =
        Ctx.Profile.time(HotswapMetric::InPlaceClusterLoad);
    // The replacement templates above are all the saddr=off encoding form
    // (global address in a 64-bit VGPR pair). The SGPR-relative (_SADDR)
    // cluster_load shares the mnemonic but has a different operand layout, so
    // reusing the off-form opcode would re-encode its scalar saddr and 32-bit
    // vaddr as a 64-bit vaddr plus an inline offset -- a corrupt address that
    // faults the GPU at runtime. Leave the _SADDR form unchanged here so the
    // later trampoline pass can preserve the cluster-load opcode and wrap it
    // with the A0-required M0 wg_mask clear/restore sequence.
    if (usesSgprBaseAddress(DI.Inst, *Ctx.LS.MRI)) {
      log() << "hotswap: inplace: " << Mnemonic
            << " (SGPR-relative saddr form) left unchanged at 0x"
            << utohexstr(DI.Offset) << "\n";
    } else {
      std::optional<unsigned> NewOpcode = resolveOpcode(ReplacementAsm, Ctx.LS);
      if (NewOpcode && swapOpcode(DI, Ctx.Text, Ctx.LS, *NewOpcode)) {
        log() << "hotswap: inplace: " << Mnemonic << " -> opcode " << *NewOpcode
              << " at 0x" << utohexstr(DI.Offset) << "\n";
        Ctx.RequiredPatchApplied = true;
        S.addPatches(1);
        return 1;
      }
      return failRequiredInPlacePatch(Ctx, Mnemonic, DI.Offset);
    }
  }

  if (DI.Inst.getOpcode() == Ctx.LS.SClauseOpcode) {
    const bool InitialVmemClause = isInitialVmemClause(Ctx, Idx);
    const bool UniformScope = clauseHasUniformScope(Ctx, Idx);

    RewriteRule Rule;
    Rule.ReplaceBytes.assign(Ctx.LS.SNopBytes.begin(), Ctx.LS.SNopBytes.end());
    if (applyByteReplace(Rule, DI.Offset, DI.Size, Ctx.Text, Ctx.TextSize,
                         Ctx.LS)) {
      releaseClauseMemberRelocationProtection(Ctx, Idx);
      log() << "hotswap: inplace: "
            << (InitialVmemClause
                    ? "initial-VMEM"
                    : (UniformScope ? "uniform-scope"
                                    : "mixed/unsupported-scope"))
            << " s_clause -> s_nop 0 at 0x" << utohexstr(DI.Offset) << "\n";
      Ctx.RequiredPatchApplied = true;
      return 1;
    }
    return failRequiredInPlacePatch(Ctx, Mnemonic, DI.Offset);
  }

  // s_barrier_signal_isfirst -> s_barrier_signal: on A0, the isfirst
  // variant may return stale SCC when cluster barriers are in flight.
  // Both S_BARRIER_SIGNAL_IMM and S_BARRIER_SIGNAL_ISFIRST_IMM share
  // a single SplitBarrier:$src0 immediate operand (see SOPInstructions.td),
  // so cloning the decoded MCInst and flipping the opcode preserves the
  // original barrier-ID operand. The dummy "-1" is only used to resolve
  // the target opcode via the asm parser.
  //
  // Correctness caveat: the isfirst variant defines SCC; the non-isfirst
  // variant does not. If downstream code reads SCC expecting the result
  // of isfirst (e.g. an s_cbranch_scc1 selecting the elected wave), the
  // swap leaves that read consuming stale SCC. On A0 the isfirst result
  // is already unreliable due to the underlying race, so the swap removes
  // a known-broken code path rather than introducing a new one. But it
  // is not a semantic equivalence. Liveness/CFG-aware detection of SCC
  // consumers is undecidable in general; the proper fix lives in
  // A0-targeted Clang codegen and is out of scope for hotswap. This
  // patch is a runtime mitigation for B0 binaries running on A0.
  //
  // The _M0 form has a different tablegen mnemonic string
  // ("s_barrier_signal_isfirst m0", with the "m0" baked into the
  // mnemonic itself, not as an operand -- see S_BARRIER_SIGNAL_ISFIRST_M0
  // in SOPInstructions.td), so it does not match this equality check
  // and falls through to the dispatcher's "no match" return below.
  // The AMDGPU backend never emits the _M0 form for compute kernels.
  if (Mnemonic == "s_barrier_signal_isfirst") {
    HotswapProfile::Scope S =
        Ctx.Profile.time(HotswapMetric::InPlaceBarrierSignal);
    std::optional<unsigned> NewOpcode =
        resolveOpcode("s_barrier_signal -1", Ctx.LS);
    if (NewOpcode && swapOpcode(DI, Ctx.Text, Ctx.LS, *NewOpcode)) {
      log() << "hotswap: inplace: s_barrier_signal_isfirst -> opcode "
            << *NewOpcode << " at 0x" << utohexstr(DI.Offset) << "\n";
      Ctx.RequiredPatchApplied = true;
      S.addPatches(1);
      return 1;
    }
    return failRequiredInPlacePatch(Ctx, Mnemonic, DI.Offset);
  }

  return 0;
}

void registerInPlacePatch(HotswapPatchVTable &VT) {
  VT.applyInPlacePatches = &applyInPlacePatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
