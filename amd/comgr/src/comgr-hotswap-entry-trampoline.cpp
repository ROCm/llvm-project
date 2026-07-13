//===- comgr-hotswap-entry-trampoline.cpp - Kernel-entry stubs ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Kernel-entry redirection pass for COMGR HotSwap. This pass is
/// independent of the gfx1250 B0-to-A0 instruction patcher: it appends one
/// PC-relative entry stub per kernel descriptor and rewrites the descriptor's
/// kernel_code_entry_byte_offset to point at that stub.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

static SmallVector<uint8_t> getCodeEndBytes(const LLVMState &LS) {
  SmallVector<uint8_t> CodeEnd(LS.SCodeEndBytes.begin(),
                               LS.SCodeEndBytes.end());
  if (CodeEnd.empty())
    log() << "hotswap: error: missing cached s_code_end for entry-stub "
          << "padding.\n";
  return CodeEnd;
}

static std::optional<uint64_t> checkedAdd(uint64_t LHS, uint64_t RHS,
                                          StringRef Context) {
  std::optional<uint64_t> Result = checkedAddUnsigned(LHS, RHS);
  if (Result)
    return Result;

  log() << "hotswap: error: " << Context << " overflows uint64_t.\n";
  return std::nullopt;
}

struct KernelEntryMCInstTemplate {
  MCInst GetPc;
  MCInst AddLo;
  MCInst AddHi;
  MCInst SetPc;
};

class KernelEntryTrampolineBuilder {
public:
  explicit KernelEntryTrampolineBuilder(const LLVMState &LS) : LS(LS) {}

  bool prepare(ArrayRef<unsigned> ScratchSgprs) {
    SmallVector<unsigned, 32> Missing;
    for (unsigned ScratchSgpr : ScratchSgprs) {
      if (ScratchSgpr == std::numeric_limits<unsigned>::max()) {
        log() << "hotswap: error: kernel-entry stub scratch SGPR pair "
              << "overflows unsigned.\n";
        return false;
      }
      if (!Templates.contains(ScratchSgpr))
        Missing.push_back(ScratchSgpr);
    }
    std::sort(Missing.begin(), Missing.end());
    Missing.erase(std::unique(Missing.begin(), Missing.end()), Missing.end());
    if (Missing.empty())
      return true;

    std::string Asm;
    Asm.reserve(Missing.size() * 96);
    for (unsigned ScratchSgpr : Missing)
      appendTemplateAsm(Asm, ScratchSgpr);

    SmallVector<MCInst, 8> Insts = parseMCInsts(Asm, LS);
    if (Insts.size() != Missing.size() * 4) {
      log() << "hotswap: error: failed to parse batched entry-stub MCInst "
            << "templates: expected " << Missing.size() * 4
            << " instructions, got " << Insts.size() << ".\n";
      return false;
    }

    for (size_t I = 0; I < Missing.size(); ++I) {
      const size_t Base = I * 4;
      if (Insts[Base + 1].getNumOperands() <= 2 ||
          !Insts[Base + 1].getOperand(2).isImm() ||
          Insts[Base + 2].getNumOperands() <= 2 ||
          !Insts[Base + 2].getOperand(2).isImm()) {
        log() << "hotswap: error: malformed entry-stub MCInst template for "
              << "scratch SGPR " << Missing[I] << ".\n";
        return false;
      }
      KernelEntryMCInstTemplate Template{Insts[Base], Insts[Base + 1],
                                         Insts[Base + 2], Insts[Base + 3]};
      Templates.try_emplace(Missing[I], std::move(Template));
    }
    return true;
  }

  SmallVector<uint8_t> build(uint64_t StubVAddr, uint64_t EntryVAddr,
                             unsigned ScratchSgpr) {
    if (ScratchSgpr == std::numeric_limits<unsigned>::max()) {
      log() << "hotswap: error: kernel-entry stub scratch SGPR pair overflows "
            << "unsigned.\n";
      return {};
    }
    if (LS.KernelEntryPrefixBytes.empty() || LS.SCodeEndBytes.empty()) {
      log() << "hotswap: error: kernel-entry stub has no cached prefix or "
            << "s_code_end encoding.\n";
      return {};
    }

    KernelEntryMCInstTemplate *Template = getTemplate(ScratchSgpr);
    if (!Template)
      return {};

    SmallVector<uint8_t> Bytes(LS.KernelEntryPrefixBytes.begin(),
                               LS.KernelEntryPrefixBytes.end());
    if (!appendEncoded(Bytes, Template->GetPc))
      return {};

    // s_get_pc_i64 returns the address of the following s_add_u32 instruction.
    // Materialize the original entry with a 64-bit PC-relative add so the code
    // object can be rewritten before ROCR knows final device addresses.
    std::optional<uint64_t> PcBase =
        checkedAdd(StubVAddr, static_cast<uint64_t>(Bytes.size()),
                   "kernel-entry stub PC base");
    if (!PcBase)
      return {};
    // Unsigned subtraction is intentional: the immediate pair materializes
    // the 64-bit two's-complement delta, including backward jumps.
    const uint64_t Delta = EntryVAddr - *PcBase;

    MCInst AddLo = Template->AddLo;
    MCInst AddHi = Template->AddHi;
    AddLo.getOperand(2).setImm(static_cast<uint32_t>(Delta));
    AddHi.getOperand(2).setImm(static_cast<uint32_t>(Delta >> 32));
    if (!appendEncoded(Bytes, AddLo) || !appendEncoded(Bytes, AddHi) ||
        !appendEncoded(Bytes, Template->SetPc))
      return {};

    if (Bytes.size() > KernelEntryStubStride) {
      log() << "hotswap: error: kernel-entry stub grew past "
            << KernelEntryStubStride << " bytes.\n";
      return {};
    }
    while (Bytes.size() < KernelEntryStubStride) {
      if (Bytes.size() + LS.SCodeEndBytes.size() > KernelEntryStubStride) {
        log() << "hotswap: error: s_code_end padding does not evenly fill "
              << "kernel-entry stub stride " << KernelEntryStubStride << ".\n";
        return {};
      }
      Bytes.append(LS.SCodeEndBytes.begin(), LS.SCodeEndBytes.end());
    }
    return Bytes;
  }

private:
  bool appendEncoded(SmallVectorImpl<uint8_t> &Out, const MCInst &Inst) const {
    SmallVector<uint8_t> Encoded = encodeHotswapMCInst(Inst, LS);
    if (Encoded.empty()) {
      log() << "hotswap: error: failed to encode entry-stub MCInst opcode "
            << Inst.getOpcode() << ".\n";
      return false;
    }
    Out.append(Encoded.begin(), Encoded.end());
    return true;
  }

  KernelEntryMCInstTemplate *getTemplate(unsigned ScratchSgpr) {
    DenseMap<unsigned, KernelEntryMCInstTemplate>::iterator Existing =
        Templates.find(ScratchSgpr);
    if (Existing != Templates.end())
      return &Existing->second;

    const unsigned Scratch[] = {ScratchSgpr};
    if (!prepare(Scratch))
      return nullptr;
    return &Templates.find(ScratchSgpr)->second;
  }

  static void appendTemplateAsm(std::string &Asm, unsigned ScratchSgpr) {
    std::string ScratchPair =
        (Twine("s[") + Twine(ScratchSgpr) + ":" + Twine(ScratchSgpr + 1) + "]")
            .str();
    std::string ScratchLo = (Twine("s") + Twine(ScratchSgpr)).str();
    std::string ScratchHi = (Twine("s") + Twine(ScratchSgpr + 1)).str();
    Asm += (Twine("s_get_pc_i64 ") + ScratchPair + "\n" + "s_add_u32 " +
            ScratchLo + ", " + ScratchLo + ", 0\n" + "s_addc_u32 " + ScratchHi +
            ", " + ScratchHi + ", 0\n" + "s_set_pc_i64 " + ScratchPair + "\n")
               .str();
  }

  const LLVMState &LS;
  DenseMap<unsigned, KernelEntryMCInstTemplate> Templates;
};

SmallVector<uint8_t> buildKernelEntryTrampoline(uint64_t StubVAddr,
                                                uint64_t EntryVAddr,
                                                unsigned ScratchSgpr,
                                                const LLVMState &LS) {
  KernelEntryTrampolineBuilder Builder(LS);
  return Builder.build(StubVAddr, EntryVAddr, ScratchSgpr);
}

uint64_t computeKernelEntryPrefetchGuardBytes(uint32_t InstPrefLines) {
  const uint64_t PrefetchBytes =
      static_cast<uint64_t>(InstPrefLines) * KernelEntryInstPrefUnitBytes;
  if (PrefetchBytes <= KernelEntryStubStride)
    return 0;
  return PrefetchBytes - KernelEntryStubStride;
}

static bool hasResolvedEntryStubState(const LLVMState &LS, StringRef Context) {
  if (!LS.MCII || LS.GlobalWbOpcode >= LS.MCII->getNumOpcodes() ||
      LS.SGetPcI64Opcode >= LS.MCII->getNumOpcodes() ||
      LS.SAddU32Opcode >= LS.MCII->getNumOpcodes() ||
      LS.SAddcU32Opcode >= LS.MCII->getNumOpcodes() ||
      LS.SSetPcI64Opcode >= LS.MCII->getNumOpcodes()) {
    log() << "hotswap: error: " << Context
          << ": LLVMState lacks resolved entry-stub opcodes.\n";
    return false;
  }

  if (!LS.MRI) {
    log() << "hotswap: error: " << Context
          << ": LLVMState lacks register info.\n";
    return false;
  }

  return true;
}

static bool decodeKernelEntryStub(ArrayRef<uint8_t> Bytes, const LLVMState &LS,
                                  std::vector<InternalDecodedInst> &Decoded,
                                  StringRef Context) {
  if (Bytes.size() < KernelEntryStubStride)
    return false;

  if (!hasResolvedEntryStubState(LS, Context))
    return false;

  if (!decodeTextSection(Bytes.data(), KernelEntryStubStride, LS, Decoded)) {
    log() << "hotswap: error: " << Context << ": failed to decode "
          << KernelEntryStubStride << "-byte candidate.\n";
    return false;
  }
  return Decoded.size() >= 6;
}

static bool hasRegOperand(const MCInst &Inst, unsigned Index) {
  return Inst.getNumOperands() > Index && Inst.getOperand(Index).isReg();
}

static bool hasImmOperand(const MCInst &Inst, unsigned Index) {
  return Inst.getNumOperands() > Index && Inst.getOperand(Index).isImm();
}

static bool sameRegOperand(const MCInst &LHS, unsigned LHSIndex,
                           const MCInst &RHS, unsigned RHSIndex) {
  return hasRegOperand(LHS, LHSIndex) && hasRegOperand(RHS, RHSIndex) &&
         LHS.getOperand(LHSIndex).getReg() == RHS.getOperand(RHSIndex).getReg();
}

static bool hasEntryStubOperandShape(ArrayRef<InternalDecodedInst> Decoded,
                                     const LLVMState &LS) {
  if (Decoded.size() < 6)
    return false;

  if (Decoded[0].Inst.getOpcode() != LS.GlobalWbOpcode ||
      Decoded[1].Inst.getOpcode() != LS.VNopInst.getOpcode() ||
      Decoded[2].Inst.getOpcode() != LS.SGetPcI64Opcode ||
      Decoded[3].Inst.getOpcode() != LS.SAddU32Opcode ||
      Decoded[4].Inst.getOpcode() != LS.SAddcU32Opcode ||
      Decoded[5].Inst.getOpcode() != LS.SSetPcI64Opcode)
    return false;

  const MCInst &GlobalWb = Decoded[0].Inst;
  const MCInst &VNop = Decoded[1].Inst;
  const MCInst &GetPc = Decoded[2].Inst;
  const MCInst &AddLo = Decoded[3].Inst;
  const MCInst &AddHi = Decoded[4].Inst;
  const MCInst &SetPc = Decoded[5].Inst;

  if (GlobalWb.getNumOperands() != 1 || !GlobalWb.getOperand(0).isImm() ||
      GlobalWb.getOperand(0).getImm() != 0 || VNop.getNumOperands() != 0)
    return false;

  if (GetPc.getNumOperands() != 1 || SetPc.getNumOperands() != 1 ||
      !sameRegOperand(GetPc, 0, SetPc, 0))
    return false;

  if (AddLo.getNumOperands() != 3 || AddHi.getNumOperands() != 3 ||
      !sameRegOperand(AddLo, 0, AddLo, 1) ||
      !sameRegOperand(AddHi, 0, AddHi, 1) || !hasImmOperand(AddLo, 2) ||
      !hasImmOperand(AddHi, 2))
    return false;

  MCRegister PairReg = GetPc.getOperand(0).getReg();
  MCRegister LoReg = AddLo.getOperand(0).getReg();
  MCRegister HiReg = AddHi.getOperand(0).getReg();
  unsigned LoSubRegIndex = LS.MRI->getSubRegIndex(PairReg, LoReg);
  unsigned HiSubRegIndex = LS.MRI->getSubRegIndex(PairReg, HiReg);
  return LoSubRegIndex != 0 && HiSubRegIndex != 0 &&
         LoSubRegIndex != HiSubRegIndex && LoSubRegIndex < HiSubRegIndex;
}

static std::optional<uint64_t>
decodeEntryStubTargetVAddr(ArrayRef<InternalDecodedInst> Decoded,
                           uint64_t StubVAddr) {
  std::optional<uint64_t> PcBaseOffset =
      checkedAdd(Decoded[2].Offset, Decoded[2].Size,
                 "decoded kernel-entry stub PC-base offset");
  if (!PcBaseOffset)
    return std::nullopt;
  std::optional<uint64_t> PcBase =
      checkedAdd(StubVAddr, *PcBaseOffset, "decoded kernel-entry stub PC base");
  if (!PcBase)
    return std::nullopt;

  const uint64_t Lo =
      static_cast<uint32_t>(Decoded[3].Inst.getOperand(2).getImm());
  const uint64_t Hi =
      static_cast<uint32_t>(Decoded[4].Inst.getOperand(2).getImm());
  const uint64_t Delta = Lo | (Hi << 32);
  return *PcBase + Delta;
}

bool isKernelEntryTrampoline(ArrayRef<uint8_t> Bytes, const LLVMState &LS) {
  std::vector<InternalDecodedInst> Decoded;
  return decodeKernelEntryStub(Bytes, LS, Decoded, "isKernelEntryTrampoline") &&
         hasEntryStubOperandShape(Decoded, LS);
}

bool hasKernelEntryTrampolinePrefix(ArrayRef<uint8_t> Bytes,
                                    const LLVMState &LS) {
  return !LS.KernelEntryPrefixBytes.empty() &&
         Bytes.size() >= LS.KernelEntryPrefixBytes.size() &&
         std::equal(LS.KernelEntryPrefixBytes.begin(),
                    LS.KernelEntryPrefixBytes.end(), Bytes.begin());
}

static std::optional<uint64_t>
checkedAlignTo(uint64_t Value, uint64_t Alignment, StringRef Context) {
  if (Alignment == 0)
    return Value;

  uint64_t Remainder = Value % Alignment;
  if (Remainder == 0)
    return Value;
  return checkedAdd(Value, Alignment - Remainder, Context);
}

static std::optional<uint64_t> entryVAddr(const KernelDescriptorInfo &KD) {
  if (KD.EntryOffset >= 0)
    return checkedAdd(
        KD.VAddr, static_cast<uint64_t>(KD.EntryOffset),
        (Twine("kernel entry vaddr for '") + KD.KernelName + "'").str());

  const uint64_t Magnitude =
      KD.EntryOffset == std::numeric_limits<int64_t>::min()
          ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1
          : static_cast<uint64_t>(-KD.EntryOffset);
  if (KD.VAddr < Magnitude) {
    log() << "hotswap: error: kernel entry vaddr for '" << KD.KernelName
          << "' underflows uint64_t.\n";
    return std::nullopt;
  }
  return KD.VAddr - Magnitude;
}

static std::optional<bool> descriptorAlreadyTargetsEntryStub(
    const ElfView &Elf, const KernelDescriptorInfo &KD, const LLVMState &LS) {
  std::optional<uint64_t> Entry = entryVAddr(KD);
  if (!Entry)
    return std::nullopt;
  if (*Entry < Elf.textAddr())
    return false;

  std::optional<uint64_t> TextEnd =
      checkedAdd(Elf.textAddr(), Elf.textSize(), "entry trampoline text end");
  if (!TextEnd)
    return std::nullopt;

  const uint64_t TextOffset = *Entry - Elf.textAddr();
  if (TextOffset > Elf.textSize() ||
      KernelEntryStubStride > Elf.textSize() - TextOffset)
    return false;

  ArrayRef<uint8_t> Candidate(Elf.textData() + TextOffset,
                              KernelEntryStubStride);
  // The full idempotency matcher uses LLVM's AMDGPU disassembler. Avoid
  // running it over arbitrary original kernel entry bytes; real code objects
  // can contain byte streams that are valid executable code but still trip
  // decoder corner cases before COMGR can finish rewriting.
  if (!hasKernelEntryTrampolinePrefix(Candidate, LS))
    return false;

  std::vector<InternalDecodedInst> Decoded;
  if (!decodeKernelEntryStub(Candidate, LS, Decoded,
                             "entry trampoline idempotency matcher"))
    return false;
  if (!hasEntryStubOperandShape(Decoded, LS))
    return false;

  std::optional<uint64_t> Target = decodeEntryStubTargetVAddr(Decoded, *Entry);
  if (!Target)
    return std::nullopt;

  return *Target >= Elf.textAddr() && *Target < *TextEnd && *Target < *Entry;
}

static std::optional<uint64_t>
totalTrampolineBytes(ArrayRef<Trampoline> Trampolines) {
  uint64_t Total = 0;
  for (const Trampoline &T : Trampolines) {
    std::optional<uint64_t> NewTotal =
        checkedAdd(Total, static_cast<uint64_t>(T.Bytes.size()),
                   "existing trampoline byte count");
    if (!NewTotal)
      return std::nullopt;
    Total = *NewTotal;
  }
  return Total;
}

static std::optional<int64_t>
checkedSignedDifference(uint64_t LHS, uint64_t RHS, StringRef Context) {
  if (LHS >= RHS) {
    uint64_t Diff = LHS - RHS;
    if (Diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      log() << "hotswap: error: " << Context
            << " positive offset is not representable as int64_t.\n";
      return std::nullopt;
    }
    return static_cast<int64_t>(Diff);
  }

  uint64_t Diff = RHS - LHS;
  constexpr uint64_t Int64MinMagnitude =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
  if (Diff > Int64MinMagnitude) {
    log() << "hotswap: error: " << Context
          << " negative offset is not representable as int64_t.\n";
    return std::nullopt;
  }
  if (Diff == Int64MinMagnitude)
    return std::numeric_limits<int64_t>::min();
  return -static_cast<int64_t>(Diff);
}

static bool appendPaddingTrampoline(std::vector<Trampoline> &Out,
                                    uint64_t PadBytes, ArrayRef<uint8_t> Fill) {
  if (PadBytes == 0)
    return true;
  if (Fill.empty()) {
    log() << "hotswap: error: entry-stub alignment padding requested without "
          << "cached s_nop bytes.\n";
    return false;
  }
  if (PadBytes % Fill.size() != 0) {
    log() << "hotswap: error: entry-stub alignment padding size " << PadBytes
          << " is not a multiple of cached s_nop size " << Fill.size() << ".\n";
    return false;
  }
  if (PadBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    log() << "hotswap: error: entry-stub alignment padding size " << PadBytes
          << " exceeds size_t.\n";
    return false;
  }

  Trampoline Pad;
  while (static_cast<uint64_t>(Pad.Bytes.size()) < PadBytes)
    Pad.Bytes.append(Fill.begin(), Fill.end());
  Out.push_back(std::move(Pad));
  return true;
}

std::optional<uint32_t> appendKernelEntryTrampolines(
    const ElfView &Elf, const LLVMState &LS, unsigned MaxSgprs,
    std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups) {
  std::vector<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
  if (Descriptors.empty())
    return 0;

  struct WorkItem {
    KernelDescriptorInfo KD;
    uint32_t InstPrefLines = 0;
  };

  std::vector<WorkItem> Work;
  uint32_t MaxInstPrefLines = 0;
  for (const KernelDescriptorInfo &KD : Descriptors) {
    std::optional<bool> AlreadyHasEntryStub =
        descriptorAlreadyTargetsEntryStub(Elf, KD, LS);
    if (!AlreadyHasEntryStub)
      return std::nullopt;
    if (*AlreadyHasEntryStub)
      continue;
    std::optional<uint32_t> InstPrefLines =
        Elf.getKernelDescriptorInstPrefSize(KD.KernelName, LS.Cpu);
    if (!InstPrefLines)
      return std::nullopt;
    uint32_t StubInstPrefLines =
        std::min(*InstPrefLines, KernelEntryStubInstPrefLines);
    MaxInstPrefLines = std::max(MaxInstPrefLines, StubInstPrefLines);
    Work.push_back({KD, StubInstPrefLines});
  }
  if (Work.empty())
    return 0;

  if (KernelEntryScratchSgpr + 1 >= MaxSgprs) {
    log() << "hotswap: error: entry trampoline: fixed scratch SGPR pair s["
          << KernelEntryScratchSgpr << ":" << KernelEntryScratchSgpr + 1
          << "] does not fit below max " << MaxSgprs << ".\n";
    return std::nullopt;
  }

  KernelEntryTrampolineBuilder Builder(LS);
  const unsigned ScratchSgprs[] = {KernelEntryScratchSgpr};
  if (!Builder.prepare(ScratchSgprs))
    return std::nullopt;

  std::optional<uint64_t> ExistingGrowthBytes = totalTrampolineBytes(Growth);
  if (!ExistingGrowthBytes)
    return std::nullopt;
  uint64_t AppendOffset = *ExistingGrowthBytes;
  std::optional<uint64_t> TextEndVAddr =
      checkedAdd(Elf.textAddr(), Elf.textSize(), "entry trampoline text end");
  if (!TextEndVAddr)
    return std::nullopt;
  std::optional<uint64_t> StubPoolBaseVAddr = checkedAdd(
      *TextEndVAddr, AppendOffset, "entry trampoline stub-pool base");
  if (!StubPoolBaseVAddr)
    return std::nullopt;
  std::optional<uint64_t> AlignedStubPoolBaseVAddr =
      checkedAlignTo(*StubPoolBaseVAddr, KernelEntryStubStride,
                     "entry trampoline aligned stub-pool base");
  if (!AlignedStubPoolBaseVAddr)
    return std::nullopt;
  const uint64_t StubStart = *AlignedStubPoolBaseVAddr - *TextEndVAddr;
  std::vector<Trampoline> LocalGrowth;
  std::vector<KernelEntryTrampolineFixup> LocalFixups;
  if (!appendPaddingTrampoline(LocalGrowth, StubStart - AppendOffset,
                               LS.SNopBytes))
    return std::nullopt;
  AppendOffset = StubStart;

  for (const WorkItem &Item : Work) {
    const KernelDescriptorInfo &KD = Item.KD;
    std::optional<uint64_t> StubTextEnd = checkedAdd(
        Elf.textSize(), AppendOffset,
        (Twine("entry trampoline append offset for '") + KD.KernelName + "'")
            .str());
    if (!StubTextEnd)
      return std::nullopt;
    std::optional<uint64_t> StubVAddr = checkedAdd(
        Elf.textAddr(), *StubTextEnd,
        (Twine("entry trampoline vaddr for '") + KD.KernelName + "'").str());
    if (!StubVAddr)
      return std::nullopt;
    std::optional<uint64_t> Entry = entryVAddr(KD);
    if (!Entry)
      return std::nullopt;
    SmallVector<uint8_t> Stub =
        Builder.build(*StubVAddr, *Entry, KernelEntryScratchSgpr);
    if (Stub.empty()) {
      log() << "hotswap: error: failed to build kernel-entry trampoline for '"
            << KD.KernelName << "' at original entry vaddr 0x"
            << utohexstr(*Entry) << ".\n";
      return std::nullopt;
    }

    Trampoline T;
    T.Bytes.assign(Stub.begin(), Stub.end());
    LocalGrowth.push_back(std::move(T));
    LocalFixups.push_back({KD.KernelName, AppendOffset, Item.InstPrefLines});
    std::optional<uint64_t> NewAppendOffset = checkedAdd(
        AppendOffset, KernelEntryStubStride,
        (Twine("entry trampoline append offset after '") + KD.KernelName + "'")
            .str());
    if (!NewAppendOffset)
      return std::nullopt;
    AppendOffset = *NewAppendOffset;
  }

  const uint64_t GuardBytes =
      computeKernelEntryPrefetchGuardBytes(MaxInstPrefLines);
  if (GuardBytes != 0) {
    SmallVector<uint8_t> CodeEnd = getCodeEndBytes(LS);
    if (CodeEnd.empty() ||
        !appendPaddingTrampoline(LocalGrowth, GuardBytes, CodeEnd))
      return std::nullopt;
  }

  if (LocalFixups.empty())
    return 0;

  if (LocalFixups.size() > std::numeric_limits<uint32_t>::max()) {
    log() << "hotswap: error: kernel-entry trampoline count "
          << LocalFixups.size() << " exceeds uint32_t.\n";
    return std::nullopt;
  }

  for (Trampoline &T : LocalGrowth)
    Growth.push_back(std::move(T));
  OutFixups.insert(OutFixups.end(), LocalFixups.begin(), LocalFixups.end());

  log() << "hotswap: installed " << LocalFixups.size()
        << " kernel-entry trampoline" << (LocalFixups.size() == 1 ? "" : "s")
        << " with " << GuardBytes << " prefetch guard bytes\n";
  return static_cast<uint32_t>(LocalFixups.size());
}

bool rewriteKernelEntryDescriptorOffsets(
    WritableMemoryBuffer &OutBuf, uint64_t OldTextSize, StringRef TargetCpu,
    ArrayRef<KernelEntryTrampolineFixup> Fixups) {
  if (Fixups.empty())
    return true;

  uint8_t *Data = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart());
  Expected<ElfView> ViewOrErr = ElfView::create(Data, OutBuf.getBufferSize());
  if (!ViewOrErr) {
    log() << "hotswap: error: failed to reparse grown ELF for entry "
          << "descriptor rewrites: " << toString(ViewOrErr.takeError()) << "\n";
    return false;
  }

  bool Ok = true;
  ElfView &OutElf = *ViewOrErr;
  for (const KernelEntryTrampolineFixup &Fixup : Fixups) {
    std::optional<uint64_t> KdVAddr =
        OutElf.getKernelDescriptorVAddr(Fixup.KernelName);
    if (!KdVAddr) {
      log() << "hotswap: error: missing kernel descriptor for entry "
            << "trampoline fixup '" << Fixup.KernelName << "'.\n";
      Ok = false;
      continue;
    }
    std::optional<uint64_t> StubTextOffset = checkedAdd(
        OldTextSize, Fixup.StubTextOffset,
        (Twine("entry trampoline text offset for '") + Fixup.KernelName + "'")
            .str());
    if (!StubTextOffset) {
      Ok = false;
      continue;
    }
    std::optional<uint64_t> StubVAddr = checkedAdd(
        OutElf.textAddr(), *StubTextOffset,
        (Twine("entry trampoline vaddr for '") + Fixup.KernelName + "'").str());
    if (!StubVAddr) {
      Ok = false;
      continue;
    }
    std::optional<int64_t> NewOffset = checkedSignedDifference(
        *StubVAddr, *KdVAddr,
        (Twine("entry trampoline descriptor offset for '") + Fixup.KernelName +
         "'")
            .str());
    if (!NewOffset) {
      Ok = false;
      continue;
    }
    bool UpdatedEntry =
        OutElf.updateKernelDescriptorEntryOffset(Fixup.KernelName, *NewOffset);
    bool UpdatedInstPref = OutElf.updateKernelDescriptorInstPrefSize(
        Fixup.KernelName, TargetCpu, Fixup.InstPrefLines);
    Ok = UpdatedEntry && UpdatedInstPref && Ok;
  }
  return Ok;
}

} // namespace hotswap
} // namespace COMGR
