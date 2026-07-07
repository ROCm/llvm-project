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

static bool appendAsm(SmallVectorImpl<uint8_t> &Out, StringRef Asm,
                      const LLVMState &LS) {
  SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: failed to assemble entry-stub instruction: "
          << Asm << "\n";
    return false;
  }
  Out.append(Bytes.begin(), Bytes.end());
  return true;
}

static SmallVector<uint8_t> getCodeEndBytes(const LLVMState &LS) {
  SmallVector<uint8_t> CodeEnd = assembleSingleInst("s_code_end", LS);
  if (CodeEnd.empty())
    log() << "hotswap: error: failed to assemble s_code_end for entry-stub "
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

SmallVector<uint8_t> buildKernelEntryTrampoline(uint64_t StubVAddr,
                                                uint64_t EntryVAddr,
                                                unsigned ScratchSgpr,
                                                const LLVMState &LS) {
  if (ScratchSgpr == std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: kernel-entry stub scratch SGPR pair overflows "
          << "unsigned.\n";
    return {};
  }

  SmallVector<uint8_t> Bytes;
  std::string ScratchPair =
      (Twine("s[") + Twine(ScratchSgpr) + ":" + Twine(ScratchSgpr + 1) + "]")
          .str();
  std::string ScratchLo = (Twine("s") + Twine(ScratchSgpr)).str();
  std::string ScratchHi = (Twine("s") + Twine(ScratchSgpr + 1)).str();

  // Assemble through the MC layer instead of spelling encoded bytes; the LIT
  // test pins the generated stub's disassembly.
  if (!appendAsm(Bytes, "global_wb", LS))
    return {};
  if (!appendAsm(Bytes, "v_nop", LS))
    return {};
  if (!appendAsm(Bytes, "s_get_pc_i64 " + ScratchPair, LS))
    return {};

  // s_get_pc_i64 returns the address of the following s_add_u32 instruction.
  // Materialize the original entry with a 64-bit PC-relative add so the code
  // object can be rewritten before ROCR knows final device addresses.
  std::optional<uint64_t> PcBase =
      checkedAdd(StubVAddr, static_cast<uint64_t>(Bytes.size()),
                 "kernel-entry stub PC base");
  if (!PcBase)
    return {};
  // Unsigned subtraction is intentional: the immediate pair materializes the
  // 64-bit two's-complement delta, including backward jumps.
  const uint64_t Delta = EntryVAddr - *PcBase;
  const uint32_t Lo = static_cast<uint32_t>(Delta);
  const uint32_t Hi = static_cast<uint32_t>(Delta >> 32);

  if (!appendAsm(Bytes,
                 "s_add_u32 " + ScratchLo + ", " + ScratchLo + ", 0x" +
                     utohexstr(Lo),
                 LS))
    return {};
  if (!appendAsm(Bytes,
                 "s_addc_u32 " + ScratchHi + ", " + ScratchHi + ", 0x" +
                     utohexstr(Hi),
                 LS))
    return {};
  if (!appendAsm(Bytes, "s_set_pc_i64 " + ScratchPair, LS))
    return {};

  SmallVector<uint8_t> CodeEnd = getCodeEndBytes(LS);
  if (CodeEnd.empty())
    return {};
  if (Bytes.size() > KernelEntryStubStride) {
    log() << "hotswap: error: kernel-entry stub grew past "
          << KernelEntryStubStride << " bytes.\n";
    return {};
  }
  while (Bytes.size() < KernelEntryStubStride) {
    if (Bytes.size() + CodeEnd.size() > KernelEntryStubStride) {
      log() << "hotswap: error: s_code_end padding does not evenly fill "
            << "kernel-entry stub stride " << KernelEntryStubStride << ".\n";
      return {};
    }
    Bytes.append(CodeEnd.begin(), CodeEnd.end());
  }
  return Bytes;
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
  SmallVector<uint8_t> Prefix;
  if (!appendAsm(Prefix, "global_wb", LS))
    return false;
  if (!appendAsm(Prefix, "v_nop", LS))
    return false;

  return Bytes.size() >= Prefix.size() &&
         std::equal(Prefix.begin(), Prefix.end(), Bytes.begin());
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

static std::optional<ArrayRef<uint8_t>>
findExecutableBytesAtVAddr(const ElfView &Elf, uint64_t VAddr, uint64_t Size) {
  for (const ElfView::ELFT::Shdr &Shdr : Elf.sections()) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) ||
        !(Shdr.sh_flags & ELF::SHF_EXECINSTR) ||
        Shdr.sh_type == ELF::SHT_NOBITS)
      continue;
    if (VAddr < Shdr.sh_addr)
      continue;
    uint64_t Delta = VAddr - Shdr.sh_addr;
    if (Delta > Shdr.sh_size || Size > Shdr.sh_size - Delta)
      continue;
    if (Shdr.sh_offset > Elf.size() || Delta > Elf.size() - Shdr.sh_offset ||
        Size > Elf.size() - Shdr.sh_offset - Delta) {
      log() << "hotswap: error: executable section bytes for vaddr 0x"
            << utohexstr(VAddr) << " extend past the ELF buffer.\n";
      return std::nullopt;
    }
    return ArrayRef<uint8_t>(Elf.data() + Shdr.sh_offset + Delta, Size);
  }
  return std::nullopt;
}

static std::optional<bool> descriptorAlreadyTargetsEntryStub(
    const ElfView &Elf, const KernelDescriptorInfo &KD, const LLVMState &LS) {
  std::optional<uint64_t> Entry = entryVAddr(KD);
  if (!Entry)
    return std::nullopt;

  std::optional<uint64_t> TextEnd =
      checkedAdd(Elf.textAddr(), Elf.textSize(), "entry trampoline text end");
  if (!TextEnd)
    return std::nullopt;

  std::optional<ArrayRef<uint8_t>> Candidate =
      findExecutableBytesAtVAddr(Elf, *Entry, KernelEntryStubStride);
  if (!Candidate)
    return false;

  // The full idempotency matcher uses LLVM's AMDGPU disassembler. Avoid
  // running it over arbitrary original kernel entry bytes; real code objects
  // can contain byte streams that are valid executable code but still trip
  // decoder corner cases before COMGR can finish rewriting.
  if (!hasKernelEntryTrampolinePrefix(*Candidate, LS))
    return false;

  std::vector<InternalDecodedInst> Decoded;
  if (!decodeKernelEntryStub(*Candidate, LS, Decoded,
                             "entry trampoline idempotency matcher"))
    return false;
  if (!hasEntryStubOperandShape(Decoded, LS))
    return false;

  std::optional<uint64_t> Target = decodeEntryStubTargetVAddr(Decoded, *Entry);
  if (!Target)
    return std::nullopt;

  return *Target >= Elf.textAddr() && *Target < *TextEnd && *Target < *Entry;
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

static std::optional<unsigned> allocateEntryStubScratchSgprs(
    const ElfView &Elf, const KernelDescriptorInfo &KD, unsigned MaxSgprs) {
  constexpr unsigned ScratchSgprs = 2;
  std::optional<unsigned> SgprCount = Elf.getKernelSgprCount(KD.KernelName);
  if (!SgprCount) {
    log() << "hotswap: error: entry trampoline: failed to read SGPR count for '"
          << KD.KernelName << "'.\n";
    return std::nullopt;
  }
  if (*SgprCount > MaxSgprs) {
    log() << "hotswap: error: entry trampoline: kernel '" << KD.KernelName
          << "' uses " << *SgprCount << " SGPRs, above max " << MaxSgprs
          << ".\n";
    return std::nullopt;
  }

  unsigned ScratchBase = (*SgprCount + 1) & ~1u;
  if (ScratchBase > MaxSgprs || MaxSgprs - ScratchBase < ScratchSgprs) {
    log() << "hotswap: error: entry trampoline: kernel '" << KD.KernelName
          << "' uses " << *SgprCount << " SGPRs; no aligned scratch pair fits "
          << "below max " << MaxSgprs << ".\n";
    return std::nullopt;
  }
  return ScratchBase;
}

static bool appendFillBytes(SmallVectorImpl<uint8_t> &Out, uint64_t FillBytes,
                            ArrayRef<uint8_t> Fill, StringRef Context) {
  if (FillBytes < Out.size()) {
    log() << "hotswap: error: " << Context << " target size " << FillBytes
          << " is smaller than existing size " << Out.size() << ".\n";
    return false;
  }
  uint64_t Needed = FillBytes - Out.size();
  if (Needed == 0)
    return true;
  if (Fill.empty()) {
    log() << "hotswap: error: " << Context
          << " requested without fill bytes.\n";
    return false;
  }
  if (Needed % Fill.size() != 0) {
    log() << "hotswap: error: " << Context << " size " << Needed
          << " is not a multiple of fill size " << Fill.size() << ".\n";
    return false;
  }
  if (Needed > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    log() << "hotswap: error: " << Context << " size " << Needed
          << " exceeds size_t.\n";
    return false;
  }

  while (static_cast<uint64_t>(Out.size()) < FillBytes)
    Out.append(Fill.begin(), Fill.end());
  return true;
}

std::optional<uint32_t> appendKernelEntryTrampolines(
    const ElfView &Elf, const LLVMState &LS, unsigned MaxSgprs,
    uint64_t StubBaseVAddr, SmallVectorImpl<uint8_t> &EntryBytes,
    std::vector<KernelEntryTrampolineFixup> &OutFixups) {
  std::vector<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
  if (Descriptors.empty())
    return 0;

  if (StubBaseVAddr % KernelEntryStubStride != 0) {
    log() << "hotswap: error: entry trampoline stub base vaddr 0x"
          << utohexstr(StubBaseVAddr) << " is not aligned to "
          << KernelEntryStubStride << " bytes.\n";
    return std::nullopt;
  }

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

  std::vector<KernelEntryTrampolineFixup> LocalFixups;
  SmallVector<uint8_t> LocalBytes;
  uint64_t AppendOffset = 0;

  for (const WorkItem &Item : Work) {
    const KernelDescriptorInfo &KD = Item.KD;
    std::optional<uint64_t> StubVAddr = checkedAdd(
        StubBaseVAddr, AppendOffset,
        (Twine("entry trampoline vaddr for '") + KD.KernelName + "'").str());
    if (!StubVAddr)
      return std::nullopt;
    std::optional<unsigned> ScratchSgpr =
        allocateEntryStubScratchSgprs(Elf, KD, MaxSgprs);
    if (!ScratchSgpr)
      return std::nullopt;
    std::optional<uint64_t> Entry = entryVAddr(KD);
    if (!Entry)
      return std::nullopt;
    SmallVector<uint8_t> Stub =
        buildKernelEntryTrampoline(*StubVAddr, *Entry, *ScratchSgpr, LS);
    if (Stub.empty()) {
      log() << "hotswap: error: failed to build kernel-entry trampoline for '"
            << KD.KernelName << "' at original entry vaddr 0x"
            << utohexstr(*Entry) << ".\n";
      return std::nullopt;
    }

    if (Stub.size() != KernelEntryStubStride) {
      log() << "hotswap: error: kernel-entry trampoline for '" << KD.KernelName
            << "' has size " << Stub.size() << ", expected "
            << KernelEntryStubStride << ".\n";
      return std::nullopt;
    }

    LocalBytes.append(Stub.begin(), Stub.end());
    LocalFixups.push_back(
        {KD.KernelName, *StubVAddr, *ScratchSgpr + 2, Item.InstPrefLines});
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
        !appendFillBytes(LocalBytes, AppendOffset + GuardBytes, CodeEnd,
                         "entry-stub prefetch guard"))
      return std::nullopt;
  }

  if (LocalFixups.empty())
    return 0;

  if (LocalFixups.size() > std::numeric_limits<uint32_t>::max()) {
    log() << "hotswap: error: kernel-entry trampoline count "
          << LocalFixups.size() << " exceeds uint32_t.\n";
    return std::nullopt;
  }

  EntryBytes.append(LocalBytes.begin(), LocalBytes.end());
  OutFixups.insert(OutFixups.end(), LocalFixups.begin(), LocalFixups.end());

  log() << "hotswap: installed " << LocalFixups.size()
        << " kernel-entry trampoline" << (LocalFixups.size() == 1 ? "" : "s")
        << " with " << GuardBytes << " prefetch guard bytes\n";
  return static_cast<uint32_t>(LocalFixups.size());
}

bool rewriteKernelEntryDescriptorOffsets(
    WritableMemoryBuffer &OutBuf, StringRef TargetCpu,
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
    std::optional<int64_t> NewOffset = checkedSignedDifference(
        Fixup.StubVAddr, *KdVAddr,
        (Twine("entry trampoline descriptor offset for '") + Fixup.KernelName +
         "'")
            .str());
    if (!NewOffset) {
      Ok = false;
      continue;
    }
    bool UpdatedEntry =
        OutElf.updateKernelDescriptorEntryOffset(Fixup.KernelName, *NewOffset);
    bool UpdatedSgprs = OutElf.updateKernelDescriptorSgprCount(
        Fixup.KernelName, Fixup.RequiredSgprs);
    bool UpdatedInstPref = OutElf.updateKernelDescriptorInstPrefSize(
        Fixup.KernelName, TargetCpu, Fixup.InstPrefLines);
    Ok = UpdatedEntry && UpdatedSgprs && UpdatedInstPref && Ok;
  }
  return Ok;
}

} // namespace hotswap
} // namespace COMGR
