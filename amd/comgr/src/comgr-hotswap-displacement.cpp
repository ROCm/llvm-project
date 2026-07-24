//===- comgr-hotswap-displacement.cpp - HotSwap text displacement --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Direct `.text` displacement for HotSwap rewrites. This layer inserts
/// larger replacement sequences into `.text`, shifts the displaced code
/// forward, repairs direct PC-relative scalar branches when it can prove the
/// new encoding, and updates ELF metadata. Appended trampolines remain the
/// fallback when any check fails.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/Support/Alignment.h"

#include <algorithm>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

using Ehdr = ELF::Elf64_Ehdr;
using Shdr = ELF::Elf64_Shdr;
using Phdr = ELF::Elf64_Phdr;
using ELFT = ElfView::ELFT;
using ELFFileT = ElfView::ELFFileT;

namespace {

Error makeDisplacementError(const Twine &Msg) {
  std::string Message = Msg.str();
  log() << "hotswap: displacement unavailable: " << Message << "\n";
  return createStringError(object::object_error::parse_failed, Message);
}

Expected<const ELFT::Shdr *> findUniqueAllocatedSection(const ElfView &Elf,
                                                        uint64_t Address,
                                                        bool RequireExecutable,
                                                        StringRef Context) {
  const ELFT::Shdr *Owner = nullptr;
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) || Shdr.sh_size == 0 ||
        Address < Shdr.sh_addr || Address - Shdr.sh_addr >= Shdr.sh_size)
      continue;
    if (Owner) {
      return makeDisplacementError(Twine(Context) +
                                   " is covered by overlapping allocated "
                                   "sections");
    }
    Owner = &Shdr;
  }
  if (!Owner) {
    return makeDisplacementError(Twine(Context) +
                                 " is outside every allocated section");
  }
  if (RequireExecutable && !(Owner->sh_flags & ELF::SHF_EXECINSTR)) {
    return makeDisplacementError(Twine(Context) +
                                 " is not in an executable section");
  }
  return Owner;
}

Expected<uint64_t> remapAllocatedAddress(const ElfView &Elf,
                                         const DisplacementPlan &Plan,
                                         uint64_t OldAddress,
                                         bool RequireExecutable,
                                         StringRef Context) {
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr()) {
    return makeDisplacementError(Twine(Context) +
                                 " cannot resolve an overflowing .text");
  }
  const uint64_t OldTextEnd = Elf.textAddr() + Elf.textSize();
  if (OldAddress >= Elf.textAddr() && OldAddress < OldTextEnd) {
    if (RequireExecutable &&
        !(Elf.textSection()->sh_flags & ELF::SHF_EXECINSTR)) {
      return makeDisplacementError(Twine(Context) +
                                   " is not in an executable section");
    }
    uint64_t NewOffset = 0;
    if (!Plan.mapOffset(OldAddress - Elf.textAddr(),
                        DisplacementMapBias::BeforeInsertedBytes, NewOffset)) {
      return makeDisplacementError(Twine(Context) +
                                   " maps inside a replaced instruction");
    }
    if (NewOffset > std::numeric_limits<uint64_t>::max() - Elf.textAddr()) {
      return makeDisplacementError(Twine(Context) +
                                   " overflows after .text displacement");
    }
    return Elf.textAddr() + NewOffset;
  }

  Expected<const ELFT::Shdr *> OwnerOrErr =
      findUniqueAllocatedSection(Elf, OldAddress, RequireExecutable, Context);
  if (!OwnerOrErr)
    return OwnerOrErr.takeError();

  // The current displacement mode changes non-.text file offsets but preserves
  // their virtual addresses. Whole-object relocation extends this mapping in a
  // later layer.
  return OldAddress;
}

Expected<uint64_t> requiredGrowthAlignment(const ElfView &Elf) {
  uint64_t MaxAlign = 1;
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (Shdr.sh_offset <= Elf.textOffset())
      continue;
    MaxAlign = std::max<uint64_t>(MaxAlign, Shdr.sh_addralign);
  }

  Expected<ELFT::PhdrRange> PhdrsOrErr = Elf.file().program_headers();
  if (!PhdrsOrErr) {
    return makeDisplacementError(
        "failed to read program headers while computing displacement "
        "alignment: " +
        Twine(toString(PhdrsOrErr.takeError())));
  }
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_offset <= Elf.textOffset())
      continue;
    MaxAlign = std::max<uint64_t>(MaxAlign, Phdr.p_align);
  }
  return std::max<uint64_t>(MaxAlign, 1);
}

Error validateDebugSections(const ElfView &Elf) {
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    Expected<StringRef> NameOrErr = Elf.file().getSectionName(Shdr);
    if (!NameOrErr) {
      return makeDisplacementError(
          "failed to read section name while checking debug info: " +
          Twine(toString(NameOrErr.takeError())));
    }
    StringRef Name = *NameOrErr;
    if (Name.starts_with(".debug") || Name.starts_with(".zdebug") ||
        Name == ".eh_frame" || Name == ".eh_frame_hdr") {
      return makeDisplacementError("debug/unwind section '" + Twine(Name) +
                                   "' requires address remapping");
    }
  }
  return Error::success();
}

bool isPcSensitiveForDisplacement(const InternalDecodedInst &DI,
                                  const LLVMState &LS) {
  unsigned Opcode = DI.Inst.getOpcode();
  if (Opcode == LS.SAddPcI64Opcode || Opcode == LS.SGetPcI64Opcode ||
      Opcode == LS.SSetPcI64Opcode || Opcode == LS.SSwapPcI64Opcode ||
      Opcode == LS.SPrefetchInstPcRelOpcode ||
      Opcode == LS.SPrefetchDataPcRelOpcode)
    return true;

  const MCInstrDesc &Desc = LS.MCII->get(Opcode);
  for (const MCOperandInfo &Operand : Desc.operands())
    if (Operand.OperandType == MCOI::OPERAND_PCREL)
      return !LS.MIA->isBranch(DI.Inst);
  return false;
}

Error validateVirtualGrowth(const ElfView &Elf, uint64_t PaddedGrowth) {
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr() ||
      PaddedGrowth > std::numeric_limits<uint64_t>::max() - Elf.textAddr() -
                         Elf.textSize())
    return makeDisplacementError("displaced .text virtual end overflows");
  const uint64_t OldTextEnd = Elf.textAddr() + Elf.textSize();
  const uint64_t NewTextEnd = OldTextEnd + PaddedGrowth;

  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) || &Shdr == Elf.textSection() ||
        Shdr.sh_size == 0)
      continue;
    if (Shdr.sh_addr >= OldTextEnd && Shdr.sh_addr < NewTextEnd) {
      return makeDisplacementError(
          "displaced .text would overlap a later allocatable section");
    }
  }

  Expected<ELFT::PhdrRange> PhdrsOrErr = Elf.file().program_headers();
  if (!PhdrsOrErr) {
    return makeDisplacementError(
        "failed to read program headers while validating displacement: " +
        Twine(toString(PhdrsOrErr.takeError())));
  }
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_filesz > std::numeric_limits<uint64_t>::max() - Phdr.p_offset)
      return makeDisplacementError("program-header file range overflows");
    const uint64_t SegmentFileEnd = Phdr.p_offset + Phdr.p_filesz;
    if (Phdr.p_type == ELF::PT_LOAD && Phdr.p_offset <= Elf.textOffset() &&
        SegmentFileEnd >= Elf.textOffset() + Elf.textSize() &&
        SegmentFileEnd != Elf.textOffset() + Elf.textSize()) {
      return makeDisplacementError(
          ".text is not the last file-backed content in its PT_LOAD segment");
    }

    if (Phdr.p_type != ELF::PT_LOAD || Phdr.p_memsz == 0)
      continue;
    if (Phdr.p_vaddr >= OldTextEnd && Phdr.p_vaddr < NewTextEnd) {
      return makeDisplacementError(
          "displaced .text would overlap a later PT_LOAD segment");
    }
  }
  return Error::success();
}

Error validateTextRelocations(const ElfView &Elf) {
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr()) {
    return makeDisplacementError(
        ".text virtual range overflows while checking relocations");
  }
  const uint64_t TextEnd = Elf.textAddr() + Elf.textSize();

  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (Shdr.sh_type != ELF::SHT_REL && Shdr.sh_type != ELF::SHT_RELA)
      continue;

    if (Shdr.sh_info == Elf.textSectionIndex()) {
      Expected<StringRef> NameOrErr = Elf.file().getSectionName(Shdr);
      if (!NameOrErr) {
        return makeDisplacementError(
            "failed to read relocation section name: " +
            Twine(toString(NameOrErr.takeError())));
      }
      return makeDisplacementError("relocation section '" + Twine(*NameOrErr) +
                                   "' references .text");
    }

    // Dynamic relocation sections use sh_info == 0. r_offset identifies the
    // location to update, but the relocation's symbol or addend can separately
    // resolve to displaced code. Reject either direction until displacement
    // can rewrite relocation expressions as well as their destinations.
    if (Shdr.sh_info == 0) {
      if (Shdr.sh_type == ELF::SHT_RELA) {
        Expected<ELFT::RelaRange> RelasOrErr = Elf.file().relas(Shdr);
        if (!RelasOrErr) {
          return makeDisplacementError(
              "failed to read dynamic relocation section: " +
              Twine(toString(RelasOrErr.takeError())));
        }
        Expected<const ELFT::Shdr *> SymtabOrErr =
            Elf.file().getSection(Shdr.sh_link);
        if (!SymtabOrErr) {
          return makeDisplacementError(
              "failed to read dynamic relocation symbol table: " +
              Twine(toString(SymtabOrErr.takeError())));
        }

        for (const ELFT::Rela &Rela : *RelasOrErr) {
          if (Rela.r_offset >= Elf.textAddr() && Rela.r_offset < TextEnd) {
            return makeDisplacementError(
                "dynamic relocation section targets a displaced .text "
                "address");
          }

          if (Rela.r_addend >= 0) {
            const uint64_t Addend = static_cast<uint64_t>(Rela.r_addend);
            if (Addend >= Elf.textAddr() && Addend < TextEnd) {
              return makeDisplacementError(
                  "dynamic relocation addend references a displaced .text "
                  "address");
            }
          }

          Expected<const ELFT::Sym *> SymOrErr =
              Elf.file().getRelocationSymbol(Rela, *SymtabOrErr);
          if (!SymOrErr) {
            return makeDisplacementError(
                "failed to read dynamic relocation symbol: " +
                Twine(toString(SymOrErr.takeError())));
          }
          const ELFT::Sym *Sym = *SymOrErr;
          if (Sym &&
              (Sym->st_shndx == Elf.textSectionIndex() ||
               (Sym->st_value >= Elf.textAddr() && Sym->st_value < TextEnd))) {
            return makeDisplacementError(
                "dynamic relocation symbol references displaced .text");
          }
        }
      } else {
        Expected<ELFT::RelRange> RelsOrErr = Elf.file().rels(Shdr);
        if (!RelsOrErr) {
          return makeDisplacementError(
              "failed to read dynamic relocation section: " +
              Twine(toString(RelsOrErr.takeError())));
        }
        Expected<const ELFT::Shdr *> SymtabOrErr =
            Elf.file().getSection(Shdr.sh_link);
        if (!SymtabOrErr) {
          return makeDisplacementError(
              "failed to read dynamic relocation symbol table: " +
              Twine(toString(SymtabOrErr.takeError())));
        }

        for (const ELFT::Rel &Rel : *RelsOrErr) {
          if (Rel.r_offset >= Elf.textAddr() && Rel.r_offset < TextEnd) {
            return makeDisplacementError(
                "dynamic relocation section targets a displaced .text "
                "address");
          }

          Expected<const ELFT::Sym *> SymOrErr =
              Elf.file().getRelocationSymbol(Rel, *SymtabOrErr);
          if (!SymOrErr) {
            return makeDisplacementError(
                "failed to read dynamic relocation symbol: " +
                Twine(toString(SymOrErr.takeError())));
          }
          const ELFT::Sym *Sym = *SymOrErr;
          if (Sym &&
              (Sym->st_shndx == Elf.textSectionIndex() ||
               (Sym->st_value >= Elf.textAddr() && Sym->st_value < TextEnd))) {
            return makeDisplacementError(
                "dynamic relocation symbol references displaced .text");
          }

          // SHT_REL stores its addend at r_offset. Without interpreting each
          // AMDGPU relocation width and target section, a symbol-free record
          // cannot prove that its implicit addend is independent of .text.
          if (!Sym && Rel.getType(false) != ELF::R_AMDGPU_NONE) {
            return makeDisplacementError(
                "symbol-free dynamic REL relocation has an implicit addend");
          }
        }
      }
    }
  }
  return Error::success();
}

Error validateKernelEntryMappings(const ElfView &Elf,
                                  const DisplacementPlan &Plan) {
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr()) {
    return makeDisplacementError(
        ".text virtual range overflows while checking kernel entries");
  }
  const uint64_t TextEnd = Elf.textAddr() + Elf.textSize();

  for (const KernelDescriptorInfo &KD : Elf.kernelDescriptors()) {
    std::optional<uint64_t> OldEntry = entryVAddr(KD);
    if (!OldEntry) {
      return makeDisplacementError("failed to resolve kernel entry for '" +
                                   Twine(KD.KernelName) + "'");
    }
    if (*OldEntry < Elf.textAddr() || *OldEntry >= TextEnd) {
      return makeDisplacementError(
          "kernel entry for '" + Twine(KD.KernelName) +
          "' is outside .text and may contain an unrepairable entry stub");
    }

    uint64_t NewEntryOffset = 0;
    if (!Plan.mapOffset(*OldEntry - Elf.textAddr(),
                        DisplacementMapBias::BeforeInsertedBytes,
                        NewEntryOffset)) {
      return makeDisplacementError("kernel entry for '" + Twine(KD.KernelName) +
                                   "' maps inside a replaced range");
    }
    if (NewEntryOffset >
        std::numeric_limits<uint64_t>::max() - Elf.textAddr()) {
      return makeDisplacementError("displaced kernel entry for '" +
                                   Twine(KD.KernelName) + "' overflows");
    }
    const uint64_t NewEntry = Elf.textAddr() + NewEntryOffset;
    if (NewEntry % KernelEntryStubStride != 0) {
      return makeDisplacementError("displaced kernel entry for '" +
                                   Twine(KD.KernelName) +
                                   "' is not 256-byte aligned");
    }
  }
  return Error::success();
}

Expected<SmallVector<uint8_t>>
reencodePcrelBranch(const InternalDecodedInst &DI, uint64_t NewFrom,
                    uint64_t NewTarget, const LLVMState &LS) {
  if (DI.Inst.getOpcode() == LS.SBranchOpcode) {
    SmallVector<uint8_t> Encoded = LS.encodeSBranch(NewFrom, NewTarget);
    if (Encoded.empty()) {
      return makeDisplacementError("s_branch at old .text offset 0x" +
                                   Twine::utohexstr(DI.Offset) +
                                   " is out of range after displacement");
    }
    return Encoded;
  }

  if (DI.Inst.getNumOperands() == 0 || !DI.Inst.getOperand(0).isImm()) {
    return makeDisplacementError(
        "branch at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
        " does not expose an immediate target operand");
  }

  int64_t ByteDelta = static_cast<int64_t>(NewTarget) -
                      static_cast<int64_t>(NewFrom) -
                      static_cast<int64_t>(DI.Size);
  if (ByteDelta % MinInstSize != 0) {
    return makeDisplacementError("branch at old .text offset 0x" +
                                 Twine::utohexstr(DI.Offset) +
                                 " has unaligned displacement after rewrite");
  }

  int64_t DwordOffset = ByteDelta / MinInstSize;
  if (DwordOffset < BranchOffsetMin || DwordOffset > BranchOffsetMax) {
    return makeDisplacementError("branch at old .text offset 0x" +
                                 Twine::utohexstr(DI.Offset) +
                                 " is out of simm16 range after displacement");
  }

  MCInst NewInst = DI.Inst;
  NewInst.getOperand(0).setImm(DwordOffset);
  SmallVector<char, 16> Code;
  SmallVector<MCFixup, 4> Fixups;
  LS.MCE->encodeInstruction(NewInst, Code, Fixups, *LS.STI);
  SmallVector<uint8_t> Encoded(Code.begin(), Code.end());
  if (Encoded.size() != DI.Size) {
    return makeDisplacementError("branch at old .text offset 0x" +
                                 Twine::utohexstr(DI.Offset) +
                                 " changed encoded size during re-encode");
  }
  return Encoded;
}

bool getAbsoluteOperand(const MCOperand &Operand, int64_t &Value) {
  if (Operand.isImm()) {
    Value = Operand.getImm();
    return true;
  }
  return Operand.isExpr() && Operand.getExpr()->evaluateAsAbsolute(Value);
}

std::optional<uint64_t> addSignedOffset(uint64_t Base, int64_t Offset,
                                        StringRef Context) {
  if (Offset >= 0)
    return checkedAddUint64(Base, static_cast<uint64_t>(Offset), Context);
  const uint64_t Magnitude = Offset == std::numeric_limits<int64_t>::min()
                                 ? uint64_t{1} << 63
                                 : static_cast<uint64_t>(-Offset);
  return checkedSubUint64(Base, Magnitude, Context);
}

Expected<SmallVector<uint8_t>>
reencodeAbsoluteOperand(const InternalDecodedInst &DI, unsigned OperandIndex,
                        int64_t NewValue, const LLVMState &LS,
                        StringRef Context) {
  if (OperandIndex >= DI.Inst.getNumOperands())
    return makeDisplacementError(Twine(Context) + " has no immediate operand");

  MCInst NewInst = DI.Inst;
  MCOperand &Operand = NewInst.getOperand(OperandIndex);
  if (Operand.isImm()) {
    Operand.setImm(NewValue);
  } else if (Operand.isExpr()) {
    // AMDGPU's disassembler represents forced literals with a backend-private
    // target expression. Build a fresh literal through the public MC assembly
    // path and transplant that structured operand; do not parse printer text or
    // reproduce the target expression locally.
    for (unsigned I = 0, E = NewInst.getNumOperands(); I != E; ++I) {
      if (I != OperandIndex && NewInst.getOperand(I).isExpr()) {
        return makeDisplacementError(Twine(Context) +
                                     " has another expression operand");
      }
    }
    std::string LiteralAssembly =
        ("s_add_pc_i64 lit64(0x" +
         Twine::utohexstr(static_cast<uint64_t>(NewValue)) + ")")
            .str();
    SmallVector<uint8_t> LiteralBytes = assembleSingleInst(LiteralAssembly, LS);
    std::vector<InternalDecodedInst> LiteralDecoded;
    if (LiteralBytes.empty() ||
        !decodeTextSection(LiteralBytes.data(), LiteralBytes.size(), LS,
                           LiteralDecoded, /*WantMnemonic=*/false) ||
        LiteralDecoded.size() != 1 ||
        LiteralDecoded[0].Inst.getNumOperands() != 1 ||
        !LiteralDecoded[0].Inst.getOperand(0).isExpr()) {
      return makeDisplacementError(Twine(Context) +
                                   " could not build an MC literal operand");
    }
    int64_t LiteralValue = 0;
    if (!LiteralDecoded[0].Inst.getOperand(0).getExpr()->evaluateAsAbsolute(
            LiteralValue) ||
        LiteralValue != NewValue) {
      return makeDisplacementError(Twine(Context) +
                                   " rebuilt the wrong MC literal value");
    }
    Operand.setExpr(LiteralDecoded[0].Inst.getOperand(0).getExpr());
  } else {
    return makeDisplacementError(Twine(Context) +
                                 " does not expose an absolute immediate");
  }

  SmallVector<char, 16> Code;
  SmallVector<MCFixup, 4> Fixups;
  LS.MCE->encodeInstruction(NewInst, Code, Fixups, *LS.STI);
  if (!Fixups.empty()) {
    return makeDisplacementError(Twine(Context) +
                                 " produced an unresolved MC fixup");
  }
  if (Code.size() != DI.Size) {
    return makeDisplacementError(Twine(Context) +
                                 " changed encoded size during re-encode");
  }
  return SmallVector<uint8_t>(Code.begin(), Code.end());
}

Expected<bool> repairSGetPcPairForDisplacement(
    const ElfView &Elf, const LLVMState &LS, const DisplacementPlan &Plan,
    const InternalDecodedInst &GetPc, SmallVectorImpl<uint8_t> &NewText) {
  if (GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg()) {
    return false;
  }

  std::optional<uint64_t> AddOldOffOrErr =
      checkedAddUint64(GetPc.Offset, GetPc.Size, "s_get_pc pair offset");
  if (!AddOldOffOrErr || *AddOldOffOrErr >= Elf.textSize())
    return false;
  const uint64_t AddOldOff = *AddOldOffOrErr;

  InternalDecodedInst Add;
  bool GotAdd = false;
  bool DecodeSucceeded = decodeTextSectionStreaming(
      Elf.textData() + AddOldOff, Elf.textSize() - AddOldOff, LS,
      /*WantMnemonic=*/false, [&](const InternalDecodedInst &DI) {
        Add = DI;
        GotAdd = true;
        return false;
      });
  if (!DecodeSucceeded || !GotAdd || !Add.DecodeSucceeded ||
      Add.Inst.getOpcode() != LS.SAddNcU64Opcode ||
      Add.Inst.getNumOperands() != 3 || !Add.Inst.getOperand(0).isReg() ||
      !Add.Inst.getOperand(1).isReg() ||
      Add.Inst.getOperand(0).getReg() != GetPc.Inst.getOperand(0).getReg() ||
      Add.Inst.getOperand(1).getReg() != GetPc.Inst.getOperand(0).getReg() ||
      Plan.rangeOverlapsReplacement(AddOldOff, Add.Size)) {
    return false;
  }

  int64_t OldAddend = 0;
  if (!getAbsoluteOperand(Add.Inst.getOperand(2), OldAddend))
    return false;

  std::optional<uint64_t> OldPcBase =
      checkedAddUint64(Elf.textAddr(), AddOldOff, "s_get_pc old PC base");
  if (!OldPcBase)
    return false;
  std::optional<uint64_t> OldTarget =
      addSignedOffset(*OldPcBase, OldAddend, "s_get_pc old target");
  if (!OldTarget)
    return false;

  Expected<uint64_t> NewTargetOrErr = remapAllocatedAddress(
      Elf, Plan, *OldTarget,
      /*RequireExecutable=*/false, "s_get_pc materialized target");
  if (!NewTargetOrErr) {
    consumeError(NewTargetOrErr.takeError());
    return false;
  }

  uint64_t NewAddOff = 0;
  if (!Plan.mapOffset(AddOldOff, DisplacementMapBias::AfterInsertedBytes,
                      NewAddOff)) {
    return false;
  }
  std::optional<uint64_t> NewPcBase =
      checkedAddUint64(Elf.textAddr(), NewAddOff, "s_get_pc displaced PC base");
  if (!NewPcBase)
    return false;
  std::optional<int64_t> NewAddendOrErr = checkedSignedDifference(
      *NewTargetOrErr, *NewPcBase, "s_get_pc displaced immediate");
  if (!NewAddendOrErr)
    return false;
  const int64_t NewAddend = *NewAddendOrErr;
  if (NewAddend == OldAddend)
    return true;

  std::string Context = ("s_add for s_get_pc at old .text offset 0x" +
                         Twine::utohexstr(GetPc.Offset))
                            .str();
  Expected<SmallVector<uint8_t>> CodeOrErr =
      reencodeAbsoluteOperand(Add, /*OperandIndex=*/2, NewAddend, LS, Context);
  if (!CodeOrErr)
    return CodeOrErr.takeError();
  SmallVector<uint8_t> &Code = *CodeOrErr;
  if (NewAddOff > NewText.size() || Code.size() > NewText.size() - NewAddOff) {
    return makeDisplacementError(Context + " writes past rebuilt .text");
  }
  std::memcpy(NewText.data() + NewAddOff, Code.data(), Code.size());
  return true;
}

Expected<bool> repairSAddPcForDisplacement(const ElfView &Elf,
                                           const LLVMState &LS,
                                           const DisplacementPlan &Plan,
                                           const InternalDecodedInst &AddPc,
                                           SmallVectorImpl<uint8_t> &NewText) {
  if (AddPc.Inst.getNumOperands() != 1)
    return false;
  int64_t OldDelta = 0;
  if (!getAbsoluteOperand(AddPc.Inst.getOperand(0), OldDelta))
    return false;

  std::optional<uint64_t> OldPcOffset =
      checkedAddUint64(AddPc.Offset, AddPc.Size, "s_add_pc old PC offset");
  if (!OldPcOffset)
    return false;
  std::optional<uint64_t> OldPcBase =
      checkedAddUint64(Elf.textAddr(), *OldPcOffset, "s_add_pc old PC base");
  if (!OldPcBase)
    return false;
  std::optional<uint64_t> OldTarget =
      addSignedOffset(*OldPcBase, OldDelta, "s_add_pc old target");
  if (!OldTarget || *OldTarget < Elf.textAddr() ||
      *OldTarget - Elf.textAddr() >= Elf.textSize()) {
    return false;
  }

  uint64_t NewAddPcOff = 0;
  if (!Plan.mapOffset(AddPc.Offset, DisplacementMapBias::AfterInsertedBytes,
                      NewAddPcOff)) {
    return false;
  }
  Expected<uint64_t> NewTargetOrErr =
      remapAllocatedAddress(Elf, Plan, *OldTarget,
                            /*RequireExecutable=*/true, "s_add_pc target");
  if (!NewTargetOrErr) {
    consumeError(NewTargetOrErr.takeError());
    return false;
  }
  std::optional<uint64_t> NewPcOffset =
      checkedAddUint64(NewAddPcOff, AddPc.Size, "s_add_pc displaced PC offset");
  if (!NewPcOffset)
    return false;
  std::optional<uint64_t> NewPcBase = checkedAddUint64(
      Elf.textAddr(), *NewPcOffset, "s_add_pc displaced PC base");
  if (!NewPcBase)
    return false;
  std::optional<int64_t> NewDeltaOrErr = checkedSignedDifference(
      *NewTargetOrErr, *NewPcBase, "s_add_pc displaced immediate");
  if (!NewDeltaOrErr)
    return false;
  const int64_t NewDelta = *NewDeltaOrErr;
  if (NewDelta == OldDelta)
    return true;

  std::string Context =
      ("s_add_pc_i64 at old .text offset 0x" + Twine::utohexstr(AddPc.Offset))
          .str();
  Expected<SmallVector<uint8_t>> CodeOrErr =
      reencodeAbsoluteOperand(AddPc, /*OperandIndex=*/0, NewDelta, LS, Context);
  if (!CodeOrErr)
    return CodeOrErr.takeError();
  SmallVector<uint8_t> &Code = *CodeOrErr;
  if (NewAddPcOff > NewText.size() ||
      Code.size() > NewText.size() - NewAddPcOff) {
    return makeDisplacementError(Context + " writes past rebuilt .text");
  }
  std::memcpy(NewText.data() + NewAddPcOff, Code.data(), Code.size());
  return true;
}

Error repairBranches(const ElfView &Elf, const LLVMState &LS,
                     const DisplacementPlan &Plan,
                     SmallVectorImpl<uint8_t> &NewText) {
  if (!LS.MIA) {
    return makeDisplacementError(
        "branch analysis through LLVM MC is unavailable");
  }

  std::optional<Error> RepairError;
  bool Decoded = decodeTextSectionStreaming(
      Elf.textData(), Elf.textSize(), LS, /*WantMnemonic=*/false,
      [&](const InternalDecodedInst &DI) {
        if (Plan.rangeOverlapsReplacement(DI.Offset, DI.Size))
          return true;
        if (!DI.DecodeSucceeded) {
          RepairError.emplace(makeDisplacementError(
              "undecodable instruction at old .text offset 0x" +
              Twine::utohexstr(DI.Offset)));
          return false;
        }

        const MCInst &Inst = DI.Inst;
        if (Inst.getOpcode() == LS.SGetPcI64Opcode) {
          Expected<bool> RepairedOrErr =
              repairSGetPcPairForDisplacement(Elf, LS, Plan, DI, NewText);
          if (!RepairedOrErr) {
            RepairError.emplace(RepairedOrErr.takeError());
            return false;
          }
          if (*RepairedOrErr)
            return true;
        }
        if (Inst.getOpcode() == LS.SAddPcI64Opcode) {
          Expected<bool> RepairedOrErr =
              repairSAddPcForDisplacement(Elf, LS, Plan, DI, NewText);
          if (!RepairedOrErr) {
            RepairError.emplace(RepairedOrErr.takeError());
            return false;
          }
          if (*RepairedOrErr)
            return true;
        }
        if (isPcSensitiveForDisplacement(DI, LS)) {
          StringRef Mnemonic = "<unknown>";
          if (LS.MCIP) {
            std::pair<const char *, uint64_t> Name =
                LS.MCIP->getMnemonic(DI.Inst);
            if (Name.first)
              Mnemonic = StringRef(Name.first).rtrim();
          }
          RepairError.emplace(makeDisplacementError(
              "pc-sensitive instruction '" + Twine(Mnemonic) +
              "' at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
              " requires linked-address repair"));
          return false;
        }
        if (LS.MIA->isCall(Inst)) {
          RepairError.emplace(makeDisplacementError(
              "call at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
              " is not supported by displacement"));
          return false;
        }
        if (LS.MIA->isIndirectBranch(Inst)) {
          RepairError.emplace(
              makeDisplacementError("indirect branch at old .text offset 0x" +
                                    Twine::utohexstr(DI.Offset) +
                                    " is not supported by displacement"));
          return false;
        }
        if (!LS.MIA->isBranch(Inst))
          return true;

        uint64_t OldTarget = 0;
        if (!LS.MIA->evaluateBranch(Inst, DI.Offset, DI.Size, OldTarget)) {
          RepairError.emplace(makeDisplacementError(
              "branch at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
              " target could not be evaluated"));
          return false;
        }
        if (OldTarget >= Elf.textSize()) {
          RepairError.emplace(makeDisplacementError(
              "branch at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
              " targets outside .text"));
          return false;
        }

        uint64_t NewFrom = 0;
        uint64_t NewTarget = 0;
        if (!Plan.mapOffset(DI.Offset, DisplacementMapBias::AfterInsertedBytes,
                            NewFrom)) {
          RepairError.emplace(makeDisplacementError(
              "branch source at old .text offset 0x" +
              Twine::utohexstr(DI.Offset) + " maps inside a replaced range"));
          return false;
        }
        if (!Plan.mapOffset(OldTarget, DisplacementMapBias::BeforeInsertedBytes,
                            NewTarget)) {
          RepairError.emplace(makeDisplacementError(
              "branch target at old .text offset 0x" +
              Twine::utohexstr(OldTarget) + " maps inside a replaced range"));
          return false;
        }

        Expected<SmallVector<uint8_t>> EncodedOrErr =
            reencodePcrelBranch(DI, NewFrom, NewTarget, LS);
        if (!EncodedOrErr) {
          RepairError.emplace(EncodedOrErr.takeError());
          return false;
        }
        SmallVector<uint8_t> &Encoded = *EncodedOrErr;

        if (NewFrom > NewText.size() ||
            Encoded.size() > NewText.size() - NewFrom) {
          RepairError.emplace(makeDisplacementError(
              "re-encoded branch at old .text offset 0x" +
              Twine::utohexstr(DI.Offset) + " writes past rebuilt .text"));
          return false;
        }
        std::memcpy(NewText.data() + NewFrom, Encoded.data(), Encoded.size());
        return true;
      });
  if (!Decoded) {
    return makeDisplacementError(
        "failed to decode .text while validating branches");
  }
  if (RepairError)
    return std::move(*RepairError);
  return Error::success();
}

Error adjustSectionHeadersForTextGrowth(uint8_t *Elf, size_t ElfSize,
                                        const ElfView &OldElf, size_t Growth) {
  if (ElfSize < sizeof(Ehdr)) {
    return makeDisplacementError(
        "displaced ELF is smaller than its ELF64 header");
  }

  const uint64_t TextOffset = OldElf.textOffset();
  const uint64_t TextSize = OldElf.textSize();
  const uint64_t TextEnd = TextOffset + TextSize;

  uint64_t Shoff = 0;
  uint16_t Shentsize = 0;
  uint16_t Shnum = 0;
  std::memcpy(&Shoff, Elf + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, Elf + offsetof(Ehdr, e_shentsize), sizeof(Shentsize));
  std::memcpy(&Shnum, Elf + offsetof(Ehdr, e_shnum), sizeof(Shnum));
  if (Shentsize < sizeof(Shdr)) {
    return makeDisplacementError(
        "displaced ELF section-header entry is too small");
  }

  if (Shoff >= TextEnd) {
    uint64_t NewShoff = Shoff + Growth;
    std::memcpy(Elf + offsetof(Ehdr, e_shoff), &NewShoff, sizeof(NewShoff));
    Shoff = NewShoff;
  }

  for (uint16_t I = 0; I < Shnum; ++I) {
    uint64_t ShPos = Shoff + static_cast<uint64_t>(I) * Shentsize;
    if (ShPos > ElfSize || sizeof(Shdr) > ElfSize - ShPos) {
      return makeDisplacementError(
          "displaced ELF section-header table is out of bounds");
    }
    uint8_t *Sh = Elf + ShPos;
    uint64_t ShOffset = 0;
    std::memcpy(&ShOffset, Sh + offsetof(Shdr, sh_offset), sizeof(ShOffset));

    if (I == OldElf.textSectionIndex()) {
      uint64_t NewTextSize = TextSize + Growth;
      std::memcpy(Sh + offsetof(Shdr, sh_size), &NewTextSize,
                  sizeof(NewTextSize));
    } else if (ShOffset >= TextEnd) {
      uint64_t NewOffset = ShOffset + Growth;
      std::memcpy(Sh + offsetof(Shdr, sh_offset), &NewOffset,
                  sizeof(NewOffset));
    }
  }
  return Error::success();
}

Error adjustProgramHeadersForTextGrowth(uint8_t *Elf, size_t ElfSize,
                                        const ElfView &OldElf, size_t Growth) {
  if (ElfSize < sizeof(Ehdr)) {
    return makeDisplacementError(
        "displaced ELF is smaller than its ELF64 header");
  }

  const uint64_t TextOffset = OldElf.textOffset();
  const uint64_t TextEnd = TextOffset + OldElf.textSize();

  uint64_t Phoff = 0;
  uint16_t Phentsize = 0;
  uint16_t Phnum = 0;
  std::memcpy(&Phoff, Elf + offsetof(Ehdr, e_phoff), sizeof(Phoff));
  std::memcpy(&Phentsize, Elf + offsetof(Ehdr, e_phentsize), sizeof(Phentsize));
  std::memcpy(&Phnum, Elf + offsetof(Ehdr, e_phnum), sizeof(Phnum));
  if (Phentsize < sizeof(Phdr)) {
    return makeDisplacementError(
        "displaced ELF program-header entry is too small");
  }

  if (Phoff >= TextEnd) {
    uint64_t NewPhoff = Phoff + Growth;
    std::memcpy(Elf + offsetof(Ehdr, e_phoff), &NewPhoff, sizeof(NewPhoff));
    Phoff = NewPhoff;
  }

  for (uint16_t I = 0; I < Phnum; ++I) {
    uint64_t PhPos = Phoff + static_cast<uint64_t>(I) * Phentsize;
    if (PhPos > ElfSize || sizeof(Phdr) > ElfSize - PhPos) {
      return makeDisplacementError(
          "displaced ELF program-header table is out of bounds");
    }
    uint8_t *Ph = Elf + PhPos;
    uint64_t POffset = 0;
    uint64_t PFilesz = 0;
    uint64_t PMemsz = 0;
    std::memcpy(&POffset, Ph + offsetof(Phdr, p_offset), sizeof(POffset));
    std::memcpy(&PFilesz, Ph + offsetof(Phdr, p_filesz), sizeof(PFilesz));
    std::memcpy(&PMemsz, Ph + offsetof(Phdr, p_memsz), sizeof(PMemsz));

    if (POffset <= TextOffset && POffset + PFilesz >= TextEnd) {
      PFilesz += Growth;
      PMemsz = std::max(PMemsz, PFilesz);
      std::memcpy(Ph + offsetof(Phdr, p_filesz), &PFilesz, sizeof(PFilesz));
      std::memcpy(Ph + offsetof(Phdr, p_memsz), &PMemsz, sizeof(PMemsz));
    } else if (POffset >= TextEnd) {
      POffset += Growth;
      std::memcpy(Ph + offsetof(Phdr, p_offset), &POffset, sizeof(POffset));
    }
  }
  return Error::success();
}

Error adjustSymbolValuesForDisplacement(uint8_t *Elf, size_t ElfSize,
                                        const ElfView &OldElf,
                                        const DisplacementPlan &Plan) {
  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Elf), ElfSize));
  if (!FileOrErr) {
    return makeDisplacementError(
        "failed to parse displaced ELF for symbol repair: " +
        Twine(toString(FileOrErr.takeError())));
  }
  ELFFileT File = std::move(*FileOrErr);

  Expected<ELFT::ShdrRange> SectionsOrErr = File.sections();
  if (!SectionsOrErr) {
    return makeDisplacementError(
        "failed to read displaced ELF sections for symbol repair: " +
        Twine(toString(SectionsOrErr.takeError())));
  }
  ELFT::ShdrRange Sections = *SectionsOrErr;

  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      return makeDisplacementError(
          "failed to read symbol table during displacement: " +
          Twine(toString(SymsOrErr.takeError())));
    }

    for (const ELFT::Sym &Sym : *SymsOrErr) {
      if (Sym.st_shndx == ELF::SHN_UNDEF || Sym.st_shndx >= ELF::SHN_LORESERVE)
        continue;

      Expected<const ELFT::Shdr *> DefShdrOrErr = File.getSection(Sym.st_shndx);
      if (!DefShdrOrErr) {
        return makeDisplacementError(
            "failed to read a symbol's defining section during displacement: " +
            Twine(toString(DefShdrOrErr.takeError())));
      }
      const ELFT::Shdr &DefShdr = **DefShdrOrErr;

      const uint8_t *SymBytes = reinterpret_cast<const uint8_t *>(&Sym);
      if (SymBytes < File.base() || SymBytes > File.end() ||
          static_cast<size_t>(File.end() - SymBytes) < sizeof(ELFT::Sym)) {
        return makeDisplacementError(
            "symbol table entry is outside displaced ELF buffer");
      }
      uint64_t SymOffset = SymBytes - File.base();

      if (Sym.st_shndx == OldElf.textSectionIndex()) {
        const bool LooksLikeVAddr =
            Sym.st_value >= DefShdr.sh_addr &&
            Sym.st_value - DefShdr.sh_addr <= Plan.oldTextSize();
        const uint64_t OldOffset =
            LooksLikeVAddr ? Sym.st_value - DefShdr.sh_addr : Sym.st_value;
        if (OldOffset > Plan.oldTextSize()) {
          return makeDisplacementError(
              "text symbol value is outside the original .text section");
        }

        uint64_t NewOffset = 0;
        if (!Plan.mapOffset(OldOffset, DisplacementMapBias::BeforeInsertedBytes,
                            NewOffset)) {
          return makeDisplacementError(
              "text symbol value maps inside a replaced range");
        }
        const uint64_t NewValue =
            LooksLikeVAddr ? DefShdr.sh_addr + NewOffset : NewOffset;
        std::memcpy(Elf + SymOffset + offsetof(ELFT::Sym, st_value), &NewValue,
                    sizeof(NewValue));

        if (Sym.st_size != 0) {
          if (OldOffset > Plan.oldTextSize() ||
              Sym.st_size > Plan.oldTextSize() - OldOffset) {
            return makeDisplacementError(
                "text symbol extends outside the original .text section");
          }
          uint64_t OldEnd = OldOffset + Sym.st_size;
          uint64_t NewStart = 0;
          uint64_t NewEnd = 0;
          // An insertion at the symbol start belongs to this symbol, while
          // one exactly at the half-open end belongs to the next symbol.
          if (!Plan.mapOffset(OldOffset,
                              DisplacementMapBias::BeforeInsertedBytes,
                              NewStart) ||
              !Plan.mapOffset(OldEnd, DisplacementMapBias::BeforeInsertedBytes,
                              NewEnd) ||
              NewEnd < NewStart) {
            return makeDisplacementError(
                "text symbol boundaries cannot be remapped");
          }
          uint64_t NewSize = NewEnd - NewStart;
          std::memcpy(Elf + SymOffset + offsetof(ELFT::Sym, st_size), &NewSize,
                      sizeof(NewSize));
        }
        continue;
      }

      // Existing non-text virtual addresses are immutable. Fully linked AMDGPU
      // code objects contain baked address materializations with no
      // relocations; moving the target symbol would silently retarget those
      // instructions.
      // ElfView.GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol
      // demonstrates the linked-address dependency.
    }
  }
  return Error::success();
}

Error rewriteKernelDescriptorEntriesForDisplacement(
    WritableMemoryBuffer &OutBuf, const ElfView &OldElf,
    const DisplacementPlan &Plan) {
  uint8_t *Data = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart());
  Expected<ElfView> OutViewOrErr =
      ElfView::create(Data, OutBuf.getBufferSize());
  if (!OutViewOrErr) {
    return makeDisplacementError(
        "failed to reparse displaced ELF for descriptor repair: " +
        Twine(toString(OutViewOrErr.takeError())));
  }

  ElfView &OutElf = *OutViewOrErr;
  for (const KernelDescriptorInfo &KD : OldElf.kernelDescriptors()) {
    std::optional<uint64_t> OldEntryVAddr = entryVAddr(KD);
    if (!OldEntryVAddr) {
      return makeDisplacementError("failed to resolve kernel entry for '" +
                                   Twine(KD.KernelName) + "'");
    }
    if (*OldEntryVAddr < OldElf.textAddr() ||
        *OldEntryVAddr >= OldElf.textAddr() + OldElf.textSize())
      continue;

    uint64_t OldEntryOffset = *OldEntryVAddr - OldElf.textAddr();
    uint64_t NewEntryOffset = 0;
    if (!Plan.mapOffset(OldEntryOffset,
                        DisplacementMapBias::BeforeInsertedBytes,
                        NewEntryOffset)) {
      return makeDisplacementError("kernel descriptor entry for '" +
                                   Twine(KD.KernelName) +
                                   "' maps inside a replaced range");
    }

    std::optional<uint64_t> NewKdVAddr =
        OutElf.getKernelDescriptorVAddr(KD.KernelName);
    if (!NewKdVAddr) {
      return makeDisplacementError("missing kernel descriptor for '" +
                                   Twine(KD.KernelName) +
                                   "' after displacement");
    }

    const uint64_t NewEntryVAddr = OutElf.textAddr() + NewEntryOffset;
    std::optional<int64_t> NewKdEntryOffset = checkedSignedDifference(
        NewEntryVAddr, *NewKdVAddr,
        (Twine("displaced kernel entry offset for '") + KD.KernelName + "'")
            .str());
    if (!NewKdEntryOffset) {
      return makeDisplacementError(
          "displaced kernel entry offset is not representable for '" +
          Twine(KD.KernelName) + "'");
    }
    if (!OutElf.updateKernelDescriptorEntryOffset(KD.KernelName,
                                                  *NewKdEntryOffset)) {
      return makeDisplacementError(
          "failed to update kernel descriptor entry for '" +
          Twine(KD.KernelName) + "'");
    }
  }
  return Error::success();
}

Error applyTextDisplacement(const ElfView &Elf, const LLVMState &LS,
                            const DisplacementPlan &Plan,
                            WritableMemoryBuffer &OutBuf) {
  const size_t InputSize = Elf.size();
  const size_t NewSize = Plan.newElfSize(InputSize);
  if (OutBuf.getBufferSize() != NewSize) {
    return makeDisplacementError(
        "output buffer has incorrect size for displacement");
  }

  SmallVector<uint8_t> NewText = Plan.buildText(
      ArrayRef<uint8_t>(Elf.textData(), Elf.textSize()), LS.SNopBytes);
  if (NewText.size() != Plan.paddedTextSize()) {
    return makeDisplacementError(
        "rebuilt .text size does not match displacement plan");
  }
  if (Error Err = repairBranches(Elf, LS, Plan, NewText))
    return Err;

  uint8_t *Out = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart());
  const uint8_t *Input = Elf.data();
  const uint64_t TextOffset = Elf.textOffset();
  const uint64_t TextEnd = TextOffset + Elf.textSize();

  std::memcpy(Out, Input, TextOffset);
  std::memcpy(Out + TextOffset, NewText.data(), NewText.size());
  if (TextEnd < InputSize) {
    std::memcpy(Out + TextOffset + NewText.size(), Input + TextEnd,
                InputSize - TextEnd);
  }

  if (Error Err = adjustSectionHeadersForTextGrowth(Out, NewSize, Elf,
                                                    Plan.paddedGrowth()))
    return Err;
  if (Error Err = adjustProgramHeadersForTextGrowth(Out, NewSize, Elf,
                                                    Plan.paddedGrowth()))
    return Err;
  if (Error Err = adjustSymbolValuesForDisplacement(Out, NewSize, Elf, Plan))
    return Err;

  if (Error Err =
          rewriteKernelDescriptorEntriesForDisplacement(OutBuf, Elf, Plan))
    return Err;

  log() << "hotswap: displacement: grew ELF from " << InputSize << " to "
        << NewSize << " bytes (" << Plan.edits().size() << " edit"
        << (Plan.edits().size() == 1 ? "" : "s") << ", raw growth "
        << Plan.rawGrowth() << " bytes, padded growth " << Plan.paddedGrowth()
        << " bytes).\n";
  return Error::success();
}

} // namespace

Expected<DisplacementPlan>
DisplacementPlan::create(const ElfView &Elf,
                         ArrayRef<DisplacementEdit> InputEdits) {
  if (InputEdits.empty())
    return makeDisplacementError("no displacement edits requested");

  std::vector<DisplacementEdit> Sorted;
  Sorted.reserve(InputEdits.size());
  for (const DisplacementEdit &Edit : InputEdits)
    Sorted.push_back(Edit);

  std::stable_sort(Sorted.begin(), Sorted.end(),
                   [](const DisplacementEdit &A, const DisplacementEdit &B) {
                     return A.Offset < B.Offset;
                   });

  uint64_t RawGrowth = 0;
  std::optional<uint64_t> PrevOffset;
  uint64_t PrevEnd = 0;
  for (const DisplacementEdit &Edit : Sorted) {
    if (Edit.ReplacementBytes.empty())
      return makeDisplacementError("displacement edit has empty replacement");
    if (Edit.Offset > Elf.textSize() ||
        Edit.OriginalSize > Elf.textSize() - Edit.Offset) {
      return makeDisplacementError("displacement edit is out of .text bounds");
    }
    if (Edit.ReplacementBytes.size() < Edit.OriginalSize)
      return makeDisplacementError("displacement edit shrinks code");
    if (Edit.ReplacementBytes.size() == Edit.OriginalSize)
      return makeDisplacementError("displacement edit has no size delta");
    if (PrevOffset && Edit.Offset < PrevEnd)
      return makeDisplacementError("displacement edits overlap");
    if (PrevOffset && Edit.Offset == *PrevOffset) {
      return makeDisplacementError(
          "multiple displacement edits share an offset");
    }

    uint64_t EditGrowth = Edit.ReplacementBytes.size() - Edit.OriginalSize;
    if (EditGrowth > std::numeric_limits<uint64_t>::max() - RawGrowth)
      return makeDisplacementError("displacement growth overflows uint64_t");
    RawGrowth += EditGrowth;
    PrevOffset = Edit.Offset;
    PrevEnd = Edit.Offset + Edit.OriginalSize;
  }

  Expected<uint64_t> AlignmentOrErr = requiredGrowthAlignment(Elf);
  if (!AlignmentOrErr)
    return AlignmentOrErr.takeError();
  uint64_t Alignment = *AlignmentOrErr;
  if (!isPowerOf2_64(Alignment))
    return makeDisplacementError("post-.text alignment is not a power of two");
  uint64_t Remainder = RawGrowth % Alignment;
  uint64_t Padding = Remainder == 0 ? 0 : Alignment - Remainder;
  if (Padding > std::numeric_limits<uint64_t>::max() - RawGrowth)
    return makeDisplacementError("aligned displacement growth overflows");
  uint64_t PaddedGrowth = RawGrowth + Padding;
  if (Error Err = validateVirtualGrowth(Elf, PaddedGrowth))
    return std::move(Err);
  if (PaddedGrowth > std::numeric_limits<size_t>::max() - Elf.size())
    return makeDisplacementError("displaced ELF size overflows size_t");
  return DisplacementPlan(Elf.textSize(), RawGrowth, PaddedGrowth,
                          std::move(Sorted));
}

bool DisplacementPlan::mapOffset(uint64_t OldOffset, DisplacementMapBias Bias,
                                 uint64_t &NewOffset) const {
  if (OldOffset > OldTextSize)
    return false;

  uint64_t Delta = 0;
  for (const DisplacementEdit &Edit : Edits) {
    const uint64_t EditEnd = Edit.Offset + Edit.OriginalSize;
    const uint64_t EditDelta = Edit.ReplacementBytes.size() - Edit.OriginalSize;

    if (OldOffset < Edit.Offset)
      break;

    if (OldOffset == Edit.Offset) {
      if (Edit.OriginalSize == 0 &&
          Bias == DisplacementMapBias::AfterInsertedBytes) {
        Delta += EditDelta;
        continue;
      }
      break;
    }

    if (OldOffset < EditEnd)
      return false;

    Delta += EditDelta;
  }

  NewOffset = OldOffset + Delta;
  return true;
}

bool DisplacementPlan::rangeOverlapsReplacement(uint64_t OldOffset,
                                                uint64_t Size) const {
  if (Size == 0)
    return false;
  const uint64_t OldEnd = OldOffset + Size;
  for (const DisplacementEdit &Edit : Edits) {
    if (Edit.OriginalSize == 0)
      continue;
    const uint64_t EditEnd = Edit.Offset + Edit.OriginalSize;
    if (OldOffset < EditEnd && OldEnd > Edit.Offset)
      return true;
  }
  return false;
}

SmallVector<uint8_t>
DisplacementPlan::buildText(ArrayRef<uint8_t> OldText,
                            ArrayRef<uint8_t> SNopBytes) const {
  SmallVector<uint8_t> Out;
  Out.reserve(paddedTextSize());

  uint64_t Pos = 0;
  for (const DisplacementEdit &Edit : Edits) {
    Out.append(OldText.begin() + Pos, OldText.begin() + Edit.Offset);
    Out.append(Edit.ReplacementBytes.begin(), Edit.ReplacementBytes.end());
    Pos = Edit.Offset + Edit.OriginalSize;
  }
  Out.append(OldText.begin() + Pos, OldText.end());

  uint64_t PadBytes = paddedTextSize() - Out.size();
  while (PadBytes >= MinInstSize && SNopBytes.size() == MinInstSize) {
    Out.append(SNopBytes.begin(), SNopBytes.end());
    PadBytes -= MinInstSize;
  }
  Out.append(PadBytes, uint8_t{0});
  return Out;
}

Expected<std::unique_ptr<WritableMemoryBuffer>>
tryApplyTextDisplacementToNewBuffer(const ElfView &Elf, const LLVMState &LS,
                                    ArrayRef<DisplacementEdit> Edits) {
  if (Error Err = validateDebugSections(Elf))
    return std::move(Err);
  if (Error Err = validateTextRelocations(Elf))
    return std::move(Err);

  Expected<DisplacementPlan> PlanOrErr = DisplacementPlan::create(Elf, Edits);
  if (!PlanOrErr)
    return PlanOrErr.takeError();
  if (Error Err = validateKernelEntryMappings(Elf, *PlanOrErr))
    return std::move(Err);

  std::unique_ptr<WritableMemoryBuffer> Out =
      WritableMemoryBuffer::getNewUninitMemBuffer(
          PlanOrErr->newElfSize(Elf.size()));
  if (!Out) {
    return makeDisplacementError(
        "failed to allocate displacement output buffer");
  }
  if (Error Err = applyTextDisplacement(Elf, LS, *PlanOrErr, *Out))
    return std::move(Err);
  return std::move(Out);
}

} // namespace hotswap
} // namespace COMGR
