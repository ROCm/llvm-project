//===- displacement.cpp - HotSwap text displacement -----------------------===//
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

#include "internal.h"

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

  const ELFT::Shdr &Owner = **OwnerOrErr;
  if (Plan.relocatesTrailingSections() && Owner.sh_addr >= OldTextEnd) {
    if (OldAddress >
        std::numeric_limits<uint64_t>::max() - Plan.paddedGrowth()) {
      return makeDisplacementError(Twine(Context) +
                                   " overflows after section displacement");
    }
    return OldAddress + Plan.paddedGrowth();
  }
  return OldAddress;
}

Expected<uint64_t> requiredGrowthAlignment(const ElfView &Elf) {
  uint64_t MaxAlign = 1;
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (Shdr.sh_offset <= Elf.textOffset())
      continue;
    if (Shdr.sh_addralign > 1 && !isPowerOf2_64(Shdr.sh_addralign)) {
      return makeDisplacementError(
          "post-.text section alignment is not a power of two");
    }
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
    if (Phdr.p_align > 1 && !isPowerOf2_64(Phdr.p_align)) {
      return makeDisplacementError(
          "post-.text program-header alignment is not a power of two");
    }
    MaxAlign = std::max<uint64_t>(MaxAlign, Phdr.p_align);
  }
  return std::max<uint64_t>(MaxAlign, 1);
}

Error validateEditInstructionBoundaries(const ElfView &Elf, const LLVMState &LS,
                                        const DisplacementPlan &Plan) {
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Elf.textData(), Elf.textSize(), LS, Decoded)) {
    return makeDisplacementError(
        "failed to decode .text while validating edit boundaries");
  }

  DenseSet<uint64_t> Boundaries;
  Boundaries.insert(0);
  for (const InternalDecodedInst &DI : Decoded) {
    if (!DI.DecodeSucceeded) {
      return makeDisplacementError(
          "original .text is not a complete sequence of instructions");
    }
    Boundaries.insert(DI.Offset + DI.Size);
  }
  if (!Boundaries.contains(Elf.textSize())) {
    return makeDisplacementError(
        "original .text is not a complete sequence of instructions");
  }

  for (const DisplacementEdit &Edit : Plan.edits()) {
    const uint64_t EditEnd = Edit.Offset + Edit.OriginalSize;
    if (!Boundaries.contains(Edit.Offset) || !Boundaries.contains(EditEnd)) {
      return makeDisplacementError(
          "displacement edit does not start and end at instruction "
          "boundaries");
    }

    std::vector<InternalDecodedInst> ReplacementDecoded;
    if (!decodeTextSection(Edit.ReplacementBytes.data(),
                           Edit.ReplacementBytes.size(), LS,
                           ReplacementDecoded)) {
      return makeDisplacementError("failed to decode displacement replacement");
    }
    uint64_t ReplacementEnd = 0;
    for (const InternalDecodedInst &DI : ReplacementDecoded) {
      if (!DI.DecodeSucceeded) {
        return makeDisplacementError(
            "displacement replacement is not a complete sequence of "
            "instructions");
      }
      ReplacementEnd = DI.Offset + DI.Size;
    }
    if (ReplacementEnd != Edit.ReplacementBytes.size()) {
      return makeDisplacementError(
          "displacement replacement is not a complete sequence of "
          "instructions");
    }
  }
  return Error::success();
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

enum class DynamicTagClass {
  Value,
  Address,
  UnsupportedAddress,
  Unknown,
};

DynamicTagClass classifyDynamicTag(int64_t Tag) {
  switch (Tag) {
  case ELF::DT_NULL:
  case ELF::DT_NEEDED:
  case ELF::DT_PLTRELSZ:
  case ELF::DT_RELASZ:
  case ELF::DT_RELAENT:
  case ELF::DT_STRSZ:
  case ELF::DT_SYMENT:
  case ELF::DT_SONAME:
  case ELF::DT_RPATH:
  case ELF::DT_SYMBOLIC:
  case ELF::DT_RELSZ:
  case ELF::DT_RELENT:
  case ELF::DT_PLTREL:
  case ELF::DT_TEXTREL:
  case ELF::DT_BIND_NOW:
  case ELF::DT_INIT_ARRAYSZ:
  case ELF::DT_FINI_ARRAYSZ:
  case ELF::DT_RUNPATH:
  case ELF::DT_FLAGS:
  case ELF::DT_PREINIT_ARRAYSZ:
  case ELF::DT_RELRSZ:
  case ELF::DT_RELRENT:
  case ELF::DT_RELACOUNT:
  case ELF::DT_RELCOUNT:
  case ELF::DT_FLAGS_1:
  case ELF::DT_VERDEFNUM:
  case ELF::DT_VERNEEDNUM:
    return DynamicTagClass::Value;

  case ELF::DT_HASH:
  case ELF::DT_STRTAB:
  case ELF::DT_SYMTAB:
  case ELF::DT_RELA:
  case ELF::DT_REL:
  case ELF::DT_SYMTAB_SHNDX:
  case ELF::DT_GNU_HASH:
  case ELF::DT_VERSYM:
  case ELF::DT_VERDEF:
  case ELF::DT_VERNEED:
    return DynamicTagClass::Address;

  // These tags introduce executable entry points, pointer arrays, GOT/PLT
  // state, TLS callbacks, or relocation encodings outside the displacement
  // proof. Supporting only their container address would leave the contents
  // or an externally callable entry unproved.
  case ELF::DT_PLTGOT:
  case ELF::DT_INIT:
  case ELF::DT_FINI:
  case ELF::DT_DEBUG:
  case ELF::DT_JMPREL:
  case ELF::DT_INIT_ARRAY:
  case ELF::DT_FINI_ARRAY:
  case ELF::DT_PREINIT_ARRAY:
  case ELF::DT_RELR:
  case ELF::DT_CREL:
  case ELF::DT_ANDROID_REL:
  case ELF::DT_ANDROID_RELA:
  case ELF::DT_ANDROID_RELR:
  case ELF::DT_TLSDESC_PLT:
  case ELF::DT_TLSDESC_GOT:
    return DynamicTagClass::UnsupportedAddress;
  }
  return DynamicTagClass::Unknown;
}

bool isKnownAllocatedSectionType(uint32_t Type) {
  switch (Type) {
  case ELF::SHT_PROGBITS:
  case ELF::SHT_SYMTAB:
  case ELF::SHT_STRTAB:
  case ELF::SHT_HASH:
  case ELF::SHT_DYNAMIC:
  case ELF::SHT_NOTE:
  case ELF::SHT_NOBITS:
  case ELF::SHT_DYNSYM:
  case ELF::SHT_GROUP:
  case ELF::SHT_SYMTAB_SHNDX:
  case ELF::SHT_GNU_HASH:
  case ELF::SHT_GNU_verdef:
  case ELF::SHT_GNU_verneed:
  case ELF::SHT_GNU_versym:
    return true;
  }
  return false;
}

bool dynamicTagMatchesSection(int64_t Tag, uint32_t Type) {
  switch (Tag) {
  case ELF::DT_HASH:
    return Type == ELF::SHT_HASH;
  case ELF::DT_STRTAB:
    return Type == ELF::SHT_STRTAB;
  case ELF::DT_SYMTAB:
    return Type == ELF::SHT_DYNSYM;
  case ELF::DT_RELA:
    return Type == ELF::SHT_RELA;
  case ELF::DT_REL:
    return Type == ELF::SHT_REL;
  case ELF::DT_SYMTAB_SHNDX:
    return Type == ELF::SHT_SYMTAB_SHNDX;
  case ELF::DT_GNU_HASH:
    return Type == ELF::SHT_GNU_HASH;
  case ELF::DT_VERSYM:
    return Type == ELF::SHT_GNU_versym;
  case ELF::DT_VERDEF:
    return Type == ELF::SHT_GNU_verdef;
  case ELF::DT_VERNEED:
    return Type == ELF::SHT_GNU_verneed;
  }
  return false;
}

Error validateTrailingRelocationLayout(const ElfView &Elf) {
  if (Elf.file().getHeader().e_type != ELF::ET_DYN) {
    return makeDisplacementError(
        "trailing-section relocation requires a linked ET_DYN code object");
  }
  if (!(Elf.textSection()->sh_flags & ELF::SHF_ALLOC) ||
      !(Elf.textSection()->sh_flags & ELF::SHF_EXECINSTR)) {
    return makeDisplacementError(
        ".text must be an allocated executable section");
  }
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr() ||
      Elf.textSize() >
          std::numeric_limits<uint64_t>::max() - Elf.textOffset()) {
    return makeDisplacementError(
        ".text range overflows while validating trailing relocation");
  }
  const uint64_t TextAddressEnd = Elf.textAddr() + Elf.textSize();
  const uint64_t TextFileEnd = Elf.textOffset() + Elf.textSize();

  SmallVector<const ELFT::Shdr *, 8> AllocatedSections;
  unsigned DynamicSectionCount = 0;
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (Shdr.sh_addralign > 1 && !isPowerOf2_64(Shdr.sh_addralign)) {
      return makeDisplacementError("section alignment is not a power of two");
    }
    if (Shdr.sh_type == ELF::SHT_DYNAMIC) {
      ++DynamicSectionCount;
      if (DynamicSectionCount > 1) {
        return makeDisplacementError(
            "multiple dynamic sections are outside the displacement model");
      }
      if (!(Shdr.sh_flags & ELF::SHF_ALLOC) ||
          Shdr.sh_entsize != sizeof(ELFT::Dyn) || Shdr.sh_size == 0 ||
          Shdr.sh_size % sizeof(ELFT::Dyn) != 0) {
        return makeDisplacementError("dynamic section has an invalid layout");
      }
    }
    if (Shdr.sh_type == ELF::SHT_REL || Shdr.sh_type == ELF::SHT_RELA) {
      return makeDisplacementError(
          "relocation sections require relocation-record repair");
    }
    if (Shdr.sh_type == ELF::SHT_RELR || Shdr.sh_type == ELF::SHT_CREL ||
        Shdr.sh_type == ELF::SHT_ANDROID_REL ||
        Shdr.sh_type == ELF::SHT_ANDROID_RELA ||
        Shdr.sh_type == ELF::SHT_ANDROID_RELR ||
        Shdr.sh_type == ELF::SHT_INIT_ARRAY ||
        Shdr.sh_type == ELF::SHT_FINI_ARRAY ||
        Shdr.sh_type == ELF::SHT_PREINIT_ARRAY ||
        Shdr.sh_type == ELF::SHT_GNU_SFRAME ||
        Shdr.sh_type == ELF::SHT_LLVM_BB_ADDR_MAP ||
        Shdr.sh_type == ELF::SHT_LLVM_JT_SIZES ||
        Shdr.sh_type == ELF::SHT_LLVM_CFI_JUMP_TABLE ||
        Shdr.sh_type == ELF::SHT_LLVM_CALL_GRAPH) {
      return makeDisplacementError(
          "code object has an unsupported address-bearing section type");
    }

    bool FileBefore = false;
    bool FileAfter = false;
    if (&Shdr != Elf.textSection() && Shdr.sh_type != ELF::SHT_NOBITS &&
        Shdr.sh_size != 0) {
      if (Shdr.sh_size >
          std::numeric_limits<uint64_t>::max() - Shdr.sh_offset) {
        return makeDisplacementError("section file range overflows");
      }
      FileBefore = Shdr.sh_offset + Shdr.sh_size <= Elf.textOffset();
      FileAfter = Shdr.sh_offset >= TextFileEnd;
      if (!FileBefore && !FileAfter) {
        return makeDisplacementError(
            "section file range overlaps .text content");
      }
    }

    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) || Shdr.sh_size == 0)
      continue;
    if (!isKnownAllocatedSectionType(Shdr.sh_type)) {
      return makeDisplacementError(
          "allocated section type is outside the displacement whitelist");
    }
    if (Shdr.sh_size > std::numeric_limits<uint64_t>::max() - Shdr.sh_addr) {
      return makeDisplacementError("allocated section address range overflows");
    }
    AllocatedSections.push_back(&Shdr);
    if (&Shdr == Elf.textSection())
      continue;
    if (Shdr.sh_flags & ELF::SHF_EXECINSTR) {
      return makeDisplacementError(
          "trailing relocation supports one executable section");
    }

    const bool AddressBefore = Shdr.sh_addr + Shdr.sh_size <= Elf.textAddr();
    const bool AddressAfter = Shdr.sh_addr >= TextAddressEnd;
    if (!AddressBefore && !AddressAfter) {
      return makeDisplacementError(
          "allocated section overlaps .text in virtual memory");
    }

    if (Shdr.sh_type != ELF::SHT_NOBITS) {
      if (FileBefore != AddressBefore) {
        return makeDisplacementError(
            "allocated section file and virtual ordering disagree");
      }
    } else if ((AddressBefore && Shdr.sh_offset > Elf.textOffset()) ||
               (AddressAfter && Shdr.sh_offset < TextFileEnd)) {
      return makeDisplacementError(
          "NOBITS section file and virtual ordering disagree");
    }
  }

  for (size_t I = 0; I != AllocatedSections.size(); ++I) {
    const ELFT::Shdr &Left = *AllocatedSections[I];
    const uint64_t LeftEnd = Left.sh_addr + Left.sh_size;
    for (size_t J = I + 1; J != AllocatedSections.size(); ++J) {
      const ELFT::Shdr &Right = *AllocatedSections[J];
      const uint64_t RightEnd = Right.sh_addr + Right.sh_size;
      if (Left.sh_addr < RightEnd && Right.sh_addr < LeftEnd) {
        return makeDisplacementError(
            "allocated sections overlap in virtual memory");
      }
    }
  }

  for (const ELFT::Shdr &SymShdr : Elf.sections()) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ELFT::SymRange> SymbolsOrErr = Elf.file().symbols(&SymShdr);
    if (!SymbolsOrErr) {
      return makeDisplacementError(
          "failed to read symbols while validating absolute addresses: " +
          Twine(toString(SymbolsOrErr.takeError())));
    }
    for (const ELFT::Sym &Symbol : *SymbolsOrErr) {
      if (Symbol.st_shndx != ELF::SHN_ABS || Symbol.st_value == 0)
        continue;
      if (Symbol.getType() == ELF::STT_FUNC) {
        return makeDisplacementError(
            "absolute function symbol is outside the displacement map");
      }
      for (const ELFT::Shdr &Owner : Elf.sections()) {
        if (!(Owner.sh_flags & ELF::SHF_ALLOC) || Owner.sh_size == 0 ||
            Symbol.st_value < Owner.sh_addr ||
            Symbol.st_value - Owner.sh_addr >= Owner.sh_size)
          continue;
        return makeDisplacementError(
            "absolute symbol aliases movable allocated content");
      }
    }
  }

  Expected<ELFT::PhdrRange> PhdrsOrErr = Elf.file().program_headers();
  if (!PhdrsOrErr) {
    return makeDisplacementError(
        "failed to read program headers while validating trailing "
        "relocation: " +
        Twine(toString(PhdrsOrErr.takeError())));
  }
  bool FoundTextLoad = false;
  unsigned DynamicSegmentCount = 0;
  SmallVector<const ELFT::Phdr *, 8> LoadSegments;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_align > 1 && !isPowerOf2_64(Phdr.p_align)) {
      return makeDisplacementError(
          "program-header alignment is not a power of two");
    }
    if (Phdr.p_filesz > std::numeric_limits<uint64_t>::max() - Phdr.p_offset ||
        Phdr.p_memsz > std::numeric_limits<uint64_t>::max() - Phdr.p_vaddr) {
      return makeDisplacementError("program-header range overflows");
    }
    if (Phdr.p_type == ELF::PT_DYNAMIC) {
      ++DynamicSegmentCount;
      if (DynamicSegmentCount > 1) {
        return makeDisplacementError(
            "multiple PT_DYNAMIC segments are outside the displacement "
            "model");
      }
      if (Phdr.p_filesz == 0 || Phdr.p_filesz % sizeof(ELFT::Dyn) != 0 ||
          Phdr.p_offset > Elf.size() ||
          Phdr.p_filesz > Elf.size() - Phdr.p_offset) {
        return makeDisplacementError(
            "PT_DYNAMIC has an invalid file-backed range");
      }
    }
    const uint64_t FileEnd = Phdr.p_offset + Phdr.p_filesz;
    const uint64_t AddressEnd = Phdr.p_vaddr + Phdr.p_memsz;
    if (Phdr.p_type != ELF::PT_LOAD) {
      if (Phdr.p_filesz != 0 && Phdr.p_offset < TextFileEnd &&
          FileEnd > Elf.textOffset()) {
        return makeDisplacementError(
            "non-PT_LOAD file range overlaps .text content");
      }
      if (Phdr.p_memsz != 0 && Phdr.p_vaddr < TextAddressEnd &&
          AddressEnd > Elf.textAddr()) {
        return makeDisplacementError(
            "non-PT_LOAD address range overlaps .text");
      }
      continue;
    }
    if (Phdr.p_align != 0 &&
        Phdr.p_offset % Phdr.p_align != Phdr.p_vaddr % Phdr.p_align) {
      return makeDisplacementError(
          "PT_LOAD file and virtual alignment are incongruent");
    }
    LoadSegments.push_back(&Phdr);

    const bool ContainsTextFile =
        Phdr.p_offset <= Elf.textOffset() && FileEnd >= TextFileEnd;
    const bool ContainsTextAddress =
        Phdr.p_vaddr <= Elf.textAddr() && AddressEnd >= TextAddressEnd;
    if (ContainsTextFile || ContainsTextAddress) {
      if (!ContainsTextFile || !ContainsTextAddress || FileEnd != TextFileEnd) {
        return makeDisplacementError(
            ".text is not the last file-backed content in its PT_LOAD");
      }
      FoundTextLoad = true;
      continue;
    }

    const bool FileBefore = FileEnd <= Elf.textOffset();
    const bool FileAfter = Phdr.p_offset >= TextFileEnd;
    const bool AddressBefore = AddressEnd <= Elf.textAddr();
    const bool AddressAfter = Phdr.p_vaddr >= TextAddressEnd;
    if ((!FileBefore && !FileAfter) || (!AddressBefore && !AddressAfter) ||
        FileBefore != AddressBefore) {
      return makeDisplacementError(
          "PT_LOAD file and virtual ordering around .text disagree");
    }
  }
  if (!FoundTextLoad)
    return makeDisplacementError(".text is not covered by a PT_LOAD segment");

  for (const ELFT::Shdr *Shdr : AllocatedSections) {
    unsigned OwnerCount = 0;
    for (const ELFT::Phdr *Phdr : LoadSegments)
      if (object::isSectionInSegment<ELFT>(*Phdr, *Shdr))
        ++OwnerCount;
    if (OwnerCount != 1) {
      return makeDisplacementError(
          "allocated section is not covered by exactly one PT_LOAD segment");
    }
  }

  for (size_t I = 0; I != LoadSegments.size(); ++I) {
    const ELFT::Phdr &Left = *LoadSegments[I];
    const uint64_t LeftEnd = Left.p_vaddr + Left.p_memsz;
    for (size_t J = I + 1; J != LoadSegments.size(); ++J) {
      const ELFT::Phdr &Right = *LoadSegments[J];
      const uint64_t RightEnd = Right.p_vaddr + Right.p_memsz;
      if (Left.p_vaddr < RightEnd && Right.p_vaddr < LeftEnd) {
        return makeDisplacementError(
            "PT_LOAD segments overlap in virtual memory");
      }
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

Error validateSupportedElfNumbering(const ElfView &Elf) {
  const ELFT::Ehdr &Header = Elf.file().getHeader();
  if ((Header.e_shnum == 0 && Header.e_shoff != 0) ||
      Header.e_shnum >= ELF::SHN_LORESERVE || Header.e_phnum == ELF::PN_XNUM ||
      Header.e_shstrndx == ELF::SHN_XINDEX) {
    return makeDisplacementError(
        "ELF extended section or program-header numbering is unsupported");
  }

  for (const ELFT::Shdr &SymShdr : Elf.sections()) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ELFT::SymRange> SymbolsOrErr = Elf.file().symbols(&SymShdr);
    if (!SymbolsOrErr) {
      return makeDisplacementError(
          "failed to read symbols while checking extended section indexes: " +
          Twine(toString(SymbolsOrErr.takeError())));
    }
    for (const ELFT::Sym &Symbol : *SymbolsOrErr) {
      if (Symbol.st_shndx == ELF::SHN_XINDEX) {
        return makeDisplacementError(
            "ELF symbol extended section indexes are unsupported");
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
                        int64_t NewValue, bool ForceLiteral,
                        const LLVMState &LS, StringRef Context) {
  if (OperandIndex >= DI.Inst.getNumOperands())
    return makeDisplacementError(Twine(Context) + " has no immediate operand");

  MCInst NewInst = DI.Inst;
  MCOperand &Operand = NewInst.getOperand(OperandIndex);
  if (Operand.isImm() && !ForceLiteral) {
    Operand.setImm(NewValue);
  } else if (Operand.isImm() || Operand.isExpr()) {
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
    std::string LiteralAssembly;
    unsigned LiteralOperandIndex = 0;
    bool IsLiteral32 = false;
    if (DI.Inst.getOpcode() == LS.SAddPcI64Opcode) {
      LiteralAssembly =
          ("s_add_pc_i64 lit64(0x" +
           Twine::utohexstr(static_cast<uint64_t>(NewValue)) + ")")
              .str();
    } else if (DI.Inst.getOpcode() == LS.SAddNcU64Opcode) {
      LiteralAssembly =
          ("s_add_nc_u64 s[0:1], s[0:1], lit64(0x" +
           Twine::utohexstr(static_cast<uint64_t>(NewValue)) + ")")
              .str();
      LiteralOperandIndex = 2;
    } else if (DI.Inst.getOpcode() == LS.SAddU32Opcode) {
      LiteralAssembly =
          ("s_add_u32 s0, s0, lit(0x" +
           Twine::utohexstr(static_cast<uint32_t>(NewValue)) + ")")
              .str();
      LiteralOperandIndex = 2;
      IsLiteral32 = true;
    } else if (DI.Inst.getOpcode() == LS.SAddcU32Opcode) {
      LiteralAssembly =
          ("s_addc_u32 s1, s1, lit(0x" +
           Twine::utohexstr(static_cast<uint32_t>(NewValue)) + ")")
              .str();
      LiteralOperandIndex = 2;
      IsLiteral32 = true;
    } else {
      return makeDisplacementError(Twine(Context) +
                                   " has no MC forced-literal builder");
    }

    SmallVector<uint8_t> LiteralBytes = assembleSingleInst(LiteralAssembly, LS);
    if (LiteralBytes.empty()) {
      return makeDisplacementError(Twine(Context) +
                                   " could not assemble an MC literal");
    }
    std::vector<InternalDecodedInst> LiteralDecoded;
    if (!decodeTextSection(LiteralBytes.data(), LiteralBytes.size(), LS,
                           LiteralDecoded) ||
        LiteralDecoded.size() != 1) {
      return makeDisplacementError(Twine(Context) +
                                   " could not decode one MC literal");
    }
    if (LiteralDecoded[0].Inst.getOpcode() != DI.Inst.getOpcode()) {
      return makeDisplacementError(Twine(Context) +
                                   " rebuilt a different MC opcode");
    }
    if (LiteralOperandIndex >= LiteralDecoded[0].Inst.getNumOperands()) {
      return makeDisplacementError(Twine(Context) +
                                   " rebuilt an MC literal without an operand");
    }
    const MCOperand &LiteralOperand =
        LiteralDecoded[0].Inst.getOperand(LiteralOperandIndex);
    int64_t LiteralValue = 0;
    if (!getAbsoluteOperand(LiteralOperand, LiteralValue) ||
        (IsLiteral32 ? static_cast<uint32_t>(LiteralValue) !=
                           static_cast<uint32_t>(NewValue)
                     : static_cast<uint64_t>(LiteralValue) !=
                           static_cast<uint64_t>(NewValue))) {
      return makeDisplacementError(Twine(Context) +
                                   " rebuilt the wrong MC literal value");
    }
    Operand = LiteralOperand;
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

struct PendingAbsoluteRepair {
  PendingAbsoluteRepair(const InternalDecodedInst &DI, unsigned OperandIndex,
                        int64_t NewValue, uint64_t NewOffset, bool ForceLiteral,
                        std::string Context)
      : DI(DI), OperandIndex(OperandIndex), NewValue(NewValue),
        NewOffset(NewOffset), ForceLiteral(ForceLiteral),
        Context(std::move(Context)) {}

  InternalDecodedInst DI;
  unsigned OperandIndex;
  int64_t NewValue;
  uint64_t NewOffset;
  bool ForceLiteral;
  std::string Context;
};

bool hasProtectedEntryInRange(const DenseSet<uint64_t> &ProtectedEntries,
                              uint64_t Offset, uint64_t Size) {
  for (uint64_t I = 0; I != Size; ++I)
    if (ProtectedEntries.contains(Offset + I))
      return true;
  return false;
}

Expected<bool> repairSGetPcPairForDisplacement(
    const ElfView &Elf, const LLVMState &LS, const DisplacementPlan &Plan,
    const InternalDecodedInst &GetPc,
    const DenseSet<uint64_t> &ProtectedEntries,
    SmallVectorImpl<PendingAbsoluteRepair> &PendingRepairs) {
  if (GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg()) {
    return false;
  }

  std::optional<uint64_t> AddOldOffOrErr =
      checkedAddUint64(GetPc.Offset, GetPc.Size, "s_get_pc pair offset");
  if (!AddOldOffOrErr || *AddOldOffOrErr >= Elf.textSize())
    return false;
  const uint64_t AddOldOff = *AddOldOffOrErr;

  const uint64_t Remaining = Elf.textSize() - AddOldOff;
  const uint64_t DecodeSize =
      std::min<uint64_t>(LS.MAI->getMaxInstLength(LS.STI.get()), Remaining);
  std::vector<InternalDecodedInst> AddDecoded;
  if (!decodeTextSection(Elf.textData() + AddOldOff, DecodeSize, LS,
                         AddDecoded) ||
      AddDecoded.empty()) {
    return false;
  }
  const InternalDecodedInst &Add = AddDecoded.front();
  if (!Add.DecodeSucceeded || Add.Inst.getOpcode() != LS.SAddNcU64Opcode ||
      Add.Inst.getNumOperands() != 3 || !Add.Inst.getOperand(0).isReg() ||
      !Add.Inst.getOperand(1).isReg() ||
      Add.Inst.getOperand(0).getReg() != GetPc.Inst.getOperand(0).getReg() ||
      Add.Inst.getOperand(1).getReg() != GetPc.Inst.getOperand(0).getReg() ||
      hasProtectedEntryInRange(ProtectedEntries, AddOldOff, Add.Size) ||
      Plan.rangeOverlapsReplacement(AddOldOff, Add.Size)) {
    return false;
  }

  // The pair is a semantic unit: s_get_pc captures the address of the
  // immediately following add. An insertion at their shared boundary would
  // leave both original instructions intact while changing the captured PC
  // and permitting the inserted code to clobber the register pair.
  uint64_t AddBeforeInsertions = 0;
  uint64_t AddAfterInsertions = 0;
  if (!Plan.mapOffset(AddOldOff, DisplacementMapBias::BeforeInsertedBytes,
                      AddBeforeInsertions) ||
      !Plan.mapOffset(AddOldOff, DisplacementMapBias::AfterInsertedBytes,
                      AddAfterInsertions) ||
      AddBeforeInsertions != AddAfterInsertions) {
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
  PendingRepairs.emplace_back(Add, /*OperandIndex=*/2, NewAddend, NewAddOff,
                              /*ForceLiteral=*/false, std::move(Context));
  return true;
}

Expected<bool> repairLegacySGetPcSetPcForDisplacement(
    const ElfView &Elf, const LLVMState &LS, const DisplacementPlan &Plan,
    ArrayRef<InternalDecodedInst> Decoded, size_t GetPcIndex,
    const DenseSet<uint64_t> &ProtectedEntries,
    SmallVectorImpl<PendingAbsoluteRepair> &PendingRepairs,
    DenseSet<uint64_t> &RepairedPcSensitiveOffsets) {
  if (!LS.MRI || GetPcIndex >= Decoded.size())
    return false;
  const InternalDecodedInst &GetPc = Decoded[GetPcIndex];
  if (GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
      Decoded.size() - GetPcIndex < 4)
    return false;

  const InternalDecodedInst &AddLo = Decoded[GetPcIndex + 1];
  const InternalDecodedInst &AddHi = Decoded[GetPcIndex + 2];
  const InternalDecodedInst &SetPc = Decoded[GetPcIndex + 3];
  if (!AddLo.DecodeSucceeded || !AddHi.DecodeSucceeded ||
      !SetPc.DecodeSucceeded || GetPc.Offset + GetPc.Size != AddLo.Offset ||
      AddLo.Offset + AddLo.Size != AddHi.Offset ||
      AddHi.Offset + AddHi.Size != SetPc.Offset ||
      AddLo.Inst.getOpcode() != LS.SAddU32Opcode ||
      AddHi.Inst.getOpcode() != LS.SAddcU32Opcode ||
      SetPc.Inst.getOpcode() != LS.SSetPcI64Opcode ||
      AddLo.Inst.getNumOperands() != 3 || AddHi.Inst.getNumOperands() != 3 ||
      SetPc.Inst.getNumOperands() != 1 || !AddLo.Inst.getOperand(0).isReg() ||
      !AddLo.Inst.getOperand(1).isReg() || !AddHi.Inst.getOperand(0).isReg() ||
      !AddHi.Inst.getOperand(1).isReg() || !SetPc.Inst.getOperand(0).isReg() ||
      AddLo.Inst.getOperand(0).getReg() != AddLo.Inst.getOperand(1).getReg() ||
      AddHi.Inst.getOperand(0).getReg() != AddHi.Inst.getOperand(1).getReg() ||
      SetPc.Inst.getOperand(0).getReg() != GetPc.Inst.getOperand(0).getReg() ||
      hasProtectedEntryInRange(ProtectedEntries, AddLo.Offset, AddLo.Size) ||
      hasProtectedEntryInRange(ProtectedEntries, AddHi.Offset, AddHi.Size) ||
      hasProtectedEntryInRange(ProtectedEntries, SetPc.Offset, SetPc.Size) ||
      Plan.rangeOverlapsReplacement(AddLo.Offset, AddLo.Size) ||
      Plan.rangeOverlapsReplacement(AddHi.Offset, AddHi.Size) ||
      Plan.rangeOverlapsReplacement(SetPc.Offset, SetPc.Size))
    return false;

  const MCRegister PairReg = GetPc.Inst.getOperand(0).getReg();
  const MCRegister LoReg = AddLo.Inst.getOperand(0).getReg();
  const MCRegister HiReg = AddHi.Inst.getOperand(0).getReg();
  const unsigned LoSubRegIndex = LS.MRI->getSubRegIndex(PairReg, LoReg);
  const unsigned HiSubRegIndex = LS.MRI->getSubRegIndex(PairReg, HiReg);
  if (LoSubRegIndex == 0 || HiSubRegIndex == 0 ||
      LoSubRegIndex >= HiSubRegIndex)
    return false;

  const uint64_t InteriorOffsets[] = {AddLo.Offset, AddHi.Offset, SetPc.Offset};
  for (uint64_t Offset : InteriorOffsets) {
    uint64_t BeforeInsertions = 0;
    uint64_t AfterInsertions = 0;
    if (!Plan.mapOffset(Offset, DisplacementMapBias::BeforeInsertedBytes,
                        BeforeInsertions) ||
        !Plan.mapOffset(Offset, DisplacementMapBias::AfterInsertedBytes,
                        AfterInsertions) ||
        BeforeInsertions != AfterInsertions)
      return false;
  }

  int64_t OldLo = 0;
  int64_t OldHi = 0;
  if (!getAbsoluteOperand(AddLo.Inst.getOperand(2), OldLo) ||
      !getAbsoluteOperand(AddHi.Inst.getOperand(2), OldHi))
    return false;
  const uint64_t OldDelta =
      static_cast<uint32_t>(OldLo) |
      (static_cast<uint64_t>(static_cast<uint32_t>(OldHi)) << 32);
  std::optional<uint64_t> OldPcBase = checkedAddUint64(
      Elf.textAddr(), AddLo.Offset, "legacy s_get_pc old PC base");
  if (!OldPcBase)
    return false;
  // The two 32-bit adds implement modulo-2^64 arithmetic. Unsigned addition
  // and subtraction reproduce both positive and negative PC deltas without
  // signed overflow.
  const uint64_t OldTarget = *OldPcBase + OldDelta;
  Expected<uint64_t> NewTargetOrErr = remapAllocatedAddress(
      Elf, Plan, OldTarget,
      /*RequireExecutable=*/true, "legacy s_get_pc jump target");
  if (!NewTargetOrErr) {
    consumeError(NewTargetOrErr.takeError());
    return false;
  }

  uint64_t NewPcOffset = 0;
  if (!Plan.mapOffset(AddLo.Offset, DisplacementMapBias::AfterInsertedBytes,
                      NewPcOffset))
    return false;
  std::optional<uint64_t> NewPcBase = checkedAddUint64(
      Elf.textAddr(), NewPcOffset, "legacy s_get_pc displaced PC base");
  if (!NewPcBase)
    return false;
  const uint64_t NewDelta = *NewTargetOrErr - *NewPcBase;
  const uint32_t NewLo = static_cast<uint32_t>(NewDelta);
  const uint32_t NewHi = static_cast<uint32_t>(NewDelta >> 32);

  if (NewLo == static_cast<uint32_t>(OldLo) &&
      NewHi == static_cast<uint32_t>(OldHi)) {
    RepairedPcSensitiveOffsets.insert(SetPc.Offset);
    return true;
  }

  uint64_t NewAddHiOffset = 0;
  if (!Plan.mapOffset(AddHi.Offset, DisplacementMapBias::AfterInsertedBytes,
                      NewAddHiOffset))
    return false;
  std::string LoContext = ("s_add_co_u32 for s_get_pc at old .text offset 0x" +
                           Twine::utohexstr(GetPc.Offset))
                              .str();
  std::string HiContext =
      ("s_add_co_ci_u32 for s_get_pc at old .text offset 0x" +
       Twine::utohexstr(GetPc.Offset))
          .str();

  // An inline immediate can become a literal after displacement. Growing the
  // add would move the following half of this indivisible sequence, so reject
  // both repairs before scheduling either one unless MC can preserve their
  // original widths.
  if (AddLo.Size == MinInstSize) {
    Expected<SmallVector<uint8_t>> EncodedOrErr = reencodeAbsoluteOperand(
        AddLo, /*OperandIndex=*/2, static_cast<int64_t>(NewLo),
        /*ForceLiteral=*/false, LS, LoContext);
    if (!EncodedOrErr)
      return EncodedOrErr.takeError();
  }
  if (AddHi.Size == MinInstSize) {
    Expected<SmallVector<uint8_t>> EncodedOrErr = reencodeAbsoluteOperand(
        AddHi, /*OperandIndex=*/2, static_cast<int64_t>(NewHi),
        /*ForceLiteral=*/false, LS, HiContext);
    if (!EncodedOrErr)
      return EncodedOrErr.takeError();
  }

  PendingRepairs.emplace_back(
      AddLo, /*OperandIndex=*/2, static_cast<int64_t>(NewLo), NewPcOffset,
      /*ForceLiteral=*/AddLo.Size > MinInstSize, std::move(LoContext));
  PendingRepairs.emplace_back(
      AddHi, /*OperandIndex=*/2, static_cast<int64_t>(NewHi), NewAddHiOffset,
      /*ForceLiteral=*/AddHi.Size > MinInstSize, std::move(HiContext));
  RepairedPcSensitiveOffsets.insert(SetPc.Offset);
  return true;
}

Expected<bool> repairSAddPcForDisplacement(
    const ElfView &Elf, const LLVMState &LS, const DisplacementPlan &Plan,
    const InternalDecodedInst &AddPc,
    SmallVectorImpl<PendingAbsoluteRepair> &PendingRepairs) {
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
  PendingRepairs.emplace_back(AddPc, /*OperandIndex=*/0, NewDelta, NewAddPcOff,
                              /*ForceLiteral=*/false, std::move(Context));
  return true;
}

Error repairBranches(const ElfView &Elf, const LLVMState &LS,
                     const DisplacementPlan &Plan,
                     SmallVectorImpl<uint8_t> &NewText) {
  if (!LS.MIA) {
    return makeDisplacementError(
        "branch analysis through LLVM MC is unavailable");
  }

  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Elf.textData(), Elf.textSize(), LS, Decoded)) {
    return makeDisplacementError(
        "failed to decode .text while validating branches");
  }

  // A repaired materialization is only valid when every path executes its
  // defining get-PC first. Protect direct and declared entries so no branch,
  // ELF entry, function alias, or kernel descriptor can bypass part of the
  // sequence.
  DenseSet<uint64_t> ProtectedEntries;
  const uint64_t ElfEntry = Elf.file().getHeader().e_entry;
  if (ElfEntry != 0 && ElfEntry >= Elf.textAddr() &&
      ElfEntry - Elf.textAddr() < Elf.textSize())
    ProtectedEntries.insert(ElfEntry - Elf.textAddr());
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges()) {
    if (Range.Begin >= Elf.textAddr() &&
        Range.Begin - Elf.textAddr() < Elf.textSize())
      ProtectedEntries.insert(Range.Begin - Elf.textAddr());
  }
  for (const KernelDescriptorInfo &Descriptor : Elf.kernelDescriptors()) {
    std::optional<uint64_t> Entry = entryVAddr(Descriptor);
    if (!Entry) {
      return makeDisplacementError(
          "failed to resolve kernel entry while protecting PC sequences");
    }
    if (*Entry >= Elf.textAddr() && *Entry - Elf.textAddr() < Elf.textSize())
      ProtectedEntries.insert(*Entry - Elf.textAddr());
  }
  for (const InternalDecodedInst &DI : Decoded) {
    if (!DI.DecodeSucceeded ||
        (!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)))
      continue;
    uint64_t Target = 0;
    if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target) &&
        Target < Elf.textSize())
      ProtectedEntries.insert(Target);
  }

  SmallVector<PendingAbsoluteRepair, 4> PendingRepairs;
  DenseSet<uint64_t> RepairedPcSensitiveOffsets;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (Plan.rangeOverlapsReplacement(DI.Offset, DI.Size))
      continue;
    if (!DI.DecodeSucceeded) {
      return makeDisplacementError(
          "undecodable instruction at old .text offset 0x" +
          Twine::utohexstr(DI.Offset));
    }

    const MCInst &Inst = DI.Inst;
    if (Inst.getOpcode() == LS.SGetPcI64Opcode) {
      Expected<bool> RepairedOrErr = repairSGetPcPairForDisplacement(
          Elf, LS, Plan, DI, ProtectedEntries, PendingRepairs);
      if (!RepairedOrErr)
        return RepairedOrErr.takeError();
      if (*RepairedOrErr)
        continue;
      RepairedOrErr = repairLegacySGetPcSetPcForDisplacement(
          Elf, LS, Plan, Decoded, I, ProtectedEntries, PendingRepairs,
          RepairedPcSensitiveOffsets);
      if (!RepairedOrErr)
        return RepairedOrErr.takeError();
      if (*RepairedOrErr)
        continue;
    }
    if (Inst.getOpcode() == LS.SAddPcI64Opcode) {
      Expected<bool> RepairedOrErr =
          repairSAddPcForDisplacement(Elf, LS, Plan, DI, PendingRepairs);
      if (!RepairedOrErr)
        return RepairedOrErr.takeError();
      if (*RepairedOrErr)
        continue;
    }
    if (RepairedPcSensitiveOffsets.contains(DI.Offset))
      continue;
    if (isPcSensitiveForDisplacement(DI, LS)) {
      StringRef Mnemonic = "<unknown>";
      if (LS.MCIP) {
        std::pair<const char *, uint64_t> Name = LS.MCIP->getMnemonic(DI.Inst);
        if (Name.first)
          Mnemonic = StringRef(Name.first).rtrim();
      }
      return makeDisplacementError(
          "pc-sensitive instruction '" + Twine(Mnemonic) +
          "' at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
          " requires linked-address repair");
    }
    if (LS.MIA->isCall(Inst)) {
      return makeDisplacementError("call at old .text offset 0x" +
                                   Twine::utohexstr(DI.Offset) +
                                   " is not supported by displacement");
    }
    if (LS.MIA->isIndirectBranch(Inst)) {
      return makeDisplacementError("indirect branch at old .text offset 0x" +
                                   Twine::utohexstr(DI.Offset) +
                                   " is not supported by displacement");
    }
    if (!LS.MIA->isBranch(Inst))
      continue;

    uint64_t OldTarget = 0;
    if (!LS.MIA->evaluateBranch(Inst, DI.Offset, DI.Size, OldTarget)) {
      return makeDisplacementError("branch at old .text offset 0x" +
                                   Twine::utohexstr(DI.Offset) +
                                   " target could not be evaluated");
    }
    if (OldTarget >= Elf.textSize()) {
      return makeDisplacementError("branch at old .text offset 0x" +
                                   Twine::utohexstr(DI.Offset) +
                                   " targets outside .text");
    }

    uint64_t NewFrom = 0;
    uint64_t NewTarget = 0;
    if (!Plan.mapOffset(DI.Offset, DisplacementMapBias::AfterInsertedBytes,
                        NewFrom)) {
      return makeDisplacementError("branch source at old .text offset 0x" +
                                   Twine::utohexstr(DI.Offset) +
                                   " maps inside a replaced range");
    }
    if (!Plan.mapOffset(OldTarget, DisplacementMapBias::BeforeInsertedBytes,
                        NewTarget)) {
      return makeDisplacementError("branch target at old .text offset 0x" +
                                   Twine::utohexstr(OldTarget) +
                                   " maps inside a replaced range");
    }

    Expected<SmallVector<uint8_t>> EncodedOrErr =
        reencodePcrelBranch(DI, NewFrom, NewTarget, LS);
    if (!EncodedOrErr)
      return EncodedOrErr.takeError();
    SmallVector<uint8_t> &Encoded = *EncodedOrErr;

    if (NewFrom > NewText.size() || Encoded.size() > NewText.size() - NewFrom) {
      return makeDisplacementError("re-encoded branch at old .text offset 0x" +
                                   Twine::utohexstr(DI.Offset) +
                                   " writes past rebuilt .text");
    }
    std::memcpy(NewText.data() + NewFrom, Encoded.data(), Encoded.size());
  }

  // Forced AMDGPU literals are backend-private target expressions. Rebuilding
  // one through the MC parser resets LS.Ctx, so do that only after the decoded
  // records are no longer inspected.
  for (PendingAbsoluteRepair &Repair : PendingRepairs) {
    Expected<SmallVector<uint8_t>> CodeOrErr =
        reencodeAbsoluteOperand(Repair.DI, Repair.OperandIndex, Repair.NewValue,
                                Repair.ForceLiteral, LS, Repair.Context);
    if (!CodeOrErr)
      return CodeOrErr.takeError();
    SmallVector<uint8_t> &Code = *CodeOrErr;
    if (Repair.NewOffset > NewText.size() ||
        Code.size() > NewText.size() - Repair.NewOffset) {
      return makeDisplacementError(Repair.Context +
                                   " writes past rebuilt .text");
    }
    std::memcpy(NewText.data() + Repair.NewOffset, Code.data(), Code.size());
  }
  return Error::success();
}

Error adjustSectionHeadersForTextGrowth(uint8_t *Elf, size_t ElfSize,
                                        const ElfView &OldElf, size_t Growth,
                                        bool RelocateAddresses) {
  if (ElfSize < sizeof(Ehdr)) {
    return makeDisplacementError(
        "displaced ELF is smaller than its ELF64 header");
  }

  const uint64_t TextOffset = OldElf.textOffset();
  const uint64_t TextSize = OldElf.textSize();
  const uint64_t TextEnd = TextOffset + TextSize;
  const uint64_t TextAddressEnd = OldElf.textAddr() + TextSize;

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
    if (Shoff > std::numeric_limits<uint64_t>::max() - Growth)
      return makeDisplacementError(
          "section-header offset overflows after displacement");
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
    const ELFT::Shdr &OldShdr = OldElf.sections()[I];

    if (I == OldElf.textSectionIndex()) {
      if (TextSize > std::numeric_limits<uint64_t>::max() - Growth)
        return makeDisplacementError(".text size overflows after displacement");
      uint64_t NewTextSize = TextSize + Growth;
      std::memcpy(Sh + offsetof(Shdr, sh_size), &NewTextSize,
                  sizeof(NewTextSize));
      continue;
    }

    if (OldShdr.sh_offset >= TextEnd) {
      if (OldShdr.sh_offset > std::numeric_limits<uint64_t>::max() - Growth)
        return makeDisplacementError(
            "section file offset overflows after displacement");
      uint64_t NewOffset = OldShdr.sh_offset + Growth;
      std::memcpy(Sh + offsetof(Shdr, sh_offset), &NewOffset,
                  sizeof(NewOffset));
    }
    if (RelocateAddresses && (OldShdr.sh_flags & ELF::SHF_ALLOC) &&
        OldShdr.sh_addr >= TextAddressEnd) {
      if (OldShdr.sh_addr > std::numeric_limits<uint64_t>::max() - Growth)
        return makeDisplacementError(
            "section virtual address overflows after displacement");
      uint64_t NewAddress = OldShdr.sh_addr + Growth;
      std::memcpy(Sh + offsetof(Shdr, sh_addr), &NewAddress,
                  sizeof(NewAddress));
    }
  }
  return Error::success();
}

Error adjustProgramHeadersForTextGrowth(uint8_t *Elf, size_t ElfSize,
                                        const ElfView &OldElf, size_t Growth,
                                        bool RelocateAddresses) {
  if (ElfSize < sizeof(Ehdr)) {
    return makeDisplacementError(
        "displaced ELF is smaller than its ELF64 header");
  }

  const uint64_t TextOffset = OldElf.textOffset();
  const uint64_t TextEnd = TextOffset + OldElf.textSize();
  const uint64_t TextAddressEnd = OldElf.textAddr() + OldElf.textSize();

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
    if (Phoff > std::numeric_limits<uint64_t>::max() - Growth)
      return makeDisplacementError(
          "program-header offset overflows after displacement");
    uint64_t NewPhoff = Phoff + Growth;
    std::memcpy(Elf + offsetof(Ehdr, e_phoff), &NewPhoff, sizeof(NewPhoff));
    Phoff = NewPhoff;
  }

  Expected<ELFT::PhdrRange> OldPhdrsOrErr = OldElf.file().program_headers();
  if (!OldPhdrsOrErr) {
    return makeDisplacementError(
        "failed to read old program headers during displacement: " +
        Twine(toString(OldPhdrsOrErr.takeError())));
  }
  ELFT::PhdrRange OldPhdrs = *OldPhdrsOrErr;
  for (uint16_t I = 0; I < Phnum; ++I) {
    uint64_t PhPos = Phoff + static_cast<uint64_t>(I) * Phentsize;
    if (PhPos > ElfSize || sizeof(Phdr) > ElfSize - PhPos) {
      return makeDisplacementError(
          "displaced ELF program-header table is out of bounds");
    }
    uint8_t *Ph = Elf + PhPos;
    const ELFT::Phdr &OldPhdr = OldPhdrs[I];

    if (OldPhdr.p_offset <= TextOffset &&
        OldPhdr.p_filesz <=
            std::numeric_limits<uint64_t>::max() - OldPhdr.p_offset &&
        OldPhdr.p_offset + OldPhdr.p_filesz >= TextEnd) {
      if (OldPhdr.p_filesz > std::numeric_limits<uint64_t>::max() - Growth)
        return makeDisplacementError(
            "program-header file size overflows after displacement");
      uint64_t NewFileSize = OldPhdr.p_filesz + Growth;
      uint64_t NewMemorySize = std::max<uint64_t>(OldPhdr.p_memsz, NewFileSize);
      if (RelocateAddresses) {
        if (OldPhdr.p_memsz > std::numeric_limits<uint64_t>::max() - Growth)
          return makeDisplacementError(
              "program-header memory size overflows after displacement");
        NewMemorySize = OldPhdr.p_memsz + Growth;
      }
      std::memcpy(Ph + offsetof(Phdr, p_filesz), &NewFileSize,
                  sizeof(NewFileSize));
      std::memcpy(Ph + offsetof(Phdr, p_memsz), &NewMemorySize,
                  sizeof(NewMemorySize));
    } else if (OldPhdr.p_offset >= TextEnd) {
      if (OldPhdr.p_offset > std::numeric_limits<uint64_t>::max() - Growth)
        return makeDisplacementError(
            "program-header file offset overflows after displacement");
      uint64_t NewOffset = OldPhdr.p_offset + Growth;
      std::memcpy(Ph + offsetof(Phdr, p_offset), &NewOffset, sizeof(NewOffset));
    }

    if (RelocateAddresses && OldPhdr.p_memsz != 0 &&
        OldPhdr.p_vaddr >= TextAddressEnd) {
      if (OldPhdr.p_vaddr > std::numeric_limits<uint64_t>::max() - Growth)
        return makeDisplacementError(
            "program-header virtual address overflows after displacement");
      uint64_t NewVirtualAddress = OldPhdr.p_vaddr + Growth;
      std::memcpy(Ph + offsetof(Phdr, p_vaddr), &NewVirtualAddress,
                  sizeof(NewVirtualAddress));
      if (OldPhdr.p_paddr != 0) {
        if (OldPhdr.p_paddr > std::numeric_limits<uint64_t>::max() - Growth)
          return makeDisplacementError(
              "program-header physical address overflows after displacement");
        uint64_t NewPhysicalAddress = OldPhdr.p_paddr + Growth;
        std::memcpy(Ph + offsetof(Phdr, p_paddr), &NewPhysicalAddress,
                    sizeof(NewPhysicalAddress));
      }
    }
  }
  return Error::success();
}

Error adjustElfEntryForDisplacement(uint8_t *Elf, size_t ElfSize,
                                    const ElfView &OldElf,
                                    const DisplacementPlan &Plan) {
  if (ElfSize < sizeof(Ehdr))
    return makeDisplacementError(
        "displaced ELF is too small to repair e_entry");
  const uint64_t OldEntry = OldElf.file().getHeader().e_entry;
  if (OldEntry == 0)
    return Error::success();
  Expected<uint64_t> NewEntryOrErr =
      remapAllocatedAddress(OldElf, Plan, OldEntry,
                            /*RequireExecutable=*/true, "ELF e_entry");
  if (!NewEntryOrErr)
    return NewEntryOrErr.takeError();
  std::memcpy(Elf + offsetof(Ehdr, e_entry), &*NewEntryOrErr,
              sizeof(*NewEntryOrErr));
  return Error::success();
}

template <typename RecordT>
Expected<uint64_t> getRecordOffset(const ELFFileT &File,
                                   const RecordT &Record) {
  const uint8_t *Bytes = reinterpret_cast<const uint8_t *>(&Record);
  if (Bytes < File.base() || Bytes > File.end() ||
      static_cast<size_t>(File.end() - Bytes) < sizeof(RecordT)) {
    return makeDisplacementError(
        "dynamic record is outside displaced ELF buffer");
  }
  return static_cast<uint64_t>(Bytes - File.base());
}

Error adjustDynamicEntriesForDisplacement(uint8_t *Elf, size_t ElfSize,
                                          const ElfView &OldElf,
                                          const DisplacementPlan &Plan) {
  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Elf), ElfSize));
  if (!FileOrErr) {
    return makeDisplacementError(
        "failed to parse displaced ELF for dynamic-tag repair: " +
        Twine(toString(FileOrErr.takeError())));
  }
  ELFFileT File = std::move(*FileOrErr);
  Expected<ELFT::DynRange> EntriesOrErr = File.dynamicEntries();
  if (!EntriesOrErr) {
    return makeDisplacementError(
        "failed to read dynamic tags during displacement: " +
        Twine(toString(EntriesOrErr.takeError())));
  }

  for (const ELFT::Dyn &Entry : *EntriesOrErr) {
    const int64_t Tag = static_cast<int64_t>(Entry.d_tag);
    DynamicTagClass Class = classifyDynamicTag(Tag);
    if (Class == DynamicTagClass::Unknown) {
      return makeDisplacementError("unknown dynamic tag 0x" +
                                   Twine::utohexstr(Entry.d_tag) +
                                   " may carry an unclassified address");
    }
    if (Class == DynamicTagClass::UnsupportedAddress) {
      return makeDisplacementError(
          "dynamic tag 0x" + Twine::utohexstr(Entry.d_tag) +
          " introduces an unsupported address-bearing construct");
    }
    if (Entry.d_un.d_val != 0 &&
        (Tag == ELF::DT_PLTRELSZ || Tag == ELF::DT_PLTREL ||
         Tag == ELF::DT_INIT_ARRAYSZ || Tag == ELF::DT_FINI_ARRAYSZ ||
         Tag == ELF::DT_PREINIT_ARRAYSZ || Tag == ELF::DT_RELRSZ ||
         Tag == ELF::DT_RELRENT)) {
      return makeDisplacementError(
          "dynamic tag enables an unsupported pointer/relocation table");
    }
    if (Class != DynamicTagClass::Address || Entry.d_un.d_ptr == 0)
      continue;

    Expected<const ELFT::Shdr *> OwnerOrErr = findUniqueAllocatedSection(
        OldElf, Entry.d_un.d_ptr, /*RequireExecutable=*/false,
        "dynamic-tag address");
    if (!OwnerOrErr)
      return OwnerOrErr.takeError();
    if (!dynamicTagMatchesSection(Tag, (**OwnerOrErr).sh_type)) {
      return makeDisplacementError(
          "dynamic tag points to a section of the wrong type");
    }
    Expected<uint64_t> NewAddressOrErr = remapAllocatedAddress(
        OldElf, Plan, Entry.d_un.d_ptr, /*RequireExecutable=*/false,
        "dynamic-tag address");
    if (!NewAddressOrErr)
      return NewAddressOrErr.takeError();
    if (*NewAddressOrErr == Entry.d_un.d_ptr)
      continue;

    Expected<uint64_t> EntryOffsetOrErr = getRecordOffset(File, Entry);
    if (!EntryOffsetOrErr)
      return EntryOffsetOrErr.takeError();
    ELFT::Dyn NewEntry = Entry;
    NewEntry.d_un.d_ptr = *NewAddressOrErr;
    std::memcpy(Elf + *EntryOffsetOrErr, &NewEntry, sizeof(NewEntry));
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
        // ELF defines the representation; do not guess it from the numeric
        // value. ET_REL symbols are section-relative, while linked ET_EXEC and
        // ET_DYN symbols carry virtual addresses.
        const bool IsSectionRelative = File.getHeader().e_type == ELF::ET_REL;
        if (!IsSectionRelative &&
            (Sym.st_value < DefShdr.sh_addr ||
             Sym.st_value - DefShdr.sh_addr > Plan.oldTextSize())) {
          return makeDisplacementError(
              "linked text symbol value is outside the original .text "
              "section");
        }
        const uint64_t OldOffset =
            IsSectionRelative ? Sym.st_value : Sym.st_value - DefShdr.sh_addr;
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
        if (!IsSectionRelative &&
            NewOffset >
                std::numeric_limits<uint64_t>::max() - DefShdr.sh_addr) {
          return makeDisplacementError(
              "linked text symbol value overflows after displacement");
        }
        const uint64_t NewValue =
            IsSectionRelative ? NewOffset : DefShdr.sh_addr + NewOffset;
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

      // The default mode keeps non-text virtual addresses immutable. In
      // whole-object mode, keep symbols in a relocated allocated section at
      // the same section-relative byte.
      if (Plan.relocatesTrailingSections() &&
          OldElf.file().getHeader().e_type != ELF::ET_REL &&
          Sym.st_shndx < OldElf.sections().size()) {
        const ELFT::Shdr &OldDefShdr = OldElf.sections()[Sym.st_shndx];
        const uint64_t OldTextEnd = OldElf.textAddr() + OldElf.textSize();
        if ((OldDefShdr.sh_flags & ELF::SHF_ALLOC) &&
            OldDefShdr.sh_addr >= OldTextEnd) {
          if (Sym.st_value < OldDefShdr.sh_addr ||
              Sym.st_value - OldDefShdr.sh_addr > OldDefShdr.sh_size) {
            return makeDisplacementError(
                "symbol in a relocated allocated section is outside its "
                "defining section");
          }
          if (Sym.st_value >
              std::numeric_limits<uint64_t>::max() - Plan.paddedGrowth()) {
            return makeDisplacementError(
                "non-text symbol value overflows after displacement");
          }
          const uint64_t NewValue = Sym.st_value + Plan.paddedGrowth();
          std::memcpy(Elf + SymOffset + offsetof(ELFT::Sym, st_value),
                      &NewValue, sizeof(NewValue));
        }
      }
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

  const bool RelocateAddresses = Plan.relocatesTrailingSections();
  if (Error Err = adjustSectionHeadersForTextGrowth(
          Out, NewSize, Elf, Plan.paddedGrowth(), RelocateAddresses))
    return Err;
  if (Error Err = adjustProgramHeadersForTextGrowth(
          Out, NewSize, Elf, Plan.paddedGrowth(), RelocateAddresses))
    return Err;
  if (Error Err = adjustElfEntryForDisplacement(Out, NewSize, Elf, Plan))
    return Err;
  if (RelocateAddresses) {
    if (Error Err =
            adjustDynamicEntriesForDisplacement(Out, NewSize, Elf, Plan)) {
      return Err;
    }
  }
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
                         ArrayRef<DisplacementEdit> InputEdits,
                         bool RelocateTrailingSections) {
  if (InputEdits.empty())
    return makeDisplacementError("no displacement edits requested");
  if (Error Err = validateSupportedElfNumbering(Elf))
    return std::move(Err);

  std::vector<DisplacementEdit> Sorted;
  Sorted.reserve(InputEdits.size());
  for (const DisplacementEdit &Edit : InputEdits)
    Sorted.push_back(Edit);

  std::stable_sort(Sorted.begin(), Sorted.end(),
                   [](const DisplacementEdit &A, const DisplacementEdit &B) {
                     if (A.Offset != B.Offset)
                       return A.Offset < B.Offset;
                     return A.MapsOldOffsetAfterInsertion &&
                            !B.MapsOldOffsetAfterInsertion;
                   });

  uint64_t RawGrowth = 0;
  std::optional<uint64_t> PrevOffset;
  const DisplacementEdit *PrevEdit = nullptr;
  uint64_t PrevEnd = 0;
  for (const DisplacementEdit &Edit : Sorted) {
    if (Edit.ReplacementBytes.empty())
      return makeDisplacementError("displacement edit has empty replacement");
    if (Edit.MapsOldOffsetAfterInsertion && Edit.OriginalSize != 0) {
      return makeDisplacementError(
          "boundary-after displacement edit is not a pure insertion");
    }
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
      const bool FollowsBoundaryAfterInsertion =
          PrevEdit && PrevEdit->MapsOldOffsetAfterInsertion &&
          !Edit.MapsOldOffsetAfterInsertion && PrevEdit->OriginalSize == 0;
      if (!FollowsBoundaryAfterInsertion) {
        return makeDisplacementError(
            "multiple displacement edits share an offset");
      }
    }

    uint64_t EditGrowth = Edit.ReplacementBytes.size() - Edit.OriginalSize;
    if (EditGrowth > std::numeric_limits<uint64_t>::max() - RawGrowth)
      return makeDisplacementError("displacement growth overflows uint64_t");
    RawGrowth += EditGrowth;
    PrevOffset = Edit.Offset;
    PrevEdit = &Edit;
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
  if (RelocateTrailingSections) {
    if (Elf.textSize() >
            std::numeric_limits<uint64_t>::max() - Elf.textAddr() ||
        PaddedGrowth > std::numeric_limits<uint64_t>::max() - Elf.textAddr() -
                           Elf.textSize()) {
      return makeDisplacementError("displaced .text virtual end overflows");
    }
    if (Error Err = validateTrailingRelocationLayout(Elf))
      return std::move(Err);
  } else {
    if (Error Err = validateVirtualGrowth(Elf, PaddedGrowth))
      return std::move(Err);
  }
  if (PaddedGrowth > std::numeric_limits<size_t>::max() - Elf.size())
    return makeDisplacementError("displaced ELF size overflows size_t");
  return DisplacementPlan(Elf.textSize(), RawGrowth, PaddedGrowth,
                          std::move(Sorted), RelocateTrailingSections);
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
          (Edit.MapsOldOffsetAfterInsertion ||
           Bias == DisplacementMapBias::AfterInsertedBytes)) {
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

static Expected<std::vector<DisplacementEdit>>
addKernelEntryAlignmentEdits(const ElfView &Elf, const LLVMState &LS,
                             ArrayRef<DisplacementEdit> Edits) {
  if (LS.SNopBytes.size() != MinInstSize) {
    return makeDisplacementError(
        "kernel-entry alignment requires one encoded s_nop");
  }

  Expected<DisplacementPlan> InitialPlanOrErr =
      DisplacementPlan::create(Elf, Edits,
                               /*RelocateTrailingSections=*/true);
  if (!InitialPlanOrErr)
    return InitialPlanOrErr.takeError();

  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr()) {
    return makeDisplacementError(
        ".text range overflows while aligning kernel entries");
  }
  const uint64_t TextEnd = Elf.textAddr() + Elf.textSize();
  SmallVector<uint64_t, 8> KernelEntries;
  for (const KernelDescriptorInfo &Descriptor : Elf.kernelDescriptors()) {
    std::optional<uint64_t> EntryAddress = entryVAddr(Descriptor);
    if (!EntryAddress) {
      return makeDisplacementError("failed to resolve kernel entry for '" +
                                   Twine(Descriptor.KernelName) + "'");
    }
    if (*EntryAddress < Elf.textAddr() || *EntryAddress >= TextEnd)
      continue;
    KernelEntries.push_back(*EntryAddress - Elf.textAddr());
  }
  llvm::sort(KernelEntries);
  KernelEntries.erase(std::unique(KernelEntries.begin(), KernelEntries.end()),
                      KernelEntries.end());

  std::vector<DisplacementEdit> Result(Edits.begin(), Edits.end());
  uint64_t AlignmentDelta = 0;
  unsigned AlignmentEditCount = 0;
  for (uint64_t Entry : KernelEntries) {
    uint64_t MappedOffset = 0;
    if (!InitialPlanOrErr->mapOffset(
            Entry, DisplacementMapBias::BeforeInsertedBytes, MappedOffset)) {
      return makeDisplacementError(
          "kernel entry maps inside a replaced instruction");
    }
    std::optional<uint64_t> MappedAddress =
        checkedAddUint64(Elf.textAddr(), MappedOffset,
                         "displaced kernel entry before alignment");
    if (!MappedAddress) {
      return makeDisplacementError("kernel entry overflows before alignment");
    }
    MappedAddress = checkedAddUint64(*MappedAddress, AlignmentDelta,
                                     "displaced kernel entry alignment delta");
    if (!MappedAddress) {
      return makeDisplacementError("kernel entry alignment delta overflows");
    }
    std::optional<uint64_t> Aligned =
        checkedAlignTo(*MappedAddress, KernelEntryStubStride,
                       "displaced kernel entry alignment");
    if (!Aligned)
      return makeDisplacementError("kernel entry alignment overflows");
    const uint64_t Padding = *Aligned - *MappedAddress;
    if (Padding == 0)
      continue;
    if (Padding % MinInstSize != 0 ||
        Padding > std::numeric_limits<uint64_t>::max() - AlignmentDelta) {
      return makeDisplacementError(
          "kernel entry requires invalid alignment padding");
    }

    DisplacementEdit Alignment;
    Alignment.Offset = Entry;
    Alignment.MapsOldOffsetAfterInsertion = true;
    for (uint64_t I = 0; I != Padding; I += MinInstSize) {
      Alignment.ReplacementBytes.append(LS.SNopBytes.begin(),
                                        LS.SNopBytes.end());
    }
    Result.push_back(std::move(Alignment));
    AlignmentDelta += Padding;
    ++AlignmentEditCount;
  }

  if (AlignmentEditCount != 0) {
    log() << "hotswap: displacement: added " << AlignmentEditCount
          << " kernel-entry alignment insertion"
          << (AlignmentEditCount == 1 ? "" : "s") << "\n";
  }
  return Result;
}

Expected<std::unique_ptr<WritableMemoryBuffer>>
tryApplyTextDisplacementToNewBuffer(const ElfView &Elf, const LLVMState &LS,
                                    ArrayRef<DisplacementEdit> Edits,
                                    bool RelocateTrailingSections) {
  if (Error Err = validateDebugSections(Elf))
    return std::move(Err);
  if (Error Err = validateTextRelocations(Elf))
    return std::move(Err);

  std::vector<DisplacementEdit> PlannedEdits(Edits.begin(), Edits.end());
  if (RelocateTrailingSections) {
    Expected<std::vector<DisplacementEdit>> AlignedOrErr =
        addKernelEntryAlignmentEdits(Elf, LS, Edits);
    if (!AlignedOrErr)
      return AlignedOrErr.takeError();
    PlannedEdits = std::move(*AlignedOrErr);
  }

  Expected<DisplacementPlan> PlanOrErr =
      DisplacementPlan::create(Elf, PlannedEdits, RelocateTrailingSections);
  if (!PlanOrErr)
    return PlanOrErr.takeError();
  if (Error Err = validateEditInstructionBoundaries(Elf, LS, *PlanOrErr))
    return std::move(Err);
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
