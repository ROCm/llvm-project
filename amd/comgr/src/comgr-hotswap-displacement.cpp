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
/// new encoding, and updates ELF metadata. Objects with an address-bearing
/// construct outside this layer's proof and repair set fail closed.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <chrono>
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

// The non-allocatable DWARF debug sections whose .text addresses
// remapDebugSectionsForDisplacement rewrites after whole-object growth.
bool isAddressBearingDebugSection(StringRef Name) {
  return Name == ".debug_frame" || Name == ".debug_info" ||
         Name == ".debug_ranges" || Name == ".debug_line";
}

// Debug sections that carry NO .text addresses and are therefore safe to leave
// byte-identical across whole-object growth (only their file offset shifts,
// which the section-header adjust already handles).
// .debug_abbrev/.debug_str/.debug_line_str are string/structure tables;
// .debug_str_offsets/.debug_addr are index tables that do not encode raw .text
// VADDRs for these DWARF<=4 objects.
bool isAddressFreeDebugSection(StringRef Name) {
  return Name == ".debug_abbrev" || Name == ".debug_str" ||
         Name == ".debug_line_str";
}

// A .debug section this pass can safely admit into whole-object growth: either
// it has no .text addresses, or we remap the ones it has. Any other
// .debug*/.zdebug* section fails closed in validateDebugSections so we never
// ship stale .text addresses.
bool isRemappableDebugSection(StringRef Name) {
  return isAddressBearingDebugSection(Name) || isAddressFreeDebugSection(Name);
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
    if (Owner)
      return makeDisplacementError(Twine(Context) +
                                   " is covered by overlapping allocated "
                                   "sections");
    Owner = &Shdr;
  }
  if (!Owner)
    return makeDisplacementError(Twine(Context) +
                                 " is outside every allocated section");
  if (RequireExecutable && !(Owner->sh_flags & ELF::SHF_EXECINSTR))
    return makeDisplacementError(Twine(Context) +
                                 " is not in an executable section");
  return Owner;
}

Expected<uint64_t> remapAllocatedAddress(const ElfView &Elf,
                                         const DisplacementPlan &Plan,
                                         uint64_t OldAddress,
                                         bool RequireExecutable,
                                         StringRef Context) {
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr())
    return makeDisplacementError(Twine(Context) +
                                 " cannot resolve an overflowing .text");
  const uint64_t OldTextEnd = Elf.textAddr() + Elf.textSize();
  if (OldAddress >= Elf.textAddr() && OldAddress < OldTextEnd) {
    if (RequireExecutable &&
        !(Elf.textSection()->sh_flags & ELF::SHF_EXECINSTR))
      return makeDisplacementError(Twine(Context) +
                                   " is not in an executable section");
    uint64_t NewOffset = 0;
    if (!Plan.mapOffset(OldAddress - Elf.textAddr(),
                        DisplacementMapBias::BeforeInsertedBytes, NewOffset))
      return makeDisplacementError(Twine(Context) +
                                   " maps inside a replaced instruction");
    if (NewOffset > std::numeric_limits<uint64_t>::max() - Elf.textAddr())
      return makeDisplacementError(Twine(Context) +
                                   " overflows after .text displacement");
    return Elf.textAddr() + NewOffset;
  }

  Expected<const ELFT::Shdr *> OwnerOrErr =
      findUniqueAllocatedSection(Elf, OldAddress, RequireExecutable, Context);
  if (!OwnerOrErr)
    return OwnerOrErr.takeError();
  const ELFT::Shdr &Owner = **OwnerOrErr;
  if (Owner.sh_addr < OldTextEnd || !Plan.relocatesTrailingSections())
    return OldAddress;
  if (OldAddress > std::numeric_limits<uint64_t>::max() - Plan.paddedGrowth())
    return makeDisplacementError(Twine(Context) +
                                 " overflows after section displacement");
  return OldAddress + Plan.paddedGrowth();
}

Expected<uint64_t> remapLoadAddress(const ElfView &Elf,
                                    const DisplacementPlan &Plan,
                                    uint64_t OldAddress, StringRef Context) {
  if (OldAddress == 0)
    return uint64_t{0};
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr())
    return makeDisplacementError(Twine(Context) +
                                 " cannot resolve an overflowing .text");
  const uint64_t OldTextEnd = Elf.textAddr() + Elf.textSize();
  if (OldAddress >= Elf.textAddr() && OldAddress < OldTextEnd)
    return remapAllocatedAddress(Elf, Plan, OldAddress,
                                 /*RequireExecutable=*/true, Context);

  Expected<ELFT::PhdrRange> PhdrsOrErr = Elf.file().program_headers();
  if (!PhdrsOrErr)
    return makeDisplacementError(Twine(Context) +
                                 " cannot read program headers: " +
                                 Twine(toString(PhdrsOrErr.takeError())));
  const ELFT::Phdr *Owner = nullptr;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_LOAD || Phdr.p_memsz == 0 ||
        Phdr.p_memsz > std::numeric_limits<uint64_t>::max() - Phdr.p_vaddr)
      continue;
    if (OldAddress < Phdr.p_vaddr || OldAddress - Phdr.p_vaddr >= Phdr.p_memsz)
      continue;
    if (Owner)
      return makeDisplacementError(
          Twine(Context) + " is covered by overlapping PT_LOAD segments");
    Owner = &Phdr;
  }
  if (!Owner)
    return makeDisplacementError(Twine(Context) +
                                 " is outside every PT_LOAD segment");
  if (Owner->p_vaddr >= OldTextEnd && Plan.relocatesTrailingSections()) {
    if (OldAddress > std::numeric_limits<uint64_t>::max() - Plan.paddedGrowth())
      return makeDisplacementError(Twine(Context) +
                                   " overflows after segment displacement");
    return OldAddress + Plan.paddedGrowth();
  }
  if (Owner->p_vaddr >= OldTextEnd ||
      Owner->p_vaddr + Owner->p_memsz <= Elf.textAddr() ||
      OldAddress < Elf.textAddr())
    return OldAddress;
  return makeDisplacementError(
      Twine(Context) + " points into unclassified post-.text segment padding");
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
  case ELF::SHT_RELA:
  case ELF::SHT_HASH:
  case ELF::SHT_DYNAMIC:
  case ELF::SHT_NOTE:
  case ELF::SHT_NOBITS:
  case ELF::SHT_REL:
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

Error validateMetadataNotes(const ElfView &Elf) {
  StringMap<bool> MetadataKernels;
  bool SawMetadata = false;
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (Shdr.sh_type != ELF::SHT_NOTE)
      continue;
    Error Err = Error::success();
    for (ELFT::Note Note : Elf.file().notes(Shdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        return makeDisplacementError(
            "allocated note is not address-free AMDGPU metadata");
      SawMetadata = true;
      ArrayRef<uint8_t> Desc =
          Note.getDesc(std::max<uint64_t>(Shdr.sh_addralign, uint64_t{4}));
      if (Desc.empty())
        return makeDisplacementError("AMDGPU metadata note is empty");
      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false))
        return makeDisplacementError("AMDGPU metadata note is malformed");
      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap())
        return makeDisplacementError("AMDGPU metadata root is not a map");
      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator Kernels =
          RootMap.find("amdhsa.kernels");
      if (Kernels == RootMap.end() || !Kernels->second.isArray())
        continue;
      for (msgpack::DocNode &Kernel : Kernels->second.getArray()) {
        if (!Kernel.isMap())
          return makeDisplacementError(
              "AMDGPU metadata kernel entry is not a map");
        msgpack::MapDocNode &KernelMap = Kernel.getMap();
        msgpack::DocNode::MapTy::iterator Name = KernelMap.find(".name");
        msgpack::DocNode::MapTy::iterator Symbol = KernelMap.find(".symbol");
        if (Name == KernelMap.end() || !Name->second.isString() ||
            Symbol == KernelMap.end() || !Symbol->second.isString())
          return makeDisplacementError(
              "AMDGPU metadata kernel lacks string .name/.symbol fields");
        StringRef KernelName = Name->second.getString();
        if (Symbol->second.getString() != (Twine(KernelName) + ".kd").str())
          return makeDisplacementError(
              "AMDGPU metadata kernel symbol does not name its descriptor");
        if (!MetadataKernels.try_emplace(KernelName, true).second)
          return makeDisplacementError("AMDGPU metadata repeats a kernel name");
      }
    }
    if (Err)
      return makeDisplacementError(
          "failed to iterate an allocated note section: " +
          Twine(toString(std::move(Err))));
  }

  if (!SawMetadata)
    return Error::success();
  for (const KernelDescriptorInfo &Descriptor : Elf.kernelDescriptors())
    if (!MetadataKernels.contains(Descriptor.KernelName))
      return makeDisplacementError(
          "kernel descriptor is absent from AMDGPU metadata");
  for (const StringMapEntry<bool> &Kernel : MetadataKernels)
    if (!Elf.getKernelDescriptorVAddr(Kernel.first()))
      return makeDisplacementError(
          "AMDGPU metadata names a missing kernel descriptor");
  return Error::success();
}

Error validateWholeObjectLayout(const ElfView &Elf) {
  if (Elf.file().getHeader().e_type != ELF::ET_DYN)
    return makeDisplacementError(
        "whole-object displacement requires a linked ET_DYN code object");
  if (!(Elf.textSection()->sh_flags & ELF::SHF_ALLOC) ||
      !(Elf.textSection()->sh_flags & ELF::SHF_EXECINSTR))
    return makeDisplacementError(
        ".text must be an allocated executable section");
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr() ||
      Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textOffset())
    return makeDisplacementError(".text range overflows");
  const uint64_t TextAddrEnd = Elf.textAddr() + Elf.textSize();
  const uint64_t TextFileEnd = Elf.textOffset() + Elf.textSize();

  SmallVector<const ELFT::Shdr *, 16> AllocatedSections;
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    if (Shdr.sh_addralign > 1 && !isPowerOf2_64(Shdr.sh_addralign))
      return makeDisplacementError("section alignment is not a power of two");
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
        Shdr.sh_type == ELF::SHT_LLVM_CALL_GRAPH)
      return makeDisplacementError(
          "code object has an unsupported address-bearing section type");
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) || Shdr.sh_size == 0)
      continue;
    if (!isKnownAllocatedSectionType(Shdr.sh_type))
      return makeDisplacementError(
          "allocated section type is outside the displacement whitelist");
    if (Shdr.sh_size > std::numeric_limits<uint64_t>::max() - Shdr.sh_addr)
      return makeDisplacementError("allocated section address range overflows");
    AllocatedSections.push_back(&Shdr);
    if (&Shdr == Elf.textSection())
      continue;
    if (Shdr.sh_flags & ELF::SHF_EXECINSTR)
      return makeDisplacementError(
          "whole-object displacement supports one executable section");
    const bool AddressBefore = Shdr.sh_addr + Shdr.sh_size <= Elf.textAddr();
    const bool AddressAfter = Shdr.sh_addr >= TextAddrEnd;
    if (!AddressBefore && !AddressAfter)
      return makeDisplacementError(
          "allocated section overlaps .text in virtual memory");

    if (Shdr.sh_type != ELF::SHT_NOBITS) {
      if (Shdr.sh_size > std::numeric_limits<uint64_t>::max() - Shdr.sh_offset)
        return makeDisplacementError("allocated section file range overflows");
      const bool FileBefore = Shdr.sh_offset + Shdr.sh_size <= Elf.textOffset();
      const bool FileAfter = Shdr.sh_offset >= TextFileEnd;
      if ((!FileBefore && !FileAfter) || FileBefore != AddressBefore)
        return makeDisplacementError(
            "allocated section file and virtual ordering disagree");
    } else if ((AddressBefore && Shdr.sh_offset > Elf.textOffset()) ||
               (AddressAfter && Shdr.sh_offset < TextFileEnd)) {
      return makeDisplacementError(
          "NOBITS section file and virtual ordering disagree");
    }
  }

  for (size_t I = 0; I != AllocatedSections.size(); ++I) {
    const ELFT::Shdr &LHS = *AllocatedSections[I];
    const uint64_t LHSEnd = LHS.sh_addr + LHS.sh_size;
    for (size_t J = I + 1; J != AllocatedSections.size(); ++J) {
      const ELFT::Shdr &RHS = *AllocatedSections[J];
      const uint64_t RHSEnd = RHS.sh_addr + RHS.sh_size;
      if (LHS.sh_addr < RHSEnd && RHS.sh_addr < LHSEnd)
        return makeDisplacementError(
            "allocated sections overlap in virtual memory");
    }
  }

  for (const ELFT::Shdr &SymShdr : Elf.sections()) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ELFT::SymRange> SymbolsOrErr = Elf.file().symbols(&SymShdr);
    if (!SymbolsOrErr)
      return makeDisplacementError(
          "failed to read symbols while validating absolute addresses: " +
          Twine(toString(SymbolsOrErr.takeError())));
    for (const ELFT::Sym &Symbol : *SymbolsOrErr) {
      if (Symbol.st_shndx != ELF::SHN_ABS || Symbol.st_value == 0)
        continue;
      if (Symbol.getType() == ELF::STT_FUNC)
        return makeDisplacementError(
            "absolute function symbol is outside the displacement map");
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
  if (!PhdrsOrErr)
    return makeDisplacementError(
        "failed to read program headers while validating object layout: " +
        Twine(toString(PhdrsOrErr.takeError())));
  bool FoundTextLoad = false;
  SmallVector<const ELFT::Phdr *, 8> LoadSegments;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_align > 1 && !isPowerOf2_64(Phdr.p_align))
      return makeDisplacementError(
          "program-header alignment is not a power of two");
    if (Phdr.p_type != ELF::PT_LOAD)
      continue;
    if (Phdr.p_filesz > std::numeric_limits<uint64_t>::max() - Phdr.p_offset ||
        Phdr.p_memsz > std::numeric_limits<uint64_t>::max() - Phdr.p_vaddr)
      return makeDisplacementError("PT_LOAD range overflows");
    if (Phdr.p_align != 0 &&
        Phdr.p_offset % Phdr.p_align != Phdr.p_vaddr % Phdr.p_align)
      return makeDisplacementError(
          "PT_LOAD file/virtual alignment congruence is invalid");
    LoadSegments.push_back(&Phdr);

    const uint64_t FileEnd = Phdr.p_offset + Phdr.p_filesz;
    const uint64_t VAddrEnd = Phdr.p_vaddr + Phdr.p_memsz;
    const bool ContainsTextFile =
        Phdr.p_offset <= Elf.textOffset() && FileEnd >= TextFileEnd;
    const bool ContainsTextVAddr =
        Phdr.p_vaddr <= Elf.textAddr() && VAddrEnd >= TextAddrEnd;
    if (ContainsTextFile || ContainsTextVAddr) {
      if (!ContainsTextFile || !ContainsTextVAddr || FileEnd != TextFileEnd)
        return makeDisplacementError(
            ".text is not the last file-backed content in its PT_LOAD");
      FoundTextLoad = true;
      continue;
    }

    const bool FileBefore = FileEnd <= Elf.textOffset();
    const bool FileAfter = Phdr.p_offset >= TextFileEnd;
    const bool VAddrBefore = VAddrEnd <= Elf.textAddr();
    const bool VAddrAfter = Phdr.p_vaddr >= TextAddrEnd;
    if ((!FileBefore && !FileAfter) || (!VAddrBefore && !VAddrAfter) ||
        FileBefore != VAddrBefore)
      return makeDisplacementError(
          "PT_LOAD file and virtual ordering around .text disagree");
  }
  if (!FoundTextLoad)
    return makeDisplacementError(".text is not covered by a PT_LOAD segment");
  for (size_t I = 0; I != LoadSegments.size(); ++I) {
    const ELFT::Phdr &LHS = *LoadSegments[I];
    const uint64_t LHSEnd = LHS.p_vaddr + LHS.p_memsz;
    for (size_t J = I + 1; J != LoadSegments.size(); ++J) {
      const ELFT::Phdr &RHS = *LoadSegments[J];
      const uint64_t RHSEnd = RHS.p_vaddr + RHS.p_memsz;
      if (LHS.p_vaddr < RHSEnd && RHS.p_vaddr < LHSEnd)
        return makeDisplacementError(
            "PT_LOAD segments overlap in virtual memory");
    }
  }
  return validateMetadataNotes(Elf);
}

Expected<uint64_t> requiredGrowthAlignment(const ElfView &Elf) {
  // validateWholeObjectLayout proves every alignment is a power of two.
  // Therefore their maximum is also their least common multiple, and rounding
  // growth to it adds the fewest bytes that preserve every downstream file
  // and virtual-address alignment simultaneously.
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

Error validateDebugSections(const ElfView &Elf, bool AllowEhFrameRemap) {
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    Expected<StringRef> NameOrErr = Elf.file().getSectionName(Shdr);
    if (!NameOrErr) {
      return makeDisplacementError(
          "failed to read section name while checking debug info: " +
          Twine(toString(NameOrErr.takeError())));
    }
    StringRef Name = *NameOrErr;
    // .eh_frame is remapped in place by remapEhFrameForDisplacement when the
    // caller opts into trailing-section relocation. .eh_frame_hdr is a
    // binary-search table over FDEs and would also need rebuilding; block it
    // until that is implemented. Full DWARF (.debug*) stays out of scope.
    if (AllowEhFrameRemap && Name == ".eh_frame")
      continue;
    // When relocating trailing sections,
    // remapDebugSectionsForDisplacement rewrites the .text addresses in the
    // non-allocatable DWARF debug sections it understands
    // (.debug_frame/.debug_info/.debug_ranges/.debug_line). Allow those
    // through; any other .debug*/.zdebug* section (or a construct the remapper
    // cannot handle) still fails closed there. .eh_frame_hdr is a binary-search
    // table over FDEs that would need rebuilding; keep blocking it.
    if (AllowEhFrameRemap && isRemappableDebugSection(Name))
      continue;
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

/// Linked-ELF proof object for an indirect dispatch table. Every slot must be
/// either a symbol-less RELATIVE64 pointer into .text or a literal zero; any
/// other contents make the candidate ineligible.
struct RelocationTableCandidate {
  uint64_t Address = 0;
  uint64_t Size = 0;
  SmallVector<std::optional<uint64_t>, 8> Targets;
  SmallVector<bool, 8> ZeroSlots;
  bool Valid = true;
};

RelocationTableCandidate *
findContainingCandidate(std::vector<RelocationTableCandidate> &Candidates,
                        uint64_t Address) {
  std::vector<RelocationTableCandidate>::iterator It = std::upper_bound(
      Candidates.begin(), Candidates.end(), Address,
      [](uint64_t Value, const RelocationTableCandidate &Candidate) {
        return Value < Candidate.Address;
      });
  if (It == Candidates.begin())
    return nullptr;
  --It;
  return Address - It->Address < It->Size ? &*It : nullptr;
}

Expected<bool> isRuntimeImmutableRange(const ElfView &Elf, uint64_t Address,
                                       uint64_t Size) {
  if (Size == 0 || Size > std::numeric_limits<uint64_t>::max() - Address)
    return false;
  const uint64_t End = Address + Size;
  Expected<ELFT::PhdrRange> PhdrsOrErr = Elf.file().program_headers();
  if (!PhdrsOrErr)
    return PhdrsOrErr.takeError();

  bool Covered = false;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_LOAD || Phdr.p_memsz == 0)
      continue;
    if (Phdr.p_memsz > std::numeric_limits<uint64_t>::max() - Phdr.p_vaddr)
      return makeDisplacementError(
          "PT_LOAD range overflows while proving table immutability");
    const uint64_t SegmentEnd = Phdr.p_vaddr + Phdr.p_memsz;
    if (Address < SegmentEnd && Phdr.p_vaddr < End &&
        (Phdr.p_flags & ELF::PF_W))
      return false;
    Covered |= Phdr.p_vaddr <= Address && SegmentEnd >= End;
  }
  return Covered;
}

template <typename RecordT>
Expected<uint64_t> getRecordOffset(const ELFFileT &File,
                                   const RecordT &Record) {
  const uint8_t *Bytes = reinterpret_cast<const uint8_t *>(&Record);
  if (Bytes < File.base() || Bytes > File.end() ||
      static_cast<size_t>(File.end() - Bytes) < sizeof(RecordT))
    return makeDisplacementError(
        "relocation record is outside displaced ELF buffer");
  return static_cast<uint64_t>(Bytes - File.base());
}

Expected<std::vector<RelocationTableCandidate>>
discoverCompleteRelocationTables(const ElfView &Elf) {
  if (Elf.textSize() > std::numeric_limits<uint64_t>::max() - Elf.textAddr())
    return makeDisplacementError(
        ".text virtual range overflows while discovering function tables");
  std::vector<RelocationTableCandidate> Candidates;
  for (const ELFT::Shdr &SymShdr : Elf.sections()) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ELFT::SymRange> SymsOrErr = Elf.file().symbols(&SymShdr);
    if (!SymsOrErr)
      return SymsOrErr.takeError();
    for (const ELFT::Sym &Sym : *SymsOrErr) {
      if (Sym.getType() != ELF::STT_OBJECT || Sym.st_size == 0 ||
          Sym.st_size % sizeof(uint64_t) != 0 ||
          Sym.st_shndx == ELF::SHN_UNDEF || Sym.st_shndx >= ELF::SHN_LORESERVE)
        continue;
      Expected<const ELFT::Shdr *> DefShdrOrErr =
          Elf.file().getSection(Sym.st_shndx);
      if (!DefShdrOrErr)
        return DefShdrOrErr.takeError();
      const ELFT::Shdr &DefShdr = **DefShdrOrErr;
      if (DefShdr.sh_type == ELF::SHT_NOBITS ||
          !(DefShdr.sh_flags & ELF::SHF_ALLOC) ||
          (DefShdr.sh_flags & (ELF::SHF_WRITE | ELF::SHF_EXECINSTR)))
        continue;
      Expected<bool> ImmutableOrErr =
          isRuntimeImmutableRange(Elf, Sym.st_value, Sym.st_size);
      if (!ImmutableOrErr)
        return ImmutableOrErr.takeError();
      if (!*ImmutableOrErr)
        continue;
      if (Sym.st_value < DefShdr.sh_addr ||
          Sym.st_value - DefShdr.sh_addr > DefShdr.sh_size ||
          Sym.st_size > DefShdr.sh_size - (Sym.st_value - DefShdr.sh_addr))
        continue;
      std::vector<RelocationTableCandidate>::iterator Duplicate =
          llvm::find_if(Candidates, [&](const RelocationTableCandidate &C) {
            return C.Address == Sym.st_value && C.Size == Sym.st_size;
          });
      if (Duplicate != Candidates.end())
        continue;

      const uint64_t ObjectOffset =
          DefShdr.sh_offset + Sym.st_value - DefShdr.sh_addr;
      if (ObjectOffset > Elf.size() || Sym.st_size > Elf.size() - ObjectOffset)
        return makeDisplacementError(
            "function-table object extends outside the ELF image");
      const size_t SlotCount = Sym.st_size / sizeof(uint64_t);
      RelocationTableCandidate Candidate{
          Sym.st_value, Sym.st_size,
          SmallVector<std::optional<uint64_t>, 8>(SlotCount),
          SmallVector<bool, 8>(SlotCount), true};
      for (size_t I = 0; I != SlotCount; ++I) {
        uint64_t Value = 0;
        std::memcpy(&Value, Elf.data() + ObjectOffset + I * sizeof(uint64_t),
                    sizeof(Value));
        Candidate.ZeroSlots[I] = Value == 0;
      }
      Candidates.push_back(std::move(Candidate));
    }
  }

  if (Candidates.empty())
    return Candidates;
  llvm::sort(Candidates, [](const RelocationTableCandidate &LHS,
                            const RelocationTableCandidate &RHS) {
    if (LHS.Address != RHS.Address)
      return LHS.Address < RHS.Address;
    return LHS.Size < RHS.Size;
  });
  // Ambiguous overlapping object symbols are not a sound ownership proof for
  // a relocation slot. Reject both intervals and retain a sorted, disjoint set
  // so every lookup below is O(log N) with one possible owner.
  size_t FarthestEnd = 0;
  for (size_t I = 1; I != Candidates.size(); ++I) {
    uint64_t StartDelta =
        Candidates[I].Address - Candidates[FarthestEnd].Address;
    if (StartDelta < Candidates[FarthestEnd].Size) {
      Candidates[FarthestEnd].Valid = false;
      Candidates[I].Valid = false;
      if (Candidates[I].Size > Candidates[FarthestEnd].Size - StartDelta)
        FarthestEnd = I;
    } else {
      FarthestEnd = I;
    }
  }
  llvm::erase_if(Candidates, [](const RelocationTableCandidate &Candidate) {
    return !Candidate.Valid;
  });
  if (Candidates.empty())
    return Candidates;

  const uint64_t TextEnd = Elf.textAddr() + Elf.textSize();
  for (const ELFT::Shdr &RelocShdr : Elf.sections()) {
    if (RelocShdr.sh_type != ELF::SHT_RELA)
      continue;
    Expected<ELFT::RelaRange> RelasOrErr = Elf.file().relas(RelocShdr);
    if (!RelasOrErr)
      return RelasOrErr.takeError();
    for (const ELFT::Rela &Rela : *RelasOrErr) {
      RelocationTableCandidate *Candidate =
          findContainingCandidate(Candidates, Rela.r_offset);
      if (!Candidate)
        continue;
      const uint64_t SlotOffset = Rela.r_offset - Candidate->Address;
      if (SlotOffset % sizeof(uint64_t) != 0 || Rela.getSymbol(false) != 0 ||
          Rela.getType(false) != ELF::R_AMDGPU_RELATIVE64 ||
          Rela.r_addend < 0) {
        Candidate->Valid = false;
        continue;
      }
      const uint64_t Target = static_cast<uint64_t>(Rela.r_addend);
      if (Target < Elf.textAddr() || Target >= TextEnd) {
        Candidate->Valid = false;
        continue;
      }
      std::optional<uint64_t> &OldTarget =
          Candidate->Targets[SlotOffset / sizeof(uint64_t)];
      const uint64_t TextOffset = Target - Elf.textAddr();
      if (OldTarget && *OldTarget != TextOffset) {
        Candidate->Valid = false;
        continue;
      }
      OldTarget = TextOffset;
    }
  }

  llvm::erase_if(Candidates, [](const RelocationTableCandidate &Candidate) {
    if (!Candidate.Valid)
      return true;
    bool SawTarget = false;
    for (size_t I = 0; I != Candidate.Targets.size(); ++I) {
      if (Candidate.Targets[I]) {
        SawTarget = true;
        continue;
      }
      if (!Candidate.ZeroSlots[I])
        return true;
    }
    return !SawTarget;
  });
  return Candidates;
}

bool definesRegister(const InternalDecodedInst &DI, const LLVMState &LS,
                     MCRegister Register) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned DefCount = std::min(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != DefCount; ++I) {
    const MCOperand &Operand = DI.Inst.getOperand(I);
    if (Operand.isReg() && Operand.getReg() &&
        LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
      return true;
  }
  for (MCPhysReg ImplicitDef : Desc.implicit_defs())
    if (LS.MRI->regsOverlap(MCRegister(ImplicitDef), Register))
      return true;
  return false;
}

bool isProvenanceBoundary(const InternalDecodedInst &DI, const LLVMState &LS) {
  return !DI.DecodeSucceeded || LS.MIA->isBranch(DI.Inst) ||
         LS.MIA->isCall(DI.Inst) || LS.MIA->isReturn(DI.Inst) ||
         LS.MIA->isIndirectBranch(DI.Inst) || LS.MIA->isBarrier(DI.Inst);
}

std::optional<size_t> findDefBefore(ArrayRef<InternalDecodedInst> Decoded,
                                    size_t Before, const LLVMState &LS,
                                    MCRegister Register) {
  for (size_t I = Before; I != 0;) {
    --I;
    const InternalDecodedInst &Candidate = Decoded[I];
    if (isProvenanceBoundary(Candidate, LS))
      return std::nullopt;
    if (definesRegister(Candidate, LS, Register))
      return I;
  }
  return std::nullopt;
}

std::vector<RelocationTableDispatch> matchRelocationTableDispatches(
    const ElfView &Elf, ArrayRef<InternalDecodedInst> Decoded,
    const LLVMState &LS, ArrayRef<RelocationTableCandidate> Tables) {
  std::vector<RelocationTableDispatch> Dispatches;
  DenseSet<uint64_t> KernelEntries;
  for (const KernelDescriptorInfo &Descriptor : Elf.kernelDescriptors()) {
    std::optional<uint64_t> Entry = entryVAddr(Descriptor);
    if (Entry && *Entry >= Elf.textAddr() &&
        *Entry - Elf.textAddr() < Elf.textSize())
      KernelEntries.insert(*Entry - Elf.textAddr());
  }
  for (size_t CallIndex = 0; CallIndex != Decoded.size(); ++CallIndex) {
    const InternalDecodedInst &Call = Decoded[CallIndex];
    if (!Call.DecodeSucceeded || Call.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
        Call.Inst.getNumOperands() < 2 ||
        !Call.Inst.getOperand(Call.Inst.getNumOperands() - 1).isReg())
      continue;
    MCRegister TargetRegister(
        Call.Inst.getOperand(Call.Inst.getNumOperands() - 1).getReg());
    if (!TargetRegister)
      continue;

    std::optional<size_t> LoadIndex =
        findDefBefore(Decoded, CallIndex, LS, TargetRegister);
    if (!LoadIndex)
      continue;
    const InternalDecodedInst &Load = Decoded[*LoadIndex];
    if (Load.Inst.getOpcode() != LS.SLoadB64Opcode ||
        Load.Inst.getNumOperands() < 4 || !Load.Inst.getOperand(0).isReg() ||
        Load.Inst.getOperand(0).getReg() != TargetRegister ||
        !Load.Inst.getOperand(1).isReg() || !Load.Inst.getOperand(1).getReg() ||
        !Load.Inst.getOperand(3).isImm() ||
        Load.Inst.getOperand(3).getImm() != 0)
      continue;
    MCRegister BaseRegister(Load.Inst.getOperand(1).getReg());

    // The source-level table access is defined only while its dynamic index is
    // in bounds. Since the table's ELF section is non-writable, every such
    // load observes one of the relocation-complete slots proved above.
    std::optional<size_t> AddIndex =
        findDefBefore(Decoded, *LoadIndex, LS, BaseRegister);
    if (!AddIndex)
      continue;
    const InternalDecodedInst &Add = Decoded[*AddIndex];
    if (Add.Inst.getOpcode() != LS.SAddNcU64Opcode ||
        Add.Inst.getNumOperands() != 3 || !Add.Inst.getOperand(0).isReg() ||
        Add.Inst.getOperand(0).getReg() != BaseRegister ||
        !Add.Inst.getOperand(1).isReg() ||
        Add.Inst.getOperand(1).getReg() != BaseRegister ||
        !Add.Inst.getOperand(2).isImm())
      continue;

    std::optional<size_t> GetPcIndex =
        findDefBefore(Decoded, *AddIndex, LS, BaseRegister);
    if (!GetPcIndex)
      continue;
    const InternalDecodedInst &GetPc = Decoded[*GetPcIndex];
    if (GetPc.Inst.getOpcode() != LS.SGetPcI64Opcode ||
        GetPc.Inst.getNumOperands() != 1 || !GetPc.Inst.getOperand(0).isReg() ||
        GetPc.Inst.getOperand(0).getReg() != BaseRegister)
      continue;

    // s_get_pc_i64 produces the next instruction's address; s_add_nc_u64 is
    // modulo-2^64, so the unsigned addition also handles a table below .text.
    const uint64_t TableAddress =
        Elf.textAddr() + GetPc.Offset + GetPc.Size +
        static_cast<uint64_t>(Add.Inst.getOperand(2).getImm());
    ArrayRef<RelocationTableCandidate>::iterator Table =
        llvm::find_if(Tables, [&](const RelocationTableCandidate &T) {
          return T.Address == TableAddress;
        });
    if (Table == Tables.end())
      continue;

    RelocationTableDispatch Dispatch;
    Dispatch.CallOffset = Call.Offset;
    Dispatch.SequenceStart = GetPc.Offset;
    Dispatch.SequenceEnd = Call.Offset;
    bool HasInvalidTarget = false;
    for (const std::optional<uint64_t> &Target : Table->Targets) {
      if (!Target)
        continue;
      ArrayRef<InternalDecodedInst>::iterator TargetInst =
          std::lower_bound(Decoded.begin(), Decoded.end(), *Target,
                           [](const InternalDecodedInst &DI, uint64_t Offset) {
                             return DI.Offset < Offset;
                           });
      if (KernelEntries.contains(*Target) || TargetInst == Decoded.end() ||
          TargetInst->Offset != *Target || !TargetInst->DecodeSucceeded) {
        HasInvalidTarget = true;
        break;
      }
      Dispatch.Targets.push_back(*Target);
    }
    if (HasInvalidTarget)
      continue;
    llvm::sort(Dispatch.Targets);
    Dispatch.Targets.erase(
        std::unique(Dispatch.Targets.begin(), Dispatch.Targets.end()),
        Dispatch.Targets.end());
    Dispatches.push_back(std::move(Dispatch));
  }
  return Dispatches;
}

bool isInRelocatedTrailingSection(const ElfView &OldElf, uint64_t VAddr);

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

    // Dynamic relocation sections use sh_info == 0. A relocation place in
    // .text is never safe because instruction expansion changes the bytes the
    // loader would patch. A symbol-less R_AMDGPU_RELATIVE64 explicit addend,
    // however, is an exact load-bias-relative code pointer: DisplacementPlan
    // can remap it without knowing the source-language table type. RCCL device
    // dispatch tables use precisely this representation.
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

          const ELFT::Sym *Sym = nullptr;
          Expected<const ELFT::Sym *> SymOrErr =
              Elf.file().getRelocationSymbol(Rela, *SymtabOrErr);
          if (!SymOrErr) {
            return makeDisplacementError(
                "failed to read dynamic relocation symbol: " +
                Twine(toString(SymOrErr.takeError())));
          }
          Sym = *SymOrErr;

          const uint64_t Addend = static_cast<uint64_t>(Rela.r_addend);
          const bool ReferencesText =
              Addend >= Elf.textAddr() && Addend < TextEnd;
          const bool ReferencesTrailingSection =
              isInRelocatedTrailingSection(Elf, Addend);
          if (ReferencesText || ReferencesTrailingSection) {
            if (Sym || Rela.getType(false) != ELF::R_AMDGPU_RELATIVE64) {
              return makeDisplacementError(
                  "dynamic relocation addend references displaced content "
                  "in an unsupported form");
            }
            // Keep scanning: one supported pointer does not make a second
            // unsupported relocation safe.
          }

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
    // A kernel whose entry falls outside [textAddr, TextEnd) has nothing to
    // displace (e.g. a zero-body placeholder kernel whose entry sits at the
    // one-past-the-end .text boundary). Skip it here to stay consistent with
    // rewriteKernelDescriptorEntriesForDisplacement, which also skips it and
    // leaves its descriptor untouched.
    if (*OldEntry < Elf.textAddr() || *OldEntry >= TextEnd)
      continue;

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

/// Attempt to repair an `s_get_pc_i64; s_add_nc_u64 sX, sX, imm` PC-relative
/// address computation under whole-object displacement. \p Gp is the
/// decoded s_get_pc at old .text offset \p Gp.Offset. The pair materializes an
/// absolute address: pc_base is the address of the instruction following
/// s_get_pc (the s_add), and the s_add immediate is (target - pc_base). When
/// the code moves, pc_base shifts, so the immediate must be adjusted by -shift
/// to keep pointing at the same (unmoved) target.
///
/// Returns Expected<bool>: true if the pair matched and was repaired (or needed
/// no change), false if the instruction is not the expected repairable pair
/// (the caller then falls through to the strict pc-sensitive rejection,
/// fail-closed). An Error is returned only for a genuinely broken rewrite
/// (re-encode size change or a write past the rebuilt buffer).
///
/// CORRECTNESS: -shift is valid only when the computed target is OUTSIDE .text
/// (a data global / GOT slot that does not move). If the target lands inside
/// .text it would itself be displaced and -shift would be wrong, so that case
/// returns false and the caller rejects the whole displacement.
Expected<bool> repairSGetPcPairForDisplacement(
    const ElfView &Elf, const LLVMState &LS, const DisplacementPlan &Plan,
    const InternalDecodedInst &Gp, SmallVectorImpl<uint8_t> &NewText) {
  // The s_add immediately follows s_get_pc. Decode a small window at its
  // offset; the first decoded instruction is the s_add carrying the PC-relative
  // addend.
  const uint64_t AddOldOff = Gp.Offset + Gp.Size;
  if (AddOldOff >= Elf.textSize())
    return false;
  // Decode ONLY the s_add that follows s_get_pc. Decode over the FULL remaining
  // .text (not a fixed-size sub-buffer): the AMDGPU disassembler can read up to
  // getMaxInstLength() bytes for a literal, so a too-small window (e.g. an
  // s_add_nc_u64 with a lit64 near the window edge) makes it over-read past the
  // copy and crash. Streaming stops after the first instruction via the
  // early-false callback, so decoding "the whole rest" costs exactly one insn.
  InternalDecodedInst Add;
  bool Got = false;
  (void)decodeTextSectionStreaming(
      Elf.textData() + AddOldOff, Elf.textSize() - AddOldOff, LS,
      /*WantMnemonic=*/false, [&](const InternalDecodedInst &DI) -> bool {
        Add = DI;
        Got = true;
        return false; // stop after the first instruction
      });
  if (!Got || Add.Inst.getOpcode() != LS.SAddNcU64Opcode ||
      Add.Inst.getNumOperands() != 3)
    return false;

  // The PC-relative addend is operand 2. Tensile/rocBLAS-style code encodes it
  // either as a plain immediate or, when it needs the 64-bit literal form, as
  // an AMDGPU lit/lit64 target MCExpr that evaluates to an absolute constant.
  // Extract the numeric addend from whichever form is present; anything else is
  // not the expected pair (fail-closed).
  const MCOperand &AddendOp = Add.Inst.getOperand(2);
  int64_t OldImm = 0;
  if (AddendOp.isImm()) {
    OldImm = AddendOp.getImm();
  } else if (!AddendOp.isExpr() ||
             !AddendOp.getExpr()->evaluateAsAbsolute(OldImm)) {
    return false;
  }

  // s_get_pc captures the address of the following s_add. Resolve the signed
  // immediate to an actual allocated object address rather than assuming that
  // every negative target is pre-.text data and every positive target is a
  // trailing section. Exotic gaps or out-of-object values fail closed.
  std::optional<uint64_t> OldPcBase =
      checkedAddUint64(Elf.textAddr(), AddOldOff, "s_get_pc old PC base");
  if (!OldPcBase)
    return false;
  std::optional<uint64_t> OldTarget;
  if (OldImm >= 0) {
    OldTarget = checkedAddUint64(*OldPcBase, static_cast<uint64_t>(OldImm),
                                 "s_get_pc old target");
  } else {
    const uint64_t Magnitude = OldImm == std::numeric_limits<int64_t>::min()
                                   ? uint64_t{1} << 63
                                   : static_cast<uint64_t>(-OldImm);
    OldTarget = checkedSubUint64(*OldPcBase, Magnitude, "s_get_pc old target");
  }
  if (!OldTarget)
    return false;
  const uint64_t TextEnd = Elf.textAddr() + Elf.textSize();
  if (*OldTarget >= Elf.textAddr() && *OldTarget < TextEnd)
    return false;

  uint64_t NewAddOff = 0;
  if (!Plan.mapOffset(AddOldOff, DisplacementMapBias::AfterInsertedBytes,
                      NewAddOff))
    return false;
  Expected<uint64_t> NewTargetOrErr =
      remapAllocatedAddress(Elf, Plan, *OldTarget, /*RequireExecutable=*/false,
                            "s_get_pc materialized target");
  if (!NewTargetOrErr) {
    consumeError(NewTargetOrErr.takeError());
    return false;
  }
  std::optional<uint64_t> NewPcBase =
      checkedAddUint64(Elf.textAddr(), NewAddOff, "s_get_pc displaced PC base");
  if (!NewPcBase)
    return false;
  std::optional<int64_t> NewImmOrErr = checkedSignedDifference(
      *NewTargetOrErr, *NewPcBase, "s_get_pc displaced immediate");
  if (!NewImmOrErr)
    return false;
  const int64_t NewImm = *NewImmOrErr;
  if (NewImm == OldImm)
    return true; // nothing moved relative to the pair; original is still
                 // correct

  // Re-encode via the asm parser rather than mutating the MCInst directly: the
  // literal may be an AMDGPU lit/lit64 form whose encoded size must be
  // preserved (a plain MCConstantExpr re-encodes to the shorter inline-literal
  // form and would corrupt the following instruction). Print the instruction to
  // its canonical asm, substitute only the trailing addend operand with the new
  // value in the same literal form, and assemble. This keeps the MC layer as
  // the single source of encoding (no target-internal MCExpr headers, no
  // hardcoded literal-field surgery).
  if (!LS.MCIP)
    return false;
  SmallString<256> PrintedBuf;
  raw_svector_ostream PrintOS(PrintedBuf);
  LS.MCIP->printInst(&Add.Inst, /*Address=*/0, /*Annot=*/"", *LS.STI, PrintOS);
  StringRef Printed = StringRef(PrintedBuf).trim();
  size_t LastComma = Printed.rfind(',');
  if (LastComma == StringRef::npos)
    return false;
  StringRef Head = Printed.substr(0, LastComma + 1); // "...s[2:3], s[2:3],"
  StringRef AddendTok = Printed.substr(LastComma + 1).trim();
  // Preserve the printed literal wrapper (lit64(...) / lit(...)) so the encoded
  // size is unchanged; only the numeric value is replaced.
  std::string Hex =
      ("0x" + Twine::utohexstr(static_cast<uint64_t>(NewImm))).str();
  std::string NewAddend;
  if (AddendTok.starts_with("lit64("))
    NewAddend = "lit64(" + Hex + ")";
  else if (AddendTok.starts_with("lit("))
    NewAddend = "lit(" + Hex + ")";
  else
    NewAddend = Hex;
  std::string NewAsm = (Head + " " + NewAddend).str();

  SmallVector<uint8_t> Code = assembleSingleInst(NewAsm, LS);
  if (Code.size() != Add.Size) {
    return makeDisplacementError("s_add for s_get_pc at old .text offset 0x" +
                                 Twine::utohexstr(Gp.Offset) +
                                 " changed encoded size during re-encode");
  }
  if (NewAddOff + Code.size() > NewText.size()) {
    return makeDisplacementError(
        "repaired s_add for s_get_pc at old .text offset 0x" +
        Twine::utohexstr(Gp.Offset) + " writes past rebuilt .text");
  }
  std::memcpy(NewText.data() + NewAddOff, Code.data(), Code.size());
  return true;
}

/// Repair a standalone `s_add_pc_i64 lit64(delta)` under whole-object
/// displacement. Unlike the s_get_pc pair (whose target is a global outside
/// .text), this is a long-forward PC-relative CONTROL transfer whose target is
/// code INSIDE .text: PC (the address of the instruction after s_add_pc_i64)
/// plus the byte delta. Both the instruction and its target are displaced, so
/// the new delta = new_target_off - (new_add_off + size), with both offsets
/// obtained from the plan (the same both-ends-move remap repairBranches does
/// for s_branch, but s_add_pc reaches far with a 64-bit literal). The operand
/// is a raw byte delta (verified via llvm-mc), not a dword offset. Returns
/// false if the instruction is not the expected single-immediate form or the
/// target leaves .text (fail-closed). Only called for whole-object growth.
Expected<bool> repairSAddPcForDisplacement(const ElfView &Elf,
                                           const LLVMState &LS,
                                           const DisplacementPlan &Plan,
                                           const InternalDecodedInst &Ap,
                                           SmallVectorImpl<uint8_t> &NewText) {
  if (Ap.Inst.getNumOperands() < 1)
    return false;
  const MCOperand &DeltaOp = Ap.Inst.getOperand(0);
  int64_t OldDelta = 0;
  if (DeltaOp.isImm()) {
    OldDelta = DeltaOp.getImm();
  } else if (!DeltaOp.isExpr() ||
             !DeltaOp.getExpr()->evaluateAsAbsolute(OldDelta)) {
    return false;
  }

  // PC base is the address of the NEXT instruction (AMDGPU PC semantics,
  // verified: 0x3BB8F4 + 12 + 0x278c8 lands on a clean boundary). Target is a
  // .text offset that must stay inside .text (control transfer); reject
  // otherwise (fail-closed).
  const int64_t OldPcBase = static_cast<int64_t>(Ap.Offset + Ap.Size);
  const int64_t OldTarget = OldPcBase + OldDelta;
  const int64_t TextSize = static_cast<int64_t>(Elf.textSize());
  if (OldTarget < 0 || OldTarget >= TextSize)
    return false;

  uint64_t NewApOff = 0;
  uint64_t NewTargetOff = 0;
  if (!Plan.mapOffset(Ap.Offset, DisplacementMapBias::AfterInsertedBytes,
                      NewApOff) ||
      !Plan.mapOffset(static_cast<uint64_t>(OldTarget),
                      DisplacementMapBias::BeforeInsertedBytes, NewTargetOff))
    return false;
  const int64_t NewPcBase =
      static_cast<int64_t>(NewApOff) + static_cast<int64_t>(Ap.Size);
  const int64_t NewDelta = static_cast<int64_t>(NewTargetOff) - NewPcBase;
  if (NewDelta == OldDelta)
    return true; // both ends shifted equally; original delta still correct

  // Re-encode preserving the lit/lit64 wrapper so the encoded size is unchanged
  // (same technique as the s_get_pc pair repair).
  if (!LS.MCIP)
    return false;
  SmallString<128> PrintedBuf;
  raw_svector_ostream PrintOS(PrintedBuf);
  LS.MCIP->printInst(&Ap.Inst, /*Address=*/0, /*Annot=*/"", *LS.STI, PrintOS);
  StringRef Printed = StringRef(PrintedBuf).trim();
  // s_add_pc_i64 has a single operand: "s_add_pc_i64 lit64(0x...)".
  size_t Space = Printed.find(' ');
  if (Space == StringRef::npos)
    return false;
  StringRef Mnem = Printed.substr(0, Space);
  StringRef DeltaTok = Printed.substr(Space + 1).trim();
  std::string Hex =
      ("0x" + Twine::utohexstr(static_cast<uint64_t>(NewDelta))).str();
  std::string NewOperand;
  if (DeltaTok.starts_with("lit64("))
    NewOperand = "lit64(" + Hex + ")";
  else if (DeltaTok.starts_with("lit("))
    NewOperand = "lit(" + Hex + ")";
  else
    NewOperand = Hex;
  std::string NewAsm = (Mnem + " " + NewOperand).str();

  SmallVector<uint8_t> Code = assembleSingleInst(NewAsm, LS);
  if (Code.size() != Ap.Size) {
    return makeDisplacementError("s_add_pc_i64 at old .text offset 0x" +
                                 Twine::utohexstr(Ap.Offset) +
                                 " changed encoded size during re-encode");
  }
  if (NewApOff + Code.size() > NewText.size()) {
    return makeDisplacementError(
        "repaired s_add_pc_i64 at old .text offset 0x" +
        Twine::utohexstr(Ap.Offset) + " writes past rebuilt .text");
  }
  std::memcpy(NewText.data() + NewApOff, Code.data(), Code.size());
  return true;
}

Error repairBranches(const ElfView &Elf, const LLVMState &LS,
                     const DisplacementPlan &Plan,
                     const DirectControlFlowInfo &ControlFlow,
                     SmallVectorImpl<uint8_t> &NewText) {
  if (!LS.MIA) {
    return makeDisplacementError(
        "branch analysis through LLVM MC is unavailable");
  }
  for (uint64_t OldTarget : ControlFlow.Targets) {
    uint64_t NewTarget = 0;
    if (!Plan.mapOffset(OldTarget, DisplacementMapBias::BeforeInsertedBytes,
                        NewTarget))
      return makeDisplacementError(
          "linked control-flow target at old .text offset 0x" +
          Twine::utohexstr(OldTarget) + " maps inside a replaced range");
  }

  // Stream the whole .text one instruction at a time instead of materializing a
  // ~1.7M-element decode vector: this validates and repairs branches in a
  // single pass, examining every instruction in the same order with the same
  // rejection checks as a full decode would, only without the intermediate
  // storage. The mnemonic string is skipped (WantMnemonic=false) since it is
  // read solely by this function's diagnostic paths, which reconstruct it on
  // demand.
  std::optional<Error> RepairErr;
  auto onInst = [&](const InternalDecodedInst &DI) -> bool {
    if (!DI.DecodeSucceeded) {
      RepairErr.emplace(makeDisplacementError(
          "undecodable instruction at old .text offset 0x" +
          Twine::utohexstr(DI.Offset)));
      return false;
    }
    if (Plan.rangeOverlapsReplacement(DI.Offset, DI.Size))
      return true;

    const MCInst &Inst = DI.Inst;
    if (ControlFlow.RelocatableIndirectTransfers.contains(DI.Offset))
      return true;
    // Whole-object displacement can repair an s_get_pc_i64; s_add pair
    // that computes an absolute address of a global outside .text: the code
    // moved, so adjust the s_add immediate by -shift. The entry-workaround path
    // (RelocateTrailingSections=false) keeps the strict reject unchanged. A
    // pair that does not match the expected form, or whose target is inside
    // .text, falls through to the strict rejection below (fail-closed).
    if (Plan.relocatesTrailingSections() &&
        Inst.getOpcode() == LS.SGetPcI64Opcode) {
      Expected<bool> RepairedOrErr =
          repairSGetPcPairForDisplacement(Elf, LS, Plan, DI, NewText);
      if (!RepairedOrErr) {
        RepairErr.emplace(RepairedOrErr.takeError());
        return false;
      }
      if (*RepairedOrErr)
        return true;
      // Not a repairable pair; fall through to the strict rejection.
    }
    // Whole-object displacement can also repair a standalone s_add_pc_i64
    // (long-forward PC-relative control transfer into .text): remap its delta
    // so it still reaches the moved target. Entry path keeps the strict reject.
    if (Plan.relocatesTrailingSections() &&
        Inst.getOpcode() == LS.SAddPcI64Opcode) {
      Expected<bool> RepairedOrErr =
          repairSAddPcForDisplacement(Elf, LS, Plan, DI, NewText);
      if (!RepairedOrErr) {
        RepairErr.emplace(RepairedOrErr.takeError());
        return false;
      }
      if (*RepairedOrErr)
        return true;
      // Not repairable; fall through to the strict rejection.
    }
    if (isPcSensitiveForDisplacement(DI, LS)) {
      // DI.Mnemonic was not populated (WantMnemonic=false above); reconstruct
      // it here on the cold diagnostic path so the message keeps naming the
      // offender.
      StringRef Mnemonic = "<unknown>";
      if (LS.MCIP) {
        std::pair<const char *, uint64_t> Mnem = LS.MCIP->getMnemonic(DI.Inst);
        if (Mnem.first)
          Mnemonic = StringRef(Mnem.first).rtrim();
      }
      RepairErr.emplace(makeDisplacementError(
          "pc-sensitive instruction '" + Twine(Mnemonic) +
          "' at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
          " requires linked-address repair"));
      return false;
    }
    if (LS.MIA->isCall(Inst)) {
      RepairErr.emplace(makeDisplacementError(
          "call at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
          " is not supported by displacement"));
      return false;
    }
    if (LS.MIA->isIndirectBranch(Inst)) {
      RepairErr.emplace(makeDisplacementError(
          "indirect branch at old .text offset 0x" +
          Twine::utohexstr(DI.Offset) + " is not supported by displacement"));
      return false;
    }
    if (!LS.MIA->isBranch(Inst))
      return true;

    uint64_t OldTarget = 0;
    if (!LS.MIA->evaluateBranch(Inst, DI.Offset, DI.Size, OldTarget)) {
      RepairErr.emplace(makeDisplacementError(
          "branch at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
          " target could not be evaluated"));
      return false;
    }
    if (OldTarget >= Elf.textSize()) {
      RepairErr.emplace(makeDisplacementError("branch at old .text offset 0x" +
                                              Twine::utohexstr(DI.Offset) +
                                              " targets outside .text"));
      return false;
    }

    uint64_t NewFrom = 0;
    uint64_t NewTarget = 0;
    if (!Plan.mapOffset(DI.Offset, DisplacementMapBias::AfterInsertedBytes,
                        NewFrom)) {
      RepairErr.emplace(makeDisplacementError(
          "branch source at old .text offset 0x" + Twine::utohexstr(DI.Offset) +
          " maps inside a replaced range"));
      return false;
    }
    if (!Plan.mapOffset(OldTarget, DisplacementMapBias::BeforeInsertedBytes,
                        NewTarget)) {
      RepairErr.emplace(makeDisplacementError(
          "branch target at old .text offset 0x" + Twine::utohexstr(OldTarget) +
          " maps inside a replaced range"));
      return false;
    }

    Expected<SmallVector<uint8_t>> EncodedOrErr =
        reencodePcrelBranch(DI, NewFrom, NewTarget, LS);
    if (!EncodedOrErr) {
      RepairErr.emplace(EncodedOrErr.takeError());
      return false;
    }
    SmallVector<uint8_t> &Encoded = *EncodedOrErr;

    if (NewFrom + Encoded.size() > NewText.size()) {
      RepairErr.emplace(makeDisplacementError(
          "re-encoded branch at old .text offset 0x" +
          Twine::utohexstr(DI.Offset) + " writes past rebuilt .text"));
      return false;
    }
    std::memcpy(NewText.data() + NewFrom, Encoded.data(), Encoded.size());
    return true;
  };

  bool Decoded = decodeTextSectionStreaming(Elf.textData(), Elf.textSize(), LS,
                                            /*WantMnemonic=*/false, onInst);
  if (RepairErr)
    return std::move(*RepairErr);
  if (!Decoded)
    return makeDisplacementError(
        "failed to decode .text while validating branches");
  return Error::success();
}

Error adjustSectionHeadersForTextGrowth(uint8_t *Elf, size_t ElfSize,
                                        const ElfView &OldElf, size_t Growth,
                                        bool RelocateVAddr) {
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
      // Relocate the virtual address of any ALLOCATABLE section after
      // .text so the grown .text pushes it forward instead of overlapping it.
      // Non-allocatable sections (.symtab, .strtab, .comment) keep sh_addr 0.
      if (RelocateVAddr) {
        uint64_t ShFlags = 0;
        std::memcpy(&ShFlags, Sh + offsetof(Shdr, sh_flags), sizeof(ShFlags));
        uint64_t ShAddr = 0;
        std::memcpy(&ShAddr, Sh + offsetof(Shdr, sh_addr), sizeof(ShAddr));
        if ((ShFlags & ELF::SHF_ALLOC) && ShAddr >= OldElf.textAddr()) {
          if (ShAddr > std::numeric_limits<uint64_t>::max() - Growth)
            return makeDisplacementError(
                "section virtual address overflows after displacement");
          uint64_t NewAddr = ShAddr + Growth;
          std::memcpy(Sh + offsetof(Shdr, sh_addr), &NewAddr, sizeof(NewAddr));
        }
      }
    }
  }
  return Error::success();
}

Error adjustProgramHeadersForTextGrowth(uint8_t *Elf, size_t ElfSize,
                                        const ElfView &OldElf, size_t Growth,
                                        bool RelocateVAddr) {
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
      // Also relocate this segment's virtual/physical address so the
      // loader maps it past the grown .text instead of on top of it.
      if (RelocateVAddr) {
        uint64_t PVaddr = 0, PPaddr = 0;
        std::memcpy(&PVaddr, Ph + offsetof(Phdr, p_vaddr), sizeof(PVaddr));
        std::memcpy(&PPaddr, Ph + offsetof(Phdr, p_paddr), sizeof(PPaddr));
        if (PVaddr > std::numeric_limits<uint64_t>::max() - Growth ||
            PPaddr > std::numeric_limits<uint64_t>::max() - Growth)
          return makeDisplacementError(
              "program-header address overflows after displacement");
        PVaddr += Growth;
        PPaddr += Growth;
        std::memcpy(Ph + offsetof(Phdr, p_vaddr), &PVaddr, sizeof(PVaddr));
        std::memcpy(Ph + offsetof(Phdr, p_paddr), &PPaddr, sizeof(PPaddr));
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
  Expected<uint64_t> NewEntryOrErr = remapAllocatedAddress(
      OldElf, Plan, OldEntry, /*RequireExecutable=*/true, "ELF e_entry");
  if (!NewEntryOrErr)
    return NewEntryOrErr.takeError();
  std::memcpy(Elf + offsetof(Ehdr, e_entry), &*NewEntryOrErr,
              sizeof(*NewEntryOrErr));
  return Error::success();
}

Error adjustDynamicEntriesForDisplacement(uint8_t *Elf, size_t ElfSize,
                                          const ElfView &OldElf,
                                          const DisplacementPlan &Plan) {
  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Elf), ElfSize));
  if (!FileOrErr)
    return makeDisplacementError(
        "failed to parse displaced ELF for dynamic-tag repair: " +
        Twine(toString(FileOrErr.takeError())));
  ELFFileT File = std::move(*FileOrErr);
  Expected<ELFT::DynRange> EntriesOrErr = File.dynamicEntries();
  if (!EntriesOrErr)
    return makeDisplacementError(
        "failed to read dynamic tags during displacement: " +
        Twine(toString(EntriesOrErr.takeError())));

  for (const ELFT::Dyn &Entry : *EntriesOrErr) {
    const int64_t Tag = static_cast<int64_t>(Entry.d_tag);
    DynamicTagClass Class = classifyDynamicTag(Tag);
    if (Class == DynamicTagClass::Unknown)
      return makeDisplacementError("unknown dynamic tag 0x" +
                                   Twine::utohexstr(Entry.d_tag) +
                                   " may carry an unclassified address");
    if (Class == DynamicTagClass::UnsupportedAddress)
      return makeDisplacementError(
          "dynamic tag 0x" + Twine::utohexstr(Entry.d_tag) +
          " introduces an unsupported address-bearing construct");
    if (Entry.d_un.d_val != 0 &&
        (Tag == ELF::DT_PLTRELSZ || Tag == ELF::DT_PLTREL ||
         Tag == ELF::DT_INIT_ARRAYSZ || Tag == ELF::DT_FINI_ARRAYSZ ||
         Tag == ELF::DT_PREINIT_ARRAYSZ || Tag == ELF::DT_RELRSZ ||
         Tag == ELF::DT_RELRENT))
      return makeDisplacementError(
          "dynamic tag enables an unsupported pointer/relocation table");
    if (Class != DynamicTagClass::Address || Entry.d_un.d_ptr == 0)
      continue;

    Expected<const ELFT::Shdr *> OwnerOrErr = findUniqueAllocatedSection(
        OldElf, Entry.d_un.d_ptr, /*RequireExecutable=*/false,
        "dynamic-tag address");
    if (!OwnerOrErr)
      return OwnerOrErr.takeError();
    if (!dynamicTagMatchesSection(Tag, (**OwnerOrErr).sh_type))
      return makeDisplacementError(
          "dynamic tag points to a section of the wrong type");
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

bool isInRelocatedTrailingSection(const ElfView &OldElf, uint64_t VAddr) {
  const uint64_t OldTextEnd = OldElf.textAddr() + OldElf.textSize();
  for (const ELFT::Shdr &Shdr : OldElf.sections()) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) || &Shdr == OldElf.textSection() ||
        Shdr.sh_size == 0 || Shdr.sh_addr < OldTextEnd)
      continue;
    if (VAddr >= Shdr.sh_addr && VAddr - Shdr.sh_addr < Shdr.sh_size)
      return true;
  }
  return false;
}

/// Repair the two address-bearing fields in dynamic relocation records:
///
/// * r_offset is the loader write location. When displacement moves its
/// allocated
///   section, the place moves by the padded text growth.
/// * a symbol-less R_AMDGPU_RELATIVE64 r_addend is a runtime-dereferenced
///   allocated address. Map it through the same DisplacementPlan used for
///   branches, symbols, descriptors, and debug information.
///
/// This structural rewrite is both smaller and more exact than recovering the
/// source-level dataflow: hotswap preserves every allocated target exactly
/// once, so the ELF relocation already supplies all required proof.
Error adjustRelocationsForDisplacement(uint8_t *Elf, size_t ElfSize,
                                       const ElfView &OldElf,
                                       const DisplacementPlan &Plan) {
  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Elf), ElfSize));
  if (!FileOrErr) {
    return makeDisplacementError(
        "failed to parse displaced ELF for relocation repair: " +
        Twine(toString(FileOrErr.takeError())));
  }
  ELFFileT File = std::move(*FileOrErr);

  Expected<ELFT::ShdrRange> SectionsOrErr = File.sections();
  if (!SectionsOrErr) {
    return makeDisplacementError(
        "failed to read displaced ELF sections for relocation repair: " +
        Twine(toString(SectionsOrErr.takeError())));
  }

  const bool UsesVirtualRelocationPlaces =
      OldElf.file().getHeader().e_type != ELF::ET_REL;
  uint64_t ShiftedPlaces = 0;
  uint64_t RemappedAddends = 0;

  for (const ELFT::Shdr &RelocShdr : *SectionsOrErr) {
    if (RelocShdr.sh_type == ELF::SHT_RELA) {
      Expected<ELFT::RelaRange> RelasOrErr = File.relas(RelocShdr);
      if (!RelasOrErr) {
        return makeDisplacementError(
            "failed to read RELA records during displacement: " +
            Twine(toString(RelasOrErr.takeError())));
      }
      for (const ELFT::Rela &Rela : *RelasOrErr) {
        ELFT::Rela NewRela = Rela;
        bool Changed = false;

        if (Plan.relocatesTrailingSections() && UsesVirtualRelocationPlaces &&
            isInRelocatedTrailingSection(OldElf, NewRela.r_offset)) {
          if (NewRela.r_offset >
              std::numeric_limits<uint64_t>::max() - Plan.paddedGrowth())
            return makeDisplacementError(
                "relocation place overflows after displacement");
          NewRela.r_offset += Plan.paddedGrowth();
          Changed = true;
          ++ShiftedPlaces;
        }

        if (NewRela.getSymbol(false) == 0 &&
            NewRela.getType(false) == ELF::R_AMDGPU_RELATIVE64) {
          const uint64_t Addend = static_cast<uint64_t>(NewRela.r_addend);
          Expected<uint64_t> NewAddendOrErr = remapLoadAddress(
              OldElf, Plan, Addend, "R_AMDGPU_RELATIVE64 addend");
          if (!NewAddendOrErr)
            return NewAddendOrErr.takeError();
          if (*NewAddendOrErr != Addend) {
            static_assert(sizeof(NewRela.r_addend) == sizeof(*NewAddendOrErr));
            std::memcpy(&NewRela.r_addend, &*NewAddendOrErr,
                        sizeof(NewRela.r_addend));
            Changed = true;
            ++RemappedAddends;
          }
        }

        if (!Changed)
          continue;
        Expected<uint64_t> OffsetOrErr = getRecordOffset(File, Rela);
        if (!OffsetOrErr)
          return OffsetOrErr.takeError();
        std::memcpy(Elf + *OffsetOrErr, &NewRela, sizeof(NewRela));
      }
      continue;
    }

    if (RelocShdr.sh_type != ELF::SHT_REL ||
        !Plan.relocatesTrailingSections() || !UsesVirtualRelocationPlaces)
      continue;
    Expected<ELFT::RelRange> RelsOrErr = File.rels(RelocShdr);
    if (!RelsOrErr) {
      return makeDisplacementError(
          "failed to read REL records during displacement: " +
          Twine(toString(RelsOrErr.takeError())));
    }
    for (const ELFT::Rel &Rel : *RelsOrErr) {
      if (!isInRelocatedTrailingSection(OldElf, Rel.r_offset))
        continue;
      if (Rel.r_offset >
          std::numeric_limits<uint64_t>::max() - Plan.paddedGrowth())
        return makeDisplacementError(
            "relocation place overflows after displacement");
      ELFT::Rel NewRel = Rel;
      NewRel.r_offset += Plan.paddedGrowth();
      Expected<uint64_t> OffsetOrErr = getRecordOffset(File, Rel);
      if (!OffsetOrErr)
        return OffsetOrErr.takeError();
      std::memcpy(Elf + *OffsetOrErr, &NewRel, sizeof(NewRel));
      ++ShiftedPlaces;
    }
  }

  if (ShiftedPlaces != 0 || RemappedAddends != 0)
    log() << "hotswap: displacement: shifted " << ShiftedPlaces
          << " relocation place(s), remapped " << RemappedAddends
          << " R_AMDGPU_RELATIVE64 code pointer(s)\n";
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

      // Whole-object displacement moves every allocated section after .text as
      // one unit. Keep its symbols attached to the same section-relative byte.
      // PC-relative code references are repaired separately from their original
      // target address; relocation-backed references observe this updated
      // symbol value.
      if (Plan.relocatesTrailingSections() &&
          OldElf.file().getHeader().e_type != ELF::ET_REL &&
          Sym.st_shndx < OldElf.sections().size()) {
        const ELFT::Shdr &OldDefShdr = OldElf.sections()[Sym.st_shndx];
        const uint64_t OldTextEnd = OldElf.textAddr() + OldElf.textSize();
        if ((OldDefShdr.sh_flags & ELF::SHF_ALLOC) &&
            OldDefShdr.sh_addr >= OldTextEnd &&
            Sym.st_value >= OldDefShdr.sh_addr &&
            Sym.st_value - OldDefShdr.sh_addr <= OldDefShdr.sh_size) {
          if (Sym.st_value >
              std::numeric_limits<uint64_t>::max() - Plan.paddedGrowth())
            return makeDisplacementError(
                "non-text symbol value overflows after displacement");
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

// Locate a section by name in the OLD ELF's section table. Debug sections sit
// AFTER .text, so their bytes in the grown OUTPUT buffer live at
// OLD sh_offset + paddedGrowth (the apply step copied the whole trailing region
// forward). Callers apply that shift when addressing the output; this helper
// only resolves the header.
const ELFT::Shdr *findSectionByName(const ElfView &Elf, StringRef Want) {
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    Expected<StringRef> NameOrErr = Elf.file().getSectionName(Shdr);
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      return nullptr;
    }
    if (*NameOrErr == Want)
      return &Shdr;
  }
  return nullptr;
}

// Map an absolute .text virtual address through the displacement plan. Objects
// that carry DWARF use vaddr == textAddr + offset in .text; addresses outside
// [textAddr, textEnd] pass through unchanged. An address at exactly textEnd
// maps to textEnd + paddedGrowth (the whole trailing region shifts). \p Bias
// selects the boundary rule at an inserted-bytes edge (BeforeInsertedBytes for
// the low end of a range, so the start of a patched instruction stays put; the
// high end also uses BeforeInsertedBytes so a range that ends at a patch
// boundary keeps its end pinned to that instruction). Returns false only on
// internal mapOffset failure (address lands inside a replaced instruction),
// which is a genuine corruption signal for the caller to fail closed on.
bool mapTextVAddr(const ElfView &Elf, const DisplacementPlan &Plan,
                  uint64_t OldVAddr, uint64_t &NewVAddr) {
  const uint64_t TextAddr = Elf.textAddr();
  const uint64_t TextEnd = TextAddr + Elf.textSize();
  if (OldVAddr < TextAddr || OldVAddr > TextEnd) {
    NewVAddr = OldVAddr;
    return true;
  }
  uint64_t NewOff = 0;
  if (!Plan.mapOffset(OldVAddr - TextAddr,
                      DisplacementMapBias::BeforeInsertedBytes, NewOff))
    return false;
  NewVAddr = TextAddr + NewOff;
  return true;
}

/// Remap the single FDE-per-object `.debug_frame`. Unlike `.eh_frame`,
/// `.debug_frame` addresses are ABSOLUTE: an FDE stores initial_location and
/// address_range as target-sized (8-byte) values. CIE vs FDE is distinguished
/// by the CIE id field: 0xffffffff (DW_CIE_ID for 32-bit DWARF) marks a CIE.
/// For each FDE describing .text, remap initial_location via the plan and
/// recompute address_range = mapped_end - mapped_start. Fails closed on 64-bit
/// DWARF, non-8-byte addresses, or a record that overruns the section.
Error remapDebugFrameForDisplacement(const ElfView &Elf,
                                     const DisplacementPlan &Plan,
                                     uint8_t *Base, uint64_t Size) {
  auto readU32 = [&](uint64_t Off) -> uint32_t {
    uint32_t V;
    std::memcpy(&V, Base + Off, sizeof(V));
    return V;
  };
  auto readU64 = [&](uint64_t Off) -> uint64_t {
    uint64_t V;
    std::memcpy(&V, Base + Off, sizeof(V));
    return V;
  };
  auto writeU64 = [&](uint64_t Off, uint64_t V) {
    std::memcpy(Base + Off, &V, sizeof(V));
  };

  const uint64_t TextAddr = Elf.textAddr();
  const uint64_t TextEnd = TextAddr + Elf.textSize();
  uint64_t Pos = 0;
  uint32_t Remapped = 0;
  while (Pos + 4 <= Size) {
    const uint32_t Length = readU32(Pos);
    if (Length == 0)
      break; // terminator
    if (Length == 0xffffffffu)
      return makeDisplacementError(
          ".debug_frame uses 64-bit DWARF (unsupported)");
    const uint64_t RecordStart = Pos;
    const uint64_t RecordEnd = Pos + 4 + Length;
    if (RecordEnd > Size || RecordStart + 8 > Size)
      return makeDisplacementError(".debug_frame record overruns section");
    const uint32_t CieId = readU32(RecordStart + 4);
    if (CieId == dwarf::DW_CIE_ID) {
      Pos = RecordEnd; // CIE: no addresses to remap
      continue;
    }
    // FDE: initial_location (8) then address_range (8) follow the CIE pointer.
    const uint64_t LocOff = RecordStart + 8;
    const uint64_t RangeOff = RecordStart + 16;
    if (RangeOff + 8 > RecordEnd)
      return makeDisplacementError(".debug_frame FDE is truncated");
    const uint64_t OldStart = readU64(LocOff);
    const uint64_t OldRange = readU64(RangeOff);
    if (OldStart < TextAddr || OldStart >= TextEnd) {
      Pos = RecordEnd; // FDE does not describe .text
      continue;
    }
    if (OldStart + OldRange < OldStart || OldStart + OldRange > TextEnd)
      return makeDisplacementError(".debug_frame FDE range leaves .text");
    uint64_t NewStart = 0, NewEnd = 0;
    if (!mapTextVAddr(Elf, Plan, OldStart, NewStart) ||
        !mapTextVAddr(Elf, Plan, OldStart + OldRange, NewEnd) ||
        NewEnd < NewStart)
      return makeDisplacementError(".debug_frame FDE cannot be remapped");
    writeU64(LocOff, NewStart);
    writeU64(RangeOff, NewEnd - NewStart);
    ++Remapped;
    Pos = RecordEnd;
  }
  log() << "hotswap: displacement: remapped " << Remapped
        << " .debug_frame FDE(s)\n";
  return Error::success();
}

/// Remap `.debug_ranges`. Each compile unit's list is a sequence of (start,end)
/// address pairs. A (0,0) pair terminates a list. A base-address-selection
/// entry has start == 0xffffffffffffffff and sets the base for subsequent
/// entries; this object emits none (verified: no 0xff..ff entries) and
/// dwarfdump shows the pairs are offsets from the CU's DW_AT_low_pc. We
/// therefore treat each nonzero pair value as (Base + stored) to recover the
/// absolute .text address, remap that, and write back (new_absolute - Base). \p
/// Base is the CU low_pc, which is unchanged here (it precedes all patches), so
/// this is exact. Fails closed if a base-selection entry appears (we do not
/// track multiple bases).
Error remapDebugRangesForDisplacement(const ElfView &Elf,
                                      const DisplacementPlan &Plan,
                                      uint8_t *SecBase, uint64_t Size,
                                      uint64_t CuBase) {
  auto readU64 = [&](uint64_t Off) -> uint64_t {
    uint64_t V;
    std::memcpy(&V, SecBase + Off, sizeof(V));
    return V;
  };
  auto writeU64 = [&](uint64_t Off, uint64_t V) {
    std::memcpy(SecBase + Off, &V, sizeof(V));
  };

  uint64_t Pos = 0;
  uint32_t Remapped = 0;
  while (Pos + 16 <= Size) {
    const uint64_t Start = readU64(Pos);
    const uint64_t End = readU64(Pos + 8);
    if (Start == 0 && End == 0) {
      Pos += 16; // end-of-list terminator; keep scanning for the next CU list
      continue;
    }
    if (Start == std::numeric_limits<uint64_t>::max())
      return makeDisplacementError(
          ".debug_ranges base-address selection is unsupported");
    // Reconstruct absolute .text addresses, remap, and re-encode as offsets
    // from the (unchanged) CU base.
    const uint64_t OldStartAbs = CuBase + Start;
    const uint64_t OldEndAbs = CuBase + End;
    uint64_t NewStartAbs = 0, NewEndAbs = 0;
    if (!mapTextVAddr(Elf, Plan, OldStartAbs, NewStartAbs) ||
        !mapTextVAddr(Elf, Plan, OldEndAbs, NewEndAbs) ||
        NewEndAbs < NewStartAbs || NewStartAbs < CuBase || NewEndAbs < CuBase)
      return makeDisplacementError(".debug_ranges entry cannot be remapped");
    writeU64(Pos, NewStartAbs - CuBase);
    writeU64(Pos + 8, NewEndAbs - CuBase);
    ++Remapped;
    Pos += 16;
  }
  log() << "hotswap: displacement: remapped " << Remapped
        << " .debug_ranges entr(ies)\n";
  return Error::success();
}

// One attribute (attr, form) from a .debug_abbrev declaration.
struct AbbrevAttr {
  uint64_t Attr = 0;
  uint64_t Form = 0;
  int64_t ImplicitConst = 0; // only for DW_FORM_implicit_const
};
// One abbreviation declaration: its tag/children and attribute list.
struct AbbrevDecl {
  bool HasChildren = false;
  SmallVector<AbbrevAttr, 8> Attrs;
};

// Parse the .debug_abbrev table starting at \p Off into a code -> declaration
// map. Only the fields the DIE walker needs are retained. Fails closed on a
// truncated table.
Expected<DenseMap<uint64_t, AbbrevDecl>>
parseAbbrevTable(const uint8_t *Base, uint64_t Size, uint64_t Off) {
  DenseMap<uint64_t, AbbrevDecl> Table;
  const uint8_t *P = Base + Off;
  const uint8_t *End = Base + Size;
  while (P < End) {
    unsigned N = 0;
    uint64_t Code = decodeULEB128(P, &N, End);
    P += N;
    if (Code == 0)
      break; // end of this abbrev table
    if (P >= End)
      return makeDisplacementError(".debug_abbrev declaration is truncated");
    uint64_t Tag = decodeULEB128(P, &N, End);
    (void)Tag;
    P += N;
    if (P >= End)
      return makeDisplacementError(".debug_abbrev declaration is truncated");
    AbbrevDecl Decl;
    Decl.HasChildren = (*P != 0);
    ++P;
    while (P < End) {
      uint64_t Attr = decodeULEB128(P, &N, End);
      P += N;
      uint64_t Form = decodeULEB128(P, &N, End);
      P += N;
      if (Attr == 0 && Form == 0)
        break; // end of attribute list
      AbbrevAttr A;
      A.Attr = Attr;
      A.Form = Form;
      if (Form == dwarf::DW_FORM_implicit_const) {
        A.ImplicitConst = decodeSLEB128(P, &N, End);
        P += N;
      }
      Decl.Attrs.push_back(A);
    }
    Table[Code] = std::move(Decl);
  }
  return Table;
}

// Advance \p P past a single attribute value of \p Form in a DWARF32 unit
// bounded by \p UnitEnd. Handles the fixed-size forms via getFixedFormByteSize
// and the variable-length forms (LEB128, counted blocks, inline strings) by
// hand. Fails closed on any form whose size we cannot determine so we never
// desynchronize the DIE walk and miswrite a later low_pc/high_pc.
Error skipFormValue(const uint8_t *Base, uint64_t &P, uint64_t UnitEnd,
                    uint64_t Form, int64_t ImplicitConst,
                    const dwarf::FormParams &FP) {
  const uint8_t *End = Base + UnitEnd;
  auto need = [&](uint64_t N) -> Error {
    if (N > UnitEnd - P)
      return makeDisplacementError(".debug_info attribute overruns unit");
    return Error::success();
  };
  switch (Form) {
  case dwarf::DW_FORM_implicit_const:
  case dwarf::DW_FORM_flag_present:
    (void)ImplicitConst;
    return Error::success(); // no bytes in the DIE
  case dwarf::DW_FORM_udata:
  case dwarf::DW_FORM_ref_udata:
  case dwarf::DW_FORM_strx:
  case dwarf::DW_FORM_addrx:
  case dwarf::DW_FORM_loclistx:
  case dwarf::DW_FORM_rnglistx:
  case dwarf::DW_FORM_GNU_str_index:
  case dwarf::DW_FORM_GNU_addr_index: {
    unsigned N = 0;
    decodeULEB128(Base + P, &N, End);
    P += N;
    return Error::success();
  }
  case dwarf::DW_FORM_sdata: {
    unsigned N = 0;
    decodeSLEB128(Base + P, &N, End);
    P += N;
    return Error::success();
  }
  case dwarf::DW_FORM_string: {
    while (P < UnitEnd && Base[P] != 0)
      ++P;
    if (P >= UnitEnd)
      return makeDisplacementError(".debug_info inline string is unterminated");
    ++P; // consume NUL
    return Error::success();
  }
  case dwarf::DW_FORM_block1: {
    if (Error E = need(1))
      return E;
    uint64_t Len = Base[P];
    ++P;
    if (Error E = need(Len))
      return E;
    P += Len;
    return Error::success();
  }
  case dwarf::DW_FORM_block2: {
    if (Error E = need(2))
      return E;
    uint16_t Len = 0;
    std::memcpy(&Len, Base + P, 2);
    P += 2;
    if (Error E = need(Len))
      return E;
    P += Len;
    return Error::success();
  }
  case dwarf::DW_FORM_block4: {
    if (Error E = need(4))
      return E;
    uint32_t Len = 0;
    std::memcpy(&Len, Base + P, 4);
    P += 4;
    if (Error E = need(Len))
      return E;
    P += Len;
    return Error::success();
  }
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_exprloc: {
    unsigned N = 0;
    uint64_t Len = decodeULEB128(Base + P, &N, End);
    P += N;
    if (Error E = need(Len))
      return E;
    P += Len;
    return Error::success();
  }
  default:
    break;
  }
  std::optional<uint8_t> FS =
      dwarf::getFixedFormByteSize(static_cast<dwarf::Form>(Form), FP);
  if (!FS)
    return makeDisplacementError(".debug_info uses an unsupported form " +
                                 Twine(Form));
  if (Error E = need(*FS))
    return E;
  P += *FS;
  return Error::success();
}

/// Remap the CU's `DW_AT_low_pc`/`DW_AT_high_pc` in `.debug_info`. We parse the
/// matching `.debug_abbrev` table to learn each attribute's form so we know the
/// exact byte layout: low_pc is DW_FORM_addr (absolute) and high_pc is either
/// DW_FORM_addr (absolute) or a DW_FORM_data* constant that encodes an OFFSET
/// from low_pc. For this object low_pc is DW_FORM_addr and high_pc is
/// DW_FORM_data4 (verified via .debug_abbrev). Remapping rule (general):
/// new_high = mapOffset(low_off + high) - mapOffset(low_off), so a data-form
/// high_pc grows only by the shift its range end crosses; an absolute-form
/// high_pc is remapped directly; a low_pc that also moved is handled by
/// remapping it too. Every other attribute is skipped using its form's fixed
/// byte size (getFixedFormByteSize) or its LEB/block/string length. Fails
/// closed on DWARF64, addr_size != 8, an abbrev code with no declaration, or a
/// form whose length we cannot determine.
Error remapDebugInfoForDisplacement(const ElfView &Elf,
                                    const DisplacementPlan &Plan,
                                    uint8_t *InfoBase, uint64_t InfoSize,
                                    const uint8_t *AbbrevBase,
                                    uint64_t AbbrevSize) {
  auto readU32 = [&](uint64_t Off) -> uint32_t {
    uint32_t V;
    std::memcpy(&V, InfoBase + Off, sizeof(V));
    return V;
  };
  auto readU16 = [&](uint64_t Off) -> uint16_t {
    uint16_t V;
    std::memcpy(&V, InfoBase + Off, sizeof(V));
    return V;
  };

  uint64_t Pos = 0;
  uint32_t RemappedLow = 0, RemappedHigh = 0;
  while (Pos + 4 <= InfoSize) {
    const uint32_t UnitLength = readU32(Pos);
    if (UnitLength == 0)
      break;
    if (UnitLength == 0xffffffffu)
      return makeDisplacementError(
          ".debug_info uses 64-bit DWARF (unsupported)");
    const uint64_t UnitStart = Pos;
    const uint64_t UnitEnd = Pos + 4 + UnitLength;
    if (UnitEnd > InfoSize)
      return makeDisplacementError(".debug_info unit overruns section");
    uint64_t P = UnitStart + 4;
    const uint16_t Version = readU16(P);
    P += 2;
    if (Version < 2 || Version > 4)
      return makeDisplacementError(".debug_info version " + Twine(Version) +
                                   " is unsupported");
    // DWARF v2-v4 CU header: version, abbrev_offset (sec_offset), addr_size.
    const uint32_t AbbrevOff = readU32(P);
    P += 4;
    const uint8_t AddrSize = InfoBase[P];
    P += 1;
    if (AddrSize != 8)
      return makeDisplacementError(".debug_info addr_size != 8 is unsupported");

    Expected<DenseMap<uint64_t, AbbrevDecl>> TableOrErr =
        parseAbbrevTable(AbbrevBase, AbbrevSize, AbbrevOff);
    if (!TableOrErr)
      return TableOrErr.takeError();
    const DenseMap<uint64_t, AbbrevDecl> &Table = *TableOrErr;

    dwarf::FormParams FP;
    FP.Version = Version;
    FP.AddrSize = AddrSize;
    FP.Format = dwarf::DWARF32;

    // Walk DIEs. We only rewrite low_pc/high_pc; all other attributes are
    // skipped by size. low_pc is captured per-DIE so a sibling high_pc data
    // form is interpreted relative to the correct low_pc.
    const uint8_t *End = InfoBase + UnitEnd;
    while (P < UnitEnd) {
      unsigned N = 0;
      uint64_t Code = decodeULEB128(InfoBase + P, &N, End);
      P += N;
      if (Code == 0)
        continue; // null DIE (end of siblings)
      DenseMap<uint64_t, AbbrevDecl>::const_iterator It = Table.find(Code);
      if (It == Table.end())
        return makeDisplacementError(
            ".debug_info references unknown abbrev code");
      const AbbrevDecl &Decl = It->second;

      bool HaveLow = false;
      uint64_t LowVAddr = 0;
      uint64_t LowOff = 0; // .text offset of low_pc
      for (const AbbrevAttr &A : Decl.Attrs) {
        if (A.Attr == dwarf::DW_AT_low_pc) {
          if (A.Form != dwarf::DW_FORM_addr)
            return makeDisplacementError(
                "DW_AT_low_pc uses an unsupported form");
          uint64_t Old = 0;
          std::memcpy(&Old, InfoBase + P, sizeof(Old));
          uint64_t New = 0;
          if (!mapTextVAddr(Elf, Plan, Old, New))
            return makeDisplacementError("DW_AT_low_pc cannot be remapped");
          std::memcpy(InfoBase + P, &New, sizeof(New));
          HaveLow = Old >= Elf.textAddr();
          LowVAddr = Old;
          LowOff = (Old >= Elf.textAddr()) ? Old - Elf.textAddr() : 0;
          ++RemappedLow;
          P += 8;
          continue;
        }
        if (A.Attr == dwarf::DW_AT_high_pc) {
          if (A.Form == dwarf::DW_FORM_addr) {
            uint64_t Old = 0;
            std::memcpy(&Old, InfoBase + P, sizeof(Old));
            uint64_t New = 0;
            if (!mapTextVAddr(Elf, Plan, Old, New))
              return makeDisplacementError(
                  "DW_AT_high_pc (addr) cannot be remapped");
            std::memcpy(InfoBase + P, &New, sizeof(New));
            ++RemappedHigh;
            P += 8;
            continue;
          }
          // Data forms encode an OFFSET from low_pc. Grow it by the shift its
          // range end crosses: new = map(low_off + old) - map(low_off).
          std::optional<uint8_t> FS =
              dwarf::getFixedFormByteSize(static_cast<dwarf::Form>(A.Form), FP);
          if (!FS || (*FS != 1 && *FS != 2 && *FS != 4 && *FS != 8))
            return makeDisplacementError(
                "DW_AT_high_pc uses an unsupported data form");
          if (!HaveLow)
            return makeDisplacementError(
                "DW_AT_high_pc data form without a .text low_pc");
          uint64_t OldHigh = 0;
          std::memcpy(&OldHigh, InfoBase + P, *FS);
          uint64_t NewLowOff = 0, NewEndOff = 0;
          if (!Plan.mapOffset(LowOff, DisplacementMapBias::BeforeInsertedBytes,
                              NewLowOff) ||
              !Plan.mapOffset(LowOff + OldHigh,
                              DisplacementMapBias::BeforeInsertedBytes,
                              NewEndOff) ||
              NewEndOff < NewLowOff)
            return makeDisplacementError(
                "DW_AT_high_pc (data) cannot be remapped");
          uint64_t NewHigh = NewEndOff - NewLowOff;
          uint64_t Limit = (*FS == 8) ? std::numeric_limits<uint64_t>::max()
                                      : ((uint64_t(1) << (*FS * 8)) - 1);
          if (NewHigh > Limit)
            return makeDisplacementError(
                "DW_AT_high_pc overflows its data form");
          std::memcpy(InfoBase + P, &NewHigh, *FS);
          (void)LowVAddr;
          ++RemappedHigh;
          P += *FS;
          continue;
        }
        // Any other attribute: advance past its encoded value.
        if (Error Err = skipFormValue(InfoBase, P, UnitEnd, A.Form,
                                      A.ImplicitConst, FP))
          return Err;
      }
    }
    Pos = UnitEnd;
  }
  log() << "hotswap: displacement: remapped " << RemappedLow << " low_pc and "
        << RemappedHigh << " high_pc in .debug_info\n";
  return Error::success();
}

/// Remap the FDEs in the displaced output's `.eh_frame` so each still describes
/// its kernel after displacement. This object's FDEs are per-kernel with a
/// `"zR"` CIE using DW_EH_PE_pcrel|sdata4 for initial_location: the target is
/// `field_vaddr + sdata4`. For each FDE:
///   - initial_location: recompute sdata4 = new_entry_vaddr - field_vaddr,
///     where new_entry_vaddr = textAddr + mapOffset(old_entry - textAddr).
///   - address_range (u32 length): += growth of that kernel, i.e.
///     mapOffset(old_end) - mapOffset(old_entry) - old_range.
/// .eh_frame sits before .text and is not itself moved, so field_vaddr is
/// unchanged. Only the pcrel encoding (0x1b) is handled; any other CIE
/// augmentation/encoding is rejected so we never silently miswrite unwind data.
Error remapEhFrameForDisplacement(const ElfView &OldElf,
                                  const DisplacementPlan &Plan,
                                  WritableMemoryBuffer &OutBuf) {
  // Locate .eh_frame in the OLD ELF (section table not yet reparsed on Out; its
  // file offset in Out is identical because .eh_frame precedes .text).
  const ELFT::Shdr *EhShdr = nullptr;
  uint64_t EhVAddr = 0;
  for (const ELFT::Shdr &Shdr : OldElf.sections()) {
    Expected<StringRef> NameOrErr = OldElf.file().getSectionName(Shdr);
    if (!NameOrErr)
      return makeDisplacementError("failed to read section name for .eh_frame");
    if (*NameOrErr == ".eh_frame") {
      EhShdr = &Shdr;
      EhVAddr = Shdr.sh_addr;
      break;
    }
  }
  if (!EhShdr)
    return Error::success(); // no unwind tables to fix

  if (EhShdr->sh_offset > OutBuf.getBufferSize() ||
      EhShdr->sh_size > OutBuf.getBufferSize() - EhShdr->sh_offset)
    return makeDisplacementError(".eh_frame is out of bounds in displaced ELF");
  uint8_t *Base =
      reinterpret_cast<uint8_t *>(OutBuf.getBufferStart()) + EhShdr->sh_offset;
  const uint64_t Size = EhShdr->sh_size;
  const uint64_t OldTextEnd = OldElf.textAddr() + OldElf.textSize();

  auto readU32 = [&](uint64_t Off) -> uint32_t {
    uint32_t V;
    std::memcpy(&V, Base + Off, sizeof(V));
    return V;
  };
  auto writeU32 = [&](uint64_t Off, uint32_t V) {
    std::memcpy(Base + Off, &V, sizeof(V));
  };

  uint64_t Pos = 0;
  uint32_t Remapped = 0;
  while (Pos + 8 <= Size) {
    const uint32_t Length = readU32(Pos);
    if (Length == 0)
      break; // terminator
    if (Length == 0xffffffffu)
      return makeDisplacementError(".eh_frame uses 64-bit DWARF (unsupported)");
    const uint64_t RecordStart = Pos;
    const uint64_t RecordEnd = Pos + 4 + Length;
    if (RecordEnd > Size)
      return makeDisplacementError(".eh_frame record overruns section");
    const uint32_t CiePtr = readU32(Pos + 4);
    if (CiePtr == 0) {
      // CIE: skip (we assume the single pcrel|sdata4 CIE this compiler emits).
      Pos = RecordEnd;
      continue;
    }
    // FDE. initial_location is the 4-byte field at RecordStart+8.
    const uint64_t LocFieldOff = RecordStart + 8;
    const uint64_t RangeFieldOff = RecordStart + 12;
    const uint64_t LocFieldVAddr = EhVAddr + LocFieldOff;
    const int32_t StoredLoc = static_cast<int32_t>(readU32(LocFieldOff));
    const uint64_t OldTarget =
        static_cast<uint64_t>(static_cast<int64_t>(LocFieldVAddr) + StoredLoc);
    const uint32_t OldRange = readU32(RangeFieldOff);

    if (OldTarget < OldElf.textAddr() || OldTarget >= OldTextEnd) {
      // FDE does not describe .text (unexpected here); leave it untouched.
      Pos = RecordEnd;
      continue;
    }
    const uint64_t OldStartOff = OldTarget - OldElf.textAddr();
    const uint64_t OldEndOff = OldStartOff + OldRange;
    uint64_t NewStartOff = 0, NewEndOff = 0;
    if (!Plan.mapOffset(OldStartOff, DisplacementMapBias::BeforeInsertedBytes,
                        NewStartOff) ||
        !Plan.mapOffset(OldEndOff, DisplacementMapBias::BeforeInsertedBytes,
                        NewEndOff) ||
        NewEndOff < NewStartOff)
      return makeDisplacementError(".eh_frame FDE range cannot be remapped");

    const uint64_t NewTargetVAddr = OldElf.textAddr() + NewStartOff;
    const int64_t NewStoredLoc = static_cast<int64_t>(NewTargetVAddr) -
                                 static_cast<int64_t>(LocFieldVAddr);
    if (NewStoredLoc < std::numeric_limits<int32_t>::min() ||
        NewStoredLoc > std::numeric_limits<int32_t>::max())
      return makeDisplacementError(".eh_frame FDE pcrel offset overflows i32");
    const uint64_t NewRange = NewEndOff - NewStartOff;
    if (NewRange > std::numeric_limits<uint32_t>::max())
      return makeDisplacementError(".eh_frame FDE range overflows u32");

    writeU32(LocFieldOff, static_cast<uint32_t>(NewStoredLoc));
    writeU32(RangeFieldOff, static_cast<uint32_t>(NewRange));
    ++Remapped;
    Pos = RecordEnd;
  }
  log() << "hotswap: displacement: remapped " << Remapped
        << " .eh_frame FDE(s)\n";
  return Error::success();
}

// Read the CU's DW_AT_low_pc from the OLD .debug_info to use as the
// .debug_ranges base. low_pc is the first DW_FORM_addr after the CU header in
// abbrev-code-1 (compile_unit); we resolve it via the abbrev table rather than
// assuming a byte offset. Returns the low_pc, or std::nullopt if the CU has no
// .text low_pc (in which case .debug_ranges has nothing to rebase).
Expected<std::optional<uint64_t>> readCuLowPc(const uint8_t *InfoBase,
                                              uint64_t InfoSize,
                                              const uint8_t *AbbrevBase,
                                              uint64_t AbbrevSize) {
  if (InfoSize < 11)
    return std::optional<uint64_t>();
  uint32_t UnitLength = 0;
  std::memcpy(&UnitLength, InfoBase, 4);
  if (UnitLength == 0xffffffffu)
    return makeDisplacementError(".debug_info uses 64-bit DWARF (unsupported)");
  uint16_t Version = 0;
  std::memcpy(&Version, InfoBase + 4, 2);
  if (Version < 2 || Version > 4)
    return makeDisplacementError(".debug_info version is unsupported");
  uint32_t AbbrevOff = 0;
  std::memcpy(&AbbrevOff, InfoBase + 6, 4);
  uint8_t AddrSize = InfoBase[10];
  if (AddrSize != 8)
    return makeDisplacementError(".debug_info addr_size != 8 is unsupported");

  Expected<DenseMap<uint64_t, AbbrevDecl>> TableOrErr =
      parseAbbrevTable(AbbrevBase, AbbrevSize, AbbrevOff);
  if (!TableOrErr)
    return TableOrErr.takeError();
  const uint64_t UnitEnd = 4 + UnitLength;
  const uint8_t *End = InfoBase + std::min<uint64_t>(UnitEnd, InfoSize);
  uint64_t P = 11;
  unsigned N = 0;
  uint64_t Code = decodeULEB128(InfoBase + P, &N, End);
  P += N;
  DenseMap<uint64_t, AbbrevDecl>::const_iterator It = TableOrErr->find(Code);
  if (It == TableOrErr->end())
    return makeDisplacementError(
        ".debug_info root DIE has unknown abbrev code");
  dwarf::FormParams FP;
  FP.Version = Version;
  FP.AddrSize = AddrSize;
  FP.Format = dwarf::DWARF32;
  for (const AbbrevAttr &A : It->second.Attrs) {
    if (A.Attr == dwarf::DW_AT_low_pc && A.Form == dwarf::DW_FORM_addr) {
      uint64_t Low = 0;
      std::memcpy(&Low, InfoBase + P, sizeof(Low));
      return std::optional<uint64_t>(Low);
    }
    if (Error Err =
            skipFormValue(InfoBase, P, UnitEnd, A.Form, A.ImplicitConst, FP))
      return std::move(Err);
  }
  return std::optional<uint64_t>();
}

// Locate a debug section in the OUTPUT buffer and hand back a writable pointer
// plus size. Debug sections follow .text, so their output bytes live at
// OLD sh_offset + Growth. Returns nullptr Base (with success) when the section
// is absent.
Error locateOutputDebugSection(const ElfView &OldElf,
                               WritableMemoryBuffer &OutBuf, uint64_t Growth,
                               StringRef Name, uint8_t *&Base, uint64_t &Size) {
  Base = nullptr;
  Size = 0;
  const ELFT::Shdr *Shdr = findSectionByName(OldElf, Name);
  if (!Shdr)
    return Error::success();
  const uint64_t TextEnd = OldElf.textOffset() + OldElf.textSize();
  uint64_t OutOff = Shdr->sh_offset;
  if (Shdr->sh_offset >= TextEnd)
    OutOff += Growth;
  if (OutOff > OutBuf.getBufferSize() ||
      Shdr->sh_size > OutBuf.getBufferSize() - OutOff)
    return makeDisplacementError("debug section '" + Twine(Name) +
                                 "' is out of bounds in displaced ELF");
  Base = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart()) + OutOff;
  Size = Shdr->sh_size;
  return Error::success();
}

// Remap the .text addresses in the non-allocatable DWARF debug sections after a
// whole-object growth. This is the .debug counterpart to
// remapEhFrameForDisplacement
// and runs in the same in-place manner (no size change) for .debug_frame,
// .debug_info, and .debug_ranges. .debug_line, whose program length changes, is
// re-synthesized separately by the caller.
Error remapDebugSectionsForDisplacement(const ElfView &OldElf,
                                        const DisplacementPlan &Plan,
                                        WritableMemoryBuffer &OutBuf) {
  const uint64_t Growth = Plan.paddedGrowth();

  uint8_t *AbbrevBase = nullptr;
  uint64_t AbbrevSize = 0;
  if (Error Err = locateOutputDebugSection(
          OldElf, OutBuf, Growth, ".debug_abbrev", AbbrevBase, AbbrevSize))
    return Err;

  uint8_t *FrameBase = nullptr;
  uint64_t FrameSize = 0;
  if (Error Err = locateOutputDebugSection(
          OldElf, OutBuf, Growth, ".debug_frame", FrameBase, FrameSize))
    return Err;
  if (FrameBase)
    if (Error Err =
            remapDebugFrameForDisplacement(OldElf, Plan, FrameBase, FrameSize))
      return Err;

  uint8_t *InfoBase = nullptr;
  uint64_t InfoSize = 0;
  if (Error Err = locateOutputDebugSection(OldElf, OutBuf, Growth,
                                           ".debug_info", InfoBase, InfoSize))
    return Err;
  // Recover the CU base for .debug_ranges from the OLD low_pc (before we
  // rewrite it in place). low_pc is unchanged here, so old == new base.
  std::optional<uint64_t> CuBase;
  if (InfoBase) {
    if (!AbbrevBase)
      return makeDisplacementError(".debug_info present without .debug_abbrev");
    Expected<std::optional<uint64_t>> BaseOrErr =
        readCuLowPc(InfoBase, InfoSize, AbbrevBase, AbbrevSize);
    if (!BaseOrErr)
      return BaseOrErr.takeError();
    CuBase = *BaseOrErr;
    if (Error Err = remapDebugInfoForDisplacement(
            OldElf, Plan, InfoBase, InfoSize, AbbrevBase, AbbrevSize))
      return Err;
  }

  uint8_t *RangesBase = nullptr;
  uint64_t RangesSize = 0;
  if (Error Err = locateOutputDebugSection(
          OldElf, OutBuf, Growth, ".debug_ranges", RangesBase, RangesSize))
    return Err;
  if (RangesBase) {
    if (!CuBase)
      return makeDisplacementError(
          ".debug_ranges present without a CU .text low_pc base");
    if (Error Err = remapDebugRangesForDisplacement(OldElf, Plan, RangesBase,
                                                    RangesSize, *CuBase))
      return Err;
  }
  return Error::success();
}

// State needed to decode a .debug_line program's address-advancing opcodes.
struct LineProgramHeader {
  uint32_t UnitLength = 0;   // value of the first 4 bytes
  uint64_t UnitEnd = 0;      // 4 + UnitLength (one unit only; see below)
  uint64_t ProgramStart = 0; // first opcode after the prologue
  uint8_t MinInstLength = 0;
  uint8_t MaxOpsPerInst = 0;
  int8_t LineBase = 0;
  uint8_t LineRange = 0;
  uint8_t OpcodeBase = 0;
  SmallVector<uint8_t, 16> StdOpcodeLengths; // [1 .. OpcodeBase-1]
};

// Parse the (single) DWARF v2-v4 line-number program header enough to walk the
// opcode stream. Fails closed on DWARF64, VLIW (max_ops!=1), or a truncated
// header. Only one program unit is supported; a second unit is rejected by the
// caller.
Expected<LineProgramHeader> parseLineProgramHeader(const uint8_t *Base,
                                                   uint64_t Size) {
  if (Size < 15)
    return makeDisplacementError(".debug_line unit is too small");
  LineProgramHeader H;
  std::memcpy(&H.UnitLength, Base, 4);
  if (H.UnitLength == 0xffffffffu)
    return makeDisplacementError(".debug_line uses 64-bit DWARF (unsupported)");
  H.UnitEnd = 4 + static_cast<uint64_t>(H.UnitLength);
  if (H.UnitEnd > Size)
    return makeDisplacementError(".debug_line unit overruns section");
  uint16_t Version = 0;
  std::memcpy(&Version, Base + 4, 2);
  if (Version < 2 || Version > 4)
    return makeDisplacementError(".debug_line version " + Twine(Version) +
                                 " is unsupported");
  uint32_t HeaderLength = 0;
  std::memcpy(&HeaderLength, Base + 6, 4);
  const uint64_t AfterHeaderLen = 10;
  H.ProgramStart = AfterHeaderLen + HeaderLength;
  if (H.ProgramStart > H.UnitEnd)
    return makeDisplacementError(".debug_line prologue overruns unit");
  uint64_t P = AfterHeaderLen;
  H.MinInstLength = Base[P++];
  H.MaxOpsPerInst =
      Base[P++]; // v4+ field; present for the v4 objects we target
  if (H.MaxOpsPerInst != 1)
    return makeDisplacementError(".debug_line VLIW (max_ops_per_inst != 1) is "
                                 "unsupported");
  P++; // default_is_stmt
  H.LineBase = static_cast<int8_t>(Base[P++]);
  H.LineRange = Base[P++];
  H.OpcodeBase = Base[P++];
  if (H.LineRange == 0)
    return makeDisplacementError(".debug_line line_range is zero");
  for (unsigned I = 1; I < H.OpcodeBase; ++I) {
    if (P >= H.ProgramStart)
      return makeDisplacementError(
          ".debug_line standard_opcode_lengths overruns prologue");
    H.StdOpcodeLengths.push_back(Base[P++]);
  }
  return H;
}

// Re-synthesize a .debug_line program so every row's address is remapped
// through the displacement plan while the decoded (line, column, file, flags)
// values stay identical. The prologue is copied verbatim; the opcode stream is
// rebuilt. For each opcode we track the OLD running address and the desired NEW
// address (mapTextVAddr of the old address). An address advance is re-emitted
// as an explicit DW_LNS_advance_pc whenever the mapped delta differs from the
// encoded delta (i.e. the advance straddles a patch); otherwise the original
// bytes are copied. Special opcodes that must change their address advance are
// decomposed into DW_LNS_advance_pc + DW_LNS_advance_line + DW_LNS_copy,
// preserving the row. set_address operands are remapped directly. Returns the
// new full-section bytes. Fails closed on a second program unit or any opcode
// we do not model.
Expected<SmallVector<uint8_t>>
resynthesizeDebugLine(const ElfView &Elf, const DisplacementPlan &Plan,
                      const uint8_t *Base, uint64_t Size) {
  Expected<LineProgramHeader> HOrErr = parseLineProgramHeader(Base, Size);
  if (!HOrErr)
    return HOrErr.takeError();
  const LineProgramHeader &H = *HOrErr;
  if (H.UnitEnd != Size)
    return makeDisplacementError(
        ".debug_line has multiple program units (unsupported)");

  SmallVector<uint8_t> Out;
  // Copy the prologue (unit_length placeholder included) verbatim; unit_length
  // is fixed up at the end.
  Out.append(Base, Base + H.ProgramStart);

  auto emitByte = [&](uint8_t B) { Out.push_back(B); };
  // DW_LNS_advance_pc's operand is an "operation advance" scaled by
  // min_inst_length (max_ops_per_inst == 1 here), so the byte delta must be an
  // exact multiple of min_inst_length. All .text addresses are
  // min_inst-aligned, so this always holds; a non-multiple would be a decode
  // bug, so fail closed.
  auto emitAdvancePc = [&](uint64_t ByteDelta) -> Error {
    if (ByteDelta == 0)
      return Error::success();
    if (ByteDelta % H.MinInstLength != 0)
      return makeDisplacementError(
          ".debug_line address delta is not a multiple of min_inst_length");
    Out.push_back(dwarf::DW_LNS_advance_pc);
    uint8_t Buf[16];
    unsigned N = encodeULEB128(ByteDelta / H.MinInstLength, Buf);
    Out.append(Buf, Buf + N);
    return Error::success();
  };
  auto emitAdvanceLine = [&](int64_t Delta) {
    if (Delta == 0)
      return;
    Out.push_back(dwarf::DW_LNS_advance_line);
    uint8_t Buf[16];
    unsigned N = encodeSLEB128(Delta, Buf);
    Out.append(Buf, Buf + N);
  };

  const uint8_t *End = Base + Size;
  uint64_t P = H.ProgramStart;
  uint64_t OldAddr = 0; // running address in the input program
  uint64_t NewAddr = 0; // running address emitted so far in the rebuilt program
  // Pure address-advance opcodes (advance_pc/const_add_pc/fixed_advance_pc) are
  // NOT copied to the output; we only track OldAddr through them. Intermediate
  // stops may land inside a replaced instruction, so we never remap them. When
  // a ROW is about to be emitted (copy/special/end_sequence), reconcile: emit
  // one DW_LNS_advance_pc so the rebuilt address equals map(OldAddr). Only row
  // addresses are remapped, and a row inside a replaced range is a genuine
  // fail-closed condition (the row would name a deleted instruction).
  auto reconcileToRow = [&]() -> Error {
    uint64_t Mapped = 0;
    if (!mapTextVAddr(Elf, Plan, OldAddr, Mapped))
      return makeDisplacementError(
          ".debug_line row address is inside a replaced instruction");
    if (Mapped < NewAddr)
      return makeDisplacementError(
          ".debug_line address advance went backwards");
    if (Error Err = emitAdvancePc(Mapped - NewAddr))
      return Err;
    NewAddr = Mapped;
    return Error::success();
  };

  while (P < H.UnitEnd) {
    const uint8_t Op = Base[P];
    if (Op == 0) {
      // Extended opcode: 0, ULEB length, sub-opcode, body.
      unsigned N = 0;
      uint64_t Len = decodeULEB128(Base + P + 1, &N, End);
      const uint64_t BodyStart = P + 1 + N;
      const uint64_t Next = BodyStart + Len;
      if (Len == 0 || Next > H.UnitEnd)
        return makeDisplacementError(
            ".debug_line extended opcode is malformed");
      const uint8_t Sub = Base[BodyStart];
      if (Sub == dwarf::DW_LNE_set_address) {
        if (Len != 9)
          return makeDisplacementError(
              ".debug_line set_address is not 8 bytes (addr_size != 8)");
        uint64_t OldSet = 0;
        std::memcpy(&OldSet, Base + BodyStart + 1, sizeof(OldSet));
        uint64_t NewSet = 0;
        if (!mapTextVAddr(Elf, Plan, OldSet, NewSet))
          return makeDisplacementError(
              ".debug_line set_address cannot be remapped");
        // set_address assigns the running address absolutely in both streams.
        Out.push_back(0);
        Out.push_back(9);
        Out.push_back(dwarf::DW_LNE_set_address);
        uint8_t AddrBytes[8];
        std::memcpy(AddrBytes, &NewSet, sizeof(NewSet));
        Out.append(AddrBytes, AddrBytes + 8);
        OldAddr = OldSet;
        NewAddr = NewSet;
      } else if (Sub == dwarf::DW_LNE_end_sequence) {
        // end_sequence emits the closing row at the current address; reconcile
        // the rebuilt address to map(OldAddr) first, then copy the opcode.
        if (Error Err = reconcileToRow())
          return Err;
        Out.append(Base + P, Base + Next);
        OldAddr = 0;
        NewAddr = 0;
      } else {
        // Other extended opcodes (set_discriminator, define_file, vendor) do
        // not touch the address; copy verbatim.
        Out.append(Base + P, Base + Next);
      }
      P = Next;
      continue;
    }

    if (Op < H.OpcodeBase) {
      // Standard opcode.
      switch (Op) {
      case dwarf::DW_LNS_advance_pc: {
        unsigned N = 0;
        uint64_t Adv = decodeULEB128(Base + P + 1, &N, End);
        OldAddr += Adv * H.MinInstLength; // track only; do not emit
        P += 1 + N;
        break;
      }
      case dwarf::DW_LNS_fixed_advance_pc: {
        uint16_t Adv = 0;
        std::memcpy(&Adv, Base + P + 1, sizeof(Adv));
        OldAddr += Adv; // fixed_advance_pc is not scaled by min_inst_length
        P += 3;
        break;
      }
      case dwarf::DW_LNS_const_add_pc: {
        const uint8_t Adjust = (255 - H.OpcodeBase) / H.LineRange;
        OldAddr += static_cast<uint64_t>(Adjust) * H.MinInstLength;
        P += 1;
        break;
      }
      case dwarf::DW_LNS_copy:
        // Emits a row at the current address; catch the rebuilt address up.
        if (Error Err = reconcileToRow())
          return Err;
        emitByte(dwarf::DW_LNS_copy);
        P += 1;
        break;
      default: {
        // A standard opcode with no address effect: copy the opcode plus its
        // ULEB operands verbatim (operand count from standard_opcode_lengths).
        const uint8_t NumOperands = H.StdOpcodeLengths[Op - 1];
        uint64_t Q = P + 1;
        for (uint8_t I = 0; I < NumOperands; ++I) {
          unsigned N = 0;
          decodeULEB128(Base + Q, &N, End);
          Q += N;
          if (Q > H.UnitEnd)
            return makeDisplacementError(
                ".debug_line standard opcode operand overruns unit");
        }
        Out.append(Base + P, Base + Q);
        P = Q;
        break;
      }
      }
      continue;
    }

    // Special opcode: advances address and line and emits a row. Because we
    // drop the pure address-advance opcodes and only reconcile at rows, NewAddr
    // can lag OldAddr's mapped position here; a verbatim copy of the special
    // opcode would apply its own (small) address delta from the lagging NewAddr
    // and desync the stream. So always decompose: correct the address
    // explicitly to the mapped row target, apply the line delta, and emit the
    // row with DW_LNS_copy, which resets the same sticky flags
    // (discriminator/basic_block/prologue_end/epilogue_begin) a special opcode
    // does, so the decoded row is identical apart from the remapped address.
    const uint8_t Adjusted = Op - H.OpcodeBase;
    const uint64_t AddrAdv =
        static_cast<uint64_t>(Adjusted / H.LineRange) * H.MinInstLength;
    const int64_t LineAdv = H.LineBase + (Adjusted % H.LineRange);
    // Only the ROW address (OldAddr + AddrAdv) is remapped; the running base
    // may sit at an intermediate stop inside a replaced range, which we never
    // map.
    OldAddr += AddrAdv;
    uint64_t MappedTarget = 0;
    if (!mapTextVAddr(Elf, Plan, OldAddr, MappedTarget))
      return makeDisplacementError(
          ".debug_line special-opcode row is inside a replaced instruction");
    if (MappedTarget < NewAddr)
      return makeDisplacementError(".debug_line special opcode went backwards");
    if (Error Err = emitAdvancePc(MappedTarget - NewAddr))
      return Err;
    emitAdvanceLine(LineAdv);
    emitByte(dwarf::DW_LNS_copy);
    NewAddr = MappedTarget;
    P += 1;
  }

  // Fix unit_length = total bytes - 4.
  const uint32_t NewUnitLength = static_cast<uint32_t>(Out.size() - 4);
  std::memcpy(Out.data(), &NewUnitLength, sizeof(NewUnitLength));
  return Out;
}

// Shift the sh_offset of every section whose file offset is at or after
// \p Threshold by \p Delta, shift e_shoff if it is past the threshold, and set
// the size of the section at \p GrownIndex to \p GrownSize. Non-allocatable
// sections only: no sh_addr / program-header changes (the caller guarantees the
// grown section is non-allocatable, so no segment maps it).
Error shiftSectionOffsetsAfter(uint8_t *Elf, size_t ElfSize, uint64_t Threshold,
                               uint64_t Delta, uint16_t GrownIndex,
                               uint64_t GrownSize) {
  if (ElfSize < sizeof(Ehdr))
    return makeDisplacementError("displaced ELF is smaller than its header");
  uint64_t Shoff = 0;
  uint16_t Shentsize = 0, Shnum = 0;
  std::memcpy(&Shoff, Elf + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, Elf + offsetof(Ehdr, e_shentsize), sizeof(Shentsize));
  std::memcpy(&Shnum, Elf + offsetof(Ehdr, e_shnum), sizeof(Shnum));
  if (Shentsize < sizeof(Shdr))
    return makeDisplacementError("section-header entry is too small");
  if (Shoff >= Threshold) {
    uint64_t NewShoff = Shoff + Delta;
    std::memcpy(Elf + offsetof(Ehdr, e_shoff), &NewShoff, sizeof(NewShoff));
    Shoff = NewShoff;
  }
  for (uint16_t I = 0; I < Shnum; ++I) {
    uint64_t ShPos = Shoff + static_cast<uint64_t>(I) * Shentsize;
    if (ShPos > ElfSize || sizeof(Shdr) > ElfSize - ShPos)
      return makeDisplacementError("section-header table is out of bounds");
    uint8_t *Sh = Elf + ShPos;
    if (I == GrownIndex) {
      std::memcpy(Sh + offsetof(Shdr, sh_size), &GrownSize, sizeof(GrownSize));
      continue;
    }
    uint64_t ShOffset = 0;
    std::memcpy(&ShOffset, Sh + offsetof(Shdr, sh_offset), sizeof(ShOffset));
    if (ShOffset >= Threshold) {
      uint64_t NewOffset = ShOffset + Delta;
      std::memcpy(Sh + offsetof(Shdr, sh_offset), &NewOffset,
                  sizeof(NewOffset));
    }
  }
  return Error::success();
}

// Re-synthesize .debug_line (if present) and, when its byte length changed,
// return a NEW output buffer with the section spliced in and every trailing
// section's file offset shifted by the length delta. .debug_line is
// non-allocatable, so no vaddr/segment fixups are needed. Returns the original
// buffer unchanged when there is no .debug_line or its length did not change.
Expected<std::unique_ptr<WritableMemoryBuffer>>
growDebugLineForDisplacement(const ElfView &OldElf,
                             const DisplacementPlan &Plan,
                             std::unique_ptr<WritableMemoryBuffer> OutBuf) {
  const ELFT::Shdr *LineShdr = findSectionByName(OldElf, ".debug_line");
  if (!LineShdr)
    return std::move(OutBuf);

  const uint64_t Growth = Plan.paddedGrowth();
  const uint64_t TextEnd = OldElf.textOffset() + OldElf.textSize();
  const uint64_t OutLineOff =
      LineShdr->sh_offset + (LineShdr->sh_offset >= TextEnd ? Growth : 0);
  const uint64_t OldLineSize = LineShdr->sh_size;
  if (OutLineOff > OutBuf->getBufferSize() ||
      OldLineSize > OutBuf->getBufferSize() - OutLineOff)
    return makeDisplacementError(
        ".debug_line is out of bounds in displaced ELF");

  const uint8_t *OldLine =
      reinterpret_cast<const uint8_t *>(OutBuf->getBufferStart()) + OutLineOff;
  Expected<SmallVector<uint8_t>> NewLineOrErr =
      resynthesizeDebugLine(OldElf, Plan, OldLine, OldLineSize);
  if (!NewLineOrErr)
    return NewLineOrErr.takeError();
  SmallVector<uint8_t> &NewLine = *NewLineOrErr;

  if (NewLine.size() == OldLineSize) {
    // Same length: splice in place, no header shift.
    std::memcpy(reinterpret_cast<uint8_t *>(OutBuf->getBufferStart()) +
                    OutLineOff,
                NewLine.data(), NewLine.size());
    log() << "hotswap: displacement: .debug_line re-synthesized in place ("
          << NewLine.size() << " bytes)\n";
    return std::move(OutBuf);
  }
  if (NewLine.size() < OldLineSize)
    return makeDisplacementError(".debug_line re-synthesis shrank the program");

  const uint64_t LineDelta = NewLine.size() - OldLineSize;
  const size_t OldTotal = OutBuf->getBufferSize();
  const size_t NewTotal = OldTotal + LineDelta;
  std::unique_ptr<WritableMemoryBuffer> NewBuf =
      WritableMemoryBuffer::getNewUninitMemBuffer(NewTotal);
  if (!NewBuf)
    return makeDisplacementError("failed to allocate .debug_line grow buffer");

  const uint8_t *Src =
      reinterpret_cast<const uint8_t *>(OutBuf->getBufferStart());
  uint8_t *Dst = reinterpret_cast<uint8_t *>(NewBuf->getBufferStart());
  const uint64_t TailOff = OutLineOff + OldLineSize;
  std::memcpy(Dst, Src, OutLineOff);
  std::memcpy(Dst + OutLineOff, NewLine.data(), NewLine.size());
  std::memcpy(Dst + OutLineOff + NewLine.size(), Src + TailOff,
              OldTotal - TailOff);

  // Shift section offsets/shoff for everything after .debug_line's old bytes.
  const uint16_t LineIndex =
      static_cast<uint16_t>(LineShdr - OldElf.sections().begin());
  if (Error Err = shiftSectionOffsetsAfter(Dst, NewTotal, TailOff, LineDelta,
                                           LineIndex, NewLine.size()))
    return std::move(Err);
  log() << "hotswap: displacement: .debug_line re-synthesized, grew "
        << OldLineSize << " -> " << NewLine.size() << " bytes (+" << LineDelta
        << "); shifted trailing sections\n";
  return std::move(NewBuf);
}

Error validateDisplacementEditBoundaries(const ElfView &Elf,
                                         const LLVMState &LS,
                                         const DisplacementPlan &Plan) {
  size_t EditIndex = 0;
  std::optional<Error> BoundaryError;
  bool Decoded = decodeTextSectionStreaming(
      Elf.textData(), Elf.textSize(), LS, /*WantMnemonic=*/false,
      [&](const InternalDecodedInst &DI) {
        if (!DI.DecodeSucceeded) {
          BoundaryError.emplace(makeDisplacementError(
              "old .text contains an undecodable instruction"));
          return false;
        }
        while (EditIndex != Plan.edits().size() &&
               Plan.edits()[EditIndex].Offset == DI.Offset &&
               Plan.edits()[EditIndex].OriginalSize == 0)
          ++EditIndex;
        if (EditIndex != Plan.edits().size() &&
            Plan.edits()[EditIndex].Offset == DI.Offset) {
          const DisplacementEdit &Edit = Plan.edits()[EditIndex];
          if (Edit.OriginalSize != DI.Size) {
            BoundaryError.emplace(makeDisplacementError(
                "growing edit does not replace exactly one decoded "
                "instruction"));
            return false;
          }
          ++EditIndex;
        }
        if (EditIndex != Plan.edits().size() &&
            Plan.edits()[EditIndex].Offset < DI.Offset + DI.Size) {
          BoundaryError.emplace(makeDisplacementError(
              "displacement edit begins inside a decoded instruction"));
          return false;
        }
        return true;
      });
  if (BoundaryError)
    return std::move(*BoundaryError);
  if (!Decoded)
    return makeDisplacementError(
        "failed to decode .text while validating edit boundaries");
  while (EditIndex != Plan.edits().size() &&
         Plan.edits()[EditIndex].Offset == Elf.textSize() &&
         Plan.edits()[EditIndex].OriginalSize == 0)
    ++EditIndex;
  if (EditIndex != Plan.edits().size())
    return makeDisplacementError(
        "displacement edit is not on a decoded instruction boundary");

  for (const DisplacementEdit &Edit : Plan.edits()) {
    std::optional<Error> ReplacementError;
    bool ReplacementDecoded = decodeTextSectionStreaming(
        Edit.ReplacementBytes.data(), Edit.ReplacementBytes.size(), LS,
        /*WantMnemonic=*/false, [&](const InternalDecodedInst &DI) {
          if (!DI.DecodeSucceeded) {
            ReplacementError.emplace(makeDisplacementError(
                "replacement contains an undecodable instruction"));
            return false;
          }
          if (isPcSensitiveForDisplacement(DI, LS) ||
              LS.MIA->isBranch(DI.Inst) || LS.MIA->isCall(DI.Inst) ||
              LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst)) {
            ReplacementError.emplace(makeDisplacementError(
                "replacement contains unrelocated control flow or a "
                "PC-sensitive instruction"));
            return false;
          }
          return true;
        });
    if (ReplacementError)
      return std::move(*ReplacementError);
    if (!ReplacementDecoded)
      return makeDisplacementError(
          "replacement bytes do not form a complete instruction sequence");
  }
  return Error::success();
}

Error applyTextDisplacement(const ElfView &Elf, const LLVMState &LS,
                            const DisplacementPlan &Plan,
                            const DirectControlFlowInfo &ControlFlow,
                            WritableMemoryBuffer &OutBuf) {
  const size_t InputSize = Elf.size();
  const size_t NewSize = Plan.newElfSize(InputSize);
  if (OutBuf.getBufferSize() != NewSize) {
    return makeDisplacementError(
        "output buffer has incorrect size for displacement");
  }

  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::duration D) {
    return std::chrono::duration<double, std::milli>(D).count();
  };
  Clock::time_point T0 = Clock::now();

  SmallVector<uint8_t> NewText = Plan.buildText(
      ArrayRef<uint8_t>(Elf.textData(), Elf.textSize()), LS.SNopBytes);
  if (NewText.size() != Plan.paddedTextSize()) {
    return makeDisplacementError(
        "rebuilt .text size does not match displacement plan");
  }
  Clock::time_point T1 = Clock::now();
  log() << "hotswap: TIMING   displacement buildText: " << ms(T1 - T0)
        << " ms\n";
  if (Error Err = repairBranches(Elf, LS, Plan, ControlFlow, NewText))
    return Err;
  bool ValidNewText = true;
  bool NewTextDecoded = decodeTextSectionStreaming(
      NewText.data(), NewText.size(), LS, /*WantMnemonic=*/false,
      [&](const InternalDecodedInst &DI) {
        ValidNewText &= DI.DecodeSucceeded;
        return ValidNewText;
      });
  if (!NewTextDecoded || !ValidNewText)
    return makeDisplacementError(
        "rebuilt .text does not form a complete instruction sequence");
  Clock::time_point T2 = Clock::now();
  log() << "hotswap: TIMING   displacement repairBranches "
           "(whole .text redecode): "
        << ms(T2 - T1) << " ms\n";

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

  const bool RelocateVAddr = Plan.relocatesTrailingSections();
  if (Error Err = adjustSectionHeadersForTextGrowth(
          Out, NewSize, Elf, Plan.paddedGrowth(), RelocateVAddr))
    return Err;
  if (Error Err = adjustProgramHeadersForTextGrowth(
          Out, NewSize, Elf, Plan.paddedGrowth(), RelocateVAddr))
    return Err;
  if (Error Err = adjustElfEntryForDisplacement(Out, NewSize, Elf, Plan))
    return Err;
  if (RelocateVAddr)
    if (Error Err =
            adjustDynamicEntriesForDisplacement(Out, NewSize, Elf, Plan))
      return Err;
  if (Error Err = adjustRelocationsForDisplacement(Out, NewSize, Elf, Plan))
    return Err;
  if (Error Err = adjustSymbolValuesForDisplacement(Out, NewSize, Elf, Plan))
    return Err;
  Clock::time_point T3 = Clock::now();
  log() << "hotswap: TIMING   displacement memcpy+metadata: " << ms(T3 - T2)
        << " ms\n";

  if (Error Err =
          rewriteKernelDescriptorEntriesForDisplacement(OutBuf, Elf, Plan))
    return Err;
  Clock::time_point T4 = Clock::now();
  log() << "hotswap: TIMING   displacement descriptor rewrite: " << ms(T4 - T3)
        << " ms\n";

  // .eh_frame FDEs describe per-kernel .text ranges; fix them to the new layout
  // when the caller opted into relocation. validateDebugSections only
  // admits .eh_frame in that mode, so this is the matching repair.
  if (RelocateVAddr)
    if (Error Err = remapEhFrameForDisplacement(Elf, Plan, OutBuf))
      return Err;
  log() << "hotswap: TIMING   displacement eh_frame remap: "
        << ms(Clock::now() - T4) << " ms\n";

  // Non-allocatable DWARF debug sections (.debug_frame/.debug_info/
  // .debug_ranges) reference .text addresses that are now stale; rewrite them
  // in place. validateDebugSections only admits these sections in this mode.
  // .debug_line is re-synthesized separately by the caller because its byte
  // length changes.
  if (RelocateVAddr)
    if (Error Err = remapDebugSectionsForDisplacement(Elf, Plan, OutBuf))
      return Err;

  log() << "hotswap: displacement: grew ELF from " << InputSize << " to "
        << NewSize << " bytes (" << Plan.edits().size() << " edit"
        << (Plan.edits().size() == 1 ? "" : "s") << ", raw growth "
        << Plan.rawGrowth() << " bytes, padded growth " << Plan.paddedGrowth()
        << " bytes).\n";
  return Error::success();
}

} // namespace

Expected<std::vector<RelocationTableDispatch>>
analyzeRelocationTableDispatches(const ElfView &Elf,
                                 ArrayRef<InternalDecodedInst> Decoded,
                                 const LLVMState &LS) {
  Expected<std::vector<RelocationTableCandidate>> TablesOrErr =
      discoverCompleteRelocationTables(Elf);
  if (!TablesOrErr)
    return TablesOrErr.takeError();
  return matchRelocationTableDispatches(Elf, Decoded, LS, *TablesOrErr);
}

Expected<DisplacementPlan>
DisplacementPlan::create(const ElfView &Elf,
                         ArrayRef<DisplacementEdit> InputEdits,
                         bool RelocateTrailingSections) {
  if (InputEdits.empty())
    return makeDisplacementError("no displacement edits requested");

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
    if (Edit.MapsOldOffsetAfterInsertion && Edit.OriginalSize != 0)
      return makeDisplacementError(
          "boundary-after displacement edit is not a pure insertion");
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
      if (!PrevEdit || !PrevEdit->MapsOldOffsetAfterInsertion ||
          PrevEdit->OriginalSize != 0 || Edit.OriginalSize == 0)
        return makeDisplacementError(
            "multiple displacement edits share an offset");
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
  // When relocating trailing sections, a grown .text that would
  // overlap later allocatable content is fine: the apply step shifts that
  // content's vaddr forward. Skip the overlap bail in that mode.
  if (!RelocateTrailingSections)
    if (Error Err = validateVirtualGrowth(Elf, PaddedGrowth))
      return std::move(Err);
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

Expected<std::unique_ptr<WritableMemoryBuffer>>
tryApplyTextDisplacementToNewBuffer(const ElfView &Elf, const LLVMState &LS,
                                    ArrayRef<DisplacementEdit> Edits,
                                    bool RelocateTrailingSections) {
  if (RelocateTrailingSections)
    if (Error Err = validateWholeObjectLayout(Elf))
      return std::move(Err);
  if (Error Err = validateDebugSections(
          Elf, /*AllowEhFrameRemap=*/RelocateTrailingSections))
    return std::move(Err);
  if (Error Err = validateTextRelocations(Elf))
    return std::move(Err);

  Expected<DisplacementPlan> PlanOrErr =
      DisplacementPlan::create(Elf, Edits, RelocateTrailingSections);
  if (!PlanOrErr)
    return PlanOrErr.takeError();
  if (Error Err = validateDisplacementEditBoundaries(Elf, LS, *PlanOrErr))
    return std::move(Err);
  if (Error Err = validateKernelEntryMappings(Elf, *PlanOrErr))
    return std::move(Err);

  DirectControlFlowInfo ControlFlow;
  bool NeedsLinkedControlFlowProof = false;
  struct IndirectTransferScanner {
    const LLVMState &LS;
    bool &Found;

    bool operator()(const InternalDecodedInst &DI) const {
      Found |= DI.Inst.getOpcode() == LS.SSwapPcI64Opcode ||
               DI.Inst.getOpcode() == LS.SSetPcI64Opcode;
      return true;
    }
  };
  IndirectTransferScanner FindIndirectTransfer{LS, NeedsLinkedControlFlowProof};
  if (!decodeTextSectionStreaming(Elf.textData(), Elf.textSize(), LS,
                                  /*WantMnemonic=*/false, FindIndirectTransfer))
    return makeDisplacementError(
        "failed to scan .text for indirect control flow");
  if (NeedsLinkedControlFlowProof) {
    std::vector<InternalDecodedInst> Decoded;
    if (!decodeTextSection(Elf.textData(), Elf.textSize(), LS, Decoded))
      return makeDisplacementError(
          "failed to decode .text for linked control-flow proof");
    std::optional<DirectControlFlowInfo> Info =
        analyzeDirectControlFlow(Elf, Decoded, LS);
    if (!Info)
      return makeDisplacementError("linked control-flow proof failed");
    ControlFlow = std::move(*Info);
  }

  std::unique_ptr<WritableMemoryBuffer> Out =
      WritableMemoryBuffer::getNewUninitMemBuffer(
          PlanOrErr->newElfSize(Elf.size()));
  if (!Out) {
    return makeDisplacementError(
        "failed to allocate displacement output buffer");
  }
  if (Error Err = applyTextDisplacement(Elf, LS, *PlanOrErr, ControlFlow, *Out))
    return std::move(Err);

  // .debug_line's program length can change when an address advance straddles a
  // patch; re-synthesize it and splice the (possibly larger) section back in,
  // shifting trailing non-allocatable sections. Only relevant when
  // validateDebugSections admitted the debug sections.
  if (RelocateTrailingSections) {
    Expected<std::unique_ptr<WritableMemoryBuffer>> GrownOrErr =
        growDebugLineForDisplacement(Elf, *PlanOrErr, std::move(Out));
    if (!GrownOrErr)
      return GrownOrErr.takeError();
    return std::move(*GrownOrErr);
  }
  return std::move(Out);
}

} // namespace hotswap
} // namespace COMGR
