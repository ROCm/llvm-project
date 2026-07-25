//===- comgr-hotswap-eh-frame.cpp - HotSwap unwind remapping ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Validated `.eh_frame` remapping for HotSwap text displacement.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"

#include <cstring>
#include <limits>
#include <vector>

using namespace llvm;

namespace COMGR {
namespace hotswap {

namespace {

using ELFT = ElfView::ELFT;

struct EhFramePatch {
  uint64_t LocationOffset;
  int32_t Location;
  uint64_t RangeOffset;
  uint32_t Range;
};

Error makeEhFrameError(const Twine &Msg) {
  std::string Message = Msg.str();
  log() << "hotswap: displacement unavailable: " << Message << "\n";
  return createStringError(object::object_error::parse_failed, Message);
}

Expected<const ELFT::Shdr *> findEhFrameSection(const ElfView &Elf) {
  const ELFT::Shdr *Found = nullptr;
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    Expected<StringRef> NameOrErr = Elf.file().getSectionName(Shdr);
    if (!NameOrErr) {
      return makeEhFrameError(
          "failed to read a section name while locating .eh_frame: " +
          Twine(toString(NameOrErr.takeError())));
    }
    if (*NameOrErr != ".eh_frame")
      continue;
    if (Found)
      return makeEhFrameError("ELF contains multiple .eh_frame sections");
    Found = &Shdr;
  }
  return Found;
}

Expected<bool> cfiSetLocationRequiresRemap(const dwarf::CFIProgram &Program,
                                           const DisplacementPlan &Plan,
                                           uint64_t OldTextBegin,
                                           uint64_t OldTextEnd) {
  for (const dwarf::CFIProgram::Instruction &Inst : Program) {
    if (Inst.Opcode != dwarf::DW_CFA_set_loc)
      continue;
    if (Inst.Ops.empty())
      return makeEhFrameError(".eh_frame has malformed DW_CFA_set_loc");
    const uint64_t OldLocation = Inst.Ops.front();
    if (OldLocation < OldTextBegin || OldLocation >= OldTextEnd)
      continue;
    uint64_t NewOffset = 0;
    if (!Plan.mapOffset(OldLocation - OldTextBegin,
                        DisplacementMapBias::AfterInsertedBytes, NewOffset) ||
        NewOffset != OldLocation - OldTextBegin) {
      return true;
    }
  }
  return false;
}

bool advancesCfiLocation(const dwarf::CFIProgram &Program) {
  for (const dwarf::CFIProgram::Instruction &Inst : Program) {
    switch (Inst.Opcode) {
    case dwarf::DW_CFA_advance_loc:
    case dwarf::DW_CFA_advance_loc1:
    case dwarf::DW_CFA_advance_loc2:
    case dwarf::DW_CFA_advance_loc4:
    case dwarf::DW_CFA_MIPS_advance_loc8:
      return true;
    default:
      break;
    }
  }
  return false;
}

Expected<bool>
hasUnsupportedAddressExpression(const dwarf::CFIProgram &Program) {
  for (const dwarf::CFIProgram::Instruction &Inst : Program) {
    if (!Inst.Expression)
      continue;
    for (const DWARFExpression::Operation &Op : *Inst.Expression) {
      if (Op.isError())
        return makeEhFrameError(
            ".eh_frame contains a malformed CFI expression");
      if (Op.getCode() == dwarf::DW_OP_addr ||
          Op.getCode() == dwarf::DW_OP_addrx) {
        return true;
      }
    }
  }
  return false;
}

Expected<int32_t> encodePcRelativeSData4(uint64_t Target,
                                         uint64_t FieldAddress) {
  if (Target >= FieldAddress) {
    uint64_t Delta = Target - FieldAddress;
    if (Delta > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
      return makeEhFrameError(".eh_frame FDE pcrel offset overflows i32");
    return static_cast<int32_t>(Delta);
  }

  uint64_t Delta = FieldAddress - Target;
  const uint64_t NegativeLimit = uint64_t{1}
                                 << (std::numeric_limits<int32_t>::digits);
  if (Delta > NegativeLimit)
    return makeEhFrameError(".eh_frame FDE pcrel offset overflows i32");
  if (Delta == NegativeLimit)
    return std::numeric_limits<int32_t>::min();
  return -static_cast<int32_t>(Delta);
}

Error validateEhFrameRecordBounds(StringRef Data) {
  uint64_t Offset = 0;
  while (Offset < Data.size()) {
    if (sizeof(uint32_t) > Data.size() - Offset)
      return makeEhFrameError(".eh_frame has a truncated record length");

    uint32_t RecordLength = 0;
    std::memcpy(&RecordLength, Data.data() + Offset, sizeof(RecordLength));
    Offset += sizeof(RecordLength);
    if (RecordLength == 0) {
      for (; Offset < Data.size(); ++Offset) {
        if (Data[Offset] != 0)
          return makeEhFrameError(
              ".eh_frame has nonzero data after its terminator");
      }
      return Error::success();
    }

    uint64_t ExtendedLength = RecordLength;
    if (RecordLength == dwarf::DW_LENGTH_DWARF64) {
      if (sizeof(uint64_t) > Data.size() - Offset) {
        return makeEhFrameError(
            ".eh_frame has a truncated 64-bit record length");
      }
      std::memcpy(&ExtendedLength, Data.data() + Offset,
                  sizeof(ExtendedLength));
      Offset += sizeof(ExtendedLength);
    }
    if (ExtendedLength > Data.size() - Offset)
      return makeEhFrameError(".eh_frame has a truncated record");
    Offset += ExtendedLength;
  }
  return Error::success();
}

} // namespace

Error remapEhFrameForDisplacement(const ElfView &OldElf,
                                  const DisplacementPlan &Plan,
                                  WritableMemoryBuffer &OutBuf) {
  // Parse through LLVM so every FDE is tied to its actual CIE instead of
  // guessing a record layout. The fixed-width pcrel|sdata4 address fields can
  // be rewritten in place. Location-changing CFI is accepted only when its
  // encoded locations remain valid; changing those instruction streams could
  // alter record sizes and is outside this remapper's contract.
  Expected<const ELFT::Shdr *> OldShdrOrErr = findEhFrameSection(OldElf);
  if (!OldShdrOrErr)
    return OldShdrOrErr.takeError();
  const ELFT::Shdr *OldShdr = *OldShdrOrErr;
  if (!OldShdr)
    return Error::success();

  if (OldShdr->sh_offset > OldElf.size() ||
      OldShdr->sh_size > OldElf.size() - OldShdr->sh_offset) {
    return makeEhFrameError(".eh_frame is outside the input ELF");
  }
  if (OldShdr->sh_type != ELF::SHT_PROGBITS ||
      !(OldShdr->sh_flags & ELF::SHF_ALLOC) ||
      (OldShdr->sh_flags & ELF::SHF_COMPRESSED)) {
    return makeEhFrameError(
        ".eh_frame must be an allocated, uncompressed SHT_PROGBITS section");
  }

  uint32_t OldEhFrameIndex = 0;
  bool FoundOldEhFrameIndex = false;
  for (const ELFT::Shdr &Shdr : OldElf.sections()) {
    if (&Shdr == OldShdr) {
      FoundOldEhFrameIndex = true;
      break;
    }
    ++OldEhFrameIndex;
  }
  if (!FoundOldEhFrameIndex)
    return makeEhFrameError("failed to resolve .eh_frame section index");
  for (const ELFT::Shdr &Shdr : OldElf.sections()) {
    if ((Shdr.sh_type == ELF::SHT_REL || Shdr.sh_type == ELF::SHT_RELA) &&
        Shdr.sh_info == OldEhFrameIndex) {
      return makeEhFrameError(".eh_frame relocation records are unsupported");
    }
  }

  uint8_t *OutData = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart());
  Expected<ElfView> OutElfOrErr =
      ElfView::create(OutData, OutBuf.getBufferSize());
  if (!OutElfOrErr) {
    return makeEhFrameError(
        "failed to parse displaced ELF while locating .eh_frame: " +
        Twine(toString(OutElfOrErr.takeError())));
  }

  Expected<const ELFT::Shdr *> OutShdrOrErr = findEhFrameSection(*OutElfOrErr);
  if (!OutShdrOrErr)
    return OutShdrOrErr.takeError();
  const ELFT::Shdr *OutShdr = *OutShdrOrErr;
  if (!OutShdr)
    return makeEhFrameError(".eh_frame disappeared from displaced ELF");
  if (OutShdr->sh_size != OldShdr->sh_size)
    return makeEhFrameError(".eh_frame size changed during displacement");
  if (OutShdr->sh_type != ELF::SHT_PROGBITS ||
      !(OutShdr->sh_flags & ELF::SHF_ALLOC) ||
      (OutShdr->sh_flags & ELF::SHF_COMPRESSED)) {
    return makeEhFrameError(
        "displaced .eh_frame is not an allocated, uncompressed SHT_PROGBITS "
        "section");
  }
  if (OutShdr->sh_offset > OutBuf.getBufferSize() ||
      OutShdr->sh_size > OutBuf.getBufferSize() - OutShdr->sh_offset) {
    return makeEhFrameError(".eh_frame is outside the displaced ELF");
  }

  StringRef OldData(
      reinterpret_cast<const char *>(OldElf.data() + OldShdr->sh_offset),
      OldShdr->sh_size);
  if (Error Err = validateEhFrameRecordBounds(OldData))
    return Err;
  DWARFDataExtractor Extractor(OldData, /*IsLittleEndian=*/true,
                               /*AddressSize=*/sizeof(uint64_t));
  const Triple::ArchType Arch = Triple("amdgcn-amd-amdhsa").getArch();
  DWARFDebugFrame Frame(Arch, /*IsEH=*/true, OldShdr->sh_addr);
  if (Error Err = Frame.parse(Extractor)) {
    return makeEhFrameError("failed to parse .eh_frame: " +
                            Twine(toString(std::move(Err))));
  }

  if (OldElf.textSize() >
      std::numeric_limits<uint64_t>::max() - OldElf.textAddr()) {
    return makeEhFrameError(".text range overflows while remapping .eh_frame");
  }
  const uint64_t OldTextBegin = OldElf.textAddr();
  const uint64_t OldTextEnd = OldTextBegin + OldElf.textSize();
  const uint64_t NewTextBegin = OutElfOrErr->textAddr();

  std::vector<EhFramePatch> Patches;
  for (const dwarf::FrameEntry &Entry : Frame.entries()) {
    if (Entry.getKind() == dwarf::FrameEntry::FK_CIE) {
      const dwarf::CIE &Cie = static_cast<const dwarf::CIE &>(Entry);
      if (Cie.getPersonalityAddress()) {
        return makeEhFrameError(
            ".eh_frame personality pointers are unsupported");
      }
      Expected<bool> AddressExpressionOrErr =
          hasUnsupportedAddressExpression(Cie.cfis());
      if (!AddressExpressionOrErr)
        return AddressExpressionOrErr.takeError();
      if (*AddressExpressionOrErr) {
        return makeEhFrameError(
            ".eh_frame CFI address expressions are unsupported");
      }
      Expected<bool> SetLocOrErr = cfiSetLocationRequiresRemap(
          Cie.cfis(), Plan, OldTextBegin, OldTextEnd);
      if (!SetLocOrErr)
        return SetLocOrErr.takeError();
      if (*SetLocOrErr) {
        return makeEhFrameError(
            ".eh_frame CIE has DW_CFA_set_loc that requires remapping");
      }
      continue;
    }
    const dwarf::FDE &Fde = static_cast<const dwarf::FDE &>(Entry);
    const dwarf::CIE *Cie = Fde.getLinkedCIE();
    if (!Cie)
      return makeEhFrameError(".eh_frame FDE does not reference a CIE");

    const uint32_t Encoding = Cie->getFDEPointerEncoding();
    if (Encoding != (dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4)) {
      return makeEhFrameError(
          ".eh_frame FDE pointer encoding is unsupported (expected "
          "DW_EH_PE_pcrel|DW_EH_PE_sdata4)");
    }
    if (Fde.getLSDAAddress())
      return makeEhFrameError(".eh_frame LSDA pointers are unsupported");
    Expected<bool> FdeAddressExpressionOrErr =
        hasUnsupportedAddressExpression(Fde.cfis());
    if (!FdeAddressExpressionOrErr)
      return FdeAddressExpressionOrErr.takeError();
    if (*FdeAddressExpressionOrErr) {
      return makeEhFrameError(
          ".eh_frame CFI address expressions are unsupported");
    }

    const uint64_t RecordOffset = Fde.getOffset();
    if (RecordOffset > OldShdr->sh_size ||
        sizeof(uint32_t) > OldShdr->sh_size - RecordOffset) {
      return makeEhFrameError(".eh_frame FDE header is out of bounds");
    }
    uint32_t RecordLength = 0;
    std::memcpy(&RecordLength, OldData.data() + RecordOffset,
                sizeof(RecordLength));
    if (RecordLength == dwarf::DW_LENGTH_DWARF64) {
      return makeEhFrameError(
          ".eh_frame uses 64-bit DWARF records (unsupported)");
    }
    if (RecordOffset > std::numeric_limits<uint64_t>::max() - 4 ||
        RecordLength > OldShdr->sh_size - RecordOffset - 4) {
      return makeEhFrameError(".eh_frame FDE record is out of bounds");
    }
    const uint64_t RecordEnd = RecordOffset + 4 + RecordLength;
    const uint64_t LocationOffset = RecordOffset + 8;
    const uint64_t RangeOffset = LocationOffset + sizeof(int32_t);
    if (RangeOffset > RecordEnd || sizeof(uint32_t) > RecordEnd - RangeOffset) {
      return makeEhFrameError(".eh_frame FDE address fields are truncated");
    }

    const uint64_t OldStart = Fde.getInitialLocation();
    const uint64_t OldRange = Fde.getAddressRange();
    if (OldRange > std::numeric_limits<uint64_t>::max() - OldStart)
      return makeEhFrameError(".eh_frame FDE address range overflows");
    const uint64_t OldEnd = OldStart + OldRange;

    Expected<bool> FdeSetLocOrErr =
        cfiSetLocationRequiresRemap(Fde.cfis(), Plan, OldTextBegin, OldTextEnd);
    if (!FdeSetLocOrErr)
      return FdeSetLocOrErr.takeError();
    if (*FdeSetLocOrErr) {
      return makeEhFrameError(
          ".eh_frame FDE has DW_CFA_set_loc that requires remapping");
    }

    const bool DescribesText =
        OldRange == 0 ? OldStart >= OldTextBegin && OldStart < OldTextEnd
                      : OldEnd > OldTextBegin && OldStart < OldTextEnd;
    if (!DescribesText) {
      if (OutShdr->sh_addr != OldShdr->sh_addr) {
        return makeEhFrameError("moved .eh_frame has an FDE outside .text");
      }

      uint32_t SectionIndex = 0;
      for (const ELFT::Shdr &OldTargetShdr : OldElf.sections()) {
        if (OldTargetShdr.sh_size >
            std::numeric_limits<uint64_t>::max() - OldTargetShdr.sh_addr) {
          return makeEhFrameError(
              "section address range overflows while checking .eh_frame");
        }
        const uint64_t SectionEnd =
            OldTargetShdr.sh_addr + OldTargetShdr.sh_size;
        const bool ContainsFde =
            OldRange == 0
                ? OldStart >= OldTargetShdr.sh_addr && OldStart < SectionEnd
                : OldStart >= OldTargetShdr.sh_addr && OldEnd <= SectionEnd;
        if (!ContainsFde) {
          ++SectionIndex;
          continue;
        }

        Expected<const ELFT::Shdr *> OutTargetShdrOrErr =
            OutElfOrErr->file().getSection(SectionIndex);
        if (!OutTargetShdrOrErr) {
          return makeEhFrameError(
              "failed to find displaced section containing an .eh_frame "
              "FDE: " +
              Twine(toString(OutTargetShdrOrErr.takeError())));
        }
        if ((*OutTargetShdrOrErr)->sh_addr != OldTargetShdr.sh_addr) {
          return makeEhFrameError(
              ".eh_frame FDE outside .text targets a moved section");
        }
        break;
      }
      continue;
    }
    if (OldStart < OldTextBegin || OldEnd > OldTextEnd) {
      return makeEhFrameError(".eh_frame FDE partially overlaps .text");
    }

    const uint64_t OldStartOffset = OldStart - OldTextBegin;
    const uint64_t OldEndOffset = OldEnd - OldTextBegin;
    uint64_t NewStartOffset = 0;
    uint64_t NewEndOffset = 0;
    if (!Plan.mapOffset(OldStartOffset,
                        DisplacementMapBias::BeforeInsertedBytes,
                        NewStartOffset) ||
        !Plan.mapOffset(OldEndOffset, DisplacementMapBias::BeforeInsertedBytes,
                        NewEndOffset) ||
        NewEndOffset < NewStartOffset) {
      return makeEhFrameError(".eh_frame FDE range cannot be remapped");
    }

    const uint64_t NewRange = NewEndOffset - NewStartOffset;
    if (NewRange > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return makeEhFrameError(".eh_frame FDE range overflows sdata4");
    }
    if (NewRange != OldRange &&
        (advancesCfiLocation(Cie->cfis()) || advancesCfiLocation(Fde.cfis()))) {
      return makeEhFrameError(
          ".eh_frame FDE has location-changing CFI that requires remapping");
    }

    if (NewStartOffset > std::numeric_limits<uint64_t>::max() - NewTextBegin) {
      return makeEhFrameError(".eh_frame FDE start address overflows");
    }
    const uint64_t NewStart = NewTextBegin + NewStartOffset;
    if (NewRange > std::numeric_limits<uint64_t>::max() - NewStart)
      return makeEhFrameError(".eh_frame FDE end address overflows");
    if (LocationOffset >
        std::numeric_limits<uint64_t>::max() - OutShdr->sh_addr) {
      return makeEhFrameError(".eh_frame FDE field address overflows");
    }
    Expected<int32_t> NewLocationOrErr =
        encodePcRelativeSData4(NewStart, OutShdr->sh_addr + LocationOffset);
    if (!NewLocationOrErr)
      return NewLocationOrErr.takeError();

    Patches.push_back({OutShdr->sh_offset + LocationOffset, *NewLocationOrErr,
                       OutShdr->sh_offset + RangeOffset,
                       static_cast<uint32_t>(NewRange)});
  }

  for (const EhFramePatch &Patch : Patches) {
    std::memcpy(OutData + Patch.LocationOffset, &Patch.Location,
                sizeof(Patch.Location));
    std::memcpy(OutData + Patch.RangeOffset, &Patch.Range, sizeof(Patch.Range));
  }
  log() << "hotswap: displacement: remapped " << Patches.size()
        << " .eh_frame FDE(s)\n";
  return Error::success();
}

} // namespace hotswap
} // namespace COMGR
