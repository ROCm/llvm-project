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
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

using namespace llvm;

namespace COMGR {
namespace hotswap {

namespace {

using ELFT = ElfView::ELFT;

struct EhFramePatch {
  uint64_t Offset;
  SmallVector<uint8_t, 8> Bytes;
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

Expected<uint64_t> amdgpuRelocationWriteWidth(uint32_t Type) {
  // TODO: Replace this target-internal width mapping when
  // https://github.com/ROCm/llvm-project/issues/3601 exposes it through LLVM.
  switch (Type) {
  case ELF::R_AMDGPU_NONE:
    return 0;
  case ELF::R_AMDGPU_REL16:
    return 2;
  case ELF::R_AMDGPU_ABS32_LO:
  case ELF::R_AMDGPU_ABS32_HI:
  case ELF::R_AMDGPU_REL32:
  case ELF::R_AMDGPU_ABS32:
  case ELF::R_AMDGPU_GOTPCREL:
  case ELF::R_AMDGPU_GOTPCREL32_LO:
  case ELF::R_AMDGPU_GOTPCREL32_HI:
  case ELF::R_AMDGPU_REL32_LO:
  case ELF::R_AMDGPU_REL32_HI:
    return 4;
  case ELF::R_AMDGPU_ABS64:
  case ELF::R_AMDGPU_REL64:
  case ELF::R_AMDGPU_RELATIVE64:
    return 8;
  default:
    return makeEhFrameError(
        "dynamic relocation has unknown AMDGPU write width (type " +
        Twine(Type) + ")");
  }
}

Expected<bool> relocationWriteOverlaps(uint64_t Offset, uint64_t Width,
                                       uint64_t RangeBegin, uint64_t RangeEnd) {
  if (Width == 0)
    return false;
  if (Offset > std::numeric_limits<uint64_t>::max() - Width)
    return makeEhFrameError("dynamic relocation write range overflows");
  const uint64_t End = Offset + Width;
  return Offset < RangeEnd && End > RangeBegin;
}

Error rejectRelocationsTargetingEhFrame(const ElfView &Elf,
                                        uint32_t EhFrameIndex,
                                        uint64_t EhFrameBegin,
                                        uint64_t EhFrameEnd) {
  for (const ELFT::Shdr &Shdr : Elf.sections()) {
    const bool IsRel = Shdr.sh_type == ELF::SHT_REL;
    const bool IsRela = Shdr.sh_type == ELF::SHT_RELA;
    const bool IsOpaquePackedRelocation =
        Shdr.sh_type == ELF::SHT_RELR || Shdr.sh_type == ELF::SHT_ANDROID_REL ||
        Shdr.sh_type == ELF::SHT_ANDROID_RELA ||
        Shdr.sh_type == ELF::SHT_ANDROID_RELR || Shdr.sh_type == ELF::SHT_CREL;

    if ((IsRel || IsRela || IsOpaquePackedRelocation) &&
        Shdr.sh_info == EhFrameIndex) {
      return makeEhFrameError(".eh_frame relocation records are unsupported");
    }

    if (IsOpaquePackedRelocation) {
      if (Shdr.sh_flags & ELF::SHF_ALLOC) {
        return makeEhFrameError(
            "allocated packed relocation records cannot be checked for "
            ".eh_frame targets");
      }
      continue;
    }

    // Section-specific REL/RELA records were handled by sh_info above.
    // Dynamic relocation sections use sh_info == 0 and absolute virtual
    // addresses in r_offset, so enumerate them and reject writes that touch
    // any byte of .eh_frame.
    if ((!IsRel && !IsRela) || Shdr.sh_info != 0)
      continue;

    if (IsRela) {
      Expected<ELFT::RelaRange> RelasOrErr = Elf.file().relas(Shdr);
      if (!RelasOrErr) {
        return makeEhFrameError(
            "failed to read dynamic RELA records while checking .eh_frame: " +
            Twine(toString(RelasOrErr.takeError())));
      }
      for (const ELFT::Rela &Rela : *RelasOrErr) {
        Expected<uint64_t> WidthOrErr =
            amdgpuRelocationWriteWidth(Rela.getType(false));
        if (!WidthOrErr)
          return WidthOrErr.takeError();
        Expected<bool> OverlapsOrErr = relocationWriteOverlaps(
            Rela.r_offset, *WidthOrErr, EhFrameBegin, EhFrameEnd);
        if (!OverlapsOrErr)
          return OverlapsOrErr.takeError();
        if (*OverlapsOrErr) {
          return makeEhFrameError("dynamic RELA record writes into .eh_frame");
        }
      }
      continue;
    }

    Expected<ELFT::RelRange> RelsOrErr = Elf.file().rels(Shdr);
    if (!RelsOrErr) {
      return makeEhFrameError(
          "failed to read dynamic REL records while checking .eh_frame: " +
          Twine(toString(RelsOrErr.takeError())));
    }
    for (const ELFT::Rel &Rel : *RelsOrErr) {
      Expected<uint64_t> WidthOrErr =
          amdgpuRelocationWriteWidth(Rel.getType(false));
      if (!WidthOrErr)
        return WidthOrErr.takeError();
      Expected<bool> OverlapsOrErr = relocationWriteOverlaps(
          Rel.r_offset, *WidthOrErr, EhFrameBegin, EhFrameEnd);
      if (!OverlapsOrErr)
        return OverlapsOrErr.takeError();
      if (*OverlapsOrErr)
        return makeEhFrameError("dynamic REL record writes into .eh_frame");
    }
  }
  return Error::success();
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

template <typename T>
EhFramePatch makeLittleEndianPatch(uint64_t Offset, T Value) {
  EhFramePatch Patch;
  Patch.Offset = Offset;
  Patch.Bytes.resize(sizeof(T));
  for (unsigned I = 0; I != sizeof(T); ++I)
    Patch.Bytes[I] = static_cast<uint8_t>(
        static_cast<std::make_unsigned_t<T>>(Value) >> (I * 8));
  return Patch;
}

Expected<uint64_t> mapCfiLocation(const DisplacementPlan &Plan,
                                  uint64_t OldLocation, uint64_t OldFdeStart,
                                  uint64_t OldFdeEnd, uint64_t OldTextBegin,
                                  uint64_t OldTextEnd, uint64_t NewTextBegin) {
  if (OldLocation < OldFdeStart || OldLocation > OldFdeEnd)
    return makeEhFrameError(".eh_frame CFI location is outside its FDE range");
  if (OldLocation < OldTextBegin || OldLocation > OldTextEnd)
    return makeEhFrameError(".eh_frame CFI location is outside .text");

  uint64_t NewOffset = 0;
  if (!Plan.mapOffset(OldLocation - OldTextBegin,
                      DisplacementMapBias::AfterInsertedBytes, NewOffset)) {
    return makeEhFrameError(
        ".eh_frame CFI location maps inside a replaced instruction");
  }
  if (NewOffset > std::numeric_limits<uint64_t>::max() - NewTextBegin)
    return makeEhFrameError(".eh_frame CFI location overflows");
  return NewTextBegin + NewOffset;
}

Error consumeCfiUleb(DataExtractor &Data, DataExtractor::Cursor &Cursor) {
  (void)Data.getULEB128(Cursor);
  if (!Cursor) {
    consumeError(Cursor.takeError());
    return makeEhFrameError(".eh_frame has a truncated CFI ULEB128 operand");
  }
  return Error::success();
}

Error consumeCfiSleb(DataExtractor &Data, DataExtractor::Cursor &Cursor) {
  (void)Data.getSLEB128(Cursor);
  if (!Cursor) {
    consumeError(Cursor.takeError());
    return makeEhFrameError(".eh_frame has a truncated CFI SLEB128 operand");
  }
  return Error::success();
}

Error consumeCfiExpression(DataExtractor &Data, DataExtractor::Cursor &Cursor) {
  const uint64_t Length = Data.getULEB128(Cursor);
  if (!Cursor) {
    consumeError(Cursor.takeError());
    return makeEhFrameError(".eh_frame has a truncated CFI expression length");
  }
  (void)Data.getBytes(Cursor, Length);
  if (!Cursor) {
    consumeError(Cursor.takeError());
    return makeEhFrameError(".eh_frame has a truncated CFI expression");
  }
  return Error::success();
}

Expected<std::vector<EhFramePatch>> remapFdeCfiProgram(
    StringRef ProgramData, uint64_t ProgramSectionOffset,
    uint64_t CodeAlignment, const DisplacementPlan &Plan, uint64_t OldFdeStart,
    uint64_t OldFdeEnd, uint64_t NewFdeStart, uint64_t NewFdeEnd,
    uint64_t OldTextBegin, uint64_t OldTextEnd, uint64_t NewTextBegin) {
  if (CodeAlignment == 0)
    return makeEhFrameError(".eh_frame CFI has zero code alignment");

  DataExtractor Data(ProgramData, /*IsLittleEndian=*/true);
  DataExtractor::Cursor Cursor(0);
  uint64_t OldLocation = OldFdeStart;
  uint64_t NewLocation = NewFdeStart;
  std::vector<EhFramePatch> Patches;

  std::function<Error(uint64_t, uint64_t, unsigned, uint8_t)> RemapAdvance =
      [&](uint64_t Operand, uint64_t OperandOffset, unsigned OperandSize,
          uint8_t PrimaryOpcode) -> Error {
    if (Operand >
        (std::numeric_limits<uint64_t>::max() - OldLocation) / CodeAlignment) {
      return makeEhFrameError(".eh_frame CFI advance location overflows");
    }
    const uint64_t OldNext = OldLocation + Operand * CodeAlignment;
    Expected<uint64_t> NewNextOrErr =
        mapCfiLocation(Plan, OldNext, OldFdeStart, OldFdeEnd, OldTextBegin,
                       OldTextEnd, NewTextBegin);
    if (!NewNextOrErr)
      return NewNextOrErr.takeError();
    const uint64_t NewNext = *NewNextOrErr;
    if (NewNext < NewLocation)
      return makeEhFrameError(".eh_frame CFI location moves backwards");
    const uint64_t NewDelta = NewNext - NewLocation;
    if (NewDelta % CodeAlignment != 0) {
      return makeEhFrameError(
          ".eh_frame CFI advance is not divisible by its code alignment");
    }
    const uint64_t NewOperand = NewDelta / CodeAlignment;
    const uint64_t MaxOperand = OperandSize == 8
                                    ? std::numeric_limits<uint64_t>::max()
                                    : (uint64_t{1} << (OperandSize * 8)) - 1;
    if (NewOperand > MaxOperand)
      return makeEhFrameError(
          ".eh_frame CFI advance no longer fits its encoding");

    if (PrimaryOpcode != 0) {
      if (NewOperand > 0x3f)
        return makeEhFrameError(
            ".eh_frame CFI advance no longer fits DW_CFA_advance_loc");
      Patches.push_back(makeLittleEndianPatch<uint8_t>(
          ProgramSectionOffset + OperandOffset,
          static_cast<uint8_t>(PrimaryOpcode | NewOperand)));
    } else {
      EhFramePatch Patch;
      Patch.Offset = ProgramSectionOffset + OperandOffset;
      Patch.Bytes.resize(OperandSize);
      for (unsigned I = 0; I != OperandSize; ++I)
        Patch.Bytes[I] = static_cast<uint8_t>(NewOperand >> (I * 8));
      Patches.push_back(std::move(Patch));
    }
    OldLocation = OldNext;
    NewLocation = NewNext;
    return Error::success();
  };

  while (Cursor && Cursor.tell() < ProgramData.size()) {
    const uint64_t OpcodeOffset = Cursor.tell();
    const uint8_t EncodedOpcode = Data.getU8(Cursor);
    if (!Cursor)
      break;

    const uint8_t PrimaryOpcode = EncodedOpcode & 0xc0;
    if (PrimaryOpcode != 0) {
      if (PrimaryOpcode == dwarf::DW_CFA_advance_loc) {
        if (Error Err = RemapAdvance(EncodedOpcode & 0x3f, OpcodeOffset, 1,
                                     dwarf::DW_CFA_advance_loc))
          return std::move(Err);
      } else if (PrimaryOpcode == dwarf::DW_CFA_offset) {
        if (Error Err = consumeCfiUleb(Data, Cursor))
          return std::move(Err);
      } else if (PrimaryOpcode != dwarf::DW_CFA_restore) {
        return makeEhFrameError(".eh_frame has an invalid primary CFI opcode");
      }
      continue;
    }

    switch (EncodedOpcode) {
    case dwarf::DW_CFA_nop:
    case dwarf::DW_CFA_remember_state:
    case dwarf::DW_CFA_restore_state:
    case dwarf::DW_CFA_GNU_window_save:
    case dwarf::DW_CFA_AARCH64_negate_ra_state_with_pc:
      break;
    case dwarf::DW_CFA_set_loc: {
      const uint64_t OperandOffset = Cursor.tell();
      const uint64_t Target = Data.getUnsigned(Cursor, sizeof(uint64_t));
      if (!Cursor)
        break;
      Expected<uint64_t> NewTargetOrErr =
          mapCfiLocation(Plan, Target, OldFdeStart, OldFdeEnd, OldTextBegin,
                         OldTextEnd, NewTextBegin);
      if (!NewTargetOrErr)
        return NewTargetOrErr.takeError();
      Patches.push_back(makeLittleEndianPatch<uint64_t>(
          ProgramSectionOffset + OperandOffset, *NewTargetOrErr));
      OldLocation = Target;
      NewLocation = *NewTargetOrErr;
      break;
    }
    case dwarf::DW_CFA_advance_loc1:
    case dwarf::DW_CFA_advance_loc2:
    case dwarf::DW_CFA_advance_loc4: {
      const unsigned OperandSize =
          EncodedOpcode == dwarf::DW_CFA_advance_loc1   ? 1
          : EncodedOpcode == dwarf::DW_CFA_advance_loc2 ? 2
                                                        : 4;
      const uint64_t OperandOffset = Cursor.tell();
      const uint64_t Operand = Data.getUnsigned(Cursor, OperandSize);
      if (!Cursor)
        break;
      if (Error Err =
              RemapAdvance(Operand, OperandOffset, OperandSize, /*Primary=*/0))
        return std::move(Err);
      break;
    }
    case dwarf::DW_CFA_restore_extended:
    case dwarf::DW_CFA_undefined:
    case dwarf::DW_CFA_same_value:
    case dwarf::DW_CFA_def_cfa_register:
    case dwarf::DW_CFA_def_cfa_offset:
    case dwarf::DW_CFA_GNU_args_size:
      if (Error Err = consumeCfiUleb(Data, Cursor))
        return std::move(Err);
      break;
    case dwarf::DW_CFA_def_cfa_offset_sf:
      if (Error Err = consumeCfiSleb(Data, Cursor))
        return std::move(Err);
      break;
    case dwarf::DW_CFA_LLVM_def_aspace_cfa:
    case dwarf::DW_CFA_LLVM_def_aspace_cfa_sf:
      if (Error Err = consumeCfiUleb(Data, Cursor))
        return std::move(Err);
      if (EncodedOpcode == dwarf::DW_CFA_LLVM_def_aspace_cfa) {
        if (Error Err = consumeCfiUleb(Data, Cursor))
          return std::move(Err);
      } else if (Error Err = consumeCfiSleb(Data, Cursor)) {
        return std::move(Err);
      }
      if (Error Err = consumeCfiUleb(Data, Cursor))
        return std::move(Err);
      break;
    case dwarf::DW_CFA_offset_extended:
    case dwarf::DW_CFA_register:
    case dwarf::DW_CFA_def_cfa:
    case dwarf::DW_CFA_val_offset:
      if (Error Err = consumeCfiUleb(Data, Cursor))
        return std::move(Err);
      if (Error Err = consumeCfiUleb(Data, Cursor))
        return std::move(Err);
      break;
    case dwarf::DW_CFA_offset_extended_sf:
    case dwarf::DW_CFA_def_cfa_sf:
    case dwarf::DW_CFA_val_offset_sf:
      if (Error Err = consumeCfiUleb(Data, Cursor))
        return std::move(Err);
      if (Error Err = consumeCfiSleb(Data, Cursor))
        return std::move(Err);
      break;
    case dwarf::DW_CFA_def_cfa_expression:
      if (Error Err = consumeCfiExpression(Data, Cursor))
        return std::move(Err);
      break;
    case dwarf::DW_CFA_expression:
    case dwarf::DW_CFA_val_expression:
      if (Error Err = consumeCfiUleb(Data, Cursor))
        return std::move(Err);
      if (Error Err = consumeCfiExpression(Data, Cursor))
        return std::move(Err);
      break;
    default:
      return makeEhFrameError(".eh_frame has an unsupported CFI opcode");
    }
  }

  if (Error Err = Cursor.takeError()) {
    consumeError(std::move(Err));
    return makeEhFrameError(".eh_frame has a truncated CFI instruction");
  }
  if (OldLocation > OldFdeEnd || NewLocation > NewFdeEnd)
    return makeEhFrameError(".eh_frame CFI location exceeds its FDE range");
  return Patches;
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
  // be rewritten in place. CFI location instructions are also rewritten when
  // their remapped operand still fits the instruction's existing encoding, so
  // no record or section needs to change size.
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
  if (OldShdr->sh_size >
      std::numeric_limits<uint64_t>::max() - OldShdr->sh_addr)
    return makeEhFrameError(".eh_frame virtual address range overflows");
  if (Error Err = rejectRelocationsTargetingEhFrame(
          OldElf, OldEhFrameIndex, OldShdr->sh_addr,
          OldShdr->sh_addr + OldShdr->sh_size))
    return Err;

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
  if (OutElfOrErr->textSize() >
      std::numeric_limits<uint64_t>::max() - NewTextBegin) {
    return makeEhFrameError(
        "displaced .text range overflows while remapping .eh_frame");
  }
  const uint64_t NewTextEnd = NewTextBegin + OutElfOrErr->textSize();

  std::vector<EhFramePatch> Patches;
  unsigned RemappedFdeCount = 0;
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

    const bool DescribesText =
        OldRange == 0 ? OldStart >= OldTextBegin && OldStart < OldTextEnd
                      : OldEnd > OldTextBegin && OldStart < OldTextEnd;
    if (!DescribesText) {
      const bool OverlapsDisplacedText =
          OldRange == 0 ? OldStart >= NewTextBegin && OldStart < NewTextEnd
                        : OldEnd > NewTextBegin && OldStart < NewTextEnd;
      if (OverlapsDisplacedText) {
        return makeEhFrameError(
            ".eh_frame FDE outside old .text overlaps displaced .text");
      }
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
    if (NewRange != OldRange && advancesCfiLocation(Cie->cfis())) {
      return makeEhFrameError(
          ".eh_frame CIE has location-changing CFI that cannot be remapped "
          "per FDE");
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

    uint64_t CfiOffset = RangeOffset + sizeof(uint32_t);
    if (Cie->getAugmentationString().starts_with("z")) {
      StringRef AugmentationAndCfi = OldData.slice(CfiOffset, RecordEnd);
      DataExtractor AugmentationData(AugmentationAndCfi,
                                     /*IsLittleEndian=*/true);
      DataExtractor::Cursor Cursor(0);
      const uint64_t AugmentationLength = AugmentationData.getULEB128(Cursor);
      if (!Cursor) {
        consumeError(Cursor.takeError());
        return makeEhFrameError(
            ".eh_frame has a truncated FDE augmentation length");
      }
      (void)AugmentationData.getBytes(Cursor, AugmentationLength);
      if (!Cursor) {
        consumeError(Cursor.takeError());
        return makeEhFrameError(
            ".eh_frame has truncated FDE augmentation data");
      }
      CfiOffset += Cursor.tell();
      consumeError(Cursor.takeError());
    }
    if (CfiOffset > RecordEnd)
      return makeEhFrameError(".eh_frame FDE CFI offset is out of bounds");

    Expected<std::vector<EhFramePatch>> CfiPatchesOrErr = remapFdeCfiProgram(
        OldData.slice(CfiOffset, RecordEnd), CfiOffset,
        Cie->getCodeAlignmentFactor(), Plan, OldStart, OldEnd, NewStart,
        NewStart + NewRange, OldTextBegin, OldTextEnd, NewTextBegin);
    if (!CfiPatchesOrErr)
      return CfiPatchesOrErr.takeError();

    Patches.push_back(
        makeLittleEndianPatch<int32_t>(LocationOffset, *NewLocationOrErr));
    Patches.push_back(makeLittleEndianPatch<uint32_t>(
        RangeOffset, static_cast<uint32_t>(NewRange)));
    Patches.insert(Patches.end(),
                   std::make_move_iterator(CfiPatchesOrErr->begin()),
                   std::make_move_iterator(CfiPatchesOrErr->end()));
    ++RemappedFdeCount;
  }

  for (const EhFramePatch &Patch : Patches) {
    if (Patch.Offset > OutShdr->sh_size ||
        Patch.Bytes.size() > OutShdr->sh_size - Patch.Offset) {
      return makeEhFrameError(".eh_frame patch is outside the section");
    }
    std::memcpy(OutData + OutShdr->sh_offset + Patch.Offset, Patch.Bytes.data(),
                Patch.Bytes.size());
  }
  log() << "hotswap: displacement: remapped " << RemappedFdeCount
        << " .eh_frame FDE(s)\n";
  return Error::success();
}

} // namespace hotswap
} // namespace COMGR
