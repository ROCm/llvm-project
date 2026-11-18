#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/RelocationResolver.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/AMDGPU/SQTTEvent.h"
#include "llvm/TargetParser/Triple.h"
#include <limits>
#include <optional>

using namespace llvm;
using namespace llvm::sqtt;

Event Event::functionEntry(const Function &F) {
  return Event(EventType::FunctionEntry, F.getName());
}

Event Event::functionExit(const Function &F) {
  return Event(EventType::FunctionExit, F.getName());
}

Event Event::userEntry(StringRef Payload) {
  return Event(EventType::UserEntry, Payload);
}

Event Event::userExit(StringRef Payload) {
  return Event(EventType::UserExit, Payload);
}

Event Event::setEventTableBase(uint32_t Index) {
  return Event(EventType::SetEventTableBase, Index);
}

Metadata *Event::toMetadata(LLVMContext &Ctx) const {
  auto *IntTy = Type::getInt32Ty(Ctx);
  Metadata *TypeMD = ConstantAsMetadata::get(
      ConstantInt::get(IntTy, static_cast<uint32_t>(Type)));
  Metadata *PayloadMD;
  if (isSetEventTableBase())
    PayloadMD = ConstantAsMetadata::get(ConstantInt::get(IntTy, getIndex()));
  else
    PayloadMD = MDString::get(Ctx, getPayload());
  return MDNode::get(Ctx, {TypeMD, PayloadMD});
}

Event Event::fromMetadata(Metadata *MD) {
  auto *Node = cast<MDNode>(MD);

  Metadata *TypeMD = Node->getOperand(0);
  ConstantAsMetadata *TypeCA = cast<ConstantAsMetadata>(TypeMD);
  ConstantInt *TypeCI = cast<ConstantInt>(TypeCA->getValue());
  EventType Type = static_cast<EventType>(TypeCI->getZExtValue());

  if (Type == EventType::SetEventTableBase) {
    auto *Index = mdconst::extract<ConstantInt>(Node->getOperand(1));
    return setEventTableBase(static_cast<uint32_t>(Index->getZExtValue()));
  }

  StringRef Payload = cast<MDString>(Node->getOperand(1))->getString();
  return {Type, Payload};
}

bool Event::hasStringPayload(EventType Type) {
  switch (Type) {
  case EventType::FunctionEntry:
  case EventType::FunctionExit:
  case EventType::UserEntry:
  case EventType::UserExit:
    return true;
  default:
    return false;
  }
}

EventRecord EventRecord::fromMetadata(Metadata *MD) {
  auto *Node = cast<MDNode>(MD);
  Metadata *Op0 = Node->getOperand(0);
  if (isa<ConstantAsMetadata>(Op0)) {
    return {{Event::fromMetadata(MD)}};
  }

  SmallVector<Event, 4> Events;
  for (unsigned I = 0; I != Node->getNumOperands(); ++I) {
    Events.push_back(Event::fromMetadata(Node->getOperand(I)));
  }
  return {Events};
}

Metadata *EventRecord::toMetadata(LLVMContext &Ctx) const {
  SmallVector<Metadata *, 4> Ops;
  for (const Event &Ev : Events)
    Ops.push_back(Ev.toMetadata(Ctx));

  if (Ops.size() == 1)
    return Ops.front();

  return MDNode::get(Ctx, Ops);
}

static std::optional<EventType> isValidEventType(uint16_t Type) {
  if (Type >= to_underlying(EventType::Begin) &&
      Type < to_underlying(EventType::End))
    return static_cast<EventType>(Type);
  return std::nullopt;
}

// Relocatable objects store pointer fields as R_AMDGPU_ABS64 addends. Those
// relocations live in a separate section. A linked image already has the
// values in the section bytes.
static Expected<DenseMap<uint64_t, uint64_t>>
relocatedValues(const object::SectionRef &Target) {
  const object::ObjectFile *O = Target.getObject();
  DenseMap<uint64_t, uint64_t> Values;
  if (!O->isRelocatableObject())
    return Values;

  auto [Supports, Resolver] = object::getRelocationResolver(*O);
  if (!Supports)
    return Values;

  for (const object::SectionRef &Sec : O->sections()) {
    Expected<object::section_iterator> RelocatedOrErr =
        Sec.getRelocatedSection();
    if (!RelocatedOrErr) {
      consumeError(RelocatedOrErr.takeError());
      continue;
    }
    if (*RelocatedOrErr == O->section_end() || **RelocatedOrErr != Target)
      continue;

    for (const object::RelocationRef &Reloc : Sec.relocations()) {
      if (!Supports(Reloc.getType())) {
        return make_error<StringError>("unsupported relocation type " +
                                           Twine(Reloc.getType()),
                                       inconvertibleErrorCode());
        continue;
      }
      object::symbol_iterator Sym = Reloc.getSymbol();
      if (Sym == O->symbol_end())
        continue;
      auto SymValue = Sym->getValue();
      if (!SymValue)
        return SymValue.takeError();
      Values[Reloc.getOffset()] =
          object::resolveRelocation(Resolver, Reloc, *SymValue, /*LocData=*/0);
    }
  }
  return Values;
}

static Error makeError(const Twine &Message) {
  return make_error<StringError>(Message, inconvertibleErrorCode());
}

class SectionInfo {
  uint64_t BaseAddress;
  StringRef Contents;
  DenseMap<uint64_t, uint64_t> Relocs;

  SectionInfo(uint64_t BaseAddress, StringRef Contents,
              DenseMap<uint64_t, uint64_t> Relocs)
      : BaseAddress(BaseAddress), Contents(Contents),
        Relocs(std::move(Relocs)) {}

public:
  static Expected<SectionInfo> get(const object::SectionRef &Section,
                                   StringRef Name) {
    if (Section.isVirtual())
      return makeError("SQTT section " + Name + " has no contents");

    auto Contents = Section.getContents();
    if (!Contents)
      return Contents.takeError();

    auto Relocs = relocatedValues(Section);
    if (!Relocs)
      return Relocs.takeError();

    return SectionInfo(Section.getAddress(), *Contents, std::move(*Relocs));
  }

  uint64_t begin_address() const { return BaseAddress; }

  uint64_t end_address() const { return BaseAddress + Contents.size(); }

  uint64_t size() const { return Contents.size(); }

  // The contents from Address to the end of the section.
  StringRef contentsFrom(uint64_t Address) const {
    return Contents.substr(Address - BaseAddress);
  }

  bool contains(uint64_t Address, uint64_t Size = 1) const {
    if (Address < BaseAddress)
      return false;
    uint64_t Offset = Address - BaseAddress;
    return Offset <= Contents.size() && Contents.size() - Offset >= Size;
  }

  template <typename T> T readValue(uint64_t Address) const {
    return support::endian::read<T, llvm::endianness::little>(
        Contents.data() + (Address - BaseAddress));
  }

  uint64_t readPointer(uint64_t Address) const {
    // We assume that relocations are only used for pointers (labels to data or
    // string sections), and not for regular data. This is not in general true,
    // but it holds in our context.
    auto It = Relocs.find(Address - BaseAddress);
    if (It != Relocs.end())
      return It->second;
    return readValue<uint64_t>(Address);
  }
};

Expected<std::vector<BinarySQTTRecord>>
sqtt::parseBinarySQTTRecords(const object::ObjectFile &O) {
  if (!O.isELF())
    return makeError("SQTT is currently only supported for ELF targets");

  if (O.getArch() != Triple::amdgpu)
    return makeError("SQTT is only supported for AMDGPU targets");

  std::optional<object::SectionRef> Events, DataStr;
  // The data section is named after the format version of its records.
  SmallVector<object::SectionRef, 1> Data;
  for (const object::SectionRef &Section : O.sections()) {
    auto SectionName = Section.getName();
    if (!SectionName)
      return SectionName.takeError();

    if (*SectionName == sqtt::EventsTableSection)
      Events = Section;
    else if (*SectionName == sqtt::DataStringsSection)
      DataStr = Section;
    else if (SectionName->starts_with(sqtt::DataTableSectionPrefix))
      Data.push_back(Section);
  }

  // Linking compilation units built with different format versions leaves one
  // data section per version.
  if (Data.size() > 1)
    return makeError("multiple SQTT sections " +
                     Twine(sqtt::DataTableSectionPrefix));

  if (Data.empty())
    return makeError("SQTT section " + Twine(sqtt::DataTableSectionPrefix) +
                     " not found");

  if (!DataStr)
    return makeError("SQTT section " + Twine(sqtt::DataStringsSection) +
                     " not found");

  if (!Events)
    return makeError("SQTT section " + Twine(sqtt::EventsTableSection) +
                     " not found");

  auto DataSectionName = Data.front().getName();
  if (!DataSectionName)
    return DataSectionName.takeError();

  StringRef VersionStr = *DataSectionName;
  VersionStr.consume_front(sqtt::DataTableSectionPrefix);
  uint32_t Version;
  if (VersionStr.getAsInteger(10, Version) || Version != sqtt::FormatVersion)
    return makeError("SQTT section " + *DataSectionName +
                     " has an unsupported format version");

  Expected<SectionInfo> EventsInfo =
      SectionInfo::get(*Events, sqtt::EventsTableSection);
  if (!EventsInfo)
    return EventsInfo.takeError();
  Expected<SectionInfo> DataInfo =
      SectionInfo::get(Data.front(), *DataSectionName);
  if (!DataInfo)
    return DataInfo.takeError();
  Expected<SectionInfo> DataStringsInfo =
      SectionInfo::get(*DataStr, sqtt::DataStringsSection);
  if (!DataStringsInfo)
    return DataStringsInfo.takeError();

  if (EventsInfo->size() % 8 != 0)
    return makeError("SQTT section " + Twine(sqtt::EventsTableSection) +
                     " size is not a multiple of 8");

  // Record IDs and table base indices are 32-bit.
  if (EventsInfo->size() / 8 > std::numeric_limits<uint32_t>::max())
    return makeError("SQTT section " + Twine(sqtt::EventsTableSection) +
                     " has more than 2^32 - 1 entries");

  // Each record is {u64 pc, u16 count, [u16 type, u64 payload]*count}.
  constexpr uint64_t RecordHeaderSize = 10;
  constexpr uint64_t EventSize = 10;

  std::vector<BinarySQTTRecord> Records;
  for (uint64_t Address = EventsInfo->begin_address();
       Address < EventsInfo->end_address(); Address += 8) {
    uint64_t Id = (Address - EventsInfo->begin_address()) / 8;
    uint64_t EventDataPtr = EventsInfo->readPointer(Address);
    if (!DataInfo->contains(EventDataPtr, RecordHeaderSize))
      return makeError("SQTT record ID " + Twine(Id) +
                       " points outside the data section");

    uint64_t PC = DataInfo->readPointer(EventDataPtr);
    uint16_t EventCount = DataInfo->readValue<uint16_t>(EventDataPtr + 8);
    if (!DataInfo->contains(EventDataPtr,
                            RecordHeaderSize + EventCount * EventSize))
      return makeError("SQTT record ID " + Twine(Id) +
                       " has a truncated record");

    std::vector<Event> Events;
    for (uint16_t I = 0; I < EventCount; ++I) {
      uint64_t EventDataAddress =
          EventDataPtr + RecordHeaderSize + I * EventSize;
      uint16_t EventType = DataInfo->readValue<uint16_t>(EventDataAddress);
      std::optional<sqtt::EventType> EventTypeOpt = isValidEventType(EventType);
      if (!EventTypeOpt)
        return makeError("Invalid SQTT event type " + Twine(EventType));
      uint64_t Payload = DataInfo->readPointer(EventDataAddress + 2);
      if (sqtt::Event::hasStringPayload(*EventTypeOpt)) {
        if (!DataStringsInfo->contains(Payload))
          return makeError("SQTT string payload " + Twine(Payload) +
                           " is outside __sqtt_data_str");
        StringRef PayloadStr = DataStringsInfo->contentsFrom(Payload);
        size_t End = PayloadStr.find('\0');
        if (End == StringRef::npos)
          return makeError("SQTT string payload " + Twine(Payload) +
                           " is not null-terminated");
        Events.emplace_back(*EventTypeOpt, PayloadStr.take_front(End));
      } else {
        assert(*EventTypeOpt == sqtt::EventType::SetEventTableBase);
        if (!EventsInfo->contains(Payload) ||
            (Payload - EventsInfo->begin_address()) % 8 != 0)
          return makeError("SQTT event table base " + Twine(Payload) +
                           " is not an entry of " +
                           Twine(sqtt::EventsTableSection));
        uint32_t Index = (Payload - EventsInfo->begin_address()) / 8;
        Events.emplace_back(*EventTypeOpt, Index);
      }
    }
    Records.emplace_back(Events, PC);
  }
  return Records;
}
