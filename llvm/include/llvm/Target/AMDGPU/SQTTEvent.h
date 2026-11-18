//===- SQTTEvent.h - AMDGPU SQTT event metadata -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// SQTT event metadata shared by the AMDGPU backend.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_AMDGPU_SQTTEVENT_H
#define LLVM_TARGET_AMDGPU_SQTTEVENT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/AMDGPUSQTTEventType.h"
#include "llvm/Support/Error.h"
#include <cassert>
#include <cstdint>
#include <vector>

namespace llvm {
class Function;
class LLVMContext;
class Metadata;
namespace object {
class ObjectFile;
} // namespace object

namespace sqtt {

const char EventsTableMetadata[] = "llvm.sqtt.events";
const char EventsTableSection[] = "__sqtt_events";
const char DataTableSectionPrefix[] = "__sqtt_data_v";
const char DataStringsSection[] = "__sqtt_data_str";
const char EventsTableCUStart[] = "sqtt_events_cu_base";

constexpr uint32_t SectionBase = 0xffffffff;

constexpr uint32_t FormatVersion = 1;

class BinarySQTTRecord;

class Event {
  EventType Type;
  union {
    StringRef StringPayload;
    uint32_t IntPayload;
  };

public:
  Event(EventType Type, StringRef Payload)
      : Type(Type), StringPayload(Payload) {
    assert(hasStringPayload(Type));
  }
  Event(EventType Type, uint32_t Payload) : Type(Type), IntPayload(Payload) {
    assert(!hasStringPayload(Type));
  }

  static Event functionEntry(const Function &F);
  static Event functionExit(const Function &F);
  static Event userEntry(StringRef Payload);
  static Event userExit(StringRef Payload);
  static Event setEventTableBase(uint32_t Index);

  EventType getType() const { return Type; }
  bool isSetEventTableBase() const {
    return Type == EventType::SetEventTableBase;
  }
  StringRef getPayload() const { return StringPayload; }
  uint32_t getIndex() const { return IntPayload; }

  Metadata *toMetadata(LLVMContext &Ctx) const;
  static Event fromMetadata(Metadata *MD);
  LLVM_ABI static bool hasStringPayload(EventType Type);
};

class EventRecord {
  SmallVector<Event, 4> Events;

public:
  EventRecord(ArrayRef<Event> Events) : Events(Events) {}
  ArrayRef<Event> events() const { return Events; }
  size_t size() const { return Events.size(); }

  Metadata *toMetadata(LLVMContext &Ctx) const;
  static EventRecord fromMetadata(Metadata *MD);
};

class BinarySQTTRecord : public EventRecord {
  uint64_t TTraceDataAddress;

public:
  BinarySQTTRecord(ArrayRef<Event> Events, uint64_t TTraceDataAddress)
      : EventRecord(Events), TTraceDataAddress(TTraceDataAddress) {}
  uint64_t getTTraceDataAddress() const { return TTraceDataAddress; }
};

LLVM_ABI Expected<std::vector<BinarySQTTRecord>>
parseBinarySQTTRecords(const object::ObjectFile &O);

} // namespace sqtt
} // namespace llvm

#endif
