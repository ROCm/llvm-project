//===- SQTTConfig.cpp - SQTT marker configuration -------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Parses SQTT marker pass options and their environment fallbacks.
///
//===----------------------------------------------------------------------===//

#include "SQTTConfig.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>

using namespace llvm;

namespace {

cl::OptionCategory SQTTMarkerCategory("AMD SQTT marker options");

#define SQTT_STRING_OPTION(Variable, Name, Description, Value)                 \
  cl::opt<std::string> Variable(Name, cl::desc(Description),                   \
                                cl::value_desc(Value), cl::init(""),           \
                                cl::cat(SQTTMarkerCategory))

SQTT_STRING_OPTION(InstrumentBarriersOpt, "sqtt-marker-instrument-barriers",
                   "Instrument AMDGPU barriers (fallback: "
                   "SQTT_INSTRUMENT_BARRIERS)",
                   "0|1");
SQTT_STRING_OPTION(MemBarrierOpt, "sqtt-marker-mem-barrier",
                   "Select marker reorder boundaries (fallback: "
                   "SQTT_MEM_BARRIER)",
                   "none|asm|fence");
SQTT_STRING_OPTION(ScopeWaveOpt, "sqtt-marker-scope-wave",
                   "Select waves that emit markers (fallback: "
                   "SQTT_SCOPE_WAVE)",
                   "mask");
SQTT_STRING_OPTION(ScopeSimdOpt, "sqtt-marker-scope-simd",
                   "Select SIMDs that emit markers (fallback: "
                   "SQTT_SCOPE_SIMD)",
                   "mask");
SQTT_STRING_OPTION(ScopeCuOpt, "sqtt-marker-scope-cu",
                   "Select CUs that emit markers (fallback: SQTT_SCOPE_CU)",
                   "mask");
SQTT_STRING_OPTION(ScopeWgOpt, "sqtt-marker-scope-wg",
                   "Select workgroups that emit markers (fallback: "
                   "SQTT_SCOPE_WG)",
                   "mask");
SQTT_STRING_OPTION(ShaderClockBitsOpt, "sqtt-marker-shader-clock-bits",
                   "Set gfx12 marker clock bits (fallback: "
                   "SQTT_SHADER_CLOCK_BITS)",
                   "bits");
SQTT_STRING_OPTION(ShaderClockShiftOpt, "sqtt-marker-shader-clock-shift",
                   "Set gfx12 marker clock shift (fallback: "
                   "SQTT_SHADER_CLOCK_SHIFT)",
                   "bits");
SQTT_STRING_OPTION(InstrumentFunctionsOpt, "sqtt-marker-instrument-functions",
                   "Set the function instrumentation threshold (fallback: "
                   "SQTT_INSTRUMENT_FUNCTIONS)",
                   "N|cost:N");
SQTT_STRING_OPTION(InstrumentMemoryOpt, "sqtt-marker-instrument-memory",
                   "Configure memory operation markers (fallback: "
                   "SQTT_INSTRUMENT_MEMORY)",
                   "N:M");
SQTT_STRING_OPTION(TraceAddressesOpt, "sqtt-marker-trace-addresses",
                   "Select address trace categories (fallback: "
                   "SQTT_TRACE_ADDRESSES)",
                   "memory|lds|memory,lds");

#undef SQTT_STRING_OPTION

struct SelectedValue {
  StringRef Value;
  StringRef Name;
};

using ConfigOption = cl::opt<std::string>;
using ValueSelector = function_ref<std::optional<SelectedValue>(
    const ConfigOption &, StringRef, const char *)>;

} // namespace

static std::optional<SelectedValue> environmentValue(const char *Name) {
  const char *Value = std::getenv(Name);
  if (!Value || Value[0] == '\0')
    return std::nullopt;
  return SelectedValue{Value, Name};
}

static uint32_t parseMask(SelectedValue Input, uint32_t Default) {
  if (Input.Value == "-1")
    return 0xFFFFFFFF;
  std::string Text = Input.Value.str();
  char *End = nullptr;
  unsigned long Value = std::strtoul(Text.c_str(), &End, 0);
  if (End == Text.c_str() || *End != '\0') {
    errs() << "sqtt: warning: invalid value for " << Input.Name << "='"
           << Input.Value << "', using default\n";
    return Default;
  }
  return static_cast<uint32_t>(Value);
}

static bool parseBool(SelectedValue Input) {
  StringRef Value = Input.Value;
  return Value.equals_insensitive("1") || Value.equals_insensitive("y") ||
         Value.equals_insensitive("yes") || Value.equals_insensitive("true") ||
         Value.equals_insensitive("on");
}

static MemBarrierMode parseMemBarrier(SelectedValue Input,
                                      MemBarrierMode Default) {
  StringRef Value = Input.Value;
  if (Value == "0" || Value.equals_insensitive("none") ||
      Value.equals_insensitive("off"))
    return MemBarrierMode::None;
  if (Value == "1" || Value.equals_insensitive("asm") ||
      Value.equals_insensitive("compiler") ||
      Value.equals_insensitive("clobber"))
    return MemBarrierMode::AsmClobber;
  if (Value == "2" || Value.equals_insensitive("fence") ||
      Value.equals_insensitive("on") || Value.equals_insensitive("hw"))
    return MemBarrierMode::Fence;
  errs() << "sqtt: warning: invalid value for " << Input.Name << "='"
         << Input.Value << "', expected one of "
         << "{none|asm|fence|0|1|2}, using default\n";
  return Default;
}

static unsigned parseUnsigned(SelectedValue Input, unsigned Default) {
  unsigned Result = 0;
  if (Input.Value.getAsInteger(10, Result)) {
    errs() << "sqtt: warning: invalid value for " << Input.Name << "='"
           << Input.Value << "', using default\n";
    return Default;
  }
  return Result;
}

static SQTTConfig parseConfig(ValueSelector Select) {
  SQTTConfig Config;
  auto Get = [&](const ConfigOption &Option, StringRef OptionName,
                 const char *EnvironmentName) {
    return Select(Option, OptionName, EnvironmentName);
  };

  if (auto Value = Get(InstrumentBarriersOpt, "sqtt-marker-instrument-barriers",
                       "SQTT_INSTRUMENT_BARRIERS"))
    Config.InstrumentBarriers = parseBool(*Value);
  if (auto Value =
          Get(MemBarrierOpt, "sqtt-marker-mem-barrier", "SQTT_MEM_BARRIER"))
    Config.MemBarrier = parseMemBarrier(*Value, MemBarrierMode::Fence);
  if (auto Value =
          Get(ScopeWaveOpt, "sqtt-marker-scope-wave", "SQTT_SCOPE_WAVE"))
    Config.WaveMask = parseMask(*Value, 0xFFFFFFFF);
  if (auto Value =
          Get(ScopeSimdOpt, "sqtt-marker-scope-simd", "SQTT_SCOPE_SIMD"))
    Config.SimdMask = parseMask(*Value, 0xF);
  if (auto Value = Get(ScopeCuOpt, "sqtt-marker-scope-cu", "SQTT_SCOPE_CU"))
    Config.CuMask = parseMask(*Value, 0x3);
  if (auto Value = Get(ScopeWgOpt, "sqtt-marker-scope-wg", "SQTT_SCOPE_WG"))
    Config.WgMask = parseMask(*Value, 0xFFFFFFFF);
  if (auto Value = Get(ShaderClockBitsOpt, "sqtt-marker-shader-clock-bits",
                       "SQTT_SHADER_CLOCK_BITS"))
    Config.ShaderClockBits = parseUnsigned(*Value, 0);
  if (auto Value = Get(ShaderClockShiftOpt, "sqtt-marker-shader-clock-shift",
                       "SQTT_SHADER_CLOCK_SHIFT"))
    Config.ShaderClockShift = parseUnsigned(*Value, 4);

  if (auto Value =
          Get(InstrumentFunctionsOpt, "sqtt-marker-instrument-functions",
              "SQTT_INSTRUMENT_FUNCTIONS")) {
    StringRef Text = Value->Value;
    if (Text.consume_front("cost:"))
      Config.Mode = CostMode::WeightedCost;
    Text.getAsInteger(10, Config.FunctionThreshold);
  }

  if (auto Value = Get(InstrumentMemoryOpt, "sqtt-marker-instrument-memory",
                       "SQTT_INSTRUMENT_MEMORY")) {
    auto [ChunkText, GapText] = Value->Value.split(':');
    unsigned Chunk = 0, Gap = 0;
    if (!ChunkText.getAsInteger(10, Chunk) && !GapText.empty() &&
        !GapText.getAsInteger(10, Gap) && Chunk > 0) {
      Config.MemoryChunkSize = Chunk;
      Config.MemoryMaxGap = Gap;
    } else {
      errs() << "sqtt: warning: invalid value for " << Value->Name << "='"
             << Value->Value << "', expected N:M\n";
    }
  }

  if (auto Value = Get(TraceAddressesOpt, "sqtt-marker-trace-addresses",
                       "SQTT_TRACE_ADDRESSES")) {
    SmallVector<StringRef, 2> Parts;
    Value->Value.split(Parts, ',');
    for (StringRef Part : Parts) {
      StringRef Category = Part.trim();
      if (Category == "memory")
        Config.TraceMemoryAddrs = true;
      else if (Category == "lds")
        Config.TraceLDSAddrs = true;
      else
        errs() << "sqtt: warning: unknown " << Value->Name << " category '"
               << Category << "'\n";
    }
  }

  if (Config.hasAddressTracing() && Config.MemoryChunkSize) {
    errs() << "sqtt: error: SQTT_TRACE_ADDRESSES and SQTT_INSTRUMENT_MEMORY "
              "are mutually exclusive\n";
    Config.TraceMemoryAddrs = Config.TraceLDSAddrs = false;
  }
  return Config;
}

SQTTConfig SQTTConfig::fromEnvironment() {
  return parseConfig([](const ConfigOption &, StringRef, const char *EnvName) {
    return environmentValue(EnvName);
  });
}

SQTTConfig SQTTConfig::fromCommandLine() {
  return parseConfig([](const ConfigOption &Option, StringRef OptionName,
                        const char *EnvName) {
    if (Option.getNumOccurrences() != 0)
      return std::optional<SelectedValue>{{Option.getValue(), OptionName}};
    return environmentValue(EnvName);
  });
}
