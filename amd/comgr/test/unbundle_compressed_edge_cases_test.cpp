#include "amd_comgr.h"
#include "common.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/OffloadBundle.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fstream>
#endif

using namespace llvm;

static constexpr char Magic[] = "__CLANG_OFFLOAD_BUNDLE__";
static constexpr char Host[] = "host-x86_64-unknown-linux-gnu";
static constexpr char Device[] = "hip-amdgcn-amd-amdhsa-unknown-gfx900";
static constexpr char Specific[] =
    "hip-amdgcn-amd-amdhsa-unknown-gfx900:xnack+";

#if defined(__linux__)
static bool isInCacheMapping(const char *Pointer, size_t Size) {
  std::ifstream Maps("/proc/self/maps");
  std::string Line;
  uintptr_t Begin = reinterpret_cast<uintptr_t>(Pointer);
  uintptr_t End = Begin + Size;
  while (std::getline(Maps, Line)) {
    size_t Dash = Line.find('-');
    size_t Space = Line.find(' ', Dash);
    if (Dash == std::string::npos || Space == std::string::npos ||
        Line.find("llvmcache-") == std::string::npos)
      continue;

    uintptr_t MappingBegin;
    uintptr_t MappingEnd;
    if (StringRef(Line.data(), Dash).getAsInteger(16, MappingBegin) ||
        StringRef(Line.data() + Dash + 1, Space - Dash - 1)
            .getAsInteger(16, MappingEnd))
      continue;
    if (Begin >= MappingBegin && End <= MappingEnd)
      return true;
  }
  return false;
}
#endif

static void appendInteger(std::string &Buffer, uint64_t Value) {
  size_t Offset = Buffer.size();
  Buffer.resize(Offset + sizeof(Value));
  support::endian::write64le(Buffer.data() + Offset, Value);
}

static std::string bundle(ArrayRef<std::pair<StringRef, StringRef>> Entries) {
  uint64_t Offset = sizeof(Magic) - 1 + sizeof(uint64_t);
  for (const auto &Entry : Entries)
    Offset += 3 * sizeof(uint64_t) + Entry.first.size();
  std::string Result = Magic;
  appendInteger(Result, Entries.size());
  for (const auto &Entry : Entries) {
    appendInteger(Result, Offset);
    appendInteger(Result, Entry.second.size());
    appendInteger(Result, Entry.first.size());
    Result += Entry.first;
    Offset += Entry.second.size();
  }
  for (const auto &Entry : Entries)
    Result += Entry.second;
  return Result;
}

static std::string compress(StringRef Input, compression::Format Format,
                            uint16_t Version) {
  auto Buffer = MemoryBuffer::getMemBuffer(Input, "", false);
  auto Compressed = object::CompressedOffloadBundle::compress(
      compression::Params(Format), *Buffer, Version == 1 ? 2 : Version);
  if (!Compressed)
    fail("compress: %s", toString(Compressed.takeError()).c_str());
  std::string Result = (*Compressed)->getBuffer().str();
  if (Version == 1) {
    Result.erase(8, sizeof(uint32_t));
    support::endian::write16le(Result.data() + 4, 1);
  }
  return Result;
}

static std::vector<std::string> unbundle(StringRef Input,
                                         std::vector<const char *> Targets,
                                         bool ExpectSuccess = true,
                                         bool CheckExecutable = false,
                                         bool VerifyCacheMapping = false,
                                         bool ExpectCacheMapping = false) {
  amd_comgr_data_set_t Inputs, Outputs;
  amd_comgr_data_t InputData;
  amd_comgr_action_info_t Action;
  checkError(amd_comgr_create_data_set(&Inputs), "create inputs");
  checkError(amd_comgr_create_data_set(&Outputs), "create outputs");
  auto InputKind = CheckExecutable ? AMD_COMGR_DATA_KIND_OBJ_BUNDLE
                                   : AMD_COMGR_DATA_KIND_BC_BUNDLE;
  auto OutputKind =
      CheckExecutable ? AMD_COMGR_DATA_KIND_EXECUTABLE : AMD_COMGR_DATA_KIND_BC;
  checkError(amd_comgr_create_data(InputKind, &InputData), "create input");
  checkError(amd_comgr_set_data(InputData, Input.size(), Input.data()),
             "set input");
  checkError(amd_comgr_set_data_name(InputData, "bundle"), "set name");
  checkError(amd_comgr_data_set_add(Inputs, InputData), "add input");
  checkError(amd_comgr_release_data(InputData), "release input");
  checkError(amd_comgr_create_action_info(&Action), "create action");
  checkError(amd_comgr_action_info_set_language(Action, AMD_COMGR_LANGUAGE_HIP),
             "set language");
  checkError(amd_comgr_action_info_set_bundle_entry_ids(Action, Targets.data(),
                                                        Targets.size()),
             "set targets");
  amd_comgr_status_t Status =
      amd_comgr_do_action(AMD_COMGR_ACTION_UNBUNDLE, Action, Inputs, Outputs);
  if (ExpectSuccess)
    checkError(Status, "unbundle");
  else if (Status == AMD_COMGR_STATUS_SUCCESS)
    fail("malformed bundle was accepted");
  checkError(amd_comgr_destroy_action_info(Action), "destroy action");
  checkError(amd_comgr_destroy_data_set(Inputs), "destroy inputs");

  std::vector<amd_comgr_data_t> Data;
  if (ExpectSuccess) {
    checkCount("outputs", Outputs, OutputKind, Targets.size());
    Data.resize(Targets.size());
    for (size_t Index = 0; Index < Data.size(); ++Index)
      checkError(amd_comgr_action_data_get_data(Outputs, OutputKind, Index,
                                                &Data[Index]),
                 "get output");
  }
  checkError(amd_comgr_destroy_data_set(Outputs), "destroy outputs");

  std::vector<std::string> Result(Data.size());
  for (size_t Count = Data.size(); Count; --Count) {
    size_t Index = Count - 1;
    size_t Size = 0;
    checkError(amd_comgr_get_data(Data[Index], &Size, nullptr), "get size");
    Result[Index].resize(Size);
    checkError(amd_comgr_get_data(Data[Index], &Size, Result[Index].data()),
               "get bytes");
    const char *Pointer = nullptr;
    size_t PointerSize = 0;
    checkError(amd_comgr_get_data_pointer(Data[Index], &PointerSize, &Pointer),
               "borrow output");
    if (PointerSize != Size || StringRef(Pointer, PointerSize) != Result[Index])
      fail("borrowed output differs from copied output");
    bool CheckMember = CheckExecutable && StringRef(Targets[Index]) != Host;
    if (VerifyCacheMapping && CheckMember) {
      bool IsAligned = (reinterpret_cast<uintptr_t>(Pointer) & 1) == 0;
      if (!IsAligned)
        fail("cached ELF is not two-byte aligned");
#if defined(__linux__)
      bool InCacheMapping = isInCacheMapping(Pointer, PointerSize);
      if (ExpectCacheMapping && !InCacheMapping)
        fail("aligned cached ELF does not share the cache mapping");
      if (!ExpectCacheMapping && InCacheMapping)
        fail("misaligned cached ELF was not copied");
#endif
    }
    amd_comgr_metadata_node_t Metadata = {};
    if (CheckMember) {
      size_t IsaSize = 0;
      checkError(amd_comgr_get_data_isa_name(Data[Index], &IsaSize, nullptr),
                 "get output ISA size");
      std::string Isa(IsaSize, '\0');
      checkError(amd_comgr_get_data_isa_name(Data[Index], &IsaSize, Isa.data()),
                 "get output ISA");
      if (Isa.find("amdgcn-amd-amdhsa--gfx900") != 0)
        fail("unexpected output ISA");
      checkError(amd_comgr_get_data_metadata(Data[Index], &Metadata),
                 "get output metadata");
      size_t SymbolCount = 0;
      checkError(amd_comgr_iterate_symbols(
                     Data[Index],
                     [](amd_comgr_symbol_t Symbol, void *UserData) {
                       amd_comgr_symbol_type_t Type;
                       auto Status = amd_comgr_symbol_get_info(
                           Symbol, AMD_COMGR_SYMBOL_INFO_TYPE, &Type);
                       if (Status == AMD_COMGR_STATUS_SUCCESS)
                         ++*static_cast<size_t *>(UserData);
                       return Status;
                     },
                     &SymbolCount),
                 "iterate output symbols");
      if (!SymbolCount)
        fail("no output symbols");
    }
    checkError(amd_comgr_release_data(Data[Index]), "release output");
    if (CheckMember) {
      amd_comgr_metadata_kind_t Kind;
      checkError(amd_comgr_get_metadata_kind(Metadata, &Kind),
                 "get metadata kind after releasing data");
      if (Kind != AMD_COMGR_METADATA_KIND_MAP)
        fail("unexpected metadata kind");
      checkError(amd_comgr_destroy_metadata(Metadata),
                 "destroy output metadata");
    }
  }
  return Result;
}

static void checkOutputs(const std::vector<std::string> &Actual,
                         const std::vector<std::string> &Expected) {
  if (Actual != Expected)
    fail("unbundled outputs differ");
}

static void testFormat(compression::Format Format, uint16_t Version) {
  auto Compress = [&](StringRef Input) {
    return compress(Input, Format, Version);
  };
  std::string First = bundle({{Device, "first"}});
  std::string Second = bundle({{Device, "second"}});
  checkOutputs(unbundle(Compress(First), {Host, Device}), {"", "first"});
  checkOutputs(unbundle(Compress(First) + Compress(Second), {Host, Device}),
               {"", "first"});
  checkOutputs(unbundle(Compress(First), {Device, Device}), {"first", "first"});

  for (std::string Input :
       {bundle({{Device, "first"}, {Device, "second"}}),
        bundle({{Device, "generic"}, {Specific, "specific"}}),
        bundle({{Host, "host"}, {Device, "device"}, {Specific, "specific"}})})
    checkOutputs(unbundle(Compress(Input), {Host, Specific, Device}),
                 unbundle(Input, {Host, Specific, Device}));

  // Compare the private in-memory compatibility predicate with Clang's
  // file-based unbundler, including asymmetric feature matching and kind/triple
  // mismatches. Check expected results as well so both paths cannot silently
  // select the wrong member.
  struct CompatibilityCase {
    const char *Provided;
    const char *Requested;
    bool Compatible;
  };
  for (const auto &Case : {
           CompatibilityCase{Host, Host, true},
           CompatibilityCase{Device, Specific, true},
           CompatibilityCase{
               "hip-amdgcn-amd-amdhsa-unknown-gfx906:xnack-",
               "hip-amdgcn-amd-amdhsa-unknown-gfx906:sramecc+:xnack-", true},
           CompatibilityCase{
               "hip-amdgcn-amd-amdhsa-unknown-gfx906:sramecc+:xnack-",
               "hip-amdgcn-amd-amdhsa-unknown-gfx906:xnack-", false},
           CompatibilityCase{Specific, Device, false},
           CompatibilityCase{Specific, Specific, true},
           CompatibilityCase{
               Specific, "hip-amdgcn-amd-amdhsa-unknown-gfx900:xnack-", false},
           CompatibilityCase{
               "hip-amdgcn-amd-amdhsa-unknown-gfx906:sramecc+:xnack-",
               "hip-amdgcn-amd-amdhsa-unknown-gfx906:xnack-:sramecc+", true},
           CompatibilityCase{Device, "hipv4-amdgcn-amd-amdhsa-unknown-gfx900",
                             true},
           CompatibilityCase{"hipv4-amdgcn-amd-amdhsa-unknown-gfx900", Device,
                             true},
           CompatibilityCase{Device, "openmp-amdgcn-amd-amdhsa-unknown-gfx900",
                             true},
           CompatibilityCase{Device, "hip-amdgcn-amd-amdhsa-unknown-gfx906",
                             false},
           CompatibilityCase{Device, "hip-amdgcn-amd-amdpal-unknown-gfx900",
                             false},
       }) {
    std::string Input = bundle({{Case.Provided, "payload"}});
    auto Actual = unbundle(Compress(Input), {Case.Requested});
    if (Actual != unbundle(Input, {Case.Requested}))
      fail("in-memory/file mismatch: %s -> %s", Case.Provided, Case.Requested);
    if (Actual != std::vector<std::string>{Case.Compatible ? "payload" : ""})
      fail("unexpected compatibility: %s -> %s", Case.Provided, Case.Requested);
  }

  checkOutputs(unbundle(Compress(First), {Host}), {Compress(First)});
  checkOutputs(
      unbundle(Compress(First), {"hip-amdgcn-amd-amdhsa-unknown-gfx906"}),
      {""});
  checkOutputs(unbundle(Compress(bundle({{Device, ""}})), {Device}), {""});
  unbundle(Compress(bundle({{"invalid", "payload"}})), {Device}, false);

  std::string Truncated = First;
  support::endian::write64le(Truncated.data() + sizeof(Magic) - 1, 2);
  unbundle(Compress(Truncated), {Device}, false);
  for (size_t Field :
       {sizeof(Magic) + 7, sizeof(Magic) + 15, sizeof(Magic) + 23}) {
    std::string Overflow = First;
    support::endian::write64le(Overflow.data() + Field, UINT64_MAX);
    unbundle(Compress(Overflow), {Device}, false);
  }

  std::string Overreported = Compress(First);
  if (Version == 3)
    support::endian::write64le(Overreported.data() + 16, First.size() + 32);
  else
    support::endian::write32le(Overreported.data() + (Version == 1 ? 8 : 12),
                               First.size() + 32);
  auto Compressed = MemoryBuffer::getMemBuffer(Overreported, "", false);
  auto Decompressed = object::CompressedOffloadBundle::decompress(*Compressed);
  if (!Decompressed)
    fail("decompress: %s", toString(Decompressed.takeError()).c_str());
  if ((*Decompressed)->getBuffer() != First)
    fail("decompression did not return the actual output size");
  checkOutputs(unbundle(Overreported, {Device}), {"first"});

  if (Version != 1) {
    std::string BadSize = Compress(First);
    if (Version == 3)
      support::endian::write64le(BadSize.data() + 8, 0);
    else
      support::endian::write32le(BadSize.data() + 8, 0);
    unbundle(BadSize, {Device}, false);
  }
}

int main(int argc, char **argv) {
  bool CachedExecutablesOnly = false;
  bool VerifyCacheMapping = false;
  for (int Index = 1; Index < argc; ++Index) {
    StringRef Argument(argv[Index]);
    if (Argument == "--cached-executables")
      CachedExecutablesOnly = true;
    else if (Argument == "--verify-cache-mapping")
      VerifyCacheMapping = true;
    else
      fail("unknown argument: %s", argv[Index]);
  }
  auto Executable = MemoryBuffer::getFile(TEST_OBJ_DIR "/shared.so");
  if (!Executable)
    fail("failed to open source/shared.so");
  std::string ExecutableContents = (*Executable)->getBuffer().str();
  // Keep the cache entry large enough for MemoryBuffer::getFile to map it.
  ExecutableContents.append(64 * 1024, '\0');
  for (auto Format : {compression::Format::Zlib, compression::Format::Zstd}) {
    if (compression::getReasonIfUnsupported(Format))
      continue;
    for (uint16_t Version : {1, 2, 3}) {
      if (!CachedExecutablesOnly)
        testFormat(Format, Version);
      for (size_t Padding = 0; Padding < alignof(std::max_align_t); ++Padding) {
        std::string HostBytes(Padding, '\0');
        std::string Input =
            bundle({{Host, HostBytes}, {Device, ExecutableContents}});
        checkOutputs(
            unbundle(compress(Input, Format, Version), {Device}, true, true,
                     VerifyCacheMapping, /*ExpectCacheMapping=*/true),
            {ExecutableContents});
        // Cached output framing places the ELF after the host bytes and two
        // size fields. Varying the host length exercises every cache alignment;
        // an odd host length places the device output at an odd offset, which
        // must take the copying fallback.
        checkOutputs(unbundle(compress(Input, Format, Version), {Host, Device},
                              true, true, VerifyCacheMapping,
                              /*ExpectCacheMapping=*/Padding % 2 == 0),
                     {HostBytes, ExecutableContents});
      }
    }
  }
  if (CachedExecutablesOnly)
    return 0;
  unbundle("CCOB", {Device}, false);
  unbundle(StringRef("CCOB\x04\x00\x00\x00", 8), {Device}, false);
  return 0;
}
