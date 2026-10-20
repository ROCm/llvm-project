#include "unbundle_test.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fstream>
#endif

using namespace llvm;
using namespace UnbundleTest;

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

static std::vector<std::string>
unbundle(StringRef Input, std::vector<const char *> Targets,
         bool ExpectSuccess = true, bool CheckExecutable = false,
         bool VerifyCacheMapping = false, bool ExpectCacheMapping = false) {
  auto Data = unbundleData(Input, Targets, ExpectSuccess, CheckExecutable);

  std::vector<std::string> Result(Data.size());
  for (size_t Count = Data.size(); Count; --Count) {
    size_t Index = Count - 1;
    size_t NameSize = 0;
    checkError(amd_comgr_get_data_name(Data[Index], &NameSize, nullptr),
               "get name size");
    std::string Name(NameSize, '\0');
    checkError(amd_comgr_get_data_name(Data[Index], &NameSize, Name.data()),
               "get name");
    std::string ExpectedName = std::string("bundle-") + Targets[Index] +
                               (CheckExecutable ? ".o" : ".bc");
    if (StringRef(Name.c_str()) != ExpectedName)
      fail("unexpected unbundled output name");
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
    if (reinterpret_cast<uintptr_t>(Pointer) % alignof(std::max_align_t) != 0)
      fail("borrowed output is not properly aligned");
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
  checkOutputs(unbundle(Compress("not a binary bundle"), {Device}), {""});
  unbundle(Compress(StringRef(Magic, sizeof(Magic))), {Device}, false);
  unbundle(Compress(First), {"invalid"}, false);
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
  std::string Log;
  raw_string_ostream LogStream(Log);
  auto Decompressed =
      object::CompressedOffloadBundle::decompress(*Compressed, &LogStream);
  if (!Decompressed)
    fail("decompress: %s", toString(Decompressed.takeError()).c_str());
  if ((*Decompressed)->getBuffer() != First)
    fail("decompression did not return the actual output size");
  if (!StringRef(Log).contains("Hashes match: Yes"))
    fail("decompression hash did not use the actual output size");
  checkOutputs(unbundle(Overreported, {Device}), {"first"});

  std::string Underreported = Compress(First);
  if (Version == 3)
    support::endian::write64le(Underreported.data() + 16, First.size() - 1);
  else
    support::endian::write32le(Underreported.data() + (Version == 1 ? 8 : 12),
                               First.size() - 1);
  unbundle(Underreported, {Device}, false);

  std::string BadMethod = Compress(First);
  support::endian::write16le(BadMethod.data() + 6, UINT16_MAX);
  unbundle(BadMethod, {Device}, false);

  size_t HeaderSize = Version == 1 ? 20 : Version == 2 ? 24 : 32;
  for (size_t Length = 8; Length < HeaderSize; ++Length)
    unbundle(StringRef(Compress(First)).take_front(Length), {Device}, false);

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
        checkOutputs(unbundle(Input, {Host, Device}, true, true),
                     {HostBytes, ExecutableContents});
        checkOutputs(unbundle(compress(Input, Format, Version), {Device}, true,
                              true, VerifyCacheMapping,
                              /*ExpectCacheMapping=*/true),
                     {ExecutableContents});
        // Cached output framing aligns every member relative to the mapped
        // entry. Varying the host length exercises each padding value and
        // verifies that the device output still shares the cache mapping.
        checkOutputs(unbundle(compress(Input, Format, Version), {Host, Device},
                              true, true, VerifyCacheMapping,
                              /*ExpectCacheMapping=*/true),
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
