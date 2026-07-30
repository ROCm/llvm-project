//===- TranslationCacheTest.cpp - translation cache unit tests ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "hotswap/cache/translation-cache.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TempDir {
  llvm::SmallString<128> Path;
  bool Valid = false;

  explicit TempDir(const char *Prefix) {
    std::error_code Ec = llvm::sys::fs::createUniqueDirectory(Prefix, Path);
    Valid = !Ec;
  }

  ~TempDir() {
    if (Valid)
      llvm::sys::fs::remove_directories(Path);
  }

  std::string file(const char *Name) const {
    llvm::SmallString<256> P(Path);
    llvm::sys::path::append(P, Name);
    return std::string(P);
  }
};

struct ScopedEnv {
  std::string Name;
  std::string OldValue;
  bool HadOldValue = false;

  ScopedEnv(const char *Name, const std::string &Value) : Name(Name) {
    if (const char *Old = std::getenv(Name)) {
      OldValue = Old;
      HadOldValue = true;
    }
    setenv(Name, Value.c_str(), 1);
  }

  ~ScopedEnv() {
    if (HadOldValue)
      setenv(Name.c_str(), OldValue.c_str(), 1);
    else
      unsetenv(Name.c_str());
  }
};

llvm::MemoryBufferRef bufRef(llvm::ArrayRef<uint8_t> V) {
  return llvm::MemoryBufferRef(
      llvm::StringRef(reinterpret_cast<const char *>(V.data()), V.size()), "");
}

// Minimal AMDGPU MsgPack metadata: an `amdhsa.kernels` array with one
// `.name`, which is all `listKernelNames` reads.
std::string amdgpuMetadataBlob(llvm::StringRef KernelName) {
  llvm::msgpack::Document Doc;
  llvm::msgpack::MapDocNode Root = Doc.getRoot().getMap(/*Convert=*/true);

  llvm::msgpack::DocNode Version = Doc.getArrayNode();
  Version.getArray().push_back(Doc.getNode(static_cast<uint64_t>(1)));
  Version.getArray().push_back(Doc.getNode(static_cast<uint64_t>(2)));
  Root["amdhsa.version"] = Version;

  llvm::msgpack::DocNode Kernel = Doc.getMapNode();
  Kernel.getMap()[".name"] = Doc.getNode(KernelName, /*Copy=*/true);

  llvm::msgpack::DocNode Kernels = Doc.getArrayNode();
  Kernels.getArray().push_back(Kernel);
  Root["amdhsa.kernels"] = Kernels;

  std::string Blob;
  Doc.writeToBlob(Blob);
  return Blob;
}

// Offset of the source byte the cache tests flip to perturb the SHA-256
// without disturbing any ELF structure the key builder parses. It lives in
// the reserved gap fakeAmdgpuElf() leaves between the ELF header and the
// metadata note.
constexpr size_t HashPerturbOffset = 80;
constexpr size_t NoteOffset = 128;
static_assert(sizeof(llvm::ELF::Elf64_Ehdr) <= HashPerturbOffset &&
                  HashPerturbOffset < NoteOffset,
              "perturb byte must lie in the ELF header-to-note padding gap");

// Build a 64-bit little-endian AMDGPU ELF carrying an NT_AMDGPU_METADATA
// note naming one kernel, so that listKernelNames succeeds and the
// translation-cache key builder accepts the object.
//
// Layout: [Elf64_Ehdr][padding to NoteOffset][note][.shstrtab][section
// headers]. The padding gap holds HashPerturbOffset.
llvm::SmallVector<uint8_t> fakeAmdgpuElf() {
  using namespace llvm;
  const std::string Blob = amdgpuMetadataBlob("cache_probe_kernel");

  // ELF note: Nhdr, then name and desc each padded to 4 bytes. n_namesz
  // counts the trailing NUL.
  constexpr StringLiteral NoteName = "AMDGPU";
  const uint32_t NameSz = NoteName.size() + 1;
  const uint32_t DescSz = Blob.size();
  const uint32_t NamePadded = alignTo(NameSz, 4);
  const uint32_t NoteSize =
      sizeof(ELF::Elf64_Nhdr) + NamePadded + alignTo(DescSz, 4);

  // Section header string table; offsets are derived as names are appended.
  std::string ShStr(1, '\0');
  auto addSectionName = [&](StringRef Name) {
    uint32_t Offset = ShStr.size();
    ShStr.append(Name.begin(), Name.end());
    ShStr.push_back('\0');
    return Offset;
  };
  const uint32_t NoteNameOffset = addSectionName(".note");
  const uint32_t ShStrNameOffset = addSectionName(".shstrtab");

  const uint32_t ShStrOffset = NoteOffset + NoteSize;
  const uint32_t ShdrOffset = alignTo(ShStrOffset + ShStr.size(), 8);
  const uint32_t Total = ShdrOffset + 3 * sizeof(ELF::Elf64_Shdr);

  SmallVector<uint8_t> D(Total, 0);
  auto writeStruct = [&](size_t Offset, const auto &S) {
    std::memcpy(D.data() + Offset, &S, sizeof(S));
  };

  ELF::Elf64_Ehdr Ehdr = {};
  Ehdr.e_ident[ELF::EI_MAG0] = 0x7f;
  Ehdr.e_ident[ELF::EI_MAG1] = 'E';
  Ehdr.e_ident[ELF::EI_MAG2] = 'L';
  Ehdr.e_ident[ELF::EI_MAG3] = 'F';
  Ehdr.e_ident[ELF::EI_CLASS] = ELF::ELFCLASS64;
  Ehdr.e_ident[ELF::EI_DATA] = ELF::ELFDATA2LSB;
  Ehdr.e_ident[ELF::EI_VERSION] = ELF::EV_CURRENT;
  Ehdr.e_ident[ELF::EI_OSABI] = ELF::ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ELF::ET_DYN;
  Ehdr.e_machine = ELF::EM_AMDGPU;
  Ehdr.e_version = ELF::EV_CURRENT;
  Ehdr.e_flags = 0x49; // arbitrary stable EF_AMDGPU flags hashed into the key
  Ehdr.e_ehsize = sizeof(ELF::Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(ELF::Elf64_Shdr);
  Ehdr.e_shnum = 3;
  Ehdr.e_shstrndx = 2;
  Ehdr.e_shoff = ShdrOffset;
  writeStruct(0, Ehdr);

  ELF::Elf64_Nhdr Nhdr = {};
  Nhdr.n_namesz = NameSz;
  Nhdr.n_descsz = DescSz;
  Nhdr.n_type = ELF::NT_AMDGPU_METADATA;
  writeStruct(NoteOffset, Nhdr);
  std::memcpy(D.data() + NoteOffset + sizeof(Nhdr), NoteName.data(),
              NoteName.size());
  std::memcpy(D.data() + NoteOffset + sizeof(Nhdr) + NamePadded, Blob.data(),
              DescSz);

  std::memcpy(D.data() + ShStrOffset, ShStr.data(), ShStr.size());

  // Section headers: [0] null, [1] .note (SHT_NOTE), [2] .shstrtab.
  ELF::Elf64_Shdr Shdrs[3] = {};
  Shdrs[1].sh_name = NoteNameOffset;
  Shdrs[1].sh_type = ELF::SHT_NOTE;
  Shdrs[1].sh_offset = NoteOffset;
  Shdrs[1].sh_size = NoteSize;
  Shdrs[1].sh_addralign = 4;
  Shdrs[2].sh_name = ShStrNameOffset;
  Shdrs[2].sh_type = ELF::SHT_STRTAB;
  Shdrs[2].sh_offset = ShStrOffset;
  Shdrs[2].sh_size = ShStr.size();
  Shdrs[2].sh_addralign = 1;
  std::memcpy(D.data() + ShdrOffset, Shdrs, sizeof(Shdrs));

  return D;
}

void writeTextFile(const std::string &Path, llvm::StringRef Text) {
  std::error_code Ec;
  llvm::raw_fd_ostream Os(Path, Ec);
  ASSERT_FALSE(Ec) << "cannot write " << Path << ": " << Ec.message();
  Os << Text;
}

void writeBinaryFile(const std::string &Path,
                     const std::vector<uint8_t> &Bytes) {
  std::error_code Ec;
  llvm::raw_fd_ostream Os(Path, Ec, llvm::sys::fs::OF_None);
  ASSERT_FALSE(Ec) << "cannot write " << Path << ": " << Ec.message();
  Os.write(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
}

COMGR::hotswap::TranslationCacheRequest
makeRequest(llvm::MemoryBufferRef Source, const std::string &RulesPath,
            const std::string &SourceGfx = "gfx1250",
            const std::string &TargetGfx = "gfx942") {
  COMGR::hotswap::TranslationCacheRequest Request;
  Request.SourceObject = Source;
  Request.SourceGfx = SourceGfx;
  Request.TargetGfx = TargetGfx;
  Request.SourceIsa = "amdgcn-amd-amdhsa--" + SourceGfx;
  Request.TargetIsa = "amdgcn-amd-amdhsa--" + TargetGfx;
  Request.CodeIsa = "amdgcn-amd-amdhsa--gfx942";
  Request.HotswapRulesPath = RulesPath;
  Request.CacheDirectory = llvm::sys::path::parent_path(RulesPath).str();
  Request.DeviceLibrariesIdentity = "unit-test-device-libraries-identity";
  Request.CacheDisabled = false;
  Request.OrigMach = 0x49;
  Request.EnableWritelaneRewrite = true;
  Request.EnableWaveNative = true;
  Request.StrictMode = true;
  return Request;
}

COMGR::hotswap::PipelineResult makeSuccessfulResult(
    std::vector<uint8_t> Hsaco = {0x7f, 'E', 'L', 'F', 1, 2, 3}) {
  COMGR::hotswap::PipelineResult Result;
  Result.Success = true;
  Result.Hsaco = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(reinterpret_cast<const char *>(Hsaco.data()),
                      Hsaco.size()),
      "");
  Result.LiftedCount = 7;
  Result.TotalCount = 7;
  return Result;
}

} // namespace

TEST(TranslationCache, FirstRunMissWriteSecondRunHit) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);

  auto First = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(First.Status, COMGR::hotswap::TranslationCacheStatus::Miss);

  auto Result = makeSuccessfulResult();
  auto Write = COMGR::hotswap::writeTranslationCache(Request, Result);
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess)
      << Write.Reason;

  auto Second = COMGR::hotswap::lookupTranslationCache(Request);
  ASSERT_EQ(Second.Status, COMGR::hotswap::TranslationCacheStatus::Hit)
      << Second.Reason;
  ASSERT_TRUE(Second.Result.Hsaco && Result.Hsaco);
  EXPECT_EQ(Second.Result.Hsaco->getBuffer(), Result.Hsaco->getBuffer());
  EXPECT_EQ(Second.Result.LiftedCount, Result.LiftedCount);
  EXPECT_EQ(Second.Result.TotalCount, Result.TotalCount);
}

TEST(TranslationCache, KernelNameParticipatesInCacheKey) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto WholeObject = makeRequest(bufRef(Source), Rules);

  auto WholeWrite = COMGR::hotswap::writeTranslationCache(
      WholeObject, makeSuccessfulResult());
  ASSERT_EQ(WholeWrite.Status,
            COMGR::hotswap::TranslationCacheStatus::WriteSuccess)
      << WholeWrite.Reason;

  auto PerKernel = WholeObject;
  PerKernel.KernelName = "cache_probe_kernel";
  auto PerKernelLookup = COMGR::hotswap::lookupTranslationCache(PerKernel);
  EXPECT_EQ(PerKernelLookup.Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);
  EXPECT_NE(PerKernelLookup.key, WholeWrite.key);

  auto PerKernelWrite = COMGR::hotswap::writeTranslationCache(
      PerKernel, makeSuccessfulResult({0x7f, 'E', 'L', 'F', 4, 5, 6}));
  ASSERT_EQ(PerKernelWrite.Status,
            COMGR::hotswap::TranslationCacheStatus::WriteSuccess)
      << PerKernelWrite.Reason;

  auto OtherKernel = WholeObject;
  OtherKernel.KernelName = "other_kernel";
  auto OtherKernelLookup = COMGR::hotswap::lookupTranslationCache(OtherKernel);
  EXPECT_EQ(OtherKernelLookup.Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);
  EXPECT_NE(OtherKernelLookup.key, PerKernelWrite.key);

  auto WholeObjectLookup = COMGR::hotswap::lookupTranslationCache(WholeObject);
  EXPECT_EQ(WholeObjectLookup.Status,
            COMGR::hotswap::TranslationCacheStatus::Hit);
}

TEST(TranslationCache, ChangedInputHashCausesMiss) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  ASSERT_EQ(
      COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult())
          .Status,
      COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  Source[HashPerturbOffset] ^= 0x1;
  auto Changed = makeRequest(bufRef(Source), Rules);
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Changed);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, ChangedIsaCausesMiss) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  ASSERT_EQ(
      COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult())
          .Status,
      COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  auto ChangedSourceIsa =
      makeRequest(bufRef(Source), Rules, "gfx1200", "gfx942");
  EXPECT_EQ(COMGR::hotswap::lookupTranslationCache(ChangedSourceIsa).Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);

  auto ChangedTargetIsa =
      makeRequest(bufRef(Source), Rules, "gfx1250", "gfx950");
  EXPECT_EQ(COMGR::hotswap::lookupTranslationCache(ChangedTargetIsa).Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, ChangedOptLevelCausesMiss) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  Request.OptLevel = 2;
  auto Write =
      COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  auto Changed = Request;
  Changed.OptLevel = 0;
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Changed);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, OldHotswapCacheDirDoesNotEnableCache) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv OldCacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", "");
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  Request.CacheDirectory = "";

  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Disabled);
}

TEST(TranslationCache, CorruptMetadataIsInvalid) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  auto Write =
      COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  writeTextFile(Write.MetadataPath, "not-json\n");
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Invalid);
  EXPECT_NE(Lookup.Reason.find("parse"), std::string::npos);
}

TEST(TranslationCache, CorruptObjectIsInvalid) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  auto Write =
      COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  writeBinaryFile(Write.ObjectPath, {1, 2, 3, 4});
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Invalid);
  EXPECT_NE(Lookup.Reason.find("cached_object_sha256"), std::string::npos);
}

TEST(TranslationCache, ReadonlyMissDoesNotWrite) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  Request.CacheReadonly = true;

  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Miss);

  auto Write =
      COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  EXPECT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::Disabled);

  auto Second = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Second.Status, COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, BypassedStatusHasStableString) {
  EXPECT_STREQ(COMGR::hotswap::translationCacheStatusString(
                   COMGR::hotswap::TranslationCacheStatus::Bypassed),
               "bypassed");
}

TEST(TranslationCache, SkipKernelListMatchesExactKernelName) {
  ScopedEnv Skip("HSA_HOTSWAP_CACHE_SKIP_KERNELS",
                 "other_kernel, target_kernel ,third_kernel");

  std::vector<std::string> Kernels = {"first_kernel", "target_kernel"};
  EXPECT_EQ(COMGR::hotswap::skippedKernelForTranslationCache(
                Kernels, "other_kernel, target_kernel ,third_kernel"),
            "target_kernel");
}

TEST(TranslationCache, SkipKernelListDoesNotUseSubstringMatching) {
  ScopedEnv Skip("HSA_HOTSWAP_CACHE_SKIP_KERNELS", "target");

  std::vector<std::string> Kernels = {"target_kernel"};
  EXPECT_TRUE(
      COMGR::hotswap::skippedKernelForTranslationCache(Kernels, "target")
          .empty());
}
