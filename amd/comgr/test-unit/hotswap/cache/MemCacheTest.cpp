//===- MemCacheTest.cpp - In-memory translation cache tests ---------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/cache/mem-cache.h"
#include "hotswap/cache/pipeline.h"
#include "hotswap/cache/translation-cache.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace COMGR::hotswap;

namespace {

// A unique temporary directory removed on scope exit. Local to this TU (the
// TranslationCacheTest.cpp TempDir lives in that TU's anon namespace and is
// not shared). Used by the two-tier disk/mem population test below.
struct MemTempDir {
  llvm::SmallString<128> Path;
  explicit MemTempDir(const char *P) {
    llvm::sys::fs::createUniqueDirectory(P, Path);
  }
  ~MemTempDir() {
    if (!Path.empty())
      llvm::sys::fs::remove_directories(Path);
  }
};

// Minimal AMDGPU MsgPack metadata naming one kernel -- all listKernelNames
// (and hence the cache key builder) reads. Adapted from TranslationCacheTest.
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

// Build a 64-bit AMDGPU ELF with an NT_AMDGPU_METADATA note naming `kernel`.
// Parameterizing the kernel name gives each call a distinct source hash, so
// distinct requests get distinct cache keys. Adapted from the fake ELF in
// TranslationCacheTest.cpp (kept local so that test stays untouched).
std::string makeFakeAmdgpuElf(llvm::StringRef kernel) {
  using namespace llvm;
  constexpr size_t NoteOffset = 128;
  const std::string Blob = amdgpuMetadataBlob(kernel);

  constexpr StringLiteral NoteName = "AMDGPU";
  const uint32_t NameSz = NoteName.size() + 1;
  const uint32_t DescSz = Blob.size();
  const uint32_t NamePadded = alignTo(NameSz, 4);
  const uint32_t NoteSize =
      sizeof(ELF::Elf64_Nhdr) + NamePadded + alignTo(DescSz, 4);

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
  Ehdr.e_flags = 0x49;
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

  return std::string(reinterpret_cast<const char *>(D.data()), D.size());
}

TranslationCacheRequest makeRequest(const std::string &elf,
                                    llvm::StringRef target = "gfx950") {
  TranslationCacheRequest req;
  req.SourceObject = llvm::MemoryBufferRef(elf, "source");
  req.SourceGfx = "gfx1250";
  req.TargetGfx = target;
  req.SourceIsa = "amdgcn-amd-amdhsa--gfx1250";
  req.TargetIsa = std::string("amdgcn-amd-amdhsa--") + target.str();
  req.CodeIsa = req.TargetIsa;
  req.CacheDirectory = ""; // disk tier irrelevant here
  req.CacheDisabled = true;
  return req;
}

// A producer that hands back a fresh HSACO of the requested size, counting how
// many times it actually ran. The bytes are arbitrary but deterministic.
PipelineResult makeProduced(std::atomic<int> &calls, size_t bytes,
                            bool success = true) {
  calls.fetch_add(1, std::memory_order_relaxed);
  PipelineResult r;
  if (success) {
    std::string data(bytes, 'X');
    r.Hsaco = llvm::MemoryBuffer::getMemBufferCopy(data, "hsaco");
    r.Success = true;
    r.LiftedCount = 7; // arbitrary attribution to verify round-trip
    r.ScaledDispatchFactor = 2;
  } else {
    r.Success = false;
    r.FailReason = "forced failure";
  }
  return r;
}

class MemCacheTest : public ::testing::Test {
protected:
  void SetUp() override { resetMemCacheForTesting(kBudget); }
  // Restore the default source-bucket hash after every test so a degenerate
  // hash installed by the collision test cannot leak into another test.
  void TearDown() override { setMemCacheHashFnForTesting(nullptr); }
  static constexpr size_t kBudget = 8ull << 20; // 8 MiB
};

// --- Basic miss-then-hit --------------------------------------------------

TEST_F(MemCacheTest, MissThenHit) {
  std::string elf = makeFakeAmdgpuElf("kern");
  auto req = makeRequest(elf);
  std::atomic<int> calls{0};

  auto r1 =
      getOrComputeTranslation(req, [&] { return makeProduced(calls, 1024); });
  EXPECT_EQ(r1.Status, MemCacheStatus::Computed);
  ASSERT_NE(r1.Entry, nullptr);
  EXPECT_EQ(calls.load(), 1);
  EXPECT_EQ(r1.Entry->Attribution.LiftedCount, 7);
  EXPECT_EQ(r1.Entry->Attribution.ScaledDispatchFactor, 2u);

  auto r2 =
      getOrComputeTranslation(req, [&] { return makeProduced(calls, 1024); });
  EXPECT_EQ(r2.Status, MemCacheStatus::Hit);
  ASSERT_NE(r2.Entry, nullptr);
  EXPECT_EQ(calls.load(), 1); // producer NOT called again
  // Same underlying buffer shared, zero-copy.
  EXPECT_EQ(r1.Entry->Hsaco.get(), r2.Entry->Hsaco.get());
}

// --- Key sensitivity: different target => different entry -----------------

TEST_F(MemCacheTest, DistinctKeysDoNotAlias) {
  std::string elf = makeFakeAmdgpuElf("kern");
  std::atomic<int> calls{0};
  auto a = getOrComputeTranslation(makeRequest(elf, "gfx950"),
                                   [&] { return makeProduced(calls, 1024); });
  auto b = getOrComputeTranslation(makeRequest(elf, "gfx942"),
                                   [&] { return makeProduced(calls, 1024); });
  EXPECT_EQ(a.Status, MemCacheStatus::Computed);
  EXPECT_EQ(b.Status, MemCacheStatus::Computed);
  EXPECT_EQ(calls.load(), 2);
  EXPECT_NE(a.Entry->Hsaco.get(), b.Entry->Hsaco.get());
}

// --- Single-flight coalescing --------------------------------------------

TEST_F(MemCacheTest, ConcurrentIdenticalCoalesceToOneProducer) {
  std::string elf = makeFakeAmdgpuElf("kern");
  auto req = makeRequest(elf);
  constexpr int kThreads = 16;

  std::atomic<int> calls{0};
  std::atomic<int> producerEntered{0};
  std::atomic<bool> release{false};

  // Producer blocks until all waiters have parked, so we deterministically
  // force coalescing rather than racing.
  auto producer = [&]() -> PipelineResult {
    producerEntered.fetch_add(1, std::memory_order_relaxed);
    // Wait until the test releases us (after confirming N-1 waiters parked).
    while (!release.load(std::memory_order_acquire))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return makeProduced(calls, 4096);
  };

  std::vector<std::thread> threads;
  std::vector<MemCacheResult> results(kThreads);
  for (int i = 0; i < kThreads; ++i)
    threads.emplace_back(
        [&, i] { results[i] = getOrComputeTranslation(req, producer); });

  // Wait until the leader entered the producer, then give the other threads a
  // best-effort chance to park. We do NOT assert an exact parked-waiter count:
  // a thread may be between "found the in-flight entry" and "incremented
  // Waiters" when we sample, which is a benign transient. The load-bearing
  // invariant is that the producer runs EXACTLY ONCE regardless of how the
  // non-leaders interleave -- that is what we assert below.
  while (producerEntered.load() == 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  // Opportunistically confirm coalescing pressure built up (not an equality
  // assertion; just that at least some waiters parked before release).
  size_t waiters = waitForMemCacheWaitersForTesting(req, kThreads - 1, 2000);
  EXPECT_GT(waiters, 0u);

  release.store(true, std::memory_order_release);
  for (auto &t : threads)
    t.join();

  // THE invariant: the producer ran exactly once; every thread got the same
  // buffer; exactly one thread is the leader (Computed) and the rest either
  // coalesced onto it or (if they arrived after publish) hit the ready map.
  EXPECT_EQ(calls.load(), 1);
  EXPECT_EQ(producerEntered.load(), 1);
  int computed = 0, coalesced = 0, hit = 0;
  const llvm::MemoryBuffer *buf = nullptr;
  for (auto &r : results) {
    ASSERT_NE(r.Entry, nullptr);
    if (!buf)
      buf = r.Entry->Hsaco.get();
    EXPECT_EQ(r.Entry->Hsaco.get(), buf); // all share one buffer
    switch (r.Status) {
    case MemCacheStatus::Computed:
      ++computed;
      break;
    case MemCacheStatus::Coalesced:
      ++coalesced;
      break;
    case MemCacheStatus::Hit:
      ++hit;
      break;
    default:
      ADD_FAILURE() << "unexpected status " << static_cast<int>(r.Status);
      break;
    }
  }
  EXPECT_EQ(computed, 1);                   // exactly one leader
  EXPECT_EQ(coalesced + hit, kThreads - 1); // everyone else shared it
}

// --- Coalesced waiters of a failed leader get the leader's status ----------

// When the single-flight leader's producer fails, every coalesced waiter must
// observe the SAME deterministic failure: Status == ProducerFailed, Entry ==
// null, AND the leader's opaque ProducerStatus code -- never a generic error
// that varies with leader-vs-waiter timing.
TEST_F(MemCacheTest, CoalescedWaitersOfFailedLeaderGetLeaderStatus) {
  std::string elf = makeFakeAmdgpuElf("kern");
  auto req = makeRequest(elf);
  constexpr int kThreads = 16;
  constexpr int kFailCode = 7; // arbitrary nonzero opaque producer code

  std::atomic<int> calls{0};
  std::atomic<int> producerEntered{0};
  std::atomic<bool> release{false};

  // Leader parks in the producer until all waiters have coalesced, then fails
  // (no Hsaco) carrying a specific opaque status code.
  auto producer = [&]() -> PipelineResult {
    producerEntered.fetch_add(1, std::memory_order_relaxed);
    while (!release.load(std::memory_order_acquire))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    PipelineResult r = makeProduced(calls, 0, /*success=*/false);
    r.ProducerStatus = kFailCode;
    return r;
  };

  std::vector<std::thread> threads;
  std::vector<MemCacheResult> results(kThreads);
  for (int i = 0; i < kThreads; ++i)
    threads.emplace_back(
        [&, i] { results[i] = getOrComputeTranslation(req, producer); });

  while (producerEntered.load() == 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  size_t waiters = waitForMemCacheWaitersForTesting(req, kThreads - 1, 2000);
  EXPECT_GT(waiters, 0u);

  release.store(true, std::memory_order_release);
  for (auto &t : threads)
    t.join();

  // The producer ran exactly once (the leader); every thread saw the failure.
  EXPECT_EQ(calls.load(), 1);
  EXPECT_EQ(producerEntered.load(), 1);
  for (auto &r : results) {
    EXPECT_EQ(r.Status, MemCacheStatus::ProducerFailed);
    EXPECT_EQ(r.Entry, nullptr);
    // Deterministic: leader AND every coalesced waiter report the same code.
    EXPECT_EQ(r.ProducerStatus, kFailCode);
  }
}

// --- Byte-budget LRU eviction --------------------------------------------

TEST_F(MemCacheTest, ByteBudgetEvictsLeastRecentlyUsed) {
  std::atomic<int> calls{0};
  // Each entry now costs retained source bytes + output bytes. Size the
  // budget from the real per-entry footprint so exactly 3 entries fit
  // (>=3x, strictly <4x), independent of the fake-ELF source size.
  const size_t kOut = 1024;
  const size_t kSrc = makeFakeAmdgpuElf("a").size();
  const size_t kPerEntry = kSrc + kOut;
  resetMemCacheForTesting(3 * kPerEntry + kPerEntry / 2); // room for 3, not 4

  auto insert = [&](llvm::StringRef k) {
    std::string elf = makeFakeAmdgpuElf(k);
    return getOrComputeTranslation(makeRequest(elf),
                                   [&] { return makeProduced(calls, 1024); });
  };

  auto a = insert("a");
  auto b = insert("b");
  auto c = insert("c");
  EXPECT_EQ(memCacheEntryCountForTesting(), 3u);

  // Touch 'a' so it is MRU; inserting 'd' should evict 'b' (now LRU).
  std::string elfA = makeFakeAmdgpuElf("a");
  auto aHit = getOrComputeTranslation(
      makeRequest(elfA), [&] { return makeProduced(calls, 1024); });
  EXPECT_EQ(aHit.Status, MemCacheStatus::Hit);

  auto d = insert("d");
  EXPECT_EQ(d.Status, MemCacheStatus::Computed);
  EXPECT_LE(memCacheEntryCountForTesting(), 3u);

  // 'a' still cached (recently touched); 'b' evicted (recompute => Computed).
  std::string elfAgainA = makeFakeAmdgpuElf("a");
  EXPECT_EQ(getOrComputeTranslation(makeRequest(elfAgainA),
                                    [&] { return makeProduced(calls, 1024); })
                .Status,
            MemCacheStatus::Hit);
  std::string elfB = makeFakeAmdgpuElf("b");
  EXPECT_EQ(getOrComputeTranslation(makeRequest(elfB),
                                    [&] { return makeProduced(calls, 1024); })
                .Status,
            MemCacheStatus::Computed);
}

// --- Eviction never frees a buffer a caller still holds -------------------

TEST_F(MemCacheTest, EvictedEntrySurvivesWhileReferenced) {
  resetMemCacheForTesting(2 * 1024);
  std::atomic<int> calls{0};

  std::string elfA = makeFakeAmdgpuElf("a");
  auto a = getOrComputeTranslation(makeRequest(elfA),
                                   [&] { return makeProduced(calls, 1024); });
  ASSERT_NE(a.Entry, nullptr);
  const llvm::MemoryBuffer *aBuf = a.Entry->Hsaco.get();

  // Fill past budget to force 'a' out of the map while we still hold a.Entry.
  for (char c = 'b'; c <= 'e'; ++c) {
    std::string elf = makeFakeAmdgpuElf(llvm::StringRef(&c, 1));
    getOrComputeTranslation(makeRequest(elf),
                            [&] { return makeProduced(calls, 1024); });
  }

  // 'a' is evicted from the map...
  std::string elfA2 = makeFakeAmdgpuElf("a");
  EXPECT_EQ(getOrComputeTranslation(makeRequest(elfA2),
                                    [&] { return makeProduced(calls, 1024); })
                .Status,
            MemCacheStatus::Computed);
  // ...but our reference is still valid and unchanged (no UAF).
  EXPECT_EQ(a.Entry->Hsaco.get(), aBuf);
  EXPECT_EQ(a.Entry->Hsaco->getBufferSize(), 1024u);
}

// --- Producer failure is not cached --------------------------------------

TEST_F(MemCacheTest, ProducerFailureNotCached) {
  std::string elf = makeFakeAmdgpuElf("kern");
  auto req = makeRequest(elf);
  std::atomic<int> calls{0};

  auto r1 = getOrComputeTranslation(
      req, [&] { return makeProduced(calls, 0, /*success=*/false); });
  EXPECT_EQ(r1.Status, MemCacheStatus::ProducerFailed);
  EXPECT_EQ(r1.Entry, nullptr);

  // Next call must re-run the producer (failure was not cached).
  auto r2 = getOrComputeTranslation(
      req, [&] { return makeProduced(calls, 1024, /*success=*/true); });
  EXPECT_EQ(r2.Status, MemCacheStatus::Computed);
  ASSERT_NE(r2.Entry, nullptr);
  EXPECT_EQ(calls.load(), 2);
}

// --- Disabled tier (budget 0) bypasses cache -----------------------------

TEST_F(MemCacheTest, ZeroBudgetDisablesTier) {
  resetMemCacheForTesting(0);
  std::string elf = makeFakeAmdgpuElf("kern");
  auto req = makeRequest(elf);
  std::atomic<int> calls{0};

  auto r1 =
      getOrComputeTranslation(req, [&] { return makeProduced(calls, 1024); });
  auto r2 =
      getOrComputeTranslation(req, [&] { return makeProduced(calls, 1024); });
  EXPECT_EQ(r1.Status, MemCacheStatus::Disabled);
  EXPECT_EQ(r2.Status, MemCacheStatus::Disabled);
  EXPECT_EQ(calls.load(), 2); // no caching
  EXPECT_EQ(memCacheEntryCountForTesting(), 0u);
}

// --- Uncacheable request (empty key) bypasses safely ---------------------

TEST_F(MemCacheTest, UncacheableRequestBypasses) {
  // Empty source object => translationCacheKey returns "" => bypass.
  TranslationCacheRequest req;
  req.SourceObject = llvm::MemoryBufferRef(llvm::StringRef(""), "empty");
  req.SourceGfx = "gfx1250";
  req.TargetGfx = "gfx950";
  std::atomic<int> calls{0};
  auto r =
      getOrComputeTranslation(req, [&] { return makeProduced(calls, 1024); });
  // Producer still ran (we do not drop the request), just nothing cached.
  EXPECT_EQ(calls.load(), 1);
  EXPECT_EQ(memCacheEntryCountForTesting(), 0u);
}

// --- Every key-determining transform field must distinguish entries --------

TEST_F(MemCacheTest, KeyFieldSensitivity) {
  std::string elf = makeFakeAmdgpuElf("kern");

  // Each lambda mutates one field of a copy of the base request; the mutated
  // request must NOT alias the base (both Computed, distinct buffers).
  auto checkField = [&](const char *label,
                        std::function<void(TranslationCacheRequest &)> mutate) {
    SCOPED_TRACE(label);
    std::atomic<int> calls{0};
    resetMemCacheForTesting(8u << 20);
    auto base = makeRequest(elf);
    auto variant = base;
    mutate(variant);
    auto a = getOrComputeTranslation(base,
                                     [&] { return makeProduced(calls, 1024); });
    auto b = getOrComputeTranslation(variant,
                                     [&] { return makeProduced(calls, 1024); });
    EXPECT_EQ(a.Status, MemCacheStatus::Computed);
    EXPECT_EQ(b.Status, MemCacheStatus::Computed); // NOT Hit -> distinct keys
    ASSERT_NE(a.Entry, nullptr);
    ASSERT_NE(b.Entry, nullptr);
    EXPECT_NE(a.Entry->Hsaco.get(), b.Entry->Hsaco.get());
  };

  checkField("TargetGfx/TargetIsa", [](TranslationCacheRequest &r) {
    r.TargetGfx = "gfx942";
    r.TargetIsa = "amdgcn-amd-amdhsa--gfx942";
    r.CodeIsa = r.TargetIsa;
  });
  checkField("OptLevel", [](TranslationCacheRequest &r) { r.OptLevel = 3; });
  checkField("OrigMach", [](TranslationCacheRequest &r) { r.OrigMach = 42; });
  checkField("StrictMode",
             [](TranslationCacheRequest &r) { r.StrictMode = true; });
  checkField("EnableWaveNative",
             [](TranslationCacheRequest &r) { r.EnableWaveNative = false; });
  checkField("DeviceLibrariesIdentity", [](TranslationCacheRequest &r) {
    r.DeviceLibrariesIdentity = "devlibs-v2";
  });
}

// --- A forced hash collision must never mis-serve a different source -------

TEST_F(MemCacheTest, HashCollisionDoesNotMisServe) {
  // Degenerate hash: every source lands in one bucket, so the exact-memcmp
  // guard in lookupLocked is the ONLY thing keeping distinct sources apart.
  setMemCacheHashFnForTesting([](const void *, size_t) { return 0ull; });
  resetMemCacheForTesting(8u << 20);
  std::atomic<int> calls{0};

  // Producer whose OUTPUT content encodes which source it was built for, so a
  // mis-serve (returning Y's bytes for X) is directly detectable. The tag byte
  // is the source's first content byte after the fake-ELF header pad.
  auto taggedProducer = [&](char tag) {
    return [&, tag]() -> PipelineResult {
      calls.fetch_add(1, std::memory_order_relaxed);
      PipelineResult r;
      std::string data(1024, tag);
      r.Hsaco = llvm::MemoryBuffer::getMemBufferCopy(data, "hsaco");
      r.Success = true;
      return r;
    };
  };

  // Two DISTINCT sources forced into the same bucket.
  std::string elfA = makeFakeAmdgpuElf("A");
  std::string elfB = makeFakeAmdgpuElf("B");
  ASSERT_NE(elfA, elfB); // genuinely different source bytes

  auto a = getOrComputeTranslation(makeRequest(elfA), taggedProducer('A'));
  auto b = getOrComputeTranslation(makeRequest(elfB), taggedProducer('B'));
  ASSERT_NE(a.Entry, nullptr);
  ASSERT_NE(b.Entry, nullptr);

  // THE invariant: neither request received the other's bytes. A's output is
  // all 'A', B's is all 'B' -- a collision alias would have handed one the
  // other's buffer.
  EXPECT_EQ(a.Entry->Hsaco->getBuffer()[0], 'A');
  EXPECT_EQ(b.Entry->Hsaco->getBuffer()[0], 'B');
  EXPECT_NE(a.Entry->Hsaco.get(), b.Entry->Hsaco.get());

  // Re-request each source repeatedly under the forced collision: every result
  // must carry its OWN tag, never the other's. (Whether a given call is a Hit
  // or a recompute is unspecified under a forced collision -- the single-slot
  // design may thrash -- but the CONTENT must always be correct.)
  for (int i = 0; i < 4; ++i) {
    auto ra = getOrComputeTranslation(makeRequest(elfA), taggedProducer('A'));
    auto rb = getOrComputeTranslation(makeRequest(elfB), taggedProducer('B'));
    ASSERT_NE(ra.Entry, nullptr);
    ASSERT_NE(rb.Entry, nullptr);
    EXPECT_EQ(ra.Entry->Hsaco->getBuffer()[0], 'A') << "iter " << i;
    EXPECT_EQ(rb.Entry->Hsaco->getBuffer()[0], 'B') << "iter " << i;
  }

  setMemCacheHashFnForTesting(nullptr); // restore (TearDown also does this)
}

// --- Reentrant producer for the same key must not deadlock ----------------

TEST_F(MemCacheTest, ReentrantProducerDoesNotDeadlock) {
  resetMemCacheForTesting(8u << 20);
  std::string elf = makeFakeAmdgpuElf("R");
  auto req = makeRequest(elf);
  std::atomic<int> calls{0};
  std::atomic<int> depth{0};
  std::function<PipelineResult()> producer = [&]() -> PipelineResult {
    if (depth.fetch_add(1) == 0) {
      // Re-enter for the same key from within the producer. The reentrancy
      // guard must run this inline (a bypass compute), not block on ourselves.
      auto inner = getOrComputeTranslation(req, producer);
      (void)inner; // must return, not hang
    }
    return makeProduced(calls, 1024);
  };
  auto r = getOrComputeTranslation(req, producer);
  EXPECT_NE(r.Entry, nullptr); // completed without deadlock
  EXPECT_GE(calls.load(), 1);
}

// --- Disk hit repopulates the mem tier, then serves a pure mem hit --------

TEST_F(MemCacheTest, DiskHitPopulatesMemThenMemHit) {
  resetMemCacheForTesting(8u << 20);
  MemTempDir dir("hotswap_memcache_test");
  std::atomic<int> calls{0};
  std::string elf = makeFakeAmdgpuElf("T");

  // A request whose disk tier is ENABLED and points at our temp dir.
  TranslationCacheRequest req = makeRequest(elf);
  req.CacheDirectory = std::string(dir.Path);
  req.CacheDisabled = false;

  // (1) Cold: producer runs the transform AND writes disk.
  auto cold = getOrComputeTranslation(req, [&] {
    PipelineResult r = makeProduced(calls, 1024);
    writeTranslationCache(req, r); // real disk write
    return r;
  });
  EXPECT_EQ(cold.Status, MemCacheStatus::Computed);

  // (2) Cross-tier: clear mem (fresh instance), keep disk. Producer now does a
  //     real disk lookup instead of the transform; on hit it returns the disk
  //     result, which repopulates mem.
  resetMemCacheForTesting(8u << 20);
  int producerRuns = 0;
  auto xproc = getOrComputeTranslation(req, [&] {
    ++producerRuns;
    TranslationCacheLookup lk = lookupTranslationCache(req);
    if (lk.Status == TranslationCacheStatus::Hit && lk.Result.Hsaco)
      return std::move(lk.Result);    // disk hit -> no transform
    return makeProduced(calls, 1024); // (should not happen)
  });
  EXPECT_EQ(xproc.Status, MemCacheStatus::Computed);
  EXPECT_EQ(producerRuns, 1); // producer ran once (the disk read)

  // (3) Warm: pure mem hit, producer NOT run again.
  int producerRuns2 = 0;
  auto warm = getOrComputeTranslation(req, [&] {
    ++producerRuns2;
    return makeProduced(calls, 1024);
  });
  EXPECT_EQ(warm.Status, MemCacheStatus::Hit);
  EXPECT_EQ(producerRuns2, 0);
  ASSERT_NE(warm.Entry, nullptr);
  ASSERT_NE(xproc.Entry, nullptr);
  EXPECT_EQ(warm.Entry->Hsaco.get(), xproc.Entry->Hsaco.get());
}

} // namespace
