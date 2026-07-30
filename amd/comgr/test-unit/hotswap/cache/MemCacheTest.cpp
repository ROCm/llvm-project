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

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace COMGR::hotswap;

namespace {

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

} // namespace
