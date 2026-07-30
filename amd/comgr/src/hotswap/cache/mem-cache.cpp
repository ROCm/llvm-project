//===- mem-cache.cpp - In-memory hotswap translation cache ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mem-cache.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace COMGR::hotswap {

namespace {

// Default in-memory budget when AMD_COMGR_HOTSWAP_MEM_CACHE_BYTES is unset.
// Transpiled HSACOs range from a few MB to hundreds of MB; 1 GiB holds a
// working set of several large objects without an unbounded footprint.
constexpr size_t kDefaultBudgetBytes = 1ull << 30; // 1 GiB

// Parses the budget env var once. Returns the default when unset, 0 (disabled)
// when explicitly "0"/empty, or the parsed byte count. Never throws.
size_t resolveBudgetFromEnv() {
  const char *env = std::getenv("AMD_COMGR_HOTSWAP_MEM_CACHE_BYTES");
  if (!env)
    return kDefaultBudgetBytes;
  llvm::StringRef value(env);
  value = value.trim();
  if (value.empty())
    return kDefaultBudgetBytes;
  unsigned long long parsed = 0;
  if (value.getAsInteger(10, parsed)) // true on parse failure
    return kDefaultBudgetBytes;       // malformed -> fall back to default
  return static_cast<size_t>(parsed);
}

// ---- xxHash64 over source content (vendored from ROCr hotswap_cache.cpp) ----
// Used only to bucket entries; identity is settled by an exact memcmp, so
// correctness never depends on hash strength.
namespace xxh {
constexpr uint64_t kP1 = 11400714785074694791ULL;
constexpr uint64_t kP2 = 14029467366897019727ULL;
constexpr uint64_t kP3 = 1609587929392839161ULL;
constexpr uint64_t kP4 = 9650029242287828579ULL;
constexpr uint64_t kP5 = 2870177450012600261ULL;
inline uint64_t rol(uint64_t v, unsigned c) {
  return (v << c) | (v >> (64 - c));
}
inline uint64_t rd64(const unsigned char *b) {
  uint64_t v;
  std::memcpy(&v, b, 8);
  return v;
}
inline uint32_t rd32(const unsigned char *b) {
  uint32_t v;
  std::memcpy(&v, b, 4);
  return v;
}
inline uint64_t round(uint64_t a, uint64_t in) {
  a += in * kP2;
  a = rol(a, 31);
  return a * kP1;
}
inline uint64_t merge(uint64_t a, uint64_t v) {
  a ^= round(0, v);
  return a * kP1 + kP4;
}
inline uint64_t hash(const void *data, size_t size) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  const unsigned char *cur = bytes, *const end = bytes + size;
  constexpr uint64_t kSeed = 0x4f1bbcdc6762f36bULL;
  uint64_t h = 0;
  if (size >= 32) {
    const unsigned char *const be = end - 32;
    uint64_t l1 = kSeed + kP1 + kP2, l2 = kSeed + kP2, l3 = kSeed,
             l4 = kSeed - kP1;
    do {
      l1 = round(l1, rd64(cur));
      cur += 8;
      l2 = round(l2, rd64(cur));
      cur += 8;
      l3 = round(l3, rd64(cur));
      cur += 8;
      l4 = round(l4, rd64(cur));
      cur += 8;
    } while (cur <= be);
    h = rol(l1, 1) + rol(l2, 7) + rol(l3, 12) + rol(l4, 18);
    h = merge(h, l1);
    h = merge(h, l2);
    h = merge(h, l3);
    h = merge(h, l4);
  } else {
    h = kSeed + kP5;
  }
  h += size;
  while (static_cast<size_t>(end - cur) >= 8) {
    h ^= round(0, rd64(cur));
    h = rol(h, 27) * kP1 + kP4;
    cur += 8;
  }
  if (static_cast<size_t>(end - cur) >= 4) {
    h ^= static_cast<uint64_t>(rd32(cur)) * kP1;
    h = rol(h, 23) * kP2 + kP3;
    cur += 4;
  }
  while (cur != end) {
    h ^= static_cast<uint64_t>(*cur++) * kP5;
    h = rol(h, 11) * kP1;
  }
  h ^= h >> 33;
  h *= kP2;
  h ^= h >> 29;
  h *= kP3;
  return h ^ (h >> 32);
}
} // namespace xxh

// Cheap in-memory bucket key: xxHash64(source) + source_size + the transform
// fields already present as strings on the request. Deliberately does NOT
// re-hash the rules file CONTENTS or re-derive the loaded-image identity (both
// process-constant and expensive) -- those distinguish nothing within a
// process. The disk tier's SHA-256 key still folds them in for cross-process
// correctness. Empty string => uncacheable (empty/missing source).
std::string deriveMemBucketKey(const TranslationCacheRequest &request) {
  const void *src = request.SourceObject.getBufferStart();
  const size_t n = request.SourceObject.getBufferSize();
  if (src == nullptr || n == 0)
    return std::string();
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%016llx:%zx|",
                static_cast<unsigned long long>(xxh::hash(src, n)), n);
  std::string key(buf);
  key += request.SourceGfx;
  key.push_back('\x1f');
  key += request.TargetGfx;
  key.push_back('\x1f');
  key += request.SourceIsa;
  key.push_back('\x1f');
  key += request.TargetIsa;
  key.push_back('\x1f');
  key += request.CodeIsa;
  key.push_back('\x1f');
  key += request.HotswapRulesPath;
  key.push_back('\x1f');
  key += request.KernelName;
  key.push_back('\x1f');
  key += (request.StrictMode ? '1' : '0');
  key += (request.EnableWritelaneRewrite ? '1' : '0');
  key += (request.EnableWaveNative ? '1' : '0');
  key += (request.ForceScaledModrep ? '1' : '0');
  key += (request.AssumeHipGlobalOffsetZero ? '1' : '0');
  // Transform inputs not implied by the source bytes. Without these, two
  // identical-source requests differing only in these fields would collide
  // into one bucket and the source-only memcmp would wrongly match.
  // elf_machine/elf_flags and the source hash are omitted: the memcmp over
  // the source bytes already covers them. rules-file CONTENTS and the loaded-
  // image build identity are omitted because they are constant across a single
  // process (the disk key still folds them in for cross-process identity).
  {
    char nbuf[48];
    std::snprintf(nbuf, sizeof(nbuf), "|om=%d|ol=%u", request.OrigMach,
                  request.OptLevel);
    key += nbuf;
  }
  key.push_back('\x1f');
  key += request.DeviceLibrariesIdentity;
  return key;
}

// A retained copy of the source bytes, for the exact-compare that makes a hash
// collision impossible to mis-serve.
struct SourceBytes {
  std::vector<unsigned char> data;
  bool equals(const void *p, size_t n) const {
    return data.size() == n && std::memcmp(data.data(), p, n) == 0;
  }
};
using SourceBytesRef = std::shared_ptr<const SourceBytes>;

SourceBytesRef captureSource(const TranslationCacheRequest &request) {
  const auto *p = reinterpret_cast<const unsigned char *>(
      request.SourceObject.getBufferStart());
  const size_t n = request.SourceObject.getBufferSize();
  auto s = std::make_shared<SourceBytes>();
  s->data.assign(p, p + n);
  return s;
}

// One shared translation, plus its LRU position. The buffer is immutable once
// published.
struct Node {
  MemCacheEntryRef Entry;
  SourceBytesRef Source; // retained source bytes, for exact-compare
  // Iterator into the LRU list identifying this key's recency slot. Valid while
  // the node is in the map.
  std::list<std::string>::iterator LruIt;
};

// A cold miss in progress. The leader runs the producer; waiters block on CV.
struct Flight {
  std::mutex Mu;
  std::condition_variable Cv;
  bool Done = false;
  MemCacheEntryRef Entry; // published result (null on producer failure)
  SourceBytesRef Source;  // leader's source, for exact-compare on coalesce
  std::thread::id Leader; // the producer thread (reentrancy guard)
  size_t Waiters = 0;     // parked waiters, for the coalescing test hook
};

class MemCache {
public:
  explicit MemCache(size_t budget) : Budget(budget) {}

  size_t budget() const { return Budget; }

  MemCacheResult getOrCompute(const TranslationCacheRequest &request,
                              const TranslationProducer &producer);

  MemCacheMetrics snapshot() const;

#ifdef COMGR_HOTSWAP_MEM_CACHE_TESTING
  void resetForTesting(size_t budget);
  size_t entryCountForTesting() const;
  size_t inFlightCountForTesting() const;
  size_t waitForWaitersForTesting(const std::string &key, size_t count,
                                  unsigned timeoutMs);
#endif

private:
  // Look up a ready entry, refreshing its LRU position. Caller holds Mu.
  MemCacheEntryRef lookupLocked(const std::string &key,
                                const TranslationCacheRequest &request);
  // Insert a freshly produced entry and enforce the budget. Caller holds Mu.
  void insertLocked(const std::string &key, MemCacheEntryRef entry,
                    SourceBytesRef source);
  void evictToBudgetLocked();

  mutable std::mutex Mu; // guards Map, Lru, InFlight, and metric gauges
  std::unordered_map<std::string, Node> Map;
  std::list<std::string> Lru; // front = most recently used
  std::unordered_map<std::string, std::shared_ptr<Flight>> InFlight;

  const size_t Budget;
  size_t LiveBytes = 0;

  // Metrics. Guarded by Mu except the atomics, which are bumped off-lock.
  std::atomic<uint64_t> Lookups{0};
  std::atomic<uint64_t> Hits{0};
  std::atomic<uint64_t> Coalesced{0};
  std::atomic<uint64_t> Computed{0};
  std::atomic<uint64_t> ProducerCalls{0};
  std::atomic<uint64_t> ProducerFailures{0};
  std::atomic<uint64_t> Evictions{0};
  std::atomic<uint64_t> ReentrantComputes{0};
  uint64_t PeakLiveBytes = 0; // guarded by Mu
};

MemCacheEntryRef
MemCache::lookupLocked(const std::string &key,
                       const TranslationCacheRequest &request) {
  auto it = Map.find(key);
  if (it == Map.end())
    return nullptr;
  // Exact-compare guards against an xxHash64 collision handing back the wrong
  // translation. A mismatch (astronomically rare) is treated as a miss.
  const void *p = request.SourceObject.getBufferStart();
  const size_t n = request.SourceObject.getBufferSize();
  if (!it->second.Source || !it->second.Source->equals(p, n))
    return nullptr;
  // Move to front (most-recently-used).
  Lru.splice(Lru.begin(), Lru, it->second.LruIt);
  it->second.LruIt = Lru.begin();
  return it->second.Entry;
}

void MemCache::evictToBudgetLocked() {
  // Drop least-recently-used entries until we are within budget. Never evict
  // below one entry unless that single entry itself exceeds the budget (in
  // which case we still drop it -- it will live via the caller's shared_ptr).
  while (LiveBytes > Budget && !Lru.empty()) {
    const std::string &victimKey = Lru.back();
    auto it = Map.find(victimKey);
    if (it != Map.end()) {
      LiveBytes -= it->second.Entry->Bytes;
      if (it->second.Source)
        LiveBytes -= it->second.Source->data.size();
      Map.erase(it);
      Evictions.fetch_add(1, std::memory_order_relaxed);
    }
    Lru.pop_back();
  }
}

void MemCache::insertLocked(const std::string &key, MemCacheEntryRef entry,
                            SourceBytesRef source) {
  // If a concurrent leader already inserted this key (possible only across
  // distinct flights, which we prevent, but be defensive), replace accounting.
  auto existing = Map.find(key);
  if (existing != Map.end()) {
    LiveBytes -= existing->second.Entry->Bytes;
    if (existing->second.Source)
      LiveBytes -= existing->second.Source->data.size();
    Lru.erase(existing->second.LruIt);
    Map.erase(existing);
  }
  Lru.push_front(key);
  Node node;
  node.Entry = std::move(entry);
  node.Source = std::move(source);
  node.LruIt = Lru.begin();
  LiveBytes += node.Entry->Bytes;
  if (node.Source)
    LiveBytes += node.Source->data.size();
  Map.emplace(key, std::move(node));
  if (LiveBytes > PeakLiveBytes)
    PeakLiveBytes = LiveBytes;
  evictToBudgetLocked();
}

// Builds a shared, immutable entry from a producer result, moving the HSACO
// buffer out. Returns null if the result carries no usable HSACO.
MemCacheEntryRef adopt(PipelineResult &result) {
  if (!result.Success || !result.Hsaco)
    return nullptr;
  auto entry = std::make_shared<MemCacheEntry>();
  entry->Bytes = result.Hsaco->getBufferSize();
  entry->Hsaco = std::shared_ptr<llvm::MemoryBuffer>(result.Hsaco.release());
  entry->Attribution.C5SuppressedCount = result.C5SuppressedCount;
  entry->Attribution.C5SuppressionReason = result.C5SuppressionReason;
  entry->Attribution.UsesScratchPrivateSegment =
      result.UsesScratchPrivateSegment;
  entry->Attribution.SourcePrivateSegmentFixedSize =
      result.SourcePrivateSegmentFixedSize;
  entry->Attribution.TargetEnablePrivateSegment =
      result.TargetEnablePrivateSegment;
  entry->Attribution.TargetPrivateSegmentFixedSize =
      result.TargetPrivateSegmentFixedSize;
  entry->Attribution.LiftedCount = result.LiftedCount;
  entry->Attribution.TotalCount = result.TotalCount;
  entry->Attribution.ScaledDispatchFactor = result.ScaledDispatchFactor;
  return entry;
}

MemCacheResult MemCache::getOrCompute(const TranslationCacheRequest &request,
                                      const TranslationProducer &producer) {
  Lookups.fetch_add(1, std::memory_order_relaxed);

  MemCacheResult result;

  // Budget 0 => tier disabled: run the producer directly, cache nothing.
  if (Budget == 0) {
    PipelineResult produced = producer();
    ProducerCalls.fetch_add(1, std::memory_order_relaxed);
    result.Status = MemCacheStatus::Disabled;
    MemCacheEntryRef entry = adopt(produced);
    if (!entry)
      ProducerFailures.fetch_add(1, std::memory_order_relaxed);
    result.Entry = std::move(entry);
    result.LeaderResult = std::move(produced);
    return result;
  }

  const std::string key = deriveMemBucketKey(request);

  // Uncacheable request (empty source, missing gfx, no kernels, unreadable
  // rules): run the producer directly and cache nothing. Never terminate or
  // fabricate a key that could collide across distinct uncacheable inputs.
  if (key.empty()) {
    PipelineResult produced = producer();
    ProducerCalls.fetch_add(1, std::memory_order_relaxed);
    MemCacheEntryRef entry = adopt(produced);
    if (!entry)
      ProducerFailures.fetch_add(1, std::memory_order_relaxed);
    // Report Computed when it produced usable bytes, else ProducerFailed. The
    // entry is returned to the caller but not inserted into the map.
    result.Status =
        entry ? MemCacheStatus::Computed : MemCacheStatus::ProducerFailed;
    result.Entry = std::move(entry);
    result.LeaderResult = std::move(produced);
    return result;
  }

  // Now that the request is known cacheable (non-empty key implies a non-null,
  // non-empty source), capture the source bytes for the exact-compare guard.
  SourceBytesRef source = captureSource(request);

  std::shared_ptr<Flight> flight;
  bool isLeader = false;
  {
    std::unique_lock<std::mutex> lock(Mu);
    // Fast path: ready in the map.
    if (MemCacheEntryRef hit = lookupLocked(key, request)) {
      Hits.fetch_add(1, std::memory_order_relaxed);
      result.Status = MemCacheStatus::Hit;
      result.Entry = std::move(hit);
      return result;
    }
    // Is an identical request already in flight?
    auto it = InFlight.find(key);
    if (it != InFlight.end()) {
      flight = it->second;
    } else {
      flight = std::make_shared<Flight>();
      flight->Leader = std::this_thread::get_id();
      InFlight.emplace(key, flight);
      isLeader = true;
    }
  }

  if (isLeader) {
    // Leader runs the producer with NO cache lock held.
    PipelineResult produced = producer();
    ProducerCalls.fetch_add(1, std::memory_order_relaxed);
    MemCacheEntryRef entry = adopt(produced);

    {
      std::unique_lock<std::mutex> lock(Mu);
      if (entry)
        insertLocked(key, entry, source);
      InFlight.erase(key);
    }
    // Publish to waiters.
    {
      std::lock_guard<std::mutex> flock(flight->Mu);
      flight->Entry = entry;
      flight->Source = source;
      flight->Done = true;
    }
    flight->Cv.notify_all();

    if (!entry)
      ProducerFailures.fetch_add(1, std::memory_order_relaxed);
    Computed.fetch_add(1, std::memory_order_relaxed);
    result.Status =
        entry ? MemCacheStatus::Computed : MemCacheStatus::ProducerFailed;
    result.Entry = std::move(entry);
    result.LeaderResult = std::move(produced);
    return result;
  }

  // Waiter. Reentrancy guard: if this same thread is the leader of this flight
  // (a producer that recursively asks for its own key), do NOT block on
  // ourselves -- run the producer inline instead. Cannot cache (the leader
  // owns the flight), so this is a bypass compute.
  if (flight->Leader == std::this_thread::get_id()) {
    ReentrantComputes.fetch_add(1, std::memory_order_relaxed);
    PipelineResult produced = producer();
    ProducerCalls.fetch_add(1, std::memory_order_relaxed);
    MemCacheEntryRef entry = adopt(produced);
    if (!entry)
      ProducerFailures.fetch_add(1, std::memory_order_relaxed);
    result.Status =
        entry ? MemCacheStatus::Computed : MemCacheStatus::ProducerFailed;
    result.Entry = std::move(entry);
    result.LeaderResult = std::move(produced);
    return result;
  }

  MemCacheEntryRef coalesced;
  SourceBytesRef flightSource;
  {
    std::unique_lock<std::mutex> flock(flight->Mu);
    ++flight->Waiters;
    flight->Cv.wait(flock, [&] { return flight->Done; });
    --flight->Waiters;
    coalesced = flight->Entry;
    flightSource = flight->Source;
  }
  const void *p = request.SourceObject.getBufferStart();
  const size_t n = request.SourceObject.getBufferSize();
  const bool sameSource = flightSource && flightSource->equals(p, n);
  if (coalesced && sameSource) {
    Coalesced.fetch_add(1, std::memory_order_relaxed);
    result.Status = MemCacheStatus::Coalesced;
    result.Entry = std::move(coalesced);
    return result;
  }
  if (coalesced && !sameSource) {
    // xxHash64 collision between two concurrent distinct sources: the leader's
    // flight was for a different source, so we cannot use its bytes. RETRY the
    // whole operation: by now the leader has published + erased its flight, so
    // our retry either finds our own source in the map (lookupLocked memcmp),
    // or -- more likely -- misses (leader stored the OTHER source) and WE
    // become a fresh leader that runs + caches our result. This makes a
    // collided request self-heal into a normal cached entry instead of
    // perpetually bypassing. Bounded: retry loops only while the collision
    // persists.
    return getOrCompute(request, producer); // tail-recursive retry
  }
  // The leader's producer failed; nothing to hand back.
  result.Status = MemCacheStatus::ProducerFailed;
  return result;
}

MemCacheMetrics MemCache::snapshot() const {
  MemCacheMetrics m;
  m.Lookups = Lookups.load(std::memory_order_relaxed);
  m.Hits = Hits.load(std::memory_order_relaxed);
  m.Coalesced = Coalesced.load(std::memory_order_relaxed);
  m.Computed = Computed.load(std::memory_order_relaxed);
  m.ProducerCalls = ProducerCalls.load(std::memory_order_relaxed);
  m.ProducerFailures = ProducerFailures.load(std::memory_order_relaxed);
  m.Evictions = Evictions.load(std::memory_order_relaxed);
  m.ReentrantComputes = ReentrantComputes.load(std::memory_order_relaxed);
  m.BudgetBytes = Budget;
  {
    std::lock_guard<std::mutex> lock(Mu);
    m.LiveBytes = LiveBytes;
    m.PeakLiveBytes = PeakLiveBytes;
    m.Entries = Map.size();
    m.InFlight = InFlight.size();
  }
  return m;
}

#ifdef COMGR_HOTSWAP_MEM_CACHE_TESTING
void MemCache::resetForTesting(size_t) {
  std::lock_guard<std::mutex> lock(Mu);
  Map.clear();
  Lru.clear();
  InFlight.clear();
  LiveBytes = 0;
  PeakLiveBytes = 0;
  Lookups = 0;
  Hits = 0;
  Coalesced = 0;
  Computed = 0;
  ProducerCalls = 0;
  ProducerFailures = 0;
  Evictions = 0;
  ReentrantComputes = 0;
}

size_t MemCache::entryCountForTesting() const {
  std::lock_guard<std::mutex> lock(Mu);
  return Map.size();
}

size_t MemCache::inFlightCountForTesting() const {
  std::lock_guard<std::mutex> lock(Mu);
  return InFlight.size();
}

size_t MemCache::waitForWaitersForTesting(const std::string &key, size_t count,
                                          unsigned timeoutMs) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  for (;;) {
    std::shared_ptr<Flight> flight;
    {
      std::lock_guard<std::mutex> lock(Mu);
      auto it = InFlight.find(key);
      if (it != InFlight.end())
        flight = it->second;
    }
    size_t observed = 0;
    if (flight) {
      std::lock_guard<std::mutex> flock(flight->Mu);
      observed = flight->Waiters;
    }
    if (observed >= count || std::chrono::steady_clock::now() >= deadline)
      return observed;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
#endif

// Process-global instance. Budget is fixed for the process lifetime (resolved
// once); tests swap the whole instance via resetMemCacheForTesting.
MemCache &instance() {
  static MemCache *cache = new MemCache(resolveBudgetFromEnv());
  return *cache;
}

#ifdef COMGR_HOTSWAP_MEM_CACHE_TESTING
// A rebindable pointer so tests can install an instance with an overridden
// budget. In non-test builds instance() is a plain function-local static.
std::mutex &testInstanceMu() {
  static std::mutex m;
  return m;
}
MemCache *&testInstancePtr() {
  static MemCache *p = nullptr;
  return p;
}
MemCache &activeInstance() {
  std::lock_guard<std::mutex> lock(testInstanceMu());
  MemCache *&p = testInstancePtr();
  if (!p)
    p = &instance();
  return *p;
}
#else
MemCache &activeInstance() { return instance(); }
#endif

} // namespace

MemCacheResult getOrComputeTranslation(const TranslationCacheRequest &request,
                                       const TranslationProducer &producer) {
  return activeInstance().getOrCompute(request, producer);
}

MemCacheMetrics snapshotMemCacheMetrics() {
  return activeInstance().snapshot();
}

size_t memCacheBudgetBytes() { return activeInstance().budget(); }

#ifdef COMGR_HOTSWAP_MEM_CACHE_TESTING
void resetMemCacheForTesting(size_t budgetBytesOverride) {
  std::lock_guard<std::mutex> lock(testInstanceMu());
  // Own the heap instances WE create here so we can free the previous one
  // without ever deleting the process-immortal instance() singleton (which
  // testInstancePtr() may alias from activeInstance()'s lazy init).
  static MemCache *ownedTestInstance = nullptr;
  delete ownedTestInstance;
  ownedTestInstance = new MemCache(budgetBytesOverride);
  testInstancePtr() = ownedTestInstance;
}

size_t memCacheEntryCountForTesting() {
  return activeInstance().entryCountForTesting();
}

size_t memCacheInFlightCountForTesting() {
  return activeInstance().inFlightCountForTesting();
}

size_t waitForMemCacheWaitersForTesting(const TranslationCacheRequest &request,
                                        size_t count, unsigned timeoutMs) {
  const std::string key = deriveMemBucketKey(request);
  return activeInstance().waitForWaitersForTesting(key, count, timeoutMs);
}
#endif

} // namespace COMGR::hotswap
