//===- mem-cache.h - In-memory hotswap translation cache ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A process-global, in-memory tier that sits in front of the on-disk
// translation cache. It coalesces concurrent identical rewrites (single
// flight), serves repeat in-process lookups without re-reading or re-hashing
// disk, and bounds its footprint with a byte-budget LRU.
//
// Ownership model: an entry holds the transpiled HSACO as a shared, immutable
// buffer. Callers receive a shared_ptr to the entry, so eviction never frees a
// buffer a caller still holds -- an evicted-but-referenced entry simply lives
// until its last user drops it. This decouples the cache's budget from the
// consumers' lifetimes, which is required because -- unlike the ROCr-resident
// prototype -- this module cannot observe loaded-executable lifetimes.
//
// This module is built with -fno-exceptions / -fno-rtti (llvm_update_compile
// _flags), so it must not rely on C++ exceptions for control flow.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_MEM_CACHE_H
#define HOTSWAP_TRANSPILER_MEM_CACHE_H

#include "pipeline.h"
#include "translation-cache.h"

#include "llvm/Support/MemoryBuffer.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace COMGR::hotswap {

// Attribution carried alongside the cached HSACO. Mirrors the sidecar fields
// PipelineResult persists, minus the (moved-out) Hsaco buffer and the
// per-run timing block, which are meaningless for a cache hit.
struct MemCacheAttribution {
  int C5SuppressedCount = 0;
  std::string C5SuppressionReason;
  bool UsesScratchPrivateSegment = false;
  uint32_t SourcePrivateSegmentFixedSize = 0;
  bool TargetEnablePrivateSegment = false;
  uint32_t TargetPrivateSegmentFixedSize = 0;
  int LiftedCount = 0;
  int TotalCount = 0;
  unsigned ScaledDispatchFactor = 1;
};

// An immutable cached translation. Shared by every hit/coalesced caller; kept
// alive by the map (strong) and any outstanding caller references.
struct MemCacheEntry {
  std::shared_ptr<llvm::MemoryBuffer> Hsaco;
  MemCacheAttribution Attribution;
  size_t Bytes = 0; // == Hsaco->getBufferSize(), cached for budget accounting.
};

using MemCacheEntryRef = std::shared_ptr<const MemCacheEntry>;

enum class MemCacheStatus {
  Disabled,  // budget == 0; producer ran, nothing cached.
  Hit,       // served from the in-memory map (no producer call).
  Coalesced, // an identical in-flight request produced this; no producer call.
  Computed,  // this caller was the single-flight leader; producer ran.
  ProducerFailed, // producer returned no HSACO; nothing cached.
};

struct MemCacheResult {
  MemCacheStatus Status = MemCacheStatus::Disabled;
  MemCacheEntryRef Entry; // null iff ProducerFailed.
  // The full producer output for the leader path (Computed). Empty/moved-from
  // for Hit/Coalesced (those never ran the producer). Lets the leader's caller
  // recover pipeline timings / failure detail that the shared entry omits.
  // For Hit/Coalesced, reconstruct what the caller needs from Entry.
  PipelineResult LeaderResult;
};

// Producer computes a translation on a cold miss. It is invoked by exactly one
// thread (the single-flight leader) with NO cache lock held, so it may block on
// disk I/O / COMGR without serializing the cache. It must be self-contained
// (typically: disk-cache lookup, else run the transpile pipeline, else write
// disk). A producer returning a PipelineResult whose Hsaco is null or whose
// Success is false is treated as a failure and not cached.
using TranslationProducer = std::function<PipelineResult()>;

// The single entry point. Looks up `request` in the in-memory tier; on a miss,
// runs `producer` exactly once across all concurrent identical requests and
// caches the result. `request` must carry the same key-determining fields the
// disk tier uses so the two tiers agree on identity.
MemCacheResult getOrComputeTranslation(const TranslationCacheRequest &request,
                                       const TranslationProducer &producer);

// Observability snapshot. All counters are monotonic since process start
// except the live_* gauges.
struct MemCacheMetrics {
  uint64_t Lookups = 0;
  uint64_t Hits = 0;
  uint64_t Coalesced = 0;
  uint64_t Computed = 0;
  uint64_t ProducerCalls = 0;
  uint64_t ProducerFailures = 0;
  uint64_t Evictions = 0;
  uint64_t ReentrantComputes = 0;
  uint64_t BudgetBytes = 0;
  uint64_t LiveBytes = 0;
  uint64_t PeakLiveBytes = 0;
  size_t Entries = 0;
  size_t InFlight = 0;
};

MemCacheMetrics snapshotMemCacheMetrics();

// Effective byte budget for this process, resolved once from
// AMD_COMGR_HOTSWAP_MEM_CACHE_BYTES (0 disables the tier; unset uses the
// built-in default). Exposed for tests and diagnostics.
size_t memCacheBudgetBytes();

#ifdef COMGR_HOTSWAP_MEM_CACHE_TESTING
// Test-only controls. Not part of the public surface.
void resetMemCacheForTesting(size_t budgetBytesOverride);
size_t memCacheEntryCountForTesting();
size_t memCacheInFlightCountForTesting();
// Blocks until `count` waiters are parked on the in-flight entry for `request`,
// or the deadline elapses. Returns the observed waiter count. Lets a test pin
// the coalescing race deterministically.
size_t waitForMemCacheWaitersForTesting(const TranslationCacheRequest &request,
                                        size_t count, unsigned timeoutMs);
#endif

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_MEM_CACHE_H
