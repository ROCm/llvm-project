//===- comgr-hotswap-internal.h - HotSwap internal types and declarations -===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal header for the HotSwap ISA rewriting subsystem. Shared by all
/// comgr-hotswap-*.cpp compilation units. Not part of the public COMGR API.
///
/// Module structure:
///   comgr-hotswap-elf.cpp       ELF parsing, binary helpers, trampoline growth
///   comgr-hotswap-llvm.cpp      LLVM MC infrastructure (disasm/asm/encode)
///   comgr-hotswap-b0a0.cpp      GFX1250 B0-to-A0 policy + public API
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_INTERNAL_H
#define COMGR_HOTSWAP_INTERNAL_H

#include "amd_comgr.h"
#include "comgr-env.h"
#include "comgr.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR {
namespace hotswap {

// -- Logging ------------------------------------------------------------------
//
// Single output stream for all hotswap diagnostics (errors, warnings, and
// verbose traces). Returns llvm::errs() if AMD_COMGR_EMIT_VERBOSE_LOGS is set
// (via COMGR::env::shouldEmitVerboseLogs()) and llvm::nulls() otherwise, so
// hotswap output stays quiet in normal use but callers can opt in to the full
// diagnostic trail without relinking. Every function that returns a null /
// empty / failure result should emit here with a `"hotswap: error: ..."` or
// `"hotswap: ..."` prefix so the failure path is traceable.

inline llvm::raw_ostream &log() {
  return COMGR::env::shouldEmitVerboseLogs() ? llvm::errs() : llvm::nulls();
}

inline std::optional<uint64_t> checkedAddUint64(uint64_t LHS, uint64_t RHS,
                                                llvm::StringRef Context) {
  std::optional<uint64_t> Result = llvm::checkedAddUnsigned(LHS, RHS);
  if (Result)
    return Result;

  log() << "hotswap: error: " << Context << " overflows uint64_t.\n";
  return std::nullopt;
}

inline std::optional<uint64_t> checkedSubUint64(uint64_t LHS, uint64_t RHS,
                                                llvm::StringRef Context) {
  if (LHS < RHS) {
    log() << "hotswap: error: " << Context << " underflows uint64_t.\n";
    return std::nullopt;
  }
  return LHS - RHS;
}

// -- HotSwap rewrite profiling -----------------------------------------------
//
// Two-level gate:
//   1. Compile time -- the whole facility is compiled in only when the build
//      defines ENABLE_HOTSWAP_PROFILE (e.g. cmake -DENABLE_HOTSWAP_PROFILE=ON,
//      which passes -DENABLE_HOTSWAP_PROFILE to the compiler). In a normal
//      build every type below collapses to the no-op stubs in the #else
//      branch, so there is no code, no static state, and no per-call overhead.
//   2. Run time  -- when compiled in, it stays dormant until HOTSWAP_PROFILE is
//      set (and not "0"); otherwise every hook is a no-op and, crucially, no
//      clock is ever read (the Scope constructor only samples the clock when
//      the owning session is enabled).
//
// Design: a typed, per-rewrite session (HotswapProfile) accumulates wall-clock
// samples into a fixed std::array indexed by HotswapMetric -- no string keys,
// no map lookups, and no mutex on the hot path. Each retargetCodeObject() call
// owns one session (via HotswapProfileSession), threads it through PatchContext
// so deep patch sites record with a single typed call, and merges the whole
// array into the process-wide HotswapProfileSink exactly once at rewrite end
// under a single lock. The sink aggregates across every code object rewritten
// in the process and dumps to stderr at process exit.
//
// Row families (see HotswapMetric / the metricInfo() table for the full list):
//   phase:*  coarse pipeline stages in retargetCodeObject (elf_parse, initLLVM,
//            decode, b0a0_dispatch, grow_elf, ...). These partition
//            phase:rewrite_total; phase:unaccounted (computed at dump time)
//            absorbs the untimed remainder so the phases sum to the total.
//   strat:*  the B0-to-A0 patch strategies (inplace, trampoline, wmma_*,
//            scratch, ...). A strategy with sub-rules reports a parent total
//            plus indented children (e.g. strat:trampoline/ds_2addr).
//   jump:*   trampoline placement outcomes (nop_sled, short_s_branch,
//            far_long_s_add_pc, declined_far); the "calls" column is the count.

// Typed identifiers for every timed / counted bucket. HotswapProfile stores one
// HotswapSample per value; enum order is the dump order, and children are
// printed indented under the preceding parent (see metricInfo()). Keep \c Count
// last: it is the array size, not a real metric.
enum class HotswapMetric : uint8_t {
  // phase:* -- coarse pipeline stages; these partition RewriteTotal.
  RewriteTotal,
  ElfParse,
  InitLLVM,
  Decode,
  B0A0Dispatch,
  NopSledScan, // child of B0A0Dispatch
  CfgBuild,    // child of B0A0Dispatch
  Liveness,    // child of B0A0Dispatch
  FixupTrampolines,
  EntryTrampolines,
  GrowElf,
  DebugSections,
  KdRewrite,
  ScratchVerify,
  Unaccounted, // synthetic: RewriteTotal - sum(other phases), set at dump time
  // strat:* -- B0-to-A0 patch strategies (parents followed by their children).
  InPlace,
  InPlaceSClause,     // child of InPlace
  InPlaceClusterLoad, // child of InPlace
  InPlaceSBarrier,    // child of InPlace
  Trampoline,
  TrampolineDs2Addr,     // child of Trampoline
  TrampolineTensor,      // child of Trampoline
  TrampolineClusterLoad, // child of Trampoline
  TrampolineAddtid,      // child of Trampoline
  WmmaSplit,
  Scratch,
  WmmaScale16,
  WmmaHazard,
  Vop3px2Src2,
  // jump:* -- trampoline placement outcomes; the "calls" column is the count.
  JumpNopSled,
  JumpShort,
  JumpLong,
  JumpDeclined,
  Count
};

// One accumulated bucket. Nanos / Calls / Patches are running sums; MinNanos /
// MaxNanos bound a single timed scope's duration (kept for the richer dump).
struct HotswapSample {
  uint64_t Nanos = 0;
  uint64_t Calls = 0;
  uint64_t Patches = 0;
  uint64_t MinNanos = std::numeric_limits<uint64_t>::max();
  uint64_t MaxNanos = 0;
};

#ifdef ENABLE_HOTSWAP_PROFILE

inline uint64_t profNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Static display metadata for one metric.
struct HotswapMetricInfo {
  const char *Name;     // dump label, e.g. "phase:elf_parse"
  uint8_t Indent;       // 0 = top-level row, 1 = child (printed indented)
  bool PartitionsTotal; // true for the coarse phases that sum to RewriteTotal
};

inline const HotswapMetricInfo &metricInfo(HotswapMetric M) {
  static constexpr HotswapMetricInfo Table[] = {
      {"phase:rewrite_total", 0, false},
      {"phase:elf_parse", 0, true},
      {"phase:initLLVM", 0, true},
      {"phase:decode", 0, true},
      {"phase:b0a0_dispatch", 0, true},
      {"b0a0:nop_sled_scan", 1, false},
      {"b0a0:cfg_build", 1, false},
      {"b0a0:liveness", 1, false},
      {"phase:fixup_trampolines", 0, true},
      {"phase:entry_trampolines", 0, true},
      {"phase:grow_elf", 0, true},
      {"phase:debug_sections", 0, true},
      {"phase:kd_rewrite", 0, true},
      {"phase:scratch_verify", 0, true},
      {"phase:unaccounted", 0, false},
      {"strat:inplace", 0, false},
      {"strat:inplace/s_clause", 1, false},
      {"strat:inplace/cluster_load", 1, false},
      {"strat:inplace/s_barrier", 1, false},
      {"strat:trampoline", 0, false},
      {"strat:trampoline/ds_2addr", 1, false},
      {"strat:trampoline/tensor_tdm", 1, false},
      {"strat:trampoline/cluster_load", 1, false},
      {"strat:trampoline/addtid", 1, false},
      {"strat:wmma_split", 0, false},
      {"strat:scratch_fp8", 0, false},
      {"strat:wmma_scale16", 0, false},
      {"strat:wmma_hazard", 0, false},
      {"strat:vop3px2_src2", 0, false},
      {"jump:nop_sled", 0, false},
      {"jump:short_s_branch", 0, false},
      {"jump:far_long_s_add_pc", 0, false},
      {"jump:declined_far", 0, false},
  };
  static_assert(sizeof(Table) / sizeof(Table[0]) ==
                    static_cast<size_t>(HotswapMetric::Count),
                "metricInfo table must have one row per HotswapMetric");
  return Table[static_cast<size_t>(M)];
}

// Process-wide aggregator. One instance (a function-local static in get())
// accumulates the per-rewrite sessions merged into it and dumps at process
// exit. All mutation goes through merge(), which takes the lock exactly once
// per rewrite -- the hot path never touches it.
class HotswapProfileSink {
public:
  static constexpr size_t NumMetrics =
      static_cast<size_t>(HotswapMetric::Count);

  static HotswapProfileSink &get() {
    static HotswapProfileSink Instance;
    return Instance;
  }

  void merge(const std::array<HotswapSample, NumMetrics> &Samples) {
    std::scoped_lock Lock(Mtx);
    for (size_t I = 0; I < NumMetrics; ++I) {
      const HotswapSample &S = Samples[I];
      if (S.Calls == 0 && S.Nanos == 0)
        continue;
      HotswapSample &D = Totals[I];
      D.Nanos += S.Nanos;
      D.Calls += S.Calls;
      D.Patches += S.Patches;
      if (S.MinNanos != std::numeric_limits<uint64_t>::max())
        D.MinNanos = std::min(D.MinNanos, S.MinNanos);
      D.MaxNanos = std::max(D.MaxNanos, S.MaxNanos);
      HasData = true;
    }
  }

  // Read access to an accumulated bucket (used by the profile unit tests).
  const HotswapSample &total(HotswapMetric M) const {
    return Totals[static_cast<size_t>(M)];
  }

  ~HotswapProfileSink() { dump(); }

private:
  HotswapProfileSink() = default;

  void printRow(llvm::StringRef Display, const HotswapSample &S,
                unsigned Indent) const {
    std::string Label = std::string(Indent * 2, ' ') + Display.str();
    const double TotalUs = S.Nanos / 1000.0;
    const double AvgUs = S.Calls ? TotalUs / S.Calls : 0.0;
    const double MinUs = S.MinNanos == std::numeric_limits<uint64_t>::max()
                             ? 0.0
                             : S.MinNanos / 1000.0;
    const double MaxUs = S.MaxNanos / 1000.0;
    fprintf(stderr, "%-28s %8llu %12.1f %11.3f %11.3f %11.3f %9llu\n",
            Label.c_str(), static_cast<unsigned long long>(S.Calls), TotalUs,
            AvgUs, MinUs, MaxUs, static_cast<unsigned long long>(S.Patches));
  }

  void dump() {
    if (!HasData)
      return;
    // #4: make the phases partition rewrite_total. Everything not covered by a
    // timed phase (input copy, pool/guard setup, symbol insertion, no-growth
    // copy, failure paths) lands in phase:unaccounted so the phase rows sum to
    // the total.
    const uint64_t TotalNanos =
        Totals[static_cast<size_t>(HotswapMetric::RewriteTotal)].Nanos;
    uint64_t PhaseSum = 0;
    for (size_t I = 0; I < NumMetrics; ++I)
      if (metricInfo(static_cast<HotswapMetric>(I)).PartitionsTotal)
        PhaseSum += Totals[I].Nanos;
    if (TotalNanos > PhaseSum) {
      HotswapSample &U =
          Totals[static_cast<size_t>(HotswapMetric::Unaccounted)];
      U.Nanos = TotalNanos - PhaseSum;
      U.Calls = std::max<uint64_t>(U.Calls, 1);
    }

    fprintf(stderr,
            "\n=== HotSwap COMGR rewrite profile (HOTSWAP_PROFILE) ===\n");
    fprintf(stderr, "%-28s %8s %12s %11s %11s %11s %9s\n", "name", "calls",
            "total_us", "avg_us", "min_us", "max_us", "patches");
    for (size_t I = 0; I < NumMetrics; ++I) {
      const HotswapSample &S = Totals[I];
      if (S.Calls == 0 && S.Nanos == 0)
        continue;
      const HotswapMetricInfo &Info = metricInfo(static_cast<HotswapMetric>(I));
      printRow(Info.Name, S, Info.Indent);
    }
    fprintf(stderr,
            "======================================================\n");
  }

  std::mutex Mtx;
  std::array<HotswapSample, NumMetrics> Totals{};
  bool HasData = false;
};

// Per-rewrite session. Accumulates locally into Samples (no lock, no string
// lookup) and, when enabled, merges into the process-wide sink once via
// merge(). The runtime gate (HOTSWAP_PROFILE) is read once here and consulted
// by Scope so a disabled session never reads the clock.
class HotswapProfile {
public:
  static constexpr size_t NumMetrics =
      static_cast<size_t>(HotswapMetric::Count);

  HotswapProfile() {
    const char *V = getenv("HOTSWAP_PROFILE");
    Enabled = V && V[0] != '\0' && llvm::StringRef(V) != "0";
  }
  HotswapProfile(const HotswapProfile &) = delete;
  HotswapProfile &operator=(const HotswapProfile &) = delete;

  bool enabled() const { return Enabled; }

  // RAII wall-clock recorder for one metric. The constructor samples the clock
  // only when the owning session is enabled (this centralizes the runtime
  // gate), so a disabled session reads no clock at all. Records on destruction,
  // including any patches attributed via addPatches().
  class Scope {
  public:
    Scope(HotswapProfile *Profile, HotswapMetric Metric)
        : Profile(Profile), Metric(Metric),
          StartNs(Profile && Profile->Enabled ? profNowNs() : 0) {}
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    Scope(Scope &&) = delete;
    Scope &operator=(Scope &&) = delete;
    ~Scope() {
      if (!Profile || !Profile->Enabled)
        return;
      const uint64_t Elapsed = profNowNs() - StartNs;
      HotswapSample &S = Profile->Samples[static_cast<size_t>(Metric)];
      S.Nanos += Elapsed;
      S.Calls += 1;
      S.Patches += Patches;
      S.MinNanos = std::min(S.MinNanos, Elapsed);
      S.MaxNanos = std::max(S.MaxNanos, Elapsed);
    }

    // Attribute \p N applied patches to this scope's bucket.
    void addPatches(uint64_t N) { Patches += N; }

  private:
    HotswapProfile *Profile;
    HotswapMetric Metric;
    uint64_t StartNs;
    uint64_t Patches = 0;
  };

  // Time the enclosing scope under \p Metric.
  Scope time(HotswapMetric Metric) { return Scope(this, Metric); }

  // Count \p N occurrences of \p Metric with no timing (used by the count-only
  // jump:* outcome tallies). No-op when the session is disabled.
  void count(HotswapMetric Metric, uint64_t N = 1) {
    if (!Enabled)
      return;
    Samples[static_cast<size_t>(Metric)].Calls += N;
  }

  // Merge this session's samples into the process-wide sink under a single
  // lock. No-op when disabled.
  void merge() {
    if (!Enabled)
      return;
    HotswapProfileSink::get().merge(Samples);
  }

  // Read access to a locally accumulated bucket (used by the profile tests).
  const HotswapSample &sample(HotswapMetric Metric) const {
    return Samples[static_cast<size_t>(Metric)];
  }

private:
  bool Enabled = false;
  std::array<HotswapSample, NumMetrics> Samples{};
};

// Per-rewrite RAII: opens the RewriteTotal timer on construction and, on
// destruction, closes it and merges the session into the process-wide sink
// exactly once. Member declaration order makes destruction close the timer
// (Total) before the merge (MergeAtExit) runs.
class HotswapProfileSession {
public:
  HotswapProfileSession() : Total(Profile.time(HotswapMetric::RewriteTotal)) {}

  HotswapProfile &profile() { return Profile; }

private:
  struct Merger {
    HotswapProfile &Profile;
    explicit Merger(HotswapProfile &Profile) : Profile(Profile) {}
    ~Merger() { Profile.merge(); }
  };

  HotswapProfile Profile;
  Merger MergeAtExit{Profile};
  HotswapProfile::Scope Total;
};

#else // !ENABLE_HOTSWAP_PROFILE

// Profiling compiled out. These no-op shims keep every hotswap-*.cpp call site
// valid at zero cost: enabled() is always false, time() yields an inert Scope,
// and count() / merge() vanish. No static state, no atexit dump, no per-call
// work, and -- since the Scope constructor takes no clock -- no timing calls.

class HotswapProfile {
public:
  class Scope {
  public:
    Scope() = default;
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    // User-declared (empty) destructor so `Scope S = time(...)` locals are
    // treated as RAII guards and not flagged -Wunused-variable in the
    // compiled-out build; still trivially inlined away at zero cost.
    ~Scope() {}
    void addPatches(uint64_t) {}
  };

  bool enabled() const { return false; }
  Scope time(HotswapMetric) { return Scope(); }
  void count(HotswapMetric, uint64_t = 1) {}
  void merge() {}
};

class HotswapProfileSession {
public:
  HotswapProfile &profile() { return Profile; }

private:
  HotswapProfile Profile;
};

#endif // ENABLE_HOTSWAP_PROFILE

// -- Trampoline and NOP sled --------------------------------------------------

struct Trampoline {
  uint64_t OriginalOffset = 0;
  uint32_t OriginalSize = 0;
  llvm::SmallVector<uint8_t> Bytes;
  // When set, both edges use an s_add_pc_i64 long branch instead of s_branch
  // (reaches anywhere, no scratch reg, no SCC). Set when the appended pool is
  // beyond s_branch's +-128 KB reach; widens the reserved branch-back slot.
  bool Long = false;
  // The branch-back is already present at the end of Bytes. Used by required
  // far patches whose backward edge cannot use s_add_pc_i64 on gfx1250 A0.
  bool PreEncodedBack = false;
};

// Kernel-entry stubs are appended as normal .text growth. Keep each entry on
// the same 256-byte alignment expected by AMDGPU kernel descriptors.
static constexpr uint64_t KernelEntryStubStride = 256;
static constexpr uint64_t KernelEntryInstPrefUnitBytes = 128;
static_assert(KernelEntryStubStride % KernelEntryInstPrefUnitBytes == 0,
              "entry-stub stride must be an integral prefetch span");
static constexpr uint32_t KernelEntryStubInstPrefLines =
    KernelEntryStubStride / KernelEntryInstPrefUnitBytes;

struct KernelDescriptorInfo {
  std::string KernelName;
  uint64_t VAddr = 0;
  int64_t EntryOffset = 0;
};

struct KernelClusterDims {
  unsigned X = 0;
  unsigned Y = 0;
  unsigned Z = 0;
};

struct NopSled {
  uint64_t Start = 0;
  uint64_t End = 0;
  uint64_t WritePos = 0;
  uint64_t FunctionStart = 0;
  uint64_t FunctionEnd = 0;
};

enum class MaskWorkaroundPolicy {
  None,
  A0,
  B0,
};

// -- Rewrite rule -------------------------------------------------------------

struct RewriteRule {
  std::string ReplaceMnemonic;
  llvm::SmallVector<uint8_t> ReplaceBytes;
};

// -- Named constants ----------------------------------------------------------

// Kernel descriptor size from upstream AMDHSAKernelDescriptor.h. Field
// offsets are resolved via offsetof(amdhsa::kernel_descriptor_t, field)
// at the access site so the struct definition stays the single source
// of truth and the *_OFFSET constants do not get spelled out twice.
static constexpr uint64_t KdSize = sizeof(llvm::amdhsa::kernel_descriptor_t);

// Maximum distance (bytes) between an instruction and a NOP sled for the
// sled to be considered reachable by a single s_branch.
static constexpr uint64_t MaxSledDistance = 131072;

// Minimum size (bytes) of a consecutive NOP run to be usable as a sled.
static constexpr uint64_t MinNopSledSize = 8;

// Minimum AMDGPU instruction size (one dword).
static constexpr uint32_t MinInstSize = 4;

// s_add_pc_i64 long-branch encoded sizes: 8 bytes for a forward (32-bit
// literal) offset, 12 for a backward (64-bit literal) one. The back slot
// reserves the max; unused tail bytes are s_nop-padded. emitToTrampoline picks
// the long path only when a short s_branch cannot reach the site's exact pool
// offset on either edge (computed from the already-queued trampolines).
static constexpr uint32_t LongBranchFwdBytes = 8;
static constexpr uint32_t LongBranchMaxBytes = 12;

// s_branch encoding: 16-bit signed dword offset field bounds. Used by
// LLVMState::encodeSBranch to reject out-of-range branches before handing
// them to MCCodeEmitter.
static constexpr int64_t BranchOffsetMin = -32768;
static constexpr int64_t BranchOffsetMax = 32767;

// MCInst operand layout for ds_load_addtid_b32 / ds_store_addtid_b32. Shared
// between the trampoline patch (comgr-hotswap-patch-trampoline.cpp) and the
// unit tests that pin the layout (HotswapMCTest.cpp) so a tablegen change
// upstream is caught in one place.
//   operand 0: vdst (load) / data0 (store) -- VGPR register
//   operand 1: combined offset             -- immediate
//   operand 2: gds                         -- immediate (0 = LDS, 1 = GDS)
static constexpr unsigned AddtidOpReg = 0;
static constexpr unsigned AddtidOpOffset = 1;
static constexpr unsigned AddtidOpGds = 2;

// -- ElfView ------------------------------------------------------------------
//
// Thin wrapper around llvm::object::ELFFile<ELF64LE> that owns the structural
// view of a mutable code-object buffer. The caller retains ownership of the
// bytes; ElfView exposes LLVM's ELF iterators through member methods and
// caches the .text section lookup.

class ElfView {
public:
  using ELFT = llvm::object::ELF64LE;
  using ELFFileT = llvm::object::ELFFile<ELFT>;

  struct FunctionTextRange {
    uint64_t Begin = 0;
    uint64_t End = 0;
    const ELFT::Sym *Symbol = nullptr;
    const ELFT::Shdr *Symtab = nullptr;
  };

  /// Parse \p Data / \p Size into an ElfView. Fails if the bytes are not a
  /// valid ELF64 or if no `.text` section is found.
  static llvm::Expected<ElfView> create(uint8_t *Data, size_t Size);

  ElfView(ElfView &&) = default;
  ElfView &operator=(ElfView &&) = default;
  ElfView(const ElfView &) = delete;
  ElfView &operator=(const ElfView &) = delete;

  const ELFFileT &file() const { return File; }
  size_t size() const { return File.getBufSize(); }

  /// Writable view of the underlying bytes. The caller that constructed this
  /// ElfView via `create(uint8_t *, size_t)` retains ownership of the buffer;
  /// ElfView just exposes a typed, mutable alias onto `ELFFile::base()`. Safe
  /// because the factory was handed a `uint8_t *` and the buffer outlives
  /// this ElfView.
  uint8_t *data() { return const_cast<uint8_t *>(File.base()); }
  const uint8_t *data() const { return File.base(); }

  /// Section header range, cached at construction time. The underlying
  /// storage is the file buffer, which lives at least as long as this
  /// ElfView, so the range is always valid to iterate.
  ELFT::ShdrRange sections() const { return Sections; }

  /// Return the cached `.text` section header. Never null for a successfully
  /// constructed ElfView.
  const ELFT::Shdr *textSection() const { return TextSection; }

  uint64_t textOffset() const { return TextSection->sh_offset; }
  uint64_t textSize() const { return TextSection->sh_size; }
  uint64_t textAddr() const { return TextSection->sh_addr; }

  /// Index of the `.text` section in the section header table.
  unsigned textSectionIndex() const { return TextSectionIndex; }

  /// Pointer into the buffer for the first byte of `.text`.
  uint8_t *textData() { return data() + textOffset(); }
  const uint8_t *textData() const { return data() + textOffset(); }

  /// Enumerate function symbol ranges in `.text` using virtual addresses.
  /// Zero-size symbols extend to the next function symbol or `.text` end.
  std::vector<FunctionTextRange> functionTextRanges() const;

  /// Find the kernel function symbol whose range includes \p TextAddress.
  /// Returns "" if no matching function symbol exists.
  std::string findKernelAtAddress(uint64_t TextAddress) const;

  /// Find the section-relative `.text` range of the function containing
  /// \p TextOffset, or std::nullopt if no sized function symbol covers it.
  std::optional<FunctionTextRange>
  findFunctionTextRangeAtOffset(uint64_t TextOffset) const;

  /// Return a pointer to \p Len bytes at virtual address \p VAddr, resolved
  /// through the allocatable section that contains it (any section, not just
  /// `.text` -- e.g. the appended trampoline pool). Returns nullptr if no
  /// section covers the range or it falls outside the buffer.
  const uint8_t *dataAtVAddr(uint64_t VAddr, uint64_t Len) const;

  /// Pointer to the kernel_descriptor for \p KernelName inside the buffer,
  /// or nullptr if not found.
  uint8_t *findKernelDescriptor(llvm::StringRef KernelName);

  /// Enumerate kernel descriptor symbols named "<kernel>.kd" and read their
  /// current kernel_code_entry_byte_offset values.
  std::vector<KernelDescriptorInfo> kernelDescriptors() const;

  /// Return the virtual address of the kernel descriptor symbol for
  /// \p KernelName, or std::nullopt when the descriptor is not present.
  std::optional<uint64_t>
  getKernelDescriptorVAddr(llvm::StringRef KernelName) const;

  /// Rewrite kernel_code_entry_byte_offset for \p KernelName.
  bool updateKernelDescriptorEntryOffset(llvm::StringRef KernelName,
                                         int64_t NewEntryOffset);

  /// Ensure kernel metadata reserves at least \p RequiredSgprs SGPRs. When
  /// \p UpdateDescriptor is true, also update the pre-gfx10 kernel descriptor
  /// field; it is reserved and must remain unchanged on gfx10+.
  bool updateKernelDescriptorSgprCount(llvm::StringRef KernelName,
                                       unsigned RequiredSgprs,
                                       bool UpdateDescriptor = true);

  /// Read COMPUTE_PGM_RSRC3.INST_PREF_SIZE for \p KernelName.
  std::optional<uint32_t>
  getKernelDescriptorInstPrefSize(llvm::StringRef KernelName,
                                  llvm::StringRef TargetCpu) const;

  /// Rewrite COMPUTE_PGM_RSRC3.INST_PREF_SIZE for \p KernelName.
  bool updateKernelDescriptorInstPrefSize(llvm::StringRef KernelName,
                                          llvm::StringRef TargetCpu,
                                          uint32_t InstPrefLines);

  /// Read the VGPR count from the kernel descriptor for \p KernelName.
  /// Returns std::nullopt if the descriptor is not found.
  std::optional<unsigned> getKernelVgprCount(llvm::StringRef KernelName,
                                             unsigned VgprGranuleSize) const;

  /// Read `group_segment_fixed_size` from the kernel descriptor for
  /// \p KernelName, i.e. the **static** (compile-time-fixed) LDS allocation
  /// per work-group in bytes. Returns std::nullopt if the descriptor symbol
  /// is missing.
  ///
  /// This is the only LDS quantity visible in the ELF. Dynamic LDS is
  /// allocated by the host at dispatch time (carried in the AQL packet's
  /// `group_segment_size` and propagated to the device via the
  /// `hidden_dynamic_lds_size` kernarg) and is *not* included here, so the
  /// returned value is a lower bound on the total LDS the kernel may
  /// touch. Callers that need to flag potential overflow of gfx1250 A0's
  /// 16-bit M0 limit can use this as a "definitely exceeds"
  /// check; "static fits, dynamic pushes over" cannot be detected
  /// statically. See AMDGPUUsage "Code Object V3 Kernel Descriptor"
  /// (GROUP_SEGMENT_FIXED_SIZE).
  std::optional<uint32_t>
  getKernelStaticLdsSize(llvm::StringRef KernelName) const;

  /// Read the SGPR count for \p KernelName from the \c amdhsa.kernels
  /// msgpack metadata note (\c .sgpr_count key), falling back to the kernel
  /// descriptor when the metadata note is absent. On GFX10+ the kernel
  /// descriptor's \c GRANULATED_WAVEFRONT_SGPR_COUNT is architecturally
  /// reserved, so metadata is the only reliable source when present.
  /// Returns std::nullopt if the matching metadata is malformed, the kernel is
  /// missing from present metadata, or the descriptor fallback is unavailable.
  std::optional<unsigned> getKernelSgprCount(llvm::StringRef KernelName) const;

  /// Read fixed \c .cluster_dims metadata for \p KernelName when present.
  /// Returns std::nullopt when the metadata note or key is absent, or when the
  /// matching metadata is malformed.
  std::optional<KernelClusterDims>
  getKernelClusterDims(llvm::StringRef KernelName) const;

  /// Update the RSRC1 VGPR granule count in the kernel descriptor for
  /// \p KernelName by adding \p ExtraVgprs. The SGPR granule field is
  /// not updated because it is reserved on GFX10+.
  void updateKernelDescriptor(llvm::StringRef KernelName, unsigned ExtraVgprs,
                              unsigned VgprGranuleSize);

  /// Virtual address at which growWithTrampolines appends the trampoline pool:
  /// the first page-aligned address above every existing allocatable section.
  /// Callers that pre-compute branch/stub targets (B0-to-A0 trampolines,
  /// kernel-entry stubs) must resolve pool positions against this value so the
  /// baked branches land on the pool's final location. Single source of truth
  /// shared with growWithTrampolines. std::nullopt on sh_addr+sh_size overflow.
  std::optional<uint64_t> trampolinePoolVAddr() const;

  /// Grow the ELF by appending the trampoline pool at a fresh virtual address
  /// (trampolinePoolVAddr()) in a new PT_LOAD segment, leaving every existing
  /// section, symbol, and segment in place. Returns a null unique_ptr on
  /// failure.
  ///
  /// Appending (rather than growing `.text` and shifting everything after it)
  /// preserves the absolute/PC-relative addresses baked into a fully-linked
  /// AMDGPU code object, which carries no relocations to fix up.
  std::unique_ptr<llvm::WritableMemoryBuffer>
  growWithTrampolines(llvm::ArrayRef<Trampoline> Trampolines,
                      llvm::ArrayRef<uint8_t> SNopBytes) const;

private:
  ElfView(ELFFileT File, ELFT::ShdrRange Sections,
          const ELFT::Shdr *TextSection, unsigned TextSectionIndex)
      : File(std::move(File)), Sections(Sections), TextSection(TextSection),
        TextSectionIndex(TextSectionIndex) {}

  ELFFileT File;
  ELFT::ShdrRange Sections;
  const ELFT::Shdr *TextSection;
  unsigned TextSectionIndex;
};

// -- Free-function ELF helpers (no ELF state required) ------------------------

/// Overwrite instruction bytes at \p InstOffset with \p Rule.ReplaceBytes,
/// padding remaining bytes with s_nop instructions sourced from \p
/// LS.SNopBytes. Returns false on bounds violation or if \p LS has no cached
/// s_nop encoding.
struct LLVMState;
[[nodiscard]] bool applyByteReplace(const RewriteRule &Rule,
                                    uint64_t InstOffset, uint32_t InstSize,
                                    uint8_t *Text, uint64_t TextSize,
                                    const LLVMState &LS);

/// Find the nearest NOP sled to \p Offset with at least \p Needed bytes of
/// free space. Returns nullptr if none found within MaxSledDistance.
NopSled *findNearestSled(std::vector<NopSled> &Sleds, uint64_t Offset,
                         uint64_t Needed);

// -- RewriteConfig ------------------------------------------------------------
//
// ISA-specific parameters that drive the generic rewriting infrastructure.
// Constructed by the policy layer (e.g. GFX1250 B0-to-A0 in PR #2203) and
// threaded through the MC helpers (buildTrampoline below) and the policy
// PatchContext so infrastructure has zero ISA assumptions.
//
// Instruction-encoding bits (s_branch / s_nop opcodes) are deliberately NOT
// members of this struct -- they are derived from the MC layer at initLLVM()
// time and exposed via LLVMState (SBranchOpcode, SNopBytes plus the
// encodeSBranch method), so the policy layer never has to hardcode target
// opcode values.

struct RewriteConfig {
  std::string SourceIsa;
  std::string TargetIsa;
  std::string TargetCpu;
  unsigned MaxVgprs = 0;
  unsigned MaxSgprs = 0;
  unsigned VgprGranuleSize = 0;
  bool RunB0A0Patches = true;
  MaskWorkaroundPolicy MaskPolicy = MaskWorkaroundPolicy::None;
};

// -- LLVM MC context ----------------------------------------------------------
//
// Bundle of per-ISA LLVM MC objects. Populated by initLLVM, consumed by the
// decode/encode helpers and by the downstream policy layer. Also caches a
// handful of AMDGPU instruction primitives (s_branch MC opcode, pre-encoded
// s_nop bytes) and exposes the encodeSBranch method -- this keeps all
// target-specific opcode knowledge inside the MC layer and off the policy /
// infrastructure layer.

struct LLVMState {
  const llvm::Target *Target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCObjectFileInfo> MOFI;
  std::unique_ptr<llvm::MCDisassembler> MCD;
  std::unique_ptr<llvm::MCInstPrinter> MCIP;
  std::unique_ptr<llvm::MCCodeEmitter> MCE;
  /// Target-provided branch / call / relocation analysis. May be null on
  /// targets that do not implement MCInstrAnalysis; callers must check
  /// before dispatching. Cached here so downstream patch passes can ask
  /// `MIA->isBranch(Inst)` / `isCall(Inst)` / `evaluateBranch(...)` instead
  /// of matching mnemonic strings.
  std::unique_ptr<llvm::MCInstrAnalysis> MIA;
  std::string Cpu;

  /// MC opcode index for `s_branch`, resolved once at initLLVM() via the
  /// asm parser. Used by encodeSBranch() below to construct a fresh MCInst
  /// per call.
  unsigned SBranchOpcode = 0;

  /// MC opcode index for `s_nop`. Resolved via the asm parser at initLLVM()
  /// time so decoded-stream consumers (e.g. buildNopSledMap) can match NOPs
  /// by opcode rather than mnemonic string.
  unsigned SNopOpcode = 0;

  /// Pre-encoded bytes for `s_nop 0` (MinInstSize bytes). Populated at
  /// initLLVM() time via MCCodeEmitter and used by applyByteReplace() and
  /// NOP-sled padding paths instead of a hardcoded encoding.
  llvm::SmallVector<uint8_t, 4> SNopBytes;

  /// Cached `v_nop` MCInst, resolved at initLLVM() time. Used by the WMMA
  /// co-execution hazard patch to build trampolines without string
  /// round-trips.
  llvm::MCInst VNopInst;

  /// MC opcodes for the kernel-entry stub sequence, resolved once at
  /// initLLVM() time by parsing representative asm snippets. The idempotency
  /// matcher compares decoded opcodes against these cached values instead of
  /// matching disassembled mnemonic strings.
  unsigned GlobalWbOpcode = 0;
  unsigned SGetPcI64Opcode = 0;
  unsigned SAddU32Opcode = 0;
  unsigned SAddcU32Opcode = 0;
  unsigned SSetPcI64Opcode = 0;

  bool Valid = false;

  /// Encode a relative `s_branch` from \p FromOffset to \p ToOffset and
  /// return the MinInstSize encoded bytes. Returns an empty vector if the
  /// delta is unaligned, out of the 16-bit signed dword range, or if this
  /// LLVMState is not valid / has no cached s_branch opcode. Uses
  /// MCCodeEmitter for the encoding so no hardcoded opcode bits appear in
  /// the hotswap code. Empty-on-failure matches the convention used by
  /// encodeMCInst() and assembleSingleInst() so the same idiom applies
  /// uniformly across the MC layer.
  [[nodiscard]] llvm::SmallVector<uint8_t>
  encodeSBranch(uint64_t FromOffset, uint64_t ToOffset) const;
};

// -- Decoded instruction ------------------------------------------------------

struct InternalDecodedInst {
  uint64_t Offset = 0;
  uint32_t Size = 0;
  llvm::MCInst Inst;
  std::string Mnemonic;
};

// -- Function declarations (LLVM MC layer) ------------------------------------

/// Initialize LLVM MC infrastructure for the AMDGPU subtarget described by
/// \p TI (produced by Comgr's parseTargetIdentifier). The triple is built
/// from TI.Arch/Vendor/OS/Environ and features are threaded through to
/// createMCSubtargetInfo so the MC layer sees the same subtarget view the
/// caller asked for. AMDGPU MC registration is delegated to
/// COMGR::ensureLLVMInitialized(); the amdgcn Target lookup itself is cached
/// in a thread-safe function-local static.
LLVMState initLLVM(const TargetIdentifier &TI);

/// Disassemble \p Text into \p Decoded using \p LS. Unknown bytes are encoded
/// as MinInstSize-sized entries with mnemonic "<unknown>".
[[nodiscard]] bool decodeTextSection(const uint8_t *Text, uint64_t TextSize,
                                     const LLVMState &LS,
                                     std::vector<InternalDecodedInst> &Decoded);

/// Assemble a single instruction string, returning its encoded bytes.
llvm::SmallVector<uint8_t> assembleSingleInst(llvm::StringRef AsmStr,
                                              const LLVMState &LS);

/// Join \p AsmLines into a single newline-terminated assembly source string,
/// as expected by assembleSingleInst (which accepts multiple instructions).
std::string joinAsmLines(llvm::ArrayRef<std::string> AsmLines);

/// Assemble \p AsmLines and append a branch-back to the next instruction
/// after the original (\p OriginalOffset + \p OriginalSize). The branch-back
/// is encoded via LLVMState::encodeSBranch, so no ISA-specific opcode needs
/// to flow in from the caller.
///
/// NOTE: no production caller remains (WMMA-split now defers edge encoding to
/// emitToTrampoline / fixupTrampolineBranches). Kept only as a self-contained
/// helper exercised by the unit tests; prefer emitToTrampoline for new code.
Trampoline buildTrampoline(llvm::ArrayRef<std::string> AsmLines,
                           uint64_t OriginalOffset, uint32_t OriginalSize,
                           uint64_t TrampolineTextOffset, const LLVMState &LS);

/// Overload that accepts pre-decoded MCInst instructions directly,
/// encoding them via MCCodeEmitter without a string round-trip.
Trampoline buildTrampoline(llvm::ArrayRef<llvm::MCInst> Insts,
                           uint64_t OriginalOffset, uint32_t OriginalSize,
                           uint64_t TrampolineTextOffset, const LLVMState &LS);

/// Return true iff any register operand of \p WmmaInst overlaps the
/// destination operand of \p ValuInst (for WMMA/VALU co-execution hazard
/// detection). Delegates aliasing to MCRegisterInfo::regsOverlap so
/// sub-registers and tuple aliases are handled without a manual range
/// computation.
bool checkVgprOverlap(const llvm::MCInst &WmmaInst,
                      const llvm::MCInst &ValuInst,
                      const llvm::MCRegisterInfo &MRI);

/// WMMA/SWMMAC A0 vs B0 v_nop spacing requirement.
struct WmmaNopReq {
  int A0Nops = 4;
  int B0Nops = 4;
};

/// Classify the A0/B0 v_nop requirement for a WMMA/SWMMAC mnemonic.
WmmaNopReq classifyWmmaNops(llvm::StringRef Mnemonic);

/// Patch the VOP3PX2 scale_src2 field (bits [58:50]) to VGPR0 encoding
/// (0x100) in a 16-byte instruction buffer. Returns true if the field
/// was modified (false if already set to the target value).
bool patchScaleSrc2(uint8_t *InstBytes);

// -- VGPR liveness types ------------------------------------------------------

/// Per-instruction def/use bitvectors over the VGPR index space. Populated by
/// getInstRegDefUse() during liveness analysis; each bit position corresponds
/// to one VGPR (index matches AMDGPU VGPR numbering, e.g. bit 5 = V5).
struct RegDefUse {
  llvm::BitVector Defs;
  llvm::BitVector Uses;
};

/// A basic block in the decoded-instruction CFG. Offsets are byte offsets
/// into .text; \c InstIndices stores positions in the flat Decoded vector;
/// \c Successors / \c Predecessors are indices into CFG::Blocks.
struct BasicBlock {
  uint64_t StartOffset = 0;
  uint64_t EndOffset = 0;
  llvm::SmallVector<size_t> InstIndices;
  llvm::SmallVector<unsigned> Successors;
  llvm::SmallVector<unsigned> Predecessors;
};

/// Control-flow graph over the decoded instruction stream. \c OffsetToBlock
/// is the inverted index mapping a .text byte offset to its owning block
/// index in \c Blocks, used to resolve branch-target / fall-through edges
/// during CFG construction.
struct CFG {
  std::vector<BasicBlock> Blocks;
  llvm::DenseMap<uint64_t, unsigned> OffsetToBlock;
};

/// Dataflow-liveness result for a kernel's VGPR set. \c LiveBefore[i] and
/// \c LiveAfter[i] are the live-in / live-out bitvectors for Decoded[i].
/// \c Converged is false when the iterative solver hit its iteration cap;
/// callers fall back to a conservative all-VGPRs-live analysis in that case.
struct LivenessInfo {
  std::vector<llvm::BitVector> LiveBefore;
  std::vector<llvm::BitVector> LiveAfter;
  bool Converged = false;
};

/// Allocates scratch VGPRs for a patch point, preferring to reuse dead slots
/// from the kernel's existing allocation before extending the allocation past
/// the kernel descriptor's reported VGPR count. Constructed per patch site
/// with the live-set at that site and the kernel's current / maximum VGPR
/// counts.
struct VgprAllocator {
  llvm::BitVector LiveAtPoint;
  unsigned KdAllocatedVgprs = 0;
  unsigned NextAboveKd = 0;
  unsigned MaxVgprs = 0;
  unsigned ExtraAllocated = 0;

  VgprAllocator(const llvm::BitVector &Live, unsigned KdVgprs, unsigned Max)
      : LiveAtPoint(Live), KdAllocatedVgprs(KdVgprs), NextAboveKd(KdVgprs),
        MaxVgprs(Max) {}

  /// Allocate one VGPR not currently marked live. Returns std::nullopt if
  /// the kernel's existing VGPR pool is saturated and there is no headroom
  /// below MaxVgprs for an additional allocation.
  std::optional<unsigned> alloc() {
    if (int V = LiveAtPoint.find_last_unset_in(0, KdAllocatedVgprs); V != -1) {
      LiveAtPoint.set(V);
      return V;
    }
    if (NextAboveKd >= MaxVgprs)
      return std::nullopt;
    unsigned V = NextAboveKd++;
    ExtraAllocated++;
    LiveAtPoint.set(V);
    return V;
  }

  unsigned extraVgprsNeeded() const { return ExtraAllocated; }
};

/// Allocates scratch SGPRs for a patch point. Unlike VGPRs (which have full
/// dataflow liveness), SGPRs have no liveness analysis, so we always allocate
/// above the kernel descriptor's reported SGPR count. This is conservative
/// but safe: no SGPR currently in use by the kernel can be clobbered.
struct SgprAllocator {
  unsigned KdAllocatedSgprs = 0;
  unsigned NextAboveKd = 0;
  unsigned MaxSgprs = 0;

  SgprAllocator(unsigned KdSgprs, unsigned Max)
      : KdAllocatedSgprs(KdSgprs), NextAboveKd(KdSgprs), MaxSgprs(Max) {}

  /// Allocate one SGPR above the kernel's current count. Returns
  /// std::nullopt if no headroom remains below MaxSgprs.
  std::optional<unsigned> alloc() {
    if (NextAboveKd >= MaxSgprs)
      return std::nullopt;
    return NextAboveKd++;
  }

  unsigned extraSgprsNeeded() const { return NextAboveKd - KdAllocatedSgprs; }
};

/// Bookkeeping for a single patch site's scratch allocation. \c Offset is
/// the .text byte offset of the patch; \c ScratchRegs is the bitvector of
/// VGPRs the patch claimed at that site. Consumed by the post-patch
/// verifier (verifyPatchCorrectness) to check the patches are mutually
/// consistent across the kernel.
struct ScratchPatchInfo {
  uint64_t Offset = 0;
  llvm::BitVector ScratchRegs;
};

// -- Patch types --------------------------------------------------------------

/// Per-kernel counters accumulated by the patch passes. Reported via log()
/// at the end of the rewrite and exposed through the public
/// amd_comgr_hotswap_result_t once that result struct is wired up.
struct KernelPatchStats {
  unsigned ExtraVgprs = 0;
  unsigned ExtraSgprs = 0;
  unsigned ScratchReused = 0;
  unsigned ScratchAboveKd = 0;
};

/// Mutable per-run context threaded through all patch passes. Bundles the
/// input config, decoded instruction stream, raw .text bytes, MC state,
/// output streams (trampolines / scratch info), and the shared ELF view +
/// liveness result so patch passes have a single parameter to pass around.
struct PatchContext {
  const RewriteConfig &Config;
  std::vector<InternalDecodedInst> &Decoded;
  uint8_t *Text = nullptr;
  uint64_t TextSize = 0;
  // .text-relative offset at which the appended trampoline pool begins
  // (trampolinePoolVAddr() - textAddr()). Trampoline branch offsets are
  // computed against this, not TextSize, since the pool no longer sits
  // immediately after .text.
  uint64_t PoolBaseOffset = 0;
  const LLVMState &LS;
  std::vector<Trampoline> &OutTrampolines;
  std::vector<NopSled> &NopSleds;
  ElfView &Elf;
  const LivenessInfo &Liveness;
  llvm::StringMap<KernelPatchStats> &KernelStats;
  std::vector<ScratchPatchInfo> &OutScratchPatches;
  // Per-rewrite profiling session (opt-in via HOTSWAP_PROFILE; a no-op when
  // profiling is compiled out). Patch passes record typed timings / counts
  // through this without any string lookup or hot-path mutex.
  HotswapProfile &Profile;
  // Required patches are transformations whose unpatched original code is
  // unsafe to return when the selected rewrite policy needs the patch.
  bool RequiredPatchFailed = false;
  bool RequiredPatchApplied = false;
};

// -- Trampoline emission helpers (defined in comgr-hotswap-b0a0.cpp) ----------

[[nodiscard]] bool emitToNopSled(PatchContext &Ctx, NopSled &Sled,
                                 uint64_t InstOffset, uint32_t InstSize,
                                 llvm::ArrayRef<uint8_t> Replacement);
[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    llvm::ArrayRef<uint8_t> Replacement,
                                    bool AllowSafeFarReturn = false);

// Encode an s_add_pc_i64 PC-relative long branch from \p FromOffset to
// \p TargetOffset (.text byte offsets). Exposed for unit testing the offset
// math / encoding. Returns empty on failure.
llvm::SmallVector<uint8_t> encodeLongBranch(const LLVMState &LS,
                                            uint64_t FromOffset,
                                            uint64_t TargetOffset);

// Encode an SCC-neutral PC-relative long branch through an aligned SGPR pair.
// s_get_pc_i64 captures the next instruction's PC, s_add_nc_u64 applies the
// two's-complement delta without reading or writing SCC, and s_set_pc_i64
// transfers control. Exposed for unit testing the offset math and register
// constraints. Returns std::nullopt after logging the specific failure.
std::optional<llvm::SmallVector<uint8_t>>
encodeSccNeutralLongBranch(const LLVMState &LS, uint64_t FromOffset,
                           uint64_t TargetOffset, unsigned SgprBase);
[[nodiscard]] bool emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset,
                                       uint32_t InstSize,
                                       llvm::ArrayRef<uint8_t> Replacement,
                                       bool AllowSafeFarReturn = false);

// -- Patch dispatch vtable ----------------------------------------------------
//
// Function-pointer dispatch table that replaces the prior LLVM_ATTRIBUTE_WEAK
// + `#if !defined(_MSC_VER)` override pattern. PE/COFF does not honour weak
// the way ELF does, so on Windows the weak stubs silently won every patch
// call and the feature was a no-op (issue ROCm/llvm-project#2479).
//
// Patch modules supply their implementations through register*Patch
// functions invoked by installHotswapPatches(). The membership list is
// comgr-hotswap-patches.def; each entry there corresponds to one slot
// below and one register*Patch function in a sibling
// comgr-hotswap-patch-*.cpp. nullptr slots are treated as no-op by the
// dispatcher, so an unmigrated pass family (e.g. scratch) is safe to
// leave unbound until its first strong override lands.
//
// The singleton accessor below eagerly installs every registered slot in
// its own initializer, so production callers never observe an empty
// vtable. installHotswapPatches() is still exported for unit tests that
// want to drive the install against a local HotswapPatchVTable.

struct HotswapPatchVTable {
  // Per-instruction passes: called in declaration order; first non-zero
  // return wins for an instruction (matches the pre-vtable dispatcher
  // behaviour in applyGfx1250B0toA0Rules).
  uint32_t (*applyInPlacePatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyTrampolinePatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyWmmaSplitPatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyScratchPatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyWmmaScale16Patches)(PatchContext &, size_t) = nullptr;

  // Whole-kernel passes: called once per kernel after the per-instruction
  // loop completes.
  uint32_t (*applyWmmaHazardPatch)(PatchContext &) = nullptr;
  uint32_t (*applyVop3px2Src2Fix)(PatchContext &) = nullptr;
};

/// Walk comgr-hotswap-patches.def and bind every patch module's
/// implementation into \p VT by calling its register*Patch function.
/// A missing register*Patch produces a link error, which is the
/// loud-failure shape the weak-symbol pattern lacked. Production code
/// never calls this directly; it runs inside getHotswapPatchVTable()'s
/// initializer. Exposed here so unit tests can drive the install against
/// a local HotswapPatchVTable.
void installHotswapPatches(HotswapPatchVTable &VT);

/// Process-wide HotswapPatchVTable singleton (Meyers-style). The
/// initializer eagerly calls installHotswapPatches() on its own storage,
/// so every reference returned here is to a fully bound vtable. C++11
/// [stmt.dcl]/4 guarantees the initializer runs exactly once and is safe
/// under concurrent first access, which removes the need for an explicit
/// std::call_once at the entry point and any inter-TU static-init order
/// contract on the patch modules.
HotswapPatchVTable &getHotswapPatchVTable();

// Forward-declare every patch module's installer from the central .def
// registry. Patch modules define these in their comgr-hotswap-patch-*.cpp;
// installHotswapPatches() consumes them; unit tests under test-unit/ also
// invoke them directly. A patches.def line with no matching definition
// produces a libamd_comgr / HotswapMCTests link error.
#define HOTSWAP_PATCH(Name) void register##Name##Patch(HotswapPatchVTable &);
#include "comgr-hotswap-patches.def"
#undef HOTSWAP_PATCH

// -- Function declarations (kernel-entry trampoline pass) ---------------------

struct KernelEntryTrampolineFixup {
  std::string KernelName;
  uint64_t StubTextOffset = 0;
  unsigned RequiredSgprs = 0;
  uint32_t InstPrefLines = 0;
};

/// Build a 256-byte, entry-aligned HotSwap kernel-entry stub at
/// \p StubVAddr that jumps to \p EntryVAddr using PC-relative address
/// materialization. Returns an empty vector if MC assembly fails.
llvm::SmallVector<uint8_t> buildKernelEntryTrampoline(uint64_t StubVAddr,
                                                      uint64_t EntryVAddr,
                                                      unsigned ScratchSgpr,
                                                      const LLVMState &LS);

/// Structural matcher for the entry stubs produced by
/// buildKernelEntryTrampoline, used to keep the rewrite idempotent.
bool isKernelEntryTrampoline(llvm::ArrayRef<uint8_t> Bytes,
                             const LLVMState &LS);

/// Cheap raw-byte prefilter for the entry stubs produced by
/// buildKernelEntryTrampoline. This is intentionally weaker than
/// isKernelEntryTrampoline and exists to avoid running the disassembler over
/// arbitrary original kernel entry bytes during idempotency checks.
bool hasKernelEntryTrampolinePrefix(llvm::ArrayRef<uint8_t> Bytes,
                                    const LLVMState &LS);

/// Compute the trailing readable guard needed after an appended kernel-entry
/// stub pool so CP instruction prefetches from the last stub cannot run past
/// mapped .text bytes.
uint64_t computeKernelEntryPrefetchGuardBytes(uint32_t InstPrefLines);

/// Append one entry stub per kernel descriptor that does not already target a
/// HotSwap entry stub. The stubs are appended to \p Growth and descriptor
/// rewrites are recorded in \p OutFixups for application after ELF growth.
std::optional<uint32_t> appendKernelEntryTrampolines(
    const ElfView &Elf, const LLVMState &LS, unsigned MaxSgprs,
    std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups);

/// Apply descriptor rewrites recorded by appendKernelEntryTrampolines after
/// the ELF has been grown.
bool rewriteKernelEntryDescriptorOffsets(
    llvm::WritableMemoryBuffer &OutBuf, uint64_t PoolVAddr,
    llvm::StringRef TargetCpu,
    llvm::ArrayRef<KernelEntryTrampolineFixup> Fixups);

/// Add a `<kernel_name>.stub` STT_FUNC symbol to the code object's `.symtab`
/// for each appended kernel-entry stub, so tools that resolve a dispatch's
/// entry address to a name (e.g. rocgdb `info dispatches`, which reads the
/// non-alloc `.symtab`) report the stub instead of a bare address. Returns a
/// newly allocated buffer with the grown `.symtab` / `.strtab`, or nullptr if
/// no symbols were added (empty fixups, missing `.symtab`, or a structural
/// problem) -- callers treat nullptr as "keep the existing buffer", since the
/// symbol is a debugging aid and its absence is not a correctness failure.
///
/// Only the trailing non-alloc `.symtab` / `.strtab` sections grow, so no
/// virtual addresses, program headers, or relocations change; `.dynsym` (used
/// by the loader) is left untouched.
std::unique_ptr<llvm::WritableMemoryBuffer> addKernelEntryTrampolineSymbols(
    llvm::WritableMemoryBuffer &In, unsigned TextSectionIndex, uint64_t TextAddr,
    uint64_t OldTextSize, llvm::ArrayRef<KernelEntryTrampolineFixup> Fixups);

// -- Function declarations (GFX1250 hotswap policy layer) ---------------------

struct Gfx1250RewriteOptions {
  bool RunB0A0Patches = true;
  bool RunEntryTrampolines = false;
  MaskWorkaroundPolicy MaskPolicy = MaskWorkaroundPolicy::None;
};

/// Run the selected GFX1250 hotswap rewrite passes on \p ElfData / \p ElfSize.
/// \p TargetIdent is the parsed target ISA (produced upstream by Comgr's
/// parseTargetIdentifier() or the hotswap-local stepping parser); it is
/// threaded into the MC init so the subtarget triple and feature flags are
/// preserved rather than being reconstructed from just the processor name. On
/// success \p Out is populated with an owned buffer containing the rewritten
/// code object. The caller can transfer the buffer directly to a comgr
/// DataObject via DataObject::setData(std::unique_ptr<MemoryBuffer>).
amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<llvm::MemoryBuffer> &Out);

} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_INTERNAL_H
