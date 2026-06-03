//===- InstrProfilingPlatformROCm.cpp - Profile data ROCm platform -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the Linux/Unix device profile drain, which decouples device counter
// collection from the host by walking HSA code objects (no host-side per-TU
// shadow). Windows has no HSA runtime, so it keeps the legacy HIP host-shadow
// mechanism in InstrProfilingPlatformROCmWindows.cpp; the CMake selects exactly
// one of the two by platform. Decoupled (host-uninstrumented) collection is
// only supported here, on Linux.
//
// Device-side profile data drain for AMDGPU via HSA introspection.
//
// At process exit this walks every loaded HSA code object on every GPU agent,
// finds the device-side __llvm_profile_sections bounds table (emitted by
// InstrProfilingPlatformGPU.c), copies its counters/data/names regions back to
// the host, and writes a target-prefixed .profraw via __llvm_write_custom_profile.
//
// The drain is decoupled from the host-side profile write: it runs from an
// atexit handler registered in a constructor, so device counters are collected
// whether or not the host translation units were instrumented, and without any
// host-side per-TU shadow variable, CUID matching, or hipModuleLoad
// interception.
//
// All HSA and HIP entry points are resolved with dlopen/dlsym (via the
// interception helpers) so libclang_rt.profile still links and runs on hosts
// without ROCm installed.
//
//===----------------------------------------------------------------------===//

// Host-only: this drains device counters from the host process. The device
// side (the __llvm_profile_sections bounds table) is emitted by
// InstrProfilingPlatformGPU.c. When this file is compiled for a GPU target as
// part of the device profile runtime build it reduces to an empty TU. It also
// reduces to an empty TU on Windows, which uses
// InstrProfilingPlatformROCmWindows.cpp instead.
#if !defined(__NVPTX__) && !defined(__AMDGPU__) && !defined(_WIN32)

extern "C" {
#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingPort.h"
}

#include "interception/interception.h"
// C library headers only (not <cstdio> etc.): clang_rt.profile is built with
// -nostdinc++ and avoids the C++ standard library (see profile/CMakeLists.txt).
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/*  Minimal HSA type/enum stubs                                               */
/*                                                                            */
/*  compiler-rt cannot depend on ROCm headers at build time, so mirror just   */
/*  the handful of HSA declarations the drain needs. Values match             */
/*  hsa/hsa.h and hsa/hsa_ven_amd_loader.h.                                   */
/* -------------------------------------------------------------------------- */

typedef uint32_t prof_hsa_status_t;
#define PROF_HSA_STATUS_SUCCESS ((prof_hsa_status_t)0x0)
#define PROF_HSA_STATUS_INFO_BREAK ((prof_hsa_status_t)0x1)

typedef struct {
  uint64_t handle;
} prof_hsa_agent_t;
typedef struct {
  uint64_t handle;
} prof_hsa_executable_t;
typedef struct {
  uint64_t handle;
} prof_hsa_executable_symbol_t;

typedef uint32_t prof_hsa_agent_info_t;
#define PROF_HSA_AGENT_INFO_NAME ((prof_hsa_agent_info_t)0)
#define PROF_HSA_AGENT_INFO_DEVICE ((prof_hsa_agent_info_t)17)

typedef uint32_t prof_hsa_device_type_t;
#define PROF_HSA_DEVICE_TYPE_GPU ((prof_hsa_device_type_t)1)

typedef uint32_t prof_hsa_symbol_kind_t;
#define PROF_HSA_SYMBOL_KIND_VARIABLE ((prof_hsa_symbol_kind_t)0)

typedef uint32_t prof_hsa_executable_symbol_info_t;
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_TYPE                                    \
  ((prof_hsa_executable_symbol_info_t)0)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH                             \
  ((prof_hsa_executable_symbol_info_t)1)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME                                    \
  ((prof_hsa_executable_symbol_info_t)2)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS                        \
  ((prof_hsa_executable_symbol_info_t)21)

#define PROF_HSA_EXTENSION_AMD_LOADER ((uint16_t)0x201)

typedef uint32_t prof_hsa_loader_storage_type_t;

typedef struct {
  prof_hsa_agent_t agent;
  prof_hsa_executable_t executable;
  prof_hsa_loader_storage_type_t code_object_storage_type;
  const void *code_object_storage_base;
  size_t code_object_storage_size;
  size_t code_object_storage_offset;
  const void *segment_base;
  size_t segment_size;
} prof_hsa_loader_segment_descriptor_t;

typedef prof_hsa_status_t (*hsa_init_ty)(void);
typedef prof_hsa_status_t (*hsa_iterate_agents_ty)(
    prof_hsa_status_t (*)(prof_hsa_agent_t, void *), void *);
typedef prof_hsa_status_t (*hsa_agent_get_info_ty)(prof_hsa_agent_t,
                                                   prof_hsa_agent_info_t,
                                                   void *);
typedef prof_hsa_status_t (*hsa_executable_iterate_agent_symbols_ty)(
    prof_hsa_executable_t, prof_hsa_agent_t,
    prof_hsa_status_t (*)(prof_hsa_executable_t, prof_hsa_agent_t,
                          prof_hsa_executable_symbol_t, void *),
    void *);
typedef prof_hsa_status_t (*hsa_executable_symbol_get_info_ty)(
    prof_hsa_executable_symbol_t, prof_hsa_executable_symbol_info_t, void *);
typedef prof_hsa_status_t (*hsa_system_get_major_extension_table_ty)(
    uint16_t, uint16_t, size_t, void *);
typedef prof_hsa_status_t (*hsa_loader_query_segment_descriptors_ty)(
    prof_hsa_loader_segment_descriptor_t *, size_t *);

/* First two members of hsa_ven_amd_loader_1_00_pfn_t. Only
 * query_segment_descriptors is used; query_host_address keeps the offset. */
typedef struct {
  void *query_host_address;
  hsa_loader_query_segment_descriptors_ty query_segment_descriptors;
} prof_hsa_loader_pfn_t;

/* HIP: only hipMemcpy is needed, for device-to-host copies. */
typedef int (*hipMemcpy_ty)(void *, const void *, size_t, int);

/* -------------------------------------------------------------------------- */
/*  Resolved runtime entry points                                             */
/* -------------------------------------------------------------------------- */

static hsa_iterate_agents_ty pHsaIterateAgents = nullptr;
static hsa_agent_get_info_ty pHsaAgentGetInfo = nullptr;
static hsa_executable_iterate_agent_symbols_ty pHsaExecIterAgentSyms = nullptr;
static hsa_executable_symbol_get_info_ty pHsaSymGetInfo = nullptr;
static hsa_loader_query_segment_descriptors_ty pQuerySegDescs = nullptr;
static hipMemcpy_ty pHipMemcpy = nullptr;

/* 0 = not yet attempted, 1 = ready, -1 = unavailable.
 * Accessed with acquire/release atomics: a thread that observes RuntimeState==1
 * (acquire) is guaranteed to also see the fully-written p* function pointers
 * (which are published before the release store of RuntimeState=1 below). */
static int RuntimeState = 0;

static int isVerboseMode(void) {
  static int IsVerbose = -1;
  if (IsVerbose == -1)
    IsVerbose = getenv("LLVM_PROFILE_VERBOSE") != nullptr;
  return IsVerbose;
}

/* -------------------------------------------------------------------------- */
/*  One-time runtime resolution                                               */
/* -------------------------------------------------------------------------- */

/* Publish the terminal RuntimeState (release) and map it to the 0/-1 return
 * convention used by loadRuntimePointers(). */
static int setRuntimeState(int S) {
  __atomic_store_n(&RuntimeState, S, __ATOMIC_RELEASE);
  return S > 0 ? 0 : -1;
}

static int loadRuntimePointers(void) {
  int State = __atomic_load_n(&RuntimeState, __ATOMIC_ACQUIRE);
  if (State)
    return State > 0 ? 0 : -1;

  if (!__interception::DynamicLoaderAvailable()) {
    if (isVerboseMode())
      PROF_NOTE("%s", "Dynamic library loading not available - "
                      "device profiling disabled\n");
    return setRuntimeState(-1);
  }

  void *Hsa = __interception::OpenLibrary("libhsa-runtime64.so");
  if (!Hsa)
    Hsa = __interception::OpenLibrary("libhsa-runtime64.so.1");
  if (!Hsa) {
    if (isVerboseMode())
      PROF_NOTE("%s", "libhsa-runtime64.so not loadable - "
                      "device profiling disabled\n");
    return setRuntimeState(-1);
  }

  hsa_init_ty pHsaInit =
      (hsa_init_ty)__interception::LookupSymbol(Hsa, "hsa_init");
  hsa_system_get_major_extension_table_ty pGetExtTable =
      (hsa_system_get_major_extension_table_ty)__interception::LookupSymbol(
          Hsa, "hsa_system_get_major_extension_table");
  pHsaIterateAgents = (hsa_iterate_agents_ty)__interception::LookupSymbol(
      Hsa, "hsa_iterate_agents");
  pHsaAgentGetInfo = (hsa_agent_get_info_ty)__interception::LookupSymbol(
      Hsa, "hsa_agent_get_info");
  pHsaExecIterAgentSyms =
      (hsa_executable_iterate_agent_symbols_ty)__interception::LookupSymbol(
          Hsa, "hsa_executable_iterate_agent_symbols");
  pHsaSymGetInfo =
      (hsa_executable_symbol_get_info_ty)__interception::LookupSymbol(
          Hsa, "hsa_executable_symbol_get_info");

  if (!pHsaInit || !pGetExtTable || !pHsaIterateAgents || !pHsaAgentGetInfo ||
      !pHsaExecIterAgentSyms || !pHsaSymGetInfo) {
    PROF_WARN("%s", "required HSA symbols missing - device profiling disabled\n");
    return setRuntimeState(-1);
  }

  /* Bring HSA up now (idempotent, refcounted). Doing this in the library
   * constructor guarantees HSA registers its own atexit(hsa_shut_down)
   * before we register atexit(drainDevices); atexit is LIFO, so our drain
   * runs while HSA is still alive. */
  prof_hsa_status_t St = pHsaInit();
  if (St != PROF_HSA_STATUS_SUCCESS && St != PROF_HSA_STATUS_INFO_BREAK) {
    if (isVerboseMode())
      PROF_NOTE("hsa_init failed (0x%x) - device profiling disabled\n", St);
    return setRuntimeState(-1);
  }

  prof_hsa_loader_pfn_t LoaderApi;
  __builtin_memset(&LoaderApi, 0, sizeof(LoaderApi));
  St = pGetExtTable(PROF_HSA_EXTENSION_AMD_LOADER, 1, sizeof(LoaderApi),
                    &LoaderApi);
  if (St != PROF_HSA_STATUS_SUCCESS || !LoaderApi.query_segment_descriptors) {
    PROF_WARN("AMD loader extension unavailable (0x%x) - "
              "device profiling disabled\n",
              St);
    return setRuntimeState(-1);
  }
  pQuerySegDescs = LoaderApi.query_segment_descriptors;

  /* HIP lookup is best-effort across deployment shapes:
   *   1. The vast majority of HIP-using programs already have libamdhip64
   *      loaded (the application or one of its libraries linked it directly).
   *      Resolving via the process namespace catches that case without us
   *      having to know the SONAME, and works even when there is no
   *      development "libamdhip64.so" symlink (runtime-only ROCm installs).
   *   2. If hipMemcpy is not in the namespace, fall back to dlopen, trying
   *      versioned SONAMEs first (which is what the dynamic linker actually
   *      loads at program start) before the unversioned dev symlink.
   *   3. On Windows just dlopen the DLL by name. */
  pHipMemcpy =
      (hipMemcpy_ty)__interception::LookupSymbolDefault("hipMemcpy");
  if (!pHipMemcpy) {
#ifdef _WIN32
    static const char *const HipLibNames[] = {"amdhip64.dll", nullptr};
#else
    /* Order: most recent ROCm major first, then older majors, then the
     * unversioned development symlink as a last resort. Update this list
     * when ROCm bumps libamdhip64 SONAME. */
    static const char *const HipLibNames[] = {
        "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5",
        "libamdhip64.so.4", "libamdhip64.so",   nullptr};
#endif
    for (int i = 0; HipLibNames[i] != nullptr; ++i) {
      void *Hip = __interception::OpenLibrary(HipLibNames[i]);
      if (!Hip)
        continue;
      pHipMemcpy =
          (hipMemcpy_ty)__interception::LookupSymbol(Hip, "hipMemcpy");
      if (pHipMemcpy) {
        if (isVerboseMode())
          PROF_NOTE("HIP resolved via dlopen(%s)\n", HipLibNames[i]);
        break;
      }
    }
  } else if (isVerboseMode()) {
    PROF_NOTE("%s",
              "HIP resolved via existing process namespace (RTLD_DEFAULT)\n");
  }
  if (!pHipMemcpy) {
    PROF_WARN("%s", "hipMemcpy unavailable - device profiling disabled\n");
    return setRuntimeState(-1);
  }

  if (isVerboseMode())
    PROF_NOTE("%s", "HSA + HIP runtime resolved for device profiling\n");
  return setRuntimeState(1);
}

static int memcpyDeviceToHost(void *Dst, const void *Src, size_t Size) {
  return pHipMemcpy ? pHipMemcpy(Dst, Src, Size, 2 /* hipMemcpyDeviceToHost */)
                    : -1;
}

/* -------------------------------------------------------------------------- */
/*  free()-based scope guard                                                  */
/* -------------------------------------------------------------------------- */

namespace {
struct UniqueFree {
  void *Ptr;
  explicit UniqueFree(void *P = nullptr) : Ptr(P) {}
  ~UniqueFree() { free(Ptr); }
  UniqueFree(const UniqueFree &) = delete;
  UniqueFree &operator=(const UniqueFree &) = delete;
  char *get() const { return static_cast<char *>(Ptr); }
  void reset(void *P) {
    free(Ptr);
    Ptr = P;
  }
};
} // namespace

/* -------------------------------------------------------------------------- */
/*  Copy one device bounds table to the host and emit a .profraw              */
/* -------------------------------------------------------------------------- */

/* Plausibility cap for any single device-side profile section. Device
 * profile data for a single linked code object is typically <10 MiB; a few
 * hundred MiB would already be unprecedented. Setting this to 256 MiB
 * catches a corrupted/uninitialized bounds table early (where End-Begin
 * can compute to multi-GiB) before we try to malloc/memcpy bogus memory. */
#define PROF_MAX_SECTION_BYTES ((size_t)256 * 1024 * 1024)

/* uintptr_t-based size of a [Begin, End] device range, with validation.
 * Returns -1 if End < Begin (would wrap to a huge size_t) or if the span
 * exceeds the per-section cap. On success, *OutSize is the byte count. */
static int computeRangeSize(const char *Label, const void *Begin,
                            const void *End, size_t *OutSize) {
  uintptr_t B = (uintptr_t)Begin;
  uintptr_t E = (uintptr_t)End;
  if (E < B) {
    PROF_WARN("device %s range invalid: end %p < begin %p\n", Label, End,
              Begin);
    return -1;
  }
  size_t Sz = (size_t)(E - B);
  if (Sz > PROF_MAX_SECTION_BYTES) {
    PROF_WARN("device %s range %zu bytes exceeds %zu-byte cap; refusing "
              "to copy (likely corrupted bounds table)\n",
              Label, Sz, (size_t)PROF_MAX_SECTION_BYTES);
    return -1;
  }
  *OutSize = Sz;
  return 0;
}

/* Returns 1 if a device .profraw was written, 0 if there was nothing to write
 * (empty counters/data sections), and -1 on error. The caller distinguishes
 * these so an empty section is never miscounted as a successful drain. */
static int processDeviceSections(void *DeviceSectionsAddr, const char *Target) {
  __llvm_profile_gpu_sections HostSections;
  if (memcpyDeviceToHost(&HostSections, DeviceSectionsAddr,
                         sizeof(HostSections)) != 0) {
    PROF_ERR("%s\n", "failed to copy device bounds table from device");
    return -1;
  }

  const void *DevCntsBegin = HostSections.CountersStart;
  const void *DevCntsEnd = HostSections.CountersStop;
  const void *DevDataBegin = HostSections.DataStart;
  const void *DevDataEnd = HostSections.DataStop;
  const void *DevNamesBegin = HostSections.NamesStart;
  const void *DevNamesEnd = HostSections.NamesStop;

  size_t CountersSize, DataSize, NamesSize;
  if (computeRangeSize("counters", DevCntsBegin, DevCntsEnd, &CountersSize) !=
          0 ||
      computeRangeSize("data", DevDataBegin, DevDataEnd, &DataSize) != 0 ||
      computeRangeSize("names", DevNamesBegin, DevNamesEnd, &NamesSize) != 0)
    return -1;

  /* DataSize must be an integral number of __llvm_profile_data records;
   * otherwise either the table layout has changed under us or the bounds
   * point at the wrong section. Refuse to relocate it - the per-record
   * loop below would walk off the end. */
  if (DataSize % sizeof(__llvm_profile_data) != 0) {
    PROF_WARN("device data section size %zu is not a multiple of "
              "sizeof(__llvm_profile_data)=%zu\n",
              DataSize, sizeof(__llvm_profile_data));
    return -1;
  }

  if (isVerboseMode())
    PROF_NOTE("Section pointers: Cnts=[%p,%p]=%zu Data=[%p,%p]=%zu "
              "Names=[%p,%p]=%zu\n",
              DevCntsBegin, DevCntsEnd, CountersSize, DevDataBegin, DevDataEnd,
              DataSize, DevNamesBegin, DevNamesEnd, NamesSize);

  if (CountersSize == 0 || DataSize == 0)
    return 0;

  UniqueFree CntsOwner, DataOwner, NamesOwner;
  char *HostCounters = (char *)malloc(CountersSize);
  CntsOwner.reset(HostCounters);
  char *HostData = (char *)malloc(DataSize);
  DataOwner.reset(HostData);
  char *HostNames = NamesSize ? (char *)malloc(NamesSize) : nullptr;
  if (NamesSize)
    NamesOwner.reset(HostNames);

  if (!HostCounters || !HostData || (NamesSize && !HostNames)) {
    PROF_ERR("%s\n", "failed to allocate host memory for device sections");
    return -1;
  }

  if (memcpyDeviceToHost(HostData, DevDataBegin, DataSize) != 0 ||
      memcpyDeviceToHost(HostCounters, DevCntsBegin, CountersSize) != 0 ||
      (NamesSize &&
       memcpyDeviceToHost(HostNames, DevNamesBegin, NamesSize) != 0)) {
    PROF_ERR("%s\n", "failed to copy profile sections from device");
    return -1;
  }

  if (isVerboseMode())
    PROF_NOTE("Copied device sections: Counters=%zu, Data=%zu, Names=%zu\n",
              CountersSize, DataSize, NamesSize);

  // Lay the buffer out as [Data][PaddingBeforeCounters][Counters][Names] to
  // match what lprofWriteDataImpl expects (CountersDelta = Counters - Data).
  const uint64_t NumData = DataSize / sizeof(__llvm_profile_data);
  uint64_t PadBeforeCounters, PadAfterCounters, PadAfterBitmap, PadAfterNames,
      PadAfterVTable, PadAfterVNames;
  if (__llvm_profile_get_padding_sizes_for_counters(
          DataSize, CountersSize, /*NumBitmapBytes=*/0, NamesSize,
          /*VTableSize=*/0, /*VNameSize=*/0, &PadBeforeCounters,
          &PadAfterCounters, &PadAfterBitmap, &PadAfterNames, &PadAfterVTable,
          &PadAfterVNames) != 0) {
    PROF_ERR("%s\n", "failed to get padding sizes");
    return -1;
  }

  size_t BufSize = DataSize + PadBeforeCounters + CountersSize + NamesSize;
  UniqueFree BufOwner(malloc(BufSize));
  char *Buf = BufOwner.get();
  if (!Buf) {
    PROF_ERR("%s\n", "failed to allocate contiguous buffer");
    return -1;
  }
  __builtin_memset(Buf, 0, BufSize);

  char *BufData = Buf;
  char *BufCounters = Buf + DataSize + PadBeforeCounters;
  char *BufNames = BufCounters + CountersSize;

  __builtin_memcpy(BufData, HostData, DataSize);
  __builtin_memcpy(BufCounters, HostCounters, CountersSize);
  if (NamesSize)
    __builtin_memcpy(BufNames, HostNames, NamesSize);

  // Relocate each record's CounterPtr from the device-relative offset to the
  // file-layout-relative offset (Data section precedes Counters in the file).
  // Validate every resolved device counter address lies within the copied
  // counters region; out-of-range entries indicate a stale/mismatched bounds
  // table and would otherwise produce a .profraw with counters pointing at
  // unrelated memory.
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)BufData;
  int BadRecords = 0;
  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      uintptr_t DeviceDataStructAddr =
          (uintptr_t)DevDataBegin + (uintptr_t)(i * sizeof(__llvm_profile_data));
      uintptr_t DeviceCountersAddr =
          DeviceDataStructAddr + (uintptr_t)DeviceCounterPtrOffset;
      uintptr_t CntsB = (uintptr_t)DevCntsBegin;
      uintptr_t CntsE = (uintptr_t)DevCntsEnd;
      /* Allow CountersAddr == CntsE for a zero-counter record at the very
       * end of the section. */
      if (DeviceCountersAddr < CntsB || DeviceCountersAddr > CntsE) {
        BadRecords++;
        if (isVerboseMode())
          PROF_NOTE("record %llu: device counter addr %p outside "
                    "[%p,%p]; zeroing CounterPtr\n",
                    (unsigned long long)i, (void *)DeviceCountersAddr,
                    DevCntsBegin, DevCntsEnd);
        // CounterPtr is IntPtrT (pointer-sized): zero exactly that field so we
        // never clobber adjacent record fields on a 32-bit host.
        __builtin_memset((char *)RelocatedData +
                             i * sizeof(__llvm_profile_data) +
                             offsetof(__llvm_profile_data, CounterPtr),
                         0, sizeof(RelocatedData[i].CounterPtr));
      } else {
        ptrdiff_t OffsetIntoCountersSection =
            (ptrdiff_t)(DeviceCountersAddr - CntsB);
        ptrdiff_t NewRelativeOffset =
            (ptrdiff_t)DataSize + (ptrdiff_t)PadBeforeCounters +
            OffsetIntoCountersSection -
            (ptrdiff_t)(i * sizeof(__llvm_profile_data));
        __builtin_memcpy((char *)RelocatedData +
                             i * sizeof(__llvm_profile_data) +
                             offsetof(__llvm_profile_data, CounterPtr),
                         &NewRelativeOffset, sizeof(NewRelativeOffset));
      }
    }
    // Zero the fields the writer does not expect to be populated.
    __builtin_memset((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                         offsetof(__llvm_profile_data, BitmapPtr),
                     0,
                     sizeof(RelocatedData[i].BitmapPtr) +
                         sizeof(RelocatedData[i].FunctionPointer) +
                         sizeof(RelocatedData[i].Values));
  }
  if (BadRecords > 0)
    PROF_WARN("%d/%llu device profile record(s) had out-of-range "
              "counter pointers (zeroed)\n",
              BadRecords, (unsigned long long)NumData);

  int Ret = __llvm_write_custom_profile(
      Target, (__llvm_profile_data *)BufData,
      (__llvm_profile_data *)(BufData + DataSize), BufCounters,
      BufCounters + CountersSize, BufNames, BufNames + NamesSize, nullptr);

  if (Ret != 0) {
    PROF_ERR("%s\n", "failed to write device profile");
    return -1;
  }
  if (isVerboseMode())
    PROF_NOTE("Wrote device profile (target=%s)\n", Target);
  return 1;
}

/* -------------------------------------------------------------------------- */
/*  HSA walk                                                                  */
/* -------------------------------------------------------------------------- */

#define PROF_MAX_GPU_AGENTS 64

namespace {
struct GpuAgent {
  prof_hsa_agent_t agent;
  char arch[64];
};

struct WalkState {
  GpuAgent agents[PROF_MAX_GPU_AGENTS];
  int num_agents;
  int total_found;
  int total_drained;
};

/* Per (agent, executable) symbol-iteration state. */
struct SymbolState {
  const char *arch;
  int found;
  int drained;
};
} // namespace

/* The canonical device bounds table symbol from InstrProfilingPlatformGPU.c. */
static const char ProfileSectionsSymbol[] = "__llvm_profile_sections";

/* Dedup distinct (Data,Counters,Names) tuples: a single linked device code
 * object exposes one __llvm_profile_sections, but the same bounds may be seen
 * via multiple agents, so drain each unique counter set only once. Also used
 * to generate collision-free target names. */
namespace {
struct BoundsTuple {
  const void *data;
  const void *cnts;
  const void *names;
};
} // namespace

#define PROF_MAX_SEEN_BOUNDS 256
static BoundsTuple SeenBounds[PROF_MAX_SEEN_BOUNDS];
static int NumSeenBounds = 0;

static int alreadySeenBounds(const void *D, const void *C, const void *N) {
  for (int i = 0; i < NumSeenBounds; ++i)
    if (SeenBounds[i].data == D && SeenBounds[i].cnts == C &&
        SeenBounds[i].names == N)
      return 1;
  if (NumSeenBounds < PROF_MAX_SEEN_BOUNDS) {
    SeenBounds[NumSeenBounds].data = D;
    SeenBounds[NumSeenBounds].cnts = C;
    SeenBounds[NumSeenBounds].names = N;
    NumSeenBounds++;
  }
  return 0;
}

static prof_hsa_status_t onSymbol(prof_hsa_executable_t, prof_hsa_agent_t,
                                  prof_hsa_executable_symbol_t Sym,
                                  void *Data) {
  SymbolState *S = (SymbolState *)Data;

  prof_hsa_symbol_kind_t Kind;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Kind) !=
          PROF_HSA_STATUS_SUCCESS ||
      Kind != PROF_HSA_SYMBOL_KIND_VARIABLE)
    return PROF_HSA_STATUS_SUCCESS;

  uint32_t NameLen = 0;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
                     &NameLen) != PROF_HSA_STATUS_SUCCESS ||
      NameLen != sizeof(ProfileSectionsSymbol) - 1)
    return PROF_HSA_STATUS_SUCCESS;

  char NameBuf[64];
  if (NameLen + 1 > sizeof(NameBuf))
    return PROF_HSA_STATUS_SUCCESS;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME, NameBuf) !=
      PROF_HSA_STATUS_SUCCESS)
    return PROF_HSA_STATUS_SUCCESS;
  NameBuf[NameLen] = '\0';

  if (strcmp(NameBuf, ProfileSectionsSymbol) != 0)
    return PROF_HSA_STATUS_SUCCESS;

  uint64_t Addr = 0;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                     &Addr) != PROF_HSA_STATUS_SUCCESS ||
      Addr == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "failed to read __llvm_profile_sections address\n");
    return PROF_HSA_STATUS_SUCCESS;
  }

  S->found++;

  // Read the bounds table first to dedup before the full copy.
  __llvm_profile_gpu_sections Sec;
  if (memcpyDeviceToHost(&Sec, (void *)(uintptr_t)Addr, sizeof(Sec)) != 0) {
    PROF_WARN("%s", "failed to copy device bounds table\n");
    return PROF_HSA_STATUS_SUCCESS;
  }
  if (alreadySeenBounds(Sec.DataStart, Sec.CountersStart, Sec.NamesStart)) {
    if (isVerboseMode())
      PROF_NOTE("%s", "device bounds already drained, skipping\n");
    return PROF_HSA_STATUS_SUCCESS;
  }

  // Generate a collision-free target. Multiple distinct device code objects on
  // the same arch (e.g. non-RDC multi-TU) must not clobber each other's file.
  static int DrainIndex = 0;
  char Target[96];
  if (DrainIndex == 0)
    snprintf(Target, sizeof(Target), "%s", S->arch);
  else
    snprintf(Target, sizeof(Target), "%s.%d", S->arch, DrainIndex);

  // Only a >0 result means a .profraw was actually written; an empty section
  // (0) or an error (<0) must not be counted as a drain or advance DrainIndex.
  if (processDeviceSections((void *)(uintptr_t)Addr, Target) > 0) {
    S->drained++;
    DrainIndex++;
  }

  return PROF_HSA_STATUS_SUCCESS;
}

static prof_hsa_status_t collectAgent(prof_hsa_agent_t Agent, void *Data) {
  prof_hsa_device_type_t DevType;
  if (pHsaAgentGetInfo(Agent, PROF_HSA_AGENT_INFO_DEVICE, &DevType) !=
          PROF_HSA_STATUS_SUCCESS ||
      DevType != PROF_HSA_DEVICE_TYPE_GPU)
    return PROF_HSA_STATUS_SUCCESS;

  WalkState *W = (WalkState *)Data;
  if (W->num_agents >= PROF_MAX_GPU_AGENTS)
    return PROF_HSA_STATUS_SUCCESS;

  GpuAgent &GA = W->agents[W->num_agents++];
  GA.agent = Agent;
  char Name[64];
  __builtin_memset(Name, 0, sizeof(Name));
  pHsaAgentGetInfo(Agent, PROF_HSA_AGENT_INFO_NAME, Name);
  size_t N = strnlen(Name, sizeof(GA.arch) - 1);
  __builtin_memcpy(GA.arch, Name, N);
  GA.arch[N] = '\0';
  if (!GA.arch[0])
    strncpy(GA.arch, "amdgpu", sizeof(GA.arch) - 1);

  if (isVerboseMode())
    PROF_NOTE("GPU agent %d: %s\n", W->num_agents - 1, GA.arch);
  return PROF_HSA_STATUS_SUCCESS;
}

/* Reentrancy guard and "we drained data at least once" flag. Both the host
 * write path and the atexit handler call drainDevices(); a successful walk
 * with non-empty results latches DrainCompleted so we never re-emit duplicate
 * .profraw files, but transient no-op outcomes ("runtime not yet loadable",
 * "no GPU agents", "no loaded segments", "no instrumented sections found")
 * stay retryable so the final atexit drain can still pick up code objects
 * that loaded later. The InProgress flag prevents a concurrent call from
 * another thread (or a re-entrant call on the same thread, e.g. a library
 * destructor that triggers another drain) from running the walk concurrently
 * and corrupting the global SeenBounds table. Both flags are accessed with
 * acquire/release atomics so the guard holds across threads. */
static int DrainInProgress = 0;
static int DrainCompleted = 0;

static int drainDevices(void) {
  if (__atomic_load_n(&DrainCompleted, __ATOMIC_ACQUIRE))
    return 0;

  /* Claim the drain with an atomic CAS. A failed CAS means either another
   * thread is already draining, or this is a reentrant call on the same
   * thread (e.g. a library destructor that triggers another drain); both
   * must bail without touching the global SeenBounds table. The acquire/
   * release ordering also publishes the worker thread's writes to threads
   * that observe DrainCompleted later. */
  int Expected = 0;
  if (!__atomic_compare_exchange_n(&DrainInProgress, &Expected, 1,
                                   /*weak=*/0, __ATOMIC_ACQ_REL,
                                   __ATOMIC_ACQUIRE))
    return 0;

  /* Mirror the early-exit paths so we always release the in-progress flag. */
  struct InProgressGuard {
    ~InProgressGuard() {
      __atomic_store_n(&DrainInProgress, 0, __ATOMIC_RELEASE);
    }
  } _Guard;

  if (loadRuntimePointers() != 0) {
    /* Runtime unavailable: don't latch DrainCompleted, allow a later call
     * (e.g. atexit, after the host has dlopen'd HIP) to retry. */
    return 0;
  }

  WalkState W;
  __builtin_memset(&W, 0, sizeof(W));
  prof_hsa_status_t St = pHsaIterateAgents(collectAgent, &W);
  if (St != PROF_HSA_STATUS_SUCCESS && St != PROF_HSA_STATUS_INFO_BREAK) {
    PROF_WARN("hsa_iterate_agents failed (0x%x)\n", St);
    return -1;
  }
  if (W.num_agents == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "no GPU agents present; nothing to drain (will retry)\n");
    return 0;
  }

  /* query_segment_descriptors ships in every loader-extension version and is
   * more permissive than iterate_executables on ROCm. It yields the loaded
   * (agent, executable) pairs directly. */
  size_t NumSegs = 0;
  St = pQuerySegDescs(nullptr, &NumSegs);
  if (St != PROF_HSA_STATUS_SUCCESS) {
    PROF_WARN("query_segment_descriptors(count) failed (0x%x)\n", St);
    return -1;
  }
  if (NumSegs == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s",
                "no loaded segments; nothing to drain (will retry)\n");
    return 0;
  }

  prof_hsa_loader_segment_descriptor_t *Segs =
      (prof_hsa_loader_segment_descriptor_t *)calloc(NumSegs, sizeof(*Segs));
  if (!Segs) {
    PROF_ERR("%s\n", "failed to allocate segment descriptor array");
    return -1;
  }
  UniqueFree SegsOwner(Segs);

  St = pQuerySegDescs(Segs, &NumSegs);
  if (St != PROF_HSA_STATUS_SUCCESS) {
    PROF_WARN("query_segment_descriptors(fetch) failed (0x%x)\n", St);
    return -1;
  }

  if (isVerboseMode())
    PROF_NOTE("query_segment_descriptors: %zu segments\n", NumSegs);

  /* Walk unique (agent, executable) pairs. */
  enum { kMaxPairs = 512 };
  uint64_t SeenAgents[kMaxPairs];
  uint64_t SeenExecs[kMaxPairs];
  int NumPairs = 0;
  int IterFailures = 0;

  for (size_t i = 0; i < NumSegs; ++i) {
    if (Segs[i].executable.handle == 0 || Segs[i].agent.handle == 0)
      continue;

    int Seen = 0;
    for (int j = 0; j < NumPairs; ++j)
      if (SeenAgents[j] == Segs[i].agent.handle &&
          SeenExecs[j] == Segs[i].executable.handle) {
        Seen = 1;
        break;
      }
    if (Seen)
      continue;
    if (NumPairs < kMaxPairs) {
      SeenAgents[NumPairs] = Segs[i].agent.handle;
      SeenExecs[NumPairs] = Segs[i].executable.handle;
      NumPairs++;
    }

    const char *Arch = nullptr;
    for (int k = 0; k < W.num_agents; ++k)
      if (W.agents[k].agent.handle == Segs[i].agent.handle) {
        Arch = W.agents[k].arch;
        break;
      }
    if (!Arch)
      continue; /* not a GPU agent we collected */

    SymbolState S;
    __builtin_memset(&S, 0, sizeof(S));
    S.arch = Arch;
    if (isVerboseMode())
      PROF_NOTE("walking executable 0x%llx on %s\n",
                (unsigned long long)Segs[i].executable.handle, Arch);
    prof_hsa_status_t IterSt =
        pHsaExecIterAgentSyms(Segs[i].executable, Segs[i].agent, onSymbol, &S);
    if (IterSt != PROF_HSA_STATUS_SUCCESS &&
        IterSt != PROF_HSA_STATUS_INFO_BREAK) {
      PROF_WARN("hsa_executable_iterate_agent_symbols on executable 0x%llx "
                "failed (0x%x)\n",
                (unsigned long long)Segs[i].executable.handle, IterSt);
      IterFailures++;
    }
    W.total_found += S.found;
    W.total_drained += S.drained;
  }

  if (isVerboseMode())
    PROF_NOTE("walk complete: agents=%d pairs=%d found=%d drained=%d "
              "iter-failures=%d\n",
              W.num_agents, NumPairs, W.total_found, W.total_drained,
              IterFailures);

  if (W.total_found > 0 && W.total_drained == 0) {
    PROF_WARN("found %d device profile symbol(s) but drained 0\n",
              W.total_found);
    return -1;
  }
  /* Latch only if we actually drained data, or if we successfully walked
   * everything and confirmed there is no instrumented code object loaded
   * (no symbols found, no per-executable iteration failures). The "no
   * instrumented code object" case is genuinely terminal for an exit drain
   * but harmless to repeat if anyone calls back in (and the host-write
   * forwarder may run before atexit). */
  if (W.total_drained > 0)
    __atomic_store_n(&DrainCompleted, 1, __ATOMIC_RELEASE);
  return (IterFailures > 0) ? -1 : 0;
}

/* -------------------------------------------------------------------------- */
/*  Public entry points                                                       */
/* -------------------------------------------------------------------------- */

/* Called from the host write path (InstrProfilingFile.c) when the host TUs are
 * instrumented. Independent of, and idempotent with, the atexit drain. */
extern "C" int __llvm_profile_hip_collect_device_data(void) {
  return drainDevices();
}

/* Legacy registration entry points from the previous host-shadow design, kept
 * as no-ops so objects compiled against the old runtime still link. */
extern "C" void __llvm_profile_offload_register_shadow_variable(void *) {}
extern "C" void
__llvm_profile_offload_register_section_shadow_variable(void *) {}
extern "C" void __llvm_profile_offload_register_dynamic_module(int, void **,
                                                               const void *) {}
extern "C" void __llvm_profile_offload_unregister_dynamic_module(void *) {}

/* -------------------------------------------------------------------------- */
/*  Constructor                                                               */
/* -------------------------------------------------------------------------- */

static void atexitDrain(void) { (void)drainDevices(); }

__attribute__((constructor)) static void profROCmInit(void) {
  // Resolve and hsa_init now so HSA's atexit(hsa_shut_down) is registered
  // before our atexit(drainDevices); LIFO then runs our drain while HSA is
  // still alive. Failure here is non-fatal: a host-only program without ROCm
  // simply gets no device drain.
  (void)loadRuntimePointers();
  atexit(atexitDrain);
}

#endif // !defined(__NVPTX__) && !defined(__AMDGPU__) && !defined(_WIN32)
