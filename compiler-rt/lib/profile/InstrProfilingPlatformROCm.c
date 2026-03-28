//===- InstrProfilingPlatformROCm.c - Profile data ROCm platform ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingPort.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/*  Platform abstraction for dynamic library loading                          */
/* -------------------------------------------------------------------------- */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef HMODULE DylibHandle;

static DylibHandle DylibOpen(const char *Name) { return LoadLibraryA(Name); }

static void *DylibSym(DylibHandle H, const char *Sym) {
  return (void *)(uintptr_t)GetProcAddress(H, Sym);
}

static int DylibAvailable(void) { return 1; }

#else /* POSIX */
#include <dlfcn.h>

/* Use weak references for dl* functions to avoid requiring -ldl at link time.
 *
 * The profile runtime is a static library, so its dependencies must be
 * explicitly linked by the user. Unlike sanitizer runtimes (which are often
 * shared libraries with their own dependencies), adding -ldl globally would
 * affect all profiling users, including those not using HIP/ROCm.
 *
 * With weak references:
 * - Programs without -ldl link successfully (dl* resolve to NULL)
 * - HIP programs get -ldl from the HIP runtime, so dl* work normally
 * - OpenMP offload programs without HIP gracefully skip device profiling
 */
#pragma weak dlopen
#pragma weak dlsym
#pragma weak dlerror

typedef void *DylibHandle;

static DylibHandle DylibOpen(const char *Name) {
  return dlopen(Name, RTLD_LAZY | RTLD_LOCAL);
}

static void *DylibSym(DylibHandle H, const char *Sym) { return dlsym(H, Sym); }

static int DylibAvailable(void) { return dlopen != NULL; }

#endif /* _WIN32 */

static int ProcessDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex,
                                   const char *Target);

static int IsVerboseMode() {
  static int IsVerbose = -1;
  if (IsVerbose == -1)
    IsVerbose = getenv("LLVM_PROFILE_VERBOSE") != NULL;
  return IsVerbose;
}

/* -------------------------------------------------------------------------- */
/*  Dynamic loading of HIP runtime symbols                                   */
/* -------------------------------------------------------------------------- */

typedef int (*hipGetSymbolAddressTy)(void **, const void *);
typedef int (*hipMemcpyTy)(void *, const void *, size_t, int);
typedef int (*hipModuleGetGlobalTy)(void **, size_t *, void *, const char *);
typedef int (*hipGetDeviceCountTy)(int *);
typedef int (*hipGetDeviceTy)(int *);
typedef int (*hipSetDeviceTy)(int);

/* hipDeviceProp_t layout for HIP 6.x+ (R0600).
 * We only need gcnArchName at offset 1160. Pad to 4096 to safely
 * accommodate future struct growth without recompilation. */
typedef struct {
  char padding[1160];
  char gcnArchName[256];
  char tail_padding[2680];
} HipDevicePropMinimal;
typedef int (*hipGetDevicePropertiesTy)(HipDevicePropMinimal *, int);

static hipGetSymbolAddressTy pHipGetSymbolAddress = NULL;
static hipMemcpyTy pHipMemcpy = NULL;
static hipModuleGetGlobalTy pHipModuleGetGlobal = NULL;
static hipGetDeviceCountTy pHipGetDeviceCount = NULL;
static hipGetDeviceTy pHipGetDevice = NULL;
static hipSetDeviceTy pHipSetDevice = NULL;
static hipGetDevicePropertiesTy pHipGetDeviceProperties = NULL;

#define MAX_DEVICES 16
static int NumDevices = 0;
static char DeviceArchNames[MAX_DEVICES][256];

/* -------------------------------------------------------------------------- */
/*  Device-to-host copies                                                     */
/*  Keep HIP-only to avoid an HSA dependency.                                 */
/* -------------------------------------------------------------------------- */

static void EnsureHipLoaded(void) {
  static int Initialized = 0;
  if (Initialized)
    return;
  Initialized = 1;

  if (!DylibAvailable()) {
    if (IsVerboseMode())
      PROF_NOTE("%s", "Dynamic library loading not available - "
                      "HIP profiling disabled\n");
    return;
  }

#ifdef _WIN32
  static const char HipLibName[] = "amdhip64.dll";
#else
  static const char HipLibName[] = "libamdhip64.so";
#endif

  DylibHandle Handle = DylibOpen(HipLibName);
  if (!Handle) {
#ifndef _WIN32
    if (dlerror)
      fprintf(stderr, "compiler-rt: failed to open %s: %s\n", HipLibName,
              dlerror());
#endif
    return;
  }

  pHipGetSymbolAddress =
      (hipGetSymbolAddressTy)DylibSym(Handle, "hipGetSymbolAddress");
  pHipMemcpy = (hipMemcpyTy)DylibSym(Handle, "hipMemcpy");
  pHipModuleGetGlobal =
      (hipModuleGetGlobalTy)DylibSym(Handle, "hipModuleGetGlobal");
  pHipGetDeviceCount =
      (hipGetDeviceCountTy)DylibSym(Handle, "hipGetDeviceCount");
  pHipGetDevice = (hipGetDeviceTy)DylibSym(Handle, "hipGetDevice");
  pHipSetDevice = (hipSetDeviceTy)DylibSym(Handle, "hipSetDevice");
  pHipGetDeviceProperties =
      (hipGetDevicePropertiesTy)DylibSym(Handle, "hipGetDevicePropertiesR0600");
  if (!pHipGetDeviceProperties)
    pHipGetDeviceProperties =
        (hipGetDevicePropertiesTy)DylibSym(Handle, "hipGetDeviceProperties");

  if (pHipGetDeviceCount && pHipGetDeviceProperties) {
    int Count = 0;
    if (pHipGetDeviceCount(&Count) == 0) {
      if (Count > MAX_DEVICES)
        Count = MAX_DEVICES;
      HipDevicePropMinimal Prop;
      for (int i = 0; i < Count; ++i) {
        memset(&Prop, 0, sizeof(Prop));
        if (pHipGetDeviceProperties(&Prop, i) == 0) {
          strncpy(DeviceArchNames[i], Prop.gcnArchName,
                  sizeof(DeviceArchNames[i]) - 1);
          DeviceArchNames[i][sizeof(DeviceArchNames[i]) - 1] = '\0';
          if (IsVerboseMode())
            PROF_NOTE("Device %d arch: %s\n", i, DeviceArchNames[i]);
        }
      }
      NumDevices = Count;
    }
  }
}

/* -------------------------------------------------------------------------- */
/*  Public wrappers that forward to the loaded HIP symbols                   */
/* -------------------------------------------------------------------------- */

static int hipGetSymbolAddress(void **devPtr, const void *symbol) {
  EnsureHipLoaded();
  return pHipGetSymbolAddress ? pHipGetSymbolAddress(devPtr, symbol) : -1;
}

static int hipMemcpy(void *dest, const void *src, size_t len,
                     int kind /*2=DToH*/) {
  EnsureHipLoaded();
  return pHipMemcpy ? pHipMemcpy(dest, src, len, kind) : -1;
}

/* Copy from device to host using HIP.
 * This requires that the device section symbols are registered with CLR,
 * otherwise hipMemcpy may attempt a CPU path and crash. */
static int memcpyDeviceToHost(void *Dst, const void *Src, size_t Size) {
  return hipMemcpy(Dst, Src, Size, 2 /* DToH */);
}

static int hipModuleGetGlobal(void **DevPtr, size_t *Bytes, void *Module,
                              const char *Name) {
  EnsureHipLoaded();
  return pHipModuleGetGlobal ? pHipModuleGetGlobal(DevPtr, Bytes, Module, Name)
                             : -1;
}

static int hipGetDevice(int *DeviceId) {
  EnsureHipLoaded();
  return pHipGetDevice ? pHipGetDevice(DeviceId) : -1;
}

static int hipSetDevice(int DeviceId) {
  EnsureHipLoaded();
  return pHipSetDevice ? pHipSetDevice(DeviceId) : -1;
}

static const char *getDeviceArchName(int DeviceId) {
  if (DeviceId < 0 || DeviceId >= NumDevices || !DeviceArchNames[DeviceId][0])
    return "amdgpu";
  return DeviceArchNames[DeviceId];
}

/* -------------------------------------------------------------------------- */
/*  Dynamic module tracking                                                   */
/* -------------------------------------------------------------------------- */

/* Per-TU profile entry inside a dynamic module.
 * A single dynamic module may contain multiple TUs (e.g. -fgpu-rdc). */
typedef struct {
  void *DeviceVar; /* device address of __llvm_offload_prf_<CUID>      */
  int Processed;   /* 0 = not yet collected, 1 = data already copied   */
} OffloadDynamicTUInfo;

/* One entry per hipModuleLoad call. */
typedef struct {
  void *ModulePtr;           /* hipModule_t handle                        */
  OffloadDynamicTUInfo *TUs; /* array of per-TU entries                 */
  int NumTUs;
  int CapTUs;
} OffloadDynamicModuleInfo;

static OffloadDynamicModuleInfo *DynamicModules = NULL;
static int NumDynamicModules = 0;
static int CapDynamicModules = 0;

/* -------------------------------------------------------------------------- */
/*  ELF symbol enumeration (Linux only)                                       */
/*                                                                            */
/*  AMDGPU code objects are always ELF, but <elf.h> is a Linux system header. */
/*  Dynamic module profiling (hipModuleLoadData) is currently Linux-only. */
/* -------------------------------------------------------------------------- */

#if defined(__linux__)
#include <elf.h>

/* Callback invoked for every matching symbol name found in the ELF image.
 * Return 0 to continue iteration, non-zero to stop. */
typedef int (*SymbolCallback)(const char *Name, void *UserData);

/* If Image is a clang offload bundle (__CLANG_OFFLOAD_BUNDLE__), find the
 * first embedded code object that is a valid ELF and return a pointer to it.
 * Otherwise return Image unchanged. Returns NULL if no ELF is found. */
static const void *UnwrapOffloadBundle(const void *Image) {
  static const char BundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  if (memcmp(Image, BundleMagic, 24) != 0)
    return Image; /* Not a bundle, return as-is. */

  const char *Buf = (const char *)Image;
  uint64_t NumEntries;
  memcpy(&NumEntries, Buf + 24, sizeof(uint64_t));

  /* Walk the entry table (starts at offset 32). */
  const char *Cursor = Buf + 32;
  for (uint64_t I = 0; I < NumEntries; ++I) {
    uint64_t EntryOffset, EntrySize, IDSize;
    memcpy(&EntryOffset, Cursor, 8);
    Cursor += 8;
    memcpy(&EntrySize, Cursor, 8);
    Cursor += 8;
    memcpy(&IDSize, Cursor, 8);
    Cursor += 8;
    /* Skip the entry ID string. */
    Cursor += IDSize;

    /* Check if this entry contains an ELF. */
    if (EntrySize >= sizeof(Elf64_Ehdr)) {
      const Elf64_Ehdr *E = (const Elf64_Ehdr *)(Buf + EntryOffset);
      if (E->e_ident[EI_MAG0] == ELFMAG0 && E->e_ident[EI_MAG1] == ELFMAG1 &&
          E->e_ident[EI_MAG2] == ELFMAG2 && E->e_ident[EI_MAG3] == ELFMAG3) {
        return (const void *)(Buf + EntryOffset);
      }
    }
  }

  PROF_WARN("%s", "Offload bundle contains no valid ELF entries\n");
  return NULL;
}

/* Parse an AMDGPU code-object ELF and invoke CB for every global symbol whose
 * name starts with PREFIX.  Image may be NULL (e.g. hipModuleLoad from file)
 * or a clang offload bundle containing an ELF;
 * in that case the function unwraps the bundle first. */
static void EnumerateElfSymbols(const void *Image, const char *Prefix,
                                SymbolCallback CB, void *UserData) {
  if (!Image)
    return;

  /* Handle clang offload bundle wrapping. */
  Image = UnwrapOffloadBundle(Image);
  if (!Image)
    return;

  const Elf64_Ehdr *Ehdr = (const Elf64_Ehdr *)Image;
  if (Ehdr->e_ident[EI_MAG0] != ELFMAG0 || Ehdr->e_ident[EI_MAG1] != ELFMAG1 ||
      Ehdr->e_ident[EI_MAG2] != ELFMAG2 || Ehdr->e_ident[EI_MAG3] != ELFMAG3) {
    if (IsVerboseMode())
      PROF_NOTE("%s", "Image is not a valid ELF, skipping enumeration\n");
    return;
  }

  size_t PrefixLen = strlen(Prefix);
  const char *Base = (const char *)Image;
  const Elf64_Shdr *Shdrs = (const Elf64_Shdr *)(Base + Ehdr->e_shoff);

  for (int i = 0; i < Ehdr->e_shnum; ++i) {
    if (Shdrs[i].sh_type != SHT_SYMTAB)
      continue;

    const Elf64_Sym *Syms = (const Elf64_Sym *)(Base + Shdrs[i].sh_offset);
    int NumSyms = Shdrs[i].sh_size / sizeof(Elf64_Sym);
    /* String table is the section referenced by sh_link. */
    const char *StrTab = Base + Shdrs[Shdrs[i].sh_link].sh_offset;

    for (int j = 0; j < NumSyms; ++j) {
      if (Syms[j].st_name == 0)
        continue;
      const char *Name = StrTab + Syms[j].st_name;
      if (strncmp(Name, Prefix, PrefixLen) == 0) {
        if (CB(Name, UserData))
          return;
      }
    }
  }
}

/* State passed through the enumeration callback. */
typedef struct {
  void *Module; /* hipModule_t */
  OffloadDynamicModuleInfo *ModInfo;
} EnumState;

/* Grow the TU array inside a module entry and register one __llvm_offload_prf_*
 * symbol. Also pre-registers the corresponding per-TU section symbols with CLR
 * (needed so hipMemcpy can copy from those device addresses later). */
static int RegisterPrfSymbol(const char *Name, void *UserData) {
  EnumState *S = (EnumState *)UserData;
  OffloadDynamicModuleInfo *MI = S->ModInfo;

  /* Look up the per-TU pointer variable, then dereference to get the
   * address of __llvm_profile_sections. */
  void *DevicePtrVar = NULL;
  size_t Bytes = 0;
  if (hipModuleGetGlobal(&DevicePtrVar, &Bytes, S->Module, Name) != 0) {
    PROF_WARN("Failed to get symbol %s for module %p\n", Name, S->Module);
    return 0; /* continue */
  }
  void *DeviceVar = NULL;
  if (hipMemcpy(&DeviceVar, DevicePtrVar, sizeof(void *), 2 /*DToH*/) != 0) {
    PROF_WARN("Failed to read sections pointer for %s\n", Name);
    return 0;
  }

  /* Grow TU array if needed. */
  if (MI->NumTUs >= MI->CapTUs) {
    int NewCap = MI->CapTUs ? MI->CapTUs * 2 : 4;
    OffloadDynamicTUInfo *New = (OffloadDynamicTUInfo *)realloc(
        MI->TUs, NewCap * sizeof(OffloadDynamicTUInfo));
    if (!New) {
      PROF_ERR("%s\n", "Failed to grow TU array");
      return 0;
    }
    MI->TUs = New;
    MI->CapTUs = NewCap;
  }
  OffloadDynamicTUInfo *TU = &MI->TUs[MI->NumTUs++];
  TU->DeviceVar = DeviceVar;
  TU->Processed = 0;

  (void)Name; /* CUID suffix available for future per-TU section lookup */

  return 0; /* continue enumeration */
}

#endif /* defined(__linux__) */

/* -------------------------------------------------------------------------- */
/*  Registration / un-registration helpers                                   */
/* -------------------------------------------------------------------------- */

void __llvm_profile_offload_register_dynamic_module(int ModuleLoadRc,
                                                    void **Ptr,
                                                    const void *Image) {
  if (IsVerboseMode())
    PROF_NOTE("Registering loaded module %d: rc=%d, module=%p, image=%p\n",
              NumDynamicModules, ModuleLoadRc, *Ptr, Image);

  if (ModuleLoadRc)
    return;

  if (NumDynamicModules >= CapDynamicModules) {
    int NewCap = CapDynamicModules ? CapDynamicModules * 2 : 64;
    OffloadDynamicModuleInfo *New = (OffloadDynamicModuleInfo *)realloc(
        DynamicModules, NewCap * sizeof(OffloadDynamicModuleInfo));
    if (!New)
      return;
    DynamicModules = New;
    CapDynamicModules = NewCap;
  }

  OffloadDynamicModuleInfo *MI = &DynamicModules[NumDynamicModules++];
  MI->ModulePtr = *Ptr;
  MI->TUs = NULL;
  MI->NumTUs = 0;
  MI->CapTUs = 0;

  /* Enumerate all __llvm_offload_prf_<CUID> symbols in the ELF image.
   * For each one, look it up via hipModuleGetGlobal (which also registers
   * the device address with CLR for later hipMemcpy) and store the entry.
   *
   * ELF parsing requires <elf.h> which is Linux-only.  On other platforms,
   * dynamic module profiling is not yet supported. */
#if defined(__linux__)
  EnumState State = {*Ptr, MI};
  EnumerateElfSymbols(Image, "__llvm_offload_prf_", RegisterPrfSymbol, &State);
#else
  (void)Image;
  if (IsVerboseMode())
    PROF_NOTE("%s",
              "Dynamic module profiling not supported on this platform\n");
#endif

  if (MI->NumTUs == 0) {
    PROF_WARN("No __llvm_offload_prf_* symbols found in module %p\n", *Ptr);
  } else if (IsVerboseMode()) {
    PROF_NOTE("Module %p: registered %d TU(s)\n", *Ptr, MI->NumTUs);
  }
}

void __llvm_profile_offload_unregister_dynamic_module(void *Ptr) {
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];

    if (MI->ModulePtr != Ptr)
      continue;

    if (IsVerboseMode())
      PROF_NOTE("Unregistering module %p (%d TUs)\n", MI->ModulePtr,
                MI->NumTUs);

    /* Process every TU in this module. */
    for (int t = 0; t < MI->NumTUs; ++t) {
      OffloadDynamicTUInfo *TU = &MI->TUs[t];
      if (TU->Processed) {
        if (IsVerboseMode())
          PROF_NOTE("Module %p TU %d already processed, skipping\n", Ptr, t);
        continue;
      }
      /* Use a globally unique index as TU index for the output filename. */
      int TUIndex = i * 1000 + t;
      if (TU->DeviceVar) {
        int CurDev = 0;
        hipGetDevice(&CurDev);
        const char *ArchName = getDeviceArchName(CurDev);
        if (ProcessDeviceOffloadPrf(TU->DeviceVar, TUIndex, ArchName) == 0)
          TU->Processed = 1;
        else
          PROF_WARN("Failed to process profile data for module %p TU %d\n", Ptr,
                    t);
      }
    }
    return;
  }

  if (IsVerboseMode())
    PROF_WARN("Unregister called for unknown module %p\n", Ptr);
}

/* Grow a void* array, doubling capacity (or starting at InitCap). */
static int GrowPtrArray(void ***Arr, int *Num, int *Cap, int InitCap) {
  if (*Num < *Cap)
    return 0;
  int NewCap = *Cap ? *Cap * 2 : InitCap;
  void **New = (void **)realloc(*Arr, NewCap * sizeof(void *));
  if (!New)
    return -1;
  *Arr = New;
  *Cap = NewCap;
  return 0;
}

static void **OffloadShadowVariables = NULL;
static int NumShadowVariables = 0;
static int CapShadowVariables = 0;

void __llvm_profile_offload_register_shadow_variable(void *ptr) {
  if (GrowPtrArray(&OffloadShadowVariables, &NumShadowVariables,
                   &CapShadowVariables, 64))
    return;
  OffloadShadowVariables[NumShadowVariables++] = ptr;
}

static void **OffloadSectionShadowVariables = NULL;
static int NumSectionShadowVariables = 0;
static int CapSectionShadowVariables = 0;

void __llvm_profile_offload_register_section_shadow_variable(void *ptr) {
  if (GrowPtrArray(&OffloadSectionShadowVariables, &NumSectionShadowVariables,
                   &CapSectionShadowVariables, 64))
    return;
  OffloadSectionShadowVariables[NumSectionShadowVariables++] = ptr;
}

static int ProcessDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex,
                                   const char *Target) {
  __llvm_profile_gpu_sections HostSections;

  if (hipMemcpy(&HostSections, DeviceOffloadPrf, sizeof(HostSections),
                2 /*DToH*/) != 0) {
    PROF_ERR("%s\n", "Failed to copy offload prf structure from device");
    return -1;
  }

  const void *DevCntsBegin = HostSections.CountersStart;
  const void *DevDataBegin = HostSections.DataStart;
  const void *DevNamesBegin = HostSections.NamesStart;
  const void *DevUniformCntsBegin = HostSections.UniformCountersStart;
  const void *DevCntsEnd = HostSections.CountersStop;
  const void *DevDataEnd = HostSections.DataStop;
  const void *DevNamesEnd = HostSections.NamesStop;
  const void *DevUniformCntsEnd = HostSections.UniformCountersStop;

  size_t CountersSize = (const char *)DevCntsEnd - (const char *)DevCntsBegin;
  size_t DataSize = (const char *)DevDataEnd - (const char *)DevDataBegin;
  size_t NamesSize = (const char *)DevNamesEnd - (const char *)DevNamesBegin;
  size_t UniformCountersSize =
      (const char *)DevUniformCntsEnd - (const char *)DevUniformCntsBegin;

  if (IsVerboseMode())
    PROF_NOTE("Section pointers: Cnts=[%p,%p]=%zu Data=[%p,%p]=%zu "
              "Names=[%p,%p]=%zu UCnts=[%p,%p]=%zu\n",
              DevCntsBegin, DevCntsEnd, CountersSize, DevDataBegin, DevDataEnd,
              DataSize, DevNamesBegin, DevNamesEnd, NamesSize,
              DevUniformCntsBegin, DevUniformCntsEnd, UniformCountersSize);

  if (CountersSize == 0 || DataSize == 0)
    return 0;

  int ret = -1;
  int NamesReused = 0, CntsReused = 0, UCntsReused = 0, DataReused = 0;

  char *HostDataBegin = NULL;
  char *HostCountersBegin = NULL;
  char *HostNamesBegin = NULL;
  char *HostUniformCountersBegin = NULL;

  /* Sections using linker-defined __start_/__stop_ bounds are shared across
     TU structs in RDC mode. Deduplicate by caching the last copied range. */
  static const void *CachedDevNamesBegin = NULL;
  static char *CachedHostNames = NULL;
  static size_t CachedNamesSize = 0;

  static const void *CachedDevCntsBegin = NULL;
  static char *CachedHostCnts = NULL;
  static size_t CachedCntsSize = 0;

  static const void *CachedDevDataBegin = NULL;
  static char *CachedHostData = NULL;
  static size_t CachedDataSize = 0;

  static const void *CachedDevUCntsBegin = NULL;
  static char *CachedHostUCnts = NULL;
  static size_t CachedUCntsSize = 0;

  if (CountersSize > 0 && DevCntsBegin == CachedDevCntsBegin &&
      CountersSize == CachedCntsSize) {
    HostCountersBegin = CachedHostCnts;
    CntsReused = 1;
    if (IsVerboseMode())
      PROF_NOTE("Reusing cached counters section (%zu bytes)\n", CountersSize);
  } else if (CountersSize > 0) {
    HostCountersBegin = (char *)malloc(CountersSize);
  }

  if (DataSize > 0 && DevDataBegin == CachedDevDataBegin &&
      DataSize == CachedDataSize) {
    HostDataBegin = CachedHostData;
    DataReused = 1;
    if (IsVerboseMode())
      PROF_NOTE("Reusing cached data section (%zu bytes)\n", DataSize);
  } else if (DataSize > 0) {
    HostDataBegin = (char *)malloc(DataSize);
  }

  if (NamesSize > 0 && DevNamesBegin == CachedDevNamesBegin &&
      NamesSize == CachedNamesSize) {
    HostNamesBegin = CachedHostNames;
    NamesReused = 1;
    if (IsVerboseMode())
      PROF_NOTE("Reusing cached names section (%zu bytes)\n", NamesSize);
  } else if (NamesSize > 0) {
    HostNamesBegin = (char *)malloc(NamesSize);
  }

  if (UniformCountersSize > 0 && DevUniformCntsBegin == CachedDevUCntsBegin &&
      UniformCountersSize == CachedUCntsSize) {
    HostUniformCountersBegin = CachedHostUCnts;
    UCntsReused = 1;
    if (IsVerboseMode())
      PROF_NOTE("Reusing cached ucnts section (%zu bytes)\n",
                UniformCountersSize);
  } else if (UniformCountersSize > 0) {
    HostUniformCountersBegin = (char *)malloc(UniformCountersSize);
  }

  if ((DataSize > 0 && !HostDataBegin) ||
      (CountersSize > 0 && !HostCountersBegin) ||
      (NamesSize > 0 && !HostNamesBegin) ||
      (UniformCountersSize > 0 && !HostUniformCountersBegin)) {
    PROF_ERR("%s\n", "Failed to allocate host memory for device sections");
    goto cleanup;
  }

  if ((DataSize > 0 && !DataReused &&
       memcpyDeviceToHost(HostDataBegin, DevDataBegin, DataSize) != 0) ||
      (CountersSize > 0 && !CntsReused &&
       memcpyDeviceToHost(HostCountersBegin, DevCntsBegin, CountersSize) !=
           0) ||
      (NamesSize > 0 && !NamesReused &&
       memcpyDeviceToHost(HostNamesBegin, DevNamesBegin, NamesSize) != 0) ||
      (UniformCountersSize > 0 && !UCntsReused &&
       memcpyDeviceToHost(HostUniformCountersBegin, DevUniformCntsBegin,
                          UniformCountersSize) != 0)) {
    PROF_ERR("%s\n", "Failed to copy profile sections from device");
    goto cleanup;
  }

  if (!CntsReused && CountersSize > 0) {
    CachedDevCntsBegin = DevCntsBegin;
    CachedHostCnts = HostCountersBegin;
    CachedCntsSize = CountersSize;
  }
  if (!DataReused && DataSize > 0) {
    CachedDevDataBegin = DevDataBegin;
    CachedHostData = HostDataBegin;
    CachedDataSize = DataSize;
  }
  if (!NamesReused && NamesSize > 0) {
    CachedDevNamesBegin = DevNamesBegin;
    CachedHostNames = HostNamesBegin;
    CachedNamesSize = NamesSize;
  }
  if (!UCntsReused && UniformCountersSize > 0) {
    CachedDevUCntsBegin = DevUniformCntsBegin;
    CachedHostUCnts = HostUniformCountersBegin;
    CachedUCntsSize = UniformCountersSize;
  }

  if (IsVerboseMode())
    PROF_NOTE("Copied device sections: Counters=%zu, Data=%zu, Names=%zu, "
              "UniformCounters=%zu\n",
              CountersSize, DataSize, NamesSize, UniformCountersSize);

  // Arrange buffer as [Data][Padding][Counters][Names] to match the layout
  // expected by lprofWriteDataImpl (CountersDelta = CountersBegin - DataBegin).
  const uint64_t NumData = DataSize / sizeof(__llvm_profile_data);
  const uint64_t NumBitmapBytes = 0;
  const uint64_t NumUniformCounters = UniformCountersSize / sizeof(uint64_t);
  const uint64_t VTableSectionSize = 0;
  const uint64_t VNamesSize = 0;
  uint64_t PaddingBytesBeforeCounters, PaddingBytesAfterCounters,
      PaddingBytesAfterBitmapBytes, PaddingBytesAfterUniformCounters,
      PaddingBytesAfterNames, PaddingBytesAfterVTable, PaddingBytesAfterVNames;

  if (__llvm_profile_get_padding_sizes_for_counters(
          DataSize, CountersSize, NumBitmapBytes, NumUniformCounters, NamesSize,
          VTableSectionSize, VNamesSize, &PaddingBytesBeforeCounters,
          &PaddingBytesAfterCounters, &PaddingBytesAfterBitmapBytes,
          &PaddingBytesAfterUniformCounters, &PaddingBytesAfterNames,
          &PaddingBytesAfterVTable, &PaddingBytesAfterVNames) != 0) {
    PROF_ERR("%s\n", "Failed to get padding sizes");
    goto cleanup;
  }

  size_t ContiguousBufferSize =
      DataSize + PaddingBytesBeforeCounters + CountersSize + NamesSize;
  char *ContiguousBuffer = (char *)malloc(ContiguousBufferSize);
  if (!ContiguousBuffer) {
    PROF_ERR("%s\n", "Failed to allocate contiguous buffer");
    goto cleanup;
  }
  memset(ContiguousBuffer, 0, ContiguousBufferSize);

  char *BufDataBegin = ContiguousBuffer;
  char *BufCountersBegin =
      ContiguousBuffer + DataSize + PaddingBytesBeforeCounters;
  char *BufNamesBegin = BufCountersBegin + CountersSize;

  memcpy(BufDataBegin, HostDataBegin, DataSize);
  memcpy(BufCountersBegin, HostCountersBegin, CountersSize);
  memcpy(BufNamesBegin, HostNamesBegin, NamesSize);

  // Reorder uniform counters to match data record order. With per-function
  // counter layout, the linker may reorder __llvm_prf_cnts and
  // __llvm_prf_ucnts independently. Use UniformCounterPtr from each data
  // record to extract the right uniform counters in data-record order,
  // so that llvm-profdata's sequential walk works correctly.
  char *ReorderedUniformCounters = NULL;
  size_t ReorderedUniformSize = 0;
  __llvm_profile_data *RawData = (__llvm_profile_data *)HostDataBegin;
  if (HostUniformCountersBegin && NumData > 0) {
    size_t TotalCounters = 0;
    for (uint64_t i = 0; i < NumData; ++i)
      TotalCounters += RawData[i].NumCounters;
    ReorderedUniformSize = TotalCounters * sizeof(uint64_t);
    ReorderedUniformCounters = (char *)calloc(1, ReorderedUniformSize);
    if (ReorderedUniformCounters) {
      size_t DstOffset = 0;
      for (uint64_t i = 0; i < NumData; ++i) {
        uint32_t NC = RawData[i].NumCounters;
        ptrdiff_t UCPtrOff = (ptrdiff_t)RawData[i].UniformCounterPtr;
        if (NC > 0 && UCPtrOff != 0) {
          const char *DevDataAddr =
              (const char *)DevDataBegin + (i * sizeof(__llvm_profile_data));
          ptrdiff_t SrcOff =
              (DevDataAddr + UCPtrOff) - (const char *)DevUniformCntsBegin;
          if (SrcOff >= 0 &&
              (size_t)(SrcOff + NC * sizeof(uint64_t)) <= UniformCountersSize)
            memcpy(ReorderedUniformCounters + DstOffset,
                   HostUniformCountersBegin + SrcOff, NC * sizeof(uint64_t));
        }
        DstOffset += NC * sizeof(uint64_t);
      }
    }
  }

  // Relocate CounterPtr in data records for file layout.
  // CounterPtr is device-relative offset; adjust for file layout where
  // Data section comes first, then Counters section.
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)BufDataBegin;
  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      const char *DeviceDataStructAddr =
          (const char *)DevDataBegin + (i * sizeof(__llvm_profile_data));
      const char *DeviceCountersAddr =
          DeviceDataStructAddr + DeviceCounterPtrOffset;
      ptrdiff_t OffsetIntoCountersSection =
          DeviceCountersAddr - (const char *)DevCntsBegin;

      ptrdiff_t NewRelativeOffset = DataSize + PaddingBytesBeforeCounters +
                                    OffsetIntoCountersSection -
                                    (i * sizeof(__llvm_profile_data));
      memcpy((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                 offsetof(__llvm_profile_data, CounterPtr),
             &NewRelativeOffset, sizeof(NewRelativeOffset));
    }
    {
      ptrdiff_t Zero = 0;
      memcpy((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                 offsetof(__llvm_profile_data, UniformCounterPtr),
             &Zero, sizeof(Zero));
    }
    memset((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
               offsetof(__llvm_profile_data, BitmapPtr),
           0,
           sizeof(RelocatedData[i].BitmapPtr) +
               sizeof(RelocatedData[i].FunctionPointer) +
               sizeof(RelocatedData[i].Values));
  }

  // Relocate UniformCounterPtr for file layout. The ucnts section sits at
  // offset UCFileOffset from the data section start. After reordering, function
  // i's ucnts start at cumulative offset within the ucnts section.
  ptrdiff_t UCFileOffset = DataSize + PaddingBytesBeforeCounters +
                           CountersSize + PaddingBytesAfterCounters + 0 +
                           PaddingBytesAfterBitmapBytes;
  if (HostUniformCountersBegin) {
    size_t CumulativeUCOffset = 0;
    for (uint64_t i = 0; i < NumData; ++i) {
      ptrdiff_t NewUCRelativeOffset =
          UCFileOffset + (ptrdiff_t)CumulativeUCOffset -
          (ptrdiff_t)(i * sizeof(__llvm_profile_data));
      memcpy((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                 offsetof(__llvm_profile_data, UniformCounterPtr),
             &NewUCRelativeOffset, sizeof(NewUCRelativeOffset));
      CumulativeUCOffset += RelocatedData[i].NumCounters * sizeof(uint64_t);
    }
  }

  char TUIndexStr[16];
  snprintf(TUIndexStr, sizeof(TUIndexStr), "%d", TUIndex);

  char *UCBegin = ReorderedUniformCounters ? ReorderedUniformCounters
                                           : HostUniformCountersBegin;
  size_t UCSize =
      ReorderedUniformCounters ? ReorderedUniformSize : UniformCountersSize;

  ret = __llvm_write_custom_profile(
      Target, TUIndexStr, (__llvm_profile_data *)BufDataBegin,
      (__llvm_profile_data *)(BufDataBegin + DataSize), BufCountersBegin,
      BufCountersBegin + CountersSize, UCBegin,
      UCBegin ? UCBegin + UCSize : NULL, BufNamesBegin,
      BufNamesBegin + NamesSize, NULL);

  free(ContiguousBuffer);
  free(ReorderedUniformCounters);

  if (ret != 0) {
    PROF_ERR("%s\n", "Failed to write device profile using shared API");
  } else if (IsVerboseMode()) {
    PROF_NOTE("%s\n", "Successfully wrote device profile using shared API");
  }

cleanup:
  if (!CntsReused)
    free(HostCountersBegin);
  if (!DataReused)
    free(HostDataBegin);
  if (!NamesReused)
    free(HostNamesBegin);
  if (!UCntsReused)
    free(HostUniformCountersBegin);
  return ret;
}

static int ProcessShadowVariable(void *ShadowVar, int TUIndex,
                                 const char *Target) {
  void *DevicePtrVar = NULL;
  if (hipGetSymbolAddress(&DevicePtrVar, ShadowVar) != 0) {
    PROF_WARN("Failed to get symbol address for shadow variable %p\n",
              ShadowVar);
    return -1;
  }
  // The shadow variable is a pointer to __llvm_profile_sections (defined
  // in the GPU profile runtime). Dereference to get the struct address.
  void *DeviceOffloadPrf = NULL;
  if (hipMemcpy(&DeviceOffloadPrf, DevicePtrVar, sizeof(void *), 2 /*DToH*/) !=
      0) {
    PROF_WARN("Failed to read sections pointer from shadow variable %p\n",
              ShadowVar);
    return -1;
  }
  return ProcessDeviceOffloadPrf(DeviceOffloadPrf, TUIndex, Target);
}

/* Check if HIP runtime is available and loaded */
static int IsHipAvailable(void) {
  EnsureHipLoaded();
  return pHipMemcpy != NULL && pHipGetSymbolAddress != NULL;
}

/* -------------------------------------------------------------------------- */
/*  Collect device-side profile data                                          */
/* -------------------------------------------------------------------------- */

int __llvm_profile_hip_collect_device_data(void) {
  if (NumShadowVariables == 0 && NumDynamicModules == 0)
    return 0;

  if (!IsHipAvailable())
    return 0;

  int Ret = 0;

  /* Shadow variables (static-linked kernels).
   * Iterate over all devices to collect profile data from each GPU. */
  if (NumShadowVariables > 0) {
    int OrigDevice = -1;
    hipGetDevice(&OrigDevice);

    for (int Dev = 0; Dev < NumDevices; ++Dev) {
      if (hipSetDevice(Dev) != 0) {
        if (IsVerboseMode())
          PROF_NOTE("Failed to set device %d, skipping\n", Dev);
        continue;
      }
      const char *ArchName = getDeviceArchName(Dev);
      if (IsVerboseMode())
        PROF_NOTE("Collecting static profile data from device %d (%s)\n", Dev,
                  ArchName);
      for (int i = 0; i < NumShadowVariables; ++i) {
        if (ProcessShadowVariable(OffloadShadowVariables[i], i, ArchName) != 0)
          Ret = -1;
      }
    }

    if (OrigDevice >= 0)
      hipSetDevice(OrigDevice);
  }

  /* Dynamically-loaded modules — warn about any unprocessed TUs */
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];
    for (int t = 0; t < MI->NumTUs; ++t) {
      if (!MI->TUs[t].Processed) {
        PROF_WARN("Dynamic module %p TU %d was not processed before exit\n",
                  MI->ModulePtr, t);
        Ret = -1;
      }
    }
  }

  return Ret;
}
