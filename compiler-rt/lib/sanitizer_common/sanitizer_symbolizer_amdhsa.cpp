//===-- sanitizer_symbolizer_amdhsa.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Symbolizes AMDGPU code objects through the AMD Code Object Manager (Comgr).
//
//===----------------------------------------------------------------------===//
#include "sanitizer_platform.h"

#if SANITIZER_AMDHSA
#  include <dlfcn.h>  // For dlopen, dlsym

#  include "sanitizer_atomic.h"
#  include "sanitizer_mutex.h"
#  include "sanitizer_symbolizer_amdhsa.h"

namespace __sanitizer {

static COMgrFunctions comgr;

// Comgr handle
static void* comgr_handle = nullptr;
// Mutex to protect the Comgr handle
static StaticSpinMutex comgr_init_mu;

// Default Comgr library path
static const char kDefaultComgrPath[] = "libamd_comgr.so";

static atomic_uint8_t comgr_symbols_loaded{0};

static bool ComgrSymbolsLoaded() {
  return atomic_load(&comgr_symbols_loaded, memory_order_acquire) != 0;
}

static void getSourceLocation(const char* Result, void* ScopedString) {
  InternalScopedString* ScopedStringObj = (InternalScopedString*)ScopedString;
  ScopedStringObj->Append(Result);
}

// Use an explicit dlopen handle: nothing requires the process to link
// libamd_comgr, and RTLD_NEXT only searches objects loaded after this one.
// Symbolization runs at report time, so dlopen is safe here.
void AMDGPUCodeObjectSymbolizer::InitCOMgr() {
  SpinMutexLock l(&comgr_init_mu);
  if (ComgrSymbolsLoaded())
    return;

  if (!comgr_handle) {
    const char* path = GetEnv("SANITIZER_COMGR_PATH");
    if (!path || !path[0])
      path = kDefaultComgrPath;
    // Whoever loaded the code object we are about to symbolize has usually
    // mapped Comgr already, so prefer `RTLD_NOLOAD`. Fall back to `RTLD_NOW`
    // for the case where it is not yet loaded.
    comgr_handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (!comgr_handle)
      comgr_handle = dlopen(path, RTLD_NOW);
    if (!comgr_handle) {
      const char* err = dlerror();
      VReport(2, "Comgr Init: dlopen(%s) failed: %s\n", path,
              err ? err : "unknown error");
      return;
    }
  }

  struct ComgrSymbolEntry {
    const char* name;
    void** slot;
  };
  // Comgr Symbol Entries Table
  const ComgrSymbolEntry kSymbols[] = {
      {"amd_comgr_create_data", (void**)&comgr.create_data},
      {"amd_comgr_set_data", (void**)&comgr.set_data},
      {"amd_comgr_set_data_from_file_slice",
       (void**)&comgr.set_data_from_file_slice},
      {"amd_comgr_create_symbolizer_info", (void**)&comgr.create_symbolizer},
      {"amd_comgr_symbolize", (void**)&comgr.symbolize},
      {"amd_comgr_destroy_symbolizer_info", (void**)&comgr.destroy_symbolizer},
      {"amd_comgr_release_data", (void**)&comgr.release_data},
  };

  bool success = true;
  for (uptr i = 0; i < ARRAY_SIZE(kSymbols); ++i) {
    void* sym = dlsym(comgr_handle, kSymbols[i].name);
    if (!sym) {
      VReport(2, "Comgr Init: Failed to load %s from Comgr handle\n",
              kSymbols[i].name);
      success = false;
    }
    *kSymbols[i].slot = sym;
  }
  if (!success) {
    internal_memset(&comgr, 0, sizeof(comgr));
    VReport(1, "Comgr Init: Failed to load Comgr symbolizer functions\n");
    return;
  }
  atomic_store(&comgr_symbols_loaded, 1, memory_order_release);
}

void AMDGPUCodeObjectSymbolizer::Init(int fd, u64 off, u64 size) {
  InitCOMgr();
  if (!ComgrSymbolsLoaded())
    return;

  if (comgr.create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &codeobject))
    return;

  object_cnt = comgr_objects::data;

  if (fd != -1) {
    if (comgr.set_data_from_file_slice(codeobject, fd, off, size)) {
      Release();
      return;
    }
  } else {
    // fd == -1 means the code object is already resident in host memory and
    // `off` is the address of its first byte.
    if (comgr.set_data(codeobject, size, reinterpret_cast<const char*>(off))) {
      Release();
      return;
    }
  }

  if (comgr.create_symbolizer(codeobject, &getSourceLocation, &symbolizer)) {
    Release();
    return;
  }

  object_cnt = comgr_objects::data_and_symb;
  init = true;
}

bool AMDGPUCodeObjectSymbolizer::SymbolizePC(uptr addr,
                                             InternalScopedString& source_loc) {
  if (!init)
    return false;
  comgr.symbolize(symbolizer, addr, true, (void*)&source_loc);
  return true;
}

void AMDGPUCodeObjectSymbolizer::Release() {
  // fall-through is avoided to silence warnings.
  switch (object_cnt) {
    case comgr_objects::data_and_symb: {
      comgr.destroy_symbolizer(symbolizer);
      comgr.release_data(codeobject);
      break;
    }
    case comgr_objects::data: {
      comgr.release_data(codeobject);
      break;
    }
    default: {
    }
  }
  // Init() calls Release() on a partial failure, so leave the object empty
  // rather than releasing the same handles again from the caller's Release().
  object_cnt = comgr_objects::no_objs;
  init = false;
}
}  // namespace __sanitizer
#endif  // SANITIZER_AMDHSA
