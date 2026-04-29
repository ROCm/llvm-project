//===-- lib/openmp/omp_alloc.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define ALLOC_DEBUG 1

#include "flang/Runtime/OpenMP/omp_alloc.h"
#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/OpenMP/omp_util.h"
#include "flang/Support/Fortran.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

namespace Fortran::runtime::omp {

static bool debugEnabled;

// ====================== OPENMP ======================

// Declare OpenMP memory management routines to avoid importing
// definitions via "omp.h" (and thus create a dependency to the
// OpenMP runtime library code).
extern "C" int omp_get_default_device(void);
extern "C" void *omp_target_alloc(std::size_t, int);
extern "C" void omp_target_free(void *, int);

// Track which device each pointer was allocated on so that
// OpenMPFree can pass the correct device ID to omp_target_free,
// even if omp_set_default_device() was called between ALLOCATE
// and DEALLOCATE.
static PointerDeviceMap allocDeviceMap;

/// Allocate \p AllocationSize bytes on the current default OpenMP device.
static void *OpenMPAlloc(std::size_t AllocationSize, std::int64_t *) {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[OMP_ALLOC] %s(%zu) (%s:%d)\n", __PRETTY_FUNCTION__,
        AllocationSize, __FILE__, __LINE__);
  }
#endif
  int device{omp_get_default_device()};
  void *pointer{omp_target_alloc(AllocationSize, device)};
  if (pointer) {
    allocDeviceMap.insert(pointer, device);
  }
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr,
        "[OMP_ALLOC] pointer of size %zu allocated at %p"
        " on device %d.\n",
        AllocationSize, pointer, device);
  }
#endif
  return pointer;
}

/// Free a pointer previously allocated by OpenMPAlloc on the correct device.
static void OpenMPFree(void *pointer) {
  int device{allocDeviceMap.removeAndGet(pointer)};
  if (device == -1) {
    Terminator{__FILE__, __LINE__}.Crash(
        "OpenMPFree: pointer %p was not allocated by OpenMPAlloc", pointer);
  }
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[OMP_ALLOC] %s(%p) device %d (%s:%d)\n",
        __PRETTY_FUNCTION__, pointer, device, __FILE__, __LINE__);
  }
#endif
  omp_target_free(pointer, device);
}

/// Register the OpenMP alloc/free pair in the runtime's allocator registry.
static void registerOpenMPAllocator() {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(
        stderr, "[OMP_ALLOC] registering OpenMP device memory allocator\n");
  }
#endif // ALLOC_DEBUG
  allocatorRegistry.Register(1, {&OpenMPAlloc, &OpenMPFree});
}

/// Return the value of environment variable \p envirable, or \p defaultValue.
static const char *getStringFromEnvironment(
    const char *envirable, const char *defaultValue = "") {
  if (auto value{std::getenv(envirable)}) {
    return value;
  }
  return defaultValue;
}

/// Return the integer value of environment variable \p envirable, or \p defaultValue.
static int getIntFromEnvironment(
    const char *envirable, const int defaultValue = 0) {
  int result = defaultValue;
  char *end;
  if (auto value{std::getenv(envirable)}) {
    auto number{std::strtoul(value, &end, 10)};
    if (number > 0 && number < std::numeric_limits<int>::max() &&
        *end == '\0') {
      result = number;
    } else {
      std::fprintf(stderr, "Fortran runtime: %s=%s is invalid; ignored\n",
          envirable, value);
    }
  }
  return result;
}

/// Split \p str at the first ':' into (before, after). If no colon, the
/// second element is empty.
static std::pair<std::string_view, std::string_view> splitAtColon(
    std::string_view str) {
  const char *data = str.data();
  size_t len = str.size();
  const char *colon = static_cast<const char *>(std::memchr(data, ':', len));
  if (!colon) {
    return {str, std::string_view()};
  }
  size_t colon_pos = colon - data;
  return {std::string_view(data, colon_pos),
      std::string_view(colon + 1, len - colon_pos - 1)};
}

extern "C" {
void RTDEF(OpenMPRegisterAllocator)() {
#if ALLOC_DEBUG
  debugEnabled = false;
  if (getIntFromEnvironment("OMP_ALLOC_DEBUG", 0) != 0) {
    debugEnabled = true;
  }
  if (debugEnabled) {
    std::fprintf(stderr, "[OMP_ALLOC] %s (%s:%d)\n", __PRETTY_FUNCTION__,
        __FILE__, __LINE__);
  }
#endif

  // Determine what allocator to register via very simplistic parsing of syntax
  // ALLOCATOR:MEMORY_KIND.  Proper values are: OPENMP
  const char *allocator_env = getStringFromEnvironment("OMP_ALLOC", "openmp");
  char allocator[256];
  std::strncpy(allocator, allocator_env, sizeof(allocator) - 1);
  allocator[sizeof(allocator) - 1] = '\0';
  for (char *p = allocator; *p; ++p)
    *p = ::toupper(*p);
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[OMP_ALLOC] requesting allocator: %s\n", allocator);
  }
#endif // ALLOC_DEBUG
  std::pair<std::string_view, std::string_view> allocSpec{
      splitAtColon(allocator)};
  if (allocSpec.first != "OPENMP") {
    std::fprintf(stderr,
        "[OMP_ALLOC] warning: wrong allocator ('%.*s') specified, "
        "using 'OPENMP' instead.\n",
        static_cast<int>(allocSpec.first.size()), allocSpec.first.data());
    allocSpec.first = "OPENMP";
  }
  if (allocSpec.first == "OPENMP") {
    if (!allocSpec.second.empty()) {
      std::fprintf(stderr,
          "[OMP_ALLOC] warning: OpenMP allocator does not "
          "accept allocator option type '%.*s'.\n",
          static_cast<int>(allocSpec.second.size()), allocSpec.second.data());
    }
    registerOpenMPAllocator();
  }
}

void RTDEF(OpenMPAllocatableSetAllocIdx)(Descriptor &descriptor, int pos) {
  if (descriptor.IsAllocatable() && !descriptor.IsAllocated()) {
#if ALLOC_DEBUG
    if (debugEnabled) {
      std::fprintf(
          stderr, "[OMP_ALLOC] OpenMPAllocatableSetAllocIdx = %d \n", pos);
    }
#endif
    descriptor.SetAllocIdx(pos);
  }
}
} // extern "C"

} // namespace Fortran::runtime::omp
