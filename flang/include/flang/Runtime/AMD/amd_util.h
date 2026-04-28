//===-- include/flang/Runtime/AMD/amd_util.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_AMD_UTIL_H_
#define FORTRAN_RUNTIME_AMD_UTIL_H_

#include <cstddef>

namespace Fortran::runtime::amd {

// A simple thread-safe map from pointer to integer value (e.g. device ID).
// Implemented as a dynamically-grown array with linear search to avoid
// pulling in C++ runtime dependencies via std::unordered_map.
class PointerDeviceMap {
public:
  // Record that |pointer| is associated with |device|.
  void insert(void *pointer, int device);

  // Look up and remove the entry for |pointer|.  Returns the associated
  // device ID, or -1 if the pointer was not found.
  int removeAndGet(void *pointer);

  // Dump the pointer-device table
  void dump() const;

private:
  struct Entry {
    void *pointer;
    int device;
  };

  void grow();

  Entry *entries_{nullptr};
  std::size_t count_{0};
  std::size_t capacity_{0};
};

} // namespace Fortran::runtime::amd

#endif // FORTRAN_RUNTIME_AMD_UTIL_H_
