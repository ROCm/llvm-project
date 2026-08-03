// COM: System libc++ should only suppress embedded libc++ when requested.
// REQUIRES: system-libcxx
// RUN: compile-hip-cxx-headers --mode=system-libcxx %s | %FileCheck %s

// CHECK: RESULT: PASS

#include <cstddef>

static_assert(sizeof(std::size_t) >= sizeof(unsigned), "size_t size");

extern "C" __attribute__((global)) void test_kernel(int *out) { out[0] = 1; }
