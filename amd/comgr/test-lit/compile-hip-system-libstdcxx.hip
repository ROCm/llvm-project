// COM: System libstdc++ must not be mixed with Comgr's embedded libc++.
// RUN: compile-hip-cxx-headers --mode=libstdcxx-conflict %s | %FileCheck %s

// CHECK: RESULT: {{PASS|SKIPPED}}

#include <array>
#include <exception>
#include <stdexcept>

static_assert(std::tuple_size<std::array<int, 3>>::value == 3, "array size");

extern "C" __attribute__((global)) void test_kernel(int *out) {
  std::array<int, 3> a = {1, 2, 3};
  out[0] = a[0] + a[1] + a[2];
}
