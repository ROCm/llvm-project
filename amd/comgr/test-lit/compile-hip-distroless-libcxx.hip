// COM: Force Comgr's embedded libc++ fallback path.
// RUN: compile-hip-cxx-headers --mode=distroless %s | %FileCheck %s

// CHECK: Embedded libc++ headers: active
// CHECK: RESULT: PASS

#include <__config>

#ifndef _LIBCPP_VERSION
#error "expected libc++ header"
#endif

extern "C" __attribute__((global)) void test_kernel(int *out) {
  out[0] = 42;
}
