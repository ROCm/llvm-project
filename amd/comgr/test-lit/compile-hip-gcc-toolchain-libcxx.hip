// REQUIRES: system-linux

// COM: An explicit --gcc-toolchain should prevent fallback to sysroot or host
// COM: C++ headers when deciding whether to use embedded libc++.
// RUN: rm -rf %t.sysroot %t.gcc-toolchain
// RUN: mkdir -p %t.sysroot/usr/include/c++/999 %t.gcc-toolchain
// RUN: touch %t.sysroot/usr/include/c++/999/cstddef
// RUN: compile-hip-cxx-headers --mode=gcc-toolchain %s \
// RUN:   %t.sysroot %t.gcc-toolchain | %FileCheck %s

// CHECK: Embedded libc++ headers: active
// CHECK: RESULT: PASS

#include <__config>

#ifndef _LIBCPP_VERSION
#error "expected libc++ header"
#endif

extern "C" __attribute__((global)) void test_kernel(int *out) {
  out[0] = 42;
}
