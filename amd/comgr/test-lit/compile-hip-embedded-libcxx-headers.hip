// COM: The embedded libc++ header set must be non-empty on every host. When
// COM: cmake/trace_headers.py mishandled host path separators the set came out
// COM: empty, the build still succeeded, and the failure only surfaced at
// COM: runtime as "'type_traits' file not found". Not gated on a platform: the
// COM: missing coverage on non-Linux hosts is what let that ship.
// COM:
// COM: --mode=distroless only forces the embedded set to be mounted; it does
// COM: not remove the host's C++ headers, and Comgr injects the embedded
// COM: directory with -idirafter (last). -nostdlibinc drops the host paths
// COM: while leaving that -idirafter and the clang resource dir intact, so the
// COM: embedded set is the only source.
// RUN: env AMD_COMGR_DRIVER_OPTIONS_APPEND="-nostdlibinc" \
// RUN:     compile-hip-cxx-headers --mode=distroless %s | %FileCheck %s

// Match a non-zero count, not just the log line: that line reports the skip
// decision and prints even when the set is empty.
// CHECK-DAG: Embedded {{[1-9][0-9]*}} libc++ headers at:
// CHECK-DAG: Embedded libc++ headers: active
// CHECK: RESULT: PASS

// A seed header for trace_headers.py, so this exercises the full traced graph.
#include <type_traits>

#ifndef _LIBCPP_VERSION
#error "expected the embedded libc++ headers"
#endif

static_assert(std::is_same<std::remove_cv<const int>::type, int>::value,
              "remove_cv/is_same");
static_assert(std::integral_constant<int, 7>::value == 7, "integral_constant");

extern "C" __attribute__((global)) void test_kernel(int *out) {
  out[0] = std::is_same<int, int>::value ? 1 : 0;
}
