// COM: The embedded libc++ header set must be non-empty and usable on every
// COM: host platform. The set is built at configure time by
// COM: cmake/trace_headers.py from clang -E -H output. When that script
// COM: mishandled host path separators the set came out empty,
// COM: EmbedLibcxxHeaders.cmake still emitted a valid
// COM: getLibcxxHeaderFiles() { return {}; }, the build succeeded, and the
// COM: failure only surfaced at runtime as "'type_traits' file not found".
// COM:
// COM: This test is deliberately not gated on a host platform. The embedded
// COM: header path is platform independent, and the lack of coverage on
// COM: non-Linux hosts is what allowed that failure to ship.
// RUN: compile-hip-cxx-headers --mode=distroless %s | %FileCheck %s

// A zero count means trace_headers.py produced a manifest with no libc++
// entries, so hasEmbeddedLibcxxHeaders() is false and nothing is mounted into
// the VFS. Match a non-zero count explicitly rather than just the log line,
// which reports the skip decision and not whether any headers exist.
// CHECK-DAG: Embedded {{[1-9][0-9]*}} libc++ headers at:
// CHECK-DAG: Embedded libc++ headers: active
// CHECK: RESULT: PASS

// Include libc++ detail headers rather than <type_traits> itself. The embedded
// set is added with -idirafter, so on a host that has a system C++ library the
// top-level names resolve to it instead, and mixing the two breaks the
// #include_next chain. Only the __-prefixed names are guaranteed to come from
// the embedded set. These are transitive dependencies of <type_traits>, which
// is what MIOpen includes and what trace_headers.py must discover; 171 of the
// 181 entries in a current manifest are detail headers like these.
#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cv.h>

#ifndef _LIBCPP_VERSION
#error "expected the embedded libc++ headers"
#endif

static_assert(std::is_same<std::remove_cv<const int>::type, int>::value,
              "remove_cv/is_same");
static_assert(std::integral_constant<int, 7>::value == 7, "integral_constant");

extern "C" __attribute__((global)) void test_kernel(int *out) {
  out[0] = std::is_same<int, int>::value ? 1 : 0;
}
