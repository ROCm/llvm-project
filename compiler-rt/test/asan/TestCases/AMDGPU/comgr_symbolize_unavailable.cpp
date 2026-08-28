// RUN: %clangxx_asan -O2 %s -o %t -Wl,-rpath,%compiler_rt_libdir
// RUN: env SANITIZER_COMGR_PATH=%t.does_not_exist.so \
// RUN:   %env_asan_opts=verbosity=2 not %run %t 2>&1 | FileCheck %s
//
// A wavefront error reported through __asan_report_nonself_error must still
// produce a complete report when libamd_comgr cannot be loaded: the frame is
// printed unsymbolized rather than dropped or crashing. SANITIZER_COMGR_PATH
// points at a file that does not exist to force the dlopen failure.
//
// REQUIRES: linux, stable-runtime, rocm
// UNSUPPORTED: android

#include <cstdint>

using uptr = unsigned long;
using u64 = uint64_t;
using u32 = uint32_t;
using s64 = int64_t;

extern "C" void __asan_report_nonself_error(
    uptr *nonself_callstack, u32 n_nonself_callstack, uptr *nonself_addrs,
    u32 n_nonself_addrs, u64 *nonself_tids, u32 n_nonself_tids, bool is_write,
    u32 access_size, bool is_abort, const char *nonself_name,
    s64 nonself_adjust_vma, int nonself_fd, u64 nonself_file_extent_size,
    u64 nonself_file_extent_start = /*default*/ 0);

int main() {
  int stack_arr[2];
  // Two lanes faulting just past the array, so the shadow classifies this as
  // a stack overflow.
  uptr addrs[2] = {(uptr)&stack_arr[2], (uptr)&stack_arr[2] + 4};
  uptr callstack[1] = {(uptr)__builtin_return_address(0)};
  // Fixed prefix is device id and workgroup x/y/z, then one id per lane.
  u64 wi_ids[6] = {0, 1, 2, 3, /*lane ids*/ 40, 41};

  __asan_report_nonself_error(callstack, 1, addrs, 2, wi_ids, 6,
                              /*is_write=*/true, /*access_size=*/4,
                              /*is_abort=*/true, "amdgpu",
                              /*vma_adjust=*/0, /*fd=*/-1,
                              /*file_extent_size=*/0,
                              /*file_extent_start=*/0);
  return 0;
}

// CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on amdgpu device 0 at pc
// CHECK: WRITE of size 4 in workgroup id (1,2,3)
// Comgr is loaded lazily, so the dlopen failure surfaces while the frame prints.
// CHECK: Comgr Init: dlopen({{.*does_not_exist\.so}}) failed
// CHECK: #0 0x{{[0-9a-f]+}} in <unavailable>
// CHECK: Thread ids and accessed addresses:
// CHECK: 40 : 0x{{[0-9a-f]+}} 41 : 0x{{[0-9a-f]+}}
// CHECK: Address 0x{{[0-9a-f]+}} is located in stack of thread T0
// CHECK: Shadow bytes around the buggy address:
