// RUN: %clangxx_asan -O2 %s -o %t -Wl,-rpath,%compiler_rt_libdir
// RUN: %env_asan_opts=verbosity=2 not %run %t 2>&1 | FileCheck %s
//
// Exercises the real Comgr path: libamd_comgr is dlopen'd and handed a file
// slice that is not an AMDGPU code object (the test binary itself). Whether
// Comgr can make sense of it is not the point -- the report must complete and
// the Comgr objects must be released rather than leaked or double-freed.
//
// The frame's contents are deliberately not checked: depending on the Comgr
// version a host ELF either symbolizes to "?? at ??:0:0" or is rejected, which
// falls back to "<unavailable>".
//
// REQUIRES: linux, stable-runtime, rocm, comgr
// UNSUPPORTED: android

#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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
  int fd = open("/proc/self/exe", O_RDONLY);
  if (fd < 0)
    return 1;
  struct stat st;
  if (fstat(fd, &st))
    return 1;

  int stack_arr[2];
  uptr addrs[1] = {(uptr)&stack_arr[2]};
  uptr callstack[1] = {(uptr)__builtin_return_address(0)};
  u64 wi_ids[5] = {7, 11, 22, 33, /*lane id*/ 5};

  __asan_report_nonself_error(callstack, 1, addrs, 1, wi_ids, 5,
                              /*is_write=*/false, /*access_size=*/4,
                              /*is_abort=*/true, "amdgpu",
                              /*vma_adjust=*/0, fd,
                              /*file_extent_size=*/(u64)st.st_size,
                              /*file_extent_start=*/0);
  return 0;
}

// CHECK-NOT: Comgr Init: dlopen
// CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on amdgpu device 7 at pc
// CHECK: READ of size 4 in workgroup id (11,22,33)
// A frame is always printed, either Comgr's answer or the <unavailable> fallback.
// CHECK: #0 0x{{[0-9a-f]+}} in {{.+}}
// CHECK: Thread ids and accessed addresses:
// CHECK: 05 : 0x{{[0-9a-f]+}}
// CHECK: Shadow bytes around the buggy address:
