// Device leak reporting: __asan_report_nonself_leak accumulates allocations
// per (pc, device) and prints them when called with device_id == -1. The
// allocating pc is symbolized through Comgr against a real code object built
// from the device half of this file.
//
// RUN: %clang -x c++ --target=amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -g -O0 \
// RUN:   -DAMDGPU_DEVICE_CODE %s -o %t.co
// RUN: %clangxx_asan -O2 -DCODE_OBJECT_PATH='"%t.co"' %s -o %t \
// RUN:   -Wl,-rpath,%compiler_rt_libdir
// RUN: %env_asan_opts=detect_leaks=1 %run %t 2>&1 | FileCheck %s
//
// REQUIRES: linux, stable-runtime, rocm, comgr, amdgpu-codegen
// UNSUPPORTED: android

#ifdef AMDGPU_DEVICE_CODE

__attribute__((amdgpu_kernel)) void asan_nonself_test_kernel(int *p) {
  p[0] = 42;
}

#else

#  include <cstdint>
#  include <cstdio>
#  include <elf.h>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>

using u64 = uint64_t;
using s64 = int64_t;

extern "C" void __asan_report_nonself_leak(u64 alloc_pc, u64 alloc_size,
                                           int device_id,
                                           const char *device_name,
                                           s64 vma_adjust, int fd,
                                           u64 file_extent_size,
                                           u64 file_extent_start =
                                               /*default*/ 0);

// Unrelocated address of the kernel, standing in for the device pc a GPU
// runtime would record as the allocation site.
static u64 FirstFunctionAddress(const unsigned char *elf) {
  auto *eh = reinterpret_cast<const Elf64_Ehdr *>(elf);
  auto *sh = reinterpret_cast<const Elf64_Shdr *>(elf + eh->e_shoff);
  for (unsigned i = 0; i < eh->e_shnum; ++i) {
    if (sh[i].sh_type != SHT_SYMTAB)
      continue;
    auto *sym = reinterpret_cast<const Elf64_Sym *>(elf + sh[i].sh_offset);
    for (unsigned j = 0; j < sh[i].sh_size / sizeof(Elf64_Sym); ++j)
      if (ELF64_ST_TYPE(sym[j].st_info) == STT_FUNC && sym[j].st_value)
        return sym[j].st_value;
  }
  return 0;
}

int main() {
  int fd = open(CODE_OBJECT_PATH, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "cannot open %s\n", CODE_OBJECT_PATH);
    return 1;
  }
  struct stat st;
  if (fstat(fd, &st))
    return 1;

  void *map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED)
    return 1;
  u64 alloc_pc = FirstFunctionAddress((const unsigned char *)map);
  if (!alloc_pc) {
    fprintf(stderr, "no function symbol in code object\n");
    return 1;
  }

  // Two allocations from the same pc must fold into one entry of 256 bytes.
  __asan_report_nonself_leak(alloc_pc, 100, /*device_id=*/0, "amdgpu",
                             /*vma_adjust=*/0, fd, (u64)st.st_size, 0);
  __asan_report_nonself_leak(alloc_pc, 156, /*device_id=*/0, "amdgpu",
                             /*vma_adjust=*/0, fd, (u64)st.st_size, 0);

  // device_id == -1 flushes the table and prints the summary.
  __asan_report_nonself_leak(0, 0, /*device_id=*/-1, "amdgpu",
                             /*vma_adjust=*/0, -1, 0, 0);
  return 0;
}

#endif // AMDGPU_DEVICE_CODE

// CHECK: Leak of 256 byte(s) in 2 allocation(s) on amdgpu device 0 from:
// CHECK-NEXT: #0 0x{{[0-9a-f]+}} in asan_nonself_test_kernel(int*) at {{.*}}comgr_symbolize_leak.cpp:{{[0-9]+}}
// CHECK: SUMMARY: AddressSanitizer: 256 byte(s) leaked in 2 allocation(s) on amdgpu device 0.
