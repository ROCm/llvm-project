// End-to-end check of AMDGPU code-object symbolization: build a real code
// object from the device half of this file, then report a device fault whose
// pc points into it. Comgr must resolve that pc back to the kernel name and
// this source file.
//
// RUN: %clang -x c++ --target=amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -g -O0 \
// RUN:   -DAMDGPU_DEVICE_CODE %s -o %t.co
// RUN: %clangxx_asan -O2 -DCODE_OBJECT_PATH='"%t.co"' %s -o %t \
// RUN:   -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
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

// Unrelocated address of the kernel, which is what a GPU runtime would report
// as the device pc once it has subtracted the load bias.
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
  u64 device_pc = FirstFunctionAddress((const unsigned char *)map);
  if (!device_pc) {
    fprintf(stderr, "no function symbol in code object\n");
    return 1;
  }

  int stack_arr[2];
  uptr addrs[1] = {(uptr)&stack_arr[2]};
  uptr callstack[1] = {(uptr)device_pc};
  // Fixed prefix is device id and workgroup x/y/z, then one id per lane.
  u64 wi_ids[5] = {3, 5, 6, 7, /*lane id*/ 9};

  __asan_report_nonself_error(callstack, 1, addrs, 1, wi_ids, 5,
                              /*is_write=*/true, /*access_size=*/4,
                              /*is_abort=*/true, "amdgpu",
                              /*vma_adjust=*/0, fd,
                              /*file_extent_size=*/(u64)st.st_size,
                              /*file_extent_start=*/0);
  return 0;
}

#endif // AMDGPU_DEVICE_CODE

// CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on amdgpu device 3 at pc
// CHECK: WRITE of size 4 in workgroup id (5,6,7)
// CHECK: #0 0x{{[0-9a-f]+}} in asan_nonself_test_kernel(int*) at {{.*}}comgr_symbolize_code_object.cpp:{{[0-9]+}}
// CHECK: Thread ids and accessed addresses:
// CHECK: 09 : 0x{{[0-9a-f]+}}
// CHECK: Address 0x{{[0-9a-f]+}} is located in stack of thread T0
// CHECK: Shadow bytes around the buggy address:
