// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t 2>&1 | FileCheck %s
//
// Regression test for HIP managed memory backing: a NO_REGISTER reservation
// registered with hsa_amd_svm_attributes_set. ROCr with SVM pointer support
// reports such ranges as HSA_EXT_POINTER_TYPE_SVM rather than RESERVED_ADDR;
// ASan must still own them, so pointer_info reports the user allocation and a
// use after hsa_amd_vmem_address_free is diagnosed.
//
// REQUIRES: linux, stable-runtime, rocm, hsa-vmem
// UNSUPPORTED: android

#include "hsa_amd_test_helpers.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdio.h>

int main() {
  HSA_CHECK(hsa_init());

  const size_t kSize = 4 << 20;
  void *mem = nullptr;
  HSA_CHECK(hsa_amd_vmem_address_reserve_align(
      &mem, kSize, /*address=*/0, /*alignment=*/4096,
      /*flags=*/HSA_AMD_VMEM_ADDRESS_NO_REGISTER));

  hsa_amd_svm_attribute_pair_t attr = {HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG,
                                       HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED};
  HSA_CHECK(hsa_amd_svm_attributes_set(mem, kSize, &attr, 1));

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(info);
  HSA_CHECK(hsa_amd_pointer_info(mem, &info, nullptr, nullptr, nullptr));
  fprintf(stderr, "hostBase matches user base: %d size: %zu\n",
          info.hostBaseAddress == mem, info.sizeInBytes);
  fflush(stderr);

  HSA_CHECK(hsa_amd_vmem_address_free(mem, kSize));
  auto *p = reinterpret_cast<volatile char *>(mem);
  p[16] = 1; // Use-after-free

  fprintf(stderr, "expected use-after-free report\n");
  return 0;
}

// CHECK: hostBase matches user base: 1 size: 4194304
// CHECK: ERROR: AddressSanitizer: heap-use-after-free
// CHECK: SUMMARY: AddressSanitizer: heap-use-after-free {{.*}}hsa_amd_svm_reserve_uaf.cpp:43:{{[0-9]+}} in main
