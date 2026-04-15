// RUN: %libomptarget-compilexx-run-and-check-generic
//
// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: amdgpu_asan

#include <omp.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  const int Device = omp_get_default_device();
  if (Device == omp_get_initial_device()) {
    printf("PASS\n");
    return 0;
  }

  constexpr int Iterations = 32;
  const std::array<int, 8> Sizes = {32, 64, 96, 128, 256, 512, 1024, 2048};
  int Failed = 0;

  for (int Iter = 0; Iter < Iterations; ++Iter) {
    int N = Sizes[Iter % Sizes.size()];
    int32_t *P =
        reinterpret_cast<int32_t *>(omp_target_alloc(N * sizeof(int32_t), Device));
    if (!P) {
      Failed = 1;
      continue;
    }

#pragma omp target teams distribute parallel for is_device_ptr(P)
    for (int I = 0; I < N; ++I)
      P[I] = I + Iter;

    std::vector<int32_t> Host(N);
    if (omp_target_memcpy(Host.data(), P, N * sizeof(int32_t), 0, 0,
                          omp_get_initial_device(), Device)) {
      Failed = 1;
      omp_target_free(P, Device);
      continue;
    }

    for (int I = 0; I < N; ++I) {
      if (Host[I] != I + Iter) {
        Failed = 1;
        break;
      }
    }

    omp_target_free(P, Device);
  }

  printf("%s\n", Failed ? "FAIL" : "PASS");
  return Failed ? 1 : 0;
}

// CHECK: PASS
