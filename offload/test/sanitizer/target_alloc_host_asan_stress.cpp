// RUN: %libomptarget-compilexx-run-and-check-generic
//
// REQUIRES: gpu
// REQUIRES: host_asan

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

  constexpr int NumThreads = 8;
  constexpr int Iterations = 64;
  const std::array<int, 10> Sizes = {8,  16,  24,  32,  48,
                                      64, 96, 128, 256, 512};

  int Failed = 0;
#pragma omp parallel num_threads(NumThreads) reduction(| : Failed)
  {
    int Tid = omp_get_thread_num();
    for (int Iter = 0; Iter < Iterations; ++Iter) {
      int N = Sizes[(Tid + Iter) % Sizes.size()];
      int32_t *P = reinterpret_cast<int32_t *>(
          omp_target_alloc(N * sizeof(int32_t), Device));
      if (!P) {
        Failed = 1;
        continue;
      }

#pragma omp target teams distribute parallel for is_device_ptr(P)
      for (int I = 0; I < N; ++I)
        P[I] = Tid ^ Iter ^ I;

      std::vector<int32_t> Host(N);
      if (omp_target_memcpy(Host.data(), P, N * sizeof(int32_t), 0, 0,
                            omp_get_initial_device(), Device)) {
        Failed = 1;
        omp_target_free(P, Device);
        continue;
      }

      for (int I = 0; I < N; ++I) {
        if (Host[I] != (Tid ^ Iter ^ I)) {
          Failed = 1;
          break;
        }
      }
      omp_target_free(P, Device);
    }
  }

  printf("%s\n", Failed ? "FAIL" : "PASS");
  return Failed ? 1 : 0;
}

// CHECK: PASS
