// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: %libomptarget-run-generic | %fcheck-generic --check-prefix=CHECK
// RUN: env LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=0 %libomptarget-run-generic | %fcheck-generic --check-prefix=CHECK
// RUN: env LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=1024 %libomptarget-run-generic | %fcheck-generic --check-prefix=CHECK
// clang-format on

#include <omp.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

static bool runAllocPressure(int Device) {
  constexpr int Repetitions = 16;
  constexpr int NumThreads = 8;
  constexpr int TinyAllocBytes = 8;
  constexpr int TinyAllocIterations = 512;
  const std::array<int, 28> Sizes = {
      1,   2,   3,    4,    5,    7,    8,    9,    15,   16,
      17,  31,  32,   33,   63,   64,   65,   127,  128,  129,
      255, 256, 257,  511,  512,  513,  1023, 1024};

  int Failed = 0;
#pragma omp parallel num_threads(NumThreads) reduction(| : Failed)
  {
    const int Tid = omp_get_thread_num();
    for (int Rep = 0; Rep < Repetitions; ++Rep) {
      for (int N : Sizes) {
        int8_t *P =
            reinterpret_cast<int8_t *>(omp_target_alloc(N * sizeof(int8_t), Device));
        if (!P) {
          Failed = 1;
          continue;
        }

#pragma omp target teams distribute parallel for is_device_ptr(P)
        for (int I = 0; I < N; ++I)
          P[I] = static_cast<int8_t>((I ^ Tid ^ Rep) & 0x7f);

        std::vector<int8_t> HostBuffer(N);
        if (omp_target_memcpy(HostBuffer.data(), P, N * sizeof(int8_t), 0, 0,
                              omp_get_initial_device(), Device)) {
          Failed = 1;
          omp_target_free(P, Device);
          continue;
        }

        for (int I = 0; I < N; ++I) {
          int8_t Expected = static_cast<int8_t>((I ^ Tid ^ Rep) & 0x7f);
          if (HostBuffer[I] != Expected) {
            Failed = 1;
            break;
          }
        }

        omp_target_free(P, Device);
      }

      // Emulate the reported tiny-allocation pressure pattern: many parallel
      // 8-byte alloc/free operations in tight loops.
      for (int TinyIter = 0; TinyIter < TinyAllocIterations; ++TinyIter) {
        int8_t *Tiny = reinterpret_cast<int8_t *>(
            omp_target_alloc(TinyAllocBytes * sizeof(int8_t), Device));
        if (!Tiny) {
          Failed = 1;
          continue;
        }

#pragma omp target is_device_ptr(Tiny)
        {
          Tiny[0] = static_cast<int8_t>((Tid + TinyIter) & 0x7f);
          Tiny[TinyAllocBytes - 1] = static_cast<int8_t>((Rep + Tid) & 0x7f);
        }

        std::array<int8_t, TinyAllocBytes> HostTiny{};
        if (omp_target_memcpy(HostTiny.data(), Tiny, TinyAllocBytes, 0, 0,
                              omp_get_initial_device(), Device)) {
          Failed = 1;
          omp_target_free(Tiny, Device);
          continue;
        }

        if (HostTiny[0] != static_cast<int8_t>((Tid + TinyIter) & 0x7f) ||
            HostTiny[TinyAllocBytes - 1] !=
                static_cast<int8_t>((Rep + Tid) & 0x7f))
          Failed = 1;

        omp_target_free(Tiny, Device);
      }
    }
  }

  return Failed == 0;
}

int main() {
  int Device = omp_get_default_device();
  if (Device == omp_get_initial_device()) {
    // Fallback for environments without offload hardware.
    printf("PASS\n");
    return 0;
  }

  bool Success = runAllocPressure(Device);
  printf("%s\n", Success ? "PASS" : "FAIL");
  return Success ? 0 : 1;
}

// CHECK: PASS
