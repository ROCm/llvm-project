// clang-format off
// Cross-platform correctness test for Xteam Scans.
// Tests both inclusive and exclusive scan (prefix sum) with the segmented
// xteam scan kernel variant. This test does NOT rely on
// LIBOMPTARGET_KERNEL_TRACE, so it can run on any GPU target once the
// scan codegen supports it.
//
// Currently UNSUPPORTED on NVPTX due to a compiler verifier assertion
// (BasicBlock::getNumber) in the scan codegen path for NVPTX targets.
//
// RUN: %libomptarget-compile-generic -O2 -fopenmp-target-ignore-env-vars -fopenmp-target-xteam-scan -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -lm -latomic
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on

#include <stdio.h>
#include <stdlib.h>

#define N 2000000

int main() {
  int *in = (int *)malloc(sizeof(int) * N);
  int *out = (int *)malloc(sizeof(int) * N);

  for (int i = 0; i < N; i++) {
    in[i] = 1;
    out[i] = 0;
  }

  // --- Inclusive scan ---
  int sum1 = 0;
#pragma omp target teams distribute parallel for reduction(inscan, + : sum1) \
    map(tofrom : in [0:N], out [0:N])
  for (int i = 0; i < N; i++) {
    sum1 += in[i];
#pragma omp scan inclusive(sum1)
    out[i] = sum1;
  }

  int checksum = 0;
  for (int i = 0; i < N; i++) {
    checksum += in[i];
    if (checksum != out[i]) {
      printf("Inclusive Scan: FAIL at %d. Expected %d, got %d\n", i, checksum,
             out[i]);
      free(in);
      free(out);
      return 1;
    }
  }
  printf("Inclusive Scan: Success\n");

  // --- Exclusive scan ---
  int sum2 = 0;
  for (int i = 0; i < N; i++)
    out[i] = 0;

#pragma omp target teams distribute parallel for reduction(inscan, + : sum2) \
    map(tofrom : in [0:N], out [0:N])
  for (int i = 0; i < N; i++) {
    out[i] = sum2;
#pragma omp scan exclusive(sum2)
    sum2 += in[i];
  }

  checksum = 0;
  for (int i = 0; i < N; i++) {
    if (checksum != out[i]) {
      printf("Exclusive Scan: FAIL at %d. Expected %d, got %d\n", i, checksum,
             out[i]);
      free(in);
      free(out);
      return 1;
    }
    checksum += in[i];
  }
  printf("Exclusive Scan: Success\n");

  free(in);
  free(out);
  return 0;
}

/// CHECK: Inclusive Scan: Success
/// CHECK: Exclusive Scan: Success
