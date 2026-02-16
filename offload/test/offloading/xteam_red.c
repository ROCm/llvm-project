// clang-format off
// Cross-platform correctness test for Xteam Reductions.
// Tests sum, max, and min reductions with int and double types.
// This test does NOT rely on LIBOMPTARGET_KERNEL_TRACE, so it can run on
// any GPU target (AMDGPU and NVPTX).
//
// RUN: %libomptarget-compile-generic -O2 -fopenmp-target-fast
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on

#include <stdio.h>

#define N 10000

int main() {
  double a[N];
  int b[N];
  int rc = 0;

  for (int i = 0; i < N; i++) {
    a[i] = (double)i;
    b[i] = i;
  }

  // --- Sum reduction (double) ---
  double sum_d = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum_d)
  for (int i = 0; i < N; i++)
    sum_d += a[i];

  double expected_sum = (double)(N - 1) * N / 2.0;
  if (sum_d != expected_sum) {
    printf("FAIL: sum(double) = %f, expected %f\n", sum_d, expected_sum);
    rc = 1;
  }

  // --- Sum reduction (int) ---
  int sum_i = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum_i)
  for (int i = 0; i < N; i++)
    sum_i += b[i];

  int expected_sum_i = (N - 1) * N / 2;
  if (sum_i != expected_sum_i) {
    printf("FAIL: sum(int) = %d, expected %d\n", sum_i, expected_sum_i);
    rc = 1;
  }

  // --- Max reduction (int) ---
  int max_i = 0;
#pragma omp target teams distribute parallel for reduction(max : max_i)
  for (int i = 0; i < N; i++)
    if (b[i] > max_i)
      max_i = b[i];

  if (max_i != N - 1) {
    printf("FAIL: max(int) = %d, expected %d\n", max_i, N - 1);
    rc = 1;
  }

  // --- Min reduction (int) ---
  int min_i = N;
#pragma omp target teams distribute parallel for reduction(min : min_i)
  for (int i = 0; i < N; i++)
    if (b[i] < min_i)
      min_i = b[i];

  if (min_i != 0) {
    printf("FAIL: min(int) = %d, expected 0\n", min_i);
    rc = 1;
  }

  // --- Max reduction (double) ---
  double max_d = 0.0;
#pragma omp target teams distribute parallel for reduction(max : max_d)
  for (int i = 0; i < N; i++)
    if (a[i] > max_d)
      max_d = a[i];

  if (max_d != (double)(N - 1)) {
    printf("FAIL: max(double) = %f, expected %f\n", max_d, (double)(N - 1));
    rc = 1;
  }

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: Success
