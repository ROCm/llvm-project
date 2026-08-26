// clang-format off
// This test verifies the AMDGPU grid heuristic for cross-team reduction
// kernels: the team count saturates the device with the desired number of
// waves per CU. A 512-thread team is 8 waves, so with the desired 16 waves
// per CU the heuristic picks two teams per CU.
// Which heuristic computes the team count is selected per device by the envar
// config table, so pin both knobs to keep this test on the path it is written
// for, independently of the device it runs on: the occupancy-based team count
// is a different heuristic that replaces this one, and without the adjust factor
// the team count stays at the one-team-per-CU default.
// RUN: %libomptarget-compile-generic -fopenmp-target-fast
// RUN: env LIBOMPTARGET_DEBUG=1 OMPX_XTEAMREDUCTION_OCCUPANCY_BASED_OPT=0 \
// RUN:   LIBOMPTARGET_AMDGPU_ADJUST_XTEAM_RED_TEAMS=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on
#include <stdio.h>

int main() {
  int N = 1000000;

  double a[N];

  for (int i = 0; i < N; i++)
    a[i] = i;

  double sum1;
  sum1 = 0;

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j < N; j = j + 1)
    sum1 += a[j];

  printf("sum1=%f\n", sum1);

  return 0;
}
// clang-format off
/// CHECK: xteam-red:NumCUs=[[#CU_COUNT:]]
/// CHECK-SAME: xteam-red:NumGroups=[[#CU_COUNT+CU_COUNT]]
/// CHECK: Launching kernel {{.*}} with {{\[}}[[#CU_COUNT+CU_COUNT]],1,1] blocks and [512,1,1] threads in SPMD mode
/// CHECK: sum1=499999500000.000000

