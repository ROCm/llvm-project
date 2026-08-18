// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
// RUN: %libomptarget-compileopt-generic
// RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
//
// REQUIRES: amdgpu

// Every worker thread of a generic-mode kernel has to reach the parallel
// region, including the threads the launch geometry adds beyond a round number
// of warps: a block hosting the main thread in a single thread above the
// workers has a size that is not a multiple of the warp size. A worker state
// machine that admits fewer workers than the runtime hands work to drops the
// iterations assigned to the difference, and does so silently.
//
// The printf keeps the kernel in generic mode, which is where the state machine
// lives; if it ever stops doing so the Generic-mode check below fails rather
// than the test quietly covering nothing.

#include <omp.h>
#include <stdio.h>

#define NUM_TEAMS 4
#define TEAM_WORK 1024
#define N (NUM_TEAMS * TEAM_WORK)

int main() {
  int A[N];
  for (int i = 0; i < N; ++i)
    A[i] = -1;

#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(TEAM_WORK)          \
    map(tofrom : A[0 : N])
#pragma omp distribute
  for (int j = 0; j < NUM_TEAMS; ++j) {
#pragma omp parallel for
    for (int i = j * TEAM_WORK; i < (j + 1) * TEAM_WORK; ++i) {
      A[i] = i;
      if (i == 1)
        printf("thread %d wrote element %d\n", omp_get_thread_num(), i);
    }
    if (j == 1)
      printf("team %d wrote element %d\n", j, A[j * TEAM_WORK]);
  }

  int Missing = 0;
  for (int i = 0; i < N; ++i)
    if (A[i] != i)
      ++Missing;

  // CHECK: in Generic mode
  // CHECK: missing iterations: 0
  printf("missing iterations: %d\n", Missing);
  return Missing != 0;
}
