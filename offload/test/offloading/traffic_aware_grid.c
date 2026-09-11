// Check that the grid is sized from the per-iteration memory traffic the
// compiler recorded: with the policy enabled, a kernel that streams a lot per
// iteration must get fewer blocks than one that streams almost nothing.
//
// The comparison is deliberately relative. The absolute block counts depend on
// the device's resident thread capacity, so only the ordering is portable.
//
// The analysis that supplies the estimate only runs at -O1 and above.
//
// RUN: %libomptarget-compile-generic -O2
// RUN: env LIBOMPTARGET_TRAFFIC_AWARE_GRID=1 %libomptarget-run-generic 2>&1 \
// RUN:  | %fcheck-generic
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DISABLED
//
// REQUIRES: amdgpu

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 18)
#define STREAMS 12

// 4 bytes per iteration over one stream: latency bound, wants the device
// oversubscribed.
static int light(const int *a) {
  int blocks = 0;
  long long sum = 0;
#pragma omp target teams distribute parallel for map(to : a[0 : N])            \
    reduction(+ : sum) reduction(max : blocks)
  for (int i = 0; i < N; ++i) {
    blocks = omp_get_num_teams();
    sum += a[i];
  }
  return blocks;
}

// 96 bytes per iteration over twelve streams: bandwidth bound, wants few
// blocks so that their working sets stay resident.
static int heavy(double *const *s) {
  int blocks = 0;
  double sum = 0;
  const double *s0 = s[0], *s1 = s[1], *s2 = s[2], *s3 = s[3];
  const double *s4 = s[4], *s5 = s[5], *s6 = s[6], *s7 = s[7];
  const double *s8 = s[8], *s9 = s[9], *s10 = s[10], *s11 = s[11];
#pragma omp target teams distribute parallel for map(                          \
        to : s0[0 : N], s1[0 : N], s2[0 : N], s3[0 : N], s4[0 : N], s5[0 : N], \
            s6[0 : N], s7[0 : N], s8[0 : N], s9[0 : N], s10[0 : N],            \
            s11[0 : N]) reduction(+ : sum) reduction(max : blocks)
  for (int i = 0; i < N; ++i) {
    blocks = omp_get_num_teams();
    sum += s0[i] + s1[i] + s2[i] + s3[i] + s4[i] + s5[i] + s6[i] + s7[i] +
           s8[i] + s9[i] + s10[i] + s11[i];
  }
  return blocks;
}

int main(void) {
  int *a = (int *)malloc(N * sizeof(int));
  double *s[STREAMS];
  for (int i = 0; i < N; ++i)
    a[i] = 1;
  for (int k = 0; k < STREAMS; ++k) {
    s[k] = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; ++i)
      s[k][i] = 1.0;
  }

  int light_blocks = light(a);
  int heavy_blocks = heavy(s);
  printf("light=%d heavy=%d\n", light_blocks, heavy_blocks);

  // CHECK: heavy kernel gets fewer blocks
  // DISABLED: both kernels get the same number of blocks
  printf("%s\n", heavy_blocks < light_blocks
                     ? "heavy kernel gets fewer blocks"
                     : (heavy_blocks == light_blocks
                            ? "both kernels get the same number of blocks"
                            : "heavy kernel got MORE blocks"));

  free(a);
  for (int k = 0; k < STREAMS; ++k)
    free(s[k]);
  return 0;
}
