// RUN: %libomptarget-compilexx-run-and-check-generic

#include <cstdio>
#include <omp.h>

int main() {
  int i;

  i = -1;
#pragma omp target map(to : i) map(from : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1

  i = -1;
#pragma omp target map(from : i) map(to : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1

  i = -1;
#pragma omp target map(alloc : i) map(to : i) map(from : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1

  i = -1;
#pragma omp target map(alloc : i) map(from : i) map(to : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1

  i = -1;
#pragma omp target map(to : i) map(alloc : i) map(from : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1

  i = -1;
#pragma omp target map(from : i) map(alloc : i) map(to : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1

  i = -1;
#pragma omp target map(to : i) map(from : i) map(alloc : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1

  i = -1;
#pragma omp target map(from : i) map(to : i) map(alloc : i)
  i += 2;
  printf("%d\n", i);
  // CHECK: 1
}