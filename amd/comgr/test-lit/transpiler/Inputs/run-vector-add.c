//===- run-vector-add.c - Launch a raised vector-add code object ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loads a code object produced by the transpiler raiser and dispatches its
// vector-add kernel over `n` floats, then checks every element against the sum
// computed on the host. Prints RESULT: PASS or RESULT: FAIL so a lit test can
// FileCheck the outcome.
//
// Usage: run-vector-add <code-object> <kernel> <n> <block>
//
// The raised kernel takes the source kernarg segment as a single by-value
// block, so the arguments are passed as one buffer through
// HIP_LAUNCH_PARAM_BUFFER_POINTER rather than as a pointer array.
//
//===----------------------------------------------------------------------===//

#define __HIP_PLATFORM_AMD__

#include <hip/hip_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(Expr)                                                        \
  do {                                                                         \
    hipError_t Status = (Expr);                                                \
    if (Status != hipSuccess) {                                                \
      fprintf(stderr, "%s failed: %s\n", #Expr, hipGetErrorString(Status));    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Mirrors the kernarg segment the source kernel declares: three global
// pointers at offsets 0, 8 and 16.
struct VectorAddArgs {
  void *A;
  void *B;
  void *C;
};

int main(int Argc, char **Argv) {
  if (Argc != 5) {
    fprintf(stderr, "usage: %s <code-object> <kernel> <n> <block>\n", Argv[0]);
    return 2;
  }
  const char *CodeObject = Argv[1];
  const char *KernelName = Argv[2];
  const int N = atoi(Argv[3]);
  const int Block = atoi(Argv[4]);
  if (N <= 0 || Block <= 0 || N % Block != 0) {
    fprintf(stderr, "n must be a positive multiple of block\n");
    return 2;
  }

  const size_t Bytes = (size_t)N * sizeof(float);
  float *HostA = (float *)malloc(Bytes);
  float *HostB = (float *)malloc(Bytes);
  float *HostC = (float *)malloc(Bytes);
  if (!HostA || !HostB || !HostC) {
    fprintf(stderr, "host allocation failed\n");
    return 1;
  }
  // Small integers, so the sums are exact in float and a mismatch is a real
  // translation error rather than rounding.
  for (int I = 0; I < N; ++I) {
    HostA[I] = (float)(I % 1024);
    HostB[I] = (float)(2 * (I % 512) + 1);
  }

  hipModule_t Module;
  hipFunction_t Kernel;
  CHECK_HIP(hipModuleLoad(&Module, CodeObject));
  CHECK_HIP(hipModuleGetFunction(&Kernel, Module, KernelName));

  struct VectorAddArgs Args;
  CHECK_HIP(hipMalloc(&Args.A, Bytes));
  CHECK_HIP(hipMalloc(&Args.B, Bytes));
  CHECK_HIP(hipMalloc(&Args.C, Bytes));
  CHECK_HIP(hipMemcpy(Args.A, HostA, Bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(Args.B, HostB, Bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemset(Args.C, 0, Bytes));

  size_t ArgsSize = sizeof(Args);
  void *Config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &Args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &ArgsSize,
                    HIP_LAUNCH_PARAM_END};
  CHECK_HIP(hipModuleLaunchKernel(Kernel, (unsigned)(N / Block), 1, 1,
                                  (unsigned)Block, 1, 1, 0, NULL, NULL,
                                  Config));
  CHECK_HIP(hipStreamSynchronize(NULL));
  CHECK_HIP(hipMemcpy(HostC, Args.C, Bytes, hipMemcpyDeviceToHost));

  int Mismatches = 0;
  for (int I = 0; I < N; ++I) {
    float Expected = HostA[I] + HostB[I];
    if (HostC[I] == Expected)
      continue;
    // Report the first few so a failure names the lanes it broke on.
    if (Mismatches < 8)
      printf("mismatch at %d: got %f, expected %f\n", I, HostC[I], Expected);
    ++Mismatches;
  }
  printf("RESULT: %s (%d of %d elements wrong)\n", Mismatches ? "FAIL" : "PASS",
         Mismatches, N);

  CHECK_HIP(hipFree(Args.A));
  CHECK_HIP(hipFree(Args.B));
  CHECK_HIP(hipFree(Args.C));
  CHECK_HIP(hipModuleUnload(Module));
  free(HostA);
  free(HostB);
  free(HostC);
  return Mismatches != 0;
}
