// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -check-prefixes=DEFAULT,ALL %s
// RUN: %clang_cc1 -target-cpu gfx900 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -check-prefixes=CPU,ALL %s

// RUN: %clang_cc1 -menable-no-nans -mno-amdgpu-ieee -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -check-prefixes=NOIEEE,ALL %s

// expected-no-diagnostics

#define N 100

int callable(int);

// Check that the target attributes are set on the generated kernel
int func() {
  // ALL-LABEL: amdgpu_kernel void @__omp_offloading{{.*}} #0

  int arr[N];

#pragma omp target teams thread_limit(42)
  for (int i = 0; i < N; i++) {
    arr[i] = callable(arr[i]);
  }

  return arr[0];
}

int callable(int x) {
  // ALL-LABEL: @_Z8callablei(i32 noundef %x) #2
  return x + 1;
}

// DEFAULT: attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,65" "kernel" "no-trapping-math"="true" "omp_target_thread_limit"="42" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
// CPU: attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,65" "kernel" "no-trapping-math"="true" "omp_target_thread_limit"="42" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
// NOIEEE: attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,65" "amdgpu-ieee"="false" "kernel" "no-nans-fp-math"="true" "no-trapping-math"="true" "omp_target_thread_limit"="42" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }

// DEFAULT: attributes #2 = { convergent mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CPU: attributes #2 = { convergent mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
// NOIEEE: attributes #2 = { convergent mustprogress noinline nounwind optnone "amdgpu-ieee"="false" "no-nans-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
