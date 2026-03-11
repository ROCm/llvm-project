// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// expected-no-diagnostics

void nested_loop(int *a, int n) {
#pragma omp target teams map(tofrom : a[0:n])
  {
#pragma omp loop bind(parallel)
    for (int i = 0; i < n; ++i)
      a[i] = i;
  }
}

// CHECK: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z11nested_loopPii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2,
