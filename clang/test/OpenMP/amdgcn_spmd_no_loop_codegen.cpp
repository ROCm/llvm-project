// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-assume-teams-oversubscription -fopenmp-assume-threads-oversubscription -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefix=DEVICE-ASSUME
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefix=DEVICE-DEFAULT
// expected-no-diagnostics

void promote(int *a, int n) {
#pragma omp target teams distribute parallel for map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

void promote_generic_loop(int *a, int n) {
#pragma omp target teams loop bind(parallel) map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

void promote_simd(int *a, int n) {
#pragma omp target teams distribute parallel for simd map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

void no_promote_num_teams(int *a, int n) {
#pragma omp target teams distribute parallel for num_teams(4) map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

void no_promote_reduction(int *a, int n) {
  int sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum) map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i) {
    a[i] = i;
    sum += a[i];
  }
}

// DEVICE-ASSUME: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden addrspace(1) constant i32 1
// DEVICE-ASSUME: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden addrspace(1) constant i32 1
// DEVICE-ASSUME: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z7promotePii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 6,
// DEVICE-ASSUME: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z20promote_generic_loopPii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 6,
// DEVICE-ASSUME: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z12promote_simdPii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 6,
// DEVICE-ASSUME: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z20no_promote_num_teamsPii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2, i32 1, i32 256, i32 4, i32 4, i32 0, i32 0 },
// DEVICE-ASSUME: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z20no_promote_reductionPii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2, i32 1, i32 256, i32 0, i32 0, i32 4, i32 1024 },

// DEVICE-DEFAULT: @__omp_rtl_assume_teams_oversubscription = weak_odr hidden addrspace(1) constant i32 0
// DEVICE-DEFAULT: @__omp_rtl_assume_threads_oversubscription = weak_odr hidden addrspace(1) constant i32 0
// DEVICE-DEFAULT: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z7promotePii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2,
// DEVICE-DEFAULT: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z12promote_simdPii_l{{[0-9]+}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2,
