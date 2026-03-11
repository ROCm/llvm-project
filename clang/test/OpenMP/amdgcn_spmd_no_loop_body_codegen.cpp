// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-assume-teams-oversubscription -fopenmp-assume-threads-oversubscription -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// expected-no-diagnostics

void direct_body(int *a, int n) {
#pragma omp target teams distribute parallel for map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

void direct_body_simd(int *a, int n) {
#pragma omp target teams distribute parallel for simd map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

void fallback_num_threads(int *a, int n) {
#pragma omp target teams distribute parallel for num_threads(32) map(tofrom : a[0:n])
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

// CHECK-LABEL: define weak_odr protected amdgpu_kernel void @{{.*}}__Z11direct_bodyPii
// CHECK: call void @__kmpc_specialized_kernel_init()
// CHECK-DAG: call i32 @__kmpc_get_hardware_thread_id_in_block()
// CHECK-DAG: call i32 @llvm.amdgcn.workgroup.id.x()
// CHECK-NOT: __kmpc_parallel_60
// CHECK-NOT: __kmpc_distribute_static_init
// CHECK: ret void

// CHECK-LABEL: define weak_odr protected amdgpu_kernel void @{{.*}}__Z16direct_body_simdPii
// CHECK: call void @__kmpc_specialized_kernel_init()
// CHECK-DAG: call i32 @__kmpc_get_hardware_thread_id_in_block()
// CHECK-DAG: call i32 @llvm.amdgcn.workgroup.id.x()
// CHECK-NOT: __kmpc_parallel_60
// CHECK-NOT: __kmpc_distribute_static_init
// CHECK: ret void

// CHECK-LABEL: define weak_odr protected amdgpu_kernel void @{{.*}}__Z20fallback_num_threadsPii
// CHECK: __kmpc_distribute_static_init
