// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fcuda-is-device -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

#define __device__ __attribute__((device))

// CHECK-LABEL: @test_user_markers(
// CHECK: call void @llvm.amdgcn.sqtt.event(metadata [[ENTRY:![0-9]+]], i32 poison, i1 true)
// CHECK: call void @llvm.amdgcn.sqtt.event(metadata [[EXIT:![0-9]+]], i32 poison, i1 true)
extern "C" __device__ void test_user_markers() {
  __builtin_amdgcn_sqtt_user_entry("foo");
  __builtin_amdgcn_sqtt_user_exit("foo");
}

// CHECK: [[ENTRY]] = !{i32 2, !"foo"}
// CHECK: [[EXIT]] = !{i32 3, !"foo"}
