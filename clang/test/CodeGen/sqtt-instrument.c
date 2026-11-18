// SQTT instrumentation is always on for AMDGPU, with or without the flag.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s -disable-llvm-passes | FileCheck --check-prefix=SQTT %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s -fsqtt-instrument -disable-llvm-passes | FileCheck --check-prefix=SQTT %s
// The attribute is only added for AMDGPU.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -fsqtt-instrument -disable-llvm-passes | FileCheck --check-prefix=NONE %s

int test1(int x) {
// SQTT: @test1(i32 {{.*}}%x) #[[ATTR1:[0-9]+]]
  return x;
}

int test2(int) __attribute__((no_instrument_function));
int test2(int x) {
// SQTT: @test2(i32 {{.*}}%x) #[[ATTR2:[0-9]+]]
  return x;
}

// SQTT: attributes #[[ATTR1]] = {{.*}}"amdgpu-sqtt-instrument"

// SQTT: attributes #[[ATTR2]] =
// SQTT-NOT: "amdgpu-sqtt-instrument"

// NONE-NOT: "amdgpu-sqtt-instrument"
