// Verify AMD_COMGR_CLANG_EXECUTABLE is honored and appears in the verbose log.
// RUN: AMD_COMGR_CLANG_EXECUTABLE=%clang \
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:    compile-opencl-minimal %s %t.bin 1.2 | %FileCheck %s

// CHECK: Clang Executable: {{.*}}clang{{.*}}
// CHECK: ReturnStatus: AMD_COMGR_STATUS_SUCCESS

__kernel void test(__global int* out) {
  *out = 42;
}
