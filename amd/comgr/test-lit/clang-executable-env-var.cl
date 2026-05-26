// Verify AMD_COMGR_CLANG_EXECUTABLE is honored and appears in the verbose log.
// RUN: AMD_COMGR_CLANG_EXECUTABLE=%clang \
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:    compile-opencl-minimal %s %t.bin 1.2 | %FileCheck %s

// Verify LLVM_PATH back-compat: <LLVM_PATH>/bin/clang is used when
// AMD_COMGR_CLANG_EXECUTABLE is not set.
// RUN: LLVM_PATH=%S/.. \
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:    compile-opencl-minimal %s %t.bin 1.2 | %FileCheck --check-prefix=LEGACY %s

// CHECK: Clang Executable: {{.*[/\\]}}clang
// CHECK: ReturnStatus: AMD_COMGR_STATUS_SUCCESS

// LEGACY: Clang Executable: {{.*[/\\]}}bin{{[/\\]}}clang
// LEGACY: ReturnStatus: AMD_COMGR_STATUS_SUCCESS

__kernel void test(__global int* out) {
  *out = 42;
}
