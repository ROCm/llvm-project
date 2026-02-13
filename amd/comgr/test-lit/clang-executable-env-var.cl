// COM: Test AMD_COMGR_CLANG_EXECUTABLE environment variable

// COM: Test 1: Default behavior (no environment variable set)
// COM: This should use the default clang from the build configuration
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:    compile-opencl-minimal %s %t_default.bin 1.2 | %FileCheck --check-prefix=DEFAULT %s

// DEFAULT: Clang Executable: {{.*}}/clang
// DEFAULT: ReturnStatus: AMD_COMGR_STATUS_SUCCESS

// COM: Test 2: Override with valid clang executable
// COM: Set AMD_COMGR_CLANG_EXECUTABLE to point to the clang from the build
// RUN: AMD_COMGR_CLANG_EXECUTABLE=%clang \
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:    compile-opencl-minimal %s %t_override.bin 1.2 | %FileCheck --check-prefix=OVERRIDE %s

// OVERRIDE: Clang Executable: {{.*}}/clang
// OVERRIDE: ReturnStatus: AMD_COMGR_STATUS_SUCCESS

// COM: Test 3: Set to invalid path (should fail)
// RUN: AMD_COMGR_CLANG_EXECUTABLE=/nonexistent/path/to/clang \
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:    %not compile-opencl-minimal %s %t_invalid.bin 1.2 | %FileCheck --check-prefix=ERROR %s

// ERROR: Error: Clang executable not found: /nonexistent/path/to/clang

__kernel void test(__global int* out) {
  *out = 42;
}
