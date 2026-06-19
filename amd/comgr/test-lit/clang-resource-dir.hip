// REQUIRES: system-linux

// RUN: env -u LLVM_PATH AMD_COMGR_USE_VFS=1 \
// RUN:   AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:   compile-hip-minimal %s %t.bin | %FileCheck %s

// CHECK: Embedded {{[0-9]+}} libc++ headers at: include/c++/v1
// CHECK: Embedded {{[0-9]+}} clang builtins at: [[RESOURCE:[^" ]*lib/clang/[^" ]+]]/include
// CHECK: clang "-cc1"
// CHECK-SAME: "-resource-dir" "[[RESOURCE]]"
// CHECK: ReturnStatus: AMD_COMGR_STATUS_SUCCESS

#include <stddef.h>

__attribute__((global)) void use_size_t(size_t *out) {
  *out = sizeof(size_t);
}
