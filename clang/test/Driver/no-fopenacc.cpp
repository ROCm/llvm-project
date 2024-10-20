// Verify that -fopenacc produces an error message.

// RUN: not %clang -fopenacc %s 2>&1 | FileCheck %s
// CHECK: {{clang: error: unknown argument: '-fopenacc'}}

// RUN: not %clang -cc1 -fopenacc %s 2>&1 | FileCheck -check-prefix CHECK-CC1 %s
// CHECK-CC1: {{error: unknown argument: '-fopenacc'}}
