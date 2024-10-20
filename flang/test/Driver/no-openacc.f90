! Verify that -fopenacc produces an error message.

! RUN: not %flang -fopenacc %s 2>&1 | FileCheck %s
! CHECK: {{flang.*: error: unknown argument: '-fopenacc'}}

! RUN: not %flang_fc1 -fopenacc %s 2>&1 | FileCheck --check-prefix CHECK-FC1 %s
! CHECK-FC1: {{error: unknown argument: '-fopenacc'}}
