! Check that flang can handle mixed C and fortran inputs.

! AOCC team xfails this test as its thought to be f18.
! UNSUPPORTED

! RUN: %clang --driver-mode=flang -### -fsyntax-only %S/Inputs/one.f90 %S/Inputs/other.c 2>&1 | FileCheck --check-prefixes=CHECK-SYNTAX-ONLY %s
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}flang{{[^"/]*}}" "-fc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/one.f90"
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}clang{{[^"/]*}}" "-cc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/other.c"
