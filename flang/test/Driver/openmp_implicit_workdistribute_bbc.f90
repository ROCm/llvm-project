! Verifies that bbc accepts `-fopenmp-implicit-workdistribute=` and produces
! valid FIR. The pass currently does not transform the IR for implicit mode,
! so we only check that compilation succeeds and emits a plausible main.

! RUN: bbc -fopenmp -emit-fir -fopenmp-implicit-workdistribute=device %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -fopenmp -emit-fir -fopenmp-implicit-workdistribute=host %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -fopenmp -emit-fir -fopenmp-implicit-workdistribute=none %s -o - \
! RUN:   | FileCheck %s

! CHECK-LABEL: func.func @_QQmain
program p
  integer :: a(4), b(4)
  a = b
end program
