! RUN: %flang_fc1 -fopenmp -E %s 2>&1 | FileCheck %s
      program main
! CHECK: k01=1+ 1
      k01=1+
!$   &  1

! CHECK: !$omp parallel private(k01)
!$omp parallel
!$omp+ private(k01)
!$omp end parallel
      end
