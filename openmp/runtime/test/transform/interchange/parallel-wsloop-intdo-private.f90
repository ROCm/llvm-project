
! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

program interchange_wsloop_intdo
  integer :: i, j, k
  print *, 'do'

  !$OMP PARALLEL DO NUM_THREADS(4) PRIVATE(k)
  !$OMP INTERCHANGE
  do i = 7, 15, 3
    do j = -1, 1
      k = i + j
      print '("i=", I0, " j=", I0, " k=", I0)', i, j, k
    end do
  end do
  !$OMP END INTERCHANGE
  !$OMP END PARALLEL DO

  print *, 'done'
end program


! CHECK:      do
! CHECK-DAG: i=7 j=-1 k=6
! CHECK-DAG: i=10 j=-1 k=9
! CHECK-DAG: i=13 j=-1 k=12
! CHECK-DAG: i=7 j=0 k=7
! CHECK-DAG: i=10 j=0 k=10
! CHECK-DAG: i=13 j=0 k=13
! CHECK-DAG: i=7 j=1 k=8
! CHECK-DAG: i=10 j=1 k=11
! CHECK-DAG: i=13 j=1 k=14
! CHECK-NEXT: done
