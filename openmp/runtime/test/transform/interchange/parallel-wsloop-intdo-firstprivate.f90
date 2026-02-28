
! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines

program interchange_wsloop_intdo
  integer :: i, j, k
  print *, 'do'

  k = 1
  !$OMP PARALLEL DO NUM_THREADS(3) FIRSTPRIVATE(k)
  !$OMP INTERCHANGE
  do i = 7, 15, 3
    do j = -1, 1
      k = k + 1
      print '("i=", I0, " j=", I0, " k=", I0)', i, j, k
    end do
  end do
  !$OMP END INTERCHANGE
  !$OMP END PARALLEL DO

  print *, 'done'
end program


! CHECK:      do
! CHECK-DAG: i=7 j=-1 k=2
! CHECK-DAG: i=10 j=-1 k=3
! CHECK-DAG: i=13 j=-1 k=4
! CHECK-DAG: i=7 j=0 k=2
! CHECK-DAG: i=10 j=0 k=3
! CHECK-DAG: i=13 j=0 k=4
! CHECK-DAG: i=7 j=1 k=2
! CHECK-DAG: i=10 j=1 k=3
! CHECK-DAG: i=13 j=1 k=4
! CHECK-NEXT: done
