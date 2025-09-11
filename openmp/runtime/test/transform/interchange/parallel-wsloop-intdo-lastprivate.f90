
! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines
! XFAIL: *

program interchange_wsloop_intdo
  integer :: i, j, k
  print *, 'do'


  !$OMP PARALLEL DO LASTPRIVATE(k)
  !$OMP INTERCHANGE
  do i = 7, 15, 3
    do j = -1, 1
      k = i*10 + j
    end do
  end do
  !$OMP END INTERCHANGE
  !$OMP END PARALLEL DO

  print *, 'done'
  print '("k=", I0)', k
end program


! CHECK:      do
! CHECK-NEXT: done
! CHECK-NEXT: k=131
